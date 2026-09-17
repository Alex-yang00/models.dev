import { expect, test } from "bun:test";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

import { groups, providers, syncProvider } from "../src/sync/index.js";
import { fetchNovitaAIModels, NovitaAIResponse, novitaAi, type NovitaAIModel } from "../src/sync/providers/novita-ai.js";

function novitaAiModel(overrides: Partial<NovitaAIModel> = {}): NovitaAIModel {
  return {
    id: "deepseek/deepseek-v3.2",
    object: "model",
    created: 1_765_440_000,
    owned_by: "novita",
    ...overrides,
  };
}

test("parses Novita AI API response", () => {
  const parsed = NovitaAIResponse.parse({
    data: [
      novitaAiModel(),
      novitaAiModel({ id: "meta-llama/llama-3.3-70b-instruct", created: 1_733_635_200 }),
    ],
  });
  expect(parsed.data).toHaveLength(2);
  expect(parsed.data[0]?.id).toBe("deepseek/deepseek-v3.2");
  expect(parsed.data[1]?.id).toBe("meta-llama/llama-3.3-70b-instruct");
});

test("accepts the standard OpenAI list marker when present", () => {
  expect(NovitaAIResponse.parse({ object: "list", data: [novitaAiModel()] }).data).toHaveLength(1);
});

test("maps Novita catalog metadata onto existing models", () => {
  const translated = novitaAi.translateModel(novitaAiModel({
    display_name: "GLM 5.3 Flash",
    description: "Updated description",
    context_size: 1_048_576,
    max_output_tokens: 131_072,
    features: ["function-calling", "structured-outputs", "reasoning"],
    input_modalities: ["text", "image"],
    output_modalities: ["text"],
    pricing: {
      prompt: { price_per_m_decimal: "0.15" },
      completion: { price_per_m_decimal: "0.5" },
      input_cache_read: { price_per_m_decimal: "0.03" },
    },
  }), {
    existing: () => ({}),
    authored: () => ({ base_model: "deepseek/deepseek-v3.2", name: "Old", description: "Old", attachment: false, reasoning: false, tool_call: false, open_weights: true, limit: { context: 1, output: 1 }, modalities: { input: ["text"], output: ["text"] } }),
  });
  expect(translated?.model).toMatchObject({ name: "GLM 5.3 Flash", limit: { context: 1_048_576, output: 131_072 }, cost: { input: 0.15, output: 0.5, cache_read: 0.03 }, modalities: { input: ["text", "image"] } });
});

test("rejects invalid Novita AI API responses", () => {
  expect(() => NovitaAIResponse.parse({ object: "list", data: [{ id: "bad", object: "not-model", created: 1, owned_by: "" }] }))
    .toThrow();
  expect(() => NovitaAIResponse.parse({ object: "list", data: [{ id: "", object: "model", created: -1, owned_by: "" }] }))
    .toThrow();
});

test("Novita AI sync preserves authored metadata for existing models", () => {
  const authored = {
    base_model: "deepseek/deepseek-v3.2",
    name: "Deepseek V3.2",
    description: "DeepSeek chat model for instruction following, coding, and analysis",
    family: "deepseek",
    release_date: "2025-12-01",
    last_updated: "2025-12-01",
    attachment: false,
    reasoning: true,
    reasoning_options: [{ type: "toggle" } as const],
    temperature: true,
    tool_call: true,
    structured_output: true,
    open_weights: true,
    cost: { input: 0.269, output: 0.4, cache_read: 0.1345 },
    limit: { context: 163_840, output: 65_536 },
    interleaved: { field: "reasoning_content" },
    modalities: { input: ["text"], output: ["text"] },
  };

  const translated = novitaAi.translateModel(novitaAiModel(), {
    existing: () => authored,
    authored: () => authored,
  });

  expect(translated).toMatchObject({ id: "deepseek/deepseek-v3.2", model: {
    base_model: authored.base_model,
    reasoning_options: authored.reasoning_options,
    interleaved: authored.interleaved,
    cost: authored.cost,
  } });
});

test("Novita AI sync creates non-reasoning models with a known lab base and a price", () => {
  const translated = novitaAi.translateModel(novitaAiModel({ id: "deepseek/deepseek-v3", context_size: 8192, max_output_tokens: 4096, features: [], pricing: { prompt: { price_per_m_decimal: "0.1" }, completion: { price_per_m_decimal: "0.2" } } }), {
    existing: () => undefined,
    authored: () => undefined,
  });
  expect(translated?.id).toBe("deepseek/deepseek-v3");
  expect(translated?.model).toMatchObject({ base_model: "deepseek/deepseek-v3", limit: { context: 8192, output: 4096 }, cost: { input: 0.1, output: 0.2 } });
});

test("Novita AI sync skips new models with unknown lab, price, or reasoning controls", () => {
  const context = { existing: () => undefined, authored: () => undefined };
  const price = { prompt: { price_per_m_decimal: "0.1" }, completion: { price_per_m_decimal: "0.2" } };
  expect(novitaAi.translateModel(novitaAiModel({ id: "novita/unknown-model", pricing: price }), context)).toBeUndefined();
  expect(novitaAi.translateModel(novitaAiModel({ id: "deepseek/deepseek-v3", features: [] }), context)).toBeUndefined();
  expect(novitaAi.translateModel(novitaAiModel({ id: "deepseek/deepseek-v3", features: ["reasoning"], pricing: price }), context)).toBeUndefined();
});

test("Novita AI sync treats explicit zero prices without tiers as free", () => {
  const model = novitaAiModel({
    id: "inclusionai/ling-3.0-flash-fin",
    input_token_price_per_m: 0,
    output_token_price_per_m: 0,
    features: ["reasoning"],
  });
  expect(novitaAi.translateModel(model, { existing: () => undefined, authored: () => undefined })).toBeUndefined();
  const authored = { base_model: "inclusionai/ling-3.0-flash-fin", reasoning_options: [] };
  const translated = novitaAi.translateModel(model, { existing: () => authored, authored: () => authored });
  expect(translated?.model).toMatchObject({
    base_model: "inclusionai/ling-3.0-flash-fin", cost: { input: 0, output: 0 },
  });
});

test("Novita AI sync does not mistake tier-only pricing for free", () => {
  const translated = novitaAi.translateModel(novitaAiModel({
    id: "deepseek/deepseek-v3",
    input_token_price_per_m: 0,
    output_token_price_per_m: 0,
    is_tiered_billing: true,
    features: [],
    tiered_billing_configs: [{
      min_tokens: 1, max_tokens: 10_000,
      pricing: { prompt: { price_per_m_decimal: "0.5" }, completion: { price_per_m_decimal: "2" } },
    }],
  }), { existing: () => undefined, authored: () => undefined });
  expect(translated?.model).toMatchObject({ cost: { input: 0.5, output: 2 } });
});

test("Novita AI sync reuses a verified lab alias and fixed R1 controls", () => {
  const context = { existing: () => undefined, authored: () => undefined };
  const pricing = { prompt: { price_per_m_decimal: "0.89" }, completion: { price_per_m_decimal: "0.89" } };
  expect(novitaAi.translateModel(novitaAiModel({
    id: "deepseek/deepseek_v3", features: [], pricing,
  }), context)?.model).toMatchObject({ base_model: "deepseek/deepseek-v3" });
  expect(novitaAi.translateModel(novitaAiModel({
    id: "deepseek/deepseek-r1", features: ["reasoning"], pricing,
  }), context)?.model).toMatchObject({ base_model: "deepseek/deepseek-r1", reasoning_options: [] });
});

test("Novita AI sync maps tiered context prices and cache-write", () => {
  const pricing = (input: string, output: string, cacheWrite: string) => ({
    prompt: { price_per_m_decimal: input },
    completion: { price_per_m_decimal: output },
    input_cache_write: { price_per_m_decimal: cacheWrite },
  });
  const result = novitaAi.translateModel(novitaAiModel({
    id: "deepseek/deepseek-v3",
    features: [],
    is_tiered_billing: true,
    tiered_billing_configs: [
      { min_tokens: 256_000, max_tokens: 1_000_000, pricing: pricing("0.5", "3", "0.625") },
      { min_tokens: 1, max_tokens: 256_000, pricing: pricing("0.4", "2.4", "0.5") },
    ],
  }), { existing: () => undefined, authored: () => undefined });
  expect(result?.model).toMatchObject({ cost: {
    input: 0.4, output: 2.4, cache_write: 0.5,
    tiers: [{ tier: { type: "context", size: 256_000 }, input: 0.5, output: 3, cache_write: 0.625 }],
  } });
});

test("Novita AI sync preserves inherited capabilities when features are absent", () => {
  const authored = { base_model: "deepseek/deepseek-v3.2", cost: { input: 0.1, output: 0.2 } };
  const resolved = { ...authored, reasoning: true, tool_call: true, modalities: { input: ["text" as const], output: ["text" as const] } };
  const translated = novitaAi.translateModel(novitaAiModel(), {
    authored: () => authored,
    existing: () => resolved,
  });
  expect(translated?.model).toMatchObject({ base_model: authored.base_model });
  expect(translated?.model).not.toHaveProperty("reasoning", false);
  expect(translated?.model).not.toHaveProperty("tool_call", false);
});

test("Novita AI sync updates existing inline model capabilities", () => {
  const existing = {
    name: "Old", description: "Old", reasoning: false, tool_call: false,
    attachment: false, open_weights: false, release_date: "2025-01-01", last_updated: "2025-01-01",
    limit: { context: 8192, output: 4096 }, modalities: { input: ["text" as const], output: ["text" as const] },
  };
  const translated = novitaAi.translateModel(novitaAiModel({
    id: "novita/custom-model",
    display_name: "Updated", features: ["reasoning", "function-calling"],
    input_modalities: ["text", "image"],
  }), { authored: () => existing, existing: () => existing });
  expect(translated?.model).toMatchObject({
    name: "Updated", reasoning: true, tool_call: true, attachment: true,
    modalities: { input: ["text", "image"] },
  });
});

test("Novita AI sync removes local models absent from API response", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "sync-novita-ai-"));
  const modelsDir = path.join(dir, "providers", "novita-ai", "models");
  await mkdir(modelsDir, { recursive: true });
  await Bun.write(path.join(modelsDir, "deepseek", "deepseek-v3.2.toml"), [
    'name = "Deepseek V3.2"',
    'description = "DeepSeek chat model for instruction following, coding, and analysis"',
    'family = "deepseek"',
    'release_date = "2025-12-01"',
    'last_updated = "2025-12-01"',
    "attachment = false",
    "reasoning = true",
    "reasoning_options = [{ type = \"toggle\" }]",
    "temperature = true",
    "tool_call = true",
    "structured_output = true",
    "open_weights = true",
    "",
    "[interleaved]",
    'field = "reasoning_content"',
    "",
    "[cost]",
    "input = 0.269",
    "output = 0.4",
    "cache_read = 0.1345",
    "",
    "[limit]",
    "context = 163_840",
    "output = 65_536",
    "",
    "[modalities]",
    'input = ["text"]',
    'output = ["text"]',
    "",
  ].join("\n"));

  try {
    const result = await syncProvider({
      ...novitaAi,
      modelsDir,
      async fetchModels() {
        return {
          object: "list",
          data: [novitaAiModel({ id: "meta-llama/llama-3.3-70b-instruct" })],
        };
      },
    });
    expect(result.deleted).toBe(1);
    expect(await Bun.file(path.join(modelsDir, "deepseek", "deepseek-v3.2.toml")).exists()).toBe(false);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("Novita AI sync tracks remote-only IDs", () => {
  expect(providers["novita-ai"]).toBe(novitaAi);
  expect(groups.aggregators).toContain("novita-ai");
  expect(novitaAi.sourceID?.(novitaAiModel())).toBe("deepseek/deepseek-v3.2");
  expect(novitaAi.sourceID?.(novitaAiModel({ id: "novita/new-model" }))).toBe("novita/new-model");
});

test("Novita AI sync requires NOVITA_API_KEY", async () => {
  const original = process.env.NOVITA_API_KEY;
  delete process.env.NOVITA_API_KEY;
  try {
    await expect(novitaAi.fetchModels()).rejects.toThrow("Novita AI sync requires NOVITA_API_KEY");
  } finally {
    if (original !== undefined) process.env.NOVITA_API_KEY = original;
  }
});

test("fetchNovitaAIModels passes Authorization header", async () => {
  let request: Request | undefined;
  const fetcher = async (_url: string, _init?: RequestInit) => {
    request = new Request(_url, _init);
    return new Response(JSON.stringify({ object: "list", data: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  const result = await fetchNovitaAIModels("test-key", fetcher);
  expect(result).toEqual({ object: "list", data: [] });
  expect(request?.method).toBe("GET");
  expect(request?.url).toBe("https://api.novita.ai/openai/v1/models");
  expect(request?.headers.get("authorization")).toBe("Bearer test-key");
});

test("fetchNovitaAIModels throws on HTTP error", async () => {
  const fetcher = async () =>
    new Response("Unauthorized", { status: 401, statusText: "Unauthorized" });

  await expect(fetchNovitaAIModels("bad-key", fetcher)).rejects.toThrow("401 Unauthorized");
});
