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
    authored: () => ({ base_model: "test/base", name: "Old", description: "Old", attachment: false, reasoning: false, tool_call: false, open_weights: true, limit: { context: 1, output: 1 }, modalities: { input: ["text"], output: ["text"] } }),
  });
  expect(translated?.model).toMatchObject({ name: "GLM 5.3 Flash", reasoning: true, tool_call: true, structured_output: true, limit: { context: 1_048_576, output: 131_072 }, cost: { input: 0.15, output: 0.5, cache_read: 0.03 }, modalities: { input: ["text", "image"], output: ["text"] } });
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

  expect(translated).toMatchObject({ id: "deepseek/deepseek-v3.2", model: authored });
});

test("Novita AI sync creates unknown remote models from API metadata", () => {
  const translated = novitaAi.translateModel(novitaAiModel({ id: "novita/unknown-model", context_size: 8192, max_output_tokens: 4096, pricing: { prompt: { price_per_m_decimal: "0.1" }, completion: { price_per_m_decimal: "0.2" } } }), {
    existing: () => undefined,
    authored: () => undefined,
  });
  expect(translated?.id).toBe("novita/unknown-model");
  expect(translated?.model).toMatchObject({ limit: { context: 8192, output: 4096 }, cost: { input: 0.1, output: 0.2 } });
});

test("Novita AI sync retains local models absent from API response", async () => {
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
    expect(result.deleted).toBe(0);
    expect(result.unchanged).toBe(1);
    expect(await Bun.file(path.join(modelsDir, "deepseek", "deepseek-v3.2.toml")).exists()).toBe(true);
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
