import { z } from "zod";

import { AuthoredModel } from "../../schema.js";
import type { ExistingModel, SyncProvider, SyncedBaseModel, SyncedModel } from "../index.js";

const API_ENDPOINT = "https://api.novita.ai/openai/v1/models";

export const NovitaAIModel = z.object({
  id: z.string().min(1),
  object: z.literal("model"),
  created: z.number().int().nonnegative(),
  owned_by: z.string(),
  title: z.string().optional(),
  display_name: z.string().optional(),
  description: z.string().optional(),
  context_size: z.number().int().positive().optional(),
  max_output_tokens: z.number().int().positive().optional(),
  features: z.array(z.string()).optional(),
  input_modalities: z.array(z.string()).optional(),
  output_modalities: z.array(z.string()).optional(),
  pricing: z.object({
    prompt: z.object({ price_per_m_decimal: z.string().optional() }).passthrough().optional(),
    completion: z.object({ price_per_m_decimal: z.string().optional() }).passthrough().optional(),
    input_cache_read: z.object({ price_per_m_decimal: z.string().optional() }).passthrough().optional(),
  }).passthrough().optional(),
}).passthrough();

export const NovitaAIResponse = z.object({
  // Novita's endpoint currently omits the OpenAI-compatible top-level object.
  // Keep accepting the standard value if the API adds it later.
  object: z.literal("list").optional(),
  data: z.array(NovitaAIModel),
}).passthrough();

export type NovitaAIModel = z.infer<typeof NovitaAIModel>;

function preserveAuthoredModel(id: string, authored: ExistingModel): SyncedModel {
  if (authored.base_model !== undefined) return authored as SyncedBaseModel;

  const parsed = AuthoredModel.safeParse({ id, ...authored });
  if (!parsed.success) {
    parsed.error.cause = { provider: "novita-ai", model: id };
    throw parsed.error;
  }
  const { id: _id, ...model } = parsed.data;
  return model;
}

function decimalPrice(value: string | undefined) {
  if (value === undefined) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined;
}

function modalities(values: string[] | undefined, fallback: ExistingModel["modalities"] | undefined) {
  if (values === undefined || values.length === 0) return fallback;
  const allowed = new Set(["text", "audio", "image", "video", "pdf"]);
  const result = values.map((value) => value.toLowerCase()).filter((value) => allowed.has(value));
  return result.length > 0 ? [...new Set(result)] : fallback;
}

export async function fetchNovitaAIModels(key: string, fetcher: typeof fetch = fetch) {
  const response = await fetcher(API_ENDPOINT, {
    method: "GET",
    headers: { Authorization: `Bearer ${key}` },
  });
  if (!response.ok) {
    throw new Error(`Novita AI models request failed: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

export const novitaAi = {
  id: "novita-ai",
  name: "Novita AI",
  modelsDir: "providers/novita-ai/models",
  skipCreates: true,
  deleteMissing: false,
  sourceID(model) {
    return model.id;
  },
  skippedNotice(ids) {
    if (ids.length === 0) return [];
    return [
      `${ids.length} Novita AI models returned by the API were not created because the catalog requires hand-authored metadata for new models.`,
      `Skipped remote IDs: ${ids.map((id) => `\`${id}\``).join(", ")}`,
    ];
  },
  async fetchModels() {
    const key = process.env.NOVITA_API_KEY;
    if (key === undefined) throw new Error("Novita AI sync requires NOVITA_API_KEY");
    return fetchNovitaAIModels(key);
  },
  parseModels(raw) {
    return NovitaAIResponse.parse(raw).data;
  },
  translateModel(model, context) {
    const authored = context.authored(model.id);
    if (authored === undefined) return undefined;
    const translated = { ...preserveAuthoredModel(model.id, authored) } as Record<string, unknown>;
    if (model.display_name ?? model.title) translated.name = model.display_name ?? model.title;
    if (model.description) translated.description = model.description;
    if (model.context_size !== undefined || model.max_output_tokens !== undefined) {
      translated.limit = {
        ...authored.limit,
        ...(model.context_size !== undefined ? { context: model.context_size } : {}),
        ...(model.max_output_tokens !== undefined ? { output: model.max_output_tokens } : {}),
      };
    }
    const input = modalities(model.input_modalities, authored.modalities?.input);
    const output = modalities(model.output_modalities, authored.modalities?.output);
    if (input !== undefined && output !== undefined) translated.modalities = { input, output };
    const features = new Set(model.features ?? []);
    if (model.features !== undefined) {
      translated.reasoning = features.has("reasoning");
      translated.tool_call = features.has("function-calling");
      translated.structured_output = features.has("structured-outputs");
    }
    const pricing = model.pricing;
    const inputCost = decimalPrice(pricing?.prompt?.price_per_m_decimal);
    const outputCost = decimalPrice(pricing?.completion?.price_per_m_decimal);
    const cacheRead = decimalPrice(pricing?.input_cache_read?.price_per_m_decimal);
    if (inputCost !== undefined || outputCost !== undefined || cacheRead !== undefined) {
      translated.cost = {
        ...authored.cost,
        ...(inputCost !== undefined ? { input: inputCost } : {}),
        ...(outputCost !== undefined ? { output: outputCost } : {}),
        ...(cacheRead !== undefined ? { cache_read: cacheRead } : {}),
      };
    }
    return { id: model.id, model: translated as SyncedModel };
  },
} satisfies SyncProvider<NovitaAIModel>;
