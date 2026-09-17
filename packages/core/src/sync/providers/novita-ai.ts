import { z } from "zod";

import { describeModel } from "../../describe.js";
import type { ExistingModel, SyncProvider, SyncedFullModel, SyncedModel } from "../index.js";
import { factorBaseModel, resolveModelMetadataBaseModel } from "./openrouter.js";

const API_ENDPOINT = "https://api.novita.ai/openai/v1/models";
const BASE_MODEL_ALIASES: Record<string, string> = {
  "deepseek/deepseek_v3": "deepseek/deepseek-v3",
};
const Price = z.object({ price_per_m_decimal: z.string().optional() }).passthrough();
const Pricing = z.object({
  prompt: Price.optional(),
  completion: Price.optional(),
  input_cache_read: Price.optional(),
  input_cache_write: Price.optional(),
}).passthrough();

export const NovitaAIModel = z.object({
  id: z.string().min(1),
  object: z.literal("model"),
  created: z.number().int().nonnegative(),
  owned_by: z.string(),
  input_token_price_per_m: z.number().optional(),
  output_token_price_per_m: z.number().optional(),
  title: z.string().optional(),
  display_name: z.string().optional(),
  description: z.string().optional(),
  context_size: z.number().int().positive().optional(),
  max_output_tokens: z.number().int().positive().optional(),
  features: z.array(z.string()).optional(),
  input_modalities: z.array(z.string()).optional(),
  output_modalities: z.array(z.string()).optional(),
  pricing: Pricing.optional(),
  is_tiered_billing: z.boolean().optional(),
  tiered_billing_configs: z.array(z.object({
    min_tokens: z.number().int().nonnegative(),
    max_tokens: z.number().int().positive(),
    pricing: Pricing,
  }).passthrough()).optional(),
}).passthrough();

export const NovitaAIResponse = z.object({
  // Novita's endpoint currently omits the OpenAI-compatible top-level object.
  // Keep accepting the standard value if the API adds it later.
  object: z.literal("list").optional(),
  data: z.array(NovitaAIModel),
}).passthrough();

export type NovitaAIModel = z.infer<typeof NovitaAIModel>;

function decimalPrice(value: string | undefined) {
  if (value === undefined) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined;
}

type Modality = "text" | "audio" | "image" | "video" | "pdf";

function modalities(values: string[] | undefined, fallback: Modality[] | undefined) {
  if (values === undefined || values.length === 0) return fallback;
  const allowed = new Set<Modality>(["text", "audio", "image", "video", "pdf"]);
  const result = values
    .map((value) => value.toLowerCase() === "file" ? "pdf" : value.toLowerCase())
    .filter((value): value is Modality => allowed.has(value as Modality));
  return result.length > 0 ? [...new Set(result)] : fallback;
}

function dateFromTimestamp(timestamp: number) {
  return new Date(timestamp * 1000).toISOString().slice(0, 10);
}

function price(pricing: z.infer<typeof Pricing> | undefined) {
  const input = decimalPrice(pricing?.prompt?.price_per_m_decimal);
  const output = decimalPrice(pricing?.completion?.price_per_m_decimal);
  if (input === undefined || output === undefined) return undefined;
  return {
    input,
    output,
    cache_read: decimalPrice(pricing?.input_cache_read?.price_per_m_decimal),
    cache_write: decimalPrice(pricing?.input_cache_write?.price_per_m_decimal),
  };
}

function cost(model: NovitaAIModel, existing: ExistingModel | undefined) {
  if (model.is_tiered_billing !== true) {
    // Novita uses zero top-level prices without a pricing object for free models.
    if (model.pricing === undefined && model.input_token_price_per_m === 0 && model.output_token_price_per_m === 0) {
      return { input: 0, output: 0 };
    }
    return price(model.pricing) ?? existing?.cost;
  }
  const bands = [...model.tiered_billing_configs ?? []].sort((a, b) => a.min_tokens - b.min_tokens);
  if (bands.length === 0 || bands[0]?.min_tokens > 1 || bands.some((band, index) =>
    band.max_tokens <= band.min_tokens || (index > 0 && band.min_tokens <= bands[index - 1]!.min_tokens)
  )) return existing?.cost;
  const base = price(bands[0]!.pricing);
  if (base === undefined || bands.some((band) => price(band.pricing) === undefined)) return existing?.cost;
  return {
    ...base,
    tiers: bands.slice(1).map((band) => ({
      ...price(band.pricing)!,
      tier: { type: "context" as const, size: band.min_tokens },
    })),
  };
}

function buildNovitaModel(model: NovitaAIModel, existing: ExistingModel | undefined, resolved: ExistingModel | undefined): SyncedModel | undefined {
  const baseModel = existing?.base_model ?? BASE_MODEL_ALIASES[model.id] ?? resolveModelMetadataBaseModel(model.id);
  // New provider entries require a lab model. Do not create fabricated inline lab facts.
  if (existing === undefined && baseModel === undefined) return undefined;
  const name = model.display_name ?? model.title ?? existing?.name ?? model.id;
  const input = modalities(model.input_modalities, resolved?.modalities?.input) ?? ["text"];
  const output = modalities(model.output_modalities, resolved?.modalities?.output) ?? ["text"];
  const features = model.features === undefined ? undefined : new Set(model.features);
  const reasoning = features?.has("reasoning") ?? resolved?.reasoning ?? false;
  const toolCall = features?.has("function-calling") ?? resolved?.tool_call ?? false;
  const structuredOutput = features?.has("structured-outputs") ?? resolved?.structured_output ?? false;
  const context = model.context_size ?? resolved?.limit?.context ?? 0;
  const outputLimit = model.max_output_tokens ?? resolved?.limit?.output ?? context;
  const modelCost = cost(model, existing);
  // DeepSeek R1 is fixed-reasoning on Novita, as with its already curated R1 variants.
  const reasoningOptions = existing?.reasoning_options ?? (model.id === "deepseek/deepseek-r1" ? [] : undefined);
  if (existing === undefined && (modelCost === undefined || (reasoning && reasoningOptions === undefined))) return undefined;
  const values: SyncedFullModel = {
    name,
    description: model.description || existing?.description || describeModel({ id: model.id, name, reasoning, tool_call: toolCall, structured_output: structuredOutput || undefined, open_weights: existing?.open_weights ?? false, limit: { context, output: outputLimit }, modalities: { input, output } }),
    family: existing?.family,
    release_date: existing?.release_date ?? dateFromTimestamp(model.created),
    last_updated: existing?.last_updated ?? dateFromTimestamp(model.created),
    attachment: input.some((value) => value !== "text"),
    reasoning,
    tool_call: toolCall,
    structured_output: structuredOutput,
    temperature: existing?.temperature,
    open_weights: existing?.open_weights ?? false,
    cost: modelCost,
    limit: { context, output: outputLimit },
    modalities: { input, output },
  };
  if (baseModel !== undefined) return factorBaseModel(baseModel, {
    ...values,
    // These are lab facts, not claims made by the Novita catalog endpoint.
    open_weights: existing?.open_weights,
    release_date: existing?.release_date,
    last_updated: existing?.last_updated,
    temperature: existing?.temperature,
    reasoning_options: reasoningOptions,
    interleaved: existing?.interleaved,
  }, values.limit, existing?.base_model_omit);
  return {
    ...existing,
    ...values,
    reasoning_options: existing?.reasoning_options,
    interleaved: existing?.interleaved,
    status: existing?.status,
    knowledge: existing?.knowledge,
  } as SyncedModel;
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
  // The endpoint exposes the metadata needed to author new provider models.
  skipCreates: false,
  deleteMissing: true,
  trackMissingModels: false,
  sourceID(model) {
    return model.id;
  },
  skippedNotice(ids) {
    return ids.length === 0 ? [] : [`Novita models needing lab metadata, pricing, or verified reasoning controls: ${ids.join(", ")}`];
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
    const translated = buildNovitaModel(model, context.authored(model.id), context.existing(model.id));
    return translated === undefined ? undefined : { id: model.id, model: translated };
  },
} satisfies SyncProvider<NovitaAIModel>;
