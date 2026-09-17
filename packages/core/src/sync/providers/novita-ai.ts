import { z } from "zod";
import path from "node:path";

import { describeModel } from "../../describe.js";
import { inferKimiFamily, ModelFamilyValues } from "../../family.js";
import type { ExistingModel, SyncProvider, SyncedFullModel, SyncedModel } from "../index.js";

const API_ENDPOINT = "https://api.novita.ai/openai/v1/models";
const MODELS_DIR = path.join(import.meta.dirname, "..", "..", "..", "..", "..", "models");

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

function inferFamily(id: string, name: string) {
  const kimi = inferKimiFamily(id, name);
  if (kimi !== undefined) return kimi;
  const target = `${id} ${name}`.toLowerCase();
  return [...ModelFamilyValues].sort((a, b) => b.length - a.length).find((family) =>
    new RegExp(`(^|[^a-z0-9])${family.toLowerCase().replace(/[.*+?^${}()|[\\]\\]/g, "\\$&")}(?=$|[^a-z0-9])`).test(target));
}

function buildNovitaModel(model: NovitaAIModel, existing: ExistingModel | undefined): SyncedModel {
  const name = model.display_name ?? model.title ?? existing?.name ?? model.id;
  const input = modalities(model.input_modalities, existing?.modalities?.input) ?? ["text"];
  const output = modalities(model.output_modalities, existing?.modalities?.output) ?? ["text"];
  const features = model.features === undefined ? undefined : new Set(model.features);
  const reasoning = features?.has("reasoning") ?? existing?.reasoning ?? false;
  const toolCall = features?.has("function-calling") ?? existing?.tool_call ?? false;
  const structuredOutput = features?.has("structured-outputs") ?? existing?.structured_output ?? false;
  const context = model.context_size ?? existing?.limit?.context ?? 0;
  const outputLimit = model.max_output_tokens ?? existing?.limit?.output ?? context;
  const inputCost = decimalPrice(model.pricing?.prompt?.price_per_m_decimal);
  const outputCost = decimalPrice(model.pricing?.completion?.price_per_m_decimal);
  const cacheRead = decimalPrice(model.pricing?.input_cache_read?.price_per_m_decimal);
  const cost = inputCost !== undefined && outputCost !== undefined
    ? { input: inputCost, output: outputCost, cache_read: cacheRead }
    : existing?.cost ?? { input: 0, output: 0 };
  const values: SyncedFullModel = {
    name,
    description: model.description ?? existing?.description ?? describeModel({ id: model.id, name, family: inferFamily(model.id, name), reasoning, tool_call: toolCall, structured_output: structuredOutput || undefined, open_weights: existing?.open_weights ?? true, limit: { context, output: outputLimit }, modalities: { input, output } }),
    family: existing?.family ?? inferFamily(model.id, name),
    release_date: existing?.release_date ?? dateFromTimestamp(model.created),
    last_updated: existing?.last_updated ?? dateFromTimestamp(model.created),
    attachment: input.some((value) => value !== "text"),
    reasoning,
    tool_call: toolCall,
    structured_output: structuredOutput,
    temperature: existing?.temperature ?? true,
    open_weights: existing?.open_weights ?? true,
    cost,
    limit: { context, output: outputLimit },
    modalities: { input, output },
  };
  if (existing?.base_model === undefined) return values;
  return {
    ...existing,
    ...values,
    base_model: existing.base_model,
    ...(existing.base_model_omit === undefined ? {} : { base_model_omit: existing.base_model_omit }),
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
  deleteMissing: false,
  sourceID(model) {
    return model.id;
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
    return { id: model.id, model: buildNovitaModel(model, context.authored(model.id)) };
  },
} satisfies SyncProvider<NovitaAIModel>;
