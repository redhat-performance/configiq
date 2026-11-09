import { z } from 'zod'

const rate = z.number().nonnegative()
const tokenLimit = z.number().int().positive().nullable().optional()
const billingSchema = z.object({
  currency: z.literal('USD'),
  request_price: rate.default(0),
  input_cache_read_per_million: rate.optional(),
  input_cache_write_per_million: rate.optional(),
  // Conditions are evaluated per representative request, never monthly totals.
  overrides: z.array(z.object({
    min_prompt_tokens: z.number().int().nonnegative(),
    input_per_million: rate.optional(),
    output_per_million: rate.optional(),
    request_price: rate.optional(),
  }).strict()).default([]),
  unsupported: z.array(z.string()).default([]),
}).strict()

export const hostedModelSchema = z.object({
  id: z.string().trim().min(1),
  name: z.string().trim().min(1),
  provider: z.string(),
  tier: z.enum(['fast', 'balanced', 'frontier']),
  price_per_m_input: rate,
  price_per_m_output: rate,
  context_window: tokenLimit,
  updated_at: z.string().nullable().optional(),
  source: z.enum(['openrouter', 'litellm', 'override']).optional(),
  hugging_face_id: z.string().nullable().optional(),
  model_author: z.string().optional(),
  serving_provider: z.string().nullable().optional(),
  max_input_tokens: tokenLimit,
  max_output_tokens: tokenLimit,
  max_context_tokens: tokenLimit,
  billing: billingSchema.optional(),
})

export type FrontierModel = z.infer<typeof hostedModelSchema>

/** Reject a corrupt refresh as a unit so the hook retains its last-good data. */
export function parseHostedModels(raw: unknown) {
  return z.object({
    models: z.array(hostedModelSchema),
    updated_at: z.string().nullable().optional(),
    stale: z.boolean().optional(),
  }).parse(raw)
}
