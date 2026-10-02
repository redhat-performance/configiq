import type { ModelFacets, ModelTypeFacet } from '@/lib/model-metadata'
import { checkpointName, normalizeModelId } from '@/lib/model-metadata'

/**
 * Scored, forgiving search over model facets.
 *
 * The old filter was a substring test against the raw Hugging Face ID, so it
 * could only find a model you could already spell. This scores each model
 * against every facet — vendor, family, parameter count, quantization, type,
 * context length — so "70b", "vision", "nvfp4" and "nemotron" all land, and a
 * small typo still hits via a bounded edit distance.
 *
 * Multi-term queries are AND: every term must match something, and the score
 * is the sum of each term's best field match. A term that matches nothing
 * eliminates the model, which keeps "70b vision" from returning every 70B
 * text model.
 */

/** Field weights, highest first. A term takes its single best field. */
const SCORE = {
  exactId: 1000,
  exactFamily: 700,
  idPrefix: 400,
  familyPrefix: 350,
  familyContains: 300,
  checkpointContains: 250,
  params: 240,
  vendor: 200,
  context: 190,
  quantization: 180,
  type: 160,
  tested: 150,
  typo: 90,
  subsequence: 50,
} as const

/** Words a user might type for each type facet. */
const TYPE_SYNONYMS: Record<ModelTypeFacet, string[]> = {
  vision: ['vision', 'multimodal', 'vl', 'image', 'visual'],
  moe: ['moe', 'mixture', 'experts', 'expert', 'sparse'],
  text: ['text', 'text-only', 'textonly', 'language'],
}

/** Bit-width shorthands that should match a family of quantization formats. */
const QUANT_SHORTHAND: Record<string, string[]> = {
  '4bit': ['INT4', 'NVFP4', 'MXFP4', 'FP4', 'AWQ', 'GPTQ', 'W4A16'],
  '4-bit': ['INT4', 'NVFP4', 'MXFP4', 'FP4', 'AWQ', 'GPTQ', 'W4A16'],
  '8bit': ['FP8', 'INT8', 'W8A8'],
  '8-bit': ['FP8', 'INT8', 'W8A8'],
}

/** Terms meaning "any quantized checkpoint" / "no quantization". */
const ANY_QUANT = new Set(['quantized', 'quant', 'compressed'])
const NO_QUANT = new Set(['unquantized', 'full-precision', 'fp32'])

export function tokenizeQuery(query: string): string[] {
  return query.toLowerCase().trim().split(/[\s,]+/).filter(Boolean)
}

/**
 * Parse a size term into billions of parameters. Accepts 70b, 70, 0.6b, 480m
 * and 2.4t so that a user can type the size however they say it.
 */
function parseParamTerm(term: string): number | null {
  const match = term.match(/^(\d+(?:\.\d+)?)(b|m|t)?$/)
  if (!match) return null
  const value = Number(match[1])
  if (!Number.isFinite(value)) return null
  switch (match[2]) {
    case 'm': return value / 1_000
    case 't': return value * 1_000
    default: return value
  }
}

/** Parse a context term (128k, 1m, 32768) into a token count. */
function parseContextTerm(term: string): number | null {
  const match = term.match(/^(\d+(?:\.\d+)?)(k|m)$/)
  if (match) {
    const value = Number(match[1])
    if (!Number.isFinite(value)) return null
    return match[2] === 'k' ? value * 1024 : value * 1024 * 1024
  }
  // A bare integer is only a plausible context length once it is large enough
  // that it cannot be a parameter count.
  const bare = Number(term)
  return Number.isInteger(bare) && bare >= 1024 ? bare : null
}

/** Whether two parameter counts are close enough to be the same model size. */
function paramsMatch(actual: number, wanted: number): boolean {
  // 70 should match 70.6 but not 7 or 700. Scale the tolerance so small models
  // stay distinguishable (4 must not match 8) while large ones tolerate the
  // rounding vendors publish.
  return Math.abs(actual - wanted) <= Math.max(0.5, wanted * 0.05)
}

/** Levenshtein distance, abandoned once it provably exceeds `max`. */
function boundedLevenshtein(a: string, b: string, max: number): number {
  if (a === b) return 0
  if (Math.abs(a.length - b.length) > max) return max + 1
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i)
  for (let i = 1; i <= a.length; i++) {
    const curr = [i]
    let rowMin = i
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
      if (curr[j] < rowMin) rowMin = curr[j]
    }
    if (rowMin > max) return max + 1
    prev = curr
  }
  return prev[b.length]
}

/** Typo tolerance scales with term length — no fuzz on very short terms. */
function typoBudget(term: string): number {
  if (term.length <= 3) return 0
  return term.length <= 6 ? 1 : 2
}

/** Whether every character of `term` appears in `target` in order. */
function isSubsequence(term: string, target: string): boolean {
  let i = 0
  for (const ch of target) {
    if (ch === term[i]) i++
    if (i === term.length) return true
  }
  return term.length === 0
}

/**
 * Best score for a single query term against one model. Zero means the term
 * does not describe this model at all.
 */
function scoreTerm(item: ModelFacets, term: string): number {
  const id = normalizeModelId(item.id)
  const family = item.family.toLowerCase()
  const checkpoint = checkpointName(item.id).toLowerCase()
  const vendor = item.vendor.toLowerCase()

  if (id === term) return SCORE.exactId
  if (family === term) return SCORE.exactFamily
  if (id.startsWith(term) || checkpoint.startsWith(term)) return SCORE.idPrefix
  if (family.startsWith(term)) return SCORE.familyPrefix
  if (family.includes(term)) return SCORE.familyContains
  if (id.includes(term)) return SCORE.checkpointContains

  const wantedParams = parseParamTerm(term)
  if (wantedParams !== null) {
    if (item.paramsB !== null && paramsMatch(item.paramsB, wantedParams)) return SCORE.params
    if (item.activeParamsB !== null && paramsMatch(item.activeParamsB, wantedParams)) return SCORE.params
  }

  if (vendor && (vendor === term || vendor.startsWith(term))) return SCORE.vendor

  const wantedContext = parseContextTerm(term)
  if (wantedContext !== null && item.contextLength !== null &&
      Math.abs(item.contextLength - wantedContext) <= wantedContext * 0.1) {
    return SCORE.context
  }

  const quant = item.quantization?.toUpperCase() ?? null
  if (quant) {
    if (quant.toLowerCase() === term || quant.toLowerCase().startsWith(term)) return SCORE.quantization
    if (QUANT_SHORTHAND[term]?.includes(quant)) return SCORE.quantization
    if (ANY_QUANT.has(term)) return SCORE.quantization
  } else if (NO_QUANT.has(term)) {
    return SCORE.quantization
  }

  for (const facet of item.types) {
    if (TYPE_SYNONYMS[facet].includes(term)) return SCORE.type
  }
  // "dense" is the absence of the moe facet rather than a facet of its own.
  if (term === 'dense' && !item.types.includes('moe')) return SCORE.type

  if (term === 'tested' && item.isTested) return SCORE.tested
  if (term === 'catalog' && item.inCatalog) return SCORE.tested

  const budget = typoBudget(term)
  if (budget > 0) {
    const words = checkpoint.split(/[-_.]+/).filter(Boolean)
    for (const word of [...words, family, vendor]) {
      if (word && boundedLevenshtein(term, word, budget) <= budget) return SCORE.typo
    }
  }

  if (term.length >= 3 && isSubsequence(term, checkpoint)) return SCORE.subsequence

  return 0
}

/**
 * Total score for a model against a full query. Zero means at least one term
 * matched nothing, so the model is not a result.
 */
export function scoreModel(item: ModelFacets, query: string): number {
  const terms = tokenizeQuery(query)
  if (terms.length === 0) return 0

  let total = 0
  for (const term of terms) {
    const score = scoreTerm(item, term)
    if (score === 0) return 0
    total += score
  }

  // Nudge, not a reorder: only breaks ties between equally good text matches.
  if (item.isTested) total += 5
  else if (item.inCatalog) total += 2
  return total
}

/**
 * Filter and rank `items` against `query`. An empty query returns the input
 * untouched so the caller can keep its own browse ordering.
 */
export function searchModels<T extends ModelFacets>(items: T[], query: string): T[] {
  if (!query.trim()) return items
  return items
    .map(item => ({ item, score: scoreModel(item, query) }))
    .filter(entry => entry.score > 0)
    .sort((a, b) => b.score - a.score || a.item.id.localeCompare(b.item.id))
    .map(entry => entry.item)
}
