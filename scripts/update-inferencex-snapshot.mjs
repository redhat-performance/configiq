import { readFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { parseInferenceXImageTag } from './inferencex-image-tag.mjs'

const apiBase = 'https://inferencex.semianalysis.com/api/v1'
const availabilityUrl = `${apiBase}/availability`
const optionsUrl = `${apiBase}/views/options`
const appConfigPath = resolve('public/config.json')
const inputDirectory = process.argv[2] === '--from-dir' ? process.argv[3] : null

// Only verified InferenceX IDs can be matched against ConfigIQ's approved models.
const modelIds = {
  dsv4: 'deepseek-ai/DeepSeek-V4-Pro',
  gptoss120b: 'openai/gpt-oss-120b',
  'kimik2.5': 'moonshotai/Kimi-K2.5',
  glm5: 'zai-org/GLM-5',
  'glm5.1': 'zai-org/GLM-5.1',
  'glm5.3': 'zai-org/GLM-5.3',
  'minimaxm2.5': 'MiniMaxAI/MiniMax-M2.5',
  minimaxm3: 'MiniMaxAI/MiniMax-M3',
  'qwen3.5': 'Qwen/Qwen3.5-397B-A17B',
}

const hardwareNames = {
  b200: 'B200', b300: 'B300', gb200: 'GB200', gb300: 'GB300',
  h100: 'H100', h200: 'H200', mi300x: 'MI300X', mi325x: 'MI325X',
  mi355x: 'MI355X', rtx6000pro: 'RTX PRO 6000', tpuv7: 'TPU v7',
}

async function getJson(url) {
  const response = await fetch(url, { headers: { Accept: 'application/json' }, signal: AbortSignal.timeout(60_000) })
  if (!response.ok) throw new Error(`InferenceX returned HTTP ${response.status} for ${url}`)
  return response.json()
}

async function getDiscovery(name, url) {
  return inputDirectory
    ? JSON.parse(await readFile(join(inputDirectory, name), 'utf8'))
    : getJson(url)
}

const availability = await getDiscovery('availability.json', availabilityUrl)
const options = await getDiscovery('options.json', optionsUrl)
const appConfig = JSON.parse(await readFile(appConfigPath, 'utf8'))
if (!Array.isArray(availability) || !Array.isArray(options.models)) {
  throw new Error('InferenceX discovery response changed')
}
// `approvedModels` is the durable name; `testedModels` is the transition key.
const approvedModelIds = appConfig.approvedModels ?? appConfig.testedModels
if (!Array.isArray(approvedModelIds) || approvedModelIds.length === 0
  || !approvedModelIds.every(model => typeof model === 'string')) {
  throw new Error('ConfigIQ approved model list is missing or invalid')
}

const approvedModels = new Set(approvedModelIds)
const skippedMappings = Object.entries(modelIds)
  .filter(([, modelId]) => !approvedModels.has(modelId))
  .map(([sourceModel]) => sourceModel)
if (skippedMappings.length > 0) {
  console.warn(`InferenceX mappings skipped because the model is not approved: ${skippedMappings.join(', ')}`)
}
const approvedSourceModels = new Set(Object.entries(modelIds)
  .filter(([, modelId]) => approvedModels.has(modelId))
  .map(([sourceModel]) => sourceModel))
const availableKeys = new Set(availability
  .filter(row => row.benchmark_type === 'single_turn' && row.isl === 8192 && row.osl === 1024
    && approvedSourceModels.has(row.model))
  .map(row => row.model))
const models = options.models.filter(model => model.dbKeys.some(key => availableKeys.has(key)))
const benchmarkUrls = models.map(model => ({
  name: model.name,
  slug: model.compareSlug,
  url: `${apiBase}/benchmarks?model=${encodeURIComponent(model.name)}`,
}))

async function getModelResults(model) {
  const results = inputDirectory
    ? JSON.parse(await readFile(join(inputDirectory, `${model.slug}.json`), 'utf8')).rows
    : await getJson(model.url)
  if (!Array.isArray(results)) throw new Error(`InferenceX benchmark response changed for ${model.name}`)
  return results
}

const responses = []
for (let index = 0; index < benchmarkUrls.length; index += 3) {
  responses.push(...await Promise.all(benchmarkUrls.slice(index, index + 3).map(getModelResults)))
}

const finite = value => typeof value === 'number' && Number.isFinite(value) && value >= 0
const positiveInteger = value => Number.isInteger(value) && value > 0
const rows = responses.flat().flatMap(result => {
  const metrics = result.metrics ?? {}
  if (result.benchmark_type !== 'single_turn' || result.isl !== 8192 || result.osl !== 1024
    || !availableKeys.has(result.model) || !positiveInteger(result.conc)
    || !['model', 'hardware', 'framework', 'precision', 'date', 'run_url'].every(key => typeof result[key] === 'string' && result[key])
    || !positiveInteger(result.decode_tp) || !positiveInteger(result.num_decode_gpu)
    || (result.disagg && (!positiveInteger(result.prefill_tp) || !positiveInteger(result.num_prefill_gpu)))
    || !['median_ttft', 'median_tpot', 'median_e2el', 'output_tput_per_gpu'].every(key => finite(metrics[key]))) return []

  const model = modelIds[result.model]
  const accelerator = hardwareNames[result.hardware] ?? String(result.hardware).toUpperCase()
  const physicalGpus = result.disagg
    ? result.num_prefill_gpu + result.num_decode_gpu
    : Math.max(result.num_prefill_gpu ?? 0, result.num_decode_gpu)
  if (!positiveInteger(physicalGpus)) return []
  const prefillTp = result.disagg ? result.prefill_tp : (result.prefill_tp ?? result.decode_tp)
  const prefillEp = result.disagg ? result.prefill_ep : (result.prefill_ep ?? result.decode_ep)
  const topology = [
    result.disagg ? 'disaggregated' : 'aggregated',
    result.is_multinode ? 'multi-node' : 'single-node',
    `P TP${prefillTp}${positiveInteger(prefillEp) ? `/EP${prefillEp}` : ''}`,
    `D TP${result.decode_tp}${positiveInteger(result.decode_ep) ? `/EP${result.decode_ep}` : ''}`,
    `${physicalGpus} GPUs`,
    result.disagg ? `workers P${result.prefill_num_workers}/D${result.decode_num_workers}` : null,
    result.prefill_dp_attention || result.decode_dp_attention
      ? `DP attention P${Boolean(result.prefill_dp_attention)}/D${Boolean(result.decode_dp_attention)}` : null,
  ].filter(Boolean).join(' · ')
  const imageTag = parseInferenceXImageTag(result.image)
  return [{
    benchmark_source: 'inferencex',
    source_id: result.id,
    source_run_url: result.run_url,
    measurement_date: result.date,
    source_model: result.model,
    source_hardware: result.hardware,
    accelerator,
    model,
    version: `InferenceX ${String(result.precision).toUpperCase()}`,
    backend: result.framework,
    backend_version: imageTag,
    precision: result.precision,
    spec_decoding: result.spec_method,
    request_type: 'single_turn',
    inferencex_topology: topology,
    inferencex_curve: `${result.curve_date ?? result.date} · curve ${result.curve_workflow_run_id}`,
    inferencex_offload: result.offload_mode ?? null,
    isl: result.isl,
    osl: result.osl,
    tp: result.decode_tp,
    tp_prefill: prefillTp,
    tp_decode: result.decode_tp,
    concurrency: result.conc,
    ttft_median: metrics.median_ttft * 1000,
    tpot_median: metrics.median_tpot * 1000,
    request_latency_median: metrics.median_e2el,
    'output_tok/sec': metrics.output_tput_per_gpu * physicalGpus,
  }]
})
rows.sort((left, right) => left.model.localeCompare(right.model) || left.accelerator.localeCompare(right.accelerator)
  || left.version.localeCompare(right.version) || left.tp - right.tp || left.concurrency - right.concurrency)
if (rows.length === 0) throw new Error('No 8k/1k single-turn rows found; snapshot was not changed')

process.stdout.write(`${JSON.stringify({
  sourceUrl: 'https://inferencex.semianalysis.com/api',
  availabilityUrl,
  benchmarkUrls: benchmarkUrls.map(model => model.url),
  modelFilter: 'public/config.json#approvedModels',
  retrievedAt: new Date().toISOString(),
  workload: { isl: 8192, osl: 1024, benchmarkType: 'single_turn' },
  rows,
})}\n`)
