/**
 * Client for the deployment artifact service.
 *
 * The renderer turns an `include=engine` section into a pinned, render-checked
 * bundle. Nothing is sized here: the request carries the evaluated limits and
 * the workload they were measured against, and the service either reproduces
 * them or refuses with a reason. The browser never assembles manifests.
 */

import { parseEngineSection, type EngineSection } from './engine'

export class DeployError extends Error {
  constructor(
    public readonly code: string,
    message: string,
  ) {
    super(message)
    this.name = 'DeployError'
  }
}

/** One target the service can render, with its pins and availability. */
export interface DeployTargetInfo {
  id: string
  name: string
  description: string
  kind: string
  /** Simulator whose evaluated limits this target renders. */
  simulator: string
  modes: Array<'agg' | 'disagg'>
  backends: string[]
  /** Engine releases the target's pinned image runs. */
  engineVersions: string[]
  /** Release pins the artifact records, verbatim. */
  pins: Record<string, string>
  available: boolean
  unavailableReason: string | null
  artifactSchemaVersion: string
  experimental: boolean
}

export interface DeployFile {
  /** Path relative to the bundle root. */
  path: string
  contents: string
  /** POSIX mode; scripts are `0o755` so a zip extract keeps them runnable. */
  mode: number
}

export type DeployCheckId =
  | 'kustomize-build'
  | 'resources'
  | 'engine-limits'
  | 'images'
  | 'placeholders'

export interface DeployCheck {
  id: DeployCheckId | string
  status: 'passed' | 'skipped' | 'failed'
  detail: string
}

export interface DeployArtifact {
  target: string
  artifactSchemaVersion: string
  engine: EngineSection
  pins: Record<string, string>
  checks: DeployCheck[]
  warnings: string[]
  files: DeployFile[]
}

/**
 * One render request.
 *
 * `engine` comes from the result page, together with the workload and system
 * that produced it. The rest are explicit choices the user made in the deploy
 * dialog; an omitted optional field is resolved by the service, never by this
 * client.
 */
export interface DeployRequestInput {
  target: string
  engine: EngineSection
  system: string
  isl: number
  osl: number
  concurrency?: number | null
  namespace?: string | null
  namePrefix?: string | null
  image?: string | null
  allowEngineVersionMismatch?: boolean
  servedModelName?: string | null
  modelConfigData?: Record<string, unknown> | null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function errorMessage(payload: unknown, fallback: string): { code: string; message: string } {
  if (isRecord(payload)) {
    const error = payload.error
    if (isRecord(error) && typeof error.message === 'string') {
      return {
        code: typeof error.code === 'string' ? error.code : 'DEPLOY_FAILED',
        message: error.message,
      }
    }
    if (typeof payload.detail === 'string' && payload.detail.length > 0) {
      return { code: 'DEPLOY_FAILED', message: payload.detail }
    }
    if (typeof payload.message === 'string' && payload.message.length > 0) {
      return { code: 'DEPLOY_FAILED', message: payload.message }
    }
  }
  return { code: 'DEPLOY_FAILED', message: fallback }
}

async function readError(res: Response, fallback: string): Promise<DeployError> {
  let payload: unknown = null
  try {
    payload = await res.json()
  } catch {
    // A non-JSON error body keeps the HTTP fallback message.
  }
  const { code, message } = errorMessage(payload, fallback)
  return new DeployError(code, message)
}

function parseTargets(payload: unknown): DeployTargetInfo[] {
  if (!isRecord(payload) || !Array.isArray(payload.targets)) {
    throw new DeployError('DEPLOY_INVALID_RESPONSE', 'The deployment target catalog was not a list of targets.')
  }
  return payload.targets.filter(isRecord).map(raw => {
    const modes = Array.isArray(raw.modes)
      ? raw.modes.filter((mode): mode is 'agg' | 'disagg' => mode === 'agg' || mode === 'disagg')
      : []
    const strings = (value: unknown): string[] =>
      Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
    const pins = isRecord(raw.pins)
      ? Object.fromEntries(Object.entries(raw.pins).filter((entry): entry is [string, string] => typeof entry[1] === 'string'))
      : {}
    return {
      id: typeof raw.id === 'string' ? raw.id : '',
      name: typeof raw.name === 'string' ? raw.name : (typeof raw.id === 'string' ? raw.id : 'Deployment target'),
      description: typeof raw.description === 'string' ? raw.description : '',
      kind: typeof raw.kind === 'string' ? raw.kind : '',
      simulator: typeof raw.simulator === 'string' ? raw.simulator : '',
      modes,
      backends: strings(raw.backends),
      engineVersions: strings(raw.engine_versions),
      pins,
      available: raw.available === true,
      unavailableReason: typeof raw.unavailable_reason === 'string' ? raw.unavailable_reason : null,
      artifactSchemaVersion: typeof raw.artifact_schema_version === 'string' ? raw.artifact_schema_version : '',
      experimental: raw.experimental === true,
    }
  }).filter(target => target.id.length > 0)
}

function parseChecks(value: unknown): DeployCheck[] {
  if (!Array.isArray(value)) return []
  return value.filter(isRecord).flatMap(raw => {
    const status = raw.status
    if (status !== 'passed' && status !== 'skipped' && status !== 'failed') return []
    return [{
      id: typeof raw.id === 'string' ? raw.id : 'check',
      status,
      detail: typeof raw.detail === 'string' ? raw.detail : '',
    }]
  })
}

function parseArtifact(payload: unknown): DeployArtifact {
  if (!isRecord(payload)) {
    throw new DeployError('DEPLOY_INVALID_RESPONSE', 'The deployment service returned an empty artifact.')
  }
  const engine = parseEngineSection(payload.engine)
  if (engine == null) {
    throw new DeployError('DEPLOY_INVALID_RESPONSE', 'The deployment service returned an artifact without engine limits.')
  }
  const files = Array.isArray(payload.files)
    ? payload.files.filter(isRecord).flatMap(raw => {
        if (typeof raw.path !== 'string' || typeof raw.contents !== 'string') return []
        return [{
          path: raw.path,
          contents: raw.contents,
          mode: typeof raw.mode === 'number' ? raw.mode : 0o644,
        }]
      })
    : []
  if (files.length === 0) {
    throw new DeployError('DEPLOY_INVALID_RESPONSE', 'The deployment service returned no files.')
  }
  const pins = isRecord(payload.pins)
    ? Object.fromEntries(Object.entries(payload.pins).filter((entry): entry is [string, string] => typeof entry[1] === 'string'))
    : {}
  return {
    target: typeof payload.target === 'string' ? payload.target : '',
    artifactSchemaVersion: typeof payload.artifact_schema_version === 'string' ? payload.artifact_schema_version : '',
    engine,
    pins,
    checks: parseChecks(payload.checks),
    warnings: Array.isArray(payload.warnings)
      ? payload.warnings.filter((warning): warning is string => typeof warning === 'string')
      : [],
    files,
  }
}

/** The request body the service expects, with only the chosen values sent. */
export function buildDeployRequestBody(input: DeployRequestInput): Record<string, unknown> {
  const body: Record<string, unknown> = {
    target: input.target,
    engine: input.engine,
    system: input.system,
    isl: input.isl,
    osl: input.osl,
  }
  if (input.concurrency != null && Number.isInteger(input.concurrency) && input.concurrency > 0) {
    body.concurrency = input.concurrency
  }
  if (input.namespace != null && input.namespace.trim().length > 0) body.namespace = input.namespace.trim()
  if (input.namePrefix != null && input.namePrefix.trim().length > 0) body.name_prefix = input.namePrefix.trim()
  if (input.image != null && input.image.trim().length > 0) body.image = input.image.trim()
  if (input.allowEngineVersionMismatch) body.allow_engine_version_mismatch = true
  if (input.servedModelName != null && input.servedModelName.trim().length > 0) {
    body.served_model_name = input.servedModelName.trim()
  }
  if (input.modelConfigData != null) body.model_config_data = input.modelConfigData
  return body
}

/** List the targets this deployment can render, with their pins. */
export async function fetchDeployTargets(signal?: AbortSignal): Promise<DeployTargetInfo[]> {
  let res: Response
  try {
    res = await fetch('/api/deploy/targets', {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal,
    })
  } catch (err: unknown) {
    if (signal?.aborted) throw err
    throw new DeployError('DEPLOY_UNAVAILABLE', 'The deployment service is unreachable.')
  }
  if (!res.ok) {
    throw await readError(res, 'The deployment target catalog could not be read.')
  }
  return parseTargets(await res.json())
}

/**
 * Render an artifact and return its file list plus the render-gate checks.
 *
 * The service verifies the bundle before answering, so a successful response is
 * the proof the artifact installs; a refusal arrives here as a `DeployError`.
 */
export async function renderDeployArtifact(
  input: DeployRequestInput,
  signal?: AbortSignal,
): Promise<DeployArtifact> {
  let res: Response
  try {
    res = await fetch('/api/deploy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(buildDeployRequestBody(input)),
      signal,
    })
  } catch (err: unknown) {
    if (signal?.aborted) throw err
    throw new DeployError('DEPLOY_UNAVAILABLE', 'The deployment service is unreachable.')
  }
  if (!res.ok) {
    throw await readError(res, 'The deployment artifact could not be rendered.')
  }
  return parseArtifact(await res.json())
}

export interface DeployDownload {
  blob: Blob
  fileName: string
}

function fileNameFromDisposition(header: string | null, fallback: string): string {
  if (!header) return fallback
  const match = /filename="?([^";]+)"?/i.exec(header)
  return match?.[1]?.trim() || fallback
}

/**
 * Render the same artifact as a zip archive and hand back its bytes.
 *
 * The JSON form is what the dialog displays; the archive form is what the user
 * downloads, so the service's zip writer preserves the scripts' executable bit
 * instead of this client reconstructing it.
 */
export async function downloadDeployArtifact(
  input: DeployRequestInput,
  signal?: AbortSignal,
): Promise<DeployDownload> {
  let res: Response
  try {
    res = await fetch('/api/deploy?format=zip', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/zip' },
      body: JSON.stringify(buildDeployRequestBody(input)),
      signal,
    })
  } catch (err: unknown) {
    if (signal?.aborted) throw err
    throw new DeployError('DEPLOY_UNAVAILABLE', 'The deployment service is unreachable.')
  }
  if (!res.ok) {
    throw await readError(res, 'The deployment archive could not be rendered.')
  }
  const fallback = `configiq-${input.target}-${input.engine.mode}.zip`
  return {
    blob: await res.blob(),
    fileName: fileNameFromDisposition(res.headers.get('content-disposition'), fallback),
  }
}

/** Hand a downloaded bundle to the browser without leaving the page. */
export function saveDeployDownload({ blob, fileName }: DeployDownload): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = fileName
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}
