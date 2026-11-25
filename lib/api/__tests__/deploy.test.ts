// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { EngineSection } from '../engine'
import {
  buildDeployRequestBody,
  DeployError,
  downloadDeployArtifact,
  fetchDeployTargets,
  renderDeployArtifact,
  saveDeployDownload,
  type DeployRequestInput,
} from '../deploy'

function engineSection(overrides: Partial<EngineSection> = {}): EngineSection {
  return {
    model: 'Qwen/Qwen3-32B',
    backend: 'vllm',
    backend_version: '0.31.0',
    mode: 'agg',
    agg: {
      replicas: { value: 1, source: 'evaluated' },
      tensor_parallel_size: { value: 2, source: 'evaluated' },
      pipeline_parallel_size: { value: 1, source: 'evaluated' },
      data_parallel_size: { value: 1, source: 'evaluated' },
      context_parallel_size: { value: 1, source: 'evaluated' },
      decode_context_parallel_size: { value: 1, source: 'evaluated' },
      moe_tensor_parallel_size: { value: 2, source: 'evaluated' },
      moe_expert_parallel_size: { value: 1, source: 'evaluated' },
      max_model_len: { value: 5000, source: 'evaluated' },
      max_num_seqs: { value: 128, source: 'evaluated' },
      max_num_batched_tokens: { value: 8192, source: 'evaluated' },
      block_size: { value: 64, source: 'evaluated' },
      gpu_memory_utilization: { value: 0.9, source: 'evaluated' },
      enable_prefix_caching: { value: true, source: 'evaluated' },
      enable_chunked_prefill: { value: null, source: 'not-modeled' },
      kv_cache_dtype: { value: null, source: 'not-modeled' },
      quantization: { value: null, source: 'not-modeled' },
    },
    prefill: null,
    decode: null,
    ...overrides,
  }
}

function input(overrides: Partial<DeployRequestInput> = {}): DeployRequestInput {
  return {
    target: 'llm-d',
    engine: engineSection(),
    system: 'h200',
    isl: 2048,
    osl: 128,
    ...overrides,
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

const ARTIFACT_BODY = {
  target: 'llm-d',
  artifact_schema_version: '1',
  engine: engineSection(),
  pins: { engine: 'vllm 0.31.0', 'llm-d': 'v1.0.0' },
  checks: [
    { id: 'kustomize-build', status: 'passed', detail: 'rendered 4 resources' },
    { id: 'images', status: 'skipped', detail: 'no image assertion' },
  ],
  warnings: ['engine release differs from the target pin'],
  files: [
    { path: 'install.sh', contents: '#!/usr/bin/env bash\n', mode: 0o755 },
    { path: 'kustomization.yaml', contents: 'resources: []\n', mode: 0o644 },
  ],
}

const TARGETS_BODY = {
  targets: [
    {
      id: 'llm-d',
      name: 'llm-d well-lit path',
      description: 'A pinned model server.',
      kind: 'llm-d',
      simulator: 'aisim',
      modes: ['agg', 'disagg'],
      backends: ['vllm'],
      engine_versions: ['0.31.0'],
      pins: { 'llm-d': 'v1.0.0' },
      available: true,
      unavailable_reason: null,
      artifact_schema_version: '1',
      experimental: true,
    },
    {
      id: 'dynamo-dgd',
      name: 'Dynamo DynamoGraphDeployment',
      description: 'One DGD.',
      kind: 'dynamo-dgd',
      simulator: 'aisim',
      modes: ['agg'],
      backends: ['vllm'],
      engine_versions: ['0.24.0'],
      pins: {},
      available: false,
      unavailable_reason: 'the pinned source tree is not installed',
      artifact_schema_version: '1',
      experimental: true,
    },
  ],
}

beforeEach(() => {
  vi.unstubAllGlobals()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('buildDeployRequestBody', () => {
  it('sends only what the caller chose', () => {
    expect(buildDeployRequestBody(input())).toEqual({
      target: 'llm-d',
      engine: engineSection(),
      system: 'h200',
      isl: 2048,
      osl: 128,
    })
  })

  it('trims the optional text fields and drops blank ones', () => {
    const body = buildDeployRequestBody(input({
      namespace: '  team-a  ',
      namePrefix: '   ',
      image: ' registry.example/vllm:0.31.0 ',
      servedModelName: '',
    }))

    expect(body.namespace).toBe('team-a')
    expect(body.image).toBe('registry.example/vllm:0.31.0')
    expect(body.name_prefix).toBeUndefined()
    expect(body.served_model_name).toBeUndefined()
  })

  it('omits a mismatch allowance that was not granted', () => {
    const allowed = buildDeployRequestBody(input({ allowEngineVersionMismatch: true }))
    const refused = buildDeployRequestBody(input({ allowEngineVersionMismatch: false }))

    expect(allowed.allow_engine_version_mismatch).toBe(true)
    expect(refused.allow_engine_version_mismatch).toBeUndefined()
  })

  it('sends a concurrency only when it is a positive integer', () => {
    expect(buildDeployRequestBody(input({ concurrency: 64 })).concurrency).toBe(64)
    expect(buildDeployRequestBody(input({ concurrency: 0 })).concurrency).toBeUndefined()
    expect(buildDeployRequestBody(input({ concurrency: -1 })).concurrency).toBeUndefined()
    expect(buildDeployRequestBody(input({ concurrency: 1.5 })).concurrency).toBeUndefined()
    expect(buildDeployRequestBody(input({ concurrency: null })).concurrency).toBeUndefined()
  })

  it('forwards the model config only when the evaluation supplied one', () => {
    const modelConfigData = { hidden_size: 5120, num_hidden_layers: 64 }

    expect(buildDeployRequestBody(input({ modelConfigData })).model_config_data).toEqual(modelConfigData)
    expect(buildDeployRequestBody(input({ modelConfigData: null })).model_config_data).toBeUndefined()
  })
})

describe('fetchDeployTargets', () => {
  it('parses the catalog, keeping the availability reason per target', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(TARGETS_BODY))
    vi.stubGlobal('fetch', fetchMock)

    const targets = await fetchDeployTargets()

    expect(fetchMock.mock.calls[0][0]).toBe('/api/deploy/targets')
    expect(targets).toHaveLength(2)
    expect(targets[0]).toEqual({
      id: 'llm-d',
      name: 'llm-d well-lit path',
      description: 'A pinned model server.',
      kind: 'llm-d',
      simulator: 'aisim',
      modes: ['agg', 'disagg'],
      backends: ['vllm'],
      engineVersions: ['0.31.0'],
      pins: { 'llm-d': 'v1.0.0' },
      available: true,
      unavailableReason: null,
      artifactSchemaVersion: '1',
      experimental: true,
    })
    expect(targets[1].available).toBe(false)
    expect(targets[1].unavailableReason).toBe('the pinned source tree is not installed')
  })

  it('drops an entry without an id rather than rendering a blank option', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      targets: [{ name: 'nameless' }, { id: 'llm-d', name: 'llm-d' }],
    })))

    const targets = await fetchDeployTargets()

    expect(targets.map(target => target.id)).toEqual(['llm-d'])
  })

  it('rejects a body that is not a target list', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ targets: 'nope' })))

    await expect(fetchDeployTargets()).rejects.toMatchObject({
      name: 'DeployError',
      code: 'DEPLOY_INVALID_RESPONSE',
    })
  })

  it('surfaces the service error code and message', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      detail: 'the pinned source tree is not installed',
    }, 503)))

    await expect(fetchDeployTargets()).rejects.toMatchObject({
      code: 'DEPLOY_FAILED',
      message: 'the pinned source tree is not installed',
    })
  })

  it('reports an unreachable service without leaking the fetch error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('fetch failed')))

    await expect(fetchDeployTargets()).rejects.toMatchObject({
      code: 'DEPLOY_UNAVAILABLE',
    })
  })
})

describe('renderDeployArtifact', () => {
  it('posts the request and returns the files, checks, and pins', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(ARTIFACT_BODY))
    vi.stubGlobal('fetch', fetchMock)

    const artifact = await renderDeployArtifact(input())

    expect(fetchMock.mock.calls[0][0]).toBe('/api/deploy')
    const init = fetchMock.mock.calls[0][1] as RequestInit
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body as string)).toEqual(buildDeployRequestBody(input()))

    expect(artifact.target).toBe('llm-d')
    expect(artifact.artifactSchemaVersion).toBe('1')
    expect(artifact.engine.mode).toBe('agg')
    expect(artifact.pins).toEqual({ engine: 'vllm 0.31.0', 'llm-d': 'v1.0.0' })
    expect(artifact.checks).toEqual([
      { id: 'kustomize-build', status: 'passed', detail: 'rendered 4 resources' },
      { id: 'images', status: 'skipped', detail: 'no image assertion' },
    ])
    expect(artifact.warnings).toEqual(['engine release differs from the target pin'])
    expect(artifact.files.map(file => file.path)).toEqual(['install.sh', 'kustomization.yaml'])
    expect(artifact.files[0].mode).toBe(0o755)
  })

  it('defaults a missing file mode to a non-executable one', async () => {
    const body = {
      ...ARTIFACT_BODY,
      files: [{ path: 'kustomization.yaml', contents: 'resources: []\n' }],
    }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(body)))

    const artifact = await renderDeployArtifact(input())

    expect(artifact.files[0].mode).toBe(0o644)
  })

  it('refuses an artifact with no engine limits', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ ...ARTIFACT_BODY, engine: null })))

    await expect(renderDeployArtifact(input())).rejects.toMatchObject({
      code: 'DEPLOY_INVALID_RESPONSE',
    })
  })

  it('refuses an artifact with no files', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ ...ARTIFACT_BODY, files: [] })))

    await expect(renderDeployArtifact(input())).rejects.toMatchObject({
      code: 'DEPLOY_INVALID_RESPONSE',
    })
  })

  it('surfaces a refusal as the service stated it', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      detail: 'the evaluated limits cannot be reproduced by this target',
    }, 422)))

    const error = await renderDeployArtifact(input()).catch((err: unknown) => err)

    expect(error).toBeInstanceOf(DeployError)
    expect(error).toMatchObject({
      code: 'DEPLOY_FAILED',
      message: 'the evaluated limits cannot be reproduced by this target',
    })
  })

  it('surfaces a structured error body', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      error: { code: 'TARGET_UNAVAILABLE', message: 'kustomize is not installed' },
    }, 503)))

    await expect(renderDeployArtifact(input())).rejects.toMatchObject({
      code: 'TARGET_UNAVAILABLE',
      message: 'kustomize is not installed',
    })
  })

  it('does not fall back to a message when the request is aborted', async () => {
    const abort = new DOMException('aborted', 'AbortError')
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(abort))
    const controller = new AbortController()
    controller.abort()

    await expect(renderDeployArtifact(input(), controller.signal)).rejects.toBe(abort)
  })
})

describe('downloadDeployArtifact', () => {
  it('asks the same endpoint for an archive and keeps the served filename', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('PK\u0003\u0004', {
      status: 200,
      headers: {
        'Content-Type': 'application/zip',
        'Content-Disposition': 'attachment; filename="configiq-llm-d-agg.zip"',
      },
    }))
    vi.stubGlobal('fetch', fetchMock)

    const download = await downloadDeployArtifact(input())

    expect(fetchMock.mock.calls[0][0]).toBe('/api/deploy?format=zip')
    expect(download.fileName).toBe('configiq-llm-d-agg.zip')
    expect(download.blob.size).toBeGreaterThan(0)
  })

  it('falls back to a deterministic filename when none is served', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('PK', {
      status: 200,
      headers: { 'Content-Type': 'application/zip' },
    })))

    const download = await downloadDeployArtifact(input())

    expect(download.fileName).toBe('configiq-llm-d-agg.zip')
  })

  it('surfaces a refusal instead of saving an error body as an archive', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      detail: 'the deployment service did not return an archive',
    }, 502)))

    await expect(downloadDeployArtifact(input())).rejects.toMatchObject({
      code: 'DEPLOY_FAILED',
      message: 'the deployment service did not return an archive',
    })
  })
})

describe('saveDeployDownload', () => {
  it('hands the archive to the browser and releases the object URL', () => {
    const createObjectURL = vi.fn(() => 'blob:artifact')
    const revokeObjectURL = vi.fn()
    vi.stubGlobal('URL', { ...URL, createObjectURL, revokeObjectURL })
    const clicks: string[] = []
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      clicks.push(this.download)
    })

    saveDeployDownload({ blob: new Blob(['PK']), fileName: 'configiq-llm-d-agg.zip' })

    expect(createObjectURL).toHaveBeenCalledTimes(1)
    expect(clicks).toEqual(['configiq-llm-d-agg.zip'])
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:artifact')
    click.mockRestore()
  })
})
