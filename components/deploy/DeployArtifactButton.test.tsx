// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/api/deploy', async importOriginal => {
  const actual = await importOriginal<typeof import('@/lib/api/deploy')>();
  return {
    ...actual,
    fetchDeployTargets: vi.fn(),
    renderDeployArtifact: vi.fn(),
    downloadDeployArtifact: vi.fn(),
    saveDeployDownload: vi.fn(),
  };
});

import {
  DeployError,
  downloadDeployArtifact,
  fetchDeployTargets,
  renderDeployArtifact,
  saveDeployDownload,
  type DeployArtifact,
  type DeployTargetInfo,
} from '@/lib/api/deploy';
import type { EngineSection } from '@/lib/api/engine';
import { DeployArtifactButton } from './DeployArtifactButton';

let root: ReturnType<typeof createRoot>;
let host: HTMLDivElement;

function engine(): EngineSection {
  const evaluated = (value: boolean | number | string) => ({ value, source: 'evaluated' as const });
  return {
    model: 'meta-llama/Llama-3.1-70B',
    backend: 'vllm',
    backend_version: '0.31.0',
    mode: 'agg',
    agg: {
      replicas: evaluated(1),
      tensor_parallel_size: evaluated(8),
      pipeline_parallel_size: evaluated(1),
      data_parallel_size: evaluated(1),
      context_parallel_size: evaluated(1),
      decode_context_parallel_size: evaluated(1),
      moe_tensor_parallel_size: evaluated(8),
      moe_expert_parallel_size: evaluated(1),
      max_model_len: evaluated(4096),
      max_num_seqs: evaluated(32),
      max_num_batched_tokens: evaluated(8192),
      block_size: evaluated(64),
      gpu_memory_utilization: evaluated(0.9),
      enable_prefix_caching: evaluated(true),
      enable_chunked_prefill: evaluated(true),
      kv_cache_dtype: evaluated('fp8'),
      quantization: evaluated('fp8'),
    },
    prefill: null,
    decode: null,
  };
}

function target(overrides: Partial<DeployTargetInfo> = {}): DeployTargetInfo {
  return {
    id: 'llm-d',
    name: 'llm-d well-lit path',
    description: 'An llm-d v1.0.0 model server plus the pinned router release.',
    kind: 'llm-d',
    simulator: 'aisim',
    modes: ['agg', 'disagg'],
    backends: ['vllm'],
    engineVersions: ['0.31.0'],
    pins: { engine: 'vllm 0.31.0', 'llm-d': 'v1.0.0' },
    available: true,
    unavailableReason: null,
    artifactSchemaVersion: '1',
    experimental: true,
    ...overrides,
  };
}

function artifact(overrides: Partial<DeployArtifact> = {}): DeployArtifact {
  return {
    target: 'llm-d',
    artifactSchemaVersion: '1',
    engine: engine(),
    pins: { engine: 'vllm 0.31.0', 'llm-d': 'v1.0.0' },
    checks: [
      { id: 'kustomize-build', status: 'passed', detail: 'rendered 4 resources' },
      { id: 'engine-limits', status: 'passed', detail: 'TP 8, max seqs 32' },
    ],
    warnings: [],
    files: [
      { path: 'install.sh', contents: '#!/usr/bin/env bash\n', mode: 0o755 },
      { path: 'kustomization.yaml', contents: 'resources: []\n', mode: 0o644 },
    ],
    ...overrides,
  };
}

function buttons(): HTMLButtonElement[] {
  return Array.from(document.body.querySelectorAll('button'))
    .filter(candidate => candidate.closest('[aria-hidden="true"]') == null);
}

function buttonNamed(text: string): HTMLButtonElement {
  const found = buttons().find(candidate => candidate.textContent?.includes(text));
  if (!found) throw new Error(`expected a button containing "${text}"`);
  return found;
}

async function renderButton(props: Partial<React.ComponentProps<typeof DeployArtifactButton>> = {}) {
  await act(async () => {
    root.render(
      <DeployArtifactButton
        engine={engine()}
        system="h200"
        isl={2048}
        osl={128}
        concurrency={32}
        disabledReason={null}
        {...props}
      />,
    );
  });
}

async function openDialog() {
  await act(async () => buttonNamed('Deploy artifact').click());
  // Resolve the target fetch and let the dialog paint.
  await act(async () => {});
  await act(async () => {});
}

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  vi.mocked(fetchDeployTargets).mockReset();
  vi.mocked(renderDeployArtifact).mockReset();
  vi.mocked(downloadDeployArtifact).mockReset();
  vi.mocked(saveDeployDownload).mockReset();
  vi.mocked(fetchDeployTargets).mockResolvedValue([target()]);
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
});

describe('DeployArtifactButton', () => {
  it('opens the dialog and renders the artifact from the evaluated limits', async () => {
    vi.mocked(renderDeployArtifact).mockResolvedValue(artifact());
    await renderButton();
    await openDialog();

    const renderButtonEl = await act(async () => buttonNamed('Render artifact'));
    await act(async () => renderButtonEl.click());

    expect(fetchDeployTargets).toHaveBeenCalledTimes(1);
    expect(renderDeployArtifact).toHaveBeenCalledWith(
      expect.objectContaining({
        target: 'llm-d',
        engine: engine(),
        system: 'h200',
        isl: 2048,
        osl: 128,
        concurrency: 32,
      }),
      expect.any(AbortSignal),
    );
    expect(document.body.textContent).toContain('Render gate passed for llm-d');
    expect(document.body.textContent).toContain('kustomize-build');
    expect(document.body.textContent).toContain('2 files');
  });

  it('downloads the archive only after the user asks for it', async () => {
    vi.mocked(renderDeployArtifact).mockResolvedValue(artifact());
    vi.mocked(downloadDeployArtifact).mockResolvedValue({
      blob: new Blob(['PK']),
      fileName: 'configiq-llm-d-agg.zip',
    });
    await renderButton();
    await openDialog();

    await act(async () => buttonNamed('Render artifact').click());

    expect(downloadDeployArtifact).not.toHaveBeenCalled();
    await act(async () => buttonNamed('Download .zip').click());

    expect(downloadDeployArtifact).toHaveBeenCalledTimes(1);
    expect(saveDeployDownload).toHaveBeenCalledWith({
      blob: expect.any(Blob),
      fileName: 'configiq-llm-d-agg.zip',
    });
  });

  it('shows a refusal with the reason the service gave', async () => {
    vi.mocked(renderDeployArtifact).mockRejectedValue(
      new DeployError('DEPLOY_TARGET_MISMATCH', 'the evaluated limits cannot be reproduced'),
    );
    await renderButton();
    await openDialog();

    await act(async () => buttonNamed('Render artifact').click());

    expect(document.body.textContent).toContain('The artifact was refused');
    expect(document.body.textContent).toContain('the evaluated limits cannot be reproduced');
  });

  it('disables the action while a calculation is running, naming the reason', async () => {
    await renderButton({ disabledReason: 'A calculation is running; render when it finishes.' });

    const action = buttonNamed('Deploy artifact');
    expect(action.disabled).toBe(true);
    expect(action.getAttribute('aria-label')).toContain('render when it finishes');
  });

  it('names the missing engine limits when nothing else blocks the action', async () => {
    await renderButton({ engine: null });

    expect(buttonNamed('Deploy artifact').getAttribute('aria-label'))
      .toContain('did not report the engine limits');
  });

  it('prefers the stale-input reason when that is why the action is blocked', async () => {
    await renderButton({ engine: null, disabledReason: 'The inputs changed after this result.' });

    expect(buttonNamed('Deploy artifact').getAttribute('aria-label'))
      .toContain('The inputs changed after this result.');
  });

  it('reports a target list it could not read', async () => {
    vi.mocked(fetchDeployTargets).mockRejectedValue(new DeployError('DEPLOY_UNAVAILABLE', 'unreachable'));
    await renderButton();
    await openDialog();

    expect(document.body.textContent).toContain('Deployment targets are unavailable');
    expect(document.body.textContent).toContain('unreachable');
    expect(buttonNamed('Render artifact').disabled).toBe(true);
  });

  it('names an unavailable target instead of hiding it', async () => {
    const unavailable = target({
      id: 'dynamo-dgd',
      name: 'Dynamo DynamoGraphDeployment',
      available: false,
      unavailableReason: 'the pinned source tree is not installed',
    });
    vi.mocked(fetchDeployTargets).mockResolvedValue([target(), unavailable]);
    await renderButton();
    await openDialog();

    expect(document.body.textContent).toContain('Some targets are unavailable here');
    expect(document.body.textContent).toContain('the pinned source tree is not installed');
  });

  it('aborts and clears a render when the dialog is pointed at a different result', async () => {
    let resolveRender: (value: DeployArtifact) => void = () => {};
    vi.mocked(renderDeployArtifact).mockReturnValue(
      new Promise<DeployArtifact>(resolve => {
        resolveRender = resolve;
      }),
    );
    await renderButton();
    await openDialog();

    await act(async () => buttonNamed('Render artifact').click());
    expect(buttonNamed('Render artifact').disabled).toBe(true);

    const nextEngine = engine();
    if (nextEngine.agg == null) throw new Error('expected aggregated limits');
    nextEngine.agg.max_num_seqs = { value: 8, source: 'evaluated' };
    await renderButton({ engine: nextEngine });

    // The late answer belongs to the previous result and must not restore it.
    await act(async () => {
      resolveRender(artifact());
    });
    await act(async () => {});

    expect(document.body.textContent).not.toContain('Render gate passed for llm-d');
    expect(buttonNamed('Render artifact').disabled).toBe(false);
  });
});
