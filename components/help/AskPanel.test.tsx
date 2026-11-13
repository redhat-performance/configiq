// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AskPanel } from './AskPanel';

vi.mock('next/link', () => ({
  default: ({
    href,
    children,
    ...rest
  }: {
    href: string;
    children: React.ReactNode;
  } & React.AnchorHTMLAttributes<HTMLAnchorElement>) =>
    React.createElement('a', { href, ...rest }, children),
}));

vi.mock('@/contexts/SettingsContext', () => ({
  useSettings: () => ({ simulator: 'aisimulate', inferenceBackend: 'vllm', backendVersion: '0.9.0' }),
}));

let root: ReturnType<typeof createRoot>;
let host: HTMLDivElement;

function streamResponse(events: unknown[]): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of events) {
        controller.enqueue(new TextEncoder().encode(`${JSON.stringify(event)}\n`));
      }
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { 'Content-Type': 'application/x-ndjson' } });
}

async function render() {
  await act(async () => {
    root.render(<AskPanel currentRoute="/predict" model="test/model" hide={() => undefined} />);
  });
}

async function ask(question: string) {
  const input = document.querySelector<HTMLTextAreaElement>('#assistant-question');
  expect(input).not.toBeNull();

  const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set;
  await act(async () => {
    setter?.call(input, question);
    input!.dispatchEvent(new Event('input', { bubbles: true }));
  });

  const button = Array.from(document.querySelectorAll('button')).find(
    (candidate) => candidate.textContent === 'Ask',
  );
  expect(button).not.toBeUndefined();

  await act(async () => {
    button!.click();
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

describe('AskPanel', () => {
  it('shows a verified answer with its citations and provenance', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        streamResponse([
          { type: 'retrieval', sections: [{ id: 'a', title: 'Concurrency', section: 'Predict', url: '/docs/overview#c', body: 'x', score: 1 }] },
          {
            type: 'result',
            status: 'answered',
            answer: 'Set concurrency in the workload section.',
            citations: [{ url: '/docs/overview#c', title: 'Concurrency', section: 'Predict' }],
            links: [{ label: 'Open Predict performance', href: '/predict?v=1&simulator=aisimulate' }],
            evidence: ['Set concurrency in the workload section.'],
            unsupportedNumbers: [],
            model: 'test/model',
          },
        ])),
    );

    await render();
    await ask('Where do I set concurrency?');

    expect(document.body.textContent).toContain('Set concurrency in the workload section.');
    const citation = Array.from(document.querySelectorAll('a')).find(
      (anchor) => anchor.getAttribute('href') === '/docs/overview#c',
    );
    expect(citation).toBeDefined();
    expect(document.body.textContent).toContain('test/model');
  });

  it('replaces unverified prose with the sources instead of showing the numbers', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        streamResponse([
          { type: 'retrieval', sections: [] },
          {
            type: 'result',
            status: 'suppressed',
            answer: '',
            citations: [{ url: '/docs/costs#tco', title: 'Total cost', section: 'Costs' }],
            links: [],
            evidence: ['Total cost includes hardware, power, and cooling.'],
            unsupportedNumbers: ['96 GB'],
            model: 'test/model',
          },
        ])),
    );

    await render();
    await ask('How much VRAM do I need?');

    expect(document.body.textContent).not.toContain('96 GB of VRAM');
    expect(document.body.textContent).toContain('could not verify the numbers');
    expect(document.body.textContent).toContain('Unverified: 96 GB');
  });
});
