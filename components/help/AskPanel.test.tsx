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
            // The route withholds this prose; the panel must too, even though
            // the field is nonempty. Rendering `event.answer` unconditionally
            // would put an unverified number on screen.
            answer: 'You need 96 GB of VRAM for this model.',
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

  it('withholds an unfinished answer and keeps its sources', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        streamResponse([
          { type: 'retrieval', sections: [] },
          {
            type: 'result',
            status: 'incomplete',
            // The route withholds this trace; the panel must not render it.
            answer: 'The user is asking about concurrency. I should explain the fields.',
            citations: [{ url: '/docs/workloads#c', title: 'Workloads', section: 'Recommend' }],
            links: [],
            evidence: [],
            unsupportedNumbers: [],
            model: 'test/model',
          },
        ])),
    );

    await render();
    await ask('How do I set concurrency?');

    expect(document.body.textContent).not.toContain('The user is asking');
    expect(document.body.textContent).toContain('did not finish this answer');
    expect(document.body.textContent).toContain('Workloads');
  });

  it('leaves the question box usable when the stream ends without a result', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        streamResponse([{ type: 'retrieval', sections: [] }])),
    );

    await render();
    await ask('Where do I set concurrency?');

    const input = document.querySelector<HTMLTextAreaElement>('#assistant-question');
    expect(input?.disabled).toBe(false);
    expect(document.body.textContent).toContain('did not finish this answer');
  });
});
