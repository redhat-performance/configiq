// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/app-config', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/app-config')>();
  return {
    ...actual,
    getAppConfig: () => ({
      ...actual.getAppConfig(),
      suggestedModelNames: ['Nemotron'],
      modelRequestUrl: '',
    }),
  };
});

import { ComboBox, type ComboBoxItem } from './ModelComboBox';
import { modelFacets } from '@/lib/model-metadata';

const IDS = [
  'Qwen/Qwen3-32B',
  'Qwen/Qwen3-32B-FP8',
  'Qwen/Qwen3-8B',
  'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8',
];

function items(): ComboBoxItem[] {
  return IDS.map((id) => ({
    ...modelFacets(id, { testedModels: ['Qwen/Qwen3-8B'] }),
    value: id,
    label: id,
    group: id.slice(0, id.indexOf('/')),
  }));
}

let container: HTMLDivElement;
let root: ReturnType<typeof createRoot>;

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
    .IS_REACT_ACT_ENVIRONMENT = true;
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

async function mount(props: Partial<React.ComponentProps<typeof ComboBox>> = {}) {
  const onChange = vi.fn();
  await act(async () => {
    root.render(
      <ComboBox id="mb" value="" onChange={onChange} items={items()} {...props} />,
    );
    await Promise.resolve();
  });
  return { onChange };
}

/**
 * PatternFly's TextInputGroupMain puts the `id` prop on its wrapper div and
 * forwards only role/aria-* to the inner input, so the field has to be found
 * by role rather than by id.
 */
function input(): HTMLInputElement {
  const el = container.querySelector<HTMLInputElement>('input[role="combobox"]');
  if (!el) throw new Error('combobox input not rendered');
  return el;
}

/**
 * Model ids of the rendered rows. Reads the label span specifically — the
 * option also contains the metadata chips, so its full textContent would come
 * back as "Qwen/Qwen3-32B32B". CSS module class names are hashed, hence the
 * substring match.
 */
function optionLabels(): string[] {
  return [...document.querySelectorAll('[id^="mb-opt-"]')].map(
    (el) => el.querySelector('[class*="optionLabel"]')?.textContent ?? '',
  );
}

/** React delegates focus through `focusin`; a bare `focus` event never reaches it. */
async function open() {
  await act(async () => {
    input().dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    await Promise.resolve();
  });
}

async function type(text: string) {
  await act(async () => {
    const el = input();
    const setter = Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      'value',
    )?.set;
    setter?.call(el, text);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    await Promise.resolve();
  });
}

async function key(k: string) {
  await act(async () => {
    input().dispatchEvent(
      new KeyboardEvent('keydown', { key: k, bubbles: true }),
    );
    await Promise.resolve();
  });
}

describe('browsing', () => {
  it('lists every model, quantized variants included', async () => {
    await mount();
    await open();
    expect(optionLabels()).toEqual(expect.arrayContaining(IDS));
    expect(optionLabels()).toHaveLength(IDS.length);
  });

  it('groups rows by vendor, ordered case-insensitively', async () => {
    await mount();
    await open();
    const groups = [...document.querySelectorAll('.pf-v6-c-menu__group-title')]
      .map((el) => el.textContent?.trim());
    expect(groups).toEqual(['nvidia', 'Qwen']);
  });
});

describe('searching', () => {
  it('shows every matching variant', async () => {
    await mount();
    await open();
    await type('qwen3-32b');
    const labels = optionLabels();
    expect(labels).toContain('Qwen/Qwen3-32B');
    expect(labels).toContain('Qwen/Qwen3-32B-FP8');
  });

  it('drops vendor grouping in favour of one ranked list', async () => {
    await mount();
    await open();
    await type('qwen');
    expect(document.querySelectorAll('.pf-v6-c-menu__group-title')).toHaveLength(0);
  });

  it('finds models by size rather than substring', async () => {
    await mount();
    await open();
    await type('120b');
    expect(optionLabels()).toEqual(['nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8']);
  });

  it('renders a No matches row when nothing scores', async () => {
    await mount();
    await open();
    await type('zzzznothing');
    expect(optionLabels()).toHaveLength(0);
    expect(document.body.textContent).toContain('No matches');
  });
});

describe('keyboard navigation', () => {
  it('selects the row the arrow keys land on', async () => {
    const { onChange } = await mount();
    await open();
    await key('ArrowDown');
    const first = optionLabels()[0];
    await key('Enter');
    expect(onChange).toHaveBeenCalledWith(first);
  });

  it('keeps focus index aligned with the rendered rows after filtering', async () => {
    const { onChange } = await mount();
    await open();
    await type('120b');
    await key('ArrowDown');
    await key('Enter');
    expect(onChange).toHaveBeenCalledWith('nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8');
  });

  it('closes on Escape without selecting', async () => {
    const { onChange } = await mount();
    await open();
    await key('ArrowDown');
    await key('Escape');
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe('tested-only filter', () => {
  it('replaces a non-tested selection when switched on', async () => {
    const { onChange } = await mount({
      value: 'Qwen/Qwen3-32B',
      supportedModels: ['Qwen/Qwen3-8B'],
    });
    await act(async () => {
      container.querySelector<HTMLInputElement>('#mb-validated-only')?.click();
      await Promise.resolve();
    });
    expect(onChange).toHaveBeenCalledWith('Qwen/Qwen3-8B');
  });

  it('leaves an already-tested selection alone', async () => {
    const { onChange } = await mount({
      value: 'Qwen/Qwen3-8B',
      supportedModels: ['Qwen/Qwen3-8B'],
    });
    await act(async () => {
      container.querySelector<HTMLInputElement>('#mb-validated-only')?.click();
      await Promise.resolve();
    });
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe('custom model entry', () => {
  it('offers an unknown id when allowCustom is set', async () => {
    await mount({ allowCustom: true });
    await open();
    await type('my-org/My-Model');
    expect(document.body.textContent).toContain('my-org/My-Model');
  });

  it('selects the typed id on Enter', async () => {
    const { onChange } = await mount({ allowCustom: true });
    await open();
    await type('my-org/My-Model');
    await key('Enter');
    expect(onChange).toHaveBeenCalledWith('my-org/My-Model');
  });
});
