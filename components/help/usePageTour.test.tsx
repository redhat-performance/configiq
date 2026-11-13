// @vitest-environment happy-dom
import * as React from 'react';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { usePageTour } from './usePageTour';
import { PageTourContext, type PageTourContextValue } from './PageTourProvider';
import { HELP_TOURS } from '@/lib/help/tours';

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

function Probe({ onValue }: { onValue: (value: PageTourContextValue) => void }) {
  const value = usePageTour();
  onValue(value);
  return <span>{value.tour ? 'tour' : 'no-tour'}</span>;
}

describe('usePageTour', () => {
  it('falls back to a no-op tour outside a provider instead of throwing', async () => {
    let seen: PageTourContextValue | null = null;

    await act(async () => {
      root.render(<Probe onValue={(value) => { seen = value; }} />);
    });

    expect(container.textContent).toBe('no-tour');
    expect(seen).not.toBeNull();
    expect(seen!.tour).toBeUndefined();
    expect(seen!.showTour).toBe(false);
    // A page rendered outside the shell can still call the handlers.
    expect(() => seen!.startTour()).not.toThrow();
    expect(() => seen!.completeTour()).not.toThrow();
  });

  it('returns the provider value when one is mounted', async () => {
    const tour = HELP_TOURS[0];
    const value: PageTourContextValue = {
      tour,
      showTour: true,
      startTour: () => {},
      completeTour: () => {},
    };
    let seen: PageTourContextValue | null = null;

    await act(async () => {
      root.render(
        <PageTourContext.Provider value={value}>
          <Probe onValue={(next) => { seen = next; }} />
        </PageTourContext.Provider>,
      );
    });

    expect(seen).toBe(value);
    expect(container.textContent).toBe('tour');
  });
});
