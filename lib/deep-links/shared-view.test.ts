import { describe, expect, it, vi } from 'vitest';
import { z } from 'zod';
import { defineLink } from './common';
import { applyLinkValues, collectLinkValues, defineAccessors } from './shared-view';

const link = defineLink('/probe', {
  alpha: z.coerce.number(),
  beta: z.string(),
  // A field the page reads from settings rather than from this table.
  gamma: z.string(),
});

function accessors(readAlpha: () => number | undefined, writeAlpha: (value: number) => void) {
  return defineAccessors(link, {
    alpha: { read: readAlpha, write: writeAlpha },
    beta: { read: () => 'from-page', write: () => undefined },
    gamma: { read: () => 'from-page', write: () => undefined },
  });
}

describe('collectLinkValues', () => {
  it('reads every bound field and drops the ones this view has no value for', () => {
    const table = defineAccessors(link, {
      alpha: { read: () => undefined, write: () => undefined },
      beta: { read: () => 'b', write: () => undefined },
      gamma: { read: () => 'g', write: () => undefined },
    });

    expect(collectLinkValues(table)).toEqual({ beta: 'b', gamma: 'g' });
  });

  it('adds values the page always sends but does not own as state', () => {
    const table = accessors(() => 1, () => undefined);

    expect(collectLinkValues(table, { gamma: 'shared' })).toEqual({
      alpha: 1,
      beta: 'from-page',
      gamma: 'shared',
    });
  });
});

describe('applyLinkValues', () => {
  it('writes only the fields the link carried', () => {
    const writeAlpha = vi.fn();
    const writeBeta = vi.fn();
    const table = defineAccessors(link, {
      alpha: { read: () => undefined, write: writeAlpha },
      beta: { read: () => undefined, write: writeBeta },
      gamma: { read: () => undefined, write: () => undefined },
    });

    applyLinkValues(link.parse('?v=1&alpha=7').values, table);

    expect(writeAlpha).toHaveBeenCalledWith(7, { alpha: 7 });
    expect(writeBeta).not.toHaveBeenCalled();
  });

  it('ignores a field the page does not bind', () => {
    const table = defineAccessors(link, {
      alpha: { read: () => undefined, write: () => undefined },
      beta: { read: () => undefined, write: () => undefined },
      gamma: { read: () => undefined, write: () => undefined },
    });

    // `delta` is not in the contract, so it never reaches a write.
    expect(() => applyLinkValues({ alpha: 1, delta: 2 } as never, table)).not.toThrow();
  });
});

describe('a view built from its accessor table', () => {
  it('lands a recipient on the view the link describes', () => {
    const sender = { alpha: 5, beta: 'sent' };
    const senderTable = defineAccessors(link, {
      alpha: { read: () => sender.alpha, write: () => undefined },
      beta: { read: () => sender.beta, write: () => undefined },
      gamma: { read: () => undefined, write: () => undefined },
    });

    const url = link.build(collectLinkValues(senderTable));

    // The recipient starts somewhere else; the link replaces what it carries
    // and leaves the field it omits alone.
    const recipient = { alpha: 0, beta: 'theirs' };
    const recipientTable = defineAccessors(link, {
      alpha: { read: () => recipient.alpha, write: (value) => { recipient.alpha = value } },
      beta: { read: () => recipient.beta, write: (value) => { recipient.beta = value } },
      gamma: { read: () => undefined, write: () => undefined },
    });

    applyLinkValues(link.parse(url.slice(url.indexOf('?'))).values, recipientTable);

    expect(recipient).toEqual(sender);
  });
});
