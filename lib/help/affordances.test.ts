import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { HELP_AFFORDANCES } from './affordances';
import { DOCS_SECTIONS } from '@/lib/docs/manifest';

const ROOT = process.cwd();

/** Map an app route to the file that renders it. */
function routeSource(route: string): string {
  if (route === '/docs') return path.join(ROOT, 'app/docs/page.tsx');
  return path.join(ROOT, 'app', route.replace(/^\//, ''), 'page.tsx');
}

function generatedDoc(slug: string): { html: string } {
  const file = path.join(ROOT, 'lib/docs/generated', `${slug}.json`);
  expect(
    existsSync(file),
    `${file} is missing; run npm run docs:build before this test`,
  ).toBe(true);
  return JSON.parse(readFileSync(file, 'utf8')) as { html: string };
}

describe('help affordances', () => {
  it('has unique ids', () => {
    const ids = HELP_AFFORDANCES.map((entry) => entry.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('points every affordance at a route that exists', () => {
    for (const entry of HELP_AFFORDANCES) {
      expect(existsSync(routeSource(entry.route)), `${entry.id} -> ${entry.route}`).toBe(true);
    }
  });

  it('points every documentation reference at a real section and anchor', () => {
    const slugs = new Set(DOCS_SECTIONS.map((section) => section.slug));

    for (const entry of HELP_AFFORDANCES) {
      if (!entry.doc) continue;
      expect(slugs.has(entry.doc), `${entry.id} -> docs/${entry.doc}`).toBe(true);

      if (!entry.anchor) continue;
      const { html } = generatedDoc(entry.doc);
      expect(
        html.includes(`id="${entry.anchor}"`),
        `${entry.id} -> /docs/${entry.doc}#${entry.anchor}`,
      ).toBe(true);
    }
  });
});
