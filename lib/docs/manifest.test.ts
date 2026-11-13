import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { DOCS_SECTIONS } from './manifest';

const ROOT = process.cwd();

interface PackageJson {
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
  scripts?: Record<string, string>;
}

const pkg = JSON.parse(readFileSync(path.join(ROOT, 'package.json'), 'utf8')) as PackageJson;

describe('docs manifest', () => {
  it('lists sections in order with unique slugs', () => {
    expect(DOCS_SECTIONS.length).toBeGreaterThan(0);

    const slugs = DOCS_SECTIONS.map((section) => section.slug);
    expect(new Set(slugs).size).toBe(slugs.length);

    for (const section of DOCS_SECTIONS) {
      expect(section.slug).toMatch(/^[a-z0-9-]+$/);
      expect(section.title.trim().length).toBeGreaterThan(0);
    }
  });

  it('points every section at a Markdown file that exists', () => {
    for (const section of DOCS_SECTIONS) {
      expect(section.file.endsWith('.md')).toBe(true);
      expect(existsSync(path.join(ROOT, section.file)), section.file).toBe(true);
    }
  });

  it('has a one-sentence description for every index entry', () => {
    for (const section of DOCS_SECTIONS) {
      expect(section.description.trim().length, section.slug).toBeGreaterThan(0);
      expect(section.description.trim().endsWith('.'), section.slug).toBe(true);
    }
  });

  it('links every guide from the overview', () => {
    // The overview is the entry point in the rendered docs and on GitHub, so a
    // section that is missing there is only reachable by guessing its URL.
    const overview = readFileSync(path.join(ROOT, 'docs/overview.md'), 'utf8');
    for (const section of DOCS_SECTIONS) {
      if (section.slug === 'overview') continue;
      const relative = section.file.replace(/^docs\//, '');
      expect(overview, `docs/overview.md does not link to ${section.file}`).toContain(
        `(${relative})`,
      );
    }
  });

  it('builds docs before type-check, test, locally and in CI', () => {
    expect(pkg.scripts?.['docs:build']).toBeDefined();
    expect(pkg.scripts?.predev).toContain('docs:build');
    expect(pkg.scripts?.prebuild).toContain('docs:build');
    // `npm test` imports the generated index through the assistant route, so a
    // clean checkout fails without this exactly like `tsc` does.
    expect(pkg.scripts?.pretest).toContain('docs:build');

    // `lib/docs/generated/` is not checked in, so a clean CI checkout must build
    // it before `tsc` resolves the docs route's imports.
    const workflow = readFileSync(path.join(ROOT, '.github/workflows/ci.yml'), 'utf8');
    const build = workflow.indexOf('npm run docs:build');
    const typeCheck = workflow.indexOf('npm run type-check');
    expect(build).toBeGreaterThan(-1);
    expect(typeCheck).toBeGreaterThan(build);
  });

  it('has a docs:check that verifies instead of rewriting', () => {
    expect(pkg.scripts?.['docs:check']).toBeDefined();
    expect(pkg.scripts?.['docs:check']).toContain('--check');
    expect(pkg.scripts?.['docs:check']).not.toBe(pkg.scripts?.['docs:build']);
  });

  it('does not ship a Markdown parser to the browser', () => {
    // Docs are rendered to HTML at build time. A runtime markdown package here
    // means someone reintroduced client-side parsing.
    const runtime = Object.keys(pkg.dependencies ?? {});
    const forbidden = ['react-markdown', 'remark', 'remark-gfm', 'remark-parse', 'rehype', 'unified'];
    for (const name of forbidden) {
      expect(runtime).not.toContain(name);
    }
  });
});
