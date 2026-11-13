import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

/**
 * The API reference must describe the service that exists.
 *
 * These are the facts most likely to drift: an endpoint list, the request
 * defaults quoted in the docs, and the required fields of an endpoint. The
 * OpenAPI contract is the source of truth, so the prose is checked against it
 * rather than against another copy of the prose.
 */

const ROOT = process.cwd();
const DOCS = readFileSync(path.join(ROOT, 'docs/api.md'), 'utf8');
const OPENAPI = readFileSync(
  path.join(ROOT, 'services/aisimulators/docs/api/openapi.yaml'),
  'utf8',
);

function openApiPaths(): Set<string> {
  return new Set(
    [...OPENAPI.matchAll(/^ {2}(\/[a-z0-9/-]+):$/gm)].map((match) => match[1]),
  );
}

function documentedEndpoints(): Set<string> {
  return new Set(
    [...DOCS.matchAll(/^\| (?:GET|POST) \| `(\/[^`]+)` \|/gm)].map((match) => match[1]),
  );
}

/** Lines of one named schema under `components.schemas`. */
function schema(name: string): string[] {
  const lines = OPENAPI.split('\n');
  const start = lines.findIndex((line) => line === `    ${name}:`);
  expect(start, `schema ${name} is missing from the OpenAPI contract`).toBeGreaterThan(-1);
  const end = lines.findIndex(
    (line, index) => index > start && /^ {4}[A-Za-z]/.test(line),
  );
  return lines.slice(start, end === -1 ? lines.length : end);
}

function propertyDefault(name: string, property: string): string | null {
  const lines = schema(name);
  const start = lines.findIndex((line) => line === `        ${property}:`);
  expect(start, `${name}.${property} is missing`).toBeGreaterThan(-1);
  for (let index = start + 1; index < lines.length; index += 1) {
    const match = /^ {10}default: (.+)$/.exec(lines[index]);
    if (match) return match[1].trim();
    if (/^ {8}\S/.test(lines[index])) break;
  }
  return null;
}

function recommendDefaults(): Map<string, string> {
  const start = DOCS.indexOf('`POST /recommend`:');
  const end = DOCS.indexOf('`objective=\'min_gpus\'`', start);
  expect(start, 'the /recommend defaults table is missing').toBeGreaterThan(-1);

  const table = DOCS.slice(start, end === -1 ? undefined : end);
  return new Map(
    [...table.matchAll(/^\| `([^`]+)` \| `([^`]+)` \|/gm)].map((match) => [
      match[1],
      match[2],
    ]),
  );
}

describe('API reference facts', () => {
  it('lists exactly the endpoints the OpenAPI contract declares', () => {
    expect([...documentedEndpoints()].sort()).toEqual([...openApiPaths()].sort());
  });

  it('documents /predict and marks /estimate as its deprecated alias', () => {
    expect(openApiPaths().has('/predict')).toBe(true);
    expect(OPENAPI).toContain('  /predict:');

    const lines = OPENAPI.split('\n');
    const start = lines.findIndex((line) => line === '  /estimate:');
    expect(start).toBeGreaterThan(-1);
    const end = lines.findIndex((line, index) => index > start && /^ {2}\S/.test(line));
    const block = lines.slice(start, end === -1 ? lines.length : end).join('\n');
    expect(block).toContain('deprecated: true');
    expect(DOCS).toMatch(/`\/estimate`\s*\|\s*Deprecated alias for `\/predict`/);
  });

  it('quotes the request defaults the contract declares', () => {
    const documented = recommendDefaults();
    expect(documented.size).toBeGreaterThan(0);

    for (const field of ['isl', 'osl', 'top_n', 'objective', 'sla_mode']) {
      expect(documented.get(field), `docs/api.md documents no default for ${field}`).toBeDefined();
      expect(documented.get(field), `${field} default drifted from OpenAPI`).toBe(
        propertyDefault('RecommendRequest', field),
      );
    }
  });

  it('describes the capacity objective\'s larger default comparison set', () => {
    const row = DOCS.split('\n').find((line) => line.startsWith('| `top_n` |'));
    expect(row).toBeDefined();
    expect(row).toContain('concurrency_capacity');
    expect(row).toContain('`8`');
    expect(row).toContain('`6`');
    // The contract says the same thing; the prose is not a second source.
    const block = schema('RecommendRequest').join('\n');
    expect(block).toMatch(/concurrency_capacity[\s\S]*8 options[\s\S]*6 otherwise/);
  });

  it('requires the fields the memory endpoint requires', () => {
    const lines = schema('MemoryRequest');
    const required = lines.find((line) => /^ {6}required:/.test(line));
    expect(required).toBeDefined();
    expect(required).toContain('model_path');
    expect(required).toContain('system');
    expect(DOCS).toContain('`POST /memory` requires `model_path` and `system`');
  });
});
