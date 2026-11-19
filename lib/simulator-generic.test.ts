import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

/**
 * Keep the frontend simulator-generic in prose as well as in code.
 *
 * Which simulator accepts, needs, or rejects a field is capability data that
 * changes between releases. A comment that states it for a named simulator goes
 * stale without a code change, and reading it invites the next change to branch
 * on the ID instead of the catalog flag. The service README, its OpenAPI
 * contract, and catalog metadata are the source for those facts.
 *
 * `lib/api/` is exempt because it is the documented boundary where simulator
 * identity is allowed to change behavior.
 */
const SIMULATOR = /\b(?:AISimulate|BLIS|aisimulate|blis)\b/;
const CAPABILITY =
  /\b(?:accepts?|rejects?|requires?|needs?|cannot|can't|does not|doesn't|takes?|supports?)\b|(?<!\/)\bmodels?\b/i;

/** True when a comment states a capability for a named simulator. */
function statesSimulatorCapability(text: string): boolean {
  return text
    .split(/(?<=[.!?])\s+/)
    .some(sentence => SIMULATOR.test(sentence) && CAPABILITY.test(sentence));
}

function commentText(line: string): string | null {
  const trimmed = line.trim();
  if (trimmed.startsWith('//')) return trimmed.slice(2);
  if (trimmed.startsWith('/*')) return trimmed.slice(2);
  if (trimmed.startsWith('*')) return trimmed.slice(1);
  return null;
}

function sourceFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'node_modules' || entry.name.startsWith('.')) return [];
      return sourceFiles(path);
    }
    return /\.tsx?$/.test(entry.name) ? [path] : [];
  });
}

function offenders(): string[] {
  const roots = ['app', 'components', 'lib'].flatMap(root => sourceFiles(root));
  return roots.flatMap(path => {
    if (path.startsWith(join('lib', 'api'))) return [];
    return readFileSync(path, 'utf8')
      .split('\n')
      .flatMap((line, index) => {
        const comment = commentText(line);
        if (comment === null || !statesSimulatorCapability(comment)) return [];
        return [`${path}:${index + 1}: ${line.trim()}`];
      });
  });
}

describe('simulator-generic frontend', () => {
  it('detects the note shape this guard exists for', () => {
    expect(statesSimulatorCapability('AISimulate does not accept a fabric.')).toBe(true);
    expect(statesSimulatorCapability('BLIS needs an inter-node fabric for P/D.')).toBe(true);
    expect(statesSimulatorCapability('The simulator rejects a fabric it does not model.')).toBe(false);
    expect(statesSimulatorCapability('AISimulate catalog, from the /models endpoint.')).toBe(false);
  });

  it('states capabilities instead of naming a simulator that accepts or needs them', () => {
    const found = offenders();
    expect(
      found,
      'Describe the catalog capability or contract rule, and put simulator-specific facts in the service README/OpenAPI:\n' +
        found.join('\n'),
    ).toEqual([]);
  });
});
