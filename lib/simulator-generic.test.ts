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
const SIMULATOR = /\b(?:AISimulate|AISim|BLIS|aisimulate|aisim|blis)\b/;
const CAPABILITY =
  /\b(?:accepts?|rejects?|requires?|needs?|cannot|can't|does not|doesn't|takes?|supports?)\b|(?<!\/)\bmodels?\b/i;

/** True when a comment states a capability for a named simulator. */
function statesSimulatorCapability(text: string): boolean {
  return text
    .split(/(?<=[.!?])\s+/)
    .some(sentence => SIMULATOR.test(sentence) && CAPABILITY.test(sentence));
}

function commentText(line: string): string | null {
  let trimmed = line.trim();
  if (!(trimmed.startsWith('//') || trimmed.startsWith('/*') || trimmed.startsWith('*'))) {
    return null;
  }
  if (trimmed === '*/') return '';
  trimmed = trimmed.replace(/^\/\/|^\/\*|^\*/, '');
  if (trimmed.endsWith('*/')) trimmed = trimmed.slice(0, -2).trimEnd();
  return trimmed;
}

interface CommentBlock {
  /** 1-based line where the block starts. */
  line: number;
  text: string;
}

/**
 * Group consecutive comment lines so a note wrapped across `//` lines is read
 * as one statement rather than as fragments too short to name a capability.
 * A code line or a blank line ends the block.
 */
function commentBlocks(lines: string[]): CommentBlock[] {
  const blocks: CommentBlock[] = [];
  let current: CommentBlock | null = null;
  const flush = () => {
    if (current !== null) blocks.push(current);
    current = null;
  };
  lines.forEach((line, index) => {
    const comment = commentText(line);
    if (comment === null) {
      flush();
      return;
    }
    const text = comment.trim();
    if (current === null) {
      current = { line: index + 1, text };
      return;
    }
    current.text = [current.text, text].filter(Boolean).join(' ');
  });
  flush();
  return blocks;
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
    return commentBlocks(readFileSync(path, 'utf8').split('\n'))
      .filter(block => statesSimulatorCapability(block.text))
      .map(block => `${path}:${block.line}: ${block.text}`);
  });
}

describe('simulator-generic frontend', () => {
  it('detects the note shape this guard exists for', () => {
    expect(statesSimulatorCapability('AISim does not accept a fabric.')).toBe(true);
    expect(statesSimulatorCapability('BLIS needs an inter-node fabric for P/D.')).toBe(true);
    expect(statesSimulatorCapability('The simulator rejects a fabric it does not model.')).toBe(false);
    expect(statesSimulatorCapability('AISim catalog, from the /models endpoint.')).toBe(false);
  });

  it('reads a note wrapped across comment lines as one statement', () => {
    const blocks = commentBlocks([
      'const backend = catalog.backends[0];',
      '// BLIS',
      '// needs an inter-node fabric for P/D.',
      '',
      '// The simulator rejects a fabric it does not model.',
    ]);

    expect(blocks.map(block => block.line)).toEqual([2, 5]);
    expect(statesSimulatorCapability(blocks[0].text)).toBe(true);
    expect(statesSimulatorCapability(blocks[1].text)).toBe(false);
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
