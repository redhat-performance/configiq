/**
 * Server-only registry loader. Browser code imports the pure contracts from
 * `registry-types` instead, so `node:fs` never reaches a client bundle.
 */

import { readFile } from 'node:fs/promises'
import { parseRegistry, type TestedModelsRegistry } from './registry-types'

export * from './registry-types'

export function registryPath(): string {
  return `${process.env.TESTED_MODELS_DIR ?? '/tested-models'}/registry.json`
}

export async function readTestedModelsRegistry(): Promise<TestedModelsRegistry> {
  return parseRegistry(JSON.parse(await readFile(registryPath(), 'utf8')))
}
