import { expect, type Page } from '@playwright/test'

export interface CatalogSystem {
  id: string
  name: string
  memory_bytes?: number
}

interface SimulatorCatalogEntry {
  id: string
  label?: string
  version?: string | null
  default?: boolean
  available?: boolean
  systems?: CatalogSystem[]
  models?: Array<string | { id?: string }>
  backends?: Array<{ id?: string }>
  objectives?: string[]
}

interface CatalogResponse {
  simulators?: SimulatorCatalogEntry[]
}

/** One simulator's own catalogue, as the deployed service reports it. */
export interface LiveSimulatorCatalog {
  id: string
  label: string
  /** Release of the simulator itself, not of the engine it models. */
  version: string | null
  /** Whether a profile with no saved simulator resolves to this entry. */
  default: boolean
  available: boolean
  systems: CatalogSystem[]
  models: string[]
  backends: string[]
  objectives: string[]
}

// The catalog aggregates every simulator's systems and model metadata, so it is
// slower than a page request. Give it room rather than failing a case that is
// otherwise fine when several workers hit the deployment at once.
const CATALOG_TIMEOUT_MS = 60_000

class SimulatorCatalogError extends Error {}

/**
 * Read every simulator the deployed catalog advertises.
 *
 * A live case drives a page whose simulator is a user choice, so it must know
 * which simulator is under test before it can assert the request and response
 * name it. A catalog that reports no simulators cannot answer that question,
 * and living with the ambiguity would let a case pass while proving nothing.
 */
export async function readSimulatorCatalogs(page: Page): Promise<LiveSimulatorCatalog[]> {
  const response = await page.request.get('/api/catalog', { timeout: CATALOG_TIMEOUT_MS })
  expect(response.ok(), 'The deployed catalog must be available').toBeTruthy()
  const catalog = await response.json() as CatalogResponse
  expect(
    Array.isArray(catalog.simulators),
    'The deployed catalog must report its simulators so a live case can name the one it tests',
  ).toBeTruthy()
  return (catalog.simulators ?? []).map(entry => ({
    id: entry.id,
    label: typeof entry.label === 'string' && entry.label.length > 0 ? entry.label : entry.id,
    version: typeof entry.version === 'string' && entry.version.length > 0 ? entry.version : null,
    default: entry.default === true,
    available: entry.available !== false,
    systems: Array.isArray(entry.systems) ? entry.systems : [],
    models: (entry.models ?? []).flatMap(model =>
      typeof model === 'string'
        ? [model]
        : typeof model.id === 'string' && model.id.length > 0 ? [model.id] : []),
    backends: (entry.backends ?? []).flatMap(backend =>
      typeof backend.id === 'string' && backend.id.length > 0 ? [backend.id] : []),
    objectives: (entry.objectives ?? []).filter(objective => typeof objective === 'string'),
  }))
}

/**
 * Read the catalogue of one simulator, or of the catalog's default simulator
 * when no id is named.
 *
 * Naming a simulator that the deployment does not advertise, or one whose
 * capabilities it could not read, fails here: a case must not silently run
 * against another simulator's catalogue.
 */
export async function readSimulatorCatalog(page: Page, simulator?: string): Promise<LiveSimulatorCatalog> {
  const catalogs = await readSimulatorCatalogs(page)
  const entry = simulator
    ? catalogs.find(candidate => candidate.id === simulator)
    : catalogs.find(candidate => candidate.default) ?? catalogs[0]
  if (!entry) {
    throw new SimulatorCatalogError(
      simulator
        ? `The deployed catalog does not list ${simulator}`
        : 'The deployed catalog marks no default simulator',
    )
  }
  expect(entry.available, `${entry.id} must be available in the deployed catalog`).toBe(true)
  expect(entry.systems.length, `${entry.id} must report GPU systems`).toBeGreaterThan(0)
  return entry
}
