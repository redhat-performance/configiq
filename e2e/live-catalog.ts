import { expect, type Page } from '@playwright/test'

interface CatalogSystem {
  id: string
  name: string
}

interface SimulatorCatalogEntry {
  id: string
  available: boolean
  systems: CatalogSystem[]
  objectives?: string[]
}

interface CatalogResponse {
  simulators?: SimulatorCatalogEntry[]
  systems?: CatalogSystem[]
  objectives?: string[]
}

// The catalog aggregates every simulator's systems and model metadata, so it is
// slower than a page request. Give it room rather than failing a case that is
// otherwise fine when several workers hit the deployment at once.
const CATALOG_TIMEOUT_MS = 60_000

export async function readAisimulateCatalogDetails(page: Page): Promise<{ systems: CatalogSystem[]; objectives: string[] }> {
  const response = await page.request.get('/api/catalog', { timeout: CATALOG_TIMEOUT_MS })
  expect(response.ok(), 'The deployed catalog must be available').toBeTruthy()
  const catalog = await response.json() as CatalogResponse
  if (Array.isArray(catalog.simulators)) {
    const aisim = catalog.simulators.find(entry => entry.id === 'aisim')
    expect(aisim?.available, 'AISim must be available in the deployed catalog').toBe(true)
    expect(aisim?.systems.length, 'AISim must report GPU systems').toBeGreaterThan(0)
    return { systems: aisim!.systems, objectives: aisim!.objectives ?? [] }
  }
  // Older deployed environments reported one unscoped simulator catalog.
  expect(catalog.systems?.length, 'The deployed catalog must report GPU systems').toBeGreaterThan(0)
  return { systems: catalog.systems!, objectives: catalog.objectives ?? [] }
}

export async function readAisimulateCatalog(page: Page): Promise<CatalogSystem[]> {
  return (await readAisimulateCatalogDetails(page)).systems
}
