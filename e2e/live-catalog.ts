import { expect, type Page } from '@playwright/test'

interface CatalogSystem {
  id: string
  name: string
}

interface SimulatorCatalogEntry {
  id: string
  available: boolean
  systems: CatalogSystem[]
}

interface CatalogResponse {
  simulators?: SimulatorCatalogEntry[]
  systems?: CatalogSystem[]
}

export async function readAisimulateCatalog(page: Page): Promise<CatalogSystem[]> {
  const response = await page.request.get('/api/catalog')
  expect(response.ok(), 'The deployed catalog must be available').toBeTruthy()
  const catalog = await response.json() as CatalogResponse
  if (Array.isArray(catalog.simulators)) {
    const aisimulate = catalog.simulators.find(entry => entry.id === 'aisimulate')
    expect(aisimulate?.available, 'AISimulate must be available in the deployed catalog').toBe(true)
    expect(aisimulate?.systems.length, 'AISimulate must report GPU systems').toBeGreaterThan(0)
    return aisimulate!.systems
  }
  // Older deployed environments reported one unscoped simulator catalog.
  expect(catalog.systems?.length, 'The deployed catalog must report GPU systems').toBeGreaterThan(0)
  return catalog.systems!
}
