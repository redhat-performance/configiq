/**
 * A complete, purchasable server configuration.
 *
 * The shared costing feed currently exposes indicative per-GPU prices. Those
 * are useful catalogue data, but they are not a defensible substitute for the
 * price of a complete server (chassis, CPUs, memory, networking and support).
 * Hybrid Savings therefore ranks purchased hardware only when the exact GPU
 * topology has a complete-server estimate.
 */
export interface ServerPurchaseConfiguration {
  gpuCount: number
  purchasePrice: number
  installationCost: number
  sourceLabel: string
  sourceUrl: string | null
  sourceDate: string
  indicative: boolean
}

type PurchaseCatalogue = Record<string, Record<number, ServerPurchaseConfiguration>>

const FALLBACK_SOURCE_DATE = '2026-08-01'

function configuration(
  gpuCount: number,
  purchasePrice: number,
  installationCost: number,
  sourceLabel: string,
  sourceUrl: string | null,
  sourceDate = FALLBACK_SOURCE_DATE,
): ServerPurchaseConfiguration {
  return {
    gpuCount,
    purchasePrice,
    installationCost,
    sourceLabel,
    sourceUrl,
    sourceDate,
    indicative: true,
  }
}

const l40s = {
  1: configuration(1, 35_000, 2_500, 'Thinkmate NVIDIA L40S server catalogue', 'https://www.thinkmate.com/systems/servers/gpx/l40s'),
  2: configuration(2, 56_000, 3_900, 'Thinkmate NVIDIA L40S server catalogue', 'https://www.thinkmate.com/systems/servers/gpx/l40s'),
  4: configuration(4, 94_000, 6_600, 'Thinkmate NVIDIA L40S server catalogue', 'https://www.thinkmate.com/systems/servers/gpx/l40s'),
  8: configuration(8, 150_000, 10_500, 'Thinkmate NVIDIA L40S server catalogue', 'https://www.thinkmate.com/systems/servers/gpx/l40s'),
}

const a100 = {
  1: configuration(1, 34_000, 2_400, 'Exxact public-sector server price schedule', 'https://www.gsaadvantage.gov/ref_text/GS35F0278Y/0ZQME9.3VGZ8X_GS-35F0278Y_EXXACTPRICELIST2024.PDF'),
  2: configuration(2, 59_000, 4_100, 'Exxact A100 server catalogue and price schedule', 'https://configurator.exxactcorp.com/configure/TWS-194019223'),
  4: configuration(4, 95_000, 6_700, 'Public four-GPU A100 80 GB server and component-price range', 'https://www.gsaadvantage.gov/ref_text/GS35F0278Y/0ZQME9.3VGZ8X_GS-35F-0278Y_EXXACTPRICELIST2024.PDF'),
  8: configuration(8, 160_000, 11_200, 'Public eight-GPU A100 80 GB server market range', 'https://www.qubrid.com/bare-metal-gpu-servers'),
}

const h100 = {
  1: configuration(1, 48_000, 3_900, 'Exxact HGX H100 server catalogue', 'https://www.exxactcorp.com/Exxact-TS4-101818584-E101818584'),
  2: configuration(2, 98_000, 6_900, 'Exxact HGX H100 server catalogue', 'https://www.exxactcorp.com/Exxact-TS4-101818584-E101818584'),
  8: configuration(8, 285_000, 22_300, 'Exxact eight-GPU HGX H100 configured system price', 'https://configurator.exxactcorp.com/configure/TS4-193475697'),
}

const h200 = {
  1: configuration(1, 65_000, 6_000, 'H200 SXM GPU market price plus single-GPU server chassis', 'https://cpq.exxactcorp.com/quote/view_quote_revs.php?qo_id=173436', '2026-04-28'),
  2: configuration(2, 110_000, 7_700, 'Derived from Exxact one- and four-GPU H200 system prices', 'https://www.exxactcorp.com/Exxact-TS4-180208180-E180208180'),
  8: configuration(8, 370_000, 18_900, 'Exxact and GSA eight-GPU H200 system prices', 'https://www.exxactcorp.com/Exxact-TS4-118380266-E118380266'),
}

export const PURCHASE_CATALOGUE: PurchaseCatalogue = {
  l40s,
  a100_sxm: a100,
  h100_sxm: h100,
  h200_sxm: h200,
  b200_sxm: {
    8: configuration(8, 404_000, 28_300, 'Exeton eight-GPU HGX B200 optimized configuration range', 'https://exeton.com/configure/ts4-169219634', '2026-09-09'),
  },
  l4: {
    1: configuration(1, 18_000, 1_300, 'OEM and GPU-server reseller market range', 'https://www.thinkmate.com/systems/servers/gpx/a10'),
    4: configuration(4, 24_000, 1_700, 'Public four-GPU L4 server and OEM chassis market range', 'https://jarvislabs.ai/blog/l4-gpu-price'),
  },
  rtx_pro_6000_server: {
    1: configuration(1, 34_766, 2_434, 'B3IQ complete one-GPU RTX PRO 6000 Blackwell server configuration', 'https://www.b3iq.org/machines/rtx-pro-6000', '2026-09-09'),
    8: configuration(8, 165_000, 11_550, 'Blackwell Cloud complete eight-GPU RTX PRO 6000 server configuration', 'https://www.blackwellcloud.com/nvidia-rtx-pro-6000-gpu-servers', '2026-09-09'),
  },
}

/** Whether at least one complete-server topology can be purchased. */
export function hasServerPurchaseConfiguration(systemId: string): boolean {
  return Object.keys(PURCHASE_CATALOGUE[systemId] ?? {}).length > 0
}

function validTopology(value: number): number | null {
  if (!Number.isFinite(value) || value < 0.5) return null
  const rounded = Math.round(value)
  return Math.abs(rounded - value) <= 0.1 ? rounded : null
}

/**
 * Return every complete server that can contain one model replica. Larger
 * servers remain separate candidates because they may pack several replicas
 * and become cheaper at higher demand; their full server price is preserved.
 */
export function resolveCompatibleServerPurchaseConfigurations(
  systemId: string,
  gpusPerReplica: number,
): ServerPurchaseConfiguration[] {
  const requiredGpus = validTopology(gpusPerReplica)
  if (requiredGpus == null) return []

  return Object.values(PURCHASE_CATALOGUE[systemId] ?? {})
    .filter(configuration => configuration.gpuCount >= requiredGpus)
    .sort((left, right) => (
      left.gpuCount - right.gpuCount || left.purchasePrice - right.purchasePrice
    ))
}
