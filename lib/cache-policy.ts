// Client-side cache lifetimes for data fetched through the same-origin API.
// Keep these aligned with the corresponding API response Cache-Control headers.
export const CLIENT_CACHE_TTL_MS = {
  catalog: 10 * 60 * 1000,
  costings: 60 * 60 * 1000,
} as const
