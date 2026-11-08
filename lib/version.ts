// Application version and build metadata
// Auto-generated at build time

export const VERSION = {
  number: process.env.NEXT_PUBLIC_GIT_VERSION || '0.2.0',
  gitCommit: process.env.NEXT_PUBLIC_GIT_COMMIT || 'dev',
  environment: process.env.NEXT_PUBLIC_VERCEL_ENV || process.env.NODE_ENV || 'development'
} as const

export function getVersionString(): string {
  const version = VERSION.number.startsWith('v') ? VERSION.number : `v${VERSION.number}`;
  return version;
}

export function getShortCommit(): string {
  return VERSION.gitCommit.substring(0, 7)
}
