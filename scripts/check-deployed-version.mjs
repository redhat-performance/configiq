#!/usr/bin/env node

// Resolve the build identity of a deployed ConfigIQ instance.
//
// The e2e workflows read /api/version before running any case, then check out
// that commit so the specs and the deployment are the same code. Without this,
// a release host that lags main fails every case with assertions about fields
// the deployed build does not send yet.
//
// Outputs (for GITHUB_OUTPUT): commit, version.
// Exits non-zero when the deployment does not report a usable commit, so a
// workflow never falls back to testing a different revision.

import { appendFileSync } from 'node:fs'

const baseUrl = (process.env.PLAYWRIGHT_BASE_URL ?? '').replace(/\/+$/, '')
if (!baseUrl) {
  console.error('PLAYWRIGHT_BASE_URL is required (for example https://configiq.dev)')
  process.exit(2)
}

let response
try {
  response = await fetch(`${baseUrl}/api/version`, {
    headers: { accept: 'application/json' },
    signal: AbortSignal.timeout(15_000),
  })
} catch (error) {
  console.error(`Could not reach ${baseUrl}/api/version: ${error instanceof Error ? error.message : String(error)}`)
  process.exit(1)
}

if (!response.ok) {
  console.error(`${baseUrl}/api/version returned HTTP ${response.status}. Deploy a build that exposes build metadata.`)
  process.exit(1)
}

const body = await response.json().catch(() => null)
if (body == null || typeof body !== 'object') {
  console.error(`${baseUrl}/api/version did not return a JSON object.`)
  process.exit(1)
}

const reportedCommit = typeof body.gitCommit === 'string' ? body.gitCommit : ''
const version = typeof body.version === 'string' ? body.version : 'unknown'
const commit = /^[0-9a-f]{40}$/.test(reportedCommit) ? reportedCommit : ''
if (!commit) {
  console.error(`${baseUrl}/api/version does not report a valid git commit. Deploy a build that reports its commit.`)
  process.exit(1)
}
const shortCommit = commit.substring(0, 7)

console.log(`Deployed ${baseUrl}: ${version} (${shortCommit})`)

if (process.env.GITHUB_OUTPUT) {
  appendFileSync(process.env.GITHUB_OUTPUT, `commit=${commit}\nversion=${version}\n`)
}

if (process.env.GITHUB_STEP_SUMMARY) {
  appendFileSync(process.env.GITHUB_STEP_SUMMARY, [
    '### Deployed build',
    '',
    `- Host: ${baseUrl}`,
    `- Version: ${version}`,
    `- Commit: ${shortCommit}`,
    '',
  ].join('\n'))
}

const expected = process.env.EXPECTED_COMMIT
if (expected && expected !== commit) {
  const message = `The deployment is serving ${shortCommit}, not ${expected.substring(0, 7)}`
  if (process.env.REQUIRE_MATCH === '1') {
    console.error(`${message}. Refusing to run tests against a different revision.`)
    process.exit(1)
  }
  console.log(`::notice::${message}. This run is pinned to the deployed commit.`)
}
