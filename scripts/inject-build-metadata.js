#!/usr/bin/env node

// Inject build metadata into environment variables
// Run this in package.json build script before next build

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function execGit(command) {
  try {
    return execSync(command, { encoding: 'utf8', stderr: 'ignore' }).trim();
  } catch (e) {
    return null;
  }
}

// Get git commit hash
const gitCommit = execGit('git rev-parse HEAD') || 'unknown';

// Get version from git tag (most recent tag) - null if no tag exists
const gitVersion = execGit('git describe --tags --abbrev=0');

// Write to .env.local for Next.js to pick up
const envPath = path.join(__dirname, '..', '.env.local');

// Clear previous auto-generated metadata
let envContent = '';
if (fs.existsSync(envPath)) {
  envContent = fs.readFileSync(envPath, 'utf8');
  // Remove old auto-generated section and all known metadata keys
  envContent = envContent
    .replace(/# Auto-generated build metadata[^\n]*\n/g, '')
    .replace(/NEXT_PUBLIC_BUILD_TIME=.*\n/g, '')
    .replace(/NEXT_PUBLIC_GIT_COMMIT=.*\n/g, '')
    .replace(/NEXT_PUBLIC_GIT_VERSION=.*\n/g, '')
    .trim();
}

// The version and commit identify the build; a wall-clock timestamp adds no
// information and would make identical builds look different.
const metadataLines = [
  '# Auto-generated build metadata - DO NOT EDIT',
  `NEXT_PUBLIC_GIT_COMMIT=${gitCommit}`,
];

if (gitVersion) {
  metadataLines.push(`NEXT_PUBLIC_GIT_VERSION=${gitVersion}`);
}

const metadata = '\n' + metadataLines.join('\n') + '\n';

fs.writeFileSync(envPath, envContent + '\n' + metadata);

console.log('✅ Build metadata injected:');
console.log(`   Git commit: ${gitCommit.substring(0, 7)}`);
if (gitVersion) {
  console.log(`   Git version: ${gitVersion}`);
}
