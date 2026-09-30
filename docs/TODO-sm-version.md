# TODO: Add sm_version to AISimulators systems API

## Context

CodeRabbit review flagged that NVFP4 KV cache architecture warnings should use numeric compute capability (SM version) instead of string-based architecture names for better future-proofing.

## Current State

The AISimulators `/systems` API returns:
```json
{
  "id": "b200_sxm",
  "architecture": "blackwell",  // ← string only
  "memory_bytes": 193273528320,
  ...
}
```

## Requested Enhancement

Add `sm_version` field to system specs:

```json
{
  "id": "b200_sxm",
  "architecture": "blackwell",
  "sm_version": 100,  // ← NEW: numeric compute capability
  ...
}
```

## Mapping

- SM90 = Hopper (H100, H200)
- SM100 = Blackwell datacenter (B200, B300, GB200, GB300)
- SM103 = Blackwell variant
- SM120 = Blackwell consumer (RTX 5090)

## Benefit

Enables precise hardware capability checks in ConfigIQ:

```typescript
// Future-proof check
if (testKVCachePrecision === 'NVFP4' && gpuSmVersion < 100) {
  warnings.push('NVFP4 requires SM100+ (Blackwell)');
}
```

Instead of string matching against an ever-growing list of architecture names.

## Priority

Low — AISimulators handles software fallback gracefully, so warnings are informational only.

## Related

- PR #57: NVFP4 precision support
- CodeRabbit review: https://github.com/redhat-performance/configiq/pull/57
