/**
 * Precision metadata used by prompts, calculators, and glossary copy.
 *
 * Kept in `lib/` so non-component modules (the glossary, the help index) can
 * import it without depending on the Predict page.
 */
export const PRECISION_METADATA = {
  FP16: { bytes: 2.0, label: 'FP16/BF16', desc: '2 bytes/param (full precision)' },
  FP8: { bytes: 1.0, label: 'FP8', desc: '1 byte (8-bit float)' },
  INT8: { bytes: 1.0, label: 'INT8', desc: '1 byte (8-bit integer)' },
  INT4: { bytes: 0.5, label: 'INT4', desc: '0.5 bytes (4-bit integer)' },
  MXFP4: { bytes: 0.56, label: 'MXFP4', desc: '0.56 bytes (Microscaling FP4, Hopper+)' },
  NVFP4: { bytes: 0.58, label: 'NVFP4', desc: '0.58 bytes (NVIDIA FP4, Blackwell only)' },
} as const;

export const KV_PRECISION_METADATA = {
  FP16: { bytes: 2.0, label: 'FP16', desc: '2 bytes per element (standard)' },
  FP8: { bytes: 1.0, label: 'FP8', desc: '1 byte per element (half the memory, slight quality loss)' },
  NVFP4: { bytes: 0.58, label: 'NVFP4', desc: '0.58 bytes (NVIDIA FP4, Blackwell only)' },
} as const;
