import { describe, expect, it } from 'vitest'
import { parseInferenceXImageTag } from './inferencex-image-tag.mjs'

describe('InferenceX image version tags', () => {
  it('keeps the release tag when the image is pinned to a digest', () => {
    expect(parseInferenceXImageTag('lmsysorg/sglang:v0.5.22-rocm720-mi35x@sha256:c4b61360'))
      .toBe('v0.5.22-rocm720-mi35x')
  })

  it('handles registry ports and images without tags', () => {
    expect(parseInferenceXImageTag('registry.example:5000/vllm:v0.24.0')).toBe('v0.24.0')
    expect(parseInferenceXImageTag('lmsysorg/sglang@sha256:c4b61360')).toBeNull()
  })
})
