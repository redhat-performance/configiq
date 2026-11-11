import { describe, expect, it } from 'vitest'
import {
  backendVersionsComparable,
  backendVersionsMatch,
  normalizeBackendVersion,
} from '../benchmark-versions'

describe('backendVersionsMatch', () => {
  it('matches a bare release pin to a vendor build', () => {
    expect(backendVersionsMatch('0.5.14', 'v0.5.14-rocm720-mi35x')).toBe(true)
    expect(backendVersionsMatch('v0.5.14-rocm720-mi35x', '0.5.14')).toBe(false)
  })

  it('keeps different releases and different vendor builds distinct', () => {
    expect(backendVersionsMatch('0.5.14', 'v0.5.21-rocm720-mi35x')).toBe(false)
    expect(backendVersionsMatch('v0.5.14-rocm720-mi35x', 'v0.5.14-cuda130')).toBe(false)
  })

  it('treats a pre-release suffix as part of the release core', () => {
    expect(normalizeBackendVersion('tensorrt-llm', '1.3.0rc29')).toBe('v1.3.0rc29')
    expect(backendVersionsMatch('1.3.0rc29', 'v1.3.0rc29-vendor')).toBe(true)
    expect(backendVersionsMatch('1.3.0rc20', 'v1.3.0rc29')).toBe(false)
  })
})

describe('backendVersionsComparable', () => {
  it('compares a catalog release with its measured vendor build in either order', () => {
    expect(backendVersionsComparable('0.5.14', 'v0.5.14-rocm720-mi35x')).toBe(true)
    expect(backendVersionsComparable('v0.5.14-rocm720-mi35x', '0.5.14')).toBe(true)
  })

  it('still keeps distinct builds and releases apart', () => {
    expect(backendVersionsComparable('v0.5.14-rocm720-mi35x', 'v0.5.14-cuda130')).toBe(false)
    expect(backendVersionsComparable('0.5.14', 'v0.5.21-rocm720-mi35x')).toBe(false)
  })
})
