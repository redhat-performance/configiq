export function parseInferenceXImageTag(image) {
  if (typeof image !== 'string') return null
  const imageWithoutDigest = image.split('@', 1)[0]
  const imageName = imageWithoutDigest.slice(imageWithoutDigest.lastIndexOf('/') + 1)
  const separator = imageName.lastIndexOf(':')
  return separator >= 0 && separator < imageName.length - 1
    ? imageName.slice(separator + 1)
    : null
}
