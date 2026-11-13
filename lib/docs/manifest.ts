import manifest from '../../docs/manifest.json';

export interface DocsSection {
  slug: string;
  title: string;
  /** Repository-relative Markdown path, used by the build script. */
  file: string;
  order: number;
}

export interface DocsManifest {
  sections: DocsSection[];
}

const parsed = manifest as DocsManifest;

/** Every rendered documentation section, in navigation order. */
export const DOCS_SECTIONS: DocsSection[] = [...parsed.sections].sort(
  (a, b) => a.order - b.order,
);

export function getDocsSection(slug: string): DocsSection | undefined {
  return DOCS_SECTIONS.find((section) => section.slug === slug);
}
