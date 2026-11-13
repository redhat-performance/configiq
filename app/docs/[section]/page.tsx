import type { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { DOCS_SECTIONS, getDocsSection } from '@/lib/docs/manifest';
import { GENERATED_DOCS } from '@/lib/docs/generated';
import type { GeneratedDoc } from '@/lib/docs/types';
import styles from '../docs.module.css';

interface DocsPageProps {
  params: Promise<{ section: string }>;
}

export function generateStaticParams() {
  return DOCS_SECTIONS.map((section) => ({ section: section.slug }));
}

export async function generateMetadata({ params }: DocsPageProps): Promise<Metadata> {
  const { section } = await params;
  const entry = getDocsSection(section);
  return {
    title: entry ? `${entry.title} · ConfigIQ docs` : 'ConfigIQ docs',
  };
}

export default async function DocsPage({ params }: DocsPageProps) {
  const { section } = await params;
  const entry = getDocsSection(section);
  const doc = GENERATED_DOCS[section as keyof typeof GENERATED_DOCS] as GeneratedDoc | undefined;

  if (!entry || !doc) notFound();

  const currentIndex = DOCS_SECTIONS.findIndex((item) => item.slug === section);
  const previous = currentIndex > 0 ? DOCS_SECTIONS[currentIndex - 1] : undefined;
  const next =
    currentIndex >= 0 && currentIndex < DOCS_SECTIONS.length - 1
      ? DOCS_SECTIONS[currentIndex + 1]
      : undefined;

  return (
    <div className={styles.page}>
      <div className={styles.layout}>
        <article className={styles.article}>
          <header className={styles.header}>
            <p className={styles.eyebrow}>Documentation</p>
          </header>

          {/* Content is repository-owned Markdown rendered at build time. It is
              never user-supplied and there is no client-side parser. */}
          <div dangerouslySetInnerHTML={{ __html: doc.html }} />

          <nav aria-label="Documentation pages" style={{ display: 'flex', gap: 24, marginTop: 40 }}>
            {previous && (
              <Link href={`/docs/${previous.slug}`} className={styles.tocLink}>
                ← {previous.title}
              </Link>
            )}
            {next && (
              <Link
                href={`/docs/${next.slug}`}
                className={styles.tocLink}
                style={{ marginLeft: 'auto' }}
              >
                {next.title} →
              </Link>
            )}
          </nav>
        </article>

        {doc.toc.length > 0 && (
          <nav className={styles.toc} aria-label="On this page">
            <p className={styles.tocTitle}>On this page</p>
            <ul className={styles.tocList}>
              {doc.toc.map((item) => (
                <li key={item.id}>
                  <a
                    href={`#${item.id}`}
                    className={`${styles.tocLink} ${item.level === 3 ? styles.tocLevel3 : ''}`}
                  >
                    {item.title}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
        )}
      </div>
    </div>
  );
}
