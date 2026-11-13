import type { Metadata } from 'next';
import Link from 'next/link';
import { DOCS_SECTIONS } from '@/lib/docs/manifest';
import styles from './docs.module.css';

export const metadata: Metadata = {
  title: 'Documentation · ConfigIQ',
  description:
    'Guides for inference sizing, GPU comparison, cost modeling, the REST API, and the in-app help system.',
};

export default function DocsIndexPage() {
  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <p className={styles.eyebrow}>Documentation</p>
        <h1 className={styles.title}>ConfigIQ documentation</h1>
        <p className={styles.lede}>
          Guides for using the tools and for running the services behind them.
          Every page has a stable address you can share.
        </p>
      </header>

      <nav aria-label="Documentation pages">
        <ul className={styles.indexGrid}>
          {DOCS_SECTIONS.map((section) => (
            <li key={section.slug}>
              <Link href={`/docs/${section.slug}`} className={styles.indexCard}>
                <span className={styles.indexTitle}>{section.title}</span>
                <span className={styles.indexDescription}>{section.description}</span>
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
}
