import { redirect } from 'next/navigation';
import { DOCS_SECTIONS } from '@/lib/docs/manifest';

export default function DocsIndexPage() {
  redirect(`/docs/${DOCS_SECTIONS[0]?.slug ?? 'overview'}`);
}
