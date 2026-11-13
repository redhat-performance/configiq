#!/usr/bin/env node
/**
 * Render the docs to HTML at build time.
 *
 * Outputs:
 *   lib/docs/generated/<slug>.json   rendered HTML + table of contents
 *   lib/docs/generated/index.ts      static registry imported by the route
 *   public/help-index.json           search entries for the help popover
 *
 * Fails the build on a link to a Markdown file that is not in the manifest, or
 * on a `/docs/<slug>#anchor` link whose anchor does not exist. That is the
 * mechanism that keeps cross-document links from rotting.
 */
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkRehype from 'remark-rehype';
import rehypeSlug from 'rehype-slug';
import rehypeStringify from 'rehype-stringify';

const ROOT = process.cwd();
const MANIFEST_PATH = path.join(ROOT, 'docs/manifest.json');
const GENERATED_DIR = path.join(ROOT, 'lib/docs/generated');
const HELP_INDEX_PATH = path.join(ROOT, 'public/help-index.json');

const LINK_RE = /^(?:\.\/)?([\w./-]+\.md)(#[^)]*)?$/;
const RELATIVE_RE = /^\.{1,2}\//;
const GITHUB_BLOB = 'https://github.com/redhat-performance/configiq/blob/main/';

function normalisePath(value) {
  return value.split(path.sep).join('/');
}

function slugFromFile(file, fileToSlug) {
  const normalised = normalisePath(file);
  return fileToSlug.get(normalised);
}

function walk(node, visit) {
  visit(node);
  if (Array.isArray(node.children)) {
    for (const child of node.children) walk(child, visit);
  }
}

function rewriteLinks(tree, section, fileToSlug, errors) {
  const dir = path.posix.dirname(normalisePath(section.file));

  walk(tree, (node) => {
    if (node.type !== 'link' || typeof node.url !== 'string') return;
    const match = LINK_RE.exec(node.url);

    if (!match) {
      // Relative non-Markdown links (specs, screenshots) are not served by the
      // app. Point them at the repository so they still resolve.
      if (RELATIVE_RE.test(node.url)) {
        const absolute = path.posix.normalize(path.posix.join(dir, node.url));
        node.url = `${GITHUB_BLOB}${absolute}`;
      }
      return;
    }

    const [, relative, hash = ''] = match;
    const absolute = path.posix.normalize(path.posix.join(dir, relative));
    const slug = slugFromFile(absolute, fileToSlug);

    if (!slug) {
      errors.push(
        `[${section.slug}] link to "${node.url}" points at a Markdown file that is not in docs/manifest.json`,
      );
      return;
    }

    node.url = `/docs/${slug}${hash}`;
  });
}

function decodeEntities(value) {
  return value
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, ' ');
}

function htmlToText(html) {
  return decodeEntities(
    html
      .replace(/<pre[\s\S]*?<\/pre>/g, ' ')
      .replace(/<code[\s\S]*?<\/code>/g, ' ')
      .replace(/<[^>]+>/g, ' ')
      .replace(/\s+/g, ' ')
      .trim(),
  );
}

function extractHeadings(html) {
  const headings = [];
  const re = /<h([1-3])[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/h\1>/g;
  let match;
  while ((match = re.exec(html)) !== null) {
    headings.push({
      level: Number(match[1]),
      id: match[2],
      title: htmlToText(match[3]),
      index: match.index,
      end: re.lastIndex,
    });
  }
  return headings;
}

/** Split rendered HTML into searchable sections at h1/h2 boundaries. */
function extractSearchSections(html, section) {
  const headings = extractHeadings(html).filter((heading) => heading.level <= 2);
  const entries = [];

  if (headings.length === 0) {
    entries.push({
      id: `docs/${section.slug}`,
      kind: 'docs',
      title: section.title,
      section: section.title,
      body: htmlToText(html),
      url: `/docs/${section.slug}`,
    });
    return entries;
  }

  // Intro before the first heading belongs to the document root.
  const intro = htmlToText(html.slice(0, headings[0].index));
  if (intro) {
    entries.push({
      id: `docs/${section.slug}`,
      kind: 'docs',
      title: section.title,
      section: section.title,
      body: intro,
      url: `/docs/${section.slug}`,
    });
  }

  headings.forEach((heading, position) => {
    const next = headings[position + 1];
    const body = htmlToText(html.slice(heading.end, next ? next.index : html.length));
    const isDocumentTitle = heading.level === 1;
    entries.push({
      id: `docs/${section.slug}#${heading.id}`,
      kind: 'docs',
      title: heading.title,
      section: section.title,
      body,
      url: isDocumentTitle
        ? `/docs/${section.slug}`
        : `/docs/${section.slug}#${heading.id}`,
    });
  });

  return entries;
}

function collectIds(html) {
  const ids = new Set();
  const re = /\sid="([^"]+)"/g;
  let match;
  while ((match = re.exec(html)) !== null) ids.add(match[1]);
  return ids;
}

function validateAnchors(rendered, errors) {
  const re = /href="\/docs\/([\w-]+)#([^"]+)"/g;
  for (const [slug, html] of rendered) {
    let match;
    while ((match = re.exec(html)) !== null) {
      const [, targetSlug, anchor] = match;
      const target = rendered.get(targetSlug);
      if (!target) {
        errors.push(`[${slug}] link to unknown docs section "${targetSlug}"`);
        continue;
      }
      if (!collectIds(target).has(anchor)) {
        errors.push(`[${slug}] link to missing anchor "#${anchor}" in /docs/${targetSlug}`);
      }
    }
  }
}

async function main() {
  const manifest = JSON.parse(await readFile(MANIFEST_PATH, 'utf8'));
  const sections = [...manifest.sections].sort((a, b) => a.order - b.order);
  const fileToSlug = new Map(
    sections.map((section) => [normalisePath(section.file), section.slug]),
  );

  const errors = [];
  const rendered = new Map();
  const searchEntries = [];

  for (const section of sections) {
    let source;
    try {
      source = await readFile(path.join(ROOT, section.file), 'utf8');
    } catch {
      errors.push(`[${section.slug}] cannot read ${section.file}`);
      continue;
    }

    const processor = unified()
      .use(remarkParse)
      .use(remarkGfm)
      .use(() => (tree) => rewriteLinks(tree, section, fileToSlug, errors))
      .use(remarkRehype)
      .use(rehypeSlug)
      .use(rehypeStringify);

    const file = await processor.process(source);
    const html = String(file);
    rendered.set(section.slug, html);

    const headings = extractHeadings(html);
    const toc = headings
      .filter((heading) => heading.level === 2 || heading.level === 3)
      .map((heading) => ({ id: heading.id, title: heading.title, level: heading.level }));

    searchEntries.push(...extractSearchSections(html, section));

    await mkdir(GENERATED_DIR, { recursive: true });
    await writeFile(
      path.join(GENERATED_DIR, `${section.slug}.json`),
      `${JSON.stringify({ slug: section.slug, title: section.title, html, toc }, null, 2)}\n`,
    );
  }

  validateAnchors(rendered, errors);

  if (errors.length > 0) {
    console.error('Docs build failed:');
    for (const error of errors) console.error(`  - ${error}`);
    process.exit(1);
  }

  const indexFile = `// Generated by scripts/build-docs.mjs. Do not edit.\n${sections
    .map(
      (section, position) =>
        `import doc${position} from './${section.slug}.json';\n`,
    )
    .join('')}
export const GENERATED_DOCS = {
${sections.map((section, position) => `  '${section.slug}': doc${position},`).join('\n')}
} as const;
`;
  await writeFile(path.join(GENERATED_DIR, 'index.ts'), indexFile);

  await mkdir(path.dirname(HELP_INDEX_PATH), { recursive: true });
  await writeFile(
    HELP_INDEX_PATH,
    `${JSON.stringify({ entries: searchEntries }, null, 2)}\n`,
  );

  // The assistant retrieves from a generated module rather than reading
  // `public/` at runtime, so the corpus in the production image is the corpus
  // the build validated.
  await writeFile(
    path.join(GENERATED_DIR, 'help-index.json'),
    `${JSON.stringify({ entries: searchEntries }, null, 2)}\n`,
  );

  console.log(
    `docs: rendered ${sections.length} sections and ${searchEntries.length} search entries`,
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
