/**
 * Every route, and what a shared view means for it.
 *
 * A shared view is a contract plus the wiring that honours it. Both are
 * enforced where they are written — `defineAccessors` makes a contract field a
 * compile error until the page binds it, and the request census makes an
 * unclassified request key a test failure. Neither notices a *new page*: it
 * simply has no contract, and nothing says whether that was a decision.
 *
 * This registry is that decision, once per route. A page either names the link
 * it shares or gives a reason it has nothing to share, and a test fails when a
 * route exists in `app/` that the registry does not mention. Adding a page with
 * inputs therefore cannot ship without answering the question.
 */

import {
  gpuExplorerLink,
  hybridSavingsLink,
  kvCacheLink,
  recommendLink,
  validationCompareLink,
} from './pages';
import { predictLink } from './predict';

export interface SharedViewRoute {
  /** The route as served, with dynamic segments written literally. */
  path: string;
  /** Human name, for messages and docs. */
  name: string;
  /** The contract this page's links speak, or why it has none. */
  sharedView:
    | { kind: 'link'; fields: readonly string[] }
    | { kind: 'none'; reason: string };
}

/**
 * Routes are listed in nav order, then the pages without a shared view.
 *
 * `fields` is derived from the contract rather than repeated, so it cannot go
 * stale: a field added to a page's contract appears here, and its binding
 * obligation is already a compile error elsewhere.
 */
export const PAGE_REGISTRY: readonly SharedViewRoute[] = [
  {
    path: '/predict',
    name: 'Predict performance',
    sharedView: { kind: 'link', fields: Object.keys(predictLink.fields) },
  },
  {
    path: '/recommend',
    name: 'Recommend sizing',
    sharedView: { kind: 'link', fields: Object.keys(recommendLink.fields) },
  },
  {
    path: '/kv-cache',
    name: 'KV cache calculator',
    sharedView: { kind: 'link', fields: Object.keys(kvCacheLink.fields) },
  },
  {
    path: '/gpu-explorer',
    name: 'GPU explorer',
    sharedView: { kind: 'link', fields: Object.keys(gpuExplorerLink.fields) },
  },
  {
    path: '/hybrid-savings',
    name: 'Hybrid savings',
    sharedView: { kind: 'link', fields: Object.keys(hybridSavingsLink.fields) },
  },
  {
    path: '/validation/compare',
    name: 'Validation compare',
    sharedView: { kind: 'link', fields: Object.keys(validationCompareLink.fields) },
  },
  {
    path: '/compare',
    name: 'Compare results',
    sharedView: {
      kind: 'none',
      reason:
        'it renders the result snapshots saved in this browser; each entry carries the link of the page that produced it, and the set itself has no URL',
    },
  },
  {
    path: '/',
    name: 'Home',
    sharedView: { kind: 'none', reason: 'the landing page links to the tools and holds no inputs' },
  },
  {
    path: '/cluster-cost',
    name: 'Cluster cost',
    sharedView: {
      kind: 'none',
      reason:
        'its inputs describe a reader-owned cluster and its rates, which are not a sizing view anyone else can reproduce',
    },
  },
  {
    path: '/routing',
    name: 'Routing economics',
    sharedView: {
      kind: 'none',
      reason:
        'it is a reference model over fleet tiers rather than a sized view, and has no link contract yet',
    },
  },
  {
    path: '/sources',
    name: 'Cost sources and assumptions',
    sharedView: {
      kind: 'none',
      reason: 'it edits the reader\'s own saved assumptions, and a link never carries those',
    },
  },
  {
    path: '/settings',
    name: 'Settings',
    sharedView: {
      kind: 'none',
      reason: 'it edits the reader\'s saved defaults; a link changes a view and never settings',
    },
  },
  {
    path: '/docs',
    name: 'Documentation',
    sharedView: { kind: 'none', reason: 'documentation index; there is nothing to reproduce' },
  },
  {
    path: '/docs/[section]',
    name: 'Documentation section',
    sharedView: { kind: 'none', reason: 'documentation content addressed by its own URL' },
  },
  {
    path: '/tested-models',
    name: 'Tested models',
    sharedView: {
      kind: 'none',
      reason: 'it browses the tested-model registry, whose rows are addressed in the URL already',
    },
  },
  {
    path: '/tested-models/[pairId]',
    name: 'Tested model detail',
    sharedView: {
      kind: 'none',
      reason: 'the row it shows is named by the path segment, not by query parameters',
    },
  },
];

/** The registry entries that share a view. */
export function sharedViewRoutes(): readonly SharedViewRoute[] {
  return PAGE_REGISTRY.filter(route => route.sharedView.kind === 'link');
}
