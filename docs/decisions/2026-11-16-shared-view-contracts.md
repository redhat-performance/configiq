# ADR: Enforce shared-view contracts structurally

**Date**: 2026-11-16
**Status**: Accepted

## Context

A shared view — the link behind "Copy link" — is a published interface. Someone
pastes one into a ticket, a bookmark, or a chat with a colleague, and it is
expected to reproduce the sized view it describes. That expectation is what
makes an omission in a link a bug rather than a missing feature: the recipient
gets a number that looks authoritative and describes a different experiment.

Each shareable page used to describe its fields three times: the contract in
`lib/deep-links/`, the builder that exported the current view, and the applier
that read a link back. Nothing tied them together, so adding an input meant
remembering two more edits. Three of those omissions had already shipped — the
recommend page never carried its per-pool context limits, the KV calculator
collapsed two independently configured pools into one, and the comparison page
could not name the measured row it was showing.

The peer services avoid this class of bug differently: in `services/`, the
Pydantic schema *is* the runtime validator, so a request that omits a field is
rejected at the boundary. The deep-link contract is only a parser; nothing
validates that the page wires every field.

## Decision

Three layers, each closing a different hole, and each enforced by something
that fails automatically rather than by review.

1. **One total accessor table per page** (`defineAccessors`, `useSharedView`).
   Each contract field has exactly one row saying how the view exports it and
   how a link applies it, and the export and import directions are both derived
   from that row. The table is total over the fields the page owns, so adding a
   field to `defineLink` is a compile error until the page binds it. Pages whose
   provenance trio (`backend`, `backendVersion`, `simulator`) comes from
   settings mark their contract with `withProvenance`, which narrows the
   obligation to the fields the page really owns.

2. **A request census per page** (`auditRequestCensus`). Totality proves a
   contract field reaches page state; it cannot see a request field no contract
   field produces. Each page declares how every request key relates to its
   contract — the fields it derives from, or why a shared view cannot carry it —
   and the audit reads the payload the page actually sends. An unclassified key,
   a key naming a field that does not exist, and a contract field that never
   reaches the request all fail. Writing the predict census is what found two
   override fields no link carried.

3. **A route registry with golden fixtures** (`lib/deep-links/registry.ts`).
   Neither of the first two layers notices a *new page*: it simply has no
   contract. Every route under `app/` is registered as either sharing a view or
   having a reason not to, and a test reads the filesystem so a new page cannot
   be added without answering the question. One golden link per page is pinned
   as an inline snapshot, so a change to a link's shape is a reviewable diff.

Adding a field to a contract is additive and does not bump `LINK_VERSION`. A
rename, removal, or change of meaning does: bump the version, keep reading the
previous one, and update the golden fixtures in the same commit. A link that
declares a version this build does not understand is refused whole rather than
half-applied.

## Rejected alternatives

- **A text check over page sources** (`scripts/check-deep-links.mjs` style).
  Tried first: a grep for contract field names across the pages produced
  fourteen false positives and no signal, because a field name appears in
  comments, labels, and unrelated state. It cannot tell a bound field from a
  coincidence, and it fails on refactors that are correct.
- **Requiring a full link contract for every page.** `/cluster-cost` and
  `/compare` describe a reader-owned cluster and this browser's saved
  estimates. A link cannot reproduce either, so the registry records that
  instead of inventing a contract to satisfy a rule.
- **Bumping `LINK_VERSION` for additive fields.** Every bump means every old
  link stops opening. Additions are backwards compatible while the parser
  ignores unknown keys and omitted fields keep the recipient's value.

## Consequences

- Adding an input to a shareable page means: contract field, accessor row
  (compiler-enforced), and — when it reaches the API — a census entry. The
  compiler and the census both say so.
- Adding a page means a registry entry, which forces the "does this share a
  view?" decision at the right time.
- The census needs the real payload. Builders therefore live in `lib/api/` (or,
  for the comparison page, beside it) and are imported by both the page and the
  test, rather than reconstructed in the test.
- Known gap: `/validation/compare` has no page-level mount test, because its
  flow needs a benchmark dataset fixture. Its staging merge and query builders
  are unit-tested and its contract is censused; a mount test is worth adding
  when a dataset fixture exists.
- `/routing` has inputs but no link. That is recorded as a reason, not as
  compliance: turning it into a shared view is a small, separate piece of work.
