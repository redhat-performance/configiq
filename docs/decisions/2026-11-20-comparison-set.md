# ADR: Compare result snapshots as a local set

**Date**: 2026-11-20
**Status**: Accepted

## Context

Sizing, performance, and frontier results are shown one at a time. A user
choosing between a larger model on last year's GPUs, a smaller model on newer
ones, or a different simulator has to keep the previous numbers in their head or
in a spreadsheet. The comparison page that used to exist was removed together
with the estimate export actions: it carried an implicit baseline and could not
say which simulator, engine version, or workload produced a row.

What a comparison needs is not a server-side notebook. It is a small set of
result snapshots the user deliberately kept, each carrying enough provenance to
be interpreted later, shown side by side with the differences made explicit. The
snapshots are measured figures, not requests, and the feedback that produced
them — replay commands, input files, simulator bundles — does not belong in a
comparison: it is large, tied to a container layout, and does not answer "which
configuration is better for this workload".

## Decision

**The comparison set lives in the browser.** It is stored under the
`localStorage` key `gc-comparison-set` as a versioned document (`schemaVersion:
1`). Nothing is sent to a ConfigIQ service, so a comparison is private to the
browser it was built in and disappears when site data is cleared. There is no
account model, and this ADR deliberately does not add one.

**An entry is a result, not a request.** Each entry stores the figures the page
displayed together with the identity needed to read them: the shared-view link
fields, model, GPU system, simulator, serving mode, sizing goal, backend and
engine version, workload shape, and the measured latency, throughput, memory,
and cost inputs. Each entry uses `realGpuCount` for cost, and cost arithmetic
stays in `lib/comparison/cost.ts` pinned to `HOURS_PER_MONTH` and
`AMORT_MONTHS_5YR`, so two rows cannot disagree about the month length or the
amortization window. The replay record and its artifacts are not stored.

**Adding is idempotent.** A snapshot is upserted on its link, simulator, engine
version, and measured figures together. Re-saving the same result moves it rather
than duplicating it; changing any of those parts makes it a different snapshot.
The four cost figures are deliberately outside that identity: a run saved with
costings off and the same run opened later with prices filled in are one result,
not two.

**A missing cost is completed, not recomputed.** A result page records the money
figures it could see at save time. If costings were off, or the price catalogue
had not arrived, those figures stayed unavailable and the row looked free. The
saved-results page reads the infrastructure catalogue unconditionally and fills
only the gaps: a figure the page computed is never replaced, a snapshot with no
published price stays unavailable rather than becoming zero, and each monthly /
five-year pair is derived from the other when only one was saved so the two
cannot disagree. Before writing, the page re-reads the stored set and copies the
completed figures onto it by entry id, so a result another tab added or removed
in the meantime is left alone. The detail view holds the entry's id rather than a
copy and reads the completed document, so it cannot show a stale figure while it
is open. The table, the detail view, and an export all read that one document.

**A sizing row keeps its target.** A recommendation is defined by what it was
asked to achieve, so a sizing snapshot carries that goal — a latency bound
(TTFT/TPOT or end-to-end), an arrival rate, or a resident-sequence count —
beside the measured figures. Predict entries have no goal. The field is
additive: documents written before it existed stay readable, and their rows
show no goal. A sizing result does not report total VRAM, so that figure stays
unavailable rather than being inferred from the catalog.

**The set is bounded and refuses growth rather than evicting entries.** It holds
at most 100 entries and 1 MB. Overflow is reported to the user and the save is
refused: silently dropping a row the user chose to keep would make the
comparison they are looking at incomplete without saying so.

**The baseline is chosen, never assumed.** Each row offers a radio control, and
the differences are rendered relative to that row. A user who has not chosen a
baseline sees absolute values, not a delta against whichever row happened to be
first. Deltas use ▲/▼ as direction without a good/bad judgement, because higher
is better for throughput and lower is better for latency.

**Import and export are explicit.** The set exports to CSV for a spreadsheet and
JSON for re-import. An imported JSON document is validated as a whole — schema
version, required fields, and bounds — before it replaces anything. A document
that fails validation changes nothing and says what was wrong. Import never
executes content or fetches a simulator.

**The page renders a semantic table.** It uses a plain `<table>` with a CSS
module modeled on the existing metric table and PatternFly's `Toolbar` for the
action bar, not `@patternfly/react-table`. The table is the comparison; its
cells need explicit alignment, measurement units, and delta styling that the
generic table component does not express without overriding most of it. The
chosen baseline is marked once per section band, above its column, rather than
repeated in every cell.

The page is **Saved results**, under Estimate in the navigation. `/compare`
keeps its path so existing links continue to work. It is registered in the
shared-view registry and its route labels, but it has no view of its own to
share: the set is browser-local, so a link cannot name it. Each entry instead
carries the shared-view link of the result page that produced it, so a saved row
can reopen the view behind its figures.

## Consequences

- A comparison is per browser and per device. There is no sync, no sharing of
  the set itself, and no server-side recovery.
- The schema is versioned from the start. A future change that reinterprets a
  stored field must bump `schemaVersion` and migrate or refuse the old document;
  adding a field is additive.
- Because replay artifacts are excluded, a comparison cannot reproduce a row.
  Reproducing a result remains the job of the page's Debug panel and the
  candidate's replay command, where the inputs and versions are shown.
- The bounds are part of the contract. Raising them is a schema change for the
  1 MB limit and a UI decision for the entry count, not a quiet constant edit.
- If a server-side or shared comparison is ever wanted, it needs an identity
  model and a storage service; this ADR does not provision either. The stored
  entry shape is the starting point for that migration.
