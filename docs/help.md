# In-app help

Every page has a help button in the masthead. It opens one popover with three
ways to get unstuck: search the documentation and the page itself, ask a
question, or take a guided tour.

## Search

Search covers three local sources:

- documentation sections and their headings
- glossary definitions for the jargon a page uses
- the named controls on each tool

The controls on the page you are on are listed before you type, and a matching
control is labelled as a page result rather than a documentation result.

It does not index the model or GPU catalog. Those are live facts owned by the
API, and copying them into a search index would let the answer go stale. Read
them where they are shown: the GPU explorer for hardware, or the tool you are
already using for the models and systems it offers.

A control result opens the page it lives on. When that control has a
documentation section, a **Docs** link beside it opens the explanation directly.

## Ask

The question box explains the documentation: what a page is for, what a control
changes, and what a term means. It reads the same sections as search, so it
answers from the documentation this site ships with rather than from the live
API.

It does not read the model or GPU catalog, current pricing offers, or the
results of a calculation. For a specific model, GPU, or configuration, open the
page that shows it. It also never runs sizing: a sizing question ends with a
link to the right tool, carrying the values you named, so you press **Calculate**
(or **Compare**) and the result appears in the full page. A value only travels
when the page's link contract accepts it, and the link says which fields it sets
and which named values it left out, with the reason: a value the contract
refuses, a question that offers two alternatives, an input the destination page
does not have, or an input the selected goal does not use. The model and GPU
system come from your settings unless the question names its own, in which case
the question wins. Anything else the question did not name keeps the value the
page already had. The answer cites the documentation sections it used. If a
number cannot be traced to a source or to your question, the panel shows the
sources without the prose rather than presenting an unverified value.

Ask is optional and can be switched off by the deployment. When it is off, the
button, search, and tours still work.

## Tours

Tours are manual. Nothing opens one automatically, and no "seen" flag is
stored, so a tour starts only when you choose **Take a tour**. The tour follows
the page's current layout, skips a step whose target is missing, and closes on
Escape, **Skip tour**, or **Got it** on the last step. It never changes your
inputs.

## Documentation

The documentation index is at [/docs](/docs), and every page has a stable
address such as `/docs/settings#sharing-a-view`. A deep link carries the
simulator, backend, and backend version that produced the view; opening one
never changes your saved settings. A link writes only the values that differ
from the page's starting state, so a partial link keeps your own values for
whatever it leaves out. See
[Settings and simulators](settings.md#sharing-a-view) for what happens when the
link's simulator differs from yours and for the fields a link always carries.

## Configuration

Self-hosted deployments opt in to the question box with:

| Variable | Meaning |
|---|---|
| `ASSISTANT_ENABLED` | Set to `true` to show the question box |
| `ASSISTANT_MODEL` | Model id served by the inference endpoint |
| `ASSISTANT_INFERENCE_URL` | OpenAI-compatible endpoint that answers questions |
| `TRITON_INFERENCE_URL` | FIL classifier endpoint; a legacy fallback for the above |

Without `ASSISTANT_ENABLED`, `ASSISTANT_MODEL`, and an inference endpoint the
assistant reports that it is unavailable. Search, tours, and the documentation
are build-time and local, so they need no configuration.

The assistant and the FIL classifiers are separate services. The FIL endpoint
serves `/v2/models/<name>/infer` and cannot answer a chat question, so
`ASSISTANT_INFERENCE_URL` points at the assistant's own server.
`TRITON_INFERENCE_URL` is only read when the assistant endpoint is unset, which
keeps a deployment that predates the split working as it did before; setting
both always prefers the assistant endpoint.

The assistant also offers the model read-only catalog, pricing, and memory
lookups as tools. An endpoint that does not support tool calling ignores them,
and the answer then comes from the documentation alone; either way, a number in
the answer is checked against a documentation section or a tool result before
the prose is shown.
