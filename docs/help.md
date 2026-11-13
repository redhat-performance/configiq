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
API, and copying them into a search index would let the answer go stale. Ask
about a specific model or GPU in the question box instead, where the answer is
read from the catalog at the time you ask.

A control result opens the page it lives on. When that control has a
documentation section, a **Docs** link beside it opens the explanation directly.

## Ask

The question box explains the documentation. It can also read a small set of
read-only facts: the GPU, model, backend, and network catalogs, current pricing
offers, and a memory estimate for one model on one system.

It never runs sizing. A sizing question ends in a link to the right tool with
the values you named already filled in, so you press **Calculate** (or
**Compare**) yourself and can see the result in the full page. The answer cites
the documentation sections it used. If a number cannot be traced to a source,
the panel shows the sources without the prose rather than presenting an
unverified value.

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
never changes your saved settings. See
[Settings and simulators](settings.md#sharing-a-view) for what happens when the
link's simulator differs from yours.

## Configuration

Self-hosted deployments opt in to the question box with:

| Variable | Meaning |
|---|---|
| `ASSISTANT_ENABLED` | Set to `true` to show the question box |
| `ASSISTANT_MODEL` | Model id served by the inference endpoint |
| `TRITON_INFERENCE_URL` | OpenAI-compatible inference endpoint the assistant calls |

Without all three the assistant reports that it is unavailable. Search, tours,
and the documentation are build-time and local, so they need no configuration.
