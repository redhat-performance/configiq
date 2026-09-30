# Sizing widget developer guide

Use this guide when changing the public sizing component. See the
[integration guide](./embeddable-sizing-widget.md) for host setup and API
examples.

## Code map

| Path | Purpose |
| --- | --- |
| `public/widgets/configiq-sizing-widget-v1.js` | Component, styles, validation, and request lifecycle |
| `tests/widgets/configiq-sizing-widget.test.mjs` | Public contract and lifecycle tests |
| `tests/widgets/fixtures/configiq-sizing-widget-host.html` | Cross-origin host preview |
| `next.config.js` | Widget CORS and cache headers |
| `app/predict/performance-prefill.ts` | Full-workflow query handoff |

## Component boundaries

The component owns its controls, validation, requests, status messages,
results, accessibility, and responsive layout. The host supplies catalog
mappings, seed values, an endpoint or proxy, theme tokens, and an optional link
to the full workflow.

Use the `config` property for models, GPUs, and seed values. Keep attributes for
small settings such as endpoint, timeout, theme, heading level, and links. Use
documented `--configiq-widget-*` properties for host styling.

## Preserve these behaviors

- Map the six inputs through `buildSizingRequest`.
- Validate responses through `normalizeSizingResponse` before rendering.
- Cancel active requests and ignore late responses after an edit.
- Preserve in-progress edits when an equivalent `config` object arrives.
- Keep widget instances isolated.
- Accept only HTTP(S) full-workflow links.
- Keep status and metric labels available to assistive technology.

The V1 URL is a mutable, backward-compatible channel. Browsers revalidate it,
and shared caches retain it for up to five minutes. Publish breaking API changes
under a new major component URL.

## Validate a change

Run the focused tests first, then the repository checks:

```bash
npm test -- tests/widgets/configiq-sizing-widget.test.mjs
npm run type-check
npm run lint
npm run build
```

Add or update tests for every public property, attribute, state transition, or
failure path that changes.

For visual changes, review the native and dark themes at 1440 and 375 pixels.
Capture every affected success, empty, loading, invalid, or unavailable state,
then update the images in `docs/screenshots/embeddable-sizing-widget/`.
