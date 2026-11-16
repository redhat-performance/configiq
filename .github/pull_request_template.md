## What changed

<!-- One or two sentences a reviewer can hold in their head. -->

## Checks

- [ ] `npm run type-check`, `npm run lint`, and the test suite pass locally
- [ ] Shared views — new or changed inputs are bound through the page's link
      contract (`defineAccessors` / `useSharedView`), and any new API request
      key is classified in the page's census
      (`lib/deep-links/request-census.test.ts`)
- [ ] New pages are registered in `lib/deep-links/registry.ts`, with a link
      contract and golden snapshot or a reason the page has nothing to share
- [ ] Link contract changes are additive, or `LINK_VERSION` is bumped with the
      previous version still readable and the golden fixtures updated
- [ ] Commits carry a `Signed-off-by:` trailer (`git commit -s`)

## Notes for review

<!-- Optional: anything surprising, deferred, or worth a second pair of eyes. -->
