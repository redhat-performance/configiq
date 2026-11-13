# Contributing guide

This document covers everything you need to start contributing to ConfigIQ alongside other engineers.

## 1. Getting access

Ask the repo owner to add you as a collaborator. Once you accept the email invite you will have push access to create branches and open pull requests.

You will also need:

- **Node.js 20 or later** — check with `node -v`
- **npm 10 or later** — check with `npm -v`
- **Git** — check with `git --version`
- **GitHub CLI (optional but useful)** — [cli.github.com](https://cli.github.com)

---

## 2. First-time setup

```bash
# Clone the repo
git clone https://github.com/redhat-performance/configiq.git
cd configiq

# Install dependencies (also wires up pre-commit hooks automatically)
npm install

# Start the dev server
npm run dev
```

The app runs at **http://localhost:3000**.

If port 3000 is in use, Next.js will move to 3001. Check the terminal output for the actual URL.

### Useful commands

| Command | What it does |
|---|---|
| `npm run dev` | Start dev server with hot reload |
| `npm run build` | Production build — run this before opening a PR |
| `npm run type-check` | TypeScript check without building |
| `npm run lint` | ESLint check |
| `npm test` | Run the test suite (builds the docs first) |
| `npm run docs:build` | Render Markdown into the app and the help index |
| `npm run docs:check` | Verify the generated docs match the Markdown sources |

`docs:build` runs automatically before `dev`, `build`, and `test`, so you rarely
call it by hand. `docs:check` never writes anything; use it to confirm a change
that touches documentation is complete before you commit.

---

## 3. How the codebase is structured

```
app/                    Next.js App Router pages
  layout.tsx            Root layout — fonts, PatternFly CSS imports
  page.tsx              Homepage
  predict/              Predict performance tool
  recommend/            Recommend sizing tool
  kv-cache/             KV cache calculator
  gpu-explorer/         GPU Explorer
  hybrid-savings/       Hybrid Savings
  routing/              Routing Economics
  cluster-cost/         Cluster cost calculator
  docs/                 Documentation index and rendered Markdown pages
  api/                  Same-origin proxy routes for the services

components/
  layout/
    AppShell.tsx        Top nav bar and PatternFly Page wrapper
  help/                 Help popover: search, ask, and tours

lib/
  api/                  AISimulators API clients and response adapters
  assistant/            Assistant tools and prompt (server-side only)
  deep-links/           Shared-view link format and parsing
  docs/                 Docs manifest and generated output
  gpu-math/             Legacy static catalogs and historical fallback code
  help/                 Search corpus, glossary, tours, and affordances
  pricing/              Pricing data and helpers
  hooks/                Same-origin catalog and pricing data hooks
  utils/
    format.ts           Number and unit formatting helpers

scripts/
  build-docs.mjs        Renders docs/ and writes the help index

docs/                   Architecture docs, guides, and decision records
public/                 Static assets
```

The most important rule: **GPU math belongs in AISimulators**. Components call
the `lib/api/` clients through the same-origin `app/api/` routes and display
results. Do not add new sizing formulas to `lib/gpu-math/`.

---

## 4. Day-to-day workflow

The `main` branch is protected. Nobody can push to it directly — all changes go through a pull request that requires:

- CI to pass (type-check + lint + build)
- At least one approving review from another engineer

### Step-by-step

**1. Always start from an up-to-date main**

```bash
git checkout main
git pull origin main
```

**2. Create a feature branch**

Use a short, descriptive name in this format:

```bash
git checkout -b feature/gpu-explorer-filters
git checkout -b fix/mobile-nav-overflow
git checkout -b chore/update-model-catalog
```

Prefixes: `feature/` for new functionality, `fix/` for bug fixes, `chore/` for maintenance.

**3. Make your changes and commit**

Pre-commit hooks run automatically when you `git commit`. They check lint and types on staged files and block the commit if anything fails — fix the error and try again.

```bash
git add app/predict/page.tsx
git commit -s -m "Add GPU memory breakdown chart to estimate panel"
```

Write commit messages in the imperative ("Add", "Fix", "Remove") and describe what the change does, not what files changed.

Always commit with `-s`. Every commit must carry a `Signed-off-by:` trailer
matching the committer identity, and `-s` adds it for you. For a commit you
already made, fix it with `git commit --amend -s`.

**4. Push your branch**

```bash
git push origin feature/gpu-explorer-filters
```

**5. Open a pull request**

```bash
gh pr create --title "Add GPU memory breakdown chart" --body "Short description of what and why"
```

Or open one from github.com. Target branch should always be `main`.

CI runs automatically. You will see checks appear on the PR page.

**6. Request a review**

Tag the other engineer for review. While you wait, you can start reviewing their open PRs.

**7. Address feedback**

Push additional commits to the same branch. The CI re-runs and any previous approval is dismissed automatically, so they will need to re-approve after you push changes.

**8. Merge**

Once CI passes and the review is approved, merge the PR. Delete the branch after merging — GitHub shows a button for this.

**9. Pull main and start the next branch**

```bash
git checkout main
git pull origin main
git checkout -b feature/next-thing
```

---

## 5. Avoiding conflicts

Two engineers working at the same time will sometimes touch the same files. The most likely conflict points in this repo are:

| File | Why it conflicts |
|---|---|
| `app/globals.css` | All shared CSS lives here |
| `app/predict/Performance.tsx` | Large file, most active page |
| `lib/api/` | API contracts and response adapters |
| `package.json` | Dependency changes |

**Practical rules to avoid painful merges:**

- Keep PRs small and short-lived. A PR that takes a week to land will conflict with everything.
- Coordinate verbally before both editing the same section (e.g. "I'm touching the estimate panel today, can you work on the model gallery?").
- Pull `main` before starting a new branch, not just at the start of the day.
- If your branch is more than a day old and others have merged, rebase before opening a PR:

```bash
git fetch origin
git rebase origin/main
```

If you hit a merge conflict during rebase, Git will pause and show the conflicting file. Open it, resolve the marked sections, then:

```bash
git add <conflicting-file>
git rebase --continue
```

---

## 6. Code conventions

Follow these so the codebase stays consistent across contributors.

### PatternFly only for UI

Do not install Tailwind, shadcn/ui, Material UI, or any other component library. PatternFly v6 is the only component library. If you need a UI pattern, check [patternfly.org](https://www.patternfly.org) first.

### Red Hat design system for colors and fonts

Use the established CSS variables — do not introduce arbitrary hex colors:

```css
/* Current theme tokens */
var(--gc-brand-red)     /* #ee0000 — brand red, logo only */
var(--gc-link)          /* #0066cc — interactive elements */
var(--gc-text)          /* #151515 — primary text */
var(--gc-text-2)        /* #3c3f42 — body and secondary text */
var(--border)           /* #d2d2d2 — borders */

/* Compatibility tokens */
var(--rh-red-50)        /* #ffeaea — selected state backgrounds */
var(--rh-gray-95)       /* #151515 — primary text */
var(--rh-gray-50)       /* #6a6e73 — legacy secondary text */
var(--rh-gray-20)       /* #d2d2d2 — borders */
var(--rh-gray-10)       /* #f0f0f0 — page background */

/* Font variables */
var(--font-display)     /* Red Hat Display — headings */
var(--font-body)        /* Red Hat Text — body copy */
var(--font-mono)        /* Red Hat Mono — code, labels */
```

Font weights: use `400` (regular) and `500` (medium) for body text. `700` bold is for Display headings only.

### Sentence case everywhere

Red Hat brand standard. No title case.

```
✓ What model are you serving?
✓ See full breakdown
✗ What Model Are You Serving?
✗ See Full Breakdown
```

### TypeScript strict — no `any`

Define proper interfaces. The CI type-check step will catch `any` types.

### Server components by default

Add `"use client"` only when the component uses browser APIs, state (`useState`), or event handlers. If in doubt, check whether removing it causes an error.

### Documentation is rendered at build time

User-facing documentation lives in `docs/*.md`, and `scripts/build-docs.mjs`
renders it into the app. Never parse Markdown in the browser or add a
Markdown-rendering package to the client bundle.

To add or change a page:

1. Edit the Markdown file, or add the new file under `docs/`.
2. Register it in `docs/manifest.json` with a unique `slug`, a `title`, a
   one-sentence `description`, the repo-relative `file`, and an `order`.
3. Link it from `docs/overview.md`; the manifest test fails if a section is
   not reachable from the overview.
4. Run `npm run docs:build`.

Links between documents use the file name: `[Costs](costs.md)`. They are
rewritten to `/docs/costs`. Links that point at an anchor must match a heading
in that document, or the build fails.

The same build produces `public/help-index.json`, which powers the in-app help
search, so a documentation change is also a change to what search can find.

### No unnecessary comments

Only add a comment when the *why* is non-obvious — a hidden constraint, a tricky invariant, a workaround for a specific bug. Well-named variables and functions are self-documenting.

---

## 7. PR checklist

Before marking a PR ready for review:

- [ ] `npm run type-check` passes locally
- [ ] `npm run lint` passes locally
- [ ] `npm test` passes locally
- [ ] `npm run build` passes locally
- [ ] Documentation changes are registered in `docs/manifest.json` and
      `npm run docs:check` passes
- [ ] New GPU math is in AISimulators, not in a component or `lib/gpu-math/`
- [ ] No new third-party UI libraries added to `package.json`
- [ ] All UI text uses sentence case
- [ ] No arbitrary hardcoded hex colors — use established `--gc-*`, `--rh-*`, or
      `--font-*` variables as appropriate

---

## 8. What is intentionally not here yet

Do not add these until there is a real requirement:

- Database (Prisma, PostgreSQL, Supabase)
- Authentication (NextAuth.js, Clerk)
- Tailwind CSS
- Monorepo tooling (Turborepo, Nx)

These will be introduced in later phases.

---

## 9. Getting help

- **CLAUDE.md** and **AGENTS.md** — context files for coding agents; also useful background for any contributor
- **docs/** — architecture decisions and tech stack rationale
- Open a GitHub issue if you find a bug or want to propose a feature before building it
