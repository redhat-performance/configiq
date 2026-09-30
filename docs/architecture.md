# Architecture

## Application structure

ConfigIQ is a single Next.js application using the App Router.

```
Browser → nginx → Next.js container → AISimulators / AICostings gateways
```

The Next.js application runs in a Podman container managed by systemd. nginx
terminates TLS and forwards browser traffic to the Next.js container. The
Next.js API routes make server-side requests to AISimulators and AICostings.
GPU recommendations and memory estimation run in the separate AISimulators
service; pricing data comes from AICostings.

## Key principle: math is isolated

GPU sizing formulas live in the AISimulators service. React components call the
same-origin Next.js API routes, which proxy requests to AISimulators. The
legacy `lib/gpu-math/` code remains for historical and fallback use only.

This means:
- Formulas are testable without rendering components
- The same logic can be reused across multiple pages
- API response adapters keep the service contract separate from UI components

## Adding persistence later

If persistence is added later, the plan is:
- Add PostgreSQL via Prisma
- Extend the existing Next.js Route Handlers in `app/api/`
- Keep GPU calculations in AISimulators rather than reimplementing them in Next.js
- No rewriting of existing components needed
