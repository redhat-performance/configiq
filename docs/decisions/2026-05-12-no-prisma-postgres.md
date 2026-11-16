# ADR: No Prisma or PostgreSQL

**Date**: 2026-05-12
**Status**: Accepted

## Decision

The application does not use a database.

## Reasons

- No persistence requirement for the current calculator tools
- Database schema is clearer once saved-scenario UX is actually designed
- Premature schema design leads to rewrites

## Consequences

- All state is ephemeral (in-component or URL params)
- No schema, migrations, or database to operate
