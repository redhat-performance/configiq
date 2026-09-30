# Multi-stage Next.js 14 build
ARG NEXT_PUBLIC_AISIMULATORS_API_URL=https://aisimulators.dev

FROM node:20-alpine AS builder

RUN apk add --no-cache git

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN NEXT_PUBLIC_AISIMULATORS_API_URL=${NEXT_PUBLIC_AISIMULATORS_API_URL} npm run build

# Runtime stage
FROM node:20-alpine

WORKDIR /app

ENV NODE_ENV=production

COPY --from=builder --chown=1001:0 /app/.next/standalone ./
COPY --from=builder --chown=1001:0 /app/.next/static ./.next/static
COPY --from=builder --chown=1001:0 /app/public ./public

RUN chmod -R g=u /app

USER 1001

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/api/health', (r) => {if (r.statusCode !== 200) throw new Error(r.statusCode)})"

CMD ["node", "server.js"]
