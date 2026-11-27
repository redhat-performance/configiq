# Multi-stage Next.js 16 build
FROM node:24-alpine AS builder

RUN apk add --no-cache git

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# Runtime stage
FROM node:24-alpine

LABEL org.opencontainers.image.source=https://github.com/redhat-performance/configiq
LABEL org.opencontainers.image.licenses=Apache-2.0

WORKDIR /app

ENV NODE_ENV=production

COPY --from=builder --chown=1001:0 /app/.next/standalone ./
COPY --from=builder --chown=1001:0 /app/.next/static ./.next/static
COPY --from=builder --chown=1001:0 /app/public ./public

# The Apache 2.0 text travels with every published image, so a pulled image can
# answer what it is licensed under without going back to the repository.
COPY LICENSE /licenses/LICENSE

RUN chmod -R g=u /app

USER 1001

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/api/health', (r) => {if (r.statusCode !== 200) throw new Error(r.statusCode)})"

CMD ["node", "server.js"]
