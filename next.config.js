/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  // Keep the live development compiler isolated from production builds.
  // Running `next build` while `next dev` is active must not replace the CSS
  // and chunk files that the browser is currently using.
  distDir: process.env.NODE_ENV === 'development' ? '.next-dev' : '.next',
  async redirects() {
    return [
      {
        source: '/performance',
        destination: '/predict',
        permanent: true,
      },
    ];
  },
  async headers() {
    return [
      {
        source: '/widgets/:path*',
        headers: [
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Cross-Origin-Resource-Policy', value: 'cross-origin' },
          { key: 'Cache-Control', value: 'public, max-age=0, s-maxage=300, must-revalidate' },
        ],
      },
    ];
  },
  transpilePackages: [
    "@patternfly/react-core",
    "@patternfly/react-charts",
    "@patternfly/react-icons",
    "@patternfly/react-table",
  ],
  webpack(config, { dev }) {
    if (dev) {
      config.cache = { type: 'memory' };
    }
    config.ignoreWarnings = [
      ...(config.ignoreWarnings || []),
      {
        module: /@patternfly[\\/]react-styles[\\/]css[\\/]components[\\/](ActionList|OverflowMenu)/,
        message: /autoprefixer: start value has mixed support/,
      },
    ];
    return config;
  },
};

module.exports = nextConfig;
