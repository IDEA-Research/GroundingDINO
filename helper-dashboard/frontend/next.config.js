/** @type {import('next').NextConfig} */
//
// WARNING: `rewrites()` below is fully evaluated at `next build` time,
// not at `next start` time. Once you build with a given
// NEXT_PUBLIC_API_BASE, that backend URL is baked into the compiled
// output and changing the env later has no effect. scripts/dev.sh
// handles this by writing a sentinel `.next/.api_base_baked` and
// rebuilding whenever the env differs. If you launch `next start`
// manually, pass NEXT_PUBLIC_API_BASE to BOTH `npm run build` and
// `npx next start` (and rebuild whenever you change it).
//
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    const backend =
      process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
    return [
      // Keep /api/chat/message handled by app route handler
      // (frontend/app/api/chat/message/route.ts) instead of rewrite proxy.
      {
        source: "/api/chat/message",
        destination: "/api/chat/message",
      },
      {
        // Exclude both /api/chat/message and /api/chat/message/*
        // so app route handlers can serve async chat enqueue + polling.
        source: "/api/:path((?!chat/message(?:/.*)?$).*)",
        destination: `${backend}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
