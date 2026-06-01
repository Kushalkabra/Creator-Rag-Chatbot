/** @type {import('next').NextConfig} */
const backendUrl = (process.env.BACKEND_URL ?? process.env.RAILWAY_BACKEND_URL ?? "")
  .trim()
  .replace(/\/$/, "");

const nextConfig = {
  async rewrites() {
    if (!backendUrl) {
      return [];
    }
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
