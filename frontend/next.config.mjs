/** @type {import('next').NextConfig} */
function normalizeBackendUrl(raw) {
  let url = (raw ?? "").trim().replace(/\/$/, "");
  if (!url) {
    return "";
  }
  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    url = `https://${url}`;
  }
  return url;
}

const backendUrl = normalizeBackendUrl(
  process.env.BACKEND_URL ?? process.env.RAILWAY_BACKEND_URL
);

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
