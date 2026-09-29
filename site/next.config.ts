import type { NextConfig } from "next";

// A static export, so the site can be served by anything -- GitHub Pages
// publishes it under /<repo>, which .github/workflows/site.yml passes in as
// PAGES_BASE_PATH. Unset locally and on Vercel, where it serves from /.
const nextConfig: NextConfig = {
  output: "export",
  basePath: process.env.PAGES_BASE_PATH || undefined,
  // /card/ becomes card/index.html, which a static host serves without rewrites.
  trailingSlash: true,
};

export default nextConfig;
