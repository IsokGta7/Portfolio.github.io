// The site is exported as static HTML for GitHub Pages. When it is served from
// a project page (https://<owner>.github.io/<repo>/) the deploy workflow sets
// NEXT_PUBLIC_BASE_PATH to "/<repo>"; for a user page it stays empty.
const basePath = (process.env.NEXT_PUBLIC_BASE_PATH ?? "").replace(/\/+$/, "")

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  basePath,
  assetPrefix: basePath || undefined,
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
}

export default nextConfig
