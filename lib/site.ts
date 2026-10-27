import { basePath } from "@/lib/base-path"

// Public address of the site, used for canonical links, Open Graph and the
// sitemap. The deploy workflow passes the URL GitHub Pages reports; local builds
// fall back to the user page on github.io.
export const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL || `https://isokgta7.github.io${basePath}`).replace(/\/+$/, "")

export function absoluteUrl(path: string): string {
  return `${siteUrl}${path.startsWith("/") ? path : `/${path}`}`
}
