import { basePath } from "@/lib/base-path"

// Public address of the site, used for canonical links, Open Graph and the
// sitemap. The deploy workflow passes the URL GitHub Pages reports; local builds
// fall back to the user page on github.io.
export const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL || `https://isokgta7.github.io${basePath}`).replace(/\/+$/, "")

export function absoluteUrl(path: string): string {
  return `${siteUrl}${path.startsWith("/") ? path : `/${path}`}`
}

export const githubUrl = "https://github.com/IsokGta7"
export const sourceUrl = "https://github.com/IsokGta7/Portfolio.github.io"

// Isaac fills these in. Each one stays hidden on the site while it is empty.
//   linkedin: full profile URL, e.g. "https://www.linkedin.com/in/<handle>/"
//   email:    plain address, rendered as a mailto: link
//   cvUrl:    a full URL, or a path under public/ such as "/cv.pdf"
export const contact = {
  linkedin: "",
  email: "",
  cvUrl: "",
}
