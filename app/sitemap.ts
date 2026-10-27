import type { MetadataRoute } from "next"
import { languages } from "@/config/translations"
import { absoluteUrl } from "@/lib/site"

export const dynamic = "force-static"

// Paths after the language segment, each published in every language.
const paths = ["", "number-prediction/", "flower-classifier/", "bitcoin-prediction/", "cpp-projects/", "python-projects/"]

export default function sitemap(): MetadataRoute.Sitemap {
  return paths.flatMap((path) =>
    languages.map((lang) => ({
      url: absoluteUrl(`/${lang}/${path}`),
      alternates: {
        languages: Object.fromEntries(languages.map((l) => [l, absoluteUrl(`/${l}/${path}`)])),
      },
    })),
  )
}
