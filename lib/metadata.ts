import type { Metadata } from "next"
import { languages, translations, type Language } from "@/config/translations"
import { absoluteUrl } from "@/lib/site"

const ogLocale: Record<Language, string> = { es: "es", en: "en" }

/**
 * Metadata for one page in one language. `path` is the part after the language
 * segment, with a trailing slash ("" for the home page, "number-prediction/").
 */
export function pageMetadata(lang: Language, path: string, title?: string, description?: string): Metadata {
  const { meta } = translations[lang]
  const url = absoluteUrl(`/${lang}/${path}`)
  const fullTitle = title ? `${title} · ${meta.siteName}` : meta.title
  const desc = description ?? meta.description

  return {
    title: title ?? { absolute: meta.title },
    description: desc,
    alternates: {
      canonical: url,
      languages: {
        ...Object.fromEntries(languages.map((l) => [l, absoluteUrl(`/${l}/${path}`)])),
        "x-default": absoluteUrl(`/es/${path}`),
      },
    },
    openGraph: {
      type: "website",
      url,
      siteName: meta.siteName,
      title: fullTitle,
      description: desc,
      locale: ogLocale[lang],
      alternateLocale: languages.filter((l) => l !== lang).map((l) => ogLocale[l]),
    },
    twitter: {
      card: "summary",
      title: fullTitle,
      description: desc,
    },
  }
}
