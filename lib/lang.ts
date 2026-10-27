import { notFound } from "next/navigation"
import { isLanguage, type Language } from "@/config/translations"

export interface LangParams {
  params: Promise<{ lang: string }>
}

// Reads the [lang] segment; anything other than a known language is a 404.
export async function resolveLang(params: LangParams["params"]): Promise<Language> {
  const { lang } = await params
  if (!isLanguage(lang)) notFound()
  return lang
}
