import Link from "next/link"
import { ArrowLeft } from "lucide-react"
import { translations, type Language } from "@/config/translations"

interface DemoHeaderProps {
  lang: Language
  title: string
  description: string
}

export function DemoHeader({ lang, title, description }: DemoHeaderProps) {
  return (
    <header className="mb-8 sm:mb-12">
      <Link
        href={`/${lang}/#demos`}
        className="inline-flex items-center gap-1.5 rounded-sm text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
      >
        <ArrowLeft className="h-4 w-4" aria-hidden="true" />
        {translations[lang].demos.backToDemos}
      </Link>
      <h1 className="mt-4 text-3xl font-bold tracking-tight sm:text-4xl">{title}</h1>
      <p className="mt-3 max-w-2xl text-base text-muted-foreground">{description}</p>
    </header>
  )
}
