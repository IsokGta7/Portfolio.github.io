import type { Metadata } from "next"
import { DemoHeader } from "@/components/demo-header"
import { NumberPredictionDemo } from "@/components/demos"
import { SiteLayout } from "@/components/site-layout"
import { translations } from "@/config/translations"
import { resolveLang, type LangParams } from "@/lib/lang"
import { pageMetadata } from "@/lib/metadata"

export async function generateMetadata({ params }: LangParams): Promise<Metadata> {
  const lang = await resolveLang(params)
  const t = translations[lang].demoPages.numberPrediction
  return pageMetadata(lang, "number-prediction/", t.title, t.description)
}

export default async function Page({ params }: LangParams) {
  const lang = await resolveLang(params)
  const t = translations[lang].demoPages.numberPrediction

  return (
    <SiteLayout lang={lang}>
      <div className="container py-12 sm:py-16">
        <DemoHeader lang={lang} title={t.title} description={t.description} />
        <NumberPredictionDemo lang={lang} />
      </div>
    </SiteLayout>
  )
}
