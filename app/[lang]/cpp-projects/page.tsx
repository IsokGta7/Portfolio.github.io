import type { Metadata } from "next"
import { ProjectLayout } from "@/components/layouts/project-layout"
import { DemoHeader } from "@/components/demo-header"
import { CppProjectsContent } from "@/components/project-contents/cpp-projects-content"
import { translations } from "@/config/translations"
import { resolveLang, type LangParams } from "@/lib/lang"
import { pageMetadata } from "@/lib/metadata"

export async function generateMetadata({ params }: LangParams): Promise<Metadata> {
  const lang = await resolveLang(params)
  const t = translations[lang].projects.cpp
  return pageMetadata(lang, "cpp-projects/", t.title, t.description)
}

export default async function Page({ params }: LangParams) {
  const lang = await resolveLang(params)
  const t = translations[lang].projects.cpp

  return (
    <ProjectLayout lang={lang}>
      <DemoHeader title={t.title} description={t.description} />
      <CppProjectsContent lang={lang} />
    </ProjectLayout>
  )
}
