import { ProjectLayout } from "@/components/layouts/project-layout"
import { ProjectSection } from "@/components/sections/project-section"
import { translations } from "@/config/translations"
import { resolveLang, type LangParams } from "@/lib/lang"

export default async function Home({ params }: LangParams) {
  const lang = await resolveLang(params)
  const { sections } = translations[lang]

  return (
    <ProjectLayout lang={lang}>
      <ProjectSection
        id="ai"
        title={sections.ai.title}
        subtitle={sections.ai.subtitle}
        projects={sections.ai.projects}
        language={lang}
      />
      <ProjectSection
        id="programming"
        title={sections.programming.title}
        subtitle={sections.programming.subtitle}
        skills={sections.programming.skills}
        language={lang}
      />
      <ProjectSection
        id="systems"
        title={sections.systems.title}
        subtitle={sections.systems.subtitle}
        skills={sections.systems.skills}
        language={lang}
      />
      <ProjectSection id="posts" title={sections.posts.title} subtitle={sections.posts.subtitle} language={lang} />
    </ProjectLayout>
  )
}
