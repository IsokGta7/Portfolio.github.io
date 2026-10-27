"use client"

import { useState } from "react"
import { ProjectLayout } from "@/components/layouts/project-layout"
import { ProjectSection } from "@/components/sections/project-section"
import { translations, type Language } from "@/config/translations"

export function ProjectsPage() {
  const [language, setLanguage] = useState<Language>("es")
  const { sections } = translations[language]

  return (
    <ProjectLayout language={language} onLanguageChange={setLanguage}>
      <ProjectSection
        id="ai"
        title={sections.ai.title}
        subtitle={sections.ai.subtitle}
        projects={sections.ai.projects}
        language={language}
      />
      <ProjectSection
        id="programming"
        title={sections.programming.title}
        subtitle={sections.programming.subtitle}
        skills={sections.programming.skills}
        language={language}
      />
      <ProjectSection
        id="systems"
        title={sections.systems.title}
        subtitle={sections.systems.subtitle}
        skills={sections.systems.skills}
        language={language}
      />
      <ProjectSection id="posts" title={sections.posts.title} subtitle={sections.posts.subtitle} language={language} />
    </ProjectLayout>
  )
}
