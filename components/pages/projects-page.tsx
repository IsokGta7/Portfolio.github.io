"use client"

import { useState } from "react"
import { ProjectLayout } from "@/components/layouts/project-layout"
import { ProjectSection } from "@/components/sections/project-section"
import { useTranslation } from "@/hooks/use-translation"
import { translations } from "@/config/translations"

export function ProjectsPage() {
  const [language, setLanguage] = useState<"es" | "en">("es")
  const t = useTranslation(translations[language], language)

  return (
    <ProjectLayout language={language} onLanguageChange={setLanguage}>
      {/* AI Section */}
      <ProjectSection
        id="ai"
        title={t?.sections?.ai?.title ?? ""}
        subtitle={t?.sections?.ai?.subtitle ?? ""}
        projects={t?.sections?.ai?.projects ?? []}
        language={language}
      />

      {/* Programming Section */}
      <ProjectSection
        id="programming"
        title={t?.sections?.programming?.title ?? ""}
        subtitle={t?.sections?.programming?.subtitle ?? ""}
        skills={t?.sections?.programming?.skills ?? []}
        language={language}
      />

      {/* Systems Design Section */}
      <ProjectSection
        id="systems"
        title={t?.sections?.systems?.title ?? ""}
        subtitle={t?.sections?.systems?.subtitle ?? ""}
        skills={t?.sections?.systems?.skills ?? []}
        language={language}
      />

      {/* Posts Section */}
      <ProjectSection
        id="posts"
        title={t?.sections?.posts?.title ?? ""}
        subtitle={t?.sections?.posts?.subtitle ?? ""}
        language={language}
      />
    </ProjectLayout>
  )
}

