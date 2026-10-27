import { ProjectLayout } from "@/components/layouts/project-layout"
import { DemoHeader } from "@/components/demo-header"
import { CppProjectsContent } from "@/components/project-contents/cpp-projects-content"
import { translations } from "@/config/translations"

export default function Page() {
  const t = translations.es.projects.cpp

  return (
    <ProjectLayout>
      <DemoHeader title={t.title} description={t.description} />
      <CppProjectsContent />
    </ProjectLayout>
  )
}
