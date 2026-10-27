import { ProjectLayout } from "@/components/layouts/project-layout"
import { DemoHeader } from "@/components/demo-header"
import { PythonProjectsContent } from "@/components/project-contents/python-projects-content"
import { translations } from "@/config/translations"

export default function Page() {
  const t = translations.es.projects.python

  return (
    <ProjectLayout>
      <DemoHeader title={t.title} description={t.description} />
      <PythonProjectsContent />
    </ProjectLayout>
  )
}
