import { ProjectLayout } from "@/components/layouts/project-layout"
import { DemoHeader } from "@/components/demo-header"
import { TextDetectionDemo } from "@/components/demos"
import { translations } from "@/config/translations"

export default function Page() {
  const t = translations.es.projects.textDetection

  return (
    <ProjectLayout>
      <DemoHeader title={t.title} description={t.description} />
      <TextDetectionDemo />
    </ProjectLayout>
  )
}
