import { ProjectLayout } from "@/components/layouts/project-layout"
import { DemoHeader } from "@/components/demo-header"
import { NumberPredictionDemo } from "@/components/demos"
import { translations } from "@/config/translations"

export default function Page() {
  const t = translations.es.projects.numberPrediction

  return (
    <ProjectLayout>
      <DemoHeader title={t.title} description={t.description} />
      <NumberPredictionDemo />
    </ProjectLayout>
  )
}
