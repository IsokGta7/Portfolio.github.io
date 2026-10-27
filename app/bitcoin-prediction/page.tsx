import { ProjectLayout } from "@/components/layouts/project-layout"
import { DemoHeader } from "@/components/demo-header"
import { BitcoinPredictionDemo } from "@/components/demos"
import { translations } from "@/config/translations"

export default function Page() {
  const t = translations.es.projects.bitcoinPrediction

  return (
    <ProjectLayout>
      <DemoHeader title={t.title} description={t.description} />
      <BitcoinPredictionDemo />
    </ProjectLayout>
  )
}
