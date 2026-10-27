import { ProjectLayout } from "@/components/layouts/project-layout"
import { DemoHeader } from "@/components/demo-header"
import { FlowerPredictionDemo } from "@/components/demos"

export default function Page() {
  return (
    <ProjectLayout>
      <DemoHeader title="Clasificador de Flores" description="Clasificación en tiempo real usando cámara o imágenes" />
      <FlowerPredictionDemo />
    </ProjectLayout>
  )
}
