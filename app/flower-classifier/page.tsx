import { ProjectLayout } from "@/components/layouts/project-layout"
import { FlowerPredictionContent } from "@/components/project-contents/flower-prediction-content"

// Rename the page component
export default function FlowerClassifierPage() {
  return (
    <ProjectLayout>
      <FlowerPredictionContent />
    </ProjectLayout>
  )
}