import type { DemoKey } from "@/config/translations"

// Route slug of each demo page, in the order they appear on the home page.
export const demoRoutes: { key: DemoKey; slug: string }[] = [
  { key: "numberPrediction", slug: "number-prediction" },
  { key: "flowerClassifier", slug: "flower-classifier" },
  { key: "bitcoinPrediction", slug: "bitcoin-prediction" },
]
