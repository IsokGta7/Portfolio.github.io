"use client"

import dynamic from "next/dynamic"

// The demos run TensorFlow.js, the camera and the canvas, which only exist in
// the browser, so each one is loaded on the client only. The placeholder keeps
// the space the widget will take so the page does not jump when it mounts.
function DemoPlaceholder() {
  return <div className="mx-auto h-[28rem] max-w-md animate-pulse rounded-lg border bg-muted/40" aria-hidden="true" />
}

export const NumberPredictionDemo = dynamic(
  () => import("@/components/project-contents/number-prediction-content").then((m) => m.NumberPredictionContent),
  { ssr: false, loading: DemoPlaceholder },
)

export const FlowerPredictionDemo = dynamic(
  () => import("@/components/project-contents/flower-prediction-content").then((m) => m.FlowerPredictionContent),
  { ssr: false, loading: DemoPlaceholder },
)

export const BitcoinPredictionDemo = dynamic(
  () => import("@/components/project-contents/bitcoin-prediction-content").then((m) => m.BitcoinPredictionContent),
  { ssr: false, loading: DemoPlaceholder },
)

export const TextDetectionDemo = dynamic(
  () => import("@/components/project-contents/text-detection-content").then((m) => m.TextDetectionContent),
  { ssr: false, loading: DemoPlaceholder },
)
