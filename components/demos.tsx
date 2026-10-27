"use client"

import dynamic from "next/dynamic"

// The demos run TensorFlow.js, the camera and the canvas, which only exist in
// the browser, so each one is loaded on the client only. Each placeholder
// reserves roughly the height its widget takes, so the page does not jump.
function placeholder(heightClass: string) {
  function DemoPlaceholder() {
    return <div className={`mx-auto w-full animate-pulse rounded-lg border bg-muted/40 ${heightClass}`} aria-hidden="true" />
  }
  return DemoPlaceholder
}

export const NumberPredictionDemo = dynamic(
  () => import("@/components/project-contents/number-prediction-content").then((m) => m.NumberPredictionContent),
  { ssr: false, loading: placeholder("h-[26.5rem] max-w-md sm:h-[35rem]") },
)

export const FlowerPredictionDemo = dynamic(
  () => import("@/components/project-contents/flower-prediction-content").then((m) => m.FlowerPredictionContent),
  { ssr: false, loading: placeholder("h-[39rem] max-w-md") },
)

export const BitcoinPredictionDemo = dynamic(
  () => import("@/components/project-contents/bitcoin-prediction-content").then((m) => m.BitcoinPredictionContent),
  { ssr: false, loading: placeholder("h-[55rem]") },
)
