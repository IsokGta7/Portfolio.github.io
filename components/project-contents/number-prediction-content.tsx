"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { Eraser, Wand2 } from "lucide-react"
import type * as TF from "@tensorflow/tfjs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { translations, type Language } from "@/config/translations"
import { withBasePath } from "@/lib/base-path"

type PointerLikeEvent = React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>

export function NumberPredictionContent({ lang }: { lang: Language }) {
  const t = translations[lang].projects.numberPrediction
  const [prediction, setPrediction] = useState<number | null>(null)
  const [model, setModel] = useState<TF.LayersModel | null>(null)
  const [isModelLoading, setIsModelLoading] = useState(true)
  const [modelFailed, setModelFailed] = useState(false)
  // TensorFlow.js is imported when the demo mounts, not with the page.
  const tfRef = useRef<typeof TF | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const isDrawing = useRef(false)
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    ctxRef.current = ctx
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    // Drawing coordinates stay in CSS pixels; the scale maps them to device pixels.
    ctx.scale(dpr, dpr)

    ctx.fillStyle = "black"
    ctx.fillRect(0, 0, rect.width, rect.height)
    ctx.lineWidth = 8
    ctx.lineCap = "round"
    ctx.strokeStyle = "white"
  }, [])

  useEffect(() => {
    let loaded: TF.LayersModel | null = null
    let cancelled = false

    import("@tensorflow/tfjs")
      .then((tf) => {
        tfRef.current = tf
        return tf.loadLayersModel(withBasePath("/assets/models/numbers/model_numbers.json"))
      })
      .then((m) => {
        if (cancelled) {
          m.dispose()
          return
        }
        loaded = m
        setModel(m)
      })
      .catch((error: unknown) => {
        console.error("Model loading failed:", error)
        if (!cancelled) setModelFailed(true)
      })
      .finally(() => {
        if (!cancelled) setIsModelLoading(false)
      })

    return () => {
      cancelled = true
      loaded?.dispose()
    }
  }, [])

  const getCanvasCoordinates = (e: PointerLikeEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const point = "touches" in e ? e.touches[0] : e
    return { x: point.clientX - rect.left, y: point.clientY - rect.top }
  }

  const startDrawing = (e: PointerLikeEvent) => {
    if (!ctxRef.current) return
    isDrawing.current = true
    const { x, y } = getCanvasCoordinates(e)
    ctxRef.current.beginPath()
    ctxRef.current.moveTo(x, y)
  }

  const draw = (e: PointerLikeEvent) => {
    if (!isDrawing.current || !ctxRef.current) return
    const { x, y } = getCanvasCoordinates(e)
    ctxRef.current.lineTo(x, y)
    ctxRef.current.stroke()
  }

  const stopDrawing = () => {
    isDrawing.current = false
  }

  // Downscales the drawing to the 28x28 grayscale input MNIST models expect,
  // white strokes on black, values normalised to 0..1.
  const preprocessCanvas = useCallback(() => {
    const canvas = canvasRef.current!
    const tempCanvas = document.createElement("canvas")
    const tempCtx = tempCanvas.getContext("2d")!
    tempCanvas.width = 28
    tempCanvas.height = 28
    tempCtx.drawImage(canvas, 0, 0, 28, 28)

    const { data } = tempCtx.getImageData(0, 0, 28, 28)
    const pixels: number[] = []
    for (let i = 0; i < data.length; i += 4) {
      pixels.push(data[i] / 255)
    }
    return tfRef.current!.tensor4d(pixels, [1, 28, 28, 1])
  }, [])

  const handlePredict = useCallback(async () => {
    const tf = tfRef.current
    if (!model || !tf) return
    const input = preprocessCanvas()
    const output = model.predict(input) as TF.Tensor
    const best = output.argMax(1)
    try {
      const [result] = await best.data()
      setPrediction(result)
    } catch (error) {
      console.error("Prediction failed:", error)
    } finally {
      tf.dispose([input, output, best])
    }
  }, [model, preprocessCanvas])

  const handleClear = () => {
    const canvas = canvasRef.current
    if (!ctxRef.current || !canvas) return
    const rect = canvas.getBoundingClientRect()
    ctxRef.current.fillStyle = "black"
    ctxRef.current.fillRect(0, 0, rect.width, rect.height)
    setPrediction(null)
  }

  return (
    <div className="mx-auto max-w-md">
      <Card className="border-2">
        <CardHeader>
          <CardTitle className="min-h-7 text-center text-lg" aria-live="polite">
            {prediction !== null ? `${t.result} ${prediction}` : ""}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="relative aspect-square rounded-lg bg-black">
            <canvas
              ref={canvasRef}
              className="h-full w-full touch-none"
              role="img"
              aria-label={t.canvasLabel}
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
              onTouchStart={startDrawing}
              onTouchMove={draw}
              onTouchEnd={stopDrawing}
            />
          </div>

          <div className="flex justify-center gap-4">
            <Button onClick={handlePredict} className="w-full md:w-auto" disabled={!model}>
              <Wand2 className="mr-2 h-4 w-4" />
              {isModelLoading ? t.loading : t.predict}
            </Button>
            <Button variant="outline" onClick={handleClear} className="w-full md:w-auto">
              <Eraser className="mr-2 h-4 w-4" />
              {t.clear}
            </Button>
          </div>

          {modelFailed && (
            <p role="alert" className="text-center text-sm text-destructive">
              {t.modelError}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
