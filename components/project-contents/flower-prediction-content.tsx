"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { Camera, Upload } from "lucide-react"
import type * as TF from "@tensorflow/tfjs"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { withBasePath } from "@/lib/base-path"

// Output order of the trained model.
const classNames = ["Margarita", "Diente de león", "Rosas", "Girasoles", "Tulipanes"]
const INPUT_SIZE = 180

export function FlowerPredictionContent() {
  const [prediction, setPrediction] = useState<string>("Esperando predicción...")
  const [model, setModel] = useState<TF.LayersModel | null>(null)
  // TensorFlow.js is imported when the demo mounts, not with the page.
  const tfRef = useRef<typeof TF | null>(null)
  const [isModelLoading, setIsModelLoading] = useState(true)
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null)
  const [selectedDevice, setSelectedDevice] = useState<string>("user")

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const processedCanvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let loaded: TF.LayersModel | null = null
    let cancelled = false

    import("@tensorflow/tfjs")
      .then((tf) => {
        tfRef.current = tf
        return tf.loadLayersModel(withBasePath("/assets/models/flowers/model_flowers.json"))
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
        console.error("Error loading model:", error)
        if (!cancelled) setPrediction("No se pudo cargar el modelo. Intenta recargar la página.")
      })
      .finally(() => {
        if (!cancelled) setIsModelLoading(false)
      })

    return () => {
      cancelled = true
      loaded?.dispose()
    }
  }, [])

  // Classifies whatever is currently drawn on the hidden input canvas.
  const classifyCanvas = useCallback(async () => {
    const canvas = canvasRef.current
    const tf = tfRef.current
    if (!model || !canvas || !tf) return

    const output = tf.tidy(() => {
      const input = tf.browser.fromPixels(canvas).toFloat().expandDims(0)
      return model.predict(input) as TF.Tensor
    })

    try {
      const scores = Array.from(await output.data())
      const predictedClass = scores.indexOf(Math.max(...scores))
      setPrediction(classNames[predictedClass])

      const processedCtx = processedCanvasRef.current?.getContext("2d")
      processedCtx?.drawImage(canvas, 0, 0, INPUT_SIZE, INPUT_SIZE)
    } catch (error) {
      console.error("Prediction error:", error)
      setPrediction("Error en la predicción")
    } finally {
      output.dispose()
    }
  }, [model])

  const captureFrame = useCallback(() => {
    const ctx = canvasRef.current?.getContext("2d")
    if (!ctx || !videoRef.current) return
    ctx.drawImage(videoRef.current, 0, 0, INPUT_SIZE, INPUT_SIZE)
    void classifyCanvas()
  }, [classifyCanvas])

  useEffect(() => {
    if (!cameraStream || !model) return
    const interval = setInterval(captureFrame, 2000)
    return () => clearInterval(interval)
  }, [cameraStream, model, captureFrame])

  useEffect(() => {
    return () => cameraStream?.getTracks().forEach((track) => track.stop())
  }, [cameraStream])

  const startCamera = async () => {
    if (!model) return

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: selectedDevice },
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setCameraStream(stream)
    } catch (error) {
      console.error("Error accessing camera:", error)
      alert("No se pudo acceder a la cámara.")
    }
  }

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      canvasRef.current?.getContext("2d")?.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE)
      URL.revokeObjectURL(url)
      void classifyCanvas()
    }
    img.src = url
  }

  return (
    <div className="mx-auto max-w-md">
      <Card className="border-2">
        <CardHeader>
          <CardTitle className="text-center text-lg" aria-live="polite">
            {prediction}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col items-center gap-4">
            <video
              ref={videoRef}
              width={INPUT_SIZE}
              height={INPUT_SIZE}
              autoPlay
              playsInline
              muted
              className="rounded-lg border-2"
            />

            <canvas ref={canvasRef} width={INPUT_SIZE} height={INPUT_SIZE} className="hidden" />

            <div className="flex w-full gap-4">
              <Select value={selectedDevice} onValueChange={setSelectedDevice}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Seleccionar cámara" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="user">Cámara Frontal</SelectItem>
                  <SelectItem value="environment">Cámara Trasera</SelectItem>
                </SelectContent>
              </Select>

              <Button onClick={startCamera} disabled={isModelLoading}>
                <Camera className="mr-2 h-4 w-4" />
                {cameraStream ? "Cambiar Cámara" : "Iniciar Cámara"}
              </Button>
            </div>

            <label className="w-full">
              <input type="file" accept="image/*" onChange={handleImageUpload} className="peer sr-only" />
              <span className="inline-flex h-10 w-full cursor-pointer items-center justify-center gap-2 rounded-md border border-input bg-background px-4 text-sm font-medium hover:bg-accent hover:text-accent-foreground peer-focus-visible:ring-2 peer-focus-visible:ring-ring peer-focus-visible:ring-offset-2">
                <Upload className="h-4 w-4" />
                Subir Imagen
              </span>
            </label>
          </div>

          <div className="text-center">
            <h3 className="mb-2">Imagen Preprocesada</h3>
            <canvas
              ref={processedCanvasRef}
              width={INPUT_SIZE}
              height={INPUT_SIZE}
              className="mx-auto rounded-lg border"
            />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
