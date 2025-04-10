"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { motion } from "framer-motion"
import { Camera, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import * as tf from "@tensorflow/tfjs"

const classNames = ['Margarita', 'Diente de león', 'Rosas', 'Girasoles', 'Tulipanes']

export function FlowerPredictionContent() {
  const [prediction, setPrediction] = useState<string>("Esperando predicción...")
  const [model, setModel] = useState<tf.LayersModel | null>(null)
  const [isModelLoading, setIsModelLoading] = useState(true)
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null)
  const [selectedDevice, setSelectedDevice] = useState<string>("user")
  
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const processedCanvasRef = useRef<HTMLCanvasElement>(null)
  const predictionInterval = useRef<NodeJS.Timeout | null>(null)

  // Load model
  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadLayersModel('/assets/models/flowers/model_flowers.json')
        setModel(loadedModel)
        setIsModelLoading(false)
      } catch (error) {
        console.error("Error loading model:", error)
        setIsModelLoading(false)
      }
    }

    loadModel()
    return () => model?.dispose()
  }, [])

  // Prediction logic with useCallback
  const predict = useCallback(async () => {
    if (!model || !videoRef.current || !canvasRef.current) return

    const ctx = canvasRef.current.getContext('2d')
    if (!ctx) return

    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height)
    const imgTensor = tf.browser.fromPixels(canvasRef.current).expandDims(0)

    try {
      const predictions = await model.predict(imgTensor).data()
      const predictedClass = predictions.indexOf(Math.max(...predictions))
      setPrediction(classNames[predictedClass])
      
      // Update processed canvas
      if (processedCanvasRef.current) {
        const processedCtx = processedCanvasRef.current.getContext('2d')
        processedCtx?.drawImage(videoRef.current!, 0, 0, processedCanvasRef.current.width, processedCanvasRef.current.height)
      }

      imgTensor.dispose()
    } catch (error) {
      console.error("Prediction error:", error)
      setPrediction("Error en la predicción")
    }
  }, [model])

  // Camera handling and prediction interval
  useEffect(() => {
    if (cameraStream && model) {
      predictionInterval.current = setInterval(predict, 2000)
    }
    
    return () => {
      if (predictionInterval.current) {
        clearInterval(predictionInterval.current)
      }
    }
  }, [cameraStream, model, predict])

  // Start/stop camera
  const startCamera = async () => {
    if (!model) return

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: selectedDevice }
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.style.display = "block"
      }
      setCameraStream(stream)
    } catch (error) {
      console.error("Error accessing camera:", error)
      alert("No se pudo acceder a la cámara.")
    }
  }

  // Handle image upload
  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file || !canvasRef.current) return

    const reader = new FileReader()
    reader.onload = async (e) => {
      const img = new Image()
      img.src = e.target?.result as string
      
      img.onload = () => {
        const ctx = canvasRef.current?.getContext('2d')
        ctx?.drawImage(img, 0, 0, canvasRef.current!.width, canvasRef.current!.height)
        predict()
      }
    }
    reader.readAsDataURL(file)
  }

  // Cleanup effects
  useEffect(() => {
    return () => {
      cameraStream?.getTracks().forEach(track => track.stop())
      if (predictionInterval.current) {
        clearInterval(predictionInterval.current)
      }
    }
  }, [cameraStream])

  return (
    <div className="container mx-auto px-4 py-8">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-8">
        <h1 className="text-3xl md:text-4xl font-bold mb-4">Clasificador de Flores</h1>
        <p className="text-muted-foreground">
          Clasificación en tiempo real usando cámara o imágenes
        </p>
      </motion.div>

      <div className="max-w-md mx-auto">
        <Card className="border-2">
          <CardHeader>
            <CardTitle className="text-center text-lg">
              {prediction}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-col gap-4 items-center">
              <video 
                ref={videoRef}
                width="180" 
                height="180" 
                autoPlay
                className="rounded-lg border-2"
              />
              
              <canvas 
                ref={canvasRef}
                width="180" 
                height="180" 
                className="hidden"
              />
              
              <div className="flex gap-4 w-full">
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

              <div className="relative w-full">
                <input 
                  type="file" 
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                <Button variant="outline" className="w-full">
                  <Upload className="mr-2 h-4 w-4" />
                  Subir Imagen
                </Button>
              </div>
            </div>

            <div className="text-center">
              <h3 className="mb-2">Imagen Preprocesada</h3>
              <canvas 
                ref={processedCanvasRef}
                width="180" 
                height="180" 
                className="border rounded-lg mx-auto"
              />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}