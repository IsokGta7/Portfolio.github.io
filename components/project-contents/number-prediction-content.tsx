"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { motion } from "framer-motion"
import { Eraser, Wand2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { DrawingCanvas } from "@/components/ui/drawing-canvas"
import { ProjectDetails } from "@/components/project-details"
import * as tf from "@tensorflow/tfjs"

export function NumberPredictionContent() {
  const [prediction, setPrediction] = useState<number | null>(null)
  const [model, setModel] = useState<tf.LayersModel | null>(null)
  const [isModelLoading, setIsModelLoading] = useState(true)
  const [showDetails, setShowDetails] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const isDrawing = useRef(false)
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null)

  // Initialize canvas context
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctxRef.current = ctx
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)
    
    // Set initial canvas state
    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.lineWidth = 8
    ctx.lineCap = 'round'
    ctx.strokeStyle = 'white'
  }, [])

  // Model loading
  useEffect(() => {
    const loadModel = async () => {
      try {
        const model = await tf.loadLayersModel('/assets/models/numbers/model_numbers.json')
        setModel(model)
        setIsModelLoading(false)
        console.log('Model loaded, test prediction:', await model.predict(tf.zeros([1, 28, 28, 1])).data())
      } catch (error) {
        console.error("Model loading failed:", error)
        setIsModelLoading(false)
      }
    }

    loadModel()
    return () => model?.dispose()
  }, [])

  // Drawing handlers
  const startDrawing = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    if (!ctxRef.current) return
    
    isDrawing.current = true
    const { offsetX, offsetY } = getCanvasCoordinates(e)
    ctxRef.current.beginPath()
    ctxRef.current.moveTo(offsetX, offsetY)
  }, [])

  const draw = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing.current || !ctxRef.current) return
    
    const { offsetX, offsetY } = getCanvasCoordinates(e)
    ctxRef.current.lineTo(offsetX, offsetY)
    ctxRef.current.stroke()
  }, [])

  const stopDrawing = useCallback(() => {
    isDrawing.current = false
  }, [])

  // Coordinate calculation
  const getCanvasCoordinates = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    
    if (e.nativeEvent instanceof TouchEvent) {
      return {
        offsetX: (e.nativeEvent.touches[0].clientX - rect.left) * (canvas.width / rect.width),
        offsetY: (e.nativeEvent.touches[0].clientY - rect.top) * (canvas.height / rect.height)
      }
    }
    
    return {
      offsetX: (e.nativeEvent.clientX - rect.left) * (canvas.width / rect.width),
      offsetY: (e.nativeEvent.clientY - rect.top) * (canvas.height / rect.height)
    }
  }

  // Preprocessing
  const preprocessCanvas = useCallback(() => {
    const canvas = canvasRef.current!
    const tempCanvas = document.createElement('canvas')
    const tempCtx = tempCanvas.getContext('2d')!
    
    tempCanvas.width = 28
    tempCanvas.height = 28
    tempCtx.drawImage(canvas, 0, 0, 28, 28)
    
    // Invert colors (MNIST format)
    tempCtx.globalCompositeOperation = 'difference'
    tempCtx.fillStyle = 'white'
    tempCtx.fillRect(0, 0, 28, 28)
    
    const imageData = tempCtx.getImageData(0, 0, 28, 28)
    const pixels = []
    
    for (let i = 0; i < imageData.data.length; i += 4) {
      pixels.push((255 - imageData.data[i]) / 255) // Invert and normalize
    }

    return tf.tensor4d(pixels, [1, 28, 28, 1])
  }, [])

  // Prediction handler
  const handlePredict = useCallback(async () => {
    if (!model) return
    
    try {
      const tensor = preprocessCanvas()
      const prediction = model.predict(tensor) as tf.Tensor
      const [result] = await prediction.argMax(1).data()
      
      setPrediction(result)
      console.log('Prediction result:', result)
      
      tensor.dispose()
      prediction.dispose()
    } catch (error) {
      console.error('Prediction failed:', error)
    }
  }, [model, preprocessCanvas])

  // Clear handler
  const handleClear = useCallback(() => {
    if (!ctxRef.current) return
    
    ctxRef.current.fillStyle = 'black'
    ctxRef.current.fillRect(0, 0, ctxRef.current.canvas.width, ctxRef.current.canvas.height)
    setPrediction(null)
  }, [])

  return (
    <div className="container mx-auto px-4 py-8">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-8">
        <h1 className="text-3xl md:text-4xl font-bold mb-4">Number Recognition</h1>
        <p className="text-muted-foreground">
          Draw a number and click predict to see the AI recognition
        </p>
      </motion.div>

      <div className="max-w-md mx-auto">
        <Card className="border-2">
          <CardHeader>
            <CardTitle className="text-center text-lg">
              {prediction !== null ? `Predicted Number: ${prediction}` : " "}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="relative aspect-square bg-black rounded-lg">
              <canvas
                ref={canvasRef}
                className="w-full h-full touch-none"
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
                onTouchStart={startDrawing}
                onTouchMove={draw}
                onTouchEnd={stopDrawing}
              />
            </div>

            <div className="flex gap-4 justify-center">
              <Button 
                onClick={handlePredict} 
                className="w-full md:w-auto"
                disabled={!model}
              >
                <Wand2 className="mr-2 h-4 w-4" />
                {isModelLoading ? "Loading Model..." : "Predict"}
              </Button>
              <Button 
                variant="outline" 
                onClick={handleClear} 
                className="w-full md:w-auto"
              >
                <Eraser className="mr-2 h-4 w-4" />
                Clear
              </Button>
            </div>
          </CardContent>
        </Card>

        {showDetails && <ProjectDetails />}
        
        <div className="mt-8 text-center">
          <Button variant="link" onClick={() => setShowDetails(!showDetails)}>
            {showDetails ? "Hide Details" : "Show Details"}
          </Button>
        </div>
      </div>
    </div>
  )
}