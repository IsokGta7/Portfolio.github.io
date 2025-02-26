"use client"

import { useState, useRef, useEffect } from "react"
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

  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadLayersModel('assets/models/numbers/model_numbers.json')
        console.log("Model loaded successfully")
        setModel(loadedModel)
        setIsModelLoading(false)
      } catch (error) {
        console.error("Error loading model:", error)
        setIsModelLoading(false)
      }
    }

    loadModel()
    
    return () => {
      if (model) {
        model.dispose()
      }
    }
  }, [])

  const preprocessCanvas = (canvas: HTMLCanvasElement) => {
    const tmpCanvas = document.createElement('canvas')
    const tmpCtx = tmpCanvas.getContext('2d')
    if (!tmpCtx) return null

    // Match original preprocessing
    tmpCanvas.width = 28
    tmpCanvas.height = 28
    tmpCtx.drawImage(canvas, 0, 0, 28, 28)
    
    // Invert colors (MNIST expects white digits on black background)
    tmpCtx.globalCompositeOperation = 'difference'
    tmpCtx.fillStyle = 'white'
    tmpCtx.fillRect(0, 0, 28, 28)

    const imageData = tmpCtx.getImageData(0, 0, 28, 28)
    const grayscale = []
    
    for (let i = 0; i < imageData.data.length; i += 4) {
      // Normalize and invert
      grayscale.push((255 - imageData.data[i]) / 255)
    }

    return tf.tensor4d(grayscale, [1, 28, 28, 1])
  }

  const handlePredict = async () => {
    if (!canvasRef.current || !model) return

    try {
      const tensor = preprocessCanvas(canvasRef.current)
      if (!tensor) return

      // Add batch dimension and normalize
      const prediction = model.predict(tensor) as tf.Tensor
      const predictionData = await prediction.data()
      const result = predictionData.indexOf(Math.max(...predictionData))
      
      setPrediction(result)
      console.log("Prediction result:", result)
      
      // Cleanup tensors
      tf.dispose([tensor, prediction])
    } catch (error) {
      console.error("Prediction error:", error)
    }
  }

  const handleClear = () => {
    if (!canvasRef.current) return
    const ctx = canvasRef.current.getContext('2d')
    if (!ctx) return
    
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    setPrediction(null)
  }

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
            <DrawingCanvas ref={canvasRef} />

            <div className="flex gap-4 justify-center">
              <Button 
                onClick={handlePredict} 
                className="w-full md:w-auto"
                disabled={isModelLoading}
              >
                <Wand2 className="mr-2 h-4 w-4" />
                {isModelLoading ? "Loading Model..." : "Predict"}
              </Button>
              <Button variant="outline" onClick={handleClear} className="w-full md:w-auto">
                <Eraser className="mr-2 h-4 w-4" />
                Clear
              </Button>
            </div>
          </CardContent>
        </Card>

        {showDetails && <ProjectDetails />}
        
        <div className="mt-8 text-center">
          <Button variant="link" onClick={() => setShowDetails(!showDetails)}>
            {showDetails ? "Hide Technical Details" : "Show Technical Details"}
          </Button>
        </div>
      </div>
    </div>
  )
}