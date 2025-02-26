"use client"

import { useState, useRef, useEffect } from "react"
import { Home, Camera, Upload } from "lucide-react"
import { motion } from "framer-motion"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

const translations = {
  es: {
    title: "Clasificador de Flores en Tiempo Real",
    useCamera: "Usar Cámara",
    frontCamera: "Cámara Frontal",
    backCamera: "Cámara Trasera",
    chooseFile: "Seleccionar Archivo",
    prediction: "Predicción:",
    preprocessed: "Imagen Preprocesada",
    home: "Inicio",
    or: "o",
    noCamera: "No se puede acceder a la cámara",
    startCamera: "Iniciar Cámara",
    stopCamera: "Detener Cámara",
  },
  en: {
    title: "Real-time Flower Classifier",
    useCamera: "Use Camera",
    frontCamera: "Front Camera",
    backCamera: "Back Camera",
    chooseFile: "Choose File",
    prediction: "Prediction:",
    preprocessed: "Preprocessed Image",
    home: "Home",
    or: "or",
    noCamera: "Cannot access camera",
    startCamera: "Start Camera",
    stopCamera: "Stop Camera",
  },
}

export default function FlowerClassifierPage() {
  const [prediction, setPrediction] = useState<string | null>(null)
  const [language, setLanguage] = useState<"es" | "en">("es")
  const [isDarkMode, setIsDarkMode] = useState(true)
  const [isCameraActive, setIsCameraActive] = useState(false)
  const [selectedCamera, setSelectedCamera] = useState<string>("user")
  const [hasCamera, setHasCamera] = useState(true)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const t = translations[language]

  useEffect(() => {
    // Set dark mode by default
    document.documentElement.classList.add("dark")

    // Check if camera is available
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then(() => setHasCamera(true))
      .catch(() => setHasCamera(false))
  }, [])

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: selectedCamera },
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsCameraActive(true)
        // Start prediction loop
        predictFromVideo()
      }
    } catch (err) {
      console.error("Error accessing camera:", err)
      setHasCamera(false)
    }
  }

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
      tracks.forEach((track) => track.stop())
      videoRef.current.srcObject = null
      setIsCameraActive(false)
    }
  }

  const predictFromVideo = () => {
    if (!isCameraActive) return

    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    const context = canvas.getContext("2d")
    if (!context) return

    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Get image data for prediction
    const imageData = canvas.toDataURL("image/jpeg")

    // Here you would send the image data to your ML model
    // For demo, we'll just rotate between some flower types
    const flowers = ["Tulipanes", "Rosas", "Margaritas", "Girasoles"]
    setPrediction(flowers[Math.floor(Math.random() * flowers.length)])

    // Continue prediction loop
    requestAnimationFrame(predictFromVideo)
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      const img = new Image()
      img.onload = () => {
        const canvas = canvasRef.current
        if (!canvas) return

        const context = canvas.getContext("2d")
        if (!context) return

        // Draw uploaded image to canvas
        context.drawImage(img, 0, 0, canvas.width, canvas.height)

        // Here you would send the image data to your ML model
        // For demo, we'll just use a random prediction
        const flowers = ["Tulipanes", "Rosas", "Margaritas", "Girasoles"]
        setPrediction(flowers[Math.floor(Math.random() * flowers.length)])
      }
      img.src = e.target?.result as string
    }
    reader.readAsDataURL(file)
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4">
          <div className="flex h-14 items-center justify-between">
            <Button variant="ghost" size="icon" asChild>
              <a href="/">
                <Home className="h-5 w-5" />
                <span className="sr-only">{t.home}</span>
              </a>
            </Button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 md:py-12">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold mb-4">{t.title}</h1>
        </motion.div>

        <div className="max-w-md mx-auto">
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="text-center text-lg">{prediction && `${t.prediction} ${prediction}`}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Camera/Canvas Display */}
              <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`absolute inset-0 w-full h-full object-cover ${isCameraActive ? "block" : "hidden"}`}
                />
                <canvas
                  ref={canvasRef}
                  width={640}
                  height={480}
                  className="absolute inset-0 w-full h-full object-cover"
                />
              </div>

              {/* Controls */}
              <div className="space-y-4">
                {hasCamera && (
                  <div className="flex flex-col gap-2">
                    <Select value={selectedCamera} onValueChange={setSelectedCamera}>
                      <SelectTrigger>
                        <SelectValue placeholder={t.frontCamera} />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="user">{t.frontCamera}</SelectItem>
                        <SelectItem value="environment">{t.backCamera}</SelectItem>
                      </SelectContent>
                    </Select>
                    <Button onClick={isCameraActive ? stopCamera : startCamera} className="w-full">
                      {isCameraActive ? (
                        <>
                          <Camera className="mr-2 h-4 w-4" />
                          {t.stopCamera}
                        </>
                      ) : (
                        <>
                          <Camera className="mr-2 h-4 w-4" />
                          {t.startCamera}
                        </>
                      )}
                    </Button>
                  </div>
                )}

                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <span className="w-full border-t" />
                  </div>
                  <div className="relative flex justify-center text-xs uppercase">
                    <span className="bg-background px-2 text-muted-foreground">{t.or}</span>
                  </div>
                </div>

                <div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  <Button variant="outline" className="w-full" onClick={() => fileInputRef.current?.click()}>
                    <Upload className="mr-2 h-4 w-4" />
                    {t.chooseFile}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}

