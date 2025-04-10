"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useTranslation } from "@/hooks/use-translation"
import { translations } from "@/config/translations"

export function TextDetectionContent() {
  const [language, setLanguage] = useState<"es" | "en">("es")
  const [image, setImage] = useState<string | null>(null)
  const [detectedText, setDetectedText] = useState<string | null>(null)

  const t = useTranslation(translations, language)

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setImage(e.target?.result as string)
        // Here you would normally send the image to your ML model
        setDetectedText("Sample detected text")
      }
      reader.readAsDataURL(file)
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-8">
        <h1 className="text-3xl md:text-4xl font-bold mb-4">{t.projects.textDetection.title}</h1>
        <p className="text-muted-foreground">{t.projects.textDetection.description}</p>
      </motion.div>

      <div className="max-w-md mx-auto">
        <Card className="border-2">
          <CardHeader>
            <CardTitle className="text-center text-lg">{t.projects.textDetection.uploadPrompt}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {image && (
              <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                <img
                  src={image || "/placeholder.svg"}
                  alt="Uploaded"
                  className="absolute inset-0 w-full h-full object-cover"
                />
              </div>
            )}

            <div>
              <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" id="image-upload" />
              <Button
                variant="outline"
                className="w-full"
                onClick={() => document.getElementById("image-upload")?.click()}
              >
                <Upload className="mr-2 h-4 w-4" />
                {t.projects.textDetection.upload}
              </Button>
            </div>

            {detectedText && (
              <div className="mt-4">
                <h3 className="font-semibold mb-2">{t.projects.textDetection.result}</h3>
                <p className="text-muted-foreground">{detectedText}</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

