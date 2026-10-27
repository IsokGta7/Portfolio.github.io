"use client"

import { useState } from "react"
import { Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { translations } from "@/config/translations"

export function TextDetectionContent() {
  const [image, setImage] = useState<string | null>(null)
  const [detectedText, setDetectedText] = useState<string | null>(null)
  const t = translations.es.projects.textDetection

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setImage(typeof e.target?.result === "string" ? e.target.result : null)
        // Here you would normally send the image to your ML model
        setDetectedText("Sample detected text")
      }
      reader.readAsDataURL(file)
    }
  }

  return (
    <div className="mx-auto max-w-md">
      <Card className="border-2">
        <CardHeader>
          <CardTitle className="text-center text-lg">{t.uploadPrompt}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {image && (
            <div className="relative aspect-video overflow-hidden rounded-lg bg-black">
              {/* eslint-disable-next-line @next/next/no-img-element -- a local data URL, nothing to optimise */}
              <img src={image} alt="" className="absolute inset-0 h-full w-full object-cover" />
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
              {t.upload}
            </Button>
          </div>

          {detectedText && (
            <div className="mt-4">
              <h3 className="mb-2 font-semibold">{t.result}</h3>
              <p className="text-muted-foreground">{detectedText}</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
