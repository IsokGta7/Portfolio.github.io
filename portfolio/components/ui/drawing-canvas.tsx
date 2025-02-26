"use client"

import { forwardRef, useEffect, useRef } from "react"

interface DrawingCanvasProps {
  width?: number
  height?: number
}

export const DrawingCanvas = forwardRef<HTMLCanvasElement, DrawingCanvasProps>((props, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas size
    canvas.width = props.width || 280
    canvas.height = props.height || 280

    // Set drawing styles
    ctx.strokeStyle = "white"
    ctx.lineWidth = 20
    ctx.lineCap = "round"
    ctx.lineJoin = "round"

    let isDrawing = false
    let lastX = 0
    let lastY = 0

    const draw = (e: MouseEvent | TouchEvent) => {
      if (!isDrawing || !canvas) return

      const rect = canvas.getBoundingClientRect()
      const scaleX = canvas.width / rect.width
      const scaleY = canvas.height / rect.height

      let currentX: number
      let currentY: number

      if (e instanceof MouseEvent) {
        currentX = (e.clientX - rect.left) * scaleX
        currentY = (e.clientY - rect.top) * scaleY
      } else {
        currentX = (e.touches[0].clientX - rect.left) * scaleX
        currentY = (e.touches[0].clientY - rect.top) * scaleY
      }

      ctx.beginPath()
      ctx.moveTo(lastX, lastY)
      ctx.lineTo(currentX, currentY)
      ctx.stroke()

      lastX = currentX
      lastY = currentY
    }

    const startDrawing = (e: MouseEvent | TouchEvent) => {
      if (!canvas) return
      isDrawing = true
      const rect = canvas.getBoundingClientRect()
      const scaleX = canvas.width / rect.width
      const scaleY = canvas.height / rect.height

      if (e instanceof MouseEvent) {
        lastX = (e.clientX - rect.left) * scaleX
        lastY = (e.clientY - rect.top) * scaleY
      } else {
        lastX = (e.touches[0].clientX - rect.left) * scaleX
        lastY = (e.touches[0].clientY - rect.top) * scaleY
      }
    }

    const stopDrawing = () => {
      isDrawing = false
    }

    // Mouse Events
    canvas.addEventListener("mousedown", startDrawing)
    canvas.addEventListener("mousemove", draw)
    canvas.addEventListener("mouseup", stopDrawing)
    canvas.addEventListener("mouseout", stopDrawing)

    // Touch Events
    canvas.addEventListener("touchstart", (e: TouchEvent) => {
      e.preventDefault()
      startDrawing(e)
    })
    canvas.addEventListener("touchmove", (e: TouchEvent) => {
      e.preventDefault()
      draw(e)
    })
    canvas.addEventListener("touchend", (e: TouchEvent) => {
      e.preventDefault()
      stopDrawing()
    })

    return () => {
      if (!canvas) return

      // Cleanup Mouse Events
      canvas.removeEventListener("mousedown", startDrawing)
      canvas.removeEventListener("mousemove", draw)
      canvas.removeEventListener("mouseup", stopDrawing)
      canvas.removeEventListener("mouseout", stopDrawing)

      // Cleanup Touch Events
      canvas.removeEventListener("touchstart", startDrawing)
      canvas.removeEventListener("touchmove", draw)
      canvas.removeEventListener("touchend", stopDrawing)
    }
  }, [props.width, props.height])

  return (
    <div className="relative aspect-square bg-black rounded-lg overflow-hidden">
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full touch-none" {...props} />
    </div>
  )
})

DrawingCanvas.displayName = "DrawingCanvas"

