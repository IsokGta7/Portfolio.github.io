"use client"

import { forwardRef, useEffect, type RefObject } from "react"

export const DrawingCanvas = forwardRef<HTMLCanvasElement>((props, ref) => {
  useEffect(() => {
    const canvas = ref as React.MutableRefObject<HTMLCanvasElement>
    if (!canvas.current) return

    const ctx = canvas.current.getContext("2d")
    if (!ctx) return

    // Set canvas size
    canvas.current.width = 280
    canvas.current.height = 280

    // Set drawing styles to match original
    ctx.strokeStyle = "white"
    ctx.lineWidth = 8  // Matched original line width
    ctx.lineCap = "round"
    ctx.lineJoin = "round"

    let isDrawing = false
    let lastX = 0
    let lastY = 0

    function draw(e: MouseEvent | TouchEvent) {
      if (!isDrawing || !canvas.current) return

      const ctx = canvas.current.getContext("2d")
      if (!ctx) return

      const rect = canvas.current.getBoundingClientRect()
      const scaleX = canvas.current.width / rect.width
      const scaleY = canvas.current.height / rect.height

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

    function startDrawing(e: MouseEvent | TouchEvent) {
      if (!canvas.current) return

      isDrawing = true
      const rect = canvas.current.getBoundingClientRect()
      const scaleX = canvas.current.width / rect.width
      const scaleY = canvas.current.height / rect.height

      if (e instanceof MouseEvent) {
        lastX = (e.clientX - rect.left) * scaleX
        lastY = (e.clientY - rect.top) * scaleY
      } else {
        lastX = (e.touches[0].clientX - rect.left) * scaleX
        lastY = (e.touches[0].clientY - rect.top) * scaleY
      }
    }

    function stopDrawing() {
      isDrawing = false
    }

    // Mouse Events
    canvas.current.addEventListener("mousedown", startDrawing)
    canvas.current.addEventListener("mousemove", draw)
    canvas.current.addEventListener("mouseup", stopDrawing)
    canvas.current.addEventListener("mouseout", stopDrawing)

    // Touch Events
    canvas.current.addEventListener("touchstart", (e: TouchEvent) => {
      e.preventDefault()
      startDrawing(e)
    })
    canvas.current.addEventListener("touchmove", (e: TouchEvent) => {
      e.preventDefault()
      draw(e)
    })
    canvas.current.addEventListener("touchend", (e: TouchEvent) => {
      e.preventDefault()
      stopDrawing()
    })

    return () => {
      if (!canvas.current) return

      // Cleanup Mouse Events
      canvas.current.removeEventListener("mousedown", startDrawing)
      canvas.current.removeEventListener("mousemove", draw)
      canvas.current.removeEventListener("mouseup", stopDrawing)
      canvas.current.removeEventListener("mouseout", stopDrawing)

      // Cleanup Touch Events
      canvas.current.removeEventListener("touchstart", startDrawing)
      canvas.current.removeEventListener("touchmove", draw)
      canvas.current.removeEventListener("touchend", stopDrawing)
    }
  }, [ref])

  return (
    <div className="relative aspect-square bg-black rounded-lg overflow-hidden">
      <canvas 
        ref={ref} 
        className="absolute inset-0 w-full h-full touch-none"
        style={{ touchAction: "none" }}  // Important for mobile
      />
    </div>
  )
})

DrawingCanvas.displayName = "DrawingCanvas"

