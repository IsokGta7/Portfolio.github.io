"use client" // Add this at the top if using client components in layout

import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { useEffect, useState } from "react"
import "./globals.css"

const inter = Inter({ subsets: ["latin"] })

// Server-side metadata
export const metadata: Metadata = {
  title: "Portfolio - Projects",
  description: "AI and Machine Learning Projects Portfolio",
  generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        {/* Delay rendering until mounted to prevent hydration mismatch */}
        {mounted && children}
        
        {/* Optional: Add loading state */}
        {!mounted && (
          <div className="flex h-screen items-center justify-center">
            <div className="animate-pulse">Loading AI components...</div>
          </div>
        )}
      </body>
    </html>
  )
}