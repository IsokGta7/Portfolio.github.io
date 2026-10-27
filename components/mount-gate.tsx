"use client"

import { useEffect, useState } from "react"

export function MountGate({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  // Delay rendering until mounted to prevent hydration mismatch
  if (!mounted) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="animate-pulse">Loading AI components...</div>
      </div>
    )
  }

  return <>{children}</>
}
