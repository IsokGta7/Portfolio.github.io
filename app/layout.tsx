import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { MountGate } from "@/components/mount-gate"
import "./globals.css"

const inter = Inter({ subsets: ["latin"] })

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
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <MountGate>{children}</MountGate>
      </body>
    </html>
  )
}
