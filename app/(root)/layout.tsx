import type { ReactNode } from "react"
import "../globals.css"

// Only the language redirect at "/" uses this layout; every real page lives
// under app/[lang], which renders its own <html lang>.
export default function RootRedirectLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  )
}
