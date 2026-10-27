import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

interface SectionProps {
  id: string
  title: string
  subtitle?: string
  children: ReactNode
  className?: string
}

// Every home-page section shares this spacing and heading scale.
export function Section({ id, title, subtitle, children, className }: SectionProps) {
  const headingId = `${id}-title`
  return (
    <section id={id} aria-labelledby={headingId} className={cn("scroll-mt-16 py-16 sm:py-24", className)}>
      <div className="container">
        <header className="mb-8 max-w-2xl sm:mb-12">
          <h2 id={headingId} className="text-2xl font-semibold tracking-tight sm:text-3xl">
            {title}
          </h2>
          {subtitle && <p className="mt-3 text-base text-muted-foreground">{subtitle}</p>}
        </header>
        {children}
      </div>
    </section>
  )
}
