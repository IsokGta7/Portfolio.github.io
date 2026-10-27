"use client"

import { useEffect, useState, type ReactNode } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Moon, Sun, Menu, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { translations, type Language } from "@/config/translations"

interface ProjectLayoutProps {
  children: ReactNode
  lang: Language
}

const sectionIds = ["ai", "programming", "systems", "posts"] as const

// The same page in the other language: swap the first path segment.
function useOtherLanguageHref(lang: Language) {
  const pathname = usePathname() ?? `/${lang}/`
  const other: Language = lang === "es" ? "en" : "es"
  const rest = pathname.replace(/^\/(es|en)(?=\/|$)/, "")
  return { other, href: `/${other}${rest || "/"}` }
}

export function ProjectLayout({ children, lang }: ProjectLayoutProps) {
  const [isDarkMode, setIsDarkMode] = useState(true)
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const t = translations[lang]
  const switchTo = useOtherLanguageHref(lang)

  useEffect(() => {
    setIsDarkMode(document.documentElement.classList.contains("dark"))
  }, [])

  const toggleTheme = () => {
    const next = !isDarkMode
    setIsDarkMode(next)
    try {
      localStorage.setItem("theme", next ? "dark" : "light")
    } catch {
      // Storage can be unavailable (private mode); the toggle still works for this visit.
    }
    document.documentElement.classList.toggle("dark", next)
  }

  const themeLabel = isDarkMode ? t.ui.lightMode : t.ui.darkMode

  const languageLink = (
    <Button variant="ghost" size="sm" asChild>
      <Link href={switchTo.href} hrefLang={switchTo.other} lang={switchTo.other} aria-label={t.ui.switchLanguage}>
        {switchTo.other.toUpperCase()}
      </Link>
    </Button>
  )

  const themeButton = (
    <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label={themeLabel} title={themeLabel}>
      {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
    </Button>
  )

  return (
    <div className="min-h-screen bg-background text-foreground">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[60] focus:rounded-md focus:bg-background focus:px-4 focus:py-2 focus:ring-2 focus:ring-ring"
      >
        {t.ui.skipToContent}
      </a>

      <nav className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4">
          <div className="flex h-16 items-center justify-between">
            <Link href={`/${lang}/`} className="rounded-md text-xl font-bold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
              Portfolio
            </Link>

            <div className="hidden items-center gap-4 md:flex">
              {sectionIds.map((id) => (
                <Button key={id} variant="ghost" size="sm" asChild>
                  <Link href={`/${lang}/#${id}`}>{t.nav[id]}</Link>
                </Button>
              ))}
              {languageLink}
              {themeButton}
            </div>

            <div className="flex items-center gap-2 md:hidden">
              {languageLink}
              {themeButton}
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                aria-label={t.ui.menu}
                aria-expanded={isMenuOpen}
                aria-controls="mobile-menu"
              >
                {isMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
              </Button>
            </div>
          </div>
        </div>

        {isMenuOpen && (
          <div id="mobile-menu" className="border-t bg-background md:hidden">
            <div className="flex flex-col">
              {sectionIds.map((id) => (
                <Button key={id} variant="ghost" className="h-12 justify-start rounded-none" asChild>
                  <Link href={`/${lang}/#${id}`} onClick={() => setIsMenuOpen(false)}>
                    {t.nav[id]}
                  </Link>
                </Button>
              ))}
            </div>
          </div>
        )}
      </nav>

      <main id="main" className="container mx-auto px-4 py-8 md:py-12">
        {children}
      </main>
    </div>
  )
}
