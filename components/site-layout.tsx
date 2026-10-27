"use client"

import { useEffect, useState, type ReactNode } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Moon, Sun, Menu, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { translations, type Language } from "@/config/translations"
import { sourceUrl } from "@/lib/site"

interface SiteLayoutProps {
  children: ReactNode
  lang: Language
}

const sectionIds = ["about", "projects", "demos", "contact"] as const

// The same page in the other language: swap the first path segment.
function useOtherLanguageHref(lang: Language) {
  const pathname = usePathname() ?? `/${lang}/`
  const other: Language = lang === "es" ? "en" : "es"
  const rest = pathname.replace(/^\/(es|en)(?=\/|$)/, "")
  return { other, href: `/${other}${rest || "/"}` }
}

export function SiteLayout({ children, lang }: SiteLayoutProps) {
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
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[60] focus:rounded-md focus:bg-background focus:px-4 focus:py-2 focus:ring-2 focus:ring-ring"
      >
        {t.ui.skipToContent}
      </a>

      <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
        <nav className="container flex h-16 items-center justify-between gap-4" aria-label={t.ui.menu}>
          <Link href={`/${lang}/`} className="rounded-md text-lg font-semibold tracking-tight">
            Isaac Rodríguez
          </Link>

          <div className="hidden items-center gap-1 md:flex">
            {sectionIds.map((id) => (
              <Button key={id} variant="ghost" size="sm" asChild>
                <Link href={`/${lang}/#${id}`}>{t.nav[id]}</Link>
              </Button>
            ))}
            {languageLink}
            {themeButton}
          </div>

          <div className="flex items-center gap-1 md:hidden">
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
        </nav>

        {isMenuOpen && (
          <ul id="mobile-menu" className="container flex flex-col border-t py-2 md:hidden">
            {sectionIds.map((id) => (
              <li key={id}>
                <Button variant="ghost" className="h-12 w-full justify-start" asChild>
                  <Link href={`/${lang}/#${id}`} onClick={() => setIsMenuOpen(false)}>
                    {t.nav[id]}
                  </Link>
                </Button>
              </li>
            ))}
          </ul>
        )}
      </header>

      <main id="main" className="flex-1">
        {children}
      </main>

      <footer className="border-t">
        <div className="container flex flex-col gap-2 py-8 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
          <p>© {new Date().getFullYear()} Isaac Rodríguez</p>
          <a href={sourceUrl} className="rounded-sm underline underline-offset-4 hover:text-foreground">
            {t.footer.source}
          </a>
        </div>
      </footer>
    </div>
  )
}
