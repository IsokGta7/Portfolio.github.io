"use client"

import { useEffect, useState, type ReactNode } from "react"
import Link from "next/link"
import { Moon, Sun, Menu, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { translations, type Language } from "@/config/translations"

interface ProjectLayoutProps {
  children: ReactNode
  language?: Language
  onLanguageChange?: (language: Language) => void
}

const sectionIds = ["ai", "programming", "systems", "posts"] as const

export function ProjectLayout({ children, language = "es", onLanguageChange }: ProjectLayoutProps) {
  const [isDarkMode, setIsDarkMode] = useState(true)
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const t = translations[language]

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

  const themeLabel = isDarkMode ? "Light mode" : "Dark mode"

  return (
    <div className="min-h-screen bg-background text-foreground">
      <nav className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4">
          <div className="flex h-16 items-center justify-between">
            <Link href="/" className="text-xl font-bold">
              Portfolio
            </Link>

            <div className="hidden items-center gap-4 md:flex">
              {sectionIds.map((id) => (
                <Button key={id} variant="ghost" size="sm" asChild>
                  <Link href={`/#${id}`}>{t.nav[id]}</Link>
                </Button>
              ))}

              {onLanguageChange && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="sm">
                      {language === "es" ? "Español" : "English"}
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => onLanguageChange("es")}>Español</DropdownMenuItem>
                    <DropdownMenuItem onClick={() => onLanguageChange("en")}>English</DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}

              <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label={themeLabel}>
                {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </Button>
            </div>

            <div className="flex items-center gap-2 md:hidden">
              {onLanguageChange && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon">
                      {language === "es" ? "ES" : "EN"}
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => onLanguageChange("es")}>Español</DropdownMenuItem>
                    <DropdownMenuItem onClick={() => onLanguageChange("en")}>English</DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}

              <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label={themeLabel}>
                {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </Button>

              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                aria-label="Menu"
                aria-expanded={isMenuOpen}
              >
                {isMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
              </Button>
            </div>
          </div>
        </div>

        {isMenuOpen && (
          <div className="border-t bg-background md:hidden">
            <div className="flex flex-col">
              {sectionIds.map((id) => (
                <Button key={id} variant="ghost" className="h-12 justify-start rounded-none" asChild>
                  <Link href={`/#${id}`} onClick={() => setIsMenuOpen(false)}>
                    {t.nav[id]}
                  </Link>
                </Button>
              ))}
            </div>
          </div>
        )}
      </nav>

      <main className="container mx-auto px-4 py-8 md:py-12">{children}</main>
    </div>
  )
}
