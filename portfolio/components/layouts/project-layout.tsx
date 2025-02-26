"use client"

import { useState, type ReactNode } from "react"
import { Moon, Sun, Menu, X } from "lucide-react"
import { motion } from "framer-motion"

import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { useTranslation } from "@/hooks/use-translation"
import { translations } from "@/config/translations"

interface ProjectLayoutProps {
  children: ReactNode
  language: "es" | "en"
  onLanguageChange: (language: "es" | "en") => void
}

export function ProjectLayout({ children, language, onLanguageChange }: ProjectLayoutProps) {
  const [isDarkMode, setIsDarkMode] = useState(true)
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const t = useTranslation(translations[language], language)

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode)
    document.documentElement.classList.toggle("dark")
  }

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id)
    if (element) {
      element.scrollIntoView({ behavior: "smooth" })
    }
    setIsMenuOpen(false)
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4">
          <div className="flex h-16 items-center justify-between">
            <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="text-xl font-bold">
              Portfolio
            </motion.div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-4">
              <Button variant="ghost" size="sm" onClick={() => scrollToSection("ai")}>
                {t?.nav?.ai}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => scrollToSection("programming")}>
                {t?.nav?.programming}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => scrollToSection("systems")}>
                {t?.nav?.systems}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => scrollToSection("posts")}>
                {t?.nav?.posts}
              </Button>

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

              <Button variant="ghost" size="icon" onClick={toggleTheme}>
                {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </Button>
            </div>

            {/* Mobile Menu Button */}
            <div className="flex md:hidden items-center gap-2">
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

              <Button variant="ghost" size="icon" onClick={toggleTheme}>
                {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </Button>

              <Button variant="ghost" size="icon" onClick={() => setIsMenuOpen(!isMenuOpen)}>
                {isMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
              </Button>
            </div>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="md:hidden border-t bg-background"
          >
            <div className="flex flex-col">
              <Button variant="ghost" className="justify-start rounded-none h-12" onClick={() => scrollToSection("ai")}>
                {t?.nav?.ai}
              </Button>
              <Button
                variant="ghost"
                className="justify-start rounded-none h-12"
                onClick={() => scrollToSection("programming")}
              >
                {t?.nav?.programming}
              </Button>
              <Button
                variant="ghost"
                className="justify-start rounded-none h-12"
                onClick={() => scrollToSection("systems")}
              >
                {t?.nav?.systems}
              </Button>
              <Button
                variant="ghost"
                className="justify-start rounded-none h-12"
                onClick={() => scrollToSection("posts")}
              >
                {t?.nav?.posts}
              </Button>
            </div>
          </motion.div>
        )}
      </nav>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 md:py-12">{children}</main>
    </div>
  )
}

