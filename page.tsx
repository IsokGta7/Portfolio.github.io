"use client"

import { useState, useEffect } from "react"
import { Moon, Sun, Menu, X } from "lucide-react"
import { motion } from "framer-motion"

import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { translations } from "@/lib/translations"
import { MobileProjectCard } from "@/components/mobile-project-card"
import { DesktopProjectCard } from "@/components/desktop-project-card"
import { SectionHeader } from "@/components/section-header"

export default function Page() {
  const [isDarkMode, setIsDarkMode] = useState(false)
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [language, setLanguage] = useState<"es" | "en">("es")
  const [isMobile, setIsMobile] = useState(false)

  const t = translations[language]

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768)
    }

    checkMobile()
    window.addEventListener("resize", checkMobile)

    return () => window.removeEventListener("resize", checkMobile)
  }, [])

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

  const ProjectCard = isMobile ? MobileProjectCard : DesktopProjectCard

  return (
    <div className={`min-h-screen bg-background text-foreground transition-colors duration-300`}>
      {/* Navigation */}
      <nav className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4">
          <div className="flex h-14 md:h-16 items-center justify-between">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-lg md:text-xl font-bold"
            >
              Portfolio
            </motion.div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-4">
              <Button variant="ghost" size="sm" onClick={() => scrollToSection("ai")}>
                {t.nav.ai}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => scrollToSection("programming")}>
                {t.nav.programming}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => scrollToSection("systems")}>
                {t.nav.systems}
              </Button>
              <Button variant="ghost" size="sm" onClick={() => scrollToSection("posts")}>
                {t.nav.posts}
              </Button>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="sm">
                    {language === "es" ? "Español" : "English"}
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => setLanguage("es")}>Español</DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setLanguage("en")}>English</DropdownMenuItem>
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
                  <DropdownMenuItem onClick={() => setLanguage("es")}>Español</DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setLanguage("en")}>English</DropdownMenuItem>
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
                {t.nav.ai}
              </Button>
              <Button
                variant="ghost"
                className="justify-start rounded-none h-12"
                onClick={() => scrollToSection("programming")}
              >
                {t.nav.programming}
              </Button>
              <Button
                variant="ghost"
                className="justify-start rounded-none h-12"
                onClick={() => scrollToSection("systems")}
              >
                {t.nav.systems}
              </Button>
              <Button
                variant="ghost"
                className="justify-start rounded-none h-12"
                onClick={() => scrollToSection("posts")}
              >
                {t.nav.posts}
              </Button>
            </div>
          </motion.div>
        )}
      </nav>

      {/* AI Section */}
      <section id="ai" className="container mx-auto px-4 py-8 md:py-24">
        <SectionHeader title={t.sections.ai.title} subtitle={t.sections.ai.subtitle} />

        <div className="grid gap-4 md:gap-6 md:grid-cols-2 lg:grid-cols-3">
          {t.sections.ai.projects.map((project, index) => (
            <ProjectCard
              key={index}
              title={project.title}
              description={project.description}
              link={project.link}
              buttonText={language === "es" ? "Ver Proyecto" : "View Project"}
              index={index}
            />
          ))}
        </div>
      </section>

      {/* Programming Section */}
      <section id="programming" className="container mx-auto px-4 py-8 md:py-24">
        <SectionHeader title={t.sections.programming.title} subtitle={t.sections.programming.subtitle} />

        <div className="grid gap-4 md:gap-6 md:grid-cols-2">
          {t.sections.programming.skills.map((skill, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className={`p-6 rounded-lg border ${isMobile ? "border-l-4 border-l-primary" : "hover:shadow-lg transition-shadow"}`}
            >
              <h3 className="text-xl font-semibold mb-2">{skill.name}</h3>
              <p className="text-muted-foreground">{skill.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Systems Design Section */}
      <section id="systems" className="container mx-auto px-4 py-8 md:py-24">
        <SectionHeader title={t.sections.systems.title} subtitle={t.sections.systems.subtitle} />

        <div className="grid gap-4 md:gap-6 md:grid-cols-3">
          {t.sections.systems.skills.map((skill, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className={`p-6 rounded-lg border ${isMobile ? "border-l-4 border-l-primary" : "hover:shadow-lg transition-shadow"}`}
            >
              <h3 className="text-xl font-semibold mb-2">{skill.name}</h3>
              <p className="text-muted-foreground">{skill.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Posts Section */}
      <section id="posts" className="container mx-auto px-4 py-8 md:py-24">
        <SectionHeader title={t.sections.posts.title} subtitle={t.sections.posts.subtitle} />

        <div className="text-center text-muted-foreground">{language === "es" ? "Próximamente" : "Coming soon"}</div>
      </section>
    </div>
  )
}

