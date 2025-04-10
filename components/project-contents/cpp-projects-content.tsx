"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useTranslation } from "@/hooks/use-translation"
import { translations } from "@/config/translations"
import { CodeBlock } from "@/components/ui/code-block"

export function CppProjectsContent() {
  const [language, setLanguage] = useState<"es" | "en">("es")
  const t = useTranslation(translations, language)

  const cppProjects = t.

 | "en">("es")
  const t = useTranslation(translations, language)

  const cppProjects = t.projects.cpp

  return (
    <div className="container mx-auto px-4 py-8">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-8">
        <h1 className="text-3xl md:text-4xl font-bold mb-4">{t.projects.cpp.title}</h1>
        <p className="text-muted-foreground">{t.projects.cpp.description}</p>
      </motion.div>

      <div className="grid gap-8 md:grid-cols-2">
        {cppProjects.map((project, index) => (
          <Card key={index}>
            <CardHeader>
              <CardTitle>{project.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-4">{project.description}</p>
              <CodeBlock language="cpp" code={project.sampleCode} />
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}

