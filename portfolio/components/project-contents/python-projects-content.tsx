"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useTranslation } from "@/hooks/use-translation"
import { translations } from "@/config/translations"
import { CodeBlock } from "@/components/ui/code-block"

export function PythonProjectsContent() {
  const [language, setLanguage] = useState<"es" | "en">("es")
  const t = useTranslation(translations, language)

  const pythonProjects = t.projects.python

  return (
    <div className="container mx-auto px-4 py-8">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-8">
        <h1 className="text-3xl md:text-4xl font-bold mb-4">{t.projects.python.title}</h1>
        <p className="text-muted-foreground">{t.projects.python.description}</p>
      </motion.div>

      <div className="grid gap-8 md:grid-cols-2">
        {pythonProjects.map((project, index) => (
          <Card key={index}>
            <CardHeader>
              <CardTitle>{project.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-4">{project.description}</p>
              <CodeBlock language="python" code={project.sampleCode} />
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}

