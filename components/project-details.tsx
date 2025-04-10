"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent } from "@/components/ui/card"
import ReactMarkdown from "react-markdown"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism"
import { useTranslation } from "@/hooks/use-translation"
import { translations } from "@/config/translations"

interface ProjectDetailsProps {
  projectKey: keyof typeof translations.es.projects
}

export function ProjectDetails({ projectKey }: ProjectDetailsProps) {
  const [language, setLanguage] = useState<"es" | "en">("es")
  const [activeTab, setActiveTab] = useState("markdown")

  const t = useTranslation(translations[language], language)
  const projectDetails = t?.projects?.[projectKey]

  if (!projectDetails) {
    return null
  }

  return (
    <Card className="mt-8">
      <CardContent className="p-6">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="markdown">{t?.common?.markdownExplanation ?? "Markdown Explanation"}</TabsTrigger>
            <TabsTrigger value="notebook">{t?.common?.jupyterNotebook ?? "Jupyter Notebook"}</TabsTrigger>
          </TabsList>
          <TabsContent value="markdown" className="mt-4">
            <div className="prose dark:prose-invert max-w-none">
              <ReactMarkdown
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || "")
                    return !inline && match ? (
                      <SyntaxHighlighter
                        {...props}
                        children={String(children).replace(/\n$/, "")}
                        style={vscDarkPlus}
                        language={match[1]}
                        PreTag="div"
                      />
                    ) : (
                      <code {...props} className={className}>
                        {children}
                      </code>
                    )
                  },
                }}
              >
                {projectDetails.markdownContent}
              </ReactMarkdown>
            </div>
          </TabsContent>
          <TabsContent value="notebook" className="mt-4">
            <div className="prose dark:prose-invert max-w-none">
              <ReactMarkdown
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || "")
                    return !inline && match ? (
                      <SyntaxHighlighter
                        {...props}
                        children={String(children).replace(/\n$/, "")}
                        style={vscDarkPlus}
                        language={match[1]}
                        PreTag="div"
                      />
                    ) : (
                      <code {...props} className={className}>
                        {children}
                      </code>
                    )
                  },
                }}
              >
                {`# Jupyter Notebook Content

\`\`\`json
${projectDetails.notebookContent}
\`\`\`
`}
              </ReactMarkdown>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

