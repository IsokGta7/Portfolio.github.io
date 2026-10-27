import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { CodeBlock } from "@/components/ui/code-block"
import { translations, type Language } from "@/config/translations"

export function CppProjectsContent({ lang }: { lang: Language }) {
  const t = translations[lang].projects.cpp

  return (
    <div className="grid gap-8 md:grid-cols-2">
      {t.projects.map((project) => (
        <Card key={project.title}>
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
  )
}
