import { Code, ExternalLink, Lock } from "lucide-react"
import { Card } from "@/components/ui/card"
import type { CaseStudy } from "@/config/projects"
import { translations, type Language } from "@/config/translations"

export function CaseStudyCard({ project, lang }: { project: CaseStudy; lang: Language }) {
  const text = project.text[lang]
  const labels = translations[lang].projects.labels

  return (
    <Card className="flex h-full flex-col p-6">
      <article className="flex h-full flex-col">
        <h3 className="text-lg font-semibold tracking-tight">{text.name}</h3>

        <dl className="mt-4 space-y-3 text-sm">
          <div>
            <dt className="font-medium">{labels.problem}</dt>
            <dd className="text-muted-foreground">{text.problem}</dd>
          </div>
          <div>
            <dt className="font-medium">{labels.role}</dt>
            <dd className="text-muted-foreground">{text.role}</dd>
          </div>
          <div>
            <dt className="font-medium">{labels.outcome}</dt>
            <dd className="text-muted-foreground">{text.outcome}</dd>
          </div>
          <div>
            <dt className="sr-only">{labels.stack}</dt>
            <dd>
              <ul className="flex flex-wrap gap-2" aria-label={labels.stack}>
                {project.stack.map((item) => (
                  <li key={item} className="rounded-md bg-secondary px-2 py-1 text-xs font-medium text-secondary-foreground">
                    {item}
                  </li>
                ))}
              </ul>
            </dd>
          </div>
        </dl>

        <div className="mt-auto flex flex-wrap gap-x-6 gap-y-2 pt-6 text-sm font-medium">
          {project.live && (
            <a href={project.live} className="inline-flex items-center gap-1.5 rounded-sm underline-offset-4 hover:underline">
              <ExternalLink className="h-4 w-4" aria-hidden="true" />
              {labels.live}
              <span className="sr-only">: {text.name}</span>
            </a>
          )}
          {project.repo && (
            <a href={project.repo} className="inline-flex items-center gap-1.5 rounded-sm underline-offset-4 hover:underline">
              <Code className="h-4 w-4" aria-hidden="true" />
              {labels.repo}
              <span className="sr-only">: {text.name}</span>
            </a>
          )}
          {project.privateRepo && (
            <span className="inline-flex items-center gap-1.5 text-muted-foreground">
              <Lock className="h-4 w-4" aria-hidden="true" />
              {labels.privateRepo}
            </span>
          )}
        </div>
      </article>
    </Card>
  )
}
