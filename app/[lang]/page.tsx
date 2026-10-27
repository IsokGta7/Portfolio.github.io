import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { CaseStudyCard } from "@/components/case-study-card"
import { ContactLinks } from "@/components/contact-links"
import { Section } from "@/components/section"
import { SiteLayout } from "@/components/site-layout"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { demoRoutes } from "@/config/demos"
import { caseStudies } from "@/config/projects"
import { translations } from "@/config/translations"
import { resolveLang, type LangParams } from "@/lib/lang"

export default async function Home({ params }: LangParams) {
  const lang = await resolveLang(params)
  const t = translations[lang]

  return (
    <SiteLayout lang={lang}>
      <section aria-labelledby="hero-title" className="border-b">
        <div className="container py-20 sm:py-28">
          <p className="text-sm font-medium uppercase tracking-wider text-muted-foreground">{t.hero.role}</p>
          <h1 id="hero-title" className="mt-3 text-4xl font-bold tracking-tight sm:text-5xl">
            Isaac Rodríguez
          </h1>
          <p className="mt-6 max-w-2xl text-lg text-muted-foreground">{t.hero.summary}</p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Button asChild size="lg">
              <Link href={`/${lang}/#projects`}>
                {t.hero.viewProjects}
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <Link href={`/${lang}/#contact`}>{t.nav.contact}</Link>
            </Button>
          </div>
        </div>
      </section>

      <Section id="about" title={t.about.title}>
        <div className="grid gap-10 lg:grid-cols-2">
          <div className="max-w-prose space-y-4 text-base leading-relaxed">
            {t.about.paragraphs.map((paragraph) => (
              <p key={paragraph}>{paragraph}</p>
            ))}
          </div>
          <div className="grid gap-6 sm:grid-cols-2">
            {t.about.skillGroups.map((group) => (
              <div key={group.title}>
                <h3 className="text-lg font-semibold tracking-tight">{group.title}</h3>
                <dl className="mt-3 space-y-3 text-sm">
                  {group.skills.map((skill) => (
                    <div key={skill.name}>
                      <dt className="font-medium">{skill.name}</dt>
                      <dd className="text-muted-foreground">{skill.description}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            ))}
          </div>
        </div>
      </Section>

      <Section id="projects" title={t.projects.title} subtitle={t.projects.subtitle} className="border-t">
        <ul className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {caseStudies.map((project) => (
            <li key={project.id}>
              <CaseStudyCard project={project} lang={lang} />
            </li>
          ))}
        </ul>
      </Section>

      <Section id="demos" title={t.demos.title} subtitle={t.demos.subtitle} className="border-t">
        <ul className="grid gap-6 md:grid-cols-3">
          {demoRoutes.map(({ key, slug }) => {
            const demo = t.demoPages[key]
            return (
              <li key={key}>
                <Card className="flex h-full flex-col p-6">
                  <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{demo.technique}</p>
                  <h3 className="mt-2 text-lg font-semibold tracking-tight">{demo.title}</h3>
                  <p className="mt-2 text-sm text-muted-foreground">{demo.description}</p>
                  <Link
                    href={`/${lang}/${slug}/`}
                    className="mt-auto inline-flex items-center gap-1.5 self-start rounded-sm pt-6 text-sm font-medium underline-offset-4 hover:underline"
                  >
                    {t.demos.open}
                    <span className="sr-only">: {demo.title}</span>
                    <ArrowRight className="h-4 w-4" aria-hidden="true" />
                  </Link>
                </Card>
              </li>
            )
          })}
        </ul>
      </Section>

      <Section id="contact" title={t.contact.title} subtitle={t.contact.subtitle} className="border-t">
        <ContactLinks lang={lang} />
      </Section>
    </SiteLayout>
  )
}
