import type { ReactNode } from "react"
import { FileText, Github, Linkedin, Mail } from "lucide-react"
import { Button } from "@/components/ui/button"
import { translations, type Language } from "@/config/translations"
import { withBasePath } from "@/lib/base-path"
import { contact, githubUrl } from "@/lib/site"

interface ContactLink {
  href: string
  label: string
  icon: ReactNode
}

// GitHub is always shown; the rest appear only once they are filled in lib/site.ts.
export function ContactLinks({ lang }: { lang: Language }) {
  const t = translations[lang].contact
  const iconClass = "h-4 w-4"

  const links: ContactLink[] = [{ href: githubUrl, label: t.github, icon: <Github className={iconClass} aria-hidden="true" /> }]
  if (contact.linkedin) {
    links.push({ href: contact.linkedin, label: t.linkedin, icon: <Linkedin className={iconClass} aria-hidden="true" /> })
  }
  if (contact.email) {
    links.push({ href: `mailto:${contact.email}`, label: t.email, icon: <Mail className={iconClass} aria-hidden="true" /> })
  }
  if (contact.cvUrl) {
    const href = contact.cvUrl.startsWith("/") ? withBasePath(contact.cvUrl) : contact.cvUrl
    links.push({ href, label: t.cv, icon: <FileText className={iconClass} aria-hidden="true" /> })
  }

  return (
    <ul className="flex flex-wrap gap-3">
      {links.map((link, i) => (
        <li key={link.href}>
          <Button variant={i === 0 ? "default" : "outline"} asChild>
            <a href={link.href}>
              {link.icon}
              {link.label}
            </a>
          </Button>
        </li>
      ))}
    </ul>
  )
}
