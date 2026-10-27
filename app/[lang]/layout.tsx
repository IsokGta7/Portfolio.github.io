import type { Metadata } from "next"
import type { ReactNode } from "react"
import { notFound } from "next/navigation"
import { Inter } from "next/font/google"
import { isLanguage, languages, translations } from "@/config/translations"
import { pageMetadata } from "@/lib/metadata"
import { siteUrl } from "@/lib/site"
import "../globals.css"

const inter = Inter({ subsets: ["latin"] })

export const dynamicParams = false

export function generateStaticParams() {
  return languages.map((lang) => ({ lang }))
}

interface LayoutProps {
  children: ReactNode
  params: Promise<{ lang: string }>
}

export async function generateMetadata({ params }: Omit<LayoutProps, "children">): Promise<Metadata> {
  const { lang } = await params
  if (!isLanguage(lang)) return {}
  const { meta } = translations[lang]
  return {
    metadataBase: new URL(`${siteUrl}/`),
    ...pageMetadata(lang, ""),
    title: { default: meta.title, template: `%s · ${meta.siteName}` },
    authors: [{ name: meta.siteName }],
  }
}

// Applies the saved theme before the first paint so a light-mode visitor does
// not see a dark flash. Dark is the default when nothing is saved.
const themeScript = `try{var t=localStorage.getItem("theme")||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");document.documentElement.classList.toggle("dark",t==="dark")}catch(e){}`

export default async function LanguageLayout({ children, params }: LayoutProps) {
  const { lang } = await params
  if (!isLanguage(lang)) notFound()

  return (
    <html lang={lang} className="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className={inter.className}>{children}</body>
    </html>
  )
}
