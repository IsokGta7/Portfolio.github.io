import type { Metadata } from "next"
import Link from "next/link"
import { withBasePath } from "@/lib/base-path"
import { absoluteUrl } from "@/lib/site"

// A static export cannot redirect on the server, so "/" sends the visitor to
// the default language with a meta refresh, and offers plain links meanwhile.
export const metadata: Metadata = {
  title: "Isaac Rodríguez",
  robots: { index: false },
  alternates: { canonical: absoluteUrl("/es/") },
}

export default function RootRedirect() {
  return (
    <main className="flex min-h-screen items-center justify-center p-4">
      <meta httpEquiv="refresh" content={`0; url=${withBasePath("/es/")}`} />
      <p>
        <Link href="/es/" hrefLang="es" className="underline underline-offset-4">
          Español
        </Link>
        {" · "}
        <Link href="/en/" hrefLang="en" lang="en" className="underline underline-offset-4">
          English
        </Link>
      </p>
    </main>
  )
}
