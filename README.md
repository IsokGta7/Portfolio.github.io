# Isaac Rodríguez · Portfolio

Personal portfolio of Isaac Rodríguez, in Spanish (`/es/`, the default) and English (`/en/`). It presents a short
about section, case studies of recent projects, three machine-learning demos that run entirely in the browser with
TensorFlow.js, and contact links.

The site is built with Next.js 15 (App Router), React 19, Tailwind CSS and a few shadcn/ui components, and is exported
as plain static HTML for GitHub Pages.

## Running it locally

Node.js 20 or newer is needed (the deploy uses Node 22).

```bash
npm install
npm run dev        # development server on http://localhost:3000, opens /es/
npm run build      # static export into out/
npm run lint
npx tsc --noEmit
```

`npm run build` fails on any type or lint error. To look at the exported site, serve `out/` with any static file
server; opening the HTML files directly from disk does not work because the pages use absolute paths.

## Where things live

- `app/[lang]/` holds every page, generated once per language. `app/(root)/page.tsx` is the bare `/` page, which sends
  visitors to `/es/` with a meta refresh, since a static export cannot redirect on the server.
- `config/translations.ts` has all the interface text in both languages. `config/projects.ts` has the project case
  studies and `config/demos.ts` the demo routes.
- `lib/site.ts` has the site URL, the GitHub link and the contact details.
- `components/project-contents/` has the three demos. Their models are in `public/assets/models/`, and TensorFlow.js is
  only downloaded when a demo page is opened.
- `app/sitemap.ts` and `app/robots.ts` produce `sitemap.xml` and `robots.txt` at build time.

## Contact details to fill in

Open `lib/site.ts` and fill the `contact` object:

| Field      | What to put there                                                    |
| ---------- | -------------------------------------------------------------------- |
| `linkedin` | full profile URL                                                     |
| `email`    | the address only; the site turns it into a `mailto:` link            |
| `cvUrl`    | a full URL, or a path such as `/cv.pdf` for a file placed in `public/` |

Each one stays hidden until it has a value. The GitHub link is always shown.

## Deployment

`.github/workflows/pages.yml` runs on every push to `main` (and on demand). It installs with `npm ci`, builds, and
publishes `out/` to GitHub Pages. In the repository settings, Pages must use "GitHub Actions" as its source.

Where the site is served from depends on the repository name, and the workflow works it out:

- for a repository named `<owner>.github.io` the site is served from the domain root, so the base path is empty;
- for any other name it is served from `/<repository-name>/`, and the workflow sets `NEXT_PUBLIC_BASE_PATH` to that.

The workflow also passes the published address as `NEXT_PUBLIC_SITE_URL`, which the canonical links, Open Graph tags,
sitemap and robots.txt use. Renaming the repository to `IsokGta7.github.io` therefore needs no code change; the next
push to `main` deploys at the root.

To reproduce a project-page build locally:

```bash
NEXT_PUBLIC_BASE_PATH=/Portfolio.github.io npm run build
```

(In Git Bash on Windows, prefix it with `MSYS_NO_PATHCONV=1` so the path is not rewritten.)

## The Bitcoin demo

The Bitcoin demo reads one-minute BTC/USDT prices from Binance in the browser. `api.binance.com` is blocked in some
countries, so the demo falls back to Binance's public market-data mirror, and if neither responds it says so and
plots a sample series generated in the browser instead of real prices.
