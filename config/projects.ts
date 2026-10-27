import type { Language } from "@/config/translations"

interface ProjectText {
  name: string
  problem: string
  role: string
  outcome: string
}

export interface CaseStudy {
  id: string
  stack: string[]
  live?: string
  repo?: string
  /** True when the code is not public, so the card says so instead of linking. */
  privateRepo?: boolean
  text: Record<Language, ProjectText>
}

export const caseStudies: CaseStudy[] = [
  {
    id: "ciensite",
    stack: ["Astro 7", "Decap CMS", "Netlify Functions"],
    live: "https://ciensite.netlify.app",
    repo: "https://github.com/IsokGta7/pagina-web-mlany",
    text: {
      es: {
        name: "ciensite",
        problem: "Un colectivo de divulgación científica necesitaba un sitio para publicar su contenido y recibir comentarios de sus lectores.",
        role: "Desarrollo",
        outcome: "Sitio estático en producción, con contenido editable desde Decap CMS y comentarios moderados mediante Netlify Functions.",
      },
      en: {
        name: "ciensite",
        problem: "A science-communication collective needed a site to publish its content and take comments from its readers.",
        role: "Development",
        outcome: "A static site in production, with content edited through Decap CMS and comments moderated by Netlify Functions.",
      },
    },
  },
  {
    id: "quincena",
    stack: ["React 19", "Supabase", "PWA"],
    privateRepo: true,
    text: {
      es: {
        name: "quincena",
        problem: "Saber cuánto dinero personal queda disponible hasta el próximo día de pago.",
        role: "Desarrollo",
        outcome: "PWA de flujo de efectivo personal; Row Level Security en Supabase hace que cada usuario vea solo sus propios datos.",
      },
      en: {
        name: "quincena",
        problem: "Knowing how much personal money is left until the next payday.",
        role: "Development",
        outcome: "A personal cash-flow PWA; Row Level Security in Supabase means each user sees only their own data.",
      },
    },
  },
  {
    id: "buho-soporte-digital",
    stack: ["React", "Supabase"],
    repo: "https://github.com/IsokGta7/buho-soporte-digital",
    text: {
      es: {
        name: "buho-soporte-digital",
        problem: "Una universidad necesitaba una mesa de ayuda de TI para atender las solicitudes de soporte.",
        role: "Desarrollo",
        outcome: "Mesa de ayuda de TI con acceso basado en roles.",
      },
      en: {
        name: "buho-soporte-digital",
        problem: "A university needed an IT help desk to handle its support requests.",
        role: "Development",
        outcome: "An IT help desk with role-based access.",
      },
    },
  },
  {
    id: "fastkv",
    stack: ["C++20", "CMake", "GoogleTest", "Google Benchmark"],
    repo: "https://github.com/IsokGta7/FastKV",
    text: {
      es: {
        name: "FastKV",
        problem: "Un almacén clave-valor que atienda accesos concurrentes, escrito en C++ moderno.",
        role: "Desarrollo",
        outcome: "Almacén clave-valor concurrente en C++20, con pruebas en GoogleTest y mediciones de rendimiento con Google Benchmark, compilado con CMake.",
      },
      en: {
        name: "FastKV",
        problem: "A key-value store that serves concurrent access, written in modern C++.",
        role: "Development",
        outcome: "A concurrent key-value store in C++20, tested with GoogleTest, measured with Google Benchmark and built with CMake.",
      },
    },
  },
  {
    id: "black-jack",
    stack: ["C++17"],
    repo: "https://github.com/IsokGta7/Black-Jack",
    text: {
      es: {
        name: "Black-Jack",
        problem: "Un juego de blackjack que se juega en la terminal.",
        role: "Desarrollo",
        outcome: "Juego de terminal en C++17 con pruebas automatizadas.",
      },
      en: {
        name: "Black-Jack",
        problem: "A game of blackjack played in the terminal.",
        role: "Development",
        outcome: "A C++17 terminal game with automated tests.",
      },
    },
  },
  {
    id: "programacion-paralela",
    stack: ["C", "MPI", "OpenMP"],
    repo: "https://github.com/IsokGta7/Programacion-Paralela-en-C",
    text: {
      es: {
        name: "Programacion-Paralela-en-C",
        problem: "Repartir el trabajo de un programa en C entre varios procesos e hilos.",
        role: "Desarrollo",
        outcome: "Programas paralelos con MPI y OpenMP, con integración continua.",
      },
      en: {
        name: "Programacion-Paralela-en-C",
        problem: "Splitting a C program's work across several processes and threads.",
        role: "Development",
        outcome: "Parallel programs with MPI and OpenMP, with continuous integration.",
      },
    },
  },
]
