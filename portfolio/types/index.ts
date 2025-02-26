export interface Project {
  title: string
  description: string
  link: string
}

export interface Skill {
  name: string
  description: string
}

export interface Section {
  title: string
  subtitle: string
  projects?: Project[]
  skills?: Skill[]
}

export interface Navigation {
  ai: string
  programming: string
  systems: string
  posts: string
}

export interface Translations {
  nav: Navigation
  sections: {
    ai: Section
    programming: Section
    systems: Section
    posts: Section
  }
}

