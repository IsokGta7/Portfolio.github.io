"use client"

import { motion } from "framer-motion"
import { ProjectCard } from "@/components/ui/project-card"
import { SkillCard } from "@/components/ui/skill-card"
import type { Project, Skill } from "@/types"

interface ProjectSectionProps {
  id: string
  title: string
  subtitle: string
  projects?: Project[]
  skills?: Skill[]
  language: "es" | "en"
}

export function ProjectSection({ id, title, subtitle, projects, skills, language }: ProjectSectionProps) {
  return (
    <section id={id} className="py-12 md:py-24">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="text-center mb-12"
      >
        <h2 className="text-4xl md:text-6xl font-bold mb-6">{title}</h2>
        <p className="text-lg md:text-xl text-muted-foreground">{subtitle}</p>
      </motion.div>

      {projects && (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {projects.map((project, index) => (
            <ProjectCard
              key={index}
              title={project.title}
              description={project.description}
              link={project.link}
              language={language}
              index={index}
            />
          ))}
        </div>
      )}

      {skills && (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {skills.map((skill, index) => (
            <SkillCard key={index} name={skill.name} description={skill.description} index={index} />
          ))}
        </div>
      )}

      {!projects && !skills && (
        <div className="text-center text-muted-foreground">{language === "es" ? "Próximamente" : "Coming soon"}</div>
      )}
    </section>
  )
}

