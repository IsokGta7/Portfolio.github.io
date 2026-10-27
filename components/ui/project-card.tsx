"use client"

import { motion } from "framer-motion"
import Link from "next/link"
import { ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import type { Language } from "@/config/translations"

interface ProjectCardProps {
  title: string
  description: string
  link: string
  language: Language
  index: number
}

export function ProjectCard({ title, description, link, language, index }: ProjectCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.1 }}
    >
      <Card className="h-full transition-transform hover:scale-[1.02]">
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent>
          <Button variant="secondary" asChild className="w-full">
            <Link href={link}>
              <ChevronRight className="mr-2 h-4 w-4" />
              {language === "es" ? "Ver Proyecto" : "View Project"}
            </Link>
          </Button>
        </CardContent>
      </Card>
    </motion.div>
  )
}

