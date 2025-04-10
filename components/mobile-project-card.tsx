import { motion } from "framer-motion"
import { ChevronRight } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

interface MobileProjectCardProps {
  title: string
  description: string
  link: string
  buttonText: string
  index: number
}

export function MobileProjectCard({ title, description, link, buttonText, index }: MobileProjectCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      whileInView={{ opacity: 1, x: 0 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.1 }}
    >
      <Card className="relative overflow-hidden border-l-4 border-l-primary">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg line-clamp-1">{title}</CardTitle>
          <CardDescription className="text-sm line-clamp-2">{description}</CardDescription>
        </CardHeader>
        <CardContent className="pb-3">
          <Button variant="ghost" asChild className="p-0 h-auto hover:bg-transparent">
            <a href={link} className="flex items-center text-primary">
              {buttonText}
              <ChevronRight className="h-4 w-4 ml-1" />
            </a>
          </Button>
        </CardContent>
      </Card>
    </motion.div>
  )
}

