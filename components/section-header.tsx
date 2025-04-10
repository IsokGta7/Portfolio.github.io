import { motion } from "framer-motion"

interface SectionHeaderProps {
  title: string
  subtitle: string
}

export function SectionHeader({ title, subtitle }: SectionHeaderProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="text-center mb-8 md:mb-12"
    >
      <h2 className="text-3xl md:text-6xl font-bold mb-3 md:mb-6">{title}</h2>
      <p className="text-base md:text-xl text-muted-foreground max-w-2xl mx-auto">{subtitle}</p>
    </motion.div>
  )
}

