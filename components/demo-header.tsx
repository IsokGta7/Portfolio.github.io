interface DemoHeaderProps {
  title: string
  description: string
}

export function DemoHeader({ title, description }: DemoHeaderProps) {
  return (
    <header className="mb-8 text-center">
      <h1 className="mb-4 text-3xl font-bold md:text-4xl">{title}</h1>
      <p className="text-muted-foreground">{description}</p>
    </header>
  )
}
