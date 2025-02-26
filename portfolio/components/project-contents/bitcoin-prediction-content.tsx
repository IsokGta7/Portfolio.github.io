"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useTranslation } from "@/hooks/use-translation"
import { translations } from "@/config/translations"

// Generate sample data
const generateSampleData = (count: number) => {
  const now = new Date()
  return Array.from({ length: count }, (_, i) => {
    const time = new Date(now.getTime() - (count - i) * 60000)
    return {
      time: time.toLocaleTimeString(),
      prediction: Math.random() * 0.5 + 0.3,
      realValue: Math.random() * 0.5 + 0.3,
    }
  })
}

export function BitcoinPredictionContent() {
  const [language, setLanguage] = useState<"es" | "en">("es")
  const [data, setData] = useState(generateSampleData(20))

  const t = useTranslation(translations, language)

  useEffect(() => {
    const interval = setInterval(() => {
      setData((prevData) => {
        const newData = [
          ...prevData.slice(1),
          {
            time: new Date().toLocaleTimeString(),
            prediction: Math.random() * 0.5 + 0.3,
            realValue: Math.random() * 0.5 + 0.3,
          },
        ]
        return newData
      })
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const chartTheme = {
    backgroundColor: "transparent",
    textColor: "#94a3b8",
    gridColor: "#1f2937",
    tooltipBackground: "#1f2937",
    tooltipText: "#f8fafc",
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-8">
        <h1 className="text-3xl md:text-4xl font-bold mb-4">{t.projects.bitcoinPrediction.title}</h1>
        <p className="text-muted-foreground">{t.projects.bitcoinPrediction.description}</p>
      </motion.div>

      <div className="grid gap-8">
        <Card>
          <CardHeader>
            <CardTitle>{t.projects.bitcoinPrediction.predictionChart}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={data}
                  margin={{
                    top: 5,
                    right: 30,
                    left: 20,
                    bottom: 5,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.gridColor} />
                  <XAxis dataKey="time" stroke={chartTheme.textColor} tick={{ fill: chartTheme.textColor }} />
                  <YAxis stroke={chartTheme.textColor} tick={{ fill: chartTheme.textColor }} domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: chartTheme.tooltipBackground,
                      color: chartTheme.tooltipText,
                      border: "none",
                    }}
                  />
                  <Line type="monotone" dataKey="prediction" stroke="#ef4444" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t.projects.bitcoinPrediction.comparison}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={data}
                  margin={{
                    top: 5,
                    right: 30,
                    left: 20,
                    bottom: 5,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.gridColor} />
                  <XAxis dataKey="time" stroke={chartTheme.textColor} tick={{ fill: chartTheme.textColor }} />
                  <YAxis stroke={chartTheme.textColor} tick={{ fill: chartTheme.textColor }} domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: chartTheme.tooltipBackground,
                      color: chartTheme.tooltipText,
                      border: "none",
                    }}
                  />
                  <Legend
                    verticalAlign="top"
                    height={36}
                    wrapperStyle={{
                      color: chartTheme.textColor,
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="realValue"
                    name={t.projects.bitcoinPrediction.realValue}
                    stroke="#3b82f6"
                    dot={false}
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="prediction"
                    name={t.projects.bitcoinPrediction.prediction}
                    stroke="#ef4444"
                    dot={false}
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

