"use client"

import { useEffect, useState } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts"
import type { LayersModel, Tensor } from "@tensorflow/tfjs"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { translations, type Language } from "@/config/translations"
import { withBasePath } from "@/lib/base-path"

interface ChartPoint {
  time: string
  prediction: number | null
  realValue: number
}

interface Series {
  times: number[]
  closes: number[]
}

type Status = "loading" | "live" | "sample" | "error"

// The model reads the last 60 one-minute closes and predicts the next one.
const WINDOW = 60
const HISTORY = 120
const REFRESH_MS = 60_000
const FETCH_TIMEOUT_MS = 8_000

// api.binance.com is blocked in some countries; the public market-data mirror
// is tried next, and a built-in sample series is used when neither answers.
const KLINE_HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]

async function fetchCloses(): Promise<Series> {
  for (const host of KLINE_HOSTS) {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS)
    try {
      const response = await fetch(`${host}/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=${HISTORY}`, {
        signal: controller.signal,
      })
      if (!response.ok) continue
      const rows: unknown = await response.json()
      if (!Array.isArray(rows) || rows.length <= WINDOW) continue
      const candles = rows as [number, string, string, string, string][]
      return { times: candles.map((c) => c[0]), closes: candles.map((c) => parseFloat(c[4])) }
    } catch {
      // Network error, CORS refusal or timeout: try the next host.
    } finally {
      clearTimeout(timer)
    }
  }
  throw new Error("Binance is unreachable")
}

// A smooth, clearly artificial series so the chart and the model still have
// something to show offline. The UI labels it as sample data.
function sampleSeries(): Series {
  const start = Date.UTC(2024, 0, 1, 12, 0)
  const times: number[] = []
  const closes: number[] = []
  for (let i = 0; i < HISTORY; i++) {
    times.push(start + i * 60_000)
    closes.push(42_000 + i * 6 + Math.sin(i / 7) * 120 + Math.sin(i / 2.3) * 35)
  }
  return { times, closes }
}

// Each 60-point window is z-score normalised on its own before it goes into the
// model, and the output is mapped back with the same mean and deviation.
async function predictSeries(model: LayersModel, closes: number[]): Promise<(number | null)[]> {
  const tf = await import("@tensorflow/tfjs")
  const predictions: (number | null)[] = closes.map(() => null)
  const stats: { mean: number; std: number }[] = []
  const windows: number[][][] = []

  for (let end = WINDOW; end < closes.length; end++) {
    const slice = closes.slice(end - WINDOW, end)
    const mean = slice.reduce((a, b) => a + b, 0) / WINDOW
    const std = Math.sqrt(slice.reduce((a, b) => a + (b - mean) ** 2, 0) / WINDOW) || 1
    stats.push({ mean, std })
    windows.push(slice.map((v) => [(v - mean) / std]))
  }
  if (windows.length === 0) return predictions

  const output = tf.tidy(() => model.predict(tf.tensor3d(windows)) as Tensor)
  const values = await output.data()
  output.dispose()

  values.forEach((v, i) => {
    predictions[WINDOW + i] = v * stats[i].std + stats[i].mean
  })
  return predictions
}

// Axis text and grid follow the theme through currentColor (the chart sits in a
// text-muted-foreground box); the tooltip reads the theme tokens directly.
const tooltipStyle = {
  backgroundColor: "hsl(var(--popover))",
  color: "hsl(var(--popover-foreground))",
  border: "1px solid hsl(var(--border))",
  borderRadius: "var(--radius)",
}

const legendText = (value: string) => <span className="text-foreground">{value}</span>

export function BitcoinPredictionContent({ lang }: { lang: Language }) {
  const t = translations[lang].demoPages.bitcoinPrediction
  const [data, setData] = useState<ChartPoint[]>([])
  const [status, setStatus] = useState<Status>("loading")

  useEffect(() => {
    let cancelled = false
    let model: LayersModel | null = null
    let timer: ReturnType<typeof setTimeout> | undefined

    const refresh = async () => {
      let series: Series
      let live = true
      try {
        series = await fetchCloses()
      } catch {
        series = sampleSeries()
        live = false
      }
      if (cancelled || !model) return

      const predictions = await predictSeries(model, series.closes)
      if (cancelled) return

      setData(
        series.closes.map((close, i) => ({
          time: new Date(series.times[i]).toLocaleTimeString(lang, { hour: "2-digit", minute: "2-digit" }),
          realValue: close,
          prediction: predictions[i],
        })),
      )
      setStatus(live ? "live" : "sample")
      if (live) timer = setTimeout(() => void refresh(), REFRESH_MS)
    }

    const start = async () => {
      try {
        const tf = await import("@tensorflow/tfjs")
        const loaded = await tf.loadLayersModel(withBasePath("/assets/models/bitcoin/model_btc.json"))
        if (cancelled) {
          loaded.dispose()
          return
        }
        model = loaded
        await refresh()
      } catch (error) {
        console.error("Bitcoin demo failed to start:", error)
        if (!cancelled) setStatus("error")
      }
    }

    void start()
    return () => {
      cancelled = true
      clearTimeout(timer)
      model?.dispose()
    }
  }, [lang])

  const statusText: Record<Status, string> = {
    loading: t.loading,
    live: t.live,
    sample: t.unreachable,
    error: t.modelError,
  }
  const isWarning = status === "sample" || status === "error"

  return (
    <div className="grid gap-8">
      <p
        role="status"
        className={
          isWarning
            ? "rounded-md border border-amber-500/60 bg-amber-500/10 p-4 text-sm"
            : "text-center text-sm text-muted-foreground"
        }
      >
        {statusText[status]}
      </p>

      <Card>
        <CardHeader>
          <CardTitle>{t.predictionChart}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] w-full text-muted-foreground">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" strokeOpacity={0.25} />
                <XAxis dataKey="time" stroke="currentColor" tick={{ fill: "currentColor" }} minTickGap={24} />
                <YAxis stroke="currentColor" tick={{ fill: "currentColor" }} domain={["auto", "auto"]} width={72} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="prediction" name={t.prediction} stroke="#ef4444" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{t.comparison}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] w-full text-muted-foreground">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="currentColor" strokeOpacity={0.25} />
                <XAxis dataKey="time" stroke="currentColor" tick={{ fill: "currentColor" }} minTickGap={24} />
                <YAxis stroke="currentColor" tick={{ fill: "currentColor" }} domain={["auto", "auto"]} width={72} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend formatter={legendText} />
                <Line type="monotone" dataKey="realValue" name={t.realValue} stroke="#3b82f6" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="prediction" name={t.prediction} stroke="#ef4444" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
