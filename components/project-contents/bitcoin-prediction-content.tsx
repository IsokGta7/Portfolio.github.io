"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { motion } from "framer-motion"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useTranslation } from "@/hooks/use-translation"
import { translations } from "@/config/translations"
import * as tf from "@tensorflow/tfjs"

interface ChartData {
  time: string
  prediction: number | null
  realValue: number | null
}

export function BitcoinPredictionContent() {
  const [language] = useState<"es" | "en">("es")
  const [data, setData] = useState<ChartData[]>([])
  const [model, setModel] = useState<tf.LayersModel | null>(null)
  const [dataMean, setDataMean] = useState<number>(0)
  const [dataStd, setDataStd] = useState<number>(1)
  const recentData = useRef<number[]>([])
  const isMounted = useRef(true)

  const t = useTranslation(translations, language)

  // Load TensorFlow.js model with cleanup
  useEffect(() => {
    isMounted.current = true
    let modelInstance: tf.LayersModel | null = null
    
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadLayersModel('/assets/models/bitcoin/model_btc.json')
        if (isMounted.current) {
          modelInstance = loadedModel
          setModel(loadedModel)
        }
      } catch (error) {
        console.error('Error loading model:', error)
      }
    }
  
    loadModel()
    return () => {
      isMounted.current = false
      modelInstance?.dispose()
    }
  }, [])

  // Fetch Binance data with error handling
  const fetchBinanceData = useCallback(async () => {
    try {
      const response = await fetch('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=10')
      const binanceData = await response.json()
      return binanceData.map((candle: string[]) => parseFloat(candle[4]))
    } catch (error) {
      console.error('Error fetching Binance data:', error)
      return []
    }
  }, [])

  const calculatePrediction = useCallback(() => {
    if (!model || !dataMean || !dataStd || recentData.current.length < 60) return null
    
    try {
      const inputTensor = tf.tensor(recentData.current)
        .sub(dataMean)
        .div(dataStd)
        .reshape([1, 60, 1])

      const predictionTensor = model.predict(inputTensor) as tf.Tensor
      const prediction = predictionTensor.dataSync()[0] * dataStd + dataMean
      
      tf.dispose([inputTensor, predictionTensor])
      return prediction
    } catch (error) {
      console.error('Prediction calculation failed:', error)
      return null
    }
  }, [model, dataMean, dataStd])

  const updateCharts = useCallback(async () => {
    if (!isMounted.current || !model) return

    try {
      const closingPrices = await fetchBinanceData()
      if (!closingPrices.length) return

      recentData.current = [...recentData.current, ...closingPrices].slice(-60)

      // Update normalization parameters if needed
      if (isMounted.current && (!dataMean || !dataStd)) {
        const tensorPrices = tf.tensor(recentData.current)
        const [mean, std] = [
          tensorPrices.mean().dataSync()[0],
          tensorPrices.sub(tensorPrices.mean()).square().mean().sqrt().dataSync()[0]
        ]
        if (isMounted.current) {
          setDataMean(mean)
          setDataStd(std)
        }
        tensorPrices.dispose()
      }

      if (isMounted.current) {
        setData(prev => {
          const newData = [...prev, {
            time: new Date().toLocaleTimeString(),
            prediction: calculatePrediction(),
            realValue: closingPrices[closingPrices.length - 1]
          }].slice(-100)
          return newData
        })
      }
    } catch (error) {
      console.error('Update error:', error)
    }
  }, [model, dataMean, dataStd, fetchBinanceData, calculatePrediction])

  useEffect(() => {
    const interval = setInterval(updateCharts, 10000)
    return () => clearInterval(interval)
  }, [updateCharts])

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
        <h1 className="text-3xl md:text-4xl font-bold mb-4">
          {t?.projects?.bitcoinPrediction?.title || "BTC/USDT Predictions"}
        </h1>
        <p className="text-muted-foreground">
          {t?.projects?.bitcoinPrediction?.description || "Real-time price predictions using AI model"}
        </p>
      </motion.div>

      <div className="grid gap-8">
        <Card>
          <CardHeader>
            <CardTitle>
              {t?.projects?.bitcoinPrediction?.predictionChart || "Prediction Chart"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.gridColor} />
                  <XAxis dataKey="time" stroke={chartTheme.textColor} />
                  <YAxis stroke={chartTheme.textColor} domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: chartTheme.tooltipBackground,
                      color: chartTheme.tooltipText,
                      border: "none",
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="prediction"
                    stroke="#ef4444"
                    dot={false}
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>
              {t?.projects?.bitcoinPrediction?.comparison || "Real vs Predicted Comparison"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.gridColor} />
                  <XAxis dataKey="time" stroke={chartTheme.textColor} />
                  <YAxis stroke={chartTheme.textColor} domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: chartTheme.tooltipBackground,
                      color: chartTheme.tooltipText,
                      border: "none",
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="realValue"
                    name={t?.projects?.bitcoinPrediction?.realValue || "Real Value"}
                    stroke="#3b82f6"
                    dot={false}
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="prediction"
                    name={t?.projects?.bitcoinPrediction?.prediction || "Prediction"}
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