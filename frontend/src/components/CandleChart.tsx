import {
  CandlestickSeries,
  LineStyle,
  createChart,
  type IChartApi,
  type IPriceLine,
  type UTCTimestamp,
} from 'lightweight-charts'
import { useEffect, useMemo, useRef } from 'react'
import type { Candle } from '../lib/api'

function parseIsoDateToUtcTimestamp(iso: string): UTCTimestamp | null {
  // expected 'YYYY-MM-DD'
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso)
  if (!m) return null
  const y = Number(m[1])
  const mo = Number(m[2])
  const d = Number(m[3])
  if (!Number.isFinite(y) || !Number.isFinite(mo) || !Number.isFinite(d)) return null
  const ms = Date.UTC(y, mo - 1, d, 0, 0, 0)
  return Math.floor(ms / 1000) as UTCTimestamp
}

export function CandleChart(props: {
  candles: Candle[]
  supportLines?: number[]
  resistanceLines?: number[]
  boxRange?: { is_box: boolean; top?: number; bottom?: number }
}) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<any>(null)
  const supportPriceLinesRef = useRef<IPriceLine[]>([])
  const resistancePriceLinesRef = useRef<IPriceLine[]>([])
  const boxPriceLinesRef = useRef<IPriceLine[]>([])

  const seriesData = useMemo(() => {
    return props.candles
      .map((c) => {
        const t = parseIsoDateToUtcTimestamp(c.time)
        if (!t) return null
        return { time: t, open: c.open, high: c.high, low: c.low, close: c.close }
      })
      .filter((x): x is NonNullable<typeof x> => Boolean(x))
  }, [props.candles])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const isDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
    const chart = createChart(el, {
      autoSize: true,
      height: 520,
      layout: {
        background: { color: isDark ? '#111827' : '#ffffff' },
        textColor: isDark ? '#e5e7eb' : '#111827',
      },
      grid: {
        vertLines: { color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.08)' },
        horzLines: { color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.08)' },
      },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false },
      crosshair: { mode: 1 },
    })
    chartRef.current = chart

    seriesRef.current = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })

    // 중요: 차트/시리즈 생성 직후 데이터 세팅(초기 effect 레이스로 빈 화면 방지)
    try {
      seriesRef.current.setData(seriesData)
      chart.timeScale().fitContent()
    } catch {
      // ignore
    }

    return () => {
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const chart = chartRef.current
    const series = seriesRef.current
    if (!chart || !series) return

    series.setData(seriesData)
    chart.timeScale().fitContent()
  }, [seriesData])

  useEffect(() => {
    const series = seriesRef.current
    if (!series) return

    // 기존 라인 제거
    for (const line of supportPriceLinesRef.current) series.removePriceLine(line)
    for (const line of resistancePriceLinesRef.current) series.removePriceLine(line)
    for (const line of boxPriceLinesRef.current) series.removePriceLine(line)
    supportPriceLinesRef.current = []
    resistancePriceLinesRef.current = []
    boxPriceLinesRef.current = []

    const supports = props.supportLines ?? []
    const resistances = props.resistanceLines ?? []
    const box = props.boxRange

    // 지지선: 파란색 점선
    for (const price of supports) {
      const line = series.createPriceLine({
        price,
        color: '#2563eb',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: true,
        title: '지지선',
      })
      supportPriceLinesRef.current.push(line)
    }

    // 저항선: 빨간색 점선
    for (const price of resistances) {
      const line = series.createPriceLine({
        price,
        color: '#ef4444',
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: true,
        title: '저항선',
      })
      resistancePriceLinesRef.current.push(line)
    }

    // 박스권: 상단/하단 주황색 점선 (사각형 채움은 추후 확장)
    if (box?.is_box && typeof box.top === 'number' && typeof box.bottom === 'number') {
      for (const [price, title] of [
        [box.top, '박스 상단'],
        [box.bottom, '박스 하단'],
      ] as const) {
        const line = series.createPriceLine({
          price,
          color: '#f59e0b',
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          axisLabelVisible: true,
          title,
        })
        boxPriceLinesRef.current.push(line)
      }
    }
  }, [props.supportLines, props.resistanceLines, props.boxRange])

  return <div ref={containerRef} className="chart" />
}

