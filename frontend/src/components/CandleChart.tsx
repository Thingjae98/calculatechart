import {
  CandlestickSeries,
  LineStyle,
  createChart,
  type IChartApi,
  type IPriceLine,
  type UTCTimestamp,
} from 'lightweight-charts'
import { useEffect, useMemo, useRef } from 'react'
import type { Candle } from '../lib/api' // 경로가 다르다면 기존 파일에 맞게 수정해주세요

function parseIsoDateToUtcTimestamp(iso: string): UTCTimestamp | null {
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

  // 데이터 안전 필터링 (NaN이나 유효하지 않은 값이 있으면 렌더링 중단 방지)
  const seriesData = useMemo(() => {
    if (!props.candles || !Array.isArray(props.candles)) return []
    
    return props.candles
      .map((c) => {
        const t = parseIsoDateToUtcTimestamp(c.time)
        if (!t || isNaN(c.open) || isNaN(c.high) || isNaN(c.low) || isNaN(c.close)) return null
        return { time: t, open: c.open, high: c.high, low: c.low, close: c.close }
      })
      .filter((x): x is NonNullable<typeof x> => Boolean(x))
      // lightweight-charts는 시간이 반드시 오름차순이어야 하므로 강제 정렬
      .sort((a, b) => (a.time as number) - (b.time as number))
  }, [props.candles])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const isDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
    const chart = createChart(el, {
      width: el.clientWidth,  
      height: 520, // 강제 높이 지정
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

    return () => {
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
    }
  }, [])

  useEffect(() => {
    const chart = chartRef.current
    const series = seriesRef.current
    if (!chart || !series) return

    // 데이터가 있을 때만 렌더링 시도
    if (seriesData.length > 0) {
      try {
        series.setData(seriesData)
        chart.timeScale().fitContent()
      } catch (err) {
        console.error("차트 데이터 세팅 중 에러 발생:", err)
      }
    }
  }, [seriesData])

  useEffect(() => {
    const series = seriesRef.current
    if (!series) return

    // 기존 라인 초기화
    for (const line of supportPriceLinesRef.current) series.removePriceLine(line)
    for (const line of resistancePriceLinesRef.current) series.removePriceLine(line)
    for (const line of boxPriceLinesRef.current) series.removePriceLine(line)
    supportPriceLinesRef.current = []
    resistancePriceLinesRef.current = []
    boxPriceLinesRef.current = []

    const supports = props.supportLines ?? []
    const resistances = props.resistanceLines ?? []
    const box = props.boxRange

    // 숫자 확인용 안전 함수 (NaN이 차트에 들어가면 뻗어버림)
    const isValidPrice = (p: any) => typeof p === 'number' && !isNaN(p) && p > 0;

    for (const price of supports) {
      if (isValidPrice(price)) {
        const line = series.createPriceLine({
          price, color: '#2563eb', lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true, title: '지지선',
        })
        supportPriceLinesRef.current.push(line)
      }
    }

    for (const price of resistances) {
      if (isValidPrice(price)) {
        const line = series.createPriceLine({
          price, color: '#ef4444', lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true, title: '저항선',
        })
        resistancePriceLinesRef.current.push(line)
      }
    }

    if (box?.is_box && isValidPrice(box.top) && isValidPrice(box.bottom)) {
      for (const [price, title] of [[box.top, '박스 상단'], [box.bottom, '박스 하단']] as const) {
        if (isValidPrice(price)) {
          const line = series.createPriceLine({
            price, color: '#f59e0b', lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true, title,
          })
          boxPriceLinesRef.current.push(line)
        }
      }
    }
  }, [props.supportLines, props.resistanceLines, props.boxRange])

  // 가장 중요한 부분: 컨테이너에 강제로 최소 높이와 너비를 부여합니다.
  return <div ref={containerRef} className="chart" style={{ width: '100%', height: '520px', minHeight: '520px' }} />
}