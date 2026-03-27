import {
  CandlestickSeries,
  LineStyle,
  createChart,
  type IChartApi,
  type IPriceLine,
  type UTCTimestamp,
} from 'lightweight-charts'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Candle } from '../lib/api' // 본인의 파일 경로에 맞게 확인해주세요

function parseIsoDateToUtcTimestamp(iso: string): UTCTimestamp | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso)
  if (!m) return null
  const ms = Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]), 0, 0, 0)
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
  
  // 🔥 에러를 화면에 직접 띄워주는 상태값 추가
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

const seriesData = useMemo(() => {
    try {
      if (!props.candles || !Array.isArray(props.candles)) return []

      const validCandles = props.candles
        .map((c) => {
          const t = parseIsoDateToUtcTimestamp(c.time)
          
          // 🔥 핵심 수정: 문자로 넘어왔을 수도 있는 가격 데이터를 숫자로 확실하게 변환!
          const open = Number(c.open)
          const high = Number(c.high)
          const low = Number(c.low)
          const close = Number(c.close)

          // 변환 후 하나라도 정상적인 숫자가 아니라면 해당 캔들만 무시
          if (!t || isNaN(open) || isNaN(high) || isNaN(low) || isNaN(close)) return null
          
          return { time: t, open, high, low, close }
        })
        .filter((x): x is NonNullable<typeof x> => Boolean(x))

      // 1. 차트 필수 조건: 시간 오름차순 정렬
      validCandles.sort((a, b) => (a.time as number) - (b.time as number))

      // 2. 차트 필수 조건: 중복 날짜 완벽 제거 (중복 시 차트 뻗음 방지)
      const uniqueCandles = []
      const seen = new Set()
      for (const c of validCandles) {
        if (!seen.has(c.time)) {
          seen.add(c.time)
          uniqueCandles.push(c)
        }
      }
      return uniqueCandles
    } catch (e: any) {
      setErrorMsg("데이터 변환 실패: " + e.message)
      return []
    }
  }, [props.candles])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    try {
      const isDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
      const chart = createChart(el, {
        autoSize: true, // 크기 자동 맞춤
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
    } catch (e: any) {
      setErrorMsg("차트 생성 실패: " + e.message)
    }

    return () => {
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
        seriesRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (!chartRef.current || !seriesRef.current) return
    if (seriesData.length === 0) return

    try {
      seriesRef.current.setData(seriesData)
      chartRef.current.timeScale().fitContent()
      setErrorMsg(null) // 성공 시 에러 메시지 초기화
    } catch (err: any) {
      setErrorMsg("차트 그리기 실패: " + err.message)
    }
  }, [seriesData])

  useEffect(() => {
    const series = seriesRef.current
    if (!series) return

    try {
      for (const line of supportPriceLinesRef.current) series.removePriceLine(line)
      for (const line of resistancePriceLinesRef.current) series.removePriceLine(line)
      for (const line of boxPriceLinesRef.current) series.removePriceLine(line)
      supportPriceLinesRef.current = []
      resistancePriceLinesRef.current = []
      boxPriceLinesRef.current = []

      const isValidPrice = (p: any) => typeof p === 'number' && !isNaN(p) && p > 0

      for (const price of props.supportLines ?? []) {
        if (isValidPrice(price)) {
          supportPriceLinesRef.current.push(
            series.createPriceLine({ price, color: '#2563eb', lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true, title: '지지선' })
          )
        }
      }

      for (const price of props.resistanceLines ?? []) {
        if (isValidPrice(price)) {
          resistancePriceLinesRef.current.push(
            series.createPriceLine({ price, color: '#ef4444', lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true, title: '저항선' })
          )
        }
      }

      const box = props.boxRange
      if (box?.is_box && isValidPrice(box.top) && isValidPrice(box.bottom)) {
        for (const [price, title] of [[box.top, '박스 상단'], [box.bottom, '박스 하단']] as const) {
          boxPriceLinesRef.current.push(
            series.createPriceLine({ price: price as number, color: '#f59e0b', lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true, title })
          )
        }
      }
    } catch (e: any) {
      console.error("라인 오버레이 실패:", e)
    }
  }, [props.supportLines, props.resistanceLines, props.boxRange])

  return (
    <div style={{ position: 'relative', width: '100%', minHeight: '520px', border: '1px solid #e5e7eb', borderRadius: '8px', overflow: 'hidden', backgroundColor: '#f9fafb' }}>
      {/* 에러 발생 시 빨간색 글씨로 차트 껍데기 안에 즉시 원인을 보여줍니다 */}
      {errorMsg && (
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', color: 'red', fontWeight: 'bold', zIndex: 10 }}>
          {errorMsg}
        </div>
      )}
      {/* 차트가 들어갈 진짜 껍데기 */}
      <div ref={containerRef} style={{ width: '100%', height: '520px' }} />
    </div>
  )
}