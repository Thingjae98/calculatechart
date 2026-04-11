import {
  CandlestickSeries,
  LineStyle,
  createChart,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
} from 'lightweight-charts'
import { useEffect, useMemo, useRef, useState } from 'react'

// 🔥 경로 에러를 원천 차단하기 위해, 다른 파일에서 불러오지 않고 직접 타입을 정의합니다!
export interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

const toMessage = (e: unknown): string => (e instanceof Error ? e.message : String(e))

function parseAndDedup(candles: Candle[]) {
  const parsePrice = (v: unknown): number => Number(String(v ?? '').replace(/,/g, ''))

  const valid = candles.map((c) => {
    const timeStr = (typeof c.time === 'string' ? c.time : '').substring(0, 10)
    const open = parsePrice(c.open)
    const high = parsePrice(c.high)
    const low = parsePrice(c.low)
    const close = parsePrice(c.close)
    if (
      !Number.isFinite(open) || !Number.isFinite(high) ||
      !Number.isFinite(low) || !Number.isFinite(close) ||
      !/^\d{4}-\d{2}-\d{2}$/.test(timeStr)
    ) return null
    return { time: timeStr, open, high, low, close }
  }).filter((x): x is NonNullable<typeof x> => Boolean(x))

  valid.sort((a, b) => a.time.localeCompare(b.time))

  const unique: typeof valid = []
  let lastTime = ''
  for (const c of valid) {
    if (c.time !== lastTime) { unique.push(c); lastTime = c.time }
  }
  return unique
}

export function CandleChart(props: {
  candles: Candle[]
  predictedCandles?: Candle[]
  supportLines?: number[]
  resistanceLines?: number[]
  boxRange?: { is_box: boolean; top?: number; bottom?: number }
  onLoadMore?: () => void
  freshLoadId?: number
}) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const predictedSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const supportPriceLinesRef = useRef<IPriceLine[]>([])
  const resistancePriceLinesRef = useRef<IPriceLine[]>([])
  const boxPriceLinesRef = useRef<IPriceLine[]>([])

  // load-more 관련 ref (stale closure 방지)
  const onLoadMoreRef = useRef<(() => void) | undefined>(undefined)
  const loadMoreTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // fresh load vs load-more 구분
  const prevFreshLoadIdRef = useRef<number>(-1)
  const prevSeriesLengthRef = useRef<number>(0)

  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  useEffect(() => {
    onLoadMoreRef.current = props.onLoadMore
  }, [props.onLoadMore])

  const seriesData = useMemo(() => {
    try {
      if (!props.candles?.length) return []
      const result = parseAndDedup(props.candles)
      if (result.length === 0 && props.candles.length > 0) {
        throw new Error(`데이터 형식이 맞지 않습니다.\n샘플:\n${JSON.stringify(props.candles[0], null, 2)}`)
      }
      return result
    } catch (e) {
      setErrorMsg(toMessage(e))
      return []
    }
  }, [props.candles])

  const predictedData = useMemo(() => {
    if (!props.predictedCandles?.length) return []
    return parseAndDedup(props.predictedCandles)
  }, [props.predictedCandles])

  // ── 차트 초기화 (마운트 1회) ─────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    try {
      const isDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
      const chart = createChart(el, {
        autoSize: true,
        layout: {
          background: { color: isDark ? '#111827' : '#ffffff' },
          textColor: isDark ? '#e5e7eb' : '#111827',
        },
        grid: {
          vertLines: { color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.08)' },
          horzLines: { color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.08)' },
        },
        rightPriceScale: { borderVisible: false },
        timeScale: { borderVisible: false, rightOffset: 10 },
        crosshair: { mode: 1 },
      })
      chartRef.current = chart

      // 실제 캔들 시리즈
      seriesRef.current = chart.addSeries(CandlestickSeries, {
        upColor: '#22c55e',
        downColor: '#ef4444',
        borderUpColor: '#22c55e',
        borderDownColor: '#ef4444',
        wickUpColor: '#22c55e',
        wickDownColor: '#ef4444',
      })

      // 예측 캔들 시리즈 (보라색)
      predictedSeriesRef.current = chart.addSeries(CandlestickSeries, {
        upColor: 'rgba(99, 102, 241, 0.75)',
        downColor: 'rgba(239, 68, 68, 0.55)',
        borderUpColor: 'rgba(99, 102, 241, 0.75)',
        borderDownColor: 'rgba(239, 68, 68, 0.55)',
        wickUpColor: 'rgba(99, 102, 241, 0.75)',
        wickDownColor: 'rgba(239, 68, 68, 0.55)',
      })

      // 스크롤 감지 → 왼쪽 끝 도달 시 과거 데이터 추가 로드
      chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (!range || !onLoadMoreRef.current) return
        if (range.from < 5) {
          if (loadMoreTimerRef.current) clearTimeout(loadMoreTimerRef.current)
          loadMoreTimerRef.current = setTimeout(() => {
            onLoadMoreRef.current?.()
          }, 350)
        }
      })
    } catch (e) {
      setErrorMsg('차트 라이브러리 초기화 실패: ' + toMessage(e))
    }

    return () => {
      if (loadMoreTimerRef.current) clearTimeout(loadMoreTimerRef.current)
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
        seriesRef.current = null
        predictedSeriesRef.current = null
      }
    }
  }, [])

  // ── 실제 캔들 데이터 업데이트 ─────────────────────────────────────
  useEffect(() => {
    const chart = chartRef.current
    const series = seriesRef.current
    if (!chart || !series || seriesData.length === 0) return

    try {
      const freshLoadId = props.freshLoadId ?? 0
      const isFreshLoad = freshLoadId !== prevFreshLoadIdRef.current

      if (isFreshLoad) {
        // 새 종목 / 새 날짜: 전체 fit
        series.setData(seriesData)
        chart.timeScale().fitContent()
        prevFreshLoadIdRef.current = freshLoadId
      } else {
        // 과거 데이터 추가 로드: 스크롤 위치 유지
        const addedCount = seriesData.length - prevSeriesLengthRef.current
        const prevRange = chart.timeScale().getVisibleLogicalRange()
        series.setData(seriesData)
        if (prevRange && addedCount > 0) {
          chart.timeScale().setVisibleLogicalRange({
            from: prevRange.from + addedCount,
            to: prevRange.to + addedCount,
          })
        }
      }

      prevSeriesLengthRef.current = seriesData.length
      setErrorMsg(null)
    } catch (err) {
      setErrorMsg(`차트에 데이터를 그리는 중 오류: ${toMessage(err)}`)
    }
  }, [seriesData, props.freshLoadId])

  // ── 예측 캔들 업데이트 ──────────────────────────────────────────
  useEffect(() => {
    const series = predictedSeriesRef.current
    if (!series) return
    try {
      series.setData(predictedData)
      // 예측 캔들이 생기면 차트를 오른쪽으로 확장해서 보여줌
      if (predictedData.length > 0) {
        chartRef.current?.timeScale().fitContent()
      }
    } catch (e) {
      console.error('예측 캔들 오류:', toMessage(e))
    }
  }, [predictedData])

  // ── 지지/저항/박스 라인 업데이트 ────────────────────────────────
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

      const isValidPrice = (p: unknown): p is number =>
        typeof p === 'number' && Number.isFinite(p) && p > 0

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
        for (const [price, title] of [[box.top, '박스 상단'], [box.bottom, '박스 하단']] as [number, string][]) {
          boxPriceLinesRef.current.push(
            series.createPriceLine({ price, color: '#f59e0b', lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true, title })
          )
        }
      }
    } catch (e) {
      console.error('라인 오버레이 실패:', toMessage(e))
    }
  }, [props.supportLines, props.resistanceLines, props.boxRange])

  return (
    <div style={{ position: 'relative', width: '100%', minHeight: '520px', border: '1px solid #e5e7eb', borderRadius: '8px', overflow: 'hidden', backgroundColor: '#f9fafb' }}>
      {errorMsg && (
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'rgba(0,0,0,0.85)', color: '#ff8a8a', padding: '20px', zIndex: 50, overflow: 'auto', fontSize: '15px', whiteSpace: 'pre-wrap' }}>
          <h3 style={{ fontWeight: 'bold', marginBottom: '10px', color: '#ff4d4d' }}>🚨 원인 발견</h3>
          {errorMsg}
        </div>
      )}
      {/* 예측 구간 범례 */}
      {predictedData.length > 0 && (
        <div style={{ position: 'absolute', top: 8, right: 8, zIndex: 10, display: 'flex', alignItems: 'center', gap: 6, backgroundColor: 'rgba(0,0,0,0.55)', borderRadius: 6, padding: '4px 10px', fontSize: 12, color: '#a5b4fc', pointerEvents: 'none' }}>
          <span style={{ width: 10, height: 10, backgroundColor: 'rgba(99,102,241,0.8)', borderRadius: 2, display: 'inline-block' }} />
          AI 예측 구간 (7 영업일)
        </div>
      )}
      <div ref={containerRef} style={{ width: '100%', height: '520px' }} />
    </div>
  )
}
