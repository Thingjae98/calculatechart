import {
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  LineStyle,
  createChart,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
} from 'lightweight-charts'
import { useEffect, useMemo, useRef, useState } from 'react'

export interface Candle {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

const toMessage = (e: unknown): string =>
  e instanceof Error ? e.message : String(e)

function parseAndDedup(candles: Candle[], keepVolume = false) {
  const parsePrice = (v: unknown): number =>
    Number(String(v ?? '').replace(/,/g, ''))

  const valid = candles
    .map((c) => {
      const timeStr = (typeof c.time === 'string' ? c.time : '').substring(0, 10)
      const open = parsePrice(c.open)
      const high = parsePrice(c.high)
      const low = parsePrice(c.low)
      const close = parsePrice(c.close)
      if (
        !Number.isFinite(open) ||
        !Number.isFinite(high) ||
        !Number.isFinite(low) ||
        !Number.isFinite(close) ||
        !/^\d{4}-\d{2}-\d{2}$/.test(timeStr)
      )
        return null
      const out: { time: string; open: number; high: number; low: number; close: number; volume?: number } = { time: timeStr, open, high, low, close }
      if (keepVolume && typeof c.volume === 'number' && Number.isFinite(c.volume)) {
        out.volume = c.volume
      }
      return out
    })
    .filter((x): x is NonNullable<typeof x> => Boolean(x))

  valid.sort((a, b) => a.time.localeCompare(b.time))

  const unique: typeof valid = []
  let lastTime = ''
  for (const c of valid) {
    if (c.time !== lastTime) {
      unique.push(c)
      lastTime = c.time
    }
  }
  return unique
}

// 단순이동평균 계산
function calcSMA(data: { time: string; close: number }[], period: number) {
  const result: { time: string; value: number }[] = []
  for (let i = period - 1; i < data.length; i++) {
    let sum = 0
    for (let j = i - period + 1; j <= i; j++) sum += data[j].close
    result.push({ time: data[i].time, value: sum / period })
  }
  return result
}

const MA_CONFIG = [
  { period: 5, color: '#eab308', label: '5일' },
  { period: 20, color: '#f97316', label: '20일' },
  { period: 60, color: '#3b82f6', label: '60일' },
  { period: 120, color: '#a855f7', label: '120일' },
] as const

export function CandleChart(props: {
  candles: Candle[]
  predictedCandles?: Candle[]
  supportLines?: number[]
  resistanceLines?: number[]
  boxRange?: { is_box: boolean; top?: number; bottom?: number }
  onLoadMore?: () => void
  freshLoadId?: number
  predDays?: number
}) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const predictedSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const maSeriesRefs = useRef<ISeriesApi<'Line'>[]>([])
  const supportPriceLinesRef = useRef<IPriceLine[]>([])
  const resistancePriceLinesRef = useRef<IPriceLine[]>([])
  const boxPriceLinesRef = useRef<IPriceLine[]>([])

  const onLoadMoreRef = useRef<(() => void) | undefined>(undefined)
  const loadMoreTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const prevFreshLoadIdRef = useRef<number>(-1)
  const prevSeriesLengthRef = useRef<number>(0)

  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  useEffect(() => {
    onLoadMoreRef.current = props.onLoadMore
  }, [props.onLoadMore])

  // ── 데이터 파싱 ──────────────────────────────────────────────────
  const seriesData = useMemo(() => {
    try {
      if (!props.candles?.length) return []
      const result = parseAndDedup(props.candles, true)
      if (result.length === 0 && props.candles.length > 0) {
        throw new Error(
          `데이터 형식이 맞지 않습니다.\n샘플:\n${JSON.stringify(props.candles[0], null, 2)}`,
        )
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

  // ── 차트 초기화 ──────────────────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    try {
      const isDark =
        window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
      const chart = createChart(el, {
        autoSize: true,
        layout: {
          background: { color: isDark ? '#111827' : '#ffffff' },
          textColor: isDark ? '#e5e7eb' : '#374151',
        },
        grid: {
          vertLines: {
            color: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.06)',
          },
          horzLines: {
            color: isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.06)',
          },
        },
        rightPriceScale: {
          borderVisible: false,
          scaleMargins: { top: 0.05, bottom: 0.22 },
        },
        timeScale: { borderVisible: false, rightOffset: 12 },
        crosshair: { mode: 1 },
      })
      chartRef.current = chart

      // 캔들 시리즈
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
        upColor: 'rgba(99, 102, 241, 0.8)',
        downColor: 'rgba(239, 68, 68, 0.55)',
        borderUpColor: 'rgba(99, 102, 241, 0.8)',
        borderDownColor: 'rgba(239, 68, 68, 0.55)',
        wickUpColor: 'rgba(99, 102, 241, 0.8)',
        wickDownColor: 'rgba(239, 68, 68, 0.55)',
      })

      // 거래량 히스토그램 시리즈 (하단 20%)
      volumeSeriesRef.current = chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      })
      chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.82, bottom: 0 },
      })

      // 이동평균선 시리즈 (5, 20, 60, 120일)
      maSeriesRefs.current = MA_CONFIG.map((ma) =>
        chart.addSeries(LineSeries, {
          color: ma.color,
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        }),
      )

      // 스크롤 감지
      chart
        .timeScale()
        .subscribeVisibleLogicalRangeChange((range) => {
          if (!range || !onLoadMoreRef.current) return
          if (range.from < 5) {
            if (loadMoreTimerRef.current)
              clearTimeout(loadMoreTimerRef.current)
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
        volumeSeriesRef.current = null
        maSeriesRefs.current = []
      }
    }
  }, [])

  // ── 캔들 + 거래량 + 이평선 데이터 업데이트 ──────────────────────
  useEffect(() => {
    const chart = chartRef.current
    const series = seriesRef.current
    if (!chart || !series || seriesData.length === 0) return

    try {
      const freshLoadId = props.freshLoadId ?? 0
      const isFreshLoad = freshLoadId !== prevFreshLoadIdRef.current

      // 캔들
      const candleOnly = seriesData.map(({ time, open, high, low, close }) => ({
        time,
        open,
        high,
        low,
        close,
      }))

      if (isFreshLoad) {
        series.setData(candleOnly)
        chart.timeScale().fitContent()
        prevFreshLoadIdRef.current = freshLoadId
      } else {
        const addedCount = seriesData.length - prevSeriesLengthRef.current
        const prevRange = chart.timeScale().getVisibleLogicalRange()
        series.setData(candleOnly)
        if (prevRange && addedCount > 0) {
          chart.timeScale().setVisibleLogicalRange({
            from: prevRange.from + addedCount,
            to: prevRange.to + addedCount,
          })
        }
      }

      // 거래량
      if (volumeSeriesRef.current) {
        const volData = seriesData
          .filter((d) => typeof d.volume === 'number' && d.volume > 0)
          .map((d) => ({
            time: d.time,
            value: d.volume!,
            color:
              d.close >= d.open
                ? 'rgba(34, 197, 94, 0.35)'
                : 'rgba(239, 68, 68, 0.35)',
          }))
        volumeSeriesRef.current.setData(volData)
      }

      // 이동평균선
      const closeData = seriesData.map((d) => ({
        time: d.time,
        close: d.close,
      }))
      MA_CONFIG.forEach((ma, idx) => {
        const maSeries = maSeriesRefs.current[idx]
        if (maSeries) {
          maSeries.setData(calcSMA(closeData, ma.period))
        }
      })

      prevSeriesLengthRef.current = seriesData.length
      setErrorMsg(null)
    } catch (err) {
      setErrorMsg(`차트 데이터 오류: ${toMessage(err)}`)
    }
  }, [seriesData, props.freshLoadId])

  // ── 예측 캔들 ────────────────────────────────────────────────────
  useEffect(() => {
    const series = predictedSeriesRef.current
    if (!series) return
    try {
      series.setData(predictedData)
      if (predictedData.length > 0) {
        chartRef.current?.timeScale().fitContent()
      }
    } catch (e) {
      console.error('예측 캔들 오류:', toMessage(e))
    }
  }, [predictedData])

  // ── 지지/저항/박스 라인 ──────────────────────────────────────────
  useEffect(() => {
    const series = seriesRef.current
    if (!series) return

    try {
      for (const line of supportPriceLinesRef.current)
        series.removePriceLine(line)
      for (const line of resistancePriceLinesRef.current)
        series.removePriceLine(line)
      for (const line of boxPriceLinesRef.current)
        series.removePriceLine(line)
      supportPriceLinesRef.current = []
      resistancePriceLinesRef.current = []
      boxPriceLinesRef.current = []

      const isValid = (p: unknown): p is number =>
        typeof p === 'number' && Number.isFinite(p) && p > 0

      for (const price of props.supportLines ?? []) {
        if (isValid(price)) {
          supportPriceLinesRef.current.push(
            series.createPriceLine({
              price,
              color: '#2563eb',
              lineWidth: 1,
              lineStyle: LineStyle.Dotted,
              axisLabelVisible: true,
              title: '지지',
            }),
          )
        }
      }

      for (const price of props.resistanceLines ?? []) {
        if (isValid(price)) {
          resistancePriceLinesRef.current.push(
            series.createPriceLine({
              price,
              color: '#ef4444',
              lineWidth: 1,
              lineStyle: LineStyle.Dotted,
              axisLabelVisible: true,
              title: '저항',
            }),
          )
        }
      }

      const box = props.boxRange
      if (box?.is_box && isValid(box.top) && isValid(box.bottom)) {
        for (const [price, title] of [
          [box.top, '박스 상단'],
          [box.bottom, '박스 하단'],
        ] as [number, string][]) {
          boxPriceLinesRef.current.push(
            series.createPriceLine({
              price,
              color: '#f59e0b',
              lineWidth: 1,
              lineStyle: LineStyle.Dotted,
              axisLabelVisible: true,
              title,
            }),
          )
        }
      }
    } catch (e) {
      console.error('라인 오버레이 실패:', toMessage(e))
    }
  }, [props.supportLines, props.resistanceLines, props.boxRange])

  const predDaysLabel = props.predDays ?? predictedData.length

  return (
    <div className="chartWrap">
      {errorMsg && (
        <div className="chartError">
          <strong>차트 오류</strong>
          <pre>{errorMsg}</pre>
        </div>
      )}
      {/* 범례 */}
      <div className="chartLegend">
        {MA_CONFIG.map((ma) => (
          <span key={ma.period} style={{ color: ma.color }}>
            ― {ma.label}
          </span>
        ))}
        {predictedData.length > 0 && (
          <span style={{ color: '#818cf8' }}>
            ■ AI {predDaysLabel}일 예측
          </span>
        )}
      </div>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  )
}
