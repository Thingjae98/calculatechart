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

export function CandleChart(props: {
  candles: Candle[]
  supportLines?: number[]
  resistanceLines?: number[]
  boxRange?: { is_box: boolean; top?: number; bottom?: number }
}) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const supportPriceLinesRef = useRef<IPriceLine[]>([])
  const resistancePriceLinesRef = useRef<IPriceLine[]>([])
  const boxPriceLinesRef = useRef<IPriceLine[]>([])
  
  // 에러 원인을 화면에 직접 띄워줄 상태
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const seriesData = useMemo(() => {
    try {
      if (!props.candles || !Array.isArray(props.candles) || props.candles.length === 0) return []

      const parsePrice = (v: unknown): number => Number(String(v ?? '').replace(/,/g, ''));

      const validCandles = props.candles.map((c: Candle) => {
        const rawTime = typeof c.time === 'string' ? c.time : '';
        const timeStr = rawTime.substring(0, 10);

        const open = parsePrice(c.open);
        const high = parsePrice(c.high);
        const low = parsePrice(c.low);
        const close = parsePrice(c.close);

        if (!Number.isFinite(open) || !Number.isFinite(high) || !Number.isFinite(low) || !Number.isFinite(close) || !/^\d{4}-\d{2}-\d{2}$/.test(timeStr)) {
          return null;
        }
        return { time: timeStr, open, high, low, close };
      }).filter((x): x is NonNullable<typeof x> => Boolean(x));

      if (validCandles.length === 0 && props.candles.length > 0) {
        throw new Error(`데이터 형식이 맞지 않아 캔들을 그릴 수 없습니다.\n원본 데이터 샘플:\n${JSON.stringify(props.candles[0], null, 2)}`);
      }

      validCandles.sort((a, b) => a.time.localeCompare(b.time));
      const uniqueCandles = [];
      let lastTime = '';
      for (const c of validCandles) {
        if (c.time !== lastTime) {
          uniqueCandles.push(c);
          lastTime = c.time;
        }
      }

      return uniqueCandles;
    } catch (e: unknown) {
      setErrorMsg(toMessage(e));
      return [];
    }
  }, [props.candles])

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
    } catch (e: unknown) {
      setErrorMsg("차트 라이브러리 초기화 실패: " + toMessage(e))
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
      setErrorMsg(null)
    } catch (err: unknown) {
      setErrorMsg(`차트에 데이터를 그리는 중 에러 발생!\n내용: ${toMessage(err)}`)
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
        const boxLines: Array<[number, string]> = [
          [box.top, '박스 상단'],
          [box.bottom, '박스 하단'],
        ]
        for (const [price, title] of boxLines) {
          boxPriceLinesRef.current.push(
            series.createPriceLine({ price, color: '#f59e0b', lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true, title })
          )
        }
      }
    } catch (e: unknown) {
      console.error("라인 오버레이 실패:", toMessage(e))
    }
  }, [props.supportLines, props.resistanceLines, props.boxRange])

  return (
    <div style={{ position: 'relative', width: '100%', minHeight: '520px', border: '1px solid #e5e7eb', borderRadius: '8px', overflow: 'hidden', backgroundColor: '#f9fafb' }}>
      {errorMsg && (
        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'rgba(0, 0, 0, 0.85)', color: '#ff8a8a', padding: '20px', zIndex: 50, overflow: 'auto', fontSize: '15px', whiteSpace: 'pre-wrap' }}>
          <h3 style={{fontWeight: 'bold', marginBottom: '10px', color: '#ff4d4d'}}>🚨 원인 발견</h3>
          {errorMsg}
        </div>
      )}
      <div ref={containerRef} style={{ width: '100%', height: '520px' }} />
    </div>
  )
}