export type Candle = {
  time: string // 'YYYY-MM-DD'
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

type RawCandle = {
  time?: string
  date?: string
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export type OhlcvResponse = {
  ticker: string
  data: unknown
  stock_name?: string
  support_lines?: unknown
  resistance_lines?: unknown
  box_range?: unknown
  score?: number
  score_breakdown?: unknown
  error?: string
}

export type RecommendationItem = {
  ticker: string
  stock_name: string
  score: number
}
export interface PredictionSignal {
  type: 'positive' | 'negative' | 'neutral'
  label: string
  desc: string
}

export interface PredictionResult {
  ticker: string
  stock_name: string
  current_price: number
  prediction_score: number
  outlook_short: string
  outlook_mid: string
  summary: string
  signals: PredictionSignal[]
  error?: string
}

export async function fetchPrediction(
  ticker: string,
  startDate: string,
  endDate: string,
): Promise<PredictionResult> {
  const params = new URLSearchParams({ start_date: startDate, end_date: endDate })
  const url = buildApiUrl(`/api/stock/${encodeURIComponent(ticker)}/predict?${params}`)
  const res = await fetch(url)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const body = await res.json()
  if (body?.error) throw new Error(body.error)
  return body
}

function toQuery(params: Record<string, string>) {
  const usp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) usp.set(k, v)
  return usp.toString()
}
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() ?? ''

function buildApiUrl(path: string) {
  if (!API_BASE_URL) return path
  const base = API_BASE_URL.endsWith('/') ? API_BASE_URL.slice(0, -1) : API_BASE_URL
  const p = path.startsWith('/') ? path : `/${path}`
  return `${base}${p}`
}

export async function fetchOhlcv(args: {
  ticker: string
  start_date: string
  end_date: string
}) {
  const url = buildApiUrl(`/api/stock/${encodeURIComponent(args.ticker)}?${toQuery({
    start_date: args.start_date,
    end_date: args.end_date,
  })}`)

  const res = await fetch(url, { method: 'GET' })
  const body = (await res.json()) as OhlcvResponse

  if (!res.ok) {
    throw new Error(body?.error ?? `HTTP ${res.status}`)
  }
  if (body?.error) {
    throw new Error(body.error)
  }
  if (!body || !('data' in body) || !Array.isArray(body.data)) {
    throw new Error('서버 응답 형식이 올바르지 않습니다.')
  }

  const normalized: Candle[] = []
  for (const c of body.data as RawCandle[]) {
    const time = typeof c.time === 'string' ? c.time : typeof c.date === 'string' ? c.date : null
    if (!time) continue

    const candle: Candle = { time, open: c.open, high: c.high, low: c.low, close: c.close }
    if (typeof c.volume === 'number' && Number.isFinite(c.volume)) {
      candle.volume = c.volume
    }
    normalized.push(candle)
  }

  const supportLines = Array.isArray(body.support_lines)
    ? (body.support_lines as unknown[]).map((v) => Number(v)).filter((n) => Number.isFinite(n))
    : []
  const resistanceLines = Array.isArray(body.resistance_lines)
    ? (body.resistance_lines as unknown[]).map((v) => Number(v)).filter((n) => Number.isFinite(n))
    : []

  const boxRangeRaw = body.box_range as Record<string, unknown> | undefined
  const boxRange =
    boxRangeRaw && typeof boxRangeRaw.is_box === 'boolean'
      ? {
          is_box: boxRangeRaw.is_box as boolean,
          top: typeof boxRangeRaw.top === 'number' ? (boxRangeRaw.top as number) : undefined,
          bottom: typeof boxRangeRaw.bottom === 'number' ? (boxRangeRaw.bottom as number) : undefined,
        }
      : { is_box: false as const }

  return {
    ticker: body.ticker,
    stock_name: body.stock_name ?? body.ticker,
    data: normalized,
    support_lines: supportLines,
    resistance_lines: resistanceLines,
    box_range: boxRange,
    score: typeof body.score === 'number' ? body.score : 0,
    score_breakdown: body.score_breakdown ?? {},
  }
}

export async function fetchRecommendations(limit = 10) {
  const res = await fetch(buildApiUrl(`/api/recommendations?limit=${limit}`), { method: 'GET' })
  const body = (await res.json()) as { top?: unknown; error?: string }
  if (!res.ok) throw new Error(body?.error ?? `HTTP ${res.status}`)
  if (body?.error) throw new Error(body.error)
  if (!Array.isArray(body?.top)) return []

  return body.top
    .map((x) => x as Record<string, unknown>)
    .filter((x) => typeof x.ticker === 'string' && typeof x.stock_name === 'string' && typeof x.score === 'number')
    .map(
      (x) =>
        ({
          ticker: x.ticker as string,
          stock_name: x.stock_name as string,
          score: x.score as number,
        }) as RecommendationItem,
    )

  
}

