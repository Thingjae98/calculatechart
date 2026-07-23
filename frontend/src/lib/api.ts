export type Candle = {
  time: string // 'YYYY-MM-DD'
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export type RawCandle = {
  time?: string
  date?: string
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export type BoxRange = {
  is_box: boolean
  top?: number
  bottom?: number
}

export type FibonacciLevels = {
  swing_high: number
  swing_low: number
  levels: Record<string, number>  // e.g. {"0.382": 12345, "0.618": 15678, ...}
  direction: 'up' | 'down'
}

export type IchimokuValues = {
  tenkan: number        // 전환선 (9일)
  kijun: number         // 기준선 (26일)
  senkou_a: number      // 선행스팬1
  senkou_b: number      // 선행스팬2
  chikou: number        // 후행스팬 (현재 종가)
  cloud_top: number
  cloud_bottom: number
  cloud_bullish: boolean
  price_above_cloud: boolean
  price_below_cloud: boolean
  tenkan_above_kijun: boolean
}

export type OhlcvResponse = {
  ticker: string
  data: RawCandle[]
  stock_name?: string
  support_lines?: number[]
  resistance_lines?: number[]
  box_range?: BoxRange
  fibonacci?: FibonacciLevels
  ichimoku?: IchimokuValues
  score?: number
  score_breakdown?: Record<string, number>
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

export interface SellTargets {
  short_term: number | null
  long_term: number | null
  stop_loss: number | null
  short_term_desc: string | null
  long_term_desc: string | null
  stop_loss_desc: string | null
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
  sell_targets?: SellTargets
  predicted_candles?: Candle[]
  error?: string
}

export type SearchResult = {
  ticker: string
  name: string
}

export type SearchResponse = {
  results: SearchResult[]
  error?: string
}

/** 종목 자동완성 검색 */
export async function searchStocks(query: string): Promise<SearchResponse> {
  if (!query.trim()) return { results: [] }
  try {
    const res = await fetch(buildApiUrl(`/api/search?q=${encodeURIComponent(query.trim())}`))
    if (!res.ok) return { results: [], error: `서버 응답 오류 (${res.status})` }
    const body = await res.json()
    const results = Array.isArray(body?.results) ? body.results : []
    return { results }
  } catch (e) {
    const msg = e instanceof TypeError && String(e).includes('fetch')
      ? '서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인하세요.'
      : '검색 중 오류가 발생했습니다.'
    return { results: [], error: msg }
  }
}

/**
 * 응답 JSON 파싱 가드.
 * 백엔드가 콜드/프록시 오류로 HTML 등 JSON 아닌 응답을 주면 res.json()이
 * "Failed to execute 'json' on 'Response'" 같은 영어 기술 에러를 던지는데,
 * 이 앱의 주 사용자는 비전문가라 그 문구가 화면에 그대로 노출되면 안 된다.
 */
async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  try {
    return (await res.json()) as T
  } catch {
    throw new Error('서버가 잠시 응답하지 못했습니다. 잠시 후 다시 시도해주세요.')
  }
}

/** Render 무료 플랜 cold start 해결용: 서버를 미리 깨움 */
export async function pingServer(): Promise<boolean> {
  try {
    const res = await fetch(buildApiUrl('/api/ping'), { method: 'GET' })
    return res.ok
  } catch {
    return false
  }
}

/**
 * 주가 예측 요청.
 * 백엔드가 자체적으로 365일 데이터를 조회하므로 날짜 파라미터 불필요.
 */
export async function fetchPrediction(
  ticker: string,
  nDays: number = 7,
): Promise<PredictionResult> {
  const params = new URLSearchParams({ n_days: String(nDays) })
  const url = buildApiUrl(`/api/stock/${encodeURIComponent(ticker)}/predict?${params}`)
  let res: Response
  try {
    res = await fetch(url)
  } catch {
    throw new Error('서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인하세요.')
  }
  if (!res.ok) throw new Error(`서버 오류가 발생했습니다 (${res.status}). 잠시 후 다시 시도해주세요.`)
  const body = await parseJsonOrThrow<PredictionResult>(res)
  if (body?.error) throw new Error(body.error)
  return body
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() ?? ''

function buildApiUrl(path: string) {
  if (!API_BASE_URL) return path
  const base = API_BASE_URL.endsWith('/') ? API_BASE_URL.slice(0, -1) : API_BASE_URL
  const p = path.startsWith('/') ? path : `/${path}`
  return `${base}${p}`
}

/**
 * OHLCV 데이터 조회.
 *
 * 두 가지 모드:
 * 1. 초기 조회: ticker만 전달 → 백엔드가 최근 365일 데이터 + 점수 반환
 * 2. 스크롤 추가 로드: ticker + before_date + days_back → 지정 날짜 이전 N일 데이터 반환
 */
export async function fetchOhlcv(args: {
  ticker: string
  before_date?: string   // 'YYYY-MM-DD' — 이 날짜 이전 데이터 요청 (스크롤 로드용)
  days_back?: number     // before_date 기준 며칠 이전까지 (기본 180)
}) {
  const params: Record<string, string> = {}
  if (args.before_date) {
    params.before_date = args.before_date
  }
  if (args.days_back) {
    params.days_back = String(args.days_back)
  }

  const query = new URLSearchParams(params).toString()
  const path = `/api/stock/${encodeURIComponent(args.ticker)}${query ? `?${query}` : ''}`
  const url = buildApiUrl(path)

  let res: Response
  try {
    res = await fetch(url, { method: 'GET' })
  } catch {
    throw new Error('서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인하세요.')
  }
  const body = await parseJsonOrThrow<OhlcvResponse>(res)

  if (!res.ok) {
    throw new Error(body?.error ?? `서버 오류가 발생했습니다 (${res.status}). 잠시 후 다시 시도해주세요.`)
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

  const boxRangeRaw = body.box_range as Partial<BoxRange> | undefined
  const boxRange: BoxRange =
    boxRangeRaw && typeof boxRangeRaw.is_box === 'boolean'
      ? {
          is_box: boxRangeRaw.is_box,
          top: typeof boxRangeRaw.top === 'number' ? boxRangeRaw.top : undefined,
          bottom: typeof boxRangeRaw.bottom === 'number' ? boxRangeRaw.bottom : undefined,
        }
      : { is_box: false }

  const fibRaw = body.fibonacci as Partial<FibonacciLevels> | undefined
  const fibonacci: FibonacciLevels | undefined =
    fibRaw && typeof fibRaw.swing_high === 'number' && fibRaw.levels
      ? (fibRaw as FibonacciLevels)
      : undefined

  const ichiRaw = body.ichimoku as Partial<IchimokuValues> | undefined
  const ichimoku: IchimokuValues | undefined =
    ichiRaw && typeof ichiRaw.tenkan === 'number' && typeof ichiRaw.kijun === 'number'
      ? (ichiRaw as IchimokuValues)
      : undefined

  return {
    ticker: body.ticker,
    stock_name: body.stock_name ?? body.ticker,
    data: normalized,
    support_lines: supportLines,
    resistance_lines: resistanceLines,
    box_range: boxRange,
    fibonacci,
    ichimoku,
    score: typeof body.score === 'number' ? body.score : 0,
    score_breakdown: body.score_breakdown ?? {},
  }
}

function isRecommendationItem(x: unknown): x is RecommendationItem {
  if (!x || typeof x !== 'object') return false
  const r = x as Record<string, unknown>
  return (
    typeof r.ticker === 'string' &&
    typeof r.stock_name === 'string' &&
    typeof r.score === 'number'
  )
}

export async function fetchRecommendations(limit = 10): Promise<RecommendationItem[]> {
  let res: Response
  try {
    res = await fetch(buildApiUrl(`/api/recommendations?limit=${limit}`), { method: 'GET' })
  } catch {
    // 다른 API와 동일하게 네트워크 실패도 한국어로 — 원시 'Failed to fetch' 노출 방지
    throw new Error('서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.')
  }
  const body = await parseJsonOrThrow<{ top?: unknown; error?: string }>(res)
  if (!res.ok) throw new Error(body?.error ?? `서버 오류가 발생했습니다 (${res.status}). 잠시 후 다시 시도해주세요.`)
  if (body?.error) throw new Error(body.error)
  if (!Array.isArray(body?.top)) return []

  return body.top.filter(isRecommendationItem)
}

// ── '종목 현재 상황' 모달 ──────────────────────────────────
// 브리핑(정적 JSON, 하루 1회)과 달리 버튼을 누른 시각에 백엔드가 계산한다.
// 요약(summary)은 LLM이 아니라 백엔드 규칙이 만든다 — 같은 입력이면 같은 문장.

export type StockNowSignal = {
  type: 'positive' | 'negative' | 'neutral'
  label: string
  desc?: string
}

export type StockNowNews = {
  title: string
  summary: string
  link: string
  published: string
}

export type StockNow = {
  ticker: string
  name: string
  market: string
  is_etf: boolean
  /** 조회를 누른 시각 'YYYY-MM-DD HH:mm' */
  checked_at: string
  /**
   * 데이터 기준 영업일 'YYYY-MM-DD'. 장중이면 당일 날짜가 들어오고 값도 갱신된다
   * (2026-07-23 10:42 실측: 75초 간격 조회에서 종가·거래량 모두 변동).
   * 다만 체결 즉시 값은 아니므로 화면에는 '몇 분 전 시세일 수 있음'으로 표기한다.
   */
  as_of: string
  close: string
  close_raw: number
  change_pct: number | null
  support: string | null
  resistance: string | null
  score: number
  signals: StockNowSignal[]
  news: StockNowNews[]
  summary: { headline: string; detail: string; caution: string }
  /** 'ai' = Gemini가 뉴스까지 읽고 씀 / 'rule' = 숫자만 보고 규칙으로 만듦 */
  summary_source: 'ai' | 'rule'
  /** false면 네이버 API 키가 없어 뉴스 칸이 비어 있다는 뜻 */
  news_enabled: boolean
}

export async function fetchStockNow(query: string): Promise<StockNow> {
  let res: Response
  try {
    res = await fetch(buildApiUrl(`/api/stock/${encodeURIComponent(query.trim())}/now`), {
      method: 'GET',
    })
  } catch {
    throw new Error('서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.')
  }
  const body = await parseJsonOrThrow<StockNow & { error?: string }>(res)
  if (!res.ok) {
    throw new Error(body?.error ?? `서버 오류가 발생했습니다 (${res.status}). 잠시 후 다시 시도해주세요.`)
  }
  if (body?.error) throw new Error(body.error)
  return body
}
