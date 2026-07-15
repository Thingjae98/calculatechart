// 데일리 마켓 브리핑 데이터 타입 + 로더
// 데이터는 정적 JSON(/briefings/latest.json)으로 제공된다.
// 매일 아침 GitHub Action(스케줄)이 웹 조사 → JSON 생성 → 커밋 → Vercel 자동배포.

export type Tone = 'up' | 'down' | 'flat'

/** 지수/개별종목 등락 항목 */
export interface BriefingQuote {
  name: string
  /** 표시용 값 (예: "5,637.08"). 숫자 포맷은 생성 단계에서 완료 */
  value: string
  /** 전일 대비 등락률(%). 모르면 null */
  change_pct: number | null
}

/** 매크로 지표 (환율/금리/유가 등) — 등락 표기가 %가 아닐 수 있어 문자열 change 사용 */
export interface BriefingMacro {
  name: string
  value: string
  /** 표시용 변화 문자열 (예: "+3.2원", "-0.04%p", "+1.1%"). 모르면 null */
  change: string | null
  tone: Tone
}

/** 반도체 투톱 개별 전망 */
export interface SemiOutlook {
  name: string
  ticker: string
  /** "상승" | "하락" | "보합/횡보" 등 자유 문자열 */
  direction: string
  tone: Tone
  /** 핵심 모멘텀/리스크 */
  momentum: string
  /** 오늘의 단기 전략 */
  strategy: string
}

/**
 * 시가총액 상위 종목 카드.
 *
 * 숫자(close/change_pct/support/resistance)는 백엔드 알고리즘이 계산해 워크플로우가
 * 주입한다 — 모델이 쓴 값을 덮어쓰므로 지어낸 숫자가 들어갈 수 없다.
 * comment만 모델이 웹 조사로 작성한다.
 */
export interface TopCapItem {
  ticker: string
  name: string
  /** 'KOSPI' | 'KOSDAQ' */
  market: string
  /** 표시용 종가 (예: "205,000") */
  close: string
  /** 전일 대비 등락률(%). 모르면 null */
  change_pct: number | null
  /** 표시용 지지선 (예: "198,000"). 탐지 실패 시 null */
  support: string | null
  /** 표시용 저항선. 탐지 실패 시 null */
  resistance: string | null
  /** 쉬운 말 한 줄 코멘트 (모델 작성) */
  comment?: string
}

export interface DailyBriefing {
  /** 'YYYY-MM-DD' (KST 기준 브리핑 대상일) */
  date: string
  /** ISO8601 생성 시각 */
  generated_at: string
  /** 샘플/플레이스홀더 데이터이면 true — UI에 경고 배너 표시 */
  is_sample?: boolean

  // 1. 글로벌 마켓 서머리
  global: {
    /** S&P500 / Nasdaq / Dow */
    indices: BriefingQuote[]
    sox: {
      value: string
      change_pct: number | null
      /** NVDA, TSM 등 핵심 반도체 개별주 */
      components: BriefingQuote[]
    }
    /** 원/달러, 미 10년물, WTI 등 */
    macro: BriefingMacro[]
    /** 글로벌 핵심 한 줄 뉴스 2~3개 */
    headlines: string[]
  }

  // 2. 국내 시장 방향성
  domestic: {
    /** "상승 우세" | "하락 우세" | "보합/관망" 등 */
    direction: string
    tone: Tone
    /** 근거 및 시나리오 (수급/매크로) */
    rationale: string[]
    /** 국내 핵심 뉴스 2~3개 (거시경제·정책·대기업·반도체/IT) — optional */
    headlines?: string[]
    /** 지지선/저항선·트리거 한 줄 */
    levels: string
  }

  // 3. 반도체 투톱
  semis: {
    samsung: SemiOutlook
    hynix: SemiOutlook
  }

  /**
   * 4. 시가총액 상위 종목 (코스피 5 + 코스닥 5)
   * optional — 백엔드(Render)가 콜드/다운이면 워크플로우가 이 섹션을 생략한다.
   * 브리핑 본문은 백엔드와 무관하게 항상 생성되어야 하므로 없으면 섹션을 숨긴다.
   */
  top_caps?: {
    kospi: TopCapItem[]
    kosdaq: TopCapItem[]
    /** 데이터 기준 영업일 'YYYY-MM-DD' */
    as_of?: string
  }

  // 5. 개장 후 관전 포인트
  checklist: string[]
}

/** 브리핑 로드 결과 */
export interface BriefingLoad {
  data: DailyBriefing | null
  error: string | null
}

/**
 * 최신 브리핑 로드. Vite BASE_URL 기준 정적 파일.
 * NetworkFirst 서비스워커 캐시가 오프라인 폴백을 담당한다.
 */
export async function fetchLatestBriefing(): Promise<BriefingLoad> {
  const url = `${import.meta.env.BASE_URL}briefings/latest.json`
  try {
    const res = await fetch(url, { cache: 'no-cache' })
    if (!res.ok) return { data: null, error: `브리핑을 불러오지 못했습니다 (${res.status})` }
    const body = (await res.json()) as DailyBriefing
    if (!body || typeof body.date !== 'string' || !body.global) {
      return { data: null, error: '브리핑 형식이 올바르지 않습니다.' }
    }
    return { data: body, error: null }
  } catch {
    return { data: null, error: '브리핑을 불러올 수 없습니다. 네트워크를 확인하세요.' }
  }
}

/** 등락률 → 부호 색상 tone */
export function pctTone(pct: number | null): Tone {
  if (pct == null) return 'flat'
  if (pct > 0) return 'up'
  if (pct < 0) return 'down'
  return 'flat'
}

/** 등락률 표시 문자열 (+1.23% / -0.45%) */
export function fmtPct(pct: number | null): string {
  if (pct == null) return '—'
  const sign = pct > 0 ? '+' : ''
  return `${sign}${pct.toFixed(2)}%`
}
