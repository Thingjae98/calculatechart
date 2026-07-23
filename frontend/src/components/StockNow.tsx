import './StockNow.css'
import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import { createPortal } from 'react-dom'
import { fetchStockNow, type StockNow as StockNowData } from '../lib/api'

/**
 * '종목 현재 상황' 모달.
 *
 * 브리핑은 하루 한 번 굽는 정적 JSON이라 "지금 이 종목이 어떤지"에는 답하지 못한다.
 * 이 모달이 그 자리를 메운다 — 검색한 시각에 백엔드가 계산해서 바로 보여준다.
 *
 * 화면 원칙(CLAUDE.md): 검색창 하나, 결과는 큰 숫자 + 쉬운 말 3줄 + 신호 카드 + 뉴스.
 * 전문 용어는 쓰지 않는다.
 */

/** 점수 → 색 토큰 (ChartAnalysis의 scoreColor와 같은 기준) */
function scoreColor(score: number): string {
  if (score >= 65) return 'var(--score-good)'
  if (score >= 50) return 'var(--score-watch)'
  if (score >= 35) return 'var(--score-warn)'
  return 'var(--score-bad)'
}

function pctClass(pct: number | null): string {
  if (pct == null || pct === 0) return 'snFlat'
  return pct > 0 ? 'snUp' : 'snDown'
}

function fmtPct(pct: number | null): string {
  if (pct == null) return '—'
  return `${pct > 0 ? '+' : ''}${pct.toFixed(2)}%`
}

export function StockNow() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  // 로딩이 길어지면 문구를 바꾼다. Render 무료 플랜은 15분 무활동 시 슬립이라
  // 장 마감 후 첫 클릭은 콜드스타트로 30~60초가 걸린다 — 그냥 "분석 중"만 띄우면
  // 멈춘 걸로 오해하고 창을 닫는다.
  const [slow, setSlow] = useState(false)
  const [data, setData] = useState<StockNowData | null>(null)
  const [error, setError] = useState<string | null>(null)

  const inputRef = useRef<HTMLInputElement>(null)
  const openerRef = useRef<HTMLButtonElement>(null)
  const dialogRef = useRef<HTMLDivElement>(null)

  const close = useCallback(() => {
    setOpen(false)
    // 모달을 연 버튼으로 포커스를 돌려준다 — 키보드/스크린리더 사용자가 길을 잃지 않게
    openerRef.current?.focus()
  }, [])

  // Esc로 닫기 + 열려 있는 동안 배경 스크롤 잠금
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
    }
    document.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    inputRef.current?.focus()
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [open, close])

  const search = useCallback(async (e?: FormEvent) => {
    e?.preventDefault()
    const q = query.trim()
    if (!q || loading) return

    setLoading(true)
    setSlow(false)
    setError(null)
    setData(null)
    const slowTimer = window.setTimeout(() => setSlow(true), 8000)
    try {
      setData(await fetchStockNow(q))
    } catch (err) {
      setError(err instanceof Error ? err.message : '알 수 없는 오류가 발생했습니다.')
    } finally {
      window.clearTimeout(slowTimer)
      setLoading(false)
      setSlow(false)
    }
  }, [query, loading])

  return (
    <>
      <button
        ref={openerRef}
        className="snOpenBtn"
        onClick={() => setOpen(true)}
      >
        🔎 종목 현재 상황
      </button>

      {/*
        모달을 body로 포털한다. 이 버튼은 .topNav 안에 있는데 .topNav에는
        backdrop-filter가 걸려 있고, backdrop-filter는 자손의 position:fixed에
        대한 컨테이닝 블록을 만든다 — 즉 그대로 두면 "화면 전체"가 아니라
        "내비게이션 박스(높이 60px)" 기준으로 잡혀 모달이 상단에 짓눌려 잘린다.
        z-index를 올리는 걸로는 해결되지 않는다(기준점 자체가 틀린 문제).
      */}
      {open && createPortal(
        <div
          className="snBackdrop"
          onClick={(e) => {
            if (e.target === e.currentTarget) close()
          }}
        >
          <div
            className="snDialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="snTitle"
            ref={dialogRef}
          >
            <header className="snHeader">
              <h2 id="snTitle" className="snTitle">종목 현재 상황</h2>
              <button className="snClose" onClick={close} aria-label="닫기">✕</button>
            </header>

            <form className="snSearch" onSubmit={search}>
              <input
                ref={inputRef}
                className="snInput"
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="종목명·ETF·코드 (예: 삼성전자, KODEX 200)"
                aria-label="종목 또는 ETF 검색"
                enterKeyHint="search"
              />
              <button className="snGo" type="submit" disabled={loading || !query.trim()}>
                검색
              </button>
            </form>

            <div className="snBody">
              {!data && !error && !loading && (
                <p className="snHint">
                  종목이나 ETF 이름을 넣고 검색하면, <strong>지금 시점 기준</strong>으로
                  가격·흐름·뉴스를 정리해 드려요.
                </p>
              )}

              {error && <p className="snError">{error}</p>}

              {data && <StockNowResult data={data} />}
            </div>

            {/* 분석 중 오버레이 — 모달 안쪽만 덮는다 (닫기 버튼은 계속 눌러야 하므로 헤더는 제외) */}
            {loading && (
              <div className="snLoading" role="status" aria-live="polite">
                <div className="snSpinner" aria-hidden="true" />
                <p className="snLoadingText">분석 중이에요. 잠시만 기다려주세요…</p>
                {slow && (
                  <p className="snLoadingSub">
                    서버를 깨우는 중입니다. 처음 열 때는 1분까지 걸릴 수 있어요.
                  </p>
                )}
              </div>
            )}
          </div>
        </div>,
        document.body,
      )}
    </>
  )
}

function StockNowResult({ data }: { data: StockNowData }) {
  return (
    <div className="snResult">
      <div className="snNameRow">
        <span className="snName">{data.name}</span>
        <span className="snTicker">{data.ticker}</span>
        {data.is_etf && <span className="snBadge">ETF</span>}
      </div>

      <div className="snPriceRow">
        <span className="snPrice">{data.close}원</span>
        <span className={`snPct ${pctClass(data.change_pct)}`}>{fmtPct(data.change_pct)}</span>
        <span className="snScore" style={{ color: scoreColor(data.score) }}>
          {data.score}점
        </span>
      </div>

      <p className="snStamp">
        {data.checked_at} 조회 · {data.as_of} 종가 기준
        <span className="snStampNote">(장중에는 시세가 20분쯤 늦게 반영돼요)</span>
      </p>

      <section className="snSummary">
        <h3 className="snSummaryHead">{data.summary.headline}</h3>
        <p>{data.summary.detail}</p>
        <p className="snCaution">{data.summary.caution}</p>
        {/* 누가 쓴 문장인지 밝힌다 — 가족이 이걸 사람 말처럼 받아들이면 곤란하다 */}
        <p className="snSource">
          {data.summary_source === 'ai' ? 'AI가 뉴스까지 읽고 정리했어요' : '차트 숫자로 자동 정리했어요'}
        </p>
      </section>

      {data.signals.length > 0 && (
        <section className="snSection">
          <h3 className="snSectionHead">차트가 말해주는 것</h3>
          <ul className="snSignals">
            {data.signals.map((s, i) => (
              <li key={i} className={`snSignal snSignal-${s.type}`}>
                <span className="snSignalLabel">{s.label}</span>
                {s.desc && <span className="snSignalDesc">{s.desc}</span>}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="snSection">
        <h3 className="snSectionHead">최근 뉴스</h3>
        {data.news.length > 0 ? (
          <ul className="snNews">
            {data.news.map((n, i) => (
              <li key={i}>
                <a href={n.link} target="_blank" rel="noopener noreferrer">{n.title}</a>
              </li>
            ))}
          </ul>
        ) : (
          <p className="snHint">
            {data.news_enabled
              ? '최근 뉴스를 찾지 못했어요.'
              : '뉴스 기능이 아직 설정되지 않았어요.'}
          </p>
        )}
      </section>
    </div>
  )
}
