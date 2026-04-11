import './App.css'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { BoxRange, Candle, PredictionResult, RecommendationItem } from './lib/api'
import { CandleChart } from './components/CandleChart'
import { fetchOhlcv, fetchRecommendations, fetchPrediction, pingServer } from './lib/api'

const PRED_DAY_OPTIONS = [7, 14, 21, 30] as const

function scoreColor(score: number) {
  if (score >= 70) return '#22c55e'
  if (score >= 50) return '#f59e0b'
  return '#ef4444'
}

function scoreLabel(score: number) {
  if (score >= 70) return '매수 유리'
  if (score >= 50) return '관망'
  return '매수 주의'
}

function App() {
  const toYmd = (d: Date) => d.toISOString().slice(0, 10)

  const [ticker, setTicker] = useState('005930')
  const [endDate, setEndDate] = useState(() => toYmd(new Date()))
  const [startDate, setStartDate] = useState(
    () => toYmd(new Date(Date.now() - 1000 * 60 * 60 * 24 * 180)), // 6개월
  )

  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [predLoading, setPredLoading] = useState(false)
  const [predError, setPredError] = useState<string | null>(null)
  const [predictedCandles, setPredictedCandles] = useState<Candle[]>([])
  const [predDays, setPredDays] = useState<number>(7)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [candles, setCandles] = useState<Candle[]>([])
  const [supportLines, setSupportLines] = useState<number[]>([])
  const [resistanceLines, setResistanceLines] = useState<number[]>([])
  const [stockName, setStockName] = useState('')
  const [chartScore, setChartScore] = useState(0)
  const [boxRange, setBoxRange] = useState<BoxRange>({ is_box: false })

  const [freshLoadId, setFreshLoadId] = useState(0)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const hasMoreHistoryRef = useRef(true)

  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([])
  const [recError, setRecError] = useState<string | null>(null)
  const [recLoading, setRecLoading] = useState(false)
  const [serverWaking, setServerWaking] = useState(false)

  const canSubmit = useMemo(() => {
    const t = ticker.trim()
    const s = startDate.trim()
    const e = endDate.trim()
    if (!t || !s || !e) return false
    return s.replace(/-/g, '') <= e.replace(/-/g, '')
  }, [ticker, startDate, endDate])

  // ── 조회 ─────────────────────────────────────────────────────────
  async function load(overrideTicker?: string) {
    if (!canSubmit && !overrideTicker) return
    const t = (overrideTicker ?? ticker).trim()

    setLoading(true)
    setError(null)
    setFreshLoadId((id) => id + 1)
    setPredictedCandles([])
    setPrediction(null)
    hasMoreHistoryRef.current = true

    try {
      const result = await fetchOhlcv({
        ticker: t,
        start_date: startDate.trim(),
        end_date: endDate.trim(),
      })
      setCandles(result.data)
      setSupportLines(result.support_lines ?? [])
      setResistanceLines(result.resistance_lines ?? [])
      setStockName(result.stock_name ?? t)
      setChartScore(result.score ?? 0)
      setBoxRange(result.box_range ?? { is_box: false })
    } catch (err) {
      setError(err instanceof Error ? err.message : '요청 중 오류가 발생했습니다.')
      setCandles([])
      setSupportLines([])
      setResistanceLines([])
      setStockName('')
      setChartScore(0)
      setBoxRange({ is_box: false })
    } finally {
      setLoading(false)
    }
  }

  // ── 과거 데이터 추가 로드 ─────────────────────────────────────────
  async function loadMore() {
    if (isLoadingMore || !candles.length || !hasMoreHistoryRef.current) return

    const earliest = candles[0].time
    const earliestDate = new Date(earliest)
    const newEnd = new Date(earliestDate.getTime() - 86400000)
    const newStart = new Date(newEnd.getTime() - 86400000 * 180) // 6개월씩 추가

    setIsLoadingMore(true)
    try {
      const result = await fetchOhlcv({
        ticker: ticker.trim(),
        start_date: toYmd(newStart),
        end_date: toYmd(newEnd),
      })
      if (result.data.length === 0) {
        hasMoreHistoryRef.current = false
      } else {
        setCandles((prev) => [...result.data, ...prev])
      }
    } catch {
      // 무시
    } finally {
      setIsLoadingMore(false)
    }
  }

  // ── N일 주가 예측 ─────────────────────────────────────────────────
  async function onPredict(days: number) {
    setPredDays(days)
    setPredLoading(true)
    setPredError(null)
    try {
      const result = await fetchPrediction(ticker.trim(), startDate, endDate, days)
      setPrediction(result)
      setPredictedCandles(result.predicted_candles ?? [])
    } catch (err) {
      setPrediction(null)
      setPredictedCandles([])
      setPredError(err instanceof Error ? err.message : '예측 요청 실패')
    } finally {
      setPredLoading(false)
    }
  }

  // ── 추천 종목 ─────────────────────────────────────────────────────
  async function loadRecommendations() {
    setRecError(null)
    setRecLoading(true)
    try {
      const top = await fetchRecommendations(10)
      setRecommendations(top)
    } catch (err) {
      setRecommendations([])
      setRecError(err instanceof Error ? err.message : '추천 목록 로드 실패')
    } finally {
      setRecLoading(false)
    }
  }

  useEffect(() => {
    let wakeTimer: ReturnType<typeof setTimeout> | null = null
    const init = async () => {
      wakeTimer = setTimeout(() => setServerWaking(true), 2000)
      await pingServer()
      if (wakeTimer) clearTimeout(wakeTimer)
      setServerWaking(false)
      void load()
      void loadRecommendations()
    }
    void init()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 현재가
  const lastPrice = candles.length > 0 ? candles[candles.length - 1].close : 0

  return (
    <div className="app">
      {/* ── 헤더: 검색 + 기간 ────────────────────────────────────── */}
      <header className="header">
        <div className="headerTop">
          <h1 className="logo">차트 분석</h1>
          <span className="logoSub">명재의 차트 점수기반 추천</span>
        </div>

        <form className="searchBar" onSubmit={(e) => { e.preventDefault(); void load() }}>
          <input
            className="searchInput"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            placeholder="종목명 또는 종목코드 (예: 삼성전자)"
          />
          <button className="btnPrimary" type="submit" disabled={!canSubmit || loading}>
            {loading ? '조회 중…' : '조회'}
          </button>
        </form>

        <div className="dateRow">
          <label className="dateField">
            <span>시작일</span>
            <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
          </label>
          <label className="dateField">
            <span>종료일</span>
            <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
          </label>
        </div>
      </header>

      <main className="main">
        {/* ── 서버 상태 / 에러 ──────────────────────────────────── */}
        {serverWaking && (
          <div className="alertInfo">서버를 깨우는 중입니다… 처음 접속 시 30~60초 소요</div>
        )}
        {isLoadingMore && (
          <div className="alertInfo">과거 데이터 불러오는 중…</div>
        )}
        {error && <div className="alert">오류: {error}</div>}
        {predError && <div className="alert">예측 오류: {predError}</div>}
        {recError && <div className="alert">추천 오류: {recError}</div>}

        {/* ── 메인 차트 카드 ────────────────────────────────────── */}
        <section className="card chartCard">
          <div className="chartHeader">
            <div>
              <div className="stockTitle">{stockName || ticker.trim()}</div>
              <div className="stockPrice">
                {lastPrice > 0 && <>{lastPrice.toLocaleString()}원</>}
                <span className="stockMeta">
                  {' '}· {candles.length}봉
                </span>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
              <div className="scoreBox" style={{ borderColor: scoreColor(chartScore) }}>
                <div className="scoreNum" style={{ color: scoreColor(chartScore) }}>{chartScore}</div>
                <div className="scoreLabel">{scoreLabel(chartScore)}</div>
                <div className="scoreBar">
                  <div
                    className="scoreBarFill"
                    style={{ width: `${chartScore}%`, backgroundColor: scoreColor(chartScore) }}
                  />
                </div>
              </div>
              {prediction?.sell_targets && (
                <div className="sellTargetsInline">
                  {prediction.sell_targets.short_term && (
                    <div className="sellInlineRow sellInlineShort">
                      <span className="sellInlineLabel">단기 목표가</span>
                      <span className="sellInlinePrice">{prediction.sell_targets.short_term.toLocaleString()}원</span>
                    </div>
                  )}
                  {prediction.sell_targets.long_term && (
                    <div className="sellInlineRow sellInlineLong">
                      <span className="sellInlineLabel">장기 목표가</span>
                      <span className="sellInlinePrice">{prediction.sell_targets.long_term.toLocaleString()}원</span>
                    </div>
                  )}
                  {prediction.sell_targets.stop_loss && (
                    <div className="sellInlineRow sellInlineStop">
                      <span className="sellInlineLabel">손절가</span>
                      <span className="sellInlinePrice">{prediction.sell_targets.stop_loss.toLocaleString()}원</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          <CandleChart
            candles={candles}
            predictedCandles={predictedCandles}
            supportLines={supportLines}
            resistanceLines={resistanceLines}
            boxRange={boxRange}
            freshLoadId={freshLoadId}
            onLoadMore={loadMore}
            predDays={predDays}
          />
        </section>

        {/* ── N일 예측 버튼 바 ──────────────────────────────────── */}
        <section className="predBar">
          <span className="predBarLabel">주가 예측</span>
          <div className="predBtnGroup">
            {PRED_DAY_OPTIONS.map((d) => (
              <button
                key={d}
                className={`predBtn ${predDays === d && predictedCandles.length > 0 ? 'predBtnActive' : ''}`}
                onClick={() => void onPredict(d)}
                disabled={!canSubmit || predLoading}
              >
                {d}일
              </button>
            ))}
          </div>
          {predLoading && <span className="predBarLoading">분석 중…</span>}
        </section>

        {/* ── 예측 결과 카드 ────────────────────────────────────── */}
        {prediction && !prediction.error && (
          <section className="card predCard">
            <div className="predCardHeader">
              <div>
                <strong>{prediction.stock_name}</strong> {predDays}일 예측
              </div>
              <div
                className="predScore"
                style={{ color: scoreColor(prediction.prediction_score) }}
              >
                {prediction.prediction_score >= 70
                  ? '매수 추천'
                  : prediction.prediction_score >= 40
                    ? '관망'
                    : '매수 위험'}{' '}
                {prediction.prediction_score}/100
              </div>
            </div>

            <div className="predOutlook">
              <div>단기(1~2주): <strong>{prediction.outlook_short}</strong></div>
              <div>중기(1달): <strong>{prediction.outlook_mid}</strong></div>
            </div>
            <p className="predSummary">{prediction.summary}</p>

            <div className="signalList">
              {prediction.signals.map((sig, i) => (
                <div key={i} className="signalItem">
                  <span className="signalDot">
                    {sig.type === 'positive' ? '🟢' : sig.type === 'negative' ? '🔴' : '🟡'}
                  </span>
                  <div>
                    <div className="signalLabel">{sig.label}</div>
                    <div className="signalDesc">{sig.desc}</div>
                  </div>
                </div>
              ))}
            </div>

          </section>
        )}

        {/* ── 추천 종목 ────────────────────────────────────────── */}
        <section className="card recCard">
          <div className="recHeader">
            <strong>추천 종목 Top 10</strong>
            <button className="btnSmall" onClick={loadRecommendations} disabled={recLoading}>
              {recLoading ? '계산 중…' : '새로고침'}
            </button>
          </div>
          <div className="recList">
            {recLoading ? (
              <div className="recEmpty">추천 종목을 계산 중입니다… (최초 1~2분 소요)</div>
            ) : recommendations.length === 0 ? (
              <div className="recEmpty">추천 데이터가 없습니다.</div>
            ) : (
              recommendations.map((item, idx) => (
                <button
                  key={`${item.ticker}-${idx}`}
                  className="recItem"
                  onClick={() => {
                    const name = item.stock_name || item.ticker
                    setTicker(name)
                    void load(name)
                  }}
                >
                  <span className="recRank">{idx + 1}</span>
                  <span className="recName">{item.stock_name}</span>
                  <span className="recTicker">{item.ticker}</span>
                  <span
                    className="recScore"
                    style={{ color: scoreColor(item.score) }}
                  >
                    {item.score}
                  </span>
                </button>
              ))
            )}
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
