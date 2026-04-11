import './App.css'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { BoxRange, Candle, PredictionResult, RecommendationItem } from './lib/api'
import { CandleChart } from './components/CandleChart'
import { fetchOhlcv, fetchRecommendations, fetchPrediction, pingServer } from './lib/api'

function App() {
  const toYmd = (d: Date) => d.toISOString().slice(0, 10)

  const [ticker, setTicker] = useState('005930')
  const [endDate, setEndDate] = useState(() => toYmd(new Date()))
  const [startDate, setStartDate] = useState(() => toYmd(new Date(Date.now() - 1000 * 60 * 60 * 24 * 90)))

  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [predLoading, setPredLoading] = useState(false)
  const [predError, setPredError] = useState<string | null>(null)
  const [predictedCandles, setPredictedCandles] = useState<Candle[]>([])

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [candles, setCandles] = useState<Candle[]>([])
  const [supportLines, setSupportLines] = useState<number[]>([])
  const [resistanceLines, setResistanceLines] = useState<number[]>([])
  const [stockName, setStockName] = useState('')
  const [chartScore, setChartScore] = useState(0)
  const [boxRange, setBoxRange] = useState<BoxRange>({ is_box: false })

  // 차트 신선 로드 식별자 — 바뀌면 차트가 fitContent()로 리셋됨
  const [freshLoadId, setFreshLoadId] = useState(0)
  // 과거 데이터 추가 로드 중 여부
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  // 더 이상 로드할 과거 데이터가 없을 때 true
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
    const norm = (v: string) => v.replace(/-/g, '')
    return norm(s) <= norm(e)
  }, [ticker, startDate, endDate])

  // ── 신선 조회 (새 종목 or 새 날짜) ──────────────────────────────
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

  // ── 과거 데이터 추가 로드 (스크롤 왼쪽 끝 도달 시) ───────────────
  async function loadMore() {
    if (isLoadingMore || !candles.length || !hasMoreHistoryRef.current) return

    const earliest = candles[0].time // 'YYYY-MM-DD'
    const earliestDate = new Date(earliest)
    const newEnd = new Date(earliestDate.getTime() - 1000 * 60 * 60 * 24)   // 하루 전
    const newStart = new Date(newEnd.getTime() - 1000 * 60 * 60 * 24 * 90)  // 90일 추가

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
        // 기존 캔들 앞에 추가 (freshLoadId는 바꾸지 않아 차트 스크롤 위치 유지)
        setCandles((prev) => [...result.data, ...prev])
      }
    } catch {
      // 로드 실패 시 조용히 무시
    } finally {
      setIsLoadingMore(false)
    }
  }

  // ── 주가 예측 ────────────────────────────────────────────────────
  async function onPredict() {
    setPredLoading(true)
    setPredError(null)
    try {
      const result = await fetchPrediction(ticker.trim(), startDate, endDate)
      setPrediction(result)
      setPredictedCandles(result.predicted_candles ?? [])
    } catch (err) {
      setPrediction(null)
      setPredError(err instanceof Error ? err.message : '예측 요청 실패')
    } finally {
      setPredLoading(false)
    }
  }

  // ── 추천 종목 로드 ────────────────────────────────────────────────
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

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    await load()
  }

  useEffect(() => {
    // Render 무료 플랜: 서버 슬립 상태일 수 있으므로 먼저 ping으로 깨운 뒤 조회
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

  return (
    <div className="app">
      <header className="header">
        <div className="title">
          <h1>차트 분석</h1>
          <p className="subtitle">명재의 차트 점수기반 추천</p>
        </div>

        <form className="form" onSubmit={onSubmit}>
          <label className="field">
            <span>종목코드/종목명</span>
            <input
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder="예: 005930 또는 삼성전자"
            />
          </label>
          <label className="field">
            <span>시작일</span>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </label>
          <label className="field">
            <span>종료일</span>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </label>
          <button className="btn" type="submit" disabled={!canSubmit || loading}>
            {loading ? '불러오는 중…' : '조회'}
          </button>
          <button className="btn" type="button" onClick={onPredict} disabled={!canSubmit || predLoading}>
            {predLoading ? '분석 중…' : '📈 주가 예측'}
          </button>
        </form>
      </header>

      <main className="main">
        {serverWaking && (
          <div className="alertInfo">
            ⏳ 서버를 깨우는 중입니다… 처음 접속 시 30~60초 정도 걸릴 수 있습니다.
          </div>
        )}
        {isLoadingMore && (
          <div className="alertInfo" style={{ marginBottom: 8 }}>
            ↩ 과거 데이터 불러오는 중…
          </div>
        )}
        {error ? <div className="alert">오류: {error}</div> : null}
        {predError ? <div className="alert">예측 오류: {predError}</div> : null}
        {recError ? <div className="alert">추천 오류: {recError}</div> : null}

        <div className="card">
          <div className="cardHeader">
            <div className="meta">
              <div className="metaTitle">
                {stockName || ticker.trim()}
              </div>
              <div className="metaSub">
                {candles.length > 0 ? `${candles[0].time} ~ ${candles[candles.length - 1].time}` : `${startDate} ~ ${endDate}`}
                {' '}· {candles.length}개 봉 · 차트점수 {chartScore}/100
              </div>
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
          />
        </div>

        {prediction && !prediction.error && (
          <div className="card predictionCard">
            <div className="cardHeader">
              <div className="meta">
                <div className="metaTitle">
                  {prediction.stock_name} 예측 분석
                  <span style={{ marginLeft: 12, fontSize: 14, color: prediction.prediction_score >= 70 ? '#22c55e' : prediction.prediction_score >= 50 ? '#f59e0b' : '#ef4444' }}>
                    종합점수 {prediction.prediction_score}/100
                  </span>
                </div>
                <div className="metaSub">현재가 {prediction.current_price.toLocaleString()}원</div>
              </div>
            </div>

            <div style={{ padding: '12px 0', borderBottom: '1px solid #2d3748', marginBottom: 12 }}>
              <div style={{ fontSize: 14, color: '#94a3b8', marginBottom: 4 }}>단기(1~2주): <strong style={{ color: '#e2e8f0' }}>{prediction.outlook_short}</strong> &nbsp;|&nbsp; 중기(1달): <strong style={{ color: '#e2e8f0' }}>{prediction.outlook_mid}</strong></div>
              <div style={{ fontSize: 14, color: '#cbd5e1' }}>{prediction.summary}</div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {prediction.signals.map((sig, i) => (
                <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                  <span style={{ fontSize: 16 }}>
                    {sig.type === 'positive' ? '🟢' : sig.type === 'negative' ? '🔴' : '🟡'}
                  </span>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0' }}>{sig.label}</div>
                    <div style={{ fontSize: 12, color: '#94a3b8' }}>{sig.desc}</div>
                  </div>
                </div>
              ))}
            </div>

            {prediction.sell_targets && (prediction.sell_targets.short_term || prediction.sell_targets.long_term) && (
              <div style={{ marginTop: 16, padding: '12px 0', borderTop: '1px solid #2d3748' }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#e2e8f0', marginBottom: 8 }}>추천 매도가</div>
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                  {prediction.sell_targets.short_term && (
                    <div style={{ flex: 1, minWidth: 180, padding: 12, borderRadius: 8, backgroundColor: 'rgba(249, 115, 22, 0.15)', border: '1px solid rgba(249, 115, 22, 0.3)' }}>
                      <div style={{ fontSize: 12, color: '#fb923c', marginBottom: 4 }}>📌 단기 매도</div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: '#fdba74' }}>{prediction.sell_targets.short_term.toLocaleString()}원</div>
                      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{prediction.sell_targets.short_term_desc}</div>
                    </div>
                  )}
                  {prediction.sell_targets.long_term && (
                    <div style={{ flex: 1, minWidth: 180, padding: 12, borderRadius: 8, backgroundColor: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.3)' }}>
                      <div style={{ fontSize: 12, color: '#f87171', marginBottom: 4 }}>🎯 장기 매도</div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: '#fca5a5' }}>{prediction.sell_targets.long_term.toLocaleString()}원</div>
                      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{prediction.sell_targets.long_term_desc}</div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        <div className="card recommendationCard">
          <div className="cardHeader">
            <div className="meta">
              <div className="metaTitle">추천 종목 Top 10</div>
              <div className="metaSub">점수 높은 순</div>
            </div>
            <button className="btn" type="button" onClick={loadRecommendations} disabled={recLoading}>
              {recLoading ? '계산 중…' : '추천 새로고침'}
            </button>
          </div>
          <div className="recommendList">
            {recLoading ? (
              <div className="recommendEmpty">추천 종목 계산 중입니다… (최초 1~2분 소요)</div>
            ) : recommendations.length === 0 ? (
              <div className="recommendEmpty">추천 데이터가 없습니다.</div>
            ) : (
              recommendations.map((item, idx) => (
                <button
                  key={`${item.ticker}-${idx}`}
                  className="recommendItem"
                  onClick={() => {
                    const name = item.stock_name || item.ticker
                    setTicker(name)
                    void load(name)
                  }}
                  type="button"
                >
                  <span>
                    {idx + 1}. {item.stock_name} ({item.ticker})
                  </span>
                  <strong>{item.score}</strong>
                </button>
              ))
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
