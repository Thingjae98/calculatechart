import './App.css'
import { useEffect, useMemo, useState } from 'react'
import type { BoxRange, Candle, PredictionResult, RecommendationItem } from './lib/api'
import { CandleChart } from './components/CandleChart'
import { fetchOhlcv, fetchRecommendations, fetchPrediction } from './lib/api'

function App() {
  const [ticker, setTicker] = useState('005930')
  const toYmd = (d: Date) => d.toISOString().slice(0, 10)
  const [endDate, setEndDate] = useState(() => toYmd(new Date()))
  const [startDate, setStartDate] = useState(() => toYmd(new Date(Date.now() - 1000 * 60 * 60 * 24 * 90)))
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [predLoading, setPredLoading] = useState(false)
  const [predError, setPredError] = useState<string | null>(null)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [candles, setCandles] = useState<Candle[]>([])
  const [supportLines, setSupportLines] = useState<number[]>([])
  const [resistanceLines, setResistanceLines] = useState<number[]>([])
  const [stockName, setStockName] = useState('')
  const [chartScore, setChartScore] = useState(0)
  const [boxRange, setBoxRange] = useState<BoxRange>({ is_box: false })
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([])
  const [recError, setRecError] = useState<string | null>(null)

  const canSubmit = useMemo(() => {
    const t = ticker.trim()
    const s = startDate.trim()
    const e = endDate.trim()
    if (!t || !s || !e) return false
    // YYYY-MM-DD 또는 YYYYMMDD 일 때 문자열 비교로도 전후 판단 가능하지만,
    // 구분자 없는 쪽과 섞일 수 있어 숫자만 남겨 비교
    const norm = (v: string) => v.replace(/-/g, '')
    return norm(s) <= norm(e)
  }, [ticker, startDate, endDate])

  async function load() {
    if (!canSubmit) return

    setLoading(true)
    setError(null)
    try {
      const result = await fetchOhlcv({
        ticker: ticker.trim(),
        start_date: startDate.trim(),
        end_date: endDate.trim(),
      })
      setCandles(result.data)
      setSupportLines(result.support_lines ?? [])
      setResistanceLines(result.resistance_lines ?? [])
      setStockName(result.stock_name ?? ticker.trim())
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

  async function onPredict() {
    setPredLoading(true)
    setPredError(null)
    try {
      const result = await fetchPrediction(ticker.trim(), startDate, endDate)
      setPrediction(result)
    } catch (err) {
      setPrediction(null)
      setPredError(err instanceof Error ? err.message : '예측 요청 실패')
    } finally {
      setPredLoading(false)
    }
  }

  async function loadRecommendations() {
    setRecError(null)
    try {
      const top = await fetchRecommendations(10)
      setRecommendations(top)
    } catch (err) {
      setRecommendations([])
      setRecError(err instanceof Error ? err.message : '추천 목록 로드 실패')
    }
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    await load()
  }

  useEffect(() => {
    // 첫 진입 시 1회 자동 조회
    void load()
    void loadRecommendations()
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
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              placeholder="YYYY-MM-DD 또는 YYYYMMDD"
            />
          </label>
          <label className="field">
            <span>종료일</span>
            <input
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              placeholder="YYYY-MM-DD 또는 YYYYMMDD"
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
                {startDate} ~ {endDate} · {candles.length}개 봉 · 차트점수 {chartScore}/100
              </div>
            </div>
          </div>
          <CandleChart
            candles={candles}
            supportLines={supportLines}
            resistanceLines={resistanceLines}
            boxRange={boxRange}
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
            <button className="btn" type="button" onClick={loadRecommendations}>
              추천 새로고침
            </button>
          </div>
          <div className="recommendList">
            {recommendations.length === 0 ? (
              <div className="recommendEmpty">추천 데이터가 없습니다.</div>
            ) : (
              recommendations.map((item, idx) => (
                <button
                  key={`${item.ticker}-${idx}`}
                  className="recommendItem"
                  onClick={() => setTicker(item.stock_name || item.ticker)}
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
