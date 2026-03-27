import './App.css'
import { useEffect, useMemo, useState } from 'react'
import type { Candle, RecommendationItem } from './lib/api'
import { fetchOhlcv, fetchRecommendations } from './lib/api'
import { CandleChart } from './components/CandleChart'

function App() {
  const [ticker, setTicker] = useState('005930')
  const toYmd = (d: Date) => d.toISOString().slice(0, 10)
  const [endDate, setEndDate] = useState(() => toYmd(new Date()))
  const [startDate, setStartDate] = useState(() => toYmd(new Date(Date.now() - 1000 * 60 * 60 * 24 * 90)))

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [candles, setCandles] = useState<Candle[]>([])
  const [supportLines, setSupportLines] = useState<number[]>([])
  const [resistanceLines, setResistanceLines] = useState<number[]>([])
  const [stockName, setStockName] = useState('')
  const [chartScore, setChartScore] = useState(0)
  const [boxRange, setBoxRange] = useState<{ is_box: boolean; top?: number; bottom?: number }>({ is_box: false })
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([])

  const canSubmit = useMemo(() => {
    const t = ticker.trim()
    return t.length > 0 && startDate.trim().length > 0 && endDate.trim().length > 0
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
      setBoxRange((result as any).box_range ?? { is_box: false })
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

  async function loadRecommendations() {
    try {
      const top = await fetchRecommendations(10)
      setRecommendations(top)
    } catch {
      setRecommendations([])
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
        </form>
      </header>

      <main className="main">
        {error ? <div className="alert">오류: {error}</div> : null}
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
