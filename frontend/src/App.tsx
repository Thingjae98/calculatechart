import './App.css'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { BoxRange, Candle, FibonacciLevels, IchimokuValues, PredictionResult, RecommendationItem, SearchResult } from './lib/api'
import { CandleChart } from './components/CandleChart'
import { fetchOhlcv, fetchRecommendations, fetchPrediction, pingServer, searchStocks } from './lib/api'

const PRED_DAY_OPTIONS = [7, 14, 21, 30] as const

function scoreColor(score: number) {
  if (score >= 65) return '#22c55e'
  if (score >= 50) return '#f59e0b'
  if (score >= 35) return '#f97316'
  return '#ef4444'
}

function scoreLabel(score: number) {
  if (score >= 75) return '강한 상승 추세'
  if (score >= 60) return '매수 고려'
  if (score >= 50) return '관망'
  if (score >= 35) return '약세 주의'
  return '매도 권고'
}

function App() {
  const [ticker, setTicker] = useState('005930')
  const [resolvedTicker, setResolvedTicker] = useState('')

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
  const [fibonacci, setFibonacci] = useState<FibonacciLevels | undefined>(undefined)
  const [ichimoku, setIchimoku] = useState<IchimokuValues | undefined>(undefined)

  const [showAllMA, setShowAllMA] = useState(false)
  const [showFibonacci, setShowFibonacci] = useState(false)

  const [freshLoadId, setFreshLoadId] = useState(0)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const hasMoreHistoryRef = useRef(true)

  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([])
  const [recError, setRecError] = useState<string | null>(null)
  const [recLoading, setRecLoading] = useState(false)
  const [serverWaking, setServerWaking] = useState(false)

  // ── 자동완성 상태 ──
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [showDropdown, setShowDropdown] = useState(false)
  const [selectedIdx, setSelectedIdx] = useState(-1)
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const canSubmit = useMemo(() => !!ticker.trim(), [ticker])

  // ── 자동완성 검색 (디바운스 250ms) ──
  const handleSearchInput = useCallback((value: string) => {
    setTicker(value)
    setSelectedIdx(-1)

    if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
    if (!value.trim()) {
      setSearchResults([])
      setShowDropdown(false)
      setSearchLoading(false)
      setSearchError(null)
      return
    }

    setSearchLoading(true)
    setSearchError(null)
    setShowDropdown(true)

    searchTimerRef.current = setTimeout(async () => {
      const resp = await searchStocks(value)
      setSearchResults(resp.results)
      setSearchError(resp.error ?? null)
      setSearchLoading(false)
      setShowDropdown(true)
    }, 250)
  }, [])

  // ── 자동완성 항목 선택 ──
  const selectSearchItem = useCallback((item: SearchResult) => {
    setTicker(item.name)
    setShowDropdown(false)
    setSearchResults([])
    void load(item.name)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── 바깥 클릭 시 드롭다운 닫기 ──
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node) &&
          inputRef.current && !inputRef.current.contains(e.target as Node)) {
        setShowDropdown(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // ── 키보드 네비게이션 ──
  const handleSearchKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (!showDropdown || searchResults.length === 0) return

    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIdx(prev => Math.min(prev + 1, searchResults.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIdx(prev => Math.max(prev - 1, 0))
    } else if (e.key === 'Enter' && selectedIdx >= 0) {
      e.preventDefault()
      selectSearchItem(searchResults[selectedIdx])
    } else if (e.key === 'Escape') {
      setShowDropdown(false)
    }
  }, [showDropdown, searchResults, selectedIdx, selectSearchItem])

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
    setShowDropdown(false)

    try {
      const result = await fetchOhlcv({ ticker: t })
      setCandles(result.data)
      setResolvedTicker(result.ticker)
      setSupportLines(result.support_lines ?? [])
      setResistanceLines(result.resistance_lines ?? [])
      setStockName(result.stock_name ?? t)
      setChartScore(result.score ?? 0)
      setBoxRange(result.box_range ?? { is_box: false })
      setFibonacci(result.fibonacci)
      setIchimoku(result.ichimoku)
    } catch (err) {
      setError(err instanceof Error ? err.message : '요청 중 오류가 발생했습니다.')
      setCandles([])
      setSupportLines([])
      setResistanceLines([])
      setStockName('')
      setChartScore(0)
      setBoxRange({ is_box: false })
      setFibonacci(undefined)
      setIchimoku(undefined)
    } finally {
      setLoading(false)
    }
  }

  // ── 과거 데이터 추가 로드 ─────────────────────────────────────────
  async function loadMore() {
    if (isLoadingMore || !candles.length || !hasMoreHistoryRef.current) return

    const tickerCode = resolvedTicker || ticker.trim()
    const earliest = candles[0].time
    const earliestDate = new Date(earliest)

    // 최대 3년(약 1095일) 이전 데이터까지만 로드
    const threeYearsAgo = new Date()
    threeYearsAgo.setFullYear(threeYearsAgo.getFullYear() - 3)
    if (earliestDate <= threeYearsAgo) {
      hasMoreHistoryRef.current = false
      return
    }

    const newEnd = new Date(earliestDate.getTime() - 86400000)

    setIsLoadingMore(true)
    try {
      const result = await fetchOhlcv({
        ticker: tickerCode,
        before_date: newEnd.toISOString().slice(0, 10),
        days_back: 180,
      })
      if (result.data.length === 0) {
        hasMoreHistoryRef.current = false
      } else {
        const filtered = result.data.filter((c) => c.time < earliest)
        if (filtered.length === 0) {
          hasMoreHistoryRef.current = false
        } else {
          setCandles((prev) => [...filtered, ...prev])
        }
      }
    } catch {
      hasMoreHistoryRef.current = false
    } finally {
      setIsLoadingMore(false)
    }
  }

  // ── N일 주가 예측 ─────────────────────────────────────────────────
  async function onPredict(days: number) {
    const tickerCode = resolvedTicker || ticker.trim()
    setPredDays(days)
    setPredLoading(true)
    setPredError(null)
    try {
      const result = await fetchPrediction(tickerCode, days)
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

  const lastPrice = candles.length > 0 ? candles[candles.length - 1].close : 0

  return (
    <div className="app">
      {/* ── 헤더: 검색 ────────────────────────────────────── */}
      <header className="header">
        <div className="headerTop">
          <h1 className="logo">차트 분석</h1>
          <span className="logoSub">명재의 차트 점수기반 추천</span>
        </div>

        <form className="searchBar" onSubmit={(e) => { e.preventDefault(); void load() }}>
          <div className="searchInputWrap">
            <input
              ref={inputRef}
              className="searchInput"
              value={ticker}
              onChange={(e) => handleSearchInput(e.target.value)}
              onFocus={() => { if (searchResults.length > 0 || searchError) setShowDropdown(true) }}
              onKeyDown={handleSearchKeyDown}
              placeholder="종목명 또는 종목코드 (예: 삼성전자)"
              autoComplete="off"
            />
            {showDropdown && (
              <div className="searchDropdown" ref={dropdownRef}>
                {searchLoading ? (
                  <div className="searchDropdownStatus">검색 중…</div>
                ) : searchError ? (
                  <div className="searchDropdownStatus searchDropdownError">{searchError}</div>
                ) : searchResults.length === 0 ? (
                  <div className="searchDropdownStatus">검색 결과가 없습니다</div>
                ) : (
                  searchResults.map((item, idx) => (
                    <button
                      key={item.ticker}
                      type="button"
                      className={`searchDropdownItem ${idx === selectedIdx ? 'searchDropdownItemActive' : ''}`}
                      onMouseDown={(e) => { e.preventDefault(); selectSearchItem(item) }}
                      onMouseEnter={() => setSelectedIdx(idx)}
                    >
                      <span className="searchDropdownName">{item.name}</span>
                      <span className="searchDropdownTicker">{item.ticker}</span>
                    </button>
                  ))
                )}
              </div>
            )}
          </div>
          <button className="btnPrimary" type="submit" disabled={!canSubmit || loading}>
            {loading ? '조회 중…' : '조회'}
          </button>
        </form>
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
                      <span className="sellInlineLabel">{prediction.sell_targets.short_term_desc ?? '단기 목표'}</span>
                      <span className="sellInlinePrice">{prediction.sell_targets.short_term.toLocaleString()}원</span>
                    </div>
                  )}
                  {prediction.sell_targets.long_term && (
                    <div className="sellInlineRow sellInlineLong">
                      <span className="sellInlineLabel">{prediction.sell_targets.long_term_desc ?? '장기 목표'}</span>
                      <span className="sellInlinePrice">{prediction.sell_targets.long_term.toLocaleString()}원</span>
                    </div>
                  )}
                  {prediction.sell_targets.stop_loss && (
                    <div className="sellInlineRow sellInlineStop">
                      <span className="sellInlineLabel">{prediction.sell_targets.stop_loss_desc ?? '손절가'}</span>
                      <span className="sellInlinePrice">{prediction.sell_targets.stop_loss.toLocaleString()}원</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          <div className="chartToggles">
            <button
              className={`predBtn ${showAllMA ? 'predBtnActive' : ''}`}
              onClick={() => setShowAllMA((v) => !v)}
            >
              {showAllMA ? '이평선 전체 ✓' : '이평선 전체'}
            </button>
            <button
              className={`predBtn ${showFibonacci ? 'predBtnActive' : ''}`}
              onClick={() => setShowFibonacci((v) => !v)}
            >
              {showFibonacci ? '피보나치 ✓' : '피보나치'}
            </button>
          </div>

          <CandleChart
            key={`${stockName}-${freshLoadId}`}
            candles={candles}
            predictedCandles={predictedCandles}
            supportLines={supportLines}
            resistanceLines={resistanceLines}
            boxRange={boxRange}
            fibonacci={fibonacci}
            ichimoku={ichimoku}
            showAllMA={showAllMA}
            showFibonacci={showFibonacci}
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
                {scoreLabel(prediction.prediction_score)}{' '}
                {prediction.prediction_score}/100
              </div>
            </div>

            <div className="predOutlook">
              <div className="predOutlookItem">
                <span className="predOutlookLabel">단기</span>
                <strong>{prediction.outlook_short}</strong>
              </div>
              <div className="predOutlookItem">
                <span className="predOutlookLabel">중기</span>
                <strong>{prediction.outlook_mid}</strong>
              </div>
            </div>
            <p className="predSummary">{prediction.summary}</p>

            {/* 매도 목표/손절 요약 카드 */}
            {prediction.sell_targets && (
              <div className="predTargetCards">
                {prediction.sell_targets.short_term && (
                  <div className="predTargetCard predTargetShort">
                    <div className="predTargetLabel">1차 목표</div>
                    <div className="predTargetPrice">{prediction.sell_targets.short_term.toLocaleString()}원</div>
                    {prediction.sell_targets.short_term_desc && (
                      <div className="predTargetDesc">{prediction.sell_targets.short_term_desc}</div>
                    )}
                  </div>
                )}
                {prediction.sell_targets.long_term && (
                  <div className="predTargetCard predTargetLong">
                    <div className="predTargetLabel">최종 목표</div>
                    <div className="predTargetPrice">{prediction.sell_targets.long_term.toLocaleString()}원</div>
                    {prediction.sell_targets.long_term_desc && (
                      <div className="predTargetDesc">{prediction.sell_targets.long_term_desc}</div>
                    )}
                  </div>
                )}
                {prediction.sell_targets.stop_loss && (
                  <div className="predTargetCard predTargetStop">
                    <div className="predTargetLabel">손절가</div>
                    <div className="predTargetPrice">{prediction.sell_targets.stop_loss.toLocaleString()}원</div>
                    {prediction.sell_targets.stop_loss_desc && (
                      <div className="predTargetDesc">{prediction.sell_targets.stop_loss_desc}</div>
                    )}
                  </div>
                )}
              </div>
            )}

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
            <div>
              <strong>추천 종목 Top 10</strong>
              <span className="recHeaderSub">바닥 반등 + 매집 신호 기반</span>
            </div>
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
                  <span className="recScoreLabel" style={{ color: scoreColor(item.score) }}>
                    {scoreLabel(item.score)}
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
