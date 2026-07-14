import './App.css'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Briefing } from './components/Briefing'
import { ChartAnalysis } from './components/ChartAnalysis'
import { InstallButton } from './components/InstallButton'

type Section = 'briefing' | 'chart'

function App() {
  const [active, setActive] = useState<Section>('briefing')
  const briefingRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<HTMLDivElement>(null)

  const scrollTo = useCallback((s: Section) => {
    const el = s === 'briefing' ? briefingRef.current : chartRef.current
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    el?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' })
    setActive(s)
  }, [])

  // 스크롤 위치에 따라 내비게이션 활성 표시 갱신
  useEffect(() => {
    const targets = [briefingRef.current, chartRef.current].filter(Boolean) as HTMLDivElement[]
    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            setActive(e.target === chartRef.current ? 'chart' : 'briefing')
          }
        }
      },
      { rootMargin: '-45% 0px -45% 0px', threshold: 0 },
    )
    targets.forEach((t) => observer.observe(t))
    return () => observer.disconnect()
  }, [])

  return (
    <div className="app">
      {/* ── 스티키 내비게이션 (앱 최적화: 빠른 이동) ── */}
      <nav className="topNav">
        <span className="topNavBrand">명재가족</span>
        <div className="topNavTabs">
          <InstallButton />
          <button
            className={`topNavTab ${active === 'briefing' ? 'topNavTabActive' : ''}`}
            onClick={() => scrollTo('briefing')}
          >
            📅 브리핑
          </button>
          <button
            className={`topNavTab ${active === 'chart' ? 'topNavTabActive' : ''}`}
            onClick={() => scrollTo('chart')}
          >
            📈 차트
          </button>
        </div>
      </nav>

      {/* 메인 페이지: 오늘의 브리핑 */}
      <div ref={briefingRef}>
        <Briefing />
      </div>

      {/* 하단: 차트 분석 */}
      <div ref={chartRef} className="chartSection">
        <ChartAnalysis />
      </div>
    </div>
  )
}

export default App
