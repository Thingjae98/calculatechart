import { useEffect, useState } from 'react'
import './Briefing.css'
import type { DailyBriefing, SemiOutlook, Tone, TopCapItem } from '../lib/briefing'
import { fetchLatestBriefing, fmtPct, pctTone } from '../lib/briefing'

function toneClass(tone: Tone) {
  return tone === 'up' ? 'toneUp' : tone === 'down' ? 'toneDown' : 'toneFlat'
}

function formatDateKo(iso: string) {
  // 'YYYY-MM-DD' → 'YYYY년 M월 D일 (요일)'
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso)
  if (!m) return iso
  const [, y, mo, d] = m
  const date = new Date(Number(y), Number(mo) - 1, Number(d))
  const dow = ['일', '월', '화', '수', '목', '금', '토'][date.getDay()]
  return `${y}년 ${Number(mo)}월 ${Number(d)}일 (${dow})`
}

function QuoteRow({ name, value, change_pct }: { name: string; value: string; change_pct: number | null }) {
  const tone = pctTone(change_pct)
  return (
    <div className="bQuoteRow">
      <span className="bQuoteName">{name}</span>
      <span className="bQuoteRight">
        <span className="bQuoteValue">{value}</span>
        <span className={`bQuotePct ${toneClass(tone)}`}>{fmtPct(change_pct)}</span>
      </span>
    </div>
  )
}

function SemiCard({ s }: { s: SemiOutlook }) {
  return (
    <div className="bSemiCard">
      <div className="bSemiHead">
        <div>
          <span className="bSemiName">{s.name}</span>
          <span className="bSemiTicker">{s.ticker}</span>
        </div>
        <span className={`bDirBadge ${toneClass(s.tone)}`}>{s.direction}</span>
      </div>
      <div className="bSemiField">
        <span className="bSemiLabel">핵심 모멘텀·리스크</span>
        <p>{s.momentum}</p>
      </div>
      <div className="bSemiField">
        <span className="bSemiLabel">오늘의 단기 전략</span>
        <p>{s.strategy}</p>
      </div>
    </div>
  )
}

/**
 * 시총 상위 종목 카드 — 숫자(종가/등락/지지/저항)는 백엔드 알고리즘 값, comment만 모델 작성.
 * 지지/저항은 "여기서 버팀 / 여기서 막힘"이 바로 읽히도록 라벨을 쉬운 말로 둔다.
 */
function TopCapCard({ s }: { s: TopCapItem }) {
  const tone = pctTone(s.change_pct)
  return (
    <div className="bTopCard">
      <div className="bTopHead">
        <span className="bTopName">{s.name}</span>
        <span className={`bTopPct ${toneClass(tone)}`}>{fmtPct(s.change_pct)}</span>
      </div>
      <div className="bTopPrice">
        {s.close}원 <span className="bTopPriceNote">전일 종가</span>
      </div>

      {/* 지지/저항 — 둘 다 없으면 줄 자체를 숨긴다(빈 칸이 더 혼란스러움) */}
      {(s.support || s.resistance) && (
        <div className="bTopLevels">
          <span className="bTopLevel">
            <span className="bTopLevelLabel">지지 (여기서 버팀)</span>
            <span className="bTopLevelVal">{s.support ? `${s.support}원` : '—'}</span>
          </span>
          <span className="bTopLevel">
            <span className="bTopLevelLabel">저항 (여기서 막힘)</span>
            <span className="bTopLevelVal">{s.resistance ? `${s.resistance}원` : '—'}</span>
          </span>
        </div>
      )}

      {s.comment && <p className="bTopComment">{s.comment}</p>}
    </div>
  )
}

/**
 * 시총 상위 섹션. 코스피 5 + 코스닥 5 = 10장을 한 번에 쌓으면 화면이 너무 길어져
 * "단순함 우선" 원칙과 충돌하므로, 탭으로 한 번에 5장만 보여준다.
 */
function TopCapsSection({ caps }: { caps: NonNullable<DailyBriefing['top_caps']> }) {
  const [market, setMarket] = useState<'kospi' | 'kosdaq'>('kospi')
  const items = market === 'kospi' ? caps.kospi : caps.kosdaq

  return (
    <div className="bCard">
      <h2 className="bCardTitle">
        4. 시가총액 상위 <span className="bCardSub">지지·저항 한눈에</span>
      </h2>

      {/* 탭: 선택 상태를 색만이 아니라 aria-selected로도 알린다 */}
      <div className="bTabs" role="tablist" aria-label="시장 선택">
        <button
          role="tab"
          aria-selected={market === 'kospi'}
          className={`bTab ${market === 'kospi' ? 'bTabActive' : ''}`}
          onClick={() => setMarket('kospi')}
        >
          코스피
        </button>
        <button
          role="tab"
          aria-selected={market === 'kosdaq'}
          className={`bTab ${market === 'kosdaq' ? 'bTabActive' : ''}`}
          onClick={() => setMarket('kosdaq')}
        >
          코스닥
        </button>
      </div>

      {items.length === 0 ? (
        <div className="bTopEmpty">이 시장의 종목 정보를 불러오지 못했습니다.</div>
      ) : (
        <div className="bTopGrid">
          {items.map((s) => <TopCapCard key={s.ticker} s={s} />)}
        </div>
      )}
    </div>
  )
}

export function Briefing() {
  const [data, setData] = useState<DailyBriefing | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    void (async () => {
      const { data, error } = await fetchLatestBriefing()
      if (!alive) return
      setData(data)
      setError(error)
      setLoading(false)
    })()
    return () => { alive = false }
  }, [])

  if (loading) {
    return (
      <section className="briefing">
        <div className="bLoading">오늘의 브리핑을 불러오는 중…</div>
      </section>
    )
  }

  if (error || !data) {
    return (
      <section className="briefing">
        <div className="bError">
          {error ?? '브리핑 데이터가 없습니다.'}
          <div className="bErrorSub">아직 오늘 브리핑이 생성되지 않았을 수 있어요. 잠시 후 다시 확인해 주세요.</div>
        </div>
      </section>
    )
  }

  const g = data.global

  return (
    <section className="briefing">
      {/* 헤더 */}
      <header className="bHeader">
        <div className="bTitleRow">
          <span className="bKicker">📅 데일리 마켓 브리핑</span>
          <span className="bDate">{formatDateKo(data.date)}</span>
        </div>
      </header>

      {data.is_sample && (
        <div className="bSampleBanner">
          ⚠️ 예시(샘플) 데이터입니다. 실제 시장 수치가 아니며, 매일 아침 자동 브리핑이 생성되면 교체됩니다.
        </div>
      )}

      {/* 1. 글로벌 마켓 서머리 */}
      <div className="bCard">
        <h2 className="bCardTitle">1. 글로벌 마켓 서머리 <span className="bCardSub">야간 시장 요약</span></h2>

        <div className="bBlock">
          <div className="bBlockLabel">미국 3대 지수</div>
          {g.indices.map((q) => <QuoteRow key={q.name} {...q} />)}
        </div>

        <div className="bBlock">
          <div className="bBlockLabel">
            필라델피아 반도체 지수 (SOX)
            <span className={`bInlinePct ${toneClass(pctTone(g.sox.change_pct))}`}>{fmtPct(g.sox.change_pct)}</span>
            <span className="bInlineVal">{g.sox.value}</span>
          </div>
          {g.sox.components.map((q) => <QuoteRow key={q.name} {...q} />)}
        </div>

        <div className="bBlock">
          <div className="bBlockLabel">주요 매크로 지표</div>
          {g.macro.map((m) => (
            <div key={m.name} className="bQuoteRow">
              <span className="bQuoteName">{m.name}</span>
              <span className="bQuoteRight">
                <span className="bQuoteValue">{m.value}</span>
                {m.change && <span className={`bQuotePct ${toneClass(m.tone)}`}>{m.change}</span>}
              </span>
            </div>
          ))}
        </div>

        <div className="bBlock">
          <div className="bBlockLabel">글로벌 핵심 뉴스</div>
          <ul className="bNewsList">
            {g.headlines.map((h, i) => <li key={i}>{h}</li>)}
          </ul>
        </div>
      </div>

      {/* 2. 국내 시장 방향성 */}
      <div className="bCard">
        <h2 className="bCardTitle">2. 오늘의 국내 증시 방향성 <span className="bCardSub">KOSPI / KOSDAQ</span></h2>
        <div className={`bDirection ${toneClass(data.domestic.tone)}`}>
          <span className="bDirectionIcon">
            {data.domestic.tone === 'up' ? '▲' : data.domestic.tone === 'down' ? '▼' : '■'}
          </span>
          <span className="bDirectionText">{data.domestic.direction}</span>
        </div>
        <ul className="bBulletList">
          {data.domestic.rationale.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
        {data.domestic.headlines && data.domestic.headlines.length > 0 && (
          <div className="bBlock bDomesticNews">
            <div className="bBlockLabel">국내 핵심 뉴스</div>
            <ul className="bNewsList">
              {data.domestic.headlines.map((h, i) => <li key={i}>{h}</li>)}
            </ul>
          </div>
        )}
        <div className="bLevels">
          <span className="bLevelsLabel">지지·저항 / 트리거</span>
          {data.domestic.levels}
        </div>
      </div>

      {/* 3. 반도체 투톱 */}
      <div className="bCard">
        <h2 className="bCardTitle">3. 반도체 투톱 <span className="bCardSub">삼성전자 · SK하이닉스</span></h2>
        <div className="bSemiGrid">
          <SemiCard s={data.semis.samsung} />
          <SemiCard s={data.semis.hynix} />
        </div>
      </div>

      {/* 4. 시총 상위 — 백엔드가 콜드/다운이면 워크플로우가 생략하므로 없을 수 있다 */}
      {data.top_caps && <TopCapsSection caps={data.top_caps} />}

      {/* 5. 개장 후 관전 포인트 */}
      <div className="bCard">
        <h2 className="bCardTitle">5. 핵심 체크리스트 <span className="bCardSub">개장 후 관전 포인트</span></h2>
        <ul className="bCheckList">
          {data.checklist.map((c, i) => <li key={i}>{c}</li>)}
        </ul>
      </div>

      <div className="bFooter">
        생성 시각: {new Date(data.generated_at).toLocaleString('ko-KR')} · 투자 판단의 참고용이며 투자 책임은 본인에게 있습니다.
      </div>
    </section>
  )
}
