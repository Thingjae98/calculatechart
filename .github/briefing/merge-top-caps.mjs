// 브리핑 JSON에 시총 상위 종목을 병합한다.
//
// 원칙: 숫자는 백엔드가 진실이고, 모델은 코멘트만 쓴다.
// 백엔드가 계산한 종가/등락률/지지/저항으로 항목을 만들고, 모델이 쓴 comment만
// ticker로 매칭해 붙인다. 모델이 숫자를 지어냈더라도 여기서 전부 덮어써진다.
//
// 사용: node merge-top-caps.mjs <top-caps.json> <briefing.json> [...더 많은 briefing.json]

import { readFileSync, writeFileSync } from 'node:fs'

const [topCapsPath, ...briefingPaths] = process.argv.slice(2)

if (!topCapsPath || briefingPaths.length === 0) {
  console.error('사용법: node merge-top-caps.mjs <top-caps.json> <briefing.json...>')
  process.exit(1)
}

const readJson = (p) => JSON.parse(readFileSync(p, 'utf8'))

const topCaps = readJson(topCapsPath)
const kospi = Array.isArray(topCaps.kospi) ? topCaps.kospi : []
const kosdaq = Array.isArray(topCaps.kosdaq) ? topCaps.kosdaq : []

if (kospi.length === 0 && kosdaq.length === 0) {
  // 백엔드가 콜드/다운이었던 경우. 섹션 없이 브리핑은 그대로 나가야 한다.
  console.log('top-caps 데이터 없음 — top_caps 섹션 생략')
  process.exit(0)
}

/** 모델이 쓴 코멘트를 ticker로 찾아 붙인다 (숫자는 백엔드 값 유지) */
function withComments(authoritative, fromModel) {
  const byTicker = new Map()
  for (const it of Array.isArray(fromModel) ? fromModel : []) {
    if (it && typeof it.ticker === 'string') byTicker.set(it.ticker, it)
  }
  return authoritative.map((item) => {
    const raw = byTicker.get(item.ticker)?.comment
    const comment = typeof raw === 'string' && raw.trim() ? raw.trim() : undefined
    return comment ? { ...item, comment } : { ...item }
  })
}

for (const path of briefingPaths) {
  const briefing = readJson(path)
  const modelSide = briefing.top_caps ?? {}

  briefing.top_caps = {
    kospi: withComments(kospi, modelSide.kospi),
    kosdaq: withComments(kosdaq, modelSide.kosdaq),
    ...(topCaps.as_of ? { as_of: topCaps.as_of } : {}),
  }

  writeFileSync(path, JSON.stringify(briefing, null, 2) + '\n', 'utf8')

  const withText = [...briefing.top_caps.kospi, ...briefing.top_caps.kosdaq].filter((i) => i.comment).length
  console.log(
    `${path} ← top_caps 병합: 코스피 ${kospi.length} / 코스닥 ${kosdaq.length} (코멘트 ${withText}건)`,
  )
}
