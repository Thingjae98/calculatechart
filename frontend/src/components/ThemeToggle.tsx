import { useEffect, useState } from 'react'
import { applyTheme, getStoredMode, storeMode, type ResolvedTheme, type ThemeMode } from '../lib/theme'

/** 시스템 → 라이트 → 다크 → 시스템 순환 */
const NEXT: Record<ThemeMode, ThemeMode> = {
  system: 'light',
  light: 'dark',
  dark: 'system',
}

const LABEL: Record<ThemeMode, string> = {
  system: '시스템 설정',
  light: '밝게',
  dark: '어둡게',
}

/** 아이콘은 텍스트가 아니라 그림이므로 aria-hidden — 의미는 aria-label이 전달한다 */
function Icon({ mode }: { mode: ThemeMode }) {
  const common = { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true }
  if (mode === 'light') {
    return (
      <svg {...common}>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
      </svg>
    )
  }
  if (mode === 'dark') {
    return (
      <svg {...common}>
        <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
      </svg>
    )
  }
  // 시스템: 모니터 아이콘
  return (
    <svg {...common}>
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <path d="M8 21h8M12 17v4" />
    </svg>
  )
}

export function ThemeToggle() {
  const [mode, setMode] = useState<ThemeMode>(() => getStoredMode())
  const [resolved, setResolved] = useState<ResolvedTheme>(() => applyTheme(getStoredMode()))

  // mode가 바뀌면 즉시 반영 + 저장
  useEffect(() => {
    setResolved(applyTheme(mode))
    storeMode(mode)
  }, [mode])

  // 'system'일 때만 OS 설정 변화를 따라간다.
  // (수동 선택 상태에서 따라가면 사용자의 선택을 덮어쓰게 된다)
  useEffect(() => {
    if (mode !== 'system') return
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)')
    if (!mq) return
    const onChange = () => setResolved(applyTheme('system'))
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [mode])

  const next = NEXT[mode]
  return (
    <button
      type="button"
      className="themeToggle"
      onClick={() => setMode(next)}
      // 아이콘만 있는 버튼이므로 레이블 필수
      aria-label={`화면 밝기: 현재 ${LABEL[mode]}${mode === 'system' ? ` (${resolved === 'dark' ? '어둡게' : '밝게'})` : ''}. 누르면 ${LABEL[next]}(으)로 바뀝니다`}
      title={`화면 밝기: ${LABEL[mode]}`}
    >
      <Icon mode={mode} />
    </button>
  )
}
