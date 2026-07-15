// 라이트/다크 테마 관리
//
// 설계: CSS는 `@media (prefers-color-scheme)`를 쓰지 않고 `:root[data-theme]`만 본다.
// 시스템 설정 해석은 JS가 전담해서 data-theme를 'light' | 'dark'로 **확정**해 둔다.
// 이유: 미디어쿼리를 남겨두면 "시스템 다크 + 수동 라이트" 같은 조합을 표현하려고
// 다크 토큰을 CSS 파일마다 두 벌씩 복제해야 한다(index/App/Briefing 3곳).
// 확정 방식이면 CSS는 라이트 1벌 + 다크 1벌로 끝난다.
//
// 첫 페인트 전 적용은 index.html의 인라인 스크립트가 담당한다(깜빡임 방지).
// 이 파일과 그 스크립트는 STORAGE_KEY / data-theme 규칙을 공유하므로 함께 고쳐야 한다.

export type ThemeMode = 'system' | 'light' | 'dark'
export type ResolvedTheme = 'light' | 'dark'

export const THEME_STORAGE_KEY = 'cc-theme'

const isMode = (v: unknown): v is ThemeMode => v === 'system' || v === 'light' || v === 'dark'

/** 저장된 사용자 선택. 없거나 깨졌으면 'system' */
export function getStoredMode(): ThemeMode {
  try {
    const raw = localStorage.getItem(THEME_STORAGE_KEY)
    return isMode(raw) ? raw : 'system'
  } catch {
    // 사파리 프라이빗 모드 등에서 localStorage 접근이 막힐 수 있다
    return 'system'
  }
}

export function storeMode(mode: ThemeMode): void {
  try {
    if (mode === 'system') localStorage.removeItem(THEME_STORAGE_KEY)
    else localStorage.setItem(THEME_STORAGE_KEY, mode)
  } catch {
    /* 저장 실패는 무시 — 이번 세션에만 적용된다 */
  }
}

export function systemPrefersDark(): boolean {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
}

export function resolveTheme(mode: ThemeMode): ResolvedTheme {
  if (mode === 'system') return systemPrefersDark() ? 'dark' : 'light'
  return mode
}

/** 해석된 테마를 <html data-theme>에 반영 */
export function applyTheme(mode: ThemeMode): ResolvedTheme {
  const resolved = resolveTheme(mode)
  document.documentElement.dataset.theme = resolved
  return resolved
}

/** 현재 <html data-theme> 값 */
export function currentTheme(): ResolvedTheme {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'
}

/**
 * 확정된 테마를 구독한다. CSS가 아니라 JS로 색을 정하는 곳(캔들 차트)에서 쓴다.
 * data-theme 변화를 직접 관찰하므로, 시스템 변경이든 수동 토글이든 모두 잡힌다.
 */
export function subscribeTheme(cb: (t: ResolvedTheme) => void): () => void {
  const observer = new MutationObserver(() => cb(currentTheme()))
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
  return () => observer.disconnect()
}
