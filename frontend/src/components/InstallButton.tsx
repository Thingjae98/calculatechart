import { useEffect, useState } from 'react'

// beforeinstallprompt 이벤트 타입 (표준 미포함)
type BeforeInstallPromptEvent = Event & {
  prompt: () => Promise<void>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>
}

function isStandalone() {
  return (
    window.matchMedia('(display-mode: standalone)').matches ||
    (navigator as unknown as { standalone?: boolean }).standalone === true
  )
}

/**
 * PWA 설치 버튼.
 * - 미설치 브라우저에서만 노출, 설치된 앱(standalone)에선 숨김.
 * - Android/데스크톱 Chrome: 네이티브 설치 프롬프트 호출.
 * - iOS Safari: beforeinstallprompt 미지원 → "공유 → 홈 화면에 추가" 안내 표시.
 */
function readDeferred(): BeforeInstallPromptEvent | null {
  const e = (window as unknown as { __deferredInstall?: Event }).__deferredInstall
  return e ? (e as BeforeInstallPromptEvent) : null
}

function isIosSafari() {
  const ua = navigator.userAgent
  return /iphone|ipad|ipod/i.test(ua) && /safari/i.test(ua) && !/crios|fxios|edgios/i.test(ua)
}

export function InstallButton() {
  // 초기값을 렌더 시점에 계산 (effect 내 setState 회피)
  const [deferred, setDeferred] = useState<BeforeInstallPromptEvent | null>(() =>
    isStandalone() ? null : readDeferred(),
  )
  const [visible, setVisible] = useState<boolean>(() =>
    isStandalone() ? false : !!readDeferred() || isIosSafari(),
  )
  const [showIosHint, setShowIosHint] = useState(false)

  useEffect(() => {
    if (isStandalone()) return

    const onInstallable = () => {
      const e = readDeferred()
      if (e) {
        setDeferred(e)
        setVisible(true)
      }
    }
    const onInstalled = () => {
      setVisible(false)
      setDeferred(null)
    }

    window.addEventListener('pwa-installable', onInstallable)
    window.addEventListener('appinstalled', onInstalled)

    return () => {
      window.removeEventListener('pwa-installable', onInstallable)
      window.removeEventListener('appinstalled', onInstalled)
    }
  }, [])

  if (!visible) return null

  const onClick = async () => {
    if (deferred) {
      await deferred.prompt()
      const choice = await deferred.userChoice
      if (choice.outcome === 'accepted') setVisible(false)
      setDeferred(null)
      ;(window as unknown as { __deferredInstall?: Event }).__deferredInstall = undefined
    } else {
      setShowIosHint(true) // iOS 안내
    }
  }

  return (
    <>
      <button className="installBtn" onClick={onClick} aria-label="앱 설치">
        <span aria-hidden="true">⬇</span> 앱 설치
      </button>

      {showIosHint && (
        <div className="installHint" role="dialog" aria-modal="true" onClick={() => setShowIosHint(false)}>
          <div className="installHintCard" onClick={(e) => e.stopPropagation()}>
            <div className="installHintTitle">홈 화면에 앱으로 추가</div>
            <p>
              Safari 하단(또는 상단)의 <strong>공유 버튼</strong>
              <span aria-hidden="true"> ⬆️ </span>을 누른 뒤,
              <br />
              <strong>‘홈 화면에 추가’</strong>를 선택하면 앱처럼 설치됩니다.
            </p>
            <button className="installHintClose" onClick={() => setShowIosHint(false)}>
              확인
            </button>
          </div>
        </div>
      )}
    </>
  )
}
