import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// PWA 설치 프롬프트 이벤트는 React 마운트보다 먼저 발생할 수 있어 미리 잡아둔다.
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault()
  ;(window as unknown as { __deferredInstall?: Event }).__deferredInstall = e
  window.dispatchEvent(new Event('pwa-installable'))
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
