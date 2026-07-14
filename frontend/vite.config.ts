import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.svg', 'apple-touch-icon-180x180.png'],
      manifest: {
        name: '모닝 마켓 브리핑',
        short_name: '마켓브리핑',
        description: '매일 아침 한국 증시 브리핑 + 반도체 투톱(삼성전자·SK하이닉스) 전략 + 차트 분석',
        lang: 'ko',
        theme_color: '#aa3bff',
        background_color: '#16171d',
        display: 'standalone',
        orientation: 'portrait',
        start_url: '/',
        scope: '/',
        icons: [
          { src: 'pwa-64x64.png', sizes: '64x64', type: 'image/png' },
          { src: 'pwa-192x192.png', sizes: '192x192', type: 'image/png' },
          { src: 'pwa-512x512.png', sizes: '512x512', type: 'image/png' },
          { src: 'maskable-icon-512x512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,svg,png,ico,woff2}'],
        // 브리핑 JSON은 최신을 우선하되 오프라인이면 캐시로 폴백
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname.startsWith('/briefings/'),
            handler: 'NetworkFirst',
            options: {
              cacheName: 'briefings',
              networkTimeoutSeconds: 5,
              expiration: { maxEntries: 30, maxAgeSeconds: 60 * 60 * 24 * 14 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
        ],
        // 백엔드 주가 API는 캐시하지 않음 (항상 실시간)
        navigateFallbackDenylist: [/^\/api\//],
      },
      devOptions: {
        enabled: false,
      },
    }),
  ],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
