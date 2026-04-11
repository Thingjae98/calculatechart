# CLAUDE.md - CalculateChart 개발 컨텍스트

## 프로젝트 개요

한국 주식(KRX) 차트 기술적 분석 + 점수화 + N일 예측 웹 앱.
개발자(명재)가 가족(부모님, 누나)에게 제공하려는 목적으로 만든 개인 프로젝트.

## 아키텍처

- **프론트엔드**: React 19 + TypeScript + Vite. `lightweight-charts` v5로 캔들/거래량/이평선 렌더링.
- **백엔드**: FastAPI (Python). `pykrx`로 KRX OHLCV 데이터 취득, `scipy`로 지지/저항선 탐지.
- **배포**: Render.com 무료 플랜 (cold start 이슈 → `/api/ping` 웜업으로 대응).

## 핵심 파일

- `backend/main.py` — 전체 백엔드 로직 (엔드포인트, 기술적 분석, 예측 캔들 생성)
- `frontend/src/App.tsx` — 메인 UI (검색, 예측 버튼, 추천 목록)
- `frontend/src/components/CandleChart.tsx` — 차트 컴포넌트 (캔들+거래량+이평선+예측+스크롤 로드)
- `frontend/src/lib/api.ts` — API 클라이언트 + TypeScript 타입 정의

## 개발 명령어

```bash
# 백엔드
cd backend && uvicorn main:app --reload --port 8000

# 프론트엔드
cd frontend && npm run dev

# 빌드
cd frontend && npm run build
```

## 주의 사항

- `pykrx`는 동기 라이브러리 → `ThreadPoolExecutor`로 감싸서 비동기 호출 + 타임아웃 적용
- `pandas_ta`는 설치 실패 가능 → import 실패 시 수동 RSI 계산으로 폴백
- 추천 종목은 1시간 메모리 캐시. 캐시 없으면 220종목 순회하므로 수 분 소요
- CandleChart에서 `freshLoadId` prop으로 신규 로드 vs 스크롤 추가 로드를 구분
- 예측 캔들은 프론트에서 별도 CandlestickSeries(보라색)로 렌더링

## 커스텀 지표 설명

- **수박지표**: 볼린저밴드 스퀴즈(밴드폭 수축) + 이동평균선 수렴(5/20/60/224일 편차 < 2%)이 동시에 발생하면 추세 전환 임박 신호
- **계단지표**: ATR의 0.5배를 그리드 크기로 하여 종가를 계단화. 동일 레벨 3회 이상 유지 + 레벨 상향 시 상승 확인

## 향후 개선 방향

- 일목균형표 추가 (한국 차트 분석의 핵심)
- 보조지표 패널 분리 (RSI/MACD 차트 아래 별도 영역)
- 캔들 패턴 자동 인식 (해머, 도지, 장악형 등)
- 주봉/월봉 지원
- 피보나치 되돌림/확장
