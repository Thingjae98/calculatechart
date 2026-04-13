# CLAUDE.md - CalculateChart 개발 컨텍스트

## 프로젝트 개요

한국 주식(KRX) 차트 기술적 분석 + 점수화 + N일 예측 웹 앱.
개발자(명재)가 가족(부모님, 누나)에게 제공하려는 목적으로 만든 개인 프로젝트.

## 아키텍처

- **프론트엔드**: React 19 + TypeScript + Vite. `lightweight-charts` v5로 캔들/거래량/이평선 렌더링.
- **백엔드**: FastAPI (Python). `pykrx`로 KRX OHLCV 데이터 취득, `scipy`로 지지/저항선 탐지.
- **배포**: Render.com 무료 플랜 (cold start 이슈 → `/api/ping` 웜업으로 대응). 프론트엔드는 Vercel.

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `backend/main.py` | 전체 백엔드 (엔드포인트, 기술적 분석, 예측 캔들, 추천) |
| `frontend/src/App.tsx` | 메인 UI (검색 자동완성, 예측 버튼, 추천 목록) |
| `frontend/src/components/CandleChart.tsx` | 차트 컴포넌트 (캔들+거래량+이평선+예측+스크롤 로드) |
| `frontend/src/lib/api.ts` | API 클라이언트 + TypeScript 타입 정의 |
| `frontend/src/App.css` | 전체 UI 스타일 (드롭다운, 카드, 차트) |
| `frontend/src/index.css` | CSS 변수 (다크모드/라이트모드 테마) |
| `frontend/index.html` | HTML 엔트리 (탭 타이틀: "차트 분석") |

## 개발 명령어

```bash
# 백엔드 (반드시 먼저 실행)
cd backend && uvicorn main:app --reload --port 8000

# 프론트엔드 (Vite 프록시가 /api/* → localhost:8000 전달)
cd frontend && npm run dev

# 빌드
cd frontend && npm run build
```

## 배포 설정

- **백엔드**: `render.yaml` → Render.com 자동 배포
- **프론트엔드**: Vercel에 `VITE_API_BASE_URL` 환경변수 필수 설정
- 로컬 개발 시 `VITE_API_BASE_URL`은 비워두면 Vite 프록시 사용

## 주의 사항

- `pykrx`는 동기 라이브러리 → `ThreadPoolExecutor`로 감싸서 비동기 호출 + 타임아웃 적용
- `pandas_ta`는 설치 실패 가능 → import 실패 시 수동 RSI 계산으로 폴백
- 종목 리스트(`_load_listing`)는 fdr 우선 + pykrx 폴백, 1시간 캐시 + startup 사전 로드
- 추천 종목은 1시간 메모리 캐시. 캐시 없으면 220종목 순회하므로 수 분 소요
- CandleChart에서 `freshLoadId` prop으로 신규 로드 vs 스크롤 추가 로드를 구분
- 예측 캔들은 프론트에서 별도 CandlestickSeries(반투명)로 렌더링
- OHLCV 데이터에서 NaN/0값 행은 백엔드+프론트엔드 양쪽에서 필터링

## 예측 모델 핵심 로직

`_generate_predicted_candles()` 함수 (backend/main.py):

1. **변동성 분석 (먼저 수행)**: 과거 120일 일일 변화율 → 연환산 변동성·P95·중위값 계산
2. **레짐 분류**: 극단적(연환산 60%↑ or 일변동 10%↑) / 안정적(20%↓) / 보통
3. **적응형 댐프닝**:
   - 극단적: drift ±0.15%, 수익률 캡 3%, sigma를 중위값 수준으로 축소
   - 장기(21~30일): 수익률 캡 4%, 누적 이탈 ±25%
   - 단기(7~14일): 수익률 캡 6%, 누적 이탈 ±35%
4. **누적 가격 캡**: 시작가 대비 최대 ±N% 초과 불가 (음수/폭등 방지)
5. **GARCH 안정화**: 0.92/0.08 가중 + running_sigma 상한(base의 2배)
6. **지지/저항 자석효과**: S/R 레벨 근처에서 반발/끌림력 적용

## 지지/저항선 탐지

- `scipy.signal.find_peaks`로 봉우리/골 탐지
- 최근성 가중치: 최근 레벨에 보너스 (1.0~1.5배)
- 클러스터링: 2% 이내 레벨 병합, 최대 3개씩 표시
- 유효 범위: 현재가 ±40% 이내만 표시

## 커스텀 지표 설명

- **수박지표**: 볼린저밴드 스퀴즈(밴드폭 수축) + 이동평균선 수렴(5/20/60/224일 편차 < 2%)이 동시에 발생하면 추세 전환 임박 신호
- **계단지표**: ATR의 0.5배를 그리드 크기로 하여 종가를 계단화. 동일 레벨 3회 이상 유지 + 레벨 상향 시 상승 확인
