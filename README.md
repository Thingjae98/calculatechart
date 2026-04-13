# CalculateChart - 차트 점수 기반 주식 분석기

한국 주식(KRX) 차트를 기술적 분석으로 점수화하고, AI 기반 주가 예측까지 제공하는 웹 앱입니다.

## 주요 기능

- **캔들스틱 차트** + 거래량 히스토그램 + 이동평균선(5/20/60/120일)
- **차트 점수** (0~100): 지지선 근접, RSI 과매도, 장기 추세, 거래량 반등, 돌파/눌림목을 종합 평가
- **N일 주가 예측** (7/14/21/30일): 단기는 모멘텀 중심, 장기는 다중 SMA(20/60/120) 평균 회귀 기반 미래 캔들 생성
- **지지/저항선**: 피봇 포인트 + 거래량·터치 횟수·최근성 종합 점수로 가장 유의미한 1개씩 표시
- **수렴→돌파→눌림목 감지**: BB/박스 돌파 + 거래량 확인 + 건전한 되돌림 판별
- **커스텀 지표**: 수박지표(BB 스퀴즈+이평 수렴), 계단지표(ATR 기반 스텝 지지)
- **추천 종목 Top 10**: KRX 전 종목 스코어링 후 상위 추천
- **스크롤 기반 과거 데이터 자동 로드**: 차트를 왼쪽으로 스크롤하면 추가 데이터 자동 로드
- **Render cold start 대응**: 서버 슬립 시 자동 웨이크업 + 안내 메시지

## 기술 스택

| 계층 | 기술 |
|------|------|
| Frontend | React 19, TypeScript, Vite, lightweight-charts v5 |
| Backend | FastAPI, pandas, numpy, scipy, pykrx, FinanceDataReader |
| Deploy | Render.com (backend), Vercel/Render (frontend) |

## 로컬 개발

### 백엔드

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 프론트엔드

```bash
cd frontend
npm install
npm run dev
```

개발 시 Vite 프록시가 `/api/*` 요청을 `http://127.0.0.1:8000`으로 전달합니다.

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `VITE_API_BASE_URL` | (빈 문자열) | 프론트엔드 → 백엔드 API URL |
| `RECOMMEND_SAMPLE_SIZE` | 220 | 추천 스코어링할 종목 수 |
| `PYKRX_TIMEOUT_SEC` | 30 | pykrx 호출 타임아웃(초) |

## 프로젝트 구조

```
calculatechart/
├── backend/
│   ├── main.py              # FastAPI 서버 + 기술적 분석 로직
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # 메인 UI 컴포넌트
│   │   ├── App.css           # 스타일
│   │   ├── components/
│   │   │   └── CandleChart.tsx  # 차트 (캔들+거래량+이평선+예측)
│   │   ├── lib/
│   │   │   └── api.ts        # API 클라이언트
│   │   └── index.css         # 글로벌 CSS 변수
│   ├── package.json
│   └── vite.config.ts
├── render.yaml               # Render 배포 설정
└── README.md
```

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/ping` | 서버 웨이크업 |
| GET | `/api/stock/{ticker}` | OHLCV + 점수 + 지지/저항선 |
| GET | `/api/stock/{ticker}/predict?n_days=7` | 기술적 분석 예측 + 미래 캔들 |
| GET | `/api/recommendations?limit=10` | 추천 종목 Top N |

## 기술적 분석 지표

| 지표 | 점수 영향 | 설명 |
|------|-----------|------|
| 이동평균 정배열 | +15 | 단기 > 중기 > 장기 MA |
| RSI(14) | +15/-15 | 과매도(<=30) / 과매수(>=70) |
| MACD | +10/-10 | 골든/데드크로스 + 히스토그램 방향 |
| 볼린저 밴드 | +10/-10 | 하단 지지 / 상단 저항 |
| 수박지표 | +10/+5 | BB 스퀴즈 + 이평 수렴 감지 |
| 계단지표 | +5/-5 | ATR 기반 스텝 지지레벨 방향 |
| 돌파 후 눌림목 | +7 | BB/박스 돌파 후 거래량 감소하며 건전한 조정 |
