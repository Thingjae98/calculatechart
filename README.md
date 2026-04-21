# 📈 CalculateChart — 한국 주식 기술적 분석기

> KRX 전 종목을 기술적 분석으로 자동 점수화하고, AI 주가 예측까지 제공하는 풀스택 주식 분석 웹 앱

**🔗 라이브 데모: [calculatechart.vercel.app](https://calculatechart.vercel.app)**

---

## 📸 미리보기

> *(아래 항목을 직접 스크린샷 찍어 삽입해주세요)*
>
> 권장 캡처 항목:
> 1. 캔들스틱 차트 + 이동평균선 화면
> 2. 차트 점수 패널 (0~100 점수 표시)
> 3. 추천 종목 Top 10 화면
>
> 삽입 방법: `![메인 차트](./docs/screenshot-main.png)` 형태로 추가

---

## ✨ 주요 기능

| 기능 | 설명 |
|---|---|
| 📊 캔들스틱 차트 | 거래량 히스토그램 + 이동평균선(5/20/60/120일) 통합 표시 |
| 🎯 차트 점수 (0~100) | RSI, MACD, 볼린저 밴드 등 7개 지표를 종합한 자동 점수화 |
| 🔮 N일 주가 예측 | 7/14/21/30일 단위 미래 캔들 생성 (단기 모멘텀 + 장기 SMA 평균 회귀) |
| 📌 지지/저항선 | 피봇 포인트 + 거래량·터치 횟수·최근성 종합 점수로 유의미한 선 자동 표시 |
| 🏆 추천 종목 Top 10 | KRX 전 종목 스코어링 후 상위 종목 자동 추천 |
| 🔄 스크롤 과거 데이터 로드 | 차트 좌측 스크롤 시 과거 데이터 자동 페이징 |

---

## 🛠 기술 스택

| 계층 | 기술 |
|---|---|
| **Frontend** | React 19, TypeScript, Vite, lightweight-charts v5 |
| **Backend** | FastAPI, pandas, numpy, scipy, pykrx, FinanceDataReader |
| **배포** | Vercel (Frontend), Render (Backend) |

### 프론트/백엔드 완전 분리 아키텍처

```
[브라우저 - Vercel]          [API 서버 - Render]
React 19 + TypeScript   →   FastAPI + Python
       ↓                          ↓
   캔들차트 렌더링           KRX 데이터 수집 + 기술적 분석
   점수 표시 UI             RSI / MACD / 볼린저 밴드 계산
   예측 캔들 시각화          미래 캔들 생성 알고리즘
```

---

## 📐 기술적 분석 지표

| 지표 | 점수 영향 | 설명 |
|---|---|---|
| 이동평균 정배열 | +15 | 단기 > 중기 > 장기 MA |
| RSI(14) | +15 / -15 | 과매도(≤30) / 과매수(≥70) |
| MACD | +10 / -10 | 골든/데드크로스 + 히스토그램 방향 |
| 볼린저 밴드 | +10 / -10 | 하단 지지 / 상단 저항 |
| 수박지표 | +10 / +5 | BB 스퀴즈 + 이평 수렴 감지 (자체 개발) |
| 계단지표 | +5 / -5 | ATR 기반 스텝 지지레벨 방향 (자체 개발) |
| 돌파 후 눌림목 | +7 | BB/박스 돌파 후 거래량 감소하며 건전한 조정 |

---

## 💡 트러블슈팅 & 기술적 의사결정

### Render Cold Start 대응
Render 무료 티어는 일정 시간 요청이 없으면 서버가 슬립 상태로 전환됩니다.  
이를 해결하기 위해 **서버 웨이크업 감지 → 자동 재시도 + 사용자 안내 메시지** 흐름을 구현했습니다.

```python
# /api/ping 엔드포인트로 Cold Start 상태 감지
# 프론트엔드에서 최초 요청 전 ping 후 응답 대기 UX 처리
```

### 지지/저항선 신뢰도 점수 시스템
단순 피봇 포인트만으로는 의미 없는 선이 너무 많이 표시되는 문제가 있었습니다.  
거래량, 터치 횟수, 최근성을 가중 합산한 **신뢰도 점수**를 도입해 가장 유의미한 선 1개씩만 표시하도록 개선했습니다.

---

## 🚀 로컬 실행

### 백엔드 (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 프론트엔드 (React)
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

> Vite 개발 서버는 `/api/*` 요청을 `http://127.0.0.1:8000`으로 자동 프록시합니다.

---

## 🗂 프로젝트 구조

```
calculatechart/
├── backend/
│   ├── main.py              # FastAPI 서버 + 기술적 분석 로직 전체
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # 메인 UI + 상태 관리
│   │   ├── components/
│   │   │   └── CandleChart.tsx  # 차트 (캔들+거래량+이평선+예측)
│   │   └── lib/
│   │       └── api.ts        # API 클라이언트
│   └── vite.config.ts
├── render.yaml               # Render 배포 설정
├── CHANGELOG.md              # 버전별 변경 이력
└── ROADMAP.md                # 향후 개발 계획
```

---

## 🌱 향후 개발 계획 (ROADMAP)

- [ ] 백테스팅 기능 — 과거 데이터로 전략 수익률 검증
- [ ] 알림 기능 — 목표 점수 도달 시 이메일/슬랙 알림
- [ ] 즐겨찾기 종목 관리
- [ ] 차트 패턴 자동 감지 (헤드앤숄더, 삼중 바닥 등)

---

## 📄 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `VITE_API_BASE_URL` | (빈 문자열) | 프론트엔드 → 백엔드 API URL |
| `RECOMMEND_SAMPLE_SIZE` | 220 | 추천 스코어링할 종목 수 |
| `PYKRX_TIMEOUT_SEC` | 30 | pykrx 호출 타임아웃(초) |
