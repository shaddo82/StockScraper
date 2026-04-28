# StockScraper

FastAPI 기반 미국 주식 조회 및 관심 종목 관리 서비스입니다.

## Tech Stack

- Python 3.11
- FastAPI
- Uvicorn
- yfinance
- pytest

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

브라우저에서 `http://localhost:8000` 접속

## Test

```bash
pytest
```

## Docker Run

```bash
docker build -t stockscraper:local .
docker run --rm -p 10000:10000 stockscraper:local
```

브라우저에서 `http://localhost:10000` 또는 `http://localhost:10000/docs` 접속

## CI

- `.github/workflows/ci.yml`: Python 3.11 테스트 CI
- `.github/workflows/docker-verify.yml`: Docker 빌드 + 헬스체크 CI

## Render Deploy

`render.yaml`이 포함되어 있어 Render Blueprint로 배포할 수 있습니다.

1. Render에서 GitHub 저장소 연결
2. `Blueprint` 생성 시 저장소 루트의 `render.yaml` 사용
3. 배포 완료 후 `/api/health` 확인
