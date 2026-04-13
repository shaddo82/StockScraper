from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import datetime

app = FastAPI(title="Stock Dashboard")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 기본 주식 목록
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


@app.get("/api/stocks")
def get_stocks():
    """현재 주식 가격 및 변동률 조회"""
    stocks = []
    for ticker in TICKERS:
        try:
            data = yf.Ticker(ticker)
            hist = data.history(period="5d")  # 5일 데이터로 변동률 계산

            if len(hist) >= 2:
                current_price = hist["Close"].iloc[-1]
                previous_price = hist["Close"].iloc[-2]
                change_percent = ((current_price - previous_price) / previous_price) * 100

                stocks.append({
                    "ticker": ticker,
                    "price": round(current_price, 2),
                    "change": round(change_percent, 2),
                    "previous_price": round(previous_price, 2)
                })
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    return {
        "stocks": stocks,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/health")
def health_check():
    """헬스 체크"""
    return {"status": "ok"}


# 정적 파일 마운트
app.mount("/", StaticFiles(directory="static", html=True), name="static")
