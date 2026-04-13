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
    """현재 주식 가격 조회"""
    stocks = []
    for ticker in TICKERS:
        data = yf.Ticker(ticker)
        hist = data.history(period="1d")
        if not hist.empty:
            price = hist["Close"].iloc[-1]
            stocks.append({
                "ticker": ticker,
                "price": round(price, 2),
                "change": 0  # 추후 추가
            })
    return {"stocks": stocks, "timestamp": datetime.now().isoformat()}

@app.get("/api/health")
def health_check():
    """헬스 체크"""
    return {"status": "ok"}

# 정적 파일 마운트
app.mount("/", StaticFiles(directory="static", html=True), name="static")
