from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import yfinance as yf
from typing import List
import json
import os

app = FastAPI()

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 종목 저장 파일
STOCKS_FILE = "stocks.json"

# 기본 종목 리스트
DEFAULT_STOCKS = ["AAPL", "GOOGL", "MSFT"]

# 최대 종목 개수
MAX_STOCKS = 20


class StockRequest(BaseModel):
    """종목 추가 요청 모델"""
    symbol: str


def load_stocks() -> List[str]:
    """저장된 종목 리스트 로드"""
    if os.path.exists(STOCKS_FILE):
        with open(STOCKS_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_STOCKS.copy()


def save_stocks(stocks: List[str]) -> None:
    """종목 리스트 저장"""
    with open(STOCKS_FILE, "w") as f:
        json.dump(stocks, f)


def calculate_price_change(current_price: float, previous_price: float) -> float:
    """가격 변화율 계산"""
    if previous_price == 0:
        return 0.0
    return ((current_price - previous_price) / previous_price) * 100


def is_valid_symbol(symbol: str) -> bool:
    """종목 코드 유효성 확인 (yfinance에서 데이터 가져올 수 있는지 확인)"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        return len(data) > 0
    except Exception:
        return False


@app.get("/api/stocks")
async def get_stocks():
    """주식 데이터 조회 API"""
    try:
        symbols = load_stocks()
        stocks = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="2d")

                if len(data) < 2:
                    continue

                current_price = data["Close"].iloc[-1]
                previous_price = data["Close"].iloc[-2]
                change_percent = calculate_price_change(current_price, previous_price)

                stocks.append({
                    "symbol": symbol,
                    "current_price": float(current_price),
                    "previous_price": float(previous_price),
                    "change_percent": float(change_percent)
                })

            except Exception as e:
                print(f"⚠️  {symbol} 데이터 수집 실패: {e}")
                continue

        return {
            "stocks": stocks,
            "count": len(stocks)
        }

    except Exception as e:
        print(f"❌ API 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/api/stocks/list")
async def get_stock_list():
    """현재 종목 리스트 조회"""
    try:
        symbols = load_stocks()
        return {
            "symbols": symbols,
            "count": len(symbols),
            "max_stocks": MAX_STOCKS
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/api/stocks/add")
async def add_stock(request: StockRequest):
    """종목 추가 API"""
    symbol = request.symbol.upper().strip()

    # 종목 코드 유효성 확인
    if not symbol or len(symbol) > 5:
        raise HTTPException(
            status_code=400,
            detail="유효하지 않은 종목 코드입니다 (1-5자)"
        )

    # 종목 존재 여부 확인
    if not is_valid_symbol(symbol):
        raise HTTPException(
            status_code=400,
            detail=f"'{symbol}' 종목을 찾을 수 없습니다"
        )

    # 현재 종목 리스트 로드
    symbols = load_stocks()

    # 중복 확인
    if symbol in symbols:
        raise HTTPException(
            status_code=400,
            detail=f"'{symbol}' 종목은 이미 추가되어 있습니다"
        )

    # 최대 개수 확인
    if len(symbols) >= MAX_STOCKS:
        raise HTTPException(
            status_code=400,
            detail=f"최대 {MAX_STOCKS}개까지만 추가할 수 있습니다"
        )

    # 종목 추가
    symbols.append(symbol)
    save_stocks(symbols)

    return {
        "message": f"'{symbol}' 종목이 추가되었습니다",
        "symbol": symbol,
        "total_count": len(symbols)
    }


@app.delete("/api/stocks/remove/{symbol}")
async def remove_stock(symbol: str):
    """종목 삭제 API"""
    symbol = symbol.upper().strip()

    # 현재 종목 리스트 로드
    symbols = load_stocks()

    # 종목 존재 확인
    if symbol not in symbols:
        raise HTTPException(
            status_code=404,
            detail=f"'{symbol}' 종목을 찾을 수 없습니다"
        )

    # 종목 삭제
    symbols.remove(symbol)
    save_stocks(symbols)

    return {
        "message": f"'{symbol}' 종목이 삭제되었습니다",
        "symbol": symbol,
        "total_count": len(symbols)
    }


@app.get("/api/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "stocks_file_exists": os.path.exists(STOCKS_FILE)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
