from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List
from types import SimpleNamespace
import json
import os
import time

from app import config
from app.model_loader import get_model_info, load_model
from ml.features import build_latest_feature_frame

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = SimpleNamespace(Ticker=None)

app = FastAPI()

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 종목 저장 파일
STOCKS_FILE = "stocks.json"

# 기본 종목 리스트
DEFAULT_STOCKS = ["AAPL", "GOOGL", "MSFT"]

# 최대 종목 개수
MAX_STOCKS = 20

# yfinance는 무료 API라 Render에서 짧은 시간에 반복 호출하면 rate limit이 자주 발생합니다.
STOCKS_CACHE_TTL_SECONDS = 60
_stocks_cache = {
    "expires_at": 0.0,
    "data": None,
}

# 한글/별칭 회사명 -> 티커 매핑
COMPANY_NAME_TO_SYMBOL = {
    "테슬라": "TSLA",
    "마이크로소프트": "MSFT",
    "마이크로 소프트": "MSFT",
    "애플": "AAPL",
    "구글": "GOOGL",
    "알파벳": "GOOGL",
    "엔비디아": "NVDA",
    "메타": "META",
    "아마존": "AMZN",
}


def build_unavailable_stock(symbol: str, error: str) -> dict:
    return {
        "symbol": symbol,
        "current_price": None,
        "previous_price": None,
        "change_percent": None,
        "prediction": {
            "symbol": symbol,
            "prediction": None,
            "direction": "예측 불가",
            "confidence": None,
            "error": error,
        },
        "error": error,
    }


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


def normalize_symbol_input(raw_symbol: str) -> str:
    """사용자 입력(티커/회사명)을 표준 티커로 변환"""
    cleaned = raw_symbol.strip()
    if not cleaned:
        return ""

    # 공백/대소문자 영향을 줄인 키로 회사명 별칭 처리
    alias_key = cleaned.replace(" ", "").lower()
    for company_name, symbol in COMPANY_NAME_TO_SYMBOL.items():
        normalized_name = company_name.replace(" ", "").lower()
        if alias_key == normalized_name:
            return symbol

    return cleaned.upper()


def _predict_stock_direction_from_history(symbol: str, history) -> dict:
    try:
        if history is None or len(history) < 15:
            raise HTTPException(
                status_code=400,
                detail=f"'{symbol}' 예측에 필요한 히스토리 데이터가 부족합니다",
            )

        feature_frame = build_latest_feature_frame(history)
        model = load_model()
        prediction = int(model.predict(feature_frame)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(feature_frame)[0]
            if len(probabilities) > 1:
                probability = float(probabilities[1])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "symbol": symbol,
        "prediction": prediction,
        "direction": "상승" if prediction == 1 else "하락",
        "confidence": probability,
        "model_info": get_model_info(),
    }


def _predict_stock_direction(symbol: str) -> dict:
    if yf.Ticker is None:
        raise HTTPException(
            status_code=503,
            detail="yfinance is not installed in this environment",
        )

    ticker = yf.Ticker(symbol)
    history = ticker.history(period="6mo")
    return _predict_stock_direction_from_history(symbol, history)


def _get_stock_prediction_or_none(symbol: str) -> dict:
    try:
        return _predict_stock_direction(symbol)
    except Exception as exc:
        return {
            "symbol": symbol,
            "prediction": None,
            "direction": "예측 불가",
            "confidence": None,
            "error": str(exc),
        }


# ============ ROOT ROUTE ============
@app.get("/")
async def root():
    """메인 페이지 (index.html 반환)"""
    return FileResponse("static/index.html")


# ============ API ROUTES ============
@app.get("/api/stocks")
async def get_stocks():
    """주식 데이터 조회 API"""
    try:
        cached_data = _stocks_cache["data"]
        if cached_data is not None and time.time() < _stocks_cache["expires_at"]:
            return cached_data

        symbols = load_stocks()
        stocks = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="6mo")

                if len(data) < 2:
                    stocks.append(
                        build_unavailable_stock(
                            symbol,
                            "가격 데이터가 부족합니다",
                        )
                    )
                    continue

                current_price = data["Close"].iloc[-1]
                previous_price = data["Close"].iloc[-2]
                change_percent = calculate_price_change(current_price, previous_price)

                try:
                    prediction = _predict_stock_direction_from_history(symbol, data)
                except Exception as exc:
                    prediction = {
                        "symbol": symbol,
                        "prediction": None,
                        "direction": "예측 불가",
                        "confidence": None,
                        "error": str(exc),
                    }

                stocks.append({
                    "symbol": symbol,
                    "current_price": float(current_price),
                    "previous_price": float(previous_price),
                    "change_percent": float(change_percent),
                    "prediction": prediction,
                })

            except Exception as e:
                print(f"⚠️  {symbol} 데이터 수집 실패: {e}")
                stocks.append(build_unavailable_stock(symbol, str(e)))

        response = {
            "stocks": stocks,
            "count": len(stocks)
        }
        _stocks_cache["data"] = response
        _stocks_cache["expires_at"] = time.time() + STOCKS_CACHE_TTL_SECONDS
        return response

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
    symbol = normalize_symbol_input(request.symbol)

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
        "stocks_file_exists": os.path.exists(STOCKS_FILE),
        "model_file_exists": os.path.exists(config.MODEL_PATH),
    }


@app.get("/api/model/info")
async def model_info():
    """현재 모델 로드 상태 조회"""
    return get_model_info()


@app.get("/api/stocks/predict/{symbol}")
async def predict_stock(symbol: str):
    """다음 거래일 상승/하락 예측 API"""
    normalized_symbol = normalize_symbol_input(symbol)
    if not normalized_symbol:
        raise HTTPException(status_code=400, detail="유효하지 않은 종목 코드입니다")
    return _predict_stock_direction(normalized_symbol)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
