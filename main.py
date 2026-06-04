import hashlib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List
from types import SimpleNamespace
import json
import os
import time
import uuid

import pandas as pd

from app import config
from app.feedback import read_feedback_logs, save_feedback
from app.model_loader import get_model_info, load_model, reload_model
from app.prediction_logger import read_prediction_logs, save_prediction_log
from app.verification import read_verification_logs, save_verification_log
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
_company_name_cache = {}


def clear_stocks_cache() -> None:
    _stocks_cache["expires_at"] = 0.0
    _stocks_cache["data"] = None

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
    "코카콜라": "KO",
    "코카 콜라": "KO",
}

SYMBOL_TO_COMPANY_NAME = {
    symbol: company_name.replace(" ", "")
    for company_name, symbol in COMPANY_NAME_TO_SYMBOL.items()
}


def get_company_name(symbol: str) -> str:
    symbol = symbol.upper().strip()
    if symbol in _company_name_cache:
        return _company_name_cache[symbol]

    fallback_name = SYMBOL_TO_COMPANY_NAME.get(symbol, symbol)
    if yf.Ticker is None:
        return fallback_name

    try:
        info = yf.Ticker(symbol).get_info()
        name = info.get("shortName") or info.get("longName") or fallback_name
    except Exception:
        name = fallback_name

    _company_name_cache[symbol] = name
    return name


def format_stock_label(symbol: str) -> str:
    company_name = get_company_name(symbol)
    return f"{company_name}({symbol})" if company_name and company_name != symbol else symbol


def build_unavailable_stock(symbol: str, error: str) -> dict:
    return {
        "symbol": symbol,
        "display_name": format_stock_label(symbol),
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


class FeedbackRequest(BaseModel):
    prediction_id: str
    symbol: str
    prediction: int
    correct_label: int
    confidence: float | None = None
    deployment: str
    model_uri: str


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
    clear_stocks_cache()


def calculate_price_change(current_price: float, previous_price: float) -> float:
    """가격 변화율 계산"""
    if previous_price == 0:
        return 0.0
    return ((current_price - previous_price) / previous_price) * 100


def _download_stock_histories(symbols: List[str]) -> dict:
    if not symbols or not hasattr(yf, "download"):
        return {}

    try:
        data = yf.download(
            tickers=" ".join(symbols),
            period="6mo",
            group_by="ticker",
            progress=False,
            threads=False,
        )
    except Exception as exc:
        print(f"⚠️  yfinance batch download 실패: {exc}")
        return {}

    if data is None or len(data) == 0:
        return {}

    if len(symbols) == 1:
        return {symbols[0]: data.dropna(how="all")}

    if not isinstance(data.columns, pd.MultiIndex):
        return {}

    histories = {}
    first_level = set(data.columns.get_level_values(0))
    second_level = set(data.columns.get_level_values(1))
    for symbol in symbols:
        if symbol in first_level:
            histories[symbol] = data[symbol].dropna(how="all")
        elif symbol in second_level:
            histories[symbol] = data.xs(symbol, axis=1, level=1).dropna(how="all")
    return histories


def _fetch_stock_history(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        return ticker.history(period="6mo")
    except Exception as exc:
        print(f"⚠️  {symbol} 개별 데이터 조회 실패: {exc}")
        return None


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


def _canary_bucket(symbol: str) -> float:
    digest = hashlib.sha256(symbol.upper().encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def _should_use_challenger(symbol: str) -> bool:
    if not config.CANARY_ENABLED:
        return False
    if config.CANARY_TRAFFIC_RATIO <= 0:
        return False
    if config.CANARY_TRAFFIC_RATIO >= 1:
        return True
    return _canary_bucket(symbol) < config.CANARY_TRAFFIC_RATIO


def _select_model_route(symbol: str) -> tuple[str, str]:
    if _should_use_challenger(symbol):
        return config.CHALLENGER_MODEL_URI, "challenger"
    return config.MODEL_URI, "champion"


def _load_model_for_symbol(symbol: str):
    model_uri, route = _select_model_route(symbol)
    try:
        if route == "challenger":
            model = load_model(model_uri=model_uri, allow_local_fallback=False)
        else:
            model = load_model()
        return model, route, model_uri
    except FileNotFoundError:
        if route != "challenger":
            raise
        model = load_model()
        return model, "champion-fallback", config.MODEL_URI


def _predict_stock_direction_from_history(symbol: str, history) -> dict:
    try:
        if history is None or len(history) < 15:
            raise HTTPException(
                status_code=400,
                detail=f"'{symbol}' 예측에 필요한 히스토리 데이터가 부족합니다",
            )

        feature_frame = build_latest_feature_frame(history)
        model, deployment, model_uri = _load_model_for_symbol(symbol)
        prediction = int(model.predict(feature_frame)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(feature_frame)[0]
            if len(probabilities) > 1:
                probability = float(probabilities[1])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    reference_time = str(pd.Timestamp(history.index[-1]).date())
    reference_close = float(history["Close"].iloc[-1])
    prediction_id = str(uuid.uuid4())
    result = {
        "prediction_id": prediction_id,
        "symbol": symbol,
        "reference_time": reference_time,
        "reference_close": reference_close,
        "prediction": prediction,
        "direction": "상승" if prediction == 1 else "하락",
        "confidence": probability,
        "deployment": deployment,
        "model_uri": model_uri,
        "model_info": get_model_info(model_uri),
    }
    save_prediction_log(
        prediction_id=prediction_id,
        symbol=symbol,
        reference_time=reference_time,
        reference_close=reference_close,
        prediction=prediction,
        direction=result["direction"],
        confidence=probability,
        deployment=deployment,
        model_uri=model_uri,
    )
    return result


def _to_float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int_or_none(value: str | int | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _empty_deployment_metric() -> dict:
    return {
        "prediction_count": 0,
        "avg_confidence": None,
        "low_confidence_count": 0,
        "feedback_count": 0,
        "wrong_feedback_count": 0,
        "wrong_feedback_rate": 0.0,
        "verification_count": 0,
        "verified_correct_count": 0,
        "verified_accuracy": None,
    }


def _get_deployment_metric(metrics: dict[str, dict], deployment: str) -> dict:
    if deployment not in metrics:
        metrics[deployment] = _empty_deployment_metric()
    return metrics[deployment]


def _build_ops_status() -> dict:
    prediction_logs = read_prediction_logs()
    feedback_logs = read_feedback_logs()
    verification_logs = read_verification_logs()
    low_confidence_count = 0
    deployment_counts: dict[str, int] = {}
    deployment_metrics: dict[str, dict] = {}
    confidence_sums: dict[str, float] = {}

    for row in prediction_logs:
        confidence = _to_float_or_none(row.get("confidence"))
        deployment = row.get("deployment") or "unknown"
        metric = _get_deployment_metric(deployment_metrics, deployment)
        metric["prediction_count"] += 1
        if confidence is not None and confidence < 0.65:
            low_confidence_count += 1
            metric["low_confidence_count"] += 1
        if confidence is not None:
            confidence_sums[deployment] = confidence_sums.get(deployment, 0.0) + confidence
        deployment_counts[deployment] = deployment_counts.get(deployment, 0) + 1

    wrong_feedback_count = sum(
        1
        for row in feedback_logs
        if row.get("prediction") != row.get("correct_label")
    )
    for row in feedback_logs:
        deployment = row.get("deployment") or "unknown"
        metric = _get_deployment_metric(deployment_metrics, deployment)
        metric["feedback_count"] += 1
        if row.get("prediction") != row.get("correct_label"):
            metric["wrong_feedback_count"] += 1

    for row in verification_logs:
        deployment = row.get("deployment") or "unknown"
        metric = _get_deployment_metric(deployment_metrics, deployment)
        metric["verification_count"] += 1
        if str(row.get("correct")) in ("1", "true", "True"):
            metric["verified_correct_count"] += 1

    for deployment, metric in deployment_metrics.items():
        if metric["prediction_count"]:
            metric["avg_confidence"] = confidence_sums.get(deployment, 0.0) / metric["prediction_count"]
        if metric["feedback_count"]:
            metric["wrong_feedback_rate"] = metric["wrong_feedback_count"] / metric["feedback_count"]
        if metric["verification_count"]:
            metric["verified_accuracy"] = metric["verified_correct_count"] / metric["verification_count"]

    return {
        "prediction_count": len(prediction_logs),
        "feedback_count": len(feedback_logs),
        "verification_count": len(verification_logs),
        "low_confidence_count": low_confidence_count,
        "deployment_counts": deployment_counts,
        "deployment_metrics": deployment_metrics,
        "wrong_feedback_count": wrong_feedback_count,
        "wrong_feedback_rate": (
            wrong_feedback_count / len(feedback_logs)
            if feedback_logs
            else 0.0
        ),
        "recent_predictions": prediction_logs[-10:],
        "recent_feedback": feedback_logs[-10:],
        "recent_verifications": verification_logs[-10:],
    }


def _find_prediction_log(prediction_id: str) -> dict[str, str]:
    for row in reversed(read_prediction_logs()):
        if row.get("prediction_id") == prediction_id:
            return row
    raise HTTPException(status_code=404, detail="prediction_id에 해당하는 예측 로그가 없습니다")


def _first_close_after(symbol: str, reference_time: str) -> tuple[str, float]:
    if yf.Ticker is None:
        raise HTTPException(status_code=503, detail="yfinance is not installed in this environment")

    reference_date = pd.Timestamp(reference_time).date()
    history = yf.Ticker(symbol).history(period="10d")
    if history is None or history.empty or "Close" not in history.columns:
        raise HTTPException(status_code=400, detail="검증할 실제 가격 데이터가 없습니다")

    close_history = history.dropna(subset=["Close"])
    for index, row in close_history.iterrows():
        actual_date = pd.Timestamp(index).date()
        if actual_date > reference_date:
            return str(actual_date), float(row["Close"])

    raise HTTPException(status_code=400, detail="아직 다음 거래일 가격 데이터가 없습니다")


def _verify_prediction_result(prediction_id: str) -> dict:
    prediction_log = _find_prediction_log(prediction_id)
    reference_time = prediction_log.get("reference_time")
    reference_close = _to_float_or_none(prediction_log.get("reference_close"))
    prediction = _to_int_or_none(prediction_log.get("prediction"))
    symbol = normalize_symbol_input(prediction_log.get("symbol", ""))

    if not reference_time or reference_close is None or prediction is None or not symbol:
        raise HTTPException(status_code=400, detail="검증에 필요한 예측 로그 필드가 부족합니다")

    actual_time, actual_close = _first_close_after(symbol, reference_time)
    actual_label = 1 if actual_close > reference_close else 0
    correct = prediction == actual_label
    result = {
        "prediction_id": prediction_id,
        "symbol": symbol,
        "prediction": prediction,
        "actual_label": actual_label,
        "correct": correct,
        "reference_time": reference_time,
        "reference_close": reference_close,
        "actual_time": actual_time,
        "actual_close": actual_close,
        "deployment": prediction_log.get("deployment") or "unknown",
        "model_uri": prediction_log.get("model_uri") or "",
    }
    save_verification_log(**result)
    return result


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
        histories = _download_stock_histories(symbols)

        for symbol in symbols:
            try:
                data = histories.get(symbol)
                if data is None or len(data) < 2:
                    data = _fetch_stock_history(symbol)

                if data is None or len(data) < 2:
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
                    "display_name": format_stock_label(symbol),
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
            "stocks": [
                {"symbol": symbol, "display_name": format_stock_label(symbol)}
                for symbol in symbols
            ],
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


@app.get("/api/model/canary")
async def canary_info():
    """카나리 배포 설정 조회"""
    return {
        "enabled": config.CANARY_ENABLED,
        "traffic_ratio": config.CANARY_TRAFFIC_RATIO,
        "champion_model_uri": config.MODEL_URI,
        "challenger_model_uri": config.CHALLENGER_MODEL_URI,
    }


@app.get("/api/ops/status")
async def ops_status():
    """운영 로그와 사용자 피드백 통계 조회"""
    return _build_ops_status()


@app.post("/api/model/reload")
async def reload_serving_model():
    """현재 설정된 모델 URI에서 모델을 강제로 다시 로드"""
    try:
        return reload_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/stocks/predict/{symbol}")
async def predict_stock(symbol: str):
    """다음 거래일 상승/하락 예측 API"""
    normalized_symbol = normalize_symbol_input(symbol)
    if not normalized_symbol:
        raise HTTPException(status_code=400, detail="유효하지 않은 종목 코드입니다")
    return _predict_stock_direction(normalized_symbol)


@app.post("/api/predictions/feedback")
async def prediction_feedback(payload: FeedbackRequest):
    """사용자 피드백 저장 API"""
    if payload.prediction not in (0, 1) or payload.correct_label not in (0, 1):
        raise HTTPException(status_code=400, detail="prediction과 correct_label은 0 또는 1이어야 합니다")

    save_feedback(
        prediction_id=payload.prediction_id,
        symbol=normalize_symbol_input(payload.symbol),
        prediction=payload.prediction,
        correct_label=payload.correct_label,
        confidence=payload.confidence,
        deployment=payload.deployment,
        model_uri=payload.model_uri,
    )
    return {"status": "feedback saved"}


@app.post("/api/predictions/verify/{prediction_id}")
async def verify_prediction(prediction_id: str):
    """실제 다음 거래일 종가로 예측 결과를 자동 검증"""
    return _verify_prediction_result(prediction_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
