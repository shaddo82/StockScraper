"""종목 추가/삭제 관리 모듈"""
import json
import os
from typing import List, Tuple
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STOCKS_FILE = "stocks.json"
DEFAULT_STOCKS = ["AAPL", "GOOGL", "MSFT"]
MAX_STOCKS = 20


def load_stocks() -> List[str]:
    """저장된 종목 리스트 로드"""
    try:
        if os.path.exists(STOCKS_FILE):
            with open(STOCKS_FILE, "r") as f:
                stocks = json.load(f)
                logger.info(f"✅ {len(stocks)}개 종목 로드 완료")
                return stocks
        return DEFAULT_STOCKS.copy()
    except Exception as e:
        logger.error(f"❌ 종목 로드 실패: {e}")
        return DEFAULT_STOCKS.copy()


def save_stocks(stocks: List[str]) -> bool:
    """종목 리스트 저장"""
    try:
        with open(STOCKS_FILE, "w") as f:
            json.dump(stocks, f)
        logger.info(f"✅ {len(stocks)}개 종목 저장 완료")
        return True
    except Exception as e:
        logger.error(f"❌ 종목 저장 실패: {e}")
        return False


def is_valid_symbol(symbol: str) -> bool:
    """
    종목 코드 유효성 확인
    yfinance에서 데이터를 가져올 수 있는지 확인
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        is_valid = len(data) > 0

        if is_valid:
            logger.info(f"✅ '{symbol}' 유효한 종목코드")
        else:
            logger.warning(f"⚠️  '{symbol}' 데이터 없음")

        return is_valid
    except Exception as e:
        logger.error(f"❌ '{symbol}' 유효성 확인 실패: {e}")
        return False


def validate_symbol_format(symbol: str) -> Tuple[bool, str]:
    """
    종목 코드 형식 검증

    Returns:
        (유효성, 오류메시지)
    """
    symbol = symbol.strip()

    if not symbol:
        return False, "종목 코드는 공백일 수 없습니다"

    if len(symbol) > 5:
        return False, "종목 코드는 5자 이하여야 합니다"

    if not symbol.replace("-", "").isalnum():
        return False, "종목 코드는 문자와 숫자만 포함해야 합니다"

    return True, ""


def add_stock(symbol: str) -> Tuple[bool, str]:
    """
    종목 추가

    Args:
        symbol: 종목 코드

    Returns:
        (성공여부, 메시지)
    """
    symbol = symbol.upper().strip()

    # 형식 검증
    is_valid, error_msg = validate_symbol_format(symbol)
    if not is_valid:
        return False, error_msg

    # 유효성 확인
    if not is_valid_symbol(symbol):
        return False, f"'{symbol}' 종목을 찾을 수 없습니다"

    # 현재 종목 리스트 로드
    symbols = load_stocks()

    # 중복 확인
    if symbol in symbols:
        return False, f"'{symbol}' 종목은 이미 추가되어 있습니다"

    # 최대 개수 확인
    if len(symbols) >= MAX_STOCKS:
        return False, f"최대 {MAX_STOCKS}개까지만 추가할 수 있습니다"

    # 종목 추가 및 저장
    symbols.append(symbol)
    if save_stocks(symbols):
        return True, f"'{symbol}' 종목이 추가되었습니다 ({len(symbols)}/{MAX_STOCKS})"
    else:
        return False, "종목 저장 중 오류가 발생했습니다"


def remove_stock(symbol: str) -> Tuple[bool, str]:
    """
    종목 삭제

    Args:
        symbol: 종목 코드

    Returns:
        (성공여부, 메시지)
    """
    symbol = symbol.upper().strip()

    # 현재 종목 리스트 로드
    symbols = load_stocks()

    # 종목 존재 확인
    if symbol not in symbols:
        return False, f"'{symbol}' 종목을 찾을 수 없습니다"

    # 기본 종목 삭제 방지 (선택사항)
    if symbol in DEFAULT_STOCKS and len(symbols) == 1:
        return False, "최소 1개의 종목은 유지해야 합니다"

    # 종목 삭제 및 저장
    symbols.remove(symbol)
    if save_stocks(symbols):
        return True, f"'{symbol}' 종목이 삭제되었습니다 ({len(symbols)}/{MAX_STOCKS})"
    else:
        return False, "종목 삭제 중 오류가 발생했습니다"


def get_stock_count() -> int:
    """현재 종목 개수 반환"""
    return len(load_stocks())


def reset_to_default() -> Tuple[bool, str]:
    """기본 종목 리스트로 리셋"""
    if save_stocks(DEFAULT_STOCKS.copy()):
        return True, f"기본 종목으로 리셋되었습니다 ({len(DEFAULT_STOCKS)}개)"
    else:
        return False, "리셋 중 오류가 발생했습니다"
