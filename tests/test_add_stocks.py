import pytest
import json
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app, load_stocks, save_stocks, is_valid_symbol, calculate_price_change, MAX_STOCKS, STOCKS_FILE, _stocks_cache

# FastAPI TestClient 생성
client = TestClient(app)


# ============ Fixture (테스트 데이터 준비) ============

@pytest.fixture
def reset_stocks():
    """각 테스트 전후로 stocks.json 초기화"""
    original_file = None
    _stocks_cache["data"] = None
    _stocks_cache["expires_at"] = 0.0

    # 원본 파일 백업
    if os.path.exists(STOCKS_FILE):
        with open(STOCKS_FILE, "r") as f:
            original_file = json.load(f)

    yield  # 테스트 실행

    # 테스트 후 원본 복구
    if original_file:
        with open(STOCKS_FILE, "w") as f:
            json.dump(original_file, f)
    elif os.path.exists(STOCKS_FILE):
        os.remove(STOCKS_FILE)

    _stocks_cache["data"] = None
    _stocks_cache["expires_at"] = 0.0


# ============ 1. calculate_price_change 함수 테스트 ============

class TestCalculatePriceChange:
    """가격 변화율 계산 함수 테스트"""

    def test_positive_change(self):
        """양수 변화 테스트"""
        result = calculate_price_change(110, 100)
        assert result == 10.0, "10% 상승해야 함"

    def test_negative_change(self):
        """음수 변화 테스트"""
        result = calculate_price_change(90, 100)
        assert result == -10.0, "10% 하락해야 함"

    def test_no_change(self):
        """변화 없음 테스트"""
        result = calculate_price_change(100, 100)
        assert result == 0.0, "0% 변화해야 함"

    def test_zero_previous_price(self):
        """이전 가격이 0일 때"""
        result = calculate_price_change(100, 0)
        assert result == 0.0, "0으로 나누지 않아야 함"

    def test_large_change(self):
        """큰 변화 테스트"""
        result = calculate_price_change(200, 100)
        assert result == 100.0, "100% 상승해야 함"


# ============ 2. load_stocks & save_stocks 테스트 ============

class TestStocksFileOperations:
    """종목 파일 로드/저장 테스트"""

    def test_save_and_load(self, reset_stocks):
        """저장 후 로드 테스트"""
        test_stocks = ["AAPL", "GOOGL", "MSFT"]
        save_stocks(test_stocks)
        loaded = load_stocks()
        assert loaded == test_stocks, "저장된 종목과 로드된 종목이 같아야 함"

    def test_load_default_stocks(self, reset_stocks):
        """파일 없을 때 기본값 로드"""
        if os.path.exists(STOCKS_FILE):
            os.remove(STOCKS_FILE)

        loaded = load_stocks()
        assert len(loaded) > 0, "기본 종목이 반환되어야 함"
        assert "AAPL" in loaded, "AAPL이 포함되어야 함"

    def test_save_empty_list(self, reset_stocks):
        """빈 리스트 저장"""
        save_stocks([])
        loaded = load_stocks()
        assert loaded == [], "빈 리스트가 저장되어야 함"

    def test_save_with_special_characters(self, reset_stocks):
        """특수 문자 포함 저장"""
        test_stocks = ["BRK.B", "BRK.A", "TSM"]
        save_stocks(test_stocks)
        loaded = load_stocks()
        assert loaded == test_stocks, "특수 문자가 포함된 종목도 저장되어야 함"


# ============ 3. is_valid_symbol 함수 테스트 (Mock 사용) ============

class TestIsValidSymbol:
    """종목 코드 유효성 확인 테스트"""

    @patch("main.yf.Ticker")
    def test_valid_symbol(self, mock_ticker):
        """유효한 종목 테스트"""
        mock_instance = MagicMock()
        mock_instance.history.return_value = MagicMock(
            __len__=MagicMock(return_value=1)
        )
        mock_ticker.return_value = mock_instance

        result = is_valid_symbol("AAPL")
        assert result is True, "유효한 종목이어야 함"

    @patch("main.yf.Ticker")
    def test_invalid_symbol(self, mock_ticker):
        """유효하지 않은 종목 테스트"""
        mock_instance = MagicMock()
        mock_instance.history.return_value = MagicMock(
            __len__=MagicMock(return_value=0)
        )
        mock_ticker.return_value = mock_instance

        result = is_valid_symbol("INVALID")
        assert result is False, "유효하지 않은 종목이어야 함"

    @patch("main.yf.Ticker")
    def test_exception_handling(self, mock_ticker):
        """예외 처리 테스트"""
        mock_ticker.side_effect = Exception("API Error")

        result = is_valid_symbol("ERROR")
        assert result is False, "예외 발생 시 False 반환해야 함"


# ============ 4. API: GET /api/stocks/list 테스트 ============

class TestGetStockList:
    """GET /api/stocks/list 엔드포인트 테스트"""

    def test_get_stock_list_status(self, reset_stocks):
        """상태 코드 확인"""
        response = client.get("/api/stocks/list")
        assert response.status_code == 200, "200 OK 반환해야 함"

    def test_get_stock_list_structure(self, reset_stocks):
        """응답 구조 확인"""
        response = client.get("/api/stocks/list")
        data = response.json()

        assert "symbols" in data, "symbols 키가 있어야 함"
        assert "count" in data, "count 키가 있어야 함"
        assert "max_stocks" in data, "max_stocks 키가 있어야 함"

    def test_get_stock_list_content(self, reset_stocks):
        """응답 내용 확인"""
        response = client.get("/api/stocks/list")
        data = response.json()

        assert isinstance(data["symbols"], list), "symbols는 리스트여야 함"
        assert data["count"] == len(data["symbols"]), "count가 정확해야 함"
        assert data["max_stocks"] == MAX_STOCKS, "max_stocks이 일치해야 함"


# ============ 4-1. API: GET /api/stocks 카드 데이터 테스트 ============

class TestGetStocks:
    """GET /api/stocks 엔드포인트 테스트"""

    @patch("main.load_stocks")
    @patch("main._download_stock_histories")
    def test_stock_card_remains_when_price_history_is_short(
        self,
        mock_download_histories,
        mock_load_stocks,
    ):
        """가격 데이터가 부족해도 종목 카드 데이터는 유지"""
        mock_load_stocks.return_value = ["AAPL"]
        mock_download_histories.return_value = {"AAPL": []}

        response = client.get("/api/stocks")
        data = response.json()

        assert response.status_code == 200, "200 OK 반환해야 함"
        assert data["count"] == 1, "데이터 부족 종목도 응답에 남아야 함"
        assert data["stocks"][0]["symbol"] == "AAPL", "AAPL 카드가 유지되어야 함"
        assert data["stocks"][0]["current_price"] is None, "가격은 None이어야 함"
        assert "error" in data["stocks"][0], "실패 이유가 있어야 함"


# ============ 5. API: POST /api/stocks/add 테스트 ============

class TestAddStock:
    """POST /api/stocks/add 엔드포인트 테스트"""

    @patch("main.is_valid_symbol")
    def test_add_stock_success(self, mock_valid, reset_stocks):
        """종목 추가 성공"""
        mock_valid.return_value = True

        response = client.post("/api/stocks/add", json={"symbol": "NVDA"})
        assert response.status_code == 200, "200 OK 반환해야 함"

        data = response.json()
        assert data["symbol"] == "NVDA", "반환된 symbol이 맞아야 함"
        assert "message" in data, "message가 있어야 함"

    @patch("main.is_valid_symbol")
    def test_add_duplicate_stock(self, mock_valid, reset_stocks):
        """중복 종목 추가 실패"""
        mock_valid.return_value = True

        # 첫 번째 추가
        client.post("/api/stocks/add", json={"symbol": "AAPL"})

        # 두 번째 추가 (중복)
        response = client.post("/api/stocks/add", json={"symbol": "AAPL"})
        assert response.status_code == 400, "400 Bad Request 반환해야 함"
        assert "이미 추가" in response.json()["detail"], "중복 메시지가 있어야 함"

    @patch("main.is_valid_symbol")
    def test_add_invalid_symbol(self, mock_valid, reset_stocks):
        """유효하지 않은 종목 추가"""
        mock_valid.return_value = False

        response = client.post("/api/stocks/add", json={"symbol": "FAKE"})
        assert response.status_code == 400, "400 Bad Request 반환해야 함"
        assert "찾을 수 없습니다" in response.json()["detail"], "에러 메시지가 있어야 함"

    @patch("main.is_valid_symbol")
    def test_add_empty_symbol(self, mock_valid, reset_stocks):
        """빈 종목 코드"""
        mock_valid.return_value = True

        response = client.post("/api/stocks/add", json={"symbol": ""})
        assert response.status_code == 400, "400 Bad Request 반환해야 함"

    @patch("main.is_valid_symbol")
    def test_add_long_symbol(self, mock_valid, reset_stocks):
        """너무 긴 종목 코드 (5자 초과)"""
        mock_valid.return_value = True

        response = client.post("/api/stocks/add", json={"symbol": "TOOLONG"})
        assert response.status_code == 400, "400 Bad Request 반환해야 함"
        assert "1-5자" in response.json()["detail"], "길이 제한 메시지가 있어야 함"

    @patch("main.is_valid_symbol")
    def test_add_lowercase_symbol(self, mock_valid, reset_stocks):
        """소문자 종목 코드 (자동 변환)"""
        mock_valid.return_value = True

        response = client.post("/api/stocks/add", json={"symbol": "nvda"})
        assert response.status_code == 200, "200 OK 반환해야 함"
        assert response.json()["symbol"] == "NVDA", "대문자로 변환되어야 함"

    @patch("main.is_valid_symbol")
    def test_add_korean_company_name(self, mock_valid, reset_stocks):
        """한글 회사명으로 종목 추가"""
        mock_valid.return_value = True
        save_stocks(["AAPL", "GOOGL", "MSFT"])

        response = client.post("/api/stocks/add", json={"symbol": "테슬라"})
        assert response.status_code == 200, "200 OK 반환해야 함"
        assert response.json()["symbol"] == "TSLA", "테슬라는 TSLA로 변환되어야 함"

    @patch("main.is_valid_symbol")
    def test_add_korean_company_name_with_space(self, mock_valid, reset_stocks):
        """띄어쓰기 포함 한글 회사명 처리"""
        mock_valid.return_value = True
        save_stocks(["AAPL", "GOOGL"])

        response = client.post("/api/stocks/add", json={"symbol": "마이크로 소프트"})
        assert response.status_code == 200, "200 OK 반환해야 함"
        assert response.json()["symbol"] == "MSFT", "마이크로 소프트는 MSFT로 변환되어야 함"

    @patch("main.is_valid_symbol")
    def test_add_korean_company_name_duplicate(self, mock_valid, reset_stocks):
        """기본 종목과 매핑되는 한글 회사명 중복 처리"""
        mock_valid.return_value = True

        response = client.post("/api/stocks/add", json={"symbol": "애플"})
        assert response.status_code == 400, "기본 보유 종목은 중복으로 거부되어야 함"
        assert "이미 추가" in response.json()["detail"], "중복 메시지가 있어야 함"

    @patch("main.is_valid_symbol")
    def test_add_max_stocks_exceeded(self, mock_valid, reset_stocks):
        """최대 종목 개수 초과"""
        mock_valid.return_value = True

        # 기본 종목을 포함해 MAX_STOCKS 개까지 채우기
        current_count = len(load_stocks())
        for i in range(MAX_STOCKS - current_count):
            symbol = f"T{i:04d}"
            client.post("/api/stocks/add", json={"symbol": symbol})

        # 초과 추가
        response = client.post("/api/stocks/add", json={"symbol": "EXTRA"})
        assert response.status_code == 400, "400 Bad Request 반환해야 함"
        assert "최대" in response.json()["detail"], "최대값 메시지가 있어야 함"


# ============ 6. API: DELETE /api/stocks/remove/{symbol} 테스트 ============

class TestRemoveStock:
    """DELETE /api/stocks/remove/{symbol} 엔드포인트 테스트"""

    @patch("main.is_valid_symbol")
    def test_remove_stock_success(self, mock_valid, reset_stocks):
        """종목 삭제 성공"""
        mock_valid.return_value = True

        # 먼저 추가
        client.post("/api/stocks/add", json={"symbol": "TSLA"})

        # 삭제
        response = client.delete("/api/stocks/remove/TSLA")
        assert response.status_code == 200, "200 OK 반환해야 함"
        assert response.json()["symbol"] == "TSLA", "반환된 symbol이 맞아야 함"

    def test_remove_nonexistent_stock(self, reset_stocks):
        """존재하지 않는 종목 삭제"""
        response = client.delete("/api/stocks/remove/NONEXIST")
        assert response.status_code == 404, "404 Not Found 반환해야 함"
        assert "찾을 수 없습니다" in response.json()["detail"], "에러 메시지가 있어야 함"

    @patch("main.is_valid_symbol")
    def test_remove_lowercase_symbol(self, mock_valid, reset_stocks):
        """소문자 종목 코드 삭제 (자동 변환)"""
        mock_valid.return_value = True

        # 대문자로 추가
        client.post("/api/stocks/add", json={"symbol": "AMD"})

        # 소문자로 삭제
        response = client.delete("/api/stocks/remove/amd")
        assert response.status_code == 200, "200 OK 반환해야 함"

    @patch("main.is_valid_symbol")
    def test_remove_and_verify(self, mock_valid, reset_stocks):
        """삭제 후 리스트에서 제거되었는지 확인"""
        mock_valid.return_value = True

        # 추가
        client.post("/api/stocks/add", json={"symbol": "META"})

        # 삭제
        client.delete("/api/stocks/remove/META")

        # 리스트 확인
        response = client.get("/api/stocks/list")
        symbols = response.json()["symbols"]
        assert "META" not in symbols, "삭제된 종목이 리스트에 없어야 함"


# ============ 7. API: GET / (루트) 테스트 ============

class TestRootRoute:
    """GET / 엔드포인트 테스트"""

    def test_root_route_status(self):
        """루트 경로 상태 코드"""
        response = client.get("/")
        assert response.status_code == 200, "200 OK 반환해야 함"

    def test_root_route_returns_html(self):
        """루트 경로가 HTML 반환"""
        response = client.get("/")
        assert "text/html" in response.headers.get("content-type", ""), "HTML 콘텐츠 타입이어야 함"


# ============ 8. API: GET /api/health 테스트 ============

class TestHealthCheck:
    """GET /api/health 엔드포인트 테스트"""

    def test_health_check_status(self):
        """헬스 체크 상태 코드"""
        response = client.get("/api/health")
        assert response.status_code == 200, "200 OK 반환해야 함"

    def test_health_check_structure(self):
        """헬스 체크 응답 구조"""
        response = client.get("/api/health")
        data = response.json()

        assert "status" in data, "status 키가 있어야 함"
        assert "stocks_file_exists" in data, "stocks_file_exists 키가 있어야 함"
        assert data["status"] == "healthy", "상태가 healthy여야 함"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
