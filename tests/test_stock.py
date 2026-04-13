# tests/test_stock.py

def calculate_price_change(current, previous):
    """가격 변화율 계산"""
    return ((current - previous) / previous) * 100


def test_price_change():
    """가격 변화율이 올바르게 계산되는지 테스트"""
    result = calculate_price_change(120, 100)
    assert result == 20.0
