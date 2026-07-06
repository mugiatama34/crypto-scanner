"""
signal_validation.py

Elliott Wave Principle (Frost/Prechter), Japanese Candlestick Charting
Techniques (Nison), Fibonacci Trading (Boroden), Wyckoff yontemi ve
Van Tharp risk yonetimi kaynaklarindan cikarilan kurallarin kod karsiligi.

Mevcut OKX tabanli tarama script'ine entegre edilmek uzere tasarlandi.
Girdi: pandas DataFrame, kolonlar: ['open', 'high', 'low', 'close', 'volume']
       index: zaman damgasi (4H bar), artan sirada.

Kullanim:
    from signal_validation import evaluate_signal
    result = evaluate_signal(df, equity=1000, risk_pct=0.015)
    if result.is_valid:
        send_telegram_alert(result)
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Yardimci gostergeler
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def find_local_lows(series: pd.Series, order: int = 3) -> list:
    """Basit yerel dip bulucu: her iki tarafinda 'order' bar daha yuksek olan noktalar."""
    lows = []
    for i in range(order, len(series) - order):
        window = series.iloc[i - order: i + order + 1]
        if series.iloc[i] == window.min():
            lows.append(i)
    return lows


# ---------------------------------------------------------------------------
# Sonuc objesi
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    is_valid: bool = False
    confidence: str = "none"        # "low" | "medium" | "high"
    checks: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    stop_price: float = None
    position_size: float = None
    fib_zone: tuple = None


# ---------------------------------------------------------------------------
# 1) Double Bottom yapisi (Murphy)
# ---------------------------------------------------------------------------

def check_double_bottom(df: pd.DataFrame, min_bar_gap: int = 10,
                         level_tolerance: float = 0.025,
                         volume_breakout_mult: float = 1.3) -> dict:
    closes = df["close"]
    lows = find_local_lows(df["low"], order=3)

    if len(lows) < 2:
        return {"passed": False, "reason": "Yeterli sayida dip bulunamadi"}

    low2_idx, low1_idx = lows[-1], lows[-2]
    if low2_idx - low1_idx < min_bar_gap:
        return {"passed": False, "reason": f"Dipler arasi mesafe yetersiz ({low2_idx - low1_idx} bar < {min_bar_gap})"}

    low1_price = df["low"].iloc[low1_idx]
    low2_price = df["low"].iloc[low2_idx]
    level_diff = abs(low2_price - low1_price) / low1_price
    if level_diff > level_tolerance:
        return {"passed": False, "reason": f"Dip seviyeleri farkli (%{level_diff*100:.2f} > tolerans)"}

    mid_slice = df.iloc[low1_idx:low2_idx + 1]
    peak_mid = mid_slice["high"].max()

    last_close = closes.iloc[-1]
    breakout_confirmed = last_close > peak_mid  # kapanis bazli, fitil degil

    avg_vol_20 = df["volume"].iloc[-21:-1].mean()
    last_vol = df["volume"].iloc[-1]
    volume_ok = last_vol > avg_vol_20 * volume_breakout_mult

    passed = breakout_confirmed and volume_ok
    return {
        "passed": passed,
        "low1_idx": low1_idx, "low2_idx": low2_idx,
        "low1_price": low1_price, "low2_price": low2_price,
        "peak_mid": peak_mid,
        "breakout_confirmed": breakout_confirmed,
        "volume_ok": volume_ok,
        "reason": None if passed else "Kirilim teyidi veya hacim kosulu saglanmadi",
    }


# ---------------------------------------------------------------------------
# 2) RSI Positive Divergence (Nison)
# ---------------------------------------------------------------------------

def check_rsi_divergence(df: pd.DataFrame, low1_idx: int, low2_idx: int,
                          oversold_threshold: float = 35,
                          min_rsi_gap: float = 5) -> dict:
    rsi = compute_rsi(df["close"])
    rsi1, rsi2 = rsi.iloc[low1_idx], rsi.iloc[low2_idx]
    price1, price2 = df["low"].iloc[low1_idx], df["low"].iloc[low2_idx]

    price_condition = price2 <= price1
    rsi_condition = rsi2 > rsi1
    oversold_condition = rsi1 < oversold_threshold
    gap_condition = (rsi2 - rsi1) >= min_rsi_gap

    passed = price_condition and rsi_condition and oversold_condition and gap_condition
    return {
        "passed": passed,
        "rsi1": rsi1, "rsi2": rsi2,
        "reason": None if passed else "RSI diverjans kosullari saglanmadi",
    }


# ---------------------------------------------------------------------------
# 3) Fibonacci 0.618-0.786 Confluence (Boroden)
# ---------------------------------------------------------------------------

def check_fibonacci_zone(df: pd.DataFrame, swing_high: float, swing_low: float,
                          tolerance: float = 0.003) -> dict:
    current_price = df["close"].iloc[-1]
    fib618 = swing_high - 0.618 * (swing_high - swing_low)
    fib786 = swing_high - 0.786 * (swing_high - swing_low)

    zone_low = min(fib618, fib786) * (1 - tolerance)
    zone_high = max(fib618, fib786) * (1 + tolerance)

    in_zone = zone_low <= current_price <= zone_high
    return {
        "passed": in_zone,
        "fib618": fib618, "fib786": fib786,
        "zone": (zone_low, zone_high),
        "current_price": current_price,
        "reason": None if in_zone else "Fiyat 0.618-0.786 bandinin disinda",
    }


# ---------------------------------------------------------------------------
# 4) Wyckoff hacim teyidi (opsiyonel filtre)
# ---------------------------------------------------------------------------

def check_wyckoff_volume(df: pd.DataFrame, low1_idx: int, low2_idx: int,
                          max_ratio: float = 1.1) -> dict:
    vol1 = df["volume"].iloc[low1_idx]
    vol2 = df["volume"].iloc[low2_idx]
    passed = vol2 <= vol1 * max_ratio
    return {
        "passed": passed,
        "vol1": vol1, "vol2": vol2,
        "reason": None if passed else "Ikinci dipte hacim beklenenden yuksek (arz tukenmesi teyit edilemedi)",
    }


# ---------------------------------------------------------------------------
# 5) Elliott pozisyon filtresi (sanity check, tetikleyici degil)
# ---------------------------------------------------------------------------

def check_elliott_sanity(df: pd.DataFrame, major_swing_high: float,
                          major_swing_low: float) -> dict:
    current_price = df["close"].iloc[-1]
    fib618_major = major_swing_high - 0.618 * (major_swing_high - major_swing_low)

    below_major_fib = current_price < fib618_major
    return {
        "below_major_fib618": below_major_fib,
        "note": ("Muhtemel duzeltme (B/2 dalgasi), trend donusu henuz teyitli degil"
                 if below_major_fib else "Buyuk trend retracement seviyesinin uzerinde"),
    }


# ---------------------------------------------------------------------------
# 6) ATR bazli stop-loss ve pozisyon buyuklugu (Van Tharp / Wilder)
# ---------------------------------------------------------------------------

def compute_stop_and_size(df: pd.DataFrame, entry_price: float, equity: float,
                           risk_pct: float = 0.015, atr_mult: float = 3.0) -> dict:
    atr = compute_atr(df).iloc[-1]
    stop_distance = atr_mult * atr
    stop_price = entry_price - stop_distance
    position_size = (equity * risk_pct) / stop_distance
    return {
        "atr": atr,
        "stop_distance": stop_distance,
        "stop_price": stop_price,
        "position_size": position_size,
    }


# ---------------------------------------------------------------------------
# Ana degerlendirme fonksiyonu
# ---------------------------------------------------------------------------

def evaluate_signal(df: pd.DataFrame, swing_high: float = None, swing_low: float = None,
                     major_swing_high: float = None, major_swing_low: float = None,
                     equity: float = 1000, risk_pct: float = 0.015) -> SignalResult:
    """
    Tum kurallari sirayla calistirir. swing_high/low verilmezse son 60 bar
    icindeki max/min kullanilir (kaba yaklasim - kendi swing detection
    mantiginla degistirmen onerilir).
    """
    result = SignalResult()

    if swing_high is None:
        swing_high = df["high"].iloc[-60:].max()
    if swing_low is None:
        swing_low = df["low"].iloc[-60:].min()
    if major_swing_high is None:
        major_swing_high = df["high"].max()
    if major_swing_low is None:
        major_swing_low = df["low"].min()

    # 1) Double Bottom
    db = check_double_bottom(df)
    result.checks["double_bottom"] = db
    if not db["passed"]:
        result.reasons.append(f"Double Bottom: {db['reason']}")
        return result  # temel yapi yoksa devam etmenin anlami yok

    # 2) RSI Divergence
    rsi_check = check_rsi_divergence(df, db["low1_idx"], db["low2_idx"])
    result.checks["rsi_divergence"] = rsi_check
    if not rsi_check["passed"]:
        result.reasons.append(f"RSI Divergence: {rsi_check['reason']}")
        return result

    # 3) Fibonacci Zone
    fib_check = check_fibonacci_zone(df, swing_high, swing_low)
    result.checks["fibonacci_zone"] = fib_check
    result.fib_zone = fib_check["zone"]
    if not fib_check["passed"]:
        result.reasons.append(f"Fibonacci Zone: {fib_check['reason']}")
        return result

    # 4) Wyckoff hacim (opsiyonel - basarisiz olsa da sinyali reddetmez, notlanir)
    wyckoff_check = check_wyckoff_volume(df, db["low1_idx"], db["low2_idx"])
    result.checks["wyckoff_volume"] = wyckoff_check
    if not wyckoff_check["passed"]:
        result.reasons.append(f"[Bilgi] Wyckoff: {wyckoff_check['reason']}")

    # 5) Elliott sanity check -> confidence belirler, reddetmez
    elliott_check = check_elliott_sanity(df, major_swing_high, major_swing_low)
    result.checks["elliott_sanity"] = elliott_check

    confidence = "high"
    if elliott_check["below_major_fib618"]:
        confidence = "low"
        result.reasons.append(f"[Uyari] Elliott: {elliott_check['note']}")
    elif not wyckoff_check["passed"]:
        confidence = "medium"

    # 6) Stop-loss ve pozisyon buyuklugu
    entry_price = df["close"].iloc[-1]
    risk_calc = compute_stop_and_size(df, entry_price, equity, risk_pct)
    result.checks["risk"] = risk_calc
    result.stop_price = risk_calc["stop_price"]
    result.position_size = risk_calc["position_size"]

    # confidence dusukse pozisyonu kucult
    if confidence == "low":
        result.position_size *= 0.5
    elif confidence == "medium":
        result.position_size *= 0.75

    result.is_valid = True
    result.confidence = confidence
    return result


if __name__ == "__main__":
    # Hizli test icin sahte veri
    rng = pd.date_range("2026-01-01", periods=200, freq="4h")
    np.random.seed(42)
    price = 3000 + np.cumsum(np.random.randn(200) * 10)
    df_test = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(200) * 5,
        "low": price - np.random.rand(200) * 5,
        "close": price + np.random.randn(200),
        "volume": np.random.randint(100, 1000, 200),
    }, index=rng)

    res = evaluate_signal(df_test, equity=1000, risk_pct=0.015)
    print("is_valid:", res.is_valid)
    print("confidence:", res.confidence)
    print("reasons:", res.reasons)
