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


def find_local_highs(series: pd.Series, order: int = 3) -> list:
    """find_local_lows'un aynasi: her iki tarafinda 'order' bar daha dusuk olan noktalar."""
    highs = []
    for i in range(order, len(series) - order):
        window = series.iloc[i - order: i + order + 1]
        if series.iloc[i] == window.max():
            highs.append(i)
    return highs


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
    # Motor 1 (A-B Fibonacci+ABC) ve Elliott sanity check'ten gelen ek puan.
    # SADECE bilgi amacli / raporlama icindir; gate'i VEYA confidence
    # kademesini ETKILEMEZ.
    bonus_score: float = 0.0
    bonus_notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1) Double Bottom yapisi (Murphy)
#    2026-07-10: min_bar_gap 10->8, level_tolerance %2.5->%5 olarak gevsetildi.
#    Gerekce: ilk canli tarama (56 coin, 0 sinyal) sonuclarinin analizi -
#    reddedilenlerin buyuk kismi 8-9 bar / %3-6 araliginda kumelenmisti (sinirda),
#    yani gate degil signal_validation.py'nin kendi filtreleri (structural
#    tolerans) darbogazdi. RSI/Fib bandı zaten sonraki elemeyi yapiyor.
# ---------------------------------------------------------------------------

def check_double_bottom(df: pd.DataFrame, min_bar_gap: int = 8,
                         level_tolerance: float = 0.05,
                         volume_breakout_mult: float = 1.15,
                         breakout_window: int = 5) -> dict:
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

    avg_vol_20 = df["volume"].iloc[-21:-1].mean()

    # 2026-07-10: tek bara (sadece en son mum) bakmak yerine, son
    # `breakout_window` bar icinde kirilim + hacim spike'inin AYNI barda
    # cakismasi araniyor. Gerekce: canli tarama verisi kirilim ve hacim
    # artisinin cogunlukla ayni ana denk gelmedigini gosterdi (19/50 coin
    # bu yuzden eleniyordu) - bu bir esik sorunu degil, zamanlama penceresi
    # sorunuydu. En yakin (en guncel) eslesme tercih edilir.
    breakout_confirmed = False
    volume_ok = False
    breakout_bars_ago = None
    for offset in range(1, breakout_window + 1):
        bar = df.iloc[-offset]
        if bar["close"] <= peak_mid:
            continue
        vol_ok_this_bar = bar["volume"] > avg_vol_20 * volume_breakout_mult
        if not breakout_confirmed:
            breakout_confirmed = True
            volume_ok = vol_ok_this_bar
            breakout_bars_ago = offset
        if vol_ok_this_bar:
            volume_ok = True
            breakout_bars_ago = offset
            break

    passed = breakout_confirmed and volume_ok
    return {
        "passed": passed,
        "low1_idx": low1_idx, "low2_idx": low2_idx,
        "low1_price": low1_price, "low2_price": low2_price,
        "peak_mid": peak_mid,
        "breakout_confirmed": breakout_confirmed,
        "volume_ok": volume_ok,
        "breakout_bars_ago": breakout_bars_ago,
        "reason": None if passed else "Kirilim teyidi veya hacim kosulu saglanmadi",
    }


# ---------------------------------------------------------------------------
# 1b) Double Top yapisi (Murphy) - check_double_bottom'un SHORT aynasi
# ---------------------------------------------------------------------------

def check_double_top(df: pd.DataFrame, min_bar_gap: int = 8,
                      level_tolerance: float = 0.05,
                      volume_breakdown_mult: float = 1.15,
                      breakdown_window: int = 5) -> dict:
    highs = find_local_highs(df["high"], order=3)

    if len(highs) < 2:
        return {"passed": False, "reason": "Yeterli sayida tepe bulunamadi"}

    high2_idx, high1_idx = highs[-1], highs[-2]
    if high2_idx - high1_idx < min_bar_gap:
        return {"passed": False, "reason": f"Tepeler arasi mesafe yetersiz ({high2_idx - high1_idx} bar < {min_bar_gap})"}

    high1_price = df["high"].iloc[high1_idx]
    high2_price = df["high"].iloc[high2_idx]
    level_diff = abs(high2_price - high1_price) / high1_price
    if level_diff > level_tolerance:
        return {"passed": False, "reason": f"Tepe seviyeleri farkli (%{level_diff*100:.2f} > tolerans)"}

    mid_slice = df.iloc[high1_idx:high2_idx + 1]
    trough_mid = mid_slice["low"].min()

    avg_vol_20 = df["volume"].iloc[-21:-1].mean()

    # check_double_bottom'daki ayni mantik: son `breakdown_window` bar
    # icinde kirilim (trough_mid'in ALTINA kapanis) + hacim spike'i ayni
    # barda mi cakisiyor.
    breakdown_confirmed = False
    volume_ok = False
    breakdown_bars_ago = None
    for offset in range(1, breakdown_window + 1):
        bar = df.iloc[-offset]
        if bar["close"] >= trough_mid:
            continue
        vol_ok_this_bar = bar["volume"] > avg_vol_20 * volume_breakdown_mult
        if not breakdown_confirmed:
            breakdown_confirmed = True
            volume_ok = vol_ok_this_bar
            breakdown_bars_ago = offset
        if vol_ok_this_bar:
            volume_ok = True
            breakdown_bars_ago = offset
            break

    passed = breakdown_confirmed and volume_ok
    return {
        "passed": passed,
        "high1_idx": high1_idx, "high2_idx": high2_idx,
        "high1_price": high1_price, "high2_price": high2_price,
        "trough_mid": trough_mid,
        "breakdown_confirmed": breakdown_confirmed,
        "volume_ok": volume_ok,
        "breakdown_bars_ago": breakdown_bars_ago,
        "reason": None if passed else "Kirilim (asagi) teyidi veya hacim kosulu saglanmadi",
    }




# ---------------------------------------------------------------------------
# 2) RSI Positive Divergence (Nison)
# ---------------------------------------------------------------------------

def check_rsi_divergence(df: pd.DataFrame, low1_idx: int, low2_idx: int,
                          oversold_threshold: float = 35,
                          min_rsi_gap: float = 5,
                          min_bar_gap: int = 6) -> dict:
    rsi = compute_rsi(df["close"])
    rsi1, rsi2 = rsi.iloc[low1_idx], rsi.iloc[low2_idx]
    price1, price2 = df["low"].iloc[low1_idx], df["low"].iloc[low2_idx]

    price_condition = price2 <= price1
    rsi_condition = rsi2 > rsi1
    oversold_condition = rsi1 < oversold_threshold
    gap_condition = (rsi2 - rsi1) >= min_rsi_gap
    # Iki dip birbirine cok yakinsa (ayni gurultu hareketinin parcasi olabilir)
    # diverjans yapisal degil, tesadufi sayilir. 4H barda 6 bar =~ 24 saat.
    bar_distance_condition = (low2_idx - low1_idx) >= min_bar_gap

    passed = (price_condition and rsi_condition and oversold_condition
              and gap_condition and bar_distance_condition)
    reason = None
    if not passed:
        if not bar_distance_condition:
            reason = f"Diverjans bar mesafesi yetersiz ({low2_idx - low1_idx} bar < {min_bar_gap})"
        else:
            reason = "RSI diverjans kosullari saglanmadi"
    return {
        "passed": passed,
        "rsi1": rsi1, "rsi2": rsi2,
        "bar_gap": low2_idx - low1_idx,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# 2b) RSI Negative Divergence (Nison) - check_rsi_divergence'in SHORT aynasi
# ---------------------------------------------------------------------------

def check_rsi_negative_divergence(df: pd.DataFrame, high1_idx: int, high2_idx: int,
                                   overbought_threshold: float = 65,
                                   min_rsi_gap: float = 5,
                                   min_bar_gap: int = 6) -> dict:
    rsi = compute_rsi(df["close"])
    rsi1, rsi2 = rsi.iloc[high1_idx], rsi.iloc[high2_idx]
    price1, price2 = df["high"].iloc[high1_idx], df["high"].iloc[high2_idx]

    price_condition = price2 >= price1          # fiyat esit/daha yuksek tepe yapiyor
    rsi_condition = rsi2 < rsi1                  # RSI daha DUSUK tepe yapiyor (negatif diverjans)
    overbought_condition = rsi1 > overbought_threshold
    gap_condition = (rsi1 - rsi2) >= min_rsi_gap
    bar_distance_condition = (high2_idx - high1_idx) >= min_bar_gap

    passed = (price_condition and rsi_condition and overbought_condition
              and gap_condition and bar_distance_condition)
    reason = None
    if not passed:
        if not bar_distance_condition:
            reason = f"Diverjans bar mesafesi yetersiz ({high2_idx - high1_idx} bar < {min_bar_gap})"
        else:
            reason = "RSI negatif diverjans kosullari saglanmadi"
    return {
        "passed": passed,
        "rsi1": rsi1, "rsi2": rsi2,
        "bar_gap": high2_idx - high1_idx,
        "reason": reason,
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
# 3b) Fibonacci bandina MESAFE ve YON (izleme listesi icin - gate degil)
# ---------------------------------------------------------------------------

def check_fibonacci_proximity(df: pd.DataFrame, swing_high: float, swing_low: float,
                               lookback: int = 5, swing_high_time=None,
                               swing_low_time=None) -> dict:
    """check_fibonacci_zone'un 'izleme' versiyonu: fiyat bandin DISINDAYSA
    ne kadar uzakta oldugunu (%) ve son `lookback` barda banda dogru mu
    yoksa uzaga mi hareket ettigini raporlar. Gate degildir, sinyali
    reddetmez/onaylamaz - sadece 'yaklasiyor' bilgisini tasir.

    swing_high_time/swing_low_time: bu seviyelerin HANGI mumdan geldigini
    izlemek icin (grafikte dogrulama yapabilmek adina) - hesaba katilmaz,
    sadece raporlanir."""
    current_price = df["close"].iloc[-1]
    fib618 = swing_high - 0.618 * (swing_high - swing_low)
    fib786 = swing_high - 0.786 * (swing_high - swing_low)
    zone_low, zone_high = min(fib618, fib786), max(fib618, fib786)

    base = {
        "swing_high": round(float(swing_high), 6),
        "swing_high_time": swing_high_time,
        "swing_low": round(float(swing_low), 6),
        "swing_low_time": swing_low_time,
        "zone": (round(float(zone_low), 6), round(float(zone_high), 6)),
        "current_price": round(float(current_price), 6),
    }

    if zone_low <= current_price <= zone_high:
        return {**base, "in_zone": True, "distance_pct": 0.0, "direction": "bandin icinde"}

    if current_price > zone_high:
        distance_pct = (current_price - zone_high) / zone_high * 100
    else:
        distance_pct = (zone_low - current_price) / zone_low * 100

    past_idx = -lookback - 1 if len(df) > lookback else 0
    past_price = df["close"].iloc[past_idx]

    if current_price > zone_high:
        approaching = current_price < past_price   # ustten iniyorsa banda yaklasiyor
    else:
        approaching = current_price > past_price   # alttan yukseliyorsa banda yaklasiyor

    return {
        **base,
        "in_zone": False,
        "distance_pct": round(float(distance_pct), 2),
        "direction": "yaklasiyor" if approaching else "uzaklasiyor",
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


def check_wyckoff_volume_short(df: pd.DataFrame, high1_idx: int, high2_idx: int,
                                max_ratio: float = 1.1) -> dict:
    """check_wyckoff_volume'un short/Upthrust aynasi: ikinci tepede hacim
    ilkine gore dusukse, talebin tukendigini (upthrust) isaret eder."""
    vol1 = df["volume"].iloc[high1_idx]
    vol2 = df["volume"].iloc[high2_idx]
    passed = vol2 <= vol1 * max_ratio
    return {
        "passed": passed,
        "vol1": vol1, "vol2": vol2,
        "reason": None if passed else "Ikinci tepede hacim beklenenden yuksek (talep tukenmesi/upthrust teyit edilemedi)",
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
                           risk_pct: float = 0.015, atr_mult: float = 3.0,
                           direction: str = "long") -> dict:
    """direction='long': stop entry'nin ALTINA (ATR bazli) konur.
    direction='short': stop entry'nin ÜSTÜNE konur. Pozisyon buyuklugu
    hesabi (risk_pct * equity / stop_distance) her iki yonde de aynidir -
    ATR bazli stop mesafesi degismez, sadece yonu degisir."""
    atr = compute_atr(df).iloc[-1]
    stop_distance = atr_mult * atr
    if direction == "short":
        stop_price = entry_price + stop_distance
    else:
        stop_price = entry_price - stop_distance
    position_size = (equity * risk_pct) / stop_distance
    return {
        "atr": atr,
        "stop_distance": stop_distance,
        "stop_price": stop_price,
        "position_size": position_size,
        "direction": direction,
    }


# ---------------------------------------------------------------------------
# Ana degerlendirme fonksiyonu
# ---------------------------------------------------------------------------

def evaluate_signal_short(df: pd.DataFrame, swing_high: float = None, swing_low: float = None,
                           equity: float = 1000, risk_pct: float = 0.015) -> SignalResult:
    """
    evaluate_signal()'in SHORT aynasi. GATE:
      1) Double Top yapisi (kirilim asagi + hacim)
      2) RSI negatif diverjans >=5 puan VE iki tepe arasi >=6 bar
      3) Fiyat Fibonacci 0.618-0.786 bandinda (check_fibonacci_zone yon
         bagimsiz - sadece bir bant kontrolu, aynen yeniden kullanilir)

    NOT: evaluate_confluence_entry() bu fonksiyonu SADECE `.is_valid`
    icin cagirir (short confidence icin "tam teyitli" tier'ini belirlemek
    amacli) - bu yuzden Wyckoff/Motor3/bonus katmanlarini (evaluate_signal
    ile ayni ayrintida) tekrar etmiyor, sadece 3 temel gate kontrol edilir.
    """
    result = SignalResult()

    if swing_high is None:
        swing_high = df["high"].iloc[-60:].max()
    if swing_low is None:
        swing_low = df["low"].iloc[-60:].min()

    dt = check_double_top(df)
    result.checks["double_top"] = dt
    if not dt["passed"]:
        result.reasons.append(f"Double Top: {dt['reason']}")
        return result

    rsi_check = check_rsi_negative_divergence(df, dt["high1_idx"], dt["high2_idx"])
    result.checks["rsi_negative_divergence"] = rsi_check
    if not rsi_check["passed"]:
        result.reasons.append(f"RSI Negative Divergence: {rsi_check['reason']}")
        return result

    fib_check = check_fibonacci_zone(df, swing_high, swing_low)
    result.checks["fibonacci_zone"] = fib_check
    result.fib_zone = fib_check["zone"]
    if not fib_check["passed"]:
        result.reasons.append(f"Fibonacci Zone: {fib_check['reason']}")
        return result

    entry_price = df["close"].iloc[-1]
    risk_calc = compute_stop_and_size(df, entry_price, equity, risk_pct, direction="short")
    result.checks["risk"] = risk_calc
    result.stop_price = risk_calc["stop_price"]
    result.position_size = risk_calc["position_size"]

    result.is_valid = True
    result.confidence = "low"  # evaluate_confluence_entry kendi confidence'ini ayrica hesaplar
    return result


def evaluate_signal(df: pd.DataFrame, swing_high: float = None, swing_low: float = None,
                     major_swing_high: float = None, major_swing_low: float = None,
                     equity: float = 1000, risk_pct: float = 0.015,
                     fib_confluence_result: dict = None,
                     ab_fibonacci_result: dict = None) -> SignalResult:
    """
    ══════════════════════════════════════════════════════════════
    GATE / CONFIDENCE / RISK HIYERARSISI
    (Boroden confluence + Nison/Wyckoff teyit + Van Tharp risk mantigina
    dayanarak 3 bagimsiz motorun tek stratejide birlestirilmis hali.)

    GATE (uçü de zorunlu, biri bile gecmezse sinyal reddedilir):
      1) Double Bottom yapisi (kirilim + hacim)
      2) RSI pozitif diverjans >=5 puan VE iki dip arasi >=6 bar
      3) Fiyat Fibonacci 0.618-0.786 bandinda

    CONFIDENCE (gate gectikten sonra kademelendirir, gate'i etkilemez):
      low    = sadece gate
      medium = gate + (Wyckoff hacim TEYIDI  VEYA  Motor 3 confluence)
      high   = gate + Wyckoff hacim TEYIDI  VE  Motor 3 confluence

    BONUS (yalnizca bilgi/raporlama, ASLA gate veya confidence kademesini
    degistirmez):
      + Elliott sanity check (buyuk trend retracement uzerinde mi)
      + Motor 1 (A-B Fibonacci + ABC uzantisi) sinyali varsa

    RISK:
      Stop-loss   = HER ZAMAN ATR bazli (confidence'tan bagimsiz)
      Pozisyon    = confidence kademesine gore: low=0.5R, medium=1.0R, high=1.5R

    fib_confluence_result: okx_double_bottom_divergence_v1.find_fib_confluence_signal
        ciktisi (Motor 3). None ise confluence yok sayilir.
    ab_fibonacci_result: okx_double_bottom_divergence_v1.find_ab_fibonacci_signal
        ciktisi (Motor 1). None ise bonusa katkisi olmaz.

    swing_high/low verilmezse son 60 bar icindeki max/min kullanilir (kaba
    yaklasim - cagiran taraf find_ab_fibonacci_signal'dan gelen gercek A-B
    swing'ini gecmeyi tercih etmeli).
    ══════════════════════════════════════════════════════════════
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

    # ── GATE 1: Double Bottom ────────────────────────────────────────
    db = check_double_bottom(df)
    result.checks["double_bottom"] = db
    if not db["passed"]:
        result.reasons.append(f"Double Bottom: {db['reason']}")
        return result  # temel yapi yoksa devam etmenin anlami yok

    # ── GATE 2: RSI Positive Divergence (>=5 puan, >=6 bar mesafe) ───
    rsi_check = check_rsi_divergence(df, db["low1_idx"], db["low2_idx"])
    result.checks["rsi_divergence"] = rsi_check
    if not rsi_check["passed"]:
        result.reasons.append(f"RSI Divergence: {rsi_check['reason']}")
        return result

    # ── GATE 3: Fibonacci 0.618-0.786 Zone ───────────────────────────
    fib_check = check_fibonacci_zone(df, swing_high, swing_low)
    result.checks["fibonacci_zone"] = fib_check
    result.fib_zone = fib_check["zone"]
    if not fib_check["passed"]:
        result.reasons.append(f"Fibonacci Zone: {fib_check['reason']}")
        return result

    # ── Gate gecildi. Simdi confidence kademesini belirleyen 2 katman ──
    wyckoff_check = check_wyckoff_volume(df, db["low1_idx"], db["low2_idx"])
    result.checks["wyckoff_volume"] = wyckoff_check
    if not wyckoff_check["passed"]:
        result.reasons.append(f"[Confidence] Wyckoff hacim teyidi yok: {wyckoff_check['reason']}")

    motor3_confluence = fib_confluence_result is not None
    result.checks["motor3_fib_confluence"] = fib_confluence_result
    if not motor3_confluence:
        result.reasons.append("[Confidence] Motor 3 confluence yok (bagimsiz iki swing cakismiyor)")

    if wyckoff_check["passed"] and motor3_confluence:
        confidence = "high"
    elif wyckoff_check["passed"] or motor3_confluence:
        confidence = "medium"
    else:
        confidence = "low"

    # ── BONUS (gate/confidence'i etkilemez, sadece raporlanir) ───────
    elliott_check = check_elliott_sanity(df, major_swing_high, major_swing_low)
    result.checks["elliott_sanity"] = elliott_check
    if not elliott_check["below_major_fib618"]:
        result.bonus_score += 0.5
        result.bonus_notes.append("Elliott: buyuk trend retracement seviyesinin uzerinde")
    else:
        result.bonus_notes.append(f"Elliott: {elliott_check['note']}")

    if ab_fibonacci_result is not None:
        result.bonus_score += 0.5
        result.bonus_notes.append("Motor 1: A-B Fibonacci + ABC uzantisi sinyali de mevcut")

    # ── RISK: stop ATR bazli (sabit), pozisyon confidence'a gore ─────
    entry_price = df["close"].iloc[-1]
    risk_calc = compute_stop_and_size(df, entry_price, equity, risk_pct)
    result.checks["risk"] = risk_calc
    result.stop_price = risk_calc["stop_price"]
    result.position_size = risk_calc["position_size"]

    position_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5}[confidence]
    result.position_size *= position_multiplier

    result.is_valid = True
    result.confidence = confidence
    return result


# ---------------------------------------------------------------------------
# IZLEME LISTESI (watchlist) - gate degil, erken uyaridir
# ---------------------------------------------------------------------------

def evaluate_watchlist(df: pd.DataFrame, swing_high: float = None, swing_low: float = None,
                        swing_high_time=None, swing_low_time=None,
                        max_fib_distance_pct: float = 8.0) -> dict:
    """
    evaluate_signal() SIKI bir gate'tir: uc kosul da SU AN saglanmadikca
    None doner. Bu fonksiyon ise "yapisal olarak hazir ama henuz tetiklenmemis"
    coinleri yakalar - kullanicinin "1-2 gun surebilir ama oraya gelirse
    islem yaparim" mantigina karsilik gelir.

    swing_high/swing_low CAGIRAN TARAFTAN gelmezse (yani Motor 1 bir A-B
    swing'i bulamadiysa), varsayilan olarak son 60 barin (4H'de ~10 gun)
    en yuksek/en dusuk noktasi kullanilir - VE bu noktalarin zaman
    damgalari da otomatik hesaplanir (grafikte dogrulama yapabilmen icin).

    Donus: None (izlemeye deger degil - dip yapisi bile yok) VEYA bir dict:
      {
        "structure_ready": True,          # iki dip + mesafe + tolerans gecti
        "breakout_confirmed": bool,
        "volume_ok": bool,
        "rsi_divergence_ok": bool,
        "in_fib_zone": bool,
        "fib_distance_pct": float | None, # bandin disindaysa % mesafe
        "fib_direction": str | None,      # "yaklasiyor" | "uzaklasiyor"
        "swing_high": float, "swing_high_time": str,  # bandin dayandigi ust nokta
        "swing_low": float, "swing_low_time": str,    # bandin dayandigi alt nokta
        "missing_steps": [str, ...],      # hangi adimlar eksik (siradaki oncelik)
      }

    NOT: Bu fonksiyonun ciktisi ISLEM TETIKLEMEZ. Sadece "izle" sinyalidir.
    Gercek giris karari HER ZAMAN evaluate_signal()'in gate'inden gecer.
    """
    if swing_high is None:
        window = df.iloc[-60:]
        swing_high = window["high"].max()
        swing_high_time = window["high"].idxmax().isoformat()
    if swing_low is None:
        window = df.iloc[-60:]
        swing_low = window["low"].min()
        swing_low_time = window["low"].idxmin().isoformat()

    db = check_double_bottom(df)
    if "low1_idx" not in db:
        return None  # dip yapisi (2 dip + mesafe + tolerans) bile yok, izlemeye deger degil

    missing_steps = []

    breakout_confirmed = db.get("breakout_confirmed", False)
    volume_ok = db.get("volume_ok", False)
    if not breakout_confirmed:
        missing_steps.append("kirilim_bekleniyor")
    elif not volume_ok:
        missing_steps.append("hacim_teyidi_bekleniyor")

    rsi_check = check_rsi_divergence(df, db["low1_idx"], db["low2_idx"])
    if not rsi_check["passed"]:
        missing_steps.append("rsi_diverjans_bekleniyor")

    fib_proximity = check_fibonacci_proximity(
        df, swing_high, swing_low,
        swing_high_time=swing_high_time, swing_low_time=swing_low_time,
    )
    if not fib_proximity["in_zone"]:
        missing_steps.append("fib_bandi_bekleniyor")
        # cok uzaksa (banda yaklasma ihtimali dusuk gorunuyorsa) izleme listesine alma
        if fib_proximity["distance_pct"] > max_fib_distance_pct:
            return None

    return {
        "structure_ready": True,
        "low1_price": round(float(db["low1_price"]), 6),
        "low1_time": df.index[db["low1_idx"]].isoformat(),
        "low2_price": round(float(db["low2_price"]), 6),
        "low2_time": df.index[db["low2_idx"]].isoformat(),
        "peak_mid": round(float(db["peak_mid"]), 6),
        "breakout_confirmed": bool(breakout_confirmed),
        "volume_ok": bool(volume_ok),
        "rsi_divergence_ok": bool(rsi_check["passed"]),
        "in_fib_zone": bool(fib_proximity["in_zone"]),
        "fib_zone_low": fib_proximity.get("zone", (None, None))[0],
        "fib_zone_high": fib_proximity.get("zone", (None, None))[1],
        "fib_distance_pct": fib_proximity.get("distance_pct"),
        "fib_direction": fib_proximity.get("direction"),
        "swing_high": fib_proximity.get("swing_high"),
        "swing_high_time": fib_proximity.get("swing_high_time"),
        "swing_low": fib_proximity.get("swing_low"),
        "swing_low_time": fib_proximity.get("swing_low_time"),
        "missing_steps": missing_steps,
    }


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
