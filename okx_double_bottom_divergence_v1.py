# ╔══════════════════════════════════════════════════════════════════════╗
# ║  OKX — BÜYÜK RALLİ SONRASI FİBO BÖLGESINDE                        ║
# ║        İKİLİ DİP + RSI POZİTİF UYUMSUZLUĞU TARAYICI              ║
# ║  4 Saatlik | Nisan 2026+ | v1.1                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# ARANAN YAPI:
#
#   [BÜYÜK RALLİ]  ≥%30 yükseliş
#        │
#        ▼
#   ──[ZİRVE]──────────────────────────────────────────────
#        │  Fibo 0.618–0.786 bölgesine düşüş
#        ▼
#   ──[DİP 1]──  RSI₁ kaydedilir
#        │  Toparlanma (en az %5)
#        ▼
#   ──[TEPE]────
#        │  Tekrar düşüş
#        ▼
#   ──[DİP 2]──  Fiyat ≤ Dip1  AMMA  RSI₂ > RSI₁
#        │
#        └─► POZİTİF UYUMSUZLUK ✅ + FİBO BÖLGE TEYIDI ✅
#
# Bağımlılıklar için requirements.txt dosyasına bakın.
# Çalıştırma:
#   python okx_double_bottom_divergence_v1.py
#   python okx_double_bottom_divergence_v1.py --test-symbol EUL/USDT --debug
#
# ── OKX API KİMLİK BİLGİLERİ (opsiyonel) ──────────────────────────────
# Bu script yalnızca herkese açık (public) OHLCV uç noktalarını kullanır,
# bu yüzden API anahtarı gerektirmez. Yine de kimlik bilgisi gereken bir
# uç nokta eklenirse aşağıdaki ortam değişkenleri otomatik okunur:
#   OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE
# Kod içine ASLA sabit (hardcoded) anahtar yazmayın.

import argparse
import ccxt
import json
import logging
import os
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── LOGLAMA ────────────────────────────────────────────────────────────
# Bulunan sinyaller konsola basılmaz; signals.log (insan-okunur) ve
# signals.json (makine-okunur) dosyalarına yazılır. GitHub Actions
# çalıştırmalarında bu dosyalar artifact olarak toplanabilir.
LOG_FILE     = os.environ.get("SIGNALS_LOG_FILE", "signals.log")
SIGNALS_JSON = os.environ.get("SIGNALS_JSON_FILE", "signals.json")

logger = logging.getLogger("okx_scanner")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()

_file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
_file_handler.setLevel(logging.INFO)
logger.addHandler(_file_handler)

# Konsola sadece durum/özet mesajları düşer (WARNING), tek tek bulunan
# sinyaller yalnızca dosyaya (INFO) yazılır.
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("%(message)s"))
_console_handler.setLevel(logging.WARNING)
logger.addHandler(_console_handler)


def log_status(msg):
    """Konsol + dosyaya yazılan durum/özet mesajı."""
    logger.warning(msg)


def log_signal(msg):
    """Sadece dosyaya (signals.log) yazılan sinyal detayı."""
    logger.info(msg)


# ── OKX BAĞLANTISI ──────────────────────────────────────────────────────
OKX_API_KEY        = os.environ.get("OKX_API_KEY", "")
OKX_API_SECRET     = os.environ.get("OKX_API_SECRET", "")
OKX_API_PASSPHRASE = os.environ.get("OKX_API_PASSPHRASE", "")


def build_exchange():
    """ccxt OKX exchange nesnesini oluşturur.

    Kimlik bilgileri kod içine yazılmaz; ortam değişkenlerinden okunur.
    Herkese açık OHLCV verisi için kimlik bilgisi gerekmez.
    """
    config = {"enableRateLimit": True}
    if OKX_API_KEY and OKX_API_SECRET and OKX_API_PASSPHRASE:
        config.update({
            "apiKey": OKX_API_KEY,
            "secret": OKX_API_SECRET,
            "password": OKX_API_PASSPHRASE,
        })
    return ccxt.okx(config)


# ── ZAMAN AYARLARI ────────────────────────────────────────────────────
START_DATE_MS    = int(datetime(2026, 4, 1, tzinfo=timezone.utc).timestamp() * 1000)
TIMEFRAME        = "4h"
CANDLES_PER_REQ  = 300

# ── BÜYÜK RALLİ PARAMETRELERİ ────────────────────────────────────────
MIN_RALLY_PCT    = 0.30    # Ralli büyüklüğü: en az %30 yükseliş
MIN_RALLY_CANDLES = 5      # Ralli süresi: en az 5 mum (= 20 saat)

# ── FİBONACCİ DÜZELTİ BÖLGE ─────────────────────────────────────────
# Zirveden dipten gelen yükselişin %61.8 – %78.6 geri çekilme bölgesi
FIB_LOWER        = 0.618   # Bölge alt sınırı
FIB_UPPER        = 0.786   # Bölge üst sınırı
FIB_TOLERANCE    = 0.03    # Her iki sınıra ±%3 esneklik
#   → Fiili arama bölgesi: 0.618×(1-0.03) = 0.5995 … 0.786×(1+0.03) = 0.8096

# ── İKİLİ DİP PARAMETRELERİ ──────────────────────────────────────────
MIN_BOUNCE_PCT   = 0.04    # Dip1 → Tepe arası min toparlanma (%4)
MAX_DIP2_ABOVE   = 0.03    # Dip2, Dip1'den en fazla %3 yüksekte olabilir
                            # (Dip2 ≤ Dip1 × 1.03)
MIN_RSI_DIVERGE  = 1.5     # RSI uyumsuzluğu: RSI₂ − RSI₁ ≥ 1.5 puan

# ── RSI ──────────────────────────────────────────────────────────────
RSI_PERIOD       = 14

# ── PENCERE ──────────────────────────────────────────────────────────
LOOKBACK_CANDLES = 300     # Kaç mum geriye bakılacak

# ── DİĞER ────────────────────────────────────────────────────────────
PAUSE_SEC        = 0.22
EXPORT_CSV       = True


def log_settings():
    log_status("✅  Ayarlar yüklendi — İkili Dip + RSI Diverjans Tarayıcı v1.1")
    log_status(f"   Zaman Dilimi    : {TIMEFRAME}")
    log_status(f"   Min Ralli       : %{MIN_RALLY_PCT*100:.0f}")
    log_status(f"   Fibo Bölge      : {FIB_LOWER:.3f} – {FIB_UPPER:.3f}  (±%{FIB_TOLERANCE*100:.0f} tolerans)")
    log_status(f"   Min RSI Diverjans: {MIN_RSI_DIVERGE} puan")


# ─── FONKSİYONLAR ─────────────────────────────────────────────────────

def compute_rsi(close, period=14):
    """Wilder EWM yöntemiyle RSI hesaplar."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def fetch_ohlcv_paginated(exchange, symbol, timeframe, since_ms, pause=0.22):
    """
    OKX API pagination: Nisan 2026'dan bugüne tüm 4h mumları çeker.
    4h'te Nisan'dan bu yana ~220 mum → genellikle tek istekle biter.
    """
    all_candles   = []
    current_since = since_ms
    now_ms        = int(datetime.now(timezone.utc).timestamp() * 1000)

    while current_since < now_ms:
        try:
            batch = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_since,
                limit=CANDLES_PER_REQ,
                params={"price": "last"}
            )
        except Exception:
            time.sleep(pause * 3)
            break

        if not batch:
            break

        all_candles.extend(batch)
        last_ts = batch[-1][0]

        if last_ts >= now_ms or len(batch) < CANDLES_PER_REQ:
            break

        current_since = last_ts + 1
        time.sleep(pause)

    if len(all_candles) < 20:
        return None

    df = pd.DataFrame(all_candles, columns=["ts","open","high","low","close","volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="ts").sort_values("ts")
    df = df.set_index("dt").drop(columns=["ts"]).astype(float)
    return df


def find_double_bottom_divergence(df, lookback):
    """
    ══════════════════════════════════════════════════════════════
    ANA ALGORİTMA — 5 Adım

    ADIM 1 — BÜYÜK RALLİ TESPİTİ
      Pencere içinde en güçlü dip→zirve hareketi bulunur.
      Şart: Yükseliş ≥ %30, süre ≥ 5 mum

    ADIM 2 — FİBONACCİ DÜZELTİ BÖLGESİ
      Ralli zirvesinden hesaplanan 0.618–0.786 geri çekilme bölgesi.
      Dip1 bu bölgede mi?

    ADIM 3 — DİP 1 TESPİTİ
      Zirveden sonra oluşan ilk anlamlı düşük (lokal minimum).

    ADIM 4 — TEPE (ARA TOPARLANMA)
      Dip1'den sonra en az %MIN_BOUNCE_PCT yükselen ilk tepe.

    ADIM 5 — DİP 2 + RSI POZİTİF UYUMSUZLUĞU
      • Dip2 fiyatı ≤ Dip1 × (1 + MAX_DIP2_ABOVE)   [fiyat teyidi]
      • RSI_dip2 > RSI_dip1 + MIN_RSI_DIVERGE        [diverjans teyidi]
      • Dip2, Fibo bölgesinde veya çok yakınında      [bölge teyidi]
    ══════════════════════════════════════════════════════════════
    """
    if len(df) < lookback + 5:
        return None

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    rsi    = compute_rsi(close, RSI_PERIOD)

    window     = df.iloc[-lookback:]
    win_close  = window["close"]
    win_high   = window["high"]
    win_low    = window["low"]
    win_rsi    = rsi.iloc[-lookback:]

    n = len(window)

    # ── ADIM 1: BÜYÜK RALLİ ──────────────────────────────────────────
    # Pencere içinde en büyük dip→zirve kombinasyonunu bul.
    # Yöntem: her olası dip pozisyonu için, sonrasındaki zirveyi kontrol et.

    best_rally = None   # (dip_pos, peak_pos, dip_price, peak_price, rally_pct)

    # Arama alanı: pencerenin ilk %70'i (sonrasında düzeltme + ikili dip için yer kalsın)
    search_limit = int(n * 0.70)

    for dip_pos in range(0, search_limit - MIN_RALLY_CANDLES):
        d_price = win_low.iloc[dip_pos]

        # Bu dipten sonraki en yüksek zirveyi bul
        future   = win_high.iloc[dip_pos + MIN_RALLY_CANDLES : search_limit + MIN_RALLY_CANDLES]
        if len(future) == 0:
            continue

        peak_rel = future.idxmax()
        p_price  = future[peak_rel]
        peak_pos = window.index.get_loc(peak_rel)

        rally_pct = (p_price - d_price) / d_price
        if rally_pct < MIN_RALLY_PCT:
            continue

        # En büyük ralliyi seç
        if best_rally is None or rally_pct > best_rally[4]:
            best_rally = (dip_pos, peak_pos, d_price, p_price, rally_pct)

    if best_rally is None:
        return None

    rally_dip_pos, rally_peak_pos, rally_dip_price, rally_peak_price, rally_pct = best_rally
    rally_range = rally_peak_price - rally_dip_price

    # ── ADIM 2: FİBONACCİ DÜZELTİ BÖLGESİ ──────────────────────────
    # 0.618 geri çekilme = zirve - 0.618 × ralli_aralığı
    # 0.786 geri çekilme = zirve - 0.786 × ralli_aralığı
    fib618_price = rally_peak_price - FIB_LOWER * rally_range
    fib786_price = rally_peak_price - FIB_UPPER * rally_range

    # Bölge sınırları (tolerans dahil)
    zone_upper = fib618_price * (1 + FIB_TOLERANCE)   # 0.618'in biraz üstü
    zone_lower = fib786_price * (1 - FIB_TOLERANCE)   # 0.786'nın biraz altı

    # ── ADIM 3: DİP 1 ────────────────────────────────────────────────
    # Ralli zirvesinden sonra oluşan ilk düşük nokta
    post_peak = window.iloc[rally_peak_pos + 1:]
    if len(post_peak) < 6:
        return None

    # İlk anlamlı dip: zirve sonrası en düşük Low (pencerenin geri kalanının ilk %60'ında)
    dip1_search_end = max(3, int(len(post_peak) * 0.60))
    dip1_idx   = post_peak["low"].iloc[:dip1_search_end].idxmin()
    dip1_pos   = post_peak.index.get_loc(dip1_idx)   # post_peak içi pozisyon
    dip1_price = post_peak["low"][dip1_idx]
    dip1_rsi   = win_rsi[dip1_idx]

    # Dip1 fiyatı, ralli zirvesinin altında olmalı
    if dip1_price >= rally_peak_price:
        return None

    # Dip1 fibo bölgede mi? (ana şart)
    if not (zone_lower <= dip1_price <= zone_upper):
        return None

    # ── ADIM 4: ARA TEPE (BOUNCE) ────────────────────────────────────
    # Dip1'den sonra en az %MIN_BOUNCE_PCT toparlanma
    post_dip1 = post_peak.iloc[dip1_pos + 1:]
    if len(post_dip1) < 3:
        return None

    bounce_idx   = post_dip1["high"].idxmax()
    bounce_price = post_dip1["high"][bounce_idx]
    bounce_pos   = post_dip1.index.get_loc(bounce_idx)

    bounce_pct = (bounce_price - dip1_price) / dip1_price
    if bounce_pct < MIN_BOUNCE_PCT:
        return None   # Toparlanma çok zayıf → gerçek ikili dip yapısı yok

    # ── ADIM 5: DİP 2 + RSI POZİTİF UYUMSUZLUĞU ─────────────────────
    # Bounce'dan sonra oluşan düşük nokta
    post_bounce = post_dip1.iloc[bounce_pos + 1:]
    if len(post_bounce) < 2:
        return None

    dip2_idx   = post_bounce["low"].idxmin()
    dip2_price = post_bounce["low"][dip2_idx]
    dip2_rsi   = win_rsi[dip2_idx]

    # Dip2 fiyat kontrolü: Dip1'den çok yüksekte olmamalı
    if dip2_price > dip1_price * (1 + MAX_DIP2_ABOVE):
        return None   # Dip2 çok yüksek → ikili dip yapısı değil

    # Dip2 fiyatı da fibo bölgede veya çok yakınında olmalı (±%5 genişletilmiş)
    zone_lower_ext = fib786_price * (1 - FIB_TOLERANCE - 0.02)
    zone_upper_ext = fib618_price * (1 + FIB_TOLERANCE + 0.02)
    if not (zone_lower_ext <= dip2_price <= zone_upper_ext):
        return None

    # RSI POZİTİF UYUMSUZLUK: Fiyat ≤ dip1 ama RSI > dip1_rsi
    rsi_divergence = dip2_rsi - dip1_rsi
    if rsi_divergence < MIN_RSI_DIVERGE:
        return None   # Diverjans yok veya negatif

    # ── ÇIKIŞ DEĞERLERİ ──────────────────────────────────────────────
    # Mevcut fiyat
    current_price   = close.iloc[-1]
    current_rsi     = rsi.iloc[-1]

    # Dip2'nin fibo bölgedeki konumu (0.618 mi 0.786 mı?)
    dip2_fib_level  = (rally_peak_price - dip2_price) / rally_range

    # Hedef: Ralli zirvesinin %50–%61.8 geri alımı (olası çıkış hedefi)
    target_50  = dip2_price + 0.500 * (rally_peak_price - dip2_price)
    target_618 = dip2_price + 0.618 * (rally_peak_price - dip2_price)

    # Risk/Reward (stop = Dip2'nin %2 altı)
    stop_loss   = dip2_price * 0.98
    risk        = dip2_price - stop_loss
    reward_50   = target_50 - current_price
    rr_ratio    = round(reward_50 / risk, 2) if risk > 0 else 0

    return {
        # Ralli
        "rally_dip"       : round(rally_dip_price, 6),
        "rally_peak"      : round(rally_peak_price, 6),
        "rally_pct"       : round(rally_pct * 100, 1),
        # Fibo bölge
        "fib618_price"    : round(fib618_price, 6),
        "fib786_price"    : round(fib786_price, 6),
        # İkili Dip
        "dip1_price"      : round(dip1_price, 6),
        "dip1_rsi"        : round(dip1_rsi, 2),
        "bounce_price"    : round(bounce_price, 6),
        "bounce_pct"      : round(bounce_pct * 100, 1),
        "dip2_price"      : round(dip2_price, 6),
        "dip2_rsi"        : round(dip2_rsi, 2),
        "dip2_fib_level"  : round(dip2_fib_level, 3),
        # RSI Diverjans
        "rsi_divergence"  : round(rsi_divergence, 2),
        # Mevcut durum
        "current_price"   : round(current_price, 6),
        "current_rsi"     : round(current_rsi, 2),
        # Hedefler
        "target_50pct"    : round(target_50, 6),
        "target_618pct"   : round(target_618, 6),
        "stop_loss"       : round(stop_loss, 6),
        "rr_ratio"        : rr_ratio,
        "candle_total"    : len(df),
    }


def _json_default(obj):
    """json.dump için numpy/pandas tiplerini native Python tiplerine çevirir."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"'{type(obj)}' JSON'a çevrilemiyor")


def write_signal_outputs(results, skipped_data, skipped_crit, total_scanned):
    """Bulunan sinyalleri konsol yerine signals.json / signals.log dosyalarına yazar."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeframe": TIMEFRAME,
        "parameters": {
            "min_rally_pct": MIN_RALLY_PCT,
            "fib_lower": FIB_LOWER,
            "fib_upper": FIB_UPPER,
            "fib_tolerance": FIB_TOLERANCE,
            "min_bounce_pct": MIN_BOUNCE_PCT,
            "max_dip2_above": MAX_DIP2_ABOVE,
            "min_rsi_diverge": MIN_RSI_DIVERGE,
            "rsi_period": RSI_PERIOD,
        },
        "summary": {
            "total_scanned": total_scanned,
            "skipped_no_data": skipped_data,
            "skipped_criteria_not_met": skipped_crit,
            "matched": len(results),
        },
        "signals": results,
    }

    with open(SIGNALS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    for result in results:
        log_signal(
            f"✅  {result['symbol']:<16} | "
            f"Ralli=%{result['rally_pct']:.0f} | "
            f"Dip1={result['dip1_price']} RSI={result['dip1_rsi']:.1f} | "
            f"Dip2={result['dip2_price']} RSI={result['dip2_rsi']:.1f} | "
            f"Diverjans=+{result['rsi_divergence']:.1f} | "
            f"Fib={result['dip2_fib_level']:.3f} | "
            f"R/R={result['rr_ratio']}"
        )

    log_status(f"💾  Sinyaller → {SIGNALS_JSON}")
    log_status(f"💾  Log       → {LOG_FILE}")


# ─── ANA TARAYICI ──────────────────────────────────────────────────────

def run_scanner():
    log_settings()
    log_status("🔌  OKX'e bağlanılıyor...")
    exchange = build_exchange()
    exchange.load_markets()

    usdt_pairs = [
        s for s, m in exchange.markets.items()
        if m.get("quote") == "USDT"
        and m.get("spot", False)
        and m.get("active", False)
        and "/USDT" in s
        and ":" not in s
    ]

    total = len(usdt_pairs)
    log_status(f"📊  {total} aktif USDT Spot çifti")
    log_status(f"⏳  Tahmini süre: ~{total * PAUSE_SEC / 60:.0f} dakika\n")

    results      = []
    skipped_data = 0
    skipped_crit = 0

    for symbol in tqdm(usdt_pairs, desc="🔍 Taranıyor", unit="coin"):

        df = fetch_ohlcv_paginated(
            exchange, symbol, TIMEFRAME,
            since_ms=START_DATE_MS,
            pause=PAUSE_SEC
        )

        if df is None or len(df) < 25:
            skipped_data += 1
            time.sleep(PAUSE_SEC)
            continue

        lookback = min(LOOKBACK_CANDLES, len(df) - 3)
        result   = find_double_bottom_divergence(df, lookback)

        if result is None:
            skipped_crit += 1
            time.sleep(PAUSE_SEC)
            continue

        result["symbol"] = symbol
        results.append(result)

        time.sleep(PAUSE_SEC)

    # ── Özet ─────────────────────────────────────────────────────────
    log_status(f"\n{'─'*65}")
    log_status(f"  Veri yetersiz  : {skipped_data}")
    log_status(f"  Kriter tutmadı : {skipped_crit}")
    log_status(f"  ✅  Eşleşen    : {len(results)}")

    write_signal_outputs(results, skipped_data, skipped_crit, total)

    if not results:
        log_status("\n❌  Hiç aday bulunamadı.")
        log_status("   İpucu → MIN_RALLY_PCT=0.25, MIN_RSI_DIVERGE=1.0 dene.")
        return None

    # ── DataFrame ────────────────────────────────────────────────────
    col_map = {
        "symbol"         : "Sembol",
        "current_price"  : "Mevcut Fiyat",
        "current_rsi"    : "RSI (Şimdi)",
        "rsi_divergence" : "RSI Diverjans (+puan)",
        "dip2_fib_level" : "Dip2 Fibo Seviyesi",
        "dip1_price"     : "Dip 1 Fiyatı",
        "dip1_rsi"       : "Dip 1 RSI",
        "bounce_price"   : "Ara Tepe",
        "bounce_pct"     : "Bounce%",
        "dip2_price"     : "Dip 2 Fiyatı",
        "dip2_rsi"       : "Dip 2 RSI",
        "fib618_price"   : "Fibo 0.618",
        "fib786_price"   : "Fibo 0.786",
        "rally_pct"      : "Ralli Büyüklük%",
        "rally_dip"      : "Ralli Başlangıç",
        "rally_peak"     : "Ralli Zirvesi",
        "target_50pct"   : "Hedef %50 Geri Alım",
        "target_618pct"  : "Hedef %61.8 Geri Alım",
        "stop_loss"      : "Stop Loss (Dip2-%2)",
        "rr_ratio"        : "R/R Oranı",
        "candle_total"   : "Toplam Mum",
    }

    df_out = pd.DataFrame(results)
    df_out = df_out[[c for c in col_map if c in df_out.columns]]
    df_out = df_out.rename(columns=col_map)

    # RSI diverjansına göre sırala (en güçlü diverjans önce)
    df_out = df_out.sort_values("RSI Diverjans (+puan)", ascending=False)
    df_out = df_out.reset_index(drop=True)

    log_status(f"\n{'═'*65}")
    log_status(f"  🎯  {len(df_out)} adet 'İkili Dip + RSI Diverjans' adayı! (detaylar {SIGNALS_JSON} / {LOG_FILE} dosyalarında)\n")

    if EXPORT_CSV:
        fn = "okx_double_bottom_divergence.csv"
        df_out.to_csv(fn, index=False, encoding="utf-8-sig")
        log_status(f"💾  CSV → {fn}")

    return df_out


# ─── TEK COİN DETAYLI TEST ───────────────────────────────────────────
def debug_test_symbol(test_symbol, debug=True):
    """
    Belirli bir coini test eder. Hangi adımda elendiğini görmek için
    debug=True kullanılabilir. Otomatik taramanın bir parçası değildir;
    yalnızca --test-symbol ile açıkça istendiğinde çalışır.
    """
    ex = build_exchange()
    ex.load_markets()

    log_status(f"\n🔬  {test_symbol} — Detaylı Test ({TIMEFRAME})")
    df_t = fetch_ohlcv_paginated(ex, test_symbol, TIMEFRAME, START_DATE_MS, pause=0.3)

    if df_t is None:
        log_status("⚠️  Veri çekilemedi.")
        return

    log_status(f"   Mum sayısı  : {len(df_t)}")
    log_status(f"   İlk mum     : {df_t.index[0]}")
    log_status(f"   Son mum     : {df_t.index[-1]}")
    log_status(f"   Fiyat aralığı: {df_t['low'].min():.6f} – {df_t['high'].max():.6f}")

    if debug:
        # Her adımı manuel çalıştır
        window = df_t.iloc[-min(LOOKBACK_CANDLES, len(df_t)-3):]

        # En büyük ralliyi bul ve göster
        log_status("\n── DEBUG: Ralli Taraması ──")
        n = len(window)
        best = None
        for dp in range(0, int(n*0.70) - MIN_RALLY_CANDLES):
            d_p = window["low"].iloc[dp]
            fut = window["high"].iloc[dp + MIN_RALLY_CANDLES : int(n*0.70)+MIN_RALLY_CANDLES]
            if len(fut) == 0: continue
            pp  = fut.max()
            rp  = (pp - d_p) / d_p
            if best is None or rp > best[2]:
                best = (d_p, pp, rp)
        if best:
            log_status(f"   En büyük ralli : {best[2]*100:.1f}%  ({best[0]:.6f} → {best[1]:.6f})")
            rng = best[1] - best[0]
            log_status(f"   Fibo 0.618     : {best[1] - 0.618*rng:.6f}")
            log_status(f"   Fibo 0.786     : {best[1] - 0.786*rng:.6f}")
        else:
            log_status("   Ralli bulunamadı")

    result_t = find_double_bottom_divergence(df_t, min(LOOKBACK_CANDLES, len(df_t)-3))

    if result_t:
        log_status("\n   ✅  ADAY — Tüm kriterler karşılandı!")
        for k, v in result_t.items():
            log_status(f"      {k:<22}: {v}")
    else:
        log_status("\n   ❌  Kriterler karşılanmıyor.")
        if debug:
            log_status("   → MIN_RALLY_PCT veya MIN_RSI_DIVERGE değerlerini gevşetmeyi dene.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="OKX Büyük Ralli Sonrası Fibo Bölgesinde İkili Dip + RSI Pozitif Uyumsuzluğu Tarayıcı"
    )
    parser.add_argument(
        "--test-symbol",
        default=os.environ.get("TEST_SYMBOL"),
        help="Taramaya ek olarak tek bir sembolü detaylı test et (örn. EUL/USDT)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("DEBUG_MODE", "").lower() in ("1", "true", "yes"),
        help="--test-symbol ile birlikte adım adım debug çıktısı üretir",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df_results = run_scanner()

    if args.test_symbol:
        debug_test_symbol(args.test_symbol, debug=args.debug)
