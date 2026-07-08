# ╔══════════════════════════════════════════════════════════════════════╗
# ║  OKX — A-B FİBONACCİ GERİ ÇEKİLME + ABC UZANTISI TARAYICI          ║
# ║  4 Saatlik | v2.0                                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# ARANAN YAPI (her coin için 90 → 60 → 30 → 15 → 7 → 3 gün sırasıyla denenir,
# ilk eşleşmede durulur):
#
#   Pencere içindeki en yüksek ve en düşük fiyat bulunur.
#   A = kronolojik olarak ilk oluşan uç (dip ya da zirve)
#   B = kronolojik olarak sonra oluşan uç
#
#   A dip, B zirve  → yükseliş bacağı  (Fibo seviyeleri B'den aşağı yönde)
#   A zirve, B dip  → düşüş bacağı    (Fibo seviyeleri B'den yukarı yönde)
#
#   Fibo 0.618  : A-B'nin %61.8 geri çekilme seviyesi
#   Fibo 1.272  : A-B mesafesinin B'den ters yönde %127.2 uzantısı (ABC)
#
#   SİNYAL: Güncel fiyat HEM 0.618 HEM 1.272 seviyesine ±%5 tolerans
#           içinde olmalı (nadir ama güçlü bir sinyal) + RSI teyidi.
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

from signal_validation import evaluate_signal

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


def timeframe_to_candles_per_day(timeframe):
    """'4h', '1h', '15m' gibi ccxt zaman dilimi string'ini günlük mum sayısına çevirir."""
    unit  = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "h":
        return max(1, round(24 / value))
    if unit == "m":
        return max(1, round((24 * 60) / value))
    if unit == "d":
        return max(1, round(1 / value))
    raise ValueError(f"Desteklenmeyen zaman dilimi: {timeframe}")


CANDLES_PER_DAY = timeframe_to_candles_per_day(TIMEFRAME)

# ── ZAMAN ARALIĞI ÖNCELİKLENDİRME ────────────────────────────────────
# Uzun aralıktan kısaya doğru denenir; ilk eşleşme bulunan aralıkta durulur.
TIME_WINDOWS_DAYS = [90, 60, 30, 15, 7, 3]

# ── FİBONACCİ SEVİYELERİ (A-B GERİ ÇEKİLME / ABC UZANTISI) ───────────
FIB_RETRACEMENT  = 0.618   # A-B hareketinin %61.8 geri çekilme seviyesi
FIB_EXTENSION    = 1.272   # A-B mesafesinin B'den ters yönde %127.2 uzantısı
SIGNAL_TOLERANCE = 0.05    # Güncel fiyat her iki seviyeye de ±%5 yakın olmalı

# ── RSI ──────────────────────────────────────────────────────────────
RSI_PERIOD       = 14
MIN_RSI_DIVERGE  = 1.5     # A noktası ile güncel RSI arasındaki minimum diverjans (puan)

# ── ZAMAN BAZLI CONFLUENCE PUANLAMASI (opsiyonel — sinyali reddetmez) ──
# Fibo 0.618/1.272 + RSI teyidi geçen bir sinyal, aşağıdaki iki zaman
# bazlı kontrolle EK puan alır. Bu kontroller hiçbir zaman bir sinyali
# reddetmez (gate değil); sadece confidence_score'a katkı sağlar.
TIME_PROJECTION_RATIOS       = [0.382, 0.618, 1.0, 1.272, 1.618]
TIME_CONFLUENCE_TOLERANCE_BARS = 2    # 4h'de ±2 mum = ±8 saat
TIME_SYMMETRY_TOLERANCE       = 0.2   # Geçmiş swing sürelerine göre ±%20
BASE_CONFIDENCE_SCORE         = 1.0   # Fibo+RSI kapısını geçen her sinyalin taban puanı

# ── SİNYAL TİPİ B: FİB CONFLUENCE + RSI OVERSOLD (bağımsız tarayıcı) ──
# İki bağımsız swing'in 0.618/0.786/1.272/1.618 seviyeleri güncel fiyatta
# çakışıyor mu + RSI aşırı satımda mı kontrol eder. find_ab_fibonacci_signal
# ve signal_validation.evaluate_signal'dan tamamen bağımsız, üçüncü bir
# tarayıcıdır.
FIB_CONFLUENCE_LEVELS    = [0.618, 0.786, 1.272, 1.618]
FIB_CONFLUENCE_TOLERANCE = float(os.environ.get("FIB_CONFLUENCE_TOLERANCE", 0.03))
RSI_OVERSOLD_THRESHOLD   = 35

# ── DİĞER ────────────────────────────────────────────────────────────
PAUSE_SEC        = 0.22
EXPORT_CSV       = True
MIN_CANDLES      = 40      # En kısa pencere (3g=18 mum) + RSI ısınma dönemi için asgari veri

# ── LİKİDİTE / STABLECOIN FİLTRELERİ ─────────────────────────────────
MIN_24H_VOLUME_USDT = float(os.environ.get("MIN_24H_VOLUME_USDT", 1_000_000))

STABLECOIN_BASES = {
    "USDC", "USDG", "USAT", "DAI", "TUSD", "FDUSD", "USDD", "USDP", "PYUSD",
    "GUSD", "USDK", "LUSD", "SUSD", "USTC", "USDE", "EUR", "EURT",
}


def is_stablecoin_pair(symbol):
    """'USDC/USDT' gibi iki tarafı da sabit fiyatlı parite mi kontrol eder."""
    base = symbol.split("/")[0].upper()
    return base in STABLECOIN_BASES


def log_settings():
    log_status("✅  Ayarlar yüklendi — A-B Fibonacci + ABC Uzantısı Tarayıcı v2.0")
    log_status(f"   Zaman Dilimi     : {TIMEFRAME}")
    log_status(f"   Zaman Aralıkları : {', '.join(str(d) + 'g' for d in TIME_WINDOWS_DAYS)} (önce uzun, sonra kısa)")
    log_status(f"   Fibo Seviyeleri  : {FIB_RETRACEMENT} (geri çekilme) / {FIB_EXTENSION} (ABC uzantısı)")
    log_status(f"   Sinyal Toleransı : ±%{SIGNAL_TOLERANCE*100:.0f}")
    log_status(f"   Min RSI Diverjans: {MIN_RSI_DIVERGE} puan")
    log_status(f"   Zaman Confluence : oranlar={TIME_PROJECTION_RATIOS}, tolerans=±{TIME_CONFLUENCE_TOLERANCE_BARS} bar (opsiyonel puan)")
    log_status(f"   Zaman Simetrisi  : tolerans=±%{TIME_SYMMETRY_TOLERANCE*100:.0f} (opsiyonel puan)")
    log_status(f"   Sinyal Tipi B    : Fib seviyeleri={FIB_CONFLUENCE_LEVELS}, tolerans=±%{FIB_CONFLUENCE_TOLERANCE*100:.0f}, RSI oversold<{RSI_OVERSOLD_THRESHOLD}")


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


def find_ab_fibonacci_signal(df, verbose=False):
    """
    ══════════════════════════════════════════════════════════════
    ANA ALGORİTMA — A-B Fibonacci Geri Çekilme + ABC Uzantısı

    Zaman aralıkları öncelik sırasıyla denenir: 90 → 60 → 30 → 15 → 7 → 3
    gün. İlk eşleşme bulunan aralıkta durulur, daha kısa aralıklara
    geçilmez.

    Her aralıkta:
      A = dönemin kronolojik olarak ilk oluşan ucu (en yüksek ya da en düşük)
      B = dönemin kronolojik olarak sonra oluşan ucu

      A dip, B zirve  → yükseliş bacağı (Fibo seviyeleri B'den aşağı yönde)
      A zirve, B dip  → düşüş bacağı   (Fibo seviyeleri B'den yukarı yönde)

      Fibo 0.618  : B - 0.618×|B-A|  (yükselişte) / B + 0.618×|B-A|  (düşüşte)
      Fibo 1.272  : B - 1.272×|B-A|  (yükselişte) / B + 1.272×|B-A|  (düşüşte)
        (A-B'nin ters yönünde, B'den itibaren projekte edilir → ABC uzantısı)

    SİNYAL ŞARTI:
      Güncel fiyat HEM 0.618 HEM 1.272 seviyesine ±SIGNAL_TOLERANCE
      içinde olmalı (iki koşul birlikte).

    RSI TEYİDİ:
      Yükseliş bacağında (A dip)  : RSI(güncel) − RSI(A) ≥ MIN_RSI_DIVERGE
      Düşüş bacağında   (A zirve) : RSI(A) − RSI(güncel) ≥ MIN_RSI_DIVERGE

    ÖNEMLİ: A/B, pencerenin SON (henüz kapanmamış/en güncel) mumu HARİÇ
    tutularak aranır. Aksi halde B = güncel mum olabilir; bu durumda
    "güncel fiyatın B'den hesaplanan hedefe yakınlığı" testi kendi
    verisiyle kendini doğrulayan anlamsız bir kontrole dönüşür (B zaten
    güncel mumun high/low'undan türetildiği için current_price'a
    yapay şekilde yakın çıkar).

    OPSİYONEL ZAMAN CONFLUENCE PUANI (gate değil, sinyali reddetmez):
      • Zaman projeksiyonu: A-B süresi (bar) × [0.382, 0.618, 1.0, 1.272,
        1.618] ile olası "dönüş" noktaları projekte edilir; B'den bugüne
        kadar geçen bar sayısı bu noktalardan birine ±2 bar yakınsa
        time_confluence=True.
      • Zaman simetrisi: Bu taramada (aynı sembol için) A-B'den ÖNCE
        denenmiş daha uzun pencerelerin kendi A-B swing süreleri, "geçmiş
        düzeltme süreleri" için kaba bir yaklaşım olarak kullanılır (script
        genel bir çoklu-pivot dedektörü içermediğinden). Güncel swing
        süresi bunların ortalamasına ±%20 yakınsa symmetry_match=True.
      • confidence_score = BASE_CONFIDENCE_SCORE + 0.5×time_confluence
        + 0.5×symmetry_match  (aralık: 1.0 – 2.0)
    ══════════════════════════════════════════════════════════════
    """
    close = df["close"]
    rsi   = compute_rsi(close, RSI_PERIOD)

    current_price = close.iloc[-1]
    current_rsi   = rsi.iloc[-1]
    current_pos   = len(df) - 1

    if pd.isna(current_rsi):
        return None

    # Bu taramada (cascade sırasında) hesaplanan A-B swing süreleri (bar).
    # Eşleşme bulunduğunda, o ana kadar denenmiş (daha uzun) pencerelerin
    # swing süreleri "geçmiş düzeltme süresi" yaklaşımı olarak kullanılır.
    past_swing_durations = []

    for window_days in TIME_WINDOWS_DAYS:
        window_candles = window_days * CANDLES_PER_DAY

        if len(df) < window_candles:
            if verbose:
                log_status(f"   [{window_days}g] atlandı → yetersiz veri ({len(df)} < {window_candles} mum)")
            continue

        window = df.iloc[-window_candles:]

        # Son mum A/B aramasından hariç tutulur (bkz. yukarıdaki ÖNEMLİ notu).
        confirmed_window = window.iloc[:-1]
        if len(confirmed_window) < 2:
            if verbose:
                log_status(f"   [{window_days}g] atlandı → teyitli mum sayısı yetersiz")
            continue

        high_time = confirmed_window["high"].idxmax()
        low_time  = confirmed_window["low"].idxmin()

        if high_time == low_time:
            if verbose:
                log_status(f"   [{window_days}g] atlandı → zirve ve dip aynı mumda (dejenere)")
            continue

        high_price = window["high"][high_time]
        low_price  = window["low"][low_time]

        if low_time < high_time:
            a_time, a_price, a_type = low_time, low_price, "dip"
            b_time, b_price, b_type = high_time, high_price, "zirve"
        else:
            a_time, a_price, a_type = high_time, high_price, "zirve"
            b_time, b_price, b_type = low_time, low_price, "dip"

        distance = abs(b_price - a_price)
        if distance <= 0:
            continue

        a_pos = df.index.get_loc(a_time)
        b_pos = df.index.get_loc(b_time)
        swing_duration_bars = b_pos - a_pos

        if a_type == "dip":   # Yükseliş bacağı (A dip → B zirve)
            fib618_price  = b_price - FIB_RETRACEMENT * distance
            ext1272_price = b_price - FIB_EXTENSION * distance
        else:                 # Düşüş bacağı (A zirve → B dip)
            fib618_price  = b_price + FIB_RETRACEMENT * distance
            ext1272_price = b_price + FIB_EXTENSION * distance

        if fib618_price <= 0 or ext1272_price <= 0:
            continue

        # Bu pencerenin (geçerli) swing süresi, sonraki (daha kısa) bir
        # pencerede eşleşme bulunursa zaman simetrisi kıyaslaması için
        # "geçmiş düzeltme süresi" adayı olarak saklanır.
        if swing_duration_bars > 0:
            past_swing_durations.append(swing_duration_bars)

        dist_fib618_pct  = abs(current_price - fib618_price) / fib618_price
        dist_ext1272_pct = abs(current_price - ext1272_price) / ext1272_price

        if verbose:
            log_status(
                f"   [{window_days}g] A={a_type}@{a_price:.6f} B={b_type}@{b_price:.6f} | "
                f"Fib618={fib618_price:.6f} (Δ%{dist_fib618_pct*100:.2f}) | "
                f"Ext1272={ext1272_price:.6f} (Δ%{dist_ext1272_pct*100:.2f})"
            )

        if dist_fib618_pct > SIGNAL_TOLERANCE or dist_ext1272_pct > SIGNAL_TOLERANCE:
            continue   # Bu aralıkta eşleşme yok → bir sonraki (daha kısa) aralığı dene

        a_rsi = rsi.loc[a_time]
        if pd.isna(a_rsi):
            if verbose:
                log_status(f"   [{window_days}g] atlandı → A noktasında RSI henüz hesaplanamıyor (ısınma dönemi)")
            continue

        if a_type == "dip":
            rsi_divergence = current_rsi - a_rsi
        else:
            rsi_divergence = a_rsi - current_rsi

        if rsi_divergence < MIN_RSI_DIVERGE:
            if verbose:
                log_status(f"   [{window_days}g] atlandı → RSI teyidi yetersiz (diverjans={rsi_divergence:.2f} < {MIN_RSI_DIVERGE})")
            continue   # RSI teyidi sağlanmadı → bu aralıkta sinyal yok

        # ── Opsiyonel zaman confluence puanı (gate değil) ─────────────
        current_bars_since_b = current_pos - b_pos

        projected_turn_bars = [swing_duration_bars * r for r in TIME_PROJECTION_RATIOS]
        time_confluence = any(
            abs(current_bars_since_b - p) <= TIME_CONFLUENCE_TOLERANCE_BARS
            for p in projected_turn_bars
        )

        # past_swing_durations'ın son elemanı bu swing'in kendisi; simetri
        # kıyaslaması "geçmiş" (bu taramada daha önce denenmiş daha uzun
        # pencerelerin) süreleriyle yapılır, kendisiyle değil.
        past_durations = past_swing_durations[:-1]
        symmetry_match = False
        avg_past_duration = None
        if past_durations:
            avg_past_duration = sum(past_durations) / len(past_durations)
            if avg_past_duration > 0:
                symmetry_match = abs(swing_duration_bars - avg_past_duration) / avg_past_duration < TIME_SYMMETRY_TOLERANCE

        confidence_score = BASE_CONFIDENCE_SCORE + (0.5 if time_confluence else 0) + (0.5 if symmetry_match else 0)

        if verbose:
            log_status(
                f"   [{window_days}g] zaman: swing={swing_duration_bars} bar, B'den bugüne={current_bars_since_b} bar, "
                f"projeksiyon={[round(p,1) for p in projected_turn_bars]} → confluence={time_confluence} | "
                f"geçmiş ort.={round(avg_past_duration,1) if avg_past_duration is not None else 'yok'} → simetri={symmetry_match} | "
                f"confidence_score={confidence_score}"
            )

        return {
            "window_days"        : window_days,
            "direction"          : "yukselis_sonrasi_duzeltme" if a_type == "dip" else "dusus_sonrasi_tepki",
            "a_type"             : a_type,
            "a_price"            : round(float(a_price), 6),
            "a_time"             : a_time.isoformat(),
            "b_type"             : b_type,
            "b_price"            : round(float(b_price), 6),
            "b_time"             : b_time.isoformat(),
            "fib618_price"       : round(float(fib618_price), 6),
            "ext1272_price"      : round(float(ext1272_price), 6),
            "current_price"      : round(float(current_price), 6),
            "dist_to_fib618_pct" : round(float(dist_fib618_pct) * 100, 3),
            "dist_to_ext1272_pct": round(float(dist_ext1272_pct) * 100, 3),
            "a_rsi"              : round(float(a_rsi), 2),
            "current_rsi"        : round(float(current_rsi), 2),
            "rsi_divergence"     : round(float(rsi_divergence), 2),
            "candle_total"       : len(df),
            "swing_duration_bars"  : int(swing_duration_bars),
            "current_bars_since_b" : int(current_bars_since_b),
            "time_confluence"       : bool(time_confluence),
            "symmetry_match"        : bool(symmetry_match),
            "past_swing_sample_size": len(past_durations),
            "confidence_score"      : round(float(confidence_score), 2),
        }

    return None


def compute_fib_levels(a_price, b_price, a_type, ratios):
    """Bir A-B swing'i için verilen fibonacci oranlarında seviye fiyatları hesaplar.

    find_ab_fibonacci_signal ile aynı yön mantığı: A dip ise seviyeler B'den
    aşağı yönde (retracement/uzantı destek gibi), A zirve ise B'den yukarı
    yönde (direnç gibi) projekte edilir.
    """
    distance = abs(b_price - a_price)
    levels = {}
    for r in ratios:
        levels[r] = (b_price - r * distance) if a_type == "dip" else (b_price + r * distance)
    return levels


def find_fib_confluence_signal(df, verbose=False):
    """
    ══════════════════════════════════════════════════════════════
    SİNYAL TİPİ B — Fib Confluence + RSI Oversold (BAĞIMSIZ tarayıcı)

    find_ab_fibonacci_signal (Sinyal A-B/ABC) ve signal_validation.
    evaluate_signal (Double Bottom, Sinyal Tipi A) ile paylaşılan bir
    kod yolu YOKTUR — tamamen ayrı, kendi başına çalışan üçüncü bir
    tarayıcıdır.

    MANTIK:
      1) Aynı sembol için TIME_WINDOWS_DAYS (90→3 gün) pencerelerinin
         her biri kendi A-B swing'ini üretir. Script genel bir
         çoklu-pivot swing dedektörü içermediğinden, "iki bağımsız
         swing" burada FARKLI zaman ölçeklerinde hesaplanan bu A-B
         uçları olarak ele alınmıştır (yaklaşımdır, gerçek ardışık
         geçmiş pivot taraması değildir). Küçük bir pencere büyük bir
         pencereyle birebir AYNI A-B çiftini üretirse (büyük pencerenin
         ekstremumu küçük pencerenin içine de düşerse) bu tekrar
         sayılmaz — gerçekte bağımsız iki swing yoktur.
      2) Her swing için 0.618/0.786/1.272/1.618 fibonacci seviyeleri
         hesaplanır (compute_fib_levels).
      3) FARKLI iki pencereden gelen swing'lerin herhangi bir seviye
         çifti, güncel fiyata AYRI AYRI ±FIB_CONFLUENCE_TOLERANCE
         içindeyse → fib_confluence (iki bağımsız swing güncel fiyat
         bölgesinde çakışıyor).
      4) Güncel RSI < RSI_OVERSOLD_THRESHOLD (aşırı satım) olmalı.

    fib_confluence VE rsi_oversold birlikte sağlanırsa sinyal üretilir.
    Bu, find_ab_fibonacci_signal'ın gate/skorlama mantığını etkilemez.
    ══════════════════════════════════════════════════════════════
    """
    close = df["close"]
    rsi   = compute_rsi(close, RSI_PERIOD)

    current_price = close.iloc[-1]
    current_rsi   = rsi.iloc[-1]

    if pd.isna(current_rsi):
        return None

    if current_rsi >= RSI_OVERSOLD_THRESHOLD:
        if verbose:
            log_status(f"   [FibConfluence] atlandı → RSI oversold değil (RSI={current_rsi:.2f} ≥ {RSI_OVERSOLD_THRESHOLD})")
        return None

    swings = []
    for window_days in TIME_WINDOWS_DAYS:
        window_candles = window_days * CANDLES_PER_DAY
        if len(df) < window_candles:
            continue

        window = df.iloc[-window_candles:]
        confirmed_window = window.iloc[:-1]   # bkz. find_ab_fibonacci_signal'daki B-özreferans notu
        if len(confirmed_window) < 2:
            continue

        high_time = confirmed_window["high"].idxmax()
        low_time  = confirmed_window["low"].idxmin()
        if high_time == low_time:
            continue

        high_price = window["high"][high_time]
        low_price  = window["low"][low_time]

        if low_time < high_time:
            a_time, a_price, a_type = low_time, low_price, "dip"
            b_time, b_price, b_type = high_time, high_price, "zirve"
        else:
            a_time, a_price, a_type = high_time, high_price, "zirve"
            b_time, b_price, b_type = low_time, low_price, "dip"

        if abs(b_price - a_price) <= 0:
            continue

        # Daha küçük bir pencere, daha büyük bir pencereyle AYNI A-B çiftini
        # üretebilir (büyük pencerenin ekstremumu küçük pencerenin içine de
        # düşerse). Bu durumda gerçekte "bağımsız" iki swing yoktur — aynı
        # swing'i iki kez saymamak için atlanır.
        if any(sw["a_time"] == a_time and sw["b_time"] == b_time for sw in swings):
            if verbose:
                log_status(f"   [FibConfluence] pencere={window_days}g atlandı → daha önceki bir pencereyle aynı A-B çifti (bağımsız değil)")
            continue

        levels = compute_fib_levels(a_price, b_price, a_type, FIB_CONFLUENCE_LEVELS)
        if any(lv <= 0 for lv in levels.values()):
            continue

        swings.append({
            "window_days": window_days,
            "a_type": a_type, "a_price": a_price, "a_time": a_time,
            "b_type": b_type, "b_price": b_price, "b_time": b_time,
            "levels": levels,
        })

        if verbose:
            level_str = ", ".join(f"{r}={v:.6f}" for r, v in levels.items())
            log_status(f"   [FibConfluence] pencere={window_days}g A({a_type})={a_price:.6f} B({b_type})={b_price:.6f} → {level_str}")

    if len(swings) < 2:
        if verbose:
            log_status("   [FibConfluence] atlandı → en az 2 bağımsız swing bulunamadı")
        return None

    best_match = None
    for i in range(len(swings)):
        for j in range(i + 1, len(swings)):
            swing_i, swing_j = swings[i], swings[j]
            for ratio_i, level_i in swing_i["levels"].items():
                dist_i = abs(current_price - level_i) / level_i
                if dist_i > FIB_CONFLUENCE_TOLERANCE:
                    continue
                for ratio_j, level_j in swing_j["levels"].items():
                    dist_j = abs(current_price - level_j) / level_j
                    if dist_j > FIB_CONFLUENCE_TOLERANCE:
                        continue

                    combined_dist = dist_i + dist_j
                    if best_match is None or combined_dist < best_match["combined_dist"]:
                        best_match = {
                            "combined_dist": combined_dist,
                            "swing_i": swing_i, "ratio_i": ratio_i, "level_i": level_i, "dist_i": dist_i,
                            "swing_j": swing_j, "ratio_j": ratio_j, "level_j": level_j, "dist_j": dist_j,
                        }

    if best_match is None:
        if verbose:
            log_status("   [FibConfluence] atlandı → hiçbir swing çifti güncel fiyatta çakışmadı")
        return None

    swing_i, swing_j = best_match["swing_i"], best_match["swing_j"]

    return {
        "current_price"        : round(float(current_price), 6),
        "current_rsi"          : round(float(current_rsi), 2),
        "rsi_oversold"          : True,
        "swing_1_window_days"   : swing_i["window_days"],
        "swing_1_a_type"        : swing_i["a_type"],
        "swing_1_a_price"       : round(float(swing_i["a_price"]), 6),
        "swing_1_a_time"        : swing_i["a_time"].isoformat(),
        "swing_1_b_type"        : swing_i["b_type"],
        "swing_1_b_price"       : round(float(swing_i["b_price"]), 6),
        "swing_1_b_time"        : swing_i["b_time"].isoformat(),
        "swing_1_fib_ratio"     : best_match["ratio_i"],
        "swing_1_fib_price"     : round(float(best_match["level_i"]), 6),
        "swing_1_dist_pct"      : round(float(best_match["dist_i"]) * 100, 3),
        "swing_2_window_days"   : swing_j["window_days"],
        "swing_2_a_type"        : swing_j["a_type"],
        "swing_2_a_price"       : round(float(swing_j["a_price"]), 6),
        "swing_2_a_time"        : swing_j["a_time"].isoformat(),
        "swing_2_b_type"        : swing_j["b_type"],
        "swing_2_b_price"       : round(float(swing_j["b_price"]), 6),
        "swing_2_b_time"        : swing_j["b_time"].isoformat(),
        "swing_2_fib_ratio"     : best_match["ratio_j"],
        "swing_2_fib_price"     : round(float(best_match["level_j"]), 6),
        "swing_2_dist_pct"      : round(float(best_match["dist_j"]) * 100, 3),
        "candle_total"          : len(df),
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


def validation_result_to_dict(symbol, validation, volume_24h_usdt=None):
    """signal_validation.SignalResult'ı JSON'a yazılabilir bir sözlüğe çevirir."""
    return {
        "symbol": symbol,
        "confidence": validation.confidence,
        "reasons": validation.reasons,
        "stop_price": round(float(validation.stop_price), 6) if validation.stop_price is not None else None,
        "position_size": round(float(validation.position_size), 6) if validation.position_size is not None else None,
        "fib_zone": (
            [round(float(validation.fib_zone[0]), 6), round(float(validation.fib_zone[1]), 6)]
            if validation.fib_zone is not None else None
        ),
        "volume_24h_usdt": round(float(volume_24h_usdt), 2) if volume_24h_usdt is not None else None,
    }


def write_signal_outputs(results, validation_results, fib_confluence_results,
                          skipped_data, skipped_crit, total_scanned):
    """Bulunan sinyalleri konsol yerine signals.json / signals.log dosyalarına yazar."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeframe": TIMEFRAME,
        "parameters": {
            "time_windows_days": TIME_WINDOWS_DAYS,
            "fib_retracement": FIB_RETRACEMENT,
            "fib_extension": FIB_EXTENSION,
            "signal_tolerance_pct": SIGNAL_TOLERANCE * 100,
            "min_rsi_diverge": MIN_RSI_DIVERGE,
            "rsi_period": RSI_PERIOD,
            "time_projection_ratios": TIME_PROJECTION_RATIOS,
            "time_confluence_tolerance_bars": TIME_CONFLUENCE_TOLERANCE_BARS,
            "time_symmetry_tolerance_pct": TIME_SYMMETRY_TOLERANCE * 100,
            "base_confidence_score": BASE_CONFIDENCE_SCORE,
            "fib_confluence_levels": FIB_CONFLUENCE_LEVELS,
            "fib_confluence_tolerance_pct": FIB_CONFLUENCE_TOLERANCE * 100,
            "rsi_oversold_threshold": RSI_OVERSOLD_THRESHOLD,
        },
        "summary": {
            "total_scanned": total_scanned,
            "skipped_no_data": skipped_data,
            "skipped_criteria_not_met": skipped_crit,
            "matched": len(results),
            "matched_double_bottom_validation": len(validation_results),
            "matched_fib_confluence": len(fib_confluence_results),
        },
        "signals": results,
        "double_bottom_validation_signals": validation_results,
        "fib_confluence_signals": fib_confluence_results,
    }

    with open(SIGNALS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    for result in results:
        log_signal(
            f"✅  {result['symbol']:<16} | "
            f"Pencere={result['window_days']}g | "
            f"A({result['a_type']})={result['a_price']} | "
            f"B({result['b_type']})={result['b_price']} | "
            f"Fib618={result['fib618_price']} (Δ%{result['dist_to_fib618_pct']:.2f}) | "
            f"Ext1272={result['ext1272_price']} (Δ%{result['dist_to_ext1272_pct']:.2f}) | "
            f"RSI Diverjans=+{result['rsi_divergence']:.2f} | "
            f"confidence={result['confidence_score']} "
            f"(time_confluence={result['time_confluence']}, symmetry={result['symmetry_match']})"
        )

    for v in validation_results:
        log_signal(
            f"✅  [Double Bottom Validation] {v['symbol']:<16} | "
            f"confidence={v['confidence']} | "
            f"stop={v['stop_price']} | pos_size={v['position_size']} | "
            f"reasons={v['reasons']}"
        )

    for f_sig in fib_confluence_results:
        log_signal(
            f"✅  [Fib Confluence] {f_sig['symbol']:<16} | "
            f"RSI={f_sig['current_rsi']} (oversold) | "
            f"Swing1({f_sig['swing_1_window_days']}g)={f_sig['swing_1_fib_ratio']}→{f_sig['swing_1_fib_price']} "
            f"(Δ%{f_sig['swing_1_dist_pct']:.2f}) | "
            f"Swing2({f_sig['swing_2_window_days']}g)={f_sig['swing_2_fib_ratio']}→{f_sig['swing_2_fib_price']} "
            f"(Δ%{f_sig['swing_2_dist_pct']:.2f})"
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
        and not is_stablecoin_pair(s)
    ]
    log_status(f"📊  {len(usdt_pairs)} aktif USDT Spot çifti (stablecoin hariç)")

    # ── Hacim/likidite filtresi ─────────────────────────────────────
    log_status(f"📈  24 saatlik hacim filtresi uygulanıyor (≥{MIN_24H_VOLUME_USDT:,.0f} USDT)...")
    volume_by_symbol = {}
    try:
        tickers = exchange.fetch_tickers(usdt_pairs)
    except Exception as exc:
        tickers = {}
        log_status(f"⚠️  Ticker verisi çekilemedi, hacim filtresi atlanıyor ({exc})")

    if tickers:
        liquid_pairs = []
        for s in usdt_pairs:
            vol = (tickers.get(s) or {}).get("quoteVolume")
            if vol is not None and vol >= MIN_24H_VOLUME_USDT:
                volume_by_symbol[s] = vol
                liquid_pairs.append(s)
        usdt_pairs = liquid_pairs

    total = len(usdt_pairs)
    log_status(f"📊  {total} likit USDT Spot çifti taranacak")
    log_status(f"⏳  Tahmini süre: ~{total * PAUSE_SEC / 60:.0f} dakika\n")

    results                = []
    validation_results     = []
    fib_confluence_results = []
    skipped_data       = 0
    skipped_crit       = 0

    for symbol in tqdm(usdt_pairs, desc="🔍 Taranıyor", unit="coin"):

        # DataFrame (OHLCV) burada üretilir.
        df = fetch_ohlcv_paginated(
            exchange, symbol, TIMEFRAME,
            since_ms=START_DATE_MS,
            pause=PAUSE_SEC
        )

        if df is None or len(df) < MIN_CANDLES:
            skipped_data += 1
            time.sleep(PAUSE_SEC)
            continue

        result = find_ab_fibonacci_signal(df)
        vol_24h = volume_by_symbol.get(symbol)

        # signal_validation.evaluate_signal() doğrulama katmanı: A-B tespiti
        # bir sinyal bulduysa, onun swing_high/swing_low'unu kullanır (son 60
        # bar max/min varsayılanından daha güvenilir); bulamadıysa
        # evaluate_signal kendi varsayılanına düşer.
        swing_high = swing_low = None
        if result is not None:
            swing_high = max(result["a_price"], result["b_price"])
            swing_low  = min(result["a_price"], result["b_price"])

        validation = evaluate_signal(df, swing_high=swing_high, swing_low=swing_low)

        if validation.is_valid:
            validation_results.append(validation_result_to_dict(symbol, validation, vol_24h))
        else:
            log_signal(f"ℹ️  [Double Bottom Validation] {symbol:<16} reddedildi | reasons={validation.reasons}")

        # Sinyal Tipi B: Fib Confluence + RSI Oversold — bağımsız üçüncü tarayıcı.
        fib_confluence_result = find_fib_confluence_signal(df)
        if fib_confluence_result is not None:
            fib_confluence_result["symbol"] = symbol
            fib_confluence_result["volume_24h_usdt"] = round(float(vol_24h), 2) if vol_24h is not None else None
            fib_confluence_results.append(fib_confluence_result)

        if result is None:
            skipped_crit += 1
            time.sleep(PAUSE_SEC)
            continue

        result["symbol"] = symbol
        result["volume_24h_usdt"] = round(float(vol_24h), 2) if vol_24h is not None else None
        results.append(result)

        time.sleep(PAUSE_SEC)

    # ── Özet ─────────────────────────────────────────────────────────
    log_status(f"\n{'─'*65}")
    log_status(f"  Veri yetersiz  : {skipped_data}")
    log_status(f"  Kriter tutmadı : {skipped_crit}")
    log_status(f"  ✅  Eşleşen (A-B Fibonacci)         : {len(results)}")
    log_status(f"  ✅  Eşleşen (Double Bottom Doğrulama): {len(validation_results)}")
    log_status(f"  ✅  Eşleşen (Fib Confluence — Tip B) : {len(fib_confluence_results)}")

    write_signal_outputs(results, validation_results, fib_confluence_results, skipped_data, skipped_crit, total)

    # Double Bottom doğrulama CSV'si, A-B Fibonacci sonucu olsun ya da
    # olmasın bağımsız olarak yazılır (iki strateji birbirinden ayrı).
    if EXPORT_CSV and validation_results:
        val_col_map = {
            "symbol"          : "Sembol",
            "confidence"      : "Güven",
            "reasons"         : "Nedenler",
            "stop_price"      : "Stop Fiyatı",
            "position_size"   : "Pozisyon Büyüklüğü",
            "fib_zone"        : "Fibo 0.618-0.786 Bandı",
            "volume_24h_usdt" : "24s Hacim (USDT)",
        }
        df_val = pd.DataFrame(validation_results)
        df_val = df_val[[c for c in val_col_map if c in df_val.columns]]
        df_val = df_val.rename(columns=val_col_map)

        val_fn = "okx_double_bottom_signals.csv"
        df_val.to_csv(val_fn, index=False, encoding="utf-8-sig")
        log_status(f"💾  CSV (Double Bottom)  → {val_fn}")

    # Fib Confluence (Sinyal Tipi B) CSV'si de bağımsız olarak yazılır.
    if EXPORT_CSV and fib_confluence_results:
        fib_col_map = {
            "symbol"              : "Sembol",
            "current_price"       : "Güncel Fiyat",
            "current_rsi"         : "Güncel RSI",
            "swing_1_window_days" : "Swing1 Pencere (gün)",
            "swing_1_a_type"      : "Swing1 A Tipi",
            "swing_1_b_type"      : "Swing1 B Tipi",
            "swing_1_fib_ratio"   : "Swing1 Fib Oranı",
            "swing_1_fib_price"   : "Swing1 Fib Fiyatı",
            "swing_1_dist_pct"    : "Swing1 Uzaklık%",
            "swing_2_window_days" : "Swing2 Pencere (gün)",
            "swing_2_a_type"      : "Swing2 A Tipi",
            "swing_2_b_type"      : "Swing2 B Tipi",
            "swing_2_fib_ratio"   : "Swing2 Fib Oranı",
            "swing_2_fib_price"   : "Swing2 Fib Fiyatı",
            "swing_2_dist_pct"    : "Swing2 Uzaklık%",
            "volume_24h_usdt"     : "24s Hacim (USDT)",
        }
        df_fib = pd.DataFrame(fib_confluence_results)
        df_fib = df_fib[[c for c in fib_col_map if c in df_fib.columns]]
        df_fib = df_fib.rename(columns=fib_col_map)

        fib_fn = "okx_fib_confluence_signals.csv"
        df_fib.to_csv(fib_fn, index=False, encoding="utf-8-sig")
        log_status(f"💾  CSV (Fib Confluence) → {fib_fn}")

    if not results:
        log_status("\n❌  A-B Fibonacci: Hiç aday bulunamadı.")
        log_status("   İpucu → SIGNAL_TOLERANCE=0.07, MIN_RSI_DIVERGE=1.0 dene.")
        return None

    # ── DataFrame ────────────────────────────────────────────────────
    col_map = {
        "symbol"              : "Sembol",
        "window_days"         : "Zaman Aralığı (gün)",
        "direction"           : "Yön",
        "a_type"              : "A Tipi",
        "a_price"             : "A Fiyatı",
        "a_time"              : "A Zamanı",
        "b_type"              : "B Tipi",
        "b_price"             : "B Fiyatı",
        "b_time"              : "B Zamanı",
        "fib618_price"        : "Fibo 0.618",
        "ext1272_price"       : "Fibo 1.272 (ABC)",
        "current_price"       : "Güncel Fiyat",
        "dist_to_fib618_pct"  : "0.618'e Uzaklık%",
        "dist_to_ext1272_pct" : "1.272'ye Uzaklık%",
        "a_rsi"               : "A RSI",
        "current_rsi"         : "Güncel RSI",
        "rsi_divergence"      : "RSI Diverjans",
        "candle_total"        : "Toplam Mum",
        "volume_24h_usdt"     : "24s Hacim (USDT)",
        "swing_duration_bars" : "Swing Süresi (bar)",
        "current_bars_since_b": "B'den Bugüne (bar)",
        "time_confluence"     : "Zaman Confluence",
        "symmetry_match"      : "Zaman Simetrisi",
        "confidence_score"    : "Confidence Skoru",
    }

    df_out = pd.DataFrame(results)
    df_out = df_out[[c for c in col_map if c in df_out.columns]]
    df_out = df_out.rename(columns=col_map)

    # Önce confidence skoruna (zaman confluence + simetri dahil), sonra
    # RSI diverjansına göre sırala (en güçlü sinyal önce)
    df_out = df_out.sort_values(["Confidence Skoru", "RSI Diverjans"], ascending=[False, False])
    df_out = df_out.reset_index(drop=True)

    log_status(f"\n{'═'*65}")
    log_status(f"  🎯  {len(df_out)} adet 'A-B Fibonacci + ABC Uzantısı' adayı! (detaylar {SIGNALS_JSON} / {LOG_FILE} dosyalarında)\n")

    if EXPORT_CSV:
        # Dosya adı stratejiyi yansıtır: bu bir double-bottom taraması DEĞİL,
        # A-B Fibonacci geri çekilme + ABC uzantısı sinyalleridir. Gerçek
        # double-bottom sonuçları yukarıda ayrı bir CSV'ye yazıldı.
        fn = "okx_ab_fibonacci_signals.csv"
        df_out.to_csv(fn, index=False, encoding="utf-8-sig")
        log_status(f"💾  CSV (A-B Fibonacci) → {fn}")

    return df_out


# ─── TEK COİN DETAYLI TEST ───────────────────────────────────────────
def debug_test_symbol(test_symbol, debug=True):
    """
    Belirli bir coini test eder. Hangi pencerede/adımda elendiğini görmek
    için debug=True kullanılabilir. Otomatik taramanın bir parçası değildir;
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
        log_status("\n── DEBUG: Zaman Aralığı Taraması ──")

    result_t = find_ab_fibonacci_signal(df_t, verbose=debug)

    if result_t:
        log_status("\n   ✅  ADAY — Tüm kriterler karşılandı!")
        for k, v in result_t.items():
            log_status(f"      {k:<22}: {v}")
    else:
        log_status("\n   ❌  Kriterler karşılanmıyor.")
        if debug:
            log_status("   → SIGNAL_TOLERANCE veya MIN_RSI_DIVERGE değerlerini gevşetmeyi dene.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="OKX A-B Fibonacci Geri Çekilme + ABC Uzantısı Tarayıcı"
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
