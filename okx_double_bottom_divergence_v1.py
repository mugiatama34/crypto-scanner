# ╔══════════════════════════════════════════════════════════════════════╗
# ║  OKX — FIBONACCI CONFLUENCE TARAYICI (tek strateji)                 ║
# ║  4 Saatlik | v4.0 (2026-07-10 pivot: Confluence artık ana strateji)  ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# GATE (find_confluence_candidates + evaluate_confluence_entry, ikisi de zorunlu):
#   1) İki BAĞIMSIZ swing'in (90/60/30 gün pencerelerinden) retracement/
#      extension seviyeleri (0.618/0.786/1.272/1.618) güncel fiyatta
#      AYRI AYRI ±%3 tolerans içinde çakışıyor (Boroden confluence)
#   2) Güncel RSI < 35 (aşırı satım)
#
# CONFIDENCE (gate geçildikten sonra kademelendirir, gate'i etkilemez):
#   low    = sadece gate (confluence + RSI oversold)
#   medium = + Double Bottom YAPISI var (signal_validation.check_double_bottom,
#            2 dip + mesafe + tolerans - kırılım/hacim şart değil)
#   high   = + Double Bottom TAM TEYİTLİ (signal_validation.evaluate_signal.is_valid,
#            yani kırılım+hacim+RSI diverjans+fib bandı da AYRICA sağlanmış)
#
# RİSK: stop-loss her zaman ATR bazlı (compute_stop_and_size); pozisyon
# büyüklüğü confidence kademesine göre 0.5R / 1.0R / 1.5R çarpanı alır.
#
# İZLEME LİSTESİ: gate'i tam geçemeyen ama toplam mesafesi makul bir
# sınırın (WATCHLIST_MAX_COMBINED_DIST_PCT) altında olan confluence
# adayları, önceki tura göre yaklaşıyor/uzaklaşıyor bilgisiyle raporlanır.
#
# NOT: Önceki mimaride (v3.0) Double Bottom ana gate, Confluence bonus
# katmanıydı. Kullanıcı tercihiyle 2026-07-10'da bu ilişki TERSİNE
# ÇEVRİLDİ. Motor 1 (A-B Fibonacci + ABC uzantısı, tek swing) tamamen
# kaldırıldı - confluence zaten onun daha güçlü (2 bağımsız swing) hali.
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

from signal_validation import (
    evaluate_signal,
    check_double_bottom, compute_stop_and_size,
)

warnings.filterwarnings("ignore")

# ── LOGLAMA ────────────────────────────────────────────────────────────
# Bulunan sinyaller konsola basılmaz; signals.log (insan-okunur) ve
# signals.json (makine-okunur) dosyalarına yazılır. GitHub Actions
# çalıştırmalarında bu dosyalar artifact olarak toplanabilir.
LOG_FILE     = os.environ.get("SIGNALS_LOG_FILE", "signals.log")
SIGNALS_JSON = os.environ.get("SIGNALS_JSON_FILE", "signals.json")
STATE_FILE   = os.environ.get("WATCHLIST_STATE_FILE", "watchlist_state.json")

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


def load_previous_watchlist_state():
    """Onceki tarama turunun watchlist snapshot'ini yukler (GitHub Actions
    cache'inden gelir). Dosya yoksa veya bozuksa bos dict doner - sistem
    kendi kendini onarir, hata firlatmaz."""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log_status(f"⚠️  Onceki watchlist state okunamadi, sifirdan baslaniyor ({exc})")
        return {}


def save_watchlist_state(watchlist_results):
    """Bu turun watchlist sonuclarini bir sonraki tur icin kaydeder
    (GitHub Actions cache adimi bunu tasir). reference_id, hangi 2 swing'in
    karsilastirildigini parmak izi olarak tutar - boylece bir sonraki tur,
    ayni swing ciftiyle mi kiyaslandigini yoksa referansin degisip
    degismedigini ayirt edebilir."""
    state = {
        entry["symbol"]: {
            "distance_pct": entry.get("combined_dist_pct"),
            "reference_id": entry.get("reference_id"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        for entry in watchlist_results
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def compute_momentum(symbol, current_distance_pct, previous_state, current_reference_id=None):
    """Onceki tura gore mesafe degisimini hesaplar. Pozitif deger =
    confluence koridoruna yaklasiyor (toplam mesafe kucaliyor), negatif =
    uzaklasiyor.

    ONEMLI: Karsilastirilan swing cifti (reference_id) degistiyse - yani
    yeni bir A-B swing tespit edildiyse - mesafe kiyaslamasi ELMA-ARMUT
    olur. Bu durumda numerik momentum yerine "referans_degisti" donulur,
    yanlis sinyal verilmez.
    """
    prev = previous_state.get(symbol)
    if prev is None or prev.get("distance_pct") is None or current_distance_pct is None:
        return None, "yeni"

    prev_reference_id = prev.get("reference_id")
    reference_changed = (
        prev_reference_id is not None and current_reference_id is not None
        and prev_reference_id != current_reference_id
    )
    if reference_changed:
        return None, "referans_degisti"

    momentum = round(float(prev["distance_pct"]) - float(current_distance_pct), 3)
    if momentum > 0.05:
        trend = "iyilesiyor"
    elif momentum < -0.05:
        trend = "kotulesiyor"
    else:
        trend = "sabit"
    return momentum, trend



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
# NOT: 15/7/3 gün pencereleri kaldırıldı (2026-07 refactor). Gerekçe: 3g=18
# mum, 7g=42 mum gibi kısa pencereler RSI'nin (14 periyot) ısınma süresine
# zar zor yetiyor ve Motor 3'ün (Fib Confluence) "bağımsız, yapısal olarak
# anlamlı swing" varsayımını zayıflatarak gürültüyü artırıyordu.
TIME_WINDOWS_DAYS = [90, 60, 30]

# ── RSI ──────────────────────────────────────────────────────────────
RSI_PERIOD       = 14

# ── CONFLUENCE (ANA STRATEJİ, 2026-07-10 pivot) ───────────────────────
# İki bağımsız swing'in retracement/extension seviyeleri (0.618/0.786/
# 1.272/1.618) güncel fiyatta çakışıyor mu + RSI aşırı satımda mı
# kontrol eder. find_confluence_candidates ana gate'in TEK kaynağıdır;
# Double Bottom (signal_validation.py) artık sadece confidence katmanı.
#
# 2026-07-10 (2. revizyon): confluence artık SADECE retracement-extension
# CAPRAZ eslesmesi ariyor (bir swing'in 0.618/0.786'si, BASKA bir swing'in
# 1.272/1.618'i ile cakisiyor mu) - iki retracement veya iki extension'in
# kendi aralarinda cakismasi confluence sayilmiyor. Ayrica iki swing'in
# HEM baslangic (A) HEM bitis (B) noktalari birbirinden farkli olmali -
# sadece biri farkliysa (ornegin ayni B, farkli A) gercekten bagimsiz
# sayilmiyor.
RETRACEMENT_RATIOS      = [0.618, 0.786]
EXTENSION_RATIOS        = [1.272, 1.618]
FIB_CONFLUENCE_LEVELS    = RETRACEMENT_RATIOS + EXTENSION_RATIOS
FIB_CONFLUENCE_TOLERANCE = float(os.environ.get("FIB_CONFLUENCE_TOLERANCE", 0.03))
RSI_OVERSOLD_THRESHOLD   = 35
# Izleme listesi: gate'i (within_tolerance) tam gecemeyen ama toplam mesafesi
# bu esigin altinda olan confluence adaylari "yaklasiyor/uzaklasiyor" olarak
# raporlanir. FIB_CONFLUENCE_TOLERANCE'in (%3+%3=%6 max gate) uzerinde,
# gurultuyu elemek icin makul bir tavan.
WATCHLIST_MAX_COMBINED_DIST_PCT = float(os.environ.get("WATCHLIST_MAX_COMBINED_DIST_PCT", 16.0))

# ── DİĞER ────────────────────────────────────────────────────────────
PAUSE_SEC        = 0.22
EXPORT_CSV       = True
MIN_CANDLES      = 40      # En kısa pencere (30g) + RSI ısınma dönemi için asgari veri

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
    log_status("✅  Ayarlar yüklendi — OKX Confluence Tarayıcı v4.0 (Double Bottom = bonus katmanı)")
    log_status(f"   Zaman Dilimi     : {TIMEFRAME}")
    log_status(f"   Zaman Aralıkları : {', '.join(str(d) + 'g' for d in TIME_WINDOWS_DAYS)} (önce uzun, sonra kısa)")
    log_status(f"   Confluence       : Fib seviyeleri={FIB_CONFLUENCE_LEVELS}, tolerans=±%{FIB_CONFLUENCE_TOLERANCE*100:.0f}, RSI oversold<{RSI_OVERSOLD_THRESHOLD}")
    log_status(f"   İzleme sınırı    : toplam mesafe ≤ %{WATCHLIST_MAX_COMBINED_DIST_PCT:.0f}")


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



def compute_fib_levels(a_price, b_price, a_type, ratios):
    """Bir A-B swing'i için verilen fibonacci oranlarında seviye fiyatları hesaplar.

    find_confluence_candidates ile aynı yön mantığı: A dip ise seviyeler B'den
    aşağı yönde (retracement/uzantı destek gibi), A zirve ise B'den yukarı
    yönde (direnç gibi) projekte edilir.
    """
    distance = abs(b_price - a_price)
    levels = {}
    for r in ratios:
        levels[r] = (b_price - r * distance) if a_type == "dip" else (b_price + r * distance)
    return levels


def find_confluence_candidates(df, watch_tolerance=0.08, verbose=False):
    """
    ══════════════════════════════════════════════════════════════
    CONFLUENCE ADAYI TESPITI (2026-07 refactor: artik ANA GATE'in
    kaynagi. Eskiden "Motor 3 / Sinyal Tipi B" adiyla sadece bonus
    puan veren bagimsiz bir tarayiciydi; artik stratejinin merkezi.)

    MANTIK (Boroden confluence teorisi):
      1) TIME_WINDOWS_DAYS (90/60/30 gun) pencerelerinin her biri kendi
         A-B swing'ini uretir. Farkli zaman olceklerinden gelen bu uclar
         "bagimsiz swing" adayidir - ama GERCEK bagimsizlik icin iki
         swing'in HEM A HEM B noktalari birbirinden FARKLI olmali (sadece
         B'nin ayni olmasi -ornegin ayni guncel tepe/dip iki farkli
         pencerede de "son uc" olarak secilirse- bagimsiz sayilmaz).
      2) Her swing icin retracement (0.618/0.786) VE extension (1.272/
         1.618) seviyeleri hesaplanir.
      3) SADECE CAPRAZ eslesme aranir: bir swing'in RETRACEMENT seviyesi,
         BASKA bir (A ve B'si farkli) swing'in EXTENSION seviyesiyle
         cakisiyor mu. Iki retracement'in veya iki extension'in kendi
         aralarinda cakismasi confluence SAYILMAZ - WIF/TRX ornegindeki
         gercek confluence deseni budur (bir swing'in geri cekilme
         bolgesi, baska bir swing'in projeksiyon hedefiyle ortusuyor).
      4) TUM gecerli (swing_i, swing_j, ratio_i, ratio_j) kombinasyonlari
         arasindan, guncel fiyata TOPLAM mesafesi (dist_i + dist_j) en
         kucuk olan cift secilir - toleransin icinde olsun ya da olmasin.
         Bu, hem GATE (tam eslesme) hem WATCHLIST (yaklasan ama henuz
         tam eslesmeyen) icin TEK bir ortak kaynak olmasini saglar.

    RSI aşırı satim SARTI BURADA UYGULANMAZ (eskiden erken donus vardi,
    bu yuzden RSI oversold degilken watchlist adaylari hic bulunamiyordu).
    Onun yerine current_rsi ve rsi_oversold bilgisi sonuca eklenir; GATE
    kararini veren ust katman (run_scanner) bunu kullanir.

    Donus: None (2'den az bagimsiz swing varsa) VEYA best-match dict:
      {
        ... swing_1_*/swing_2_* alanlari (eskisiyle ayni format) ...,
        "dist_i_pct", "dist_j_pct": her ayagin ayri ayri % mesafesi,
        "combined_dist_pct": toplam mesafe (watchlist siralamasi icin),
        "within_tolerance": HEM dist_i HEM dist_j <= FIB_CONFLUENCE_TOLERANCE mi,
        "current_rsi", "rsi_oversold": RSI durumu (gate karari ust katmanda),
      }
    ══════════════════════════════════════════════════════════════
    """
    close = df["close"]
    rsi   = compute_rsi(close, RSI_PERIOD)

    current_price = close.iloc[-1]
    current_rsi   = rsi.iloc[-1]

    if pd.isna(current_rsi):
        return None, "rsi_hesaplanamiyor"

    swings = []
    for window_days in TIME_WINDOWS_DAYS:
        window_candles = window_days * CANDLES_PER_DAY
        if len(df) < window_candles:
            continue

        window = df.iloc[-window_candles:]
        confirmed_window = window.iloc[:-1]
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

        if any(sw["a_time"] == a_time and sw["b_time"] == b_time for sw in swings):
            if verbose:
                log_status(f"   [Confluence] pencere={window_days}g atlandı → aynı A-B çifti (bağımsız değil)")
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
            log_status(f"   [Confluence] pencere={window_days}g A({a_type})={a_price:.6f} B({b_type})={b_price:.6f} → {level_str}")

    if len(swings) < 2:
        if verbose:
            log_status("   [Confluence] atlandı → en az 2 geçerli pencere swing'i bulunamadı (veri yetersiz)")
        return None, "yetersiz_pencere"

    independent_pair_exists = False
    best_match = None
    for i in range(len(swings)):
        for j in range(i + 1, len(swings)):
            swing_i, swing_j = swings[i], swings[j]

            # Gercek bagimsizlik: HEM A HEM B noktalari farkli olmali.
            # Sadece biri (ozellikle guncel/son ekstremum olan B) ortaksa
            # bu iki swing gercekte bagimsiz degil - ayni referans noktasina
            # dayanan farkli boy olcumleri, gercek "iki ayri swing" degil.
            shares_a_point = (
                swing_i["a_time"] == swing_j["a_time"] or swing_i["a_time"] == swing_j["b_time"]
                or swing_i["b_time"] == swing_j["a_time"] or swing_i["b_time"] == swing_j["b_time"]
            )
            if shares_a_point:
                if verbose:
                    log_status(
                        f"   [Confluence] {swing_i['window_days']}g × {swing_j['window_days']}g atlandı "
                        f"→ ortak nokta paylaşıyorlar (gerçekten bağımsız değil)"
                    )
                continue

            independent_pair_exists = True

            for ratio_i, level_i in swing_i["levels"].items():
                dist_i = abs(current_price - level_i) / level_i
                # cok uzak (watch_tolerance'in cok disinda) kombinasyonlari
                # erken elemek performans icin - ama tam esitsizlik degil,
                # gevsek bir on-filtre (watch_tolerance zaten gate toleransindan genis)
                if dist_i > watch_tolerance * 2:
                    continue
                for ratio_j, level_j in swing_j["levels"].items():
                    # SADECE CAPRAZ eslesme: biri retracement biri extension
                    # olmali. Iki retracement veya iki extension'in kendi
                    # aralarinda cakismasi confluence sayilmaz.
                    i_is_retracement = ratio_i in RETRACEMENT_RATIOS
                    j_is_retracement = ratio_j in RETRACEMENT_RATIOS
                    if i_is_retracement == j_is_retracement:
                        continue

                    dist_j = abs(current_price - level_j) / level_j
                    if dist_j > watch_tolerance * 2:
                        continue

                    combined_dist = dist_i + dist_j
                    if best_match is None or combined_dist < best_match["combined_dist"]:
                        best_match = {
                            "combined_dist": combined_dist,
                            "swing_i": swing_i, "ratio_i": ratio_i, "level_i": level_i, "dist_i": dist_i,
                            "swing_j": swing_j, "ratio_j": ratio_j, "level_j": level_j, "dist_j": dist_j,
                        }

    if best_match is None:
        if not independent_pair_exists:
            reason = "tumu_ortak_nokta_paylasiyor"
            if verbose:
                log_status("   [Confluence] atlandı → bulunan tüm swing çiftleri ortak nokta paylaşıyor (gerçekten bağımsız değil)")
        else:
            reason = "cok_uzak_veya_capraz_tip_yok"
            if verbose:
                log_status("   [Confluence] atlandı → bağımsız çiftler var ama hiçbiri makul mesafede çapraz eşleşme vermiyor")
        return None, reason

    swing_i, swing_j = best_match["swing_i"], best_match["swing_j"]
    within_tolerance = (
        best_match["dist_i"] <= FIB_CONFLUENCE_TOLERANCE
        and best_match["dist_j"] <= FIB_CONFLUENCE_TOLERANCE
    )

    return {
        "current_price"        : round(float(current_price), 6),
        "current_rsi"          : round(float(current_rsi), 2),
        "rsi_oversold"          : bool(current_rsi < RSI_OVERSOLD_THRESHOLD),
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
        "combined_dist_pct"     : round(float(best_match["combined_dist"]) * 100, 3),
        "within_tolerance"      : bool(within_tolerance),
        "candle_total"          : len(df),
    }, None


def evaluate_confluence_entry(df, candidate, equity=1000, risk_pct=0.015):
    """
    ══════════════════════════════════════════════════════════════
    YENI ANA GATE (2026-07-10 pivot): Confluence artik birincil strateji,
    Double Bottom bonus/confidence katmanina indirgendi (eskiden tam
    tersiydi). Kullanicinin gercek giris mantigi: iki bagimsiz swing'in
    retracement + extension seviyelerinin fiyatta cakismasi.

    GATE (ikisi de zorunlu):
      1) candidate["within_tolerance"] - HEM swing_1 HEM swing_2'nin
         fib seviyesi guncel fiyata FIB_CONFLUENCE_TOLERANCE icinde
      2) candidate["rsi_oversold"] - RSI < RSI_OVERSOLD_THRESHOLD

    CONFIDENCE (gate gectikten sonra kademelendirir):
      low    = sadece gate
      medium = gate + Double Bottom YAPISI var (2 dip + mesafe + tolerans,
               kirilim/hacim/RSI/fib şart degil - sadece yapi)
      high   = gate + Double Bottom TAM TEYITLI (evaluate_signal.is_valid)
               yani kirilim+hacim+RSI diverjans+fib bandi da AYRICA saglanmis

    RISK: stop ATR bazli (confidence'tan bagimsiz), pozisyon 0.5R/1.0R/1.5R.

    Donus: None (gate gecmedi) VEYA dict (sinyal detaylari).
    ══════════════════════════════════════════════════════════════
    """
    if candidate is None or not candidate.get("within_tolerance") or not candidate.get("rsi_oversold"):
        return None

    entry_price = candidate["current_price"]

    db = check_double_bottom(df)
    double_bottom_structure = "low1_idx" in db

    full_db_validation = evaluate_signal(df)
    double_bottom_fully_confirmed = bool(full_db_validation.is_valid)

    if double_bottom_fully_confirmed:
        confidence = "high"
    elif double_bottom_structure:
        confidence = "medium"
    else:
        confidence = "low"

    risk_calc = compute_stop_and_size(df, entry_price, equity, risk_pct)
    position_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5}[confidence]
    position_size = risk_calc["position_size"] * position_multiplier

    return {
        "confidence": confidence,
        "double_bottom_structure": double_bottom_structure,
        "double_bottom_fully_confirmed": double_bottom_fully_confirmed,
        "entry_price": round(float(entry_price), 6),
        "stop_price": round(float(risk_calc["stop_price"]), 6),
        "position_size": round(float(position_size), 6),
        "swing_1_window_days": candidate["swing_1_window_days"],
        "swing_1_fib_ratio": candidate["swing_1_fib_ratio"],
        "swing_1_fib_price": candidate["swing_1_fib_price"],
        "swing_1_dist_pct": candidate["swing_1_dist_pct"],
        "swing_2_window_days": candidate["swing_2_window_days"],
        "swing_2_fib_ratio": candidate["swing_2_fib_ratio"],
        "swing_2_fib_price": candidate["swing_2_fib_price"],
        "swing_2_dist_pct": candidate["swing_2_dist_pct"],
        "current_rsi": candidate["current_rsi"],
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


def write_signal_outputs(validation_results, watchlist_results, skipped_data, skipped_crit, total_scanned):
    """Bulunan sinyalleri (tek strateji: Double Bottom gate) VE izleme
    listesini signals.json / signals.log dosyalarına yazar."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeframe": TIMEFRAME,
        "strategy": "double_bottom_gate_v2 (2026-07 refactor: tek gate + confidence hiyerarsisi)",
        "parameters": {
            "time_windows_days": TIME_WINDOWS_DAYS,
            "rsi_period": RSI_PERIOD,
            "fib_confluence_levels": FIB_CONFLUENCE_LEVELS,
            "fib_confluence_tolerance_pct": FIB_CONFLUENCE_TOLERANCE * 100,
            "rsi_oversold_threshold": RSI_OVERSOLD_THRESHOLD,
        },
        "summary": {
            "total_scanned": total_scanned,
            "skipped_no_data": skipped_data,
            "skipped_gate_rejected": skipped_crit,
            "matched": len(validation_results),
            "watchlist": len(watchlist_results),
        },
        "signals": validation_results,
        "watchlist": watchlist_results,
    }

    with open(SIGNALS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    for v in validation_results:
        log_signal(
            f"✅  {v['symbol']:<16} | confidence={v['confidence']:<6} | "
            f"db_yapi={v['double_bottom_structure']} | db_tam_teyit={v['double_bottom_fully_confirmed']} | "
            f"entry={v['entry_price']} | stop={v['stop_price']} | pos_size={v['position_size']}"
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

    # Motor 1 (A-B Fibonacci+ABC) ve Motor 3 (Fib Confluence) artık kendi
    # başlarına sonuç biriktirmiyor — evaluate_signal() içine bonus/confidence
    # girdisi olarak akıyorlar. Tek gerçek sinyal listesi: validation_results.
    validation_results = []
    watchlist_results   = []
    skipped_data        = 0
    skipped_crit         = 0
    previous_watchlist_state = load_previous_watchlist_state()
    log_status(f"📈  Onceki tur state: {len(previous_watchlist_state)} coin (momentum karsilastirmasi icin)")

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

        # 2026-07-10 pivot: Confluence artik ANA GATE. Double Bottom
        # (evaluate_confluence_entry icinde) bonus/confidence katmani.
        # Motor 1 (A-B/ABC) artik hic kullanilmiyor - confluence zaten
        # onun yaptigi isin daha guclu (2-swing) versiyonu.
        candidate, no_candidate_reason = find_confluence_candidates(df)
        vol_24h = volume_by_symbol.get(symbol)

        confluence_signal = evaluate_confluence_entry(df, candidate)

        if confluence_signal is not None:
            confluence_signal["symbol"] = symbol
            confluence_signal["volume_24h_usdt"] = round(float(vol_24h), 2) if vol_24h is not None else None
            validation_results.append(confluence_signal)
            log_signal(
                f"✅  {symbol:<16} | confidence={confluence_signal['confidence']:<6} | "
                f"swing1={confluence_signal['swing_1_fib_ratio']}@%{confluence_signal['swing_1_dist_pct']} | "
                f"swing2={confluence_signal['swing_2_fib_ratio']}@%{confluence_signal['swing_2_dist_pct']} | "
                f"stop={confluence_signal['stop_price']} | pos_size={confluence_signal['position_size']}"
            )
        else:
            skipped_crit += 1
            if candidate is None:
                reason_map = {
                    "yetersiz_pencere": "En az 2 geçerli pencere swing'i bulunamadı (veri yetersiz)",
                    "tumu_ortak_nokta_paylasiyor": "Bulunan swing çiftlerinin TÜMÜ ortak nokta paylaşıyor (gerçekten bağımsız değil)",
                    "cok_uzak_veya_capraz_tip_yok": "Bağımsız çiftler var ama hiçbiri makul mesafede çapraz (retracement×extension) eşleşme vermiyor",
                    "rsi_hesaplanamiyor": "RSI henüz hesaplanamıyor (ısınma dönemi)",
                }
                reason_text = reason_map.get(no_candidate_reason, no_candidate_reason)
                log_signal(f"ℹ️  [Confluence] {symbol:<16} reddedildi | reasons=['{reason_text}']")
            else:
                reasons = []
                if not candidate.get("within_tolerance"):
                    reasons.append(f"Confluence tolerans dışı (toplam mesafe %{candidate['combined_dist_pct']})")
                if not candidate.get("rsi_oversold"):
                    reasons.append(f"RSI oversold değil (RSI={candidate['current_rsi']})")
                log_signal(f"ℹ️  [Confluence] {symbol:<16} reddedildi | reasons={reasons}")

                # Gate'i tam geçemedi ama toplam mesafesi makul bir sınırın
                # altındaysa izleme listesine al. Bu İŞLEM TETİKLEMEZ.
                if (not candidate["within_tolerance"]
                        and candidate["combined_dist_pct"] <= WATCHLIST_MAX_COMBINED_DIST_PCT):
                    reference_id = (
                        f"{candidate['swing_1_a_time']}|{candidate['swing_1_b_time']}|"
                        f"{candidate['swing_2_a_time']}|{candidate['swing_2_b_time']}"
                    )
                    momentum, trend = compute_momentum(
                        symbol, candidate["combined_dist_pct"], previous_watchlist_state,
                        current_reference_id=reference_id,
                    )
                    watch = {
                        "symbol": symbol,
                        "trend": trend,
                        "momentum_pct": momentum,
                        "combined_dist_pct": candidate["combined_dist_pct"],
                        "current_rsi": candidate["current_rsi"],
                        "rsi_oversold": candidate["rsi_oversold"],
                        "swing_1_window_days": candidate["swing_1_window_days"],
                        "swing_1_fib_ratio": candidate["swing_1_fib_ratio"],
                        "swing_1_fib_price": candidate["swing_1_fib_price"],
                        "swing_1_dist_pct": candidate["swing_1_dist_pct"],
                        "swing_1_a_time": candidate["swing_1_a_time"],
                        "swing_1_b_time": candidate["swing_1_b_time"],
                        "swing_2_window_days": candidate["swing_2_window_days"],
                        "swing_2_fib_ratio": candidate["swing_2_fib_ratio"],
                        "swing_2_fib_price": candidate["swing_2_fib_price"],
                        "swing_2_dist_pct": candidate["swing_2_dist_pct"],
                        "swing_2_a_time": candidate["swing_2_a_time"],
                        "swing_2_b_time": candidate["swing_2_b_time"],
                        "reference_id": reference_id,
                        "volume_24h_usdt": round(float(vol_24h), 2) if vol_24h is not None else None,
                    }
                    watchlist_results.append(watch)
                    log_signal(
                        f"👀  [İzleme] {symbol:<16} | toplam_mesafe=%{watch['combined_dist_pct']} | "
                        f"trend={trend} (momentum={momentum}) | rsi_oversold={watch['rsi_oversold']}"
                    )

        time.sleep(PAUSE_SEC)

    # ── Özet ─────────────────────────────────────────────────────────
    log_status(f"\n{'─'*65}")
    log_status(f"  Veri yetersiz  : {skipped_data}")
    log_status(f"  Kriter tutmadı (gate reddedildi) : {skipped_crit}")
    log_status(f"  ✅  Eşleşen (Confluence — tek strateji): {len(validation_results)}")

    write_signal_outputs(validation_results, watchlist_results, skipped_data, skipped_crit, total)
    save_watchlist_state(watchlist_results)

    # ── İzleme Listesi CSV'si (0 sinyal olsa bile HER ZAMAN yazılır) ──
    if watchlist_results:
        watch_col_map = {
            "symbol"              : "Sembol",
            "trend"                : "Trend",
            "momentum_pct"         : "Momentum (önceki tura göre)",
            "combined_dist_pct"    : "Toplam Mesafe%",
            "current_rsi"          : "Güncel RSI",
            "rsi_oversold"         : "RSI Oversold mu",
            "swing_1_window_days"  : "Swing1 Pencere (gün)",
            "swing_1_fib_ratio"    : "Swing1 Fib Oranı",
            "swing_1_fib_price"    : "Swing1 Fib Fiyatı",
            "swing_1_dist_pct"     : "Swing1 Uzaklık%",
            "swing_1_a_time"       : "Swing1 A Zamanı",
            "swing_1_b_time"       : "Swing1 B Zamanı",
            "swing_2_window_days"  : "Swing2 Pencere (gün)",
            "swing_2_fib_ratio"    : "Swing2 Fib Oranı",
            "swing_2_fib_price"    : "Swing2 Fib Fiyatı",
            "swing_2_dist_pct"     : "Swing2 Uzaklık%",
            "swing_2_a_time"       : "Swing2 A Zamanı",
            "swing_2_b_time"       : "Swing2 B Zamanı",
            "volume_24h_usdt"      : "24s Hacim (USDT)",
        }
        df_watch = pd.DataFrame(watchlist_results)
        df_watch = df_watch[[c for c in watch_col_map if c in df_watch.columns]]
        df_watch = df_watch.rename(columns=watch_col_map)
        # "iyilesiyor" trendindekiler once, sonra toplam mesafeye gore
        trend_rank = {"iyilesiyor": 3, "yeni": 1, "sabit": 1, "referans_degisti": 1, "kotulesiyor": 0}
        df_watch["_trend_rank"] = df_watch["Trend"].map(trend_rank)
        df_watch = df_watch.sort_values(["_trend_rank", "Toplam Mesafe%"], ascending=[False, True])
        df_watch = df_watch.drop(columns=["_trend_rank"]).reset_index(drop=True)

        if EXPORT_CSV:
            watch_fn = "okx_watchlist.csv"
            df_watch.to_csv(watch_fn, index=False, encoding="utf-8-sig")
            log_status(f"👀  CSV (İzleme Listesi, {len(df_watch)} coin) → {watch_fn}")

    if not validation_results:
        log_status("\n❌  Hiç aday bulunamadı (Confluence gate'ini geçen sinyal yok).")
        return None

    # ── Tek birleşik sinyal CSV çıktısı ────────────────────────────────
    col_map = {
        "symbol"                        : "Sembol",
        "confidence"                     : "Güven",
        "double_bottom_structure"        : "Double Bottom Yapısı Var mı",
        "double_bottom_fully_confirmed"  : "Double Bottom Tam Teyitli mi",
        "entry_price"                    : "Giriş Fiyatı",
        "stop_price"                     : "Stop Fiyatı",
        "position_size"                  : "Pozisyon Büyüklüğü",
        "swing_1_window_days"            : "Swing1 Pencere (gün)",
        "swing_1_fib_ratio"              : "Swing1 Fib Oranı",
        "swing_1_fib_price"              : "Swing1 Fib Fiyatı",
        "swing_1_dist_pct"               : "Swing1 Uzaklık%",
        "swing_2_window_days"            : "Swing2 Pencere (gün)",
        "swing_2_fib_ratio"              : "Swing2 Fib Oranı",
        "swing_2_fib_price"              : "Swing2 Fib Fiyatı",
        "swing_2_dist_pct"               : "Swing2 Uzaklık%",
        "current_rsi"                    : "Güncel RSI",
        "volume_24h_usdt"                : "24s Hacim (USDT)",
    }

    df_out = pd.DataFrame(validation_results)
    df_out = df_out[[c for c in col_map if c in df_out.columns]]
    df_out = df_out.rename(columns=col_map)

    # Confidence kademesine göre sırala (high → medium → low)
    confidence_rank = {"high": 2, "medium": 1, "low": 0}
    df_out["_rank"] = df_out["Güven"].map(confidence_rank)
    df_out = df_out.sort_values("_rank", ascending=False)
    df_out = df_out.drop(columns=["_rank"]).reset_index(drop=True)

    log_status(f"\n{'═'*65}")
    log_status(f"  🎯  {len(df_out)} adet Confluence sinyali! (detaylar {SIGNALS_JSON} / {LOG_FILE} dosyalarında)\n")

    if EXPORT_CSV:
        fn = "okx_double_bottom_signals.csv"
        df_out.to_csv(fn, index=False, encoding="utf-8-sig")
        log_status(f"💾  CSV → {fn}")

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
        log_status("\n── DEBUG: Confluence Adayı Taraması (90/60/30 gün) ──")
    candidate_t, no_candidate_reason_t = find_confluence_candidates(df_t, verbose=debug)

    log_status("\n── SONUÇ: Confluence Gate (tek strateji) ──")
    if candidate_t is None:
        log_status(f"   ❌  Aday bulunamadı — neden: {no_candidate_reason_t} — izleme listesine de girmiyor.")
        return

    log_status(f"   Swing1: {candidate_t['swing_1_window_days']}g, {candidate_t['swing_1_fib_ratio']} → "
               f"{candidate_t['swing_1_fib_price']} (Δ%{candidate_t['swing_1_dist_pct']})")
    log_status(f"   Swing2: {candidate_t['swing_2_window_days']}g, {candidate_t['swing_2_fib_ratio']} → "
               f"{candidate_t['swing_2_fib_price']} (Δ%{candidate_t['swing_2_dist_pct']})")
    log_status(f"   Toplam mesafe: %{candidate_t['combined_dist_pct']} | within_tolerance={candidate_t['within_tolerance']}")
    log_status(f"   RSI: {candidate_t['current_rsi']} | oversold={candidate_t['rsi_oversold']}")

    confluence_signal_t = evaluate_confluence_entry(df_t, candidate_t)

    if confluence_signal_t is not None:
        log_status(f"\n   ✅  SİNYAL GEÇERLİ — confidence={confluence_signal_t['confidence']}")
        log_status(f"      double_bottom_structure       : {confluence_signal_t['double_bottom_structure']}")
        log_status(f"      double_bottom_fully_confirmed : {confluence_signal_t['double_bottom_fully_confirmed']}")
        log_status(f"      entry_price : {confluence_signal_t['entry_price']}")
        log_status(f"      stop_price  : {confluence_signal_t['stop_price']}")
        log_status(f"      position_size: {confluence_signal_t['position_size']}")
    else:
        reasons = []
        if not candidate_t.get("within_tolerance"):
            reasons.append(f"Confluence tolerans dışı (toplam mesafe %{candidate_t['combined_dist_pct']})")
        if not candidate_t.get("rsi_oversold"):
            reasons.append(f"RSI oversold değil (RSI={candidate_t['current_rsi']})")
        log_status(f"\n   ❌  Gate reddedildi. reasons={reasons}")

        if (not candidate_t["within_tolerance"]
                and candidate_t["combined_dist_pct"] <= WATCHLIST_MAX_COMBINED_DIST_PCT):
            log_status(f"\n   👀  İZLEME LİSTESİNDE — toplam mesafe %{candidate_t['combined_dist_pct']}, "
                       f"tolerans sınırı %{WATCHLIST_MAX_COMBINED_DIST_PCT}")
        else:
            log_status("\n   (İzleme listesine de girmiyor — çok uzak)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="OKX Fibonacci Confluence Tarayıcı (tek strateji, v4.0)"
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
