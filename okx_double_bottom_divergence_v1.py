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

# ── DİĞER ────────────────────────────────────────────────────────────
PAUSE_SEC        = 0.22
EXPORT_CSV       = True


def log_settings():
    log_status("✅  Ayarlar yüklendi — A-B Fibonacci + ABC Uzantısı Tarayıcı v2.0")
    log_status(f"   Zaman Dilimi     : {TIMEFRAME}")
    log_status(f"   Zaman Aralıkları : {', '.join(str(d) + 'g' for d in TIME_WINDOWS_DAYS)} (önce uzun, sonra kısa)")
    log_status(f"   Fibo Seviyeleri  : {FIB_RETRACEMENT} (geri çekilme) / {FIB_EXTENSION} (ABC uzantısı)")
    log_status(f"   Sinyal Toleransı : ±%{SIGNAL_TOLERANCE*100:.0f}")
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
    ══════════════════════════════════════════════════════════════
    """
    close = df["close"]
    rsi   = compute_rsi(close, RSI_PERIOD)

    current_price = close.iloc[-1]
    current_rsi   = rsi.iloc[-1]

    if pd.isna(current_rsi):
        return None

    for window_days in TIME_WINDOWS_DAYS:
        window_candles = window_days * CANDLES_PER_DAY

        if len(df) < window_candles:
            if verbose:
                log_status(f"   [{window_days}g] atlandı → yetersiz veri ({len(df)} < {window_candles} mum)")
            continue

        window = df.iloc[-window_candles:]

        high_time = window["high"].idxmax()
        low_time  = window["low"].idxmin()

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

        if a_type == "dip":   # Yükseliş bacağı (A dip → B zirve)
            fib618_price  = b_price - FIB_RETRACEMENT * distance
            ext1272_price = b_price - FIB_EXTENSION * distance
        else:                 # Düşüş bacağı (A zirve → B dip)
            fib618_price  = b_price + FIB_RETRACEMENT * distance
            ext1272_price = b_price + FIB_EXTENSION * distance

        if fib618_price <= 0 or ext1272_price <= 0:
            continue

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
        }

    return None


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
            "time_windows_days": TIME_WINDOWS_DAYS,
            "fib_retracement": FIB_RETRACEMENT,
            "fib_extension": FIB_EXTENSION,
            "signal_tolerance_pct": SIGNAL_TOLERANCE * 100,
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
            f"Pencere={result['window_days']}g | "
            f"A({result['a_type']})={result['a_price']} | "
            f"B({result['b_type']})={result['b_price']} | "
            f"Fib618={result['fib618_price']} (Δ%{result['dist_to_fib618_pct']:.2f}) | "
            f"Ext1272={result['ext1272_price']} (Δ%{result['dist_to_ext1272_pct']:.2f}) | "
            f"RSI Diverjans=+{result['rsi_divergence']:.2f}"
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

        result = find_ab_fibonacci_signal(df)

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
    }

    df_out = pd.DataFrame(results)
    df_out = df_out[[c for c in col_map if c in df_out.columns]]
    df_out = df_out.rename(columns=col_map)

    # RSI diverjansına göre sırala (en güçlü diverjans önce)
    df_out = df_out.sort_values("RSI Diverjans", ascending=False)
    df_out = df_out.reset_index(drop=True)

    log_status(f"\n{'═'*65}")
    log_status(f"  🎯  {len(df_out)} adet 'A-B Fibonacci + ABC Uzantısı' adayı! (detaylar {SIGNALS_JSON} / {LOG_FILE} dosyalarında)\n")

    if EXPORT_CSV:
        fn = "okx_double_bottom_divergence.csv"
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
