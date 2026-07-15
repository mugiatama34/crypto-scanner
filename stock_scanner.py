# ╔══════════════════════════════════════════════════════════════════════╗
# ║  S&P 500 — FIBONACCI CONFLUENCE + DOUBLE BOTTOM/TOP SPEC TARAYICI    ║
# ║  1 Saatlik | yfinance | v1.0 (2026-07-15)                            ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# Bu dosya, okx_double_bottom_divergence_v1.py (kripto scanner) ile AYNI
# analiz motorunu kullanır - confluence gate, spec gate (Double Bottom/
# Top + Wyckoff/Elliott/zaman simetrisi katmanları) tamamen borsaya özel
# DEĞİLDİR, sadece bir pandas DataFrame (OHLCV) üzerinde çalışır. Bu
# yüzden yeniden yazılmadı, doğrudan import edildi.
#
# BU DOSYAYA ÖZEL (borsaya özel) olan kısımlar:
#   - yfinance ile veri çekme
#   - S&P 500 sembol listesi (Wikipedia'dan, başarısız olursa yedek liste)
#   - ABD piyasa saati kontrolü (hafta içi 09:30-16:00 ET)
#   - Likidite filtresi (ortalama günlük dolar hacmi)
#   - Kendi state dosyası (crypto'nunkiyle KARIŞMAZ)
#
# NOT: yfinance ve Wikipedia'ya bu geliştirme ortamından erişim yoktu,
# bu yüzden veri çekme kodu sadece MANTIK olarak yazıldı ve sentetik
# veriyle test edildi - GERÇEK veri çekimi ancak GitHub Actions'ta
# (tam internet erişimi olan ortamda) doğrulanabilir. İlk birkaç
# çalıştırmada beklenmeyen bir hata çıkarsa normal karşıla, birlikte
# düzeltiriz.
#
# Çalıştırma:
#   python stock_scanner.py
#   python stock_scanner.py --test-symbol AAPL --debug

import argparse
import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from okx_double_bottom_divergence_v1 import (
    find_confluence_candidates,
    evaluate_confluence_entry,
    find_big_wave_abc_pattern,
    evaluate_double_bottom_spec_signal,
    get_btc_trend,
    send_telegram_message,
    format_signal_telegram_message,
    format_spec_signal_telegram_message,
    _json_default,
)

# ── AYARLAR ──────────────────────────────────────────────────────────
TIMEFRAME             = "1h"
LOOKBACK_PERIOD        = os.environ.get("STOCK_LOOKBACK_PERIOD", "60d")
PAUSE_SEC              = float(os.environ.get("STOCK_PAUSE_SEC", 0.3))
MIN_AVG_DOLLAR_VOLUME  = float(os.environ.get("MIN_AVG_DOLLAR_VOLUME", 10_000_000))
TELEGRAM_COOLDOWN_HOURS = float(os.environ.get("TELEGRAM_COOLDOWN_HOURS", 4))
EXPORT_CSV             = True

STOCK_STATE_FILE   = os.environ.get("STOCK_STATE_FILE", "stock_state.json")
LOG_FILE           = os.environ.get("STOCK_SIGNALS_LOG_FILE", "stock_signals.log")
SIGNALS_JSON       = os.environ.get("STOCK_SIGNALS_JSON_FILE", "stock_signals.json")

# ── LOGLAMA (crypto scanner ile ayni format, ayri dosyaya yazar) ──────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("stock_scanner")


def log_status(msg):
    logger.info(msg)


def log_signal(msg):
    logger.info(msg)


# ── ABD PİYASA SAATİ KONTROLÜ ──────────────────────────────────────────
def is_market_open_now():
    """NYSE/NASDAQ şu an açık mı? Hafta içi 09:30-16:00 ET. Resmi tatiller
    KONTROL EDİLMİYOR (bilinçli basitleştirme) - tatil günü çalışırsa
    veri zaten güncellenmemiş olur, script hata vermeden ama anlamsız
    şekilde eski veriyi tekrar tarar. İleride tatil takvimi eklenebilir."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:  # Cumartesi=5, Pazar=6
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close


# ── S&P 500 SEMBOL LİSTESİ ─────────────────────────────────────────────
_FALLBACK_SP500 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK-B", "AVGO", "TSLA",
    "JPM", "LLY", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ",
    "NFLX", "BAC", "CRM", "ABBV", "CVX", "KO", "AMD", "PEP", "WMT", "TMO",
    "ORCL", "MRK", "ADBE", "ACN", "LIN", "MCD", "CSCO", "ABT", "DHR", "INTU",
    "IBM", "TXN", "PM", "GE", "CAT", "VZ", "AMGN", "NOW", "UNP", "QCOM",
]


def fetch_sp500_symbols():
    """Wikipedia'dan güncel S&P 500 listesini çeker. Başarısız olursa
    (ağ, sayfa formatı değişikliği vb.) küçük, sabit bir yedek listeye
    düşer - script ASLA tamamen durmaz.

    NOT: pd.read_html(url) DOĞRUDAN çağrılırsa Wikipedia genelde 403
    Forbidden döner - User-Agent göndermeyen isteklere karşı koruma var.
    Bunun için requests ile tarayıcı gibi bir User-Agent'la sayfa HTML'i
    çekilip pd.read_html'e METİN olarak veriliyor (URL değil)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; stock-scanner/1.0)"}
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=15,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        symbols = tables[0]["Symbol"].astype(str).tolist()
        # yfinance nokta yerine tire ister: BRK.B -> BRK-B
        symbols = [s.strip().replace(".", "-") for s in symbols]
        symbols = sorted(set(symbols))
        log_status(f"📊  S&P 500 listesi Wikipedia'dan çekildi: {len(symbols)} sembol")
        return symbols
    except Exception as exc:
        log_status(f"⚠️  S&P 500 listesi çekilemedi, {len(_FALLBACK_SP500)} sembollük yedek listeye düşülüyor: {exc}")
        return list(_FALLBACK_SP500)


# ── VERİ ÇEKME ──────────────────────────────────────────────────────────
def fetch_stock_ohlcv(symbol, period=LOOKBACK_PERIOD, interval=TIMEFRAME):
    """yfinance'tan OHLCV DataFrame çeker, scanner'ın beklediği kolon
    isimlerine (open/high/low/close/volume, küçük harf) çevirir.
    Veri yoksa/hatalıysa None döner - çağıran taraf bunu 'insufficient_data'
    olarak ele alır, script durmaz."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    # yfinance bazen MultiIndex kolon döner (tek sembol bile olsa) - düzelt.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        return None
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def compute_avg_dollar_volume(df, lookback=20):
    """Son `lookback` barın ortalama (kapanış fiyatı × hacim) çarpımı -
    likidite filtresi için basit bir dolar-hacim tahmini."""
    recent = df.iloc[-lookback:]
    return float((recent["close"] * recent["volume"]).mean())


# ── STATE (crypto scanner'dan TAMAMEN BAĞIMSIZ, ayrı dosya) ───────────
def _load_stock_state():
    empty = {"telegram_notified": {}, "spec_notified": {}}
    if not os.path.exists(STOCK_STATE_FILE):
        return empty
    try:
        with open(STOCK_STATE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {"telegram_notified": raw.get("telegram_notified", {}), "spec_notified": raw.get("spec_notified", {})}
    except Exception as exc:
        log_status(f"⚠️  Önceki hisse state okunamadı, sıfırdan başlanıyor ({exc})")
        return empty


def save_stock_state(telegram_notified_state, spec_notified_state):
    with open(STOCK_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"telegram_notified": telegram_notified_state, "spec_notified": spec_notified_state},
            f, indent=2, ensure_ascii=False,
        )


def should_notify_telegram_stock(symbol, direction, previous_state, cooldown_hours=TELEGRAM_COOLDOWN_HOURS):
    """Kripto scanner'daki should_notify_telegram ile AYNI mantık, ayrı
    (hisse-özel) state üzerinde çalışır."""
    prev = previous_state.get(symbol)
    if prev is None:
        return True
    if prev.get("direction") != direction:
        return True
    try:
        notified_at = datetime.fromisoformat(prev["notified_at"])
    except Exception:
        return True
    elapsed_hours = (datetime.now(timezone.utc) - notified_at).total_seconds() / 3600
    return elapsed_hours >= cooldown_hours


def should_notify_spec_stock(symbol, b_time_iso, previous_state):
    prev = previous_state.get(symbol)
    if prev is None:
        return True
    return prev.get("b_time") != b_time_iso


# ── ANA TARAMA ──────────────────────────────────────────────────────────
def run_stock_scanner():
    log_status("✅  Ayarlar yüklendi — S&P 500 Confluence + Spec Tarayıcı v1.0")
    log_status(f"   Zaman Dilimi     : {TIMEFRAME} (yfinance period={LOOKBACK_PERIOD})")
    log_status(f"   Likidite filtresi: ortalama günlük ≥ ${MIN_AVG_DOLLAR_VOLUME:,.0f}")
    telegram_status = "✅ aktif" if (os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID")) else "❌ yapılandırılmamış"
    log_status(f"   Telegram         : {telegram_status}")

    if not is_market_open_now():
        log_status("⏸️  ABD borsası şu an kapalı (hafta içi 09:30-16:00 ET dışında) — tarama atlanıyor.")
        return None

    symbols = fetch_sp500_symbols()

    # SPY (S&P 500 ETF), BTC'nin kripto tarafındaki rolünün aynısı: genel
    # piyasa trend uyum katmanı için referans.
    market_trend = None
    try:
        spy_df = fetch_stock_ohlcv("SPY")
        if spy_df is not None:
            market_trend = get_btc_trend(spy_df)  # fonksiyon jenerik, isim tarihsel
        log_status(f"📈  SPY (piyasa) trend: {market_trend}")
    except Exception as exc:
        log_status(f"⚠️  SPY trend hesaplanamadı (uyum katmanı atlanacak): {exc}")

    previous_state = _load_stock_state()
    telegram_notified_this_run = dict(previous_state["telegram_notified"])
    spec_notified_this_run = dict(previous_state["spec_notified"])

    validation_results = []
    spec_results = []
    abc_pattern_results = []
    skipped_data = 0
    skipped_liquidity = 0
    skipped_crit = 0

    for symbol in symbols:
        df = fetch_stock_ohlcv(symbol)
        time.sleep(PAUSE_SEC)

        if df is None or len(df) < 50:
            skipped_data += 1
            continue

        avg_dollar_vol = compute_avg_dollar_volume(df)
        if avg_dollar_vol < MIN_AVG_DOLLAR_VOLUME:
            skipped_liquidity += 1
            continue

        # ── Ana Confluence Gate (kripto ile AYNI fonksiyon) ──────────
        candidate, _reason = find_confluence_candidates(df)
        confluence_signal = evaluate_confluence_entry(df, candidate)
        if confluence_signal is not None:
            confluence_signal["symbol"] = symbol
            validation_results.append(confluence_signal)
            log_signal(
                f"✅  {symbol:<8} | confidence={confluence_signal['confidence']:<6} | "
                f"yön={confluence_signal['direction']} | entry={confluence_signal['entry_price']} | "
                f"stop={confluence_signal['stop_price']}"
            )
            if should_notify_telegram_stock(symbol, confluence_signal["direction"], telegram_notified_this_run):
                text = "📈 [HİSSE] " + format_signal_telegram_message(confluence_signal)
                if send_telegram_message(text):
                    log_status(f"📨  Telegram bildirimi gönderildi → {symbol}")
                    telegram_notified_this_run[symbol] = {
                        "direction": confluence_signal["direction"],
                        "notified_at": datetime.now(timezone.utc).isoformat(),
                    }
        else:
            skipped_crit += 1

        # ── Büyük Dalga ABC Deseni (EK/BAĞIMSIZ rapor) ───────────────
        try:
            abc_pattern = find_big_wave_abc_pattern(df)
        except Exception as exc:
            abc_pattern = None
            log_status(f"⚠️  [Büyük Dalga ABC] {symbol} için hata (yok sayıldı): {exc}")
        if abc_pattern is not None:
            abc_pattern["symbol"] = symbol
            abc_pattern_results.append(abc_pattern)

        # ── Spec Raporu (EK/BAĞIMSIZ, kripto ile AYNI fonksiyon) ─────
        try:
            spec_result = evaluate_double_bottom_spec_signal(df, btc_trend=market_trend)
        except Exception as exc:
            spec_result = None
            log_status(f"⚠️  [Spec] {symbol} için hata (yok sayıldı): {exc}")

        if spec_result is not None:
            spec_result["symbol"] = symbol
            spec_results.append(spec_result)

            if spec_result["final_signal"] == "VALID":
                log_signal(
                    f"📐  [Spec] {symbol:<8} | {spec_result['direction']} | confidence={spec_result['confidence']} | "
                    f"entry={spec_result['entry']} stop={spec_result['stop']} r={spec_result['r_multiple']}"
                )
                if should_notify_spec_stock(symbol, spec_result.get("b_time", ""), spec_notified_this_run):
                    spec_text = "📐 [HİSSE SPEC] " + format_spec_signal_telegram_message(symbol, spec_result)
                    if send_telegram_message(spec_text):
                        log_status(f"📨  [Spec] Telegram bildirimi gönderildi → {symbol}")
                        spec_notified_this_run[symbol] = {
                            "b_time": spec_result.get("b_time", ""),
                            "notified_at": datetime.now(timezone.utc).isoformat(),
                        }
            elif spec_result["final_signal"] == "CONFLICT":
                log_signal(f"⚡  [Spec] {symbol:<8} | CONFLICT — hem LONG hem SHORT gate'i geçiyor, sinyal sayılmadı")

    save_stock_state(telegram_notified_this_run, spec_notified_this_run)

    # ── Özet ─────────────────────────────────────────────────────────
    log_status(f"\n{'─'*65}")
    log_status(f"  Veri yetersiz    : {skipped_data}")
    log_status(f"  Düşük likidite   : {skipped_liquidity}")
    log_status(f"  Kriter tutmadı   : {skipped_crit}")
    log_status(f"  ✅  Eşleşen (Confluence): {len(validation_results)}")
    log_status(f"  📐  Spec VALID          : {sum(1 for r in spec_results if r['final_signal']=='VALID')}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_trend_spy": market_trend,
        "summary": {
            "total_symbols": len(symbols),
            "skipped_no_data": skipped_data,
            "skipped_low_liquidity": skipped_liquidity,
            "skipped_gate_rejected": skipped_crit,
            "matched_confluence": len(validation_results),
            "spec_valid": sum(1 for r in spec_results if r["final_signal"] == "VALID"),
        },
        "confluence_signals": validation_results,
        "spec_signals": spec_results,
    }
    with open(SIGNALS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
    log_status(f"💾  Sinyaller → {SIGNALS_JSON}")
    log_status(f"💾  Log       → {LOG_FILE}")

    if EXPORT_CSV and validation_results:
        pd.DataFrame(validation_results).to_csv("stock_confluence_signals.csv", index=False, encoding="utf-8-sig")
        log_status("💾  CSV → stock_confluence_signals.csv")
    if EXPORT_CSV and abc_pattern_results:
        pd.DataFrame(abc_pattern_results).to_csv("stock_big_wave_abc.csv", index=False, encoding="utf-8-sig")
        log_status("💾  CSV → stock_big_wave_abc.csv")
    if EXPORT_CSV and spec_results:
        spec_rows = [{
            "symbol": r["symbol"], "final_signal": r["final_signal"],
            "direction": r.get("direction") or r.get("direction_checked"),
            "confidence": r.get("confidence"), "entry": r.get("entry"), "stop": r.get("stop"),
        } for r in spec_results]
        pd.DataFrame(spec_rows).to_csv("stock_double_bottom_spec.csv", index=False, encoding="utf-8-sig")
        log_status("💾  CSV → stock_double_bottom_spec.csv")

    return validation_results


def debug_test_symbol(symbol):
    log_status(f"\n{'='*65}\n  DEBUG: {symbol}\n{'='*65}")
    df = fetch_stock_ohlcv(symbol)
    if df is None:
        log_status("❌  Veri çekilemedi.")
        return
    log_status(f"   {len(df)} bar çekildi, son kapanış: {df['close'].iloc[-1]}")

    candidate, reason = find_confluence_candidates(df, verbose=True)
    log_status(f"\n── Confluence Gate ──")
    if candidate is None:
        log_status(f"   ❌  Aday bulunamadı — neden: {reason}")
    else:
        confluence_signal = evaluate_confluence_entry(df, candidate)
        if confluence_signal:
            log_status(f"   ✅  SİNYAL — {confluence_signal}")
        else:
            log_status(f"   ❌  Gate reddedildi (within_tolerance={candidate['within_tolerance']}, "
                       f"rsi_oversold={candidate['rsi_oversold']}, rsi_overbought={candidate['rsi_overbought']})")

    spec_result = evaluate_double_bottom_spec_signal(df)
    log_status(f"\n── Spec Raporu ──\n   {spec_result['final_signal']}")


def parse_args():
    parser = argparse.ArgumentParser(description="S&P 500 Confluence + Spec Tarayıcı")
    parser.add_argument("--test-symbol", default=os.environ.get("TEST_SYMBOL"))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.test_symbol:
        debug_test_symbol(args.test_symbol)
    else:
        run_stock_scanner()
