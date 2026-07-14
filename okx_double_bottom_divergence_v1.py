# ╔══════════════════════════════════════════════════════════════════════╗
# ║  OKX — FIBONACCI CONFLUENCE TARAYICI (tek strateji)                 ║
# ║  4 Saatlik | v4.1 (2026-07-10: pivot-tabanlı swing tespiti)          ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# GATE (find_confluence_candidates + evaluate_confluence_entry, ikisi de zorunlu):
#   1) İki BAĞIMSIZ swing'in (find_zigzag_pivots ile fiyat yapısından
#      bulunan gerçek dönüş noktalarından, TAKVIM PENCERESİNDEN DEĞİL)
#      retracement/extension seviyeleri (0.618/0.786/1.272/1.618) güncel
#      fiyatta AYRI AYRI ±%3 tolerans içinde ÇAPRAZ çakışıyor (bir swing
#      retracement, diğeri extension - Boroden confluence)
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
# NOT (v4.0→v4.1): İlk confluence sürümü 90/60/30 günlük takvim
# pencereleri kullanıyordu. Canlı veri, bu pencerelerin çoğu zaman AYNI
# güncel ekstremumu paylaştığını (gerçek bağımsızlık yok) gösterdi -
# reddedilenlerin %55'i bu yüzdendi. v4.1, swing tespitini fiyatın kendi
# yerel dönüş noktalarına (ZigZag pivot) taşıyarak bunu kaynağından
# çözdü.
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
import requests
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

from signal_validation import (
    evaluate_signal, evaluate_signal_short,
    check_double_bottom, check_double_top, compute_stop_and_size,
    check_rsi_divergence, check_rsi_negative_divergence, check_fibonacci_zone,
    check_wyckoff_volume, check_wyckoff_volume_short,
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


TELEGRAM_COOLDOWN_HOURS = float(os.environ.get("TELEGRAM_COOLDOWN_HOURS", 4))


def _load_state_file():
    """STATE_FILE'i butun olarak yukler: {'watchlist': {...}, 'telegram_notified': {...},
    'spec_notified': {...}}. Dosya yoksa/bozuksa/eski formattaysa (duz
    watchlist dict'i) bos/uyumlu bir yapiya geri doner - sistem kendi
    kendini onarir, hata firlatmaz."""
    empty = {"watchlist": {}, "telegram_notified": {}, "spec_notified": {}}
    if not os.path.exists(STATE_FILE):
        return empty
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if any(k in raw for k in empty):
            return {
                "watchlist": raw.get("watchlist", {}),
                "telegram_notified": raw.get("telegram_notified", {}),
                "spec_notified": raw.get("spec_notified", {}),
            }
        # Eski format (2026-07-10 oncesi): dosyanin tamami duz watchlist dict'iydi.
        return {"watchlist": raw, "telegram_notified": {}, "spec_notified": {}}
    except Exception as exc:
        log_status(f"⚠️  Onceki state okunamadi, sifirdan baslaniyor ({exc})")
        return empty


def load_previous_watchlist_state():
    """Onceki tarama turunun watchlist snapshot'ini yukler (GitHub Actions
    cache'inden gelir)."""
    return _load_state_file()["watchlist"]


def load_previous_telegram_state():
    """Onceki tarama turunda hangi coin/yon icin ne zaman Telegram bildirimi
    gonderildigini yukler (cooldown kontrolu icin)."""
    return _load_state_file()["telegram_notified"]


def load_previous_spec_notified_state():
    """Spec raporu (2026-07-14) icin: hangi coin+B_zamani kombinasyonu
    daha once VALID sinyal olarak bildirildi. Sembol+B_zamani ayni
    kaldigi surece TEKRAR bildirim gonderilmez (spec madde 10)."""
    return _load_state_file()["spec_notified"]


def should_notify_spec(symbol, b_time_iso, previous_spec_state):
    """Ayni sembol + ayni B_zamani kombinasyonu daha once bildirildiyse
    tekrar gondermez. B_zamani DEGISTIYSE (yeni bir Double Bottom/Top
    olustu demektir) bu YENI bilgi sayilir, hemen bildirir."""
    prev = previous_spec_state.get(symbol)
    if prev is None:
        return True
    return prev.get("b_time") != b_time_iso


def should_notify_telegram(symbol, direction, previous_telegram_state, cooldown_hours=TELEGRAM_COOLDOWN_HOURS):
    """Ayni coin+yon icin son TELEGRAM_COOLDOWN_HOURS saat icinde bildirim
    gonderildiyse TEKRAR gondermez (spam onleme). Yon degistiyse (ornegin
    short'tan long'a) bu YENI bilgi sayilir, cooldown'u atlar - hemen bildirir."""
    prev = previous_telegram_state.get(symbol)
    if prev is None:
        return True
    if prev.get("direction") != direction:
        return True  # yon degisti, yeni bilgi - cooldown gecerli degil
    try:
        notified_at = datetime.fromisoformat(prev["notified_at"])
    except Exception:
        return True  # bozuk kayit - guvenli tarafta kal, bildir
    elapsed_hours = (datetime.now(timezone.utc) - notified_at).total_seconds() / 3600
    return elapsed_hours >= cooldown_hours


def save_watchlist_state(watchlist_results, telegram_notified_state=None, spec_notified_state=None):
    """Bu turun watchlist sonuclarini VE telegram/spec bildirim gecmisini
    bir sonraki tur icin kaydeder (GitHub Actions cache adimi bunu tasir).
    reference_id, hangi 2 swing'in karsilastirildigini parmak izi olarak
    tutar - boylece bir sonraki tur, ayni swing ciftiyle mi kiyaslandigini
    yoksa referansin degisip degismedigini ayirt edebilir."""
    watchlist_state = {
        entry["symbol"]: {
            "distance_pct": entry.get("combined_dist_pct"),
            "reference_id": entry.get("reference_id"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        for entry in watchlist_results
    }
    full_state = {
        "watchlist": watchlist_state,
        "telegram_notified": telegram_notified_state or {},
        "spec_notified": spec_notified_state or {},
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(full_state, f, indent=2, ensure_ascii=False)


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



# ── TELEGRAM BİLDİRİMLERİ ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID")


def send_telegram_message(text):
    """Telegram Bot API'ye HTML formatlı mesaj gönderir. Kimlik bilgisi
    eksikse veya ağ hatası olursa SESSİZCE atlar (uyarı loglar) - bildirim
    hatası ana taramayı ASLA durdurmamalı."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            log_status(f"⚠️  Telegram gönderim hatası ({resp.status_code}): {resp.text[:200]}")
            return False
        return True
    except Exception as exc:
        log_status(f"⚠️  Telegram gönderim hatası (ağ): {exc}")
        return False


def _fmt_time_short(iso_str):
    """ISO zaman damgasini 'DD/MM SS:DD' gibi kisa, okunakli bir bicime cevirir
    (TradingView'de mumu bulmak icin)."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d/%m %H:%M")
    except Exception:
        return iso_str


def format_signal_telegram_message(signal):
    """Confluence sinyalini Telegram icin okunakli HTML mesajina cevirir.
    A/B noktalarinin fiyat+zamanini da icerir - boylece TradingView'de
    elle Fibonacci cizip dogrulama yapilabilir (sadece 'bacakX' numarasi
    grafikte hicbir seye karsilik gelmez, iste bu yuzden eklendi)."""
    direction_label = "🟢 LONG" if signal["direction"] == "long" else "🔴 SHORT"
    confidence_emoji = {"low": "🟡", "medium": "🟠", "high": "🟢"}.get(signal["confidence"], "⚪")

    s1_a_t = _fmt_time_short(signal["swing_1_a_time"])
    s1_b_t = _fmt_time_short(signal["swing_1_b_time"])
    s2_a_t = _fmt_time_short(signal["swing_2_a_time"])
    s2_b_t = _fmt_time_short(signal["swing_2_b_time"])

    return (
        f"{direction_label} — <b>{signal['symbol']}</b>\n"
        f"Güven: {confidence_emoji} {signal['confidence'].upper()}\n"
        f"\n"
        f"Giriş: <code>{signal['entry_price']}</code>\n"
        f"Stop: <code>{signal['stop_price']}</code>\n"
        f"Pozisyon: <code>{signal['position_size']}</code>\n"
        f"\n"
        f"<b>Swing1</b> ({signal['swing_1_fib_ratio']} → {signal['swing_1_fib_price']}, Δ%{signal['swing_1_dist_pct']}):\n"
        f"  A ({signal['swing_1_a_type']}): {signal['swing_1_a_price']} @ {s1_a_t}\n"
        f"  B ({signal['swing_1_b_type']}): {signal['swing_1_b_price']} @ {s1_b_t}\n"
        f"<b>Swing2</b> ({signal['swing_2_fib_ratio']} → {signal['swing_2_fib_price']}, Δ%{signal['swing_2_dist_pct']}):\n"
        f"  A ({signal['swing_2_a_type']}): {signal['swing_2_a_price']} @ {s2_a_t}\n"
        f"  B ({signal['swing_2_b_type']}): {signal['swing_2_b_price']} @ {s2_b_t}\n"
        f"\n"
        f"RSI: {signal['current_rsi']}\n"
        f"Yapı var mı: {signal['structure_present']} | Tam teyitli: {signal['fully_confirmed']}"
    )



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
# NOT: Takvim pencereleri (90/60/30 gün) 2026-07-10'da kaldırıldı - swing
# tespiti artık ZIGZAG_PCT_THRESHOLD ile fiyat yapısından yapılıyor
# (bkz. find_zigzag_pivots). Gerekçe: iç içe pencereler çoğu zaman aynı
# güncel ekstremumu paylaşıyor, "bağımsız" swing üretmiyordu.

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
# 2026-07-10: SHORT tarafi eklendi. RSI_OVERBOUGHT, long'un (oversold<35)
# aynasi - fiyat yukaridan bir confluence direncine yaklasirken RSI>65 ise
# short kurulumu olarak degerlendirilir.
RSI_OVERBOUGHT_THRESHOLD = 65
# Izleme listesi: gate'i (within_tolerance) tam gecemeyen ama toplam mesafesi
# bu esigin altinda olan confluence adaylari "yaklasiyor/uzaklasiyor" olarak
# raporlanir. FIB_CONFLUENCE_TOLERANCE'in (%3+%3=%6 max gate) uzerinde,
# gurultuyu elemek icin makul bir tavan.
WATCHLIST_MAX_COMBINED_DIST_PCT = float(os.environ.get("WATCHLIST_MAX_COMBINED_DIST_PCT", 16.0))

# ── PIVOT TESPITI (2026-07-10, 2. pivot: takvim pencereleri yerine
#    gercek fiyat yapisindan swing tespiti) ────────────────────────────
# Fiyat, mevcut ekstremden bu yuzde kadar ters yone hareket edince bir
# pivot (donus noktasi) onaylanir. 4H kripto icin %5 makul bir baslangic -
# cok kucuk (%1-2) gurultuyu pivot sanar, cok buyuk (%15+) neredeyse hic
# pivot bulamaz.
ZIGZAG_PCT_THRESHOLD = float(os.environ.get("ZIGZAG_PCT_THRESHOLD", 0.05))
# 2026-07-13: cok kisa sureli (%esigi gecen ama sadece birkac mumde olan)
# sicramalarin "gecerli dalga" sayilmasini onlemek icin minimum bacak
# suresi. 8 bar = 32 saat (4H mumda) - bunun altindaki hareketler gurultu
# sayilip pivot listesine hic girmiyor (find_zigzag_pivots icinde "swallow"
# edilip bir sonraki gercek pivotla birlestiriliyor).
MIN_LEG_DURATION_BARS = int(os.environ.get("MIN_LEG_DURATION_BARS", 8))

# ── DİĞER ────────────────────────────────────────────────────────────
PAUSE_SEC        = 0.22
EXPORT_CSV       = True
MIN_CANDLES      = 40      # En kısa pencere (30g) + RSI ısınma dönemi için asgari veri

# ── LİKİDİTE / STABLECOIN FİLTRELERİ ─────────────────────────────────
MIN_24H_VOLUME_USDT = float(os.environ.get("MIN_24H_VOLUME_USDT", 1_000_000))

# ── DOUBLE BOTTOM/TOP SPEC RAPORU (2026-07-14, TAMAMEN AYRI 3. rapor) ──
# Kullanicinin Proje'den aldigi detayli spec'e gore: Double Bottom/Top
# ana gate, RSI diverjansi + Fibonacci bandi sarti, Wyckoff/confluence/
# zaman simetrisi/BTC uyumu confidence katmanlari, Elliott carpani,
# DUSUK/ORTA/YUKSEK siniflandirma. Ana confluence gate'ini (find_
# confluence_candidates) HICBIR SEKILDE etkilemez - ayri CSV + ayri
# Telegram bildirimi (okx_double_bottom_spec.csv).
# NOT: Kullanici Fib toleransini spec'in ±%0.5'i yerine mevcut ±%3'te
# tutmayi tercih etti (2026-07-14 karari).
SPEC_MIN_BAR_GAP        = 10
SPEC_LEVEL_TOLERANCE    = 0.03
SPEC_FIB_ZONE_TOLERANCE = FIB_CONFLUENCE_TOLERANCE  # ±%3, kullanicinin tercihi
SPEC_CONFLUENCE_TOLERANCE = 0.025  # confidence katmani icin ±%2.5 (spec: %2-3 araligi)
SPEC_TIME_SYMMETRY_BAR_TOLERANCE = 2

STABLECOIN_BASES = {
    "USDC", "USDG", "USAT", "DAI", "TUSD", "FDUSD", "USDD", "USDP", "PYUSD",
    "GUSD", "USDK", "LUSD", "SUSD", "USTC", "USDE", "EUR", "EURT",
}


def is_stablecoin_pair(symbol):
    """'USDC/USDT' gibi iki tarafı da sabit fiyatlı parite mi kontrol eder."""
    base = symbol.split("/")[0].upper()
    return base in STABLECOIN_BASES


def log_settings():
    log_status("✅  Ayarlar yüklendi — OKX Confluence Tarayıcı v4.1 (pivot-tabanlı) (Double Bottom = bonus katmanı)")
    log_status(f"   Zaman Dilimi     : {TIMEFRAME}")
    log_status(f"   Pivot Tespiti    : ZigZag ±%{ZIGZAG_PCT_THRESHOLD*100:.0f} (takvim penceresi değil, fiyat yapısından)")
    log_status(f"   Confluence       : Fib seviyeleri={FIB_CONFLUENCE_LEVELS}, tolerans=±%{FIB_CONFLUENCE_TOLERANCE*100:.0f}, RSI oversold<{RSI_OVERSOLD_THRESHOLD} (long) / overbought>{RSI_OVERBOUGHT_THRESHOLD} (short)")
    log_status(f"   İzleme sınırı    : toplam mesafe ≤ %{WATCHLIST_MAX_COMBINED_DIST_PCT:.0f}")
    telegram_status = "✅ aktif" if (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID) else "❌ yapılandırılmamış"
    log_status(f"   Telegram         : {telegram_status}")


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


def find_zigzag_pivots(df, pct_threshold=ZIGZAG_PCT_THRESHOLD, min_leg_bars=MIN_LEG_DURATION_BARS):
    """
    Fiyat serisinden GERÇEK yerel dönüş noktalarını (pivot high/low) bulur.
    Klasik "ZigZag" mantığı: mevcut yönde yeni bir ekstrem oluştukça takip
    eder, fiyat ekstremden `pct_threshold` kadar ters yöne hareket edince
    bir pivot onaylanır ve yön değişir. Bu, takvim pencereleri (90/60/30
    gün) yerine fiyatın kendi yapısından gelen, doğası gereği birbirinden
    AYRIŞAN (farklı A/B noktalarına sahip) swing adayları üretir.

    2026-07-13: Ham zigzag TAMAMLANDIKTAN SONRA, `_merge_short_legs` ile
    ardışık pivotlar arasındaki süresi `min_leg_bars`'ın altında kalan
    "bacaklar" (örn. güçlü bir trend içindeki birkaç saatlik ani bir
    whipsaw) ÇİFT HALİNDE elenir - bu, dip/zirve alternansını HER ZAMAN
    korur (tek pivot silmek yerine çift silindiği için sıra bozulmaz).

    Dönüş: kronolojik sırayla pivot noktaları listesi:
      [{"time": Timestamp, "price": float, "type": "dip"|"zirve"}, ...]
    """
    if len(df) < 3:
        return []

    highs = df["high"].values
    lows = df["low"].values
    times = df.index

    pivots = []
    trend = None  # henuz belirlenmedi
    extreme_idx = 0
    extreme_price = df["close"].iloc[0]

    for i in range(1, len(df)):
        high_i, low_i = highs[i], lows[i]

        if trend is None:
            if high_i >= extreme_price * (1 + pct_threshold):
                trend = "up"
                extreme_idx, extreme_price = i, high_i
            elif low_i <= extreme_price * (1 - pct_threshold):
                trend = "down"
                extreme_idx, extreme_price = i, low_i
            continue

        if trend == "up":
            if high_i > extreme_price:
                extreme_idx, extreme_price = i, high_i
            elif low_i <= extreme_price * (1 - pct_threshold):
                pivots.append({"time": times[extreme_idx], "price": float(extreme_price), "type": "zirve"})
                trend = "down"
                extreme_idx, extreme_price = i, low_i
        else:  # trend == "down"
            if low_i < extreme_price:
                extreme_idx, extreme_price = i, low_i
            elif high_i >= extreme_price * (1 + pct_threshold):
                pivots.append({"time": times[extreme_idx], "price": float(extreme_price), "type": "dip"})
                trend = "up"
                extreme_idx, extreme_price = i, high_i

    # Su an olusmakta olan (henuz reversal ile onaylanmamis) son ekstrem de
    # eklenir - en guncel, "canli" swing ucu olarak kullanilabilir.
    if trend is not None:
        pivots.append({
            "time": times[extreme_idx], "price": float(extreme_price),
            "type": "zirve" if trend == "up" else "dip",
        })

    if pivots:
        # Baslangic ankraji: ilk onaylanan pivotun ZIT tipinde, serinin ilk
        # barindan. Bu, ilk swing'in (baslangic -> ilk pivot) de bir aday
        # olarak degerlendirilmesini saglar.
        first_type = "dip" if pivots[0]["type"] == "zirve" else "zirve"
        pivots.insert(0, {"time": times[0], "price": float(df["close"].iloc[0]), "type": first_type})

    pivots = _merge_short_legs(df, pivots, min_leg_bars)
    return pivots


def _merge_short_legs(df, pivots, min_leg_bars):
    """Ardisik pivot ciftleri arasindaki bar mesafesi min_leg_bars'in
    altindaysa, o cifti (HER ZAMAN IKISINI BIRDEN) listeden cikarir.
    Alternans (dip/zirve/dip/zirve...) boylece daima korunur - tek pivot
    silmek alternansi bozardi, cift silmek bozmaz. Kalan komsu pivotlar
    yeniden birbirine cok yakin hale gelebileceginden, degisiklik
    kalmayana kadar tekrar taranir."""
    if min_leg_bars <= 0 or len(pivots) < 3:
        return pivots

    pivots = list(pivots)
    time_to_idx = {t: i for i, t in enumerate(df.index)}

    changed = True
    while changed and len(pivots) >= 3:
        changed = False
        for i in range(len(pivots) - 1):
            idx1 = time_to_idx.get(pivots[i]["time"])
            idx2 = time_to_idx.get(pivots[i + 1]["time"])
            if idx1 is None or idx2 is None:
                continue
            if (idx2 - idx1) < min_leg_bars:
                del pivots[i:i + 2]
                changed = True
                break

    return pivots




def find_big_wave_abc_pattern(df, max_report_dist_pct=20.0):
    """
    ══════════════════════════════════════════════════════════════
    EK / BAĞIMSIZ RAPOR — find_confluence_candidates'tan TAMAMEN AYRI,
    onu hiçbir şekilde etkilemez, ana gate'e girmez. Sadece ayrı bir
    CSV'ye (okx_big_wave_abc.csv) yazılır.

    DESEN: "Büyük dalganın retracement'i" + "ondan SONRA gelen küçük,
    bağımsız bir ABC düzeltmesinin extension'ı" (C henüz tamamlanmamış
    olabilir - bu yüzden extension bir PROJEKSİYON/hedef olarak
    kullanılıyor, kesinleşmiş bir nokta değil).

      1) Büyük dalga = zigzag pivotları arasındaki TÜM bacaklar içinde
         fiyat aralığı (|B-A|) EN BÜYÜK olan tek bacak.
      2) Bu büyük dalganın retracement seviyeleri (0.618/0.786) hesaplanır.
      3) Büyük dalganın bitişinden SONRA gelen, onunla ortak nokta
         paylaşmayan bacaklar arasından EN GÜNCEL (kronolojik son) olanı
         "küçük ABC bacağı" olarak seçilir - bu, düzeltmenin henüz
         tamamlanmamış "C" ayağını temsil eder. Bu bacağın extension
         seviyeleri (1.272/1.618) hesaplanır - yani "C nereye gidebilir"
         projeksiyonu.
      4) Büyük dalga retracement'i ile küçük bacak extension'ı arasından,
         GÜNCEL FİYATA toplam mesafesi en küçük kombinasyon seçilir.

    Donus: None (yeterli bacak yok VEYA en iyi eslesme bile cok uzak)
    VEYA detay dict.
    ══════════════════════════════════════════════════════════════
    """
    pivots = find_zigzag_pivots(df)
    if len(pivots) < 4:
        return None

    current_price = df["close"].iloc[-1]

    legs = []
    for k in range(len(pivots) - 1):
        p_a, p_b = pivots[k], pivots[k + 1]
        if abs(p_b["price"] - p_a["price"]) <= 0:
            continue
        legs.append({
            "leg_id": k,
            "a_type": p_a["type"], "a_price": p_a["price"], "a_time": p_a["time"],
            "b_type": p_b["type"], "b_price": p_b["price"], "b_time": p_b["time"],
            "magnitude": abs(p_b["price"] - p_a["price"]),
        })
    if not legs:
        return None

    big_wave = max(legs, key=lambda l: l["magnitude"])
    big_retracement = compute_fib_levels(big_wave["a_price"], big_wave["b_price"], big_wave["a_type"], RETRACEMENT_RATIOS)

    candidate_legs = [
        l for l in legs
        if l["leg_id"] != big_wave["leg_id"]
        and l["a_time"] > big_wave["b_time"]
        and l["a_time"] not in (big_wave["a_time"], big_wave["b_time"])
        and l["b_time"] not in (big_wave["a_time"], big_wave["b_time"])
    ]
    if not candidate_legs:
        return None

    small_leg = max(candidate_legs, key=lambda l: l["leg_id"])
    small_extension = compute_fib_levels(small_leg["a_price"], small_leg["b_price"], small_leg["a_type"], EXTENSION_RATIOS)

    best = None
    for ratio_big, level_big in big_retracement.items():
        if level_big <= 0:
            continue
        dist_big = abs(current_price - level_big) / level_big
        for ratio_small, level_small in small_extension.items():
            if level_small <= 0:
                continue
            dist_small = abs(current_price - level_small) / level_small
            combined = dist_big + dist_small
            if best is None or combined < best["combined_dist"]:
                best = {
                    "combined_dist": combined,
                    "ratio_big": ratio_big, "level_big": level_big, "dist_big": dist_big,
                    "ratio_small": ratio_small, "level_small": level_small, "dist_small": dist_small,
                }

    if best is None or best["combined_dist"] * 100 > max_report_dist_pct:
        return None

    within_tolerance = (
        best["dist_big"] <= FIB_CONFLUENCE_TOLERANCE and best["dist_small"] <= FIB_CONFLUENCE_TOLERANCE
    )

    return {
        "current_price": round(float(current_price), 6),
        "big_wave_a_type": big_wave["a_type"], "big_wave_a_price": round(float(big_wave["a_price"]), 6),
        "big_wave_a_time": big_wave["a_time"].isoformat(),
        "big_wave_b_type": big_wave["b_type"], "big_wave_b_price": round(float(big_wave["b_price"]), 6),
        "big_wave_b_time": big_wave["b_time"].isoformat(),
        "big_wave_retracement_ratio": best["ratio_big"],
        "big_wave_retracement_price": round(float(best["level_big"]), 6),
        "big_wave_dist_pct": round(float(best["dist_big"]) * 100, 3),
        "small_leg_a_type": small_leg["a_type"], "small_leg_a_price": round(float(small_leg["a_price"]), 6),
        "small_leg_a_time": small_leg["a_time"].isoformat(),
        "small_leg_b_type": small_leg["b_type"], "small_leg_b_price": round(float(small_leg["b_price"]), 6),
        "small_leg_b_time": small_leg["b_time"].isoformat(),
        "small_leg_c_extension_ratio": best["ratio_small"],
        "small_leg_c_extension_price": round(float(best["level_small"]), 6),
        "small_leg_dist_pct": round(float(best["dist_small"]) * 100, 3),
        "combined_dist_pct": round(float(best["combined_dist"]) * 100, 3),
        "within_tolerance": bool(within_tolerance),
    }


def find_confluence_candidates(df, watch_tolerance=0.08, verbose=False):
    """
    ══════════════════════════════════════════════════════════════
    CONFLUENCE ADAYI TESPITI — ANA GATE'in TEK kaynağı.

    2026-07-13 (3. revizyon): Genel "herhangi bağımsız iki bacağın çapraz
    eşleşmesi" yaklaşımından, SPESİFİK bir desene kilitlendi - kullanıcının
    tarif ettiği ve TA literatüründeki popüler/klasik confluence şekli:

      1) find_zigzag_pivots ile fiyat serisinin gerçek pivot noktaları
         bulunur; ardışık her pivot çifti bir "bacak" oluşturur.
      2) BÜYÜK DALGA = tüm bacaklar içinde fiyat aralığı (|B-A|) EN BÜYÜK
         olan TEK bacak. SADECE retracement seviyeleri (0.618/0.786)
         hesaplanır - büyük dalga hiçbir zaman extension rolünde olmaz.
      3) KÜÇÜK ABC BACAĞI = büyük dalganın bitişinden SONRA gelen, onunla
         ortak nokta paylaşmayan bacaklar arasından EN GÜNCEL (kronolojik
         son) olanı. SADECE extension seviyeleri (1.272/1.618) hesaplanır -
         henüz tamamlanmamış "C" ayağının projeksiyonu olarak yorumlanır.
      4) Büyük dalganın retracement'i ile küçük bacağın extension'ı
         arasından, GÜNCEL FİYATA toplam mesafesi en küçük kombinasyon
         seçilir.

    NOT: Bu artık find_big_wave_abc_pattern (okx_big_wave_abc.csv) ile
    AYNI temel mantığı kullanıyor - tek fark, bu fonksiyon RSI/watchlist/
    gate entegrasyonu için gerekli ek alanları (leg_id, duration_bars,
    rsi_oversold/overbought) da hesaplayıp döndürüyor.

    RSI aşırı satim/alim SARTI BURADA UYGULANMAZ - current_rsi ve
    rsi_oversold/rsi_overbought bilgisi sonuca eklenir; GATE kararını
    veren üst katman (evaluate_confluence_entry) bunu kullanır.

    Donus: (None, reason_kodu) VEYA (best-match dict, None).
    ══════════════════════════════════════════════════════════════
    """
    close = df["close"]
    rsi   = compute_rsi(close, RSI_PERIOD)

    current_price = close.iloc[-1]
    current_rsi   = rsi.iloc[-1]

    if pd.isna(current_rsi):
        return None, "rsi_hesaplanamiyor"

    pivots = find_zigzag_pivots(df)
    if verbose:
        for p in pivots:
            log_status(f"   [Pivot] {p['time']} {p['type']:<6} {p['price']:.6f}")

    legs = []
    for k in range(len(pivots) - 1):
        p_a, p_b = pivots[k], pivots[k + 1]
        a_time, a_price, a_type = p_a["time"], p_a["price"], p_a["type"]
        b_time, b_price, b_type = p_b["time"], p_b["price"], p_b["type"]

        if abs(b_price - a_price) <= 0:
            continue

        retracement_levels = compute_fib_levels(a_price, b_price, a_type, RETRACEMENT_RATIOS)
        extension_levels = compute_fib_levels(a_price, b_price, a_type, EXTENSION_RATIOS)
        if any(lv <= 0 for lv in retracement_levels.values()) or any(lv <= 0 for lv in extension_levels.values()):
            continue

        duration_bars = df.index.get_loc(b_time) - df.index.get_loc(a_time)

        legs.append({
            "leg_id": k,
            "duration_bars": int(duration_bars),
            "a_type": a_type, "a_price": a_price, "a_time": a_time,
            "b_type": b_type, "b_price": b_price, "b_time": b_time,
            "magnitude": abs(b_price - a_price),
            "retracement_levels": retracement_levels,
            "extension_levels": extension_levels,
        })

        if verbose:
            log_status(f"   [Confluence] bacak={k} A({a_type})={a_price:.6f} B({b_type})={b_price:.6f} ({duration_bars} bar)")

    if len(legs) < 2:
        if verbose:
            log_status("   [Confluence] atlandı → en az 2 pivot bacağı bulunamadı (fiyat hareketi yetersiz)")
        return None, "yetersiz_pivot"

    big_wave = max(legs, key=lambda l: l["magnitude"])

    candidate_legs = [
        l for l in legs
        if l["leg_id"] != big_wave["leg_id"]
        and l["a_time"] > big_wave["b_time"]
        and l["a_time"] not in (big_wave["a_time"], big_wave["b_time"])
        and l["b_time"] not in (big_wave["a_time"], big_wave["b_time"])
    ]
    if not candidate_legs:
        if verbose:
            log_status("   [Confluence] atlandı → büyük dalgadan sonra bağımsız bir küçük ABC bacağı yok")
        return None, "kucuk_abc_bacagi_yok"

    small_leg = max(candidate_legs, key=lambda l: l["leg_id"])

    best_match = None
    for ratio_big, level_big in big_wave["retracement_levels"].items():
        dist_big = abs(current_price - level_big) / level_big
        if dist_big > watch_tolerance * 2:
            continue
        for ratio_small, level_small in small_leg["extension_levels"].items():
            dist_small = abs(current_price - level_small) / level_small
            if dist_small > watch_tolerance * 2:
                continue
            combined_dist = dist_big + dist_small
            if best_match is None or combined_dist < best_match["combined_dist"]:
                best_match = {
                    "combined_dist": combined_dist,
                    "ratio_big": ratio_big, "level_big": level_big, "dist_big": dist_big,
                    "ratio_small": ratio_small, "level_small": level_small, "dist_small": dist_small,
                }

    if best_match is None:
        if verbose:
            log_status("   [Confluence] atlandı → büyük dalga retracement'i ile küçük bacak extension'ı makul mesafede değil")
        return None, "cok_uzak"

    within_tolerance = (
        best_match["dist_big"] <= FIB_CONFLUENCE_TOLERANCE
        and best_match["dist_small"] <= FIB_CONFLUENCE_TOLERANCE
    )

    return {
        "current_price"         : round(float(current_price), 6),
        "current_rsi"           : round(float(current_rsi), 2),
        "rsi_oversold"          : bool(current_rsi < RSI_OVERSOLD_THRESHOLD),
        "rsi_overbought"        : bool(current_rsi > RSI_OVERBOUGHT_THRESHOLD),
        "swing_1_leg_id"        : big_wave["leg_id"],
        "swing_1_duration_bars" : big_wave["duration_bars"],
        "swing_1_a_type"        : big_wave["a_type"],
        "swing_1_a_price"       : round(float(big_wave["a_price"]), 6),
        "swing_1_a_time"        : big_wave["a_time"].isoformat(),
        "swing_1_b_type"        : big_wave["b_type"],
        "swing_1_b_price"       : round(float(big_wave["b_price"]), 6),
        "swing_1_b_time"        : big_wave["b_time"].isoformat(),
        "swing_1_fib_ratio"     : best_match["ratio_big"],
        "swing_1_fib_price"     : round(float(best_match["level_big"]), 6),
        "swing_1_dist_pct"      : round(float(best_match["dist_big"]) * 100, 3),
        "swing_2_leg_id"        : small_leg["leg_id"],
        "swing_2_duration_bars" : small_leg["duration_bars"],
        "swing_2_a_type"        : small_leg["a_type"],
        "swing_2_a_price"       : round(float(small_leg["a_price"]), 6),
        "swing_2_a_time"        : small_leg["a_time"].isoformat(),
        "swing_2_b_type"        : small_leg["b_type"],
        "swing_2_b_price"       : round(float(small_leg["b_price"]), 6),
        "swing_2_b_time"        : small_leg["b_time"].isoformat(),
        "swing_2_fib_ratio"     : best_match["ratio_small"],
        "swing_2_fib_price"     : round(float(best_match["level_small"]), 6),
        "swing_2_dist_pct"      : round(float(best_match["dist_small"]) * 100, 3),
        "combined_dist_pct"     : round(float(best_match["combined_dist"]) * 100, 3),
        "within_tolerance"      : bool(within_tolerance),
        "candle_total"          : len(df),
    }, None


def evaluate_confluence_entry(df, candidate, equity=1000, risk_pct=0.015):
    """
    ══════════════════════════════════════════════════════════════
    ANA GATE (2026-07-10 pivot + short ekleme). Confluence birincil
    strateji; Double Bottom/Top bonus/confidence katmani.

    YÖN (2026-07-10, 2. ekleme): Confluence seviye tespiti yön bağımsız
    (aynı matematik). Yönü belirleyen RSI:
      candidate["rsi_oversold"]   (RSI<35) → LONG
      candidate["rsi_overbought"] (RSI>65) → SHORT
      İkisi de değilse (nötr RSI)          → sinyal yok

    GATE (ikisi de zorunlu):
      1) candidate["within_tolerance"]
      2) rsi_oversold (LONG) VEYA rsi_overbought (SHORT)

    CONFIDENCE (gate gectikten sonra kademelendirir):
      LONG:  medium = Double Bottom yapısı | high = Double Bottom tam teyitli
      SHORT: medium = Double Top yapısı    | high = Double Top tam teyitli

    RISK: stop ATR bazlı - LONG'da entry'nin ALTINA, SHORT'ta entry'nin
    ÜSTÜNE (compute_stop_and_size'ın direction parametresi). Pozisyon
    0.5R/1.0R/1.5R aynı şekilde.

    Donus: None (gate gecmedi/RSI notr) VEYA dict (sinyal detaylari,
    "direction": "long"|"short" alanıyla).
    ══════════════════════════════════════════════════════════════
    """
    if candidate is None or not candidate.get("within_tolerance"):
        return None

    if candidate.get("rsi_oversold"):
        direction = "long"
    elif candidate.get("rsi_overbought"):
        direction = "short"
    else:
        return None  # RSI notr bolgede - ne long ne short

    entry_price = candidate["current_price"]

    if direction == "long":
        db = check_double_bottom(df)
        structure_present = "low1_idx" in db
        full_validation = evaluate_signal(df)
        fully_confirmed = bool(full_validation.is_valid)
    else:
        dt = check_double_top(df)
        structure_present = "high1_idx" in dt
        full_validation = evaluate_signal_short(df)
        fully_confirmed = bool(full_validation.is_valid)

    if fully_confirmed:
        confidence = "high"
    elif structure_present:
        confidence = "medium"
    else:
        confidence = "low"

    risk_calc = compute_stop_and_size(df, entry_price, equity, risk_pct, direction=direction)
    position_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5}[confidence]
    position_size = risk_calc["position_size"] * position_multiplier

    return {
        "direction": direction,
        "confidence": confidence,
        "structure_present": structure_present,
        "fully_confirmed": fully_confirmed,
        "entry_price": round(float(entry_price), 6),
        "stop_price": round(float(risk_calc["stop_price"]), 6),
        "position_size": round(float(position_size), 6),
        "swing_1_leg_id": candidate["swing_1_leg_id"],
        "swing_1_duration_bars": candidate["swing_1_duration_bars"],
        "swing_1_fib_ratio": candidate["swing_1_fib_ratio"],
        "swing_1_fib_price": candidate["swing_1_fib_price"],
        "swing_1_dist_pct": candidate["swing_1_dist_pct"],
        "swing_1_a_type": candidate["swing_1_a_type"],
        "swing_1_a_price": candidate["swing_1_a_price"],
        "swing_1_a_time": candidate["swing_1_a_time"],
        "swing_1_b_type": candidate["swing_1_b_type"],
        "swing_1_b_price": candidate["swing_1_b_price"],
        "swing_1_b_time": candidate["swing_1_b_time"],
        "swing_2_leg_id": candidate["swing_2_leg_id"],
        "swing_2_duration_bars": candidate["swing_2_duration_bars"],
        "swing_2_fib_ratio": candidate["swing_2_fib_ratio"],
        "swing_2_fib_price": candidate["swing_2_fib_price"],
        "swing_2_dist_pct": candidate["swing_2_dist_pct"],
        "swing_2_a_type": candidate["swing_2_a_type"],
        "swing_2_a_price": candidate["swing_2_a_price"],
        "swing_2_a_time": candidate["swing_2_a_time"],
        "swing_2_b_type": candidate["swing_2_b_type"],
        "swing_2_b_price": candidate["swing_2_b_price"],
        "swing_2_b_time": candidate["swing_2_b_time"],
        "current_rsi": candidate["current_rsi"],
    }



def check_shortening_of_thrust(pivots, end_leg_id, direction):
    """Wyckoff 'shortening of thrust': son 3 bacağın genliğinin (|B-A|)
    küçülüp küçülmediğini kontrol eder - bir trendin ivme kaybettiğinin
    işareti. direction='long' ise dip bacaklarına, 'short' ise tepe
    bacaklarına bakar (basitleştirilmiş: her iki yönde de son 3 bacağın
    genel genlik trendine bakılır, spesifik dip/tepe ayrımı yapılmaz -
    zigzag bacakları zaten alternatif dip/tepe içerir)."""
    if end_leg_id < 2:
        return {"passed": False, "reason": "Yeterli bacak geçmişi yok (en az 3 bacak gerekli)"}
    legs_by_id = {p: None for p in range(end_leg_id + 1)}
    # pivots listesinden bacak genliklerini yeniden hesapla
    magnitudes = []
    for k in range(max(0, end_leg_id - 2), end_leg_id + 1):
        if k + 1 >= len(pivots):
            continue
        mag = abs(pivots[k + 1]["price"] - pivots[k]["price"])
        magnitudes.append(mag)
    if len(magnitudes) < 3:
        return {"passed": False, "reason": "Yeterli bacak geçmişi yok (en az 3 bacak gerekli)"}
    shortening = magnitudes[0] > magnitudes[1] > magnitudes[2]
    return {
        "passed": shortening,
        "magnitudes": magnitudes,
        "reason": None if shortening else "Ardışık bacak genlikleri küçülmüyor",
    }


def check_confluence_match(pivots, primary_leg_id, primary_level, tolerance=SPEC_CONFLUENCE_TOLERANCE):
    """İkinci BAĞIMSIZ bir bacağın herhangi bir fib seviyesi, ana swing'in
    eşleşen seviyesine ±tolerance içinde mi? (confidence katmanı - gate
    değil). find_confluence_candidates'tan BAĞIMSIZ, basitleştirilmiş bir
    kontrol: sadece 'başka bir bağımsız bacak var mı ve seviyesi yakın mı'
    sorusuna bakar, rol (retracement/extension) ayrımı yapmaz."""
    for k in range(len(pivots) - 1):
        if k == primary_leg_id:
            continue
        p_a, p_b = pivots[k], pivots[k + 1]
        if p_a["time"] == pivots[primary_leg_id]["time"] or p_a["time"] == pivots[primary_leg_id + 1]["time"]:
            continue
        if p_b["time"] == pivots[primary_leg_id]["time"] or p_b["time"] == pivots[primary_leg_id + 1]["time"]:
            continue
        if abs(p_b["price"] - p_a["price"]) <= 0:
            continue
        levels = compute_fib_levels(p_a["price"], p_b["price"], p_a["type"], FIB_CONFLUENCE_LEVELS)
        for ratio, level in levels.items():
            if level <= 0:
                continue
            if abs(primary_level - level) / level <= tolerance:
                return {"passed": True, "matched_leg_id": k, "matched_ratio": ratio, "matched_level": level}
    return {"passed": False, "reason": "Bağımsız ikinci bir swing'in fib seviyesi yakın mesafede değil"}


def check_time_symmetry(df, a_time, b_time, ratios=(0.618, 1.0, 1.272), bar_tolerance=SPEC_TIME_SYMMETRY_BAR_TOLERANCE):
    """Güncel barın, swing süresinin (B-A) 0.618/1.0/1.272 katı kadar
    B'den sonra oluşan bir 'zaman izdüşümü' noktasına ±bar_tolerance
    içinde olup olmadığını kontrol eder (Gann/zaman-fiyat simetrisi,
    confidence katmanı - gate değil)."""
    try:
        a_idx = df.index.get_loc(a_time)
        b_idx = df.index.get_loc(b_time)
    except KeyError:
        return {"passed": False, "reason": "Zaman indeksleri bulunamadı"}
    swing_duration = b_idx - a_idx
    if swing_duration <= 0:
        return {"passed": False, "reason": "Geçersiz swing süresi"}
    current_idx = len(df) - 1
    for ratio in ratios:
        projected_idx = b_idx + round(swing_duration * ratio)
        if abs(current_idx - projected_idx) <= bar_tolerance:
            return {"passed": True, "matched_ratio": ratio, "projected_idx": projected_idx}
    return {"passed": False, "reason": "Güncel bar hiçbir zaman izdüşümü noktasına yakın değil"}


def get_btc_trend(btc_df, sma_period=20):
    """BTC/USDT 4H df'inden basit bir trend yönü çıkarır: son kapanış,
    SMA(sma_period)'in üzerindeyse 'up', altındaysa 'down', çok yakınsa
    (±%0.5) 'neutral'. BTC verisi alınamazsa None döner (opsiyonel katman
    - alınamazsa sessizce atlanır, hata fırlatmaz)."""
    if btc_df is None or len(btc_df) < sma_period:
        return None
    sma = btc_df["close"].iloc[-sma_period:].mean()
    current = btc_df["close"].iloc[-1]
    diff_pct = (current - sma) / sma
    if diff_pct > 0.005:
        return "up"
    elif diff_pct < -0.005:
        return "down"
    return "neutral"


def _evaluate_spec_direction(df, direction, pivots, btc_trend=None, equity=1000, risk_pct=0.015):
    """Tek bir yön (long veya short) için spec'teki tam gate + confidence
    + risk zincirini çalıştırır. evaluate_double_bottom_spec_signal
    tarafından her iki yön için de çağrılır."""
    failed_conditions = []
    passed_conditions = []

    if direction == "long":
        structure_check = check_double_bottom(df, min_bar_gap=SPEC_MIN_BAR_GAP, level_tolerance=SPEC_LEVEL_TOLERANCE)
        structure_key = "double_bottom"
    else:
        structure_check = check_double_top(df, min_bar_gap=SPEC_MIN_BAR_GAP, level_tolerance=SPEC_LEVEL_TOLERANCE)
        structure_key = "double_top"

    if not structure_check["passed"]:
        failed_conditions.append(f"{structure_key}: {structure_check['reason']}")
        return {"gate_passed": False, "failed_conditions": failed_conditions, "passed_conditions": passed_conditions}
    passed_conditions.append(f"{structure_key}: teyitli")

    if direction == "long":
        idx1, idx2 = structure_check["low1_idx"], structure_check["low2_idx"]
        rsi_check = check_rsi_divergence(df, idx1, idx2)
    else:
        idx1, idx2 = structure_check["high1_idx"], structure_check["high2_idx"]
        rsi_check = check_rsi_negative_divergence(df, idx1, idx2)

    if not rsi_check["passed"]:
        failed_conditions.append(f"rsi_divergence: {rsi_check['reason']}")
        return {"gate_passed": False, "failed_conditions": failed_conditions, "passed_conditions": passed_conditions}
    passed_conditions.append("rsi_divergence: teyitli")

    swing_high = df["high"].iloc[-60:].max()
    swing_low = df["low"].iloc[-60:].min()
    fib_check = check_fibonacci_zone(df, swing_high, swing_low, tolerance=SPEC_FIB_ZONE_TOLERANCE)
    if not fib_check["passed"]:
        failed_conditions.append(f"fib_zone: {fib_check['reason']}")
        return {"gate_passed": False, "failed_conditions": failed_conditions, "passed_conditions": passed_conditions}
    passed_conditions.append("fib_zone: teyitli")

    # ── GATE GEÇİLDİ - Confidence katmanları ────────────────────────
    layers = {}

    if direction == "long":
        wyckoff_check = check_wyckoff_volume(df, idx1, idx2)
    else:
        wyckoff_check = check_wyckoff_volume_short(df, idx1, idx2)
    thrust_check = None
    # Wyckoff spring/upthrust VEYA shortening of thrust - hangisi bacak
    # bilgisiyle eslesiyorsa onu kullan (spec: "VEYA")
    wyckoff_or_thrust = wyckoff_check["passed"]
    if not wyckoff_or_thrust and pivots:
        nearest_leg_id = max((k for k in range(len(pivots) - 1) if pivots[k + 1]["time"] <= df.index[idx2]), default=None)
        if nearest_leg_id is not None:
            thrust_check = check_shortening_of_thrust(pivots, nearest_leg_id, direction)
            wyckoff_or_thrust = thrust_check["passed"]
    layers["wyckoff_spring_or_thrust"] = bool(wyckoff_or_thrust)

    confluence_check = {"passed": False}
    if pivots:
        nearest_leg_id = max((k for k in range(len(pivots) - 1) if pivots[k + 1]["time"] <= df.index[idx2]), default=None)
        if nearest_leg_id is not None:
            primary_level = fib_check.get("fib618") or swing_high
            confluence_check = check_confluence_match(pivots, nearest_leg_id, primary_level)
    layers["confluence"] = bool(confluence_check["passed"])

    time_symmetry_check = check_time_symmetry(df, df.index[idx1], df.index[idx2])
    layers["time_symmetry"] = bool(time_symmetry_check["passed"])

    btc_layer = 0.0
    if btc_trend is not None:
        if direction == "long":
            btc_layer = 0.5 if btc_trend in ("up", "neutral") else -0.5
        else:
            btc_layer = 0.5 if btc_trend in ("down", "neutral") else -0.5
    layers["btc_trend_alignment"] = btc_layer

    # ── Elliott bağlamsal çarpan ─────────────────────────────────────
    major_high = df["high"].max()
    major_low = df["low"].min()
    entry_price = df["close"].iloc[-1]
    major_fib618 = major_high - 0.618 * (major_high - major_low)
    if direction == "long":
        elliott_multiplier = 0.7 if entry_price < major_fib618 else 1.0
    else:
        elliott_multiplier = 0.7 if entry_price > major_fib618 else 1.0

    layer1 = 1.0 if layers["wyckoff_spring_or_thrust"] else 0.0
    layer2 = 1.0 if layers["confluence"] else 0.0
    layer4 = layers["btc_trend_alignment"]
    final_score = (1 + layer1 + layer2 + layer4) * elliott_multiplier
    if layers["time_symmetry"]:
        final_score += 0.5

    if final_score < 1.5:
        confidence = "DÜŞÜK"
        r_multiple, risk_pct_label, position_multiplier = "0.5R", "%0.5", 0.5
    elif final_score < 2.5:
        confidence = "ORTA"
        r_multiple, risk_pct_label, position_multiplier = "1.0R", "%1", 1.0
    else:
        confidence = "YÜKSEK"
        r_multiple, risk_pct_label, position_multiplier = "1.5R", "%1.5-2", 1.5

    risk_calc = compute_stop_and_size(df, entry_price, equity, risk_pct, direction=direction)

    return {
        "gate_passed": True,
        "failed_conditions": failed_conditions,
        "passed_conditions": passed_conditions,
        "layers": layers,
        "elliott_multiplier": elliott_multiplier,
        "final_score": round(final_score, 3),
        "confidence": confidence,
        "entry": round(float(entry_price), 6),
        "stop": round(float(risk_calc["stop_price"]), 6),
        "r_multiple": r_multiple,
        "position_size_pct": risk_pct_label,
        "position_size": round(float(risk_calc["position_size"]) * position_multiplier, 6),
        "b_time": df.index[idx2].isoformat(),  # tekrar sinyal engelleme (spec madde 10) icin
    }


def evaluate_double_bottom_spec_signal(df, btc_trend=None, equity=1000, risk_pct=0.015):
    """
    ══════════════════════════════════════════════════════════════
    SPEC RAPORU (2026-07-14) — Kullanıcının Proje'den aldığı detaylı
    kriterlere göre TAMAMEN AYRI bir strateji. Ana confluence gate'ini
    (find_confluence_candidates/evaluate_confluence_entry) HİÇBİR
    ŞEKİLDE etkilemez - ayrı CSV (okx_double_bottom_spec.csv) + ayrı
    Telegram bildirimi.

    GATE: Double Bottom/Top + RSI diverjansı + Fibonacci bandı (long/short
    ayna simetrik). Her iki yön de GATE'i geçerse "CONFLICT" - sinyal
    sayılmaz. Confidence katmanları (Wyckoff/confluence/zaman simetrisi/
    BTC uyumu) + Elliott çarpanı ile DÜŞÜK/ORTA/YÜKSEK sınıflandırma.

    Donus: dict - "final_signal": "VALID"|"REJECTED"|"CONFLICT"
    ══════════════════════════════════════════════════════════════
    """
    pivots = find_zigzag_pivots(df)

    long_result = _evaluate_spec_direction(df, "long", pivots, btc_trend, equity, risk_pct)
    short_result = _evaluate_spec_direction(df, "short", pivots, btc_trend, equity, risk_pct)

    if long_result["gate_passed"] and short_result["gate_passed"]:
        return {
            "final_signal": "CONFLICT", "direction": None, "confidence": None,
            "long_detail": long_result, "short_detail": short_result,
        }

    if long_result["gate_passed"]:
        chosen, direction = long_result, "LONG"
    elif short_result["gate_passed"]:
        chosen, direction = short_result, "SHORT"
    else:
        # Ikisi de gate'i gecemedi - REJECTED. Hangisi daha az eksikle
        # reddedildiyse "direction_checked" olarak o gosterilir.
        direction = "LONG" if len(long_result["failed_conditions"]) <= len(short_result["failed_conditions"]) else "SHORT"
        return {
            "final_signal": "REJECTED", "direction_checked": direction, "confidence": None,
            "long_detail": long_result, "short_detail": short_result,
        }

    return {
        "final_signal": "VALID",
        "direction": direction,
        "confidence": chosen["confidence"],
        "entry": chosen["entry"], "stop": chosen["stop"],
        "r_multiple": chosen["r_multiple"], "position_size_pct": chosen["position_size_pct"],
        "position_size": chosen["position_size"],
        "b_time": chosen["b_time"],
        "score_breakdown": {
            "gate": "PASSED",
            "wyckoff_spring_or_thrust": chosen["layers"]["wyckoff_spring_or_thrust"],
            "confluence": chosen["layers"]["confluence"],
            "time_symmetry": chosen["layers"]["time_symmetry"],
            "btc_trend_alignment": chosen["layers"]["btc_trend_alignment"],
            "elliott_multiplier": chosen["elliott_multiplier"],
            "final_score": chosen["final_score"],
        },
        "passed_conditions": chosen["passed_conditions"],
        "long_detail": long_result, "short_detail": short_result,
    }


def format_spec_signal_telegram_message(symbol, spec_result):
    """VALID spec sinyalini Telegram icin kisa, spec'in istedigi formatta
    HTML mesajina cevirir. Sadece VALID sinyaller icin cagrilir."""
    direction = spec_result["direction"]
    emoji = "🟢" if direction == "LONG" else "🔴"
    reasons_summary = ", ".join(spec_result["passed_conditions"])
    return (
        f"{emoji} <b>{symbol}</b> {direction} [SPEC] — Güven: {spec_result['confidence']}\n"
        f"Giriş: <code>{spec_result['entry']}</code> | Stop: <code>{spec_result['stop']}</code> | "
        f"R: {spec_result['r_multiple']}\n"
        f"Gerekçe: {reasons_summary}"
    )


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
            "zigzag_pct_threshold": ZIGZAG_PCT_THRESHOLD,
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
            f"yön={v['direction']:<5} | yapı={v['structure_present']} | tam_teyit={v['fully_confirmed']} | "
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
    abc_pattern_results = []   # EK/BAĞIMSIZ rapor - ana gate'i etkilemez
    spec_results        = []   # EK/BAĞIMSIZ rapor (2026-07-14 spec) - ana gate'i etkilemez
    skipped_data        = 0
    skipped_crit         = 0
    previous_watchlist_state = load_previous_watchlist_state()
    previous_telegram_state = load_previous_telegram_state()
    telegram_notified_this_run = dict(previous_telegram_state)  # kopya - bu tur guncellenecek
    previous_spec_notified_state = load_previous_spec_notified_state()
    spec_notified_this_run = dict(previous_spec_notified_state)
    log_status(f"📈  Onceki tur state: {len(previous_watchlist_state)} coin (momentum karsilastirmasi icin)")

    # BTC 4H trend'i (spec raporunun opsiyonel BTC uyum katmani icin) -
    # taramanin basinda BIR KEZ cekilir, hata olursa None kalir (katman
    # sessizce atlanir, ana taramayi durdurmaz).
    btc_trend = None
    try:
        btc_df = fetch_ohlcv_paginated(exchange, "BTC/USDT", TIMEFRAME, since_ms=START_DATE_MS, pause=PAUSE_SEC)
        btc_trend = get_btc_trend(btc_df)
        log_status(f"₿  BTC 4H trend: {btc_trend}")
    except Exception as exc:
        log_status(f"⚠️  BTC trend hesaplanamadı (spec BTC katmanı atlanacak): {exc}")

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

        # EK/BAĞIMSIZ rapor: büyük dalga retracement'i + küçük ABC
        # düzeltmesinin extension'ı. Ana gate'i, confidence'ı veya
        # watchlist'i HİÇBİR ŞEKİLDE etkilemez - sadece ayrı bir CSV'ye
        # yazılır. try/except ile korunuyor: burada bir hata olursa ana
        # tarama asla etkilenmemeli.
        try:
            abc_pattern = find_big_wave_abc_pattern(df)
        except Exception as exc:
            abc_pattern = None
            log_status(f"⚠️  [Büyük Dalga ABC] {symbol} için hata (yok sayıldı): {exc}")
        if abc_pattern is not None:
            abc_pattern["symbol"] = symbol
            abc_pattern_results.append(abc_pattern)
            log_signal(
                f"🌊  [Büyük Dalga ABC] {symbol:<16} | büyük {abc_pattern['big_wave_retracement_ratio']}@%{abc_pattern['big_wave_dist_pct']} | "
                f"küçük-C {abc_pattern['small_leg_c_extension_ratio']}@%{abc_pattern['small_leg_dist_pct']} | "
                f"toplam=%{abc_pattern['combined_dist_pct']} | tolerans_içinde={abc_pattern['within_tolerance']}"
            )
        vol_24h = volume_by_symbol.get(symbol)

        # EK/BAĞIMSIZ rapor (2026-07-14 spec): Double Bottom/Top ana gate,
        # RSI diverjansı + Fib bandı, Wyckoff/confluence/zaman simetrisi/
        # BTC uyumu confidence katmanları. Ana confluence gate'ini
        # HİÇBİR ŞEKİLDE etkilemez. try/except ile korunuyor.
        try:
            spec_result = evaluate_double_bottom_spec_signal(df, btc_trend=btc_trend)
        except Exception as exc:
            spec_result = None
            log_status(f"⚠️  [Spec] {symbol} için hata (yok sayıldı): {exc}")

        if spec_result is not None:
            spec_result["symbol"] = symbol
            spec_result["volume_24h_usdt"] = round(float(vol_24h), 2) if vol_24h is not None else None
            spec_results.append(spec_result)

            if spec_result["final_signal"] == "VALID":
                log_signal(
                    f"📐  [Spec] {symbol:<16} | {spec_result['direction']} | confidence={spec_result['confidence']} | "
                    f"entry={spec_result['entry']} stop={spec_result['stop']} r={spec_result['r_multiple']}"
                )
                if should_notify_spec(symbol, spec_result.get("b_time", ""), previous_spec_notified_state):
                    spec_text = format_spec_signal_telegram_message(symbol, spec_result)
                    if send_telegram_message(spec_text):
                        log_status(f"📨  [Spec] Telegram bildirimi gönderildi → {symbol}")
                        spec_notified_this_run[symbol] = {
                            "b_time": spec_result.get("b_time", ""),
                            "notified_at": datetime.now(timezone.utc).isoformat(),
                        }
            elif spec_result["final_signal"] == "CONFLICT":
                log_signal(f"⚡  [Spec] {symbol:<16} | CONFLICT — hem LONG hem SHORT gate'i geçiyor, sinyal sayılmadı")
            else:
                log_signal(
                    f"ℹ️  [Spec] {symbol:<16} | REJECTED | "
                    f"long_failed={spec_result['long_detail']['failed_conditions']} | "
                    f"short_failed={spec_result['short_detail']['failed_conditions']}"
                )

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
            if should_notify_telegram(symbol, confluence_signal["direction"], previous_telegram_state):
                telegram_text = format_signal_telegram_message(confluence_signal)
                if send_telegram_message(telegram_text):
                    log_status(f"📨  Telegram bildirimi gönderildi → {symbol}")
                    telegram_notified_this_run[symbol] = {
                        "direction": confluence_signal["direction"],
                        "notified_at": datetime.now(timezone.utc).isoformat(),
                    }
            else:
                log_status(f"🔕  Telegram bildirimi atlandı (cooldown, {TELEGRAM_COOLDOWN_HOURS}sa) → {symbol}")
        else:
            skipped_crit += 1
            if candidate is None:
                reason_map = {
                    "yetersiz_pivot": "En az 2 pivot bacağı bulunamadı (fiyat hareketi yetersiz)",
                    "kucuk_abc_bacagi_yok": "Büyük dalgadan sonra bağımsız bir küçük ABC bacağı yok",
                    "cok_uzak": "Büyük dalga retracement'i ile küçük bacak extension'ı makul mesafede değil",
                    "rsi_hesaplanamiyor": "RSI henüz hesaplanamıyor (ısınma dönemi)",
                }
                reason_text = reason_map.get(no_candidate_reason, no_candidate_reason)
                log_signal(f"ℹ️  [Confluence] {symbol:<16} reddedildi | reasons=['{reason_text}']")
            else:
                reasons = []
                if not candidate.get("within_tolerance"):
                    reasons.append(f"Confluence tolerans dışı (toplam mesafe %{candidate['combined_dist_pct']})")
                if not candidate.get("rsi_oversold") and not candidate.get("rsi_overbought"):
                    reasons.append(f"RSI ne oversold(<{RSI_OVERSOLD_THRESHOLD}) ne overbought(>{RSI_OVERBOUGHT_THRESHOLD}) (RSI={candidate['current_rsi']})")
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
                        "swing_1_leg_id": candidate["swing_1_leg_id"],
        "swing_1_duration_bars": candidate["swing_1_duration_bars"],
                        "swing_1_fib_ratio": candidate["swing_1_fib_ratio"],
                        "swing_1_fib_price": candidate["swing_1_fib_price"],
                        "swing_1_dist_pct": candidate["swing_1_dist_pct"],
                        "swing_1_a_time": candidate["swing_1_a_time"],
                        "swing_1_b_time": candidate["swing_1_b_time"],
                        "swing_2_leg_id": candidate["swing_2_leg_id"],
        "swing_2_duration_bars": candidate["swing_2_duration_bars"],
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
    save_watchlist_state(watchlist_results, telegram_notified_this_run, spec_notified_this_run)

    # ── İzleme Listesi CSV'si (0 sinyal olsa bile HER ZAMAN yazılır) ──
    if watchlist_results:
        watch_col_map = {
            "symbol"              : "Sembol",
            "trend"                : "Trend",
            "momentum_pct"         : "Momentum (önceki tura göre)",
            "combined_dist_pct"    : "Toplam Mesafe%",
            "current_rsi"          : "Güncel RSI",
            "rsi_oversold"         : "RSI Oversold mu",
            "swing_1_leg_id"        : "Swing1 Bacak No",
            "swing_1_duration_bars" : "Swing1 Süre (bar)",
            "swing_1_fib_ratio"    : "Swing1 Fib Oranı",
            "swing_1_fib_price"    : "Swing1 Fib Fiyatı",
            "swing_1_dist_pct"     : "Swing1 Uzaklık%",
            "swing_1_a_time"       : "Swing1 A Zamanı",
            "swing_1_b_time"       : "Swing1 B Zamanı",
            "swing_2_leg_id"        : "Swing2 Bacak No",
            "swing_2_duration_bars" : "Swing2 Süre (bar)",
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

    # ── Büyük Dalga + ABC Deseni CSV'si (EK/BAĞIMSIZ, 0 olsa bile yazılır) ──
    if abc_pattern_results:
        abc_col_map = {
            "symbol"                       : "Sembol",
            "combined_dist_pct"            : "Toplam Mesafe%",
            "within_tolerance"             : "Tolerans İçinde mi",
            "big_wave_retracement_ratio"   : "Büyük Dalga Retracement Oranı",
            "big_wave_retracement_price"   : "Büyük Dalga Retracement Fiyatı",
            "big_wave_dist_pct"            : "Büyük Dalga Uzaklık%",
            "big_wave_a_type"              : "Büyük Dalga A Tipi",
            "big_wave_a_price"             : "Büyük Dalga A Fiyatı",
            "big_wave_a_time"              : "Büyük Dalga A Zamanı",
            "big_wave_b_type"              : "Büyük Dalga B Tipi",
            "big_wave_b_price"             : "Büyük Dalga B Fiyatı",
            "big_wave_b_time"              : "Büyük Dalga B Zamanı",
            "small_leg_c_extension_ratio"  : "Küçük Bacak C Extension Oranı",
            "small_leg_c_extension_price"  : "Küçük Bacak C Extension Fiyatı",
            "small_leg_dist_pct"           : "Küçük Bacak Uzaklık%",
            "small_leg_a_type"             : "Küçük Bacak A Tipi",
            "small_leg_a_price"            : "Küçük Bacak A Fiyatı",
            "small_leg_a_time"             : "Küçük Bacak A Zamanı",
            "small_leg_b_type"             : "Küçük Bacak B Tipi (C öncesi)",
            "small_leg_b_price"            : "Küçük Bacak B Fiyatı (C öncesi)",
            "small_leg_b_time"             : "Küçük Bacak B Zamanı (C öncesi)",
            "current_price"                : "Güncel Fiyat",
        }
        df_abc = pd.DataFrame(abc_pattern_results)
        df_abc = df_abc[[c for c in abc_col_map if c in df_abc.columns]]
        df_abc = df_abc.rename(columns=abc_col_map)
        df_abc = df_abc.sort_values("Toplam Mesafe%").reset_index(drop=True)

        if EXPORT_CSV:
            abc_fn = "okx_big_wave_abc.csv"
            df_abc.to_csv(abc_fn, index=False, encoding="utf-8-sig")
            log_status(f"🌊  CSV (Büyük Dalga ABC, {len(df_abc)} coin) → {abc_fn}")

    # ── Spec Raporu CSV'si (2026-07-14, EK/BAĞIMSIZ, 0 olsa bile yazılır) ──
    if spec_results:
        spec_rows = []
        for r in spec_results:
            row = {
                "symbol": r["symbol"],
                "final_signal": r["final_signal"],
                "direction": r.get("direction") or r.get("direction_checked"),
                "confidence": r.get("confidence"),
                "entry": r.get("entry"),
                "stop": r.get("stop"),
                "r_multiple": r.get("r_multiple"),
                "final_score": (r.get("score_breakdown") or {}).get("final_score"),
                "passed_conditions": "; ".join(r.get("passed_conditions", [])) if r.get("passed_conditions") else None,
                "long_failed": "; ".join(r["long_detail"]["failed_conditions"]),
                "short_failed": "; ".join(r["short_detail"]["failed_conditions"]),
                "volume_24h_usdt": r.get("volume_24h_usdt"),
            }
            spec_rows.append(row)

        spec_col_map = {
            "symbol": "Sembol", "final_signal": "Sonuç", "direction": "Yön",
            "confidence": "Güven", "entry": "Giriş", "stop": "Stop", "r_multiple": "R",
            "final_score": "Skor", "passed_conditions": "Geçen Koşullar",
            "long_failed": "LONG Reddedilme Nedeni", "short_failed": "SHORT Reddedilme Nedeni",
            "volume_24h_usdt": "24s Hacim (USDT)",
        }
        df_spec = pd.DataFrame(spec_rows)
        df_spec = df_spec[[c for c in spec_col_map if c in df_spec.columns]]
        df_spec = df_spec.rename(columns=spec_col_map)
        signal_rank = {"VALID": 2, "CONFLICT": 1, "REJECTED": 0}
        df_spec["_rank"] = df_spec["Sonuç"].map(signal_rank)
        df_spec = df_spec.sort_values("_rank", ascending=False).drop(columns=["_rank"]).reset_index(drop=True)

        if EXPORT_CSV:
            spec_fn = "okx_double_bottom_spec.csv"
            df_spec.to_csv(spec_fn, index=False, encoding="utf-8-sig")
            valid_count = sum(1 for r in spec_results if r["final_signal"] == "VALID")
            log_status(f"📐  CSV (Spec Raporu, {valid_count} VALID / {len(df_spec)} toplam) → {spec_fn}")

    if not validation_results:
        log_status("\n❌  Hiç aday bulunamadı (Confluence gate'ini geçen sinyal yok).")
        return None

    # ── Tek birleşik sinyal CSV çıktısı ────────────────────────────────
    col_map = {
        "symbol"                        : "Sembol",
        "confidence"                     : "Güven",
        "direction"                       : "Yön",
        "structure_present"               : "Yapı Var mı (DB/DT)",
        "fully_confirmed"                 : "Tam Teyitli mi (DB/DT)",
        "entry_price"                    : "Giriş Fiyatı",
        "stop_price"                     : "Stop Fiyatı",
        "position_size"                  : "Pozisyon Büyüklüğü",
        "swing_1_leg_id"                  : "Swing1 Bacak No",
        "swing_1_duration_bars"          : "Swing1 Süre (bar)",
        "swing_1_fib_ratio"              : "Swing1 Fib Oranı",
        "swing_1_fib_price"              : "Swing1 Fib Fiyatı",
        "swing_1_dist_pct"               : "Swing1 Uzaklık%",
        "swing_2_leg_id"                  : "Swing2 Bacak No",
        "swing_2_duration_bars"          : "Swing2 Süre (bar)",
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

    log_status(f"   Swing1: bacak{candidate_t['swing_1_leg_id']} ({candidate_t['swing_1_duration_bars']} bar), {candidate_t['swing_1_fib_ratio']} → "
               f"{candidate_t['swing_1_fib_price']} (Δ%{candidate_t['swing_1_dist_pct']})")
    log_status(f"   Swing2: bacak{candidate_t['swing_2_leg_id']} ({candidate_t['swing_2_duration_bars']} bar), {candidate_t['swing_2_fib_ratio']} → "
               f"{candidate_t['swing_2_fib_price']} (Δ%{candidate_t['swing_2_dist_pct']})")
    log_status(f"   Toplam mesafe: %{candidate_t['combined_dist_pct']} | within_tolerance={candidate_t['within_tolerance']}")
    log_status(f"   RSI: {candidate_t['current_rsi']} | oversold={candidate_t['rsi_oversold']}")

    confluence_signal_t = evaluate_confluence_entry(df_t, candidate_t)

    if confluence_signal_t is not None:
        log_status(f"\n   ✅  SİNYAL GEÇERLİ — confidence={confluence_signal_t['confidence']}")
        log_status(f"      direction                     : {confluence_signal_t['direction']}")
        log_status(f"      structure_present             : {confluence_signal_t['structure_present']}")
        log_status(f"      fully_confirmed               : {confluence_signal_t['fully_confirmed']}")
        log_status(f"      entry_price : {confluence_signal_t['entry_price']}")
        log_status(f"      stop_price  : {confluence_signal_t['stop_price']}")
        log_status(f"      position_size: {confluence_signal_t['position_size']}")
    else:
        reasons = []
        if not candidate_t.get("within_tolerance"):
            reasons.append(f"Confluence tolerans dışı (toplam mesafe %{candidate_t['combined_dist_pct']})")
        if not candidate_t.get("rsi_oversold") and not candidate_t.get("rsi_overbought"):
            reasons.append(f"RSI ne oversold(<{RSI_OVERSOLD_THRESHOLD}) ne overbought(>{RSI_OVERBOUGHT_THRESHOLD}) (RSI={candidate_t['current_rsi']})")
        log_status(f"\n   ❌  Gate reddedildi. reasons={reasons}")

        if (not candidate_t["within_tolerance"]
                and candidate_t["combined_dist_pct"] <= WATCHLIST_MAX_COMBINED_DIST_PCT):
            log_status(f"\n   👀  İZLEME LİSTESİNDE — toplam mesafe %{candidate_t['combined_dist_pct']}, "
                       f"tolerans sınırı %{WATCHLIST_MAX_COMBINED_DIST_PCT}")
        else:
            log_status("\n   (İzleme listesine de girmiyor — çok uzak)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="OKX Fibonacci Confluence Tarayıcı (tek strateji, v4.1)"
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
