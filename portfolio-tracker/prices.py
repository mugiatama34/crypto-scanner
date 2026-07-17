"""Yahoo Finance (yfinance) uzerinden guncel ve gecmis fiyat verisi.

API anahtari gerekmez. Yfinance'in dondurdugu her deger, agdan
cekilemedigi (baglanti sorunu, yanlis sembol vb.) durumlarda None/bos
olabilir - cagiran taraf (portfolio.py / app.py) bunu son islem fiyati
gibi bir fallback ile karsilar, uygulama hic bir zaman bu yuzden
patlamaz.
"""

from datetime import datetime, timedelta

import yfinance as yf

CACHE_MAX_AGE_MINUTES = 240  # onbellekteki fiyat 4 saatten eskiyse yeniden cek


def _fetch_current_price(symbol):
    try:
        hist = yf.Ticker(symbol).history(period="1d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


def _fetch_history(symbol, start_date):
    """{tarih_str: kapanis_fiyati} sozlugu doner."""
    try:
        hist = yf.Ticker(symbol).history(start=start_date)
        if hist.empty:
            return {}
        return {idx.strftime("%Y-%m-%d"): float(val) for idx, val in hist["Close"].items()}
    except Exception:
        return {}


def get_cached_prices(conn, symbols):
    """Onbellekteki fiyatlari {symbol: (price, fetched_at)} olarak doner."""
    if not symbols:
        return {}
    placeholders = ",".join("?" for _ in symbols)
    rows = conn.execute(
        f"SELECT symbol, price, fetched_at FROM price_cache WHERE symbol IN ({placeholders})",
        list(symbols),
    ).fetchall()
    return {r["symbol"]: (r["price"], r["fetched_at"]) for r in rows}


def save_prices_to_cache(conn, prices):
    now = datetime.utcnow().isoformat()
    for symbol, price in prices.items():
        conn.execute(
            """INSERT INTO price_cache (symbol, price, fetched_at) VALUES (?, ?, ?)
               ON CONFLICT(symbol) DO UPDATE SET price = excluded.price, fetched_at = excluded.fetched_at""",
            (symbol, price, now),
        )
    conn.commit()


def get_current_prices(conn, symbols, force_refresh=False):
    """Guncel fiyatlari onbellekten (taze ise) ya da Yahoo Finance'ten
    ceker. Donus: ({symbol: price}, en_son_cekim_zamani veya None)."""
    if not symbols:
        return {}, None

    cached = get_cached_prices(conn, symbols)
    now = datetime.utcnow()

    def is_stale(symbol):
        if symbol not in cached:
            return True
        fetched_at = datetime.fromisoformat(cached[symbol][1])
        return now - fetched_at > timedelta(minutes=CACHE_MAX_AGE_MINUTES)

    needs_fetch = force_refresh or any(is_stale(s) for s in symbols)

    if needs_fetch:
        fresh = {}
        for symbol in symbols:
            price = _fetch_current_price(symbol)
            if price is not None:
                fresh[symbol] = price
        if fresh:
            save_prices_to_cache(conn, fresh)
        cached = get_cached_prices(conn, symbols)

    prices = {s: cached[s][0] for s in symbols if s in cached}
    fetched_at_values = [cached[s][1] for s in symbols if s in cached]
    last_fetched_at = max(fetched_at_values) if fetched_at_values else None
    return prices, last_fetched_at


def get_price_history(symbols, start_date):
    """Her sembol icin {tarih_str: kapanis_fiyati} sozlugunu bir arada
    {symbol: {...}} olarak doner. Agdan cekilemeyen semboller sozlukte
    yer almaz (bos donmez, sadece o sembol eksik kalir)."""
    history = {}
    for symbol in symbols:
        symbol_history = _fetch_history(symbol, start_date)
        if symbol_history:
            history[symbol] = symbol_history
    return history


def price_on_or_before(history_for_symbol, target_date):
    """history_for_symbol: {tarih_str: fiyat}. target_date'te ya da ondan
    once en yakin (borsanin acik oldugu) gundeki kapanis fiyatini bulur."""
    candidates = [d for d in history_for_symbol if d <= target_date]
    if not candidates:
        return None
    return history_for_symbol[max(candidates)]
