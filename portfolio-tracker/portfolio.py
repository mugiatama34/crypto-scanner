"""Portfoy hesaplamalari: pozisyon ozetleri, gerceklesen/gerceklesmeyen K/Z,
nakit bakiyesi, girisden itibaren (all-time) performans ve gunluk/haftalik/
aylik/YTD donem karsilastirmalari.

Donem karsilastirmalari, o donemin basindaki net degeri (o tarihteki
hisse adetleri x o tarihteki kapanis fiyati + o tarihteki nakit) bugunku
net degerle kiyaslar; donem icinde yapilan yeni yatirim/cekimler bu
kiyaslamadan ayristirilir (yoksa yeni para yatirmak sahte bir "kazanc"
gibi gorunurdu).
"""

from collections import defaultdict

from fifo import match_fifo
from prices import price_on_or_before

EPSILON = 1e-9


def get_transactions_by_symbol(conn):
    rows = conn.execute("SELECT * FROM transactions ORDER BY tx_date, id").fetchall()
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["symbol"]].append(row)
    return grouped


def get_symbol_tags(conn):
    """Her sembol icin en son girilen etiketi kullanir."""
    rows = conn.execute(
        "SELECT symbol, tag FROM transactions WHERE tag IS NOT NULL AND tag != '' "
        "ORDER BY tx_date, id"
    ).fetchall()
    tags = {}
    for row in rows:
        tags[row["symbol"]] = row["tag"]
    return tags


def compute_portfolio(conn, price_lookup=None):
    """Tum semboller icin FIFO'yu calistirip pozisyon listesi ve toplamlari dondurur.

    price_lookup: {symbol: guncel_fiyat}. Fiyati olmayan (veya henuz API
    baglanmamis) semboller icin son islem fiyati kullanilir.
    """
    price_lookup = price_lookup or {}
    grouped = get_transactions_by_symbol(conn)
    tags = get_symbol_tags(conn)

    positions = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    total_realized_pnl = 0.0
    total_realized_cost = 0.0

    for symbol, txs in grouped.items():
        open_lots, closed = match_fifo(txs)

        symbol_realized_pnl = sum(m.realized_pnl for m in closed)
        symbol_realized_cost = sum(m.quantity * m.buy_price for m in closed)
        total_realized_pnl += symbol_realized_pnl
        total_realized_cost += symbol_realized_cost

        open_qty = sum(lot.quantity for lot in open_lots)
        if open_qty <= EPSILON:
            continue  # pozisyon tamamen kapanmis, ozet tabloda gosterilmiyor

        cost_basis = sum(lot.quantity * lot.price for lot in open_lots)
        avg_cost = cost_basis / open_qty

        current_price = price_lookup.get(symbol)
        price_is_live = current_price is not None
        if current_price is None:
            current_price = txs[-1]["price"]

        market_value = open_qty * current_price
        unrealized_pnl = market_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > EPSILON else 0.0

        total_market_value += market_value
        total_cost_basis += cost_basis

        positions.append({
            "symbol": symbol,
            "tag": tags.get(symbol),
            "open_qty": open_qty,
            "avg_cost": avg_cost,
            "current_price": current_price,
            "price_is_live": price_is_live,
            "cost_basis": cost_basis,
            "market_value": market_value,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "realized_pnl": symbol_realized_pnl,
        })

    positions.sort(key=lambda p: p["market_value"], reverse=True)

    total_unrealized_pnl = total_market_value - total_cost_basis
    total_unrealized_pnl_pct = (
        total_unrealized_pnl / total_cost_basis * 100 if total_cost_basis > EPSILON else 0.0
    )
    total_realized_pnl_pct = (
        total_realized_pnl / total_realized_cost * 100 if total_realized_cost > EPSILON else 0.0
    )

    return {
        "positions": positions,
        "total_market_value": total_market_value,
        "total_cost_basis": total_cost_basis,
        "total_unrealized_pnl": total_unrealized_pnl,
        "total_unrealized_pnl_pct": total_unrealized_pnl_pct,
        "total_realized_pnl": total_realized_pnl,
        "total_realized_pnl_pct": total_realized_pnl_pct,
    }


def compute_position_detail(conn, symbol, price_lookup=None):
    """Tek bir sembol icin FIFO detayi: acik lotlar, kapanan eslesmeler,
    tum islem gecmisi ve ozet rakamlar. Sembole ait hic islem yoksa None
    doner."""
    price_lookup = price_lookup or {}
    txs = conn.execute(
        "SELECT * FROM transactions WHERE symbol = ? ORDER BY tx_date, id", (symbol,)
    ).fetchall()
    if not txs:
        return None

    open_lots, closed = match_fifo(txs)

    open_qty = sum(lot.quantity for lot in open_lots)
    cost_basis = sum(lot.quantity * lot.price for lot in open_lots)
    avg_cost = cost_basis / open_qty if open_qty > EPSILON else 0.0

    current_price = price_lookup.get(symbol)
    price_is_live = current_price is not None
    if current_price is None:
        current_price = txs[-1]["price"]

    market_value = open_qty * current_price
    unrealized_pnl = market_value - cost_basis
    unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > EPSILON else 0.0

    realized_pnl = sum(m.realized_pnl for m in closed)
    realized_cost = sum(m.quantity * m.buy_price for m in closed)
    realized_pnl_pct = (realized_pnl / realized_cost * 100) if realized_cost > EPSILON else 0.0

    tags = get_symbol_tags(conn)

    return {
        "symbol": symbol,
        "tag": tags.get(symbol),
        "open_qty": open_qty,
        "avg_cost": avg_cost,
        "cost_basis": cost_basis,
        "current_price": current_price,
        "price_is_live": price_is_live,
        "market_value": market_value,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "realized_pnl": realized_pnl,
        "realized_pnl_pct": realized_pnl_pct,
        "open_lots": open_lots,
        "closed_matches": closed,
        "transactions": list(reversed(txs)),
    }


def compute_open_quantities_as_of(conn, as_of_date):
    """Her sembol icin, belirli bir tarihte elde olan (acik) adedi
    hesaplar - o tarihten sonraki islemler sayilmaz."""
    grouped = get_transactions_by_symbol(conn)
    quantities = {}
    for symbol, txs in grouped.items():
        filtered = [t for t in txs if t["tx_date"] <= as_of_date]
        if not filtered:
            continue
        open_lots, _ = match_fifo(filtered)
        qty = sum(lot.quantity for lot in open_lots)
        if qty > EPSILON:
            quantities[symbol] = qty
    return quantities


def compute_cash_balance_as_of(conn, as_of_date):
    deposits = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM cash_flows "
        "WHERE flow_type = 'deposit' AND flow_date <= ?",
        (as_of_date,),
    ).fetchone()["s"]
    withdrawals = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM cash_flows "
        "WHERE flow_type = 'withdrawal' AND flow_date <= ?",
        (as_of_date,),
    ).fetchone()["s"]
    buys = conn.execute(
        "SELECT COALESCE(SUM(total), 0) AS s FROM transactions "
        "WHERE tx_type = 'buy' AND tx_date <= ?",
        (as_of_date,),
    ).fetchone()["s"]
    sells = conn.execute(
        "SELECT COALESCE(SUM(total), 0) AS s FROM transactions "
        "WHERE tx_type = 'sell' AND tx_date <= ?",
        (as_of_date,),
    ).fetchone()["s"]
    return deposits - withdrawals - buys + sells


def compute_net_worth_as_of(conn, as_of_date, price_history):
    """price_history: {symbol: {tarih_str: kapanis_fiyati}}.

    Donus: (net_deger, tum_fiyatlar_bulundu_mu). Bir sembolun o tarihteki
    fiyati bulunamazsa tum_fiyatlar_bulundu_mu False doner - cagiran taraf
    bu durumda "eksik veri" olarak isaretlemeli, 0 degerinde varsaymamali.
    """
    quantities = compute_open_quantities_as_of(conn, as_of_date)
    market_value = 0.0
    all_priced = True
    for symbol, qty in quantities.items():
        price = price_on_or_before(price_history.get(symbol, {}), as_of_date)
        if price is None:
            all_priced = False
            continue
        market_value += qty * price
    cash = compute_cash_balance_as_of(conn, as_of_date)
    return market_value + cash, all_priced


def compute_net_deposits_between(conn, start_date, end_date):
    """start_date haric, end_date dahil araliktaki net yatirim (yatirim -
    cekim). Donem performansindan yeni sermaye etkisini ayiklamak icin."""
    deposits = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM cash_flows "
        "WHERE flow_type = 'deposit' AND flow_date > ? AND flow_date <= ?",
        (start_date, end_date),
    ).fetchone()["s"]
    withdrawals = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM cash_flows "
        "WHERE flow_type = 'withdrawal' AND flow_date > ? AND flow_date <= ?",
        (start_date, end_date),
    ).fetchone()["s"]
    return deposits - withdrawals


def compute_period_performance(conn, periods, current_net_worth, price_history, today_str):
    """periods: {etiket: baslangic_tarih_str}.

    Donus: {etiket: {"pnl": ..., "pnl_pct": ...} veya None (fiyat verisi
    eksikse)}.
    """
    results = {}
    for label, start_date in periods.items():
        start_net_worth, all_priced = compute_net_worth_as_of(conn, start_date, price_history)
        if not all_priced:
            results[label] = None
            continue
        net_deposits = compute_net_deposits_between(conn, start_date, today_str)
        pnl = current_net_worth - start_net_worth - net_deposits
        pnl_pct = (pnl / start_net_worth * 100) if start_net_worth > EPSILON else 0.0
        results[label] = {"pnl": pnl, "pnl_pct": pnl_pct}
    return results


def compute_cash_balance(conn):
    """Nakit bakiyesi = yatirilan sermaye - cekilen - alimlar + satimlar."""
    deposits = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM cash_flows WHERE flow_type = 'deposit'"
    ).fetchone()["s"]
    withdrawals = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM cash_flows WHERE flow_type = 'withdrawal'"
    ).fetchone()["s"]
    buys = conn.execute(
        "SELECT COALESCE(SUM(total), 0) AS s FROM transactions WHERE tx_type = 'buy'"
    ).fetchone()["s"]
    sells = conn.execute(
        "SELECT COALESCE(SUM(total), 0) AS s FROM transactions WHERE tx_type = 'sell'"
    ).fetchone()["s"]
    return deposits - withdrawals - buys + sells


def compute_net_deposited(conn):
    """Toplam yatirilan sermaye (cekilenler dusulmus): girisden itibaren
    performansin paydasi olarak kullanilir."""
    deposits = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM cash_flows WHERE flow_type = 'deposit'"
    ).fetchone()["s"]
    withdrawals = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM cash_flows WHERE flow_type = 'withdrawal'"
    ).fetchone()["s"]
    return deposits - withdrawals


def compute_all_time_performance(conn, total_market_value, cash_balance):
    """Girisden itibaren (all-time) performans: mevcut net deger ile
    yatirilan net sermaye arasindaki fark."""
    net_deposited = compute_net_deposited(conn)
    net_worth = total_market_value + cash_balance
    pnl = net_worth - net_deposited
    pnl_pct = (pnl / net_deposited * 100) if net_deposited > EPSILON else 0.0
    return {
        "net_deposited": net_deposited,
        "net_worth": net_worth,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
    }
