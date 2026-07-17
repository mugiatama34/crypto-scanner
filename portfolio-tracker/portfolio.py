"""Portfoy hesaplamalari: pozisyon ozetleri, gerceklesen/gerceklesmeyen K/Z,
nakit bakiyesi ve girisden itibaren (all-time) performans.

Zaman dilimi (gunluk/haftalik/aylik/YTD) performans karsilastirmalari,
gecmis piyasa fiyati gerektirdigi icin fiyat API'si baglandiginda
(4. adim) eklenecek.
"""

from collections import defaultdict

from fifo import match_fifo

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
