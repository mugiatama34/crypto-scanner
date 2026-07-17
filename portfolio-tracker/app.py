from datetime import date, timedelta

from flask import Flask, abort, flash, redirect, render_template, request, url_for

import db
import portfolio
import prices

app = Flask(__name__)
app.secret_key = "portfolio-tracker-local"  # sadece flash mesajlari icin, tek kullanicili yerel araç

db.init_db()


@app.route("/")
def dashboard():
    conn = db.get_connection()
    symbols = list(portfolio.get_transactions_by_symbol(conn).keys())

    price_lookup, last_fetched_at = prices.get_current_prices(conn, symbols)

    result = portfolio.compute_portfolio(conn, price_lookup)
    cash_balance = portfolio.compute_cash_balance(conn)
    all_time = portfolio.compute_all_time_performance(
        conn, result["total_market_value"], cash_balance
    )

    today = date.today()
    periods = {
        "daily": (today - timedelta(days=1)).isoformat(),
        "weekly": (today - timedelta(days=7)).isoformat(),
        "monthly": (today - timedelta(days=30)).isoformat(),
        "ytd": date(today.year, 1, 1).isoformat(),
    }
    price_history = prices.get_price_history(symbols, min(periods.values())) if symbols else {}
    period_performance = portfolio.compute_period_performance(
        conn, periods, all_time["net_worth"], price_history, today.isoformat()
    )

    conn.close()

    return render_template(
        "dashboard.html",
        positions=result["positions"],
        total_market_value=result["total_market_value"],
        total_unrealized_pnl=result["total_unrealized_pnl"],
        total_unrealized_pnl_pct=result["total_unrealized_pnl_pct"],
        total_realized_pnl=result["total_realized_pnl"],
        total_realized_pnl_pct=result["total_realized_pnl_pct"],
        cash_balance=cash_balance,
        all_time=all_time,
        period_performance=period_performance,
        last_fetched_at=last_fetched_at,
        today=today.isoformat(),
    )


@app.route("/positions/<symbol>")
def position_detail(symbol):
    symbol = symbol.upper()
    conn = db.get_connection()
    price_lookup, _ = prices.get_current_prices(conn, [symbol])
    detail = portfolio.compute_position_detail(conn, symbol, price_lookup)
    conn.close()

    if detail is None:
        abort(404)

    return render_template("position.html", p=detail)


@app.route("/prices/refresh", methods=["POST"])
def refresh_prices():
    conn = db.get_connection()
    symbols = list(portfolio.get_transactions_by_symbol(conn).keys())
    prices.get_current_prices(conn, symbols, force_refresh=True)
    conn.close()

    flash("Fiyatlar güncellendi.", "success")
    return redirect(url_for("dashboard"))


@app.route("/cash/add", methods=["POST"])
def add_cash_flow():
    form = request.form
    flow_type = form.get("flow_type")
    flow_date = form.get("flow_date")

    try:
        amount = float(form.get("amount", ""))
    except ValueError:
        flash("Tutar sayısal olmalı.", "error")
        return redirect(url_for("dashboard"))

    if flow_type not in ("deposit", "withdrawal") or not flow_date or amount <= 0:
        flash("Tür, tarih ve pozitif bir tutar girmelisin.", "error")
        return redirect(url_for("dashboard"))

    note = form.get("note", "").strip() or None

    conn = db.get_connection()
    conn.execute(
        "INSERT INTO cash_flows (flow_date, flow_type, amount, note) VALUES (?, ?, ?, ?)",
        (flow_date, flow_type, amount, note),
    )
    conn.commit()
    conn.close()

    label = "Yatırım" if flow_type == "deposit" else "Çekim"
    flash(f"{label} kaydedildi.", "success")
    return redirect(url_for("dashboard"))


@app.route("/transactions")
def transactions():
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT * FROM transactions ORDER BY tx_date DESC, id DESC"
    ).fetchall()
    conn.close()

    tags = sorted({r["tag"] for r in rows if r["tag"]})
    return render_template(
        "transactions.html",
        transactions=rows,
        tags=tags,
        today=date.today().isoformat(),
    )


@app.route("/transactions/add", methods=["POST"])
def add_transaction():
    form = request.form
    symbol = form.get("symbol", "").strip().upper()
    tx_type = form.get("tx_type")
    tx_date = form.get("tx_date")
    tag = form.get("tag", "").strip() or None
    note = form.get("note", "").strip() or None

    try:
        quantity = float(form.get("quantity", ""))
        price = float(form.get("price", ""))
    except ValueError:
        flash("Adet ve fiyat sayısal olmalı.", "error")
        return redirect(url_for("transactions"))

    if not symbol or tx_type not in ("buy", "sell") or not tx_date:
        flash("Sembol, işlem tipi ve tarih zorunlu.", "error")
        return redirect(url_for("transactions"))

    if quantity <= 0 or price <= 0:
        flash("Adet ve fiyat pozitif olmalı.", "error")
        return redirect(url_for("transactions"))

    total = round(quantity * price, 2)

    conn = db.get_connection()
    conn.execute(
        """INSERT INTO transactions (symbol, tx_type, tx_date, quantity, price, total, tag, note)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, tx_type, tx_date, quantity, price, total, tag, note),
    )
    conn.commit()
    conn.close()

    flash(f"{symbol} işlemi eklendi.", "success")
    return redirect(url_for("transactions"))


@app.route("/transactions/<int:tx_id>/delete", methods=["POST"])
def delete_transaction(tx_id):
    conn = db.get_connection()
    conn.execute("DELETE FROM transactions WHERE id = ?", (tx_id,))
    conn.commit()
    conn.close()

    flash("İşlem silindi.", "success")
    return redirect(url_for("transactions"))


if __name__ == "__main__":
    app.run(debug=True)
