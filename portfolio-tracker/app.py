from datetime import date

from flask import Flask, flash, redirect, render_template, request, url_for

import db

app = Flask(__name__)
app.secret_key = "portfolio-tracker-local"  # sadece flash mesajlari icin, tek kullanicili yerel araç

db.init_db()


@app.route("/")
def index():
    return redirect(url_for("transactions"))


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
