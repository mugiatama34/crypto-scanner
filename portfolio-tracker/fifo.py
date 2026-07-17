"""FIFO (First In, First Out) eslestirme mantigi.

Bir sembole ait tum alim/satim islemlerini alir; satislari, en eski
acik alim lotundan baslayarak kapatir. Boylece hangi satisin hangi
alimi kapattigi ve gerceklesen kar/zarar net olarak hesaplanir.
"""

from collections import deque
from dataclasses import dataclass

EPSILON = 1e-9


@dataclass
class OpenLot:
    tx_id: int
    date: str
    quantity: float
    price: float


@dataclass
class ClosedMatch:
    symbol: str
    buy_tx_id: int
    buy_date: str
    buy_price: float
    sell_tx_id: int
    sell_date: str
    sell_price: float
    quantity: float

    @property
    def realized_pnl(self):
        return round((self.sell_price - self.buy_price) * self.quantity, 2)


def match_fifo(transactions):
    """transactions: tek bir sembole ait islem satirlari (dict-benzeri, sozluk
    erisimi destekleyen: t["tx_type"], t["tx_date"], t["quantity"], t["price"], t["id"]).

    Donus: (open_lots, closed_matches)
    """
    ordered = sorted(transactions, key=lambda t: (t["tx_date"], t["id"]))
    buy_queue = deque()
    closed = []

    for tx in ordered:
        if tx["tx_type"] == "buy":
            buy_queue.append(OpenLot(tx["id"], tx["tx_date"], tx["quantity"], tx["price"]))
        else:
            remaining = tx["quantity"]
            while remaining > EPSILON and buy_queue:
                lot = buy_queue[0]
                matched_qty = min(lot.quantity, remaining)
                closed.append(ClosedMatch(
                    symbol=tx["symbol"],
                    buy_tx_id=lot.tx_id,
                    buy_date=lot.date,
                    buy_price=lot.price,
                    sell_tx_id=tx["id"],
                    sell_date=tx["tx_date"],
                    sell_price=tx["price"],
                    quantity=matched_qty,
                ))
                lot.quantity -= matched_qty
                remaining -= matched_qty
                if lot.quantity <= EPSILON:
                    buy_queue.popleft()
            # remaining > 0 burada kalirsa, elde olandan fazla satis girilmis
            # demektir (veri girisi hatasi) - bu fazlalik yok sayilir.

    open_lots = [lot for lot in buy_queue if lot.quantity > EPSILON]
    return open_lots, closed
