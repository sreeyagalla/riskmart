"""Independent re-computation of the headline risk numbers straight from the SQL tables.

Deliberately shares NO code with riskmart.risk / the factor_changes view: changes are computed
with plain Python from factor_levels, bonds are priced by summing discounted cash flows, and the
quantiles use a hand-rolled linear-interpolation percentile.
"""
import math
import sqlite3


def _pct(sorted_vals, q):
    """Linear-interpolation quantile (same convention as numpy's default, written by hand)."""
    pos = q * (len(sorted_vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _price(y, c, T, f=2):
    n = int(round(T * f))
    return sum(100 * c / f / (1 + y / f) ** k for k in range(1, n + 1)) + 100 / (1 + y / f) ** n


def compute(db_path, cl, lookback=500):
    db = sqlite3.connect(str(db_path))
    kind = dict(db.execute("SELECT series_id, kind FROM series_meta"))
    lv = {}
    for sid, d, x in db.execute("SELECT series_id, obs_date, level FROM factor_levels ORDER BY obs_date"):
        lv.setdefault(sid, []).append((d, x))
    dates = [d for d, _ in lv["SP500"]]
    chg = {}
    for sid, rows in lv.items():
        xs = [x for _, x in rows]
        chg[sid] = [(xs[i] / xs[i - 1] - 1) if kind[sid] == "price" else (xs[i] - xs[i - 1]) / 100
                    for i in range(1, len(xs))]
    n = len(dates) - 1
    idx = range(n - lookback, n)
    last = {sid: rows[-1][1] for sid, rows in lv.items()}
    pos = db.execute("SELECT position_id, asset_class, instrument, factor_id, spread_factor_id, invert, "
                     "notional_usd, coupon, maturity_years, freq FROM positions").fetchall()

    def pos_pnl(p):
        _, _, inst, f, sf, inv, N, cpn, mat, fq = p
        out = []
        for i in idx:
            if inst == "DELTA1":
                r = chg[f][i]
                out.append(N * ((1 / (1 + r) - 1) if inv else r))
            else:
                y0 = last[f] / 100 + (last[sf] / 100 if sf else 0)
                dy = chg[f][i] + (chg[sf][i] if sf else 0)
                out.append(N * (_price(y0 + dy, cpn, mat, fq) / _price(y0, cpn, mat, fq) - 1))
        return out

    per = {p[0]: pos_pnl(p) for p in pos}
    port = [sum(per[k][j] for k in per) for j in range(lookback)]
    s = sorted(port)
    var = -_pct(s, 1 - cl)
    q = _pct(s, 1 - cl)
    tail = [x for x in port if x <= q]
    classes = {}
    for p in pos:
        classes.setdefault(p[1], []).append(p[0])
    by_class = {}
    for ac, ids in classes.items():
        v = sorted(sum(per[k][j] for k in ids) for j in range(lookback))
        by_class[ac] = -_pct(v, 1 - cl)
    wi = min(range(lookback), key=lambda j: port[j])
    mean = sum(port) / lookback
    sd = math.sqrt(sum((x - mean) ** 2 for x in port) / (lookback - 1))
    # CS01 by explicit one-bp spread bump on each credit bond
    cs01 = 0.0
    for p in pos:
        if p[4]:
            y0 = last[p[3]] / 100 + last[p[4]] / 100
            cs01 += p[6] * (_price(y0 + 1e-4, p[7], p[8], p[9]) / _price(y0, p[7], p[8], p[9]) - 1)
    return {"var_hist": var, "es_hist": -sum(tail) / len(tail), "by_class": by_class,
            "worst_pnl": port[wi], "worst_date": dates[1:][n - lookback + wi],
            "var_param_sigma": sd, "cs01": cs01,
            "exposure": {ac: sum(p[6] for p in pos if p[1] == ac) for ac in classes},
            "gross": sum(abs(p[6]) for p in pos)}
