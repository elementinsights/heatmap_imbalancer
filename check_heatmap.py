# inspect_12h_window.py
import os, math, requests, datetime as dt
from collections import defaultdict
from dotenv import load_dotenv

API_HOST = "https://open-api-v4.coinglass.com"
COIN_ENDPOINT = "/api/futures/liquidation/aggregated-heatmap/model2"
PAIR_ENDPOINT = "/api/futures/liquidation/heatmap/model2"

COIN          = os.getenv("COIN", "ETH").upper().strip()
TIMEFRAME     = "12h"
WINDOW_PCT    = 1.5         # ± window around current price
BIN_STEP_PCT  = 0.25        # size of % ranges (bins)
THRESHOLD_USD = 100_000_000 # only print bins >= this total

def get_api_key():
    load_dotenv()
    k = os.getenv("COINGLASS_API_KEY")
    if not k:
        raise SystemExit("Missing COINGLASS_API_KEY in .env")
    return k

def timeframe_variants(tf: str):
    tf = tf.strip().lower()
    variants = {tf}
    if tf.endswith("h"):
        n = int(tf[:-1])
        variants |= {f"{n}hour", f"{n} hours"}
        if n % 24 == 0:
            d = n // 24
            variants |= {f"{d}d", f"{d}day", f"{d} days"}
    order = [
        "12h","24h","48h","72h","1d","2d","3d",
        "12hour","24hour","48hour","72hour",
        "12 hours","24 hours","48 hours","72 hours",
        "1day","2day","3day"
    ]
    return [v for v in order if v in variants]

def try_fetch(url, headers, params):
    r = requests.get(url, headers=headers, params=params, timeout=20)
    try:
        j = r.json()
    except Exception:
        j = {}
    return r, j

def fetch_any_model2(currency: str, timeframe: str):
    headers = {"CG-API-KEY": get_api_key(), "accept": "application/json"}

    # Try COIN endpoint with multiple param spellings
    url = f"{API_HOST}{COIN_ENDPOINT}"
    for v in timeframe_variants(timeframe):
        for params in (
            {"currency": currency, "range": v},
            {"symbol":  currency,  "range": v},
            {"coin":    currency,  "range": v},
            {"currency": currency, "interval": v},
        ):
            r, j = try_fetch(url, headers, params)
            if r.status_code == 200 and str(j.get("code")) == "0" and "data" in j:
                return j["data"], {"endpoint":"coin","params":params}

    # Fallback to PAIR <COIN>USDT
    url = f"{API_HOST}{PAIR_ENDPOINT}"
    pair = f"{currency}USDT"
    for v in timeframe_variants(timeframe):
        for params in (
            {"symbol": pair, "range": v},
            {"symbol": pair, "interval": v},
        ):
            r, j = try_fetch(url, headers, params)
            if r.status_code == 200 and str(j.get("code")) == "0" and "data" in j:
                return j["data"], {"endpoint":"pair","params":params}

    raise RuntimeError("Failed to fetch model-2 data (coin and pair endpoints).")

def get_current_price(price_candlesticks):
    if not price_candlesticks or len(price_candlesticks[-1]) < 5:
        raise RuntimeError("Missing/short price_candlesticks.")
    return float(price_candlesticks[-1][4])  # [ts,o,h,l,c,v]

def aggregate_per_level(y_axis, liq_triples):
    """Sum magnitudes per y-level (level = price bin)."""
    levels = []
    for v in (y_axis or []):
        try: levels.append(float(v))
        except Exception: levels.append(float("nan"))
    sums = defaultdict(float)
    for row in (liq_triples or []):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        _, yi, amt = row[:3]
        try:
            yi = int(yi); amt = float(amt)
        except Exception:
            continue
        if 0 <= yi < len(levels) and math.isfinite(amt):
            sums[yi] += abs(amt)  # magnitude
    out = []
    for idx, lvl in enumerate(levels):
        if math.isfinite(lvl):
            out.append({"level": lvl, "total_usd": sums.get(idx, 0.0)})
    return out

def within_window(rows, price, pct):
    tol = pct / 100.0
    window = [r for r in rows if price and abs(r["level"] - price)/price <= tol]
    below  = [r for r in window if r["level"] <  price]
    above  = [r for r in window if r["level"] >  price]
    return below, above

def bin_key(level, price, step_pct):
    """Return a labeled % range bin relative to price."""
    delta_pct = (level - price) / price * 100.0
    # snap to bins like [-1.5,-1.25), [-1.25,-1.0), ..., [0.0,0.25), ...
    # use floor for both signs
    k = math.floor(delta_pct / step_pct)
    lo = k * step_pct
    hi = (k + 1) * step_pct
    label = f"{lo:+.2f}% to {hi:+.2f}%"
    return (k, label)

def summarize_side(rows, price, step_pct, threshold):
    """Aggregate totals by % bins; return only bins >= threshold."""
    bins = defaultdict(float)
    counts = defaultdict(int)
    # sum per bin (per-level totals are already magnitudes)
    for r in rows:
        _, label = bin_key(r["level"], price, step_pct)
        bins[label] += r["total_usd"]
        counts[label] += 1
    # filter and sort by distance from price (closer first)
    def bin_center(label):
        lo, _, hi = label.partition(" to ")
        lo_v = float(lo.strip("%+ "))
        hi_v = float(hi.strip("%"))
        return (lo_v + hi_v) / 2.0
    picked = [(label, total, counts[label]) for label, total in bins.items() if total >= threshold]
    picked.sort(key=lambda x: abs(bin_center(x[0])))  # nearest first
    return picked

def fmt_money(n):
    n = float(n)
    absn = abs(n)
    if absn >= 1e12: return f"${n/1e12:.1f}T"
    if absn >= 1e9:  return f"${n/1e9:.1f}B"
    if absn >= 1e6:  return f"${n/1e6:.1f}M"
    if absn >= 1e3:  return f"${n/1e3:.1f}K"
    return f"${n:.0f}"

def main():
    data, meta = fetch_any_model2(COIN, TIMEFRAME)
    price = get_current_price(data.get("price_candlesticks", []))
    rows  = aggregate_per_level(data.get("y_axis", []), data.get("liquidation_leverage_data", []))
    below, above = within_window(rows, price, WINDOW_PCT)

    print(f"\n=== {COIN} Model-2 {TIMEFRAME} · Price = ${price:,.2f} · Window ±{WINDOW_PCT}% ===")
    print(f"Endpoint: {meta['endpoint']}  Params: {meta['params']}")
    print(f"Bin size: {BIN_STEP_PCT:.2f}%   Threshold per bin: {fmt_money(THRESHOLD_USD)}\n")

    above_bins = summarize_side(above, price, BIN_STEP_PCT, THRESHOLD_USD)
    below_bins = summarize_side(below, price, BIN_STEP_PCT, THRESHOLD_USD)

    if above_bins:
        print("ABOVE (≥ threshold):")
        for label, total, count in above_bins:
            print(f"  {label:>18}  total {fmt_money(total):>8}  ({count} level(s))")
    else:
        print("ABOVE: no bins ≥ threshold")

    if below_bins:
        print("\nBELOW (≥ threshold):")
        for label, total, count in below_bins:
            print(f"  {label:>18}  total {fmt_money(total):>8}  ({count} level(s))")
    else:
        print("\nBELOW: no bins ≥ threshold")

    # (Optional) show the single closest big levels (≥ threshold) for quick inspection
    BIG_LEVEL = THRESHOLD_USD
    close_above = sorted([r for r in above if r["total_usd"] >= BIG_LEVEL],
                         key=lambda r: (r["level"] - price))
    close_below = sorted([r for r in below if r["total_usd"] >= BIG_LEVEL],
                         key=lambda r: (price - r["level"]))
    if close_above or close_below:
        print("\nTop individual levels ≥ threshold (nearest first):")
        for r in close_above[:5]:
            dpct = (r["level"] - price) / price * 100
            print(f"  ABOVE  {r['level']:>12,.2f}  ({dpct:+.3f}%)  {fmt_money(r['total_usd'])}")
        for r in close_below[:5]:
            dpct = (r["level"] - price) / price * 100
            print(f"  BELOW  {r['level']:>12,.2f}  ({dpct:+.3f}%)  {fmt_money(r['total_usd'])}")

if __name__ == "__main__":
    main()
