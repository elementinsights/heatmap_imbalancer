# check_heatmap_snapshot.py — RIGHT NOW (snapshot) liquidation viewer
# Imbalance and tables are computed from ONLY the latest x_index.

import os, math, requests, warnings
from collections import defaultdict
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

# --- Silence urllib3 warning on macOS Python compiled with LibreSSL ---
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

API_HOST = "https://open-api-v4.coinglass.com"
COIN_ENDPOINT = "/api/futures/liquidation/aggregated-heatmap/model2"
PAIR_ENDPOINT = "/api/futures/liquidation/heatmap/model2"

# -------------------- Secrets / Config --------------------
def get_api_key() -> str:
    """Prefer Streamlit secrets, then .env, then env var."""
    try:
        key = st.secrets.get("COINGLASS_API_KEY")
        if key:
            return key
    except Exception:
        pass
    load_dotenv()
    key = os.getenv("COINGLASS_API_KEY")
    if key:
        return key
    raise RuntimeError("Missing COINGLASS_API_KEY (add to .env or .streamlit/secrets.toml)")

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
    """Try COIN first; fall back to <COIN>USDT pair."""
    headers = {"CG-API-KEY": get_api_key(), "accept": "application/json"}

    # COIN endpoint
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

    # PAIR endpoint
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

# -------------------- Snapshot aggregation (RIGHT NOW) --------------------
def aggregate_totals_by_level_snapshot(y_axis, liq_triples, min_cell_usd: float = 0.0):
    """
    Use ONLY the latest time slice (max x_index). Sum abs(cell) per price level.
    """
    # levels from y_axis
    levels = []
    for v in (y_axis or []):
        try:
            levels.append(float(v))
        except Exception:
            levels.append(float("nan"))
    if not levels:
        return []

    # find latest x_index
    latest_x = None
    for row in (liq_triples or []):
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            try:
                x = int(row[0])
                latest_x = x if latest_x is None else max(latest_x, x)
            except Exception:
                pass
    if latest_x is None:
        return []

    # sum only latest column
    sums = defaultdict(float)
    for row in (liq_triples or []):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        x, yi, amt = row[:3]
        try:
            x = int(x); yi = int(yi); amt = float(amt)
        except Exception:
            continue
        if x == latest_x and 0 <= yi < len(levels) and math.isfinite(amt):
            mag = abs(amt)
            if mag >= float(min_cell_usd):
                sums[yi] += mag

    return [{"level": lvl, "total_usd": sums.get(idx, 0.0)}
            for idx, lvl in enumerate(levels) if math.isfinite(lvl)]

# -------------------- Window & binning --------------------
def within_window(rows, price, pct):
    tol = pct / 100.0
    window = [r for r in rows if price and abs(r["level"] - price)/price <= tol]
    below  = [r for r in window if r["level"] <  price]
    above  = [r for r in window if r["level"] >  price]
    return below, above

def bin_key(level, price, step_pct):
    delta_pct = (level - price) / price * 100.0
    k = math.floor(delta_pct / step_pct)
    lo = k * step_pct
    hi = (k + 1) * step_pct
    label = f"{lo:+.2f}% to {hi:+.2f}%"
    return (k, label)

def summarize_side(rows, price, step_pct):
    bins = defaultdict(float)
    counts = defaultdict(int)
    for r in rows:
        _, label = bin_key(r["level"], price, step_pct)
        bins[label] += r["total_usd"]
        counts[label] += 1

    def bin_center(lbl):
        lo, _, hi = lbl.partition(" to ")
        lo_v = float(lo.strip("%+ "))
        hi_v = float(hi.strip("%"))
        return (lo_v + hi_v) / 2.0

    picked = [(label, total, counts[label]) for label, total in bins.items()]
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

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Coinglass Liquidation Data — Snapshot", layout="wide")

with st.sidebar:
    st.title("Heatmap Checker — Snapshot")
    coin = st.text_input("Coin (symbol)", value=os.getenv("COIN", "ETH").upper().strip())
    timeframes_str = st.text_input("Timeframes (comma-separated)", value=os.getenv("TIMEFRAMES", "12h,24h,72h"))
    window_pct = st.number_input("Window ±%", min_value=0.1, max_value=20.0, value=float(os.getenv("WINDOW_PCT", 6.0)), step=0.1)
    # Controls per your spec:
    show_tables = st.checkbox("Show tables", value=False)
    show_nearest = st.checkbox("Show Nearest Big Levels", value=False)
    run_btn = st.button("Fetch Snapshot", type="primary")

st.title(f"{coin} Liquidation Data")

@st.cache_data(ttl=60)
def run_for_timeframe(tf: str, coin: str, window_pct: float, bin_step: float = 0.25):
    data, _meta = fetch_any_model2(coin, tf)
    price = get_current_price(data.get("price_candlesticks", []))
    rows  = aggregate_totals_by_level_snapshot(
        data.get("y_axis", []),
        data.get("liquidation_leverage_data", []),
        0.0  # no min cell filter (you asked to remove it)
    )
    below, above = within_window(rows, price, window_pct)

    # Totals / imbalance (snapshot within window)
    t_above = sum(r["total_usd"] for r in above)
    t_below = sum(r["total_usd"] for r in below)
    denom = (t_above + t_below) or 1.0
    imbalance = (t_above - t_below) / denom

    # Bin summaries (snapshot) for tables
    above_bins = summarize_side(above, price, bin_step)
    below_bins = summarize_side(below, price, bin_step)

    # Nearest levels by proximity (top 5 each)
    close_above = sorted(above, key=lambda r: r["level"] - price)[:5]
    close_below = sorted(below, key=lambda r: price - r["level"])[:5]

    return {
        "price": price,
        "below": below,
        "above": above,
        "t_above": t_above,
        "t_below": t_below,
        "imbalance": imbalance,
        "above_bins": above_bins,
        "below_bins": below_bins,
        "close_above": close_above,
        "close_below": close_below,
    }

if run_btn:
    try:
        tfs = [s.strip() for s in timeframes_str.split(',') if s.strip()]
        first_out = run_for_timeframe(tfs[0], coin, window_pct)
        price_ref = first_out["price"]
        lo = price_ref * (1 - window_pct/100)
        hi = price_ref * (1 + window_pct/100)

        # --- Header: one-row flexbox, space-between ---
        st.markdown(
            f"""
            <div style='font-size:18px; display:flex; justify-content:space-between; gap: 1.5rem; flex-wrap: wrap; padding-bottom: 30px;'>
              <div style=><strong>Current Price:</strong> ${price_ref:,.2f}</div>
              <div><strong>{window_pct:.2f}% Below Current Price:</strong> ${lo:,.2f}</div>
              <div><strong>{window_pct:.2f}% Above Current Price:</strong> ${hi:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Render timeframes (3 per row)
        chunks = [tfs[i:i+3] for i in range(0, len(tfs), 3)]
        for row_idx, chunk in enumerate(chunks):
            cols = st.columns(len(chunk))
            for i, tf in enumerate(chunk):
                with cols[i]:
                    try:
                        out = first_out if (row_idx == 0 and i == 0) else run_for_timeframe(tf, coin, window_pct)
                        st.subheader(tf.upper())

                        # Imbalance metric
                        st.metric("Imbalance", f"{out['imbalance']*100:+.2f}%")

                        # Totals row (flex)
                        st.markdown(
                            f"""
                            <div style='display:flex; justify-content:normal; gap: 3rem; flex-wrap: wrap; margin-bottom: 6px; padding-bottom:30px;'>
                              <div><strong>Above:</strong> {fmt_money(out['t_above'])[1:] if fmt_money(out['t_above']).startswith('$') else fmt_money(out['t_above'])}</div>
                              <div><strong>Below:</strong> {fmt_money(out['t_below'])[1:] if fmt_money(out['t_below']).startswith('$') else fmt_money(out['t_below'])}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # ---------- Bin tables (snapshot) ----------
                        if show_tables:
                            def render_bin_table(title, bins):
                                st.markdown(f"**Clusters {title}**")
                                if not bins:
                                    st.info("No data in window.")
                                    return
                                df = pd.DataFrame({
                                    "% Bin": [b[0] for b in bins],
                                    "Total (USD)": [float(b[1]) for b in bins],
                                    "Levels": [int(b[2]) for b in bins],
                                })
                                max_total = df["Total (USD)"].max()

                                def _hi(row):
                                    style = "background-color:#00ff7f;color:#000;" if row["Total (USD)"] == max_total else ""
                                    return [style] * len(row)

                                styler = (
                                    df.style
                                      .apply(_hi, axis=1)
                                      .format({"Total (USD)": lambda x: fmt_money(x)})
                                )
                                st.dataframe(styler, width="stretch")

                            render_bin_table("Above", out["above_bins"])
                            render_bin_table("Below", out["below_bins"])

                        # ---------- Nearest Big Levels ----------
                        if show_nearest and (out["close_above"] or out["close_below"]):
                            st.markdown("**Nearest Big Levels**")
                            rows = []
                            for r in out["close_above"]:
                                dpct = (r["level"] - out["price"]) / out["price"] * 100
                                rows.append({"Side":"ABOVE","Level": f"{r['level']:,.2f}", "%Δ": f"{dpct:+.3f}%", "Total (USD)": float(r['total_usd'])})
                            for r in out["close_below"]:
                                dpct = (r["level"] - out["price"]) / out["price"] * 100
                                rows.append({"Side":"BELOW","Level": f"{r['level']:,.2f}", "%Δ": f"{dpct:+.3f}%", "Total (USD)": float(r['total_usd'])})

                            df = pd.DataFrame(rows)
                            if not df.empty:
                                max_total = df["Total (USD)"].max()

                                def _hi(row):
                                    style = "background-color:#00ff7f;color:#000;" if row["Total (USD)"] == max_total else ""
                                    return [style] * len(row)

                                styler = (
                                    df.style
                                      .apply(_hi, axis=1)
                                      .format({"Total (USD)": lambda x: fmt_money(x)})
                                )
                                st.dataframe(styler, width="stretch")
                    except Exception as e:
                        st.error(f"{tf}: {e}")
    except Exception as e:
        st.error(str(e))
else:
    st.info("Configure inputs on the left, then click **Fetch Snapshot**.")
