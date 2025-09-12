# app.py — Coinglass Model-2 Heatmap (totals only): 12h, 48h, 72h with optional combined table
import os, math, requests, datetime as dt
from collections import defaultdict
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh

API_HOST = "https://open-api-v4.coinglass.com"
COIN_ENDPOINT = "/api/futures/liquidation/aggregated-heatmap/model2"
PAIR_ENDPOINT = "/api/futures/liquidation/heatmap/model2"
TIMEFRAMES = ["12h", "24h", "72h"]  # shown in a single column


# ---------- page ----------
st.set_page_config(page_title="Model-2 Heatmap Viewer", layout="wide")
st_autorefresh(interval=60_000, key="auto5m")  # rerun every 5 minutes

# Trim side padding
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem;}
</style>
""", unsafe_allow_html=True)

# ---------- helpers ----------
def get_api_key() -> str:
    load_dotenv()
    k = os.getenv("COINGLASS_API_KEY")
    if not k:
        st.error("Missing COINGLASS_API_KEY in .env")
        st.stop()
    return k

def try_fetch(url, headers, params):
    r = requests.get(url, headers=headers, params=params, timeout=20)
    try:
        j = r.json()
    except Exception:
        j = {}
    return r, j

def timeframe_variants(tf: str):
    """Common encodings for '12h','24h','48h','72h' etc."""
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

@st.cache_data(ttl=300)  # cache per (coin, timeframe) for 5 minutes
def fetch_coin_model2_raw(currency: str, timeframe: str):
    """COIN (aggregated) Model-2 only. Return (data, used_params, fetched_at_utc)."""
    headers = {"CG-API-KEY": get_api_key(), "accept": "application/json"}
    url = f"{API_HOST}{COIN_ENDPOINT}"
    variants = timeframe_variants(timeframe)
    param_options = []
    for v in variants:
        param_options += [
            {"currency": currency, "range": v},
            {"symbol":  currency,  "range": v},
            {"coin":    currency,  "range": v},
            {"currency": currency, "interval": v},
        ]
    last_error = None
    for params in param_options:
        r, j = try_fetch(url, headers, params)
        if r.status_code != 200:
            last_error = f"HTTP {r.status_code}: {r.text[:200]}"
            continue
        if str(j.get("code")) == "0" and "data" in j:
            return j["data"], params, dt.datetime.utcnow()
        last_error = f"code={j.get('code')} msg={j.get('msg')} (params={params})"
    raise RuntimeError(last_error or "coin endpoint failed")

@st.cache_data(ttl=300)
def fetch_pair_model2_raw(symbol_pair: str, timeframe: str):
    """PAIR Model-2 only. Return (data, used_params, fetched_at_utc)."""
    headers = {"CG-API-KEY": get_api_key(), "accept": "application/json"}
    url = f"{API_HOST}{PAIR_ENDPOINT}"
    variants = timeframe_variants(timeframe)
    param_options = []
    for v in variants:
        param_options += [
            {"symbol": symbol_pair, "range": v},
            {"symbol": symbol_pair, "interval": v},
        ]
    last_error = None
    for params in param_options:
        r, j = try_fetch(url, headers, params)
        if r.status_code != 200:
            last_error = f"HTTP {r.status_code}: {r.text[:200]}"
            continue
        if str(j.get("code")) == "0" and "data" in j:
            return j["data"], params, dt.datetime.utcnow()
        last_error = f"code={j.get('code')} msg={j.get('msg')} (params={params})"
    raise RuntimeError(last_error or "pair endpoint failed")

def default_pair_for_coin(coin: str) -> str:
    return f"{coin.upper().strip()}USDT"

def fetch_any_model2(currency: str, timeframe: str):
    """Try COIN first (with multiple range spellings); if it fails, use PAIR <COIN>USDT."""
    try:
        data, params, ts = fetch_coin_model2_raw(currency, timeframe)
        return data, params, ts, "coin"
    except Exception as e_coin:
        try:
            pair = default_pair_for_coin(currency)
            data, params, ts = fetch_pair_model2_raw(pair, timeframe)
            return data, params, ts, "pair"
        except Exception as e_pair:
            raise RuntimeError(f"{e_coin} | {e_pair}")

def get_current_price(price_candlesticks):
    if not price_candlesticks or len(price_candlesticks[-1]) < 5:
        raise RuntimeError("Missing/short price_candlesticks.")
    return float(price_candlesticks[-1][4])  # [ts,o,h,l,c,v]

def aggregate_totals_by_level(y_axis, liq_triples):
    levels = []
    for v in (y_axis or []):
        try: levels.append(float(v))
        except Exception: levels.append(float("nan"))
    if not levels: return []
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
            sums[yi] += amt
    return [{"level": lvl, "total_usd": sums.get(idx, 0.0)}
            for idx, lvl in enumerate(levels) if math.isfinite(lvl)]

def split_window(rows, price, pct):
    tol = pct / 100.0
    window = [r for r in rows if price and abs(r["level"] - price)/price <= tol]
    below  = sorted([r for r in window if r["level"] <  price], key=lambda r: price - r["level"])
    above  = sorted([r for r in window if r["level"] >  price], key=lambda r: r["level"] - price)
    return below, above

def imbalance_above_below(below, above):
    total_below = sum(r["total_usd"] for r in below)
    total_above = sum(r["total_usd"] for r in above)
    denom = total_above + total_below
    return ((total_above - total_below)/denom if denom else 0.0,
            total_above, total_below)

def fmt_compact(n: float) -> str:
    """Return 1 decimal compact suffix (K/M/B/T)."""
    n = float(n)
    absn = abs(n)
    if absn >= 1e12:
        return f"{n/1e12:.1f}T"
    if absn >= 1e9:
        return f"{n/1e9:.1f}B"
    if absn >= 1e6:
        return f"{n/1e6:.1f}M"
    if absn >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:.0f}"

def styled(df):
    # Keep 'level' as price with 2 decimals, totals compact like 6.1B
    return df.style.format({
        "level": "{:,.2f}",
        "total_usd": lambda x: fmt_compact(x)
    })

def combine_below_above(below, above):
    """Single table with a 'side' column, preserving proximity order (below nearest first, then above)."""
    rows = [{"side": "Below", **r} for r in below] + [{"side": "Above", **r} for r in above]
    return rows

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    currency   = st.text_input("Currency (coin symbol)", value="BTC").upper().strip()
    pct_win    = st.slider("± Window (%)", 0.5, 20.0, 6.0, 0.5)
    show_tables = st.checkbox("Show combined table (Below & Above)", value=False)
    if st.button("Refresh now"):
        st.cache_data.clear()

st.title("Coinglass Model-2 Heatmap")

def render_panel(timeframe: str, container):
    with container:
        try:
            data, used_params, fetched_at_utc, source_kind = fetch_any_model2(currency, timeframe)
            price = get_current_price(data.get("price_candlesticks", []))
        except Exception as e:
            st.error(f"[{timeframe}] Fetch error: {e}")
            return

        st.subheader(f"{timeframe}")
        st.metric(f"{currency} (Model-2 {timeframe})", f"${price:,.2f}")
        st.caption(f"Params: {used_params} · Last updated: {fetched_at_utc:%Y-%m-%d %H:%M:%S} UTC")

        rows = aggregate_totals_by_level(data.get("y_axis", []),
                                         data.get("liquidation_leverage_data", []))
        below, above = split_window(rows, price, pct_win)

        # Combined table
        if show_tables:
            st.markdown(f"**Levels within ±{pct_win}% (combined)**")
            combined = combine_below_above(below, above)
            df = pd.DataFrame(combined, columns=["side", "level", "total_usd"])
            st.dataframe(styled(df), use_container_width=True)

        # Metrics with compact numbers
        pos_imb, tot_above, tot_below = imbalance_above_below(below, above)
        st.markdown("**Imbalance**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Above Total", f"${fmt_compact(tot_above)}")
        m2.metric("Below Total", f"${fmt_compact(tot_below)}")
        m3.metric("Above/Below Imbalance", f"{pos_imb:.2%}")

# Layout: single column with 12h, 48h, 72h panels stacked
col = st.container()
for tf in TIMEFRAMES:
    render_panel(tf, col)
    st.divider()
