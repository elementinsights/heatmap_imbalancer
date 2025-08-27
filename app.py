# app.py — Streamlit viewer (totals only) showing 12h + 24h side-by-side with auto-refresh
import os, math, requests, datetime as dt
from collections import defaultdict
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh  # <-- NEW

API_HOST = "https://open-api-v4.coinglass.com"
ENDPOINT  = "/api/futures/liquidation/aggregated-heatmap/model2"  # coin-level (all exchanges)

st.set_page_config(page_title="Model-2 Heatmap Viewer", layout="wide")

# Auto-refresh every 5 minutes (300,000 ms)
st_autorefresh(interval=300_000, key="auto5m")

# Tighten left/right padding
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

@st.cache_data(ttl=300)  # cache each timeframe for 5 minutes
def fetch_coin_model2(currency: str, timeframe: str):
    headers = {"CG-API-KEY": get_api_key(), "accept": "application/json"}
    url = f"{API_HOST}{ENDPOINT}"
    param_options = [
        {"currency": currency, "range": timeframe},
        {"symbol": currency,  "range": timeframe},
        {"coin": currency,    "range": timeframe},
        {"currency": currency, "interval": timeframe},
    ]
    last_error = None
    for params in param_options:
        r, j = try_fetch(url, headers, params)
        if r.status_code != 200:
            last_error = f"HTTP {r.status_code}: {r.text[:200]}"
            continue
        if str(j.get("code")) == "0" and "data" in j:
            return j["data"], params, dt.datetime.utcnow()
        last_error = f"code={j.get('code')} msg={j.get('msg')}"
    raise RuntimeError(last_error or "Could not fetch Model-2 heatmap")

def get_current_price(price_candlesticks):
    if not price_candlesticks or len(price_candlesticks[-1]) < 5:
        raise RuntimeError("Missing/short price_candlesticks.")
    return float(price_candlesticks[-1][4])  # [ts,o,h,l,c,v]

def aggregate_totals_by_level(y_axis, liq_triples):
    levels = []
    for v in (y_axis or []):
        try:
            levels.append(float(v))
        except Exception:
            levels.append(float("nan"))
    if not levels:
        return []

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

def styled(df):
    return df.style.format({"level": "{:,.2f}", "total_usd": "{:,.0f}"})

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    currency  = st.text_input("Currency (coin symbol)", value="BTC").upper().strip()
    pct_win   = st.slider("± Window (%)", min_value=0.5, max_value=20.0, value=6.0, step=0.5)
    if st.button("Refresh now"):
        st.cache_data.clear()

st.title("Coinglass Model-2 Heatmap — 12h & 24h (auto-refresh every 5 min)")

def render_panel(timeframe: str, col):
    with col:
        try:
            data, used_params, fetched_at_utc = fetch_coin_model2(currency, timeframe)
            price = get_current_price(data.get("price_candlesticks", []))
        except Exception as e:
            st.error(f"[{timeframe}] Fetch error: {e}")
            return

        st.subheader(f"{timeframe}")
        st.metric(f"{currency} (Model-2 {timeframe})", f"${price:,.2f}")
        st.caption(f"Params: {used_params}  ·  Last updated: {fetched_at_utc:%Y-%m-%d %H:%M:%S} UTC")

        rows = aggregate_totals_by_level(data.get("y_axis", []),
                                         data.get("liquidation_leverage_data", []))
        below, above = split_window(rows, price, pct_win)

        st.markdown(f"**Levels within ±{pct_win}%**")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Below**")
            st.dataframe(styled(pd.DataFrame(below)) if below else pd.DataFrame(columns=["level","total_usd"]),
                         use_container_width=True)
        with cc2:
            st.markdown("**Above**")
            st.dataframe(styled(pd.DataFrame(above)) if above else pd.DataFrame(columns=["level","total_usd"]),
                         use_container_width=True)

        pos_imb, tot_above, tot_below = imbalance_above_below(below, above)
        st.markdown("**Imbalance**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Above Total", f"${tot_above:,.0f}")
        m2.metric("Below Total", f"${tot_below:,.0f}")
        m3.metric("Above/Below Imbalance", f"{pos_imb:.2%}")

# Two panels side-by-side for 12h and 24h
col12h, col24h = st.columns(2)
render_panel("12h", col12h)
render_panel("24h", col24h)
