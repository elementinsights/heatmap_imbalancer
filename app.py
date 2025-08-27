# file: btc_price_from_model2_12h.py
import os, sys, requests
from dotenv import load_dotenv

API_HOST = "https://open-api-v4.coinglass.com"
ENDPOINT = "/api/futures/liquidation/aggregated-heatmap/model2"  # coin-level (all exchanges)

def get_api_key():
    load_dotenv()
    key = os.getenv("COINGLASS_API_KEY")
    if not key:
        print("Missing COINGLASS_API_KEY in .env", file=sys.stderr)
        sys.exit(1)
    return key

def fetch_price_coin_model2_12h():
    headers = {"CG-API-KEY": get_api_key(), "accept": "application/json"}
    url = f"{API_HOST}{ENDPOINT}"

    # Different tenants name the coin param differently; try a few.
    # Server error you saw also asked for 'range', so we pass range=12h.
    param_options = [
        {"currency": "BTC", "range": "12h"},
        {"symbol": "BTC", "range": "12h"},
        {"coin": "BTC", "range": "12h"},
        # Some accounts use "interval" instead of "range":
        {"currency": "BTC", "interval": "12h"},
    ]

    last_err = None
    for params in param_options:
        try:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        except Exception as e:
            last_err = f"{url} {params} -> network/json error: {e}"
            continue

        if r.status_code != 200:
            last_err = f"{url} {params} -> HTTP {r.status_code}: {r.text[:200]}"
            continue

        if str(j.get("code")) != "0" or "data" not in j:
            last_err = f"{url} {params} -> code={j.get('code')} msg={j.get('msg')}"
            continue

        data = j["data"]
        candles = data.get("price_candlesticks") or []
        if not candles or len(candles[-1]) < 5:
            last_err = f"{url} {params} -> missing/short price_candlesticks"
            continue

        try:
            close_price = float(candles[-1][4])  # [ts, o, h, l, c, v]
            return close_price
        except Exception:
            last_err = f"{url} {params} -> close not parseable: {candles[-1][4]}"

    raise RuntimeError(last_err or "Could not fetch price.")

if __name__ == "__main__":
    try:
        price = fetch_price_coin_model2_12h()
        print(f"BTC (Model2 12h) price: ${price:,.2f}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
