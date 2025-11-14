import os
import json
import time
import logging
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

# ------------------------
# Config & Globals
# ------------------------

OANDA_API_URLS = {
    "practice": "https://api-fxpractice.oanda.com/v3",
    "live": "https://api-fxtrade.oanda.com/v3",
}

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

logger = logging.getLogger(__name__)


# ------------------------
# Google Sheets helpers
# ------------------------

def get_gspread_client() -> gspread.Client:
    """Authenticate to Google Sheets using a service account JSON in env GOOGLE_CREDS_JSON."""
    creds_json = os.environ["GOOGLE_CREDS_JSON"]
    info = json.loads(creds_json)
    credentials = Credentials.from_service_account_info(info, scopes=SCOPES)
    client = gspread.authorize(credentials)
    return client


def write_dataframe_to_sheet(df: pd.DataFrame, sheet_name: str, tab_name: str) -> None:
    """Create/replace the Oanda-Screener tab with the provided DataFrame."""
    gc = get_gspread_client()
    sh = gc.open(sheet_name)

    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=str(len(df) + 10), cols=str(len(df.columns) + 10))

    ws.clear()
    values = [df.columns.tolist()] + df.fillna("").astype(str).values.tolist()
    ws.update("A1", values)
    logger.info("Updated sheet '%s' tab '%s' with %d rows", sheet_name, tab_name, len(df))


# ------------------------
# Oanda helpers
# ------------------------

def get_oanda_session():
    token = os.environ["OANDA_API_TOKEN"]
    env = os.getenv("OANDA_ENV", "practice").lower()
    base_url = OANDA_API_URLS.get(env, OANDA_API_URLS["practice"])

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    return session, base_url


def fetch_instruments(session: requests.Session, base_url: str, account_id: str) -> List[Dict[str, Any]]:
    """Fetch all tradable currency instruments for the account."""
    url = f"{base_url}/accounts/{account_id}/instruments"
    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()
    instruments = data.get("instruments", [])

    fx_instruments = [
        ins for ins in instruments
        if ins.get("type") == "CURRENCY" and ins.get("tradeable", True)
    ]
    logger.info("Fetched %d FX instruments from Oanda", len(fx_instruments))
    return fx_instruments


def fetch_candles(
    session: requests.Session,
    base_url: str,
    instrument: str,
    granularity: str,
    count: int,
):
    """Fetch up to `count` candles for an instrument at a given granularity."""
    url = f"{base_url}/instruments/{instrument}/candles"
    params = {
        "granularity": granularity,
        "count": count,
        "price": "M",
    }
    resp = session.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    candles = [c for c in data.get("candles", []) if c.get("complete", False)]

    closes = [float(c["mid"]["c"]) for c in candles]
    highs = [float(c["mid"]["h"]) for c in candles]
    lows = [float(c["mid"]["l"]) for c in candles]
    volumes = [int(c.get("volume", 0)) for c in candles]

    return closes, highs, lows, volumes


# ------------------------
# Indicator helpers
# ------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    if len(series) < period + 1:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else None


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    if len(series) < slow + signal:
        return None, None, None

    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line

    return float(macd.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])


def compute_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None

    h = pd.Series(highs)
    l = pd.Series(lows)
    c = pd.Series(closes)

    prev_close = c.shift(1)

    tr = pd.concat(
        [
            (h - l),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()
    val = atr.iloc[-1]
    return float(val) if pd.notna(val) else None


# ------------------------
# Screener logic
# ------------------------

def build_instrument_row(
    session: requests.Session,
    base_url: str,
    instrument_name: str,
) -> Optional[Dict[str, Any]]:
    """Compute all metrics for a single instrument."""
    # Daily data for returns & pseudo-ATH
    daily_closes, daily_highs, daily_lows, _ = fetch_candles(
        session, base_url, instrument_name, granularity="D", count=400
    )

    # 15m data for intraday indicators
    m15_closes, m15_highs, m15_lows, m15_volumes = fetch_candles(
        session, base_url, instrument_name, granularity="M15", count=800
    )

    if not m15_closes:
        logger.warning("No 15m data for %s", instrument_name)
        return None

    last_price = m15_closes[-1]

    # ---- % down from ATH (over available daily history) ----
    pct_down_from_ath = None
    ret_1d = ret_7d = ret_14d = None

    if daily_closes:
        daily_series = pd.Series(daily_closes)
        ath = daily_series.max()
        if ath > 0:
            pct_down_from_ath = (last_price / ath - 1.0) * 100.0

        # returns over 1/7/14 days (if enough data)
        if len(daily_series) >= 2:
            ret_1d = (daily_series.iloc[-1] / daily_series.iloc[-2] - 1.0) * 100.0
        if len(daily_series) >= 8:
            ret_7d = (daily_series.iloc[-1] / daily_series.iloc[-8] - 1.0) * 100.0
        if len(daily_series) >= 15:
            ret_14d = (daily_series.iloc[-1] / daily_series.iloc[-15] - 1.0) * 100.0

    # ---- 24h volume & range ----
    # 96 * 15m candles = 24h
    lookback_24h = 96
    vol_24h = None
    range_24h_pct = None

    if len(m15_closes) >= 2:
        idx_start_24h = max(0, len(m15_closes) - lookback_24h)
        slice_highs = m15_highs[idx_start_24h:]
        slice_lows = m15_lows[idx_start_24h:]
        slice_volumes = m15_volumes[idx_start_24h:]

        vol_24h = float(np.sum(slice_volumes))
        high_24h = max(slice_highs)
        low_24h = min(slice_lows)
        if last_price > 0 and high_24h >= low_24h:
            range_24h_pct = (high_24h - low_24h) / last_price * 100.0

    # ---- RSI / MA / MACD / ATR on 15m ----
    s15 = pd.Series(m15_closes)

    rsi_14_15m = compute_rsi(s15, period=14)

    ma180 = s15.rolling(window=180, min_periods=180).mean()
    ma720 = s15.rolling(window=720, min_periods=720).mean()

    ma180_val = float(ma180.iloc[-1]) if len(s15) >= 180 and pd.notna(ma180.iloc[-1]) else None
    ma720_val = float(ma720.iloc[-1]) if len(s15) >= 720 and pd.notna(ma720.iloc[-1]) else None

    macd_val, macd_signal, macd_hist = compute_macd(s15)

    atr14_15m = compute_atr(m15_highs, m15_lows, m15_closes, period=14)

    # ---- Sparkline: last 30 closes as comma-separated string ----
    spark_n = 30
    spark_slice = m15_closes[-spark_n:]
    sparkline = ",".join(f"{p:.5f}" for p in spark_slice)

    row = {
        "pair": instrument_name,
        "last_price": last_price,
        "%_down_from_ATH": pct_down_from_ath,
        "return_1d_%": ret_1d,
        "return_7d_%": ret_7d,
        "return_14d_%": ret_14d,
        "vol_24h": vol_24h,
        "range_24h_%": range_24h_pct,
        "RSI14_15m": rsi_14_15m,
        "MA180_15m": ma180_val,
        "MA720_15m": ma720_val,
        "MACD": macd_val,
        "MACD_signal": macd_signal,
        "MACD_hist": macd_hist,
        "ATR14_15m": atr14_15m,
        "sparkline": sparkline,
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }

    return row


def run_screener_once():
    session, base_url = get_oanda_session()
    account_id = os.environ["OANDA_ACCOUNT_ID"]

    instruments = fetch_instruments(session, base_url, account_id)

    rows: List[Dict[str, Any]] = []
    for ins in instruments:
        name = ins.get("name")
        if not name:
            continue
        try:
            logger.info("Processing %s", name)
            row = build_instrument_row(session, base_url, name)
            if row:
                rows.append(row)
        except Exception as exc:
            logger.exception("Failed to build row for %s: %s", name, exc)

    if not rows:
        logger.warning("No rows built this run.")
        return

    df = pd.DataFrame(rows)

    # Keep a stable ordering
    columns = [
        "pair",
        "last_price",
        "%_down_from_ATH",
        "return_1d_%",
        "return_7d_%",
        "return_14d_%",
        "vol_24h",
        "range_24h_%",
        "RSI14_15m",
        "MA180_15m",
        "MA720_15m",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "ATR14_15m",
        "sparkline",
        "updated_at",
    ]
    df = df[columns]

    sheet_name = os.getenv("GOOGLE_SHEET_NAME", "Active-Investing")
    tab_name = os.getenv("OANDA_SCREENER_TAB", "Oanda-Screener")

    write_dataframe_to_sheet(df, sheet_name, tab_name)


# ------------------------
# Main loop
# ------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    interval_seconds = int(os.getenv("SCREENER_INTERVAL_SECONDS", "900"))  # default 15 minutes

    logger.info("Starting Oanda screener loop, interval=%ss", interval_seconds)

    while True:
        try:
            run_screener_once()
        except Exception as exc:
            logger.exception("Error in screener loop: %s", exc)
        logger.info("Sleeping for %s seconds...", interval_seconds)
        time.sleep(interval_seconds)
