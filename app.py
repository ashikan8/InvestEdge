import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.optimize as sco
import yfinance as yf
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

try:
    from sklearn.covariance import LedoitWolf
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# ---------- Flask ----------
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ---------- Ticker normalization ----------
TICKER_FIXES = {
    "APPL": "AAPL",
    "TESLA": "TSLA",
    "GOOGLE": "GOOGL",
    "JPY": "JPY=X",
}

def normalize_tickers(raw) -> List[str]:
    if isinstance(raw, str):
        raw = raw.split(",")
    items = []
    for t in raw:
        t = (t or "").strip().upper()
        if not t:
            continue
        t = TICKER_FIXES.get(t, t)
        items.append(t)
    seen, out = set(), []
    for t in items:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

# ---------- Data utilities ----------
def choose_interval_period(now: datetime) -> Tuple[str, str]:
    weekday = now.weekday()
    if weekday >= 5:
        return "1d", "5y"
    m_open  = datetime.strptime("09:30", "%H:%M").time()
    m_close = datetime.strptime("16:00", "%H:%M").time()
    return ("30m", "60d") if (m_open <= now.time() <= m_close) else ("1d", "5y")

def get_price_data(tickers: List[str]) -> Tuple[pd.DataFrame, str]:
    interval, period = choose_interval_period(datetime.now())
    try:
        raw = yf.download(tickers, interval=interval, period=period, progress=False, auto_adjust=False)
        data = raw["Adj Close"]
        if isinstance(data, pd.Series):
            name = tickers[0] if isinstance(tickers, (list, tuple)) else str(tickers)
            data = data.to_frame(name=name)
        data = data.dropna(axis=1, how="all")
        data = data.ffill().dropna()
        return data, interval
    except Exception as e:
        print(f"[get_price_data] error: {e}", file=sys.stderr)
        return pd.DataFrame(), "unknown"

def bars_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    diffs_min = np.median(np.diff(index.values).astype("timedelta64[m]").astype(float))
    if diffs_min >= 24 * 60 - 1:
        return 252.0
    per_day = max(1, int(round((6.5 * 60) / diffs_min)))
    return 252.0 * per_day

# ---------- Returns & estimators ----------
def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

def exp_weighted_mean(returns: pd.DataFrame, halflife: int = 63) -> pd.Series:
    n = len(returns)
    if n == 0:
        return pd.Series(np.zeros(returns.shape[1]), index=returns.columns)
    decay = np.log(0.5) / halflife
    w = np.exp(decay * np.arange(n)[::-1])
    w = w / w.sum()
    return pd.Series(np.dot(w, returns.values), index=returns.columns)

def spy_mean_same_frequency(returns_index: pd.DatetimeIndex, interval: str) -> Optional[float]:
    try:
        period = "5y" if interval == "1d" else "60d"
        spy = yf.download("SPY", interval=interval, period=period, progress=False, auto_adjust=False)["Adj Close"]
        spy = spy.ffill().dropna()
        spy = spy.reindex(returns_index, method="ffill").dropna()
        rets = spy.pct_change().dropna()
        return float(rets.mean()) if len(rets) else None
    except Exception:
        return None

def robust_covariance(returns: pd.DataFrame) -> np.ndarray:
    if _HAS_SKLEARN and returns.shape[0] >= returns.shape[1]:
        try:
            return LedoitWolf().fit(returns.values).covariance_
        except Exception:
            pass
    return np.cov(returns.values, rowvar=False)

# ---------- Optimizer ----------
def optimize_sharpe_positive_only(returns: pd.DataFrame, interval: str) -> np.ndarray:
    # Stabilized expected returns
    mu_ew = exp_weighted_mean(returns, halflife=63)
    mu_spy = spy_mean_same_frequency(returns.index, interval)
    if mu_spy is None:
        mu = mu_ew.values
    else:
        mu = (0.6 * mu_ew.values) + (0.4 * mu_spy)

    Sigma = robust_covariance(returns)
    n = len(mu)

    bounds = tuple((0.0, 1.0) for _ in range(n))
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(max(1e-16, w @ Sigma @ w)))
        return -(ret / vol) if vol > 0 else 0.0

    res = sco.minimize(neg_sharpe, x0, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": 2000, "ftol": 1e-12})

    if (not res.success) or np.any(np.isnan(res.x)):
        w = x0
    else:
        w = np.clip(res.x, 0.0, 1.0)

    # Rule: if mu[i] > 0, ensure weight > 0
    for i in range(n):
        if mu[i] > 0 and w[i] <= 1e-6:
            w[i] = 1e-6  # small positive floor

    # Zero out explicitly negative-return assets
    for i in range(n):
        if mu[i] <= 0:
            w[i] = 0.0

    # Re-normalize
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = np.full(n, 1.0 / n)
    return w

# ---------- Integer shares ----------
def allocate_integer_shares(tickers: List[str],
                            weights_raw: Dict[str, float],
                            last_prices: Dict[str, float],
                            portfolio_value: float) -> Dict[str, int]:
    prices = np.array([last_prices[t] for t in tickers], dtype=float)
    w = np.array([weights_raw[t] for t in tickers], dtype=float)
    dollar_targets = portfolio_value * w
    shares = (dollar_targets / prices).astype(int)
    return {t: int(s) for t, s in zip(tickers, shares)}

# ---------- Backtests ----------
def backtest_static(prices: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
    rets = prices.pct_change().dropna()
    if rets.empty:
        return {}
    port = pd.Series(rets.values @ weights, index=rets.index)
    bpyr = bars_per_year(rets.index)
    curve = (1 + port).cumprod()
    return {
        "total_return": float(curve.iloc[-1] - 1.0),
        "cagr": float(curve.iloc[-1] ** (bpyr / len(curve)) - 1.0),
        "vol": float(port.std() * np.sqrt(bpyr)),
        "sharpe": float((port.mean() * bpyr) / (port.std() * np.sqrt(bpyr))) if port.std() > 0 else 0.0
    }

# ---------- Routes ----------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/optimize", methods=["POST"])
def optimize_api():
    payload = request.get_json(force=True) or {}
    tickers = normalize_tickers(payload.get("tickers", ""))
    portfolio_value = float(payload.get("portfolio_value", 100000))

    if not tickers:
        return jsonify({"error": "No tickers provided."}), 400

    prices_df, interval = get_price_data(tickers)
    if prices_df.empty:
        return jsonify({"error": "No price data found.", "tickers_cleaned": tickers}), 400

    valid = [t for t in tickers if t in prices_df.columns]
    invalid = [t for t in tickers if t not in prices_df.columns]
    if not valid:
        return jsonify({"error": "None of the provided tickers returned data.", "invalid_tickers": invalid}), 400

    prices_df = prices_df[valid]
    rets = simple_returns(prices_df)
    if rets.empty:
        return jsonify({"error": "Insufficient return history.", "valid_tickers": valid}), 400

    weights_vec = optimize_sharpe_positive_only(rets, interval)

    weight_map_raw = {t: 0.0 for t in tickers}
    for t, w in zip(valid, weights_vec):
        weight_map_raw[t] = float(w)
    weights_display = {t: round(weight_map_raw[t], 6) for t in tickers}

    last_prices = {t: (float(prices_df[t].iloc[-1]) if t in prices_df.columns else 0.0) for t in tickers}
    share_targets = allocate_integer_shares(tickers, weight_map_raw, last_prices, portfolio_value)

    static_bt = backtest_static(prices_df, np.array([weight_map_raw[t] for t in valid]))

    response = {
        "timestamp": str(prices_df.index[-1]),
        "interval": interval,
        "tickers_cleaned": tickers,
        "invalid_tickers": invalid,
        "weights": weights_display,
        "last_prices": last_prices,
        "share_targets": share_targets,
        "backtest_static": static_bt
    }
    return jsonify(response)

# ---------- Entrypoint ----------
def main():
    print("Run with: python app.py web")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        app.run(host="0.0.0.0", port=5050, debug=True)
    else:
        main()

