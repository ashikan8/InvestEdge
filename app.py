import os
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

EW_HALFLIFE = 63      # bars (~3 months if daily)
BLEND_ALPHA = 0.60    # 60% asset EW mean, 40% SPY mean

# Flask 
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# Ticker normalization / common fixes 
TICKER_FIXES = {
    "APPL": "AAPL",
    "TESLA": "TSLA",
    "GOOGLE": "GOOGL",
    "JPY": "JPY=X",  # USD/JPY FX pair on Yahoo
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
    # de-duplicate preserving order
    seen, out = set(), []
    for t in items:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

#Time & data utilities 
def choose_interval_period(now: datetime) -> Tuple[str, str]:
    """Use 30m during market hours, otherwise 1d (longer history for daily)."""
    weekday = now.weekday()
    if weekday >= 5:
        return "1d", "5y"
    m_open  = datetime.strptime("09:30", "%H:%M").time()
    m_close = datetime.strptime("16:00", "%H:%M").time()
    return ("30m", "60d") if (m_open <= now.time() <= m_close) else ("1d", "5y")

def get_price_data(tickers: List[str]) -> Tuple[pd.DataFrame, str]:
    """Download Adj Close (total return), forward-fill small gaps, drop leading NaNs."""
    interval, period = choose_interval_period(datetime.now())
    try:
        raw = yf.download(tickers, interval=interval, period=period, progress=False, auto_adjust=False)
        data = raw["Adj Close"]
        if isinstance(data, pd.Series):
            # yfinance returns a Series for a single ticker; convert to DataFrame
            name = tickers[0] if isinstance(tickers, (list, tuple)) else str(tickers)
            data = data.to_frame(name=name)
        data = data.dropna(axis=1, how="all")
        data = data.ffill().dropna()
        return data, interval
    except Exception as e:
        print(f"[get_price_data] error: {e}", file=sys.stderr)
        return pd.DataFrame(), "unknown"

def bars_per_year(index: pd.DatetimeIndex) -> float:
    """Infer annualization factor from timestamp spacing (daily vs intraday)."""
    if len(index) < 3:
        return 252.0
    diffs_min = np.median(np.diff(index.values).astype("timedelta64[m]").astype(float))
    if diffs_min >= 24 * 60 - 1:  # daily or coarser
        return 252.0
    per_day = max(1, int(round((6.5 * 60) / diffs_min)))  # ~6.5 trading hours/day
    return 252.0 * per_day

# Returns & estimators
def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

def exp_weighted_mean(returns: pd.DataFrame, halflife: int = EW_HALFLIFE) -> pd.Series:
    """Exponentially-weighted mean (recency-aware) of each asset's returns."""
    n = len(returns)
    if n == 0:
        return pd.Series(np.zeros(returns.shape[1]), index=returns.columns)
    decay = np.log(0.5) / halflife
    w = np.exp(decay * np.arange(n)[::-1])
    w = w / w.sum()
    return pd.Series(np.dot(w, returns.values), index=returns.columns)

def spy_mean_same_frequency(returns_index: pd.DatetimeIndex, interval: str) -> Optional[float]:
    """Mean per-bar return of SPY aligned to the same frequency as the panel."""
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
    """Ledoit–Wolf covariance; fallback to sample covariance if LW not available."""
    if _HAS_SKLEARN and returns.shape[0] >= returns.shape[1]:
        try:
            return LedoitWolf().fit(returns.values).covariance_
        except Exception:
            pass
    return np.cov(returns.values, rowvar=False)

# Optimizer 
def optimize_max_sharpe_positive_assets(returns: pd.DataFrame, interval: str) -> np.ndarray:
    """
    Max-Sharpe using stabilized expected returns:
      - mu = BLEND_ALPHA * EWMean(asset) + (1 - BLEND_ALPHA) * Mean(SPY)
      - Sigma = Ledoit–Wolf covariance (fallback: sample)
    Rule: set weight = 0 only for assets with non-positive expected return.
          All assets with positive expected return get > 0 weight.
    """
    # Stabilized expected returns (per bar)
    mu_ew = exp_weighted_mean(returns, halflife=EW_HALFLIFE)
    mu_spy = spy_mean_same_frequency(returns.index, interval)
    if mu_spy is None:
        mu = mu_ew.values
    else:
        mu = (BLEND_ALPHA * mu_ew.values) + ((1.0 - BLEND_ALPHA) * mu_spy)

    Sigma = robust_covariance(returns)
    n = len(mu)

    # Long-only, fully invested
    bounds = tuple((0.0, 1.0) for _ in range(n))
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(max(1e-16, w @ Sigma @ w)))
        return -(ret / vol) if vol > 0 else 0.0

    res = sco.minimize(
        neg_sharpe, x0, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-12}
    )

    if (not res.success) or np.any(np.isnan(res.x)):
        w = x0.copy()
    else:
        w = np.clip(res.x, 0.0, 1.0)

    eps = 1e-6
    for i in range(n):
        if mu[i] <= 0:
            w[i] = 0.0
        elif w[i] <= eps:
            w[i] = eps

    # Re-normalize to sum to 1
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = np.full(n, 1.0 / n)

    return w

# Integer share allocation (largest remainder, ≥1 each if feasible) 
def allocate_integer_shares_largest_remainder(
    tickers: List[str],
    weights_raw: Dict[str, float],
    last_prices: Dict[str, float],
    portfolio_value: float,
    require_one_each: bool = True
) -> Tuple[Dict[str, int], Optional[str]]:
    prices = np.array([last_prices[t] for t in tickers], dtype=float)
    if np.any(prices <= 0):
        return {t: 0 for t in tickers}, "Some tickers have invalid (non-positive) prices."

    if require_one_each:
        min_cost = float(np.sum(prices))
        if portfolio_value + 1e-9 < min_cost:
            return ({t: 0 for t in tickers},
                    f"Portfolio value (${portfolio_value:,.2f}) is less than cost of one share of each (${min_cost:,.2f}).")

    w = np.array([weights_raw[t] for t in tickers], dtype=float)
    dollar_targets = portfolio_value * w
    frac_shares = dollar_targets / prices

    # baseline: 1 each if required, else floor
    base = np.maximum(1, np.floor(frac_shares)).astype(int) if require_one_each else np.floor(frac_shares).astype(int)
    spent = float(np.sum(base * prices))
    cash_left = portfolio_value - spent

    remainders = frac_shares - base
    min_price = float(np.min(prices))
    while cash_left >= min_price - 1e-9:
        affordable = np.where(prices <= cash_left + 1e-9)[0]
        if affordable.size == 0:
            break
        idx = affordable[np.argmax(remainders[affordable])]
        base[idx] += 1
        cash_left -= float(prices[idx])
        remainders[idx] -= 1.0

    return ({t: int(s) for t, s in zip(tickers, base)}, None)

# Backtest (simple static buy & hold over available window) 
def backtest_static(prices: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
    rets = prices.pct_change().dropna()
    if rets.empty:
        return {"total_return": None, "cagr": None, "vol": None, "sharpe": None, "max_dd": None}
    port = pd.Series(rets.values @ weights, index=rets.index)
    bpyr = bars_per_year(rets.index)
    curve = (1 + port).cumprod()

    total_return = float(curve.iloc[-1] - 1.0)
    cagr = float(curve.iloc[-1] ** (bpyr / len(curve)) - 1.0)
    vol = float(port.std() * np.sqrt(bpyr))
    sharpe = float((port.mean() * bpyr) / vol) if vol > 0 else 0.0
    max_dd = float(((curve - curve.cummax()) / curve.cummax()).min())
    return {"total_return": total_return, "cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": max_dd}

# Routes 
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
        return jsonify({"error": "No price data found. Check ticker symbols.", "tickers_cleaned": tickers}), 400

    valid = [t for t in tickers if t in prices_df.columns]
    invalid = [t for t in tickers if t not in prices_df.columns]
    if not valid:
        return jsonify({"error": "None of the provided tickers returned data.", "invalid_tickers": invalid}), 400

    prices_df = prices_df[valid]
    rets = simple_returns(prices_df)
    if rets.empty:
        return jsonify({"error": "Insufficient return history for optimization.", "valid_tickers": valid}), 400

    # Max-Sharpe with stabilized mu; only zero out non-positive mu 
    weights_vec = optimize_max_sharpe_positive_assets(rets, interval)

    # Build maps aligned to requested order (0 for invalid)
    weight_map_raw = {t: 0.0 for t in tickers}
    for t, w in zip(valid, weights_vec):
        weight_map_raw[t] = float(w)
    weights_display = {t: round(weight_map_raw[t], 6) for t in tickers}

    # Last prices for ALL tickers (0.0 for invalid so .toFixed works in UI)
    last_prices = {t: (float(prices_df[t].iloc[-1]) if t in prices_df.columns else 0.0) for t in tickers}

    # Integer shares (≥1 share each if feasible)
    share_targets, alloc_err = allocate_integer_shares_largest_remainder(
        tickers, weight_map_raw, last_prices, portfolio_value, require_one_each=True
    )
    if alloc_err:
        return jsonify({
            "error": alloc_err,
            "timestamp": str(prices_df.index[-1]),
            "interval": "auto" if interval in ("30m", "1d") else interval,
            "tickers_cleaned": tickers,
            "invalid_tickers": invalid,
            "weights": weights_display,
            "weights_raw": weight_map_raw,
            "last_prices": last_prices,
            "share_targets": {t: 0 for t in tickers}
        }), 400

    # Backtest
    static_bt = backtest_static(prices_df, np.array([weight_map_raw[t] for t in valid]))

    response = {
        "timestamp": str(prices_df.index[-1]),
        "interval": "auto" if interval in ("30m", "1d") else interval,
        "tickers_cleaned": tickers,
        "invalid_tickers": invalid,
        "weights": weights_display,     # for UI
        "weights_raw": weight_map_raw,  # debug detail
        "last_prices": last_prices,
        "share_targets": share_targets,
        "backtest_static": static_bt
    }
    return jsonify(response)

# Entrypoint (Railway binds PORT)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
