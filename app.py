import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from zoneinfo import ZoneInfo

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
MARKET_TZ = ZoneInfo("America/New_York")  # market-hours check must use ET, not server local time
RISK_FREE_ANNUAL = 0.02   # annualized risk-free rate used in every Sharpe calculation
DEFAULT_TARGET_VOL = 0.15  # fallback annualized vol target for the "max return for target risk" strategy
STRATEGIES = ("max_sharpe", "min_variance", "risk_parity", "max_return_target_risk")

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

def _parse_optional_fraction(value):
    """Parse an optional fraction in (0, 1]. Returns None if unset, the float if
    valid, or False if present but invalid (so the caller can return a 400)."""
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    if not (0.0 < f <= 1.0):
        return False
    return f

#Time & data utilities
def choose_interval_period(now: datetime) -> Tuple[str, str]:
    """Use 30m during market hours, otherwise 1d (longer history for daily). `now` must be tz-aware in MARKET_TZ."""
    weekday = now.weekday()
    if weekday >= 5:
        return "1d", "5y"
    m_open  = now.replace(hour=9, minute=30, second=0, microsecond=0).time()
    m_close = now.replace(hour=16, minute=0, second=0, microsecond=0).time()
    return ("30m", "60d") if (m_open <= now.time() <= m_close) else ("1d", "5y")

def get_price_data(tickers: List[str]) -> Tuple[pd.DataFrame, str]:
    """Download Adj Close (total return), forward-fill small gaps, drop leading NaNs."""
    interval, period = choose_interval_period(datetime.now(MARKET_TZ))
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

def fetch_spy_returns(interval: str) -> Optional[pd.Series]:
    """Download SPY once and compute its own bar-to-bar returns, for reuse across many lookback windows."""
    try:
        period = "5y" if interval == "1d" else "60d"
        spy = yf.download("SPY", interval=interval, period=period, progress=False, auto_adjust=False)["Adj Close"]
        if isinstance(spy, pd.DataFrame):  # newer yfinance returns a 1-col frame for single tickers
            spy = spy.iloc[:, 0]
        spy = spy.ffill().dropna()
        return spy.pct_change().dropna()
    except Exception:
        return None

def spy_mean_in_range(spy_rets: Optional[pd.Series], start, end) -> Optional[float]:
    """Mean of an already-downloaded SPY return series restricted to [start, end].

    Slicing SPY's own return series by date range (rather than reindexing SPY
    prices onto the asset panel's timestamp grid before differencing) avoids
    forward-filling stale prices and manufacturing artificial zero-return bars,
    which would dilute the mean toward 0.
    """
    if spy_rets is None:
        return None
    window = spy_rets[(spy_rets.index >= start) & (spy_rets.index <= end)]
    return float(window.mean()) if len(window) else None

def robust_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf covariance; fallback to sample covariance if LW not available.

    Shrinkage helps most exactly when sample size is small relative to the
    number of assets (sample covariance is then ill-conditioned/singular), so
    it must not be gated behind "enough samples" — that would skip shrinkage
    precisely when it's needed.
    """
    if _HAS_SKLEARN:
        try:
            return LedoitWolf().fit(returns.values).covariance_
        except Exception:
            pass
    return np.cov(returns.values, rowvar=False)

_SPY_RETS_UNSET = object()  # sentinel distinguishing "fetch it for me" from an explicit, already-failed None

def _blended_mu(returns: pd.DataFrame, interval: str, spy_rets) -> np.ndarray:
    """Per-bar expected returns: BLEND_ALPHA * EWMean(asset) + (1-alpha) * Mean(SPY).

    `spy_rets`: pass an already-downloaded SPY return series (or None, if a
    fetch already failed) to avoid a fresh network call on every invocation —
    important inside the walk-forward backtest. Pass the _SPY_RETS_UNSET
    sentinel to fetch fresh.
    """
    mu_ew = exp_weighted_mean(returns, halflife=EW_HALFLIFE)
    if spy_rets is _SPY_RETS_UNSET:
        spy_rets = fetch_spy_returns(interval)
    mu_spy = spy_mean_in_range(spy_rets, returns.index.min(), returns.index.max())
    if mu_spy is None:
        return mu_ew.values
    return (BLEND_ALPHA * mu_ew.values) + ((1.0 - BLEND_ALPHA) * mu_spy)

def _effective_max_weight(n_active: int, requested: Optional[float]) -> Tuple[float, bool]:
    """Clamp the per-asset cap so a long-only, fully-invested portfolio stays feasible.

    With n_active assets each capped at `cap`, weights can only sum to 1 when
    n_active * cap >= 1. If the requested cap is too tight, raise it to 1/n_active
    (equal weight). Returns (effective_cap, was_adjusted).
    """
    if requested is None or requested <= 0 or requested >= 1.0:
        return 1.0, False
    floor = 1.0 / n_active
    if requested < floor - 1e-12:
        return floor, True
    return requested, False

def _solve_weights(objective, n: int, bounds, x0: np.ndarray, extra_constraints=None) -> np.ndarray:
    """Run SLSQP for a long-only, fully-invested objective; clip + renormalize the result."""
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if extra_constraints:
        cons += extra_constraints
    res = sco.minimize(
        objective, x0, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"maxiter": 2000, "ftol": 1e-12}
    )
    w = np.array(x0, dtype=float) if ((not res.success) or np.any(np.isnan(res.x))) else np.asarray(res.x, dtype=float)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    w = np.clip(w, lo, hi)
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)

# Optimizer (long-only, fully invested, optional per-asset cap)
def optimize_portfolio(
    returns: pd.DataFrame,
    interval: str,
    strategy: str = "max_sharpe",
    max_weight: Optional[float] = None,
    target_vol: Optional[float] = None,
    spy_rets=_SPY_RETS_UNSET,
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Returns (weights, note). `note` is an optional human-readable string, e.g.
    when the max-weight cap had to be relaxed for feasibility.

    strategy:
      max_sharpe             - maximize (E[r] - rf) / vol; non-positive-mu assets
                               are excluded (pinned to 0), the rest get a floor weight
      min_variance           - global minimum-variance portfolio (ignores mu)
      risk_parity            - equal risk contribution (ignores mu)
      max_return_target_risk - maximize E[r] s.t. annualized vol <= target_vol
    """
    n = returns.shape[1]
    Sigma = robust_covariance(returns)
    bpyr = bars_per_year(returns.index)
    rf_bar = RISK_FREE_ANNUAL / bpyr
    eps = 1e-6

    def cap_note(cap: float, count: int) -> str:
        noun = "asset" if count == 1 else "assets"
        return f"Max-weight cap raised to {cap:.0%} so {count} {noun} can sum to 100%."

    # ----- strategies that ignore expected returns -----
    if strategy == "min_variance":
        cap, adj = _effective_max_weight(n, max_weight)
        bounds = tuple((0.0, cap) for _ in range(n))
        w = _solve_weights(lambda w: float(w @ Sigma @ w), n, bounds, np.full(n, 1.0 / n))
        return w, (cap_note(cap, n) if adj else None)

    if strategy == "risk_parity":
        cap, adj = _effective_max_weight(n, max_weight)
        bounds = tuple((eps, cap) for _ in range(n))  # floor keeps every risk contribution defined

        def rp_obj(w):
            contrib = w * (Sigma @ w)           # each asset's contribution to portfolio variance
            return float(np.sum((contrib - contrib.mean()) ** 2))  # 0 when all contributions are equal

        w = _solve_weights(rp_obj, n, bounds, np.full(n, 1.0 / n))
        return w, (cap_note(cap, n) if adj else None)

    # ----- strategies that depend on expected returns (mu) -----
    mu = _blended_mu(returns, interval, spy_rets)

    if strategy == "max_return_target_risk":
        cap, adj = _effective_max_weight(n, max_weight)
        bounds = tuple((0.0, cap) for _ in range(n))
        tv = target_vol if (target_vol and target_vol > 0) else DEFAULT_TARGET_VOL
        target_var_bar = (tv ** 2) / bpyr      # convert annualized vol target to per-bar variance
        cons = [{"type": "ineq", "fun": lambda w: target_var_bar - float(w @ Sigma @ w)}]
        w = _solve_weights(lambda w: -float(w @ mu), n, bounds, np.full(n, 1.0 / n), extra_constraints=cons)
        realized_vol = float(np.sqrt(max(0.0, w @ Sigma @ w)) * np.sqrt(bpyr))
        notes = []
        if adj:
            notes.append(f"max-weight cap raised to {cap:.0%} for feasibility")
        if realized_vol > tv + 1e-4:
            notes.append(f"target volatility {tv:.0%} not reachable; closest is {realized_vol:.0%}")
        return w, ("; ".join(notes).capitalize() if notes else None)

    # ----- default: max_sharpe -----
    positive = mu > 0
    n_pos = int(positive.sum())
    notes: List[str] = []
    capped = max_weight is not None and 0.0 < max_weight < 1.0

    if not capped:
        # No cap: exclude non-positive-mu assets, floor the survivors so they
        # can't collapse to 0. If nothing has positive expected return, there's
        # nothing to optimize toward.
        if n_pos == 0:
            return np.full(n, 1.0 / n), "No asset has a positive expected return; using equal weights."
        active = positive
        cap, lo = 1.0, eps
    else:
        # A per-asset cap is a diversification constraint: to have weights sum
        # to 100% with each <= cap, at least ceil(1/cap) assets must be held.
        # Honor the cap by WIDENING the active set to the best-ranked assets
        # (by expected return) rather than silently relaxing the cap. Only
        # relax the cap when there genuinely aren't enough assets to meet it.
        k_needed = int(np.ceil(1.0 / max_weight))
        if n < k_needed:
            cap = 1.0 / n  # impossible to cap below this and still total 100%
            n_active = n
            notes.append(f"only {n} assets, so the cap was relaxed to {cap:.0%}")
        else:
            cap = max_weight
            n_active = max(n_pos, k_needed)  # widen past positive-mu assets if the cap demands it
        order = np.argsort(mu)[::-1]                 # highest expected return first
        active = np.zeros(n, dtype=bool)
        active[order[:n_active]] = True
        if n_active > n_pos and n >= k_needed:
            notes.append(f"added {n_active - n_pos} lower-ranked asset(s) so no holding exceeds {cap:.0%}")
        lo = 0.0  # forced-in assets may go to 0; the sum + cap constraints still apply

    na = int(active.sum())
    bounds = tuple((lo, cap) if active[i] else (0.0, 0.0) for i in range(n))
    x0 = np.where(active, 1.0 / na, 0.0)

    def neg_sharpe(w):
        ret = float(w @ mu) - rf_bar
        vol = float(np.sqrt(max(1e-16, w @ Sigma @ w)))
        return -(ret / vol) if vol > 0 else 0.0

    w = _solve_weights(neg_sharpe, n, bounds, x0)
    return w, ("; ".join(notes).capitalize() if notes else None)

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

    w = np.array([weights_raw[t] for t in tickers], dtype=float)
    # Only tickers the optimizer actually assigned a weight to count as "held
    # positions" — require_one_each must not force a share of a ticker the
    # optimizer deliberately zeroed out.
    active = w > 0

    if require_one_each:
        min_cost = float(np.sum(prices[active]))
        if portfolio_value + 1e-9 < min_cost:
            return ({t: 0 for t in tickers},
                    f"Portfolio value (${portfolio_value:,.2f}) is less than cost of one share of each held position (${min_cost:,.2f}).")

    dollar_targets = portfolio_value * w
    frac_shares = dollar_targets / prices

    # baseline: 1 each for active positions if required, else floor
    if require_one_each:
        base = np.where(active, np.maximum(1, np.floor(frac_shares)), 0).astype(int)
    else:
        base = np.floor(frac_shares).astype(int)
    spent = float(np.sum(base * prices))
    cash_left = portfolio_value - spent

    remainders = frac_shares - base
    min_price = float(np.min(prices[active])) if active.any() else float(np.min(prices))
    while cash_left >= min_price - 1e-9:
        affordable = np.where(active & (prices <= cash_left + 1e-9))[0]
        if affordable.size == 0:
            break
        idx = affordable[np.argmax(remainders[affordable])]
        base[idx] += 1
        cash_left -= float(prices[idx])
        remainders[idx] -= 1.0

    return ({t: int(s) for t, s in zip(tickers, base)}, None)

def allocate_fractional_shares(
    tickers: List[str],
    weights_raw: Dict[str, float],
    last_prices: Dict[str, float],
    portfolio_value: float
) -> Dict[str, float]:
    # Calculate fractional shares for each ticker
    shares = {}
    for t in tickers:
        price = last_prices.get(t, 0.0)
        weight = weights_raw.get(t, 0.0)
        if price > 0:
            shares[t] = round((portfolio_value * weight) / price, 4)
        else:
            shares[t] = 0.0
    return shares

EMPTY_BACKTEST_STATS = {"total_return": None, "cagr": None, "vol": None, "sharpe": None, "max_dd": None}

def _backtest_stats(port: pd.Series, bpyr: float) -> Dict[str, float]:
    """Summary stats (total return, CAGR, vol, Sharpe, max drawdown) for a bar-return series."""
    if port.empty:
        return dict(EMPTY_BACKTEST_STATS)
    curve = (1 + port).cumprod()
    total_return = float(curve.iloc[-1] - 1.0)
    cagr = float(curve.iloc[-1] ** (bpyr / len(curve)) - 1.0) if curve.iloc[-1] > 0 else None
    vol = float(port.std() * np.sqrt(bpyr))
    # Sharpe uses an excess return over the risk-free rate: (annualized return - rf) / annualized vol.
    sharpe = float((port.mean() * bpyr - RISK_FREE_ANNUAL) / vol) if vol > 0 else 0.0
    max_dd = float(((curve - curve.cummax()) / curve.cummax()).min())
    return {"total_return": total_return, "cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": max_dd}

# In-sample fit (simple static buy & hold over the SAME window the optimizer was fit on)
def backtest_static(prices: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
    """
    NOTE: this measures fit quality, not predictive performance — the weights
    were chosen specifically to maximize Sharpe over this exact window, so
    this is mechanically guaranteed to look good. See backtest_walkforward
    for an out-of-sample estimate.
    """
    rets = prices.pct_change().dropna()
    if rets.empty:
        return dict(EMPTY_BACKTEST_STATS)
    port = pd.Series(rets.values @ weights, index=rets.index)
    bpyr = bars_per_year(rets.index)
    return _backtest_stats(port, bpyr)

# Walk-forward (out-of-sample) backtest
def backtest_walkforward(
    prices: pd.DataFrame,
    interval: str,
    strategy: str = "max_sharpe",
    max_weight: Optional[float] = None,
    target_vol: Optional[float] = None,
    min_lookback_bars: int = 90,
) -> Dict[str, object]:
    """
    At each rebalance point, fit weights using only data strictly before that
    point (no look-ahead), then hold those weights forward until the next
    rebalance. Chaining these forward-only periods into one equity curve
    gives an honest, out-of-sample estimate of how the strategy would have
    performed — unlike backtest_static, which fits and tests on the same
    window. Uses the SAME strategy and constraints the user selected, so the
    out-of-sample numbers describe the portfolio actually being recommended.
    """
    rets_full = simple_returns(prices)
    n = len(rets_full)
    if n <= min_lookback_bars:
        return {**EMPTY_BACKTEST_STATS, "rebalances": 0}

    bpyr = bars_per_year(rets_full.index)
    rebalance_every = max(5, int(round(bpyr / 12)))  # ~monthly rebalancing, regardless of bar frequency

    spy_rets = fetch_spy_returns(interval)  # fetch once, reuse across every rebalance window's fit

    segments: List[pd.Series] = []
    i = min_lookback_bars
    rebalance_count = 0
    while i < n:
        train = rets_full.iloc[:i]  # strictly past data only relative to this rebalance point
        weights, _ = optimize_portfolio(
            train, interval, strategy=strategy, max_weight=max_weight,
            target_vol=target_vol, spy_rets=spy_rets
        )
        rebalance_count += 1
        j = min(i + rebalance_every, n)
        hold = rets_full.iloc[i:j]
        segments.append(pd.Series(hold.values @ weights, index=hold.index))
        i = j

    port = pd.concat(segments) if segments else pd.Series(dtype=float)
    stats = _backtest_stats(port, bpyr)
    return {**stats, "rebalances": rebalance_count, "out_of_sample_bars": int(len(port))}

# Routes 
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/optimize", methods=["POST"])
def optimize_api():
    payload = request.get_json(force=True) or {}
    tickers = normalize_tickers(payload.get("tickers", ""))
    try:
        portfolio_value = float(payload.get("portfolio_value", 100000))
    except (TypeError, ValueError):
        return jsonify({"error": "portfolio_value must be a number."}), 400
    if portfolio_value <= 0:
        return jsonify({"error": "portfolio_value must be positive."}), 400

    strategy = (payload.get("strategy") or "max_sharpe")
    if strategy not in STRATEGIES:
        return jsonify({"error": f"Unknown strategy '{strategy}'. Expected one of {list(STRATEGIES)}."}), 400

    max_weight = _parse_optional_fraction(payload.get("max_weight"))
    if max_weight is False:
        return jsonify({"error": "max_weight must be a number between 0 and 1."}), 400
    target_vol = _parse_optional_fraction(payload.get("target_volatility"))
    if target_vol is False:
        return jsonify({"error": "target_volatility must be a number between 0 and 1."}), 400

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

    # Optimize under the selected strategy and per-asset cap.
    weights_vec, opt_note = optimize_portfolio(
        rets, interval, strategy=strategy, max_weight=max_weight, target_vol=target_vol
    )

    # Build maps aligned to requested order (0 for invalid)
    weight_map_raw = {t: 0.0 for t in tickers}
    for t, w in zip(valid, weights_vec):
        weight_map_raw[t] = float(w)
    weights_display = {t: round(weight_map_raw[t], 6) for t in tickers}

    # Last prices for ALL tickers (0.0 for invalid so .toFixed works in UI)
    last_prices = {t: (float(prices_df[t].iloc[-1]) if t in prices_df.columns else 0.0) for t in tickers}

    # Fractional shares (partial stocks)
    share_targets = allocate_fractional_shares(
        tickers, weight_map_raw, last_prices, portfolio_value
    )

    # Backtests: in-sample fit quality vs. an honest out-of-sample estimate.
    # backtest_in_sample fits and tests on the same window, so it always looks
    # good — it measures how well the optimizer fit the data, not how it would
    # have performed going forward. backtest_walkforward refits on a rolling
    # basis using only past data at each rebalance, so it's the number that
    # should actually inform a real decision.
    in_sample_bt = backtest_static(prices_df, np.array([weight_map_raw[t] for t in valid]))
    walkforward_bt = backtest_walkforward(
        prices_df, interval, strategy=strategy, max_weight=max_weight, target_vol=target_vol
    )

    response = {
        "timestamp": str(prices_df.index[-1]),
        "interval": "auto" if interval in ("30m", "1d") else interval,
        "strategy": strategy,
        "note": opt_note,
        "tickers_cleaned": tickers,
        "invalid_tickers": invalid,
        "weights": weights_display,     # for UI
        "weights_raw": weight_map_raw,  # debug detail
        "last_prices": last_prices,
        "share_targets": share_targets,
        "backtest_in_sample": in_sample_bt,
        "backtest_walkforward": walkforward_bt
    }
    return jsonify(response)

# Entrypoint (Railway binds PORT)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").strip().lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
