# InvestEdge

A web application that builds optimized stock portfolios using modern portfolio theory, robust statistical estimation, and out-of-sample backtesting.

**Live app:** [investedgeus.com](https://investedgeus.com)

InvestEdge turns a list of stock tickers into an optimized, ready-to-buy portfolio. Enter your tickers and how much you want to invest, then pick a strategy. InvestEdge pulls five years of market data, estimates risk and return, and solves for the best allocation. It then shows how that portfolio would have actually performed, using an out-of-sample backtest.

---

## Features

- **Four optimization strategies.** Pick the one that matches your goal:
  | Strategy | What it does |
  |---|---|
  | **Max Sharpe** | Maximizes risk-adjusted return, (expected return − risk-free rate) ÷ volatility |
  | **Minimum Variance** | Finds the lowest-risk mix of your assets |
  | **Risk Parity** | Balances the portfolio so each asset contributes an equal share of total risk |
  | **Max Return for Target Risk** | Maximizes expected return while keeping volatility at or below a target you set |
- **Position limits.** Set a maximum weight per asset to force diversification. InvestEdge warns you when the cap is costing you return or raising your risk.
- **Ready-to-trade output.** Converts weights into exact fractional share counts for your portfolio size, using the latest prices.
- **Two kinds of backtest.**
  - *In-sample fit:* how the portfolio performed over the window it was optimized on.
  - *Walk-forward (out-of-sample):* re-optimizes monthly using **only past data**, with no look-ahead, giving a realistic estimate of how the strategy would have performed.
- **Performance metrics.** Total return, CAGR, annualized volatility, Sharpe ratio, and max drawdown.
- **Accounts and history.** Sign up with email/password or Google. Your past optimizations are saved to your account (guests keep history in the browser).
- **Smart ticker cleanup.** Normalizes input, removes duplicates, and fixes common typos (e.g. `APPL` → `AAPL`, `TESLA` → `TSLA`).

---

## How It Works

1. **Data.** Downloads five years of daily adjusted-close prices from Yahoo Finance (`yfinance`), plus SPY as the market benchmark.
2. **Expected returns.** A blend that reduces the noise of raw historical averages:
   - 60%: an **exponentially weighted mean** of each asset's returns (63-day half-life, so recent data counts more)
   - 40%: a **CAPM market prior**, rf + β·(E[market] − rf), with **Blume-adjusted betas** shrunk toward 1.0
3. **Risk.** A **Ledoit-Wolf shrinkage covariance matrix** (scikit-learn). It stays stable even when there are few observations relative to the number of assets, where a plain sample covariance breaks down.
4. **Optimization.** Each strategy is solved as a constrained problem with **SciPy's SLSQP** solver. Portfolios are long-only and fully invested, with optional per-asset caps. Infeasible inputs are handled gracefully: if a target volatility is impossible, for example, the closest feasible (minimum-risk) portfolio is returned with an explanation.
5. **Backtesting.** The walk-forward engine refits the chosen strategy roughly every 21 trading days on data strictly before each rebalance date, then chains the holding periods into one out-of-sample equity curve.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Backend | Python, Flask, Gunicorn |
| Quant / data | NumPy, pandas, SciPy, scikit-learn, yfinance |
| Database | SQLite (user accounts and saved optimization history) |
| Auth | Werkzeug password hashing, Google OAuth (Authlib) |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Railway (via `Procfile`) |

---

## Running Locally

```bash
git clone https://github.com/ashikan8/InvestEdge.git
cd InvestEdge
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:8000**.

### Optional environment variables

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Session signing key (**required in production**) |
| `DATABASE_PATH` | Where the SQLite database lives (point at a persistent volume in production) |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Enables "Sign in with Google" |
| `COOKIE_SECURE` | Set to `true` when serving over HTTPS |
| `PORT` | Port to bind (default `8000`) |

---

## API

`POST /optimize`

```json
{
  "tickers": "AAPL, MSFT, NVDA, JPM, XOM",
  "portfolio_value": 10000,
  "strategy": "max_sharpe",
  "max_weight": 0.35,
  "target_volatility": 0.15
}
```

`strategy` is one of `max_sharpe`, `min_variance`, `risk_parity`, `max_return_target_risk`.
The response includes the weights, share counts, latest prices, any notes or warnings, and both in-sample and walk-forward backtest statistics.

---

## Project Structure

```
InvestEdge/
├── app.py            # Flask API: data, estimators, optimizers, backtests, auth, history
├── index.html        # Web interface
├── requirements.txt  # Python dependencies
└── Procfile          # Deployment entrypoint
```

---

## Disclaimer

InvestEdge is an educational tool. Its results are based on historical data and statistical estimates, and past performance does not guarantee future results. This is not financial advice.
