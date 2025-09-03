import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import scipy.optimize as sco
import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

LOG_FILE = "allocations_log.csv"

app = Flask(__name__)
CORS(app)

def portfolio_performance(weights, mean_returns, cov_matrix):
   ret = np.dot(weights, mean_returns)
   vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
   sharpe = ret / vol if vol > 0 else 0
   return ret, vol, sharpe

def optimize_portfolio(returns):
   mean_returns = returns.mean()
   cov_matrix = returns.cov()
   n = len(mean_returns)
   def neg_sharpe(weights):
       r, v, s = portfolio_performance(weights, mean_returns, cov_matrix)
       return -s
   constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})
   bounds = tuple((0, 1) for _ in range(n))
   init_guess = n * [1. / n]
   result = sco.minimize(neg_sharpe, init_guess,
                         method="SLSQP",
                         bounds=bounds,
                         constraints=constraints)
   return result.x, mean_returns, cov_matrix

def run_backtest(prices, weights):
   returns = prices.pct_change().dropna()
   port_returns = returns.dot(weights)
   cumulative = (1 + port_returns).cumprod()
   total_return = cumulative.iloc[-1] - 1
   cagr = (cumulative.iloc[-1]) ** (252 / len(cumulative)) - 1
   vol = port_returns.std() * np.sqrt(252)
   sharpe = (port_returns.mean() * 252) / vol if vol > 0 else 0
   running_max = cumulative.cummax()
   drawdown = (cumulative - running_max) / running_max
   max_dd = drawdown.min()
   return {
       "total_return": total_return,
       "cagr": cagr,
       "vol": vol,
       "sharpe": sharpe,
       "max_dd": max_dd
   }

def get_price_data(tickers):
   now = datetime.now()
   weekday = now.weekday()
   if weekday >= 5:
       print(f"{YELLOW}📅 Weekend detected — using daily (1d) data fallback.{RESET}")
       interval, period = "1d", "2y"
   else:
       current_time = now.time()
       if datetime.strptime("09:30", "%H:%M").time() <= current_time <= datetime.strptime("16:00", "%H:%M").time():
           print(f"{GREEN}📈 Market open — using intraday (30m) data.{RESET}")
           interval, period = "30m", "60d"
       else:
           print(f"{YELLOW}📅 Market closed — using daily (1d) data fallback.{RESET}")
           interval, period = "1d", "2y"
   try:
       data = yf.download(tickers, interval=interval, period=period, progress=False)["Close"]
       if isinstance(data, pd.Series):
           data = data.to_frame()
       data = data.dropna()
       return data
   except Exception as e:
       print(f"{RED}Failed to download data: {e}{RESET}")
       sys.exit(1)

@app.route('/optimize', methods=['POST'])
def optimize_api():
   data = request.get_json()
   tickers = [t.strip().upper() for t in data.get('tickers', '').split(',')]
   portfolio_value = float(data.get('portfolio_value', 100000))
   prices = get_price_data(tickers)
   returns = prices.pct_change().dropna()
   weights, mean_returns, cov_matrix = optimize_portfolio(returns)
   weights = np.round(weights, 4)
   last_prices = {t: float(prices[t].iloc[-1]) for t in tickers}
   share_targets = {t: int(portfolio_value * w / last_prices[t]) for t, w in zip(tickers, weights)}
   response = {
       "timestamp": str(prices.index[-1]),
       "interval": data.get('interval', '1d'),
       "weights": {t: float(w) for t, w in zip(tickers, weights)},
       "last_prices": last_prices,
       "share_targets": share_targets
   }
   return jsonify(response)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/about")
def about():
    return """
    <html>
    <head><title>About InvestEdge</title></head>
    <body style="font-family:Inter,Arial,sans-serif;max-width:700px;margin:40px auto;padding:24px;">
      <h2>About InvestEdge</h2>
      <p>
      InvestEdge is a research-driven portfolio tool built by three high-school students at the intersection of finance and computer science. Our platform uses a Max-Sharpe (risk-adjusted return) optimization framework—supported by back-testing and transparent data—to propose diversified portfolio weights and visualize risk/return trade-offs. Our goal is to help users make more informed investment decisions through clear, accessible analytics. InvestEdge is an educational resource and does not provide financial advice.
      </p>
      <a href="/" style="color:#27e98a;">&#8592; Back to Home</a>
    </body>
    </html>
    """

if __name__ == "__main__":
   import sys
   if len(sys.argv) > 1 and sys.argv[1] == "web":
       app.run(host="0.0.0.0", port=5050, debug=True)
   else:
       main()
