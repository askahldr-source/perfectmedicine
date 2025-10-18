#!/usr/bin/env python3
"""
option_chain.py

Fetch option chain for a ticker using yfinance, compute mid price and Black-Scholes Greeks.

Usage:
    python option_chain.py AAPL                # all expiries, both calls and puts printed (first expiry shown by default)
    python option_chain.py AAPL 2025-11-21     # specific expiry (YYYY-MM-DD)
    python option_chain.py AAPL 2025-11-21 calls
    python option_chain.py AAPL all puts out.csv

Requires:
    pip install yfinance pandas numpy scipy

Outputs:
    - prints a DataFrame to stdout (head)
    - if an output CSV filename is provided, saves the full table
"""

import sys
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

def bs_greeks(S, K, t, r, sigma, option_type):
    # t: time to expiry in years (float)
    # sigma: volatility (annual)
    if t <= 0 or sigma <= 0:
        # degenerate case
        return dict(delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan, rho=np.nan)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    if option_type.lower().startswith('c'):
        delta = cdf_d1
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(t)) - r * K * np.exp(-r * t) * cdf_d2)
        rho = K * t * np.exp(-r * t) * cdf_d2
    else:
        delta = cdf_d1 - 1
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(t)) + r * K * np.exp(-r * t) * norm.cdf(-d2))
        rho = -K * t * np.exp(-r * t) * norm.cdf(-d2)
    gamma = pdf_d1 / (S * sigma * np.sqrt(t))
    vega = S * pdf_d1 * np.sqrt(t)
    # convert vega to per-1% vol if desired: divide by 100
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

def annualized_hist_vol(price_series, days=90):
    # annualized std dev based on log returns
    if len(price_series) < 2:
        return np.nan
    returns = np.log(price_series / price_series.shift(1)).dropna()
    if len(returns) == 0:
        return np.nan
    daily_std = returns[-days:].std()
    return daily_std * np.sqrt(252)

def fetch_and_compute(ticker, expiry_arg="all", side="both", out_csv=None, risk_free=0.05):
    tkr = yf.Ticker(ticker)
    expiries = tkr.options
    if not expiries:
        raise ValueError(f"No option expiries found for {ticker}")
    # choose expiry
    if expiry_arg.lower() in ("all", ""):
        expiries_to_fetch = expiries
    else:
        if expiry_arg not in expiries:
            raise ValueError(f"Expiry {expiry_arg} not found for {ticker}. Available: {expiries}")
        expiries_to_fetch = [expiry_arg]

    # get historical closing prices for fallback vol
    hist = tkr.history(period="1y")['Close']
    hist_vol = annualized_hist_vol(hist)

    rows = []
    S0 = tkr.history(period="1d")['Close'].iloc[-1]

    today = dt.datetime.utcnow().date()

    for expiry in expiries_to_fetch:
        chain = tkr.option_chain(expiry)
        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            if side.lower() == "calls" and kind != "call":
                continue
            if side.lower() == "puts" and kind != "put":
                continue
            # ensure expected columns present
            for col in ['strike', 'bid', 'ask', 'impliedVolatility', 'lastPrice', 'openInterest', 'volume']:
                if col not in df.columns:
                    df[col] = np.nan
            # compute days to expiry
            exp_date = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
            days = (exp_date - today).days
            t_years = max(days / 365.0, 0.0)
            for _, r in df.iterrows():
                K = float(r['strike'])
                bid = r['bid'] if not pd.isna(r['bid']) else np.nan
                ask = r['ask'] if not pd.isna(r['ask']) else np.nan
                mid = np.nan
                if not pd.isna(bid) and not pd.isna(ask):
                    mid = (bid + ask) / 2.0
                else:
                    # fallback to lastPrice if bid/ask missing
                    mid = r['lastPrice'] if not pd.isna(r['lastPrice']) else np.nan
                iv = r.get('impliedVolatility', np.nan)
                # yfinance impliedVol often is in decimal (e.g. 0.25). If it's > 1, assume percent and divide.
                if not pd.isna(iv):
                    try:
                        iv = float(iv)
                        if iv > 1.0:
                            iv = iv / 100.0
                    except Exception:
                        iv = np.nan
                if pd.isna(iv) or iv <= 0:
                    iv_used = hist_vol if not pd.isna(hist_vol) else 0.3
                else:
                    iv_used = iv
                greeks = bs_greeks(S=float(S0), K=K, t=t_years, r=risk_free, sigma=iv_used, option_type=kind)
                rows.append({
                    'expiry': expiry,
                    'type': kind,
                    'contract': r.get('contractSymbol', ''),
                    'strike': K,
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'lastPrice': r.get('lastPrice', np.nan),
                    'impliedVol': iv,
                    'vol_used': iv_used,
                    'days': days,
                    'openInterest': r.get('openInterest', np.nan),
                    'volume': r.get('volume', np.nan),
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'vega': greeks['vega'],
                    'theta': greeks['theta'],
                    'rho': greeks['rho'],
                    'underlying': float(S0)
                })
    out = pd.DataFrame(rows)
    if out_csv:
        out.to_csv(out_csv, index=False)
        print(f"Saved to {out_csv}")
    # print top rows grouped by expiry
    pd.set_option('display.max_columns', 20)
    if len(out) == 0:
        print("No options matched your filters.")
    else:
        # show top 10 rows
        print(out.sort_values(['expiry','type','strike']).head(50).to_string(index=False))
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: python option_chain.py TICKER [EXPIRY|'all'] [calls|puts|both] [out.csv]")
        sys.exit(1)
    ticker = sys.argv[1].upper()
    expiry = sys.argv[2] if len(sys.argv) >= 3 else "all"
    side = sys.argv[3] if len(sys.argv) >= 4 else "both"
    out_csv = sys.argv[4] if len(sys.argv) >= 5 else None
    try:
        df = fetch_and_compute(ticker, expiry_arg=expiry, side=side, out_csv=out_csv)
    except Exception as e:
        print("Error:", e)
        sys.exit(2)

if __name__ == "__main__":
    main()