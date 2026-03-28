import yfinance as yf
import pandas as pd
import numpy as np

# Tickers
# SPYS&P 500 ETF: Core equity exposure
# GLD: Gold ETF
# TLT: 20+ Year Treasury bonds
# EEM: Emerging markets ETF
# IEF7-10 Year: Treasury bonds

TICKERS = ["SPY", "GLD", "TLT", "EEM", "IEF"]

# Weights per ticker for a typical moderate risk institutional-style portfolio 
WEIGHTS = np.array([0.40, 0.15, 0.20, 0.15, 0.10])

# Periods of stress
STRESS_PERIODS = {
    "2008 Financial Crisis": ("2008-01-01", "2009-03-31"),
    "COVID-19 Crash":        ("2020-01-01", "2020-06-30"),
    "2022 Rate Shock":       ("2022-01-01", "2022-12-31"),
}

def fetch_data(start: str = "2005-01-01", end: str = "2024-12-31") -> dict:
    tickers = TICKERS + ["^VIX"]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    raw = raw.rename(columns={"^VIX": "VIX"})

    prices = raw[TICKERS].dropna()
    asset_returns = prices.pct_change().dropna()
    portfolio_returns = asset_returns.dot(WEIGHTS)
    portfolio_returns.name = "portfolio"

    vix = raw["VIX"].reindex(asset_returns.index).ffill()

    return {
        "prices": prices,
        "asset_returns": asset_returns,
        "portfolio_returns": portfolio_returns,
        "vix": vix,
    }
