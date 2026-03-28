import numpy as np
import pandas as pd
from scipy import stats

from data_loader import STRESS_PERIODS


# ── Exception flagging ────────────────────────────────────────────────────────
# An exception occurs when the actual loss exceeds the VaR estimate.
# VaR is expressed as a positive number, so exception = return < -VaR

def compute_exceptions(returns: pd.Series, var_series: pd.Series) -> pd.Series:
    aligned = var_series.reindex(returns.index).ffill().dropna()
    r_aligned = returns.reindex(aligned.index)
    return r_aligned < -aligned


# ── Kupiec POF Test ───────────────────────────────────────────────────────────
# Tests whether the actual exception rate matches the expected rate.
# H0: exception rate = (1 - confidence level)
# Uses a likelihood ratio test, follows chi-squared distribution with 1 degree of freedom.

def kupiec_pof(exceptions: pd.Series, confidence: float = 0.99) -> dict:
    n = len(exceptions)
    x = exceptions.sum()
    p = 1 - confidence
    p_hat = x / n if n > 0 else 0.0

    eps = 1e-10
    p_hat_c = np.clip(p_hat, eps, 1 - eps)
    p_c = np.clip(p, eps, 1 - eps)

    lr = -2 * (
        x * np.log(p_c / p_hat_c) + (n - x) * np.log((1 - p_c) / (1 - p_hat_c))
    )
    p_value = 1 - stats.chi2.cdf(lr, df=1)

    return {
        "n_obs": n,
        "n_exceptions": int(x),
        "expected_exceptions": round(n * p, 1),
        "exception_rate": round(p_hat * 100, 2),
        "expected_rate": round(p * 100, 2),
        "lr_stat": round(lr, 4),
        "p_value": round(p_value, 4),
        "reject_h0": p_value < 0.05,
    }


# ── Basel Traffic Light ───────────────────────────────────────────────────────
# Basel II/III standard: count exceptions in last 250 trading days.
# Green (0-4): model acceptable
# Yellow (5-9): investigation required
# Red (10+):   model rejected, capital add-on applied

def basel_traffic_light(exceptions: pd.Series, window: int = 250) -> dict:
    recent = exceptions.iloc[-window:] if len(exceptions) >= window else exceptions
    n_exc = int(recent.sum())

    if n_exc <= 4:
        zone = "Green"
    elif n_exc <= 9:
        zone = "Yellow"
    else:
        zone = "Red"

    return {
        "n_exceptions_250d": n_exc,
        "zone": zone,
    }


# ── Full backtest across all models ──────────────────────────────────────────

def run_backtest(portfolio_returns: pd.Series, var_df: pd.DataFrame, confidence: float = 0.99) -> dict:
    results = {}
    for model in var_df.columns:
        exc = compute_exceptions(portfolio_returns, var_df[model])
        results[model] = {
            "exceptions": exc,
            "kupiec": kupiec_pof(exc, confidence),
            "basel": basel_traffic_light(exc),
        }
    return results


# ── Stress period breakdown ───────────────────────────────────────────────────

def stress_period_summary(portfolio_returns: pd.Series, var_df: pd.DataFrame, confidence: float = 0.99) -> pd.DataFrame:
    rows = []
    for period, (start, end) in STRESS_PERIODS.items():
        r_slice = portfolio_returns.loc[start:end]
        for model in var_df.columns:
            var_slice = var_df[model].reindex(r_slice.index).ffill().dropna()
            r_aligned = r_slice.reindex(var_slice.index)
            exc = (r_aligned < -var_slice).sum()
            n = len(var_slice)
            expected = n * (1 - confidence)
            rows.append({
                "Period": period,
                "Model": model,
                "N Days": n,
                "Exceptions": exc,
                "Expected": round(expected, 1),
                "Exception Rate (%)": round(exc / n * 100, 2) if n > 0 else 0,
                "Overshoot Ratio": round(exc / expected, 2) if expected > 0 else None,
            })
    return pd.DataFrame(rows)