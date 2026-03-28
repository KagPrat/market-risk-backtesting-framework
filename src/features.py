import pandas as pd
import numpy as np

FEATURE_COLS = [
    "realized_vol_21d",
    "realized_vol_63d",
    "vix_level",
    "vix_change_1d",
    "vix_change_5d",
    "abs_return_1d",
    "abs_return_5d",
    "return_skew_63d",
    "return_kurt_63d",
    "momentum_21d",
]


def build_features(portfolio_returns: pd.Series, vix: pd.Series) -> pd.DataFrame:
    r = portfolio_returns.copy()
    v = vix.copy()

    df = pd.DataFrame(index=r.index)

    # Realized volatility: annualized rolling std over 1-month and 1-quarter windows
    df["realized_vol_21d"] = r.rolling(21).std() * np.sqrt(252)
    df["realized_vol_63d"] = r.rolling(63).std() * np.sqrt(252)

    # VIX: market's implied volatility — spikes during fear/crisis periods
    df["vix_level"] = v
    df["vix_change_1d"] = v.pct_change(1)   # sudden VIX jumps signal emerging stress
    df["vix_change_5d"] = v.pct_change(5)   # weekly VIX trend

    # Recent loss magnitude — proxy for whether we're in a shock right now
    df["abs_return_1d"] = r.abs()
    df["abs_return_5d"] = r.abs().rolling(5).mean()

    # Distribution shape — fat tails and negative skew = more crash risk
    df["return_skew_63d"] = r.rolling(63).skew()
    df["return_kurt_63d"] = r.rolling(63).kurt()

    # Momentum — trend direction over past month
    df["momentum_21d"] = (1 + r).rolling(21).apply(np.prod, raw=True) - 1

    # Target: next day's absolute return (what XGBoost is trying to predict)
    df["target_abs_return"] = r.abs().shift(-1)

    return df.dropna()