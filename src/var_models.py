import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

from features import build_features, FEATURE_COLS


# ── 1. Parametric VaR ────────────────────────────────────────────────────────
# Assumes returns are normally distributed.
# VaR = -(mean - z * std) where z is the normal distribution quantile

def parametric_var(returns: pd.Series, confidence: float = 0.99, window: int = 252) -> pd.Series:
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std()
    z = stats.norm.ppf(1 - confidence)
    return -(mu + z * sigma).dropna()

# ── 2. Historical Simulation ─────────────────────────────────────────────────
# No distribution assumption — uses actual past returns.
# VaR = the worst (1-confidence) percentile of returns in the window

def historical_var(returns: pd.Series, confidence: float = 0.99, window: int = 252) -> pd.Series:
    return (-returns.rolling(window).quantile(1 - confidence)).dropna()


# ── 3. GARCH(1,1) ────────────────────────────────────────────────────────────
# Models volatility clustering — calm periods followed by volatile periods.
# Re-fits every `step` days for speed. Uses t-distribution for fat tails.

def garch_var(returns: pd.Series, confidence: float = 0.99, window: int = 252, step: int = 21) -> pd.Series:
    r_scaled = returns * 100  # arch library works better with scaled returns
    var_dict = {}

    indices = list(range(window, len(r_scaled), step))

    last_var = np.nan
    for i in indices:
        train = r_scaled.iloc[:i]
        try:
            model = arch_model(train, vol="Garch", p=1, q=1, dist="t", rescale=False)
            res = model.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=1, reindex=False)
            cond_vol = np.sqrt(fc.variance.values[-1, 0]) / 100
            mu = train.mean().item() / 100
            df_t = res.params.get("nu", 10)
            z = stats.t.ppf(1 - confidence, df=df_t)
            last_var = -(mu + z * cond_vol)
        except Exception:
            pass
        var_dict[r_scaled.index[i]] = last_var

    var_series = pd.Series(var_dict).sort_index()
    return var_series.reindex(returns.index).ffill().dropna()

# ── 4. ML VaR (XGBoost) ──────────────────────────────────────────────────────
# Uses engineered features (VIX, realized vol, skew etc.) to predict next day's
# absolute return. Walk-forward cross-validation respects time order.
# VaR = predicted volatility * normal distribution quantile

def ml_var(
    portfolio_returns: pd.Series,
    vix: pd.Series,
    confidence: float = 0.99,
    n_splits: int = 5,
) -> tuple[pd.Series, XGBRegressor]:
    feat_df = build_features(portfolio_returns, vix)
    X = feat_df[FEATURE_COLS]
    y = feat_df["target_abs_return"]

    z = stats.norm.ppf(confidence)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds = pd.Series(index=X.index, dtype=float)

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y.iloc[test_idx])], verbose=False)
        preds.iloc[test_idx] = model.predict(X_test)

    # Final model trained on all data for feature importance display
    final_model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    final_model.fit(X, y)

    return (preds * z).dropna(), final_model

# ── Wrapper: compute all four models ─────────────────────────────────────────

def compute_all_vars(
    portfolio_returns: pd.Series,
    vix: pd.Series,
    confidence: float = 0.99,
    garch_step: int = 21,
) -> tuple[pd.DataFrame, object]:

    print("Computing Parametric VaR...")
    p_var = parametric_var(portfolio_returns, confidence)

    print("Computing Historical Simulation VaR...")
    h_var = historical_var(portfolio_returns, confidence)

    print("Computing GARCH(1,1) VaR...")
    g_var = garch_var(portfolio_returns, confidence, step=garch_step)

    print("Computing ML VaR...")
    m_var, ml_model = ml_var(portfolio_returns, vix, confidence)

    var_df = pd.DataFrame({
        "Parametric":    p_var,
        "Historical Sim": h_var,
        "GARCH(1,1)":    g_var,
        "ML (XGBoost)":  m_var,
    }).dropna()

    return var_df, ml_model