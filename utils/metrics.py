"""
Evaluation metrics for financial forecasting.

Two categories:
  - Price-level metrics (MSE, RMSE, R²): affected by AIE, don't use alone
  - Honest metrics (IC, return-space R², DA, Sharpe): what quant firms use

The whole point of Part 2 is that price-level R² alone is misleading.
A zero-parameter persistence model achieves R²>0.95 on daily stock prices.
These honest metrics are what actually tell you if your model is doing anything.

Vivian Chan | 2026
"""

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score


def persistence_baseline(y_true):
    """
    Predict P_{t+1} = P_t (today's price = yesterday's price).

    If this gets R² > 0.95, any model reporting similar R² isn't
    demonstrating intelligence — it's just riding autocorrelation.
    This is the Autocorrelation Inflation Effect (AIE).
    """
    yt = y_true.flatten()
    yp = yt[:-1]
    ya = yt[1:]

    r2   = r2_score(ya, yp)
    rmse = float(np.sqrt(mean_squared_error(ya, yp)))
    da   = float(np.mean(np.sign(np.diff(ya)) == np.sign(np.diff(yp))))
    ic, _ = stats.spearmanr(np.diff(yp), np.diff(ya))

    return {
        'R2_price': round(float(r2), 4),
        'RMSE':     round(rmse, 4),
        'DirAcc':   round(da, 4),
        'IC':       round(float(ic), 4),
    }


def return_space_r2(y_true, y_pred):
    """
    R² computed on percentage returns instead of price levels.
    This eliminates autocorrelation inflation.

    Expected value is near zero — markets are largely efficient
    at daily frequency. That's not a failure, that's reality.
    """
    yt = y_true.flatten()
    yp = y_pred.flatten()
    rt = np.diff(yt) / (np.abs(yt[:-1]) + 1e-8)
    rp = np.diff(yp) / (np.abs(yp[:-1]) + 1e-8)
    return round(float(r2_score(rt, rp)), 4)


def information_coefficient(y_true, y_pred):
    """
    IC = Spearman rank correlation of predicted vs actual returns.
    This is the standard signal quality metric at quant firms.

    IC > 0.05: practically meaningful
    IC > 0.10: strong signal
    IC < 0:    model is anticorrelated (bad)
    """
    yt = y_true.flatten()
    yp = y_pred.flatten()
    rt = np.diff(yt)
    rp = np.diff(yp)
    if len(rt) < 3:
        return float('nan')
    ic, _ = stats.spearmanr(rp, rt)
    return round(float(ic), 4)


def directional_accuracy(y_true, y_pred):
    """
    Fraction of days where predicted direction = actual direction.
    50% = random. Anything below 50% means your model is
    systematically wrong (which is what we saw in Day 1 baseline).
    """
    yt = y_true.flatten()
    yp = y_pred.flatten()
    return round(float(
        np.mean(np.sign(np.diff(yt)) == np.sign(np.diff(yp)))
    ), 4)


def sharpe_ratio(y_true, y_pred, cost_bps=0):
    """
    Annualized Sharpe of a predicted-direction long/short strategy.
    cost_bps: transaction cost per trade in basis points.
    5bps = institutional, 10bps = retail, 50bps = high cost.
    """
    yt  = y_true.flatten()
    yp  = y_pred.flatten()
    sig = np.sign(np.diff(yp))
    ret = sig * np.diff(yt) / (np.abs(yt[:-1]) + 1e-8)
    chg = np.abs(np.diff(np.concatenate([[0], sig])))
    ret -= chg * cost_bps / 10000
    std  = ret.std()
    if std == 0 or np.isnan(std):
        return float('nan')
    return round(float(np.sqrt(252) * ret.mean() / (std + 1e-8)), 4)


def max_drawdown(y_pred):
    """Max drawdown of predicted price path."""
    yp   = y_pred.flatten()
    peak = np.maximum.accumulate(yp)
    dd   = (yp - peak) / (peak + 1e-8)
    return round(float(dd.min()), 4)


def leakage_inflation_ratio(part1_metric, part2_metric):
    """
    LIR = |Part1_metric / Part2_metric|

    Quantifies how much data leakage inflated Part 1 reported metrics.
    LIR > 1: Part 1 overstated performance.
    Average LIR across my experiments: ~1.84x for DA.

    Returns nan if denominator is 0 or negative.
    """
    if part2_metric is None or part2_metric == 0 or part2_metric < 0:
        return float('nan')
    return round(abs(part1_metric / part2_metric), 4)


def full_evaluate(y_true, y_pred, label='', cost_bps=0):
    """Full metric suite — both honest and price-level."""
    yt = y_true.flatten()
    yp = y_pred.flatten()

    metrics = {
        # price-level (AIE-affected — use for comparison to prior work only)
        'MSE':         round(float(mean_squared_error(yt, yp)), 4),
        'RMSE':        round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        'R2_price':    round(float(r2_score(yt, yp)), 4),
        # honest metrics
        'R2_returns':  return_space_r2(y_true, y_pred),
        'IC':          information_coefficient(y_true, y_pred),
        'DirAcc':      directional_accuracy(y_true, y_pred),
        'Sharpe':      sharpe_ratio(y_true, y_pred, cost_bps),
        'MaxDrawdown': max_drawdown(y_pred),
    }

    if label:
        print(f"\n  [{label}]")
        for k, v in metrics.items():
            note = ''
            if k == 'R2_price' and isinstance(v, float) and v > 0.95:
                note = '  <- WARNING: AIE inflation likely'
            if k == 'IC' and isinstance(v, float) and abs(v) > 0.05:
                note = '  <- meaningful signal'
            print(f"    {k:15s}: {v}{note}")

    return metrics
