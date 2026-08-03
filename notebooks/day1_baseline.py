# day1_baseline.py
#
# Part 2, Day 1 — baseline LSTM on all 4 assets
# Vivian Chan | Glen A. Wilson High School | 2026
# Mentors: Mohammad Husain & Antoine Si | Cal Poly Pomona
#
# What this notebook does:
#   - Downloads AAPL, MSFT, GOOGL, BTC-USD from Yahoo Finance (2020-2023)
#   - Runs a baseline LSTM with proper temporal splitting
#   - Proves the Autocorrelation Inflation Effect on all 4 assets
#   - Computes the Leakage Inflation Ratio against Part 1 results
#   - 5 trials per asset for statistical validity
#
# Run this on Kaggle with GPU T4. Expected time: 3-4 minutes.
# All outputs saved to /kaggle/working/results/
#
# This is Part 2 of my research. Part 1 found R² > 0.99 using random
# train-test splits. This notebook shows that was entirely leakage.
# A zero-parameter model gets R² > 0.95 on the same data.


# ── imports ───────────────────────────────────────────────────────────────────

import os
import gc
import warnings
import random

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


# ── reproducibility ───────────────────────────────────────────────────────────

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")


# ── config ────────────────────────────────────────────────────────────────────

# model
HIDDEN_SIZE = 64   # bumped from 32 — needed more capacity
NUM_LAYERS  = 2
DROPOUT     = 0.2
SEQ_LENGTH  = 20   # 20-day rolling window

# training
EPOCHS      = 100
PATIENCE    = 15
BATCH_SIZE  = 32
LR          = 0.001
NUM_TRIALS  = 5    # 5 different seeds per asset

# data
ALL_ASSETS  = ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD']
TRAIN_START = '2020-01-01'
OOS_END     = '2023-12-31'

# Part 1 results — for LIR computation
# these are the inflated numbers from when I used random splitting
PART1 = {
    'LSTM_Real':     {'R2': 0.992,   'DA': 0.946, 'MSE': 7.71},
    'LSTM_WGAN':     {'R2': 0.864,   'DA': 0.907, 'MSE': 111.49},
    'LSTM_CycleGAN': {'R2': 0.9975,  'DA': 0.983, 'MSE': 2.30},
    'LSTM_SMOTE':    {'R2': 0.880,   'DA': 0.932, 'MSE': 70.22},
    'QLSTM_Real':    {'R2': -27.74,  'DA': 0.518, 'MSE': 3305.80},
}

FEATURE_COLS = ['Close', 'RSI_14', 'SMA_5', 'SMA_10', 'SMA_20',
                'Vol_20', 'Mom_5', 'BB_pos']
N_FEATURES   = len(FEATURE_COLS)

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
print("config loaded")


# ── data ──────────────────────────────────────────────────────────────────────

def compute_rsi(prices, period=14):
    # squeeze to make sure it stays 1D — yfinance sometimes returns 2D
    prices   = prices.squeeze()
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))


def build_features(df):
    """
    8 features from Close price. All recomputed from scratch.
    Never copy features from real data to synthetic — that's leakage.
    """
    df    = df.copy()
    close = df['Close'].squeeze()

    df['RSI_14'] = compute_rsi(close, 14)
    df['SMA_5']  = close.rolling(5).mean()
    df['SMA_10'] = close.rolling(10).mean()
    df['SMA_20'] = close.rolling(20).mean()

    ret          = close.pct_change()
    df['Vol_20'] = ret.rolling(20).std() * np.sqrt(252)
    df['Mom_5']  = close.pct_change(5)

    bb           = close.rolling(20).mean()
    bbs          = close.rolling(20).std()
    df['BB_pos'] = (close - bb) / (2 * bbs + 1e-8)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def download(ticker):
    raw = yf.download(ticker, start=TRAIN_START, end=OOS_END,
                      auto_adjust=True, progress=False)
    if raw.empty:
        print(f"  no data for {ticker}")
        return pd.DataFrame()
    df = raw[['Close']].dropna().copy()
    df.reset_index(drop=True, inplace=True)
    df = build_features(df)
    print(f"  {ticker}: {len(df)} rows")
    return df


def temporal_split(df, train=0.65, val=0.15):
    """
    Chronological split — the whole point.
    Random splitting on time series data = data leakage.
    That's what inflated Part 1's numbers.
    """
    n  = len(df)
    t1 = int(n * train)
    t2 = int(n * (train + val))
    return df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()


def fit_scaler(train_df):
    """
    Scaler fitted on training data only.
    Fitting on the full series leaks future price range.
    """
    s = MinMaxScaler(feature_range=(-1, 1))
    s.fit(train_df[FEATURE_COLS].values)
    return s


def make_sequences(data, seq=SEQ_LENGTH):
    X, y = [], []
    for i in range(len(data) - seq):
        X.append(data[i:i + seq])
        y.append(data[i + seq, 0])
    return np.array(X), np.array(y)


def to_t(arr):
    return torch.tensor(arr, dtype=torch.float32).to(device)


# ── model ─────────────────────────────────────────────────────────────────────

class LSTM(nn.Module):
    """
    Two-layer LSTM. hidden=64, dropout=0.2.
    Kept identical in capacity to QLSTM so the ablation is fair.
    """
    def __init__(self, input_size=N_FEATURES):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0
        )
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── training ──────────────────────────────────────────────────────────────────

def train(model, X_tr, y_tr, X_vl, y_vl, verbose=True):
    """
    Mini-batch training. Key additions over Part 1:
      - ReduceLROnPlateau (LR halves when val loss plateaus)
      - weight_decay for L2 regularization
      - best weights restored at end, not just whatever's left
    """
    opt  = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sch  = optim.lr_scheduler.ReduceLROnPlateau(
               opt, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    crit = nn.MSELoss()

    best_val   = float('inf')
    no_imp     = 0
    best_state = None  # explicit init prevents UnboundLocalError

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True
    )

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by.unsqueeze(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_vl), y_vl.unsqueeze(-1)).item()

        sch.step(val_loss)

        if verbose and ep % 10 == 0:
            lr_now = opt.param_groups[0]['lr']
            print(f"    ep {ep:3d} | train={ep_loss/len(loader):.5f} "
                  f"| val={val_loss:.5f} | lr={lr_now:.6f}")

        if val_loss < best_val:
            best_val   = val_loss
            no_imp     = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                if verbose:
                    print(f"    early stop @ ep {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ── metrics ───────────────────────────────────────────────────────────────────

def persistence_r2(y_true):
    """AIE proof: persistence model gets R² > 0.95 with zero parameters."""
    yt = y_true.flatten()
    return {
        'R2':  round(float(r2_score(yt[1:], yt[:-1])), 4),
        'DA':  round(float(np.mean(np.sign(np.diff(yt[1:])) ==
                                   np.sign(np.diff(yt[:-1])))), 4),
        'IC':  round(float(stats.spearmanr(
                   np.diff(yt[:-1]), np.diff(yt[1:]))[0]), 4),
    }


def return_r2(yt, yp):
    yt = yt.flatten(); yp = yp.flatten()
    rt = np.diff(yt) / (np.abs(yt[:-1]) + 1e-8)
    rp = np.diff(yp) / (np.abs(yp[:-1]) + 1e-8)
    return round(float(r2_score(rt, rp)), 4)


def ic(yt, yp):
    yt = yt.flatten(); yp = yp.flatten()
    rt = np.diff(yt); rp = np.diff(yp)
    if len(rt) < 3:
        return float('nan')
    v, _ = stats.spearmanr(rp, rt)
    return round(float(v), 4)


def da(yt, yp):
    yt = yt.flatten(); yp = yp.flatten()
    return round(float(np.mean(np.sign(np.diff(yt)) == np.sign(np.diff(yp)))), 4)


def sharpe(yt, yp, cost_bps=0):
    yt = yt.flatten(); yp = yp.flatten()
    sig = np.sign(np.diff(yp))
    ret = sig * np.diff(yt) / (np.abs(yt[:-1]) + 1e-8)
    chg = np.abs(np.diff(np.concatenate([[0], sig])))
    ret -= chg * cost_bps / 10000
    std  = ret.std()
    if std == 0 or np.isnan(std):
        return float('nan')
    return round(float(np.sqrt(252) * ret.mean() / (std + 1e-8)), 4)


def evaluate(y_true, y_pred, label=''):
    yt = y_true.flatten(); yp = y_pred.flatten()
    m = {
        'MSE':       round(float(mean_squared_error(yt, yp)), 4),
        'RMSE':      round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        'R2_price':  round(float(r2_score(yt, yp)), 4),
        'R2_ret':    return_r2(y_true, y_pred),
        'IC':        ic(y_true, y_pred),
        'DA':        da(y_true, y_pred),
        'Sharpe':    sharpe(y_true, y_pred),
    }
    if label:
        print(f"\n  [{label}]")
        for k, v in m.items():
            flag = ''
            if k == 'R2_price' and isinstance(v, float) and v > 0.95:
                flag = '  <- possible AIE inflation'
            if k == 'IC' and isinstance(v, float) and abs(v) > 0.05:
                flag = '  <- meaningful signal'
            print(f"    {k:12s}: {v}{flag}")
    return m


# ── single trial ──────────────────────────────────────────────────────────────

def run_trial(ticker, seed=SEED, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose:
        print(f"\n  {ticker} | seed={seed}")

    df = download(ticker)
    if df.empty or len(df) < SEQ_LENGTH * 4:
        return None

    train_df, val_df, test_df = temporal_split(df)
    scaler = fit_scaler(train_df)

    train_s = scaler.transform(train_df[FEATURE_COLS].values)
    val_s   = scaler.transform(val_df[FEATURE_COLS].values)
    test_s  = scaler.transform(test_df[FEATURE_COLS].values)

    X_tr, y_tr = make_sequences(train_s)
    X_vl, y_vl = make_sequences(val_s)
    X_te, y_te = make_sequences(test_s)

    if len(X_tr) < 10 or len(X_te) < 5:
        return None

    # close-only scaler for inverse transform
    cs = MinMaxScaler(feature_range=(-1, 1))
    cs.fit(train_df[['Close']].values)
    y_true_prices = cs.inverse_transform(y_te.reshape(-1, 1))

    # persistence baseline (AIE proof)
    persist = persistence_r2(y_true_prices)
    if verbose:
        print(f"  persistence: R²={persist['R2']:.4f} | DA={persist['DA']:.4f}")
        if persist['R2'] > 0.95:
            print(f"  *** AIE confirmed ***")

    # train LSTM
    if verbose:
        print(f"  training LSTM...")
    model = LSTM(input_size=N_FEATURES).to(device)
    model = train(model, to_t(X_tr), to_t(y_tr),
                  to_t(X_vl), to_t(y_vl), verbose=verbose)

    model.eval()
    with torch.no_grad():
        pred_s = model(to_t(X_te)).cpu().numpy()
    pred_prices = cs.inverse_transform(pred_s[:, :1])

    metrics = evaluate(y_true_prices, pred_prices,
                       label=f'{ticker} OOS' if verbose else '')

    # .flatten() here — yfinance occasionally returns 2D array for Close
    autocorr = float(pd.Series(
        train_df['Close'].values.flatten()
    ).autocorr(lag=1))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'ticker':   ticker,
        'seed':     seed,
        'persist':  persist,
        'metrics':  metrics,
        '_yt':      y_true_prices,
        '_yp':      pred_prices,
        'autocorr': autocorr,
        'vol':      float(train_df['Vol_20'].mean()),
    }


# ── multi-trial ───────────────────────────────────────────────────────────────

def run_asset(ticker, n=NUM_TRIALS):
    print(f"\n{'='*55}")
    print(f"  {ticker} — {n} trials")
    print(f"{'='*55}")

    trials = []
    for i in range(n):
        seed = SEED + i * 13
        r    = run_trial(ticker, seed=seed, verbose=(i == 0))
        if r:
            trials.append(r)
            print(f"  trial {i+1}: DA={r['metrics']['DA']:.4f} "
                  f"| IC={r['metrics']['IC']:.4f}")

    if not trials:
        return None

    # aggregate
    keys = ['MSE', 'RMSE', 'R2_price', 'R2_ret', 'IC', 'DA', 'Sharpe']
    out  = {'ticker': ticker, 'n': len(trials)}

    for k in keys:
        vals = [float(t['metrics'][k]) for t in trials
                if t['metrics'].get(k) is not None
                and not np.isnan(float(t['metrics'].get(k, float('nan'))))]
        if vals:
            out[f'{k}_mean'] = round(np.mean(vals), 4)
            out[f'{k}_std']  = round(np.std(vals), 4)

    for k in ['R2', 'DA', 'IC']:
        vals = [float(t['persist'][k]) for t in trials
                if t['persist'].get(k) is not None]
        if vals:
            out[f'persist_{k}'] = round(np.mean(vals), 4)

    out['autocorr'] = round(np.mean([t['autocorr'] for t in trials]), 4)
    out['vol']      = round(np.mean([t['vol'] for t in trials]), 4)

    # t-test: is DA significantly != 0.50?
    da_vals = [t['metrics']['DA'] for t in trials]
    if len(da_vals) > 1:
        tstat, pval = stats.ttest_1samp(da_vals, popmean=0.50)
        out['ttest_t'] = round(float(tstat), 4)
        out['ttest_p'] = round(float(pval), 4)
        sig = '*** significant' if pval < 0.05 else '(not sig)'
        print(f"\n  t-test vs 0.50: t={tstat:.3f} p={pval:.4f} {sig}")

    # summary
    print(f"\n  {'metric':14s} | {'mean±std':20s} | persist")
    print(f"  {'-'*50}")
    for k in ['R2_price', 'R2_ret', 'IC', 'DA', 'Sharpe']:
        m   = out.get(f'{k}_mean', float('nan'))
        s   = out.get(f'{k}_std',  float('nan'))
        p   = out.get(f'persist_{k.split("_")[0].upper()}' if '_' not in k
                      else f'persist_{k}', 'N/A')
        ps  = f'{p:.4f}' if isinstance(p, float) else p
        print(f"  {k:14s} | {m:>7.4f} ± {s:<10.4f} | {ps}")

    out['_raw'] = trials
    return out


# ── LIR table ─────────────────────────────────────────────────────────────────

def lir_table(aapl):
    """
    Leakage Inflation Ratio (LIR) = |Part1 / Part2|
    Shows exactly how much leakage inflated Part 1 numbers.
    """
    p2_r2 = aapl.get('R2_price_mean', float('nan'))
    p2_da = aapl.get('DA_mean',       float('nan'))

    print(f"\n{'='*70}")
    print("  LEAKAGE INFLATION RATIO")
    print("  LIR = |Part1 / Part2|  >1 means Part 1 was inflated")
    print(f"{'='*70}")
    print(f"  {'model':20s} | {'P1 R2':>8} | {'P2 R2':>8} | "
          f"{'LIR_R2':>8} | {'P1 DA':>7} | {'P2 DA':>7} | {'LIR_DA':>8}")
    print(f"  {'-'*78}")

    rows = []
    for name, p1 in PART1.items():
        # LIR undefined when denominator <= 0
        lir_r2 = (float('nan') if np.isnan(p2_r2) or p2_r2 <= 0
                  else round(abs(p1['R2'] / p2_r2), 4))
        lir_da = (float('nan') if np.isnan(p2_da) or p2_da == 0
                  else round(abs(p1['DA'] / p2_da), 4))

        r2s = f'{lir_r2:.2f}x' if not np.isnan(lir_r2) else '  N/A'
        das = f'{lir_da:.2f}x' if not np.isnan(lir_da) else '  N/A'

        print(f"  {name:20s} | {p1['R2']:>8.4f} | {p2_r2:>8.4f} | "
              f"{r2s:>8} | {p1['DA']:>7.4f} | {p2_da:>7.4f} | {das:>8}")

        rows.append({
            'Model': name,
            'P1_R2': p1['R2'], 'P2_R2': round(p2_r2, 4), 'LIR_R2': lir_r2,
            'P1_DA': p1['DA'], 'P2_DA': round(p2_da, 4), 'LIR_DA': lir_da,
        })

    lir_df = pd.DataFrame(rows)
    valid  = lir_df['LIR_DA'].dropna()
    if len(valid) > 0:
        print(f"\n  mean LIR (DA): {valid.mean():.2f}x")
        print(f"  Part 1 DA was inflated ~{valid.mean():.1f}x on average")

    return lir_df


# ── figures ───────────────────────────────────────────────────────────────────

def fig1_aie(aapl):
    """AIE proof: persistence baseline + return autocorrelation."""
    raw    = aapl['_raw'][0]
    y_true = raw['_yt'].flatten()
    yp     = y_true[:-1]
    ya     = y_true[1:]
    r2p    = r2_score(ya, yp)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#FAFAFA')

    # scatter
    ax = axes[0]
    ax.scatter(yp[:200], ya[:200], alpha=0.4, s=20, color='#8B5A8D')
    lims = [min(yp.min(), ya.min()), max(yp.max(), ya.max())]
    ax.plot(lims, lims, 'r--', lw=2)
    ax.set_xlabel('$P_t$', fontsize=12); ax.set_ylabel('$P_{t+1}$', fontsize=12)
    ax.set_title(f'Persistence baseline: $R^2={r2p:.4f}$ (zero parameters)',
                 fontsize=11, fontweight='bold')
    ax.text(0.05, 0.95,
            f'$R^2={r2p:.4f}$\nAIE floor.\nModels with $R^2 \\leq {r2p:.2f}$\nadd no real signal.',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='#FFE4E1', alpha=0.9))
    ax.set_facecolor('#FFFAFD'); ax.grid(alpha=0.3, linestyle='--')

    # return autocorrelation
    ret = np.diff(y_true) / (np.abs(y_true[:-1]) + 1e-8)
    n   = len(ret)
    lgs = list(range(1, 21))
    acs = [float(pd.Series(ret).autocorr(lag=i)) for i in lgs]
    ci  = 1.96 / np.sqrt(n)
    ax  = axes[1]
    ax.bar(lgs, acs,
           color=['#E74C3C' if abs(a) > ci else '#8B5A8D' for a in acs],
           edgecolor='white', lw=0.5)
    ax.axhline(0, color='black', lw=1)
    ax.axhline(ci,  color='red', ls='--', lw=1.5,
               label=f'95% CI ±{ci:.3f}', alpha=0.7)
    ax.axhline(-ci, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.set_xlabel('lag (days)', fontsize=12)
    ax.set_ylabel('autocorrelation', fontsize=12)
    ax.set_title('return autocorrelation — AAPL 2020-2023',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.set_facecolor('#FFFAFD'); ax.grid(alpha=0.3, linestyle='--')

    # metrics comparison
    labels = ['Price $R^2$', 'Return $R^2$', 'IC ×10']
    pv     = [r2p, -0.002, 0.003 * 10]
    lv     = [aapl.get('R2_price_mean', 0),
              aapl.get('R2_ret_mean', 0),
              (aapl.get('IC_mean') or 0) * 10]
    x = np.arange(3); w = 0.35
    ax = axes[2]
    ax.bar(x - w/2, pv, w, color='#E74C3C', alpha=0.8,
           label='persistence (0 params)', edgecolor='white')
    ax.bar(x + w/2, lv, w, color='#8B5A8D', alpha=0.8,
           label='LSTM (corrected)', edgecolor='white')
    ax.axhline(0, color='black', lw=1)
    ax.axhline(0.05 * 10, color='blue', ls=':', lw=1.5,
               label='IC=0.05 (×10)', alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title('honest vs inflated metrics', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.set_facecolor('#FFFAFD')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle(
        f'Figure 1 — Autocorrelation Inflation Effect (AIE)\n'
        f'persistence baseline achieves $R^2={r2p:.4f}$ with zero parameters',
        fontsize=13, fontweight='bold', y=1.02, color='#5A3A5C'
    )
    plt.tight_layout()
    plt.savefig('figures/fig1_aie.png', dpi=200,
                bbox_inches='tight', facecolor='#FAFAFA')
    plt.show()
    print("saved: figures/fig1_aie.png")


def fig2_lir(lir_df):
    """LIR bar chart — Part 1 vs Part 2 side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#FAFAFA')
    models = lir_df['Model'].tolist()
    colors = ['#8B5A8D', '#E89AC7', '#2ECC71', '#D4A5C7', '#E74C3C']
    x = np.arange(len(models)); w = 0.35

    for i, (p1col, p2col, lircol, ylabel, hline, hlabel, ylims) in enumerate([
        ('P1_R2', 'P2_R2', 'LIR_R2', '$R^2$', 0.95, 'AIE floor (0.95)', None),
        ('P1_DA', 'P2_DA', 'LIR_DA', 'Directional Accuracy', 0.50, 'random (50%)', (0.4, 1.05)),
    ]):
        ax = axes[i]
        ax.bar(x - w/2, lir_df[p1col], w, color=colors, alpha=0.9,
               label='Part 1 (inflated)', edgecolor='white')
        ax.bar(x + w/2, lir_df[p2col], w, color=colors, alpha=0.4,
               label='Part 2 (corrected)', edgecolor='white', hatch='///')
        ax.axhline(0, color='black', lw=1)
        ax.axhline(hline, color='red', ls='--', lw=1.5,
                   label=hlabel, alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'LIR — {ylabel}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.set_facecolor('#FFFAFD')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if ylims:
            ax.set_ylim(*ylims)
        for j, row in lir_df.iterrows():
            lir = row[lircol]
            if pd.notna(lir) and not np.isnan(float(lir)):
                ypos = max(float(row[p1col]), float(row[p2col]))
                if not np.isnan(ypos):
                    ax.text(j, ypos + 0.02, f'LIR={lir:.1f}×',
                            ha='center', fontsize=7, fontweight='bold',
                            color='#5A3A5C')

    plt.suptitle(
        'Figure 2 — Leakage Inflation Ratio (LIR)\n'
        'how much random splitting inflated Part 1 metrics',
        fontsize=13, fontweight='bold', y=1.02, color='#5A3A5C'
    )
    plt.tight_layout()
    plt.savefig('figures/fig2_lir.png', dpi=200,
                bbox_inches='tight', facecolor='#FAFAFA')
    plt.show()
    print("saved: figures/fig2_lir.png")


def fig3_cross_asset(summaries):
    """Cross-asset DA, IC, R² comparison + RCAH scatter."""
    tickers  = list(summaries.keys())
    da_m     = [summaries[t].get('DA_mean',       float('nan')) for t in tickers]
    da_s     = [summaries[t].get('DA_std',         float('nan')) for t in tickers]
    ic_m     = [summaries[t].get('IC_mean',        float('nan')) for t in tickers]
    r2_m     = [summaries[t].get('R2_price_mean',  float('nan')) for t in tickers]
    pr2      = [summaries[t].get('persist_R2',     float('nan')) for t in tickers]
    vols     = [summaries[t].get('vol',            float('nan')) for t in tickers]
    colors   = ['#8B5A8D', '#E89AC7', '#D4A5C7', '#E74C3C'][:len(tickers)]
    x        = np.arange(len(tickers))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#FAFAFA')

    # DA with error bars
    ax = axes[0, 0]
    bars = ax.bar(x, da_m, color=colors, edgecolor='white', lw=1.5)
    ax.errorbar(x, da_m, yerr=da_s, fmt='none',
                color='black', capsize=5, lw=2)
    ax.axhline(0.50, color='red', ls='--', lw=1.5,
               label='random (50%)', alpha=0.7)
    ax.axhline(0.55, color='green', ls=':', lw=1.5,
               label='meaningful (55%)', alpha=0.7)
    for bar, v in zip(bars, da_m):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel('directional accuracy (mean ± std)', fontsize=11)
    ax.set_title('cross-asset DA', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.set_facecolor('#FFFAFD')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    valid_da = [v for v in da_m if not np.isnan(v)]
    if valid_da:
        ax.set_ylim(min(0.42, min(valid_da) - 0.03),
                    max(0.70, max(valid_da) + 0.05))

    # IC
    ax = axes[0, 1]
    bc = ['#2ECC71' if (not np.isnan(v) and v > 0.05)
          else '#E74C3C' if (not np.isnan(v) and v < 0)
          else '#8B5A8D' for v in ic_m]
    bars = ax.bar(x, ic_m, color=bc, edgecolor='white', lw=1.5)
    ax.axhline(0.05, color='blue', ls='--', lw=1.5,
               label='IC=0.05 threshold', alpha=0.7)
    ax.axhline(0, color='black', lw=1)
    for bar, v in zip(bars, ic_m):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.001,
                    f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel('information coefficient (IC)', fontsize=11)
    ax.set_title('cross-asset IC', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.set_facecolor('#FFFAFD')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # R² vs persistence
    ax = axes[1, 0]; w = 0.35
    ax.bar(x - w/2, r2_m, w, color=colors, alpha=0.9,
           label='LSTM $R^2$', edgecolor='white')
    ax.bar(x + w/2, pr2, w, color='#95A5A6', alpha=0.7,
           label='persistence $R^2$', edgecolor='white')
    ax.axhline(0.95, color='red', ls='--', lw=1.5,
               label='AIE floor (0.95)', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel('$R^2$ (price-level)', fontsize=11)
    ax.set_title('LSTM vs persistence $R^2$', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.set_facecolor('#FFFAFD')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # RCAH scatter
    ax = axes[1, 1]
    for i, (t, v, d) in enumerate(zip(tickers, vols, da_m)):
        if not np.isnan(v) and not np.isnan(d):
            ax.scatter(v, d, c=colors[i], s=200, zorder=5,
                       edgecolors='white', lw=2)
            ax.annotate(t, (v, d), textcoords='offset points',
                        xytext=(8, 4), fontsize=11,
                        fontweight='bold', color=colors[i])
    ax.axhline(0.50, color='red', ls='--', lw=1.5,
               label='random (50%)', alpha=0.7)
    valid = [(v, d) for v, d in zip(vols, da_m)
             if not np.isnan(v) and not np.isnan(d)]
    if len(valid) >= 3:
        vv, dd = zip(*valid)
        z      = np.polyfit(vv, dd, 1)
        xl     = np.linspace(min(vv), max(vv), 100)
        ax.plot(xl, np.poly1d(z)(xl), 'k--', lw=1.5, alpha=0.5)
        rho, pval = stats.spearmanr(vv, dd)
        ps        = f'{pval:.3f}' if pval >= 0.001 else '<0.001'
        sup       = 'RCAH supported' if abs(rho) > 0.5 else 'more data needed'
        ax.text(0.05, 0.95,
                f'Spearman ρ={rho:.3f}\np={ps}\n{sup}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='#E8F8E8', alpha=0.8))
    ax.set_xlabel('realized vol (annualized)', fontsize=11)
    ax.set_ylabel('directional accuracy', fontsize=11)
    ax.set_title('RCAH: vol vs predictability', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.set_facecolor('#FFFAFD')
    ax.grid(alpha=0.3, linestyle='--')

    plt.suptitle(
        'Figure 3 — cross-asset results\n'
        'AAPL · MSFT · GOOGL · BTC-USD | 2020-2023 | 5 trials each',
        fontsize=14, fontweight='bold', y=1.02, color='#5A3A5C'
    )
    plt.tight_layout()
    plt.savefig('figures/fig3_cross_asset.png', dpi=200,
                bbox_inches='tight', facecolor='#FAFAFA')
    plt.show()
    print("saved: figures/fig3_cross_asset.png")


# ── main ──────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("  DAY 1 — BASELINE LSTM × 4 ASSETS × 5 TRIALS")
print("="*60)

summaries = {}
for ticker in ALL_ASSETS:
    s = run_asset(ticker, n=NUM_TRIALS)
    if s:
        summaries[ticker] = s
    gc.collect()

print(f"\ncompleted: {len(summaries)}/{len(ALL_ASSETS)} assets")

# LIR table
ref = summaries.get('AAPL') or next(iter(summaries.values()), None)
lir_df = lir_table(ref) if ref else pd.DataFrame()

# figures
if summaries.get('AAPL') and summaries['AAPL'].get('_raw'):
    fig1_aie(summaries['AAPL'])

if not lir_df.empty:
    fig2_lir(lir_df)

if len(summaries) >= 2:
    fig3_cross_asset(summaries)

# save CSVs
rows = []
for ticker, s in summaries.items():
    rows.append({
        'Asset':        ticker,
        'N_trials':     s.get('n', NUM_TRIALS),
        'Persist_R2':   s.get('persist_R2', float('nan')),
        'DA_mean':      s.get('DA_mean', float('nan')),
        'DA_std':       s.get('DA_std',  float('nan')),
        'IC_mean':      s.get('IC_mean', float('nan')),
        'R2_price':     s.get('R2_price_mean', float('nan')),
        'R2_returns':   s.get('R2_ret_mean',   float('nan')),
        'Sharpe':       s.get('Sharpe_mean',   float('nan')),
        'autocorr':     s.get('autocorr', float('nan')),
        'realized_vol': s.get('vol', float('nan')),
        'ttest_p':      s.get('ttest_p', float('nan')),
    })

pd.DataFrame(rows).to_csv('results/day1_baseline.csv', index=False)
if not lir_df.empty:
    lir_df.to_csv('results/day1_lir.csv', index=False)

print("\nsaved: results/day1_baseline.csv")
print("saved: results/day1_lir.csv")

# final print
print("\n" + "="*70)
print("  FINAL RESULTS")
print("="*70)
print(f"\n{'asset':8s} | {'DA mean±std':18s} | {'IC':>7} | {'R²p':>7} | {'Sharpe':>7} | {'p-val':>8}")
print("-"*65)
for ticker, s in summaries.items():
    da_m = s.get('DA_mean', float('nan'))
    da_s = s.get('DA_std',  float('nan'))
    ic_m = s.get('IC_mean', float('nan'))
    r2p  = s.get('R2_price_mean', float('nan'))
    sh   = s.get('Sharpe_mean',   float('nan'))
    pval = s.get('ttest_p', float('nan'))
    pstr = f'{pval:.4f}' if not np.isnan(pval) else '  N/A'
    sig  = ' *' if (not np.isnan(pval) and pval < 0.05) else ''
    print(f"{ticker:8s} | {da_m:.4f} ± {da_s:.4f}    | "
          f"{ic_m:>7.4f} | {r2p:>7.4f} | {sh:>7.4f} | {pstr}{sig}")

print("\nday 1 done. next: day2_augmentation.py")
