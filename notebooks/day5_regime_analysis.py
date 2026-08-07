# day5_regime_analysis.py
#
# Part 2, Day 5 — regime-conditional analysis (RCAH)
# Vivian Chan | Glen A. Wilson High School | 2026
# Mentors: Mohammad Husain & Antoine Si | Cal Poly Pomona
#
# What this notebook does:
#   - Splits each asset's history into 3 regimes: COVID_Crash (Jan-Mar
#     2020), Recovery (Apr 2020 - Dec 2021), Rate_Hike (Jan 2022 - Dec
#     2023).
#   - For each (asset, regime): trains a baseline LSTM and a CycleGAN-
#     augmented LSTM on ONLY that regime's data, computes AugBenefit =
#     Hybrid DA - Real DA.
#   - Tests the Regime-Conditional Augmentation Hypothesis (RCAH):
#     Spearman correlation between realized volatility and AugBenefit,
#     across all 12 (asset, regime) rows.
#   - ρ > 0.5 and p < 0.05 -> RCAH supported. ρ < 0.3 -> not confirmed,
#     and that's still a publishable, honest result.
#
# Data reality check I'm being upfront about: COVID_Crash is only ~60
# calendar days. After the ~19-row feature warmup (SMA_20/BB/Vol_20 all
# need a 20-day rolling window), that leaves very few sequences to work
# with. Rather than pretend that's not a problem, this notebook uses a
# shorter sequence length and smaller CycleGAN window JUST for Day 5
# (documented below, not silently different), and skips any (asset,
# regime) pair that still doesn't have enough data after that -- with a
# printed reason, not a silently wrong number.
#
# This is supposedly my most original finding, so it needs to be an
# honest one even if that means some cells come back N/A.
#
# Run this on Kaggle with GPU T4. Expected time: ~2-3 hours for 4 assets
# x 3 regimes x 2 conditions x 3 trials (each regime is small, so this is
# lighter per-cell than Day 2-4, there's just more cells).


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
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

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

# LSTM model (same architecture as Day 1-4, hidden size unchanged)
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
EPOCHS      = 100
PATIENCE    = 15
BATCH_SIZE  = 32
LR          = 0.001
NUM_TRIALS  = 3   # fewer than Day 1-4's 5/10 -- regimes are small, 3 keeps this honest without ballooning runtime

# Day-5-specific: shorter sequence length and CycleGAN window so the
# smallest regime (COVID_Crash) has any usable sequences at all. This is
# a deliberate, documented adaptation -- NOT the same SEQ_LENGTH as
# Day 1-4, because those regimes have ~250-500 rows and COVID has ~40.
SEQ_LENGTH = 10

# data
ALL_ASSETS  = ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD']
TRAIN_START = '2020-01-01'
OOS_END     = '2023-12-31'

REGIMES = [
    ('COVID_Crash', '2020-01-01', '2020-03-31'),
    ('Recovery',     '2020-04-01', '2021-12-31'),
    ('Rate_Hike',    '2022-01-01', '2023-12-31'),
]

FEATURE_COLS = ['Close', 'RSI_14', 'SMA_5', 'SMA_10', 'SMA_20',
                'Vol_20', 'Mom_5', 'BB_pos']
N_FEATURES   = len(FEATURE_COLS)

# CycleGAN -- windows shrunk to fit inside a ~40-row regime; see note above.
GAN_WINDOW   = 10
GAN_HIDDEN   = 32
GAN_EPOCHS   = 40
GAN_BATCH    = 8
GAN_LR       = 2e-4
LAMBDA_CYCLE = 10.0
LAMBDA_ID    = 5.0
BLOCK_SIZE   = 3
GAN_SEED     = 42

AUG_RATIO = 1.0

# regime split + minimum-data guards
TRAIN_RATIO = 0.70   # more train, less val -- small regimes can't afford a big val slice
VAL_RATIO   = 0.10
MIN_FEATURE_ROWS = SEQ_LENGTH + 6   # need at least a handful of sequences past the one needed to train

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
print("config loaded")


# ── data ──────────────────────────────────────────────────────────────────────

def compute_rsi(prices, period=14):
    prices   = prices.squeeze()
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))


def build_features(df):
    """Same 8 features as Day 1-4. Keeps whatever extra columns (like
    Date) the caller already put on df -- only Close-derived columns are
    added, nothing is dropped except NaN warmup rows."""
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


def download_with_dates(ticker):
    """Same as Day 1-4's download(), except it keeps the Date column
    instead of dropping the index -- Day 5 needs real dates to slice
    into regimes. Feature columns are NOT computed here; that happens
    per-regime after slicing, matching how Day 1-4 always compute
    features fresh rather than reuse a shared frame.

    close.squeeze() handles the same yfinance MultiIndex quirk Day 1-4
    guard against (raw['Close'] can come back as a 1-column DataFrame
    instead of a Series). Building df from close.index/close.values
    after dropna() keeps dates correctly aligned to whichever rows
    survived -- slicing raw's index by position would misalign if any
    row other than a leading one got dropped."""
    raw = yf.download(ticker, start=TRAIN_START, end=OOS_END,
                      auto_adjust=True, progress=False)
    if raw.empty:
        print(f"  no data for {ticker}")
        return pd.DataFrame()
    close = raw['Close'].squeeze().dropna()
    df = pd.DataFrame({'Date': close.index, 'Close': close.values})
    df.reset_index(drop=True, inplace=True)
    print(f"  {ticker}: {len(df)} raw rows, {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


def temporal_split(df, train=TRAIN_RATIO, val=VAL_RATIO):
    n  = len(df)
    t1 = int(n * train)
    t2 = int(n * (train + val))
    return df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()


def fit_scaler(train_df):
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
    return torch.tensor(np.asarray(arr), dtype=torch.float32).to(device)


def make_windows(series_1d, window):
    if len(series_1d) < window:
        return np.empty((0, window), dtype=np.float32)
    n = len(series_1d) - window + 1
    return np.stack([series_1d[i:i + window] for i in range(n)]).astype(np.float32)


def regime_stats(regime_df):
    """Realized volatility (annualized) and lag-1 autocorrelation over
    the WHOLE regime window (not just its train split) -- these describe
    the regime itself, same as Day 1's autocorr_lag1/realized_vol."""
    close = regime_df['Close'].values.astype(float)
    if len(close) < 3:
        return float('nan'), float('nan')
    rets = np.diff(close) / close[:-1]
    vol = float(np.std(rets) * np.sqrt(252))
    ac  = float(pd.Series(close).autocorr(lag=1))
    return round(vol, 4), round(ac, 4)


# ── cyclegan (same design as Day 2, smaller window/batch for regime data) ────

def block_bootstrap_returns(returns, target_len, block_size, rng):
    n = len(returns)
    block_size = max(1, min(block_size, n))
    chunks = []
    total = 0
    while total < target_len:
        start = rng.integers(0, max(1, n - block_size + 1))
        chunk = returns[start:start + block_size]
        chunks.append(chunk)
        total += len(chunk)
    return np.concatenate(chunks)[:target_len]


class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm1d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator1D(nn.Module):
    def __init__(self, channels=1, hidden=GAN_HIDDEN, n_blocks=2):
        super().__init__()
        layers = [
            nn.Conv1d(channels, hidden, kernel_size=3, padding=1),
            nn.InstanceNorm1d(hidden),
            nn.ReLU(inplace=True),
        ]
        layers += [ResidualBlock1D(hidden) for _ in range(n_blocks)]
        layers += [nn.Conv1d(hidden, channels, kernel_size=3, padding=1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Discriminator1D(nn.Module):
    def __init__(self, channels=1, hidden=GAN_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def train_cyclegan_for_regime(label, train_df, verbose=False):
    """Same CycleGAN design as Day 2, trained on one regime's train
    slice. Returns (None, real_returns) if there isn't enough data --
    callers fall back to no augmentation for that regime rather than
    crash or fabricate a result."""
    real_returns = train_df['Close'].pct_change().dropna().values.astype(np.float32)
    real_windows = make_windows(real_returns, GAN_WINDOW)

    if len(real_windows) < GAN_BATCH:
        print(f"  WARNING: {label} has too few return windows for CycleGAN "
              f"({len(real_windows)} < {GAN_BATCH}); no augmentation for this regime.")
        return None, real_returns

    mu    = float(real_returns.mean())
    scale = float(3.0 * real_returns.std() + 1e-8)

    rng = np.random.default_rng(GAN_SEED)
    boot_returns = block_bootstrap_returns(real_returns, len(real_returns), BLOCK_SIZE, rng)
    boot_windows = make_windows(boot_returns, GAN_WINDOW)

    real_z = to_t(((real_windows - mu) / scale)[:, None, :])
    boot_z = to_t(((boot_windows - mu) / scale)[:, None, :])
    n = min(len(real_z), len(boot_z))

    torch.manual_seed(GAN_SEED)
    G_AB = Generator1D().to(device)
    G_BA = Generator1D().to(device)
    D_A  = Discriminator1D().to(device)
    D_B  = Discriminator1D().to(device)

    opt_G = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()),
                        lr=GAN_LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(list(D_A.parameters()) + list(D_B.parameters()),
                        lr=GAN_LR, betas=(0.5, 0.999))
    adv_loss = nn.MSELoss()
    cyc_loss = nn.L1Loss()

    n_batches = max(1, n // GAN_BATCH)
    for epoch in range(1, GAN_EPOCHS + 1):
        perm_a = torch.randperm(len(real_z))
        perm_b = torch.randperm(len(boot_z))

        for b in range(n_batches):
            idx_a = perm_a[b * GAN_BATCH:(b + 1) * GAN_BATCH]
            idx_b = perm_b[b * GAN_BATCH:(b + 1) * GAN_BATCH]
            if len(idx_a) == 0 or len(idx_b) == 0:
                continue
            real_a = real_z[idx_a]
            real_b = boot_z[idx_b]

            opt_G.zero_grad()
            fake_b = G_AB(real_a)
            fake_a = G_BA(real_b)
            rec_a  = G_BA(fake_b)
            rec_b  = G_AB(fake_a)
            idt_a  = G_BA(real_a)
            idt_b  = G_AB(real_b)

            d_fake_a = D_A(fake_a)
            d_fake_b = D_B(fake_b)
            loss_g = (
                adv_loss(d_fake_a, torch.ones_like(d_fake_a))
                + adv_loss(d_fake_b, torch.ones_like(d_fake_b))
                + LAMBDA_CYCLE * (cyc_loss(rec_a, real_a) + cyc_loss(rec_b, real_b))
                + LAMBDA_ID * (cyc_loss(idt_a, real_a) + cyc_loss(idt_b, real_b))
            )
            loss_g.backward()
            opt_G.step()

            opt_D.zero_grad()
            d_a_real = D_A(real_a)
            d_a_fake = D_A(fake_a.detach())
            d_b_real = D_B(real_b)
            d_b_fake = D_B(fake_b.detach())
            loss_d = 0.5 * (
                adv_loss(d_a_real, torch.ones_like(d_a_real))
                + adv_loss(d_a_fake, torch.zeros_like(d_a_fake))
                + adv_loss(d_b_real, torch.ones_like(d_b_real))
                + adv_loss(d_b_fake, torch.zeros_like(d_b_fake))
            )
            loss_d.backward()
            opt_D.step()

    return G_BA, real_returns


def generate_cyclegan_dataset(G_BA, real_returns, anchor_price, scaler, target_len, seed,
                              window=GAN_WINDOW, block_size=BLOCK_SIZE):
    rng = np.random.default_rng(seed)
    raw_boot = block_bootstrap_returns(real_returns, target_len, block_size, rng)

    if G_BA is not None and len(real_returns) >= window:
        mu    = float(real_returns.mean())
        scale = float(3.0 * real_returns.std() + 1e-8)
        pad = (-len(raw_boot)) % window
        boot_for_gan = raw_boot if pad == 0 else np.concatenate([raw_boot, raw_boot[:pad]])
        boot_z = (boot_for_gan - mu) / scale
        n_windows = len(boot_z) // window
        boot_z = boot_z[:n_windows * window].reshape(n_windows, window)

        G_BA.eval()
        with torch.no_grad():
            inp = to_t(boot_z[:, None, :])
            refined = G_BA(inp).cpu().numpy()[:, 0, :]
        synthetic_returns = (refined * scale + mu).reshape(-1)[:target_len]
    else:
        synthetic_returns = raw_boot[:target_len]

    prices = [anchor_price]
    for r in synthetic_returns:
        prices.append(prices[-1] * (1.0 + float(r)))
    synth_df = build_features(pd.DataFrame({'Close': np.array(prices)}))

    if len(synth_df) < SEQ_LENGTH + 1:
        return np.empty((0, SEQ_LENGTH, N_FEATURES)), np.empty((0,))

    synth_scaled = scaler.transform(synth_df[FEATURE_COLS].values)
    return make_sequences(synth_scaled)


# ── model + training (same architecture as Day 1-4) ───────────────────────────

class LSTM(nn.Module):
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


def train(model, X_tr, y_tr, X_vl, y_vl, verbose=False):
    opt  = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sch  = optim.lr_scheduler.ReduceLROnPlateau(
               opt, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    crit = nn.MSELoss()

    best_val   = float('inf')
    no_imp     = 0
    best_state = None

    batch_size = min(BATCH_SIZE, max(1, len(X_tr)))
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    for ep in range(1, EPOCHS + 1):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by.unsqueeze(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_vl), y_vl.unsqueeze(-1)).item()
        sch.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            no_imp     = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def da(yt, yp):
    yt = yt.flatten(); yp = yp.flatten()
    return round(float(np.mean(np.sign(np.diff(yt)) == np.sign(np.diff(yp)))), 4)


# ── per-regime pipeline ───────────────────────────────────────────────────────

def run_regime(ticker, regime_name, regime_df):
    label = f"{ticker}/{regime_name}"
    vol, autocorr = regime_stats(regime_df)
    row = {'Asset': ticker, 'Regime': regime_name, 'Vol': vol, 'AutoCorr': autocorr,
           'Real_DA': float('nan'), 'Hybrid_DA': float('nan'), 'AugBenefit': float('nan'),
           'N_train': 0, 'N_test': 0}

    feat_df = build_features(regime_df[['Close']].copy())
    if len(feat_df) < MIN_FEATURE_ROWS:
        print(f"  SKIP {label}: only {len(feat_df)} feature rows after warmup "
              f"(need >= {MIN_FEATURE_ROWS})")
        return row

    train_df, val_df, test_df = temporal_split(feat_df)
    if len(train_df) < SEQ_LENGTH + 2 or len(val_df) < SEQ_LENGTH + 1 or len(test_df) < SEQ_LENGTH + 1:
        print(f"  SKIP {label}: train/val/test too small after split "
              f"({len(train_df)}/{len(val_df)}/{len(test_df)} rows)")
        return row

    scaler = fit_scaler(train_df)
    X_tr, y_tr = make_sequences(scaler.transform(train_df[FEATURE_COLS].values))
    X_vl, y_vl = make_sequences(scaler.transform(val_df[FEATURE_COLS].values))
    X_te, y_te = make_sequences(scaler.transform(test_df[FEATURE_COLS].values))
    if len(X_tr) < 5 or len(X_vl) < 1 or len(X_te) < 2:
        print(f"  SKIP {label}: too few sequences ({len(X_tr)} train / "
              f"{len(X_vl)} val / {len(X_te)} test)")
        return row

    cs = MinMaxScaler(feature_range=(-1, 1))
    cs.fit(train_df[['Close']].values)
    y_true_prices = cs.inverse_transform(y_te.reshape(-1, 1))
    anchor_price = float(train_df['Close'].iloc[0])
    target_len = int(round(AUG_RATIO * len(train_df)))

    X_vl_t, y_vl_t = to_t(X_vl), to_t(y_vl)
    X_te_t = to_t(X_te)

    print(f"  {label}: vol={vol}, autocorr={autocorr}, "
          f"train/val/test={len(X_tr)}/{len(X_vl)}/{len(X_te)} sequences")

    # -- real (baseline) trials --
    real_da_vals = []
    for i in range(NUM_TRIALS):
        seed = SEED + i * 13
        torch.manual_seed(seed); np.random.seed(seed)
        model = LSTM().to(device)
        model = train(model, to_t(X_tr), to_t(y_tr), X_vl_t, y_vl_t)
        model.eval()
        with torch.no_grad():
            pred_s = model(X_te_t).cpu().numpy()
        pred_prices = cs.inverse_transform(pred_s[:, :1])
        real_da_vals.append(da(y_true_prices, pred_prices))

    # -- hybrid (CycleGAN-augmented) trials --
    G_BA, real_returns = train_cyclegan_for_regime(label, train_df)
    hybrid_da_vals = []
    for i in range(NUM_TRIALS):
        seed = SEED + i * 13
        torch.manual_seed(seed); np.random.seed(seed)
        X_synth, y_synth = generate_cyclegan_dataset(
            G_BA, real_returns, anchor_price, scaler, target_len, seed)
        if len(X_synth) > 0:
            X_tr_aug = np.concatenate([X_tr, X_synth])
            y_tr_aug = np.concatenate([y_tr, y_synth])
        else:
            X_tr_aug, y_tr_aug = X_tr, y_tr
        model = LSTM().to(device)
        model = train(model, to_t(X_tr_aug), to_t(y_tr_aug), X_vl_t, y_vl_t)
        model.eval()
        with torch.no_grad():
            pred_s = model(X_te_t).cpu().numpy()
        pred_prices = cs.inverse_transform(pred_s[:, :1])
        hybrid_da_vals.append(da(y_true_prices, pred_prices))

    real_mean   = round(float(np.mean(real_da_vals)), 4)
    hybrid_mean = round(float(np.mean(hybrid_da_vals)), 4)
    aug_benefit = round(hybrid_mean - real_mean, 4)
    print(f"  {label}: Real_DA={real_mean} | Hybrid_DA={hybrid_mean} | "
          f"AugBenefit={aug_benefit:+.4f}")

    row.update({
        'Real_DA': real_mean, 'Hybrid_DA': hybrid_mean, 'AugBenefit': aug_benefit,
        'N_train': len(X_tr), 'N_test': len(X_te),
    })
    return row


# ── rcah test + figure ─────────────────────────────────────────────────────────

def rcah_test(rows_df):
    """Spearman correlation between realized volatility and AugBenefit,
    across every (asset, regime) row that wasn't skipped."""
    valid = rows_df.dropna(subset=['Vol', 'AugBenefit'])
    if len(valid) < 3:
        print(f"\n  RCAH test: only {len(valid)} valid rows -- not enough to test.")
        return float('nan'), float('nan')
    rho, pval = stats.spearmanr(valid['Vol'], valid['AugBenefit'])
    print(f"\n  RCAH test (n={len(valid)}): Spearman rho={rho:.4f}, p={pval:.4f}")
    if rho > 0.5 and pval < 0.05:
        print("  -> RCAH supported: augmentation benefit rises with realized volatility.")
    elif rho < 0.3:
        print("  -> RCAH not confirmed. Still a valid, honest result to report.")
    else:
        print("  -> RCAH inconclusive at this sample size.")
    return round(float(rho), 4), round(float(pval), 4)


def fig6_rcah_scatter(rows_df, rho, pval):
    valid = rows_df.dropna(subset=['Vol', 'AugBenefit'])
    if valid.empty:
        print("  skipping fig6 -- no valid rows to plot")
        return

    colors = {'AAPL': '#8B5A8D', 'MSFT': '#E89AC7', 'GOOGL': '#D4A5C7', 'BTC-USD': '#E74C3C'}
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#FAFAFA')

    for _, r in valid.iterrows():
        ax.scatter(r['Vol'], r['AugBenefit'], s=180, zorder=5,
                  color=colors.get(r['Asset'], '#8B5A8D'), edgecolors='white', linewidth=1.5)
        ax.annotate(f"{r['Asset']}\n{r['Regime']}", (r['Vol'], r['AugBenefit']),
                   textcoords='offset points', xytext=(8, 4), fontsize=8)

    if len(valid) >= 3:
        z = np.polyfit(valid['Vol'], valid['AugBenefit'], 1)
        xline = np.linspace(valid['Vol'].min(), valid['Vol'].max(), 100)
        ax.plot(xline, np.poly1d(z)(xline), 'k--', lw=1.5, alpha=0.5, label='trend')

    ax.axhline(0, color='black', lw=1)
    pstr = f'{pval:.3f}' if not np.isnan(pval) else 'N/A'
    ax.text(0.05, 0.95, f'Spearman rho={rho:.3f}\np={pstr}',
           transform=ax.transAxes, fontsize=10, va='top',
           bbox=dict(boxstyle='round', facecolor='#E8F8E8', alpha=0.8))
    ax.set_xlabel('realized volatility (annualized)', fontsize=11)
    ax.set_ylabel('augmentation benefit (Hybrid DA - Real DA)', fontsize=11)
    ax.set_title('Figure 6 — RCAH: volatility vs augmentation benefit\n'
                'each point is one (asset, regime) pair', fontsize=12,
                fontweight='bold', color='#5A3A5C')
    ax.set_facecolor('#FFFAFD'); ax.grid(alpha=0.3, linestyle='--')
    if len(valid) >= 3:
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('figures/fig6_rcah_scatter.png', dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
    plt.show()
    print("saved: figures/fig6_rcah_scatter.png")


# ── main ──────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  DAY 5 — REGIME-CONDITIONAL ANALYSIS (RCAH)")
print("="*70)

rows = []
for ticker in ALL_ASSETS:
    print(f"\n{'='*65}\n  {ticker}\n{'='*65}")
    full_df = download_with_dates(ticker)
    if full_df.empty:
        continue

    for regime_name, start, end in REGIMES:
        mask = (full_df['Date'] >= start) & (full_df['Date'] <= end)
        regime_df = full_df.loc[mask].reset_index(drop=True)
        if regime_df.empty:
            print(f"  SKIP {ticker}/{regime_name}: no rows in this date range")
            rows.append({'Asset': ticker, 'Regime': regime_name, 'Vol': float('nan'),
                        'AutoCorr': float('nan'), 'Real_DA': float('nan'),
                        'Hybrid_DA': float('nan'), 'AugBenefit': float('nan'),
                        'N_train': 0, 'N_test': 0})
            continue
        rows.append(run_regime(ticker, regime_name, regime_df))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

rows_df = pd.DataFrame(rows)
rows_df.to_csv('results/day5_regime_analysis.csv', index=False)
print(f"\nsaved: results/day5_regime_analysis.csv ({len(rows_df)} rows)")

rho, pval = rcah_test(rows_df)
fig6_rcah_scatter(rows_df, rho, pval)

print("\n" + "="*90)
print("  DAY 5 FINAL RESULTS")
print("="*90)
print(rows_df.to_string(index=False))
print(f"\nRCAH: Spearman rho={rho}, p={pval}")
print("\nday 5 done. next: day6 quantum ablation study")
