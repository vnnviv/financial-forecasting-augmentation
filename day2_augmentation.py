
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

# LSTM model / training (identical to Day 1 -- same architecture, same
# hyperparameters, so any DA change is attributable to the training data,
# not a model change)
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
SEQ_LENGTH  = 20
EPOCHS      = 100
PATIENCE    = 15
BATCH_SIZE  = 32
LR          = 0.001
NUM_TRIALS  = 5     # same trial count as Day 1 -- paired comparison

# data
ALL_ASSETS  = ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD']
TRAIN_START = '2020-01-01'
OOS_END     = '2023-12-31'

FEATURE_COLS = ['Close', 'RSI_14', 'SMA_5', 'SMA_10', 'SMA_20',
                'Vol_20', 'Mom_5', 'BB_pos']
N_FEATURES   = len(FEATURE_COLS)

# CycleGAN
GAN_WINDOW   = 32     # length of one return window fed to the GAN
GAN_HIDDEN   = 32     # conv channel width
GAN_EPOCHS   = 60
GAN_BATCH    = 16
GAN_LR       = 2e-4
LAMBDA_CYCLE = 10.0
LAMBDA_ID    = 5.0
BLOCK_SIZE   = 5      # bootstrap block length, in days
GAN_SEED     = 42     # one CycleGAN trained per asset, not per trial

# augmentation
AUG_RATIO = 1.0   # synthetic days added to train = AUG_RATIO * len(train_df)

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
print("config loaded")


# ── data (identical to Day 1) ─────────────────────────────────────────────────

def compute_rsi(prices, period=14):
    prices   = prices.squeeze()
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))


def build_features(df):
    """
    8 features from Close price. Used for BOTH real and synthetic price
    series -- always called fresh, never fed pre-computed real features.
    That's what makes the synthetic augmentation leakage-safe.
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


# ── cyclegan -- synthetic return generation ───────────────────────────────────
#
# domain A = windows of real daily returns.
# domain B = windows of block-bootstrapped returns -- a naive synthetic
#            baseline built by resampling contiguous blocks of the real
#            return series with replacement. preserves local structure
#            within a block, destroys everything beyond it.
#
# G_AB / G_BA learn to translate between the two domains with adversarial
# + cycle-consistency + identity losses (standard CycleGAN). after
# training, G_BA turns fresh bootstrap draws into windows that look more
# like real return dynamics than the naive bootstrap alone.

def block_bootstrap_returns(returns, target_len, block_size, rng):
    """Resample contiguous blocks of `returns` with replacement until
    reaching target_len. This is domain B -- the raw material CycleGAN
    learns to refine into something closer to domain A."""
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


def make_windows(series_1d, window):
    """Overlapping windows of length `window` -- training data for the GAN."""
    if len(series_1d) < window:
        return np.empty((0, window), dtype=np.float32)
    n = len(series_1d) - window + 1
    return np.stack([series_1d[i:i + window] for i in range(n)]).astype(np.float32)


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
    """Fully-convolutional generator -- output length always equals input
    length (stride-1 convs only), so window length never has to be
    tracked through the network."""
    def __init__(self, channels=1, hidden=GAN_HIDDEN, n_blocks=3):
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
    """PatchGAN-style 1D discriminator -- scores overlapping patches of the
    window rather than the whole window at once (standard CycleGAN design)."""
    def __init__(self, channels=1, hidden=GAN_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden, hidden * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden * 2, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def train_cyclegan_for_asset(ticker, train_df, verbose=True):
    """One CycleGAN per asset (not per trial) -- training it is the
    expensive part (~60 min on a T4), and the downstream LSTM trials only
    need different *synthetic draws* and *init seeds*, not a re-trained
    generator each time. Returns (G_BA, real_returns) -- real_returns is
    reused later as the bootstrap source and for the z-scale stats.
    Returns (None, real_returns) if there's too little data to train on,
    in which case callers fall back to raw block-bootstrap (no refinement).
    """
    # .squeeze() guards against yfinance sometimes handing back 'Close' as a
    # 1-column DataFrame instead of a Series (same issue build_features()
    # guards against) -- without it pct_change()/.values comes back 2D and
    # every downstream window/tensor picks up a stray trailing axis.
    real_returns = train_df['Close'].squeeze().pct_change().dropna().values.astype(np.float32)
    real_windows = make_windows(real_returns, GAN_WINDOW)

    if len(real_windows) < GAN_BATCH:
        print(f"  WARNING: {ticker} has too few return windows to train a "
              f"CycleGAN ({len(real_windows)} < {GAN_BATCH}); falling back "
              f"to raw block-bootstrap augmentation (no GAN refinement).")
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

    if verbose:
        print(f"  training CycleGAN on {ticker}: {n} windows, {GAN_EPOCHS} epochs")

    n_batches = max(1, n // GAN_BATCH)
    for epoch in range(1, GAN_EPOCHS + 1):
        perm_a = torch.randperm(len(real_z))
        perm_b = torch.randperm(len(boot_z))
        g_total, d_total = 0.0, 0.0

        for b in range(n_batches):
            idx_a = perm_a[b * GAN_BATCH:(b + 1) * GAN_BATCH]
            idx_b = perm_b[b * GAN_BATCH:(b + 1) * GAN_BATCH]
            if len(idx_a) == 0 or len(idx_b) == 0:
                continue
            real_a = real_z[idx_a]
            real_b = boot_z[idx_b]

            # ── generators (translate + cycle back + identity) ──
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

            # ── discriminators ──
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

            g_total += loss_g.item()
            d_total += loss_d.item()

        if verbose and epoch % 15 == 0:
            print(f"    gan epoch {epoch:3d}/{GAN_EPOCHS} | "
                  f"G={g_total / n_batches:.4f} | D={d_total / n_batches:.4f}")

    return G_BA, real_returns


def generate_synthetic_dataset(G_BA, real_returns, anchor_price, scaler,
                               target_len, seed, window=GAN_WINDOW,
                               block_size=BLOCK_SIZE):
    """
    Builds one synthetic (X, y) dataset the SAME way real data is built:
    bootstrap/GAN-refine returns -> compound into a price path -> run
    build_features() (the exact function used on real prices) -> scale
    with the REAL train scaler -> make_sequences(). No real feature value
    is ever copied onto a synthetic row.
    """
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
        # no trained generator available -- fall back to raw bootstrap
        synthetic_returns = raw_boot[:target_len]

    prices = [anchor_price]
    for r in synthetic_returns:
        prices.append(prices[-1] * (1.0 + float(r)))
    synth_df = build_features(pd.DataFrame({'Close': np.array(prices)}))

    if len(synth_df) < SEQ_LENGTH + 1:
        return np.empty((0, SEQ_LENGTH, N_FEATURES)), np.empty((0,))

    synth_scaled = scaler.transform(synth_df[FEATURE_COLS].values)
    return make_sequences(synth_scaled)


# ── model (identical to Day 1) ────────────────────────────────────────────────

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


# ── training (identical to Day 1) ─────────────────────────────────────────────

def train(model, X_tr, y_tr, X_vl, y_vl, verbose=True):
    opt  = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sch  = optim.lr_scheduler.ReduceLROnPlateau(
               opt, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    crit = nn.MSELoss()

    best_val   = float('inf')
    no_imp     = 0
    best_state = None

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

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
            print(f"    ep {ep:3d} | train={ep_loss / len(loader):.5f} "
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


# ── metrics (identical to Day 1) ──────────────────────────────────────────────

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
        'MSE':      round(float(mean_squared_error(yt, yp)), 4),
        'RMSE':     round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        'R2_price': round(float(r2_score(yt, yp)), 4),
        'R2_ret':   return_r2(y_true, y_pred),
        'IC':       ic(y_true, y_pred),
        'DA':       da(y_true, y_pred),
        'Sharpe':   sharpe(y_true, y_pred),
    }
    if label:
        print(f"\n  [{label}]")
        for k, v in m.items():
            print(f"    {k:12s}: {v}")
    return m


# ── per-asset pipeline ─────────────────────────────────────────────────────────

def load_day1_baseline():
    """Accepts either the repo's canonical results/day1_baseline.csv
    (DA_mean column) or the day1_cross_asset_results.csv some earlier
    Day 1 script versions save at the notebook root (LSTM_DirAcc column)
    -- normalizes either into the same DA_mean-indexed-by-Asset schema."""
    candidates = [
        ('results/day1_baseline.csv', {}),
        ('day1_cross_asset_results.csv', {'LSTM_DirAcc': 'DA_mean'}),
    ]
    for path, rename in candidates:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).rename(columns=rename)
        if 'Asset' in df.columns and 'DA_mean' in df.columns:
            print(f"  loaded Day 1 baseline from {path}")
            return df.set_index('Asset')

    print("  NOTE: no Day 1 baseline CSV found (checked results/day1_baseline.csv "
          "and day1_cross_asset_results.csv in the working directory) -- run "
          "Day 1 first for the baseline comparison columns. Continuing with "
          "augmentation-only results for now.")
    return None


def run_asset_augmented(ticker, baseline_row=None, n=NUM_TRIALS):
    print(f"\n{'='*65}")
    print(f"  {ticker} — CycleGAN augmentation ({n} trials)")
    print(f"{'='*65}")

    df = download(ticker)
    if df.empty or len(df) < SEQ_LENGTH * 4:
        print(f"  SKIP {ticker}: insufficient data")
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
        print(f"  SKIP {ticker}: too few sequences")
        return None

    cs = MinMaxScaler(feature_range=(-1, 1))
    cs.fit(train_df[['Close']].values)
    y_true_prices = cs.inverse_transform(y_te.reshape(-1, 1))

    G_BA, real_returns = train_cyclegan_for_asset(ticker, train_df)
    anchor_price = float(train_df['Close'].squeeze().iloc[0])
    target_len   = int(round(AUG_RATIO * len(train_df)))

    X_vl_t, y_vl_t = to_t(X_vl), to_t(y_vl)
    X_te_t         = to_t(X_te)

    trials = []
    for i in range(n):
        seed = SEED + i * 13
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_synth, y_synth = generate_synthetic_dataset(
            G_BA, real_returns, anchor_price, scaler, target_len, seed=seed)

        if len(X_synth) > 0:
            X_tr_aug = np.concatenate([X_tr, X_synth])
            y_tr_aug = np.concatenate([y_tr, y_synth])
        else:
            X_tr_aug, y_tr_aug = X_tr, y_tr

        model = LSTM(input_size=N_FEATURES).to(device)
        model = train(model, to_t(X_tr_aug), to_t(y_tr_aug), X_vl_t, y_vl_t,
                      verbose=(i == 0))

        model.eval()
        with torch.no_grad():
            pred_s = model(X_te_t).cpu().numpy()
        pred_prices = cs.inverse_transform(pred_s[:, :1])

        metrics = evaluate(y_true_prices, pred_prices,
                           label=f'{ticker} augmented OOS' if i == 0 else '')
        print(f"  trial {i+1}: DA={metrics['DA']:.4f} | IC={metrics['IC']:.4f} "
              f"| n_real={len(X_tr)} | n_synth={len(X_synth)}")
        trials.append({'metrics': metrics, 'n_synth': len(X_synth)})

    if not trials:
        return None

    print(f"  integrity check: {trials[0]['n_synth']} synthetic examples "
          f"added to TRAIN only; validation and test are 100% real.")

    keys = ['MSE', 'RMSE', 'R2_price', 'R2_ret', 'IC', 'DA', 'Sharpe']
    out  = {'ticker': ticker, 'n': len(trials)}
    for k in keys:
        vals = [float(t['metrics'][k]) for t in trials
                if t['metrics'].get(k) is not None
                and not np.isnan(float(t['metrics'].get(k, float('nan'))))]
        if vals:
            out[f'{k}_mean'] = round(np.mean(vals), 4)
            out[f'{k}_std']  = round(np.std(vals), 4)
    out['n_synth_mean'] = round(np.mean([t['n_synth'] for t in trials]), 1)

    if baseline_row is not None:
        baseline_da = float(baseline_row['DA_mean'])
        out['baseline_DA'] = round(baseline_da, 4)
        out['delta_pp'] = round((out.get('DA_mean', float('nan')) - baseline_da) * 100, 2)

        da_vals = [t['metrics']['DA'] for t in trials]
        if len(da_vals) > 1:
            tstat, pval = stats.ttest_1samp(da_vals, popmean=baseline_da)
            out['ttest_t'] = round(float(tstat), 4)
            out['ttest_p'] = round(float(pval), 4)
            sig = '*** significant' if pval < 0.05 else '(not sig)'
            print(f"\n  vs Day 1 baseline (DA={baseline_da:.4f}): "
                  f"t={tstat:.3f} p={pval:.4f} {sig} | "
                  f"delta={out['delta_pp']:+.2f}pp")

    print(f"\n  {'metric':12s} | {'mean±std':20s}")
    print(f"  {'-'*40}")
    for k in ['R2_price', 'R2_ret', 'IC', 'DA', 'Sharpe']:
        m = out.get(f'{k}_mean', float('nan'))
        s = out.get(f'{k}_std', float('nan'))
        print(f"  {k:12s} | {m:>7.4f} ± {s:<10.4f}")

    return out


# ── figures ────────────────────────────────────────────────────────────────────

def fig4_comparison(summaries):
    tickers  = list(summaries.keys())
    baseline = [summaries[t].get('baseline_DA', float('nan')) for t in tickers]
    aug_mean = [summaries[t].get('DA_mean', float('nan')) for t in tickers]
    aug_std  = [summaries[t].get('DA_std', float('nan')) for t in tickers]
    delta    = [summaries[t].get('delta_pp', float('nan')) for t in tickers]

    x = np.arange(len(tickers)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#FAFAFA')

    ax.bar(x - w/2, baseline, w, color='#95A5A6', alpha=0.85,
           label='Day 1 baseline (real only)', edgecolor='white')
    ax.bar(x + w/2, aug_mean, w, color='#8B5A8D', alpha=0.9,
           label='CycleGAN-augmented', edgecolor='white')
    ax.errorbar(x + w/2, aug_mean, yerr=aug_std, fmt='none',
                color='black', capsize=5, lw=2)
    ax.axhline(0.50, color='red', ls='--', lw=1.5, label='random (50%)', alpha=0.7)

    for xi, b, d in zip(x, aug_mean, delta):
        if not np.isnan(b) and not np.isnan(d):
            ax.text(xi + w/2, b + 0.01, f'{d:+.1f}pp', ha='center',
                    fontsize=9, fontweight='bold', color='#5A3A5C')

    ax.set_xticks(x); ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel('directional accuracy', fontsize=11)
    ax.set_title('Figure 4 — CycleGAN augmentation vs Day 1 baseline\n'
                 'directional accuracy, real held-out test set only',
                 fontsize=12, fontweight='bold', color='#5A3A5C')
    ax.legend(fontsize=9); ax.set_facecolor('#FFFAFD')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('figures/fig4_cyclegan_comparison.png', dpi=200,
                bbox_inches='tight', facecolor='#FAFAFA')
    plt.show()
    print("saved: figures/fig4_cyclegan_comparison.png")


# ── main ──────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  DAY 2 — CYCLEGAN AUGMENTATION")
print("="*70)

baseline_df = load_day1_baseline()

summaries = {}
for ticker in ALL_ASSETS:
    baseline_row = None
    if baseline_df is not None and ticker in baseline_df.index:
        baseline_row = baseline_df.loc[ticker]
    s = run_asset_augmented(ticker, baseline_row=baseline_row, n=NUM_TRIALS)
    if s:
        summaries[ticker] = s
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\ncompleted: {len(summaries)}/{len(ALL_ASSETS)} assets")

if summaries and any('baseline_DA' in s for s in summaries.values()):
    fig4_comparison(summaries)
elif summaries:
    print("skipping fig4 (no Day 1 baseline loaded -- run day1_baseline.py first)")

rows = []
for ticker, s in summaries.items():
    rows.append({
        'Asset':               ticker,
        'N_trials':            s.get('n', NUM_TRIALS),
        'Baseline_DA':         s.get('baseline_DA', float('nan')),
        'CycleGAN_DA_mean':    s.get('DA_mean', float('nan')),
        'CycleGAN_DA_std':     s.get('DA_std', float('nan')),
        'Delta_pp':            s.get('delta_pp', float('nan')),
        'CycleGAN_IC_mean':    s.get('IC_mean', float('nan')),
        'CycleGAN_R2_price':   s.get('R2_price_mean', float('nan')),
        'CycleGAN_R2_returns': s.get('R2_ret_mean', float('nan')),
        'CycleGAN_Sharpe':     s.get('Sharpe_mean', float('nan')),
        'n_synth_examples':    s.get('n_synth_mean', float('nan')),
        'ttest_p':             s.get('ttest_p', float('nan')),
    })

if rows:
    pd.DataFrame(rows).to_csv('results/day2_cyclegan_results.csv', index=False)
    print("\nsaved: results/day2_cyclegan_results.csv")

print("\n" + "="*75)
print("  DAY 2 FINAL RESULTS")
print("="*75)
print(f"\n{'asset':8s} | {'baseline DA':>11} | {'aug DA (mean±std)':20s} | "
      f"{'delta':>8} | {'p-val':>8}")
print("-"*70)
for ticker, s in summaries.items():
    bda  = s.get('baseline_DA', float('nan'))
    m    = s.get('DA_mean', float('nan'))
    sd   = s.get('DA_std', float('nan'))
    dpp  = s.get('delta_pp', float('nan'))
    pval = s.get('ttest_p', float('nan'))
    bstr = f'{bda:.4f}' if not np.isnan(bda) else '  N/A'
    dstr = f'{dpp:+.2f}pp' if not np.isnan(dpp) else '   N/A'
    pstr = f'{pval:.4f}' if not np.isnan(pval) else '  N/A'
    print(f"{ticker:8s} | {bstr:>11} | {m:.4f} ± {sd:.4f}       | "
          f"{dstr:>8} | {pstr:>8}")

print("\nday 2 done. next: day3_wgan_smote.py")
