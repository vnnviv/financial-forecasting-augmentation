# day4_statistical_validation.py
#
# What this notebook does:
#   - Upgrades every condition from 5 trials to 10 -- baseline (real only),
#     CycleGAN, WGAN-GP, and SMOTE-TS, all four re-run here with matching
#     seeds so every method is properly paired against baseline trial-for-
#     trial, not just compared against a saved mean.
#   - For each asset, each generator (CycleGAN, WGAN-GP) is still trained
#     once, same as Day 2/Day 3 -- only the LSTM trials scale to 10.
#   - Three tests per (asset, method) pair, all using the same 10 paired
#     trials:
#       1. paired t-test on the 10 DA values (method vs baseline)
#       2. Wilcoxon signed-rank test -- non-parametric, doesn't assume DA
#          is normally distributed across trials
#       3. McNemar's test on direction-correct/incorrect agreement,
#          pooled across all 10 trials' test-set predictions (valid
#          pairing because baseline and method share the same seed and
#          therefore the same test set every trial)
#   - Flags anything with p < 0.05 in the final printout.
#
# Any result without a p-value here is anecdotal -- that's the whole
# point of today.
#
# Run this on Kaggle with GPU T4. Expected time: ~2-3 hours for 4 assets
# x 4 conditions x 10 trials (the generators are still trained once per
# asset each, so this scales in LSTM trials, not GAN training).


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

# LSTM model / training (identical to Day 1-3)
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
SEQ_LENGTH  = 20
EPOCHS      = 100
PATIENCE    = 15
BATCH_SIZE  = 32
LR          = 0.001
NUM_TRIALS  = 10    # upgraded from 5 -- stronger statistical claims

# data
ALL_ASSETS  = ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD']
TRAIN_START = '2020-01-01'
OOS_END     = '2023-12-31'

FEATURE_COLS = ['Close', 'RSI_14', 'SMA_5', 'SMA_10', 'SMA_20',
                'Vol_20', 'Mom_5', 'BB_pos']
N_FEATURES   = len(FEATURE_COLS)

# CycleGAN (Day 2)
GAN_WINDOW   = 32
GAN_HIDDEN   = 32
GAN_EPOCHS   = 60
GAN_BATCH    = 16
GAN_LR       = 2e-4
LAMBDA_CYCLE = 10.0
LAMBDA_ID    = 5.0
BLOCK_SIZE   = 5
GAN_SEED     = 42

# WGAN-GP (Day 3)
WGAN_HIDDEN = 64
WGAN_EPOCHS = 60
WGAN_BATCH  = 16
WGAN_LR     = 1e-4
N_CRITIC    = 5
LAMBDA_GP   = 10.0

# SMOTE-TS (Day 3)
SMOTE_K = 5

# augmentation
AUG_RATIO = 1.0

os.makedirs('results', exist_ok=True)
print("config loaded")


# ── data (identical to Day 1-3) ───────────────────────────────────────────────

def compute_rsi(prices, period=14):
    prices   = prices.squeeze()
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))


def build_features(df):
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


def make_windows(series_1d, window):
    if len(series_1d) < window:
        return np.empty((0, window), dtype=np.float32)
    n = len(series_1d) - window + 1
    return np.stack([series_1d[i:i + window] for i in range(n)]).astype(np.float32)


# ── cyclegan (identical to Day 2) ─────────────────────────────────────────────

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
    real_returns = train_df['Close'].squeeze().pct_change().dropna().values.astype(np.float32)
    real_windows = make_windows(real_returns, GAN_WINDOW)

    if len(real_windows) < GAN_BATCH:
        print(f"  WARNING: {ticker} has too few return windows for CycleGAN; skipping.")
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

        if verbose and epoch % 20 == 0:
            print(f"    cyclegan epoch {epoch:3d}/{GAN_EPOCHS}")

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


# ── wgan-gp (identical to Day 3) ──────────────────────────────────────────────

class WGANGenerator(nn.Module):
    def __init__(self, latent_dim=GAN_WINDOW, hidden=WGAN_HIDDEN, window=GAN_WINDOW):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, window), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class WGANCritic(nn.Module):
    def __init__(self, window=GAN_WINDOW, hidden=WGAN_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def gradient_penalty(critic, real, fake):
    batch = real.size(0)
    eps = torch.rand(batch, 1, device=real.device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(interp)
    grads = torch.autograd.grad(
        outputs=scores, inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grads = grads.view(batch, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp_for_asset(ticker, train_df, verbose=True):
    real_returns = train_df['Close'].squeeze().pct_change().dropna().values.astype(np.float32)
    real_windows = make_windows(real_returns, GAN_WINDOW)

    if len(real_windows) < WGAN_BATCH:
        print(f"  WARNING: {ticker} has too few return windows for WGAN-GP; skipping.")
        return None, (0.0, 1.0)

    mu    = float(real_returns.mean())
    scale = float(3.0 * real_returns.std() + 1e-8)
    real_z = to_t((real_windows - mu) / scale)
    n = len(real_z)

    torch.manual_seed(GAN_SEED)
    G = WGANGenerator().to(device)
    C = WGANCritic().to(device)
    opt_G = optim.Adam(G.parameters(), lr=WGAN_LR, betas=(0.5, 0.9))
    opt_C = optim.Adam(C.parameters(), lr=WGAN_LR, betas=(0.5, 0.9))

    if verbose:
        print(f"  training WGAN-GP on {ticker}: {n} windows, {WGAN_EPOCHS} epochs")

    n_batches = max(1, n // WGAN_BATCH)
    for epoch in range(1, WGAN_EPOCHS + 1):
        perm = torch.randperm(n)

        for b in range(n_batches):
            idx = perm[b * WGAN_BATCH:(b + 1) * WGAN_BATCH]
            if len(idx) == 0:
                continue
            real_batch = real_z[idx]
            bs = real_batch.size(0)

            for _ in range(N_CRITIC):
                z = torch.randn(bs, GAN_WINDOW, device=device)
                fake_batch = G(z).detach()
                opt_C.zero_grad()
                gp = gradient_penalty(C, real_batch, fake_batch)
                loss_c = C(fake_batch).mean() - C(real_batch).mean() + LAMBDA_GP * gp
                loss_c.backward()
                nn.utils.clip_grad_norm_(C.parameters(), 1.0)
                opt_C.step()

            z = torch.randn(bs, GAN_WINDOW, device=device)
            fake_batch = G(z)
            opt_G.zero_grad()
            loss_g = -C(fake_batch).mean()
            loss_g.backward()
            nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()

        if verbose and epoch % 20 == 0:
            print(f"    wgan epoch {epoch:3d}/{WGAN_EPOCHS}")

    return G, (mu, scale)


def generate_wgan_dataset(G, mu_scale, anchor_price, scaler, target_len, seed, window=GAN_WINDOW):
    if G is None:
        return np.empty((0, SEQ_LENGTH, N_FEATURES)), np.empty((0,))

    mu, scale = mu_scale
    rng = np.random.default_rng(seed)
    n_windows = int(np.ceil(target_len / window))

    G.eval()
    with torch.no_grad():
        z = to_t(rng.standard_normal((n_windows, window)).astype(np.float32))
        gen = G(z).cpu().numpy()
    synthetic_returns = (gen.reshape(-1) * scale + mu)[:target_len]

    prices = [anchor_price]
    for r in synthetic_returns:
        prices.append(prices[-1] * (1.0 + float(r)))
    synth_df = build_features(pd.DataFrame({'Close': np.array(prices)}))

    if len(synth_df) < SEQ_LENGTH + 1:
        return np.empty((0, SEQ_LENGTH, N_FEATURES)), np.empty((0,))

    synth_scaled = scaler.transform(synth_df[FEATURE_COLS].values)
    return make_sequences(synth_scaled)


# ── smote-ts (identical to Day 3) ─────────────────────────────────────────────

def smote_neighbor_table(X_tr, k=SMOTE_K):
    n = len(X_tr)
    if n < k + 1:
        return None
    flat = X_tr.reshape(n, -1)
    dists = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.argsort(dists, axis=1)[:, :k]


def smote_ts_augment(X_tr, y_tr, neighbors, target_len, seed):
    if neighbors is None:
        return np.empty((0,) + X_tr.shape[1:]), np.empty((0,))

    rng = np.random.default_rng(seed)
    n, k = neighbors.shape
    anchors = rng.integers(0, n, size=target_len)
    picks   = rng.integers(0, k, size=target_len)
    lams    = rng.uniform(0.0, 1.0, size=target_len)

    X_synth = np.empty((target_len,) + X_tr.shape[1:], dtype=X_tr.dtype)
    y_synth = np.empty(target_len, dtype=y_tr.dtype)
    for idx in range(target_len):
        i = anchors[idx]
        j = neighbors[i, picks[idx]]
        lam = lams[idx]
        X_synth[idx] = X_tr[i] + lam * (X_tr[j] - X_tr[i])
        y_synth[idx] = y_tr[i] + lam * (y_tr[j] - y_tr[i])
    return X_synth, y_synth


# ── model (identical to Day 1-3) ──────────────────────────────────────────────

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


# ── training (identical to Day 1-3) ───────────────────────────────────────────

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


# ── metrics (identical to Day 1-3, plus direction_correct for McNemar's) ─────

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


def direction_correct(yt, yp):
    """Per-test-point boolean array of direction-correct predictions --
    da() is just this array's mean. Kept separate because McNemar's test
    needs the per-point array itself, not the aggregate."""
    yt = yt.flatten(); yp = yp.flatten()
    return np.sign(np.diff(yt)) == np.sign(np.diff(yp))


def da(yt, yp):
    return round(float(np.mean(direction_correct(yt, yp))), 4)


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


# ── statistical tests ─────────────────────────────────────────────────────────

def paired_ttest(baseline_da, method_da):
    """Paired t-test on the 10 trial-matched DA values (same seeds both sides)."""
    if len(baseline_da) < 2 or len(method_da) < 2:
        return float('nan'), float('nan')
    tstat, pval = stats.ttest_rel(method_da, baseline_da)
    return round(float(tstat), 4), round(float(pval), 4)


def wilcoxon_test(baseline_da, method_da):
    """Non-parametric alternative to the t-test -- doesn't assume DA is
    normally distributed across trials. Falls back to NaN if every paired
    difference is exactly zero (scipy raises on that input)."""
    diffs = np.array(method_da) - np.array(baseline_da)
    if np.allclose(diffs, 0):
        return float('nan'), 1.0
    try:
        stat, pval = stats.wilcoxon(baseline_da, method_da)
        return round(float(stat), 4), round(float(pval), 4)
    except ValueError:
        return float('nan'), float('nan')


def mcnemar_test(baseline_correct, method_correct):
    """McNemar's test on pooled per-test-point direction agreement.
    Pooling is valid here because baseline and method share the same seed
    (and therefore the same test set) trial-for-trial -- position j in
    trial i's baseline array and position j in trial i's method array are
    predictions on the exact same test point.

    Uses the exact binomial test on discordant pairs when there are few
    of them (< 25), the standard chi-square approximation with continuity
    correction otherwise -- this avoids depending on statsmodels, which
    isn't guaranteed to be on the Kaggle image."""
    baseline_correct = np.asarray(baseline_correct, dtype=bool)
    method_correct   = np.asarray(method_correct, dtype=bool)
    b = int(np.sum(baseline_correct & ~method_correct))   # baseline right, method wrong
    c = int(np.sum(~baseline_correct & method_correct))   # baseline wrong, method right
    n_discordant = b + c

    if n_discordant == 0:
        return {'b': b, 'c': c, 'statistic': float('nan'), 'p_value': 1.0, 'test': 'exact'}
    if n_discordant < 25:
        p = stats.binomtest(min(b, c), n_discordant, 0.5).pvalue
        return {'b': b, 'c': c, 'statistic': float('nan'), 'p_value': round(float(p), 4), 'test': 'exact'}
    chi2_stat = (abs(b - c) - 1) ** 2 / n_discordant
    p = 1 - stats.chi2.cdf(chi2_stat, df=1)
    return {'b': b, 'c': c, 'statistic': round(float(chi2_stat), 4),
            'p_value': round(float(p), 4), 'test': 'chi2'}


# ── per-asset, per-condition pipeline ─────────────────────────────────────────

CONDITIONS = ['baseline', 'cyclegan', 'wgan_gp', 'smote_ts']


def run_trials(condition, augment_fn, X_tr, y_tr, X_vl_t, y_vl_t, X_te_t, cs,
               y_true_prices, n=NUM_TRIALS):
    trials = []
    for i in range(n):
        seed = SEED + i * 13
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_synth, y_synth = augment_fn(seed)
        if len(X_synth) > 0:
            X_tr_aug = np.concatenate([X_tr, X_synth])
            y_tr_aug = np.concatenate([y_tr, y_synth])
        else:
            X_tr_aug, y_tr_aug = X_tr, y_tr

        model = LSTM(input_size=N_FEATURES).to(device)
        model = train(model, to_t(X_tr_aug), to_t(y_tr_aug), X_vl_t, y_vl_t, verbose=False)

        model.eval()
        with torch.no_grad():
            pred_s = model(X_te_t).cpu().numpy()
        pred_prices = cs.inverse_transform(pred_s[:, :1])

        metrics = evaluate(y_true_prices, pred_prices)
        correct = direction_correct(y_true_prices, pred_prices)
        trials.append({'metrics': metrics, 'correct': correct, 'n_synth': len(X_synth)})

    da_vals = [t['metrics']['DA'] for t in trials]
    print(f"  [{condition}] DA over {n} trials: "
          f"{np.mean(da_vals):.4f} ± {np.std(da_vals):.4f}")
    return trials


def run_asset(ticker):
    print(f"\n{'='*65}")
    print(f"  {ticker} — statistical validation ({NUM_TRIALS} trials x 4 conditions)")
    print(f"{'='*65}")

    df = download(ticker)
    if df.empty or len(df) < SEQ_LENGTH * 4:
        print(f"  SKIP {ticker}: insufficient data")
        return None

    train_df, val_df, test_df = temporal_split(df)
    scaler = fit_scaler(train_df)
    X_tr, y_tr = make_sequences(scaler.transform(train_df[FEATURE_COLS].values))
    X_vl, y_vl = make_sequences(scaler.transform(val_df[FEATURE_COLS].values))
    X_te, y_te = make_sequences(scaler.transform(test_df[FEATURE_COLS].values))
    if len(X_tr) < 10 or len(X_te) < 5:
        print(f"  SKIP {ticker}: too few sequences")
        return None

    cs = MinMaxScaler(feature_range=(-1, 1))
    cs.fit(train_df[['Close']].values)
    y_true_prices = cs.inverse_transform(y_te.reshape(-1, 1))

    anchor_price = float(train_df['Close'].squeeze().iloc[0])
    target_len   = int(round(AUG_RATIO * len(train_df)))

    X_vl_t, y_vl_t = to_t(X_vl), to_t(y_vl)
    X_te_t         = to_t(X_te)

    print("  training generators (once per asset)...")
    G_cyclegan, real_returns = train_cyclegan_for_asset(ticker, train_df, verbose=False)
    G_wgan, mu_scale         = train_wgan_gp_for_asset(ticker, train_df, verbose=False)
    neighbors                = smote_neighbor_table(X_tr, k=SMOTE_K)

    augmenters = {
        'baseline': lambda seed: (np.empty((0, SEQ_LENGTH, N_FEATURES)), np.empty((0,))),
        'cyclegan': lambda seed: generate_cyclegan_dataset(
            G_cyclegan, real_returns, anchor_price, scaler, target_len, seed),
        'wgan_gp':  lambda seed: generate_wgan_dataset(
            G_wgan, mu_scale, anchor_price, scaler, target_len, seed),
        'smote_ts': lambda seed: smote_ts_augment(X_tr, y_tr, neighbors, target_len, seed),
    }

    condition_trials = {}
    for condition in CONDITIONS:
        condition_trials[condition] = run_trials(
            condition, augmenters[condition], X_tr, y_tr,
            X_vl_t, y_vl_t, X_te_t, cs, y_true_prices,
        )

    baseline_trials = condition_trials['baseline']
    baseline_da = [t['metrics']['DA'] for t in baseline_trials]
    baseline_correct_pooled = np.concatenate([t['correct'] for t in baseline_trials])

    results = {}
    for condition in CONDITIONS:
        trials = condition_trials[condition]
        da_vals = [t['metrics']['DA'] for t in trials]
        summary = {
            'n': len(trials),
            'DA_mean': round(float(np.mean(da_vals)), 4),
            'DA_std':  round(float(np.std(da_vals)), 4),
            'IC_mean': round(float(np.mean([t['metrics']['IC'] for t in trials])), 4),
            'R2_price_mean': round(float(np.mean([t['metrics']['R2_price'] for t in trials])), 4),
            'Sharpe_mean':   round(float(np.mean([t['metrics']['Sharpe'] for t in trials])), 4),
        }
        if condition != 'baseline':
            correct_pooled = np.concatenate([t['correct'] for t in trials])
            tstat, tpval = paired_ttest(baseline_da, da_vals)
            wstat, wpval = wilcoxon_test(baseline_da, da_vals)
            mc = mcnemar_test(baseline_correct_pooled, correct_pooled)
            summary.update({
                'delta_pp': round((summary['DA_mean'] - np.mean(baseline_da)) * 100, 2),
                'ttest_t': tstat, 'ttest_p': tpval,
                'wilcoxon_stat': wstat, 'wilcoxon_p': wpval,
                'mcnemar_b': mc['b'], 'mcnemar_c': mc['c'],
                'mcnemar_stat': mc['statistic'], 'mcnemar_p': mc['p_value'],
                'mcnemar_test': mc['test'],
            })
        results[condition] = summary

    return results


# ── main ──────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  DAY 4 — STATISTICAL VALIDATION (10 TRIALS)")
print("="*70)

all_results = {}
for ticker in ALL_ASSETS:
    r = run_asset(ticker)
    if r:
        all_results[ticker] = r
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\ncompleted: {len(all_results)}/{len(ALL_ASSETS)} assets")

# results/day4_statistical_validation.csv -- DA summary per (asset, condition)
summary_rows = []
for ticker, conditions in all_results.items():
    for condition, s in conditions.items():
        summary_rows.append({
            'Asset': ticker, 'Condition': condition, 'N_trials': s['n'],
            'DA_mean': s['DA_mean'], 'DA_std': s['DA_std'],
            'IC_mean': s['IC_mean'], 'R2_price_mean': s['R2_price_mean'],
            'Sharpe_mean': s['Sharpe_mean'],
        })
if summary_rows:
    pd.DataFrame(summary_rows).to_csv('results/day4_statistical_validation.csv', index=False)
    print("saved: results/day4_statistical_validation.csv")

# results/day4_p_values_table.csv -- the three tests per (asset, method)
pval_rows = []
for ticker, conditions in all_results.items():
    for condition in ['cyclegan', 'wgan_gp', 'smote_ts']:
        s = conditions[condition]
        pval_rows.append({
            'Asset': ticker, 'Method': condition,
            'Baseline_DA': conditions['baseline']['DA_mean'],
            'Method_DA': s['DA_mean'], 'Delta_pp': s['delta_pp'],
            'TTest_t': s['ttest_t'], 'TTest_p': s['ttest_p'],
            'Wilcoxon_stat': s['wilcoxon_stat'], 'Wilcoxon_p': s['wilcoxon_p'],
            'McNemar_b': s['mcnemar_b'], 'McNemar_c': s['mcnemar_c'],
            'McNemar_stat': s['mcnemar_stat'], 'McNemar_p': s['mcnemar_p'],
            'McNemar_test_type': s['mcnemar_test'],
            'Significant_any': bool(
                (not np.isnan(s['ttest_p']) and s['ttest_p'] < 0.05)
                or (not np.isnan(s['wilcoxon_p']) and s['wilcoxon_p'] < 0.05)
                or (not np.isnan(s['mcnemar_p']) and s['mcnemar_p'] < 0.05)
            ),
        })
if pval_rows:
    pd.DataFrame(pval_rows).to_csv('results/day4_p_values_table.csv', index=False)
    print("saved: results/day4_p_values_table.csv")

print("\n" + "="*95)
print("  DAY 4 FINAL RESULTS")
print("="*95)
print(f"\n{'asset':8s} | {'method':9s} | {'baseline DA':>11} | {'method DA':>11} | "
      f"{'delta':>8} | {'t-test p':>9} | {'wilcoxon p':>10} | {'mcnemar p':>10}")
print("-"*95)
for row in pval_rows:
    sig = ' *' if row['Significant_any'] else ''
    print(f"{row['Asset']:8s} | {row['Method']:9s} | {row['Baseline_DA']:>11.4f} | "
          f"{row['Method_DA']:>11.4f} | {row['Delta_pp']:>+7.2f}pp | "
          f"{row['TTest_p']:>9.4f} | {row['Wilcoxon_p']:>10.4f} | "
          f"{row['McNemar_p']:>10.4f}{sig}")

print("\nday 4 done. next: day5 regime-conditional analysis (RCAH)")
