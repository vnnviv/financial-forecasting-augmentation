"""
Day 9 — reproducibility re-run for the Phase 2 paper.

Regenerates, in ONE session, the four numbers that are currently either
single-seed, unverified, or inconsistent between text and code:

  EXP 1  persistence R^2 from pinned raw data      -> Table "Cross-Ticker", col 1
         + writes data/raw/*.csv so the repo pins the exact prices used
  EXP 2  quantum ablation with per-trial DA        -> Table "Quantum Ablation" std devs
  EXP 3  augmentation at 10 seeds, 4 conditions    -> Table "Cross-Ticker" DA columns,
         + Sharpe cost curve on every trial           the three tests, and the
                                                      Sharpe-vs-cost table

Model, training and generator code is taken verbatim from
day4_statistical_validation.py and day6_qlstm_ablation02.py so the numbers
are comparable to the originals rather than to a reimplementation.

Vivian Chan | 2026
"""

import os, json, math, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf

# PennyLane is not in the Kaggle image. Install it on first import rather than
# letting a missing package take down a multi-hour run at line 28.
try:
    import pennylane as qml
except ModuleNotFoundError:
    import subprocess, sys
    print("pennylane not found - installing (needs Internet ON in notebook settings)")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pennylane'],
                   check=True)
    import pennylane as qml
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

# ── what to run ───────────────────────────────────────────────────────────────
RUN_EXP1_PERSISTENCE = True    # ~1 min
RUN_EXP2_QUANTUM     = True    # ~25-40 min  (QLSTM is CPU-only)
RUN_EXP3_AUGMENT     = True    # ~2-4 h on GPU  (4 assets x 4 conditions x 10 seeds)

# If a previous run already wrote data/raw/*.csv, set this True to train on the
# pinned files instead of re-downloading. This is the switch that makes the
# whole study reproducible after yfinance re-adjusts historical prices.
USE_PINNED_DATA = False

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

# ── config: Day 1-5 LSTM (unchanged) ──────────────────────────────────────────
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
SEQ_LENGTH  = 20
EPOCHS      = 100
PATIENCE    = 15
BATCH_SIZE  = 32
LR          = 0.001
NUM_TRIALS  = 10

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

SMOTE_K   = 5
AUG_RATIO = 1.0

# ── config: Day 6 quantum ablation (capacity-matched, hidden=32) ──────────────
Q_HIDDEN   = 32
Q_EPOCHS   = 30
Q_PATIENCE = 8
Q_BATCH    = 16
Q_LR       = 0.001
Q_TRIALS   = 5       # original table used 3; 5 verifies those and adds two
N_QUBITS   = 4
N_QLAYERS  = 1

# ── cost grid for the Sharpe table ────────────────────────────────────────────
COST_BPS = [0, 5, 10, 20, 50]

os.makedirs('results', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
print("config loaded")
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
QDEVICE = torch.device('cpu')


# ── quantum gate ───────────────────────────────────────────────────────────────
#
# Each of the QLSTM's 4 gates is: Linear(concat -> n_qubits) -> variational
# quantum circuit (angle-embed, entangle, measure) -> Linear(n_qubits ->
# hidden_size). qml.qnn.TorchLayer wraps the QNode as an nn.Module and
# handles the torch<->pennylane tensor conversion itself, which is what
# sidesteps the "use sourceTensor.clone().detach()" warning my original
# manual circuit code (torch.tensor([expval1, expval2, ...])) used to
# throw -- there's no manual tensor construction left to get wrong.

_dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(_dev, interface="torch")
def _quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.templates.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]


def make_vqc():
    weight_shapes = {"weights": (N_QLAYERS, N_QUBITS)}
    return qml.qnn.TorchLayer(_quantum_circuit, weight_shapes)


class QuantumGate(nn.Module):
    def __init__(self, in_features, hidden_size=Q_HIDDEN):
        super().__init__()
        self.pre  = nn.Linear(in_features, N_QUBITS)
        self.vqc  = make_vqc()
        self.post = nn.Linear(N_QUBITS, hidden_size)

    def forward(self, x):
        # scale into (-pi/2, pi/2) before angle embedding -- unscaled
        # Linear output can be any magnitude, and rotation angles wrap
        # every 2*pi, so an unscaled input makes the embedding meaningless
        angles = torch.tanh(self.pre(x)) * (np.pi / 2)
        q_out = self.vqc(angles)
        return self.post(q_out)


print("quantum gate defined "
      f"({N_QUBITS} qubits, {N_QLAYERS} entangling layer(s))")


# ── models ────────────────────────────────────────────────────────────────────

class ClassicalLSTM32(nn.Module):
    """Standard 2-layer LSTM, hidden=32 -- the capacity-matched baseline
    for this ablation specifically (not the hidden=64 used in Day 1-5)."""
    def __init__(self, input_size=N_FEATURES, hidden_size=Q_HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class QLSTMCell(nn.Module):
    """Same 4-gate structure as a classical LSTM cell -- forget, input,
    update (cell candidate), output -- except each gate is a QuantumGate
    instead of a plain Linear layer."""
    def __init__(self, input_size, hidden_size=Q_HIDDEN):
        super().__init__()
        self.hidden_size = hidden_size
        concat = input_size + hidden_size
        self.forget_gate = QuantumGate(concat, hidden_size)
        self.input_gate  = QuantumGate(concat, hidden_size)
        self.update_gate = QuantumGate(concat, hidden_size)
        self.output_gate = QuantumGate(concat, hidden_size)

    def forward(self, x, h, c):
        v = torch.cat([x, h], dim=-1)
        f = torch.sigmoid(self.forget_gate(v))
        i = torch.sigmoid(self.input_gate(v))
        g = torch.tanh(self.update_gate(v))
        o = torch.sigmoid(self.output_gate(v))
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class QLSTM(nn.Module):
    """Same forward signature as ClassicalLSTM -- (batch, seq, features)
    in, (batch, 1) out -- so both models plug into the exact same train()
    function unchanged. Loops the cell over timesteps by hand since a
    quantum gate can't be expressed as a single nn.LSTM call."""
    def __init__(self, input_size=N_FEATURES, hidden_size=Q_HIDDEN):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = QLSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        h = torch.zeros(batch, self.hidden_size, device=x.device)
        c = torch.zeros(batch, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :], h, c)
        return self.fc(h)


print("models defined (ClassicalLSTM + QLSTM, both hidden=32)")


# ── training (same loop for both models) ──────────────────────────────────────

def qtrain(model, X_tr, y_tr, X_vl, y_vl, verbose=True):
    opt  = optim.Adam(model.parameters(), lr=Q_LR, weight_decay=1e-5)
    crit = nn.MSELoss()

    best_val   = float('inf')
    no_imp     = 0
    best_state = None

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=Q_BATCH, shuffle=True)

    for ep in range(1, Q_EPOCHS + 1):
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

        if verbose and ep % 5 == 0:
            print(f"    ep {ep:3d}/{Q_EPOCHS} | train={ep_loss / len(loader):.5f} "
                  f"| val={val_loss:.5f}")

        if val_loss < best_val:
            best_val   = val_loss
            no_imp     = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= Q_PATIENCE:
                if verbose:
                    print(f"    early stop @ ep {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ── metrics (identical to Day 1-5) ────────────────────────────────────────────



# ══════════════════════════════════════════════════════════════════════════════
#  pinned-data layer
#
#  yfinance re-adjusts historical closes over time (splits, dividend
#  restatements), so a download today does not reproduce a download from
#  months ago. Every price series this run touches is written to
#  data/raw/<TICKER>.csv on first use; committing that folder is what makes
#  the persistence R^2 in the paper reproducible by a reader.
# ══════════════════════════════════════════════════════════════════════════════

RAW_DIR = 'data/raw'


def raw_path(ticker):
    return os.path.join(RAW_DIR, f"{ticker.replace('-', '_')}.csv")


def load_prices(ticker):
    """Returns a one-column Close DataFrame, either pinned or freshly pulled."""
    path = raw_path(ticker)
    if USE_PINNED_DATA and os.path.exists(path):
        px = pd.read_csv(path)
        print(f"  {ticker}: {len(px)} rows (pinned, {path})")
        return px[['Close']].copy()

    raw = yf.download(ticker, start=TRAIN_START, end=OOS_END,
                      auto_adjust=True, progress=False)
    if raw.empty:
        print(f"  no data for {ticker}")
        return pd.DataFrame()
    px = raw[['Close']].dropna().copy()
    px.insert(0, 'Date', raw.index[-len(px):].strftime('%Y-%m-%d'))
    px.reset_index(drop=True, inplace=True)
    px.to_csv(path, index=False)
    print(f"  {ticker}: {len(px)} rows (downloaded; pinned to {path})")
    return px[['Close']].copy()


def get_df(ticker):
    """Replaces day4's download(): same output, but routed through the pin."""
    px = load_prices(ticker)
    if px.empty:
        return pd.DataFrame()
    return build_features(px)


def to_tc(arr):
    """CPU tensor — the QLSTM's PennyLane layer has no CUDA path."""
    return torch.tensor(arr, dtype=torch.float32, device=QDEVICE)


def sharpe_curve(y_true, y_pred, grid=COST_BPS):
    return {str(c): sharpe(y_true, y_pred, cost_bps=c) for c in grid}


def msd(vals):
    """mean / population std / n, skipping nans — matches the originals."""
    v = [float(x) for x in vals if x is not None and not np.isnan(float(x))]
    if not v:
        return {'mean': float('nan'), 'std': float('nan'), 'n': 0}
    return {'mean': round(float(np.mean(v)), 4),
            'std':  round(float(np.std(v)), 4),
            'n':    len(v)}


RESULTS = {}


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 1 — persistence R^2 from pinned data
#
#  Fixes: column 1 of the cross-ticker table. Reports the value this run
#  produces next to the value currently in the paper, so any drift caused by
#  yfinance re-adjustment is documented rather than silently inherited.
# ══════════════════════════════════════════════════════════════════════════════

PAPER_PERSIST_R2 = {'AAPL': 0.984, 'MSFT': 0.976, 'GOOGL': 0.971, 'BTC-USD': 0.945}


def exp1_persistence():
    print("\n" + "=" * 78)
    print("  EXP 1 — persistence baseline from pinned data")
    print("=" * 78)

    rows = []
    for ticker in ALL_ASSETS:
        df = get_df(ticker)
        if df.empty:
            continue

        # same chronological split, same test segment, as every other table
        _, _, test_df = temporal_split(df)
        prices = test_df['Close'].squeeze().values.astype(float)

        # P_hat_{t+1} = P_t
        actual = prices[1:]
        naive  = prices[:-1]
        r2     = r2_score(actual, naive)
        rmse   = float(np.sqrt(mean_squared_error(actual, naive)))
        da_p   = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(naive))))

        paper = PAPER_PERSIST_R2[ticker]
        drift = r2 - paper
        flag  = '  <- DIFFERS FROM PAPER' if abs(drift) > 0.002 else ''
        print(f"  {ticker:8s}  R2={r2:.4f}  (paper {paper:.3f}, drift {drift:+.4f})"
              f"  RMSE={rmse:.4f}  DA={da_p:.4f}{flag}")

        rows.append({'asset': ticker, 'persistence_R2': round(float(r2), 4),
                     'paper_R2': paper, 'drift': round(float(drift), 4),
                     'RMSE': round(rmse, 4), 'persistence_DA': round(da_p, 4),
                     'n_test': int(len(prices))})

    RESULTS['exp1_persistence'] = rows
    pd.DataFrame(rows).to_csv('results/day9_persistence.csv', index=False)
    print("\n  saved: results/day9_persistence.csv")
    print("  saved: data/raw/*.csv  <- COMMIT THIS FOLDER")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2 — quantum ablation with per-trial DA
#
#  Fixes: the identical +/-0.71% std devs in the quantum table. Prints every
#  trial's DA so the std can be checked by hand, and reports statistics at
#  n=3 (the published configuration) and at n=Q_TRIALS.
# ══════════════════════════════════════════════════════════════════════════════

def exp2_quantum(ticker='AAPL'):
    print("\n" + "=" * 78)
    print(f"  EXP 2 — quantum ablation, {ticker}, hidden={Q_HIDDEN}, {Q_TRIALS} trials")
    print("=" * 78)

    df = get_df(ticker)
    if df.empty:
        return None

    train_df, val_df, test_df = temporal_split(df)
    scaler = fit_scaler(train_df)
    X_tr, y_tr = make_sequences(scaler.transform(train_df[FEATURE_COLS].values))
    X_vl, y_vl = make_sequences(scaler.transform(val_df[FEATURE_COLS].values))
    X_te, y_te = make_sequences(scaler.transform(test_df[FEATURE_COLS].values))

    cs = MinMaxScaler(feature_range=(-1, 1))
    cs.fit(train_df[['Close']].values)
    y_true_prices = cs.inverse_transform(y_te.reshape(-1, 1))

    # classical on GPU if present; quantum forced to CPU
    X_vl_g, y_vl_g, X_te_g = to_t(X_vl), to_t(y_vl), to_t(X_te)
    X_vl_c, y_vl_c, X_te_c = to_tc(X_vl), to_tc(y_vl), to_tc(X_te)

    out = {}
    for name, cls, dev_, Xv, yv, Xt in [
        ('classical', ClassicalLSTM32, device,   X_vl_g, y_vl_g, X_te_g),
        ('qlstm',     QLSTM,           QDEVICE,  X_vl_c, y_vl_c, X_te_c),
    ]:
        print(f"\n  -- {name} --")
        per_trial = []
        for i in range(Q_TRIALS):
            seed = SEED + i * 13
            torch.manual_seed(seed); np.random.seed(seed)

            model = cls().to(dev_)
            model = qtrain(model,
                           torch.tensor(X_tr, dtype=torch.float32, device=dev_),
                           torch.tensor(y_tr, dtype=torch.float32, device=dev_),
                           Xv, yv, verbose=False)
            model.eval()
            with torch.no_grad():
                pred_s = model(Xt).cpu().numpy()
            pred_prices = cs.inverse_transform(pred_s[:, :1])

            m = {'seed': seed,
                 'DA':       da(y_true_prices, pred_prices),
                 'R2_price': round(float(r2_score(y_true_prices.flatten(),
                                                  pred_prices.flatten())), 4),
                 'IC':       ic(y_true_prices, pred_prices)}
            per_trial.append(m)
            print(f"    trial {i+1} (seed {seed}): DA={m['DA']:.4f}  "
                  f"R2={m['R2_price']:.4f}  IC={m['IC']:.4f}")

        das = [t['DA'] for t in per_trial]
        out[name] = {
            'per_trial': per_trial,
            'DA_first3': msd(das[:3]),      # the published configuration
            'DA_all':    msd(das),
            'R2_price':  msd([t['R2_price'] for t in per_trial]),
            'IC':        msd([t['IC'] for t in per_trial]),
        }
        print(f"    DA over first 3 : {out[name]['DA_first3']['mean']:.4f} "
              f"+/- {out[name]['DA_first3']['std']:.4f}")
        print(f"    DA over all {Q_TRIALS}  : {out[name]['DA_all']['mean']:.4f} "
              f"+/- {out[name]['DA_all']['std']:.4f}")

    c3, q3 = out['classical']['DA_first3'], out['qlstm']['DA_first3']
    cA, qA = out['classical']['DA_all'],    out['qlstm']['DA_all']
    out['gap_pp_first3'] = round((c3['mean'] - q3['mean']) * 100, 2)
    out['gap_pp_all']    = round((cA['mean'] - qA['mean']) * 100, 2)
    out['std_identical_first3'] = bool(abs(c3['std'] - q3['std']) < 1e-9)

    print(f"\n  gap (3 trials) : {out['gap_pp_first3']:+.2f} pp   "
          f"[paper reports +1.13 pp]")
    print(f"  gap ({Q_TRIALS} trials) : {out['gap_pp_all']:+.2f} pp")
    print(f"  std devs identical at n=3? {out['std_identical_first3']}"
          f"   (paper prints 0.71% for BOTH models — if this is False, "
          f"one of them is a transcription error)")

    RESULTS['exp2_quantum'] = out
    with open('results/day9_quantum.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("  saved: results/day9_quantum.json")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 3 — augmentation at 10 seeds, with the cost curve on every trial
#
#  Fixes: (a) the Sharpe-vs-cost table, which currently reports a single
#  seed while every other number in the paper is a mean over trials;
#  (b) the DA columns of the cross-ticker table, so they come from the same
#  10-seed run as the p-values rather than from the 5-seed run.
# ══════════════════════════════════════════════════════════════════════════════

def exp3_augmentation():
    print("\n" + "=" * 78)
    print(f"  EXP 3 — augmentation, {NUM_TRIALS} seeds x {len(CONDITIONS)} conditions")
    print("=" * 78)

    all_assets_out = {}

    for ticker in ALL_ASSETS:
        print(f"\n{'-' * 78}\n  {ticker}\n{'-' * 78}")
        df = get_df(ticker)
        if df.empty or len(df) < SEQ_LENGTH * 4:
            print(f"  SKIP {ticker}")
            continue

        train_df, val_df, test_df = temporal_split(df)
        scaler = fit_scaler(train_df)
        X_tr, y_tr = make_sequences(scaler.transform(train_df[FEATURE_COLS].values))
        X_vl, y_vl = make_sequences(scaler.transform(val_df[FEATURE_COLS].values))
        X_te, y_te = make_sequences(scaler.transform(test_df[FEATURE_COLS].values))
        if len(X_tr) < 10 or len(X_te) < 5:
            print(f"  SKIP {ticker}: too few sequences")
            continue

        cs = MinMaxScaler(feature_range=(-1, 1))
        cs.fit(train_df[['Close']].values)
        y_true_prices = cs.inverse_transform(y_te.reshape(-1, 1))

        anchor_price = float(train_df['Close'].squeeze().iloc[0])
        target_len   = int(round(AUG_RATIO * len(train_df)))
        X_vl_t, y_vl_t, X_te_t = to_t(X_vl), to_t(y_vl), to_t(X_te)

        print("  training generators (once per asset, train segment only)...")
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

        cond_out = {}
        for condition in CONDITIONS:
            print(f"\n  [{condition}]")
            trials = []
            for i in range(NUM_TRIALS):
                seed = SEED + i * 13
                torch.manual_seed(seed); np.random.seed(seed)

                X_synth, y_synth = augmenters[condition](seed)
                if len(X_synth) > 0:
                    X_in = np.concatenate([X_tr, X_synth])
                    y_in = np.concatenate([y_tr, y_synth])
                else:
                    X_in, y_in = X_tr, y_tr

                model = LSTM(input_size=N_FEATURES).to(device)
                model = train(model, to_t(X_in), to_t(y_in),
                              X_vl_t, y_vl_t, verbose=False)
                model.eval()
                with torch.no_grad():
                    pred_s = model(X_te_t).cpu().numpy()
                pred_prices = cs.inverse_transform(pred_s[:, :1])

                trials.append({
                    'seed':    seed,
                    'DA':      da(y_true_prices, pred_prices),
                    'R2_ret':  return_r2(y_true_prices, pred_prices),
                    'IC':      ic(y_true_prices, pred_prices),
                    # the fix: cost curve on EVERY trial, not just the first
                    'sharpe':  sharpe_curve(y_true_prices, pred_prices),
                    'correct': direction_correct(y_true_prices, pred_prices),
                })
                print(f"    seed {seed}: DA={trials[-1]['DA']:.4f}  "
                      f"Sharpe@0={trials[-1]['sharpe']['0']}")

            das = [t['DA'] for t in trials]
            curve = {str(c): msd([t['sharpe'][str(c)] for t in trials])
                     for c in COST_BPS}
            cond_out[condition] = {
                'DA':        msd(das),
                'DA_trials': das,
                'R2_ret':    msd([t['R2_ret'] for t in trials]),
                'IC':        msd([t['IC'] for t in trials]),
                'sharpe_by_cost': curve,
                '_correct':  np.concatenate([t['correct'] for t in trials]),
            }
            print(f"    DA: {cond_out[condition]['DA']['mean']:.4f} "
                  f"+/- {cond_out[condition]['DA']['std']:.4f}")
            print("    Sharpe by cost (mean +/- std over "
                  f"{NUM_TRIALS} seeds):")
            for c in COST_BPS:
                s = curve[str(c)]
                print(f"      {c:>2} bps: {s['mean']:+.4f} +/- {s['std']:.4f}")

        # ── paired tests, baseline vs each augmentation ──────────────────────
        base_da      = cond_out['baseline']['DA_trials']
        base_correct = cond_out['baseline']['_correct']
        tests = {}
        for condition in CONDITIONS:
            if condition == 'baseline':
                continue
            t_stat, t_p = paired_ttest(base_da, cond_out[condition]['DA_trials'])
            w_stat, w_p = wilcoxon_test(base_da, cond_out[condition]['DA_trials'])
            mc = mcnemar_test(base_correct, cond_out[condition]['_correct'])
            delta = (cond_out[condition]['DA']['mean'] - cond_out['baseline']['DA']['mean']) * 100
            tests[condition] = {
                'delta_pp':   round(float(delta), 2),
                'ttest_p':    None if np.isnan(t_p) else round(float(t_p), 4),
                'wilcoxon_p': None if np.isnan(w_p) else round(float(w_p), 4),
                'mcnemar_p':  mc['p_value'],
                'all_three_sig': bool(
                    (not np.isnan(t_p) and t_p < 0.05) and
                    (not np.isnan(w_p) and w_p < 0.05) and
                    (mc['p_value'] is not None and mc['p_value'] < 0.05)),
                'survives_bonferroni_4': bool(
                    (not np.isnan(t_p) and t_p < 0.0125) and
                    (not np.isnan(w_p) and w_p < 0.0125) and
                    (mc['p_value'] is not None and mc['p_value'] < 0.0125)),
            }
            print(f"\n  {ticker} {condition}: delta={tests[condition]['delta_pp']:+.2f} pp  "
                  f"t={tests[condition]['ttest_p']}  w={tests[condition]['wilcoxon_p']}  "
                  f"mc={tests[condition]['mcnemar_p']}  "
                  f"all3={tests[condition]['all_three_sig']}")

        for c in cond_out.values():
            c.pop('_correct', None)
        all_assets_out[ticker] = {'conditions': cond_out, 'tests': tests}

    RESULTS['exp3_augmentation'] = all_assets_out
    with open('results/day9_augmentation.json', 'w') as f:
        json.dump(all_assets_out, f, indent=2)
    print("\n  saved: results/day9_augmentation.json")
    return all_assets_out


# ══════════════════════════════════════════════════════════════════════════════
#  paper-ready tables
# ══════════════════════════════════════════════════════════════════════════════

def print_paper_tables():
    print("\n" + "=" * 78)
    print("  PAPER-READY NUMBERS")
    print("=" * 78)

    aug = RESULTS.get('exp3_augmentation')
    if aug:
        print("\n  Sharpe vs transaction cost (baseline LSTM, mean over "
              f"{NUM_TRIALS} seeds)")
        print("  " + "-" * 62)
        print(f"  {'Asset':10s}" + "".join(f"{str(c)+' bps':>11}" for c in COST_BPS))
        for ticker, blk in aug.items():
            curve = blk['conditions']['baseline']['sharpe_by_cost']
            cells = "".join(f"{curve[str(c)]['mean']:>11.2f}" for c in COST_BPS)
            print(f"  {ticker:10s}{cells}")
        print("\n  (std devs — if any exceeds the mean, say so in the caption)")
        for ticker, blk in aug.items():
            curve = blk['conditions']['baseline']['sharpe_by_cost']
            cells = "".join(f"{curve[str(c)]['std']:>11.2f}" for c in COST_BPS)
            print(f"  {ticker:10s}{cells}")

        print("\n  Cross-ticker DA (%), all from this 10-seed run")
        print("  " + "-" * 62)
        print(f"  {'Asset':10s}{'LSTM':>12}{'WGAN-GP':>12}{'CycleGAN':>12}{'SMOTE-TS':>12}")
        for ticker, blk in aug.items():
            cc = blk['conditions']
            row = "".join(f"{cc[k]['DA']['mean'] * 100:>12.2f}"
                          for k in ['baseline', 'wgan_gp', 'cyclegan', 'smote_ts'])
            print(f"  {ticker:10s}{row}")

    q = RESULTS.get('exp2_quantum')
    if q:
        print("\n  Quantum ablation (AAPL)")
        print("  " + "-" * 62)
        for n_label, key in [('3 trials (published)', 'DA_first3'),
                             (f'{Q_TRIALS} trials', 'DA_all')]:
            c, qq = q['classical'][key], q['qlstm'][key]
            print(f"  {n_label:22s} classical {c['mean']*100:.2f}% +/- {c['std']*100:.2f}%"
                  f"   QLSTM {qq['mean']*100:.2f}% +/- {qq['std']*100:.2f}%")

    p1 = RESULTS.get('exp1_persistence')
    if p1:
        print("\n  Persistence R^2")
        print("  " + "-" * 62)
        for r in p1:
            note = '  <- update the table' if abs(r['drift']) > 0.002 else '  ok'
            print(f"  {r['asset']:10s} this run {r['persistence_R2']:.4f}   "
                  f"paper {r['paper_R2']:.3f}{note}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__' or True:
    if RUN_EXP1_PERSISTENCE:
        exp1_persistence()
    if RUN_EXP2_QUANTUM:
        exp2_quantum('AAPL')
    if RUN_EXP3_AUGMENT:
        exp3_augmentation()

    print_paper_tables()

    with open('results/day9_all_results.json', 'w') as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print("\n" + "=" * 78)
    print("  DONE — download results/day9_all_results.json and send it back")
    print("=" * 78)
