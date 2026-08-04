temping
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

# LSTM model / training (identical to Day 1/Day 2 -- same architecture and
# hyperparameters throughout, so any DA change is attributable to the
# training data, not a model change)
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
SEQ_LENGTH  = 20
EPOCHS      = 100
PATIENCE    = 15
BATCH_SIZE  = 32
LR          = 0.001
NUM_TRIALS  = 5     # same trial count as Day 1/Day 2 -- paired comparison

# data
ALL_ASSETS  = ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD']
TRAIN_START = '2020-01-01'
OOS_END     = '2023-12-31'

FEATURE_COLS = ['Close', 'RSI_14', 'SMA_5', 'SMA_10', 'SMA_20',
                'Vol_20', 'Mom_5', 'BB_pos']
N_FEATURES   = len(FEATURE_COLS)

# WGAN-GP
GAN_WINDOW  = 32      # length of one generated return window
GAN_HIDDEN  = 64       # MLP hidden width
WGAN_EPOCHS = 60
WGAN_BATCH  = 16
WGAN_LR     = 1e-4
N_CRITIC    = 5        # critic steps per generator step (standard WGAN-GP)
LAMBDA_GP   = 10.0      # gradient penalty weight
GAN_SEED    = 42        # one WGAN-GP trained per asset, not per trial

# SMOTE-TS
SMOTE_K = 5   # neighbors to interpolate against

# augmentation
AUG_RATIO = 1.0   # synthetic days/sequences added to train = AUG_RATIO * len(train)

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
print("config loaded")


# ── data (identical to Day 1/Day 2) ───────────────────────────────────────────

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
    8 features from Close price. Used for both real and WGAN-GP synthetic
    price series -- always called fresh, never fed pre-computed real
    features. That's what makes the WGAN-GP augmentation leakage-safe.
    (SMOTE-TS doesn't go through this -- see the SMOTE-TS section below.)
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


# ── wgan-gp -- synthetic return generation ────────────────────────────────────
#
# unconditional: the generator maps noise straight to a return window,
# no domain-translation step like Day 2's CycleGAN needed. Trained once
# per asset against a Wasserstein critic with a gradient penalty.

def make_windows(series_1d, window):
    if len(series_1d) < window:
        return np.empty((0, window), dtype=np.float32)
    n = len(series_1d) - window + 1
    return np.stack([series_1d[i:i + window] for i in range(n)]).astype(np.float32)


class WGANGenerator(nn.Module):
    def __init__(self, latent_dim=GAN_WINDOW, hidden=GAN_HIDDEN, window=GAN_WINDOW):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, window), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class WGANCritic(nn.Module):
    def __init__(self, window=GAN_WINDOW, hidden=GAN_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def gradient_penalty(critic, real, fake):
    """Standard WGAN-GP penalty -- pushes the critic's gradient norm on
    interpolated samples toward 1, which is what makes training stable
    instead of blowing up to NaN partway through."""
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
    """One WGAN-GP per asset (not per trial), same reasoning as Day 2's
    per-asset CycleGAN: training is the expensive part, downstream trials
    only need different noise draws and LSTM init seeds. Returns
    (G, (mu, scale)) -- G is None if there isn't enough data to train on,
    in which case callers skip WGAN-GP augmentation for that asset."""
    real_returns = train_df['Close'].squeeze().pct_change().dropna().values.astype(np.float32)
    real_windows = make_windows(real_returns, GAN_WINDOW)

    if len(real_windows) < WGAN_BATCH:
        print(f"  WARNING: {ticker} has too few return windows to train "
              f"WGAN-GP ({len(real_windows)} < {WGAN_BATCH}); skipping "
              f"WGAN-GP augmentation for this asset.")
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
        c_total, g_total = 0.0, 0.0

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
                c_total += loss_c.item()

            z = torch.randn(bs, GAN_WINDOW, device=device)
            fake_batch = G(z)
            opt_G.zero_grad()
            loss_g = -C(fake_batch).mean()
            loss_g.backward()
            nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()
            g_total += loss_g.item()

        if verbose and epoch % 15 == 0:
            print(f"    wgan epoch {epoch:3d}/{WGAN_EPOCHS} | "
                  f"C={c_total / (n_batches * N_CRITIC):.4f} | "
                  f"G={g_total / n_batches:.4f}")

    return G, (mu, scale)


def generate_wgan_dataset(G, mu_scale, anchor_price, scaler, target_len, seed,
                          window=GAN_WINDOW):
    """Same downstream pattern as Day 2: generate returns -> compound into
    a price path -> build_features() from scratch -> scale with the real
    train scaler -> make_sequences(). No real feature value is ever
    copied onto a synthetic row."""
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


# ── smote-ts -- interpolated real sequences ───────────────────────────────────
#
# no training, no price reconstruction. Interpolates directly between real
# training sequences that are already in scaled feature space. The
# neighbor table is built once per asset (it only depends on X_tr, which
# doesn't change across trials) and reused for every trial's random draws.

def smote_neighbor_table(X_tr, k=SMOTE_K):
    """k nearest real neighbors of every real training sequence, by
    Euclidean distance in flattened feature space."""
    n = len(X_tr)
    if n < k + 1:
        return None
    flat = X_tr.reshape(n, -1)
    dists = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.argsort(dists, axis=1)[:, :k]


def smote_ts_augment(X_tr, y_tr, neighbors, target_len, seed):
    """For each synthetic example: pick a real anchor sequence, pick one
    of its k nearest real neighbors, interpolate both X and y at a random
    ratio in (0, 1). This is the whole method -- no separate feature
    recomputation step, because interpolating real feature vectors IS
    SMOTE, not an approximation of it."""
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


# ── model (identical to Day 1/Day 2) ──────────────────────────────────────────

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


# ── training (identical to Day 1/Day 2) ───────────────────────────────────────

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


# ── metrics (identical to Day 1/Day 2) ────────────────────────────────────────

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


# ── per-asset, per-method pipeline ────────────────────────────────────────────

def load_prior_results():
    """Loads whatever's already sitting in the working directory: Day 1's
    baseline (accepting either the repo's results/day1_baseline.csv or
    the day1_cross_asset_results.csv some earlier Day 1 scripts save at
    the root) and Day 2's CycleGAN result, if present. Both are optional
    -- this notebook still runs and still saves its own results without
    them, it just skips those columns/bars in the comparison."""
    baseline_df = None
    for path, rename in [('results/day1_baseline.csv', {}),
                         ('day1_cross_asset_results.csv', {'LSTM_DirAcc': 'DA_mean'})]:
        if os.path.exists(path):
            df = pd.read_csv(path).rename(columns=rename)
            if 'Asset' in df.columns and 'DA_mean' in df.columns:
                print(f"  loaded Day 1 baseline from {path}")
                baseline_df = df.set_index('Asset')
                break
    if baseline_df is None:
        print("  NOTE: no Day 1 baseline CSV found -- continuing without it.")

    cyclegan_df = None
    if os.path.exists('results/day2_cyclegan_results.csv'):
        cyclegan_df = pd.read_csv('results/day2_cyclegan_results.csv').set_index('Asset')
        print("  loaded Day 2 CycleGAN results from results/day2_cyclegan_results.csv")
    else:
        print("  NOTE: no Day 2 CycleGAN results found -- fig5 will skip that column.")

    return baseline_df, cyclegan_df


def run_trials(method, augment_fn, X_tr, y_tr, X_vl_t, y_vl_t, X_te_t, cs,
               y_true_prices, n=NUM_TRIALS):
    """Runs n LSTM trials for one augmentation method. `augment_fn(seed)`
    returns (X_synth, y_synth) for that trial -- everything else (real
    data, validation, test) is identical across methods, only the
    augmentation source changes."""
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
        model = train(model, to_t(X_tr_aug), to_t(y_tr_aug), X_vl_t, y_vl_t,
                      verbose=(i == 0))

        model.eval()
        with torch.no_grad():
            pred_s = model(X_te_t).cpu().numpy()
        pred_prices = cs.inverse_transform(pred_s[:, :1])

        metrics = evaluate(y_true_prices, pred_prices,
                           label=f'{method} OOS' if i == 0 else '')
        print(f"  [{method}] trial {i+1}: DA={metrics['DA']:.4f} | "
              f"IC={metrics['IC']:.4f} | n_synth={len(X_synth)}")
        trials.append({'metrics': metrics, 'n_synth': len(X_synth)})
    return trials


def aggregate(trials, baseline_da=None):
    keys = ['MSE', 'RMSE', 'R2_price', 'R2_ret', 'IC', 'DA', 'Sharpe']
    out = {'n': len(trials)}
    for k in keys:
        vals = [float(t['metrics'][k]) for t in trials
                if t['metrics'].get(k) is not None
                and not np.isnan(float(t['metrics'].get(k, float('nan'))))]
        if vals:
            out[f'{k}_mean'] = round(np.mean(vals), 4)
            out[f'{k}_std']  = round(np.std(vals), 4)
    out['n_synth_mean'] = round(np.mean([t['n_synth'] for t in trials]), 1)

    if baseline_da is not None:
        out['baseline_DA'] = round(float(baseline_da), 4)
        out['delta_pp'] = round((out.get('DA_mean', float('nan')) - baseline_da) * 100, 2)
        da_vals = [t['metrics']['DA'] for t in trials]
        if len(da_vals) > 1:
            tstat, pval = stats.ttest_1samp(da_vals, popmean=baseline_da)
            out['ttest_t'] = round(float(tstat), 4)
            out['ttest_p'] = round(float(pval), 4)
    return out


def run_asset(ticker, baseline_row):
    print(f"\n{'='*65}")
    print(f"  {ticker} — WGAN-GP + SMOTE-TS ({NUM_TRIALS} trials each)")
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

    baseline_da = float(baseline_row['DA_mean']) if baseline_row is not None else None

    # -- WGAN-GP --
    G, mu_scale = train_wgan_gp_for_asset(ticker, train_df)
    wgan_trials = run_trials(
        'WGAN-GP',
        lambda seed: generate_wgan_dataset(G, mu_scale, anchor_price, scaler, target_len, seed),
        X_tr, y_tr, X_vl_t, y_vl_t, X_te_t, cs, y_true_prices,
    )
    wgan_summary = aggregate(wgan_trials, baseline_da)
    print(f"  integrity check (WGAN-GP): {wgan_trials[0]['n_synth']} synthetic "
          f"examples added to TRAIN only; validation and test are 100% real.")

    # -- SMOTE-TS --
    neighbors = smote_neighbor_table(X_tr, k=SMOTE_K)
    smote_trials = run_trials(
        'SMOTE-TS',
        lambda seed: smote_ts_augment(X_tr, y_tr, neighbors, target_len, seed),
        X_tr, y_tr, X_vl_t, y_vl_t, X_te_t, cs, y_true_prices,
    )
    smote_summary = aggregate(smote_trials, baseline_da)
    print(f"  integrity check (SMOTE-TS): {smote_trials[0]['n_synth']} synthetic "
          f"examples added to TRAIN only; validation and test are 100% real.")

    return {'wgan_gp': wgan_summary, 'smote_ts': smote_summary}


# ── figures ────────────────────────────────────────────────────────────────────

def fig5_all_methods(summaries, cyclegan_df):
    tickers = list(summaries.keys())
    baseline = [summaries[t]['wgan_gp'].get('baseline_DA', float('nan')) for t in tickers]
    cyclegan = [
        float(cyclegan_df.loc[t]['CycleGAN_DA_mean'])
        if cyclegan_df is not None and t in cyclegan_df.index else float('nan')
        for t in tickers
    ]
    wgan  = [summaries[t]['wgan_gp'].get('DA_mean', float('nan')) for t in tickers]
    smote = [summaries[t]['smote_ts'].get('DA_mean', float('nan')) for t in tickers]

    x = np.arange(len(tickers)); w = 0.2
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('#FAFAFA')

    ax.bar(x - 1.5*w, baseline, w, color='#95A5A6', alpha=0.9, label='baseline (real only)', edgecolor='white')
    ax.bar(x - 0.5*w, cyclegan, w, color='#8B5A8D', alpha=0.9, label='CycleGAN (Day 2)', edgecolor='white')
    ax.bar(x + 0.5*w, wgan,     w, color='#E74C3C', alpha=0.9, label='WGAN-GP', edgecolor='white')
    ax.bar(x + 1.5*w, smote,    w, color='#2ECC71', alpha=0.9, label='SMOTE-TS', edgecolor='white')
    ax.axhline(0.50, color='black', ls='--', lw=1.5, label='random (50%)', alpha=0.6)

    ax.set_xticks(x); ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel('directional accuracy', fontsize=11)
    ax.set_title('Figure 5 — all augmentation methods vs baseline\n'
                 'directional accuracy, real held-out test set only',
                 fontsize=12, fontweight='bold', color='#5A3A5C')
    ax.legend(fontsize=9); ax.set_facecolor('#FFFAFD')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('figures/fig5_all_methods_comparison.png', dpi=200,
                bbox_inches='tight', facecolor='#FAFAFA')
    plt.show()
    print("saved: figures/fig5_all_methods_comparison.png")


# ── main ──────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  DAY 3 — WGAN-GP + SMOTE-TS AUGMENTATION")
print("="*70)

baseline_df, cyclegan_df = load_prior_results()

summaries = {}
for ticker in ALL_ASSETS:
    baseline_row = None
    if baseline_df is not None and ticker in baseline_df.index:
        baseline_row = baseline_df.loc[ticker]
    s = run_asset(ticker, baseline_row)
    if s:
        summaries[ticker] = s
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\ncompleted: {len(summaries)}/{len(ALL_ASSETS)} assets")

if summaries:
    fig5_all_methods(summaries, cyclegan_df)

rows = []
for ticker, methods in summaries.items():
    for method_name, s in methods.items():
        rows.append({
            'Asset':          ticker,
            'Method':         method_name,
            'N_trials':       s.get('n', NUM_TRIALS),
            'Baseline_DA':    s.get('baseline_DA', float('nan')),
            'DA_mean':        s.get('DA_mean', float('nan')),
            'DA_std':         s.get('DA_std', float('nan')),
            'Delta_pp':       s.get('delta_pp', float('nan')),
            'IC_mean':        s.get('IC_mean', float('nan')),
            'R2_price_mean':  s.get('R2_price_mean', float('nan')),
            'R2_returns_mean': s.get('R2_ret_mean', float('nan')),
            'Sharpe_mean':    s.get('Sharpe_mean', float('nan')),
            'n_synth_mean':   s.get('n_synth_mean', float('nan')),
            'ttest_p':        s.get('ttest_p', float('nan')),
        })

if rows:
    pd.DataFrame(rows).to_csv('results/day3_wgan_smote_results.csv', index=False)
    print("\nsaved: results/day3_wgan_smote_results.csv")

print("\n" + "="*85)
print("  DAY 3 FINAL RESULTS")
print("="*85)
print(f"\n{'asset':8s} | {'method':9s} | {'baseline DA':>11} | "
      f"{'DA (mean±std)':20s} | {'delta':>8} | {'p-val':>8}")
print("-"*80)
for ticker, methods in summaries.items():
    for method_name, s in methods.items():
        bda  = s.get('baseline_DA', float('nan'))
        m    = s.get('DA_mean', float('nan'))
        sd   = s.get('DA_std', float('nan'))
        dpp  = s.get('delta_pp', float('nan'))
        pval = s.get('ttest_p', float('nan'))
        bstr = f'{bda:.4f}' if not np.isnan(bda) else '  N/A'
        dstr = f'{dpp:+.2f}pp' if not np.isnan(dpp) else '   N/A'
        pstr = f'{pval:.4f}' if not np.isnan(pval) else '  N/A'
        print(f"{ticker:8s} | {method_name:9s} | {bstr:>11} | "
              f"{m:.4f} ± {sd:.4f}       | {dstr:>8} | {pstr:>8}")

print("\nday 3 done. next: day4 statistical validation (10 trials, paired t-tests)")
