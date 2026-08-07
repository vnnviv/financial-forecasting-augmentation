# day6_qlstm_ablation.py
#
# Part 2, Day 6 — quantum ablation study
# Vivian Chan | Glen A. Wilson High School | 2026
# Mentors: Mohammad Husain & Antoine Si | Cal Poly Pomona
#
# What this notebook does:
#   - Isolates the quantum contribution: a capacity-matched classical
#     LSTM (hidden=32, back down from the hidden=64 used everywhere else
#     in this project -- 32 is what the original QLSTM ran at, so this
#     is the fair-fight size, not the main pipeline's size) vs a QLSTM
#     where each of the 4 gates (forget, input, update, output) runs
#     through a small variational quantum circuit instead of a plain
#     Linear layer. Same data, same training loop, same hyperparameters
#     otherwise -- only the gate computation differs.
#   - Uses qml.qnn.TorchLayer for the quantum gates, which handles the
#     tensor conversion between PennyLane and PyTorch internally. That's
#     what avoids the "use sourceTensor.clone().detach()" UserWarning my
#     original hand-rolled circuit code used to throw -- TorchLayer never
#     does the naive torch.tensor(already_a_tensor) thing that caused it.
#   - AAPL only. QLSTM is slow (a forward pass runs 4 quantum circuits x
#     20 timesteps per sample, and PennyLane's simulator isn't fast), and
#     the plan explicitly says to scope this to one asset and say so, not
#     pretend a 4-asset quantum study happened in one Kaggle session.
#
# Expected finding, going in: QLSTM performs badly -- close to what Part
# 1 originally reported (R² ~ -27.7, DA ~ 51.8%). If that holds up, it's
# not a failure of this notebook, it's the point: a quantum circuit
# squeezed through a 4-qubit bottleneck has less representational
# capacity than a plain Linear layer at the same hidden size, and
# capacity is exactly what this ablation is testing for.
#
# Run this on Kaggle with GPU T4 (PennyLane's default.qubit simulator
# runs on CPU regardless -- the GPU only helps the classical side).
# Expected time: 4-5 hours, almost all of it QLSTM.


# ── imports ───────────────────────────────────────────────────────────────────

# pennylane isn't on Kaggle's default image (unlike yfinance, which
# usually is). subprocess instead of a "!pip install" cell-magic line so
# this still runs whether it's pasted into a notebook cell or executed
# as a plain script.
import subprocess
subprocess.run(['pip', 'install', 'pennylane', '-q'], check=True)

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
import pennylane as qml
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
print(f"device: {device} (quantum circuits run on CPU regardless)")


# ── config ────────────────────────────────────────────────────────────────────

# capacity-matched hidden size -- NOT the hidden=64 used in Day 1-5.
# This is what makes it a fair ablation: same hidden size on both sides.
HIDDEN_SIZE = 32
SEQ_LENGTH  = 20
EPOCHS      = 30    # cut from 100 -- QLSTM epochs are much more expensive than classical
PATIENCE    = 8
BATCH_SIZE  = 16
LR          = 0.001
NUM_TRIALS  = 3      # cut from 5 -- see above

# quantum circuit
N_QUBITS  = 4
N_QLAYERS = 1

# data -- AAPL only, see header note
ALL_ASSETS  = ['AAPL']
TRAIN_START = '2020-01-01'
OOS_END     = '2023-12-31'

FEATURE_COLS = ['Close', 'RSI_14', 'SMA_5', 'SMA_10', 'SMA_20',
                'Vol_20', 'Mom_5', 'BB_pos']
N_FEATURES   = len(FEATURE_COLS)

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
print("config loaded")


# ── data (identical to Day 1-5) ───────────────────────────────────────────────

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


def to_t(arr, dev=None):
    return torch.tensor(np.asarray(arr), dtype=torch.float32).to(dev or device)


# PennyLane's default.qubit simulator only runs on CPU -- it has no CUDA
# path. Everything in ClassicalLSTM stays on `device` (cuda if available,
# for speed); QLSTM has to run entirely on CPU instead, or its classical
# pre/post Linear layers end up on cuda while the quantum layer's
# internal state stays on cpu, and PyTorch throws "Expected all tensors
# to be on the same device". This only shows up with an actual GPU
# present -- a CPU-only environment never surfaces the mismatch, which
# is exactly the failure mode this comment exists to prevent recurring.
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
    def __init__(self, in_features, hidden_size=HIDDEN_SIZE):
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

class ClassicalLSTM(nn.Module):
    """Standard 2-layer LSTM, hidden=32 -- the capacity-matched baseline
    for this ablation specifically (not the hidden=64 used in Day 1-5)."""
    def __init__(self, input_size=N_FEATURES, hidden_size=HIDDEN_SIZE):
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
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE):
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
    def __init__(self, input_size=N_FEATURES, hidden_size=HIDDEN_SIZE):
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

def train(model, X_tr, y_tr, X_vl, y_vl, verbose=True):
    opt  = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
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

        if verbose and ep % 5 == 0:
            print(f"    ep {ep:3d}/{EPOCHS} | train={ep_loss / len(loader):.5f} "
                  f"| val={val_loss:.5f}")

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


# ── metrics (identical to Day 1-5) ────────────────────────────────────────────

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


def evaluate(y_true, y_pred, label=''):
    yt = y_true.flatten(); yp = y_pred.flatten()
    m = {
        'MSE':      round(float(mean_squared_error(yt, yp)), 4),
        'RMSE':     round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        'R2_price': round(float(r2_score(yt, yp)), 4),
        'R2_ret':   return_r2(y_true, y_pred),
        'IC':       ic(y_true, y_pred),
        'DA':       da(y_true, y_pred),
    }
    if label:
        print(f"\n  [{label}]")
        for k, v in m.items():
            print(f"    {k:12s}: {v}")
    return m


# ── per-model trial pipeline ──────────────────────────────────────────────────

def run_trials(model_name, model_cls, X_tr, y_tr, X_vl_t, y_vl_t, X_te_t, cs,
               y_true_prices, n=NUM_TRIALS, model_device=None):
    model_device = model_device or device
    trials = []
    for i in range(n):
        seed = SEED + i * 13
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = model_cls().to(model_device)
        model = train(model, to_t(X_tr, model_device), to_t(y_tr, model_device),
                      X_vl_t, y_vl_t, verbose=(i == 0))

        model.eval()
        with torch.no_grad():
            pred_s = model(X_te_t).cpu().numpy()
        pred_prices = cs.inverse_transform(pred_s[:, :1])

        metrics = evaluate(y_true_prices, pred_prices,
                          label=f'{model_name} OOS' if i == 0 else '')
        print(f"  [{model_name}] trial {i+1}: DA={metrics['DA']:.4f} | R2_price={metrics['R2_price']:.4f}")
        trials.append(metrics)

    keys = ['MSE', 'RMSE', 'R2_price', 'R2_ret', 'IC', 'DA']
    out = {'n': len(trials)}
    for k in keys:
        vals = [float(t[k]) for t in trials if t.get(k) is not None and not np.isnan(float(t.get(k, float('nan'))))]
        if vals:
            out[f'{k}_mean'] = round(np.mean(vals), 4)
            out[f'{k}_std']  = round(np.std(vals), 4)
    return out


def run_asset(ticker):
    print(f"\n{'='*65}\n  {ticker} — classical vs quantum LSTM ({NUM_TRIALS} trials each)\n{'='*65}")

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

    # classical stays on `device` (cuda if available); QLSTM needs its
    # own cpu-only copies of val/test, see the QDEVICE note above to_t()
    X_vl_t, y_vl_t = to_t(X_vl), to_t(y_vl)
    X_te_t = to_t(X_te)
    X_vl_t_cpu, y_vl_t_cpu = to_t(X_vl, QDEVICE), to_t(y_vl, QDEVICE)
    X_te_t_cpu = to_t(X_te, QDEVICE)

    print("\n  -- classical (hidden=32) --")
    classical = run_trials('ClassicalLSTM', ClassicalLSTM, X_tr, y_tr,
                          X_vl_t, y_vl_t, X_te_t, cs, y_true_prices,
                          model_device=device)

    print("\n  -- quantum (4 qubits, hidden=32) --")
    quantum = run_trials('QLSTM', QLSTM, X_tr, y_tr,
                        X_vl_t_cpu, y_vl_t_cpu, X_te_t_cpu, cs, y_true_prices,
                        model_device=QDEVICE)

    return {'classical': classical, 'quantum': quantum}


# ── figure ────────────────────────────────────────────────────────────────────

def fig7_classical_vs_qlstm(results):
    tickers = list(results.keys())
    c_da = [results[t]['classical'].get('DA_mean', float('nan')) for t in tickers]
    q_da = [results[t]['quantum'].get('DA_mean', float('nan')) for t in tickers]
    c_r2 = [results[t]['classical'].get('R2_price_mean', float('nan')) for t in tickers]
    q_r2 = [results[t]['quantum'].get('R2_price_mean', float('nan')) for t in tickers]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('#FAFAFA')
    x = np.arange(len(tickers)); w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, c_da, w, color='#8B5A8D', alpha=0.9, label='Classical LSTM', edgecolor='white')
    ax.bar(x + w/2, q_da, w, color='#E74C3C', alpha=0.9, label='QLSTM', edgecolor='white')
    ax.axhline(0.50, color='black', ls='--', lw=1.5, label='random (50%)', alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel('directional accuracy', fontsize=11)
    ax.set_title('DA: classical vs quantum', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.set_facecolor('#FFFAFD'); ax.grid(axis='y', alpha=0.3, linestyle='--')

    ax = axes[1]
    ax.bar(x - w/2, c_r2, w, color='#8B5A8D', alpha=0.9, label='Classical LSTM', edgecolor='white')
    ax.bar(x + w/2, q_r2, w, color='#E74C3C', alpha=0.9, label='QLSTM', edgecolor='white')
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel('R² (price)', fontsize=11)
    ax.set_title('R²: classical vs quantum\n(QLSTM often goes deeply negative)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.set_facecolor('#FFFAFD'); ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('Figure 7 — classical LSTM vs QLSTM, capacity-matched (hidden=32)',
                fontsize=13, fontweight='bold', y=1.02, color='#5A3A5C')
    plt.tight_layout()
    plt.savefig('figures/fig7_lstm_vs_qlstm.png', dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
    plt.show()
    print("saved: figures/fig7_lstm_vs_qlstm.png")


# ── main ──────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  DAY 6 — QUANTUM ABLATION STUDY")
print("="*70)

results = {}
for ticker in ALL_ASSETS:
    r = run_asset(ticker)
    if r:
        results[ticker] = r
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\ncompleted: {len(results)}/{len(ALL_ASSETS)} assets")

if results:
    fig7_classical_vs_qlstm(results)

rows = []
for ticker, r in results.items():
    c, q = r['classical'], r['quantum']
    rows.append({
        'Asset': ticker,
        'Classical_DA_mean': c.get('DA_mean', float('nan')),
        'Classical_DA_std':  c.get('DA_std', float('nan')),
        'Classical_R2_price': c.get('R2_price_mean', float('nan')),
        'Classical_IC': c.get('IC_mean', float('nan')),
        'QLSTM_DA_mean': q.get('DA_mean', float('nan')),
        'QLSTM_DA_std':  q.get('DA_std', float('nan')),
        'QLSTM_R2_price': q.get('R2_price_mean', float('nan')),
        'QLSTM_IC': q.get('IC_mean', float('nan')),
        'DA_gap_pp': round((c.get('DA_mean', float('nan')) - q.get('DA_mean', float('nan'))) * 100, 2),
    })
if rows:
    pd.DataFrame(rows).to_csv('results/day6_ablation_results.csv', index=False)
    print("saved: results/day6_ablation_results.csv")

print("\n" + "="*80)
print("  DAY 6 FINAL RESULTS")
print("="*80)
for row in rows:
    print(f"\n{row['Asset']}:")
    print(f"  classical: DA={row['Classical_DA_mean']:.4f}±{row['Classical_DA_std']:.4f} | "
          f"R2={row['Classical_R2_price']:.4f} | IC={row['Classical_IC']:.4f}")
    print(f"  QLSTM:     DA={row['QLSTM_DA_mean']:.4f}±{row['QLSTM_DA_std']:.4f} | "
          f"R2={row['QLSTM_R2_price']:.4f} | IC={row['QLSTM_IC']:.4f}")
    print(f"  gap: classical beats QLSTM by {row['DA_gap_pp']:+.2f}pp DA")

print("\nday 6 done. next: day7 review + buffer -- check everything, build the master spreadsheet, rest")
