"""
LSTM model and training loop.

ClassicalLSTM: two-layer LSTM, hidden=64, dropout=0.2
  - capacity-matched to QLSTM for fair ablation comparison
  - trained with mini-batch SGD + ReduceLROnPlateau

Key difference from Part 1:
  Part 1 used a random shuffle in the DataLoader on time series data.
  This notebook keeps things chronological throughout.

Vivian Chan | 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
EPOCHS      = 100
PATIENCE    = 15
BATCH_SIZE  = 32
LR          = 0.001


class ClassicalLSTM(nn.Module):
    """
    Standard two-layer LSTM for stock price prediction.
    Input: (batch, seq_length, n_features)
    Output: (batch, 1) — next-day Close price (scaled)
    """
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_model(model, X_tr, y_tr, X_vl, y_vl,
                epochs=EPOCHS, patience=PATIENCE,
                lr=LR, verbose=True):
    """
    Mini-batch training with:
      - ReduceLROnPlateau scheduler (halves LR when val loss stalls)
      - L2 regularization via weight_decay
      - Gradient clipping to avoid exploding gradients
      - Best-weight restoration at end

    Returns the model loaded with the best validation weights.
    """
    optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, mode='min', factor=0.5,
                     patience=5, min_lr=1e-5)
    criterion  = nn.MSELoss()
    best_val   = float('inf')
    no_improve = 0
    best_state = None  # initialize explicitly — avoids UnboundLocalError

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    for epoch in range(1, epochs + 1):
        # training
        model.train()
        ep_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by.unsqueeze(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(X_vl), y_vl.unsqueeze(-1)
            ).item()

        scheduler.step(val_loss)

        if verbose and epoch % 10 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    ep {epoch:3d}/{epochs} | "
                  f"train={ep_loss/len(loader):.5f} | "
                  f"val={val_loss:.5f} | lr={lr_now:.6f}")

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            # clone each tensor so we don't alias the live model state
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    early stop @ epoch {epoch} "
                          f"(best val={best_val:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        if verbose:
            print("    WARNING: model never improved — check your data")

    return model
