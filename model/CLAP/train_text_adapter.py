"""Train the text adapter: an affine map from CLAP text embeddings
to CLAP audio embeddings.

Loss: mean(1 - cos(W t + b, a)) + lambda * ||W - I||_F^2
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

AUG_MODE    = "--aug" in sys.argv
NPZ_PATH    = os.path.join(PROJECT_ROOT, "model", "kaggle_training_cfm",
                           "adapter_embeddings_aug.npz" if AUG_MODE else "adapter_embeddings.npz")
OUT_ADAPTER = os.path.join(PROJECT_ROOT, "model", "kaggle_training_cfm",
                           "adapter_aug.pt" if AUG_MODE else "adapter.pt")

DIM        = 512
LR         = 1e-3
LAMBDA     = 0.1
MAX_EPOCHS = 500
PATIENCE   = 30
N_FOLDS    = 5

_mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
DEVICE  = "mps" if _mps_ok else ("cuda" if torch.cuda.is_available() else "cpu")


class TextAdapter(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(t @ self.W.T + self.b, dim=-1)


def _loss(a_hat: torch.Tensor, a: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    cos_loss  = (1.0 - (a_hat * a).sum(dim=-1)).mean()
    I         = torch.eye(W.shape[0], device=W.device)
    frob_loss = ((W - I) ** 2).sum()
    return cos_loss + LAMBDA * frob_loss


def _train_one_fold(T_tr, A_tr, T_va, A_va):
    model = TextAdapter(DIM).to(DEVICE)
    opt   = AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    T_tr = torch.from_numpy(T_tr).to(DEVICE)
    A_tr = torch.from_numpy(A_tr).to(DEVICE)
    T_va = torch.from_numpy(T_va).to(DEVICE)
    A_va = torch.from_numpy(A_va).to(DEVICE)

    best_cos   = -1.0
    best_state = None
    no_improve = 0

    for _ in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        loss = _loss(model(T_tr), A_tr, model.W)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_cos = (model(T_va) * A_va).sum(dim=-1).mean().item()

        if val_cos > best_cos + 1e-6:
            best_cos   = val_cos
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            break

    return best_cos, best_state


def _train_full(T, A, epochs: int) -> TextAdapter:
    model = TextAdapter(DIM).to(DEVICE)
    opt   = AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    T_t = torch.from_numpy(T).to(DEVICE)
    A_t = torch.from_numpy(A).to(DEVICE)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = _loss(model(T_t), A_t, model.W)
        loss.backward()
        opt.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                cos = (model(T_t) * A_t).sum(dim=-1).mean().item()
            print(f"  epoch {epoch+1:>4}/{epochs}  "
                  f"loss={loss.item():.6f}  train_cos={cos:.4f}")

    return model


def main():
    if not os.path.isfile(NPZ_PATH):
        print(f"File not found: {NPZ_PATH}")
        if AUG_MODE:
            print("Run first: python CLAP/build_adapter_dataset_aug.py")
        else:
            print("Run first: python CLAP/build_adapter_dataset_aug.py")
        return

    data = np.load(NPZ_PATH, allow_pickle=True)
    T    = data["T"].astype(np.float32)
    A    = data["A"].astype(np.float32)
    N    = len(T)

    mode_str = "AUGMENTED" if AUG_MODE else "ORIGINALE"
    print(f"Dataset  : {N} samples  dim={T.shape[1]}  [{mode_str}]")
    print(f"Device   : {DEVICE}")
    print(f"λ={LAMBDA}  lr={LR}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}\n")

    baseline = (T * A).sum(axis=1).mean()
    print(f"Baseline cosine (W=I, b=0) : {baseline:.4f}\n")

    print(f"=== {N_FOLDS}-FOLD CROSS-VALIDATION ===")
    rng = np.random.default_rng(42)
    val_scores = []

    if AUG_MODE and "names" in data:
        names = data["names"]
        unique_presets = np.array(sorted(set(names)))
        rng.shuffle(unique_presets)
        preset_folds = np.array_split(unique_presets, N_FOLDS)

        for k, val_presets in enumerate(preset_folds):
            val_set = set(val_presets)
            tr_idx  = np.where([n not in val_set for n in names])[0]
            va_idx  = np.where([n in val_set     for n in names])[0]
            T_tr, A_tr = T[tr_idx], A[tr_idx]
            T_va, A_va = T[va_idx], A[va_idx]

            t0 = time.time()
            best_cos, _ = _train_one_fold(T_tr, A_tr, T_va, A_va)
            elapsed = time.time() - t0

            val_scores.append(best_cos)
            print(f"  Fold {k+1}/{N_FOLDS}  "
                  f"tr={len(tr_idx)}  va={len(va_idx)}  "
                  f"val_cos={best_cos:.4f}  ({elapsed:.1f}s)")
    else:
        idx   = rng.permutation(N)
        folds = np.array_split(idx, N_FOLDS)

        for k, val_idx in enumerate(folds):
            tr_idx = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
            T_tr, A_tr = T[tr_idx], A[tr_idx]
            T_va, A_va = T[val_idx], A[val_idx]

            t0 = time.time()
            best_cos, _ = _train_one_fold(T_tr, A_tr, T_va, A_va)
            elapsed = time.time() - t0

            val_scores.append(best_cos)
            print(f"  Fold {k+1}/{N_FOLDS}  val_cos={best_cos:.4f}  ({elapsed:.1f}s)")

    mean_cos = float(np.mean(val_scores))
    std_cos  = float(np.std(val_scores))
    delta    = mean_cos - float(baseline)
    print(f"\nCV result  : {mean_cos:.4f} ± {std_cos:.4f}")
    print(f"Baseline   : {baseline:.4f}")
    print(f"Δ          : {delta:+.4f}")
    if delta < 0.01:
        print("  [WARNING] Marginal improvement: "
              "the gap may already be small.")

    print(f"\n=== Final training (all {N} samples, {MAX_EPOCHS} epochs) ===")
    final_model = _train_full(T, A, MAX_EPOCHS)

    final_model.eval()
    with torch.no_grad():
        T_t = torch.from_numpy(T).to(DEVICE)
        A_t = torch.from_numpy(A).to(DEVICE)
        final_cos = (final_model(T_t) * A_t).sum(dim=-1).mean().item()
    print(f"\nFinal training cosine : {final_cos:.4f}")

    torch.save({
        "state_dict": {k: v.cpu() for k, v in final_model.state_dict().items()},
        "dim"       : DIM,
        "cv_mean"   : mean_cos,
        "cv_std"    : std_cos,
        "baseline"  : float(baseline),
        "lambda"    : LAMBDA,
        "lr"        : LR,
        "augmented" : AUG_MODE,
    }, OUT_ADAPTER)
    print(f"Adapter saved : {OUT_ADAPTER}")
    print(f"\nCV: {mean_cos:.4f} ± {std_cos:.4f}  "
          f"(baseline {baseline:.4f}, Δ={delta:+.4f})")
    if AUG_MODE:
        print("\nTo use this adapter, set:")
        print("  ADAPTER_PATH = '.../kaggle_training_cfm/adapter_aug.pt'")


if __name__ == "__main__":
    main()
