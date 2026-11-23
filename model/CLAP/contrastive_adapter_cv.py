"""Contrastive adapter, phase 1: five-fold cross-validation of adapter
variants that add an InfoNCE term to the adapter loss.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

QUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(QUI))
NPZ = os.path.join(ROOT, "model", "kaggle_training_cfm", "adapter_embeddings_aug_space.npz")
OUT = os.path.join(QUI, "contrastive_adapter_cv.json")
LR, LAMBDA, MAX_EP, PAT, NF = 1e-3, 0.1, 500, 30, 5
torch.manual_seed(0)

VARIANTI = {
    "attuale": dict(alpha_cos=1.0, alpha_nce=0.0, tau=None),
    "misto_t007": dict(alpha_cos=1.0, alpha_nce=1.0, tau=0.07),
    "misto_t003": dict(alpha_cos=1.0, alpha_nce=1.0, tau=0.03),
    "contrastivo": dict(alpha_cos=0.0, alpha_nce=1.0, tau=0.07),
}


class Adapter(torch.nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.W = torch.nn.Parameter(torch.eye(d))
        self.b = torch.nn.Parameter(torch.zeros(d))

    def forward(self, t):
        return F.normalize(t @ self.W.T + self.b, dim=-1)


def loss_fn(m, T, A, idx, P, cfg):
    z = m(T)
    l = LAMBDA * ((m.W - torch.eye(m.W.shape[0])) ** 2).sum()
    if cfg["alpha_cos"]:
        l = l + cfg["alpha_cos"] * (1 - (z * A).sum(-1)).mean()
    if cfg["alpha_nce"]:
        l = l + cfg["alpha_nce"] * F.cross_entropy(z @ P.T / cfg["tau"], idx)
    return l


def metriche(z, A, pos, P_tutti):
    cos = float((z * A).sum(-1).mean())
    S = z @ P_tutti.T
    giusto = S[np.arange(len(z)), pos]
    rango = 1 + (S > giusto[:, None]).sum(1)
    zz = z @ z.T
    disp = float(zz[~np.eye(len(z), dtype=bool)].mean())
    return {"cos": cos, "rango_mediano": float(np.median(rango)), "R@1": float((rango <= 1).mean()),
            "R@5": float((rango <= 5).mean()), "R@10": float((rango <= 10).mean()), "dispersione": disp}


def main():
    d = np.load(NPZ, allow_pickle=True)
    T, A, names = d["T"].astype(np.float32), d["A"].astype(np.float32), d["names"]
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    preset = np.array(sorted(set(names)))
    pid = {p: i for i, p in enumerate(preset)}
    P_tutti = np.stack([A[np.where(names == p)[0][0]] for p in preset])
    pos = np.array([pid[n] for n in names])
    rng = np.random.default_rng(42)
    perm = preset.copy(); rng.shuffle(perm)
    folds = np.array_split(perm, NF)

    ris = {"grezzo": [], **{k: [] for k in VARIANTI}}
    epoche = {k: [] for k in VARIANTI}
    t0 = time.time()
    for k, val in enumerate(folds):
        vs = set(val)
        va = np.array([n in vs for n in names]); tr = ~va
        ris["grezzo"].append(metriche(T[va], A[va], pos[va], P_tutti))
        tr_pres = np.array(sorted(set(names[tr])))
        tr_id = {p: i for i, p in enumerate(tr_pres)}
        P_tr = torch.from_numpy(np.stack([P_tutti[pid[p]] for p in tr_pres]))
        va_pres = np.array(sorted(set(names[va])))
        va_id = {p: i for i, p in enumerate(va_pres)}
        P_va = torch.from_numpy(np.stack([P_tutti[pid[p]] for p in va_pres]))
        Ttr, Atr = torch.from_numpy(T[tr]), torch.from_numpy(A[tr])
        Tva, Ava = torch.from_numpy(T[va]), torch.from_numpy(A[va])
        itr = torch.tensor([tr_id[n] for n in names[tr]])
        iva = torch.tensor([va_id[n] for n in names[va]])
        for nome, cfg in VARIANTI.items():
            m = Adapter(); opt = AdamW(m.parameters(), lr=LR, weight_decay=0.0)
            best, stato, ep_best, noimp = None, None, 0, 0
            for ep in range(MAX_EP):
                m.train(); opt.zero_grad()
                loss_fn(m, Ttr, Atr, itr, P_tr, cfg).backward(); opt.step()
                m.eval()
                with torch.no_grad():
                    if nome == "attuale":
                        crit = -float((m(Tva) * Ava).sum(-1).mean())
                    else:
                        crit = float(loss_fn(m, Tva, Ava, iva, P_va, cfg))
                if best is None or crit < best - 1e-6:
                    best, stato, ep_best, noimp = crit, {kk: v.clone() for kk, v in m.state_dict().items()}, ep + 1, 0
                else:
                    noimp += 1
                    if noimp >= PAT:
                        break
            m.load_state_dict(stato); m.eval()
            with torch.no_grad():
                z = m(Tva).numpy()
            r = metriche(z, A[va], pos[va], P_tutti); r["norma_b"] = float(m.b.norm()); r["epoca"] = ep_best
            ris[nome].append(r); epoche[nome].append(ep_best)
        print(f"  fold {k + 1}/{NF} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    tab = {}
    for nome, lst in ris.items():
        tab[nome] = {m_: (float(np.mean([x[m_] for x in lst])), float(np.std([x[m_] for x in lst])))
                     for m_ in lst[0]}
    candidati = [n for n in VARIANTI if tab[n]["cos"][0] >= 0.55]
    scelta = min(candidati, key=lambda n: (tab[n]["rango_mediano"][0], -tab[n]["R@10"][0])) if candidati else None
    json.dump({"tabella": tab, "fold": ris, "scelta": scelta, "regola": "cos >= 0,55, poi rango mediano minimo, poi R@10",
               "epoche_migliori": epoche}, open(OUT, "w"), indent=1)
    print(f"\n{'variant':12s} {'cos':>13s} {'median rank':>12s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'spread':>9s} {'|b|':>6s}")
    for nome, t in tab.items():
        print(f"{nome:12s} {t['cos'][0]:.3f}±{t['cos'][1]:.3f} {t['rango_mediano'][0]:8.1f}±{t['rango_mediano'][1]:<4.0f} "
              f"{t['R@1'][0]:6.3f} {t['R@5'][0]:6.3f} {t['R@10'][0]:6.3f} {t['dispersione'][0]:9.3f} "
              f"{t.get('norma_b', (0, 0))[0]:6.2f}")
    print(f"\nselected: {scelta}   (results in {OUT})")


if __name__ == "__main__":
    main()
