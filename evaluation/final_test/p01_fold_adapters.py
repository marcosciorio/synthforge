"""Train one text adapter for each of the five folds of the reference set.
"""

import json
import os

import numpy as np
import torch

import config as K
import train_text_adapter as TA


def main():
    d = np.load(K.NPZ_ADAPTER, allow_pickle=True)
    T, A = d["T"].astype(np.float32), d["A"].astype(np.float32)
    names = np.array([str(n) for n in d["names"]])
    preset = np.array(sorted(set(names)))
    np.random.default_rng(42).shuffle(preset)
    folds = np.array_split(preset, K.N_FOLD)
    fold_map = {str(n): k for k, f in enumerate(folds) for n in f}
    print(f"[..] {len(T)} pairs, {len(preset)} presets, {K.N_FOLD} folds (adapter: {K.VARIANTE_ADAPTER})")
    riepilogo = []
    for k, f in enumerate(folds):
        fuori = set(f)
        tr = np.array([n not in fuori for n in names])
        modello = TA._train_full(T[tr], A[tr], TA.MAX_EPOCHS)
        modello.eval()
        with torch.no_grad():
            Tv = torch.from_numpy(T[~tr]).to(TA.DEVICE)
            Av = torch.from_numpy(A[~tr]).to(TA.DEVICE)
            cos_fuori = (modello(Tv) * Av).sum(-1).mean().item()
        torch.save({"state_dict": {a: v.cpu() for a, v in modello.state_dict().items()},
                    "dim": TA.DIM, "fold": k, "cos_fuori": cos_fuori},
                   os.path.join(K.DIR_FOLD, f"fold{k}.pt"))
        riepilogo.append(cos_fuori)
        print(f"[OK] fold {k}: {int(tr.sum())} training pairs, held-out cosine {cos_fuori:.4f}")
    json.dump({"fold_map": fold_map, "cos_fuori": riepilogo, "variante": K.VARIANTE_ADAPTER},
              open(os.path.join(K.DIR_FOLD, "fold_map.json"), "w"), indent=1)
    print(f"[OK] mean cosine on the held-out folds: {np.mean(riepilogo):.4f} ± {np.std(riepilogo):.4f}")


if __name__ == "__main__":
    main()
