"""Convergence check of the number of RK4 steps (50 against 100).
"""

import json
import sys

import numpy as np
import torch

import config as K
import engine as M


def main(n=20):
    b = M.carica_tutto()
    fold = M.adapter_fold()
    prompt = json.load(open(K.PROMPT_SVILUPPO))[:n]
    righe = []
    for p in prompt:
        e = M.adatta(fold[p["nome"]], M.embedding_testo(b, [p["testi"]["algoritmica"]]))
        x0 = M.rumore(b.spec.dim, K.seed_di(p["indice"], 0)).unsqueeze(0)
        ref, ref_emb = M.riferimento_sviluppo(b, p["nome"])
        r = {"nome": p["nome"]}
        xs = {}
        for passi in (K.N_PASSI, K.N_PASSI_CONTROLLO):
            x = M.genera(b.vf, e, x0, 2.0, passi)[0]
            xs[passi] = x
            gen = M.C.render_preset(b.plugin, M.a_preset(b, x))
            r[f"sim_{passi}"] = float(np.dot(ref_emb, M.emb_audio(b, gen)))
        r["delta"] = abs(r[f"sim_{K.N_PASSI}"] - r[f"sim_{K.N_PASSI_CONTROLLO}"])
        r["mse_x"] = float(np.mean((xs[K.N_PASSI] - xs[K.N_PASSI_CONTROLLO]) ** 2))
        righe.append(r)
        print(f"  {p['nome'][:40]:40s} sim50={r[f'sim_{K.N_PASSI}']:.4f} "
              f"sim100={r[f'sim_{K.N_PASSI_CONTROLLO}']:.4f} Δ={r['delta']:.4f}")
    delta = float(np.mean([r["delta"] for r in righe]))
    scelta = K.N_PASSI if delta < K.SOGLIA_PASSI else K.N_PASSI_CONTROLLO
    out = {"delta_medio": delta, "mse_x_medio": float(np.mean([r["mse_x"] for r in righe])),
           "soglia": K.SOGLIA_PASSI, "n_passi": scelta, "righe": righe}
    json.dump(out, open(K.CONTROLLO_PASSI, "w"), indent=1)
    print(f"[OK] mean Δ {delta:.4f} (threshold {K.SOGLIA_PASSI}) -> steps = {scelta}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
