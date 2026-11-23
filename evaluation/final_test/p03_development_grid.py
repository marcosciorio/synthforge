"""Hyperparameter grid on the development set: guidance scale for each
method, restart time and search type for the retrieval method.
"""

import argparse
import json
import time

import numpy as np
import torch

import config as K
import engine as M

CAMPI = ("nome", "testo", "metodo", "s", "t_star")
RICERCHE = ("", "_grezzo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    passi = json.load(open(K.CONTROLLO_PASSI))["n_passi"]
    b = M.carica_tutto()
    fold = M.adapter_fold()
    prompt = json.load(open(K.PROMPT_SVILUPPO))[: a.limit]
    fatte = M.chiavi_fatte(K.SVILUPPO_JSONL, CAMPI)
    S = K.GRIGLIA_S
    t0 = time.time()
    for i, p in enumerate(prompt):
        nome = p["nome"]
        ref = None
        for testo, frase in p["testi"].items():
            celle = [("diretta", s, 0.0) for s in S] + [("adapter", s, 0.0) for s in S]
            for r in RICERCHE:
                celle += [(f"retrieval{r}", s, ts) for ts in K.GRIGLIA_TSTAR for s in S]
                celle += [(f"solo_retrieval{r}", None, 1.0)]
            mancanti = [c for c in celle if (nome, testo, c[0], c[1], c[2]) not in fatte]
            if not mancanti:
                continue
            if ref is None:
                ref, ref_emb = M.riferimento_sviluppo(b, nome)
            e_raw = M.embedding_testo(b, [frase])
            e_ad = M.adatta(fold[nome], e_raw)
            cat = M.categoria_dal_testo(frase)
            rec = {"": M.recupera(b, e_ad.cpu().numpy()[0], cat, escludi=nome),
                   "_grezzo": M.recupera(b, e_raw.cpu().numpy()[0], cat, escludi=nome)}
            x0 = M.rumore(b.spec.dim, K.seed_di(p["indice"], 0)).unsqueeze(0)

            def ricerca_di(m):
                return "_grezzo" if m.endswith("_grezzo") else ""

            gruppi = {}
            for m, s, ts in mancanti:
                if m.startswith("solo_"):
                    continue
                if m == "retrieval_grezzo" and rec["_grezzo"] == rec[""]:
                    continue
                gruppi.setdefault((m, ts), []).append(s)
            uscite = {}
            for (m, ts), scale in gruppi.items():
                cond = e_raw if m == "diretta" else e_ad
                B = len(scale)
                xr = None
                if m.startswith("retrieval"):
                    x_ret_np = M.C.encode_preset(b.spec, b.pool_params[rec[ricerca_di(m)]])
                    xr = torch.from_numpy(x_ret_np).unsqueeze(0).expand(B, -1)
                xs = M.genera(b.vf, cond, x0.expand(B, -1), scale, passi, t_star=ts, x_ret=xr)
                for s, x in zip(scale, xs):
                    uscite[(m, s, ts)] = x

            misurate = {}
            for c in mancanti:
                m, s, ts = c
                r = ricerca_di(m)
                stesso = r == "_grezzo" and rec["_grezzo"] == rec[""]
                chiave_base = (m.replace("_grezzo", ""), s, ts) if stesso else c
                if chiave_base not in misurate:
                    if m.startswith("solo_"):
                        preset = b.pool_params[rec[r]]
                    else:
                        x = uscite.get(chiave_base)
                        if x is None:
                            x = M.genera(b.vf, e_ad, x0, chiave_base[1], passi, t_star=ts,
                                         x_ret=torch.from_numpy(M.C.encode_preset(b.spec, b.pool_params[rec[""]])).unsqueeze(0))[0]
                        preset = M.a_preset(b, x)
                    gen = M.C.render_preset(b.plugin, preset)
                    misurate[chiave_base], _ = M.metriche_rapide(b, ref, ref_emb, gen)
                M.aggiungi(K.SVILUPPO_JSONL, {"nome": nome, "testo": testo, "metodo": m, "s": s,
                                              "t_star": ts, "categoria_testo": cat,
                                              "recuperato": rec[r] if m.startswith(("retrieval", "solo_")) else None,
                                              "stesso_preset": rec["_grezzo"] == rec[""],
                                              "passi": passi, **misurate[chiave_base]})
        dt = time.time() - t0
        print(f"  [{i+1}/{len(prompt)}] {nome[:40]:40s} ({dt/60:.1f} min, "
              f"~{dt/(i+1)*(len(prompt)-i-1)/60:.0f} min remaining)", flush=True)
    print(f"[OK] development grid -> {K.SVILUPPO_JSONL}")


if __name__ == "__main__":
    main()
