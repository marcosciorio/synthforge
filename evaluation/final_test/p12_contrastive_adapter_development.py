"""Contrastive adapter, phase 2: generation on the development set with the
fold adapters trained with the contrastive term, compared with the standard adapter.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from torch.optim import AdamW

import config as K
import engine as M
from config import C

PESO_NCE, TAU, LAMBDA, LR, EPOCHE = 0.5, 0.07, 0.1, 1e-3, 500
DIR_FOLD_NEW = os.path.join(K.RISULTATI, "contrastive_fold_adapters")
OUT_JSONL = os.path.join(K.RISULTATI, "development_contrastive.jsonl")
OUT_MD = os.path.join(K.DIR_ANALISI, "contrastive_adapter_development.md")
CELLE = [("adapter", 2.0, 0.0), ("retrieval", 3.0, 0.5), ("solo_retrieval", None, 1.0)]
CAMPI = ("nome", "testo", "metodo", "s", "t_star")


def addestra_fold():
    os.makedirs(DIR_FOLD_NEW, exist_ok=True)
    fm_path = os.path.join(DIR_FOLD_NEW, "fold_map.json")
    if os.path.isfile(fm_path):
        print("[OK] contrastive fold adapters already present")
        return
    d = np.load(K.NPZ_ADAPTER, allow_pickle=True)
    T = d["T"].astype(np.float32); A = d["A"].astype(np.float32)
    T /= np.linalg.norm(T, axis=1, keepdims=True); A /= np.linalg.norm(A, axis=1, keepdims=True)
    names = np.array([str(n) for n in d["names"]])
    preset = np.array(sorted(set(names)))
    np.random.default_rng(42).shuffle(preset)
    folds = np.array_split(preset, K.N_FOLD)
    fold_map = {str(n): k for k, f in enumerate(folds) for n in f}
    vecchia = json.load(open(os.path.join(K.DIR_FOLD, "fold_map.json")))["fold_map"]
    assert vecchia == fold_map, "folds differ from p01"
    torch.manual_seed(0)
    cos_fuori = []
    for k, f in enumerate(folds):
        fuori = set(f)
        tr = np.array([n not in fuori for n in names])
        tr_pres = sorted(set(names[tr])); tid = {p: i for i, p in enumerate(tr_pres)}
        P = torch.from_numpy(np.stack([A[np.where(names == p)[0][0]] for p in tr_pres]))
        Tt, At = torch.from_numpy(T[tr]), torch.from_numpy(A[tr])
        idx = torch.tensor([tid[n] for n in names[tr]])
        m = C.TextAdapter(512); opt = AdamW(m.parameters(), lr=LR, weight_decay=0.0)
        I = torch.eye(512)
        for _ in range(EPOCHE):
            opt.zero_grad()
            z = m(Tt)
            loss = (1 - (z * At).sum(-1)).mean() + LAMBDA * ((m.W - I) ** 2).sum() \
                + PESO_NCE * F.cross_entropy(z @ P.T / TAU, idx)
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            c = float((m(torch.from_numpy(T[~tr])) * torch.from_numpy(A[~tr])).sum(-1).mean())
        cos_fuori.append(c)
        torch.save({"state_dict": {a: v.cpu() for a, v in m.state_dict().items()}, "dim": 512, "fold": k,
                    "cos_fuori": c, "perdita": f"(1-cos) + {PESO_NCE}*InfoNCE(tau={TAU})"},
                   os.path.join(DIR_FOLD_NEW, f"fold{k}.pt"))
        print(f"[OK] fold {k}: held-out cosine {c:.4f}", flush=True)
    json.dump({"fold_map": fold_map, "cos_fuori": cos_fuori, "variante": "contrastivo_p05"},
              open(fm_path, "w"), indent=1)


def carica_fold_nuovi():
    m = json.load(open(os.path.join(DIR_FOLD_NEW, "fold_map.json")))
    mod = {k: M.carica_adapter(os.path.join(DIR_FOLD_NEW, f"fold{k}.pt")) for k in range(K.N_FOLD)}
    return {n: mod[k] for n, k in m["fold_map"].items()}


def genera_sviluppo(limite):
    passi = json.load(open(K.CONTROLLO_PASSI))["n_passi"]
    b = M.carica_tutto()
    fold = carica_fold_nuovi()
    prompt = json.load(open(K.PROMPT_SVILUPPO))[:limite]
    fatte = M.chiavi_fatte(OUT_JSONL, CAMPI)
    t0 = time.time()
    for i, p in enumerate(prompt):
        nome = p["nome"]; ref = None
        for testo, frase in p["testi"].items():
            mancanti = [c for c in CELLE if (nome, testo, *c) not in fatte]
            if not mancanti:
                continue
            if ref is None:
                ref, ref_emb = M.riferimento_sviluppo(b, nome)
            e_ad = M.adatta(fold[nome], M.embedding_testo(b, [frase]))
            cat = M.categoria_dal_testo(frase)
            rec = M.recupera(b, e_ad.cpu().numpy()[0], cat, escludi=nome)
            x0 = M.rumore(b.spec.dim, K.seed_di(p["indice"], 0)).unsqueeze(0)
            for m, s, ts in mancanti:
                if m == "solo_retrieval":
                    preset = b.pool_params[rec]
                else:
                    xr = None
                    if m == "retrieval":
                        xr = torch.from_numpy(M.C.encode_preset(b.spec, b.pool_params[rec])).unsqueeze(0)
                    x = M.genera(b.vf, e_ad, x0, [s], passi, t_star=ts, x_ret=xr)[0]
                    preset = M.a_preset(b, x)
                gen = M.C.render_preset(b.plugin, preset)
                met, _ = M.metriche_rapide(b, ref, ref_emb, gen)
                M.aggiungi(OUT_JSONL, {"nome": nome, "testo": testo, "metodo": m, "s": s, "t_star": ts,
                                       "categoria_testo": cat, "recuperato": rec if m != "adapter" else None,
                                       "passi": passi, **met})
        dt = time.time() - t0
        print(f"  [{i+1}/{len(prompt)}] {nome[:40]:40s} ({dt/60:.1f} min, "
              f"~{dt/(i+1)*(len(prompt)-i-1)/60:.0f} min remaining)", flush=True)


def leggi(percorso):
    return [json.loads(r) for r in open(percorso)]


def confronto():
    nuovo = {(r["nome"], r["testo"], r["metodo"]): r for r in leggi(OUT_JSONL)}
    vecchio = {}
    for r in leggi(K.SVILUPPO_JSONL):
        for m, s, ts in CELLE:
            if r["metodo"] == m and r["s"] == s and abs(r["t_star"] - ts) < 1e-9:
                vecchio[(r["nome"], r["testo"], m)] = r
    righe, pv = [], []
    for m, _, _ in CELLE:
        chiavi = [k for k in nuovo if k[2] == m and k in vecchio]
        nomi = sorted({k[0] for k in chiavi})
        out = {"metodo": m, "n_preset": len(nomi)}
        for met in ("sim_audio", "MSS"):
            a = np.array([np.mean([vecchio[k][met] for k in chiavi if k[0] == n]) for n in nomi])
            b_ = np.array([np.mean([nuovo[k][met] for k in chiavi if k[0] == n]) for n in nomi])
            d = b_ - a
            p = float(wilcoxon(b_, a).pvalue) if np.any(d != 0) else 1.0
            out[met] = {"attuale": float(a.mean()), "nuovo": float(b_.mean()), "diff": float(d.mean()),
                        "quota_nuovo_meglio": float(np.mean(d > 0) if met == "sim_audio" else np.mean(d < 0)), "p": p}
        if m != "adapter":
            uguali = [nuovo[k]["recuperato"] == vecchio[k]["recuperato"] for k in chiavi]
            out["stesso_preset_recuperato"] = float(np.mean(uguali))
        righe.append(out); pv.append(out["sim_audio"]["p"])
    ordine = np.argsort(pv); corr = [0.0] * len(pv); prec = 0.0
    for j, i in enumerate(ordine):
        prec = max(prec, min(1.0, (len(pv) - j) * pv[i])); corr[i] = prec
    for r, c in zip(righe, corr):
        r["sim_audio"]["p_holm"] = c
    migliora = any(r["sim_audio"]["p_holm"] < 0.05 and r["sim_audio"]["diff"] > 0 for r in righe)
    peggiora = any(r["sim_audio"]["p_holm"] < 0.05 and r["sim_audio"]["diff"] < 0 for r in righe)
    esito = "proceeds to phase 3" if migliora and not peggiora else "does not proceed to phase 3"
    L = ["# Contrastive adapter on the development set (phase 2)", "",
         "New = (1 - cos) + 0.5 InfoNCE(tau 0.07); current = standard adapter. 100 D_ref presets, algorithmic and "
         "rephrased texts, mean per preset; paired Wilcoxon, Holm over 3 comparisons (sim_audio). MSS: lower is better.", "",
         "| method | sim current | sim new | diff | new better | p Holm | MSS current | MSS new | p MSS | same preset |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in righe:
        s_, ms = r["sim_audio"], r["MSS"]
        L.append(f"| {r['metodo']} | {s_['attuale']:.4f} | {s_['nuovo']:.4f} | {s_['diff']:+.4f} | "
                 f"{s_['quota_nuovo_meglio']:.0%} | {s_['p_holm']:.4f} | {ms['attuale']:.3f} | {ms['nuovo']:.3f} | "
                 f"{ms['p']:.4f} | {r.get('stesso_preset_recuperato', float('nan')):.0%} |")
    L += ["", f"**Outcome (rule fixed in advance):** {esito}."]
    open(OUT_MD, "w").write("\n".join(L) + "\n")
    json.dump(righe, open(OUT_MD.replace(".md", ".json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    addestra_fold()
    genera_sviluppo(a.limit)
    confronto()
