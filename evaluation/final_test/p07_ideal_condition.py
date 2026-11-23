"""Diagnostic analysis: generation conditioned on the CLAP embedding of the
true audio (ideal condition), and the nearest reference preset to that audio.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.stats import wilcoxon

import config as K
from config import C

ORACOLO_JSONL = os.path.join(K.RISULTATI, "ideal_condition.jsonl")
RIEPILOGO = os.path.join(K.DIR_ANALISI, "ideal_condition_summary.md")
SCALE = {"oracolo_s1": 1.0, "oracolo_s2": 2.0}
CAMPI = ("nome", "metodo", "k")


def esegui(limite):
    import engine as M
    from p05_final_test import preset_vero

    passi = json.load(open(K.PARAMETRI))["n_passi"]
    b = M.carica_tutto()
    prompt = json.load(open(K.PROMPT_TEST))[:limite]
    fatte = M.chiavi_fatte(ORACOLO_JSONL, CAMPI)
    seeds = list(range(K.K_SEED_TEST))
    t0 = time.time()
    for i, p in enumerate(prompt):
        nome = p["nome"]
        ref, sr = sf.read(p["wav"], dtype="float32")
        ref = C._to_mono(ref)
        ref_emb = M.emb_audio(b, ref)
        x_vero = C.encode_preset(b.spec, preset_vero(nome))

        def misura(x, gen):
            g = M.emb_audio(b, gen)
            return {"sim_audio": float(np.dot(ref_emb, g)),
                    "MSS": C.metric_mss(ref, gen, C.RENDER_SR),
                    "RMS_cosine": C.metric_rms_cosine(ref, gen, C.RENDER_SR),
                    "MSE_param": float(np.mean((x - x_vero) ** 2))}

        def scrivi(metodo, k, m, extra):
            M.aggiungi(ORACOLO_JSONL, {"nome": nome, "metodo": metodo, "k": k, **m,
                                       "passi": passi, **extra})

        if (nome, "oracolo_vicino", -1) not in fatte:
            idx = np.array([b.indice_pool[n] for n in b.pool_ok])
            vicino = b.pool_ok[int(np.argmax(b.pool_audio_embs[idx] @ ref_emb))]
            x = C.encode_preset(b.spec, b.pool_params[vicino])
            scrivi("oracolo_vicino", -1, misura(x, C.render_preset(b.plugin, b.pool_params[vicino])),
                   {"recuperato": vicino})

        cond = torch.from_numpy(ref_emb.astype(np.float32)).unsqueeze(0).to(C.DEVICE)
        X0 = torch.stack([M.rumore(b.spec.dim, K.seed_di(p["indice"], k)) for k in seeds])
        for metodo, s in SCALE.items():
            if all((nome, metodo, k) in fatte for k in seeds):
                continue
            xs = M.genera(b.vf, cond, X0, s, passi)
            for k, x in zip(seeds, xs):
                if (nome, metodo, k) in fatte:
                    continue
                preset = M.a_preset(b, x)
                scrivi(metodo, k, misura(x, C.render_preset(b.plugin, preset)),
                       {"preset": {n: float(v) for n, v in preset.items()}})
        dt = time.time() - t0
        print(f"  [{i+1}/{len(prompt)}] {nome[:40]:40s} ({dt/60:.1f} min, "
              f"~{dt/(i+1)*(len(prompt)-i-1)/60:.0f} min remaining)", flush=True)
    print(f"[OK] ideal condition -> {ORACOLO_JSONL}")


def ic_bootstrap(v, rng):
    m = rng.choice(v, size=(K.N_BOOTSTRAP, len(v)), replace=True).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def analisi():
    rng = np.random.default_rng(0)
    t = pd.read_json(K.TEST_JSONL, lines=True)
    o = pd.read_json(ORACOLO_JSONL, lines=True)
    casuale = t[t.metodo == "casuale_categoria"].groupby("nome").sim_audio.mean()

    col = {m: o[o.metodo == m].groupby("nome").sim_audio.mean() for m in ["oracolo_s1", "oracolo_s2", "oracolo_vicino"]}
    for testo in K.TESTI:
        for m in ["adapter", "retrieval", "solo_retrieval"]:
            col[f"{m} | {testo}"] = t[(t.metodo == m) & (t.testo == testo)].groupby("nome").sim_audio.mean()
    sim = pd.DataFrame(col).dropna()
    rc = casuale.reindex(sim.index)
    score = sim.sub(rc, axis=0).div(1 - rc, axis=0)

    righe = []
    for c in score.columns:
        lo, hi = ic_bootstrap(score[c].values, rng)
        righe.append({"condizione": c, "n": len(score), "sim_audio": sim[c].mean(),
                      "score": score[c].mean(), "ic95_basso": lo, "ic95_alto": hi,
                      "score_mediana": score[c].median(),
                      "quota_sopra_casuale": (score[c] > 0).mean()})
    medie = pd.DataFrame(righe)

    confronti = [("oracolo_s1", None), ("oracolo_s2", None), ("oracolo_vicino", None)]
    confronti += [("oracolo_s1", f"adapter | {t_}") for t_ in K.TESTI]
    confronti += [("oracolo_vicino", f"solo_retrieval | {t_}") for t_ in K.TESTI]
    confronti += [("oracolo_s1", "oracolo_vicino"), ("oracolo_s2", "oracolo_s1")]
    righe = []
    for a, bb in confronti:
        d = score[a] - (0.0 if bb is None else score[bb])
        w = wilcoxon(d, zero_method="wilcox")
        lo, hi = ic_bootstrap(d.values, rng)
        nz = d[d != 0]
        rk = pd.Series(np.abs(nz)).rank().values
        r = (rk[nz.values > 0].sum() - rk[nz.values < 0].sum()) / rk.sum()
        righe.append({"A": a, "B": bb or "casuale_categoria", "diff_media": d.mean(),
                      "ic95_basso": lo, "ic95_alto": hi, "r_rank_biserial": r, "p": w.pvalue})
    conf = pd.DataFrame(righe)
    ordine = np.argsort(conf.p.values)
    m = len(conf)
    holm = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * conf.p.values[ordine]))
    conf.loc[conf.index[ordine], "p_holm"] = holm

    altre = o.groupby("metodo")[["MSS", "RMS_cosine", "MSE_param"]].mean()
    with open(RIEPILOGO, "w") as f:
        f.write("# Ideal condition test\n\n"
                "Condition = CLAP embedding of the true audio. score: 0 = random preset of the "
                "true category, 1 = reference (same random baseline as the final test).\n\n"
                "## Means\n\n```\n" + medie.to_string(index=False) + "\n```\n\n"
                "## Comparisons (paired Wilcoxon, Holm within this family)\n\n```\n"
                + conf.to_string(index=False) + "\n```\n\n"
                "## Other metrics (ideal condition)\n\n```\n" + altre.to_string() + "\n```\n")
    print(open(RIEPILOGO).read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    if not a.analysis_only:
        esegui(a.limit)
    analisi()
