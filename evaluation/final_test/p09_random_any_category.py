"""Random preset of any category as a reference, measured with all the
metrics of the test, and comparisons against the methods.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import wilcoxon

import config as K
from config import C

OUT_JSONL = os.path.join(K.RISULTATI, "random_any_category.jsonl")
OUT_MD = os.path.join(K.DIR_ANALISI, "random_any_category.md")
OUT_CSV = os.path.join(K.DIR_ANALISI, "random_any_category_means.csv")


def esegui():
    import engine as M
    from p05_final_test import preset_vero
    b = M.carica_tutto()
    prompt = json.load(open(K.PROMPT_TEST))
    fatte = M.chiavi_fatte(OUT_JSONL, ("nome", "j"))
    t0 = time.time()
    for i, p in enumerate(prompt):
        nome = p["nome"]
        if all((nome, j) in fatte for j in range(K.N_RANDOM)):
            continue
        ref, sr = sf.read(p["wav"], dtype="float32")
        ref = C._to_mono(ref)
        ref_emb = M.emb_audio(b, ref)
        x_vero = C.encode_preset(b.spec, preset_vero(nome))
        rng = np.random.default_rng(K.seed_di(p["indice"], 998))
        for j, n in enumerate(rng.choice(b.pool_ok, size=K.N_RANDOM, replace=False)):
            if (nome, j) in fatte:
                continue
            gen = C.render_preset(b.plugin, b.pool_params[n])
            x = C.encode_preset(b.spec, b.pool_params[n])
            g = M.emb_audio(b, gen)
            M.aggiungi(OUT_JSONL, {
                "nome": nome, "j": j, "preset": str(n), "categoria_preset": b.pool_categories.get(n, ""),
                "categoria_vera": p["categoria_vera"],
                "sim_audio": float(np.dot(ref_emb, g)), "MSS": C.metric_mss(ref, gen, C.RENDER_SR),
                "wMFCC": C.metric_wmfcc(ref, gen, C.RENDER_SR), "SOT": C.metric_sot(ref, gen, C.RENDER_SR),
                "RMS_cosine": C.metric_rms_cosine(ref, gen, C.RENDER_SR),
                "MSE_param": float(np.mean((x - x_vero) ** 2))})
        dt = time.time() - t0
        print(f"  [{i+1}/{len(prompt)}] {nome[:40]:40s} ({dt/60:.1f} min)", flush=True)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    out = np.empty(m)
    out[o] = adj
    return out


def analisi():
    rng = np.random.default_rng(0)
    q = pd.read_json(OUT_JSONL, lines=True)
    t = pd.read_json(K.TEST_JSONL, lines=True)
    metr = ["sim_audio", "MSS", "wMFCC", "SOT", "RMS_cosine", "MSE_param"]
    qa = q.groupby("nome")[metr].mean()
    cond = ["diretta", "adapter", "retrieval", "solo_retrieval"]
    righe, conf = [], []
    for testo in K.TESTI:
        for c in cond:
            x = t[(t.metodo == c) & (t.testo == testo)].groupby("nome")[metr].mean().reindex(qa.index)
            s_any = (x.sim_audio - qa.sim_audio) / (1 - qa.sim_audio)
            boot = rng.choice(s_any.values, size=(K.N_BOOTSTRAP, len(s_any))).mean(1)
            righe.append({"testo": testo, "metodo": c, "score_any": s_any.mean(),
                          "ic95_basso": np.percentile(boot, 2.5), "ic95_alto": np.percentile(boot, 97.5),
                          "sopra": (s_any > 0).mean(), **{m: x[m].mean() for m in metr}})
            for m in ["score", "MSS"]:
                d = s_any if m == "score" else (x.MSS - qa.MSS)
                conf.append({"metrica": m, "testo": testo, "metodo": c, "diff_media": d.mean(),
                             "p": wilcoxon(d).pvalue})
    righe.append({"testo": "-", "metodo": "casuale_qualsiasi", **{m: qa[m].mean() for m in metr}})
    cat = t[t.metodo == "casuale_categoria"].groupby("nome")[metr].mean().reindex(qa.index)
    righe.append({"testo": "-", "metodo": "casuale_categoria", **{m: cat[m].mean() for m in metr}})
    medie = pd.DataFrame(righe)
    conf = pd.DataFrame(conf)
    for m in ["score", "MSS"]:
        sel = conf.metrica == m
        conf.loc[sel, "p_holm"] = holm(conf.loc[sel, "p"].values)
    stessa = (q.categoria_preset == q.categoria_vera).mean()
    medie.to_csv(OUT_CSV, index=False)
    conf.to_csv(OUT_CSV.replace("_means", "_comparisons"), index=False)
    with open(OUT_MD, "w") as f:
        f.write("# Reference: random preset of any category\n\n"
                f"5 D_ref presets per sound, drawn from the whole pool; share of the same category as the sound: {stessa:.3f}.\n\n"
                "## Means\n\n```\n" + medie.round(4).to_string(index=False) + "\n```\n\n"
                "## Comparisons with the any-category random preset (Wilcoxon, Holm over 12 per metric)\n\n```\n"
                + conf.round(5).to_string(index=False) + "\n```\n")
    print(open(OUT_MD).read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    if not a.analysis_only:
        esegui()
    analisi()
