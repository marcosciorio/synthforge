"""Right against wrong description: is each generated sound closer to its own
reference than sounds generated from the other descriptions?
"""

import json
import os
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import wilcoxon

import config as K
from config import C
import engine as M

OUT_NPZ = os.path.join(K.RISULTATI, "right_description.npz")
OUT_MD = os.path.join(K.DIR_ANALISI, "right_description.md")
COND = ["diretta", "adapter", "retrieval", "solo_retrieval"]


def rms(a):
    return librosa.feature.rms(y=a, frame_length=2048, hop_length=512)[0]


def cos_mat(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o])); out = np.empty(m); out[o] = adj
    return out


def main():
    b = M.carica_tutto()
    P = json.load(open(K.PROMPT_TEST))
    nomi = [p["nome"] for p in P]
    t = pd.read_json(K.TEST_JSONL, lines=True)
    rng = np.random.default_rng(20261117)
    refs, ref_emb, ref_rms = [], [], []
    for p in P:
        a, sr = sf.read(p["wav"], dtype="float32"); a = C._to_mono(a)
        refs.append(a); ref_emb.append(M.emb_audio(b, a)); ref_rms.append(rms(a))
    n_fr = min(len(r) for r in ref_rms)
    RE = np.array(ref_emb); RR = np.array([r[:n_fr] for r in ref_rms])
    sbagliati = {i: rng.choice([j for j in range(len(P)) if j != i], size=3, replace=False) for i in range(len(P))}
    salva, righe = {}, []
    t0 = time.time()
    for testo in K.TESTI:
        for c in COND:
            sel = t[(t.metodo == c) & (t.testo == testo) & ((t.k == 0) | (c == "solo_retrieval"))].set_index("nome")
            gen_emb, gen_rms, gens = [], [], []
            cache = {}
            for nome in nomi:
                r = sel.loc[nome]
                if c == "solo_retrieval":
                    chiave = r["recuperato"]
                    if chiave not in cache:
                        cache[chiave] = C.render_preset(b.plugin, b.pool_params[chiave])
                    g = cache[chiave]
                else:
                    g = C.render_preset(b.plugin, {k: float(v) for k, v in r["preset"].items()})
                gens.append(g); gen_emb.append(M.emb_audio(b, g)); gen_rms.append(rms(g)[:n_fr])
            S_clap = cos_mat(RE, np.array(gen_emb))
            S_rms = cos_mat(RR, np.array(gen_rms))
            mss_g = np.array([C.metric_mss(refs[i], gens[i], C.RENDER_SR) for i in range(len(P))])
            mss_s = np.array([np.mean([C.metric_mss(refs[i], gens[j], C.RENDER_SR) for j in sbagliati[i]])
                              for i in range(len(P))])
            salva[f"{c}|{testo}|clap"] = S_clap; salva[f"{c}|{testo}|rms"] = S_rms
            salva[f"{c}|{testo}|mss_g"] = mss_g; salva[f"{c}|{testo}|mss_s"] = mss_s
            off = ~np.eye(len(P), dtype=bool)
            for metr, S in [("CLAP", S_clap), ("RMS_cos", S_rms)]:
                g_ = np.diag(S); s_ = np.array([S[i, off[i]].mean() for i in range(len(P))])
                rango = np.array([1 + (S[i, off[i]] > S[i, i]).sum() for i in range(len(P))])
                righe.append({"metrica": metr, "testo": testo, "condizione": c, "giusto": g_.mean(),
                              "sbagliato": s_.mean(), "diff": (g_ - s_).mean(), "p": wilcoxon(g_ - s_).pvalue,
                              "rango_mediano": float(np.median(rango)), "quota_giusto_meglio": float((g_ > s_).mean())})
            righe.append({"metrica": "MSS", "testo": testo, "condizione": c, "giusto": mss_g.mean(),
                          "sbagliato": mss_s.mean(), "diff": (mss_g - mss_s).mean(),
                          "p": wilcoxon(mss_g - mss_s).pvalue, "rango_mediano": np.nan,
                          "quota_giusto_meglio": float((mss_g < mss_s).mean())})
            print(f"  {testo:11s} {c:15s} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    np.savez_compressed(OUT_NPZ, nomi=np.array(nomi), **salva)
    d = pd.DataFrame(righe)
    for m in d.metrica.unique():
        s = d.metrica == m
        d.loc[s, "p_holm"] = holm(d.loc[s, "p"].values)
    d.to_csv(OUT_MD.replace(".md", ".csv"), index=False)
    with open(OUT_MD, "w") as f:
        f.write("# Right against wrong description\n\n"
                "giusto = sound generated from the description of sound i, compared with reference i; "
                "sbagliato = mean over the sounds generated from the descriptions of the other sounds, compared with the same "
                "reference i (CLAP and RMS-cos on all, MSS on 3 at random). Seed 0. Rank: position of the right sound "
                "among the 122 generated for reference i (about 61.5 at random).\n\n```\n"
                + d.round(4).to_string(index=False) + "\n```\n")
    print(open(OUT_MD).read())


if __name__ == "__main__":
    main()
