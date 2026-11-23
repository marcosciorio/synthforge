"""Analysis of the final test: normalised score, planned paired comparisons
(Wilcoxon signed-rank test, Holm correction), bootstrap confidence intervals.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config as K

METRICHE = ["sim_audio", "MSS", "wMFCC", "SOT", "RMS_cosine", "CLAP_testo", "MSE_param"]
ORDINE_METODI = ["casuale_categoria", "diretta", "adapter", "retrieval", "solo_retrieval", "riferimento"]


def tabella_md(df):
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def ic_bootstrap(v, n=K.N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


def holm(p):
    p = np.asarray(p, float)
    ordine = np.argsort(p)
    corr = np.empty_like(p)
    corrente = 0.0
    for r, i in enumerate(ordine):
        corrente = max(corrente, min(1.0, (len(p) - r) * p[i]))
        corr[i] = corrente
    return corr


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = pd.Series(np.abs(d)).rank().values
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def per_suono(df):
    m = df.groupby(["nome", "testo", "metodo"])[[c for c in METRICHE if c in df]].mean().reset_index()
    cas = m[m.metodo == "casuale_categoria"].set_index(["nome", "testo"])
    rif = m[m.metodo == "riferimento"].set_index(["nome", "testo"])
    idx = m.set_index(["nome", "testo"]).index
    sim_c = cas["sim_audio"].reindex(idx).values
    m["score"] = (m["sim_audio"].values - sim_c) / (1.0 - sim_c)
    clap_c = cas["CLAP_testo"].reindex(idx).values
    clap_r = rif["CLAP_testo"].reindex(idx).values
    m["score_testo"] = (m["CLAP_testo"].values - clap_c) / (clap_r - clap_c)
    return m


def confronti(m, metrica):
    w = m.pivot_table(index="nome", columns=["testo", "metodo"], values=metrica)
    coppie = []
    for t in K.TESTI:
        coppie += [((t, "adapter"), (t, "diretta")), ((t, "retrieval"), (t, "adapter")),
                   ((t, "retrieval"), (t, "solo_retrieval"))]
        coppie += [((t, x), (t, "casuale_categoria")) for x in K.METODI]
    for x in K.METODI:
        coppie += [(("algoritmica", x), ("guidata", x)), (("guidata", x), ("libera", x)),
                   (("algoritmica", x), ("libera", x))]
    righe = []
    for a, b in coppie:
        d = (w[a] - w[b]).dropna().values
        lo, hi = ic_bootstrap(d)
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        righe.append({"metrica": metrica, "A": f"{a[1]} | {a[0]}", "B": f"{b[1]} | {b[0]}", "n": len(d),
                      "diff_media": d.mean(), "ic95_basso": lo, "ic95_alto": hi,
                      "diff_mediana": np.median(d), "r_rank_biserial": rank_biserial(d), "p": p})
    out = pd.DataFrame(righe)
    out["p_holm"] = holm(out["p"].values)
    return out


def main():
    df = pd.read_json(K.TEST_JSONL, lines=True)
    m = per_suono(df)
    n_suoni = m["nome"].nunique()

    medie = []
    for (t, x), g in m.groupby(["testo", "metodo"]):
        r = {"testo": t, "metodo": x, "n": len(g)}
        for c in ["score", "score_testo"] + METRICHE:
            if c in g and g[c].notna().any():
                v = g[c].dropna().values
                r[c] = v.mean()
                if c == "score":
                    r["score_ic95_basso"], r["score_ic95_alto"] = ic_bootstrap(v)
        medie.append(r)
    medie = pd.DataFrame(medie)
    medie["ordine"] = medie["metodo"].map({x: i for i, x in enumerate(ORDINE_METODI)})
    medie = medie.sort_values(["testo", "ordine"]).drop(columns="ordine")
    medie.to_csv(os.path.join(K.DIR_ANALISI, "test_means.csv"), index=False)

    conf = pd.concat([confronti(m, "score"), confronti(m, "MSS")], ignore_index=True)
    conf.to_csv(os.path.join(K.DIR_ANALISI, "test_comparisons.csv"), index=False)

    md = ["# Final test on the D_test sounds\n",
          f"{n_suoni} sounds, {K.K_SEED_TEST} seeds per condition, adapter: {K.VARIANTE_ADAPTER}.\n",
          "score: 0 = random preset of the same category, 1 = reference audio "
          "(CLAP audio-audio similarity).\n",
          "\n## Means by text type and method\n",
          tabella_md(medie[["testo", "metodo", "n", "score", "score_ic95_basso", "score_ic95_alto",
                            "score_testo", "MSS", "wMFCC", "SOT", "RMS_cosine", "MSE_param"]]),
          "\n\n## Planned comparisons (paired Wilcoxon, Holm)\n"]
    for metrica, g in conf.groupby("metrica", sort=False):
        md += [f"\n### {metrica}\n", tabella_md(g.drop(columns="metrica"))]
    open(os.path.join(K.DIR_ANALISI, "test_summary.md"), "w").write("\n".join(md))
    print(f"[OK] analysis -> {K.DIR_ANALISI}")


if __name__ == "__main__":
    main()
