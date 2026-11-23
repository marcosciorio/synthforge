"""Agreement between the words chosen by people in the guided descriptions
and the words computed by the captioner (Cohen's kappa).
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

import config as K

CAPTION_DIR = os.path.join(K.ROOT, "data", "Dataset_TAL_48kHz", "in_pool", "captions")
CAMPI = ["category", "brightness", "loudness", "weight", "texture", "timeShape"]

POLARITA = {
    "brightness": ({"bright", "sharp"}, {"dark", "muffled"}, {"warm", "mellow"}),
    "loudness":   ({"loud", "powerful", "aggressive"}, {"soft", "gentle"}),
    "weight":     ({"heavy", "deep", "full"}, {"light", "thin"}),
    "texture":    ({"rough", "harsh", "buzzy", "metallic", "glitchy", "wobbly"},
                   {"smooth", "clean", "lush", "hollow"}),
    "timeShape":  ({"short", "pulsing"}, {"long", "evolving", "slow"}, {"fading"}),
}
OPPOSTI = {"brightness": (0, 1), "loudness": (0, 1), "weight": (0, 1), "texture": (0, 1),
           "timeShape": (0, 1)}


def gruppo(campo, v):
    for i, g in enumerate(POLARITA[campo]):
        if v in g:
            return i
    return None


def kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    po = np.mean([x == y for x, y in zip(a, b)])
    va = pd.Series(a).value_counts(normalize=True)
    vb = pd.Series(b).value_counts(normalize=True)
    pe = float(sum(va.get(v, 0) * vb.get(v, 0) for v in set(a) | set(b)))
    return po, pe, (po - pe) / (1 - pe) if pe < 1 else np.nan, n


def main():
    desc = json.load(open(K.DESCRIZIONI))
    cap = {os.path.basename(p)[:4]: p for p in glob.glob(os.path.join(CAPTION_DIR, "*.json"))}
    righe = []
    for r in desc:
        c = json.load(open(cap[r["sample_id"][:4]]))["choices"]
        h = r["scelte_unificate"]
        for campo in CAMPI:
            vh, va = h.get(campo), c.get(campo)
            if vh in (None, "", "None") or va in (None, "", "None"):
                continue
            righe.append({"nome": r["sample_id"], "esperienza": r.get("esperienza_label", "?"),
                          "campo": campo, "persona": str(vh).lower(), "captioner": str(va).lower()})
    d = pd.DataFrame(righe)
    d.to_csv(os.path.join(K.DIR_ANALISI, "tag_agreement.csv"), index=False)

    tab = []
    for campo in CAMPI:
        x = d[d.campo == campo]
        po, pe, k, n = kappa(x.persona, x.captioner)
        riga = {"campo": campo, "n": n, "concordanza": po, "attesa_caso": pe, "kappa": k}
        if campo in POLARITA:
            gp = x.persona.map(lambda v: gruppo(campo, v))
            ga = x.captioner.map(lambda v: gruppo(campo, v))
            ok = gp.notna() & ga.notna()
            gp, ga = gp[ok], ga[ok]
            pp, pep, kp, _ = kappa(gp, ga)
            o1, o2 = OPPOSTI[campo]
            riga.update({"polarita": pp, "polarita_caso": pep, "kappa_polarita": kp,
                         "opposti": float(np.mean(((gp == o1) & (ga == o2)) | ((gp == o2) & (ga == o1))))})
        tab.append(riga)
    tab = pd.DataFrame(tab)

    conf_b = pd.crosstab(d[d.campo == "brightness"].persona, d[d.campo == "brightness"].captioner,
                         rownames=["persona"], colnames=["captioner"])

    desc_campi = d[d.campo != "category"].assign(ok=lambda z: z.persona == z.captioner)
    per_suono = desc_campi.groupby("nome").agg(concordi=("ok", "sum"), campi=("ok", "size"),
                                              esperienza=("esperienza", "first"))
    per_suono["quota"] = per_suono.concordi / per_suono.campi
    t = pd.read_json(K.TEST_JSONL, lines=True)
    casuale = t[t.metodo == "casuale_categoria"].groupby("nome").sim_audio.mean()
    legami = []
    for metodo in ["adapter", "retrieval", "solo_retrieval"]:
        s = t[(t.metodo == metodo) & (t.testo == "guidata")].groupby("nome").sim_audio.mean()
        sc = ((s - casuale) / (1 - casuale)).reindex(per_suono.index)
        rho, p = spearmanr(per_suono.quota, sc, nan_policy="omit")
        legami.append({"method (guided text)": metodo, "spearman_rho": rho, "p": p})
    legami = pd.DataFrame(legami)

    esp = per_suono.groupby("esperienza").quota.agg(["count", "mean"])
    gruppi = [g.quota.values for _, g in per_suono.groupby("esperienza") if len(g) > 1]
    p_esp = kruskal(*gruppi).pvalue if len(gruppi) > 1 else np.nan

    with open(os.path.join(K.DIR_ANALISI, "tag_agreement.md"), "w") as f:
        f.write("# Agreement between the tags chosen by people and by the captioner (122 sounds)\n\n"
                "Person: `scelte_unificate` of the guided description. Captioner: `choices` "
                "of the algorithmic caption of the same sound. Only fields present in both.\n\n"
                "## By field\n\n```\n" + tab.round(3).to_string(index=False) + "\n```\n\n"
                "Polarity groups: " + "; ".join(
                    f"{c}: " + " / ".join("{" + ",".join(sorted(g)) + "}" for g in gs)
                    for c, gs in POLARITA.items()) + ". `opposti` = share of cases in which person and captioner "
                "fall in the two opposite groups (for example bright against dark).\n\n"
                "## Brightness: confusion matrix\n\n```\n" + conf_b.to_string() + "\n```\n\n"
                "## Agreement per sound and final test score (guided text)\n\n"
                f"Mean share of agreeing descriptive fields per sound: {per_suono.quota.mean():.3f} "
                f"(median {per_suono.quota.median():.3f}, n = {len(per_suono)}).\n\n```\n"
                + legami.round(4).to_string(index=False) + "\n```\n\n"
                "## By declared experience\n\n```\n" + esp.round(3).to_string() + "\n```\n\n"
                f"Kruskal-Wallis: p = {p_esp:.3f}\n")
    print(open(os.path.join(K.DIR_ANALISI, "tag_agreement.md")).read())


if __name__ == "__main__":
    main()
