"""Choose the parameters of each method from the development grid,
with the one-standard-error rule.
"""

import json
import os

import numpy as np
import pandas as pd

import config as K


def _md(df):
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def scegli(df):
    per_preset = df.groupby(["s", "t_star", "nome"])["sim_audio"].mean().reset_index()
    g = per_preset.groupby(["s", "t_star"])["sim_audio"]
    tab = pd.DataFrame({"media": g.mean(), "se": g.std(ddof=1) / np.sqrt(g.count()), "n": g.count()}).reset_index()
    migliore = tab.loc[tab["media"].idxmax()]
    candidati = tab[tab["media"] >= migliore["media"] - migliore["se"]]
    scelta = candidati.sort_values(["s", "media"], ascending=[True, False]).iloc[0]
    return tab.sort_values("media", ascending=False), migliore, scelta


def main():
    df = pd.read_json(K.SVILUPPO_JSONL, lines=True)
    passi = json.load(open(K.CONTROLLO_PASSI))["n_passi"]
    out = {"n_passi": passi, "forcing": K.FORCING, "variante_adapter": K.VARIANTE_ADAPTER,
           "n_preset_sviluppo": int(df["nome"].nunique())}
    md = ["# Grid on the development set\n",
          f"{out['n_preset_sviluppo']} D_ref presets, 2 texts each, {passi} RK4 steps. "
          "Metric: CLAP audio-audio similarity with the reference (mean per preset, then over presets).\n"]
    for metodo in ("diretta", "adapter", "retrieval", "retrieval_grezzo"):
        d = df[df["metodo"] == metodo]
        tab, migliore, scelta = scegli(d)
        out[metodo] = {"s": float(scelta["s"]), "t_star": float(scelta["t_star"]),
                       "media": float(scelta["media"]), "se": float(scelta["se"]),
                       "migliore_assoluta": {"s": float(migliore["s"]), "t_star": float(migliore["t_star"]),
                                             "media": float(migliore["media"])}}
        per_testo = {}
        for testo, dt in d.groupby("testo"):
            _, m_t, _ = scegli(dt)
            per_testo[testo] = {"s": float(m_t["s"]), "t_star": float(m_t["t_star"])}
        out[metodo]["migliore_per_testo"] = per_testo
        md += [f"\n## {metodo}\n", _md(tab),
               f"\n\nSelected: s = {scelta['s']:g}" + (f", t* = {scelta['t_star']:g}" if metodo.startswith("retrieval") else "")
               + f" (mean {scelta['media']:.4f} ± {scelta['se']:.4f}). Best by text type: {per_testo}.\n"]
    grezzo = out.pop("retrieval_grezzo")
    adattato = out["retrieval"]
    vince_grezzo = grezzo["media"] > adattato["media"]
    out["retrieval"] = dict(grezzo if vince_grezzo else adattato, ricerca="grezzo" if vince_grezzo else "adattato",
                            alternativa={"ricerca": "adattato" if vince_grezzo else "grezzo",
                                         **(adattato if vince_grezzo else grezzo)})
    md.append(f"\n## Preset search\n\nAdapted: {adattato['media']:.4f} ± {adattato['se']:.4f}; "
              f"raw: {grezzo['media']:.4f} ± {grezzo['se']:.4f}. Selected: **{out['retrieval']['ricerca']}**.\n")
    for nome_r in ("solo_retrieval", "solo_retrieval_grezzo"):
        rif = df[df["metodo"] == nome_r].groupby("nome")["sim_audio"].mean()
        out[nome_r + "_media"] = float(rif.mean())
        md.append(f"\nReference {nome_r} (t* = 1): {rif.mean():.4f} ± {rif.std(ddof=1)/np.sqrt(len(rif)):.4f}\n")
    if "stesso_preset" in df:
        q = df[df["metodo"] == "solo_retrieval"]["stesso_preset"].mean()
        out["frazione_stesso_preset"] = float(q)
        md.append(f"\nThe two searches find the same preset for {100*q:.1f}% of the sentences.\n")
    json.dump(out, open(K.PARAMETRI, "w"), indent=1)
    open(os.path.join(K.DIR_ANALISI, "development_grid.md"), "w").write("\n".join(md))
    print(json.dumps({k: v for k, v in out.items() if k in ("n_passi", "diretta", "adapter", "retrieval")}, indent=1))


if __name__ == "__main__":
    main()
