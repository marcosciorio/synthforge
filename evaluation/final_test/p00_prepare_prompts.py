"""Build the prompts of the test set (algorithmic, guided and free
descriptions) and of the development set.
"""

import json
import os

import numpy as np

import config as K
from config import C


def prompt_test():
    descr = json.load(open(K.DESCRIZIONI))
    trad = {k: v["tradotta"] for k, v in json.load(open(K.TRADUZIONI)).items() if not k.startswith("_")}
    per_numero = {f[:4]: f[:-5] for f in os.listdir(C.CAPTION_DIR) if f.endswith(".json")}
    righe = []
    for r in sorted(descr, key=lambda r: r["sample_id"]):
        nome = r["sample_id"]
        if not os.path.isfile(os.path.join(C.CAPTION_DIR, nome + ".json")):
            print(f"  [name] '{nome}' -> '{per_numero[nome[:4]]}'")
            nome = per_numero[nome[:4]]
        cap = json.load(open(os.path.join(C.CAPTION_DIR, nome + ".json")))
        wav = os.path.join(C.AUDIO_DIR, nome + ".wav")
        assert os.path.isfile(wav), f"missing reference audio: {wav}"
        righe.append({
            "nome": nome,
            "categoria_vera": (cap.get("choices") or {}).get("category", ""),
            "wav": wav,
            "testi": {
                "algoritmica": cap["caption"],
                "guidata": r["guidata_unificata"],
                "libera": trad.get(nome, r["libera_originale"]),
            },
            "libera_tradotta": nome in trad,
            "esperienza": r.get("esperienza_label"),
        })
    for i, r in enumerate(righe):
        r["indice"] = i
    return righe


def prompt_sviluppo():
    rng = np.random.default_rng(42)
    test = {r["sample_id"] for r in json.load(open(K.DESCRIZIONI))}
    emb = np.load(C.EMB_NPZ, allow_pickle=True)
    nomi = [str(n) for n in emb["names"]]
    per_cat = {}
    for n in nomi:
        cp = os.path.join(C.NIP_CAPTION_DIR, n + ".json")
        jp = os.path.join(C.NIP_JSON_DIR, n + ".json")
        if n in test or not (os.path.isfile(cp) and os.path.isfile(jp)):
            continue
        cat = (json.load(open(cp)).get("choices") or {}).get("category", "")
        per_cat.setdefault(cat, []).append(n)
    tot = sum(len(v) for v in per_cat.values())
    quote = {c: K.N_SVILUPPO * len(v) / tot for c, v in per_cat.items()}
    n_cat = {c: int(q) for c, q in quote.items()}
    for c in sorted(quote, key=lambda c: quote[c] - n_cat[c], reverse=True)[: K.N_SVILUPPO - sum(n_cat.values())]:
        n_cat[c] += 1
    scelti = []
    for c in sorted(per_cat):
        scelti += list(rng.choice(sorted(per_cat[c]), size=n_cat[c], replace=False))
    righe = []
    for i, n in enumerate(sorted(scelti)):
        cap = json.load(open(os.path.join(C.NIP_CAPTION_DIR, n + ".json")))
        caps = cap.get("captions") or [cap["caption"]]
        varianti = caps[2:7]
        righe.append({
            "nome": n, "indice": i,
            "categoria_vera": (cap.get("choices") or {}).get("category", ""),
            "testi": {"algoritmica": caps[0],
                      "variante": varianti[int(rng.integers(len(varianti)))] if varianti else caps[0]},
        })
    return righe, n_cat


def main():
    t = prompt_test()
    json.dump(t, open(K.PROMPT_TEST, "w"), indent=1, ensure_ascii=False)
    print(f"[OK] test: {len(t)} sounds, {sum(r['libera_tradotta'] for r in t)} free descriptions translated -> {K.PROMPT_TEST}")
    s, n_cat = prompt_sviluppo()
    json.dump(s, open(K.PROMPT_SVILUPPO, "w"), indent=1, ensure_ascii=False)
    print(f"[OK] development set: {len(s)} D_ref presets {n_cat} -> {K.PROMPT_SVILUPPO}")


if __name__ == "__main__":
    main()
