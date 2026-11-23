"""Final test on the 122 test sounds, in a single run.
"""

import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

import config as K
import engine as M
from config import C

CAMPI = ("nome", "testo", "metodo", "k")


def preset_vero(nome):
    raw = json.load(open(os.path.join(C.JSON_DIR, nome + ".json")))
    return {k: float(v) for k, v in (raw.get("parameters", raw) or {}).items() if isinstance(v, (int, float))}


def misura(b, ref, ref_emb, gen, x_vero, x_gen, e_testi):
    g = M.emb_audio(b, gen)
    base = {"sim_audio": float(np.dot(ref_emb, g)),
            "MSS": C.metric_mss(ref, gen, C.RENDER_SR),
            "wMFCC": C.metric_wmfcc(ref, gen, C.RENDER_SR),
            "SOT": C.metric_sot(ref, gen, C.RENDER_SR),
            "RMS_cosine": C.metric_rms_cosine(ref, gen, C.RENDER_SR),
            "MSE_param": float(np.mean((x_gen - x_vero) ** 2))}
    return base, {t: float(np.dot(e, g)) for t, e in e_testi.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    P = json.load(open(K.PARAMETRI))
    passi = P["n_passi"]
    b = M.carica_tutto()
    prompt = json.load(open(K.PROMPT_TEST))[: a.limit]
    fatte = M.chiavi_fatte(K.TEST_JSONL, CAMPI)
    seeds = list(range(K.K_SEED_TEST))
    comuni = {"passi": passi, "variante_adapter": K.VARIANTE_ADAPTER}
    t0 = time.time()
    for i, p in enumerate(prompt):
        nome = p["nome"]
        ref, sr = sf.read(p["wav"], dtype="float32")
        ref = C._to_mono(ref)
        assert sr == C.RENDER_SR, f"{nome}: sr {sr}"
        ref_emb = M.emb_audio(b, ref)
        x_vero = C.encode_preset(b.spec, preset_vero(nome))
        e_raw = {t: M.embedding_testo(b, [f]) for t, f in p["testi"].items()}
        e_raw_np = {t: e.cpu().numpy()[0] for t, e in e_raw.items()}

        def scrivi(testo, metodo, k, base, clap, extra):
            M.aggiungi(K.TEST_JSONL, {"nome": nome, "testo": testo, "metodo": metodo, "k": k,
                                      **base, "CLAP_testo": clap, **extra, **comuni})

        if not all((nome, t, "riferimento", -1) in fatte for t in K.TESTI):
            for t in K.TESTI:
                scrivi(t, "riferimento", -1, {"sim_audio": 1.0, "MSS": 0.0, "MSE_param": 0.0},
                       float(np.dot(e_raw_np[t], ref_emb)), {})
        if not all((nome, t, "casuale_categoria", j) in fatte for t in K.TESTI for j in range(K.N_RANDOM)):
            rng = np.random.default_rng(K.seed_di(p["indice"], 999))
            stessa = [n for n in b.pool_ok if b.pool_categories.get(n, "") == p["categoria_vera"]]
            pool = stessa or b.pool_ok
            for j, n in enumerate(rng.choice(pool, size=K.N_RANDOM, replace=len(pool) < K.N_RANDOM)):
                x = C.encode_preset(b.spec, b.pool_params[n])
                base, clap = misura(b, ref, ref_emb, C.render_preset(b.plugin, b.pool_params[n]), x_vero, x, e_raw_np)
                for t in K.TESTI:
                    scrivi(t, "casuale_categoria", j, base, clap[t], {"preset": str(n)})

        for testo, frase in p["testi"].items():
            if all((nome, testo, m, k) in fatte for m in K.METODI for k in seeds) and \
               (nome, testo, "solo_retrieval", -1) in fatte:
                continue
            e_ad = M.adatta(b.adapter, e_raw[testo])
            cat = M.categoria_dal_testo(frase)
            ricerca = P["retrieval"].get("ricerca", "adattato")
            e_cerca = e_ad if ricerca == "adattato" else e_raw[testo]
            rec = M.recupera(b, e_cerca.cpu().numpy()[0], cat)
            x_ret = C.encode_preset(b.spec, b.pool_params[rec])
            X0 = torch.stack([M.rumore(b.spec.dim, K.seed_di(p["indice"], k)) for k in seeds])
            uscite = {
                "diretta": M.genera(b.vf, e_raw[testo], X0, P["diretta"]["s"], passi),
                "adapter": M.genera(b.vf, e_ad, X0, P["adapter"]["s"], passi),
                "retrieval": M.genera(b.vf, e_ad, X0, P["retrieval"]["s"], passi,
                                      t_star=P["retrieval"]["t_star"],
                                      x_ret=torch.from_numpy(x_ret).unsqueeze(0).expand(len(seeds), -1)),
            }
            info = {"categoria_testo": cat, "recuperato": rec, "ricerca": ricerca}
            if (nome, testo, "solo_retrieval", -1) not in fatte:
                base, clap = misura(b, ref, ref_emb, C.render_preset(b.plugin, b.pool_params[rec]),
                                    x_vero, x_ret, {testo: e_raw_np[testo]})
                scrivi(testo, "solo_retrieval", -1, base, clap[testo], info)
            for metodo, xs in uscite.items():
                for k, x in zip(seeds, xs):
                    if (nome, testo, metodo, k) in fatte:
                        continue
                    preset = M.a_preset(b, x)
                    base, clap = misura(b, ref, ref_emb, C.render_preset(b.plugin, preset),
                                        x_vero, x, {testo: e_raw_np[testo]})
                    scrivi(testo, metodo, k, base, clap[testo],
                           {**info, "preset": {n: float(v) for n, v in preset.items()}})
        dt = time.time() - t0
        print(f"  [{i+1}/{len(prompt)}] {nome[:40]:40s} ({dt/60:.1f} min, "
              f"~{dt/(i+1)*(len(prompt)-i-1)/60:.0f} min remaining)", flush=True)
    print(f"[OK] test -> {K.TEST_JSONL}")


if __name__ == "__main__":
    main()
