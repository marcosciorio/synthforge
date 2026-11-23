"""Functions shared by the final test scripts: batched generation with an
explicit initial noise (paired comparisons across methods), retrieval with
category forcing and leave-one-out search on the development set.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

import config as K
from config import C


def carica_adapter(percorso):
    ck = torch.load(percorso, map_location=C.DEVICE, weights_only=False)
    a = C.TextAdapter(ck.get("dim", 512)).to(C.DEVICE)
    a.load_state_dict(ck["state_dict"])
    a.eval()
    return a


def carica_tutto(plugin=True):
    b = C.load_all(load_plugin_=plugin)
    b.adapter = carica_adapter(K.ADAPTER_FINALE)
    b.pool_ok = [n for n in b.pool_names if b.pool_params.get(n)]
    b.indice_pool = {n: i for i, n in enumerate(b.pool_names)}
    print(f"[OK] adapter: {K.VARIANTE_ADAPTER} ({K.ADAPTER_FINALE})")
    return b


@torch.no_grad()
def embedding_testo(b, frasi):
    e = b.clap.get_text_embedding(list(frasi), use_tensor=True)
    return F.normalize(e.float(), dim=-1).to(C.DEVICE)


@torch.no_grad()
def adatta(adapter, e):
    return F.normalize(adapter(e), dim=-1)


def categoria_dal_testo(frase):
    return C.resolve_prompt_category(frase, {})


def recupera(b, e_adattato_np, categoria, escludi=None):
    nomi = b.pool_ok if escludi is None else [n for n in b.pool_ok if n != escludi]
    idx = np.array([b.indice_pool[n] for n in nomi])
    sims = b.pool_audio_embs[idx] @ e_adattato_np
    if K.FORCING and categoria:
        maschera = np.array([b.pool_categories.get(n, "") == categoria for n in nomi])
        if maschera.any():
            j = np.where(maschera)[0][int(np.argmax(sims[maschera]))]
            return nomi[int(j)]
    return nomi[int(np.argmax(sims))]


def rumore(dim, seed):
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(dim, generator=g)


def _campo(vf, x, t, cond, s):
    vc = vf(x, t, cond)
    if bool(torch.all(s == 1.0)):
        return vc
    return (1 - s) * vf(x, t, None) + s * vc


@torch.no_grad()
def genera(vf, cond, x0, s, passi, t_star=0.0, x_ret=None):
    x0 = x0.to(C.DEVICE)
    B = x0.shape[0]
    s = torch.as_tensor(s, dtype=torch.float32, device=C.DEVICE).reshape(-1, 1).expand(B, 1)
    if cond.shape[0] != B:
        cond = cond.expand(B, -1)
    dt = 1.0 / passi
    if x_ret is None:
        x, t0, n = x0.clone(), 0.0, passi
    else:
        x_ret = x_ret.to(C.DEVICE)
        x, t0 = (1.0 - t_star) * x0 + t_star * x_ret, t_star
        n = max(1, int(round(passi * (1.0 - t_star))))
    t = torch.full((B, 1), float(t0), device=C.DEVICE)
    for _ in range(n):
        k1 = _campo(vf, x, t, cond, s)
        k2 = _campo(vf, x + dt * k1 / 2, t + dt / 2, cond, s)
        k3 = _campo(vf, x + dt * k2 / 2, t + dt / 2, cond, s)
        k4 = _campo(vf, x + dt * k3, t + dt, cond, s)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + dt
    return x.cpu().numpy()


def a_preset(b, x_np):
    return C.enforce_audio_logic(b.spec.decode(x_np))


def emb_audio(b, audio):
    return C.embed_audio_clap(b.clap, audio, C.RENDER_SR)


def metriche_rapide(b, ref_mono, ref_emb, gen_mono):
    g = emb_audio(b, gen_mono)
    return {"sim_audio": float(np.dot(ref_emb, g)),
            "MSS": C.metric_mss(ref_mono, gen_mono, C.RENDER_SR)}, g


def metriche_complete(b, ref_mono, ref_emb, gen_mono, e_testo_np):
    g = emb_audio(b, gen_mono)
    return {
        "sim_audio":   float(np.dot(ref_emb, g)),
        "MSS":         C.metric_mss(ref_mono, gen_mono, C.RENDER_SR),
        "wMFCC":       C.metric_wmfcc(ref_mono, gen_mono, C.RENDER_SR),
        "SOT":         C.metric_sot(ref_mono, gen_mono, C.RENDER_SR),
        "RMS_cosine":  C.metric_rms_cosine(ref_mono, gen_mono, C.RENDER_SR),
        "CLAP_testo":  float(np.dot(e_testo_np, g)),
    }


def adapter_fold():
    import os
    m = json.load(open(os.path.join(K.DIR_FOLD, "fold_map.json")))
    if m.get("variante") != K.VARIANTE_ADAPTER:
        raise RuntimeError("fold adapters were trained with another variant: run p01_fold_adapters.py again")
    modelli = {k: carica_adapter(os.path.join(K.DIR_FOLD, f"fold{k}.pt")) for k in range(K.N_FOLD)}
    return {n: modelli[k] for n, k in m["fold_map"].items()}


def riferimento_sviluppo(b, nome):
    ref = C.render_preset(b.plugin, b.pool_params[nome])
    return ref, emb_audio(b, ref)


def chiavi_fatte(percorso, campi):
    fatte = set()
    try:
        with open(percorso) as f:
            for riga in f:
                r = json.loads(riga)
                fatte.add(tuple(r[c] for c in campi))
    except FileNotFoundError:
        pass
    return fatte


def aggiungi(percorso, riga):
    with open(percorso, "a") as f:
        f.write(json.dumps(riga, ensure_ascii=False) + "\n")
