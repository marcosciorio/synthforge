"""Fixed-preset control: the reference preset closest to the centroid of the
CLAP audio embeddings, used for every test sound.
"""

import json, os
import numpy as np, pandas as pd, soundfile as sf
import config as K
from config import C
import engine as M
from p05_final_test import preset_vero

OUT = os.path.join(K.RISULTATI, "fixed_preset.jsonl")
b = M.carica_tutto()
idx = np.array([b.indice_pool[n] for n in b.pool_ok])
E = b.pool_audio_embs[idx]; E = E / np.linalg.norm(E, axis=1, keepdims=True)
c = E.mean(0); c /= np.linalg.norm(c)
fisso = b.pool_ok[int(np.argmax(E @ c))]
print("fixed preset:", fisso, b.pool_categories.get(fisso))
gen = C.render_preset(b.plugin, b.pool_params[fisso]); g = M.emb_audio(b, gen)
x = C.encode_preset(b.spec, b.pool_params[fisso])
if os.path.exists(OUT): os.remove(OUT)
for p in json.load(open(K.PROMPT_TEST)):
    ref, sr = sf.read(p["wav"], dtype="float32"); ref = C._to_mono(ref)
    re_ = M.emb_audio(b, ref)
    M.aggiungi(OUT, {"nome": p["nome"], "preset": fisso, "sim_audio": float(np.dot(re_, g)),
                     "MSS": C.metric_mss(ref, gen, C.RENDER_SR), "wMFCC": C.metric_wmfcc(ref, gen, C.RENDER_SR),
                     "SOT": C.metric_sot(ref, gen, C.RENDER_SR), "RMS_cosine": C.metric_rms_cosine(ref, gen, C.RENDER_SR),
                     "MSE_param": float(np.mean((x - C.encode_preset(b.spec, preset_vero(p["nome"]))) ** 2))})
d = pd.read_json(OUT, lines=True)
print(d[["sim_audio", "MSS", "wMFCC", "SOT", "RMS_cosine", "MSE_param"]].mean().round(4).to_string())
q = pd.read_json(os.path.join(K.RISULTATI, "random_any_category.jsonl"), lines=True).groupby("nome").sim_audio.mean()
d = d.set_index("nome"); s = (d.sim_audio - q) / (1 - q)
print("score against the any-category random preset", round(s.mean(), 4), "share above zero", round((s > 0).mean(), 3))
