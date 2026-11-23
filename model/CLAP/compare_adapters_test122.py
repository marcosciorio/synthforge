"""Compare text adapters on the 122 test sounds: cosine similarity between
adapted text and audio embeddings, and text-to-audio retrieval.
"""

import os, sys, json, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(os.path.dirname(SCRIPT_DIR))
INPOOL     = os.path.join(ROOT, "data", "Dataset_TAL_48kHz", "in_pool")
NOTINPOOL  = os.path.join(ROOT, "data", "Dataset_TAL_48kHz", "not_in_pool")
HUMAN_JSON = os.path.join(ROOT, "listening_test", "backend", "results_server",
                          "unified_descriptions.json")
CLAP_CKPT  = os.path.join(ROOT, "model", "kaggle_training_cfm",
                          "music_audioset_epoch_15_esc_90.14.pt")
BASE_NPZ   = os.path.join(ROOT, "model", "kaggle_training_cfm", "adapter_embeddings.npz")
CACHE_NPZ  = os.path.join(SCRIPT_DIR, "test122_audio_embeddings.npz")
CLAP_SR    = 48000
DEVICE     = "cpu"


class TextAdapter(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        return F.normalize(t @ self.W.T + self.b, dim=-1)


def l2norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def load_audio_48k(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != CLAP_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=CLAP_SR)
    audio = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    audio = (audio / 32767.0).astype(np.float32)
    return audio.reshape(1, -1).astype(np.float32)


_CLAP = None
def clap():
    global _CLAP
    if _CLAP is None:
        import laion_clap
        m = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=DEVICE)
        m.load_ckpt(CLAP_CKPT)
        m.eval()
        _CLAP = m
    return _CLAP


def audio_emb(path):
    with torch.no_grad():
        e = clap().get_audio_embedding_from_data(x=load_audio_48k(path), use_tensor=False)[0]
    return l2norm(e.astype(np.float32))


def text_emb(texts):
    with torch.no_grad():
        e = clap().get_text_embedding(texts, use_tensor=False)
    return l2norm(np.asarray(e, dtype=np.float32))


def check_pipeline(n=3):
    d = np.load(BASE_NPZ, allow_pickle=True)
    names, A = list(d["names"]), l2norm(d["A"].astype(np.float32))
    cs = []
    for i in range(n):
        e = audio_emb(os.path.join(NOTINPOOL, "audio", names[i] + ".wav"))
        cs.append(float(e @ A[i]))
    return cs


def get_audio_embeddings(names):
    if os.path.isfile(CACHE_NPZ):
        c = np.load(CACHE_NPZ, allow_pickle=True)
        if list(c["names"]) == names:
            print(f"Audio embeddings loaded from cache: {CACHE_NPZ}")
            return c["A"].astype(np.float32)
    print("Computing the audio embeddings of the 122 test sounds (CPU)...")
    A = np.stack([audio_emb(os.path.join(INPOOL, "audio", n + ".wav")) for n in names])
    np.savez_compressed(CACHE_NPZ, A=A, names=np.array(names))
    print(f"  saved to {CACHE_NPZ}")
    return A


def metrics(T, A):
    S = T @ A.T
    matched = np.diag(S)
    ranks = (S > matched[:, None]).sum(axis=1)
    return {"cos": float(matched.mean()), "cos_std": float(matched.std()),
            "top1": float((ranks < 1).mean()), "top5": float((ranks < 5).mean()),
            "rank_median": float(np.median(ranks) + 1), "n": int(len(T))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", default=[], help="name=path.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(INPOOL, "captions"))}
    fix = {f.replace("'", ""): f for f in files}
    human = {fix.get(h["sample_id"], h["sample_id"]): h
             for h in json.load(open(HUMAN_JSON, encoding="utf-8"))}
    names = sorted(human)
    assert set(names) == files
    assert len(names) == 122, len(names)
    caps = {n: json.load(open(os.path.join(INPOOL, "captions", n + ".json"), encoding="utf-8"))
            for n in names}

    cs = check_pipeline()
    print("Audio pipeline check (cosine with adapter_embeddings.npz):", [round(c, 4) for c in cs])

    A = get_audio_embeddings(names)

    sources = {
        "algoritmica": [caps[n]["caption"] for n in names],
        "algoritmica + space": [caps[n]["caption"] + (f", {caps[n]['choices']['space']}"
                                if caps[n]["choices"].get("space") else "") for n in names],
        "umana guidata (unificata)": [human[n]["guidata_unificata"] for n in names],
        "umana guidata (originale)": [human[n]["guidata_originale"] for n in names],
        "umana libera": [human[n]["libera_originale"] for n in names],
    }
    T_raw = {k: text_emb(v) for k, v in sources.items()}

    adapters = {"no adapter": None}
    for spec in args.adapter:
        k, p = spec.split("=", 1)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = TextAdapter(ck.get("dim", 512)); m.load_state_dict(ck["state_dict"]); m.eval()
        adapters[k] = m

    res = {}
    for an, m in adapters.items():
        for sn, T in T_raw.items():
            if m is not None:
                with torch.no_grad():
                    T = m(torch.from_numpy(T)).numpy()
            res[f"{an} | {sn}"] = metrics(T, A)

    print(f"\n{'adapter':<14} {'text':<28} {'cos':>7} {'top1':>6} {'top5':>6} {'med. rank':>9}")
    for k, r in res.items():
        an, sn = k.split(" | ")
        print(f"{an:<14} {sn:<28} {r['cos']:>7.4f} {100*r['top1']:>5.1f}% {100*r['top5']:>5.1f}% {r['rank_median']:>9.1f}")
    print("chance: top1 0.8%, top5 4.1%, median rank ~61")

    if args.out:
        json.dump({"pipeline_check": cs, "results": res}, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
