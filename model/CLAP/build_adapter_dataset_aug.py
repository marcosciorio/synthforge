"""Build the text and audio CLAP embeddings used to train the text adapter.
"""

import os, json, sys, time
import numpy as np
import torch

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

AUG_JSON      = os.path.join(PROJECT_ROOT, "model", "kaggle_training_cfm", "augmented_captions.json")
BASE_NPZ      = os.path.join(PROJECT_ROOT, "model", "kaggle_training_cfm", "adapter_embeddings.npz")
OLD_AUG_NPZ   = os.path.join(PROJECT_ROOT, "model", "kaggle_training_cfm", "adapter_embeddings_aug.npz")
OUT_NPZ       = os.path.join(PROJECT_ROOT, "model", "kaggle_training_cfm", "adapter_embeddings_aug.npz")
OUT_NPZ_TMP   = os.path.join(PROJECT_ROOT, "model", "kaggle_training_cfm", "adapter_embeddings_aug.new.npz")
CLAP_CKPT = os.path.join(PROJECT_ROOT, "model", "kaggle_training_cfm",
                         "music_audioset_epoch_15_esc_90.14.pt")

_mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
DEVICE  = "mps" if _mps_ok else ("cuda" if torch.cuda.is_available() else "cpu")


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def load_name_to_audio() -> dict[str, np.ndarray]:
    if os.path.isfile(BASE_NPZ):
        print(f"Loading the original audio embeddings: {BASE_NPZ}")
        orig = np.load(BASE_NPZ, allow_pickle=True)
        orig_names = list(orig["names"])
        orig_A     = orig["A"]
        name_to_audio = {}
        for i, name in enumerate(orig_names):
            name_to_audio[str(name)] = orig_A[i]
        print(f"  {len(name_to_audio)} presets")
        return name_to_audio

    if os.path.isfile(OLD_AUG_NPZ):
        print(f"adapter_embeddings.npz not found, using {OLD_AUG_NPZ} (deduplicated by name)")
        old = np.load(OLD_AUG_NPZ, allow_pickle=True)
        old_A, old_names = old["A"], old["names"]
        name_to_audio = {}
        for i, name in enumerate(old_names):
            if name not in name_to_audio:
                name_to_audio[name] = old_A[i]
        print(f"  {len(name_to_audio)} presets (deduplicated from {len(old_names)} rows)")
        return name_to_audio

    print("Error: neither adapter_embeddings.npz nor adapter_embeddings_aug.npz was found.")
    sys.exit(1)


def main():
    print(f"Loading augmented captions: {AUG_JSON}")
    with open(AUG_JSON, encoding="utf-8") as f:
        aug: dict = json.load(f)
    n_captions = sum(len(v) for v in aug.values())
    print(f"  {len(aug)} presets  |  {n_captions} captions in total")

    name_to_audio = load_name_to_audio()

    try:
        import laion_clap
    except ImportError:
        print("Error: install laion-clap with pip install laion-clap --no-deps")
        sys.exit(1)

    if not os.path.isfile(CLAP_CKPT):
        print(f"CLAP checkpoint not found: {CLAP_CKPT}")
        sys.exit(1)

    print(f"\nDevice: {DEVICE}")
    print("Loading CLAP...")
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=DEVICE)
    clap.load_ckpt(CLAP_CKPT)
    clap.eval()
    print("CLAP loaded.\n")

    all_T, all_A, all_names = [], [], []
    skipped = []
    t0   = time.time()
    done = 0

    for preset_name, captions in aug.items():
        audio_emb = name_to_audio.get(preset_name)
        if audio_emb is None:
            skipped.append(preset_name)
            continue

        with torch.no_grad():
            t_embs = clap.get_text_embedding(captions, use_tensor=False)

        for t_emb in t_embs:
            all_T.append(t_emb.astype(np.float32))
            all_A.append(audio_emb.astype(np.float32))
            all_names.append(preset_name)
            done += 1

        if done % 200 == 0 or done == n_captions:
            print(f"  {done}/{n_captions}  ({time.time() - t0:.1f}s)")

    if skipped:
        print(f"\nPresets missing from the audio embeddings ({len(skipped)}): {skipped}")

    T         = l2norm(np.stack(all_T))
    A         = l2norm(np.stack(all_A))
    names_arr = np.array(all_names)

    print(f"\nT: {T.shape}  A: {A.shape}")

    cos = (T * A).sum(axis=1)
    print("\n=== Text-audio gap (augmented set) ===")
    print(f"  cosine mean : {cos.mean():.4f}")
    print(f"  cosine std  : {cos.std():.4f}")
    print(f"  min / max   : {cos.min():.4f} / {cos.max():.4f}")

    np.savez_compressed(OUT_NPZ_TMP, T=T, A=A, names=names_arr)
    os.replace(OUT_NPZ_TMP, OUT_NPZ)
    print(f"\n✓  Saved: {OUT_NPZ}")
    print("Next step: python CLAP/train_text_adapter.py --aug")


if __name__ == "__main__":
    main()
