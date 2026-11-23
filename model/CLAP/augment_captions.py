"""Collect the seven text variants of each reference preset
(algorithmic caption, preset name and five rephrasings).
"""

import json
import os
import glob

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CAPTION_DIR = os.path.join(ROOT, "data", "Dataset_TAL_48kHz", "not_in_pool", "captions")
OUT_PATH    = os.path.join(ROOT, "model", "kaggle_training_cfm", "augmented_captions.json")


def augment(d: dict) -> list:
    variants = d.get("captions") or [d["caption"]]

    seen, result = set(), []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result


def main():
    files = sorted(glob.glob(os.path.join(CAPTION_DIR, "*.json")))
    print(f"Found {len(files)} caption files")

    augmented = {}

    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            d = json.load(f)

        preset_name = os.path.splitext(os.path.basename(fpath))[0]
        variants    = augment(d)
        augmented[preset_name] = variants

        if list(augmented.keys()).index(preset_name) < 3:
            print(f"\n── {preset_name} ──")
            for i, v in enumerate(variants, 1):
                print(f"  v{i}: {v}")

    n_total   = sum(len(v) for v in augmented.values())
    n_presets = len(augmented)
    avg       = n_total / n_presets
    print(f"\n✓  {n_presets} presets  |  {n_total} total captions  |  avg {avg:.1f} per preset")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)
    print(f"✓  saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
