"""Algorithmic captioner for TAL-NoiseMaker presets.

Describes each preset with a closed vocabulary along seven dimensions
(category, brightness, texture, loudness, weight, timeShape, space), using
the AudioCommons timbral features and the preset parameters.
"""

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


CATEGORY_PREFIX_MAP = {
    "LD": "lead", "BS": "bass", "BASS": "bass", "FX": "fx", "PD": "pad",
    "DR": "drum", "KB": "keyboard", "KY": "keyboard",
    "ARP": "arp", "CH": "chord", "CD": "chord", "PL": "pluck",
}

_LFO_DEST = {0:"off",1:"cutoff",2:"osc1",3:"osc2",4:"pw",5:"fm",6:"lfo",7:"osc1+2"}
LFO_ATTIVO = 0.1
def _attivo(amount: float) -> bool:
    return abs(amount - 0.5) > LFO_ATTIVO
def _lfo_dest(val: float) -> str:
    return _LFO_DEST[min(int(round(val * 7)), 7)]

FEATURES = ["brightness", "depth", "hardness", "roughness", "warmth", "sharpness", "boominess"]

RESID_MARGIN = 0.4


def extract_category(stem: str) -> Optional[str]:
    body = stem.split("_", 1)[1] if "_" in stem else stem
    prefix = body.strip().split(" ", 1)[0].upper()
    return CATEGORY_PREFIX_MAP.get(prefix)


def fit_corpus_stats(all_features: Dict[str, Dict], all_params: Dict[str, Dict]) -> Dict:
    stems = sorted(all_features.keys())
    X = np.array([[all_features[s].get(f, 0.0) for f in FEATURES] for s in stems])
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std

    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    w = Vt[0].copy()
    bright_idx = FEATURES.index("brightness")
    if w[bright_idx] < 0:
        w = -w
    pc1_explained = float((S[0] ** 2) / np.sum(S ** 2))

    sti = Z @ w
    sti_pct = tuple(float(np.percentile(sti, p)) for p in (20, 35, 65, 80))

    resid_raw = Z - np.outer(sti, w)
    resid_std = resid_raw.std(axis=0)
    resid_std[resid_std == 0] = 1.0

    roughness_vals = X[:, FEATURES.index("roughness")]
    roughness_pct = tuple(float(np.percentile(roughness_vals, p)) for p in (20, 35, 65, 80))

    drive_vals = np.array([all_params[s].get("filterdrive", 0.0) for s in stems])
    sub_vals   = np.array([all_params[s].get("osc3volume", 0.0) for s in stems])
    drive_pct = tuple(float(np.percentile(drive_vals, p)) for p in (20, 35, 65, 80))
    sub_pct   = tuple(float(np.percentile(sub_vals, p)) for p in (20, 35, 65, 80))

    return {
        "mean": mean.tolist(), "std": std.tolist(), "pc1_loadings": w.tolist(),
        "pc1_explained_variance": pc1_explained,
        "resid_std": resid_std.tolist(),
        "sti_percentiles": sti_pct,
        "roughness_percentiles": roughness_pct,
        "drive_percentiles": drive_pct,
        "sub_osc_percentiles": sub_pct,
    }


def compute_sti_and_residuals(feats: Dict, stats: Dict) -> Tuple[float, Dict[str, float]]:
    mean = np.array(stats["mean"]); std = np.array(stats["std"]); w = np.array(stats["pc1_loadings"])
    resid_std = np.array(stats["resid_std"])
    z = np.array([(feats.get(f, 0.0) - mean[i]) / std[i] for i, f in enumerate(FEATURES)])
    sti = float(z @ w)
    resid_raw = z - sti * w
    resid = {f: float(resid_raw[i] / resid_std[i]) for i, f in enumerate(FEATURES)}
    return sti, resid


def pick_brightness(feats: Dict, stats: Dict) -> str:
    sti, resid = compute_sti_and_residuals(feats, stats)
    p20, p35, p65, p80 = stats["sti_percentiles"]
    if sti > p80:
        return "sharp" if resid["sharpness"] > RESID_MARGIN else "bright"
    if sti > p65:
        return "bright"
    if sti < p20:
        return "muffled" if resid["warmth"] < -RESID_MARGIN else "dark"
    if sti < p35:
        return "dark"
    return "warm" if resid["warmth"] > RESID_MARGIN else "mellow"


def pick_loudness(feats: Dict, params: Dict, stats: Dict) -> Optional[str]:
    sti, resid = compute_sti_and_residuals(feats, stats)
    p20, p35, p65, p80 = stats["sti_percentiles"]
    drive = params.get("filterdrive", 0.0)
    sub = params.get("osc3volume", 0.0)
    drive_p80 = stats["drive_percentiles"][3]
    sub_p65 = stats["sub_osc_percentiles"][2]

    if sti > p80:
        return "aggressive" if (resid["roughness"] > RESID_MARGIN or drive > drive_p80) else "loud"
    if sti > p65:
        return "loud"
    if sti < p20:
        return "powerful" if sub > sub_p65 else "soft"
    if sti < p35:
        return "gentle" if resid["warmth"] > RESID_MARGIN else "soft"
    return None


def pick_weight(feats: Dict, params: Dict, stats: Dict) -> str:
    sti, resid = compute_sti_and_residuals(feats, stats)
    p20, p35, p65, p80 = stats["sti_percentiles"]
    sub = params.get("osc3volume", 0.0)
    sub_p65 = stats["sub_osc_percentiles"][2]

    if sti < p20:
        return "heavy" if (resid["boominess"] > RESID_MARGIN or sub > sub_p65) else "deep"
    if sti < p35:
        return "deep" if resid["depth"] > RESID_MARGIN else "full"
    if sti > p80:
        return "thin" if resid["boominess"] < -RESID_MARGIN else "light"
    if sti > p65:
        return "light"
    return "full"


def pick_texture(feats: Dict, stats: Dict) -> Optional[str]:
    sti, resid = compute_sti_and_residuals(feats, stats)
    r20, r35, r65, r80 = stats["roughness_percentiles"]
    r = feats.get("roughness", 0.0)
    sti_p35, sti_p65 = stats["sti_percentiles"][1], stats["sti_percentiles"][2]

    if r > r80:
        return "harsh" if resid["sharpness"] > RESID_MARGIN else "rough"
    if r > r65:
        return "rough"
    if r < r20:
        return "smooth"
    if r < r35:
        return "clean"
    if sti > sti_p65:
        return "metallic" if resid["hardness"] > RESID_MARGIN else "buzzy"
    if sti < sti_p35:
        return "lush" if resid["warmth"] > RESID_MARGIN else "hollow"
    return None


def time_shape_from_params(p: Dict) -> Optional[str]:
    A = p.get("ampattack", 0); D = p.get("ampdecay", 0)
    S = p.get("ampsustain", 0); R = p.get("amprelease", 0)

    lfo1_amt = p.get("lfo1amount", 0); lfo1_rate = p.get("lfo1rate", 0)
    lfo1_d   = _lfo_dest(p.get("lfo1destination", 0))
    lfo2_amt = p.get("lfo2amount", 0); lfo2_rate = p.get("lfo2rate", 0)
    lfo2_d   = _lfo_dest(p.get("lfo2destination", 0))
    delay_sync = p.get("delaysync", 0); delay_fb = p.get("delayfeedback", 0)
    delay_wet = p.get("delaywet", 0)
    fad_amt  = p.get("freeadamount", 0); fad_dest = p.get("freeaddestination", 0)
    fad_decay = p.get("freeaddecay", 0)
    filter_a = p.get("filterattack", 0); filter_d = p.get("filterdecay", 0)
    filter_s = p.get("filtersustain", 1); filter_c = p.get("filtercontour", 0.5)

    pulsing = (
        (_attivo(lfo1_amt) and lfo1_d == "cutoff" and lfo1_rate > 0.4) or
        (_attivo(lfo2_amt) and lfo2_d == "cutoff" and lfo2_rate > 0.4) or
        (delay_sync > 0.5 and delay_fb > 0.35 and delay_wet > 0.1)
    )
    evolving = (
        (_attivo(lfo1_amt) and lfo1_rate < 0.3 and lfo1_d == "cutoff") or
        (_attivo(lfo2_amt) and lfo2_rate < 0.3 and lfo2_d == "cutoff") or
        (_attivo(fad_amt) and 0.0 < fad_dest < 0.65 and fad_decay > 0.05)
    )
    filter_pluck = (
        S > 0.7 and filter_a < 0.15 and (
            (filter_s < 0.2 and filter_d < 0.5 and filter_c > 0.3) or
            (filter_d < 0.2 and filter_c > 0.7)
        )
    )
    fading = S < 0.2 and D > 0.4 and R > 0.3

    if pulsing:                return "pulsing"
    if evolving:               return "evolving"
    if A > 0.4:                return "slow"
    if filter_pluck:           return "short"
    if R > 0.6 or S > 0.8:    return "long"
    if fading:                 return "fading"
    if R < 0.25 and S < 0.2:  return "short"
    return None


def space_from_params(p: Dict) -> Optional[str]:
    rev_wet = p.get("reverbwet", 0); rev_decay = p.get("reverbdecay", 0)
    dly_wet = p.get("delaywet", 0); dly_time = p.get("delaytime", 0)
    dly_sync = p.get("delaysync", 0)
    chorus_on = (p.get("chorus1enable", 0) > 0.5 or
                 p.get("chorus2enable", 0) > 0.5)

    cavernous = rev_wet > 0.4 and rev_decay > 0.65
    echoey = dly_wet > 0.25 and (dly_sync > 0.5 or dly_time > 0.05)
    spacious = rev_wet > 0.25
    dry = rev_wet <= 0.05 and dly_wet <= 0.05 and not chorus_on

    if cavernous:  return "cavernous"
    if echoey:     return "echoey"
    if spacious:   return "spacious"
    if chorus_on:  return "chorused"
    if dry:        return "dry"
    return None


def compose_sentence(choices: Dict) -> str:
    category = choices.get("category") or "synthesizer"
    cat_words = set(category.lower().split())
    descriptor_order = ["brightness", "texture", "loudness", "weight"]
    descriptors = [
        choices[k] for k in descriptor_order
        if choices.get(k) and choices[k].lower() not in cat_words
    ]
    def a_or_an(s): return "an" if s[:1].lower() in "aeiou" else "a"
    def join_and(items):
        if not items: return ""
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    subject = join_and(descriptors)
    sentence = (f"The sound of {a_or_an(subject)} {subject} {category}"
                if subject else f"The sound of {a_or_an(category)} {category}")
    if choices.get("timeShape"):
        sentence += f", {choices['timeShape']}"
    return sentence


def _init_ac_worker():
    import timbral_models


def _extract_one(args):
    audio_path, stem = args
    import timbral_models
    result = timbral_models.timbral_extractor(str(audio_path), verbose=False)
    return stem, {f: float(result[f]) for f in FEATURES}


if __name__ == "__main__":
    _in_pool  = Path(__file__).resolve().parents[1] / "data" / "Dataset_TAL_48kHz" / "in_pool"
    audio_dir = _in_pool / "audio"
    json_dir  = _in_pool / "json"
    out_dir   = _in_pool / "captions"
    out_dir.mkdir(parents=True, exist_ok=True)
    import sys
    STATS_DREF = _in_pool.parent / "not_in_pool" / "_corpus_stats.json"
    usa_stats_dref = "--local-stats" not in sys.argv and STATS_DREF.exists()
    riusa = "--reuse-features" in sys.argv

    all_wav = sorted(audio_dir.glob("*.wav"))
    if not all_wav:
        print(f"❌ No .wav files found in {audio_dir}")
        raise SystemExit(1)

    print(f"Found {len(all_wav)} audio files.\n")

    print("Step 1/3: extracting AudioCommons features...")
    max_workers = max(1, mp.cpu_count() - 1)
    tasks = [(wav, wav.stem) for wav in all_wav]
    all_features: Dict[str, Dict] = {}

    if riusa:
        for wav, stem in list(tasks):
            cj = out_dir / f"{stem}.json"
            if cj.exists():
                all_features[stem] = json.load(open(cj))["features"]
        tasks = [t for t in tasks if t[1] not in all_features]
        print(f"  features reused: {len(all_features)}, to extract: {len(tasks)}")
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_ac_worker) as ex:
        futs = {ex.submit(_extract_one, t): t for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(tasks), desc="Extracting AC",
                        unit="file", dynamic_ncols=True):
            try:
                stem, feats = fut.result()
                all_features[stem] = feats
            except Exception as e:
                _, wav_stem = futs[fut]
                print(f"⚠️  {wav_stem}: {e}")
                all_features[wav_stem] = {f: 0.0 for f in FEATURES}

    print("\nStep 2/3: fitting corpus statistics (z-scores, PCA, percentiles)...")
    all_params: Dict[str, Dict] = {}
    for json_file in json_dir.glob("*.json"):
        with open(json_file) as fh:
            data = json.load(fh)
        all_params[json_file.stem] = data.get("parameters", data)

    if usa_stats_dref:
        stats = json.load(open(STATS_DREF))
        print(f"  corpus statistics loaded from D_ref: {STATS_DREF}")
    else:
        stats = fit_corpus_stats(all_features, all_params)
    print(f"  PC1 (Spectral Tilt Index) explains {stats['pc1_explained_variance']:.1%} of the variance")
    print(f"  STI percentiles (20/35/65/80): "
          f"{', '.join(f'{v:.2f}' for v in stats['sti_percentiles'])}")
    print(f"  roughness percentiles: "
          f"{', '.join(f'{v:.1f}' for v in stats['roughness_percentiles'])}")

    if not usa_stats_dref:
        with open(out_dir.parent / "_corpus_stats.json", "w") as fh:
            json.dump(stats, fh, indent=2)

    print("\nStep 3/3: captioning...")
    count = 0
    for json_file in tqdm(sorted(json_dir.glob("*.json")), desc="Captioning",
                          unit="preset", dynamic_ncols=True):
        with open(json_file) as fh:
            data = json.load(fh)

        stem   = json_file.stem
        params = data.get("parameters", data)
        feats  = all_features.get(stem)
        if feats is None:
            print(f"⚠️  No features for {stem}, skipped.")
            continue

        choices = {
            "category":   extract_category(stem),
            "brightness": pick_brightness(feats, stats),
            "loudness":   pick_loudness(feats, params, stats),
            "weight":     pick_weight(feats, params, stats),
            "texture":    pick_texture(feats, stats),
            "timeShape":  time_shape_from_params(params),
            "space":      space_from_params(params),
        }
        caption = compose_sentence(choices)

        out = {
            "name":        stem,
            "preset_name": data.get("preset_name", stem),
            "caption":     caption,
            "choices":     choices,
            "features":    feats,
        }
        with open(out_dir / json_file.name, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        count += 1

    print(f"\n✓ {count} presets captioned → {out_dir}")
