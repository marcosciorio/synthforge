"""Dataset builder for the handcrafted TAL-NoiseMaker presets.

Reads presets in .noisemakerpreset, .fxp or .vstpreset format, renders them
with DawDreamer and saves audio and parameters in a common format.
"""

import dawdreamer as daw
import json
import os
import glob
import numpy as np
import soundfile as sf
import xml.etree.ElementTree as ET
from pathlib import Path


VST_PATH           = "/Library/Audio/Plug-Ins/VST3/TAL-NoiseMaker.vst3"
INPUT_DIR          = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Presets_TAL_originali")
OUTPUT_DIR         = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Dataset_TAL_48kHz")
LOCKED_PARAMS_FILE = "locked_parameters.json"

SAMPLE_RATE   = 48000
DURATION      = 4.0
NOTE_DURATION = 2.0
MIDI_NOTE     = 60
VELOCITY      = 100
MASTER_VOLUME = 0.8

SILENCE_THRESHOLD = 1e-4
PEAK_TARGET_DB    = -3.0

OUTPUT_AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
OUTPUT_JSON_DIR  = os.path.join(OUTPUT_DIR, "json")
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR,  exist_ok=True)

ALIAS_MAP = {
    "cutoff":               "filtercutoff",
    "resonance":            "filterresonance",
    "volume":               "mastervolume",
    "highpass":             "masterhighpass",
    "detune":               "masterdetune",
    "oscmastertune":        "oscmasterfinetune",
    "envelopeeditordest1":  "envelopedestination",
    "envelopeeditorspeed":  "envelopespeed",
    "envelopeeditoramount": "envelopeamount",
    "delayfactorl":         "delayx2l",
    "delayfactorr":         "delayx2r",
    "velocitycutoff":       "velocityfilter",
}

IGNORE_XML_PARAMS = {"tab1open", "tab2open", "tab3open", "programname", "version"}


def normalizza_nome(testo: str) -> str:
    return testo.lower().replace("_", "").replace(" ", "")


def normalizza_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < SILENCE_THRESHOLD:
        return audio
    return audio * (10 ** (target_db / 20.0) / peak)


def carica_noisemakerpreset(file_path: str, synth, name_to_index: dict) -> dict | None:
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        program = root.find(".//program")
        if program is None:
            return None
        attribs = program.attrib
    except Exception as e:
        print(f"  ⚠️  XML error: {e}")
        return None

    matched = aliased = fuzzy = 0
    unmatched = []

    for name, value in attribs.items():
        nome_xml = normalizza_nome(name)

        if nome_xml in IGNORE_XML_PARAMS:
            continue

        try:
            val = max(0.0, min(1.0, float(value)))
        except (ValueError, TypeError):
            continue

        if nome_xml in name_to_index:
            synth.set_parameter(name_to_index[nome_xml], val)
            matched += 1
            continue

        nome_vst = ALIAS_MAP.get(nome_xml)
        if nome_vst and nome_vst in name_to_index:
            synth.set_parameter(name_to_index[nome_vst], val)
            aliased += 1
            continue

        found = False
        for vst_name, vst_idx in name_to_index.items():
            if (nome_xml in vst_name or vst_name in nome_xml) and \
               len(nome_xml) >= 4 and len(vst_name) >= 4:
                synth.set_parameter(vst_idx, val)
                fuzzy += 1
                found = True
                break
        if not found:
            unmatched.append(nome_xml)

    return {"attribs": attribs, "stats": {"matched": matched, "aliased": aliased,
                                           "fuzzy": fuzzy, "unmatched": unmatched}}


def carica_fxp(file_path: str, synth) -> bool:
    try:
        ok = synth.load_preset(file_path)
        return bool(ok)
    except Exception as e:
        print(f"  ⚠️  load_preset error: {e}")
        return False


def carica_vstpreset(file_path: str, synth) -> bool:
    try:
        ok = synth.load_vst3_preset(file_path)
        return bool(ok)
    except AttributeError:
        try:
            ok = synth.load_preset(file_path)
            return bool(ok)
        except Exception as e:
            print(f"  ⚠️  load_preset error (VST3 fallback): {e}")
            return False
    except Exception as e:
        print(f"  ⚠️  load_vst3_preset error: {e}")
        return False


def estrai_whitelist_da_xml(attribs: dict, locked_params: list) -> dict:
    result = {}
    for name, value in attribs.items():
        nome_xml = normalizza_nome(name)
        nome_xml = ALIAS_MAP.get(nome_xml, nome_xml)
        if nome_xml in locked_params:
            try:
                result[nome_xml] = float(value)
            except (ValueError, TypeError):
                pass
    return result


def estrai_whitelist_da_synth(synth, locked_params: list,
                               vst_idx_to_name: dict) -> dict:
    result = {}
    for idx, val in synth.get_patch():
        name_norm = normalizza_nome(vst_idx_to_name.get(idx, ""))
        if name_norm in locked_params:
            result[name_norm] = float(val)
    return result


def main():
    print("=" * 70)
    print("TAL-NoiseMaker dataset builder")
    print("Formats: .noisemakerpreset | .fxp | .vstpreset")
    print("=" * 70)

    print(f"\n[1/4] Loading parameter whitelist: '{LOCKED_PARAMS_FILE}'")
    if not os.path.exists(LOCKED_PARAMS_FILE):
        print(f"❌ File not found: {LOCKED_PARAMS_FILE}")
        return

    with open(LOCKED_PARAMS_FILE, "r") as f:
        raw = json.load(f)

    LOCKED_PARAMETERS = []
    if isinstance(raw, dict):
        for cat, params in raw.items():
            LOCKED_PARAMETERS.extend(normalizza_nome(p) for p in params)
        print(f"   ✓ {len(raw)} categories, {len(LOCKED_PARAMETERS)} parameters")
    elif isinstance(raw, list):
        LOCKED_PARAMETERS = [normalizza_nome(p) for p in raw]
        print(f"   ✓ {len(LOCKED_PARAMETERS)} parameters")

    print(f"\n[2/4] Initialising DawDreamer...")
    engine = daw.RenderEngine(SAMPLE_RATE, 512)
    synth  = engine.make_plugin_processor("TAL-NoiseMaker", VST_PATH)

    vst_params      = synth.get_plugin_parameters_description()
    name_to_index   = {normalizza_nome(p["name"]): p["index"] for p in vst_params}
    idx_to_name     = {p["index"]: p["name"] for p in vst_params}
    print(f"   ✓ {Path(VST_PATH).name} ({len(name_to_index)} VST parameters)")

    print(f"\n[3/4] Searching presets in: {INPUT_DIR}")
    extensions = ["*.noisemakerpreset", "*.fxp", "*.vstpreset"]
    tal_files  = []
    for ext in extensions:
        found = glob.glob(os.path.join(INPUT_DIR, "**", ext), recursive=True)
        tal_files.extend(found)
    tal_files.sort()

    if not tal_files:
        print("❌ No presets found.")
        return

    counts = {ext.lstrip("*"): sum(1 for f in tal_files if f.endswith(ext.lstrip("*")))
              for ext in extensions}
    print(f"   ✓ {len(tal_files)} presets in total:")
    for ext, n in counts.items():
        if n > 0:
            print(f"     {ext:25s} {n}")

    print(f"\n[4/4] Processing...\n")

    n_ok = n_silent = n_xml_err = n_load_err = n_err = 0
    rms_vals = []
    unmatched_global = {}
    fmt_counts = {".noisemakerpreset": 0, ".fxp": 0, ".vstpreset": 0}

    for i, file_path in enumerate(tal_files):
        preset_name = Path(file_path).stem
        ext         = Path(file_path).suffix.lower()
        progress    = f"[{i+1:04d}/{len(tal_files)}]"

        try:
            ai_vector = {}
            match_stats = {}

            if ext == ".noisemakerpreset":
                result = carica_noisemakerpreset(file_path, synth, name_to_index)
                if result is None:
                    print(f"{progress} ❌ XML error: {preset_name}")
                    n_xml_err += 1
                    continue
                match_stats = result["stats"]
                for u in match_stats["unmatched"]:
                    if u not in IGNORE_XML_PARAMS:
                        unmatched_global[u] = unmatched_global.get(u, 0) + 1
                ai_vector = estrai_whitelist_da_xml(result["attribs"], LOCKED_PARAMETERS)

            elif ext == ".fxp":
                if not carica_fxp(file_path, synth):
                    print(f"{progress} ❌ FXP load error: {preset_name}")
                    n_load_err += 1
                    continue
                ai_vector = estrai_whitelist_da_synth(synth, LOCKED_PARAMETERS, idx_to_name)

            elif ext == ".vstpreset":
                if not carica_vstpreset(file_path, synth):
                    print(f"{progress} ❌ VST3 load error: {preset_name}")
                    n_load_err += 1
                    continue
                ai_vector = estrai_whitelist_da_synth(synth, LOCKED_PARAMETERS, idx_to_name)

            else:
                continue

            if "mastervolume" in name_to_index:
                synth.set_parameter(name_to_index["mastervolume"], MASTER_VOLUME)
            key_mv = "mastervolume" if "mastervolume" in LOCKED_PARAMETERS else \
                     "volume"       if "volume"       in LOCKED_PARAMETERS else None
            if key_mv:
                ai_vector[key_mv] = MASTER_VOLUME

            synth.clear_midi()
            synth.add_midi_note(MIDI_NOTE, VELOCITY, 0, NOTE_DURATION)
            engine.load_graph([(synth, [])])
            engine.render(DURATION)
            audio = engine.get_audio()

            if np.max(np.abs(audio)) < SILENCE_THRESHOLD:
                print(f"{progress} ⚠️  Silent: {preset_name}")
                n_silent += 1
                continue

            audio_norm = normalizza_audio(audio, target_db=PEAK_TARGET_DB)
            rms = float(np.sqrt(np.mean(audio_norm ** 2)))
            rms_vals.append(rms)

            base = f"{i:04d}_{preset_name}"
            sf.write(os.path.join(OUTPUT_AUDIO_DIR, f"{base}.wav"),
                     audio_norm.T, SAMPLE_RATE)

            with open(os.path.join(OUTPUT_JSON_DIR, f"{base}.json"), "w") as f:
                json.dump({
                    "preset_name":   preset_name,
                    "preset_format": ext.lstrip("."),
                    "parameters":    ai_vector,
                    "audio_info": {
                        "rms":         rms,
                        "peak_db":     PEAK_TARGET_DB,
                        "normalized":  True,
                        "sample_rate": SAMPLE_RATE,
                        "duration_s":  DURATION,
                        "midi_note":   MIDI_NOTE,
                    }
                }, f, indent=2)

            n_ok += 1
            fmt_counts[ext] += 1
            n_applied = sum(match_stats.get(k, 0) for k in ("matched", "aliased", "fuzzy")) \
                        if match_stats else "n/a"
            print(f"{progress} ✅ OK  RMS={rms:.4f}  [{ext.lstrip('.')}]  {preset_name}")

        except Exception as e:
            print(f"{progress} ❌ Error: {e}")
            n_err += 1

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  ✅ Saved:            {n_ok}")
    print(f"     .noisemakerpreset  {fmt_counts['.noisemakerpreset']}")
    print(f"     .fxp               {fmt_counts['.fxp']}")
    print(f"     .vstpreset         {fmt_counts['.vstpreset']}")
    print(f"  ⚠️  Silent:           {n_silent}")
    print(f"  ❌ XML errors:       {n_xml_err}")
    print(f"  ❌ Load errors:      {n_load_err}")
    print(f"  ❌ Render errors:    {n_err}")

    if rms_vals:
        print(f"\n  RMS  min={min(rms_vals):.4f}  "
              f"max={max(rms_vals):.4f}  mean={np.mean(rms_vals):.4f}")

    if unmatched_global:
        top = sorted(unmatched_global.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n  Most frequent unrecognised XML parameters:")
        for name, count in top:
            print(f"    {name:35s} ({count} presets)")
        print("  → Add them to ALIAS_MAP if they affect the sound.")

    with open(os.path.join(OUTPUT_DIR, "dataset_stats.json"), "w") as f:
        json.dump({
            "total": len(tal_files), "saved": n_ok,
            "by_format": fmt_counts,
            "silent": n_silent, "xml_errors": n_xml_err,
            "load_errors": n_load_err, "render_errors": n_err,
            "rms": {"min":  float(min(rms_vals)) if rms_vals else 0,
                    "max":  float(max(rms_vals)) if rms_vals else 0,
                    "mean": float(np.mean(rms_vals)) if rms_vals else 0},
            "unmatched_params": unmatched_global,
            "config": {"sample_rate": SAMPLE_RATE, "duration": DURATION,
                       "midi_note": MIDI_NOTE, "master_volume": MASTER_VOLUME,
                       "peak_target_db": PEAK_TARGET_DB}
        }, f, indent=2)

    print(f"\n  📂 Audio:  {OUTPUT_AUDIO_DIR}")
    print(f"  📂 JSON:   {OUTPUT_JSON_DIR}")
    print(f"  📊 Stats:  {os.path.join(OUTPUT_DIR, 'dataset_stats.json')}")


if __name__ == "__main__":
    main()