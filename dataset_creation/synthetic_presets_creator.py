"""Synthetic preset generator for TAL-NoiseMaker.

Samples random presets from per-parameter priors, applies the logical
constraints between parameters and renders each preset with DawDreamer.
"""

import dawdreamer as daw
import numpy as np
import json
import os
import glob
import soundfile as sf
import multiprocessing
import time
from pathlib import Path

VST_PATH     = "/Library/Audio/Plug-Ins/VST3/TAL-NoiseMaker.vst3"
OUTPUT_DIR   = "./test_synthetic_presets"
LOCKED_PARAMS_FILE = "locked_parameters.json"

AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio_wav")
JSON_DIR  = os.path.join(OUTPUT_DIR, "params_json")

SAMPLE_RATE   = 48000
DURATION      = 4.0
NOTE_DURATION = 2.0
MIDI_NOTE     = 60
VELOCITY      = 100
MASTER_VOLUME = 0.8

TARGET_PRESETS    = 1000
RMS_THRESHOLD_MIN = 0.01
RMS_THRESHOLD_MAX = 0.85


_EPS = 0.05

CATEGORICAL_CHOICES = {
    "lfo1destination": np.linspace(0.0, 1.0, 8).tolist(),
    "lfo1waveform":    np.linspace(0.0, 1.0, 6).tolist(),
    "lfo1keytrigger":  [0.0, 1.0],
    "lfo2destination": np.linspace(0.0, 1.0, 8).tolist(),
    "lfo2waveform":    np.linspace(0.0, 1.0, 6).tolist(),
    "lfo2keytrigger":  [0.0, 1.0],
    "osc1waveform": [0.0, 0.5, 1.0],
    "osc1tune":     np.linspace(0.0, 1.0, 49).tolist(),
    "osc2waveform": np.linspace(0.0, 1.0, 5).tolist(),
    "osc2tune":     np.linspace(0.0, 1.0, 49).tolist(),
    "filtertype":        np.linspace(0.0, 1.0, 12).tolist(),
    "freeaddestination": np.linspace(0.0, 1.0, 6).tolist(),
    "oscsync":      [0.0, 1.0],
    "chorus1enable":[0.0, 1.0],
    "chorus2enable":[0.0, 1.0],
    "delaysync":    [0.0, 1.0],
    "delayx2l":     [0.0, 1.0],
    "delayx2r":     [0.0, 1.0],
}


BIPOLAR_NEUTRAL = {
    "lfo1amount",
    "lfo2amount",
    "osc1finetune",
    "osc2finetune",
    "osc1pw",
    "filtercontour",
    "freeadamount",
    "masterdetune",
}
P_BIPOLAR_NEUTRAL = 0.25

P_REVERB_OFF = 0.35
P_DELAY_OFF  = 0.35
P_OSC_OFF    = 0.10

def sample_param(p_name):
    if p_name in CATEGORICAL_CHOICES:
        return round(float(np.random.choice(CATEGORICAL_CHOICES[p_name])), 7)

    if p_name in BIPOLAR_NEUTRAL and np.random.random() < P_BIPOLAR_NEUTRAL:
        return 0.5

    if p_name == "reverbwet" and np.random.random() < P_REVERB_OFF:
        return 0.0
    if p_name == "delaywet" and np.random.random() < P_DELAY_OFF:
        return 0.0
    if p_name in ("osc1volume", "osc2volume") and np.random.random() < P_OSC_OFF:
        return 0.0

    return round(float(np.random.uniform(0.0, 1.0)), 7)


def enforce_audio_logic(params):
    def g(key, default=0.0):
        return float(params.get(key, default))

    if g("lfo1destination") == 0.0:
        for p in ["lfo1waveform", "lfo1rate", "lfo1phase", "lfo1keytrigger"]:
            if p in params: params[p] = 0.0
        if "lfo1amount" in params: params["lfo1amount"] = 0.5

    if abs(g("lfo1amount") - 0.5) < _EPS:
        for p in ["lfo1waveform", "lfo1rate", "lfo1phase", "lfo1keytrigger"]:
            if p in params: params[p] = 0.0
        if "lfo1destination" in params: params["lfo1destination"] = 0.0

    if g("lfo2destination") == 0.0:
        for p in ["lfo2waveform", "lfo2rate", "lfo2phase", "lfo2keytrigger"]:
            if p in params: params[p] = 0.0
        if "lfo2amount" in params: params["lfo2amount"] = 0.5

    if abs(g("lfo2amount") - 0.5) < _EPS:
        for p in ["lfo2waveform", "lfo2rate", "lfo2phase", "lfo2keytrigger"]:
            if p in params: params[p] = 0.0
        if "lfo2destination" in params: params["lfo2destination"] = 0.0

    if g("osc1waveform") != 0.5:
        if "osc1pw" in params: params["osc1pw"] = 0.5

    if g("osc2volume") < _EPS and g("osc2fm") < _EPS:
        if "oscsync" in params: params["oscsync"] = 0.0

    if g("osc1volume") < _EPS and g("osc2volume") < _EPS:
        if "ringmodulation" in params: params["ringmodulation"] = 0.0
        if "oscsync" in params:        params["oscsync"]        = 0.0

    if g("filtertype") == 1.0:
        for p in ["filtercutoff", "filterresonance",
                  "filterattack", "filterdecay", "filtersustain", "filterrelease",
                  "filterdrive"]:
            if p in params: params[p] = 0.0
        if "filtercontour" in params: params["filtercontour"] = 0.5

    if abs(g("filtercontour") - 0.5) < _EPS:
        for p in ["filterattack", "filterdecay", "filtersustain", "filterrelease"]:
            if p in params: params[p] = 0.0

    if g("freeaddestination") == 0.0:
        for p in ["freeadattack", "freeaddecay"]:
            if p in params: params[p] = 0.0
        if "freeadamount" in params: params["freeadamount"] = 0.5

    if abs(g("freeadamount") - 0.5) < _EPS:
        for p in ["freeadattack", "freeaddecay"]:
            if p in params: params[p] = 0.0
        if "freeaddestination" in params: params["freeaddestination"] = 0.0

    if g("reverbwet") < _EPS:
        for p in ["reverbdecay", "reverbpredelay", "reverbhighcut", "reverblowcut"]:
            if p in params: params[p] = 0.0

    if g("delaywet") < _EPS:
        for p in ["delaytime", "delaysync", "delayhighshelf", "delaylowshelf",
                  "delayfeedback", "delayx2l", "delayx2r"]:
            if p in params: params[p] = 0.0

    return params

worker_engine        = None
worker_synth         = None
worker_active_params = None
worker_static_params = None

def normalize_name(text):
    return text.lower().replace("_", "").replace(" ", "")

def init_worker(whitelist):
    global worker_engine, worker_synth, worker_active_params, worker_static_params
    worker_engine = daw.RenderEngine(SAMPLE_RATE, 512)
    worker_synth  = worker_engine.make_plugin_processor("TAL-NoiseMaker", VST_PATH)

    vst_params    = worker_synth.get_plugin_parameters_description()
    name_to_index = {normalize_name(p["name"]): p["index"] for p in vst_params}

    worker_active_params = []
    for p_name in whitelist:
        if p_name in name_to_index:
            worker_active_params.append({"index": name_to_index[p_name], "json_name": p_name})

    worker_static_params = []
    if "volume" in name_to_index:
        worker_static_params.append({"index": name_to_index["volume"], "value": MASTER_VOLUME})

def process_single_preset(file_index):
    global worker_engine, worker_synth, worker_active_params, worker_static_params

    np.random.seed((os.getpid() * int(time.time())) % 123456789 + file_index)

    for sp in worker_static_params:
        worker_synth.set_parameter(sp["index"], sp["value"])

    current_config = {
        p["json_name"]: sample_param(p["json_name"])
        for p in worker_active_params
    }

    current_config = enforce_audio_logic(current_config)

    for p in worker_active_params:
        worker_synth.set_parameter(p["index"], current_config[p["json_name"]])

    worker_synth.clear_midi()
    worker_synth.add_midi_note(MIDI_NOTE, VELOCITY, 0, NOTE_DURATION)
    worker_engine.load_graph([(worker_synth, [])])
    worker_engine.render(DURATION)
    audio = worker_engine.get_audio()

    rms = np.sqrt(np.mean(audio ** 2))
    if not (RMS_THRESHOLD_MIN <= rms <= RMS_THRESHOLD_MAX):
        return False

    file_id  = f"{file_index:05d}"
    wav_path = os.path.join(AUDIO_DIR, f"synth_{file_id}.wav")
    sf.write(wav_path, audio.T, SAMPLE_RATE)

    json_path = os.path.join(JSON_DIR, f"synth_{file_id}.json")
    with open(json_path, "w") as f:
        json.dump({"synth_parameters": current_config}, f, indent=4)

    return True

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)
    Path(JSON_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Loading parameters from '{LOCKED_PARAMS_FILE}'...")
    with open(LOCKED_PARAMS_FILE, "r") as f:
        locked_params = json.load(f)

    WHITELIST = []
    for category in locked_params.values():
        WHITELIST.extend([normalize_name(p) for p in category])

    existing_files = glob.glob(os.path.join(AUDIO_DIR, "synth_*.wav"))
    start_index = (
        max(int(os.path.splitext(os.path.basename(f))[0].split("_")[1])
            for f in existing_files) + 1
        if existing_files else 0
    )

    num_cores = multiprocessing.cpu_count()
    print(f"\nGenerating presets on {num_cores} cores in parallel")
    print(f"Starting index: {start_index:05d}")
    print(f"WAV -> {AUDIO_DIR}")
    print(f"JSON -> {JSON_DIR}")
    print(f"\nSampling priors:")
    print(f"  P_BIPOLAR_NEUTRAL = {P_BIPOLAR_NEUTRAL}  ({', '.join(sorted(BIPOLAR_NEUTRAL))})")
    print(f"  P_REVERB_OFF      = {P_REVERB_OFF}")
    print(f"  P_DELAY_OFF       = {P_DELAY_OFF}")
    print(f"  P_OSC_OFF         = {P_OSC_OFF}")

    start_time    = time.time()
    success_count = start_index
    current_attempt_index = start_index

    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(WHITELIST,)
    ) as pool:
        while success_count < TARGET_PRESETS:
            batch_size       = num_cores * 4
            indices_to_process = list(range(current_attempt_index,
                                            current_attempt_index + batch_size))
            results = pool.map(process_single_preset, indices_to_process)

            for was_successful in results:
                current_attempt_index += 1
                if was_successful:
                    success_count += 1
                    if success_count % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"Progress: {success_count}/{TARGET_PRESETS} "
                              f"({elapsed:.1f}s)")
                    if success_count >= TARGET_PRESETS:
                        break

    elapsed_total = time.time() - start_time
    print(f"\nGeneration completed in {elapsed_total:.1f} s")
