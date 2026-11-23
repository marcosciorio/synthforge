#!/usr/bin/env python3
"""Generate a TAL-NoiseMaker preset with the MIDI map of the 63 parameters
already set, so that the MIDI learn step is not needed.
"""

import os
import sys
import xml.etree.ElementTree as ET

ORDER = [
    "programname", "unknown", "volume",
    "filtertype", "cutoff", "resonance", "keyfollow", "filtercontour",
    "filterattack", "filterdecay", "filtersustain", "filterrelease",
    "ampattack", "ampdecay", "ampsustain", "amprelease",
    "osc1volume", "osc2volume", "osc3volume", "oscmastertune",
    "osc1tune", "osc2tune", "osc1finetune", "osc2finetune",
    "osc1waveform", "osc2waveform", "oscsync",
    "lfo1waveform", "lfo2waveform", "lfo1rate", "lfo2rate",
    "lfo1amount", "lfo2amount", "lfo1destination", "lfo2destination",
    "lfo1Phase", "lfo2Phase", "osc2fm", "osc2phase", "osc1pw", "osc1phase",
    "transpose", "freeadattack", "freeaddecay", "freeadamount", "freeaddestination",
    "lfo1sync", "lfo1keytrigger", "lfo2sync", "lfo2keytrigger",
    "portamento", "portamentomode", "voices",
    "velocityvolume", "velocitycontour", "velocityfilter",
    "pitchwheelcutoff", "pitchwheelpitch",
    "ringmodulation", "chorus1enable", "chorus2enable",
    "reverbwet", "reverbdecay", "reverbpredelay", "reverbhighcut", "reverblowcut",
    "oscbitcrusher", "highpass", "detune", "vintagenoise",
    "panic", "midilearn",
    "envelopeeditordest1", "envelopeeditorspeed", "envelopeeditoramount",
    "envelopeOneShot", "envelopefixtempo", "envelopereset",
    "filterdrive",
    "delaywet", "delaytime", "delaysync", "delayfactorl", "delayfactorR",
    "delayhighshelf", "delaylowshelf", "delayfeedback", "midiclear", "midilock",
]
assert len(ORDER) == 89

RENAME = {
    "filtercutoff": "cutoff", "filterresonance": "resonance",
    "masterdetune": "detune", "masterhighpass": "highpass",
    "lfo1phase": "lfo1Phase", "lfo2phase": "lfo2Phase",
    "delayx2l": "delayfactorl", "delayx2r": "delayfactorR",
}

GROUPS = {
    "OSC 1": ["osc1waveform", "osc1tune", "osc1finetune", "osc1volume", "osc1pw", "osc1phase"],
    "OSC 2": ["osc2waveform", "osc2tune", "osc2finetune", "osc2volume", "osc2fm", "osc2phase"],
    "MIX": ["osc3volume", "ringmodulation", "oscsync", "oscbitcrusher", "vintagenoise"],
    "FILTER": ["filtertype", "filtercutoff", "filterresonance", "filtercontour", "filterdrive",
               "filterattack", "filterdecay", "filtersustain", "filterrelease"],
    "AMP": ["ampattack", "ampdecay", "ampsustain", "amprelease"],
    "LFO 1": ["lfo1destination", "lfo1waveform", "lfo1rate", "lfo1amount", "lfo1phase", "lfo1keytrigger"],
    "LFO 2": ["lfo2destination", "lfo2waveform", "lfo2rate", "lfo2amount", "lfo2phase", "lfo2keytrigger"],
    "FREE AD": ["freeaddestination", "freeadamount", "freeadattack", "freeaddecay"],
    "FX": ["reverbwet", "reverbdecay", "reverbpredelay", "reverbhighcut", "reverblowcut",
           "delaywet", "delaytime", "delayfeedback", "delayhighshelf", "delaylowshelf",
           "delaysync", "delayx2l", "delayx2r", "chorus1enable", "chorus2enable"],
    "MASTER": ["masterdetune", "masterhighpass"],
}

SAFE_CC = list(range(20, 32)) + list(range(33, 64)) + list(range(70, 90))

DEFAULT_SRC = os.path.expanduser(
    "~/Library/Application Support/ToguAudioLine/TAL-NoiseMaker/presets/default.noisemakerpreset")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "SynthForge_MIDI_Setup.noisemakerpreset")
TAL_PRESETS_DIR = os.path.expanduser(
    "~/Library/Application Support/ToguAudioLine/TAL-NoiseMaker/presets")


def build_mappings():
    model_names = [n for names in GROUPS.values() for n in names]
    assert len(model_names) == 63
    mappings = []
    for i, name in enumerate(model_names):
        xml_name = RENAME.get(name, name)
        param_index = ORDER.index(xml_name) - 1
        mappings.append((param_index, SAFE_CC[i]))
    assert len({idx for idx, _ in mappings}) == len(mappings), "duplicate indices"
    return mappings


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    if not os.path.isfile(src):
        print(f"Source preset not found: {src}")
        print("Pass a valid .noisemakerpreset file as argument.")
        sys.exit(1)

    tree = ET.parse(src)
    root = tree.getroot()
    program = root.find("programs").find("program")
    program.set("programname", "SynthForge MIDI Setup")

    midimap = root.find("midimap")
    if midimap is None:
        midimap = ET.SubElement(root, "midimap")
    else:
        for child in list(midimap):
            midimap.remove(child)

    for param_index, cc in build_mappings():
        ET.SubElement(midimap, "map", {"param": str(param_index), "controllernumber": str(cc)})

    tree.write(OUT_PATH, encoding="UTF-8", xml_declaration=True)
    print(f"Written: {OUT_PATH}")

    if os.path.isdir(TAL_PRESETS_DIR):
        dest = os.path.join(TAL_PRESETS_DIR, os.path.basename(OUT_PATH))
        with open(OUT_PATH, "rb") as f_in, open(dest, "wb") as f_out:
            f_out.write(f_in.read())
        print(f"Also copied to: {dest}")


if __name__ == "__main__":
    main()
