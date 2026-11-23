#!/usr/bin/env python3
"""Send test MIDI control-change messages on a virtual MIDI port, to check the
mapping written by generate_midi_setup_preset.py.
"""

import argparse
import sys
import time

import mido

PORT_NAME = "SynthForge Test"

MAPPING = [
    ("osc_1_waveform", 20), ("osc_1_tune", 21), ("osc_1_fine_tune", 22),
    ("osc_1_volume", 23), ("osc_1_pw", 24), ("osc_1_phase", 25),
    ("osc_2_waveform", 26), ("osc_2_tune", 27), ("osc_2_fine_tune", 28),
    ("osc_2_volume", 29), ("osc_2_fm", 30), ("osc_2_phase", 31),
    ("osc_3_volume", 33), ("ringmodulation", 34), ("osc_sync", 35),
    ("osc_bitcrusher", 36), ("vintage_noise", 37),
    ("filter_type", 38), ("filter_cutoff", 39), ("filter_resonance", 40),
    ("filter_contour", 41), ("filter_drive", 42), ("filter_attack", 43),
    ("filter_decay", 44), ("filter_sustain", 45), ("filter_release", 46),
    ("amp_attack", 47), ("amp_decay", 48), ("amp_sustain", 49), ("amp_release", 50),
    ("lfo_1_destination", 51), ("lfo_1_waveform", 52), ("lfo_1_rate", 53),
    ("lfo_1_amount", 54), ("lfo_1_phase", 55), ("lfo_1_keytrigger", 56),
    ("lfo_2_destination", 57), ("lfo_2_waveform", 58), ("lfo_2_rate", 59),
    ("lfo_2_amount", 60), ("lfo_2_phase", 61), ("lfo_2_keytrigger", 62),
    ("free_ad_destination", 63), ("free_ad_amount", 70), ("free_ad_attack", 71),
    ("free_ad_decay", 72),
    ("reverb_wet", 73), ("reverb_decay", 74), ("reverb_pre_delay", 75),
    ("reverb_high_cut", 76), ("reverb_low_cut", 77),
    ("delay_wet", 78), ("delay_time", 79), ("delay_feedback", 80),
    ("delay_high_shelf", 81), ("delay_low_shelf", 82), ("delay_sync", 83),
    ("delay_x2_l", 84), ("delay_x2_r", 85),
    ("chorus_1_enable", 86), ("chorus_2_enable", 87),
    ("master_detune", 88), ("master_high_pass", 89),
]

SPOT_CHECK = ["osc_1_waveform", "filter_cutoff", "filter_resonance", "amp_release"]


def sweep(port, cc, name):
    print(f"  CC {cc:3d}  ({name}) -> 0 ... 127 ... 64")
    for value in (0, 127, 64):
        port.send(mido.Message("control_change", channel=0, control=cc, value=value))
        time.sleep(0.6)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--spot", action="store_true", help="four representative controls (default)")
    group.add_argument("--all", action="store_true", help="all 63 controls, in order")
    group.add_argument("--cc", type=int, help="send only this CC number")
    args = parser.parse_args()

    print(f"Opening the virtual MIDI port '{PORT_NAME}'...")
    try:
        port = mido.open_output(PORT_NAME, virtual=True)
    except Exception as exc:
        print(f"Could not open the virtual MIDI port: {exc}")
        sys.exit(1)
    print(f"OK. In the DAW, set the MIDI input of the TAL-NoiseMaker track to '{PORT_NAME}'.")
    input("Press Enter to start sending the CC messages...")

    if args.cc is not None:
        sweep(port, args.cc, next((n for n, c in MAPPING if c == args.cc), "?"))
    elif args.all:
        for name, cc in MAPPING:
            sweep(port, cc, name)
    else:
        for name in SPOT_CHECK:
            cc = next(c for n, c in MAPPING if n == name)
            sweep(port, cc, name)

    print("Done.")


if __name__ == "__main__":
    main()
