#pragma once

#include <juce_core/juce_core.h>
#include <array>

struct ParamCCMapping
{
    const char* vst3Name;
    int cc;
};

inline const std::array<ParamCCMapping, 63>& getParamCCMap()
{
    static const std::array<ParamCCMapping, 63> table { {

        { "osc_1_waveform", 20 },
        { "osc_1_tune", 21 },
        { "osc_1_fine_tune", 22 },
        { "osc_1_volume", 23 },
        { "osc_1_pw", 24 },
        { "osc_1_phase", 25 },

        { "osc_2_waveform", 26 },
        { "osc_2_tune", 27 },
        { "osc_2_fine_tune", 28 },
        { "osc_2_volume", 29 },
        { "osc_2_fm", 30 },
        { "osc_2_phase", 31 },

        { "osc_3_volume", 33 },
        { "ringmodulation", 34 },
        { "osc_sync", 35 },
        { "osc_bitcrusher", 36 },
        { "vintage_noise", 37 },

        { "filter_type", 38 },
        { "filter_cutoff", 39 },
        { "filter_resonance", 40 },
        { "filter_contour", 41 },
        { "filter_drive", 42 },
        { "filter_attack", 43 },
        { "filter_decay", 44 },
        { "filter_sustain", 45 },
        { "filter_release", 46 },

        { "amp_attack", 47 },
        { "amp_decay", 48 },
        { "amp_sustain", 49 },
        { "amp_release", 50 },

        { "lfo_1_destination", 51 },
        { "lfo_1_waveform", 52 },
        { "lfo_1_rate", 53 },
        { "lfo_1_amount", 54 },
        { "lfo_1_phase", 55 },
        { "lfo_1_keytrigger", 56 },

        { "lfo_2_destination", 57 },
        { "lfo_2_waveform", 58 },
        { "lfo_2_rate", 59 },
        { "lfo_2_amount", 60 },
        { "lfo_2_phase", 61 },
        { "lfo_2_keytrigger", 62 },

        { "free_ad_destination", 63 },
        { "free_ad_amount", 70 },
        { "free_ad_attack", 71 },
        { "free_ad_decay", 72 },

        { "reverb_wet", 73 },
        { "reverb_decay", 74 },
        { "reverb_pre_delay", 75 },
        { "reverb_high_cut", 76 },
        { "reverb_low_cut", 77 },
        { "delay_wet", 78 },
        { "delay_time", 79 },
        { "delay_feedback", 80 },
        { "delay_high_shelf", 81 },
        { "delay_low_shelf", 82 },
        { "delay_sync", 83 },
        { "delay_x2_l", 84 },
        { "delay_x2_r", 85 },
        { "chorus_1_enable", 86 },
        { "chorus_2_enable", 87 },

        { "master_detune", 88 },
        { "master_high_pass", 89 },
    } };
    return table;
}
