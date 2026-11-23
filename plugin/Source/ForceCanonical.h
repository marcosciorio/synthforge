#pragma once

#include <juce_core/juce_core.h>
#include <map>

namespace ForceCanonical
{
    using Patch = std::map<juce::String, double>;

    inline const std::map<juce::String, Patch>& timeShapeTable()
    {
        static const std::map<juce::String, Patch> table {

            { "pulsing", {
                { "lfo_1_amount", 0.9 }, { "lfo_1_rate", 0.6 }, { "lfo_1_destination", 0.142857 },
            } },
            { "evolving", {
                { "lfo_1_amount", 0.85 }, { "lfo_1_rate", 0.15 }, { "lfo_1_destination", 0.142857 },
                { "lfo_2_amount", 0.0 }, { "lfo_2_destination", 0.0 },
                { "delay_sync", 0.0 },
            } },
            { "slow", {
                { "amp_attack", 0.6 },
                { "lfo_1_amount", 0.0 }, { "lfo_1_destination", 0.0 },
                { "lfo_2_amount", 0.0 }, { "lfo_2_destination", 0.0 },
                { "delay_sync", 0.0 },
                { "free_ad_amount", 0.0 }, { "free_ad_destination", 0.0 },
            } },
            { "long", {
                { "amp_attack", 0.1 }, { "amp_sustain", 0.85 }, { "amp_release", 0.8 },
                { "lfo_1_amount", 0.0 }, { "lfo_1_destination", 0.0 },
                { "lfo_2_amount", 0.0 }, { "lfo_2_destination", 0.0 },
                { "delay_sync", 0.0 },
                { "free_ad_amount", 0.0 }, { "free_ad_destination", 0.0 },
                { "filter_attack", 0.3 },
            } },
            { "fading", {
                { "amp_attack", 0.1 }, { "amp_decay", 0.5 }, { "amp_sustain", 0.1 }, { "amp_release", 0.45 },
                { "lfo_1_amount", 0.0 }, { "lfo_1_destination", 0.0 },
                { "lfo_2_amount", 0.0 }, { "lfo_2_destination", 0.0 },
                { "delay_sync", 0.0 },
                { "free_ad_amount", 0.0 }, { "free_ad_destination", 0.0 },
            } },
            { "short", {
                { "amp_attack", 0.1 }, { "amp_sustain", 0.1 }, { "amp_release", 0.1 },
                { "lfo_1_amount", 0.0 }, { "lfo_1_destination", 0.0 },
                { "lfo_2_amount", 0.0 }, { "lfo_2_destination", 0.0 },
                { "delay_sync", 0.0 },
                { "free_ad_amount", 0.0 }, { "free_ad_destination", 0.0 },
            } },
        };
        return table;
    }

    inline const std::map<juce::String, Patch>& spaceTable()
    {
        static const std::map<juce::String, Patch> table {
            { "cavernous", { { "reverb_wet", 0.6 }, { "reverb_decay", 0.8 } } },
            { "echoey",    { { "delay_wet", 0.4 }, { "delay_time", 0.3 }, { "reverb_wet", 0.1 } } },
            { "spacious",  { { "reverb_wet", 0.3 }, { "reverb_decay", 0.3 }, { "delay_wet", 0.1 } } },
            { "chorused",  { { "chorus_1_enable", 1.0 }, { "reverb_wet", 0.1 }, { "delay_wet", 0.1 } } },
            { "dry",       { { "reverb_wet", 0.0 }, { "delay_wet", 0.0 },
                              { "chorus_1_enable", 0.0 }, { "chorus_2_enable", 0.0 } } },
        };
        return table;
    }

    inline void applyPatch(juce::DynamicObject& target, const std::map<juce::String, Patch>& table,
                            const juce::String& key)
    {
        if (key.isEmpty())
            return;
        auto it = table.find(key.toLowerCase());
        if (it == table.end())
            return;
        for (auto& [vst3Name, value] : it->second)
            if (target.hasProperty(vst3Name))
                target.setProperty(vst3Name, value);
    }

    inline juce::var apply(const juce::var& baseParams, const juce::String& forceTimeShape,
                            const juce::String& forceSpace)
    {
        auto* srcObj = baseParams.getDynamicObject();
        if (srcObj == nullptr)
            return baseParams;

        juce::DynamicObject::Ptr result = new juce::DynamicObject();
        for (auto& prop : srcObj->getProperties())
            result->setProperty(prop.name, prop.value);

        applyPatch(*result, timeShapeTable(), forceTimeShape);
        applyPatch(*result, spaceTable(), forceSpace);

        return juce::var(result.get());
    }
}
