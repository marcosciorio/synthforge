#pragma once

#include <juce_core/juce_core.h>
#include <array>
#include <map>

struct VocabField
{
    juce::String key;
    juce::String buttonLabel;
    juce::StringArray words;
};

inline const std::array<VocabField, 7>& getVocabFields()
{
    static const std::array<VocabField, 7> fields { {
        { "category",   "SOUND TYPE",     juce::StringArray { "lead", "bass", "fx", "pad", "drum", "keyboard", "arp", "chord", "pluck" } },
        { "brightness", "BRIGHT OR DARK", juce::StringArray { "bright", "dark", "warm", "mellow", "sharp", "muffled" } },
        { "loudness",   "LOUD OR SOFT",   juce::StringArray { "soft", "gentle", "loud", "powerful", "aggressive" } },
        { "weight",     "WEIGHT OR BODY", juce::StringArray { "deep", "full", "thin", "light", "heavy" } },
        { "texture",    "TEXTURE",        juce::StringArray { "smooth", "rough", "clean", "harsh", "buzzy", "metallic", "hollow", "lush" } },
        { "space",      "SPACE",          juce::StringArray { "dry", "cavernous", "spacious", "echoey", "chorused" } },
        { "timeShape",  "TIME",           juce::StringArray { "short", "long", "slow", "fading", "pulsing", "evolving" } },
    } };
    return fields;
}

inline juce::String joinWithAnd(const juce::StringArray& items)
{
    if (items.isEmpty())
        return {};
    if (items.size() == 1)
        return items[0];

    juce::StringArray allButLast(items);
    auto last = allButLast[allButLast.size() - 1];
    allButLast.remove(allButLast.size() - 1);
    return allButLast.joinIntoString(", ") + " and " + last;
}

inline juce::String aOrAn(const juce::String& word)
{
    if (word.isEmpty())
        return "a";
    auto c = juce::CharacterFunctions::toLowerCase(word[0]);
    return juce::String("aeiou").containsChar(c) ? "an" : "a";
}

inline juce::String composeSentence(const juce::StringPairArray& choices)
{
    auto category = choices.getValue("category", {}).trim();
    if (category.isEmpty())
        category = "synthesizer";

    juce::StringArray catWords;
    catWords.addTokens(category.toLowerCase(), " ", "");

    static const juce::StringArray descriptorKeys { "brightness", "texture", "space", "loudness", "weight" };
    juce::StringArray descriptors;
    for (auto& key : descriptorKeys)
    {
        auto word = choices.getValue(key, {}).trim();
        if (word.isNotEmpty() && ! catWords.contains(word.toLowerCase()))
            descriptors.add(word);
    }

    juce::String sentence;
    if (! descriptors.isEmpty())
    {
        auto subject = joinWithAnd(descriptors);
        sentence = "The sound of " + aOrAn(subject) + " " + subject + " " + category;
    }
    else
    {
        sentence = "The sound of " + aOrAn(category) + " " + category;
    }

    auto timeShape = choices.getValue("timeShape", {}).trim();
    if (timeShape.isNotEmpty())
        sentence << ", " << timeShape;

    return sentence;
}

inline const std::map<juce::String, juce::StringArray>& getTimeShapeSynonyms()
{
    static const std::map<juce::String, juce::StringArray> table {
        { "short",    juce::StringArray { "quick", "brief", "snappy", "punchy", "staccato" } },
        { "long",     juce::StringArray { "sustained", "held", "drone", "continuous" } },
        { "slow",     juce::StringArray { "gradual", "building" } },
        { "fading",   juce::StringArray { "dying", "decaying", "vanishing" } },
        { "pulsing",  juce::StringArray { "throbbing", "pumping", "rhythmic" } },
        { "evolving", juce::StringArray { "morphing", "shifting", "transforming" } },
    };
    return table;
}

inline const std::map<juce::String, juce::StringArray>& getSpaceSynonyms()
{
    static const std::map<juce::String, juce::StringArray> table {
        { "dry",       juce::StringArray { "close", "direct", "unprocessed" } },
        { "cavernous", juce::StringArray { "huge", "massive", "cathedral" } },
        { "spacious",  juce::StringArray { "roomy", "reverb", "reverberant", "ambient" } },
        { "echoey",    juce::StringArray { "echo", "echoes", "delay" } },
        { "chorused",  juce::StringArray { "chorus", "detuned" } },
    };
    return table;
}

inline juce::String resolveForceValue(const juce::StringPairArray& choices,
                                       const juce::String& fieldKey,
                                       const juce::String& promptContinuation)
{
    auto explicitValue = choices.getValue(fieldKey, {}).trim();
    if (explicitValue.isNotEmpty())
        return explicitValue;

    const VocabField* field = nullptr;
    for (auto& f : getVocabFields())
        if (f.key == fieldKey)
            field = &f;
    if (field == nullptr)
        return {};

    for (auto& word : field->words)
        if (promptContinuation.containsWholeWordIgnoreCase(word))
            return word;

    auto& synonyms = (fieldKey == "timeShape") ? getTimeShapeSynonyms() : getSpaceSynonyms();
    for (auto& [canonical, words] : synonyms)
        for (auto& word : words)
            if (promptContinuation.containsWholeWordIgnoreCase(word))
                return canonical;

    return {};
}
