#pragma once

#include <juce_graphics/juce_graphics.h>
#include "BinaryData.h"

namespace SynthForgeFonts
{
    inline juce::Typeface::Ptr rajdhaniBoldTypeface()
    {
        static auto tf = juce::Typeface::createSystemTypefaceFor(
            BinaryData::RajdhaniBold_ttf, (size_t) BinaryData::RajdhaniBold_ttfSize);
        return tf;
    }

    inline juce::Typeface::Ptr rajdhaniSemiBoldTypeface()
    {
        static auto tf = juce::Typeface::createSystemTypefaceFor(
            BinaryData::RajdhaniSemiBold_ttf, (size_t) BinaryData::RajdhaniSemiBold_ttfSize);
        return tf;
    }

    inline juce::Typeface::Ptr interRegularTypeface()
    {
        static auto tf = juce::Typeface::createSystemTypefaceFor(
            BinaryData::InterRegular_ttf, (size_t) BinaryData::InterRegular_ttfSize);
        return tf;
    }

    inline juce::Typeface::Ptr interSemiBoldTypeface()
    {
        static auto tf = juce::Typeface::createSystemTypefaceFor(
            BinaryData::InterSemiBold_ttf, (size_t) BinaryData::InterSemiBold_ttfSize);
        return tf;
    }

    inline juce::Typeface::Ptr interBoldTypeface()
    {
        static auto tf = juce::Typeface::createSystemTypefaceFor(
            BinaryData::InterBold_ttf, (size_t) BinaryData::InterBold_ttfSize);
        return tf;
    }

    inline juce::Typeface::Ptr jetBrainsMonoTypeface()
    {
        static auto tf = juce::Typeface::createSystemTypefaceFor(
            BinaryData::JetBrainsMonoRegular_ttf, (size_t) BinaryData::JetBrainsMonoRegular_ttfSize);
        return tf;
    }

    inline juce::Font rajdhaniBold(float height)     { return juce::Font(rajdhaniBoldTypeface()).withHeight(height); }
    inline juce::Font rajdhaniSemiBold(float height) { return juce::Font(rajdhaniSemiBoldTypeface()).withHeight(height); }
    inline juce::Font interRegular(float height)     { return juce::Font(interRegularTypeface()).withHeight(height); }
    inline juce::Font interSemiBold(float height)    { return juce::Font(interSemiBoldTypeface()).withHeight(height); }
    inline juce::Font interBold(float height)        { return juce::Font(interBoldTypeface()).withHeight(height); }
    inline juce::Font jetBrainsMono(float height)    { return juce::Font(jetBrainsMonoTypeface()).withHeight(height); }
}
