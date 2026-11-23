#pragma once

#include <juce_graphics/juce_graphics.h>

namespace SynthForgeColours
{
    using juce::Colour;

    inline const Colour bgOuter   = Colour::fromRGB(0x1a, 0x1a, 0x1a);
    inline const Colour bgInner   = Colour::fromRGB(0x23, 0x23, 0x23);
    inline const Colour bgSection = Colour::fromRGB(0x2f, 0x2f, 0x2f);
    inline const Colour bgControl = Colour::fromRGB(0x1a, 0x1a, 0x1a);

    inline const Colour borderDark = Colour::fromRGB(0x0d, 0x0d, 0x0d);
    inline const Colour borderMid  = Colour::fromRGB(0x3d, 0x3d, 0x3d);
    inline const Colour borderHi   = Colour::fromRGB(0x4a, 0x4a, 0x4a);

    inline const Colour textPrimary   = Colour::fromRGB(0xe8, 0xe8, 0xe8);
    inline const Colour textSecondary = Colour::fromRGB(0xa8, 0xa8, 0xa8);
    inline const Colour textDim       = Colour::fromRGB(0x70, 0x70, 0x70);
    inline const Colour textOnAccent  = juce::Colours::white;

    inline const Colour orangeLight      = Colour::fromRGB(0xe8, 0x98, 0x38);
    inline const Colour orangeLightShade = Colour::fromRGB(0xc6, 0x78, 0x20);
    inline const Colour orangeDark       = Colour::fromRGB(0xb8, 0x48, 0x08);
    inline const Colour orangeDarkShade  = Colour::fromRGB(0x8a, 0x36, 0x00);
    inline const Colour teal             = Colour::fromRGB(0x3e, 0x88, 0x78);
    inline const Colour tealShade        = Colour::fromRGB(0x2c, 0x65, 0x58);

    inline const Colour knobTick = Colour::fromRGB(0x5a, 0x5a, 0x5a);

    inline const Colour statusOk    = Colour::fromRGB(0x4a, 0xde, 0x80);
    inline const Colour statusWarn  = Colour::fromRGB(0xfa, 0xcc, 0x15);
    inline const Colour statusError = Colour::fromRGB(0xf8, 0x71, 0x71);

    inline const Colour descriptorSelectedBg = Colour::fromRGB(0x25, 0x1c, 0x15);
}
