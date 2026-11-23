#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "SynthForgeColours.h"
#include "SynthForgeFonts.h"

class SynthForgeLookAndFeel final : public juce::LookAndFeel_V4
{
public:
    SynthForgeLookAndFeel()
    {
        using namespace SynthForgeColours;
        setColour(juce::TooltipWindow::textColourId, textPrimary);
        setColour(juce::TooltipWindow::backgroundColourId, bgControl);
        setColour(juce::TooltipWindow::outlineColourId, borderDark);
    }

    void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                           float sliderPosProportional, float rotaryStartAngle, float rotaryEndAngle,
                           juce::Slider& slider) override
    {
        using namespace SynthForgeColours;
        auto bounds = juce::Rectangle<float>((float) x, (float) y, (float) width, (float) height).reduced(4.0f);
        auto radius = juce::jmin(bounds.getWidth(), bounds.getHeight()) / 2.0f;
        auto centre = bounds.getCentre();
        auto angle  = rotaryStartAngle + sliderPosProportional * (rotaryEndAngle - rotaryStartAngle);
        auto enabled = slider.isEnabled();

        constexpr int numTicks = 11;
        auto tickInner = radius * (25.0f / 35.0f);
        auto tickOuter = radius * (29.0f / 35.0f);
        for (int i = 0; i < numTicks; ++i)
        {
            auto t = (float) i / (float) (numTicks - 1);
            auto a = rotaryStartAngle + t * (rotaryEndAngle - rotaryStartAngle);
            auto p1 = centre.getPointOnCircumference(tickInner, a);
            auto p2 = centre.getPointOnCircumference(tickOuter, a);
            g.setColour(enabled ? knobTick : knobTick.withAlpha(0.5f));
            g.drawLine({ p1, p2 }, 1.4f);
        }

        auto ringRadius = radius * (25.0f / 35.0f);
        auto faceRadius = radius * (23.0f / 35.0f);

        g.setColour(juce::Colour::fromRGB(0x0a, 0x0a, 0x0a));
        g.fillEllipse(centre.x - ringRadius, centre.y - ringRadius, ringRadius * 2.0f, ringRadius * 2.0f);

        juce::ColourGradient faceGrad(
            enabled ? juce::Colour::fromRGB(0x4a, 0x4a, 0x4a) : juce::Colour::fromRGB(0x33, 0x33, 0x33),
            centre.x - faceRadius * 0.3f, centre.y - faceRadius * 0.4f,
            enabled ? juce::Colour::fromRGB(0x15, 0x15, 0x15) : juce::Colour::fromRGB(0x18, 0x18, 0x18),
            centre.x + faceRadius * 0.7f, centre.y + faceRadius * 0.7f,
            true);
        faceGrad.addColour(0.7, enabled ? juce::Colour::fromRGB(0x2a, 0x2a, 0x2a) : juce::Colour::fromRGB(0x24, 0x24, 0x24));
        g.setGradientFill(faceGrad);
        g.fillEllipse(centre.x - faceRadius, centre.y - faceRadius, faceRadius * 2.0f, faceRadius * 2.0f);
        g.setColour(juce::Colour::fromRGB(0x0a, 0x0a, 0x0a));
        g.drawEllipse(centre.x - faceRadius, centre.y - faceRadius, faceRadius * 2.0f, faceRadius * 2.0f, 0.5f);

        auto hlW = radius * (14.0f / 35.0f);
        auto hlH = radius * (3.0f / 35.0f);
        auto hlY = centre.y - faceRadius * (13.0f / 23.0f);
        g.setColour(juce::Colours::white.withAlpha(0.05f));
        g.fillEllipse(centre.x - hlW, hlY - hlH, hlW * 2.0f, hlH * 2.0f);

        auto indInner = radius * (8.0f / 35.0f);
        auto indOuter = radius * (20.0f / 35.0f);
        juce::Path tick;
        tick.startNewSubPath(0.0f, -indOuter);
        tick.lineTo(0.0f, -indInner);
        tick.applyTransform(juce::AffineTransform::rotation(angle).translated(centre));
        g.setColour(enabled ? textPrimary : textDim);
        g.strokePath(tick, juce::PathStrokeType(2.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    }

    void drawButtonBackground(juce::Graphics& g, juce::Button& button, const juce::Colour&,
                               bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override
    {
        using namespace SynthForgeColours;
        auto bounds = button.getLocalBounds().toFloat().reduced(0.5f);
        auto id = button.getComponentID();
        bool enabled = button.isEnabled();
        bool active = button.getToggleState();
        float corner = (id == "help") ? 4.0f : 3.0f;

        juce::Colour top, bottom;

        if (id == "gen")
        {
            top = orangeDark; bottom = orangeDarkShade;
        }
        else if (id == "vocab")
        {
            top = bottom = active ? descriptorSelectedBg : bgControl;
        }
        else if (id == "help")
        {
            top = bottom = bgSection;
        }
        else if (id == "cat" || id == "force" || id == "upload" || id == "arrow")
        {
            top = juce::Colour::fromRGB(0x3a, 0x3a, 0x3a);
            bottom = (id == "upload") ? juce::Colour::fromRGB(0x26, 0x26, 0x26) : juce::Colour::fromRGB(0x2a, 0x2a, 0x2a);
        }
        else
        {
            top = bottom = bgControl;
        }

        if (! enabled)
        {
            top = top.withMultipliedAlpha(0.5f);
            bottom = bottom.withMultipliedAlpha(0.5f);
        }
        else if (shouldDrawButtonAsDown)
        {
            top = top.darker(0.2f); bottom = bottom.darker(0.2f);
        }
        else if (shouldDrawButtonAsHighlighted)
        {
            top = top.brighter(0.08f); bottom = bottom.brighter(0.08f);
        }

        juce::ColourGradient grad(top, bounds.getX(), bounds.getY(), bottom, bounds.getX(), bounds.getBottom(), false);
        g.setGradientFill(grad);
        g.fillRoundedRectangle(bounds, corner);

        juce::Colour borderColour = borderDark;
        if (id == "vocab" && active)
            borderColour = orangeLightShade;
        else if (id == "arrow" && shouldDrawButtonAsHighlighted && enabled)
            borderColour = orangeLightShade;
        g.setColour(borderColour);
        g.drawRoundedRectangle(bounds, corner, 1.0f);
    }

    void drawButtonText(juce::Graphics& g, juce::TextButton& button, bool, bool shouldDrawButtonAsHighlighted) override
    {
        using namespace SynthForgeColours;
        auto id = button.getComponentID();

        if (id == "arrow")
        {
            drawArrowIcon(g, button, shouldDrawButtonAsHighlighted);
            return;
        }

        auto bounds = button.getLocalBounds().reduced(6);
        bool active = button.getToggleState();
        bool enabled = button.isEnabled();
        bool isVocab = (id == "vocab");
        bool isUpload = (id == "upload");

        if (isVocab)
            bounds.removeFromRight(16);

        juce::Rectangle<int> uploadIconArea;
        if (isUpload)
        {
            uploadIconArea = bounds.removeFromLeft(16);
            bounds.removeFromLeft(8);
        }

        juce::Colour textColour;
        juce::Font font;

        if (id == "gen")
        {
            textColour = juce::Colours::white;
            font = SynthForgeFonts::rajdhaniBold(15.0f).withExtraKerningFactor(0.15f);
        }
        else if (id == "vocab")
        {
            textColour = active ? orangeLight : textPrimary;
            font = SynthForgeFonts::interSemiBold(12.0f).withExtraKerningFactor(0.1f);
        }
        else if (id == "help")
        {
            textColour = textSecondary;
            font = SynthForgeFonts::interSemiBold(14.0f);
        }
        else if (id == "upload" || id == "cat" || id == "force")
        {
            textColour = active ? orangeLight : textPrimary;
            font = SynthForgeFonts::rajdhaniBold(14.0f).withExtraKerningFactor(0.16f);
        }
        else
        {
            textColour = textPrimary;
            font = SynthForgeFonts::interSemiBold(12.0f);
        }

        if (! enabled)
            textColour = textDim;

        g.setColour(textColour);
        g.setFont(font);
        g.drawFittedText(button.getButtonText(), bounds, juce::Justification::centred, 2);

        if (isUpload)
        {
            auto iconBounds = uploadIconArea.toFloat().reduced(1.0f);
            g.setColour(textColour);
            g.drawRoundedRectangle(iconBounds, 2.0f, 1.3f);
            juce::Path arrow;
            auto cx = iconBounds.getCentreX();
            auto cy = iconBounds.getCentreY();
            arrow.startNewSubPath(cx, cy + 3.0f);
            arrow.lineTo(cx, cy - 3.0f);
            arrow.startNewSubPath(cx - 2.5f, cy - 0.8f);
            arrow.lineTo(cx, cy - 3.0f);
            arrow.lineTo(cx + 2.5f, cy - 0.8f);
            g.strokePath(arrow, juce::PathStrokeType(1.3f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
        }

        if (isVocab)
        {
            auto chevronArea = button.getLocalBounds().removeFromRight(20).toFloat();
            juce::Path chevron;
            auto cx = chevronArea.getCentreX();
            auto cy = chevronArea.getCentreY();
            chevron.startNewSubPath(cx - 4.0f, cy - 2.5f);
            chevron.lineTo(cx, cy + 2.5f);
            chevron.lineTo(cx + 4.0f, cy - 2.5f);
            g.setColour(textDim);
            g.strokePath(chevron, juce::PathStrokeType(1.5f));
        }
    }

private:

    void drawArrowIcon(juce::Graphics& g, juce::TextButton& button, bool highlighted)
    {
        using namespace SynthForgeColours;
        bool isUndo = button.getButtonText() == "undo";
        bool enabled = button.isEnabled();

        juce::Path icon;
        if (isUndo)
        {
            icon.startNewSubPath(9.0f, 14.0f);
            icon.lineTo(4.0f, 9.0f);
            icon.lineTo(9.0f, 4.0f);
            icon.startNewSubPath(4.0f, 9.0f);
            icon.lineTo(15.0f, 9.0f);
            icon.addArc(10.0f, 9.0f, 10.0f, 10.0f, 0.0f, juce::MathConstants<float>::pi, false);
            icon.lineTo(11.0f, 19.0f);
        }
        else
        {
            icon.startNewSubPath(15.0f, 14.0f);
            icon.lineTo(20.0f, 9.0f);
            icon.lineTo(15.0f, 4.0f);
            icon.startNewSubPath(20.0f, 9.0f);
            icon.lineTo(9.0f, 9.0f);
            icon.addArc(4.0f, 9.0f, 10.0f, 10.0f, 0.0f, -juce::MathConstants<float>::pi, false);
            icon.lineTo(13.0f, 19.0f);
        }

        auto bounds = button.getLocalBounds().toFloat().reduced(8.0f);
        icon.scaleToFit(bounds.getX(), bounds.getY(), bounds.getWidth(), bounds.getHeight(), true);

        auto colour = ! enabled ? textDim : (highlighted ? orangeLight : textPrimary);
        g.setColour(colour);
        g.strokePath(icon, juce::PathStrokeType(2.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    }
};
