#pragma once

#include <array>
#include <memory>
#include <vector>
#include <juce_audio_processors/juce_audio_processors.h>
#include "PluginProcessor.h"
#include "SynthForgeLookAndFeel.h"
#include "SynthForgeClient.h"
#include "Vocabulary.h"
#include "ForceCanonical.h"

class SynthForgeAudioProcessorEditor final : public juce::AudioProcessorEditor,
                                             private juce::Timer
{
public:
    explicit SynthForgeAudioProcessorEditor(SynthForgeAudioProcessor&);
    ~SynthForgeAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    bool serverReady = false;

    struct HistoryEntry
    {
        juce::var params;
        juce::var x1;
    };

    void refreshPromptFromChoices();
    void showVocabMenu(juce::TextButton& button, const VocabField& field);
    void chooseAudioFile();
    void refreshKnobAvailability();

    void applyFidelityDefault();
    int fidelityMode = -1;
    bool hasCategorySource() const;
    juce::String resolveCategory(const juce::String& promptContinuation) const;
    void triggerGenerate();
    void handleGenerateResponse(const SynthForgeClient::Response& response);
    void pushHistory(const SynthForgeClient::Response& response);
    void applyHistory(int index);
    void updateHistoryButtons();

    void sendCurrentParams();
    void setShowingHelp(bool shouldShowHelp);
    void setStatus(const juce::String& message, juce::Colour colour);

    SynthForgeAudioProcessor& processorRef;
    SynthForgeLookAndFeel lookAndFeelCustom;

    juce::Component mainPage;
    juce::Component helpPage;

    juce::Rectangle<int> generateBoxBounds, describeBoxBounds, sourceBoxBounds;

    juce::Label titleLabel;
    juce::TextButton helpButton { "?" };

    juce::Label sectionGenerate, sectionSource, sectionDescribe;

    juce::Label stepLabel, stepValueLabel;
    juce::Label fideltyLabel, fideltyValueLabel;
    juce::Label retrievalLabel, retrievalValueLabel;
    juce::Label memoryLabel, memoryValueLabel;
    juce::Label statusLabel;
    juce::Label brandFooterLabel;

    juce::Slider stepKnob, fideltyKnob, retrievalKnob, memoryKnob;
    juce::TextButton retrievalCategoryToggle { "CAT" };

    juce::Label promptPrefixLabel;
    juce::TextEditor promptEditor;
    juce::TextButton genButton { "GEN" };
    juce::TextButton uploadButton { "UPLOAD SOUND" };
    juce::TextButton undoButton { "undo" };
    juce::TextButton redoButton { "redo" };

    juce::TextEditor helpText;
    juce::TextButton backButton { "BACK" };

    std::array<std::unique_ptr<juce::TextButton>, 7> vocabButtons;

    juce::TextButton forceToggle { "FORCE" };
    juce::StringPairArray choices;
    juce::String uploadedAudioPath;
    std::unique_ptr<juce::FileChooser> fileChooser;

    juce::var lastX1;
    std::vector<HistoryEntry> history;
    int historyPos = -1;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SynthForgeAudioProcessorEditor)
};
