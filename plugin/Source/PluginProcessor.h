#pragma once

#include <atomic>
#include <memory>
#include <juce_audio_processors/juce_audio_processors.h>
#include "MidiCCMap.h"
#include "SynthForgeClient.h"

class SynthForgeAudioProcessor final : public juce::AudioProcessor
{
public:
    SynthForgeAudioProcessor();
    ~SynthForgeAudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    void sendParameters(const juce::var& paramsObject);

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }

    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return true; }
    bool isMidiEffect() const override { return true; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

private:
    void ensureServerRunning();

    juce::MidiBuffer pendingMidi;
    juce::SpinLock pendingMidiLock;
    std::unique_ptr<juce::ChildProcess> serverProcess;

    std::shared_ptr<std::atomic<bool>> abortRegistration { std::make_shared<std::atomic<bool>>(false) };
    std::shared_ptr<std::atomic<bool>> hasRegistered { std::make_shared<std::atomic<bool>>(false) };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SynthForgeAudioProcessor)
};
