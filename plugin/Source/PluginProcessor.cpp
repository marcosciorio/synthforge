#include "PluginProcessor.h"
#include "PluginEditor.h"

SynthForgeAudioProcessor::SynthForgeAudioProcessor()
    : AudioProcessor(BusesProperties())
{
    ensureServerRunning();

    auto abortFlag = abortRegistration;
    auto registeredFlag = hasRegistered;
    juce::Thread::launch([abortFlag, registeredFlag]
    {
        constexpr int maxAttempts = 300;
        for (int i = 0; i < maxAttempts && ! abortFlag->load(); ++i)
        {
            if (SynthForgeClient::isServerReachable())
            {
                if (! abortFlag->load())
                {
                    SynthForgeClient::registerClient();
                    registeredFlag->store(true);
                }
                return;
            }
            juce::Thread::sleep(200);
        }
    });
}

SynthForgeAudioProcessor::~SynthForgeAudioProcessor()
{
    abortRegistration->store(true);
    if (hasRegistered->load())
        SynthForgeClient::unregisterClient();
}

void SynthForgeAudioProcessor::ensureServerRunning()
{
    if (SynthForgeClient::isServerReachable())
        return;

    juce::String root = juce::SystemStats::getEnvironmentVariable("SYNTHFORGE_PROJECT_ROOT", {});
   #ifdef SYNTHFORGE_PROJECT_ROOT
    if (root.isEmpty())
        root = juce::String(juce::CharPointer_UTF8(SYNTHFORGE_PROJECT_ROOT));
   #endif
    if (root.isEmpty())
        return;
    juce::File pythonBin(root + "/synth/bin/python3");
    juce::File script(root + "/model/kaggle_training_cfm/synthforge_server.py");
    if (! pythonBin.existsAsFile() || ! script.existsAsFile())
        return;

    serverProcess = std::make_unique<juce::ChildProcess>();
    serverProcess->start(juce::StringArray { pythonBin.getFullPathName(), script.getFullPathName() },
                          juce::ChildProcess::wantStdOut);

}

void SynthForgeAudioProcessor::prepareToPlay(double, int) {}

void SynthForgeAudioProcessor::releaseResources() {}

bool SynthForgeAudioProcessor::isBusesLayoutSupported(const BusesLayout&) const
{
    return true;
}

void SynthForgeAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{

    buffer.clear();

    const juce::SpinLock::ScopedLockType lock(pendingMidiLock);
    if (! pendingMidi.isEmpty())
    {
        midi.addEvents(pendingMidi, 0, -1, 0);
        pendingMidi.clear();
    }
}

void SynthForgeAudioProcessor::sendParameters(const juce::var& paramsObject)
{
    auto* obj = paramsObject.getDynamicObject();
    if (obj == nullptr)
        return;

    constexpr int midiChannel = 1;
    juce::MidiBuffer newEvents;
    for (auto& mapping : getParamCCMap())
    {
        auto value = obj->getProperty(mapping.vst3Name);
        if (value.isVoid())
            continue;
        auto v01 = juce::jlimit(0.0f, 1.0f, (float) value);
        auto ccValue = juce::roundToInt(v01 * 127.0f);
        newEvents.addEvent(juce::MidiMessage::controllerEvent(midiChannel, mapping.cc, ccValue), 0);
    }

    const juce::SpinLock::ScopedLockType lock(pendingMidiLock);
    pendingMidi.addEvents(newEvents, 0, -1, 0);
}

juce::AudioProcessorEditor* SynthForgeAudioProcessor::createEditor()
{
    return new SynthForgeAudioProcessorEditor(*this);
}

void SynthForgeAudioProcessor::getStateInformation(juce::MemoryBlock&) {}

void SynthForgeAudioProcessor::setStateInformation(const void*, int) {}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SynthForgeAudioProcessor();
}
