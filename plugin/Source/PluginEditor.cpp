#include "PluginEditor.h"
#include "SynthForgeClient.h"
#include "SynthForgeColours.h"
#include "SynthForgeFonts.h"
#include <functional>

using namespace SynthForgeColours;

namespace
{
    constexpr int kKnobSize = 70;
    constexpr int kHeaderH = 28;

    void styleKnobLabel(juce::Label& label, const juce::String& text)
    {
        label.setText(text, juce::dontSendNotification);
        label.setJustificationType(juce::Justification::centred);
        label.setFont(SynthForgeFonts::interSemiBold(10.0f).withExtraKerningFactor(0.15f));
        label.setColour(juce::Label::textColourId, textSecondary);
    }

    void styleValueLabel(juce::Label& label)
    {
        label.setJustificationType(juce::Justification::centred);
        label.setFont(SynthForgeFonts::jetBrainsMono(10.0f).withExtraKerningFactor(0.05f));
        label.setColour(juce::Label::textColourId, textPrimary);
        label.setColour(juce::Label::backgroundColourId, bgControl);
        label.setColour(juce::Label::outlineColourId, borderDark);
    }

    void styleSectionHeader(juce::Label& label, const juce::String& text)
    {
        label.setText(text, juce::dontSendNotification);
        label.setJustificationType(juce::Justification::centred);
        label.setFont(SynthForgeFonts::rajdhaniBold(13.0f).withExtraKerningFactor(0.19f));
        label.setColour(juce::Label::textColourId, juce::Colours::white);

    }

    void styleKnob(juce::Slider& knob, double min, double max, double step, double defaultValue)
    {
        knob.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        knob.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
        knob.setRange(min, max, step);
        knob.setDoubleClickReturnValue(true, defaultValue);
        knob.setValue(defaultValue, juce::dontSendNotification);
    }

    juce::Rectangle<int> layoutKnobBlock(juce::Rectangle<int> area, juce::Label& label, juce::Slider& knob, juce::Label& valueLabel)
    {
        label.setBounds(area.removeFromTop(14));
        area.removeFromTop(6);
        knob.setBounds(area.removeFromTop(kKnobSize).withSizeKeepingCentre(kKnobSize, kKnobSize));
        area.removeFromTop(6);
        valueLabel.setBounds(area.removeFromTop(18).withSizeKeepingCentre(44, 18));
        return area;
    }

    void paintSectionModule(juce::Graphics& g, juce::Rectangle<int> box, juce::Colour headerTop, juce::Colour headerBottom)
    {
        g.setColour(bgSection);
        g.fillRect(box);

        auto header = box.removeFromTop(kHeaderH).toFloat();
        juce::ColourGradient grad(headerTop, header.getX(), header.getY(), headerBottom, header.getX(), header.getBottom(), false);
        g.setGradientFill(grad);
        g.fillRect(header);

        g.setColour(borderDark);
        g.drawRect(box.withTop(box.getY() - kHeaderH), 1);
        g.drawLine(header.getX(), header.getBottom(), header.getRight(), header.getBottom(), 1.0f);
    }
}

SynthForgeAudioProcessorEditor::SynthForgeAudioProcessorEditor(SynthForgeAudioProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p)
{
    setLookAndFeel(&lookAndFeelCustom);

    titleLabel.setText("SynthForge", juce::dontSendNotification);
    titleLabel.setJustificationType(juce::Justification::centred);
    titleLabel.setFont(SynthForgeFonts::rajdhaniBold(30.0f).withExtraKerningFactor(0.12f));
    titleLabel.setColour(juce::Label::textColourId, textPrimary);
    addAndMakeVisible(titleLabel);

    helpButton.setComponentID("help");
    helpButton.setTooltip("How SynthForge works");
    helpButton.onClick = [this] { setShowingHelp(true); };
    addAndMakeVisible(helpButton);

    addAndMakeVisible(mainPage);
    addChildComponent(helpPage);

    styleSectionHeader(sectionGenerate, "GENERATE");
    styleSectionHeader(sectionSource, "SOURCE");
    styleSectionHeader(sectionDescribe, "DESCRIBE");
    for (auto* l : { &sectionGenerate, &sectionSource, &sectionDescribe })
        mainPage.addAndMakeVisible(l);

    styleKnobLabel(stepLabel, "STEP");
    styleKnobLabel(fideltyLabel, "FIDELITY");
    styleKnobLabel(retrievalLabel, "RETRIEVAL");
    styleKnobLabel(memoryLabel, "MEMORY");
    for (auto* l : { &stepLabel, &fideltyLabel, &retrievalLabel, &memoryLabel })
        mainPage.addAndMakeVisible(l);

    for (auto* l : { &stepValueLabel, &fideltyValueLabel, &retrievalValueLabel, &memoryValueLabel })
    {
        styleValueLabel(*l);
        mainPage.addAndMakeVisible(l);
    }

    styleKnob(stepKnob, 10.0, 200.0, 1.0, 50.0);
    styleKnob(fideltyKnob, 1.0, 8.0, 0.1, 3.0);
    styleKnob(retrievalKnob, 0.0, 1.0, 0.01, 0.5);
    styleKnob(memoryKnob, 0.0, 1.0, 0.01, 0.0);
    for (auto* k : { &stepKnob, &fideltyKnob, &retrievalKnob, &memoryKnob })
        mainPage.addAndMakeVisible(k);

    stepKnob.onValueChange = [this]
    {
        stepValueLabel.setText(juce::String((int) stepKnob.getValue()), juce::dontSendNotification);
    };
    fideltyKnob.onValueChange = [this]
    {
        fideltyValueLabel.setText(juce::String(fideltyKnob.getValue(), 2), juce::dontSendNotification);
    };
    retrievalKnob.onValueChange = [this]
    {
        retrievalValueLabel.setText(juce::String(retrievalKnob.getValue(), 2), juce::dontSendNotification);
        applyFidelityDefault();
    };
    memoryKnob.onValueChange = [this]
    {
        memoryValueLabel.setText(juce::String(memoryKnob.getValue(), 2), juce::dontSendNotification);
        refreshKnobAvailability();
        applyFidelityDefault();
        if (memoryKnob.getValue() > 0.0001)
            setStatus("Memory on: generation restarts from the previous preset, Retrieval is locked.", statusWarn);
    };
    stepKnob.onValueChange();
    fideltyKnob.onValueChange();
    retrievalKnob.onValueChange();
    memoryKnob.onValueChange();

    retrievalCategoryToggle.setComponentID("cat");
    retrievalCategoryToggle.setClickingTogglesState(true);

    retrievalCategoryToggle.setToggleState(true, juce::dontSendNotification);
    mainPage.addAndMakeVisible(retrievalCategoryToggle);

    promptPrefixLabel.setText("THE SOUND OF", juce::dontSendNotification);
    promptPrefixLabel.setJustificationType(juce::Justification::centredLeft);
    promptPrefixLabel.setFont(SynthForgeFonts::interBold(10.0f).withExtraKerningFactor(0.15f));
    promptPrefixLabel.setColour(juce::Label::textColourId, textSecondary);
    mainPage.addAndMakeVisible(promptPrefixLabel);

    promptEditor.setMultiLine(false);
    promptEditor.setReturnKeyStartsNewLine(false);
    promptEditor.setFont(SynthForgeFonts::interRegular(13.0f));
    promptEditor.setTextToShowWhenEmpty("a warm analog pad, slow attack...", textDim);
    promptEditor.setColour(juce::TextEditor::backgroundColourId, bgControl);
    promptEditor.setColour(juce::TextEditor::textColourId, textPrimary);
    promptEditor.setColour(juce::TextEditor::outlineColourId, borderDark);
    promptEditor.setColour(juce::TextEditor::focusedOutlineColourId, orangeDarkShade);
    promptEditor.onTextChange = [this] { refreshKnobAvailability(); };
    mainPage.addAndMakeVisible(promptEditor);

    genButton.setComponentID("gen");
    genButton.onClick = [this] { triggerGenerate(); };
    mainPage.addAndMakeVisible(genButton);

    for (size_t i = 0; i < getVocabFields().size(); ++i)
    {
        const auto& field = getVocabFields()[i];
        auto button = std::make_unique<juce::TextButton>(field.buttonLabel);
        button->setComponentID("vocab");
        button->onClick = [this, &field, raw = button.get()] { showVocabMenu(*raw, field); };
        mainPage.addAndMakeVisible(*button);
        vocabButtons[i] = std::move(button);
    }

    forceToggle.setComponentID("force");
    forceToggle.setClickingTogglesState(true);
    forceToggle.setTooltip("Forces the last generated preset to match the chosen TIME and/or SPACE exactly "
                            "(from the menus or detected in the text). Instant: toggle it on and off "
                            "to compare with the original without regenerating.");
    forceToggle.onClick = [this]
    {
        sendCurrentParams();
        setStatus(forceToggle.getToggleState()
                       ? "FORCE on: time and space forced on the last preset."
                       : "FORCE off: original generated values.",
                   statusOk);
    };
    mainPage.addAndMakeVisible(forceToggle);

    uploadButton.setComponentID("upload");
    uploadButton.setTooltip("Load an audio file to use as a starting point or as a target");
    uploadButton.onClick = [this] { chooseAudioFile(); };
    mainPage.addAndMakeVisible(uploadButton);

    undoButton.setComponentID("arrow");
    redoButton.setComponentID("arrow");
    undoButton.setTooltip("Previous preset");
    redoButton.setTooltip("Next preset");
    undoButton.onClick = [this] { applyHistory(historyPos - 1); setStatus("Previous preset.", statusOk); };
    redoButton.onClick = [this] { applyHistory(historyPos + 1); setStatus("Next preset.", statusOk); };
    mainPage.addAndMakeVisible(undoButton);
    mainPage.addAndMakeVisible(redoButton);

    statusLabel.setJustificationType(juce::Justification::centred);
    statusLabel.setFont(SynthForgeFonts::jetBrainsMono(10.0f).withExtraKerningFactor(0.05f));
    mainPage.addAndMakeVisible(statusLabel);

    brandFooterLabel.setText("SynthForge", juce::dontSendNotification);
    brandFooterLabel.setJustificationType(juce::Justification::centred);
    brandFooterLabel.setFont(SynthForgeFonts::rajdhaniSemiBold(16.0f).withExtraKerningFactor(0.25f));
    brandFooterLabel.setColour(juce::Label::textColourId, juce::Colour::fromRGB(0x4a, 0x4a, 0x4a));
    mainPage.addAndMakeVisible(brandFooterLabel);

    backButton.setComponentID("help");
    backButton.onClick = [this] { setShowingHelp(false); };
    helpPage.addAndMakeVisible(backButton);

    helpText.setMultiLine(true);
    helpText.setReadOnly(true);
    helpText.setScrollbarsShown(true);
    helpText.setFont(SynthForgeFonts::interRegular(14.0f));
    helpText.setColour(juce::TextEditor::backgroundColourId, bgControl);
    helpText.setColour(juce::TextEditor::textColourId, textPrimary);
    helpText.setColour(juce::TextEditor::outlineColourId, borderDark);
    helpText.setText(
        "SYNTHFORGE - HOW IT WORKS\n\n"
        "SynthForge turns a text description (or an audio example) into TAL-NoiseMaker "
        "parameters, sent live as MIDI CC.\n\n"
        "STEP\n"
        "Number of integration steps of the generative model (CFM). More steps = higher "
        "quality / smoother result, but slower. Default 50 (100 gives almost the same "
        "preset in twice the time).\n\n"
        "FIDELITY\n"
        "How strongly the result follows your text prompt (classifier-free guidance). "
        "Higher = stricter adherence to the words you typed; lower = more creative "
        "freedom for the model. It resets to the value chosen in the tests whenever "
        "the mode changes: 2 when generating from noise (RETRIEVAL at 0, or an uploaded "
        "sound with MEMORY at maximum), 3 when starting from a preset (RETRIEVAL or "
        "MEMORY above 0).\n\n"
        "RETRIEVAL\n"
        "Active whenever MEMORY is at 0. Controls how much the generation starts from "
        "the closest matching handcrafted preset in the dataset, instead of pure noise. "
        "Low = mostly a fresh generation; high = stays close to that preset, lightly "
        "nudged by your prompt.\n"
        "  CAT button: only clickable if you've typed something or picked a SOUND TYPE. "
        "When lit, restricts the preset search to that category - either the one you "
        "picked explicitly, or one detected inside your free text (e.g. typing \"a deep "
        "bass\" is enough, no need to also pick it from the SOUND TYPE menu).\n\n"
        "MEMORY\n"
        "Locked at 0 until you have generated at least one preset (or loaded an audio "
        "file) in this session - it needs something to start from. Once active, it "
        "restarts generation from your LAST preset instead of a database match, adjusted "
        "by whatever you type next. This also disables RETRIEVAL, since the two starting "
        "points are mutually exclusive.\n\n"
        "UPLOAD SOUND\n"
        "Loads an audio file to use as a reference.\n"
        "  - With MEMORY at 0: the audio is used to find the closest preset (like "
        "RETRIEVAL, but querying by sound instead of by text).\n"
        "  - With MEMORY between 0 and max: the audio is not used - generation restarts "
        "from your last preset, as described under MEMORY.\n"
        "  - With MEMORY at maximum: the audio itself becomes the direct target - the "
        "model generates from scratch trying to match that sound, and no text prompt is "
        "required.\n\n"
        "GEN\n"
        "Runs the generation with the current settings and sends the result to "
        "TAL-NoiseMaker as MIDI CC.\n\n"
        "UNDO / REDO (the arrow buttons)\n"
        "Step back and forward through everything you've generated in this session.\n\n"
        "EXAMPLES\n"
        "  - \"a warm analog pad\" with Memory=0, Retrieval=0.5 -> finds the closest "
        "pad-like preset and lightly regenerates it toward your description.\n"
        "  - Generate something, then type \"brighter, more aggressive\" with Memory=0.6 "
        "-> keeps the character of what you just made, pushed in that direction.\n"
        "  - Upload a kick drum sample with Memory at maximum -> generates a "
        "TAL-NoiseMaker patch that tries to match that kick, no text needed.",
        false);
    helpPage.addAndMakeVisible(helpText);

    refreshKnobAvailability();
    updateHistoryButtons();

    setSize(1100, 440);

    serverReady = SynthForgeClient::isServerReachable();
    if (serverReady)
    {
        setStatus("Ready.", statusOk);
    }
    else
    {
        genButton.setEnabled(false);
        setStatus("Starting the generation engine...", statusWarn);
    }
    startTimer(1000);
}

SynthForgeAudioProcessorEditor::~SynthForgeAudioProcessorEditor()
{
    stopTimer();
    setLookAndFeel(nullptr);
}

void SynthForgeAudioProcessorEditor::timerCallback()
{
    if (serverReady)
    {
        stopTimer();
        return;
    }
    if (SynthForgeClient::isServerReachable())
    {
        serverReady = true;
        genButton.setEnabled(true);
        setStatus("Ready.", statusOk);
        stopTimer();
    }
}

void SynthForgeAudioProcessorEditor::setStatus(const juce::String& message, juce::Colour colour)
{
    statusLabel.setColour(juce::Label::textColourId, colour);
    static const juce::String bullet(juce::CharPointer_UTF8("\xe2\x97\x8f"));
    statusLabel.setText(message.isEmpty() ? juce::String() : (bullet + "  " + message), juce::dontSendNotification);
}

void SynthForgeAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(bgInner);

    paintSectionModule(g, generateBoxBounds, orangeLight, orangeLightShade);
    paintSectionModule(g, sourceBoxBounds, orangeLight, orangeLightShade);
    paintSectionModule(g, describeBoxBounds, orangeDark, orangeDarkShade);

    auto frame = getLocalBounds().toFloat().reduced(2.0f);
    g.setColour(juce::Colours::black);
    g.drawRoundedRectangle(frame, 6.0f, 1.0f);
}

void SynthForgeAudioProcessorEditor::resized()
{
    auto area = getLocalBounds();
    area.removeFromLeft(20);
    area.removeFromRight(20);
    area.removeFromTop(18);
    area.removeFromBottom(14);

    auto brandRow = area.removeFromTop(28);
    helpButton.setBounds(brandRow.removeFromLeft(28));
    brandRow.removeFromRight(28);
    titleLabel.setBounds(brandRow);
    area.removeFromTop(22);

    mainPage.setBounds(area);
    helpPage.setBounds(area);

    auto mainPageOffset = mainPage.getPosition();
    auto main = mainPage.getLocalBounds();

    auto brandFooterArea = main.removeFromBottom(31);
    brandFooterArea.removeFromTop(8);
    brandFooterLabel.setBounds(brandFooterArea);

    auto footerRow = main.removeFromBottom(30);
    undoButton.setBounds(footerRow.removeFromLeft(34));
    redoButton.setBounds(footerRow.removeFromRight(34));
    statusLabel.setBounds(footerRow);

    auto uploadRow = main.removeFromBottom(74);
    uploadRow.removeFromTop(18);
    uploadRow.removeFromBottom(14);
    uploadButton.setBounds(uploadRow.withSizeKeepingCentre(200, uploadRow.getHeight()));

    main.removeFromBottom(10);
    auto topRow = main.removeFromTop(180);

    const int sideW = 320;
    auto generateArea = topRow.removeFromLeft(sideW);
    topRow.removeFromLeft(10);
    auto sourceArea = topRow.removeFromRight(sideW);
    topRow.removeFromRight(10);
    auto describeArea = topRow;

    generateBoxBounds = generateArea.translated(mainPageOffset.x, mainPageOffset.y);
    sourceBoxBounds = sourceArea.translated(mainPageOffset.x, mainPageOffset.y);
    describeBoxBounds = describeArea.translated(mainPageOffset.x, mainPageOffset.y);

    sectionGenerate.setBounds(generateArea.removeFromTop(kHeaderH));
    {
        auto body = generateArea.reduced(16, 20);
        int blockW = (body.getWidth() - 22) / 2;
        auto stepBlock = body.removeFromLeft(blockW);
        body.removeFromLeft(22);
        auto fideltyBlock = body;
        layoutKnobBlock(stepBlock, stepLabel, stepKnob, stepValueLabel);
        layoutKnobBlock(fideltyBlock, fideltyLabel, fideltyKnob, fideltyValueLabel);
    }

    sectionSource.setBounds(sourceArea.removeFromTop(kHeaderH));
    {
        auto body = sourceArea.reduced(16, 20);
        int blockW = 96, catW = 44, gap = 18;
        auto retrievalBlock = body.removeFromLeft(blockW);
        body.removeFromLeft(gap);
        auto catStrip = body.removeFromLeft(catW);
        body.removeFromLeft(gap);
        auto memoryBlock = body;
        layoutKnobBlock(retrievalBlock, retrievalLabel, retrievalKnob, retrievalValueLabel);
        layoutKnobBlock(memoryBlock, memoryLabel, memoryKnob, memoryValueLabel);
        retrievalCategoryToggle.setBounds(catStrip.withSizeKeepingCentre(catW, 24));
    }

    sectionDescribe.setBounds(describeArea.removeFromTop(kHeaderH));
    {
        auto body = describeArea.reduced(20, 18);
        auto promptRow = body.removeFromTop(32);
        genButton.setBounds(promptRow.removeFromRight(70));
        promptRow.removeFromRight(8);
        promptPrefixLabel.setBounds(promptRow.removeFromLeft(90));
        promptRow.removeFromLeft(8);
        promptEditor.setBounds(promptRow);

        body.removeFromTop(12);

        const int cols = 4, rows = 2, gap = 6;
        int cellW = (body.getWidth() - gap * (cols - 1)) / cols;
        int cellH = (body.getHeight() - gap * (rows - 1)) / rows;
        cellH = juce::jmin(cellH, 30);

        for (int i = 0; i < (int) vocabButtons.size(); ++i)
        {
            int r = i / cols, c = i % cols;
            juce::Rectangle<int> cell(body.getX() + c * (cellW + gap),
                                       body.getY() + r * (cellH + gap),
                                       cellW, cellH);
            vocabButtons[(size_t) i]->setBounds(cell);
        }

        {
            int forceIndex = (int) vocabButtons.size();
            int r = forceIndex / cols, c = forceIndex % cols;
            juce::Rectangle<int> cell(body.getX() + c * (cellW + gap),
                                       body.getY() + r * (cellH + gap),
                                       cellW, cellH);
            forceToggle.setBounds(cell);
        }
    }

    auto help = helpPage.getLocalBounds();
    backButton.setBounds(help.removeFromTop(28).removeFromLeft(90));
    help.removeFromTop(8);
    helpText.setBounds(help);
}

void SynthForgeAudioProcessorEditor::setShowingHelp(bool shouldShowHelp)
{
    mainPage.setVisible(! shouldShowHelp);
    helpPage.setVisible(shouldShowHelp);
}

bool SynthForgeAudioProcessorEditor::hasCategorySource() const
{
    if (promptEditor.getText().trim().isNotEmpty())
        return true;
    return choices.getValue("category", {}).isNotEmpty();
}

juce::String SynthForgeAudioProcessorEditor::resolveCategory(const juce::String& promptContinuation) const
{
    auto explicitCategory = choices.getValue("category", {});
    if (explicitCategory.isNotEmpty())
        return explicitCategory;

    static const juce::StringArray categoryOrder { "arp", "bass", "chord", "drum", "fx",
                                                   "keyboard", "lead", "pad", "pluck" };
    for (auto& word : categoryOrder)
        if (promptContinuation.containsWholeWordIgnoreCase(word))
            return word;

    return {};
}

void SynthForgeAudioProcessorEditor::refreshKnobAvailability()
{
    bool hasStartingPoint = ! lastX1.isVoid() || uploadedAudioPath.isNotEmpty();
    memoryKnob.setEnabled(hasStartingPoint);
    if (! hasStartingPoint)
        memoryKnob.setValue(0.0, juce::dontSendNotification);

    bool locked = hasStartingPoint && memoryKnob.getValue() > 0.0001;
    bool catAvailable = hasCategorySource();
    retrievalKnob.setEnabled(! locked);
    retrievalCategoryToggle.setEnabled(! locked && catAvailable);
    retrievalCategoryToggle.setTooltip(catAvailable
        ? "Restrict the retrieval to the category chosen or detected in the text"
        : "Select a category or write a description first");
}

void SynthForgeAudioProcessorEditor::applyFidelityDefault()
{
    bool memoryActive = memoryKnob.isEnabled() && memoryKnob.getValue() > 0.0001;
    bool memoryMaxed = memoryKnob.isEnabled() && memoryKnob.getValue() >= memoryKnob.getMaximum() - 0.001;
    bool hasAudio = uploadedAudioPath.isNotEmpty();
    bool fromNoise = (hasAudio && memoryMaxed)
                  || (! memoryActive && retrievalKnob.getValue() <= 0.0001);
    int mode = fromNoise ? 0 : 1;
    if (mode == fidelityMode)
        return;
    fidelityMode = mode;
    fideltyKnob.setValue(fromNoise ? 2.0 : 3.0, juce::sendNotificationSync);
}

void SynthForgeAudioProcessorEditor::refreshPromptFromChoices()
{
    auto sentence = composeSentence(choices);
    static const juce::String prefix("The sound of ");
    if (sentence.startsWithIgnoreCase(prefix))
        sentence = sentence.substring(prefix.length());
    promptEditor.setText(sentence, juce::dontSendNotification);
    refreshKnobAvailability();
}

void SynthForgeAudioProcessorEditor::showVocabMenu(juce::TextButton& button, const VocabField& field)
{
    juce::PopupMenu menu;
    menu.addItem(1, "- clear -");
    menu.addSeparator();
    int id = 2;
    for (auto& word : field.words)
        menu.addItem(id++, word, true, choices.getValue(field.key, {}) == word);

    menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(button),
        [this, &field, &button, words = field.words](int result)
        {
            if (result == 0)
                return;
            if (result == 1)
            {
                choices.remove(juce::StringRef(field.key));
                button.setButtonText(field.buttonLabel);
                button.setToggleState(false, juce::dontSendNotification);
            }
            else
            {
                auto word = words[result - 2];
                choices.set(field.key, word);
                button.setButtonText(word.toUpperCase());
                button.setToggleState(true, juce::dontSendNotification);
            }
            refreshPromptFromChoices();
        });
}

void SynthForgeAudioProcessorEditor::chooseAudioFile()
{
    fileChooser = std::make_unique<juce::FileChooser>(
        "Choose an audio file", juce::File(), "*.wav;*.aif;*.aiff;*.mp3");

    fileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
        [this](const juce::FileChooser& chooser)
        {
            auto file = chooser.getResult();
            if (file.existsAsFile())
            {
                uploadedAudioPath = file.getFullPathName();
                setStatus("Audio loaded: " + file.getFileName(), statusOk);
                refreshKnobAvailability();
                applyFidelityDefault();
            }
        });
}

void SynthForgeAudioProcessorEditor::triggerGenerate()
{
    auto continuation = promptEditor.getText().trim();
    auto prompt = continuation.isNotEmpty() ? ("The sound of " + continuation) : juce::String();

    auto steps = (int) stepKnob.getValue();
    auto cfg = fideltyKnob.getValue();
    auto memoryValue = memoryKnob.getValue();
    auto retrievalValue = retrievalKnob.getValue();
    bool memoryMaxed = memoryKnob.isEnabled() && memoryValue >= memoryKnob.getMaximum() - 0.001;
    bool memoryActive = memoryKnob.isEnabled() && memoryValue > 0.0001;
    bool hasAudio = uploadedAudioPath.isNotEmpty();

    juce::Component::SafePointer<SynthForgeAudioProcessorEditor> safeThis(this);
    auto dispatch = [safeThis](std::function<SynthForgeClient::Response()> call)
    {
        juce::Thread::launch([safeThis, call]
        {
            auto response = call();
            juce::MessageManager::callAsync([safeThis, response]
            {
                if (auto* editor = safeThis.getComponent())
                    editor->handleGenerateResponse(response);
            });
        });
    };

    if (hasAudio && memoryMaxed)
    {
        genButton.setEnabled(false);
        setStatus("Generating from the loaded audio...", statusWarn);
        auto audioPath = uploadedAudioPath;
        dispatch([audioPath, steps, cfg] { return SynthForgeClient::generateFromAudio(audioPath, steps, cfg); });
        return;
    }

    if (memoryActive)
    {
        if (lastX1.isVoid())
        {
            setStatus("Memory is on but there is no previous preset yet: generate once with Memory at zero.", statusError);
            return;
        }
        if (prompt.isEmpty())
        {
            setStatus("Describe how to change the sound before generating.", statusError);
            return;
        }
        genButton.setEnabled(false);
        setStatus(hasAudio ? "Generating (audio not used: Memory is not at maximum)..."
                           : "Generating...", statusWarn);
        auto previousX1 = lastX1;
        dispatch([prompt, steps, cfg, memoryValue, previousX1]
        {
            return SynthForgeClient::memoryEdit(prompt, steps, cfg, memoryValue, previousX1);
        });
        return;
    }

    if (prompt.isEmpty() && ! hasAudio)
    {
        setStatus("Write a prompt or load an audio file before generating.", statusError);
        return;
    }
    genButton.setEnabled(false);
    setStatus("Generating...", statusWarn);
    juce::String category;
    if (retrievalCategoryToggle.getToggleState())
        category = resolveCategory(continuation);
    auto audioForQuery = hasAudio ? uploadedAudioPath : juce::String();
    constexpr double tau = 0.3;
    dispatch([prompt, audioForQuery, steps, cfg, retrievalValue, category]
    {
        return SynthForgeClient::retrieveAndEdit(prompt, audioForQuery, steps, cfg, retrievalValue, tau, category);
    });
}

void SynthForgeAudioProcessorEditor::handleGenerateResponse(const SynthForgeClient::Response& response)
{
    genButton.setEnabled(true);
    if (! response.ok)
    {
        setStatus("Error: " + response.error, statusError);
        return;
    }

    pushHistory(response);
    lastX1 = response.x1;
    sendCurrentParams();
    refreshKnobAvailability();

    juce::String status = "Preset generated and sent to TAL-NoiseMaker as MIDI CC.";
    if (response.matchedPreset.isNotEmpty())
        status << "  (closest preset: \"" << response.matchedPreset << "\", sim=" << juce::String(response.similarity, 2) << ")";
    setStatus(status, statusOk);
}

void SynthForgeAudioProcessorEditor::pushHistory(const SynthForgeClient::Response& response)
{
    if (historyPos + 1 < (int) history.size())
        history.resize((size_t) historyPos + 1);
    history.push_back({ response.params, response.x1 });
    historyPos = (int) history.size() - 1;
    updateHistoryButtons();
}

void SynthForgeAudioProcessorEditor::applyHistory(int index)
{
    if (index < 0 || index >= (int) history.size())
        return;
    historyPos = index;
    lastX1 = history[(size_t) historyPos].x1;
    sendCurrentParams();
    refreshKnobAvailability();
    updateHistoryButtons();
}

void SynthForgeAudioProcessorEditor::sendCurrentParams()
{
    if (historyPos < 0 || historyPos >= (int) history.size())
        return;

    const auto& baseParams = history[(size_t) historyPos].params;
    juce::var toSend = baseParams;

    if (forceToggle.getToggleState())
    {
        auto continuation = promptEditor.getText().trim();
        auto forceTimeShape = resolveForceValue(choices, "timeShape", continuation);
        auto forceSpace     = resolveForceValue(choices, "space", continuation);
        if (forceTimeShape.isNotEmpty() || forceSpace.isNotEmpty())
            toSend = ForceCanonical::apply(baseParams, forceTimeShape, forceSpace);
    }

    processorRef.sendParameters(toSend);
}

void SynthForgeAudioProcessorEditor::updateHistoryButtons()
{
    undoButton.setEnabled(historyPos > 0);
    redoButton.setEnabled(historyPos < (int) history.size() - 1);
}
