cmake_minimum_required(VERSION 3.22)

project(SynthForge VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment target")

include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 8.0.4
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(JUCE)

set(SYNTHFORGE_PROJECT_ROOT "" CACHE PATH "Root folder of the project (Python environment and inference server)")

juce_add_plugin(SynthForge
    COMPANY_NAME "QMUL"
    PLUGIN_MANUFACTURER_CODE Qmul
    PLUGIN_CODE Sfg1
    FORMATS AU VST3 Standalone
    PRODUCT_NAME "SynthForge"
    IS_SYNTH FALSE
    IS_MIDI_EFFECT TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    COPY_PLUGIN_AFTER_BUILD TRUE
)

target_sources(SynthForge PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
)

target_compile_definitions(SynthForge PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
)

if(SYNTHFORGE_PROJECT_ROOT)
    target_compile_definitions(SynthForge PRIVATE SYNTHFORGE_PROJECT_ROOT="${SYNTHFORGE_PROJECT_ROOT}")
endif()

juce_add_binary_data(SynthForgeFonts SOURCES
    assets/fonts/Rajdhani-Bold.ttf
    assets/fonts/Rajdhani-SemiBold.ttf
    assets/fonts/Inter-Regular.ttf
    assets/fonts/Inter-SemiBold.ttf
    assets/fonts/Inter-Bold.ttf
    assets/fonts/JetBrainsMono-Regular.ttf
)

target_link_libraries(SynthForge PRIVATE
    SynthForgeFonts
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_gui_extra
    PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)
