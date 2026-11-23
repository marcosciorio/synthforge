# Text to Synthesizer Parameters via Conditional Flow Matching

Code of my Master's thesis in Computer Engineering (Sound Engineering) at the University of Naples Federico II, co-supervised at Queen Mary University of London.

The system generates a preset for the TAL-NoiseMaker synthesizer from a text description of the sound, for example *"a warm analog pad"*. A generative model based on conditional flow matching produces the 63 synthesizer parameters from a CLAP audio embedding. At inference time the condition comes from the text: a text adapter maps the CLAP text embedding into the audio embedding space, and generation can also restart from the closest human-made preset in a reference library. **SynthForge**, an AU/VST3 plugin, brings the system into a DAW.

## Repository structure

The folders follow the stages of the project.

| Folder | Content |
|---|---|
| `dataset_creation` | Synthetic preset generation (sampling priors and logical constraints), rendering of the handcrafted presets, Kaggle notebook that builds the training shards with CLAP embeddings |
| `captioning` | Algorithmic captioner with a closed vocabulary, and the analyses behind its thresholds |
| `model` | Training notebook of the flow matching model, text adapter, contrastive adapter experiment, inference server of SynthForge |
| `evaluation` | Evaluation: shared rendering and metrics code, final test scripts (`final_test/p00`–`p12`) |
| `listening_test` | Website used to collect human descriptions of the test sounds (FastAPI backend, React frontend) |
| `plugin` | SynthForge plugin (C++, JUCE): sends the generated parameters to TAL-NoiseMaker as MIDI control-change messages |

## Requirements

- Python 3.11, with the environment in `synth/` at the repository root:
  `python3.11 -m venv synth && synth/bin/pip install -r requirements.txt`
- [TAL-NoiseMaker](https://tal-software.com/products/tal-noisemaker), free VST3/AU synthesizer, installed in the standard plugin folder
- The LAION-CLAP checkpoint `music_audioset_epoch_15_esc_90.14.pt`, placed in `model/kaggle_training_cfm/`
- For the plugin: CMake 3.22 or later and a C++17 compiler; JUCE 8.0.4 is downloaded by CMake

Datasets, trained weights and the Listening Test responses are not included in the repository. The scripts expect the data in `data/Dataset_TAL_48kHz/` and the model files in `model/kaggle_training_cfm/`. The final test also reads the Listening Test descriptions from `listening_test/backend/results_server/unified_descriptions.json` and the English translations of the free descriptions from `evaluation/final_test/free_translations.json`.

## Usage

**Final test.** From `evaluation/final_test`:

```sh
./run_all.sh           # quick run on two presets, then the full run
```

The script runs prompt preparation, fold adapters, step check, development grid, parameter selection, test and analysis. Scripts `p07`–`p12` contain the additional analyses and run separately.

**Plugin.** From `plugin`:

```sh
cmake -B build -DSYNTHFORGE_PROJECT_ROOT="/path/to/this/repository"
cmake --build build --config Release
```

The plugin starts the inference server (`model/kaggle_training_cfm/synthforge_server.py`) with the Python environment in `synth/` under the project root. Load the preset produced by `generate_midi_setup_preset.py` in TAL-NoiseMaker once, so that the control-change messages are mapped to the synthesizer parameters (see `MIDI_CC_MAPPING.md`).

**Listening Test.** Backend: `uvicorn main:app` in `listening_test/backend`. Frontend: `npm install` and `npm run dev` in `listening_test/frontend`.

## Third-party components

- JUCE, under its own licence
- Inter, JetBrains Mono and Rajdhani fonts, embedded in the plugin, under the SIL Open Font License 1.1
- LAION-CLAP, DawDreamer and pedalboard, installed as Python packages
