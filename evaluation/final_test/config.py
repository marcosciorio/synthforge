"""Shared settings of the final test: paths, parameter grids and seeds.
"""

import os
import sys

QUI  = os.path.dirname(os.path.abspath(__file__))
VAL  = os.path.dirname(QUI)
ROOT = os.path.dirname(VAL)
sys.path.insert(0, os.path.join(VAL, "common"))
sys.path.insert(0, os.path.join(ROOT, "model", "CLAP"))

import common as C

VARIANTE_ADAPTER = "space"
_KT = os.path.join(ROOT, "model", "kaggle_training_cfm")
_ADAPTER = {
    "attuale": (os.path.join(_KT, "adapter_aug.pt"),       os.path.join(_KT, "adapter_embeddings_aug.npz")),
    "space":   (os.path.join(_KT, "adapter_aug_space.pt"), os.path.join(_KT, "adapter_embeddings_aug_space.npz")),
}
ADAPTER_FINALE, NPZ_ADAPTER = _ADAPTER[VARIANTE_ADAPTER]

DESCRIZIONI = os.path.join(ROOT, "listening_test", "backend", "results_server",
                           "unified_descriptions.json")
TRADUZIONI  = os.path.join(QUI, "free_translations.json")

RISULTATI       = os.environ.get("FINAL_TEST_RESULTS", os.path.join(QUI, "results"))
DIR_FOLD        = os.path.join(RISULTATI, "fold_adapters")
PROMPT_TEST     = os.path.join(RISULTATI, "prompt_test.json")
PROMPT_SVILUPPO = os.path.join(RISULTATI, "prompt_development.json")
CONTROLLO_PASSI = os.path.join(RISULTATI, "step_check.json")
SVILUPPO_JSONL  = os.path.join(RISULTATI, "development.jsonl")
PARAMETRI       = os.path.join(RISULTATI, "selected_parameters.json")
TEST_JSONL      = os.path.join(RISULTATI, "test.jsonl")
DIR_ANALISI     = os.path.join(RISULTATI, "analysis")

N_PASSI           = 50
N_PASSI_CONTROLLO = 100
SOGLIA_PASSI      = 0.01
FORCING           = True
SEED_BASE         = 20261110

N_SVILUPPO     = 100
N_FOLD         = 5
GRIGLIA_S      = [1.0, 2.0, 3.0, 4.0]
GRIGLIA_TSTAR  = [0.3, 0.5, 0.7]

K_SEED_TEST = 3
N_RANDOM    = 5
METODI      = ["diretta", "adapter", "retrieval"]
TESTI       = ["algoritmica", "guidata", "libera"]
N_BOOTSTRAP = 10000

for d in (RISULTATI, DIR_FOLD, DIR_ANALISI):
    os.makedirs(d, exist_ok=True)


def seed_di(indice: int, k: int) -> int:
    return SEED_BASE + 1000 * indice + k
