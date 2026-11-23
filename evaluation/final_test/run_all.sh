#!/bin/zsh
# Runs the final test from start to end: a quick run on two presets, then the full run.
#
#   ./run_all.sh           quick run, then full run
#   ./run_all.sh --quick   quick run only, in results_quick/
#   ./run_all.sh --force   run on battery power as well
#
# Each step resumes from the missing results if the run is interrupted.
set -e
cd "$(dirname "$0")"
PY="$PWD/../../synth/bin/python"

if [[ "$*" != *--force* ]] && ! pmset -g batt | grep -q "AC Power"; then
  echo "[STOP] Running on battery: connect the power adapter (or use --force)."
  exit 1
fi

passo() {
  local nome=$1; shift
  echo "\n==== $nome ($(date '+%H:%M')) ===="
  caffeinate -dims "$@" 2>&1 | tee -a "$LOG"
  [[ ${pipestatus[1]} -eq 0 ]] || { echo "[ERROR] $nome failed, see $LOG"; exit 1; }
}

esegui() {
  export FINAL_TEST_RESULTS="$1"
  mkdir -p "$1"; LOG="$1/log_$(date '+%Y%m%d_%H%M').txt"
  local L=(); [[ -n "$2" ]] && L=(--limit "$2")
  passo "0. prompts"               $PY p00_prepare_prompts.py
  passo "1. fold adapters"         $PY p01_fold_adapters.py
  passo "2. step check"            $PY p02_step_check.py ${2:-20}
  passo "3. development grid"      $PY p03_development_grid.py $L
  passo "4. parameter selection"   $PY p04_select_parameters.py
  passo "5. final test"            $PY p05_final_test.py $L
  passo "6. analysis"              $PY p06_analysis.py
}

echo "#### QUICK RUN (2 presets) ####"
rm -rf results_quick
esegui "$PWD/results_quick" 2
echo "[OK] quick run completed"
[[ "$*" == *--quick* ]] && exit 0

echo "#### FULL RUN ####"
esegui "$PWD/risultati" ""
echo "\n[DONE] Results in results/analysis/ ($(date '+%H:%M'))"
