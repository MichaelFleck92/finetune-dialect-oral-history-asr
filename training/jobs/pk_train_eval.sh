#!/bin/bash
# pk_train_eval.sh NAME ARGS...  ->  parakeet_lora.py mit ARGS, danach dev-Auswertung (Segmente, greedy)
# jedes Epochen-Zwischenstands (bzw. des Laufs bei einer Epoche). Gegenstück zu train_eval.sh (Whisper),
# 16.11.2026. Läuft in der NeMo-Umgebung (requirements-nemo.txt) über PY_NEMO, Standard ~/venv-nemo.
# Logs in training/runs/NAME/.
name=$1; shift
cd "$(dirname "$0")/../.." || exit 1      # Repo-Wurzel
P=${PY_NEMO:-$HOME/venv-nemo/bin/python}
F='NeMo W|NeMo I|warnings.warn|^ +warn|OneLogger|telemetry|CUDACachingAllocator'
mkdir -p "training/runs/$name"
$P training/parakeet_lora.py --name "$name" "$@" 2>&1 | grep --line-buffered -v -E "$F" | tee "training/runs/$name/console.log"
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && { echo "TRAIN FAILED $name ($rc)"; exit "$rc"; }
dirs=$(ls -d training/runs/$name/ep* 2>/dev/null | sort -V)
[ -z "$dirs" ] && dirs="training/runs/$name"
for d in $dirs; do
  $P training/parakeet_eval.py --modus segmente --split dev --adapter "$d" --name "${d#training/runs/}" 2>&1 \
    | grep --line-buffered -v -E "$F" | tee "$d/console_eval_dev.log" | grep --line-buffered -E "^WER|^  [A-Z]" \
    | sed -u "s|^|DEV ${d#training/runs/}: |"
done
