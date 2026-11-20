#!/bin/bash
# train_eval.sh NAME ARGS...  ->  whisper_lora.py (large-v3) mit ARGS, danach dev-Auswertung
# jedes Epochen-Zwischenstands (bzw. des Laufs bei einer Epoche). Logs in training/runs/NAME/.
# Python der Whisper-Umgebung (requirements.txt) über PY, Standard python3.
name=$1; shift
cd "$(dirname "$0")/../.." || exit 1      # Repo-Wurzel
P=${PY:-python3}
M=openai/whisper-large-v3
mkdir -p "training/runs/$name"
$P training/whisper_lora.py --model $M --name "$name" "$@" 2>&1 | grep --line-buffered -v -i pynvml | tee "training/runs/$name/console.log"
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && { echo "TRAIN FAILED $name ($rc)"; exit "$rc"; }
dirs=$(ls -d training/runs/$name/ep* 2>/dev/null | sort -V)
[ -z "$dirs" ] && dirs="training/runs/$name"
for d in $dirs; do
  $P training/whisper_eval.py --model $M --adapter "$d" --name "${d#training/runs/}" --split dev 2>&1 \
    | grep --line-buffered -v -i pynvml | tee "$d/console_eval_dev.log" | grep --line-buffered -E "^WER|^  [A-Z]" \
    | sed -u "s|^|DEV ${d#training/runs/}: |"
done
