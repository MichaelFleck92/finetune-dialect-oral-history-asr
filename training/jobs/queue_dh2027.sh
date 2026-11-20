#!/bin/bash
# Warteschlange für den DH2027-Vergleich auf dem Datensatz mit längeren Segmenten (Specs von Qwen3-235B,
# Segmente bis 28 s): Whisper large-v3 und Parakeet v3, je ohne und mit LoRA. Einstellungen aus der
# Abstimmung auf dem bisherigen Datensatz übernommen (Whisper lr 5e-4, Parakeet lr 2e-3, je 3 Epochen,
# r = 16), keine neue Abstimmung; test genau einmal mit Epoche 3. Parakeet wegen der bis 28 s langen Segmente
# mit 2 x 8 Segmenten je Schritt und Joint-Teilbatch 1 (längste 2: 15,0 GB; 2 x Teilbatch 2: 20,5 GB, 4 x 1: 21,7 GB).
# Aufruf:  bash training/jobs/queue_dh2027.sh   (PY / PY_NEMO wie in train_eval.sh / pk_train_eval.sh)
cd "$(dirname "$0")/../.." || exit 1      # Repo-Wurzel
if grep -q "w-r16-lr5e-4-e3" training/jobs/queue.txt training/jobs/done.txt 2>/dev/null; then
  echo "== schon eingetragen, nichts geändert"; exit 0
fi
W=${PY:-python3}; N=${PY_NEMO:-$HOME/venv-nemo/bin/python}; M=openai/whisper-large-v3
F='grep --line-buffered -v -i -E "pynvml|NeMo W|NeMo I|warnings.warn|OneLogger|telemetry"'
cat >> training/jobs/queue.txt <<Q
# ohne Feintuning, neue Segmente (dev, test)
$W training/whisper_eval.py --model $M --name baseline-large-v3 --split dev 2>&1 | $F
$W training/whisper_eval.py --model $M --name baseline-large-v3 --split test 2>&1 | $F
$N training/parakeet_eval.py --modus segmente --split dev 2>&1 | $F
$N training/parakeet_eval.py --modus segmente --split test 2>&1 | $F
# LoRA
bash training/jobs/pk_train_eval.sh pk-r16-lr2e-3-e3 --epochs 3 --lr 2e-3 --batch 2 --accum 8 --fused-batch 1
bash training/jobs/train_eval.sh w-r16-lr5e-4-e3 --epochs 3 --lr 5e-4
# test: Segmente und ganze Interviews (Stücke wie bisher, training/longform)
$N training/parakeet_eval.py --modus segmente --split test --adapter training/runs/pk-r16-lr2e-3-e3/ep3 --name pk-r16-lr2e-3-e3/ep3 2>&1 | $F
$N training/parakeet_eval.py --modus interviews --adapter training/runs/pk-r16-lr2e-3-e3/ep3 --name pk-r16-lr2e-3-e3/ep3 2>&1 | $F
$W training/whisper_eval.py --model $M --adapter training/runs/w-r16-lr5e-4-e3/ep3 --name w-r16-lr5e-4-e3/ep3 --split test 2>&1 | $F
$W training/longform_eval.py --model $M --adapter training/runs/w-r16-lr5e-4-e3/ep3 --name w-r16-lr5e-4-e3/ep3 --tag greedy 2>&1 | $F
Q
if ! pgrep -f "training/jobs/queue.sh" >/dev/null; then    # nur nohup im Hintergrund, sonst hält die Sitzung
  nohup bash training/jobs/queue.sh >> training/jobs/queue.log 2>&1 < /dev/null &
fi
echo "Warteschlange läuft; Log: training/jobs/queue.log"
