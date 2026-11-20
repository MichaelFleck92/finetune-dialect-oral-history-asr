#!/bin/bash
# Lernraten auf den längeren Segmenten wählen (nur dev), danach test einmal je Modell für den gewählten Stand.
# Schon vorhanden (queue_dh2027.sh): Whisper lr 5e-4, Parakeet lr 2e-3, je 3 Epochen mit dev nach jeder Epoche.
# Neu: Whisper 2e-4 und 1e-3; Parakeet 1e-3 und 4e-3 (auf dem früheren Datensatz lag das Parakeet-Optimum
# zwischen 1e-3 und 2e-3 am oberen Rand des Rasters). Sonst alles wie bisher (r = 16, 16 Segmente je Schritt).
# Wahl: niedrigste dev-WER über Lernrate x Epoche (training/select_dev.py), dann jobs/test_best.sh.
# Dauer auf einer 24-GB-GPU: Parakeet je ~1 h 40 min, Whisper je ~2 h 40 min, test ~15 min - zusammen ~9 h.
# Aufruf:  bash training/jobs/queue_lr_dh2027.sh   (PY / PY_NEMO / LONGFORM_AUDIO wie in test_best.sh)
cd "$(dirname "$0")/../.." || exit 1      # Repo-Wurzel
if grep -q "w-r16-lr2e-4-e3" training/jobs/queue.txt training/jobs/done.txt 2>/dev/null; then
  echo "schon eingetragen, nichts geändert"; exit 0
fi
cat >> training/jobs/queue.txt <<'Q'
# Lernraten-Suche auf den längeren Segmenten (dev)
bash training/jobs/pk_train_eval.sh pk-r16-lr1e-3-e3 --epochs 3 --lr 1e-3 --batch 2 --accum 8 --fused-batch 1
bash training/jobs/pk_train_eval.sh pk-r16-lr4e-3-e3 --epochs 3 --lr 4e-3 --batch 2 --accum 8 --fused-batch 1
bash training/jobs/train_eval.sh w-r16-lr2e-4-e3 --epochs 3 --lr 2e-4
bash training/jobs/train_eval.sh w-r16-lr1e-3-e3 --epochs 3 --lr 1e-3
# test einmal für den auf dev gewählten Stand je Modell
bash training/jobs/test_best.sh
Q
if ! pgrep -f "training/jobs/queue.sh" >/dev/null; then    # nur nohup im Hintergrund, sonst hält die Sitzung
  nohup bash training/jobs/queue.sh >> training/jobs/queue.log 2>&1 < /dev/null &
fi
echo "Warteschlange läuft; Log: training/jobs/queue.log"
