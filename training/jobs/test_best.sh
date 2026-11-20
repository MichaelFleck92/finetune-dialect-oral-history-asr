#!/bin/bash
# test_best.sh: je Modell den auf dev gewählten Stand (training/select_dev.py) einmal auf test auswerten,
# Segmente und ganze Interviews (greedy). Schon vorhandene test-Ergebnisse werden nicht wiederholt.
# PY / PY_NEMO wie in train_eval.sh / pk_train_eval.sh; LONGFORM_AUDIO: Ordner mit den ganzen Aufnahmen
# (<interview>.wav/.flac), Standard work/audio16k.
cd "$(dirname "$0")/../.." || exit 1      # Repo-Wurzel
W=${PY:-python3}; N=${PY_NEMO:-$HOME/venv-nemo/bin/python}; M=openai/whisper-large-v3
AUD=(); [ -n "$LONGFORM_AUDIO" ] && AUD=(--audio-dir "$LONGFORM_AUDIO")
F='NeMo W|NeMo I|warnings.warn|^ +warn|OneLogger|telemetry|pynvml|CUDACachingAllocator'
$W training/select_dev.py pk-r16-lr w-r16-lr | sed -u "s|^|AUSWAHL |" || exit 1
pk=$($W training/select_dev.py --nur pk-r16-lr) || exit 1
w=$($W training/select_dev.py --nur w-r16-lr) || exit 1

d=training/runs/$pk
[ -f "$d/eval_test_greedy.json" ] || $N training/parakeet_eval.py --modus segmente --split test --adapter "$d" --name "$pk" 2>&1 \
  | grep --line-buffered -v -E "$F" | sed -u "s|^WER|TEST $pk: WER|"
[ -d "$d/longform_greedy" ] || $N training/parakeet_eval.py --modus interviews --adapter "$d" --name "$pk" "${AUD[@]}" 2>&1 \
  | grep --line-buffered -v -E "$F"

d=training/runs/$w
[ -f "$d/eval_test.json" ] || $W training/whisper_eval.py --model $M --adapter "$d" --name "$w" --split test 2>&1 \
  | grep --line-buffered -v -E "$F" | sed -u "s|^WER|TEST $w: WER|"
[ -d "$d/longform_greedy" ] || $W training/longform_eval.py --model $M --adapter "$d" --name "$w" --tag greedy "${AUD[@]}" 2>&1 \
  | grep --line-buffered -v -E "$F"
echo "TEST_BEST fertig: Parakeet $pk, Whisper $w"
