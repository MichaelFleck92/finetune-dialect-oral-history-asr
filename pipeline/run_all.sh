#!/bin/sh
# Ganze Kette in der richtigen Reihenfolge. Jede Stufe ueberspringt, was schon
# aktuell ist (Hashes von Eingabe und Code), und rechnet nur Geaendertes neu.
# Die Emissionen (Stufe 3a) sind der teure Schritt: ~4 h fuer 72 h Audio, CPU.
set -e
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
python3 pipeline/draft_specs.py --check         # 0  Dateien aus config/ pruefen
python3 pipeline/llm_specs.py                   # 0c Specs von Qwen3-235B (Qwen-API, QWEN_API_BASE), vom Parser geprueft
python3 pipeline/ocr_zwangsarbeiter.py          # 0b abfotografierte Transkripte -> Text (nur Zwangsarbeiter)
python3 pipeline/parse_transcripts.py           # 1  Transkript + Spec -> Turns (wortgetreu geprueft)
python3 pipeline/convert_audio.py               # 2  Audio -> 16 kHz mono
python3 pipeline/emissions.py                   # 3a wav2vec2-Emissionen (teuer, zwischengespeichert)
python3 pipeline/align.py                       # 3b Forced Alignment ohne Handmarken
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 pipeline/eval_markers.py --csv work/eval/marker_eval.csv   # 3c Genauigkeit gegen Handmarken; grobe Faelle: Whisper-large-v3 + Hoerprobe (--hoerprobe)
python3 pipeline/segment.py                     # 4  Wortzeiten -> Segmente + Guete
python3 pipeline/export_dataset.py              # 5  Schneiden, Metadaten, Aufteilung, Manifest
python3 pipeline/check_dataset.py               # 6  Abschlusskontrolle (jede Datei bitgenau)
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 pipeline/eval_segments.py --n 500 --n-drop 150 --shift 150   # 7 unabhaengige Pruefung
python3 pipeline/eval_edges.py                  # 7b Kantenregeln: Verlust, Dialekt, Wirkung auf die bewertete Hoerprobe
python3 pipeline/make_samples.py --n 80 --seed 28   # 8  Hoerprobe mit Bewertung (Stand v2: seed 7, archiviert)
python3 pipeline/report.py > work/eval/report.md   # 9  alle Kennzahlen fuer README und Kapitel
