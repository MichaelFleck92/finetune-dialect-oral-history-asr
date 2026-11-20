#!/usr/bin/env python3
"""
Beispiel: den Datensatz fuer ein Whisper-Feintuning laden.

  pip install "datasets[audio]" transformers
  python3 pipeline/load_example.py

Felder, die man beim Laden meist braucht:
  text           Zieltext, einheitlich ueber alle Projekte (ohne Fuellwoerter
                 und Zoegerungszeichen - so schreiben die Roter-Stern-Transkripte)
  text_verbatim  wie im Transkript (Rechnitz enthaelt dort ahm/mhm und …)
  split          train / dev / test - fest, nach Personengruppen
  quality        hoch / mittel
  role           zeitzeuge / interviewer / weitere
  consent_anonymise  True / False / None (None = keine Angabe gefunden)
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "dataset"


def load_plain(split=None, min_quality="mittel", role=None, drop_anonymise=False, text_field="text"):
    order = {"mittel": 1, "hoch": 2}
    rows = []
    for line in (DS / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if split and r["split"] != split:
            continue
        if order[r["quality"]] < order[min_quality]:
            continue
        if role and r["role"] != role:
            continue
        if drop_anonymise and r["consent_anonymise"]:
            continue
        r["audio"] = str(DS / r["file_name"])
        r["target"] = r[text_field]
        rows.append(r)
    return rows


def load_hf(split="train", **kw):
    """Mit der datasets-Bibliothek, 16 kHz, Spalten audio/text."""
    from datasets import Dataset, Audio
    rows = load_plain(split, **kw)
    ds = Dataset.from_list([{"audio": r["audio"], "text": r["target"], "role": r["role"],
                             "interview": r["interview"], "duration": r["duration"]} for r in rows])
    return ds.cast_column("audio", Audio(sampling_rate=16000))


if __name__ == "__main__":
    for sp in ("train", "dev", "test"):
        rows = load_plain(sp)
        h = sum(r["duration"] for r in rows) / 3600
        print(f"{sp:5} {len(rows):6} Segmente {h:6.2f} h  aus {len({r['interview'] for r in rows}):3} Interviews")
    r = load_plain("train")[0]
    print("\nBeispielzeile:")
    for k in ("file_name", "text", "text_verbatim", "duration", "role", "quality", "interview", "split"):
        print(f"  {k:14} {r[k]!r}")
