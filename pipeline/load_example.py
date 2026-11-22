#!/usr/bin/env python3
"""
Example: load the dataset for Whisper fine-tuning.

  pip install "datasets[audio]" transformers
  python3 pipeline/load_example.py

Fields usually needed when loading:
  text           target text, uniform across all projects (without fillers
                 and hesitation marks - this is how the Roter Stern transcripts are written)
  text_verbatim  as in the transcript (Rechnitz contains ahm/mhm and … there)
  split          train / dev / test - fixed, by groups of persons
  quality        hoch / mittel (high / medium)
  role           zeitzeuge / interviewer / weitere (interviewee / interviewer / other)
  consent_anonymise  True / False / None (None = no statement found)
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
    """With the datasets library, 16 kHz, columns audio/text."""
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
