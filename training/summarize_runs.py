#!/usr/bin/env python3
"""
All evaluations in training/runs/ as a table (one row per eval_*.json).

  python3 training/summarize_runs.py --split dev
  python3 training/summarize_runs.py --split dev --csv training/runs/uebersicht_dev.csv

Columns: run (folder, epoch checkpoints as <lauf>/ep<k>), decoding,
WER/CER overall and per project, plus the training settings from lora.json
(empty for the base model). Sorted by WER.
"""
import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import RUNS

CFG = ["epoch", "epochs", "lr", "r", "alpha", "scope", "spec_augment", "splits", "seed", "dev_loss_after"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    rows = []
    for f in sorted(RUNS.rglob(f"eval_{a.split}*.json")):
        e = json.loads(f.read_text(encoding="utf-8"))
        lj = f.parent / "lora.json"
        cfg = json.loads(lj.read_text(encoding="utf-8")) if lj.exists() else {}
        tag = f.stem[len(f"eval_{a.split}"):].lstrip("_")
        row = {"lauf": str(f.parent.relative_to(RUNS)), "tag": tag, "dekodierung": e.get("decoding", ""),
               "wer": e["gesamt"]["wer"], "cer": e["gesamt"]["cer"]}
        for p, s in e["je_projekt"].items():
            row[f"wer {p}"] = s["wer"]
        for k in CFG:
            v = cfg.get(k)
            row[k] = "+".join(v) if isinstance(v, list) else v
        rows.append(row)
    if not rows:
        sys.exit(f"keine eval_{a.split}*.json in {RUNS}")
    rows.sort(key=lambda r: r["wer"])
    cols = ["lauf", "tag", "wer", "cer"] + sorted({k for r in rows for k in r if k.startswith("wer ")}) + CFG
    fmt = lambda k, v: "" if v is None else f"{100*v:.1f}" if k.startswith(("wer", "cer")) else str(v)
    width = {k: max(len(k), *(len(fmt(k, r.get(k))) for r in rows)) for k in cols}
    print("  ".join(k.ljust(width[k]) for k in cols))
    for r in rows:
        print("  ".join(fmt(k, r.get(k)).ljust(width[k]) for k in cols))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols + ["dekodierung"], extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"-> {a.csv}")


if __name__ == "__main__":
    main()
