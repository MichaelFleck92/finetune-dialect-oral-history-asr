#!/usr/bin/env python3
"""
Names on complete interviews: learned vocabulary or generalisation?

  python3 training/longform_train_overlap.py baseline-large-v3:greedy dh-w-r16-lr2e-4-e3/ep2:greedy \
      parakeet-tdt-0.6b-v3:greedy dh-pk-r16-lr2e-3-e3/ep3:greedy

Runs in pairs (base model, fine-tuned model), mentions and recognition as in
longform_entities.py (names and dates, without "other"). Per mention: does its word sequence
(normalised like the WER) occur in the training text (dataset/metadata.jsonl, split train, field
text - what the model saw in training)? Then the recognition rate per pair separately
for "seen" / "never seen" and how many of the mentions that only the fine-tuned model
gets right were in the training data. Only figures are printed.
-> training/longform/training_ueberlappung.json
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import ROOT, words
from longform_entities import collect

MAXN = 8                                   # longest mention searched for, in words


def train_ngrams():
    grams = Counter()
    for line in (ROOT / "dataset" / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["split"] != "train":
            continue
        w = words(r["text"])
        for n in range(1, MAXN + 1):
            for i in range(len(w) - n + 1):
                grams[tuple(w[i:i + n])] += 1
    return grams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="<lauf>:<tag> paarweise: Ausgangsmodell, feingetuntes Modell, ...")
    a = ap.parse_args()
    if len(a.runs) % 2:
        sys.exit("Läufe paarweise angeben")
    runs = [tuple(r.split(":")) if ":" in r else (r, "") for r in a.runs]
    mentions, _, _ = collect(runs)
    core = [m for m in mentions if m["typ"] != "MISC"]
    grams = train_ngrams()
    seen = [grams[tuple(words(m["text"]))] > 0 for m in core]
    out = {"nennungen": len(core), "gesehen": sum(seen), "paare": {}}
    print(f"Namen und Daten: {len(core)}, davon im Training gesehen {sum(seen)} ({100 * sum(seen) / len(core):.1f} %)")
    for p in range(0, len(runs), 2):
        a0, b0 = p, p + 1
        name = f"{a.runs[a0]} -> {a.runs[b0]}"
        res = {}
        for s, lab in [(True, "gesehen"), (False, "nie gesehen")]:
            ms = [m for m, x in zip(core, seen) if x == s]
            ra = sum(m["erkannt"][a0] for m in ms) / len(ms)
            rb = sum(m["erkannt"][b0] for m in ms) / len(ms)
            res[lab] = {"n": len(ms), "vorher": round(ra, 4), "nachher": round(rb, 4)}
            print(f"{name}  {lab:12} n={len(ms):4}  {100 * ra:5.1f} -> {100 * rb:5.1f} %")
        only_b = [x for m, x in zip(core, seen) if m["erkannt"][b0] and not m["erkannt"][a0]]
        only_a = [x for m, x in zip(core, seen) if m["erkannt"][a0] and not m["erkannt"][b0]]
        res["nur_feingetunt"] = {"n": len(only_b), "im_training": sum(only_b)}
        res["nur_ausgangsmodell"] = {"n": len(only_a), "im_training": sum(only_a)}
        print(f"{name}  nur feingetunt erkannt: {len(only_b)}, davon im Training {sum(only_b)}; "
              f"nur Ausgangsmodell: {len(only_a)}, davon im Training {sum(only_a)}")
        out["paare"][name] = res
    path = ROOT / "training" / "longform" / "training_ueberlappung.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
