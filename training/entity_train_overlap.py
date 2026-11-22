#!/usr/bin/env python3
"""
Has the fine-tuned model learned the names from the training transcripts - or
does it also recognise names that never occur there?

  python3 training/entity_train_overlap.py --a baseline-large-v3 --b l3-r16-lr5e-4-e3/ep3 --tag beam5fb

Reads the mentions from entity_eval.py (training/entities/nennungen_<...>.csv) and
the transcripts of the training part (dataset/metadata.jsonl, field text - what
the model saw in training). Per mention: does the word sequence of the entity
(normalised like the WER) occur in the training text, and how often? Then the
recognition rate of both models separately for "seen in training" / "never seen"
and by frequency, and how many of the mentions that only the fine-tuned model
gets right were in the training data. Only figures are printed.
-> training/entities/training_ueberlappung_<split>[_<tag>]_<a>_vs_<b>.json
"""
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import ROOT, DATASET, words

ENT = ROOT / "training" / "entities"
NAMES = {"PER": "Person", "LOC": "Ort", "ORG": "Organisation", "DATUM": "Datum", "MISC": "Sonstiges"}
BINS = [("nie gesehen", 0, 0), ("1-4 mal", 1, 4), ("5-49 mal", 5, 49), ("50+ mal", 50, 10 ** 9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="baseline-large-v3")
    ap.add_argument("--b", default="l3-r16-lr5e-4-e3/ep3")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="beam5fb")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    stem = f"{a.split}{sfx}_{a.a}_vs_{a.b}".replace("/", "-")
    rows = list(csv.DictReader(open(ENT / f"nennungen_{stem}.csv", encoding="utf-8"), delimiter=";"))
    ka, kb = f"erkannt_{a.a}", f"erkannt_{a.b}"

    # word sequences of the entities and how often they occur in the training text
    seqs = {r["text"]: tuple(words(r["text"])) for r in rows}
    lengths = {len(s) for s in seqs.values() if s}
    wanted = set(seqs.values())
    count = Counter()
    n_train = 0
    for line in (DATASET / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        if m["split"] != "train":
            continue
        n_train += 1
        w = words(m["text"])
        for n in lengths:
            for i in range(len(w) - n + 1):
                g = tuple(w[i:i + n])
                if g in wanted:
                    count[g] += 1

    def freq(r):
        return count[seqs[r["text"]]]

    def table(ms, title):
        out = {}
        print(f"\n{title} ({len(ms)} Nennungen)")
        print(f"  {'im Training':14} {'n':>5}  {'erkannt A':>9} {'erkannt B':>9} {'Gewinn':>8}")
        for name, lo, hi in BINS + [("gesehen (1+)", 1, 10 ** 9)]:
            sub = [r for r in ms if lo <= freq(r) <= hi]
            if not sub:
                continue
            ra = sum(int(r[ka]) for r in sub) / len(sub)
            rb = sum(int(r[kb]) for r in sub) / len(sub)
            out[name] = {"n": len(sub), "erkannt_a": ra, "erkannt_b": rb}
            print(f"  {name:14} {len(sub):5}  {100*ra:8.1f}% {100*rb:8.1f}% {100*(rb-ra):+7.1f}")
        return out

    core = [r for r in rows if r["typ"] != "MISC"]
    res = {"a": a.a, "b": a.b, "split": a.split, "tag": a.tag, "train_segmente": n_train}
    print(f"{a.a} (A) gegen {a.b} (B), {a.split}{' ' + a.tag if a.tag else ''}; Trainingstext: {n_train} Segmente")
    res["namen_und_daten"] = table(core, "Namen und Daten")
    for t in ("PER", "LOC", "ORG", "DATUM"):
        res[NAMES[t]] = table([r for r in core if r["typ"] == t], NAMES[t])
    res["sonstiges"] = table([r for r in rows if r["typ"] == "MISC"], "Sonstiges")

    # the mentions that only one model gets right
    print("\nNur von einem Modell getroffen (Namen und Daten):")
    res["nur_eines"] = {}
    for who, cond in (("nur B", lambda r: r[kb] == "1" and r[ka] == "0"), ("nur A", lambda r: r[ka] == "1" and r[kb] == "0"),
                      ("beide", lambda r: r[ka] == "1" and r[kb] == "1"), ("keines", lambda r: r[ka] == "0" and r[kb] == "0")):
        sub = [r for r in core if cond(r)]
        seen = [r for r in sub if freq(r) > 0]
        per_type = defaultdict(lambda: [0, 0])
        for r in sub:
            per_type[r["typ"]][0] += 1
            per_type[r["typ"]][1] += freq(r) > 0
        res["nur_eines"][who] = {"n": len(sub), "gesehen": len(seen), "verschiedene_schreibungen": len({r["text"] for r in sub}),
                                 "je_typ": {NAMES[t]: {"n": v[0], "gesehen": v[1]} for t, v in per_type.items()}}
        print(f"  {who:7} {len(sub):4} Nennungen ({len({r['text'] for r in sub})} Schreibungen): im Training gesehen {len(seen):4} "
              f"({100*len(seen)/max(1, len(sub)):.0f} %), nie gesehen {len(sub) - len(seen):4}  | "
              + ", ".join(f"{NAMES[t]} {v[1]}/{v[0]} gesehen" for t, v in sorted(per_type.items())))
    out = ENT / f"training_ueberlappung_{stem}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
