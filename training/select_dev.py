#!/usr/bin/env python3
"""
Selection on dev (18.11.2026): among all epoch checkpoints of the runs whose name starts with the prefix,
the checkpoint with the lowest dev WER (segments, greedy). On a tie (to 0.01 percentage points)
the checkpoint with fewer epochs wins, then the smaller learning rate. The rule is fixed before the results;
test is evaluated only for the chosen checkpoint (jobs/test_best.sh).

  python3 training/select_dev.py pk-r16-lr w-r16-lr        # table per prefix, -> training/runs/auswahl_dev.json
  python3 training/select_dev.py --nur pk-r16-lr           # only the chosen checkpoint (for scripts)

Reads training/runs/<lauf>/ep<k>/eval_dev.json (Whisper) or eval_dev_greedy.json (Parakeet).
"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import RUNS

FILES = ("eval_dev.json", "eval_dev_greedy.json")


def lr_of(name):
    m = re.search(r"-lr([0-9.]+e-?[0-9]+)", name)
    return float(m.group(1)) if m else float("inf")


def candidates(prefix):
    out = []
    for run in sorted(RUNS.glob(prefix + "*")):
        for ep in sorted(run.glob("ep*")):
            f = next((ep / n for n in FILES if (ep / n).exists()), None)
            if f is None:
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            out.append({"stand": f"{run.name}/{ep.name}", "lr": lr_of(run.name), "epoche": int(ep.name[2:]),
                        "wer": d["gesamt"]["wer"], "je_projekt": {p: v["wer"] for p, v in d.get("je_projekt", {}).items()}})
    return out


def best(cands):
    return min(cands, key=lambda c: (round(100 * c["wer"], 2), c["epoche"], c["lr"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="*")
    ap.add_argument("--nur", default=None, help="Präfix; gibt nur den gewählten Stand aus")
    a = ap.parse_args()
    if a.nur:
        c = candidates(a.nur)
        if not c:
            sys.exit(f"keine dev-Ergebnisse für {a.nur}*")
        print(best(c)["stand"])
        return
    res = {}
    for p in a.prefixes:
        c = candidates(p)
        if not c:
            print(f"{p}*: keine dev-Ergebnisse")
            continue
        b = best(c)
        res[p] = {"gewaehlt": b["stand"], "kandidaten": c}
        print(f"{p}*: {len(c)} Stände")
        for x in sorted(c, key=lambda x: (x["lr"], x["epoche"])):
            proj = "  ".join(f"{k} {100 * v:.1f} %" for k, v in x["je_projekt"].items())
            print(f"  {'*' if x is b else ' '} {x['stand']:28} dev WER {100 * x['wer']:6.2f} %   {proj}")
    if res:
        (RUNS / "auswahl_dev.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
