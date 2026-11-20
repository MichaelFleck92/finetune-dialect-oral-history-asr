#!/usr/bin/env python3
"""
Ganze Interviews auswerten: Ausgabe je Stück gegen die Transkriptwörter,
deren Zeit (Alignment, work/aligned) in das Stück fällt.

  python3 training/longform_score.py baseline-large-v3:beam5fb l3-r16-lr5e-4-e3/ep3:beam5fb ...

Läufe als <lauf>:<tag> (Ordner training/runs/<lauf>/longform_<tag>/). Der erste
Lauf ist die Bezugsgröße für Unterschiede und Bootstrap.
  alle     WER über alle Stücke mit Transkriptwörtern; Sprache, die das Transkript
           auslässt (Joker im Alignment), zählt hier als Fehler - also streng.
  sauber   nur Stücke ohne ausgelassene Sprache (kein Joker überlappt das Stück
           um mehr als 0,3 s) - der faire Vergleich.
  ohne Transkript: Wörter, die ein Modell in Stücke ganz ohne Transkriptwörter
           schreibt (nicht transkribierte Sprache oder erfundener Text).
Bootstrap über die Interviews (95 %) für den Unterschied zum ersten Lauf.
-> training/longform/ergebnis.json. Ausgegeben werden nur Zahlen.
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import ROOT, RUNS, words

LF = ROOT / "training" / "longform"
PROJ = {"rs": "Roter Stern", "re": "Rechnitz", "za": "Zwangsarbeiter"}


def reference(iv, audio_s):
    """Transkriptwörter mit Zeitpunkt (Mitte) und die Zeitspannen ausgelassener Sprache."""
    a = json.loads((ROOT / "work" / "aligned" / f"{iv}.json").read_text(encoding="utf-8"))
    ws = a["words"]
    t, last = [], 0.0
    for w in ws:                                        # nicht alignierte Wörter: Zeit des Vorgängers
        if w.get("s") is not None and w.get("e") is not None:
            last = (w["s"] + w["e"]) / 2
        t.append(last)
    gaps = []
    for j in a.get("stars", []):
        g0 = ws[j - 1]["e"] if j > 0 and ws[j - 1].get("e") is not None else (0.0 if j == 0 else t[j - 1])
        g1 = ws[j]["s"] if j < len(ws) and ws[j].get("s") is not None else (audio_s if j >= len(ws) else t[j])
        gaps.append((g0, g1))
    return [(t[i], words(w["w"])) for i, w in enumerate(ws)], gaps


def score_run(run):
    name, tag = run.split(":") if ":" in run else (run, "")
    d = RUNS / name / ("longform" + (f"_{tag}" if tag else ""))
    per = {}
    for f in sorted((LF / "chunks").glob("*.json")):
        iv = f.stem
        if not (d / f"{iv}.json").exists():           # Lauf (noch) ohne dieses Interview
            continue
        meta = json.loads(f.read_text(encoding="utf-8"))
        out = json.loads((d / f"{iv}.json").read_text(encoding="utf-8"))["chunks"]
        ref, gaps = reference(iv, meta["audio_s"])
        mids = np.array([m for m, _ in ref])
        r = {"alle": [0, 0], "sauber": [0, 0], "ohne_transkript_woerter": 0, "stuecke": len(out), "stuecke_sauber": 0}
        for s, e, text in out:
            rw = [w for (m, ws) in ref if s <= m < e for w in ws] if ((mids >= s) & (mids < e)).any() else []
            hw = words(text)
            if not rw:
                r["ohne_transkript_woerter"] += len(hw)
                continue
            err = Levenshtein.distance(rw, hw)
            r["alle"][0] += err; r["alle"][1] += len(rw)
            if not any(min(e, g1) - max(s, g0) > 0.3 for g0, g1 in gaps):
                r["sauber"][0] += err; r["sauber"][1] += len(rw); r["stuecke_sauber"] += 1
        per[iv] = r
    return per


def main():
    runs = sys.argv[1:]
    if not runs:
        sys.exit(__doc__)
    res = {run: score_run(run) for run in runs}
    ivs = sorted(set.intersection(*(set(per) for per in res.values())))
    res = {run: {k: per[k] for k in ivs} for run, per in res.items()}
    rate = lambda per, keys, kind: sum(per[k][kind][0] for k in keys) / max(1, sum(per[k][kind][1] for k in keys))
    out = {"laeufe": runs, "interviews": ivs, "je_lauf": {}}
    base = res[runs[0]]
    rng = np.random.default_rng(1)
    picks = [rng.choice(ivs, len(ivs)) for _ in range(10000)]
    first = next(iter(res.values()))
    print(f"Ganze Interviews: {len(ivs)} Test-Interviews, {sum(v['stuecke'] for v in first.values())} Stücke, "
          f"davon {sum(v['stuecke_sauber'] for v in first.values())} sauber; "
          f"Transkriptwörter alle {sum(v['alle'][1] for v in first.values())}, sauber {sum(v['sauber'][1] for v in first.values())}")
    print(f"{'Lauf':42} {'WER alle':>9} {'WER sauber':>11} {'Diff sauber (95 %)':>24} {'Wörter ohne Transkript':>23}")
    for run, per in res.items():
        row = {"alle": rate(per, ivs, "alle"), "sauber": rate(per, ivs, "sauber"),
               "ohne_transkript_woerter": sum(per[k]["ohne_transkript_woerter"] for k in ivs),
               "je_projekt": {PROJ[p]: {kind: rate(per, [k for k in ivs if k.startswith(p + "_")], kind) for kind in ("alle", "sauber")}
                              for p in PROJ if any(k.startswith(p + "_") for k in ivs)},
               "je_interview": {k: {kind: per[k][kind][0] / max(1, per[k][kind][1]) for kind in ("alle", "sauber")} for k in ivs}}
        diff = ""
        if run != runs[0]:
            dd = [rate(base, p, "sauber") - rate(per, p, "sauber") for p in picks]
            row["gewinn_sauber"] = rate(base, ivs, "sauber") - row["sauber"]
            row["gewinn_sauber_ci95"] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
            diff = f"{100*row['gewinn_sauber']:+.1f} ({100*row['gewinn_sauber_ci95'][0]:+.1f} bis {100*row['gewinn_sauber_ci95'][1]:+.1f})"
        out["je_lauf"][run] = row
        print(f"{run:42} {100*row['alle']:8.1f}% {100*row['sauber']:10.1f}% {diff:>24} {row['ohne_transkript_woerter']:23d}")
    print("je Projekt (WER sauber):")
    for run in runs:
        print(f"  {run:40} " + "  ".join(f"{p} {100*v['sauber']:.1f} %" for p, v in out["je_lauf"][run]["je_projekt"].items()))
    (LF / "ergebnis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {LF / 'ergebnis.json'}")


if __name__ == "__main__":
    main()
