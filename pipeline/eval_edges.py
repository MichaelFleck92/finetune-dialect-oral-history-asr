#!/usr/bin/env python3
"""
Schnittkanten-Regeln abwaegen (nach der Hoerprobe vom 28.09.2026).

Die Hoerprobe an Stand v2 fand die meisten Fehler an den Segmentraendern.
Strengere Kantenregeln kosten aber Material - und wenn sie an absoluten
CTC-Werten haengen, vor allem Dialekt (dieselbe Schieflage wie der v1-Filter).
Dieses Skript rechnet Varianten durch und vergleicht mit Stand v2
(archive/v2_2026-09-27):

  Verlust    behaltene Stunden gegenueber v2 (ohne ausgeschlossene Interviews)
  Dialekt    Verlust bei Interviews mit niedrigem Median-CTC-Wert (< -1,5:
             Dialekt, schwierige Aufnahme) gegenueber Hochsprache (> -0,3)
  Hoerprobe  was mit den 80 bewerteten v2-Segmenten geschaehe

Die Begrenzung des Randpolsters auf die Dekodierung (seit v2.2) ist hier
abgeschaltet: Sie aendert nur Schnittzeiten, keine Woerter, und wuerde sonst
jede Variante gleich verschieben.

  python3 pipeline/eval_edges.py   -> work/eval/kantenregeln.csv
"""
import collections, csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment as SG
from common import ROOT, WORK, load_specs

ARCH = ROOT / "archive/v2_2026-09-27"


def variant(q, cap, absolute, pause):
    """Kantenpruefung als Funktion. pause: 'keine' | 'sprecherwechsel' (ein Wort)
    | 'bis_pause' (bei jedem Rand ohne Pause kuerzen, bis eine Pause kommt)."""
    def trim(seg, aligned, pos, thr, emis, inv, stats):
        for side in (-1, +1):
            cascade = False
            while seg:
                w = seg[0] if side < 0 else seg[-1]
                k = pos[w["i"]] + side
                nb = aligned[k] if 0 <= k < len(aligned) else None
                t = absolute if absolute is not None else thr[w["spk"]]
                bad = w["sc"] < t
                if not bad and nb is not None:
                    if pause == "bis_pause" or (pause == "sprecherwechsel" and not cascade and nb["spk"] != w["spk"]):
                        bad = SG.no_pause(w, nb, side, emis, inv)
                if not bad:
                    break
                seg = seg[1:] if side < 0 else seg[:-1]
                cascade = True
        return seg
    return trim


VARIANTS = [
    ("nur Tokenisierung, Nachbarn gedehnter Wörter", dict(q=None, cap=None, absolute=-1e9, pause="keine")),
    ("CTC absolut < -5, bis zur Pause kürzen", dict(q=None, cap=None, absolute=-5.0, pause="bis_pause")),
    ("CTC absolut < -5, ohne Pausenregel", dict(q=None, cap=None, absolute=-5.0, pause="keine")),
    ("CTC < p10 des Sprechers, bis zur Pause kürzen", dict(q=0.10, cap=-3.0, absolute=None, pause="bis_pause")),
    ("CTC < p10 des Sprechers, ohne Pausenregel", dict(q=0.10, cap=-3.0, absolute=None, pause="keine")),
    ("CTC < p10 des Sprechers, Sprecherwechsel 1 Wort (gewählt)",
     dict(q=0.10, cap=-3.0, absolute=None, pause="sprecherwechsel")),
]


def main():
    specs = {s["id"]: s for s in load_specs()}
    votes = {f: v for f, v in list(csv.reader(open(ARCH / "hoerprobe/bewertung.csv", encoding="utf-8"),
                                               delimiter=";"))[1:]}
    meta = {}
    for l in open(ARCH / "dataset/metadata.jsonl", encoding="utf-8"):
        x = json.loads(l)
        if x["file_name"] in votes:
            meta[x["file_name"]] = x
    old_h = {}
    for f in (ARCH / "work/segments").glob("*.json"):
        S = json.loads(f.read_text())
        old_h[S["id"]] = sum(o["duration"] for o in S["segments"] if o["keep"])
    medsc = {}
    for f in (WORK / "aligned").glob("*.json"):
        A = json.loads(f.read_text())
        medsc[A["id"]] = float(np.median([w["sc"] for w in A["words"] if "sc" in w]))
    ids = [i for i in specs if not specs[i].get("exclude") and old_h.get(i)]
    orig_trim, orig_q, orig_cap = SG.trim_edges, SG.EDGE_Q, SG.EDGE_SC_MAX
    orig_after = SG.speech_after
    SG.speech_after = lambda *a: None                       # Polstergrenze aus (siehe oben)
    rows = []
    for name, v in VARIANTS:
        SG.trim_edges = variant(**v)
        if v["q"] is not None:
            SG.EDGE_Q, SG.EDGE_SC_MAX = v["q"], v["cap"]
        new_h, kept = {}, {}
        for i in ids:
            A, out, stats = SG.build(i)
            k = [o for o in out if SG.keep_rule(o)[0]]
            new_h[i] = sum(o["duration"] for o in k)
            kept[i] = k
        SG.trim_edges, SG.EDGE_Q, SG.EDGE_SC_MAX = orig_trim, orig_q, orig_cap
        loss = {i: 1 - new_h[i] / old_h[i] for i in ids}
        dia = [loss[i] for i in ids if medsc[i] < -1.5]
        std = [loss[i] for i in ids if medsc[i] > -0.3]
        eff = collections.Counter()
        for f, vote in votes.items():
            m = meta[f]
            if m["interview"] not in kept:
                continue
            s, e = m["source_start"], m["source_end"]
            cov = [o for o in kept[m["interview"]] if o["end"] > s + 0.05 and o["start"] < e - 0.05]
            if any(abs(o["start"] - s) < 0.03 and abs(o["end"] - e) < 0.03 for o in cov):
                st = "unveraendert"
            elif not cov:
                st = "entfallen"
            else:
                st = "gekuerzt"
            eff[("passt" if vote == "passt" else "beanstandet", st)] += 1
        rows.append({"variante": name,
                     "stunden": round(sum(new_h.values()) / 3600, 2),
                     "verlust_pct": round(100 * (1 - sum(new_h.values()) / sum(old_h[i] for i in ids)), 1),
                     "verlust_dialekt_pct": round(100 * float(np.mean(dia)), 1),
                     "verlust_hochsprache_pct": round(100 * float(np.mean(std)), 1),
                     "r_dialekt": round(float(np.corrcoef([medsc[i] for i in ids], [loss[i] for i in ids])[0, 1]), 2),
                     **{f"{a}_{b}": eff[(a, b)] for a in ("beanstandet", "passt")
                        for b in ("unveraendert", "gekuerzt", "entfallen")}})
        print(rows[-1], flush=True)
    SG.speech_after = orig_after
    out = WORK / "eval" / "kantenregeln.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("->", out)


if __name__ == "__main__":
    main()
