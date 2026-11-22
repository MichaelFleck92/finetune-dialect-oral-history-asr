#!/usr/bin/env python3
"""
Why is the gain on complete interviews smaller than on the test segments?
Breaks down the "clean" chunks from longform_score.py (figures only):

  python3 training/longform_analysis.py baseline-large-v3:beam5fb l3-r16-lr5e-4-e3/ep3:beam5fb

1. the same words: every word of a test segment that lies in a clean chunk,
   scored once as a segment (predictions_test_<tag>.csv) and once in the
   complete interview - wrong or missing, plus insertions between two words of
   the same segment; by segment length.
2. clean chunks by origin of the words (wholly / partly / not from test segments),
   number of speakers and chunk length.
3. error types (wrong / missing / inserted), errors by position in the chunk
   (training only knows segments up to 20.4 s), insertions at the edges (1 s each)
   and in the middle per minute.
-> training/longform/analyse.json
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import ROOT, RUNS, words

LF = ROOT / "training" / "longform"
SEG_BINS = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 99)]


def ref_words(iv, audio_s):
    """Transcript words (midpoint, speaker, normalised words) and gaps as in longform_score."""
    a = json.loads((ROOT / "work" / "aligned" / f"{iv}.json").read_text(encoding="utf-8"))
    ws, out, last = a["words"], [], 0.0
    for w in ws:
        if w.get("s") is not None and w.get("e") is not None:
            last = (w["s"] + w["e"]) / 2
        out.append((last, w.get("spk"), words(w["w"])))
    t = [m for m, _, _ in out]
    gaps = []
    for j in a.get("stars", []):
        g0 = ws[j - 1]["e"] if j > 0 and ws[j - 1].get("e") is not None else (0.0 if j == 0 else t[j - 1])
        g1 = ws[j]["s"] if j < len(ws) and ws[j].get("s") is not None else (audio_s if j >= len(ws) else t[j])
        gaps.append((g0, g1))
    return out, gaps


def flags(r, h):
    """Per reference word: wrong or missing (0/1); insertions before each word (index len(r) = at the end); error types."""
    f, ins, kinds = [0] * len(r), [0] * (len(r) + 1), defaultdict(int)
    for op in Levenshtein.editops(r, h):
        kinds[op.tag] += 1
        if op.tag == "insert":
            ins[op.src_pos] += 1
        else:
            f[op.src_pos] = 1
    return f, ins, kinds


def main():
    runs = sys.argv[1:3]
    if len(runs) != 2:
        sys.exit(__doc__)
    names, tags = zip(*(r.split(":") for r in runs))
    segs = defaultdict(list)
    for line in open(ROOT / "dataset" / "metadata.jsonl", encoding="utf-8"):
        s = json.loads(line)
        if s["split"] == "test":
            segs[s["interview"]].append((s["source_start"], s["source_end"], s["file_name"], s["duration"]))
    pred = []
    for n, t in zip(names, tags):
        with open(RUNS / n / f"predictions_test_{t}.csv", encoding="utf-8") as f:
            pred.append({r["file_name"]: (words(r["referenz"]), words(r["hypothese"])) for r in csv.DictReader(f)})

    same = defaultdict(lambda: [0, [0, 0], [0, 0]])       # length class -> [words, [seg, chunk] base, [seg, chunk] FT]
    groups = defaultdict(lambda: [[0, 0], [0, 0]])        # group -> per run [errors, words]
    pos = defaultdict(lambda: [[0, 0], [0, 0]])
    kinds = {k: [defaultdict(int), defaultdict(int)] for k in ("Segmente", "Stücke")}
    edge, secs = [defaultdict(int), defaultdict(int)], defaultdict(float)
    n_words = {"Segmente": 0, "Stücke": 0}
    n_same = 0
    for f in sorted((LF / "chunks").glob("*.json")):
        iv = f.stem
        meta = json.loads(f.read_text(encoding="utf-8"))
        outs = [json.loads((RUNS / n / f"longform_{t}" / f"{iv}.json").read_text(encoding="utf-8"))["chunks"]
                for n, t in zip(names, tags)]
        ref, gaps = ref_words(iv, meta["audio_s"])
        flat = [(i, j) for i, (_, _, ws) in enumerate(ref) for j in range(len(ws))]
        sg = sorted(segs[iv])
        seg_keys = [[(i, j) for i, j in flat if a <= ref[i][0] <= b] for a, b, _, _ in sg]
        in_seg = {key for keys in seg_keys for key in keys}
        cflag, cins, clean = [{}, {}], [{}, {}], {}
        for ci, (s, e, _) in enumerate(outs[0]):
            keys = [(i, j) for i, j in flat if s <= ref[i][0] < e]
            if not keys:
                continue
            ok = not any(min(e, g1) - max(s, g0) > 0.3 for g0, g1 in gaps)
            for key in keys:
                clean[key] = ok
            if not ok:
                continue
            rw = [ref[i][2][j] for i, j in keys]
            wt = [ref[i][0] - s for i, _ in keys]
            share = sum(key in in_seg for key in keys) / len(keys)
            spk = {ref[i][1] for i, _ in keys}
            gs = ["alle sauberen",
                  "ganz aus Test-Segmenten" if share == 1 else ("keine Test-Segmente" if share == 0 else "teils aus Test-Segmenten"),
                  "ein Sprecher" if len(spk) == 1 else "Sprecherwechsel",
                  "Stück < 15 s" if e - s < 15 else ("Stück 15-22 s" if e - s < 22 else "Stück ab 22 s")]
            secs["Rand"] += min(2.0, e - s); secs["Mitte"] += max(0.0, e - s - 2.0)
            n_words["Stücke"] += len(rw)
            for r in range(2):
                fl, ins, kd = flags(rw, words(outs[r][ci][2]))
                err = sum(fl) + sum(ins)
                for g in gs:
                    groups[g][r][0] += err; groups[g][r][1] += len(rw)
                for k, v in kd.items():
                    kinds["Stücke"][r][k] += v
                for p, key in enumerate(keys):
                    cflag[r][key] = fl[p]; cins[r][key] = ins[p + 1] if p + 1 < len(keys) else 0
                    b = "0-10 s" if wt[p] < 10 else ("10-20 s" if wt[p] < 20 else "ab 20 s")
                    pos[b][r][0] += fl[p] + ins[p]; pos[b][r][1] += 1
                for p in range(len(rw) + 1):
                    if ins[p]:
                        tm = 0.0 if p == 0 else ((e - s) if p == len(rw) else (wt[p - 1] + wt[p]) / 2)
                        edge[r]["Rand" if tm < 1.0 or tm > e - s - 1.0 else "Mitte"] += ins[p]
        for k, (a, b, fn, d) in enumerate(sg):
            keys = seg_keys[k]
            assert [ref[i][2][j] for i, j in keys] == pred[0][fn][0], fn   # segment text = transcript words
            n_words["Segmente"] += len(keys)
            sf = [flags(*pred[r][fn]) for r in range(2)]
            for r in range(2):
                for kk, v in sf[r][2].items():
                    kinds["Segmente"][r][kk] += v
            if not all(clean.get(key, False) for key in keys):
                continue
            n_same += 1
            acc = same[next(x for x in SEG_BINS if x[0] <= d < x[1])]
            for p, key in enumerate(keys):
                inner = p + 1 < len(keys)                   # insertion between two words of the same segment
                acc[0] += 1
                for r in range(2):
                    acc[1 + r][0] += sf[r][0][p] + (sf[r][1][p + 1] if inner else 0)
                    acc[1 + r][1] += cflag[r][key] + (cins[r][key] if inner else 0)

    pct = lambda a, n: 100 * a / max(1, n)
    res = {"laeufe": runs}
    print(f"1. dieselben Wörter ({n_same} von {sum(map(len, segs.values()))} Test-Segmenten liegen ganz in sauberen Stücken)")
    print(f"   {'Segmentlänge':14}{'Wörter':>8}   {'Basis Seg -> Stück':>20}   {'FT Seg -> Stück':>18}   Gewinn Seg / Stück")
    tot = [0, [0, 0], [0, 0]]
    res["dieselben_woerter"] = {}
    for bn in SEG_BINS + ["alle"]:
        if bn == "alle":
            a, label = tot, "alle"
        else:
            a = same[bn]; label = f"{bn[0]}-{bn[1]} s" if bn[1] < 99 else f"ab {bn[0]} s"
            tot = [tot[0] + a[0], [tot[1][0] + a[1][0], tot[1][1] + a[1][1]], [tot[2][0] + a[2][0], tot[2][1] + a[2][1]]]
        v = [pct(a[1][0], a[0]), pct(a[1][1], a[0]), pct(a[2][0], a[0]), pct(a[2][1], a[0])]
        res["dieselben_woerter"][label] = {"woerter": a[0], "basis_seg_stueck": v[:2], "ft_seg_stueck": v[2:]}
        print(f"   {label:14}{a[0]:8d}   {v[0]:8.1f} -> {v[1]:5.1f} %   {v[2]:7.1f} -> {v[3]:5.1f} %   {v[0] - v[2]:+5.1f} / {v[1] - v[3]:+5.1f}")
    print(f"2. saubere Stücke{'':20}{'Wörter':>8}{'Basis':>9}{'FT':>9}{'Gewinn':>8}")
    res["gruppen"] = {}
    for g in ["alle sauberen", "ganz aus Test-Segmenten", "teils aus Test-Segmenten", "keine Test-Segmente",
              "ein Sprecher", "Sprecherwechsel", "Stück < 15 s", "Stück 15-22 s", "Stück ab 22 s"]:
        a = groups[g]; v = [pct(*a[0]), pct(*a[1])]
        res["gruppen"][g] = {"woerter": a[0][1], "wer": v}
        print(f"   {g:33}{a[0][1]:8d}{v[0]:8.1f}%{v[1]:8.1f}%{v[0] - v[1]:+8.1f}")
    print("3. Fehlerarten je 100 Referenzwörter (falsch / fehlt / eingefügt)")
    res["fehlerarten"] = {}
    for label in ("Segmente", "Stücke"):
        for r in range(2):
            v = [pct(kinds[label][r][k], n_words[label]) for k in ("replace", "delete", "insert")]
            res["fehlerarten"][f"{names[r]} {label}"] = v
            print(f"   {names[r][:22]:22} {label:9} {v[0]:5.1f} / {v[1]:5.1f} / {v[2]:5.1f}")
    print("   Fehler nach Position im Stück (ohne Einfügungen am Stückende):")
    res["position"] = {}
    for b in ["0-10 s", "10-20 s", "ab 20 s"]:
        a = pos[b]; v = [pct(*a[0]), pct(*a[1])]
        res["position"][b] = {"woerter": a[0][1], "fehler": v}
        print(f"   {b:10}{a[0][1]:8d} Wörter   Basis {v[0]:5.1f} %   FT {v[1]:5.1f} %   Gewinn {v[0] - v[1]:+.1f}")
    res["einfuegungen_je_minute"] = {names[r]: {k: 60 * edge[r][k] / secs[k] for k in secs} for r in range(2)}
    print("   Einfügungen je Minute, Rand (erste und letzte Sekunde) / Mitte: " +
          "   ".join(f"{names[r][:18]} {60 * edge[r]['Rand'] / secs['Rand']:.1f} / {60 * edge[r]['Mitte'] / secs['Mitte']:.1f}" for r in range(2)))
    (LF / "analyse.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {LF / 'analyse.json'}")


if __name__ == "__main__":
    main()
