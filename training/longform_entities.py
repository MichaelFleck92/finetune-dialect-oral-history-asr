#!/usr/bin/env python3
"""
Names on complete interviews: the same reviewed mentions as in
entity_eval.py (test segments), searched for in the output of longform_eval.py.
Every word of a test segment lies in a chunk via its alignment time; the
chunk output is aligned against all transcript words of the chunk (Levenshtein as
in entity_eval). A mention counts as recognised if all its words are "equal".

  python3 training/longform_entities.py baseline-large-v3:beam5fb baseline-large-v3:beam5fb-buch1 \
      --liste namensliste.txt

The first run is the reference (McNemar, bootstrap over interviews). "sauber" = the
mention lies in a chunk without omitted speech (as in longform_score.py). With
--liste: PER/LOC/ORG separately for in / not in the list, plus inserted list names
(more frequent in the chunk output than in the chunk's transcript words).
--projekte rs,re: only these projects (prefix before the first _ in the interview name).
-> training/longform/nennungen[_<projekte>].json. Only figures are printed.
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import ROOT, RUNS, words
from entity_eval import ENT, TYPES, NAMES, word_spans, align, mcnemar
from longform_analysis import LF, ref_words

PROJ = {"rs": "Roter Stern", "re": "Rechnitz", "za": "Zwangsarbeiter"}


def load_mentions():
    """Interview -> [(segment start, end, normalised words, [(type, text, word indices)], segment, word spans, text)]."""
    keep = {}
    with open(ENT / "entitaeten_test_geprueft.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            if r["entscheidung"] in ("stimmt", "typ") and r["typ"] in TYPES:
                keep[r["text"]] = r["typ"]
    meta = {}
    for line in (ROOT / "dataset" / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["split"] == "test":
            meta[r["file_name"]] = r
    ents = {}
    for line in (ENT / "test_spacy.jsonl").read_text(encoding="utf-8").splitlines():
        seg = json.loads(line)
        spans = word_spans(meta[seg["file_name"]]["text"])
        es = []
        for e in seg["entities"]:
            idx = [i for i, (_, s, x) in enumerate(spans) if s < e["end"] and x > e["start"]]
            if keep.get(e["text"]) and idx:
                es.append((keep[e["text"]], e["text"], idx))
        ents[seg["file_name"]] = (es, spans)
    out = defaultdict(list)
    for f, r in meta.items():
        es, spans = ents.get(f, ([], word_spans(r["text"])))
        out[r["interview"]].append((r["source_start"], r["source_end"], words(r["text"]), es, f, spans, r["text"]))
    return {iv: sorted(v, key=lambda x: x[:2]) for iv, v in out.items()}


def snippet(text, spans, a, b):
    """Original text of the normalised words a..b-1 (spans from word_spans)."""
    a, b = max(0, a), min(len(spans), b)
    return text[spans[a][1]:spans[b - 1][2]] if b > a else ""


def word_map(ref, hyp):
    """Word-by-word alignment (Levenshtein editops): per reference word (position in hyp, aligned?) -
    equal or substituted: True, deleted: False at the place where it would be; plus the inserted positions."""
    m, ins, i, j = [None] * len(ref), set(), 0, 0
    for op in Levenshtein.editops(ref, hyp):
        while i < op.src_pos:                               # same distance before
            m[i] = (j, True); i += 1; j += 1
        if op.tag == "replace":
            m[i] = (j, True); i += 1; j += 1
        elif op.tag == "delete":
            m[i] = (j, False); i += 1
        else:
            ins.add(j); j += 1
    while i < len(ref):
        m[i] = (j, True); i += 1; j += 1
    return m, ins


def best_span(target, hw, h0, h1, ins, n=2, w=3):
    """Output range [h0, h1) of the mention, extended at the edges by up to n inserted words
    if that comes closer to the name (e.g. a name as two words); if a range up to w words
    away is clearly closer (+10 points, at least 60 %), that one - the alignment slips when the output
    omits part of a sentence; -> (a, b, similarity)."""
    best = None
    for l in range(n + 1):
        if l and h0 - l not in ins:
            break
        for r in range(n + 1):
            if r and h1 + r - 1 not in ins:
                break
            a, b = h0 - l, h1 + r
            sim = fuzz.ratio(target, " ".join(hw[a:b])) / 100
            if best is None or sim > best[2]:
                best = (a, b, sim)
    mapped, L = best[2], len(target.split())
    for a in range(max(0, h0 - w), min(len(hw), h1 + w)):
        for b in range(a + max(1, L - 1), min(len(hw), a + L + 1) + 1):
            sim = fuzz.ratio(target, " ".join(hw[a:b])) / 100
            if sim >= max(0.6, mapped + 0.1) and sim > best[2]:
                best = (a, b, sim)
    return best


def count_names(ws, by_first):
    c = defaultdict(int)
    for i, w in enumerate(ws):
        for seq in by_first.get(w, ()):
            if tuple(ws[i:i + len(seq)]) == seq:
                c[seq] += 1
    return c


def collect(runs, lst=frozenset(), ctx=6, projekte=None):
    """Per mention: type, in the list?, clean?, per run recognised / output / similarity (fuzz.ratio
    as in entity_eval) and context (ctx words before and after each); plus inserted list names."""
    by_first = defaultdict(list)
    for s in lst:
        by_first[s[0]].append(s)
    segs = load_mentions()
    mentions, inserted, unplaced = [], [[0, 0] for _ in runs], 0
    for f in sorted((LF / "chunks").glob("*.json")):
        iv = f.stem
        if projekte and iv.split("_")[0] not in projekte:
            continue
        meta = json.loads(f.read_text(encoding="utf-8"))
        outs = [json.loads((RUNS / n / ("longform" + (f"_{t}" if t else "")) / f"{iv}.json").read_text(encoding="utf-8"))["chunks"]
                for n, t in runs]
        ref, gaps = ref_words(iv, meta["audio_s"])
        flat = [(i, j) for i, (_, _, ws) in enumerate(ref) for j in range(len(ws))]
        status, pos, hyps, clean = [{} for _ in runs], [{} for _ in runs], [{} for _ in runs], {}
        for ci, (s, e, _) in enumerate(outs[0]):
            keys = [(i, j) for i, j in flat if s <= ref[i][0] < e]
            if not keys:
                continue
            ok = not any(min(e, g1) - max(s, g0) > 0.3 for g0, g1 in gaps)
            rw = [ref[i][2][j] for i, j in keys]
            cr = count_names(rw, by_first) if lst else {}
            for r, out in enumerate(outs):
                assert out[ci][:2] == [s, e], iv                   # all runs with the same chunks
                sp = word_spans(out[ci][2])
                hw = [w for w, _, _ in sp]
                st = align(rw, hw)
                wm, ins = word_map(rw, hw)
                hyps[r][ci] = (out[ci][2], sp, hw, ins)
                for p, key in enumerate(keys):
                    status[r][key] = st[p][0] == "equal"
                    pos[r][key] = (ci, wm[p])
                if lst:
                    ch = count_names(hw, by_first)
                    k = sum(max(0, v - cr.get(q, 0)) for q, v in ch.items())
                    inserted[r][0] += k
                    inserted[r][1] += k if ok else 0
            for key in keys:
                clean[key] = ok
        for a0, b0, sw, es, fn, spans, text in segs.get(iv, []):
            keys = [(i, j) for i, j in flat if a0 <= ref[i][0] <= b0]
            assert [ref[i][2][j] for i, j in keys] == sw, iv          # segment text = transcript words
            for typ, etext, idx in es:
                ks = [keys[i] for i in idx]
                unplaced += not all(k in clean for k in ks)
                target = " ".join(sw[i] for i in idx)
                i0, i1 = idx[0], idx[-1] + 1
                m = {"interview": iv, "file_name": fn, "typ": typ, "text": etext, "in_liste": tuple(words(etext)) in lst,
                     "sauber": all(clean.get(k, False) for k in ks), "zeit": round(ref[ks[0][0]][0] - a0, 2),
                     "ref_vor": snippet(text, spans, i0 - ctx, i0), "ref_nach": snippet(text, spans, i1, i1 + ctx),
                     "erkannt": [all(status[r].get(k, False) for k in ks) for r in range(len(runs))],
                     "ausgabe": [], "vor": [], "nach": [], "aehnlichkeit": [], "stelle": []}
                for r in range(len(runs)):
                    parts = defaultdict(list)                       # chunk -> alignment of the mention words
                    for k in ks:
                        if k in pos[r]:
                            parts[pos[r][k][0]].append(pos[r][k][1])
                    got, vor, nach, stelle = [], "", "", None
                    for n, ci in enumerate(sorted(parts)):
                        htext, sp, hw, ins = hyps[r][ci]
                        hit = [j for j, ok in parts[ci] if ok]
                        h0, h1 = (min(hit), max(hit) + 1) if hit else (parts[ci][0][0],) * 2   # empty = omitted
                        x0, x1, _ = best_span(target, hw, h0, h1, ins)
                        if x1 > x0:
                            got.append(snippet(htext, sp, x0, x1))
                        if n == 0:
                            vor = snippet(htext, sp, x0 - ctx, x0)
                            stelle = (ci, x0, x1)                  # chunk and word range [x0, x1) in the output
                        nach = snippet(htext, sp, x1, x1 + ctx)
                    norm = " ".join(w for g in got for w in words(g))
                    m["ausgabe"].append(" ".join(got)); m["vor"].append(vor); m["nach"].append(nach); m["stelle"].append(stelle)
                    m["aehnlichkeit"].append(round(fuzz.ratio(target, norm) / 100, 3))
                mentions.append(m)
    return mentions, inserted, unplaced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="<lauf>:<tag>, der erste ist die Bezugsgröße")
    ap.add_argument("--liste", default=None, help="Namensliste (eine Schreibung je Zeile)")
    ap.add_argument("--projekte", default=None, help="Komma-Liste der Projektkürzel, z. B. rs")
    a = ap.parse_args()
    runs = [tuple(r.split(":")) if ":" in r else (r, "") for r in a.runs]
    lst = {tuple(words(l)) for l in Path(a.liste).read_text(encoding="utf-8").splitlines() if words(l)} if a.liste else set()
    projekte = a.projekte.split(",") if a.projekte else None
    mentions, inserted, unplaced = collect(runs, lst, projekte=projekte)
    names = [f"{n}:{t}" if t else n for n, t in runs]
    core = [m for m in mentions if m["typ"] != "MISC"]
    pol = [m for m in mentions if m["typ"] in ("PER", "LOC", "ORG")]
    groups = {"Namen und Daten": core, "Namen und Daten, sauber": [m for m in core if m["sauber"]],
              **{NAMES[t]: [m for m in mentions if m["typ"] == t] for t in TYPES}, "alle mit Sonstiges": mentions}
    for p, pn in PROJ.items():
        groups[f"Projekt {pn}"] = [m for m in core if m["interview"].startswith(p + "_")]
    if lst:
        groups["PER/LOC/ORG"] = pol
        groups["PER/LOC/ORG in der Liste"] = [m for m in pol if m["in_liste"]]
        groups["PER/LOC/ORG nicht in der Liste"] = [m for m in pol if not m["in_liste"]]
        for p, pn in PROJ.items():
            groups[f"PER/LOC/ORG {pn}"] = [m for m in pol if m["interview"].startswith(p + "_")]
    groups = {g: ms for g, ms in groups.items() if ms}
    print(f"Ganze Interviews, geprüfte Nennungen der Test-Segmente: {len(mentions)} "
          f"(davon {sum(m['sauber'] for m in mentions)} in sauberen Stücken, {unplaced} nicht in einem Stück)")
    out = {"laeufe": names, "liste": a.liste, "listengroesse": len(lst), "nennungen": len(mentions),
           "nicht_in_stueck": unplaced, "gruppen": {}}
    print(f"{'Gruppe':32} {'n':>5}  " + "  ".join(f"{'Lauf ' + str(r + 1):>8}" for r in range(len(runs)))
          + "   neu richtig / neu falsch gegen Lauf 1, p")
    for g, ms in groups.items():
        rates = [sum(m["erkannt"][r] for m in ms) / len(ms) for r in range(len(runs))]
        row = {"n": len(ms), "erkannt": dict(zip(names, rates)), "gegen_1": {}}
        cmp = []
        for r in range(1, len(runs)):
            c = sum(m["erkannt"][r] and not m["erkannt"][0] for m in ms)
            b = sum(m["erkannt"][0] and not m["erkannt"][r] for m in ms)
            row["gegen_1"][names[r]] = {"neu_richtig": c, "neu_falsch": b, "p_mcnemar": mcnemar(b, c)}
            cmp.append(f"{c:3d} / {b:3d}, p = {mcnemar(b, c):.2g}")
        out["gruppen"][g] = row
        print(f"{g:32} {len(ms):5}  " + "  ".join(f"{100*x:7.1f}%" for x in rates) + "   " + ";  ".join(cmp))
    rng = np.random.default_rng(1)
    out["bootstrap_interviews"] = {}
    for g in ("Namen und Daten", "PER/LOC/ORG"):
        if g not in groups:
            continue
        ms_g = groups[g]
        ivs = sorted({m["interview"] for m in ms_g})
        by_iv = {iv: [m for m in ms_g if m["interview"] == iv] for iv in ivs}
        picks = [[m for iv in rng.choice(ivs, len(ivs)) for m in by_iv[iv]] for _ in range(10000)]
        out["bootstrap_interviews"][g] = {}
        for r in range(1, len(runs)):
            d = [sum(m["erkannt"][r] - m["erkannt"][0] for m in ms) / len(ms) for ms in picks]
            lo, hi = np.percentile(d, [2.5, 97.5])
            diff = sum(m["erkannt"][r] - m["erkannt"][0] for m in ms_g) / len(ms_g)
            out["bootstrap_interviews"][g][names[r]] = {"diff": diff, "ci95": [float(lo), float(hi)]}
            print(f"{g}: Lauf {r + 1} - Lauf 1 {100*diff:+.1f} Prozentpunkte, 95 % über {len(ivs)} Interviews: "
                  f"{100*lo:+.1f} bis {100*hi:+.1f}")
    if lst:
        out["eingefuegte_listennamen"] = {n: {"alle": x[0], "sauber": x[1]} for n, x in zip(names, inserted)}
        for r, n in enumerate(names):
            print(f"Lauf {r + 1} {n:40} eingefügte Listennamen {inserted[r][0]} (in sauberen Stücken {inserted[r][1]})")
    out["projekte"] = projekte
    path = LF / ("nennungen" + (f"_{'-'.join(projekte)}" if projekte else "") + ".json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
