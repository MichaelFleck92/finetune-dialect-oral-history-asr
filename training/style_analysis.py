#!/usr/bin/env python3
"""
Schritt 3: echte Verbesserung oder nur Transkriptionsstil? Vergleicht zwei Läufe
(A = Ausgangsmodell, B = feingetunt) auf demselben Teil, ohne Text auszugeben.

  python3 training/style_analysis.py --a baseline-large-v3 --b <lauf> --tag beam5fb

1. Fehler je Wortklasse des Transkripts (Funktionswort, Partikel, Nomen/Name,
   anderes Inhaltswort, Zahl; dazu eingefügte Wörter) und Anteil am Gewinn.
2. Stilfreie WER: Partikeln auf beiden Seiten ignoriert, unmittelbare
   Wiederholungen zusammengefasst, fast gleiche Schreibung (Zeichenähnlichkeit
   >= 0,8, z. B. hab/habe) nicht als Fehler. Bleibt der Gewinn, liegt er nicht in
   diesen Stilfragen. Mit Bootstrap über die Interviews.
3. Passung zum Audio ohne Transkript: CTC-Likelihood jedes Textes unter dem
   unabhängigen Akustikmodell der Pipeline (wav2vec2, gespeicherte Emissionen in
   work/emissions). Das Modell kennt die Konventionen des Archivs nicht; es
   bevorzugt allerdings lautnahe Schreibung (Dialekt "begumaster" vor
   "Bürgermeister") und misst darum Auslassen/Hinzufügen besser als Normalisierung.
   Dazu die Kreuztabelle WER-Urteil gegen Akustik-Urteil je Segment.
-> training/runs/<b>/stil_<split>[_<tag>].json (nur Zahlen; Lauf-Schrägstriche als -).
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import ROOT, RUNS, DATASET, words, norm_word

DISCOURSE = set("ja jaja na naja nein nee ne gell also halt eh eben schon doch mal denn eigentlich irgendwie so ach oh aha genau jo nun okay ok".split())
FUNCTION = set("""der die das den dem des ein eine einen einem einer eines ich du er sie es wir ihr mich mir dich dir ihn ihm uns euch
sich mein meine meinen meinem meiner sein seine seinen seinem seiner ihre ihren ihrem ihrer unser unsere dieser diese dieses diesem diesen
und oder aber weil dass wenn als wie ob sondern bis da wo was wer in im ins an am auf aus bei beim mit nach von vom vor zu zum zur über
unter um durch für gegen ohne seit hinter neben zwischen bin bist ist sind seid war waren wäre wären gewesen hab habe hat haben hatte
hatten hätte werden wird wurde wurden worden kann konnte können muss musste müssen soll sollte darf durfte will wollte mag möchte nicht
kein keine keinen nichts nie dann dort hier jetzt noch auch nur sehr immer wieder gar man denen""".split())
FPS = 50


def classes(text):
    toks, out = text.split(), []
    for k, t in enumerate(toks):
        ws = words(t)
        low = ws[0] if ws else ""
        if any(c.isdigit() for c in t):
            cl = "Zahl"
        elif low in DISCOURSE:
            cl = "Partikel"
        elif low in FUNCTION:
            cl = "Funktionswort"
        elif t[:1].isupper() and k > 0 and not toks[k - 1].endswith((".", "!", "?", ":")):
            cl = "Nomen/Name"
        else:
            cl = "anderes Inhaltswort"
        out += [(w, cl) for w in ws]
    return out


def style_blind(seq):
    seq = [w for w in seq if w not in DISCOURSE]
    return [w for i, w in enumerate(seq) if i == 0 or w != seq[i - 1]]


def blind_errors(ref, hyp):
    return sum(1 for o in Levenshtein.editops(ref, hyp)
               if not (o.tag == "replace" and fuzz.ratio(ref[o.src_pos], hyp[o.dest_pos]) >= 80))


def ctc_targets(text, vocab):
    ids = []
    for w in text.split():
        nw = norm_word(w)
        if not nw:
            continue
        if ids:
            ids.append(vocab["|"])
        ids += [vocab["|"] if ch == "|" else vocab.get(ch, vocab["<unk>"]) for ch in nw]
    return ids


def boot(per_iv, fn, n=10000, seed=1):
    ivs = sorted(per_iv)
    rng = np.random.default_rng(seed)
    vals = [fn([per_iv[iv] for iv in rng.choice(ivs, len(ivs))]) for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="baseline-large-v3")
    ap.add_argument("--b", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="", help="Dekodierung beider Läufe, z. B. beam5fb")
    ap.add_argument("--ohne-akustik", action="store_true")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    meta = {}
    for line in (DATASET / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["split"] == a.split:
            meta[r["file_name"]] = r
    P = {k: {r["file_name"]: r["hypothese"] for r in csv.DictReader(open(RUNS / k / f"predictions_{a.split}{sfx}.csv", encoding="utf-8"))}
         for k in (a.a, a.b)}
    files = sorted(P[a.a])
    assert set(files) == set(P[a.b]) == set(meta), "Läufe und Teil passen nicht zusammen"
    res = {"a": a.a, "b": a.b, "split": a.split, "tag": a.tag, "segmente": len(files)}

    # 1 + 2
    per = {k: defaultdict(lambda: [0, 0]) for k in (a.a, a.b)}
    iv_std = defaultdict(lambda: np.zeros(6))           # je Interview: Fehler A, B, Wörter; stilfrei A, B, Wörter
    for f in files:
        ref = classes(meta[f]["text"])
        rw = [w for w, _ in ref]
        r2 = style_blind(rw)
        row = []
        for k in (a.a, a.b):
            hw = words(P[k][f])
            for _, cl in ref:
                per[k][cl][1] += 1
            for o in Levenshtein.editops(rw, hw):
                if o.tag in ("replace", "delete"):
                    per[k][ref[o.src_pos][1]][0] += 1
                else:
                    w = hw[o.dest_pos]
                    per[k]["eingefügt: " + ("Partikel" if w in DISCOURSE else "Funktionswort" if w in FUNCTION else "anderes")][0] += 1
            row.append((Levenshtein.distance(rw, hw), blind_errors(r2, style_blind(hw))))
        iv_std[meta[f]["interview"]] += np.array([row[0][0], row[1][0], len(rw), row[0][1], row[1][1], len(r2)])
    N = sum(v[1] for c, v in per[a.a].items() if not c.startswith("eingefügt"))
    gain = sum(per[a.a][c][0] - per[a.b][c][0] for c in per[a.a])
    res["wortklassen"] = {c: {"anteil_woerter": per[a.a][c][1] / N if per[a.a][c][1] else None,
                              "fehler_a": per[a.a][c][0], "fehler_b": per[a.b][c][0], "woerter": per[a.a][c][1],
                              "gewinn_pp": 100 * (per[a.a][c][0] - per[a.b][c][0]) / N,
                              "anteil_gewinn": (per[a.a][c][0] - per[a.b][c][0]) / gain if gain else None}
                          for c in per[a.a]}
    print(f"{a.a} gegen {a.b} auf {a.split}{' (' + a.tag + ')' if a.tag else ''}: {len(files)} Segmente, {N} Referenzwörter")
    print(f"{'Wortklasse':26} {'Anteil':>7} {'Fehler A':>9} {'Fehler B':>9} {'Gewinn':>9} {'Anteil am Gewinn':>17}")
    for c, v in sorted(res["wortklassen"].items(), key=lambda x: (x[0].startswith("eingefügt"), -(x[1]["woerter"] or 0))):
        if v["woerter"]:
            print(f"{c:26} {100*v['anteil_woerter']:6.1f} % {100*v['fehler_a']/v['woerter']:8.1f} % {100*v['fehler_b']/v['woerter']:8.1f} % "
                  f"{v['gewinn_pp']:6.2f} pp {100*(v['anteil_gewinn'] or 0):15.0f} %")
        else:
            print(f"{c:26} {'':>7} {v['fehler_a']:9d} {v['fehler_b']:9d} {v['gewinn_pp']:6.2f} pp {100*(v['anteil_gewinn'] or 0):15.0f} %")
    T = sum(iv_std.values())
    wer = lambda xs: sum(x[0] for x in xs) / sum(x[2] for x in xs) - sum(x[1] for x in xs) / sum(x[2] for x in xs)
    blind = lambda xs: sum(x[3] for x in xs) / sum(x[5] for x in xs) - sum(x[4] for x in xs) / sum(x[5] for x in xs)
    res["wer"] = {"a": T[0] / T[2], "b": T[1] / T[2], "gewinn": T[0] / T[2] - T[1] / T[2], "ci95": boot(iv_std, wer)}
    res["wer_stilfrei"] = {"a": T[3] / T[5], "b": T[4] / T[5], "gewinn": T[3] / T[5] - T[4] / T[5], "ci95": boot(iv_std, blind)}
    for key, name in (("wer", "WER"), ("wer_stilfrei", "WER stilfrei")):
        v = res[key]
        print(f"{name:13} {100*v['a']:5.1f} % -> {100*v['b']:5.1f} %, Gewinn {100*v['gewinn']:4.1f} pp "
              f"(95 % über {len(iv_std)} Interviews: {100*v['ci95'][0]:.1f} bis {100*v['ci95'][1]:.1f})")

    # 3
    if not a.ohne_akustik:
        emis, vocab, nll = {}, None, {k: [] for k in ("ref", a.a, a.b)}
        for f in files:
            m = meta[f]
            iv = m["interview"]
            if iv not in emis:
                emis[iv] = torch.from_numpy(np.load(ROOT / f"work/emissions/{iv}.npy").astype(np.float32))
                vocab = json.loads((ROOT / f"work/emissions/{iv}.json").read_text())["vocab"]
            lp = emis[iv][int(round(m["source_start"] * FPS)):int(round(m["source_end"] * FPS))]
            T_ = lp.shape[0]
            for k, text in (("ref", m["text"]), (a.a, P[a.a][f]), (a.b, P[a.b][f])):
                tg = ctc_targets(text, vocab)
                if not tg:
                    nll[k].append(float(-lp[:, 0].sum()))
                    continue
                nll[k].append(float(torch.nn.functional.ctc_loss(lp.unsqueeze(1), torch.tensor([tg]), torch.tensor([T_]),
                                                                 torch.tensor([len(tg)]), blank=0, reduction="sum", zero_infinity=False)))
        nll = {k: np.array(v) for k, v in nll.items()}
        dur = np.array([meta[f]["duration"] for f in files])
        same = np.array([words(P[a.a][f]) == words(P[a.b][f]) for f in files])
        d = nll[a.a] - nll[a.b]
        ok = ~same & np.isfinite(nll[a.a]) & np.isfinite(nll[a.b])
        cross = defaultdict(int)
        for i, f in enumerate(files):
            if not ok[i]:
                continue
            rw = words(meta[f]["text"])
            ea, eb = Levenshtein.distance(rw, words(P[a.a][f])), Levenshtein.distance(rw, words(P[a.b][f]))
            w = "wer_b_besser" if eb < ea else "wer_b_schlechter" if eb > ea else "wer_gleich"
            cross[f"{w}|{'akustik_b_besser' if d[i] > 0 else 'akustik_a_besser'}"] += 1
        res["akustik"] = {"verschieden": int((~same).sum()), "gewertet": int(ok.sum()), "b_passt_besser": int((d[ok] > 0).sum()),
                          "a_passt_besser": int((d[ok] < 0).sum()), "vorsprung_b_median_nats_s": float(np.median(d[ok] / dur[ok])),
                          "unmoeglich_lang_a": int(np.isinf(nll[a.a]).sum()), "unmoeglich_lang_b": int(np.isinf(nll[a.b]).sum()),
                          "kreuztabelle": dict(cross)}
        v = res["akustik"]
        print(f"Akustik (wav2vec2, CTC): {v['verschieden']} Segmente mit verschiedenem Text, B passt besser in "
              f"{v['b_passt_besser']} ({100*v['b_passt_besser']/max(1, v['gewertet']):.0f} %), A in {v['a_passt_besser']}; "
              f"Median-Vorsprung {v['vorsprung_b_median_nats_s']:.2f} nats/s")
        for w in ("wer_b_besser", "wer_gleich", "wer_b_schlechter"):
            x, y = cross[f"{w}|akustik_b_besser"], cross[f"{w}|akustik_a_besser"]
            print(f"  {w:17} Akustik stimmt B zu: {x:5}  bevorzugt A: {y:5}  ({100*x/max(1, x+y):.0f} % / {100*y/max(1, x+y):.0f} %)")
    out = RUNS / a.b / f"stil_{a.split}{sfx}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
