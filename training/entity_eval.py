#!/usr/bin/env python3
"""
Wie gut erkennen zwei Läufe die Namen im Transkript? (Schritt 4)

  python3 training/entity_eval.py --a baseline-large-v3 --b large-v3-lora
  python3 training/entity_eval.py --a ... --b ... --ungeprueft   # nur zum Testen

Liest training/entities/<split>_spacy.jsonl (entities_tag.py) und die geprüfte
Liste entitaeten_<split>_geprueft.csv (entities_review.py); gewertet werden nur
Schreibungen mit "stimmt" oder geändertem Typ, im geprüften Typ. --ungeprueft
nimmt alle spaCy-Vorschläge außer "?" (Ergebnis ist dann als ungeprüft markiert).

Je Nennung: die Wörter der Entität (normalisiert wie die WER) und ihre Zuordnung
zur Ausgabe des Modells (Levenshtein-Ausrichtung des ganzen Segments).
  erkannt  alle Wörter der Entität stehen genau so in der Ausgabe
  nah      Zeichenähnlichkeit zur zugeordneten Ausgabe >= 0,8 (z. B. eine Endung falsch)
Hauptergebnis sind Personen, Orte, Organisationen und Daten ("Namen und Daten");
"Sonstiges" (Gruppen, Ereignisse, Adjektive, Dialekt- und Alltagswörter) steht
getrennt. Dazu die Fehlerrate auf Namenswörtern gegen die auf allen übrigen Wörtern,
der McNemar-Test (gepaart je Nennung) und ein Bootstrap über die Interviews.
-> training/entities/ergebnis_<split>[_<tag>]_<a>_vs_<b>.json (nur Zahlen) und
   nennungen_<split>[_<tag>]_<a>_vs_<b>.csv (je Nennung, mit Text - bleibt lokal).
   --tag beam5fb wertet die Ausgaben mit Strahlsuche + Rückfall (beide Läufe gleich);
   <lauf>@<tag> wählt die Dekodierung je Lauf (z. B. baseline-large-v3@beam5fb-pnamen).
Ausgegeben werden nur Zahlen.
"""
import argparse, csv, json, math, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import ROOT, RUNS, FILLERS, norm_word, words

ENT = ROOT / "training" / "entities"
TYPES = ["PER", "LOC", "ORG", "MISC", "DATUM"]
NAMES = {"PER": "Person", "LOC": "Ort", "ORG": "Organisation", "MISC": "Sonstiges", "DATUM": "Datum"}
CLOSE = 0.8


def word_spans(text):
    """Normalisierte Wörter wie whisper_common.words, je mit Zeichenbereich des Tokens."""
    out = []
    for m in re.finditer(r"\S+", text):
        for w in norm_word(m.group()).replace("'", "").split("|"):
            if w and w not in FILLERS:
                out.append((w, m.start(), m.end()))
    return out


def align(ref, hyp):
    """Je Referenzwort: (Art, Bereich in hyp) aus den Levenshtein-Opcodes."""
    st = [None] * len(ref)
    for op in Levenshtein.opcodes(ref, hyp):
        for i in range(op.src_start, op.src_end):
            st[i] = (op.tag, (op.dest_start, op.dest_end))
    return st


def mcnemar(b, c):
    """Exakter zweiseitiger McNemar-Test (Binomial, p = 0,5) für b gegen c."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="baseline-large-v3")
    ap.add_argument("--b", default="large-v3-lora")
    ap.add_argument("--split", default="test")
    ap.add_argument("--geprueft", default=None, help="CSV aus der Prüfseite (Standard: training/entities/entitaeten_<split>_geprueft.csv)")
    ap.add_argument("--ungeprueft", action="store_true")
    ap.add_argument("--tag", default="", help="Dekodierung beider Läufe, z. B. beam5fb (predictions_<split>_<tag>.csv)")
    a = ap.parse_args()
    meta = {}
    for line in (ROOT / "dataset" / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["split"] == a.split:
            meta[r["file_name"]] = r
    keep = None
    if not a.ungeprueft:
        path = Path(a.geprueft) if a.geprueft else ENT / f"entitaeten_{a.split}_geprueft.csv"
        if not path.exists():
            sys.exit(f"fehlt: {path} (Prüfseite training/entities/pruefen_{a.split}.html, oder --ungeprueft)")
        keep = {}
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter=";"):
                if r["entscheidung"] in ("stimmt", "typ") and r["typ"] in TYPES:
                    keep[r["text"]] = r["typ"]
    sfx = f"_{a.tag}" if a.tag else ""
    def pred_path(run):                                  # <lauf>@<tag> wählt die Dekodierung je Lauf
        if "@" in run:
            name, t = run.split("@", 1)
            return RUNS / name / f"predictions_{a.split}_{t}.csv"
        return RUNS / run / f"predictions_{a.split}{sfx}.csv"
    hyp = {k: {r["file_name"]: r["hypothese"] for r in csv.DictReader(open(pred_path(k), encoding="utf-8"))}
           for k in (a.a, a.b)}
    mentions, word_err = [], {k: defaultdict(lambda: [0, 0]) for k in (a.a, a.b)}
    for line in (ENT / f"{a.split}_spacy.jsonl").read_text(encoding="utf-8").splitlines():
        seg = json.loads(line)
        f = seg["file_name"]
        text = meta[f]["text"]
        spans = word_spans(text)
        ref = [w for w, _, _ in spans]
        assert ref == words(text), f"Normalisierung weicht ab: {f}"
        ents = []
        for e in seg["entities"]:
            typ = (keep.get(e["text"]) if keep is not None else (e["label"] if e["label"] in TYPES else None))
            idx = [i for i, (_, s, x) in enumerate(spans) if s < e["end"] and x > e["start"]]
            if typ and idx:
                ents.append((e, typ, idx))
        is_ent = {i: ("Sonstiges" if typ == "MISC" else "Namen") for _, typ, idx in ents for i in idx}
        res = {}
        for k in (a.a, a.b):
            h = words(hyp[k][f])
            st = align(ref, h)
            for i in range(len(ref)):
                cls = is_ent.get(i, "übrige")
                word_err[k][cls][1] += 1
                word_err[k][cls][0] += st[i][0] != "equal"
            res[k] = (h, st)
        for e, typ, idx in ents:
            m = {"file_name": f, "interview": seg["interview"], "project": seg["project"], "text": e["text"], "typ": typ}
            target = " ".join(ref[i] for i in idx)
            for k in (a.a, a.b):
                h, st = res[k]
                js = sorted({j for i in idx for j in range(*st[i][1])})
                got = " ".join(h[j] for j in js)
                exact = all(st[i][0] == "equal" for i in idx)
                sim = fuzz.ratio(target, got) / 100
                m[k] = {"erkannt": exact, "nah": exact or sim >= CLOSE, "aehnlichkeit": round(sim, 3), "ausgabe": got}
            mentions.append(m)
    if not mentions:
        sys.exit("keine gewerteten Nennungen")

    def recall(ms, k, key):
        return sum(m[k][key] for m in ms) / max(1, len(ms))

    core = [m for m in mentions if m["typ"] != "MISC"]
    groups = {"Namen und Daten": core, **{NAMES[t]: [m for m in mentions if m["typ"] == t] for t in TYPES}, "alle mit Sonstiges": mentions}
    groups = {g: ms for g, ms in groups.items() if ms}
    proj = {"rs": "Roter Stern", "re": "Rechnitz", "za": "Zwangsarbeiter"}
    for p, name in proj.items():
        ms = [m for m in core if m["project"] == p]
        if ms:
            groups[f"Projekt {name}"] = ms
    out = {"split": a.split, "a": a.a, "b": a.b, "geprueft": keep is not None, "nah_ab": CLOSE, "gruppen": {}}
    print(f"Entitäten {a.split}: {len(mentions)} Nennungen ({'geprüft' if keep is not None else 'UNGEPRÜFT, nur Test'}) "
          f"- {a.a} gegen {a.b}")
    print(f"{'Gruppe':26} {'n':>5}  {'erkannt A':>9} {'erkannt B':>9}  {'nah A':>6} {'nah B':>6}  {'nur A':>5} {'nur B':>5} {'p (McNemar)':>11}")
    for g, ms in groups.items():
        b = sum(m[a.a]["erkannt"] and not m[a.b]["erkannt"] for m in ms)
        c = sum(m[a.b]["erkannt"] and not m[a.a]["erkannt"] for m in ms)
        row = {"n": len(ms), "erkannt_a": recall(ms, a.a, "erkannt"), "erkannt_b": recall(ms, a.b, "erkannt"),
               "nah_a": recall(ms, a.a, "nah"), "nah_b": recall(ms, a.b, "nah"), "nur_a": b, "nur_b": c, "p_mcnemar": mcnemar(b, c)}
        out["gruppen"][g] = row
        print(f"{g:26} {len(ms):5}  {100*row['erkannt_a']:8.1f}% {100*row['erkannt_b']:8.1f}%  {100*row['nah_a']:5.1f}% "
              f"{100*row['nah_b']:5.1f}%  {b:5} {c:5} {row['p_mcnemar']:11.2g}")
    rng = np.random.default_rng(1)
    out["bootstrap_interviews"] = {}
    for g in ("Namen und Daten", "alle mit Sonstiges"):
        ms_g = groups[g]
        ivs = sorted({m["interview"] for m in ms_g})
        by_iv = {iv: [m for m in ms_g if m["interview"] == iv] for iv in ivs}
        diffs = []
        for _ in range(10000):
            ms = [m for iv in rng.choice(ivs, len(ivs)) for m in by_iv[iv]]
            diffs.append(recall(ms, a.b, "erkannt") - recall(ms, a.a, "erkannt"))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        d = out["gruppen"][g]["erkannt_b"] - out["gruppen"][g]["erkannt_a"]
        out["bootstrap_interviews"][g] = {"interviews": len(ivs), "diff_erkannt": d, "ci95": [float(lo), float(hi)]}
        print(f"{g}: Unterschied erkannt (B - A) {100*d:+.1f} Prozentpunkte, "
              f"95 % über {len(ivs)} Interviews gebootstrapt: {100*lo:+.1f} bis {100*hi:+.1f}")
    out["wortfehler"] = {k: {cls: {"fehler": v[0], "woerter": v[1], "rate": v[0] / max(1, v[1])} for cls, v in word_err[k].items()}
                         for k in (a.a, a.b)}
    for k in (a.a, a.b):
        w = out["wortfehler"][k]
        print(f"Wortfehler {k:20} " + ", ".join(f"{cls} {100*w[cls]['rate']:5.1f} % ({w[cls]['woerter']} Wörter)"
                                                 for cls in ("Namen", "Sonstiges", "übrige") if cls in w))
    tag = (f"{a.split}{sfx}_{a.a}_vs_{a.b}" + ("" if keep is not None else "_ungeprueft")).replace("/", "-").replace("@", "_")
    (ENT / f"ergebnis_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(ENT / f"nennungen_{tag}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["file_name", "interview", "typ", "text", f"erkannt_{a.a}", f"erkannt_{a.b}", f"ausgabe_{a.a}", f"ausgabe_{a.b}"])
        for m in mentions:
            w.writerow([m["file_name"], m["interview"], m["typ"], m["text"], int(m[a.a]["erkannt"]), int(m[a.b]["erkannt"]),
                        m[a.a]["ausgabe"], m[a.b]["ausgabe"]])
    print(f"-> training/entities/ergebnis_{tag}.json, nennungen_{tag}.csv")


if __name__ == "__main__":
    main()
