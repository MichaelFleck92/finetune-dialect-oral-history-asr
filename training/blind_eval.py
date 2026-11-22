#!/usr/bin/env python3
"""
Evaluate step 2: blind ratings against the key.

  python3 training/blind_eval.py --schluessel training/blind/schluessel_<lauf>.json bewertung_X.csv bewertung_Y.csv

For each rating, the verdict (-2 = A clearly better ... +2 = B clearly better)
is turned so that positive means "fine-tuned model better". Control pairs (same
output) count only as attention checks. Per person and combined:
shares better/equal/worse, mean rating, sign test (exact, two-sided, without
"equal"), bootstrap over the interviews; agreement between the two persons
(Cohen's kappa on better/equal/worse, weighted kappa on the five-point
scale); error types per model, if the page asked for them (no longer since
06.10.2026). Totals unweighted (every interview with the same number of pairs)
and weighted by interview size.
-> training/blind/ergebnis_<lauf>.json; only figures are printed.
"""
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FLAGS = ["name", "sinn", "erfunden", "fehlt"]


def sign_test(pos, neg):
    n, k = pos + neg, min(pos, neg)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def kappa(x, y, cats, weights=None):
    """Cohen's kappa; weights=None unweighted, "linear" linearly weighted."""
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    obs = np.zeros((k, k))
    for a, b in zip(x, y):
        obs[idx[a], idx[b]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.array([[0 if i == j else 1 for j in range(k)] for i in range(k)], float) if weights is None else \
        np.array([[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)])
    return 1 - (w * obs).sum() / max(1e-12, (w * exp).sum())


def summary(scores, ivs, iv_size, rng):
    s = np.array(scores, float)
    pos, neg = int((s > 0).sum()), int((s < 0).sum())
    by = defaultdict(list)
    for v, iv in zip(s, ivs):
        by[iv].append(v)
    keys = sorted(by)
    mean_iv = np.mean([np.mean(by[iv]) for iv in keys])
    w = np.array([iv_size[iv] for iv in keys], float)
    mean_w = float(np.sum(w * [np.mean(by[iv]) for iv in keys]) / w.sum())
    boot = []
    for _ in range(10000):
        pick = rng.choice(keys, len(keys))
        boot.append(np.mean([np.mean(by[iv]) for iv in pick]))
    return {"n": len(s), "besser": pos, "gleich": int((s == 0).sum()), "schlechter": neg,
            "anteil_besser": pos / len(s), "anteil_schlechter": neg / len(s), "mittel": float(s.mean()),
            "mittel_je_interview": float(mean_iv), "mittel_gewichtet": mean_w,
            "ci95_mittel_interviews": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p_vorzeichentest": sign_test(pos, neg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schluessel", required=True)
    ap.add_argument("bewertungen", nargs="+")
    a = ap.parse_args()
    key = json.loads(Path(a.schluessel).read_text(encoding="utf-8"))
    items, ft = key["items"], key["b"]
    iv_size = key["interviews"]
    rng = np.random.default_rng(1)
    raters, flags, ctrl = {}, defaultdict(Counter), defaultdict(list)
    for path in a.bewertungen:
        for r in csv.DictReader(open(path, encoding="utf-8"), delimiter=";"):
            if r["urteil"] == "":
                continue
            it = items[r["item"]]
            u = int(r["urteil"])
            if it["kontrolle"]:
                ctrl[r["bewerter"]].append(u)
                continue
            score = -u if it["A"] == ft else u          # > 0: fine-tuned model better
            raters.setdefault(r["bewerter"], {})[r["item"]] = score
            for side in ("A", "B"):                      # error types only if the page asked for them
                model = "feingetunt" if it[side] == ft else "ausgang"
                for f in FLAGS:
                    if r.get(f"{side}_{f}") not in (None, ""):
                        flags[(r["bewerter"], model)][f] += int(r[f"{side}_{f}"])
    out = {"lauf": ft, "ausgang": key["a"], "je_person": {}, "kontrollen": {}}
    print(f"Blinde Bewertung: {key['a']} gegen {ft}; positiv = {ft} besser")
    for name, sc in raters.items():
        ids = sorted(sc)
        s = summary([sc[i] for i in ids], [items[i]["interview"] for i in ids], iv_size, rng)
        out["je_person"][name] = s
        c = ctrl.get(name, [])
        out["kontrollen"][name] = {"n": len(c), "gleich": sum(v == 0 for v in c)}
        print(f"  {name:12} n={s['n']:3}  besser {s['besser']:3} ({100*s['anteil_besser']:.0f} %), gleich {s['gleich']:3}, "
              f"schlechter {s['schlechter']:3} ({100*s['anteil_schlechter']:.0f} %); Mittel {s['mittel']:+.2f} "
              f"(je Interview {s['mittel_je_interview']:+.2f}, gewichtet {s['mittel_gewichtet']:+.2f}, "
              f"95 % {s['ci95_mittel_interviews'][0]:+.2f} bis {s['ci95_mittel_interviews'][1]:+.2f}); "
              f"Vorzeichentest p = {s['p_vorzeichentest']:.2g}; Kontrollen 'gleich' {sum(v == 0 for v in c)}/{len(c)}")
    names = list(raters)
    if len(names) >= 2:
        x, y = raters[names[0]], raters[names[1]]
        common = sorted(set(x) & set(y))
        d = lambda v: (v > 0) - (v < 0)
        k3 = kappa([d(x[i]) for i in common], [d(y[i]) for i in common], [-1, 0, 1])
        k5 = kappa([x[i] for i in common], [y[i] for i in common], [-2, -1, 0, 1, 2], "linear")
        agree = sum(d(x[i]) == d(y[i]) for i in common) / max(1, len(common))
        mean = {i: (x[i] + y[i]) / 2 for i in common}
        s = summary([mean[i] for i in common], [items[i]["interview"] for i in common], iv_size, rng)
        out["gemeinsam"] = {**s, "paare": len(common), "uebereinstimmung_richtung": agree, "kappa_richtung": k3, "kappa_gewichtet_5": k5}
        print(f"  gemeinsam    {len(common)} Paare: Richtung gleich beurteilt {100*agree:.0f} %, Kappa {k3:.2f}, "
              f"gewichtetes Kappa (5 Stufen) {k5:.2f}; Mittel beider {s['mittel']:+.2f} "
              f"(95 % über Interviews {s['ci95_mittel_interviews'][0]:+.2f} bis {s['ci95_mittel_interviews'][1]:+.2f}), "
              f"Vorzeichentest p = {s['p_vorzeichentest']:.2g}")
    out["fehlerarten"] = {f"{n}|{m}": dict(c) for (n, m), c in flags.items()}
    for (n, m), c in sorted(flags.items()):
        print(f"  Fehlerarten {n:12} {m:10} " + ", ".join(f"{f} {c[f]}" for f in FLAGS))
    path = Path(a.schluessel).parent / f"ergebnis_{ft.replace('/', '-').replace('@', '-')}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
