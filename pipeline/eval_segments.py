#!/usr/bin/env python3
"""
Independent check of the segments with a different recogniser (Whisper-small).

v1 checked the wav2vec2 alignment with the same wav2vec2 model - both
figures measured how well this one model understands the audio. Here:

  1. CER of Whisper-small against the segment text, for a random,
     duration-proportional sample of kept AND discarded segments.
     The absolute CER reflects dialect and recording; what is informative is
     the comparison: kept vs. discarded, high vs. medium. The two strata are
     drawn separately; anything across both (calibration) is extrapolated to
     the whole set with the stratum sizes (columns stratum_s, stratum_n).
  2. Shift test: the same segments with boundaries shifted by -1 s and +1 s.
     If the boundaries are right, Whisper recognises most of the target text
     in the original. This does not depend on how strong the dialect is -
     only on where the boundaries lie.

Result: work/eval/segments_whisper.csv and a summary.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 pipeline/eval_segments.py --n 500 --shift 150
"""
import argparse, collections, csv, json, os, random, re, sys, wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, SR, load_specs
from align import norm_word

OUT = WORK / "eval"


def norm(s):
    s = re.sub(r"\[(unv|gap|nv)\]", " ", s)
    return " ".join(w for w in (norm_word(t) for t in s.split()) if w).replace("|", " ")


def cer(ref, hyp):
    from rapidfuzz.distance import Levenshtein
    if not ref:
        return 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


MIN_OTHERS = 3      # this many other kept segments are needed for the interview median


def bad_flags(rows):
    """'Bad' per row of the sample: Whisper is more than 0.3 above what it
    otherwise achieves for this interview, or recognises practically nothing
    (CER > 0.9). The reference is the median CER of the OTHER kept segments
    of the same interview in the sample; if there are fewer than MIN_OTHERS,
    of the other kept segments of the same project (the projects lie far
    apart: median 0.16 / 0.30 / 0.36). Until 01.10.2026 the segment itself
    was included - with 1-2 drawn segments per interview (14 of 82) it was
    then hardly ever 'bad' (review): kept 4.4 instead of 5.8 %."""
    cer_ = [float(r["cer"]) for r in rows]
    keep = [r["kind"] == "keep" for r in rows]
    iv = [r["id"] for r in rows]
    proj = [r["id"].split("_", 1)[0] for r in rows]
    out = []
    for i in range(len(rows)):
        others = [cer_[j] for j in range(len(rows)) if j != i and keep[j] and iv[j] == iv[i]]
        if len(others) < MIN_OTHERS:
            others = [cer_[j] for j in range(len(rows)) if j != i and keep[j] and proj[j] == proj[i]] \
                or [cer_[j] for j in range(len(rows)) if j != i and keep[j]]
        out.append(cer_[i] - float(np.median(others)) > 0.3 or cer_[i] > 0.9)
    return out


class Wav:
    def __init__(self, path):
        self.w = wave.open(str(path))
        self.n = self.w.getnframes()

    def read(self, s, e):
        a, b = max(0, int(s * SR)), min(self.n, int(e * SR))
        self.w.setpos(a)
        return np.frombuffer(self.w.readframes(b - a), dtype="<i2").astype(np.float32) / 32768.0


def candidate_rules():
    rules = {"nichts verwerfen": lambda r: False}
    for sr in (-1.0, -1.5, -2.0, -2.5):
        rules[f"score_rel < {sr}"] = (lambda sr: lambda r: r["score_rel"] < sr)(sr)
    for sa in (-3.5, -4.5):
        rules[f"score < {sa}"] = (lambda sa: lambda r: r["score"] < sa)(sa)
    for em in (-6.0, -8.0, -10.0):
        rules[f"edge_min < {em}"] = (lambda em: lambda r: r["edge_min"] < em)(em)
    rules["score_rel < -1.5 oder score < -4.5"] = lambda r: r["score_rel"] < -1.5 or r["score"] < -4.5
    rules["score_rel < -2.0 oder score < -4.5"] = lambda r: r["score_rel"] < -2.0 or r["score"] < -4.5
    rules["(v1) score < -3.0"] = lambda r: r["score"] < -3.0
    return rules


def current_strata():
    """Seconds per stratum (kept / discarded) in the current version, without
    excluded interviews - for CSVs from before 01.10.2026, which do not yet
    record the stratum sizes."""
    tot = collections.Counter()
    for spec in load_specs():
        f = WORK / "segments" / f"{spec['id']}.json"
        if not f.exists() or spec.get("exclude"):
            continue
        for o in json.loads(f.read_text())["segments"]:
            tot["keep" if o["keep"] else "drop"] += o["duration"]
    return dict(tot)


def calibration(rows, n_boot=2000, seed=1):
    """Justify the thresholds for keep_rule. "Bad" means: for this segment,
    Whisper is clearly above what it otherwise achieves for the same person
    (CER - median of the other kept segments of this interview > 0.3, see
    bad_flags) or recognises practically nothing (CER > 0.9). So the figure
    measures mismatched pairs, not dialect. Rules are compared by hit rate
    and loss.

    Kept and discarded segments are drawn separately (500 vs. 150, each
    stratum by duration), so the discarded ones are strongly over-represented.
    Until 01.10.2026 the sample was evaluated unweighted - "finds 73 % of the
    bad ones" and "13 % bad" held only for the sample (review).
    Now every drawn segment stands for its share of the whole set
    (stratum duration / number drawn): shares, hit rate and loss apply to
    the hours of the whole set. 95 % intervals from a stratified bootstrap
    (the "bad" classification stays fixed)."""
    rows = [dict(r) for r in rows]
    for r in rows:
        for k in ("cer", "score", "score_rel", "edge_min", "duration"):
            r[k] = float(r[k])
    kind = np.array([r["kind"] for r in rows])
    if rows[0].get("stratum_s"):
        strata, source = {r["kind"]: float(r["stratum_s"]) for r in rows}, "Stichprobe"
    else:
        strata, source = current_strata(), "aktueller Stand"
    n = {k: int(np.sum(kind == k)) for k in ("keep", "drop")}
    w = np.array([strata[k] / n[k] for k in kind])          # seconds of the whole set per drawn segment
    bad = np.array(bad_flags(rows), float)
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(kind == k) for k in ("keep", "drop")]
    boot = [np.concatenate([rng.choice(i, len(i)) for i in idx]) for _ in range(n_boot)]

    def est(flag, ix):
        ww, bb, ff = w[ix], bad[ix], flag[ix]
        hit, flagged, allbad = np.sum(ww * ff * bb), np.sum(ww * ff), np.sum(ww * bb)
        return (hit / flagged if flagged else np.nan, hit / allbad if allbad else np.nan)

    def interval(values):
        values = values[np.isfinite(values)]
        return tuple(float(v) for v in np.percentile(values, [2.5, 97.5])) if values.size else (np.nan, np.nan)

    out = []
    for name, f in candidate_rules().items():
        flag = np.array([bool(f(r)) for r in rows], float)
        precision, recall = est(flag, np.arange(len(rows)))
        bs = np.array([est(flag, ix) for ix in boot])
        out.append({"rule": name, "n_sample": int(flag.sum()),
                    "flag_h": float(np.sum(w * flag)) / 3600, "precision": precision, "recall": recall,
                    "good_lost_h": float(np.sum(w * flag * (1 - bad))) / 3600,
                    "precision_ci": interval(bs[:, 0]), "recall_ci": interval(bs[:, 1])})
    return {"n": n, "strata_h": {k: v / 3600 for k, v in strata.items()}, "strata_source": source,
            "bad_share_h": float(np.sum(w * bad) / np.sum(w)), "bad_share_sample": float(bad.mean()),
            "rules": out}


def calibrate(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cal = calibration(rows)
    print(f"Stichprobe {len(rows)} Segmente ({cal['n']['keep']} behalten, {cal['n']['drop']} verworfen); "
          f"hochgerechnet auf {cal['strata_h']['keep']:.2f} h behalten und {cal['strata_h']['drop']:.2f} h "
          f"verworfen (Schichtgroessen: {cal['strata_source']}): schlecht {100*cal['bad_share_h']:.1f} % der "
          f"Stunden (ungewichtet {100*cal['bad_share_sample']:.1f} %)\n")
    print(f"{'Regel':38} {'Stichpr.':>8} {'verwirft h':>10} {'davon schlecht':>18} {'erwischt':>16} {'Verlust gute h':>14}")
    for x in cal["rules"]:
        ci = lambda v, c: "     -" if v != v else f"{100*v:3.0f} % ({100*c[0]:.0f}-{100*c[1]:.0f})"
        print(f"{x['rule']:38} {x['n_sample']:8} {x['flag_h']:10.2f} {ci(x['precision'], x['precision_ci']):>18} "
              f"{ci(x['recall'], x['recall_ci']):>16} {x['good_lost_h']:14.2f}")


def compare_pairs(tsv, arch, device="mps"):
    """Whisper pre-check for the re-check: old against new cut of the same
    segments (samples/nachpruefung.tsv). Measures whether the new cut lets
    more of the target text be heard - does not replace listening.
    -> work/eval/nachpruefung_whisper.csv"""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    arch = Path(arch)
    old = {json.loads(l)["file_name"]: json.loads(l) for l in (arch / "dataset/metadata.jsonl").read_text().splitlines()}
    new = {json.loads(l)["file_name"]: json.loads(l) for l in (WORK.parent / "dataset/metadata.jsonl").read_text().splitlines()}
    proc = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device).eval()

    def asr(x):
        feats = proc(x, sampling_rate=SR, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(feats, language="de", task="transcribe", max_new_tokens=220, num_beams=1)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    wavs, rows = {}, []
    for r in csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"):
        mo, mn = old[r["alt"]], new[r["neu"]]
        w = wavs.setdefault(mn["interview"], Wav(WORK / "audio16k" / f"{mn['interview']}.wav"))
        ref = norm(mn["text"])
        ho, hn = asr(w.read(mo["source_start"], mo["source_end"])), asr(w.read(mn["source_start"], mn["source_end"]))
        rows.append({"paar": r["paar"], "cer_alt": round(cer(ref, norm(ho)), 4), "cer_neu": round(cer(ref, norm(hn)), 4),
                     "hyp_alt": ho, "hyp_neu": hn})
    out = OUT / "nachpruefung_whisper.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    d = np.array([x["cer_neu"] - x["cer_alt"] for x in rows])
    print(f"{len(rows)} Paare: neu besser {int(np.sum(d < -0.01))}, gleich {int(np.sum(abs(d) <= 0.01))}, "
          f"neu schlechter {int(np.sum(d > 0.01))} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="Stichprobe behaltener Segmente")
    ap.add_argument("--n-drop", type=int, default=150, help="Stichprobe verworfener Segmente")
    ap.add_argument("--shift", type=int, default=150, help="davon mit Verschiebetest")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--calibrate", default=None, help="nur auswerten: CSV einer frueheren Pruefung")
    ap.add_argument("--vergleich", default=None, help="samples/nachpruefung.tsv: alter gegen neuen Schnitt")
    ap.add_argument("--archiv", default="archive/v2.1_2026-09-28", help="Stand der alten Schnitte")
    a = ap.parse_args()
    if a.calibrate:
        return calibrate(a.calibrate)
    if a.vergleich:
        return compare_pairs(a.vergleich, a.archiv, a.device)
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    rng = random.Random(a.seed)
    segs = []
    for spec in load_specs():
        f = WORK / "segments" / f"{spec['id']}.json"
        # excluded interviews belong neither to "kept" nor to
        # "discarded by the filter"
        if not f.exists() or spec.get("exclude"):
            continue
        for k, o in enumerate(json.loads(f.read_text())["segments"]):
            segs.append((spec["id"], spec["project"], k, o))
    kept = [x for x in segs if x[3]["keep"]]
    drop = [x for x in segs if not x[3]["keep"]]
    # draw in proportion to duration (long segments count as much as their duration)
    def draw(pool, n):
        w = [x[3]["duration"] for x in pool]
        idx = set()
        while len(idx) < min(n, len(pool)):
            idx.add(rng.choices(range(len(pool)), weights=w)[0])
        return [pool[i] for i in sorted(idx)]
    sample = [("keep", x) for x in draw(kept, a.n)] + [("drop", x) for x in draw(drop, a.n_drop)]
    # record the stratum sizes: the calibration uses them to extrapolate to the whole set
    stratum = {"keep": (sum(x[3]["duration"] for x in kept), len(kept)),
               "drop": (sum(x[3]["duration"] for x in drop), len(drop))}
    shift_ids = set(rng.sample(range(a.n), min(a.shift, a.n)))
    proc = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(a.device).eval()

    def asr(x):
        feats = proc(x, sampling_rate=SR, return_tensors="pt").input_features.to(a.device)
        with torch.no_grad():
            ids = model.generate(feats, language="de", task="transcribe", max_new_tokens=220, num_beams=1)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    wavs = {}
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for n, (kind, (iid, proj, k, o)) in enumerate(sample, 1):
        w = wavs.setdefault(iid, Wav(WORK / "audio16k" / f"{iid}.wav"))
        ref = norm(o["text"])
        hyp = asr(w.read(o["start"], o["end"]))
        row = {"kind": kind, "id": iid, "project": proj, "seg": k, "start": o["start"],
               "duration": o["duration"], "quality": o.get("quality") or "", "drop_reason": o["drop_reason"],
               "role": o["role"], "score": o["score"], "score_rel": o["score_rel"], "edge_min": o["edge_min"],
               "cer": round(cer(ref, norm(hyp)), 4), "ref": o["text"], "hyp": hyp,
               "stratum_s": round(stratum[kind][0], 2), "stratum_n": stratum[kind][1]}
        if kind == "keep" and (n - 1) in shift_ids:
            for d in (-1.0, 1.0):
                h2 = asr(w.read(o["start"] + d, o["end"] + d))
                row[f"cer_shift{d:+.0f}"] = round(cer(ref, norm(h2)), 4)
        rows.append(row)
        if n % 50 == 0:
            print(f"  {n}/{len(sample)}", flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(OUT / "segments_whisper.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    summarize(rows)


def summarize(rows):
    def line(name, sub):
        if not sub:
            return
        c = np.array([r["cer"] for r in sub])
        print(f"  {name:34} n={len(c):4}  CER Median {np.median(c):.3f}  Mittel {c.mean():.3f}  >0.6 {np.mean(c>0.6)*100:4.1f}%")
    print("\nWhisper-small gegen Segmenttext (dauerproportionale Stichprobe):")
    kept = [r for r in rows if r["kind"] == "keep"]
    line("behalten, alle", kept)
    for p in ("rs", "re", "za"):
        line(f"behalten, Projekt {p}", [r for r in kept if r["project"] == p])
    for q in ("hoch", "mittel"):
        line(f"behalten, Stufe {q}", [r for r in kept if r["quality"] == q])
    line("verworfen, alle", [r for r in rows if r["kind"] == "drop"])
    for why in sorted({r["drop_reason"] for r in rows if r["kind"] == "drop"}):
        line(f"verworfen: {why}", [r for r in rows if r["kind"] == "drop" and r["drop_reason"] == why])
    sh = [r for r in kept if "cer_shift+1" in r]
    if sh:
        o = np.array([r["cer"] for r in sh]); m = np.array([r["cer_shift-1"] for r in sh])
        p = np.array([r["cer_shift+1"] for r in sh])
        best = np.mean((o <= m) & (o <= p))
        print(f"\nVerschiebetest (n={len(sh)}): Original am besten in {best*100:.1f}%  |  "
              f"CER Original {np.median(o):.3f}, -1 s {np.median(m):.3f}, +1 s {np.median(p):.3f} (Median)")
    if len(rows) > 20:
        def ranks(x):
            return np.argsort(np.argsort(np.asarray(x, float)))
        c = ranks([r["cer"] for r in rows])
        for f in ("score", "score_rel", "edge_min"):
            rho = np.corrcoef(ranks([r[f] for r in rows]), c)[0, 1]
            print(f"  Spearman({f}, CER) = {rho:+.2f}  (Stichprobe beider Schichten, ungewichtet)")


if __name__ == "__main__":
    main()
