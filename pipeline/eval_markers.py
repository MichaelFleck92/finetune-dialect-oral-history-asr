#!/usr/bin/env python3
"""
Independent accuracy measurement: alignment against the hand-set time marks.

The alignment no longer uses the marks - so they are a yardstick that does
not come from the same assumptions as the result (in v1 this existed only for
one interview). A mark stands at the end of a paragraph; it is compared with
the end of the last word before it.

The marks themselves are imprecise: they lag behind the speech (key press
after hearing). Until 29.09.2026 this delay was estimated per interview as the
median of (word end - mark) - from the same marks that were then measured; a
constant offset of the alignment thus disappeared into the delay (review,
point 2). Since then Whisper-large-v3 measures the delay independently of the
alignment (delays(): N_DELAY marks per interview, where do the words before
the mark end?), and only this delay is corrected for; the offset of the
alignment against Whisper stays in the error and is reported. For
comparison: leave-one-out, the old delay and the raw values without correction.

Gross deviations (> 10 s) are decided case by case: where is the passage
before the mark spoken - where the alignment puts it, or where the mark (plus
delay) stands? For screening, the delay estimated from the same marks is
subtracted (sufficient for a threshold of 10 s; with the independently
measured delay one more case would lie just above it - it stays undecided
and counts as an error in the figures). Until 29.09.2026 this was decided by the greedy decoding
of the same wav2vec2 model that provides the alignment's anchors -
it almost inevitably had to agree with the alignment (review). Since then a
recogniser that has nothing in common with the alignment decides
(Whisper-large-v3: different architecture, different training data), plus the ear:

  * Per case, one window before each of the two candidate positions (WIN_BEFORE /
    WIN_AFTER). Whisper transcribes without guidance - the target text is not
    passed in as a prompt. The N_REF words before the mark are searched fuzzily
    in what was heard; the word times tell where the passage ends.
  * Verdict: match below the chance threshold -> unclear. If the passage ends
    at most TOL_S from exactly one candidate position, that one is right
    ("marke_falsch" = alignment right, "alignment_falsch" = mark right);
    if it lies further from both -> "beide_falsch". Only windows with reliable
    word times give the position (Whisper's word times occasionally collapse
    onto one point in time); if two windows disagree -> unclear.
    Both since the listening check of 30.09.2026, which found exactly such a case.
  * Chance threshold: 99th percentile of the match scores a target text
    reaches in the windows of other interviews (where it does not occur).
  * Positive control: one undisputed mark per interview (deviation <= 2 s)
    - does Whisper find the passage where it must be?
  * Listening check (blind, A/B in random order): all cases that Whisper
    does not decide, in which it contradicts the alignment or in which the
    passage stands in both windows (then the word times alone decide), plus a
    random sample of the rest as a check on Whisper. Where the ear commits,
    its verdict applies.

The Whisper results per window are cached (work/eval/marker_whisper.json);
only what is missing is recomputed.

  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 pipeline/eval_markers.py --csv work/eval/marker_eval.csv
  python3 pipeline/eval_markers.py --hoerprobe       # next round: samples/marker/, then samples/marker_2/ ...
  python3 pipeline/eval_markers.py --hoerprobe-csv samples/marker/marker_hoerprobe.csv --csv work/eval/marker_eval.csv

With --csv, marker_grob.csv (every gross deviation with times, Whisper and
ear verdict), marker_kontrolle.csv (positive control) and
marker_schiedsrichter.json (threshold, control, agreement) are written next to
it - for report.py.
"""
import argparse, bisect, csv, html, json, math, random, re, shutil, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORK, SR, load_specs, write_json

WHISPER = "openai/whisper-large-v3"
GROSS_S = 10.0          # from this deviation on (after subtracting the delay), cases are decided individually
N_REF = 12              # this many words before the mark are searched for
WIN_BEFORE, WIN_AFTER = 24.0, 4.0   # Whisper window around a candidate position (28 s < 30 s)
TOL_S = 5.0             # passage ends this close to the candidate position -> spoken there
NULL_Q = 99             # chance threshold: percentile of matches in other interviews
CONTROL_DEV_S = 2.0     # undisputed mark for the positive control
CONTROL_SEED, EAR_SEED = 30, 31
N_DELAY, DELAY_SEED = 10, 32   # measure the delay of the marks independently: this many marks per interview
MIN_DELAY = 3                   # fewer valid measurements -> no independent delay, no correction
CLIP_BEFORE, CLIP_AFTER = 12.0, 2.0  # listening check: excerpt around each candidate position
EVAL = WORK / "eval"
CACHE = EVAL / "marker_whisper.json"
EAR_TSV = EVAL / "marker_hoerprobe.tsv"
EAR_CSV = EVAL / "marker_hoerprobe_bewertung.csv"
EAR_DIR = ROOT / "samples" / "marker"
VERDICTS = ("marke_falsch", "alignment_falsch", "beide_falsch", "unklar")


def evaluate(path):
    d = json.loads(Path(path).read_text())
    W = d["words"]
    # Multi-part recordings: the time marks restart at zero in each part
    # (group interview "Teil 2", one two-part interview). If the clock jumps back,
    # the next part applies from there, with its offset in the concatenated audio.
    parts = json.loads((WORK / "audio16k" / f"{d['id']}.json").read_text())["parts"]
    offs = [p["offset_s"] for p in parts]
    part, prev_t = 0, -1.0
    rows = []
    for h in sorted(d.get("hand_markers", []), key=lambda h: h["after_word"]):
        if h["t"] < prev_t - 30 and part + 1 < len(offs):
            part += 1
        prev_t = h["t"]
        wi = h["after_word"]
        if wi >= len(W) or "e" not in W[wi]:
            continue
        nxt = next((W[j] for j in range(wi + 1, min(len(W), wi + 4)) if "s" in W[j]), None)
        rows.append({"t": h["t"] + offs[part], "end": W[wi]["e"], "next": nxt["s"] if nxt else None, "wi": wi})
    return d, rows


def summarize(rows):
    if len(rows) < 3:
        return None
    t = np.array([r["t"] for r in rows])
    e = np.array([r["end"] for r in rows])
    # The mark stands at the end of a paragraph: it is compared with the end of the
    # last word before it. (The start of the next word is no use - in between
    # there is often untranscribed conversation, ~200 s in one interview.)
    err = e - t
    lag = float(np.median(err))
    dev = np.abs(err - lag)
    return {"n": len(rows), "lag_s": round(lag, 2), "median_abs_s": round(float(np.median(dev)), 2),
            "p90_abs_s": round(float(np.percentile(dev, 90)), 2),
            "within_2s": round(float(np.mean(dev <= 2)), 3), "within_5s": round(float(np.mean(dev <= 5)), 3),
            "gross_gt_10s": int(np.sum(dev > GROSS_S)), "gross_gt_30s": int(np.sum(dev > 30))}

# ------------------------------------------------------------ Whisper as referee

def reference(W, wi):
    """The N_REF words before the mark: normalised (for searching) and as in the transcript."""
    ws = W[max(0, wi - N_REF + 1): wi + 1]
    return (" ".join(w["norm"] for w in ws).replace("|", " "),
            re.sub(r"\s+([,.;:!?…])", r"\1", " ".join(w["w"] for w in ws)))


class Hearer:
    """Whisper-large-v3 over windows of the 16 kHz audio, with a cache."""

    def __init__(self, device="mps"):
        self.cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
        if self.cache.get("model") != WHISPER:
            self.cache = {"model": WHISPER, "revision": None, "windows": {}}
        self.device, self.asr, self.wavs, self.dirty = device, None, {}, False

    def window(self, iid, c):
        """Window before candidate position c -> key (result in the cache)."""
        a, b = max(0.0, c - WIN_BEFORE), max(0.0, c - WIN_BEFORE) + WIN_BEFORE + WIN_AFTER
        key = f"{iid}:{a:.2f}:{b:.2f}"
        if key in self.cache["windows"]:
            return key
        if self.asr is None:
            import torch
            from transformers import pipeline
            from huggingface_hub import snapshot_download
            self.asr = pipeline("automatic-speech-recognition", model=WHISPER, device=self.device,
                                torch_dtype=torch.float32)
            self.cache["revision"] = Path(snapshot_download(WHISPER, local_files_only=True)).name
        from eval_segments import Wav
        w = self.wavs.setdefault(iid, Wav(WORK / "audio16k" / f"{iid}.wav"))
        # no prompt, no guidance: Whisper only hears the audio
        out = self.asr({"raw": w.read(a, b), "sampling_rate": SR}, return_timestamps="word",
                       generate_kwargs={"language": "de", "task": "transcribe"})
        words = [[ch["text"].strip(), None if ch["timestamp"][0] is None else round(a + ch["timestamp"][0], 2),
                  None if ch["timestamp"][1] is None else round(a + ch["timestamp"][1], 2)]
                 for ch in out["chunks"]]
        self.cache["windows"][key] = {"a": a, "b": b, "text": out["text"].strip(), "words": words}
        self.dirty = True
        if len(self.cache["windows"]) % 10 == 0:
            self.save()
        return key

    def save(self):
        if self.dirty:
            write_json(CACHE, self.cache)
            self.dirty = False


_HYP = {}


def heard(win):
    """What was heard in a window, normalised like the transcript (align.norm_word):
    text, character end of each word, time end of each word, word without duration?
    Once per window."""
    if id(win) not in _HYP:
        from align import norm_word
        hyp, pos, ends, flat = "", [], [], []
        for w, s, e in win["words"]:
            n = norm_word(w).replace("|", " ")
            if not n:
                continue
            hyp += (" " if hyp else "") + n
            pos.append(len(hyp)); ends.append(e)
            flat.append(s is None or e is None or e - s < 0.02)
        _HYP[id(win)] = (hyp, pos, ends, flat)
    return _HYP[id(win)]


def match(ref, win):
    """How well does the target text occur in what was heard (0-100), where does
    it end, and are the word times usable there? If Whisper heard less than the
    target text is long, the whole is compared - otherwise a single matching
    word would already give 100. The word times occasionally collapse (many
    words on the same point in time, without duration - one interview,
    listening check 30.09.2026): if more than half of the words found have no
    duration, the window says nothing about WHERE the passage ends."""
    from rapidfuzz import fuzz
    hyp, pos, ends, flat = heard(win)
    if not hyp or not ref:
        return 0.0, None, False
    if len(hyp) >= len(ref):
        al = fuzz.partial_ratio_alignment(ref, hyp)
        score, dstart, dend = al.score, al.dest_start, al.dest_end
    else:
        score, dstart, dend = fuzz.ratio(ref, hyp), 0, len(hyp)
    k0 = min(bisect.bisect_right(pos, dstart), len(ends) - 1)
    k1 = max(k0, min(bisect.bisect_left(pos, dend), len(ends) - 1))
    t = ends[k1] if ends[k1] is not None else win["b"]
    ok = sum(flat[k0:k1 + 1]) * 2 <= (k1 - k0 + 1)
    return round(float(score), 1), t, ok


def locate(ref, keys, hearer):
    """All windows of a case -> [(score, end of passage, time reliable, window)],
    best score first."""
    res = [(*match(ref, hearer.cache["windows"][k]), k) for k in keys]
    return sorted(res, key=lambda r: -r[0])


def whisper_verdict(hits, c_al, c_mk):
    """hits: windows above the chance threshold. -> (verdict, end of passage).
    Only windows with reliable word times say anything about the position; if
    two of them disagree, the case stays unclear. Neither candidate position
    is preferred, not even at equal scores."""
    good = [h for h in hits if h[2] and h[1] is not None]
    if not good:
        return "unklar", None
    ends = [h[1] for h in good]
    if max(ends) - min(ends) > TOL_S:
        return "unklar", None
    t = good[0][1]
    d_al, d_mk = abs(t - c_al), abs(t - c_mk)
    if d_al <= TOL_S < d_mk:
        return "marke_falsch", t
    if d_mk <= TOL_S < d_al:
        return "alignment_falsch", t
    if d_al > TOL_S and d_mk > TOL_S:
        return "beide_falsch", t
    return "unklar", t


def collect(files):
    """All interviews with valid marks -> (figures, gross cases, control candidates)."""
    excluded = {s["id"] for s in load_specs() if s.get("exclude")}
    per, cases, calm = [], [], []
    for f in files:
        d, rows = evaluate(f)
        if not d.get("hand_markers_valid", True):
            per.append({"id": d["id"], "invalid": True}); continue
        s = summarize(rows)
        if not s:
            continue
        W = d["words"]
        per.append({"id": d["id"], "rows": rows, "s": s, "excluded": d["id"] in excluded,
                    "audio_s": d["audio_s"],
                    # marks whose word is not aligned are missing from rows - reported, not hidden
                    "skipped": len(d.get("hand_markers", [])) - len(rows)})
        for r in rows:
            dev = abs(r["end"] - r["t"] - s["lag_s"])
            ref, ref_text = reference(W, r["wi"])
            r["ref"], r["ref_text"] = ref, ref_text
            x = {"id": d["id"], "wi": r["wi"], "t": r["t"], "end": r["end"], "lag": s["lag_s"],
                 "c_al": r["end"], "c_mk": round(r["t"] + s["lag_s"], 2), "dev": round(dev, 2),
                 "ref": ref, "ref_text": ref_text, "excluded": d["id"] in excluded, "audio_s": d["audio_s"]}
            if dev > GROSS_S:
                cases.append(x)
            elif dev <= CONTROL_DEV_S:
                calm.append(x)
    return per, cases, calm


def arbitrate(cases, calm, hearer):
    """Whisper verdict for every gross case, chance threshold, positive control."""
    rng = random.Random(CONTROL_SEED)
    controls = []
    for iid in sorted({x["id"] for x in calm}):         # one undisputed mark per interview
        controls.append(dict(rng.choice([x for x in calm if x["id"] == iid])))
    todo = [(x, c) for x in cases for c in (x["c_al"], x["c_mk"])] + [(x, x["c_al"]) for x in controls]
    for n, (x, c) in enumerate(todo, 1):
        x.setdefault("keys", []).append(hearer.window(x["id"], c))
        if n % 20 == 0:
            print(f"  Whisper {n}/{len(todo)}", flush=True)
    hearer.save()
    for x in cases + controls:
        x["keys"] = list(dict.fromkeys(x["keys"]))
    # chance threshold: target text against the windows of a case from another interview
    null = [locate(x["ref"], y["keys"], hearer)[0][0]
            for x in cases + controls for y in cases + controls if y["id"] != x["id"]]
    thr = float(math.ceil(np.percentile(null, NULL_Q)))
    for x in cases + controls:
        res = locate(x["ref"], x["keys"], hearer)
        hits = [r for r in res if r[0] >= thr]
        v, t = whisper_verdict(hits, x["c_al"], x["c_mk"])
        # If the candidate positions are less than one window apart, the
        # passage can stand in both windows - then the word times alone decide
        x.update({"hits": len(hits), "flat": sum(not r[2] for r in hits), "w_score": res[0][0], "w_end": t,
                  "w_text": hearer.cache["windows"][res[0][3]]["text"],
                  "d_al": None if t is None else round(abs(t - x["c_al"]), 2),
                  "d_mk": None if t is None else round(abs(t - x["c_mk"]), 2), "v_whisper": v})
    for x in controls:
        x["found"] = x["d_al"] is not None and x["d_al"] <= TOL_S
    info = {"model": WHISPER, "revision": hearer.cache.get("revision"), "tol_s": TOL_S, "n_ref": N_REF,
            "regel": "Ort nur aus Fenstern mit verlaesslichen Wortzeiten; widersprechen sich zwei Fenster "
                     "-> unklar (seit Hoerprobe 30.09.2026)",
            "window_s": [WIN_BEFORE, WIN_AFTER], "null_n": len(null), "null_q": NULL_Q,
            "null_median": round(float(np.median(null)), 1), "threshold": thr,
            "controls_n": len(controls), "controls_found": sum(x["found"] for x in controls)}
    return controls, info

def delays(per, hearer, thr):
    """Delay of the marks, measured independently of the alignment (review, point 2).

    Previously the delay was estimated per interview as the median of
    (word end - mark) - from the same marks that are then measured. A constant
    offset of the alignment disappeared into the delay. Now: N_DELAY random
    marks per interview (without the gross cases); Whisper hears a 28 s window
    centred between the word end according to the alignment and the mark
    (neither position preferred) and says where the words before the mark
    end (w). A measurement is valid above the chance threshold with reliable
    word times. Per interview:
      delay D   = median(mark - w)         how late the mark was set
      offset A  = median(word end - w)     how far the alignment lies from Whisper
    word end - mark = A - D: the old delay mixed both. Only D is corrected
    for now; A stays in the error."""
    rng = random.Random(DELAY_SEED)
    todo = []
    for p in per:
        if p.get("invalid"):
            continue
        pool = [r for r in p["rows"] if abs(r["end"] - r["t"] - p["s"]["lag_s"]) <= GROSS_S]
        todo += [(p, r) for r in sorted(rng.sample(pool, min(N_DELAY, len(pool))), key=lambda r: r["wi"])]
    meas = []
    for n, (p, r) in enumerate(todo, 1):
        mid = (r["end"] + r["t"]) / 2
        key = hearer.window(p["id"], mid + WIN_BEFORE - 22.0)     # window [mid - 22, mid + 6]
        sc, tw, ok = match(r["ref"], hearer.cache["windows"][key])
        meas.append({"id": p["id"], "wi": r["wi"], "t": r["t"], "end": r["end"], "score": sc, "w_end": tw,
                     "ok": ok, "valid": sc >= thr and ok and tw is not None})
        if n % 20 == 0:
            print(f"  Whisper (Verzug) {n}/{len(todo)}", flush=True)
    hearer.save()
    est = {}
    for p in per:
        if p.get("invalid"):
            continue
        v = [m for m in meas if m["id"] == p["id"] and m["valid"]]
        tried = sum(1 for m in meas if m["id"] == p["id"])
        if len(v) >= MIN_DELAY:
            est[p["id"]] = {"D": round(float(np.median([m["t"] - m["w_end"] for m in v])), 2),
                            "A": round(float(np.median([m["end"] - m["w_end"] for m in v])), 2),
                            "n": len(v), "tried": tried}
        else:
            est[p["id"]] = {"D": None, "A": None, "n": len(v), "tried": tried}
    return meas, est


def errors(p, variant, est=None):
    """Deviation |word end - mark - correction| per mark of an interview.
    insample: median of the same marks (until 29.09.2026); loo: median of the
    other marks of the same interview; whisper: delay D from the independent
    measurement (if missing, no correction); roh: no correction."""
    err = np.array([r["end"] - r["t"] for r in p["rows"]])
    if variant == "insample":
        return np.abs(err - np.median(err))
    if variant == "loo":
        return np.abs(err - np.array([np.median(np.delete(err, i)) for i in range(len(err))]))
    if variant == "whisper":
        D = (est or {}).get(p["id"], {}).get("D")
        return np.abs(err + (D or 0.0))
    return np.abs(err)


# ------------------------------------------------------------ listening check

PAGE_HELP = """<div class="sub">__N__ Stellen &middot; __DATE__</div>
<details class="help" open><summary>So wird bewertet</summary>
<dl><dt>Aufgabe</dt><dd>Über jedem Fall steht eine Stelle aus dem Transkript (die letzten Wörter vor einer
handgesetzten Zeitmarke). A und B sind zwei Ausschnitte von je 14 Sekunden aus derselben Aufnahme, an zwei
verschiedenen Stellen. Wenn die Stelle in einem Ausschnitt gesprochen wird, endet sie dort etwa zwei
Sekunden vor Schluss. Welcher Ausschnitt welcher ist, ist je Fall zufällig verteilt.</dd>
<dt>Antworten</dt><dd><i>in A</i> / <i>in B</i> – die Stelle ist nur dort zu hören. <i>in beiden</i> – in
beiden zu hören (z. B. wiederholt). <i>in keinem</i> – in keinem der beiden zu hören. <i>unsicher</i> –
nicht zu entscheiden. Dialekt, der hochsprachlich geschrieben ist, und fehlende Füllwörter sind normal.
Die Notiz ist frei.</dd></dl></details>"""


def must_hear(x):
    """Cases the ear always gets: Whisper does not decide, contradicts the
    alignment, or the passage stands in both windows (then the word times
    alone decide)."""
    return x["v_whisper"] != "marke_falsch" or x["hits"] >= 2


def ear_paths(r):
    """Files of listening-check round r: assignment (A/B key), rating, page
    folder, name of the downloaded CSV, prefix of the case IDs."""
    if r == 1:
        return EAR_TSV, EAR_CSV, EAR_DIR, "marker_hoerprobe.csv", "m"
    return (EVAL / f"marker_hoerprobe_{r}.tsv", EVAL / f"marker_hoerprobe_{r}_bewertung.csv",
            ROOT / "samples" / f"marker_{r}", f"marker_hoerprobe_{r}.csv", f"r{r}m")


def ear_rounds():
    """Rounds created: 1, 2, ... as long as the assignment exists."""
    r = 1
    while ear_paths(r)[0].exists():
        r += 1
    return list(range(1, r))


def case_key(x):
    return (x["id"], int(x["wi"]), round(float(x["t"]), 2))


def build_ear(cases, date):
    """Next round of the blind A/B listening check. Round 1: all cases from
    must_hear, plus a random sample of the rest (a quarter, at least 5) as a
    check on Whisper. Every further round: only cases from must_hear that
    have not yet been on any page (e.g. after a rule change). Earlier rounds
    stay untouched."""
    from make_samples import PAGE_AB
    r = len(ear_rounds()) + 1
    tsv, _, folder, download, prefix = ear_paths(r)
    seen = {case_key(m) for q in ear_rounds()
            for m in csv.DictReader(open(ear_paths(q)[0], encoding="utf-8"), delimiter="\t")}
    rng = random.Random(EAR_SEED + r - 1)
    if r == 1:
        sure = [x for x in cases if not must_hear(x)]
        pick = [x for x in cases if must_hear(x)] \
            + rng.sample(sure, min(len(sure), max(5, math.ceil(len(sure) / 4))))
    else:
        pick = [x for x in cases if must_hear(x) and case_key(x) not in seen]
    if not pick:
        print("Keine offenen Faelle - keine neue Runde noetig."); return
    rng.shuffle(pick)
    if folder.exists():
        if any(folder.glob("*.csv")):
            sys.exit(f"{folder.relative_to(ROOT)} enthaelt schon eine Bewertung - nicht ueberschrieben")
        shutil.rmtree(folder)
    audio = folder / "audio"
    audio.mkdir(parents=True)
    cards, mapping = [], []
    for n, x in enumerate(pick, 1):
        fid = f"{prefix}{n:02d}"
        a_is_al = rng.random() < 0.5
        clips = {}
        for side, c in (("a", x["c_al"] if a_is_al else x["c_mk"]), ("b", x["c_mk"] if a_is_al else x["c_al"])):
            s = min(max(0.0, c - CLIP_BEFORE), max(0.0, x["audio_s"] - CLIP_BEFORE - CLIP_AFTER))
            clips[side] = s
            res = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{s:.3f}",
                                  "-t", f"{CLIP_BEFORE + CLIP_AFTER:.3f}", "-i", str(WORK / "audio16k" / f"{x['id']}.wav"),
                                  "-c:a", "flac", str(audio / f"{fid}_{side}.flac")], capture_output=True, text=True)
            if res.returncode:
                sys.exit(f"ffmpeg: {res.stderr[:200]}")
        mapping.append({"fall": fid, "id": x["id"], "wi": x["wi"], "t": x["t"], "a_ist_alignment": int(a_is_al),
                        "start_a": round(clips["a"], 3), "start_b": round(clips["b"], 3), "whisper": x["v_whisper"],
                        "grund": "pflicht" if must_hear(x) else "stichprobe"})
        cards.append(f"""<div class="card" data-id="{fid}">
<div class="top"><b>{fid}</b></div>
<div class="txt">… {html.escape(x['ref_text'])}</div>
<div class="row"><span class="lbl">A</span><audio controls preload="none" src="audio/{fid}_a.flac"></audio></div>
<div class="row"><span class="lbl">B</span><audio controls preload="none" src="audio/{fid}_b.flac"></audio></div>
<div class="row"><button class="v" data-v="a">in A</button><button class="v" data-v="b">in B</button>
<button class="v" data-v="beide">in beiden</button><button class="v" data-v="keine">in keinem</button>
<button class="v" data-v="unsicher">unsicher</button></div>
<input class="note" type="text" placeholder="Notiz (optional)"></div>""")
    page = PAGE_AB
    key = "markerhoerprobe-__DATE__" if r == 1 else f"markerhoerprobe-r{r}-__DATE__"
    for old, new in (("<h1>Nachprüfung: welche Fassung passt besser?</h1>",
                      "<h1>Zeitmarken: wo wird die Stelle gesprochen?</h1>"),
                     ('const KEY="nachpruefung-__DATE__"', f'const KEY="{key}"'),
                     ('"paar;urteil;notiz"', '"fall;urteil;notiz"'),
                     ('a.download="nachpruefung.csv"', f'a.download="{download}"'),
                     ("button.sel[data-v=passt]", "button.v.sel")):
        assert page.count(old) == 1, old
        page = page.replace(old, new)
    page, k = re.subn(r'<div class="sub">.*?</details>', lambda m: PAGE_HELP, page, count=1, flags=re.S)
    assert k == 1
    page = page.replace("__N__", str(len(pick))).replace("__DATE__", date).replace("__CARDS__", "\n".join(cards))
    (folder / "marker_hoerprobe.html").write_text(page, encoding="utf-8")
    with open(tsv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mapping[0]), delimiter="\t"); w.writeheader(); w.writerows(mapping)
    c = {v: sum(1 for m in mapping if m["whisper"] == v) for v in VERDICTS}
    print(f"Runde {r}: {len(pick)} Faelle -> {folder.relative_to(ROOT)}/marker_hoerprobe.html (Zuordnung: "
          f"{tsv.relative_to(ROOT)}); Pflicht {sum(m['grund'] == 'pflicht' for m in mapping)}, Stichprobe "
          f"{sum(m['grund'] == 'stichprobe' for m in mapping)}; laut Whisper "
          + ", ".join(f"{k} {v}" for k, v in c.items() if v))


def import_ear(path):
    """Assign a downloaded rating to the matching round (by the case IDs)."""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline=""), delimiter=";"))
    m = re.fullmatch(r"r(\d+)m\d+|m\d+", rows[0]["fall"]) if rows else None
    if not m:
        sys.exit(f"{path}: keine Faelle erkannt")
    r = int(m.group(1)) if m.group(1) else 1
    tsv, dst = ear_paths(r)[:2]
    if not tsv.exists():
        sys.exit(f"Runde {r} ist nicht angelegt ({tsv.relative_to(ROOT)} fehlt)")
    shutil.copyfile(path, dst)
    print(f"Bewertung Runde {r} uebernommen -> {dst.relative_to(ROOT)}")


def ear_verdicts():
    """Ratings of all rounds -> {(id, wi, t): (verdict, note)}. A/B are mapped
    back: heard in the excerpt at the alignment -> mark wrong, at the mark ->
    alignment wrong, in neither -> both wrong."""
    out = {}
    for r in ear_rounds():
        tsv, rated = ear_paths(r)[:2]
        if not rated.exists():
            continue
        cases = {q["fall"]: q for q in csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t")}
        for q in csv.DictReader(open(rated, encoding="utf-8-sig", newline=""), delimiter=";"):
            m = cases.get(q["fall"])
            if not m or q["urteil"] == "offen":
                continue
            al = m["a_ist_alignment"] == "1"
            v = {"a": "marke_falsch" if al else "alignment_falsch", "b": "alignment_falsch" if al else "marke_falsch",
                 "beide": "unklar", "keine": "beide_falsch", "unsicher": "unklar"}[q["urteil"]]
            out[case_key(m)] = (v, q.get("notiz", ""))
    return out

# ------------------------------------------------------------ main program

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(WORK / "aligned"))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--hoerprobe", action="store_true",
                    help="naechste Runde der blinden A/B-Hoerprobe anlegen (nur offene Faelle)")
    ap.add_argument("--hoerprobe-csv", default=None,
                    help="heruntergeladene marker_hoerprobe*.csv uebernehmen (Runde an den Fall-Kennungen)")
    a = ap.parse_args()
    if a.hoerprobe_csv:
        import_ear(a.hoerprobe_csv)
    per, cases, calm = collect(sorted(Path(a.dir).glob("*.json")))
    hearer = Hearer(a.device)
    controls, info = arbitrate(cases, calm, hearer)
    if a.hoerprobe:
        date = json.loads((ROOT / "dataset" / "manifest.json").read_text())["created"][:10]
        return build_ear(cases, date)
    ear = ear_verdicts()
    for x in cases:
        v, note = ear.get(case_key(x), (None, ""))
        x["v_ohr"], x["notiz"] = v, note
        x["verdict"] = v if v in ("marke_falsch", "alignment_falsch", "beide_falsch") else x["v_whisper"]
    heard = [x for x in cases if x["v_ohr"] is not None]
    info.update({"ohr_n": len(heard),
                 "ohr_entschieden": sum(x["v_ohr"] != "unklar" for x in heard),
                 "ohr_beide_entschieden": sum(x["v_ohr"] != "unklar" and x["v_whisper"] != "unklar" for x in heard),
                 "ohr_gleich_whisper": sum(x["v_ohr"] == x["v_whisper"] for x in heard if x["v_ohr"] != "unklar"
                                           and x["v_whisper"] != "unklar"),
                 "offen_fuers_ohr": sum(must_hear(x) and x["v_ohr"] is None for x in cases)})
    meas, est = delays(per, hearer, info["threshold"])
    info.update({"verzug_messungen": len(meas), "verzug_gueltig": sum(m["valid"] for m in meas),
                 "verzug_je_interview": N_DELAY, "verzug_ohne_schaetzung": sorted(k for k, v in est.items()
                                                                                   if v["D"] is None)})
    # ---- output per interview
    print(f"Schiedsrichter {WHISPER}: Zufallsschwelle {info['threshold']:.0f} ({NULL_Q}. Perzentil von "
          f"{info['null_n']} Zufallspaaren, Median {info['null_median']}); Positivkontrolle: gefunden "
          f"{info['controls_found']} von {info['controls_n']} unstrittigen Marken; Hoerprobe: {len(heard)} Faelle")
    print(f"Verzug unabhaengig gemessen: {info['verzug_gueltig']} von {info['verzug_messungen']} Messungen gueltig"
          + (f"; ohne Schaetzung (keine Korrektur): {', '.join(info['verzug_ohne_schaetzung'])}"
             if info["verzug_ohne_schaetzung"] else ""))
    print(f"\n{'Interview':26} {'n':>4} {'alt':>6} {'D':>6} {'A':>6} {'Mess':>4}   |Fehler| Med / <=2s / <=5s "
          f"(Whisper-Verzug)   roh   {'>10s':>5}")
    out, allrows = [], {"datensatz": [], "ausgeschlossen": []}
    for p in per:
        if p.get("invalid"):
            print(f"{p['id']:26} (Zeitmarken beziehen sich nicht auf diese Aufnahme - nicht gewertet)"); continue
        s, g = p["s"], ("ausgeschlossen" if p["excluded"] else "datensatz")
        e = est[p["id"]]
        mine = [x for x in cases if x["id"] == p["id"]]
        verd = {v: sum(1 for x in mine if x["verdict"] == v) for v in VERDICTS}
        dev = {v: errors(p, v, est) for v in ("insample", "loo", "whisper", "roh")}
        wrong_marker = {(x["wi"], round(x["t"], 2)) for x in mine if x["verdict"] == "marke_falsch"}
        wrong = np.array([(r["wi"], round(r["t"], 2)) in wrong_marker for r in p["rows"]])
        allrows[g].append((dev, wrong))
        dw = dev["whisper"]
        out.append({"id": p["id"], "teil": g, **s, "ausgelassen_nicht_aligniert": p["skipped"],
                    "verzug_whisper_s": e["D"], "versatz_alignment_s": e["A"], "messungen_whisper": e["n"],
                    "median_abs_whisper_s": round(float(np.median(dw)), 2),
                    "p90_abs_whisper_s": round(float(np.percentile(dw, 90)), 2),
                    "within_2s_whisper": round(float(np.mean(dw <= 2)), 3),
                    "within_5s_whisper": round(float(np.mean(dw <= 5)), 3),
                    "median_abs_roh_s": round(float(np.median(dev["roh"])), 2),
                    **{f"grob_{v}": verd[v] for v in VERDICTS}})
        fmt = lambda x: "   -  " if x is None else f"{x:+6.2f}"
        print(f"{p['id']:26} {s['n']:4} {-s['lag_s']:+6.2f} {fmt(e['D'])} {fmt(e['A'])} {e['n']:4}   "
              f"{np.median(dw):5.2f} / {np.mean(dw<=2):4.2f} / {np.mean(dw<=5):4.2f}              "
              f"{np.median(dev['roh']):5.2f} {s['gross_gt_10s']:5}"
              + ("   grob: " + ", ".join(f"{v} {n}" for v, n in verd.items() if n) if mine else "")
              + ("   [ausgeschlossen]" if p["excluded"] else "")
              + (f"   [{p['skipped']} Marken ohne aligniertes Wort]" if p["skipped"] else ""))
    print("  (alt = bisheriger Verzug aus denselben Marken; D = Verzug laut Whisper; A = Versatz Alignment gegen Whisper)")
    for g, parts in allrows.items():
        if not parts:
            continue
        print(f"\n{g.upper()}:")
        for v, name in (("whisper", "Verzug laut Whisper (unabhaengig)"), ("loo", "Verzug leave-one-out"),
                        ("insample", "Verzug aus denselben Marken (alt)"), ("roh", "ohne Korrektur (obere Schranke)")):
            d = np.concatenate([x[0][v] for x in parts])
            print(f"  {name:36} n={len(d)}  Median {np.median(d):.2f} s  p90 {np.percentile(d,90):.2f} s  "
                  f"<=2 s {np.mean(d<=2)*100:.1f}%  <=5 s {np.mean(d<=5)*100:.1f}%")
        ok = np.concatenate([x[0]["whisper"][~x[1]] for x in parts])
        print(f"  {'ohne widerlegte Marken (Whisper-Verzug)':36} n={len(ok)}  Median {np.median(ok):.2f} s  p90 "
              f"{np.percentile(ok,90):.2f} s  <=2 s {np.mean(ok<=2)*100:.1f}%  <=5 s {np.mean(ok<=5)*100:.1f}%")
        mine = [x for x in cases if x["excluded"] == (g == "ausgeschlossen")]
        print("  grobe Faelle: " + ", ".join(f"{v} {sum(1 for x in mine if x['verdict'] == v)}" for v in VERDICTS)
              + " (Whisper allein: " + ", ".join(f"{v} {sum(1 for x in mine if x['v_whisper'] == v)}"
                                                 for v in VERDICTS) + ")")
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
        cols = ["id", "wi", "t", "end", "marke_plus_verzug", "abweichung_s", "soll_text", "whisper_wert",
                "whisper_ende", "d_alignment", "d_marke", "fenster_ueber_schwelle", "urteil_whisper", "urteil_ohr",
                "urteil", "notiz", "whisper_gehoert"]
        with open(Path(a.csv).with_name("marker_grob.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(cols)
            for x in cases:
                w.writerow([x["id"], x["wi"], x["t"], x["end"], x["c_mk"], x["dev"], x["ref_text"], x["w_score"],
                            x["w_end"], x["d_al"], x["d_mk"], x["hits"], x["v_whisper"], x["v_ohr"] or "",
                            x["verdict"], x["notiz"], x["w_text"]])
        with open(Path(a.csv).with_name("marker_kontrolle.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["id", "wi", "t", "end", "abweichung_s", "soll_text", "whisper_wert",
                                            "whisper_ende", "d_alignment", "gefunden"])
            for x in controls:
                w.writerow([x["id"], x["wi"], x["t"], x["end"], x["dev"], x["ref_text"], x["w_score"], x["w_end"],
                            x["d_al"], int(x["found"])])
        with open(Path(a.csv).with_name("marker_verzug.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["id", "wi", "t", "end", "whisper_wert", "whisper_ende",
                                            "wortzeiten_verlaesslich", "gueltig", "marke_minus_whisper",
                                            "alignment_minus_whisper"])
            for m in meas:
                w.writerow([m["id"], m["wi"], m["t"], m["end"], m["score"], m["w_end"], int(m["ok"]), int(m["valid"]),
                            "" if m["w_end"] is None else round(m["t"] - m["w_end"], 2),
                            "" if m["w_end"] is None else round(m["end"] - m["w_end"], 2)])
        write_json(Path(a.csv).with_name("marker_schiedsrichter.json"), info)


if __name__ == "__main__":
    main()
