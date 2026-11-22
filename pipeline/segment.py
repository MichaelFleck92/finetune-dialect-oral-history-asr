#!/usr/bin/env python3
"""
Stage 4: word times -> training segments (audio/text pairs).

Cutting rules
  * never across a speaker change, never across [unv]/[gap] (there is speech
    without text there), never across unaligned words or pauses > HARD_GAP
  * the word right next to an [unv]/[gap] is dropped: its boundary towards the
    wildcard is uncertain (v1 kept these segments, 724 of them); likewise the
    neighbours of a stretched word (> MAX_WORD_S) - the alignment is wrong
    there, and the listening check found wrong cuts at such places
  * segment edges are checked (listening check 28.09.2026: 23 of 80 segments
    flagged, mostly at the edges):
    - edge words are dropped as long as their CTC score is below the 10th
      percentile of their speaker (and below -3): position uncertain, often a
      short word on the wrong side of a pause. Relative to the speaker, because
      a fixed threshold (-5) cost 11 % of the material for dialect and 5 % for
      standard German - the same bias as the absolute filter in v1.
    - At a speaker change without a pause (distance < EDGE_GAP and the greedy
      decoding hears speech just outside) the first or last word is dropped -
      the voices overlap there. Trimming up to the next real pause cost 4-7 %
      of the material, again mostly for dialect, for few additionally fixed
      cases.
    Whatever becomes too short as a result is dropped.
  * Pauses inside a segment are listened to: if the greedy decoding yields
    letters there, there is speech without text. From MIN_LETTERS (about one
    word) on, the segment is split; shorter remnants ("aeh", "mhm") stay - the
    transcripts never write fillers, and the model should learn that.
  * length up to MAX_S (Whisper window 30 s), cut no earlier than after MIN_CUT_S, preferably at
    sentence end + pause
  * adaptive edges: never beyond the middle of the pause to the neighbouring
    word, so no overlap (v1: 925 overlapping pairs). A little more on the
    right, because CTC word ends come early. At the segment end the padding
    also ends before the next speech the greedy decoding hears: the second
    listening check (28.09.2026) found four of eight edge errors where the
    padding reached into untranscribed speech ("weil | dann haben", "oder |
    halt"). v2.2 limited both edges; the blind A/B re-check gave clearly
    better cuts at the end, none at the start (1 better, 1-2 worse) -
    CTC word starts tend to come late, a tighter start clips the onset.
    Since v2.3 only the end is limited.

Quality per segment (basis for quality/keep, see keep_rule):
  score      mean CTC score of the words
  score_rel  score minus the median of the segments of the same person in the
             interview - measures "worse than usual for this speaker" instead of
             "hard to understand". v1 filtered absolutely and thus mostly threw
             away dialect (yield 34-43 % for strong dialect, ~90 % for standard German).
  edge_min   worst score of the first/last two words (segment edges)

  python3 pipeline/segment.py [--only ID]
"""
import argparse, collections, json, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, load_specs, write_json, sha1_file, code_hash

FPS = 50.0
# Longer segments (16.11.2026, variant F): cutting starts only at 28 s, and no earlier than after
# MIN_CUT_S; pauses split only from 4 s. Before: target 20 s, cut from 4 s, pause 2 s -> median 7.8 s,
# longest 20.4 s; now median 10.3 s, mean 12.0 s, 67 % of the time in segments of 15 s or more (before 35 %).
MIN_S, TARGET_S, MAX_S = 1.5, 28.0, 28.0
MIN_CUT_S = 16.0        # earliest cut point within a long run
HARD_GAP = 4.0          # longer pause: always split (pauses containing speech are split by MIN_LETTERS anyway)
MAX_WORD_S = 1.5        # longer single word = stretched, alignment not credible there
MIN_LETTERS = 5         # this many letters in a pause = untranscribed word
PAD_L, PAD_R = 0.15, 0.30
EDGE_Q = 0.10           # edge word below this percentile of its speaker ...
EDGE_SC_MAX = -3.0      # ... and below this value: position uncertain -> drop
EDGE_GAP = 0.15         # speaker change with a shorter distance ...
EDGE_LETTERS = 3        # ... and this many letters just outside = overlap
SPEECH_MARGIN = 0.04    # distance of the cut from the next heard speech
FILLERS = {"äh", "ähm", "öhm", "ehm", "ahm", "hm", "hmm", "mhm", "mh", "äähm", "ähh", "hmhm", "aha"}
FILLERS.discard("aha")  # 'aha' is a word


def speech_letters(emis, inv, f0, f1, blank=0):
    """Letters of the greedy decoding in the frame range [f0, f1)."""
    if f1 - f0 < 3:
        return ""
    ids = emis[f0:f1].argmax(-1)
    out, prev = [], -1
    for k in ids.tolist():
        if k != prev and k != blank:
            ch = inv.get(k, "")
            if len(ch) == 1 and ch.isalpha():
                out.append(ch)
        prev = k
    return "".join(out)


def _is_letter(ch):
    return len(ch) == 1 and ch.isalpha()


def speech_after(emis, inv, t_end, limit):
    """Time of the first letter of other speech after the word end t_end (up to
    limit), else None. First wait for the word-end character or >= 3 empty
    frames - the word itself often still lingers in the decoding."""
    f0 = max(0, int(t_end * FPS) - 2)
    ids = emis[f0:max(f0, int(limit * FPS) + 1)].argmax(-1).tolist()
    done, blank = False, 0
    for k, i in enumerate(ids):
        ch, t = inv.get(i, ""), (f0 + k) / FPS
        if not done:
            if ch == "|":
                done = True
            elif i == 0 and t > t_end:
                blank += 1
                done = blank >= 3
            elif _is_letter(ch):
                blank = 0
        elif _is_letter(ch):
            return t
    return None


def norm_text(tokens):
    """Target text for training: as in the transcript, but uniform across all
    projects - fillers (written only in Rechnitz) and hesitation marks
    (… / ... / free-standing dashes) are dropped."""
    out = []
    # punctuation tokens hang on the preceding word in the alignment ("geschluckt. -"):
    # split at spaces first, otherwise free-standing dashes remain
    tokens = [x for t in tokens for x in t.split()]
    # "Quartier…mhm" is one token: split at hesitation marks first
    tokens = [x for t in tokens for x in re.split(r"(?<=…)|(?<=\.\.\.)", t) if x]
    for t in tokens:
        core = t.strip(".,;:!?…\"„“”»«()–-").lower()
        if core in FILLERS:
            # attach the filler's punctuation to the previous word
            tail = re.sub(r"^[^.,;:!?]*", "", t.rstrip("…"))
            if out and tail and out[-1][-1:] not in ".,;:!?":
                out[-1] += tail
            continue
        t = t.replace("…", " ").replace("...", " ")
        if re.fullmatch(r"[-–—]+", t.strip(" .,;:!?")):
            # free-standing dash, also with punctuation ("-,", "--,"; 01.10.2026):
            # the punctuation goes to the previous word, as with fillers
            tail = re.sub(r"^[^.,;:!?]*", "", t.strip())
            if out and tail and out[-1][-1:] not in ".,;:!?":
                out[-1] += tail
            continue
        out.append(t.strip())
    s = " ".join(x for x in out if x)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;:])(?=[,.;:!?])", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,;:")
    # dash on the last word = broken-off sentence ("und-", "haben Sie-?"); it has
    # no sound (listening check 01.10.2026: 296 target texts ended like this). Dashes
    # inside stay - often a suspended hyphen ("Nacht- und Nebelaktion").
    while True:                                  # until nothing more drops off ("vor- --,")
        t = re.sub(r"[-–—]+(?=[.?!…\"“”»«')]*$)", "", s).rstrip(" ,;:")
        if t == s:
            return s
        s = t


def truncation(tokens):
    """Word break-off 'kein-' - but not a suspended hyphen ('Nacht- und Nebel')."""
    for a, b in zip(tokens, tokens[1:] + [""]):
        if re.search(r"\w-$", a) and b.lower().strip(",.") not in ("und", "oder", "bzw", "bis", "sowie"):
            return True
    return False


def runs(W, marks_hard, emis, inv, stats):
    """Words -> runs that must not be split."""
    out, cur = [], []
    drop_next = False
    def close():
        nonlocal cur
        if cur:
            out.append(cur)
        cur = []
    for k, w in enumerate(W):
        if "s" not in w:
            close(); stats["wort_nicht_aligniert"] += 1; continue
        if (w["e"] - w["s"]) > MAX_WORD_S:
            # stretched: as with the wildcard, the neighbours are dropped too
            if cur:
                cur.pop(); stats["wort_neben_gedehnt"] += 1
            close(); stats["wort_gedehnt"] += 1
            drop_next = True
            continue
        if drop_next:
            drop_next = False
            stats["wort_neben_gedehnt"] += 1
            continue
        if w["i"] in marks_hard:            # wildcard before this word
            # the word before (already in the run) and this word are dropped
            if cur:
                cur.pop(); stats["wort_neben_joker"] += 1
            close(); stats["wort_neben_joker"] += 1
            continue
        if cur:
            prev = cur[-1]
            gap = w["s"] - prev["e"]
            if w["spk"] != prev["spk"] or gap > HARD_GAP:
                close()
            elif gap > 0.25:
                letters = speech_letters(emis, inv, int(prev["e"] * FPS) + 2, int(w["s"] * FPS) - 1)
                if len(letters) >= MIN_LETTERS:
                    close(); stats["pause_mit_sprache_getrennt"] += 1
                elif letters:
                    w = dict(w); w["_filler_before"] = letters
                    stats["kurzer_rest_in_pause"] += 1
        cur.append(w)
    close()
    return out


def no_pause(w, nb, side, emis, inv):
    """Is there no pause between w and the outer neighbouring word nb?"""
    gap = (w["s"] - nb["e"]) if side < 0 else (nb["s"] - w["e"])
    if gap >= EDGE_GAP:
        return False
    # just outside (0.06 s distance for the fuzzy CTC boundaries, 0.4 s window)
    t0, t1 = (w["s"] - 0.46, w["s"] - 0.06) if side < 0 else (w["e"] + 0.06, w["e"] + 0.46)
    return len(speech_letters(emis, inv, max(0, int(t0 * FPS)), int(t1 * FPS))) >= EDGE_LETTERS


def edge_thresholds(W):
    """CTC threshold for edge words per speaker (see EDGE_Q)."""
    by = collections.defaultdict(list)
    for w in W:
        if "sc" in w:
            by[w["spk"]].append(w["sc"])
    allsc = [x for v in by.values() for x in v]
    return {k: min(EDGE_SC_MAX, float(np.quantile(v if len(v) >= 200 else allsc, EDGE_Q)))
            for k, v in by.items()}


def trim_edges(seg, aligned, pos, thr, emis, inv, stats):
    """Drop uncertain edge words (see the module description)."""
    for side in (-1, +1):
        if not seg:
            break
        w = seg[0] if side < 0 else seg[-1]
        k = pos[w["i"]] + side
        nb = aligned[k] if 0 <= k < len(aligned) else None
        if nb is not None and nb["spk"] != w["spk"] and no_pause(w, nb, side, emis, inv):
            seg = seg[1:] if side < 0 else seg[:-1]
            stats["rand_weg_sprecherwechsel"] += 1
        while seg:
            w = seg[0] if side < 0 else seg[-1]
            if w["sc"] >= thr[w["spk"]]:
                break
            seg = seg[1:] if side < 0 else seg[:-1]
            stats["rand_weg_ctc"] += 1
    return seg


def cut(run):
    segs, cur = [], []
    for w in run:
        if cur and w["e"] - cur[0]["s"] > MAX_S:
            # hard limit reached: cut at the best point, if necessary here
            # (without "or len(cur)" the words ended up in two segments)
            bi = best_cut(cur) or len(cur)
            segs.append(cur[:bi]); cur = cur[bi:]
        cur.append(w)
        if cur[-1]["e"] - cur[0]["s"] >= TARGET_S:
            bi = best_cut(cur)
            if bi:
                segs.append(cur[:bi]); cur = cur[bi:]
    if cur:
        segs.append(cur)
    return segs


def best_cut(cur):
    best, bi = -1.0, None
    for j in range(len(cur) - 1, 0, -1):
        if cur[j - 1]["e"] - cur[0]["s"] < MIN_CUT_S:
            break
        gap = cur[j]["s"] - cur[j - 1]["e"]
        if gap < 0.15:
            continue
        sc = gap + (1.0 if re.search(r"[.!?…]$", cur[j - 1]["w"]) else
                    0.4 if re.search(r"[,;:–-]$", cur[j - 1]["w"]) else 0.0)
        if sc > best:
            best, bi = sc, j
    return bi if best >= 0.3 else None


def build(iid):
    A = json.loads((WORK / "aligned" / f"{iid}.json").read_text())
    T = json.loads((WORK / "turns" / f"{iid}.json").read_text())
    em = json.loads((WORK / "emissions" / f"{iid}.json").read_text())
    emis = np.load(WORK / "emissions" / f"{iid}.npy").astype(np.float32)
    inv = {v: k for k, v in em["vocab"].items()}
    turns = {t["idx"]: t for t in T["turns"]}
    W = A["words"]
    hard = {m["word"] for m in A["marks"] if m["kind"] in ("unv", "gap")}
    stats = collections.Counter()
    aligned = [w for w in W if "s" in w]
    pos = {w["i"]: k for k, w in enumerate(aligned)}
    thr = edge_thresholds(W)
    segs = []
    for run in runs(W, hard, emis, inv, stats):
        for seg in cut(run):
            seg = trim_edges(seg, aligned, pos, thr, emis, inv, stats)
            if len(seg) < 2:
                stats["zu_kurz_woerter"] += 1; continue
            segs.append(seg)
    # edges: never beyond half the pause to the neighbouring word (across segments too)
    out = []
    for seg in segs:
        k0, k1 = pos[seg[0]["i"]], pos[seg[-1]["i"]]
        prev_e = aligned[k0 - 1]["e"] if k0 > 0 else 0.0
        next_s = aligned[k1 + 1]["s"] if k1 + 1 < len(aligned) else A["audio_s"]
        s = max(0.0, seg[0]["s"] - min(PAD_L, max(0.0, (seg[0]["s"] - prev_e) / 2)))
        e = min(A["audio_s"], seg[-1]["e"] + min(PAD_R, max(0.0, (next_s - seg[-1]["e"]) / 2)))
        # padding at the end never reaches into speech the decoding hears afterwards
        # (only shorten, never extend; not at the start, see the module description)
        t = speech_after(emis, inv, seg[-1]["e"], e + 0.05)
        if t is not None and t - SPEECH_MARGIN < e:
            e = min(e, max(seg[-1]["e"] + 0.02, t - SPEECH_MARGIN)); stats["polster_vor_sprache_gekuerzt"] += 1
        dur = e - s
        if dur < MIN_S:
            stats["zu_kurz_s"] += 1; continue
        toks = [w["w"] for w in seg]
        text_v = re.sub(r"\s+([,.;:!?…])", r"\1", " ".join(toks)).strip()
        text = norm_text(toks)
        if len(text.split()) < 2:
            stats["zu_kurz_text"] += 1; continue
        if re.search(r"[\[\]#{}]", text):
            stats["restauszeichnung"] += 1; continue
        t = turns[seg[0]["turn"]]
        scs = [w["sc"] for w in seg]
        edge = scs[:2] + scs[-2:]
        gaps = [b["s"] - a["e"] for a, b in zip(seg, seg[1:])]
        out.append({
            "start": round(s, 3), "end": round(e, 3), "duration": round(dur, 3),
            "text": text, "text_verbatim": text_v,
            "speaker": t["speaker"], "speaker_name": t["name"], "role": t["role"], "person": t["person"],
            "turn": t["idx"], "n_words": len(seg), "words_per_s": round(len(seg) / dur, 2),
            "score": round(float(np.mean(scs)), 4), "edge_min": round(float(min(edge)), 3),
            "max_gap": round(max(gaps) if gaps else 0.0, 2),
            "fillers_untranscribed": sum(1 for w in seg if w.get("_filler_before")),
            "has_truncation": truncation(toks),
            "words": [[w["w"], round(w["s"] - s, 3), round(w["e"] - s, 3)] for w in seg],
        })
    # relative score per person
    by = collections.defaultdict(list)
    for o in out:
        by[o["person"]].append(o["score"])
    med = {k: float(np.median(v)) for k, v in by.items()}
    for o in out:
        o["score_rel"] = round(o["score"] - med[o["person"]], 4)
    return A, out, stats


def keep_rule(o):
    """-> (keep?, level, reason).

    Set on 27.09.2026 with `eval_segments.py --calibrate` (Whisper-small,
    duration-proportional sample; "bad" = CER more than 0.3 above the
    median of the other segments of the same interview, i.e. a mismatched pair
    rather than dialect). The figures - extrapolated to the whole set since
    01.10.2026 - are in work/eval/report.md ("Kalibrierung der Verwerfregel"):
    the chosen rule is more precise than v1 (score < -3.0, mostly hits dialect)
    and loses less good material; edge_min < -8 hardly hit bad segments and has
    been removed. Only a small part of the bad material is caught.
    Caution: thresholds and figures come from the same sample
    (in-sample) - a reliable estimate needs a new one.
    """
    if not (0.7 <= o["words_per_s"] <= 5.5):
        return False, None, "sprechrate"
    if o["score"] < -4.5:
        return False, None, "score_absolut"
    if o["score_rel"] < -1.5:
        return False, None, "score_relativ"
    q = "hoch" if o["score_rel"] >= -0.5 else "mittel"
    return True, q, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    outdir = WORK / "segments"; outdir.mkdir(parents=True, exist_ok=True)
    tot = collections.Counter(); hours = collections.Counter()
    for spec in load_specs():
        iid = spec["id"]
        if a.only and not any(o in iid for o in a.only.split(",")):
            continue
        if not (WORK / "aligned" / f"{iid}.json").exists():
            continue
        A, segs, stats = build(iid)
        for o in segs:
            keep, q, why = keep_rule(o)
            if spec.get("exclude"):             # whole interview excluded (reason in the spec)
                keep, q, why = False, None, "interview_ausgeschlossen"
            o["keep"], o["quality"], o["drop_reason"] = keep, q, why
        write_json(outdir / f"{iid}.json", {
            "id": iid, "audio_s": A["audio_s"], "segments": segs, "stats": dict(stats),
            "aligned_sha1": sha1_file(WORK / "aligned" / f"{iid}.json"),
            "code": code_hash("segment.py")})
        k = [o for o in segs if o["keep"]]
        tot["segmente"] += len(segs); tot["behalten"] += len(k)
        hours["alle"] += sum(o["duration"] for o in segs); hours["behalten"] += sum(o["duration"] for o in k)
        print(f"{iid:26} {len(segs):5} Segm. {len(k):5} behalten {sum(o['duration'] for o in k)/3600:5.2f} h "
              f"(Audio {A['audio_s']/3600:4.2f} h, {100*sum(o['duration'] for o in k)/A['audio_s']:3.0f}%)  "
              + " ".join(f"{kk}={v}" for kk, v in sorted(stats.items())))
    print(f"\nGESAMT {tot['segmente']} Segmente, {tot['behalten']} behalten, "
          f"{hours['behalten']/3600:.2f} h von {hours['alle']/3600:.2f} h")


if __name__ == "__main__":
    main()
