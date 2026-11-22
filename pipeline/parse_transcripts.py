#!/usr/bin/env python3
"""
Stage 1: transcript -> speaker turns, controlled by the interview's spec.

The spec (pipeline/specs/<id>.json) fixes what v1 guessed: which tags are
speakers, who has which role, where the conversation starts and ends, which
lines are page headers and what individual annotations mean.
The specs are written by pipeline/llm_specs.py (Qwen3-235B from the structural
overview of pipeline/inventory.py); they can be checked by hand.

The parser itself decides nothing about content. It only cuts things out
(header, tags, annotations, time marks) and sets markers; it adds no text.
This is checked at the end: the words of the turns must occur in the same
order in the source text (after the ligatures are repaired).

Output: work/turns/<id>.json - turns with speaker, role, person and text.
The text contains only three markers:
  [unv]  unintelligible/uncertain      -> wildcard in the alignment, segment boundary
  [gap]  speech without text in audio  -> wildcard in the alignment, segment boundary
  [nv]   laughter, gestures, coughing  -> no wildcard (blank frames absorb it), no boundary
Hand-set time marks are removed and only recorded for checking.

  python3 pipeline/parse_transcripts.py [--only ID] [--report]
"""
import argparse, collections, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, WORK, SPECS, load_specs, resolve, write_json, sha1_file,
                    sha1_text, code_hash)
from sources import read_lines, build_vocab, detect_glyphs, repair_glyphs
from notes import classify

OUT = WORK / "turns"
# format profiles per transcript type (tag style, page headers, closing credits): config/profiles.json (local,
# contains names from page headers); template config/profiles.example.json
PROFILES = json.loads((ROOT / "config" / "profiles.json").read_text(encoding="utf-8"))

# ------------------------------------------------------------------ tags

def tolerant(tag):
    """'Hr. Muster' -> Hr[\\s.]*Muster\\.? ; 'A.K.' -> A[\\s.]*K\\.?
    The transcripts write the same tag with varying dots and spaces
    ([A.K] / [A.K.] / [Anna  Muster] / Hr..  Muster)."""
    parts = [p for p in re.split(r"[\s.]+", tag) if p]
    return r"[\s.]*".join(re.escape(p) for p in parts) + r"\.?"


class Tags:
    def __init__(self, spec, style):
        self.style = style
        self.map = {}                       # tag variant -> canonical tag
        for t in spec.get("speakers", {}):
            self.map[t] = t
        for a, t in spec.get("tag_aliases", {}).items():
            self.map[a] = t
        alts = sorted(self.map, key=len, reverse=True)
        pat = "|".join(f"(?P<t{i}>{tolerant(a)})" for i, a in enumerate(alts))
        self.alts = alts
        if style == "bracket":
            self.line = re.compile(r"^\s*\[\s*(?:" + pat + r")\s*\]\s*:?\s*") if alts else None
            self.inline = re.compile(r"(?<=\s)\[\s*(?:" + pat + r")\s*\]\s*:?\s+") if alts else None
        else:
            # "_" instead of ":" is a typo in the transcript ("okay. ZZ_ Und mein Onkel")
            self.line = re.compile(r"^\s*(?:" + pat + r")\s*[:_]\s*") if alts else None
            self.inline = re.compile(r"(?<=[.!?…\"“”)\]»'’])\s+(?:" + pat + r")\s*[:_]\s+") if alts else None
        # embedded interjections "[I: Okay.]" - in both styles
        self.bracket_turn = re.compile(r"\[\s*(?:" + pat + r")\s*:\s*([^\[\]]+?)\s*\]") if alts else None
        self.regex = [(re.compile(r), t) for r, t in spec.get("tag_regex", {}).items()]

    def which(self, m):
        for k, v in m.groupdict().items():
            if v is not None and k.startswith("t"):
                return self.map[self.alts[int(k[1:])]]
        return None

    def match_line(self, s):
        for rx, t in self.regex:
            m = rx.match(s)
            if m:
                return t, m.end()
        if self.line:
            m = self.line.match(s)
            if m:
                return self.which(m), m.end()
        return None, 0

# ------------------------------------------------------------------ time marks

TS_ANY = re.compile(r"\(?#\s*[\d:.,\-]*\d[\d:.,\-]*\s*#?\)?|##")
TS_OK = re.compile(r"^\(?#\s*(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,-](\d{1,2}))?\s*#?\)?$")


def parse_ts(raw):
    m = TS_OK.match(raw.strip())
    if not m:
        return None
    h, mi, s, frac = m.groups()
    t = int(h or 0) * 3600 + int(mi) * 60 + int(s)
    if frac:
        t += int(frac) / (10 ** len(frac))
    return float(t)

# ------------------------------------------------------------------ annotations

NOTE = re.compile(r"\[[^\[\]]*\]|\((?!\s*\d{1,2}\s*\)\d)[^()]*\)")
MIXED = re.compile(r"\[[^\[\]()]{1,80}\)|\([^()\[\]]{1,80}\]")
UNSURE_PREV = re.compile(r"(\S+?)\s*[\[(]\s*\?\s*[\])]")
# Bracket-free placeholders during processing - otherwise the "[gap]"
# inserted for an inner note would prevent the outer note from being
# recognised as a whole ("[Ab 1:20:00 erzaehlt XY ... (wie viele ...) ...]").
PH = {"unv": "\ue000unv\ue001", "gap": "\ue000gap\ue001", "nv": "\ue000nv\ue001"}
MARK = {"unclear": f" {PH['unv']} ", "gap": f" {PH['gap']} ", "nonverbal": f" {PH['nv']} ",
        "pause": " ", "editorial": " "}


def process_text(text, spec, prof, stats):
    """Annotations/time marks/placeholders -> markers. Only removes."""
    for rx, rep in spec.get("text_sub", []):       # deletions only, e.g. OCR prefix "X.:"
        text, n = re.subn(rx, rep, text)
        stats["text_sub"] += n
    ts = []
    def ts_sub(m):
        t = parse_ts(m.group(0))
        ts.append({"raw": m.group(0).strip(), "t": t, "pos": m.start()})
        stats["zeitmarken" if t is not None else "zeitmarken_unlesbar"] += 1
        return " \x01 "
    text = TS_ANY.sub(ts_sub, text)
    text = re.sub(r"\[ocr:(?:[^\[\]]|\[[^\[\]]*\])*\]", f" {PH['unv']} ", text)   # uncertain OCR reading
    text = re.sub(r"\[#\d{1,3}\]", " ", text)                       # counter readings (OCR)
    # "(19)39er": added century like "[19]41" - editorial, what is spoken is "39er"
    text, n = re.subn(r"\(\s*\d{1,2}\s*\)(?=\d)", "", text)
    stats["notiz:editorial"] += n
    text = re.sub(r"\bX{3,}\w*|\bxxx\w*", f" {PH['unv']} ", text)          # placeholders
    # "Schwoich(?)" / "Dirndl [?]": the word before it is uncertain
    text, n = UNSURE_PREV.subn(f" {PH['unv']} ", text)
    stats["unsicher_markiert"] += n
    overrides = spec.get("notes", {})
    bdef = prof.get("bracket_default", "gap")
    def note_sub(m):
        s = m.group(0)
        if s in ("[unv]", "[gap]", "[nv]"):               # markers from the source (OCR)
            return f" {PH[s[1:-1]]} "
        inner, kind = s[1:-1], s[0]
        cls = classify(inner, overrides, bdef, kind)
        stats["notiz:" + cls] += 1
        stats["_notes"][f"{s[:80]} -> {cls}"] += 1
        if cls == "spoken":
            return " " + inner.strip() + " "
        return MARK[cls]
    for _ in range(2):                          # nested brackets
        text = NOTE.sub(note_sub, text)
    text = MIXED.sub(note_sub, text)            # "[Muster oder Mustar)"
    # Unclosed annotation (wrapped across a turn or text boundary):
    # what comes after or before it is not reliably speech.
    if "[" in text or "]" in text:
        stats["_leftover"][text[:100]] += 1
        stats["klammer_offen_als_luecke"] += 1
        i = text.rfind("[")
        if i >= 0 and "]" not in text[i:]:
            text = text[:i] + f" {PH['gap']} "
        j = text.find("]")
        if j >= 0 and "[" not in text[:j]:
            text = f" {PH['gap']} " + text[j + 1:]
        text = text.replace("[", " ").replace("]", " ")
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub("\ue000(unv|gap|nv)\ue001", r"[\1]", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?…“”])", r"\1", text)
    # time marks: record the position as a word index in the cleaned text. The
    # placeholder can stick to punctuation ("\x01…"), so search within the token.
    out_ts, words = [], []
    for tok in text.split(" "):
        while "\x01" in tok:
            t = ts[len(out_ts)] if len(out_ts) < len(ts) else {"raw": "?", "t": None}
            out_ts.append({"raw": t["raw"], "t": t["t"], "word": len(words)})
            tok = tok.replace("\x01", "", 1)
        tok = tok.strip()
        if tok:
            words.append(tok)
    out = " ".join(words).strip()
    if re.search(r"[\x00-\x1f\ue000-\uf8ff]", out):
        raise AssertionError(f"Steuer-/Platzhalterzeichen im Text: {out[:120]!r}")
    return out, out_ts

# ------------------------------------------------------------------ turns

def find_line(lines, prefix, start=0):
    for i in range(start, len(lines)):
        if lines[i].strip().startswith(prefix):
            return i
    return None


def parse(spec, vocab):
    prof = dict(PROFILES[spec["profile"]])
    style = spec.get("tag_style") or prof["tag_style"]
    src = ROOT / spec["transcript"] if spec["transcript"].startswith("work/") else resolve(spec["transcript"])
    raw_lines = read_lines(src)
    stats = collections.Counter()
    stats["_notes"] = collections.Counter(); stats["_leftover"] = collections.Counter()
    glyph = {}
    lines = raw_lines
    gflag = spec.get("glyph_repair")
    if gflag is None:
        gflag = prof.get("glyph_repair", False)
    if gflag:
        gmap = detect_glyphs(lines, vocab)
        lines, fixes, unverified = repair_glyphs(lines, vocab, gmap)
        # Ligatures that come out as a lower-case letter (h/i/l/s/t/x for fh, tf,
        # fb ...) cannot be told apart from real letters automatically -
        # the few cases are listed, checked, in the spec ("Brohabrik": "Brotfabrik")
        for bad, good in spec.get("glyph_words", {}).items():
            rx = re.compile(r"(?<![\wÄÖÜäöüß])" + re.escape(bad) + r"(?![\wÄÖÜäöüß])")
            n = sum(len(rx.findall(l)) for l in lines)
            if n:
                lines = [rx.sub(good, l) for l in lines]
                fixes[bad] = good
        glyph = {"map": gmap, "fixes": fixes, "unverified": sorted(unverified)}
    tags = Tags(spec, style)
    # --- start and end of the conversation
    if spec.get("body_start"):
        b0 = find_line(lines, spec["body_start"])
        if b0 is None:
            raise ValueError(f"{spec['id']}: body_start nicht gefunden: {spec['body_start']!r}")
    else:
        b0 = next((i for i, l in enumerate(lines) if l.strip() and tags.match_line(l.strip())[0]), None)
        if b0 is None:
            raise ValueError(f"{spec['id']}: keine Zeile beginnt mit einem Kuerzel der Spec")
    b1 = len(lines)
    if spec.get("body_end"):
        e = find_line(lines, spec["body_end"], b0 + 1)
        if e is None:
            raise ValueError(f"{spec['id']}: body_end nicht gefunden: {spec['body_end']!r}")
        b1 = e
    if prof.get("body_end"):            # format knowledge (e.g. closing credits of the BIK PDFs): the earlier end applies
        rx = re.compile(prof["body_end"])
        e = next((i for i in range(b0 + 1, b1) if rx.match(lines[i].strip())), None)
        if e is not None:
            b1 = e
    drops = [re.compile(r) for r in prof.get("drop_lines", []) + spec.get("drop_lines", [])]
    gaps = [re.compile(r) for r in spec.get("gap_lines", [])]
    body_raw = []
    for i in range(b0, b1):
        s = lines[i].replace("\x0c", "").strip()
        if not s:
            continue
        if any(rx.search(s) for rx in gaps):          # e.g. "Teil 2": gap, no text
            stats["zeile_als_luecke"] += 1
            body_raw.append((i, "[gap]"))
            continue
        if any(rx.search(s) for rx in drops):
            stats["zeile_verworfen"] += 1
            stats["_notes"][f"(verworfene Zeile) {s[:60]}"] += 1
            continue
        body_raw.append((i, s))
    # --- split into turns
    turns, cur = [], None
    untagged = []
    para = prof.get("join", "paragraph") == "paragraph"
    alt = spec.get("paragraph_speakers")
    for k, (i, s) in enumerate(body_raw):
        if style == "none":
            spk = alt["order"][k % len(alt["order"])] if alt else "UNB"
            turns.append({"tag": spk, "raw": s, "line": i}); continue
        t, end = tags.match_line(s)
        if t:
            if cur:
                turns.append(cur)
            cur = {"tag": t, "raw": s[end:].strip(), "line": i}
        elif cur is not None:
            cur["raw"] += " " + s
            if para and s not in ("[gap]", "[unv]", "[nv]"):
                untagged.append((i, s[:140]))
        else:
            stats["vor_erstem_kuerzel"] += 1
    if cur:
        turns.append(cur)
    # --- split off embedded tags in the middle of a line and "[I: Okay.]"
    split = []
    for t in turns:
        pieces = [(t["tag"], t["raw"])]
        if tags.bracket_turn:
            nxt = []
            for tag, raw in pieces:
                pos = 0
                for m in tags.bracket_turn.finditer(raw):
                    who = tags.which(m)
                    nxt.append((tag, raw[pos:m.start()]))
                    nxt.append((who, m.group(m.lastindex)))
                    stats["einwurf_abgetrennt"] += 1
                    pos = m.end()
                nxt.append((tag, raw[pos:]))
            pieces = nxt
        if tags.inline:
            nxt = []
            for tag, raw in pieces:
                pos, cur_tag = 0, tag
                for m in tags.inline.finditer(raw):
                    nxt.append((cur_tag, raw[pos:m.start()]))
                    cur_tag = tags.which(m)
                    stats["kuerzel_im_satz_getrennt"] += 1
                    pos = m.end()
                nxt.append((cur_tag, raw[pos:]))
            pieces = nxt
        for tag, raw in pieces:
            if raw.strip():
                split.append({"tag": tag, "raw": raw.strip(), "line": t["line"]})
    # --- clean the text, assign speakers
    speakers = spec.get("speakers", {})
    out = []
    for t in split:
        txt, ts = process_text(t["raw"], spec, prof, stats)
        if not re.sub(r"\[(unv|gap|nv)\]", "", txt).strip():
            if txt.strip():
                out.append(None)             # marker only: attach to the previous one
                out[-1] = {"marker_only": txt}
            continue
        sp = speakers.get(t["tag"], {"name": t["tag"], "role": "unbekannt",
                                     "person": f"p_{spec['id']}_{t['tag']}"})
        if t["tag"] not in speakers and t["tag"] != "UNB":
            stats["_unknown_speaker:" + t["tag"]] += 1
        out.append({"speaker": t["tag"], "name": sp["name"], "role": sp["role"],
                    "person": sp["person"], "text": txt, "timestamps": ts, "line": t["line"]})
    # attach marker-only turns ("[gap]" as a line of its own) to the previous turn
    turns2 = []
    for t in out:
        if t is None:
            continue
        if "marker_only" in t:
            if turns2:
                turns2[-1]["text"] += " " + t["marker_only"].strip()
            continue
        turns2.append(t)
    for k, t in enumerate(turns2):
        t["idx"] = k
        t["n_words"] = len(re.sub(r"\[(unv|gap|nv)\]", " ", t["text"]).split())
    # --- verbatim? Output words must be a subsequence of the source text
    src_words = re.findall(r"[\wÄÖÜäöüß]+", " ".join(s for _, s in body_raw).lower())
    out_words = re.findall(r"[\wÄÖÜäöüß]+", " ".join(
        re.sub(r"\[(unv|gap|nv)\]", " ", t["text"]) for t in turns2).lower())
    j = 0
    for w in out_words:
        while j < len(src_words) and src_words[j] != w:
            j += 1
        if j == len(src_words):
            raise AssertionError(f"{spec['id']}: Ausgabe nicht wortgetreu (Wort {w!r} nicht in Reihenfolge)")
        j += 1
    notes = stats.pop("_notes"); left = stats.pop("_leftover")
    rec = {"id": spec["id"], "project": spec["project"], "source": spec["transcript"],
           "source_sha1": sha1_file(src), "spec_sha1": sha1_text(json.dumps(spec, sort_keys=True, default=str)),
           "code": code_hash("parse_transcripts.py", "sources.py", "notes.py"),
           "meta": {k: spec.get(k) for k in ("interviewee", "interviewer", "date", "place", "consent")},
           "speakers": speakers, "turns": turns2, "glyphs": glyph,
           "stats": dict(stats), "body_lines": [b0, b1],
           "untagged_paragraphs": untagged[:200], "notes_seen": dict(notes.most_common()),
           "leftover_brackets": dict(left.most_common(20))}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--report", action="store_true", help="Pruefliste je Interview ausgeben")
    a = ap.parse_args()
    specs = [s for s in load_specs() if not a.only or any(o in s["id"] for o in a.only.split(","))]
    # vocabulary for the ligature repair: only the clean docx transcripts,
    # the same every time (v1 read it from the last run of stage 1)
    vocab = build_vocab(l for s in load_specs() if (s["transcript"] or "").endswith(".docx")
                        for l in read_lines(resolve(s["transcript"])))
    ok, bad = 0, []
    for s in specs:
        if s.get("status") == "draft":
            bad.append((s["id"], "Spec ist noch Entwurf")); continue
        try:
            rec = parse(s, vocab)
        except Exception as e:
            bad.append((s["id"], str(e)[:200])); continue
        write_json(OUT / f"{s['id']}.json", rec)
        ok += 1
        st = rec["stats"]
        spk = collections.Counter()
        for t in rec["turns"]:
            spk[(t["speaker"], t["role"][:3])] += t["n_words"]
        print(f"{s['id']:26} {len(rec['turns']):4} Turns {sum(spk.values()):6} W  "
              + " ".join(f"{k[0]}/{k[1]}:{v}" for k, v in spk.most_common())
              + f" | gap {st.get('notiz:gap',0)} unv {st.get('notiz:unclear',0)+st.get('unsicher_markiert',0)}"
              + (f" | Glyphen {len(rec['glyphs'].get('fixes', {}))}" if rec["glyphs"] else "")
              + (f" | UNTAGGED {len(rec['untagged_paragraphs'])}" if rec["untagged_paragraphs"] else "")
              + (f" | UNBEKANNT {[k for k in st if k.startswith('_unknown')]}" if any(k.startswith('_unknown') for k in st) else ""))
        if a.report:
            for i, u in rec["untagged_paragraphs"][:30]:
                print(f"     untagged z.{i}: {u}")
            for k, v in list(rec["notes_seen"].items())[:60]:
                print(f"     {v:3}  {k}")
            if rec["leftover_brackets"]:
                print(f"     RESTKLAMMERN: {rec['leftover_brackets']}")
    print(f"\n{ok} geparst, {len(bad)} nicht:")
    for b in bad:
        print("   ", b)


if __name__ == "__main__":
    main()
