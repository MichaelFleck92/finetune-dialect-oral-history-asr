#!/usr/bin/env python3
"""
Read transcripts (docx / pdf / rtf / OCR txt) and repair broken ligatures
in 13 PDFs.

The readers return lines or paragraphs, nothing else. Which of them are
header, speaker tag, annotation or conversation text is decided by the parser
on the basis of the interview's spec.
"""
import collections, itertools, re, subprocess, unicodedata, zipfile
from pathlib import Path

# ------------------------------------------------------------------ readers

def docx_paragraphs(path):
    """Paragraph texts from a docx.

    Two traps (both occurred in v1):
    * <w:t...> must not match <w:tab>, <w:tabs>, <w:tblPr> - otherwise the
      expression swallows the tags in between as "text".
    * Text boxes contain <w:p> inside <w:p>; the non-greedy expression then
      ends at the inner </w:p>. Text boxes/images are removed beforehand.
    """
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    xml = re.sub(r"<mc:AlternateContent\b.*?</mc:AlternateContent>", " ", xml, flags=re.S)
    xml = re.sub(r"<w:txbxContent\b.*?</w:txbxContent>", " ", xml, flags=re.S)
    xml = re.sub(r"<w:pict\b.*?</w:pict>", " ", xml, flags=re.S)
    # footnote/endnote references are not text
    xml = re.sub(r"<w:(footnote|endnote)Reference\b[^>]*/>", " ", xml)
    out = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        t = re.sub(r"<w:tab\b[^>]*/?>", "\t", p)
        t = re.sub(r"<w:br\b[^>]*/?>", " ", t)
        s = "".join(re.findall(r"<w:t(?=[ >])[^>]*>(.*?)</w:t>", t, re.S))
        s = (s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
              .replace("&quot;", '"').replace("&apos;", "'"))
        out.append(unicodedata.normalize("NFC", s))
    return out


LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"}


def pdf_lines(path):
    from pdfminer.high_level import extract_text
    txt = extract_text(str(path))
    for k, v in LIGATURES.items():
        txt = txt.replace(k, v)
    txt = re.sub(r"\(cid:\d+\)", " ", txt)
    return [unicodedata.normalize("NFC", l) for l in txt.split("\n")]


def rtf_lines(path):
    """RTF via macOS textutil; the file contains embedded objects on which
    home-made RTF stripping fails."""
    r = subprocess.run(["textutil", "-convert", "txt", "-stdout", str(path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"textutil scheiterte an {path}: {r.stderr[:200]}")
    return [unicodedata.normalize("NFC", l.rstrip()) for l in r.stdout.split("\n")]


def read_lines(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".docx":
        return docx_paragraphs(path)
    if ext == ".rtf":
        return rtf_lines(path)
    if ext == ".txt":
        return [unicodedata.normalize("NFC", l.rstrip())
                for l in path.read_text(encoding="utf-8").split("\n")]
    if ext == ".pdf":
        return pdf_lines(path)
    raise ValueError(f"unbekanntes Format: {path}")

# ------------------------------------------------------------------ ligatures

# The PDFs of one transcription team ("Interview Hr./Fr. X.pdf", 13 of them) use
# fonts with a broken Unicode mapping: ligatures (tt, ti, ft, ff, fi, ...)
# come out as arbitrary replacement characters - and different ones per document:
#   (a separate mapping per document, e.g. @=tt L=ft O=ti or W=ff)
# v1 only knew @0LOSQ in three documents; all other cases ("MuAer",
# "s7mmt", "Karto^eln") went into training unrepaired.
#
# Procedure per document: every character that stands inside a word after a
# letter and does not belong there is tried with all ligatures.
# If one ligature for this character yields mostly known
# words, the character counts as a replacement character of this document.
LIGS = ["tt", "ti", "ft", "ff", "fi", "fl", "ffi", "ffl", "tf", "st", "fb", "fh", "fk", "di", "ch", "tz"]
NOTE_SPAN = re.compile(r"(?:(?<=\s)|^)\[[^\[\]]{1,200}\]")
TOKEN = re.compile(r"\S+")
EDGE = ".,;:!?\"„“”()»«–—…'’‘´"


def build_vocab(texts):
    v = collections.Counter()
    for t in texts:
        for w in re.findall(r"[A-Za-zÄÖÜäöüß]+", t):
            v[w.lower()] += 1
    return v


def _candidate_positions(w, chars=None):
    """Positions of suspicious characters in word w (without edge punctuation)."""
    out = []
    for i in range(1, len(w)):
        c = w[i]
        if chars is not None and c not in chars:
            continue
        if c.isalpha() and c.islower():
            continue
        if c in "-'’‘´.":
            continue
        prev = w[i - 1]
        nxt = w[i + 1] if i + 1 < len(w) else ""
        # two replacement characters in a row ("S7K" = Stift) count as letters
        # for each other
        g_prev = chars is not None and prev in chars
        g_next = chars is not None and nxt in chars
        if c.isalpha():                       # capital letter inside a word
            # When repairing (chars known), also right after the capital
            # initial letter: "SOLing" = Stifting, "OXizier" = Offizier
            # (review 01.10.2026: 17 tokens in 5 PDFs). Not when detecting -
            # there these would mostly be abbreviations.
            after_cap = chars is not None and i == 1 and prev.isupper() and bool(nxt)
            if not (prev.islower() or g_prev or after_cap) or (nxt and not (nxt.islower() or g_next)):
                continue
        else:                                 # digit / special character
            if not (prev.isalpha() or g_prev):
                continue
            if nxt and not (nxt.isalpha() or g_next):
                continue
        out.append(i)
    return out


def _protect_notes(line):
    """Replace real annotations [..] (opening bracket at the start of a word) with
    placeholders so that their brackets do not count as replacement characters."""
    keep = []
    def sub(m):
        keep.append(m.group(0)); return f"\x00{len(keep)-1}\x00"
    return NOTE_SPAN.sub(sub, line), keep


def _restore_notes(line, keep):
    return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], line)


def detect_glyphs(lines, vocab, min_hits=3, min_share=0.4):
    """-> {replacement character: most frequent ligature} for this document.

    Evidence is a known word or a compound with a known head word (many
    affected words are rare or compounds)."""
    known = [w for w, n in vocab.items() if len(w) >= 6 and n >= 2]
    tails = {}
    for k in known:
        tails.setdefault(k[-4:], []).append(k)
    def is_word(c):
        c = c.lower()
        if c in vocab:
            return True
        return any(c.endswith(k) and len(c) > len(k) for k in tails.get(c[-4:], ()))
    hits = collections.defaultdict(collections.Counter)
    occ = collections.Counter()
    text, _ = _protect_notes("\n".join(lines))
    for m in TOKEN.finditer(text):
        w = m.group().strip(EDGE)
        if len(w) < 3 or "@" in w and "." in w or "/" in w:
            continue
        for i in _candidate_positions(w):
            c = w[i]
            occ[c] += 1
            for lg in LIGS:
                if is_word(w[:i] + lg + w[i + 1:]):
                    hits[c][lg] += 1
                    break
    out = {}
    for c, cnt in hits.items():
        lg, n = cnt.most_common(1)[0]
        if n >= min_hits and sum(cnt.values()) / max(1, occ[c]) >= min_share:
            out[c] = lg
    return out


def repair_glyphs(lines, vocab, glyphs):
    """Resolve replacement characters word by word (precedence see fix()).

    -> (lines, {broken: repaired}, {unchecked repairs})
    """
    if not glyphs:
        return list(lines), {}, set()
    chars = set(glyphs)
    cache, fixes, unsure = {}, {}, set()

    def fix(w):
        """Precedence: (1) document rule, if it yields a known word;
        (2) another ligature combination that yields a known word;
        (3) otherwise the document rule (unchecked, reported).
        Compound matches do not count here - they turned "Scho]en" into
        "Schoffen" instead of "Schotten"."""
        if w in cache:
            return cache[w]
        if w[0].isdigit() or not any(ch.isalpha() and ch not in chars for ch in w):
            cache[w] = w; return w            # number: "100", "70er", "40er-Jahre"
        pos = _candidate_positions(w, chars)
        if not pos or len(pos) > 4:
            cache[w] = w; return w
        def build(combo):
            out, k = [], 0
            for i, ch in enumerate(w):
                if k < len(pos) and i == pos[k]:
                    out.append(combo[k]); k += 1
                else:
                    out.append(ch)
            return "".join(out)
        dom = build([glyphs[w[i]] for i in pos])
        if dom.lower() in vocab:
            best = dom
        else:
            options = [[lg for lg in LIGS] for _ in pos]
            exact = [(vocab[c.lower()], c) for c in
                     (build(cb) for cb in itertools.islice(itertools.product(*options), 5000))
                     if c.lower() in vocab]
            if exact:
                best = max(exact)[1]
            else:
                best = dom
                unsure.add(dom)
        cache[w] = best
        if best != w:
            fixes[w] = best
        return best

    # on the whole document, so that annotations wrapped across lines
    # ("[Anm: Stalin-Denkmal,\nRussen-Denkmal]") are protected as a whole
    prot, keep = _protect_notes("\n".join(lines))
    toks = []
    for m in re.finditer(r"\S+|\s+", prot):
        t = m.group()
        if not t.isspace():
            core = t.strip(EDGE)
            if core and any(ch in chars for ch in core[1:]):
                r = fix(core)
                t = t.replace(core, r, 1) if r != core else t
        toks.append(t)
    out = _restore_notes("".join(toks), keep).split("\n")
    assert len(out) == len(lines)
    return out, fixes, unsure
