#!/usr/bin/env python3
"""
Transkripte einlesen (docx / pdf / rtf / OCR-txt) und kaputte Ligaturen
in 13 PDFs reparieren.

Die Leser liefern Zeilen bzw. Absaetze, sonst nichts. Was davon Kopf,
Sprecherkuerzel, Anmerkung oder Gespraechstext ist, entscheidet der Parser
anhand der Spec des Interviews.
"""
import collections, itertools, re, subprocess, unicodedata, zipfile
from pathlib import Path

# ------------------------------------------------------------------ Leser

def docx_paragraphs(path):
    """Absatztexte aus einer docx.

    Zwei Fallen (beide in v1 aufgetreten):
    * <w:t...> darf nicht auf <w:tab>, <w:tabs>, <w:tblPr> passen - sonst
      frisst der Ausdruck die dazwischenliegenden Tags als "Text".
    * Textfelder enthalten <w:p> in <w:p>; der nicht-greedy Ausdruck endet dann
      am inneren </w:p>. Textfelder/Bilder werden vorher entfernt.
    """
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    xml = re.sub(r"<mc:AlternateContent\b.*?</mc:AlternateContent>", " ", xml, flags=re.S)
    xml = re.sub(r"<w:txbxContent\b.*?</w:txbxContent>", " ", xml, flags=re.S)
    xml = re.sub(r"<w:pict\b.*?</w:pict>", " ", xml, flags=re.S)
    # Fussnoten-/Endnotenverweise sind kein Text
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
    """RTF ueber macOS textutil; die Datei enthaelt eingebettete Objekte, an
    denen eigene RTF-Entkleidung scheitert."""
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

# ------------------------------------------------------------------ Ligaturen

# Die PDFs eines Transkriptionsteams ("Interview Hr./Fr. X.pdf", 13 Stueck) benutzen
# Schriften mit kaputter Unicode-Zuordnung: Ligaturen (tt, ti, ft, ff, fi, ...)
# kommen als beliebige Ersatzzeichen heraus - und zwar je Dokument andere:
#   (je Dokument eine eigene Zuordnung, z. B. @=tt L=ft O=ti oder W=ff)
# v1 kannte nur @0LOSQ in drei Dokumenten; alle anderen Faelle ("MuAer",
# "s7mmt", "Karto^eln") gingen unrepariert ins Training.
#
# Vorgehen je Dokument: jedes Zeichen, das mitten im Wort nach einem
# Buchstaben steht und dort nicht hingehoert, wird mit allen Ligaturen
# probiert. Ergibt eine Ligatur fuer dieses Zeichen ueberwiegend bekannte
# Woerter, gilt das Zeichen als Ersatzzeichen dieses Dokuments.
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
    """Positionen verdaechtiger Zeichen im Wort w (ohne Randsatzzeichen)."""
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
        # zwei Ersatzzeichen hintereinander ("S7K" = Stift) zaehlen fuereinander
        # als Buchstaben
        g_prev = chars is not None and prev in chars
        g_next = chars is not None and nxt in chars
        if c.isalpha():                       # Grossbuchstabe mitten im Wort
            # Beim Reparieren (chars bekannt) auch direkt nach dem grossen
            # Anfangsbuchstaben: "SOLing" = Stifting, "OXizier" = Offizier
            # (Review 01.10.2026: 17 Token in 5 PDFs). Beim Erkennen nicht -
            # dort waeren das meist Abkuerzungen.
            after_cap = chars is not None and i == 1 and prev.isupper() and bool(nxt)
            if not (prev.islower() or g_prev or after_cap) or (nxt and not (nxt.islower() or g_next)):
                continue
        else:                                 # Ziffer / Sonderzeichen
            if not (prev.isalpha() or g_prev):
                continue
            if nxt and not (nxt.isalpha() or g_next):
                continue
        out.append(i)
    return out


def _protect_notes(line):
    """Echte Anmerkungen [..] (oeffnende Klammer am Wortanfang) durch
    Platzhalter ersetzen, damit ihre Klammern nicht als Ersatzzeichen gelten."""
    keep = []
    def sub(m):
        keep.append(m.group(0)); return f"\x00{len(keep)-1}\x00"
    return NOTE_SPAN.sub(sub, line), keep


def _restore_notes(line, keep):
    return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], line)


def detect_glyphs(lines, vocab, min_hits=3, min_share=0.4):
    """-> {Ersatzzeichen: haeufigste Ligatur} fuer dieses Dokument.

    Als Beleg zaehlt ein bekanntes Wort oder ein Kompositum mit bekanntem
    Grundwort (viele betroffene Woerter sind selten oder zusammengesetzt)."""
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
    """Ersatzzeichen je Wort aufloesen (Vorrang siehe fix()).

    -> (Zeilen, {kaputt: repariert}, {ungepruefte Reparaturen})
    """
    if not glyphs:
        return list(lines), {}, set()
    chars = set(glyphs)
    cache, fixes, unsure = {}, {}, set()

    def fix(w):
        """Vorrang: (1) Dokument-Regel, wenn sie ein bekanntes Wort ergibt;
        (2) eine andere Ligatur-Kombination, die ein bekanntes Wort ergibt;
        (3) sonst die Dokument-Regel (ungeprueft, wird gemeldet).
        Komposita-Treffer entscheiden hier nicht mit - sie liessen "Scho]en"
        zu "Schoffen" statt "Schotten" werden."""
        if w in cache:
            return cache[w]
        if w[0].isdigit() or not any(ch.isalpha() and ch not in chars for ch in w):
            cache[w] = w; return w            # Zahl: "100", "70er", "40er-Jahre"
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

    # auf dem ganzen Dokument, damit ueber Zeilen umbrochene Anmerkungen
    # ("[Anm: Stalin-Denkmal,\nRussen-Denkmal]") als Ganzes geschuetzt sind
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
