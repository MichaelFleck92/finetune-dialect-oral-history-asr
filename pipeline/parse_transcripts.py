#!/usr/bin/env python3
"""
Stufe 1: Transkript -> Sprecherturns, gesteuert von der Spec des Interviews.

Die Spec (pipeline/specs/<id>.json) legt fest, was in v1 geraten wurde: welche
Kuerzel Sprecher sind, wer welche Rolle hat, wo das Gespraech beginnt und
endet, welche Zeilen Seitenkopf sind und was einzelne Anmerkungen bedeuten.
Die Specs schreibt pipeline/llm_specs.py (Qwen3-235B aus der Strukturuebersicht
von pipeline/inventory.py); sie sind von Hand pruefbar.

Der Parser selbst entscheidet nichts Inhaltliches. Er schneidet nur heraus
(Kopf, Kuerzel, Anmerkungen, Zeitmarken) und setzt Markierungen; er fuegt
keinen Text hinzu. Das wird am Ende geprueft: die Woerter der Turns muessen
in derselben Reihenfolge im Quelltext stehen (nach Reparatur der Ligaturen).

Ausgabe: work/turns/<id>.json - Turns mit Sprecher, Rolle, Person und Text.
Im Text stehen nur drei Markierungen:
  [unv]  unverstaendlich/unsicher      -> Joker im Alignment, Segmentgrenze
  [gap]  Sprache ohne Text im Audio    -> Joker im Alignment, Segmentgrenze
  [nv]   Lachen, Gesten, Husten        -> kein Joker (Blank-Frames nehmen es auf), keine Grenze
Handgesetzte Zeitmarken werden entfernt und nur zur Kontrolle mitgeschrieben.

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
# Format-Profile je Transkriptart (Kuerzelstil, Seitenkoepfe, Abspann): config/profiles.json (lokal,
# enthaelt Namen aus Seitenkoepfen); Vorlage config/profiles.example.json
PROFILES = json.loads((ROOT / "config" / "profiles.json").read_text(encoding="utf-8"))

# ------------------------------------------------------------------ Kuerzel

def tolerant(tag):
    """'Hr. Muster' -> Hr[\\s.]*Muster\\.? ; 'A.K.' -> A[\\s.]*K\\.?
    Die Transkripte schreiben dasselbe Kuerzel mit wechselnden Punkten und
    Leerzeichen ([A.K] / [A.K.] / [Anna  Kurz] / Hr..  Muster)."""
    parts = [p for p in re.split(r"[\s.]+", tag) if p]
    return r"[\s.]*".join(re.escape(p) for p in parts) + r"\.?"


class Tags:
    def __init__(self, spec, style):
        self.style = style
        self.map = {}                       # Kuerzel-Variante -> kanonisches Kuerzel
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
            # "_" statt ":" ist ein Tippfehler im Transkript ("okay. ZZ_ Und mein Onkel")
            self.line = re.compile(r"^\s*(?:" + pat + r")\s*[:_]\s*") if alts else None
            self.inline = re.compile(r"(?<=[.!?…\"“”)\]»'’])\s+(?:" + pat + r")\s*[:_]\s+") if alts else None
        # eingebettete Einwuerfe "[I: Okay.]" - in beiden Stilen
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

# ------------------------------------------------------------------ Zeitmarken

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

# ------------------------------------------------------------------ Anmerkungen

NOTE = re.compile(r"\[[^\[\]]*\]|\((?!\s*\d{1,2}\s*\)\d)[^()]*\)")
MIXED = re.compile(r"\[[^\[\]()]{1,80}\)|\([^()\[\]]{1,80}\]")
UNSURE_PREV = re.compile(r"(\S+?)\s*[\[(]\s*\?\s*[\])]")
# Waehrend der Verarbeitung klammerfreie Platzhalter - sonst verhindern die
# eingesetzten "[gap]" eines inneren Vermerks, dass der aeussere Vermerk als
# Ganzes erkannt wird ("[Ab 1:20:00 erzaehlt EK ... (wie viele ...) ...]").
PH = {"unv": "\ue000unv\ue001", "gap": "\ue000gap\ue001", "nv": "\ue000nv\ue001"}
MARK = {"unclear": f" {PH['unv']} ", "gap": f" {PH['gap']} ", "nonverbal": f" {PH['nv']} ",
        "pause": " ", "editorial": " "}


def process_text(text, spec, prof, stats):
    """Anmerkungen/Zeitmarken/Platzhalter -> Markierungen. Entfernt nur."""
    for rx, rep in spec.get("text_sub", []):       # nur Loeschungen, z.B. OCR-Praefix "X.:"
        text, n = re.subn(rx, rep, text)
        stats["text_sub"] += n
    ts = []
    def ts_sub(m):
        t = parse_ts(m.group(0))
        ts.append({"raw": m.group(0).strip(), "t": t, "pos": m.start()})
        stats["zeitmarken" if t is not None else "zeitmarken_unlesbar"] += 1
        return " \x01 "
    text = TS_ANY.sub(ts_sub, text)
    text = re.sub(r"\[ocr:(?:[^\[\]]|\[[^\[\]]*\])*\]", f" {PH['unv']} ", text)   # unsichere OCR-Lesung
    text = re.sub(r"\[#\d{1,3}\]", " ", text)                       # Zaehlerstaende (OCR)
    # "(19)39er": ergaenztes Jahrhundert wie "[19]41" - redaktionell, gesprochen wird "39er"
    text, n = re.subn(r"\(\s*\d{1,2}\s*\)(?=\d)", "", text)
    stats["notiz:editorial"] += n
    text = re.sub(r"\bX{3,}\w*|\bxxx\w*", f" {PH['unv']} ", text)          # Platzhalter
    # "Schwoich(?)" / "Dirndl [?]": das Wort davor ist unsicher
    text, n = UNSURE_PREV.subn(f" {PH['unv']} ", text)
    stats["unsicher_markiert"] += n
    overrides = spec.get("notes", {})
    bdef = prof.get("bracket_default", "gap")
    def note_sub(m):
        s = m.group(0)
        if s in ("[unv]", "[gap]", "[nv]"):               # Markierungen aus der Quelle (OCR)
            return f" {PH[s[1:-1]]} "
        inner, kind = s[1:-1], s[0]
        cls = classify(inner, overrides, bdef, kind)
        stats["notiz:" + cls] += 1
        stats["_notes"][f"{s[:80]} -> {cls}"] += 1
        if cls == "spoken":
            return " " + inner.strip() + " "
        return MARK[cls]
    for _ in range(2):                          # verschachtelte Klammern
        text = NOTE.sub(note_sub, text)
    text = MIXED.sub(note_sub, text)            # "[Reiner oder Raner)"
    # Nicht geschlossene Anmerkung (ueber eine Turn- oder Textgrenze
    # umbrochen): was danach bzw. davor steht, ist keine gesicherte Rede.
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
    # Zeitmarken: Position als Wortindex im bereinigten Text festhalten. Der
    # Platzhalter kann an Satzzeichen kleben ("\x01…"), deshalb im Token suchen.
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

# ------------------------------------------------------------------ Turns

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
        # Ligaturen, die als Kleinbuchstabe herauskommen (h/i/l/s/t/x fuer fh, tf,
        # fb ...), sind automatisch nicht von echten Buchstaben zu unterscheiden -
        # die wenigen Faelle stehen gepruefte in der Spec ("Brohabrik": "Brotfabrik")
        for bad, good in spec.get("glyph_words", {}).items():
            rx = re.compile(r"(?<![\wÄÖÜäöüß])" + re.escape(bad) + r"(?![\wÄÖÜäöüß])")
            n = sum(len(rx.findall(l)) for l in lines)
            if n:
                lines = [rx.sub(good, l) for l in lines]
                fixes[bad] = good
        glyph = {"map": gmap, "fixes": fixes, "unverified": sorted(unverified)}
    tags = Tags(spec, style)
    # --- Gespraechsbeginn und -ende
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
    if prof.get("body_end"):            # Formatwissen (z. B. Abspann der BIK-PDFs): das frühere Ende gilt
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
        if any(rx.search(s) for rx in gaps):          # z.B. "Teil 2": Luecke, kein Text
            stats["zeile_als_luecke"] += 1
            body_raw.append((i, "[gap]"))
            continue
        if any(rx.search(s) for rx in drops):
            stats["zeile_verworfen"] += 1
            stats["_notes"][f"(verworfene Zeile) {s[:60]}"] += 1
            continue
        body_raw.append((i, s))
    # --- in Turns zerlegen
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
    # --- eingebettete Kuerzel mitten in der Zeile und "[I: Okay.]" abtrennen
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
    # --- Text bereinigen, Sprecher zuordnen
    speakers = spec.get("speakers", {})
    out = []
    for t in split:
        txt, ts = process_text(t["raw"], spec, prof, stats)
        if not re.sub(r"\[(unv|gap|nv)\]", "", txt).strip():
            if txt.strip():
                out.append(None)             # reine Markierung: an den Vorgaenger haengen
                out[-1] = {"marker_only": txt}
            continue
        sp = speakers.get(t["tag"], {"name": t["tag"], "role": "unbekannt",
                                     "person": f"p_{spec['id']}_{t['tag']}"})
        if t["tag"] not in speakers and t["tag"] != "UNB":
            stats["_unknown_speaker:" + t["tag"]] += 1
        out.append({"speaker": t["tag"], "name": sp["name"], "role": sp["role"],
                    "person": sp["person"], "text": txt, "timestamps": ts, "line": t["line"]})
    # reine Markierungs-Turns ("[gap]" als eigene Zeile) dem vorigen Turn anhaengen
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
    # --- Wortgetreu? Ausgabewoerter muessen Teilfolge des Quelltexts sein
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
    # Wortschatz fuer die Ligaturreparatur: nur die sauberen docx-Transkripte,
    # jedes Mal gleich (v1 las ihn aus dem letzten Lauf von Stufe 1)
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
