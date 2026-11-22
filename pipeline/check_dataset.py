#!/usr/bin/env python3
"""
Final check of the delivered dataset. Exits with an error code if anything
is wrong.

  * Files: every row has its file, no file without a row
  * Content: EVERY FLAC is decoded and compared bit for bit with the excerpt
    from work/audio16k - same samples, no offset, exact length
    (v1 checked the duration of 25 samples). This does not check against the
    original recordings: the conversion to work/audio16k (stage 2) resamples.
  * no overlap between two segments of the same interview
  * Texts: no markup, no abbreviations, no fillers, no replacement
    characters of broken ligatures, no placeholders (XXX, (?)), no dash at
    the end (broken-off sentence)
  * Split: no person (except interviewers) and no group of persons in more
    than one part; no interviewer segments in dev/test

  python3 pipeline/check_dataset.py [--quick]
"""
import argparse, collections, json, re, subprocess, sys, wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, DATASET, SR

BAD_TEXT = [
    ("Auszeichnung", re.compile(r"[\[\]{}#␣␦]")),
    ("Steuerzeichen", re.compile(r"[\x00-\x1f\x7f\u200b-\u200f\ue000-\uf8ff]")),
    ("Kuerzel im Text", re.compile(r"(^|\s)(I|ZZ|[A-ZÄÖÜ]\.[A-ZÄÖÜ]\.?|Hr\.\s*\w+|Fr\.\s*\w+)\s*[:_]\s")),
    ("freistehender Strich", re.compile(r"(^|\s)[-–—]+[.,;:!?…]*(\s|$)")),
    ("Strich am Ende", re.compile(r"[-–—][.?!…\"“”»«')]*$")),
    ("Jahrhundert-Ergaenzung", re.compile(r"\(\s*\d{1,2}\s*\)\d")),
    ("Platzhalter", re.compile(r"XXX|\(\?\)|\?\)")),
    ("Fuellwort", re.compile(r"\b(äh|ähm|öhm|ehm|ahm|mhm|hmm?)\b", re.I)),
    ("Ligatur-Ersatzzeichen", re.compile(r"[a-zäöüß][@\\`^\]\[][a-zäöüß]|[a-zäöüß][A-Z0-9@][a-zäöüß]{2}")),
]


def glyph_residue(word, chars):
    """Is a replacement character of this document still inside a word?
    Deliberately written independently of sources.py - the regex above missed
    cases right after the capital initial letter ("SOfting", "OXizier"; review
    01.10.2026)."""
    if not word or word[0].isdigit():          # digits ("100" if 0 = tt)
        return False
    for i in range(1, len(word)):
        c = word[i]
        if c not in chars:
            continue
        prev, nxt = word[i - 1], word[i + 1] if i + 1 < len(word) else ""
        g_prev = prev in chars and i > 1       # the initial letter is never a replacement character ("SS", S = ti)
        if nxt and not (nxt.islower() or nxt in chars):
            continue
        if c.isalpha():                        # not in abbreviations ("PKWs", "OK")
            if prev.islower() or g_prev or (i == 1 and prev.isupper() and nxt):
                return True
        elif prev.isalpha() or g_prev:
            return True
    return False


def decode(p):
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(p), "-f", "s16le", "-ac", "1",
                        "-ar", str(SR), "-"], capture_output=True)
    return np.frombuffer(r.stdout, dtype="<i2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Inhaltsvergleich nur fuer 300 Dateien")
    a = ap.parse_args()
    rows = [json.loads(l) for l in (DATASET / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    fails = []
    print(f"Segmente laut metadata.jsonl: {len(rows)}")
    names = [r["file_name"] for r in rows]
    if len(set(names)) != len(names):
        fails.append("doppelte Dateinamen")
    on_disk = {str(p.relative_to(DATASET)) for p in (DATASET / "audio").rglob("*.flac")}
    miss = set(names) - on_disk; extra = on_disk - set(names)
    if miss:
        fails.append(f"{len(miss)} Dateien fehlen")
    if extra:
        fails.append(f"{len(extra)} Dateien ohne Zeile")
    print(f"Dateien: {len(on_disk)}  fehlend {len(miss)}  ueberzaehlig {len(extra)}")

    # content, bit for bit
    by_iv = collections.defaultdict(list)
    for r in rows:
        by_iv[r["interview"]].append(r)
    todo = rows if not a.quick else [rows[i] for i in np.random.default_rng(0).choice(len(rows), 300, replace=False)]
    todo_iv = collections.defaultdict(list)
    for r in todo:
        todo_iv[r["interview"]].append(r)
    bad_content, dur_err = [], 0
    for iid, rs in todo_iv.items():
        with wave.open(str(WORK / "audio16k" / f"{iid}.wav")) as w:
            src = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        def chk(r):
            x = decode(DATASET / r["file_name"])
            s0 = int(round(r["source_start"] * SR))
            y = src[s0:s0 + len(x)]
            same = len(x) > 0 and len(y) == len(x) and np.array_equal(x, y)
            return r["file_name"], same, abs(len(x) - int(round(r["duration"] * SR)))
        with ThreadPoolExecutor(8) as ex:
            for fn, same, de in ex.map(chk, rs):
                if not same:
                    bad_content.append(fn)
                dur_err = max(dur_err, de)
    if bad_content:
        fails.append(f"{len(bad_content)} Dateien nicht bitgleich mit dem Quellausschnitt, z.B. {bad_content[:3]}")
    if dur_err:
        fails.append(f"Laenge weicht um bis zu {dur_err} Samples von duration ab")
    print(f"Inhalt geprueft: {len(todo)} Dateien, nicht bitgleich {len(bad_content)}, "
          f"groesste Laengenabweichung {dur_err} Samples")

    # overlaps
    ov = 0
    for iid, rs in by_iv.items():
        rs = sorted(rs, key=lambda r: r["source_start"])
        ov += sum(1 for x, y in zip(rs, rs[1:]) if y["source_start"] < x["source_end"] - 1e-6)
    if ov:
        fails.append(f"{ov} ueberlappende Segmentpaare")
    print(f"Ueberlappende Paare: {ov}")

    # texts
    for name, rx in BAD_TEXT:
        hits = [r for r in rows if rx.search(r["text"])]
        print(f"Text '{name}': {len(hits)}" + (f"  z.B. {hits[0]['text'][:80]!r}" if hits else ""))
        if hits and name != "Ligatur-Ersatzzeichen":
            fails.append(f"{len(hits)} Texte mit {name}")
        elif len(hits) > 20:
            fails.append(f"{len(hits)} Texte mit moeglichen Ligatur-Ersatzzeichen")
    # replacement characters per document (glyphs.map of the PDFs with broken ligatures)
    glyph_chars = {}
    for f in (WORK / "turns").glob("*.json"):
        g = json.loads(f.read_text(encoding="utf-8")).get("glyphs") or {}
        if g.get("map"):
            glyph_chars[f.stem] = set(g["map"])
    left = [(r["interview"], w.strip(".,;:!?\"„“”()»«–—…'’‘´")) for r in rows if r["interview"] in glyph_chars
            for w in r["text"].split() if glyph_residue(w.strip(".,;:!?\"„“”()»«–—…'’‘´"), glyph_chars[r["interview"]])]
    print(f"Text 'Ersatzzeichen je Dokument' ({len(glyph_chars)} PDFs): {len(left)}"
          + (f"  z.B. {left[:3]}" if left else ""))
    if left:
        fails.append(f"{len(left)} Woerter mit unreparierter Ligatur, z.B. {left[:5]}")
    short = [r for r in rows if len(r["text"].split()) < 2]
    if short:
        fails.append(f"{len(short)} Texte mit weniger als 2 Woertern")

    # split
    parts_of = collections.defaultdict(set)
    grp_parts = collections.defaultdict(set)
    for r in rows:
        grp_parts[r["group"]].add(r["split"])
        if r["role"] != "interviewer":
            parts_of[r["person"]].add(r["split"])
    leak_p = {p: s for p, s in parts_of.items() if len(s) > 1}
    leak_g = {g: s for g, s in grp_parts.items() if len(s) > 1}
    iv_devtest = sum(1 for r in rows if r["role"] == "interviewer" and r["split"] != "train")
    if leak_p:
        fails.append(f"Personen in mehreren Teilen: {list(leak_p.items())[:5]}")
    if leak_g:
        fails.append(f"Gruppen in mehreren Teilen: {list(leak_g.items())[:5]}")
    if iv_devtest:
        fails.append(f"{iv_devtest} Interviewer-Segmente in Dev/Test")
    sp = collections.Counter(r["split"] for r in rows)
    hs = collections.Counter()
    for r in rows:
        hs[r["split"]] += r["duration"]
    print("Aufteilung: " + ", ".join(f"{k} {sp[k]} Segm. / {hs[k]/3600:.2f} h" for k in ("train", "dev", "test")))
    print(f"Personen in mehreren Teilen: {len(leak_p)}   Gruppen: {len(leak_g)}   Interviewer in Dev/Test: {iv_devtest}")
    print(f"\nGesamt {sum(r['duration'] for r in rows)/3600:.2f} h | Rollen "
          + str(dict(collections.Counter(r['role'] for r in rows)))
          + " | Qualitaet " + str(dict(collections.Counter(r['quality'] for r in rows))))
    print()
    if fails:
        print("NICHT BESTANDEN:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("bestanden.")


if __name__ == "__main__":
    main()
