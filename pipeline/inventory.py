#!/usr/bin/env python3
"""
Structural overview per transcript - the basis from which the spec is
written (by pipeline/llm_specs.py or by hand).

Shows only what the structure needs: header (contact data masked),
candidates for speaker tags with frequencies, tags in the middle of lines,
all bracketed annotations, time marks, placeholders and the last lines.
The conversation text itself is not printed.

  python3 pipeline/inventory.py rs_muster_a [re_beispiel ...]   # individual ones
  python3 pipeline/inventory.py --all > work/inventory.txt
"""
import argparse, collections, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_specs, resolve, ROOT
from sources import read_lines

MASK_KEYS = re.compile(r"^\s*(Kontakt\w*|Adresse|Anschrift|Telefon\w*|Tel\.?|Handy|Mobil\w*|"
                       r"E-?Mail|Wohnort|Straße|Strasse|PLZ|Geburtsdatum|Geboren am)\s*[:\-]?", re.I)
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d /\-]{6,}\d)(?!\d)")
EMAIL = re.compile(r"\S+@\S+\.\w+")
TS = re.compile(r"#[^#\n]{0,24}#|#\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,-]\d+)?")
TS_OK = re.compile(r"^#\s*\d{1,2}:\d{2}:\d{2}(?:[.,-]\d{1,2})?\s*#$")
TAG_LINE = re.compile(r"^\s*([^\s:\[\]][^:\[\]]{0,38}?)\s*:\s")
TAG_BRACKET = re.compile(r"^\s*\[\s*([^\]]{1,40}?)\s*\]")
INLINE = re.compile(r"(?<=[.!?…\"“”)\]]\s)([A-ZÄÖÜ][\wÄÖÜäöüß.\- ]{0,20}?):\s")
NOTE = re.compile(r"\[[^\]\n]{1,120}\]|\([^)\n]{1,80}\)|\{[^}\n]{1,80}\}")


def mask(line):
    if MASK_KEYS.match(line):
        k = MASK_KEYS.match(line).group(0)
        return k + " <maskiert>"
    line = EMAIL.sub("<email>", line)
    line = PHONE.sub("<nummer>", line)
    return line


def inventory(spec, max_notes=80):
    p = resolve(spec["transcript"]) if not spec["transcript"].startswith("work/") \
        else ROOT / spec["transcript"]
    lines = read_lines(p)
    nonempty = [(i, l.strip()) for i, l in enumerate(lines) if l.strip()]
    words = sum(len(l.split()) for _, l in nonempty)
    tags = collections.Counter(); tag_ex = {}
    btags = collections.Counter(); btag_ex = {}
    for i, l in nonempty:
        m = TAG_BRACKET.match(l)
        if m:
            btags[m.group(1)] += 1; btag_ex.setdefault(m.group(1), (i, l[m.end():m.end() + 50]))
            continue
        m = TAG_LINE.match(l)
        if m and len(m.group(1).split()) <= 4:
            tags[m.group(1)] += 1; tag_ex.setdefault(m.group(1), (i, l[m.end():m.end() + 50]))
    # first line starting with a frequent tag = probably the start of the conversation
    frequent = {k for k, n in tags.items() if n >= 5} | {k for k, n in btags.items() if n >= 5}
    first_turn = next((i for i, l in nonempty
                       if (TAG_LINE.match(l) and TAG_LINE.match(l).group(1) in frequent)
                       or (TAG_BRACKET.match(l) and TAG_BRACKET.match(l).group(1) in frequent)), None)
    inline = collections.Counter(); inline_ex = collections.defaultdict(list)
    notes = collections.Counter()
    ts_all, ts_bad = 0, []
    odd = collections.Counter()
    for i, l in nonempty:
        for m in INLINE.finditer(l):
            inline[m.group(1)] += 1
            if len(inline_ex[m.group(1)]) < 2:
                inline_ex[m.group(1)].append(mask(l[max(0, m.start() - 40):m.end() + 30]))
        for m in NOTE.finditer(l):
            notes[m.group(0)] += 1
        for m in TS.finditer(l):
            ts_all += 1
            if not TS_OK.match(m.group(0).strip()):
                ts_bad.append((i, m.group(0)))
        for m in re.finditer(r"XXX+|\(\?\)|\?\)|\b\w+\d{3}\b|\bNN\b|\.\.\.|…|\b(?:äh|ähm|ahm|öhm|mhm|hm)\b", l, re.I):
            odd[m.group(0) if not re.match(r"\w+\d{3}", m.group(0)) else "<wort+ziffern>"] += 1
    out = []
    out.append(f"##### {spec['id']}  |  {spec['transcript']}")
    out.append(f"Zeilen {len(nonempty)}, Woerter {words}, Audio {len(spec['audio'])} Teil(e); "
               f"Zeitmarken {ts_all} (davon ungewoehnlich {len(ts_bad)}: {ts_bad[:6]})")
    stop = first_turn if first_turn is not None else (nonempty[min(25, len(nonempty) - 1)][0] + 1)
    out.append(f"--- Kopf (bis Zeile {stop}):")
    for i, l in nonempty:
        if i >= stop:
            break
        out.append(f"  {i:4}: {mask(l)[:170]}")
    if first_turn is not None:
        out.append(f"--- erste Gespraechszeile: {first_turn}: {nonempty[[j for j,_ in nonempty].index(first_turn)][1][:90]!r}")
    out.append("--- Kuerzel am Zeilenanfang 'X:' (Anzahl, erstes Vorkommen):")
    for k, n in tags.most_common(14):
        ex = "<maskiert>" if MASK_KEYS.match(k + ":") else mask(tag_ex[k][1])
        out.append(f"  {n:5}  {k!r:34} z.{tag_ex[k][0]}: {ex!r}")
    if btags:
        out.append("--- Kuerzel am Zeilenanfang '[X]':")
        for k, n in btags.most_common(14):
            out.append(f"  {n:5}  [{k}]  z.{btag_ex[k][0]}: {mask(btag_ex[k][1])!r}")
    if inline:
        out.append("--- 'X:' mitten in der Zeile (nach Satzende):")
        for k, n in inline.most_common(8):
            out.append(f"  {n:5}  {k!r}  z.B. {inline_ex[k]}")
    out.append(f"--- Anmerkungen in Klammern ({len(notes)} verschiedene, {sum(notes.values())} gesamt):")
    for k, n in notes.most_common(max_notes):
        out.append(f"  {n:4}  {mask(k)[:110]}")
    if odd:
        out.append(f"--- Auffaelliges: {dict(odd.most_common(12))}")
    out.append("--- letzte Zeilen:")
    for i, l in nonempty[-5:]:
        out.append(f"  {i:4}: {mask(l)[:150]}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--notes", type=int, default=80)
    a = ap.parse_args()
    specs = [s for s in load_specs() if a.all or s["id"] in a.ids
             or any(s["id"].startswith(x) for x in a.ids)]
    for s in specs:
        if not s["transcript"]:
            print(f"##### {s['id']}: kein Transkript in der Spec"); continue
        try:
            print(inventory(s, a.notes))
        except FileNotFoundError as e:
            print(f"##### {s['id']}: Datei fehlt ({e})")
        print()


if __name__ == "__main__":
    main()
