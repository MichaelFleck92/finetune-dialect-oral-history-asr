#!/usr/bin/env python3
"""
Stufe 0b fuer das Projekt "Zwangsarbeiter": abfotografierte Transkriptseiten
lesen und dabei die Sprecher auseinanderhalten.

Wer spricht, steht in diesen Transkripten nicht als Kuerzel da, sondern in der
Schrift: die Fragen des Interviewers sind kursiv, die Antworten aufrecht.

  Text     macOS Vision (pipeline/bin/vision_ocr), zeilenweise mit Kaestchen
  Kursiv   Schraeglage je Zeile (line_slant): die Zeile wird schrittweise
           geschert; beim richtigen Winkel stehen die Senkrechten untereinander
           und die Spaltensumme bekommt scharfe Spitzen.

Gegenueber v1:
  * Jede Seite wird zweimal gelesen, mit und ohne Sprachkorrektur. Woerter, an
    denen sich beide Lesungen unterscheiden, gelten als unsicher und werden als
    [ocr:...] markiert - der Parser laesst sie nicht ins Training.
  * Zaehlerstaende der Mappen ("Chieskiu088", "(047)", "XXX084") werden als
    [#...] abgetrennt, "XXX" (Platzhalter der Transkribierenden) als [unv].
  * Die Kursiv-Pruefung ist nicht mehr der Wechsel der Turns (der ergibt sich
    zwangslaeufig aus dem Zusammenfassen), sondern eine Sichtpruefung von
    Zeilenausschnitten - siehe --crops.

  python3 pipeline/ocr_zwangsarbeiter.py [--only ID] [--crops N]
"""
import argparse, difflib, re, subprocess, sys, tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORK, load_specs, resolve

VISION = Path(__file__).resolve().parent / "bin" / "vision_ocr"
ITALIC_DEG = 5.0          # gemessen: aufrecht -6..+4 Grad, kursiv 9..17 (einmal 23)
OUT = WORK / "ocr"


def line_slant(crop, lo=-6, hi=26, step=1.0):
    if crop.size == 0 or crop.shape[0] < 12:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    g = cv2.GaussianBlur(g, (3, 3), 0)
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if bw.sum() / 255.0 < crop.shape[0] * 2:
        return None
    h, w = bw.shape
    pad = int(h * np.tan(np.deg2rad(hi))) + 4
    canvas = np.zeros((h, w + 2 * pad), np.uint8)
    canvas[:, pad:pad + w] = bw
    best, best_score = 0.0, -1.0
    for a in np.arange(lo, hi + 1e-9, step):
        t = np.tan(np.deg2rad(a))
        M = np.float32([[1, t, -t * h / 2], [0, 1, 0]])
        sh = cv2.warpAffine(canvas, M, (canvas.shape[1], h), flags=cv2.INTER_NEAREST)
        score = float(np.var(sh.sum(0, dtype=np.float64) / 255.0))
        if score > best_score:
            best_score, best = score, float(a)
    return best


def run_vision(path, correction=True):
    cmd = [str(VISION), str(path)] + ([] if correction else ["--no-correction"])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"vision_ocr scheiterte an {path}: {r.stderr[:200]}")
    rows = []
    for l in r.stdout.split("\n"):
        parts = l.split("\t")
        if len(parts) < 3 or not parts[2].strip():
            continue
        x0, y0, x1, y1 = (int(float(v)) for v in parts[0].split())
        rows.append({"box": (x0, y0, x1, y1), "conf": float(parts[1]), "text": parts[2].strip()})
    return rows


def orient(path, tmpdir):
    """Die Fotos sind quer aufgenommen; die Drehung mit dem meisten erkannten
    Text gewinnt."""
    im = Image.open(path)
    best = None
    for rot, tf in ((270, Image.ROTATE_270), (90, Image.ROTATE_90),
                    (0, None), (180, Image.ROTATE_180)):
        cand = im if tf is None else im.transpose(tf)
        tmp = Path(tmpdir) / f"o_{path.stem}_{rot}.png"
        cand.save(tmp)
        rows = run_vision(tmp)
        n = sum(len(r["text"]) for r in rows)
        if best is None or n > best[0]:
            best = (n, rot, cand, tmp, rows)
        if n > 1500:
            break
    return best


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def mark_uncertain(text_c, text_n):
    """Woerter der korrigierten Lesung, die in der unkorrigierten anders lauten,
    als [ocr:...] markieren (Satzzeichen-Unterschiede zaehlen nicht)."""
    a, b = text_c.split(), text_n.split()
    key = lambda w: re.sub(r"[^\wÄÖÜäöüß]", "", w).lower()
    sm = difflib.SequenceMatcher(a=[key(w) for w in a], b=[key(w) for w in b], autojunk=False)
    out, n = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        for w in a[i1:i2]:
            if tag == "equal" or not key(w):
                out.append(w)
            else:
                out.append(f"[ocr:{w}]"); n += 1
    return " ".join(out), n


HEAD = re.compile(r"^\s*Interview mit .+geführt von", re.I)
# Archivkopf, Stempel und Seitenfuss (kein gesprochener Text)
BOILER = re.compile(
    r"^\s*(COM\s+\d|AL\s+\d{4}-|OHA-WISOG|GESPERRT|Interviewer\s*:|Interviewte/?r?\s*:|"
    r"\d+\s*Band|Seite\s+\d+|\d{1,3}\s*$)", re.I)
# Ein Transkript kennzeichnet Fragen zusaetzlich mit "I.:" (OCR: "1:" oder "l:")
QPREFIX = re.compile(r"^\s*(?:I\.?|1|l)\s*[:.]\s*")
# Zaehlerstaende der Mappen: drei Ziffern, direkt an ein Wort oder ein
# Anfuehrungszeichen geklebt ("Chieskiu088", "Rohrzug200", „Lebring“207)
# oder allein geklammert ("(047)", "(XXX007)"). Kein gesprochener Text.
GLUED = re.compile(r"(?<=[A-Za-zÄÖÜäöüß\"“”„)])(\d{3})(?![\d])")
BRACKETED = re.compile(r"\(\s*(XXX)?\s*(\d{3})\s*\)")
XXX_NUM = re.compile(r"\bXXX(\d{3})\b")


def clean_counters(t):
    n0 = len(re.findall(r"\d{3}|XXX", t))
    t = BRACKETED.sub(lambda m: (" [unv]" if m.group(1) else "") + f" [#{m.group(2)}] ", t)
    t = XXX_NUM.sub(lambda m: f" [unv] [#{m.group(1)}] ", t)
    t = GLUED.sub(lambda m: f" [#{m.group(1)}] ", t)
    t = re.sub(r"\bXXX\b", "[unv]", t)
    n = n0 - len(re.findall(r"(?<!#)\d{3}(?!\])|XXX", t))
    return re.sub(r"\s{2,}", " ", t).strip(), n


def read_page(path, tmpdir):
    n, rot, img_pil, tmp, rows_c = orient(path, tmpdir)
    rows_n = run_vision(tmp, correction=False)
    img = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    out = []
    for r in rows_c:
        x0, y0, x1, y1 = r["box"]
        crop = img[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + 3]
        ang = line_slant(crop)
        match = max(rows_n, key=lambda q: box_iou(q["box"], r["box"]), default=None)
        if match and box_iou(match["box"], r["box"]) > 0.5:
            text, n_unsure = mark_uncertain(r["text"], match["text"])
        else:
            text, n_unsure = r["text"], 0
        out.append({"y": y0, "x": x0, "box": r["box"], "text": text, "raw": r["text"],
                    "conf": r["conf"], "angle": ang, "unsure": n_unsure})
    out.sort(key=lambda q: (q["y"], q["x"]))
    return out, rot, img


def assign_style(lines):
    """Kursiv ja/nein; nicht messbare Zeilen erben den Stil der Vorgaengerzeile."""
    prev = False
    for l in lines:
        if l["angle"] is None:
            l["italic"] = prev
        else:
            l["italic"] = l["angle"] >= ITALIC_DEG
        prev = l["italic"]


def dehyphenate(lines, vocab, justified):
    """Trennstriche am Zeilenende zusammenziehen - im Blocksatz die
    Regel, sonst nur wenn dabei ein bekanntes Wort entsteht; ein Bindestrich
    am Zeilenende kann dort auch einen Wortabbruch im Sprechen bedeuten."""
    out, i, joined = [], 0, 0
    while i < len(lines):
        cur = dict(lines[i])
        while (i + 1 < len(lines)
               and re.search(r"[A-Za-zÄÖÜäöüß]-$", cur["text"].strip())
               and lines[i + 1]["italic"] == cur["italic"]):
            nxt = lines[i + 1]["text"].strip()
            head = cur["text"].strip()[:-1]
            tail = nxt.split(" ", 1)
            merged = head.split()[-1] + tail[0]
            known = re.sub(r"[^\wÄÖÜäöüß]", "", merged).lower() in vocab
            if not known and not (justified and tail[0][:1].islower()):
                break
            cur["text"] = head + nxt
            joined += 1
            i += 1
        out.append(cur); i += 1
    return out, joined


def build(pages, vocab):
    alll = [r for pg in pages for r in pg]
    n_hy = sum(1 for r in alll if re.search(r"[A-Za-zÄÖÜäöüß]-$", r["text"].strip()))
    justified = bool(alll) and n_hy / len(alll) > 0.05
    head, turns, cur, n_join = [], [], None, 0
    for lines in pages:
        lines = [l for l in lines if not BOILER.match(l["raw"])]
        lines, j = dehyphenate(lines, vocab, justified)
        n_join += j
        for r in lines:
            if HEAD.match(r["text"]) and not turns:
                head.append(r["text"]); continue
            if cur is None or cur["italic"] != r["italic"]:
                if cur:
                    turns.append(cur)
                cur = {"italic": r["italic"], "lines": []}
            cur["lines"].append(r["text"])
    if cur:
        turns.append(cur)
    first_q = next((i for i, t in enumerate(turns) if t["italic"]), 0)
    if first_q:
        head += [" ".join(t["lines"]) for t in turns[:first_q]]
        turns = turns[first_q:]
    return head, turns, n_join, justified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--crops", type=int, default=0,
                    help="je Dokument N zufaellige Zeilenausschnitte fuer die Sichtpruefung ablegen")
    ap.add_argument("--force", action="store_true", help="auch vorhandene Ergebnisse neu lesen")
    a = ap.parse_args()
    if not VISION.exists():
        sys.exit(f"vision_ocr fehlt: swiftc -O pipeline/bin/vision_ocr.swift -o {VISION}")
    from sources import read_lines, build_vocab
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [s for s in load_specs() if s["project"] == "za" and (not a.only or a.only in s["id"])]
    # Wortschatz aus den sauberen docx-Transkripten (deterministisch, nicht
    # aus einem frueheren Lauf wie in v1)
    vocab = build_vocab(l for s in load_specs() if (s["transcript"] or "").endswith(".docx")
                        for l in read_lines(resolve(s["transcript"])))
    rng = np.random.default_rng(0)
    for s in specs:
        folder = resolve(s["ocr_images"])
        pngs = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".png")
        out_txt = OUT / f"{s['id']}.txt"
        if (out_txt.exists() and not a.force and not a.crops
                and out_txt.stat().st_mtime > max(p.stat().st_mtime for p in pngs)):
            continue                          # aktuell (Vision braucht ~3 min je Mappe);
                                              # nach Aenderungen an der Leselogik: --force
        pages, audit, crops = [], [], []
        with tempfile.TemporaryDirectory() as td:
            for i, p in enumerate(pngs, 1):
                lines, rot, img = read_page(p, td)
                assign_style(lines)
                pages.append(lines)
                for l in lines:
                    audit.append((p.name, l))
                    crops.append((p, rot, l))
                print(f"  [{i}/{len(pngs)}] {p.name} Drehung {rot}  {len(lines)} Zeilen, "
                      f"{sum(l['italic'] for l in lines)} kursiv, "
                      f"{sum(l['unsure'] for l in lines)} unsichere Woerter", flush=True)
        head, turns, n_join, justified = build(pages, vocab)
        body, n_cnt = [], 0
        for t in turns:
            tag = "I" if t["italic"] else "ZZ"
            txt = " ".join(t["lines"])
            if t["italic"]:
                txt = QPREFIX.sub("", txt)
            txt, n = clean_counters(txt)
            n_cnt += n
            body.append(f"{tag}: {txt}")
        (OUT / f"{s['id']}.txt").write_text("\n".join(head + [""] + body) + "\n", encoding="utf-8")
        (OUT / f"{s['id']}_zeilen.tsv").write_text(
            "seite\ty\twinkel\tkursiv\tkonfidenz\tunsicher\ttext\n" + "\n".join(
                f"{pg}\t{l['y']}\t{l['angle']}\t{'ja' if l['italic'] else 'nein'}\t"
                f"{l['conf']:.3f}\t{l['unsure']}\t{l['text']}" for pg, l in audit) + "\n",
            encoding="utf-8")
        if a.crops:
            cdir = OUT / f"{s['id']}_crops"; cdir.mkdir(exist_ok=True)
            rots = {0: None, 90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}
            for k in rng.choice(len(crops), size=min(a.crops, len(crops)), replace=False):
                pth, rot, l = crops[k]
                im = Image.open(pth)
                im = im if rots[rot] is None else im.transpose(rots[rot])
                img = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
                x0, y0, x1, y1 = l["box"]
                crop = img[max(0, y0 - 6):y1 + 6, max(0, x0 - 6):x1 + 6]
                tag = "K" if l["italic"] else "A"
                cv2.imwrite(str(cdir / f"{k:04d}_{tag}_{l['angle']}.png"), crop)
        w = sum(len(l.split()) for l in body)
        ni = sum(1 for t in turns if t["italic"])
        print(f"{s['id']}: {len(pngs)} Seiten, {len(turns)} Turns ({ni} kursiv), {w} Woerter, "
              f"{n_join} Trennungen zusammengezogen (Blocksatz: {justified}), "
              f"{n_cnt} Zaehler/XXX abgetrennt\n", flush=True)


if __name__ == "__main__":
    main()
