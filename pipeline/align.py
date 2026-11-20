#!/usr/bin/env python3
"""
Stufe 3b: Forced Alignment ohne handgesetzte Zeitmarken.

v1 benutzte die Zeitmarken der Transkripte als feste Stuetzpunkte. Die Marken
hinken aber ~1,9 s nach (gemessen an einem Interview mit Handmarken) - als harte Grenze
verschoben sie den Beginn jedes folgenden Turns um im Median 1,65 s. Deshalb:

  * Stuetzpunkte kommen nur aus dem Audio selbst: Greedy-Dekodierung der
    wav2vec2-Emissionen, eindeutige Wort-Trigramme, die im Transkript genauso
    vorkommen, in zeitlich monotoner Folge (laengste steigende Teilfolge).
    Die Blockgrenze liegt zwischen zwei Woertern, die der Erkenner exakt so
    gehoert hat.
  * In grossen Luecken wird lokal weitergesucht (auch Bigramme). Findet sich
    nichts, wird der Block nicht "in der Mitte geteilt" (v1), sondern bleibt
    unaligniert, wenn er fuer einen Durchgang zu gross ist.
  * Ein Joker-Token (<star>, wie beim MMS-Aligner) steht an [unv]/[gap] und am
    Anfang/Ende der Aufnahme: dort liegt Sprache ohne Text, die sonst die
    Nachbarwoerter verschleppt.
  * Text ohne Ton (ein Interview: Luecke im Tonmitschnitt) faellt als
    unmoegliche Sprechrate zwischen zwei Stuetzpunkten auf und wird nicht
    aligniert statt in wenige Sekunden gequetscht.
  * Zwei Woerter, die im Transkript nur durch ein Zoegerungszeichen verbunden
    sind ("Jahren…seit", 434 Mal, v.a. Rechnitz), werden getrennt aligniert;
    als ein Wort bekamen sie eine gedehnte Zeitspanne und verzerrten die
    Schnittkanten (Hoerprobe 28.09.2026).

Die handgesetzten Zeitmarken werden nur mitgeschrieben - fuer die
unabhaengige Genauigkeitsmessung (pipeline/eval_markers.py).

  python3 pipeline/align.py [--only ID] [--force] [--turn-stars]
"""
import argparse, bisect, json, re, sys, time
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
from num2words import num2words

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, load_specs, write_json, sha1_file, code_hash

FPS = 50.0
CELL_BUDGET = 1.2e9          # Frames x Token je Block (forced_align, CPU)
MAX_BLOCK_S = 900            # groessere Bloecke werden lokal weiter unterteilt
MIN_ANCHOR_GAP_S = 20.0      # Mindestabstand der Stuetzpunkte
MAX_WPS = 5.0                # hoechstens 300 Woerter/min zwischen zwei Stuetzpunkten
OUT = WORK / "aligned"

# ------------------------------------------------------------ Normalisierung

ABBR = {"z.b.": "zum beispiel", "usw.": "und so weiter", "bzw.": "beziehungsweise",
        "d.h.": "das heisst", "dr.": "doktor", "prof.": "professor", "hr.": "herr",
        "fr.": "frau", "st.": "sankt", "nr.": "nummer", "ca.": "circa", "km": "kilometer",
        "kg": "kilo", "u.a.": "unter anderem", "evtl.": "eventuell", "bzgl.": "bezüglich",
        "%": "prozent", "&": "und", "+": "plus", "§": "paragraph", "m²": "quadratmeter",
        "mag.": "magister", "mag.a": "magistra", "ing.": "ingenieur", "dipl.": "diplom"}
ALLOWED = set("abcdefghijklmnopqrstuvwxyzäöüíó'")


def spell_number(tok, ordinal=False, year=True):
    n = int(tok)
    if ordinal and n < 100:
        return num2words(n, lang="de", to="ordinal")
    if year and 1100 <= n <= 1999:              # Jahreszahl: neunzehnhundert...
        h, r = divmod(n, 100)
        return num2words(h, lang="de") + "hundert" + (num2words(r, lang="de") if r else "")
    return num2words(n, lang="de")


# Tausenderpunkt, Dezimalkomma und Prozent so, wie es gesprochen wird. Bis
# 01.10.2026 wurde jede Ziffernfolge einzeln ausgeschrieben ("20.000" ->
# zwanzignull, "95%" ohne prozent, "100%ig" -> einhundertig; Review). Die
# Lesarten sind an der Greedy-Dekodierung aller 45 betroffenen Token geprueft:
# "10.000" zehntausend, "100.000" hunderttausend (ohne "ein"), "80%" achtzig
# prozent, "100%ig" hundertprozentig, "2,7%" zwei komma sieben prozent,
# "5,5 Jahre" fuenfeinhalb, "8,8" (Flak) acht acht, "1,90" eins neunzig.
PERCENT = re.compile(r"(\d+)(?:,(\d+))?%(?:-?(ig\w*))?")
THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+(?!\d)")
DECIMAL = re.compile(r"(\d+),(\d+)")


def _amount(digits):
    """Menge, keine Jahreszahl; "einhundert"/"eintausend" am Anfang ohne "ein"."""
    s = spell_number(digits, year=False).replace(" ", "|")
    return re.sub(r"^ein(?=hundert|tausend)", "", s)


def _decimal(a, b, komma=False):
    if komma:                                   # "komma" nur bei Prozent gehoert
        return _amount(a) + "|komma|" + _amount(b)
    if b == "5":                                # 5,5 -> fuenfeinhalb, 1,5 -> eineinhalb
        return {"0": "", "1": "ein"}.get(a, _amount(a)) + "einhalb"
    return _amount(a) + "|" + _amount(b)


def _percent(m):
    num = _decimal(m.group(1), m.group(2), komma=True) if m.group(2) else _amount(m.group(1))
    return num + ("prozent" + m.group(3) if m.group(3) else "|prozent")


def norm_word(w):
    """Originalwort -> Schreibung im Modellvokabular ('' = nicht alignierbar).
    '|' trennt Woerter innerhalb eines Tokens ("zum|beispiel", "acht|acht")."""
    lw = w.lower().strip("„“”\"»«()…,;:!?")
    if lw in ABBR:
        return ABBR[lw].replace(" ", "|")
    if re.fullmatch(r"\d{1,2}\.", lw):
        lw = spell_number(lw[:-1], ordinal=True)
    lw = PERCENT.sub(_percent, lw)
    lw = THOUSANDS.sub(lambda m: _amount(m.group().replace(".", "")), lw)
    lw = DECIMAL.sub(lambda m: _decimal(m.group(1), m.group(2)), lw)
    lw = re.sub(r"\d+", lambda m: spell_number(m.group()), lw)
    # erst nach dem Ausschreiben: "dreißig" (30-39, 1930-1939) behielt sonst kein ss
    lw = lw.replace("ß", "ss").replace("’", "'").replace("‘", "'").replace("´", "'")
    lw = lw.replace("-", "")                     # Nazi-Deutschland -> ein Wort
    import unicodedata
    out = []
    for ch in lw:
        if ch in ALLOWED or ch == "|":
            out.append(ch)
        elif ch.isalpha():
            base = unicodedata.normalize("NFKD", ch)[0].lower()
            if base in ALLOWED:
                out.append(base)
    return re.sub(r"\|+", "|", "".join(out)).strip("|'")

# ------------------------------------------------------------ Woerter

def split_joined(tok):
    """'Jahren…seit' -> ['Jahren…', 'seit']: Zoegerungszeichen zwischen zwei Woertern."""
    return re.sub(r"(\w)(…|\.\.\.)(?=\w)", r"\1\2 ", tok).split()


def build_words(turns, turn_stars=False):
    """-> words [{orig, norm, turn, speaker}], stars {vor Wortindex}, marks, handmarken"""
    words, stars, marks, hand = [], set(), [], []
    stars.add(0)
    for t in turns:
        if turn_stars and words:
            stars.add(len(words))
        toks = t["text"].split()
        # Wortindex je Token fuer die Zeitmarken (die im Turn als Wortposition stehen)
        tok_word = []
        for tok in toks:
            m = re.fullmatch(r"\[(unv|gap|nv)\][,.;:!?…]*", tok)
            if m:
                kind = m.group(1)
                marks.append({"word": len(words), "kind": kind, "turn": t["idx"]})
                if kind in ("unv", "gap"):
                    stars.add(len(words))
                tok_word.append(None)
                continue
            last = None                           # Zeitmarken zaehlen Transkript-Token:
            for part in split_joined(tok):        # letztes Wort des Tokens
                nw = norm_word(part)
                if not nw:
                    if words:                     # Satzzeichen an das Vorgaengerwort haengen
                        words[-1]["orig"] += " " + part
                    continue
                last = len(words)
                words.append({"orig": part, "norm": nw, "turn": t["idx"], "speaker": t["speaker"]})
            tok_word.append(last)
        # Zeitmarken: 'word' zaehlt die Woerter des bereinigten Turntexts ohne Markierungen
        plain_idx = [k for k, tok in enumerate(toks) if not re.fullmatch(r"\[(unv|gap|nv)\][,.;:!?…]*", tok)]
        for ts in t.get("timestamps", []):
            if ts.get("t") is None:
                continue
            k = ts["word"]                        # Anzahl Woerter davor im Turn
            wi = None
            for j in range(min(k, len(plain_idx)) - 1, -1, -1):
                wi = tok_word[plain_idx[j]]
                if wi is not None:
                    break
            if wi is not None:
                hand.append({"after_word": wi, "t": ts["t"], "turn": t["idx"]})
    stars.add(len(words))
    return words, stars, marks, hand

# ------------------------------------------------------------ ASR-Stuetzpunkte

def greedy_words(emis, inv_vocab, blank=0):
    ids = emis.argmax(-1)
    chars, prev = [], -1
    for t, k in enumerate(ids.tolist()):
        if k != prev and k != blank:
            chars.append((inv_vocab.get(k, ""), t))
        prev = k
    words, cur, st, en = [], [], None, None
    for ch, t in chars:
        if ch == "|":
            if cur:
                words.append(("".join(cur), st, en))
            cur, st = [], None
        elif len(ch) == 1 and (ch.isalpha() or ch in "'"):
            if st is None:
                st = t
            cur.append(ch); en = t
    if cur:
        words.append(("".join(cur), st, en))
    return words


def _lis(cands):
    """laengste streng steigende Teilfolge nach (ref, frame)."""
    cands.sort()
    tails, idx, back = [], [], []
    for k, (ri, fr, _) in enumerate(cands):
        i = bisect.bisect_left(tails, fr)
        if i == len(tails):
            tails.append(fr); idx.append(k)
        else:
            tails[i] = fr; idx[i] = k
        back.append(idx[i - 1] if i else -1)
    if not tails:
        return []
    out, k = [], idx[len(tails) - 1]
    while k != -1:
        out.append(cands[k]); k = back[k]
    return out[::-1]


def find_anchors(ref, asr, r0, r1, a0, a1, n):
    """Eindeutige n-Gramme in ref[r0:r1] und asr[a0:a1] -> [(ref_idx, frame, asr_idx)]
    Stuetzpunkt = Grenze zwischen 1. und 2. Wort des n-Gramms."""
    def grams(seq, lo, hi):
        d = {}
        for i in range(lo, hi - n + 1):
            d.setdefault(tuple(seq[i:i + n]), []).append(i)
        return d
    rg = grams(ref, r0, r1)
    aw = [w[0] for w in asr]
    ag = grams(aw, a0, a1)
    cands = []
    for g, ri in rg.items():
        if len(ri) != 1:
            continue
        aj = ag.get(g)
        if not aj or len(aj) != 1:
            continue
        if sum(len(x) for x in g) < 9:            # zu kurz: "und da ist" ist wertlos
            continue
        i, j = ri[0], aj[0]
        fr = (asr[j][2] + asr[j + 1][1]) // 2     # zwischen Wort 1 und 2
        cands.append((i + 1, fr, j + 1))
    return _lis(cands)


def anchor_blocks(words, asr, T, stats):
    ref = [w["norm"] for w in words]
    pts = [(0, 0, 0)] + find_anchors(ref, asr, 0, len(ref), 0, len(asr), 3) + [(len(ref), T, len(asr))]
    # rekursiv verfeinern: grosse Luecken lokal nach Tri-/Bigrammen durchsuchen
    for n in (3, 2):
        out = [pts[0]]
        for (r0, f0, a0), (r1, f1, a1) in zip(pts, pts[1:]):
            if (f1 - f0) / FPS > MAX_BLOCK_S / 3 and r1 - r0 > 30:
                inner = find_anchors(ref, asr, r0, r1, a0, a1, n)
                inner = [p for p in inner if r0 < p[0] < r1 and f0 < p[1] < f1]
                out.extend(inner)
            out.append((r1, f1, a1))
        pts = out
    # ausduennen und Plausibilitaet: Sprechrate zwischen zwei Stuetzpunkten
    kept = [pts[0]]
    for p in pts[1:-1]:
        if (p[1] - kept[-1][1]) / FPS >= MIN_ANCHOR_GAP_S:
            kept.append(p)
    kept.append(pts[-1])
    # einzelne Ausreisser (Trigramm an falscher Stelle) entfernen: ein Punkt,
    # der zu beiden Nachbarn eine unmoegliche Rate ergibt, faellt heraus
    changed = True
    while changed and len(kept) > 2:
        changed = False
        for k in range(1, len(kept) - 1):
            a, b, c = kept[k - 1], kept[k], kept[k + 1]
            rate_ab = (b[0] - a[0]) / max(0.2, (b[1] - a[1]) / FPS)
            rate_bc = (c[0] - b[0]) / max(0.2, (c[1] - b[1]) / FPS)
            rate_ac = (c[0] - a[0]) / max(0.2, (c[1] - a[1]) / FPS)
            if (rate_ab > MAX_WPS or rate_bc > MAX_WPS) and rate_ac <= MAX_WPS:
                del kept[k]; stats["stuetzpunkt_verworfen"] += 1; changed = True
                break
    blocks = []
    for (r0, f0, _), (r1, f1, _) in zip(kept, kept[1:]):
        if r1 <= r0:
            continue
        rate = (r1 - r0) / max(0.2, (f1 - f0) / FPS)
        blocks.append({"w0": r0, "w1": r1, "f0": int(f0), "f1": int(f1), "rate": round(rate, 2),
                       "ok": rate <= MAX_WPS})
    stats["stuetzpunkte"] += len(kept) - 2
    return blocks

# ------------------------------------------------------------ Alignment

def align_block(emis_star, words, w0, w1, f0, f1, stars, vocab, star_id):
    """Woerter [w0, w1) auf Frames [f0, f1) -> [(wi, s_frame, e_frame_exkl, score)] oder None.
    stars: Wortindizes, vor denen ein Joker steht (0 = Aufnahmebeginn,
    len(words) = Aufnahmeende)."""
    toks, owner = [], []
    if w0 in stars:
        toks.append(star_id); owner.append(-2)
    for k, wi in enumerate(range(w0, w1)):
        if k:
            if wi in stars:
                toks.append(star_id); owner.append(-2)
            else:
                toks.append(vocab["|"]); owner.append(-1)
        for ch in words[wi]["norm"]:
            if ch == "|":
                toks.append(vocab["|"]); owner.append(-1)
            else:
                toks.append(vocab.get(ch, vocab["<unk>"])); owner.append(wi)
    if w1 in stars:
        toks.append(star_id); owner.append(-2)
    T = f1 - f0
    repeats = sum(1 for x, y in zip(toks, toks[1:]) if x == y)
    if T < len(toks) + repeats + 1:
        return None
    lp = emis_star[f0:f1].unsqueeze(0).contiguous()
    tg = torch.tensor([toks], dtype=torch.int32)
    try:
        al, sc = AF.forced_align(lp, tg, blank=0)
    except Exception:
        return None
    spans = AF.merge_tokens(al[0], sc[0].float(), blank=0)
    if len(spans) != len(toks):
        return None
    res = {}
    for sp, ow in zip(spans, owner):
        if ow < 0:
            continue
        d = res.setdefault(ow, [sp.start, sp.end, []])
        d[0] = min(d[0], sp.start); d[1] = max(d[1], sp.end)
        d[2].append(float(sp.score))
    return [(wi, f0 + s, f0 + e, float(np.mean(scs))) for wi, (s, e, scs) in sorted(res.items())]


def run(spec, turn_stars=False):
    iid = spec["id"]
    tj = json.loads((WORK / "turns" / f"{iid}.json").read_text())
    em_meta = json.loads((WORK / "emissions" / f"{iid}.json").read_text())
    emis = np.load(WORK / "emissions" / f"{iid}.npy").astype(np.float32)
    T = emis.shape[0]
    vocab = em_meta["vocab"]
    inv = {v: k for k, v in vocab.items()}
    star_id = emis.shape[1]
    emis_star = torch.from_numpy(np.concatenate([emis, np.zeros((T, 1), np.float32)], 1))
    words, stars, marks, hand = build_words(tj["turns"], turn_stars)
    stats = __import__("collections").Counter()
    asr = greedy_words(torch.from_numpy(emis), inv)
    blocks = anchor_blocks(words, asr, T, stats)
    out = {}
    queue = [(b, 0) for b in blocks]
    done_blocks = []
    while queue:
        b, depth = queue.pop(0)
        n_tok = sum(len(words[i]["norm"]) + 1 for i in range(b["w0"], b["w1"]))
        cells = (b["f1"] - b["f0"]) * n_tok
        if not b["ok"]:
            # Irgendwo in diesem Block steht Text ohne Ton. Mit lokalen
            # Bigramm-Stuetzpunkten den alignierbaren Teil herauslösen; was
            # danach noch zu schnell ist, bleibt unaligniert.
            ref = [w["norm"] for w in words]
            fr = [x[1] for x in asr]
            a0, a1 = bisect.bisect_left(fr, b["f0"]), bisect.bisect_right(fr, b["f1"])
            inner = [p for p in find_anchors(ref, asr, b["w0"], b["w1"], a0, a1, 2)
                     if b["w0"] < p[0] < b["w1"] and b["f0"] < p[1] < b["f1"]]
            if inner and depth < 4 and b.get("_refined") is None:
                pts = [(b["w0"], b["f0"])] + [(p[0], p[1]) for p in inner] + [(b["w1"], b["f1"])]
                for (r0, f0), (r1, f1) in zip(pts, pts[1:]):
                    if r1 > r0 and f1 > f0:
                        rate = (r1 - r0) / max(0.2, (f1 - f0) / FPS)
                        queue.append(({"w0": r0, "w1": r1, "f0": int(f0), "f1": int(f1), "rate": round(rate, 2),
                                       "ok": rate <= MAX_WPS, "_refined": True}, depth + 1))
                continue
            b["status"] = "text_ohne_ton"; stats["woerter_text_ohne_ton"] += b["w1"] - b["w0"]
            done_blocks.append(b); continue
        if cells > CELL_BUDGET:
            # lokal nach Bigrammen mit engerem Abstand suchen
            ref = [w["norm"] for w in words]
            a0 = bisect.bisect_left([a[1] for a in asr], b["f0"])
            a1 = bisect.bisect_right([a[1] for a in asr], b["f1"])
            inner = [p for p in find_anchors(ref, asr, b["w0"], b["w1"], a0, a1, 2)
                     if b["w0"] < p[0] < b["w1"] and b["f0"] < p[1] < b["f1"]]
            if inner and depth < 4:
                pts = [(b["w0"], b["f0"])] + [(p[0], p[1]) for p in inner[::max(1, len(inner) // 8)]] \
                      + [(b["w1"], b["f1"])]
                for (r0, f0), (r1, f1) in zip(pts, pts[1:]):
                    if r1 > r0 and f1 > f0:
                        rate = (r1 - r0) / max(0.2, (f1 - f0) / FPS)
                        queue.append(({"w0": r0, "w1": r1, "f0": int(f0), "f1": int(f1),
                                       "rate": round(rate, 2), "ok": rate <= MAX_WPS}, depth + 1))
                continue
            b["status"] = "zu_gross_ohne_stuetzpunkt"; stats["woerter_zu_gross"] += b["w1"] - b["w0"]
            done_blocks.append(b); continue
        r = align_block(emis_star, words, b["w0"], b["w1"], b["f0"], b["f1"], stars, vocab, star_id)
        if r is None:
            b["status"] = "alignment_fehlgeschlagen"; stats["woerter_fehlgeschlagen"] += b["w1"] - b["w0"]
        else:
            b["status"] = "ok"
            for wi, s, e, sc in r:
                out[wi] = (s, e, sc)
        done_blocks.append(b)
    wl = []
    for wi, w in enumerate(words):
        rec = {"i": wi, "w": w["orig"], "norm": w["norm"], "turn": w["turn"], "spk": w["speaker"]}
        if wi in out:
            s, e, sc = out[wi]
            rec.update({"s": round(s / FPS, 3), "e": round(e / FPS, 3), "sc": round(sc, 4)})
        wl.append(rec)
    for b in done_blocks:
        b.pop("_refined", None)
    res = {"id": iid, "audio_s": em_meta["audio_s"], "n_words": len(words), "n_aligned": len(out),
           "turn_stars": turn_stars, "stars": sorted(stars), "marks": marks, "hand_markers": hand,
           "hand_markers_valid": spec.get("hand_markers_valid", True),
           "blocks": sorted(done_blocks, key=lambda b: b["w0"]), "stats": dict(stats),
           "words": wl, "turns_sha1": sha1_file(WORK / "turns" / f"{iid}.json"),
           "emissions_sha1": em_meta["audio_sha1"], "code": code_hash("align.py")}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--turn-stars", action="store_true", help="Joker auch an jedem Sprecherwechsel")
    ap.add_argument("--out", default=None, help="anderes Ausgabeverzeichnis (Varianten)")
    a = ap.parse_args()
    out_dir = Path(a.out) if a.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [s for s in load_specs() if not a.only or any(o in s["id"] for o in a.only.split(","))]
    for s in specs:
        iid = s["id"]
        if not (WORK / "emissions" / f"{iid}.npy").exists():
            print(f"-- {iid}: Emissionen fehlen noch"); continue
        tgt = out_dir / f"{iid}.json"
        if tgt.exists() and not a.force:
            old = json.loads(tgt.read_text())
            # emissions_sha1 = Hash des 16-kHz-Audios, aus dem die Emissionen stammen
            em_audio = json.loads((WORK / "emissions" / f"{iid}.json").read_text())["audio_sha1"]
            if old.get("turns_sha1") == sha1_file(WORK / "turns" / f"{iid}.json") \
                    and old.get("emissions_sha1") == em_audio \
                    and old.get("code") == code_hash("align.py"):
                continue                              # aktuell
        t0 = time.time()
        r = run(s, a.turn_stars)
        write_json(tgt, r)
        st = r["stats"]
        print(f"{iid:26} {r['n_aligned']:6}/{r['n_words']:6} W ({100*r['n_aligned']/max(1,r['n_words']):5.1f}%) "
              f"Stuetzp. {st.get('stuetzpunkte',0):4}  Bloecke {len(r['blocks']):4}  "
              + " ".join(f"{k}={v}" for k, v in st.items() if k.startswith("woerter") or k.endswith("verworfen"))
              + f"  {time.time()-t0:5.1f}s", flush=True)


if __name__ == "__main__":
    main()
