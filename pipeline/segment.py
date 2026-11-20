#!/usr/bin/env python3
"""
Stufe 4: Wortzeiten -> Trainingssegmente (Audio/Text-Paare).

Schnittregeln
  * nie ueber einen Sprecherwechsel, nie ueber [unv]/[gap] (dort liegt Sprache
    ohne Text), nie ueber nicht alignierte Woerter oder Pausen > HARD_GAP
  * das Wort direkt neben einem [unv]/[gap] wird weggelassen: seine Grenze zum
    Joker hin ist unsicher (v1 behielt diese Segmente, 724 Stueck); ebenso die
    Nachbarn eines gedehnten Wortes (> MAX_WORD_S) - dort stimmt das Alignment
    nicht, und die Hoerprobe fand an solchen Stellen falsche Schnitte
  * Schnittkanten werden geprueft (Hoerprobe 28.09.2026: 23 von 80 Segmenten
    beanstandet, meist an den Raendern):
    - Randwoerter fallen weg, solange ihr CTC-Wert unter dem 10. Perzentil
      ihres Sprechers liegt (und unter -3): Lage unsicher, oft ein kurzes Wort
      auf der falschen Seite einer Pause. Relativ zum Sprecher, weil eine feste
      Schwelle (-5) bei Dialekt 11 % und bei Hochsprache 5 % des Materials
      kostete - dieselbe Schieflage wie der absolute Filter in v1.
    - Beim Sprecherwechsel ohne Pause (Abstand < EDGE_GAP und die
      Greedy-Dekodierung hoert knapp ausserhalb Sprache) faellt das erste bzw.
      letzte Wort weg - dort ueberlappen die Stimmen. Bis zur naechsten echten
      Pause zu kuerzen kostete 4-7 % des Materials, wieder vor allem bei
      Dialekt, fuer wenige zusaetzlich behobene Faelle.
    Was dadurch zu kurz wird, entfaellt.
  * Pausen im Segment werden abgehoert: liefert die Greedy-Dekodierung dort
    Buchstaben, liegt Sprache ohne Text vor. Ab MIN_LETTERS (etwa ein Wort)
    wird getrennt; kuerzere Reste ("aeh", "mhm") bleiben - die Transkripte
    schreiben Fuellwoerter grundsaetzlich nicht, das Modell soll das lernen.
  * Laenge bis MAX_S (Whisper-Fenster 30 s), Schnitt fruehestens nach MIN_CUT_S, bevorzugt an
    Satzende + Pause
  * Raender adaptiv: nie ueber die Mitte der Pause zum Nachbarwort hinaus,
    also keine Ueberlappung (v1: 925 ueberlappende Paare). Rechts etwas mehr,
    weil CTC-Wortenden frueh liegen. Am Segmentende endet das Polster
    ausserdem vor der naechsten Sprache, die die Greedy-Dekodierung hoert: Die
    zweite Hoerprobe (28.09.2026) fand vier von acht Kantenfehlern dort, wo das
    Polster in ungeschriebene Sprache reichte ("weil | dann haben", "oder |
    halt"). v2.2 begrenzte beide Kanten; die blinde A/B-Nachpruefung ergab am
    Ende klar bessere Schnitte, am Anfang keine (1 besser, 1-2 schlechter) -
    CTC-Wortanfaenge liegen eher spaet, ein engerer Anfang kappt den Anlaut.
    Seit v2.3 wird deshalb nur das Ende begrenzt.

Guete je Segment (Grundlage fuer quality/keep, siehe keep_rule):
  score      mittlerer CTC-Wert der Woerter
  score_rel  score minus Median der Segmente derselben Person im Interview -
             misst "schlechter als sonst bei diesem Sprecher" statt "schwer
             verstaendlich". v1 filterte absolut und warf damit v.a. Dialekt
             weg (Ausbeute 34-43 % bei starkem Dialekt, ~90 % bei Hochsprache).
  edge_min   schlechtester Wert der ersten/letzten zwei Woerter (Schnittkanten)

  python3 pipeline/segment.py [--only ID]
"""
import argparse, collections, json, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, load_specs, write_json, sha1_file, code_hash

FPS = 50.0
# Laengere Segmente (16.11.2026, Variante F): erst ab 28 s wird geschnitten, und zwar fruehestens nach
# MIN_CUT_S; Pausen trennen erst ab 4 s. Vorher: Ziel 20 s, Schnitt ab 4 s, Pause 2 s -> Median 7,8 s,
# laengstes 20,4 s; jetzt Median 10,3 s, Mittel 12,0 s, 67 % der Zeit in Segmenten ab 15 s (vorher 35 %).
MIN_S, TARGET_S, MAX_S = 1.5, 28.0, 28.0
MIN_CUT_S = 16.0        # fruehester Schnittpunkt innerhalb eines langen Laufs
HARD_GAP = 4.0          # laengere Pause: immer trennen (Pausen mit Sprache trennt MIN_LETTERS ohnehin)
MAX_WORD_S = 1.5        # laengeres Einzelwort = gedehnt, Alignment dort unglaubwuerdig
MIN_LETTERS = 5         # so viele Buchstaben in einer Pause = ungeschriebenes Wort
PAD_L, PAD_R = 0.15, 0.30
EDGE_Q = 0.10           # Randwort unter diesem Perzentil seines Sprechers ...
EDGE_SC_MAX = -3.0      # ... und unter diesem Wert: Lage unsicher -> weglassen
EDGE_GAP = 0.15         # Sprecherwechsel mit kuerzerem Abstand ...
EDGE_LETTERS = 3        # ... und so vielen Buchstaben knapp ausserhalb = Ueberlappung
SPEECH_MARGIN = 0.04    # Abstand des Schnitts zur naechsten gehoerten Sprache
FILLERS = {"äh", "ähm", "öhm", "ehm", "ahm", "hm", "hmm", "mhm", "mh", "äähm", "ähh", "hmhm", "aha"}
FILLERS.discard("aha")  # 'aha' ist ein Wort


def speech_letters(emis, inv, f0, f1, blank=0):
    """Buchstaben der Greedy-Dekodierung im Frame-Bereich [f0, f1)."""
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
    """Zeit des ersten Buchstabens fremder Sprache nach dem Wortende t_end (bis
    limit), sonst None. Erst das Wortende-Zeichen oder >= 3 leere Frames
    abwarten - das Wort selbst klingt in der Dekodierung oft noch nach."""
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
    """Zieltext fuers Training: wie im Transkript, aber einheitlich ueber alle
    Projekte - Fuellwoerter (nur in Rechnitz geschrieben) und Zoegerungszeichen
    (… / ... / freistehende Gedankenstriche) entfallen."""
    out = []
    # Satzzeichen-Token haengen im Alignment am Vorgaengerwort ("geschluckt. -"):
    # erst an Leerzeichen trennen, sonst bleiben freistehende Striche stehen
    tokens = [x for t in tokens for x in t.split()]
    # "Quartier…mhm" ist ein Token: erst an Zoegerungszeichen trennen
    tokens = [x for t in tokens for x in re.split(r"(?<=…)|(?<=\.\.\.)", t) if x]
    for t in tokens:
        core = t.strip(".,;:!?…\"„“”»«()–-").lower()
        if core in FILLERS:
            # Satzzeichen des Fuellworts an das vorige Wort haengen
            tail = re.sub(r"^[^.,;:!?]*", "", t.rstrip("…"))
            if out and tail and out[-1][-1:] not in ".,;:!?":
                out[-1] += tail
            continue
        t = t.replace("…", " ").replace("...", " ")
        if re.fullmatch(r"[-–—]+", t.strip(" .,;:!?")):
            # freistehender Strich, auch mit Satzzeichen ("-,", "--,"; 01.10.2026):
            # das Satzzeichen geht wie beim Fuellwort an das vorige Wort
            tail = re.sub(r"^[^.,;:!?]*", "", t.strip())
            if out and tail and out[-1][-1:] not in ".,;:!?":
                out[-1] += tail
            continue
        out.append(t.strip())
    s = " ".join(x for x in out if x)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;:])(?=[,.;:!?])", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" ,;:")
    # Strich am letzten Wort = abgebrochener Satz ("und-", "haben Sie-?"); er hat
    # keinen Laut (Hoerprobe 01.10.2026: 296 Zieltexte endeten so). Striche im
    # Inneren bleiben - oft Ergaenzungsstrich ("Nacht- und Nebelaktion").
    while True:                                  # bis nichts mehr wegfaellt ("vor- --,")
        t = re.sub(r"[-–—]+(?=[.?!…\"“”»«')]*$)", "", s).rstrip(" ,;:")
        if t == s:
            return s
        s = t


def truncation(tokens):
    """Wortabbruch 'kein-' - aber nicht Ergaenzungsstrich ('Nacht- und Nebel')."""
    for a, b in zip(tokens, tokens[1:] + [""]):
        if re.search(r"\w-$", a) and b.lower().strip(",.") not in ("und", "oder", "bzw", "bis", "sowie"):
            return True
    return False


def runs(W, marks_hard, emis, inv, stats):
    """Woerter -> Laeufe, die nicht getrennt werden duerfen."""
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
            # gedehnt: wie beim Joker fallen auch die Nachbarn weg
            if cur:
                cur.pop(); stats["wort_neben_gedehnt"] += 1
            close(); stats["wort_gedehnt"] += 1
            drop_next = True
            continue
        if drop_next:
            drop_next = False
            stats["wort_neben_gedehnt"] += 1
            continue
        if w["i"] in marks_hard:            # Joker vor diesem Wort
            # Wort davor (schon im Lauf) und dieses Wort fallen weg
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
    """Liegt zwischen w und dem Nachbarwort nb ausserhalb keine Pause?"""
    gap = (w["s"] - nb["e"]) if side < 0 else (nb["s"] - w["e"])
    if gap >= EDGE_GAP:
        return False
    # knapp ausserhalb (0,06 s Abstand fuer die unscharfen CTC-Grenzen, 0,4 s Fenster)
    t0, t1 = (w["s"] - 0.46, w["s"] - 0.06) if side < 0 else (w["e"] + 0.06, w["e"] + 0.46)
    return len(speech_letters(emis, inv, max(0, int(t0 * FPS)), int(t1 * FPS))) >= EDGE_LETTERS


def edge_thresholds(W):
    """CTC-Schwelle fuer Randwoerter je Sprecher (siehe EDGE_Q)."""
    by = collections.defaultdict(list)
    for w in W:
        if "sc" in w:
            by[w["spk"]].append(w["sc"])
    allsc = [x for v in by.values() for x in v]
    return {k: min(EDGE_SC_MAX, float(np.quantile(v if len(v) >= 200 else allsc, EDGE_Q)))
            for k, v in by.items()}


def trim_edges(seg, aligned, pos, thr, emis, inv, stats):
    """Unsichere Randwoerter weglassen (siehe Modulbeschreibung)."""
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
            # harte Grenze erreicht: am besten Punkt schneiden, notfalls hier
            # (ohne "or len(cur)" landeten die Woerter in zwei Segmenten)
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
    # Raender: nie ueber die halbe Pause zum Nachbarwort (auch ueber Segmente hinweg)
    out = []
    for seg in segs:
        k0, k1 = pos[seg[0]["i"]], pos[seg[-1]["i"]]
        prev_e = aligned[k0 - 1]["e"] if k0 > 0 else 0.0
        next_s = aligned[k1 + 1]["s"] if k1 + 1 < len(aligned) else A["audio_s"]
        s = max(0.0, seg[0]["s"] - min(PAD_L, max(0.0, (seg[0]["s"] - prev_e) / 2)))
        e = min(A["audio_s"], seg[-1]["e"] + min(PAD_R, max(0.0, (next_s - seg[-1]["e"]) / 2)))
        # Polster am Ende nie in Sprache hinein, die die Dekodierung danach hoert
        # (nur kuerzen, nie verlaengern; am Anfang nicht, siehe Modulbeschreibung)
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
    # Relativwert je Person
    by = collections.defaultdict(list)
    for o in out:
        by[o["person"]].append(o["score"])
    med = {k: float(np.median(v)) for k, v in by.items()}
    for o in out:
        o["score_rel"] = round(o["score"] - med[o["person"]], 4)
    return A, out, stats


def keep_rule(o):
    """-> (behalten?, Stufe, Grund).

    Festgelegt am 27.09.2026 mit `eval_segments.py --calibrate` (Whisper-small,
    dauerproportionale Stichprobe; "schlecht" = CER mehr als 0,3 ueber dem
    Median der uebrigen Segmente desselben Interviews, also Fehlpaar statt
    Dialekt). Die Kennzahlen - seit 01.10.2026 auf den Bestand hochgerechnet -
    stehen in work/eval/report.md ("Kalibrierung der Verwerfregel"): Die
    gewaehlte Regel ist genauer als v1 (score < -3.0, trifft v.a. Dialekt) und
    verliert weniger gutes Material; edge_min < -8 traf kaum schlechte Segmente
    und ist entfernt. Erfasst wird nur ein kleiner Teil des schlechten
    Materials.
    Achtung: Schwellen und Kennzahlen stammen aus derselben Stichprobe
    (in-sample) - fuer eine belastbare Schaetzung braucht es eine neue.
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
            if spec.get("exclude"):             # ganzes Interview ausgeschlossen (Grund in der Spec)
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
