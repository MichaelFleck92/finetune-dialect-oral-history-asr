#!/usr/bin/env python3
"""
Alle Kennzahlen fuer README und Kapitel - aus den Daten berechnet, nicht von
Hand abgeschrieben (v1: viele Zahlen der Doku liessen sich nicht nachrechnen).

  python3 pipeline/report.py > work/eval/report.md

Ausnahmen, die nicht aus den Daten kommen, stehen in der Doku als solche:
Angaben zu v1 (aus der Begutachtung des Stands vom 23.09.2026) und die
Sichtpruefungen von Hand (Schriftlage). Die blinde Hoerprobe der groben
Markenabweichungen geht als Bewertungsdatei ein (work/eval/marker_hoerprobe_bewertung.csv).
"""
import collections, csv, json, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORK, DATASET, load_specs, PROJECTS


# Interview mit eigenem Markenformat (Auswertung getrennt), aus config/report.json
FORMAT_B_ID = (json.loads((ROOT / "config" / "report.json").read_text(encoding="utf-8")).get("format_b_id")
               if (ROOT / "config" / "report.json").exists() else None)


def h(x):
    return f"{x/3600:.2f}"


def pct(x):
    return f"{100*x:.0f} %"


def main():
    specs = {s["id"]: s for s in load_specs()}
    proj_of = {i: s["project"] for i, s in specs.items()}
    P = list(PROJECTS)
    print(f"# Kennzahlen (erzeugt mit pipeline/report.py)\n")
    # ---------------- Bestand
    audio = {i: json.loads((WORK / "audio16k" / f"{i}.json").read_text())["duration_s"] for i in specs}
    turns = {i: json.loads((WORK / "turns" / f"{i}.json").read_text()) for i in specs}
    words = {i: sum(t["n_words"] for t in turns[i]["turns"]) for i in specs}
    print("## Bestand\n\n| Projekt | Interviews | Audio (h) | Wörter im Transkript | Interviewdatum |\n|---|---|---|---|---|")
    for p in P:
        ids = [i for i in specs if proj_of[i] == p]
        dates = sorted(specs[i]["date"] for i in ids if specs[i].get("date"))
        nod = sum(1 for i in ids if not specs[i].get("date"))
        span = f"{dates[0][:4]}–{dates[-1][:4]}" if dates else "-"
        print(f"| {PROJECTS[p]} | {len(ids)} | {h(sum(audio[i] for i in ids))} | {sum(words[i] for i in ids):,} | "
              f"{span}{f' ({nod} ohne Datum)' if nod else ''} |".replace(",", "."))
    print(f"| **gesamt** | {len(specs)} | {h(sum(audio.values()))} | {sum(words.values()):,} | |\n".replace(",", "."))
    fmt = collections.Counter(Path(s["transcript"]).suffix.lower().lstrip(".") for s in specs.values())
    print("Transkriptformate: " + ", ".join(f"{k} {v}" for k, v in fmt.most_common())
          + " (txt = OCR der abfotografierten Seiten)")
    print("Interviews ohne Datum: " + ", ".join(sorted(i for i, s in specs.items() if not s.get("date"))) + "\n")
    nspk = collections.Counter(len({t['speaker'] for t in turns[i]['turns']}) for i in specs)
    print("Sprecher je Interview (laut Spec, ohne Parser-Artefakte): "
          + ", ".join(f"{k}: {v}" for k, v in sorted(nspk.items())) + "\n")
    rw = collections.Counter()
    for i in specs:
        for t in turns[i]["turns"]:
            rw[t["role"]] += t["n_words"]
    tw = sum(rw.values())
    print("Wortanteile nach Rolle: " + ", ".join(f"{k} {100*v/tw:.1f} %" for k, v in rw.most_common()) + "\n")
    # ---------------- Einwilligung
    cons = collections.Counter()
    for s in specs.values():
        v = s["consent"]["anonymise_requested"]
        cons["ja" if v else "nein" if v is False else "keine Angabe"] += 1
    print(f"Anonymisierungswunsch: {dict(cons)}")
    for s in sorted(specs.values(), key=lambda s: s["interviewee"]):
        if s["consent"]["anonymise_requested"]:
            print(f"- {s['interviewee']}: {s['consent']['evidence']}")
    # ---------------- Parsing
    st = collections.Counter(); gly_docs, gly_words = 0, 0
    for i in specs:
        for k, v in turns[i]["stats"].items():
            st[k] += v
        g = turns[i].get("glyphs") or {}
        if g.get("fixes"):
            gly_docs += 1; gly_words += len(g["fixes"])
    print("\n## Aufbereitung\n")
    print(f"Ligaturen repariert: {gly_words} verschiedene Wörter in {gly_docs} PDFs\n")
    print("Anmerkungen nach Klasse: " + ", ".join(f"{k[6:]} {v}" for k, v in sorted(st.items()) if k.startswith("notiz:")) + "\n")
    for k in ("zeile_verworfen", "zeile_als_luecke", "einwurf_abgetrennt", "kuerzel_im_satz_getrennt",
              "klammer_offen_als_luecke", "unsicher_markiert", "zeitmarken", "zeitmarken_unlesbar", "text_sub"):
        print(f"- {k}: {st.get(k, 0)}")
    # Schriftlage der abfotografierten Transkripte (ocr_zwangsarbeiter.py)
    ocr = [x for f in sorted((WORK / "ocr").glob("*_zeilen.tsv"))
           for x in csv.DictReader(open(f, encoding="utf-8"), delimiter="\t")]
    if ocr:
        up = sorted(float(x["winkel"]) for x in ocr if x["kursiv"] != "ja")
        it = sorted(float(x["winkel"]) for x in ocr if x["kursiv"] == "ja")
        hist = collections.Counter(round(w) for w in it)
        print(f"\nSchriftlage (OCR, {len(ocr)} Zeilen): aufrecht {len(up)} Zeilen, {up[0]:+.0f} bis {up[-1]:+.0f} Grad; "
              f"kursiv {len(it)} Zeilen, {it[0]:+.0f} bis {it[-1]:+.0f} Grad; kursiv je Grad: "
              + ", ".join(f"{k}: {v}" for k, v in sorted(hist.items())))
    # ---------------- Alignment
    print("\n## Alignment\n")
    al = {i: json.loads((WORK / "aligned" / f"{i}.json").read_text()) for i in specs
          if (WORK / "aligned" / f"{i}.json").exists()}
    nw = sum(a["n_words"] for a in al.values()); na = sum(a["n_aligned"] for a in al.values())
    ast = collections.Counter()
    for a in al.values():
        for k, v in a["stats"].items():
            ast[k] += v
    print(f"Wörter aligniert: {na:,} von {nw:,} ({100*na/nw:.2f} %)".replace(",", "."))
    # seit v2.1: verklebte Token ("Jahren…seit") getrennt, Zahlwoerter mit ß richtig normalisiert
    import re as _re
    from align import spell_number
    glued = collections.Counter()
    for i in specs:
        for t in turns[i]["turns"]:
            glued[i] += sum(1 for tok in t["text"].split() if _re.search(r"\w(…|\.\.\.)\w", tok))
    sz = sum(1 for a in al.values() for w in a["words"]
             if any("ß" in spell_number(d) for d in _re.findall(r"\d+", w["w"])))
    print(f"Verklebte Token getrennt aligniert: {sum(glued.values())} (" + ", ".join(
        f"{i} {n}" for i, n in glued.most_common(3)) + f"); Zahlwörter mit ß im ausgeschriebenen Wort: {sz}")
    print(f"Stützpunkte aus dem Audio: {ast['stuetzpunkte']} (≈ {ast['stuetzpunkte']/(sum(a['audio_s'] for a in al.values())/3600):.0f} je Stunde), verworfen {ast.get('stuetzpunkt_verworfen', 0)}")
    segs = {i: json.loads((WORK / "segments" / f"{i}.json").read_text())["segments"] for i in specs
            if (WORK / "segments" / f"{i}.json").exists()}
    for k in ("woerter_text_ohne_ton", "woerter_zu_gross", "woerter_fehlgeschlagen"):
        per = {i: a["stats"].get(k, 0) for i, a in al.items() if a["stats"].get(k)}
        print(f"- {k}: {sum(per.values())} {per}")
        if k == "woerter_text_ohne_ton":
            # Dauer geschaetzt mit der Sprechrate desselben Interviews
            for i, n in per.items():
                rate = np.median([o["words_per_s"] for o in segs.get(i, []) if o["keep"]])
                print(f"  {i}: bei {rate:.2f} Wörtern/s (Median der Segmente) ≈ {n/rate/60:.1f} min")
    ev = WORK / "eval" / "marker_eval.csv"
    if ev.exists():
        rows = list(csv.DictReader(open(ev)))
        with_markers = sorted(i for i, a in al.items() if a.get("hand_markers"))
        excluded = sorted(i for i in with_markers if not al[i].get("hand_markers_valid", True))
        print(f"\nInterviews mit Zeitmarken: {len(with_markers)}; nicht gewertet (Marken beziehen sich nicht "
              f"auf diese Aufnahme): {', '.join(excluded) or '-'}")
        drin = [r for r in rows if r.get("teil") != "ausgeschlossen"]
        raus = [r for r in rows if r.get("teil") == "ausgeschlossen"]
        print(f"Handmarken-Maßstab: {len(drin)} Interviews im Datensatz, {sum(int(r['n']) for r in drin)} Marken; "
              f"getrennt ausgewiesen (vom Datensatz ausgeschlossen): "
              + (", ".join(f"{r['id']} ({r['n']} Marken)" for r in raus) or "-")
              + " (Details: work/eval/marker_eval.csv)")
        sj0 = WORK / "eval" / "marker_schiedsrichter.json"
        info0 = json.loads(sj0.read_text()) if sj0.exists() else {}
        D_of = {r["id"]: (float(r["verzug_whisper_s"]) if r.get("verzug_whisper_s") else None) for r in rows}
        A_of = {r["id"]: (float(r["versatz_alignment_s"]) if r.get("versatz_alignment_s") else None) for r in rows}
        in_ds = {r["id"] for r in drin}
        if any(v is not None for v in D_of.values()):
            # Zusammenfassung ueber die Interviews des Datensatzes; das Gruppengespraech eigens
            Ds = sorted(((i, v) for i, v in D_of.items() if v is not None and i in in_ds), key=lambda x: -abs(x[1]))
            As = [v for i, v in A_of.items() if v is not None and i in in_ds]
            print(f"Verzug der Marken, unabhängig vom Alignment gemessen: Whisper-large-v3 hört an je bis zu "
                  f"{info0.get('verzug_je_interview', '?')} Marken je Interview, wo die Wörter vor der Marke enden "
                  f"({info0.get('verzug_gueltig', '?')} von {info0.get('verzug_messungen', '?')} Messungen gültig; "
                  f"work/eval/marker_verzug.csv)")
            print(f"  Verzug D (Marke − Wortende laut Whisper): Median {np.median([v for _, v in Ds]):+.2f} s; "
                  f"|D| ≤ 0,5 s bei {sum(abs(v) <= 0.5 for _, v in Ds)} von {len(Ds)}; darüber: "
                  + ", ".join(f"{i} {v:+.2f} s" for i, v in Ds if abs(v) > 0.5))
            print(f"  Versatz A (Wortende laut Alignment − laut Whisper): Median {np.median(As):+.2f} s, "
                  f"Spanne {min(As):+.2f} bis {max(As):+.2f} s; |A| ≤ 0,5 s bei {sum(abs(v) <= 0.5 for v in As)} "
                  f"von {len(As)}")
            As_big = sorted(((i, A_of[i]) for i in in_ds if A_of.get(i) is not None and abs(A_of[i]) > 0.5),
                            key=lambda x: x[1])
            raw = {r["id"]: r.get("median_abs_roh_s") for r in rows}
            if As_big:
                print("    |A| > 0,5 s: " + ", ".join(f"{i} {v:+.2f} s (Marke gegen Alignment ohne Korrektur: Median "
                                                     f"{raw[i]} s)" for i, v in As_big))
            lag = sorted(((r["id"], float(r["lag_s"])) for r in drin), key=lambda x: -abs(x[1]))
            print(f"  Zum Vergleich der bisherige Verzug aus denselben Marken (Median Wortende − Marke = A − D): "
                  + ", ".join(f"{i} {v:+.2f} s (D {D_of[i]:+.2f}, A {A_of[i]:+.2f})" for i, v in lag if abs(v) > 0.5)
                  + " (übrige ≤ 0,5 s)")
            for r in raus:
                if D_of.get(r["id"]) is not None:
                    print(f"  Ausgeschlossen, {r['id']}: D {D_of[r['id']]:+.2f} s, A {A_of[r['id']]:+.2f} s")
            miss = [i for i, v in D_of.items() if v is None]
            print(f"  Ohne unabhängige Schätzung (zu wenige gültige Messungen – nicht korrigiert): {', '.join(miss) or '-'}; "
                  f"Marken ohne aligniertes Wort (nicht gewertet): "
                  f"{sum(int(r.get('ausgelassen_nicht_aligniert') or 0) for r in rows)}")
        gf = WORK / "eval" / "marker_grob.csv"
        grob = list(csv.DictReader(open(gf, encoding="utf-8"))) if gf.exists() else []
        sj = WORK / "eval" / "marker_schiedsrichter.json"
        info = json.loads(sj.read_text()) if sj.exists() else {}
        if grob and info:
            print(f"Grobe Abweichungen (> 10 s) entscheidet ein vom Alignment unabhängiger Erkenner "
                  f"({info['model']}, ohne Vorgabe des Soll-Texts) und die blinde Hörprobe: Wo endet die Stelle "
                  f"vor der Marke – höchstens {info['tol_s']:.0f} s neben dem Alignment oder neben der Marke?")
            kf = WORK / "eval" / "marker_kontrolle.csv"
            kd = np.array([float(r["d_alignment"]) for r in csv.DictReader(open(kf, encoding="utf-8"))
                           if r["d_alignment"]]) if kf.exists() else np.array([])
            print(f"  Zufallsschwelle {info['threshold']:.0f} von 100 ({info['null_q']}. Perzentil von "
                  f"{info['null_n']} Paaren aus Soll-Text und Fenstern fremder Interviews, Median "
                  f"{info['null_median']}); Positivkontrolle: Whisper findet die Stelle an "
                  f"{info['controls_found']} von {info['controls_n']} unstrittigen Marken (je Interview eine, "
                  f"Abweichung ≤ 2 s) am richtigen Ort"
                  + (f", Ende im Median {np.median(kd):.2f} s daneben, höchstens {kd.max():.2f} s" if kd.size else "")
                  + " (work/eval/marker_kontrolle.csv)")
            if info.get("ohr_n"):
                print(f"  Hörprobe (blind, A/B): {info['ohr_n']} Fälle angehört, {info['ohr_entschieden']} "
                      f"entschieden; wo Ohr und Whisper beide entschieden: gleich in {info['ohr_gleich_whisper']} von "
                      f"{info.get('ohr_beide_entschieden', '?')}; noch nicht angehört, obwohl Pflicht: "
                      f"{info.get('offen_fuers_ohr', '?')}")
            else:
                print("  Hörprobe: noch nicht bewertet – Urteile unten allein von Whisper")
            names = {"marke_falsch": "Marke falsch (Alignment richtig)", "alignment_falsch": "Alignment falsch",
                     "beide_falsch": "beide daneben", "unklar": "unklar"}
            for teil, sel in (("im Datensatz", [g for g in grob if g["id"] not in {r["id"] for r in raus}]),
                              ("ausgeschlossen", [g for g in grob if g["id"] in {r["id"] for r in raus}])):
                if not sel:
                    continue
                cf = collections.Counter(g["urteil"] for g in sel)
                cw = collections.Counter(g["urteil_whisper"] for g in sel)
                print(f"  {teil}: {len(sel)} Fälle – " + ", ".join(f"{names[k]} {cf[k]}" for k in names)
                      + " (Whisper allein: " + ", ".join(f"{cw[k]}" for k in names) + ")")
            c = collections.Counter((g["id"], g["urteil"]) for g in grob)
            print("  je Interview: " + ", ".join(f"{i} {v} {n}" for (i, v), n in sorted(c.items())))
        wrong = {(g["id"], int(g["wi"]), round(float(g["t"]), 2)) for g in grob if g.get("urteil") == "marke_falsch"}
        from eval_markers import evaluate, errors
        out_ids = {r["id"] for r in raus}
        est = {i: {"D": v} for i, v in D_of.items()}
        grp = {"Roter Stern Format A": [], "Format B (ein Interview)": [], "Rechnitz": []}
        n_iv = collections.Counter()
        ok, grp_x = [], []
        var = {"whisper": [], "loo": [], "insample": [], "roh": []}
        for f in sorted((WORK / "aligned").glob("*.json")):
            d, rr = evaluate(f)
            if not d.get("hand_markers_valid", True) or len(rr) < 3:
                continue
            pe = {"id": d["id"], "rows": rr}
            devs = list(errors(pe, "whisper", est))     # korrigiert nur um den unabhängig gemessenen Verzug
            if d["id"] in out_ids:
                grp_x += devs; continue
            for v in var:
                var[v] += list(errors(pe, v, est))
            k = "Format B (ein Interview)" if d["id"] == FORMAT_B_ID else ("Rechnitz" if d["id"].startswith("re_") else "Roter Stern Format A")
            grp[k] += devs
            n_iv[k] += 1
            ok += [x for x, r in zip(devs, rr) if (d["id"], r["wi"], round(r["t"], 2)) not in wrong]
        grp["gesamt (Datensatz)"] = [x for v in list(grp.values()) for x in v]
        n_iv["gesamt (Datensatz)"] = sum(n_iv.values())
        grp["gesamt ohne widerlegte Marken"] = ok
        n_iv["gesamt ohne widerlegte Marken"] = n_iv["gesamt (Datensatz)"]
        if grp_x:
            grp["Rechnitzer Gruppengespräch (ausgeschlossen)"] = grp_x
            n_iv["Rechnitzer Gruppengespräch (ausgeschlossen)"] = len(out_ids)
        print("\nAbweichung Wortende − Marke nach Abzug des unabhängig gemessenen Verzugs:")
        print("\n| Maßstab | Interviews | Marken | Median | p90 | ≤ 2 s | ≤ 5 s |\n|---|---|---|---|---|---|---|")
        for k, v in grp.items():
            v = np.array(v)
            print(f"| {k} | {n_iv[k]} | {len(v)} | {np.median(v):.2f} s | {np.percentile(v,90):.2f} s | "
                  f"{100*np.mean(v<=2):.1f} % | {100*np.mean(v<=5):.1f} % |")
        print(f"\nKorrektur des Verzugs im Vergleich ({n_iv['gesamt (Datensatz)']} Interviews im Datensatz, "
              f"{len(var['roh'])} Marken):")
        print("\n| Korrektur | Median | p90 | ≤ 2 s | ≤ 5 s |\n|---|---|---|---|---|")
        for v, name in (("whisper", "Verzug laut Whisper (unabhängig, gewählt)"),
                        ("loo", "Verzug aus den übrigen Marken desselben Interviews (leave-one-out)"),
                        ("insample", "Verzug aus denselben Marken (bis 29.09.2026)"),
                        ("roh", "keine Korrektur (obere Schranke)")):
            x = np.array(var[v])
            print(f"| {name} | {np.median(x):.2f} s | {np.percentile(x,90):.2f} s | {100*np.mean(x<=2):.1f} % | "
                  f"{100*np.mean(x<=5):.1f} % |")
    # ---------------- Segmente / Datensatz
    print("\n## Datensatz\n")
    excl = {i: s["exclude"] for i, s in specs.items() if s.get("exclude")}
    for i, why in excl.items():
        print(f"Ausgeschlossen: {i} ({h(sum(o['duration'] for o in segs.get(i, [])))} h Kandidaten) – {why}")
    segs = {i: v for i, v in segs.items() if i not in excl}
    sst = collections.Counter()
    for i in segs:
        for k, v in json.loads((WORK / "segments" / f"{i}.json").read_text())["stats"].items():
            sst[k] += v
    print("Wörter an Schnittkanten weggelassen: "
          + ", ".join(f"{k} {sst.get(k, 0)}" for k in ("wort_neben_joker", "wort_neben_gedehnt",
                                                      "rand_weg_ctc", "rand_weg_sprecherwechsel"))
          + f"; Randpolster vor gehörter Sprache gekürzt: {sst.get('polster_vor_sprache_gekuerzt', 0)} Kanten")
    drop = collections.Counter(o["drop_reason"] for v in segs.values() for o in v if not o["keep"])
    ndrop = sum(drop.values())
    cand = sum(o["duration"] for v in segs.values() for o in v)
    kept = sum(o["duration"] for v in segs.values() for o in v if o["keep"])
    print(f"Verworfene Segmente: {ndrop} ({h(cand - kept)} h) – nach Grund: "
          + ", ".join(f"{k} {v}" for k, v in drop.most_common()))
    npass = {i: sum(o["duration"] for o in v if o["keep"]) / max(1e-9, sum(o["duration"] for o in v))
             for i, v in segs.items()}
    cover = {i: sum(o["duration"] for o in v) / audio[i] for i, v in segs.items()}
    print(f"Kandidaten {sum(len(v) for v in segs.values())} ({h(cand)} h), behalten {h(kept)} h; "
          f"Durchlass je Interview {100*min(npass.values()):.0f}–{100*max(npass.values()):.0f} % "
          f"(Median {100*np.median(list(npass.values())):.0f} %); Abdeckung (Kandidaten/Audio) "
          f"{100*min(cover.values()):.0f}–{100*max(cover.values()):.0f} %, niedrigste: "
          + ", ".join(f"{i} {100*c:.0f} %" for i, c in sorted(cover.items(), key=lambda x: x[1])[:8]) + "\n")
    if (DATASET / "metadata.jsonl").exists():
        rows = [json.loads(l) for l in (DATASET / "metadata.jsonl").read_text().splitlines()]
        tot = sum(x["duration"] for x in rows)
        print("| Projekt | Segmente | Stunden | Anteil am Datensatz | Ausbeute (Datensatz/Audio) |\n|---|---|---|---|---|")
        for p in P:
            r = [x for x in rows if x["project"] == p]
            a = sum(audio[i] for i in specs if proj_of[i] == p)
            d = sum(x["duration"] for x in r)
            print(f"| {PROJECTS[p]} | {len(r):,} | {h(d)} | {pct(d/tot)} | {pct(d/a)} |".replace(",", "."))
        print(f"| **gesamt** | {len(rows):,} | {h(tot)} | 100 % | {pct(tot/sum(audio.values()))} |\n".replace(",", "."))
        for av in sorted((ROOT / "archive").glob("v*_*"), key=lambda x: x.name.rsplit("_", 1)[-1])[-1:]:  # juengster Stand
            am = av / "dataset/metadata.jsonl"
            if am.exists():
                prev = [json.loads(l) for l in am.read_text(encoding="utf-8").splitlines()]
                pt = sum(x["duration"] for x in prev)
                print(f"Gegenüber {av.name}: {len(rows) - len(prev):+d} Segmente, {(tot - pt)/3600:+.2f} h\n")
        d = np.array([x["duration"] for x in rows])
        print(f"Segmentlänge: Median {np.median(d):.1f} s, 10./90. Perzentil {np.percentile(d,10):.1f}/{np.percentile(d,90):.1f} s, "
              f"< 5 s {100*np.mean(d<5):.0f} %, 5–15 s {100*np.mean((d>=5)&(d<15)):.0f} %, ≥ 15 s {100*np.mean(d>=15):.0f} %, max {d.max():.1f} s")
        # Satzgrenzen: beginnt der Zieltext klein bzw. endet er ohne Satzzeichen, liegt der Schnitt mitten im Satz
        low = np.mean([x["text"][:1].islower() for x in rows])
        nop = np.mean([not re.search(r'[.!?…"“”]$', x["text"]) for x in rows])
        print(f"Schnitt mitten im Satz: {100*low:.0f} % der Zieltexte beginnen klein, {100*nop:.0f} % enden ohne Satzzeichen\n")
        for f in ("role", "quality", "split"):
            c = collections.Counter(x[f] for x in rows); ch = collections.Counter()
            for x in rows:
                ch[x[f]] += x["duration"]
            print(f"- {f}: " + ", ".join(f"{k} {v} Segm./{h(ch[k])} h" for k, v in c.most_common()))
        ivs = collections.defaultdict(set)
        for x in rows:
            ivs[x["split"]].add(x["interview"])
        print("- Interviews je Teil: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(ivs.items())))
        yi = {i: sum(x["duration"] for x in rows if x["interview"] == i) / audio[i] for i in specs}
        print(f"\nAusbeute je Interview (ohne ausgeschlossene): Median "
              f"{100*np.median([v for i, v in yi.items() if i not in excl]):.0f} %")
        persons = {x["person"] for x in rows if x["role"] == "zeitzeuge"}
        groups = collections.defaultdict(set)
        for x in rows:
            groups[x["group"]].add(x["interview"])
        print(f"\nZeitzeug:innen (Personen): {len(persons)}; Interviews: {len({x['interview'] for x in rows})}; "
              f"Personengruppen: {len(groups)}; mit mehreren Interviews: "
              + "; ".join(", ".join(sorted(v)) for v in sorted(groups.values(), key=sorted) if len(v) > 1))
        print(f"consent_anonymise: " + str(dict(collections.Counter(str(x['consent_anonymise']) for x in rows))))
        print(f"Segmente ohne Interviewdatum: {sum(1 for x in rows if not x.get('date'))}")
        # je Interview: wo geht Audio verloren? Abdeckung (Kandidaten/Audio) -> Durchlass des
        # Guetefilters -> Ausbeute; in dev/test entfallen zusaetzlich die Interviewer-Segmente
        split_of = {x["interview"]: x["split"] for x in rows}
        print("\n| Interview | Audio (h) | Abdeckung | Durchlass | Ausbeute | Teil | Interviewer-Anteil |\n|---|---|---|---|---|---|---|")
        for i in sorted(specs, key=lambda i: yi[i]):
            iv = sum(o["duration"] for o in segs.get(i, []) if o["keep"] and o["role"] == "interviewer") / audio[i]
            print(f"| {i} | {h(audio[i])} | {pct(cover.get(i, 0))} | {pct(npass.get(i, 0))} | {pct(yi[i])} | "
                  f"{split_of.get(i, 'ausgeschlossen' if i in excl else '-')} | {pct(iv)} |")
    ew = WORK / "eval" / "segments_whisper.csv"
    if ew.exists():
        print("\n## Unabhängige Prüfung (Whisper-small)\n")
        r = list(csv.DictReader(open(ew, encoding="utf-8")))
        kept = [x for x in r if x["kind"] == "keep"]
        dropped = [x for x in r if x["kind"] == "drop"]
        from eval_segments import bad_flags, MIN_OTHERS
        for x, b in zip(r, bad_flags(r)):
            x["_bad"] = b
        bad = lambda x: x["_bad"]
        print(f"Stichprobe: {len(kept)} behaltene, {len(dropped)} verworfene Segmente aus "
              f"{len({x['id'] for x in r})} der {sum(1 for s in specs.values() if not s.get('exclude'))} "
              f"Interviews im Datensatz; 'schlecht' = CER mehr als 0,3 über dem Median der übrigen behaltenen "
              f"Segmente des Interviews (bei weniger als {MIN_OTHERS}: des Projekts) oder > 0,9; beide Schichten "
              f"getrennt und je nach Dauer gezogen - die Anteile gelten je Schicht (nach Stunden)\n")
        print("| Teilmenge | n | CER (Median) | Anteil > 0,6 | „schlecht“ |\n|---|---|---|---|---|")
        subsets = [("behalten, alle", kept)] + [(f"– {PROJECTS[p]}", [x for x in kept if x["project"] == p]) for p in P] \
            + [(f"– Stufe {q}", [x for x in kept if x["quality"] == q]) for q in ("hoch", "mittel")] \
            + [("verworfen", dropped)]
        for name, sub in subsets:
            c = np.array([float(x["cer"]) for x in sub])
            print(f"| {name} | {len(sub)} | {np.median(c):.3f} | {100*np.mean(c>0.6):.1f} % | "
                  f"{100*np.mean([bad(x) for x in sub]):.1f} % |")
        sh = [x for x in kept if x.get("cer_shift+1")]
        if sh:
            o = np.array([float(x["cer"]) for x in sh]); m = np.array([float(x["cer_shift-1"]) for x in sh])
            p = np.array([float(x["cer_shift+1"]) for x in sh])
            print(f"\nVerschiebetest n={len(sh)}: Original am besten {100*np.mean((o<=m)&(o<=p)):.0f} %; "
                  f"Median CER {np.median(o):.3f} vs. {np.median(m):.3f} (−1 s) / {np.median(p):.3f} (+1 s)")
        from eval_segments import calibration
        cal = calibration(r)
        print(f"\nKalibrierung der Verwerfregel: Stichprobe {len(r)} ({cal['n']['keep']} behaltene, "
              f"{cal['n']['drop']} verworfene Segmente, je Schicht nach Dauer gezogen), hochgerechnet auf den "
              f"Bestand ({cal['strata_h']['keep']:.2f} h behalten, {cal['strata_h']['drop']:.2f} h verworfen; "
              f"Schichtgrößen: {cal['strata_source']}): {100*cal['bad_share_h']:.1f} % der Stunden schlecht "
              f"(ungewichtet in der Stichprobe {100*cal['bad_share_sample']:.1f} %); 95 %-Intervalle aus einem "
              f"geschichteten Bootstrap\n")
        print("| Regel | in der Stichprobe | verwirft (h) | davon schlecht | findet von allen schlechten | "
              "Verlust guter Segmente (h) |\n|---|---|---|---|---|---|")
        span = lambda v, c: "–" if v != v else f"{100*v:.0f} % ({100*c[0]:.0f}–{100*c[1]:.0f} %)"
        for x in cal["rules"]:
            print(f"| {x['rule']} | {x['n_sample']} | {x['flag_h']:.2f} | {span(x['precision'], x['precision_ci'])} | "
                  f"{span(x['recall'], x['recall_ci'])} | {x['good_lost_h']:.2f} |")
    # ---------------- Hoerprobe(n): menschliche Bewertung
    probes = [(ROOT / "archive/v2_2026-09-27/hoerprobe", "Stand v2 (27.09.2026, seed 7)"),
              (ROOT / "archive/v2.1_2026-09-28/hoerprobe", "Stand v2.1 (28.09.2026, seed 28, Fehlerarten)"),
              (ROOT / "samples", "aktueller Stand")]
    shown = False
    for d, label in probes:
        tsv, csvf = d / "pruefliste.tsv", d / "bewertung.csv"
        if not (tsv.exists() and csvf.exists()):
            continue
        sample = {r["file_name"]: r for r in csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t")}
        how = {r.get("ziehung") for r in sample.values()} - {None, ""}   # seit 01.10.2026 in der TSV
        if how:
            label += ", " + ", ".join(sorted(how))
        from make_samples import read_votes, FLAGS   # altes (file_name;bewertung) und neues Format
        allv = [v for v in read_votes(csvf) if v["file_name"] in sample]
        full = [v for v in allv if v["urteil"] != "offen"]
        votes = [(v["file_name"], v["alt"] or v["urteil"]) for v in full]
        if not votes:
            continue
        if not shown:
            print("\n## Hörprobe (menschliche Bewertung)\n")
            shown = True
        c = collections.Counter(v for _, v in votes)
        n, k = len(votes), c["passt"]
        z = 1.96                                 # Wilson-Intervall, 95 %
        mid = (k / n + z * z / (2 * n)) / (1 + z * z / n)
        half = z / (1 + z * z / n) * np.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n))
        print(f"{label}: {n} von {len(sample)} bewertet – "
              + ", ".join(f"{x} {m} ({100*m/n:.0f} %)" for x, m in c.most_common())
              + f"; passt 95 %-Intervall {100*(mid-half):.0f}–{100*(mid+half):.0f} %")
        byp = collections.defaultdict(collections.Counter)
        for f, v in votes:
            byp[sample[f]["projekt"]][v] += 1
        print("  je Projekt: " + "; ".join(f"{p}: passt {cc['passt']}/{sum(cc.values())}"
                                            for p, cc in sorted(byp.items())))
        fc = collections.Counter(f for v in full for f in v["flags"])
        if fc:
            print("  Fehlerarten (mehrere je Segment möglich): "
                  + ", ".join(f"{g} {lbl} {fc[k]}" for k, g, lbl in FLAGS if fc[k]))
        byq = collections.defaultdict(collections.Counter)
        for f, v in votes:
            byq[sample[f]["qualitaet"]][v] += 1
        print("  je Gütestufe: " + "; ".join(f"{q}: passt {cc['passt']}/{sum(cc.values())} "
                                              f"({100*cc['passt']/sum(cc.values()):.0f} %)"
                                              for q, cc in sorted(byq.items())))
        uf = d / "ursachen.csv"                  # Einordnung der Beanstandungen (von Hand)
        if uf.exists():
            u = list(csv.DictReader(open(uf, encoding="utf-8"), delimiter=";"))
            print(f"  Ursachen der {len(u)} Beanstandungen (von Hand eingeordnet, {uf.relative_to(ROOT)}): "
                  + ", ".join(f"{k} {n}" for k, n in collections.Counter(x["ursache"] for x in u).most_common()))
        ew2 = d.parent / "work/eval/segments_whisper.csv"
        if ew2.exists():                         # automatisches Urteil desselben Stands
            r2 = list(csv.DictReader(open(ew2, encoding="utf-8")))
            from eval_segments import bad_flags      # dieselbe Einstufung wie fuer den aktuellen Stand
            bad2 = np.mean([b for x, b in zip(r2, bad_flags(r2)) if x["kind"] == "keep"])
            print(f"  zum Vergleich, Whisper-Kriterium an diesem Stand: {100*bad2:.1f} % der behaltenen Segmente 'schlecht'")
    for d, label, wf in ((ROOT / "archive/v2.2_2026-09-28/nachpruefung", "v2.1 gegen v2.2 (Polstergrenze an beiden Kanten)",
                          ROOT / "archive/v2.2_2026-09-28/nachpruefung/nachpruefung_whisper.csv"),
                         (ROOT / "samples", "aktuell", WORK / "eval" / "nachpruefung_whisper.csv")):
        nt, nf = d / "nachpruefung.tsv", d / "nachpruefung.csv"
        if not nt.exists():
            continue
        n_pairs = sum(1 for _ in open(nt, encoding="utf-8")) - 1
        print(f"\nNachprüfung (A/B, blind) {label}: {n_pairs} Paare – "
              + ("bewertet:" if nf.exists() else "noch nicht bewertet"))
        if wf.exists():
            w = list(csv.DictReader(open(wf, encoding="utf-8")))
            dd = np.array([float(x["cer_neu"]) - float(x["cer_alt"]) for x in w])
            print(f"  Whisper-Vorprüfung (eval_segments.py --vergleich): neu besser {int(np.sum(dd < -0.01))}, "
                  f"gleich {int(np.sum(abs(dd) <= 0.01))}, neu schlechter {int(np.sum(dd > 0.01))}")
        if nf.exists():
            import contextlib, io
            from make_samples import summary_compare, compare_verdicts
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                summary_compare(nf, nt)
            print("  " + buf.getvalue().strip().replace("\n", "\n  "))
            res = compare_verdicts(nf, nt)
            print("  gesamt: " + ", ".join(f"{v} {sum(1 for _, x, _ in res if x == v)}"
                                           for v in ("neu besser", "gleich", "alt besser")))
            if wf.exists():                       # stimmt Whisper mit dem Gehoer ueberein?
                wv = {x["paar"]: ("neu besser" if float(x["cer_neu"]) - float(x["cer_alt"]) < -0.01 else
                                  "alt besser" if float(x["cer_neu"]) - float(x["cer_alt"]) > 0.01 else "gleich") for x in w}
                agree = sum(1 for p, v, _ in res if wv.get(p["paar"]) == v)
                print(f"  Whisper-Vorprüfung trifft das Hörurteil in {agree} von {len(res)} Paaren")
    kf = WORK / "eval" / "kantenregeln.csv"
    if kf.exists():
        print("\nSchnittkanten-Regeln im Vergleich (pipeline/eval_edges.py; Regeln für Randwörter, ohne die "
              "Polstergrenze (seit v2.2); Verlust gegenüber Stand v2, Wirkung auf die 80 bewerteten v2-Segmente "
              "ohne Gruppengespräch)\n")
        print("| Variante | Stunden | Verlust | Dialekt / Hochsprache | r | Beanstandete: unverändert / gekürzt / entfallen "
              "| „passt“: unverändert / gekürzt / entfallen |\n|---|---|---|---|---|---|---|")
        for x in csv.DictReader(open(kf, encoding="utf-8")):
            print(f"| {x['variante']} | {x['stunden']} | {x['verlust_pct']} % | {x['verlust_dialekt_pct']} % / "
                  f"{x['verlust_hochsprache_pct']} % | {x['r_dialekt']} | {x['beanstandet_unveraendert']} / "
                  f"{x['beanstandet_gekuerzt']} / {x['beanstandet_entfallen']} | {x['passt_unveraendert']} / "
                  f"{x['passt_gekuerzt']} / {x['passt_entfallen']} |")
    # ---------------- Rechenzeit (Emissionen, aus den Protokollen)
    runs = []
    for f in sorted((WORK / "logs").glob("emissions*.log")):
        for m in re.finditer(r"\]\s+(\S+)\s+([\d.]+) min in\s+([\d.]+) min", f.read_text(errors="replace")):
            runs.append((m.group(1), float(m.group(2)), float(m.group(3))))
    if runs:
        a_min = sum(x[1] for x in runs); c_min = sum(x[2] for x in runs)
        print(f"\n## Rechenzeit\n\nEmissionen (CPU) laut work/logs/emissions*.log: {len({x[0] for x in runs})} von "
              f"{len(specs)} Interviews protokolliert, {a_min/60:.1f} h Audio in {c_min/60:.1f} h "
              f"(Faktor {a_min/c_min:.1f}); hochgerechnet auf {h(sum(audio.values()))} h Audio: "
              f"{sum(audio.values())/60/(a_min/c_min)/60:.1f} h")


if __name__ == "__main__":
    main()
