#!/usr/bin/env python3
"""
Step 2: package for the blind rating of two runs on the test segments (A against B).

  python3 training/blind_rating.py --a baseline-large-v3 --b <endmodell> --tag beam5fb
  python3 training/blind_rating.py --a w-r16-lr5e-4-e3/ep3 --b pk-r16-lr2e-3-e3/ep3@greedy --tag "" \
      --out samples/blind_whisper_parakeet

Runs as <lauf>[@<tag>] (predictions_test[_<tag>].csv); without @, --tag applies.

Sample from the test part (persons not in training): per interview
--je-interview random segments in which the two outputs differ, plus
--kontrollen pairs with identical output as attention checks (should get
"equal"). Sides A/B random per segment. Both outputs are set uniformly
(lower case, no punctuation, numbers spelled out) so that the writing habits
of the fine-tuned model do not give it away; the archive transcript is not
shown - the rating is against what is heard.

-> --out (default samples/blindbewertung/; index.html, Anleitung.txt, audio/b###.flac; no names in
   file names, no key - so it can be passed on; an existing folder is not overwritten) and
   training/blind/schluessel_<b>.json
   (which side is which model - stays here). Only figures are printed.
"""
import argparse, csv, json, re, shutil, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import ROOT, RUNS, DATASET, words, norm_word

OUT = ROOT / "samples" / "blindbewertung"
KEY = ROOT / "training" / "blind"


def neutral(text):
    """Uniform setting: lower case, no punctuation, numbers spelled out."""
    out = []
    for tok in text.split():
        if any(c.isdigit() for c in tok):              # norm_word writes ß as ss (alignment vocabulary);
            out.append(norm_word(tok).replace("|", " ").replace("issig", "ißig"))  # in number words only "dreißig"
            continue
        t = re.sub(r"[^\w'\-]", "", tok.lower()).strip("-'")
        if t:
            out.append(t)
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="baseline-large-v3")
    ap.add_argument("--b", required=True)
    ap.add_argument("--tag", default="beam5fb")
    ap.add_argument("--je-interview", type=int, default=8)
    ap.add_argument("--kontrollen", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="Zielordner (Standard samples/blindbewertung)")
    a = ap.parse_args()
    out = Path(a.out).resolve() if a.out else OUT
    if out.exists():
        sys.exit(f"{out} gibt es schon - anderen --out wählen oder den alten Ordner selbst wegräumen")

    def pred_file(run):
        name, tag = run.split("@", 1) if "@" in run else (run, a.tag)
        return RUNS / name / f"predictions_test{'_' + tag if tag else ''}.csv"
    meta = {}
    for line in (DATASET / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["split"] == "test":
            meta[r["file_name"]] = r
    P = {k: {r["file_name"]: r["hypothese"] for r in csv.DictReader(open(pred_file(k), encoding="utf-8"))}
         for k in (a.a, a.b)}
    rng = np.random.default_rng(a.seed)
    diff, same = defaultdict(list), []
    for f in sorted(P[a.a]):
        (same.append(f) if neutral(P[a.a][f]) == neutral(P[a.b][f]) else diff[meta[f]["interview"]].append(f))
    picked = []
    for iv in sorted(diff):
        fs = diff[iv]
        picked += [(f, False) for f in rng.choice(fs, min(a.je_interview, len(fs)), replace=False)]
    picked += [(f, True) for f in rng.choice(same, min(a.kontrollen, len(same)), replace=False)]
    order = rng.permutation(len(picked))
    (out / "audio").mkdir(parents=True)
    KEY.mkdir(parents=True, exist_ok=True)
    items, key = [], {}
    for n, i in enumerate(order, 1):
        f, ctrl = picked[i]
        iid = f"b{n:03d}"
        b_left = bool(rng.integers(2))                    # True: run B is shown as "A" on the left
        left, right = (P[a.b][f], P[a.a][f]) if b_left else (P[a.a][f], P[a.b][f])
        shutil.copyfile(DATASET / f, out / "audio" / f"{iid}.flac")
        items.append({"id": iid, "A": neutral(left), "B": neutral(right), "d": round(meta[f]["duration"], 1)})
        key[iid] = {"file_name": f, "interview": meta[f]["interview"], "project": meta[f]["project"], "kontrolle": ctrl,
                    "A": a.b if b_left else a.a, "B": a.a if b_left else a.b}
    (KEY / f"schluessel_{a.b.replace('/', '-').replace('@', '-')}.json").write_text(json.dumps(
        {"a": a.a, "b": a.b, "tag": a.tag, "seed": a.seed, "je_interview": a.je_interview,
         "interviews": {iv: len(fs) for iv, fs in sorted(diff.items())}, "gleiche_ausgabe_gesamt": len(same),
         "items": key}, ensure_ascii=False, indent=1), encoding="utf-8")
    minutes = sum(meta[key[i]["file_name"]]["duration"] for i in key) / 60
    (out / "Anleitung.txt").write_text(GUIDE.format(n=len(items), minutes=round(minutes)), encoding="utf-8")
    data = json.dumps({"items": items}, ensure_ascii=False, separators=(",", ":"))
    (out / "index.html").write_text(PAGE.replace("__DATA__", data.replace("</", "<\\/")), encoding="utf-8")
    size = sum(p.stat().st_size for p in (out / "audio").iterdir()) / 1e6
    print(f"-> {out}: {len(items)} Paare ({len(items) - min(a.kontrollen, len(same))} verschieden aus {len(diff)} Interviews, "
          f"{min(a.kontrollen, len(same))} Kontrollen), Audio {size:.0f} MB; gleiche Ausgabe insgesamt: {len(same)} von {len(P[a.a])}")
    print(f"-> Schlüssel: {KEY / ('schluessel_' + a.b.replace('/', '-').replace('@', '-') + '.json')} (nicht weitergeben)")


GUIDE = """Blinde Bewertung – Anleitung
===========================

Worum es geht
Zu {n} kurzen Ausschnitten aus Oral-History-Interviews gibt es je zwei maschinelle
Abschriften, A und B. Bitte hören Sie jeden Ausschnitt und beurteilen Sie, welche
Abschrift das Gesprochene besser wiedergibt. Von welchem System welche Abschrift
stammt, ist verborgen und wechselt zufällig.

So geht's
1. Die ZIP-Datei entpacken (nicht direkt aus der ZIP-Datei heraus öffnen).
2. index.html mit Chrome oder Firefox öffnen (Doppelklick oder ins Browserfenster ziehen).
3. Oben Ihren Namen oder ein Kürzel eintragen.
4. Je Ausschnitt anhören (Leertaste), dann urteilen mit den Tasten 1 bis 5; die Seite
   springt zum nächsten Ausschnitt.
5. Zwischendurch und am Ende "CSV speichern" - die Datei bewertung_<Name>.csv landet im
   Download-Ordner. Bitte diese Datei zurückschicken.

Dauer: zusammen {minutes} Minuten Audio, mit dem Bewerten etwa 45 bis 60 Minuten. Pausen
sind möglich: Die Eingaben bleiben im selben Browser gespeichert, solange die Seite aus
demselben Ordner geöffnet wird. Sicher ist nur die gespeicherte CSV-Datei.

Maßstab ist allein das Gehörte. Dialekt darf hochdeutsch geschrieben sein, Füllwörter
(äh, ähm) zählen nicht. Beide Abschriften sind gleich gesetzt (klein, ohne Satzzeichen).

Bitte unabhängig voneinander bewerten und sich vorher nicht absprechen.

Vertraulich: Die Aufnahmen stammen aus Interviews mit Zeitzeuginnen und Zeitzeugen und
enthalten Namen. Bitte nicht weitergeben und den Ordner nach der Bewertung löschen.
"""


PAGE = r"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blinde Bewertung</title>
<style>
:root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e2e0dc;--card:#fff;--ok:#2d6a4f;--acc:#7a3b2e;--chip:#f3f1ee;--cur:#fff6dc;--sel:#1a1a1a;--selfg:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#17181a;--fg:#e8e6e3;--mut:#9b9b9b;--line:#2e3033;--card:#1f2124;--ok:#74c69d;--acc:#d99a86;--chip:#26282b;--cur:#2f2a1c;--sel:#e8e6e3;--selfg:#17181a}}
*{box-sizing:border-box}
body{margin:0;padding:20px 16px 80px;background:var(--bg);color:var(--fg);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{max-width:860px;margin:0 auto}
h1{font-size:1.35rem;margin:0 0 4px}.sub{color:var(--mut);font-size:.9rem;margin-bottom:10px}
.help{font-size:.88rem;color:var(--mut);background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 14px;margin-bottom:10px}
.help summary{cursor:pointer;color:var(--fg);font-weight:600}.help p{margin:6px 0}
kbd{font:inherit;font-size:.8rem;border:1px solid var(--line);border-radius:4px;padding:0 4px;background:var(--chip);color:var(--fg)}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);margin-bottom:12px;display:flex;gap:8px 14px;align-items:center;flex-wrap:wrap;z-index:2;font-size:.9rem}
input[type=text]{font:inherit;font-size:.9rem;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
button{font:inherit;font-size:.85rem;padding:4px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.card{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:10px;padding:12px 16px;margin-bottom:12px}
.card.cur{background:var(--cur);outline:2px solid var(--acc)}.card.done{border-left-color:var(--ok)}
body.open .card.done{display:none}
.top{font-size:.8rem;color:var(--mut);margin-bottom:6px}
audio{width:100%;height:34px;margin-bottom:6px}
.t{display:grid;grid-template-columns:28px 1fr;gap:8px;padding:6px 0;border-top:1px solid var(--line)}
.lab{font-weight:700;color:var(--acc)}
.scale{display:flex;gap:4px;flex-wrap:wrap;margin-top:8px}
.scale button{font-size:.82rem;border-radius:14px;background:var(--chip)}
.scale button[aria-pressed=true]{background:var(--sel);color:var(--selfg);border-color:var(--sel)}
</style></head><body><div class="wrap">
<h1>Blinde Bewertung zweier Abschriften</h1>
<div class="sub" id="sub"></div>
<details class="help" open><summary>So wird bewertet</summary>
<p>Zu jedem Ausschnitt gibt es zwei maschinelle Abschriften, <b>A</b> und <b>B</b>. Welche von welchem System stammt, ist
verborgen und wechselt zufällig. Beide sind gleich gesetzt (klein, ohne Satzzeichen, Zahlen ausgeschrieben) – bitte nur den
Wortlaut vergleichen.</p>
<p><b>Frage:</b> Welche Abschrift gibt wieder, was gesprochen wird? Maßstab ist allein das Gehörte, nicht schöne Sprache.
Dialekt darf hochdeutsch geschrieben sein; Füllwörter (äh, ähm) zählen nicht.</p>
<p>Skala: <kbd>1</kbd> A deutlich besser · <kbd>2</kbd> A etwas besser · <kbd>3</kbd> gleich gut · <kbd>4</kbd> B etwas besser ·
<kbd>5</kbd> B deutlich besser.</p>
<p>Tastatur: <kbd>Leertaste</kbd> abspielen/anhalten, <kbd>1</kbd>–<kbd>5</kbd> Urteil (springt weiter), <kbd>↓</kbd><kbd>↑</kbd>
blättern. Die Eingaben bleiben in diesem Browser gespeichert. Zum Schluss „CSV speichern“ und die Datei zurückschicken.</p></details>
<div class="bar"><label>Ihr Name <input type="text" id="rater" placeholder="Name oder Kürzel"></label>
<label><input type="checkbox" id="open"> nur offene</label><span id="prog"></span><button id="csv">CSV speichern</button></div>
<div id="list"></div></div>
<script>
const D=__DATA__,IT=D.items,KEY="blindbewertung";
const SC=[[-2,"1 A deutlich besser"],[-1,"2 A etwas besser"],[0,"3 gleich gut"],[1,"4 B etwas besser"],[2,"5 B deutlich besser"]];
let st={};try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}};
const rater=document.getElementById("rater");rater.value=st._rater||"";rater.oninput=()=>{st._rater=rater.value.trim();save()};
document.getElementById("sub").textContent=`${IT.length} Ausschnitte · zusammen ${Math.round(IT.reduce((s,x)=>s+x.d,0)/60)} Minuten Audio`;
const list=document.getElementById("list");const cards=[];let cur=0;
const get=id=>st[id]||(st[id]={u:null});
function prog(){const done=IT.filter(x=>st[x.id]&&st[x.id].u!==null).length;document.getElementById("prog").textContent=`${done} von ${IT.length} bewertet`}
function paint(c,x){const s=st[x.id]||{};c.classList.toggle("done",s.u!==null&&s.u!==undefined);
  c.querySelectorAll(".scale button").forEach(b=>b.setAttribute("aria-pressed",String(s.u!==undefined&&s.u!==null&&+b.dataset.v===s.u)))}
IT.forEach((x,i)=>{const c=document.createElement("div");c.className="card";
  const top=document.createElement("div");top.className="top";top.textContent=`${x.id} · ${String(x.d).replace(".",",")} s`;
  const au=document.createElement("audio");au.controls=true;au.preload="none";au.src=`audio/${x.id}.flac`;
  c.append(top,au);
  for(const side of ["A","B"]){const t=document.createElement("div");t.className="t";
    const l=document.createElement("div");l.className="lab";l.textContent=side;
    const p=document.createElement("div");p.textContent=x[side]||"(leer)";
    t.append(l,p);c.append(t)}
  const sc=document.createElement("div");sc.className="scale";
  for(const [v,lab] of SC){const b=document.createElement("button");b.textContent=lab;b.dataset.v=v;
    b.onclick=()=>{const s=get(x.id);s.u=s.u===v?null:v;save();paint(c,x);prog()};sc.append(b)}
  c.append(sc);c.onclick=e=>{if(!["BUTTON","INPUT","AUDIO","LABEL"].includes(e.target.tagName)){cur=i;focus(false)}};
  list.append(c);cards.push(c);paint(c,x)});
function vis(){return cards.filter(c=>c.offsetParent!==null)}
function focus(scroll=true){cards.forEach(c=>c.classList.remove("cur"));const c=cards[cur];if(!c)return;c.classList.add("cur");if(scroll)c.scrollIntoView({block:"center"})}
function step(d){const v=vis();let i=v.indexOf(cards[cur]);i=i<0?0:Math.min(v.length-1,Math.max(0,i+d));cur=cards.indexOf(v[i]);
  document.querySelectorAll("audio").forEach(a=>a.pause());focus()}
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT"&&e.target.type==="text")return;if(e.metaKey||e.ctrlKey)return;
  const c=cards[cur],x=IT[cur];if(!c)return;
  if(e.key===" "){e.preventDefault();const a=c.querySelector("audio");a.paused?a.play():a.pause();return}
  if(["1","2","3","4","5"].includes(e.key)){const s=get(x.id);s.u=+e.key-3;save();paint(c,x);prog();step(1);return}
  if(e.key==="ArrowDown"||e.key==="j"){e.preventDefault();step(1)}if(e.key==="ArrowUp"||e.key==="k"){e.preventDefault();step(-1)}});
document.addEventListener("play",e=>{document.querySelectorAll("audio").forEach(a=>{if(a!==e.target)a.pause()})},true);
document.getElementById("open").onchange=e=>{document.body.classList.toggle("open",e.target.checked);step(0)};
document.getElementById("csv").onclick=()=>{if(!st._rater){alert("Bitte zuerst oben Ihren Namen oder ein Kürzel eintragen.");rater.focus();return}
  const q=s=>'"'+String(s).replace(/"/g,'""')+'"';const rows=["bewerter;item;urteil"];
  for(const x of IT){const s=st[x.id]||{};rows.push([q(st._rater),x.id,s.u===null||s.u===undefined?"":s.u].join(";"))}
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\n")+"\n"],{type:"text/csv"}));
  a.download=`bewertung_${st._rater.replace(/[^\w-]+/g,"_")}.csv`;a.click()};
focus(false);prog();
</script></body></html>
"""

if __name__ == "__main__":
    main()
