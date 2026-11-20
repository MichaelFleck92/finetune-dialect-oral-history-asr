#!/usr/bin/env python3
"""
Hoerprobe mit Bewertung: samples/pruefliste.html

Zufaellige, dauerproportionale Stichprobe aus dem fertigen Datensatz (jede
Stunde Material hat dieselbe Chance, gezogen zu werden - keine Schichtung, die
man hinterher umrechnen muesste). Je Probe: Audio, Text und ein Urteil (18.11.2026):

  passt                Ausschnitt und Text stimmen ueberein
  falsch geschnitten   Fehler der Pipeline: Wort am Anfang/Ende abgeschnitten, zu viel Ton,
                       andere Stimme im Ausschnitt, Text gehoert an eine andere Stelle
  Transkript falsch    Fehler des Archivtranskripts: Schnitt stimmt, aber im Inneren fehlen
                       Woerter, stehen zu viel, sind falsch oder umformuliert
  unsicher             nicht zu entscheiden

"falsch geschnitten" und "Transkript falsch" koennen beide zutreffen; die Einzelheiten
(welche Kante, welche Art) sind optional. Tastatur: 1-4 Urteil, Leertaste Abspielen,
j/k naechste/vorige Probe.

Die Bewertungen bleiben im Browser gespeichert und lassen sich als CSV
herunterladen; pipeline/make_samples.py --summary samples/bewertung.csv
fasst sie zusammen (Anteile mit 95-%-Intervall nach Wilson, je Projekt; auch die
frueheren Formate). Rein lokal - die Seite laedt das Audio aus dataset/.

  python3 pipeline/make_samples.py --n 150 --seed 2027
  python3 pipeline/make_samples.py --n 30 --seed 1001 --gleichverteilt

--gleichverteilt zieht jedes Segment mit derselben Chance statt nach Dauer -
dann beschreibt die Quote Segmente statt Stunden. Die Guetestufe steht nicht auf
dem Pruefblatt (sichtbar beeinflusste sie das Urteil); --mit-stufe zeigt sie wieder.
In pruefliste.tsv steht sie weiter, ebenso die Art der Ziehung.

Nachpruefung nach einer Regelaenderung: dieselben bewerteten Segmente vorher
(Archiv) und jetzt, blind als A/B - welche Fassung vorn steht, ist je Paar
zufaellig. Nur Paare, deren Schnitt sich geaendert hat.

  python3 pipeline/make_samples.py --vergleich archive/v2.1_2026-09-28
  python3 pipeline/make_samples.py --summary-vergleich samples/nachpruefung.csv
"""
import argparse, collections, csv, html, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, DATASET

OUT = ROOT / "samples"

# Hauptkategorien (Spalte, Knopf) und Einzelheiten (Spalte, Kategorie, Knopf) - Reihenfolge = Spalten der CSV
CATS = [("schnitt", "falsch geschnitten"), ("transkript", "Transkript falsch")]
FLAGS = [
    ("anfang_abgeschnitten", "schnitt", "Anfang: Wort abgeschnitten"),
    ("anfang_zu_viel", "schnitt", "Anfang: zu viel Ton"),
    ("ende_abgeschnitten", "schnitt", "Ende: Wort abgeschnitten"),
    ("ende_zu_viel", "schnitt", "Ende: zu viel Ton"),
    ("andere_stimme", "schnitt", "andere Stimme"),
    ("text_anderer_inhalt", "schnitt", "Text gehört an eine andere Stelle"),
    ("text_fehlt", "transkript", "Wörter fehlen"),
    ("text_zu_viel", "transkript", "Wörter zu viel"),
    ("text_falsches_wort", "transkript", "falsches Wort"),
    ("text_umformuliert", "transkript", "umformuliert"),
]
FLAG_CAT = {k: c for k, c, _ in FLAGS}


def pick(rows, n, seed, uniform=False):
    rng = random.Random(seed)
    if uniform:                                  # jedes Segment gleich wahrscheinlich
        return [rows[i] for i in rng.sample(range(len(rows)), min(n, len(rows)))]
    w = [r["duration"] for r in rows]
    idx = set()
    while len(idx) < min(n, len(rows)):
        idx.add(rng.choices(range(len(rows)), weights=w)[0])
    return [rows[i] for i in sorted(idx, key=lambda i: rng.random())]


PAGE = """<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hörprobe</title>
<style>
:root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e2e0dc;--card:#fff;--ok:#2d6a4f;--bad:#b3261e;--warn:#9a6a00;--acc:#7a3b2e;--chip:#f3f1ee}
@media (prefers-color-scheme:dark){:root{--bg:#17181a;--fg:#e8e6e3;--mut:#9b9b9b;--line:#2e3033;--card:#1f2124;--ok:#74c69d;--bad:#f2b8b5;--warn:#e0b050;--acc:#d99a86;--chip:#26282b}}
*{box-sizing:border-box}
body{margin:0;padding:24px 16px 80px;background:var(--bg);color:var(--fg);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{max-width:860px;margin:0 auto}
h1{font-size:1.4rem;margin:0 0 4px}.sub{color:var(--mut);font-size:.9rem;margin-bottom:14px}
.help{font-size:.88rem;color:var(--mut);background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin-bottom:14px}
.help summary{cursor:pointer;color:var(--fg);font-weight:600}
.help dl{margin:8px 0 0}.help dt{font-weight:600;color:var(--fg);margin-top:6px}.help dd{margin:0 0 0 12px}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0 10px;border-bottom:1px solid var(--line);margin-bottom:16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2;font-size:.9rem}
.card{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px;scroll-margin-top:70px}
.card.cur{box-shadow:0 0 0 2px var(--acc)}
.card[data-state=passt]{border-left-color:var(--ok)}.card[data-state=fehler]{border-left-color:var(--bad)}.card[data-state=unsicher]{border-left-color:var(--warn)}
body.open .card[data-state=passt],body.open .card[data-state=fehler],body.open .card[data-state=unsicher]{display:none}
.top{font-size:.8rem;color:var(--mut);display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.txt{padding:8px 12px;border-left:3px solid var(--acc);margin:8px 0}
audio{width:100%;height:34px}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:8px}
.lbl{font-size:.78rem;color:var(--mut);min-width:52px}
button{font:inherit;font-size:.85rem;padding:5px 10px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer}
button.f{font-size:.8rem;padding:3px 9px;border-radius:14px;background:var(--chip)}
button.sel[data-v=passt]{background:var(--ok);color:#fff;border-color:var(--ok)}
button.sel[data-v=unsicher]{background:var(--warn);color:#fff;border-color:var(--warn)}
button.sel[data-v=schnitt],button.sel[data-v=transkript],button.f.sel{background:var(--bad);color:#fff;border-color:var(--bad)}
kbd{font:inherit;font-size:.72rem;opacity:.6;margin-left:4px}
.det{display:none}
.card[data-cs~=schnitt] .det.schnitt,.card[data-cs~=transkript] .det.transkript{display:flex}
.note{width:100%;margin-top:8px;font:inherit;font-size:.85rem;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg)}
</style></head><body><div class="wrap">
<h1>Hörprobe: stimmen Ausschnitt und Text überein?</h1>
<div class="sub">__N__ Segmente, __HOW__ (seed __SEED__) &middot; Datensatz vom __DATE__</div>
<details class="help" open><summary>So wird bewertet</summary>
<dl>
<dt>passt <kbd>1</kbd></dt><dd>Ausschnitt und Text stimmen überein. Fehlende Füllwörter (äh, mhm) und Dialekt,
der hochsprachlich geschrieben ist, sind gewollt und kein Fehler.</dd>
<dt>falsch geschnitten <kbd>2</kbd></dt><dd>Der Fehler liegt am Schnitt: das erste oder letzte Wort ist nur halb
oder gar nicht zu hören, vor dem ersten bzw. nach dem letzten Wort ist Sprache zu hören, die nicht im Text steht,
eine andere Stimme spricht hinein, oder der Text gehört zu einer anderen Stelle.</dd>
<dt>Transkript falsch <kbd>3</kbd></dt><dd>Der Schnitt stimmt, aber das Archivtranskript gibt das Gesprochene im
Inneren nicht richtig wieder: Wörter fehlen oder stehen zu viel, ein Wort ist falsch („Punker“ statt „Bunker“),
oder der Satz ist umformuliert.</dd>
<dd>Beides kann zutreffen. Die Einzelheiten darunter sind freiwillig.</dd>
<dt>unsicher <kbd>4</kbd></dt><dd>wenn es sich nicht entscheiden lässt. Die Notiz ist frei.</dd>
<dt>Tastatur</dt><dd><kbd>Leertaste</kbd> abspielen/anhalten, <kbd>j</kbd>/<kbd>k</kbd> nächste/vorige Probe
(spielt sie ab), <kbd>1</kbd>–<kbd>4</kbd> Urteil für die markierte Probe.</dd>
</dl></details>
<div class="bar"><span id="count"></span><label><input type="checkbox" id="open"> nur offene zeigen</label>
<label><input type="checkbox" id="auto" checked> nach „passt“ weiter</label>
<button id="csv">Bewertungen als CSV speichern</button></div>
__CARDS__
</div>
<script>
const CATS=__CATS__, FLAGS=__FLAGS__, FCAT=__FCAT__;
const KEY="segmentbewertung-__SEED__-__DATE__";
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const cards=[...document.querySelectorAll(".card")];
let cur=0;
function get(id){const s=st[id]||(st[id]={});s.c=s.c||[];s.f=s.f||[];return s}
function verdict(s){return !s?undefined:s.u?"unsicher":s.p?"passt":(s.c&&s.c.length)?"fehler":undefined}
function tidy(id){const s=st[id];if(s&&!s.p&&!s.u&&!s.c.length&&!s.f.length&&!s.n)delete st[id]}
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){};upd()}
function upd(){document.getElementById("count").textContent=cards.filter(c=>verdict(st[c.dataset.id])).length+" von "+cards.length+" bewertet"}
function render(c){const s=st[c.dataset.id]||{c:[],f:[]};
  c.querySelectorAll("button.v").forEach(b=>{const v=b.dataset.v;
    const on=v==="passt"?!!s.p:v==="unsicher"?!!s.u:(s.c||[]).includes(v);b.classList.toggle("sel",on);b.setAttribute("aria-pressed",on)});
  c.querySelectorAll("button.f").forEach(b=>{const on=(s.f||[]).includes(b.dataset.f);b.classList.toggle("sel",on);b.setAttribute("aria-pressed",on)});
  const n=c.querySelector(".note");if(document.activeElement!==n)n.value=s.n||"";
  c.dataset.cs=(s.c||[]).join(" ");c.dataset.state=verdict(s)||""}
function setCur(i,play){if(i<0||i>=cards.length)return;cards.forEach(x=>x.classList.remove("cur"));cur=i;const c=cards[i];
  c.classList.add("cur");c.scrollIntoView({block:"nearest",behavior:"smooth"});
  if(play){document.querySelectorAll("audio").forEach(a=>{if(a!==c.querySelector("audio"))a.pause()});const a=c.querySelector("audio");a.currentTime=0;a.play().catch(()=>{})}}
function next(play){let i=cur+1;const open=document.body.classList.contains("open");
  while(open&&i<cards.length&&verdict(st[cards[i].dataset.id]))i++;setCur(i,play)}
function press(c,v){const id=c.dataset.id,s=get(id);
  if(v==="passt"){s.p=!s.p;if(s.p){s.c=[];s.f=[];s.u=false}}
  else if(v==="unsicher"){s.u=!s.u;if(s.u)s.p=false}
  else{if(s.c.includes(v)){s.c=s.c.filter(x=>x!==v);s.f=s.f.filter(f=>FCAT[f]!==v)}else{s.c.push(v);s.p=false}}
  tidy(id);render(c);save();
  if(v==="passt"&&st[id]&&st[id].p&&document.getElementById("auto").checked)next(true)}
cards.forEach((c,i)=>{const id=c.dataset.id;
  c.addEventListener("click",()=>{if(cur!==i)setCur(i,false)});
  c.querySelector("audio").addEventListener("play",()=>{if(cur!==i)setCur(i,false)});
  c.querySelectorAll("button.v").forEach(b=>b.onclick=()=>press(c,b.dataset.v));
  c.querySelectorAll("button.f").forEach(b=>b.onclick=()=>{const s=get(id),f=b.dataset.f;
    if(s.f.includes(f))s.f=s.f.filter(x=>x!==f);else{s.f.push(f);if(!s.c.includes(FCAT[f]))s.c.push(FCAT[f]);s.p=false}
    tidy(id);render(c);save()});
  c.querySelector(".note").oninput=e=>{get(id).n=e.target.value;tidy(id);save()};
  render(c)});
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT"){if(e.key==="Escape")e.target.blur();return}
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  const c=cards[cur];
  if(e.key===" "){e.preventDefault();const a=c.querySelector("audio");a.paused?a.play().catch(()=>{}):a.pause()}
  else if(e.key==="j"||e.key==="ArrowDown"){e.preventDefault();next(true)}
  else if(e.key==="k"||e.key==="ArrowUp"){e.preventDefault();setCur(cur-1,true)}
  else if("1234".includes(e.key)&&e.key.length===1){press(c,["passt","schnitt","transkript","unsicher"][+e.key-1])}});
document.getElementById("open").onchange=e=>document.body.classList.toggle("open",e.target.checked);
function csvText(){const q=x=>'"'+String(x).replace(/"/g,'""')+'"';
  const rows=[["file_name","urteil"].concat(CATS).concat(FLAGS).concat(["notiz"]).join(";")];
  for(const c of cards){const s=st[c.dataset.id];if(!s)continue;   // auch Notizen ohne Urteil ("offen")
    rows.push([c.dataset.id,verdict(s)||"offen"].concat(CATS.map(k=>(s.c||[]).includes(k)?1:0))
      .concat(FLAGS.map(f=>(s.f||[]).includes(f)?1:0)).concat([q(s.n||"")]).join(";"))}
  return rows.join("\\n")+"\\n"}
document.getElementById("csv").onclick=()=>{const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([csvText()],{type:"text/csv"}));a.download="bewertung.csv";a.click()};
upd();setCur(0,false);
</script></body></html>"""


def build(rows, seed, date, how="zufällig nach Dauer gezogen", show_tier=False):
    det = "\n".join(
        f'<div class="row det {cat}"><span class="lbl">genauer</span>'
        + "".join(f'<button class="f" data-f="{k}">{html.escape(lbl)}</button>' for k, c, lbl in FLAGS if c == cat)
        + "</div>" for cat, _ in CATS)
    cards = []
    for r in rows:
        src = "../dataset/" + r["file_name"]
        cards.append(f"""<div class="card" data-id="{html.escape(r['file_name'])}">
<div class="top"><b>{html.escape(r['interviewee'] or r['interview'])}</b><span>{r['project']}</span>
<span>{r['role']}</span>{f"<span>{r['quality']}</span>" if show_tier else ""}<span>{r['duration']:.1f} s</span></div>
<audio controls preload="none" src="{html.escape(src)}"></audio>
<div class="txt">{html.escape(r['text'])}</div>
<div class="row"><button class="v" data-v="passt">passt<kbd>1</kbd></button><button class="v" data-v="schnitt">falsch geschnitten<kbd>2</kbd></button>
<button class="v" data-v="transkript">Transkript falsch<kbd>3</kbd></button><button class="v" data-v="unsicher">unsicher<kbd>4</kbd></button></div>
{det}
<input class="note" type="text" placeholder="Notiz (optional): was ist zu hören?"></div>""")
    page = (PAGE.replace("__N__", str(len(rows))).replace("__SEED__", str(seed)).replace("__HOW__", how)
            .replace("__DATE__", date).replace("__CATS__", json.dumps([k for k, _ in CATS]))
            .replace("__FLAGS__", json.dumps([k for k, _, _ in FLAGS])).replace("__FCAT__", json.dumps(FLAG_CAT))
            .replace("__CARDS__", "\n".join(cards)))
    OUT.mkdir(exist_ok=True)
    (OUT / "pruefliste.html").write_text(page, encoding="utf-8")
    with open(OUT / "pruefliste.tsv", "w", encoding="utf-8") as fh:
        fh.write("file_name\tprojekt\trolle\tqualitaet\tdauer\tziehung\ttext\n")
        for r in rows:
            fh.write(f"{r['file_name']}\t{r['project']}\t{r['role']}\t{r['quality']}\t{r['duration']:.2f}\t"
                     f"{how}\t{r['text']}\n")


def read_votes(path):
    """Bewertungs-CSV -> [{file_name, urteil (passt|fehler|unsicher|offen), cats, flags, notiz, alt}].
    "offen" = nur eine Notiz, noch kein Urteil. cats: Teilmenge von {schnitt, transkript}, aus den
    Spalten der Hauptkategorien oder - in frueheren Formaten - aus den Einzelheiten abgeleitet.
    Aeltestes Format (Stand v2): file_name;bewertung mit passt|schnitt|text|unsicher."""
    rows = [r for r in csv.reader(open(path, encoding="utf-8-sig", newline=""), delimiter=";") if r]
    head, body = rows[0], rows[1:]
    out = []
    if "urteil" in head:
        for r in body:
            d = dict(zip(head, r))
            flags = [k for k, _, _ in FLAGS if d.get(k) == "1"]
            cats = {k for k, _ in CATS if d.get(k) == "1"} | {FLAG_CAT[f] for f in flags}
            out.append({"file_name": d["file_name"], "urteil": d["urteil"], "alt": None,
                        "cats": cats, "flags": flags, "notiz": d.get("notiz", "")})
    else:
        for r in body:
            out.append({"file_name": r[0], "urteil": "fehler" if r[1] in ("schnitt", "text") else r[1], "alt": r[1],
                        "cats": {"schnitt"} if r[1] == "schnitt" else ({"transkript"} if r[1] == "text" else set()),
                        "flags": [], "notiz": ""})
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return c - h, c + h


def summary(path, show_notes=False):
    allv = read_votes(path)
    votes = [v for v in allv if v["urteil"] != "offen"]   # "offen": nur Notiz, noch kein Urteil
    n = len(votes)
    if not n:
        print("noch keine Urteile"); return
    def line(label, k, m):
        lo, hi = wilson(k, m)
        return f"  {label:22} {k:4} / {m:<4} {100*k/m:5.1f} %  (95 %: {100*lo:.1f} bis {100*hi:.1f})"
    print(f"{n} bewertet" + (f"; dazu {len(allv) - n} nur mit Notiz" if len(allv) > n else ""))
    groups = [("passt", lambda v: v["urteil"] == "passt"),
              ("falsch geschnitten", lambda v: "schnitt" in v["cats"]),
              ("Transkript falsch", lambda v: "transkript" in v["cats"]),
              ("  davon beides", lambda v: {"schnitt", "transkript"} <= v["cats"]),
              ("unsicher", lambda v: v["urteil"] == "unsicher")]
    for label, f in groups:
        print(line(label, sum(map(f, votes)), n))
    by = collections.defaultdict(list)
    for v in votes:                              # Projekt steht im Dateinamen (audio/rs_x/rs_x_...)
        by[v["file_name"].split("/")[1].split("_")[0]].append(v)
    for p, vs in sorted(by.items()):
        m = len(vs)
        print(f"  {p}: " + ", ".join(f"{lab.strip()} {sum(map(f, vs))}/{m}" for lab, f in groups if not lab.startswith("  ")))
    fc = collections.Counter(f for v in votes for f in v["flags"])
    if fc:
        print("Einzelheiten (freiwillig, mehrere je Segment möglich):")
        for k, c, lbl in FLAGS:
            if fc[k]:
                print(f"  {dict(CATS)[c]}: {lbl:36} {fc[k]}")
    notes = [v for v in allv if v["notiz"]]
    if notes and show_notes:
        print("Notizen:")
        for v in notes:
            print(f"  {v['file_name']}  [{v['urteil']}]  {v['notiz']}")
    elif notes:
        print(f"{len(notes)} Notizen (anzeigen mit --notizen)")


PAGE_AB = PAGE.split("<h1>")[0] + """<h1>Nachprüfung: welche Fassung passt besser?</h1>
<div class="sub">__N__ Segmente aus der letzten Hörprobe, deren Schnitt sich geändert hat &middot; __DATE__</div>
<details class="help" open><summary>So wird bewertet</summary>
<dl><dt>A und B</dt><dd>Derselbe Text, zwei Schnitte desselben Ausschnitts. Welche Fassung die neue ist,
ist je Paar zufällig verteilt. Beide anhören und wählen, welche besser zum Text passt – oder „gleich“,
wenn kein Unterschied zu hören ist. Die Notiz ist frei.</dd></dl></details>
<div class="bar"><span id="count"></span><label><input type="checkbox" id="open"> nur offene zeigen</label>
<button id="csv">Bewertungen als CSV speichern</button></div>
__CARDS__
</div>
<script>
const KEY="nachpruefung-__DATE__";
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const cards=[...document.querySelectorAll(".card")];
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){};upd()}
function upd(){document.getElementById("count").textContent=cards.filter(c=>st[c.dataset.id]&&st[c.dataset.id].v).length+" von "+cards.length+" bewertet"}
function render(c){const s=st[c.dataset.id]||{};
  c.querySelectorAll("button.v").forEach(b=>{const on=s.v===b.dataset.v;b.classList.toggle("sel",on);b.setAttribute("aria-pressed",on)});
  const n=c.querySelector(".note");if(document.activeElement!==n)n.value=s.n||"";
  c.dataset.state=s.v?"passt":""}
cards.forEach(c=>{const id=c.dataset.id;
  c.querySelectorAll("button.v").forEach(b=>b.onclick=()=>{const s=st[id]||(st[id]={});
    s.v=s.v===b.dataset.v?undefined:b.dataset.v;if(!s.v&&!s.n)delete st[id];render(c);save()});
  c.querySelector(".note").oninput=e=>{const s=st[id]||(st[id]={});s.n=e.target.value;if(!s.v&&!s.n)delete st[id];save()};
  render(c)});
document.getElementById("open").onchange=e=>document.body.classList.toggle("open",e.target.checked);
function csvText(){const q=x=>'"'+String(x).replace(/"/g,'""')+'"';
  const rows=["paar;urteil;notiz"];
  for(const c of cards){const s=st[c.dataset.id];if(!s)continue;rows.push([c.dataset.id,s.v||"offen",q(s.n||"")].join(";"))}
  return rows.join("\\n")+"\\n"}
document.getElementById("csv").onclick=()=>{const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([csvText()],{type:"text/csv"}));a.download="nachpruefung.csv";a.click()};
upd();
</script></body></html>"""


def build_compare(arch, seed=1):
    """Bewertete Segmente eines archivierten Stands gegen den aktuellen Datensatz."""
    arch = Path(arch).resolve()
    old = {}
    for l in (arch / "dataset/metadata.jsonl").read_text(encoding="utf-8").splitlines():
        x = json.loads(l); old[x["file_name"]] = x
    rated = [r["file_name"] for r in csv.DictReader(open(arch / "hoerprobe/pruefliste.tsv", encoding="utf-8"), delimiter="\t")]
    cur = collections.defaultdict(list)
    for l in (DATASET / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        x = json.loads(l); cur[x["interview"]].append(x)
    rng = random.Random(seed)
    pairs, gone = [], []
    for f in rated:
        m = old[f]; s, e = m["source_start"], m["source_end"]
        same = [x for x in cur[m["interview"]] if x["text"] == m["text"] and x["source_end"] > s and x["source_start"] < e]
        if not same:
            gone.append(f); continue
        n = same[0]
        if abs(n["source_start"] - s) < 0.005 and abs(n["source_end"] - e) < 0.005:
            continue
        a_is_new = rng.random() < 0.5
        pairs.append({"paar": f"p{len(pairs)+1:02d}", "alt": f, "neu": n["file_name"], "a_ist_neu": a_is_new,
                      "text": n["text"], "name": n["interviewee"] or n["interview"],
                      "d_anfang": round(n["source_start"] - s, 3), "d_ende": round(n["source_end"] - e, 3)})
    date = json.loads((DATASET / "manifest.json").read_text())["created"][:10]
    cards = []
    for p in pairs:
        src_old = "../" + str((arch / "hoerprobe" / p["alt"]).relative_to(ROOT))
        src_new = "../dataset/" + p["neu"]
        a, b = (src_new, src_old) if p["a_ist_neu"] else (src_old, src_new)
        cards.append(f"""<div class="card" data-id="{p['paar']}">
<div class="top"><b>{html.escape(p['name'])}</b><span>{p['paar']}</span></div>
<div class="row"><span class="lbl">A</span><audio controls preload="none" src="{html.escape(a)}"></audio></div>
<div class="row"><span class="lbl">B</span><audio controls preload="none" src="{html.escape(b)}"></audio></div>
<div class="txt">{html.escape(p['text'])}</div>
<div class="row"><button class="v" data-v="a">A passt besser</button><button class="v" data-v="gleich">gleich</button>
<button class="v" data-v="b">B passt besser</button></div>
<input class="note" type="text" placeholder="Notiz (optional)"></div>""")
    page = PAGE_AB.replace("__N__", str(len(pairs))).replace("__DATE__", date).replace("__CARDS__", "\n".join(cards))
    page = page.replace("button.sel[data-v=passt]", "button.v.sel")
    (OUT / "nachpruefung.html").write_text(page, encoding="utf-8")
    with open(OUT / "nachpruefung.tsv", "w", encoding="utf-8") as fh:
        fh.write("paar\talt\tneu\ta_ist_neu\td_anfang\td_ende\n")
        for p in pairs:
            fh.write(f"{p['paar']}\t{p['alt']}\t{p['neu']}\t{int(p['a_ist_neu'])}\t{p['d_anfang']}\t{p['d_ende']}\n")
    print(f"{len(pairs)} Paare -> samples/nachpruefung.html (+ .tsv); entfallen: {len(gone)} {gone}")


def compare_verdicts(path, tsv=None):
    """nachpruefung.csv + nachpruefung.tsv -> [(Paar-Zeile, Urteil, Notiz)] mit
    Urteil "neu besser" | "gleich" | "alt besser" (A/B zurueckuebersetzt)."""
    tsv = Path(tsv or Path(path).with_name("nachpruefung.tsv"))
    pairs = {r["paar"]: r for r in csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t")}
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8-sig", newline=""), delimiter=";"):
        p = pairs.get(r["paar"])
        if not p or r["urteil"] == "offen":
            continue
        new_is_a = p["a_ist_neu"] == "1"
        verdict = {"gleich": "gleich", "a": "neu besser" if new_is_a else "alt besser",
                   "b": "alt besser" if new_is_a else "neu besser"}[r["urteil"]]
        out.append((p, verdict, r.get("notiz", "")))
    return out


def edge_kind(p):
    """Welche Kante hat sich im Paar geaendert: nur Anfang | nur Ende | beide."""
    a, e = abs(float(p["d_anfang"])) > 0.005, abs(float(p["d_ende"])) > 0.005
    return "beide" if a and e else ("nur Anfang" if a else "nur Ende")


def summary_compare(path, tsv=None, before_csv=None):
    """Auswertung der Nachpruefung: getrennt nach dem frueheren Urteil und nach
    der geaenderten Kante."""
    arch = Path(before_csv) if before_csv else ROOT / "archive" / "v2.1_2026-09-28" / "hoerprobe" / "bewertung.csv"
    before = {v["file_name"]: v for v in read_votes(arch)} if arch.exists() else {}
    res = compare_verdicts(path, tsv)
    kinds = ("neu besser", "gleich", "alt besser")
    for title, key in (("vorher", lambda p: before.get(p["alt"], {}).get("urteil", "?")), ("Kante", edge_kind)):
        c = collections.Counter((key(p), v) for p, v, _ in res)
        for g in sorted({k[0] for k in c}):
            tot = sum(n for k, n in c.items() if k[0] == g)
            print(f"{title} {g} ({tot}): " + ", ".join(f"{v} {c[(g, v)]}" for v in kinds))
    notes = [f"  {p['paar']} ({v}): {n}" for p, v, n in res if n]
    if notes:
        print("Notizen:"); print("\n".join(notes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gleichverteilt", action="store_true",
                    help="jedes Segment gleich wahrscheinlich statt nach Dauer")
    ap.add_argument("--mit-stufe", action="store_true", help="Guetestufe auf dem Pruefblatt zeigen")
    ap.add_argument("--summary", default=None, help="heruntergeladene bewertung.csv auswerten")
    ap.add_argument("--notizen", action="store_true", help="bei --summary auch die Notizen ausgeben")
    ap.add_argument("--vergleich", default=None, help="archivierter Stand mit bewerteter Hoerprobe -> A/B-Nachpruefung")
    ap.add_argument("--summary-vergleich", default=None, help="heruntergeladene nachpruefung.csv auswerten")
    a = ap.parse_args()
    if a.summary:
        return summary(a.summary, a.notizen)
    if a.summary_vergleich:
        return summary_compare(a.summary_vergleich)
    if a.vergleich:
        return build_compare(a.vergleich)
    rows = [json.loads(l) for l in (DATASET / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    date = json.loads((DATASET / "manifest.json").read_text())["created"][:10]
    sel = pick(rows, a.n, a.seed, uniform=a.gleichverteilt)
    how = "zufällig, jedes Segment gleich wahrscheinlich" if a.gleichverteilt else "zufällig nach Dauer gezogen"
    build(sel, a.seed, date, how=how, show_tier=a.mit_stufe)
    print(f"{len(sel)} Proben -> samples/pruefliste.html (+ .tsv)")


if __name__ == "__main__":
    main()
