#!/usr/bin/env python3
"""
Prüfseite für die spaCy-Entitäten (lokal, ohne Audio). Am besten über den lokalen
Server öffnen (python3 -m http.server 8765, dann
http://localhost:8765/training/entities/pruefen_test.html), damit der Browser die
Entscheidungen sicher speichert; als Datei geöffnet klappt das nicht in jedem Browser.

  python3 training/entities_review.py --split test
  -> training/entities/pruefen_<split>.html

Je verschiedener Schreibung eine Zeile mit Vorschlag, Zahl der Nennungen und
bis zu drei Stellen im Transkript. Entscheidung je Zeile: stimmt / anderer Typ /
kein Name / Grenze falsch; Tastatur: Enter stimmt, 1-5 Typ, 0 kein Name,
9 Grenze falsch, Pfeile oder j/k blättern, u zurücksetzen. Die Entscheidungen
bleiben im Browser gespeichert; "CSV speichern" lädt
entitaeten_<split>_geprueft.csv herunter - diese Datei liest entity_eval.py.
Nur Standardbibliothek; gibt nur Zahlen aus.
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENT = ROOT / "training" / "entities"
ORDER = ["PER", "LOC", "ORG", "DATUM", "?", "MISC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    text = {}
    for line in (ROOT / "dataset" / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["split"] == a.split:
            text[r["file_name"]] = r["text"]
    groups = defaultdict(lambda: {"labs": Counter(), "ctx": []})
    for line in (ENT / f"{a.split}_spacy.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        t = text[r["file_name"]]
        for e in r["entities"]:
            g = groups[e["text"]]
            g["labs"][e["label"]] += 1
            if len(g["ctx"]) < 3:
                s, x = e["start"], e["end"]
                g["ctx"].append([("…" if s > 70 else "") + t[max(0, s - 70):s], t[s:x],
                                 t[x:x + 70] + ("…" if x + 70 < len(t) else ""), r["interview"]])
    entries = []
    for t, g in groups.items():
        lab = g["labs"].most_common(1)[0][0]
        entries.append({"t": t, "lab": lab, "labs": dict(g["labs"]), "n": sum(g["labs"].values()), "ctx": g["ctx"]})
    entries.sort(key=lambda e: (ORDER.index(e["lab"]) if e["lab"] in ORDER else 9, -e["n"], e["t"].lower()))
    data = json.dumps({"split": a.split, "entries": entries}, ensure_ascii=False, separators=(",", ":"))
    out = ENT / f"pruefen_{a.split}.html"
    out.write_text(PAGE.replace("__DATA__", data.replace("</", "<\\/")), encoding="utf-8")
    c = Counter(e["lab"] for e in entries)
    print(f"-> {out}: {len(entries)} Schreibungen ({', '.join(f'{k} {c[k]}' for k in ORDER if c[k])}), "
          f"{sum(e['n'] for e in entries)} Nennungen")


PAGE = r"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Entitäten prüfen</title>
<style>
:root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e2e0dc;--card:#fff;--ok:#2d6a4f;--bad:#b3261e;--warn:#9a6a00;--acc:#7a3b2e;--chip:#f3f1ee;--cur:#fff6dc;--mark:#fde7a6}
@media (prefers-color-scheme:dark){:root{--bg:#17181a;--fg:#e8e6e3;--mut:#9b9b9b;--line:#2e3033;--card:#1f2124;--ok:#74c69d;--bad:#f2b8b5;--warn:#e0b050;--acc:#d99a86;--chip:#26282b;--cur:#2f2a1c;--mark:#5a4a14}}
*{box-sizing:border-box}
body{margin:0;padding:20px 16px 80px;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{max-width:960px;margin:0 auto}
h1{font-size:1.35rem;margin:0 0 4px}.sub{color:var(--mut);font-size:.88rem;margin-bottom:10px}
.help{font-size:.86rem;color:var(--mut);background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 14px;margin-bottom:10px}
.help summary{cursor:pointer;color:var(--fg);font-weight:600}.help p{margin:6px 0}
kbd{font:inherit;font-size:.8rem;border:1px solid var(--line);border-radius:4px;padding:0 4px;background:var(--chip);color:var(--fg)}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);margin-bottom:10px;display:flex;gap:8px 14px;align-items:center;flex-wrap:wrap;z-index:2;font-size:.88rem}
select,button{font:inherit;font-size:.84rem;padding:3px 8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.row{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:8px;padding:8px 12px;margin-bottom:6px}
.row.cur{background:var(--cur);outline:2px solid var(--acc)}
.row[data-d=stimmt],.row[data-d=typ]{border-left-color:var(--ok)}.row[data-d=kein]{border-left-color:var(--bad)}.row[data-d=grenze]{border-left-color:var(--warn)}
body.open .row[data-d]:not([data-d=""]){display:none}
.head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.ent{font-weight:600;font-size:1.02rem}.meta{color:var(--mut);font-size:.8rem}
.ctx{font-size:.86rem;color:var(--mut);margin:3px 0 0 0;padding-left:10px;border-left:2px solid var(--line)}
.ctx mark{background:var(--mark);color:var(--fg);border-radius:3px;padding:0 2px}
.btns{display:flex;gap:4px;flex-wrap:wrap;margin-top:6px}
.btns button{font-size:.78rem;padding:2px 8px;border-radius:12px;background:var(--chip)}
.btns button[aria-pressed=true]{background:var(--fg);color:var(--bg)}
.btns button.no[aria-pressed=true]{background:var(--bad);color:#fff;border-color:var(--bad)}
.btns button.ok[aria-pressed=true]{background:var(--ok);color:#fff;border-color:var(--ok)}
</style></head><body><div class="wrap">
<h1>Entitäten prüfen</h1>
<div class="sub" id="sub"></div>
<details class="help" open><summary>So wird geprüft</summary>
<p>Je Zeile eine Schreibung, die spaCy im Transkript als Name erkannt hat, mit Vorschlag und bis zu drei Stellen.
Gewertet werden später nur bestätigte Zeilen – aufhören ist jederzeit möglich; die häufigsten stehen oben.</p>
<p><b>stimmt</b> <kbd>Enter</kbd> – Name, Typ passt. <b>Typ</b> <kbd>1</kbd> Person <kbd>2</kbd> Ort <kbd>3</kbd> Organisation
(auch Behörde, Partei, Firma, Truppe) <kbd>4</kbd> Sonstiges (Ereignis, Werk, Volksgruppe …) <kbd>5</kbd> Datum.
<b>kein Name</b> <kbd>0</kbd> – gewöhnliches Wort. <b>Grenze falsch</b> <kbd>9</kbd> – Name, aber zu viel oder zu wenig
markiert (wird nicht gewertet). Blättern mit <kbd>↓</kbd><kbd>↑</kbd> oder <kbd>j</kbd><kbd>k</kbd>, zurücksetzen mit <kbd>u</kbd>.</p>
<p><b>Regeln:</b> Vorname allein, Nachname, Spitzname, voller Name → Person. Verwandtschafts- und Anredewörter allein
(Sohn, Tochter, Mutter, Mama, Onkel) → kein Name, auch wenn eine bestimmte Person gemeint ist. Titel oder Verwandtschaftswort
mit Name (Onkel Franz, Frau Muster, Doktor Beispiel) → Person. Vorname in einem Ortsnamen (Maria in Mariatrost) → Ort, nach Zusammenhang.
Volksgruppen und Nationalitäten als Hauptwort (die Russen, Engländer) → Sonstiges. Adjektive (russisch, deutsch), Feiertage,
„der Krieg“, die Anrede „Sie“ → kein Name. Heer, Partei, Behörde, Firma (SS, Wehrmacht, NSDAP, Gendarmerie, Puch) → Organisation.
Straße, Gebäude, Bezirk, Dorf (Stubenberggasse, Schloss Rechnitz, Thondorf) → Ort.</p>
<p>Gilt für alle Nennungen dieser Schreibung. Die Entscheidungen bleiben in diesem Browser gespeichert; zum Schluss
„CSV speichern“ und die Datei nach <code>training/entities/</code> legen.</p></details>
<div class="bar"><label>Typ <select id="f"></select></label><label><input type="checkbox" id="open"> nur offene</label>
<span id="prog"></span><button id="csv">CSV speichern</button></div>
<div id="list"></div>
</div>
<script>
const D=__DATA__,E=D.entries,KEY="entitaeten:"+D.split;
const NAMES={PER:"Person",LOC:"Ort",ORG:"Organisation",MISC:"Sonstiges",DATUM:"Datum","?":"unklar"};
const TYPES=["PER","LOC","ORG","MISC","DATUM"];
let st={};try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}};
const f=document.getElementById("f");f.add(new Option("alle","alle"));
for(const k of ["PER","LOC","ORG","DATUM","?","MISC"]){const n=E.filter(e=>e.lab===k).length;if(n)f.add(new Option(`${NAMES[k]} (${n})`,k))}
document.getElementById("sub").textContent=`${D.split}-Teil · ${E.length} Schreibungen · ${E.reduce((s,e)=>s+e.n,0)} Nennungen · Vorschläge von spaCy (de_core_news_lg), Jahreszahlen per Regel`;
const list=document.getElementById("list");let rows=[],cur=0;
function decide(e,d,typ){if(!d){delete st[e.t]}else{st[e.t]={d,typ:typ||e.lab}}save()}
function paint(r,e){const s=st[e.t]||{};r.dataset.d=s.d||"";
  r.querySelectorAll("button").forEach(b=>{const on=b.dataset.d===s.d&&(b.dataset.d!=="typ"||b.dataset.typ===s.typ);b.setAttribute("aria-pressed",on)})}
function build(){list.replaceChildren();rows=[];
  for(const e of E){if(f.value!=="alle"&&e.lab!==f.value)continue;
    const r=document.createElement("div");r.className="row";r.tabIndex=-1;
    const h=document.createElement("div");h.className="head";
    const b=document.createElement("span");b.className="ent";b.textContent=e.t;
    const m=document.createElement("span");m.className="meta";
    m.textContent=`Vorschlag ${NAMES[e.lab]||e.lab} · ${e.n}×`+(Object.keys(e.labs).length>1?" · auch "+Object.entries(e.labs).filter(([k])=>k!==e.lab).map(([k,v])=>`${NAMES[k]||k} ${v}×`).join(", "):"");
    h.append(b,m);r.append(h);
    for(const c of e.ctx){const p=document.createElement("div");p.className="ctx";const mk=document.createElement("mark");mk.textContent=c[1];
      const iv=document.createElement("span");iv.className="meta";iv.textContent="  · "+c[3];p.append(c[0],mk,c[2],iv);r.append(p)}
    const bs=document.createElement("div");bs.className="btns";
    const mkb=(label,d,typ,cls)=>{const x=document.createElement("button");x.textContent=label;x.dataset.d=d;if(typ)x.dataset.typ=typ;if(cls)x.className=cls;
      x.onclick=()=>{const s=st[e.t];decide(e,s&&s.d===d&&(!typ||s.typ===typ)?null:d,typ);paint(r,e);prog()};bs.append(x)};
    if(e.lab!=="?")mkb("✓ stimmt","stimmt",e.lab,"ok");
    TYPES.forEach((t,k)=>{if(t!==e.lab)mkb(`${k===3?4:k===4?5:k+1} ${NAMES[t]}`,"typ",t)});
    mkb("0 kein Name","kein",null,"no");mkb("9 Grenze falsch","grenze",null);
    r.append(bs);r.onclick=ev=>{if(ev.target.tagName!=="BUTTON"){cur=rows.indexOf(r);focus()}};
    list.append(r);rows.push(r);r._e=e;paint(r,e)}
  cur=Math.min(cur,Math.max(0,rows.length-1));focus(false);prog()}
function vis(){return rows.filter(r=>r.offsetParent!==null)}
function focus(scroll=true){rows.forEach(r=>r.classList.remove("cur"));const r=rows[cur];if(!r)return;r.classList.add("cur");if(scroll)r.scrollIntoView({block:"center"})}
function step(dir){const v=vis();if(!v.length)return;let i=v.indexOf(rows[cur]);i=i<0?0:Math.min(v.length-1,Math.max(0,i+dir));cur=rows.indexOf(v[i]);focus()}
function prog(){const done=E.filter(e=>st[e.t]),men=done.reduce((s,e)=>s+e.n,0);
  document.getElementById("prog").textContent=`${done.length} von ${E.length} geprüft (${men} von ${E.reduce((s,e)=>s+e.n,0)} Nennungen)`}
document.addEventListener("keydown",ev=>{if(ev.target.tagName==="SELECT"||ev.metaKey||ev.ctrlKey)return;const r=rows[cur];if(!r)return;const e=r._e;
  const map={"1":"PER","2":"LOC","3":"ORG","4":"MISC","5":"DATUM"};let did=true;
  if(ev.key==="Enter"&&e.lab!=="?")decide(e,"stimmt",e.lab);else if(map[ev.key])decide(e,map[ev.key]===e.lab?"stimmt":"typ",map[ev.key]);
  else if(ev.key==="0")decide(e,"kein");else if(ev.key==="9")decide(e,"grenze");else if(ev.key==="u")decide(e,null);
  else if(ev.key==="ArrowDown"||ev.key==="j"){step(1);ev.preventDefault();return}else if(ev.key==="ArrowUp"||ev.key==="k"){step(-1);ev.preventDefault();return}
  else did=false;
  if(did){ev.preventDefault();paint(r,e);prog();if(ev.key!=="u")step(1)}});
f.onchange=()=>{cur=0;build()};
document.getElementById("open").onchange=ev=>{document.body.classList.toggle("open",ev.target.checked);cur=0;step(0)};
document.getElementById("csv").onclick=()=>{const q=x=>'"'+String(x).replace(/"/g,'""')+'"';
  const lines=["text;vorschlag;entscheidung;typ;nennungen"];
  for(const e of E){const s=st[e.t];lines.push([q(e.t),e.lab,s?s.d:"offen",s&&(s.d==="stimmt"||s.d==="typ")?s.typ:"",e.n].join(";"))}
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([lines.join("\n")+"\n"],{type:"text/csv"}));
  a.download=`entitaeten_${D.split}_geprueft.csv`;a.click()};
build();
</script></body></html>
"""

if __name__ == "__main__":
    main()
