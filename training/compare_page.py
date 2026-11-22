#!/usr/bin/env python3
"""
Listening page: audio, transcript and the outputs of two runs side by side.

  python3 training/compare_page.py --a baseline-large-v3 --b large-v3-lora

Reads training/runs/<a|b>/predictions_test.csv and eval_test.json as well as
dataset/metadata.jsonl. Contains all segments of the test part; the default is
a fixed random selection of --per-interview segments per interview. What the
WER counts is marked: on the normalised words (whisper_common.words)
wrong, inserted or missing – not case and punctuation.
The audio is loaded from ../dataset/audio: open the page through the local
server (launch.json "hoerprobe"), not as a file.
-> samples/vergleich_<b>.html
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import ROOT, DATASET, RUNS, words


def token_words(text):
    """Tokens of the text, their normalised words and the token of each word."""
    toks = text.split()
    seq, owner = [], []
    for k, t in enumerate(toks):
        for w in words(t):
            seq.append(w)
            owner.append(k)
    return toks, seq, owner


def diff(ref, hyp):
    """Hypothesis as [text, kind, title] pieces: kind "" correct, "s" wrong
    (title = transcript), "i" inserted, "d" missing (text from the transcript),
    "f" filler (does not count). Plus the error count and reference words."""
    rt, rw, ro = token_words(ref)
    ht, hw, ho = token_words(hyp)
    kind, title, before = [""] * len(ht), {}, defaultdict(list)
    for k, t in enumerate(ht):
        if not words(t) and any(c.isalpha() for c in t):
            kind[k] = "f"
    for op in Levenshtein.opcodes(rw, hw):
        if op.tag == "replace":
            toks = sorted({ho[j] for j in range(op.dest_start, op.dest_end)})
            src = sorted({ro[i] for i in range(op.src_start, op.src_end)})
            for k in toks:
                kind[k] = "s"
            title[toks[0]] = " ".join(filter(None, [title.get(toks[0]), " ".join(rt[i] for i in src)]))
        elif op.tag == "insert":
            for j in range(op.dest_start, op.dest_end):
                if kind[ho[j]] != "s":
                    kind[ho[j]] = "i"
        elif op.tag == "delete":
            at = ho[op.dest_start] if op.dest_start < len(hw) else len(ht)
            before[at] += [ro[i] for i in range(op.src_start, op.src_end)]
    out = []
    for k in range(len(ht) + 1):
        if before.get(k):
            out.append([" ".join(rt[i] for i in dict.fromkeys(before[k])), "d", ""])
        if k < len(ht):
            out.append([ht[k], kind[k], title.get(k, "")] if title.get(k) else [ht[k], kind[k]])
    return out, Levenshtein.distance(rw, hw), len(rw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="baseline-large-v3", help="Lauf A (Ausgangsmodell)")
    ap.add_argument("--b", default="large-v3-lora", help="Lauf B (feingetunt)")
    ap.add_argument("--label-a", default="large-v3")
    ap.add_argument("--label-b", default="LoRA")
    ap.add_argument("--per-interview", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    meta = {}
    for line in (DATASET / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        meta[r["file_name"]] = r
    pa = {r["file_name"]: r for r in csv.DictReader(open(RUNS / a.a / "predictions_test.csv", encoding="utf-8"))}
    pb = {r["file_name"]: r for r in csv.DictReader(open(RUNS / a.b / "predictions_test.csv", encoding="utf-8"))}
    assert set(pa) == set(pb), "die Läufe haben verschiedene Segmente"
    ea, eb = (json.loads((RUNS / n / "eval_test.json").read_text(encoding="utf-8")) for n in (a.a, a.b))
    files = sorted(pa)
    rows, per_iv = [], defaultdict(lambda: [0, 0, 0, 0])
    for f in files:
        m = meta[f]
        da, ea_n, n = diff(m["text"], pa[f]["hypothese"])
        db, eb_n, _ = diff(m["text"], pb[f]["hypothese"])
        rows.append({"f": f, "s": m["speaker_name"], "p": m["project"], "iv": m["interview"],
                     "d": m["duration"], "r": m["text"], "a": da, "b": db, "ea": ea_n, "eb": eb_n, "n": n})
        t = per_iv[m["interview"]]
        t[0] += 1; t[1] += ea_n; t[2] += eb_n; t[3] += n
    rng = np.random.default_rng(a.seed)
    by_iv = defaultdict(list)
    for k, r in enumerate(rows):
        by_iv[r["iv"]].append(k)
    for ks in by_iv.values():
        for k in rng.choice(ks, min(a.per_interview, len(ks)), replace=False):
            rows[k]["k"] = 1
    interviews = [{"iv": iv, "s": rows[by_iv[iv][0]]["s"], "p": rows[by_iv[iv][0]]["p"], "n": t[0],
                   "wa": t[1] / max(1, t[3]), "wb": t[2] / max(1, t[3])}
                  for iv, t in sorted(per_iv.items(), key=lambda x: -x[1][1] / max(1, x[1][3]))]
    info = {"a": a.label_a, "b": a.label_b, "run_a": a.a, "run_b": a.b, "model": ea["model"],
            "seed": a.seed, "per_iv": a.per_interview,
            "wer_a": ea["gesamt"]["wer"], "wer_b": eb["gesamt"]["wer"],
            "proj": [[p, ea["je_projekt"][p]["wer"], eb["je_projekt"][p]["wer"]] for p in ea["je_projekt"]]}
    data = json.dumps({"info": info, "iv": interviews, "rows": rows}, ensure_ascii=False, separators=(",", ":"))
    out = Path(a.out) if a.out else ROOT / "samples" / f"vergleich_{a.b}.html"
    out.write_text(PAGE.replace("__DATA__", data.replace("</", "<\\/")), encoding="utf-8")
    better = sum(r["eb"] < r["ea"] for r in rows); worse = sum(r["eb"] > r["ea"] for r in rows)
    print(f"-> {out} ({out.stat().st_size/1e6:.1f} MB): {len(rows)} Segmente, Auswahl "
          f"{sum(r.get('k', 0) for r in rows)}; {a.label_b} besser {better}, schlechter {worse}")


PAGE = r"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Whisper-Vergleich</title>
<style>
:root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e2e0dc;--card:#fff;--ok:#2d6a4f;--bad:#b3261e;--acc:#7a3b2e;--chip:#f3f1ee;
--sub-bg:#fbe3e1;--sub-fg:#8c1d18;--ins-bg:#fdf0d5;--ins-fg:#7a5200;--del:#9b9b9b;--good-bg:#e3f1e8}
@media (prefers-color-scheme:dark){:root{--bg:#17181a;--fg:#e8e6e3;--mut:#9b9b9b;--line:#2e3033;--card:#1f2124;--ok:#74c69d;--bad:#f2b8b5;--acc:#d99a86;--chip:#26282b;
--sub-bg:#4a2523;--sub-fg:#f2b8b5;--ins-bg:#45391d;--ins-fg:#ecc979;--del:#7d7d7d;--good-bg:#1f3a2b}}
*{box-sizing:border-box}
body{margin:0;padding:24px 16px 80px;background:var(--bg);color:var(--fg);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:1.4rem;margin:0 0 4px}.sub{color:var(--mut);font-size:.9rem;margin-bottom:14px}
.help{font-size:.88rem;color:var(--mut);background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin-bottom:12px}
.help summary{cursor:pointer;color:var(--fg);font-weight:600}
.help p{margin:6px 0}
table{border-collapse:collapse;font-size:.85rem;margin:6px 0;width:100%}
th,td{text-align:left;padding:3px 8px;border-bottom:1px solid var(--line)}th{color:var(--mut);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0 10px;border-bottom:1px solid var(--line);margin-bottom:14px;display:flex;gap:8px 14px;align-items:center;flex-wrap:wrap;z-index:2;font-size:.88rem}
.bar label{display:flex;gap:6px;align-items:center}
select{font:inherit;font-size:.85rem;padding:3px 6px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);max-width:100%}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{border:0;border-radius:0;border-right:1px solid var(--line)}.seg button:last-child{border-right:0}
button{font:inherit;font-size:.85rem;padding:4px 10px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer}
button[aria-pressed=true]{background:var(--fg);color:var(--bg)}
#count{color:var(--mut);width:100%}
.card{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:10px;padding:12px 16px;margin-bottom:12px}
.card.better{border-left-color:var(--ok)}.card.worse{border-left-color:var(--bad)}
.top{font-size:.8rem;color:var(--mut);display:flex;gap:4px 10px;flex-wrap:wrap;margin-bottom:8px}.top b{color:var(--fg)}
audio{width:100%;height:34px;margin-bottom:4px}
.line{display:grid;grid-template-columns:86px 1fr auto;gap:10px;padding:6px 0;border-top:1px solid var(--line);align-items:baseline}
audio + .line{border-top:0}
.who{font-size:.78rem;color:var(--mut);font-weight:600}
.wer{font-size:.78rem;color:var(--mut);font-variant-numeric:tabular-nums;white-space:nowrap}
.wer.win{color:var(--ok);font-weight:600}
.ref .txt{border-left:3px solid var(--acc);padding-left:10px}
.t span{border-radius:3px}
.mark .s{background:var(--sub-bg);color:var(--sub-fg);padding:0 2px;cursor:help}
.mark .i{background:var(--ins-bg);color:var(--ins-fg);padding:0 2px}
.mark .d{color:var(--del);text-decoration:line-through;font-size:.92em}
#list:not(.mark) .d{display:none}
.f{color:var(--mut)}
.legend span{padding:0 4px;border-radius:3px}
.more{display:block;margin:6px auto 0}
@media (max-width:560px){.line{grid-template-columns:1fr auto}.who{grid-column:1/-1;margin-bottom:-6px}.bar{position:static}}
</style></head><body><div class="wrap">
<h1 id="h1"></h1>
<div class="sub" id="sub"></div>
<details class="help"><summary>Was wird gezeigt?</summary>
<p>Je Segment das Audio, das <b>Transkript</b> (Referenz) und was die beiden Modelle daraus machen –
Greedy-Dekodierung, Deutsch, ohne Zeitmarken, wie in <code>training/whisper_eval.py</code>.
Alle Segmente stammen aus dem Test-Teil; die Personen waren nicht im Training.</p>
<p class="legend">Markiert wird, was die WER zählt – auf normalisierten Wörtern (klein, ohne Satzzeichen,
Zahlen ausgeschrieben, ohne Füllwörter): <span class="mark"><span class="t"><span class="s">falsch</span></span></span>
(Mauszeiger zeigt das Transkript), <span class="mark"><span class="t"><span class="i">zusätzlich</span></span></span>,
<span class="mark"><span class="t"><span class="d">fehlt</span></span></span> (aus dem Transkript),
<span class="f">äh</span> Füllwort, zählt nicht. Groß- und Kleinschreibung und Satzzeichen zählen nicht.</p>
<p>Das Transkript ist selbst nicht fehlerfrei; ein „Fehler“ des Modells kann auch eine Abweichung des Transkripts vom Gesprochenen sein.
Rechnitz und Zwangsarbeiter sind im Test-Teil mit je nur einem Interview vertreten.</p>
<table id="ivtab"></table></details>
<div class="bar">
<span class="seg" id="selSet"><button data-v="probe">Stichprobe</button><button data-v="alle">alle</button></span>
<span class="seg" id="selRes"><button data-v="alle">alle</button><button data-v="besser"></button><button data-v="schlechter"></button><button data-v="gleich">gleich</button></span>
<label>Interview <select id="selIv"></select></label>
<label>Sortierung <select id="selSort"><option value="folge">Interview, Zeit</option><option value="gewinn">größte Verbesserung</option><option value="verlust">größte Verschlechterung</option></select></label>
<label><input type="checkbox" id="mark" checked> Unterschiede markieren</label>
<span id="count"></span></div>
<div id="list"></div>
<button class="more" id="more">mehr zeigen</button>
</div>
<script>
const D=__DATA__;
const I=D.info,R=D.rows,P={rs:"Roter Stern",re:"Rechnitz",za:"Zwangsarbeiter"};
const pct=x=>(100*x).toFixed(1).replace(".",",")+" %";
const st={set:"probe",res:"alle",iv:"alle",sort:"folge",mark:true,limit:40};
try{Object.assign(st,JSON.parse(localStorage.getItem("vergleich:"+I.run_b)||"{}"),{limit:40})}catch(e){}
document.getElementById("h1").textContent=`${I.model.replace("openai/","")}: Ausgangsmodell gegen ${I.b}`;
document.getElementById("sub").textContent=`Test-Teil, ${R.length.toLocaleString("de")} Segmente · WER gesamt ${pct(I.wer_a)} → ${pct(I.wer_b)} · `+
  I.proj.map(p=>`${p[0]} ${pct(p[1])} → ${pct(p[2])}`).join(" · ")+` · Stichprobe: ${I.per_iv} zufällige Segmente je Interview (seed ${I.seed})`;
const tab=document.getElementById("ivtab");
tab.innerHTML=`<tr><th>Interview</th><th>Projekt</th><th class="num">Segmente</th><th class="num">WER ${I.a}</th><th class="num">WER ${I.b}</th></tr>`;
for(const v of D.iv){const tr=tab.insertRow();[v.s,P[v.p]||v.p,v.n,pct(v.wa),pct(v.wb)].forEach((x,k)=>{const td=tr.insertCell();td.textContent=x;if(k>1)td.className="num"})}
const sel=document.getElementById("selIv");
sel.add(new Option("alle","alle"));for(const v of D.iv)sel.add(new Option(`${v.s} (${v.p}, ${v.n})`,v.iv));
document.querySelector('#selRes [data-v=besser]').textContent=`${I.b} besser`;
document.querySelector('#selRes [data-v=schlechter]').textContent=`${I.b} schlechter`;
function chunks(el,cs){for(const c of cs){if(el.childNodes.length)el.append(" ");const s=document.createElement("span");s.textContent=c[0];
  if(c[1])s.className=c[1];if(c[2])s.title="Transkript: "+c[2];else if(c[1]==="i")s.title="nicht im Transkript";else if(c[1]==="d")s.title="fehlt";else if(c[1]==="f")s.title="Füllwort, zählt nicht";el.append(s)}}
function line(label,cls,fill,wer,win){const d=document.createElement("div");d.className="line "+cls;
  const a=document.createElement("div");a.className="who";a.textContent=label;
  const t=document.createElement("div");t.className="t txt";fill(t);
  const w=document.createElement("div");w.className="wer"+(win?" win":"");w.textContent=wer;d.append(a,t,w);return d}
function card(r){const c=document.createElement("div");c.className="card"+(r.eb<r.ea?" better":r.eb>r.ea?" worse":"");
  const top=document.createElement("div");top.className="top";
  const b=document.createElement("b");b.textContent=r.s;top.append(b);
  for(const x of [P[r.p]||r.p,r.d.toFixed(1).replace(".",",")+" s",r.f.split("/").pop()]){const s=document.createElement("span");s.textContent=x;top.append(s)}
  const au=document.createElement("audio");au.controls=true;au.preload="none";au.src="../dataset/"+r.f;
  const n=Math.max(1,r.n),wa=r.ea/n,wb=r.eb/n;
  c.append(top,au,
    line("Transkript","ref",t=>t.textContent=r.r,`${r.n} Wörter`),
    line(I.a,"hyp",t=>chunks(t,r.a),"WER "+pct(wa),false),
    line(I.b,"hyp",t=>chunks(t,r.b),"WER "+pct(wb),wb<wa));
  return c}
function filtered(){let xs=R.filter(r=>(st.set==="alle"||r.k)&&(st.iv==="alle"||r.iv===st.iv)&&
  (st.res==="alle"||(st.res==="besser"?r.eb<r.ea:st.res==="schlechter"?r.eb>r.ea:r.eb===r.ea)));
  const g=r=>(r.ea-r.eb)/Math.max(1,r.n);
  if(st.sort==="gewinn")xs=xs.slice().sort((x,y)=>g(y)-g(x)||y.n-x.n);
  if(st.sort==="verlust")xs=xs.slice().sort((x,y)=>g(x)-g(y)||y.n-x.n);
  return xs}
function render(){const xs=filtered(),list=document.getElementById("list");list.replaceChildren();
  xs.slice(0,st.limit).forEach(r=>list.append(card(r)));
  const bt=xs.filter(r=>r.eb<r.ea).length,wo=xs.filter(r=>r.eb>r.ea).length;
  document.getElementById("count").textContent=`${xs.length} Segmente · ${I.b} besser ${bt}, schlechter ${wo}, gleich ${xs.length-bt-wo}`+(xs.length>st.limit?` · gezeigt ${st.limit}`:"");
  document.getElementById("more").hidden=xs.length<=st.limit;
  document.querySelectorAll(".seg button").forEach(b=>b.setAttribute("aria-pressed",b.parentNode.id==="selSet"?st.set===b.dataset.v:st.res===b.dataset.v));
  sel.value=st.iv;document.getElementById("selSort").value=st.sort;document.getElementById("mark").checked=st.mark;
  document.getElementById("list").classList.toggle("mark",st.mark);
  try{localStorage.setItem("vergleich:"+I.run_b,JSON.stringify({set:st.set,res:st.res,iv:st.iv,sort:st.sort,mark:st.mark}))}catch(e){}}
document.querySelectorAll("#selSet button").forEach(b=>b.onclick=()=>{st.set=b.dataset.v;st.limit=40;render()});
document.querySelectorAll("#selRes button").forEach(b=>b.onclick=()=>{st.res=b.dataset.v;st.limit=40;render()});
sel.onchange=e=>{st.iv=e.target.value;st.limit=40;render()};
document.getElementById("selSort").onchange=e=>{st.sort=e.target.value;st.limit=40;render()};
document.getElementById("mark").onchange=e=>{st.mark=e.target.checked;render()};
document.getElementById("more").onclick=()=>{st.limit+=40;render()};
document.addEventListener("play",e=>{document.querySelectorAll("audio").forEach(a=>{if(a!==e.target)a.pause()})},true);
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
