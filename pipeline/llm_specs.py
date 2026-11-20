#!/usr/bin/env python3
"""
Stufe 0c: Specs von einem Sprachmodell schreiben lassen – Qwen3-235B über eine Qwen-API (16.11.2026).

Eine Spec legt fest, was der Parser nicht raten soll: wer welches Kürzel hat (Name, Rolle), welche
Schreibvarianten eines Kürzels es gibt, wo das Gespräch beginnt und endet und welche Zeilen kein Gespräch
sind (Seitenköpfe, Teilüberschriften). Diese Entscheidungen trifft Qwen3-235B aus der Strukturübersicht
(inventory.py) plus Anfang und Ende des Transkripts mit Zeilennummern.

Das Modell gibt nie Transkripttext aus, nur diese Entscheidungen (JSON, per Schema erzwungen, Temperatur 0).
Jede Antwort wird sofort mit dem Parser geprüft (parse_transcripts.parse, inkl. Wortprüfung: jedes Wort der
Ausgabe steht in derselben Reihenfolge im Quelltext). Meldet der Parser einen Fehler oder bleiben Wörter
ohne bekannten Sprecher, bekommt das Modell die Meldung zurück und darf bis zu --retries Mal korrigieren.

Nicht vom Modell, sondern aus der lokalen Projektkonfiguration (config/interviews.json): Dateien und
Audioteile, Format-Profil (pipeline/specs/_profiles.json), Einwilligung, Ausschluss. Anmerkungen in Klammern
klassifiziert weiter notes.py nach festen Regeln.

Anbindung: jede OpenAI-kompatible Chat-Completions-API, die Qwen3-235B-A22B-Instruct-2507 anbietet
(gehostet oder selbst betrieben, z. B. mit vLLM), über Umgebungsvariablen:
  QWEN_API_BASE   Basis-URL bis einschließlich /v1 (Pflicht)
  QWEN_API_KEY    Schlüssel, falls der Endpunkt einen verlangt
  QWEN_MODEL      Modellname beim Anbieter (Standard: qwen3-235b-a22b-instruct-2507)

  python3 pipeline/llm_specs.py                      # alle Interviews ohne gültige Spec
  python3 pipeline/llm_specs.py --only re_,rs_a --force
  python3 pipeline/llm_specs.py --compare <ordner>/turns     # gegen Turns aus anderen Specs (nur Zahlen)

-> pipeline/specs/<id>.json, work/llm_specs/<id>.json (Antworten, Versuche, Parser-Meldungen),
   work/llm_specs/_vergleich.json (mit --compare). Antworten werden zwischengespeichert (work/llm_cache/).
"""
import argparse, collections, difflib, hashlib, json, os, re, sys, time
import urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORK, SPECS, resolve, slug, write_json
from sources import read_lines
from inventory import inventory, mask

CONFIG = ROOT / "config" / "interviews.json"
SPLIT_SRC = ROOT / "config" / "split.json"
OUT = WORK / "llm_specs"
CACHE = WORK / "llm_cache"
MODEL = os.environ.get("QWEN_MODEL", "qwen3-235b-a22b-instruct-2507")
HEAD, TAIL = 150, 40          # Zeilen vom Anfang/Ende des Transkripts, die das Modell wörtlich sieht

ROLES = ["interviewer", "zeitzeuge", "weitere"]
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["interviewee", "interviewer", "date", "speakers", "tag_aliases", "body_start", "body_end",
                 "drop_lines", "gap_lines", "tag_regex", "paragraph_order"],
    "properties": {
        "interviewee": {"type": "string"},
        "interviewer": {"type": "string"},
        "date": {"type": ["string", "null"]},
        "speakers": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["tag", "name", "role"],
            "properties": {"tag": {"type": "string"}, "name": {"type": "string"},
                           "role": {"type": "string", "enum": ROLES}}}},
        "tag_aliases": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["variant", "tag"],
            "properties": {"variant": {"type": "string"}, "tag": {"type": "string"}}}},
        "body_start": {"type": ["string", "null"]},
        "body_end": {"type": ["string", "null"]},
        "drop_lines": {"type": "array", "items": {"type": "string"}},
        "gap_lines": {"type": "array", "items": {"type": "string"}},
        "tag_regex": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["regex", "tag"],
            "properties": {"regex": {"type": "string"}, "tag": {"type": "string"}}}},
        "paragraph_order": {"type": "array", "items": {"type": "string"}},
    }}

SYSTEM = """Du bereitest Transkripte von Oral-History-Interviews für ein automatisches Text-Audio-Alignment vor.
Ein deterministischer Parser schneidet später das Gespräch heraus und ordnet jeden Absatz einem Sprecher zu.
Du lieferst nur die Entscheidungen, die der Parser dafür braucht, als JSON. Du schreibst nie Transkripttext
um und erfindest nichts: Kürzel, Zeilenanfänge und Namen übernimmst du zeichengenau aus der Vorlage."""

TASK = """Felder:
- speakers: jedes Sprecherkürzel, das im Gespräch vorkommt, genau so geschrieben wie am Zeilenanfang
  (ohne Doppelpunkt bzw. ohne eckige Klammern), mit vollem Namen (so weit aus Kopf/Text erkennbar, sonst
  das Kürzel) und Rolle: "interviewer" (stellt die Fragen, auch Co-Interviewer), "zeitzeuge" (interviewte
  Person, deren Erinnerungen erfragt werden – bei Paar- und Gruppeninterviews ALLE Interviewten), "weitere"
  (nur Personen, die bloß nebenbei mitsprechen, z. B. Angehörige, die einzelne Einwürfe machen).
- tag_aliases: Schreibvarianten eines Kürzels, die im Text vorkommen (andere Abkürzung, voller Name statt
  Kürzel, Tippfehler), jeweils mit dem Kürzel aus speakers. Punkte und Leerzeichen gleicht der Parser
  selbst aus ("A.K." = "A. K." = "AK."), dafür keine Varianten anlegen.
- body_start: null, wenn das Gespräch mit der ersten Zeile beginnt, die mit einem Kürzel aus speakers
  anfängt. Sonst der zeichengenaue Anfang (mindestens 15 Zeichen) der ersten Gesprächszeile.
- body_end: der zeichengenaue Anfang (mindestens 10 Zeichen) der ersten Zeile NACH dem Gespräch (z. B.
  Abspann, Anhang, Unterschriften, Biografie), oder null, wenn das Gespräch bis zum Ende geht.
- drop_lines: zeichengenaue Anfänge von Zeilen innerhalb des Gesprächs, die kein gesprochener Text sind und
  ohne Lücke entfallen können (wiederkehrende Seitenköpfe/-füße, Dateinamen, Seitenzahlen-Zeilen).
- gap_lines: zeichengenaue Anfänge von Zeilen innerhalb des Gesprächs, an denen gesprochene Sprache fehlt
  (z. B. "Teil 2", "Ende Band 1", Zusammenfassungen statt Wortlaut).
- tag_regex: nur nötig, wenn ein Sprecherwechsel nicht als "Kürzel:" oder "[Kürzel]" am Zeilenanfang steht
  (z. B. "TB; Text"); dann ein Python-Regex, der am Zeilenanfang das Kürzel samt Trennzeichen erfasst,
  und das Kürzel aus speakers. Sonst leer.
- paragraph_order: nur wenn das Transkript gar keine Kürzel hat und die Absätze abwechselnd gesprochen
  werden: die Kürzel in der Reihenfolge des ersten Absatzes, z. B. ["I", "ZZ"]. Sonst leer.
- interviewee, interviewer: Namen laut Kopf (sonst laut speakers). date: Datum des Interviews als
  JJJJ-MM-TT, falls angegeben, sonst null.
"""


# ---------------------------------------------------------------- Qwen-API, mit Zwischenspeicher

def chat(body, attempts=8):
    """POST /chat/completions (OpenAI-kompatibel); wiederholt bei Überlast und Netzfehlern."""
    base = os.environ.get("QWEN_API_BASE")
    if not base:
        sys.exit("QWEN_API_BASE fehlt: Basis-URL einer OpenAI-kompatiblen Qwen-API (bis einschließlich /v1)")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("QWEN_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["QWEN_API_KEY"]
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    for k in range(attempts):
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in (408, 409, 429) and e.code < 500:
                raise RuntimeError(f"Qwen-API: HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        if k + 1 < attempts:
            time.sleep(min(60, 2 ** k))
    raise RuntimeError(f"Qwen-API nach {attempts} Versuchen nicht erreichbar")


def parse_json(raw):
    """JSON-Antwort; verträgt einen Denkblock oder Codezaun davor/danach (Endpunkte ohne Schema-Zwang)."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    i, j = raw.find("{"), raw.rfind("}")
    return json.loads(raw[i:j + 1] if i >= 0 and j > i else raw)


def ask(messages, model, usage, fmt="schema", max_tokens=4000):
    """Chat mit JSON-Schema; Antwort (dict) aus dem Zwischenspeicher, wenn genau diese Anfrage schon lief."""
    req = {"model": model, "system": SYSTEM, "messages": messages, "schema": SCHEMA,
           "temperature": 0, "max_tokens": max_tokens}
    if fmt != "schema":
        req["format"] = fmt
    key = json.dumps(req, ensure_ascii=False, sort_keys=True)
    f = CACHE / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if f.exists():
        usage["cache"] += 1
        return json.loads(f.read_text(encoding="utf-8"))["answer"]
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYSTEM}]
                        + [{"role": m["role"], "content": m["text"]} for m in messages]}
    if fmt == "schema":
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "spec", "strict": True, "schema": SCHEMA}}
    else:                                            # ohne Schema-Zwang: Schema in den Prompt
        if fmt == "object":
            body["response_format"] = {"type": "json_object"}
        body["messages"][-1]["content"] += "\n\nAntworte nur mit JSON nach diesem Schema:\n" + json.dumps(SCHEMA, ensure_ascii=False)
    t = time.time()
    r = chat(body)
    u = r.get("usage") or {}
    usage["calls"] += 1
    usage["in"] += u.get("prompt_tokens", 0)
    usage["out"] += u.get("completion_tokens", 0)
    usage["seconds"] += time.time() - t
    choice = r["choices"][0]
    if choice.get("finish_reason") == "length":
        usage["abgeschnitten"] += 1
    ans = parse_json(choice["message"]["content"] or "")
    write_json(f, {"answer": ans, "usage": u})
    return ans


# ---------------------------------------------------------------- Prompt und Spec

def numbered(lines, a, b):
    return "\n".join(f"{i:5}| {mask(lines[i].strip())[:300]}" for i in range(a, b) if lines[i].strip())


def prompt(meta):
    p = resolve(meta["transcript"]) if not meta["transcript"].startswith("work/") else ROOT / meta["transcript"]
    lines = read_lines(p)
    n = len(lines)
    inv = inventory({"id": meta["id"], "transcript": meta["transcript"], "audio": meta.get("audio", [])})
    head = numbered(lines, 0, min(HEAD, n))
    tail = numbered(lines, max(HEAD, n - TAIL), n) if n > HEAD else ""
    fmt = {"bracket": "Kürzel stehen in eckigen Klammern am Zeilenanfang: [Kürzel] Text",
           "none": "Das Transkript hat keine Sprecherkürzel."}.get(meta.get("tag_style"), "")
    return (f"{TASK}\nStrukturübersicht des Transkripts (Gesprächstext ausgelassen, Kontaktdaten maskiert):\n"
            f"{inv}\n\n{('Hinweis zum Format: ' + fmt) if fmt else ''}\n"
            f"Anfang des Transkripts (Zeilennummer| Zeile):\n{head}\n"
            + (f"\n[... Zeilen {HEAD}–{n - TAIL - 1} ausgelassen ...]\n\nEnde des Transkripts:\n{tail}\n" if tail else "")
            + "\nGib die Entscheidungen als JSON zurück.")


def source(meta, vocab):
    """Zeilen des Transkripts roh (wie im Prompt) und so, wie der Parser sie sieht (Ligaturen repariert)."""
    from parse_transcripts import PROFILES
    from sources import detect_glyphs, repair_glyphs
    p = resolve(meta["transcript"]) if not meta["transcript"].startswith("work/") else ROOT / meta["transcript"]
    raw = read_lines(p)
    fixed = raw
    if PROFILES[meta["profile"]].get("glyph_repair", False):
        fixed, _, _ = repair_glyphs(raw, vocab, detect_glyphs(raw, vocab))
    assert len(fixed) == len(raw)
    return raw, fixed


LINE_NO = re.compile(r"^\s*\d{1,5}\s*[:|]\s?")


def anchor(raw, fixed, value, start=0):
    """Zeilenanfang aus der Antwort -> (Präfix, wie ihn der Parser findet; Zeilenindex) oder (Wert, None).
    Gesucht wird in den rohen und den reparierten Zeilen, notfalls ohne eine mitkopierte Zeilennummer
    ("718: ..." aus der Übersicht). Zurück kommt der Anfang der reparierten Zeile, lang genug, dass ab
    `start` keine frühere Zeile denselben Anfang hat."""
    v = (value or "").strip()
    if not v:
        return None, None
    for cand in dict.fromkeys([v, LINE_NO.sub("", v).strip()]):
        if not cand:
            continue
        for i in range(start, len(raw)):
            if raw[i].strip().startswith(cand) or fixed[i].strip().startswith(cand):
                line = fixed[i].strip()
                for n in (len(cand), 40, 80, len(line)):
                    pre = line[:max(n, 1)]
                    first = next(j for j in range(start, len(fixed)) if fixed[j].strip().startswith(pre))
                    if first == i:
                        return pre, i
                return line, i
    for cand in dict.fromkeys([v, LINE_NO.sub("", v).strip()]):     # notfalls irgendwo in der Zeile (nach Kürzel)
        if len(cand) >= 12:
            for i in range(start, len(raw)):
                if cand in raw[i] or cand in fixed[i]:
                    line = fixed[i].strip()
                    for n in (40, 80, len(line)):
                        pre = line[:n]
                        if next(j for j in range(start, len(fixed)) if fixed[j].strip().startswith(pre)) == i:
                            return pre, i
                    return line, i
    return v, None


def build_spec(meta, ans, raw=None, fixed=None):
    """Antwort des Modells + Projektmetadaten -> Spec im Format des Parsers."""
    speakers = {}
    for s in ans["speakers"]:
        t = s["tag"].strip().strip("[]").rstrip(":").strip()
        if t and t not in speakers:
            name = s["name"].strip() or t
            # Personen-ID über Interviews hinweg nur bei vollem Namen (Vor- und Nachname); Bezeichnungen wie
            # "Ehefrau" oder ein bloßes Kürzel bleiben je Interview getrennt (sonst falsche Personengruppen)
            full = len(name.split()) >= 2 and re.match(r"^[A-ZÄÖÜ]", name) and name != t
            speakers[t] = {"name": name, "role": s["role"],
                           "person": "p_" + slug(name) if full else f"p_{meta['id']}_{slug(t) or 'x'}"}
    aliases = {a["variant"].strip().strip("[]").rstrip(":").strip(): a["tag"].strip().strip("[]").rstrip(":").strip()
               for a in ans["tag_aliases"] if a["variant"].strip()}
    aliases = {v: t for v, t in aliases.items() if t in speakers and v not in speakers}
    spec = {"id": meta["id"], "project": meta["project"], "status": "llm", "transcript": meta["transcript"],
            "audio": meta.get("audio", []), "profile": meta["profile"],
            "interviewee": ans["interviewee"], "interviewer": ans["interviewer"], "date": ans["date"],
            "speakers": speakers, "tag_aliases": aliases,
            "body_start": ans["body_start"] or None, "body_end": ans["body_end"] or None,
            "drop_lines": [x.strip() for x in ans["drop_lines"] if len(x.strip()) >= 3],
            "gap_lines": [x.strip() for x in ans["gap_lines"] if len(x.strip()) >= 3],
            "tag_regex": {r["regex"]: r["tag"] for r in ans["tag_regex"] if r["tag"] in speakers},
            "notes": {}, "consent": meta.get("consent", {"anonymise_requested": None, "evidence": None}),
            "llm": {"model": MODEL, "date": time.strftime("%Y-%m-%d")}}
    for k in ("exclude", "ocr_images", "hand_markers_valid", "tag_style", "text_sub"):
        if k in meta:
            spec[k] = meta[k]
    if ans["paragraph_order"] and meta.get("tag_style") == "none":
        spec["paragraph_speakers"] = {"order": [t for t in ans["paragraph_order"] if t in speakers]}
    if raw is not None:                                 # Zeilenanfänge so, wie der Parser sie findet
        i0 = 0
        if spec["body_start"]:
            spec["body_start"], i0 = anchor(raw, fixed, spec["body_start"])
            i0 = i0 or 0
        if spec["body_end"]:
            spec["body_end"], _ = anchor(raw, fixed, spec["body_end"], i0 + 1)
        for k in ("drop_lines", "gap_lines"):           # nie eine Gesprächszeile (> 15 Wörter) verwerfen
            keep = []
            for x in spec[k]:
                pre, i = anchor(raw, fixed, x)
                hits = [l for l in fixed if l.strip().startswith(pre)]
                if hits and max(len(l.split()) for l in hits) > 15:
                    spec.setdefault("_verworfen", []).append(k)
                    continue
                keep.append(pre)
            spec[k] = keep
    spec["llm"]["verworfene_zeilenregeln"] = len(spec.pop("_verworfen", []))
    spec["drop_lines"] = ["^" + re.escape(x) for x in spec["drop_lines"]]
    spec["gap_lines"] = ["^" + re.escape(x) for x in spec["gap_lines"]]
    for rx in list(spec["tag_regex"]):                  # ungültige Regex gar nicht erst an den Parser
        try:
            re.compile(rx)
        except re.error:
            del spec["tag_regex"][rx]
    return spec


def problems(rec, spec=None, fixed=None):
    """Was nach einem erfolgreichen Parse noch nicht stimmt (Rückmeldung an das Modell)."""
    out = []
    st = rec["stats"]
    if spec is not None and fixed is not None:
        from parse_transcripts import Tags, PROFILES
        prof = PROFILES[spec["profile"]]
        tags = Tags(spec, spec.get("tag_style") or prof["tag_style"])
        b0, b1 = rec["body_lines"]
        after = sum(1 for l in fixed[b1:] if l.strip() and tags.match_line(l.strip())[0])
        if after >= 5:
            out.append(f"Nach body_end folgen noch {after} Zeilen, die mit einem Sprecherkürzel beginnen – "
                       f"body_end liegt zu früh (oder auf einer Zeile, die mitten im Gespräch wiederkehrt).")
        first = next((i for i, l in enumerate(fixed) if l.strip() and tags.match_line(l.strip())[0]), b0)
        before = sum(1 for l in fixed[first:b0] if l.strip() and tags.match_line(l.strip())[0])
        if before >= 5:
            out.append(f"Vor body_start stehen schon {before} Zeilen, die mit einem Sprecherkürzel beginnen – "
                       f"body_start liegt zu spät.")
        for k, label in (("drop_lines", "drop_lines"), ("gap_lines", "gap_lines")):
            for rx in spec.get(k, []):
                r = re.compile(rx)
                long = [l for l in fixed[b0:b1] if r.search(l.strip()) and len(l.split()) > 15]
                if long:
                    out.append(f"Der {label}-Eintrag {rx[1:41]!r} trifft {len(long)} Zeile(n) mit mehr als 15 "
                               f"Wörtern – das ist Gesprächstext, kein Seitenkopf/keine Lücke.")
    words = collections.Counter()
    for t in rec["turns"]:
        if t["role"] == "unbekannt":
            words[t["speaker"]] += t["n_words"]
    for tag, n in words.most_common():
        if n >= 20:
            out.append(f"Das Kürzel {tag!r} kommt am Zeilenanfang vor ({n} Wörter), steht aber nicht in speakers.")
    by_role = collections.Counter()
    by_spk = collections.Counter()
    for t in rec["turns"]:
        by_role[t["role"]] += t["n_words"]
        by_spk[(t["speaker"], t["role"])] += t["n_words"]
    if not by_role.get("zeitzeuge"):
        out.append("Kein Sprecher hat die Rolle zeitzeuge – wer wird interviewt?")
    elif by_role.get("interviewer", 0) > by_role["zeitzeuge"]:
        out.append(f"Die als interviewer markierten Sprecher sprechen {by_role['interviewer']} Wörter, die als "
                   f"zeitzeuge markierten nur {by_role['zeitzeuge']} – Rollen bzw. paragraph_order prüfen "
                   f"(nur beibehalten, wenn der Interviewer hier wirklich mehr spricht).")
    nonint = sum(n for (_, r), n in by_spk.items() if r != "interviewer")
    for (tag, r), n in by_spk.items():
        if r == "weitere" and nonint and n >= 0.15 * nonint:
            out.append(f"{tag!r} hat die Rolle weitere, spricht aber {n} von {nonint} Wörtern der "
                       f"Nicht-Interviewer – wenn die Person mitinterviewt wird, ist die Rolle zeitzeuge.")
    total = sum(t["n_words"] for t in rec["turns"])
    if total < 300:
        out.append(f"Es bleiben nur {total} Wörter Gespräch übrig – Gesprächsanfang/-ende oder Kürzel prüfen.")
    if st.get("vor_erstem_kuerzel", 0) > 30:
        out.append(f"{st['vor_erstem_kuerzel']} Zeilen nach body_start stehen vor dem ersten Sprecherkürzel.")
    return out


# ---------------------------------------------------------------- Vergleich mit früheren Turns

WORD = re.compile(r"[\wÄÖÜäöüß]+")


def word_rows(turns):
    rows = []
    for t in turns:
        for w in WORD.findall(re.sub(r"\[(unv|gap|nv)\]", " ", t["text"]).lower()):
            rows.append((w, t["speaker"], t["role"]))
    return rows


def compare(old, new):
    """Wörter gemeinsam, Rolle/Sprecher je Wort gleich (Sprecher über Mehrheit zugeordnet), Wechsel."""
    P, L = word_rows(old["turns"]), word_rows(new["turns"])
    sm = difflib.SequenceMatcher(None, [r[0] for r in P], [r[0] for r in L], autojunk=False)
    pairs = [(b.a + k, b.b + k) for b in sm.get_matching_blocks() for k in range(b.size)]
    vote = collections.defaultdict(collections.Counter)
    for a, b in pairs:
        vote[L[b][1]][P[a][1]] += 1
    m = {l: c.most_common(1)[0][0] for l, c in vote.items()}
    n = max(1, len(pairs))
    roles = collections.Counter((P[a][2], L[b][2]) for a, b in pairs)
    tp = fp = fn = 0
    for (a0, b0), (a1, b1) in zip(pairs, pairs[1:]):
        cp, cl = P[a0][1] != P[a1][1], m[L[b0][1]] != m[L[b1][1]]
        tp += cp and cl; fp += cl and not cp; fn += cp and not cl
    return {"woerter_alt": len(P), "woerter_neu": len(L), "gemeinsam": len(pairs),
            "rolle_gleich": round(sum(P[a][2] == L[b][2] for a, b in pairs) / n, 4),
            "sprecher_gleich": round(sum(m[L[b][1]] == P[a][1] for a, b in pairs) / n, 4),
            "rollen_paare": {f"{a}->{b}": c for (a, b), c in roles.items() if a != b},
            "wechsel": {"beide": tp, "nur_neu": fp, "nur_alt": fn}}


# ---------------------------------------------------------------- Ablauf

def run_one(meta, vocab, a, usage):
    from parse_transcripts import parse
    raw, fixed = source(meta, vocab)
    base = prompt(meta)
    messages = [{"role": "user", "text": base}]
    log = {"id": meta["id"], "model": a.model, "versuche": []}
    spec = rec = None
    for attempt in range(a.retries + 1):
        ans = ask(messages, a.model, usage, a.response_format)
        spec = build_spec(meta, ans, raw, fixed)
        err, probs = None, []
        try:
            rec = parse(spec, vocab)
            probs = problems(rec, spec, fixed)
        except Exception as e:                       # Parser-Fehler inkl. Wortprüfung
            err, rec = str(e)[:400], None
        log["versuche"].append({"antwort": ans, "fehler": err, "probleme": probs})
        if err is None and not probs:
            break
        msg = ("Der Parser meldet einen Fehler: " + err) if err else "Der Parser meldet: " + " ".join(probs)
        if err and "Steuer-/Platzhalterzeichen" in err:
            msg += (" Das heißt meist: Nach dem Gespräch folgt noch Text, der nicht zum Gespräch gehört (Abspann,"
                    " Unterschrift, Formular, Logo) – body_end muss auf die erste Zeile davon zeigen.")
        if err and "nicht gefunden" in err:
            msg += (" Kopiere nur die ersten 15 bis 25 Zeichen der Zeile, zeichengenau und ohne Zeilennummer –"
                    " oder setze body_start auf null, wenn das Gespräch mit der ersten Kürzelzeile beginnt.")
        if err and "nicht wortgetreu" in err:
            msg += " Das heißt meist: ein drop_lines- oder gap_lines-Eintrag oder ein tag_regex trifft Gesprächstext."
        messages += [{"role": "assistant", "text": json.dumps(ans, ensure_ascii=False)},
                     {"role": "user", "text": msg + "\nKorrigiere die Angaben und gib das vollständige JSON neu zurück."}]
    log["ok"] = rec is not None
    log["rest_probleme"] = log["versuche"][-1]["probleme"] if rec is not None else None
    write_json(OUT / f"{meta['id']}.json", log)
    if rec is not None:
        write_json(SPECS / f"{meta['id']}.json", spec)
    return rec, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="Komma-Liste von ID-Anfängen")
    ap.add_argument("--force", action="store_true", help="auch Interviews mit vorhandener Spec neu")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--response-format", default="schema", choices=["schema", "object", "none"],
                    help="schema: JSON-Schema erzwingen (Standard); object/none für Endpunkte ohne Schema-Zwang")
    ap.add_argument("--price", default=None, help="US$ je Mio. Token ein,aus (nur für die Kostenangabe), z. B. 0.3,1.2")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--compare", default=None, help="Ordner mit früheren Turns (<id>.json) zum Vergleich")
    a = ap.parse_args()
    from parse_transcripts import build_vocab
    metas = json.loads(CONFIG.read_text(encoding="utf-8"))["interviews"]
    if a.only:
        metas = [m for m in metas if any(m["id"].startswith(o) for o in a.only.split(","))]
    SPECS.mkdir(parents=True, exist_ok=True)
    if SPLIT_SRC.exists():
        (SPECS / "_split.json").write_text(SPLIT_SRC.read_text(encoding="utf-8"), encoding="utf-8")
    vocab = build_vocab(l for m in json.loads(CONFIG.read_text(encoding="utf-8"))["interviews"]
                        if m["transcript"].endswith(".docx") for l in read_lines(resolve(m["transcript"])))
    usage = collections.Counter()
    summary, cmp = [], {}
    t0 = time.time()
    for m in metas:
        if (SPECS / f"{m['id']}.json").exists() and not a.force and not a.compare:
            continue
        rec, log = run_one(m, vocab, a, usage)
        n_try = len(log["versuche"])
        line = f"{m['id']:26} {'ok ' if log['ok'] else 'FEHLER'} Versuche {n_try}"
        if rec is not None:
            line += f"  {len(rec['turns']):4} Turns {sum(t['n_words'] for t in rec['turns']):6} W"
            if log["rest_probleme"]:
                line += f"  offen: {len(log['rest_probleme'])}"
        else:
            line += f"  {log['versuche'][-1]['fehler'][:90]}"
        if a.compare and rec is not None:
            f = Path(a.compare) / f"{m['id']}.json"
            if f.exists():
                c = compare(json.loads(f.read_text(encoding="utf-8")), rec)
                cmp[m["id"]] = c
                line += (f"  | Wörter {c['gemeinsam']}/{c['woerter_alt']}  Rolle {100*c['rolle_gleich']:.1f} %"
                         f"  Sprecher {100*c['sprecher_gleich']:.1f} %")
        print(line, flush=True)
        summary.append({"id": m["id"], "ok": log["ok"], "versuche": n_try, "offen": log["rest_probleme"]})
    cost = ""
    if a.price:
        p_in, p_out = (float(x) for x in a.price.split(","))
        cost = f"~{usage['in'] / 1e6 * p_in + usage['out'] / 1e6 * p_out:.2f} US$, "
    print(f"\n{sum(s['ok'] for s in summary)}/{len(summary)} Specs gültig; API {usage['calls']} Aufrufe "
          f"(+{usage['cache']} aus dem Zwischenspeicher), {usage['in']:,} Token ein / {usage['out']:,} aus, "
          f"{cost}{time.time() - t0:.0f} s; abgeschnitten {usage['abgeschnitten']}")
    if cmp:
        W = sum(c["woerter_alt"] for c in cmp.values())
        G = sum(c["gemeinsam"] for c in cmp.values())
        R = sum(c["rolle_gleich"] * c["gemeinsam"] for c in cmp.values())
        S = sum(c["sprecher_gleich"] * c["gemeinsam"] for c in cmp.values())
        print(f"Vergleich mit {a.compare}: {len(cmp)} Interviews, Wörter erfasst {100*G/W:.2f} %, "
              f"Rolle je Wort gleich {100*R/G:.2f} %, Sprecher je Wort gleich {100*S/G:.2f} %")
        write_json(OUT / "_vergleich.json", {"model": a.model, "interviews": cmp, "usage": dict(usage),
                                            "summe": {"woerter_alt": W, "gemeinsam": G, "rolle": R / G, "sprecher": S / G}})


if __name__ == "__main__":
    main()
