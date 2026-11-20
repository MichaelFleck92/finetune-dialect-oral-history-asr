#!/usr/bin/env python3
"""
Gemeinsame Helfer: Pfade, Interview-Katalog (aus den Specs), Hashes, Audio.

Jede Stufe liest den Katalog aus pipeline/specs/*.json. Eine Spec ist die
einzige Stelle, an der festgelegt wird, welches Transkript zu welchen
Audioteilen gehoert, wer welche Rolle hat und wo der Gespraechstext beginnt.
"""
import hashlib, json, re, subprocess, unicodedata, wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PIPE = ROOT / "pipeline"
SPECS = PIPE / "specs"
SRC = ROOT / "Interviews"
WORK = ROOT / "work"
DATASET = ROOT / "dataset"
SR = 16000

PROJECTS = {
    "rs": "Roter Stern über Graz",
    "re": "Rechnitz (BDA)",
    "za": "Zwangsarbeiter",
}


def nfc(s):
    return unicodedata.normalize("NFC", s)


def slug(s):
    s = unicodedata.normalize("NFKD", s.replace("ß", "ss")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def resolve(rel):
    """Pfad aus einer Spec (NFC) auf die Platte abbilden. APFS findet NFC und
    NFD meist beide; zur Sicherheit wird notfalls im Verzeichnis verglichen."""
    p = ROOT / rel
    if p.exists():
        return p
    cur = ROOT
    for part in Path(rel).parts:
        cand = cur / part
        if not cand.exists():
            hits = [c for c in cur.iterdir() if nfc(c.name) == nfc(part)] if cur.is_dir() else []
            if not hits:
                raise FileNotFoundError(rel)
            cand = hits[0]
        cur = cand
    return cur


def rel(p):
    # absolute() statt resolve(): Interviews/ darf ein symbolischer Link auf den Datenordner sein
    return nfc(str(Path(p).absolute().relative_to(ROOT)))


def load_specs():
    out = []
    for f in sorted(SPECS.glob("*.json")):
        if f.name.startswith("_"):
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        s["_spec_file"] = f
        out.append(s)
    return out


def spec_by_id(iid):
    return json.loads((SPECS / f"{iid}.json").read_text(encoding="utf-8"))


def sha1_file(p, chunk=1 << 20):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha1_text(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def code_hash(*names):
    """Hash der beteiligten Pipeline-Dateien - landet in jeder Ausgabe, damit
    veraltete Zwischenstaende erkannt werden (v1 hatte genau dieses Problem)."""
    h = hashlib.sha1()
    for n in names:
        h.update((PIPE / n).read_bytes())
    return h.hexdigest()[:12]


def audio_duration(path):
    """Dauer in Sekunden ueber die ffmpeg-Kopfzeile (ffprobe fehlt hier)."""
    if Path(path).suffix.lower() == ".wav":
        try:
            with wave.open(str(path)) as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            pass
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path)],
                       capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", r.stderr)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def read_wav16(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
