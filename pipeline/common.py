#!/usr/bin/env python3
"""
Shared helpers: paths, interview catalogue (from the specs), hashes, audio.

Every stage reads the catalogue from pipeline/specs/*.json. A spec is the
only place that defines which transcript belongs to which audio parts, who
has which role and where the conversation text begins.
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
    """Map a path from a spec (NFC) to the disk. APFS usually finds both NFC
    and NFD; to be safe, the directory is compared if necessary."""
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
    # absolute() instead of resolve(): Interviews/ may be a symbolic link to the data folder
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
    """Hash of the pipeline files involved - stored in every output so that
    stale intermediate results are detected (v1 had exactly this problem)."""
    h = hashlib.sha1()
    for n in names:
        h.update((PIPE / n).read_bytes())
    return h.hexdigest()[:12]


def audio_duration(path):
    """Duration in seconds from the ffmpeg header (ffprobe is missing here)."""
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
