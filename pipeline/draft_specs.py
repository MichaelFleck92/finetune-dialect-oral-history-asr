#!/usr/bin/env python3
"""
Stage 0: create draft specs for all interviews under Interviews/.

Only creates specs that do not exist yet - completed specs are never
overwritten. The draft contains the transcript and the audio parts in playback order;
speakers, roles, conversation boundaries and non-conversation lines are set by pipeline/llm_specs.py.

  python3 pipeline/draft_specs.py            # create missing specs
  python3 pipeline/draft_specs.py --check    # only show what is missing
"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SRC, SPECS, nfc, slug, rel, write_json, audio_duration

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".mts"}
TEXT_EXT = {".docx", ".pdf", ".rtf", ".txt"}

# Where transcript and audio are not in the same folder or the file names do not match, the
# mapping is in config/sources.json (local, not in the repo, because the file names contain names); template:
# config/sources.example.json. Multi-part recordings in playback order.
_SRC = json.loads((ROOT / "config" / "sources.json").read_text(encoding="utf-8")) \
    if (ROOT / "config" / "sources.json").exists() else {}
RE_TON = _SRC.get("re_ton", "")
RECHNITZ = {k: tuple(v) for k, v in _SRC.get("rechnitz", {}).items()}
ZA_DIR = _SRC.get("za_dir", "")
ZWANGSARBEITER = {k: tuple(v) for k, v in _SRC.get("zwangsarbeiter", {}).items()}


def part_key(p):
    """Playback order of multi-part recordings: 'Name 27072023' before
    'Name 2 27072023', 'Name 1' before 'Name 2'."""
    stem = nfc(p.stem)
    m = re.search(r"(?:teil|part|[_\s])(\d)(?:\b|_)", stem, re.I)
    return (int(m.group(1)) if m else 1, stem)


def empty_spec(iid, project):
    return {"id": iid, "project": project, "status": "draft",
            "transcript": None, "audio": [], "interviewee": None, "date": None,
            "tag_style": None, "body_start": None, "body_end": None,
            "speakers": {}, "notes": {}, "consent": {"anonymise_requested": None,
                                                     "evidence": None},
            "glyph_repair": False, "remarks": ""}


def drafts():
    out = []
    rs = SRC / "RoterStern-Align"
    for d in sorted(rs.iterdir()) if rs.exists() else []:
        if not d.is_dir():
            continue
        files = [p for p in sorted(d.rglob("*")) if p.is_file() and not p.name.startswith(".")]
        tx = [p for p in files if p.suffix.lower() in TEXT_EXT]
        au = sorted([p for p in files if p.suffix.lower() in AUDIO_EXT], key=part_key)
        s = empty_spec("rs_" + slug(nfc(d.name)), "rs")
        s["transcript"] = rel(tx[0]) if len(tx) == 1 else None
        if len(tx) != 1:
            s["remarks"] = f"{len(tx)} Textdateien im Ordner - von Hand festlegen"
        s["audio"] = [rel(p) for p in au]
        out.append(s)
    for key, (tx, auds) in RECHNITZ.items():
        s = empty_spec("re_" + key, "re")
        s["transcript"] = "Interviews/Rechnitz-Align/" + tx
        s["audio"] = [RE_TON + a for a in auds]
        out.append(s)
    for key, (folder, wav) in ZWANGSARBEITER.items():
        s = empty_spec("za_" + key, "za")
        s["transcript"] = f"work/ocr/za_{key}.txt"
        s["ocr_images"] = ZA_DIR + "Zwangsarbeiter-audio-transcripts/" + folder
        s["audio"] = [ZA_DIR + wav]
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    SPECS.mkdir(parents=True, exist_ok=True)
    new = 0
    for s in drafts():
        f = SPECS / f"{s['id']}.json"
        missing = [x for x in s["audio"] if not (ROOT / x).exists()]
        if s["transcript"] and not s["transcript"].startswith("work/") \
                and not (ROOT / s["transcript"]).exists():
            missing.append(s["transcript"])
        durs = [audio_duration(ROOT / x) for x in s["audio"] if (ROOT / x).exists()]
        info = f"{len(s['audio'])} Teil(e), {sum(d or 0 for d in durs)/60:6.1f} min"
        if f.exists():
            continue
        new += 1
        print(f"{'(fehlt) ' if a.check else 'neu     '}{s['id']:28} {info}"
              + (f"   FEHLT: {missing}" if missing else ""))
        if not a.check:
            write_json(f, s)
    print(f"{new} {'fehlende' if a.check else 'neue'} Specs")


if __name__ == "__main__":
    main()
