#!/usr/bin/env python3
"""
Stage 5: cut out the segments and write them as a dataset.

  dataset/audio/<interview>/<interview>_<start_ms>.flac   16 kHz mono, lossless
  dataset/metadata.jsonl                                  one row per segment
  dataset/words.jsonl                                     word times per segment
  dataset/manifest.json                                   provenance, code version, figures

Compared with v1:
  * File names contain the start time (ms). If the segmentation changes,
    the name changes - an old file can never be reused under a new text
    (v1 --reuse-audio only compared the running number).
    Only what the manifest says was cut from the same source with the same
    boundaries is reused.
  * The split is now fixed (pipeline/specs/_split.json), by groups of persons:
    interviews in which the same person speaks (persons interviewed twice,
    married couples, a person appearing as a "friend" in another interview)
    always stay together. Interviewers speak in many interviews -
    their segments go into training only; the interviewer segments of
    dev/test interviews are left out, so that no voice and no room is on
    both sides. New interviews go into training.

  python3 pipeline/export_dataset.py [--dry-run]
"""
import argparse, collections, hashlib, json, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORK, SPECS, DATASET, SR, load_specs, write_json, sha1_file, code_hash, PROJECTS

SPLIT_FILE = SPECS / "_split.json"
TEST_SHARE, DEV_SHARE = 0.10, 0.10


# ------------------------------------------------------------ groups of persons

def person_groups(specs):
    """Union-find over interviews that share a non-interviewer person."""
    parent = {s["id"]: s["id"] for s in specs}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    owner = {}
    for s in specs:
        for sp in s["speakers"].values():
            if sp["role"] == "interviewer":
                continue
            p = sp["person"]
            if p in owner:
                parent[find(s["id"])] = find(owner[p])
            else:
                owner[p] = s["id"]
    groups = collections.defaultdict(list)
    for s in specs:
        groups[find(s["id"])].append(s["id"])
    return {g: sorted(v) for g, v in groups.items()}


def make_split(specs, hours):
    """Create the fixed split (only if _split.json is missing): per project
    ~10 % test / ~10 % dev by hours, groups in a fixed random order."""
    groups = person_groups(specs)
    proj = {s["id"]: s["project"] for s in specs}
    assign = {}
    for p in PROJECTS:
        gs = [g for g, ids in groups.items() if proj[ids[0]] == p]
        gs.sort(key=lambda g: hashlib.md5(("rs-split-2026:" + g).encode()).hexdigest())
        total = sum(hours.get(i, 0) for g in gs for i in groups[g])
        acc = {"test": 0.0, "dev": 0.0}
        for g in gs:
            h = sum(hours.get(i, 0) for i in groups[g])
            if len(gs) <= 3:                        # forced labourers: 1 test, rest training
                sp = "test" if acc["test"] == 0 else "train"
            elif acc["test"] + h <= TEST_SHARE * total * 1.3 and acc["test"] < TEST_SHARE * total:
                sp = "test"
            elif acc["dev"] + h <= DEV_SHARE * total * 1.3 and acc["dev"] < DEV_SHARE * total:
                sp = "dev"
            else:
                sp = "train"
            if sp != "train":
                acc[sp] += h
            for i in groups[g]:
                assign[i] = {"split": sp, "group": g}
    return {"_doc": "Feste Aufteilung nach Personengruppen (erzeugt von export_dataset.py, "
                    "danach nur bewusst von Hand aendern). Neue Interviews: Training.",
            "created": time.strftime("%Y-%m-%d"), "interviews": dict(sorted(assign.items()))}


# ------------------------------------------------------------ cutting

def cut(src, dst, start, dur):
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                        "-i", str(src), "-ac", "1", "-ar", str(SR), "-c:a", "flac",
                        "-compression_level", "5", str(dst)], capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip()[:200]


def source_part(parts, t):
    """Time in the concatenated 16 kHz audio -> (original file, time within it)."""
    cur = parts[0]
    for p in parts:
        if t >= p["offset_s"]:
            cur = p
    return cur["source"], round(t - cur["offset_s"], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    specs = load_specs()
    segfiles = {s["id"]: WORK / "segments" / f"{s['id']}.json" for s in specs}
    missing = [i for i, f in segfiles.items() if not f.exists()]
    if missing:
        sys.exit(f"ABBRUCH: Segmente fehlen fuer {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    segs = {i: json.loads(f.read_text()) for i, f in segfiles.items()}
    hours = {i: sum(o["duration"] for o in d["segments"] if o["keep"]) / 3600 for i, d in segs.items()}
    if SPLIT_FILE.exists():
        split = json.loads(SPLIT_FILE.read_text())
    else:
        split = make_split(specs, hours)
        write_json(SPLIT_FILE, split)
        print(f"Aufteilung neu festgelegt -> {SPLIT_FILE.relative_to(ROOT)}")
    groups = person_groups(specs)
    grp_of = {i: g for g, ids in groups.items() for i in ids}
    # consistency: a group must not be spread over several parts
    for g, ids in groups.items():
        parts = {split["interviews"].get(i, {"split": "train"})["split"] for i in ids}
        if len(parts) > 1:
            sys.exit(f"ABBRUCH: Personengruppe {ids} liegt in mehreren Teilen {parts} - _split.json anpassen")

    old_manifest = {}
    mf = DATASET / "manifest.json"
    if mf.exists():
        old_manifest = {e["file_name"]: e for e in json.loads(mf.read_text()).get("files", [])}
    rows, words_rows, files, jobs = [], [], [], []
    stats = collections.Counter()
    for s in specs:
        iid = s["id"]
        sp = split["interviews"].get(iid, {"split": "train"})["split"]
        parts = json.loads((WORK / "audio16k" / f"{iid}.json").read_text())["parts"]
        wav = WORK / "audio16k" / f"{iid}.wav"
        src_key = hashlib.sha1("".join(p["sha1"] for p in parts).encode()).hexdigest()[:16]
        for o in segs[iid]["segments"]:
            if not o["keep"]:
                continue
            seg_split = sp
            if sp != "train" and o["role"] == "interviewer":
                stats["interviewer_in_devtest_ausgelassen"] += 1
                continue
            rel = f"audio/{iid}/{iid}_{int(round(o['start'] * 1000)):08d}.flac"
            key = f"{src_key}:{o['start']:.3f}:{o['end']:.3f}"
            src_file, src_t = source_part(parts, o["start"])
            rows.append({
                "file_name": rel, "text": o["text"], "text_verbatim": o["text_verbatim"],
                "duration": o["duration"], "split": seg_split,
                "interview": iid, "project": s["project"], "group": grp_of[iid],
                "speaker": o["speaker"], "speaker_name": o["speaker_name"], "role": o["role"],
                "person": o["person"], "interviewee": s.get("interviewee"),
                "interviewer": s.get("interviewer"), "date": s.get("date"),
                "quality": o["quality"], "score": o["score"], "score_rel": o["score_rel"],
                "edge_min": o["edge_min"], "words_per_s": o["words_per_s"], "n_words": o["n_words"],
                "has_truncation": o["has_truncation"], "fillers_untranscribed": o["fillers_untranscribed"],
                "consent_anonymise": s["consent"]["anonymise_requested"],
                "consent_evidence": s["consent"]["evidence"],
                "source_start": o["start"], "source_end": o["end"],
                "source_file": src_file, "source_file_start": src_t, "language": "de"})
            words_rows.append({"file_name": rel, "words": o["words"]})
            files.append({"file_name": rel, "key": key})
            dst = DATASET / rel
            if not a.dry_run:
                if dst.exists() and old_manifest.get(rel, {}).get("key") == key:
                    stats["wiederverwendet"] += 1
                else:
                    jobs.append((wav, dst, o["start"], o["duration"]))
    print(f"{len(rows)} Segmente, {sum(r['duration'] for r in rows)/3600:.2f} h; "
          f"zu schneiden {len(jobs)}, wiederverwendet {stats['wiederverwendet']}")
    if a.dry_run:
        return
    errs = []
    with ThreadPoolExecutor(a.workers) as ex:
        for ok, err in ex.map(lambda j: cut(*j), jobs):
            if not ok:
                errs.append(err)
    if errs:
        sys.exit(f"ABBRUCH: {len(errs)} Schnittfehler, z.B. {errs[:2]}")
    want = {r["file_name"] for r in rows}
    gone = 0
    for f in sorted((DATASET / "audio").rglob("*.flac")) if (DATASET / "audio").exists() else []:
        if str(f.relative_to(DATASET)) not in want:
            f.unlink(); gone += 1
    for d in sorted((DATASET / "audio").iterdir(), reverse=True) if (DATASET / "audio").exists() else []:
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    (DATASET / "metadata.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                            encoding="utf-8")
    (DATASET / "words.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in words_rows) + "\n",
                                         encoding="utf-8")
    by = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        by[r["split"]]["segmente"] += 1; by[r["split"]]["sekunden"] += r["duration"]
        by[r["project"]]["segmente"] += 1; by[r["project"]]["sekunden"] += r["duration"]
    em_rev = json.loads(next((WORK / "emissions").glob("*.json")).read_text()).get("revision")
    write_json(mf, {
        "created": time.strftime("%Y-%m-%d %H:%M"), "segments": len(rows),
        "hours": round(sum(r["duration"] for r in rows) / 3600, 3),
        "by": {k: {"segmente": v["segmente"], "stunden": round(v["sekunden"] / 3600, 3)} for k, v in by.items()},
        "interviews": len({r["interview"] for r in rows}),
        "persons_zeitzeug_innen": len({r["person"] for r in rows if r["role"] == "zeitzeuge"}),
        "model": {"alignment": "jonatasgrosman/wav2vec2-large-xlsr-53-german", "revision": em_rev},
        "code": {n: code_hash(n) for n in ("parse_transcripts.py", "sources.py", "notes.py", "align.py",
                                           "segment.py", "export_dataset.py", "convert_audio.py",
                                           "emissions.py")},
        "specs_sha1": hashlib.sha1("".join(sha1_file(f) for f in sorted(SPECS.glob("*.json"))).encode()).hexdigest(),
        "stats": dict(stats), "files": files})
    print(f"{gone} verwaiste Dateien entfernt; Aufteilung: "
          + ", ".join(f"{k} {v['segmente']} Segm./{v['sekunden']/3600:.2f} h" for k, v in by.items()
                      if k in ("train", "dev", "test")))


if __name__ == "__main__":
    main()
