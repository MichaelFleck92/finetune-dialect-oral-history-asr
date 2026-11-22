#!/usr/bin/env python3
"""
Stage 2: original recordings -> work/audio16k/<id>.wav (16 kHz, mono, 16 bit).

Multi-part recordings are concatenated in the order of the spec; the part
boundaries go to work/audio16k/<id>.json. For stereo, it is first checked
whether downmixing cancels speech (channels in antiphase) - in that case the
louder channel is taken instead of the sum.

  python3 pipeline/convert_audio.py [--only ID] [--force]
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (WORK, SR, load_specs, resolve, audio_duration, sha1_file,
                    write_json)

OUT = WORK / "audio16k"


def channel_plan(src):
    """'mix' or 'c0'/'c1': 3-minute sample from the middle, decoded in stereo."""
    dur = audio_duration(src) or 0
    start = max(0.0, dur / 2 - 90)
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.1f}", "-t", "180",
                        "-i", str(src), "-vn", "-ac", "2", "-ar", str(SR), "-f", "s16le", "-"],
                       capture_output=True)
    x = np.frombuffer(r.stdout, dtype="<i2").astype(np.float32)
    if x.size < 2 * SR:
        return "mix", {}
    L, R = x[0::2], x[1::2]
    rl, rr = np.sqrt(np.mean(L ** 2)) + 1e-9, np.sqrt(np.mean(R ** 2)) + 1e-9
    rm = np.sqrt(np.mean(((L + R) / 2) ** 2)) + 1e-9
    corr = float(np.corrcoef(L, R)[0, 1]) if rl > 1 and rr > 1 else 1.0
    info = {"rms_db_L": round(20 * np.log10(rl), 1), "rms_db_R": round(20 * np.log10(rr), 1),
            "mix_loss_db": round(20 * np.log10(max(rl, rr) / rm), 1), "corr_LR": round(corr, 3)}
    # Compared with the louder channel, the sum normally loses ~0-6 dB (one channel
    # empty = 6 dB). Much more means: the channels cancel each other out.
    if info["mix_loss_db"] > 9:
        return ("c0" if rl >= rr else "c1"), info
    return "mix", info


def n_channels(src):
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(src)],
                       capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "Audio:" in line:
            return 1 if "mono" in line else 2
    return 1


def up_to_date(spec, parts, tgt, side):
    """Output up to date? Same audio parts in the same order as in the spec,
    source files unchanged (size+mtime, else SHA-1 - if that matches, size
    and mtime are updated so that the next run is fast)."""
    if not (tgt.exists() and side.exists()):
        return False
    old = json.loads(side.read_text(encoding="utf-8"))
    if [p["source"] for p in old["parts"]] != list(spec["audio"]):
        return False
    touched = False
    for rec, path in zip(old["parts"], parts):
        st = path.stat()
        if rec.get("size") == st.st_size and rec.get("mtime_ns") == st.st_mtime_ns:
            continue
        if sha1_file(path) != rec["sha1"]:
            return False
        rec["size"], rec["mtime_ns"] = st.st_size, st.st_mtime_ns
        touched = True
    if touched:
        write_json(side, old)
    return True


def convert(spec, force=False):
    tgt = OUT / f"{spec['id']}.wav"
    side = OUT / f"{spec['id']}.json"
    parts = [resolve(a) for a in spec["audio"]]
    if not force and up_to_date(spec, parts, tgt, side):
        return "vorhanden", None
    OUT.mkdir(parents=True, exist_ok=True)
    plans, infos = [], []
    for p in parts:
        if n_channels(p) == 2:
            plan, info = channel_plan(p)
        else:
            plan, info = "mono", {}
        plans.append(plan); infos.append(info)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    for p in parts:
        cmd += ["-i", str(p)]
    chains = []
    for i, plan in enumerate(plans):
        pan = {"c0": "pan=mono|c0=c0", "c1": "pan=mono|c0=c1"}.get(plan, "aformat=channel_layouts=mono")
        chains.append(f"[{i}:a:0]{pan},aresample={SR}[a{i}]")
    graph = ";".join(chains)
    if len(parts) > 1:
        graph += ";" + "".join(f"[a{i}]" for i in range(len(parts))) + \
                 f"concat=n={len(parts)}:v=0:a=1[out]"
    else:
        graph = graph.replace("[a0]", "[out]")
    cmd += ["-filter_complex", graph, "-map", "[out]", "-ac", "1", "-ar", str(SR),
            "-sample_fmt", "s16", str(tgt)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tgt.unlink(missing_ok=True)
        return "FEHLER", r.stderr.strip()[:300]
    durs = [audio_duration(p) for p in parts]
    offs, acc = [], 0.0
    for d in durs:
        offs.append(round(acc, 3)); acc += d or 0
    write_json(side, {"id": spec["id"], "parts": [
        {"source": a, "offset_s": o, "duration_s": round(d or 0, 3), "channels": pl,
         "sha1": sha1_file(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
         **({"stereo_check": inf} if inf else {})}
        for a, p, o, d, pl, inf in zip(spec["audio"], parts, offs, durs, plans, infos)],
        "duration_s": round(audio_duration(tgt), 3)})
    return "ok", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    specs = [s for s in load_specs()
             if not a.only or any(o in s["id"] for o in a.only.split(","))]
    for i, s in enumerate(specs, 1):
        st, err = convert(s, a.force)
        if st != "vorhanden":
            print(f"[{i}/{len(specs)}] {st:8} {s['id']}" + (f"  {err}" if err else ""), flush=True)
    n = len(list(OUT.glob("*.wav")))
    print(f"{n} Dateien in work/audio16k")


if __name__ == "__main__":
    main()
