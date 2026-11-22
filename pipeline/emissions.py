#!/usr/bin/env python3
"""
Stage 3a: compute and cache wav2vec2 emissions per interview.

This is the expensive step (CPU, ~15x real time). Everything else - anchors,
forced alignment, checks - works on work/emissions/<id>.npy and can therefore
be rerun as often as needed without touching the model.

MPS (Apple GPU) returns unusable logits for this model with torch 2.3.1
(greedy output "eeeeee", checked again on 26.09.2026) - hence CPU.

  python3 pipeline/emissions.py [--only ID] [--force]
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, SR, load_specs, read_wav16, sha1_file, write_json

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
FPS = 50.0                      # hop of 320 samples = 20 ms
OUT = WORK / "emissions"


def load_model():
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).eval()
    # Note: transformers reports pos_conv_embed as "newly initialized". That
    # is a false alarm - torch remaps weight_g/weight_v on loading; checked on
    # 25.09.2026 against the checkpoint tensors (bit-identical).
    return proc, model


def model_revision():
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(MODEL_ID, local_files_only=True)).name
    except Exception:
        return "unknown"


@torch.no_grad()
def compute(audio, model, win_s=20.0, ctx_s=1.0):
    """Log-probs for the whole file, window by window with 1 s of context on each side."""
    win, ctx = int(win_s * SR), int(ctx_s * SR)
    out, n = [], len(audio)
    for start in range(0, n, win):
        a, b = max(0, start - ctx), min(n, start + win + ctx)
        seg = audio[a:b]
        if len(seg) < 400:
            break
        x = torch.from_numpy(seg).unsqueeze(0)
        x = (x - x.mean()) / (x.std() + 1e-7)
        lg = model(x).logits.float().log_softmax(-1)[0]
        lo = int(round((start - a) / SR * FPS))
        hi = lo + int(round((min(n, start + win) - start) / SR * FPS))
        out.append(lg[lo:min(hi, lg.shape[0])].numpy())
    return np.concatenate(out, 0)


def probe(emis, proc):
    """Catch silent errors (wrong device, broken weights)."""
    mid = emis.shape[0] // 2
    ids = torch.from_numpy(emis[mid:mid + 750].argmax(-1)).unsqueeze(0)
    txt = proc.batch_decode(ids)[0]
    return sum(c.isalpha() for c in txt), txt[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [s for s in load_specs()
             if not a.only or any(o in s["id"] for o in a.only.split(","))]
    todo = []
    for s in specs:
        wav = WORK / "audio16k" / f"{s['id']}.wav"
        meta = OUT / f"{s['id']}.json"
        if not wav.exists():
            print(f"!! fehlt: {wav.name}"); continue
        if meta.exists() and not a.force:
            m = json.loads(meta.read_text())
            if m.get("audio_sha1") == sha1_file(wav):
                continue
        todo.append((s["id"], wav))
    print(f"{len(specs) - len(todo)} vorhanden, {len(todo)} zu berechnen", flush=True)
    if not todo:
        return
    torch.set_num_threads(max(1, torch.get_num_threads()))
    proc, model = load_model()
    rev = model_revision()
    vocab = proc.tokenizer.get_vocab()
    t_all = time.time()
    for k, (iid, wav) in enumerate(todo, 1):
        t0 = time.time()
        audio = read_wav16(wav)
        emis = compute(audio, model)
        n_let, txt = probe(emis, proc)
        if n_let < 30:
            raise RuntimeError(f"{iid}: Probe-Dekodierung ergab {n_let} Buchstaben ({txt!r})")
        np.save(OUT / f"{iid}.npy", emis.astype(np.float16))
        write_json(OUT / f"{iid}.json", {
            "id": iid, "model": MODEL_ID, "revision": rev, "fps": FPS,
            "frames": int(emis.shape[0]), "audio_s": round(len(audio) / SR, 3),
            "audio_sha1": sha1_file(wav), "vocab": vocab, "probe": txt})
        dt = time.time() - t0
        print(f"[{k}/{len(todo)}] {iid:28} {len(audio)/SR/60:6.1f} min in {dt/60:5.1f} min "
              f"({len(audio)/SR/dt:4.1f}x)  gesamt {(time.time()-t_all)/3600:4.2f} h  "
              f"probe: {txt[:40]!r}", flush=True)


if __name__ == "__main__":
    main()
