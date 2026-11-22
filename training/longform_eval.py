#!/usr/bin/env python3
"""
Transcribe complete interviews: the chunks from longform_chunks.py (cut at pauses,
the same for all models) with the same decoding as whisper_eval.py
(German, without timestamps, --beams, --fallback, --prompts).

  python3 training/longform_eval.py --model openai/whisper-large-v3 --name baseline-large-v3 --beams 5 --fallback --tag beam5fb
  python3 training/longform_eval.py --model openai/whisper-large-v3 --adapter training/runs/<lauf> --name <lauf> --beams 5 --fallback --tag beam5fb

Whisper's own long-form mode needs timestamps; the fine-tuned model was trained
without timestamps - hence fixed chunks here instead of timestamp jumps.
Audio: complete recording per interview as <interview>.wav or .flac (16 kHz mono) in --audio-dir,
default work/audio16k (stage 2). On the Mac: --dtype fp16.
-> training/runs/<name>/longform[_<tag>]/<interview>.json ({"chunks": [[s, e, text], ...]}).
Scoring with longform_score.py. Only figures are printed.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import ROOT, RUNS, SR, default_device, load_model
from whisper_eval import compression_ratio, CR_MAX, FALLBACK_T

LF = ROOT / "training" / "longform"


def load_file(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def interview_audio(iv, audio_dir=None):
    """Complete recording of an interview: <audio_dir>/<iv>.wav or .flac (default work/audio16k)."""
    d = Path(audio_dir) if audio_dir else ROOT / "work" / "audio16k"
    for ext in (".wav", ".flac"):
        if (d / f"{iv}{ext}").exists():
            return load_file(d / f"{iv}{ext}")
    raise FileNotFoundError(f"keine Aufnahme für {iv} in {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interviews", default=None, help="Komma-Liste; Standard: alle in training/longform/chunks")
    ap.add_argument("--audio-dir", default=None, help="Ordner mit <interview>.wav/.flac (Standard work/audio16k)")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="auto: bf16 auf cuda, sonst fp32")
    a = ap.parse_args()
    dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}.get(a.dtype, torch.bfloat16 if a.device == "cuda" else None)
    proc, model = load_model(a.model, a.adapter, a.device, dtype)
    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8")) if a.prompts else None
    out_dir = RUNS / a.name / ("longform" + (f"_{a.tag}" if a.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((LF / "chunks").glob("*.json"))
    if a.interviews:
        files = [f for f in files if f.stem in a.interviews.split(",")]
    bs = max(1, a.batch // a.beams)
    gen = dict(language="de", task="transcribe", max_new_tokens=200)
    t0, n_fb, n_ch = time.time(), 0, 0
    for f in files:
        meta = json.loads(f.read_text(encoding="utf-8"))
        iv, chunks = meta["interview"], meta["chunks"]
        audio = interview_audio(iv, a.audio_dir)
        pid = proc.get_prompt_ids(prompts[iv.split("_")[0]], return_tensors="pt").to(a.device) if prompts else None
        g = dict(gen, prompt_ids=pid) if pid is not None else gen
        texts = []
        for i in range(0, len(chunks), bs):
            part = chunks[i:i + bs]
            seg = [audio[int(s * SR):int(e * SR)] for s, e in part]
            x = proc.feature_extractor(seg, sampling_rate=SR, return_tensors="pt").input_features.to(a.device, model.dtype)
            with torch.no_grad():
                ids = model.generate(x, num_beams=a.beams, **g)
            out = [h.strip() for h in proc.batch_decode(ids, skip_special_tokens=True)]
            for k, h in enumerate(out):
                if not a.fallback or compression_ratio(h) <= CR_MAX:
                    continue
                n_fb += 1
                for t in FALLBACK_T:
                    torch.manual_seed(a.seed * 100003 + (n_ch + i + k) * 10 + int(t * 10))
                    with torch.no_grad():
                        c = model.generate(x[k:k + 1], do_sample=True, temperature=t, num_beams=1, **g)
                    cand = proc.batch_decode(c, skip_special_tokens=True)[0].strip()
                    if compression_ratio(cand) < compression_ratio(out[k]):
                        out[k] = cand
                    if compression_ratio(cand) <= CR_MAX:
                        break
            texts += out
            if (i // bs) % 20 == 19:
                print(f"    {iv}: {len(texts)}/{len(chunks)} Stücke, {time.time()-t0:.0f} s, Rückfälle {n_fb}", flush=True)
        n_ch += len(chunks)
        (out_dir / f"{iv}.json").write_text(json.dumps({"interview": iv, "model": a.model, "adapter": a.adapter, "beams": a.beams,
                                                        "fallback": a.fallback, "prompts": a.prompts, "device": a.device,
                                                        "dtype": str(model.dtype).replace("torch.", ""),
                                                        "chunks": [[s, e, t] for (s, e), t in zip(chunks, texts)]},
                                                       ensure_ascii=False), encoding="utf-8")
        print(f"  {iv}: {len(chunks)} Stücke, {sum(e - s for s, e in chunks)/60:.0f} min Audio, "
              f"bisher {time.time()-t0:.0f} s, Rückfälle {n_fb}", flush=True)
    print(f"LONGFORM {a.name} {a.tag}: {len(files)} Interviews, {n_ch} Stücke, {time.time()-t0:.0f} s, {n_fb} Rückfälle -> {out_dir}")


if __name__ == "__main__":
    main()
