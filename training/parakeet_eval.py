#!/usr/bin/env python3
"""
NVIDIA Parakeet (NeMo, e.g. nvidia/parakeet-tdt-0.6b-v3) on the same data as Whisper (15.11.2026).

  --modus segmente:   test segments as in whisper_eval.py
                      -> training/runs/<name>/eval_<split>_<tag>.json, predictions_<split>_<tag>.csv
  --modus interviews: complete interviews, the same chunks from longform_chunks.py as longform_eval.py
                      -> training/runs/<name>/longform_<tag>/<interview>.json ({"chunks": [[s, e, text], ...]})

Decoding: the model's NeMo default (TDT greedy, language detected automatically; Parakeet has no
language setting), without a list and without fine-tuning. Reference, normalisation and WER as in whisper_eval.py
(whisper_common.words); scoring afterwards with entity_eval.py, longform_score.py,
longform_entities.py as for Whisper. Runs in its own environment with nemo_toolkit[asr]
(requirements-nemo.txt, e.g. ~/venv-nemo). Only figures are printed.

  ~/venv-nemo/bin/python training/parakeet_eval.py --modus segmente
  ~/venv-nemo/bin/python training/parakeet_eval.py --modus interviews

With a LoRA adapter from parakeet_lora.py (16.11.2026; merged into the base weights):
  ~/venv-nemo/bin/python training/parakeet_eval.py --modus segmente --split dev \
      --adapter training/runs/<lauf> --name <lauf>
"""
import argparse, csv, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parakeet_common import MODEL, ROOT, RUNS, SR, split_rows, load_audio, load_model, texte   # noqa: E402
from whisper_eval import score, PROJECTS                             # noqa: E402
from longform_eval import LF, interview_audio                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modus", required=True, choices=["segmente", "interviews"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter", default=None, help="Ordner mit lora.pt/lora.json aus parakeet_lora.py")
    ap.add_argument("--name", default="parakeet-tdt-0.6b-v3", help="Ausgabe nach training/runs/<name>/")
    ap.add_argument("--tag", default="greedy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="nur eine feste Zufallsauswahl (Test)")
    ap.add_argument("--interviews", default=None, help="Komma-Liste; Standard: alle in training/longform/chunks")
    ap.add_argument("--audio-dir", default=None, help="Ordner mit <interview>.wav/.flac (Standard work/audio16k)")
    a = ap.parse_args()

    import torch
    model = load_model(a.model, a.adapter)
    dev = next(model.parameters()).device
    print(f"{a.model}{' + ' + a.adapter if a.adapter else ''} auf {dev}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f} Mio. Parameter", flush=True)
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if a.modus == "segmente":
        rows = split_rows(a.split, a.limit)
        print(f"{a.split}: {len(rows)} Segmente, {sum(r['duration'] for r in rows) / 3600:.2f} h", flush=True)
        with ThreadPoolExecutor(8) as pool:
            audio = list(pool.map(load_audio, rows))
        hyps = []
        for i in range(0, len(rows), 256):
            with torch.inference_mode():
                hyps += texte(model.transcribe(audio[i:i + 256], batch_size=a.batch, verbose=False))
            print(f"  {len(hyps)}/{len(rows)}  {time.time() - t0:.0f} s", flush=True)
        res = {"model": a.model, "adapter": a.adapter, "split": a.split, "segments": len(rows),
               "hours": round(sum(r["duration"] for r in rows) / 3600, 3),
               "decoding": "NeMo-Standard (TDT greedy), Sprache automatisch, ohne Liste",
               "device": str(dev), "seconds": round(time.time() - t0, 1),
               "gesamt": score([(r["text"], h) for r, h in zip(rows, hyps)]),
               "je_projekt": {PROJECTS[p]: score([(r["text"], h) for r, h in zip(rows, hyps) if r["project"] == p])
                              for p in PROJECTS if any(r["project"] == p for r in rows)}}
        (out / f"eval_{a.split}_{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        with open(out / f"predictions_{a.split}_{a.tag}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file_name", "project", "referenz", "hypothese", "wer_segment"])
            for r, h in zip(rows, hyps):
                w.writerow([r["file_name"], r["project"], r["text"], h, score([(r["text"], h)])["wer"]])
        g = res["gesamt"]
        print(f"WER {100 * g['wer']:.1f} %  CER {100 * g['cer']:.1f} %  ({g['ref_words']} Referenzwörter, {res['seconds']:.0f} s)")
        for p, s in res["je_projekt"].items():
            print(f"  {p:15} WER {100 * s['wer']:.1f} %  CER {100 * s['cer']:.1f} %  ({s['ref_words']} Wörter)")
        return

    out_dir = out / f"longform_{a.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((LF / "chunks").glob("*.json"))
    if a.interviews:
        files = [f for f in files if f.stem in a.interviews.split(",")]
    n_ch = 0
    for f in files:
        meta = json.loads(f.read_text(encoding="utf-8"))
        iv, chunks = meta["interview"], meta["chunks"]
        audio = interview_audio(iv, a.audio_dir)
        seg = [audio[int(s * SR):int(e * SR)] for s, e in chunks]
        with torch.inference_mode():
            txt = texte(model.transcribe(seg, batch_size=a.batch, verbose=False))
        n_ch += len(chunks)
        (out_dir / f"{iv}.json").write_text(json.dumps({"interview": iv, "model": a.model, "adapter": a.adapter,
                                                        "decoding": "NeMo-Standard (TDT greedy)", "device": str(dev),
                                                        "chunks": [[s, e, t] for (s, e), t in zip(chunks, txt)]},
                                                       ensure_ascii=False), encoding="utf-8")
        print(f"  {iv}: {len(chunks)} Stücke, {sum(e - s for s, e in chunks) / 60:.0f} min Audio, "
              f"bisher {time.time() - t0:.0f} s", flush=True)
    print(f"LONGFORM {a.name} {a.tag}: {len(files)} Interviews, {n_ch} Stücke, {time.time() - t0:.0f} s -> {out_dir}")


if __name__ == "__main__":
    main()
