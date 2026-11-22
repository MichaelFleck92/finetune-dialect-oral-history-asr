#!/usr/bin/env python3
"""
WER of a Whisper model on one part of the dataset (default: test).

  python3 training/whisper_eval.py --model openai/whisper-tiny --name baseline-tiny
  python3 training/whisper_eval.py --model openai/whisper-tiny --adapter training/runs/tiny-lora --name tiny-lora

German, without timestamps, greedy by default (--beams 5: beam search); on an
Nvidia GPU (cuda) in bf16, else in fp32. --fallback: outputs that repeat
themselves (gzip compression ratio > 2.4, as in openai-whisper) are decoded
again with temperature 0.2 ... 1.0; the first one without repetition is kept,
else the least repetitive. --prompts: Whisper prompt per project (text before the
transcript); without --prompts everything decodes as before. The reference is `text`. WER and CER over the
whole part (sum of errors / sum of reference words or characters), plus
per project; both sides normalised as in whisper_common.words.
-> training/runs/<name>/eval_<split>[_<tag>].json and predictions_<split>[_<tag>].csv
"""
import argparse, csv, json, os, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import RUNS, SR, default_device, split_rows, load_audio, words, load_model

PROJECTS = {"rs": "Roter Stern", "re": "Rechnitz", "za": "Zwangsarbeiter"}
CR_MAX = 2.4                                   # openai-whisper: above this, an output counts as a loop
FALLBACK_T = (0.2, 0.4, 0.6, 0.8, 1.0)


def compression_ratio(text):
    b = text.encode("utf-8")
    return len(b) / len(zlib.compress(b)) if b else 0.0


def score(pairs):
    """[(reference, hypothesis)] -> WER, CER, number of reference words."""
    we = wn = ce = cn = 0
    for ref, hyp in pairs:
        r, h = words(ref), words(hyp)
        we += Levenshtein.distance(r, h); wn += len(r)
        ce += Levenshtein.distance(" ".join(r), " ".join(h)); cn += len(" ".join(r))
    return {"wer": round(we / max(1, wn), 4), "cer": round(ce / max(1, cn), 4), "ref_words": wn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-tiny")
    ap.add_argument("--adapter", default=None, help="Ordner mit lora.pt / lora.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="nur eine feste Zufallsauswahl")
    ap.add_argument("--batch", type=int, default=32, help="bei Strahlsuche geteilt durch --beams")
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--fallback", action="store_true", help="Wiederholungsschleifen neu dekodieren")
    ap.add_argument("--seed", type=int, default=0, help="fuer das Ziehen beim Rueckfall")
    ap.add_argument("--tag", default="", help="Zusatz im Dateinamen, z. B. beam5")
    ap.add_argument("--prompts", default=None, help="JSON {Projekt: Prompttext}; Prompt je Projekt")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    suffix = f"_{a.tag}" if a.tag else ""
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if a.device == "cuda" else None
    proc, model = load_model(a.model, a.adapter, a.device, dtype)
    rows = split_rows(a.split, a.limit)
    print(f"{a.model}{' + ' + a.adapter if a.adapter else ''} auf {a.split}: {len(rows)} Segmente, "
          f"{sum(r['duration'] for r in rows)/3600:.2f} h, Geraet {a.device} {model.dtype}", flush=True)
    t0, n_fb = time.time(), 0
    bs = max(1, a.batch // a.beams)
    gen = dict(language="de", task="transcribe", max_new_tokens=200)
    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8")) if a.prompts else None
    if prompts:                                          # one prompt per project -> batches from one project only
        batches = []
        for p in sorted({r["project"] for r in rows}):
            idx = [j for j, r in enumerate(rows) if r["project"] == p]
            pid = proc.get_prompt_ids(prompts[p], return_tensors="pt").to(a.device)
            batches += [(idx[i:i + bs], pid) for i in range(0, len(idx), bs)]
    else:
        batches = [(list(range(i, min(i + bs, len(rows)))), None) for i in range(0, len(rows), bs)]
    hyps, done = [None] * len(rows), 0
    with ThreadPoolExecutor(8) as pool:                  # ffmpeg decoding in parallel
        for bi, (idx, pid) in enumerate(batches):
            batch = [rows[j] for j in idx]
            audio = list(pool.map(load_audio, batch))
            feats = proc.feature_extractor(audio, sampling_rate=SR, return_tensors="pt").input_features
            x = feats.to(a.device, model.dtype)
            g = dict(gen, prompt_ids=pid) if pid is not None else gen
            with torch.no_grad():
                ids = model.generate(x, num_beams=a.beams, **g)
            out_txt = [h.strip() for h in proc.batch_decode(ids, skip_special_tokens=True)]
            for k, h in enumerate(out_txt):
                if not a.fallback or compression_ratio(h) <= CR_MAX:
                    continue
                n_fb += 1
                for t in FALLBACK_T:
                    torch.manual_seed(a.seed * 100003 + idx[k] * 10 + int(t * 10))
                    with torch.no_grad():
                        c = model.generate(x[k:k + 1], do_sample=True, temperature=t, num_beams=1, **g)
                    cand = proc.batch_decode(c, skip_special_tokens=True)[0].strip()
                    if compression_ratio(cand) < compression_ratio(out_txt[k]):
                        out_txt[k] = cand
                    if compression_ratio(cand) <= CR_MAX:
                        break
            for k, j in enumerate(idx):
                hyps[j] = out_txt[k]
            done += len(idx)
            if bi % 10 == 0:
                print(f"  {done}/{len(rows)}  {time.time()-t0:.0f} s", flush=True)
    decoding = (("greedy" if a.beams == 1 else f"Strahlsuche {a.beams}") + ", de, ohne Zeitmarken"
                + (f", Rueckfall bei Wiederholung (Kompressionsrate > {CR_MAX})" if a.fallback else ""))
    res = {"model": a.model, "adapter": a.adapter, "split": a.split, "segments": len(rows),
           "hours": round(sum(r["duration"] for r in rows) / 3600, 3), "decoding": decoding,
           "beams": a.beams, "fallback": a.fallback, "fallbacks": n_fb, "prompts": a.prompts,
           "device": a.device, "dtype": str(model.dtype).replace("torch.", ""),
           "seconds": round(time.time() - t0, 1), "gesamt": score([(r["text"], h) for r, h in zip(rows, hyps)]),
           "je_projekt": {PROJECTS[p]: score([(r["text"], h) for r, h in zip(rows, hyps) if r["project"] == p])
                          for p in PROJECTS if any(r["project"] == p for r in rows)}}
    (out / f"eval_{a.split}{suffix}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / f"predictions_{a.split}{suffix}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file_name", "project", "referenz", "hypothese", "wer_segment"])
        for r, h in zip(rows, hyps):
            w.writerow([r["file_name"], r["project"], r["text"], h, score([(r["text"], h)])["wer"]])
    g = res["gesamt"]
    print(f"WER {100*g['wer']:.1f} %  CER {100*g['cer']:.1f} %  ({g['ref_words']} Referenzwoerter, "
          f"{res['seconds']:.0f} s{f', {n_fb} Rueckfaelle' if a.fallback else ''})")
    for p, s in res["je_projekt"].items():
        print(f"  {p:15} WER {100*s['wer']:.1f} %  CER {100*s['cer']:.1f} %  ({s['ref_words']} Woerter)")


if __name__ == "__main__":
    main()
