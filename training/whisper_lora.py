#!/usr/bin/env python3
"""
LoRA fine-tuning of Whisper on the train part of the dataset.

  python3 training/whisper_lora.py --model openai/whisper-tiny --name tiny-lora --epochs 1

The base model stays frozen; only the LoRA matrices in the attention and MLP
layers of the encoder and decoder are learned (whisper_common.add_lora).
The target is `text` with the prefix <|de|><|transcribe|><|notimestamps|>. As a
check, the loss on a fixed dev selection before training and after every
epoch. -> training/runs/<name>/ (lora.pt, lora.json, log.csv)

Then: python3 training/whisper_eval.py --model <the same> --adapter training/runs/<name> --name <name> --split dev
With several epochs, every checkpoint lies in training/runs/<name>/ep<k>/ (can be evaluated like a run).
Tuning is done on dev only; test is kept for the end.

Memory (Apple GPU): the encoder compares 1500 frames pairwise; that costs
about 1 GB per segment in the batch (measured 02.10.2026: batch 2 / 4 / 8 = 2.4 / 4.7
/ 8.0 GB, same throughput). Batch 16 at once pushed 24 GB into swap.
Hence batch 4 times 4 gradient steps (acts like batch 16), and the
MPS memory limit is lowered: better an error than a swapping machine.

Nvidia GPU (cuda): base model in bf16, LoRA weights in fp32, computed
under autocast (bf16). --grad-ckpt stores only the inputs per layer and
recomputes the rest in the backward pass (slower, much less memory).
Measured 02.10.2026, whisper-large-v3 on an Nvidia GPU with 24 GB, 16 segments per step:
batch 2 x 8: 5.3 s, 11.8 GB; batch 16 with --grad-ckpt: 5.5 s, 8.1 GB;
batch 4 x 4: 4.1 s, 20.0 GB (longest transcripts: 20.1 GB) -> 4 x 4.
"""
import argparse, contextlib, csv, functools, json, os, sys, time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")   # share of the recommended GPU memory
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less waste on cuda
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whisper_common import RUNS, SR, default_device, split_rows, load_audio, add_lora, lora_state

TARGETS = "q_proj,k_proj,v_proj,out_proj,fc1,fc2"


class Segments(Dataset):
    """Segment -> (log-mel features 80 x 3000, token IDs of the target text)."""

    def __init__(self, rows, proc):
        self.rows, self.fe, self.tok = rows, proc.feature_extractor, proc.tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        feats = self.fe(load_audio(r), sampling_rate=SR, return_tensors="np").input_features[0]
        return feats, self.tok(r["text"]).input_ids


def collate(batch, sot):
    feats = torch.from_numpy(np.stack([f for f, _ in batch]))
    labels = torch.full((len(batch), max(len(ids) for _, ids in batch)), -100, dtype=torch.long)
    for k, (_, ids) in enumerate(batch):
        labels[k, :len(ids)] = torch.tensor(ids)
    if bool((labels[:, 0] == sot).all()):
        labels = labels[:, 1:]                   # the model itself puts <|startoftranscript|> in front
    return feats, labels


def autocast(device):
    """bf16 computation on cuda, nothing otherwise (MPS/CPU compute in fp32)."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()


@torch.no_grad()
def dev_loss(model, loader, device):
    """Mean loss per target token on the dev selection."""
    model.eval()
    tot = n = 0
    for feats, labels in loader:
        k = int((labels != -100).sum())
        with autocast(device):
            loss = model(input_features=feats.to(device), labels=labels.to(device)).loss
        tot += float(loss) * k
        n += k
    model.train()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-tiny")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4, help="Segmente je Durchgang (Speicher!)")
    ap.add_argument("--accum", type=int, default=4, help="Durchgaenge je Optimierungsschritt")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", default=TARGETS)
    ap.add_argument("--scope", default="all", choices=["all", "encoder", "decoder"],
                    help="LoRA in Encoder und Decoder oder nur in einem von beiden")
    ap.add_argument("--spec-augment", action="store_true",
                    help="SpecAugment: im Training zufaellig Zeit- und Frequenzbereiche der Merkmale ausblenden")
    ap.add_argument("--splits", default="train", help="Trainingsteile, z. B. train,dev fuer einen Schlusslauf")
    ap.add_argument("--limit", type=int, default=None, help="nur eine feste Zufallsauswahl der Trainingsteile")
    ap.add_argument("--dev-limit", type=int, default=256, help="0 = kein dev-Verlust")
    ap.add_argument("--workers", type=int, default=min(6, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true", help="Gradient-Checkpointing (Speicher sparen)")
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    torch.manual_seed(a.seed)
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)
    proc = WhisperProcessor.from_pretrained(a.model)
    proc.tokenizer.set_prefix_tokens(language="german", task="transcribe", predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(a.model)
    if a.device == "cuda":
        model.to(torch.bfloat16)                 # frozen base; LoRA is created in fp32 afterwards
    if a.spec_augment:                           # masks according to the model configuration, active only in training
        model.config.apply_spec_augment = True
    targets = a.targets.split(",")
    n_layers = add_lora(model, targets, a.rank, a.alpha, a.dropout, a.scope)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(a.device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    n_train, n_all = sum(p.numel() for p in params), sum(p.numel() for p in model.parameters())
    splits = a.splits.split(",")
    train_rows = [r for sp in splits for r in split_rows(sp)]
    if a.limit and a.limit < len(train_rows):
        keep = np.random.default_rng(a.seed).choice(len(train_rows), a.limit, replace=False)
        train_rows = [train_rows[i] for i in sorted(keep)]
    use_dev = a.dev_limit > 0 and "dev" not in splits
    dev_rows = split_rows("dev", a.dev_limit, a.seed) if use_dev else []
    col = functools.partial(collate, sot=proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"))
    extra = {"num_workers": a.workers, "persistent_workers": True, "prefetch_factor": 4} if a.workers else {}
    extra["pin_memory"] = a.device == "cuda"
    tl = DataLoader(Segments(train_rows, proc), batch_size=a.batch, shuffle=True, collate_fn=col,
                    generator=torch.Generator().manual_seed(a.seed), **extra)
    dl = DataLoader(Segments(dev_rows, proc), batch_size=a.batch, collate_fn=col, **extra) if use_dev else None
    total = a.epochs * -(-len(tl) // a.accum)        # optimisation steps
    warm = max(1, min(100, total // 10))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    print(f"{a.model}: LoRA r={a.rank} in {n_layers} Schichten ({a.scope}), {n_train:,} von {n_all:,} Parametern "
          f"trainierbar ({100*n_train/n_all:.1f} %); {'+'.join(splits)} {len(train_rows)} Segmente "
          f"({sum(r['duration'] for r in train_rows)/3600:.2f} h), {total} Schritte zu {a.batch} x {a.accum} "
          f"Segmenten; Geraet {a.device}{', Gradient-Checkpointing' if a.grad_ckpt else ''}"
          f"{', SpecAugment' if a.spec_augment else ''}", flush=True)
    d0 = dev_loss(model, dl, a.device) if use_dev else None
    if use_dev:
        print(f"dev-Verlust vor dem Training: {d0:.3f}", flush=True)
    cfg = {"base": a.model, "r": a.rank, "alpha": a.alpha, "dropout": a.dropout, "targets": targets,
           "scope": a.scope, "spec_augment": a.spec_augment, "splits": splits,
           "epochs": a.epochs, "batch": a.batch, "accum": a.accum, "lr": a.lr, "steps": total, "seed": a.seed,
           "device": a.device, "base_dtype": "bfloat16" if a.device == "cuda" else "float32", "grad_ckpt": a.grad_ckpt,
           "train_segments": len(train_rows), "train_hours": round(sum(r["duration"] for r in train_rows) / 3600, 3),
           "trainable_params": n_train, "dev_loss_before": None if d0 is None else round(d0, 4)}

    def save(folder, **more):
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(lora_state(model), folder / "lora.pt")
        (folder / "lora.json").write_text(json.dumps(
            {**cfg, **more, "seconds": round(time.time() - t0, 1), "created": time.strftime("%Y-%m-%d %H:%M")},
            ensure_ascii=False, indent=1), encoding="utf-8")

    log, recent, step, t0, d = [], [], 0, time.time(), None
    for ep in range(a.epochs):
        for i, (feats, labels) in enumerate(tl):
            with autocast(a.device):
                loss = model(input_features=feats.to(a.device), labels=labels.to(a.device), use_cache=False).loss
            (loss / a.accum).backward()
            recent.append(float(loss))
            if (i + 1) % a.accum and i + 1 < len(tl):
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or step == total:
                el = time.time() - t0
                mem = (f", GPU {torch.mps.driver_allocated_memory()/1e9:.1f} GB" if a.device == "mps" else
                       f", GPU max {torch.cuda.max_memory_allocated()/1e9:.1f} GB" if a.device == "cuda" else "")
                log.append({"schritt": step, "epoche": ep + 1, "verlust": round(float(np.mean(recent)), 4),
                            "lr": round(sched.get_last_lr()[0], 7), "sekunden": round(el, 1)})
                print(f"  Schritt {step}/{total}  Verlust {np.mean(recent):.3f}  "
                      f"{el/step:.2f} s/Schritt, noch ~{(total-step)*el/step/60:.0f} min{mem}", flush=True)
                recent = []
        if use_dev:
            d = dev_loss(model, dl, a.device)
            log.append({"schritt": step, "epoche": ep + 1, "dev_verlust": round(d, 4), "sekunden": round(time.time() - t0, 1)})
            print(f"Epoche {ep+1}: dev-Verlust {d:.3f} (vorher {d0:.3f})", flush=True)
        if a.epochs > 1:                         # checkpoint per epoch, can be evaluated like a run of its own
            save(out / f"ep{ep+1}", epoch=ep + 1, dev_loss_after=None if d is None else round(d, 4))
    save(out, epoch=a.epochs, dev_loss_after=None if d is None else round(d, 4))
    with open(out / "log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["schritt", "epoche", "verlust", "lr", "dev_verlust", "sekunden"])
        w.writeheader(); w.writerows(log)
    print(f"-> {out}/lora.pt ({(out / 'lora.pt').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
