#!/usr/bin/env python3
"""
LoRA-Feintuning von Parakeet (NVIDIA nvidia/parakeet-tdt-0.6b-v3, NeMo) auf dem train-Teil des Datensatzes –
Gegenstück zu whisper_lora.py für den Vergleich Whisper/Parakeet, je ohne und mit Feintuning (16.11.2026).

  ~/venv-nemo/bin/python training/parakeet_lora.py --name pk-r16-lr5e-4-e3 --epochs 3 --lr 5e-4

Gleich wie bei Whisper: dieselben Segmente (dataset/metadata.jsonl, Audio über ffmpeg wie bei der Auswertung),
Ziel ist derselbe Text (`text`, mit Groß-/Kleinschreibung und Satzzeichen; Parakeets Tokenizer kennt beides),
dieselbe LoRA-Klasse (whisper_common.LoRALinear, r = 16, alpha = 32, Dropout 0,05), AdamW (weight decay 0,01),
lineares Aufwärmen (höchstens 100 Schritte) und lineares Abklingen, Gradienten-Norm auf 1 begrenzt, je Schritt
16 Segmente, dev-Verlust auf derselben festen Auswahl (256 Segmente) vor dem Training und nach jeder Epoche,
Zwischenstand je Epoche in training/runs/<name>/ep<k>/. Abgestimmt wird nur auf dev; test bleibt für den Schluss.

Parakeet-spezifisch: LoRA in Encoder (Attention + Feed-Forward) und Joint-Netz (parakeet_common.TARGETS).
Verlust ist der TDT-Transducer-Verlust des Modells (NeMo, über numba-cuda), gemittelt je Ziel-Token
("mean_volume", wie die Kreuzentropie bei Whisper; NeMo-Standard wäre je Segment). Das Joint-Netz rechnet in
Teilbatches (NeMo "fused", --fused-batch). Basis bleibt in fp32 eingefroren, gerechnet unter autocast (bf16).
BatchNorm in den Faltungsmodulen bleibt im Auswertungsmodus (die gelernten Statistiken der Basis bleiben
unverändert, sonst würde das eingefrorene Modell trotzdem driften). Dropout des Modells bleibt wie konfiguriert
(Encoder 0,1, Joint 0,2). SpecAugment nur mit --spec-augment (Whisper-Endmodell: ohne).
Ausgegeben werden nur Zahlen. -> training/runs/<name>/ (lora.pt, lora.json, log.csv)

Danach: ~/venv-nemo/bin/python training/parakeet_eval.py --modus segmente --split dev --adapter training/runs/<name> --name <name>
Probe: --max-steps N bricht nach N Optimierungsschritten ab (Zeit, Speicher, Verlauf prüfen).
"""
import argparse, csv, json, os, sys, time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if not hasattr(np, "row_stack"):     # numba-cuda 0.30 (TDT-Verlust) ruft np.row_stack, in NumPy 2.4 entfernt
    np.row_stack = np.vstack

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parakeet_common import MODEL, RUNS, TARGETS, split_rows, load_audio, add_lora, lora_state   # noqa: E402


class Segments(Dataset):
    """Segment -> (Audio float32 16 kHz, Token-Ids des Zieltexts). Tokenisiert wird vorab im Hauptprozess
    (der NeMo-Tokenizer muss so nicht in die Worker-Prozesse)."""

    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.ids = [tokenizer.text_to_ids(r["text"]) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return load_audio(self.rows[i]), self.ids[i]


def collate(batch):
    sig_len = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
    sig = torch.zeros(len(batch), int(sig_len.max()))
    tr_len = torch.tensor([len(ids) for _, ids in batch], dtype=torch.long)
    tr = torch.zeros(len(batch), max(1, int(tr_len.max())), dtype=torch.long)    # Auffüllen mit 0 wie NeMo
    for k, (x, ids) in enumerate(batch):
        sig[k, :len(x)] = torch.from_numpy(x)
        tr[k, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return sig, sig_len, tr, tr_len


def batch_loss(model, batch, device):
    """TDT-Verlust je Ziel-Token (mean_volume) und Zahl der Ziel-Token."""
    sig, sig_len, tr, tr_len = (t.to(device, non_blocking=True) for t in batch)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        enc, enc_len = model.forward(input_signal=sig, input_signal_length=sig_len)
        dec, _, _ = model.decoder(targets=tr, target_length=tr_len)
        loss, _, _, _ = model.joint(encoder_outputs=enc, decoder_outputs=dec, encoder_lengths=enc_len,
                                    transcripts=tr, transcript_lengths=tr_len, compute_wer=False)
    return loss, int(tr_len.sum())


def set_train(model, on):
    """train/eval umschalten; BatchNorm bleibt immer im Auswertungsmodus (eingefrorene Statistiken)."""
    model.train(on)
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


@torch.no_grad()
def dev_loss(model, loader, device):
    """Mittlerer Verlust je Ziel-Token auf der dev-Auswahl. Der TDT-Verlust nimmt im Training mit
    Wahrscheinlichkeit omega (Modell: 0,1) je Teilbatch den gewöhnlichen RNN-T-Verlust; für einen
    vergleichbaren dev-Verlust ist omega hier 0."""
    set_train(model, False)
    tdt = model.joint.loss._loss
    omega, tdt.omega = tdt.omega, 0.0
    tot = n = 0
    for batch in loader:
        loss, k = batch_loss(model, batch, device)
        tot += float(loss) * k
        n += k
    tdt.omega = omega
    set_train(model, True)
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4, help="Segmente je Durchgang (Speicher!)")
    ap.add_argument("--accum", type=int, default=4, help="Durchgänge je Optimierungsschritt")
    ap.add_argument("--fused-batch", type=int, default=2,
                    help="Teilbatch für Joint-Netz + Verlust (Speicher! längste 4 Segmente: 2 -> 17,6 GB, 4 -> zu viel)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", default=TARGETS)
    ap.add_argument("--scope", default="all", choices=["all", "encoder", "joint"])
    ap.add_argument("--spec-augment", action="store_true", help="SpecAugment laut Modellkonfiguration")
    ap.add_argument("--splits", default="train", help="Trainingsteile, z. B. train,dev für einen Schlusslauf")
    ap.add_argument("--limit", type=int, default=None, help="nur eine feste Zufallsauswahl der Trainingsteile")
    ap.add_argument("--dev-limit", type=int, default=256, help="0 = kein dev-Verlust")
    ap.add_argument("--max-steps", type=int, default=None, help="Probe: nach so vielen Schritten aufhören")
    ap.add_argument("--workers", type=int, default=min(6, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import nemo.collections.asr as nemo_asr
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)
    model = nemo_asr.models.ASRModel.from_pretrained(a.model)
    if not a.spec_augment:
        model.spec_augmentation = None
    model.joint.set_fused_batch_size(a.fused_batch)
    model.joint.loss.reduction = "mean_volume"
    targets = a.targets.split(",")
    n_layers = add_lora(model, targets, a.rank, a.alpha, a.dropout, a.scope)
    model.to(device)
    set_train(model, True)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train, n_all = sum(p.numel() for p in params), sum(p.numel() for p in model.parameters())
    splits = a.splits.split(",")
    train_rows = [r for sp in splits for r in split_rows(sp)]
    if a.limit and a.limit < len(train_rows):
        keep = np.random.default_rng(a.seed).choice(len(train_rows), a.limit, replace=False)
        train_rows = [train_rows[i] for i in sorted(keep)]
    use_dev = a.dev_limit > 0 and "dev" not in splits
    dev_rows = split_rows("dev", a.dev_limit, a.seed) if use_dev else []
    extra = {"num_workers": a.workers, "persistent_workers": True, "prefetch_factor": 4} if a.workers else {}
    extra["pin_memory"] = device == "cuda"
    tl = DataLoader(Segments(train_rows, model.tokenizer), batch_size=a.batch, shuffle=True, collate_fn=collate,
                    generator=torch.Generator().manual_seed(a.seed), **extra)
    dl = DataLoader(Segments(dev_rows, model.tokenizer), batch_size=a.batch, collate_fn=collate, **extra) if use_dev else None
    total = a.epochs * -(-len(tl) // a.accum)        # Optimierungsschritte
    warm = max(1, min(100, total // 10))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    print(f"{a.model}: LoRA r={a.rank} in {n_layers} Schichten ({a.scope}), {n_train:,} von {n_all:,} Parametern "
          f"trainierbar ({100*n_train/n_all:.1f} %); {'+'.join(splits)} {len(train_rows)} Segmente "
          f"({sum(r['duration'] for r in train_rows)/3600:.2f} h), {total} Schritte zu {a.batch} x {a.accum} "
          f"Segmenten (Joint-Teilbatch {a.fused_batch}); Gerät {device}"
          f"{', SpecAugment' if a.spec_augment else ''}"
          f"{f', Probe: Abbruch nach {a.max_steps} Schritten' if a.max_steps else ''}", flush=True)
    t0 = time.time()
    d0 = dev_loss(model, dl, device) if use_dev else None
    if use_dev:
        print(f"dev-Verlust vor dem Training: {d0:.3f} ({time.time() - t0:.0f} s)", flush=True)
    cfg = {"base": a.model, "r": a.rank, "alpha": a.alpha, "dropout": a.dropout, "targets": targets,
           "scope": a.scope, "spec_augment": a.spec_augment, "splits": splits,
           "epochs": a.epochs, "batch": a.batch, "accum": a.accum, "fused_batch": a.fused_batch, "lr": a.lr,
           "steps": total, "max_steps": a.max_steps, "seed": a.seed, "device": device, "base_dtype": "float32",
           "autocast": "bfloat16" if device == "cuda" else None, "loss": "tdt, mean_volume",
           "train_segments": len(train_rows), "train_hours": round(sum(r["duration"] for r in train_rows) / 3600, 3),
           "trainable_params": n_train, "dev_loss_before": None if d0 is None else round(d0, 4)}

    def save(folder, **more):
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(lora_state(model), folder / "lora.pt")
        (folder / "lora.json").write_text(json.dumps(
            {**cfg, **more, "seconds": round(time.time() - t0, 1), "created": time.strftime("%Y-%m-%d %H:%M")},
            ensure_ascii=False, indent=1), encoding="utf-8")

    log, recent, step, d, t1, stop = [], [], 0, None, time.time(), False
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(a.epochs):
        for i, batch in enumerate(tl):
            loss, _ = batch_loss(model, batch, device)
            (loss / a.accum).backward()
            recent.append(float(loss))
            if (i + 1) % a.accum and i + 1 < len(tl):
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            stop = bool(a.max_steps and step >= a.max_steps)
            if step % 25 == 0 or step == total or stop:
                el = time.time() - t1
                mem = f", GPU max {torch.cuda.max_memory_allocated()/1e9:.1f} GB" if device == "cuda" else ""
                log.append({"schritt": step, "epoche": ep + 1, "verlust": round(float(np.mean(recent)), 4),
                            "lr": round(sched.get_last_lr()[0], 7), "sekunden": round(el, 1)})
                print(f"  Schritt {step}/{total}  Verlust {np.mean(recent):.3f}  "
                      f"{el/step:.2f} s/Schritt, noch ~{(total-step)*el/step/60:.0f} min{mem}", flush=True)
                recent = []
            if stop:
                break
        if use_dev:
            d = dev_loss(model, dl, device)
            log.append({"schritt": step, "epoche": ep + 1, "dev_verlust": round(d, 4), "sekunden": round(time.time() - t0, 1)})
            print(f"Epoche {ep+1}: dev-Verlust {d:.3f} (vorher {d0:.3f})", flush=True)
        if a.epochs > 1 and not stop:            # Zwischenstand je Epoche, auswertbar wie ein eigener Lauf
            save(out / f"ep{ep+1}", epoch=ep + 1, dev_loss_after=None if d is None else round(d, 4))
        if stop:
            break
    save(out, epoch=ep + 1, steps_done=step, dev_loss_after=None if d is None else round(d, 4))
    with open(out / "log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["schritt", "epoche", "verlust", "lr", "dev_verlust", "sekunden"])
        w.writeheader(); w.writerows(log)
    print(f"-> {out}/lora.pt ({(out / 'lora.pt').stat().st_size/1e6:.1f} MB), {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
