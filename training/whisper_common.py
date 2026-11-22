#!/usr/bin/env python3
"""
Shared code for Whisper evaluation and LoRA fine-tuning (training/).

Only the finished dataset is read (dataset/metadata.jsonl and the FLACs),
not work/ - the dataset/ folder is enough for training and evaluation.
LoRA is self-written here (one class, see below), so that no further package
is needed; the principle is the same as in peft.
"""
import json, math, subprocess, sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
RUNS = ROOT / "training" / "runs"
SR = 16000

sys.path.insert(0, str(ROOT / "pipeline"))
from align import norm_word        # the same normalisation as eval_segments.py

# the transcript writes no fillers (text), Whisper sometimes does
FILLERS = {"äh", "ähm", "öhm", "ehm", "ahm", "hm", "hmm", "mhm", "mh", "äähm", "ähh", "hmhm"}


def default_device():
    """Nvidia GPU (cuda), else Apple GPU (mps), else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def split_rows(split, limit=None, seed=0):
    """Segments of one part (train/dev/test), optionally a fixed random selection."""
    rows = [json.loads(l) for l in (DATASET / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == split]
    if limit and limit < len(rows):
        keep = np.random.default_rng(seed).choice(len(rows), limit, replace=False)
        rows = [rows[i] for i in sorted(keep)]
    return rows


def load_audio(row):
    """FLAC -> float32, 16 kHz mono (ffmpeg; torchaudio has no FLAC backend here)."""
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(DATASET / row["file_name"]),
                          "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def words(text):
    """Words for the WER: as align.norm_word (lower case, no punctuation, numbers
    spelled out, ss instead of ß, no hyphen), plus without apostrophes and
    without fillers. So "1945" against "neunzehnhundertfünfundvierzig" does not
    count as an error - the transcript writes numbers sometimes one way, sometimes the other."""
    out = []
    for tok in text.split():
        for w in norm_word(tok).replace("'", "").split("|"):
            if w and w not in FILLERS:
                out.append(w)
    return out


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B(A(x)). W stays frozen, only A (r x in) and
    B (out x r) are learned. B starts at zero: before training the model
    computes exactly like the base model."""

    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(self.drop(x))) * self.scale


def add_lora(model, targets, r, alpha, dropout, scope="all"):
    """Freeze all weights and replace the named linear layers with
    LoRALinear - in the encoder and decoder (scope "all") or only in one of
    them ("encoder", "decoder"). -> number of replaced layers."""
    for p in model.parameters():
        p.requires_grad_(False)
    root = {"all": model, "encoder": model.model.encoder, "decoder": model.model.decoder}[scope]
    n = 0
    for _, mod in list(root.named_modules()):
        for child, sub in list(mod.named_children()):
            if child in targets and isinstance(sub, nn.Linear):
                setattr(mod, child, LoRALinear(sub, r, alpha, dropout))
                n += 1
    return n


def lora_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if ".lora_" in k}


@torch.no_grad()
def merge_lora(model):
    """Merge LoRA into the base weights (W + scale * B A) and turn the layers
    back into nn.Linear. For evaluation in bf16: this way the result is
    rounded once instead of the base and LoRA paths separately."""
    for mod in list(model.modules()):
        for name, sub in list(mod.named_children()):
            if isinstance(sub, LoRALinear):
                sub.base.weight += sub.scale * (sub.lora_b.weight @ sub.lora_a.weight)
                setattr(mod, name, sub.base)


def load_model(base, adapter=None, device="cpu", dtype=None):
    """Whisper model (+ processor), optionally with a trained LoRA adapter.
    dtype (e.g. torch.bfloat16 on cuda): merge the adapter in fp32, then convert."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(base)
    model = WhisperForConditionalGeneration.from_pretrained(base)
    if adapter:
        cfg = json.loads((Path(adapter) / "lora.json").read_text(encoding="utf-8"))
        add_lora(model, cfg["targets"], cfg["r"], cfg["alpha"], 0.0, cfg.get("scope", "all"))
        state = torch.load(Path(adapter) / "lora.pt")
        expected = {k for k in model.state_dict() if ".lora_" in k}
        assert expected == set(state), "Adapter passt nicht zum Modell"
        model.load_state_dict(state, strict=False)
        if dtype is not None:
            merge_lora(model)
    if dtype is not None:
        model.to(dtype)
    return proc, model.to(device).eval()
