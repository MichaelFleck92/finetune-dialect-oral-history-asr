#!/usr/bin/env python3
"""
Gemeinsames fuer Whisper-Auswertung und LoRA-Feintuning (training/).

Gelesen wird nur der fertige Datensatz (dataset/metadata.jsonl und die FLACs),
nicht work/ - fuer Training und Auswertung genuegt der Ordner dataset/.
LoRA ist hier selbst geschrieben (eine Klasse, s.u.), damit kein weiteres Paket
noetig ist; das Prinzip ist dasselbe wie bei peft.
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
from align import norm_word        # dieselbe Normalisierung wie eval_segments.py

# Das Transkript schreibt keine Fuellwoerter (text), Whisper manchmal schon
FILLERS = {"äh", "ähm", "öhm", "ehm", "ahm", "hm", "hmm", "mhm", "mh", "äähm", "ähh", "hmhm"}


def default_device():
    """Nvidia-GPU (cuda), sonst Apple-GPU (mps), sonst cpu."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def split_rows(split, limit=None, seed=0):
    """Segmente eines Teils (train/dev/test), optional eine feste Zufallsauswahl."""
    rows = [json.loads(l) for l in (DATASET / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == split]
    if limit and limit < len(rows):
        keep = np.random.default_rng(seed).choice(len(rows), limit, replace=False)
        rows = [rows[i] for i in sorted(keep)]
    return rows


def load_audio(row):
    """FLAC -> float32, 16 kHz mono (ffmpeg; torchaudio hat hier kein FLAC-Backend)."""
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(DATASET / row["file_name"]),
                          "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def words(text):
    """Woerter fuer die WER: wie align.norm_word (klein, ohne Satzzeichen, Zahlen
    ausgeschrieben, ss statt ß, ohne Bindestrich), dazu ohne Apostroph und ohne
    Fuellwoerter. So zaehlt "1945" gegen "neunzehnhundertfünfundvierzig" nicht
    als Fehler - das Transkript schreibt Zahlen mal so, mal so."""
    out = []
    for tok in text.split():
        for w in norm_word(tok).replace("'", "").split("|"):
            if w and w not in FILLERS:
                out.append(w)
    return out


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B(A(x)). W bleibt eingefroren, gelernt werden nur
    A (r x in) und B (out x r). B startet bei null: vor dem Training rechnet das
    Modell exakt wie das Ausgangsmodell."""

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
    """Alle Gewichte einfrieren und die genannten Linear-Schichten durch
    LoRALinear ersetzen - in Encoder und Decoder (scope "all") oder nur in einem
    von beiden ("encoder", "decoder"). -> Zahl der ersetzten Schichten."""
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
    """LoRA in die Basisgewichte einrechnen (W + scale * B A) und die Schichten
    wieder zu nn.Linear machen. Fuer die Auswertung in bf16: so wird das
    Ergebnis einmal gerundet statt Basis- und LoRA-Pfad getrennt."""
    for mod in list(model.modules()):
        for name, sub in list(mod.named_children()):
            if isinstance(sub, LoRALinear):
                sub.base.weight += sub.scale * (sub.lora_b.weight @ sub.lora_a.weight)
                setattr(mod, name, sub.base)


def load_model(base, adapter=None, device="cpu", dtype=None):
    """Whisper-Modell (+ Prozessor), optional mit einem trainierten LoRA-Adapter.
    dtype (z. B. torch.bfloat16 auf cuda): Adapter in fp32 einrechnen, dann umwandeln."""
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
