#!/usr/bin/env python3
"""
Shared code for Parakeet (NeMo) in training/: evaluation (parakeet_eval.py) and LoRA fine-tuning
(parakeet_lora.py), 16.11.2026.

Runs in its own NeMo environment (e.g. ~/venv-nemo: nemo_toolkit[asr] 3.0, torch 2.15, numba-cuda
for the TDT loss). Data, normalisation and the LoRA class come from whisper_common.py, so that both models
work with the same segments, the same reference and the same LoRA implementation.

LoRA targets (counterpart to Whisper's q/k/v/out_proj, fc1, fc2): in the FastConformer encoder the attention projections
linear_q, linear_k, linear_v, linear_out and the two feed-forward blocks per layer (linear1, linear2); in the
joint network the projections of the encoder and the prediction network (enc, pred). Not: linear_pos (relative positions,
no counterpart in Whisper), the output layer of the joint network (Whisper's output layer also stays
frozen) and the LSTM of the prediction network (not a linear layer).
"""
import json, sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transformers     # noqa: E402,F401  load before the placeholder (checks on import whether torchaudio is present)
_STUB = False
try:                    # torchaudio no longer exists for NeMo 3's torch version; pipeline/align.py
    import torchaudio   # noqa: F401  imports it, only norm_word (normalisation) is needed here
except ImportError:
    import types
    _ta = types.ModuleType("torchaudio")
    _ta.functional = types.ModuleType("torchaudio.functional")
    sys.modules.update({"torchaudio": _ta, "torchaudio.functional": _ta.functional})
    _STUB = True
from whisper_common import (ROOT, RUNS, SR, split_rows, load_audio, words,          # noqa: E402,F401
                            LoRALinear, lora_state, merge_lora)
if _STUB:               # remove the placeholder again so that NeMo sees torchaudio as not installed
    sys.modules.pop("torchaudio.functional", None)
    sys.modules.pop("torchaudio", None)

MODEL = "nvidia/parakeet-tdt-0.6b-v3"
TARGETS = "linear_q,linear_k,linear_v,linear_out,linear1,linear2,enc,pred"


def add_lora(model, targets, r, alpha, dropout, scope="all"):
    """Freeze all weights and replace the named linear layers with LoRALinear – in the encoder and
    joint network (scope "all") or only in one of them ("encoder", "joint"). -> number of replaced layers."""
    for p in model.parameters():
        p.requires_grad_(False)
    roots = {"all": [model.encoder, model.joint], "encoder": [model.encoder], "joint": [model.joint]}[scope]
    n = 0
    for root in roots:
        for _, mod in list(root.named_modules()):
            for child, sub in list(mod.named_children()):
                if child in targets and isinstance(sub, nn.Linear):
                    setattr(mod, child, LoRALinear(sub, r, alpha, dropout))
                    n += 1
    return n


def load_model(name=MODEL, adapter=None, device=None):
    """Parakeet from NeMo, optionally with a trained LoRA adapter (merged into the base weights)."""
    import nemo.collections.asr as nemo_asr
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = nemo_asr.models.ASRModel.from_pretrained(name)     # NeMo already puts the model on the GPU itself
    if adapter:
        cfg = json.loads((Path(adapter) / "lora.json").read_text(encoding="utf-8"))
        assert cfg["base"] == name, f"Adapter gehört zu {cfg['base']}, nicht zu {name}"
        add_lora(model, cfg["targets"], cfg["r"], cfg["alpha"], 0.0, cfg.get("scope", "all"))
        state = torch.load(Path(adapter) / "lora.pt")
        expected = {k for k in model.state_dict() if ".lora_" in k}
        assert expected == set(state), "Adapter passt nicht zum Modell"
        model.load_state_dict(state, strict=False)
        model.to(device)                                       # new LoRA layers on the same device
        merge_lora(model)
    return model.to(device).eval()


def texte(out):
    """Depending on the version, NeMo returns str, Hypothesis or (best, all)."""
    if isinstance(out, tuple):
        out = out[0]
    return [(o.text if hasattr(o, "text") else str(o)).strip() for o in out]
