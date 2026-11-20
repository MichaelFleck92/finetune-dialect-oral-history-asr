#!/usr/bin/env python3
"""
Gemeinsames für Parakeet (NeMo) in training/: Auswertung (parakeet_eval.py) und LoRA-Feintuning
(parakeet_lora.py), 16.11.2026.

Läuft in der eigenen NeMo-Umgebung (z. B. ~/venv-nemo: nemo_toolkit[asr] 3.0, torch 2.15, numba-cuda
für den TDT-Verlust). Daten, Normalisierung und LoRA-Klasse kommen aus whisper_common.py, damit beide Modelle
mit denselben Segmenten, derselben Referenz und derselben LoRA-Umsetzung arbeiten.

LoRA-Ziele (Gegenstück zu Whisper q/k/v/out_proj, fc1, fc2): im FastConformer-Encoder die Attention-Projektionen
linear_q, linear_k, linear_v, linear_out und die beiden Feed-Forward-Blöcke je Schicht (linear1, linear2); im
Joint-Netz die Projektionen von Encoder und Prediction-Netz (enc, pred). Nicht: linear_pos (relative Positionen,
bei Whisper kein Gegenstück), die Ausgabeschicht des Joint-Netzes (bei Whisper bleibt die Ausgabeschicht ebenfalls
fest) und das LSTM des Prediction-Netzes (keine Linear-Schicht).
"""
import json, sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transformers     # noqa: E402,F401  vor dem Platzhalter laden (prüft beim Import, ob torchaudio da ist)
_STUB = False
try:                    # torchaudio gibt es für die torch-Version von NeMo 3 nicht mehr; pipeline/align.py
    import torchaudio   # noqa: F401  importiert es, gebraucht wird hier nur norm_word (Normalisierung)
except ImportError:
    import types
    _ta = types.ModuleType("torchaudio")
    _ta.functional = types.ModuleType("torchaudio.functional")
    sys.modules.update({"torchaudio": _ta, "torchaudio.functional": _ta.functional})
    _STUB = True
from whisper_common import (ROOT, RUNS, SR, split_rows, load_audio, words,          # noqa: E402,F401
                            LoRALinear, lora_state, merge_lora)
if _STUB:               # Platzhalter wieder entfernen, damit NeMo torchaudio als nicht vorhanden sieht
    sys.modules.pop("torchaudio.functional", None)
    sys.modules.pop("torchaudio", None)

MODEL = "nvidia/parakeet-tdt-0.6b-v3"
TARGETS = "linear_q,linear_k,linear_v,linear_out,linear1,linear2,enc,pred"


def add_lora(model, targets, r, alpha, dropout, scope="all"):
    """Alle Gewichte einfrieren und die genannten Linear-Schichten durch LoRALinear ersetzen – in Encoder und
    Joint-Netz (scope "all") oder nur in einem von beiden ("encoder", "joint"). -> Zahl der ersetzten Schichten."""
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
    """Parakeet aus NeMo, optional mit trainiertem LoRA-Adapter (in die Basisgewichte eingerechnet)."""
    import nemo.collections.asr as nemo_asr
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = nemo_asr.models.ASRModel.from_pretrained(name)     # NeMo legt das Modell selbst schon auf die GPU
    if adapter:
        cfg = json.loads((Path(adapter) / "lora.json").read_text(encoding="utf-8"))
        assert cfg["base"] == name, f"Adapter gehört zu {cfg['base']}, nicht zu {name}"
        add_lora(model, cfg["targets"], cfg["r"], cfg["alpha"], 0.0, cfg.get("scope", "all"))
        state = torch.load(Path(adapter) / "lora.pt")
        expected = {k for k in model.state_dict() if ".lora_" in k}
        assert expected == set(state), "Adapter passt nicht zum Modell"
        model.load_state_dict(state, strict=False)
        model.to(device)                                       # neue LoRA-Schichten auf dasselbe Gerät
        merge_lora(model)
    return model.to(device).eval()


def texte(out):
    """NeMo gibt je nach Version str, Hypothesis oder (beste, alle) zurück."""
    if isinstance(out, tuple):
        out = out[0]
    return [(o.text if hasattr(o, "text") else str(o)).strip() for o in out]
