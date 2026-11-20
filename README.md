# Dialect Oral History ASR

Code for preparing Austrian German oral history interviews, partly in strong dialect, as
speech recognition training data, and for fine-tuning and comparing two ASR models on them:
OpenAI Whisper large-v3 and NVIDIA Parakeet TDT 0.6B v3, each without and with LoRA.

The repository accompanies a short paper for DH2027 (Galway). **It contains no data:**
audio, transcripts, speaker names and all per-interview metadata stay with the archive. They live in
local, git-ignored folders (`Interviews/`, `config/*.json`, `work/`, `dataset/`). Templates with
made-up example values are provided under `config/*.example.json`.

> Status (20 Nov 2026): pipeline and training code complete; learning rate and epoch chosen on dev with the
> longer segments described here; results below are for the selected checkpoints.

## Overview

```
transcripts (docx/pdf/rtf, photographed pages)        recordings (wav/mp3/m4a, multi-part)
        │                                                     │
 0c  LLM writes a "spec" per interview (Qwen3-235B)        2  16 kHz mono, parts concatenated
        │  speakers, roles, tag variants, start/end           │
 1   deterministic parser → speaker turns              3a  wav2vec2 CTC emissions (cached)
        │  removes only; word-order check                     │
        └──────────────────────────┬──────────────────────────┘
                              3b  forced alignment without hand-set time marks
                                   (anchors from the recogniser itself, wildcards for gaps)
                              4   segmentation: one speaker, ≤ 28 s, edge checks, quality filter
                              5–6 export (FLAC + metadata, fixed split) and bit-exact check
                                   │
             ┌─────────────────────┴─────────────────────┐
     Whisper large-v3 + LoRA                     Parakeet TDT 0.6B v3 + LoRA
             └─────────────────────┬─────────────────────┘
        evaluation: test segments, whole interviews (chunked at pauses), names and dates
```

## Pipeline (`pipeline/`)

| Stage | Script | What it does |
|---|---|---|
| 0 | `draft_specs.py` | finds transcript and audio parts per interview (`config/interviews.json`, `config/sources.json`) |
| 0b | `ocr_zwangsarbeiter.py` | OCR of photographed transcript pages (macOS Vision, `bin/vision_ocr.swift`) |
| 0c | `llm_specs.py` | **an LLM writes the spec**: speaker tags with name and role, tag variants, where the conversation starts and ends, non-speech lines |
| 1 | `parse_transcripts.py` | applies the spec; only *removes* (header, tags, annotations, time marks) and checks that every output word appears in the source in the same order |
| 2 | `convert_audio.py` | audio → 16 kHz mono, parts concatenated |
| 3a | `emissions.py` | wav2vec2 emissions (expensive, cached) |
| 3b | `align.py` | forced alignment without hand-set time marks |
| 3c | `eval_markers.py` | accuracy against the hand-set time marks (independent check) |
| 4 | `segment.py` | word times → segments, edge checks, quality filter |
| 5 | `export_dataset.py` | cut FLAC files, metadata, fixed train/dev/test split by person groups |
| 6 | `check_dataset.py` | every file bit-exact against its slice of the source audio |
| 7–9 | `eval_segments.py`, `eval_edges.py`, `make_samples.py`, `report.py` | independent check with Whisper-small, edge rules, listening test, all figures |

`sh pipeline/run_all.sh` runs everything in order; every stage records hashes of its inputs and code and
only recomputes what changed.

### Stage 0c: specs from an LLM, verified by a parser

The transcripts follow at least six conventions (tags like `I:`/`ZZ:`, `[A.K.]`, `Hr. X:`, tags explained
in the header, no tags at all, OCR from photographed pages). Instead of guessing them with regular
expressions, an LLM decides the structure and a deterministic parser executes it:

* Input to the model: the structural inventory of the transcript (`inventory.py`; tag candidates with counts,
  all bracketed annotations, header with contact data masked) plus the first 150 and last 40 lines.
* Output: JSON under a schema (temperature 0) – speakers (tag, name, role: interviewer / interviewee /
  other), tag variants, conversation start and end, non-speech lines. **The model never outputs
  transcript text**, so it cannot alter what is said.
* Every answer is parsed immediately. Parser errors and plausibility problems (tags at line start that are
  not in the speaker list, speech after the declared end, the "interviewer" talking more than the
  interviewee, a "non-speech" rule that hits a line of more than 15 words) go back to the model, which may
  correct itself up to two times. Rules that would drop a line of speech are rejected outright.
* Model: Qwen3-235B-A22B-Instruct-2507 through a Qwen chat-completions API. For 84 interviews: 84/84 valid specs,
  about 10 minutes and under US$ 0.50 in total. Agreement with the specs used for an earlier version of the
  dataset, per word: 99.8 % of the words captured, same speaker 99.9 %, same role 99.1 % (almost all differences
  are interviewee vs. other person, which does not affect the dataset).

Annotations in brackets are classified by fixed rules (`notes.py`): unclear or missing speech becomes a
wildcard in the alignment and a hard segment boundary; laughter and gestures do not.

### Stage 4: longer segments

Segments never cross a speaker change, unwritten speech (`[unv]`, `[gap]`, speech the recogniser hears in a
pause), unaligned or implausibly stretched words, or pauses of 4 s or more. Within one speaker, a segment is
cut only when it would exceed 28 s, at the best point after at least 16 s (sentence end plus pause).

| | earlier segmentation | this version |
|---|---|---|
| segments | 19,020 | 14,933 |
| hours (without interviewers in dev/test) | 46.8 | 49.9 |
| median / mean duration | 7.8 / 8.9 s | 10.3 / 12.0 s |
| 90th percentile / longest | 17.1 / 20.4 s | 23.8 / 28.4 s |
| share of audio in segments ≥ 15 s | 35 % | 67 % |

Split (fixed by interview and person group; interviewers only in training):
train 12,061 segments / 39.2 h, dev 1,282 / 4.6 h, test 1,590 / 6.1 h.

## Training and evaluation (`training/`)

* `whisper_lora.py`, `parakeet_lora.py`: LoRA written from scratch (`whisper_common.LoRALinear`, no `peft`),
  identical for both models: r = 16, α = 32, dropout 0.05, AdamW, linear warm-up and decay, 16 segments per
  step, gradient norm clipped at 1. Whisper: attention and MLP of encoder and decoder (1.8 % of the
  parameters). Parakeet: attention and feed-forward of the FastConformer encoder plus the joint network's
  projections (1.7 %), TDT loss averaged per target token, BatchNorm statistics frozen.
* Learning rate and epoch chosen on dev only (`select_dev.py`, see below); test evaluated once for the selected
  checkpoint of each model (`jobs/test_best.sh`).
* `whisper_eval.py`, `parakeet_eval.py`: test segments (WER/CER after normalisation: lower case, no
  punctuation, numbers spelled out, fillers removed).
* `longform_chunks.py`, `longform_eval.py`, `longform_score.py`: whole interviews cut at pauses into chunks of
  ≤ 28 s, scored against the aligned transcript; bootstrap over interviews.
* `entities_tag.py`, `entity_eval.py`, `longform_entities.py`: names and dates (manually checked spaCy
  annotations), recognised exactly or findable by fuzzy search.
* `jobs/`: queue, train-and-evaluate scripts for a single NVIDIA GPU with 24 GB (Python interpreters via
  `PY` and `PY_NEMO`).

### Model selection on dev (20 Nov 2026)

Learning rates (`jobs/queue_lr_dh2027.sh`): Whisper 2·10⁻⁴, 5·10⁻⁴, 1·10⁻³; Parakeet 1·10⁻³, 2·10⁻³, 4·10⁻³; each
3 epochs with dev after every epoch, everything else fixed (r = 16, 16 segments per step). Selection rule, fixed
before the results: lowest dev WER over learning rate × epoch; ties (0.01 points) go to fewer epochs, then the
smaller learning rate.

| dev WER (1,282 segments, 35,316 words) | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| Whisper + LoRA, lr 2·10⁻⁴ | 19.60 % | **18.34 %** | 18.68 % |
| Whisper + LoRA, lr 5·10⁻⁴ | 20.37 % | 18.76 % | 18.50 % |
| Whisper + LoRA, lr 1·10⁻³ | 23.32 % | 20.66 % | 19.44 % |
| Parakeet + LoRA, lr 1·10⁻³ | 22.18 % | 21.13 % | 20.77 % |
| Parakeet + LoRA, lr 2·10⁻³ | 22.43 % | 21.34 % | **20.39 %** |
| Parakeet + LoRA, lr 4·10⁻³ | 25.24 % | 22.00 % | 20.55 % |

Selected: Whisper lr 2·10⁻⁴, epoch 2; Parakeet lr 2·10⁻³, epoch 3 (the setting chosen earlier on the shorter
segments). The best three settings of each model lie within 0.4 points on dev. The selected Whisper learning rate is the smallest tested; Parakeet still improves from epoch 2 to 3 at
every learning rate, so more epochs might help (not tested). Against the previously used Whisper checkpoint
(lr 5·10⁻⁴, epoch 3) the selected one is slightly better on test: segments +0.3 points (0.0 to 0.7), whole
interviews +0.4 (0.1 to 0.7), names and dates ±0.0 (−1.8 to 2.4).

### Human evaluation

* **Segment rating** (`pipeline/make_samples.py --n 150 --seed 2027`): 150 segments drawn at random in proportion
  to duration from the whole dataset; one rater judges each against the audio as *fits*, *wrongly cut*
  (pipeline error: word cut off at an edge, extra speech, another voice, text belongs elsewhere) or *transcript
  wrong* (archive error: the cut is right but words inside are missing, added, wrong or paraphrased), or
  *unsure*. Shares with Wilson 95 % intervals (`--summary`).
  Result (19 Nov 2026): **fits 128 of 150 (85.3 %, 95 % 78.8–90.1)**; wrongly cut 11 (7.3 %, 4.1–12.7), mostly
  extra speech at an edge (8) or another voice (2); transcript wrong 12 (8.0 %, 4.6–13.5), mostly a wrong word
  (6); both 1, unsure 0. By collection: Roter Stern 103/123 fit, Rechnitz 20/22, Zwangsarbeiter 5/5.
* **Blind comparison of the final models** (`training/blind_rating.py`, `blind_eval.py`): 8 test segments per
  interview where the outputs of Whisper + LoRA and Parakeet + LoRA differ (96 pairs from the 12 test interviews;
  the outputs are identical for 308 of 1,590 segments), plus 4 identical pairs as attention checks; sides
  randomised, both outputs set alike (lower case, no punctuation, numbers spelled out), the archive transcript is
  not shown. Five-point scale; sign test and bootstrap over interviews.
  Result (one rater, 20 Nov 2026): **Whisper + LoRA better in 35 of 96 pairs (36 %), Parakeet + LoRA better in 17
  (18 %)**, no difference in 44 (46 %); sign test p = 0.018. Mean on the scale from −2 to +2 (positive = Parakeet +
  LoRA better): −0.22 (95 % over interviews −0.32 to −0.11). Clear preferences are rare (Whisper 5, Parakeet 2).
  Attention checks 4/4 rated equal. The listener's judgement agrees with the WER difference.

### Results (test, greedy decoding)

Checkpoints selected on dev as above: Whisper + LoRA lr 2·10⁻⁴ after epoch 2, Parakeet + LoRA lr 2·10⁻³ after
epoch 3.
Test segments: 1,590 segments, 49,134 words. Whole interviews: the 12 test interviews (8.5 h), cut at pauses into
1,296 chunks of ≤ 28 s, scored against the aligned transcript on the 1,106 chunks without untranscribed speech.
Names and dates: 731 manually checked mentions (persons, places, organisations, dates), counted as recognised
when every word matches.

| | WER segments | WER whole interviews | names + dates (whole interviews) | training (to the selected checkpoint) |
|---|---|---|---|---|
| Whisper large-v3 | 19.8 % | 20.8 % | 59.4 % | – |
| Whisper large-v3 + LoRA | **14.0 %** | **17.7 %** | **72.4 %** | 1 h 41 min (2 epochs) |
| Parakeet TDT 0.6B v3 | 24.5 % | 26.2 % | 52.7 % | – |
| Parakeet TDT 0.6B v3 + LoRA | 15.8 % | 19.6 % | 68.7 % | 95 min (3 epochs) |

WER by collection (test segments, without → with LoRA):

| | Roter Stern | Rechnitz | Zwangsarbeiter |
|---|---|---|---|
| Whisper large-v3 | 13.1 → 9.7 % | 31.4 → 17.9 % | 51.1 → 39.9 % |
| Parakeet TDT 0.6B v3 | 17.3 → 11.1 % | 39.1 → 21.2 % | 55.8 → 42.6 % |

Differences in percentage points with 95 % bootstrap intervals over the 12 test interviews (positive = second
system better):

| | WER segments | WER whole interviews | names + dates |
|---|---|---|---|
| Whisper → Whisper + LoRA | +5.8 (3.1 to 8.8) | +3.2 (1.4 to 4.9) | +13.0 (8.2 to 17.8) |
| Parakeet → Parakeet + LoRA | +8.7 (5.4 to 12.4) | +6.6 (4.0 to 9.1) | +16.0 (9.9 to 20.7) |
| Whisper + LoRA → Parakeet + LoRA | −1.9 (−2.5 to −1.2) | −2.0 (−2.5 to −1.3) | −3.7 (−6.2 to −1.1) |
| Whisper → Parakeet + LoRA | +4.0 (1.6 to 6.4) | +1.2 (−0.4 to 2.6) | +9.3 (4.2 to 13.9) |

Compared with the earlier, shorter segmentation (same models and the settings chosen there – Whisper lr 5·10⁻⁴,
both epoch 3 –, whole interviews, paired), the longer segments change neither WER (Whisper + LoRA 18.1 → 18.0 %,
Parakeet + LoRA 19.8 → 19.6 %, both within ±0.4) nor names and dates measurably.

Compute on one NVIDIA GPU (24 GB): recognition of the 8.5 h of whole test interviews takes 73 s with Parakeet and about 260 s with
Whisper (greedy); training steps take 2.5 s (Parakeet, max. 14.1 GB) and 4.0 s (Whisper, max. 20.6 GB).

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt                 # pipeline + Whisper
# Parakeet needs its own environment (NeMo pins torch):
python3 -m venv ~/venv-nemo && ~/venv-nemo/bin/pip install -r requirements-nemo.txt
```

Data layout (all git-ignored):

```
Interviews/            original transcripts and recordings (or a symbolic link to them)
config/interviews.json per interview: files, format profile, consent, exclusion   (template: *.example.json)
config/sources.json    transcript ↔ audio mapping where it is not given by the folder structure
config/profiles.json   one format profile per transcript type
config/split.json      fixed train/dev/test assignment
work/, dataset/        intermediate results and the exported dataset
```

Stage 0c needs an OpenAI-compatible chat-completions API that serves Qwen3-235B-A22B-Instruct-2507 (hosted, or
self-hosted e.g. with vLLM); no extra package is needed:

```bash
export QWEN_API_BASE=https://<endpoint>/v1   # required
export QWEN_API_KEY=<key>                    # if the endpoint needs one
export QWEN_MODEL=qwen3-235b-a22b-instruct-2507   # model name at that endpoint
python3 pipeline/llm_specs.py                # --response-format object|none for endpoints without JSON-schema support
```

## AI use

The structure of each transcript (stage 0c) is decided by Qwen3-235B as described above. The code was
written with the help of an AI coding assistant (Claude, Anthropic) and reviewed by the author.

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). It covers the code only: audio,
transcripts and all data derived from them are not part of this repository and remain with the archive.
