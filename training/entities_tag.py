#!/usr/bin/env python3
"""
Mark entities in the transcripts of one part - locally with spaCy.

  ~/venvs/roterstern-ner/bin/python training/entities_tag.py --split test

Runs in its own environment (spaCy 3.8 + de_core_news_lg), so that the
pipeline environment (torch, numpy) stays untouched; otherwise needs only the
standard library. Per interview, the segments are joined in time order into one
text (one line per segment) so that spaCy has context; entities across a
segment boundary are discarded.

Marked are:
  spacy   PER, LOC, ORG, MISC from the NER model
  propn   proper nouns (part of speech PROPN) outside any entity - possible
          gaps of the model, type "?" until reviewed
  regel   years 1800-2029 and dates, type DATUM

-> training/entities/<split>_spacy.jsonl (per segment: file_name, interview,
   entities with character positions in `text`). Only figures are printed.
"""
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

import spacy

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training" / "entities"
MODEL = "de_core_news_lg"
DATE = re.compile(r"\b(?:\d{1,2}\.\s?\d{1,2}\.\s?)?(?:18|19|20)\d{2}\b|\b\d{1,2}\.\s?(?:Jänner|Januar|Februar|Feber|März|"
                  r"April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    nlp = spacy.load(MODEL)
    rows = [json.loads(l) for l in (ROOT / "dataset" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == a.split]
    by_iv = defaultdict(list)
    for r in rows:
        by_iv[r["interview"]].append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    out_rows, count, uniq = [], Counter(), defaultdict(set)
    for iv, segs in sorted(by_iv.items()):
        segs.sort(key=lambda r: r["source_start"])
        offs, parts, pos = [], [], 0
        for r in segs:
            offs.append((pos, pos + len(r["text"])))
            parts.append(r["text"])
            pos += len(r["text"]) + 1
        doc = nlp("\n".join(parts))
        ents = [[] for _ in segs]

        def put(start, end, text, label, source):
            for k, (s0, s1) in enumerate(offs):
                if s0 <= start and end <= s1:
                    ents[k].append({"start": start - s0, "end": end - s0, "text": text, "label": label, "source": source})
                    count[(source, label)] += 1
                    uniq[label].add(text)
                    return

        covered = set()
        for e in doc.ents:
            put(e.start_char, e.end_char, e.text, e.label_, "spacy")
            covered.update(range(e.start, e.end))
        for t in doc:
            if t.pos_ == "PROPN" and t.i not in covered and t.text[:1].isupper():
                put(t.idx, t.idx + len(t.text), t.text, "?", "propn")
        for m in DATE.finditer(doc.text):
            put(m.start(), m.end(), m.group(), "DATUM", "regel")
        for r, e in zip(segs, ents):
            out_rows.append({"file_name": r["file_name"], "interview": iv, "project": r["project"],
                             "entities": sorted(e, key=lambda x: x["start"])})
    path = OUT / f"{a.split}_spacy.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"split": a.split, "segments": len(out_rows), "interviews": len(by_iv), "spacy": spacy.__version__,
            "model": f"{nlp.meta['name']} {nlp.meta['version']}",
            "mentions": {f"{s}:{l}": n for (s, l), n in sorted(count.items())},
            "unique": {l: len(v) for l, v in sorted(uniq.items())}}
    (OUT / f"{a.split}_spacy_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{a.split}: {len(out_rows)} Segmente aus {len(by_iv)} Interviews, Modell {meta['model']} (spaCy {meta['spacy']})")
    for (s, l), n in sorted(count.items()):
        print(f"  {s:6} {l:6} {n:5} Nennungen, {len(uniq[l]) if s != 'propn' or l == '?' else '':>5} verschiedene")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
