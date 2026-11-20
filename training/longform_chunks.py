#!/usr/bin/env python3
"""
Ganze Interviews in Stücke von höchstens 28 s schneiden - an Pausen, nach der
Lautstärke (ohne Transkript, ohne Modell; für alle Modelle dieselben Stücke).

  python3 training/longform_chunks.py --split test

Lautstärke je 20 ms; leise ist, was unter Rauschboden + 35 % des Abstands zum
90. Perzentil liegt. Pausen = mindestens 0,3 s leise. Geschnitten wird an der ersten
langen Pause (>= 2 s) im Fenster, sonst an der letzten Pause bis 28 s, sonst hart an
der leisesten Stelle zwischen 20 und 28 s. Stille zwischen den Stücken fällt weg;
Stücke mit weniger als 15 % lauten Rahmen werden nicht transkribiert.
-> training/longform/chunks/<interview>.json. Ausgegeben werden nur Zahlen; zur
Kontrolle der Anteil der Transkriptwörter (Alignment), die in einem Stück liegen.
"""
import argparse, json, wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training" / "longform" / "chunks"
SR, HOP = 16000, 320                      # 20 ms
MAXLEN, MINPAUSE, LONGPAUSE = 28.0, 0.3, 2.0


def read_wav(path):
    with wave.open(str(path)) as f:
        assert f.getframerate() == SR and f.getnchannels() == 1
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def chunks_for(audio):
    n = len(audio) // HOP
    rms = np.sqrt(np.mean(audio[:n * HOP].reshape(n, HOP) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    floor, p90 = np.percentile(db, 10), np.percentile(db, 90)
    quiet = db < floor + 0.35 * (p90 - floor)
    pauses, i = [], 0                      # (start_s, end_s) leiser Läufe
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * HOP / SR >= MINPAUSE:
                pauses.append((i * HOP / SR, j * HOP / SR))
            i = j
        else:
            i += 1
    total = n * HOP / SR
    out, pos = [], 0.0
    if pauses and pauses[0][0] == 0.0:
        pos = max(0.0, pauses[0][1] - 0.1)
    while pos < total - 0.5:
        win = [p for p in pauses if pos + 1.0 < p[0] <= pos + MAXLEN]
        long_ = [p for p in win if p[1] - p[0] >= LONGPAUSE]
        late = [p for p in win if p[0] > pos + 3.0]
        if long_:
            cut, nxt = long_[0][0] + 0.1, long_[0][1] - 0.1
        elif late:
            cut, nxt = late[-1][0] + 0.1, late[-1][1] - 0.1
        elif pos + MAXLEN >= total:
            cut, nxt = total, total
        else:
            a, b = int((pos + 20) * SR / HOP), int((pos + MAXLEN) * SR / HOP)
            cut = (a + int(np.argmin(db[a:b]))) * HOP / SR
            nxt = cut
        cut = min(cut, pos + MAXLEN)
        fa, fb = int(pos * SR / HOP), max(int(pos * SR / HOP) + 1, int(cut * SR / HOP))
        if cut - pos >= 0.5 and (~quiet[fa:fb]).mean() >= 0.15:
            out.append([round(pos, 2), round(cut, 2)])
        pos = max(nxt, cut)
    return out, total, {"floor_db": round(float(floor), 1), "p90_db": round(float(p90), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    ivs = sorted({json.loads(l)["interview"] for l in (ROOT / "dataset" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
                  if json.loads(l)["split"] == a.split})
    OUT.mkdir(parents=True, exist_ok=True)
    T = C = W = Win = 0
    for iv in ivs:
        ch, total, lev = chunks_for(read_wav(ROOT / "work" / "audio16k" / f"{iv}.wav"))
        (OUT / f"{iv}.json").write_text(json.dumps({"interview": iv, "audio_s": round(total, 2), "chunks": ch, "pegel": lev,
                                                    "regeln": {"max_s": MAXLEN, "pause_s": MINPAUSE, "lange_pause_s": LONGPAUSE}}),
                                        encoding="utf-8")
        words = [w for w in json.loads((ROOT / "work" / "aligned" / f"{iv}.json").read_text(encoding="utf-8"))["words"]
                 if w.get("s") is not None and w.get("e") is not None]
        mids = np.array([(w["s"] + w["e"]) / 2 for w in words])
        inside = sum(((mids >= s) & (mids < e)).sum() for s, e in ch)
        dur = sum(e - s for s, e in ch)
        lens = [e - s for s, e in ch]
        T += total; C += dur; W += len(mids); Win += inside
        print(f"{iv:24} {total/3600:4.2f} h: {len(ch):4} Stücke (Median {np.median(lens):4.1f} s, max {max(lens):4.1f} s), "
              f"{100*dur/total:3.0f} % der Zeit, Transkriptwörter in Stücken {100*inside/len(mids):5.1f} %")
    print(f"gesamt: {T/3600:.2f} h Audio, {C/3600:.2f} h in Stücken ({100*C/T:.0f} %), Transkriptwörter in Stücken {100*Win/W:.1f} % von {W}")


if __name__ == "__main__":
    main()
