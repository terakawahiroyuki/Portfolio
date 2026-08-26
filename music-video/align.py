#!/usr/bin/env python3
"""Force-align the known lyrics to the extracted vocal track with pocketsphinx."""
import os as _os
SP = _os.environ.get("MV_WORKDIR", _os.path.dirname(_os.path.abspath(__file__)) or ".")
AUDIO = _os.environ.get("MV_AUDIO", _os.path.join(SP, "Let_It_Ride.mp3"))
import json
import re
import wave

from pocketsphinx import Decoder


# --- parse lyrics from the ffmetadata dump ---
sections = []  # [{name, lines: [[word,...], ...]}]
with open(f"{SP}/meta.txt") as f:
    txt = f.read()
m = re.search(r"lyrics-eng=(.*?)\ncomment=", txt, re.S)
raw = m.group(1)
lines = [l.rstrip("\\").strip() for l in raw.split("\n")]
cur = None
for l in lines:
    if not l:
        continue
    if l.startswith("[") and l.endswith("]"):
        cur = {"name": l[1:-1], "lines": []}
        sections.append(cur)
    else:
        cur["lines"].append(l)

with open(f"{SP}/lyrics.json", "w") as f:
    json.dump(sections, f, ensure_ascii=False, indent=1)
print(f"{len(sections)} sections, {sum(len(s['lines']) for s in sections)} lines")


def norm_word(w):
    w = w.lower().replace("’", "'").replace("…", "")
    w = re.sub(r"[^a-z']", "", w)
    return w.strip("'") if w.strip("'") else w


# flat word list with (section_idx, line_idx, word_idx)
flat = []
for si, s in enumerate(sections):
    for li, l in enumerate(s["lines"]):
        for wi, w in enumerate(l.split()):
            nw = norm_word(w)
            if nw:
                flat.append({"si": si, "li": li, "wi": wi, "word": nw, "disp": w})
print(f"{len(flat)} words to align")

dec = Decoder(samprate=16000, beam=1e-100, wbeam=1e-80, pbeam=1e-100, silprob=0.3)
# explicit pronunciations for out-of-dictionary compounds
for w, ph in [("bassline", "B AE S L AY N"),
              ("feelgood", "F IY L G UH D"),
              ("oldschool", "OW L D S K UW L")]:
    dec.add_word(w, ph, True)
oov = sorted({f["word"] for f in flat if not dec.lookup_word(f["word"])})
print("OOV after add_word:", oov)
align_words = [f["word"] for f in flat]

text = " ".join(align_words)
dec.set_align_text(text)
dec.start_utt()
with wave.open(f"{SP}/vocals16k.wav", "rb") as wv:
    data = wv.readframes(wv.getnframes())
dec.process_raw(data, full_utt=True)
dec.end_utt()

segs = [(s.word, s.start_frame / 100.0, s.end_frame / 100.0, s.ascore) for s in dec.seg()]
words_only = [s for s in segs if not (s[0].startswith("<") or s[0].startswith("["))]
print(f"aligned segments: {len(segs)} total, {len(words_only)} words (expected {len(flat)})")

out = []
j = 0
for s in segs:
    w = s[0]
    if w.startswith("<") or w.startswith("["):
        continue
    base = re.sub(r"\(\d+\)$", "", w)
    assert base == align_words[j], f"mismatch at {j}: {base} != {align_words[j]}"
    f = flat[j]
    out.append({"si": f["si"], "li": f["li"], "wi": f["wi"], "disp": f["disp"],
                "s": round(s[1], 2), "e": round(s[2], 2), "score": s[3]})
    j += 1

with open(f"{SP}/aligned_words.json", "w") as f:
    json.dump(out, f, ensure_ascii=False)
print("first words:", [(o["disp"], o["s"]) for o in out[:6]])
print("last words:", [(o["disp"], o["s"]) for o in out[-4:]])

# line-level summary
print("\n--- line timings ---")
for si, s in enumerate(sections):
    for li in range(len(s["lines"])):
        ws = [o for o in out if o["si"] == si and o["li"] == li]
        if ws:
            print(f"[{s['name']:>13} {li:2d}] {ws[0]['s']:7.2f} - {ws[-1]['e']:7.2f}  {s['lines'][li][:50]}")
