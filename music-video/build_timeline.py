#!/usr/bin/env python3
"""Build timeline.json (sections + lyric line timings) for the renderer.

Uses aligned_words.json from pocketsphinx when available and sane;
otherwise falls back to a beat-grid hypothesis guided by the structural
novelty peaks from analyze.py.
"""
import os as _os
SP = _os.environ.get("MV_WORKDIR", _os.path.dirname(_os.path.abspath(__file__)) or ".")
AUDIO = _os.environ.get("MV_AUDIO", _os.path.join(SP, "Let_It_Ride.mp3"))
import json
import os
import sys

import numpy as np


A = np.load(f"{SP}/analysis.npz")
DUR = float(A["duration"])
beat_times = A["beat_times"]
dbp = int(A["downbeat_phase"])
downbeats = beat_times[dbp::4]

with open(f"{SP}/lyrics.json") as f:
    sections_raw = json.load(f)

KIND = {"Intro": "intro", "Verse 1": "verse", "Verse 2": "verse", "Verse 3": "verse3",
        "Chorus": "chorus", "Final Chorus": "finalchorus", "Breakdown": "breakdown",
        "Outro": "outro"}


def snap_downbeat(t):
    i = int(np.argmin(np.abs(downbeats - t)))
    return float(downbeats[i])


def cluster_fix(ws):
    """Repair a line the aligner stretched across an instrumental gap:
    clamp absurd word durations, keep the densest word cluster, and pack
    stray leading/trailing words tightly against it."""
    ws = [dict(w) for w in ws]
    for w in ws:
        if w["e"] - w["s"] > 1.5:
            w["e"] = round(w["s"] + 1.5, 2)
    clusters, cur = [], [0]
    for k in range(1, len(ws)):
        if ws[k]["s"] - ws[cur[-1]]["e"] > 3.0:
            clusters.append(cur)
            cur = []
        cur.append(k)
    clusters.append(cur)
    best = max(clusters, key=len)
    i0, i1 = best[0], best[-1]
    for n, k in enumerate(range(i0 - 1, -1, -1)):
        e = round(ws[i0]["s"] - 0.05 - 0.25 * n, 2)
        ws[k]["s"], ws[k]["e"] = round(e - 0.2, 2), e
    for n, k in enumerate(range(i1 + 1, len(ws))):
        s = round(ws[i1]["e"] + 0.05 + 0.25 * n, 2)
        ws[k]["s"], ws[k]["e"] = s, round(s + 0.2, 2)
    return ws


def build_from_alignment(path):
    with open(path) as f:
        words = json.load(f)
    lines = []
    for si, s in enumerate(sections_raw):
        for li, text in enumerate(s["lines"]):
            ws = [w for w in words if w["si"] == si and w["li"] == li]
            if not ws:
                continue
            dur = ws[-1]["e"] - ws[0]["s"]
            if dur > 8 or dur / len(ws) > 2.5:
                fixed = cluster_fix(ws)
                if len(fixed) >= len(ws) * 0.5:
                    print(f"  cluster-fixed [{s['name']} {li}]: "
                          f"{ws[0]['s']:.2f}-{ws[-1]['e']:.2f} -> {fixed[0]['s']:.2f}-{fixed[-1]['e']:.2f}")
                    ws = fixed
            lines.append(dict(sec=s["name"], kind=KIND[s["name"]], text=text, si=si, li=li,
                              t0=ws[0]["s"], t1=ws[-1]["e"],
                              words=[{"w": w["disp"], "s": w["s"], "e": w["e"]} for w in ws]))
    return lines


def sanity(lines):
    """Returns (ok, report). Checks monotonicity and durations."""
    bad = 0
    prev_end = 0
    msgs = []
    for ln in lines:
        d = ln["t1"] - ln["t0"]
        nw = len(ln.get("words") or [])
        if d <= 0.15 or d > 14 or (nw and d / nw > 3.5):
            bad += 1
            msgs.append(f"  suspicious: [{ln['sec']} {ln['li']}] {ln['t0']:.2f}-{ln['t1']:.2f} '{ln['text'][:36]}'")
        if ln["t0"] < prev_end - 2.5:
            bad += 1
            msgs.append(f"  overlap: [{ln['sec']} {ln['li']}] starts {ln['t0']:.2f} before prev end {prev_end:.2f}")
        prev_end = max(prev_end, ln["t1"])
    return bad, msgs


def build_fallback():
    """Beat-grid hypothesis informed by novelty peaks."""
    bars = downbeats
    bar_len = float(np.median(np.diff(bars)))

    def bar_at(t):
        return float(bars[np.argmin(np.abs(bars - t))])

    # section start times (novelty-guided hypothesis)
    starts = {
        "Intro": 0.0,
        "Verse 1": bar_at(9.8),
        "Chorus": bar_at(48.4),
        "Verse 2": bar_at(88.3),
        "Chorus#2": bar_at(126.3),
        "Breakdown": bar_at(165.3),
        "Verse 3": bar_at(185.3),
        "Final Chorus": bar_at(232.1),
        "Outro": bar_at(252.6),
    }
    bars_per_line = {"Intro": 1.0, "Verse 1": 1.0, "Chorus": 2.0, "Verse 2": 1.0,
                     "Chorus#2": 2.0, "Breakdown": 1.0, "Verse 3": 1.0,
                     "Final Chorus": 2.0, "Outro": 1.0}
    lines = []
    keys = list(starts)
    chorus_seen = 0
    for si, s in enumerate(sections_raw):
        name = s["name"]
        key = name
        if name == "Chorus":
            chorus_seen += 1
            key = "Chorus" if chorus_seen == 1 else "Chorus#2"
        t = starts[key]
        bpl = bars_per_line[key] * bar_len
        if name == "Intro":
            t = 1.5
            bpl = 9.8 / max(len(s["lines"]), 1) * 0.9
        for li, text in enumerate(s["lines"]):
            t0 = t + li * bpl
            t1 = t0 + bpl * 0.92
            lines.append(dict(sec=name, kind=KIND[name], text=text, si=si, li=li,
                              t0=round(t0, 2), t1=round(t1, 2), words=[]))
    return lines


use_fallback = "--provisional" in sys.argv or not os.path.exists(f"{SP}/aligned_words.json")
if not use_fallback:
    lines = build_from_alignment(f"{SP}/aligned_words.json")
    bad, msgs = sanity(lines)
    print(f"alignment: {len(lines)} lines, {bad} suspicious")
    print("\n".join(msgs[:30]))
    if bad > len(lines) * 0.25:
        print("too many suspicious lines -> falling back to grid")
        lines = build_fallback()
else:
    lines = build_fallback()
    print(f"fallback grid: {len(lines)} lines")

# section spans from line times; switch scenes early across long instrumental gaps
sections = []
by_sec = {}
for ln in lines:
    by_sec.setdefault(ln["si"], []).append(ln)
si_sorted = sorted(by_sec)
prev_end = 0.0
for si in si_sorted:
    ls = by_sec[si]
    name = sections_raw[si]["name"]
    first = min(l["t0"] for l in ls)
    if KIND[name] == "intro":
        t0 = 0.0
    elif first - prev_end > 6.0:
        t0 = snap_downbeat(prev_end + 1.2)
    else:
        t0 = snap_downbeat(first - 0.15)
    sections.append(dict(name=name, kind=KIND[name], t0=round(t0, 2)))
    prev_end = max(l["t1"] for l in ls)
# monotonic, then close each at the next start
for k in range(1, len(sections)):
    if sections[k]["t0"] <= sections[k - 1]["t0"]:
        sections[k]["t0"] = sections[k - 1]["t0"] + 4.0
for k, s in enumerate(sections):
    s["t1"] = round(sections[k + 1]["t0"] if k + 1 < len(sections) else DUR, 2)

with open(f"{SP}/timeline.json", "w") as f:
    json.dump({"sections": sections, "lines": lines}, f, ensure_ascii=False, indent=1)

print("\nsections:")
for s in sections:
    print(f"  {s['name']:>14} {s['t0']:7.2f} - {s['t1']:7.2f}")
print(f"saved timeline.json ({len(lines)} lines)")
