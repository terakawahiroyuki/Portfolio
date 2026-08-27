#!/usr/bin/env python3
"""Let It Ride — 30s vertical TikTok cut (1080x1920).

Uses the pre-verse build ("Keep it moving...") into the full first chorus,
start/end snapped to downbeats. Reuses analysis.npz + timeline.json from the
full MV pipeline; layout recomposed for 9:16 with TikTok UI safe zones.
"""
import os as _os
SP = _os.environ.get("MV_WORKDIR", _os.path.dirname(_os.path.abspath(__file__)) or ".")
AUDIO = _os.environ.get("MV_AUDIO", _os.path.join(SP, "Let_It_Ride.mp3"))
import argparse
import math
import json
import random
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import imageio_ffmpeg

W, H = 1080, 1920
FPS = 30

CREAM = (255, 243, 214)
GOLD = (255, 201, 60)
ORANGE = (255, 138, 0)
CORAL = (255, 78, 80)
MAGENTA = (216, 78, 119)
TEAL = (31, 167, 160)
BROWN = (43, 27, 14)


def lerp_c(c1, c2, u):
    return tuple(int(round(c1[k] + (c2[k] - c1[k]) * u)) for k in range(3))


def ease_out(u):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


def ease_in_out(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------- data
A = np.load(f"{SP}/analysis.npz")
tempo = float(A["tempo"])
beat_times = A["beat_times"]
dbp = int(A["downbeat_phase"])
downbeats = beat_times[dbp::4]
bass, mid_e, high_e = A["bass"], A["mid"], A["high"]
onset, rms, spec = A["onset"], A["rms"], A["spec"]
beat_len = 60.0 / tempo

with open(f"{SP}/timeline.json") as f:
    TL = json.load(f)

# clip range: downbeat before "Keep it moving" (58.88) .. end of chorus line 7 (87.96)
START = float(downbeats[np.searchsorted(downbeats, 58.88) - 1])
END = 87.96
CH_T0 = 67.55          # chorus scene switch
DUR = END - START
NF = int(round(DUR * FPS))
print(f"clip {START:.2f} -> {END:.2f}  ({DUR:.2f}s, {NF} frames)")

LINES = [ln for ln in TL["lines"] if START - 0.5 <= ln["t0"] <= END - 0.3]
for ln in LINES:
    print(f"  [{ln['kind']:>7}] {ln['t0']:6.2f}-{ln['t1']:6.2f} {ln['text'][:44]}")


def E(ts):
    i = min(max(int(ts * FPS), 0), len(bass) - 1)
    b = int(np.searchsorted(beat_times, ts, side="right") - 1)
    tsb = ts - (beat_times[b] if b >= 0 else -beat_len)
    dbs = downbeats[downbeats <= ts]
    tsd = ts - (dbs[-1] if len(dbs) else -10)
    return dict(bass=float(bass[i]), mid=float(mid_e[i]), high=float(high_e[i]),
                onset=float(onset[i]), rms=float(rms[i]),
                pulse=math.exp(-tsb * 6.0), dpulse=math.exp(-max(tsd, 0) * 3.2),
                beat=b, spec=spec[min(i, len(spec) - 1)])


# ---------------------------------------------------------------- fonts/sprites
F = f"{SP}/fonts"
fon_verse = ImageFont.truetype(f"{F}/anton.ttf", 72)
fon_chorus = ImageFont.truetype(f"{F}/anton.ttf", 80)
fon_bungee = ImageFont.truetype(f"{F}/bungee.ttf", 116)
fon_tag = ImageFont.truetype(f"{F}/archivo_black.ttf", 30)
fon_end = ImageFont.truetype(f"{F}/anton.ttf", 130)
fon_cta = ImageFont.truetype(f"{F}/archivo_black.ttf", 36)
fon_note = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 44)


def text_sprite(text, font, fill, stroke, sw=7, shadow=True):
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    bb = tmp.textbbox((0, 0), text, font=font, stroke_width=sw)
    w, h = bb[2] - bb[0] + 20, bb[3] - bb[1] + 22
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    ox, oy = 10 - bb[0], 10 - bb[1]
    if shadow:
        dr.text((ox + 5, oy + 6), text, font=font, fill=(20, 8, 4, 160),
                stroke_width=sw, stroke_fill=(20, 8, 4, 160))
    dr.text((ox, oy), text, font=font, fill=fill + (255,), stroke_width=sw, stroke_fill=stroke + (255,))
    return img


def wrap(text, font, maxw):
    words, lines, cur = text.split(), [], ""
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    for w_ in words:
        t = (cur + " " + w_).strip()
        if tmp.textlength(t, font=font) > maxw and cur:
            lines.append(cur); cur = w_
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines


def vgrad(stops):
    col = np.zeros((H, 3), dtype=np.float32)
    ys = [int(s[0] * (H - 1)) for s in stops]
    for k in range(len(stops) - 1):
        y0, y1 = ys[k], ys[k + 1]
        c0, c1 = np.array(stops[k][1], np.float32), np.array(stops[k + 1][1], np.float32)
        n = max(y1 - y0, 1)
        col[y0:y1] = c0 + (c1 - c0) * np.linspace(0, 1, n)[:, None]
    col[ys[-1]:] = np.array(stops[-1][1], np.float32)
    return col[:, None, :]


GRAD_STREET = vgrad([(0, (255, 148, 40)), (0.45, (255, 196, 90)), (0.75, (255, 232, 170)), (1, (255, 240, 200))])
GRAD_PARTY = vgrad([(0, (255, 120, 40)), (1, (255, 170, 60))])


def radial_glow(radius, color, peak=160):
    d = radius * 2
    yy, xx = np.mgrid[0:d, 0:d]
    r = np.sqrt((yy - radius) ** 2 + (xx - radius) ** 2) / radius
    a = np.clip(1 - r, 0, 1) ** 2.2 * peak
    img = np.zeros((d, d, 4), dtype=np.uint8)
    img[..., :3] = color
    img[..., 3] = a.astype(np.uint8)
    return Image.fromarray(img, "RGBA")


def retro_sun(radius, c_top, c_bot):
    d = radius * 2
    img = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    for y in range(d):
        u = y / d
        col = lerp_c(c_top, c_bot, u)
        x = radius - math.sqrt(max(radius ** 2 - (y - radius) ** 2, 0))
        dr.line([(x, y), (d - x, y)], fill=col + (255,))
    n = 7
    for k in range(n):
        yy0 = int(d * (0.52 + 0.48 * (k / n)))
        dr.rectangle([0, yy0, d, yy0 + int(2 + 12 * (k / n))], fill=(0, 0, 0, 0))
    return img


GLOW = radial_glow(430, (255, 220, 140))
GLOW_HOT = radial_glow(470, (255, 120, 60), 175)
SUNS = {r: retro_sun(r, (255, 244, 200), (255, 150, 40)) for r in (190, 205, 220, 250, 265, 285)}


def pick_sun(base_r, boost):
    keys = sorted(SUNS)
    want = base_r * (1 + 0.10 * boost)
    return SUNS[min(keys, key=lambda k: abs(k - want))]


rng = random.Random(7)
BUILDINGS = []
x = -20
while x < W + 40:
    bw = rng.randint(70, 150)
    BUILDINGS.append((x, bw, rng.randint(160, 420), rng.random()))
    x += bw + rng.randint(6, 16)

g_rng = np.random.default_rng(3)
GRAIN = [Image.fromarray(g_rng.integers(0, 26, (H // 2, W // 2), dtype=np.uint8)
                         .repeat(2, 0).repeat(2, 1), "L").point(lambda v: v // 3) for _ in range(4)]
yy, xx = np.mgrid[0:H, 0:W]
r_norm = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
VIGN = Image.fromarray((np.clip(r_norm - 0.68, 0, 1) ** 1.8 * 190).astype(np.uint8), "L")
BLACK = Image.new("RGB", (W, H), (8, 5, 3))

NOTE_SPR = [text_sprite("♪", fon_note, c, BROWN, sw=4, shadow=False) for c in (CORAL, TEAL, GOLD)]
CONF = []
for col in (GOLD, CORAL, TEAL, CREAM, MAGENTA):
    s = Image.new("RGBA", (18, 13), (0, 0, 0, 0))
    ImageDraw.Draw(s).rectangle([2, 2, 15, 10], fill=col + (235,))
    CONF.append(s)

TAG = text_sprite("electron6  ·  LET IT RIDE", fon_tag, BROWN, CREAM, sw=0, shadow=False)
END_T = text_sprite("LET IT RIDE", fon_end, CREAM, BROWN, sw=9)
CTA = text_sprite("FULL MV ON YOUTUBE", fon_cta, GOLD, BROWN, sw=4)

HOOK = "LET IT RIDE"
HOOK_LETTERS = []
for ch in HOOK:
    if ch == " ":
        HOOK_LETTERS.append(None)
        continue
    HOOK_LETTERS.append((text_sprite(ch, fon_bungee, GOLD, (90, 20, 10), sw=8),
                         text_sprite(ch, fon_bungee, CREAM, (120, 30, 12), sw=8, shadow=False)))


def is_hook(text):
    t = text.lower()
    return t.count("let it ride") >= 1 and len(t) <= 30


LINE_SPRITES = []
for ln in LINES:
    if is_hook(ln["text"]):
        LINE_SPRITES.append(dict(ln=ln, hook=True, rows=[]))
        continue
    font = fon_chorus if ln["kind"] == "chorus" else fon_verse
    stroke = (105, 26, 16) if ln["kind"] == "chorus" else BROWN
    rows = []
    for row in wrap(ln["text"].replace("’", "'").upper(), font, 940):
        rows.append(dict(base=text_sprite(row, font, CREAM, stroke),
                         hi=text_sprite(row, font, GOLD, stroke, shadow=False)))
    LINE_SPRITES.append(dict(ln=ln, hook=False, rows=rows))
for k, sp in enumerate(LINE_SPRITES):
    t1 = sp["ln"]["t1"]
    cut = t1 + 0.45
    if k + 1 < len(LINE_SPRITES):
        cut = min(cut, max(LINE_SPRITES[k + 1]["ln"]["t0"] - 0.03, t1 - 0.2))
    sp["cut"] = min(cut, END)


def karaoke_frac(ln, ts):
    ws = ln.get("words") or []
    if not ws:
        return min(max((ts - ln["t0"]) / max(ln["t1"] - ln["t0"], 0.01), 0), 1)
    if ts >= ws[-1]["e"]:
        return 1.0
    for k, w_ in enumerate(ws):
        if ts < w_["s"]:
            return k / len(ws)
        if ts <= w_["e"]:
            return (k + (ts - w_["s"]) / max(w_["e"] - w_["s"], 0.01)) / len(ws)
    return 1.0


# ---------------------------------------------------------------- drawing
def draw_skyline(dr, dy, color):
    horizon = 1470 + dy
    for bx, bw, bh, seed in BUILDINGS:
        dr.rectangle([bx, horizon - bh, bx + bw, H], fill=color)
    dr.rectangle([0, horizon, W, H], fill=color)
    for k in range(6):
        yv = horizon + 40 + k * 56
        dr.rectangle([0, yv, W, yv + 18], fill=lerp_c(color, (0, 0, 0), 0.25))


def draw_boombox(img, dr, x, y, en, t, s=1.15):
    bounce = -14 * en["pulse"] * s
    y = y + bounce
    bw, bh = 340 * s, 150 * s
    dr.rounded_rectangle([x, y, x + bw, y + bh], radius=18 * s, fill=(50, 34, 26), outline=CREAM, width=int(4 * s))
    dr.rounded_rectangle([x + 8 * s, y + 8 * s, x + bw - 8 * s, y + 30 * s], radius=8 * s, fill=(72, 50, 38))
    dr.arc([x + 80 * s, y - 52 * s, x + bw - 80 * s, y + 10 * s], 180, 360, fill=(50, 34, 26), width=int(10 * s))
    for scx in (x + 78 * s, x + bw - 78 * s):
        r0 = 52 * s
        dr.ellipse([scx - r0, y + 88 * s - r0 + 8 * s, scx + r0, y + 88 * s + r0 + 8 * s],
                   fill=(26, 18, 12), outline=GOLD, width=int(4 * s))
        r1 = (20 + 26 * en["bass"]) * s
        dr.ellipse([scx - r1, y + 96 * s - r1, scx + r1, y + 96 * s + r1], fill=ORANGE)
        r2 = max((8 + 10 * en["mid"]) * s, 2)
        dr.ellipse([scx - r2, y + 96 * s - r2, scx + r2, y + 96 * s + r2], fill=CREAM)
    cx0 = x + bw / 2 - 52 * s
    dr.rounded_rectangle([cx0, y + 52 * s, cx0 + 104 * s, y + 118 * s], radius=8 * s, fill=(210, 190, 150))
    for wx in (cx0 + 28 * s, cx0 + 76 * s):
        dr.ellipse([wx - 14 * s, y + 71 * s, wx + 14 * s, y + 99 * s], fill=(70, 50, 36))
        for a in (t * 4, t * 4 + 2.1, t * 4 + 4.2):
            dr.line([(wx, y + 85 * s), (wx + 11 * s * math.cos(a), y + 85 * s + 11 * s * math.sin(a))],
                    fill=(230, 220, 190), width=max(int(3 * s), 1))


def draw_dancer(dr, x, base_y, k, t, en, col, s=1.5):
    bounce = -30 * s * en["pulse"] * (0.6 + 0.4 * math.sin(k * 2.4))
    sway = math.sin(t * 2 * math.pi * tempo / 60 / 2 + k * 0.7) * 9 * s
    y = base_y + bounce
    hx = x + sway * 0.4
    dr.ellipse([hx - 13 * s, y - 118 * s, hx + 13 * s, y - 92 * s], fill=col)
    dr.polygon([(hx - 15 * s, y - 88 * s), (hx + 15 * s, y - 88 * s),
                (x + 20 * s, y - 20 * s), (x - 20 * s, y - 20 * s)], fill=col)
    dr.polygon([(x - 16 * s, y - 24 * s), (x - 4 * s, y - 24 * s), (x - 10 * s - sway * 0.3, y)], fill=col)
    dr.polygon([(x + 16 * s, y - 24 * s), (x + 4 * s, y - 24 * s), (x + 10 * s + sway * 0.3, y)], fill=col)
    wa = math.sin(t * 2 * math.pi * tempo / 60 + k * 0.7) * 0.5
    for sgn in (-1, 1):
        a = -math.pi / 2 + sgn * (0.55 + 0.25 * wa)
        ex = hx + sgn * 14 * s + 46 * s * math.cos(a)
        ey = y - 84 * s + 46 * s * math.sin(a)
        dr.line([(hx + sgn * 14 * s, y - 82 * s), (ex, ey)], fill=col, width=int(9 * s))
        dr.ellipse([ex - 7 * s, ey - 7 * s, ex + 7 * s, ey + 7 * s], fill=col)


def draw_rays(img, t, cx, cy, colors, n=14, speed=0.5, alpha=60, boost=0.0):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    rot = t * speed + boost
    R = 2400
    for k in range(n):
        a0 = rot + k * 2 * math.pi / n
        a1 = a0 + math.pi / n * 0.85
        dr.polygon([(cx, cy), (cx + R * math.cos(a0), cy + R * math.sin(a0)),
                    (cx + R * math.cos(a1), cy + R * math.sin(a1))], fill=colors[k % len(colors)] + (alpha,))
    img.alpha_composite(ov)


def draw_eq(dr, en, y_base, width=680, hmax=120):
    n = 24
    bw = width / n
    x0 = (W - width) / 2
    for k in range(n):
        bh = 8 + en["spec"][k] * hmax
        c = (GOLD, CORAL, TEAL)[k % 3]
        dr.rounded_rectangle([x0 + k * bw + 2, y_base - bh, x0 + (k + 1) * bw - 2, y_base], radius=5, fill=c)


def confetti(ts):
    out = []
    b1 = np.searchsorted(beat_times, ts)
    for b in range(max(0, b1 - 5), b1):
        bt = beat_times[b]
        age = ts - bt
        if not 0 <= age <= 2.4:
            continue
        r = random.Random(9000 + b)
        for k in range(24):
            x0, y0 = r.uniform(0, W), r.uniform(H * 0.2, H * 0.55)
            x_ = x0 + r.uniform(-80, 80) * age
            y_ = y0 + r.uniform(-300, -90) * age + 420 * age * age * 0.5
            if -20 < y_ < H + 20:
                out.append((x_, y_, (b * 7 + k) % len(CONF), age / 2.4))
    return out


def draw_hook(img, ts, en, y, alpha=1.0):
    widths = [(l[0].width - 36 if l else 50) for l in HOOK_LETTERS]
    total = sum(widths) + 5 * (len(HOOK_LETTERS) - 1)
    x = (W - total) // 2
    n = len(HOOK_LETTERS)
    for k, l in enumerate(HOOK_LETTERS):
        if l is None:
            x += widths[k] + 5 + 14
            continue
        spr, hi = l
        cx = (k / (n - 1)) - 0.5
        arc_y = 30 * (cx * cx * 4 - 1) * 0.5
        wave = math.sin((ts * tempo / 60) * math.pi * 2 - k * 0.55)
        by = -18 * en["pulse"] * max(wave, 0) - 10 * en["dpulse"]
        use = hi if en["pulse"] > 0.82 else spr
        if alpha < 0.999:
            use = use.copy()
            a = use.getchannel("A").point(lambda v: int(v * alpha))
            use.putalpha(a)
        img.alpha_composite(use, (int(x - 18), int(y + arc_y + by)))
        x += widths[k] + 5
    # echo second "let it ride" of the doubled hook line as a ghost? keep clean.


def draw_lines(img, ts, en, mul=1.0):
    for sp in LINE_SPRITES:
        ln = sp["ln"]
        if not (ln["t0"] - 0.22 <= ts <= sp["cut"]):
            continue
        a_in = ease_out((ts - (ln["t0"] - 0.22)) / 0.28)
        fade_w = max(sp["cut"] - ln["t1"], 0.06)
        a_out = 1.0 if ts < ln["t1"] else max(0.0, 1 - (ts - ln["t1"]) / fade_w)
        alpha = a_in * a_out * mul
        if alpha <= 0.01:
            continue
        if sp["hook"]:
            draw_hook(img, ts, en, y=830, alpha=alpha)
            continue
        rows = sp["rows"]
        frac = karaoke_frac(ln, ts)
        pop = 1.0 + 0.16 * (1 - a_in)
        total_h = sum(r["base"].height - 30 for r in rows) + 30
        y = 1130 - total_h // 2
        for ri, r in enumerate(rows):
            base, hi = r["base"], r["hi"]
            w_, h_ = base.size
            x = (W - w_) // 2
            yy_ = y + int(5 * math.sin(ts * 3 + ri)) - int(8 * en["pulse"])
            row_frac = min(max(frac * len(rows) - ri, 0), 1)
            spr = base
            if alpha < 0.999 or pop > 1.001:
                if pop > 1.001:
                    spr = spr.resize((int(w_ * pop), int(h_ * pop)), Image.BILINEAR)
                    x = (W - spr.width) // 2
                    yy_ -= (spr.height - h_) // 2
                if alpha < 0.999:
                    spr = spr.copy()
                    a = spr.getchannel("A").point(lambda v: int(v * alpha))
                    spr.putalpha(a)
                img.alpha_composite(spr, (x, int(yy_)))
            else:
                img.alpha_composite(base, (x, int(yy_)))
                if row_frac > 0:
                    cw = int(hi.width * row_frac)
                    if cw > 2:
                        img.alpha_composite(hi.crop((0, 0, cw, hi.height)), (x, int(yy_)))
            y += h_ - 30


def scene_street(img, ts, t, en):
    dr = ImageDraw.Draw(img)
    dy = int(-6 * en["pulse"])
    scx, scy = W // 2, 430
    img.alpha_composite(GLOW, (scx - 430, scy - 430))
    sun = pick_sun(200, en["bass"])
    img.alpha_composite(sun, (scx - sun.width // 2, scy - sun.height // 2))
    draw_skyline(dr, dy, (60, 30, 16))
    en2 = dict(en)
    draw_boombox(img, dr, 60, 1500 + dy, en2, t)
    for k in range(3):
        u = ((t / beat_len + k * 0.33) % 1.0)
        nx = 700 + k * 120 - 40 * u
        ny = 700 - 120 * math.sin(u * math.pi)
        if u < 0.92:
            img.alpha_composite(NOTE_SPR[k % 3], (int(nx), int(ny)))
    draw_eq(dr, en, 1800 + dy, width=600, hmax=95)


def scene_party(img, ts, t, en):
    dr = ImageDraw.Draw(img)
    draw_rays(img, t, W // 2, 560, [(255, 150, 40), (255, 100, 50)], n=16, speed=0.55,
              alpha=60, boost=0.25 * en["dpulse"])
    img.alpha_composite(GLOW_HOT, (W // 2 - 470, 560 - 470))
    sun = pick_sun(260, en["bass"])
    img.alpha_composite(sun, (W // 2 - sun.width // 2, 560 - sun.height // 2))
    n_d = 6
    for k in range(n_d):
        x = int(W * (k + 0.65) / (n_d + 0.3))
        col = (43, 22, 10) if k % 2 == 0 else (60, 26, 12)
        draw_dancer(dr, x, H - 60, k, t, en, col, s=1.5 + 0.2 * ((k * 37 % 5) / 4))
    dr.rectangle([0, H - 50, W, H], fill=(43, 22, 10))
    for (x_, y_, ci, u) in confetti(ts):
        img.alpha_composite(CONF[ci].rotate(u * 720 + ci * 40, expand=False), (int(x_), int(y_)))


def render_frame(i):
    t = i / FPS            # clip-local time
    ts = START + t         # absolute song time
    en = E(ts)
    # scene select with crossfade at chorus
    u = ease_in_out((ts - (CH_T0 - 0.8)) / 0.8)
    imgs = []
    if u < 0.999:
        base = np.repeat(np.clip(GRAD_STREET * (1 + 0.06 * en["dpulse"]), 0, 255).astype(np.uint8), W, axis=1)
        im = Image.fromarray(base).convert("RGBA")
        scene_street(im, ts, t, en)
        imgs.append(im)
    if u > 0.001:
        base = np.repeat(np.clip(GRAD_PARTY * (1 + 0.10 * en["dpulse"]), 0, 255).astype(np.uint8), W, axis=1)
        im = Image.fromarray(base).convert("RGBA")
        scene_party(im, ts, t, en)
        imgs.append(im)
    img = imgs[0] if len(imgs) == 1 else Image.blend(imgs[0], imgs[1], u)
    ue = ease_in_out((t - (DUR - 1.15)) / 0.5) if t > DUR - 1.15 else 0.0
    draw_lines(img, ts, en, mul=1.0 - ue)
    # artist tag pill (top, inside safe zone)
    pad = 14
    ov = Image.new("RGBA", (TAG.width + pad * 2 + 14, TAG.height + pad), (0, 0, 0, 0))
    d2 = ImageDraw.Draw(ov)
    d2.rounded_rectangle([0, 0, ov.width - 1, ov.height - 1], radius=ov.height // 2,
                         fill=GOLD + (235,), outline=BROWN + (255,), width=3)
    ov.alpha_composite(TAG, (pad + 7, pad // 2 - 3))
    img.alpha_composite(ov, ((W - ov.width) // 2, 120))
    # end card
    if ue > 0:
        dark = Image.new("RGBA", (W, H), (10, 6, 3, int(140 * ue)))
        img.alpha_composite(dark)
        for spr, yv in ((END_T, 800), (CTA, 1010)):
            s2 = spr.copy()
            a = s2.getchannel("A").point(lambda v: int(v * ue))
            s2.putalpha(a)
            img.alpha_composite(s2, ((W - spr.width) // 2, yv))
    out = img.convert("RGB")
    out.paste(BLACK, (0, 0), GRAIN[i % 4])
    out.paste(BLACK, (0, 0), VIGN)
    if en["onset"] > 1.05 and t < DUR - 1.2:
        arr = np.asarray(out).copy()
        arr[:, 2:, 0] = arr[:, :-2, 0]
        out = Image.fromarray(arr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", type=str, default=None)
    ap.add_argument("--out", type=str, default=f"{SP}/let_it_ride_tiktok.mp4")
    args = ap.parse_args()
    if args.png:
        for s in args.png.split(","):
            i = int(s)
            render_frame(i).save(f"{SP}/preview/tt{i:04d}.png")
            print("saved preview", i)
        return
    FF = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [FF, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", AUDIO,
           "-filter_complex",
           f"[1:a]atrim=start={START:.3f}:end={END + 0.04:.3f},asetpts=PTS-STARTPTS,"
           f"afade=t=in:st=0:d=0.05,afade=t=out:st={DUR - 0.45:.3f}:d=0.45[a]",
           "-map", "0:v", "-map", "[a]",
           "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(NF):
        proc.stdin.write(render_frame(i).tobytes())
        if i % 200 == 0:
            print(f"frame {i}/{NF} ({time.time() - t0:.0f}s)", flush=True)
    proc.stdin.close()
    proc.wait()
    print("encoded", args.out)


if __name__ == "__main__":
    main()
