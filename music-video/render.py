import os as _os
SP = _os.environ.get("MV_WORKDIR", _os.path.dirname(_os.path.abspath(__file__)) or ".")
AUDIO = _os.environ.get("MV_AUDIO", _os.path.join(SP, "Let_It_Ride.mp3"))
#!/usr/bin/env python3
"""Let It Ride — generative music video renderer.

Sunny old-school funk block-party aesthetic. Every visual element is driven
by the audio analysis (beats, band energies, spectrum) and the aligned lyric
timeline. Frames are streamed raw to ffmpeg and muxed with the original MP3.

Usage:
  python3 render.py --png 100,500,900     # dump preview frames
  python3 render.py                       # full render to let_it_ride_mv.mp4
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import imageio_ffmpeg

W, H = 1280, 720
FPS = 30

# ---------------------------------------------------------------- palette
CREAM = (255, 243, 214)
GOLD = (255, 201, 60)
ORANGE = (255, 138, 0)
CORAL = (255, 78, 80)
MAGENTA = (216, 78, 119)
TEAL = (31, 167, 160)
BROWN = (43, 27, 14)
PURPLE = (91, 58, 142)
INDIGO = (34, 27, 74)
NIGHT = (16, 12, 40)
SKYBLUE = (255, 214, 140)


def lerp(a, b, u):
    return a + (b - a) * u


def lerp_c(c1, c2, u):
    return tuple(int(round(lerp(c1[k], c2[k], u))) for k in range(3))


def ease_out(u):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


def ease_in_out(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------- analysis
A = np.load(f"{SP}/analysis.npz")
DUR = float(A["duration"])
NF = int(math.ceil(DUR * FPS))
tempo = float(A["tempo"])
beat_times = A["beat_times"]
dbp = int(A["downbeat_phase"])
bass, lowmid, mid_e, high_e = A["bass"], A["lowmid"], A["mid"], A["high"]
onset, rms, spec = A["onset"], A["rms"], A["spec"]
beat_len = 60.0 / tempo

# per-frame beat bookkeeping
vt = np.arange(NF) / FPS
bidx = np.searchsorted(beat_times, vt, side="right") - 1
tsb = vt - np.where(bidx >= 0, beat_times[np.clip(bidx, 0, None)], -beat_len)
is_down = (np.clip(bidx, 0, None) - dbp) % 4 == 0
last_db = np.maximum.accumulate(np.where(is_down & (bidx >= 0), vt - tsb, -10.0))
tsd = vt - last_db
PULSE = np.exp(-tsb * 6.0)
DPULSE = np.exp(-np.maximum(tsd, 0) * 3.2)
bar_idx = np.maximum((np.clip(bidx, 0, None) - dbp), 0) // 4


def E(i):
    i = min(max(i, 0), NF - 1)
    return dict(bass=float(bass[i]), lowmid=float(lowmid[i]), mid=float(mid_e[i]),
                high=float(high_e[i]), onset=float(onset[i]), rms=float(rms[i]),
                pulse=float(PULSE[i]), dpulse=float(DPULSE[i]),
                beat=int(bidx[i]), bar=int(bar_idx[i]), spec=spec[min(i, len(spec) - 1)])


# ---------------------------------------------------------------- timeline
with open(f"{SP}/timeline.json") as f:
    TL = json.load(f)
SECTIONS = TL["sections"]  # {name, kind, t0, t1}
LINES = TL["lines"]        # {sec, kind, text, t0, t1, words:[{w,s,e}]}

SCENE_OF_KIND = {
    "intro": "title", "verse": "street", "chorus": "party",
    "breakdown": "disco", "verse3": "dusk", "finalchorus": "party_max",
    "outro": "nightend",
}
for s in SECTIONS:
    s["scene"] = SCENE_OF_KIND[s["kind"]]

XFADE = 0.8


def scene_at(t):
    """Returns [(scene_dict, weight), ...] with transition crossfade."""
    cur = SECTIONS[0]
    for s in SECTIONS:
        if t >= s["t0"]:
            cur = s
    i = SECTIONS.index(cur)
    if i + 1 < len(SECTIONS):
        nxt = SECTIONS[i + 1]
        if t > nxt["t0"] - XFADE:
            u = ease_in_out((t - (nxt["t0"] - XFADE)) / XFADE)
            if u > 0.001:
                return [(cur, 1 - u), (nxt, u)]
    return [(cur, 1.0)]


# ---------------------------------------------------------------- fonts
F = f"{SP}/fonts"
fon = {
    "anton52": ImageFont.truetype(f"{F}/anton.ttf", 52),
    "anton64": ImageFont.truetype(f"{F}/anton.ttf", 64),
    "anton40": ImageFont.truetype(f"{F}/anton.ttf", 40),
    "anton30": ImageFont.truetype(f"{F}/anton.ttf", 30),
    "bungee90": ImageFont.truetype(f"{F}/bungee.ttf", 90),
    "bungee54": ImageFont.truetype(f"{F}/bungee.ttf", 54),
    "bungee36": ImageFont.truetype(f"{F}/bungee.ttf", 36),
    "arch24": ImageFont.truetype(f"{F}/archivo_black.ttf", 24),
    "arch20": ImageFont.truetype(f"{F}/archivo_black.ttf", 20),
    "arch32": ImageFont.truetype(f"{F}/archivo_black.ttf", 32),
}

# ---------------------------------------------------------------- gradients
def vgrad(stops):
    """stops: [(y_frac, color)] -> H x 1 x 3 uint8 column, tiled to W."""
    col = np.zeros((H, 3), dtype=np.float32)
    ys = [int(s[0] * (H - 1)) for s in stops]
    for k in range(len(stops) - 1):
        y0, y1 = ys[k], ys[k + 1]
        c0 = np.array(stops[k][1], dtype=np.float32)
        c1 = np.array(stops[k + 1][1], dtype=np.float32)
        n = max(y1 - y0, 1)
        ramp = np.linspace(0, 1, n)[:, None]
        col[y0:y1] = c0 + (c1 - c0) * ramp
    col[ys[-1]:] = np.array(stops[-1][1], dtype=np.float32)
    return np.repeat(col[:, None, :], 1, axis=1)


GRADS = {
    "title": vgrad([(0, (30, 14, 8)), (0.45, (120, 38, 18)), (0.78, (233, 110, 30)), (1, (255, 190, 90))]),
    "street": vgrad([(0, (255, 148, 40)), (0.5, (255, 196, 90)), (0.85, (255, 232, 170)), (1, (255, 240, 200))]),
    "party": vgrad([(0, (255, 120, 40)), (1, (255, 170, 60))]),
    "party_max": vgrad([(0, (255, 96, 60)), (1, (255, 160, 70))]),
    "disco": vgrad([(0, (18, 12, 48)), (0.6, (52, 30, 96)), (1, (110, 50, 120))]),
    "dusk": vgrad([(0, (40, 26, 88)), (0.45, (120, 52, 110)), (0.8, (235, 110, 70)), (1, (255, 170, 90))]),
    "nightend": vgrad([(0, (10, 8, 30)), (0.6, (24, 18, 56)), (1, (52, 34, 84))]),
}


def sky_np(scene, i, en):
    g = GRADS[scene]
    b = 1.0 + 0.10 * en["dpulse"] * (1.2 if scene in ("party", "party_max") else 0.6)
    arr = np.clip(g * b, 0, 255).astype(np.uint8)
    return np.repeat(arr, W, axis=1)


# ---------------------------------------------------------------- sprites
def radial_glow(radius, color, peak=180):
    d = radius * 2
    yy, xx = np.mgrid[0:d, 0:d]
    r = np.sqrt((yy - radius) ** 2 + (xx - radius) ** 2) / radius
    a = np.clip(1 - r, 0, 1) ** 2.2 * peak
    img = np.zeros((d, d, 4), dtype=np.uint8)
    img[..., :3] = color
    img[..., 3] = a.astype(np.uint8)
    return Image.fromarray(img, "RGBA")


def retro_sun(radius, c_top, c_bot):
    """Classic retro sun with horizontal slats in the lower half."""
    d = radius * 2
    img = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    for y in range(d):
        u = y / d
        col = lerp_c(c_top, c_bot, u)
        x = radius - math.sqrt(max(radius ** 2 - (y - radius) ** 2, 0))
        dr.line([(x, y), (d - x, y)], fill=col + (255,))
    # slats
    n = 7
    for k in range(n):
        yy0 = int(d * (0.52 + 0.48 * (k / n)))
        hgt = int(2 + 9 * (k / n))
        dr.rectangle([0, yy0, d, yy0 + hgt], fill=(0, 0, 0, 0))
    return img


SUN_GLOW = radial_glow(300, (255, 220, 140), 150)
SUN_GLOW_HOT = radial_glow(320, (255, 120, 60), 170)
SUNS = {r: retro_sun(r, (255, 244, 200), (255, 150, 40)) for r in (120, 130, 140, 150, 160, 170, 185)}
MOON = retro_sun(90, (240, 240, 255), (170, 170, 230))


def pick_sun(base_r, boost):
    keys = sorted(SUNS)
    want = base_r * (1 + 0.10 * boost)
    return SUNS[min(keys, key=lambda k: abs(k - want))]


# skyline layout (deterministic)
rng = random.Random(7)
BUILDINGS = []
x = -20
while x < W + 40:
    bw = rng.randint(60, 130)
    bh = rng.randint(90, 230)
    BUILDINGS.append((x, bw, bh, rng.random()))
    x += bw + rng.randint(4, 14)

STARS = [(rng.randint(0, W), rng.randint(0, int(H * 0.55)), rng.random()) for _ in range(90)]

# grain + vignette
g_rng = np.random.default_rng(3)
GRAIN = [Image.fromarray(g_rng.integers(0, 26, (H // 2, W // 2), dtype=np.uint8).repeat(2, 0).repeat(2, 1), "L")
         for _ in range(4)]
yy, xx = np.mgrid[0:H, 0:W]
r_norm = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
VIGN = Image.fromarray((np.clip(r_norm - 0.62, 0, 1) ** 1.8 * 190).astype(np.uint8), "L")
BLACK = Image.new("RGB", (W, H), (8, 5, 3))


def apply_post(img, i, en, scene):
    # grain + vignette
    img.paste(BLACK, (0, 0), GRAIN[i % 4].point(lambda v: v // 3))
    img.paste(BLACK, (0, 0), VIGN)
    return img


# ---------------------------------------------------------------- lyric sprites
def normline(text):
    return text.replace("’", "'")


def render_text_sprite(text, font, fill, stroke, sw=6, shadow=True):
    tmp = Image.new("RGBA", (10, 10))
    d0 = ImageDraw.Draw(tmp)
    bb = d0.textbbox((0, 0), text, font=font, stroke_width=sw)
    w, h = bb[2] - bb[0] + 16, bb[3] - bb[1] + 18
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    ox, oy = 8 - bb[0], 8 - bb[1]
    if shadow:
        dr.text((ox + 4, oy + 5), text, font=font, fill=(20, 8, 4, 160), stroke_width=sw,
                stroke_fill=(20, 8, 4, 160))
    dr.text((ox, oy), text, font=font, fill=fill + (255,), stroke_width=sw, stroke_fill=stroke + (255,))
    return img


def wrap_text(text, font, maxw):
    words = text.split()
    lines, cur = [], ""
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    for w_ in words:
        t = (cur + " " + w_).strip()
        if tmp.textlength(t, font=font) > maxw and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines


LINE_STYLE = {
    "intro":   dict(font="anton40", fill=CREAM, stroke=BROWN, hi=GOLD, y=560, maxw=900),
    "verse":   dict(font="anton52", fill=CREAM, stroke=BROWN, hi=GOLD, y=500, maxw=1080),
    "verse3":  dict(font="anton52", fill=CREAM, stroke=(24, 12, 40), hi=(255, 210, 120), y=500, maxw=1080),
    "chorus":  dict(font="anton64", fill=CREAM, stroke=(105, 26, 16), hi=GOLD, y=470, maxw=1100),
    "finalchorus": dict(font="anton64", fill=CREAM, stroke=(105, 26, 16), hi=GOLD, y=470, maxw=1100),
    "breakdown": dict(font="bungee54", fill=(255, 220, 120), stroke=(40, 16, 70), hi=(255, 255, 255), y=612, maxw=1100),
    "outro":   dict(font="anton52", fill=(230, 220, 255), stroke=(18, 12, 44), hi=(255, 255, 255), y=330, maxw=1000),
}

HOOK = "LET IT RIDE"


def is_hook(text):
    t = text.lower()
    return t.count("let it ride") >= 1 and len(t) <= 30


LINE_SPRITES = []  # per line: dict(kind, rows=[{base, hi, w, h}], words, t0, t1)
for ln in LINES:
    st = LINE_STYLE[ln["kind"]]
    font = fon[st["font"]]
    text = normline(ln["text"]).upper()
    if is_hook(ln["text"]) and ln["kind"] in ("chorus", "finalchorus", "outro", "intro"):
        LINE_SPRITES.append(dict(ln=ln, hook=True, rows=[], style=st))
        continue
    rows = []
    for row in wrap_text(text, font, st["maxw"]):
        base = render_text_sprite(row, font, st["fill"], st["stroke"])
        hi = render_text_sprite(row, font, st["hi"], st["stroke"], shadow=False)
        rows.append(dict(base=base, hi=hi))
    LINE_SPRITES.append(dict(ln=ln, hook=False, rows=rows, style=st))

# a line's display is cut when the next line starts (no overlapping captions)
for k, sp in enumerate(LINE_SPRITES):
    t1 = sp["ln"]["t1"]
    cut = t1 + 0.45
    if k + 1 < len(LINE_SPRITES):
        nxt = LINE_SPRITES[k + 1]["ln"]["t0"]
        cut = min(cut, max(nxt - 0.03, t1 - 0.2))
    sp["cut"] = cut

# hook letters (Bungee) rendered individually
HOOK_LETTERS = []
for ch in HOOK:
    if ch == " ":
        HOOK_LETTERS.append(None)
        continue
    spr = render_text_sprite(ch, fon["bungee90"], GOLD, (90, 20, 10), sw=7)
    hi = render_text_sprite(ch, fon["bungee90"], CREAM, (120, 30, 12), sw=7, shadow=False)
    HOOK_LETTERS.append((spr, hi))

TITLE_LETTERS = []
for ch in HOOK:
    if ch == " ":
        TITLE_LETTERS.append(None)
        continue
    spr = render_text_sprite(ch, ImageFont.truetype(f"{F}/anton.ttf", 150), CREAM, BROWN, sw=8)
    TITLE_LETTERS.append(spr)

SEC_TAGS = {}
for s in SECTIONS:
    label = s["name"].upper()
    t = render_text_sprite(label, fon["arch24"], BROWN, CREAM, sw=0, shadow=False)
    SEC_TAGS[s["name"]] = t

CRED1 = render_text_sprite("ELECTRON6", fon["arch32"], CREAM, BROWN, sw=4)
CRED2 = render_text_sprite("PRESENTS", fon["arch20"], GOLD, BROWN, sw=3)
END1 = render_text_sprite("LET IT RIDE", ImageFont.truetype(f"{F}/anton.ttf", 110), CREAM, (18, 12, 44), sw=8)
END2 = render_text_sprite("ELECTRON6", fon["arch32"], GOLD, (18, 12, 44), sw=4)
END3 = render_text_sprite("MADE WITH SUNO", fon["arch20"], (200, 190, 235), (18, 12, 44), sw=3)

NOTE_SPR = []
for sz, col in [(34, CORAL), (28, TEAL), (40, GOLD)]:
    NOTE_SPR.append(render_text_sprite("♪", ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", sz), col, BROWN, sw=3, shadow=False))

# confetti sprites
CONF = []
c_rng = random.Random(11)
for col in (GOLD, CORAL, TEAL, CREAM, MAGENTA):
    s = Image.new("RGBA", (14, 10), (0, 0, 0, 0))
    ImageDraw.Draw(s).rectangle([2, 2, 11, 7], fill=col + (235,))
    CONF.append(s)
SPARK = render_text_sprite("✦", ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 38), CREAM, PURPLE, sw=2, shadow=False)


# ---------------------------------------------------------------- scene pieces
def draw_skyline(img, dr, dy, color, en, windows="off", win_dim=1.0):
    horizon = H - 150 + dy
    for bi, (bx, bw, bh, seed) in enumerate(BUILDINGS):
        top = horizon - bh
        dr.rectangle([bx, top, bx + bw, H], fill=color)
        if windows != "off":
            band = int(seed * 23) % 24
            lvl = en["spec"][band] * win_dim
            cols = max(bw // 26, 2)
            rows = max(bh // 34, 2)
            lit = int(rows * min(lvl, 1.0) + 0.35)
            for cx in range(cols):
                for ry in range(lit):
                    wx = bx + 10 + cx * (bw - 14) / cols
                    wy = horizon - 18 - ry * 30
                    if wy < top + 6:
                        break
                    wcol = (255, 214, 110) if windows == "night" else (255, 240, 200)
                    dr.rectangle([wx, wy, wx + 10, wy + 14], fill=wcol)
    dr.rectangle([0, horizon, W, H], fill=color)


def draw_ground_stripes(dr, dy, c1, c2):
    horizon = H - 150 + dy
    dr.rectangle([0, horizon, W, H], fill=c1)
    for k in range(5):
        yv = horizon + 24 + k * 28
        dr.rectangle([0, yv, W, yv + 10], fill=c2)


def draw_rays(img, t, cx, cy, colors, n=14, speed=0.35, alpha=70, boost=0.0):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    rot = t * speed + boost
    R = 1600
    for k in range(n):
        a0 = rot + k * 2 * math.pi / n
        a1 = a0 + math.pi / n * 0.85
        p = [(cx, cy),
             (cx + R * math.cos(a0), cy + R * math.sin(a0)),
             (cx + R * math.cos(a1), cy + R * math.sin(a1))]
        dr.polygon(p, fill=colors[k % len(colors)] + (alpha,))
    img.alpha_composite(ov)


def draw_eq(dr, en, y_base, width=560, hmax=110, colors=(GOLD, CORAL, TEAL)):
    n = 24
    bw = width / n
    x0 = (W - width) / 2
    for k in range(n):
        v = en["spec"][k]
        bh = 6 + v * hmax
        c = colors[k % len(colors)]
        dr.rounded_rectangle([x0 + k * bw + 2, y_base - bh, x0 + (k + 1) * bw - 2, y_base],
                             radius=4, fill=c)


def draw_boombox(img, dr, x, y, en, scale=1.0):
    s = scale
    bounce = -10 * en["pulse"] * s
    y = y + bounce
    bw, bh = 340 * s, 150 * s
    dr.rounded_rectangle([x, y, x + bw, y + bh], radius=18 * s, fill=(50, 34, 26), outline=CREAM, width=int(4 * s))
    dr.rounded_rectangle([x + 8 * s, y + 8 * s, x + bw - 8 * s, y + 30 * s], radius=8 * s, fill=(72, 50, 38))
    # handle
    dr.arc([x + 80 * s, y - 52 * s, x + bw - 80 * s, y + 10 * s], 180, 360, fill=(50, 34, 26), width=int(10 * s))
    # speakers
    for scx in (x + 78 * s, x + bw - 78 * s):
        r0 = 52 * s
        dr.ellipse([scx - r0, y + 88 * s - r0 + 8 * s, scx + r0, y + 88 * s + r0 + 8 * s], fill=(26, 18, 12),
                   outline=GOLD, width=int(4 * s))
        r1 = (20 + 26 * en["bass"]) * s
        dr.ellipse([scx - r1, y + 96 * s - r1, scx + r1, y + 96 * s + r1], fill=ORANGE)
        r2 = max((8 + 10 * en["mid"]) * s, 2)
        dr.ellipse([scx - r2, y + 96 * s - r2, scx + r2, y + 96 * s + r2], fill=CREAM)
    # cassette window
    cx0 = x + bw / 2 - 52 * s
    dr.rounded_rectangle([cx0, y + 52 * s, cx0 + 104 * s, y + 118 * s], radius=8 * s, fill=(210, 190, 150))
    ang = en.get("_t", 0) * 4
    for wx in (cx0 + 28 * s, cx0 + 76 * s):
        dr.ellipse([wx - 14 * s, y + 71 * s, wx + 14 * s, y + 99 * s], fill=(70, 50, 36))
        for a in (ang, ang + 2.1, ang + 4.2):
            dr.line([(wx, y + 85 * s), (wx + 11 * s * math.cos(a), y + 85 * s + 11 * s * math.sin(a))],
                    fill=(230, 220, 190), width=max(int(3 * s), 1))


def draw_dancer(dr, x, base_y, k, t, en, col, scale=1.0):
    """Simple funky silhouette with hands up, bouncing on beats."""
    s = scale
    phase = k * 0.7
    bounce = -26 * s * en["pulse"] * (0.6 + 0.4 * math.sin(k * 2.4))
    sway = math.sin(t * 2 * math.pi * tempo / 60 / 2 + phase) * 8 * s
    y = base_y + bounce
    hx = x + sway * 0.4
    # head
    dr.ellipse([hx - 13 * s, y - 118 * s, hx + 13 * s, y - 92 * s], fill=col)
    # body
    dr.polygon([(hx - 15 * s, y - 88 * s), (hx + 15 * s, y - 88 * s),
                (x + 20 * s, y - 20 * s), (x - 20 * s, y - 20 * s)], fill=col)
    # legs
    dr.polygon([(x - 16 * s, y - 24 * s), (x - 4 * s, y - 24 * s), (x - 10 * s - sway * 0.3, y)], fill=col)
    dr.polygon([(x + 16 * s, y - 24 * s), (x + 4 * s, y - 24 * s), (x + 10 * s + sway * 0.3, y)], fill=col)
    # arms up, waving
    wa = math.sin(t * 2 * math.pi * tempo / 60 + phase) * 0.5
    for sgn in (-1, 1):
        a = -math.pi / 2 + sgn * (0.55 + 0.25 * wa)
        ex = hx + sgn * 14 * s + 46 * s * math.cos(a)
        ey = y - 84 * s + 46 * s * math.sin(a)
        dr.line([(hx + sgn * 14 * s, y - 82 * s), (ex, ey)], fill=col, width=int(9 * s))
        dr.ellipse([ex - 7 * s, ey - 7 * s, ex + 7 * s, ey + 7 * s], fill=col)


def draw_vinyl(img, t, en, cx, cy, radius, spin=1.0):
    d = radius * 2
    v = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    dr = ImageDraw.Draw(v)
    dr.ellipse([0, 0, d, d], fill=(16, 12, 18, 255), outline=(200, 180, 220, 255), width=3)
    lab = max(int(radius * 0.26), 18)
    for rr in range(radius - 16, lab + 6, -11):
        dr.ellipse([radius - rr, radius - rr, radius + rr, radius + rr], outline=(48, 40, 56, 255), width=2)
    # label
    dr.ellipse([radius - lab, radius - lab, radius + lab, radius + lab], fill=ORANGE)
    dr.ellipse([radius - lab + 2, radius - lab + 2, radius + lab - 2, radius + lab - 2], outline=CREAM, width=3)
    dr.ellipse([radius - 7, radius - 7, radius + 7, radius + 7], fill=(16, 12, 18, 255))
    ang = math.degrees(t * 2.4 * spin + 2.2 * en["pulse"])
    v = v.rotate(-ang, resample=Image.BILINEAR)
    dr2 = ImageDraw.Draw(v)
    # static glint
    dr2.arc([18, 18, d - 18, d - 18], 200, 250, fill=(235, 225, 255, 130), width=7)
    img.alpha_composite(v, (int(cx - radius), int(cy - radius)))


def confetti_positions(t, t_start, n_per_beat=26, life=2.6):
    """Deterministic confetti bursts on each beat since t_start."""
    out = []
    b0 = np.searchsorted(beat_times, t_start)
    b1 = np.searchsorted(beat_times, t)
    for b in range(max(b0, b1 - 5), b1):
        bt = beat_times[b]
        age = t - bt
        if age > life or age < 0:
            continue
        r = random.Random(9000 + b)
        for k in range(n_per_beat):
            x0 = r.uniform(0, W)
            vy0 = r.uniform(-260, -80)
            vx = r.uniform(-70, 70)
            y0 = r.uniform(H * 0.25, H * 0.6)
            x_ = x0 + vx * age
            y_ = y0 + vy0 * age + 380 * age * age * 0.5
            if -20 < y_ < H + 20:
                out.append((x_, y_, (b * 7 + k) % len(CONF), age / life))
    return out


# ---------------------------------------------------------------- lyric engine
def active_lines(t):
    out = []
    for i, sp in enumerate(LINE_SPRITES):
        ln = sp["ln"]
        if ln["t0"] - 0.22 <= t <= sp["cut"]:
            out.append(sp)
    return out


def karaoke_frac(ln, t):
    ws = ln.get("words") or []
    if not ws:
        u = (t - ln["t0"]) / max(ln["t1"] - ln["t0"], 0.01)
        return min(max(u, 0), 1)
    if t >= ws[-1]["e"]:
        return 1.0
    total = len(ws)
    for k, w_ in enumerate(ws):
        if t < w_["s"]:
            return k / total
        if t <= w_["e"]:
            wu = (t - w_["s"]) / max(w_["e"] - w_["s"], 0.01)
            return (k + wu) / total
    return 1.0


def draw_line_sprite(img, sp, t, en):
    ln, st = sp["ln"], sp["style"]
    t0, t1 = ln["t0"], ln["t1"]
    a_in = ease_out((t - (t0 - 0.22)) / 0.28)
    fade_w = max(sp["cut"] - t1, 0.06)
    a_out = 1.0 if t < t1 else max(0.0, 1 - (t - t1) / fade_w)
    alpha = a_in * a_out
    if alpha <= 0.01:
        return
    if sp["hook"]:
        hy = {"intro": 435, "outro": 260}.get(ln["kind"], st.get("y", 460) - 60)
        draw_hook(img, t, en, y=hy, alpha=alpha, t0=t0)
        return
    rows = sp["rows"]
    frac = karaoke_frac(ln, t)
    pop = 1.0 + 0.16 * (1 - a_in)
    total_h = sum(r["base"].height - 24 for r in rows) + 24
    y = st["y"] - total_h // 2
    n_rows = len(rows)
    for ri, r in enumerate(rows):
        base, hi = r["base"], r["hi"]
        w_, h_ = base.size
        x = (W - w_) // 2
        yy_ = y + int(4 * math.sin(t * 3 + ri)) - int(6 * en["pulse"])
        row_frac = min(max(frac * n_rows - ri, 0), 1)
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
        y += h_ - 24


def draw_hook(img, t, en, y=400, alpha=1.0, t0=0.0, letters=None, arc=26, spacing=6):
    letters = letters or HOOK_LETTERS
    widths = [(l[0].width if isinstance(l, tuple) else (l.width if l else 46)) for l in letters]
    widths = [(w_ - 32) for w_ in widths]
    gap = spacing
    total = sum(widths) + gap * (len(letters) - 1) + 80
    x = (W - total) // 2
    n = len(letters)
    for k, l in enumerate(letters):
        if l is None:
            x += widths[k] + gap + 14
            continue
        spr, hi = l if isinstance(l, tuple) else (l, None)
        cx = (k / (n - 1)) - 0.5
        arc_y = arc * (cx * cx * 4 - 1) * 0.5
        # letters bounce in a wave following the beat
        wave = math.sin((t * tempo / 60) * math.pi * 2 - k * 0.55)
        by = -14 * en["pulse"] * max(wave, 0) - 8 * en["dpulse"]
        use = spr
        if hi and en["pulse"] > 0.82:
            use = hi
        if alpha < 0.999:
            use = use.copy()
            a = use.getchannel("A").point(lambda v: int(v * alpha))
            use.putalpha(a)
        img.alpha_composite(use, (int(x - 20), int(y + arc_y + by)))
        x += widths[k] + gap
    return total


# section tag stinger
def draw_section_tag(img, t):
    for s in SECTIONS:
        dt = t - s["t0"]
        if 0 <= dt <= 2.6 and s["kind"] not in ("intro", "outro"):
            tag = SEC_TAGS[s["name"]]
            u = ease_out(dt / 0.4)
            fade = 1.0 if dt < 2.1 else 1 - (dt - 2.1) / 0.5
            x = int(-tag.width - 30 + (70 + tag.width) * u)
            pad = 12
            ov = Image.new("RGBA", (tag.width + pad * 2 + 16, tag.height + pad), (0, 0, 0, 0))
            d2 = ImageDraw.Draw(ov)
            d2.rounded_rectangle([0, 0, ov.width - 1, ov.height - 1], radius=ov.height // 2,
                                 fill=GOLD + (int(235 * fade),), outline=BROWN + (int(255 * fade),), width=3)
            ov.alpha_composite(tag, (pad + 8, pad // 2 - 2))
            if fade < 1:
                a = ov.getchannel("A").point(lambda v: int(v * fade))
                ov.putalpha(a)
            img.alpha_composite(ov, (x - tag.width // 2 + 40, 46))


# ---------------------------------------------------------------- scenes
def scene_title(img, t, i, en):
    dr = ImageDraw.Draw(img)
    dy = int(-4 * en["pulse"])
    # rising sun
    rise = ease_in_out(min(t / 7.0, 1))
    scy = int(H * 0.95 - rise * H * 0.52)
    img.alpha_composite(SUN_GLOW, (W // 2 - 300, scy - 300))
    sun = pick_sun(150, en["bass"])
    img.alpha_composite(sun, (W // 2 - sun.width // 2, scy - sun.height // 2))
    draw_skyline(img, dr, dy + 60, (30, 16, 10), en, windows="off")
    # credits then title
    if t < 2.6:
        u = ease_out(t / 0.7)
        fade = 1.0 if t < 2.0 else max(0.0, 1 - (t - 2.0) / 0.6)
        for spr, yv in ((CRED1, 300), (CRED2, 352)):
            s2 = spr.copy()
            a = s2.getchannel("A").point(lambda v: int(v * u * fade))
            s2.putalpha(a)
            img.alpha_composite(s2, ((W - spr.width) // 2, yv))
    else:
        # title letters drop in on successive beats
        widths = [(l.width - 26 if l else 50) for l in TITLE_LETTERS]
        total = sum(widths) + 8 * (len(TITLE_LETTERS) - 1)
        x = (W - total) // 2
        t_start = 2.6
        for k, l in enumerate(TITLE_LETTERS):
            if l is None:
                x += widths[k] + 8 + 12
                continue
            lt = t_start + k * 0.16
            u = ease_out((t - lt) / 0.30)
            if u > 0:
                y0 = -180 + (300) * u
                wave = math.sin((t * tempo / 60) * math.pi * 2 - k * 0.5)
                by = -12 * en["pulse"] * max(wave, 0)
                img.alpha_composite(l, (int(x - 20), int(y0 + by)))
            x += widths[k] + 8
        draw_eq(dr, en, H - 60, width=640, hmax=70)


def scene_street(img, t, i, en, dusk=False):
    dr = ImageDraw.Draw(img)
    dy = int(-5 * en["pulse"])
    scx, scy = int(W * 0.72), int(H * 0.30)
    if dusk:
        scy = int(H * 0.42)
        for sx, sy, ss in STARS:
            a = int(120 * ss * (0.6 + 0.4 * math.sin(t * 2 + ss * 9)))
            dr.ellipse([sx, sy, sx + 2 + ss, sy + 2 + ss], fill=(255, 244, 220, a))
    img.alpha_composite(SUN_GLOW, (scx - 300, scy - 300))
    sun = pick_sun(130 if not dusk else 140, en["bass"])
    img.alpha_composite(sun, (scx - sun.width // 2, scy - sun.height // 2))
    col = (60, 30, 16) if not dusk else (26, 16, 44)
    draw_skyline(img, dr, dy, col, en, windows=("night" if dusk else "off"))
    draw_ground_stripes(dr, dy, lerp_c(col, (0, 0, 0), 0.25), lerp_c(col, (255, 255, 255), 0.06))
    en2 = dict(en); en2["_t"] = t
    draw_boombox(img, dr, 46, H - 200 + dy, en2, scale=0.72)
    # bouncing notes over the skyline
    for k in range(3):
        nb = (en["beat"] + k) % 3
        u = ((t / beat_len + k * 0.33) % 1.0)
        nx = 950 + k * 90 - 40 * u
        ny = H - 420 - 90 * math.sin(u * math.pi)
        spr = NOTE_SPR[k % 3]
        if u < 0.92:
            img.alpha_composite(spr, (int(nx), int(ny)))
    draw_eq(dr, en, H - 26 + dy, width=430, hmax=56)


def scene_party(img, t, i, en, mx=False):
    dr = ImageDraw.Draw(img)
    cols = [(255, 150, 40), (255, 100, 50)] if not mx else [(255, 120, 60), (255, 80, 90)]
    draw_rays(img, t, W // 2, int(H * 0.42), cols, n=16, speed=0.45 if not mx else 0.7,
              alpha=60, boost=0.25 * en["dpulse"])
    img.alpha_composite(SUN_GLOW_HOT, (W // 2 - 320, int(H * 0.42) - 320))
    sun = pick_sun(160 if not mx else 170, en["bass"])
    img.alpha_composite(sun, (W // 2 - sun.width // 2, int(H * 0.42) - sun.height // 2))
    # dancers
    base_y = H - 40
    n_d = 8 if not mx else 10
    for k in range(n_d):
        x = int(W * (k + 0.7) / (n_d + 0.4))
        col = (43, 22, 10) if k % 2 == 0 else (60, 26, 12)
        draw_dancer(dr, x, base_y, k, t, en, col, scale=1.0 + 0.15 * ((k * 37 % 5) / 4))
    dr.rectangle([0, H - 34, W, H], fill=(43, 22, 10))
    if mx:
        for (x_, y_, ci, u) in confetti_positions(t, scene_start(t), n_per_beat=22):
            spr = CONF[ci]
            img.alpha_composite(spr.rotate(u * 720 + ci * 40, expand=False), (int(x_), int(y_)))


def scene_disco(img, t, i, en):
    dr = ImageDraw.Draw(img)
    draw_rays(img, t, W // 2, int(H * 0.44), [(120, 70, 200), (80, 40, 160)], n=12, speed=0.18, alpha=48)
    draw_vinyl(img, t, en, W // 2, int(H * 0.46), 210, spin=1.0)
    # sparkles on beats
    r = random.Random(400 + en["beat"])
    for k in range(10):
        if r.random() < 0.7:
            sx, sy = r.randint(60, W - 60), r.randint(40, H - 160)
            u = 1 - en["pulse"]
            s2 = SPARK.copy()
            a = s2.getchannel("A").point(lambda v: int(v * max(0.15, en["pulse"])))
            s2.putalpha(a)
            img.alpha_composite(s2, (sx, sy))
    dr.rectangle([0, H - 60, W, H], fill=(14, 10, 30))
    draw_eq(dr, en, H - 14, width=760, hmax=44, colors=(PURPLE, MAGENTA, TEAL))


def scene_nightend(img, t, i, en):
    dr = ImageDraw.Draw(img)
    for sx, sy, ss in STARS:
        a = int(150 * ss * (0.6 + 0.4 * math.sin(t * 1.6 + ss * 9)))
        dr.ellipse([sx, sy, sx + 2 + ss, sy + 2 + ss], fill=(255, 244, 220, a))
    img.alpha_composite(MOON, (int(W * 0.78) - 90, int(H * 0.20) - 90))
    draw_skyline(img, dr, 0, (18, 14, 38), en, windows="night", win_dim=0.5)
    s0 = section_by_kind("outro")
    dt = t - s0["t0"]
    # end card once outro lyrics wind down
    tail = DUR - 12.0
    if t > tail:
        u = ease_in_out((t - tail) / 1.2)
        for spr, yv in ((END1, 250), (END2, 400), (END3, 452)):
            s2 = spr.copy()
            a = s2.getchannel("A").point(lambda v: int(v * u))
            s2.putalpha(a)
            img.alpha_composite(s2, ((W - spr.width) // 2, yv))
        # spinning-down vinyl
        draw_vinyl(img, t * max(0.15, 1 - (t - tail) / 10), en, W // 2, H - 90, 70, spin=1.0)


def section_by_kind(kind):
    for s in SECTIONS:
        if s["kind"] == kind:
            return s
    return SECTIONS[-1]


_scene_start_cache = {}


def scene_start(t):
    cur = SECTIONS[0]
    for s in SECTIONS:
        if t >= s["t0"]:
            cur = s
    return cur["t0"]


SCENE_FN = {
    "title": scene_title,
    "street": scene_street,
    "party": lambda img, t, i, en: scene_party(img, t, i, en, mx=False),
    "party_max": lambda img, t, i, en: scene_party(img, t, i, en, mx=True),
    "disco": scene_disco,
    "dusk": lambda img, t, i, en: scene_street(img, t, i, en, dusk=True),
    "nightend": scene_nightend,
}


def render_frame(i):
    t = i / FPS
    en = E(i)
    layers = scene_at(t)
    imgs = []
    for s, wgt in layers:
        base = sky_np(s["scene"], i, en)
        img = Image.fromarray(base).convert("RGBA")
        SCENE_FN[s["scene"]](img, t, i, en)
        imgs.append((img, wgt))
    if len(imgs) == 1:
        img = imgs[0][0]
    else:
        img = Image.blend(imgs[0][0], imgs[1][0], imgs[1][1])
    # lyrics
    for sp in active_lines(t):
        draw_line_sprite(img, sp, t, en)
    draw_section_tag(img, t)
    # progress tape line
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, H - 4, int(W * t / DUR), H], fill=GOLD + (200,))
    out = img.convert("RGB")
    out = apply_post(out, i, en, layers[0][0]["scene"])
    # end fade
    if t > DUR - 1.6:
        u = (t - (DUR - 1.6)) / 1.6
        arr = np.asarray(out).astype(np.uint16)
        arr = (arr * max(0.0, 1 - u)).astype(np.uint8)
        out = Image.fromarray(arr)
    # onset chromatic kick
    if en["onset"] > 1.05 and t < DUR - 2:
        arr = np.asarray(out).copy()
        arr[:, 2:, 0] = arr[:, :-2, 0]
        out = Image.fromarray(arr)
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", type=str, default=None)
    ap.add_argument("--range", type=str, default=None, help="start,end seconds (test encode)")
    ap.add_argument("--out", type=str, default=f"{SP}/let_it_ride_mv.mp4")
    args = ap.parse_args()

    if args.png:
        os.makedirs(f"{SP}/preview", exist_ok=True)
        for s in args.png.split(","):
            i = int(s)
            fr = render_frame(i)
            p = f"{SP}/preview/f{i:05d}.png"
            fr.save(p)
            print("saved", p)
        return

    i0, i1 = 0, NF
    if args.range:
        a, b = args.range.split(",")
        i0, i1 = int(float(a) * FPS), int(float(b) * FPS)

    FF = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [FF, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-ss", f"{i0 / FPS:.3f}", "-i", AUDIO,
           "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
           args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(i0, i1):
        fr = render_frame(i)
        proc.stdin.write(fr.tobytes())
        if (i - i0) % 300 == 0:
            el = time.time() - t0
            done = i - i0 + 1
            eta = el / done * (i1 - i0 - done)
            print(f"frame {i}/{i1}  {el:.0f}s elapsed, eta {eta:.0f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    print("encoded", args.out, flush=True)


if __name__ == "__main__":
    main()
