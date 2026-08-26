import os as _os
SP = _os.environ.get("MV_WORKDIR", _os.path.dirname(_os.path.abspath(__file__)) or ".")
AUDIO = _os.environ.get("MV_AUDIO", _os.path.join(SP, "Let_It_Ride.mp3"))
#!/usr/bin/env python3
"""Audio analysis for the Let It Ride music video.

Decodes the MP3 with ffmpeg, computes STFT-based features with numpy only:
  - onset envelope + tempo + DP beat tracking + downbeat phase
  - per-video-frame band energies (bass/lowmid/mid/high) and RMS
  - 24-band log-spaced spectrum per video frame (for EQ bars)
  - structural novelty peaks (section boundary hints)
Saves everything to analysis.npz.

Also renders a center-channel "vocal" extraction to vocals16k.wav for
forced alignment with pocketsphinx.
"""
import json
import subprocess
import sys

import numpy as np
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
SR = 22050
FPS = 30
HOP = 512
NFFT = 2048


def decode(path, sr, channels):
    cmd = [FF, "-v", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le",
           "-ac", str(channels), "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    if channels > 1:
        x = x.reshape(-1, channels)
    return x


def stft_mag(x, nfft, hop):
    win = np.hanning(nfft).astype(np.float32)
    n = 1 + (len(x) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * win
    return np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)  # (n, nfft//2+1)


print("decoding...", flush=True)
mono = decode(AUDIO, SR, 1)
dur = len(mono) / SR
print(f"duration {dur:.2f}s ({dur//60:.0f}:{dur%60:04.1f})", flush=True)

print("stft...", flush=True)
S = stft_mag(mono, NFFT, HOP)  # frames at SR/HOP ≈ 43.07 fps
freqs = np.fft.rfftfreq(NFFT, 1 / SR)
tf = np.arange(S.shape[0]) * HOP / SR  # frame times

# --- onset envelope (log-magnitude positive spectral flux) ---
logS = np.log1p(S * 100)
flux = np.diff(logS, axis=0, prepend=logS[:1])
flux[flux < 0] = 0
onset = flux.sum(axis=1)
onset = onset / (np.percentile(onset, 98) + 1e-9)
# light smoothing
k = np.hanning(5); k /= k.sum()
onset_s = np.convolve(onset, k, mode="same")

# --- tempo via autocorrelation ---
frame_rate = SR / HOP
o = onset_s - onset_s.mean()
ac = np.correlate(o, o, mode="full")[len(o) - 1:]
ac /= (ac[0] + 1e-9)
lag_min = int(frame_rate * 60 / 180)
lag_max = int(frame_rate * 60 / 65)
lag = lag_min + np.argmax(ac[lag_min:lag_max])
tempo = 60 * frame_rate / lag
# prefer the 90-150 octave
for cand in (tempo, tempo * 2, tempo / 2):
    if 85 <= cand <= 155:
        tempo = cand
        break
period = frame_rate * 60 / tempo
print(f"tempo {tempo:.2f} BPM (period {period:.2f} frames)", flush=True)

# --- DP beat tracking (Ellis) ---
n = len(onset_s)
score = onset_s.copy()
back = np.full(n, -1, dtype=np.int64)
lo, hi = int(round(period * 0.5)), int(round(period * 2.0))
alpha = 100.0
for i in range(hi, n):
    window = np.arange(i - hi, i - lo + 1)
    trans = -alpha * (np.log(np.maximum(i - window, 1) / period)) ** 2
    vals = score[window] + trans
    j = np.argmax(vals)
    if vals[j] > 0:
        score[i] = onset_s[i] + vals[j]
        back[i] = window[j]
tail = np.arange(max(0, n - int(period * 2)), n)
b = tail[np.argmax(score[tail])]
beats = []
while b >= 0:
    beats.append(b)
    b = back[b]
beats = np.array(beats[::-1])
beat_times = beats * HOP / SR
print(f"{len(beats)} beats, first {beat_times[:4].round(2)}, last {beat_times[-2:].round(2)}", flush=True)

# --- downbeat phase: bass onset energy at beats ---
bass_bins = freqs < 160
bass_flux = flux[:, bass_bins].sum(axis=1)
phase_scores = [bass_flux[beats[p::4]].mean() for p in range(4)]
downbeat_phase = int(np.argmax(phase_scores))
print(f"downbeat phase {downbeat_phase} scores {np.round(phase_scores, 3)}", flush=True)

# --- per-video-frame features ---
nvf = int(np.ceil(dur * FPS))
vt = np.arange(nvf) / FPS


def band_energy(f_lo, f_hi):
    sel = (freqs >= f_lo) & (freqs < f_hi)
    e = np.sqrt((S[:, sel] ** 2).sum(axis=1))
    return np.interp(vt, tf, e)


def norm01(x, p_lo=10, p_hi=98):
    lo_, hi_ = np.percentile(x, p_lo), np.percentile(x, p_hi)
    return np.clip((x - lo_) / (hi_ - lo_ + 1e-9), 0, 1)


bass = norm01(band_energy(20, 160))
lowmid = norm01(band_energy(160, 500))
mid = norm01(band_energy(500, 2000))
high = norm01(band_energy(2000, 8000))
onset_v = np.clip(np.interp(vt, tf, onset_s), 0, 1.6)

# frame RMS
hop_rms = SR // FPS
pad = np.pad(mono, (0, max(0, nvf * hop_rms - len(mono))))
rms = np.sqrt((pad[: nvf * hop_rms].reshape(nvf, hop_rms) ** 2).mean(axis=1))
rms = norm01(rms, 5, 99)

# --- 24 log-spaced spectrum bands per video frame ---
NB = 24
edges = np.geomspace(40, 12000, NB + 1)
spec = np.zeros((S.shape[0], NB), dtype=np.float32)
for i in range(NB):
    sel = (freqs >= edges[i]) & (freqs < edges[i + 1])
    spec[:, i] = np.sqrt((S[:, sel] ** 2).sum(axis=1)) if sel.any() else 0
spec = np.log1p(spec * 50)
spec /= np.percentile(spec, 99, axis=0, keepdims=True) + 1e-9
spec_v = np.zeros((nvf, NB), dtype=np.float32)
for i in range(NB):
    spec_v[:, i] = np.clip(np.interp(vt, tf, spec[:, i]), 0, 1.15)

# --- structural novelty (checkerboard on cosine self-similarity) ---
NBS = 40
edges2 = np.geomspace(40, 10000, NBS + 1)
feat = np.zeros((S.shape[0], NBS), dtype=np.float32)
for i in range(NBS):
    sel = (freqs >= edges2[i]) & (freqs < edges2[i + 1])
    feat[:, i] = np.sqrt((S[:, sel] ** 2).sum(axis=1)) if sel.any() else 0
feat = np.log1p(feat * 50)
ds = max(1, int(frame_rate / 2))  # ~2 fps
m = (len(feat) // ds) * ds
featd = feat[:m].reshape(-1, ds, NBS).mean(axis=1)
featd -= featd.mean(axis=0)
normv = np.linalg.norm(featd, axis=1, keepdims=True) + 1e-9
Fn = featd / normv
SSM = Fn @ Fn.T
L = 16  # ±8 s at 2 fps
kern = np.outer(np.r_[np.ones(L), -np.ones(L)], np.r_[np.ones(L), -np.ones(L)])
kern *= np.outer(np.hanning(2 * L), np.hanning(2 * L))
nov = np.zeros(len(SSM))
for i in range(L, len(SSM) - L):
    nov[i] = (SSM[i - L:i + L, i - L:i + L] * kern).sum()
nov = np.clip(nov, 0, None)
nov /= nov.max() + 1e-9
nov_times = np.arange(len(nov)) * ds / frame_rate
# peak pick with 8 s min distance
peaks = []
order = np.argsort(nov)[::-1]
for i in order:
    if nov[i] < 0.12:
        break
    t = nov_times[i]
    if all(abs(t - p[0]) > 8 for p in peaks):
        peaks.append((float(t), float(nov[i])))
peaks.sort()
print("novelty peaks:", [f"{t:.1f}({v:.2f})" for t, v in peaks], flush=True)

np.savez_compressed(
    f"{SP}/analysis.npz",
    duration=dur, fps=FPS, tempo=tempo, downbeat_phase=downbeat_phase,
    beat_times=beat_times, bass=bass, lowmid=lowmid, mid=mid, high=high,
    onset=onset_v, rms=rms, spec=spec_v,
    novelty=np.array(peaks) if peaks else np.zeros((0, 2)),
)
print("saved analysis.npz", flush=True)

# --- vocal (center) extraction for forced alignment ---
print("vocal extraction...", flush=True)
st = decode(AUDIO, 16000, 2)
midc = (st[:, 0] + st[:, 1]) * 0.5
side = (st[:, 0] - st[:, 1]) * 0.5
NF2, H2 = 1024, 256
win2 = np.hanning(NF2).astype(np.float32)
nfr = 1 + (len(midc) - NF2) // H2
idx = np.arange(NF2)[None, :] + H2 * np.arange(nfr)[:, None]
M = np.fft.rfft(midc[idx] * win2, axis=1)
Ssp = np.fft.rfft(side[idx] * win2, axis=1)
mag = np.maximum(np.abs(M) - 1.0 * np.abs(Ssp), 0.0)
f2 = np.fft.rfftfreq(NF2, 1 / 16000)
wgt = np.ones_like(f2); wgt[f2 < 150] = 0.05; wgt[f2 > 6000] = 0.3
V = mag * wgt * np.exp(1j * np.angle(M))
vfr = np.fft.irfft(V, axis=1).astype(np.float32) * win2
out = np.zeros(len(midc), dtype=np.float32)
wsum = np.zeros(len(midc), dtype=np.float32)
for k2 in range(nfr):
    s0 = k2 * H2
    out[s0:s0 + NF2] += vfr[k2]
    wsum[s0:s0 + NF2] += win2 ** 2
out /= np.maximum(wsum, 1e-6)
out /= np.max(np.abs(out)) + 1e-9
pcm = (out * 32000).astype(np.int16)
import wave
with wave.open(f"{SP}/vocals16k.wav", "wb") as wv:
    wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(16000)
    wv.writeframes(pcm.tobytes())
print("saved vocals16k.wav", flush=True)
