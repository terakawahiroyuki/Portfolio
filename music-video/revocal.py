#!/usr/bin/env python3
"""Gentler center-channel vocal extraction -> vocals16k.wav"""
import os as _os
SP = _os.environ.get("MV_WORKDIR", _os.path.dirname(_os.path.abspath(__file__)) or ".")
AUDIO = _os.environ.get("MV_AUDIO", _os.path.join(SP, "Let_It_Ride.mp3"))
import subprocess
import wave

import numpy as np
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()

cmd = [FF, "-v", "error", "-i", AUDIO, "-f", "f32le", "-ac", "2", "-ar", "16000", "-"]
raw = subprocess.run(cmd, capture_output=True, check=True).stdout
st = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)
midc = (st[:, 0] + st[:, 1]) * 0.5
side = (st[:, 0] - st[:, 1]) * 0.5

NF, H = 1024, 256
win = np.hanning(NF).astype(np.float32)
nfr = 1 + (len(midc) - NF) // H
idx = np.arange(NF)[None, :] + H * np.arange(nfr)[:, None]
M = np.fft.rfft(midc[idx] * win, axis=1)
S = np.fft.rfft(side[idx] * win, axis=1)
# gentle: subtract only 60% of side magnitude, keep a floor of 15% of mid
mag = np.maximum(np.abs(M) - 0.6 * np.abs(S), 0.15 * np.abs(M))
f = np.fft.rfftfreq(NF, 1 / 16000)
wgt = np.ones_like(f)
wgt[f < 120] = 0.1
V = mag * wgt * np.exp(1j * np.angle(M))
vfr = np.fft.irfft(V, axis=1).astype(np.float32) * win
out = np.zeros(len(midc), dtype=np.float32)
wsum = np.zeros(len(midc), dtype=np.float32)
for k in range(nfr):
    s0 = k * H
    out[s0:s0 + NF] += vfr[k]
    wsum[s0:s0 + NF] += win ** 2
out /= np.maximum(wsum, 1e-6)
out /= np.max(np.abs(out)) + 1e-9
pcm = (out * 30000).astype(np.int16)
with wave.open(f"{SP}/vocals16k.wav", "wb") as wv:
    wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(16000)
    wv.writeframes(pcm.tobytes())
print("saved gentler vocals16k.wav", len(pcm) / 16000, "s")
