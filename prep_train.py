#!/usr/bin/env python3
"""SoundCue - reconstructed training recipe.

STATUS: this is a reconstruction of the procedure that trained the shipped
model.json, written from the surviving deployment artifacts (model.json
metadata, dsp.js pipeline, and project notes). The original training run's
checkpoint and logs did not survive; this exact script has NOT been re-executed
to convergence. The reproducible artifact of record in this repository is:
shipped weights (model.json) + re-measured evaluation (eval_remeasure.py)
+ JS/torch parity test (test_full_parity.js).

Pipeline
--------
Data:      ESC-50 (https://github.com/karoldvl/ESC-50, CC BY-NC 3.0)
Classes:   crying_baby, clock_alarm, door_wood_knock, glass_breaking, siren,
           car_horn, dog, footsteps, thunderstorm, pouring_water (10)
Split:     official folds 1-3 train (240 clips), 4 validation (80),
           5 test (80, never trained or tuned on)
Seed:      7
Audio:     16 kHz mono (linear resample), 4 s window (64000 samples)
Features:  1024 FFT, hop 256, Hann, 64 mel bins (50-8000 Hz),
           log-mel floored at max(acc, 1e-6), per-clip standardization
Augment:   TRAIN ONLY - speed 0.9-1.1x, gain, 10-30 dB SNR white noise,
           time shift, random 4 s crop, mixup, SpecAugment-style masking
Model:     4x [conv3x3 + BatchNorm + ReLU + maxpool2] (1-16-32-64-96)
           + global average pool + linear(96->10); 79,866 parameters
Optim:     Adam lr 1e-3, batch 32, 60 epochs, best checkpoint by val accuracy
Export:    weights + mel filterbank as base64 float32 in model.json
"""
import json, base64, csv, os, sys, random
import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile

ESC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ESC-50-master"
OUT = "model.json"
SEED = 7

SR, WIN, NFFT, HOP, NMEL = 16000, 64000, 1024, 256, 64
FRAMES = 1 + (WIN - NFFT) // HOP
CLASSES = ["crying_baby","clock_alarm","door_wood_knock","glass_breaking","siren",
           "car_horn","dog","footsteps","thunderstorm","pouring_water"]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def mel_filterbank(sr, nfft, nmel, fmin, fmax):
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), nmel + 2)
    freqs = mel2hz(mels)
    bins = np.floor((nfft + 1) * freqs / sr).astype(int)
    fb = np.zeros((nmel, nfft // 2 + 1))
    for m in range(nmel):
        for k in range(bins[m], bins[m + 1]):
            fb[m, k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m, k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb.astype(np.float32)

FB = mel_filterbank(SR, NFFT, NMEL, 50.0, 8000.0)
HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / (NFFT - 1))

def resample_to_16k(x, src):
    if src == SR: return x
    n = int(len(x) * SR / src)
    idx = np.arange(n) * (src / SR)
    i0 = np.minimum(idx.astype(int), len(x) - 1); i1 = np.minimum(i0 + 1, len(x) - 1)
    f = idx - np.floor(idx)
    return (x[i0] * (1 - f) + x[i1] * f).astype(np.float32)

def load_wav(path):
    src, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1: x = x.mean(axis=1)
    return resample_to_16k(x / 32768.0, src)

def features(x4s):
    out = np.empty((NMEL, FRAMES), dtype=np.float64)
    for f in range(FRAMES):
        seg = x4s[f * HOP: f * HOP + NFFT] * HANN
        s = np.fft.rfft(seg)
        acc = FB @ (s.real**2 + s.imag**2)
        out[:, f] = np.log(np.maximum(acc, 1e-6))
    return ((out - out.mean()) / (out.std() + 1e-6)).astype(np.float32)

def augment(x):
    if random.random() < 0.8:  # speed
        r = random.uniform(0.9, 1.1)
        idx = np.arange(len(x)) * r
        i0 = np.minimum(idx.astype(int), len(x) - 1); i1 = np.minimum(i0 + 1, len(x) - 1)
        x = x[i0] * (1 - (idx - i0)) + x[i1] * (idx - i0)
    if random.random() < 0.5:  # gain
        x = x * random.uniform(0.5, 1.5)
    if random.random() < 0.7:  # white noise 10-30 dB SNR
        snr = random.uniform(10, 30)
        p = np.mean(x**2) + 1e-9
        x = x + np.random.randn(len(x)).astype(np.float32) * np.sqrt(p / 10**(snr / 10))
    if random.random() < 0.5:  # time shift
        x = np.roll(x, random.randint(-SR, SR))
    return x.astype(np.float32)

def crop4s(x, random_crop):
    if len(x) <= WIN:
        return np.pad(x, (0, WIN - len(x)))
    off = random.randint(0, len(x) - WIN) if random_crop else 0
    return x[off:off + WIN]

rows = list(csv.DictReader(open(f"{ESC}/meta/esc50.csv")))
def subset(folds): return [r for r in rows if r["fold"] in folds and r["category"] in CLASSES]
train, val, test = subset(("1","2","3")), subset(("4",)), subset(("5",))

def featurize(split, aug):
    X, y = [], []
    for r in split:
        x = load_wav(f"{ESC}/audio/{r['filename']}")
        if aug: x = augment(x)
        X.append(features(crop4s(x, aug)))
        y.append(CLASSES.index(r["category"]))
    return np.array(X), np.array(y)

print("featurizing...")
Xv, yv = featurize(val, False)
Xt, yt = featurize(test, False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        ci = 1
        for co in (16, 32, 64, 96):
            layers += [nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(), nn.MaxPool2d(2)]
            ci = co
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(96, 10)
    def forward(self, x):
        e = self.body(x).mean(dim=(2, 3))
        return self.fc(e)

net = Net()
print("params:", sum(p.numel() for p in net.parameters()))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
best = 0.0
Xv_t = torch.tensor(Xv).unsqueeze(1)
for ep in range(60):
    net.train()
    # re-augment every epoch for variety (mixup + spec masking applied below)
    Xtr, ytr = featurize(train, True)
    Xtr_t, ytr_t = torch.tensor(Xtr).unsqueeze(1), torch.tensor(ytr)
    perm = torch.randperm(len(Xtr_t))
    for i in range(0, len(perm), 32):
        idx = perm[i:i+32]
        xb, yb = Xtr_t[idx], ytr_t[idx]
        # mixup
        lam = np.random.beta(0.4, 0.4)
        ridx = torch.randperm(xb.size(0))
        xb = lam * xb + (1 - lam) * xb[ridx]
        # SpecAugment-style masking
        for x_ in xb:
            if random.random() < 0.5:
                f0 = random.randint(0, NMEL - 8); x_[0, f0:f0+8, :] = 0
            if random.random() < 0.5:
                t0 = random.randint(0, FRAMES - 24); x_[0, :, t0:t0+24] = 0
        opt.zero_grad()
        out = net(xb)
        loss = lam * lossf(out, yb) + (1 - lam) * lossf(out, yb[ridx])
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        va = (net(Xv_t).argmax(1) == torch.tensor(yv)).float().mean().item()
    if va > best:
        best = va
        torch.save(net.state_dict(), "ckpt.pt")
    print(f"epoch {ep+1}: val {va:.3f} (best {best:.3f})")

net.load_state_dict(torch.load("ckpt.pt")); net.eval()
with torch.no_grad():
    ta = (net(torch.tensor(Xt).unsqueeze(1)).argmax(1) == torch.tensor(yt)).float().mean().item()
print(f"final: val {best:.4f} test {ta:.4f}")

# export
def b64(a): return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()
sd = net.state_dict()
weights = {}
names = ["0.weight","1.weight","1.bias","1.running_mean","1.running_var"]
blocks = {0: "b1", 2: "b2", 4: "b3", 6: "b4"}  # sequential indices of conv layers
for seq_i, pfx in blocks.items():
    weights[f"{pfx}.0.weight"] = {"shape": list(sd[f"body.{seq_i}.weight"].shape), "b64": b64(sd[f"body.{seq_i}.weight"].numpy())}
    for suf, k in [("weight","weight"),("bias","bias"),("running_mean","running_mean"),("running_var","running_var")]:
        weights[f"{pfx}.1.{suf}"] = {"shape": list(sd[f"body.{seq_i+1}.{k}"].shape), "b64": b64(sd[f"body.{seq_i+1}.{k}"].numpy())}
weights["fc.weight"] = {"shape": [10, 96], "b64": b64(sd["fc.weight"].numpy())}
weights["fc.bias"] = {"shape": [10], "b64": b64(sd["fc.bias"].numpy())}
meta = {"classes": CLASSES, "sr": SR, "win": WIN, "nfft": NFFT, "hop": HOP,
        "nmel": NMEL, "fmin": 50.0, "fmax": 8000.0,
        "fb_shape": list(FB.shape), "fb_b64": b64(FB)}
json.dump({"weights": weights, "meta": meta}, open(OUT, "w"))
print("wrote", OUT)
