#!/usr/bin/env python3
"""Re-measure SoundCue's surviving deployed weights on ESC-50 fold 5.

The original training run's checkpoint/logs did not survive a workspace reset.
This script reconstructs a PyTorch reference from the SHIPPED model.json
(the artifact the app actually serves), applies the documented feature
pipeline (log-mel floored at 1e-6), and re-measures test-fold metrics.

It also dumps per-clip probabilities for the JS parity harness.
"""
import json, base64, csv, sys
import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile

ROOT = "/downloads/soundcue"
ESC = "/tmp/ESC-50-master"

SR, WIN, NFFT, HOP, NMEL = 16000, 64000, 1024, 256, 64
FRAMES = 1 + (WIN - NFFT) // HOP  # 247

m = json.load(open(f"{ROOT}/model.json"))
CLASSES = m["meta"]["classes"]

def b64f32(s):
    return np.frombuffer(base64.b64decode(s), dtype=np.float32)

W = {k: b64f32(v["b64"]) for k, v in m["weights"].items()}
FB = b64f32(m["meta"]["fb_b64"]).reshape(NMEL, NFFT // 2 + 1)

HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / (NFFT - 1))

def resample_to_16k(x, src):
    if src == SR:
        return x
    n = int(len(x) * SR / src)
    idx = np.arange(n) * (src / SR)
    i0 = np.minimum(idx.astype(int), len(x) - 1)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    f = idx - np.floor(idx)
    return (x[i0] * (1 - f) + x[i1] * f).astype(np.float32)

def load_first4s(path):
    src, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x /= 32768.0
    x = resample_to_16k(x, src)
    x = x[:WIN]
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    return x

def features(x):
    """log-mel, floored at 1e-6, per-clip standardization - mirrors dsp.js."""
    out = np.empty((NMEL, FRAMES), dtype=np.float64)
    nb = NFFT // 2 + 1
    for f in range(FRAMES):
        seg = x[f * HOP: f * HOP + NFFT] * HANN
        spec = np.fft.rfft(seg)
        pow_ = spec.real**2 + spec.imag**2
        acc = FB @ pow_
        out[:, f] = np.log(np.maximum(acc, 1e-6))
    mean = out.mean()
    std = np.sqrt(((out - mean) ** 2).mean())
    return ((out - mean) / (std + 1e-6)).astype(np.float32)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        chans = [(1, 16), (16, 32), (32, 64), (64, 96)]
        self.blocks = nn.ModuleList()
        for ci, co in chans:
            self.blocks.append(nn.ModuleList([
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
            ]))
        self.fc = nn.Linear(96, 10)

    def forward(self, x):
        for conv, bn in self.blocks:
            x = torch.relu(bn(conv(x)))
            x = torch.max_pool2d(x, 2)
        emb = x.mean(dim=(2, 3))
        return self.fc(emb), emb

net = Net().eval()
sd = net.state_dict()
for i, (ci, co) in enumerate([(1,16),(16,32),(32,64),(64,96)], start=1):
    sd[f"blocks.{i-1}.0.weight"] = torch.from_numpy(W[f"b{i}.0.weight"].reshape(co, ci, 3, 3).copy())
    for suf in ["weight", "bias", "running_mean", "running_var"]:
        sd[f"blocks.{i-1}.1.{suf}"] = torch.from_numpy(W[f"b{i}.1.{suf}"].copy())
sd["fc.weight"] = torch.from_numpy(W["fc.weight"].reshape(10, 96).copy())
sd["fc.bias"] = torch.from_numpy(W["fc.bias"].copy())
net.load_state_dict(sd)

# fold 5 test set for the 10 classes
rows = list(csv.DictReader(open(f"{ESC}/meta/esc50.csv")))
test = [r for r in rows if r["fold"] == "5" and r["category"] in CLASSES]
val  = [r for r in rows if r["fold"] == "4" and r["category"] in CLASSES]
train = [r for r in rows if r["fold"] in ("1", "2", "3") and r["category"] in CLASSES]
print(f"train {len(train)} val {len(val)} test {len(test)}")

def run_split(split, dump_probs=None):
    cm = np.zeros((10, 10), dtype=int)
    probs_out = []
    with torch.no_grad():
        for r in split:
            x = load_first4s(f"{ESC}/audio/{r['filename']}")
            feats = torch.from_numpy(features(x)).view(1, 1, NMEL, FRAMES)
            logits, _ = net(feats)
            p = torch.softmax(logits, 1)[0].numpy()
            t = CLASSES.index(r["category"])
            pred = int(p.argmax())
            cm[t, pred] += 1
            if dump_probs is not None:
                dump_probs.append({"file": r["filename"], "true": r["category"],
                                   "probs": [round(float(v), 8) for v in p]})
    acc = np.trace(cm) / cm.sum()
    return acc, cm

acc_v, cm_v = run_split(val)
acc_t, cm_t = run_split(test, dump_probs := [])
json.dump(dump_probs, open(f"{ROOT}/torch_probs.json", "w"))

# baseline: mel mean/std + softmax regression (torch), seed 7, re-measured on floored features
def flat_feats(split):
    X, y = [], []
    for r in split:
        f = features(load_first4s(f"{ESC}/audio/{r['filename']}"))
        X.append(np.concatenate([f.mean(1), f.std(1)]))
        y.append(CLASSES.index(r["category"]))
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(y)

torch.manual_seed(7)
Xtr, ytr = flat_feats(train)
Xv, yv = flat_feats(val)
Xt, yt = flat_feats(test)
mu, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-6
lin = nn.Linear(128, 10)
opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
lossf = nn.CrossEntropyLoss()
for ep in range(2000):
    opt.zero_grad()
    loss = lossf(lin((Xtr - mu) / sd_), ytr)
    loss.backward(); opt.step()
with torch.no_grad():
    bv = (lin((Xv - mu) / sd_).argmax(1) == yv).float().mean().item()
    bt = (lin((Xt - mu) / sd_).argmax(1) == yt).float().mean().item()

nparams = sum(p.numel() for p in net.parameters())
res = {
    "note": "Re-measured 2026-11-16 from the surviving deployed weights (model.json) with floored log-mel features (max(acc,1e-6)); original training logs/checkpoint did not survive a workspace reset. Split: official ESC-50 folds 1-3 train / 4 val / 5 test, 10 classes, 8 clips per class on val/test.",
    "classes": CLASSES,
    "baseline": {"model": "mel mean/std + softmax regression (seed 7, re-measured)", "val": round(bv, 4), "test": round(bt, 4)},
    "cnn": {"params": nparams, "best_val": round(acc_v, 4), "test": round(acc_t, 4)},
    "confusion": cm_t.tolist(),
}
json.dump(res, open(f"{ROOT}/results.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("baseline", "cnn")}, indent=1))
print("params", nparams)
print("confusion rows (true):", cm_t.tolist())
