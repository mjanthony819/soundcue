#!/usr/bin/env python3
"""Evidence for the REMOVED few-shot enrollment feature.

Re-measures why 'teach it your own doorbell' was dropped:
enroll 3 door-knock clips (train folds, seed 7), nearest-neighbour on the
CNN's 96-d embeddings, distance gate derived from the samples' own spread.
Report false matches on the 72 non-knock fold-5 test clips and missed true
knocks among the 16 knock clips in val+test, under the auto gate and a
tightened gate.
"""
import json, base64, csv, random
import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile

ROOT = "/downloads/soundcue"
ESC = "/tmp/ESC-50-master"
SR, WIN, NFFT, HOP, NMEL = 16000, 64000, 1024, 256, 64
FRAMES = 1 + (WIN - NFFT) // HOP

m = json.load(open(f"{ROOT}/model.json"))
CLASSES = m["meta"]["classes"]
b64f32 = lambda s: np.frombuffer(base64.b64decode(s), dtype=np.float32)
W = {k: b64f32(v["b64"]) for k, v in m["weights"].items()}
FB = b64f32(m["meta"]["fb_b64"]).reshape(NMEL, NFFT // 2 + 1)
HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / (NFFT - 1))

def load_first4s(path):
    src, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1: x = x.mean(axis=1)
    x /= 32768.0
    if src != SR:
        n = int(len(x) * SR / src); idx = np.arange(n) * (src / SR)
        i0 = np.minimum(idx.astype(int), len(x)-1); i1 = np.minimum(i0+1, len(x)-1)
        f = idx - np.floor(idx); x = x[i0]*(1-f) + x[i1]*f
    x = x[:WIN]
    return np.pad(x, (0, max(0, WIN - len(x))))[:WIN].astype(np.float32)

def features(x):
    out = np.empty((NMEL, FRAMES))
    for f in range(FRAMES):
        s = np.fft.rfft(x[f*HOP:f*HOP+NFFT] * HANN)
        out[:, f] = np.log(np.maximum(FB @ (s.real**2 + s.imag**2), 1e-6))
    return ((out - out.mean()) / (out.std() + 1e-6)).astype(np.float32)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        for ci, co in [(1,16),(16,32),(32,64),(64,96)]:
            self.blocks.append(nn.ModuleList([nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co)]))
        self.fc = nn.Linear(96, 10)
    def forward(self, x):
        for conv, bn in self.blocks:
            x = torch.max_pool2d(torch.relu(bn(conv(x))), 2)
        emb = x.mean(dim=(2, 3))
        return self.fc(emb), emb

net = Net().eval()
sd = net.state_dict()
for i, (ci, co) in enumerate([(1,16),(16,32),(32,64),(64,96)], start=1):
    sd[f"blocks.{i-1}.0.weight"] = torch.from_numpy(W[f"b{i}.0.weight"].reshape(co, ci, 3, 3).copy())
    for suf in ["weight","bias","running_mean","running_var"]:
        sd[f"blocks.{i-1}.1.{suf}"] = torch.from_numpy(W[f"b{i}.1.{suf}"].copy())
sd["fc.weight"] = torch.from_numpy(W["fc.weight"].reshape(10, 96).copy())
sd["fc.bias"] = torch.from_numpy(W["fc.bias"].copy())
net.load_state_dict(sd)

rows = list(csv.DictReader(open(f"{ESC}/meta/esc50.csv")))
def subset(folds): return [r for r in rows if r["fold"] in folds and r["category"] in CLASSES]

def emb_of(fn):
    with torch.no_grad():
        _, e = net(torch.from_numpy(features(load_first4s(f"{ESC}/audio/{fn}"))).view(1,1,NMEL,FRAMES))
    return e[0].numpy()

random.seed(7)
knocks_train = [r["filename"] for r in subset(("1","2","3")) if r["category"] == "door_wood_knock"]
enroll_fns = random.sample(knocks_train, 3)
embs = [emb_of(f) for f in enroll_fns]

def cos(a, b): return 1 - float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

pd = [cos(embs[0], embs[1]), cos(embs[0], embs[2]), cos(embs[1], embs[2])]
gate_auto = min(0.5, max(pd) * 1.5)
gate_tight = gate_auto * 0.5

def dist_to_enrolled(e): return min(cos(e, g) for g in embs)

knocks_held = [r["filename"] for r in subset(("4","5")) if r["category"] == "door_wood_knock"]
others_test = [r for r in subset(("5",)) if r["category"] != "door_wood_knock"]

res = {"enroll_files": enroll_fns, "enroll_pairwise_dists": [round(float(v),4) for v in pd],
       "gate_auto": round(float(gate_auto),4), "gate_tight": round(float(gate_tight),4)}
for name, gate in [("auto", gate_auto), ("tight", gate_tight)]:
    false_hits = []
    for r in others_test:
        d = dist_to_enrolled(emb_of(r["filename"]))
        if d < gate: false_hits.append({"file": r["filename"], "true": r["category"], "d": round(float(d),4)})
    misses = []
    for f in knocks_held:
        d = dist_to_enrolled(emb_of(f))
        if d >= gate: misses.append({"file": f, "d": round(float(d),4)})
    res[name] = {"false_matches": len(false_hits), "false_match_detail": false_hits[:20],
                 "true_knocks": len(knocks_held), "missed_knocks": len(misses), "miss_detail": misses}

json.dump(res, open(f"{ROOT}/knn_eval.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("gate_auto","gate_tight")}, indent=1))
for name in ("auto","tight"):
    r = res[name]
    print(f"{name}: false matches {r['false_matches']}/72 non-knock test clips, missed {r['missed_knocks']}/{r['true_knocks']} held-out knocks")
