# Reproducing SoundCue

## What is reproducible, and how

| Artifact | Status | How to verify |
| --- | --- | --- |
| Shipped weights (`model.json`) | Surviving artifact of the original training run | Byte file in this repo; the live app loads it |
| Published metrics (val 92.5%, test 93.75%, baseline 67.5%/56.2%) | Re-measured from the shipped weights on the official ESC-50 split | `python3 eval_remeasure.py` |
| Browser/torch equivalence | Verified on all 80 test clips (max prob diff 3.8e-7, 0 class or abstain disagreements) | `node test_full_parity.js` (needs `torch_probs.json` from the step above) |
| Few-shot feature removal rationale | Re-measured (`knn_eval.json`) | `python3 eval_knn.py` |
| Training recipe (`prep_train.py`) | **Reconstructed recipe, not re-executed to convergence.** Documents the exact pipeline (data, split, features, augmentation, model, optimizer, export). The original run's checkpoint and logs did not survive; do not cite it as proven to reproduce these exact weights. | Read; run at your own compute budget (~15 min CPU) |

## Prerequisites

- Python 3.10+ with `torch` (CPU is fine), `numpy`, `scipy`
- Node.js 18+ (for the parity test)
- ESC-50: `curl -sL -o esc50.zip https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip && unzip esc50.zip`
  (CC BY-NC 3.0 — non-commercial use only)

Paths default to `/tmp/ESC-50-master`; pass another path as `sys.argv[1]` to the Python scripts.

## Steps

```bash
python3 eval_remeasure.py     # re-measures metrics from shipped weights; writes results.json + torch_probs.json
node test_full_parity.js      # JS inference vs torch on all 80 test clips; writes parity.json
python3 eval_knn.py           # evidence for the removed few-shot feature; writes knn_eval.json
```

Expected: test accuracy 0.9375, parity maxProbDiff < 1e-6 with 0 disagreements.

## Exact configuration

- Classes (10): crying_baby, clock_alarm, door_wood_knock, glass_breaking, siren, car_horn, dog,
  footsteps, thunderstorm, pouring_water
- Split: official ESC-50 folds 1-3 train (240 clips), fold 4 validation (80), fold 5 test (80).
  No clip appears in two folds.
- Seed: 7
- Audio: 16 kHz mono (linear resample), first 4 s of each 5 s clip (64000 samples), zero-padded if short
- Features: 1024-pt FFT, hop 256, Hann window, 64 mel bins (50-8000 Hz),
  log-mel **floored at max(acc, 1e-6)**, per-clip standardization (biased std, +1e-6)
- Model: 4x [conv3x3 (no bias) + BatchNorm + ReLU + maxpool2] with channels 16-32-64-96,
  global average pool, linear 96->10. 79,866 parameters.
- Augmentation (train only): speed 0.9-1.1x, gain 0.5-1.5x, white noise 10-30 dB SNR,
  time shift ±1 s, random 4 s crop, mixup (beta 0.4), SpecAugment-style freq/time masking
- Optimizer: Adam lr 1e-3, batch 32, 60 epochs, best checkpoint by validation accuracy
