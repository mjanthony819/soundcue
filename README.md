# SoundCue

**See the sounds around you.** A visual sound-awareness companion: the microphone listens continuously, a
small CNN classifies 4-second windows every 2 seconds — entirely in the browser — and turns everyday alert
sounds (door knock, baby crying, alarm clock, siren, car horn, glass breaking, dog barking, thunder,
footsteps, running water) into visual alerts with urgency tiers, confidence, an event timeline and an
exportable log. Built as an accessibility aid for Deaf and hard-of-hearing people.

> **Not a safety device.** SoundCue is an awareness aid and a research prototype, evaluated on 80 clean
> test clips of 10 sound classes — not in real homes. Do not rely on it for emergencies, smoke/CO alarms
> or medical monitoring.

**Live app:** `index.html` via GitHub Pages (link on the repo page). Offline after load — no server, no
account, no telemetry, no model API.

Built for **IdeaLab Hacks** (Devpost), solo. Built with AI assistance (disclosed on the submission).

## What's inside

- **Listen** — live mic → top-3 class probabilities, abstain threshold, editable alert tiers, deduplicated
  event timeline (same sound within 10 s merges), CSV export, screen flash for high-tier alerts.
- **Try a clip** — ten real held-out test clips (fold 5, never trained or tuned on), plus your own WAV
  upload, plus an interactive white-noise slider that shows confidence degrade until the model abstains.
- **Your sounds** — teach it a custom sound (your own buzzer/kettle) with three 4-second samples: the
  frozen CNN's 96-d embeddings + nearest neighbour, with a distance gate derived from the samples' own
  spread. Verified: enrolled on 3 door-knock clips, an unseen knock matched at cosine distance 0.07
  (gate 0.20) while 6 unseen siren clips sat at 0.39–0.48 and were all ignored.
- **Evidence** — measured results, per-class precision/recall, confusion matrix, honest limits.

## Data & model

- **Data:** [ESC-50](https://github.com/karoldvl/ESC-50) — 10 alert-relevant classes, 400 clips (40/class).
  Official folds 1–3 train (240), fold 4 validation (80), fold 5 test (80, untouched). No clip overlaps folds.
- **Model:** 79,866-parameter CNN (4 conv+BN+ReLU+maxpool blocks, GAP, linear), trained for this project in
  PyTorch (CPU), seed 7, best checkpoint by validation accuracy. Train-time augmentation: speed 0.9–1.1×,
  gain, 10–30 dB white noise, time shift, random 4 s crop, mixup, label smoothing.
- **Results (this training run):** CNN — validation 91.2% (80 clips), **test 92.5% (80 clips)**.
  Baseline (mel mean/std + logistic regression) — validation 77.5%, test 65.0%.
  Test confusions: siren→car horn ×2; single mix-ups crying baby→dog, dog→glass, footsteps→knock,
  footsteps→thunder (6 errors total).
- **In-browser inference:** pure-JS re-implementation (FFT, mel filterbank, conv/BN/ReLU/pool, FC),
  parity-tested against the PyTorch logits on held-out clips — max absolute logit difference 0.013,
  ~250 ms per 4 s window on a laptop.

## Licenses

- **App code:** MIT.
- **ESC-50 dataset:** CC BY-NC 3.0 (© the ESC-50 authors). The bundled demo clips in `app/clips/` keep
  their original per-clip licenses (7 × CC0, 3 × CC-BY) — full credits in `clip_attribution.json`,
  derived from the dataset's own LICENSE file.
- **Trained model weights (`app/model.json`):** shared under CC BY-NC 3.0 as a derivative of ESC-50.
- Non-commercial use only, consistent with the dataset license.

## Repo layout

- `app/` — the whole product: `index.html`, `style.css`, `app.js`, `dsp.js` (DSP + inference),
  `model.json` (weights + mel filterbank), `results.json` (measured metrics), `clips/` (10 held-out
  demo clips + manifest), `clip_attribution.json`.
- `prep_train.py` — full training pipeline (cache → baseline → train → export).
- `test_parity.js`, `test_knn.js` — parity and custom-sound verification.
