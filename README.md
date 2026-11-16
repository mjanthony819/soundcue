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
- **Try a clip** — ten real clips from the official fold-5 test split (held out from training; development-exposed — see Evaluation), plus your own WAV
  upload, plus an interactive white-noise slider that shows confidence degrade until the model abstains.
- **Evidence** — measured results, per-class precision/recall, confusion matrix, honest limits.

## Data & model

- **Data:** [ESC-50](https://github.com/karoldvl/ESC-50) — 10 alert-relevant classes, 400 clips (40/class).
  Official folds 1–3 train (240), fold 4 validation (80), fold 5 test (80). Fold 5 was held out from training, but its clips were inspected during development
  (interactive noise tests, pipeline fixes, re-evaluation) — read it as a development-exposed benchmark,
  not pristine independent validation. No threshold or feature was retuned on fold 5 to raise the score.
  No clip overlaps folds.
- **Model:** 79,866-parameter CNN (4 conv+BN+ReLU+maxpool blocks, GAP, linear), trained for this project in
  PyTorch (CPU), seed 7, best checkpoint by validation accuracy. Train-time augmentation: speed 0.9–1.1×,
  gain, 10–30 dB white noise, time shift, random 4 s crop, mixup, SpecAugment-style masking.
- **Results, re-measured from the shipped weights** on the fold-5 test split with the final
  feature pipeline: CNN — validation 92.5% (80 clips), **test 93.75% (80 clips)**.
  Baseline (mel mean/std + softmax regression, same features) — validation 67.5%, test 56.2%.
  Test confusions (5 errors): siren→car horn ×2, dog→glass breaking, footsteps→door knock,
  footsteps→thunder.
- **In-browser inference:** pure-JS re-implementation (FFT, mel filterbank, conv/BN/ReLU/pool, FC).
  Full parity test against the PyTorch reference on all 80 test clips: largest probability difference
  3.8×10⁻⁷, zero disagreements in predicted class, zero disagreements in the abstain decision at the
  default 0.55 threshold (`test_full_parity.js`).

## Model provenance (honest note)

The shipped `model.json` weights are the surviving artifact of the original training run; that run's
checkpoint and logs were lost. All published metrics were therefore **re-measured from the shipped
weights** on the official held-out split (`eval_remeasure.py`), not copied from training logs.
`prep_train.py` is a reconstructed training recipe — the documented procedure for this model class —
not a script proven to reproduce these exact weights.

## A feature we removed, on evidence

We built and tested few-shot enrollment ("teach it your own doorbell": 3 samples → nearest-neighbour on
the CNN's 96-d embeddings, distance gate from the samples' own spread). Re-measured evaluation
(`eval_knn.py`, results in `knn_eval.json`): enrolling on 3 door-knock clips, the automatic gate
false-fires on 36 of 72 non-knock held-out test clips, and even a gate tightened to half that distance
still false-fires on 15 of 72. An awareness tool that cries wolf is worse than none, so the feature was
removed.

## Licenses

- **App code:** MIT.
- **ESC-50 dataset:** CC BY-NC 3.0 (© the ESC-50 authors). The bundled demo clips keep their original
  per-clip licenses (7 × CC0, 3 × CC-BY) — full credits in `clip_attribution.json`, derived from the
  dataset's own LICENSE file.
- **Trained model weights (`model.json`):** shared under CC BY-NC 3.0 as a derivative of ESC-50.
- Non-commercial use only, consistent with the dataset license.

## Repo layout (flat, GitHub Pages serves the root)

- `index.html`, `style.css`, `app.js`, `dsp.js` — the whole product (DSP + inference in `dsp.js`).
- `model.json` — weights + mel filterbank (base64 float32). `results.json` — measured metrics.
- `clip-*.ogg`, `manifest.json`, `clip_attribution.json` — 10 held-out demo clips with licenses.
- `prep_train.py` — reconstructed training recipe (see provenance note).
- `eval_remeasure.py` — re-measures metrics from the shipped weights on the official split.
- `test_full_parity.js` — 80-clip JS/PyTorch parity test.
- `eval_knn.py` + `knn_eval.json` — evaluation of the removed few-shot feature (see above).
- `REPRODUCE.md` — step-by-step instructions.
