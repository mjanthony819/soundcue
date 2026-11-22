/* SoundCue DSP + model inference. Pure JS, no dependencies.
   Mirrors prep_train.py exactly: 16 kHz, 4 s window, 1024 FFT, hop 256,
   64 mel bins (50-8000 Hz), log-mel, per-clip standardization.
   CNN: 4x [conv3x3 + BN + ReLU + maxpool2] (1-16-32-64-96) + GAP + FC. */
"use strict";

const DSP = (() => {
  const SR = 16000, WIN = 64000, NFFT = 1024, HOP = 256, NMEL = 64;
  const FRAMES = 1 + Math.floor((WIN - NFFT) / HOP); // 247

  const HANN = new Float32Array(NFFT);
  for (let i = 0; i < NFFT; i++) HANN[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (NFFT - 1));

  function fftRadix2(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
      for (let i = 0; i < n; i += len) {
        let cwr = 1, cwi = 0;
        for (let j = 0; j < half; j++) {
          const a = i + j, b = i + j + half;
          const vr = re[b] * cwr - im[b] * cwi, vi = re[b] * cwi + im[b] * cwr;
          re[b] = re[a] - vr; im[b] = im[a] - vi;
          re[a] += vr; im[a] += vi;
          const t = cwr * wr - cwi * wi; cwi = cwr * wi + cwi * wr; cwr = t;
        }
      }
    }
  }

  let FB = null; // [NMEL][513] mel filterbank from model.json
  function setFilterbank(fbFlat) {
    FB = [];
    const nb = NFFT / 2 + 1;
    for (let m = 0; m < NMEL; m++) FB.push(fbFlat.subarray(m * nb, (m + 1) * nb));
  }

  // x: Float32Array(WIN) @16k -> Float32Array(NMEL*FRAMES), standardized, row-major [mel][frame]
  function computeFeatures(x) {
    const out = new Float32Array(NMEL * FRAMES);
    const re = new Float32Array(NFFT), im = new Float32Array(NFFT);
    const nb = NFFT / 2 + 1;
    const pow = new Float32Array(nb);
    for (let f = 0; f < FRAMES; f++) {
      const off = f * HOP;
      for (let i = 0; i < NFFT; i++) { re[i] = x[off + i] * HANN[i]; im[i] = 0; }
      fftRadix2(re, im);
      for (let k = 0; k < nb; k++) pow[k] = re[k] * re[k] + im[k] * im[k];
      for (let m = 0; m < NMEL; m++) {
        const row = FB[m]; let acc = 0;
        for (let k = 0; k < nb; k++) acc += pow[k] * row[k];
        out[m * FRAMES + f] = Math.log(Math.max(acc, 1e-6));
      }
    }
    let mean = 0;
    for (let i = 0; i < out.length; i++) mean += out[i];
    mean /= out.length;
    let v = 0;
    for (let i = 0; i < out.length; i++) { const d = out[i] - mean; v += d * d; }
    const inv = 1 / (Math.sqrt(v / out.length) + 1e-6);
    for (let i = 0; i < out.length; i++) out[i] = (out[i] - mean) * inv;
    return out;
  }

  // resample arbitrary-rate Float32Array to 16 kHz by linear interpolation
  function resampleTo16k(x, srcRate) {
    if (srcRate === SR) return x;
    const n = Math.floor(x.length * SR / srcRate);
    const out = new Float32Array(n);
    const ratio = srcRate / SR;
    for (let i = 0; i < n; i++) {
      const p = i * ratio, i0 = Math.floor(p), f = p - i0;
      const a = x[Math.min(i0, x.length - 1)], b = x[Math.min(i0 + 1, x.length - 1)];
      out[i] = a + (b - a) * f;
    }
    return out;
  }

  // ---- CNN ----
  let W = null; // decoded weights
  function b64ToF32(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }
  async function loadModel(url) {
    const j = await (await fetch(url)).json();
    W = {};
    for (const k in j.weights) W[k] = b64ToF32(j.weights[k].b64);
    setFilterbank(b64ToF32(j.meta.fb_b64));
    return j.meta;
  }

  // fold BN into per-channel scale/shift
  function bnParams(pfx, C) {
    const g = W[pfx + '.1.weight'], b = W[pfx + '.1.bias'],
          m = W[pfx + '.1.running_mean'], v = W[pfx + '.1.running_var'];
    const scale = new Float32Array(C), shift = new Float32Array(C);
    for (let c = 0; c < C; c++) {
      scale[c] = g[c] / Math.sqrt(v[c] + 1e-5);
      shift[c] = b[c] - m[c] * scale[c];
    }
    return { scale, shift };
  }

  function convBnReluPool(inp, C, H, Wd, pfx, CO) {
    const wt = W[pfx + '.0.weight']; // [CO][C][3][3]
    const { scale, shift } = bnParams(pfx, CO);
    const full = new Float32Array(CO * H * Wd);
    for (let co = 0; co < CO; co++) {
      const wco = wt.subarray(co * C * 9, (co + 1) * C * 9);
      const sc = scale[co], sh = shift[co];
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < Wd; x++) {
          let s = 0;
          for (let c = 0; c < C; c++) {
            const wc = wco.subarray(c * 9, c * 9 + 9);
            const base = c * H * Wd;
            for (let ky = 0; ky < 3; ky++) {
              const yy = y + ky - 1;
              if (yy < 0 || yy >= H) continue;
              const rowBase = base + yy * Wd;
              for (let kx = 0; kx < 3; kx++) {
                const xx = x + kx - 1;
                if (xx < 0 || xx >= Wd) continue;
                s += inp[rowBase + xx] * wc[ky * 3 + kx];
              }
            }
          }
          s = s * sc + sh;
          full[co * H * Wd + y * Wd + x] = s > 0 ? s : 0;
        }
      }
    }
    const OH = H >> 1, OW = Wd >> 1;
    const out = new Float32Array(CO * OH * OW);
    for (let co = 0; co < CO; co++) {
      const base = co * H * Wd;
      for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) {
        const y0 = oy * 2, x0 = ox * 2;
        let best = full[base + y0 * Wd + x0];
        if (x0 + 1 < Wd && full[base + y0 * Wd + x0 + 1] > best) best = full[base + y0 * Wd + x0 + 1];
        if (y0 + 1 < H) {
          if (full[base + (y0 + 1) * Wd + x0] > best) best = full[base + (y0 + 1) * Wd + x0];
          if (x0 + 1 < Wd && full[base + (y0 + 1) * Wd + x0 + 1] > best) best = full[base + (y0 + 1) * Wd + x0 + 1];
        }
        out[co * OH * OW + oy * OW + ox] = best;
      }
    }
    return { data: out, C: CO, H: OH, W: OW };
  }

  // features -> {logits: number[10], embedding: number[96]}
  function forward(feats) {
    let t = { data: feats, C: 1, H: NMEL, W: FRAMES };
    t = convBnReluPool(t.data, t.C, t.H, t.W, 'b1', 16);
    t = convBnReluPool(t.data, t.C, t.H, t.W, 'b2', 32);
    t = convBnReluPool(t.data, t.C, t.H, t.W, 'b3', 64);
    t = convBnReluPool(t.data, t.C, t.H, t.W, 'b4', 96);
    const emb = new Float32Array(96);
    const hw = t.H * t.W;
    for (let c = 0; c < 96; c++) {
      let s = 0;
      for (let i = 0; i < hw; i++) s += t.data[c * hw + i];
      emb[c] = s / hw;
    }
    const fcw = W['fc.weight'], fcb = W['fc.bias'];
    const logits = new Float32Array(10);
    for (let o = 0; o < 10; o++) {
      let s = fcb[o];
      for (let i = 0; i < 96; i++) s += fcw[o * 96 + i] * emb[i];
      logits[o] = s;
    }
    return { logits: Array.from(logits), embedding: Array.from(emb) };
  }

  function softmax(logits) {
    const mx = Math.max(...logits);
    const e = logits.map(v => Math.exp(v - mx));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / sum);
  }

  function cosineDist(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return 1 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
  }

  return { SR, WIN, NFFT, HOP, NMEL, FRAMES, loadModel, computeFeatures, resampleTo16k, forward, softmax, cosineDist };
})();
