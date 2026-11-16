/* Quick smoke parity: one clip per class (10 clips) JS vs torch probabilities.
   For the full 80-clip audit use test_full_parity.js. */
"use strict";
const fs = require("fs");
const src = fs.readFileSync(__dirname + "/dsp.js", "utf8");
const DSP = eval(src + "\n;DSP;");
const torch = JSON.parse(fs.readFileSync(__dirname + "/torch_probs.json", "utf8"));
const seen = new Set(); const sample = [];
for (const r of torch) if (!seen.has(r.true)) { seen.add(r.true); sample.push(r); }

function readWav(path) {
  const b = fs.readFileSync(path);
  let off = 12, fmt = null, data = null;
  while (off + 8 <= b.length) {
    const id = b.toString("ascii", off, off + 4), sz = b.readUInt32LE(off + 4);
    if (id === "fmt ") fmt = { rate: b.readUInt32LE(off + 12) };
    if (id === "data") data = b.subarray(off + 8, off + 8 + sz);
    off += 8 + sz + (sz & 1);
  }
  const x = new Float32Array(data.length / 2);
  for (let i = 0; i < x.length; i++) x[i] = data.readInt16LE(i * 2) / 32768;
  return { x, rate: fmt.rate };
}

(async () => {
  const j = JSON.parse(fs.readFileSync(__dirname + "/model.json", "utf8"));
  global.fetch = async () => ({ json: async () => j });
  await DSP.loadModel("model.json");
  let maxDiff = 0;
  for (const row of sample) {
    const { x, rate } = readWav(`/tmp/ESC-50-master/audio/${row.file}`);
    const y = DSP.resampleTo16k(x, rate);
    const padded = new Float32Array(DSP.WIN);
    padded.set(y.subarray(0, Math.min(y.length, DSP.WIN)));
    const probs = DSP.softmax(DSP.forward(DSP.computeFeatures(padded)).logits);
    for (let k = 0; k < 10; k++) maxDiff = Math.max(maxDiff, Math.abs(probs[k] - row.probs[k]));
  }
  console.log(`smoke parity over ${sample.length} clips: maxProbDiff ${maxDiff}`);
  if (maxDiff > 1e-4) { console.error("FAIL"); process.exit(1); }
  console.log("PASS");
})();
