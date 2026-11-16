/* Full parity test: deployed JS inference vs PyTorch reference, all 80 fold-5 test clips.
   Loads the app's actual dsp.js, computes features + forward in JS, and compares
   probabilities against torch_probs.json produced by eval_remeasure.py.
   Run: node test_full_parity.js            (from the repo root, with /tmp/ESC-50-master present)
*/
"use strict";
const fs = require("fs");

// load the real app DSP code
const src = fs.readFileSync(__dirname + "/dsp.js", "utf8");
const DSP = eval(src + "\n;DSP;");

const ESC = "/tmp/ESC-50-master";
const CLASSES = ["crying_baby","clock_alarm","door_wood_knock","glass_breaking","siren","car_horn","dog","footsteps","thunderstorm","pouring_water"];

function readWav(path) {
  const b = fs.readFileSync(path);
  let off = 12, fmt = null, data = null;
  while (off + 8 <= b.length) {
    const id = b.toString("ascii", off, off + 4);
    const sz = b.readUInt32LE(off + 4);
    if (id === "fmt ") fmt = { rate: b.readUInt32LE(off + 12), bits: b.readUInt16LE(off + 22), ch: b.readUInt16LE(off + 10) };
    if (id === "data") data = b.subarray(off + 8, off + 8 + sz);
    off += 8 + sz + (sz & 1);
  }
  const n = data.length / 2;
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = data.readInt16LE(i * 2) / 32768;
  return { x, rate: fmt.rate };
}

(async () => {
  // load weights exactly as the app does
  const j = JSON.parse(fs.readFileSync(__dirname + "/model.json", "utf8"));
  global.fetch = async () => ({ json: async () => j });
  await DSP.loadModel("model.json");

  const torch = JSON.parse(fs.readFileSync(__dirname + "/torch_probs.json", "utf8"));
  const THR = 0.55;
  let maxDiff = 0, argmaxDisagree = 0, abstainDisagree = 0, correctJS = 0;

  for (const row of torch) {
    const { x, rate } = readWav(`${ESC}/audio/${row.file}`);
    let y = DSP.resampleTo16k(x, rate);
    y = y.subarray(0, DSP.WIN);
    const padded = new Float32Array(DSP.WIN);
    padded.set(y.subarray(0, Math.min(y.length, DSP.WIN)));
    const feats = DSP.computeFeatures(padded);
    const { logits } = DSP.forward(feats);
    const probs = DSP.softmax(logits);
    const jsTop = probs.indexOf(Math.max(...probs));
    const tTop = row.probs.indexOf(Math.max(...row.probs));
    for (let k = 0; k < 10; k++) maxDiff = Math.max(maxDiff, Math.abs(probs[k] - row.probs[k]));
    if (jsTop !== tTop) argmaxDisagree++;
    if ((probs[jsTop] >= THR) !== (row.probs[tTop] >= THR)) abstainDisagree++;
    if (CLASSES[jsTop] === row.true) correctJS++;
  }
  const out = {
    clips: torch.length,
    maxProbDiff: maxDiff,
    argmaxDisagreements: argmaxDisagree,
    abstainDisagreementsAt055: abstainDisagree,
    jsTestAccuracy: correctJS / torch.length,
    window: "first 4 s of clip",
    features: "log-mel floored at max(acc,1e-6), per-clip standardization",
  };
  fs.writeFileSync(__dirname + "/parity.json", JSON.stringify(out, null, 1));
  console.log(JSON.stringify(out, null, 1));
  if (argmaxDisagree || abstainDisagree || maxDiff > 1e-4) { console.error("PARITY FAIL"); process.exit(1); }
  console.log("PARITY PASS");
})();
