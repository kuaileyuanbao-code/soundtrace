// Run the real app.js entry points with simulated DOM/audio and captured requests.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { File } = require("node:buffer");

const sourceArg = process.argv.indexOf("--source");
const sourcePath = sourceArg < 0 ? path.join(__dirname, "app.js") : process.argv[sourceArg + 1];
const source = fs.readFileSync(sourcePath, "utf8").replace(/\ninitialize\(\);\s*$/, "\n");
const modes = ["fast", "main", "review", "text"];
const available = () => ({ main: { available: true }, fast: { available: true }, review: { available: true }, text: { available: true } });
const fixtures = [];
let checks = 0;

function harness(mode) {
  const nodes = new Map();
  const node = selector => {
    if (!nodes.has(selector)) nodes.set(selector, { value: "", checked: false, disabled: false,
      textContent: "", hidden: false, classList: { add() {}, remove() {}, toggle() {} },
      setAttribute() {}, click() {}, pause() {}, play: async () => {}, close() {}, showModal() {} });
    return nodes.get(selector);
  };
  node("#modeSelect").value = mode;
  node("#speakerCountSelect").value = "2";
  const store = new Map();
  const requests = [];
  const warmups = [];
  let health = available();
  let rejection = null;
  let permissionError = null;
  let microphoneCalls = 0;
  const response = (body, ok = true, status = 200) => ({ ok, status, json: async () => body });
  class Context {
    sampleRate = 16000;
    destination = {};
    resume = async () => {};
    close = async () => {};
    createMediaStreamSource = () => ({ connect() {}, disconnect() {} });
    createScriptProcessor = () => ({ connect() {}, disconnect() {}, onaudioprocess: null });
    createGain = () => ({ gain: { value: 0 }, connect() {}, disconnect() {} });
  }
  const sandbox = { console, File, Blob, FormData, URL: { createObjectURL: () => "blob:test", revokeObjectURL() {} },
    performance, setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {},
    document: { querySelector: node, querySelectorAll: () => [] },
    localStorage: { getItem: key => store.get(key) || null, setItem: (key, value) => store.set(key, value), removeItem: key => store.delete(key) },
    window: { AudioContext: Context }, AudioContext: Context,
    navigator: { mediaDevices: { getUserMedia: async () => {
      microphoneCalls++;
      if (permissionError) throw permissionError;
      return { getTracks: () => [{ stop() {} }] };
    } } },
    fetch: async (url, options = {}) => {
      if (url === "/api/health") return response(health);
      if (url.startsWith('/api/warmup?mode=')) {
        warmups.push(url.split('=')[1]);
        return response({ok:true, status:'on_demand'});
      }
      if (url === "/api/jobs") {
        const body = options.body;
        const request = { mode: body.get("mode"), speaker_count: body.get("speaker_count"),
          voiceprint_ids: body.get("voiceprint_ids"), file: body.get("audio") };
        requests.push(request);
        return rejection ? response({ detail: rejection }, false, 503) : response({ id: "fake-job", mode: request.mode });
      }
      throw new Error(`Unexpected request ${url}`);
    },
  };
  const context = vm.createContext(sandbox);
  vm.runInContext(source + `\nrender = () => {}; drawAudioWave = async () => {};
    showJob = () => {}; pollJob = async () => {};
    globalThis.api = { state, startMicRecording, stopMicRecording, setAudioFile, submitTranscription, checkBackend, loadSettings, requestWarmup };
    state.health = ${JSON.stringify(available())};`, context);
  return { api: context.api, node, store, requests, warmups, setHealth: value => { health = value; context.api.state.health = value; },
    deny: () => { permissionError = Object.assign(new Error("denied"), { name: "NotAllowedError" }); },
    reject: message => { rejection = message; }, microphoneCalls: () => microphoneCalls };
}

async function capture(h, origin, mode) {
  assert.equal(h.requests.length, 1);
  const request = h.requests[0];
  assert.equal(request.mode, mode);
  assert.equal(request.speaker_count, ["fast", "text"].includes(mode) ? "0" : "2");
  if (mode === "text") assert.equal(request.voiceprint_ids, "");
  assert.equal(h.node("#modeSelect").value, mode);
  fixtures.push({ origin, selected_mode: mode, ...request, file: undefined,
    filename: request.file.name, audio_base64: Buffer.from(await request.file.arrayBuffer()).toString("base64") });
  checks++;
}

async function main() {
  for (const mode of modes) {
    const h = harness(mode);
    await h.api.requestWarmup();
    assert.deepEqual(h.warmups, [mode], 'Preload must follow restored/selected mode');
    h.node('#modeSelect').value = mode === 'fast' ? 'main' : 'fast';
    h.node('#modeSelect').onchange();
    await Promise.resolve();
    assert.equal(h.warmups.at(-1), h.node('#modeSelect').value);
    checks++;
  }
  for (const mode of modes) {
    const mic = harness(mode);
    await mic.api.startMicRecording();
    assert.equal(mic.node("#modeSelect").value, mode, "Recording must keep the selected mode");
    assert.equal(mic.node("#modeSelect").disabled, true);
    mic.api.state.recording.chunks = [new Float32Array(16000)];
    mic.api.state.recording.frames = 16000;
    // The normal four-second health refresh also runs during recording.
    await mic.api.checkBackend();
    await mic.api.stopMicRecording(true);
    await capture(mic, "microphone", mode);
    const imported = harness(mode);
    await imported.node("#audioInput").onchange({ target: { files: [
      new File([await mic.requests[0].file.arrayBuffer()], "imported.wav", { type: "audio/wav" }),
    ] } });
    await imported.node("#transcribeButton").onclick();
    await capture(imported, "import", mode);
  }
  for (const mode of modes) {
    const h = harness(mode);
    h.setHealth({ ...available(), [mode]: { available: false, missing: [`missing-${mode}-model`] } });
    await h.api.checkBackend();
    assert.equal(h.node("#modeSelect").value, mode, "Health checks must not choose another route");
    await h.api.startMicRecording();
    assert.equal(h.microphoneCalls(), 0);
    assert.match(h.node("#toast").textContent, new RegExp(`missing-${mode}-model`));
    h.api.state.audioFile = new File(["test"], "test.wav");
    await h.api.submitTranscription();
    assert.equal(h.requests.length, 0);
    assert.match(h.node("#jobDetail").textContent, new RegExp(`missing-${mode}-model`));
    assert.equal(h.node("#modeSelect").value, mode);
    checks++;
    const stale = harness(mode);
    stale.api.state.audioFile = new File(["test"], "test.wav");
    stale.reject(`所选 ${mode} 模式缺少本地模型`);
    await stale.api.submitTranscription();
    assert.equal(stale.requests.length, 1, "Do not retry in a fallback mode");
    assert.equal(stale.requests[0].mode, mode);
    assert.equal(stale.node("#modeSelect").value, mode);
    assert.match(stale.node("#jobDetail").textContent, /缺少本地模型/);
    checks++;
  }
  const denied = harness("fast");
  denied.deny();
  await denied.api.startMicRecording();
  assert.equal(denied.node("#modeSelect").value, "fast");
  assert.equal(denied.requests.length, 0);
  assert.match(denied.node("#toast").textContent, /没有获得话筒权限/);
  checks++;
  for (const mode of modes) for (const version of [3, 4]) {
    const h = harness("main");
    h.store.set("soundtrace.settings.v1", JSON.stringify({ version, mode }));
    h.api.loadSettings();
    assert.equal(h.node("#modeSelect").value, mode, "Restore the saved explicit mode");
    checks++;
  }
  const outputArg = process.argv.indexOf("--fixtures");
  if (outputArg >= 0) fs.writeFileSync(process.argv[outputArg + 1], JSON.stringify(fixtures, null, 2));
  console.log(`Passed ${checks} front-end mode cases; real app.js, simulated microphone/DOM, no model inference.`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
