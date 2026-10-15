const colors = ["#7868d7", "#dd79a9", "#5aaea5", "#8f75c9", "#d59a5f", "#639ed3"];
const STORAGE_KEYS = { history: "soundtrace.history.v1", settings: "soundtrace.settings.v1" };
const state = { rows: [], resultMode: null, name: "选择一段录音", duration: 0, audioFile: null, audioUrl: null, stopAt: null, jobId: null, health: null, recording: null, historyId: null, voiceprints: [], selectedVoiceprints: new Set(), voiceprintRecording: null };
const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const audio = $("#audioPlayer");
const speakerName = id => /^(?:spk|voice|turn)\d+$/i.test(String(id)) ? `说话人 ${Number(String(id).replace(/\D/g, ""))}` : String(id);
const speakerColor = id => {
  const numeric = Number(String(id).replace(/\D/g, ""));
  if (numeric) return colors[(numeric - 1) % colors.length];
  const hash = [...String(id)].reduce((value, character) => ((value * 31) + character.charCodeAt(0)) >>> 0, 0);
  return colors[hash % colors.length];
};
const compact = value => String(value || "").replace(/\s+/g, "").trim();
const formatTime = seconds => `${String(Math.floor((seconds || 0) / 60)).padStart(2, "0")}:${String(Math.floor((seconds || 0) % 60)).padStart(2, "0")}`;
const formatWords = text => (String(text).toLowerCase().match(/[a-z0-9']+|[\u3400-\u9fff]|[^\s]/g) || []).join(" ");
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
const modeName = mode => ({ text: "纯文字转写", main: "统一识别", review: "精细复核", conversation: "旧版短对话", fast: "快速预览", accurate: "旧版高精度" })[mode] || "识别结果";
let historySaveTimer = null;

const textRows = rows => rows.length > 0 && rows.every(row => row.speaker === null && row.speaker_analyzed === false && row.timestamp_kind === "audio_extent");
const isTextResult = () => state.resultMode === "text" || textRows(state.rows);
const plainText = () => state.rows.map(row => String(row.words ?? "")).join("\n");

function flushHistory() {
  clearTimeout(historySaveTimer);
  updateCurrentHistory();
}

function setResult(payload, fallbackMode) {
  if (!Array.isArray(payload.rows) || !payload.rows.length) throw new Error("转写结果为空或格式无效");
  const pure = payload.mode === "text" || fallbackMode === "text" || textRows(payload.rows);
  if (pure && !textRows(payload.rows)) throw new Error("纯文字结果缺少未分析人物或音频范围标记");
  state.resultMode = pure ? "text" : (payload.mode || fallbackMode || "imported");
  state.rows = payload.rows.map(row => ({ ...row }));
  if (pure && typeof payload.text === "string") state.rows = [{ ...state.rows[0], words: payload.text,
    start_time: 0, end_time: Math.max(...payload.rows.map(row => Number(row.end_time) || 0)) }];
}

function toast(message) {
  $("#toast").textContent = message;
  $("#toast").classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => $("#toast").classList.remove("show"), 2500);
}

const wait = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

async function responseJson(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `后端返回 ${response.status}`);
  return payload;
}

function setServiceState(kind, text) {
  const element = $("#serviceState");
  element.classList.remove("offline", "busy");
  if (kind) element.classList.add(kind);
  $("#serviceStateText").textContent = text;
}

function selectedModeError(mode = $("#modeSelect").value) {
  if (!["fast", "main", "review", "text"].includes(mode)) return "请选择有效的识别模式";
  if (mode === "text" && state.health && !state.health.text) return "当前后端尚不支持纯文字转写，请更新并重启本地服务";
  const availability = state.health?.[mode];
  if (availability?.available !== false) return null;
  const missing = availability.missing?.length ? `缺少本地环境或模型：${availability.missing.join("、")}` : "环境或模型未就绪，请查看本地服务状态";
  return `${modeName(mode)}无法运行：${missing}`;
}

async function checkBackend() {
  try {
    const health = await responseJson(await fetch("/api/health", { cache: "no-store" }));
    state.health = health;
    const mode = $("#modeSelect").value;
    const warming = ["main", "review"].includes(mode) && health.main?.warmup_status === "loading";
    const stateText = warming ? (health.main.warmup_message || "正在预热模型") : health.busy ? "本地 GPU 忙碌"
      : mode === "text" ? (health.text?.model_loaded ? "纯文字模型已加载" : "纯文字首次使用时加载")
      : health.main?.model_loaded ? "统一识别模型已就绪" : "后端已连接";
    setServiceState(health.busy || warming ? "busy" : "", stateText);
    const reviewOption = $('#modeSelect option[value="review"]');
    const mainOption = $('#modeSelect option[value="main"]');
    mainOption.disabled = !health.main?.available;
    mainOption.textContent = health.main?.available ? "统一识别 · 推荐" : "统一识别 · 环境未就绪";
    reviewOption.disabled = !health.review?.available;
    reviewOption.textContent = health.review?.available ? "精细复核 · 耗时更长" : "精细复核 · 环境未就绪";
    const textOption = $('#modeSelect option[value="text"]');
    textOption.disabled = health.text?.available !== true;
    textOption.textContent = health.text?.available ? "纯文字转写" : "纯文字转写 · 环境未就绪";
    if (selectedModeError()) setServiceState("offline", `${modeName($("#modeSelect").value)}不可用`);
    updateModeControls();
    return health;
  } catch {
    state.health = null;
    setServiceState("offline", "后端未连接");
    return null;
  }
}

async function requestWarmup() {
  const mode = $("#modeSelect").value;
  if (state.jobId || !state.health || selectedModeError(mode)) return;
  if (["main", "review"].includes(mode) && (state.health.main?.model_loaded || state.health.main?.warmup_status === "loading")) return;
  try {
    const response = await responseJson(await fetch(`/api/warmup?mode=${mode}`, { method: "POST" }));
    if ($("#modeSelect").value !== mode || state.jobId) return;
    if (response.status === "loading") setServiceState("busy", "正在预热模型");
    if (response.status === "cancelling") setServiceState("busy", "正在停止未使用的统一识别预热");
  } catch (error) {
    console.warn("模型预热未启动", error);
  }
}

function loadSettings() {
  try {
    const settings = JSON.parse(localStorage.getItem(STORAGE_KEYS.settings) || "{}");
    const mode = ({ conversation: "main", accurate: "review" })[settings.mode] || settings.mode;
    if (["main", "fast", "review", "text"].includes(mode)) $("#modeSelect").value = mode;
    if (settings.version >= 2 && ["0", "1", "2", "3", "4", "5", "6"].includes(String(settings.speakerCount))) {
      $("#speakerCountSelect").value = String(settings.speakerCount);
    }
    if (Array.isArray(settings.selectedVoiceprints)) state.selectedVoiceprints = new Set(settings.selectedVoiceprints.map(String));
    $("#useVoiceprints").checked = Boolean(settings.useVoiceprints);
  } catch { /* 使用默认设置 */ }
  updateModeControls();
}

function saveSettings() {
  localStorage.setItem(STORAGE_KEYS.settings, JSON.stringify({
    version: 4,
    mode: $("#modeSelect").value,
    speakerCount: $("#speakerCountSelect").value,
    useVoiceprints: $("#useVoiceprints").checked,
    selectedVoiceprints: [...state.selectedVoiceprints]
  }));
}

function updateModeControls() {
  const mode = $("#modeSelect").value;
  $(".mode-picker").classList.toggle("text-only", mode === "text");
  $("#modeHint").textContent = ({ text: "纯文字只提供文字，不进行人物识别和精细时间对齐；首次使用需要加载模型。",
    fast: "快速预览提供文字、人物和时间。", main: "统一识别提供文字、说话人和对齐时间。",
    review: "精细复核使用多模型检查文字，首次加载和复核耗时更长。" })[mode] || "请选择识别模式";
  if (state.jobId || state.recording) {
    $("#useVoiceprints").disabled = true;
    $("#speakerCountSelect").disabled = true;
    return;
  }
  const conversation = ["main", "review"].includes($("#modeSelect").value);
  const selectedCount = state.selectedVoiceprints.size;
  if (!conversation || !selectedCount) $("#useVoiceprints").checked = false;
  $("#useVoiceprints").disabled = !conversation || !selectedCount;
  $("#speakerCountSelect").disabled = !conversation || $("#useVoiceprints").checked;
  $("#voiceprintSwitchText").textContent = selectedCount ? `声纹 ${selectedCount}` : "声纹";
}

function readHistory() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEYS.history) || "[]");
    return Array.isArray(value) ? value : [];
  } catch { return []; }
}

function writeHistory(items) {
  const trimmed = items.slice(0, 20);
  while (trimmed.length) {
    try {
      localStorage.setItem(STORAGE_KEYS.history, JSON.stringify(trimmed));
      return;
    } catch { trimmed.pop(); }
  }
  localStorage.removeItem(STORAGE_KEYS.history);
}

async function loadVoiceprints() {
  try {
    const payload = await responseJson(await fetch("/api/voiceprints", { cache: "no-store" }));
    state.voiceprints = Array.isArray(payload.profiles) ? payload.profiles : [];
    const available = new Set(state.voiceprints.map(profile => String(profile.id)));
    state.selectedVoiceprints = new Set([...state.selectedVoiceprints].filter(id => available.has(id)));
    renderVoiceprints();
    updateModeControls();
    saveSettings();
  } catch (error) {
    state.voiceprints = [];
    renderVoiceprints();
    console.warn("声纹库读取失败", error);
  }
}

function renderVoiceprints() {
  const list = $("#voiceprintList");
  list.innerHTML = state.voiceprints.length ? state.voiceprints.map(profile => {
    const quality = Number(profile.quality) >= .75 ? "质量良好" : Number(profile.quality) >= .55 ? "质量一般" : "建议重新录入";
    const checked = state.selectedVoiceprints.has(String(profile.id)) ? "checked" : "";
    return `<article class="voiceprint-item">
      <input class="voiceprint-check" type="checkbox" data-voiceprint-select="${escapeHtml(profile.id)}" ${checked} aria-label="选择 ${escapeHtml(profile.name)} 参与识别">
      <div class="voiceprint-copy"><strong>${escapeHtml(profile.name)}</strong><span>${Number(profile.duration_seconds).toFixed(0)} 秒 · ${Number(profile.sample_count)} 个样本 · <b class="voiceprint-quality">${quality}</b></span></div>
      <button class="history-delete" type="button" data-voiceprint-delete="${escapeHtml(profile.id)}" aria-label="删除 ${escapeHtml(profile.name)}" title="删除">×</button>
    </article>`;
  }).join("") : `<p class="history-empty">还没有声纹档案<br>输入姓名并录制第一位成员的声音</p>`;
  $$('[data-voiceprint-select]').forEach(input => input.onchange = () => {
    input.checked ? state.selectedVoiceprints.add(input.dataset.voiceprintSelect) : state.selectedVoiceprints.delete(input.dataset.voiceprintSelect);
    updateVoiceprintSelection();
  });
  $$('[data-voiceprint-delete]').forEach(button => button.onclick = async () => {
    const profile = state.voiceprints.find(item => String(item.id) === button.dataset.voiceprintDelete);
    if (!profile || !confirm(`确定删除“${profile.name}”的声纹档案吗？`)) return;
    try {
      await responseJson(await fetch(`/api/voiceprints/${encodeURIComponent(profile.id)}`, { method: "DELETE" }));
      state.selectedVoiceprints.delete(String(profile.id));
      await loadVoiceprints();
      toast(`已删除 ${profile.name} 的声纹`);
    } catch (error) { toast(error.message); }
  });
  updateVoiceprintSelection(false);
}

function updateVoiceprintSelection(rerender = false) {
  const count = state.selectedVoiceprints.size;
  $("#voiceprintSelectionText").textContent = count ? `已选择 ${count} 位成员参与识别` : "尚未选择参与识别的成员";
  $("#selectAllVoiceprintsButton").textContent = count === state.voiceprints.length && count ? "取消全选" : "全选";
  if (rerender) renderVoiceprints();
  updateModeControls();
  saveSettings();
}

async function startVoiceprintRecording() {
  const name = $("#voiceprintName").value.trim();
  if (!name) return toast("请先输入成员姓名");
  if (state.recording || state.jobId) return toast("请等待当前录音或识别任务完成");
  if (!navigator.mediaDevices?.getUserMedia) return toast("当前浏览器不支持话筒录音");
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContextClass();
    await context.resume();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    const silentGain = context.createGain();
    silentGain.gain.value = 0;
    const recording = { name, context, stream, source, processor, silentGain, chunks: [], frames: 0, timer: null };
    processor.onaudioprocess = event => {
      const samples = new Float32Array(event.inputBuffer.getChannelData(0));
      recording.chunks.push(samples);
      recording.frames += samples.length;
    };
    source.connect(processor);
    processor.connect(silentGain);
    silentGain.connect(context.destination);
    state.voiceprintRecording = recording;
    $("#voiceprintName").disabled = true;
    $("#voiceprintRecordButton").classList.add("recording");
    $("#voiceprintRecordButton").textContent = "结束录入";
    const update = () => {
      if (state.voiceprintRecording !== recording) return;
      const seconds = recording.frames / recording.context.sampleRate;
      $("#voiceprintRecordStatus").textContent = `正在录制 ${formatTime(seconds)} · 请保持只有 ${recording.name} 一个人说话`;
      if (seconds >= 90) stopVoiceprintRecording();
    };
    update();
    recording.timer = setInterval(update, 250);
  } catch (error) {
    stream?.getTracks().forEach(track => track.stop());
    toast(error?.name === "NotAllowedError" ? "请允许浏览器使用麦克风" : `无法启动话筒：${error.message || error}`);
  }
}

async function stopVoiceprintRecording() {
  const recording = state.voiceprintRecording;
  if (!recording) return;
  state.voiceprintRecording = null;
  clearInterval(recording.timer);
  recording.processor.onaudioprocess = null;
  recording.source.disconnect();
  recording.processor.disconnect();
  recording.silentGain.disconnect();
  recording.stream.getTracks().forEach(track => track.stop());
  await recording.context.close();
  const duration = recording.frames / recording.context.sampleRate;
  const button = $("#voiceprintRecordButton");
  button.classList.remove("recording");
  if (duration < 8) {
    $("#voiceprintName").disabled = false;
    button.textContent = "开始录入";
    $("#voiceprintRecordStatus").textContent = "录音不足 8 秒，请重新录制几句完整的话。";
    return toast("声纹录音至少需要 8 秒");
  }
  button.disabled = true;
  button.textContent = "正在提取";
  $("#voiceprintRecordStatus").textContent = "CAM++ 正在提取声纹向量，原始录音随后会删除…";
  try {
    const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "").replace("T", "_");
    const file = new File([encodeWav(recording.chunks, recording.frames, recording.context.sampleRate)], `voiceprint_${stamp}.wav`, { type: "audio/wav" });
    const body = new FormData();
    body.append("name", recording.name);
    body.append("audio", file);
    const payload = await responseJson(await fetch("/api/voiceprints", { method: "POST", body }));
    state.selectedVoiceprints.add(String(payload.profile.id));
    $("#voiceprintName").value = "";
    await loadVoiceprints();
    $("#voiceprintRecordStatus").textContent = `${payload.profile.name} 的声纹已保存，可以继续录入下一位成员。`;
    toast(`已录入 ${payload.profile.name} 的声纹`);
  } catch (error) {
    $("#voiceprintRecordStatus").textContent = error.message;
    toast(error.message);
  } finally {
    $("#voiceprintName").disabled = false;
    button.disabled = false;
    button.textContent = "开始录入";
    await checkBackend();
  }
}

function saveCurrentToHistory(mode) {
  if (!state.rows.length) return;
  const id = state.historyId || (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`);
  const item = {
    id,
    name: state.name,
    createdAt: new Date().toISOString(),
    duration: state.duration,
    mode: isTextResult() ? "text" : mode,
    speakerCount: isTextResult() ? null : new Set(state.rows.map(row => row.speaker)).size,
    rows: state.rows.map(row => ({ ...row }))
  };
  writeHistory([item, ...readHistory().filter(entry => entry.id !== id)]);
  state.historyId = id;
  renderHistory();
}

function updateCurrentHistory() {
  if (!state.historyId) return;
  const items = readHistory();
  const item = items.find(entry => entry.id === state.historyId);
  if (!item) return;
  item.rows = state.rows.map(row => ({ ...row }));
  item.duration = state.duration;
  writeHistory(items);
}

function renderHistory() {
  const items = readHistory();
  $("#historyList").innerHTML = items.length ? items.map(item => {
    const date = new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(new Date(item.createdAt));
    const pure = item.mode === "text" || textRows(item.rows || []);
    const detail = pure ? `纯文字转写 · 音频长度 ${formatTime(item.duration)}` : `${modeName(item.mode)} · ${Number(item.speakerCount) || 0} 人 · ${Number(item.rows?.length) || 0} 段 · ${formatTime(item.duration)}`;
    return `<article class="history-item">
      <button class="history-open" type="button" data-history-open="${escapeHtml(item.id)}"><strong>${escapeHtml(item.name || "未命名录音")}</strong><span>${date} · ${detail}</span></button>
      <button class="history-delete" type="button" data-history-delete="${escapeHtml(item.id)}" aria-label="删除这条历史" title="删除">×</button>
    </article>`;
  }).join("") : `<p class="history-empty">还没有识别历史<br>完成一次录音识别后会自动保存在这里</p>`;
  $$('[data-history-open]').forEach(button => button.onclick = () => openHistory(button.dataset.historyOpen));
  $$('[data-history-delete]').forEach(button => button.onclick = () => {
    writeHistory(readHistory().filter(item => item.id !== button.dataset.historyDelete));
    if (state.historyId === button.dataset.historyDelete) state.historyId = null;
    renderHistory();
  });
}

function openHistory(id) {
  flushHistory();
  const item = readHistory().find(entry => entry.id === id);
  if (!item?.rows?.length) return;
  audio.pause();
  audio.removeAttribute("src");
  audio.load();
  if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  state.audioUrl = null;
  state.audioFile = null;
  state.historyId = item.id;
  setResult({ rows: item.rows, mode: item.mode }, item.mode);
  state.name = item.name || "历史记录";
  state.duration = Number(item.duration) || Math.max(...state.rows.map(row => Number(row.end_time) || 0));
  $("#historyDialog").close();
  render();
  toast("已打开历史文字；播放时请重新选择原音频");
}

function showJob(job) {
  const panel = $("#jobProgress");
  panel.hidden = false;
  panel.classList.toggle("failed", job.status === "failed");
  $("#jobStage").textContent = job.message || "正在识别";
  const mode = modeName(job.mode);
  const elapsed = job.elapsed_seconds ?? (job.started_at ? Math.max(0, Date.now() / 1000 - job.started_at) : 0);
  const detail = job.warnings?.length ? job.warnings.join("；") : "在本机处理，音频不会上传到云端";
  $("#jobDetail").textContent = `${mode} · ${job.status === "queued" ? "正在排队" : `用时 ${elapsed.toFixed(1)} 秒`} · ${detail}`;
  const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
  $("#jobPercent").textContent = `${Math.round(progress)}%`;
  $("#jobProgressBar").style.width = `${progress}%`;
}

async function pollJob(jobId, clientStarted, acceptedAt) {
  while (state.jobId === jobId) {
    const job = await responseJson(await fetch(`/api/jobs/${jobId}`, { cache: "no-store" }));
    showJob(job);
    if (job.status === "completed") {
      const payload = await responseJson(await fetch(job.result_url, { cache: "no-store" }));
      flushHistory();
      setResult(payload, job.mode);
      state.name = payload.session_id || state.name;
      state.duration = Math.max(state.duration, ...state.rows.map(row => Number(row.end_time) || 0));
      render();
      saveCurrentToHistory(payload.mode || job.mode);
      if (Number.isFinite(clientStarted) && Number.isFinite(acceptedAt)) {
        const completedAt = performance.now();
        // Best-effort observation after rendering; it must not fail recognition.
        fetch(`/api/jobs/${jobId}/client-timing`, { method: "POST",
          headers: { "Content-Type": "application/json" }, body: JSON.stringify({
            user_end_to_end_seconds: (completedAt - clientStarted) / 1000,
            client_submit_seconds: (acceptedAt - clientStarted) / 1000,
            client_after_accept_seconds: (completedAt - acceptedAt) / 1000,
          }) }).catch(() => {});
      }
      toast(`识别完成 · ${isTextResult() ? "纯文字转写" : `${job.speakers} 位说话人`} · 用时 ${(job.elapsed_seconds || 0).toFixed(1)} 秒`);
      return;
    }
    if (job.status === "failed") throw new Error(job.error || "识别失败");
    await wait(900);
  }
}

function render() {
  const pure = isTextResult();
  const speakers = [...new Set(state.rows.map(row => row.speaker))].sort();
  document.body.classList.toggle("has-audio", Boolean(state.audioFile));
  document.body.classList.toggle("has-rows", state.rows.length > 0);
  $("#recordingName").textContent = state.name;
  $("#recordingMeta").textContent = state.audioFile
    ? `本地音频 · ${formatTime(state.duration)}`
    : "支持 WAV、MP3、FLAC、OGG";
  $("#durationTime").textContent = formatTime(state.duration);
  $("#summaryText").textContent = pure ? "纯文字 · 不分析人物和精细时间，可直接编辑" : `${speakers.length} 位说话人 · ${state.rows.length} 个片段`;
  $("#speakerKey").hidden = pure;
  $("#exportTextButton").hidden = !pure;
  $("#speakerKey").innerHTML = pure ? "" : speakers.map(id => `<span><i style="background:${speakerColor(id)}"></i>${speakerName(id)}</span>`).join("");

  const query = compact($("#transcriptSearch").value).toLowerCase();
  if (pure) {
    const text = plainText();
    $("#transcriptList").innerHTML = compact(text).toLowerCase().includes(query)
      ? `<article class="plain-transcript"><textarea id="plainTextEditor" aria-label="纯文字转写内容" spellcheck="false">${escapeHtml(text)}</textarea></article>`
      : `<p class="empty">没有匹配的文字</p>`;
    const field = $("#plainTextEditor");
    if (field) {
      field.style.height = `${Math.max(160, field.scrollHeight)}px`;
      field.oninput = () => {
        state.rows = [{ ...state.rows[0], words: field.value }];
        field.style.height = "auto";
        field.style.height = `${Math.max(160, field.scrollHeight)}px`;
        clearTimeout(historySaveTimer);
        historySaveTimer = setTimeout(updateCurrentHistory, 350);
      };
    }
    if (state.audioFile) drawSyntheticWave();
    return;
  }
  const visible = state.rows.map((row, index) => ({ row, index })).filter(item => compact(item.row.words).toLowerCase().includes(query));
  $("#transcriptList").innerHTML = visible.length ? visible.map(({ row, index }) => `
    <article class="segment">
      <div class="segment-person"><strong><i class="speaker-dot" style="background:${speakerColor(row.speaker)}"></i>${speakerName(row.speaker)}</strong><time>${formatTime(row.start_time)} – ${formatTime(row.end_time)}</time></div>
      <textarea data-row="${index}" aria-label="转写文字">${escapeHtml(compact(row.words))}</textarea>
      <button class="segment-play" data-play="${index}" title="播放片段" aria-label="播放片段">▶</button>
    </article>`).join("") : `<p class="empty">没有匹配的文字</p>`;

  $$("textarea[data-row]").forEach(field => {
    field.style.height = `${Math.max(44, field.scrollHeight)}px`;
    field.oninput = () => {
      field.style.height = "auto";
      field.style.height = `${field.scrollHeight}px`;
      state.rows[Number(field.dataset.row)].words = formatWords(field.value);
      clearTimeout(historySaveTimer);
      historySaveTimer = setTimeout(updateCurrentHistory, 350);
    };
  });
  $$('[data-play]').forEach(button => button.onclick = () => playSegment(state.rows[Number(button.dataset.play)]));
  if (state.audioFile) drawSyntheticWave();
}

function canvasContext() {
  const canvas = $("#waveform");
  const ratio = devicePixelRatio || 1;
  canvas.width = Math.max(1, canvas.clientWidth * ratio);
  canvas.height = Math.max(1, canvas.clientHeight * ratio);
  return { canvas, context: canvas.getContext("2d"), ratio };
}

function drawBars(peaks) {
  const { canvas, context, ratio } = canvasContext();
  const middle = canvas.height / 2;
  context.clearRect(0, 0, canvas.width, canvas.height);
  const gradient = context.createLinearGradient(0, 0, canvas.width, 0);
  gradient.addColorStop(0, "#b39fe5");
  gradient.addColorStop(.52, "#8ea9dd");
  gradient.addColorStop(1, "#e2a6c8");
  context.fillStyle = gradient;
  peaks.forEach((peak, index) => {
    const height = Math.max(2 * ratio, peak * canvas.height * .82);
    context.fillRect(index * 4 * ratio, middle - height / 2, 2 * ratio, height);
  });
}

function drawSyntheticWave() {
  const count = Math.max(20, Math.floor($("#waveform").clientWidth / 4));
  drawBars(Array.from({ length: count }, (_, i) => .12 + Math.abs(Math.sin(i * .43) * .62 + Math.sin(i * .11) * .18)));
}

async function drawAudioWave(file) {
  const context = new AudioContext();
  const decoded = await context.decodeAudioData(await file.arrayBuffer());
  const data = decoded.getChannelData(0);
  const count = Math.max(20, Math.floor($("#waveform").clientWidth / 4));
  const block = Math.max(1, Math.floor(data.length / count));
  drawBars(Array.from({ length: count }, (_, i) => {
    let peak = 0;
    for (let j = i * block; j < Math.min(data.length, (i + 1) * block); j++) peak = Math.max(peak, Math.abs(data[j]));
    return peak;
  }));
  await context.close();
}

async function setAudioFile(file, knownDuration = 0) {
  flushHistory();
  state.audioFile = file;
  state.name = file.name.replace(/\.[^.]+$/, "");
  state.duration = knownDuration;
  state.jobId = null;
  state.historyId = null;
  $("#jobProgress").hidden = true;
  if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  state.audioUrl = URL.createObjectURL(file);
  audio.src = state.audioUrl;
  audio.onloadedmetadata = () => {
    state.duration = Number.isFinite(audio.duration) ? audio.duration : knownDuration;
    render();
  };
  render();
  try { await drawAudioWave(file); } catch { drawSyntheticWave(); }
}

function encodeWav(chunks, frameCount, sampleRate) {
  const buffer = new ArrayBuffer(44 + frameCount * 2);
  const view = new DataView(buffer);
  const writeText = (offset, text) => [...text].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)));
  writeText(0, "RIFF");
  view.setUint32(4, 36 + frameCount * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, frameCount * 2, true);
  let offset = 44;
  chunks.forEach(chunk => chunk.forEach(sample => {
    const clipped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
    offset += 2;
  }));
  return new Blob([buffer], { type: "audio/wav" });
}

function setRecordingControls(active) {
  const button = $("#micButton");
  button.classList.toggle("recording", active);
  button.setAttribute("aria-pressed", String(active));
  button.setAttribute("aria-label", active ? "停止录音并自动识别" : "开始话筒录音");
  button.title = active ? "点击停止录音并自动识别" : "点击开始录音，再次点击结束并自动识别";
  $(".mic-label").textContent = active ? "停止" : "话筒";
  $("#openAudioButton").disabled = active;
  $("#transcribeButton").disabled = active;
  $("#modeSelect").disabled = active;
  $("#speakerCountSelect").disabled = active;
  $("#useVoiceprints").disabled = active;
  if (!active) updateModeControls();
}

async function startMicRecording() {
  if (!navigator.mediaDevices?.getUserMedia) return toast("当前浏览器不支持话筒录音，请使用 localhost 页面和新版浏览器");
  if (state.jobId) return toast("请等待当前识别任务完成");
  if (state.voiceprintRecording) return toast("请先结束声纹录入");
  let stream;
  try {
    const modeError = selectedModeError();
    if (modeError) return toast(modeError);
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContextClass();
    await context.resume();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    const silentGain = context.createGain();
    silentGain.gain.value = 0;
    const recording = { context, stream, source, processor, silentGain, chunks: [], frames: 0, startedAt: performance.now(), timer: null };
    processor.onaudioprocess = event => {
      const samples = new Float32Array(event.inputBuffer.getChannelData(0));
      recording.chunks.push(samples);
      recording.frames += samples.length;
    };
    source.connect(processor);
    processor.connect(silentGain);
    silentGain.connect(context.destination);
    state.recording = recording;
    setRecordingControls(true);
    $("#jobProgress").hidden = true;
    const updateTimer = () => {
      if (state.recording !== recording) return;
      const seconds = recording.frames / recording.context.sampleRate;
      $("#recordingName").textContent = `正在录音 · ${formatTime(seconds)}`;
      $("#recordingMeta").textContent = "再次点击话筒，结束后将自动识别";
      if (seconds >= 300) stopMicRecording(true);
    };
    updateTimer();
    recording.timer = setInterval(updateTimer, 250);
  } catch (error) {
    stream?.getTracks().forEach(track => track.stop());
    const denied = error?.name === "NotAllowedError";
    toast(denied ? "没有获得话筒权限，请在浏览器地址栏允许麦克风" : `无法启动话筒：${error.message || error}`);
  }
}

async function stopMicRecording(autoTranscribe = true) {
  const recording = state.recording;
  if (!recording) return;
  state.recording = null;
  clearInterval(recording.timer);
  recording.processor.onaudioprocess = null;
  recording.source.disconnect();
  recording.processor.disconnect();
  recording.silentGain.disconnect();
  recording.stream.getTracks().forEach(track => track.stop());
  await recording.context.close();
  setRecordingControls(false);
  const duration = recording.frames / recording.context.sampleRate;
  if (duration < .3) {
    render();
    return toast("录音太短，请至少说一小句");
  }
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "").replace("T", "_");
  const file = new File([encodeWav(recording.chunks, recording.frames, recording.context.sampleRate)], `recording_${stamp}.wav`, { type: "audio/wav" });
  await setAudioFile(file, duration);
  if (autoTranscribe) await submitTranscription();
}

function playSegment(row) {
  if (!audio.src) return toast("请先选择对应音频");
  audio.currentTime = row.start_time;
  state.stopAt = row.end_time;
  audio.play();
}

$("#openAudioButton").onclick = () => $("#audioInput").click();
$("#openResultButton").onclick = () => $("#resultInput").click();
$("#audioInput").onchange = async event => {
  const file = event.target.files[0];
  if (!file) return;
  await setAudioFile(file);
};
$("#micButton").onclick = () => state.recording ? stopMicRecording(true) : startMicRecording();
$("#resultInput").onchange = async event => {
  try {
    const decoded = JSON.parse(await event.target.files[0].text());
    const payload = Array.isArray(decoded) ? { rows: decoded } : decoded;
    if (!payload || !Array.isArray(payload.rows) || !payload.rows.length) throw new Error();
    flushHistory();
    setResult(payload, "imported");
    const rows = state.rows;
    state.historyId = null;
    state.name = String(rows[0].session_id);
    state.duration = Math.max(...rows.map(row => Number(row.end_time)));
    render();
    saveCurrentToHistory("imported");
    toast(isTextResult() ? "已打开纯文字结果" : `已打开 ${rows.length} 个片段`);
  } catch { toast("无法读取该转写 JSON"); }
};
$("#masterPlayButton").onclick = () => {
  if (!audio.src) return toast("请先选择音频");
  state.stopAt = null;
  audio.paused ? audio.play() : audio.pause();
};
audio.onplay = () => $("#masterPlayButton").textContent = "Ⅱ";
audio.onpause = () => $("#masterPlayButton").textContent = "▶";
audio.ontimeupdate = () => {
  if (state.stopAt && audio.currentTime >= state.stopAt) { audio.pause(); state.stopAt = null; }
  $("#currentTime").textContent = formatTime(audio.currentTime);
  $("#playhead").style.left = `${audio.duration ? audio.currentTime / audio.duration * 100 : 0}%`;
};
$("#waveform").onclick = event => {
  if (!audio.src || !audio.duration) return;
  const box = event.currentTarget.getBoundingClientRect();
  audio.currentTime = (event.clientX - box.left) / box.width * audio.duration;
};
$("#speedSelect").onchange = event => audio.playbackRate = Number(event.target.value);
$("#transcriptSearch").oninput = render;
$("#exportButton").onclick = () => {
  flushHistory();
  if (isTextResult()) {
    const payload = { mode: "text", session_id: state.name, text: plainText(), duration_seconds: state.duration,
      speaker_analyzed: false, timestamp_kind: "audio_extent", rows: state.rows.map(row => ({ ...row, speaker: null,
        speaker_analyzed: false, timestamp_kind: "audio_extent" })) };
    return downloadResult(JSON.stringify(payload, null, 2), "application/json;charset=utf-8", `${state.name}.text.json`);
  }
  const rows = state.rows.map(row => ({
    end_time: Number(Number(row.end_time).toFixed(2)), session_id: String(row.session_id), speaker: String(row.speaker),
    start_time: Number(Number(row.start_time).toFixed(2)), words: formatWords(compact(row.words))
  }));
  downloadResult(JSON.stringify(rows, null, 2), "application/json;charset=utf-8", `${state.name}.seglst.json`);
};
function downloadResult(content, type, name) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([content], { type }));
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
$("#exportTextButton").onclick = () => {
  if (!isTextResult()) return;
  flushHistory();
  downloadResult(plainText(), "text/plain;charset=utf-8", `${state.name}.txt`);
};
async function submitTranscription() {
  if (!state.audioFile) return $("#audioInput").click();
  const button = $("#transcribeButton");
  button.disabled = true;
  $("#micButton").disabled = true;
  $("#modeSelect").disabled = true;
  $("#speakerCountSelect").disabled = true;
  $("#useVoiceprints").disabled = true;
  button.textContent = "正在提交";
  try {
    const mode = $("#modeSelect").value;
    const modeError = selectedModeError(mode);
    if (modeError) throw new Error(modeError);
    const clientStarted = performance.now();
    const body = new FormData();
    body.append("audio", state.audioFile);
    body.append("mode", mode);
    body.append("speaker_count", ["main", "review"].includes(mode) ? $("#speakerCountSelect").value : "0");
    body.append("voiceprint_ids", ["main", "review"].includes(mode) && $("#useVoiceprints").checked ? [...state.selectedVoiceprints].join(",") : "");
    const job = await responseJson(await fetch("/api/jobs", { method: "POST", body }));
    const acceptedAt = performance.now();
    state.jobId = job.id;
    showJob(job);
    button.textContent = "识别中";
    setServiceState("busy", "本地 GPU 忙碌");
    await pollJob(job.id, clientStarted, acceptedAt);
  } catch (error) {
    const message = error.message === "Failed to fetch" ? "算法服务尚未启动" : error.message;
    const panel = $("#jobProgress");
    panel.hidden = false;
    panel.classList.add("failed");
    $("#jobStage").textContent = "识别没有完成";
    $("#jobDetail").textContent = message;
    toast(message);
  }
  finally {
    state.jobId = null;
    button.disabled = false;
    $("#micButton").disabled = false;
    $("#modeSelect").disabled = false;
    updateModeControls();
    button.textContent = "开始识别";
    await checkBackend();
  }
}
$("#transcribeButton").onclick = submitTranscription;
$("#modeSelect").onchange = () => {
  updateModeControls();
  saveSettings();
  void requestWarmup();
};
$("#speakerCountSelect").onchange = saveSettings;
$("#useVoiceprints").onchange = () => {
  if ($("#useVoiceprints").checked && !state.selectedVoiceprints.size) {
    $("#useVoiceprints").checked = false;
    $("#voiceprintDialog").showModal();
    toast("请先在声纹库选择参与识别的成员");
  }
  updateModeControls();
  saveSettings();
};
$("#voiceprintButton").onclick = async () => { await loadVoiceprints(); $("#voiceprintDialog").showModal(); };
$("#closeVoiceprintButton").onclick = () => {
  if (state.voiceprintRecording) return toast("请先结束正在进行的声纹录音");
  $("#voiceprintDialog").close();
};
$("#voiceprintDialog").onclick = event => {
  if (event.target !== $("#voiceprintDialog")) return;
  if (state.voiceprintRecording) return toast("请先结束正在进行的声纹录音");
  $("#voiceprintDialog").close();
};
$("#voiceprintRecordButton").onclick = () => state.voiceprintRecording ? stopVoiceprintRecording() : startVoiceprintRecording();
$("#selectAllVoiceprintsButton").onclick = () => {
  const allSelected = state.voiceprints.length && state.selectedVoiceprints.size === state.voiceprints.length;
  state.selectedVoiceprints = allSelected ? new Set() : new Set(state.voiceprints.map(profile => String(profile.id)));
  updateVoiceprintSelection(true);
};
$("#historyButton").onclick = () => { flushHistory(); renderHistory(); $("#historyDialog").showModal(); };
$("#closeHistoryButton").onclick = () => $("#historyDialog").close();
$("#historyDialog").onclick = event => { if (event.target === $("#historyDialog")) $("#historyDialog").close(); };
$("#clearHistoryButton").onclick = () => {
  if (!readHistory().length || !confirm("确定清空本机保存的全部识别历史吗？")) return;
  localStorage.removeItem(STORAGE_KEYS.history);
  state.historyId = null;
  renderHistory();
  toast("识别历史已清空");
};
$("#serviceState").onclick = () => {
  if (!state.health) return toast("本地服务未连接，请重新双击“打开声迹”");
  const main = state.health.main;
  if ($("#modeSelect").value === "text") {
    const error = selectedModeError();
    return toast(error || `服务正常 · 纯文字${state.health.text?.model_loaded ? "模型已加载" : "将在首次识别时加载模型"}`);
  }
  if (main?.warmup_error) return toast(`模型加载失败：${main.warmup_error}`);
  toast(`服务正常 · GPU ${state.health.busy ? "正在工作" : "空闲"} · 统一识别${main?.model_loaded ? "已就绪" : "首次使用时加载"}`);
};
window.onresize = drawSyntheticWave;

async function initialize() {
  loadSettings();
  renderHistory();
  render();
  const health = await checkBackend();
  if (health) {
    await loadVoiceprints();
    await requestWarmup();
  }
  setInterval(checkBackend, 4000);
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(error => console.warn("离线外壳未注册", error));
}

initialize();
