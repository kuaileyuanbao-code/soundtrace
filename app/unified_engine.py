"""Single-file Main service and optional MOSS/FireRed/MOVER review.

No competition references, batch speaker prototypes or historical predictions
are read here. The review route is a new product pipeline, not clean-v2 replay.
"""
from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import threading
import time
import uuid
import sys
from pathlib import Path
from performance import StageTimings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODELS = ROOT / "project/user_data/model_data"
SOURCE = ROOT / "project/code/test/packages/ifly_asr_dev017_verified_20260725/code"
PREFIX = "__SOUNDTRACE_MAIN__"


def engine_python():
    configured = os.environ.get("SOUNDTRACE_MAIN_PYTHON")
    if configured:
        return Path(configured)
    candidates = [ROOT / "project/user_data/tmp_data/envs/sortformer/Scripts/python.exe",
                  ROOT / "tmp/audit_repro_full_20260911/.venv_augmented/Scripts/python.exe"]
    return next((p for p in candidates if p.is_file()), candidates[-1])


def native_model_path(path):
    """Native SentencePiece on Windows needs an ASCII directory name."""
    if os.name != "nt" or str(path).isascii():
        return path
    link = Path(os.environ["LOCALAPPDATA"]) / "Soundtrace/runtime/models" / path.name
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists():
        if link.resolve() != path.resolve():
            raise RuntimeError(f"模型目录映射已指向其他文件：{link}")
        return link
    # Create a junction with native PowerShell; no model copies or deletion.
    literal = lambda value: "'" + str(value).replace("'", "''") + "'"
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
         f"New-Item -ItemType Junction -Path {literal(link)} -Target {literal(path)} -ErrorAction Stop | Out-Null"],
        capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode or not link.exists():
        raise RuntimeError("无法为模型建立兼容路径：" + result.stderr)
    return link


def missing_requirements(review=False, text_only=False):
    if text_only:
        paths = [engine_python(), ROOT / "app/main_worker.py", ROOT / "app/text_transcription.py",
                 MODELS / "Qwen3-ASR-1.7B/config.json"]
        return [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in paths if not p.exists()]
    paths = [engine_python(), ROOT / "app/main_worker.py", SOURCE / "src/ifly_tcpwer/global_proto.py",
             ROOT / "vendor/3D-Speaker/speakerlab", MODELS / "Qwen3-ASR-1.7B/config.json",
             MODELS / "Qwen3-ForcedAligner-0.6B/config.json",
             MODELS / "diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo",
             MODELS / "3dspeaker/iic/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"]
    if review:
        paths += [ROOT / ".venv_moss/Scripts/python.exe", ROOT / "models/MOSS-Transcribe-Diarize",
                  MODELS / "FireRedASR2-AED", ROOT / "packages/FireRedASR2S/fireredasr2s",
                  ROOT / "research/third_party/mover", ROOT / "scripts/run_mover_fusion.py"]
    return [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in paths if not p.exists()]


def validate_rows(rows, duration):
    cleaned = []
    seen = set()
    for item in sorted(rows, key=lambda r: (float(r["start_time"]), float(r["end_time"]), str(r["speaker"]))):
        s, e = float(item["start_time"]), float(item["end_time"])
        if not math.isfinite(s) or not math.isfinite(e) or s < 0 or e <= s or e > duration + 0.05:
            raise ValueError(f"无效的转写时间：{s}–{e}，音频长度 {duration}")
        e = min(e, duration)
        if e <= s:
            raise ValueError("转写片段位于音频结束位置，缺少有效时长。")
        words = " ".join(str(item["words"]).split())
        if not words:
            continue
        key = (str(item["speaker"]), round(s, 3), round(e, 3), words)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({**item, "start_time": round(s, 3), "end_time": round(min(e, duration), 3), "words": words})
    if not cleaned:
        raise ValueError("模型没有识别到有效语音。")
    return cleaned


def compact_rows(rows):
    """Group adjacent tokens into readable utterances without dropping words."""
    groups = {}
    output = []
    for row in sorted(rows, key=lambda r: (r["start_time"], r["end_time"], str(r["speaker"]))):
        key = (row["session_id"], str(row["speaker"]))
        current = groups.get(key)
        gap = row["start_time"] - current["end_time"] if current else 0
        if current and -0.05 <= gap <= 0.6 and row["end_time"] - current["start_time"] <= 4.5:
            current["words"] += " " + row["words"]
            current["end_time"] = max(current["end_time"], row["end_time"])
        else:
            current = dict(row)
            groups[key] = current
            output.append(current)
    return sorted(output, key=lambda r: (r["start_time"], r["end_time"], str(r["speaker"])))


def window_plan(duration, length, overlap):
    # Stable ownership at overlap midpoints, including a short final chunk.
    windows = []
    start = 0.0
    while start < duration:
        end = min(duration, start + length)
        windows.append((start, end))
        if end >= duration:
            break
        start += length - overlap
    return [(s, e, (windows[i-1][1] + s) / 2 if i else s,
             (e + windows[i+1][0]) / 2 if i+1 < len(windows) else e)
            for i, (s, e) in enumerate(windows)]


def owned_tokens(rows, offset, keep_start, keep_end, session_id):
    output = []
    for row in rows:
        words = str(row["words"]).split()
        if not words:
            continue
        s, e = offset + float(row["start_time"]), offset + float(row["end_time"])
        step = (e - s) / len(words)
        for i, word in enumerate(words):
            a, b = s + i * step, s + (i + 1) * step
            if b > a and keep_start <= (a + b) / 2 < keep_end:
                output.append({**row, "session_id": session_id, "start_time": a, "end_time": b, "words": word})
    return output


def map_chunk_speakers(rows, main_rows):
    """Map chunk-local MOSS identities to Main by total temporal support."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    local = sorted({str(r["speaker"]) for r in rows})
    global_ids = sorted({str(r["speaker"]) for r in main_rows})
    scores = np.zeros((len(local), len(global_ids)))
    for r in rows:
        for anchor in main_rows:
            overlap = max(0, min(r["end_time"], anchor["end_time"]) - max(r["start_time"], anchor["start_time"]))
            scores[local.index(str(r["speaker"])), global_ids.index(str(anchor["speaker"]))] += overlap
    mapping = {}
    if scores.size:
        left, right = linear_sum_assignment(-scores)
        mapping = {local[a]: global_ids[b] for a, b in zip(left, right) if scores[a, b] > 0}
    # Keep unsupported voices separate instead of silently merging them.
    for speaker in local:
        first = min(r["start_time"] for r in rows if str(r["speaker"]) == speaker)
        mapping.setdefault(speaker, f"moss_extra_{round(first * 1000)}_{speaker}")
    return [{**r, "speaker": mapping[str(r["speaker"])]} for r in rows]


def stop_worker_process(process):
    """Close the protocol first; terminate the venv redirector's tree on timeout."""
    if process is None or process.poll() is not None:
        return
    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            process.kill()
        process.wait(timeout=8)


class WarmupCancelled(RuntimeError):
    """An unused background preload was cancelled; never a recognition fallback."""


class MainEngine:
    def __init__(self):
        self.process = None
        self.ready = False
        self.messages = queue.Queue()
        self.lock = threading.RLock()
        self.startup_seconds = None
        self.model_info = {}

    def state(self):
        if not self.process or self.process.poll() is not None:
            return {"observed_at": time.time(), "models": {
                name: {"state": "unloaded", "devices": []}
                for name in ("qwen", "aligner", "sortformer", "campp")}}
        return {**self.model_info.get("model_state_after", {"models": {}, "state": "loading_or_unknown"}),
                "snapshot_source": "last_worker_tensor_observation"}

    def stop(self):
        with self.lock:
            process, self.process = self.process, None
            self.ready = False
            self.model_info = {}
            stop_worker_process(process)

    def _read(self, process, messages):
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    def _receive(self, callback, timeout, cancel_event=None):
        deadline = time.monotonic() + timeout
        while True:
            if cancel_event is not None and cancel_event.is_set():
                # This runs under the engine lock in the preload thread. Wait for
                # the old process tree to exit before releasing the GPU lock.
                self.stop()
                raise WarmupCancelled("后台预热已取消，将优先准备用户选择的模式。")
            try:
                remaining = max(0.01, deadline - time.monotonic())
                line = self.messages.get(timeout=min(.1, remaining) if cancel_event is not None else remaining)
            except queue.Empty:
                if cancel_event is not None and time.monotonic() < deadline:
                    continue
                self.stop()
                raise TimeoutError("语音模型长时间未响应，已释放进程；请重试。")
            if line is None:
                self.stop()
                raise RuntimeError("统一识别进程退出，请查看任务日志后重试。")
            if line.startswith(PREFIX):
                return json.loads(line[len(PREFIX):])
            callback({"type": "log", "message": line.strip()})

    def warmup(self, callback=lambda event: None, cancel_event=None, text_only=False):
        with self.lock:
            if self.ready and self.process and self.process.poll() is None:
                if not text_only and self.model_info.get("text_only", False):
                    missing = missing_requirements()
                    if missing:
                        raise RuntimeError("统一识别环境缺少：" + ", ".join(missing))
                    request_id = uuid.uuid4().hex
                    started = time.perf_counter()
                    self.process.stdin.write(json.dumps({"id": request_id, "command": "prepare_main"}) + "\n")
                    self.process.stdin.flush()
                    while True:
                        event = self._receive(callback, 300, cancel_event)
                        if event.get("type") == "warmup":
                            callback(event)
                        elif event.get("id") == request_id:
                            if event.get("type") == "error":
                                self.stop()
                                raise RuntimeError(event["error"])
                            self.model_info.update({k: v for k, v in event.items() if k not in {"id", "type"}})
                            self.model_info["upgrade_wait_seconds"] = time.perf_counter() - started
                            break
                return
            if cancel_event is not None and cancel_event.is_set():
                raise WarmupCancelled("后台预热已取消，将优先准备用户选择的模式。")
            missing = missing_requirements(text_only=True) if text_only else missing_requirements()
            if missing:
                raise RuntimeError("统一识别环境缺少：" + ", ".join(missing))
            self.stop()
            started = time.perf_counter()
            self.messages = queue.Queue()
            self.process = subprocess.Popen(
                [str(engine_python()), "-X", "utf8", str(ROOT / "app/main_worker.py")]
                + (["--text-only"] if text_only else []), cwd=ROOT,
                env={**os.environ, "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                     "MODELSCOPE_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                encoding="utf-8", errors="replace", text=True, bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            threading.Thread(target=self._read, args=(self.process, self.messages), daemon=True).start()
            deadline = time.monotonic() + 300
            while True:
                event = self._receive(callback, max(0.01, deadline - time.monotonic()), cancel_event)
                if cancel_event is not None and cancel_event.is_set():
                    self.stop()
                    raise WarmupCancelled("后台预热已取消，将优先准备用户选择的模式。")
                if event.get("type") == "warmup":
                    if "model_state_after" in event:
                        self.model_info["model_state_after"] = event["model_state_after"]
                    callback(event)
                    continue
                if event.get("type") != "ready":
                    self.stop()
                    raise RuntimeError(event.get("error", "统一模型加载失败"))
                break
            self.startup_seconds = round(time.perf_counter() - started, 3)
            self.model_info = {k: v for k, v in event.items() if k != "type"}
            self.ready = True

    def park(self, release_between_moves=False):
        """Release GPU memory before another process runs MOSS or enrollment."""
        with self.lock:
            if not self.ready or not self.process or self.process.poll() is not None:
                return
            request_id = uuid.uuid4().hex
            self.process.stdin.write(json.dumps({"id": request_id, "command": "park",
                                                 "release_between_moves": release_between_moves}) + "\n")
            self.process.stdin.flush()
            while True:
                event = self._receive(lambda event: None, 60)
                if event.get("id") == request_id:
                    if event.get("type") == "error":
                        raise RuntimeError(event["error"])
                    self.model_info["model_state_after"] = event["model_state_after"]
                    return event

    def transcribe(self, wav, expected_count=0, callback=lambda event: None):
        with self.lock:
            started = time.perf_counter()
            state_before_warmup = self.state()
            was_ready = self.ready and self.process and self.process.poll() is None
            self.warmup(callback)
            wait_seconds = round(time.perf_counter() - started, 3)
            request_id = uuid.uuid4().hex
            self.process.stdin.write(json.dumps({"id": request_id, "wav": str(wav), "speaker_count": expected_count}) + "\n")
            self.process.stdin.flush()
            while True:
                event = self._receive(callback, 900)
                if event.get("id") != request_id:
                    continue
                if event.get("type") == "progress":
                    callback(event)
                elif event.get("type") == "error":
                    raise RuntimeError(event["error"])
                elif event.get("type") == "result":
                    response_at = time.perf_counter()
                    validate_started = time.perf_counter()
                    event["rows"] = validate_rows(event["rows"], event["duration_seconds"])
                    event["timings"]["model_wait_seconds"] = wait_seconds
                    event["timings"]["main_validation_seconds"] = round(time.perf_counter() - validate_started, 6)
                    event["timings"]["main_engine_seconds"] = round(response_at - started, 6)
                    event["timings"]["main_ipc_and_dispatch_seconds"] = round(max(0,
                        response_at - started - wait_seconds - event["timings"].get("worker_request_seconds", 0)), 6)
                    event["model_state_before_warmup"] = state_before_warmup
                    event["load_observation"] = {"incurred_in_this_call": not bool(was_ready),
                        "startup_seconds": self.startup_seconds, "timings": self.model_info.get("load_timings", {}),
                        "profiling": self.model_info.get("load_profiling", {})}
                    self.model_info.update({k: event[k] for k in ("asr_batch_size", "gpu_resident") if k in event})
                    self.model_info["model_state_after"] = event["model_state_after"]
                    return event

    def transcribe_text(self, wav, callback=lambda event: None):
        with self.lock:
            started = time.perf_counter()
            before = self.state()
            was_ready = bool(self.ready and self.process and self.process.poll() is None)
            self.warmup(callback, text_only=True)
            wait_seconds = time.perf_counter() - started
            request_id = uuid.uuid4().hex
            self.process.stdin.write(json.dumps({"id": request_id, "command": "text", "wav": str(wav)}) + "\n")
            self.process.stdin.flush()
            while True:
                event = self._receive(callback, 900)
                if event.get("id") != request_id:
                    continue
                if event.get("type") == "progress":
                    callback(event)
                elif event.get("type") == "error":
                    raise RuntimeError(event["error"])
                elif event.get("type") == "result":
                    elapsed = time.perf_counter() - started
                    event["rows"] = validate_rows(event["rows"], event["duration_seconds"])
                    event["model_state_before_warmup"] = before
                    event["load_observation"] = {"incurred_in_this_call": not was_ready,
                        "startup_seconds": self.startup_seconds, "timings": self.model_info.get("load_timings", {}),
                        "profiling": self.model_info.get("load_profiling", {})}
                    event["timings"].update(model_wait_seconds=round(wait_seconds, 6),
                        text_engine_seconds=round(elapsed, 6), text_ipc_and_dispatch_seconds=round(max(0,
                        elapsed - wait_seconds - event["timings"]["worker_request_seconds"]), 6))
                    self.model_info["model_state_after"] = event["model_state_after"]
                    return event


def run_command(command, log):
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               encoding="utf-8", errors="replace", text=True,
                               env={**os.environ, "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"},
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for line in process.stdout:
        log(line.strip())
    if process.wait() != 0:
        raise RuntimeError(f"精细复核阶段失败：{Path(command[3]).name}")
    return round(time.perf_counter() - started, 6)


def review_recording(wav, work_dir, main_rows, moss_request, moss_stop, progress, log):
    import soundfile as sf
    from firered_raw_to_seglst import normalize_tokens
    from map_timestamped_asr_to_speaker_rows import choose_assignments

    started = time.perf_counter()
    profile = StageTimings()
    chunk_observations = []
    work_dir.mkdir(parents=True, exist_ok=True)
    def save_partial(stage):
        (work_dir / "performance.json").write_text(json.dumps({"stage": stage,
            "timings": profile.seconds(), "moss_chunks": chunk_observations},
            ensure_ascii=False, indent=2), encoding="utf-8")
    audio, sr = sf.read(wav, dtype="float32")
    duration = len(audio) / sr
    moss_rows = []
    moss_chunks = work_dir / "moss_chunks"
    moss_chunks.mkdir(exist_ok=True)
    plan = window_plan(duration, 60, 5)
    try:
        for i, (s, e, keep_s, keep_e) in enumerate(plan):
            progress(55 + int(15 * i / len(plan)), "moss", f"正在复核对话 {i+1}/{len(plan)}")
            chunk = moss_chunks / f"chunk_{i:05d}.wav"
            sf.write(chunk, audio[round(s*sr):round(e*sr)], sr)
            with profile.measure("moss_request_seconds"):
                payload = moss_request(chunk)
            chunk_observations.append({k: v for k, v in payload.items()
                                       if k in {"timings", "profiling", "model_state_before", "model_state_after", "startup_observation"}})
            save_partial("moss")
            chunk_rows = owned_tokens(payload["rows"], s, keep_s, keep_e, wav.stem)
            moss_rows.extend(map_chunk_speakers(chunk_rows, main_rows))
    finally:
        with profile.measure("moss_stop_seconds"):
            moss_stop()
        save_partial("moss_stopped")
    if not moss_rows:
        raise RuntimeError("MOSS复核没有产生有效文字，未将空结果用于融合。")
    progress(72, "firered", "正在用另一套识别模型核对文字")
    fire_dir = work_dir / "firered_chunks"
    fire_dir.mkdir(exist_ok=True)
    fire_plan = window_plan(duration, 30, 2)
    for i, (s, e, _, _) in enumerate(fire_plan):
        sf.write(fire_dir / f"chunk_{i:05d}.wav", audio[round(s*sr):round(e*sr)], sr)
    raw_path = work_dir / "firered_raw.json"
    fire_timing_path = work_dir / "firered_performance.json"
    fire_seconds = run_command([str(ROOT / ".venv/Scripts/python.exe"), "-X", "utf8",
                 str(ROOT / "scripts/infer_fireredasr2_aed.py"), "--wav-dir", str(fire_dir),
                 "--model-dir", str(native_model_path(MODELS / "FireRedASR2-AED")), "--output", str(raw_path),
                 "--timings-output", str(fire_timing_path)], log)
    profile.values["firered_process_seconds"] = fire_seconds
    raw = {r["session_id"]: r for r in json.loads(raw_path.read_text(encoding="utf-8"))}
    fire_tokens = []
    for i, (s, e, keep_s, keep_e) in enumerate(fire_plan):
        item = raw[f"chunk_{i:05d}"]
        chunk_rows = [{"speaker": "spk1", "words": " ".join(normalize_tokens(t["token"])),
                       "start_time": t["start_time"], "end_time": t["end_time"]} for t in item["timestamps"]]
        fire_tokens.extend(owned_tokens(chunk_rows, s, keep_s, keep_e, wav.stem))
    if not fire_tokens:
        raise RuntimeError("FireRed复核没有产生有效时间戳。")
    assignments = choose_assignments(main_rows, [{**r, "token": r["words"]} for r in fire_tokens], "hybrid", 1.0)
    fire_rows = [{**r, "speaker": main_rows[index]["speaker"]} for r, index in zip(fire_tokens, assignments)]
    progress(86, "fusion", "正在融合三路文字、时间和人物结果")
    paths = []
    for name, rows in [("moss", moss_rows), ("main", main_rows), ("firered", fire_rows)]:
        path = work_dir / f"{name}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        paths.append(path)
    output = work_dir / "fused.json"
    command = [str(ROOT / ".venv/Scripts/python.exe"), "-X", "utf8", str(ROOT / "scripts/run_mover_fusion.py")]
    for path in paths:
        command += ["--input", str(path)]
    command += ["--output", str(output), "--strategy", "fullset", "--tcdp-collar", "5"]
    mover_timing_path = work_dir / "mover_performance.json"
    command += ["--timings-output", str(mover_timing_path)]
    profile.values["mover_process_seconds"] = run_command(command, log)
    rows = compact_rows(validate_rows(json.loads(output.read_text(encoding="utf-8")), duration))
    elapsed = time.perf_counter() - started
    details = profile.seconds()
    details["review_other_seconds"] = round(max(0, elapsed - sum(profile.values.values())), 6)
    fire = json.loads(fire_timing_path.read_text(encoding="utf-8"))
    mover = json.loads(mover_timing_path.read_text(encoding="utf-8"))
    for key in ("moss_load_and_place_seconds", "moss_inference_seconds", "moss_other_seconds"):
        details[key] = round(sum(c.get("timings", {}).get(key, 0) for c in chunk_observations), 6)
    details["moss_startup_seconds"] = round(sum(c.get("startup_observation", {}).get("startup_seconds", 0)
                                                  for c in chunk_observations), 6)
    accounted_moss = sum(details[k] for k in ("moss_load_and_place_seconds", "moss_inference_seconds",
                                            "moss_other_seconds", "moss_startup_seconds"))
    details["moss_ipc_and_dispatch_seconds"] = round(max(0, details.get("moss_request_seconds", 0) - accounted_moss), 6)
    observation = {"timings": details, "moss_chunks": chunk_observations, "firered": fire, "mover": mover}
    (work_dir / "performance.json").write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, round(elapsed, 3), observation
