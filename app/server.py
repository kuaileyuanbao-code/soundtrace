from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import atexit
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from scipy.signal import resample_poly
from starlette.concurrency import run_in_threadpool


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
JOBS_DIR = APP_DIR / "jobs"
VOICEPRINTS_DIR = APP_DIR / "voiceprints"
VOICEPRINTS_FILE = VOICEPRINTS_DIR / "profiles.json"
SCRIPTS_DIR = ROOT / "scripts"
ACCURATE_SCRIPT = SCRIPTS_DIR / "run_preservealign_pipeline.py"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
ALLOWED_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}
sys.path.insert(0, str(SCRIPTS_DIR))

from unified_engine import MainEngine, WarmupCancelled, missing_requirements, review_recording, validate_rows, stop_worker_process  # noqa: E402
from performance import StageTimings, model_state
from model_lifecycle import GIB, fast_state, move_fast, tensor_bytes, memory_budget

OBSERVED_SOURCES = ("app/main_worker.py", "app/unified_engine.py", "app/server.py", "app/performance.py",
                    "app/text_transcription.py",
                    "app/moss_worker.py", "app/app.js", "app/model_lifecycle.py",
                    "packages/FireRedASR2S/fireredasr2s/__init__.py",
                    "packages/FireRedASR2S/fireredasr2s/fireredasr2/asr.py",
                    "scripts/infer_fireredasr2_aed.py", "scripts/run_mover_fusion.py")
SOURCE_SHA256 = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in OBSERVED_SOURCES}


def infer_file(*args, **kwargs):
    # FunASR imports a separate PyTorch/model stack. Main runs in its own
    # worker and must not pay this startup/memory cost for an unused mode.
    from infer_funasr import infer_file as fast_infer_file
    return fast_infer_file(*args, **kwargs)


app = FastAPI(title="声迹本地转写服务", version="3.0")
_main_engine = MainEngine()
_main_warmup_status = {"status": "idle", "error": None}
_main_warmup_cancel = None  # Only a background preload has a cancellation token.
_warmup_request_lock = threading.Lock()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_model_lock = threading.Lock()
_active_mode = None
_cache_events = deque(maxlen=32)
_cache_monitor_stop = threading.Event()
_speaker_model = None
_speaker_model_lock = threading.Lock()
_voiceprint_lock = threading.Lock()
_inference_lock = threading.Lock()
_job_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="soundtrace-gpu")
_moss_worker = None
_moss_worker_ready = False
_moss_model_state = model_state(None)
_moss_startup_observation = {}
_moss_worker_lock = threading.Lock()
_warmup_lock = threading.Lock()
_warmup_status = {"status": "idle", "error": None}
MOSS_WORKER_PREFIX = "__SOUNDTRACE_MOSS__"
_diarizen_worker = None
_diarizen_worker_ready = False
_diarizen_worker_lock = threading.Lock()
DIARIZEN_WORKER_PREFIX = "__SOUNDTRACE_DIARIZEN__"

ACCURATE_REQUIREMENTS = (
    ROOT / ".venv" / "Scripts" / "python.exe",
    ROOT / ".venv_moss" / "Scripts" / "python.exe",
    ROOT / ".venv_pyannote" / "Scripts" / "python.exe",
    ROOT / "models" / "MOSS-Transcribe-Diarize",
    ROOT / "models" / "diarizen-wavlm-large-s80-md-v2",
    ROOT / "models" / "speaker-embedding-pyannote" / "pytorch_model.bin",
    ACCURATE_SCRIPT,
    APP_DIR / "diarizen_worker.py",
)
CONVERSATION_REQUIREMENTS = (
    ROOT / ".venv_moss" / "Scripts" / "python.exe",
    ROOT / "models" / "MOSS-Transcribe-Diarize",
    SCRIPTS_DIR / "infer_moss_transcribe_diarize.py",
)

ACCURATE_STAGES = {
    "prepare_audio": (8, "正在整理音频"),
    "prepare_chunks": (15, "正在切分长录音"),
    "moss_inference": (24, "MOSS 正在转写并区分说话人"),
    "merge_chunks": (84, "正在合并分块结果"),
    "diarizen_inference": (64, "DiariZen 正在统一全局说话人"),
    "preservealign": (88, "正在对齐文字与全局说话人"),
    "validate_output": (95, "正在校验最终结果"),
}


def accurate_available() -> bool:
    return all(path.exists() for path in ACCURATE_REQUIREMENTS)


def conversation_available() -> bool:
    return all(path.exists() for path in CONVERSATION_REQUIREMENTS)


def safe_session_id(filename: str | None) -> str:
    stem = Path(filename or "audio").stem.strip() or "audio"
    safe = "".join(
        character
        if character.isascii() and (character.isalnum() or character in "-_")
        else "_"
        for character in stem
    )
    return safe[:80] or "audio"


def load_model():
    global _model
    with _model_lock:
        if _model is None:
            from funasr import AutoModel

            _model = AutoModel(
                model="iic/SenseVoiceSmall",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 12000},
                spk_model="cam++",
                device=os.environ.get("SOUNDTRACE_DEVICE", "cuda:0"),
                disable_update=True,
            )
        else:
            try:
                move_fast(_model, _model.kwargs["device"])
            except Exception:
                # A partial GPU restore must not leave hidden GPU owners.
                try:
                    move_fast(_model, "cpu")
                except Exception:
                    pass
                _model = None
                gc.collect()
                import torch
                torch.cuda.empty_cache()
                raise
    return _model


def cache_note(action, reason, budget=None):
    event = {"at": time.time(), "action": action, "reason": reason, "memory": budget or memory_budget()}
    _cache_events.append(event)
    print("Model cache: " + json.dumps(event, ensure_ascii=False), flush=True)


def park_fast_model():
    with _model_lock:
        if _model is None:
            return
        before = fast_state(_model)
        if not any(d.startswith("cuda") for s in before.values() for d in s["devices"]):
            return
        move_fast(_model, "cpu")
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    cache_note("fast_park", "Main needs GPU; retained existing Fast weights on CPU")


def discard_inactive_cache():
    """Release only parked CPU owners; safe even while the other owner is loading.

    Recheck the active mode under the owner lock. Never block the monitor on an
    inference/constructor, and never move or evict an active GPU model here.
    """
    global _model
    budget = memory_budget()
    if budget["can_retain"]:
        return
    if _active_mode == "fast" and _main_engine.lock.acquire(blocking=False):
        try:
            states = _main_engine.state().get("models", {})
            if (_active_mode == "fast" and _main_engine.ready and states
                    and all(s.get("state") in {"cpu_loaded", "unloaded"} for s in states.values())):
                _main_engine.stop()
                _main_warmup_status.update(status="idle", error=None)
                cache_note("main_release", "memory reserve or reuse disabled", budget)
        finally:
            _main_engine.lock.release()
    elif _active_mode in {"main", "review", "text"} and _model_lock.acquire(blocking=False):
        released = False
        try:
            states = fast_state(_model)
            if (_active_mode in {"main", "review", "text"} and _model is not None
                    and all(d == "cpu" for s in states.values() for d in s["devices"])):
                _model = None
                released = True
        finally:
            _model_lock.release()
        if released:
            gc.collect()
            cache_note("fast_release", "memory reserve or reuse disabled", budget)


def prepare_fast_mode():
    global _active_mode
    stop_moss_worker()
    stop_diarizen_worker()
    if _main_engine.ready:
        models = _main_engine.state().get("models", {})
        on_gpu = any(d.startswith("cuda") for s in models.values() for d in s.get("devices", []))
        copy_bytes = _main_engine.model_info.get("cpu_cache_park_peak_bytes", 4 * GIB) if on_gpu else 0
        budget = memory_budget(copy_bytes)
        if budget["can_retain"]:
            try:
                if on_gpu:
                    _main_engine.park(release_between_moves=True)
                    cache_note("main_park", "Fast needs GPU; retained existing Main process and CPU weights", budget)
                # Observe actual post-park headroom. The monitor can release this
                # inactive CPU owner while Fast imports / loads if pressure grows.
                budget = memory_budget()
                if not budget["can_retain"]:
                    _main_engine.stop()
                    _main_warmup_status.update(status="idle", error=None)
                    cache_note("main_release", "insufficient headroom after CPU parking", budget)
            except Exception as exc:
                _main_engine.stop()
                _main_warmup_status.update(status="idle", error=None)
                cache_note("main_release", "CPU parking failed: " + str(exc))
        else:
            _main_engine.stop()
            _main_warmup_status.update(status="idle", error=None)
            cache_note("main_release", "insufficient headroom for CPU parking / Fast startup", budget)
    _active_mode = "fast"


def monitor_model_cache():
    while not _cache_monitor_stop.wait(5):
        try:
            discard_inactive_cache()
        except Exception as exc:
            cache_note("cache_monitor_error", str(exc))


def load_speaker_model():
    """Load the lightweight CAM++ voiceprint model without the ASR stack."""
    global _speaker_model
    if _speaker_model is not None:
        return _speaker_model
    with _speaker_model_lock:
        if _speaker_model is None:
            from funasr import AutoModel

            _speaker_model = AutoModel(
                model="cam++",
                device=os.environ.get("SOUNDTRACE_DEVICE", "cuda:0"),
                disable_update=True,
            )
    return _speaker_model


def read_voiceprints() -> list[dict]:
    with _voiceprint_lock:
        if not VOICEPRINTS_FILE.exists():
            return []
        try:
            payload = json.loads(VOICEPRINTS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
    profiles = payload.get("profiles", []) if isinstance(payload, dict) else []
    return [profile for profile in profiles if isinstance(profile, dict)]


def write_voiceprints(profiles: list[dict]) -> None:
    VOICEPRINTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"version": 1, "profiles": profiles},
        ensure_ascii=False,
        indent=2,
    )
    with _voiceprint_lock:
        temporary = VOICEPRINTS_DIR / f"profiles.{uuid.uuid4().hex}.tmp"
        temporary.write_text(payload, encoding="utf-8")
        try:
            os.replace(temporary, VOICEPRINTS_FILE)
        except OSError:
            VOICEPRINTS_FILE.write_text(payload, encoding="utf-8")
            temporary.unlink(missing_ok=True)


def public_voiceprint(profile: dict) -> dict:
    return {
        "id": str(profile["id"]),
        "name": str(profile["name"]),
        "sample_count": int(profile.get("sample_count", 0)),
        "duration_seconds": round(float(profile.get("duration_seconds", 0)), 1),
        "quality": round(float(profile.get("quality", 0)), 2),
        "created_at": float(profile.get("created_at", 0)),
    }


def unload_fast_model() -> None:
    global _model
    with _model_lock:
        if _model is None:
            return
        _model = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def convert_to_wav(source: Path, target: Path) -> None:
    try:
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(
            "无法读取该音频。请选择 WAV、MP3、FLAC 或 OGG 文件。"
        ) from exc
    if not sample_rate or audio.size == 0:
        raise ValueError("音频为空或采样率无效。")
    mono = np.mean(audio, axis=1)
    if sample_rate != 16000:
        divisor = int(np.gcd(sample_rate, 16000))
        mono = resample_poly(mono, 16000 // divisor, sample_rate // divisor)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, mono, 16000, subtype="PCM_16")


def normalize_speakers(rows: list[dict]) -> list[dict]:
    speaker_map: dict[str, str] = {}
    for row in rows:
        speaker = str(row["speaker"])
        if speaker not in speaker_map:
            speaker_map[speaker] = f"spk{len(speaker_map) + 1}"
        row["speaker"] = speaker_map[speaker]
    return rows


def transcribe_fast(source: Path, session_id: str, work_dir: Path) -> list[dict]:
    wav_path = work_dir / f"{session_id}.wav"
    convert_to_wav(source, wav_path)
    cancel_main_preload()
    with _inference_lock:
        prepare_fast_mode()
        model = load_model()
        discard_inactive_cache()
        rows = infer_file(model, wav_path, split_max_ms=12000, split_gap_ms=500)
        discard_inactive_cache()
    if not rows:
        raise RuntimeError("模型没有识别到有效语音。")
    return normalize_speakers(rows)


def update_job(job_id: str, **changes) -> None:
    with _job_lock:
        job = _jobs.get(job_id)
        if job is not None:
            if "progress" in changes:
                changes["progress"] = max(job.get("progress", 0), changes["progress"])
            job.update(changes)
            job["updated_at"] = time.time()


def append_job_log(job_id: str, line: str) -> None:
    clean = line.strip()
    if not clean:
        return
    with _job_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["log"] = [*job.get("log", []), clean][-24:]
            job["updated_at"] = time.time()


def public_job(job: dict) -> dict:
    result = {key: value for key, value in job.items() if key not in {"job_dir", "source", "result_path"}
              and not key.startswith("_")}
    if result.get("status") == "completed":
        result["result_url"] = f"/api/jobs/{job['id']}/result"
    return result


def run_pipeline_process(job_id: str, wav_path: Path, work_dir: Path, *extra: str) -> None:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-X",
        "utf8",
        str(ACCURATE_SCRIPT),
        "--input-wav",
        str(wav_path),
        "--work-dir",
        str(work_dir),
        "--device",
        "cuda",
        *extra,
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        append_job_log(job_id, line)
        if line.startswith("[") and "]" in line:
            stage = line[1 : line.index("]")]
            if stage in ACCURATE_STAGES and "already completed" not in line:
                progress, message = ACCURATE_STAGES[stage]
                update_job(job_id, stage=stage, progress=progress, message=message)
    return_code = process.wait()
    if return_code:
        with _job_lock:
            log = list(_jobs.get(job_id, {}).get("log", []))
        detail = "\n".join(log[-8:]) or f"流水线退出码 {return_code}"
        raise RuntimeError(detail)


def write_json_retry(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(10):
        try:
            path.write_text(payload, encoding="utf-8")
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.12 * (attempt + 1))


def precompute_accurate_moss(job_id: str, work_dir: Path) -> None:
    chunks_dir = work_dir / "02_chunks"
    chunks = sorted(chunks_dir.glob("*.wav"))
    if not chunks:
        raise RuntimeError("长录音切分完成，但没有生成可识别的音频块。")
    moss_output = work_dir / "03_moss" / "chunks.seglst.json"
    metadata_output = moss_output.with_suffix(".metadata.json")
    rows: list[dict] = []
    metadata: list[dict] = []
    started = time.perf_counter()
    for index, chunk in enumerate(chunks, start=1):
        progress = 20 + round(34 * (index - 1) / max(1, len(chunks)))
        update_job(
            job_id,
            progress=progress,
            stage="moss_inference",
            message=f"MOSS 正在识别长录音分块 {index}/{len(chunks)}",
        )
        payload = request_moss_transcription(job_id, chunk, 2048)
        chunk_rows = payload.get("rows") or []
        rows.extend(chunk_rows)
        metadata.append(
            {
                "session_id": chunk.stem,
                "segments": len(chunk_rows),
                "elapsed_sec": payload.get("elapsed_seconds", 0),
                "generated_tokens": payload.get("generated_tokens", 0),
                "generation_budget": 2048,
                "hit_token_limit": False,
                "retried_with_silence_padding": False,
                "raw_transcript": payload.get("raw_transcript", ""),
            }
        )
        write_json_retry(moss_output, rows)
        write_json_retry(metadata_output, metadata)
    if not rows:
        raise RuntimeError("MOSS 没有从长录音中识别到有效语音。")

    status_path = work_dir / "pipeline_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["stages"]["moss_inference"] = {
        "state": "completed",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "command": ["persistent-moss-worker"],
        "outputs": [str(moss_output), str(metadata_output)],
    }
    write_json_retry(status_path, status)
    append_job_log(job_id, f"Reused warm MOSS for {len(chunks)} long-audio chunks.")


def precompute_accurate_diarizen(job_id: str, work_dir: Path) -> None:
    mono_wavs = sorted((work_dir / "01_audio").glob("*.wav"))
    if len(mono_wavs) != 1:
        raise RuntimeError("长录音全局分人阶段没有找到唯一的单声道音频。")
    wav_path = mono_wavs[0]
    diarizen_dir = work_dir / "04_diarizen"
    rttm_path = diarizen_dir / "rttm" / f"{wav_path.stem}.rttm"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    started = time.perf_counter()
    update_job(job_id, progress=64, stage="diarizen_inference", message="DiariZen 正在统一全局说话人")
    try:
        payload = request_diarizen(job_id, wav_path, rttm_path)
    except RuntimeError as exc:
        if "memory" not in str(exc).lower() and "cuda" not in str(exc).lower():
            raise
        append_job_log(job_id, "Retrying DiariZen after releasing the MOSS worker.")
        stop_diarizen_worker()
        stop_moss_worker()
        payload = request_diarizen(job_id, wav_path, rttm_path)

    metadata_path = diarizen_dir / "metadata" / f"{wav_path.stem}.json"
    write_json_retry(
        metadata_path,
        {
            "session_id": wav_path.stem,
            "speaker_count": payload.get("speaker_count", 0),
            "segments": payload.get("segments", 0),
            "elapsed_seconds": payload.get("elapsed_seconds", 0),
            "min_speakers": 1,
            "max_speakers": 6,
            "batch_size": 2,
            "persistent_worker": True,
        },
    )
    write_json_retry(diarizen_dir / "failures.json", [])

    status_path = work_dir / "pipeline_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["stages"]["diarizen_inference"] = {
        "state": "completed",
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "command": ["persistent-diarizen-worker", "--batch-size", "2"],
        "outputs": [str(rttm_path)],
    }
    write_json_retry(status_path, status)
    append_job_log(
        job_id,
        f"DiariZen inference: {payload.get('elapsed_seconds', 0):.1f}s; model kept ready.",
    )


def run_accurate_job(job_id: str, wav_path: Path, work_dir: Path) -> list[dict]:
    if not accurate_available():
        raise RuntimeError("高精度流水线所需的模型或 Python 环境不完整。")

    unload_fast_model()
    update_job(job_id, progress=8, stage="prepare_audio", message="正在整理并切分长录音")
    run_pipeline_process(job_id, wav_path, work_dir, "--stop-after", "prepare_chunks")
    precompute_accurate_moss(job_id, work_dir)
    precompute_accurate_diarizen(job_id, work_dir)
    update_job(job_id, progress=84, stage="merge_chunks", message="正在合并并对齐全局说话人")
    run_pipeline_process(job_id, wav_path, work_dir)

    result_path = work_dir / "06_final" / "speaker_transcript.seglst.json"
    if not result_path.exists():
        raise RuntimeError("高精度流水线结束，但没有生成最终 SegLST。")
    rows = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("高精度流水线返回了空结果。")
    return normalize_speakers(rows)


def extract_segment_embeddings(
    rows: list[dict],
    wav_path: Path,
) -> tuple[list[dict], np.ndarray, list[float]]:
    ordered = sorted(rows, key=lambda row: (float(row["start_time"]), float(row["end_time"])))
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1)
    clips: list[np.ndarray] = []
    durations: list[float] = []
    minimum_samples = int(sample_rate * 0.8)
    for row in ordered:
        start = max(0, int(float(row["start_time"]) * sample_rate))
        end = min(len(mono), int(float(row["end_time"]) * sample_rate))
        clip = mono[start:end]
        durations.append(max(0.0, (end - start) / sample_rate))
        if len(clip) < minimum_samples:
            clip = np.pad(clip, (0, minimum_samples - len(clip)))
        clips.append(np.asarray(clip, dtype=np.float32))

    result = load_speaker_model().generate(input=clips, batch_size=len(clips))
    if not result or "spk_embedding" not in result[0]:
        raise RuntimeError("CAM++ 没有返回声纹向量。")
    embeddings = result[0]["spk_embedding"]
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]
    if len(embeddings) != len(ordered):
        raise RuntimeError("CAM++ 返回的声纹数量与语音片段不一致。")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)
    return ordered, embeddings, durations


def cluster_voice_embeddings(
    embeddings: np.ndarray,
    durations: list[float],
    expected_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(embeddings) < expected_count:
        raise ValueError("有效语音片段少于说话人数。")

    anchor_indices = [index for index, duration in enumerate(durations) if duration >= 1.0]
    if len(anchor_indices) < expected_count:
        anchor_indices = list(range(len(embeddings)))
    if len(anchor_indices) < expected_count:
        raise ValueError("可用于分人的语音片段不足。")

    from sklearn.cluster import AgglomerativeClustering

    anchor_embeddings = embeddings[anchor_indices]
    anchor_labels = AgglomerativeClustering(
        n_clusters=expected_count,
        metric="cosine",
        linkage="average",
    ).fit_predict(anchor_embeddings)
    centroids = []
    for label in range(expected_count):
        centroid = np.mean(anchor_embeddings[anchor_labels == label], axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
        centroids.append(centroid)
    centroids_array = np.stack(centroids)
    labels = np.argmax(embeddings @ centroids_array.T, axis=1)
    return labels, centroids_array


def apply_speaker_count_constraint(
    rows: list[dict],
    expected_count: int,
    wav_path: Path,
) -> list[dict]:
    """Cluster utterances by CAM++ voiceprints when the speaker count is known."""
    if expected_count <= 0 or not rows:
        return rows
    ordered = sorted(rows, key=lambda row: (float(row["start_time"]), float(row["end_time"])))
    if expected_count == 1:
        for row in ordered:
            row["speaker"] = "voice1"
        return ordered
    try:
        ordered, embeddings, durations = extract_segment_embeddings(ordered, wav_path)
        labels, _ = cluster_voice_embeddings(embeddings, durations, expected_count)
    except (RuntimeError, ValueError):
        return ordered

    speaker_names: dict[int, str] = {}
    for row, label in zip(ordered, labels):
        numeric_label = int(label)
        if numeric_label not in speaker_names:
            speaker_names[numeric_label] = f"voice{len(speaker_names) + 1}"
        row["speaker"] = speaker_names[numeric_label]
    return ordered


def match_rows_to_voiceprints(
    rows: list[dict],
    wav_path: Path,
    profiles: list[dict],
) -> list[dict]:
    """Cluster a conversation, then map each cluster one-to-one to enrolled people."""
    if not profiles:
        return rows
    ordered, embeddings, durations = extract_segment_embeddings(rows, wav_path)
    labels, conversation_centroids = cluster_voice_embeddings(
        embeddings,
        durations,
        len(profiles),
    )
    enrolled = np.asarray([profile["centroid"] for profile in profiles], dtype=np.float32)
    enrolled /= np.maximum(np.linalg.norm(enrolled, axis=1, keepdims=True), 1e-8)

    from scipy.optimize import linear_sum_assignment

    similarities = conversation_centroids @ enrolled.T
    cluster_indices, profile_indices = linear_sum_assignment(-similarities)
    mapping = {
        int(cluster): str(profiles[int(profile)]["name"])
        for cluster, profile in zip(cluster_indices, profile_indices)
    }
    for row, label in zip(ordered, labels):
        row["speaker"] = mapping.get(int(label), f"未知说话人{int(label) + 1}")
    return ordered


def enroll_voiceprint(name: str, source: Path, work_dir: Path) -> dict:
    wav_path = work_dir / "enrollment.wav"
    convert_to_wav(source, wav_path)
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1)
    duration = len(mono) / sample_rate
    if duration < 8:
        raise ValueError("声纹录音至少需要 8 秒，请连续说几句完整的话。")
    if duration > 180:
        raise ValueError("单次声纹录音请控制在 3 分钟以内。")

    window = int(sample_rate * 3.0)
    hop = int(sample_rate * 2.0)
    clips = []
    energies = []
    for start in range(0, max(1, len(mono) - int(sample_rate)), hop):
        clip = mono[start : min(len(mono), start + window)]
        if len(clip) < sample_rate:
            continue
        if len(clip) < window:
            clip = np.pad(clip, (0, window - len(clip)))
        clips.append(np.asarray(clip, dtype=np.float32))
        energies.append(float(np.sqrt(np.mean(np.square(clip)) + 1e-12)))
    if len(clips) < 3:
        raise ValueError("有效声纹片段太少，请录制更长一些。")
    energy_floor = max(0.0015, float(np.percentile(energies, 30)))
    selected = [clip for clip, energy in zip(clips, energies) if energy >= energy_floor]
    if len(selected) < 3:
        selected = [clips[index] for index in np.argsort(energies)[-3:]]
    selected = selected[:40]

    result = load_speaker_model().generate(input=selected, batch_size=len(selected))
    embeddings = result[0].get("spk_embedding") if result else None
    if embeddings is None:
        raise RuntimeError("CAM++ 没有提取到有效声纹。")
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    first_center = np.mean(embeddings, axis=0)
    first_center /= max(float(np.linalg.norm(first_center)), 1e-8)
    similarities = embeddings @ first_center
    keep_threshold = max(0.2, float(np.percentile(similarities, 20)))
    kept = embeddings[similarities >= keep_threshold]
    if len(kept) < 3:
        kept = embeddings
    centroid = np.mean(kept, axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
    quality = float(np.median(kept @ centroid))

    profile = {
        "id": uuid.uuid4().hex,
        "name": name,
        "centroid": [round(float(value), 7) for value in centroid],
        "sample_count": len(kept),
        "duration_seconds": round(duration, 1),
        "quality": round(quality, 3),
        "created_at": time.time(),
    }
    profiles = read_voiceprints()
    if any(str(item.get("name", "")).casefold() == name.casefold() for item in profiles):
        raise ValueError(f"声纹库中已经有“{name}”，请换一个名称或先删除旧档案。")
    profiles.append(profile)
    write_voiceprints(profiles)
    return public_voiceprint(profile)


def stop_moss_worker() -> None:
    global _moss_worker, _moss_worker_ready, _moss_model_state
    with _moss_worker_lock:
        process = _moss_worker
        _moss_worker = None
        _moss_worker_ready = False
        _warmup_status.update(status="idle", error=None)
        stop_worker_process(process)
        _moss_model_state = model_state(None)


def ensure_moss_worker(job_id: str | None):
    global _moss_worker, _moss_worker_ready, _moss_model_state, _moss_startup_observation
    if _moss_worker is not None and _moss_worker.poll() is None and _moss_worker_ready:
        return _moss_worker
    _moss_worker_ready = False
    update_job(job_id, progress=34, stage="model", message="首次加载 MOSS 对话模型")
    started = time.perf_counter()
    _moss_worker = subprocess.Popen(
        [
            str(ROOT / ".venv_moss" / "Scripts" / "python.exe"),
            str(APP_DIR / "moss_worker.py"),
            "--model",
            str(ROOT / "models" / "MOSS-Transcribe-Diarize"),
            "--device",
            "cuda",
            "--dtype",
            "bfloat16",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert _moss_worker.stdout is not None
    while True:
        line = _moss_worker.stdout.readline()
        if not line:
            raise RuntimeError("MOSS 常驻进程在模型加载时退出。")
        if line.startswith(MOSS_WORKER_PREFIX):
            payload = json.loads(line[len(MOSS_WORKER_PREFIX) :])
            if payload.get("type") == "ready":
                _moss_worker_ready = True
                _moss_model_state = payload.get("model_state_after", {"state": "unknown", "devices": []})
                _moss_startup_observation = {"startup_seconds": round(time.perf_counter() - started, 6),
                                           "worker_boot_seconds": payload.get("worker_boot_seconds")}
                _warmup_status.update(status="ready", error=None)
                append_job_log(job_id, "MOSS model ready; later recordings will reuse it.")
                return _moss_worker
        elif "Loading weights" not in line:
            append_job_log(job_id, line)


def request_moss_transcription(
    job_id: str,
    wav_path: Path,
    max_new_tokens: int = 2048,
) -> dict:
    global _moss_model_state
    request_id = uuid.uuid4().hex
    with _moss_worker_lock:
        was_ready = bool(_moss_worker and _moss_worker.poll() is None and _moss_worker_ready)
        process = ensure_moss_worker(job_id)
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(
            json.dumps(
                {
                    "id": request_id,
                    "wav": str(wav_path),
                    "max_new_tokens": max_new_tokens,
                }
            )
            + "\n"
        )
        process.stdin.flush()
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("MOSS 常驻进程在识别时退出。")
            if not line.startswith(MOSS_WORKER_PREFIX):
                append_job_log(job_id, line)
                continue
            message = json.loads(line[len(MOSS_WORKER_PREFIX) :])
            if message.get("id") != request_id:
                continue
            if message.get("type") == "error":
                raise RuntimeError(message.get("error") or "MOSS 识别失败。")
            if message.get("type") == "result":
                _moss_model_state = message.get("model_state_after", {"state": "unknown", "devices": []})
                message["startup_observation"] = _moss_startup_observation if not was_ready else {"startup_seconds": 0}
                return message


def stop_diarizen_worker() -> None:
    global _diarizen_worker, _diarizen_worker_ready
    with _diarizen_worker_lock:
        process = _diarizen_worker
        _diarizen_worker = None
        _diarizen_worker_ready = False
        stop_worker_process(process)


def ensure_diarizen_worker(job_id: str):
    global _diarizen_worker, _diarizen_worker_ready
    if (
        _diarizen_worker is not None
        and _diarizen_worker.poll() is None
        and _diarizen_worker_ready
    ):
        return _diarizen_worker
    _diarizen_worker_ready = False
    update_job(job_id, progress=62, stage="diarizen_model", message="首次加载全局说话人模型")
    _diarizen_worker = subprocess.Popen(
        [
            str(ROOT / ".venv_pyannote" / "Scripts" / "python.exe"),
            str(APP_DIR / "diarizen_worker.py"),
            "--diarizen-model",
            str(ROOT / "models" / "diarizen-wavlm-large-s80-md-v2"),
            "--embedding-model",
            str(ROOT / "models" / "speaker-embedding-pyannote" / "pytorch_model.bin"),
            "--batch-size",
            "2",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert _diarizen_worker.stdout is not None
    while True:
        line = _diarizen_worker.stdout.readline()
        if not line:
            raise RuntimeError("DiariZen 常驻进程在模型加载时退出。")
        if line.startswith(DIARIZEN_WORKER_PREFIX):
            payload = json.loads(line[len(DIARIZEN_WORKER_PREFIX) :])
            if payload.get("type") == "ready":
                _diarizen_worker_ready = True
                append_job_log(job_id, "DiariZen model ready; later long recordings will reuse it.")
                return _diarizen_worker
        else:
            append_job_log(job_id, line)


def request_diarizen(
    job_id: str,
    wav_path: Path,
    output_path: Path,
    min_speakers: int = 1,
    max_speakers: int = 6,
) -> dict:
    request_id = uuid.uuid4().hex
    with _diarizen_worker_lock:
        process = ensure_diarizen_worker(job_id)
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(
            json.dumps(
                {
                    "id": request_id,
                    "wav": str(wav_path),
                    "output": str(output_path),
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers,
                }
            )
            + "\n"
        )
        process.stdin.flush()
        while True:
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("DiariZen 常驻进程在识别时退出。")
            if not line.startswith(DIARIZEN_WORKER_PREFIX):
                append_job_log(job_id, line)
                continue
            message = json.loads(line[len(DIARIZEN_WORKER_PREFIX) :])
            if message.get("id") != request_id:
                continue
            if message.get("type") == "error":
                raise RuntimeError(message.get("error") or "DiariZen 分人失败。")
            if message.get("type") == "result":
                return message


def warmup_conversation_model() -> None:
    with _warmup_lock:
        if _moss_worker is not None and _moss_worker.poll() is None and _moss_worker_ready:
            _warmup_status.update(status="ready", error=None)
            return
        _warmup_status.update(status="loading", error=None)
        try:
            with _inference_lock:
                unload_fast_model()
                with _moss_worker_lock:
                    ensure_moss_worker(None)
            _warmup_status.update(status="ready", error=None)
        except Exception as exc:
            _warmup_status.update(status="failed", error=str(exc))


def run_conversation_job(
    job_id: str,
    wav_path: Path,
    work_dir: Path,
    expected_speakers: int = 0,
    voiceprint_ids: list[str] | None = None,
) -> list[dict]:
    if not conversation_available():
        raise RuntimeError("短对话分人所需的 MOSS 环境尚未准备完整。")
    duration = sf.info(wav_path).duration
    if duration > 65:
        raise ValueError("短对话分人适用于 60 秒内录音；更长音频请选择高精度模式。")

    unload_fast_model()
    update_job(job_id, progress=55, stage="transcribe", message="MOSS 正在识别文字与说话人")
    payload = request_moss_transcription(job_id, wav_path, 2048)
    rows = payload.get("rows") if payload else None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("模型没有识别到有效语音。")
    append_job_log(job_id, f"MOSS inference: {payload.get('elapsed_seconds', 0):.1f}s")
    selected_profiles: list[dict] = []
    if voiceprint_ids:
        profile_map = {str(profile.get("id")): profile for profile in read_voiceprints()}
        selected_profiles = [profile_map[profile_id] for profile_id in voiceprint_ids if profile_id in profile_map]
        if len(selected_profiles) != len(voiceprint_ids):
            raise ValueError("所选声纹档案已被删除，请重新选择。")
        update_job(job_id, progress=82, stage="voiceprint", message="CAM++ 正在匹配声纹库成员")
        rows = match_rows_to_voiceprints(rows, wav_path, selected_profiles)
        append_job_log(job_id, f"Matched enrolled voiceprints: {', '.join(profile['name'] for profile in selected_profiles)}")
    elif expected_speakers:
        update_job(job_id, progress=82, stage="voiceprint", message="CAM++ 正在按声纹校正说话人")
        try:
            rows = apply_speaker_count_constraint(rows, expected_speakers, wav_path)
            append_job_log(job_id, f"CAM++ voiceprint clustering: {expected_speakers} speakers")
        except Exception as exc:
            append_job_log(job_id, f"CAM++ clustering skipped; kept MOSS labels: {exc}")
    return rows if selected_profiles else normalize_speakers(rows)


def release_local_models(preserve_fast=False) -> None:
    global _speaker_model
    if preserve_fast and _model is not None:
        states = fast_state(_model)
        # Check the immediate CPU copy; monitor later active-model loading using
        # real memory readings instead of assuming every startup peaks together.
        required = max((tensor_bytes([getattr(_model, name)]) for name, s in states.items()
                        if any(d.startswith("cuda") for d in s["devices"])), default=0)
        budget = memory_budget(required)
        if budget["can_retain"]:
            try:
                park_fast_model()
                if not memory_budget()["can_retain"]:
                    unload_fast_model()
                    cache_note("fast_release", "insufficient headroom after CPU parking")
            except Exception as exc:
                unload_fast_model()
                cache_note("fast_release", "CPU parking failed: " + str(exc))
        else:
            unload_fast_model()
            cache_note("fast_release", "insufficient headroom for Main startup / restore", budget)
    else:
        unload_fast_model()
    with _speaker_model_lock:
        if _speaker_model is None:
            return
        _speaker_model = None
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def run_unified_job(job_id, wav_path, work_dir, mode, expected_speakers, voiceprint_ids):
    global _active_mode
    profile = StageTimings()
    with profile.measure("release_other_models_seconds"):
        stop_moss_worker()
        stop_diarizen_worker()
        release_local_models(preserve_fast=mode == "main")
        _active_mode = mode
    def on_event(event):
        if event["type"] == "log":
            append_job_log(job_id, event["message"])
        elif event["type"] == "warmup":
            update_job(job_id, stage="model", message=event["message"])
        elif event["type"] == "progress":
            progress = round(event["progress"] * 0.55) if mode == "review" else event["progress"]
            update_job(job_id, progress=progress, stage=event["stage"], message=event["message"])
    update_job(job_id, progress=10, stage="model", message="正在准备统一识别模型")
    _main_warmup_status.update(status="loading" if not _main_engine.ready else "ready", error=None)
    payload = _main_engine.transcribe(wav_path, expected_speakers, on_event)
    _main_warmup_status.update(status="ready", error=None)
    rows = payload["rows"]
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json_retry(work_dir / "main.json", rows)
    timings = payload["timings"]
    update_job(job_id, timings=timings, model_observation={
        "before_warmup": payload["model_state_before_warmup"],
        "main_worker_before": payload["model_state_before"], "main_worker_after": payload["model_state_after"],
        "load": payload["load_observation"], "profiling": payload["profiling"]})
    review_observation = None
    if mode == "review":
        with profile.measure("main_park_before_review_seconds"):
            park_observation = _main_engine.park()
        rows, review_seconds, review_observation = review_recording(
            wav_path, work_dir / "review", rows,
            lambda chunk: request_moss_transcription(job_id, chunk), stop_moss_worker,
            lambda p, s, m: update_job(job_id, progress=p, stage=s, message=m),
            lambda line: append_job_log(job_id, line),
        )
        timings["review_seconds"] = review_seconds
        timings.update(review_observation["timings"])
        timings.update(review_observation["firered"]["timings"])
        timings.update(review_observation["mover"]["timings"])
        payload["park_before_review"] = park_observation
        if expected_speakers:
            with profile.measure("speaker_count_constraint_seconds"):
                rows = apply_speaker_count_constraint(rows, expected_speakers, wav_path)
    if voiceprint_ids:
        with profile.measure("main_park_before_voiceprint_seconds"):
            payload["park_before_voiceprint"] = _main_engine.park()
        profiles = {str(p["id"]): p for p in read_voiceprints()}
        if any(profile_id not in profiles for profile_id in voiceprint_ids):
            raise ValueError("本次使用的声纹档案已删除，请重新选择。")
        update_job(job_id, progress=93, stage="voiceprint", message="正在匹配声纹库中的姓名")
        with profile.measure("voiceprint_match_seconds", gpu=True):
            rows = match_rows_to_voiceprints(rows, wav_path, [profiles[p] for p in voiceprint_ids])
    else:
        rows = normalize_speakers(rows)
    rows = validate_rows(rows, payload["duration_seconds"])
    timings.update(profile.seconds())
    update_job(job_id, model_observation={"before_warmup": payload["model_state_before_warmup"],
        "main_worker_before": payload["model_state_before"], "main_worker_after": payload["model_state_after"],
        "load": payload["load_observation"], "profiling": payload["profiling"],
        "review": review_observation, "park_before_review": payload.get("park_before_review"),
        "park_before_voiceprint": payload.get("park_before_voiceprint")})
    metadata = {k: v for k, v in payload.items() if k not in {"rows", "type", "id"}}
    metadata.update(timings=timings, pipeline="qwen-sortformer-campp" if mode == "main" else "main-moss-firered-mover")
    write_json_retry(work_dir / "metadata.json", metadata)
    update_job(job_id, timings=timings, warnings=payload["warnings"], duration_seconds=payload["duration_seconds"])
    return rows


def run_text_job(job_id, wav_path, work_dir):
    """ASR-only route, sharing the Qwen owner but bypassing all Main processing."""
    global _active_mode
    profile = StageTimings()
    with profile.measure("release_other_models_seconds"):
        stop_moss_worker()
        stop_diarizen_worker()
        release_local_models(preserve_fast=True)
        _active_mode = "text"

    def on_event(event):
        if event["type"] == "log":
            append_job_log(job_id, event["message"])
        elif event["type"] == "warmup":
            update_job(job_id, stage="model", message=event["message"])
        elif event["type"] == "progress":
            update_job(job_id, progress=event["progress"], stage=event["stage"], message=event["message"])

    update_job(job_id, progress=10, stage="model", message="正在准备纯文字识别模型")
    payload = _main_engine.transcribe_text(wav_path, on_event)
    _main_warmup_status.update(status="idle" if _main_engine.model_info.get("text_only") else "ready", error=None)
    payload["timings"].update(profile.seconds())
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json_retry(work_dir / "metadata.json", {k: v for k, v in payload.items() if k not in {"rows", "type", "id"}})
    update_job(job_id, text=payload["text"], chunks=payload["chunks"],
               merge_boundaries=payload["merge_boundaries"], speaker_analyzed=False, timestamp_kind="audio_extent",
               timings=payload["timings"], warnings=payload["warnings"], duration_seconds=payload["duration_seconds"],
               model_observation={"before_warmup": payload["model_state_before_warmup"],
                   "worker_before": payload["model_state_before"], "worker_after": payload["model_state_after"],
                   "load": payload["load_observation"], "profiling": payload["profiling"]})
    return payload["rows"]


def cancel_main_preload():
    """Signal only speculative preload; active recognition and cached models are safe."""
    with _warmup_request_lock:
        # A queued/running Main or Review task now owns this preparation.
        # A later Fast request must respect the existing task queue.
        with _job_lock:
            if any(job["mode"] in {"main", "review"} and job["status"] in {"queued", "running"}
                   for job in _jobs.values()):
                return False
        if _main_warmup_cancel is not None:
            _main_warmup_cancel.set()
            _main_warmup_status.update(message="正在停止未使用的统一识别预热")
            return True
    return False


def warmup_main_model(cancel_event=None):
    global _active_mode, _main_warmup_cancel
    if not _warmup_lock.acquire(blocking=False):
        with _warmup_request_lock:
            if _main_warmup_cancel is cancel_event:
                _main_warmup_cancel = None
                _main_warmup_status.update(status="idle", error=None)
        return
    try:
        with _inference_lock:
            if cancel_event is not None and cancel_event.is_set():
                _main_warmup_status.update(status="idle", error=None, message="统一识别将在选用时准备")
                return
            _main_warmup_status.update(status="loading", error=None)
            stop_moss_worker()
            stop_diarizen_worker()
            release_local_models(preserve_fast=True)
            _active_mode = "main"
            try:
                _main_engine.warmup(lambda event: _main_warmup_status.update(message=event["message"])
                                    if event.get("type") == "warmup" else None, cancel_event=cancel_event)
            except WarmupCancelled:
                _main_warmup_status.update(status="idle", error=None, message="统一识别将在选用时准备")
                _active_mode = None
                return
            _main_warmup_status.update(status="ready", error=None)
    except Exception as exc:
        _main_warmup_status.update(status="failed", error=str(exc))
    finally:
        with _warmup_request_lock:
            if _main_warmup_cancel is cancel_event:
                _main_warmup_cancel = None
        _warmup_lock.release()


def run_job(job_id: str) -> None:
    with _job_lock:
        job = dict(_jobs[job_id])
    source = Path(job["source"])
    job_dir = Path(job["job_dir"])
    session_id = job["session_id"]
    mode = job["mode"]
    expected_speakers = int(job.get("speaker_count") or 0)
    voiceprint_ids = list(job.get("voiceprint_ids") or [])
    started = time.perf_counter()
    profile = StageTimings()
    queue_wait = max(0, started - job.get("_queued_monotonic", started))
    update_job(job_id, model_state_at_execution_start=model_snapshot())
    update_job(job_id, started_at=time.time())
    update_job(job_id, status="running", progress=3, stage="prepare", message="正在准备音频")
    try:
        wait_started = time.perf_counter()
        if mode in {"fast", "text"}:
            cancel_main_preload()
        if mode in {"main", "review"} and _main_warmup_status["status"] == "loading":
            update_job(job_id, stage="model", message="正在等待模型预热完成")
        with _inference_lock:
            gpu_wait_seconds = round(time.perf_counter() - wait_started, 3)
            processing_started = time.perf_counter()
            update_job(job_id, model_state_after_gpu_lock=model_snapshot())
            if mode in {"main", "review", "text"}:
                wav_path = job_dir / "input" / f"{session_id}.wav"
                update_job(job_id, progress=6, message="正在转换为 16 kHz 单声道")
                with profile.measure("audio_conversion_seconds"):
                    convert_to_wav(source, wav_path)
                with profile.measure("unified_pipeline_seconds"):
                    rows = (run_text_job(job_id, wav_path, job_dir / "text") if mode == "text" else
                            run_unified_job(job_id, wav_path, job_dir / "unified", mode, expected_speakers, voiceprint_ids))
                with _job_lock:
                    timings = {**_jobs[job_id].get("timings", {}), "gpu_wait_seconds": gpu_wait_seconds}
                update_job(job_id, timings=timings)
            else:
                fast_dir = job_dir / "fast"
                wav_path = fast_dir / f"{session_id}.wav"
                update_job(job_id, progress=12, stage="convert", message="正在转换为 16 kHz 单声道")
                with profile.measure("audio_conversion_seconds"):
                    convert_to_wav(source, wav_path)
                with profile.measure("fast_release_other_models_seconds"):
                    prepare_fast_mode()
                update_job(job_id, progress=28, stage="model", message="正在加载快速识别模型")
                with profile.measure("fast_load_seconds", gpu=True):
                    model = load_model()
                discard_inactive_cache()
                update_job(job_id, progress=46, stage="transcribe", message="SenseVoice 正在识别并区分说话人")
                with profile.measure("fast_inference_seconds", gpu=True):
                    rows = infer_file(model, wav_path, split_max_ms=12000, split_gap_ms=500)
                if not rows:
                    raise RuntimeError("模型没有识别到有效语音。")
                rows = normalize_speakers(rows)
            discard_inactive_cache()
        update_job(job_id, progress=96, stage="finalize", message="正在整理转写结果")
        result_path = job_dir / "result.json"
        with profile.measure("result_write_seconds"):
            result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        speakers = 0 if mode == "text" else len({str(row["speaker"]) for row in rows})
        completed = time.perf_counter()
        with _job_lock:
            timings = dict(_jobs[job_id].get("timings", {}))
        timings.update(profile.seconds())
        top_level = ("audio_conversion_seconds", "unified_pipeline_seconds", "fast_release_other_models_seconds",
                     "fast_load_seconds", "fast_inference_seconds", "result_write_seconds")
        timings["server_processing_other_seconds"] = round(max(0, completed - processing_started
            - sum(profile.values.get(k, 0) for k in top_level)), 6)
        timings.update(queue_wait_seconds=round(queue_wait, 6), gpu_wait_seconds=gpu_wait_seconds,
                       processing_seconds=round(completed - processing_started, 6),
                       upload_receive_seconds=job.get("upload_receive_seconds", 0),
                       server_end_to_end_seconds=round(completed - job.get("_received_monotonic", started), 6))
        update_job(
            job_id,
            status="completed",
            progress=100,
            stage="completed",
            message="识别完成",
            segments=len(rows),
            speakers=speakers,
            result_path=str(result_path),
            elapsed_seconds=round(time.perf_counter() - started, 3),
            completed_at=time.time(),
            timings=timings, model_state_at_completion=model_snapshot(),
        )
        source.unlink(missing_ok=True)
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            stage="failed",
            message="识别失败",
            error=str(exc),
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
    finally:
        with _job_lock:
            observation = public_job(dict(_jobs[job_id]))
        observation.update(source_sha256=SOURCE_SHA256, observation_schema=1,
                           server_profiling=profile.info())
        write_json_retry(job_dir / "performance.json", observation)


def model_snapshot():
    return {"observed_at": time.time(), "main": _main_engine.state(),
            "fast": model_state(getattr(_model, "model", None)),
            "voiceprint": model_state(getattr(_speaker_model, "model", None)),
            "moss": dict(_moss_model_state), "fast_components": fast_state(_model),
            "cache": {"active_mode": _active_mode, "memory": memory_budget(), "events": list(_cache_events)}}


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "service": "声迹本地转写服务",
        "version": "3.0",
        "busy": any(job.get("status") == "running" for job in _jobs.values()),
        "fast": {
            "available": True,
            "model": "SenseVoiceSmall + FSMN-VAD + CAM++",
            "model_loaded": _model is not None,
            "model_state": fast_state(_model),
        },
        "cache": {"active_mode": _active_mode, "memory": memory_budget(), "events": list(_cache_events)},
        "main": {
            "available": not missing_requirements(),
            "missing": missing_requirements(),
            "model": "Qwen3-ASR-1.7B + ForcedAligner + Sortformer + CAM++",
            "model_loaded": bool(_main_engine.ready and _main_engine.process and _main_engine.process.poll() is None
                                 and not _main_engine.model_info.get("text_only", False)),
            "warmup_status": _main_warmup_status["status"],
            "warmup_error": _main_warmup_status["error"],
            "warmup_message": _main_warmup_status.get("message", "正在预热模型"),
            "startup_seconds": _main_engine.startup_seconds,
            "runtime": _main_engine.model_info,
            "model_state": _main_engine.state(),
        },
        "review": {
            "available": not missing_requirements(review=True),
            "missing": missing_requirements(review=True),
            "model": "Main + MOSS + FireRed + MOVER",
            "experimental": True,
        },
        "text": {
            "available": not missing_requirements(text_only=True),
            "missing": missing_requirements(text_only=True), "model": "Qwen3-ASR-1.7B",
            "model_loaded": bool(_main_engine.ready and _main_engine.process and _main_engine.process.poll() is None),
            "speaker_analyzed": False, "timestamp_kind": "audio_extent",
        },
    }


@app.post("/api/warmup", status_code=202)
def warmup(mode: str = "main") -> dict:
    global _main_warmup_cancel
    mode = {"conversation": "main", "accurate": "review"}.get(mode, mode)
    if mode in {"fast", "text"}:
        cancelled = cancel_main_preload()
        return {"ok": True, "status": "cancelling" if cancelled else "on_demand", "mode": mode}
    if mode not in {"main", "review"}:
        raise HTTPException(status_code=400, detail="识别模式必须是 main、review、fast 或 text。")
    if missing_requirements(review=mode == "review"):
        raise HTTPException(status_code=503, detail="统一识别所需环境尚未准备完整。")
    if (_main_engine.ready and _main_engine.process and _main_engine.process.poll() is None
            and not _main_engine.model_info.get("text_only", False)):
        return {"ok": True, "status": "ready"}
    with _warmup_request_lock:
        with _job_lock:
            pending = any(job["status"] in {"queued", "running"} for job in _jobs.values())
        if pending:
            return {"ok": True, "status": "task_pending", "mode": mode}
        if _main_warmup_cancel is None and _main_warmup_status["status"] != "loading":
            _main_warmup_cancel = threading.Event()
            _main_warmup_status.update(status="loading", error=None, message="正在准备统一识别模型")
            threading.Thread(target=warmup_main_model, args=(_main_warmup_cancel,),
                             name="soundtrace-warmup", daemon=True).start()
    return {"ok": True, "status": _main_warmup_status["status"]}


@app.get("/api/voiceprints")
def list_voiceprints() -> dict:
    return {"profiles": [public_voiceprint(profile) for profile in read_voiceprints()]}


@app.post("/api/voiceprints")
async def create_voiceprint(
    name: str = Form(...),
    audio: UploadFile = File(...),
) -> dict:
    clean_name = " ".join(name.strip().split())
    if not clean_name or len(clean_name) > 24 or any(ord(character) < 32 for character in clean_name):
        raise HTTPException(status_code=400, detail="姓名需要是 1–24 个正常字符。")
    suffix = Path(audio.filename or "voiceprint.wav").suffix.lower() or ".wav"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"声纹录音不支持 {suffix} 格式。")

    enrollment_id = uuid.uuid4().hex
    work_dir = VOICEPRINTS_DIR / "temporary" / enrollment_id
    source = work_dir / f"source{suffix}"
    work_dir.mkdir(parents=True, exist_ok=False)
    total = 0
    try:
        with source.open("wb") as file:
            while chunk := await audio.read(1024 * 1024):
                total += len(chunk)
                if total > 64 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="声纹录音超过 64 MB 上限。")
                file.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="声纹录音为空。")

        def process_enrollment() -> dict:
            with _inference_lock:
                _main_engine.park()
                return enroll_voiceprint(clean_name, source, work_dir)

        profile = await run_in_threadpool(process_enrollment)
        return {"ok": True, "profile": profile}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await audio.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@app.delete("/api/voiceprints/{profile_id}")
def delete_voiceprint(profile_id: str) -> dict:
    with _job_lock:
        in_use = any(
            job.get("status") in {"queued", "running"}
            and profile_id in (job.get("voiceprint_ids") or [])
            for job in _jobs.values()
        )
    if in_use:
        raise HTTPException(status_code=409, detail="这个声纹正在用于识别任务，请任务结束后再删除。")
    profiles = read_voiceprints()
    retained = [profile for profile in profiles if str(profile.get("id")) != profile_id]
    if len(retained) == len(profiles):
        raise HTTPException(status_code=404, detail="找不到这个声纹档案。")
    write_voiceprints(retained)
    return {"ok": True}


@app.post("/api/jobs", status_code=202)
async def create_job(
    audio: UploadFile = File(...),
    mode: str = Form("main"),
    speaker_count: int = Form(0),
    voiceprint_ids: str = Form(""),
) -> dict:
    received = time.perf_counter()
    requested_mode = mode
    mode = {"conversation": "main", "accurate": "review"}.get(mode, mode)
    if mode not in {"fast", "main", "review", "text"}:
        raise HTTPException(status_code=400, detail="识别模式必须是 main、review、fast 或 text。")
    if mode == "text":
        missing = missing_requirements(text_only=True)
        if missing:
            raise HTTPException(status_code=503, detail="纯文字识别缺少本地环境或模型：" + "、".join(missing))
        if speaker_count or voiceprint_ids.strip():
            raise HTTPException(status_code=400, detail="纯文字模式不分析说话人，请不要指定人数或声纹档案。")
    if mode in {"main", "review"}:
        missing = missing_requirements(review=mode == "review")
        if missing:
            raise HTTPException(status_code=503, detail=f"所选 {mode} 模式无法运行，缺少本地环境或模型：" + "、".join(missing))
    if speaker_count < 0 or speaker_count > 6:
        raise HTTPException(status_code=400, detail="说话人数必须为自动或 1–6 人。")
    selected_voiceprints = list(dict.fromkeys(item.strip() for item in voiceprint_ids.split(",") if item.strip()))
    if selected_voiceprints and mode == "fast":
        raise HTTPException(status_code=400, detail="声纹库适用于统一识别和精细复核。")
    if len(selected_voiceprints) > 6:
        raise HTTPException(status_code=400, detail="一次最多匹配 6 个声纹档案。")
    if selected_voiceprints:
        known_ids = {str(profile.get("id")) for profile in read_voiceprints()}
        if any(profile_id not in known_ids for profile_id in selected_voiceprints):
            raise HTTPException(status_code=400, detail="所选声纹档案不存在，请刷新声纹库。")
        speaker_count = len(selected_voiceprints)
    suffix = Path(audio.filename or "audio.wav").suffix.lower() or ".wav"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不支持 {suffix} 格式。")

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    source = job_dir / f"source{suffix}"
    job_dir.mkdir(parents=True, exist_ok=False)
    total = 0
    try:
        with source.open("wb") as file:
            while chunk := await audio.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="音频超过 512 MB 上限。")
                file.write(chunk)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        await audio.close()
    if total == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="上传的音频为空。")

    now = time.time()
    queued = time.perf_counter()
    job = {
        "id": job_id,
        "filename": audio.filename or source.name,
        "session_id": safe_session_id(audio.filename),
        "mode": mode,
        "requested_mode": requested_mode,
        "auto_routed": False,
        "speaker_count": speaker_count,
        "voiceprint_ids": selected_voiceprints,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "message": "已进入本地 GPU 队列",
        "created_at": now,
        "updated_at": now,
        "job_dir": str(job_dir),
        "source": str(source),
        "log": [],
        "_received_monotonic": received,
        "_queued_monotonic": queued,
        "upload_receive_seconds": round(queued - received, 6),
    }
    with _job_lock:
        _jobs[job_id] = job
    if mode in {"fast", "text"}:
        cancel_main_preload()
    _executor.submit(run_job, job_id)
    return public_job(job)


@app.post("/api/jobs/{job_id}/client-timing")
def record_client_timing(job_id: str, payload: dict) -> dict:
    fields = ("user_end_to_end_seconds", "client_submit_seconds", "client_after_accept_seconds")
    values = {}
    for key in fields:
        value = payload.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 86400:
            raise HTTPException(status_code=400, detail="无效的客户端计时。")
        values[key] = round(float(value), 6)
    with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="找不到该识别任务。")
        if job["status"] != "completed":
            raise HTTPException(status_code=409, detail="识别任务尚未完成。")
        job["client_timings"] = values
        directory = Path(job["job_dir"])
    write_json_retry(directory / "client_timing.json", values)
    return {"ok": True}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with _job_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="找不到该识别任务。")
        return public_job(dict(job))


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str) -> dict:
    with _job_lock:
        job = dict(_jobs.get(job_id) or {})
    if not job:
        raise HTTPException(status_code=404, detail="找不到该识别任务。")
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="识别任务尚未完成。")
    path = Path(job["result_path"])
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = {"rows": rows, "session_id": job["session_id"], "mode": job["mode"],
            "elapsed_seconds": job.get("elapsed_seconds"), "timings": job.get("timings", {}),
            "warnings": job.get("warnings", [])}
    if job["mode"] == "text":
        result.update(text=job["text"], chunks=job["chunks"], merge_boundaries=job["merge_boundaries"],
                      speaker_analyzed=False, timestamp_kind="audio_extent")
    return result


@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> dict:
    """Compatibility endpoint for older front ends; new clients use /api/jobs."""
    suffix = Path(audio.filename or "audio.wav").suffix.lower() or ".wav"
    session_id = safe_session_id(audio.filename)
    job_dir = JOBS_DIR / f"legacy-{uuid.uuid4().hex}"
    source = job_dir / f"source{suffix}"
    job_dir.mkdir(parents=True, exist_ok=False)
    try:
        with source.open("wb") as file:
            while chunk := await audio.read(1024 * 1024):
                file.write(chunk)
        rows = await run_in_threadpool(transcribe_fast, source, session_id, job_dir)
        return {"rows": rows, "session_id": session_id, "mode": "fast"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"识别失败：{exc}") from exc
    finally:
        await audio.close()
        shutil.rmtree(job_dir, ignore_errors=True)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "index.html")


app.mount("/", StaticFiles(directory=APP_DIR), name="static")
atexit.register(_main_engine.stop)


@app.on_event("startup")
def start_model_warmup():
    _cache_monitor_stop.clear()
    threading.Thread(target=monitor_model_cache, name="soundtrace-cache-guard", daemon=True).start()
    # The server cannot know the browser's saved/selected mode. The page
    # requests Main/Review preload after restoring it; Fast loads on demand.


@app.on_event("shutdown")
def stop_workers():
    _cache_monitor_stop.set()
    # Shutdown may cancel even a preload needed by a queued task. Do not use
    # the user-mode guard here; the service is closing.
    with _warmup_request_lock:
        if _main_warmup_cancel is not None:
            _main_warmup_cancel.set()
    _main_engine.stop()
    unload_fast_model()
    stop_moss_worker()
    stop_diarizen_worker()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8767)
