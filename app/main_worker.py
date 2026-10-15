"""Persistent single-recording Qwen/Sortformer/CAM++ worker.

Uses the locked competition weights and prototype logic, with prototypes scoped
to this recording. Reuses GPU models and batches windows, with an OOM fallback.
"""
from __future__ import annotations

import gc
import os
import shutil
import importlib.util
import json
import sys
import time
from pathlib import Path

from performance import StageTimings, model_state, observe_call

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SOURCE = ROOT / "project/code/test/packages/ifly_asr_dev017_verified_20260725/code"
sys.path.insert(0, str(SOURCE / "src"))
sys.path.insert(0, str(ROOT / "vendor/3D-Speaker"))
PREFIX = "__SOUNDTRACE_MAIN__"
MODELS = ROOT / "project/user_data/model_data"


def emit(value):
    print(PREFIX + json.dumps(value, ensure_ascii=False), flush=True)


def _short_track_prototypes(turns, matrix, anchors, expected_count):
    """Recover missing short-speaker evidence only for unsupported prototypes.

    Exactly K long turns from fewer than K local tracks do not independently
    support K identities. Use coherent local tracks in this case, keeping
    simultaneous tracks apart. Longer recordings retain the original recipe:
    their local track IDs may legitimately change identity over time.
    """
    import numpy as np

    if (expected_count < 2 or len(anchors) != expected_count
            or len({turns[i][2] for i in anchors}) >= expected_count):
        return None
    tracks = {}
    for i, (_, _, track) in enumerate(turns):
        tracks.setdefault(track, []).append(i)
    groups = list(tracks.values())
    if len(groups) < expected_count:
        return None
    for indices in groups:
        vectors = matrix[indices]
        # Reuse the existing acoustic distance threshold. Do not trust a local
        # track that already contains acoustically inconsistent turns.
        for offset in range(0, len(vectors), 256):
            if np.any(1 - vectors[offset:offset + 256] @ vectors.T > 0.561):
                return None

    def overlapping(left, right):
        return any(min(turns[i][1], turns[j][1]) > max(turns[i][0], turns[j][0])
                   for i in left for j in right)

    # There are at most the diarizer's few local tracks, not one group per word.
    while len(groups) > expected_count:
        candidates = []
        for i, left in enumerate(groups):
            for j in range(i + 1, len(groups)):
                right = groups[j]
                if not overlapping(left, right):
                    distance = 1 - float(matrix[left].mean(axis=0) @ matrix[right].mean(axis=0))
                    candidates.append((distance, i, j))
        if not candidates:
            # Inconsistent overlap/count evidence: keep the established route
            # rather than force an impossible merge or invent a new identity.
            return None
        _, i, j = min(candidates)
        groups[i] = groups[i] + groups.pop(j)
    prototypes = np.stack([matrix[indices].mean(axis=0) for indices in groups])
    return prototypes / np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)


def cluster_turns(turns, embeddings, expected_count=0, diagnostics=None):
    """Long-turn prototypes; short recordings may use their available turns."""
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from ifly_tcpwer.global_proto import _smooth_local_tracks

    matrix = np.asarray(embeddings, dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    anchors = np.array([i for i, (s, e, _) in enumerate(turns) if e - s >= 2.1], dtype=int)
    short_prototypes = _short_track_prototypes(turns, matrix, anchors, expected_count)
    if diagnostics is not None:
        diagnostics.update(strategy="short_track_prototypes" if short_prototypes is not None
                           else "long_turn_prototypes", long_turn_count=len(anchors),
                           long_turn_track_count=len({turns[i][2] for i in anchors}))
    minimum = expected_count or 2
    if len(anchors) < minimum:
        anchors = np.arange(len(turns))
    if len(anchors) == 1 or expected_count == 1:
        labels = np.zeros(len(anchors), dtype=int)
    else:
        kwargs = dict(metric="cosine", linkage="average")
        if expected_count:
            kwargs.update(n_clusters=min(expected_count, len(anchors)))
        else:
            kwargs.update(n_clusters=None, distance_threshold=0.561)
        labels = AgglomerativeClustering(**kwargs).fit_predict(matrix[anchors])
    prototypes = np.stack([matrix[anchors[labels == k]].mean(axis=0) for k in sorted(set(labels))])
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)
    if short_prototypes is not None:
        prototypes = short_prototypes
    emissions = matrix @ prototypes.T
    identities = _smooth_local_tracks(
        [("recording", i, s, e, track) for i, (s, e, track) in enumerate(turns)],
        emissions, 0.44, 0.3,
    )
    return [[s, e, int(identity)] for (s, e, _), identity in zip(turns, identities)]


def _short_track_seglst(session_id, timestamps, turns, diarization):
    """Resolve a short-prototype handoff without copying overlapping words.

    A weak extra local track must not move the handoff earlier than the main
    track supporting that identity. When both supported tracks are active,
    prefer the new onset instead of continuing the preceding speaker forever.
    Token text and aligner times remain unchanged.
    """
    from ifly_tcpwer.merge import Token, assign_speakers, tokens_to_segments

    support = {}
    for (s, e, track), (_, _, identity) in zip(turns, diarization):
        by_track = support.setdefault(identity, {})
        by_track[track] = by_track.get(track, 0.0) + e - s
    primary = {identity: max(tracks, key=tracks.get) for identity, tracks in support.items()}
    tokens = [Token(t['text'], float(t['start_time']), float(t['end_time'])) for t in timestamps]
    assign_speakers(tokens, diarization)
    for token in tokens:
        midpoint = (token.start + token.end) / 2
        active = [i for i, (s, e, _) in enumerate(turns) if s <= midpoint <= e]
        if len({diarization[i][2] for i in active}) < 2:
            continue
        supported = [i for i in active if turns[i][2] == primary[diarization[i][2]]]
        if supported:
            chosen = max(supported, key=lambda i: turns[i][0])
            token.speaker = f"spk{diarization[chosen][2] + 1}"
    return [dict(session_id=session_id, speaker=segment.speaker,
                 start_time=round(max(0.0, segment.start), 2),
                 end_time=round(max(segment.start, segment.end), 2),
                 words=" ".join(segment.tokens))
            for segment in tokens_to_segments(tokens) if segment.tokens]


class Runner:
    def __init__(self, text_only=False):
        self.qwen = self.sortformer = self.camp = None
        self.load_profile = StageTimings()
        with self.load_profile.measure("torch_import_seconds"):
            import torch
        load_started = time.perf_counter()
        torch.set_num_threads(max(1, int(os.environ.get("SOUNDTRACE_CPU_THREADS", "4"))))
        self.device = "cuda:0"
        self.batch_size = max(1, min(8, int(os.environ.get("SOUNDTRACE_ASR_BATCH_SIZE", "8"))))
        self.resident = os.environ.get("SOUNDTRACE_GPU_RESIDENT", "auto") != "0"
        if os.environ.get("SOUNDTRACE_GPU_RESIDENT", "auto") == "auto":
            self.resident = torch.cuda.get_device_properties(0).total_memory >= 7.5 * 1024**3
        # nagisa/DyNet opens its bundled Japanese tokenizer weights through a
        # native Windows API that cannot read the workspace's Chinese path.
        if os.name == "nt":
            spec = importlib.util.find_spec("nagisa")
            if spec and not str(spec.origin).isascii():
                packages = Path(os.environ["LOCALAPPDATA"]) / "Soundtrace/runtime/python_packages"
                source = Path(spec.origin).parent
                target = packages / "nagisa"
                if not (target / "data/nagisa_v001.model").is_file():
                    shutil.copytree(source, target, dirs_exist_ok=True)
                sys.path.insert(0, str(packages))
        emit({"type": "warmup", "message": "正在加载文字识别模型" if text_only else "正在加载文字识别和时间对齐模型", "model_state_after": self.state()})
        with self.load_profile.measure("qwen_dependency_import_seconds"):
            from qwen_asr import Qwen3ASRModel
            from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner
        with observe_call(Qwen3ForcedAligner, "from_pretrained", self.load_profile,
                          "aligner_load_and_place_seconds", gpu=True):
            with self.load_profile.measure("qwen_and_aligner_load_seconds", gpu=True):
                self.qwen = Qwen3ASRModel.from_pretrained(
                    str(MODELS / "Qwen3-ASR-1.7B"), dtype=torch.bfloat16,
                    device_map=self.device, max_inference_batch_size=self.batch_size, max_new_tokens=768,
                    forced_aligner=None if text_only else str(MODELS / "Qwen3-ForcedAligner-0.6B"),
                    forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": self.device},
                )
        self.load_profile.values["qwen_load_and_place_seconds"] = (
            self.load_profile.values["qwen_and_aligner_load_seconds"]
            - self.load_profile.values.get("aligner_load_and_place_seconds", 0))
        self.sortformer = None
        self.camp = None
        self.fbank = None
        if text_only:
            self.load_seconds = round(time.perf_counter() - load_started, 3)
            return
        self.load_speakers()
        if self.resident:
            try:
                self.move(self.sortformer, self.device, self.load_profile, "sortformer_to_gpu_seconds")
                self.move(self.camp, self.device, self.load_profile, "campp_to_gpu_seconds")
            except torch.cuda.OutOfMemoryError:
                self.resident = False
                self.park(self.load_profile)
        else:
            self.park(self.load_profile)
        self.load_seconds = round(time.perf_counter() - load_started, 3)

    def ready_info(self):
        from model_lifecycle import tensor_bytes, cpu_move_groups
        components = [self.qwen.model, self.qwen.forced_aligner.model if self.qwen.forced_aligner else None,
                      self.sortformer, self.camp]
        components = [module for module in components if module is not None]
        return {"load_seconds": self.load_seconds, "text_only": any(m is None for m in
                (self.qwen.forced_aligner, self.sortformer, self.camp)),
                "qwen_object_id": id(self.qwen),
                "speaker_dependencies_imported": any(n.startswith(("nemo.", "speakerlab.")) for n in sys.modules),
                "cpu_cache_weight_bytes": tensor_bytes(components),
                "cpu_cache_largest_model_bytes": max(tensor_bytes([m]) for m in components),
                "cpu_cache_park_peak_bytes": max(size for m in components for _, size in cpu_move_groups(m)),
                "asr_batch_size": self.batch_size, "gpu_resident": self.resident,
                "load_timings": self.load_profile.seconds(), "load_profiling": self.load_profile.info(),
                "model_state_after": self.state()}

    def prepare_main(self):
        """Upgrade the same Qwen owner; never reload ASR just to add diarization."""
        import torch
        started = time.perf_counter()
        if self.qwen.forced_aligner is None:
            from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner
            self.move(self.qwen.model, self.device, self.load_profile, "qwen_to_gpu_seconds")
            with self.load_profile.measure("aligner_load_and_place_seconds", gpu=True):
                self.qwen.forced_aligner = Qwen3ForcedAligner.from_pretrained(
                    str(MODELS / "Qwen3-ForcedAligner-0.6B"), dtype=torch.bfloat16, device_map=self.device)
        self.load_speakers()
        if self.resident:
            try:
                self.move(self.sortformer, self.device, self.load_profile, "sortformer_to_gpu_seconds")
                self.move(self.camp, self.device, self.load_profile, "campp_to_gpu_seconds")
            except torch.cuda.OutOfMemoryError:
                self.resident = False
                self.park(self.load_profile)
        else:
            self.park(self.load_profile)
        return {**self.ready_info(), "upgrade_seconds": round(time.perf_counter() - started, 6)}

    def transcribe_text(self, request):
        import soundfile as sf
        import torch
        from text_transcription import merge_windows, text_row, text_window_plan, TEXT_SINGLE_WINDOW_SECONDS

        started = time.perf_counter()
        profile = StageTimings()
        before = self.state()
        wav = Path(request["wav"])
        with profile.measure("audio_read_seconds"):
            audio, sr = sf.read(wav, dtype="float32")
        if sr != 16000 or audio.ndim != 1 or not len(audio):
            raise ValueError("Text expects nonempty 16 kHz mono WAV")
        duration = len(audio) / sr
        # Avoid unnecessary seams only for bounded short Text requests. Main
        # and the longer Text 14/2 overlap/merging path stay unchanged.
        windows = text_window_plan(duration)
        self.move(self.qwen.model, self.device, profile, "qwen_to_gpu_seconds")
        torch.cuda.reset_peak_memory_stats()
        chunks = []
        retries = 0
        # Retry the entire request after OOM; never append a failed partial batch.
        batch_size = self.batch_size
        while True:
            try:
                chunks = []
                for start in range(0, len(windows), batch_size):
                    batch = windows[start:start + batch_size]
                    emit({"type": "progress", "id": request["id"], "progress": 15 + int(70 * start / len(windows)),
                          "stage": "qwen", "message": f"正在转写文字窗口 {start + 1}–{start + len(batch)}/{len(windows)}"})
                    clips = [(audio[round(s * sr):round(e * sr)], sr) for s, e, _, _ in batch]
                    with profile.measure("asr_seconds", gpu=True):
                        results = self.qwen.transcribe(audio=clips, language="Chinese", return_time_stamps=False)
                    if len(results) != len(batch):
                        raise RuntimeError("批量识别返回的片段数量不一致。")
                    chunks.extend({"start_time": s, "end_time": e, "text": r.text,
                                   "timestamp_kind": "audio_window"} for (s, e, _, _), r in zip(batch, results))
                break
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                retries += 1
            gc.collect()
            torch.cuda.empty_cache()
        with profile.measure("text_merge_seconds"):
            text, boundaries, warnings = merge_windows(chunks)
        if not text.strip():
            raise RuntimeError("模型没有识别到有效语音。")
        return {"rows": [text_row(wav.stem, text, duration)], "text": text, "chunks": chunks,
                "merge_boundaries": boundaries, "warnings": warnings,
                "duration_seconds": duration, "windows": len(windows), "asr_batch_size": batch_size,
                "window_strategy": "single_short" if duration <= TEXT_SINGLE_WINDOW_SECONDS else "overlap_14_2",
                "single_window_limit_seconds": TEXT_SINGLE_WINDOW_SECONDS,
                "speaker_analyzed": False, "timestamp_kind": "audio_extent",
                "model_state_before": before, "model_state_after": self.state(),
                "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                "profiling": {**profile.info(), "oom_retries": retries},
                "timings": {**profile.seconds(), "worker_request_seconds": round(time.perf_counter() - started, 6)}}

    def load_speakers(self):
        import torch
        if self.sortformer is None:
            emit({"type": "warmup", "message": "正在加载说话人分离模型", "model_state_after": self.state()})
            with self.load_profile.measure("sortformer_dependency_import_seconds"):
                from nemo.collections.asr.models import SortformerEncLabelModel
            with self.load_profile.measure("sortformer_cpu_load_seconds"):
                self.sortformer = SortformerEncLabelModel.restore_from(
                    str(MODELS / "diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo"),
                    map_location="cpu",
                ).eval()
        if self.camp is None:
            emit({"type": "warmup", "message": "正在加载声纹模型", "model_state_after": self.state()})
            with self.load_profile.measure("campp_cpu_load_seconds"):
                from speakerlab.models.campplus.DTDNN import CAMPPlus
                from speakerlab.process.processor import FBank
                self.camp = CAMPPlus(feat_dim=80, embedding_size=192)
                weights = MODELS / "3dspeaker/iic/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
                self.camp.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
                self.camp.eval()
                self.fbank = FBank(n_mels=80, sample_rate=16000, mean_nor=True)

    def transcribe(self, request):
        import torch
        self.profile = StageTimings()
        self.request_state = None
        started = time.perf_counter()
        retries = 0
        while True:
            try:
                result = self._transcribe(request)
                if "timings" in result:
                    result["timings"]["worker_request_seconds"] = round(time.perf_counter() - started, 6)
                    result["profiling"]["oom_retries"] = retries
                return result
            except torch.cuda.OutOfMemoryError:
                # Restart ownership from the beginning; never append partial
                # batches from the failed attempt to the retried result.
                if not self.resident and self.batch_size == 1:
                    raise
                if self.batch_size > 1:
                    self.batch_size = max(1, self.batch_size // 2)
                else:
                    self.resident = False
                self.qwen.max_inference_batch_size = self.batch_size
                retries += 1
            # Free failed-generation tensors/tracebacks before reallocating.
            self.park(self.profile)
            emit({"type": "progress", "id": request["id"], "progress": 12,
                  "stage": "qwen", "message": "显存不足，正在用低显存方式继续识别"})

    def state(self):
        return {"observed_at": time.time(), "models": {
            "qwen": model_state(self.qwen.model if self.qwen else None),
            "aligner": model_state(self.qwen.forced_aligner.model if self.qwen and self.qwen.forced_aligner else None),
            "sortformer": model_state(self.sortformer), "campp": model_state(self.camp)}}

    def move(self, model, device, profile, key):
        # Preserve even no-op .to()/.cpu() calls; record actual source devices.
        before = model_state(model)
        with profile.measure(key, gpu=True):
            model.cpu() if device == "cpu" else model.to(device)
        return before

    def park(self, profile=None, release_between_moves=False):
        import torch
        profile = profile or StageTimings()
        from model_lifecycle import cpu_move_groups
        for module, key in [(self.qwen.model, "qwen_to_cpu_seconds"),
                            (self.qwen.forced_aligner.model if self.qwen.forced_aligner else None, "aligner_to_cpu_seconds"),
                            (self.sortformer, "sortformer_to_cpu_seconds"), (self.camp, "campp_to_cpu_seconds")]:
            if module is not None:
                if release_between_moves:
                    # Return each bounded GPU allocation to WDDM before copying
                    # the next group. Keep the original whole-model park elsewhere.
                    for group, _ in cpu_move_groups(module):
                        self.move(group, "cpu", profile, key)
                        with profile.measure("cache_cleanup_seconds", gpu=True):
                            torch.cuda.empty_cache()
                else:
                    self.move(module, "cpu", profile, key)
        with profile.measure("cache_cleanup_seconds", gpu=True):
            gc.collect()
            torch.cuda.empty_cache()
        return {"model_state_after": self.state(), "timings": profile.seconds(), "profiling": profile.info()}

    def _transcribe(self, request):
        import numpy as np
        import soundfile as sf
        import torch
        from infer_qwen_asr_windows import make_windows, ownership_ranges, normalize_tokens
        from ifly_tcpwer.merge import make_seglst

        started = time.perf_counter()
        profile = self.profile
        if self.request_state is None:
            self.request_state = self.state()
        wav = Path(request["wav"])
        with profile.measure("audio_read_seconds"):
            audio, sr = sf.read(wav, dtype="float32")
        if sr != 16000 or audio.ndim != 1:
            raise ValueError("Main expects 16 kHz mono WAV")
        duration = len(audio) / sr
        timestamps = []
        windows = make_windows(duration, 14.0, 2.0)
        if not self.resident:
            self.park(profile)
        self.move(self.qwen.model, self.device, profile, "qwen_to_gpu_seconds")
        self.move(self.qwen.forced_aligner.model, self.device, profile, "aligner_to_gpu_seconds")
        torch.cuda.reset_peak_memory_stats()
        asr_started = time.perf_counter()
        owners = ownership_ranges(windows)
        for batch_start in range(0, len(windows), self.batch_size):
            batch_end = min(len(windows), batch_start + self.batch_size)
            emit({"type": "progress", "id": request["id"], "progress": 15 + int(40 * batch_start / len(windows)),
                  "stage": "qwen", "message": f"正在识别并对齐语音 {batch_start + 1}–{batch_end}/{len(windows)}"})
            clips = [(audio[round(offset * sr):round(end * sr)], sr)
                     for offset, end in windows[batch_start:batch_end]]
            with observe_call(self.qwen, "_infer_asr", profile, "asr_seconds", gpu=True), \
                 observe_call(self.qwen.forced_aligner, "align", profile, "forced_aligner_seconds", gpu=True):
                with profile.measure("qwen_transcribe_seconds"):
                    results = self.qwen.transcribe(
                        audio=clips,
                        language="Chinese", return_time_stamps=True,
                    )
            if len(results) != len(clips):
                raise RuntimeError("批量识别返回的片段数量不一致。")
            for i, result in enumerate(results, batch_start):
                offset, _ = windows[i]
                keep_start, keep_end = owners[i]
                if normalize_tokens(result.text) and not result.time_stamps:
                    raise RuntimeError("识别到了文字，但时间对齐失败；请重试。")
                for item in result.time_stamps or []:
                    tokens = normalize_tokens(item.text)
                    s = max(0.0, offset + float(item.start_time))
                    e = min(duration, offset + float(item.end_time))
                    if not tokens or e <= s:
                        continue
                    mid = (s + e) / 2
                    if mid < keep_start or (mid >= keep_end and i + 1 < len(windows)):
                        continue
                    step = (e - s) / len(tokens)
                    timestamps.extend({"text": word, "start_time": s + j * step, "end_time": s + (j + 1) * step}
                                      for j, word in enumerate(tokens))
        timestamps.sort(key=lambda item: (item["start_time"], item["end_time"]))
        asr_seconds = time.perf_counter() - asr_started
        if not self.resident:
            self.park(profile)
        if not timestamps:
            raise RuntimeError("模型没有识别到有效语音。")

        emit({"type": "progress", "id": request["id"], "progress": 60, "stage": "sortformer",
              "message": "正在分析发言和说话人切换"})
        diar_started = time.perf_counter()
        self.move(self.sortformer, self.device, profile, "sortformer_to_gpu_seconds")
        with profile.measure("sortformer_seconds", gpu=True):
            predictions = self.sortformer.diarize(audio=[str(wav)], batch_size=1, num_workers=0, verbose=False)[0]
        tracks = {}
        turns = []
        for line in predictions:
            s, e, label = line.split()
            s, e = max(0.0, float(s)), min(duration, float(e))
            if e > s:
                tracks.setdefault(label, len(tracks))
                turns.append([s, e, tracks[label]])
        turns.sort(key=lambda turn: (turn[0], turn[1], turn[2]))
        if not self.resident:
            self.move(self.sortformer, "cpu", profile, "sortformer_to_cpu_seconds")
            with profile.measure("cache_cleanup_seconds", gpu=True):
                torch.cuda.empty_cache()
        warnings = []
        if not turns:
            turns = [[timestamps[0]["start_time"], timestamps[-1]["end_time"], 0]]
            warnings.append("发言活动证据不足，暂以单一说话人展示；可指定人数或使用声纹库。")
        emit({"type": "progress", "id": request["id"], "progress": 78, "stage": "campp",
              "message": "正在建立本次录音的声纹与人物身份"})
        self.move(self.camp, self.device, profile, "campp_to_gpu_seconds")
        embeddings = []
        from speakerlab.utils.utils import circle_pad
        with torch.inference_mode():
            for s, e, _ in turns:
                with profile.measure("campp_features_seconds"):
                    clip = torch.from_numpy(audio[int(s * sr):max(int(s * sr) + 1, int(e * sr))].copy())
                    # Long turns need bounded memory; sample up to eight seconds.
                    clip = circle_pad(clip[:8 * sr], max(1600, min(len(clip), 8 * sr)))
                    features = self.fbank(clip.unsqueeze(0))
                with profile.measure("campp_features_to_gpu_seconds", gpu=True):
                    features = features.to(self.device)
                with profile.measure("campp_seconds", gpu=True):
                    embedding = self.camp(features.unsqueeze(0))
                with profile.measure("campp_result_to_cpu_seconds", gpu=True):
                    embeddings.append(embedding.cpu().numpy()[0])
                del embedding
        expected = int(request.get("speaker_count", 0))
        speaker_clustering = {}
        with profile.measure("speaker_postprocess_seconds"):
            diarization = cluster_turns(turns, np.stack(embeddings), expected, speaker_clustering)
        if expected > len(turns):
            warnings.append("有效发言片段少于指定人数，未强行生成额外人物。")
        with profile.measure("seglst_postprocess_seconds"):
            if speaker_clustering["strategy"] == "short_track_prototypes":
                rows = _short_track_seglst(wav.stem, timestamps, turns, diarization)
                speaker_clustering["overlap_assignment"] = "supported_track_onset"
            else:
                rows = make_seglst(wav.stem, {"timestamps": timestamps}, diarization)
        # Single ASR cannot recover both simultaneous utterances. Do not copy
        # the same words to both speakers; independent review handles overlap.
        if not self.resident:
            self.park(profile)
        detail = profile.seconds()
        # Parent totals overlap their children; only these leaves form a partition.
        leaves = [k for k in profile.values if k != "qwen_transcribe_seconds"]
        detail["main_other_seconds"] = round(max(0, time.perf_counter() - started
                                                      - sum(profile.values[k] for k in leaves)), 6)
        return {"rows": rows, "diarization": diarization, "warnings": warnings,
                "speaker_clustering": speaker_clustering,
                "duration_seconds": round(duration, 3), "windows": len(windows),
                "asr_batch_size": self.batch_size, "gpu_resident": self.resident,
                "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                "model_state_before": self.request_state, "model_state_after": self.state(),
                "profiling": profile.info(),
                "timings": {**detail, "qwen_seconds": round(asr_seconds, 3),
                            "diarization_seconds": round(time.perf_counter() - diar_started, 3),
                            "main_seconds": round(time.perf_counter() - started, 3)}}


def main():
    runner = Runner(text_only="--text-only" in sys.argv)
    emit({"type": "ready", **runner.ready_info()})
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if request.get("command") == "park":
                payload = runner.park(release_between_moves=bool(request.get("release_between_moves", False)))
            elif request.get("command") == "prepare_main":
                payload = runner.prepare_main()
            elif request.get("command") == "text":
                payload = runner.transcribe_text(request)
            else:
                payload = runner.transcribe(request)
            emit({"type": "result", "id": request["id"], **payload})
        except Exception as exc:
            import traceback
            traceback.print_exc()
            runner.park()
            emit({"type": "error", "id": request.get("id", ""), "error": str(exc)})


if __name__ == "__main__":
    main()
