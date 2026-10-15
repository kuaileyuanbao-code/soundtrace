"""Cross-session CAM++ voice prototypes on top of predicted diarization turns."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from .io import read_json
from .merge import make_overlap_seglst, make_seglst


def _robust_prototype(
    embeddings: np.ndarray,
    durations: np.ndarray,
    trim_fraction: float,
    duration_power: float,
) -> np.ndarray:
    """Build a normalized centroid while optionally trimming acoustic outliers."""
    if not 0.0 <= trim_fraction < 1.0:
        raise ValueError("prototype trim fraction must satisfy 0 <= value < 1")
    weights = np.maximum(durations, 1e-3) ** duration_power
    centroid = np.average(embeddings, axis=0, weights=weights)
    centroid /= max(np.linalg.norm(centroid), 1e-8)
    if trim_fraction > 0 and len(embeddings) > 1:
        keep = max(1, int(np.ceil(len(embeddings) * (1.0 - trim_fraction))))
        selected = np.argsort(embeddings @ centroid)[-keep:]
        centroid = np.average(
            embeddings[selected],
            axis=0,
            weights=weights[selected],
        )
        centroid /= max(np.linalg.norm(centroid), 1e-8)
    return centroid


def _enforce_overlap_cannot_link(
    turns: list[tuple[str, int, float, float, int]],
    emissions: np.ndarray,
    assigned: np.ndarray,
    min_overlap: float,
    max_reassignment_loss: float,
) -> np.ndarray:
    """Prevent sufficiently overlapping turns from sharing a global speaker."""
    if min_overlap <= 0 or emissions.shape[1] < 2:
        return assigned

    assigned = assigned.copy()
    by_session: dict[str, list[int]] = defaultdict(list)
    for index, (session_id, _, _, _, _) in enumerate(turns):
        by_session[session_id].append(index)

    for indices in by_session.values():
        indices.sort(key=lambda index: turns[index][2])
        for _ in range(3):
            changed = False
            for left_position, left in enumerate(indices):
                left_end = turns[left][3]
                for right in indices[left_position + 1 :]:
                    right_start = turns[right][2]
                    if right_start >= left_end:
                        break
                    overlap = min(left_end, turns[right][3]) - max(
                        turns[left][2], right_start
                    )
                    if overlap < min_overlap or assigned[left] != assigned[right]:
                        continue

                    current = int(assigned[left])
                    alternatives = [label for label in range(emissions.shape[1]) if label != current]
                    if not alternatives:
                        continue
                    left_alt = max(alternatives, key=lambda label: emissions[left, label])
                    right_alt = max(alternatives, key=lambda label: emissions[right, label])
                    left_loss = emissions[left, current] - emissions[left, left_alt]
                    right_loss = emissions[right, current] - emissions[right, right_alt]
                    if min(left_loss, right_loss) > max_reassignment_loss:
                        continue
                    if left_loss <= right_loss:
                        assigned[left] = left_alt
                    else:
                        assigned[right] = right_alt
                    changed = True
            if not changed:
                break
    return assigned


def _smooth_local_tracks(
    turns: list[tuple[str, int, float, float, int]],
    emissions: np.ndarray,
    transition_penalty: float,
    transition_tau: float,
) -> np.ndarray:
    """Use short-range Sortformer track continuity without forcing whole-track identity."""
    assigned = emissions.argmax(axis=1)
    if transition_penalty <= 0:
        return assigned

    tracks: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (session_id, _, _, _, local_speaker) in enumerate(turns):
        tracks[(session_id, local_speaker)].append(index)

    for indices in tracks.values():
        indices.sort(key=lambda index: turns[index][2])
        scores = emissions[indices[0]].copy()
        backpointers = []
        for previous, current in zip(indices, indices[1:]):
            gap = max(0.0, turns[current][2] - turns[previous][3])
            penalty = transition_penalty * np.exp(-gap / max(transition_tau, 1e-6))
            best_previous = int(np.argmax(scores))
            updated = np.empty_like(scores)
            backpointer = np.empty(len(scores), dtype=np.int32)
            for state in range(len(scores)):
                stay = scores[state]
                change = scores[best_previous] - penalty
                if stay >= change:
                    updated[state] = stay + emissions[current, state]
                    backpointer[state] = state
                else:
                    updated[state] = change + emissions[current, state]
                    backpointer[state] = best_previous
            scores = updated
            backpointers.append(backpointer)

        state = int(np.argmax(scores))
        path = [state]
        for backpointer in reversed(backpointers):
            state = int(backpointer[state])
            path.append(state)
        for index, state in zip(indices, reversed(path)):
            assigned[index] = state
    return assigned


def _embed_turns(model, feature_extractor, wav: torch.Tensor, turns, device):
    from speakerlab.utils.utils import circle_pad

    clips = []
    for start, end, _ in turns:
        left = max(0, int(float(start) * 16000))
        right = min(wav.shape[-1], int(float(end) * 16000))
        clips.append(wav[0, left : max(left + 1600, right)])
    output = []
    with torch.inference_mode():
        for offset in range(0, len(clips), 64):
            batch_clips = clips[offset : offset + 64]
            max_length = max(1600, max(clip.shape[0] for clip in batch_clips))
            batch = torch.stack(
                [circle_pad(clip, max_length) for clip in batch_clips]
            ).unsqueeze(1)
            features = torch.vmap(feature_extractor)(batch.to(device))
            output.append(model(features).cpu().numpy())
    return np.concatenate(output)


def _load_campplus(device: torch.device, cache_dir: str | Path):
    """Load only CAM++ without importing the full diarization dependency stack."""
    from speakerlab.models.campplus.DTDNN import CAMPPlus
    from speakerlab.process.processor import FBank
    from speakerlab.utils.utils import download_model_from_modelscope

    model_cache = download_model_from_modelscope(
        "iic/speech_campplus_sv_zh_en_16k-common_advanced",
        "v1.0.0",
        str(cache_dir),
    )
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    state = torch.load(
        str(Path(model_cache) / "campplus_cn_en_common.pt"),
        map_location="cpu",
    )
    model.load_state_dict(state)
    model.eval().to(device)
    feature_extractor = FBank(
        n_mels=80,
        sample_rate=16000,
        mean_nor=True,
    )
    return model, feature_extractor


def _load_audio_16k(path: str | Path) -> torch.Tensor:
    """Load competition WAV files without depending on FFmpeg or TorchCodec."""
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sample_rate}: {path}")
    mono = audio.mean(axis=1)
    return torch.from_numpy(mono).unsqueeze(0)


def build_global_proto_seglst(
    paths: list[Path],
    diar_cache: str | Path,
    asr_cache: str | Path,
    embedding_cache: str | Path,
    model_dir: str | Path,
    device: str = "cuda:0",
    threshold: float = 0.561,
    min_duration: float = 2.1,
    track_transition_penalty: float = 0.44,
    track_transition_tau: float = 0.3,
    overlap_min_turn_duration: float = 0.65,
    overlap_min_region_duration: float = 0.0,
    overlap_cannot_link_min_duration: float = 0.0,
    overlap_cannot_link_max_loss: float = 0.15,
    prototype_trim_fraction: float = 0.0,
    prototype_duration_power: float = 0.0,
    third_party_3dspeaker: str | Path | None = None,
) -> tuple[list[dict], dict[str, list[list[float]]]]:
    """Extract turns, cluster long-turn prototypes globally, and assign every turn."""

    # These imports are intentionally delayed: normal ASR/diarization users do not
    # pay the 3D-Speaker import cost.
    import sys

    third_party = (
        Path(third_party_3dspeaker)
        if third_party_3dspeaker is not None
        else Path(__file__).resolve().parents[2] / "third_party" / "3D-Speaker"
    )
    if not (third_party / "speakerlab").is_dir():
        raise FileNotFoundError(f"3D-Speaker source directory is missing: {third_party}")
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))

    embedding_cache = Path(embedding_cache)
    embedding_cache.mkdir(parents=True, exist_ok=True)
    model = None
    feature_extractor = None
    torch_device = torch.device(device)
    all_turns: list[tuple[str, int, float, float, int]] = []
    all_embeddings = []
    for path in paths:
        turns = read_json(Path(diar_cache) / f"{path.stem}.json")
        turns = [
            [float(start), float(end), int(speaker)]
            for start, end, speaker in turns
            if float(end) > float(start)
        ]
        if not turns:
            continue
        cache = embedding_cache / f"{path.stem}.npy"
        if cache.exists():
            embeddings = np.load(cache)
        else:
            if model is None:
                model, feature_extractor = _load_campplus(torch_device, model_dir)
            embeddings = _embed_turns(
                model,
                feature_extractor,
                _load_audio_16k(path),
                turns,
                torch_device,
            )
            np.save(cache, embeddings)
        all_turns.extend(
            (path.stem, index, start, end, speaker)
            for index, (start, end, speaker) in enumerate(turns)
        )
        all_embeddings.append(embeddings)

    if not all_embeddings:
        raise RuntimeError("说话人日志为空，无法构造全局声纹原型")
    matrix = np.concatenate(all_embeddings)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    train_mask = np.asarray(
        [end - start >= min_duration for _, _, start, end, _ in all_turns]
    )
    train = matrix[train_mask]
    if len(train) < 2:
        raise RuntimeError("有效长说话人片段少于两个，无法聚类")
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    ).fit_predict(train)
    train_durations = np.asarray(
        [end - start for _, _, start, end, _ in all_turns]
    )[train_mask]
    prototypes = []
    for label in sorted(set(labels)):
        selected = labels == label
        prototypes.append(
            _robust_prototype(
                train[selected],
                train_durations[selected],
                prototype_trim_fraction,
                prototype_duration_power,
            )
        )
    emissions = matrix @ np.asarray(prototypes).T
    assigned = _smooth_local_tracks(
        all_turns, emissions, track_transition_penalty, track_transition_tau
    )
    assigned = _enforce_overlap_cannot_link(
        all_turns,
        emissions,
        assigned,
        overlap_cannot_link_min_duration,
        overlap_cannot_link_max_loss,
    )

    diarization: dict[str, list[list[float]]] = defaultdict(list)
    for (session_id, _, start, end, _), label in zip(all_turns, assigned):
        diarization[session_id].append([start, end, int(label)])

    output = []
    for path in paths:
        asr_result = read_json(Path(asr_cache) / f"{path.stem}.json")
        if overlap_min_turn_duration > 0:
            output.extend(
                make_overlap_seglst(
                    path.stem,
                    asr_result,
                    diarization[path.stem],
                    overlap_min_turn_duration,
                    overlap_min_region_duration,
                )
            )
        else:
            output.extend(make_seglst(path.stem, asr_result, diarization[path.stem]))
    return output, dict(diarization)
