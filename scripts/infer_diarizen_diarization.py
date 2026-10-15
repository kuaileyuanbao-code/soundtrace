"""Run the official DiariZen diarization model and write one RTTM per WAV.

The DiariZen v2 checkpoint is CC BY-NC 4.0. This script is intended for local
development-set evaluation. Confirm competition/licensing eligibility before
using model outputs in a prize-bearing submission.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def prepare_legacy_runtime(repo_root: Path) -> None:
    """Bridge DiariZen's pyannote 3.1 code to the local modern CUDA runtime."""
    diarizen_root = repo_root / "vendor" / "DiariZen"
    bundled_pyannote = diarizen_root / "pyannote-audio"
    sys.path[:0] = [str(diarizen_root), str(bundled_pyannote)]

    import numpy as np

    # Removed in NumPy 2, but referenced by bundled pyannote.audio 3.1.
    if not hasattr(np, "NaN"):
        np.NaN = np.nan
    if not hasattr(np, "NAN"):
        np.NAN = np.nan

    import soundfile as sf
    import torch
    import torchaudio

    # PyTorch 2.6+ changed the default to weights_only=True. These checkpoints
    # are downloaded from the two official Hugging Face repositories.
    original_torch_load = torch.load

    def trusted_checkpoint_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_checkpoint_load

    # torchaudio 2.11 delegates file IO to TorchCodec. SoundFile already reads
    # the competition's PCM WAV input and avoids an unnecessary FFmpeg runtime.
    def soundfile_load(path, *args, **kwargs):
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(samples.T.copy()), sample_rate

    torchaudio.load = soundfile_load

    # The bundled pyannote.audio package shares namespace modules (core,
    # database, metrics) with the installed pyannote packages.
    import pyannote

    installed_pyannote = repo_root / ".venv_pyannote" / "Lib" / "site-packages" / "pyannote"
    if installed_pyannote.is_dir() and str(installed_pyannote) not in pyannote.__path__:
        pyannote.__path__.append(str(installed_pyannote))


def normalized_segments(annotation):
    label_map = {}
    result = []
    for turn, _, label in annotation.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        if end <= start:
            continue
        if label not in label_map:
            label_map[label] = len(label_map) + 1
        result.append((start, end, label_map[label]))
    return sorted(result)


def write_rttm(path: Path, session_id: str, segments) -> None:
    lines = [
        f"SPEAKER {session_id} 1 {start:.3f} {end - start:.3f} "
        f"<NA> <NA> {speaker} <NA> <NA>"
        for start, end, speaker in segments
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local DiariZen diarization.")
    parser.add_argument("--wav-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--diarizen-model",
        type=Path,
        default=Path("models/diarizen-wavlm-large-s80-md-v2"),
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=Path("models/speaker-embedding-pyannote/pytorch_model.bin"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=6)
    parser.add_argument(
        "--clustering-method",
        choices=("VBxClustering", "AgglomerativeClustering"),
        default="VBxClustering",
    )
    parser.add_argument("--ahc-threshold", type=float, default=0.6)
    parser.add_argument("--min-cluster-size", type=int, default=13)
    parser.add_argument(
        "--sessions",
        help="Comma-separated session ids to run, for example 001,009,042.",
    )
    parser.add_argument(
        "--speaker-counts-seglst",
        type=Path,
        help="Use per-session speaker counts predicted by another system.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    prepare_legacy_runtime(repo_root)

    from diarizen.pipelines.inference import DiariZenPipeline

    output_dir = args.output_dir.resolve()
    rttm_dir = output_dir / "rttm"
    metadata_dir = output_dir / "metadata"
    rttm_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "inference": {
            "args": {
                "seg_duration": 16,
                "segmentation_step": 0.1,
                "batch_size": args.batch_size,
                "apply_median_filtering": True,
            }
        },
        "clustering": {
            "args": {
                "method": args.clustering_method,
                "min_speakers": args.min_speakers,
                "max_speakers": args.max_speakers,
                "ahc_criterion": "distance",
                "ahc_threshold": args.ahc_threshold,
                "min_cluster_size": args.min_cluster_size,
                "Fa": 0.07,
                "Fb": 0.8,
                "lda_dim": 128,
                "max_iters": 20,
            }
        },
    }

    pipeline = DiariZenPipeline(
        diarizen_hub=args.diarizen_model.resolve(),
        embedding_model=str(args.embedding_model.resolve()),
        config_parse=config,
    )

    speaker_counts = {}
    if args.speaker_counts_seglst:
        rows = json.loads(args.speaker_counts_seglst.read_text(encoding="utf-8"))
        labels = {}
        for row in rows:
            labels.setdefault(row["session_id"], set()).add(row["speaker"])
        speaker_counts = {session_id: len(value) for session_id, value in labels.items()}

    wavs = sorted(args.wav_dir.resolve().glob("*.wav"))
    if args.sessions:
        selected = {item.strip() for item in args.sessions.split(",") if item.strip()}
        wavs = [wav for wav in wavs if wav.stem in selected]
    if args.limit:
        wavs = wavs[: args.limit]

    failures = []
    completed = 0
    for index, wav_path in enumerate(wavs, 1):
        session_id = wav_path.stem
        rttm_path = rttm_dir / f"{session_id}.rttm"
        metadata_path = metadata_dir / f"{session_id}.json"
        if args.resume and rttm_path.exists() and metadata_path.exists():
            print(f"[{index}/{len(wavs)}] {session_id} skip", flush=True)
            completed += 1
            continue

        print(f"[{index}/{len(wavs)}] {session_id}", flush=True)
        started = time.perf_counter()
        try:
            forced_count = speaker_counts.get(session_id)
            if forced_count:
                pipeline.min_speakers = forced_count
                pipeline.max_speakers = forced_count
            else:
                pipeline.min_speakers = args.min_speakers
                pipeline.max_speakers = args.max_speakers
            annotation = pipeline(str(wav_path), sess_name=session_id)
            segments = normalized_segments(annotation)
            write_rttm(rttm_path, session_id, segments)
            speakers = {speaker for _, _, speaker in segments}
            metadata = {
                "session_id": session_id,
                "speaker_count": len(speakers),
                "segments": len(segments),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "min_speakers": args.min_speakers,
                "max_speakers": args.max_speakers,
                "ahc_threshold": args.ahc_threshold,
                "forced_speaker_count": forced_count,
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            completed += 1
        except Exception as exc:
            failures.append({"session_id": session_id, "error": repr(exc)})
            print(f"ERROR {session_id}: {exc}", flush=True)

    (output_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Completed {completed}/{len(wavs)} sessions", flush=True)


if __name__ == "__main__":
    main()
