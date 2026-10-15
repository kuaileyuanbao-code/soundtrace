"""Persistent JSON-line worker for DiariZen global speaker diarization."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from infer_diarizen_diarization import (  # noqa: E402
    normalized_segments,
    prepare_legacy_runtime,
    write_rttm,
)


PREFIX = "__SOUNDTRACE_DIARIZEN__"


def emit(value: dict) -> None:
    print(PREFIX + json.dumps(value, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diarizen-model", required=True, type=Path)
    parser.add_argument("--embedding-model", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    prepare_legacy_runtime(ROOT)
    from diarizen.pipelines.inference import DiariZenPipeline

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
                "method": "VBxClustering",
                "min_speakers": 1,
                "max_speakers": 6,
                "ahc_criterion": "distance",
                "ahc_threshold": 0.6,
                "min_cluster_size": 13,
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
    emit({"type": "ready", "batch_size": args.batch_size})

    for line in sys.stdin:
        request_id = ""
        try:
            request = json.loads(line)
            request_id = str(request["id"])
            wav_path = Path(request["wav"])
            output_path = Path(request["output"])
            min_speakers = int(request.get("min_speakers", 1))
            max_speakers = int(request.get("max_speakers", 6))
            pipeline.min_speakers = min_speakers
            pipeline.max_speakers = max_speakers
            started = time.perf_counter()
            annotation = pipeline(str(wav_path), sess_name=wav_path.stem)
            segments = normalized_segments(annotation)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_rttm(output_path, wav_path.stem, segments)
            emit(
                {
                    "type": "result",
                    "id": request_id,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "speaker_count": len({speaker for _, _, speaker in segments}),
                    "segments": len(segments),
                }
            )
        except Exception as exc:
            emit({"type": "error", "id": request_id, "error": str(exc)})


if __name__ == "__main__":
    main()
