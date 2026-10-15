"""Persistent JSON-line worker for short MOSS conversations.

This process runs inside .venv_moss and keeps the 0.9B model loaded between
recordings. Protocol messages are prefixed so third-party progress output can
share stdout without being mistaken for responses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BOOT_STARTED = time.perf_counter()
from performance import StageTimings, model_state, observe_call


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from infer_moss_transcribe_diarize import (  # noqa: E402
    DEFAULT_PROMPT,
    ModelRunner,
    normalize_speaker,
    normalize_tokens,
    subtitle_segments_from_transcript,
)


PREFIX = "__SOUNDTRACE_MOSS__"


def emit(value: dict) -> None:
    print(PREFIX + json.dumps(value, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    runner = ModelRunner(args.model, device=args.device, dtype=args.dtype)
    # ModelRunner is lazy: ready is process readiness, not loaded weights.
    emit({"type": "ready", "worker_boot_seconds": round(time.perf_counter() - BOOT_STARTED, 6),
          "model_state_after": model_state(runner._model)})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = str(request["id"])
            wav_path = Path(request["wav"])
            max_new_tokens = int(request.get("max_new_tokens", 2048))
            started = time.perf_counter()
            profile = StageTimings()
            state_before = model_state(runner._model)
            runner_module = sys.modules[ModelRunner.__module__]
            with observe_call(runner, "_ensure_loaded", profile, "moss_load_and_place_seconds", gpu=True), \
                 observe_call(runner_module, "generate_transcription", profile, "moss_inference_seconds", gpu=True):
                result = runner.transcribe(
                    wav_path,
                    prompt=request.get("prompt") or DEFAULT_PROMPT,
                    max_new_tokens=max_new_tokens,
                    decoding="greedy",
                )
            if result.generated_tokens >= max_new_tokens:
                raise RuntimeError(f"Generation hit the {max_new_tokens}-token limit")
            rows = []
            for segment in subtitle_segments_from_transcript(result.text, postprocess=False):
                tokens = normalize_tokens(segment.text)
                if not tokens or segment.end <= segment.start:
                    continue
                rows.append(
                    {
                        "end_time": round(float(segment.end), 2),
                        "session_id": wav_path.stem,
                        "speaker": normalize_speaker(segment.speaker),
                        "start_time": round(float(segment.start), 2),
                        "words": " ".join(tokens),
                    }
                )
            emit(
                {
                    "type": "result",
                    "id": request_id,
                    "rows": rows,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "generated_tokens": result.generated_tokens,
                    "raw_transcript": result.text,
                    "timings": {**profile.seconds(), "moss_other_seconds": round(max(0,
                        time.perf_counter() - started - sum(profile.values.values())), 6)},
                    "model_state_before": state_before, "model_state_after": model_state(runner._model),
                    "profiling": profile.info(),
                }
            )
        except Exception as exc:
            emit(
                {
                    "type": "error",
                    "id": str(locals().get("request_id", "")),
                    "error": str(exc),
                }
            )


if __name__ == "__main__":
    main()
