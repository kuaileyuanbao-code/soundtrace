import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

SCRIPT_STARTED = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from performance import StageTimings, model_state, observe_call


def atomic_write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_sessions(value):
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main():
    parser = argparse.ArgumentParser(
        description="Run FireRedASR2-AED and retain confidence and token timestamps."
    )
    parser.add_argument("--repo-dir", type=Path, default=Path("packages/FireRedASR2S"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/FireRedASR2-AED"))
    parser.add_argument("--wav-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sessions", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--full-precision", action="store_true")
    parser.add_argument("--timings-output", type=Path, default=None)
    args = parser.parse_args()
    profile = StageTimings()

    repo_dir = args.repo_dir.resolve()
    sys.path.insert(0, str(repo_dir))

    with profile.measure("firered_dependency_import_seconds"):
        import torch
        import torchaudio
        # Compatibility switch for diagnosis/rollback; same AED model and
        # parameters either way, and the process still exits after each task.
        eager_import = os.environ.get("SOUNDTRACE_FIRERED_EAGER_IMPORT", "0") == "1"
        if eager_import:
            import fireredasr2s.fireredasr2system
            import fireredasr2s.fireredasr2.models.fireredasr_llm
            import fireredasr2s.fireredasr2.tokenizer.llm_tokenizer
        from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
        # Previously installed by the eagerly imported full-system module.
        logging.basicConfig(level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    original_forced_align = torchaudio.functional.forced_align

    def forced_align_compat(log_probs, *positional, **keywords):
        if log_probs.dtype == torch.bfloat16:
            log_probs = log_probs.float()
        return original_forced_align(log_probs, *positional, **keywords)

    torchaudio.functional.forced_align = forced_align_compat

    selected = parse_sessions(args.sessions)
    wav_paths = sorted(args.wav_dir.glob("*.wav"))
    if selected is not None:
        wav_paths = [path for path in wav_paths if path.stem in selected]
    if args.max_files is not None:
        wav_paths = wav_paths[: args.max_files]
    if not wav_paths:
        raise ValueError("No WAV files selected")

    completed = {}
    if args.output.exists():
        for item in json.loads(args.output.read_text(encoding="utf-8-sig")):
            completed[str(item["session_id"])] = item

    config = FireRedAsr2Config(
        use_gpu=not args.cpu,
        use_half=not args.full_precision,
        beam_size=args.beam_size,
        nbest=1,
        decode_max_len=300,
        softmax_smoothing=1.25,
        aed_length_penalty=0.6,
        eos_penalty=1.0,
        return_timestamp=True,
    )
    with observe_call(torch.nn.Module, "cuda", profile, "firered_weights_to_gpu_seconds", gpu=True):
        with profile.measure("firered_load_and_place_seconds", gpu=True):
            model = FireRedAsr2.from_pretrained("aed", str(args.model_dir), config)
    state_after_load = model_state(model.model)

    pending = [path for path in wav_paths if path.stem not in completed]
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        with profile.measure("firered_inference_seconds", gpu=True):
            results = model.transcribe(
                [path.stem for path in batch],
                [str(path) for path in batch],
            )
        for result in results:
            session_id = str(result["uttid"])
            timestamps = [
                {
                    "token": str(token),
                    "start_time": round(float(start), 3),
                    "end_time": round(float(end), 3),
                }
                for token, start, end in result.get("timestamp", [])
            ]
            completed[session_id] = {
                "session_id": session_id,
                "text": str(result.get("text", "")),
                "confidence": float(result.get("confidence", 0.0)),
                "duration": float(result.get("dur_s", 0.0)),
                "rtf": float(result.get("rtf", 0.0)),
                "timestamps": timestamps,
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output, [completed[key] for key in sorted(completed)])
        print(f"completed={len(completed)}/{len(wav_paths)}", flush=True)
    if args.timings_output is not None:
        total = time.perf_counter() - SCRIPT_STARTED
        leaves = [k for k in profile.values if k != "firered_weights_to_gpu_seconds"]
        details = {**profile.seconds(), "firered_script_seconds": round(total, 6),
                   "firered_other_seconds": round(max(0, total - sum(profile.values[k] for k in leaves)), 6)}
        atomic_write_json(args.timings_output, {"timings": details, "profiling": profile.info(),
            "import_strategy": "eager_compatibility" if eager_import else "aed_only",
            "dependency_observation": {name: name in sys.modules for name in (
                "transformers", "fireredasr2s.fireredasr2system",
                "fireredasr2s.fireredasr2.models.fireredasr_llm")},
            "model_state_before": model_state(None), "model_state_after_load": state_after_load,
            "model_state_after": model_state(model.model)})


if __name__ == "__main__":
    main()
