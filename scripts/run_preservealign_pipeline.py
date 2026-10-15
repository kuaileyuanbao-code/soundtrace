"""Run the complete PreserveAlign inference pipeline for one WAV recording.

The neural front ends intentionally run in separate Python environments:
MOSS-Transcribe-Diarize produces speaker-attributed text on overlapping chunks,
while DiariZen produces meeting-level speaker activity. PreserveAlign then
reconciles chunk-local MOSS speakers with DiariZen's global speakers without
changing the recognized words or segment timestamps.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


STAGES = (
    "prepare_audio",
    "prepare_chunks",
    "moss_inference",
    "merge_chunks",
    "diarizen_inference",
    "preservealign",
    "validate_output",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Convert one WAV recording into speaker + timestamp + text SegLST "
            "with MOSS, DiariZen, and PreserveAlign."
        )
    )
    parser.add_argument("--input-wav", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument(
        "--utility-python",
        type=Path,
        default=repo_root / ".venv" / "Scripts" / "python.exe",
    )
    parser.add_argument(
        "--moss-python",
        type=Path,
        default=repo_root / ".venv_moss" / "Scripts" / "python.exe",
    )
    parser.add_argument(
        "--diarizen-python",
        type=Path,
        default=repo_root / ".venv_pyannote" / "Scripts" / "python.exe",
    )
    parser.add_argument(
        "--moss-model",
        type=Path,
        default=repo_root / "models" / "MOSS-Transcribe-Diarize",
    )
    parser.add_argument(
        "--diarizen-model",
        type=Path,
        default=repo_root / "models" / "diarizen-wavlm-large-s80-md-v2",
    )
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=(
            repo_root
            / "models"
            / "speaker-embedding-pyannote"
            / "pytorch_model.bin"
        ),
    )
    parser.add_argument("--channel-mode", choices=("max-rms", "mean", "channel"), default="max-rms")
    parser.add_argument("--channel", type=int, default=1, help="One-based channel for --channel-mode=channel.")
    parser.add_argument("--chunk-seconds", type=float, default=60.0)
    parser.add_argument("--overlap-seconds", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
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
        "--fallback",
        choices=("nearest", "overlap", "none"),
        default="nearest",
    )
    parser.add_argument(
        "--one-to-one",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use maximum-overlap one-to-one local/global speaker assignment.",
    )
    parser.add_argument(
        "--row-fallback-threshold",
        type=float,
        default=None,
        help="Optional mapping-purity threshold for row-level fallback.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--stop-after",
        choices=STAGES,
        default=None,
        help="Exit successfully after the selected stage (used by local orchestration).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command_text(command: list[object]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(payload, encoding="utf-8")
    try:
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 9:
                    # OneDrive may briefly lock rename/replace operations on Windows.
                    # A direct overwrite is safe here because this status file is
                    # diagnostic only and the pipeline has a single writer.
                    path.write_text(payload, encoding="utf-8")
                    return
                time.sleep(0.12 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def load_status(path: Path, input_wav: Path) -> dict:
    if path.exists():
        status = json.loads(path.read_text(encoding="utf-8"))
        previous_input = Path(status.get("input_wav", ""))
        if previous_input and previous_input != input_wav:
            raise SystemExit(
                f"Work directory belongs to another input: {previous_input}"
            )
        return status
    return {
        "pipeline": "MOSS + DiariZen + PreserveAlign",
        "input_wav": str(input_wav),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stages": {stage: {"state": "pending"} for stage in STAGES},
    }


def run_stage(
    name: str,
    command: list[object],
    expected_outputs: list[Path],
    status: dict,
    status_path: Path,
    repo_root: Path,
    resume: bool,
    dry_run: bool,
) -> None:
    stage = status["stages"][name]
    outputs_ready = all(path.exists() for path in expected_outputs)
    if resume and stage.get("state") == "completed" and outputs_ready:
        print(f"[{name}] already completed", flush=True)
        return

    print(f"[{name}] {command_text(command)}", flush=True)
    if dry_run:
        return

    stage.update(
        {
            "state": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": [str(item) for item in command],
        }
    )
    atomic_write_json(status_path, status)
    started = time.perf_counter()
    try:
        subprocess.run(
            [str(item) for item in command],
            cwd=repo_root,
            env={**os.environ, "PYTHONUTF8": "1"},
            check=True,
        )
    except Exception as error:
        stage.update(
            {
                "state": "failed",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": repr(error),
            }
        )
        atomic_write_json(status_path, status)
        raise

    missing = [str(path) for path in expected_outputs if not path.exists()]
    if missing:
        stage.update(
            {
                "state": "failed",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": f"Missing expected outputs: {missing}",
            }
        )
        atomic_write_json(status_path, status)
        raise RuntimeError(f"Stage {name} did not create expected outputs: {missing}")
    stage.update(
        {
            "state": "completed",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "outputs": [str(path) for path in expected_outputs],
        }
    )
    stage.pop("error", None)
    atomic_write_json(status_path, status)


def validate_seglst(path: Path, expected_session: str) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Final SegLST must be a JSON list.")
    if not rows:
        raise ValueError("Final SegLST is empty.")
    required = {"session_id", "speaker", "start_time", "end_time", "words"}
    speakers = set()
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Row {index} is missing fields: {sorted(missing)}")
        if str(row["session_id"]) != expected_session:
            raise ValueError(
                f"Row {index} has session {row['session_id']!r}, expected {expected_session!r}"
            )
        if float(row["end_time"]) <= float(row["start_time"]):
            raise ValueError(f"Row {index} has a non-positive duration.")
        if not str(row["words"]).strip():
            raise ValueError(f"Row {index} has empty words.")
        speakers.add(str(row["speaker"]))
    return {"rows": len(rows), "speakers": sorted(speakers)}


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_wav = args.input_wav.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()

    if not input_wav.is_file():
        raise SystemExit(f"Input WAV does not exist: {input_wav}")
    if input_wav.suffix.lower() != ".wav":
        raise SystemExit("--input-wav must point to a .wav file")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", input_wav.stem):
        raise SystemExit(
            "The WAV filename stem becomes session_id and may only contain "
            "letters, digits, dot, underscore, and hyphen."
        )
    if args.chunk_seconds <= 0:
        raise SystemExit("--chunk-seconds must be positive")
    if not 0 <= args.overlap_seconds < args.chunk_seconds:
        raise SystemExit("--overlap-seconds must be smaller than --chunk-seconds")
    if args.channel < 1:
        raise SystemExit("--channel must be at least 1")
    if args.min_speakers < 1 or args.max_speakers < args.min_speakers:
        raise SystemExit("Invalid speaker-count range")
    if args.row_fallback_threshold is not None and not (
        0 <= args.row_fallback_threshold <= 1
    ):
        raise SystemExit("--row-fallback-threshold must be between 0 and 1")

    required_paths = (
        args.utility_python,
        args.moss_python,
        args.diarizen_python,
        args.moss_model,
        args.diarizen_model,
        args.embedding_model,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing runtime paths: {missing}")

    session_id = input_wav.stem
    mono_dir = work_dir / "01_audio"
    chunks_dir = work_dir / "02_chunks"
    moss_dir = work_dir / "03_moss"
    diarizen_dir = work_dir / "04_diarizen"
    align_dir = work_dir / "05_preservealign"
    final_dir = work_dir / "06_final"
    mono_wav = mono_dir / f"{session_id}.wav"
    audio_manifest = mono_dir / "manifest.json"
    chunk_manifest = chunks_dir / "manifest.json"
    moss_chunks = moss_dir / "chunks.seglst.json"
    moss_merged = moss_dir / "meeting_local_speakers.seglst.json"
    diarizen_rttm = diarizen_dir / "rttm" / f"{session_id}.rttm"
    aligned = align_dir / "preservealign.seglst.json"
    final_output = final_dir / "speaker_transcript.seglst.json"
    report_path = final_dir / "run_summary.json"
    status_path = work_dir / "pipeline_status.json"

    if not args.dry_run:
        work_dir.mkdir(parents=True, exist_ok=True)
    status = load_status(status_path, input_wav)
    status["configuration"] = {
        "session_id": session_id,
        "channel_mode": args.channel_mode,
        "channel": args.channel,
        "chunk_seconds": args.chunk_seconds,
        "overlap_seconds": args.overlap_seconds,
        "moss_model": str(args.moss_model.resolve()),
        "diarizen_model": str(args.diarizen_model.resolve()),
        "embedding_model": str(args.embedding_model.resolve()),
        "clustering_method": args.clustering_method,
        "speaker_range": [args.min_speakers, args.max_speakers],
        "one_to_one": args.one_to_one,
        "fallback": args.fallback,
        "row_fallback_threshold": args.row_fallback_threshold,
    }
    if not args.dry_run:
        atomic_write_json(status_path, status)

    common_resume = ["--resume"] if args.resume else []
    run_stage(
        "prepare_audio",
        [
            args.utility_python,
            repo_root / "scripts" / "prepare_multichannel_wavs.py",
            "--input-dir",
            input_wav.parent,
            "--output-dir",
            mono_dir,
            "--pattern",
            input_wav.name,
            "--mode",
            args.channel_mode,
            "--channel",
            args.channel,
            "--strip-prefix-regex",
            "",
            "--manifest",
            audio_manifest,
            *common_resume,
        ],
        [mono_wav, audio_manifest],
        status,
        status_path,
        repo_root,
        args.resume,
        args.dry_run,
    )
    run_stage(
        "prepare_chunks",
        [
            args.utility_python,
            repo_root / "scripts" / "prepare_audio_chunks.py",
            "--input-dir",
            mono_dir,
            "--output-dir",
            chunks_dir,
            "--chunk-seconds",
            args.chunk_seconds,
            "--overlap-seconds",
            args.overlap_seconds,
            "--manifest",
            chunk_manifest,
            *common_resume,
        ],
        [chunk_manifest],
        status,
        status_path,
        repo_root,
        args.resume,
        args.dry_run,
    )
    if args.stop_after == "prepare_chunks":
        print("Stopped after prepare_chunks as requested.", flush=True)
        return
    run_stage(
        "moss_inference",
        [
            args.moss_python,
            repo_root / "scripts" / "infer_moss_transcribe_diarize.py",
            "--wav-dir",
            chunks_dir,
            "--output",
            moss_chunks,
            "--model",
            args.moss_model,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--max-new-tokens",
            args.max_new_tokens,
            *common_resume,
        ],
        [moss_chunks, moss_chunks.with_suffix(".metadata.json")],
        status,
        status_path,
        repo_root,
        args.resume,
        args.dry_run,
    )
    run_stage(
        "merge_chunks",
        [
            args.utility_python,
            repo_root / "scripts" / "merge_chunked_seglst.py",
            "--input",
            moss_chunks,
            "--chunk-manifest",
            chunk_manifest,
            "--output",
            moss_merged,
            "--speaker-mode",
            "unique-per-chunk",
        ],
        [moss_merged],
        status,
        status_path,
        repo_root,
        args.resume,
        args.dry_run,
    )
    run_stage(
        "diarizen_inference",
        [
            args.diarizen_python,
            repo_root / "scripts" / "infer_diarizen_diarization.py",
            "--wav-dir",
            mono_dir,
            "--output-dir",
            diarizen_dir,
            "--diarizen-model",
            args.diarizen_model,
            "--embedding-model",
            args.embedding_model,
            "--min-speakers",
            args.min_speakers,
            "--max-speakers",
            args.max_speakers,
            "--clustering-method",
            args.clustering_method,
            "--ahc-threshold",
            args.ahc_threshold,
            "--min-cluster-size",
            args.min_cluster_size,
            *common_resume,
        ],
        [diarizen_rttm],
        status,
        status_path,
        repo_root,
        args.resume,
        args.dry_run,
    )

    mapping_command: list[object] = [
        args.utility_python,
        repo_root / "scripts" / "map_source_speakers_to_rttm.py",
        "--input",
        moss_merged,
        "--rttm-dir",
        diarizen_dir / "rttm",
        "--output",
        aligned,
        "--fallback",
        args.fallback,
    ]
    if args.one_to_one:
        mapping_command.append("--one-to-one")
    if args.row_fallback_threshold is not None:
        mapping_command.extend(
            ["--row-fallback-threshold", args.row_fallback_threshold]
        )
    run_stage(
        "preservealign",
        mapping_command,
        [aligned],
        status,
        status_path,
        repo_root,
        args.resume,
        args.dry_run,
    )

    if args.dry_run:
        print(f"[validate_output] validate and copy to {final_output}")
        print("Dry run complete; no commands were executed.")
        return

    validation_stage = status["stages"]["validate_output"]
    if not (
        args.resume
        and validation_stage.get("state") == "completed"
        and final_output.exists()
        and report_path.exists()
    ):
        validation_stage.update(
            {
                "state": "running",
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )
        atomic_write_json(status_path, status)
        summary = validate_seglst(aligned, session_id)
        rows = json.loads(aligned.read_text(encoding="utf-8"))
        final_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(final_output, rows)
        run_summary = {
            "pipeline": "MOSS-Transcribe-Diarize + DiariZen + PreserveAlign",
            "input_wav": str(input_wav),
            "output_seglst": str(final_output),
            "session_id": session_id,
            **summary,
            "output_fields": [
                "session_id",
                "speaker",
                "start_time",
                "end_time",
                "words",
            ],
            "preserves_moss_words_and_timestamps": True,
        }
        atomic_write_json(report_path, run_summary)
        validation_stage.update(
            {
                "state": "completed",
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "outputs": [str(final_output), str(report_path)],
                **summary,
            }
        )
        atomic_write_json(status_path, status)

    print("Pipeline complete.")
    print(f"Final SegLST: {final_output}")
    print(f"Run summary:  {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Re-run with --resume to continue.", file=sys.stderr)
        raise SystemExit(130)
