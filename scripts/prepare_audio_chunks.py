import argparse
import json
from pathlib import Path

import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split long WAV recordings into bounded chunks with a time-offset manifest."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-seconds", type=float, default=60.0)
    parser.add_argument("--overlap-seconds", type=float, default=5.0)
    parser.add_argument("--limit-sessions", type=int, default=None)
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def main():
    args = parse_args()
    if args.chunk_seconds <= 0:
        raise SystemExit("--chunk-seconds must be positive")
    if args.overlap_seconds < 0 or args.overlap_seconds >= args.chunk_seconds:
        raise SystemExit("--overlap-seconds must be non-negative and smaller than the chunk")

    wav_paths = sorted(args.input_dir.glob("*.wav"))
    requested = set(args.sessions or [])
    if requested:
        wav_paths = [path for path in wav_paths if path.stem in requested]
        missing = requested - {path.stem for path in wav_paths}
        if missing:
            raise FileNotFoundError(f"Missing sessions: {sorted(missing)}")
    if args.limit_sessions is not None:
        wav_paths = wav_paths[: args.limit_sessions]
    if not wav_paths:
        raise SystemExit(f"No WAV files found under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.output_dir / "manifest.json"
    manifest = []
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = {row["chunk_id"] for row in manifest}
    produced = 0

    for wav_path in wav_paths:
        with sf.SoundFile(wav_path) as source:
            chunk_frames = max(1, round(args.chunk_seconds * source.samplerate))
            overlap_frames = round(args.overlap_seconds * source.samplerate)
            step_frames = chunk_frames - overlap_frames
            start_frame = 0
            chunk_index = 0

            while start_frame < source.frames:
                end_frame = min(source.frames, start_frame + chunk_frames)
                start_ms = round(1000 * start_frame / source.samplerate)
                end_ms = round(1000 * end_frame / source.samplerate)
                chunk_id = f"{wav_path.stem}__{start_ms:010d}__{end_ms:010d}"
                output_path = args.output_dir / f"{chunk_id}.wav"

                if not (args.resume and chunk_id in completed and output_path.exists()):
                    source.seek(start_frame)
                    samples = source.read(
                        end_frame - start_frame,
                        dtype="float32",
                        always_2d=True,
                    )
                    temp_path = output_path.with_suffix(".tmp.wav")
                    sf.write(
                        temp_path,
                        samples,
                        source.samplerate,
                        format="WAV",
                        subtype="PCM_16",
                    )
                    temp_path.replace(output_path)
                    manifest = [row for row in manifest if row["chunk_id"] != chunk_id]
                    manifest.append(
                        {
                            "chunk_id": chunk_id,
                            "chunk_index": chunk_index,
                            "session_id": wav_path.stem,
                            "source": str(wav_path.resolve()),
                            "path": str(output_path.resolve()),
                            "start_time": start_frame / source.samplerate,
                            "end_time": end_frame / source.samplerate,
                            "sample_rate": source.samplerate,
                        }
                    )
                    manifest.sort(key=lambda row: (row["session_id"], row["start_time"]))
                    atomic_write_json(manifest_path, manifest)
                    print(
                        f"{chunk_id}: {(end_frame - start_frame) / source.samplerate:.1f}s",
                        flush=True,
                    )

                produced += 1
                if args.limit_chunks is not None and produced >= args.limit_chunks:
                    print(f"Prepared {len(manifest)} chunk records in {args.output_dir}")
                    return
                if end_frame == source.frames:
                    break
                start_frame += step_frames
                chunk_index += 1

    print(f"Prepared {len(manifest)} chunk records in {args.output_dir}")


if __name__ == "__main__":
    main()
