import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert multi-channel recordings into mono WAV files for ASR inference."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*.flac")
    parser.add_argument(
        "--mode",
        choices=["channel", "max-rms", "mean"],
        default="max-rms",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=1,
        help="One-based channel used when --mode=channel.",
    )
    parser.add_argument(
        "--strip-prefix-regex",
        default=r"^\d{8}_",
        help="Regex removed from each input stem to form the output session id.",
    )
    parser.add_argument("--block-frames", type=int, default=262144)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def choose_max_rms_channel(path, block_frames):
    energy = None
    total_frames = 0
    with sf.SoundFile(path) as source:
        for block in source.blocks(
            blocksize=block_frames,
            dtype="float32",
            always_2d=True,
        ):
            block_energy = np.sum(block.astype(np.float64) ** 2, axis=0)
            energy = block_energy if energy is None else energy + block_energy
            total_frames += len(block)
    if energy is None or total_frames == 0:
        raise ValueError(f"Empty audio file: {path}")
    return int(np.argmax(energy))


def convert_file(path, output_path, mode, channel, block_frames):
    info = sf.info(path)
    if info.channels < 1:
        raise ValueError(f"Audio has no channels: {path}")

    if mode == "channel":
        selected_channel = channel - 1
        if selected_channel < 0 or selected_channel >= info.channels:
            raise ValueError(
                f"Requested channel {channel}, but {path} has {info.channels} channels"
            )
    elif mode == "max-rms":
        selected_channel = choose_max_rms_channel(path, block_frames)
    else:
        selected_channel = None

    temp_path = output_path.with_suffix(".tmp.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(path) as source, sf.SoundFile(
        temp_path,
        mode="w",
        samplerate=source.samplerate,
        channels=1,
        format="WAV",
        subtype="PCM_16",
    ) as target:
        for block in source.blocks(
            blocksize=block_frames,
            dtype="float32",
            always_2d=True,
        ):
            if selected_channel is None:
                mono = np.mean(block, axis=1, keepdims=True)
            else:
                mono = block[:, selected_channel : selected_channel + 1]
            target.write(np.clip(mono, -1.0, 1.0))
    temp_path.replace(output_path)

    return {
        "channels": info.channels,
        "frames": info.frames,
        "sample_rate": info.samplerate,
        "selected_channel": None if selected_channel is None else selected_channel + 1,
    }


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
    if args.channel < 1:
        raise SystemExit("--channel must be at least 1")
    if args.block_frames < 1:
        raise SystemExit("--block-frames must be at least 1")

    paths = sorted(path for path in args.input_dir.rglob(args.pattern) if path.is_file())
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No files matching {args.pattern!r} under {args.input_dir}")

    prefix_pattern = re.compile(args.strip_prefix_regex) if args.strip_prefix_regex else None
    manifest_path = args.manifest or args.output_dir / "manifest.json"
    manifest = []
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = {row["session_id"] for row in manifest}
    seen = set(completed)

    for index, path in enumerate(paths, 1):
        session_id = path.stem
        if prefix_pattern:
            session_id = prefix_pattern.sub("", session_id, count=1)
        if not session_id:
            raise ValueError(f"Empty session id after prefix removal: {path}")
        if session_id in seen and session_id not in completed:
            raise ValueError(f"Duplicate output session id: {session_id}")
        seen.add(session_id)

        output_path = args.output_dir / f"{session_id}.wav"
        if args.resume and session_id in completed and output_path.exists():
            print(f"[{index}/{len(paths)}] skip {session_id}", flush=True)
            continue

        details = convert_file(
            path,
            output_path,
            args.mode,
            args.channel,
            args.block_frames,
        )
        manifest = [row for row in manifest if row["session_id"] != session_id]
        manifest.append(
            {
                "session_id": session_id,
                "input": str(path.resolve()),
                "output": str(output_path.resolve()),
                "mode": args.mode,
                **details,
            }
        )
        manifest.sort(key=lambda row: row["session_id"])
        atomic_write_json(manifest_path, manifest)
        print(
            f"[{index}/{len(paths)}] {session_id}: {details['channels']}ch -> mono",
            flush=True,
        )

    print(f"Prepared {len(manifest)} recordings in {args.output_dir}")


if __name__ == "__main__":
    main()
