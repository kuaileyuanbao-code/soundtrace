import argparse
import json
import re
import time
import wave
from pathlib import Path

from moss_transcribe_diarize.app.model_runner import ModelRunner
from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
from moss_transcribe_diarize.subtitle import subtitle_segments_from_transcript


CN_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.I)


def normalize_tokens(text):
    text = (text or "").lower()
    tokens = []
    index = 0
    while index < len(text):
        character = text[index]
        if CN_RE.match(character):
            tokens.append(character)
            index += 1
            continue
        match = ASCII_WORD_RE.match(text, index)
        if match:
            tokens.append(match.group(0))
            index = match.end()
            continue
        index += 1
    return tokens


def normalize_speaker(label):
    label = str(label).lower()
    match = re.fullmatch(r"(?:s|spk)0*(\d+)", label)
    if match:
        return f"spk{int(match.group(1))}"
    return label.replace("s", "spk", 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MOSS-Transcribe-Diarize once for a directory of WAV files and emit SegLST."
    )
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--allow-token-limit",
        action="store_true",
        help="Keep outputs that consume the full generation budget instead of failing.",
    )
    parser.add_argument(
        "--retry-token-limit-with-silence-padding-seconds",
        type=float,
        default=None,
        help=(
            "For a shorter WAV that hits the token limit, right-pad with silence "
            "to this duration and retry once before failing."
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from an existing output/metadata pair and skip completed sessions.",
    )
    return parser.parse_args()


def load_json_list(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return value


def right_pad_wav(input_path, output_path, target_seconds):
    with wave.open(str(input_path), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())
    target_frames = round(target_seconds * params.framerate)
    if params.nframes >= target_frames:
        return None
    silence_frames = target_frames - params.nframes
    silence = b"\x00" * silence_frames * params.nchannels * params.sampwidth
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames + silence)
    return output_path


def main():
    args = parse_args()
    requested = set(args.sessions or [])
    wav_paths = sorted(args.wav_dir.glob("*.wav"))
    if requested:
        wav_paths = [path for path in wav_paths if path.stem in requested]
    if not wav_paths:
        raise SystemExit("No matching WAV files found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(".metadata.json")
    rows = []
    metadata = []
    completed = set()
    if args.resume and (args.output.exists() or metadata_path.exists()):
        if not args.output.exists() or not metadata_path.exists():
            raise SystemExit(
                "Resume requires both the SegLST output and metadata files to exist."
            )
        rows = load_json_list(args.output)
        metadata = load_json_list(metadata_path)
        completed = {
            item["session_id"]
            for item in metadata
            if isinstance(item, dict) and item.get("session_id")
        }
        selected = {path.stem for path in wav_paths}
        completed &= selected
        rows = [row for row in rows if row.get("session_id") in completed]
        metadata = [
            item for item in metadata if item.get("session_id") in completed
        ]
        wav_paths = [path for path in wav_paths if path.stem not in completed]
        print(
            f"Resuming with {len(completed)} completed sessions and "
            f"{len(wav_paths)} remaining.",
            flush=True,
        )

    if not wav_paths:
        print(f"Nothing to do; all requested sessions are already in {args.output}.")
        return

    runner = ModelRunner(args.model, device=args.device, dtype=args.dtype)
    started = time.time()
    total_sessions = len(completed) + len(wav_paths)

    for index, wav_path in enumerate(wav_paths, start=len(completed) + 1):
        session_id = wav_path.stem
        result = runner.transcribe(
            wav_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            decoding="greedy",
        )
        hit_token_limit = result.generated_tokens >= args.max_new_tokens
        retried_with_silence_padding = False
        if (
            hit_token_limit
            and not args.allow_token_limit
            and args.retry_token_limit_with_silence_padding_seconds is not None
        ):
            padded_path = right_pad_wav(
                wav_path,
                args.output.parent
                / f"{args.output.stem}.retry_padded"
                / wav_path.name,
                args.retry_token_limit_with_silence_padding_seconds,
            )
            if padded_path is not None:
                retried_with_silence_padding = True
                result = runner.transcribe(
                    padded_path,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    decoding="greedy",
                )
                hit_token_limit = result.generated_tokens >= args.max_new_tokens
        if hit_token_limit and not args.allow_token_limit:
            raise RuntimeError(
                f"Generation hit the {args.max_new_tokens}-token limit for {session_id}; "
                "increase --max-new-tokens or pass --allow-token-limit explicitly"
            )
        segments = subtitle_segments_from_transcript(result.text, postprocess=False)
        session_rows = []
        for segment in segments:
            tokens = normalize_tokens(segment.text)
            if not tokens or segment.end <= segment.start:
                continue
            session_rows.append(
                {
                    "end_time": round(float(segment.end), 2),
                    "session_id": session_id,
                    "speaker": normalize_speaker(segment.speaker),
                    "start_time": round(float(segment.start), 2),
                    "words": " ".join(tokens),
                }
            )
        rows.extend(session_rows)
        metadata.append(
            {
                "session_id": session_id,
                "segments": len(session_rows),
                "elapsed_sec": round(result.elapsed_sec, 3),
                "generated_tokens": result.generated_tokens,
                "generation_budget": args.max_new_tokens,
                "hit_token_limit": hit_token_limit,
                "retried_with_silence_padding": retried_with_silence_padding,
                "raw_transcript": result.text,
            }
        )

        args.output.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[{index}/{total_sessions}] {session_id}: "
            f"{len(session_rows)} segments, {result.elapsed_sec:.1f}s",
            flush=True,
        )

    print(
        f"Wrote {len(rows)} rows from {total_sessions} sessions to {args.output} "
        f"in {time.time() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
