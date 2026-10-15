import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel


CN_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.I)


def normalize_tokens(text):
    text = (text or "").lower()
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if CN_RE.match(ch):
            tokens.append(ch)
            i += 1
            continue
        match = ASCII_WORD_RE.match(text, i)
        if match:
            tokens.append(match.group(0))
            i = match.end()
            continue
        i += 1
    return tokens


def make_windows(duration, window, overlap):
    if duration <= window:
        return [(0.0, duration)]
    stride = window - overlap
    starts = list(np.arange(0.0, max(duration - window, 0.0) + 1e-6, stride))
    last_start = max(0.0, duration - window)
    if not starts or last_start - starts[-1] > 0.05:
        starts.append(last_start)
    return [(float(start), min(duration, float(start) + window)) for start in starts]


def ownership_ranges(windows):
    ranges = []
    for index, (start, end) in enumerate(windows):
        keep_start = start
        keep_end = end
        if index:
            previous_end = windows[index - 1][1]
            keep_start = (previous_end + start) / 2.0
        if index + 1 < len(windows):
            next_start = windows[index + 1][0]
            keep_end = (end + next_start) / 2.0
        ranges.append((keep_start, keep_end))
    return ranges


def atomic_write_json(path, value):
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--window", type=float, default=14.0)
    parser.add_argument("--overlap", type=float, default=2.0)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--sessions", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.window <= 1.0:
        parser.error("--window must be greater than 1 second")
    if not 0.0 <= args.overlap < args.window:
        parser.error("--overlap must satisfy 0 <= overlap < window")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
        max_inference_batch_size=1,
        max_new_tokens=768,
        forced_aligner=args.aligner,
        forced_aligner_kwargs={"dtype": dtype, "device_map": args.device},
    )

    wav_paths = sorted(args.wav_dir.glob("*.wav"))
    if args.sessions:
        selected = {
            line.strip()
            for line in args.sessions.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        }
        wav_paths = [path for path in wav_paths if path.stem in selected]

    rows = []
    diagnostics = []
    if args.resume and args.output.exists():
        rows = json.loads(args.output.read_text(encoding="utf-8"))
    completed = {row["session_id"] for row in rows}

    for wav_index, wav_path in enumerate(wav_paths, 1):
        session_id = wav_path.stem
        if session_id in completed:
            print(f"[{wav_index}/{len(wav_paths)}] {session_id} skip", flush=True)
            continue
        audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1).astype(np.float32)
        duration = len(audio) / float(sample_rate)
        windows = make_windows(duration, args.window, args.overlap)
        ownership = ownership_ranges(windows)
        clips = [
            (
                audio[
                    int(round(start * sample_rate)) : int(round(end * sample_rate))
                ],
                sample_rate,
            )
            for start, end in windows
        ]
        print(
            f"[{wav_index}/{len(wav_paths)}] {session_id}: "
            f"{duration:.2f}s -> {len(windows)} windows",
            flush=True,
        )
        results = model.transcribe(
            audio=clips,
            language=args.language,
            return_time_stamps=True,
        )

        session_rows = []
        for window_index, (result, (offset, _), (keep_start, keep_end)) in enumerate(
            zip(results, windows, ownership)
        ):
            kept = 0
            for item in getattr(result, "time_stamps", None) or []:
                tokens = normalize_tokens(getattr(item, "text", ""))
                if not tokens:
                    continue
                start = offset + float(getattr(item, "start_time"))
                end = offset + float(getattr(item, "end_time"))
                if end <= start:
                    end = start + 0.01
                midpoint = (start + end) / 2.0
                is_last = window_index + 1 == len(windows)
                if midpoint < keep_start or (midpoint >= keep_end and not is_last):
                    continue
                step = (end - start) / len(tokens)
                for token_index, token in enumerate(tokens):
                    token_start = start + token_index * step
                    token_end = start + (token_index + 1) * step
                    session_rows.append(
                        {
                            "session_id": session_id,
                            "speaker": "spk1",
                            "start_time": round(token_start, 2),
                            "end_time": round(max(token_end, token_start + 0.01), 2),
                            "words": token,
                        }
                    )
                    kept += 1
            diagnostics.append(
                {
                    "session_id": session_id,
                    "window_index": window_index,
                    "window_start": round(offset, 3),
                    "window_end": round(windows[window_index][1], 3),
                    "keep_start": round(keep_start, 3),
                    "keep_end": round(keep_end, 3),
                    "text": getattr(result, "text", ""),
                    "kept_tokens": kept,
                }
            )
        rows.extend(session_rows)
        rows.sort(
            key=lambda row: (
                row["session_id"],
                float(row["start_time"]),
                float(row["end_time"]),
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output, rows)
        atomic_write_json(args.output.with_suffix(".windows.json"), diagnostics)

    print(f"Wrote {len(rows)} token rows to {args.output}")


if __name__ == "__main__":
    main()
