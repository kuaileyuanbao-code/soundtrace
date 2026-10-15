import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore original session times and remove overlap from chunk-level SegLST."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--chunk-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--speaker-mode",
        choices=["unique-per-chunk", "preserve"],
        default="unique-per-chunk",
    )
    return parser.parse_args()


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def ownership_windows(manifest):
    grouped = defaultdict(list)
    for row in manifest:
        grouped[str(row["session_id"])].append(row)

    result = {}
    for session_id, chunks in grouped.items():
        chunks.sort(key=lambda row: float(row["start_time"]))
        for index, chunk in enumerate(chunks):
            start = float(chunk["start_time"])
            end = float(chunk["end_time"])
            left = start
            right = end
            if index > 0:
                previous_end = float(chunks[index - 1]["end_time"])
                left = (start + previous_end) / 2.0
            if index + 1 < len(chunks):
                next_start = float(chunks[index + 1]["start_time"])
                right = (end + next_start) / 2.0
            result[str(chunk["chunk_id"])] = {
                "chunk_index": int(chunk["chunk_index"]),
                "session_id": session_id,
                "start_time": start,
                "end_time": end,
                "ownership_start": left,
                "ownership_end": right,
            }
    return result


def main():
    args = parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    manifest = json.loads(args.chunk_manifest.read_text(encoding="utf-8"))
    windows = ownership_windows(manifest)

    output = []
    dropped = 0
    for row in rows:
        chunk_id = str(row["session_id"])
        if chunk_id not in windows:
            raise KeyError(f"Chunk missing from manifest: {chunk_id}")
        window = windows[chunk_id]
        chunk_start = window["start_time"]
        start = chunk_start + float(row["start_time"])
        end = chunk_start + float(row["end_time"])
        midpoint = (start + end) / 2.0
        is_last_edge = abs(midpoint - window["ownership_end"]) < 1e-9
        if midpoint < window["ownership_start"] or (
            midpoint > window["ownership_end"] or is_last_edge
        ):
            dropped += 1
            continue

        item = dict(row)
        item["session_id"] = window["session_id"]
        item["start_time"] = round(max(start, window["ownership_start"]), 3)
        item["end_time"] = round(min(end, window["ownership_end"]), 3)
        if item["end_time"] <= item["start_time"]:
            item["end_time"] = round(item["start_time"] + 0.01, 3)
        if args.speaker_mode == "unique-per-chunk":
            item["speaker"] = f"chunk{window['chunk_index']:04d}_{row['speaker']}"
        output.append(item)

    output.sort(
        key=lambda row: (
            str(row["session_id"]),
            float(row["start_time"]),
            float(row["end_time"]),
            str(row["speaker"]),
        )
    )
    atomic_write_json(args.output, output)
    print(f"Wrote {len(output)} rows to {args.output}; dropped {dropped} overlap rows")


if __name__ == "__main__":
    main()
