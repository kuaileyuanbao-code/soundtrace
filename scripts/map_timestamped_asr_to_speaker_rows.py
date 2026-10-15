import argparse
import json
from collections import defaultdict
from pathlib import Path

from firered_raw_to_seglst import normalize_tokens
from replace_text_by_token_alignment import align, flatten, load_rows


def load_timestamped_tokens(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    sessions = {}
    for item in raw:
        tokens = []
        for timestamp in item.get("timestamps", []):
            normalized = normalize_tokens(str(timestamp.get("token", "")))
            for token in normalized:
                tokens.append(
                    {
                        "token": token,
                        "start_time": float(timestamp["start_time"]),
                        "end_time": float(timestamp["end_time"]),
                    }
                )
        sessions[str(item["session_id"])] = tokens
    return sessions


def interval_distance(midpoint, row):
    start = float(row["start_time"])
    end = float(row["end_time"])
    if start <= midpoint <= end:
        return 0.0
    return min(abs(midpoint - start), abs(midpoint - end))


def interval_overlap(token, row):
    return max(
        0.0,
        min(float(token["end_time"]), float(row["end_time"]))
        - max(float(token["start_time"]), float(row["start_time"])),
    )


def timestamp_row(token, rows, preferred=None):
    midpoint = (float(token["start_time"]) + float(token["end_time"])) / 2
    overlaps = [interval_overlap(token, row) for row in rows]
    best_overlap = max(overlaps, default=0.0)
    candidates = [
        index for index, overlap in enumerate(overlaps)
        if overlap == best_overlap and overlap > 0
    ]
    if preferred in candidates:
        return preferred
    if not candidates:
        distances = [interval_distance(midpoint, row) for row in rows]
        best_distance = min(distances)
        candidates = [
            index for index, distance in enumerate(distances)
            if distance == best_distance
        ]
        if preferred in candidates:
            return preferred
    return min(
        candidates,
        key=lambda index: (
            interval_distance(midpoint, rows[index]),
            abs(
                midpoint
                - (
                    float(rows[index]["start_time"])
                    + float(rows[index]["end_time"])
                )
                / 2
            ),
            float(rows[index]["end_time"]) - float(rows[index]["start_time"]),
            index,
        ),
    )


def choose_assignments(rows, tokens, mode, slack):
    source = flatten(rows)
    candidate = [(token["token"], 0) for token in tokens]
    pairs, _, _, _ = align(source, candidate)
    assignments = []
    for candidate_index, token in enumerate(tokens):
        aligned_row = None
        if candidate_index in pairs:
            aligned_row = source[pairs[candidate_index]][1]
        timed_row = timestamp_row(token, rows, preferred=aligned_row)
        if mode == "timestamp" or aligned_row is None:
            assignments.append(timed_row)
            continue

        midpoint = (float(token["start_time"]) + float(token["end_time"])) / 2
        aligned = rows[aligned_row]
        inside = (
            float(aligned["start_time"]) - slack
            <= midpoint
            <= float(aligned["end_time"]) + slack
        )
        assignments.append(aligned_row if inside else timed_row)
    return assignments


def emit_source_rows(session_id, rows, tokens, assignments):
    grouped = defaultdict(list)
    for token, row_index in zip(tokens, assignments):
        grouped[row_index].append(token["token"])
    output = []
    for row_index, row in enumerate(rows):
        words = grouped.get(row_index)
        if not words:
            continue
        output.append({**row, "words": " ".join(words)})
    return output


def emit_token_times(session_id, rows, tokens, assignments, max_gap):
    output = []
    current = None
    for token, row_index in zip(tokens, assignments):
        speaker = rows[row_index]["speaker"]
        start = float(token["start_time"])
        end = float(token["end_time"])
        if (
            current is None
            or current["speaker"] != speaker
            or start - current["end_time"] > max_gap
        ):
            if current is not None:
                current["words"] = " ".join(current.pop("tokens"))
                current["start_time"] = round(current["start_time"], 3)
                current["end_time"] = round(current["end_time"], 3)
                output.append(current)
            current = {
                "session_id": session_id,
                "speaker": speaker,
                "start_time": start,
                "end_time": end,
                "tokens": [token["token"]],
            }
        else:
            current["end_time"] = max(current["end_time"], end)
            current["tokens"].append(token["token"])
    if current is not None:
        current["words"] = " ".join(current.pop("tokens"))
        current["start_time"] = round(current["start_time"], 3)
        current["end_time"] = round(current["end_time"], 3)
        output.append(current)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Map timestamped ASR tokens onto an existing speaker skeleton."
    )
    parser.add_argument("--speaker-time-source", required=True, type=Path)
    parser.add_argument("--timestamped-asr", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("timestamp", "hybrid"), default="hybrid")
    parser.add_argument("--slack", type=float, default=0.5)
    parser.add_argument("--timing", choices=("source", "token"), default="token")
    parser.add_argument("--max-gap", type=float, default=0.5)
    args = parser.parse_args()

    speaker_rows = load_rows(args.speaker_time_source)
    timestamped = load_timestamped_tokens(args.timestamped_asr)
    missing = set(speaker_rows) - set(timestamped)
    if missing:
        raise ValueError(f"Timestamped ASR is missing sessions: {sorted(missing)}")

    output = []
    diagnostics = {}
    for session_id in sorted(speaker_rows):
        rows = speaker_rows[session_id]
        tokens = timestamped[session_id]
        assignments = choose_assignments(rows, tokens, args.mode, args.slack)
        if args.timing == "source":
            emitted = emit_source_rows(session_id, rows, tokens, assignments)
        else:
            emitted = emit_token_times(
                session_id, rows, tokens, assignments, args.max_gap
            )
        output.extend(emitted)
        diagnostics[session_id] = {
            "tokens": len(tokens),
            "source_rows": len(rows),
            "emitted_rows": len(emitted),
            "speakers": len({row["speaker"] for row in emitted}),
        }

    output.sort(
        key=lambda row: (
            str(row["session_id"]),
            float(row["start_time"]),
            float(row["end_time"]),
            str(row["speaker"]),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    diagnostics_path = args.output.with_suffix(".diagnostics.json")
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "sessions": len(speaker_rows),
                "rows": len(output),
                "tokens": sum(len(row["words"].split()) for row in output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
