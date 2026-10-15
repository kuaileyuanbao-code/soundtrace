import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from relabel_seglst_with_rttm import best_speaker, overlap, parse_rttm_dir


def maximize_assignment(score_matrix):
    row_count = len(score_matrix)
    column_count = max((len(row) for row in score_matrix), default=0)
    size = max(row_count, column_count)
    padded = [row + [0.0] * (size - len(row)) for row in score_matrix]
    padded.extend([[0.0] * size for _ in range(size - row_count)])

    states = {0: (0.0, [])}
    for row_index in range(size):
        next_states = {}
        for used_mask, (score, assignment) in states.items():
            for column_index in range(size):
                bit = 1 << column_index
                if used_mask & bit:
                    continue
                candidate = (score + padded[row_index][column_index], assignment + [column_index])
                new_mask = used_mask | bit
                current = next_states.get(new_mask)
                if current is None or candidate[0] > current[0]:
                    next_states[new_mask] = candidate
        states = next_states
    return states[(1 << size) - 1][1][:row_count]


def build_mapping(rows, intervals, one_to_one, fallback):
    source_rows = defaultdict(list)
    for row in rows:
        source_rows[str(row["speaker"])].append(row)

    target_speakers = sorted({speaker for _, _, speaker in intervals})
    source_speakers = sorted(source_rows)
    scores = []
    for source_speaker in source_speakers:
        speaker_scores = []
        for target_speaker in target_speakers:
            target_intervals = [
                (start, end)
                for start, end, speaker in intervals
                if speaker == target_speaker
            ]
            total = 0.0
            for row in source_rows[source_speaker]:
                row_start = float(row["start_time"])
                row_end = float(row["end_time"])
                total += sum(
                    overlap(row_start, row_end, start, end)
                    for start, end in target_intervals
                )
            speaker_scores.append(total)
        scores.append(speaker_scores)

    mapping = {}
    if one_to_one:
        assignment = maximize_assignment(scores)
        for row_index, column_index in enumerate(assignment):
            if column_index < len(target_speakers) and scores[row_index][column_index] > 0:
                mapping[source_speakers[row_index]] = target_speakers[column_index]
    if fallback == "none":
        return mapping
    for source_speaker, speaker_scores in zip(source_speakers, scores):
        if source_speaker in mapping or not target_speakers:
            continue
        if speaker_scores and max(speaker_scores) > 0:
            best_index = max(range(len(speaker_scores)), key=speaker_scores.__getitem__)
            mapping[source_speaker] = target_speakers[best_index]
            continue

        if fallback == "overlap":
            continue

        source_midpoints = [
            (float(row["start_time"]) + float(row["end_time"])) / 2
            for row in source_rows[source_speaker]
        ]
        nearest = min(
            intervals,
            key=lambda interval: min(
                min(abs(midpoint - interval[0]), abs(midpoint - interval[1]))
                for midpoint in source_midpoints
            ),
            default=None,
        )
        if nearest is not None:
            mapping[source_speaker] = nearest[2]
    return mapping


def mapping_purity(rows, intervals, mapping):
    matched_duration = Counter()
    total_duration = Counter()
    for row in rows:
        source_speaker = str(row["speaker"])
        mapped_speaker = mapping.get(source_speaker)
        if mapped_speaker is None:
            continue
        start = float(row["start_time"])
        end = float(row["end_time"])
        duration = max(0.01, end - start)
        row_speaker = best_speaker(start, end, intervals, source_speaker)
        total_duration[source_speaker] += duration
        if row_speaker == mapped_speaker:
            matched_duration[source_speaker] += duration
    return {
        speaker: matched_duration[speaker] / duration
        for speaker, duration in total_duration.items()
        if duration > 0
    }


def source_group(speaker):
    if "_" in speaker:
        return speaker.split("_", 1)[0]
    return speaker


def main():
    parser = argparse.ArgumentParser(
        description="Map chunk-local ASR speakers to global RTTM speakers."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--rttm-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--one-to-one", action="store_true")
    parser.add_argument(
        "--fallback",
        choices=("nearest", "overlap", "none"),
        default="nearest",
        help=(
            "Policy for local speakers left unmatched by the one-to-one assignment. "
            "'overlap' allows many-to-one positive-overlap mapping, 'nearest' also "
            "maps zero-overlap speakers by nearest activity, and 'none' preserves "
            "their original local labels."
        ),
    )
    parser.add_argument(
        "--row-fallback-threshold",
        type=float,
        default=None,
        help=(
            "Use independent row-level RTTM assignment when duration-weighted "
            "support for a local-to-global mapping falls below this threshold."
        ),
    )
    args = parser.parse_args()
    if args.row_fallback_threshold is not None and not (
        0.0 <= args.row_fallback_threshold <= 1.0
    ):
        parser.error("--row-fallback-threshold must be between 0 and 1")

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    rttm = parse_rttm_dir(args.rttm_dir)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["session_id"]), source_group(str(row["speaker"])))].append(row)

    mappings = {}
    purities = {}
    for (session_id, group), group_rows in grouped.items():
        intervals = rttm.get(session_id, [])
        mapping = build_mapping(
            group_rows,
            intervals,
            args.one_to_one,
            args.fallback,
        )
        mappings[(session_id, group)] = mapping
        purities[(session_id, group)] = mapping_purity(
            group_rows, intervals, mapping
        )

    output = []
    mapped = 0
    row_fallback_rows = 0
    for row in rows:
        item = dict(row)
        key = (str(row["session_id"]), source_group(str(row["speaker"])))
        source_speaker = str(row["speaker"])
        speaker = mappings[key].get(source_speaker)
        purity = purities[key].get(source_speaker, 1.0)
        if (
            args.row_fallback_threshold is not None
            and purity < args.row_fallback_threshold
        ):
            speaker = best_speaker(
                float(row["start_time"]),
                float(row["end_time"]),
                rttm.get(str(row["session_id"]), []),
                source_speaker,
            )
            row_fallback_rows += 1
        if speaker is not None:
            item["speaker"] = speaker
            mapped += 1
        output.append(item)

    output.sort(
        key=lambda row: (
            str(row["session_id"]),
            float(row["start_time"]),
            float(row["end_time"]),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {len(output)} rows to {args.output}; "
        f"mapped {mapped} rows in {len(mappings)} source groups; "
        f"row fallback used for {row_fallback_rows} rows"
    )


if __name__ == "__main__":
    main()
