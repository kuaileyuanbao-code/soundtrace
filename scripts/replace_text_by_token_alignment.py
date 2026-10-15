import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_rows(path):
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["session_id"])].append(dict(row))
    for session_rows in grouped.values():
        session_rows.sort(
            key=lambda row: (
                float(row["start_time"]),
                float(row["end_time"]),
                str(row["speaker"]),
            )
        )
    return dict(grouped)


def flatten(rows):
    tokens = []
    for row_index, row in enumerate(rows):
        for token in str(row.get("words", "")).split():
            tokens.append((token, row_index))
    return tokens


def align(source, candidate):
    source_words = [item[0] for item in source]
    candidate_words = [item[0] for item in candidate]
    n = len(source_words)
    m = len(candidate_words)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0] = i
        back[i][0] = "delete"
    for j in range(1, m + 1):
        costs[0][j] = j
        back[0][j] = "insert"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = costs[i - 1][j - 1] + (
                source_words[i - 1] != candidate_words[j - 1]
            )
            deletion = costs[i - 1][j] + 1
            insertion = costs[i][j - 1] + 1
            costs[i][j], back[i][j] = min(
                (
                    (substitution, "pair"),
                    (deletion, "delete"),
                    (insertion, "insert"),
                ),
                key=lambda item: item[0],
            )

    pairs = {}
    exact = 0
    substitutions = 0
    i, j = n, m
    while i or j:
        action = back[i][j]
        if action == "pair":
            pairs[j - 1] = i - 1
            if source_words[i - 1] == candidate_words[j - 1]:
                exact += 1
            else:
                substitutions += 1
            i -= 1
            j -= 1
        elif action == "delete":
            i -= 1
        else:
            j -= 1
    return pairs, exact, substitutions, costs[n][m]


def fill_unpaired_rows(candidate, source, pairs):
    assigned = {
        candidate_index: source[source_index][1]
        for candidate_index, source_index in pairs.items()
    }
    mapped_indices = sorted(assigned)
    if not mapped_indices:
        max_source_row = max((row_index for _, row_index in source), default=0)
        return {
            index: min(candidate[index][1], max_source_row)
            for index in range(len(candidate))
        }

    for candidate_index in range(len(candidate)):
        if candidate_index in assigned:
            continue
        previous = next(
            (index for index in reversed(mapped_indices) if index < candidate_index),
            None,
        )
        following = next(
            (index for index in mapped_indices if index > candidate_index),
            None,
        )
        if previous is None:
            assigned[candidate_index] = assigned[following]
        elif following is None:
            assigned[candidate_index] = assigned[previous]
        elif candidate_index - previous <= following - candidate_index:
            assigned[candidate_index] = assigned[previous]
        else:
            assigned[candidate_index] = assigned[following]
    return assigned


def atomic_write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Keep one SegLST's speaker/time rows while replacing its text through "
            "global token alignment to another SegLST."
        )
    )
    parser.add_argument("--speaker-time-source", required=True, type=Path)
    parser.add_argument("--text-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--diagnostics-output", type=Path, default=None)
    args = parser.parse_args()

    speaker_time = load_rows(args.speaker_time_source)
    text = load_rows(args.text_source)
    missing = set(text) - set(speaker_time)
    if missing:
        raise ValueError(f"Speaker/time source is missing sessions: {sorted(missing)}")

    output = []
    diagnostics = {}
    for session_id in sorted(text):
        source_rows = speaker_time[session_id]
        source_tokens = flatten(source_rows)
        candidate_tokens = flatten(text[session_id])
        pairs, exact, substitutions, distance = align(
            source_tokens, candidate_tokens
        )
        assignments = fill_unpaired_rows(candidate_tokens, source_tokens, pairs)
        tokens_by_source_row = defaultdict(list)
        for candidate_index, (token, _) in enumerate(candidate_tokens):
            tokens_by_source_row[assignments[candidate_index]].append(token)

        emitted = 0
        for row_index, source_row in enumerate(source_rows):
            tokens = tokens_by_source_row.get(row_index)
            if not tokens:
                continue
            row = dict(source_row)
            row["words"] = " ".join(tokens)
            output.append(row)
            emitted += 1

        diagnostics[session_id] = {
            "source_tokens": len(source_tokens),
            "candidate_tokens": len(candidate_tokens),
            "exact_pairs": exact,
            "substitution_pairs": substitutions,
            "edit_distance": distance,
            "unaligned_candidate_tokens": len(candidate_tokens) - len(pairs),
            "source_rows": len(source_rows),
            "emitted_rows": emitted,
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
    atomic_write_json(args.output, output)
    diagnostics_path = (
        args.diagnostics_output
        or args.output.with_suffix(".alignment.json")
    )
    atomic_write_json(diagnostics_path, diagnostics)
    print(
        f"Wrote {len(output)} rows across {len(text)} sessions to {args.output}"
    )


if __name__ == "__main__":
    main()
