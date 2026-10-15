import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_rttm_dir(rttm_dir):
    by_session = defaultdict(list)
    speaker_mappings = defaultdict(dict)
    for path in sorted(Path(rttm_dir).glob("*.rttm")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8 or parts[0] != "SPEAKER":
                    continue
                session_id = parts[1]
                start = float(parts[3])
                dur = float(parts[4])
                raw_speaker = parts[7]
                if raw_speaker.isdigit():
                    speaker = f"spk{int(raw_speaker)}"
                else:
                    mapping = speaker_mappings[session_id]
                    if raw_speaker not in mapping:
                        mapping[raw_speaker] = f"spk{len(mapping) + 1}"
                    speaker = mapping[raw_speaker]
                by_session[session_id].append((start, start + dur, speaker))
    for intervals in by_session.values():
        intervals.sort()
    return by_session


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def best_speaker(start, end, intervals, default):
    best = (0.0, default)
    midpoint = (start + end) / 2
    for i_start, i_end, speaker in intervals:
        ov = overlap(start, end, i_start, i_end)
        if ov > best[0]:
            best = (ov, speaker)
    if best[0] > 0:
        return best[1]
    for i_start, i_end, speaker in intervals:
        if i_start <= midpoint <= i_end:
            return speaker
    nearest = min(
        intervals,
        key=lambda item: min(abs(midpoint - item[0]), abs(midpoint - item[1])),
        default=None,
    )
    return nearest[2] if nearest else default


def split_tokens(tokens, overlaps):
    total = sum(item[0] for item in overlaps)
    if total <= 0 or len(overlaps) <= 1:
        return [tokens]

    chunks = []
    prev = 0
    running = 0.0
    for idx, (duration, _, _, _) in enumerate(overlaps):
        running += duration
        if idx == len(overlaps) - 1:
            cut = len(tokens)
        else:
            cut = round(running / total * len(tokens))
            cut = max(prev + 1, min(cut, len(tokens) - (len(overlaps) - idx - 1)))
        chunks.append(tokens[prev:cut])
        prev = cut
    return chunks


def relabel_row(row, intervals, split_on_rttm):
    start = float(row["start_time"])
    end = float(row["end_time"])
    if not intervals:
        return [row]

    if not split_on_rttm:
        item = dict(row)
        item["speaker"] = best_speaker(start, end, intervals, row["speaker"])
        return [item]

    tokens = row["words"].split()
    overlaps = []
    for i_start, i_end, speaker in intervals:
        ov = overlap(start, end, i_start, i_end)
        if ov > 0:
            overlaps.append((ov, max(start, i_start), min(end, i_end), speaker))
    if not overlaps:
        item = dict(row)
        item["speaker"] = best_speaker(start, end, intervals, row["speaker"])
        return [item]

    overlaps.sort(key=lambda x: (x[1], x[2]))
    token_chunks = split_tokens(tokens, overlaps)
    out = []
    for (_, seg_start, seg_end, speaker), chunk in zip(overlaps, token_chunks):
        if not chunk:
            continue
        out_start = round(seg_start, 2)
        out_end = round(seg_end, 2)
        if out_end <= out_start:
            out_end = round(out_start + 0.01, 2)
        item = dict(row)
        item["speaker"] = speaker
        item["start_time"] = out_start
        item["end_time"] = out_end
        item["words"] = " ".join(chunk)
        out.append(item)
    return out or [row]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--rttm-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-on-rttm", action="store_true")
    parser.add_argument(
        "--sessions",
        type=Path,
        default=None,
        help="Optional text file; only these sessions are relabeled.",
    )
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    rttm = parse_rttm_dir(args.rttm_dir)
    selected = None
    if args.sessions:
        selected = {
            line.strip()
            for line in args.sessions.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        }
    out = []
    changed_sessions = set()
    for row in rows:
        if selected is not None and row["session_id"] not in selected:
            out.append(row)
            continue
        intervals = rttm.get(row["session_id"], [])
        if intervals:
            changed_sessions.add(row["session_id"])
        out.extend(relabel_row(row, intervals, args.split_on_rttm))

    out.sort(key=lambda x: (x["session_id"], float(x["start_time"]), float(x["end_time"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} rows, relabeled {len(changed_sessions)} sessions to {args.output}")


if __name__ == "__main__":
    main()
