import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_STARTED = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from performance import StageTimings

from mover_runtime import configure_mover


def normalize_output(
    rows: list[dict], preserve_speaker_labels: bool = False
) -> list[dict]:
    speaker_maps: dict[str, dict[str, str]] = {}
    normalized = []
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["session_id"]),
            float(item["start_time"]),
            float(item["end_time"]),
            str(item["speaker"]),
        ),
    ):
        session_id = str(row["session_id"])
        speaker = str(row["speaker"])
        mapping = speaker_maps.setdefault(session_id, {})
        if speaker not in mapping:
            mapping[speaker] = (
                speaker if preserve_speaker_labels else f"spk{len(mapping) + 1}"
            )
        words = " ".join(str(row["words"]).lower().split())
        if not words:
            continue
        start_time = round(float(row["start_time"]), 2)
        end_time = round(float(row["end_time"]), 2)
        if end_time <= start_time:
            end_time = round(start_time + 0.01, 2)
        normalized.append(
            {
                "session_id": session_id,
                "speaker": mapping[speaker],
                "start_time": start_time,
                "end_time": end_time,
                "words": words,
            }
        )
    normalized.sort(
        key=lambda item: (
            str(item["session_id"]),
            float(item["start_time"]),
            float(item["end_time"]),
            str(item["speaker"]),
        )
    )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official NTT MOVER implementation on SegLST inputs."
    )
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--strategy", choices=["subset", "fullset", "persegment"], default="subset"
    )
    parser.add_argument("--tcdp-collar", type=float, default=5.0)
    parser.add_argument("--subset-grouping-threshold", type=float, default=0.0)
    parser.add_argument(
        "--pseudo-word-level-timing",
        choices=[
            "character_based",
            "character_based_points",
            "full_segment",
            "equidistant_intervals",
            "equidistant_points",
        ],
        default="character_based",
    )
    parser.add_argument(
        "--time-merge-type",
        choices=["center_points", "keep_segment"],
        default="center_points",
    )
    parser.add_argument("--speaker-voting", action="store_true")
    parser.add_argument("--disable-speaker-mapping", action="store_true")
    parser.add_argument("--disable-ocr", action="store_true")
    parser.add_argument("--preserve-speaker-labels", action="store_true")
    parser.add_argument("--timings-output", type=Path, default=None)
    args = parser.parse_args()
    profile = StageTimings()

    with profile.measure("mover_dependency_import_seconds"):
        configure_mover()
        import meeteval
        from mover.mover import mover

    hypotheses = []
    for path in args.input:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise TypeError(f"Expected a SegLST list: {path}")
        hypotheses.append(meeteval.io.SegLST(rows))

    with profile.measure("mover_fusion_seconds"):
        fused = mover(
            hypotheses,
            strategy=args.strategy,
            tcdp_collar=args.tcdp_collar,
            subset_grouping_threshold=args.subset_grouping_threshold,
            pseudo_word_level_timing=args.pseudo_word_level_timing,
            time_merge_type=args.time_merge_type,
            speaker_mapping=not args.disable_speaker_mapping,
            speaker_voting=args.speaker_voting,
            order_consistency_resolution=not args.disable_ocr,
        )
    rows = normalize_output(
        list(fused), preserve_speaker_labels=args.preserve_speaker_labels
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "inputs": [str(path) for path in args.input],
                "output": str(args.output),
                "rows": len(rows),
                "sessions": len({row["session_id"] for row in rows}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.timings_output is not None:
        total = time.perf_counter() - SCRIPT_STARTED
        args.timings_output.write_text(json.dumps({"timings": {**profile.seconds(),
            "mover_script_seconds": round(total, 6),
            "mover_other_seconds": round(max(0, total - sum(profile.values.values())), 6)},
            "profiling": profile.info()}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
