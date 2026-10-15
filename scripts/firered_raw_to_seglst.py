import argparse
import json
import re
from pathlib import Path


CHINESE = re.compile(r"[\u4e00-\u9fff]")
ASCII_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


def normalize_tokens(text):
    text = (text or "").lower()
    tokens = []
    index = 0
    while index < len(text):
        if CHINESE.match(text[index]):
            tokens.append(text[index])
            index += 1
            continue
        match = ASCII_WORD.match(text, index)
        if match:
            tokens.append(match.group(0))
            index = match.end()
            continue
        index += 1
    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Convert FireRedASR2 raw inference JSON to text-only SegLST."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8-sig"))
    rows = []
    for item in raw:
        tokens = normalize_tokens(item.get("text", ""))
        if not tokens:
            continue
        duration = max(0.01, float(item.get("duration", 0.0)))
        rows.append(
            {
                "session_id": str(item["session_id"]),
                "speaker": "spk1",
                "start_time": 0.0,
                "end_time": round(duration, 2),
                "words": " ".join(tokens),
            }
        )

    rows.sort(key=lambda row: row["session_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} sessions to {args.output}")


if __name__ == "__main__":
    main()
