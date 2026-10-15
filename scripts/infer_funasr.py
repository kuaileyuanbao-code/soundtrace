import argparse
import json
import re
import sys
from pathlib import Path

from ftfy import fix_text
from funasr import AutoModel


TAG_RE = re.compile(r"<\|[^|]+?\|>")
CN_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def repair_text(text):
    text = text or ""
    try:
        text = text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        text = fix_text(text)
    text = TAG_RE.sub("", text)
    return text.strip()


def normalize_tokens(text):
    text = repair_text(text).lower()
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


def normalize_words(text):
    return " ".join(normalize_tokens(text))


def session_id_from_path(path):
    return Path(path).stem


def make_records(session_id, item, split_max_ms=0, split_gap_ms=0):
    start = item.get("start", 0)
    end = item.get("end", start)
    tokens = normalize_tokens(item.get("sentence", ""))
    if not tokens:
        return []
    spk = item.get("spk", 0)
    speaker = f"spk{int(spk) + 1}" if isinstance(spk, int) else str(spk)
    timestamps = item.get("timestamp") or []

    chunks = []
    if (split_max_ms or split_gap_ms) and len(timestamps) == len(tokens):
        chunk_start = 0
        for idx, timestamp in enumerate(timestamps):
            if idx == chunk_start:
                continue
            chunk_begin = timestamps[chunk_start][0]
            chunk_end = timestamp[1]
            prev_end = timestamps[idx - 1][1]
            gap = timestamp[0] - prev_end
            should_split = False
            if split_gap_ms and gap >= split_gap_ms:
                should_split = True
            if split_max_ms and chunk_end - chunk_begin > split_max_ms:
                should_split = True
            if should_split:
                chunks.append((chunk_start, idx))
                chunk_start = idx
        chunks.append((chunk_start, len(tokens)))
    else:
        chunks = [(0, len(tokens))]

    records = []
    for begin, finish in chunks:
        if (split_max_ms or split_gap_ms) and len(timestamps) == len(tokens):
            record_start = timestamps[begin][0]
            record_end = timestamps[finish - 1][1]
        else:
            record_start = start
            record_end = end
        records.append(
            {
                "session_id": session_id,
                "speaker": speaker,
                "start_time": round(float(record_start) / 1000.0, 2),
                "end_time": round(float(record_end) / 1000.0, 2),
                "words": " ".join(tokens[begin:finish]),
            }
        )
    return records


def infer_file(model, wav_path, split_max_ms=0, split_gap_ms=0):
    result = model.generate(
        input=str(wav_path),
        batch_size_s=60,
        merge_vad=False,
    )
    session_id = session_id_from_path(wav_path)
    records = []
    for item in result[0].get("sentence_info") or []:
        records.extend(
            make_records(
                session_id,
                item,
                split_max_ms=split_max_ms,
                split_gap_ms=split_gap_ms,
            )
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-segment-ms", type=int, default=12000)
    parser.add_argument("--model", default="iic/SenseVoiceSmall")
    parser.add_argument("--split-max-ms", type=int, default=0)
    parser.add_argument("--split-gap-ms", type=int, default=0)
    args = parser.parse_args()

    wavs = sorted(args.wav_dir.glob("*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]
    if not wavs:
        raise SystemExit(f"No wav files found in {args.wav_dir}")

    model = AutoModel(
        model=args.model,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": args.max_segment_ms},
        spk_model="cam++",
        device=args.device,
        disable_update=True,
    )

    all_records = []
    for idx, wav_path in enumerate(wavs, 1):
        print(f"[{idx}/{len(wavs)}] {wav_path.name}", flush=True)
        try:
            all_records.extend(
                infer_file(
                    model,
                    wav_path,
                    split_max_ms=args.split_max_ms,
                    split_gap_ms=args.split_gap_ms,
                )
            )
        except Exception as exc:
            print(f"ERROR {wav_path}: {exc}", file=sys.stderr, flush=True)
            all_records.append(
                {
                    "session_id": session_id_from_path(wav_path),
                    "speaker": "spk1",
                    "start_time": 0.0,
                    "end_time": 0.01,
                    "words": "嗯",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(all_records)} records to {args.output}")


if __name__ == "__main__":
    main()
