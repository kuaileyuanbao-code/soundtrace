"""Text-only window stitching. Window bounds are audio extents, not speech times."""
import math
import re


TOKEN = re.compile(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", re.UNICODE)
PUNCTUATION = ",，。.!！?？;；:：、"
# Validated on the 17.152 s failure, real short/repeated speech, and dense 20 s.
# Longer single inputs worked in experiments but increased latency; keep the
# product change bounded and preserve the existing planner above this ceiling.
TEXT_SINGLE_WINDOW_SECONDS = 20.0


def text_window_plan(duration):
    if duration <= TEXT_SINGLE_WINDOW_SECONDS:
        return [(0.0, duration, 0.0, duration)]
    from unified_engine import window_plan
    return window_plan(duration, 14.0, 2.0)


def join_text(left, right):
    left, right = left.rstrip(), right.lstrip()
    if not left or not right:
        return left or right
    # Prefer the continuing window's punctuation at a matched boundary.
    if left[-1] in PUNCTUATION and right[0] in PUNCTUATION:
        left = left.rstrip(PUNCTUATION)
    space = " " if (left and left[-1].isascii() and (left[-1].isalnum() or left[-1] in ".,!?;:)")
                         and right[0].isascii() and right[0].isalnum()) else ""
    return left + space + right


def merge_windows(chunks):
    """Conservative exact suffix/prefix matching, bounded by audio overlap.

    Ignore punctuation/case only for matching; preserve original text in output.
    Never guess word timestamps, interpolate text ownership, or globally remove
    repetitions. Ambiguous/mismatched boundaries retain text and emit a warning.
    """
    merged = ""
    boundaries = []
    previous = None
    for chunk in chunks:
        text = chunk["text"].strip()
        matched = 0
        overlap = max(0, previous["end_time"] - chunk["start_time"]) if previous else 0
        if text and merged and overlap > 0 and previous["text"].strip():
            left = list(TOKEN.finditer(previous["text"]))
            right = list(TOKEN.finditer(text))
            left_duration = previous["end_time"] - previous["start_time"]
            right_duration = chunk["end_time"] - chunk["start_time"]
            bound = math.ceil(max(len(left) * overlap / left_duration,
                                  len(right) * overlap / right_duration)) + 4
            for size in range(min(bound, len(left), len(right)), 0, -1):
                a = [m.group().casefold() for m in left[-size:]]
                b = [m.group().casefold() for m in right[:size]]
                # A single short token/character is insufficient evidence.
                if a == b and size >= 2 and sum(map(len, a)) >= 4:
                    matched = size
                    text = text[right[size - 1].end():]
                    break
            boundaries.append({"window_index": len(boundaries) + 1, "audio_overlap_seconds": overlap,
                               "matched_tokens": matched, "status": "matched" if matched else "unresolved"})
        merged = join_text(merged, text)
        previous = chunk
    warnings = []
    if any(b["status"] == "unresolved" for b in boundaries):
        warnings.append("部分重叠窗口未找到可靠的相同文字边界，已保留原文，边界可能存在重复；未执行时间对齐。")
    return merged, boundaries, warnings


def text_row(session_id, text, duration):
    return {"session_id": session_id, "speaker": None, "start_time": 0.0,
            "end_time": duration, "words": text,
            "timestamp_kind": "audio_extent", "speaker_analyzed": False}
