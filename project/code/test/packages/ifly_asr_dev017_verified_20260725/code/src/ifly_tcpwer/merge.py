"""Assign aligned ASR tokens to diarization tracks and create SegLST records."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Iterable

from .text import normalize_token


@dataclass
class Token:
    text: str
    start: float
    end: float
    speaker: str = ""


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    tokens: list[str] = field(default_factory=list)


def _distance_to_interval(time: float, start: float, end: float) -> float:
    if start <= time <= end:
        return 0.0
    return min(abs(time - start), abs(time - end))


def assign_speakers(tokens: list[Token], diarization: list[list[float]]) -> list[Token]:
    """Assign each token once, preferring active then nearest speaker tracks."""
    if not diarization:
        for token in tokens:
            token.speaker = "spk1"
        return tokens

    turns = [(float(start), float(end), f"spk{int(speaker) + 1}") for start, end, speaker in diarization]
    previous = turns[0][2]
    for token in tokens:
        midpoint = (token.start + token.end) / 2
        candidates = [turn for turn in turns if turn[0] <= midpoint <= turn[1]]
        if candidates:
            continuing = [turn for turn in candidates if turn[2] == previous]
            chosen = continuing[0] if continuing else max(candidates, key=lambda turn: turn[1] - turn[0])
        else:
            chosen = min(turns, key=lambda turn: _distance_to_interval(midpoint, turn[0], turn[1]))
        token.speaker = chosen[2]
        previous = token.speaker
    return tokens


def tokens_to_segments(
    tokens: Iterable[Token], *, max_duration: float = 4.5, max_gap: float = 1.0
) -> list[Segment]:
    segments: list[Segment] = []
    for token in tokens:
        normalized = normalize_token(token.text)
        if not normalized:
            continue
        can_append = (
            segments
            and segments[-1].speaker == token.speaker
            and token.start - segments[-1].end <= max_gap
            and token.end - segments[-1].start <= max_duration
        )
        if not can_append:
            segments.append(Segment(token.speaker, token.start, token.end, normalized))
        else:
            segments[-1].end = max(segments[-1].end, token.end)
            segments[-1].tokens.extend(normalized)
    return segments


def make_seglst(session_id: str, asr: dict, diarization: list[list[float]]) -> list[dict]:
    tokens = [
        Token(item["text"], float(item["start_time"]), float(item["end_time"]))
        for item in asr["timestamps"]
    ]
    segments = tokens_to_segments(assign_speakers(tokens, diarization))
    return [
        {
            "session_id": session_id,
            "speaker": segment.speaker,
            "start_time": round(max(0.0, segment.start), 2),
            "end_time": round(max(segment.start, segment.end), 2),
            "words": " ".join(segment.tokens),
        }
        for segment in segments
        if segment.tokens
    ]


def make_overlap_seglst(
    session_id: str,
    asr: dict,
    diarization: list[list[float]],
    min_turn_duration: float = 0.65,
    min_overlap_duration: float = 0.0,
) -> list[dict]:
    """Keep both active speakers for reliable long-turn overlaps."""
    turns = [
        (float(start), float(end), f"spk{int(speaker) + 1}")
        for start, end, speaker in diarization
        if float(end) > float(start)
    ]
    if not turns:
        return make_seglst(session_id, asr, diarization)

    per_speaker: dict[str, list[Token]] = defaultdict(list)
    previous = turns[0][2]
    for item in asr["timestamps"]:
        token = Token(item["text"], float(item["start_time"]), float(item["end_time"]))
        midpoint = (token.start + token.end) / 2
        active = [turn for turn in turns if turn[0] <= midpoint <= turn[1]]
        labels: list[str]
        distinct_active = sorted({turn[2] for turn in active})
        overlap_duration = (
            min(turn[1] for turn in active) - max(turn[0] for turn in active)
            if active
            else 0.0
        )
        if (
            len(distinct_active) > 1
            and min(turn[1] - turn[0] for turn in active) >= min_turn_duration
            and overlap_duration >= min_overlap_duration
        ):
            labels = distinct_active
        elif active:
            continuing = [turn for turn in active if turn[2] == previous]
            chosen = continuing[0] if continuing else max(active, key=lambda turn: turn[1] - turn[0])
            labels = [chosen[2]]
        else:
            chosen = min(turns, key=lambda turn: _distance_to_interval(midpoint, turn[0], turn[1]))
            labels = [chosen[2]]
        previous = labels[0]
        for label in labels:
            per_speaker[label].append(Token(token.text, token.start, token.end, label))

    output = []
    for tokens in per_speaker.values():
        for segment in tokens_to_segments(tokens):
            output.append(
                {
                    "session_id": session_id,
                    "speaker": segment.speaker,
                    "start_time": round(max(0.0, segment.start), 2),
                    "end_time": round(max(segment.start, segment.end), 2),
                    "words": " ".join(segment.tokens),
                }
            )
    output.sort(key=lambda item: (item["start_time"], item["end_time"], item["speaker"]))
    return [item for item in output if item["words"]]
