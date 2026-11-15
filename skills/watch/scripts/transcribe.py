#!/usr/bin/env python3
"""WebVTT parsing and shared transcript formatting."""
from __future__ import annotations

import html
import math
import re
import sys
from pathlib import Path

from runtime import configure_stdio

TIME = r'(?:[0-9]{2,}:)?[0-5][0-9]:[0-5][0-9][.,][0-9]{3}'
TIMING_RE = re.compile(rf'^\s*({TIME})\s+-->\s+({TIME})(?:\s+.*)?$')
TAG_RE = re.compile(r'<[^>]*>')
BLOCK_RE = re.compile(r'^(NOTE(?:\s|$)|STYLE$|REGION$)')


def normalize_segments(data: object) -> list[dict]:
    """Validate a {'segments': [{start, end, text}]} payload into chronological segments."""
    if not isinstance(data, dict) or not isinstance(data.get('segments'), list):
        raise ValueError('expected a JSON object with a segments array')
    out = []
    previous = -1.0
    for segment in data['segments']:
        if not isinstance(segment, dict):
            raise ValueError('invalid segment')
        start, end, text = segment.get('start'), segment.get('end'), segment.get('text')
        if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or isinstance(start, bool) or isinstance(end, bool)):
            raise ValueError('invalid segment timestamps')
        start, end = float(start), float(end)
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start
                or start < previous or not isinstance(text, str) or not text.strip()):
            raise ValueError('invalid segment timestamps or text')
        previous = start
        out.append({'start': round(start, 6), 'end': round(end, 6), 'text': text.strip()})
    return out


def _seconds(stamp: str) -> float:
    return sum(float(part) * 60 ** i for i, part in enumerate(reversed(stamp.replace(',', '.').split(':'))))


def parse_vtt(path: str) -> list[dict]:
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    segments = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if BLOCK_RE.match(line):
            i += 1
            while i < len(lines) and lines[i] != '':
                i += 1
            continue
        match = TIMING_RE.match(line)
        i += 1
        if not match:
            continue
        start, end = map(_seconds, match.groups())
        display = line[match.end(2):].strip()
        payload = []
        # A timing line also ends the cue, recovering a missing blank separator.
        while i < len(lines) and lines[i] != '' and not TIMING_RE.match(lines[i]):
            payload.append(lines[i])
            i += 1
        text = ' '.join(html.unescape(TAG_RE.sub('', part)).strip() for part in payload).strip()
        if text and end >= start:
            segments.append({'start': start, 'end': end, 'text': text, '_display': display})
    return [{k: s[k] for k in ('start', 'end', 'text')} for s in _dedupe(segments)]


def _dedupe(segments: list[dict]) -> list[dict]:
    """Collapse only overlapping same-display duplicates; preserve repeated speech."""
    out = []
    for seg in segments:
        if out and seg['start'] < out[-1]['end'] and seg['_display'] == out[-1]['_display']:
            previous = out[-1]
            if seg['text'] == previous['text'] or seg['text'].startswith(previous['text'] + ' '):
                previous['text'] = seg['text']
                previous['end'] = max(previous['end'], seg['end'])
                continue
        out.append(dict(seg))
    return out


def filter_range(segments: list[dict], start_seconds: float | None, end_seconds: float | None) -> list[dict]:
    if start_seconds is None and end_seconds is None:
        return segments
    lo = start_seconds if start_seconds is not None else float('-inf')
    hi = end_seconds if end_seconds is not None else float('inf')
    return [seg for seg in segments if seg['end'] >= lo and seg['start'] <= hi]


def format_transcript(segments: list[dict]) -> str:
    return '\n'.join(f"[{int(s['start']) // 60:02d}:{int(s['start']) % 60:02d}] {s['text']}" for s in segments)


if __name__ == '__main__':
    configure_stdio()
    if len(sys.argv) != 2:
        raise SystemExit('usage: transcribe.py <vtt-path>')
    print(format_transcript(parse_vtt(sys.argv[1])))
