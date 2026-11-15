#!/usr/bin/env python3
"""Probe video metadata and select frames: keyframes, scene changes, uniform samples, or cue points.

Auto-fps targets a frame budget, not a fixed rate: token cost scales with frame count,
so short videos stay dense and long ones capped. Focused ranges get a denser budget.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from runtime import diagnostic, run_text

MAX_FPS = 2.0
SCENE_THRESHOLD = 0.20
# Fewer detected shots than this means an effectively static clip (screen recording,
# talking head): sample uniformly instead. A floor, not the frame budget.
SCENE_MIN_FRAMES = 8
# Fewer keyframes than this is too sparse for coverage; fall back to uniform.
KEYFRAME_MIN = 4
MAX_READ_DIMENSION = 1998
# Near-duplicate test: mean per-channel difference (0-255) of 16x16 RGB thumbnails.
# Colour-aware, so flat frames (slides, fades) stay distinct; subtle text edits can be missed.
DEDUP_THUMB = 16
DEDUP_THRESHOLD = 2.0
SHOWINFO_TS_RE = re.compile(r"pts_time:([^\s]+)")


def parse_time(value: str | float | int | None) -> float | None:
    """Parse a finite non-negative SS, MM:SS, or HH:MM:SS time."""
    if value is None or str(value).strip() == "":
        return None
    try:
        parts = str(value).strip().split(":")
        if len(parts) > 3:
            raise ValueError()
        numbers = [float(part) for part in parts]
        if any(not math.isfinite(n) or n < 0 for n in numbers):
            raise ValueError()
        if len(parts) > 1 and (any(n >= 60 for n in numbers[1:]) or any(n != int(n) for n in numbers[:-1])):
            raise ValueError()
        return sum(n * 60 ** i for i, n in enumerate(reversed(numbers)))
    except ValueError:
        raise SystemExit("Invalid time; expected finite non-negative SS, MM:SS, or HH:MM:SS.") from None


def parse_timestamps(value: str | None) -> list[float]:
    """Comma-separated times into sorted, de-duplicated seconds; blank tokens are skipped."""
    tokens = [token.strip() for token in (value or "").split(",")]
    return sorted({float(parse_time(token)) for token in tokens if token})


def format_time(seconds: float) -> str:
    hours, rem = divmod(int(round(seconds)), 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}" if hours else f"{minutes:02d}:{sec:02d}"


def validate_controls(resolution, max_frames, start_seconds, end_seconds, fps=None):
    if not isinstance(resolution, int) or resolution < 2:
        raise SystemExit("--resolution must be an integer of at least 2 pixels")
    if max_frames is not None and (not isinstance(max_frames, int) or max_frames < 1):
        raise SystemExit("--max-frames must be a positive integer")
    start, end = parse_time(start_seconds) or 0.0, parse_time(end_seconds)
    if end is not None and end <= start:
        raise SystemExit("--end must be greater than --start")
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise SystemExit("--fps must be finite and greater than zero")


def get_metadata(video_path: str) -> dict:
    result = run_text([
        "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
        str(Path(video_path).resolve()),
    ], timeout=30)
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed: {diagnostic(result.stderr)}")
    try:
        data = json.loads(result.stdout)
        streams, fmt = data["streams"], data.get("format", {})
        if not isinstance(streams, list) or not streams:
            raise ValueError()
        video = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
        # format.duration can include a nonzero start PTS; stream durations are elapsed source time.
        elapsed = [float(s['duration']) for s in (video, audio) if s.get('duration') not in (None, 'N/A')]
        duration = max(elapsed) if elapsed else float(fmt.get('duration'))
        if any(not math.isfinite(d) or d <= 0 for d in elapsed) or not math.isfinite(duration) or duration <= 0:
            raise ValueError()
        return {
            "duration_seconds": duration, "width": video.get("width"), "height": video.get("height"),
            "codec": video.get("codec_name"), "size_bytes": int(fmt.get("size") or 0),
            "has_audio": bool(audio), "has_video": bool(video),
        }
    except (ValueError, TypeError, KeyError, AttributeError):
        raise SystemExit(f"ffprobe returned invalid metadata, streams, or duration: {diagnostic(result.stderr)}") from None


@lru_cache(maxsize=8)
def _sync_option(executable: str) -> tuple[str, str]:
    result = run_text([executable, "-hide_banner", "-h", "full"], timeout=30)
    if result.returncode == 0:
        help_text = result.stdout + result.stderr
        # FFmpeg 9 lists options with stream specifiers, e.g. "-fps_mode[:<stream_spec>]".
        if re.search(r"(?m)^\s*-fps_mode\b", help_text):
            return "-fps_mode", "vfr"
        if re.search(r"(?m)^\s*-vsync\b", help_text):
            return "-vsync", "vfr"
    raise SystemExit(f"{executable} did not advertise -fps_mode or -vsync; check the FFmpeg installation. {diagnostic(result.stderr)}")


def sync_args() -> tuple[str, str]:
    return _sync_option(shutil.which("ffmpeg") or "ffmpeg")


# ---- frame budgets ---------------------------------------------------------------------------

def _clamp_fps(fps: float, duration_seconds: float, max_frames: int) -> tuple[float, int]:
    fps = min(fps, MAX_FPS)
    return fps, min(max_frames, max(1, int(round(fps * duration_seconds))))


def auto_fps(duration_seconds: float, max_frames: int = 100) -> tuple[float, int]:
    """Pick fps that targets a sensible frame budget for full-video scans."""
    if duration_seconds <= 0:
        return 1.0, 1
    if duration_seconds <= 30:
        target = min(max_frames, max(12, int(round(duration_seconds))))
    elif duration_seconds <= 60:
        target = min(max_frames, 40)
    elif duration_seconds <= 180:
        target = min(max_frames, 60)
    elif duration_seconds <= 600:
        target = min(max_frames, 80)
    else:
        target = max_frames
    return _clamp_fps(target / duration_seconds, duration_seconds, max_frames)


def auto_fps_focus(duration_seconds: float, max_frames: int = 100) -> tuple[float, int]:
    """Denser budget for user-specified ranges: they are zooming in for detail."""
    if duration_seconds <= 0:
        return MAX_FPS, 2
    if duration_seconds <= 5:
        target = min(max_frames, max(10, int(round(duration_seconds * 6))))
    elif duration_seconds <= 15:
        target = min(max_frames, max(30, int(round(duration_seconds * 4))))
    elif duration_seconds <= 30:
        target = min(max_frames, 60)
    elif duration_seconds <= 60:
        target = min(max_frames, 80)
    else:
        target = max_frames
    return _clamp_fps(target / duration_seconds, duration_seconds, max_frames)


# ---- ffmpeg extraction -----------------------------------------------------------------------

def _scale_filter(resolution: int) -> str:
    return (
        f"scale=w='min({resolution},iw)':h='min({MAX_READ_DIMENSION},ih)':"
        "force_original_aspect_ratio=decrease:force_divisible_by=2"
    )


def _clear(out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob(f"{prefix}_*.jpg"):
        existing.unlink()


def _ffmpeg_frames(video_path: str, output: Path, vf: str, *, start: float | None = None,
                   end: float | None = None, max_frames: int | None = None,
                   input_args: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    """Run ffmpeg with showinfo logging so every written image has a source timestamp."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.6f}"]
    if end is not None:
        cmd += ["-to", f"{end:.6f}"]
    cmd += [*input_args, "-i", str(Path(video_path).resolve()), "-vf", f"{vf},showinfo", *sync_args()]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    return run_text([*cmd, "-q:v", "4", str(output)])


def _emitted_frames(files, stderr, offset, reason):
    try:
        stamps = [float(m.group(1)) for m in SHOWINFO_TS_RE.finditer(stderr)]
        # FFmpeg may log an extra filtered frame before an output cap stops it.
        if len(stamps) < len(files) or any(not math.isfinite(t) for t in stamps):
            raise ValueError()
        if any(b < a for a, b in zip(stamps, stamps[1:])):
            raise ValueError()
    except ValueError:
        raise SystemExit("ffmpeg output is missing valid frame timestamps; refusing to mislabel images.") from None
    return [{"index": i, "timestamp_seconds": round(offset + stamps[i], 6),
             "path": str(path), "reason": reason} for i, path in enumerate(files)]


def extract(
    video_path: str, out_dir: Path, fps: float, resolution: int = 512,
    max_frames: int = 100, start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> list[dict]:
    """Uniform sampling: the first real frame in each 1/rate bucket across the range."""
    validate_controls(resolution, max_frames, start_seconds, end_seconds, fps)
    duration = get_metadata(video_path)["duration_seconds"]
    start = start_seconds or 0.0
    end = min(duration, end_seconds) if end_seconds is not None else duration
    if end <= start:
        raise SystemExit("Requested range is outside the video duration")
    rate = min(fps, MAX_FPS, max_frames / (end - start))
    _clear(out_dir, "frame")
    # Accurate input seeking resets t relative to the seek point. select keeps the
    # first real frame in each bucket; the fps filter would shift the pixels.
    vf = (f"select='isnan(prev_selected_t)+gt(floor(t*{rate:.17g}),"
          f"floor(prev_selected_t*{rate:.17g}))',{_scale_filter(resolution)}")
    result = _ffmpeg_frames(video_path, out_dir / "frame_%04d.jpg", vf, start=start, end=end, max_frames=max_frames)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg frame extraction failed: {diagnostic(result.stderr)}")
    files = sorted(out_dir.glob("frame_*.jpg"))
    if not files:
        raise SystemExit("ffmpeg produced no video frames in the requested range")
    return _emitted_frames(files, result.stderr, start, "uniform")


def extract_scene_candidates(
    video_path: str, out_dir: Path, resolution: int = 512, max_frames: int | None = 100,
    start_seconds: float | None = None, end_seconds: float | None = None,
    threshold: float = SCENE_THRESHOLD,
) -> list[dict]:
    """First frame plus ffmpeg scene-change frames. max_frames lets ffmpeg stop early; None keeps all."""
    validate_controls(resolution, max_frames, start_seconds, end_seconds)
    _clear(out_dir, "frame")
    vf = f"select='eq(n\\,0)+gt(scene\\,{threshold})',{_scale_filter(resolution)}"
    result = _ffmpeg_frames(video_path, out_dir / "frame_%04d.jpg", vf, start=start_seconds,
                            end=end_seconds, max_frames=max_frames)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg scene extraction failed: {diagnostic(result.stderr)}")
    out = _emitted_frames(sorted(out_dir.glob("frame_*.jpg")), result.stderr, start_seconds or 0.0, "scene-change")
    if out:
        out[0]["reason"] = "first-frame"
    return out


def extract_at_timestamps(
    video_path: str, out_dir: Path, timestamps: list[float], resolution: int = 512,
    max_frames: int | None = None, start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> tuple[list[dict], dict]:
    """One frame at each absolute source timestamp (transcript cues).

    Cues outside an active [start, end] window are dropped; more cues than max_frames are
    even-sampled. The cue_*.jpg prefix lets these coexist with detail frame_*.jpg output.
    """
    validate_controls(resolution, max_frames, start_seconds, end_seconds)
    _clear(out_dir, "cue")
    lo = start_seconds or 0.0
    hi = end_seconds if end_seconds is not None else float("inf")
    requested = sorted({parse_time(t) for t in timestamps})
    in_window = [t for t in requested if lo <= t <= hi]
    points = in_window if max_frames is None else [in_window[i] for i in _even_indices(len(in_window), max_frames)]

    out: list[dict] = []
    for t in points:
        path = out_dir / f"cue_{len(out):04d}.jpg"
        result = _ffmpeg_frames(video_path, path, _scale_filter(resolution), start=t, max_frames=1)
        if result.returncode != 0:
            raise SystemExit(f"ffmpeg cue extraction failed: {diagnostic(result.stderr)}")
        if not path.exists():
            continue
        frame = _emitted_frames([path], result.stderr, t, "transcript-cue")[0]
        if frame["timestamp_seconds"] > hi:
            path.unlink()
            continue
        frame.update(index=len(out), requested_timestamp_seconds=t)
        out.append(frame)

    return out, {"engine": "timestamps", "candidate_count": len(requested), "selected_count": len(out),
                 "dropped_out_of_window": len(requested) - len(in_window), "fallback": False}


# ---- selection -------------------------------------------------------------------------------

def _even_indices(count: int, n: int) -> list[int]:
    """Indices of n evenly spaced items out of count, keeping first and last."""
    if n >= count:
        return list(range(count))
    if n <= 1:
        return [0]
    return [round(i * (count - 1) / (n - 1)) for i in range(n)]


def _discard(frames: list[dict]) -> None:
    for frame in frames:
        try:
            Path(frame["path"]).unlink()
        except OSError:
            pass


def _reindex(frames: list[dict]) -> list[dict]:
    for i, frame in enumerate(frames):
        frame["index"] = i
    return frames


def merge_frames(primary: list[dict], pinned: list[dict]) -> list[dict]:
    """Chronological union. Pinned cue frames are never dropped; their budget is reserved upstream."""
    return _reindex(sorted([*primary, *pinned], key=lambda f: f["timestamp_seconds"]))


def _even_sample(candidates: list[dict], n: int) -> list[dict]:
    """Keep n evenly spaced candidates (first and last included), deleting the rest."""
    selected = [candidates[i] for i in _even_indices(len(candidates), n)]
    keep = {frame["path"] for frame in selected}
    _discard([c for c in candidates if c["path"] not in keep])
    return _reindex(selected)


def _frame_delta(a: bytes, b: bytes) -> float:
    """Mean absolute per-byte difference (0-255). Mismatched lengths count as maximally different."""
    if not a or len(a) != len(b):
        return float("inf")
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _thumb_frames(paths: list[Path]) -> list[bytes]:
    """Decode a numbered JPEG sequence to DEDUP_THUMB-square RGB thumbnails in one ffmpeg pass.

    Fail-open: any error or byte-count mismatch returns [] so dedup is skipped, not extraction.
    """
    if not paths:
        return []
    m = re.match(r"(.*?)(\d+)(\.[A-Za-z0-9]+)$", paths[0].name)
    if m is None:
        return []
    prefix, digits, ext = m.groups()
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-start_number", str(int(digits)),
        "-i", str(paths[0].parent / f"{prefix}%0{len(digits)}d{ext}"),
        "-vf", f"scale={DEDUP_THUMB}:{DEDUP_THUMB},format=rgb24", "-f", "rawvideo", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
    except OSError:
        return []
    chunk = DEDUP_THUMB * DEDUP_THUMB * 3
    if result.returncode != 0 or len(result.stdout) != chunk * len(paths):
        return []
    return [result.stdout[i * chunk:(i + 1) * chunk] for i in range(len(paths))]


def dedupe_perceptual(candidates: list[dict], threshold: float = DEDUP_THRESHOLD) -> tuple[list[dict], int]:
    """Drop near-identical frames from a chronological list; returns (survivors, dropped_count)."""
    if len(candidates) <= 1:
        return candidates, 0
    return _dedupe_by_deltas(candidates, _thumb_frames([Path(c["path"]) for c in candidates]), threshold)


def _dedupe_by_deltas(candidates: list[dict], thumbs: list[bytes],
                      threshold: float = DEDUP_THRESHOLD) -> tuple[list[dict], int]:
    """Greedily drop frames within threshold of the last kept frame. Unchanged unless thumbs align 1:1."""
    if len(thumbs) != len(candidates) or len(candidates) <= 1:
        return candidates, 0
    kept, dropped, last = [candidates[0]], [], thumbs[0]
    for cand, thumb in zip(candidates[1:], thumbs[1:]):
        if _frame_delta(thumb, last) <= threshold:
            dropped.append(cand)
        else:
            kept.append(cand)
            last = thumb
    _discard(dropped)
    return _reindex(kept), len(dropped)


def _select(candidates: list[dict], engine: str, max_frames: int | None, dedup: bool,
            fallback: bool = False) -> tuple[list[dict], dict]:
    """Drop near-duplicates, then even-sample down to the cap across the whole range."""
    deduped, dropped = dedupe_perceptual(candidates) if dedup else (candidates, 0)
    selected = _even_sample(deduped, len(deduped) if max_frames is None else max_frames)
    return selected, {"engine": engine, "candidate_count": len(candidates), "deduped_count": dropped,
                      "selected_count": len(selected), "fallback": fallback}


def extract_scene_or_uniform(
    video_path: str, out_dir: Path, fps: float, target_frames: int, resolution: int = 512,
    max_frames: int | None = 100, start_seconds: float | None = None,
    end_seconds: float | None = None, dedup: bool = True,
) -> tuple[list[dict], dict]:
    """Scene changes, or uniform sampling when the clip is effectively static.

    Detection runs uncapped over the whole range before sampling down. Capping it with
    -frames:v would keep only the first cuts, dropping the tail of long videos and possibly
    misfiring the static fallback on a cut-heavy clip.
    """
    scenes = extract_scene_candidates(video_path, out_dir, resolution=resolution, max_frames=None,
                                      start_seconds=start_seconds, end_seconds=end_seconds)
    if len(scenes) >= SCENE_MIN_FRAMES:
        return _select(scenes, "scene", max_frames, dedup)
    cap = target_frames if max_frames is None else min(max_frames, target_frames)
    uniform = extract(video_path, out_dir, fps=fps, resolution=resolution, max_frames=cap,
                      start_seconds=start_seconds, end_seconds=end_seconds)
    return _select(uniform, "uniform", max_frames, dedup, fallback=True)


def extract_keyframes(
    video_path: str, out_dir: Path, resolution: int = 512, max_frames: int | None = 50,
    start_seconds: float | None = None, end_seconds: float | None = None, dedup: bool = True,
) -> tuple[list[dict], dict]:
    """Decode only keyframes (-skip_frame nokey): the cheap tier.

    Encoders place keyframes at scene cuts, so they approximate distinct moments.
    Too few keyframes falls back to uniform sampling over the same range.
    """
    validate_controls(resolution, max_frames, start_seconds, end_seconds)
    _clear(out_dir, "frame")
    result = _ffmpeg_frames(video_path, out_dir / "frame_%04d.jpg", _scale_filter(resolution),
                            start=start_seconds, end=end_seconds, input_args=("-skip_frame", "nokey"))
    files = sorted(out_dir.glob("frame_*.jpg"))
    if result.returncode != 0:
        # An interval between keyframes can make ffmpeg fail with no output; that is a fallback case.
        if files:
            raise SystemExit(f"ffmpeg keyframe extraction failed after partial output: {diagnostic(result.stderr)}")
        print(f"[watch] no keyframe candidates; trying normal decoding. {diagnostic(result.stderr, 700)}",
              file=sys.stderr)
    candidates = _emitted_frames(files, result.stderr, start_seconds or 0.0, "keyframe")
    if len(candidates) >= KEYFRAME_MIN:
        return _select(candidates, "keyframe", max_frames, dedup)

    _discard(candidates)
    duration = get_metadata(video_path)["duration_seconds"]
    end = min(end_seconds, duration) if end_seconds is not None else duration
    budget = max_frames if max_frames is not None else 100
    fps, _ = auto_fps(max(0.0, end - (start_seconds or 0.0)), max_frames=budget)
    uniform = extract(video_path, out_dir, fps=fps, resolution=resolution, max_frames=budget,
                      start_seconds=start_seconds, end_seconds=end_seconds)
    return _select(uniform, "uniform", max_frames, dedup, fallback=True)
