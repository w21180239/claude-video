"""Local transcribe.cpp backend: Silero VAD chunks, then a GGUF ASR model per chunk.

Default model is SenseVoiceSmall (Q8_0; zh, en, yue, ja, ko). It emits no timestamps, so each VAD chunk's
span becomes the segment timing. Inference runs in a child process so a native crash
cannot take down watch. The Python binding is stdlib-only (ctypes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from typing import NamedTuple
from pathlib import Path

VERSION = '0.4.2'
RELEASE_URL = f'https://github.com/handy-computer/transcribe.cpp/releases/download/v{VERSION}'
WHEEL = (f'transcribe_cpp-{VERSION}-py3-none-any.whl',
         '2e4a5d3e7ba78918969ccb1ee06f3d5c6c79664e0626d0da0ffe832f465e6d51')
NATIVE = {
    ('Windows', 'x86_64'): ('windows-x86_64-cpu-vulkan', '6094102cf57b404b050707ceb258b12ce647b92e801b5d46077214e5b030c8aa'),
    ('Linux', 'x86_64'): ('linux-x86_64-cpu-vulkan', '4fe87db8919ea34e461db7987713b580f54e6f3ed610457c5d8fb8eac1a15046'),
    ('Linux', 'aarch64'): ('linux-aarch64-cpu-vulkan', '5e9cc7b244949fc686d1272e796e965411e80cc768533f1704c3c53e4c7d695e'),
    ('Darwin', 'arm64'): ('macos-arm64-metal', '75ea3a705f3edec09942fdbb1b2b5509d7ecd75bda55158a47a3ab1e2d1105b3'),
    ('Darwin', 'x86_64'): ('macos-x86_64-cpu', 'ca414e5b28706a2f2ecea1479b3e1c5f47d0f11f43aba86e289ae5c10b1ca40c'),
}
LIBRARY_NAMES = ('transcribe.dll', 'libtranscribe.so', 'libtranscribe.dylib')
# (Hugging Face repo, file, sha256)
MODEL = ('handy-computer/SenseVoiceSmall-gguf', 'SenseVoiceSmall-Q8_0.gguf',
         '6c759ee4c9748c9b3f7a5a60ca74f0f7e685fb9d45d1378fce7cfd62f59adf29')
VAD = ('handy-computer/silero-vad-v6.2-gguf', 'silero-vad-v6.2-F32.gguf',
       '0d871a800c420fdd5d1b17a3aea4296d1c39608f0ff4a9295eb76457eb1c88e6')
HOME = Path.home() / '.cache' / 'watch' / 'transcribe-cpp'
SAMPLE_RATE = 16000
# Merge pauses under 0.5 s; force-split speech past 20 s so timestamps stay fine-grained.
VAD_OPTIONS = {'min_silence_ms': 500, 'speech_pad_ms': 200, 'max_speech_ms': 20000}


class Transcript(NamedTuple):
    segments: list[dict]
    language: str
    model: str


def site_dir() -> Path:
    return HOME / 'site'


def sentinel() -> Path:
    return HOME / '.deps-ok'


def library_path() -> Path | None:
    for name in LIBRARY_NAMES:
        for path in sorted((HOME / 'native').glob(f'*/{name}')):
            return path
    return None


def runtime_ready() -> bool:
    return (sentinel().is_file() and library_path() is not None
            and (site_dir() / 'transcribe_cpp' / '__init__.py').is_file())


def hf_hub_dir() -> Path:
    if os.environ.get('HF_HUB_CACHE'):
        return Path(os.environ['HF_HUB_CACHE'])
    if os.environ.get('HF_HOME'):
        return Path(os.environ['HF_HOME']) / 'hub'
    return Path.home() / '.cache' / 'huggingface' / 'hub'


def find_model(spec: tuple[str, str, str], configured: str = '') -> Path | None:
    """Configured path, else the Hugging Face cache (shared with apps such as Handy), else our own copy."""
    if configured:
        path = Path(configured)
        return path if path.is_file() else None
    repo, name, _ = spec
    for path in sorted((hf_hub_dir() / f"models--{repo.replace('/', '--')}" / 'snapshots').glob(f'*/{name}')):
        if path.is_file():
            return path
    own = HOME / 'models' / name
    return own if own.is_file() else None


def settings(cfg=None) -> dict:
    from config import ConfigError, get_config, positive_number
    cfg = dict(get_config(asr_override='transcribe-cpp') if cfg is None else cfg)
    if cfg['transcribe_cpp_language'] and not re.fullmatch('[a-z]{2,3}', cfg['transcribe_cpp_language']):
        raise ConfigError('WATCH_TRANSCRIBE_CPP_LANGUAGE must be a lowercase language code such as en or zh.')
    cfg['transcribe_cpp_timeout'] = (positive_number(cfg['transcribe_cpp_timeout'], 'WATCH_TRANSCRIBE_CPP_TIMEOUT')
                                     if cfg['transcribe_cpp_timeout'] else None)
    return cfg


# ---- install -------------------------------------------------------------------------------

def _download(url: str, target: Path, sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f'[setup] downloading {url}', file=sys.stderr, flush=True)
    digest = hashlib.sha256()
    fd, name = tempfile.mkstemp(prefix='.download-', dir=target.parent)
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'watch-skill'})
        with os.fdopen(fd, 'wb') as out, urllib.request.urlopen(request, timeout=60) as response:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256:
            raise SystemExit(f'Checksum mismatch for {url}; download rejected.')
        os.replace(name, target)
    except OSError as exc:
        raise SystemExit(f'Download failed for {url} ({type(exc).__name__}: {exc}).') from None
    finally:
        Path(name).unlink(missing_ok=True)


def install_runtime() -> None:
    machine = platform.machine().lower()
    machine = {'amd64': 'x86_64', 'x64': 'x86_64', 'arm64': 'arm64' if platform.system() == 'Darwin' else 'aarch64'}.get(machine, machine)
    lane = NATIVE.get((platform.system(), machine))
    if lane is None:
        raise SystemExit(f'transcribe.cpp {VERSION} ships no prebuilt library for {platform.system()} {machine}.')
    sentinel().unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix='watch-tcpp-') as temporary:
        tmp = Path(temporary)
        wheel = tmp / WHEEL[0]
        _download(f'{RELEASE_URL}/{WHEEL[0]}', wheel, WHEEL[1])
        tarball = tmp / f'transcribe-native-{VERSION}-{lane[0]}.tar.gz'
        _download(f'{RELEASE_URL}/{tarball.name}', tarball, lane[1])
        for sub in ('site', 'native'):
            shutil.rmtree(HOME / sub, ignore_errors=True)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(site_dir())
        with tarfile.open(tarball) as archive:
            archive.extractall(HOME / 'native', filter='data')
    if library_path() is None:
        raise SystemExit('transcribe.cpp archive contained no native library.')


def ensure_model(spec: tuple[str, str, str], configured: str = '') -> Path:
    found = find_model(spec, configured)
    if found:
        return found
    if configured:
        raise SystemExit(f'Configured model file not found: {configured}')
    repo, name, sha256 = spec
    target = HOME / 'models' / name
    _download(f'https://huggingface.co/{repo}/resolve/main/{name}', target, sha256)
    return target


def mark_ready() -> None:
    sentinel().write_text(VERSION + '\n', encoding='utf-8')


# ---- inference -----------------------------------------------------------------------------

def decode_pcm(media: Path, out: Path) -> Path:
    from runtime import run_text
    result = run_text(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(media.resolve()),
                       '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE), '-f', 'f32le', str(out.resolve())])
    if result.returncode != 0:
        raise SystemExit(f'ffmpeg audio decode failed: {result.stderr.strip()}')
    if not out.exists() or out.stat().st_size == 0:
        raise SystemExit('ffmpeg produced no audio; the media may have no audio track.')
    return out


def worker_command(pcm: Path, output: Path, model: Path, vad: Path, language: str) -> list[str]:
    cmd = [sys.executable, '-I', str(Path(__file__).resolve()), '--worker', str(pcm), str(output),
           '--site', str(site_dir()), '--model', str(model), '--vad', str(vad)]
    return cmd + (['--language', language] if language else [])


def transcribe_media(media: Path, work: Path, cfg=None, *, require_ready=True) -> Transcript:
    from transcribe import normalize_segments
    cfg = settings(cfg)
    if require_ready and not runtime_ready():
        raise SystemExit('transcribe.cpp runtime missing. Run setup.py --asr transcribe-cpp.')
    model = find_model(MODEL, cfg['transcribe_cpp_model'])
    vad = find_model(VAD, cfg['transcribe_cpp_vad'])
    if model is None or vad is None:
        raise SystemExit('transcribe.cpp model or VAD file missing. Run setup.py --asr transcribe-cpp.')
    print(f'[watch] local transcribe.cpp ({model.name}) transcription...', file=sys.stderr, flush=True)
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='watch-tcpp-', dir=work) as temporary:
        pcm = decode_pcm(media, Path(temporary) / 'audio.f32')
        output = Path(temporary) / 'segments.json'
        env = dict(os.environ, TRANSCRIBE_LIBRARY=str(library_path()), PYTHONIOENCODING='utf-8')
        try:
            result = subprocess.run(worker_command(pcm, output, model, vad, cfg['transcribe_cpp_language']),
                                    env=env, timeout=cfg['transcribe_cpp_timeout'])
        except subprocess.TimeoutExpired:
            raise SystemExit('transcribe.cpp timed out and was stopped. Increase WATCH_TRANSCRIBE_CPP_TIMEOUT or leave it unset.') from None
        except OSError as exc:
            raise SystemExit(f'Cannot start the transcribe.cpp worker ({type(exc).__name__}).') from None
        if result.returncode != 0:
            raise SystemExit(f'transcribe.cpp failed (exit {result.returncode}); see the error above. '
                             'Rerun setup.py --asr transcribe-cpp if the runtime is broken.')
        try:
            data = json.loads(output.read_text(encoding='utf-8'))
            segments = normalize_segments(data)
        except (OSError, ValueError, TypeError) as exc:
            raise SystemExit(f'transcribe.cpp output missing or malformed ({type(exc).__name__}).') from None
    if not segments:
        print('[watch] transcribe.cpp detected no speech.', file=sys.stderr)
    return Transcript(segments, data.get('language') or 'unknown', model.name)


def worker(argv: list[str]) -> int:
    """Child process entry. Runs with -I, so it imports only the stdlib and the binding."""
    from array import array
    from collections import Counter
    parser = argparse.ArgumentParser()
    parser.add_argument('pcm')
    parser.add_argument('output')
    parser.add_argument('--site', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--vad', required=True)
    parser.add_argument('--language', default=None)
    args = parser.parse_args(argv)
    sys.path.insert(0, args.site)
    import transcribe_cpp as tc
    # Level 3 = error; the rest is ggml allocation chatter.
    tc.set_log_callback(lambda level, message: level == 3 and print(message.rstrip(), file=sys.stderr))

    pcm = array('f')
    pcm.frombytes(Path(args.pcm).read_bytes())
    if sys.byteorder == 'big':
        pcm.byteswap()
    vad = tc.Model(args.vad).vad_session()
    asr = tc.Model(args.model).session()  # Loaded before VAD runs so a broken model fails even on silence.
    spans = vad.run(pcm, **VAD_OPTIONS).segments
    segments, languages = [], Counter()
    step = max(1, len(spans) // 10)
    for index, span in enumerate(spans, 1):
        try:
            result = asr.run(pcm[span.start_sample:span.end_sample], language=args.language)
        except tc.OutputTruncated as exc:  # Repetition loops keep their partial text.
            result = exc.partial_result
        except tc.InputTooShort:
            continue
        text = result.text.strip()
        if text:
            segments.append({'start': round(span.start_sample / SAMPLE_RATE, 3),
                             'end': round(span.end_sample / SAMPLE_RATE, 3), 'text': text})
            languages[result.language] += len(text)
        if index % step == 0 or index == len(spans):
            print(f'[transcribe-cpp] {index}/{len(spans)} speech chunks', file=sys.stderr, flush=True)
    language = languages.most_common(1)[0][0] if languages else None
    Path(args.output).write_text(json.dumps({'language': language, 'segments': segments}, ensure_ascii=False),
                                 encoding='utf-8')
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        raise SystemExit(worker(sys.argv[2:]))
    raise SystemExit('Internal worker; use watch.py or setup.py.')
