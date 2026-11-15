#!/usr/bin/env python3
"""Base preflight, engine/detail/ASR configuration, and the transcribe.cpp installer."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from config import (ASR_BACKENDS, CONFIG_DIR, CONFIG_FILE, ConfigError, DETAILS, ENGINES, get_config,
                    load_gemini_key, read_env_file, resolve_engine, write_settings)
from runtime import configure_stdio, diagnostic, run_text

REQUIRED_BINARIES = ['ffmpeg', 'ffprobe', 'yt-dlp']
_PERM_WARNED: set[str] = set()
ENV_TEMPLATE = '''# /watch configuration. No shell interpolation; last assignment wins.
# WATCH_ENGINE=auto|gemini|local. auto uses Gemini when GEMINI_API_KEY is set.
# With the gemini engine, local videos are uploaded to Google for analysis.
# Native captions always come first. WATCH_ASR_BACKEND=transcribe-cpp|none transcribes the rest.
GEMINI_API_KEY=
'''


def _check_binaries():
    return [name for name in REQUIRED_BINARIES if not shutil.which(name)]


def _check_file_permissions(path: Path) -> None:
    # Native Windows mode bits do not describe NTFS ACLs. WSL Linux homes do.
    if os.name == 'nt' or platform.system() == 'Windows' or str(path) in _PERM_WARNED:
        return
    try:
        if path.stat().st_mode & 0o044:
            _PERM_WARNED.add(str(path))
            print(f'[watch] WARNING: {path} is readable by other users. Run: chmod 600 "{path}"', file=sys.stderr)
    except OSError:
        pass


def is_first_run():
    return read_env_file(CONFIG_FILE).get('SETUP_COMPLETE') != 'true'


def _scaffold_env():
    if CONFIG_FILE.exists():
        read_env_file(CONFIG_FILE)  # Fail safely before modifying bad config.
        return
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(CONFIG_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(ENV_TEMPLATE)


def _write_setup_complete():
    _scaffold_env()
    write_settings({'SETUP_COMPLETE': 'true'}, CONFIG_FILE)


def _packages(missing):
    """ffprobe ships with the ffmpeg package."""
    return list(dict.fromkeys('ffmpeg' if name == 'ffprobe' else name for name in missing))


def _install_step(cmd, step):
    print(f"[setup] {step}: {' '.join(map(str, cmd))}", file=sys.stderr, flush=True)
    try:
        result = subprocess.run(list(map(str, cmd)))
    except OSError as exc:
        raise SystemExit(f'{step}: cannot run {cmd[0]} ({type(exc).__name__}); install or repair that executable, then retry.') from None
    if result.returncode:
        raise SystemExit(f'{step} failed (exit {result.returncode}): retry setup after checking the error above.')


def _install_hint(system, missing):
    hints = []
    if 'ffmpeg' in _packages(missing):
        hints.append('winget install --id Gyan.FFmpeg --exact' if system == 'Windows'
                     else 'apt: sudo apt install ffmpeg (or the equivalent package for your distribution)')
    if 'yt-dlp' in missing:
        hints.extend(['winget install --id yt-dlp.yt-dlp --exact', 'winget install --id DenoLand.Deno --exact']
                     if system == 'Windows' else
                     ['Install pipx, then: pipx install "yt-dlp[default,curl-cffi]"; pipx ensurepath. '
                      'For YouTube also install Deno: https://deno.com/'])
    if system == 'Windows':
        hints.append('Reopen your terminal/agent after PATH changes.')
    return '\n'.join(hints)


def _install_binaries(missing) -> bool:
    """Homebrew installs on macOS; other systems get commands to run. True when nothing is missing afterwards."""
    system = platform.system()
    if system != 'Darwin':
        print(_install_hint(system, missing), file=sys.stderr)
        return False
    if not shutil.which('brew'):
        print('[setup] Homebrew is missing. Install it from https://brew.sh, then rerun setup.', file=sys.stderr)
        return False
    _install_step(['brew', 'install', *_packages(missing)], 'Install media dependencies')
    print('[setup] Installed media dependencies with Homebrew.', file=sys.stderr)
    return not _check_binaries()


def _probe(cmd):
    try:
        result = run_text(cmd, timeout=20)
        output = result.stdout or result.stderr
        return {'ok': result.returncode == 0, 'version_line': output.splitlines()[0] if output else None,
                'output': diagnostic(output, 2500)}
    except SystemExit as exc:
        return {'ok': False, 'output': str(exc)}


def _transcribe_cpp_ready(cfg):
    # File checks only: --check must never load native code.
    from local_transcribe_cpp import MODEL, VAD, find_model, runtime_ready
    return bool(runtime_ready() and find_model(MODEL, cfg['transcribe_cpp_model'])
                and find_model(VAD, cfg['transcribe_cpp_vad']))


def _status(detailed=False):
    missing = _check_binaries()
    _check_file_permissions(CONFIG_FILE)
    cfg = get_config()
    gemini_key = bool(load_gemini_key())
    try:
        engine = resolve_engine(cfg['engine'], gemini_key)
    except ConfigError:
        engine = 'gemini'  # Chosen explicitly but the key is missing; gemini_key_present reports it.
    binaries_required = engine == 'local'
    blocked = bool(missing) and binaries_required
    tcpp_ready = _transcribe_cpp_ready(cfg)
    first_run = is_first_run()
    result = {'status': 'needs_install' if blocked else 'ready', 'can_proceed': not blocked,
              'engine': engine, 'configured_engine': cfg['engine'], 'gemini_key_present': gemini_key,
              'gemini_model': cfg['gemini_model'], 'binaries_required': binaries_required,
              'first_run': first_run, 'setup_complete': not first_run, 'missing_binaries': missing,
              'asr_backend': cfg['asr_backend'],
              'asr_ready': cfg['asr_backend'] == 'none' or tcpp_ready,
              'transcribe_cpp_ready': tcpp_ready,
              'config_file': str(CONFIG_FILE), 'watch_detail': cfg['detail'], 'platform': platform.system()}
    if detailed:
        tools = {}
        for name in REQUIRED_BINARIES:
            path = shutil.which(name)
            tools[name] = {'path': path, **(_probe([path, '--version' if name == 'yt-dlp' else '-version']) if path else {'ok': False})}
        ytdlp = shutil.which('yt-dlp')
        result['tools'] = tools
        result['youtube'] = {
            'deno': shutil.which('deno'), 'node': shutil.which('node'),
            'ejs': 'unknown (inspect the owning yt-dlp installation; no network probe)',
            'impersonation': _probe([ytdlp, '--ignore-config', '--list-impersonate-targets']) if ytdlp else None,
            'update_hint': 'Update the owning package: brew upgrade yt-dlp; pipx upgrade yt-dlp; or winget upgrade --id yt-dlp.yt-dlp --exact. Verify the executable path above.',
        }
    return result


def cmd_check():
    # ASR readiness never blocks base watch.
    status = _status()
    if status['can_proceed']:
        return 0
    print(f"[watch] Missing {', '.join(status['missing_binaries'])}. Run python3 \"{Path(__file__).resolve()}\".", file=sys.stderr)
    return 2


def cmd_json():
    print(json.dumps(_status(detailed=True), indent=2))
    return 0


def cmd_install(asr=None, detail=None, engine=None):
    if engine:
        _scaffold_env()
        write_settings({'WATCH_ENGINE': engine}, CONFIG_FILE)
    if engine == 'gemini':
        if not load_gemini_key():
            print(f'[setup] Add GEMINI_API_KEY privately to {CONFIG_FILE} (free key: https://aistudio.google.com/apikey), '
                  'then rerun --engine gemini. Local videos will be uploaded to Google for analysis.', file=sys.stderr)
            return 3
        _write_setup_complete()
        missing = _check_binaries()
        note = f" Optional: install {', '.join(_packages(missing))} for non-YouTube URLs and --engine local." if missing else ''
        print(f'[setup] Gemini engine is ready. Configuration: {CONFIG_FILE}.{note}')
        return 0
    missing = _check_binaries()
    if missing and not _install_binaries(missing):
        return 2
    _scaffold_env()
    if detail:
        write_settings({'WATCH_DETAIL': detail}, CONFIG_FILE)
    if asr == 'transcribe-cpp':
        return install_transcribe_cpp()
    if asr:
        write_settings({'WATCH_ASR_BACKEND': asr, 'SETUP_COMPLETE': 'true'}, CONFIG_FILE)
    print(f'[setup] Base watch is ready. Configuration: {CONFIG_FILE}')
    return 0


def install_transcribe_cpp():
    import local_transcribe_cpp as tcpp
    print(f'[setup] Local transcribe.cpp {tcpp.VERSION}: downloads an 18 MB runtime, plus the 250 MB SenseVoiceSmall '
          'model unless the Hugging Face cache already has it. No Python packages or PyTorch.', file=sys.stderr, flush=True)
    cfg = tcpp.settings()
    _scaffold_env()
    if not tcpp.runtime_ready():
        tcpp.install_runtime()
    model = tcpp.ensure_model(tcpp.MODEL, cfg['transcribe_cpp_model'])
    vad = tcpp.ensure_model(tcpp.VAD, cfg['transcribe_cpp_vad'])
    cfg.update(transcribe_cpp_model=str(model), transcribe_cpp_vad=str(vad))
    with tempfile.TemporaryDirectory(prefix='watch-warmup-') as temporary:
        audio = Path(temporary) / 'silence.wav'
        _install_step(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                       'anullsrc=r=16000:cl=mono', '-t', '2', str(audio)], 'Create warm-up audio')
        tcpp.transcribe_media(audio, Path(temporary), cfg, require_ready=False)
    tcpp.mark_ready()
    write_settings({'WATCH_ASR_BACKEND': 'transcribe-cpp', 'WATCH_TRANSCRIBE_CPP_MODEL': str(model),
                    'WATCH_TRANSCRIBE_CPP_VAD': str(vad), 'SETUP_COMPLETE': 'true'}, CONFIG_FILE)
    print(f'[setup] transcribe.cpp is ready (model {model}).', file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Fast, silent base preflight')
    mode.add_argument('--json', action='store_true', help='Detailed status as JSON')
    parser.add_argument('--asr', choices=sorted(ASR_BACKENDS),
                        help='Transcription for captionless videos; transcribe-cpp installs and warms it up')
    parser.add_argument('--detail', choices=sorted(DETAILS))
    parser.add_argument('--engine', choices=sorted(ENGINES))
    args = parser.parse_args()
    try:
        if args.check:
            return cmd_check()
        if args.json:
            return cmd_json()
        return cmd_install(args.asr, args.detail, args.engine)
    except (ConfigError, OSError, ValueError) as exc:
        print(f'[setup] {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    configure_stdio()
    raise SystemExit(main())
