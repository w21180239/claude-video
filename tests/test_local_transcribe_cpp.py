"""Exercise the real worker child process against a fake transcribe_cpp binding; no native code."""
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import config
import local_transcribe_cpp as tcpp
import setup
import whisper

FAKE_BINDING = '''
import os


class TranscribeError(Exception):
    pass


class OutputTruncated(TranscribeError):
    pass


class InputTooShort(TranscribeError):
    pass


class _Span:
    def __init__(self, start, end):
        self.start_sample, self.end_sample = start, end


class _Vad:
    def run(self, pcm, **options):
        assert options == {'min_silence_ms': 500, 'speech_pad_ms': 200, 'max_speech_ms': 20000}
        if os.environ.get('WATCH_TEST_MODE') == 'silence':
            return type('R', (), {'segments': ()})()
        n = len(pcm)
        return type('R', (), {'segments': (_Span(0, n // 4), _Span(n // 4, n // 4 + 10), _Span(n // 2, n))})()


class _Result:
    def __init__(self, text, language):
        self.text, self.language = text, language


class _Session:
    calls = 0

    def run(self, pcm, language=None):
        _Session.calls += 1
        if len(pcm) == 10:
            raise InputTooShort()
        if _Session.calls == 3:
            exc = OutputTruncated()
            exc.partial_result = _Result(' partial ', 'zh')
            raise exc
        return _Result(' hello world ', language or 'en')


class Model:
    def __init__(self, path):
        if os.environ.get('WATCH_TEST_MODE') == 'crash':
            raise SystemExit(7)
        self.path = path

    def vad_session(self):
        return _Vad()

    def session(self):
        return _Session()


def set_log_callback(handler):
    handler(1, 'chatter')
'''


@pytest.fixture
def runtime(tmp_path):
    home = tcpp.HOME
    (home / 'site' / 'transcribe_cpp').mkdir(parents=True)
    (home / 'site' / 'transcribe_cpp' / '__init__.py').write_text(FAKE_BINDING, encoding='utf-8')
    (home / 'native' / 'lane').mkdir(parents=True)
    (home / 'native' / 'lane' / 'transcribe.dll').write_bytes(b'')
    (home / 'models').mkdir()
    for _, name, _ in (tcpp.MODEL, tcpp.VAD):
        (home / 'models' / name).write_bytes(b'gguf')
    tcpp.mark_ready()
    return home


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / 'clip with spaces.wav'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                    'sine=frequency=440:sample_rate=16000', '-t', '2', str(path)], check=True)
    return path


def test_worker_maps_vad_spans_to_segments(runtime, audio, tmp_path, capfd):
    segments = tcpp.transcribe_media(audio, tmp_path / 'work')
    assert segments == [{'start': 0.0, 'end': 0.5, 'text': 'hello world'},
                        {'start': 1.0, 'end': 2.0, 'text': 'partial'}]
    assert segments.language == 'en' and segments.model == tcpp.MODEL[1]
    err = capfd.readouterr().err
    assert '3/3 speech chunks' in err and 'chatter' not in err


def test_configured_language_is_forwarded(runtime, audio, tmp_path, monkeypatch):
    monkeypatch.setenv('WATCH_TRANSCRIBE_CPP_LANGUAGE', 'zh')
    assert tcpp.transcribe_media(audio, tmp_path / 'work')[0]['text'] == 'hello world'


def test_silence_reports_no_speech(runtime, audio, tmp_path, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'silence')
    segments = tcpp.transcribe_media(audio, tmp_path / 'work')
    assert segments == [] and segments.no_speech


def test_worker_crash_is_a_clean_failure(runtime, audio, tmp_path, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'crash')
    with pytest.raises(SystemExit, match='exit 7'):
        tcpp.transcribe_media(audio, tmp_path / 'work')


def test_timeout_stops_worker(runtime, audio, tmp_path, monkeypatch):
    monkeypatch.setenv('WATCH_TRANSCRIBE_CPP_TIMEOUT', '5')
    real = subprocess.run
    def slow(*args, **kwargs):
        if '--worker' not in args[0]:
            return real(*args, **kwargs)
        assert kwargs['timeout'] == 5
        raise subprocess.TimeoutExpired(args[0], 5)
    monkeypatch.setattr(tcpp.subprocess, 'run', slow)
    with pytest.raises(SystemExit, match='timed out'):
        tcpp.transcribe_media(audio, tmp_path / 'work')


def test_missing_runtime_points_to_installer(audio, tmp_path):
    with pytest.raises(SystemExit, match='--install-transcribe-cpp'):
        tcpp.transcribe_media(audio, tmp_path / 'work')


def test_worker_runs_isolated_with_binding_dir():
    cmd = tcpp.worker_command(Path('a.f32'), Path('o.json'), Path('m.gguf'), Path('v.gguf'), 'en')
    assert cmd[1] == '-I' and '--site' in cmd and cmd[-2:] == ['--language', 'en']
    assert '--language' not in tcpp.worker_command(Path('a'), Path('o'), Path('m'), Path('v'), '')


def test_model_lookup_prefers_config_then_hf_cache_then_own_copy(tmp_path, monkeypatch):
    repo, name, _ = tcpp.MODEL
    assert tcpp.find_model(tcpp.MODEL) is None
    own = tcpp.HOME / 'models' / name
    own.parent.mkdir(parents=True)
    own.write_bytes(b'')
    assert tcpp.find_model(tcpp.MODEL) == own
    cached = tmp_path / 'hub' / f"models--{repo.replace('/', '--')}" / 'snapshots' / 'abc' / name
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b'')
    monkeypatch.setenv('HF_HUB_CACHE', str(tmp_path / 'hub'))
    assert tcpp.find_model(tcpp.MODEL) == cached
    assert tcpp.find_model(tcpp.MODEL, str(own)) == own
    assert tcpp.find_model(tcpp.MODEL, str(tmp_path / 'missing.gguf')) is None


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(tcpp.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(b'tampered'))
    target = tmp_path / 'model.gguf'
    with pytest.raises(SystemExit, match='Checksum mismatch'):
        tcpp._download('https://example.invalid/model.gguf', target, '0' * 64)
    assert not target.exists() and not list(tmp_path.glob('.download-*'))


def test_invalid_language_rejected(monkeypatch):
    monkeypatch.setenv('WATCH_TRANSCRIBE_CPP_LANGUAGE', 'English')
    with pytest.raises(config.ConfigError):
        tcpp.settings()


def test_whisper_dispatches_without_extracting_mp3(monkeypatch, tmp_path):
    monkeypatch.setattr(whisper, 'extract_audio', lambda *a: pytest.fail('decodes PCM itself'))
    monkeypatch.setattr(tcpp, 'transcribe_media', lambda media, work: ['ok'])
    assert whisper.transcribe_video('v.mp4', tmp_path / 'audio.mp3', backend='transcribe-cpp') == (['ok'], 'transcribe-cpp')
    with pytest.raises(SystemExit, match='API key'):
        whisper.transcribe_video('v.mp4', tmp_path / 'audio.mp3', backend='transcribe-cpp', api_key='k')


def test_status_reports_readiness_from_files_only(runtime):
    config.write_settings({'WATCH_WHISPER_BACKEND': 'transcribe-cpp'})
    status = setup._status()
    assert status['transcribe_cpp_ready'] and status['backend_ready']
    assert 'transcribe_cpp' not in sys.modules


def test_status_not_ready_without_runtime():
    config.write_settings({'WATCH_WHISPER_BACKEND': 'transcribe-cpp'})
    status = setup._status()
    assert not status['transcribe_cpp_ready'] and not status['backend_ready']


def test_installer_writes_settings_after_warmup(runtime, monkeypatch):
    monkeypatch.setattr(tcpp, 'install_runtime', lambda: pytest.fail('runtime already present'))
    assert setup.cmd_install('transcribe-cpp') == 0
    cfg = config.get_config()
    assert cfg['whisper_backend'] == 'transcribe-cpp'
    assert cfg['transcribe_cpp_model'].endswith(tcpp.MODEL[1]) and cfg['transcribe_cpp_vad'].endswith(tcpp.VAD[1])
    assert not setup.is_first_run()
