"""Setting resolution, dotenv parsing, and Gemini key lookup."""
from __future__ import annotations

import pytest

import config


@pytest.mark.parametrize('env,expected', [(None, 'balanced'), ('efficient', 'efficient'), ('bogus', 'balanced')])
def test_detail_resolution(monkeypatch, env, expected):
    if env:
        monkeypatch.setenv("WATCH_DETAIL", env)
    assert config.get_config()["detail"] == expected


def test_asr_backend_defaults_to_none_and_rejects_unknown(monkeypatch):
    assert config.get_config()["asr_backend"] == "none"
    monkeypatch.setenv("WATCH_ASR_BACKEND", "bogus")
    with pytest.raises(config.ConfigError, match="WATCH_ASR_BACKEND"):
        config.get_config()
    assert config.get_config(asr_override="none")["asr_backend"] == "none"


def test_frame_cap_mapping():
    assert config.frame_cap("efficient") == 50
    assert config.frame_cap("balanced") == 100
    assert config.frame_cap("token-burner") is None
    assert config.frame_cap("transcript") is None
    assert config.frame_cap("anything-else") == 100


@pytest.mark.parametrize('value,expected', [
    ('"a#=b" # comment', 'a#=b'), ("'C:\\Users\\a' # note", 'C:\\Users\\a'),
    ('C:\\Users\\a # note', 'C:\\Users\\a'), ('abc#def=ghi', 'abc#def=ghi'),
    ('"$HOME\\n"', '$HOME\\n'),
])
def test_literal_dotenv_values(tmp_path, value, expected):
    path = tmp_path / '.env'
    path.write_text('KEY=' + value)
    assert config.read_env_file(path)['KEY'] == expected


@pytest.mark.parametrize('value', ['"private-secret', '"private-secret" junk'])
def test_malformed_quotes_never_echo_value(tmp_path, value):
    path = tmp_path / '.env'
    path.write_text('KEY=' + value)
    with pytest.raises(config.ConfigError) as exc:
        config.read_env_file(path)
    assert 'line 1' in str(exc.value) and 'private-secret' not in str(exc.value)


def test_engine_defaults_to_auto_and_local_without_key():
    cfg = config.get_config()
    assert cfg['engine'] == 'auto'
    assert cfg['gemini_model'] == 'gemini-3.7-flash'
    assert cfg['gemini_timeout'] == 600.0
    assert config.load_gemini_key() is None
    assert config.resolve_engine('auto', has_key=False) == 'local'


def test_auto_selects_gemini_when_key_present(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    assert config.load_gemini_key() == 'test-key'
    assert config.resolve_engine('auto', has_key=True) == 'gemini'
    assert config.resolve_engine('local', has_key=True) == 'local'


def test_gemini_key_lookup_order(monkeypatch, tmp_path):
    config.CONFIG_DIR.mkdir(parents=True)
    config.CONFIG_FILE.write_text('GEMINI_API_KEY=from-user-file\n')
    (tmp_path / 'cwd' / '.env').write_text('GEMINI_API_KEY=from-cwd\n')
    assert config.load_gemini_key() == 'from-user-file'
    monkeypatch.setenv('GEMINI_API_KEY', 'from-env')
    assert config.load_gemini_key() == 'from-env'
    monkeypatch.delenv('GEMINI_API_KEY')
    config.CONFIG_FILE.write_text('GEMINI_API_KEY=\n')
    assert config.load_gemini_key() == 'from-cwd'


def test_explicit_gemini_without_key_is_a_safe_error():
    with pytest.raises(config.ConfigError, match='GEMINI_API_KEY'):
        config.resolve_engine('gemini', has_key=False)


def test_invalid_engine_and_timeout_rejected(monkeypatch):
    monkeypatch.setenv('WATCH_ENGINE', 'vertex')
    with pytest.raises(config.ConfigError, match='WATCH_ENGINE'):
        config.get_config()
    monkeypatch.setenv('WATCH_ENGINE', 'auto')
    monkeypatch.setenv('WATCH_GEMINI_TIMEOUT', '-5')
    with pytest.raises(config.ConfigError, match='WATCH_GEMINI_TIMEOUT'):
        config.get_config()
