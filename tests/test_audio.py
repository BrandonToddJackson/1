from pathlib import Path

from pipeline import audio


def test_build_filter_chain_default_uses_afftdn():
    chain = audio._build_filter_chain(None)
    assert "afftdn=nf=-25" in chain
    assert "arnndn" not in chain


def test_build_filter_chain_with_rnnoise_model():
    chain = audio._build_filter_chain(Path("/models/rnnoise.rnnn"))
    assert "arnndn=model=/models/rnnoise.rnnn" in chain
    assert "afftdn" not in chain


def test_build_filter_chain_includes_all_stages():
    chain = audio._build_filter_chain(None)
    assert "highpass=f=80" in chain
    assert "deesser" in chain
    assert "speechnorm" in chain
    assert "alimiter" in chain


def test_extract_json_stats_finds_last_balanced_object():
    stderr = 'some progress text\n{"a": 1}\nmore text\n{"output_i": "-16.0", "input_i": "-20.0"}\n'
    stats = audio._extract_json_stats(stderr)
    assert stats == {"output_i": "-16.0", "input_i": "-20.0"}


def test_extract_json_stats_returns_none_when_no_json():
    assert audio._extract_json_stats("no json here at all") is None


def test_extract_json_stats_returns_none_on_malformed_json():
    assert audio._extract_json_stats("{not: valid json}") is None


def test_measure_loudness_returns_none_on_subprocess_failure(monkeypatch, tmp_path):
    from pipeline import procutil

    def fake_run_or_raise(cmd, label):
        raise procutil.SubprocessFailedError("ffmpeg exploded")

    monkeypatch.setattr(audio, "run_or_raise", fake_run_or_raise)
    result = audio._measure_loudness(tmp_path / "in.mp4", "highpass=f=80")
    assert result is None


def test_enhance_falls_back_to_single_pass_when_measurement_missing_keys(monkeypatch, tmp_path):
    """A measurement JSON missing expected keys must not crash enhance() --
    falls back to single-pass loudnorm with a logged warning."""
    class FakeProc:
        stderr = '{"output_i": "-16.0"}'  # missing input_i/input_tp/etc

    calls = []

    def fake_run_or_raise(cmd, label):
        calls.append(cmd)
        return FakeProc()

    monkeypatch.setattr(audio, "run_or_raise", fake_run_or_raise)
    out_path = tmp_path / "out.mp4"

    result = audio.enhance(tmp_path / "in.mp4", out_path, target_lufs=-16.0)

    assert result == -16.0
    assert len(calls) == 2  # measure pass + apply pass
    # The apply-pass command must NOT contain measured_I (fell back to single-pass).
    apply_cmd = calls[1]
    af_arg = apply_cmd[apply_cmd.index("-af") + 1]
    assert "measured_I" not in af_arg
