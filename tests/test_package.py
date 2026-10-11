"""Model-free checks of the package: pooling, captures, the viewer page, the CLI."""

from __future__ import annotations

import numpy as np
import pytest

from cortexlive import capture, reference
from cortexlive.run import RunResult, Volume
from cortexlive.stream import TrPooler
from cortexlive.viewer import html
from cortexlive.viewer.payload import SCALE, scored, zq


def full_resample(src: np.ndarray, n_src: int, n_out: int) -> np.ndarray:
    """The cache's anchored 2 Hz resample, recomputed over the whole buffer."""
    denom = max(1, n_out - 1)
    n_rows = min(n_out, int(np.floor((len(src) - 1) / (n_src - 1) * denom)) + 1)
    xs, xd = np.arange(len(src)) / (n_src - 1), np.arange(n_rows) / denom
    return np.stack([np.interp(xd, xs, src[:, j]) for j in range(src.shape[1])], 1).astype(
        np.float32)


@pytest.mark.parametrize("seed", range(5))
def test_incremental_resample_matches_full_recompute(seed):
    rng = np.random.default_rng(seed)
    pool, blocks = TrPooler(3), []
    pool.anchor(400, 57)
    for _ in range(60):
        if blocks and rng.random() < 0.2:  # audio drops its provisional tail chunk
            keep = int(rng.integers(0, len(blocks)))
            pool.replace_tail(keep)
            blocks = blocks[:keep]
        block = rng.standard_normal((int(rng.integers(1, 12)), 3)).astype(np.float32)
        pool.extend(block, rate=15.0)
        blocks.append(block)
        np.testing.assert_array_equal(pool._resampled(), full_resample(
            np.concatenate(blocks), 400, 57))


class Head:
    subjects = ["sub-01", "sub-02"]
    layers = {"text": 8, "audio": 15, "video": 19}
    window = 5


def test_capture_round_trip(tmp_path):
    out = tmp_path / "captures" / "clip"
    out.mkdir(parents=True)
    vols = [Volume(s=i, lag_s=0.5 + i, tick_ms=100.0, wall_s=1.49 * i,
                   preds={s: np.full(1000, i, np.float32) for s in Head.subjects})
            for i in range(3)]
    result = RunResult(volumes=vols, media_s=4.0, wall_s=5.0, pace="realtime")
    capture.save(out, result, input_path=None, input_name="clip.mp4", device="cpu",
                 transcript="none", head=Head)
    assert "NaN" not in (out / "capture.json").read_text()  # 3 volumes, 4 quarters
    meta, arrays = capture.load(out)
    assert meta["run"]["volumes"] == meta["run"]["expected_volumes"] == 3
    assert arrays["sub-02"].shape == (3, 1000)
    assert capture.find(out.parent) == [out]


def test_finite_replaces_nan_with_null():
    assert capture.finite({"a": [1.0, float("nan")], "b": float("inf")}) == {
        "a": [1.0, None], "b": None}


def test_zq_leaves_missing_volumes_neutral():
    a = np.full((14, 1000), np.nan, np.float32)
    a[6:12] = np.random.default_rng(0).standard_normal((6, 1000))
    q = zq(a)
    assert q.dtype == np.int8 and not q[:6].any() and not q[12:].any() and q[6:12].any()


def test_statistics_never_come_from_the_start_trim():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((8, 1000)).astype(np.float32)
    a[:5] += 50  # the start-trim volumes: their windows reach before the video
    assert not scored(a)[:5].any()
    q = zq(a)
    assert np.abs(q[5:].astype(int)).mean() < 127 * 2 / SCALE  # scaled by the 3 shown rows
    short = a[:7]  # 2 rows past the trim: too few for statistics of their own
    base = (np.zeros(1000, np.float32), np.ones(1000, np.float32))
    assert not zq(short).any() and zq(short, base)[5:].any()


def test_viewer_page_is_wrapped_and_embed_is_escaped():
    page = html({"capture": {"name": "</script><b>x"}})
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    assert "</script><b>x" not in page
    assert "<title>" in page


def test_known_stimulus_from_file_name(monkeypatch):
    monkeypatch.setattr(reference, "_stimuli", lambda: {
        "s06e01a": "friends", "s01e01a": "friends", "life01": "movie10"})
    assert reference.known_stimulus("friends_s06e01a.mkv")["split"] == "validation"
    assert reference.known_stimulus("Friends_S01E01A_clip.mp4")["split"] == "training"
    assert reference.known_stimulus("life01.mkv")["source"] == "movie10"
    assert reference.known_stimulus("holiday.mp4") is None


def test_cli_help(capsys):
    from cortexlive.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "run" in capsys.readouterr().out


def test_text_stream_needs_llama_only_for_words():
    from cortexlive.stream import TextStream

    text = TextStream(layer=8, hidden=2048, device="cpu")
    text.push_tr([])
    text.push_tr([])
    assert not text.loaded  # no words: zero rows, and Llama never loads
    assert text.tr_row(1).shape == (2048,) and not text.tr_row(1).any()


class _Text:
    loaded = False

    def __init__(self, fail: bool = False):
        self.fail = fail

    def load(self):
        if self.fail:
            raise OSError("model.safetensors not found")
        self.loaded = True


class _Enc:
    def __init__(self, fail: bool = False):
        self.text = _Text(fail)


def _known_friends(monkeypatch, llama: bool):
    from cortexlive import stream

    monkeypatch.setattr(reference, "known_stimulus", lambda name: {
        "stimulus": "s06e01a", "source": "friends", "split": "validation"})
    monkeypatch.setattr(reference, "transcript", lambda known: [["hi"], [], ["there"]])
    monkeypatch.setattr(stream, "text_model_installed", lambda: llama)


def test_transcript_without_llama_runs_without_words(monkeypatch, tmp_path):
    from cortexlive.record import choose_transcript
    from cortexlive.run import RunError

    _known_friends(monkeypatch, llama=False)
    logged, enc = [], _Enc()
    known, words, label, note = choose_transcript(enc, "friends_s06e01a.mkv", "auto",
                                                  logged.append)
    assert known["stimulus"] == "s06e01a" and words is None and label == "none"
    assert note == "transcript left out: Llama 3.2 1B is not installed"  # kept in the capture
    assert not enc.text.loaded and "is not installed" in logged[-1]  # the line left on screen
    tsv = tmp_path / "words.tsv"
    tsv.write_text("words_per_tr\n['hi']\n")
    with pytest.raises(RunError, match="needs Llama"):  # asked for by path: say why it can't
        choose_transcript(enc, "clip.mp4", tsv, logged.append)


def test_transcript_loads_llama_before_the_run(monkeypatch):
    from cortexlive.record import choose_transcript

    _known_friends(monkeypatch, llama=True)
    enc = _Enc()
    _, words, label, note = choose_transcript(enc, "friends_s06e01a.mkv", "auto", lambda m: None)
    assert enc.text.loaded and words and label == "CNeuroMod transcript" and note is None
    _, words, _, _ = choose_transcript(_Enc(), "friends_s06e01a.mkv", "none", lambda m: None)
    assert words is None
    # installed but unloadable, e.g. offline with a broken download: drop the transcript
    _, words, label, note = choose_transcript(_Enc(fail=True), "friends_s06e01a.mkv", "auto",
                                              lambda m: None)
    assert words is None and label == "none" and "could not be loaded" in note


def test_llama_counts_as_installed_only_with_every_file(tmp_path):
    from cortexlive.stream import TEXT_MODEL, text_model_installed

    repo = tmp_path / f"models--{TEXT_MODEL.replace('/', '--')}"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("abc123")
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert not text_model_installed(str(tmp_path))  # an interrupted download
    for f in ("model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        (snap / f).write_text("x")
    assert text_model_installed(str(tmp_path))


def test_cli_refuses_a_transcript_without_llama_before_loading(monkeypatch, tmp_path, capsys):
    from cortexlive import cli, stream

    monkeypatch.setattr(stream, "text_model_installed", lambda: False)
    monkeypatch.setattr(stream, "LiveEncoder", lambda *a, **k: pytest.fail("loaded the models"))
    video, tsv = tmp_path / "clip.mp4", tmp_path / "words.tsv"
    video.write_bytes(b"")
    tsv.write_text("words_per_tr\n['hi']\n")
    assert cli.main(["run", str(video), "--transcript", str(tsv)]) == 2
    assert "needs Llama 3.2 1B" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["view", "export"])
def test_cli_explains_a_capture_without_predictions(command, tmp_path, capsys):
    from cortexlive import cli

    summary = tmp_path / "summary-only"
    summary.mkdir()
    (summary / "capture.json").write_text("{}")
    assert cli.main([command, str(summary)]) == 2
    assert "keeps only its summary" in capsys.readouterr().err
    assert cli.main([command, str(tmp_path / "nothing")]) == 2
    assert "not a capture" in capsys.readouterr().err


def test_viewer_page_explains_the_first_seconds():
    page = html()
    assert "Brain response starts at" in page  # the volumes before the brain responds
    assert "d.stats" in page  # live colours start from the season-6 statistics


def _hub_error(status: int, reason: str):
    import httpx
    from huggingface_hub.errors import GatedRepoError

    response = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co"))
    return GatedRepoError(f"{status} Client Error. (Request ID: x)\n\n{reason}",
                          response=response, server_message=reason)


@pytest.mark.parametrize("gated", [True, False])
def test_setup_carries_on_without_llama(monkeypatch, capsys, gated):
    import argparse

    import huggingface_hub

    from cortexlive import cli

    fetched = []

    def fake_download(repo, **kw):
        if repo == cli.TEXT_BACKBONE:
            raise (_hub_error(403, "Your request is awaiting a review.") if gated
                   else OSError("Connection reset by peer"))
        fetched.append(repo)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    monkeypatch.setattr(cli, "cmd_doctor", lambda args: 0)
    code = cli.cmd_setup(argparse.Namespace(models=None, skip_backbones=False))
    out = capsys.readouterr().out
    assert code == 0 and fetched == list(cli.BACKBONES[:2]) and "optional" in out
    if gated:  # the Hub's own reason, and how to get access
        assert "awaiting a review" in out and "hf auth login" in out
    else:  # nothing to do with access: just try again
        assert "Connection reset" in out and "setup` again" in out


def test_setup_explains_a_file_that_is_not_a_bundle(tmp_path, capsys):
    from cortexlive import cli

    page = tmp_path / "Llama-cortexlive-v0.1.0.tar.gz"
    page.write_text("<html>sign in</html>")  # e.g. a web page saved under the bundle's name
    assert cli._install_bundle(str(page)) == 1
    assert "not a cortexlive model bundle" in capsys.readouterr().err
