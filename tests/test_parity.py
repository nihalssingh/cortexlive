"""Live pooling against the offline path that built the training cache (no models).

The offline side is the extraction code itself: `resample_to` read from
scripts/modal_extract.py, then `to_tr_grid` and `align_features` as training used them.
The live side is the real stream code with only the backbone forward replaced by fixed
synthetic features. The cache also stores float16, which these tests leave out.
"""

from __future__ import annotations

import ast
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cortexlive import stream
from cortexlive.align import align_features, to_tr_grid
from cortexlive.features import AUDIO, VIDEO, canonical_n_tr
from cortexlive.stream import (
    AudioStream,
    HeadArtifact,
    LiveEncoder,
    TextStream,
    TrPooler,
    VideoStream,
)

CLIP, SR = stream.FRAMES_PER_CLIP, stream.AUDIO_SR
CHUNK = int(stream.AUDIO_CHUNK_S * SR)


def _extraction_resample():
    """`resample_to` from scripts/modal_extract.py, the 2 Hz step of the cache. It is nested
    in a Modal function, so it is read from the source rather than imported."""
    path = stream.ROOT / "scripts" / "modal_extract.py"
    fn = next(n for n in ast.walk(ast.parse(path.read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == "resample_to")
    ns = {"np": np}
    exec(compile(ast.Module([fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["resample_to"]


resample_to = _extraction_resample()


def assert_pools_like_the_cache(pool: TrPooler, seq: np.ndarray, n_out: int, modality) -> None:
    """Every live TR row, past the end too, against features.load_canonical's grid."""
    g = resample_to(seq, n_out)
    cached = to_tr_grid(g, canonical_n_tr(modality, len(g)))
    t = np.arange(len(cached) + 2)
    live = np.stack([pool.tr_row(i) for i in t])
    # linspace and TrPooler's i / (n - 1) can round the last float32 bit differently
    np.testing.assert_allclose(live, cached[np.minimum(t, len(cached) - 1)], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n_frames", [64, 703, 2049, 4517])  # one clip, then partial clips
def test_video_stream_pools_like_the_cache(n_frames):
    rng = np.random.default_rng(n_frames)
    clips = rng.standard_normal((n_frames // CLIP, CLIP // 2, 3)).astype(np.float32)
    video = VideoStream.__new__(VideoStream)  # no backbone: only the forward is replaced
    video.buf, video.pool, video.out_rate = [], TrPooler(3), stream.VIDEO_FPS / 2

    def forward(clip):  # frames carry their index, so clip boundaries are checked too
        c = int(clip[0, 0]) // CLIP
        assert clip[:, 0].tolist() == list(range(c * CLIP, (c + 1) * CLIP))
        video.pool.extend(clips[c], video.out_rate)

    video._forward = forward
    video.set_total(n_frames)
    for i in range(n_frames):
        video.push(np.array([i]))
    seq = clips.reshape(-1, 3)  # extraction drops the trailing partial clip
    # the 2 Hz grid length, computed as do_video in modal_extract.py does
    n_out = max(1, int(round(len(seq) / (stream.VIDEO_FPS / 2) * 2.0)))
    assert_pools_like_the_cache(video.pool, seq, n_out, VIDEO)


def _frames(n: int) -> int:
    """Feature frames for n samples. Any fixed rule will do: set_total asks the extractor."""
    return 0 if n < SR // 10 else (1 + (n - 400) // 160) // 2


@pytest.mark.parametrize("seconds", [5.0, 30.0, 30.05, 61.3, 95.05])  # .05: a tail too short
def test_audio_stream_pools_like_the_cache(seconds):
    rng = np.random.default_rng(int(seconds * 100))
    n_samples = int(round(seconds * SR))
    chunks = rng.standard_normal((-(-n_samples // CHUNK), _frames(CHUNK), 3)).astype(np.float32)

    def model(c, n):  # every frame depends on the whole segment, so a partial is provisional
        return chunks[c][: _frames(n)] + np.float32(np.sin(n))

    audio = AudioStream.__new__(AudioStream)  # no backbone: only the forward is replaced
    audio.wav, audio.chunk, audio.finalized = np.zeros(0, np.float32), CHUNK, 0
    audio.pool, audio._had_partial = TrPooler(3), False
    audio.fe = lambda x, **kw: {"input_features": np.empty((1, _frames(len(x)), 0))}

    def forward(seg):  # AudioStream._forward from the model's output on
        feats = model(audio.finalized, len(seg))
        audio.pool.extend(feats, len(feats) / (len(seg) / SR))

    audio._forward = forward
    audio.set_total(n_samples)
    fed = 0
    while fed < n_samples:  # uneven pushes, with the provisional tails a 1x run makes
        n = min(int(rng.uniform(0.2, 9.0) * SR), n_samples - fed)
        audio.push(np.zeros(n, np.float32))
        fed += n
        if rng.random() < 0.5:
            audio.flush_partial()
    audio.flush_partial()  # run.py flushes at the end in both paces
    # Extraction: 30 s chunks, a final one under 0.1 s dropped (do_audio in modal_extract.py)
    seq = np.concatenate([model(c, min(CHUNK, n_samples - s))
                          for c, s in enumerate(range(0, n_samples, CHUNK))
                          if n_samples - s >= SR // 10])
    n_out = max(1, int(round(n_samples / SR * 2.0)))  # as do_audio in modal_extract.py
    assert_pools_like_the_cache(audio.pool, seq, n_out, AUDIO)


class _Finished:
    """A stream that has received everything; clamps past its end like TrPooler.tr_row."""

    def __init__(self, rows):
        self.rows = rows

    def tr_row(self, t):
        return self.rows[min(t, len(self.rows) - 1)]


def test_live_window_matches_align_features(tmp_path):
    rng = np.random.default_rng(0)
    dims, k = {"text": 5, "audio": 4, "video": 3}, 2
    arrays = {}
    for m, d in dims.items():
        arrays |= {f"{m}_scaler_mean": rng.standard_normal(d),
                   f"{m}_scaler_scale": rng.uniform(0.5, 2.0, d),
                   f"{m}_pca_mean": rng.standard_normal(d),
                   f"{m}_pca_components": rng.standard_normal((k, d))}
    for s in ("sub-01", "sub-02"):
        arrays |= {f"{s}_coef": rng.standard_normal((7, 5 * k * len(dims))),
                   f"{s}_intercept": rng.standard_normal(7)}
    np.savez(tmp_path / "head.npz", **{a: v.astype(np.float32) for a, v in arrays.items()})
    (tmp_path / "meta.json").write_text(json.dumps(
        {"modalities": list(dims), "layers": {"text": 8, "audio": 15, "video": 19}, "window": 5}))
    head = HeadArtifact(tmp_path / "head.npz", tmp_path / "meta.json")

    # Grids a TR apart, like real ones: the transcript and the video run short.
    raw = {m: rng.standard_normal((n, dims[m])).astype(np.float32)
           for m, n in (("text", 60), ("audio", 61), ("video", 60))}
    enc = LiveEncoder.__new__(LiveEncoder)  # no backbones
    enc.head, enc.text = head, TextStream(layer=8, hidden=dims["text"])
    enc.text.rows = list(raw["text"])
    enc.audio = SimpleNamespace(pool=_Finished(raw["audio"]))
    enc.video = SimpleNamespace(pool=_Finished(raw["video"]))

    # Training: cut to the shortest grid (12_trimodal.py), project, concatenate, window.
    n = min(len(r) for r in raw.values())
    z = np.concatenate([head.project(m, raw[m][:n]) for m in head.modalities], axis=1)
    n_bold = 61
    x = align_features(z, n_bold, start=0)  # row s: the window for BOLD sample s
    for s in range(n_bold):  # the start-trim volumes too, which the viewer computes
        got = enc.predict(s)
        for sub in head.subjects:
            want = x[s] @ head.arrays[f"{sub}_coef"].T + head.arrays[f"{sub}_intercept"]
            np.testing.assert_allclose(got[sub], want, rtol=1e-5, atol=1e-5)
