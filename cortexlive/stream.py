"""Live inference: rolling feature extractors feeding the trained head.

The configuration is the measured one: layers text 8 / audio 15 / video 19, video at
192px (V-JEPA2 loaded with `crop_size=192` so its rotary position grid matches the
input; see modal_extract.py), and a per-subject ridge over 5-TR windows. The weights load
from `models/final_head.npz`, the layers and window from `runs/final_head/meta.json`
(both written by scripts/17_fit_final_head.py); 192 is VIDEO_CROP below.

No output delay is added: BOLD sample `s` is predicted from the stimulus window ending
at TR `s - hrf_delay` (align_features), whose input ends 2 TRs (~3 s) before time
s * TR. What the live path adds is extractor latency:

    text    none: Llama is a causal decoder, so extending its KV cache reproduces the
            full-episode features exactly
    video   one 64-frame clip (~2.1 s): clips are independent in the cache too, so
            clip-at-a-time matches it exactly
    audio   w2v-bert runs on independent 30 s chunks, so a finalized chunk arrives up
            to 30 s late. The live path re-forwards the trailing partial chunk each
            tick; those features are provisional until the chunk finalizes, because
            w2v-bert attends across a whole chunk (21_live_analysis.py's capture mode
            measures the difference: runs/live_analysis/capture_friends_s06e01a.json).

Pooling: the cache point-samples native-rate features onto a 2 Hz grid anchored to the
stream's total length, then mean-bins them to TRs. TrPooler replicates both steps
exactly when the length is known (`set_total`). Without it (open-ended input) the grid
is rebuilt from the current buffer length: rows read at the end of a 240 s feed then
score like the offline pipeline (r 0.280 vs 0.279,
runs/stream_gate/extract_parity_openended.json); rows emitted tick by tick in that mode
have no parity record.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .align import HRF_DELAY, STIMULUS_WINDOW
from .data import TR_SECONDS

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEAD_PATH = ROOT / "models" / "final_head.npz"
META_PATH = ROOT / "runs" / "final_head" / "meta.json"
LOCK_PATH = pathlib.Path.home() / ".cache" / "cortexlive" / "model.lock"

VIDEO_MODEL = "facebook/vjepa2-vitl-fpc64-256"
AUDIO_MODEL = "facebook/w2v-bert-2.0"
TEXT_MODEL = "meta-llama/Llama-3.2-1B"
VIDEO_FPS = 30
FRAMES_PER_CLIP = 64
AUDIO_SR = 16_000
AUDIO_CHUNK_S = 30.0
VIDEO_CROP = 192


# Llama 3.2 is gated: Meta approves each Hugging Face account by hand. Only transcripts
# use it, so everything else runs without it.
TEXT_ACCESS = ("accept the license at https://huggingface.co/meta-llama/Llama-3.2-1B, "
               "run `hf auth login`, then `cortexlive setup`")


def text_model_installed(cache_dir: str | None = None) -> bool:
    """Whether Llama 3.2 1B is in the Hugging Face cache (setup fetches it given access).

    Every file the load needs, since an interrupted download leaves the small ones behind.
    """
    from huggingface_hub import try_to_load_from_cache

    files = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    return all(isinstance(try_to_load_from_cache(TEXT_MODEL, f, cache_dir=cache_dir), str)
               for f in files)


def torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class ModelBusy(RuntimeError):
    """Another process already has the models loaded."""


_model_lock = None  # this process's hold on LOCK_PATH, kept until it exits


def acquire_model_lock(path: pathlib.Path = LOCK_PATH):
    """Allow one process with the models loaded per user.

    A process holding the three backbones uses about 9 GB of memory (measured on an M5
    with 24 GB), so a second one, such as `cortexlive run` beside an open viewer, would
    push the machine into heavy swapping. The lock lasts until the process exits.
    """
    global _model_lock
    if _model_lock is not None:
        return _model_lock
    import fcntl
    import os
    import sys

    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        holder = f.read().strip() or "another process"
        f.close()
        raise ModelBusy(f"the models are already loaded by {holder}: quit it first "
                        "(each one needs about 9 GB of memory)") from None
    f.seek(0)
    f.truncate()
    words = [pathlib.Path(sys.argv[0]).name] + [a for a in sys.argv[1:2] if not a.startswith("-")]
    f.write(f"{' '.join(words)} (pid {os.getpid()})")
    f.flush()
    _model_lock = f
    return f


def find_blocks(model):
    """Locate the transformer block list, whatever the architecture calls it."""
    from torch import nn

    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 8:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise RuntimeError("no transformer block list found")
    name, blocks = best
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1], blocks


def truncate(model, keep: int) -> None:
    """Physically drop blocks above `keep`: hidden_states[keep] is the last one left."""
    from torch import nn

    parent, attr, blocks = find_blocks(model)
    setattr(parent, attr, nn.ModuleList(list(blocks)[:keep]))


# --------------------------------------------------------------------------- head


class HeadArtifact:
    """The shipping model: per-modality scaler+PCA, per-subject ridge. Pure numpy."""

    def __init__(self, path: pathlib.Path = HEAD_PATH, meta_path: pathlib.Path = META_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"the trained head is not installed ({path}): "
                "cortexlive setup --models <bundle>"
            )
        self.arrays = dict(np.load(path))
        self.meta = json.loads(meta_path.read_text())
        self.modalities: list[str] = self.meta["modalities"]
        self.layers: dict[str, int] = self.meta["layers"]
        self.window: int = self.meta["window"]
        self.subjects = sorted(k[: -len("_coef")] for k in self.arrays if k.endswith("_coef"))

    def project(self, name: str, feats: np.ndarray) -> np.ndarray:
        """(n, d_raw) float -> (n, n_components), the exact train-time transform."""
        a = self.arrays
        z = (feats.astype(np.float32) - a[f"{name}_scaler_mean"]) / a[f"{name}_scaler_scale"]
        return (z - a[f"{name}_pca_mean"]) @ a[f"{name}_pca_components"].T

    def predict(self, subject: str, window_rows: np.ndarray) -> np.ndarray:
        """(window, n_components * n_modalities) oldest-first -> (n_parcels,)."""
        x = window_rows.reshape(-1)
        return x @ self.arrays[f"{subject}_coef"].T + self.arrays[f"{subject}_intercept"]

    def predict_rows(self, subject: str, x: np.ndarray) -> np.ndarray:
        """(n, window * n_components * n_modalities) flattened windows -> (n, n_parcels)."""
        return x @ self.arrays[f"{subject}_coef"].T + self.arrays[f"{subject}_intercept"]


# ---------------------------------------------------------------- feature streams


class TrPooler:
    """Native-rate feature frames -> TR rows, via the CACHE's exact pooling path.

    The head was trained on cached features pooled in two steps: linear-interpolation
    resample to 2 Hz (extraction's `resample_to`), then mean-binning onto the TR grid
    (`to_tr_grid`). Reproducing both steps matters: binning native-rate frames straight
    into TRs is a different filter, and the head notices.

    Cost. Anchored (known-length input), the resample is incremental: a 2 Hz row is a
    point sample between the two source frames around it, so only rows touching frames
    that changed since the last call -- new frames, or audio's re-forwarded provisional
    tail -- are recomputed, and per-tick cost stays flat however long the stream runs.
    Unanchored grids rescale with the buffer, so every
    row moves and the whole buffer is still resampled as it grows: open-ended sessions
    keep a cost that rises linearly with their length.
    """

    TARGET_HZ = 2.0

    def __init__(self, dim: int):
        self.dim = dim
        self._buf = np.empty((0, dim), dtype=np.float32)  # frames, contiguous in time
        self._n = 0  # frames in use; the buffer grows geometrically
        self._starts: list[int] = []  # frame offset of each extend()ed block
        self.rate: float | None = None  # frames per second, set by first block
        self._dirty = 0  # first frame index changed since the last resample
        self._grid: np.ndarray | None = None
        self._rows = 0  # valid rows in _grid
        # Exact-grid anchor for known-length streams. The cache's 2 Hz rows are POINT
        # SAMPLES (np.interp) on a grid tied to the stream's TOTAL length, and w2v-bert
        # features vary fast enough that sampling them on any other grid decorrelates
        # the rows. Anchored to the total length, the live rows reproduce the cache
        # (mean row correlation >= 0.9999 for all three streams, lowest single row 0.998
        # for video, runs/stream_gate/extract_parity.json). Open-ended input cannot
        # anchor; see the module docstring
        # for what that costs and what is still unmeasured.
        self.n_src_final: int | None = None
        self.n_out_final: int | None = None

    def anchor(self, n_src_final: int, n_out_final: int) -> None:
        self.n_src_final = max(2, n_src_final)
        self.n_out_final = max(1, n_out_final)
        self._grid, self._rows, self._dirty = None, 0, 0  # grids change meaning

    def extend(self, block: np.ndarray, rate: float) -> None:
        self.rate = rate
        block = np.atleast_2d(block).astype(np.float32)
        need = self._n + len(block)
        if need > len(self._buf):
            grown = np.empty((max(need, 2 * len(self._buf), 256), self.dim), np.float32)
            grown[: self._n] = self._buf[: self._n]
            self._buf = grown
        self._buf[self._n : need] = block
        self._starts.append(self._n)
        self._dirty = min(self._dirty, self._n)
        self._n = need

    def replace_tail(self, n_keep_blocks: int) -> None:
        """Drop provisional trailing blocks (audio partial-chunk recompute)."""
        if n_keep_blocks < len(self._starts):
            cut = self._starts[n_keep_blocks]
            del self._starts[n_keep_blocks:]
            self._dirty = min(self._dirty, cut)
            self._n = cut

    @property
    def n_blocks(self) -> int:
        return len(self._starts)

    def _resampled(self) -> np.ndarray | None:
        """The buffer so far on the 2 Hz grid -- extraction's `resample_to`, replicated.

        Anchored: source frame i sits at fraction i/(n_src_final-1), output row j at
        j/(n_out_final-1) -- the exact geometry the cache used over the whole stream.
        Rows are only emitted once their position is inside the received frames, and a
        row's value only changes if one of its two source frames does. Unanchored: both
        counts come from the current buffer, which is all an open-ended stream can do.
        """
        n_have = self._n
        if self.rate is None or n_have == 0:
            return None
        if self._grid is not None and self._dirty >= n_have:
            return self._grid[: self._rows]  # nothing changed since the last call
        src = self._buf[:n_have]
        if self.n_src_final is not None:
            n_src, n_out = self.n_src_final, self.n_out_final
            denom = max(1, n_out - 1)
            n_rows = min(n_out, int(np.floor((n_have - 1) / (n_src - 1) * denom)) + 1)
            if self._grid is None:
                self._grid = np.empty((n_out, self.dim), dtype=np.float32)
                self._rows = 0
            # Row j samples between frames floor(p) and floor(p) + 1, p = j/denom*(n_src-1),
            # so rows with p < dirty - 1 saw no changed frame and keep their values.
            keep = min(self._rows, max(0, int((self._dirty - 1) * denom / (n_src - 1))))
            # np.interp on a slice starting one frame before the first recomputed row's
            # bracket returns bit-identical values to the full-buffer call.
            i0 = max(0, int(keep / denom * (n_src - 1)) - 1)
            xs = np.arange(i0, n_have) / (n_src - 1)
            xd = np.arange(keep, n_rows) / denom
            for j in range(self.dim):
                self._grid[keep:n_rows, j] = np.interp(xd, xs, src[i0:, j])
            self._rows = n_rows
        else:
            n_out = max(1, int(round(n_have / self.rate * self.TARGET_HZ)))
            xs, xd = np.linspace(0, 1, n_have), np.linspace(0, 1, n_out)
            self._grid = np.empty((n_out, self.dim), dtype=np.float32)
            for j in range(self.dim):
                self._grid[:, j] = np.interp(xd, xs, src[:, j])
            self._rows = n_out
        self._dirty = n_have
        return self._grid[: self._rows]

    def tr_row(self, t: int) -> np.ndarray | None:
        """Row t of `to_tr_grid` over the (final) grid, clamped at the edges."""
        g = self._resampled()
        if g is None:
            return None
        n_full = self.n_out_final if self.n_out_final is not None else len(g)
        n_tr = max(1, round(n_full / self.TARGET_HZ / TR_SECONDS))
        t = min(t, n_tr - 1)  # clamp, mirroring align_features at array edges
        edges = np.linspace(0, n_full, n_tr + 1)
        lo, hi = int(np.floor(edges[t])), int(np.ceil(edges[t + 1]))
        hi = max(hi, lo + 1)
        if lo >= len(g):  # not yet received; clamp to the newest available row
            return g[-1]
        return g[lo : min(hi, len(g))].mean(axis=0)

    def seconds_covered(self) -> float:
        if self.rate is None:
            return 0.0
        return self._n / self.rate


class VideoStream:
    """Feed 192x192 RGB frames at 30 fps; V-JEPA2 layer-19 features come out.

    Exact parity with the cache: independent 64-frame clips, fp16, spatial mean over
    the 12x12 patch grid, model loaded with crop_size matching the input resolution.
    V-JEPA2's predictor is skipped: only encoder states are read, and they do not depend on it.
    """

    def __init__(self, layer: int, device: str | None = None, crop: int = VIDEO_CROP):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.layer = layer
        self.crop = crop
        self.device = device or torch_device()
        m = AutoModel.from_pretrained(VIDEO_MODEL, dtype=torch.float16, crop_size=crop)
        truncate(m, layer)
        self.model = m.to(self.device).eval()
        self.grid = (crop // m.config.patch_size) ** 2
        self.out_rate = VIDEO_FPS / m.config.tubelet_size  # 15 Hz tubelet steps
        self.buf: list[np.ndarray] = []
        self.pool = TrPooler(m.config.hidden_size)

    def set_total(self, n_frames: int) -> None:
        """Anchor the pooling grid for a known-length stream (episode playback)."""
        tub = 2  # tubelet_size: 32 feature steps per 64-frame clip
        n_src = (n_frames // FRAMES_PER_CLIP) * (FRAMES_PER_CLIP // tub)
        n_out = max(1, int(round(n_src / (VIDEO_FPS / tub) * TrPooler.TARGET_HZ)))
        self.pool.anchor(n_src, n_out)

    def push(self, frame: np.ndarray) -> None:
        """One (crop, crop, 3) uint8 frame. Forwards when a clip completes."""
        self.buf.append(frame)
        if len(self.buf) == FRAMES_PER_CLIP:
            self._forward(np.stack(self.buf))
            self.buf = []

    def _forward(self, clip: np.ndarray) -> None:
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(clip).to(self.device).permute(0, 3, 1, 2)
            x = x.half().div_(255.0).unsqueeze(0)
            out = self.model(pixel_values_videos=x, output_hidden_states=True,
                             skip_predictor=True)
            h = out.hidden_states[self.layer]
            feats = (
                h.view(1, h.shape[1] // self.grid, self.grid, h.shape[-1])
                .mean(2)
                .squeeze(0)
                .float()
                .cpu()
                .numpy()
            )
        self.pool.extend(feats, self.out_rate)


class AudioStream:
    """Feed 16 kHz mono float32 samples; w2v-bert layer-15 features come out.

    Finalized non-overlapping 30 s chunks match the cache exactly. The trailing
    partial chunk is re-forwarded on demand so fresh TRs exist at all; its features
    are provisional until the chunk completes and is re-run in final form.
    """

    def __init__(self, layer: int, device: str | None = None):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self.torch = torch
        self.layer = layer
        self.device = device or torch_device()
        m = AutoModel.from_pretrained(AUDIO_MODEL, dtype=torch.float16)
        truncate(m, layer)
        self.model = m.to(self.device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
        # Only samples after the last finalized chunk are kept: finalized audio is never
        # read again, so a push costs the same however long the stream has run.
        self.wav = np.zeros(0, dtype=np.float32)
        self.chunk = int(AUDIO_CHUNK_S * AUDIO_SR)
        self.finalized = 0  # chunks forwarded in final form
        self.pool = TrPooler(m.config.hidden_size)
        self._had_partial = False

    def set_total(self, n_samples: int) -> None:
        """Anchor the pooling grid for a known-length stream (episode playback).

        Feature counts per chunk depend only on chunk sample-length, so the final
        total is computable up front by probing the feature extractor on silence.
        n_out replicates extraction: round(len(wav)/sr * 2) -- from WAV length, even
        though the final sub-0.1 s tail chunk (if any) is dropped from features.
        """

        def count(n: int) -> int:
            if n < AUDIO_SR // 10:
                return 0
            return self.fe(np.zeros(n, np.float32), sampling_rate=AUDIO_SR,
                           return_tensors="np")["input_features"].shape[1]

        full, part = divmod(n_samples, self.chunk)
        n_src = full * count(self.chunk) + count(part)
        n_out = max(1, int(round(n_samples / AUDIO_SR * TrPooler.TARGET_HZ)))
        self.pool.anchor(n_src, n_out)

    def push(self, samples: np.ndarray) -> None:
        self.wav = np.concatenate([self.wav, samples.astype(np.float32)])
        while len(self.wav) >= self.chunk:
            if self._had_partial:  # replace the provisional tail with the final chunk
                self.pool.replace_tail(self.finalized)
                self._had_partial = False
            self._forward(self.wav[: self.chunk])
            self.finalized += 1
            self.wav = self.wav[self.chunk :]

    def flush_partial(self) -> None:
        """Forward the trailing partial chunk so the newest TRs have features."""
        seg = self.wav
        if len(seg) < AUDIO_SR // 10:
            return
        if self._had_partial:
            self.pool.replace_tail(self.finalized)
        self._forward(seg)
        self._had_partial = True

    def _forward(self, seg: np.ndarray) -> None:
        torch = self.torch
        with torch.no_grad():
            inp = self.fe(seg, sampling_rate=AUDIO_SR, return_tensors="pt")
            out = self.model(
                input_features=inp["input_features"].to(self.device).half(),
                output_hidden_states=True,
            )
            feats = out.hidden_states[self.layer].squeeze(0).float().cpu().numpy()
        # w2v-bert emits a fixed frame rate; derive it from this segment's own length
        # so partial chunks pool at the same rate as full ones.
        rate = len(feats) / (len(seg) / AUDIO_SR)
        self.pool.extend(feats, rate)


class TextStream:
    """Feed per-TR word lists; Llama layer-8 features come out, KV-cache incremental.

    Llama is causal, so extending the context reproduces the cached full-episode
    forward exactly: cached extraction ran the whole transcript in one pass, and the
    hidden state of token t never depends on tokens after t. Empty TRs are zero rows,
    matching 03_extract_text.py.

    Llama loads only when a run has words (`load`). Without a transcript every row is
    zero, so an input with no transcript never needs it.
    """

    def __init__(self, layer: int, hidden: int, device: str | None = None):
        self.layer = layer
        self.hidden = hidden  # the head's text input width
        self.device = device
        self.model = self.tokenizer = None
        self.past = None
        self.rows: list[np.ndarray] = []  # one (hidden,) per TR

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = self.device or torch_device()
        m = AutoModel.from_pretrained(TEXT_MODEL, dtype=torch.float16)
        if m.config.hidden_size != self.hidden:
            raise ValueError(f"{TEXT_MODEL} has width {m.config.hidden_size}; "
                             f"the head expects {self.hidden}")
        # Keep one block PAST the readout layer: Llama applies its final RMSNorm to the
        # LAST hidden_states entry, so a model truncated exactly at `layer` returns a
        # normalized state where the cache (extracted from the full model) stored the
        # raw interior one, a mismatch that lowers accuracy. w2v-bert does not post-norm
        # the appended state, nor does V-JEPA2 from transformers 5.4 on (pyproject.toml
        # requires 5.5), so their streams truncate exactly.
        truncate(m, self.layer + 1)
        self.model = m.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

    def push_tr(self, words: list[str]) -> None:
        if not words:
            self.rows.append(np.zeros(self.hidden, dtype=np.float32))
            return
        self.load()
        torch = self.torch
        text = " " + " ".join(str(w) for w in words)  # leading space matters for BPE
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            self.rows.append(np.zeros(self.hidden, dtype=np.float32))
            return
        with torch.no_grad():
            out = self.model(
                input_ids=torch.tensor([ids], device=self.device),
                past_key_values=self.past,
                use_cache=True,
                output_hidden_states=True,
            )
        self.past = out.past_key_values
        h = out.hidden_states[self.layer].squeeze(0).float().cpu().numpy()
        self.rows.append(h.mean(axis=0))

    def tr_row(self, t: int) -> np.ndarray | None:
        if t < len(self.rows):
            return self.rows[t]
        if self.rows:
            return self.rows[-1]  # clamp, mirroring align_features
        return None


# ------------------------------------------------------------------- orchestrator


class LiveEncoder:
    """Owns the three streams and the head; emits per-subject parcel predictions.

    Prediction for BOLD sample `s` (untrimmed timeline) uses the stimulus window
    [s - hrf - w + 1 .. s - hrf] in TR units, clamped at 0 -- byte-for-byte the
    align_features convention the head was trained under.
    """

    def __init__(self, head: HeadArtifact | None = None, device: str | None = None):
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()  # weight-loading bars would bury the run's log
        self.head = head or HeadArtifact()
        acquire_model_lock()
        self.device = device or torch_device()
        layers = self.head.layers
        self.video = VideoStream(layers["video"], self.device)
        self.audio = AudioStream(layers["audio"], self.device)
        self.text = TextStream(layers["text"], len(self.head.arrays["text_scaler_mean"]),
                               self.device)  # Llama itself loads only for a transcript

    def set_total(self, n_video_frames: int, n_audio_samples: int) -> None:
        """Anchor both pooling grids when the input's length is known up front."""
        self.video.set_total(n_video_frames)
        self.audio.set_total(n_audio_samples)

    def reset(self) -> None:
        """Forget all input so a new stream can start, without reloading the models."""
        self.video.buf = []
        self.video.pool = TrPooler(self.video.pool.dim)
        self.audio.wav = np.zeros(0, dtype=np.float32)
        self.audio.finalized = 0
        self.audio._had_partial = False
        self.audio.pool = TrPooler(self.audio.pool.dim)
        self.text.past = None
        self.text.rows = []

    def release_memory(self) -> None:
        """Hand the accelerator's cached blocks back to the system; the models stay loaded."""
        import gc

        import torch

        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def _tr_row(self, name: str, t: int) -> np.ndarray | None:
        if name == "text":
            return self.text.tr_row(t)
        return (self.video if name == "video" else self.audio).pool.tr_row(t)

    def ready_through(self) -> int:
        """Largest BOLD sample s whose whole stimulus window has features."""
        secs = min(self.video.pool.seconds_covered(), self.audio.pool.seconds_covered())
        tr_have = int(secs / TR_SECONDS)  # TRs fully covered by video+audio
        tr_have = min(tr_have, len(self.text.rows))
        return tr_have - 1 + HRF_DELAY  # window ends at s - HRF_DELAY

    def predict(self, s: int) -> dict[str, np.ndarray] | None:
        """Per-subject (n_parcels,) predictions for untrimmed BOLD sample s."""
        end = s - HRF_DELAY
        ts = [max(0, end - k) for k in range(STIMULUS_WINDOW - 1, -1, -1)]
        rows = []
        for t in ts:
            parts = []
            for name in self.head.modalities:
                raw = self._tr_row(name, t)
                if raw is None:
                    return None
                parts.append(self.head.project(name, raw[None, :])[0])
            rows.append(np.concatenate(parts))
        window = np.stack(rows)  # (window, n_mod * n_components), oldest first
        return {s_: self.head.predict(s_, window) for s_ in self.head.subjects}
