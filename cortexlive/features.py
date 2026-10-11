"""Unified access to the cached backbone features.

Three modalities, three different shapes and two different sample rates; a mistake
here fails silently, so this module is the single place that knows:

    modality  model                        layers  dim   cached rate
    text      Llama-3.2-1B                 17      2048  TR (0.671 Hz)
    audio     w2v-bert-2.0                 25      1024  2 Hz
    video     vjepa2-vitl-fpc64-256        25      1024  2 Hz

Everything is returned on the TR grid, because that is what `align_features` indexes in
and what the fMRI actually is. Pooling happens here, once, rather than being remembered
at each call site.

Layer indices are per-modality, and every backbone predicts best short of full depth
(measured, all four subjects): text layer 8 of 16 (runs/layer_sweep_all), audio 15 of
24 and video 19 of 24 (runs/av_layer_sweep; sub-03 alone peaks at 18, a 0.0001 r
difference). These are the shipped head's layers.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

from .align import to_tr_grid
from .data import TR_SECONDS

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"


@dataclass(frozen=True)
class Modality:
    name: str
    subdir: str
    cached_hz: float | None  # None = already on the TR grid
    best_layer: int | None  # None = not yet measured

    @property
    def path(self) -> pathlib.Path:
        return CACHE / self.name / self.subdir


TEXT = Modality("text", "llama-3.2-1b", None, 8)
AUDIO = Modality("audio", "w2v-bert-2.0", 2.0, 15)
VIDEO = Modality("video", "vjepa2-vitl-fpc64-256", 2.0, 19)

MODALITIES = {m.name: m for m in (TEXT, AUDIO, VIDEO)}


def readable(path: pathlib.Path) -> np.ndarray | None:
    """Open a cached array, or None if it is not a complete .npy.

    A file still being written (a download in progress) is a normal transient state:
    numpy raises EOFError or ValueError on it, which means "not ready yet", not a crash.
    """
    try:
        return np.load(path, mmap_mode="r")
    except (EOFError, ValueError, OSError):
        return None


def available(modality: Modality) -> set[str]:
    """Stimuli present in this modality's cache."""
    if not modality.path.exists():
        return set()
    return {p.stem for p in modality.path.glob("*.npy")}


def common_stimuli(*modalities: Modality) -> set[str]:
    """Stimuli cached for every requested modality -- the usable intersection."""
    sets = [available(m) for m in modalities]
    return set.intersection(*sets) if sets else set()


def n_layers(modality: Modality) -> int | None:
    files = sorted(modality.path.glob("*.npy")) if modality.path.exists() else []
    for f in files:  # skip anything still being written
        arr = readable(f)
        if arr is not None:
            return int(arr.shape[1])
    return None


def load(modality: Modality, stimulus: str, layer: int, n_tr: int) -> np.ndarray | None:
    """Return (n_tr, dim) float32 for one stimulus and layer, pooled onto the TR grid.

    Returns None when the stimulus is not cached, so callers can skip rather than crash
    mid-sweep -- extraction runs are incremental and partial caches are normal.
    """
    arr = readable(modality.path / f"{stimulus}.npy")
    if arr is None:
        return None
    if layer >= arr.shape[1]:
        raise IndexError(f"{modality.name}: layer {layer} but only {arr.shape[1]} cached")
    feats = np.asarray(arr[:, layer, :], dtype=np.float32)
    if feats.shape[0] == 0:
        return None
    return to_tr_grid(feats, n_tr)


def canonical_n_tr(modality: Modality, n_src: int) -> int:
    """How many TRs this stimulus spans, from the feature count alone."""
    if modality.cached_hz is None:
        return n_src
    return max(1, round(n_src / modality.cached_hz / TR_SECONDS))


def load_canonical(modality: Modality, stimulus: str, layer: int) -> np.ndarray | None:
    """Like `load`, but on the STIMULUS's own TR grid rather than a subject's run length.

    Two subjects watching the same episode have scans that differ by a TR or two. Pooling
    to each subject's length would re-read and re-pool the same file once per subject for
    no benefit. Pooling to the stimulus's own grid instead lets one array serve every
    subject, and `align_features` clamps the leftover 1-2 sample difference -- exactly
    what it already does for text, whose grid comes from the transcript, not the scan.
    """
    arr = readable(modality.path / f"{stimulus}.npy")
    if arr is None:  # absent, or still downloading
        return None
    if layer >= arr.shape[1]:
        raise IndexError(f"{modality.name}: layer {layer} but only {arr.shape[1]} cached")
    feats = np.asarray(arr[:, layer, :], dtype=np.float32)
    if feats.shape[0] == 0:
        return None
    return to_tr_grid(feats, canonical_n_tr(modality, feats.shape[0]))


def summary() -> str:
    """Human-readable cache state: what has been extracted, per modality."""
    lines = [f"{'modality':<8}{'model':<26}{'files':>7}{'layers':>8}{'dim':>7}{'GB':>7}{'bad':>5}"]
    for m in MODALITIES.values():
        files = sorted(m.path.glob("*.npy")) if m.path.exists() else []
        head, bad = None, 0
        for f in files:
            arr = readable(f)
            if arr is None:
                bad += 1
            elif head is None:
                head = arr
        if head is None:
            lines.append(
                f"{m.name:<8}{m.subdir:<26}{len(files):>7}{'-':>8}{'-':>7}{'-':>7}{bad:>5}"
            )
            continue
        gb = sum(f.stat().st_size for f in files) / 1e9
        lines.append(
            f"{m.name:<8}{m.subdir:<26}{len(files):>7}{head.shape[1]:>8}"
            f"{head.shape[2]:>7}{gb:>7.1f}{bad:>5}"
        )
    return "\n".join(lines)
