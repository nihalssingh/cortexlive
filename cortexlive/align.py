"""Align stimulus features to fMRI samples.

This follows the convention of the official Algonauts 2025 challenge baseline:

    excluded_samples_start = 5    drop the first 5 fMRI samples of every run
    excluded_samples_end   = 5    drop the last 5
    hrf_delay              = 3    fMRI sample s is modelled by stimulus at s - 3 TRs
    stimulus_window        = 5    ...using a 5-TR window of stimulus, not one sample

Three choices, and why.

**Why trim 5 from each end.** Scanner transients at run start, and stimulus context
that doesn't exist before t=0. Trimming also conveniently absorbs a real quirk of this
dataset: transcript .tsv files are sometimes 1-2 rows SHORTER than the fMRI run they
describe. Both are aligned from the start (verified against word onsets), so the
mismatch is trailing, and dropping the last 5 samples makes it disappear. Without the
trim you would silently read past the end of the feature array.

**Why a window and not a single lag.** The BOLD response to an event is smeared over
roughly 4-8 seconds, so one TR of stimulus cannot explain one TR of BOLD. A 5-TR window
(7.5 s) is the baseline's answer and a reasonable default.

**One difference from the baseline.** The official code applies the `stimulus_window`
to visual and audio but gives language a *single* sample at `s - hrf_delay`, on the
grounds that language features already span several TRs. Here the window applies to
every modality: hemodynamic lag varies across cortex (faster in sensory areas, slower in
association cortex), and the output is a spatial map, so a single fixed lag for one
modality would bias the comparison between regions. Set `window=1` to reproduce the
baseline's behaviour.
"""

from __future__ import annotations

import numpy as np

EXCLUDED_START = 5
EXCLUDED_END = 5
HRF_DELAY = 3
STIMULUS_WINDOW = 5


def trim_fmri(arr: np.ndarray, start: int = EXCLUDED_START, end: int = EXCLUDED_END):
    """Drop leading/trailing fMRI samples. Input (n_samples, n_parcels)."""
    if arr.shape[0] <= start + end:
        raise ValueError(f"run too short to trim: {arr.shape[0]} samples")
    return arr[start : arr.shape[0] - end]


def to_tr_grid(features: np.ndarray, n_tr: int) -> np.ndarray:
    """Pool features sampled at any rate onto the TR grid: (n_src, ...) -> (n_tr, ...).

    Modalities are cached at different rates, and getting this wrong fails silently:

        text          TR rate, 0.671 Hz   (one vector per fMRI sample)
        audio, video  2 Hz                (~2.98 frames per TR)

    `align_features` indexes its input in TR units, so handing it a 2 Hz array makes
    every lookup off by a factor of three. Nothing raises: the model trains and scores
    badly, which looks like weak features. Convert first.

    Each output row is the mean of the source frames whose centres fall inside that TR,
    which is what the fMRI sample physically integrates over.
    """
    n_src = features.shape[0]
    if n_src == n_tr:
        return features
    if n_src == 0:
        raise ValueError("empty feature array")

    # Fractional bin edges: source frame i covers [i, i+1) of the source timeline, and
    # the two timelines span the same wall-clock duration.
    edges = np.linspace(0, n_src, n_tr + 1)
    out = np.empty((n_tr, *features.shape[1:]), dtype=np.float32)
    src = features.astype(np.float32, copy=False)
    for t in range(n_tr):
        lo, hi = int(np.floor(edges[t])), int(np.ceil(edges[t + 1]))
        hi = max(hi, lo + 1)
        out[t] = src[lo : min(hi, n_src)].mean(axis=0)
    return out


def align_features(
    features: np.ndarray,
    n_fmri_trimmed: int,
    window: int = STIMULUS_WINDOW,
    hrf_delay: int = HRF_DELAY,
    start: int = EXCLUDED_START,
) -> np.ndarray:
    """Build (n_fmri_trimmed, window * n_features) aligned to trimmed fMRI samples.

    `features` is indexed in the ORIGINAL (untrimmed) stimulus timeline, so index t
    means movie time [t * 1.49, (t+1) * 1.49). For trimmed fMRI sample s, the stimulus
    window ends at `s + start - hrf_delay` and spans `window` samples back from there.

    Indices below zero are clamped to the first available sample, and indices past the
    end of `features` are clamped to the last -- which is what makes the 1-2 row
    transcript shortfall harmless rather than an IndexError.
    """
    if features.ndim != 2:
        raise ValueError(f"expected (n_samples, n_features), got {features.shape}")
    n_stim, n_feat = features.shape

    ends = np.arange(n_fmri_trimmed) + start - hrf_delay
    # (n_fmri_trimmed, window) index matrix, oldest sample first
    idx = ends[:, None] - np.arange(window - 1, -1, -1)[None, :]
    idx = np.clip(idx, 0, n_stim - 1)

    return features[idx].reshape(n_fmri_trimmed, window * n_feat)
