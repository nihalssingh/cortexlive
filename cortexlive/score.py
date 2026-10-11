"""Scoring: per-parcel correlation, per-network breakdown, and the noise ceiling.

Why a ceiling: raw per-parcel r is dominated by how reliable each parcel's signal is,
not by how good the model is. Raw r per network mostly maps signal-to-noise, and would
rank networks the same way for any model. The ceiling converts r into the fraction of
the predictable signal a model captures.

Theory for the ceiling. Write an observation as y = s + e, signal plus independent
noise. Two viewings of the same stimulus give

    corr(y1, y2) = var(s) / (var(s) + var(e)) = rho          [test-retest reliability]

A model that predicted s perfectly, scored against a single noisy observation, would
only reach

    corr(s, y)  = sqrt(rho)

so `sqrt(rho)` -- not `rho` -- is the ceiling for a model evaluated against single-run
data, which is what Friends is. Normalized score is r_model / sqrt(rho).

Assumption: rho is measured on Movie10 (Life and Hidden Figures, the only twice-shown
stimuli) and applied to Friends, which assumes reliability transfers across stimulus
sets. The dataset allows no other estimate.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np

from .data import BoldStore

ATLAS_PATH = pathlib.Path(__file__).resolve().parent.parent / "atlas" / "schaefer1000.npz"


def pearson_per_parcel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation between two (n_samples, n_parcels) arrays."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b).sum(0) / denom
    return np.nan_to_num(r, nan=0.0)


def _fisher_z(r: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return np.arctanh(np.clip(r, -1 + eps, 1 - eps))


def _bh_fdr(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg. Returns a boolean mask of parcels passing at level alpha."""
    n = p.size
    order = np.argsort(p)
    thresh = alpha * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    mask = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.max(np.nonzero(passed)[0])
        mask[order[: cutoff + 1]] = True
    return mask


@dataclass
class NoiseCeiling:
    """Per-parcel reliability and the resulting ceiling, for one subject."""

    subject: str
    reliability: np.ndarray  # (1000,) test-retest rho, Fisher-averaged over pairs
    ceiling: np.ndarray  # (1000,) sqrt(max(rho, 0)) -- the normalizer
    mask: np.ndarray  # (1000,) bool, parcels with reliability significantly > 0
    per_pair: np.ndarray  # (n_pairs, 1000) raw per-pair correlations
    pair_names: list[str] = field(default_factory=list)

    @property
    def n_reliable(self) -> int:
        return int(self.mask.sum())

    def normalize(self, r_model: np.ndarray, floor: float = 0.1) -> np.ndarray:
        """r_model / ceiling, with unreliable parcels set to nan rather than divided.

        Dividing by a near-zero ceiling manufactures enormous normalized scores out of
        noise, so parcels below `floor` are excluded instead.
        """
        out = np.full_like(r_model, np.nan, dtype=np.float64)
        ok = self.mask & (self.ceiling > floor)
        out[ok] = r_model[ok] / self.ceiling[ok]
        return out


def noise_ceiling(store: BoldStore, alpha: float = 0.05) -> NoiseCeiling:
    """Estimate per-parcel reliability from the twice-presented Movie10 stimuli."""
    pairs = store.repeated_pairs()
    if not pairs:
        raise ValueError(f"{store.subject}: no repeated stimuli found")

    per_pair, names = [], []
    for first, second in pairs:
        a, b = store.load(first), store.load(second)
        # The two viewings can differ by a TR or two; compare the common prefix.
        n = min(a.shape[0], b.shape[0])
        per_pair.append(pearson_per_parcel(a[:n], b[:n]))
        names.append(first.stimulus)

    per_pair_arr = np.stack(per_pair)  # (n_pairs, 1000)

    # Average across pairs in Fisher-z space, and test against zero at the level of
    # pairs rather than TRs. TRs within a run are heavily autocorrelated, so treating
    # them as independent samples would produce absurdly small p-values.
    z = _fisher_z(per_pair_arr)
    z_mean = z.mean(0)
    n_pairs = z.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = z_mean / (z.std(0, ddof=1) / np.sqrt(n_pairs))
    t = np.nan_to_num(t, nan=0.0)

    from scipy import stats  # local import; scipy is only needed here

    p = stats.t.sf(np.abs(t), df=n_pairs - 1) * 2
    mask = _bh_fdr(p, alpha=alpha)

    reliability = np.tanh(z_mean)
    ceiling = np.sqrt(np.clip(reliability, 0.0, None))

    return NoiseCeiling(
        subject=store.subject,
        reliability=reliability,
        ceiling=ceiling,
        mask=mask,
        per_pair=per_pair_arr,
        pair_names=names,
    )


def load_atlas() -> dict[str, np.ndarray]:
    if not ATLAS_PATH.exists():
        raise FileNotFoundError(f"{ATLAS_PATH} missing -- run scripts/fetch_atlas.py")
    z = np.load(ATLAS_PATH, allow_pickle=False)
    return {k: z[k] for k in z.files}


def per_network(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, tuple[float, int]]:
    """Mean of `values` within each Yeo-7 network -> {display_name: (mean, n_parcels)}.

    A single mean over 1000 parcels hides where a model works; this breaks it down.
    """
    atlas = load_atlas()
    network = atlas["network"]
    names = atlas["network_names"]
    display = atlas["network_display"]

    out: dict[str, tuple[float, int]] = {}
    for name, disp in zip(names, display, strict=True):
        sel = network == name
        if mask is not None:
            sel = sel & mask
        vals = values[sel]
        vals = vals[~np.isnan(vals)]
        out[str(disp)] = (float(vals.mean()) if vals.size else float("nan"), int(vals.size))
    return out
