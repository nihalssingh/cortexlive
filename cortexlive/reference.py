"""Reference data a capture is compared against.

  * display masks: parcels whose test-retest reliability supports a claim (others are
    drawn grey)
  * reference statistics: per-parcel mean and SD of the model's predictions over the
    Friends season-6 runs (50 episode parts; 49 for sub-03), the scale for whole-run maps
    of any video
  * for inputs that are CNeuroMod stimuli: the four participants' measured fMRI and a
    control baseline (ridge on five transcript speech features, no neural network)

The first two are built once from the dataset (`build`) and ship in models/reference.npz,
so any video can be viewed on any machine; the third needs data/algonauts.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import numpy as np

from . import features as F
from .align import EXCLUDED_START, HRF_DELAY, STIMULUS_WINDOW, align_features, trim_fmri
from .data import SUBJECTS, BoldStore, split_by_episode
from .score import load_atlas, noise_ceiling, pearson_per_parcel
from .stimulus import control_features, transcript_path, words_per_tr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "algonauts"
MODELS = ROOT / "models"
REFERENCE = MODELS / "reference.npz"
CONTROL = MODELS / "control_head.npz"
VAL_SEASONS = (6,)
CEILING_FLOOR = 0.25
CONTROL_ALPHAS = (1e2, 1e3, 1e4, 1e5)


# ------------------------------------------------------------------ known stimuli


@functools.cache
def _stimuli() -> dict[str, str]:
    """stimulus id -> source ('friends' / 'movie10'), for stimuli with recorded fMRI."""
    if not DATA.exists():
        return {}
    try:
        return {r.stimulus: r.source for r in BoldStore(DATA, SUBJECTS[0]).runs}
    except Exception:
        return {}


def known_stimulus(name: str) -> dict | None:
    """Recognise a CNeuroMod stimulus from a file name, e.g. friends_s06e01a.mkv."""
    stims = _stimuli()
    if not stims:
        return None
    low = Path(name).name.lower()
    m = re.search(r"s(\d{2})e(\d{2})([a-d])", low)
    hit = None
    if m and f"s{m.group(1)}e{m.group(2)}{m.group(3)}" in stims:
        hit = f"s{m.group(1)}e{m.group(2)}{m.group(3)}"
    else:
        for s in sorted(stims, key=len, reverse=True):
            if stims[s] != "friends" and re.search(rf"(^|[^a-z0-9]){re.escape(s)}([^a-z0-9]|$)",
                                                    Path(low).stem):
                hit = s
                break
    if hit is None:
        return None
    season = re.match(r"s(\d{2})e", hit)
    split = "validation" if season and int(season.group(1)) in VAL_SEASONS else "training"
    return {"stimulus": hit, "source": stims[hit], "split": split}


def transcript(known: dict) -> list[list[str]] | None:
    path = transcript_path(DATA, known["source"], known["stimulus"])
    return words_per_tr(path) if path else None


def measured(known: dict) -> dict[str, np.ndarray]:
    """subject -> (n_volumes, 1000) recorded BOLD, untrimmed (row s = volume s)."""
    out = {}
    for s in SUBJECTS:
        store = BoldStore(DATA, s)
        run = next((r for r in store.runs if r.stimulus == known["stimulus"]), None)
        if run is not None:
            out[s] = store.load(run).astype(np.float32)
    return out


# ------------------------------------------------------------------ control baseline


def _val_split(store: BoldStore):
    vg = [g for g in store.groups() if any(f"-s{n:02d}e" in g for n in VAL_SEASONS)]
    return split_by_episode(store.runs, val_groups=vg)


def _fit_control(store: BoldStore) -> dict[str, np.ndarray]:
    """scripts/02_control_baseline.py's protocol: standardized windowed features, ridge
    alpha chosen on season 6. The standardization is folded into the weights."""
    from sklearn.linear_model import Ridge

    train, val = _val_split(store)

    def xy(runs):
        xs, ys = [], []
        for run in runs:
            path = transcript_path(DATA, run.source, run.stimulus)
            if path is None:
                continue
            bold = trim_fmri(store.load(run))
            xs.append(align_features(control_features(str(path)), bold.shape[0]))
            ys.append(bold)
        return np.concatenate(xs), np.concatenate(ys)

    x_tr, y_tr = xy(train)
    x_va, y_va = xy(val)
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
    best = None
    for alpha in CONTROL_ALPHAS:
        m = Ridge(alpha=alpha).fit((x_tr - mu) / sd, y_tr)
        r = float(pearson_per_parcel(m.predict((x_va - mu) / sd), y_va).mean())
        if best is None or r > best[2]:
            best = (m, alpha, r)
    m, alpha, r = best
    return {"coef": (m.coef_ / sd).astype(np.float32),
            "intercept": (m.intercept_ - (mu / sd) @ m.coef_.T).astype(np.float32),
            "alpha": np.array(alpha), "val_r": np.array(r)}


@functools.cache
def control_model() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """subject -> (coef, intercept) of the control ridge; empty if not built."""
    if not CONTROL.exists():
        return {}
    with np.load(CONTROL) as z:
        return {s: (z[f"{s}_coef"], z[f"{s}_intercept"]) for s in SUBJECTS if f"{s}_coef" in z}


def control_predictions(known: dict, n_trimmed: dict[str, int]) -> dict[str, np.ndarray]:
    """subject -> (n_trimmed, 1000) control predictions on the trimmed timeline."""
    path = transcript_path(DATA, known["source"], known["stimulus"])
    if path is None:
        return {}
    feats = control_features(str(path))
    out = {}
    for s, (coef, icpt) in control_model().items():
        if s in n_trimmed:
            out[s] = align_features(feats, n_trimmed[s]) @ coef.T + icpt
    return out


# --------------------------------------------------------------------- reference


@functools.cache
def load() -> dict | None:
    """{'mask': {subj: bool[1000]}, 'mean': {...}, 'sd': {...}} or None if not built."""
    if not REFERENCE.exists():
        return None
    with np.load(REFERENCE) as z:
        subs = [s for s in SUBJECTS if f"{s}_mask" in z]
        return {"mask": {s: z[f"{s}_mask"].astype(bool) for s in subs},
                "mean": {s: z[f"{s}_mean"] for s in subs},
                "sd": {s: z[f"{s}_sd"] for s in subs}}


def build(head, log=print) -> None:
    """Write models/reference.npz and models/control_head.npz from the dataset and the
    feature cache (maintainers; users get both from the model bundle)."""
    video = F.Modality("video", f"{F.VIDEO.subdir}_res192v2", F.VIDEO.cached_hz,
                       F.VIDEO.best_layer)  # the live configuration
    mods = {"text": F.TEXT, "audio": F.AUDIO, "video": video}
    rows: dict[str, np.ndarray | None] = {}  # stimulus -> projected rows, shared

    def projected(stim: str) -> np.ndarray | None:
        if stim not in rows:
            parts = []
            for name in head.modalities:
                f = F.load_canonical(mods[name], stim, head.layers[name])
                if f is None:
                    rows[stim] = None
                    return None
                parts.append(head.project(name, f))
            n = min(len(p) for p in parts)
            rows[stim] = np.concatenate([p[:n] for p in parts], axis=1)
        return rows[stim]

    ref, ctrl = {}, {}
    for s in SUBJECTS:
        store = BoldStore(DATA, s)
        nc = noise_ceiling(store)
        ref[f"{s}_mask"] = (nc.mask & (nc.ceiling >= CEILING_FLOOR)).astype(np.uint8)
        preds = []
        for run in _val_split(store)[1]:
            z = projected(run.stimulus)
            if z is None:
                continue
            n = trim_fmri(store.load(run)).shape[0]
            ends = np.arange(n) + EXCLUDED_START - HRF_DELAY
            idx = np.clip(ends[:, None] - np.arange(STIMULUS_WINDOW - 1, -1, -1)[None, :],
                          0, len(z) - 1)
            preds.append(head.predict_rows(s, z[idx].reshape(n, -1)))
        p = np.concatenate(preds)
        ref[f"{s}_mean"] = p.mean(0).astype(np.float32)
        ref[f"{s}_sd"] = p.std(0).astype(np.float32)
        fit = _fit_control(store)
        ctrl.update({f"{s}_{k}": v for k, v in fit.items()})
        log(f"  {s}: {len(preds)} season-6 runs, {int(ref[f'{s}_mask'].sum())} reliable "
            f"parcels; control alpha {float(fit['alpha']):g}, val r {float(fit['val_r']):.4f}")
    MODELS.mkdir(exist_ok=True)
    np.savez_compressed(REFERENCE, **ref)
    np.savez_compressed(CONTROL, **ctrl)
    load.cache_clear()
    control_model.cache_clear()


# ----------------------------------------------------------------------- scoring


def _mean(x: np.ndarray) -> float | None:
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def score(arrays: dict[str, np.ndarray], known: dict) -> tuple[dict, dict]:
    """Accuracy of a capture's predictions against the recorded fMRI.

    Pearson r per parcel over the volumes the run predicted, on the official trimmed
    timeline (first and last 5 volumes dropped); the control baseline is scored on the
    same volumes. Returns (summary for capture.json, per-parcel arrays for scores.npz).
    """
    bold = measured(known)
    trimmed = {s: trim_fmri(b) for s, b in bold.items()}
    ctrl = control_predictions(known, {s: len(t) for s, t in trimmed.items()})
    ref = load()
    atlas = load_atlas()
    nets = [str(x) for x in atlas["network_display"]]
    net_idx = atlas["network_index"]
    samples = arrays["samples"].astype(int)
    summary, parcels = {}, {}
    for s, y in trimmed.items():
        if s not in arrays:
            continue
        rows = samples - EXCLUDED_START
        keep = (rows >= 0) & (rows < len(y))
        if keep.sum() < 3:
            continue
        i = rows[keep]
        r_live = pearson_per_parcel(arrays[s][keep], y[i])
        r_ctrl = pearson_per_parcel(ctrl[s][i], y[i]) if s in ctrl else np.full(1000, np.nan)
        mask = ref["mask"][s] if ref else np.ones(1000, bool)
        by_net = {}
        for k, name in enumerate(nets):
            m = mask & (net_idx == k)
            if m.any():
                by_net[name] = {"r": _mean(r_live[m]), "control_r": _mean(r_ctrl[m]),
                                "parcels": int(m.sum())}
        summary[s] = {"volumes_scored": int(keep.sum()), "volumes_available": int(len(y)),
                      "r": _mean(r_live), "r_reliable": _mean(r_live[mask]),
                      "control_r": _mean(r_ctrl), "control_r_reliable": _mean(r_ctrl[mask]),
                      "reliable_parcels": int(mask.sum()), "by_network": by_net}
        parcels[s] = {"r": r_live.astype(np.float32), "control_r": r_ctrl.astype(np.float32)}

    def avg(k):
        v = [d[k] for d in summary.values() if d[k] is not None]
        return float(np.mean(v)) if v else None

    keys = ("r", "r_reliable", "control_r", "control_r_reliable")
    out = {
        "stimulus": known["stimulus"],
        "split": known["split"],
        "protocol": "Pearson r per parcel over the predicted volumes of the trimmed "
                    "timeline; mean over all 1000 parcels, or over reliable parcels "
                    f"(test-retest significant and noise ceiling >= {CEILING_FLOOR}); "
                    "control = ridge on five transcript speech features",
        "per_subject": summary,
        "mean": {k: avg(k) for k in keys} if summary else None,
    }
    return out, parcels
