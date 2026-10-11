"""Llama-3.2-1B text features -> ridge, with a layer sweep.

Two questions:
  1. Does a real language model beat counting words? Given the control baseline reached
     r ~= 0.12 with five features, this is a genuine question, not a formality.
  2. Which layer carries the most brain-predictive signal? If signal saturates
     mid-stack, the upper layers need not run at all: the cleanest speedup for real
     time.

Follows the official challenge baseline's dimensionality reduction (StandardScaler ->
PCA(250)).

    uv run scripts/04_text_baseline.py [--sweep-subject sub-01]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cortexlive.align import align_features, trim_fmri
from cortexlive.data import SUBJECTS, BoldStore, split_by_episode
from cortexlive.score import noise_ceiling, pearson_per_parcel, per_network

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "algonauts"
CACHE = ROOT / "cache" / "text" / "llama-3.2-1b"
OUT = ROOT / "runs" / "text_baseline"

VAL_SEASONS = (6,)
N_COMPONENTS = 250
ALPHAS = (1e3, 1e4, 1e5, 1e6)
PCA_FIT_ROWS = 40_000


def layer_features(run, layer: int) -> np.ndarray | None:
    path = CACHE / f"{run.stimulus}.npy"
    if not path.exists():
        return None
    return np.asarray(np.load(path, mmap_mode="r")[:, layer, :], dtype=np.float32)


def build(store: BoldStore, runs, layer: int, scaler=None, pca=None):
    """Return (X, Y, scaler, pca). Fits scaler/pca when not supplied (train side)."""
    per_run = []
    for run in runs:
        feats = layer_features(run, layer)
        if feats is None:
            continue
        per_run.append((run, feats))
    if not per_run:
        raise RuntimeError(f"no cached text features for layer {layer}")

    if scaler is None:
        stacked = np.concatenate([f for _, f in per_run])
        rng = np.random.default_rng(0)
        idx = rng.choice(len(stacked), size=min(PCA_FIT_ROWS, len(stacked)), replace=False)
        scaler = StandardScaler().fit(stacked[idx])
        pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(scaler.transform(stacked[idx]))
        del stacked

    xs, ys = [], []
    for run, feats in per_run:
        reduced = pca.transform(scaler.transform(feats))
        bold = trim_fmri(store.load(run))
        xs.append(align_features(reduced, bold.shape[0]))
        ys.append(bold)
    return np.concatenate(xs), np.concatenate(ys), scaler, pca


def fit_score(store: BoldStore, train_runs, val_runs, layer: int):
    x_tr, y_tr, scaler, pca = build(store, train_runs, layer)
    x_va, y_va, _, _ = build(store, val_runs, layer, scaler, pca)

    best = None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(x_tr, y_tr)
        r = pearson_per_parcel(model.predict(x_va), y_va)
        if best is None or r.mean() > best[1].mean():
            best = (alpha, r)
    return best  # (alpha, r_per_parcel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-subject", default="sub-01")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    n_layers = np.load(next(CACHE.glob("*.npy")), mmap_mode="r").shape[1]
    print(f"layers available: {n_layers}\n")

    # --- layer sweep on one subject ---
    store = BoldStore(DATA, args.sweep_subject)
    val_groups = [g for g in store.groups() if any(f"-s{s:02d}e" in g for s in VAL_SEASONS)]
    train_runs, val_runs = split_by_episode(store.runs, val_groups=val_groups)

    sweep = {}
    print(f"=== layer sweep ({args.sweep_subject}) ===")
    print(f"{'layer':>6}{'mean r':>10}{'alpha':>10}")
    for layer in range(n_layers):
        alpha, r = fit_score(store, train_runs, val_runs, layer)
        sweep[layer] = float(r.mean())
        print(f"{layer:>6}{r.mean():>10.4f}{alpha:>10.0e}")

    best_layer = max(sweep, key=sweep.get)
    print(f"\nbest layer: {best_layer}  (r={sweep[best_layer]:.4f})")
    print(f"depth fraction: {best_layer}/{n_layers - 1} = {best_layer / (n_layers - 1):.0%}")

    # --- all subjects at the best layer ---
    summary: dict[str, object] = {
        "model": "meta-llama/Llama-3.2-1B",
        "n_components": N_COMPONENTS,
        "val_split": f"Friends season(s) {VAL_SEASONS}",
        "layer_sweep_subject": args.sweep_subject,
        "layer_sweep": sweep,
        "best_layer": int(best_layer),
        "n_layers": int(n_layers),
        "subjects": {},
    }

    for subject in SUBJECTS:
        st = BoldStore(DATA, subject)
        vg = [g for g in st.groups() if any(f"-s{s:02d}e" in g for s in VAL_SEASONS)]
        tr_runs, va_runs = split_by_episode(st.runs, val_groups=vg)
        alpha, r = fit_score(st, tr_runs, va_runs, best_layer)

        nc = noise_ceiling(st)
        r_norm = nc.normalize(r)
        np.save(OUT / f"r_per_parcel_{subject}.npy", r)

        raw_net = per_network(r)
        norm_net = per_network(r_norm)
        summary["subjects"][subject] = {  # type: ignore[index]
            "alpha": alpha,
            "r_mean": float(r.mean()),
            "r_mean_reliable": float(r[nc.mask].mean()),
            "r_normalized_mean": float(np.nanmean(r_norm)),
            "raw_per_network": {k: v[0] for k, v in raw_net.items()},
            "normalized_per_network": {k: v[0] for k, v in norm_net.items()},
        }
        print(f"\n=== {subject} (layer {best_layer}) ===")
        print(f"raw r={r.mean():+.4f}   normalized={np.nanmean(r_norm):+.4f}")
        print(f"{'network':<32}{'raw r':>9}{'normalized':>12}")
        for name, (val, _) in raw_net.items():
            print(f"{name:<32}{val:>+9.4f}{norm_net[name][0]:>+12.4f}")

    (OUT / "result.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'result.json'}")


if __name__ == "__main__":
    main()
