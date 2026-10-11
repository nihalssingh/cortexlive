"""The control baseline: ridge on transcript-derived speech features.

No backbone. Just how much speech is happening. Every later model must beat this line,
per cortical network, or its score in that network means nothing.

    uv run scripts/02_control_baseline.py

Writes runs/control_baseline/{result.json, r_per_parcel_<subject>.npy}.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from sklearn.linear_model import Ridge

from cortexlive.align import align_features, trim_fmri
from cortexlive.data import SUBJECTS, BoldStore, split_by_episode
from cortexlive.score import noise_ceiling, pearson_per_parcel, per_network
from cortexlive.stimulus import FEATURE_NAMES, control_features, transcript_path

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "algonauts"
OUT = ROOT / "runs" / "control_baseline"

# Frozen so every experiment in this project is scored on the same held-out episodes.
# Chosen as whole Friends seasons: season 6 is ~17% of Friends and holding out entire
# seasons removes any chance of same-episode leakage across the split.
VAL_SEASONS = (6,)
ALPHAS = (1e2, 1e3, 1e4, 1e5)


def build_xy(store: BoldStore, runs) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for run in runs:
        path = transcript_path(DATA, run.source, run.stimulus)
        if path is None:
            continue
        feats = control_features(str(path))
        bold = trim_fmri(store.load(run))
        xs.append(align_features(feats, bold.shape[0]))
        ys.append(bold)
    if not xs:
        raise RuntimeError("no runs had transcripts")
    return np.concatenate(xs), np.concatenate(ys)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "description": "Ridge on 5 transcript speech features x 5-TR window. No backbone.",
        "features": list(FEATURE_NAMES),
        "val_split": f"Friends season(s) {VAL_SEASONS} held out (whole seasons)",
        "subjects": {},
    }

    for subject in SUBJECTS:
        store = BoldStore(DATA, subject)
        val_groups = [g for g in store.groups() if any(f"-s{s:02d}e" in g for s in VAL_SEASONS)]
        train_runs, val_runs = split_by_episode(store.runs, val_groups=val_groups)

        x_tr, y_tr = build_xy(store, train_runs)
        x_va, y_va = build_xy(store, val_runs)

        mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
        x_tr, x_va = (x_tr - mu) / sd, (x_va - mu) / sd

        best = None
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(x_tr, y_tr)
            r = pearson_per_parcel(model.predict(x_va), y_va)
            if best is None or r.mean() > best[1].mean():
                best = (alpha, r)
        alpha, r = best  # type: ignore[misc]

        nc = noise_ceiling(store)
        r_norm = nc.normalize(r)

        np.save(OUT / f"r_per_parcel_{subject}.npy", r)
        raw_net = per_network(r)
        norm_net = per_network(r_norm)

        summary["subjects"][subject] = {  # type: ignore[index]
            "alpha": alpha,
            "n_train_samples": int(x_tr.shape[0]),
            "n_val_samples": int(x_va.shape[0]),
            "n_features": int(x_tr.shape[1]),
            "r_mean": float(r.mean()),
            "r_mean_reliable": float(r[nc.mask].mean()),
            "r_normalized_mean": float(np.nanmean(r_norm)),
            "raw_per_network": {k: v[0] for k, v in raw_net.items()},
            "normalized_per_network": {k: v[0] for k, v in norm_net.items()},
        }

        print(
            f"\n=== {subject} ===  alpha={alpha:g}  "
            f"train={x_tr.shape[0]} val={x_va.shape[0]} dims={x_tr.shape[1]}"
        )
        print(f"raw r (all parcels)      : {r.mean():+.4f}")
        print(f"raw r (reliable parcels) : {r[nc.mask].mean():+.4f}")
        print(f"ceiling-normalized       : {np.nanmean(r_norm):+.4f}")
        print(f"{'network':<32}{'raw r':>9}{'normalized':>12}")
        for name, (val, _) in raw_net.items():
            print(f"{name:<32}{val:>+9.4f}{norm_net[name][0]:>+12.4f}")

    (OUT / "result.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'result.json'}")


if __name__ == "__main__":
    main()
