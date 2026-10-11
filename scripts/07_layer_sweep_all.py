"""Score every Llama 3.2 1B layer for all four subjects.

The live text stream stops at the best layer, so that layer has to be the same for
everyone. If the peak sat at a different depth per subject, truncating at one depth would
tune the real-time system to one person.

    uv run scripts/07_layer_sweep_all.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cortexlive.align import align_features, trim_fmri
from cortexlive.data import SUBJECTS, BoldStore, split_by_episode
from cortexlive.score import pearson_per_parcel

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "algonauts"
CACHE = ROOT / "cache" / "text" / "llama-3.2-1b"
OUT = ROOT / "runs" / "layer_sweep_all"

VAL_SEASONS = (6,)
N_COMPONENTS = 250
ALPHAS = (1e3, 1e4, 1e5, 1e6)


def cached(stimulus: str, layer: int) -> np.ndarray | None:
    p = CACHE / f"{stimulus}.npy"
    if not p.exists():
        return None
    return np.asarray(np.load(p, mmap_mode="r")[:, layer, :], dtype=np.float32)


def score_layer(store: BoldStore, train_runs, val_runs, layer: int) -> float:
    tr = [(r, f) for r in train_runs if (f := cached(r.stimulus, layer)) is not None]
    va = [(r, f) for r in val_runs if (f := cached(r.stimulus, layer)) is not None]

    stacked = np.concatenate([f for _, f in tr])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(stacked), size=min(40_000, len(stacked)), replace=False)
    scaler = StandardScaler().fit(stacked[idx])
    pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(scaler.transform(stacked[idx]))
    del stacked

    def build(pairs):
        xs, ys = [], []
        for run, feats in pairs:
            bold = trim_fmri(store.load(run))
            xs.append(align_features(pca.transform(scaler.transform(feats)), bold.shape[0]))
            ys.append(bold)
        return np.concatenate(xs), np.concatenate(ys)

    x_tr, y_tr = build(tr)
    x_va, y_va = build(va)

    best = -np.inf
    for alpha in ALPHAS:
        r = pearson_per_parcel(Ridge(alpha=alpha).fit(x_tr, y_tr).predict(x_va), y_va)
        best = max(best, float(r.mean()))
    return best


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n_layers = np.load(next(CACHE.glob("*.npy")), mmap_mode="r").shape[1]

    table: dict[str, dict[int, float]] = {}
    for subject in SUBJECTS:
        store = BoldStore(DATA, subject)
        vg = [g for g in store.groups() if any(f"-s{s:02d}e" in g for s in VAL_SEASONS)]
        train_runs, val_runs = split_by_episode(store.runs, val_groups=vg)
        table[subject] = {}
        for layer in range(n_layers):
            table[subject][layer] = score_layer(store, train_runs, val_runs, layer)
        peak = max(table[subject], key=table[subject].get)
        print(f"{subject}: peak at layer {peak} (r={table[subject][peak]:.4f})")

    print(f"\n{'layer':>6}" + "".join(f"{s:>10}" for s in SUBJECTS) + f"{'mean':>10}")
    means = {}
    for layer in range(n_layers):
        vals = [table[s][layer] for s in SUBJECTS]
        means[layer] = float(np.mean(vals))
        mark = ""
        print(f"{layer:>6}" + "".join(f"{v:>10.4f}" for v in vals) + f"{means[layer]:>10.4f}{mark}")

    peaks = {s: int(max(table[s], key=table[s].get)) for s in SUBJECTS}
    mean_peak = int(max(means, key=means.get))
    print(f"\nper-subject peaks : {peaks}")
    print(f"peak of the mean  : layer {mean_peak}")

    # How much accuracy is lost by using one global depth instead of each subject's own?
    cost = {s: table[s][peaks[s]] - table[s][mean_peak] for s in SUBJECTS}
    print(
        f"cost of using layer {mean_peak} for everyone: "
        + ", ".join(f"{s} {c:+.4f}" for s, c in cost.items())
    )

    (OUT / "result.json").write_text(
        json.dumps(
            {
                "n_layers": int(n_layers),
                "per_subject": {s: {str(k): v for k, v in table[s].items()} for s in SUBJECTS},
                "mean_over_subjects": {str(k): v for k, v in means.items()},
                "per_subject_peak": peaks,
                "peak_of_mean": mean_peak,
                "cost_of_global_depth": cost,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT / 'result.json'}")


if __name__ == "__main__":
    main()
