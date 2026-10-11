"""Fit and persist the shipped head: the exact model `cortexlive` loads.

Per-subject ridge (it matched the joint MLP in 12_trimodal) over the three modalities,
written with every tensor inference needs: per-modality scaler + PCA, per-subject ridge
weights, and the config that binds them (modalities, layers, PCA size, window).

The head is trained on 256px video features although it runs live at 192px: only the
256px cache covers the training seasons, and on season 6 the live configuration scores
1.5% lower (runs/live_analysis/live_config.json). Alpha is chosen per subject on season 6,
which also reproduces 12_trimodal's 0.2629.

    uv run --extra train python scripts/17_fit_final_head.py
"""

from __future__ import annotations

import importlib
import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import Ridge

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
tm = importlib.import_module("12_trimodal")  # module name starts with a digit

LAYERS = {"text": 8, "audio": 15, "video": 19}
MODELS = ROOT / "models"
OUT = ROOT / "runs" / "final_head"


def design(z, bolds, idxs, stims):
    xs = [z[s][i].reshape(len(i), -1) for i, s in zip(idxs, stims, strict=True)]
    return np.concatenate(xs), np.concatenate(bolds)


def main() -> None:
    names = list(LAYERS)
    mods = [tm.F.MODALITIES[n] for n in names]
    corpus = tm.Corpus(mods, LAYERS)
    print(f"{len(corpus.usable)} stimuli cached in all of {names}")
    proj = corpus.fit_projections()
    z = corpus.project(proj)

    MODELS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for m in corpus.mods:
        scaler, pca = proj[m.name]
        arrays[f"{m.name}_scaler_mean"] = scaler.mean_.astype(np.float32)
        arrays[f"{m.name}_scaler_scale"] = scaler.scale_.astype(np.float32)
        arrays[f"{m.name}_pca_mean"] = pca.mean_.astype(np.float32)
        arrays[f"{m.name}_pca_components"] = pca.components_.astype(np.float32)

    scores: dict[str, dict] = {}
    for s in tm.SUBJECTS:
        x_tr, y_tr = design(z, *corpus.tensors(z, s, val=False))
        x_va, y_va = design(z, *corpus.tensors(z, s, val=True))
        best = None
        for alpha in tm.ALPHAS:
            model = Ridge(alpha=alpha).fit(x_tr, y_tr)
            r = tm.pearson_per_parcel(model.predict(x_va), y_va)
            if best is None or r.mean() > best[1].mean():
                best = (alpha, r, model)
        alpha, r, model = best
        arrays[f"{s}_coef"] = model.coef_.astype(np.float32)
        arrays[f"{s}_intercept"] = model.intercept_.astype(np.float32)
        scores[s] = {"alpha": alpha, "val_r": float(r.mean())}
        print(f"  {s}: alpha {alpha:g}  val r {r.mean():.4f}")
        del x_tr, y_tr, x_va, y_va

    meta = {
        "modalities": names,
        "layers": LAYERS,
        "n_components": tm.N_COMPONENTS,
        "window": tm.STIMULUS_WINDOW,
        "note": "trained on 256px video features; runs live at 192px, which scores 1.5% "
        "lower on season 6 (runs/live_analysis/live_config.json)",
        "per_subject": scores,
        "mean_val_r": float(np.mean([v["val_r"] for v in scores.values()])),
    }
    np.savez_compressed(MODELS / "final_head.npz", **arrays)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    size = (MODELS / "final_head.npz").stat().st_size / 1e6
    print(f"\nmean val r {meta['mean_val_r']:.4f}")
    print(f"wrote {MODELS / 'final_head.npz'} ({size:.0f} MB) and {OUT / 'meta.json'}")


if __name__ == "__main__":
    main()
