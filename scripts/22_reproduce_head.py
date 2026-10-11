"""Re-fit the shipped head from the cached features and check it against the installed one.

The recipe is 17_fit_final_head.py's: 12_trimodal's corpus (dataset, features, windows,
splits), its per-backbone scaler + PCA and its alpha grid, each alpha scored on season 6.
The only change is memory: 17 builds the 2 GB design matrix of each participant, while this
accumulates the ridge normal equations run by run in float64, so it peaks at about 4 GB.

Compares the fit with models/final_head.npz and runs/final_head/meta.json (from 17 or from
`cortexlive setup --models`) and writes runs/final_head/reproduce.json. Needs the dataset
and the feature cache (README, Reproduce); loads no backbone.

    uv run --extra train python scripts/22_reproduce_head.py
"""

from __future__ import annotations

import importlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
tm = importlib.import_module("12_trimodal")  # module name starts with a digit
fit = importlib.import_module("17_fit_final_head")

OUT = ROOT / "runs" / "final_head" / "reproduce.json"


def main() -> None:
    t0 = time.time()
    shipped = np.load(ROOT / "models" / "final_head.npz")
    meta = json.loads((ROOT / "runs" / "final_head" / "meta.json").read_text())

    corpus = tm.Corpus([tm.F.MODALITIES[n] for n in fit.LAYERS], fit.LAYERS)
    proj = corpus.fit_projections()
    z = corpus.project(proj)
    corpus.raw = None  # the projected features are all the fit needs

    out: dict = {"stimuli": len(corpus.usable), "train_stimuli": len(corpus.train_stimuli()),
                 "pca_variance_kept": {}, "projection_max_rel_diff": 0.0, "participants": {}}
    for name, (scaler, pca) in proj.items():
        out["pca_variance_kept"][name] = round(float(pca.explained_variance_ratio_.sum()), 4)
        for key, arr in (("scaler_mean", scaler.mean_), ("scaler_scale", scaler.scale_),
                         ("pca_mean", pca.mean_), ("pca_components", pca.components_)):
            ref = shipped[f"{name}_{key}"]
            diff = float(np.abs(arr.astype(np.float32) - ref).max() / (np.abs(ref).max() + 1e-12))
            out["projection_max_rel_diff"] = max(out["projection_max_rel_diff"], diff)

    width = z[corpus.usable[0]].shape[1] * tm.STIMULUS_WINDOW
    for s in tm.SUBJECTS:
        xtx, xty = np.zeros((width, width)), np.zeros((width, tm.N_PARCELS))
        sx, sy, n = np.zeros(width), np.zeros(tm.N_PARCELS), 0
        for bold, idx, stim in zip(*corpus.tensors(z, s, val=False), strict=True):
            x = z[stim][idx].reshape(len(idx), -1).astype(np.float64)
            y = bold.astype(np.float64)
            xtx += x.T @ x
            xty += x.T @ y
            sx += x.sum(0)
            sy += y.sum(0)
            n += len(x)
        mx, my = sx / n, sy / n
        a, b = xtx - n * np.outer(mx, mx), xty - n * np.outer(mx, my)  # centred, as Ridge does
        del xtx, xty
        vb, vi, vs = corpus.tensors(z, s, val=True)
        xv = np.concatenate([z[st][i].reshape(len(i), -1) for i, st in zip(vi, vs, strict=True)])
        yv = np.concatenate(vb)

        best = None
        by_alpha = {}
        for alpha in tm.ALPHAS:
            w = np.linalg.solve(a + alpha * np.eye(width), b)
            icpt = my - mx @ w
            r = float(tm.pearson_per_parcel(xv @ w.astype(np.float32) + icpt.astype(np.float32),
                                            yv).mean())
            by_alpha[f"{alpha:g}"] = round(r, 5)
            if best is None or r > best[1]:
                best = (alpha, r, w, icpt)
        alpha, r, w, icpt = best
        coef = shipped[f"{s}_coef"]
        r_shipped = float(tm.pearson_per_parcel(xv @ coef.T + shipped[f"{s}_intercept"], yv).mean())
        out["participants"][s] = {
            "train_volumes": n,
            "val_volumes": int(len(yv)),
            "val_r_by_alpha": by_alpha,
            "alpha": {"refit": alpha, "shipped": meta["per_subject"][s]["alpha"]},
            "val_r": {"refit": round(r, 5), "shipped_weights": round(r_shipped, 5),
                      "meta": round(meta["per_subject"][s]["val_r"], 5)},
            "coef_correlation": round(float(np.corrcoef(w.T.ravel(), coef.ravel())[0, 1]), 8),
            "coef_max_rel_diff": float(np.abs(w.T - coef).max() / np.abs(coef).max()),
            "intercept_max_abs_diff": float(np.abs(icpt - shipped[f"{s}_intercept"]).max()),
        }
        print(f"  {s}: alpha {alpha:g}  val r {r:.5f} (shipped weights {r_shipped:.5f})",
              flush=True)
        del a, b, xv, yv

    refit = [p["val_r"]["refit"] for p in out["participants"].values()]
    out["mean_val_r"] = {"refit": round(float(np.mean(refit)), 5),
                         "meta": round(meta["mean_val_r"], 5)}
    out["note"] = ("refit in float64 from the normal equations; the shipped head was solved by "
                   "sklearn in float32, hence coef_max_rel_diff of a few 1e-4")
    out["seconds"] = round(time.time() - t0)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nmean val r {out['mean_val_r']['refit']} (meta {out['mean_val_r']['meta']}) -> {OUT}")


if __name__ == "__main__":
    main()
