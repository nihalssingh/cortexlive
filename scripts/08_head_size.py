"""Does the size of the head matter?

Same cached text features (sub-01, Llama layer 8), same frozen split, same scoring: four
heads from 0.6 M to 86 M parameters.

  ridge        linear, closed-form                      1.25 M
  mlp-small    one hidden layer, 256 units              0.58 M
  mlp-large    two hidden layers, 2048 units            8.8 M
  mlp-xl       two hidden layers, 8192 units            85.6 M

If they tie, a small linear head is enough. If the big ones win clearly, ridge is leaving
accuracy on the table.

    uv run --extra train python scripts/08_head_size.py
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn

from cortexlive.align import align_features, trim_fmri
from cortexlive.data import BoldStore, split_by_episode
from cortexlive.score import noise_ceiling, pearson_per_parcel, per_network

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "algonauts"
CACHE = ROOT / "cache" / "text" / "llama-3.2-1b"
OUT = ROOT / "runs" / "head_size"

SUBJECT = "sub-01"
LAYER = 8
N_COMPONENTS = 250
VAL_SEASONS = (6,)
EPOCHS = 30
BATCH = 512


def cached(stim: str) -> np.ndarray | None:
    p = CACHE / f"{stim}.npy"
    if not p.exists():
        return None
    return np.asarray(np.load(p, mmap_mode="r")[:, LAYER, :], dtype=np.float32)


def make_mlp(d_in: int, hidden: list[int], d_out: int) -> nn.Module:
    layers: list[nn.Module] = []
    prev = d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.2)]
        prev = h
    layers.append(nn.Linear(prev, d_out))
    return nn.Sequential(*layers)


def train_mlp(x_tr, y_tr, x_va, y_va, hidden, device, seed=0) -> tuple[np.ndarray, int]:
    torch.manual_seed(seed)
    model = make_mlp(x_tr.shape[1], hidden, y_tr.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, total_steps=EPOCHS * max(1, len(x_tr) // BATCH)
    )
    xt = torch.tensor(x_tr, device=device)
    yt = torch.tensor(y_tr, device=device)
    xv = torch.tensor(x_va, device=device)

    best_r, best = -np.inf, None
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt) - BATCH + 1, BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            r = pearson_per_parcel(model(xv).cpu().numpy(), y_va)
        if r.mean() > best_r:
            best_r, best = r.mean(), r
    return best, n_params


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    store = BoldStore(DATA, SUBJECT)
    vg = [g for g in store.groups() if any(f"-s{s:02d}e" in g for s in VAL_SEASONS)]
    train_runs, val_runs = split_by_episode(store.runs, val_groups=vg)

    tr = [(r, f) for r in train_runs if (f := cached(r.stimulus)) is not None]
    va = [(r, f) for r in val_runs if (f := cached(r.stimulus)) is not None]

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
        return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)

    x_tr, y_tr = build(tr)
    x_va, y_va = build(va)
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    x_tr, x_va = (x_tr - mu) / sd, (x_va - mu) / sd
    print(f"device={device}  train={x_tr.shape}  val={x_va.shape}\n")

    nc = noise_ceiling(store)
    results: dict[str, dict] = {}

    # --- ridge ---
    t0 = time.time()
    best = None
    for alpha in (1e3, 1e4, 1e5, 1e6):
        r = pearson_per_parcel(Ridge(alpha=alpha).fit(x_tr, y_tr).predict(x_va), y_va)
        if best is None or r.mean() > best.mean():
            best = r
    results["ridge"] = {
        "params": x_tr.shape[1] * y_tr.shape[1] + y_tr.shape[1],
        "r": float(best.mean()),
        "seconds": round(time.time() - t0, 1),
        "normalized_per_network": {k: v[0] for k, v in per_network(nc.normalize(best)).items()},
    }

    # --- MLPs ---
    configs = [("mlp-small", [256]), ("mlp-large", [2048, 2048]), ("mlp-xl", [8192, 8192])]
    for name, hidden in configs:
        t0 = time.time()
        r, n_params = train_mlp(x_tr, y_tr, x_va, y_va, hidden, device)
        results[name] = {
            "params": int(n_params),
            "hidden": hidden,
            "r": float(r.mean()),
            "seconds": round(time.time() - t0, 1),
            "normalized_per_network": {k: v[0] for k, v in per_network(nc.normalize(r)).items()},
        }

    base = results["ridge"]["r"]
    print(f"{'head':<12}{'params':>12}{'mean r':>10}{'vs ridge':>10}{'train s':>9}")
    for name, v in results.items():
        print(
            f"{name:<12}{v['params'] / 1e6:>11.2f}M{v['r']:>10.4f}"
            f"{v['r'] - base:>+10.4f}{v['seconds']:>9.1f}"
        )

    (OUT / "result.json").write_text(
        json.dumps(
            {
                "subject": SUBJECT,
                "layer": LAYER,
                "modality": "text only",
                "note": "Tests whether head SIZE matters on our data, rather than citing "
                "the Algonauts organizers' architecture-agnostic finding.",
                "heads": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT / 'result.json'}")


if __name__ == "__main__":
    main()
