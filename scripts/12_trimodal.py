"""Trimodal readouts: a ridge per subject against one network trained on all four.

Two things are being tested here, and they are separable.

**Does joint training help?** The other scripts fit one model per subject, while TRIBE
trains one network across subjects. Four brains disagree about which parcel is which, but
they agree about what the stimulus is, so a shared trunk sees 4x the data while a
per-subject linear readout absorbs the anatomy. The `separate` arm is the same
network trained one subject at a time, so the difference is joint training and nothing
else. The shipped records (runs/trimodal/) hold only the ridge and joint arms; the shipped
model is the per-subject ridge, refit and saved by 17_fit_final_head.py.

**Does each modality earn its place?** Audio is the one to be suspicious of: speech
already reaches the model through the transcript, so w2v-BERT's 0.6 B parameters mostly
add prosody, music and non-speech sound. If `text+video` matches `all`, audio costs
real-time budget for nothing.

Memory note: the network arms never materialise the windowed design matrix.
`align_features` would expand every sample 5x, about 10 GB in float32 for four subjects;
instead each stimulus keeps one (n_tr, d) array and batches gather their own windows
through an index matrix. (The ridge arm builds it one subject at a time, about 2.4 GB.)

    uv run --extra train python scripts/12_trimodal.py --arms ridge,joint
    uv run --extra train python scripts/12_trimodal.py --modalities text,video --arms ridge,joint

The network arms set no torch seed, so a rerun matches their records only approximately.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import warnings

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn

from cortexlive import features as F
from cortexlive.align import EXCLUDED_START, HRF_DELAY, STIMULUS_WINDOW, trim_fmri
from cortexlive.data import N_PARCELS, SUBJECTS, BoldStore, split_by_episode
from cortexlive.score import noise_ceiling, pearson_per_parcel, per_network

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "algonauts"
OUT = ROOT / "runs" / "trimodal"

VAL_SEASONS = (6,)
N_COMPONENTS = 250
PCA_SAMPLE = 40_000
ALPHAS = (1e3, 1e4, 1e5, 1e6)

HIDDEN = 1024
EPOCHS = 40
BATCH = 256
LR = 1e-3


# --------------------------------------------------------------------------- data


def window_index(n_trimmed: int, n_stim: int, window: int = STIMULUS_WINDOW) -> np.ndarray:
    """(n_trimmed, window) int32 indices into a stimulus's feature array.

    Exactly the indexing `align_features` does, minus the gather -- so the 5x expansion
    happens per batch on the fly instead of once for the whole corpus.
    """
    ends = np.arange(n_trimmed) + EXCLUDED_START - HRF_DELAY
    idx = ends[:, None] - np.arange(window - 1, -1, -1)[None, :]
    return np.clip(idx, 0, n_stim - 1).astype(np.int32)


class Corpus:
    """Per-stimulus features (shared across subjects) plus per-subject BOLD and indices."""

    def __init__(self, mods: list[F.Modality], layers: dict[str, int]):
        self.mods = mods
        self.layers = layers
        self.stores = {s: BoldStore(DATA, s) for s in SUBJECTS}
        self.splits = {}
        for s, st in self.stores.items():
            vg = [g for g in st.groups() if any(f"-s{sn:02d}e" in g for sn in VAL_SEASONS)]
            self.splits[s] = split_by_episode(st.runs, val_groups=vg)

        needed = {r.stimulus for st in self.stores.values() for r in st.runs}
        common = F.common_stimuli(*mods) if mods else set()
        self.usable = sorted(needed & common)
        if not self.usable:
            raise SystemExit(
                "no stimulus is cached for every requested modality -- "
                f"have {[f'{m.name}:{len(F.available(m))}' for m in mods]}"
            )

        self.raw: dict[str, dict[str, np.ndarray]] = {}
        for m in mods:
            layer = layers[m.name]
            got = {}
            for stim in self.usable:
                f = F.load_canonical(m, stim, layer)
                if f is not None:
                    got[stim] = f
            self.raw[m.name] = got
        # Keep only stimuli that survived loading in every modality.
        self.usable = sorted(set.intersection(*(set(d) for d in self.raw.values())))
        # Modalities can disagree by one TR at the tail: video drops its final partial
        # 64-frame clip, so its canonical grid is sometimes a TR shorter than text's.
        # Cut to the shortest -- the tail is inside the 7.45 s trim_fmri discards anyway.
        for stim in self.usable:
            n = min(self.raw[m.name][stim].shape[0] for m in mods)
            for m in mods:
                self.raw[m.name][stim] = self.raw[m.name][stim][:n]

    def train_stimuli(self) -> list[str]:
        """Stimuli that are in some subject's TRAIN split -- never a val episode."""
        val = {r.stimulus for s in SUBJECTS for r in self.splits[s][1]}
        return [s for s in self.usable if s not in val]

    def fit_projections(self) -> dict[str, tuple[StandardScaler, PCA]]:
        """One scaler+PCA per modality, fit on train stimuli only."""
        proj = {}
        rng = np.random.default_rng(0)
        train = self.train_stimuli()
        for m in self.mods:
            stacked = np.concatenate([self.raw[m.name][s] for s in train])
            idx = rng.choice(len(stacked), size=min(PCA_SAMPLE, len(stacked)), replace=False)
            scaler = StandardScaler().fit(stacked[idx])
            pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(scaler.transform(stacked[idx]))
            proj[m.name] = (scaler, pca)
            ev = float(pca.explained_variance_ratio_.sum())
            print(
                f"  {m.name:<6} layer {self.layers[m.name]:>2}  PCA {N_COMPONENTS}d keeps {ev:.1%}"
            )
            del stacked
        return proj

    def project(self, proj) -> dict[str, np.ndarray]:
        """stimulus -> (n_tr, N_COMPONENTS * n_modalities) float32, modalities concatenated."""
        out = {}
        for stim in self.usable:
            parts = []
            for m in self.mods:
                scaler, pca = proj[m.name]
                parts.append(pca.transform(scaler.transform(self.raw[m.name][stim])))
            out[stim] = np.concatenate(parts, axis=1).astype(np.float32)
        return out

    def tensors(self, z: dict[str, np.ndarray], subject: str, val: bool):
        """(bold, stim_ids, index_matrix) for one subject and split."""
        runs = self.splits[subject][1 if val else 0]
        bolds, idxs, stims = [], [], []
        for run in runs:
            if run.stimulus not in z:
                continue
            bold = trim_fmri(self.stores[subject].load(run))
            bolds.append(bold.astype(np.float32))
            idxs.append(window_index(bold.shape[0], z[run.stimulus].shape[0]))
            stims.append(run.stimulus)
        return bolds, idxs, stims


# --------------------------------------------------------------------------- model


class JointHead(nn.Module):
    """Shared trunk, one linear readout per subject.

    The readout is where anatomy lives: parcel 412 is not the same tissue in two people,
    so nothing above the readout should be subject-specific, and nothing below it should
    be shared. TRIBE adds a learned subject embedding into the trunk instead; a separate
    output matrix is the simpler form of the same idea and needs no extra plumbing.
    """

    def __init__(self, d_in: int, subjects: list[str], hidden: int = HIDDEN, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.readout = nn.ModuleDict({s: nn.Linear(hidden, N_PARCELS) for s in subjects})

    def forward(self, x: torch.Tensor, subject: str) -> torch.Tensor:
        return self.readout[subject](self.trunk(x))


class Batcher:
    """Gathers windowed batches without ever materialising the full design matrix."""

    def __init__(self, z, bolds, idxs, stims, device):
        self.z = {k: torch.from_numpy(v) for k, v in z.items()}
        self.device = device
        self.runs = []
        for bold, idx, stim in zip(bolds, idxs, stims, strict=True):
            self.runs.append((torch.from_numpy(bold), torch.from_numpy(idx.astype(np.int64)), stim))
        self.offsets = np.cumsum([0] + [b.shape[0] for b, _, _ in self.runs])
        self.n = int(self.offsets[-1])

    def gather(self, rows: np.ndarray):
        xs, ys = [], []
        run_of = np.searchsorted(self.offsets, rows, side="right") - 1
        for r in np.unique(run_of):
            bold, idx, stim = self.runs[r]
            local = torch.from_numpy((rows[run_of == r] - self.offsets[r]).astype(np.int64))
            xs.append(self.z[stim][idx[local]].reshape(len(local), -1))
            ys.append(bold[local])
        return torch.cat(xs).to(self.device), torch.cat(ys).to(self.device)

    def all_rows(self) -> np.ndarray:
        return np.arange(self.n)


def evaluate(model, batchers: dict[str, Batcher], device) -> dict[str, np.ndarray]:
    model.eval()
    out = {}
    with torch.no_grad():
        for subject, b in batchers.items():
            preds, trues = [], []
            for start in range(0, b.n, 4096):
                rows = np.arange(start, min(start + 4096, b.n))
                x, y = b.gather(rows)
                preds.append(model(x, subject).float().cpu().numpy())
                trues.append(y.cpu().numpy())
            out[subject] = pearson_per_parcel(np.concatenate(preds), np.concatenate(trues))
    return out


def train(model, train_b, val_b, device, epochs=EPOCHS, tag="joint"):
    """Trains on whichever subjects are in `train_b`; early-stops on mean val r."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    steps = sum(max(1, b.n // BATCH) for b in train_b.values())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * steps)
    rng = np.random.default_rng(0)

    best_mean, best = -np.inf, None
    for epoch in range(epochs):
        model.train()
        # One shuffled pass per subject, with the subject order itself interleaved so
        # consecutive optimiser steps do not all come from the same brain.
        plan = []
        for subject, b in train_b.items():
            order = rng.permutation(b.n)
            for i in range(0, b.n - BATCH + 1, BATCH):
                plan.append((subject, order[i : i + BATCH]))
        rng.shuffle(plan)
        for subject, rows in plan:
            x, y = train_b[subject].gather(np.sort(rows))
            opt.zero_grad()
            nn.functional.mse_loss(model(x, subject), y).backward()
            opt.step()
            sched.step()

        scores = evaluate(model, val_b, device)
        mean = float(np.mean([s.mean() for s in scores.values()]))
        if mean > best_mean:
            best_mean, best = mean, {k: v.copy() for k, v in scores.items()}
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"    [{tag}] epoch {epoch:>2}  val r {mean:.4f}  (best {best_mean:.4f})")
    return best


# --------------------------------------------------------------------------- arms


def arm_ridge(corpus, z, subject) -> np.ndarray:
    """Per-subject ridge on the same features -- the baseline every earlier number used."""
    tr_b, tr_i, tr_s = corpus.tensors(z, subject, val=False)
    va_b, va_i, va_s = corpus.tensors(z, subject, val=True)

    def design(bolds, idxs, stims):
        xs = [z[s][i].reshape(len(i), -1) for i, s in zip(idxs, stims, strict=True)]
        return np.concatenate(xs), np.concatenate(bolds)

    x_tr, y_tr = design(tr_b, tr_i, tr_s)
    x_va, y_va = design(va_b, va_i, va_s)
    best = None
    for alpha in ALPHAS:
        r = pearson_per_parcel(Ridge(alpha=alpha).fit(x_tr, y_tr).predict(x_va), y_va)
        if best is None or r.mean() > best.mean():
            best = r
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modalities", default="text,audio,video")
    ap.add_argument("--text-layer", type=int, default=None)
    ap.add_argument("--audio-layer", type=int, default=None)
    ap.add_argument("--video-layer", type=int, default=None)
    ap.add_argument("--arms", default="ridge,separate,joint")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--tag", default="", help="suffix for the output file")
    args = ap.parse_args()

    names = [n.strip() for n in args.modalities.split(",") if n.strip()]
    mods = [F.MODALITIES[n] for n in names]
    arms = {a.strip() for a in args.arms.split(",")}
    cli = {"text": args.text_layer, "audio": args.audio_layer, "video": args.video_layer}
    layers = {}
    for m in mods:
        layer = cli[m.name] if cli[m.name] is not None else m.best_layer
        if layer is None:
            n = F.n_layers(m)
            if n is None:
                raise SystemExit(f"{m.name}: nothing cached at {m.path}")
            layer = n // 2
            print(f"note: {m.name} layer unmeasured, defaulting to the middle ({layer} of {n - 1})")
        layers[m.name] = layer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  modalities={names}  layers={layers}\n")

    t0 = time.time()
    corpus = Corpus(mods, layers)
    print(f"{len(corpus.usable)} stimuli cached in all of {names}")
    proj = corpus.fit_projections()
    z = corpus.project(proj)
    d_in = STIMULUS_WINDOW * N_COMPONENTS * len(mods)
    print(f"design width {d_in}  ({time.time() - t0:.0f}s to build)\n")

    train_b, val_b = {}, {}
    for s in SUBJECTS:
        train_b[s] = Batcher(z, *corpus.tensors(z, s, val=False), device)
        val_b[s] = Batcher(z, *corpus.tensors(z, s, val=True), device)
        print(f"  {s}: train {train_b[s].n:>7} TR   val {val_b[s].n:>6} TR")
    print()

    ceilings = {s: noise_ceiling(corpus.stores[s]) for s in SUBJECTS}
    results: dict[str, dict] = {}

    def record(name: str, scores: dict[str, np.ndarray]) -> None:
        # Parcels below the reliability floor are nan for every subject, so averaging
        # across subjects hits genuinely empty slices. That is the intended behaviour --
        # `per_network` drops them -- so silence the warning rather than the nans.
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        results[name] = {
            "per_subject": {s: float(r.mean()) for s, r in scores.items()},
            "mean": float(np.mean([r.mean() for r in scores.values()])),
            "normalized_mean": float(
                np.nanmean([np.nanmean(ceilings[s].normalize(r)) for s, r in scores.items()])
            ),
            "normalized_per_network": {
                k: v[0]
                for k, v in per_network(
                    np.nanmean(
                        np.stack([ceilings[s].normalize(r) for s, r in scores.items()]), axis=0
                    )
                ).items()
            },
        }
        r = results[name]
        print(
            f"{name:<10} mean r {r['mean']:.4f}   normalized {r['normalized_mean']:.3f}   "
            + " ".join(f"{s.split('-')[1]}:{v:.4f}" for s, v in r["per_subject"].items())
        )

    if "ridge" in arms:
        print("ridge (one model per subject):")
        record("ridge", {s: arm_ridge(corpus, z, s) for s in SUBJECTS})

    if "separate" in arms:
        print("\nseparate (same network, one subject at a time):")
        scores = {}
        for s in SUBJECTS:
            model = JointHead(d_in, [s]).to(device)
            got = train(model, {s: train_b[s]}, {s: val_b[s]}, device, args.epochs, tag=s)
            scores[s] = got[s]
            del model
        record("separate", scores)

    if "joint" in arms:
        print("\njoint (shared trunk, per-subject readout):")
        model = JointHead(d_in, list(SUBJECTS)).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        record("joint", train(model, train_b, val_b, device, args.epochs))
        results["joint"]["params"] = int(n_params)
        print(f"           {n_params / 1e6:.1f} M parameters in the head")

    if "separate" in results and "joint" in results:
        d = results["joint"]["mean"] - results["separate"]["mean"]
        print(f"\njoint training is worth {d:+.4f} mean r over the identical network fit alone")

    # Non-default layers get their own file, so the record for the shipped layers stays.
    custom = "".join(f"-{n}{v}" for n, v in layers.items() if v != F.MODALITIES[n].best_layer)
    tag = args.tag or "-".join(names) + custom
    path = OUT / f"{tag}.json"
    path.write_text(
        json.dumps(
            {
                "modalities": names,
                "layers": layers,
                "n_stimuli": len(corpus.usable),
                "n_components": N_COMPONENTS,
                "design_width": d_in,
                "epochs": args.epochs,
                "arms": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
