"""Measure what weight quantization costs in brain-prediction accuracy.

Answers a question that is usually argued rather than measured: does quantizing the
text backbone damage the representation the brain-encoding head depends on, and how
does that compare to simply not running the top half of the model?

Method -- simulated (fake) quantization. Weights are quantized to N bits and immediately
dequantized back to fp16, so the arithmetic runs in fp16 throughout and only the
numerical content of the weights changes. This isolates the precision effect from kernel
implementation, needs no int4 kernels, and works identically for every backbone. Per-
channel symmetric min/max, which is what a real deployment would use as a floor.

Only layer 8 is extracted and only for the validation episodes, because the head is
frozen from the full-precision fit -- the question is whether a quantized backbone still
produces features the existing head can read.

    uv run --extra extract python scripts/06_quant_sweep.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from cortexlive.align import align_features, trim_fmri
from cortexlive.data import BoldStore, split_by_episode
from cortexlive.score import noise_ceiling, pearson_per_parcel, per_network

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "algonauts"
CACHE = ROOT / "cache" / "text" / "llama-3.2-1b"
OUT = ROOT / "runs" / "quant_sweep"
MODEL_ID = "meta-llama/Llama-3.2-1B"

SUBJECT = "sub-01"
LAYER = 8
N_COMPONENTS = 250
ALPHA = 1e5
VAL_SEASONS = (6,)
BITS = (8, 6, 4, 3, 2)


def fake_quantize_(model: torch.nn.Module, bits: int) -> int:
    """Quantize-dequantize every 2-D weight in place, per output channel. Returns count."""
    qmax = 2 ** (bits - 1) - 1
    n = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                w = module.weight.data.float()
                scale = w.abs().amax(dim=1, keepdim=True) / qmax
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                module.weight.data = (torch.round(w / scale).clamp(-qmax, qmax) * scale).to(
                    module.weight.dtype
                )
                n += 1
    return n


@torch.no_grad()
def extract(model, tokenizer, words_per_tr, device) -> np.ndarray:
    ids_per_tr = [
        tokenizer(" " + " ".join(w), add_special_tokens=False)["input_ids"] if w else []
        for w in words_per_tr
    ]
    flat = [i for ids in ids_per_tr for i in ids]
    hidden = model.config.hidden_size
    feats = np.zeros((len(words_per_tr), hidden), dtype=np.float32)
    if not flat:
        return feats
    out = model(torch.tensor([flat], device=device), output_hidden_states=True)
    states = out.hidden_states[LAYER].squeeze(0).float().cpu().numpy()
    pos = 0
    for t, ids in enumerate(ids_per_tr):
        if ids:
            feats[t] = states[pos : pos + len(ids)].mean(axis=0)
            pos += len(ids)
    return feats


def main() -> None:
    import ast

    import pandas as pd

    from cortexlive.stimulus import transcript_path

    OUT.mkdir(parents=True, exist_ok=True)
    store = BoldStore(DATA, SUBJECT)
    val_groups = [g for g in store.groups() if any(f"-s{s:02d}e" in g for s in VAL_SEASONS)]
    train_runs, val_runs = split_by_episode(store.runs, val_groups=val_groups)

    def cached(stim):
        p = CACHE / f"{stim}.npy"
        return (
            None
            if not p.exists()
            else np.asarray(np.load(p, mmap_mode="r")[:, LAYER, :], np.float32)
        )

    # --- fit the head once, on full-precision features ---
    per_run = [(r, f) for r in train_runs if (f := cached(r.stimulus)) is not None]
    stacked = np.concatenate([f for _, f in per_run])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(stacked), size=min(40_000, len(stacked)), replace=False)
    scaler = StandardScaler().fit(stacked[idx])
    pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(scaler.transform(stacked[idx]))
    del stacked
    xs, ys = [], []
    for run, feats in per_run:
        bold = trim_fmri(store.load(run))
        xs.append(align_features(pca.transform(scaler.transform(feats)), bold.shape[0]))
        ys.append(bold)
    head = Ridge(alpha=ALPHA).fit(np.concatenate(xs), np.concatenate(ys))
    del xs, ys

    val = [(r, f) for r in val_runs if (f := cached(r.stimulus)) is not None]
    nc = noise_ceiling(store)

    def score(feats_by_stim: dict[str, np.ndarray]) -> np.ndarray:
        px, py = [], []
        for run, _ in val:
            f = feats_by_stim.get(run.stimulus)
            if f is None:
                continue
            bold = trim_fmri(store.load(run))
            px.append(
                head.predict(align_features(pca.transform(scaler.transform(f)), bold.shape[0]))
            )
            py.append(bold)
        return pearson_per_parcel(np.concatenate(px), np.concatenate(py))

    baseline = score({r.stimulus: f for r, f in val})
    print(f"fp16 baseline (layer {LAYER}): r = {baseline.mean():+.4f}\n")

    # --- reference points from the layer sweep, for comparison ---
    sweep = json.load(open(ROOT / "runs" / "text_baseline" / "result.json"))["layer_sweep"]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    words = {}
    for run, _ in val:
        p = transcript_path(DATA, run.source, run.stimulus)
        if p is None:
            continue
        df = pd.read_csv(p, sep="\t")
        rows = []
        for cell in df["words_per_tr"]:
            try:
                w = ast.literal_eval(cell) if isinstance(cell, str) and cell.strip() else []
            except (ValueError, SyntaxError):
                w = []
            rows.append([str(x) for x in w] if isinstance(w, list) else [])
        words[run.stimulus] = rows

    results = {"fp16": float(baseline.mean())}
    per_net = {"fp16": {k: v[0] for k, v in per_network(nc.normalize(baseline)).items()}}

    print(f"{'precision':>10}{'mean r':>10}{'vs fp16':>10}{'retained':>10}")
    print(f"{'fp16':>10}{baseline.mean():>10.4f}{0.0:>+10.4f}{100.0:>9.1f}%")

    for bits in BITS:
        model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
        n = fake_quantize_(model, bits)
        feats = {s: extract(model, tokenizer, w, device) for s, w in words.items()}
        r = score(feats)
        results[f"int{bits}"] = float(r.mean())
        per_net[f"int{bits}"] = {k: v[0] for k, v in per_network(nc.normalize(r)).items()}
        print(
            f"{'int' + str(bits):>10}{r.mean():>10.4f}{r.mean() - baseline.mean():>+10.4f}"
            f"{100 * r.mean() / baseline.mean():>9.1f}%"
        )
        del model
        if device == "mps":
            torch.mps.empty_cache()

    payload = {
        "note": f"{n} Linear layers fake-quantized per run; layer {LAYER}; "
        "head frozen from the fp16 fit",
        "subject": SUBJECT,
        "layer": LAYER,
        "mean_r": results,
        "normalized_per_network": per_net,
        "layer_sweep_reference": sweep,
    }
    (OUT / "result.json").write_text(json.dumps(payload, indent=2))

    print("\nfor comparison, from the layer sweep (fp16 throughout):")
    pct16 = 100 * sweep["16"] / baseline.mean()
    print(f"  layer 16 (full depth) : {sweep['16']:.4f}  ({pct16:.1f}% of layer 8)")
    print(f"  layer 4               : {sweep['4']:.4f}")
    print(f"  layer 0 (embeddings)  : {sweep['0']:.4f}")
    print(f"\nwrote {OUT / 'result.json'}")


if __name__ == "__main__":
    main()
