"""Extract per-TR contextualized text features from Llama-3.2-1B.

Writes cache/text/<model>/<stimulus>.npy, each (n_tr, n_layers, hidden) fp16.

ALL layers are cached, deliberately. Storing only a pre-averaged subset -- the way
TRIBE's layer_aggregation does -- would make the later layer-truncation experiment
impossible, and that experiment is where the real-time speedup comes from. All 17
hidden states of Llama-3.2-1B across Friends s1-6 is only ~10 GB.

Whole episodes go through in a single forward pass: an episode is a few thousand
tokens and Llama-3.2 has a 128k context, so no sliding window is needed and every
token sees its full preceding context.

    uv run --extra extract python scripts/03_extract_text.py [--limit N]
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from cortexlive.data import SUBJECTS, BoldStore
from cortexlive.stimulus import transcript_path

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "algonauts"
MODEL_ID = "meta-llama/Llama-3.2-1B"
OUT = ROOT / "cache" / "text" / MODEL_ID.split("/")[-1].lower()


def tr_words(path: pathlib.Path) -> list[list[str]]:
    """Per-TR word lists, in movie order."""
    df = pd.read_csv(path, sep="\t")
    out = []
    for cell in df["words_per_tr"]:
        try:
            words = ast.literal_eval(cell) if isinstance(cell, str) and cell.strip() else []
        except (ValueError, SyntaxError):
            words = []
        out.append([str(w) for w in words] if isinstance(words, list) else [])
    return out


@torch.no_grad()
def extract_episode(model, tokenizer, words_per_tr: list[list[str]], device: str) -> np.ndarray:
    """Return (n_tr, n_layers, hidden) fp16 -- mean hidden state over each TR's tokens."""
    ids_per_tr: list[list[int]] = []
    for words in words_per_tr:
        if not words:
            ids_per_tr.append([])
            continue
        # Leading space matters for BPE: " word" and "word" are different tokens.
        text = " " + " ".join(words)
        ids_per_tr.append(tokenizer(text, add_special_tokens=False)["input_ids"])

    flat = [i for ids in ids_per_tr for i in ids]
    n_layers = model.config.num_hidden_layers + 1  # + embedding output
    hidden = model.config.hidden_size

    if not flat:
        return np.zeros((len(words_per_tr), n_layers, hidden), dtype=np.float16)

    input_ids = torch.tensor([flat], device=device)
    out = model(input_ids, output_hidden_states=True)
    # (n_layers, n_tokens, hidden)
    states = torch.stack(out.hidden_states, dim=0).squeeze(1).to(torch.float16).cpu().numpy()

    feats = np.zeros((len(words_per_tr), n_layers, hidden), dtype=np.float16)
    pos = 0
    for t, ids in enumerate(ids_per_tr):
        if ids:
            feats[t] = states[:, pos : pos + len(ids), :].mean(axis=1)
            pos += len(ids)
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only process N stimuli")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}  model={MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    print(f"layers={model.config.num_hidden_layers + 1}  hidden={model.config.hidden_size}")

    # Union of stimuli across subjects, so the cache serves every subject.
    stimuli: dict[str, str] = {}
    for subject in SUBJECTS:
        for run in BoldStore(DATA, subject).runs:
            stimuli.setdefault(run.stimulus, run.source)

    todo = sorted(stimuli.items())
    if args.limit:
        todo = todo[: args.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    t0, done, skipped = time.time(), 0, 0

    for stim, source in todo:
        dest = OUT / f"{stim}.npy"
        if dest.exists():
            skipped += 1
            continue
        path = transcript_path(DATA, source, stim)
        if path is None:
            continue
        feats = extract_episode(model, tokenizer, tr_words(path), device)
        np.save(dest, feats)
        done += 1
        if done % 20 == 0:
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(todo)}  {rate:.1f} stim/s  last={stim} {feats.shape}")

    dt = time.time() - t0
    total_gb = sum(f.stat().st_size for f in OUT.glob("*.npy")) / 1e9
    print(f"\ndone: {done} extracted, {skipped} already cached, {dt / 60:.1f} min")
    print(f"cache: {OUT}  ({total_gb:.1f} GB)")


if __name__ == "__main__":
    main()
