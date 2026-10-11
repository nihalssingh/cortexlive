"""Build the Schaefer-1000 parcel -> Yeo-7 network mapping.

The Algonauts fMRI arrays are (n_samples, 1000) with no labels attached, so every
per-network breakdown in this project depends on this file. Writes
atlas/schaefer1000.npz with, for each of the 1000 parcels: its full label, its
hemisphere, and its Yeo-7 network.

Needs no dataset access -- nilearn downloads the atlas itself.

This assumes nilearn's parcel ordering matches the column ordering of the Algonauts .h5
arrays (both are Schaefer 2018 1000-parcel / 7-network); scripts/verify_parcels.py
checks it against the dataset.

    uv run scripts/fetch_atlas.py
"""

from __future__ import annotations

import pathlib

import numpy as np
from nilearn import datasets

OUT = pathlib.Path(__file__).resolve().parent.parent / "atlas" / "schaefer1000.npz"

# Yeo-7 order as it appears in Schaefer label strings.
NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

# Names to show in the visualizer. "Limbic" here is CORTICAL limbic -- orbitofrontal
# and temporal pole. It is not the amygdala or any subcortical structure; Schaefer is
# a cortex-only parcellation. Labelling it "emotion" in a UI would be wrong.
DISPLAY = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal Attention",
    "SalVentAttn": "Salience / Ventral Attention",
    "Limbic": "Limbic (cortical)",
    "Cont": "Frontoparietal Control",
    "Default": "Default Mode",
}


def parse_label(label: str) -> tuple[str, str]:
    """'7Networks_LH_Vis_1' -> ('LH', 'Vis')."""
    parts = label.split("_")
    if len(parts) < 3:
        raise ValueError(f"unexpected Schaefer label format: {label!r}")
    hemi, network = parts[1], parts[2]
    if hemi not in ("LH", "RH"):
        raise ValueError(f"unexpected hemisphere {hemi!r} in {label!r}")
    if network not in NETWORKS:
        raise ValueError(f"unexpected network {network!r} in {label!r}")
    return hemi, network


def main() -> None:
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7, resolution_mm=1)

    labels = [x.decode() if isinstance(x, bytes) else str(x) for x in atlas.labels]

    # nilearn >=0.12 prepends a "Background" entry for the 0 value in the label image.
    # Strip it, so index i here is parcel i of the 1000 -- which is what the .h5 columns
    # are. Getting this wrong shifts every parcel by one and silently scrambles every
    # network breakdown, so assert rather than slice blindly.
    if labels and labels[0] == "Background":
        labels = labels[1:]
    if len(labels) != 1000:
        raise SystemExit(f"expected 1000 parcel labels, got {len(labels)}")

    hemis, networks = zip(*(parse_label(x) for x in labels), strict=True)
    net_idx = np.array([NETWORKS.index(n) for n in networks], dtype=np.int16)

    np.savez_compressed(
        OUT,
        labels=np.array(labels),
        hemisphere=np.array(hemis),
        network=np.array(networks),
        network_index=net_idx,
        network_names=np.array(NETWORKS),
        network_display=np.array([DISPLAY[n] for n in NETWORKS]),
    )

    print(f"wrote {OUT}")
    print(f"{'network':<32} {'parcels':>7}  {'LH':>4} {'RH':>4}")
    for i, n in enumerate(NETWORKS):
        sel = net_idx == i
        lh = sum(1 for h, s in zip(hemis, sel, strict=True) if s and h == "LH")
        print(f"{DISPLAY[n]:<32} {sel.sum():>7}  {lh:>4} {sel.sum() - lh:>4}")
    print(f"{'TOTAL':<32} {len(labels):>7}")


if __name__ == "__main__":
    main()
