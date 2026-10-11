"""Project the 1000 Schaefer parcels onto an fsaverage cortical surface.

Produces atlas/surface.npz: for each hemisphere, a per-vertex parcel index (-1 outside
cortex) plus the vertex coordinates and triangle faces. Every rendering approach needs
this -- a flat 2D map, an inflated 3D mesh, or a WebGL viewer -- so it is built once,
independent of the aesthetic decision.

With this, drawing a frame is a lookup, not a computation: given 1000 predicted values,
`values[vertex_parcel]` colours the whole mesh. That matters for the live version, where
a frame must render every 1.49 s alongside three streaming backbones.

Note on space: the shipped parcellation is MNI152NLin2009cAsym while fsaverage maps
assume MNI152. The mismatch is small and irrelevant for visualisation, but it would
matter if these vertices were ever used for quantitative analysis. They are not -- all
numbers in this project come from the parcel time series directly.

    uv run scripts/fetch_surface.py
"""

from __future__ import annotations

import pathlib

import numpy as np
from nilearn import datasets, surface

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "algonauts"
OUT = ROOT / "atlas" / "surface.npz"


def main() -> None:
    nii = sorted(DATA.glob("fmri/*/atlas/*dseg_parcellation.nii.gz"))
    if not nii:
        raise SystemExit(f"no parcellation volume under {DATA}/fmri/*/atlas/")
    print(f"projecting {nii[0].name}")

    atlas_npz = ROOT / "atlas" / "schaefer1000.npz"
    if not atlas_npz.exists():
        raise SystemExit(f"{atlas_npz} missing -- run scripts/fetch_atlas.py first")
    hemisphere = np.load(atlas_npz, allow_pickle=False)["hemisphere"]

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    arrays: dict[str, np.ndarray] = {}

    for hemi, mesh_key in (("left", "pial_left"), ("right", "pial_right")):
        coords, faces = surface.load_surf_mesh(fsaverage[mesh_key])
        # Parcel IDs are categorical, so linear interpolation would invent parcels that
        # do not exist at boundaries. nilearn >=0.13 calls this mode
        # "nearest_most_frequent" -- a majority vote among nearby voxels.
        texture = surface.vol_to_surf(
            str(nii[0]),
            fsaverage[mesh_key],
            interpolation="nearest_most_frequent",
            radius=1.0,
        )
        parcel_id = np.nan_to_num(texture, nan=0.0).astype(np.int32)
        # Volume IDs are 1..1000; store 0-based with -1 for "not cortex".
        vertex_parcel = np.where(parcel_id > 0, parcel_id - 1, -1).astype(np.int16)

        # Schaefer parcels are strictly unilateral, but sampling a volume onto a surface
        # bleeds a little across the midline, which assigns a handful of LH parcels to
        # RH vertices and vice versa. Discard those -- on screen they would paint a
        # patch of one hemisphere with the other hemisphere's predicted signal.
        want = "LH" if hemi == "left" else "RH"
        wrong_side = (vertex_parcel >= 0) & (hemisphere[np.clip(vertex_parcel, 0, None)] != want)
        if wrong_side.any():
            print(f"  {hemi:<5} dropped {int(wrong_side.sum())} midline-bleed vertices")
        vertex_parcel[wrong_side] = -1

        arrays[f"{hemi}_vertex_parcel"] = vertex_parcel
        arrays[f"{hemi}_coords"] = coords.astype(np.float32)
        arrays[f"{hemi}_faces"] = faces.astype(np.int32)

        covered = int((vertex_parcel >= 0).sum())
        n_parcels = len(np.unique(vertex_parcel[vertex_parcel >= 0]))
        print(
            f"  {hemi:<5} vertices={len(vertex_parcel):>6}  "
            f"assigned={covered:>6} ({100 * covered / len(vertex_parcel):.0f}%)  "
            f"parcels touched={n_parcels}"
        )

    np.savez_compressed(OUT, **arrays)

    total = set()
    for hemi in ("left", "right"):
        vp = arrays[f"{hemi}_vertex_parcel"]
        total |= set(np.unique(vp[vp >= 0]).tolist())
    print(f"\nwrote {OUT}")
    print(f"parcels represented on the surface: {len(total)}/1000")
    if len(total) < 1000:
        missing = sorted(set(range(1000)) - total)
        print(f"  unrepresented (too small at this mesh resolution): {missing[:12]}...")


if __name__ == "__main__":
    main()
