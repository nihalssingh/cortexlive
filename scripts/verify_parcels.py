"""Verify that nilearn's Schaefer parcel order matches the dataset's .h5 column order.

This is the check the whole project's spatial story rests on. `atlas/schaefer1000.npz`
labels parcel i with a Yeo-7 network; every per-network number we report assumes
column i of the .h5 arrays is that same parcel. Both sides are Schaefer 2018
1000-parcel / 7-network, so they *should* agree -- but if they don't, nothing looks
broken, the code runs fine, and every cortical map is quietly wrong.

The dataset ships a subject parcellation volume where each voxel carries a parcel ID
in 1..1000, which gives us ground truth to check against. Two tests:

  1. HEMISPHERE. nilearn says parcel i is LH or RH. In MNI world coordinates, left is
     x < 0. Compute each parcel's centroid from the volume and compare. An ordering
     scrambled across hemispheres cannot agree by luck; one shifted by a few parcels
     still passes the 99% bar (only parcels at the LH/RH boundary change side), as does
     one reordered within a hemisphere.

  2. ANATOMY. Visual parcels should sit posterior to Default Mode parcels (lower y), and
     Somatomotor parcels above z = 0. A weaker check, but it sees y and z, so it catches
     an anterior/posterior or up/down flip that test 1 (x only) cannot.

Run once the dataset is downloaded (docs/data.md):
    uv run scripts/verify_parcels.py
"""

from __future__ import annotations

import pathlib
import sys

import nibabel as nib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
ATLAS_NPZ = ROOT / "atlas" / "schaefer1000.npz"
DATA = ROOT / "data" / "algonauts"


def parcel_centroids(nii_path: pathlib.Path) -> np.ndarray:
    """Return (1000, 3) centroids in world (MNI) coordinates, indexed by parcel ID - 1."""
    img = nib.load(nii_path)
    vol = np.asarray(img.dataobj)
    affine = img.affine

    ids = np.arange(1, 1001)
    centroids = np.full((1000, 3), np.nan)
    # np.nonzero once, then bincount per axis -- far faster than 1000 boolean masks.
    idx = np.nonzero(vol)
    labels = vol[idx].astype(np.int64)
    voxel_coords = np.stack(idx, axis=1).astype(np.float64)

    counts = np.bincount(labels, minlength=1001).astype(np.float64)
    for axis in range(3):
        sums = np.bincount(labels, weights=voxel_coords[:, axis], minlength=1001)
        with np.errstate(invalid="ignore", divide="ignore"):
            centroids[:, axis] = (sums / counts)[ids]

    # voxel -> world
    ones = np.ones((1000, 1))
    world = (affine @ np.hstack([centroids, ones]).T).T[:, :3]
    return world


def main() -> int:
    if not ATLAS_NPZ.exists():
        sys.exit(f"missing {ATLAS_NPZ} -- run scripts/fetch_atlas.py first")

    atlas = np.load(ATLAS_NPZ, allow_pickle=False)
    hemi = atlas["hemisphere"]
    network = atlas["network"]

    nii = sorted(DATA.glob("fmri/*/atlas/*dseg_parcellation.nii.gz"))
    if not nii:
        sys.exit(f"no parcellation volume under {DATA}/fmri/*/atlas/ -- "
                 "download the dataset first (docs/data.md)")

    print(f"using {nii[0].relative_to(ROOT)}\n")
    world = parcel_centroids(nii[0])

    missing = np.isnan(world[:, 0])
    if missing.any():
        print(f"WARNING: {missing.sum()} parcel IDs absent from the volume")

    ok = ~missing

    # --- Test 1: hemisphere ---
    # MNI convention: x < 0 is the left hemisphere.
    vol_is_left = world[:, 0] < 0
    nil_is_left = hemi == "LH"
    agree = vol_is_left[ok] == nil_is_left[ok]
    pct = 100 * agree.mean()

    print("TEST 1  hemisphere (nilearn LH/RH vs centroid x-sign)")
    print(f"        {agree.sum()}/{ok.sum()} agree ({pct:.1f}%)")

    if not agree.all():
        bad = np.where(ok)[0][~agree][:10]
        print(f"        first disagreements at parcel index: {bad.tolist()}")

    # --- Test 2: gross anatomy ---
    vis_y = np.nanmean(world[(network == "Vis") & ok, 1])
    dmn_y = np.nanmean(world[(network == "Default") & ok, 1])
    som_z = np.nanmean(world[(network == "SomMot") & ok, 2])

    print("\nTEST 2  anatomy")
    print(f"        Visual mean y      = {vis_y:+7.1f} mm  (should be most posterior)")
    print(f"        Default mean y     = {dmn_y:+7.1f} mm")
    print(f"        Somatomotor mean z = {som_z:+7.1f} mm  (should be superior, z > 0)")

    anatomy_ok = vis_y < dmn_y and som_z > 0

    print()
    if pct > 99.0 and anatomy_ok:
        print("PASS -- hemispheres and gross anatomy match the dataset's parcel IDs.")
        return 0
    print("FAIL -- do NOT trust any per-network result until this is resolved.")
    print("       Check whether the .h5 columns are ordered by parcel ID 1..1000,")
    print("       and whether nilearn's label list still starts with 'Background'.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
