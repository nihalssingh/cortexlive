"""Pre-rasterize the inflated cortical surface into 4 canonical views.

Writes atlas/views.npz holding, per view, a per-pixel parcel index and a per-pixel
Lambertian shade (parcel -1 = background, -2 = the medial wall, which carries no parcel
but is drawn as surface). Rendering a frame then costs one lookup per pixel:

    rgb = colormap(values[parcel_map]) * shade_map

No WebGL, no mesh traversal, no external library. The geometry is fixed -- the standard
four-panel neuroimaging layout doesn't rotate -- so all the 3D work can happen once,
here, offline, which keeps drawing cheap while the backbones run live.

Views, laid out as the field conventionally prints them:

    [ LH lateral ]  [ RH lateral ]
    [ LH medial  ]  [ RH medial  ]

    uv run scripts/build_views.py
"""

from __future__ import annotations

import pathlib

import numpy as np
from nilearn import datasets, surface

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "atlas" / "views.npz"
SURF = ROOT / "atlas" / "surface.npz"

RES = 420  # pixels per view
WALL = -2  # parcel code for the medial wall
MARGIN = 0.04

# (name, hemisphere, camera sign along x). +1 means the camera sits at +x looking in -x,
# so the visible surface is the one with the largest x.
VIEWS = (
    ("lh_lateral", "left", -1),
    ("rh_lateral", "right", +1),
    ("lh_medial", "left", +1),
    ("rh_medial", "right", -1),
)


def project(coords: np.ndarray, cam_sign: int) -> tuple[np.ndarray, np.ndarray]:
    """World -> (screen_xy in [0,1]^2, depth) for an orthographic camera on the x axis."""
    # Screen horizontal is the y (anterior-posterior) axis. With camera forward = cam_sign
    # along x and up = +z, the camera's right-hand direction works out to cam_sign * y.
    # Getting this sign wrong mirrors every panel: a left hemisphere viewed from the left
    # would show the frontal pole on the right, which reads as anatomically plausible
    # while being backwards.
    sx = coords[:, 1] * cam_sign
    sy = coords[:, 2]
    depth = coords[:, 0] * cam_sign  # larger = nearer the camera
    return np.stack([sx, sy], axis=1), depth


def rasterize(
    coords: np.ndarray,
    faces: np.ndarray,
    vertex_parcel: np.ndarray,
    cam_sign: int,
    res: int = RES,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffered triangle rasterisation. Returns (parcel_map int16, shade_map float32)."""
    xy, depth = project(coords, cam_sign)

    lo, hi = xy.min(0), xy.max(0)
    span = (hi - lo).max()
    pad = span * MARGIN
    lo, span = lo - pad, span + 2 * pad
    # Preserve aspect: centre the smaller axis inside the square canvas.
    centre = (hi + lo) / 2
    lo = centre - span / 2
    px = (xy - lo) / span * (res - 1)

    # Flip vertical: image row 0 is the top of the picture.
    px[:, 1] = (res - 1) - px[:, 1]

    # Face normals in world space, for shading.
    v0, v1, v2 = (coords[faces[:, i]] for i in range(3))
    normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norm_len == 0, 1, norm_len)

    # Light slightly above and in front of the camera, so gyri and sulci read as relief.
    light = np.array([cam_sign * 0.85, 0.25, 0.45])
    light = light / np.linalg.norm(light)
    face_shade = np.clip(np.abs(normals @ light), 0, 1) * 0.75 + 0.25

    # A face takes the parcel of its first labelled vertex; unlabelled faces are the
    # medial wall, drawn as surface so the medial views read as a brain, not a hole.
    fp = vertex_parcel[faces]
    face_parcel = np.max(fp, axis=1)
    face_parcel = np.where(face_parcel < 0, WALL, face_parcel)

    zbuf = np.full((res, res), -np.inf, dtype=np.float64)
    parcel_map = np.full((res, res), -1, dtype=np.int16)
    shade_map = np.zeros((res, res), dtype=np.float32)

    fx, fy = px[faces][:, :, 0], px[faces][:, :, 1]
    fz = depth[faces]

    # Cull back faces: only the hemisphere facing the camera is visible.
    facing = normals[:, 0] * cam_sign > 0

    x_min = np.floor(fx.min(1)).astype(int)
    x_max = np.ceil(fx.max(1)).astype(int)
    y_min = np.floor(fy.min(1)).astype(int)
    y_max = np.ceil(fy.max(1)).astype(int)

    for f in range(faces.shape[0]):
        if not facing[f]:
            continue
        x0, x1 = max(x_min[f], 0), min(x_max[f], res - 1)
        y0, y1 = max(y_min[f], 0), min(y_max[f], res - 1)
        if x0 > x1 or y0 > y1:
            continue

        ax, bx, cx = fx[f]
        ay, by, cy = fy[f]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-12:
            continue

        ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        w0 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / det
        w1 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / det
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z = w0 * fz[f, 0] + w1 * fz[f, 1] + w2 * fz[f, 2]
        sub = zbuf[y0 : y1 + 1, x0 : x1 + 1]
        win = inside & (z > sub)
        if not win.any():
            continue
        sub[win] = z[win]
        parcel_map[y0 : y1 + 1, x0 : x1 + 1][win] = face_parcel[f]
        shade_map[y0 : y1 + 1, x0 : x1 + 1][win] = face_shade[f]

    fill_speckles(parcel_map, shade_map)
    return parcel_map, shade_map


def fill_speckles(parcel_map: np.ndarray, shade_map: np.ndarray, passes: int = 2) -> None:
    """Fill pin-holes left by unlabelled vertices, in place.

    A handful of surface vertices carry no parcel, which rasterises to specks scattered
    over the cortex. They read as data ("nothing here") when they are just sampling
    noise. Only pixels almost entirely surrounded by cortex are filled, so the medial
    wall and the background are untouched.
    """
    res = parcel_map.shape[0]
    for _ in range(passes):
        holes = np.argwhere(parcel_map < 0)
        if not len(holes):
            return
        filled = 0
        for y, x in holes:
            y0, y1 = max(y - 1, 0), min(y + 2, res)
            x0, x1 = max(x - 1, 0), min(x + 2, res)
            block = parcel_map[y0:y1, x0:x1]
            neigh = block[block >= 0]
            if neigh.size < 6:  # not enclosed by cortex -> genuine background
                continue
            vals, counts = np.unique(neigh, return_counts=True)
            parcel_map[y, x] = vals[np.argmax(counts)]
            sblock = shade_map[y0:y1, x0:x1]
            shade_map[y, x] = sblock[block >= 0].mean()
            filled += 1
        if not filled:
            return


def main() -> None:
    if not SURF.exists():
        raise SystemExit(f"{SURF} missing -- run scripts/fetch_surface.py first")
    surf = np.load(SURF, allow_pickle=False)

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    inflated = {
        "left": surface.load_surf_mesh(fsaverage["infl_left"]),
        "right": surface.load_surf_mesh(fsaverage["infl_right"]),
    }

    out: dict[str, np.ndarray] = {}
    names = []
    for name, hemi, cam in VIEWS:
        coords, faces = inflated[hemi]
        vp = surf[f"{hemi}_vertex_parcel"]
        pmap, smap = rasterize(np.asarray(coords), np.asarray(faces), vp, cam)
        out[f"{name}_parcel"] = pmap
        out[f"{name}_shade"] = smap
        names.append(name)
        cover = int((pmap >= 0).sum())
        print(
            f"  {name:<11} {cover:>7} px cortex ({100 * cover / pmap.size:.0f}%)  "
            f"parcels visible={len(np.unique(pmap[pmap >= 0]))}"
        )

    out["view_names"] = np.array(names)
    out["res"] = np.array([RES])
    np.savez_compressed(OUT, **out)
    print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
