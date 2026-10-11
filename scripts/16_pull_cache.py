"""Pull the feature cache down from the Modal volume, resumably.

`modal volume get` on a whole directory is the obvious way to do this and the wrong one at
this size: it can stall partway with the process alive and no bytes moving, and its only
recovery is `--force`, which restarts the entire directory and throws away everything
already fetched.

So: fetch one file at a time, skip whatever is already present AND parseable, and retry.
`F.readable` is the completeness test rather than mere existence, because a file
interrupted mid-write exists at the right path with the wrong contents. That distinction
is the whole reason a naive `if path.exists(): continue` would silently keep a truncated
array and only fail much later, during a sweep, as an unexplained accuracy drop.

    uv run --extra remote python scripts/16_pull_cache.py --what video audio
"""

from __future__ import annotations

import argparse
import pathlib
import time

from cortexlive import features as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
VOLUME = "cortexlive-cache"


def volume():
    import modal

    return modal.Volume.from_name(VOLUME)


def remote_files(remote_dir: str) -> list[str]:
    return sorted(
        e.path.split("/")[-1] for e in volume().iterdir(remote_dir) if e.path.endswith(".npy")
    )


def local_complete(local_dir: pathlib.Path) -> set[str]:
    """Files that are present AND fully parseable -- not merely present."""
    if not local_dir.exists():
        return set()
    return {p.name for p in local_dir.glob("*.npy") if F.readable(p) is not None}


def fetch(vol, remote_dir: str, name: str, local_dir: pathlib.Path) -> bool:
    """Stream one file to a .part, then rename. The rename is the commit.

    Writing straight to the final path leaves a truncated-but-plausible array when a
    download stalls. A temp file plus atomic rename means an interrupted fetch leaves no
    file at all, so "present" and "complete" mean the same thing.
    """
    dest = local_dir / name
    tmp = local_dir / f"{name}.part"
    try:
        with open(tmp, "wb") as fh:
            for chunk in vol.read_file(f"{remote_dir}/{name}"):
                fh.write(chunk)
        if F.readable(tmp) is None:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False


def pull(remote_dir: str, local_dir: pathlib.Path, retries: int) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for stale in local_dir.glob("*.part"):
        stale.unlink()
    want = remote_files(remote_dir)
    if not want:
        print(f"{remote_dir}: nothing listed remotely")
        return
    have = local_complete(local_dir)
    # Anything present but unparseable is a casualty of an earlier interrupted pull.
    broken = {p.name for p in local_dir.glob("*.npy")} - have
    for name in broken:
        (local_dir / name).unlink()
    todo = [n for n in want if n not in have]
    print(
        f"{remote_dir}: {len(want)} remote, {len(have)} complete locally, "
        f"{len(broken)} truncated (deleted), {len(todo)} to fetch"
    )

    vol = volume()
    t0, done, failed = time.time(), 0, []
    for i, name in enumerate(todo, 1):
        ok = False
        for attempt in range(retries):
            ok = fetch(vol, remote_dir, name, local_dir)
            if ok:
                break
            print(f"  retry {attempt + 1}/{retries}: {name}")
        if ok:
            done += 1
        else:
            failed.append(name)
        if i % 20 == 0 or i == len(todo):
            rate = done / max(1e-9, time.time() - t0) * 60
            gb = sum(f.stat().st_size for f in local_dir.glob("*.npy")) / 1e9
            eta = (len(todo) - i) / rate if rate else 0
            print(f"  [{i}/{len(todo)}] {rate:.1f}/min  {gb:.1f} GB  eta {eta:.0f} min")

    print(f"{remote_dir}: fetched {done}, failed {len(failed)}")
    if failed:
        print(f"  failed: {failed[:10]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="+", default=["video", "audio"])
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    targets = {
        "video": (f"/video/{F.VIDEO.subdir}", F.VIDEO.path),
        "audio": (f"/audio/{F.AUDIO.subdir}", F.AUDIO.path),
    }
    for key in args.what:
        if key.startswith("res"):  # e.g. res192: the 192 px video features
            crop = key[3:]
            sub = f"{F.VIDEO.subdir}_res{crop}v2"
            targets[key] = (f"/video/{sub}", F.CACHE / "video" / sub)
        remote, local = targets[key]
        pull(remote, local, args.retries)
        print()

    print(F.summary())


if __name__ == "__main__":
    main()
