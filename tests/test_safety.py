"""One model process at a time, clear errors, and license notices that travel with the model."""

from __future__ import annotations

import hashlib
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

from cortexlive import cli
from cortexlive.viewer import html

HOLD = """import sys, time
from pathlib import Path
from cortexlive.stream import acquire_model_lock
acquire_model_lock(Path(sys.argv[1]))
print("held", flush=True)
time.sleep(60)
"""
TRY = """import sys
from pathlib import Path
from cortexlive.stream import ModelBusy, acquire_model_lock
try:
    acquire_model_lock(Path(sys.argv[1]))
    print("acquired")
except ModelBusy as e:
    print("busy:", e)
"""


def _try(lock: Path) -> str:
    return subprocess.run([sys.executable, "-c", TRY, str(lock)], capture_output=True,
                          text=True, timeout=60).stdout.strip()


def test_second_model_process_is_refused_until_the_first_exits(tmp_path):
    lock = tmp_path / "model.lock"
    holder = subprocess.Popen([sys.executable, "-c", HOLD, str(lock)], stdout=subprocess.PIPE,
                              text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        out = _try(lock)
        assert out.startswith("busy:") and f"pid {holder.pid}" in out
    finally:
        holder.kill()
        holder.wait()
    assert _try(lock) == "acquired"


def test_viewer_on_a_taken_port_explains_itself(tmp_path, capsys):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        code = cli.main(["--captures", str(tmp_path), "view", "--no-browser", "--port", str(port)])
    assert code == 2
    err = capsys.readouterr().err
    assert f"port {port} is already in use" in err and f"--port {port + 1}" in err


def test_bundle_is_named_for_llama_and_carries_its_notices(tmp_path, monkeypatch):
    for name in cli.BUNDLE + cli.BUNDLE_NOTICES:
        f = tmp_path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(name)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert cli.main(["bundle"]) == 0
    (out,) = (tmp_path / "dist").glob("*.tar.gz")
    assert out.name.startswith("Llama-cortexlive-")
    with tarfile.open(out) as tar:
        names = set(tar.getnames())
    assert set(cli.BUNDLE) | set(cli.BUNDLE_NOTICES) <= names

    (tmp_path / cli.BUNDLE_NOTICES[1]).unlink()
    assert cli.main(["bundle", "-o", str(tmp_path / "x.tar.gz")]) == 1


PUBLISHED = "Llama-cortexlive-v0.1.0.tar.gz"


def _bundle(tmp_path, monkeypatch, name):
    """Pack stand-in model files with `cortexlive bundle` as `name`. Returns the bundle and an
    empty checkout that ROOT and MODELS now point at, so nothing touches this one."""
    src, checkout = tmp_path / "src", tmp_path / "checkout"
    for f in cli.BUNDLE + cli.BUNDLE_NOTICES:
        (src / f).parent.mkdir(parents=True, exist_ok=True)
        (src / f).write_text(f)
    monkeypatch.setattr(cli, "ROOT", src)
    out = tmp_path / "downloads" / name
    assert cli.main(["bundle", "-o", str(out)]) == 0
    checkout.mkdir()
    monkeypatch.setattr(cli, "ROOT", checkout)
    monkeypatch.setattr(cli, "MODELS", checkout / "models")
    return out, checkout


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_setup_refuses_a_published_name_with_another_digest(tmp_path, monkeypatch, capsys):
    bundle, checkout = _bundle(tmp_path, monkeypatch, PUBLISHED)  # a rebuild: another digest
    capsys.readouterr()
    assert cli._install_bundle(str(bundle)) == 1
    err = capsys.readouterr().err
    assert "nothing was installed" in err and cli.RELEASES[PUBLISHED] in err
    assert _files(checkout) == [] and bundle.exists()  # nothing unpacked, the copy kept


def test_a_download_that_fails_the_check_leaves_nothing(tmp_path, monkeypatch, capsys):
    import urllib.request

    bundle, checkout = _bundle(tmp_path, monkeypatch, PUBLISHED)
    monkeypatch.setattr(urllib.request, "urlretrieve",  # no network: "download" the rebuild
                        lambda url, filename: shutil.copyfile(bundle, filename))
    url = f"https://github.com/nihalssingh/cortexlive/releases/download/v0.1.0/{PUBLISHED}"
    assert cli._install_bundle(url) == 1
    assert "nothing was installed" in capsys.readouterr().err
    assert _files(checkout) == []  # neither the download nor a model file


def test_setup_verifies_a_published_bundle_even_renamed(tmp_path, monkeypatch, capsys):
    bundle, checkout = _bundle(tmp_path, monkeypatch, PUBLISHED)
    monkeypatch.setitem(cli.RELEASES, PUBLISHED, hashlib.sha256(bundle.read_bytes()).hexdigest())
    capsys.readouterr()
    assert cli._install_bundle(str(bundle)) == 0
    assert f"(verified: {PUBLISHED})" in capsys.readouterr().out
    assert _files(checkout) == sorted(cli.BUNDLE)
    renamed = bundle.rename(bundle.with_name("model (1).tar.gz"))  # as a browser might
    assert cli._install_bundle(str(renamed)) == 0
    assert f"(verified: {PUBLISHED})" in capsys.readouterr().out


def test_setup_installs_a_bundle_built_locally(tmp_path, monkeypatch, capsys):
    bundle, checkout = _bundle(tmp_path, monkeypatch, "Llama-cortexlive-v0.1.0-1-gabc1234.tar.gz")
    capsys.readouterr()
    assert cli._install_bundle(str(bundle)) == 0
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    assert f"bundle sha256 {digest} (not a bundle this version knows" in capsys.readouterr().out
    assert _files(checkout) == sorted(cli.BUNDLE)


def test_setup_explains_a_bundle_cut_short(tmp_path, monkeypatch, capsys):
    bundle, checkout = _bundle(tmp_path, monkeypatch, "bundle.tar.gz")
    data = bundle.read_bytes()
    bundle.write_bytes(data[: len(data) // 2])  # an interrupted download
    capsys.readouterr()
    assert cli._install_bundle(str(bundle)) == 1
    assert "could not be read" in capsys.readouterr().err
    assert _files(checkout) == []


def test_an_archive_that_is_not_a_bundle_changes_nothing(tmp_path, monkeypatch, capsys):
    bundle, checkout = _bundle(tmp_path, monkeypatch, "bundle.tar.gz")
    head = checkout / "models" / "final_head.npz"
    head.parent.mkdir(parents=True)
    head.write_text("the installed head")
    partial = tmp_path / "partial.tar.gz"
    with tarfile.open(bundle) as src, tarfile.open(partial, "w:gz") as dst:
        member = src.getmember("models/final_head.npz")  # the head alone, without meta.json
        dst.addfile(member, src.extractfile(member))
    assert cli._install_bundle(str(partial)) == 1
    assert head.read_text() == "the installed head"


def test_the_bundle_the_readme_installs_is_pinned():
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    names = re.findall(r"releases/download/[^/\s]+/([^\s/)]+\.tar\.gz)", readme)
    assert names and all(name in cli.RELEASES for name in names)
    assert set(re.findall(r"\b[0-9a-f]{64}\b", readme)) <= set(cli.RELEASES.values())


def test_the_changelog_starts_with_this_version():
    root = Path(__file__).resolve().parent.parent
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    newest = re.search(r"^## \[(.+?)\] - \d{4}-\d{2}-\d{2}$", (root / "CHANGELOG.md").read_text(),
                       re.M)
    assert newest and newest.group(1) == version


def test_home_paths_are_expanded(monkeypatch):
    monkeypatch.delenv("CORTEXLIVE_CAPTURES", raising=False)
    args = cli.argparse.Namespace(captures="~/cortexlive-captures")
    assert cli._captures_root(args) == Path.home() / "cortexlive-captures"


def test_viewer_page_explains_how_to_start_and_offers_sound():
    page = html()
    assert 'id="soundBtn"' in page and "cortexlive.command" in page


def test_page_payload_leaves_out_where_the_video_lives():
    from cortexlive.viewer import payload

    cap = Path(__file__).resolve().parent.parent / "runs" / "live_demo" / "friends_s06e01a"
    shown = payload.capture_payload(cap)["meta"]["input"]
    assert "path" not in shown and shown["name"] == "friends_s06e01a.mkv"
