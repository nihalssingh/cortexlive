"""One model process at a time, clear errors, and license notices that travel with the model."""

from __future__ import annotations

import socket
import subprocess
import sys
import tarfile
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
