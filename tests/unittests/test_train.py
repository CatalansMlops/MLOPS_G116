"""Unit tests for train utilities."""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlops_g116.train import _is_port_open, _launch_snakeviz, _launch_tensorboard, _pick_available_port


def _bind_listening_socket() -> tuple[socket.socket, int]:
    """Create a listening socket bound to localhost."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock, sock.getsockname()[1]


def test_is_port_open_detects_listening_port() -> None:
    """Ensure _is_port_open returns True for a listening socket."""
    sock, port = _bind_listening_socket()
    try:
        assert _is_port_open(port), f"Expected port {port} to be open"
    finally:
        sock.close()


def test_is_port_open_false_for_invalid_port() -> None:
    """Ensure _is_port_open returns False for an invalid port."""
    assert not _is_port_open(0), "Expected port 0 to be closed"


def test_pick_available_port_falls_back_when_range_busy() -> None:
    """Ensure _pick_available_port returns preferred port when range is occupied."""
    sock, port = _bind_listening_socket()
    sock_next = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        try:
            sock_next.bind(("127.0.0.1", port + 1))
            sock_next.listen(1)
        except OSError:
            pytest.skip("Port range unavailable for fallback test.")
        selected = _pick_available_port(port, max_tries=2)
        assert selected == port, f"Expected preferred port {port}, got {selected}"
    finally:
        sock.close()
        sock_next.close()


def test_launch_tensorboard_skips_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure _launch_tensorboard returns early when tensorboard is unavailable."""
    monkeypatch.setattr("mlops_g116.train.shutil.which", lambda *_args: None)

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("subprocess.Popen should not be called when tensorboard is missing")

    monkeypatch.setattr("mlops_g116.train.subprocess.Popen", _raise)
    _launch_tensorboard(tmp_path, preferred_port=6006, open_browser=False)


def test_launch_snakeviz_invokes_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure _launch_snakeviz invokes subprocess when snakeviz is available."""
    monkeypatch.setitem(sys.modules, "snakeviz", SimpleNamespace())
    called: dict[str, list[str]] = {}

    def _fake_popen(cmd: list[str], *_args: object, **_kwargs: object) -> SimpleNamespace:
        called["cmd"] = cmd
        return SimpleNamespace()

    monkeypatch.setattr("mlops_g116.train.subprocess.Popen", _fake_popen)

    def _pick_port(preferred_port: int, max_tries: int = 25) -> int:
        return preferred_port + 1

    monkeypatch.setattr("mlops_g116.train._pick_available_port", _pick_port)

    profile_path = tmp_path / "profile.prof"
    profile_path.write_text("data", encoding="utf-8")
    _launch_snakeviz(profile_path, preferred_port=8080)

    assert called["cmd"][0] == sys.executable
    assert called["cmd"][2] == "snakeviz"
