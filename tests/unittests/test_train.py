"""Unit tests for train utilities."""

from __future__ import annotations

import socket

import pytest

from mlops_g116.train import _is_port_open, _pick_available_port


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
