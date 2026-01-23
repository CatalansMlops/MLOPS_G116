"""Tests for main module utilities."""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import mlops_g116.main as main_module


def test_resolve_output_prefix_uses_output_root(tmp_path: Path) -> None:
    """Ensure output prefix uses OUTPUT_ROOT when output_dir is under it."""
    output_root = tmp_path / "runs"
    output_dir = output_root / "exp1"
    output_dir.mkdir(parents=True)
    prefix = main_module._resolve_output_prefix(output_dir, str(output_root))
    assert prefix == "outputs/exp1", f"Unexpected prefix: {prefix}"


def test_resolve_output_prefix_uses_outputs_segment(tmp_path: Path) -> None:
    """Ensure output prefix falls back to outputs segment when not relative to cwd."""
    output_dir = tmp_path / "nested" / "outputs" / "run1"
    prefix = main_module._resolve_output_prefix(output_dir, None)
    assert prefix == "outputs/run1", f"Unexpected prefix: {prefix}"


def test_cleanup_output_dir_removes_empty_parents(tmp_path: Path) -> None:
    """Ensure cleanup removes output folder and prunes empty parents."""
    keep_file = tmp_path / "keep.txt"
    keep_file.write_text("keep", encoding="utf-8")
    output_dir = tmp_path / "outputs" / "run1"
    output_dir.mkdir(parents=True)
    (output_dir / "file.txt").write_text("data", encoding="utf-8")

    main_module._cleanup_output_dir(output_dir)

    assert not output_dir.exists(), "Expected output directory to be removed"
    assert not (tmp_path / "outputs").exists(), "Expected empty outputs directory to be removed"
    assert keep_file.exists(), "Expected unrelated files to remain"


def test_strip_classifier_replaces_fc1() -> None:
    """Ensure _strip_classifier replaces simple model head."""
    model = SimpleNamespace(fc1=torch.nn.Linear(4, 2))
    main_module._strip_classifier(model)
    assert isinstance(model.fc1, torch.nn.Identity), "Expected fc1 to be Identity"


def test_strip_classifier_replaces_backbone_fc() -> None:
    """Ensure _strip_classifier replaces backbone fc head."""
    backbone = SimpleNamespace(fc=torch.nn.Linear(4, 2))
    model = SimpleNamespace(backbone=backbone)
    main_module._strip_classifier(model)
    assert isinstance(model.backbone.fc, torch.nn.Identity), "Expected backbone fc to be Identity"


def test_strip_classifier_replaces_backbone_classifier() -> None:
    """Ensure _strip_classifier replaces backbone classifier head."""
    backbone = SimpleNamespace(classifier=torch.nn.Linear(4, 2))
    model = SimpleNamespace(backbone=backbone)
    main_module._strip_classifier(model)
    assert isinstance(model.backbone.classifier, torch.nn.Identity), "Expected backbone classifier to be Identity"


def test_is_port_open_detects_listening_port() -> None:
    """Ensure _is_port_open returns True for a listening socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        assert main_module._is_port_open(port), f"Expected port {port} to be open"
    finally:
        sock.close()


def test_upload_outputs_to_gcs_uploads_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure _upload_outputs_to_gcs uploads each file with normalized prefix."""
    uploads: list[tuple[str, str]] = []
    bucket_names: list[str] = []

    class DummyBlob:
        def __init__(self, name: str) -> None:
            self.name = name

        def upload_from_filename(self, filename: str) -> None:
            uploads.append((self.name, filename))

    class DummyBucket:
        def blob(self, name: str) -> DummyBlob:
            return DummyBlob(name)

    class DummyClient:
        def bucket(self, name: str) -> DummyBucket:
            bucket_names.append(name)
            return DummyBucket()

    monkeypatch.setattr(main_module.storage, "Client", lambda: DummyClient())

    output_dir = tmp_path / "outputs" / "run1"
    output_dir.mkdir(parents=True)
    (output_dir / "metrics.json").write_text("{}", encoding="utf-8")
    nested = output_dir / "reports"
    nested.mkdir()
    (nested / "figure.png").write_text("png", encoding="utf-8")

    main_module._upload_outputs_to_gcs(output_dir, "my-bucket", "/runs/exp1/")

    assert bucket_names == ["my-bucket"], f"Unexpected buckets: {bucket_names}"
    uploaded_paths = {name for name, _ in uploads}
    assert uploaded_paths == {
        "runs/exp1/metrics.json",
        "runs/exp1/reports/figure.png",
    }, f"Unexpected uploaded paths: {uploaded_paths}"


def test_launch_tensorboard_skips_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure _launch_tensorboard returns early when tensorboard is unavailable."""
    monkeypatch.setattr(main_module.shutil, "which", lambda *_args: None)

    def _raise(*_args: object, **_kwargs: object) -> None:
        """Fail if subprocess.Popen is invoked."""
        raise AssertionError("subprocess.Popen should not be called when tensorboard is missing")

    monkeypatch.setattr(main_module.subprocess, "Popen", _raise)
    main_module._launch_tensorboard(tmp_path, preferred_port=6006, open_browser=False)


def test_launch_snakeviz_invokes_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure _launch_snakeviz invokes subprocess when snakeviz is available."""
    monkeypatch.setitem(sys.modules, "snakeviz", SimpleNamespace())
    called: dict[str, list[str]] = {}

    def _fake_popen(cmd: list[str], *_args: object, **_kwargs: object) -> SimpleNamespace:
        """Capture subprocess command without spawning."""
        called["cmd"] = cmd
        return SimpleNamespace()

    monkeypatch.setattr(main_module.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(main_module, "_pick_available_port", lambda preferred_port, max_tries=25: preferred_port + 1)

    profile_path = tmp_path / "profile.prof"
    profile_path.write_text("data", encoding="utf-8")
    main_module._launch_snakeviz(profile_path, preferred_port=8080)

    assert called["cmd"][0] == main_module.sys.executable, "Expected Python executable in command"
    assert called["cmd"][2] == "snakeviz", f"Unexpected command: {called['cmd']}"


def test_resolve_output_prefix_uses_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure output prefix is relative to cwd when possible."""
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "runs" / "exp1"
    output_dir.mkdir(parents=True)

    prefix = main_module._resolve_output_prefix(output_dir, None)

    assert prefix == "runs/exp1", f"Unexpected prefix: {prefix}"
