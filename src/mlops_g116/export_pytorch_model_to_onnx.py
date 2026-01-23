"""Inspect or export ONNX models for deployment workflows."""

from __future__ import annotations

from pathlib import Path

import onnx


def inspect_onnx_graph(model_path: str | Path) -> str:
    """Load an ONNX model and return a printable graph.

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        Human-readable representation of the ONNX graph.
    """
    path = Path(model_path)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return onnx.helper.printable_graph(model.graph)


def main() -> None:
    """Print the graph for the default ONNX model path."""
    graph = inspect_onnx_graph("resnet18.onnx")
    print(graph)


if __name__ == "__main__":
    main()
