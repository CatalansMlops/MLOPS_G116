"""Pytest configuration for unit tests."""

from __future__ import annotations

import os

import matplotlib

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")
