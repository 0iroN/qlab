"""Referans JSON dosyalarini test fixture'lari olarak sunar."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "reference"


def _load(name: str) -> dict:
    path = REFERENCE_DIR / name
    if not path.is_file():
        pytest.fail(
            f"Referans dosyasi bulunamadi: {path}\n"
            "tools/generate_reference.py bir Linux makinede kosulup ciktisi "
            "reference/ altina islenmeli. Bkz. reference/README.md"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sto3g_basis_data() -> dict:
    return _load("basis_sto3g.json")


@pytest.fixture(scope="session")
def g631_basis_data() -> dict:
    return _load("basis_631g.json")


@pytest.fixture(scope="session")
def h2_sto3g_integrals() -> dict:
    return _load("integrals_h2_074.json")


@pytest.fixture(scope="session")
def h2o_sto3g_integrals() -> dict:
    return _load("integrals_h2o.json")


@pytest.fixture(scope="session")
def hf_energies() -> dict:
    return _load("hf_energies.json")
