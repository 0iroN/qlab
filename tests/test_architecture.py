"""Katman kuralının makine tarafından zorlanması.

Spec §4.1: engine paketi viz/app import edemez. Bu kural yorumla değil,
import-linter ile korunur — ihlal edildiğinde test kırılır.
"""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skip(reason="sözleşmedeki modüller Task 3'te oluşacak")
def test_import_contracts_hold():
    result = subprocess.run(
        [sys.executable, "-m", "importlinter.cli", "lint-imports"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import sözleşmesi ihlal edildi:\n{result.stdout}\n{result.stderr}"
