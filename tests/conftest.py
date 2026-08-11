"""Test oturumu genelinde konsol kodlaması tuzağını kapatır.

`pytest tests/ -s` capture'ı kapatır; bu durumda print() çıktısı doğrudan
ham konsol koderına gider. Windows'un varsayılan CP1252 konsolunda Türkçe
karakter (ör. 'ı') içeren herhangi bir print() bu yüzden UnicodeEncodeError
ile çöker — `-s` olmadan pytest'in kendi capture katmanı bunu maskeler.

Burada qlab.run::main() içinde zaten kanıtlanmış aynı desen tekrar
kullanılıyor (reconfigure() önce denenir, olmayan ortamlarda TextIOWrapper'a
geri dönülür) böylece TÜM test oturumu için bir kez uygulanır: hem şu anki
tek print() (tests/tdse/test_tunneling.py) hem de gelecekte eklenecek her
Türkçe çıktı için. Dar istisna yakalama bilinçlidir — reconfigure()/wrapping
başarısız olsa bile hiçbir ek hata yükseltilmemelidir.
"""

from __future__ import annotations

import io
import sys


def _reconfigure_utf8(stream_name: str) -> None:
    stream = getattr(sys, stream_name)
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        try:
            if hasattr(stream, "buffer"):
                setattr(
                    sys,
                    stream_name,
                    io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace"),
                )
        except (AttributeError, ValueError, OSError):
            pass


_reconfigure_utf8("stdout")
_reconfigure_utf8("stderr")


# ---------------------------------------------------------------------------
# Referans JSON fixture'lari
#
# Bunlar bir zamanlar `tests/hf/conftest.py`'de yasiyordu ve orada, komut
# satirindaki YOL SIRASINA gore kayboluyorlardi. OLCULDU (pytest 9.1.1):
#     pytest -q tests/hf/test_boys.py tests/test_units.py tests/hf/test_analysis.py
#         -> 68 passed, 32 errors ("fixture 'hf_energies' not found")
#     pytest -q tests/hf/test_analysis.py tests/test_units.py tests/hf/test_scf.py
#         -> 56 passed, 20 errors
# Desen genel: `tests/hf/A`, `tests/B`, `tests/hf/C` verildiginde pytest
# `tests/hf/` dizinine IKINCI girisinde oradaki conftest'i artik baglamiyor.
# Kanonik `pytest -q` (testpaths = ["tests"]) etkilenmiyordu, o yuzden takim
# yesil kaliyor ve delik gorunmuyordu.
#
# Iki "bariz" aciklama OLCULEREK ELENDI: `tests/__init__.py` eklemek de,
# `tests/hf/__init__.py` kaldirmak da hatayi DEGISTIRMEDI (ikisinde de yine
# 32 error). Calisan tek cozum, fixture'lari her kosuda kosulsuz yuklenen KOK
# conftest'e tasimakti -- burasi. Regresyonu
# test_referans_fixturelari_yol_SIRASINDAN_bagimsiz_bulunuyor bagliyor.
#
# `pytest.skip` degil `pytest.fail`: eksik bir referans dosyasi ATLANACAK bir
# kosul degil, bozuk bir ortamdir (ayni gerekce README'de de yazili).
# ---------------------------------------------------------------------------

import json
from pathlib import Path

import pytest

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"


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
