"""Referans dosyalari var, okunabiliyor ve beklenen anahtarlari tasiyor."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_DEPO = Path(__file__).resolve().parents[2]


def test_basis_data_has_expected_elements(sto3g_basis_data):
    for element in ("H", "He", "Li", "C", "N", "O", "F"):
        assert element in sto3g_basis_data, f"{element} STO-3G verisi eksik"


def test_hydrogen_sto3g_has_three_primitives(sto3g_basis_data):
    shells = sto3g_basis_data["H"]
    assert len(shells) == 1, "H'nin STO-3G'de tek kabugu olmali"
    assert shells[0]["angular_momentum"] == 0
    assert len(shells[0]["primitives"]) == 3


def test_oxygen_sto3g_has_s_and_p_shells(sto3g_basis_data):
    angulars = sorted(s["angular_momentum"] for s in sto3g_basis_data["O"])
    assert 0 in angulars and 1 in angulars, "O'da hem s hem p kabugu olmali"


def test_energies_reference_records_pyscf_version(hf_energies):
    assert "_pyscf_version" in hf_energies
    assert hf_energies["_pyscf_version"]


def test_h2_reference_energy_is_physically_sensible(hf_energies):
    record = hf_energies["h2_074"]
    assert record["n_electrons"] == 2
    assert record["n_basis_functions"] == 2
    assert -1.2 < record["energy_total"] < -1.0
    assert record["energy_nuclear"] > 0


def test_integral_matrices_are_square_and_consistent(h2_sto3g_integrals):
    overlap = np.array(h2_sto3g_integrals["overlap"])
    kinetic = np.array(h2_sto3g_integrals["kinetic"])
    eri = np.array(h2_sto3g_integrals["eri"])
    n = overlap.shape[0]
    assert overlap.shape == kinetic.shape == (n, n)
    assert eri.shape == (n, n, n, n)
    # rtol=0.0 acikca: varsayilan rtol=1e-7, beklenen deger 1.0 oldugu icin
    # "atol=1e-10" yazan iddiayi gercekte 1e-7'ye gevsetiyordu. Referans
    # dosyasindaki olculen sapma 2.22e-16; sinir 2e-15, ~9 kat pay.
    np.testing.assert_allclose(np.diag(overlap), 1.0, atol=2e-15, rtol=0.0)


@pytest.mark.parametrize(
    ("yollar", "hedef"),
    [
        pytest.param(
            ["tests/hf/test_boys.py", "tests/test_units.py", "tests/hf/test_analysis.py"],
            "tests/hf/test_analysis.py",
            id="analysis-ucuncu",
        ),
        pytest.param(
            ["tests/hf/test_analysis.py", "tests/test_units.py", "tests/hf/test_scf.py"],
            "tests/hf/test_scf.py",
            id="scf-ucuncu",
        ),
    ],
)
def test_referans_fixturelari_yol_SIRASINDAN_bagimsiz_bulunuyor(yollar, hedef):
    """Referans fixture'lari komut satirindaki yol SIRASINA gore kaybolmamali.

    OLCULDU (pytest 9.1.1, bu depo, HEAD 81ebc62): `tests/hf/A`, `tests/B`,
    `tests/hf/C` deseninde UCUNCU argumanin fixture'lari bulunamiyordu --
    "fixture 'hf_energies' not found". Iki bagimsiz ornek, ikisi de
    deterministik (iki kez ust uste tekrarlandi):
        pytest -q tests/hf/test_boys.py tests/test_units.py tests/hf/test_analysis.py
            -> 68 passed, 32 errors
        pytest -q tests/hf/test_analysis.py tests/test_units.py tests/hf/test_scf.py
            -> 56 passed, 20 errors
    Her iki ikili alt kume ve kanonik `pytest -q` (536 passed) YESILDI, yani
    hata yalnizca uc-arguman deseninde goruluyordu.

    SEBEP OLCULDU, ve bildirilen tanı YANLISTI: sorun `tests/__init__.py`'nin
    yoklugu DEGIL. Iki adayi da denedim, IKISI DE duzeltmedi --
    `tests/__init__.py` eklendiginde yine 32 error, `tests/hf/__init__.py`
    kaldirildiginda yine 32 error. Kalan ve calisan tek aciklama, fixture'larin
    `tests/hf/conftest.py` gibi ARA bir dizinde yasamasiydi: pytest o conftest'i
    yalnizca `tests/hf/` dizinine ILK girildiginde bagliyor, araya `tests/`
    seviyesinden bir arguman girip cikildiktan sonra ikinci kez girildiginde
    baglamiyor. Fixture'lar kok `tests/conftest.py`'ye tasindi; kok conftest her
    kosuda kosulsuz yuklenir, yani desen artik kurulamaz.

    NEDEN ONEMLI (kozmetik degil): "su uc dosyayi kostur" siradan bir
    gelistirici hareketi ve mutasyon dongusunun tavsiye edilen is akisi.
    Kaybolan fixture 32 testi ERROR'a cevirir; bir mutasyon olcumu sirasinda
    bu, mutasyonun oldurdugu testlerle karistirilabilecek SAHTE bir sinyaldir
    -- nitekim onceki bir incelemede bir kez karistirildi.

    Alt surecte kosar: pytest'in argumanlari kendi surecinin baslangicinda
    toplamasi bu kusurun ta kendisidir, yani ic icaden sinanamaz.
    """
    sonuc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", *yollar],
        cwd=str(_DEPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    cikti = sonuc.stdout + sonuc.stderr
    assert "fixture 'hf_energies' not found" not in cikti, (
        f"{hedef} icin referans fixture'lari yol sirasi yuzunden kayboldu:\n{cikti[-3000:]}"
    )
    assert " error" not in cikti, f"beklenmeyen ERROR:\n{cikti[-3000:]}"
    assert sonuc.returncode == 0, f"kod {sonuc.returncode}\n{cikti[-3000:]}"
