"""Izgara uzerinde orbital ve yogunluk degerlendirmesi.

Bu dosyanin ana dogrulama fikri: izgarada sayisal olarak hesaplanan ortusme
matrisi, motorun analitik overlap_matrix'ine oturmalidir. Boylece yeni kod
Modul 1B'nin dogrulanmis motoruna baglanir ve bagimsiz bir referans gerekmez.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import overlap_matrix
from qlab.observable.hf_obs import VolumeGrid, evaluate_basis_function
from qlab.types import Molecule


@pytest.fixture(scope="module")
def h2():
    molecule = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    return molecule, build_basis(molecule, "sto-3g")


def test_grid_around_molekulu_kapsiyor(h2):
    molecule, _ = h2
    grid = VolumeGrid.around(molecule, padding=6.0, points_per_axis=40)
    assert grid.shape == (40, 40, 40)
    ust = grid.origin + grid.spacing * (np.array(grid.shape) - 1)
    for atom in molecule.atoms:
        assert np.all(atom.position > grid.origin)
        assert np.all(atom.position < ust)


def test_hacim_elemani_eksen_araliklarinin_carpimi(h2):
    molecule, _ = h2
    grid = VolumeGrid.around(molecule, padding=6.0, points_per_axis=40)
    beklenen = float(np.prod(grid.spacing))
    assert grid.volume_element == pytest.approx(beklenen, rel=1e-12, abs=0.0)


def test_izgara_ortusmesi_analitik_ortusmeye_oturuyor(h2):
    """Olculdu: n=80 izgarada maks fark 2.70e-09.

    Bu testin gucu buradan gelir: yeni degerlendirme kodu, Modul 1B'nin
    PySCF'e karsi dogrulanmis overlap_matrix'ine baglanir. Kontraksiyon
    normalizasyonu atlanirsa fark 1e-2 mertebesine cikar ve test kirmizi olur.
    """
    molecule, basis = h2
    grid = VolumeGrid.around(molecule, padding=7.0, points_per_axis=80)
    values = np.stack([evaluate_basis_function(b, grid) for b in basis])
    numeric = np.einsum("ixyz,jxyz->ij", values, values) * grid.volume_element
    np.testing.assert_allclose(numeric, overlap_matrix(basis), atol=1e-7, rtol=0.0)


def test_s_fonksiyonu_merkezde_maksimum(h2):
    molecule, basis = h2
    grid = VolumeGrid.around(molecule, padding=6.0, points_per_axis=41)
    values = evaluate_basis_function(basis[0], grid)
    en_buyuk = np.unravel_index(np.argmax(values), values.shape)
    konum = grid.origin + grid.spacing * np.array(en_buyuk)
    np.testing.assert_allclose(konum, basis[0].center, atol=0.4, rtol=0.0)


def test_p_fonksiyonu_dugum_duzlemine_sahip():
    """p_z fonksiyonu z=merkez duzleminde tam olarak sifirdir."""
    molecule = Molecule.from_string("O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587")
    basis = build_basis(molecule, "sto-3g")
    p_fonksiyonlari = [b for b in basis if sum(b.angular) == 1]
    assert len(p_fonksiyonlari) == 3
    grid = VolumeGrid.around(molecule, padding=5.0, points_per_axis=41)
    for fonksiyon in p_fonksiyonlari:
        eksen = int(np.argmax(fonksiyon.angular))
        values = evaluate_basis_function(fonksiyon, grid)
        koordinat = grid.points[..., eksen] - fonksiyon.center[eksen]
        dugum = np.abs(koordinat) < 1e-12
        if dugum.any():
            assert np.abs(values[dugum]).max() < 1e-30
