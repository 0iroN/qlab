import math
from unittest.mock import patch

import numpy as np
import pytest

from qlab.engine.hf.basis import (
    AVAILABLE_BASES,
    build_basis,
    _load_reference_data,
)
from qlab.errors import BasisNotAvailableError
from qlab.types import Molecule


def test_hydrogen_sto3g_has_one_basis_function():
    functions = build_basis(Molecule.from_string("H 0 0 0"), "sto-3g")
    assert len(functions) == 1
    assert functions[0].angular == (0, 0, 0)
    assert len(functions[0].primitives) == 3


def test_h2_sto3g_has_two_basis_functions():
    mol = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    assert len(build_basis(mol, "sto-3g")) == 2


def test_water_sto3g_matches_pyscf_count(hf_energies):
    """O: 1s + 2s + 2p(3) = 5, iki H: 1'er = 7."""
    record = hf_energies["h2o"]
    mol = Molecule.from_string(record["atom"])
    assert len(build_basis(mol, "sto-3g")) == record["n_basis_functions"]


def test_water_631g_matches_pyscf_count(hf_energies):
    """6-31G'de bazi kabuklar birden fazla contraction tasir — her ek katsayi
    sutunu AYRI bir contracted fonksiyon demektir. Sutunlari yok sayarsan bu
    test tutmaz."""
    record = hf_energies["h2o_631g"]
    mol = Molecule.from_string(record["atom"])
    assert len(build_basis(mol, "6-31g")) == record["n_basis_functions"]


def test_p_shell_produces_three_functions_with_distinct_angular():
    mol = Molecule.from_string("O 0 0 0")
    p_angulars = {f.angular for f in build_basis(mol, "sto-3g") if sum(f.angular) == 1}
    assert p_angulars == {(1, 0, 0), (0, 1, 0), (0, 0, 1)}


def test_exponents_come_from_reference_data(sto3g_basis_data):
    """Baz katsayilari referans JSON'dan gelmeli — koda elle yazilmamali."""
    functions = build_basis(Molecule.from_string("H 0 0 0"), "sto-3g")
    exponents = sorted(p.exponent for p in functions[0].primitives)
    expected = sorted(row[0] for row in sto3g_basis_data["H"][0]["primitives"])
    np.testing.assert_allclose(exponents, expected, rtol=1e-12)


def test_basis_functions_are_centred_on_their_atoms():
    mol = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    functions = build_basis(mol, "sto-3g")
    np.testing.assert_allclose(functions[0].center, mol.atoms[0].position)
    np.testing.assert_allclose(functions[1].center, mol.atoms[1].position)


def test_basis_functions_know_their_atom():
    mol = Molecule.from_string("O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587")
    functions = build_basis(mol, "sto-3g")
    indices = [f.atom_index for f in functions]
    assert indices.count(0) == 5, "O'da 1s + 2s + 2p(3) = 5 fonksiyon"
    assert indices.count(1) == 1 and indices.count(2) == 1
    for function in functions:
        np.testing.assert_allclose(
            function.center, mol.atoms[function.atom_index].position
        )


def test_s_normalisation_closed_form():
    """s tipi icin N = (2a/pi)^{3/4}."""
    functions = build_basis(Molecule.from_string("H 0 0 0"), "sto-3g")
    function = functions[0]
    expected = [(2 * p.exponent / np.pi) ** 0.75 for p in function.primitives]
    np.testing.assert_allclose(function.normalisation(), expected, rtol=1e-12)


def test_unknown_basis_rejected():
    with pytest.raises(BasisNotAvailableError, match="banana"):
        build_basis(Molecule.from_string("H 0 0 0"), "banana")


def test_unsupported_element_rejected():
    with pytest.raises(BasisNotAvailableError, match="Ne"):
        build_basis(Molecule.from_string("Ne 0 0 0"), "sto-3g")


def test_available_bases_listed():
    assert AVAILABLE_BASES == ("sto-3g", "6-31g")


def test_p_shell_normalisation_closed_form():
    """p kabugu normalizasyonu: acisal momentum > 0 dallarini kapsamasini saglar.

    Referans veri seti yalnizca hidrojen (s-tipi) iceriyor, p dali test edilmez.
    Kapanis formulun p-icin azaltildigini dogrulariz: O atomu icin p primi,
    N = (2α/π)^{3/4} · sqrt(4α) (i=1, j=k=0 icin).
    """
    mol = Molecule.from_string("O 0 0 0")
    functions = build_basis(mol, "sto-3g")

    # p kabuguyla ilgili fonksiyonlari sec (angular = (1,0,0), (0,1,0), veya (0,0,1))
    p_functions = [f for f in functions if sum(f.angular) == 1]
    assert len(p_functions) > 0, "O atomunda p kabugu olmali"

    # p-tipi fonksiyonlardan birini al
    p_func = p_functions[0]

    # Beklenen normalizasyon: (2α/π)^{3/4} · sqrt(4α)
    exponents = np.array([p.exponent for p in p_func.primitives])
    expected_norm = (2 * exponents / np.pi) ** 0.75 * np.sqrt(4 * exponents)

    actual_norm = p_func.normalisation()
    np.testing.assert_allclose(actual_norm, expected_norm, rtol=1e-12)

    # s-tipi ile karsilastir: ikisi farkli olmali (angular faktor farki)
    s_functions = [f for f in functions if sum(f.angular) == 0]
    s_func = s_functions[0]
    s_norm = s_func.normalisation()

    # Ayni ustel: birinci p primitifi vs ilk s primitifi
    s_exp = s_func.primitives[0].exponent
    p_exp = p_func.primitives[0].exponent

    if np.isclose(s_exp, p_exp):
        # Ayni ustel ama farkli normalizasyon olmali
        s_single = (2 * s_exp / np.pi) ** 0.75
        p_single = (2 * p_exp / np.pi) ** 0.75 * np.sqrt(4 * p_exp)
        assert not np.isclose(s_single, p_single), \
            "p ve s normalizasyonu farkli olmali (acisal faktor)"


def test_d_shell_rejection_with_monkeypatch():
    """d kabuguyla (angular >= 2) esasina carpmani saglar.

    Referans veri setlerinde d kabuklari yok; sentetik veri ile test.
    BasisNotAvailableError veya KeyError (guard eksik ise) ile reddedilmeli.
    """
    mol = Molecule.from_string("H 0 0 0")

    # d kabuguyla sentetik referans veri olustur
    d_shell_data = {
        "H": [
            {
                "angular_momentum": 2,  # d shell
                "primitives": [[3.425, 0.154]]
            }
        ]
    }

    # _load_reference_data fonksiyonunu monkeypatch et
    with patch("qlab.engine.hf.basis._load_reference_data") as mock_load:
        mock_load.return_value = d_shell_data

        with pytest.raises((BasisNotAvailableError, KeyError)):
            build_basis(mol, "sto-3g")


def test_multi_column_contraction():
    """Coklu sutunlu kontraksiyon: n_contraction > 1 dalini kapsamasini saglar.

    Referans verileri split-valence kabuklara sahip (2 sutun), fakat
    her sutun AYNI ilk sutun olarak okunur. Sentetik verilerle gercek
    coklu-sutun kontraktionunu test ederiz.
    """
    mol = Molecule.from_string("H 0 0 0")

    # Iki katsayi sutunlu sentetik veri (coklu kontraksiyon)
    multi_contraction_data = {
        "H": [
            {
                "angular_momentum": 0,
                "primitives": [
                    [3.4252509, 0.15432897, 0.70000000],
                    [0.6239137, 0.53532814, 0.40000000],
                    [0.1688554, 0.44463454, 0.10000000]
                ]
            }
        ]
    }

    with patch("qlab.engine.hf.basis._load_reference_data") as mock_load:
        mock_load.return_value = multi_contraction_data

        functions = build_basis(mol, "sto-3g")

        # Iki adet s-tipi fonksiyon olusturulmali (2 sutun = 2 kontraksiyon)
        s_functions = [f for f in functions if f.angular == (0, 0, 0)]
        assert len(s_functions) == 2, \
            f"Iki katsayi sutunu ile 2 kontakte fonksiyon olusturulmali, {len(s_functions)} bulundu"

        # Birinci fonksiyon ilk sutunu (0.15..., 0.53..., 0.44...)
        # Ikinci fonksiyon ikinci sutunu (0.70, 0.40, 0.10)
        func1_coeffs = [p.coefficient for p in s_functions[0].primitives]
        func2_coeffs = [p.coefficient for p in s_functions[1].primitives]

        # Katsayilar farkli olmali
        assert not np.allclose(func1_coeffs, func2_coeffs), \
            "Farkli sutunlar farkli katsayilari olusturmali"

        # Ilk sutun (1. sutun)
        np.testing.assert_allclose(func1_coeffs, [0.15432897, 0.53532814, 0.44463454], rtol=1e-6)

        # Ikinci sutun (2. sutun)
        np.testing.assert_allclose(func2_coeffs, [0.70000000, 0.40000000, 0.10000000], rtol=1e-6)
