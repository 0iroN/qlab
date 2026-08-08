import math
from unittest.mock import patch

import numpy as np
import pytest
from scipy.integrate import quad

from qlab.engine.hf.basis import (
    AVAILABLE_BASES,
    BasisFunction,
    PrimitiveGaussian,
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


def _gauss_moment(power: int, alpha: float) -> float:
    """Tek eksende integral( x^power * exp(-alpha*x^2), x=-inf..inf ).

    scipy.integrate.quad ile SAYISAL olarak hesaplanir; basis.py'deki cift
    faktoriyel kapali formla hicbir cebir paylasmaz -- bu yuzden
    _self_overlap()'i dogrulamak icin bagimsiz bir referans olusturur.
    """
    integrand = lambda x: x**power * math.exp(-alpha * x * x)
    value, _ = quad(integrand, -np.inf, np.inf)
    return value


def _numeric_self_overlap(function) -> float:
    """Kontrakte fonksiyonun oz-ortusmesini sayisal integralle hesaplar.

    basis.py'nin _self_overlap()'i veya contraction_normalisation()'i
    CAGRILMAZ; yalnizca ham us/katsayi verisi ve normalisation() (primitif
    basina normalizasyon -- acisal kapali formdan bagimsiz, ayri bir
    fonksiyon) kullanilir. Merkez konumu integralde onemsizdir (oteleme
    degismezligi): fonksiyon sanki orijinde gibi entegre edilir.
    """
    i, j, k = function.angular
    exponents = [p.exponent for p in function.primitives]
    coefficients = [p.coefficient for p in function.primitives]
    norms = function.normalisation()
    weights = [c * n for c, n in zip(coefficients, norms)]

    toplam = 0.0
    for p in range(len(exponents)):
        for q in range(len(exponents)):
            alpha = exponents[p] + exponents[q]
            ix = _gauss_moment(2 * i, alpha)
            iy = _gauss_moment(2 * j, alpha)
            iz = _gauss_moment(2 * k, alpha)
            toplam += weights[p] * weights[q] * ix * iy * iz
    return toplam


def _water_basis():
    mol = Molecule.from_string("O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587")
    return build_basis(mol, "sto-3g")


@pytest.mark.parametrize("angular", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_p_shell_self_overlap_matches_numerical_integration(angular):
    """_self_overlap()'in acisal-momentum kismini bagimsiz sayisal integralle
    capraz dogrular.

    Task 5'te tek test edilen acisal deger (0, 0, 0) oldugu icin cift
    faktoriyel terimindeki her carpan 1'e iniyordu -- formulun acisal kismi
    hicbir testte gercekten hesaba katilmiyordu (bkz.
    .superpowers/sdd/task-5-report.md, 'Self-review findings'). Bu test, su
    molekulundeki gercek oksijen 2p fonksiyonlarinin (build_basis'ten) uc
    ekseni icin de _self_overlap()'i sayisal referansla karsilastirarak bu
    bosluk kapatir.
    """
    functions = _water_basis()
    p_functions = [f for f in functions if f.angular == angular]
    assert len(p_functions) == 1
    function = p_functions[0]

    numeric = _numeric_self_overlap(function)
    implementation = function._self_overlap()

    np.testing.assert_allclose(numeric, implementation, rtol=1e-10)


def test_d_shell_self_overlap_matches_numerical_integration():
    """(2n-1)!! carpanini l=2'de sinar -- n=0 ve n=1'de bu carpan 1'dir.

    _self_overlap()'teki cift faktoriyel n=0 icin (-1)!!=1, n=1 icin 1!!=1
    verir; yani mevcut TUM testler (s ve p kabuklari) yanlis yazilmis bir
    cift faktoriyelden etkilenmez -- hata ancak d kabugu geldiginde (n=2'de
    3!!=3) ortaya cikardi ve o ana kadar gorunmez kalirdi.

    build_basis l>=2'yi reddettigi icin (basis.py, spec bolum 13) fonksiyon
    elle kurulur; referans, test_p_shell_self_overlap_matches_numerical_
    integration ile ayni bagimsiz sayisal integraldir (_numeric_self_overlap,
    scipy.integrate.quad -- basis.py'nin kapali formuyle hicbir cebir
    paylasmaz).
    """
    function = BasisFunction(
        center=np.array([0.0, 0.0, 0.0]),
        angular=(2, 0, 0),
        primitives=(
            PrimitiveGaussian(exponent=1.9, coefficient=0.42),
            PrimitiveGaussian(exponent=0.55, coefficient=0.71),
        ),
        atom_index=0,
    )

    numeric = _numeric_self_overlap(function)
    implementation = function._self_overlap()

    np.testing.assert_allclose(numeric, implementation, rtol=1e-10)

    # Cift faktoriyel gercekten devrede mi: ayni ustel/katsayilarla kurulmus
    # s tipi bir fonksiyonun oz-ortusmesiyle ayni cikmamalidir. (Bu, "carpan
    # 1'e indi ve test yine de gecti" senaryosunu dogrudan eler.)
    s_version = BasisFunction(
        center=function.center,
        angular=(0, 0, 0),
        primitives=function.primitives,
        atom_index=0,
    )
    assert not np.isclose(implementation, s_version._self_overlap())


def test_p_shell_contraction_normalisation_matches_pyscf_water_diagonal(
    h2o_sto3g_integrals,
):
    """Ikinci, bagimsiz referans: reference/integrals_h2o.json'daki PySCF
    ortusme matrisinin kosegeni tam 1.0'dir (O'nun 2p fonksiyonlari dahil).

    contraction_normalisation() carpani uygulandiktan sonra tam-normalize
    kontrakte fonksiyonun sayisal oz-ortusme integrali de ayni kosegen
    degerine esit olmalidir. Bu, basis.py'nin formulunu degil, PySCF'in
    bagimsiz olarak hesapladigi gercek bir referans degerini kullanir.
    """
    functions = _water_basis()
    overlap = h2o_sto3g_integrals["overlap"]

    p_gorulen = 0
    for idx, function in enumerate(functions):
        if sum(function.angular) != 1:
            continue
        p_gorulen += 1
        i, j, k = function.angular
        exponents = [p.exponent for p in function.primitives]
        coefficients = [p.coefficient for p in function.primitives]
        norms = function.normalisation()
        c_norm = function.contraction_normalisation()
        weights = [c * n * c_norm for c, n in zip(coefficients, norms)]

        toplam = 0.0
        for p in range(len(exponents)):
            for q in range(len(exponents)):
                alpha = exponents[p] + exponents[q]
                ix = _gauss_moment(2 * i, alpha)
                iy = _gauss_moment(2 * j, alpha)
                iz = _gauss_moment(2 * k, alpha)
                toplam += weights[p] * weights[q] * ix * iy * iz

        # rtol=0.0 acikca verilir: assert_allclose'un varsayilan rtol'u
        # 1e-7'dir ve kosegen degeri 1.0 oldugu icin "atol=1e-10" yazan bir
        # iddia gercekte 1e-7'yi zorlar. Olculen sapma 2.22e-15 (uc p
        # fonksiyonunda da ayni); sinir 2e-14, ~9 kat pay.
        np.testing.assert_allclose(
            toplam, overlap[idx][idx], atol=2e-14, rtol=0.0
        )

    assert p_gorulen == 3, "Su/STO-3G'de O'nun 3 adet 2p fonksiyonu olmali"


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
