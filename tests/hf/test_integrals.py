"""Molekuler integral testleri (S, T, V, ERI).

Referanslar: reference/integrals_h2_074.json (PySCF, STO-3G, H2 0.74 A, yalniz
s kabugu) ve reference/integrals_h2o.json (PySCF, STO-3G, su, uc p fonksiyonu).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from qlab.engine.hf.basis import BasisFunction, PrimitiveGaussian, build_basis
from qlab.engine.hf.integrals import (
    core_hamiltonian,
    dipole_matrices,
    eri_tensor,
    kinetic_matrix,
    nuclear_matrix,
    overlap_matrix,
)
from qlab.types import Atom, Molecule


@pytest.fixture(scope="module")
def h2(hf_energies):
    molecule = Molecule.from_string(hf_energies["h2_074"]["atom"])
    return molecule, build_basis(molecule, "sto-3g")


# --------------------------------------------------------------------------
# PySCF karsilastirmalarinda tolerans: rtol ACIKCA verilmeli
# --------------------------------------------------------------------------
#
# np.testing.assert_allclose'un varsayilan rtol'u 1e-7'dir ve kriter
# |actual-desired| <= atol + rtol*|desired| seklindedir. Su molekulunun
# ~62 hartree'lik cekirdek-cekim elemanlari icin varsayilan rtol tek basina
# ~6e-6'lik bir bosluk birakir -- yani gorunurdeki atol=1e-10 aldaticidir,
# gercekte uygulanan sinir cok daha gevsektir. Bu yuzden asagida ve su
# molekulu grubunda atol VE rtol birlikte, acikca verilir.
#
# Sekiz PySCF karsilastirmasinin yedisi TEK bir kurala uyar: atol=1e-10,
# rtol=1e-10. Bu kural olculen en siki elemanda bile en az ~5.2 kat pay
# birakir (tam margin tablosu: .superpowers/sdd/b1-task-7-fix2-report.md,
# Finding 1). Tek istisna su molekulunun cekirdek-cekim matrisidir
# (test_water_nuclear_matches_pyscf) -- ayni kuralla orada pay ~1.80x'e
# duser (kirilgan), bu yuzden o test ozel olarak atol=3e-10, rtol=3e-10
# kullanir; fiziksel gerekce testin hemen ustunde.
#
# Kalan sapma ozyinelemenin kendisinden DEGIL, referansi ureten PySCF
# derlemesinin bohr sabitinden (CODATA 2010, param.BOHR = 0.52917721092)
# qlab'in kullandigi sabitten (CODATA 2018, units.ANGSTROM_TO_BOHR =
# 1.8897261246257702) gelir; iki sabit ilk 10 basamakta (0.5291772109) ayni,
# 11. basamaktan itibaren ayrisir (...92 / ...903). Bu kayma cekirdek
# konumlarina dogrudan gecer ve oradan HER integrale KENDI cekirdek-konumu
# turevi uzerinden sizar -- sizinti buyuklukle orantili DEGILDIR: olculen
# bagil sapmalar 2.36e-11 ile 8.40e-10 arasinda (~36 kat), mutlak sapmalar
# 1.42e-11 ile 2.13e-10 arasinda (~15 kat) degisir; ne biri sabit ne digeri
# sabittir. Ozyinelemenin kendisi cok daha hassastir; kanit asagidaki
# test_water_matches_pyscf_to_machine_precision_at_pyscf_geometry (PySCF'in
# KENDI bohr sabitiyle kurulan geometride dort tensor de rtol=0, atol=1e-12
# ile eslesir). O test ozyinelemeyi kilitleyen asil kapidir; buradaki rtol
# degerleri yalnizca birim-sabiti farkini yutar, integral hassasiyetini
# olcmez.


def test_overlap_matches_pyscf(h2, h2_sto3g_integrals):
    _, basis = h2
    expected = np.array(h2_sto3g_integrals["overlap"])
    np.testing.assert_allclose(overlap_matrix(basis), expected, atol=1e-10, rtol=1e-10)


def test_kinetic_matches_pyscf(h2, h2_sto3g_integrals):
    _, basis = h2
    expected = np.array(h2_sto3g_integrals["kinetic"])
    np.testing.assert_allclose(kinetic_matrix(basis), expected, atol=1e-10, rtol=1e-10)


def test_nuclear_matches_pyscf(h2, h2_sto3g_integrals):
    molecule, basis = h2
    expected = np.array(h2_sto3g_integrals["nuclear"])
    np.testing.assert_allclose(
        nuclear_matrix(basis, molecule), expected, atol=1e-10, rtol=1e-10
    )


def test_overlap_diagonal_is_unity(h2):
    """Normalizasyon ozdesligi -- PySCF referansiyla ilgisi yok.

    Kosegen deger tanim geregi 1.0'dir; olculen fark tam 0.0 (bkz.
    b1-task-7-fix-report.md, Finding 1). atol=1e-13 makine hassasiyetine
    yakin ama tam sifir degil, boylece platforma bagli toplama sirasina kucuk
    bir pay birakiyor.
    """
    _, basis = h2
    np.testing.assert_allclose(np.diag(overlap_matrix(basis)), 1.0, atol=1e-13, rtol=0.0)


def test_matrices_are_symmetric(h2):
    """M == M.T oz-tutarlilik denetimi -- PySCF referansiyla ilgisi yok.

    Olculen fark tam 0.0 (bit-birebir ayni, bkz. b1-task-7-fix-report.md,
    Finding 1), ama bu genel bir garanti degil -- bu testin kullandigi h2
    fixture'inin ozel bir sonucu. h2 s-kabugu ve es-cekirdeklidir (angular
    her iki fonksiyon icin de (0,0,0)), bu yuzden matrix[a,b] ile
    matrix[b,a]'nin ikisi de _pair_data'nin i==j==0 kisayolundan gecer
    (deger dogrudan e000 = exp(-mu*(A-B)^2), a<->b degisiminde degismez) --
    _Axis.e() ozyinelemesi hic calismaz. Acisal momentum sifirdan farkli
    oldugunda (ornegin su/p kabugu) _pair_data(a,b) ile _pair_data(b,a)
    x_pa/x_pb ve angular_a/angular_b degerlerini degistirir, ve
    ozyineleme j'yi i'den once azalttigi icin iki siralama farkli
    dugumlerden gecer -- bit-birebir eslesme artik garanti degildir.
    atol=1e-13 burada platforma bagli toplama sirasina degil, bu ozel
    s-kabugu/es-cekirdek durumuna kucuk bir pay birakiyor; bu testi su
    molekulune genisletmek isteyen biri 1e-13'un sikilasabilecegini
    bilmeli.
    """
    molecule, basis = h2
    for matrix in (
        overlap_matrix(basis),
        kinetic_matrix(basis),
        nuclear_matrix(basis, molecule),
    ):
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-13, rtol=0.0)


def test_overlap_is_positive_definite(h2):
    _, basis = h2
    assert np.all(np.linalg.eigvalsh(overlap_matrix(basis)) > 0)


def test_core_hamiltonian_is_sum_of_parts(h2):
    molecule, basis = h2
    np.testing.assert_allclose(
        core_hamiltonian(basis, molecule),
        kinetic_matrix(basis) + nuclear_matrix(basis, molecule),
        rtol=0.0,
        atol=1e-15,
    )


def test_overlap_falls_off_with_separation():
    close = build_basis(Molecule.from_string("H 0 0 0; H 0 0 0.5"), "sto-3g")
    far = build_basis(Molecule.from_string("H 0 0 0; H 0 0 3.0"), "sto-3g")
    assert overlap_matrix(close)[0, 1] > overlap_matrix(far)[0, 1]
    assert overlap_matrix(far)[0, 1] < 0.05


def test_eri_matches_pyscf(h2, h2_sto3g_integrals):
    _, basis = h2
    expected = np.array(h2_sto3g_integrals["eri"])
    np.testing.assert_allclose(eri_tensor(basis), expected, atol=1e-10, rtol=1e-10)


def test_eri_has_eightfold_symmetry(h2):
    """8-katli permutasyon simetrisi -- PySCF referansiyla ilgisi yok.

    eri_tensor ayni float degerini sekiz permutasyon konumuna da dogrudan
    atadigi icin (bkz. integrals.py, eri_tensor) fark yapisal olarak tam
    0.0'dir (bkz. b1-task-7-fix-report.md, Finding 1). atol=1e-13 kucuk bir
    pay birakiyor.
    """
    _, basis = h2
    g = eri_tensor(basis)
    for permuted in (
        np.transpose(g, (1, 0, 2, 3)),
        np.transpose(g, (0, 1, 3, 2)),
        np.transpose(g, (1, 0, 3, 2)),
        np.transpose(g, (2, 3, 0, 1)),
        np.transpose(g, (3, 2, 0, 1)),
        np.transpose(g, (2, 3, 1, 0)),
        np.transpose(g, (3, 2, 1, 0)),
    ):
        np.testing.assert_allclose(g, permuted, atol=1e-13, rtol=0.0)


def test_eri_diagonal_elements_are_positive(h2):
    _, basis = h2
    g = eri_tensor(basis)
    for i in range(g.shape[0]):
        assert g[i, i, i, i] > 0, "(ii|ii) elektron itmesi pozitif olmali"


def test_eri_cauchy_schwarz_bound(h2):
    """|(ab|cd)| <= sqrt((ab|ab)) * sqrt((cd|cd)) -- gercek bir ust sinir."""
    _, basis = h2
    g = eri_tensor(basis)
    n = g.shape[0]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    bound = np.sqrt(g[a, b, a, b]) * np.sqrt(g[c, d, c, d])
                    assert abs(g[a, b, c, d]) <= bound + 1e-12


@pytest.fixture(scope="module")
def water(hf_energies):
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    return molecule, build_basis(molecule, "sto-3g")


# Ayni tolerans kurali gecerli: bkz. bu dosyanin basindaki
# "PySCF karsilastirmalarinda tolerans: rtol ACIKCA verilmeli" notu (h2
# fixture'inin hemen ustunde). Su molekulunde buyuklukler H2'den cok daha
# genis oldugundan (cekirdek-cekim ~62 hartree'ye kadar cikiyor), tek
# istisna cekirdek-cekim testidir (test_water_nuclear_matches_pyscf) --
# gerekcesi o testin hemen ustunde.


def test_water_overlap_matches_pyscf(water, h2o_sto3g_integrals):
    _, basis = water
    expected = np.array(h2o_sto3g_integrals["overlap"])
    np.testing.assert_allclose(overlap_matrix(basis), expected, atol=1e-10, rtol=1e-10)


def test_water_kinetic_matches_pyscf(water, h2o_sto3g_integrals):
    _, basis = water
    expected = np.array(h2o_sto3g_integrals["kinetic"])
    np.testing.assert_allclose(kinetic_matrix(basis), expected, atol=1e-10, rtol=1e-10)


def test_water_nuclear_matches_pyscf(water, h2o_sto3g_integrals):
    """Cekirdek-cekim matrisi digerlerinden daha gevsek toleransla kontrol edilir.

    Sekiz PySCF karsilastirmasindaki artik hepsinde ayni kaynaktan gelir
    (bkz. dosyanin basindaki bohr-sabiti notu), ama cekirdek-cekim
    ozellikle duyarlidir: overlap, kinetik ve ERI integralleri geometriye
    yalnizca Gauss cift-merkezleri (product centers) uzerinden baglidir,
    oysa cekirdek-cekim her cekirdegi DOGRUDAN bir Coulomb merkezi olarak
    kullanir. Bu yuzden bohr-sabiti kaymasi cekirdek-cekimde digerlerinden
    daha guclu goruntur; ve oksijenin +8 yuku ile ~62 hartree'ye ulasan
    matris elemanlari, bu duyarliligin en cok ortaya ciktigi yerdir. Genel
    kuralla (atol=1e-10, rtol=1e-10) buradaki pay ~1.80x'e duser
    (kirilgan); bu yuzden bu test ozel olarak atol=3e-10, rtol=3e-10
    kullanir (pay ~5.4x, bkz. .superpowers/sdd/b1-task-7-fix2-report.md,
    Finding 1).
    """
    molecule, basis = water
    expected = np.array(h2o_sto3g_integrals["nuclear"])
    np.testing.assert_allclose(
        nuclear_matrix(basis, molecule), expected, atol=3e-10, rtol=3e-10
    )


def test_water_eri_matches_pyscf(water, h2o_sto3g_integrals):
    _, basis = water
    expected = np.array(h2o_sto3g_integrals["eri"])
    np.testing.assert_allclose(eri_tensor(basis), expected, atol=1e-10, rtol=1e-10)


def test_water_overlap_diagonal_is_unity(water):
    """Normalizasyon ozdesligi -- PySCF referansiyla ilgisi yok.

    Kosegen deger tanim geregi 1.0'dir; olculen fark 2.22e-16 (tek bir
    makine epsilonu, bkz. b1-task-7-fix-report.md, Finding 1). atol=1e-13
    bunu rahatca kapsar ve gorunurdeki eski atol=1e-10'un arkasindaki
    gizli rtol=1e-7 bosluguna kiyasla gercekten baglayicidir.
    """
    _, basis = water
    np.testing.assert_allclose(np.diag(overlap_matrix(basis)), 1.0, atol=1e-13, rtol=0.0)


def test_p_functions_on_same_atom_are_orthogonal(water):
    """Ayni merkezdeki p_x, p_y, p_z birbirine dik olmali."""
    _, basis = water
    overlap = overlap_matrix(basis)
    p_indices = [i for i, f in enumerate(basis) if sum(f.angular) == 1]
    assert len(p_indices) == 3
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert overlap[p_indices[i], p_indices[j]] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# PySCF ile kalan fark: bohr sabiti
# --------------------------------------------------------------------------

_PYSCF_BOHR_TO_ANGSTROM = 0.52917721092
"""PySCF'in param.BOHR degeri.

qlab CODATA-2018'i (units.BOHR_TO_ANGSTROM = 0.529177210903) kullanir; iki
sabit arasindaki bagil fark 3.2e-11'dir ve dogrudan cekirdek konumlarina
gecer. Bu yuzden yukaridaki testler ~5e-11 bagil fark birakir. Referans
uretilirken kullanilan sabitle ayni geometride hesaplanirsa fark makine
hassasiyetine iner -- asagidaki test tam olarak bunu sabitler.
"""


def _molecule_at_pyscf_bohr(spec: str) -> Molecule:
    atoms = []
    for entry in spec.split(";"):
        symbol, *coordinates = entry.split()
        position = np.array([float(v) for v in coordinates]) / _PYSCF_BOHR_TO_ANGSTROM
        atoms.append(Atom(symbol, position))
    return Molecule(tuple(atoms))


def test_water_matches_pyscf_to_machine_precision_at_pyscf_geometry(
    hf_energies, h2o_sto3g_integrals
):
    """Ayni bohr sabitiyle kurulan geometride dort tensor de ~1e-14'te esler.

    Bu test, yukaridaki 1e-10 seviyesindeki artigin ozyinelemeden DEGIL birim
    sabitinden geldigini kanitlar ve ozyinelemeyi 1e-10 yerine 1e-12'de
    kilitler: acisal momentum genislemesinde ileride olusacak bir hata,
    gevsek toleransin arkasina saklanamaz.
    """
    molecule = _molecule_at_pyscf_bohr(hf_energies["h2o"]["atom"])
    basis = build_basis(molecule, "sto-3g")
    computed = {
        "overlap": overlap_matrix(basis),
        "kinetic": kinetic_matrix(basis),
        "nuclear": nuclear_matrix(basis, molecule),
        "eri": eri_tensor(basis),
    }
    for name, got in computed.items():
        np.testing.assert_allclose(
            got,
            np.array(h2o_sto3g_integrals[name]),
            rtol=0.0,
            atol=1e-12,
            err_msg=f"{name} tensoru makine hassasiyetinde eslesmedi",
        )


# --------------------------------------------------------------------------
# l >= 2: su/STO-3G'nin ulasamadigi ozyineleme dallari
# --------------------------------------------------------------------------
#
# build_basis d kabugunu bilincli olarak reddediyor (basis.py, spec bolum 13),
# ama integrals.py'deki MD ozyinelemesi acisal momentumda geneldir. Su
# molekulu l<=1 kaldigi icin iki sayisal yolu HIC calistirmaz:
#   * kinetik enerjideki -1/2 j(j-1) S(i, j-2) terimi (yalniz j>=2'de sifirdan
#     farkli),
#   * _Axis.e'nin i>=2 / j>=2 adimlari.
# Asagidaki testler BasisFunction'i dogrudan kurarak bu yollari kapsar ve
# degerleri scipy.integrate.quad ile bagimsiz olarak dogrular.


def _contracted_weights(function: BasisFunction) -> list[float]:
    """d_i * N_i * kontraksiyon-normalizasyonu (basis.py'nin genel formulleri)."""
    scale = function.contraction_normalisation()
    return [
        p.coefficient * n * scale
        for p, n in zip(function.primitives, function.normalisation())
    ]


def _axis_overlap(i, j, alpha, beta, ax, bx) -> float:
    """integral( x_A^i e^{-alpha x_A^2} * x_B^j e^{-beta x_B^2} ) dx."""
    def integrand(x):
        return (
            (x - ax) ** i * np.exp(-alpha * (x - ax) ** 2)
            * (x - bx) ** j * np.exp(-beta * (x - bx) ** 2)
        )

    return quad(integrand, -np.inf, np.inf)[0]


def _axis_gradient_overlap(i, j, alpha, beta, ax, bx) -> float:
    """integral( d/dx[x_A^i e^{-alpha x_A^2}] * d/dx[x_B^j e^{-beta x_B^2}] ) dx."""
    def derivative(x, l, gamma, c):
        u = x - c
        power = l * u ** (l - 1) if l > 0 else 0.0
        return (power - 2 * gamma * u ** (l + 1)) * np.exp(-gamma * u * u)

    def integrand(x):
        return derivative(x, i, alpha, ax) * derivative(x, j, beta, bx)

    return quad(integrand, -np.inf, np.inf)[0]


def _numeric_overlap(a: BasisFunction, b: BasisFunction) -> float:
    total = 0.0
    for pa, wa in zip(a.primitives, _contracted_weights(a)):
        for pb, wb in zip(b.primitives, _contracted_weights(b)):
            value = wa * wb
            for k in range(3):
                value *= _axis_overlap(
                    a.angular[k], b.angular[k],
                    pa.exponent, pb.exponent, a.center[k], b.center[k],
                )
            total += value
    return total


def _numeric_kinetic(a: BasisFunction, b: BasisFunction) -> float:
    """T_ab = 1/2 integral(grad a . grad b) -- kismi integrasyon ozdesligi.

    integrals.py kete -1/2 d^2/dx^2 uygular; buradaki yol yalnizca BIRINCI
    turevleri kullanir, yani sinanan cebiri paylasmaz.
    """
    total = 0.0
    for pa, wa in zip(a.primitives, _contracted_weights(a)):
        for pb, wb in zip(b.primitives, _contracted_weights(b)):
            s = [
                _axis_overlap(a.angular[k], b.angular[k], pa.exponent,
                              pb.exponent, a.center[k], b.center[k])
                for k in range(3)
            ]
            g = [
                _axis_gradient_overlap(a.angular[k], b.angular[k], pa.exponent,
                                       pb.exponent, a.center[k], b.center[k])
                for k in range(3)
            ]
            total += 0.5 * wa * wb * (
                g[0] * s[1] * s[2] + s[0] * g[1] * s[2] + s[0] * s[1] * g[2]
            )
    return total


def _make_function(center, angular) -> BasisFunction:
    """build_basis d'yi reddettigi icin kabuk elle kurulur (iki primitif)."""
    return BasisFunction(
        center=np.array(center, dtype=float),
        angular=angular,
        primitives=(
            PrimitiveGaussian(exponent=1.9, coefficient=0.42),
            PrimitiveGaussian(exponent=0.55, coefficient=0.71),
        ),
        atom_index=0,
    )


_HIGH_ANGULAR_PAIRS = [
    ((2, 0, 0), (0, 0, 0)),   # i>=2, j=0 -> _Axis.e'nin i-azaltma dali
    ((0, 0, 0), (0, 0, 2)),   # j>=2 -> kinetikte j(j-1) terimi, i=0 ile
    ((2, 0, 0), (1, 0, 0)),
    ((0, 2, 0), (0, 1, 0)),
    ((0, 0, 2), (0, 0, 2)),   # her iki tarafta da l=2
    ((1, 1, 0), (1, 0, 1)),
    ((2, 0, 0), (0, 2, 0)),
]


@pytest.mark.parametrize(("angular_a", "angular_b"), _HIGH_ANGULAR_PAIRS)
def test_overlap_at_l2_matches_quadrature(angular_a, angular_b):
    a = _make_function((0.0, 0.0, 0.0), angular_a)
    b = _make_function((0.35, -0.6, 0.9), angular_b)
    computed = overlap_matrix([a, b])[0, 1]
    assert computed == pytest.approx(_numeric_overlap(a, b), rel=1e-8, abs=1e-12)


@pytest.mark.parametrize(("angular_a", "angular_b"), _HIGH_ANGULAR_PAIRS)
def test_kinetic_at_l2_matches_quadrature(angular_a, angular_b):
    a = _make_function((0.0, 0.0, 0.0), angular_a)
    b = _make_function((0.35, -0.6, 0.9), angular_b)
    computed = kinetic_matrix([a, b])[0, 1]
    assert computed == pytest.approx(_numeric_kinetic(a, b), rel=1e-8, abs=1e-12)


# --------------------------------------------------------------------------
# dipole_matrices (Task 10): quadrature check + origin-argument coverage
# --------------------------------------------------------------------------
#
# analysis.py (Task 10) always calls dipole_matrices(basis, origin=(0,0,0)) --
# for a neutral molecule the resulting dipole MOMENT is origin-independent,
# so that call site never exercises a nonzero origin. A sign error or a
# dropped origin term in _dipole_primitive_block would therefore pass every
# analysis.py-level test undetected. The two tests below close that gap
# directly at the integrals.py level, with a deliberately nonzero origin.

_DIPOLE_ORIGIN = np.array([0.2, -0.4, 1.1])


def _axis_moment(i, j, alpha, beta, ax, bx, origin_k) -> float:
    """integral( (x-origin_k) * x_A^i e^{-alpha x_A^2} * x_B^j e^{-beta x_B^2} ) dx."""
    def integrand(x):
        return (
            (x - origin_k)
            * (x - ax) ** i * np.exp(-alpha * (x - ax) ** 2)
            * (x - bx) ** j * np.exp(-beta * (x - bx) ** 2)
        )

    return quad(integrand, -np.inf, np.inf)[0]


def _numeric_dipole(a: BasisFunction, b: BasisFunction, origin: np.ndarray) -> np.ndarray:
    """<a|(r-origin)|b>, sekil (3,) -- her eksen icin bagimsiz sayisal integrasyon."""
    result = np.zeros(3)
    for pa, wa in zip(a.primitives, _contracted_weights(a)):
        for pb, wb in zip(b.primitives, _contracted_weights(b)):
            for k in range(3):
                moment = _axis_moment(
                    a.angular[k], b.angular[k], pa.exponent, pb.exponent,
                    a.center[k], b.center[k], origin[k],
                )
                other_axes = 1.0
                for m in range(3):
                    if m == k:
                        continue
                    other_axes *= _axis_overlap(
                        a.angular[m], b.angular[m], pa.exponent, pb.exponent,
                        a.center[m], b.center[m],
                    )
                result[k] += wa * wb * moment * other_axes
    return result


@pytest.mark.parametrize(("angular_a", "angular_b"), _HIGH_ANGULAR_PAIRS)
def test_dipole_at_l2_matches_quadrature_with_nonzero_origin(angular_a, angular_b):
    a = _make_function((0.0, 0.0, 0.0), angular_a)
    b = _make_function((0.35, -0.6, 0.9), angular_b)
    computed = np.array(
        [component[0, 1] for component in dipole_matrices([a, b], _DIPOLE_ORIGIN)]
    )
    expected = _numeric_dipole(a, b, _DIPOLE_ORIGIN)
    np.testing.assert_allclose(computed, expected, rtol=1e-8, atol=1e-12)


def test_dipole_matrices_shift_identity(water):
    """<mu|(r-C)|nu> = <mu|r|nu> - C*<mu|nu>, yani D(C) = D(0) - C*S.

    Su/STO-3G'de (s VE p fonksiyonlari, gercek uretim olcegi) origin C=0'dan
    kasten farkli secilerek dogrudan sinanir -- bu ozdeslik yalniz origin
    terimi dogru tasindiginda tutar; isareti ters veya eksik bir origin
    terimi bu testi kirar.
    """
    _, basis = water
    zero_x, zero_y, zero_z = dipole_matrices(basis, np.zeros(3))
    shifted_x, shifted_y, shifted_z = dipole_matrices(basis, _DIPOLE_ORIGIN)
    overlap = overlap_matrix(basis)
    for shifted, zero, origin_k in (
        (shifted_x, zero_x, _DIPOLE_ORIGIN[0]),
        (shifted_y, zero_y, _DIPOLE_ORIGIN[1]),
        (shifted_z, zero_z, _DIPOLE_ORIGIN[2]),
    ):
        np.testing.assert_allclose(
            shifted, zero - origin_k * overlap, rtol=0.0, atol=1e-10
        )
