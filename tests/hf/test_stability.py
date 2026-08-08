"""RHF ic kararlilik analizi (internal, RHF -> RHF) testleri.

Sinanan iddia: yakinsamis bir RHF cozumu enerjinin bir DURAGAN noktasidir,
ama minimum olmak zorunda DEGILDIR. Isgal-bos (occupied-virtual) orbital
dondurmelerine gore ikinci turev matrisinin (tekil/singlet orbital Hessian)
negatif bir ozdegeri varsa, yakindaki daha dusuk enerjili bir RHF cozumu
vardir ve raporlanan sonuc bir EYER NOKTASIDIR.

Bu dosyanin TASIYICI dogrulamasi, analitik Hessian'in sonlu farkla
karsilastirilmasidir (bkz. test_hessian_matches_finite_difference_*).
Depoda PySCF kararlilik referansi YOKTUR; referans uretilemedigi icin
"bagimsiz sayisal kontrol + fiziksel kabul durumlari" spesifikasyonu
uygulanir -- ayni desen Boys fonksiyonunda (scipy.integrate.quad) ve
kontraksiyon normalizasyonunda (eksen-bazli kuadratur) kullanilmisti.

Parametrelendirme (sonlu fark ile analitik ifade AYNI olmak zorunda):

    C(theta) = C exp(K),   K_ai = theta_ai,  K_ia = -theta_ai
               (a bos, i isgal; K'nin isgal-isgal ve bos-bos bloklari sifir)

Yani theta_ai, i. isgal orbitalini a. bos orbitale dogru donduren aciDIR.
Hessian H_[ia],[jb] = d^2 E / d theta_ai d theta_bj (theta = 0'da).
Satir/sutun sirasi: index = i * n_virtual + (a - n_occupied).

pytest.approx BUYUK olani alir (max(abs, rel*expected)); bu dosyadaki her
iddiada IKI parametre de acikca verilir ve sinirlar OLCULMUSTUR.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from qlab.engine.hf.analysis import stability
from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import core_hamiltonian, eri_tensor
from qlab.engine.hf.scf import solve
from qlab.engine.hf.stability import (
    StabilityResult,
    analyse_stability,
    orbital_hessian,
)
from qlab.types import Molecule

# --- Sonlu fark makinesi (uretim kodundan BAGIMSIZ yazilmistir) -----------


def _integrals(molecule: Molecule, basis_name: str):
    basis = build_basis(molecule, basis_name)
    return (
        core_hamiltonian(basis, molecule),
        eri_tensor(basis),
        molecule.nuclear_repulsion(),
    )


def _rhf_energy(core, eri, nuclear, coefficients, n_occupied) -> float:
    """RHF toplam enerjisi, VERILEN katsayilardan (SCF kosulmaz).

    E = 1/2 sum_uv P_uv (Hcore_uv + F_uv) + E_cekirdek,  P = 2 C_occ C_occ^T
    """
    occupied = coefficients[:, :n_occupied]
    density = 2.0 * occupied @ occupied.T
    coulomb = np.einsum("uvls,ls->uv", eri, density)
    exchange = np.einsum("ulvs,ls->uv", eri, density)
    fock = core + coulomb - 0.5 * exchange
    return 0.5 * float(np.einsum("uv,uv->", density, core + fock)) + nuclear


def _rotate(coefficients, n_occupied, direction, t):
    """C exp(t K(direction)); direction, duzlestirilmis theta vektorudur."""
    n_mo = coefficients.shape[1]
    n_virtual = n_mo - n_occupied
    k = np.zeros((n_mo, n_mo))
    for index, value in enumerate(direction):
        i, a = divmod(index, n_virtual)
        a += n_occupied
        k[a, i] += t * value
        k[i, a] -= t * value
    return coefficients @ expm(k)


def _second_derivative_along(core, eri, nuclear, coefficients, n_occupied, direction, h):
    """d^2 E / dt^2, t=0'da, bes noktali (O(h^4)) merkezi fark."""

    def energy(t):
        return _rhf_energy(
            core, eri, nuclear, _rotate(coefficients, n_occupied, direction, t), n_occupied
        )

    return (
        -energy(2 * h)
        + 16 * energy(h)
        - 30 * energy(0.0)
        + 16 * energy(-h)
        - energy(-2 * h)
    ) / (12 * h * h)


def _first_derivative_along(core, eri, nuclear, coefficients, n_occupied, direction, h):
    """dE/dt, t=0'da, bes noktali (O(h^4)) merkezi fark."""

    def energy(t):
        return _rhf_energy(
            core, eri, nuclear, _rotate(coefficients, n_occupied, direction, t), n_occupied
        )

    return (
        energy(-2 * h) - 8 * energy(-h) + 8 * energy(h) - energy(2 * h)
    ) / (12 * h)


def _finite_difference_hessian(core, eri, nuclear, coefficients, n_occupied, h):
    """Tam Hessian, yalnizca YONLU ikinci turevlerden kurulur.

    Kosegen: H_pp = D^2(e_p).
    Kosegen disi: H_pq = 1/2 [ D^2(e_p + e_q) - D^2(e_p) - D^2(e_q) ]
    (H simetrik oldugu icin bu ozdeslik tamdir). Boylece kosegen disi
    elemanlar da bes noktali O(h^4) dogrulukla olculur; klasik dort noktali
    capraz fark yalnizca O(h^2) verirdi.
    """
    n_rotations = n_occupied * (coefficients.shape[1] - n_occupied)
    basis_vectors = np.eye(n_rotations)

    diagonal = np.array(
        [
            _second_derivative_along(
                core, eri, nuclear, coefficients, n_occupied, basis_vectors[p], h
            )
            for p in range(n_rotations)
        ]
    )

    hessian = np.diag(diagonal)
    for p in range(n_rotations):
        for q in range(p + 1, n_rotations):
            both = _second_derivative_along(
                core,
                eri,
                nuclear,
                coefficients,
                n_occupied,
                basis_vectors[p] + basis_vectors[q],
                h,
            )
            hessian[p, q] = hessian[q, p] = 0.5 * (both - diagonal[p] - diagonal[q])
    return hessian


# --- 1. Tasiyici dogrulama: analitik Hessian == sonlu fark ----------------

# Adim boyu ve toleranslar OLCULMUSTUR (h taramasi:
# .superpowers/sdd/b1-stability-report.md). h buyudukce hata h^4 gibi buyur
# (0.01 -> 0.04, yani 4 kat h, hatayi 256 kat buyutur -- olculdu), h kucukken
# enerjideki yuvarlama h^2'ye bolunerek buyur. h = 5e-3 iki sistem icin de bu
# iki kolun kesisimine yakindir:
#     H2  (1x1) : max|analitik - sayisal| = 2.758e-09  (goreli 6.1e-10)
#     H2O (10x10): max|analitik - sayisal| = 1.111e-08  (goreli 1.4e-10)
# Sinirlar olculen degerin ~3.6 katinda baglayici tutulur.
_FD_H = 5e-3
_FD_TOLERANCE_H2 = 1e-8
_FD_TOLERANCE_H2O = 4e-8

# Gradyanin sifirdan sapmasi sonlu fark hatasi DEGIL, SCF'nin kendi yakinsama
# esigidir: dE/dtheta_ai = 4 F_ai(MO) ve solve() bu blogu ancak
# residual_tolerance (varsayilan 1e-8) mertebesinde sifirlar. Olculen max
# 2.462e-09, yani esigin ~4 kat altinda; sinir esigin kendisinde tutulur.
_GRADIENT_TOLERANCE = 1e-8


def test_hessian_matches_finite_difference_h2():
    """H2/STO-3G: tek isgal, tek bos orbital -> 1x1 Hessian.

    Konvansiyon hatasi (2 veya 4 kati) tam da burada yakalanir: analitik
    deger ile sayisal deger arasindaki oran 1.0 degilse, tureve giren
    parametrelendirme uyusmuyor demektir.
    """
    molecule = Molecule.from_string("H 0 0 0; H 0 0 0.74")
    result = solve(molecule, "sto-3g")
    core, eri, nuclear = _integrals(molecule, "sto-3g")
    n_occupied = molecule.n_electrons // 2

    analytic = orbital_hessian(core, eri, result.coefficients, n_occupied)
    numerical = _finite_difference_hessian(
        core, eri, nuclear, result.coefficients, n_occupied, _FD_H
    )

    assert analytic.shape == (1, 1)
    print(
        f"\nH2 0.74 A  analitik H = {analytic[0, 0]:.12f}  "
        f"sayisal H = {numerical[0, 0]:.12f}  "
        f"fark = {abs(analytic[0, 0] - numerical[0, 0]):.3e}  "
        f"oran = {analytic[0, 0] / numerical[0, 0]:.12f}"
    )
    np.testing.assert_allclose(
        analytic, numerical, atol=_FD_TOLERANCE_H2, rtol=0.0
    )


def test_hessian_matches_finite_difference_water(hf_energies):
    """H2O/STO-3G: 5 isgal, 2 bos -> 10x10, p fonksiyonlari ve kosegen
    disi elemanlar dahil. 1x1 durumunda gorunmeyen indeks/transpoz
    hatalari burada ortaya cikar."""
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    result = solve(molecule, "sto-3g")
    core, eri, nuclear = _integrals(molecule, "sto-3g")
    n_occupied = molecule.n_electrons // 2

    analytic = orbital_hessian(core, eri, result.coefficients, n_occupied)
    numerical = _finite_difference_hessian(
        core, eri, nuclear, result.coefficients, n_occupied, _FD_H
    )

    assert analytic.shape == (10, 10)
    deviation = np.abs(analytic - numerical)
    off_diagonal = deviation[~np.eye(10, dtype=bool)]
    print(
        f"\nH2O analitik-sayisal Hessian: max fark = {deviation.max():.3e}, "
        f"kosegen disi max fark = {off_diagonal.max():.3e}, "
        f"|H| araligi = [{np.abs(analytic).min():.3e}, {np.abs(analytic).max():.3e}]"
    )
    np.testing.assert_allclose(
        analytic, numerical, atol=_FD_TOLERANCE_H2O, rtol=0.0
    )


def test_gradient_vanishes_at_converged_solution(hf_energies):
    """Yakinsamis cozum DURAGAN olmali: her dondurme yonunde birinci turev 0.

    Bu, hem SCF'nin gercekten duragan bir noktaya indigini hem de sonlu
    fark makinesinin dogru parametreyi turevledigini gosterir -- Hessian
    karsilastirmasinin on kosuludur.
    """
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    result = solve(molecule, "sto-3g")
    core, eri, nuclear = _integrals(molecule, "sto-3g")
    n_occupied = molecule.n_electrons // 2
    n_rotations = n_occupied * (result.coefficients.shape[1] - n_occupied)

    gradient = np.array(
        [
            _first_derivative_along(
                core, eri, nuclear, result.coefficients, n_occupied, direction, _FD_H
            )
            for direction in np.eye(n_rotations)
        ]
    )
    print(f"\nH2O sayisal gradyan max |dE/dtheta| = {np.abs(gradient).max():.3e}")
    np.testing.assert_allclose(gradient, 0.0, atol=_GRADIENT_TOLERANCE, rtol=0.0)


def test_hessian_is_symmetric(hf_energies):
    """H_[ia],[jb] = H_[jb],[ia] -- ikinci turevin simetrisi.

    Kod simetrilestirmez (0.5*(H+H^T) yapmaz); simetri, (pq|rs) integral
    simetrilerinden ANALITIK olarak gelir ve burada sayisal olarak
    dogrulanir. eigvalsh yalnizca alt ucgeni okudugu icin, sessiz bir
    asimetri hukmu bozmadan gecebilirdi.

    Olculen sapma (su/STO-3G): 6.05e-15, en buyuk eleman |H| = 80.7 --
    yani eps*||H|| ~ 1.8e-14 mertebesinde, tam beklenen yuvarlama.
    Sinir 1e-13, olculenin ~16 kati.
    """
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    result = solve(molecule, "sto-3g")
    core, eri, _ = _integrals(molecule, "sto-3g")
    hessian = orbital_hessian(
        core, eri, result.coefficients, molecule.n_electrons // 2
    )
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-13, rtol=0.0)


# --- 2. Fiziksel kabul durumlari ------------------------------------------


def test_h2_equilibrium_is_stable():
    result = solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g")
    print(f"\nH2 0.74 A en dusuk ozdeger = {result.stability.lowest_eigenvalue:.6f}")
    assert result.stability.stable is True
    assert result.stability.lowest_eigenvalue > 0.0
    # Olculmus deger kilitlenir: degisirse ya Hessian ya SCF cozumu degismistir.
    assert result.stability.lowest_eigenvalue == pytest.approx(
        4.518469, abs=1e-5, rel=0.0
    )


def test_water_reference_geometry_is_stable(hf_energies):
    result = solve(Molecule.from_string(hf_energies["h2o"]["atom"]), "sto-3g")
    print(f"\nH2O en dusuk ozdeger = {result.stability.lowest_eigenvalue:.6f}")
    assert result.stability.stable is True
    assert result.stability.lowest_eigenvalue > 0.0
    assert result.stability.lowest_eigenvalue == pytest.approx(
        2.092671, abs=1e-5, rel=0.0
    )


@pytest.mark.slow
def test_h2_at_10_angstrom_is_stable():
    """10 A'da SCF hala DOGRU (simetrik) cozumu buluyor -- bu sinir
    test_physical.py::test_large_separation_breaks_down_numerically_and_we_say_so
    tarafindan kilitlenmistir. Kararlilik analizi o sinirla AYNI FIKIRDE
    olmak zorundadir: 10 A stabil."""
    result = solve(Molecule.from_string("H 0 0 0; H 0 0 10.0"), "sto-3g")
    print(
        f"\nH2 10 A  E = {result.energy_total:.6f}  "
        f"en dusuk ozdeger = {result.stability.lowest_eigenvalue:.6f}"
    )
    assert result.stability.stable is True
    assert result.stability.lowest_eigenvalue > 0.0
    assert result.stability.lowest_eigenvalue == pytest.approx(
        2.886753, abs=1e-5, rel=0.0
    )


@pytest.mark.slow
def test_h2_at_12_angstrom_is_an_unreported_saddle_point():
    """OZELLIGIN SEBEBI: 12 A'da SCF yerellesmis (H- H+) cozume yakinsar.

    O cozum gercek bir duragan noktadir (converged=True yalan degildir) ama
    minimum DEGILDIR: isgal orbitalini bos orbitale dogru dondurmek enerjiyi
    DUSURUR. Kararlilik analizi olmadan bu bilgi cikti da HICBIR YERDE
    gorunmuyordu.
    """
    result = solve(Molecule.from_string("H 0 0 0; H 0 0 12.0"), "sto-3g")
    print(
        f"\nH2 12 A  E = {result.energy_total:.6f}  "
        f"en dusuk ozdeger = {result.stability.lowest_eigenvalue:.6f}"
    )
    assert result.stability.stable is False
    assert result.stability.lowest_eigenvalue < 0.0
    # Kararsizlik ihmal edilebilir bir sayisal gurultu DEGIL: olculen deger
    # -2.922031. Bu sayi -- kararli durumlarin aksine -- TAM olarak
    # kilitlenmez, cunku 12 A'daki cozumun kendisi eigh'in dejenere
    # altuzayda yaptigi KEYFI secime baglidir (bkz. test_physical.py) ve
    # LAPACK surumune gore kayabilir. Kilitlenen sey isaret ve buyukluk
    # mertebesidir: sinir olculenin ~3'te 1'inde.
    assert result.stability.lowest_eigenvalue < -1.0


@pytest.mark.slow
def test_unstable_solution_still_reports_converged_true():
    """Kararsizlik bir HATA DEGILDIR: cozum gecerli bir duragan noktadir ve
    kullanici onu mesru olarak isteyebilir. Duzeltilen kusur SESSIZLIKTIR,
    sonucun kendisi degil -- solve() bu yuzden yukseltmez ve converged
    anlamini korur."""
    result = solve(Molecule.from_string("H 0 0 0; H 0 0 12.0"), "sto-3g")
    assert result.provenance.converged is True
    assert result.stability.stable is False


@pytest.mark.slow
def test_rotating_along_the_negative_mode_lowers_the_energy():
    """Negatif ozdegerin ANLAMI: o yonde dondurmek enerjiyi gercekten
    dusurmeli. Bu, "negatif ozdeger = eyer noktasi" yorumunun dogrudan
    fiziksel kanitidir -- isaret hatasi olsaydi enerji YUKSELIRDI."""
    molecule = Molecule.from_string("H 0 0 0; H 0 0 12.0")
    result = solve(molecule, "sto-3g")
    core, eri, nuclear = _integrals(molecule, "sto-3g")
    n_occupied = molecule.n_electrons // 2

    hessian = orbital_hessian(core, eri, result.coefficients, n_occupied)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    direction = eigenvectors[:, 0]

    reference = _rhf_energy(core, eri, nuclear, result.coefficients, n_occupied)
    rotated = _rhf_energy(
        core,
        eri,
        nuclear,
        _rotate(result.coefficients, n_occupied, direction, 0.1),
        n_occupied,
    )
    print(
        f"\nH2 12 A negatif mod boyunca: E(0) = {reference:.6f} -> "
        f"E(0.1 rad) = {rotated:.6f} (fark {rotated - reference:.6f})"
    )
    assert eigenvalues[0] < 0.0
    assert rotated < reference


# --- 3. Kenar durumlar ve API ---------------------------------------------


def test_system_without_virtual_orbitals_is_trivially_stable(hf_energies):
    """He/STO-3G: tek baz fonksiyonu, tek isgal orbital, SIFIR bos orbital.

    Dondurulebilecek hicbir cift yoktur -> Hessian 0x0'dir. Bu dal
    ozdeger cagrisina hic girmemeli (np.linalg.eigvalsh bos matriste bos
    dizi doner ve [0] indeksleme IndexError verirdi) ve "kararli, ama
    ozdeger yok" seklinde durust raporlanmalidir.
    """
    result = solve(Molecule.from_string(hf_energies["he"]["atom"]), "sto-3g")
    assert result.stability.n_rotations == 0
    assert result.stability.stable is True
    assert result.stability.lowest_eigenvalue is None
    assert result.stability.eigenvalues.shape == (0,)


def test_hessian_is_empty_when_there_are_no_occupied_orbitals():
    """orbital_hessian'in ikinci koruma dali (n_occupied == 0).

    Bu, Molecule uzerinden ULASILAMAZ bir durumdur (elektronsuz sistem
    cozulmez) ama fonksiyon toplam olsun diye korunur; testsiz birakilirsa
    "hic kosulmamis dal" olur. Dogrudan cagirilarak kapatilir.
    """
    core = np.eye(2)
    eri = np.zeros((2, 2, 2, 2))
    coefficients = np.eye(2)
    assert orbital_hessian(core, eri, coefficients, 0).shape == (0, 0)


def test_stability_can_be_turned_off_and_says_what_is_skipped():
    result = solve(
        Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g", stability=False
    )
    assert result.stability is None
    assert result.provenance.stability is None


def test_provenance_carries_machine_readable_verdict():
    result = solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g")
    verdict = result.provenance.stability
    assert verdict is not None
    assert verdict.stable is True
    assert verdict.lowest_eigenvalue == pytest.approx(
        result.stability.lowest_eigenvalue, abs=0.0, rel=0.0
    )
    assert verdict.n_rotations == result.stability.n_rotations


@pytest.mark.slow
def test_provenance_line_names_the_saddle_point():
    result = solve(Molecule.from_string("H 0 0 0; H 0 0 12.0"), "sto-3g")
    line = dict(result.provenance.as_lines())["Kararlılık"]
    print(f"\nPanel satiri: {line}")
    assert line.startswith("✗")
    assert "eyer" in line.lower()


def test_neglected_list_says_only_internal_stability_is_tested():
    """"Kararli" hukmu oldugundan GENIS okunabilir: bu analiz cozumun kuresel
    minimum oldugunu veya UHF'in daha dusuk enerji vermeyecegini soylemez.
    Sinir, panelin "neyi hesaba katmadim" satirinda yazili olmali -- ve bu
    metin scf.py'deki TEK kaynaktan (_HF_NEGLECTED) gelmeli, ikinci bir
    kopyadan degil (bkz. test_hf_scenario.py, tek-kaynak testi).
    """
    result = solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g")
    ihmal = " ".join(result.provenance.neglected).lower()
    assert "uhf" in ihmal
    assert "yalnizca ic" in ihmal


def test_analysis_helper_reproduces_the_result_attached_by_solve(hf_energies):
    """analysis.stability(result), solve()'un ilistirdigi ile ayni sonucu
    vermeli -- stability=False ile kosanlar icin sonradan hesaplama yolu."""
    molecule = Molecule.from_string(hf_energies["h2o"]["atom"])
    attached = solve(molecule, "sto-3g").stability
    recomputed = stability(solve(molecule, "sto-3g", stability=False))
    assert isinstance(recomputed, StabilityResult)
    np.testing.assert_allclose(
        recomputed.eigenvalues, attached.eigenvalues, atol=1e-12, rtol=0.0
    )
    assert recomputed.stable is attached.stable


def test_analyse_stability_flags_a_hand_built_negative_direction():
    """analyse_stability'nin karar esigi, dogrudan (SCF olmadan) sinanir:
    kucuk negatif bir ozdeger 'kararsiz' sayilmali, sifira cok yakin
    pozitif/negatif gurultu ise sayilmamalidir.

    Bu, esik dalinin (lowest > -tolerance) her iki tarafini da kapsar;
    fiziksel testler yalnizca |ozdeger| ~ 1 mertebesindeki durumlara
    deger, esigin yakinini hic yoklamaz.
    """
    from qlab.engine.hf.stability import EIGENVALUE_TOLERANCE

    noise = np.diag([-EIGENVALUE_TOLERANCE / 10, 1.0])
    genuine = np.diag([-EIGENVALUE_TOLERANCE * 10, 1.0])

    assert analyse_stability(noise).stable is True
    assert analyse_stability(genuine).stable is False

    # Esik bir PARAMETREDIR ve gecirildiginde gercekten kullanilmalidir --
    # varsayilanla ayni sonucu veren bir "gecis parametresi" sessizce
    # yok sayilabilirdi.
    assert analyse_stability(noise, tolerance=EIGENVALUE_TOLERANCE / 100).stable is False
    assert analyse_stability(genuine, tolerance=EIGENVALUE_TOLERANCE * 100).stable is True
