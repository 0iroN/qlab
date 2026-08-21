"""Dis (RHF -> UHF, uclu) kararlilik. Depo bugune kadar YALNIZ singlet bakiyordu.

Bu dosyanin TASIYICI dogrulamasi, analitik triplet Hessian'in sonlu farkla
karsilastirilmasidir. Sonlu fark makinesi uretim kodundan BAGIMSIZ yazilmistir;
paylasilan tek sey integraller ve SCF katsayilaridir.

Parametrelendirme (sonlu fark ile analitik ifade AYNI olmak zorunda):

    Ca(t) = C exp(+tK),  Cb(t) = C exp(-tK),  K_ai = theta_ai,  K_ia = -theta_ai

Ayni sonlu fark makinesi alfa ile beta AYNI yonde dondurulunce deponun mevcut
SINGLET Hessian'ini verir -- iki kanalin gercekten farkli seyler olctugunun
AYIRICI kontrolu budur ve bu dosyada kilitlidir.

SINIF UYARISI -- "lam_triplet < 0 <=> E_UHF < E_RHF" bir CIPA DEGILDIR.
Hessian, SCF'in indigi AYNI fonksiyonelin ikinci turevidir, dolayisiyla
lam < 0 "yakinda daha dusuk bir UHF cozumu var" demektir; bu cebir, fizik
degil. SAHTE-CIPA SINAVI KOSULDU: fizige ait olmayan (dogru simetrileri olan
ama fizik-disi rastgele) integrallerde hukum 51/53 tutuyor -- ve ayni sinavda
YANLIS kanal (singlet) da 50/53 tutuyor, yani sahte girdi iki kanali
AYIRAMIYOR.

Ayirim GERCEK sistemlerde ortaya cikiyor ve olculdu: H2'de 18 ayri R'de
triplet olcutu 18/18, singlet olcutu 8/18. Yani hukum fizik degil ama
DOGRU CEBIR: yanlis kanalla verildiginde gercek molekullerde 10/18 dusuyor.
Iki soru ayridir ve iki ayri cevabi vardir; bu dosya ikisini de yaziyor.

Asil FIZIK, catallanma noktasinin KENDISIDIR ve o ayri dosyada:
tests/hf/test_coulson_fischer.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from qlab.engine.hf.analysis import external_stability
from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import core_hamiltonian, eri_tensor
from qlab.engine.hf.multistart import search_uhf
from qlab.engine.hf.scf import solve
from qlab.engine.hf.stability import (
    StabilityResult,
    orbital_hessian,
    triplet_orbital_hessian,
)
from qlab.types import Molecule
from qlab.units import BOHR_TO_ANGSTROM


def _h2(r_bohr: float) -> Molecule:
    yari = r_bohr * BOHR_TO_ANGSTROM / 2
    return Molecule.from_string(f"H 0 0 {-yari}; H 0 0 {yari}")


def _zincir(n: int, r_bohr: float) -> Molecule:
    adim = r_bohr * BOHR_TO_ANGSTROM
    return Molecule.from_string("; ".join(f"H 0 0 {i * adim:.17g}" for i in range(n)))


def _integraller(molecule: Molecule, basis_name: str):
    basis = build_basis(molecule, basis_name)
    return (
        core_hamiltonian(basis, molecule),
        eri_tensor(basis),
        molecule.nuclear_repulsion(),
    )


# --- Sonlu fark makinesi (uretim kodundan BAGIMSIZ yazilmistir) -----------


def _uhf_energy(core, eri, nuclear, c_alpha, c_beta, n_occupied) -> float:
    """UHF toplam enerjisi, VERILEN katsayilardan (SCF kosulmaz).

    Pa = Ca_occ Ca_occ^T (spin orbitali basina isgal 1), Pt = Pa + Pb,
    J TOPLAM yogunluktan, K her spin icin AYRI.
    """
    p_alpha = c_alpha[:, :n_occupied] @ c_alpha[:, :n_occupied].T
    p_beta = c_beta[:, :n_occupied] @ c_beta[:, :n_occupied].T
    p_total = p_alpha + p_beta
    coulomb = np.einsum("uvls,ls->uv", eri, p_total)
    fock_alpha = core + coulomb - np.einsum("ulvs,ls->uv", eri, p_alpha)
    fock_beta = core + coulomb - np.einsum("ulvs,ls->uv", eri, p_beta)
    return 0.5 * float(
        np.sum(p_total * core)
        + np.sum(p_alpha * fock_alpha)
        + np.sum(p_beta * fock_beta)
    ) + nuclear


def _k_matrix(coefficients, n_occupied, direction, t):
    n_mo = coefficients.shape[1]
    n_virtual = n_mo - n_occupied
    k = np.zeros((n_mo, n_mo))
    for index, value in enumerate(direction):
        i, a = divmod(index, n_virtual)
        a += n_occupied
        k[a, i] += t * value
        k[i, a] -= t * value
    return k


def _zit_yonde(coefficients, n_occupied, direction, t):
    """Ca = C exp(+tK), Cb = C exp(-tK) -- TRIPLET (spin kirici) kanal."""
    k = _k_matrix(coefficients, n_occupied, direction, t)
    return coefficients @ expm(k), coefficients @ expm(-k)


def _ayni_yonde(coefficients, n_occupied, direction, t):
    """Ca = Cb = C exp(+tK) -- SINGLET (spin koruyucu) kanal, AYIRICI kontrol."""
    rotated = coefficients @ expm(_k_matrix(coefficients, n_occupied, direction, t))
    return rotated, rotated


def _second_derivative_along(core, eri, nuclear, coefficients, n_occ, direction, h, rot):
    """d^2 E / dt^2, t=0'da, bes noktali (O(h^4)) merkezi fark."""

    def energy(t):
        c_alpha, c_beta = rot(coefficients, n_occ, direction, t)
        return _uhf_energy(core, eri, nuclear, c_alpha, c_beta, n_occ)

    return (
        -energy(2 * h)
        + 16 * energy(h)
        - 30 * energy(0.0)
        + 16 * energy(-h)
        - energy(-2 * h)
    ) / (12 * h * h)


def _finite_difference_hessian(core, eri, nuclear, coefficients, n_occ, h, rot):
    """Tam Hessian, yalnizca YONLU ikinci turevlerden.

    Kosegen: H_pp = D^2(e_p).
    Kosegen disi: H_pq = 1/2 [ D^2(e_p + e_q) - D^2(e_p) - D^2(e_q) ]
    (H simetrik oldugu icin bu ozdeslik tamdir; test_stability.py'deki
    singlet makinesiyle ayni desen.)
    """
    n_rotations = n_occ * (coefficients.shape[1] - n_occ)
    basis_vectors = np.eye(n_rotations)
    diagonal = np.array(
        [
            _second_derivative_along(
                core, eri, nuclear, coefficients, n_occ, basis_vectors[p], h, rot
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
                n_occ,
                basis_vectors[p] + basis_vectors[q],
                h,
                rot,
            )
            hessian[p, q] = hessian[q, p] = 0.5 * (both - diagonal[p] - diagonal[q])
    return hessian


# --- 1. Tasiyici dogrulama: analitik triplet Hessian == sonlu fark --------

_FD_SISTEMLER = [
    ("H2 R=1.4", lambda: _h2(1.4), 1.6146, 1.305e-10),
    ("H2 R=3.0", lambda: _h2(3.0), 1.0021, 4.200e-10),
    ("H4 R=2.0", lambda: _zincir(4, 2.0), 3.0558, 3.815e-10),
    ("LiH", lambda: Molecule.from_string("Li 0 0 0; H 0 0 1.5949"), 10.1102, 4.173e-10),
]


@pytest.mark.parametrize("ad,yapici,olcek,bagil", _FD_SISTEMLER)
def test_TRIPLET_Hessian_SONLU_FARKLA_uyusuyor(ad, yapici, olcek, bagil):
    """TASIYICI DOGRULAMA: analitik ifade = enerjinin ikinci turevi.

    OLCULDU (h = 1e-3, bes noktali): bagil max fark H2 R=1.4 1.305e-10,
    H2 R=3.0 4.200e-10, H4 R=2.0 3.815e-10, LiH 4.173e-10, H2O 6.659e-10.
    Besinin de ayni mertebede olmasi, kalan farkin O(h^4) kesme + yuvarlama
    oldugunu soyler; bir transkripsiyon hatasi olsaydi sistemden sisteme
    mertebe atlardi.

    Sinir 10x olculenin ustune konur: amac kesme hatasini kirmizi
    yapmadan bir transkripsiyon hatasini yakalamaktir.
    """
    molecule = yapici()
    result = solve(molecule, "sto-3g", stability=False, max_iterations=5000)
    core, eri, nuclear = _integraller(molecule, "sto-3g")
    n_occupied = molecule.n_electrons // 2

    analitik = triplet_orbital_hessian(core, eri, result.coefficients, n_occupied)
    sayisal = _finite_difference_hessian(
        core, eri, nuclear, result.coefficients, n_occupied, 1e-3, _zit_yonde
    )

    assert analitik.shape == (n_occupied * (len(build_basis(molecule, "sto-3g"))
                                            - n_occupied),) * 2
    assert np.allclose(analitik, analitik.T, atol=1e-12)
    assert float(np.abs(analitik).max()) == pytest.approx(olcek, abs=1e-3)
    assert float(np.abs(analitik - sayisal).max()) / olcek < 10 * bagil


def test_AYNI_YONDE_dondurunce_deponun_SINGLET_Hessian_i_cikiyor():
    """AYIRICI KONTROL -- iki kanal gercekten FARKLI seyler olcuyor.

    Ayni sonlu fark makinesi, tek fark alfa ve beta'nin AYNI yonde
    dondurulmesi, deponun mevcut `orbital_hessian`'ini veriyor. OLCULDU
    (mutlak max fark / olcek): H2 R=1.4 2.498e-10 / 1.6146,
    H2 R=3.0 7.704e-10 / 1.0021, H4 R=2.0 2.399e-09 / 3.0558,
    LiH 5.906e-09 / 10.1102.

    Bu test olmadan "triplet Hessian yazdim" iddiasi, singlet'i yeniden
    yazmis olma ihtimalinden ayrilamazdi.
    """
    for yapici in (lambda: _h2(1.4), lambda: _h2(3.0), lambda: _zincir(4, 2.0)):
        molecule = yapici()
        result = solve(molecule, "sto-3g", stability=False, max_iterations=5000)
        core, eri, nuclear = _integraller(molecule, "sto-3g")
        n_occupied = molecule.n_electrons // 2

        singlet = orbital_hessian(core, eri, result.coefficients, n_occupied)
        sayisal_ayni = _finite_difference_hessian(
            core, eri, nuclear, result.coefficients, n_occupied, 1e-3, _ayni_yonde
        )
        olcek = float(np.abs(singlet).max())
        assert float(np.abs(singlet - sayisal_ayni).max()) / olcek < 1e-8

        triplet = triplet_orbital_hessian(core, eri, result.coefficients, n_occupied)
        assert float(np.abs(triplet - singlet).max()) > 1.0


def test_TRIPLET_SINGLETTEN_AYRI_HUKUM_veriyor():
    """3.0 bohr'da SINGLET "kararli" derken TRIPLET "eyer" diyor.

    OLCULDU: singlet en kucuk ozdeger +2.759810, triplet -1.002068.
    Depo bugune kadar bu noktada YALNIZCA +2.76'yi gorup "ic minimum"
    basiyordu; spin kanalindaki eyer ciktinin hicbir yerinde yoktu.
    """
    molecule = _h2(3.0)
    result = solve(molecule, "sto-3g", stability=False)
    core, eri, _ = _integraller(molecule, "sto-3g")

    singlet = float(np.linalg.eigvalsh(
        orbital_hessian(core, eri, result.coefficients, 1)).min())
    triplet = float(np.linalg.eigvalsh(
        triplet_orbital_hessian(core, eri, result.coefficients, 1)).min())

    assert singlet == pytest.approx(2.759810, abs=1e-5)
    assert triplet == pytest.approx(-1.002068, abs=1e-5)
    assert singlet > 0.0 > triplet


def test_TRIPLET_OLCUTU_gercek_sistemlerde_SINGLETTEN_iyi():
    """MUHAFIZ, CIPA DEGIL -- ama DOGRU kanal, ve bu olculdu.

    "lam < 0 <=> UHF kazanci var" hukmu H2'nin 18 ayri R'sinde sinandi:
    TRIPLET kanalla 18/18, SINGLET kanalla 8/18. Sahte-cipa sinavinda
    (fizik-disi rastgele integraller) iki kanal AYRILAMIYORDU
    (51/53 ve 50/53); ayirim gercek molekullerde ortaya cikiyor.

    Burada dokuz R kullanilir (sure icin); dokuzunda da ayni desen.
    """
    dogru_triplet = dogru_singlet = 0
    r_degerleri = (1.0, 1.4, 2.0, 2.2, 2.6, 3.0, 4.0, 6.0, 10.0)
    for r in r_degerleri:
        molecule = _h2(r)
        result = solve(molecule, "sto-3g", stability=False)
        core, eri, _ = _integraller(molecule, "sto-3g")
        lam_t = float(np.linalg.eigvalsh(
            triplet_orbital_hessian(core, eri, result.coefficients, 1)).min())
        lam_s = float(np.linalg.eigvalsh(
            orbital_hessian(core, eri, result.coefficients, 1)).min())
        kazanc = result.energy_total - search_uhf(molecule, "sto-3g", seeds=12).energy
        dogru_triplet += (lam_t < 0.0) == (kazanc > 1e-10)
        dogru_singlet += (lam_s < 0.0) == (kazanc > 1e-10)

    assert dogru_triplet == len(r_degerleri)
    # Yanlis kanal GERCEKTEN dusmeli -- bu satir olmadan test, triplet
    # yerine singlet yazilmis olsa da yesil kalirdi.
    assert dogru_singlet <= 5


# --- 2. Panel: dis kararlilik artik BASILIYOR -----------------------------


def test_PANELDE_DIS_KARARLILIK_satiri_var():
    """Bu modulun ASIL kazanci: depo "ic minimum" derken spin kanalinda EYER.

    Deponun PySCF referans kaydi h2_200 (tests/hf/test_scf.py) tam olarak
    bu durumdadir ve OLCULDU:
      E_RHF          = -0.783792654264
      E_UHF (arama)  = -0.937212833096   -> RHF 0.153420178832 hartree YUKARIDA
      ic  kararlilik = +2.546683038854   -> "kararli"
      dis kararlilik = -1.599532559297   -> EYER
    Panel bugune kadar yalnizca +2.5467'yi basiyordu.
    """
    molecule = Molecule.from_string("H 0 0 0; H 0 0 2.0")
    result = solve(molecule, "sto-3g", stability=True)

    assert result.stability.stable is True
    assert result.stability.lowest_eigenvalue == pytest.approx(
        2.546683038854, abs=1e-9
    )
    assert result.external_stability.stable is False
    assert result.external_stability.lowest_eigenvalue == pytest.approx(
        -1.599532559297, abs=1e-9
    )

    kazanc = result.energy_total - search_uhf(molecule, "sto-3g", seeds=12).energy
    assert kazanc == pytest.approx(0.153420178832, abs=1e-9)

    satirlar = dict(result.provenance.as_lines())
    assert "Kararlılık" in satirlar
    assert "Dış kararlılık" in satirlar
    assert satirlar["Kararlılık"].startswith("✓")
    assert satirlar["Dış kararlılık"].startswith("✗")
    assert "uhf" in satirlar["Dış kararlılık"].lower()


def test_stability_False_ile_DIS_analiz_de_kosulmaz():
    """"Analiz edilmedi" ile "kararli" ayni sey degildir -- ikinci kanalda da."""
    result = solve(_h2(3.0), "sto-3g", stability=False)
    assert result.stability is None
    assert result.external_stability is None
    assert "Dış kararlılık" not in dict(result.provenance.as_lines())


def test_analysis_yardimcisi_solve_un_ilistirdigiyle_ayni():
    """external_stability(result), solve()'un ilistirdigiyle AYNI olmali --
    stability=False ile kosulmus sonuclar icin sonradan hesaplama yolu."""
    molecule = _h2(3.0)
    ilistirilen = solve(molecule, "sto-3g", stability=True).external_stability
    yeniden = external_stability(solve(molecule, "sto-3g", stability=False))
    assert isinstance(yeniden, StabilityResult)
    np.testing.assert_allclose(
        yeniden.eigenvalues, ilistirilen.eigenvalues, atol=1e-12, rtol=0.0
    )
    assert yeniden.stable is ilistirilen.stable is False


def test_KAPALI_KABUK_DENGE_geometrileri_iki_kanalda_da_kararli():
    """Yanlis alarm sinavi: normal molekuller "spin eyeri" diye
    isaretlenmemeli. OLCULDU (en kucuk triplet ozdegeri):
    H2 R=1.4 +1.614595, LiH +0.337552, H2O +1.450247.
    """
    beklenen = [
        (_h2(1.4), 1.614595),
        (Molecule.from_string("Li 0 0 0; H 0 0 1.5949"), 0.337552),
        (Molecule.from_string(
            "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"), 1.450247),
    ]
    for molecule, lam in beklenen:
        result = solve(molecule, "sto-3g", stability=True)
        assert result.external_stability.stable is True
        assert result.external_stability.lowest_eigenvalue == pytest.approx(
            lam, abs=1e-5
        )


def test_dis_kararlilik_verdict_JSON_a_yazilabilir():
    """run.save() provenance'i dataclasses.asdict + json.dumps ile diske
    yaziyor; yeni alan bu yolu bozmamali."""
    import dataclasses
    import json

    result = solve(_h2(3.0), "sto-3g", stability=True)
    metin = json.dumps(dataclasses.asdict(result.provenance))
    assert "dis" in metin
    assert json.loads(metin)["external_stability"]["stable"] is False
