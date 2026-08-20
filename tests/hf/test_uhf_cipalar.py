"""UHF cipalari -- ve her birinin SINIFI.

Uc sinif var ve karistirilmazlar:
  DAIRESEL DEGIL : bagimsiz ikinci yolu var, gercekten bir sey dogruluyor
  TRANSKRIPSIYON : teorem/ozdeslik, ama mutasyona duyarli -> muhafiz
  TAUTOLOJI      : fizige ait olmayan girdide de tutuyor -> "cipa" DENMEZ

On olcumde siniflandirildi ve SAHTE-CIPA SINAVI kosuldu; sonuclar
dokstringlerde. Tautolojiler gizlenmiyor, SINIFLARIYLA yaziliyorlar.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import core_hamiltonian, overlap_matrix
from qlab.engine.hf.multistart import search_uhf
from qlab.engine.hf.scf import solve
from qlab.engine.hf.uhf import energy_from_density, solve_uhf
from qlab.types import Molecule
from qlab.units import BOHR_TO_ANGSTROM

E_H_STO3G = -0.466581849557276
UZAK_A = 20.0 * BOHR_TO_ANGSTROM  # 20 bohr = 10.583544 angstrom

# OLCULDU (cok-baslangicli, 12 tohum; hepsi 12/12 yakinsadi):
_ATOM = {
    "H": (1, -0.466581849557, 0.75),
    "Li": (2, -7.315525981281, 0.75),
    "N": (5, -53.719010162594, 3.75),
    "F": (5, -97.986504958736, 0.75),
}


def test_E_H_tek_atomdan_okunuyor():
    """Ayrisma cipasinin girdisi MOLEKULDEN BAGIMSIZ uretiliyor -- kanit."""
    h = Molecule.from_string("H 0 0 0")
    bazi = build_basis(h, "sto-3g")
    assert float(core_hamiltonian(bazi, h)[0, 0] / overlap_matrix(bazi)[0, 0]) == (
        pytest.approx(E_H_STO3G, abs=1e-14)
    )


def test_AYRISMA_LIMITI_cipasi():
    """SINIF: DAIRESEL DEGIL -- ama ZAYIF.

    2 x E_H tek atomdan hesaplanir. OLCULDU: UHF(H2, R=20 bohr) =
    -0.933163699115, fark 5.551e-16. Ayni limitte RHF 0.362302972 Ha =
    9.85877 eV sapiyor.

    ZAYIFLIGI da OLCULDU (on olcum): sekiz gercekci mutantin IKISI bu
    denetimi DOGRU KODDAN AYIRT EDILEMEZ sekilde geciyor (K'nin J desenine
    cevrilmesi 4.44e-16, K'nin yariya bolunmesi 1.11e-16). TEK BASINA asla
    yeterli sayilmaz.
    """
    mol = Molecule.from_string(f"H 0 0 0; H 0 0 {UZAK_A:.9f}")
    u = search_uhf(mol, "sto-3g", seeds=8)
    assert u.energy == pytest.approx(2 * E_H_STO3G, abs=1e-12)

    r = solve(mol, "sto-3g", stability=False, max_iterations=5000)
    assert (r.energy_total - 2 * E_H_STO3G) == pytest.approx(0.362302972, abs=1e-6)


@pytest.mark.parametrize("sembol", sorted(_ATOM))
def test_ATOM_enerjileri_ve_S2(sembol):
    """Parca toplami cipasinin PARCALARI -- ayri ayri, cok-baslangicli.

    OLCULDU: H -0.466581849557 (<S^2> 0.75), Li -7.315525981281 (0.75),
    N -53.719010162594 (3.75), F -97.986504958736 (0.75).

    DIKKAT: bu sayilar TOHUMA BAGLI. Olculdu, N atomu tekbicim bias'la
    -53.719010163, alternan bias'la -53.259271159 veriyor -- 0.46 hartree
    fark. Arama olmadan "N atomunun UHF enerjisi" diye bir sayi YOKTUR.
    """
    na, enerji, s2 = _ATOM[sembol]
    r = search_uhf(Molecule.from_string(f"{sembol} 0 0 0"), "sto-3g", seeds=12, n_alpha=na)
    assert r.energy == pytest.approx(enerji, abs=1e-9)
    assert r.result.s_squared == pytest.approx(s2, abs=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("ad", "metin", "n_alpha", "parcalar", "beklenen"),
    [
        ("H2", f"H 0 0 0; H 0 0 {UZAK_A:.9f}", None, ("H", "H"), -0.933163699115),
        ("LiH", f"Li 0 0 0; H 0 0 {UZAK_A:.9f}", None, ("Li", "H"), -7.782107830838),
        ("N2", f"N 0 0 0; N 0 0 {UZAK_A:.9f}", 7, ("N", "N"), -107.438020325188),
    ],
)
def test_PARCA_TOPLAMI_cipasi(ad, metin, n_alpha, parcalar, beklenen):
    """SINIF: DAIRESEL DEGIL ve GUCLU -- ama yalniz YETERLI ARAMA ile.

    E(AB, R=20 bohr) = E(A) + E(B); parcalar AYRI AYRI hesaplanir.
    OLCULDU: H2 fark +5.551e-16, LiH -9.592e-14, N2 +1.421e-14.

    Ayrisma cipasindan GUCLU olmasinin sebebi kapsamidir: p kabuklu (N),
    heteronukleer (LiH) ve ACIK KABUKLU parcalari (H, Li, N) iceriyor.
    On olcumde bu cipa gercek bir hata YAKALADI: Li atomu alternan bias ile
    0.085 Ha yuksek bir cozume `converged=True` ile sessizce yakinsamisti.
    """
    ab = search_uhf(Molecule.from_string(metin), "sto-3g", seeds=12, n_alpha=n_alpha)
    toplam = sum(
        search_uhf(
            Molecule.from_string(f"{p} 0 0 0"), "sto-3g", seeds=12, n_alpha=_ATOM[p][0]
        ).energy
        for p in parcalar
    )
    assert ab.energy == pytest.approx(beklenen, abs=1e-9)
    assert ab.energy == pytest.approx(toplam, abs=1e-11)


@pytest.mark.slow
def test_PARCA_TOPLAMI_cipasinin_SINIRI_HF_molekulunde():
    """DURUST OLUMSUZ: cipanin gucu KODUN degil ARAMANIN ozelligi.

    OLCULDU: HF (H + F, 20 bohr) icin parca toplami -98.453086808293, ama
    bu tohum ailesi taban durumu HIC BULMUYOR --
      seeds=8  -> -97.570681531688  (fark +8.824e-01, 1 ayri cozum)
      seeds=12 -> -97.570681531688  (+8.824e-01)
      seeds=24 -> -97.678222271884  (+7.749e-01, 3 ayri cozum)
      seeds=64 -> -97.678222271884  (+7.749e-01)
    Yani tohum sayisini SEKIZ KAT artirmak farki yalnizca 0.11 Ha
    kapatiyor ve orada duruyor.

    Bu test cipayi degil, cipanin SINIRINI kilitler. Biri ileride tohum
    ailesini iyilestirirse bu test kirmizi olur -- ve o zaman bu dokstring
    guncellenir. Sinir gizlenmiyor, olculuyor.
    """
    hf = search_uhf(Molecule.from_string(f"H 0 0 0; F 0 0 {UZAK_A:.9f}"), "sto-3g", seeds=24)
    toplam = _ATOM["H"][1] + _ATOM["F"][1]
    assert hf.energy == pytest.approx(-97.678222271884, abs=1e-9)
    assert (hf.energy - toplam) == pytest.approx(0.7749, abs=1e-3)


def test_VARYASYONEL_denetimin_ZAYIFLIGI_belgelenmistir():
    """SINIF: TRANSKRIPSIYON DENETIMI -- ve ZAYIF.

    E_UHF <= E_RHF bir teoremdir (RHF, UHF'in Ca = Cb ozel halidir).
    OLCULDU (on olcum): on gercekci mutantin YEDISI bu denetimi geciyor,
    cunku hatalarin cogu enerjiyi DUSURUYOR -- ornegin J'yi spin
    yogunlugundan kurmak H2O'da -53.26 Ha veriyor ve "ihlal yok" diyor.

    Bu test denetimin GECERLI oldugunu kilitler; TEK BASINA YETERLI
    olmadigini dokstring soyler.
    """
    for metin in ("H 0 0 0; H 0 0 2.00", "H 0 0 0; H 0 0 0.74"):
        mol = Molecule.from_string(metin)
        assert search_uhf(mol, "sto-3g", seeds=8).energy <= (
            solve(mol, "sto-3g", stability=False).energy_total + 1e-10
        )


def test_RHF_KARARLIYKEN_ozdeslik_TAUTOLOJIDIR():
    """SINIF: TAUTOLOJI -- ve bu test o gercegi BELGELER.

    "RHF kararliyken UHF = RHF" bir cipa DEGILDIR: her RHF cozumu zaten bir
    UHF duragan noktasidir. SAHTE-CIPA SINAVI KOSULDU (on olcum): molekul,
    cekirdek ya da geometri iceremeyen RASTGELE integrallerde de 32/32
    tutuyor.

    Yine de sevk edilir -- ucuz bir MUHAFIZ olarak, "cipa" adiyla DEGIL.
    """
    for metin in ("He 0 0 0", "H 0 0 0; H 0 0 0.74"):
        mol = Molecule.from_string(metin)
        assert abs(
            solve_uhf(mol, "sto-3g").energy_total
            - solve(mol, "sto-3g", stability=False).energy_total
        ) < 1e-12


def test_ENERJI_YOGUNLUKTAN_BAGIMSIZ_yeniden_hesaplaniyor():
    """M9 DELIGINI KAPATAN DENETIM.

    On olcumde tasarlanan M9 mutanti -- dogru Fock, dogru enerji, artik
    `E += 1e-3 * SUM (Mulliken spin populasyonu)^2`, N > 2'ye kapili --
    onerilen DOKUZ denetimin DOKUZUNU DA geciyordu: ayrisma 4.44e-16,
    parca toplami dogru kodla ayni, varyasyonel 1.42e-14, kararli-ozdeslik
    4.26e-14, Hessian uyumu 11/11, FCI 4.14e-13, atom <S^2> tam.

    Sebep: eklenen terim kapali kabukta SIFIR ve ayrisma limitinde
    parcalarda da butunde de AYNI toplami veriyor (H4'te SUM = 4, dort ayri
    H'de de 4).

    Bu denetim SCF BIRIKTIRICISINE degil YAKINSAMIS YOGUNLUGA bakiyor: E,
    P'den standart ifadeyle yeniden kurulur. M9'un ekledigi terim orada
    YOKTUR.

    DENETIMIN YUK TASIDIGI OLCULDU -- M9 delta ile kucultulerek:
      delta = 1e-3  -> YEDI test kiriliyor (atom enerjileri, parca toplami,
                       HF siniri VE bu denetim)
      delta = 1e-10 -> yalniz IKI test: LiH parca toplami ve bu denetim
      delta = 1e-12 -> yalniz BU DENETIM
      delta = 1e-13 -> HICBIR test yakalamiyor (3.8e-13 < 1e-12 esigi)
    Yani bu denetim olmasaydi 1e-12 mertebesindeki bir sizma sessizce
    gecerdi; literal enerji degerleri onu ancak delta buyudugunde yakaliyor.
    """
    for metin in (
        "H 0 0 0; H 0 0 0.74",
        "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        "H 0 0 0; H 0 0 2.0; H 0 0 4.0; H 0 0 6.0",
        "Li 0 0 0; H 0 0 1.60",
    ):
        mol = Molecule.from_string(metin)
        u = solve_uhf(mol, "sto-3g")
        assert abs(energy_from_density(u, mol, "sto-3g") - u.energy_total) < 1e-12

    # ...ama YUKARIDAKI DORT VAKA TEK BASINA YETMEZ ve bu OLCULDU: dordunde
    # de spin populasyonu TAM SIFIR (SUM(sp^2) = 0.000e+00), yani M9'un
    # ekledigi terim orada HIC TETIKLENMIYOR. Denetimin yuk tasimasi icin
    # N > 2 VE spin KIRIK bir vaka sart.
    #
    # OLCULDU (gerilmis H4, alternan bias): aralik 2.0 bohr -> SUM(sp^2)
    # 0.236765 / <S^2> 0.113312; 3.0 -> 3.028714 / 1.428911;
    # 4.0 -> 3.842103 / 1.894299.
    for aralik in (2.0, 3.0, 4.0):
        a = aralik * BOHR_TO_ANGSTROM
        mol = Molecule.from_string("; ".join(f"H 0 0 {i * a:.9f}" for i in range(4)))
        u = solve_uhf(
            mol, "sto-3g", initial_bias=0.35 * np.array([(-1.0) ** i for i in range(4)])
        )
        S = overlap_matrix(build_basis(mol, "sto-3g"))
        spin_pop = np.diag((u.density_alpha - u.density_beta) @ S)
        assert float(np.sum(spin_pop**2)) > 0.2  # terim GERCEKTEN tetikleniyor
        assert abs(energy_from_density(u, mol, "sto-3g") - u.energy_total) < 1e-12


def test_YOGUNLUKTAN_HESAP_spin_KIRIK_cozumde_de_tutuyor():
    """M9 yalniz N > 2'ye kapiliydi ve spin populasyonu SIFIR OLMAYAN
    durumda deger tasiyordu; denetim orada da kosulmali.

    OLCULDU: H2 R = 2.00 A'da alternan bias ile <S^2> = 0.945862386, yani
    spin GERCEKTEN kirilmis.
    """
    mol = Molecule.from_string("H 0 0 0; H 0 0 2.00")
    u = solve_uhf(mol, "sto-3g", initial_bias=0.35 * np.array([1.0, -1.0]))
    assert u.s_squared == pytest.approx(0.945862386, abs=1e-6)
    assert abs(energy_from_density(u, mol, "sto-3g") - u.energy_total) < 1e-12


def test_ARAMA_RAPORU_kendi_kendini_denetliyor():
    """Bir UHF enerjisi ARAMA PROTOKOLU olmadan bir sayi degildir."""
    r = search_uhf(Molecule.from_string("H 0 0 0; H 0 0 2.00"), "sto-3g", seeds=8)
    assert r.n_seeds == 8
    assert r.n_converged == 8
    assert r.n_distinct >= 2  # cokmus ve kirik cozum birlikte gorunuyor
    assert r.spread > 0.1
    assert "ayri cozum" in r.as_line()
