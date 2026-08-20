"""UHF cekirdegi -- ve uygulamada SESSIZ hata ureten alti sey.

Bu dosyanin isi enerji DOGRULAMAK degil (cipa kumesi Task 3'te); burada
Fock'un, enerjinin ve <S^2>'nin DOGRU KURULDUGU denetleniyor.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import core_hamiltonian, overlap_matrix
from qlab.engine.hf.scf import solve
from qlab.engine.hf.uhf import solve_uhf
from qlab.errors import InvalidInputError
from qlab.types import Molecule
from qlab.units import BOHR_TO_ANGSTROM

# OLCULDU: tek H atomunun STO-3G enerjisi, h11/S11 olarak, MOLEKULDEN BAGIMSIZ.
E_H_STO3G = -0.466581849557276


def _h2(r_bohr: float) -> Molecule:
    yari = r_bohr * BOHR_TO_ANGSTROM / 2
    return Molecule.from_string(f"H 0 0 {-yari}; H 0 0 {yari}")


def _alternan(n: int, olcek: float = 0.35) -> np.ndarray:
    return olcek * np.array([(-1.0) ** i for i in range(n)])


def test_KAPALI_KABUK_dengede_UHF_RHF_ile_ozdes():
    """TRANSKRIPSIYON DENETIMI -- CIPA DEGIL.

    Her RHF cozumu zaten bir UHF DURAGAN NOKTASIDIR; yerel minimumsa UHF
    ona geri doner. Bu bir teoremdir ve fizik hakkinda hicbir sey soylemez;
    on olcumde SAHTE-CIPA SINAVI kosuldu ve fizige ait olmayan rastgele
    integrallerde de 32/32 tuttugu goruldu.

    Yine de degerlidir: ucuz bir MUHAFIZDIR.
    """
    for metin in (
        "He 0 0 0",
        "H 0 0 0; H 0 0 0.74",
        "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
    ):
        mol = Molecule.from_string(metin)
        r = solve(mol, "sto-3g", stability=False)
        u = solve_uhf(mol, "sto-3g")
        assert abs(u.energy_total - r.energy_total) < 1e-12
        assert abs(u.s_squared) < 1e-12


def test_AYRISMA_LIMITI_iki_serbest_atom():
    """OLCULDU: UHF(H2, R=20 bohr) = -0.933163699114551 ve
    2 x E_H(STO-3G) = -0.933163699114551 -- fark 4.441e-16, <S^2> = 1.000000000000.

    E_H tek atomdan (h11/S11) hesaplanir, molekulden BAGIMSIZDIR; bu yuzden
    burada dairesel bir sey yok. RHF ise ayni limitte -0.570860727 veriyor,
    yani 0.362302972 hartree = 9.85877 eV sapiyor.
    """
    bazi = build_basis(Molecule.from_string("H 0 0 0"), "sto-3g")
    e_h = float(
        core_hamiltonian(bazi, Molecule.from_string("H 0 0 0"))[0, 0]
        / overlap_matrix(bazi)[0, 0]
    )
    assert e_h == pytest.approx(E_H_STO3G, abs=1e-14)

    u = solve_uhf(_h2(20.0), "sto-3g", initial_bias=_alternan(2))
    assert u.energy_total == pytest.approx(2 * e_h, abs=1e-12)
    assert u.s_squared == pytest.approx(1.0, abs=1e-10)

    r = solve(_h2(20.0), "sto-3g", stability=False, max_iterations=5000)
    assert (r.energy_total - 2 * e_h) == pytest.approx(0.362302972, abs=1e-6)


def test_VARYASYONEL_sinir():
    """E_UHF <= E_RHF. TEOREM, dolayisiyla transkripsiyon denetimi.

    ZAYIF oldugu OLCULDU (on olcum): on gercekci mutantin YEDISI bu
    denetimi geciyor, cunku hatalarin cogu enerjiyi DUSURUYOR. Tek basina
    asla yeterli sayilmaz.
    """
    for r_bohr in (1.4, 2.0, 3.0, 4.0, 6.0):
        mol = _h2(r_bohr)
        assert solve_uhf(mol, "sto-3g", initial_bias=_alternan(2)).energy_total <= (
            solve(mol, "sto-3g", stability=False).energy_total + 1e-10
        )


@pytest.mark.parametrize(
    ("sembol", "n_alpha", "beklenen"),
    [("H", 1, 0.75), ("He", 1, 0.0), ("Li", 2, 0.75), ("C", 4, 2.0), ("N", 5, 3.75), ("O", 5, 2.0)],
)
def test_SERBEST_ATOMLARDA_S2_tam(sembol, n_alpha, beklenen):
    """<S^2> = S(S+1), on iki hanede.

    OLCULDU: H 0.750000000000, He 0.000000000000, Li 0.750000000000,
    C 2.000000000000, N 3.750000000000, O 2.000000000000.

    Formulun SABIT DONDURMEDIGI ayrica gosterildi (on olcum): Li'nin
    yanlis (0.085 Ha yuksek) cozumu ayni formulle 0.750048 veriyor.
    """
    mol = Molecule.from_string(f"{sembol} 0 0 0")
    n_baz = len(build_basis(mol, "sto-3g"))
    u = solve_uhf(mol, "sto-3g", n_alpha=n_alpha, initial_bias=0.35 * np.ones(n_baz))
    assert u.s_squared == pytest.approx(beklenen, abs=1e-10)
    assert u.n_alpha == n_alpha
    assert u.n_beta == mol.n_electrons - n_alpha


def test_S2_ORTUSME_matrisini_ICERIYOR():
    """TUZAK: <S^2> formulunde ortusme matrisini unutmak.

    <S^2> = Sz(Sz+1) + n_beta - SUM_ij |<a_i|b_j>|^2 ve o ic carpim
    ORTOGONAL OLMAYAN AO tabaninda S ile alinir. S = I varsayilirsa sonuc
    sessizce yanlis cikar.

    Burada S'nin birim OLMADIGI bir sistem kullaniliyor: H2 R=1.4 bohr'da
    S01 = 0.6593182061.
    """
    S = overlap_matrix(build_basis(_h2(1.4), "sto-3g"))
    assert S[0, 1] == pytest.approx(0.6593182061, abs=1e-9)  # tuzak GERCEKTEN var
    u = solve_uhf(_h2(1.4), "sto-3g", initial_bias=_alternan(2))
    assert abs(u.s_squared) < 1e-12  # kapali kabuk, kirilmamis


def test_ACIK_KABUK_n_alpha_ayarlanabiliyor():
    """Triplet H2: n_alpha=2, n_beta=0 -> <S^2> = 2.0 tam.

    `solve()` tek elektron sayisini REDDEDER (kapali kabuk RHF); `solve_uhf`
    kabul eder, ve modulun varolus sebeplerinden biri budur.
    """
    u = solve_uhf(_h2(1.4), "sto-3g", n_alpha=2)
    assert (u.n_alpha, u.n_beta) == (2, 0)
    assert u.s_squared == pytest.approx(2.0, abs=1e-10)

    from qlab.errors import UnsupportedSystemError

    with pytest.raises(UnsupportedSystemError):
        solve(Molecule.from_string("H 0 0 0"), "sto-3g")
    assert solve_uhf(Molecule.from_string("H 0 0 0"), "sto-3g", n_alpha=1)


def test_BASLANGIC_TAHMINI_sonucu_DEGISTIRIYOR():
    """TUZAK ve DURUST OLUMSUZ: bias olmadan UHF, RHF'e COKUYOR.

    OLCULDU (H2, R = 2.00 A = 3.779452 bohr):
      RHF               = -0.783792654264
      UHF (bias=None)   = -0.783792654264   <- TAM cokme, <S^2> = 0
      UHF (alternan)    = -0.937212833096   <S^2> = 0.945862386
      fark              =  0.153420179 hartree = 4.175 eV

    Yani `solve_uhf`in tek basina dondurdugu sayi bir MINIMUM DEGIL bir
    DURAGAN NOKTADIR. Minimum iddiasi ancak cok-baslangicli arama ile
    yapilir -- Task 3.
    """
    mol = Molecule.from_string("H 0 0 0; H 0 0 2.00")
    rhf = solve(mol, "sto-3g", stability=False).energy_total
    biassiz = solve_uhf(mol, "sto-3g")
    alternan = solve_uhf(mol, "sto-3g", initial_bias=_alternan(2))

    assert abs(biassiz.energy_total - rhf) < 1e-12  # COKUYOR
    assert abs(biassiz.s_squared) < 1e-12
    assert alternan.energy_total == pytest.approx(-0.937212833096, abs=1e-9)
    assert alternan.s_squared == pytest.approx(0.945862386, abs=1e-6)
    assert (rhf - alternan.energy_total) == pytest.approx(0.153420179, abs=1e-6)


def test_PROVENANCE_duragan_nokta_uyarisini_TASIYOR():
    """Seffaflik metni, dondurulen sayinin bir MINIMUM OLMADIGINI soylemeli.

    Bu, Task 1'in dersinin devami: bir SCF enerjisi arama protokolu
    olmadan bir sayi degildir, ve panel bunu okuyucuya soylemek zorundadir.
    """
    u = solve_uhf(_h2(1.4), "sto-3g")
    metin = " ".join(deger for _, deger in u.provenance.as_lines())
    assert "DURAGAN NOKTADIR" in metin
    assert "multistart" in metin
    assert "s_squared" in u.provenance.diagnostics


def test_gecersiz_girdiler_reddediliyor():
    with pytest.raises(InvalidInputError, match="n_alpha"):
        solve_uhf(_h2(1.4), "sto-3g", n_alpha=5)
    with pytest.raises(InvalidInputError, match="n_alpha"):
        solve_uhf(_h2(1.4), "sto-3g", n_alpha=-1)
    with pytest.raises(InvalidInputError, match="initial_bias"):
        solve_uhf(_h2(1.4), "sto-3g", initial_bias=np.zeros(5))
