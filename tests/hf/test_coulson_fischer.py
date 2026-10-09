"""Coulson-Fischer noktasi -- Modul 1C'nin EN GUCLU cipasi.

RHF cozumunun spin kanalinda kararli olmaktan ciktigi ve UHF'in ayrildigi R.
Fiziksel bir CATALLANMA noktasidir: deneysel hicbir sayi icermez, hicbir
yerden verilmez, iki ayri yordamdan OKUNUR.

SINIF: DAIRESEL DEGIL. Iki yol paylasilan tek sey UHF enerji ifadesidir
(problemin tanimi); test edilen iki sey -- analitik Hessian formulu ve SCF --
ikinci yolda HIC KULLANILMAZ:

  (a) depo SCF'i + analitik triplet Hessian'in en kucuk ozdegerinin sifir koku
      -> Rc = 2.179694793840 bohr
  (b) SIMETRIDEN kurulmus orbitaller + bes noktali sonlu fark; SCF kosulmaz,
      analitik Hessian kullanilmaz
      -> Rc = 2.179694793998 bohr,   FARK = 1.578e-10

Fark, sonlu farkin YUVARLAMA tabanindan gelir -- kesme hatasindan degil, ve
bu OLCULDU: h kucultuldukce fark BUYUYOR (h=3e-3 -> 3.736e-10,
h=1e-3 -> 1.275e-09, h=3e-4 -> 3.734e-09), yani h=1e-3 zaten yuvarlamanin
hakim oldugu bolgededir. lam'daki ~1e-9 hata, egimin |dlam/dR| ~ 1.5 oldugu
yerde koke ~7e-10 olarak yansir; olculen 1.578e-10 bu mertebededir.

UCUNCU bir bagimsiz okuma da ayni yeri gosteriyor: SCF'in KENDI davranisi.
Rc'nin altinda UHF, RHF'e birebir cokuyor (|E_RHF - E_UHF| <= 8.9e-15);
ustunde ayriliyor ve fark d = R - Rc ile CATALLANMA usleriyle buyuyor.

BU DOSYANIN KENDI SINIRI -- OLCULDU, ve cipa tasarimi hakkinda geneldir:
bir KOK okuyan cipa, genel OLCEGE KORDUR. Hessian'in 4.0 carpani 2.0
yapildiginda Rc HIC KIPIRDAMIYOR (2.179694794'te kaliyor), cunku kuresel bir
carpan bir sifir gecisini yerinden oynatmaz. Mutasyon sinavinda uc kok-testi
de (test_COULSON_FISCHER_noktasi, test_ZINCIRLERDE_Rc_n_ile_KAYIYOR ve
test_TRIPLET_OLCUTU_...) YESIL kaldi; mutanti yalnizca BUYUKLUK karsilastiran
testler yakaladi (test_Rc_IKINCI_BAGIMSIZ_YOLDAN_SCF_SUZ'un aralik boyu
karsilastirmasi ve test_dis_kararlilik.py'nin sonlu fark testleri).

Yani kok = FIZIK, olcek = TRANSKRIPSIYON; ikisi ayri test sinifi ister ve
biri digerinin yerine gecmez. Bu dosyadaki cipalarin tek basina birakilmasi,
carpani yanlis yazilmis bir Hessian'i sevk etmeye yeterdi.

Diger mutantlarda Rc'nin nereye gittigi de olculdu: Coulomb terimi
BIRAKILIRSA (yani singlet yazilirsa) kok HIC YOK -- H2'nin singlet Hessian'i
0.6-12 bohr arasinda negatife inmiyor; Coulomb yarim birakilirsa yine kok yok;
Fock bloklarinin isareti takas edilirse yine kok yok; (ab|ij) terimi
duserse kok 4.570719109 bohr'a kayiyor (2.39 bohr hata, ama VAR).

OLCULMUS DURUST OLUMSUZ -- catallanma usleri ARAMAYA baglidir, koda degil:
tek sabit bias (0.35) ile olculdugunde UHF bazi noktalarda RHF'e GERI COKUYOR
(fark tam 0) ve us "nan" cikiyor. Ayni olcum search_uhf (12 tohum) ile temiz.
Bu, Task 3'te ilk kez olculen desendir: cipanin gucu kodun degil ARAMANIN
ozelligi. COKUS YERI MAKINEYE BAGLI ve KILITLENMEZ: ilk kayit d = 3e-3 ve
d = 1e-1 diyordu; 09.10.2026'da bu makinede (numpy 2.5.1, OpenBLAS 0.3.33)
yedi noktali izgarada (d = 1e-3 ... 2e-1) YALNIZ d = 1e-2'de, bu dosyanin bes
noktali izgarasinda YALNIZ d = 4e-3'te cokuyor; BLAS mi surum mu ayristirilmadi.
B7'den beri cokus SESSIZ DEGIL: solve_uhf(stability=True) tam o noktalarda
UHF EYERI (✗) basar, ve kararlilik izlemesi (search_uhf varsayilani, seeds = 1;
sabit bias + depodaki ozel `_follow_instability` de) bu iki izgaranin hicbir
noktasinda cokmuyor. Izgaralar d >= 1e-3: Rc'ye d <= 6e-5 yakinlikta izlemenin
kabul esiginin kor bolgesi var, orada izlemeli seeds = 1 de RHF eyerinde kalir
(✗) -- bkz. tests/hf/test_uhf_kararlilik.py (T6, KOR_BOLGE). Asagidaki
pitchfork testinin 12 tohumlu sayilari izlemeyle BIT BIT ayni.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import brentq

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import core_hamiltonian, eri_tensor, overlap_matrix
from qlab.engine.hf.multistart import search_uhf
from qlab.engine.hf.scf import solve
from qlab.engine.hf.stability import triplet_orbital_hessian
from qlab.engine.hf.uhf import solve_uhf
from qlab.types import Molecule
from qlab.units import BOHR_TO_ANGSTROM

RC_BOHR = 2.179694793840


def _h2(r_bohr: float) -> Molecule:
    yari = r_bohr * BOHR_TO_ANGSTROM / 2
    return Molecule.from_string(f"H 0 0 {-yari}; H 0 0 {yari}")


def _zincir(n: int, r_bohr: float) -> Molecule:
    adim = r_bohr * BOHR_TO_ANGSTROM
    return Molecule.from_string("; ".join(f"H 0 0 {i * adim:.17g}" for i in range(n)))


def _integraller(molecule: Molecule, basis_name: str):
    basis = build_basis(molecule, basis_name)
    return core_hamiltonian(basis, molecule), eri_tensor(basis)


# --- YOL (a): depo SCF + analitik triplet Hessian -------------------------


def _lam_scf(molecule: Molecule) -> float:
    """Triplet Hessian'in en kucuk ozdegeri, yakinsamis RHF cozumunde."""
    result = solve(molecule, "sto-3g", stability=False, max_iterations=5000)
    core, eri = _integraller(molecule, "sto-3g")
    n_occupied = molecule.n_electrons // 2
    hessian = triplet_orbital_hessian(core, eri, result.coefficients, n_occupied)
    return float(np.linalg.eigvalsh(hessian).min())


# --- YOL (b): simetri + sonlu fark; SCF YOK, analitik Hessian YOK ---------
#
# H2/STO-3G'de iki baz fonksiyonu vardir ve SIMETRI iki MO'yu TAM belirler:
#   sigma_g = (phi1 + phi2) / sqrt(2(1+S12)),  sigma_u = (phi1 - phi2) / sqrt(2(1-S12))
# Bu yuzden RHF cozumunu bulmak icin SCF'e gerek yoktur. Enerji, uhf.py ile
# ayni ifadeden VERILEN katsayilarla hesaplanir; ikinci turev bes noktali
# merkezi farkla alinir. Parametrelendirme, analitik ifadenin varsaydigiyla
# ayni olmak ZORUNDADIR: Ca = C exp(+tK), Cb = C exp(-tK).


def _uhf_enerjisi(core, eri, nuclear, c_alpha, c_beta, n_occupied) -> float:
    p_alpha = c_alpha[:, :n_occupied] @ c_alpha[:, :n_occupied].T
    p_beta = c_beta[:, :n_occupied] @ c_beta[:, :n_occupied].T
    p_total = p_alpha + p_beta
    coulomb = np.einsum("uvls,ls->uv", eri, p_total)
    exchange_alpha = np.einsum("ulvs,ls->uv", eri, p_alpha)
    exchange_beta = np.einsum("ulvs,ls->uv", eri, p_beta)
    fock_alpha = core + coulomb - exchange_alpha
    fock_beta = core + coulomb - exchange_beta
    return 0.5 * float(
        np.sum(p_total * core)
        + np.sum(p_alpha * fock_alpha)
        + np.sum(p_beta * fock_beta)
    ) + nuclear


def _lam_simetriden(r_bohr: float, h: float = 1e-3) -> float:
    """d^2E/dt^2, t=0'da -- SCF kosulmadan, analitik Hessian kullanilmadan."""
    molecule = _h2(r_bohr)
    basis = build_basis(molecule, "sto-3g")
    overlap = overlap_matrix(basis)
    core = core_hamiltonian(basis, molecule)
    eri = eri_tensor(basis)
    nuclear = molecule.nuclear_repulsion()
    s12 = overlap[0, 1]
    coefficients = np.column_stack(
        [
            np.array([1.0, 1.0]) / np.sqrt(2.0 * (1.0 + s12)),
            np.array([1.0, -1.0]) / np.sqrt(2.0 * (1.0 - s12)),
        ]
    )

    def energy(t: float) -> float:
        k = np.array([[0.0, -t], [t, 0.0]])
        return _uhf_enerjisi(
            core, eri, nuclear, coefficients @ expm(k), coefficients @ expm(-k), 1
        )

    return (
        -energy(2 * h)
        + 16 * energy(h)
        - 30 * energy(0.0)
        + 16 * energy(-h)
        - energy(-2 * h)
    ) / (12 * h * h)


# --- Cipalar --------------------------------------------------------------


def test_COULSON_FISCHER_noktasi():
    """OLCULDU: Rc = 2.179694793840 bohr = 1.153444811624 angstrom.

    Isaret dogrulamasi: lam(Rc - 1e-6) = +1.547817e-06,
    lam(Rc + 1e-6) = -1.547816e-06 -- kok gercek bir isaret degisimidir,
    tesadufi bir sifir gecisi degil.
    """
    rc = brentq(lambda r: _lam_scf(_h2(r)), 1.8, 2.6, xtol=1e-13)
    assert rc == pytest.approx(RC_BOHR, abs=1e-9)
    assert rc * BOHR_TO_ANGSTROM == pytest.approx(1.153444811624, abs=1e-9)
    assert _lam_scf(_h2(rc - 1e-6)) > 0.0
    assert _lam_scf(_h2(rc + 1e-6)) < 0.0


def test_Rc_IKINCI_BAGIMSIZ_YOLDAN_SCF_SUZ():
    """CIPAYI DAIRESEL OLMAKTAN CIKARAN test.

    Ikinci yol ne SCF kosar ne de analitik Hessian'i cagirir; orbitalleri
    simetriden kurar ve ikinci turevi sonlu farkla alir. OLCULDU:
    Rc = 2.179694793998, birinci yoldan FARK = 1.578e-10.

    OLCU MUTLAK, BAGIL DEGIL -- ve bu bir tercih degil zorunluluktur: lam
    tanim geregi Rc'de SIFIRDAN GECER, dolayisiyla bagil hata kokun yaninda
    patlar. Olculdu: R=2.1797'de mutlak fark 6.481e-10 iken bagil fark
    8.042e-05'tir. Ayni noktada mutlak olcu hicbir sey kaybetmez.
    """
    rc_fd = brentq(_lam_simetriden, 1.8, 2.6, xtol=1e-13)
    assert rc_fd == pytest.approx(2.179694793998, abs=1e-9)
    assert abs(rc_fd - RC_BOHR) < 1e-9

    # Yol (b) YALNIZ kokte degil, ARALIK BOYUNCA yol (a) ile ortusmeli --
    # aksi halde iki egri farkli olup tesadufen ayni yerde kesisebilirdi.
    # OLCULDU (mutlak fark): 6.893e-10 @1.5, 1.035e-09 @1.9, 1.275e-09 @2.1,
    # 1.185e-09 @2.3, 2.211e-10 @2.6, 4.579e-10 @3.0, 9.615e-10 @6.0 --
    # lam -2.37'den +1.35'e giderken hata ~1e-9'da SABIT kaliyor.
    for r in (1.5, 1.9, 2.1, 2.3, 2.6, 3.0, 6.0):
        assert abs(_lam_scf(_h2(r)) - _lam_simetriden(r)) < 1e-8, r


def test_Rc_ONCESINDE_UHF_RHF_ile_OZDES_SONRASINDA_DEGIL():
    """UCUNCU bagimsiz okuma: SCF'in kendi davranisi, Hessian hic kullanilmadan.

    OLCULDU -- Rc altinda |E_RHF - E_UHF|: 0.0 (d=-0.20), 8.88e-15 (-0.05),
    3.11e-15 (-0.01), 4.44e-16 (-0.001). Rc ustunde: +1.747e-07 (+0.001),
    +1.735e-05 (+0.01), +4.207e-04 (+0.05), +1.534e-01 (+1.60).
    """
    for d in (-0.20, -0.05, -0.01, -1e-3):
        r = RC_BOHR + d
        rhf = solve(_h2(r), "sto-3g", stability=False).energy_total
        uhf = search_uhf(_h2(r), "sto-3g", seeds=12).energy
        assert abs(rhf - uhf) < 1e-12, f"R = Rc{d:+}: Rc oncesinde ayrilmamali"
    for d in (1e-3, 0.01, 0.05, 1.60):
        r = RC_BOHR + d
        rhf = solve(_h2(r), "sto-3g", stability=False).energy_total
        uhf = search_uhf(_h2(r), "sto-3g", seeds=12).energy
        assert rhf - uhf > 1e-8, f"R = Rc{d:+}: Rc sonrasinda ayrilmali"


def test_CATALLANMA_USLERI_pitchfork():
    """SINIF: DAIRESEL DEGIL -- usler hicbir yerden VERILMEDI, uyumdan cikti.

    Bir catallanma (pitchfork) icin beklenen usler 2, 1/2 ve 1'dir; us ancak
    d -> 0 limitinde temizdir, o yuzden PENCERE DARALTILARAK olculur.

    OLCULDU (search_uhf, 12 tohum):
      pencere d <= 2e-1 (yedi nokta):   1.9767 / 0.4825 / 0.9727
      pencere d <= 1.6e-2 (bes nokta):  1.9960 / 0.4970 / 0.9953
    Pencere daraldikca ucu de beklentiye YAKINSIYOR -- kilitlenen sey budur.
    """
    d = np.array([1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2])
    farklar, spinler, s_kareler = [], [], []
    for adim in d:
        molecule = _h2(RC_BOHR + adim)
        rhf = solve(molecule, "sto-3g", stability=False)
        uhf = search_uhf(molecule, "sto-3g", seeds=12).result
        farklar.append(rhf.energy_total - uhf.energy_total)
        overlap = overlap_matrix(build_basis(molecule, "sto-3g"))
        p_alpha = uhf.coefficients_alpha[:, :1] @ uhf.coefficients_alpha[:, :1].T
        p_beta = uhf.coefficients_beta[:, :1] @ uhf.coefficients_beta[:, :1].T
        spinler.append(float(np.abs(np.diag((p_alpha - p_beta) @ overlap)).max()))
        s_kareler.append(uhf.s_squared)

    def us(y) -> float:
        return float(np.polyfit(np.log(d), np.log(np.array(y)), 1)[0])

    assert us(np.abs(farklar)) == pytest.approx(1.9960, abs=0.01)
    assert us(spinler) == pytest.approx(0.4970, abs=0.01)
    assert us(s_kareler) == pytest.approx(0.9953, abs=0.01)


def test_ZINCIRLERDE_Rc_n_ile_KAYIYOR():
    """OLCULDU: n=2 2.179694794, n=4 1.951986163, n=6 1.804435695 bohr.

    Uzun zincir limiti icin de olculdu ama IDDIA EDILMEZ: n=8 1.695323976,
    n=10 1.609356683, n=12 1.538971647. DURUST OLUMSUZ -- 1/n uyumu
    Rc = 1.499197 + 1.456804/n verir ama artigi 0.089'a kadar cikar; log-log
    uyumu Rc ~ n^-0.1942 verir, artik RMS 1.03e-02. IKISI DE KABA; alti nokta
    n -> sonsuz limitini secmeye YETMEZ ve bu belge o limiti SECMEZ.
    """
    beklenen = {2: 2.179694794, 4: 1.951986163, 6: 1.804435695}
    for n, rc in beklenen.items():
        kok = brentq(lambda r, n=n: _lam_scf(_zincir(n, r)), 1.3, 2.6, xtol=1e-11)
        assert kok == pytest.approx(rc, abs=1e-6), n
