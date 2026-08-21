"""UHF cozumunden site-bazli tight-binding parametrelerinin cikarilmasi.

NE YAPAR
--------
`search_uhf` ile bulunan spin-kirik cozumun alfa/beta Fock matrislerini
Lowdin ile ortogonallestirir ve site bazinda okur:

    H_eff^sigma = S^(-1/2) F^sigma S^(-1/2),   sigma in {alfa, beta}

NEDEN UHF, NEDEN RHF DEGIL (olculdu)
------------------------------------
Hidrojen halkalarinda RHF cozumu cogu geometride HF'in TABAN DURUMU DEGILDIR:
45 (N, d) noktasinin 35'i spin eyeridir ve N = 8, 12 test edilen HICBIR
geometride kararli degildir. Desen Huckel'dir -- 4n halkalarinda TB
spektrumunun HOMO/LUMO'su tam dejeneredir ve RHF dejenere seviyeyi cift
isgale zorlar. Dolayisiyla RHF indirgemesi bir eyer noktasini indirgerdi.

MODELIN SEKLI VARSAYIM DEGIL, OLCUMDUR
--------------------------------------
Spin-kirik halka cozumu ANTIFERROMANYETIKTIR ve bu su bicimde gorunur
(hepsi 21.08.2026'da bu depoda yeniden olculdu, STO-3G):

  * Mulliken spin populasyonu alterniyor  (N=8, d=2.0: -+0.517253)
  * diag(H_eff^alfa) IKI deger arasinda alterniyor (+0.250498 / -0.135858)
  * diag(H_eff^beta), alfa'nin BIR SITE kaydirilmisidir; olculen artik
    1.942890e-15 ... 9.087647e-09. Bu aralik BES sistemde olculdu ve
    sistemleri ADLIDIR, cunku aralik hangi sistemlerden geldigi
    yazilmadan tekrarlanamaz:
        (N=6,  d=2.6) 2.220446e-15    (N=6,  d=3.0) 1.942890e-15
        (N=8,  d=2.0) 5.558044e-09    (N=8,  d=2.6) 3.951947e-09
        (N=10, d=2.6) 9.087647e-09
  * hoplama SPINDEN BAGIMSIZ ve tekbicimdir: t^alfa = t^beta = -0.407145,
    sekiz bagin sekizinde de ayni. "Ayni"nin OLCUSU: N=8 d=2.0'da
    max|t^a - t^a[0]| = 3.003514e-09 ve max|t^a - t^b| = 8.213723e-10.
    Bu sayilar SIFIR degildir ve olmalari da beklenmez -- `solve_uhf`in
    artik toleransi 1e-8'dir, yani Fock elemanlarinin tabani oradadir.

Bu yuzden bagimsiz parametreler DORTTUR: eps_bar, Delta, t1, t2.

    eps_bar = kosegenin ortalamasi
    Delta   = kademeli alan, (cift indisli - tek indisli) / 2
    t_m     = halka mesafesi m olan elemanlarin ortalamasi

DELTA'NIN ISARETI KEYFI DEGIL: alfa elektronunun yerinde enerjisi, BETA
fazlasi tasiyan sitede yukselir (fazladan Coulomb, eksik degistokus). Yani
`delta` ile cift indisli sitelerin alfa-spin fazlasi ZIT isaretlidir.
Isaretin kendisi sistemden sisteme doner -- hangi alt-orgunun alfa
tasidigi UHF cozumunun kendi secimidir -- ama ZITLIK donmez. OLCULDU:
    N=8,  d=2.0: spin(site 0) = -0.517253 -> delta = +0.193178
    N=8,  d=2.6: spin(site 0) = +0.703871 -> delta = -0.265159
    N=10, d=2.6: spin(site 0) = +0.683922 -> delta = -0.257172

Yapi VARSAYILMAZ: `staggering_residual` her cagrida olculur ve sonuca
ilistirilir; cagiran onu okuyup guvenip guvenmeyecegine karar verebilir.

NE SINANMAZ / NE IDDIA EDILMEZ
------------------------------
  * Yalnizca STO-3G. 6-31G atom basina iki fonksiyon tasir ve site-bazli
    okuma orada TANIMSIZDIR (H_eff[0,1] atom-ICI elemani dondururdu).
  * Yalnizca HALKA. Acik zincirde bulk parametreli bir TB modelin hatasi,
    kesme tamamen kaldirilsa bile 0.096 Ha'da taban yapar; o taban KENAR
    etkisidir ve bu modul onu modellemez.
  * UHF bir spin ozdurumu DEGILDIR; <S^2> sonuca ilistirilir ama
    DUZELTILMEZ (spin projeksiyonu yoktur). OLCULDU: N=8 d=2.0'da
    <S^2> = 1.220731, yani cozum ne singlet ne triplet.
  * Parametreler bir ARAMA protokolunden gelir (`search_uhf`), tek bir
    SCF'ten degil; arama ozeti `search_line` alaninda tasinir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import core_hamiltonian, eri_tensor, overlap_matrix
from qlab.engine.hf.multistart import search_uhf
from qlab.errors import InvalidInputError, UnsupportedSystemError
from qlab.types import Molecule

# Kademeli yapinin kabul esigi. Gerekcesi OLCULMUSTUR: bes gercek sistemde
# artik 1.942890e-15 ... 9.087647e-09 araligindaydi, yani esik olculenin ~10
# kati ustunde ve gurultuye degil YAPIYA duyarli. Tabanin nereden geldigi de
# olculdu: `solve_uhf`in artik toleransi 1e-8'dir.
STAGGERING_TOLERANCE = 1e-7


@dataclass(frozen=True)
class DownfoldedParameters:
    """UHF'ten indirgenmis site parametreleri. Enerjiler HARTREE.

    epsilon_bar         : kosegenin ortalamasi (ortalama yerinde enerji)
    delta               : kademeli alan; spin-kirilmamis rejimde TAM SIFIR
    hoppings            : (t1, t2, ...), halka mesafesine gore
    staggering_residual : |diag(H^a) - roll(diag(H^b), 1)| -- yapinin OLCUSU
    s_squared           : UHF'in <S^2>'si; spin kirlenmesi RAPORLANIR
    search_line         : arama protokolunun tek satirlik ozeti
    """

    epsilon_bar: float
    delta: float
    hoppings: tuple[float, ...]
    n_site: int
    basis_name: str
    s_squared: float
    energy_uhf: float
    staggering_residual: float
    search_line: str


def _ring_distance(i: int, j: int, n: int) -> int:
    return min(abs(i - j), n - abs(i - j))


def _effective_hamiltonians(
    molecule: Molecule, basis_name: str, seeds: int
) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    """(H_eff^alfa, H_eff^beta, <S^2>, E_UHF, arama ozeti).

    Fock matrisleri UHF cozumunun yogunluklarindan YENIDEN kurulur --
    `UHFResult` onlari saklamaz, yalnizca katsayi ve yogunluk tasir.
    """
    basis = build_basis(molecule, basis_name)
    overlap = overlap_matrix(basis)
    core = core_hamiltonian(basis, molecule)
    eri = eri_tensor(basis)

    report = search_uhf(molecule, basis_name, seeds=seeds)
    result = report.result
    p_alpha, p_beta = result.density_alpha, result.density_beta
    coulomb = np.einsum("uvls,ls->uv", eri, p_alpha + p_beta)
    fock_alpha = core + coulomb - np.einsum("ulvs,ls->uv", eri, p_alpha)
    fock_beta = core + coulomb - np.einsum("ulvs,ls->uv", eri, p_beta)

    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    x = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T
    return (
        x @ fock_alpha @ x,
        x @ fock_beta @ x,
        float(result.s_squared),
        float(report.energy),
        report.as_line(),
    )


def downfold_uhf(
    molecule: Molecule,
    basis_name: str = "sto-3g",
    *,
    n_neighbour: int = 2,
    seeds: int = 12,
) -> DownfoldedParameters:
    """Halka molekulunden iki-siteli antiferromanyetik TB parametreleri.

    `n_neighbour` kac hoplama okunacagini belirler; varsayilan 2. Menzil 2'de
    kesilmesinin gerekcesi OLCULMUSTUR (bkz. tests/tb/test_downfold_kesme.py):
    t2 hatayi ~3 kat dusurur, t3 ve otesi RHF tarafinda hicbir sey katmiyordu.
    """
    n_site = len(molecule.atoms)
    if n_site % 2 != 0:
        raise UnsupportedSystemError(
            f"iki-siteli hucre CIFT sayida site ister, verilen: {n_site}. "
            "Tek sayida sitede kademeli alan tanimli degildir (frustrasyon)."
        )
    if n_neighbour < 1 or n_neighbour > n_site // 2:
        raise InvalidInputError(
            f"n_neighbour 1 ile {n_site // 2} arasinda olmali, "
            f"verilen: {n_neighbour}"
        )

    h_alpha, h_beta, s_squared, energy, line = _effective_hamiltonians(
        molecule, basis_name, seeds
    )
    if h_alpha.shape[0] != n_site:
        raise UnsupportedSystemError(
            f"site-bazli indirgeme atom basina TEK baz fonksiyonu ister: "
            f"{n_site} atom icin {h_alpha.shape[0]} fonksiyon uretildi "
            f"({basis_name!r}). Yalnizca 'sto-3g' desteklenir."
        )

    diagonal_alpha = np.diag(h_alpha)
    diagonal_beta = np.diag(h_beta)
    residual = float(np.abs(diagonal_alpha - np.roll(diagonal_beta, 1)).max())

    epsilon_bar = float(np.mean(diagonal_alpha))
    delta = float(np.mean(diagonal_alpha[0::2] - diagonal_alpha[1::2])) / 2.0

    hoppings = tuple(
        float(
            np.mean(
                [
                    h_alpha[i, j]
                    for i in range(n_site)
                    for j in range(n_site)
                    if _ring_distance(i, j, n_site) == m
                ]
            )
        )
        for m in range(1, n_neighbour + 1)
    )

    return DownfoldedParameters(
        epsilon_bar=epsilon_bar,
        delta=delta,
        hoppings=hoppings,
        n_site=n_site,
        basis_name=basis_name,
        s_squared=s_squared,
        energy_uhf=energy,
        staggering_residual=residual,
        search_line=line,
    )
