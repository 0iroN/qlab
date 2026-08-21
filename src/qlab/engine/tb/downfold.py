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
  * KABUK ORTALAMASI GERI ALINAMAZ ve modelin TAVANINI o belirler. Model her
    halka mesafesi m icin TEK bir t_m tasir; H_eff^alfa ise CIFT m'de A-A
    ile B-B'yi ayirir (alfa kanalinda A alt-orgusu alfa fazlasi, B beta
    fazlasi tasir ve ikisi esdeger DEGILDIR). Tek m'de kabuk tekbicimdir
    (o hoplamalar A<->B'yi baglar). OLCULDU, kabuk ici yayilim
    (N=8, d=2.0): m=1 3.004e-09, m=2 5.106e-02, m=3 1.007e-09,
    m=4 2.416e-02. Sonucu: menzil TAMAMEN acilsa bile hata sifirlanmaz --
    N=8 d=2.0'da tam menzilde 0.063140 hartree kaliyor. Menzili buyutmek
    bu tavani DELMEZ; delmek icin gereken sey alt-orguya bagli iki ayri
    cift-menzil hoplamasidir ve o, dort parametreli modelin DISINDADIR.
    (bkz. tests/tb/test_downfold_kesme.py)
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
from qlab.engine.tb.hamiltonian import Hopping, TBModel
from qlab.engine.tb.lattice import Lattice
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
    kesilmesinin gerekcesi OLCULMUSTUR (bkz. tests/tb/test_downfold_kesme.py).

    "T2 HATAYI ~3 KAT DUSURUR" CUMLESI DUZELTILDI: bu dokstring once oyle
    diyordu, ama alti sistemde olculen oran 2.25 / 2.32 / 2.57 / 2.61 /
    2.91 / 3.26 -- yani 3 kat yalnizca BIR sistemde tutuyor, ortalama 2.65.
    Dogru cumle "t2 hatayi IKI ILA UC KAT dusurur"dur.

    t3 icin de RHF tarafindan devralinan "hicbir sey katmiyordu" hukmu
    UHF'te olculdu ve NIYANSLI cikti: N=8 d=2.0'da max metriginde dokuz
    hanede hicbir sey katmiyor (0.066286861 -> 0.066286861) ama rms'te
    %5.7 katiyor. Menzil yine de 2'de kesilir, cunku modelin tavanini
    menzil degil KABUK ORTALAMASI belirliyor (modul dokstringi).
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


def model_from_downfold(
    params: DownfoldedParameters, spin: str = "alpha"
) -> TBModel:
    """Indirgenmis parametrelerden iki atomlu hucreli `TBModel`.

    Hucre IKI sitedir (A, B) ve kademeli alan `spin` ile YON degistirir:
    alfa kanalinda A yukselir B duser, beta kanalinda tersi. Hoplama
    spinden BAGIMSIZDIR (olculdu: t^alfa = t^beta, sekiz bagin sekizinde).

    Orgu vektoru 2.0 secilir ve KEYFIDIR: `finite_ring_matrix` yalnizca
    hucre indislerini modulo baglar, mesafeyi kullanmaz. Gercek geometri
    zaten indirgemeye girdi olmustur.

    ANTIPODAL MENZILDE HOPLAMA YARIYA BOLUNUR ve bu bir ayar degil bir
    DUZELTMEDIR. Halka mesafesi m < n/2 iken her sitenin IKI ortagi vardir
    (i+m ve i-m) ve asagidaki dort (tek m) / iki (cift m) kayit onlari birer
    kez baglar. m = n/2 iken iki ortak CAKISIR: ayni kayitlar ayni cifte
    duser ve `finite_ring_matrix` onlari TOPLAR, yani matrise t yerine 2t
    girer. OLCULDU (duzeltmeden once, gercek UHF t_{n/2} degerleriyle):
        N=6,  m=3, (0,3): +0.006946 yazildi, +0.003473 olmaliydi
        N=8,  m=4, (0,4): +0.006293 yazildi, +0.003147 olmaliydi
        N=10, m=5, (0,5): -0.005788 yazildi, -0.002894 olmaliydi
    Ucunde de TAM iki kat, yani hata bir isaret ya da kayma degil saf
    cift-sayimdir. Bolme, kayit sayisini azaltmaya YEGLENIR cunku sayimi
    tek yerde ve okunur tutar.
    """
    if spin not in ("alpha", "beta"):
        raise InvalidInputError(f"spin 'alpha' veya 'beta' olmali: {spin!r}")
    if len(params.hoppings) > params.n_site // 2:
        raise InvalidInputError(
            f"menzil yarim halkayi asamaz: {len(params.hoppings)} hoplama "
            f"verildi ama {params.n_site} sitelik halkada en buyuk mesafe "
            f"{params.n_site // 2}. Daha uzagi m ile n-m'yi ayni cifte "
            "yazardi (ortusme)."
        )
    yon = 1.0 if spin == "alpha" else -1.0

    lattice = Lattice(
        vectors=np.array([[2.0, 0.0, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    onsite = np.array(
        [
            params.epsilon_bar + yon * params.delta,
            params.epsilon_bar - yon * params.delta,
        ]
    )

    hoppings: list[Hopping] = []
    for menzil, ham_t in enumerate(params.hoppings, start=1):
        # Antipodal kabukta her cift IKI kez yaziliyor -- dokstringe bakiniz.
        t = ham_t / 2.0 if 2 * menzil == params.n_site else ham_t
        if menzil % 2 == 1:
            # TEK menzil: A ile B'yi baglar. Hucre icinde ve komsu hucreye.
            kayma = menzil // 2
            hoppings.append(Hopping(0, 1, (kayma, 0, 0), t))
            hoppings.append(Hopping(1, 0, (-kayma, 0, 0), t))
            hoppings.append(Hopping(1, 0, (kayma + 1, 0, 0), t))
            hoppings.append(Hopping(0, 1, (-kayma - 1, 0, 0), t))
        else:
            # CIFT menzil: ayni alt-orguyu baglar (A-A ve B-B).
            kayma = menzil // 2
            for orbital in (0, 1):
                hoppings.append(Hopping(orbital, orbital, (kayma, 0, 0), t))
                hoppings.append(Hopping(orbital, orbital, (-kayma, 0, 0), t))

    return TBModel(lattice=lattice, onsite=onsite, hoppings=tuple(hoppings))
