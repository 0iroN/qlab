"""G7 -- UHF indirgemesinin (Modul 3H) HIZLI SERITTEKI cipalari.

NEDEN BU DOSYA VAR (G7, 09.10.2026). `tests/tb`nin `-m "not slow"` seridinde
bant / Bloch / Mermin / radyal yasa tarafinin bol fizik cipasi var (analitik
dispersiyon, grafen uclari, F = -grad Omega, P(x) yasasi, W^2/mu2 = 4z ...).
Indirgeme tarafinin (`downfold.py`) DOGAYA bakan iki iddiasi ise yalniz `slow`
seritteydi: kademeli alanin KAREKOK yasasi (`test_downfold_cipa.py`) ve
kademeli alanin ISARET yasasi (`test_downfold.py::test_DELTA_ISARETI_...`).
OLCULDU (`.superpowers/sdd/araclar/mut.py`; kayitlar
`.superpowers/sdd/g7-olcum/duzeltme/tb/`):

  * M5 -- UHF Fock'unda degistokus terimini OBUR spinin yogunluguyla kuran
    mutant (`fock_alpha` ... `p_beta`, `fock_beta` ... `p_alpha`): tb hizli
    seridinin (bu dosya haric) 385 testinin HICBIRINI kirmadi; onu slow
    `test_DELTA_ISARETI_SPIN_POPULASYONUNA_BAGLI` olduruyor.
  * M6 -- Lowdin'i S^-1/2 yerine S^-1 ile yapan mutant: tb hizli seridinde
    ilk kirmizi bir OLCULMUS REGRESYON pini
    (`test_KADEMELI_ALAN_ve_ortalama_yerinde_enerji`); tb hizli seridinin
    fizik cipasi alt kumesi (99 test) YESIL kaldi.

KAPSAM YALNIZ tb'DIR. Depo genelinde tablo farkli (OLCULDU, ayni kayitlar):
`tests/bridge/test_dusen_adaylar.py::test_HF_TB_LOWDIN_ADIMI_KAYIPSIZ_...`
M6'yi bir OZDESLIKLE zaten olduruyor (Lowdin spektrumu = UHF orbital
enerjileri, esik 1e-8; mutantta 0.544); M5'i ise yalniz 1e-10 gurultu
tabanindaki bir regresyon pini ile (7.19e-10 -> 6.88e-10, bagil 4.4e-2 >
1e-3). Yani M5 acigi depo genelinde de gercekti; M6 icin "hizli seritte
yalniz regresyon" cumlesi yalniz tb icindir.

Iki test de N = 4 hidrojen halkasinda (STO-3G, 4 baz fonksiyonu, d = 2.6
bohr) kosar: bir `search_uhf` (4 tohum) 0.026 s, N = 8 / 12 tohumun
(0.88 s) ~1/34'u (bu makine, yuk altinda). Ikisi de DAIRESEL DEGILDIR:
referans indirgemenin kendi kod yolundan (Fock'u yeniden kurma + kendi X
matrisi) gelmez. Ama SINIFLARI FARKLIDIR ve her testte yazilidir: birincisi
bir FIZIK CIPASI (Hubbard modeli), ikincisi bir CAPRAZ-MODUL OZDESLIGI
(transkripsiyon sinifi; G7 sayiminda fizik cipasi SAYILMAZ).

Halka yardimcisi KOPYADIR (`downfold.hydrogen_ring` ithal edilmez): paylasilan
bir yardimciyi bozan mutant iki tarafi da ayni yonde bozar ve test yesil kalir
(gerekce `test_downfold_kesme.py`de olculdu). Lowdin karekoku de testin KENDI
ozayrisimidir; indirgemenin X'i kullanilmaz (M6 tam olarak orayi bozar).

OZEL ITHAL: `_effective_hamiltonians` alt cizgilidir ve BILEREK ithal edilir.
H_eff^alfa / H_eff^beta MATRISLERINI donduren bir kamu API'si yoktur
(`downfold_uhf` yalniz skaler parametre dondurur) ve ikinci testin iki
ozdesligi matrisleri ister. Emsali: ayni `_effective_hamiltonians` ithalini
yapan tests/tb/test_downfold.py, tests/tb/test_downfold_kesme.py ve
tests/bridge/test_dusen_adaylar.py (satir numarasiz: atif kaymasin).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import core_hamiltonian, eri_tensor, overlap_matrix
from qlab.engine.hf.multistart import search_uhf
from qlab.engine.tb import downfold as downfold_modulu
from qlab.engine.tb.downfold import _effective_hamiltonians, downfold_uhf
from qlab.types import Molecule
from qlab.units import BOHR_TO_ANGSTROM

HALKA = 4
MESAFE_BOHR = 2.6
TOHUM = 4
BAZ = "sto-3g"


def _halka(n: int, d: float) -> Molecule:
    """Komsu araligi d bohr olan duzgun N-halka (yerel kopya, bkz. dokstring)."""
    yaricap = d / (2 * np.sin(np.pi / n))
    a = yaricap * BOHR_TO_ANGSTROM
    return Molecule.from_string(
        "; ".join(
            f"H {a*np.cos(2*np.pi*k/n):.17g} {a*np.sin(2*np.pi*k/n):.17g} 0"
            for k in range(n)
        )
    )


def _karekokler(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S^1/2, S^-1/2), testin KENDI ozayrisiminden."""
    w, v = np.linalg.eigh(S)
    return v @ np.diag(np.sqrt(w)) @ v.T, v @ np.diag(1.0 / np.sqrt(w)) @ v.T


def _spin_cevir(r):
    """(P^a, P^b) -> (P^b, P^a). n_alpha = n_beta iken AYNI enerjili TAM UHF cozumu."""
    return dataclasses.replace(
        r,
        density_alpha=r.density_beta,
        density_beta=r.density_alpha,
        coefficients_alpha=r.coefficients_beta,
        coefficients_beta=r.coefficients_alpha,
        orbital_energies_alpha=r.orbital_energies_beta,
        orbital_energies_beta=r.orbital_energies_alpha,
    )


def _hubbard_orani(p, sonuc, U: float, S_yarim: np.ndarray) -> tuple[float, float]:
    """(-2 Delta / (U m_A), m_A). On kosullar burada sinanir.

    Cozum gercekten iki alt-orgulu ve spin-kirik olmali; aksi halde oran
    tanimsizdir (m = 0) ve test fizik degil gurultu okur.
    """
    spin = np.diag(S_yarim @ (sonuc.density_alpha - sonuc.density_beta) @ S_yarim)
    m_cift = float(np.mean(spin[0::2]))
    assert p.staggering_residual < 1e-7, (
        f"iki alt-orgulu yapi yok: kaydirma artigi {p.staggering_residual:.3e}"
    )
    assert abs(m_cift) > 0.5, f"cozum spin-kirik degil: m_A = {m_cift:+.6f}"
    assert np.abs(spin[0::2] + spin[1::2]).max() < 1e-6, spin  # m_B = -m_A
    return -2.0 * p.delta / (U * m_cift), m_cift


def test_G7_HUBBARD_ORTALAMA_ALANI_Delta_eksi_U_m_bolu_iki(monkeypatch):
    """SINIF: FIZIK CIPASI, DAIRESEL DEGIL -- Hubbard ortalama alan cozumu.

    FIZIK. Hubbard modelinin Hartree-Fock (ortalama alan) cozumunde alfa
    elektronunun yerinde enerjisi eps + U <n_beta>'dir: degistokus yalniz AYNI
    spinden gelir, yani alfa enerjisi BETA fazlasi olan sitede yukselir. Iki
    alt-orgulu antiferromanyette yuk duzgun (n_A = n_B = 1) oldugu icin
    n_A,beta - n_B,beta = -m_A ve

        Delta = (eps_A - eps_B) / 2 = -U m_A / 2,   yani  -2 Delta / (U m_A) = 1.

    Uc buyukluk UC AYRI yerden gelir:
      Delta : `downfold_uhf` (indirgemenin Fock'u + kendi Lowdin'i)
      U     : Lowdin bazinda (00|00) -- ERI'den, testin KENDI S^-1/2'siyle
      m_A   : Lowdin spin populasyonu diag(S^1/2 (P^a - P^b) S^1/2), cift
              siteler -- testin KENDI `search_uhf` kosusundan

    IKI ISARET SEKTORU, SPIN CEVIRMEYLE (denetim, YUKSEK bulgu). Tek bir SCF
    cozumu tek bir isaret sektoru gorur (burada Delta > 0). Bu dosyanin
    onceki surumu bu yuzden Delta'yi `abs()` ile pozitife sabitleyen mutanti
    (N8) KACIRIYORDU; tb hizli seridinin geri kalani da kaciriyor (bu dosya
    haric 385 passed, `mut_N8_tb_hizli_yenisiz.log`). Hangi AF cozumunun
    secildigi BLAS cekirdegine de bagli (OLCULDU, `isaret_blas.log`:
    N = 4, d = 3.2, 6 tohum varsayilan cekirdekte Delta < 0, Haswell'de > 0;
    `mut_N8_slow_sandybridge.log`: Sandybridge'de slow `test_DELTA_ISARETI`
    iki sisteminde de ayni isareti goruyor ve N8'i KACIRIYOR). Ikinci bir
    sistem ya da tohum aramak bu kirilganligi tekrarlardi. Onun yerine:
    n_alpha = n_beta iken UHF enerji fonksiyoneli alfa <-> beta takasinda
    DEGISMEZ, yani (P^a, P^b) -> (P^b, P^a) ayni enerjili TAM bir UHF
    cozumudur. Indirgemenin icindeki `search_uhf` sarilir (monkeypatch),
    arama bir kez daha kosar (~0.03 s) ve sonucu takas edilir; cevrik
    cozum indirgenir. Beklenen Delta -> -Delta, m_A -> -m_A, oran AYNI.
    Iki cozumde de AYNI assert kosar; hangi isaretin once secildigi
    onemsizlesir.

    OLCULDU (`olcum_tam.log`, `olcum_blas.log`; N = 4, STO-3G, 4 tohum):
        d = 2.6:  Delta = +0.309816 (cevrik -0.309816), U = 0.823486,
                  m_A = -0.762080 (cevrik +0.762080), oran 0.9874 IKISINDE
                  de; 7 OpenBLAS cekirdeginde 0.987364. Kaydirma artigi
                  <= 1.8e-15 (cekirdege bagli), <S^2> = 1.2747.
        d taramasi (N = 4): 1.8 -> 0.9715, 2.0 -> 0.9764, 2.2 -> 0.9807,
                  3.0 -> 0.9918, 3.6 -> 0.9958 (atomik sinira alttan yaklasir)
        N = 8, d = 2.0 (12 tohum): oran 0.9743, U = 0.883907 -- `downfold.py`
                  dokstringinin bagimsiz hesapla buldugu 0.883907411 ile ayni.
    Tohum 2 / 4 / 8'de Delta (6 hane) ve oran (4 hane) ayni. Tohum 1'de
    cozum RHF-benzeri (m_A = 0, kaydirma artigi 3.5e-2): on kosullar orada
    gercekten is gorur.

    Bu, `test_DELTA_ISARETI_SPIN_POPULASYONUNA_BAGLI`nin (slow, N = 8; sure
    dosyanin sonundaki notta) isaret iddiasini IKI SEKTORDE ve BUYUKLUK
    olarak (cozunurluk asagida) saniye altinda tasir.

    TOLERANS TEK TARAFLI: 0.95 < oran < 1.0 (denetim, DUSUK bulgu: iki
    tarafli |oran - 1| < 0.05 bandi Delta'yi %2.6 buyuten N5'i kaciriyordu).
      * UST SINIR 1.0 bir fizik ongorusudur ve OLCULMUSTUR. Delta Lowdin
        bazinda terim terim ayristirildi (`olcum_tam.log` bolum 3; terimlerin
        toplami Delta'yi yuvarlama duzeyinde, ~1e-15, kurar): yerinde terim
        (J_yer - K_yer = U n_beta) orani TAM 1 yapar; komsu Coulomb payi
        sifirdir (yuk duzgun, <n_A> = 1); sapmanin TAMAMI yerinde-olmayan
        degistokustan gelir. Baskin parcasi kapali formdadir: komsu dogrudan
        degistokus -(2 K_01 - K_02) / U; K_ij = (ij|ij)_L > 0 (ortusme
        yogunlugunun Coulomb oz-enerjisi) ve komsular TERS spin fazlasi
        tasidigi icin kademeyi AZALTIR. Test noktasinda (2 K_01 - K_02) / U
        = 0.012271 (K_01 = 0.005148, K_02 = 0.000192), kalan bag-degistokusu
        0.000365; toplam 0.012636 = 1 - 0.987364. On iki sistemde (N = 4,
        d = 1.6 ... 4.0; N = 6, d = 2.6 / 3.0; N = 8, d = 2.0 / 2.6) oran - 1
        HEP negatif (-0.0341 ... -0.0026); bag terimi N = 4, d >= 3.0'da
        isaret degistirir ama orada dogrudan terimin <= %4'u.
      * ALT SINIR 0.95: Hubbard-disi terimlere %5 pay. Test noktasi 0.9874;
        N = 4, d = 1.6 ... 4.0 taramasinin en kotusu 0.9659 (d = 1.6; 1.46
        kat pay).
    COZUNURLUK (durust olumsuz, mut.py `spec_olcum_sag.json`): Delta'yi
    ~%1.28'e kadar BUYUTEN (R2, %1.27: oran 0.99986) ya da ~%3.8'e kadar
    KUCULTEN (R1, %2.4: oran 0.9633) bir okuma hatasi bu assert'ten SAG
    cikar.

    KIRMA (mut.py `spec_g7_tb.json` ile olculdu; sayilar `olcum_tam.log`):
      M5 degistokus spin takasi -> oran -0.9874 (asil ve cevrik)   KIRMIZI
      M6 Lowdin S^-1            -> oran +1.0917 (ust sinirin 0.092 ustu)
      N1 Coulomb kendi spini x2 -> oran +0.1059
      N2 Coulomb yalniz 2 P^a   -> on kosul: kaydirma artigi 0.553 (oran 0.1059)
      N3 degistokus yerine J    -> oran +0.4407
      N5 Delta / 1.95           -> oran +1.0127 (eski bantta YESILDI)
      N8 abs(Delta)             -> asil +0.9874 YESIL, cevrik -0.9874 KIRMIZI
      N4 Lowdin usu -0.45       -> oran +0.9776 YESIL; onu enerji testi ve
                                   asagidaki iz isirigi olduruyor
      N7 eps_bar yalniz A       -> oran etkilenmez; iz isirigi +0.310

    EK ISIRIK -- OKUMA KATMANI (denetim, DUSUK bulgu; CAPRAZ-MODUL iz
    ozdesligi, fizik cipasi SAYILMAZ). eps_bar = tr(H_eff^alfa) / N =
    ort(eps^alfa): Lowdin benzerlik donusumu F C = S C eps genellestirilmis
    ozdeger probleminin spektrumunu korur. Referans `solve_uhf`in KENDI
    orbital enerjileridir. Taban SCF'tendir: eps^alfa son (DIIS'li) Fock'un
    ozdegerleridir, F(P_son)'unki degil. OLCULDU: test noktasinda |fark|
    <= 7.5e-10 (7 cekirdek); 26 geometri (N = 4, 6; d = 1.6 ... 4.0) x 7
    cekirdek en kotusu 1.11e-07 (Nehalem). Tolerans 1e-6 (en kotunun ~9
    kati). Oldurdugu: N7 +0.310, M6 +0.148, N3 -0.484, N4 -0.0134.

    KOR NOKTA (durust olumsuz): hoplama okumasi (t_m) bu dosyada CIPALANMADI.
    Butun t_m'nin isaretini ceviren mutant (N6) bu dosyadan SAG cikar
    (OLCULDU). t1'in isareti iki alt-orgulu halkada bir ayar secimidir;
    t2'ninki fizikseldir ama test noktasinda t2 = +0.0037 (t1 = -0.2272).
    """
    mol = _halka(HALKA, MESAFE_BOHR)
    p = downfold_uhf(mol, BAZ, seeds=TOHUM)
    sonuc = search_uhf(mol, BAZ, seeds=TOHUM).result
    baz = build_basis(mol, BAZ)
    S_yarim, X = _karekokler(overlap_matrix(baz))
    x0 = X[:, 0]
    U = float(np.einsum("a,b,c,d,abcd->", x0, x0, x0, x0, eri_tensor(baz)))

    # Spin cevirme yalniz n_alpha = n_beta iken ayni problemin cozumudur.
    assert sonuc.n_alpha == sonuc.n_beta, (sonuc.n_alpha, sonuc.n_beta)
    cagri: list[int] = []

    def _cevrik_arama(*a, **k):
        cagri.append(1)
        rapor = search_uhf(*a, **k)
        return dataclasses.replace(rapor, result=_spin_cevir(rapor.result))

    monkeypatch.setattr(downfold_modulu, "search_uhf", _cevrik_arama)
    p_cevrik = downfold_uhf(mol, BAZ, seeds=TOHUM)
    assert len(cagri) == 1, (
        f"spin cevirme sarmali {len(cagri)} kez cagrildi: indirgeme aramayi "
        "artik `downfold.search_uhf` adiyla yapmiyor, cevrik kol etkisiz"
    )

    for etiket, pp, ss in (
        ("asil", p, sonuc),
        ("spin-cevrik", p_cevrik, _spin_cevir(sonuc)),
    ):
        oran, m_cift = _hubbard_orani(pp, ss, U, S_yarim)
        assert 0.95 < oran < 1.0, (
            f"{etiket} cozumde Hubbard ortalama alani tutmuyor: "
            f"-2 Delta / (U m_A) = {oran:+.4f} "
            f"(Delta = {pp.delta:+.6f}, U = {U:.6f}, m_A = {m_cift:+.6f})"
        )

    eps_fark = p.epsilon_bar - float(np.mean(sonuc.orbital_energies_alpha))
    assert abs(eps_fark) < 1e-6, (
        f"eps_bar iz ozdesligi tutmuyor: eps_bar - ort(eps^alfa) = {eps_fark:+.3e} "
        f"(eps_bar = {p.epsilon_bar:+.9f})"
    )


def test_G7_UHF_ENERJISI_indirgenmis_Fock_matrisleriyle_KESIN_kuruluyor():
    """SINIF: CAPRAZ-MODUL OZDESLIGI (transkripsiyon sinifi) -- fizik cipasi DEGIL.

    NEDEN CIPA DEGIL (denetim, ORTA bulgu). Referans E_UHF `solve_uhf`in SCF
    biriktiricisinden gelir ve orada Fock, `downfold.py`dekiyle AYNI
    Pople-Nesbet formulu ve AYNI einsum desenleriyle ("uvls,ls->uv",
    "ulvs,ls->uv") kurulur. Yani bu test iki modulun ayni Fock'u ayni
    bicimde yazdigini dogrular; IKISINDE AYNI yapilan bir hataya KORDUR.
    Depo ayni yapiyi `uhf.energy_from_density`de "CIPA DEGIL, TRANSKRIPSIYON
    DENETIMI" diye siniflar. `solve_uhf`in kendisi hf modulunde disariya
    baglidir (or. `tests/hf/test_uhf.py::test_AYRISMA_LIMITI_iki_serbest_atom`,
    2 E_H(STO-3G)). DAIRESEL degildir: indirgemeyle ortak kodu yoktur ve
    indirgemenin Fock'unu / X'ini kullanmaz; P_L ve h_L testin KENDI
    S^1/2 / S^-1/2'siyle kurulur.

    IKI OZDESLIK:
      (1) ENERJI: E_UHF = 1/2 sum_sigma tr[P^sigma (h + F^sigma)] + V_nn.
          Iz benzerlik donusumunde degismez: P_L = S^1/2 P S^1/2,
          h_L = S^-1/2 h S^-1/2, H_eff^sigma = S^-1/2 F^sigma S^-1/2 ile ayni
          ifade Lowdin bazinda TAM tutar.
      (2) DURAGANLIK (Roothaan): F^sigma P^sigma S = S P^sigma F^sigma, yani
          [H_eff^sigma, P_L^sigma] = 0. Bu matris TAM OLARAK
          S^-1/2 (F P S - S P F) S^-1/2'dir; `solve_uhf` durmak icin onun
          max|.|'sini residual_tolerance = 1e-8'in altina indirir. Yani sinir
          yalniz olculmus degil, yakinsama olcutuyle GARANTILIDIR.
    (2) neden eklendi: (1) N2'ye (Coulomb'u 2 P^alfa'dan kuran mutant)
    KORDUR -- enerji farki yuvarlama duzeyinde (olculen -8.9e-16), cunku AF simetrisinde
    tr(P_toplam J[P_spin]) = 0. (2) N2'yi 1.163e-01 ile gorur.

    OLCULDU (`olcum_tam.log`, `olcum_blas.log`; 26 geometri: N = 4 ve 6,
    d = 1.6 ... 4.0 bohr, spin-kirik ve kirilmamis; 7 OpenBLAS cekirdegi):
      (1) en buyuk |fark| 2.49e-14 (Prescott); test noktasinda (N = 4,
          d = 2.6) |fark| <= 4.4e-15 (7 cekirdegin 5'inde 0). Yalniz
          yuvarlama: `solve_uhf` enerjiyi dondurdugu yogunluklarla kurdugu
          Fock'tan hesaplar, yani artik toleransina BAGLI DEGILDIR.
      (2) en buyuk 8.8e-09 (< 1e-8, garantiyle tutarli); test noktasinda
          cekirdege gore 1.2e-12 ... 8.3e-09.
    TOLERANS (1) 1e-13 hartree: en kotunun 4.0 kati, test noktasinda >= 22
    kat. (2) 1e-7: yakinsama olcutunun 10 kati.

    KIRMA (mut.py `spec_g7_tb.json`; sayilar `olcum_tam.log`):
                                      (1) enerji farki   (2) komutator
      M5 degistokus spin takasi        +9.731e-01        2.381e-01
      M6 Lowdin S^-1                   +1.555e-01        1.883e-02
      N1 Coulomb kendi spini x2        +4.232e-01        1.163e-01
      N2 Coulomb yalniz 2 P^alfa       -8.9e-16 (KOR)    1.163e-01
      N3 degistokus yerine J           -5.652e-01        2.130e-02
      N4 Lowdin usu -0.45              -2.152e-02        6.493e-04
    Okuma katmani mutantlari (N5, N7, N8, N6) bu teste TASARIM GEREGI
    gorunmez (`downfold_uhf`i okumaz); onlar birinci testin isidir.

    ON KOSUL: spin-kirik cozum (<S^2> > 0.5). Kirilmamis cozumde P^a = P^b ve
    M5 bu ozdesliklerde GORUNMEZ olurdu.

    `slow` DEGIL: olculen sure dosyanin sonundaki notta.
    """
    mol = _halka(HALKA, MESAFE_BOHR)
    h_alfa, h_beta, s_kare, _, _ = _effective_hamiltonians(mol, BAZ, TOHUM)
    sonuc = search_uhf(mol, BAZ, seeds=TOHUM).result
    baz = build_basis(mol, BAZ)
    S_yarim, X = _karekokler(overlap_matrix(baz))

    assert s_kare > 0.5, f"cozum spin-kirik degil: <S^2> = {s_kare:.4f}"

    h_L = X @ core_hamiltonian(baz, mol) @ X
    pa_L = S_yarim @ sonuc.density_alpha @ S_yarim
    pb_L = S_yarim @ sonuc.density_beta @ S_yarim
    kurulan = (
        0.5 * (float(np.sum(pa_L * (h_L + h_alfa))) + float(np.sum(pb_L * (h_L + h_beta))))
        + mol.nuclear_repulsion()
    )
    fark = kurulan - sonuc.energy_total
    assert abs(fark) < 1e-13, (
        f"indirgenmis Fock UHF enerjisini kurmuyor: fark {fark:+.3e} hartree "
        f"(E_UHF = {sonuc.energy_total:.12f})"
    )

    komutator = max(
        float(np.abs(h_alfa @ pa_L - pa_L @ h_alfa).max()),
        float(np.abs(h_beta @ pb_L - pb_L @ h_beta).max()),
    )
    assert komutator < 1e-7, (
        f"indirgenmis Fock UHF cozumunde duragan degil: "
        f"max|[H_eff, P_L]| = {komutator:.3e}"
    )


# --------------------------------------------------------------------------
# SURE (OLCULDU, 09.10.2026, `pytest --durations=0`, bu makine, YUK ALTINDA
# -- ayni anda baska ajanlar test kosuyordu; `.superpowers/sdd/g7-olcum/
# duzeltme/tb/sure.log`, 4 kosu): Hubbard testi 0.10 s call (spin-cevrik kol
# dahil), enerji/duraganlik testi 0.06-0.08 s call; setup/teardown < 0.005 s.
# Ikisi de G7'nin 0.3 s hedefinin altinda; `slow` isareti YOK. Karsilastirma:
# slow `test_DELTA_ISARETI_SPIN_POPULASYONUNA_BAGLI` ayni kayitta 3.06 s.
# --------------------------------------------------------------------------
