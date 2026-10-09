"""Ic (internal, RHF -> RHF) kararlilik analizi.

NEDEN VAR: yakinsamis bir SCF cozumu enerjinin bir DURAGAN noktasidir, ama
minimum olmak zorunda degildir. Bu depoda olculmus somut ornek (bkz.
tests/hf/test_physical.py::test_large_separation_breaks_down_numerically_and_we_say_so):
H2/STO-3G'de ~12 A ayrimda iki atomik orbital arasindaki eslesme ~1e-20'ye
duser, kosegen elemanlarin yuvarlama farkinin (~1e-17) altina iner ve eigh
dejenere altuzayda keyfi bir baz secer. SCF o zaman simetrik cozum yerine
YERELLESMIS (H- H+) cozume yakinsar. O cozum gercek bir duragan noktadir --
converged=True yalan degildir -- ama bir EYER NOKTASIDIR: isgal orbitalini
bos orbitale dogru dondurmek enerjiyi dusurur. Kararlilik analizi olmadan bu
bilgi ciktinin HICBIR yerinde gorunmuyordu.

KAYNAK VE FORMULASYON
---------------------
Kullanilan buyukluk, tekil (singlet) orbital dondurme Hessian'idir. Szabo &
Ostlund, "Modern Quantum Chemistry", Bolum 3 (SCF kararliligi) ile
Helgaker-Jorgensen-Olsen, "Molecular Electronic-Structure Theory", Bolum 10
(orbital dondurme parametrelendirmesi ve E^(2)) ayni nesneyi verir; asagidaki
ifade bu iki kaynagin tekil A ve B matrislerinin toplamidir:

    A_[ai],[bj] = delta_ij delta_ab (eps_a - eps_i) + 2(ai|jb) - (ab|ji)
    B_[ai],[bj] = 2(ai|bj) - (aj|bi)

Gercek (reel) orbitallerde (pq|rs) hem ilk hem ikinci ciftte simetriktir,
dolayisiyla A+B sadelesir. Bu modul, formulu EZBERDEN almak yerine
KULLANILAN parametrelendirmeden dogrudan turetir -- cunku A/B ile "enerjinin
ikinci turevi" arasindaki carpan (2 mi, 4 mu) tamamen parametrelendirme
konvansiyonuna baglidir ve tam da bu carpan sessiz hata kaynagidir.

PARAMETRELENDIRME (sonlu fark testi bununla AYNI olmak zorunda):

    C(theta) = C exp(K),    K_ai = theta_ai,  K_ia = -theta_ai
               (K'nin isgal-isgal ve bos-bos bloklari sifir)

Turetme (E, P'de kuadratiktir: E = tr(P Hcore) + 1/2 tr(P G(P)), G dogrusal):

    dE/dtheta_p       = tr( dP/dtheta_p  F )
    d2E/dtheta_p dq   = tr( d2P/dtheta_p dtheta_q  F ) + tr( dP/dp  G(dP/dq) )

Yukaridaki ikinci ifade duragan olmayan noktalarda da GECERLIDIR (duraganlik
yalnizca birinci terimi degil, gradyani sifirlar). P = 2 C_occ C_occ^T ve
C(theta)'nin theta^2'ye kadar aciliminden:

    dP/dtheta_ai            = 2 ( C_a C_i^T + C_i C_a^T )
    d2P/dtheta_ai dtheta_bj = 2 [ delta_ij (C_a C_b^T + C_b C_a^T)
                                - delta_ab (C_i C_j^T + C_j C_i^T) ]

Bunlari yerine koyup MO bazina gecince:

    H_[ia],[jb] = 4 [ delta_ij F_ab - delta_ab F_ij
                      + 4(ai|bj) - (ab|ij) - (aj|ib) ]

Yani H = 4 (A+B); 4 carpani YUKARIDAKI theta parametrelendirmesine aittir,
A+B'nin kendisine degil. Kanit ezbere degil olcumdedir:
tests/hf/test_stability.py analitik H'yi bes noktali sonlu farkla
karsilastirir (H2 icin 1x1, su icin 10x10, kosegen disi elemanlar dahil).

F_ab / F_ij, MO bazindaki Fock matrisinin bos-bos ve isgal-isgal bloklaridir.
Kanonik orbitallerde bunlar kosegendir (eps_a, eps_i) ama burada BLOK olarak
kullanilir: solve() donen katsayilari son iterasyonda (DIIS aktifse
ekstrapole edilmis) Fock'tan uretir, dolayisiyla son yogunlugun Fock'u o
bazda tam kosegen olmayabilir. Blok kullanmak bu farki bedelsiz kapsar.

NE SINANIR, NE SINANMAZ
-----------------------
Sinanan UC kanal vardir, ucu de gercek (reel) isgal-bos dondurmeler:

  IC  (`orbital_hessian`)         tekil/singlet, RHF -> RHF. Negatif ozdeger =
                                  yakinda daha dusuk enerjili baska bir RHF
                                  cozumu var.
  DIS (`triplet_orbital_hessian`) uclu/triplet, RHF -> UHF. Negatif ozdeger =
                                  spin simetrisini kiran bir UHF cozumu daha
                                  dusuk enerjili.
  UHF (`uhf_orbital_hessian`)     UHF -> UHF, alfa ve beta BAGIMSIZ doner (B7,
                                  09.10.2026). Negatif ozdeger = UHF cozumu bir
                                  EYER; yakinda daha dusuk bir UHF cozumu var.
                                  Kapali kabukta ilk iki kanali ozel hal olarak
                                  icerir (ozdeslik testte kilitli).

Dis kanal Modul 1C'de eklendi. Eklenmeden once depo, spin kanalinda EYER olan
noktalara "ic minimum" diyordu ve bu sessizlikti: olculmus somut ornek, deponun
kendi PySCF referans kaydi h2_200'dur (H2, 2.00 A) -- ic kararlilik ozdegeri
+2.546683038854 ile "kararli" derken triplet ozdegeri -1.599532559297'dir ve
gercek HF taban durumu 0.153420178832 hartree ASAGIDADIR.

Sinanmayan: kompleks (reel olmayan) orbital kararsizliklari. Uc kanalda da
"stable=True", cozumun KURESEL minimum oldugunu soylemez -- ve UHF kanalinda
bu OLCULDU: H8 halkasi d = 3.0 bohr STO-3G'de kuresel minimumun 0.0799 ve
0.0810 Ha ustunde iki YEREL minimum daha var (tests/hf/test_uhf_kararlilik.py,
T7). Bu sinir
provenance'in ihmal listesinde de yazilidir (scf.py, _HF_NEGLECTED).

INDEKS SIRASI: dondurmeler (i, a) ciftleriyle numaralanir; satir/sutun
indeksi i * n_virtual + (a - n_occupied), yani isgal indeksi disda, bos
indeksi icte.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.provenance import StabilityVerdict

# Ozdegerin "gercekten negatif" sayilmasi icin gereken esik (hartree/rad^2).
# Gerekcesi olculmustur: bu depodaki sistemlerde en buyuk Hessian elemani ~81
# (su/STO-3G), dolayisiyla eigvalsh'in sayisal gurultusu ~eps*||H|| ~ 1e-14
# mertebesindedir; esik bunun ~6 mertebe ustunde. Olculen GERCEK kararsizlik
# (H2 12 A) ise -2.92, yani esigin ~8 mertebe UZERINDE buyuklukte. Iki taraf
# arasinda 14 mertebelik bir bosluk vardir; esik o boslugun ortasina degil
# gurultu tarafina yakin konur -- amac gercek kararsizligi kacirmamak,
# sifir cevresindeki yuvarlamayi kararsizlik saymamaktir.
#
# DUZELTME (B7, 09.10.2026) -- "14 mertebelik bosluk" CURUDU ve bant olculdu
# (kamu yolu; `.superpowers/sdd/b7-olcum/uygulama/u05_tol_bandi.log`):
#   * Gurultu tarafi ~1e-14 DEGIL. Sinirlayan eigvalsh yuvarlamasi degil, SCF
#     artiginin SIFIR MODLARA bulasmasidir: UHF kanalinda kararli bir cozumde
#     en buyuk olculen -2.579e-10 (HF/6-31G, 20 bohr). RHF ic kanalinda da:
#     search_rhf(8) ile N2/STO-3G'nin en iyisinde tek bir sifir civari ozdeger
#     1.50 A'da -1.913e-9 (esigin 5.2 kat icinde), 1.80 A'da -5.6e-11,
#     2.00 A'da +8.3e-11. Goldstone modu mu SCF artigi mi AYRISTIRILMADI.
#   * Kararsizlik tarafi da yalniz H2 12 A'nin -2.92'si degil: FIZIKSEL yumusak
#     modlar var. Olculen en kucuk GERCEK UHF kararsizligi -2.116e-7 (HF/STO-3G,
#     8 bohr, pi-delik eyeri; izleme 1.054e-7 Ha asagidaki sigma-delik
#     minimumuna iniyor).
# Yani bant [2.6e-10, 2.1e-7]; esik sifir-mod tarafina 39, gercek kararsizlik
# tarafina 21 kat uzak. Esik DEGISMEDI; |lambda| < 1e-8 yumusak modlar AYIRT
# EDILEMEZ ve bu bir beyandir (uhf.py, multistart.py).
EIGENVALUE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class StabilityResult:
    """Bir RHF cozumunun ic kararlilik analizi.

    stable            : en kucuk ozdeger > -EIGENVALUE_TOLERANCE mi
    lowest_eigenvalue : en kucuk Hessian ozdegeri; dondurme yoksa None
    eigenvalues       : tum ozdegerler, artan sirada (dondurme yoksa bos)
    n_rotations       : n_occupied * n_virtual
    """

    stable: bool
    lowest_eigenvalue: float | None
    eigenvalues: np.ndarray
    n_rotations: int

    def verdict(self, channel: str = "ic") -> StabilityVerdict:
        """Provenance'a ilistirilse makine-okunur (JSON'a yazilabilir) ozet.

        `channel` hukmun HANGI kanaldan geldigini tasir ("ic", "dis" ya da
        "uhf").
        StabilityResult'in kendisi kanal tasimaz -- ayni ozdeger analizi iki
        kanal icin de aynen gecerlidir; ayrim yalnizca RAPORLAMADA anlamlidir
        ve panelin dogru cumleyi kurmasi icin gereklidir.
        """
        return StabilityVerdict(
            stable=self.stable,
            lowest_eigenvalue=self.lowest_eigenvalue,
            n_rotations=self.n_rotations,
            channel=channel,
        )


def _transform_eri(
    eri: np.ndarray,
    coefficients: np.ndarray,
    coefficients_second: np.ndarray | None = None,
) -> np.ndarray:
    """AO -> MO iki-elektron integral donusumu, dort adimda (N^5).

    Tek bir N^8 einsum yerine dort ardisik N^5 kismi kasilma. Hessian'in
    ihtiyaci olan uc blok (ov|ov, vv|oo, ov|vo) butun MO indekslerini
    kullandigi icin tam tensor uretmek, uc ayri kismi donusum yazmaktan hem
    kisa hem karsilastirilabilir maliyettedir. Bellek N^4'tur -- solve()
    zaten AO ERI tensorunu ayni buyuklukte tutuyor, yeni bir asimptotik yuk
    getirmez.

    `coefficients_second` verilirse IKINCI cift (rs) o katsayilarla donusur:
    (p q | r s) = (C1_p C1_q | C2_r C2_s). UHF Hessian'inin karisik (alfa-beta)
    blogu tam olarak budur. Verilmezse ikinci cift de `coefficients` ile
    donusur ve islem bit bit eskisidir (RHF yollari bu dali kullanir).
    """
    second = coefficients if coefficients_second is None else coefficients_second
    step = np.einsum("uvls,up->pvls", eri, coefficients, optimize=True)
    step = np.einsum("pvls,vq->pqls", step, coefficients, optimize=True)
    step = np.einsum("pqls,lr->pqrs", step, second, optimize=True)
    return np.einsum("pqrs,st->pqrt", step, second, optimize=True)


def orbital_hessian(
    core: np.ndarray,
    eri: np.ndarray,
    coefficients: np.ndarray,
    n_occupied: int,
) -> np.ndarray:
    """Tekil (singlet) orbital dondurme Hessian'i, sekil (n_ov, n_ov).

    H_[ia],[jb] = 4 [ delta_ij F_ab - delta_ab F_ij
                      + 4(ai|bj) - (ab|ij) - (aj|ib) ]

    Turetme ve konvansiyon icin modul dokstringine bakin. Bos (veya isgal)
    orbital yoksa dondurulebilecek cift de yoktur; o durumda 0x0 bir matris
    doner -- bu bir hata degildir (He/STO-3G tam olarak boyledir).
    """
    n_orbitals = coefficients.shape[1]
    n_virtual = n_orbitals - n_occupied
    if n_occupied == 0 or n_virtual == 0:
        return np.zeros((0, 0))

    occupied = slice(0, n_occupied)
    virtual = slice(n_occupied, n_orbitals)

    occupied_coefficients = coefficients[:, occupied]
    density = 2.0 * occupied_coefficients @ occupied_coefficients.T
    coulomb = np.einsum("uvls,ls->uv", eri, density)
    exchange = np.einsum("ulvs,ls->uv", eri, density)
    fock_mo = coefficients.T @ (core + coulomb - 0.5 * exchange) @ coefficients

    mo_eri = _transform_eri(eri, coefficients)
    aibj = mo_eri[virtual, occupied, virtual, occupied]
    abij = mo_eri[virtual, virtual, occupied, occupied]
    ajib = mo_eri[virtual, occupied, occupied, virtual]

    hessian = (
        np.einsum("ij,ab->iajb", np.eye(n_occupied), fock_mo[virtual, virtual])
        - np.einsum("ab,ij->iajb", np.eye(n_virtual), fock_mo[occupied, occupied])
        + 4.0 * np.einsum("aibj->iajb", aibj)
        - np.einsum("abij->iajb", abij)
        - np.einsum("ajib->iajb", ajib)
    )
    n_rotations = n_occupied * n_virtual
    return 4.0 * hessian.reshape(n_rotations, n_rotations)


def triplet_orbital_hessian(
    core: np.ndarray,
    eri: np.ndarray,
    coefficients: np.ndarray,
    n_occupied: int,
) -> np.ndarray:
    """Uclu (triplet, RHF -> UHF) orbital dondurme Hessian'i, sekil (n_ov, n_ov).

    H_[ia],[jb] = 4 [ delta_ij F_ab - delta_ab F_ij - (ab|ij) - (aj|ib) ]

    `orbital_hessian` ile TEK yapisal fark: Coulomb terimi (+4(ai|bj)) DUSER.
    Sebep parametrelendirmededir -- bu kanalda alfa ve beta ZIT yonde doner
    (Ca = C exp(+tK), Cb = C exp(-tK)), toplam yogunluk birinci mertebede
    degismez, dolayisiyla Coulomb katkilari birbirini goturur; degis-tokus
    katkilari her spin icin AYRI yogunluktan geldigi icin kalir.

    Imza ve govde bilerek singlet surumun aynisidir ki iki ifade yan yana
    okunabilsin. Ayni sebeple MO donusumu burada TEKRAR yapilir: iki
    fonksiyonu tek bir govdede birlestirmek N^5 bir donusum kazandirirdi ama
    tam da karsilastirilmasi gereken iki satiri ic ice gecirirdi.

    Bu tekrarin bedeli OLCULDU ve ihmal edilebilir; dahasi sistem buyudukce
    oran DUSUYOR (cagri basina, ve onu ureten SCF'e orani):
      H2/STO-3G   (2 bf)  0.049 ms  = SCF'in %3.36'si
      H2O/STO-3G  (7 bf)  0.059 ms  = SCF'in %0.13'u
      H2O/6-31G  (13 bf)  0.144 ms  = SCF'in %0.04'u
    Sebep, SCF'in bu donusumu her iterasyonda degil, Hessian'in yalnizca bir
    kez yapmasidir.

    OLCULDU (bes noktali sonlu fark, h=1e-3, Ca = C exp(+tK) / Cb = C exp(-tK);
    tests/hf/test_dis_kararlilik.py): bagil max fark H2 R=1.4 1.305e-10,
    H2 R=3.0 4.200e-10, H4 R=2.0 3.815e-10, LiH 4.173e-10, H2O 6.659e-10.

    AYIRICI KONTROL: ayni sonlu fark makinesi alfa ile beta AYNI yonde
    dondurulunce `orbital_hessian`i verir (bagil fark < 1e-8) -- iki kanalin
    farkli seyler olctugu boyle kilitlenir.
    """
    n_orbitals = coefficients.shape[1]
    n_virtual = n_orbitals - n_occupied
    if n_occupied == 0 or n_virtual == 0:
        return np.zeros((0, 0))

    occupied = slice(0, n_occupied)
    virtual = slice(n_occupied, n_orbitals)

    occupied_coefficients = coefficients[:, occupied]
    density = 2.0 * occupied_coefficients @ occupied_coefficients.T
    coulomb = np.einsum("uvls,ls->uv", eri, density)
    exchange = np.einsum("ulvs,ls->uv", eri, density)
    fock_mo = coefficients.T @ (core + coulomb - 0.5 * exchange) @ coefficients

    mo_eri = _transform_eri(eri, coefficients)
    abij = mo_eri[virtual, virtual, occupied, occupied]
    ajib = mo_eri[virtual, occupied, occupied, virtual]

    hessian = (
        np.einsum("ij,ab->iajb", np.eye(n_occupied), fock_mo[virtual, virtual])
        - np.einsum("ab,ij->iajb", np.eye(n_virtual), fock_mo[occupied, occupied])
        # Coulomb terimi (+4 aibj) BURADA YOK -- singlet ile tek fark budur.
        - np.einsum("abij->iajb", abij)
        - np.einsum("ajib->iajb", ajib)
    )
    n_rotations = n_occupied * n_virtual
    return 4.0 * hessian.reshape(n_rotations, n_rotations)


def _uhf_same_spin_block(
    eri: np.ndarray,
    coefficients: np.ndarray,
    fock: np.ndarray,
    n_occupied: int,
) -> np.ndarray:
    """Ayni-spin blogu H^ss, sekil (n_o n_v, n_o n_v); bos/isgal yoksa 0x0.

    H^ss_[ia],[jb] = 2 [delta_ij f_ab - delta_ab f_ij]
                     + 4(ai|bj) - 2(ab|ij) - 2(aj|ib)

    Butun integraller AYNI spinin orbitallerindendir; f bu spinin MO Fock'u.
    """
    n_orbitals = coefficients.shape[1]
    n_virtual = n_orbitals - n_occupied
    if n_occupied == 0 or n_virtual == 0:
        return np.zeros((0, 0))
    occupied = slice(0, n_occupied)
    virtual = slice(n_occupied, n_orbitals)

    fock_mo = coefficients.T @ fock @ coefficients
    mo_eri = _transform_eri(eri, coefficients)
    aibj = mo_eri[virtual, occupied, virtual, occupied]
    abij = mo_eri[virtual, virtual, occupied, occupied]
    ajib = mo_eri[virtual, occupied, occupied, virtual]

    block = (
        2.0 * np.einsum("ij,ab->iajb", np.eye(n_occupied), fock_mo[virtual, virtual])
        - 2.0 * np.einsum("ab,ij->iajb", np.eye(n_virtual), fock_mo[occupied, occupied])
        + 4.0 * np.einsum("aibj->iajb", aibj)
        - 2.0 * np.einsum("abij->iajb", abij)
        - 2.0 * np.einsum("ajib->iajb", ajib)
    )
    n_rotations = n_occupied * n_virtual
    return block.reshape(n_rotations, n_rotations)


def uhf_orbital_hessian(
    core: np.ndarray,
    eri: np.ndarray,
    coefficients_alpha: np.ndarray,
    coefficients_beta: np.ndarray,
    n_alpha: int,
    n_beta: int,
) -> np.ndarray:
    """UHF orbital dondurme Hessian'i (reel alfa ve beta dondurmeleri).

    UCUNCU KANAL ("uhf", UHF -> UHF). Ic ve dis kanal bir RHF cozumunu
    sinar; bu, bir UHF cozumunun kendisinin YEREL minimum mu yoksa EYER mi
    oldugunu sinar. Kanal eklenmeden once depoda UHF -> UHF analizi YOKTU ve
    `search_uhf` HF molekulunde (H + F, 20 bohr, STO-3G) bir eyeri (S^2 = 1,
    lambda_min = -1.549729) sessizce "en iyi" diye donduruyordu.

    PARAMETRELENDIRME (sonlu fark testi bununla AYNI olmak zorunda):

        C_s(kappa) = C_s exp(K_s),  K_s[a, i] = kappa_ia,  K_s[i, a] = -kappa_ia
        s in {alfa, beta}; iki spin BAGIMSIZ doner.

    P_s = C_s,occ C_s,occ^T (spin basina, 2 carpani YOK). Modul dokstringindeki
    turetmenin aynisi spin basina yapilinca:

        H^ss_[ia],[jb] = 2 [delta_ij f^s_ab - delta_ab f^s_ij]
                         + 4(ai|bj)_ss - 2(ab|ij)_ss - 2(aj|ib)_ss
        H^ab_[ia],[jb] = 4 (a_alfa i_alfa | b_beta j_beta)

    Karisik blokta degis-tokus YOKTUR (alfa ve beta orbitalleri arasinda K
    terimi olmaz); yalniz Coulomb baglar. Karisik blok `_transform_eri`in
    iki katsayili dalindan gelir.

    INDEKS SIRASI: once alfa blogu (i * n_v_alfa + a), sonra beta blogu.
    Bir spinde dondurme yoksa (n_s = 0 ya da n_s = baz boyutu) o blok bos
    kalir; ikisi de bossa 0x0 doner.

    KAPALI KABUK OZDESLIGI (C_alfa = C_beta, n_alfa = n_beta):
        H^aa + H^bb + H^ab + H^ba = `orbital_hessian`        (singlet)
        H^aa + H^bb - H^ab - H^ba = `triplet_orbital_hessian` (triplet)
    -- yani bu fonksiyon iki eski kanali OZEL HAL olarak icerir.

    OLCULDU (tests/hf/test_uhf_kararlilik.py, T1/T2):
      * analitik gradyanin merkezi farkiyla (h = 1e-4): bes acik kabuk
        durumunda (HF@20'nin S^2 = 0 ve S^2 = 1 eyerleri, N atomunun zemini ve
        eyeri, H2 5 bohr kirik simetri) max|An - FD| / max|H| = 6.667e-9;
        h = 3e-4'te 6.000e-8 -- oran 9 = 3^2, yani fark O(h^2) KESMEDIR,
        gurultu degil;
      * kapali kabuk ozdesligi (H2 1.4 / 3.0 bohr, su): bagil <= 3.9e-16;
      * elle turetilmis bagimsiz bir kopyayla (elestirmen e01) ayni sekiz
        durumda bagil <= 8.2e-17.
    MALIYET: HF@20 10 x 10 0.22 ms, H14 halkasi 98 x 98 0.53 ms -- ayni
    sistemde bir `solve_uhf` kosusunun (30 ms / 284 ms) ~%1'inin altinda.
    """
    occupied_alpha = coefficients_alpha[:, :n_alpha]
    occupied_beta = coefficients_beta[:, :n_beta]
    density_alpha = occupied_alpha @ occupied_alpha.T
    density_beta = occupied_beta @ occupied_beta.T
    coulomb = np.einsum("uvls,ls->uv", eri, density_alpha + density_beta)
    fock_alpha = core + coulomb - np.einsum("ulvs,ls->uv", eri, density_alpha)
    fock_beta = core + coulomb - np.einsum("ulvs,ls->uv", eri, density_beta)

    block_alpha = _uhf_same_spin_block(eri, coefficients_alpha, fock_alpha, n_alpha)
    block_beta = _uhf_same_spin_block(eri, coefficients_beta, fock_beta, n_beta)
    size_alpha, size_beta = block_alpha.shape[0], block_beta.shape[0]

    hessian = np.zeros((size_alpha + size_beta, size_alpha + size_beta))
    hessian[:size_alpha, :size_alpha] = block_alpha
    hessian[size_alpha:, size_alpha:] = block_beta
    if size_alpha and size_beta:
        mixed_eri = _transform_eri(eri, coefficients_alpha, coefficients_beta)
        mixed = 4.0 * np.einsum(
            "aibj->iajb",
            mixed_eri[n_alpha:, :n_alpha, n_beta:, :n_beta],
        ).reshape(size_alpha, size_beta)
        hessian[:size_alpha, size_alpha:] = mixed
        hessian[size_alpha:, :size_alpha] = mixed.T
    return hessian


def analyse_stability(
    hessian: np.ndarray, tolerance: float = EIGENVALUE_TOLERANCE
) -> StabilityResult:
    """Hessian ozdegerlerinden kararlilik hukmu.

    KASTEN yukseltmez: kararsiz bir cozum gecerli bir duragan noktadir ve
    kullanici onu mesru olarak isteyebilir (ornegin RHF'nin ayrisma
    davranisini incelemek icin). Duzeltilen kusur SESSIZLIKTIR, sonucun
    kendisi degil.
    """
    if hessian.shape[0] == 0:
        return StabilityResult(
            stable=True,
            lowest_eigenvalue=None,
            eigenvalues=np.zeros(0),
            n_rotations=0,
        )
    eigenvalues = np.linalg.eigvalsh(hessian)
    lowest = float(eigenvalues[0])
    return StabilityResult(
        stable=lowest > -tolerance,
        lowest_eigenvalue=lowest,
        eigenvalues=eigenvalues,
        n_rotations=hessian.shape[0],
    )
