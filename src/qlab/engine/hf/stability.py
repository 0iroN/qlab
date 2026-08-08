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
Sinanan: gercek (reel) isgal-bos dondurmeler, tekil (singlet) uzayda -- yani
RHF -> RHF ic kararliligi. Negatif ozdeger = yakinda daha dusuk enerjili bir
RHF cozumu var = raporlanan cozum eyer noktasi.

Sinanmayan: DIS kararsizliklar -- RHF -> UHF (spin kirilmasi), uclu (triplet)
uyarim, kompleks orbitaller. "stable=True", cozumun kuresel minimum oldugunu
veya UHF'in daha dusuk enerji vermeyecegini SOYLEMEZ. Bu sinir provenance'in
ihmal listesinde de yazilidir (scf.py, _HF_NEGLECTED).

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

    def verdict(self) -> StabilityVerdict:
        """Provenance'a ilistirilse makine-okunur (JSON'a yazilabilir) ozet."""
        return StabilityVerdict(
            stable=self.stable,
            lowest_eigenvalue=self.lowest_eigenvalue,
            n_rotations=self.n_rotations,
        )


def _transform_eri(eri: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """AO -> MO iki-elektron integral donusumu, dort adimda (N^5).

    Tek bir N^8 einsum yerine dort ardisik N^5 kismi kasilma. Hessian'in
    ihtiyaci olan uc blok (ov|ov, vv|oo, ov|vo) butun MO indekslerini
    kullandigi icin tam tensor uretmek, uc ayri kismi donusum yazmaktan hem
    kisa hem karsilastirilabilir maliyettedir. Bellek N^4'tur -- solve()
    zaten AO ERI tensorunu ayni buyuklukte tutuyor, yeni bir asimptotik yuk
    getirmez.
    """
    step = np.einsum("uvls,up->pvls", eri, coefficients, optimize=True)
    step = np.einsum("pvls,vq->pqls", step, coefficients, optimize=True)
    step = np.einsum("pqls,lr->pqrs", step, coefficients, optimize=True)
    return np.einsum("pqrs,st->pqrt", step, coefficients, optimize=True)


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
