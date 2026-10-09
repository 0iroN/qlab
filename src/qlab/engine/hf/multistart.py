"""Cok-baslangicli SCF arama protokolu.

NEDEN VAR: olculdu, tek baslangicli `solve()` bazi geometrilerde TABAN DURUM
OLMAYAN bir cozume yakinsiyor ve `converged=True` diyor. Sebep TEK DEGIL,
IKI -- ve ikisi de ayri ayri olculdu:

  (1) DIIS. N2 R=1.10 A'da DIIS'li kosum -106.769673858, DIIS'siz kosum
      -107.496500512 veriyor. Orada tohum degistirmeye gerek YOK; DIIS'i
      kapatmak yetiyor. Fark 0.726827 hartree = 19.78 eV.
  (2) BASLANGIC TAHMINI. R=1.30...2.00 A'da DIIS'li ve DIIS'siz kosumlar
      BIREBIR ayni sonucu veriyor (fark < 1e-9), ama farkli kosegen
      sapmalarla 2-5 AYRI cozum bulunuyor:
        R=1.30 -> 0.498217 Ha asagida (2 ayri cozum)
        R=1.50 -> 0.323357 Ha        (3)
        R=1.80 -> 0.409936 Ha        (4)
        R=2.00 -> 0.294682 Ha        (5)

ARAMA YALNIZ TOHUMU DEGISTIRIR, DIIS'i DEGIL -- ve bu bir tercih degil bir
OLCUM sonucudur. Ilk tasarim her tohumu iki DIIS ayariyla kosuyordu; olculdu,
DIIS ekseni HICBIR yerde bir sey degistirmiyor (bes N2 geometrisinde ve suda
en buyuk fark 5.68e-14) ama maliyeti IKIYE KATLIYOR (1.85 s -> 4.2 s).
Sebebi de olculdu: R=1.10'da bias verilince DIIS'li kosum da taban durumu
buluyor -- sekiz tohumun BESI. Yani DIIS yalnizca VARSAYILAN (sifir) bias'la
yanlis yone gidiyor, ve tohum degistirmek onu zaten kurtariyor.

DIIS bulgusunun kendisi yine de gerceKtir ve testte kilitlidir; oradaki hukum
"DIIS yanlis cozume goturuyor" degil, "VARSAYILAN baslangicla DIIS yanlis
cozume goturuyor"dur.

ASIL KURAL: bir SCF enerjisi, ARAMA PROTOKOLU olmadan bir sayi degildir.
`SearchReport` bu yuzden yalniz enerjiyi degil, kac tohumdan kac cozum
ciktigini ve yayilimi da tasir.

Bu modul FIZIK YAPMAZ: yalnizca `solve()`u farkli baslangiclarla tekrar
tekrar cagirir ve en dusugunu kaynagiyla birlikte dondurur. TEK ISTISNA UHF
kararlilik izlemesidir (B7, 09.10.2026): orada en iyinin UHF Hessian'ina
bakilir (`stability.uhf_orbital_hessian`), en negatif yonde bir cizgi aramasi
yapilir ve SCF yine kamu `solve_uhf` ile yeniden kosulur. Gerekce ve olcumler
asagidaki "Kararlilik izleme" blogunda; kapanis testleri
tests/hf/test_uhf_kararlilik.py.

B7 OLCUMLERI (kamu yolu; betikler `.superpowers/sdd/b7-olcum/uygulama/`):
  * HF@20 bohr STO-3G: izlemesiz 1/8/12 tohum -97.570681531688 (+0.8824 Ha),
    24/40/64 tohum -97.678222271884 (+0.7749 Ha) -- ikisi de UHF EYERI;
    izlemeli 1/8/12/24/40/64 tohumun HEPSI parca toplamina 0 ... +1.4e-14
    (<= 1 ulp; 1 inis; duzeltme olcumu f01).
  * Kok neden tohum OLCEGI: ailenin en buyuk |bias|'i 1.3008; H 1s'e tek
    bilesenli bias 7.33 < B* <= 7.34 (Hcore'da F 2p ile H 1s arasi 7.334 Ha).
  * Yan kazanc: HF R = 2.5 bohr'da izleme 1.302e-2 Ha, 6-31G R = 4 bohr'da
    0.1365 Ha indiriyor (tohum ailesi orada da eyerde kaliyordu).
  * Izlemenin dokunmadigi yerler BIT BIT ayni: atomlar (H, Li, N, F), 12
    tohumlu catallanma, H2 Yamaguchi J (seeds = 12) ve testlerdeki halka
    aramalari (kayit kosusu u11).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.integrals import core_hamiltonian, eri_tensor
from qlab.engine.hf.scf import HFResult, solve
from qlab.engine.hf.stability import analyse_stability, uhf_orbital_hessian
from qlab.engine.hf.uhf import UHFResult, solve_uhf
from qlab.errors import InvalidInputError, QLabError, SCFConvergenceError
from qlab.provenance import StabilityVerdict
from qlab.types import Molecule

# Iki cozumun "ayri" sayilmasi icin gereken enerji farki (hartree).
# Olculdu: ayni cozume farkli baslangictan inen kosularda sapma 1e-12
# mertebesinde kaliyor; ayri cozumler arasindaki en kucuk fark 0.27
# hartree'nin ustunde. Esik ikisinin arasindadir ve YALNIZCA KUME SAYIMI
# icindir -- hicbir fizik hukmu buna dayanmaz.
# UHF ICIN "0.27 Ha" BAYAT (B7): HF R = 8 bohr STO-3G'de pi-delik eyeri ile
# sigma-delik minimumu arasi 1.054e-7 Ha (izlemeyle olculdu) -- bu esik ikisini
# TEK kume sayardi; Haar baslangiclarindan 17 SCF'in buldugu pi-delik kumesi de
# 1.04e-9 icinde. Sayi DEGISMEDI: esik hukum tasimaz, izlemenin buldugu cozum
# `n_distinct`e eklenmez ve donen cozumun ne oldugunu `stability` soyler.
_AYRIM_ESIGI = 1e-6

# Varsayilan tohum sayisi. SECILMEDI, OLCULDU: N2'nin bes geometrisinde
# 8 / 16 / 32 tohum AYNI en dusuk enerjiyi buluyor (R = 1.10, 1.30, 1.50,
# 1.80, 2.00 -- besinde de birebir). N2'de 8 kosu ~1.9 s.
_VARSAYILAN_TOHUM = 8


@dataclass(frozen=True, eq=False)
class SearchReport:
    """Bir arama kosusunun sonucu. `energy` TEK BASINA raporlanmaz."""

    energy: float
    result: HFResult
    n_seeds: int
    n_converged: int
    n_distinct: int
    spread: float

    def as_line(self) -> str:
        """Seffaflik panelinin basacagi tek satir."""
        return (
            f"{self.n_seeds} tohum, {self.n_converged} yakinsadi, "
            f"{self.n_distinct} ayri cozum, yayilim {self.spread:.3e} hartree"
        )


def seed_biases(n_basis: int, seeds: int) -> list[np.ndarray]:
    """Kosegen sapma desenleri -- SCF'i farkli havzalara itmek icin.

    Desenler DETERMINISTIK: sifir (yani mevcut davranis), alternan, blok,
    tek-site, ve sabit tohumlu rastgele. Rastgele olanin tohumu sabittir
    cunku bir arama protokolu tekrarlanabilir olmak zorundadir.
    """
    if seeds < 1:
        raise InvalidInputError(f"seeds en az 1 olmali, verilen: {seeds}")
    desenler: list[np.ndarray] = [np.zeros(n_basis)]
    for olcek in (0.1, 0.35, 1.0):
        desenler.append(olcek * np.array([(-1.0) ** i for i in range(n_basis)]))
        blok = np.ones(n_basis)
        blok[: n_basis // 2] = -1.0
        desenler.append(olcek * blok)
        for site in range(min(n_basis, 4)):
            tek = np.zeros(n_basis)
            tek[site] = olcek
            desenler.append(tek)
    rng = np.random.default_rng(20260820)
    while len(desenler) < seeds:
        desenler.append(rng.normal(0.0, 0.5, n_basis))
    return desenler[:seeds]


def _ayri_sayisi(enerjiler: list[float]) -> int:
    """Esikten uzak kumelerin sayisi."""
    if not enerjiler:
        return 0
    sirali = sorted(enerjiler)
    kume = 1
    for onceki, simdiki in zip(sirali, sirali[1:]):
        if simdiki - onceki > _AYRIM_ESIGI:
            kume += 1
    return kume


def search_rhf(
    molecule: Molecule,
    basis_name: str = "sto-3g",
    seeds: int = _VARSAYILAN_TOHUM,
    max_iterations: int = 2000,
) -> SearchReport:
    """RHF'i `seeds` farkli baslangic sapmasiyla kosar, EN DUSUGUNU dondurur.

    DIIS ayari DEGISTIRILMEZ; gerekcesi olculmustur (modul dokstringi).
    """
    n_basis = len(build_basis(molecule, basis_name))
    enerjiler: list[float] = []
    en_iyi: HFResult | None = None

    for bias in seed_biases(n_basis, seeds):
        try:
            sonuc = solve(
                molecule,
                basis_name,
                stability=False,
                max_iterations=max_iterations,
                initial_bias=bias,
            )
        except (SCFConvergenceError, QLabError):
            continue
        enerjiler.append(float(sonuc.energy_total))
        if en_iyi is None or sonuc.energy_total < en_iyi.energy_total:
            en_iyi = sonuc

    if en_iyi is None:
        raise SCFConvergenceError(
            iterations=max_iterations,
            energy_history=[],
            final_residual=float("nan"),
            oscillating=False,
        )

    return SearchReport(
        energy=float(en_iyi.energy_total),
        result=en_iyi,
        n_seeds=seeds,
        n_converged=len(enerjiler),
        n_distinct=_ayri_sayisi(enerjiler),
        spread=float(max(enerjiler) - min(enerjiler)),
    )


@dataclass(frozen=True, eq=False)
class UHFSearchReport:
    """`SearchReport`in UHF karsiligi. `energy` TEK BASINA raporlanmaz.

    UHF'te arama RHF'tekinden DAHA gereklidir: olculdu, baslangic sapmasi
    olmadan UHF her zaman RHF'e coker (H2 R = 2.00 A'da fark < 1e-12), yani
    "UHF kostum" demek tek basina hicbir sey ifade etmez.

    B7 ALANLARI (09.10.2026, hepsi varsayilanli ve sonda):
      stability     : DONEN cozumun UHF -> UHF hukmu (kanal "uhf") -- tohumun
                      en iyisinin DEGIL. Izleme kapali olsa da HESAPLANIR.
      seed_energy   : tohum ailesinin en iyisinin enerjisi (izlemeden once)
      follow_cycles : kararlilik izlemesinin KABUL ettigi inis sayisi
      follow_gain   : seed_energy - energy >= 0 (izlemenin indirdigi, Ha)
    `n_distinct` ve `spread` YALNIZ tohum sonuclarindan hesaplanir; izlemenin
    buldugu cozum onlara eklenmez (tanimlari B7 oncesiyle ayni).
    """

    energy: float
    result: UHFResult
    n_seeds: int
    n_converged: int
    n_distinct: int
    spread: float
    stability: StabilityVerdict | None = None
    seed_energy: float | None = None
    follow_cycles: int = 0
    follow_gain: float = 0.0

    def as_line(self) -> str:
        satir = (
            f"{self.n_seeds} tohum, {self.n_converged} yakinsadi, "
            f"{self.n_distinct} ayri cozum, yayilim {self.spread:.3e} hartree"
        )
        if self.follow_cycles > 0:
            satir += (
                f"; kararlilik izlemesi {self.follow_cycles} dongude "
                f"{self.follow_gain:.3e} Ha indirdi"
            )
        if self.stability is not None:
            lam = self.stability.lowest_eigenvalue
            lam_metin = "donme yok" if lam is None else f"lambda_min={lam:.3e}"
            if self.stability.stable:
                satir += (
                    f"; donen cozum UHF-kararli (reel alfa/beta donmeleri, "
                    f"{lam_metin})"
                )
            else:
                satir += f"; ✗ donen cozum UHF EYERI ({lam_metin})"
        return satir


# --- Kararlilik izleme (B7, 09.10.2026) ------------------------------------
#
# NEDEN VAR: olculdu, tohum ailesi HF molekulunde (H + F, 20 bohr, STO-3G)
# parca toplamini HIC bulmuyordu -- 8/12 tohum -97.570681531688, 24/40/64 tohum
# -97.678222271884, parca toplami -98.453086808293. Sebep tohum SAYISI degil
# OLCEGI: ailenin en buyuk |bias|'i 1.3008, H 1s'i doldurmak icin gereken
# tek bilesenli bias ise 7.30 < B* <= 7.35 (Hcore kosegeninde F 2p ile H 1s
# arasi 7.334 Ha). Ailenin buldugu iki durum da UHF EYERIdir (lambda_min
# -1.764811 ve -1.549729) ve depo bunu GORMUYORDU, cunku UHF -> UHF
# kararlilik analizi yoktu. Izleme en iyiyi en negatif Hessian yonunde
# itip SCF'i yeniden kosar: 1, 8, 12, 24, 40 ve 64 tohumun hepsi parca
# toplamina iner (tests/hf/test_uhf_kararlilik.py).
#
# YALNIZ EN IYIYE uygulanir ve bu bir olcum sonucudur (u06): 13 vakada (H8
# d = 1.8 ... 3.0, H6/H10/H14 halkalari, HF R = 2.5/4/8/20) tohum ailesinin 20
# ayri eyerinin HEPSINE izleme, en iyiye izlemeden daha dusuk bir sey bulmadi
# (en buyuk fark -2.84e-14). Kararli bir en iyide izleme hicbir sey yapmaz:
# donen nesne tohumun kendisidir, sayilar bit bit aynidir.
#
# PARCALARIN YUKU (u08/u09/u10, izlemenin ENSTRUMANLI bir kopyasiyla olculdu;
# 50 eyer, 54 kabul edilen inis, hepsi en cok 2 iniste kararliya indi). Ilk
# mutasyon turunda kabul denetimi, - yon, DIIS'siz geri donus ve ara aci
# (1.3 rad) HAYATTA KALDI, cunku testlerin vakalarinda ilk DIIS'li yeniden
# baslatma zaten iniyordu; bu dordunun her biri icin tek basina fark yaratan
# bir eyer bulundu ve testte kilitlendi. IKI PARCA SIGORTADIR, olculmus bir
# ihtiyac DEGIL: aday sayisi (`_IZLEME_ADAY`) ve kucuk acilar 0.05/0.1/0.2
# (`_IZLEME_ACILARI`) -- her birini ayri ayri kaldiran mutant testlerin hepsini
# geciyor (beyanli; testte kilitli DEGIL).

# Cizgi aramasi acilari (radyan), en negatif ozvektor boyunca +/- iki yonde.
# pi/2 bir isgal-bos ciftini TAM takas eder (HF@20'nin yuk aktarimi). Ara aci
# YUK TASIR: yalniz pi/2 ile kalinsa HF@20'nin S^2 = 1 eyeri (Haar baslangici)
# EYERDE kalir; kabul 1.3 rad'da (o eyerin en negatif ozdegeri iki katli;
# ozuzayda 13 yonun 5'inde yalniz-pi/2 basarisiz, tam izgara 13/13).
# KUCUK ACILAR (0.05 / 0.1 / 0.2) SIGORTADIR, olculmus bir ihtiyac DEGIL:
# catallanmanin sonsuz kucuk kararsizligi dusunulerek konuldular ama olculen
# hicbir vakada tek baslarina fark yaratmadilar. Kamu search_uhf ile 18 vakada
# (H2 Rc + 8e-5 ... 0.5, H8 d = 1.8 ... 3.0, H6 d = 2.6, HF R = 2.5 / 8 / 20,
# 6-31G R = 4) ve depodaki ozel `_follow_instability` ile testlerde kilitli 4
# zor eyerde (M16 / M17 / M18 vakalari, yumusak HF R = 8) onlarsiz izleme ayni
# inis sayisini, ayni hukmu ve ayni minimumu veriyor (en buyuk |dE| 1.84e-12;
# duzeltme olcumu f03).
# Iki yon SART: ozvektorun isareti keyfidir ve 50 eyerin 6'sinda (u08) yalniz
# + yon daha YUKSEK bir minimuma iniyordu (H8 d = 2.6 / 3.0).
_IZLEME_ACILARI: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.3, np.pi / 2)

# Cizgi aramasinin en dusuk enerjili kac donmesinden SCF yeniden baslatilir.
# Her biri once DIIS'li, inmezse DIIS'siz denenir; ilk kabul edilen kazanir.
# DIIS'siz geri donus YUK TASIR: H8 d = 2.6 bohr'un blok-bias eyerinde DIIS'li
# yeniden baslatma 2000 iterasyonda yakinsamiyor, DIIS'siz -3.9573119765'e
# iniyor. DURUST: olculen 54 kabulun HEPSI siradaki ILK adaydan geldi; 2.-4.
# adaylar bugune kadar hic gerekmedi (`_IZLEME_ADAY = 1` mutanti testleri
# geciyor). Dort, basarisiz bir dongunun bedelini sekiz SCF ile sinirlayan bir
# SIGORTADIR, olculmus bir ihtiyac degil.
_IZLEME_ADAY = 4

# Yeni cozumun kabulu: E_yeni < E_eski - _IZLEME_KABUL. Alt sinir: denetimin
# KENDISI yuk tasir -- H2 R = Rc + 0.1'de DIIS'li yeniden baslatma AYNI eyere
# geri cokuyor (dE = 0.0); denetimsiz izleme bunu bir "inis" sayar ve rapor 2
# dongu der (dogrusu 1). Yumusak yuzeyde SCF durma olcutunun enerji izi
# kuramsal olarak ~artik^2 / (2 |lambda|) = (1e-8)^2 / (4e-7) ~ 2.5e-10.
# Testlerdeki gercek inisler esigin ~100 kati ustunde: 1.054e-7 (HF R = 8
# bohr, pi-delik eyerinden sigma-delik minimumuna) ve 1.75e-7 (H2, R = Rc +
# 1e-3); 1e-6 esik ikisini de kacirirdi (testte kilitli).
# KOR BOLGE -- esigin altinda kalan GERCEK inisler var (olculdu, kamu yolu;
# beyanli, KAPANMADI): bir catallanmanin yaninda inis dE ~ d^2 ile keyfi
# kucuktur. H2, R = Rc + d, seeds = 1: d <= 6e-5'te (lambda_min -7.7e-7 ...
# -4.6e-5, yani yumusak mod bandinin 77 ... 4600 kati DISINDA) kirik simetrili
# minimum RHF eyerinin yalniz 1.7e-13 ... 6.3e-10 Ha altinda; izleme bu inisi
# REDDEDER ve eyer "✗" ile doner (ayni noktada seeds = 12'nin tohum ailesi
# minimumu buluyor); d = 8e-5'te inis 1.12e-9 ve kabul ediliyor. Engel esiktir,
# SCF degil: depodaki ozel yolda esik 1e-13'e indirilince d = 1e-6 / 1e-5 /
# 6e-5'te tek inisle minimuma iniliyor (duzeltme olcumu f02). Hukum durust,
# sessizlik yok -- ama izlemenin HER eyerden indigi iddia EDILMEZ.
_IZLEME_KABUL = 1e-9

# Kabul edilen inis sayisinin tavani. Olculen en cok 2 (H8 halkasi d = 2.0,
# seeds = 1; 50 eyerin hicbiri 2'yi asmadi); tavan sonsuz donguye karsi bir
# sigortadir, ayar degildir. Tavana ulasilirsa donen cozumun hukmu yine
# HESAPLANIR ve eyerse "✗" basilir.
_IZLEME_MAX_DONGU = 30


def _dondur(
    coefficients: np.ndarray, kappa: np.ndarray, n_occupied: int
) -> np.ndarray:
    """C exp(K), K[a, i] = kappa_ia, K[i, a] = -kappa_ia (`uhf_orbital_hessian`
    ile AYNI parametrelendirme ve indeks sirasi: i * n_v + a)."""
    n_orbital = coefficients.shape[1]
    n_virtual = n_orbital - n_occupied
    if n_occupied == 0 or n_virtual == 0:
        return coefficients
    blok = kappa.reshape(n_occupied, n_virtual)
    K = np.zeros((n_orbital, n_orbital))
    K[n_occupied:, :n_occupied] = blok.T
    K[:n_occupied, n_occupied:] = -blok
    return coefficients @ expm(K)


def _uhf_enerjisi(
    core: np.ndarray,
    eri: np.ndarray,
    nuclear: float,
    coefficients_alpha: np.ndarray,
    coefficients_beta: np.ndarray,
    n_alpha: int,
    n_beta: int,
) -> float:
    """Verilen orbitallerin UHF enerjisi (SCF YOK) -- cizgi aramasi icin."""
    p_alpha = coefficients_alpha[:, :n_alpha] @ coefficients_alpha[:, :n_alpha].T
    p_beta = coefficients_beta[:, :n_beta] @ coefficients_beta[:, :n_beta].T
    coulomb = np.einsum("uvls,ls->uv", eri, p_alpha + p_beta)
    fock_alpha = core + coulomb - np.einsum("ulvs,ls->uv", eri, p_alpha)
    fock_beta = core + coulomb - np.einsum("ulvs,ls->uv", eri, p_beta)
    return 0.5 * float(
        np.sum((p_alpha + p_beta) * core)
        + np.sum(p_alpha * fock_alpha)
        + np.sum(p_beta * fock_beta)
    ) + nuclear


def _follow_instability(
    molecule: Molecule,
    basis_name: str,
    start: UHFResult,
    max_iterations: int,
) -> tuple[UHFResult, StabilityVerdict, int]:
    """Kararlilik izleme: (donen cozum, ONUN hukmu, kabul edilen inis sayisi).

    Dongu: UHF Hessian'i -> lambda_min > -EIGENVALUE_TOLERANCE ise DUR.
    Degilse en negatif ozvektor v boyunca +/- `_IZLEME_ACILARI` donmeleri,
    enerjilerine gore siralanir; en dusuk `_IZLEME_ADAY` tanesinden kamu
    `solve_uhf` yeniden baslatilir (once DIIS'li, sonra DIIS'siz) ve
    E_yeni < E_eski - `_IZLEME_KABUL` olan ilk sonuc kabul edilir. Hicbiri
    inmezse eyer OLDUGU GIBI doner -- hukmu "✗" olarak.
    """
    basis = build_basis(molecule, basis_name)
    core = core_hamiltonian(basis, molecule)
    eri = eri_tensor(basis)
    nuclear = molecule.nuclear_repulsion()
    n_alpha, n_beta = start.n_alpha, start.n_beta

    sonuc = start
    inis = 0
    while True:
        hessian = uhf_orbital_hessian(
            core, eri, sonuc.coefficients_alpha, sonuc.coefficients_beta,
            n_alpha, n_beta,
        )
        karar = analyse_stability(hessian)
        if karar.stable or inis >= _IZLEME_MAX_DONGU:
            return sonuc, karar.verdict(channel="uhf"), inis

        yon = np.linalg.eigh(hessian)[1][:, 0]
        n_rot_alpha = n_alpha * (sonuc.coefficients_alpha.shape[1] - n_alpha)
        adaylar = []
        for isaret in (1.0, -1.0):
            for aci in _IZLEME_ACILARI:
                kappa = isaret * aci * yon
                c_alpha = _dondur(sonuc.coefficients_alpha, kappa[:n_rot_alpha], n_alpha)
                c_beta = _dondur(sonuc.coefficients_beta, kappa[n_rot_alpha:], n_beta)
                enerji = _uhf_enerjisi(core, eri, nuclear, c_alpha, c_beta, n_alpha, n_beta)
                adaylar.append((enerji, c_alpha, c_beta))
        adaylar.sort(key=lambda aday: aday[0])

        yeni: UHFResult | None = None
        for _, c_alpha, c_beta in adaylar[:_IZLEME_ADAY]:
            for diis in (True, False):
                try:
                    deneme = solve_uhf(
                        molecule,
                        basis_name,
                        n_alpha=n_alpha,
                        initial_orbitals=(c_alpha, c_beta),
                        use_diis=diis,
                        max_iterations=max_iterations,
                    )
                except (SCFConvergenceError, QLabError):
                    continue
                if deneme.energy_total < sonuc.energy_total - _IZLEME_KABUL:
                    yeni = deneme
                    break
            if yeni is not None:
                break
        if yeni is None:
            return sonuc, karar.verdict(channel="uhf"), inis
        sonuc = yeni
        inis += 1


def search_uhf(
    molecule: Molecule,
    basis_name: str = "sto-3g",
    seeds: int = _VARSAYILAN_TOHUM,
    n_alpha: int | None = None,
    max_iterations: int = 2000,
    *,
    follow_instability: bool = True,
) -> UHFSearchReport:
    """UHF'i `seeds` farkli baslangic sapmasiyla kosar, en dusugunu izler.

    Tohum ailesi B7 oncesiyle AYNIDIR. `follow_instability=True` (varsayilan)
    tohumlarin en iyisine kararlilik izleme uygular. Donen cozumun ne oldugunu
    YALNIZ `stability` soyler: `stable` ise reel alfa/beta donmelerine karsi
    yerel minimumdur (KURESEL oldugu iddia EDILMEZ -- H8 halkasi d = 3.0
    bohr'da baska yerel minimumlar var; |lambda| < 1e-8 yumusak modlar ayirt
    edilmez); izleme inemezse (ornegin `_IZLEME_KABUL`un kor bolgesi) eyer
    oldugu gibi doner ve hukmu "✗" basilir. `follow_instability=False`
    B7 oncesinin sayilarini bit bit verir AMA hukmu yine hesaplar: donen
    cozum eyerse `stability.stable` False'tur ve `as_line()` "✗" basar --
    sayi degismez, sessizlik biter.
    """
    n_basis = len(build_basis(molecule, basis_name))
    enerjiler: list[float] = []
    en_iyi: UHFResult | None = None

    for bias in seed_biases(n_basis, seeds):
        try:
            sonuc = solve_uhf(
                molecule,
                basis_name,
                n_alpha=n_alpha,
                initial_bias=bias,
                max_iterations=max_iterations,
            )
        except (SCFConvergenceError, QLabError):
            continue
        enerjiler.append(float(sonuc.energy_total))
        if en_iyi is None or sonuc.energy_total < en_iyi.energy_total:
            en_iyi = sonuc

    if en_iyi is None:
        raise SCFConvergenceError(
            iterations=max_iterations,
            energy_history=[],
            final_residual=float("nan"),
            oscillating=False,
        )

    tohum_enerjisi = float(en_iyi.energy_total)
    if follow_instability:
        donen, hukum, dongu = _follow_instability(
            molecule, basis_name, en_iyi, max_iterations
        )
    else:
        basis = build_basis(molecule, basis_name)
        hukum = analyse_stability(
            uhf_orbital_hessian(
                core_hamiltonian(basis, molecule), eri_tensor(basis),
                en_iyi.coefficients_alpha, en_iyi.coefficients_beta,
                en_iyi.n_alpha, en_iyi.n_beta,
            )
        ).verdict(channel="uhf")
        donen, dongu = en_iyi, 0
    # Hukum DONEN cozume aittir; sonuca ve onun provenance'ina da yazilir.
    donen = replace(
        donen, stability=hukum, provenance=replace(donen.provenance, stability=hukum)
    )

    return UHFSearchReport(
        energy=float(donen.energy_total),
        result=donen,
        n_seeds=seeds,
        n_converged=len(enerjiler),
        n_distinct=_ayri_sayisi(enerjiler),
        spread=float(max(enerjiler) - min(enerjiler)),
        stability=hukum,
        seed_energy=tohum_enerjisi,
        follow_cycles=dongu,
        follow_gain=tohum_enerjisi - float(donen.energy_total),
    )

