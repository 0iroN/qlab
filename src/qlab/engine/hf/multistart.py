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
tekrar cagirir ve en dusugunu kaynagiyla birlikte dondurur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.hf.basis import build_basis
from qlab.engine.hf.scf import HFResult, solve
from qlab.engine.hf.uhf import UHFResult, solve_uhf
from qlab.errors import InvalidInputError, QLabError, SCFConvergenceError
from qlab.types import Molecule

# Iki cozumun "ayri" sayilmasi icin gereken enerji farki (hartree).
# Olculdu: ayni cozume farkli baslangictan inen kosularda sapma 1e-12
# mertebesinde kaliyor; ayri cozumler arasindaki en kucuk fark 0.27
# hartree'nin ustunde. Esik ikisinin arasindadir ve YALNIZCA KUME SAYIMI
# icindir -- hicbir fizik hukmu buna dayanmaz.
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
    """

    energy: float
    result: UHFResult
    n_seeds: int
    n_converged: int
    n_distinct: int
    spread: float

    def as_line(self) -> str:
        return (
            f"{self.n_seeds} tohum, {self.n_converged} yakinsadi, "
            f"{self.n_distinct} ayri cozum, yayilim {self.spread:.3e} hartree"
        )


def search_uhf(
    molecule: Molecule,
    basis_name: str = "sto-3g",
    seeds: int = _VARSAYILAN_TOHUM,
    n_alpha: int | None = None,
    max_iterations: int = 2000,
) -> UHFSearchReport:
    """UHF'i `seeds` farkli baslangic sapmasiyla kosar, EN DUSUGUNU dondurur."""
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

    return UHFSearchReport(
        energy=float(en_iyi.energy_total),
        result=en_iyi,
        n_seeds=seeds,
        n_converged=len(enerjiler),
        n_distinct=_ayri_sayisi(enerjiler),
        spread=float(max(enerjiler) - min(enerjiler)),
    )

