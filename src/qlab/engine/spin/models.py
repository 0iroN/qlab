"""Iki spin modeli, TEK protokol.

Modeller yalnizca ONERI uretir; kabul/red karari `metropolis.py`nindir ve o
modul hangi modeli kosturdugunu BILMEZ. Bu ayrim Modul 4'un cipasinin on
kosuludur (spec §3.4): Ising ile Heisenberg arasindaki TEK fark spin uzayi
olmalidir, kod yolu degil.

Iki model de ayni Hamiltoniyen bicimini paylasir:

    H = -J sum_<ij> s_i . s_j - h sum_i s_i^z

    Ising      : s_i in {-1, +1}         (AYRIK simetri)
    Heisenberg : s_i in S^2 (birim kure)  (SUREKLI simetri)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from qlab.errors import InvalidInputError


@runtime_checkable
class SpinModel(Protocol):
    """Bir spin modeli: kac bilesen tasidigi ve nasil oneri urettigi.

    Protokolun DAR olmasi kasitlidir. Ornekleyicinin gorebilecegi tek sey
    `n_component` ve `propose`dur; sicaklik, eslesme sabiti ve alan modele
    HIC gecmez, cunku gecerse model kabul kararina karisabilirdi ve iki modeli
    ayni cekirdekten gecirme iddiasi anlamini yitirirdi.
    """

    n_component: int

    def propose(self, spins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """HER site icin bir aday spin dizisi dondurur (kabul edilmeden).

        Girdiyi DEGISTIRMEZ; yeni bir dizi dondurur. Yerinde degisiklik,
        reddedilen bir onerinin geri alinmasini imkansiz kilardi.
        """


@dataclass(frozen=True)
class IsingModel:
    """AYRIK simetri: tek oneri isaret cevirmedir.

    Ising'de her sitenin iki durumu vardir, dolayisiyla "rastgele bir aday"
    ile "diger durum" ayni seydir ve `rng` kullanilmaz. Imzada yine de yer
    alir: protokol iki model icin de AYNI olmak zorunda.
    """

    n_component: int = 1

    def propose(self, spins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return -spins


@dataclass(frozen=True)
class HeisenbergModel:
    """SUREKLI simetri: birim kurede kucuk rastgele donme.

    `step` kabul oranini ayarlar ve FIZIK DEGILDIR -- ayni dengeye farkli
    hizda gidilir.

    OLCULEN ortalama nokta carpimi (L=16 kutup kafesi, tohum 20260821, tek
    oneri turu; her satir ayni tohumu kullanir, yani tek degisken `step`):

        step=0.05 -> 0.997451
        step=0.60 -> 0.692146   (varsayilan)
        step=2.00 -> 0.259781

    Kucuk adim aciliminda nokta carpimi ~ 1 - step^2'dir ve bu bagimsiz
    kontrol tutuyor: step=0.05 icin 1 - step^2 = 0.997500, L=1024'te olculen
    0.997502.

    OLCULMEDI -- BEYAN: varsayilan 0.6'nin "T ~ 1'de kabul orani makul" olup
    olmadigi bu gorevde SINANMADI, cunku kabul/red karari henuz yok
    (`metropolis.py`, Task 2). Buradaki tek hukum, `step` buyudukce onerinin
    baslangictan uzaklasmasidir.
    """

    step: float = 0.6
    n_component: int = 3

    def __post_init__(self) -> None:
        if self.step <= 0.0:
            raise InvalidInputError(f"step pozitif olmali, verilen: {self.step}")

    def propose(self, spins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        aday = spins + self.step * rng.normal(size=spins.shape)
        norm = np.linalg.norm(aday, axis=2, keepdims=True)
        return aday / norm
