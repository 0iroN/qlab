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

    KABUL ORANI OLCULDU (Task 2; Task 1'de olculemezdi cunku kabul/red karari
    yoktu) ve hukum NITELIKLI cikti. L=16, T=1.0, J=1, h=0, tohum 20260822,
    n_burn=500, n_sweep=1500:

        step=0.05 -> kabul 0.965898
        step=0.60 -> kabul 0.676958   (varsayilan)
        step=2.00 -> kabul 0.499771
        step=100  -> kabul 0.417086   (DOYUM)

    Kabul orani `step` ile tekduze duser AMA BIR TABANA DOYAR: `step`
    buyudukce oneri, kurede duzgun dagilmis rastgele bir yone yakinsar ve
    T=1.0'da 0.42'nin altina INEMEZ. Yani bu oneri ailesiyle erisilebilir
    aralik [0.42, 1.00)'dir ve Metropolis folklorundaki "%30-50 hedefle"
    kurali burada ancak ucundan (step ~ 2, yani oneri "kucuk donme" olmaktan
    ciktigi yerde) tutturulabilir.

    DURUST HUKUM: 0.6 icin olculen 0.677 klasik %30-50 bandinin USTUNDEDIR;
    "makul" yalnizca "ne ~0 ne ~1, erisilebilir araligin icinde" anlaminda
    dogrudur. Sicakliga bagliligi da olculdu (4 tohum, step=0.6):
    T=0.8 -> 0.6008, T=1.0 -> 0.6770, T=1.4 -> 0.7700. Olcum
    `tests/spin/test_metropolis.py::test_KABUL_ORANI_ADIMLA_DUSUYOR_ve_BIR_TABANA_DOYUYOR`
    dosyasinda kilitlidir.
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
