"""Kare kafes uzerinde spin durumu -- SPIN UZAYINDAN BAGIMSIZ.

Bu modul spinin ayrik mi surekli mi oldugunu BILMEZ. Bilmemesi bir stil
tercihi degil, Modul 4'un Mermin-Wagner cipasinin ON KOSULUDUR: Ising ile
Heisenberg AYNI kafes ve AYNI ornekleyiciden gecmezse "ayni kod, zit hukum"
iddiasi duser (spec §3.4, §5).

Spinler `(L, L, n_component)` seklindedir; Ising'de n_component = 1.
Ayri bir 2B dizi kullanmak, iki modelin kod yollarini ayirirdi.

BEYAN EDILEN YAKLASIMLAR (spec §6):
- Klasik spin; kuantum dalgalanmasi YOK.
- Kare kafes, periyodik sinir.
- `k_B = 1` bir birim secimidir, fizik degil: sicaklik `J` ve `h` ile AYNI
  enerji birimindedir; `J` bir GIRDIDIR ve kT/J = T/J'dir.
- KENAR DURUMU L = 2: karsit komsular ayni siteye denk gelir
  (`np.roll(+1)` ile `np.roll(-1)` cakisir), yani her bag CIFT katsayiyla
  sayilir. Olculdu: L=2 tekbicim Ising'de bu modul E = -8.0 verir, oysa
  torustaki AYRIK bag sayisi 4'tur ve bag basina -J ile E = -4.0 olurdu.
  L >= 3'te boyle bir cakisma yoktur (L=3: modul -18.0, ayrik bag sayisi 18).
  Alt sinir yine de 2'dir cunku L=1 sitenin KENDI komsusu olmasi demektir;
  L=2 fizik olarak anlamli bir kafes degil, sadece bag katsayisi beyan
  edilmis bir kenar durumudur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.errors import InvalidInputError


@dataclass(frozen=True, eq=False)
class SpinState:
    """`(L, L, n_component)` seklinde spin dizisi.

    `eq=False`: alan bir ndarray oldugu icin uretilen `__eq__` bir dizi
    dondurur ve uretilen `__hash__` `TypeError` verir. Depo bu tuzagi
    `engine/hf/integrals.py` ve `engine/md/state.py`de zaten odedi; ayni
    gerekce.
    """

    spins: np.ndarray

    def __post_init__(self) -> None:
        # ONCE ndarray'e cevir: liste girdisi dogrulamayi ham `AttributeError`
        # ile dusururdu (`.ndim` yok). Ayni desen `engine/md/state.py`de.
        object.__setattr__(self, "spins", np.asarray(self.spins, dtype=float))
        if self.spins.ndim != 3 or self.spins.shape[0] != self.spins.shape[1]:
            raise InvalidInputError(
                f"spins (L, L, n_component) olmali, verilen: {self.spins.shape}"
            )

    @property
    def L(self) -> int:
        return int(self.spins.shape[0])

    @property
    def n_component(self) -> int:
        return int(self.spins.shape[2])

    @classmethod
    def uniform(cls, L: int, n_component: int) -> SpinState:
        """Hepsi ayni yone bakan durum -- son bilesen 1, digerleri 0.

        SON bilesen bilincli: alan terimi de son bilesene (s^z) baglanir, yani
        tekbicim durum her zaman alanla HIZALIDIR. Ilk bilesen secilseydi
        n_component = 1'de fark gorunmez, n_component = 3'te alan terimi
        sessizce sifirlanirdi.
        """
        if L < 2:
            raise InvalidInputError(f"L en az 2 olmali, verilen: {L}")
        if n_component < 1:
            raise InvalidInputError(
                f"n_component en az 1 olmali, verilen: {n_component}"
            )
        spins = np.zeros((L, L, n_component))
        spins[..., -1] = 1.0
        return cls(spins)


def neighbour_sum(spins: np.ndarray) -> np.ndarray:
    """Dort komsunun toplami, periyodik sinir.

    Sonuc girdiyle AYNI sekildedir: her site icin komsularinin bilesen bazinda
    toplami. Model bilgisi tasimaz -- Ising'de de Heisenberg'de de ayni satir
    kosar.
    """
    return (
        np.roll(spins, 1, axis=0)
        + np.roll(spins, -1, axis=0)
        + np.roll(spins, 1, axis=1)
        + np.roll(spins, -1, axis=1)
    )


def total_energy(state: SpinState, coupling: float, field: float) -> float:
    """E = -J sum_<ij> s_i . s_j - h sum_i s_i^z

    Komsu toplami her bagi IKI kez sayar; 0.5 carpani onu duzeltir. Olculdu:
    L=6 tekbicim Ising'de (J=1, h=0) dogru deger -72.0, carpan dusurulurse
    -144.0 -- iki kat.

    Alan SON bilesene (z) etki eder. n_component = 1'de son bilesen ile ilk
    bilesen ayni seydir, dolayisiyla bu secim yalnizca Heisenberg'de
    gozlemlenebilir; `tests/spin/test_lattice.py` onu orada kilitler.
    """
    etkilesim = -0.5 * coupling * float(np.sum(state.spins * neighbour_sum(state.spins)))
    alan = -field * float(np.sum(state.spins[..., -1]))
    return etkilesim + alan


def magnetisation(state: SpinState) -> np.ndarray:
    """Bilesen bazinda ORTALAMA spin, sekil (n_component,).

    Toplam degil ortalama: boyutsuz ve L'den bagimsiz olmasi, Mermin-Wagner
    cipasinin `d ln m / d ln L` egimini anlamli kilar.

    Isaretler korunur (mutlak deger ALINMAZ). `⟨|m|⟩` sonlu kafeste ayri bir
    gozlemlenebilirdir ve `observable/` tarafinda hesaplanir; burada karistir-
    mak, simetri kirilmasini olcmeyi imkansiz kilardi.
    """
    return state.spins.reshape(-1, state.n_component).mean(axis=0)
