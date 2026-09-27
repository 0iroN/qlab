"""Spin taramasi cizim YUKU -- goruntu uretmez.

Deponun `viz/` katmani hep yuk uretir (`frames.py`, `md_frames.py`,
`density_frames.py`, `band_frames.py`); `.png` yazmak yeni bagimlilik ve yeni
katman demektir. Ayni uyari `band_frames.py`de yazili: bu bir DISIPLINDIR,
SOZLESME DEGIL -- `viz` katmaninin import-linter tarafindan korunan bir GUI
yasagi YOKTUR.

BU MODUL NEDEN VAR -- KATMAN ZORUNLULUGU. Gozlemlenebilirler
`observable/spin_obs.py`dedir ve `engine/spin/factory.py` onlari IMPORT
EDEMEZ (sozlesme: run -> viz -> observable -> engine -> scenario). Yani
tarama sonucundan Binder/khi/C ve hata cubugunu cikaran yer buradan baskasi
olamaz. Fabrika fizigi kosar, bu modul olcer, fabrika olculeni beyana cevirir.

ISI SIGASI "OLCULMEDI" DIYEBILIR VE BU SIFIR DEGILDIR. Wolff kume cevirdigi
icin enerjiyi adim basina ucuza guncelleyemez ve `energy_series = None`
dondurur; `heat_capacity` alani o zaman `None` KALIR ve `.npz`ye HIC
YAZILMAZ. Sifir yazmak "isi sigasi sifir cikti" diye okunurdu -- Modul 1B'nin
kararlilik alani dersinin aynisi, ve `observable/spin_obs.py::_seri` ayni
gerekceyle `None` gorunce hata firlatir.

KORELASYON UZUNLUGU BU YUKTE YOKTUR ve gerekcesi olculmustur: ornekleyiciler
yalnizca SON konfigurasyonu dondurur, ikinci moment tahmincisi ise tek
konfigurasyonda T_c'de 41 kat yayiliyor (3.644 ... 150.562). Tek bir sayi
raporlamak, o sayinin hangi tohuma dustugunu raporlamak olurdu. Gerekce
panelde de beyan edilir (`engine/spin/factory.py::_SPIN_IHMAL`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.spin.factory import SpinScanResult
from qlab.errors import InvalidInputError
from qlab.observable.spin_obs import (
    binder_cumulant,
    binning_error,
    heat_capacity,
    susceptibility,
)


@dataclass(frozen=True, eq=False)
class SpinPayload:
    """Sicaklik taramasi cizimi icin gereken her sey. SI birimi YOK.

    `temperatures` `coupling` ile ayni ENERJI birimindedir (k_B = 1; kT/J =
    T/coupling); kelvin YOKTUR (bkz. birim secimi beyani). `susceptibility`
    enerji^-1 birimindedir (bkz. birim beyani). Butun diziler ayni
    uzunluktadir ve senaryonun sicaklik listesiyle AYNI SIRADADIR.

    `magnetisation_error` blok-ortalamasi hatasidir. `plateau_reached` False
    olan indislerde bu sayi hatanin KENDISI DEGIL bir ALT SINIRIDIR ve
    cagiran bunu bilmek zorundadir -- iki dizi bu yuzden BIRLIKTE tasinir,
    tek bir "hata" dizisine indirgenmez.

    `heat_capacity` `None` ise "OLCULMEDI" demektir, "sifir" degil.

    `eq=False`: `frames.py:23`teki `FramePayload` `eq=True` tasiyor ve
    `p1 == p2` cagrisi numpy alanlari yuzunden ValueError atiyor (olculdu,
    `band_frames.py` dokstringi).
    """

    temperatures: np.ndarray
    magnetisation: np.ndarray
    magnetisation_error: np.ndarray
    plateau_reached: np.ndarray
    tau_int: np.ndarray
    binder: np.ndarray
    susceptibility: np.ndarray
    heat_capacity: np.ndarray | None


def build_spin_payload(result: SpinScanResult) -> SpinPayload:
    """Tarama sonucundan gozlemlenebilirleri ve HATA CUBUGUNU cikarir.

    Hata cubugu her sicaklikta `binning_error` ile hesaplanir ve plato
    bayragi ATILMAZ. Yalniz `error` dondurmek, plato yapmamis bir egrinin
    alt sinirini hatanin kendisi gibi gostermek olurdu.

    ISI SIGASI ALGORITMA BAZINDA ACILIR/KAPANIR, sicaklik bazinda degil:
    `energy_series` bir kosum ozelligidir (Wolff'ta hepsi None, Metropolis'te
    hicbiri). Karisik bir durum olusursa -- yani bazi sicakliklarda enerji
    var bazilarinda yoksa -- burada HATA firlatilir, cunku o durumda dizinin
    yarisi olculmus yarisi olculmemis olurdu ve tek bir dizi bunu anlatamaz.
    """
    scenario = result.scenario
    if len(result.samples) != len(scenario.temperatures):
        raise InvalidInputError(
            f"ornek sayisi sicaklik sayisiyla uyusmuyor: "
            f"{len(result.samples)} != {len(scenario.temperatures)}"
        )

    L = scenario.lattice_size
    m: list[float] = []
    hata: list[float] = []
    plato: list[bool] = []
    tau: list[float] = []
    binder: list[float] = []
    khi: list[float] = []
    enerjili = [s.energy_series is not None for s in result.samples]
    if any(enerjili) and not all(enerjili):
        raise InvalidInputError(
            "sicakliklarin bir kismi enerji serisi tasiyor, bir kismi tasimiyor; "
            "isi sigasi dizisinin yarisi 'olculmedi' olamaz. `energy_series` "
            "bir KOSUM ozelligidir (Wolff olcmez, Metropolis olcer)."
        )
    C: list[float] | None = [] if all(enerjili) else None

    for sicaklik, ornek in zip(scenario.temperatures, result.samples):
        seri = ornek.magnetisation_series
        b = binning_error(seri)
        m.append(float(seri.mean()))
        hata.append(b.error)
        plato.append(b.plateau_reached)
        tau.append(b.tau_int)
        binder.append(binder_cumulant(seri))
        khi.append(susceptibility(seri, L, sicaklik))
        if C is not None:
            C.append(heat_capacity(ornek.energy_series, L, sicaklik))

    return SpinPayload(
        temperatures=np.asarray(scenario.temperatures, dtype=float),
        magnetisation=np.asarray(m, dtype=float),
        magnetisation_error=np.asarray(hata, dtype=float),
        plateau_reached=np.asarray(plato, dtype=bool),
        tau_int=np.asarray(tau, dtype=float),
        binder=np.asarray(binder, dtype=float),
        susceptibility=np.asarray(khi, dtype=float),
        heat_capacity=None if C is None else np.asarray(C, dtype=float),
    )
