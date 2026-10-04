"""Bloch Hamiltoniyeni H(k) ve sonlu halka matrisi -- HARTREE biriminde.

AYAR (GAUGE) SECIMI. Iki konvansiyon var: "hucre" gauge'unda faz yalniz orgu
otelemesini tasir, exp(i k . R_hucre); "atom" gauge'unda taban konumlari da
girer, exp(i k . (R_hucre + tau_j - tau_i)). Varsayilan HUCREDIR, cunku o
konvansiyonda H(k) ters orgu vektoru altinda birebir periyodiktir ve
periyodiklik testi H uzerinden yazilabilir.

OLCULDU ve BEYAN EDILIYOR: iki gauge H'nin ELEMANLARINI 5.20 eV'ye varan
farkla degistirir ama OZDEGERLERI 8 ULP'ta ayni birakir. Sonucu: iki
uygulamayi H uzerinden karsilastirmak YANILTICIDIR; karsilastirma ozdeger
uzerinden yapilir.

`eigh` TUZAGI. `np.linalg.eigh` varsayilan UPLO='L' ile yalniz ALT ucgeni
okur; olculdu, ust ucgene +1000 eklemek ozdegerleri TAM 0.000e+00
degistiriyor. Bu modul H'nin TAMAMINI kurar ve testler H'yi dogrudan
denetler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.tb.lattice import Lattice, max_cell_for_cutoff, neighbour_shells
from qlab.engine.tb.radyal import UsselRadyalYasa
from qlab.errors import InvalidInputError

# H0 (04.10.2026): hoplama listesinin Hermitlik toleransi, listedeki en buyuk
# |toplam t|'ye GORE. Kapi olmadan tek yonlu yazilmis bir zincir rc=0 ile
# "metal ✓" basiyordu ve bant genisligi YARIYA iniyordu (3.4 eV, dogrusu
# 6.8): `eigvalsh` yalniz alt ucgeni okur, eksik es sessizce yok sayilir.
# OLCULDU: testlerin kurdugu 1904 modelde en buyuk sapma 7.22e-16 (yuvarlama;
# distance_model'in iki yonu ayri hesaplamasindan), es-eksik 0 -- yani esik
# 1000 kattan fazla pay birakir. Eleman-basina bagil olcut SECILMEDI: kucuk
# bir elemanin bagil yuvarlamasi (6e-14) ozdegeri etkilemez, olcege gore
# sapma etkiler.
_HERMIT_TOLERANSI = 1e-12


@dataclass(frozen=True, eq=False)
class Hopping:
    """`i` orbitalinden `cell` hucresindeki `j` orbitaline hoplama, HARTREE.

    Isaret konvansiyonu: H'nin kosegen disi elemani DOGRUDAN `t`'dir
    (H[i,j] += t * faz). Literaturdeki "-t" konvansiyonu KULLANILMAZ;
    donusum cagiranin isidir ve senaryo sinirinda yapilir.
    """

    i: int
    j: int
    cell: tuple[int, int, int]
    t: float


@dataclass(frozen=True, eq=False)
class TBModel:
    """Orgu + yerinde enerjiler + hoplama listesi. Hepsi HARTREE."""

    lattice: Lattice
    onsite: np.ndarray
    hoppings: tuple[Hopping, ...]

    def __post_init__(self) -> None:
        yerinde = np.asarray(self.onsite, dtype=float)
        object.__setattr__(self, "onsite", yerinde)
        if yerinde.ndim != 1 or yerinde.shape[0] != self.lattice.n_atom:
            raise InvalidInputError(
                f"onsite dizisi (n_atom,) olmali: {yerinde.shape} != "
                f"({self.lattice.n_atom},)"
            )
        if not np.all(np.isfinite(yerinde)):
            raise InvalidInputError("onsite enerjileri sonlu olmali")
        for h in self.hoppings:
            if not (0 <= h.i < self.lattice.n_atom) or not (
                0 <= h.j < self.lattice.n_atom
            ):
                raise InvalidInputError(
                    f"orbital indisi aralik disi: i={h.i}, j={h.j}, "
                    f"atom sayisi {self.lattice.n_atom}"
                )
            if not np.isfinite(h.t):
                raise InvalidInputError(f"hoplama sonlu olmali: t={h.t}")
            # Kesirli hucrede H ters orgude periyodik DEGILDIR; B1/B2'nin
            # turetmesi cokuyor (dogrulama turu: sinir 128 kat asildi).
            if not all(float(c).is_integer() for c in h.cell):
                raise InvalidInputError(
                    f"hucre bilesenleri tam sayi olmali: cell={list(h.cell)}"
                )
        self._hermit_kapisi()

    def _hermit_kapisi(self) -> None:
        """Her (i, j, R) toplami (j, i, -R) toplamina esit olmali (t gercel).

        Anahtar yalniz ilk `dimension` hucre bilesenidir -- `bloch_hamiltonian`
        otekilerini okumaz. Ayni (i, j, R) birden cok kez yazilabilir;
        karsilastirilan TOPLAMDIR. Esi yazilmamis bir anahtarin esi 0 sayilir:
        tek yonlu SIFIR hoplama H'yi degistirmez ve reddedilmez.
        """
        d = self.lattice.dimension
        toplam: dict[tuple, float] = {}
        for h in self.hoppings:
            anahtar = (h.i, h.j, tuple(h.cell[eksen] for eksen in range(d)))
            toplam[anahtar] = toplam.get(anahtar, 0.0) + float(h.t)
        olcek = max((abs(t) for t in toplam.values()), default=0.0)
        for (i, j, hucre), t in toplam.items():
            es = toplam.get((j, i, tuple(-c for c in hucre)))
            if abs(t - (0.0 if es is None else es)) > _HERMIT_TOLERANSI * olcek:
                raise InvalidInputError(
                    f"hoplama listesi Hermityen degil: (i={i}, j={j}, "
                    f"cell={list(hucre)}) toplami {t!r} hartree, esi (i={j}, j={i}, "
                    f"cell={[-c for c in hucre]}) "
                    f"{'YOK' if es is None else repr(es)}. Her hoplama iki "
                    "yonde de yazilmalidir; tek yonlu liste bant genisligini "
                    "sessizce yariya indirir."
                )

    @property
    def n_orbital(self) -> int:
        return int(self.onsite.shape[0])


def bloch_hamiltonian(
    model: TBModel, k: np.ndarray, gauge: str = "cell"
) -> np.ndarray:
    """(n_orb, n_orb) kompleks Hermityen H(k). `k` 1/bohr, sonuc hartree."""
    if gauge not in ("cell", "atom"):
        raise InvalidInputError(f"gauge 'cell' ya da 'atom' olmali: {gauge!r}")
    k = np.asarray(k, dtype=float)
    if k.shape != (3,):
        raise InvalidInputError(f"k (3,) olmali, verilen: {k.shape}")

    n = model.n_orbital
    H = np.zeros((n, n), dtype=complex)
    np.fill_diagonal(H, model.onsite)

    vektorler = model.lattice.vectors
    d = model.lattice.dimension
    for h in model.hoppings:
        kayma = np.zeros(3)
        for eksen in range(d):
            kayma = kayma + h.cell[eksen] * vektorler[eksen]
        if gauge == "atom":
            kayma = kayma + model.lattice.basis[h.j] - model.lattice.basis[h.i]
        H[h.i, h.j] += h.t * np.exp(1j * float(k @ kayma))
    return H


def uniform_model(
    lattice: Lattice, onsite: np.ndarray, cutoff_bohr: float, t: float
) -> TBModel:
    """Kesme yaricapindan hoplama listesi kuran kisayol -- TEKBICIM `t` ile.

    `neighbour_shells`in IKI tuketicisinden biridir; digeri
    `distance_model`dir (`tests/tb/test_distance_model.py` bu sayimi `ast`
    ile kilitler). Kesme yaricapindaki kayan nokta tuzagi (11 fcc orgu
    sabitinin 5'inde 12 yerine 0 komsu) ancak gercekten kullanilan bir yolda
    korunur. Elle `Hopping` listesi yazmak da mesrudur ve testlerde oyle
    yapilir; bu kisayol yuksek koordinasyonlu kafesler (fcc'de 12 komsu)
    icin vardir.

    `t` TEKBICIMDIR: her komsu ciftine ayni deger konur. Mesafeye bagli
    hoplama bu fonksiyonun isi DEGILDIR; ss-sigma satiri `distance_model`dedir.
    Yone bagli Slater-Koster (p/d orbitalleri, yon kosinusleri) depoda YOKTUR.

    `max_cell` `max_cell_for_cutoff`tan turetilir (`distance_model` gibi).
    03.10.2026'ya kadar 2 SABITTI ve kesme iki hucre otesine uzaninca kabuk
    SESSIZCE kayboluyordu -- 1B'de 3|a|'da, ama 2B/3B'de DAHA ONCE ve simetri
    kirilarak (olculdu: fcc'de kesme 2.449|a|'da 86 yerine 80 hoplama ve
    t = -0.1'de C4z altinda bant farki 4|t| = 0.400 Ha; ucgen orguda C6).
    Duzeltme olculdu: kayipsiz 226 vakanin 226'sinda (dogrulayici ayrica
    bagimsiz bir orneklemle teyit etti) hoplama listesi, H(k) ve ozdegerler
    bit bit ayni; kayipli 240 vakanin
    240'inda liste `distance_model` ile ayni. Patolojik olcek
    `neighbour_shells`in hucre sayisi kapisina takilir.
    """
    return TBModel(
        lattice=lattice,
        onsite=onsite,
        hoppings=tuple(
            Hopping(i=c.i, j=c.j, cell=c.cell, t=t)
            for c in neighbour_shells(
                lattice, cutoff_bohr, max_cell=max_cell_for_cutoff(lattice, cutoff_bohr)
            )
        ),
    )


def distance_model(
    lattice: Lattice,
    onsite: np.ndarray,
    ss_sigma: UsselRadyalYasa,
    max_cell: int | None = None,
) -> TBModel:
    """Mesafeye bagli hoplama: her komsu cifti icin t = ss_sigma(d), HARTREE.

    Slater-Koster tablosunun YALNIZ ss-sigma satiri. Tek s orbitali/atom:
    aci carpani OZDES 1, bu bir SK modeli DEGILDIR. p/d yok (V_sp-sigma,
    V_pp-sigma, V_pp-pi kapsam disi). Isaret: H'nin kosegen disi elemani
    dogrudan t'dir (`Hopping`); '-t' konvansiyonu kullanilmaz.

    KESME YASADAN TURETILIR (`ss_sigma.r_cut_bohr`); ayri ve daha kucuk bir
    kesme verilemez, cunku o zaman sert kesmenin merdiveni geri gelirdi
    (spec md-tb B3: `uniform_model` dimerde 2|t| = 0.2000 sicrar, bu yasa
    kuyrukta C^1 ile sifira iner). r_cut'ta ya da hemen otesinde (bagil
    tolerans bandinda) kalan bir komsu listeye t = 0.0 ile girer.

    `NeighbourPair.distance` src'de ILK kez burada okunur. Bu yol ile MD yolu
    (`qlab.engine.tb.mermin.tek_orbital_mermin`) AYNI yasa nesnesini
    kullanabilir ve Gamma'daki supercell'de ayni Omega ve kuvveti verir
    (`tests/tb/test_distance_model.py` M6).

    `max_cell` verilmezse `max_cell_for_cutoff`tan (ters orgu) turetilir.
    Verilirse ondan KUCUK olamaz (`InvalidInputError`): daha kucuk bir
    max_cell kabugu SESSIZCE kaybedebilir (olculdu, `max_cell_for_cutoff`).
    Cakisik taban atomlari ya da goruntuleri `neighbour_shells`te
    `InvalidInputError`dir.

    `onsite` modele aynen gecer (M6'nin onsite kolu kilitler: tekbicim c ile
    Omega + n_e c ve ayni kuvvet, yuvarlama duzeyinde ve |c| ile olcekli
    toleransla). Tekbicim c ozdegerleri c kadar kaydirir;
    `mermin_dolum`un kT = 0 dejenerelik toleransi max|eps| ile olceklendigi
    icin sahte-dejenerelik bandi |c| ile buyur (beyanli, `mermin.py`).

    MALIYET (olculdu, 24.09.2026, bu makine; x'te N atomlu supercell, a = 2,
    5 tekrarin en kucugu): distance_model + H(Gamma) N = 16/32/64/128'de
    2.2/8.6/33.7/133 ms -- `neighbour_shells`in Python dongusu, ~N^2. Omega
    ve kuvvetle birlikte kafes yolu MD yolundan 9.5/20/44/51 kat yavas. MD
    bu yuzden kafes yolundan GECMEZ (spec 2.3); iki yolun esitligi M6'da
    kilitli (N = 128'de de |dOmega| 3.6e-15, max|dF| 2.5e-15).

    PARAMETRELER SECIMDIR (spec B4): yasa turetilmedi, secildi; bu kenarda
    DOGA CIPASI YOKTUR.
    """
    if not isinstance(ss_sigma, UsselRadyalYasa):
        raise InvalidInputError(
            f"ss_sigma bir UsselRadyalYasa olmali: {type(ss_sigma).__name__}"
        )
    kesme = ss_sigma.r_cut_bohr
    gereken = max_cell_for_cutoff(lattice, kesme)
    if max_cell is None:
        max_cell = gereken
    elif isinstance(max_cell, bool) or not isinstance(max_cell, (int, np.integer)):
        raise InvalidInputError(f"max_cell bir tamsayi olmali: {max_cell!r}")
    elif max_cell < gereken:
        raise InvalidInputError(
            f"max_cell = {max_cell} kesme {kesme} bohr icin yetmez: ters orgu "
            f"siniri {gereken} (daha kucugu kabuk kaybedebilir)"
        )
    ciftler = neighbour_shells(lattice, kesme, max_cell=int(max_cell))
    t, _ = ss_sigma.deger_ve_turev(np.array([c.distance for c in ciftler], dtype=float))
    return TBModel(
        lattice=lattice,
        onsite=onsite,
        hoppings=tuple(
            Hopping(i=c.i, j=c.j, cell=c.cell, t=float(t_c))
            for c, t_c in zip(ciftler, t)
        ),
    )


def finite_ring_matrix(model: TBModel, n_cell: int) -> np.ndarray:
    """Periyodik sinirli SONLU halkanin gercek-uzay matrisi (n_cell*n_orb kare).

    Bu, `bloch_hamiltonian`'dan TAMAMEN AYRI bir kod yoludur: hicbir faz
    carpani, hicbir ters orgu, hicbir kompleks us kullanmaz -- yalnizca
    hucre indislerini modulo n_cell baglar. Task 3'un capraz kontrolu bu
    ayriliga dayanir.
    """
    if n_cell < 3:
        raise InvalidInputError(f"n_cell en az 3 olmali, verilen: {n_cell}")
    if model.lattice.dimension != 1:
        raise InvalidInputError(
            f"sonlu halka yalniz 1B orgu icin tanimli, boyut: "
            f"{model.lattice.dimension}"
        )
    n_orb = model.n_orbital
    boyut = n_cell * n_orb
    H = np.zeros((boyut, boyut), dtype=float)
    for hucre in range(n_cell):
        for i in range(n_orb):
            H[hucre * n_orb + i, hucre * n_orb + i] += model.onsite[i]
    for h in model.hoppings:
        for hucre in range(n_cell):
            hedef = (hucre + h.cell[0]) % n_cell
            H[hucre * n_orb + h.i, hedef * n_orb + h.j] += h.t
    return H
