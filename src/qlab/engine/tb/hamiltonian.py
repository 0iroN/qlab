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
# distance_model'in iki yonu ayri hesaplamasindan -- H0-1'de, 11.10.2026,
# kapandi: iki yon artik bit bit ayni), es-eksik 0 -- yani esik
# 1000 kattan fazla pay birakir. Eleman-basina bagil olcut SECILMEDI: kucuk
# bir elemanin bagil yuvarlamasi (6e-14) ozdegeri etkilemez, olcege gore
# sapma etkiler.
# H0-2 (11.10.2026): ESI HIC YAZILMAMIS sifirdan farkli bir anahtar artik
# OLCEKTEN BAGIMSIZ reddedilir. Olculdu: R = 0'da 1e11 hartree'lik bir terim
# olcegi buyutup tek yonlu 0.05 hartree'lik R != 0 hoplamasini tolerans icine
# aliyordu (kabul, alt bant genisligi 0.1 -> 0.05). Yuvarlama bir esi YOK
# EDEMEZ (H0-1'den beri `neighbour_shells` iki yonu bit bit ayni hesaplar),
# yani bu dalda tolerans gereksizdi. Esi VAR ama degeri farkli anahtarda
# olcek BILINCLI olarak butun anahtarlardir, R = 0 (k'dan bagimsiz) dahil:
# asimetri delta ise ozdeger hatasi <= delta (Weyl) ve delta <= 1e-12 max|t|,
# yani hata H olceginin 1e-12'sini asamaz. Olcegi k'ya bagli terimlerle
# sinirlamak sinifi KAPATMAZDI (dev bir R != 0 terimi ayni isi gorur) --
# beyan, kapi degil.
_HERMIT_TOLERANSI = 1e-12

# E4 (11.10.2026): `bloch_hamiltonian` fazi double'da k . kayma olarak kurar
# (kayma = sum_e c_e a_e; atom gauge'unda + tau_j - tau_i). k ilk Brillouin
# bolgesinden N ters orgu vektoru oteyse hata N ile DOGRUSAL buyur, ucurum
# yoktur. OLCULDU (HEAD 80eb21c, kapisiz; tam faz referansi Fraction +
# Decimal): 2 orbitalli zincirde (k0 = 0.37, a = 2.5) k + N G'nin ozdeger farki
# N = 1e4 / 1e6 / 1e8 / 1e10 / 1e12'de 1.0e-11 / 5.3e-10 / 5.5e-09 / 1.2e-06 /
# 2.0e-04 eV; tek orbitalli `_zincir`de (test_hamiltonian) 1e12'de 6.57e-04 eV.
# Birincil hucreye double'da INDIRGEMEK duzeltmez (ayni zincir, N = 1e12:
# 2.5e-04 eV): k'nin k . a carpimi ve G ayni mertebede yuvarlanir. Bu yuzden
# KAPI.
# OLCU TERIM BASINA VE KARTEZYENDIR (dogrulama turunda duzeltildi, ayni gun):
# yuvarlama hatasi fazin DEGERIYLE degil, toplanan CARPIMLARIN buyuklugu
# P_h(k) = sum_x |k_x| M[h, x] ile olceklenir; M[h, x] = sum_e |c_e| |a_e,x|
# (+ |tau_j,x| + |tau_i,x| atom gauge'unda). Kesin ust sinir |hata| <= 11 u P_h
# (u = 2^-53; kayma birikimi, atom gauge'unun iki toplamasi ve 3'lu nokta
# carpimi ~8 u eder). M modelde bir kez kurulur, kapi O(n_hoplama). max_h P_h
# 8.2e5'i (11 u P = 1e-9 radyan; ozdegerde <= 1e-9 x satir basina sum|t|,
# Weyl) asan k `InvalidInputError` -- KABUL edilen her k'de her terimin faz
# hatasi <= 1e-9 rad. Sinir muhafazakardir: bu yorumdaki olcumlerde gercek
# faz hatasi 11 u P'nin 17-400 kat altinda (2 orbitalli zincirde 1e5 G'de
# hucre gauge'u P = 6.3e5 kabul, atom gauge'u P = 9.0e5 red).
# Ilk olcu (k'nin KESIRLI koordinati x max_h sum_e |c_e| [+ taban kolu]) iki
# DELIK birakiyordu (olculdu, kamu yolu): (1) EGIK taban a2 = (3e4 a, a)'da k
# 1e5 hucre oteden GECIYORDU -- kesirli koordinat kucuk, ama k_x a_2,x ve
# k_y a_2,y carpimlari birbirini goturur, hatalari goturmez (faz hatasi
# 1.5e-06 rad; 11 u P = 4.6e-05); (2) atom gauge'unda duzlem DISI taban farki
# (tau_z = 0.7, 1B orgu) ile k_z = 1e14 GECIYORDU (ters orgu vektorleri
# duzlemde, k_z'yi gormez): sessiz 2.3e-03 eV kayma. Terim basina olcu ikisini
# de reddeder. Kaba Kartezyen sinirin (max|k| sum|c| ||a||_1; N'de KARESEL)
# kusurunu da tasimaz: indirgenmemis tabanda ilk bolgedeki izgaranin P'si
# ~4 pi N |c| (N'de DOGRUSAL, gercek hata gibi) -- N = 1000 / 3e4 / 6e4'te
# 1.2e4 / 3.6e5 / 7.2e5 (kabul; olculen faz hatasi 2.5e-13 / 1.5e-12 /
# 3.1e-12 rad). Hucre gauge'unda dik k (1B/2B orguda k_z) carpima girmez
# (kayma_z TAM 0: faz tam, kapi da bakmaz). Ilk bolgede P ~2 pi |c|max (uzun
# menzilde 1e2-1e3). O kosul gurultusu (E2; 17 derece donuk tabanda N = 3e4'te
# 1.2e-7 W) KAPIDA DEGIL, `band_edges` kesirli yoldan gider -- beyan.
_FAZ_HATASI_SINIRI = 1e-9
_FAZ_OLCEGI_SINIRI = _FAZ_HATASI_SINIRI / (11 * 2.0**-53)


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
        # E4: faz carpimlarinin terim basina buyuklugu M (`_FAZ_HATASI_SINIRI`
        # yorumu), bir kez: M[h, x] = sum_e |c_e| |a_e,x|; atom gauge'unda
        # + |tau_j,x| + |tau_i,x|. Kapi max_h sum_x |k_x| M[h, x]'e bakar.
        d = self.lattice.dimension
        hucre = np.array(
            [[abs(float(h.cell[e])) for e in range(d)] for h in self.hoppings], dtype=float
        ).reshape(-1, d)
        m_hucre = hucre @ np.abs(self.lattice.vectors)
        taban = np.abs(self.lattice.basis)
        i_ler = np.array([h.i for h in self.hoppings], dtype=int)
        j_ler = np.array([h.j for h in self.hoppings], dtype=int)
        m_atom = m_hucre + taban[j_ler] + taban[i_ler]
        object.__setattr__(self, "_faz_carpimlari", (m_hucre, m_atom))

    def _hermit_kapisi(self) -> None:
        """Her (i, j, R) toplami (j, i, -R) toplamina esit olmali (t gercel).

        Anahtar yalniz ilk `dimension` hucre bilesenidir -- `bloch_hamiltonian`
        otekilerini okumaz. Ayni (i, j, R) birden cok kez yazilabilir;
        karsilastirilan TOPLAMDIR. Esi yazilmamis bir anahtar: toplami TAM 0
        ise kabul (tek yonlu SIFIR hoplama H'yi degistirmez), degilse olcekten
        bagimsiz red (H0-2; modul yorumu `_HERMIT_TOLERANSI`).
        """
        d = self.lattice.dimension
        toplam: dict[tuple, float] = {}
        for h in self.hoppings:
            anahtar = (h.i, h.j, tuple(h.cell[eksen] for eksen in range(d)))
            toplam[anahtar] = toplam.get(anahtar, 0.0) + float(h.t)
        olcek = max((abs(t) for t in toplam.values()), default=0.0)
        for (i, j, hucre), t in toplam.items():
            es = toplam.get((j, i, tuple(-c for c in hucre)))
            if (t != 0.0) if es is None else (abs(t - es) > _HERMIT_TOLERANSI * olcek):
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
    if not np.all(np.isfinite(k)):
        raise InvalidInputError(f"k sonlu olmali: {k}")
    m_hucre, m_atom = model._faz_carpimlari
    carpimlar = (m_atom if gauge == "atom" else m_hucre) @ np.abs(k)
    faz_olcegi = float(np.max(carpimlar, initial=0.0))
    if not faz_olcegi <= _FAZ_OLCEGI_SINIRI:
        h = model.hoppings[int(np.argmax(carpimlar))]
        raise InvalidInputError(
            f"k cok buyuk: faz carpimlarinin buyuklugu max_h sum_x |k_x| M[h, x] = "
            f"{faz_olcegi:.3e} > {_FAZ_OLCEGI_SINIRI:.3e} (en buyugu i={h.i}, j={h.j}, "
            f"cell={list(h.cell)}, gauge={gauge!r}); double faz hatasi "
            f"{_FAZ_HATASI_SINIRI:g} rad'i asabilir ve ozdeger SESSIZCE kayardi (E4). "
            "Birincil hucreye double'da indirgemek de kurtarmaz: k'yi ilk Brillouin "
            "bolgesinde ve orgu vektorlerini indirgenmis (kisa) verin."
        )

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
