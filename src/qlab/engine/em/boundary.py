"""Sogurucu ve yansitici sinirlar: dereceli iletken PML, ve PEC.

Her sinir tek bir sey saglar: `sigma_1d(cells) -> (cells,)` elektrik
iletkenlik profili. `yee.py` sinirin TURUNU bilmez, yalniz bu diziyi gorur --
`medium` ile ayni gerekce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.errors import InvalidInputError

YUZLER = ("iki", "sol", "sag")


def _yerlestir(cells: int, profil: np.ndarray, yuzler: str) -> np.ndarray:
    """Katman profilini istenen yuz(ler)e koy.

    `yuzler` bir KOLAYLIK DEGIL, olcumun kosuludur: sogurucu yansimasi iki
    kosumun FARKINDAN okunur (test kosumu sogurucu ile, referans kosumu
    sogurucusuz ve daha uzun izgarada). Sogurucu iki uca birden konursa SOL
    uctan donen dalga da farka karisir ve olculen sey artik sag sogurucunun
    yansimasi olmaz.
    """
    d = int(profil.size)
    sig = np.zeros(cells)
    if d == 0:
        return sig
    if yuzler not in YUZLER:
        raise InvalidInputError(f"yuzler {YUZLER} icinden olmali, verilen: {yuzler!r}")
    if yuzler in ("iki", "sol"):
        sig[:d] = np.maximum(sig[:d], profil[::-1])
    if yuzler in ("iki", "sag"):
        sig[cells - d :] = np.maximum(sig[cells - d :], profil)
    return sig


@dataclass(frozen=True)
class PEC:
    """Mukemmel iletken duvar: sogurma YOK, sigma = 0 her yerde.

    Izgaranin iki ucundaki `E = 0` kosulu `yee.py`de HER sinir tipi icin
    kosar; PEC "sogurucuyu kaldir" demektir, "duvari kaldir" degil. Yani bu
    sinif tam yansitici bir bosluk kurar -- Task 4'un dairesel PEC bosluguyla
    ayni fizik, tek boyutta.
    """

    def sigma_1d(self, cells: int) -> np.ndarray:
        if cells < 3:
            raise InvalidInputError(f"cells en az 3 olmali, verilen: {cells}")
        return np.zeros(cells)

    @property
    def thickness(self) -> int:
        return 0


@dataclass(frozen=True)
class PML:
    """Dereceli iletken sogurucu sinir (1B'de eslesmis kayipli katman).

    KALINLIK 15, 12 DEGIL -- OLCULDU. Plan taslagi bir zamanlar "12 hucre PML
    30 hucre kayipli katmani HER acida gecer, kenar basina 18 hucre tasarruf"
    diyordu; bu CURUDU. Dik geliste (theta = 0) SEVK EDILEN kodla yeniden
    olculdu. PENCERE: S = 0.5, dx = 1, kaynak x = 1500, prob x = 2200,
    sogurucunun ic yuzu x = 2800, 6800 adim, Ricker wavelength = 110
    (sgf = 1), referans izgara 6400 hucre (sogurucusuz), DIKDORTGEN pencere.
    KAPI: sogurucu yerine PEC konunca |R - 1| en buyuk 4.602e-07 (GECTI).

        R(PML,     d = 12) = 8.3162e-03
        R(kayipli, d = 30) = 7.6132e-03      -> PML 1.0923 kat DAHA KOTU
        R(PML,     d = 15) = 5.2735e-03      -> GECIYOR (0.6927 kat)

    Yani d = 12 GECMIYOR ve DURUST TASARRUF KENAR BASINA 15 HUCREDIR. Hukum
    lambda boyunca da KARARLI: lam = 60/90/110/165/200'de d = 12'nin orani
    1.0166 / 1.0550 / 1.0923 / 1.1672 / 1.2006 -- BESI DE 1'in ustunde;
    d = 15'inki 0.6258 / 0.6649 / 0.6927 / 0.7451 / 0.7679 -- besi de altinda.

    AMA HUKUM `m`E BAGLIDIR ve bunu ne plan ne spec adlandiriyordu. Mutasyon
    turu sordurdu, olculdu (theta = 0, lam = 110, kayipli d = 30 = 7.6132e-03):

        m = 1 : d=12 5.9197e-03 GECER    d=15 3.8062e-03 GECER
        m = 2 : d=12 6.5265e-03 GECER    d=15 4.1410e-03 GECER
        m = 3 : d=12 8.3162e-03 GECMEZ   d=15 5.2735e-03 GECER
        m = 4 : d=12 1.0728e-02 GECMEZ   d=15 6.8193e-03 GECER

    Yani "d = 12 GECMIYOR" cumlesi `m = 3` ile birlikte verilmis bir
    hukumdur; `m` ADLANDIRILMADAN yazilamaz. `d = 15` hukmu dort `m`de de
    ayakta -- SEVK EDILEN sayi bu yuzden 15'tir, ve dayanagi tek bir m
    degeri degildir.

    DURUST OLUMSUZ: DIK GELISTE dusuk `m` daha iyidir (d = 15'te m = 1,
    m = 3'un 0.7218 kati yansitiyor; d = 12'de 0.7118 kati) ve bu modul `m`i
    EGIK geliste OLCMEDI -- egik olcum 1B-acili indirgemeyi gerektirir ve o
    Task 3'tur. `m = 3` dereceli PML'in yerlesik secimidir ve burada o
    gerekceyle sevk ediliyor, bu penceredeki bir olcum ustunlugu yuzunden
    DEGIL.

    "EN KOTU ACIDA 304 KAT KAZANC" YAZILAMAZ: ayni fizikte yalniz kayit
    uzunlugu degisince kazanc 17.1 ... 625.1 kat oynuyor, ve "en kotu aci"
    bir PENCERE ARTEFAKTIDIR -- aci kumesi 85 dereceye acilinca kazanc
    45 dereceden sonra monoton dusuyor. Kazanc GERCEKTIR, SAYISI YOKTUR.

    OLCUM ALETININ TAVANI VAR (bkz. source.gauss_cos): 1B-acili fark yontemi
    theta >= 75 derecede R > 1 donduruyor, yani fiziksel olarak imkansiz bir
    sayi. Bu bir sinir kusuru DEGIL, olcum aletinin bant genisligi
    arizasidir.

    `sigma_max` GIRILMEZ, hedef yansimadan TURETILIR:
        sigma_max = -(m + 1) * ln(r0) / (2 * d)      (eta = 1, c = 1)
    Bu, dereceli katmanin kuramsal dik-gelis yansimasini `r0`a oturtan
    kapali formdur; `m` profilin derecesidir.

    BEDEL OLCULDU ve plan taslaginin "adim basina 1.33 kat, bellek 1.30 kat"
    sayisi BU UYGULAMA ICIN YANLIS: o rakam Berenger BOLUNMUS-ALAN PML'e
    aittir (Ez'yi ikiye bolup fazladan alan tutar). 1B'de dik geliste
    bolunmus-alan PML ile eslesmis dereceli iletken AYNI semadir, dolayisiyla
    burada fazladan alan YOKTUR. Olculen (N = 3000, 4000 adim, dokuz
    tekrarin en iyisi): npml = 15 ve npml = 0 IKISI DE 0.0379 s, sure orani
    1.001 kat; tepe bellek 315.3 KiB vs 315.5 KiB, orani 1.000 kat. Yani
    fark OLCUM GURULTUSUNUN ICINDE. TEK gercek bedel kenar basina 15
    hucrelik IZGARADIR.
    (TUZAK, olculdu: ayni olcum `tracemalloc` acikken 0.111 s veriyor --
    uc kat. Bellek ve sure AYNI kosumda olculurse sure sayisi tracemalloc'un
    yukunu tasir; ikisi ayri kosuldu.)
    (2B bolunmus-alan PML Task 4'un isidir ve bedeli orada YENIDEN olculur.)
    """

    thickness: int = 15
    m: float = 3.0
    r0: float = 1e-6
    yuzler: str = "iki"

    def __post_init__(self) -> None:
        if self.yuzler not in YUZLER:
            raise InvalidInputError(
                f"yuzler {YUZLER} icinden olmali, verilen: {self.yuzler!r}"
            )
        if self.thickness < 0:
            raise InvalidInputError(
                f"PML kalinligi negatif olamaz, verilen: {self.thickness}"
            )
        if self.m <= 0.0:
            raise InvalidInputError(f"profil derecesi m > 0 olmali, verilen: {self.m}")
        if not 0.0 < self.r0 < 1.0:
            raise InvalidInputError(
                f"hedef yansima r0 (0, 1) araliginda olmali, verilen: {self.r0}"
            )

    @property
    def sigma_max(self) -> float:
        if self.thickness == 0:
            return 0.0
        return -(self.m + 1.0) * float(np.log(self.r0)) / (2.0 * self.thickness)

    def profile(self) -> np.ndarray:
        """Katman ICINDEKI profil; indis 0 ic yuz, `thickness-1` dis yuz.

        Ornek noktasi hucre MERKEZIDIR ((r + 0.5)/d), kenari degil: kenardan
        orneklense ic yuzde sigma = 0 cikar ve katmanin ilk hucresi hicbir
        sey yapmaz.
        """
        if self.thickness == 0:
            return np.zeros(0)
        r = (np.arange(self.thickness) + 0.5) / self.thickness
        return self.sigma_max * r**self.m

    def sigma_1d(self, cells: int) -> np.ndarray:
        """Iki uca da yerlestirilmis profil."""
        if cells < 3:
            raise InvalidInputError(f"cells en az 3 olmali, verilen: {cells}")
        if 2 * self.thickness >= cells:
            raise InvalidInputError(
                f"iki PML katmani izgaraya sigmali: 2 * {self.thickness} >= {cells}"
            )
        return _yerlestir(cells, self.profile(), self.yuzler)


@dataclass(frozen=True)
class LossyLayer:
    """PML'in REDDEDILEN rakibi: sabit `sigma_max`li dereceli kayipli katman.

    SEVK EDILMEZ ve sevk edilmemesi olculdu (bkz. PML dokstringi). Depoda
    durmasinin tek sebebi, "PML daha iyi" hukmunun testte YENIDEN
    olculebilmesidir: kiyas nesnesi kodda yoksa hukum de dogrulanamaz, plan
    metnine dayanan bir iddia olarak kalir.

    Fark tek satirdir: `sigma_max` hedef yansimadan TURETILMEZ, ELLE verilir.
    """

    thickness: int = 30
    m: float = 3.0
    sigma_max: float = 0.35
    yuzler: str = "iki"

    def __post_init__(self) -> None:
        if self.yuzler not in YUZLER:
            raise InvalidInputError(
                f"yuzler {YUZLER} icinden olmali, verilen: {self.yuzler!r}"
            )
        if self.thickness < 0:
            raise InvalidInputError(
                f"kalinlik negatif olamaz, verilen: {self.thickness}"
            )
        if self.sigma_max < 0.0:
            raise InvalidInputError(
                f"sigma_max negatif olamaz, verilen: {self.sigma_max}"
            )

    def profile(self) -> np.ndarray:
        if self.thickness == 0:
            return np.zeros(0)
        r = (np.arange(self.thickness) + 0.5) / self.thickness
        return self.sigma_max * r**self.m

    def sigma_1d(self, cells: int) -> np.ndarray:
        if cells < 3:
            raise InvalidInputError(f"cells en az 3 olmali, verilen: {cells}")
        if 2 * self.thickness >= cells:
            raise InvalidInputError(
                f"iki katman izgaraya sigmali: 2 * {self.thickness} >= {cells}"
            )
        return _yerlestir(cells, self.profile(), self.yuzler)
