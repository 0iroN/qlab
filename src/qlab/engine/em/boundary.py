"""Sogurucu ve yansitici sinirlar: dereceli iletken PML, ve PEC.

Her sinir DIZI saglar, davranis degil: 1B'de `sigma_1d(cells) -> (cells,)`
elektrik iletkenlik profili; 2B'de `sigma_2d(shape) -> (nx, ny)` ve
`pec_mask_2d(shape) -> (nx, ny)` bool IC ILETKEN maskesi. `yee.py` sinirin
TURUNU bilmez, yalniz bu dizileri gorur -- `medium` ile ayni gerekce.

Maskenin BURADA olmasi bilinclidir: geometri (dairenin yaricapi, kutunun
kenari) sinirin isidir, cekirdegin degil. Modul 5'in en guclu cipasi
(Bessel oranlari) ile onun pozitif kontrolu (kare kutu) AYNI `_adim_2d`den
gecer ve aralarindaki tek fark BU DIZIDIR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.errors import InvalidInputError

YUZLER = ("iki", "sol", "sag")


def _sekil(shape) -> tuple[int, int]:
    """2B sekil dogrulamasi -- TEK YERDE, dort sinir sinifi icin."""
    s = tuple(int(n) for n in shape)
    if len(s) != 2:
        raise InvalidInputError(f"2B sekil (nx, ny) olmali, verilen: {shape}")
    if min(s) < 3:
        raise InvalidInputError(
            f"2B izgara her yonde en az 3 hucre olmali, verilen: {s}"
        )
    return s


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

    def sigma_2d(self, shape: tuple[int, int]) -> np.ndarray:
        return np.zeros(_sekil(shape))

    def pec_mask_2d(self, shape: tuple[int, int]) -> np.ndarray:
        """IC iletken YOK -- bosluk izgaranin TAMAMI (kare kutu).

        Bu, Task 4'un `kare_kutu` KONTROLUDUR ve mutant DEGIL: dairesel
        boslugun oranlari Bessel kokleri, kare boslugunkiler
        sqrt(m^2+n^2)'lerdir. Iki geometri AYNI cekirdekten gecer.
        """
        return np.zeros(_sekil(shape), dtype=bool)

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
    buyuk acilarda R > 1 donduruyor, yani fiziksel olarak imkansiz bir sayi.
    Bu bir sinir kusuru DEGIL, olcum aletinin bant genisligi arizasidir --
    olculdu: `npml` 60 -> 120 -> 200 arizayi HIC oynatmiyor (theta = 55'te
    ||rho_s|-1| ucunde de 4.2533e-02), `sgf` ise oynatiyor.

    BU DOKSTRINGIN *"theta >= 75 derecede"* IBARESI Task 3'te OLCULEREK
    CURUDU: sevk edilen boruda `R > 1` bolgesi `sgf` = 9'da theta = 70-72.5,
    `sgf` = 3'te theta = 50-55'tir ve theta = 75'te ikisinde de R < 1'dir.
    Tam tablo `source.gauss_cos` dokstringindedir.

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

    2B'DE SOZ ODENDI (Task 4) ve IKI SONUC CIKTI:
    (1) BEDEL 2B'DE DE YOK. `sigma_2d` de bolunmus-alan DEGIL, iki eksende
        dereceli SKALER iletkenliktir; fazladan alan tutmaz. Olculdu
        (n = 241, 400 adim, besin en iyisi): npml = 0/15/40 icin
        0.1601/0.1524/0.1544 s, tepe bellek 7740.0 vs 7740.1 KiB (1.0000).
    (2) AMA "d = 15 YETER" HUKMU 1B'NINDIR ve 2B'ye TASINMAZ. Nokta kaynak
        her aciya birden isir, skaler iletken ise yalniz DIK bilende
        eslesir. Olculen kalan genlik orani (PEC'e gore): d = 8/15/25/40
        icin 7.999e-02 / 5.299e-02 / 2.117e-02 / 5.950e-03. 1B'de d = 15
        dik geliste 5.27e-03 veriyordu; 2B'de ayni kalinlik ON KAT daha
        kotu ve o duzeye inmek icin d = 40 gerekiyor.
    Bu bir kusur beyanidir, engel degil: Modul 5'in 2B fizik cipasi KAPALI
    bir bosluktur ve sogurucu KULLANMAZ. Kilit: tests/em/test_bessel.py.
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

    def sigma_2d(self, shape: tuple[int, int]) -> np.ndarray:
        """2B'de dort kenara yerlestirilmis profil -- ve ADI PML DEGIL.

        BEYAN, ve bu Task 4'te OLCULDU: burada kurulan sey Berenger'in
        BOLUNMUS-ALAN PML'i DEGIL, iki eksende ayri ayri dereceli SKALER
        iletkenliktir (kosede ikisinin BUYUGU alinir). 1B'de dik geliste
        ikisi AYNI semadir -- bu sinifin 1B dokstringi bunu yaziyor ve
        dogru. 2B'de DEGILDIR: skaler iletken egik gelen dalgayi
        eslestirmez, cunku eslestirme kosulu yalniz dik bilesende saglanir.
        """
        s = _sekil(shape)
        d = self.thickness
        for n in s:
            if 2 * d >= n:
                raise InvalidInputError(
                    f"iki PML katmani izgaraya sigmali: 2 * {d} >= {n} "
                    f"(sekil {s})"
                )
        profil = self.profile()
        sx = _yerlestir(s[0], profil, self.yuzler)[:, None]
        sy = _yerlestir(s[1], profil, self.yuzler)[None, :]
        return np.maximum(np.broadcast_to(sx, s), np.broadcast_to(sy, s))

    def pec_mask_2d(self, shape: tuple[int, int]) -> np.ndarray:
        """Sogurucu sinirda IC iletken yoktur."""
        return np.zeros(_sekil(shape), dtype=bool)


@dataclass(frozen=True)
class CircularPEC:
    """Dairesel PEC bosluk -- Modul 5'in EN GUCLU CIPASININ geometrisi.

    Izgaranin MERKEZINDEN `radius` uzaktaki her Ez dugumu iletkendir
    (Ez = 0). Merkez bir DUGUM uzerindedir: sekil (2R+1+2*margin)^2 secilir
    ve merkez tam ortadaki dugumdur.

    SINIR MERDIVENLENMISTIR ve bu bir kusur DEGIL, CIPANIN PENCERESIDIR:
    dairenin gercek yeri ile "r <= R" dugum kumesinin sinirlari arasindaki
    fark O(dx)'tir, dolayisiyla MUTLAK kip frekanslari yalniz 1. mertebeden
    yakinsar. ORANDA merdiven hatasi buyuk olcude gotusur; cipa bu yuzden
    ORAN olarak kurulur. Iki hukum de Task 4'te ayri ayri OLCULDU ve
    `tests/em/test_bessel.py`de ayri ayri kilitlendi.

    `sigma_2d` SIFIRDIR: bu bir sogurucu degil, TAM YANSITICI bir bosluktur.
    Cekirdek yine de `sigma_2d`yi cagirir -- sinirin TURUNU bilmedigi icin
    (bkz. modul dokstringi), ve "sogurma yok"un karsiligi sifir bir dizidir,
    ozel bir kod yolu degil.
    """

    radius: float

    def __post_init__(self) -> None:
        if not self.radius > 0.0:
            raise InvalidInputError(
                f"yaricap pozitif olmali, verilen: {self.radius}"
            )

    @property
    def thickness(self) -> int:
        return 0

    def sigma_1d(self, cells: int) -> np.ndarray:
        raise InvalidInputError(
            "CircularPEC yalniz 2B'de tanimlidir; 1B karsiligi PEC'tir "
            "(bkz. PEC dokstringi: 'ayni fizik, tek boyutta')."
        )

    def sigma_2d(self, shape: tuple[int, int]) -> np.ndarray:
        return np.zeros(_sekil(shape))

    def pec_mask_2d(self, shape: tuple[int, int]) -> np.ndarray:
        """`r > radius` olan dugumler iletken. MERDIVENLENME BURADA OLUYOR."""
        nx, ny = _sekil(shape)
        cx, cy = 0.5 * (nx - 1), 0.5 * (ny - 1)
        i = np.arange(nx)[:, None] - cx
        j = np.arange(ny)[None, :] - cy
        return (i * i + j * j) > self.radius * self.radius


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
