"""Jeodezik kareleme -- CIPA YOLU (ODE degil).

Bu dosya bir tek sayi uretir: bir radyal periyotta supurulen acinin `2*pi`den
sapmasi, `p/M` ile olceklenmis hali:

    C = lim_{p -> sonsuz} (dphi(p) - 2*pi) * p / M

`Theory` kuram secicisiyle bu sayinin kapali formu `2*pi*(2 + 2*gamma - beta)`
(metric.py). Burasi onu KARELEMEYLE, yani yorungeyi cozerek uretir.

PENCERE -- ve bu pencerenin TAM BEYANI sonucun kendisi kadar sevk edilmistir:

- Birim secimi `G = c = 1` (metric.py'de bir kez beyan edildi).
- `psi` IKAMESI: `u = (u1+u2)/2 + ((u2-u1)/2)*sin(psi)`, `psi` [-pi/2, pi/2]
  araliginda; integrand `1/sqrt(w(u))`. Ikame kok tekilligini ANALITIK olarak
  kaldirir. Ham-`u` borusu SEVK EDILMIYOR (metric.py'de olculdu: e = 1e-6'da
  isaretce ters ve 63 kat yanlis).
- `scipy.integrate.quad`, `epsabs = 1e-14`, `epsrel = 1e-13`, `limit = 400`.
- Richardson `(3e5, 1e6)`, `1/p`de dogrusal disdegerleme.

RICHARDSON OPTIMUMDAN SONRA BOZULUYOR -- bu dosyanin kendi fonksiyonuyla
YENIDEN olculdu (GR, e = 0.3, M = 1, durust hane = -log10(|C-6pi|/6pi)):

    (1e5, 3e5) -> 9.12 | (3e5, 1e6) -> 10.01 | (1e6, 3e6) -> 9.94 |
    (3e6, 1e7) -> 8.98 | (3e7, 1e8) -> 8.93

Yani optimum (3e5, 1e6)'dadir ve bir disdegerleme SONSUZA KADAR IYILESMEZ.
Sebebi olculdu ve `grav_obs.newton_noise_floor`in tabaniyla AYNI seydir:
Richardson `~p` giden bir hatayi silmez, agirliklarinin toplami kadar
BUYUTUR; ust basamagin agirligi 1.4286 ve tasidigi yuvarlama kuantumu
`ulp(2*pi)*ust`tur.

`p` ILE `richardson` AYNI SEYI BELIRLER, o yuzden IKISI BIRDEN VERILEMEZ --
bkz. `perihelion_coefficient` dokstringi. Bu kural Task 1'in "iki alan ayni
seyi belirlerse sessiz bir kazanan dogar" olcumunun bu dosyadaki uygulanisidir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad, solve_ivp

from qlab.engine.grav.metric import (
    GAUGE_NAMES,
    Theory,
    closed_form_coefficient,
    isotropic_orbit,
    isotropic_w,
    null_kernel,
    null_w,
    radial_w,
)
from qlab.errors import InvalidInputError

#: `quad` dugmeleri -- sabittir, disari acilmaz. OLCULDU (bu modulun
#: sabitleri degistirilip sevk edilen fonksiyon kosturularak), bes ayar:
#: (1.49e-8, 1.49e-8), (1e-10, 1e-9), (1e-12, 1e-11), (1e-14, 1e-13),
#: (1e-16, 1e-15). C BESINDE DE BIT BIT AYNI (18.84955591968662); gurultu
#: tabani ilk DORDUNDE bit bit ayni (1.5942e-09), (1e-16, 1e-15)'te ise
#: `IntegrationWarning` cikip 2.2123e-09'a zipliyor. Yani en siki tolerans
#: HICBIR hane kazandirmiyor, yalnizca tabani kotulestiriyor; o yuzden
#: 1e-14/1e-13 sevk ediliyor. (`lambda = 0` kosusu bes ayarda da tam 0.0 --
#: dejenereligin dugmelerden bagimsiz oldugunun bir olcumu daha.)
PSI_EPSABS: float = 1e-14
PSI_EPSREL: float = 1e-13
PSI_LIMIT: int = 400

#: Sevk edilen Richardson cifti -- yukaridaki merdivenin OPTIMUMU.
DEFAULT_RICHARDSON: tuple[float, float] = (3e5, 1e6)

#: FIZIGI KAPATMA MERDIVENI -- `grav_obs.newton_noise_floor` bunu tuketir.
#:
#: `lambda = 0` NOKTASI TEK BASINA BIR HATA CUBUGU DEGILDIR ve bu OLCULDU:
#: `lam = 0`da `w == 1` (metric.py), integrand SABIT 1.0 olur, `quad` sabiti
#: tam integre eder ve `dphi - 2*pi` BIT BIT 0.0 cikar. On yedi ayri dugme
#: ayarinda (yedi `p`, dort Richardson cifti, alti (e, M) noktasi) donen deger
#: DEGISMEDI: hepsinde tam 0.0 (tests/grav/test_hata_cubugu.py). `quad`
#: toleransi da dogrudan degistirilerek denendi (1.49e-8 ... 1e-15, dort
#: ayar): yine tam 0.0. Yani o sifir bir OLCUM DEGIL BIR SABITTIR.
#:
#: Bu merdiven fizigi kapatmayi bir NOKTA yerine bir LIMIT olarak kurar:
#: `lam -> 0` giderken kesme hatasi `lam` ile birlikte kaybolur ama boru
#: aritmetigi CALISMAYA DEVAM EDER. Merdivenin iki ucu da OLCULEREK secildi:
#:   - ust uc 1e-4: daha buyuk `lam`da kesme hatasi girmeye baslar (1e-2'de
#:     artik 2.0228e-09'a cikiyor, plato degeri 1.5942e-09);
#:   - alt uc 1e-9: `lam*M/p` makine epsilonunun altina inince `w` yeniden
#:     tam 1.0'a yuvarlanir ve basamak OLUR. OLCULEN olum esigi (sevk edilen
#:     Richardson penceresi): `lam = 1e-11`de boru hala calisiyor (-2.284e-10),
#:     `lam = 7e-12`de tam 0.0. Merdivenin alt ucu ondan ~100 kat yukarida.
#:     Arada bir KISMEN OLU bolge de var ve o da olculdu: `lam` 1e-11 ile
#:     2e-11 arasinda YALNIZ alt basamak (p = 3e5) canlidir, ust basamak
#:     (p = 1e6) tam 0.0 doner ve taban 1.594e-09 yerine 2.284e-10 gorunur --
#:     yani kismen olu bir merdiven tabani 7 KAT KUCUK gosterir.
#: Merdiven boyunca artik `lam`DAN BAGIMSIZDIR -- bes dekatta yalniz 5.08 kat
#: oynuyor. Kesme hatasi (`~lam^2`) baskin olsaydi ayni bes dekatta 1e10 kat
#: oynardi. Bu, merdivenin FIZIGI DEGIL ARITMETIGI olctugunun olcumudur.
NEWTON_PROBE_LAMBDAS: tuple[float, ...] = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)


#: SEVK EDILEN IKI AYAR -- ve bu ad artik `metric.GAUGE_NAMES`ten OKUNUR,
#: burada YENIDEN yazilmaz. Task 4 ayni adlari bos jeodezikte de ihtiyac
#: duydu; iki liste ayni seyi belirlerse sessiz bir kazanan dogar (Task 1'in
#: olcumu) ve o kazanan adlandirilmamis bir pencere dugmesidir.
#: Birinci mertebe katsayisi `C` ayardan BAGIMSIZDIR (olculdu,
#: `tests/grav/test_perihel.py`); ikinci mertebe DEGILDIR
#: (`grav_obs.second_order_coefficient` dokstringi). Bukulmede AYNI HUKUM
#: cikti (`tests/grav/test_bukulme.py`): `1+gamma` ayardan bagimsiz,
#: `(oran-2)*b/M` degil.
GAUGES: tuple[str, ...] = GAUGE_NAMES


def _donus_noktalari(p: float, e: float) -> tuple[float, float]:
    """Afel ve perihel: `u1 = (1-e)/p`, `u2 = (1+e)/p`.

    IKI AYAR DA BU FONKSIYONU KULLANIR ve bu bir ORTAK DUGMEDIR -- capraz
    kontrolun GOREMEDIGI yuzey burasidir. OLCULDU (GR, e = 0.3, M = 1,
    Richardson (3e5, 1e6)); mutant `1+e` -> `1+1.001*e`:

        temiz   schw 18.84955591968662   izo 18.849555921564487  |fark| 1.878e-09
        mutant  schw 18.852383354455476  izo 18.852383352831374  |fark| 1.624e-09

    Yani mutant iki ayari BIRLIKTE 2.827e-03 kaydiriyor ve aradaki fark
    BUYUMUYOR, KUCULUYOR: capraz kontrol tamamen SESSIZ kalir. Onu olduren sey
    capraz kontrol degil, `6*pi` CIPASIDIR. Ayni sey `quad` dugmeleri, `psi`
    ikamesi ve Richardson cifti icin de gecerlidir.
    """
    return (1.0 - e) / p, (1.0 + e) / p


def _dphi_schwarzschild(*, theory: Theory, M: float, p: float, e: float) -> float:
    """Schwarzschild ayari: integrand `1/sqrt(w(u))`, `w` `u`da DOGRUSAL.

    `Theory`den YALNIZ `lam` okunur (olculdu, kaydeden vekil).
    """
    u1, u2 = _donus_noktalari(p, e)
    orta = 0.5 * (u1 + u2)
    yari = 0.5 * (u2 - u1)

    def integrand(psi: float) -> float:
        u = orta + yari * math.sin(psi)
        return 1.0 / math.sqrt(radial_w(u, u1, u2, theory=theory, M=M))

    deger, _ = quad(
        integrand, -math.pi / 2, math.pi / 2,
        epsabs=PSI_EPSABS, epsrel=PSI_EPSREL, limit=PSI_LIMIT,
    )
    return 2.0 * deger


def _dphi_isotropic(*, theory: Theory, M: float, p: float, e: float) -> float:
    """Izotropik PPN ayari: integrand `1/sqrt(R(u)/A(u))`, ikisi de KARESEL.

    `Theory`den YALNIZ `gamma` ve `beta` okunur, `lam` HIC gorulmez (olculdu,
    kaydeden vekil). Cekirdek `metric.isotropic_orbit`ta kurulur -- 2x2 Cramer
    + kuartik deflasyon; Schwarzschild yolunun tek satirlik `w`siyle ORTAK
    HICBIR GOVDESI YOKTUR.

    Cekirdek DONGU DISINDA bir kez kurulur. SAYIM tarafi KESINDIR ve testte
    kilitlidir (GR, p = 3e5): `quad` integrandi basamak basina 21 kez cagirir,
    `isotropic_orbit` ise 1 kez; Richardson cagrisinda 42'ye 2.

    ZAMAN tarafi GURULTULUDUR ve o yuzden ARALIK olarak yazilir (bes tekrarli
    olcum, ayni makine): `isotropic_orbit` 1.08 ... 1.23 us, `isotropic_w`
    0.118 ... 0.122 us -- yani cekirdek dongu ICINDE kurulsaydi integrand
    basina ~10 KAT pahali olurdu. Iki ayarin Richardson cagrisi ise
    AYIRT EDILEMEZ: 0.0149 ... 0.0157 ms (Schwarzschild) ve 0.0143 ... 0.0169 ms
    (izotropik), araliklar tamamen ORTUSUYOR. Yani iki ayarli capraz kontrolun
    maliyeti tek cipanin ~2 KATIDIR ve bu ODENEBILIR. (Spec 7.3 kendi eledigi
    "iki-cipa-yolu" icin 14.07 ms yaziyor; O SAYIYI BEN OLCMEDIM ve baska bir
    kurulumdan geliyor, o yuzden buraya bir oran yazilmiyor.)
    """
    u1, u2 = _donus_noktalari(p, e)
    orbit = isotropic_orbit(u1, u2, theory=theory, M=M)
    orta = 0.5 * (u1 + u2)
    yari = 0.5 * (u2 - u1)

    def integrand(psi: float) -> float:
        u = orta + yari * math.sin(psi)
        return 1.0 / math.sqrt(isotropic_w(u, orbit))

    deger, _ = quad(
        integrand, -math.pi / 2, math.pi / 2,
        epsabs=PSI_EPSABS, epsrel=PSI_EPSREL, limit=PSI_LIMIT,
    )
    return 2.0 * deger


_DPHI = {
    "schwarzschild": _dphi_schwarzschild,
    "isotropic": _dphi_isotropic,
}


def _dphi(*, theory: Theory, M: float, p: float, e: float, gauge: str) -> float:
    """Bir radyal periyotta supurulen aci -- SECILEN ayarda."""
    return _DPHI[gauge](theory=theory, M=M, p=p, e=e)


def _C_tek_basamak(*, theory: Theory, M: float, p: float, e: float,
                   gauge: str) -> float:
    """Tek `p` basamaginda `(dphi - 2*pi) * p / M`."""
    return (_dphi(theory=theory, M=M, p=p, e=e, gauge=gauge) - 2.0 * math.pi) * p / M


def _kapi(*, M: float, e: float, p: float | None,
          richardson: tuple[float, float] | None, gauge: str) -> None:
    """Girdi kapilari. Hepsi `InvalidInputError` atar ve mesajinda OLCUM tasir."""
    if gauge not in _DPHI:
        raise InvalidInputError(
            f"bilinmeyen ayar: {gauge!r}. Sevk edilen ayarlar: {GAUGES}. "
            "Ayar bir KOORDINAT SECIMIDIR ve sonucun penceresine girer: "
            "birinci mertebe katsayisi C ayardan bagimsizdir (olculdu, bes PPN "
            "noktasinda iki ayar 1.497e-09 ... 3.312e-09'da usuyor), ikinci "
            "mertebe DEGILDIR (GR, e = 0.3, p = 3e5: K_schw = 85.248585, "
            "K_izo = 26.577757)."
        )
    if not (M > 0.0) or not math.isfinite(M):
        raise InvalidInputError(
            f"M pozitif ve sonlu olmali, verilen: {M!r}. C = (dphi-2pi)*p/M "
            "oldugu icin M = 0 sifira bolmedir, M < 0 ise fiziksel olarak "
            "anlamsizdir."
        )
    if not math.isfinite(e) or not (0.0 < e < 1.0):
        raise InvalidInputError(
            f"e BAGLI bir yorungenin dismerkezligi olmali (0 < e < 1), "
            f"verilen: {e!r}. Kapi SOZLESMESELDIR, sayisal degil -- olculdu "
            "(GR, M=1, Richardson (3e5,1e6)): e = 1.0'da kareleme sessizce "
            "18.84955592043523, e = 2.0'da 18.84955591693327 donduruyor. "
            "Yani inanilir gorunen bir sayi doner, ama e >= 1'de KAPALI BIR "
            "YORUNGE YOKTUR ve 'yorunge basina perihel kaymasi' tanimsizdir. "
            "e = 0 ise dairedir ve `radial_w` onu ayrica reddeder."
        )
    if richardson is None and p is None:
        raise InvalidInputError(
            "basamak tanimsiz: `richardson=None` verildiginde tek basamak `p` "
            "ZORUNLUDUR."
        )
    if richardson is not None and p is not None:
        raise InvalidInputError(
            f"`p` ile `richardson` AYNI SEYI belirler, ikisi birden verilemez "
            f"(verilen p={p!r}, richardson={richardson!r}). Iki alan ayni seyi "
            "belirlerse sessiz bir kazanan dogar ve o kazanan adlandirilmamis "
            "bir pencere dugmesidir. Richardson icin `p`yi verMEYIN; tek "
            "basamak icin `richardson=None` verin."
        )
    if richardson is not None:
        if len(richardson) != 2:
            raise InvalidInputError(
                f"richardson bir (alt, ust) CIFTI olmali, verilen: {richardson!r}"
            )
        alt, ust = richardson
        for ad, deger in (("alt", alt), ("ust", ust)):
            if not math.isfinite(deger) or not (deger > 0.0):
                raise InvalidInputError(
                    f"richardson.{ad} pozitif ve sonlu olmali, verilen: {deger!r}"
                )
        if alt == ust:
            raise InvalidInputError(
                f"richardson basamaklari AYRI olmali, verilen: {richardson!r}. "
                "Esit basamaklarda `1/p`de disdegerleme sifira bolmedir."
            )
    elif not math.isfinite(p) or not (p > 0.0):  # type: ignore[arg-type]
        raise InvalidInputError(f"p pozitif ve sonlu olmali, verilen: {p!r}")


def perihelion_coefficient(
    *,
    theory: Theory = Theory(),
    M: float,
    e: float,
    p: float | None = None,
    richardson: tuple[float, float] | None = DEFAULT_RICHARDSON,
    gauge: str = "schwarzschild",
) -> float:
    """Perihel katsayisi `C = (dphi - 2*pi) * p / M` -- KARELEME (CIPA YOLU).

    SINIF: FIZIK CIPASI. Statik koku yoktur; sayi ancak yorunge cozulerek
    cikar. (Karsilastir: ISCO'nun statik koku VAR ve o "kapali-form turev"
    rafindadir.)

    OLCULEN (GR, M = 1, e = 0.3, dosya penceresi, Richardson (3e5, 1e6)):

        C   = 18.84955591968662
        6pi = 18.84955592153876
        fark -1.852e-09, yani 10.01 DURUST HANE

    `P` ILE `RICHARDSON` ARASINDAKI ILISKI IMZADA TANIMLIDIR ve tanim bir
    SESSIZ KAZANANI onlemek icin YASAKLAYICIDIR:

        richardson = (alt, ust)  ->  basamaklar CIFTTEN gelir, `p` VERILEMEZ
        richardson = None        ->  tek basamak `p`, ve `p` ZORUNLUDUR
        ikisi birden             ->  InvalidInputError
        ikisi de yok             ->  InvalidInputError

    Planin Task 2 Step 1'i `p=3e5` ile `richardson=(3e5,1e6)`i AYNI cagrida
    veriyordu; o cagrinin ne demek oldugu tanimsizdir ve burada REDDEDILIR.

    IKI PENCERE AYNI ETIKET ALTINDA TOPLANAMAZ -- olculdu, ve ikisini ayiran
    sey planin dedigi gibi "sinyalin tanimi" degil, NEYIN SINIRLADIGIDIR:

        pencere                 C                    |C - 6pi|   taban     taban/hata
        (1) tek basamak p=3e5   18.84984008349022    2.842e-04   2.453e-10  8.63e-07
        (2) Richardson (3e5,1e6) 18.84955591968662   1.852e-09   1.594e-09  8.61e-01

    Pencere (1) KESME SINIRLIDIR: gercek hatasi gurultu tabaninin 1.16 MILYON
    KATI. Orada S/G = 7.68e+10 "10.89 hane" der ama gercek hane 4.82'dir --
    yani gurultu tabani o pencerede bir hata cubugu DEGILDIR, cunku baskin
    hatayi gormez. Pencere (2) YUVARLAMA SINIRLIDIR: taban gercek hatanin
    0.86 katidir ve S/G'nin hanesi (10.07) gercek haneyi (10.01) 0.07 hane
    icinde ongorur. `newton_noise_floor` bu yuzden bir hane ONGORUSUDUR, bir
    hane GARANTISI degil.

    `theory` VARSAYILANI GR'dir (`Theory()`), yani bu fonksiyon argumansiz
    cagrildiginda GR'yi cozer; `newton_noise_floor` SINYALI bu varsayilandan
    okur ve fizigi kapatma noktasini CAGIRANA BIRAKMAZ.

    `GAUGE` -- VE BU MODULUN GERCEK CAPRAZ KONTROLU BURADADIR.
    Iki ayar AYNI SAYIYI iki TAMAMEN AYRI govdeyle uretir:

        "schwarzschild" -- `w(u) = 1 - (2*lam*M/3)(u1+u2+u)`, `u`da DOGRUSAL;
                           `Theory`den YALNIZ `lam` okunur
        "isotropic"     -- `R(u)/A(u)`, 2x2 Cramer + kuartik deflasyon;
                           `Theory`den YALNIZ `gamma` ve `beta` okunur

    ORTAK DUGME YOK -- ve bu VARSAYILMADI, kaydeden vekille OLCULDU (bkz.
    `tests/grav/test_perihel.py`): dokunulan ad kumeleri `{"lam"}` ve
    `{"gamma", "beta"}`, kesisimleri BOS. Bu, spec 7.3'un dorduncu maddesinin
    (ortak dugme capraz kontrolu KORLESTIRIR) bu modulde ILK KEZ
    SAGLANMASIDIR: kareleme <-> `closed_form_coefficient` karsilastirmasi
    ikisi de `lam`i okudugu icin `lam` mutantina KORDU; iki ayar karsilastirmasi
    DEGIL (olculdu, mutasyon turu).

    OLCULEN FARK (Richardson (3e5, 1e6), e = 0.3, M = 1, bes PPN noktasi):
    1.497e-09 ... 3.312e-09; `e` taramasinda (GR, yedi e) 3.807e-11 ... 3.705e-09.

    CAPRAZ KONTROLUN GORMEDIGI YUZEY DE ADIYLA YAZILI: iki ayar `quad`
    dugmelerini, `psi` ikamesini, Richardson ciftini ve `_donus_noktalari`yi
    PAYLASIR. Bunlardan birini bozan bir mutant iki tarafi BIRLIKTE kaydirir
    ve capraz kontrol susar (olculdu, `_donus_noktalari` dokstringi).
    """
    _kapi(M=M, e=e, p=p, richardson=richardson, gauge=gauge)
    if richardson is None:
        return _C_tek_basamak(theory=theory, M=M, p=float(p), e=e, gauge=gauge)
    alt, ust = richardson
    f1 = _C_tek_basamak(theory=theory, M=M, p=alt, e=e, gauge=gauge)
    f2 = _C_tek_basamak(theory=theory, M=M, p=ust, e=e, gauge=gauge)
    h1, h2 = 1.0 / alt, 1.0 / ust
    return (f1 * h2 - f2 * h1) / (h2 - h1)


def _newton_probes() -> tuple[tuple[dict[str, object], float], ...]:
    """`(fizigi kapatan kwargs, o noktanin TAM degeri)` ciftleri.

    Aile `Theory(gamma=-1, beta=-lam)`dir; `lam_eff = 2 + 2*(-1) - (-lam)`
    ve `2 + 2*(-1)` kayan noktada TAM 0.0 oldugu icin `lam_eff` istenen
    degere BIT BIT esittir (olculdu: yedi `lam` degerinde tam esitlik).
    `lam = 0` ucu planin `Theory(gamma=g, beta=2+2*g)` ailesinin ta kendisidir
    (`g = -1`).

    TAM DEGER `closed_form_coefficient`tan okunur, burada YENIDEN yazilmaz --
    `2*pi*lam`in ikinci bir kopyasi ikinci bir sessiz kazanan demektir.
    """
    ciftler = []
    for lam in NEWTON_PROBE_LAMBDAS:
        t = Theory(gamma=-1.0, beta=-lam)
        ciftler.append(({"theory": t}, closed_form_coefficient(t)))
    return tuple(ciftler)


#: Boru etiketi. `newton_noise_floor` bunu `fn.__name__`DEN DEGIL buradan
#: okur: ad degisirse etiket degismemeli, ve iki farkli fonksiyon ayni boruyu
#: kullanabilmelidir.
perihelion_coefficient.PIPELINE = "kareleme"  # type: ignore[attr-defined]
perihelion_coefficient.NEWTON_PROBES = _newton_probes()  # type: ignore[attr-defined]

#: BORUYU SECEN argumanlar. Task 3 `gauge`u eklediginde `"kareleme"` etiketi
#: TEK BASINA YALAN SOYLEMEYE BASLADI: iki AYRI govde (Schwarzschild `w`si ve
#: izotropik `R/A`si) ayni etiketi donduruyordu, oysa tabanlari FARKLI
#: (1.594187e-09 <-> 2.346903e-09) ve merdiven olum esikleri de farkli
#: (p/M = 1e13 <-> 1e14). Karisik boru yasaginin kaydi ancak ayar etikete
#: yazilirsa iki ayari ayirt edebilir. Varsayilan buraya KOPYALANMAZ,
#: imzadan okunur.
perihelion_coefficient.PIPELINE_KEYS = ("gauge",)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# BOS JEODEZIK -- ISIK BUKULMESI (kareleme, CIPA YOLU)
# ---------------------------------------------------------------------------

#: Bukulme karelemesinin `theta` ikamesi icin `quad` dugmeleri. Bagli
#: yorungenin `psi` dugmeleriyle AYNI degerler ama AYRI sabitler: iki boru
#: birbirinin dugmesine guvenmez.
#:
#: VE BU BORUDA TOLERANS TAMAMEN ETKISIZ -- olculdu, varsayilmadi. Sevk
#: edilen kodda bu iki sabit degistirilip kosuldu, DORT ayar:
#: (1.49e-8, 1.49e-8) yani `quad`in kendi varsayilani, (1e-12, 1e-11),
#: (1e-14, 1e-13), (1e-16, 1e-15). DORDUNDE DE:
#:     delta(GR, M=1, b=1e5) = 4.000117813962589e-05   BIT BIT ayni
#:     oran                  = 1.9999999996887523      BIT BIT ayni
#:     gurultu tabani        = 3.651917e-10            BIT BIT ayni
#: Sebep yapisaldir: `theta` ikamesinden sonra integrand [0, pi/2]'de
#: PURUZSUZDUR ve `quad`in 21 noktali Gauss-Kronrod kurali onu daha ilk
#: bolmede makine hassasiyetinde integre eder. Yani bu boruda hane KARELEME
#: toleransiyla degil, `1/b`de Richardson penceresiyle kazanilir.
#: (1e-16, 1e-15) ayari burada da `IntegrationWarning` URETIYOR -- ama
#: DEGERI DEGISTIRMIYOR, ve bu ikisi ayri seylerdir. Bagli yorungenin `psi`
#: borusunda ayni ayar tabani 1.5942e-09'dan 2.2123e-09'a ZIPLATIYORDU;
#: burada uyari cikiyor, sayi bit bit ayni kaliyor.
NULL_EPSABS: float = 1e-14
NULL_EPSREL: float = 1e-13
NULL_LIMIT: int = 400

#: Bukulme oraninin sevk edilen Richardson cifti -- `1/b`de dogrusal.
#: OLCULEN merdiven (GR, M = 1, Schwarzschild ayari, durust hane
#: `-log10(|oran-2|/2)`):
#:   (1e3,3e3) -> 5.64 | (1e4,3e4) -> 7.64 | (1e4,1e5) -> 8.16 |
#:   (1e5,3e5) -> 9.81 | (1e5,1e6) -> 9.37 | (1e6,3e6) -> 9.30 |
#:   (1e6,1e7) -> 8.41 | (1e7,3e7) -> 7.69
#: Optimum (1e5, 3e5)'tedir ve ORADAN SONRA BOZULUYOR -- karelemenin
#: `p` merdiveninin bukulme tarafindaki birebir karsiligi.
DEFAULT_NULL_RICHARDSON: tuple[float, float] = (1e5, 3e5)


#: BUKULME BORUSUNUN FIZIGI KAPATMA MERDIVENI: `gamma = -1 + eps`.
#:
#: `M = 0` NOKTASI TEK BASINA BIR HATA CUBUGU DEGILDIR ve bu olculdu:
#: `M = 0`da `A = T = 1`, `Q(u) = u0 + u` olur ve integrand BIT BIT 1.0'a
#: duser, `delta` TAM 0.0 cikar -- iki ayarda da. Karelemedeki `lambda = 0`
#: sabitinin birebir esi. (Ustelik `M > 0` kapisi o noktayi API'den de
#: erisilemez kilar.)
#:
#: Merdiven yerine `1 + gamma -> 0` LIMITI sevk ediliyor.
#:
#: VE BU MERDIVENIN UCLARI ZORUNLU DEGILDIR -- bu, karelemenin
#: `NEWTON_PROBE_LAMBDAS`iyla arasindaki OLCULMUS FARKTIR ve orada yazan
#: gerekce buraya TASINMAZ. Karelemede ust ucu kesme hatasi, alt ucu `w`nin
#: yeniden tam 1.0'a yuvarlanmasi (basamagin OLMESI) zorluyordu; burada
#: ikisi de olmuyor. Olculdu (Schwarzschild, M = 1, Richardson (1e5, 3e5),
#: artik `|oran - (1+gamma)|`):
#:
#:   eps    5e-1      1e-1      1e-2      1e-3      1e-4      1e-9
#:   artik  1.353e-10 8.141e-11 3.082e-10 3.398e-10 3.652e-10 2.451e-10
#:   eps    1e-12     1e-14     1e-15     1e-16
#:   artik  3.448e-10 3.458e-10 3.458e-10 3.458e-10
#:
#: Yani artik ON BESTEN FAZLA DEKAT boyunca 8.14e-11 ile 3.65e-10 arasinda,
#: toplam 4.5 kat oynuyor: merdiven FIZIGI DEGIL ARITMETIGI olcuyor. Alt
#: uctaki DOYMA da olculdu: eps <= 1e-12'de `oran` BIT BIT ayni degeri
#: (3.45805844e-10) donduruyor, cunku `2*INT - pi` farkinin mutlak
#: yuvarlamasi `(1+gamma)`yi tamamen orter. OLU basamak (bit bit 0.0) HIC
#: gorulmedi: her pencerede 6/6 canli, yani bu boru `DEJENERE` OLAMAZ.
#: 1e-4 ... 1e-9 araligi bu platonun TEMSILI bir dilimidir; secim
#: karelemedeki gibi zorunlu degil, ACIKCA BIR TERCIHTIR.
DEFLECTION_PROBE_EPSILONS: tuple[float, ...] = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)


def deflection_angle(*, theory: Theory = Theory(), M: float, b: float,
                     gauge: str = "schwarzschild") -> float:
    """Isigin bukulme ACISI `delta` -- KARELEME (CIPA YOLU).

        delta = 2 * INT_0^{u0} du / sqrt((du/dphi)^2) - pi

    ve `metric.null_kernel`in `w` formunda, `u = u0*sin(theta)` ikamesiyle

        delta = 2 * INT_0^{pi/2} sqrt(u0*(1 + sin theta)*w(u)) dtheta - pi.

    INTEGRAL `u = 0`DAN BASLAR, YANI KESME YOKTUR. Bu bir ayrinti degil,
    olculmus bir gerekce: sonlu bir `r_start`tan baslayan bir surumde
    kacirilan asimptot acisi `2b/r_start`tir ve b = 1e5, r_start = 1e6'da
    0.2 rad eder -- sinyalin (4e-5 rad) 5000 KATI. O surum "oran = 1.000000"
    verir, yani GR'yi NEWTON SANIR. Burada alt sinir TAM `u = 0`dir.

    OLCULEN (GR, M = 1, dosya penceresi):

        b       delta                      delta*b/M     (delta*b/M - 4)*b/M
        1e3     4.011823809925e-03         4.011824      11.823810
        1e4     4.001178524082e-04         4.001179      11.785241
        1e5     4.000117813963e-05         4.000118      11.781396
        1e6     4.000011781358e-06         4.000012      11.781358

    Birinci mertebe `4*M/b` (GR), ikinci mertebe `15*pi/4` = 11.780972
    (`NullPotentials` dokstringi: literaturun `15*pi/4 - 4`u `r0` cinsinden,
    bu `b` cinsinden). b = 1e7'de sayi BOZULUYOR (11.791719) cunku `delta`nin
    mutlak ulp'si `b^2` ile carpilir: b = 1e6'da +-3e-04, b = 1e7'de +-3e-02.

    `beta` IKINCI MERTEBEDE GORUNUR VE O DA KAPALI FORMDUR:
    `d(delta)/d(beta) = -pi*M^2/b^2`. Olculdu (SEVK EDILEN kodla, merkezi
    fark h = 1e-3); `d(delta)/d(beta) / (-pi*M^2/b^2)`:

        b        schwarzschild   isotropic
        1e3      1.007687        1.010255
        1e4      1.000751        1.001020
        1e5      1.000815        1.000108

    Fark adimi `h` BIR PENCERE DUGMESIDIR: ayni b = 1e5'te h = 1e-4 ile iki
    ayar da 1.010710 verir, cunku turevin mutlak buyuklugu `1/b^2` ile
    duserken farkin mutlak yuvarlamasi sabit kalir ve `h` kuculdukce oran
    BOZULUR -- iyilesmez.

    `b <= b_krit` ISE FOTON YAKALANIR ve `InvalidInputError` atilir
    (`metric._en_kucuk_pozitif_kok`).
    """
    _null_kapi(M=M, b=b, gauge=gauge)
    kernel = null_kernel(theory=theory, M=M, b=b, gauge=gauge)
    u0 = kernel.turning_point

    def integrand(theta: float) -> float:
        s = math.sin(theta)
        return math.sqrt(u0 * (1.0 + s) * null_w(u0 * s, kernel))

    deger, _ = quad(
        integrand, 0.0, math.pi / 2,
        epsabs=NULL_EPSABS, epsrel=NULL_EPSREL, limit=NULL_LIMIT,
    )
    return 2.0 * deger - math.pi


def _null_kapi(*, M: float, b: float, gauge: str) -> None:
    """Bukulme borusunun girdi kapilari; mesajlarinda OLCUM tasirlar."""
    if gauge not in GAUGE_NAMES:
        raise InvalidInputError(
            f"bilinmeyen ayar: {gauge!r}. Sevk edilen ayarlar: {GAUGE_NAMES}. "
            "Ayar bukulmede de pencereye girer: birinci mertebe (`1+gamma`) "
            "ayardan bagimsizdir (iki ayar 1.159457e-10'da usuyor), ikinci "
            "mertebe DEGILDIR ((oran-2)*b/M = 7*pi/8 <-> pi/2, fark tam "
            "3*pi/8)."
        )
    if not (M > 0.0) or not math.isfinite(M):
        raise InvalidInputError(
            f"M pozitif ve sonlu olmali, verilen: {M!r}."
        )
    if not math.isfinite(b) or not (b > 0.0):
        raise InvalidInputError(f"b pozitif ve sonlu olmali, verilen: {b!r}")


# ---------------------------------------------------------------------------
# ODE -- YORUNGE MOTORU (cipa yolu DEGIL) ve KENDI BORUSU
# ---------------------------------------------------------------------------

#: ODE penceresi. `rtol`/`atol` BIRER PENCERE DUGMESIDIR ve hukme girer;
#: OLCULEN merdiven (GR, e = 0.3, M = 1, DOP853, durust hane):
#:
#:   rtol      (1e3,3e3)  (3e3,1e4)  (1e4,3e4)  (3e4,1e5)  (1e5,3e5)
#:   1e-10       5.22       4.91       4.18       2.99       2.26
#:   1e-12       5.12       6.20       5.93       4.88       3.83
#:   1e-13       5.12       6.12       7.59       5.87       4.90
#:
#: Yani ODE'nin Richardson optimumu `rtol`la YER DEGISTIRIR (1e-12'de
#: (3e3,1e4), 1e-13'te (1e4,3e4)). Spec 4.5'in "(1e4,3e4) -> 7.20" satiri bir
#: PENCERE BEYANI TASIMIYORDU; bu depoda ayni cift rtol = 1e-12'de 5.93,
#: rtol = 1e-13'te 7.59 veriyor.
ODE_RTOL: float = 1e-13
ODE_ATOL: float = 1e-15

#: ODE'nin sevk edilen Richardson cifti (yukaridaki tablonun optimumu).
DEFAULT_ODE_RICHARDSON: tuple[float, float] = (1e4, 3e4)

#: Sevk edilen integratorler. KAPALI BIR LISTE, ve sebebi olculdu:
#: `newton_noise_floor` tabani metoda gore DORT MERTEBE oynuyor (asagida).
ODE_METHODS: tuple[str, ...] = ("DOP853", "RK45", "Radau", "LSODA", "BDF")


@dataclass(frozen=True)
class Trajectory:
    """Bir radyal periyot boyunca esit arali `phi`de ornek alinmis yorunge.

    `phi` [0, dphi] araliginda `n_points` esit arali deger; `u = 1/r` ve
    `r = 1/u` ayni izgarada. `phi[0] = 0.0` perihelden baslar.

    ESIT ARALI `PHI` KARELEMEDEN ALINAMAZ ve bu ODE'nin buradaki gerekcesinin
    ASIL yarisidir: kareleme `phi(u)` uretir, yani TERS FONKSIYONU; esit arali
    `phi` istemek her nokta icin ayri bir kok bulma ya da ayri bir integral
    demektir. `viz/grav_frames.py` bu yuzden ODE'ye baglanir.
    """

    phi: "np.ndarray"
    u: "np.ndarray"
    r: "np.ndarray"


def _ode_cekirdek(*, theory: Theory, M: float, p: float, e: float,
                  rtol: float, atol: float, method: str, dense: bool):
    """Bir radyal periyodu ODE ile supuren cekirdek. `(dphi, sol)` doner.

    DENKLEM Schwarzschild ayarindadir ve `radial_w`nin ta kendisinin turevidir:

        u'' + u = M/L^2 + lam*M*u^2,
        M/L^2 = (S - a*(S^2 - P))/2,  a = 2*lam*M/3, S = u1+u2, P = u1*u2

    (yani `(u-u1)(u2-u)*w(u)` acilimindaki `u` katsayisinin yarisi -- ayni
    cekirdegin iki yazilisi; kareleme ile ODE arasindaki tek ORTAK govde bu
    iliskidir ve o yuzden ikisi CAPRAZ KONTROL DEGILDIR, spec 7.3).

    `t0 = 0` VE KUSUR NEREDEYDI -- OLCULDU, VE SPEC 4.4'TEN DAHA DAR:
    spec "`t0 = 0` ZORUNLU" diyor ve tabanin `1.0001e-9 * p` ciktigini
    yaziyor. Kusur `t0`nin DEGERINDE degil, supurulen acinin `t_olay - t0`
    yerine `t_olay` diye OKUNMASINDAYDI. Olculdu (BU dosyanin penceresinde --
    DOP853, rtol 1e-13, atol 1e-15, lam = 0, tek basamak; `t0` sevk edilen
    imzada bir arguman OLMADIGI icin olcum bu govdenin bir kopyasiyla
    yapildi):

        p     t0      dphi = t_olay - t0     dphi = t_olay  (KUSURLU)
        1e3   0       +1.135092e-09          +1.135092e-09
        1e3   1e-9    +1.137757e-09          +1.001138e-06   (= 1e-9*p + taban)
        1e4   0       +1.317702e-07          +1.317702e-07
        1e4   1e-9    +1.317702e-07          +1.013177e-05   (= 1e-9*p + taban)

    Yani cikarma yapilinca `t0` PRATIK OLARAK ETKISIZDIR (p = 1e3'te dorduncu
    hanede oynuyor, p = 1e4'te BIT BIT ayni), yapilmayinca `1e-9*p` tam olarak
    geri geliyor. Spec 4.4'un "5 800 kat" duzelmesi de aslinda bu sutunun
    duzelmesidir: dugme `t0` degil, CIKARMADIR. Sevk edilen surum `t0 = 0`
    KULLANIR *ve* cikarmayi yapar -- iki onlem birbirinin yedegidir.

    `t0`DAKI SAHTE OLAY: baslangic perihelde (`v = 0`) oldugu icin `solve_ivp`
    olay islevini `t0`da TAM 0.0 gorur ve `direction = -1` kosulunu orada da
    saglar. Olculdu, sevk edilen cagride: `t_events[0]` =
    `[0.0, 6.283248140523368]`, yani BIRINCI kok sahtedir. Ilk taslagim bu
    yuzden `t0 = 1e-9` yaziyordu; dogru cozum koku SUZMEKTIR.

    SUZGECIN ESIGI `pi`DIR AMA SINIRI `0`DAN AYIRT EDILEMEZ -- VE BU
    MUTASYONLA OLCULDU, VARSAYILMADI. `if t > math.pi` yerine `if t > 0.0`
    yazan mutant 237 testin HEPSINDEN sag cikiyor, cunku `t0 = 0`da sahte kok
    TAM `0.0`dir ve `0.0 > 0.0` zaten YANLIS. Yani bu iki esik sevk edilen
    pencerede DAVRANIS OLARAK OZDESTIR; `pi` esigi bir davranis farki icin
    degil, (a) `t0 != 0` alan bir kopyada da dogru kalsin diye, (b) cozucu
    gurultusunun uretebilecegi kucuk sahte kokleri kapsam disi biraksin diye
    secilmistir. `t >= 0.0` yazan mutant ise OLUR: sahte kok alinir, `dphi`
    sifira duser ve asagidaki kapi atar.

    "SON koku al" suzgeci OLCULEREK ELENDI: RK45 ve LSODA'da olay islevi
    `v ~ 0` civarinda gurultuye girip 4*pi'ye yakin sahte kokler uretiyor ve
    son kok `(4*pi - 2*pi)*p` = 2.513274e+05 gibi bir sayi donduruyordu --
    SESSIZCE. `pi < dphi < 3*pi` kapisi o sessiz yolu da kapatir.
    """
    u1, u2 = _donus_noktalari(p, e)
    lam = theory.lam
    a = 2.0 * lam * M / 3.0
    S = u1 + u2
    P = u1 * u2
    M_over_L2 = 0.5 * (S - a * (S * S - P))
    lamM = lam * M

    def rhs(phi, y):
        u, v = y
        return (v, M_over_L2 + lamM * u * u - u)

    def donus(phi, y):
        return y[1]
    donus.direction = -1.0

    sol = solve_ivp(
        rhs, (0.0, 4.0 * math.pi), (u2, 0.0), method=method,
        rtol=rtol, atol=atol, events=donus, dense_output=dense,
    )
    if not sol.success:
        raise InvalidInputError(
            f"ODE cozucusu basarisiz: {sol.message!r} "
            f"(method={method!r}, rtol={rtol!r}, atol={atol!r}, p={p!r})"
        )
    kokler = [float(t) for t in sol.t_events[0] if t > math.pi]
    if not kokler:
        raise InvalidInputError(
            f"radyal periyot bulunamadi (method={method!r}, p={p!r}, e={e!r}): "
            f"phi > pi olan donus noktasi yok. Bulunan kokler: "
            f"{list(sol.t_events[0])!r}"
        )
    dphi = kokler[0]
    if not (math.pi < dphi < 3.0 * math.pi):
        raise InvalidInputError(
            f"supurulen aci penceresi disinda: dphi = {dphi!r} "
            f"(method={method!r}, p={p!r}). Bagli bir yorungede bir radyal "
            "periyot 2*pi civarindadir; bu deger olay suzgecinin sahte bir "
            "kok yakaladigini soyler."
        )
    return dphi, sol


def run_trajectory(*, theory: Theory = Theory(), M: float, p: float, e: float,
                   n_points: int = 2000, rtol: float = ODE_RTOL,
                   atol: float = ODE_ATOL, method: str = "DOP853") -> Trajectory:
    """ODE yolu -- YORUNGE MOTORU, CIPA YOLU DEGIL.

    SINIF: CIPA DEGIL. Bu fonksiyon bir sayiyi belgelemez, bir egri uretir;
    belgelenen sayi `perihelion_coefficient` (kareleme) ile gelir.

    IHTIYAC OLCULDU, VE PLANIN ORANI URETILEMEDI. Plan "2000 nokta ODE
    1.35 ms, kareleme 19.44 ms (14.4 kat)" diyor. Bu depoda, ayni makinede:

        ODE + dense_output, 2000 nokta          2.482 ms
        BIRIKIMLI kareleme, 2000 nokta         12.026 ms   ( 4.8 kat)
        NOKTA BASINA ayri `quad`, 2000 nokta  207.568 ms   (83.6 kat)

    Yani "kat" sayisi tamamen karelemenin NASIL yazildigina bagli ve tek bir
    oran yazmak yaniltici olur; ucu de burada. ASIL gerekce zaman DEGILDIR:
    kareleme `phi(u)` verir, yani TERS FONKSIYONU, ve esit arali `phi`
    istendiginde her nokta ayri bir kok bulma isidir. `dense_output` bunu
    bedavaya verir.

    `n_points` ORNEKLEME YOGUNLUGUDUR, DOGRULUK DUGMESI DEGIL: cozucu adimini
    `rtol`/`atol` secer, `n_points` yalnizca `dense_output`tan kac nokta
    okundugunu belirler. Ikisi karistirilirsa "cozunurlugu artirdim ama sonuc
    degismedi" seklinde sahte bir sasirma dogar.
    """
    if not isinstance(n_points, int) or n_points < 2:
        raise InvalidInputError(
            f"n_points en az 2 olmali (tamsayi), verilen: {n_points!r}. "
            "n_points bir ORNEKLEME yogunlugudur, dogruluk dugmesi degil; "
            "dogrulugu `rtol`/`atol` belirler."
        )
    _ode_kapi(M=M, p=p, e=e, rtol=rtol, atol=atol, method=method)
    dphi, sol = _ode_cekirdek(theory=theory, M=M, p=p, e=e, rtol=rtol,
                              atol=atol, method=method, dense=True)
    phi = np.linspace(0.0, dphi, n_points)
    u = sol.sol(phi)[0]
    return Trajectory(phi=phi, u=u, r=1.0 / u)


def perihelion_coefficient_ode(
    *,
    theory: Theory = Theory(),
    M: float,
    e: float,
    p: float | None = None,
    richardson: tuple[float, float] | None = DEFAULT_ODE_RICHARDSON,
    rtol: float = ODE_RTOL,
    atol: float = ODE_ATOL,
    method: str = "DOP853",
) -> float:
    """Perihel katsayisi -- ODE BORUSUNDAN. CIPA YOLU DEGIL.

    SINIF: CIPA DEGIL, ODE BORUSUNUN KENDI OLCULEBILIR CIKTISI. Cipa
    `perihelion_coefficient`tir (kareleme, 10.01 durust hane); bu fonksiyon
    ODE'nin KENDI hata cubugunun okunabilmesi icin vardir -- spec 4.3'un
    karisik boru yasagi, "ODE'nin gurultusunu karelemenin sinyaline bolme"
    yasagidir ve ancak ODE'nin KENDI sinyali ve KENDI tabani varsa
    uygulanabilir.

    OLCULEN (GR, M = 1, e = 0.3, DOP853, rtol 1e-13, atol 1e-15,
    Richardson (1e4, 3e4)):

        C_ode = 18.8495564061          6pi = 18.84955592153876
        fark  = -4.845e-07  ->  7.59 DURUST HANE
        taban = 1.9218e-06,  S/G = 9.8082e+06,  6/6 basamak canli
        taban/gercek hata = 3.966

    Yani ODE tabani gercek hatayi 4 KAT BUYUK gosterir: bu pencerede hata
    cubugu MUHAFAZAKARDIR (karelemede oran 0.861 idi, yani hafif iyimser).
    Komsu pencerelerde kalibrasyon 1.00'e oturuyor (rtol 1e-12 & (1e4,3e4):
    1.065; rtol 1e-13 & (3e4,1e5): 1.006).

    METOT BIR PENCERE DUGMESIDIR VE SPEC 4.4'UN SATIRI BURADA DUSTU. Spec
    "taban DOP853'e ozgu degil (RK45, Radau, LSODA ayni mertebede)" diyor;
    olculdu (Richardson (1e4,3e4), rtol 1e-13, ayni suzgec):

        metot     durust hane   taban        sure
        DOP853        7.59      1.9218e-06     4.2 ms
        RK45          6.83      1.3768e-06    21.4 ms
        Radau         7.12      3.2953e-09   247.4 ms
        LSODA         5.84      2.5587e-05     3.7 ms
        BDF           5.75      3.2038e-05    61.7 ms

    Taban 3.2953e-09 ile 3.2038e-05 arasinda, yani 9722 KAT oynuyor -- DORT
    MERTEBE. "Ayni mertebede" cumlesi bu boruda YANLIS. `method` bu yuzden
    `PIPELINE_KEYS`e girer: etiket hangi integratorun kullanildigini TASIR.

    `p` ile `richardson` iliskisi `perihelion_coefficient`taki ile AYNIDIR ve
    ayni gerekceyle YASAKLAYICIDIR.
    """
    _ode_kapi(M=M, p=p, e=e, rtol=rtol, atol=atol, method=method,
              richardson=richardson)
    if richardson is None:
        return _C_ode_tek(theory=theory, M=M, p=float(p), e=e, rtol=rtol,
                          atol=atol, method=method)
    alt, ust = richardson
    f1 = _C_ode_tek(theory=theory, M=M, p=alt, e=e, rtol=rtol, atol=atol,
                    method=method)
    f2 = _C_ode_tek(theory=theory, M=M, p=ust, e=e, rtol=rtol, atol=atol,
                    method=method)
    h1, h2 = 1.0 / alt, 1.0 / ust
    return (f1 * h2 - f2 * h1) / (h2 - h1)


def _C_ode_tek(*, theory: Theory, M: float, p: float, e: float, rtol: float,
               atol: float, method: str) -> float:
    dphi, _ = _ode_cekirdek(theory=theory, M=M, p=p, e=e, rtol=rtol,
                            atol=atol, method=method, dense=False)
    return (dphi - 2.0 * math.pi) * p / M


def _ode_kapi(*, M: float, p: float | None, e: float, rtol: float,
              atol: float, method: str,
              richardson: tuple[float, float] | None = None) -> None:
    """ODE borusunun girdi kapilari."""
    if method not in ODE_METHODS:
        raise InvalidInputError(
            f"bilinmeyen integrator: {method!r}. Sevk edilenler: {ODE_METHODS}. "
            "Integrator bir PENCERE DUGMESIDIR ve hukme girer: gurultu tabani "
            "DOP853'te 1.9218e-06, Radau'da 3.2953e-09, BDF'te 3.2038e-05 "
            "(7767 kat)."
        )
    if not (M > 0.0) or not math.isfinite(M):
        raise InvalidInputError(f"M pozitif ve sonlu olmali, verilen: {M!r}.")
    if not math.isfinite(e) or not (0.0 < e < 1.0):
        raise InvalidInputError(
            f"e BAGLI bir yorungenin dismerkezligi olmali (0 < e < 1), "
            f"verilen: {e!r}."
        )
    for ad, deger in (("rtol", rtol), ("atol", atol)):
        if not math.isfinite(deger) or not (deger > 0.0):
            raise InvalidInputError(
                f"{ad} pozitif ve sonlu olmali, verilen: {deger!r}"
            )
    if richardson is None:
        if p is None:
            raise InvalidInputError(
                "basamak tanimsiz: `richardson=None` verildiginde tek basamak "
                "`p` ZORUNLUDUR."
            )
    elif p is not None:
        raise InvalidInputError(
            f"`p` ile `richardson` AYNI SEYI belirler, ikisi birden verilemez "
            f"(verilen p={p!r}, richardson={richardson!r})."
        )
    if richardson is not None:
        if len(richardson) != 2:
            raise InvalidInputError(
                f"richardson bir (alt, ust) CIFTI olmali, verilen: {richardson!r}"
            )
        alt, ust = richardson
        for ad, deger in (("alt", alt), ("ust", ust)):
            if not math.isfinite(deger) or not (deger > 0.0):
                raise InvalidInputError(
                    f"richardson.{ad} pozitif ve sonlu olmali, verilen: {deger!r}"
                )
        if alt == ust:
            raise InvalidInputError(
                f"richardson basamaklari AYRI olmali, verilen: {richardson!r}."
            )
    elif p is not None and (not math.isfinite(p) or not (p > 0.0)):
        raise InvalidInputError(f"p pozitif ve sonlu olmali, verilen: {p!r}")


#: ODE BORUSUNUN KENDI ETIKETI. Karelemenin `"kareleme"`siyle KARISTIRILAMAZ
#: -- karisik boru yasaginin kaydi tam olarak budur: ODE'nin tabani
#: (1.9218e-06) karelemenin tabaniyla (1.594187e-09) ayni etiketi tasisaydi
#: bir tablo sinyali birinden, gurultuyu otekinden alabilirdi.
perihelion_coefficient_ode.PIPELINE = "ode"  # type: ignore[attr-defined]
perihelion_coefficient_ode.NEWTON_PROBES = _newton_probes()  # type: ignore[attr-defined]

#: `method` BORUYU SECER (yukaridaki dort mertebelik taban tablosu), `gauge`
#: ise BU BORUDA YOKTUR: ODE yalnizca Schwarzschild ayarinda kosar.
perihelion_coefficient_ode.PIPELINE_KEYS = ("method",)  # type: ignore[attr-defined]
