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

from scipy.integrate import quad

from qlab.engine.grav.metric import (
    Theory,
    closed_form_coefficient,
    isotropic_orbit,
    isotropic_w,
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


#: SEVK EDILEN IKI AYAR. "Ayar" burada bir KOORDINAT SECIMIDIR: ayni fizik,
#: baska bir `r` tanimi ve baska bir metrik yazilisi. Birinci mertebe katsayisi
#: `C` ayardan BAGIMSIZDIR (olculdu, `tests/grav/test_perihel.py`); ikinci
#: mertebe DEGILDIR (`grav_obs.second_order_coefficient` dokstringi).
GAUGES: tuple[str, ...] = ("schwarzschild", "isotropic")


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
