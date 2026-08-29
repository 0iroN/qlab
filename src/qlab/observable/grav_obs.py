"""Yercekimi gozlemlenebilirleri -- ve HER SAYININ KENDI HATA CUBUGU.

Bu dosyanin ilk parcasi `newton_noise_floor`dur: spec 4.3'un "fizigi kapat,
ayni boruyu kostur" makinesi. Kural pazarlik edilmez ve KARISIK BORU YASAKTIR
-- bir cozucunun gurultu tabani OTEKI cozucunun kesme hatasina bolunemez.

BU MAKINE PLANDA YAZILDIGI HALIYLE DEJENEREYDI ve bu OLCULDU. Plan tabani
`lambda = 0` NOKTASINDAN okuyordu; ama karelemede `lam = 0` iken `w == 1`
olur, integrand SABIT 1.0'a duser, `quad` sabiti tam integre eder ve
`dphi - 2*pi` BIT BIT 0.0 cikar. On yedi ayri dugme ayarinda (yedi `p`
1e2...1e12, dort Richardson cifti, alti (e, M) noktasi) donen deger
DEGISMEDI: hepsinde tam 0.0. Bir sayi hicbir dugmeye tepki vermiyorsa
o bir OLCUM DEGIL BIR SABITTIR; ve fizik icermeyen girdide gecen bir kapi
kapi degildir. Yani plan hali, deponun kendi olcutuyle bir TAUTOLOJIYDI.

SEVK EDILEN ONARIM: fizigi kapatmak bir NOKTA degil bir LIMITTIR. Boru,
kesme hatasinin fizikle birlikte kayboldugu ama ARITMETIGIN CALISMAYA DEVAM
ETTIGI bir merdivende kosturulur (`engine.grav.geodesic.NEWTON_PROBE_LAMBDAS`,
esiklerinin ikisi de olculdu). Onarim SU UC OLCUMLE dogrulandi -- cunku bir
duzeltmenin kendisi de olculmemis bir iddiadir:

  1. TABAN KIPIRDIYOR. Richardson cifti 10 kat buyutuldugunde taban da tam
     10.00 kat buyuyor (uc bagimsiz basamakta 10.00 / 10.00 / 10.00). Ayni
     bozulmada `lambda = 0` tabani 0.0'da KALIYOR.
  2. TABANIN KAPALI-FORM ONGORUSU VAR ve tutuyor. Taban `dphi`nin ulp
     kuantumunun Richardson agirliklariyla buyutulmus halidir:
     `ulp(2*pi) * (|w1|*alt + |w2|*ust)` = 1.383021e-09; olculen 1.594187e-09,
     oran 1.153. Bu ongoru KARELEMEDEN BAGIMSIZ bir yoldan gelir.
  3. TABAN GERCEK HATAYI ONGORUYOR. Sevk edilen pencerede taban 1.594187e-09,
     GR'nin `6*pi`den gercek sapmasi 1.852140e-09 -- oran 0.8607. Bes PPN
     noktasinda ongorulen durust hane ile olculen durust hane arasindaki en
     buyuk fark 0.60 hane (GR'de 0.07).

DURUST NITELIK -- VE BU BIR KUSUR BEYANIDIR: taban YALNIZ ARITMETIGI olcer.
Kesme hatasi fizikle birlikte kapandigi icin, "fizigi kapat" sinavi kesme
hatasini YAPISAL OLARAK goremez. Olculdu: tek basamakli (Richardson kapali,
p = 3e5) pencerede gercek hata 2.842e-04, taban 2.453e-10 -- taban gercek
hatanin 1/1 158 000'i. Yani S/G ancak YUVARLAMA SINIRLI bir pencerede bir
hata cubugudur; kesme sinirli bir pencerede sessizce iyimserdir. Sevk edilen
Richardson penceresi yuvarlama sinirlidir (taban/hata = 0.861) ve panel
pencereyi bu yuzden ADIYLA yazar.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq, fsolve

from qlab.engine.grav.geodesic import (
    DEFAULT_NULL_RICHARDSON,
    DEFLECTION_PROBE_EPSILONS,
    deflection_angle,
    perihelion_coefficient,
)
from qlab.engine.grav.metric import (
    Theory,
    closed_form_coefficient,
    deflection_closed_form,
    evaluate_poly,
    null_potentials,
)
from qlab.errors import InvalidInputError

#: `apsidal_exponent`in sevk edilen `n` izgarasi. `n = -1` YOK (log
#: potansiyelde `r^(n+1)/(n+1)` 0/0'dir) ve `n <= -3` YOK (orada dairesel
#: yorunge kararsizdir, `3+n <= 0`).
APSIDAL_GRID: tuple[float, ...] = (-2.4, -2.2, -1.8, -1.6, -1.4, -1.2, -0.8,
                                   -0.4, 0.0)

#: Hukmun UC degeri. `DEJENERE`, `observable/em_obs.VERDICTS`in
#: `YARGILANAMAZ`iyla ve Modul 4'un `correlation_length -> None` deseniyle
#: AYNI STATUDEDIR: yargilayamadigini yargilamis gibi gosterme.
#:
#:   SAGLAM   -- taban olculdu ve sinyal tabani geciyor (`ratio >= 1`).
#:   CURUK    -- taban olculdu ama sinyal tabanin ALTINDA (`ratio < 1`);
#:               raporlanan sayi gurultunun icinde, panel bunu SOYLER.
#:   DEJENERE -- taban OLCULEMEDI: fizigi kapatma merdiveninin her basamagi
#:               tam 0.0 dondu, yani boru o pencerede hicbir aritmetik is
#:               yapmiyor. `ratio` burada `inf` DEGIL `nan`dir -- "sonsuz
#:               hassasiyet" bir yalan olurdu, dogru cevap "bilinmiyor"dur.
VERDICTS: tuple[str, ...] = ("SAGLAM", "CURUK", "DEJENERE")


@dataclass(frozen=True)
class NoiseFloor:
    """Bir fizik sayisinin KENDI BORUSUNDAN okunmus hata cubugu.

    `value`    : taban -- fizik kapaliyken borudan cikan en buyuk mutlak
                 sapma. Boru dejenere ise tam 0.0.
    `signal`   : ayni borunun ayni pencerede fizik ACIKKEN verdigi degerin
                 mutlak buyuklugu.
    `ratio`    : `signal / value`; dejenere boruda `nan`.
    `pipeline` : hangi boru VE taban hangi merdivenden okundu. Bu alan
                 KARISIK BORU YASAGININ kaydidir: bir tablo sinyali bir
                 borudan, tabani otekinden alamaz cunku ikisi de burada
                 yazilidir.
    `verdict`  : `VERDICTS`ten biri.
    """

    value: float
    signal: float
    ratio: float
    pipeline: str
    verdict: str


def _boru_secici_etiketi(fn: Callable[..., float], kw: dict[str, Any]) -> str:
    """`fn.PIPELINE_KEYS`teki argumanlarin DEGERLERINI etiket parcasina cevirir.

    Beyan etmeyen borular icin bos dizge doner, yani etiket HARFI HARFINE
    eskisi gibi kalir -- bu, `PIPELINE_KEYS`in var olan borulari bozmadan
    eklenebilmesinin kosuludur ve testte kilitlidir.
    """
    anahtarlar = getattr(fn, "PIPELINE_KEYS", ())
    if not anahtarlar:
        return ""
    imza = inspect.signature(fn).parameters
    parcalar = []
    for ad in anahtarlar:
        if ad in kw:
            deger = kw[ad]
        elif ad in imza and imza[ad].default is not inspect.Parameter.empty:
            deger = imza[ad].default
        else:
            raise InvalidInputError(
                f"{getattr(fn, 'PIPELINE', fn)!r} borusu {ad!r}i boru secici "
                "ilan ediyor ama ne cagride verilmis ne de imzada varsayilani "
                "var; etiket boruyu adlandiramaz."
            )
        parcalar.append(f"{ad}={deger}")
    return " | " + ", ".join(parcalar)


def newton_noise_floor(fn: Callable[..., float], **kw: Any) -> NoiseFloor:
    """Fizigi KAPAT, AYNI boruyu kostur -- ve tabanin bir sey olctugunu OLC.

    `fn` fizigi kapatma noktalarini KENDISI beyan eder; cagiran veremez.
    Beyan iki alandadir ve ikisi de ZORUNLUDUR:

        fn.PIPELINE       -- borunun adi (`fn.__name__` DEGIL: ad degisirse
                             etiket degismemeli, ve iki fonksiyon ayni boruyu
                             kullanabilmelidir)
        fn.NEWTON_PROBES  -- `(kwargs, TAM deger)` ciftleri; `kwargs` `fn`e
                             fizigi kapatan argumanlari verir
        fn.PIPELINE_KEYS  -- (istege bagli) HANGI ARGUMANLARIN boruyu SECTIGI.
                             Degerleri etikete islenir.

    `PIPELINE_KEYS` TASK 3'TE OLCULEREK EKLENDI, TASARLANMADI. Task 2'de
    `perihelion_coefficient` TEK bir boruydu ve "kareleme" onu tam
    adlandiriyordu. Task 3 ayni fonksiyona `gauge` anahtar argumanini ekledi ve
    o an ETIKET YALAN SOYLEMEYE BASLADI: iki AYRI govde (Schwarzschild `w`si ve
    izotropik `R/A`si, ortak dugmesi olmayan iki mekanizma) ayni `"kareleme"`
    etiketini donduruyordu. Olculdu, ayni cagrida yalniz `gauge` degistirilerek:
    taban 1.594187e-09 <-> 2.346903e-09, merdiven olum esigi p/M = 1e13 <-> 1e14.
    Yani KARISIK BORU YASAGININ KAYDI, tam da bu modulun iki-ayarli capraz
    kontrolu sevk edildigi anda ayirt edemez hale gelmisti. Anahtar degerleri
    artik etikete yazilir; `PIPELINE_KEYS` beyan etmeyen borular (tek govdeli
    olanlar) icin etiket HARFI HARFINE eskisi gibi kalir.

    Varsayilanlar `fn`in IMZASINDAN okunur, ikinci bir yere yazilmaz -- iki alan
    ayni seyi belirlerse sessiz bir kazanan dogar (Task 1'in olcumu).

    CAGIRAN BU ARGUMANLARDAN HICBIRINI VEREMEZ. Verebilseydi "fizigi kapatma"
    cagirana birakilmis bir dugme olurdu ve makine kendi sinavini secebilirdi;
    boyle bir cagri `InvalidInputError` ile reddedilir.

    TABAN = merdivenin CANLI basamaklarinda `|fn(...) - TAM|`in en buyugu.
    Bir basamak BIT BIT 0.0 donduruyorsa OLUDUR ve elenir: kayan noktali bir
    boru tam 0.0 donduruyorsa aritmetik yapmamistir. (Karelemede bu, `w`nin
    yeniden tam 1.0'a yuvarlanmasidir; olculdu, `richardson = (p, 10p/3)`:
    `p/M` = 1e5'te 6/6 canli, 1e8'de 5/6, 1e10'da 3/6, 1e12'de 1/6, 1e13'te
    0/6.) Hicbir canli basamak kalmazsa hukum `DEJENERE`dir. Eleme SART:
    olu bir basamagin "artigi" `|0 - TAM| = |TAM|` olurdu, yani sinyalin
    kendisi taban diye raporlanirdi.

    OLCULEN (kareleme, GR sinyali, e = 0.3, M = 1, Richardson (3e5, 1e6),
    quad 1e-14/1e-13/400): taban **1.594187e-09**, sinyal **18.84955591968662**,
    S/G **1.1824e+10**, 6/6 basamak canli, hukum `SAGLAM`.

    `CURUK` KARELEME BORUSUNDA ERISILEMEZ VE SEBEBI YAPISALDIR -- olculdu,
    varsayilmadi. Karelemede sinyalin gurultuye gomulmesi (`lam*M/p < eps`)
    ile merdivenin olmesi (`w` tam 1.0'a yuvarlanmasi) AYNI kosuldur, cunku
    `w - 1` ile `dphi - 2*pi` ayni kucuk buyuklugun (`lam*M/p`) iki yuzudur.
    `p/M`yi 1e5'ten 1e17'ye tarayan olcumde hukum SAGLAM'dan dogrudan
    DEJENERE'ye geciyor (1e12'de S/G = 2.6671e+03 SAGLAM, 1e13'te 0/6 canli
    DEJENERE); arada CURUK cikan hicbir nokta YOK.

    >>> TASK 4'TE DUSEN IDDIA -- BU DOKSTRINGIN KENDI CUMLESI. Buraya Task
    2'de "`CURUK` dali baska borular icin duruyor; N-body tarafinda `m2 = 0`
    tabaninin sinyali `~m2^2` gittigi icin orada bir KESISIM vardir (spec
    4.3)" yazilmisti. Task 4 ODE borusunu getirdi ve o cumle ILK KEZ
    SINANABILIR hale geldi: SINANDI VE DUSTU. `perihelion_coefficient_ode`
    borusunda `CURUK` SEVK EDILEN API ILE erisilebilir -- olculdu, iki ayri
    nokta:

        rtol=1e-5, atol=1e-7, richardson=(1e4,3e4):
            taban 5.5342e+02, sinyal 535.542614, S/G 0.967704  -> CURUK
        rtol=1e-6, atol=1e-8, richardson=(1e5,3e5):
            taban 5.4595e+03, sinyal 5441.593263, S/G 0.996726 -> CURUK

    Yani `CURUK`un FIZIK kapsamasi N-body'yi (Task 5) BEKLEMIYOR; bu modulde
    zaten var. AMA IDDIANIN RUHU AYAKTA, VE O DA OLCULDU: ODE'de `CURUK`
    yalnizca S/G = 1'in cok ince bir komsulugunda cikiyor ve bu YAPISALDIR --
    integrator hatasi baskin oldugunda ayni hata hem `signal`i (|C| ~ |hata|)
    hem tabani belirler, yani oran 1'e RAPTOLUNUR. Olculen: `p` buyutulerek
    (rtol 1e-13, (1e8,3e8) ve (1e10,3e10)) S/G 1.181868 ve 1.000008 cikiyor,
    yani 1'in USTUNDEN yaklasiyor. Bu boruda hicbir zaman GENIS bir CURUK
    bandi yok. Ve `DEJENERE` ODE'de HIC gorulmedi (her pencerede 6/6 canli),
    yani iki boru birbirinin tam tersi: kareleme `DEJENERE` olabilir ama
    `CURUK` olamaz, ODE `CURUK` olabilir ama `DEJENERE` olamaz. Genis ve
    guvenli bir `CURUK` bandi hala N-body'den (`m2 = 0`, sinyal `~m2^2`)
    bekleniyor -- ama artik "erisilemez" diye degil, "dar" diye.
    """
    etiket = getattr(fn, "PIPELINE", None)
    if not isinstance(etiket, str) or not etiket:
        raise InvalidInputError(
            f"{getattr(fn, '__name__', fn)!r} bir `PIPELINE` etiketi tasimiyor. "
            "Boru adi `fn.__name__`den OKUNMAZ: ad bir kod ayrintisidir, boru "
            "adi ise panelde yazilan bir beyandir."
        )
    probes = getattr(fn, "NEWTON_PROBES", None)
    if not probes:
        raise InvalidInputError(
            f"{etiket!r} borusu `NEWTON_PROBES` beyan etmiyor. Fizigi kapatma "
            "noktalarini CAGRILAN beyan eder; cagiran veremez."
        )
    for probe_kw, _tam in probes:
        cakisan = sorted(set(probe_kw) & set(kw))
        if cakisan:
            raise InvalidInputError(
                f"{cakisan} argumanlari `newton_noise_floor`a verilemez: "
                "fizigi kapatan noktayi CAGRILAN belirler. Cagiran verebilseydi "
                "'fizigi kapat' sinavi cagirana birakilmis bir dugme olurdu."
            )

    signal = abs(float(fn(**kw)))
    artiklar: list[float] = []
    for probe_kw, tam in probes:
        deger = float(fn(**{**kw, **probe_kw}))
        if deger == 0.0:
            continue          # OLU basamak: boru aritmetik yapmadi
        artiklar.append(abs(deger - tam))

    canli = len(artiklar)
    secici = _boru_secici_etiketi(fn, kw)
    boru = (f"{etiket}{secici} | fizik-kapali merdiven "
            f"{canli}/{len(probes)} canli")
    if canli == 0:
        return NoiseFloor(
            value=0.0, signal=signal, ratio=math.nan,
            pipeline=boru, verdict="DEJENERE",
        )
    value = max(artiklar)
    ratio = signal / value
    return NoiseFloor(
        value=value, signal=signal, ratio=ratio,
        pipeline=boru, verdict="CURUK" if ratio < 1.0 else "SAGLAM",
    )


def second_order_coefficient(*, theory: Theory = Theory(), M: float,
                             p: float, e: float) -> float:
    """Ikinci mertebe katsayisi `K`: `C(p) = 2*pi*lam + K*M/p + O((M/p)^2)`.

        K = (C(p) - 2*pi*lam) * p / M

    SINIF: FIZIK CIPASI. Hedefi `K = pi*lam^2*(3 + e^2/6)` -- spec 3.2, ve
    spec'in ilk hali olan `27*pi + 1.5*pi*e^2` bu ailenin `lam = 3` DILIMIDIR.

    UC AYRI PENCERE DUGMESI VAR VE UCU DE HUKME GIRER:

    (1) `p` BIR OPTIMUMDUR, VE PLANIN SAYISI BU PENCEREDE CIKMIYOR. Olculdu
        (Schwarzschild ayari, izgara lam in {0.5,1,2,3,4,6} x
        e in {0.3,0.7,0.95}, EN KOTU bagil fark):

            p = 1e4 -> 1.097e-03 | 3e4 -> 3.653e-04 | 1e5 -> 1.096e-04 |
            p = 3e5 -> 4.086e-05 | 1e6 -> 3.414e-04 | 3e6 -> 2.900e-03 |
            p = 1e7 -> 5.528e-02

        Optimum `p = 3e5`te ve orada en kotu fark 4.086e-05'tir. Planin
        6.7e-05'i BU IZGARADA VE BU PENCEREDE URETILEMEDI; 4.086e-05 Task 1'in
        yan olcumuyle de birebir ayni cikti.

        OPTIMUMUN YERI KAPALI FORMDAN ONGORULUYOR, ve ongoru KARELEMEDEN
        BAGIMSIZ bir yoldan gelir: `dphi - 2*pi`nin yuvarlama kuantumu
        `ulp(2*pi)` idi, `K` onu `(p/M)^2` ile buyutur, yani gurultu
        `ulp(2*pi)*(p/M)^2`dir. Izgaranin EN KOTU MUTLAK hatasi bu ongoruye
        boluunce (her `p`de kendi en kotu noktasinda):

            p = 1e4 -> 4.4e+06 | 3e4 -> 1.6e+05 | 1e5 -> 4.4e+03 |
            p = 3e5 -> 1.265   | 1e6 -> 0.930   | 3e6 -> 0.878   | 1e7 -> 1.540

        Yani `p <= 1e5` KESME SINIRLIDIR (oran binlerce), `p >= 3e5` YUVARLAMA
        SINIRLIDIR (oran ~1). Optimum tam bu gecisin uzerindedir; bu, Task 2'nin
        "kesme sinirli / yuvarlama sinirli pencere" ayriminin `K` tarafindaki
        karsiligidir ve orada oldugu gibi burada da gurultu tabanini bir hata
        cubugu YAPAN sey pencerenin YUVARLAMA SINIRLI olmasidir.

    (2) RICHARDSON KULLANILMAZ, VE SEBEBI ILK YAZDIGIMDAN BASKA CIKTI.
        Ilk gerekcem "disdegerlenmis `C`den okunsaydi artik `(p/M)^2` ile
        carpilip 166 buyuklugunde bir K verirdi" idi; OLCULDU VE YANLIS.
        Richardson `1/p`de dogrusal disdegerleme yapar, yani `K*M/p` terimini
        TAM OLARAK SILER. Yanlis yolun verdigi sayi (p = 3e5, GR):
        e = 0.3 -> -5.556419e-04 (dogrusu 85.24858543879077),
        e = 0.7 -> +6.100720e-05 (dogrusu 87.13363755319392).
        Yani sinyal BUYUMUYOR, TAMAMEN KAYBOLUYOR. Kural ayni kaliyor,
        gerekcesi tersine donuyor: `K` tam da Richardson'in SILDIGI terimdir.
        Bu yuzden govde `richardson=None` ile cagirir.

    (3) `AYAR` DUGMESI YOK, VE SEBEBI OLCULDU: `K` AYAR-BAGIMLIDIR.
        `C` (birinci mertebe) iki ayarda ayni sayidir; `K` DEGILDIR --
        GR, e = 0.3, p = 3e5: Schwarzschild 85.248585, izotropik 26.577757.
        Iki sebep birlikte: (a) `p` ile `e` birer KOORDINAT ETIKETIDIR ve iki
        ayarda ayni yorunge farkli (p, e) ile adlandirilir; (b) sevk edilen
        izotropik metrik 1PN'de KESILMISTIR (`metric.isotropic_potential`),
        yani ikinci mertebesi bir MODELLEME ARTIGIDIR, fizik degil. Izotropik
        K bir GURULTU DEGIL, tanimli bir sayidir (p = 1e4 ... 1e6 boyunca
        26.576587 -> 26.577775, bes hanede kararli) -- ama BASKA BIR SAYIDIR.

        SONUC, VE BU MODULUN CAPRAZ KONTROLUNUN SINIRIDIR: iki ayarli capraz
        kontrol BIRINCI MERTEBEYI belgeler, ikinci mertebeyi BELGELEMEZ.

    `2*pi*lam` cikarilmasi `closed_form_coefficient`tan okunur, burada YENIDEN
    yazilmaz. Bunun bedeli acikca beyan edilir: `K`nin dogrulugu kapali formun
    dogruluguna BAGLIDIR ve bu yuzden `K` testi `lam`i ELLE yazar (bir mutant
    iki tarafi birlikte kaydiramasin diye).
    """
    C_p = perihelion_coefficient(theory=theory, M=M, p=p, e=e, richardson=None)
    return (C_p - closed_form_coefficient(theory)) * (p / M)


# ---------------------------------------------------------------------------
# ISIK BUKULMESI -- `1 + gamma`
# ---------------------------------------------------------------------------

def deflection_ratio(
    *,
    theory: Theory = Theory(),
    M: float,
    richardson: tuple[float, float] = DEFAULT_NULL_RICHARDSON,
    gauge: str = "schwarzschild",
) -> float:
    """Isik bukulmesi orani `delta(gamma) / delta(gamma = 0)` -> `1 + gamma`.

    SINIF: FIZIK CIPASI. Sayi ancak bos jeodezik cozulerek cikar; statik bir
    koku yoktur. (Karsilastir: `critical_impact_parameter`in statik koku VAR
    ve o "kapali-form turev" rafindadir.)

    HER IKI YARI DA SAYISALDIR: pay `theory`nin `gamma`siyla, payda
    `Theory(gamma=0, beta=theory.beta)` ile AYNI boruda kosar. Newton yarisi
    icin `2*M/b` kapali formu KULLANILMAZ -- kullanilsaydi oran yari kapali
    form olurdu ve pencerenin ikinci mertebesi yalan soylerdi (asagida).

    AMA "IKI YARI" BIR CAPRAZ KONTROL DEGILDIR VE BU OLCULDU. Iki yari
    `b`yi, `M`i, `beta`yi, ayari, `quad` dugmelerini, `theta` ikamesini,
    donus noktasi arayicisini, deflasyonu ve Richardson ciftini PAYLASIR;
    ayrilan TEK sey `gamma`nin degeridir. Paylasilan govdeye sokulan bir
    bozulmanin oranda GORUNUP GORUNMEDIGI, bozulmanin `gamma`ya bagli olup
    olmamasina bakar -- uc bozulma olculdu (GR, M = 1, b = 1e5,
    Richardson (1e5,3e5); temiz artik |oran - 2| = 3.1125e-10):

        bozulma                       delta(1) kaymasi  delta(0) kaymasi  ORAN kaymasi
        u^3 katsayisi 2M -> 2.001M    5.0001e-04        1.0000e-03        9.9900e-04  GORUNUR
        theta ikamesi sin -> sin(1.0001t)  1.7808e-05   2.1458e-05        7.8533e-05  GORUNUR
        donus noktasi u0 -> 1.0000001*u0   1.0002e-07   1.0001e-07        1.1102e-10  GORUNMEZ

    Ucuncu satir bu cipanin KOR NOKTASIDIR: `gamma`dan BAGIMSIZ, CARPIMSAL bir
    bozulma iki yaride ayni bagil kaymayi yapar ve oranda SADELESIR -- kayma
    temiz artigin bile ALTINDA kalir. Onu goren sey oran degil, `delta`nin
    KENDISIDIR (`delta*b/M -> 4` ve ikinci mertebe `15*pi/4`), ve bu yuzden
    `tests/grav/test_bukulme.py` ikisini de kilitler.

    ILK YAZDIGIM AYIRT EDICI SINAV OLCULEREK ELENDI: "oran yalniz `M/b`ye
    baglidir, `delta` degildir; `M`i iki katina cikarip `b`yi de iki katina
    cikarmak ikisini ayirir" demistim. YANLIS -- `delta` bir ACIDIR, yani
    BOYUTSUZDUR ve `G = c = 1` birimlerinde tek boyutsuz birlesim `M/b`dir.
    Olculdu: `delta(M=1, b=1e5)` ile `delta(M=2, b=2e5)` BIT BIT ayni
    (4.000117813962589e-05), oran da oyle. Yani `M/b` olceklemesi ikisini
    AYIRMAZ; ayiran sey yalnizca `gamma`dan bagimsiz bir CARPANDIR ve o da
    ancak cekirdek disaridan bozularak gorulur (yukaridaki ucuncu satir).

    IKI AYAR DA BURADA ORTAK DUGME TASIR -- VE BU TASK 3'TEN FARKLIDIR.
    Perihelde Schwarzschild yolu `Theory`den yalniz `lam`i, izotropik yol
    yalniz `{gamma, beta}`yi okuyordu ve kesisim BOSTU. Bukulmede iki ayar da
    `gamma` VE `beta`yi okur (`metric.null_potentials`), yani kesisim
    `{gamma, beta}`dir. Dolayisiyla `1+gamma`nin iki ayarda ayni cikmasi bir
    CAPRAZ KONTROL DEGIL, bir AYAR-BAGIMSIZLIGI OLCUMUDUR; bu modulun gercek
    capraz kontrolu perihelde kalir.

    PENCERE -- `1/b`de RICHARDSON, ve TEK `b` PENCERESI YETMEZ:

        tek b = 1e5, Schwarzschild:  oran = 2.000027490288990  ->  4.86 hane
        Richardson (1e5, 3e5):       oran = 1.9999999996887523 ->  9.81 hane

    OLCULEN (GR, M = 1, Richardson (1e5, 3e5)):

        ayar             oran                     hane    taban        taban/hata
        schwarzschild    1.9999999996887523       9.81    3.651917e-10   1.173
        isotropic        1.999999999804698       10.01    8.436665e-11   0.432

    Iki ayarin farki 1.159457e-10. PPN ailesinde (gamma in
    {-0.5,0.5,1,1.5,2} x beta in {0,1,3}) en kotu |oran - (1+gamma)|
    Schwarzschild'de 4.3487e-09, izotropikte 2.5253e-09.

    PENCERENIN IKINCI MERTEBESI AYAR-BAGIMLIDIR -- VE PLANIN IKI SAYISI DA
    BURADA CURUDU. Olculen `(oran - 2)*b/M` (tek `b`, b -> buyuk limiti):

        Schwarzschild  2.749029   ->  `7*pi/8`  = 2.748894
        isotropic      1.570874   ->  `pi/2`    = 1.570796
        fark           1.178155   ->  `3*pi/8`  = 1.178097

    Planin FARKI (`3*pi/8`) DOGRU cikti; iki DEGERI (`15*pi/8` = 5.890486 ve
    `3*pi/2` = 4.712389) ise HER IKISI DE TAM `pi` KADAR BUYUK. Sebep
    olculebilir: `(oran-2)*b/M = c2(gamma)/2 - c2(0)` ve bu boruda
    `c2(0) = pi` (Newton yarisinin KENDI ikinci mertebesi). Planin sayilari
    `c2(0) = 0` ile, yani Newton yarisi kapali formdan alinmis bir boruyla
    uyusuyor -- oysa plan "her iki yari da sayisal" diyor. Iki cumle ayni
    anda dogru olamaz; sevk edilen boru "her iki yari da sayisal" olanidir.

    RICHARDSON ADIMLARINDAKI `M` CARPANI BIR BEYANDIR, BIR HESAP DEGIL --
    ve bu MUTASYONLA olculdu. Govde `h1, h2 = M/alt, M/ust` yazar cunku
    acilim parametresi `M/b`dir; ama `(f1*h2 - f2*h1)/(h2 - h1)` ifadesi iki
    adimin ORTAK bir carpaniyla olceklenmesine karsi DEGISMEZDIR (cebirsel:
    pay ve payda ayni `M` ile carpilir). Olculdu: `M/alt` yerine `1.0/alt`
    yazan mutant 237 testin HEPSINDEN sag cikiyor, ve bunun sebebi bir test
    acigi DEGIL, mutantin ESDEGER olmasidir -- ayni girdide bit bit ayni
    sayiyi donduruyor (M = 1, 2, 7.3 ile denendi). Satir okuyucuya `M/b`nin
    kucuk parametre oldugunu soylemek icin duruyor.

    SERTIFIKA ARALIK OLARAK VERILIR (tek `b`, Richardson YOK, bagil hata
    `|oran-2|/2`, Schwarzschild, GR):

        b/b_krit    1.001      2          20         100        1e4
        bagil hata  5.709e+00  2.460e-01  1.389e-02  2.671e-03  2.645e-05

    Planin tablosu (1.4e+01 / 4.5e-01 / 1.3e-02 / 2.3e-03) ust iki basamakta
    URETILEMEDI (2.4 ve 1.8 kat), alt iki basamakta usuyor.
    """
    if richardson is None:
        raise InvalidInputError(
            "bukulme orani TEK `b`de bir CIPA DEGILDIR: orada pencere kesme "
            "sinirlidir (b = 1e5'te 4.86 durust hane, Richardson (1e5,3e5) "
            "ile 9.81). `richardson` zorunludur; tek `b` sertifikasi icin "
            "`geodesic.deflection_angle` kullanin."
        )
    if len(richardson) != 2:
        raise InvalidInputError(
            f"richardson bir (alt, ust) CIFTI olmali, verilen: {richardson!r}"
        )
    alt, ust = richardson
    if alt == ust:
        raise InvalidInputError(
            f"richardson basamaklari AYRI olmali, verilen: {richardson!r}. "
            "Esit basamaklarda `1/b`de disdegerleme sifira bolmedir."
        )
    newton = Theory(gamma=0.0, beta=theory.beta)
    f1 = (deflection_angle(theory=theory, M=M, b=alt, gauge=gauge)
          / deflection_angle(theory=newton, M=M, b=alt, gauge=gauge))
    f2 = (deflection_angle(theory=theory, M=M, b=ust, gauge=gauge)
          / deflection_angle(theory=newton, M=M, b=ust, gauge=gauge))
    h1, h2 = M / alt, M / ust
    return (f1 * h2 - f2 * h1) / (h2 - h1)


def _deflection_probes() -> tuple[tuple[dict[str, Any], float], ...]:
    """`gamma -> -1` merdiveni: `(fizigi kapatan kwargs, TAM deger)` ciftleri.

    BUKULME BORUSUNUN "FIZIGI KAPAT" NOKTASI `M = 0` DEGILDIR ve bu olculdu:
    `M = 0`da `A = T = 1`, `Q(u) = u0 + u` olur, integrand BIT BIT 1.0'a
    duser ve `delta` TAM 0.0 cikar -- iki ayarda da. Yani o nokta tam olarak
    karelemedeki `lambda = 0` gibi bir SABITTIR, olcum degil. (Ustelik
    `M > 0` kapisi onu API'den de erisilemez kilar.)

    Sevk edilen nokta `1 + gamma -> 0` LIMITIDIR: kapali form `1 + gamma`
    oldugu icin `gamma = -1`de tam sifir beklenir, ama boru orada CALISMAYA
    DEVAM EDER (`M` hala acik, integrand hala 1'den farkli). Merdiven
    `gamma = -1 + eps`, eps = 1e-4 ... 1e-9.

    MERDIVEN BOYUNCA ARTIK EPS'TEN BAGIMSIZDIR ve bu, tabanin FIZIGI DEGIL
    ARITMETIGI olctugunun olcumudur -- sebebi de kapali formdan okunuyor:
    `delta(gamma) ~ 2*(1+gamma)*M/b` kucuk `(1+gamma)`da bir FARK olarak
    hesaplanir (`2*INT - pi`), mutlak yuvarlamasi sabit kalir, oran ise
    `(1+gamma)`ya bolununce artik sabitlenir. Olculen artiklar (M = 1):

        pencere       eps=1e-4    1e-5      1e-6      1e-7      1e-8      1e-9
        schw (1e5,3e5) 3.6519e-10 1.0239e-10 2.4597e-10 2.1037e-10 3.2671e-10 2.4509e-10
        schw (1e4,3e4) 2.4632e-08 2.4631e-08 2.4616e-08 2.4620e-08 2.4621e-08 2.4614e-08
        izo  (1e5,3e5) 4.8171e-11 4.0705e-11 8.4367e-11 4.0052e-11 7.6291e-11 1.6435e-11
        izo  (1e4,3e4) 5.1143e-09 5.1012e-09 5.1025e-09 5.0936e-09 5.0978e-09 5.1205e-09

    Yani (1e4,3e4) penceresinde merdiven ALTI DEKATTA 1.0007 kat oynuyor --
    neredeyse mukemmel bir plato; (1e5,3e5)'te 3.6 kat, cunku orada artik
    zaten ulp mertebesindedir ve isaret gurultusu goruluyor.

    TAM DEGER `metric.deflection_closed_form`tan okunur, burada YENIDEN
    yazilmaz: `1 + gamma`nin ikinci bir kopyasi ikinci bir sessiz kazanandir.
    """
    ciftler = []
    for eps in DEFLECTION_PROBE_EPSILONS:
        t = Theory(gamma=-1.0 + eps)
        ciftler.append(({"theory": t}, deflection_closed_form(t)))
    return tuple(ciftler)


#: Boru etiketi -- KARELEMEDEN DE ODE'DEN DE AYRI. Uc boru, uc etiket.
deflection_ratio.PIPELINE = "bukulme"  # type: ignore[attr-defined]
deflection_ratio.NEWTON_PROBES = _deflection_probes()  # type: ignore[attr-defined]

#: `gauge` BORUYU SECER: iki AYRI govde (Schwarzschild `A/T` ve izotropik
#: `B/b^2 - u^2 A`) ve tabanlari FARKLI (3.651917e-10 <-> 8.436665e-11).
#:
#: `richardson`, `M` VE `b` BURADA YOK, VE BU BIR OLCUTUN UYGULANMASIDIR:
#: `PIPELINE_KEYS` "tabani degistiren" degil "GOVDEYI degistiren" argumanlari
#: adlandirir. Olculdu: `richardson`i (1e4,3e4)'e cevirmek tabani 3.65e-10'dan
#: 2.46e-08'e, yani 67 KAT buyutuyor -- ama ayni satirlari kosturuyor. Ayni
#: sey ODE'de `rtol` icin de gecerli (taban 1240 kat oynuyor, govde ayni).
#: Task 3 `gauge`u eklerken bu olcutu YAZMAMISTI; Task 4 yaziyor, cunku ODE
#: `method` ile ilk kez GERCEKTEN farkli bir govde secen ikinci bir anahtar
#: getirdi.
deflection_ratio.PIPELINE_KEYS = ("gauge",)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# KAPALI-FORM TUREVLER -- SINIFI YAZILI
# ---------------------------------------------------------------------------
#
# SINIF: KAPALI-FORM TUREV, "fizik cipasi" DEGIL. Olcut deponun kendi olcutu:
# *"simulasyon bu sayi hakkinda YANLIS olabilir miydi?"* Asagidaki bes sayinin
# hicbirinde yorunge COZULMEZ; hepsi bir cebirsel kokun ya da bir NOKTADAKI
# dogrusallastirmanin geri okunmasidir. Bir sabiti geri okumak model cozmez.
#
# VE STATIK/DINAMIK ORANLARI OLCULDU -- ucunde de dinamik hem YAVAS hem DAHA
# AZ HASSAS:
#
#   sayi     statik                    dinamik                      oran
#   ISCO      4.68 us, hata 8.882e-16   189.3 ms, hata 2.333e-03    40418x
#   b_krit   13.44 us, hata 0.000e+00     6.50 ms, hata 2.395e-05      483x
#
# Planin 844x / 11180x sayilari bu makinede URETILEMEDI. Dinamik yollar SEVK
# EDILMIYOR (spec 7.5: "HAYIR -- yerine GUCLU ALAN"), yani bu iki sutun bu
# dosyanin bir kopyasiyla olculdu ve yontemleri burada yazili: dinamik ISCO
# dairesel yorungeyi 1e-8 bozup `phi` in [0, 500] boyunca izler ve esik
# asiminda kararsiz sayar (44 turda ikiye bolme); dinamik b_krit donus
# noktasinin varligini 400 noktali bir izgarada tarar ve `b`yi 44 turda
# ikiye boler. Ikisinin de hatasi YONTEMDEN gelir: dinamik ISCO'nun radyal
# salinim periyodu marjinal noktada IRAKSAR, yani sonlu bir `phi` penceresi
# esigi sistematik olarak kacirir.

def photon_sphere(*, theory: Theory = Theory(), M: float = 1.0) -> float:
    """Foton kuresinin yaricapi, `r/M` cinsinden. SINIF: KAPALI-FORM TUREV.

    Bos jeodezigin etkin potansiyeli `V(u) = u^2 * A(u)`dir ve foton kuresi
    onun ucnoktasidir: `dV/du = 0`. `A` `metric.null_potentials`tan okunur.

    OLCULEN (GR, M = 1): `r/M` = 3.0 (bit bit), `r/r_s` = 1.5 (bit bit);
    500 tekrarli iki kosuda 9.76 us. `r_s = 2*M`.

    YALNIZ `beta - gamma` FARKINA BAGLIDIR ve bu olculdu: (gamma, beta) =
    (1, 0) ile (2, 1) AYNI sayiyi verir (4.0). Sebep `A`nin ikinci mertebe
    katsayisinin `2*(beta-gamma)*M^2` olmasidir.

    `beta > gamma` ICIN FOTON KURESI YOKTUR -- olculdu, varsayilmadi:
    (gamma, beta) = (0, 1) ve (1, 3)'te `dV/du`nun (0, 1/(2M)] araliginda
    koku yok, cunku 1PN'de kesilmis `A` orada ufuk yapmiyor.
    """
    u = _foton_u(theory=theory, M=M)
    return 1.0 / (u * M)


def critical_impact_parameter(*, theory: Theory = Theory(),
                              M: float = 1.0) -> float:
    """Kritik carpisma parametresi `b_krit/M`. SINIF: KAPALI-FORM TUREV.

    `b_krit = 1 / sqrt(V(u_foton))` = `r_foton / sqrt(A(r_foton))`.

    OLCULEN (GR, M = 1): 5.196152422706632, `3*sqrt(3)`ten fark TAM
    0.000e+00 (bit bit esit); 500 tekrarli iki kosuda 10.69 ve 13.44 us. *(Planin taban blogu bu sayiyi
    `5.196152422706631` diye yaziyor -- SON BITTE, 1 ulp DUSUK; bu depoda
    hem olculen deger hem `3*math.sqrt(3)` `...632`dir.)*

    `photon_sphere` ile AYNI kapiya tabidir: `beta > gamma`da yoktur.
    """
    u = _foton_u(theory=theory, M=M)
    A = evaluate_poly(
        null_potentials(theory=theory, M=M, gauge="schwarzschild").time_poly, u
    )
    if not (A > 0.0):
        raise InvalidInputError(
            f"foton kuresinde A = {A!r} <= 0; b_krit tanimsiz "
            f"(gamma={theory.gamma!r}, beta={theory.beta!r})."
        )
    return 1.0 / (u * math.sqrt(A) * M)


def isco_radius(*, theory: Theory = Theory(), M: float = 1.0) -> float:
    """En ic kararli dairesel yorunge, `r/M`. SINIF: KAPALI-FORM TUREV.

    Kutleli parcacigin dairesel yorunge kosulu `W(u) = (1 + L^2 u^2) A(u)`
    fonksiyonunun ucnoktasidir; marjinal kararlilik `d2W/du2 = 0` demektir.
    Iki denklem de `L^2`de DOGRUSALDIR, yani `L^2` elenerek tek degiskenli
    bir denklem kalir:

        f(u) = Q1(u)*P2(u) - Q2(u)*P1(u) = 0,
        dW/du   = L^2*P1 + Q1,   d2W/du2 = L^2*P2 + Q2

    OLCULEN (GR, M = 1): `r/M` = 5.999999999999999 (6'dan 8.882e-16),
    `r/r_s` = 2.9999999999999996; 500 tekrarli iki kosuda 4.68 ve 4.90 us;
    `r_ISCO/r_foton` = 1.9999999999999998.

    YALNIZ `beta - gamma`YA BAGLIDIR (foton kuresi gibi): (1,0) ve (2,1)
    ikisi de 8.376805373 verir.

    DURUST NITELIK: `beta > gamma`da `f`nin koku VARDIR ama fiziksel DEGILDIR
    -- (0,1)'de 1.374183181, (1,3)'te 2.331811168 cikiyor ve ayni noktalarda
    foton kuresi YOK. 1PN'de kesilmis `A` orada bir kara delik tarif
    etmiyor; sayi doner ama yorumlanmamalidir. Bu yuzden `beta > gamma`
    REDDEDILIR -- sessizce inanilir gorunen bir sayi dondurmek, deponun en
    pahali arizasidir.
    """
    _ppn_kapisi(theory=theory, ad="isco_radius")
    a2 = _a2(theory=theory, M=M)

    def f(u: float) -> float:
        Q1 = -2.0 * M + 2.0 * a2 * u
        P1 = 2.0 * u - 6.0 * M * u * u + 4.0 * a2 * u ** 3
        Q2 = 2.0 * a2
        P2 = 2.0 - 12.0 * M * u + 12.0 * a2 * u * u
        return Q1 * P2 - Q2 * P1

    u = _ilk_kok(f, 1e-9 / M, 0.499 / M, ad="ISCO")
    return 1.0 / (u * M)


def lagrange_mu_c() -> float:
    """L4/L5 kararlilik esigi `mu_c`. SINIF: KAPALI-FORM TUREV.

    Dairesel kisitli uc-cisim probleminde L4 eskenar noktasi `mu` kutle orani
    kucukken kararlidir. Burada L4'un YERI de, dogrusallastirma matrisi de
    SAYISAL bulunur: `grad U = 0` cozulur, `U`nun ikinci turevleri merkezi
    farkla alinir, 4x4 matrisin ozdegerleri `numpy.linalg.eigvals` ile
    hesaplanir ve `max Re(lambda) > 0` olan en kucuk `mu` ikiye bolmeyle
    aranir.

    OLCULEN: 0.038520898328784976; kapali form `(1 - sqrt(69)/9)/2` =
    0.03852089650455137, fark 1.8242e-09, 2.15 ms. Fark tamamen ikinci turevin
    merkezi fark adimindan (h = 1e-4) ve `max Re(lambda) > 1e-7` esiginden
    gelir.

    SINIF GEREKCESI: hicbir yorunge cozulmez, tek bir NOKTADAKI
    dogrusallastirma okunur. `mu_c`nin kapali formu vardir (`27*mu*(1-mu) = 1`
    kokunun kucuk olani), yani "simulasyon bu sayi hakkinda yanilabilir
    miydi?" sorusunun cevabi -- sayinin KENDISI acisindan -- hayirdir.

    NEDEN `observable/`DA VE `engine/grav/nbody.py`DE DEGIL: cunku Task 4'te
    `nbody.py` HENUZ YOK. Task 5 onu sevk ettiginde bu govde ORADA DEGIL
    BURADA kalir, cunku bir N-body COZUCUSU kullanmiyor; bir matrisin
    ozdegerlerini okuyor.
    """
    def U(x: float, y: float, mu: float) -> float:
        r1 = math.hypot(x + mu, y)
        r2 = math.hypot(x - 1.0 + mu, y)
        return 0.5 * (x * x + y * y) + (1.0 - mu) / r1 + mu / r2

    def gradU(v, mu):
        x, y = v
        r1 = math.hypot(x + mu, y)
        r2 = math.hypot(x - 1.0 + mu, y)
        gx = x - (1.0 - mu) * (x + mu) / r1 ** 3 - mu * (x - 1.0 + mu) / r2 ** 3
        gy = y - (1.0 - mu) * y / r1 ** 3 - mu * y / r2 ** 3
        return [gx, gy]

    def maks_re(mu: float) -> float:
        x0, y0 = fsolve(gradU, [0.5 - mu, 0.86602540378], args=(mu,), xtol=1e-13)
        h = 1e-4
        Uxx = (U(x0 + h, y0, mu) - 2.0 * U(x0, y0, mu) + U(x0 - h, y0, mu)) / h ** 2
        Uyy = (U(x0, y0 + h, mu) - 2.0 * U(x0, y0, mu) + U(x0, y0 - h, mu)) / h ** 2
        Uxy = (U(x0 + h, y0 + h, mu) - U(x0 + h, y0 - h, mu)
               - U(x0 - h, y0 + h, mu) + U(x0 - h, y0 - h, mu)) / (4.0 * h * h)
        A = np.array([[0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [Uxx, Uxy, 0.0, 2.0],
                      [Uxy, Uyy, -2.0, 0.0]])
        return float(np.max(np.linalg.eigvals(A).real))

    lo, hi = 0.01, 0.06
    for _ in range(50):
        orta = 0.5 * (lo + hi)
        if maks_re(orta) < 1e-7:
            lo = orta
        else:
            hi = orta
    return 0.5 * (lo + hi)


def apsidal_exponent(*, force_exponents: tuple[float, ...] = APSIDAL_GRID) -> float:
    """`Phi(n) ~ (3+n)^s` iliskisindeki `s` ussu. SINIF: KAPALI-FORM TUREV.

    Merkezi kuvvet `F ~ -k*r^n` altinda YAKIN-DAIRESEL bir yorungenin apsidal
    acisi `Phi`dir. Burada `Phi` bir yorunge COZULEREK degil, dairesel
    noktadaki DOGRUSALLASTIRMADAN okunur:

        V_eff(r) = k*r^(n+1)/(n+1) + L^2/(2 r^2),   L^2 = k*rc^(n+3)
        omega_r^2 = V_eff''(rc)   (MERKEZI FARKLA, h = 1e-4)
        omega_phi = sqrt(L^2)/rc^2
        Phi = pi * omega_phi / omega_r

    sonra `log(Phi/pi)` ile `log(3+n)` arasinda dogru fit edilir; egim `s`dir.

    OLCULEN: -0.4999999923696353, `-1/2`den fark 7.6304e-09, 19.1 us.
    Hatanin tamami merkezi fark adimindandir.

    SINIF GEREKCESI -- VE BU BIR SINIR VAKASIDIR, o yuzden acikca yaziliyor:
    govde bir yorunge integre ETMEZ, bir noktadaki ikinci turevi alir. Deponun
    olcutu ("simulasyon bu sayi hakkinda YANLIS olabilir miydi?") burada
    "hayir"a yakindir ama "tam kapali form"dan da uzaktir: `Phi = pi/sqrt(3+n)`
    kapali formu KODA GIRMEZ, `-1/2` fit'ten cikar. Yine de yorunge
    cozulmedigi icin raf "kapali-form turev"dir; `n = -1` (log potansiyel)
    izgarada YOKTUR cunku orada `V_eff` ifadesi 0/0'dir.

    FIT'IN KESISIMI BEDAVA BIR SINAVDIR ve KAPI OLARAK KONDU -- olcularak,
    tasarlanarak degil. `log(Phi/pi) = s*log(3+n) + c` dogrusunda kapali form
    `c = 0` der ve bu bilgi EGIMDE KULLANILMAZ, yani sinav bedavadir. Olculen
    `c = -1.578485733121196e-08`.

    NEDEN GEREKLI OLDUGU MUTASYONLA OLCULDU: govdeden `math.pi` carpani
    silindiginde `log(Phi/pi)` sabit bir `-log(pi)` kadar kayar ve EGIM BIT BIT
    AYNI KALIR (-0.49999999236963555 <-> -0.4999999923696353), yani o mutant
    egimi okuyan HER testten sag cikar. Kesisim ise -1.1447299016342576'ya
    gider ve mutanti orada yakalar. Bir us fit'i OLCEGE KORDUR; kesisim onun
    gorMEDIGI yuzdur.
    """
    if len(force_exponents) < 2:
        raise InvalidInputError(
            f"en az iki kuvvet ussu gerekir, verilen: {force_exponents!r}"
        )
    xs, ys = [], []
    for n in force_exponents:
        if n == -1.0:
            raise InvalidInputError(
                "n = -1 (logaritmik potansiyel) bu govdede tanimsiz: "
                "`r^(n+1)/(n+1)` orada 0/0'dir."
            )
        if not (3.0 + n > 0.0):
            raise InvalidInputError(
                f"n > -3 olmali (aksi halde dairesel yorunge kararsizdir), "
                f"verilen: {n!r}"
            )
        k = rc = 1.0
        L2 = k * rc ** (n + 3.0)

        def Veff(r: float) -> float:
            return k * r ** (n + 1.0) / (n + 1.0) + L2 / (2.0 * r * r)

        h = 1e-4
        d2 = (Veff(rc + h) - 2.0 * Veff(rc) + Veff(rc - h)) / (h * h)
        Phi = math.pi * (math.sqrt(L2) / rc ** 2) / math.sqrt(d2)
        xs.append(math.log(3.0 + n))
        ys.append(math.log(Phi / math.pi))
    egim, kesisim = np.polyfit(np.array(xs), np.array(ys), 1)
    if abs(float(kesisim)) > 1e-6:
        raise InvalidInputError(
            f"apsidal fit'in kesisimi {float(kesisim)!r}; kapali form 0 der "
            "(olculen 1.6e-08). Bir us fit'i OLCEGE kordur, kesisim degildir: "
            "sifirdan sapan bir kesisim `Phi`nin genel carpaninin bozuldugunu "
            "soyler ve o bozulma egimde GORUNMEZ."
        )
    return float(egim)


def _a2(*, theory: Theory, M: float) -> float:
    """`A`nin ikinci mertebe katsayisi -- `null_potentials`tan OKUNUR."""
    return null_potentials(theory=theory, M=M, gauge="schwarzschild").time_poly[2]


def _ppn_kapisi(*, theory: Theory, ad: str) -> None:
    if theory.beta > theory.gamma:
        raise InvalidInputError(
            f"{ad}: beta > gamma (beta={theory.beta!r}, gamma={theory.gamma!r}) "
            "olan PPN noktalarinda 1PN'de kesilmis A(u) ufuk yapmiyor; foton "
            "kuresi YOK ve donen ISCO koku fiziksel degil (olculdu: "
            "(gamma,beta) = (0,1)'de 1.374183181, (1,3)'te 2.331811168, ve "
            "ayni noktalarda dV/du'nun koku yok). Inanilir gorunen bir sayi "
            "dondurmek yerine reddediliyor."
        )


def _foton_u(*, theory: Theory, M: float) -> float:
    _ppn_kapisi(theory=theory, ad="photon_sphere")
    a2 = _a2(theory=theory, M=M)
    f = lambda u: 2.0 * u - 6.0 * M * u * u + 4.0 * a2 * u ** 3
    return _ilk_kok(f, 1e-9 / M, 0.499 / M, ad="foton kuresi")


def _ilk_kok(f, lo: float, hi: float, *, ad: str) -> float:
    if f(lo) * f(hi) > 0.0:
        raise InvalidInputError(
            f"{ad}: (0, 1/(2M)] araliginda kok yok "
            f"(f({lo!r}) = {f(lo)!r}, f({hi!r}) = {f(hi)!r})."
        )
    return brentq(f, lo, hi, xtol=1e-18, rtol=8.9e-16, maxiter=200)
