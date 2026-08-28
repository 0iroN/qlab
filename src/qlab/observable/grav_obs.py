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

import math
from dataclasses import dataclass
from typing import Any, Callable

from qlab.errors import InvalidInputError

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


def newton_noise_floor(fn: Callable[..., float], **kw: Any) -> NoiseFloor:
    """Fizigi KAPAT, AYNI boruyu kostur -- ve tabanin bir sey olctugunu OLC.

    `fn` fizigi kapatma noktalarini KENDISI beyan eder; cagiran veremez.
    Beyan iki alandadir ve ikisi de ZORUNLUDUR:

        fn.PIPELINE       -- borunun adi (`fn.__name__` DEGIL: ad degisirse
                             etiket degismemeli, ve iki fonksiyon ayni boruyu
                             kullanabilmelidir)
        fn.NEWTON_PROBES  -- `(kwargs, TAM deger)` ciftleri; `kwargs` `fn`e
                             fizigi kapatan argumanlari verir

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

    `CURUK` BU BORUDA ERISILEMEZ VE SEBEBI YAPISALDIR -- olculdu, varsayilmadi.
    Karelemede sinyalin gurultuye gomulmesi (`lam*M/p < eps`) ile merdivenin
    olmesi (`w` tam 1.0'a yuvarlanmasi) AYNI kosuldur, cunku `w - 1` ile
    `dphi - 2*pi` ayni kucuk buyuklugun (`lam*M/p`) iki yuzudur. `p/M`yi
    1e5'ten 1e17'ye tarayan olcumde hukum SAGLAM'dan dogrudan DEJENERE'ye
    geciyor (1e12'de S/G = 2.6671e+03 SAGLAM, 1e13'te 0/6 canli DEJENERE);
    arada CURUK cikan hicbir nokta YOK. Yani bu boruda "panel soyluyor mu"
    sorusunun cevabini `DEJENERE` tasiyor. `CURUK` dali baska borular icin
    duruyor -- N-body tarafinda `m2 = 0` tabaninin sinyali `~m2^2` gittigi
    icin orada bir KESISIM vardir (spec 4.3).
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
    boru = f"{etiket} | fizik-kapali merdiven {canli}/{len(probes)} canli"
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
