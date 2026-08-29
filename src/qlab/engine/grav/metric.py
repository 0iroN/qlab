"""Metrik, `w` formu ve KURAM SECICI.

BIRIM SECIMI: `G = c = 1`. Bu bir YAKLASIM DEGILDIR, bir birim secimidir --
`M` bir uzunluktur (Schwarzschild yaricapi `r_s = 2M`), `u = 1/r` bir ters
uzunluktur, ve `radial_w` ile `closed_form_coefficient` BOYUTSUZDUR.

IKI AYAR SEVK EDILIYOR ve ikisi de bu dosyada:

- SCHWARZSCHILD ayari (`radial_w`): duzeltme `u`da KARESEL girer
  (`u'' + u = M/L^2 + lam*M*u^2`), yani enerji denkleminde KUBIK bir terim.
- IZOTROPIK PPN ayari (`isotropic_potential`): duzeltme metrik
  potansiyellerinde `u`da DOGRUSAL girer (`B = 1 + 2*gamma*M*u`).

Ikisi ayni `C = 2*pi*(2 + 2*gamma - beta)` sayisini verir; Task 3 bunu GERCEK
CAPRAZ KONTROL olarak kilitleyecek. Bu dosyanin kendi iki olcumu daha dardir
ve IKISININ GUCU AYNI DEGILDIR -- fark mutasyonla olculdu:

- kareleme (Schwarzschild ayari) <-> `closed_form_coefficient`: bes PPN
  noktasinda 2.0e-10'da usuyorlar, AMA IKISI DE `Theory.lam`i okur. Ortak
  dugme capraz kontrolu KORLESTIRIR (spec 7.3'un dorduncu maddesi): `lam`in
  formulunu `2+gamma-beta` yapan mutantta iki taraf BIRLIKTE kayar ve bu
  karsilastirma sessiz kalir. Onu oldurmek icin testteki ELLE yazilmis
  carpanlar (3, 1, 4, 5, 1) gerekir.
- izotropik acilimin ikinci mertebe katsayisi <-> ayni kapali form: sekiz
  noktada 2.3e-09'da usuyorlar, ve BU ORTAK DUGMEYI TASIMAZ -- izotropik
  taraf `gamma` ile `beta`yi dogrudan okur, `lam`i HIC gormez. Olculdu:
  `Theory.lam` mutanti bu testi de OLDURUYOR.

(`tests/grav/test_metric.py`, pencereler orada yazili.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

from qlab.errors import InvalidInputError

#: SEVK EDILEN IKI AYARIN ADLARI -- TEK KAYNAK. `geodesic.GAUGES` bunu yeniden
#: yazmaz, buradan okur; iki liste ayni seyi belirlerse sessiz bir kazanan
#: dogar (Task 1'in olcumu) ve o kazanan adlandirilmamis bir pencere
#: dugmesidir. "Ayar" burada bir KOORDINAT SECIMIDIR: ayni fizik, baska bir
#: `r` tanimi ve baska bir metrik yazilisi.
GAUGE_NAMES: tuple[str, ...] = ("schwarzschild", "isotropic")


@dataclass(frozen=True)
class Theory:
    """Hangi kuramin cozuldugunu SECEN dugmeler -- YENI BIR KUTU.

    Bunlar bir YAKLASIM DEGILDIR (yaklasim YAPILAN bir seydir: "test
    parcacigi aldik"), bir BIRIM SECIMI de degildir (`G = c = 1` oyle).
    Bunlar HANGI KURAMIN cozuldugunu secer, ve panel bunu boyle yazar.

    GR: gamma = beta = 1 <=> lam = 3. HERHANGI BASKA DEGER GR DEGILDIR.
    `lam` BIR ALAN DEGIL, TURETILMIS BIR OZELLIKTIR -- asagiya bak.

    VE IKISI AYNI KAPALI FORMUN IKI YUZUDUR:

        C = 2*pi*(2 + 2*gamma - beta)        lam_eff = 2 + 2*gamma - beta

    `LAM` NEDEN GERCEK VARSAYILANLI BIR ALAN DEGIL -- BU YASAK OLCULDU,
    TERCIH EDILMEDI. Bir dataclass alani gercek bir varsayilan tasidiginda
    "verildi" ile "verilmedi" ayirt EDILEMEZ, yani "lam ile (gamma, beta)
    catisirsa hata" kurali ancak deger tabanli bir tutarlilik sinavi olarak
    uygulanabilirdi. O halde kural su uc govdeyi de OLDURURDU:
      - Newton noktasi `Theory(gamma=-1, beta=0)` (lam_eff = 0, varsayilan 3),
      - kapali formun bes parametre satirindan DORDU (lam_eff 1/4/5/1),
      - Task 2'nin `Theory(gamma=g, beta=2+2*g)` enjeksiyonu (lam_eff = 0).
    Yani spec 4.3'un "pazarlik edilmez" hata cubugu makinesi ILK CAGRIDA
    istisna atardi. `@property` bu ailenin tamamini serbest birakir, ve
    `Theory(lam=...)` yine de `TypeError` verir cunku `lam` kurucu argumani
    DEGILDIR.
    """

    gamma: float = 1.0
    beta: float = 1.0

    @property
    def lam(self) -> float:
        """TURETILMIS -- BIR ALAN DEGIL. Tek kaynak (gamma, beta)."""
        return 2.0 + 2.0 * self.gamma - self.beta


@dataclass(frozen=True)
class IsotropicOrbit:
    """Izotropik ayarda BAGLI bir yorungenin kareleme cekirdegi -- CIPA YOLU.

    Schwarzschild ayarinin `radial_w`i ile AYNI ISI gorur ama BASKA BIR
    MEKANIZMAYLA, ve ikisinin ORTAK DUGMESI YOKTUR (olculdu, kaydeden vekil:
    Schwarzschild yolu `Theory`nin yalniz `lam`ini, bu yol yalniz `gamma` ile
    `beta`sini okur). Task 3'un capraz kontrolu bu ayrimin uzerinde durur.

    TUREV -- ve her adimi cebirsel, sayisal bir fit YOK:

        ds^2 = -A dt^2 + B (dr^2 + r^2 dphi^2),   u = 1/r_izotropik
        E = A t',  L = B r^2 phi'  (korunanlar)
        (du/dphi)^2 = X*(B/A)(u) - Y*B(u) - u^2,  X = E^2/L^2, Y = 1/L^2

    `X` ile `Y` yorungeden OKUNUR, girilmez: donus noktalarinda `du/dphi = 0`
    oldugu icin iki denklem `X` ve `Y`de DOGRUSALDIR ve 2x2 Cramer ile tam
    cozulur. (Schwarzschild yolunda karsiligi `u1`, `u2`nin `w`ye girmesidir.)

    Sonra ifade `A(u)` ile carpilir: `N(u) = (du/dphi)^2 * A(u)` DORDUNCU
    DERECEDEN bir POLINOMDUR ve `u1` ile `u2` onun kokleridir. Deflasyonla

        N(u) = (u - u1)(u2 - u) R(u),   R ikinci dereceden

    ve `psi = arcsin((u - orta)/yari)` ikamesinden sonra integrand TAM OLARAK
    `1/sqrt(R(u)/A(u))` olur -- Schwarzschild yolunun `1/sqrt(w(u))`suyla ayni
    KALIP, tamamen farkli GOVDE.

    DEFLASYON DORT KATSAYIDAN YALNIZ UCUNU KULLANIR (`n2`, `n3`, `n4`), ve
    KALAN IKISI BEDAVA BIR SINAVDIR: `n0 = -Pr*r0` ile `n1 = S*r0 - Pr*r1`
    yeniden uretilmek zorundadir. Olculdu (SEVK EDILEN kodla, sekiz
    (gamma, beta) noktasi, p = 3e5, e = 0.3, M = 1):
    |n0 artigi| 2.391e-22 ... 1.551e-21, |n1 artigi| 6.667e-18 ... 4.507e-16.
    Bu bir tautoloji DEGILDIR: `n0` ile `n1` `X`, `Y`, `r0`, `r1`, `r2`
    zincirinin HICBIR yerinde kullanilmaz.

    Alanlar:
      `time_poly`   -- `A(u) = 1 - 2*M*u + 2*beta*(M*u)^2` katsayilari (c0,c1,c2)
      `radial_poly` -- `R(u)` katsayilari, ayni sirayla
      `energy_term` -- `X = E^2/L^2`
      `inverse_L2`  -- `Y = 1/L^2`

    IKI POLINOM AYNI SIRAYLA VE AYNI BICIMDE saklanir, cunku Newton
    noktasinda (`gamma = -1, beta = 0`) `R` ile `A` OZDES olur ve `isotropic_w`
    orada BIT BIT 1.0 dondurmelidir; farkli bicimde degerlendirilen iki ozdes
    polinom bunu yarim ulp'le kacirir.
    """

    time_poly: tuple[float, float, float]
    radial_poly: tuple[float, float, float]
    energy_term: float
    inverse_L2: float


@dataclass(frozen=True)
class IsotropicPotentials:
    """Izotropik PPN metriginin iki potansiyeli, tek bir `u` noktasinda.

        ds^2 = -A(u) dt^2 + B(u) (dx^2 + dy^2 + dz^2)

    `time_factor` = A, `space_factor` = B. Ikisi de BOYUTSUZDUR ve `u = 0`da
    (sonsuzda) 1'e gider.
    """

    time_factor: float
    space_factor: float


def closed_form_coefficient(theory: Theory) -> float:
    """Perihel katsayisinin KAPALI FORMU: `C = 2*pi*(2 + 2*gamma - beta)`.

    SINIF -- ve bu dosyada iki ayri sinifi var, karistirmak pahalidir:

      * TEK BASINA cagrilip bir sabitle karsilastirildiginda bu bir
        TRANSKRIPSIYON KILIDIDIR. Cozulen bir model yoktur; deponun olcutu
        ("simulasyon bu sayi hakkinda YANLIS olabilir miydi?") HAYIR der.
      * Bir KARELEME sonucuyla karsilastirildiginda FIZIK CIPASININ REFERANS
        TARAFIDIR. Olculdu (`tests/grav/test_metric.py`, psi ikamesi,
        quad 1e-14/1e-13/400, Richardson (3e5, 1e6), e = 0.3, M = 1): bes PPN
        noktasinda kareleme ile bagil fark 9.826e-11 ... 2.003e-10; GR'de
        `6*pi = 18.84955592153876`den fark -1.852e-09, 10.01 durust hane.

    `C`nin donduruldugu deger bes noktada `2*pi*k` ile BIT BIT esittir
    (olculdu; yaklasiklik payi degil, tam esitlik).

    TEK YOL: `lam` bu fonksiyonda YENIDEN yazilmaz, `theory.lam` uzerinden
    okunur. Ikinci bir kopya, ikinci bir sessiz kazanan demektir.
    """
    return 2.0 * math.pi * theory.lam


def deflection_closed_form(theory: Theory) -> float:
    """Isik bukulmesi oraninin KAPALI FORMU: `delta(gamma)/delta(0) = 1 + gamma`.

    SINIF -- `closed_form_coefficient`taki AYNI ikilik burada da gecerlidir:
    tek basina bir sabitle karsilastirilirsa TRANSKRIPSIYON KILIDI, bir
    kareleme sonucuyla karsilastirilirsa FIZIK CIPASININ REFERANS TARAFI.
    Olculdu (`tests/grav/test_bukulme.py`, Richardson (1e5,3e5), M = 1):
    GR'de Schwarzschild ayari 1.9999999996887523 (9.81 durust hane),
    izotropik 1.999999999804698 (10.01); PPN ailesinde en kotu sapma
    4.3487e-09.

    `beta` BURAYA GIRMEZ, ve bu bir ihmal degil bir OLCUM: `beta` bukulmeye
    IKINCI mertebede girer ve o da kapali formdur --
    `d(delta)/d(beta) = -pi*M^2/b^2` (olculdu, b = 1e4: oran 1.000751).

    TEK YOL: `1 + gamma` bu depoda YALNIZ burada yazilir.
    """
    return 1.0 + theory.gamma


def radial_w(u, u1: float, u2: float, *, theory: Theory, M: float):
    """Radyal karelemenin `w` FORMU: `w(u) = 1 - a*(u1 + u2 + u)`.

    `a = 2*lam*M/3`. Bagli yorunge icin

        (du/dphi)^2 = (u - u1) * (u2 - u) * w(u)

    ve `u1 = (1-e)/p` (afel), `u2 = (1+e)/p` (perihel), `p = a_yariboyut(1-e^2)`.

    BU FORM BIR TERCIH DEGIL, SPEC 4.3'UN ON KOSULUDUR. Ayni carpan kok
    formunda `(2*lam*M/3) * (3/(2*lam*M) - u1 - u2 - u)` diye yazilabilir ve
    OLCULDU: lam = 0'da `3/(2*lam*M)` Python float aritmetiginde
    `ZeroDivisionError` atar -- yani "lambda = 0 kosusu HATA CUBUGUDUR" kurali
    kok formunda ILK CAGRIDA coker. `w` formu ayni noktada TAM 1.0 dondurur
    (olculdu: on bes (u, M) ciftinde bit bit 1.0).

    VE lam = 3'te KOK FORMUNDAN AYIRT EDILEMEZ -- ama "bit bit ayni" DEGIL,
    ve bu fark olcumle bulundu (ilk yazdigim test noktasal esitlik isteyip
    KIRMIZI dustu). Olculdu: 336 (p, e, u) noktasinda noktasal bit bit
    esitlik 248/336, en kotu bagil fark 1.205e-16 -- yani YARIM ulp
    (1 ulp = 2.220e-16). HUKUM duzeyinde ise fark HIC YOK: disdegerlenmis
    `C` uc dismerkezlikte de bit bit ayni (0.000e+00; Richardson (3e5, 1e6),
    e in {0.3, 0.7, 0.95}).

    `u` bir `float` ya da `numpy.ndarray` olabilir; govde eleman elemandir.

    `u1 < u2` KAPISI -- ve gerekcesi spec 4.6'dan DAR:
    spec "kareleme dairesel yorungede cokuyor" diyor. Olculdu (GR, M = 1,
    p = 3e5, quad 1e-14/1e-13/400; hedef `3*pi*lam^2` = 84.82300164692441):

        e        ham-u karelemesi K      psi ikamesi K
        1e-12    -4.141e+09              +84.82444
        1e-06    -5.314e+03              +84.82444
        1e-03    +7.278e+01              +84.82444

    Yani (1) `e > 0` kapisi ham-u borusunu KURTARMAZ -- e = 1e-6 kapidan gecer
    ve sonuc orada zaten hem ISARET olarak ters hem buyukluk olarak 63 kat
    yanlistir (-5.314e+03 vs +84.823); (2) `psi` ikameli boru e = 0'da bile
    dogru sayiyi verir, yani kapiya SAYISAL olarak ihtiyaci yoktur. Kapi
    burada SOZLESMESEL bir kapidir: `radial_w`in girdisi bagli bir yorungenin
    IKI AYRI donus noktasidir, `u1 == u2` bir dairedir ve orada olculecek bir
    perihel kaymasi yoktur.
    """
    if not math.isfinite(u1) or not math.isfinite(u2):
        raise InvalidInputError(
            f"donus noktalari sonlu olmali, verilen: u1={u1!r}, u2={u2!r}"
        )
    if u1 >= u2:
        raise InvalidInputError(
            f"donus noktalari AYRI olmali (u1 < u2), verilen: u1={u1!r}, "
            f"u2={u2!r}. u1 == u2 dairesel yorunge demektir, yani e = 0, ve "
            "orada perihel kaymasi tanimsizdir. Olculdu (GR, M=1, p=3e5, "
            "ham-u karelemesi): e = 1e-6'da ikinci mertebe katsayisi "
            "K = -5.314e+03 cikiyor, dogrusu 3*pi*lam^2 = 84.823002. "
            "e > 0 verin."
        )
    if not (M > 0.0) or not math.isfinite(M):
        raise InvalidInputError(
            f"M pozitif ve sonlu olmali, verilen: {M!r}. M <= 0 sessizce "
            "Newton limitine dusmez, fiziksel olarak anlamsizdir."
        )
    a = 2.0 * theory.lam * M / 3.0
    return 1.0 - a * (u1 + u2 + u)


def isotropic_potential(u, *, theory: Theory, M: float) -> IsotropicPotentials:
    """Izotropik PPN metriginin iki potansiyeli:

        A(u) = 1 - 2*M*u + 2*beta*(M*u)^2        (time_factor)
        B(u) = 1 + 2*gamma*M*u                   (space_factor)

    `u = 1/r_izotropik`. Metrik STANDART 1PN mertebesinde kesilmistir; `B`nin
    `(M*u)^2` terimi 1PN'de gerekmez ve BILINCLI olarak yoktur. Oran `B/A`
    YENIDEN kesilmez, oldugu gibi alinir -- bu bir MODELLEME SECIMIDIR ve
    getirdigi 2PN artiklari `C`yi (birinci mertebe) etkilemez.

    JEODEZIK DENKLEMDE TEORI BAGIMLILIGININ TAMAMI `B/A` ORANINDADIR:

        B/A = 1 + (2 + 2*gamma)*x + (4 - 2*beta + 4*gamma)*x^2 + O(x^3),
        x = M*u

    ve ikinci mertebe katsayisi `4 - 2*beta + 4*gamma = 2*lam`, yani
    `closed_form_coefficient(theory)/pi`. Olculdu (`x = 1e-3`, iki kademeli
    Richardson): sekiz (gamma, beta) noktasinda en kotu bagil fark 2.302e-09,
    ve `x = 1e-3` bir OPTIMUMDUR (x = 1e-2'de 1.845e-06, x = 1e-5'te
    3.442e-05) -- bir disdegerleme sonsuza kadar iyilesmez.

    NEWTON NOKTASINDA (gamma = -1, beta = 0) `A` ile `B` OZDESTIR: ikisi de
    `1 - 2*M*u`. Olculdu: 2000 rastgele (u, M) noktasinda esitsizlik sayisi 0,
    bit bit. Bu, `w == 1` ile ayni isi goren izotropik karsiliktir -- A == B
    olunca jeodezik denklem cebirsel olarak Newton'un Kepler denklemine iner,
    ve spec 4.3'un hata cubugu BU ayarda da kosulabilir hale gelir.

    DURUST NITELIK: o esitlik `beta`yi TEK BASINA kilitlemez -- olculdu,
    (gamma, beta) = (-1, 1e-16)'da da A == B cikiyor cunku `2*beta*(M*u)^2`
    terimi 0.98'in ulp'sinin altinda kaliyor. `beta`nin degerini goren sinav
    `B/A` aciliminin ikinci mertebesidir.
    """
    if not (M > 0.0) or not math.isfinite(M):
        raise InvalidInputError(
            f"M pozitif ve sonlu olmali, verilen: {M!r}."
        )
    x = M * u
    return IsotropicPotentials(
        time_factor=1.0 - 2.0 * x + 2.0 * theory.beta * x * x,
        space_factor=1.0 + 2.0 * theory.gamma * x,
    )


def isotropic_orbit(u1: float, u2: float, *, theory: Theory, M: float) -> IsotropicOrbit:
    """Izotropik ayarin CIPA YOLU: donus noktalarindan kareleme cekirdegi.

    Girdi Schwarzschild yolununkiyle AYNIDIR (`u1` afel, `u2` perihel, `M`),
    cikti FARKLI BIR NESNEDIR. Turev `IsotropicOrbit` dokstringinde.

    KAPILAR `radial_w`inkiyle AYNI GEREKCEYE dayanir ve burada YENIDEN
    yazilmistir, cunku iki yol birbirinin kapisina GUVENMEZ -- guvenselerdi
    ortak bir dugme dogardi ve capraz kontrol tam oradan korlesirdi.
    """
    if not math.isfinite(u1) or not math.isfinite(u2):
        raise InvalidInputError(
            f"donus noktalari sonlu olmali, verilen: u1={u1!r}, u2={u2!r}"
        )
    if u1 >= u2:
        raise InvalidInputError(
            f"donus noktalari AYRI olmali (u1 < u2), verilen: u1={u1!r}, "
            f"u2={u2!r}. u1 == u2 dairedir; orada olculecek perihel kaymasi "
            "yoktur ve `X`, `Y` icin kurulan 2x2 dizge TEKILDIR (det = 0)."
        )
    if not (M > 0.0) or not math.isfinite(M):
        raise InvalidInputError(
            f"M pozitif ve sonlu olmali, verilen: {M!r}."
        )
    g = theory.gamma
    b = theory.beta
    x1 = M * u1
    x2 = M * u2
    B1 = 1.0 + 2.0 * g * x1
    B2 = 1.0 + 2.0 * g * x2
    A1 = 1.0 - 2.0 * x1 + 2.0 * b * x1 * x1
    A2 = 1.0 - 2.0 * x2 + 2.0 * b * x2 * x2
    P1 = B1 / A1
    P2 = B2 / A2
    det = B1 * P2 - B2 * P1
    if det == 0.0 or not math.isfinite(det):
        raise InvalidInputError(
            f"izotropik yorunge dizgesi tekil: det={det!r} (u1={u1!r}, "
            f"u2={u2!r}, gamma={g!r}, beta={b!r}, M={M!r})."
        )
    X = (B1 * u2 * u2 - B2 * u1 * u1) / det
    Y = (P1 * u2 * u2 - P2 * u1 * u1) / det

    # N(u) = (du/dphi)^2 * A(u), dorduncu dereceden. `n0` ile `n1` BILEREK
    # HESAPLANMIYOR: deflasyon icin gerekmiyorlar ve hesaplanmadiklari icin
    # `test_perihel.py`de bedava bir SINAV oluyorlar. (Ilk yazdigimda `n1`
    # burada duruyordu ve HICBIR YERDE kullanilmiyordu -- olu koddu.)
    n2 = -1.0 - 2.0 * Y * M * M * (b - 2.0 * g)
    n3 = 2.0 * M - 4.0 * Y * b * g * M ** 3
    n4 = -2.0 * b * M * M

    S = u1 + u2
    Pr = u1 * u2
    r2 = -n4
    r1 = S * r2 - n3
    r0 = S * r1 - Pr * r2 - n2
    return IsotropicOrbit(
        time_poly=(1.0, -2.0 * M, 2.0 * b * M * M),
        radial_poly=(r0, r1, r2),
        energy_term=X,
        inverse_L2=Y,
    )


def isotropic_w(u, orbit: IsotropicOrbit):
    """Izotropik ayarin `w`si: `R(u)/A(u)`. Integrand `1/sqrt(w)` -- AYNI KALIP.

    `radial_w` ile bilerek ayni role sahiptir ve ayni bicimde tuketilir, ama
    `Theory`ye HIC dokunmaz: kuram bagimliligi zaten `isotropic_orbit`te
    tuketilmistir. Bu ayrim capraz kontrolun ON KOSULUDUR.

    `u` bir `float` ya da `numpy.ndarray` olabilir; govde eleman elemandir.
    """
    a0, a1, a2 = orbit.time_poly
    r0, r1, r2 = orbit.radial_poly
    return (r0 + r1 * u + r2 * u * u) / (a0 + a1 * u + a2 * u * u)


@dataclass(frozen=True)
class NullPotentials:
    """BOS jeodeziklerin metrik parcalari -- SECILEN ayarda, POLINOM olarak.

    Kutlesiz bir parcacik icin (`b = L/E`, `u = 1/r`):

        (du/dphi)^2 = N(u) / D(u),
        N(u) = S(u)/b^2 - u^2 * A(u),        D(u) = A(u) / T(u)

    ve iki ayar YALNIZ (A, S, T) uclusunde ayrilir:

        schwarzschild : A = 1 - 2*M*u + 2*(beta-gamma)*(M*u)^2
                        S = 1
                        T = 1 - 2*gamma*M*u          (yani g^rr)
        isotropic     : A = 1 - 2*M*u + 2*beta*(M*u)^2
                        S = B = 1 + 2*gamma*M*u
                        T = 1

    NEDEN SCHWARZSCHILD TARAFINDA `1/B` DEGIL `g^rr = 1 - 2*gamma*M*u`:
    cunku GR'de (gamma = beta = 1) bu secim metrigi TAM Schwarzschild yapar --
    `A = 1-2Mu`, `T = 1-2Mu`, `D = 1` ve `(du/dphi)^2 = 1/b^2 - u^2 + 2*M*u^3`,
    yani 1PN'de KESILMIS bir yaklasim degil, kapali cozumun kendisi. Olculdu:
    ikinci mertebe katsayisi `(delta*b/M - 4)*b/M` bu boruda b = 1e6'da
    11.781358, `15*pi/4` = 11.780972 (fark 3.9e-04, ve o farkin tamami
    yuvarlamadir: delta'nin mutlak ulp'si b^2 ile carpilinca b = 1e6'da
    +-3e-04 eder). Literaturun `15*pi/4 - 4` sayisi EN YAKIN GECIS `r0`
    cinsindendir; `b` (carpisma parametresi) cinsinden ayni sonuc `15*pi/4`tur
    (`1/b = (1/r0)(1 - M/r0 + ...)` katsayiyi tam +4 kaydirir).

    IZOTROPIK TARAF 1PN'DE KESILMISTIR ve bu bilinclidir
    (`isotropic_potential` dokstringi): orada ikinci mertebe bir MODELLEME
    ARTIGIDIR -- olculdu `3*pi` = 9.424778 (b = 1e6'da 9.424576). Yani
    bukulmenin BIRINCI mertebesi ayardan bagimsizdir (`1+gamma`; iki ayar
    1.159457e-10'da usuyor), IKINCI mertebesi DEGILDIR. Bu,
    `second_order_coefficient`in perihel tarafinda olculen hukmunun ta
    kendisidir.

    Alanlar (hepsi (c0, c1, c2) sirasinda, `P(u) = c0 + c1*u + c2*u^2`):
      `time_poly`   -- A
      `source_poly` -- S
      `damp_poly`   -- T
    """

    time_poly: tuple[float, float, float]
    source_poly: tuple[float, float, float]
    damp_poly: tuple[float, float, float]


@dataclass(frozen=True)
class NullKernel:
    """Bir BOS jeodezigin kareleme cekirdegi -- `w` FORMUNDA.

    `N(u)` DORDUNCU dereceden bir polinomdur; fiziksel donus noktasi `u0`
    onun [0, 1/(2M)] araligindaki EN KUCUK pozitif kokudur. Deflasyonla

        N(u) = (u0 - u) * Q(u),   Q ucuncu dereceden

    ve `u = u0*sin(theta)`, `theta` [0, pi/2] ikamesinden sonra

        delta = 2 * INT_0^{pi/2} sqrt(u0 * (1 + sin theta) * w(u)) dtheta - pi,
        w(u) = D(u) / Q(u)

    olur; kok tekilligi ANALITIK kalkar (`cos(theta)` ile `sqrt(1 - sin)`
    sadelesir). Bu, bagli yorungenin `psi` ikamesinin kutlesiz karsiligidir ve
    AYNI GEREKCEYLE sevk edilmistir.

    VE `w` FORMU BURADA DA SPEC 4.3'UN ON KOSULUDUR: `M -> 0`da `A = T = 1`,
    `Q(u) = u0 + u` olur, integrandin TAMAMI `sqrt(u0*(1+s)/(u0+u0*s))` = 1.0
    olur (bit bit) ve `delta` TAM 0.0 cikar. Olculdu (`M = 1e-200`, b = 1e5,
    iki ayarda da `delta == 0.0`; `M = 0`in kendisi `M > 0` kapisiyla zaten
    reddediliyor). Yani "kutleyi kapat" noktasi TEK BASINA bir hata cubugu
    DEGILDIR, tipki karelemede `lambda = 0`in olmadigi gibi. Bu borunun tabani bu yuzden bir
    NOKTADAN degil `gamma -> -1` LIMITINDEN okunur
    (`observable/grav_obs.deflection_ratio`).

    DEFLASYON BES KATSAYIDAN YALNIZ DORDUNU KULLANIR (`n1`..`n4`) ve BESINCISI
    BEDAVA BIR SINAVDIR: `u0*q0 = n0` yeniden uretilmek zorundadir. Olculdu
    (sevk edilen kod, GR, M = 1, b = 1e3 ... 1e6): |u0*q0 - n0| 1.212e-27 ...
    6.353e-22. Bu bir tautoloji DEGILDIR -- `n0` deflasyon zincirinin hicbir
    yerinde kullanilmaz.

    Alanlar:
      `turning_point`     -- `u0`
      `radial_poly`       -- `Q` katsayilari (q0, q1, q2, q3)
      `time_poly`         -- `A` (D'nin payi)
      `damp_poly`         -- `T` (D'nin paydasi)
      `constant_residual` -- `u0*q0 - n0`, BEDAVA SINAVIN artigi
    """

    turning_point: float
    radial_poly: tuple[float, float, float, float]
    time_poly: tuple[float, float, float]
    damp_poly: tuple[float, float, float]
    constant_residual: float


def null_potentials(*, theory: Theory, M: float, gauge: str) -> NullPotentials:
    """Bos jeodezigin (A, S, T) uclusu -- BU UCLUNUN TEK KAYNAGI.

    Tablo `NullPotentials` dokstringinde. Bu fonksiyon `A`yi ikinci bir yerde
    YENIDEN yazmamak icin vardir: `null_kernel`, `isco_radius`,
    `photon_sphere` ve `critical_impact_parameter` DORDU DE buradan okur.

    IZOTROPIK TARAF `isotropic_potential` ILE AYNI METRIGI tarif etmek
    ZORUNDADIR, ve bu bir GOVDE PAYLASIMIYLA degil bir SINAVLA saglanir
    (`tests/grav/test_bukulme.py`): iki yol bagimsiz yazilir, sonra buradan
    kurulan `A` ve `S` degerleri `isotropic_potential`in dondurduklerine BIT
    BIT esit mi diye bakilir. Govde paylassalardi ortak bir dugme dogardi;
    yalnizca sinav paylasinca dogan sey TUTARLILIKTIR.
    """
    if gauge not in GAUGE_NAMES:
        raise InvalidInputError(
            f"bilinmeyen ayar: {gauge!r}. Sevk edilen ayarlar: {GAUGE_NAMES}."
        )
    if not (M > 0.0) or not math.isfinite(M):
        raise InvalidInputError(f"M pozitif ve sonlu olmali, verilen: {M!r}.")
    g = theory.gamma
    b = theory.beta
    if gauge == "schwarzschild":
        return NullPotentials(
            time_poly=(1.0, -2.0 * M, 2.0 * (b - g) * M * M),
            source_poly=(1.0, 0.0, 0.0),
            damp_poly=(1.0, -2.0 * g * M, 0.0),
        )
    return NullPotentials(
        time_poly=(1.0, -2.0 * M, 2.0 * b * M * M),
        source_poly=(1.0, 2.0 * g * M, 0.0),
        damp_poly=(1.0, 0.0, 0.0),
    )


def evaluate_poly(coeffs, u):
    """Horner. `u` bir `float` ya da `numpy.ndarray` olabilir."""
    toplam = coeffs[-1] + 0.0 * u
    for k in reversed(coeffs[:-1]):
        toplam = toplam * u + k
    return toplam


def _en_kucuk_pozitif_kok(N, *, b: float, M: float) -> float:
    """`N(u) = 0`in (0, 1/(2M)] araligindaki EN KUCUK pozitif koku.

    `N(0) = S(0)/b^2 > 0` oldugu icin kok, `N`nin ilk isaret degistirdigi
    yerdedir; ust sinir `1.05/b`den baslanip 1.05 ile buyutulerek ARANIR.
    Tavan `u = 1/(2M)`dir.

    `np.roots` KULLANILMIYOR ve sebebi olculdu: dorduncu dereceden katsayi
    `-2*(beta-gamma)*M^2` GR'de TAM 0.0'dir, `np.roots` orada derecesi
    dusen bir polinomla calisip kok SAYISINI ayardan ayara degistirir;
    deflasyon ise TEK bir koke ihtiyac duyar. `brentq` her iki durumda da
    ayni sayida kok dondurur: bir tane.
    """
    f = lambda u: evaluate_poly(N, u)
    tavan = 0.5 / M
    ust = min(1.05 / b, tavan)
    while f(ust) > 0.0:
        ust *= 1.05
        if ust >= tavan:
            raise InvalidInputError(
                f"b = {b!r} (M = {M!r}) icin donus noktasi YOK: foton "
                "yakalanir ve sonsuza donmez, yani 'bukulme acisi' orada "
                "tanimsizdir. Olculdu (GR, M = 1): b_krit/M = 3*sqrt(3) = "
                "5.196152422706632; b = 5.19 yakalanir, b = 5.20 gecer."
            )
    return brentq(f, 0.0, ust, xtol=1e-18, rtol=8.9e-16, maxiter=200)


def null_kernel(*, theory: Theory, M: float, b: float, gauge: str) -> NullKernel:
    """Bos jeodezigin donus noktasi + deflasyonlu cekirdegi.

    `b` CARPISMA PARAMETRESIDIR (`b = L/E`), en yakin gecis yaricapi DEGIL.
    Ayrim olculdu ve hukme girer: ayni ikinci mertebe katsayisi `r0`
    cinsinden `15*pi/4 - 4`, `b` cinsinden `15*pi/4`tur.
    """
    if not math.isfinite(b) or not (b > 0.0):
        raise InvalidInputError(
            f"b pozitif ve sonlu olmali, verilen: {b!r}. `b` carpisma "
            "parametresidir (b = L/E), en yakin gecis yaricapi degil."
        )
    pot = null_potentials(theory=theory, M=M, gauge=gauge)
    a0, a1, a2 = pot.time_poly
    s0, s1, _ = pot.source_poly
    ib2 = 1.0 / (b * b)
    N = (s0 * ib2, s1 * ib2, -a0, -a1, -a2)
    u0 = _en_kucuk_pozitif_kok(N, b=b, M=M)
    q3 = -N[4]
    q2 = u0 * q3 - N[3]
    q1 = u0 * q2 - N[2]
    q0 = u0 * q1 - N[1]
    return NullKernel(
        turning_point=u0,
        radial_poly=(q0, q1, q2, q3),
        time_poly=pot.time_poly,
        damp_poly=pot.damp_poly,
        constant_residual=u0 * q0 - N[0],
    )


def null_w(u, kernel: NullKernel):
    """Bos jeodezigin `w`si: `D(u)/Q(u)` = `(A(u)/T(u)) / Q(u)`.

    `radial_w` ve `isotropic_w` ile AYNI role sahiptir ve ayni bicimde
    tuketilir: integrand `sqrt(u0 * (1 + sin theta) * w(u))`.

    `u` bir `float` ya da `numpy.ndarray` olabilir; govde eleman elemandir.
    """
    return (
        evaluate_poly(kernel.time_poly, u) / evaluate_poly(kernel.damp_poly, u)
    ) / evaluate_poly(kernel.radial_poly, u)
