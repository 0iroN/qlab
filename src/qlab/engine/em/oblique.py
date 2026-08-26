"""Egik gelis -- Maxwell'in 1B'ye INDIRGENMIS hali (d/dy -> i*ky).

MODELLEME SECIMI, BEYAN EDILIR: enine dalga sayisi `ky` ANALITIK olarak
dayatilir; y yonunde izgara YOKTUR. x yonundeki cozum tam Yee'dir ve yansima
katsayisi COZULUR, yazilmaz. Bedeli acikca yazilir: y yonunde izgara
dispersiyonu bulunmadigi icin tam 2B kosumundan farklidir, ve bu fark bu
modulde OLCULMEDI (tam 2B ile karsilastirma yapilmadi).

    s-kutuplanma (TM_z) : Ez, Hy, Hx   -- E gelis duzlemine DIK
    p-kutuplanma (TE_z) : Hz, Ey, Ex   -- E gelis duzlemi ICINDE

BU DOSYA IKINCI BIR COZUCU DEGILDIR. Her iki kutuplanma da `yee._adim`in
KENDISINDEN gecer; `oblique.py`nin ekledigi tek sey, enine turevden gelen
`i*ky` terimi ve onu tasiyan UCUNCU alan bilesenidir. Bunu mumkun kilan sey
bir DUALITE degiskeni degistirmesidir:

    s :  (e, h) := ( Ez, +Hy )        yardimci alan: Hx,  isaret -1
    p :  (e, h) := ( Hz, -Ey )        yardimci alan: Ex,  isaret +1

p'deki eksi isareti kozmetik degil: `_adim`in H guncellemesi
`h <- da*h + db*(e[i+1]-e[i])`dir, oysa Ey'ninki
`ey <- cay*ey - cby*(hz[i+1]-hz[i])`. `h := -Ey` konunca ikisi AYNI satir olur
ve E guncellemesindeki `-(ey[1:]-ey[:-1])` de `+(h[1:]-h[:-1])`ye doner.
Yani p-kutuplanma icin ayri bir guncelleme YAZILMADI; ayni satir bir isaret
degiskeni degistirmesiyle kullanildi.

Sonuc olculebilir bir iddiadir ve testte kilitlidir: `ky = 0`da s-kutuplanma
kaydi `yee.run_1d`in kaydiyla BIT BIT aynidir.

BIRIM SECIMI `grid.FieldState`de bir kez beyan edildi; burada tekrar
edilmiyor. Alanlar KARMASIKTIR (`i*ky` gercel alanlari tasimaz);
`FieldState`in dtype'i bu yuzden kisitlanmamisti (bkz. grid.FieldState).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from qlab.engine.em.grid import FieldRecord, FieldState, Window, dx_kapisi
from qlab.engine.em.medium import Medium
from qlab.engine.em.yee import YeeCoefficients, _adim, yee_coefficients
from qlab.errors import InvalidInputError

#: Iki kutuplanma. Adlari `s`/`p` -- optik uzlasimi; `TM_z`/`TE_z` adlari
#: dokstringde birlikte yaziliyor cunku ikisi literaturde YER DEGISTIRIYOR.
POLARIZATIONS: tuple[str, ...] = ("s", "p")


def oblique_courant_limit(ky: float) -> float:
    """S_max = 1 / sqrt(1 + (ky/2)^2) -- INDIRGENMIS SEMANIN kendi ozelligi.

    SINIF: SEMA CIPASI -- cozucuyu dogrular, fizigi DEGIL. Sayi GIRILMEZ;
    sema kararli olmaktan cikar ve esik ikili aramayla OLCULUR.
    `grid.courant_limit`in egik indirgemedeki karsiligidir ve ONU GENELLER:
    `ky = 0`da tam olarak `courant_limit(1)` = 1 dondurur.

    NEREDEN GELIYOR: indirgenmis semada enine turev TAM (izgarasiz), x
    yonunde Yee'dir; buyutme analizi

        sin(omega dt / 2)^2 = S^2 [ sin(kx/2)^2 + (ky/2)^2 ]

    verir ve kararlilik sag tarafin HER desteklenen kx icin <= 1 olmasini
    ister. Degisen sey `max sin(kx/2)`dir:

      * sonsuz/periyodik izgarada 1     -> S_max = 1/sqrt(1 + (ky/2)^2)
      * SONLU PEC izgarada (L dugum) kx_m = m*pi/(L-1), m = 1..L-2 oldugu
        icin cos(pi/(2(L-1)))           -> S_max(L) =
                            1/sqrt(cos(pi/(2(L-1)))^2 + (ky/2)^2)

    Ikincisi `ky = 0`da tam olarak `1/cos(pi/(2(L-1)))`tir -- yani
    `grid.courant_limit`te olculen SONLU IZGARA PAYININ ta kendisi, ve bu
    fonksiyon onu `ky != 0`a TASIR.

    KAPI SONSUZ IZGARA FORMUNA KONUYOR ve bu bir SECIMDIR, olcum degil:
    `courant_limit`in 1/sqrt(d)'ye konmasiyla ayni statude -- kapi izgara
    boyutunu BILMEZ ve bilmemelidir. Bedeli olculdu ve 2e-04'un altindadir
    (asagidaki `olc/PEC` sutunu).

    OLCULDU -- SEVK EDILEN `oblique_step`, KAPI ATLANMIS (esik olcumu kapiyi
    atlamak ZORUNDADIR; `grid.courant_limit` ile ayni gerekce), vakum, PEC
    duvar, L = 201, 800 adim, tohumlu rastgele baslangic (tohum 3), ikili
    arama tol 1e-7, tespit olcutu `max|e| > 1e12 * baslangic`:

        ky           0.0         0.2         0.4         0.5554
        olculen      1.00019164  0.99522660  0.98076447  0.96372008
        kapali(inf)  1.00000000  0.99503719  0.98058068  0.96354216
        kapali(PEC)  1.00003084  0.99506758  0.98060976  0.96356975
        olc/PEC      1.0001608   1.0001598   1.0001578   1.0001560

        ky           0.8         1.2         2.0
        olculen      0.92864405  0.85764172  0.70722221
        kapali(inf)  0.92847669  0.85749293  0.70710678
        kapali(PEC)  0.92850138  0.85751237  0.70711769
        olc/PEC      1.0001537   1.0001508   1.0001478

    IKI KUTUPLANMA AYNI SAYIYI VERIYOR ve fark TAM SIFIRDIR (ky = 0.4 ve
    1.2'de olculdu) -- `ObliqueCoefficients.sign`in bir GAUGE oldugunun
    ikinci, bagimsiz kaniti.

    ARTIK ASIM IKI PARCADIR ve ikisi de olculdu -- `grid.courant_limit`teki
    ayrimin AYNISI:

    (a) SONLU IZGARA payi KAPALI FORMUN KENDISINDE. Cikarilinca kalan pay
        L'den BAGIMSIZ: ky = 0.8, 4000 adim, L = 101/201/401 icin
        5.252e-06 / 5.078e-06 / 4.926e-06.
    (b) SONLU KOSUM payi ADIMLA iniyor -- ve TEK BIR SAYI DEGILDIR: adim
        sayisi ile TESPIT OLCUTU BIRLIKTE belirler. ky = 0.8, L = 201:

            adim     olcut 1e6    olcut 1e12   fark        1e12 - kapali(PEC)
              800    0.92853445   0.92864406   1.096e-04   1.427e-04
             2000    0.92850592   0.92852269   1.677e-05   2.131e-05
             4000    0.92850238   0.92850643   4.046e-06   5.049e-06
            12000    0.92850147   0.92850189   4.213e-07   5.112e-07

        Yani TESPIT OLCUTU DE BIR PENCERE DUGMESIDIR -- ama ADIM ARTINCA
        ETKISI DE KAYBOLUYOR: iki sutun birlikte kapali forma iniyor.
        Olculen sonme ussu adim^-2.08 (iki bagimsiz aralikta 2.074 ve
        2.084), ve bu `grid.courant_limit`in IZOTROPIK olcumuyle ayni:
        orada 4000 -> 20000 adimda (5 kat) pay 5.99e-06 -> 2.12e-07, yani
        28.25 kat; burada 800 -> 4000 adimda (yine 5 kat) 1.427e-04 ->
        5.049e-06, yani 28.26 kat.

    BU DOSYADA BIR ZAMANLAR SEVK EDILMIS BASKA BIR TABLO VARDI (kaynak
    surulen `run_oblique_1d`, probda 1e6 MUTLAK esigi, 800 adim): ky = 0.2 /
    0.5554 / 1.2 icin 0.99511662 / 0.96361252 / 0.85755547. O SAYILAR
    YANLIS DEGIL, BASKA BIR PENCERENIN sayilari -- ve bu olculdu: ayni
    prosedur kapi atlanarak yeniden kosuldugunda 0.99511666 / 0.96361740 /
    0.85755547 (sonuncusu BIREBIR) cikiyor, ve tohumlu prosedur AYNI 1e6
    olcutuyle kosuldugunda 0.99510612 / 0.96360522 / 0.85754172 veriyor.
    Yani iki tablo arasindaki ~1.1e-04 SEMADAN DEGIL TESPIT OLCUTUNDEN
    geliyor. Yeni tablo secildi cunku tohumlu baslangic butun `kx`leri
    dogrudan uyarir ve `tests/em/test_sema_cipalari.py`nin ev usulu odur.

    TOHUM BIR DUGME DEGIL: ky = 0.8, L = 201, 4000 adim, tohum 1/3/7 icin
    0.92850612 / 0.92850643 / 0.92850618 (yayilim 3.1e-07).

    KILIT: tests/em/test_arayuz.py.
    """
    k = float(ky)
    if not np.isfinite(k):
        raise InvalidInputError(f"ky sonlu olmali, verilen: {ky}")
    return float(1.0 / np.sqrt(1.0 + (k / 2.0) ** 2))


@dataclass(frozen=True, eq=False)
class ObliqueCoefficients:
    """Egik indirgemenin katsayilari -- ve `_adim`e VERILECEK SLOT ATAMASI.

    `main` dogrudan `yee._adim`e gider. Hangi fiziksel alanin hangi slota
    oturdugu kutuplanmaya gore degisir (dualite; bkz. modul dokstringi):

        s : main = (ca, cb, da, db) = (Ez'nin, Ez'nin, Hy'nin, Hy'nin)
        p : main = (ca, cb, da, db) = (Hz'nin, Hz'nin, Ey'nin, Ey'nin)

    `aux_a`/`aux_b` ucuncu bilesenindir (s'te Hx, p'de Ex) ve `sign` iki yerde
    birden kullanilir -- yardimci alanin guncellemesinde ve ana alana geri
    beslenen `i*ky` teriminde. Tek isaret, cunku ikisi ayni curl'un iki
    yarisidir.

    BUNUN OLCULMUS SONUCU: `sign`in GLOBAL olarak cevrilmesi FIZIGI
    DEGISTIRMEZ. Iki yerde birden gectigi icin cevirme tam olarak
    `aux -> -aux` degisken degistirmesidir ve alanlar aynen kalir; mutasyon
    turunda p'nin isareti +1 -> -1 yapildiginda `tests/em/test_arayuz.py`de
    yalniz KATSAYI SLOTLARINI sayan transkripsiyon testi dustu, hicbir fizik
    testi dusmedi. Yani `sign` bir GAUGE'dir: yardimci alanin isaret secimi.
    Yazili olmasinin sebebi fizik degil OKUNABILIRLIK -- iki kutuplanmanin
    curl'lerini ayni satirda tutmak.
    """

    main: YeeCoefficients
    aux_a: np.ndarray
    aux_b: np.ndarray
    sign: float
    polarization: str
    ky: float


def oblique_coefficients(
    medium: Medium, *, courant: float, boundary, polarization: str, ky: float
) -> ObliqueCoefficients:
    """Egik katsayilar. COURANT DENETIMI `yee_coefficients` uzerinden kosar.

    `yee_coefficients` HER IKI kutuplanma icin de cagrilir -- p'de sonucun
    yalniz bir kismi kullanilsa bile. Sebep kapinin tek kaynaktan gecmesidir:
    ikinci bir kararlilik denetimi yazilsaydi iki denetim birbirinden
    sessizce ayrilabilirdi.

    `ky`NIN BIRIMI: RADYAN BOLU HUCRE. Bu bir ayrinti degil, bu dosyanin
    `dx` ile iliskisinin TAMAMI. Enine terim `cb * (i*ky) * aux` diye
    girer ve `cb = S/eps`tir; fiziksel karsiligi `(dt/eps) * i*k_y` oldugu
    icin `ky = k_y * dx`, yani ARGUMANIN KENDISI dimensiyonsuz gruptur.
    `oblique_coefficients` `dx`i GORMEZ ve `run_oblique_1d`in `dx`i bu
    terime HIC girmez -- bu yapisaldir, olculmus degil.

    KARARLILIK: `courant_limit(1)` kapisi `ky`yi HESABA KATMAZ, oysa gercek
    sinir `ky` ile DUSER. IKINCI BIR KAPI BU YUZDEN BURADADIR ve esigi
    `oblique_courant_limit(ky)`dir; kapali formun turetimi, olcumu ve artik
    asimin iki parcasi ORADA duruyor -- tek kaynak, burada tekrar edilmiyor.

    IKI KAPI DA CANLIDIR ve ikisi de ULASILABILIR; biri otekini OLU
    birakmiyor, cunku sirayla kosuyorlar ve aralari BOS DEGIL:
      * `yee_coefficients`in kapisi `S > 1`i keser (izotropik sinir),
      * bu kapi `oblique_courant_limit(ky) < S <= 1` BANDINI keser.
    `ky = 0`da band bostur (esik tam 1.0 = `courant_limit(1)`) ve bu bir
    kusur degil, iki kapinin ORADA CAKISTIGININ ta kendisidir. Kapinin bu
    SIRADA olmasi da bir secimdir: once konsaydi izotropik kapi hicbir zaman
    yeni bir sey reddetmezdi ve OLU bir kod yolu olurdu.

    BU PARAGRAF BIR ZAMANLAR BIR DURUST OLUMSUZ TASIYORDU: *"AMA KAPI BUNU
    SINAMAZ: `ky` yeterince buyutulurse `courant_limit(1)` gecer ve sema
    yine de patlar."* O cumle DOGRUYDU ve artik GECERSIZDIR -- kapi kondu.
    Neyi kestigi OLCULDU (L = 201, 800 adim, tohum 3, kapi atlanmis):
    `S` = 0.9 `ky` = 0'da KARARLIDIR (800 adimda buyume 1.163606) ve
    izotropik kapiyi HAKLI OLARAK gecer; ayni `S` `ky` = 1.2'de 800 adimda
    `|e|`yi 4.22e+216 katina cikariyor ve baslangicin 1e12 katini daha
    46. ADIMDA asiyor. `ky` = 2.0'da 20. adimda `nan`.

    `ky` ETIKETI BIR ZAMANLAR *"ky dx"* IDI ve o etiket, `dx` kapatilirken
    ILK KEZ sinanabilir hale gelip CURUDU: kapali formu `ky*dx` ile okumak
    `dx = 4`te 0.4 icin 0.78086881 ongorur, oysa AYNI pencerede olculen esik
    0.98092060'tir (ky = 0.8'de 0.52999894'e karsi 0.92879012) -- yani %26
    sapma. `ky` RADYAN BOLU HUCREdir ve bu fonksiyon `dx`i GORMEZ.

    SEVK EDILEN CAGRILARDA PAY BUYUK: gorulen en buyuk `ky` 0.555360'tir
    (lam = 16, theta = 45, eps1 = 4) ve oradaki esik 0.963542; kullanilan
    `S` = 0.4, yani pay 2.41 kattir. `ky` = 2.0 gibi bir uc degerde bile
    esik 0.707107 ve pay 1.77 kat -- yani kapi sevk edilen hicbir cagriyi
    kesmez.

    `ey_kural` BURADA ILK KEZ GERCEKTEN CAGRILIYOR. Task 1'in raporunda
    "1B TM'de olu kod" diye isaretlenmisti ve dogruydu: 1B TM'de yari hucrede
    E YOKTUR. p-kutuplanmada VARDIR (Ey), ve secim olculdu -- bkz.
    `medium.Medium`.
    """
    if polarization not in POLARIZATIONS:
        raise InvalidInputError(
            f"polarization {POLARIZATIONS} icinden olmali, verilen: "
            f"{polarization!r}"
        )
    if not np.isfinite(ky):
        raise InvalidInputError(f"ky sonlu olmali, verilen: {ky}")
    # IZOTROPIK KAPI ONCE (S > 1'i keser), EGIK KAPI SONRA (bandi keser).
    # Sira bir SECIMDIR ve gerekcesi dokstringde: ters cevrilirse izotropik
    # kapi hicbir zaman yeni bir sey reddetmez ve OLU kalir.
    base = yee_coefficients(medium, courant=courant, boundary=boundary)
    limit = oblique_courant_limit(ky)
    if courant > limit:
        raise InvalidInputError(
            f"S = {courant} EGIK indirgemenin kararlilik sinirini asiyor: "
            f"ky = {ky} icin S_max = {limit} (= 1/sqrt(1 + (ky/2)^2)). "
            "BU SINIR `courant_limit(1)` = 1'DEN KUCUKTUR ve o kapi "
            "DEVRALINAMAZ -- enine turev TAM (izgarasiz) oldugu icin `ky` "
            "payi dogrudan buyutme carpanina girer. OLCULDU (L = 201, 800 "
            "adim, tohum 3, kapi atlanmis): S = 0.9 ky = 0'da sema "
            "KARARLIDIR (buyume 1.163606) ve izotropik kapiyi hakli olarak "
            "gecer; AYNI S ky = 1.2'de |e|yi 4.22e+216 katina cikariyor ve "
            "1e12 katini 46. ADIMDA asiyor. Olculen esik ky = 0/0.2/0.4/"
            "0.5554/0.8/1.2/2.0 icin 1.00019164/0.99522660/0.98076447/"
            "0.96372008/0.92864405/0.85764172/0.70722221 ve kapali formun "
            "ustunde kalan pay her yerde 2e-04'un altinda. Ayrinti: "
            "oblique.oblique_courant_limit."
        )

    cells = medium.cells
    sigma = np.asarray(boundary.sigma_1d(cells), dtype=float)
    eps = medium.eps_r
    d_e = sigma * courant / (2.0 * eps)

    # TAM HUCREDEKI H (s'te Hx, p'de Hz). Manyetik kaybin ESLESME kosulu
    # sigma_m/mu = sigma_e/eps oldugu icin `d_h` burada `d_e`nin AYNISIDIR --
    # yari hucredeki gibi bir ortalama alinmaz, cunku ortalanacak bir sey yok.
    mag_a = (1.0 - d_e) / (1.0 + d_e)
    mag_b = (courant / medium.mu_r) / (1.0 + d_e)

    # YARI HUCREDEKI TEGET E (yalniz p'de var: Ey). `ey_kural` tam burada.
    eps_y = medium.eps_yari()
    sig_y = 0.5 * (sigma[:-1] + sigma[1:])
    d_ey = sig_y * courant / (2.0 * eps_y)
    el_ya = (1.0 - d_ey) / (1.0 + d_ey)
    el_yb = (courant / eps_y) / (1.0 + d_ey)

    if polarization == "s":
        main = base
        aux_a, aux_b, sign = mag_a, mag_b, -1.0
    else:
        main = YeeCoefficients(
            ca=mag_a, cb=mag_b, da=el_ya, db=el_yb, dim=1, courant=courant
        )
        aux_a, aux_b, sign = base.ca, base.cb, +1.0
    return ObliqueCoefficients(
        main=main,
        aux_a=aux_a,
        aux_b=aux_b,
        sign=sign,
        polarization=polarization,
        ky=float(ky),
    )


def oblique_step(
    state: FieldState, aux: np.ndarray, coeffs: ObliqueCoefficients
) -> tuple[FieldState, np.ndarray]:
    """Bir egik adim. UC SATIR, ve ortasi `yee._adim`IN KENDISI.

    SIRA (ve sirasi bir secimdir, beyan edilir): yardimci alan ESKI `e` ile
    guncellenir, sonra `_adim` kosar (once H, sonra E), en son `i*ky` terimi
    yeni `e`ye eklenir. Yardimci alan `_adim`den ONCE guncellenmeseydi
    yarim adim kayar ve sema Yee'nin ic ice gecmis zamanlamasini kaybederdi.

    Ucta E = 0 (PEC duvar) `_adim`de kosar; `i*ky` terimi yalniz [1:-1]e
    eklenir, dolayisiyla duvari BOZMAZ.
    """
    ky = coeffs.ky
    aux_yeni = coeffs.aux_a * aux + coeffs.sign * coeffs.aux_b * (
        1j * ky
    ) * state.e
    st = _adim(state, coeffs.main)
    st.e[1:-1] += coeffs.sign * coeffs.main.cb[1:-1] * (1j * ky) * aux_yeni[1:-1]
    return st, aux_yeni


def run_oblique_1d(
    medium: Medium,
    *,
    courant: float,
    boundary,
    steps: int,
    source: Callable[[float], float] | None,
    source_index: int | None,
    probes: tuple[int, ...],
    wavelength: float,
    sgf: float,
    ky: float,
    polarization: str,
    window_fn: str = "dikdortgen",
    dx: float = 1.0,
) -> FieldRecord:
    """Cok adimli egik kosum -- ve PENCERESINI dondurmek ZORUNDA.

    `run_1d` ile AYNI sozlesme: pencere varsayilansiz istenir, kaynak YUMUSAK
    NOKTA kaynagidir, ve KAYIT ONCE yapilir (e^n prob'a yazilir, sonra e^{n+1}
    uretilir). Kayit hizasi bir SECIMDIR ve burada onemi `run_1d`dekinden
    BUYUKTUR: bu dosyanin urettigi sayilar FAZ okumalaridir ve kayit sonra
    yapilsaydi butun seri bir adim kayardi.

    DONEN KAYIT hangi alandir: s'te Ez, p'de Hz. `final.e` de oyle; `final.h`
    ise s'te +Hy, p'de **-Ey**'dir (dualite; bkz. modul dokstringi). Bu bir
    ayrinti degil beyan: `final.h`yi Ey sanan biri isaret hatasi yapar.

    Yansiyan alan IKI KOSUMUN FARKINDAN okunur (yumusak kaynak iki yonlu
    isir); arayuzsuz referans kosumu cagiranin isidir -- bkz.
    `observable/em_obs.reflection_coefficient`.

    `dx != 1` BURADA DA REDDEDILIYOR (`grid.dx_kapisi`) ve BURADA BEDELI
    EN AGIR: bu dosyada IKI enine buyukluk var ve `dx != 1`de AYRI
    BIRIMLERE dusuyorlar -- `ky` radyan/HUCRE (bkz. `oblique_coefficients`),
    kaynagin saati ise `t = n*S*dx` ile UZUNLUK basina. Sonuc BIR ACI
    HATASIDIR ve OLCULDU. Pencere: `em_obs.ObliqueWindow` varsayilanlari
    (N = 3600, npml = 60, S = 0.4, iface = 1500, kaynak 600, prob 1494,
    sgf = 9, dikdortgen), eps1 = 4, eps2 = 1, BEYAN EDILEN theta = 45 der,
    ve `wavelength` dx ile birlikte olceklenerek izgara cozunurlugu
    lambda = 16 hucrede SABIT tutuldu; adim sayisi ucunde de 10040:

        dx     W     ky          etkin theta   olculen |rho_s|
        1      16    0.55536037  45.000000     1.00000650
        2      32    0.27768018  20.704811     0.47632473
        0.5    8     1.11072073  YOK           1.03208644

    theta = 45'te (eps1 = 4, eps2 = 1) TAM IC YANSIMA vardir ve `|rho| = 1`
    ZORUNLUDUR. `dx = 2`de olculen sayi 0.47632473 -- Fresnel'in 45 der
    icin dedigi 1.00000000'a 0.5237 uzakta, 20.704811 der icin dedigi
    0.45141623'e 0.0249 uzakta, yani 21 KAT DAHA YAKIN: cozucu BASKA BIR
    ACININ sorusunu cevapliyordu. `dx = 0.5`te ise beyan edilen aci hic YOK
    (sin(theta) = 1.41421 > 1) ve olculen `|rho_s|` 1.03208644, yani 1'in
    USTUNDE -- fiziksel olarak IMKANSIZ. `t = n*S*dx` bu yuzden `t = n*S`
    oldu; `* dx` carpani KALDIRILDI.

    DURUST OLUMSUZ -- YUKARIDAKI UC SATIRDA HATA "SESSIZ" DEGILDI: sevk
    edilen `reflection_coefficient` bu yapilandirmada `|rho| = 1` kapisini
    KOSAR (`total_internal_reflection(4, 1, 45)` True, `unity_gate` 1e-2) ve
    ||rho|-1| = 0.5237 ile 1.0321'i ikisini de KESER, yani `None` doner.
    Kapi yalniz TIY'de ve aynada kosar: KISMI YANSIMADA (ornegin ayni
    ortamlarda beyan edilen theta = 20 der, `total_internal_reflection`
    False) hicbir kapi yoktur ve yanlis sayi SESSIZCE geri doner. Ustelik
    `em_obs` cozucuye `dx` HIC gecirmez -- yani bu yol pratikte de hic
    acilmamisti. Kapinin gerekcesi bu satirlar degil, `grid.dx_kapisi`daki
    BIT BIT cokme olcumudur.
    """
    dx_kapisi(dx)
    if steps < 0:
        raise InvalidInputError(f"steps negatif olamaz, verilen: {steps}")
    if (source is None) != (source_index is None):
        raise InvalidInputError(
            "source ve source_index birlikte verilir ya da birlikte None olur"
        )
    cells = medium.cells
    for p in probes:
        if not 0 <= p < cells:
            raise InvalidInputError(f"prob indisi izgara disinda: {p} (cells={cells})")
    if source_index is not None and not 0 < source_index < cells - 1:
        raise InvalidInputError(
            f"kaynak indisi PEC duvarlarin ARASINDA olmali: {source_index}"
        )
    coeffs = oblique_coefficients(
        medium, courant=courant, boundary=boundary, polarization=polarization, ky=ky
    )
    st = FieldState.zeros_1d(cells, dtype=complex)
    aux = np.zeros(cells, dtype=complex)
    pidx = np.asarray(probes, dtype=int)
    kayit = np.zeros((len(probes), steps), dtype=complex)
    for n in range(steps):
        kayit[:, n] = st.e[pidx]
        st, aux = oblique_step(st, aux, coeffs)
        if source is not None:
            st.e[source_index] += coeffs.main.cb[source_index] * source(
                n * courant
            )
    return FieldRecord(
        probes=kayit,
        probe_indices=tuple(int(p) for p in probes),
        window=Window(
            cells=cells,
            dx=dx,
            courant=courant,
            steps=steps,
            wavelength=wavelength,
            npml=int(getattr(boundary, "thickness", 0)),
            sgf=sgf,
            window_fn=window_fn,
        ),
        final=st,
    )
