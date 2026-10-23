"""Egik gelis -- Maxwell'in 1B'ye INDIRGENMIS hali (d/dy -> i*ky).

MODELLEME SECIMI, BEYAN EDILIR: enine dalga sayisi `ky` ANALITIK olarak
dayatilir; y yonunde izgara YOKTUR. x yonundeki cozum tam Yee'dir ve yansima
katsayisi COZULUR, yazilmaz. Bedeli tam 2B ile OLCULDU (G1: s / TM_z kolu
09.10.2026, p / TE_z kolu 10.10.2026 dualite yoluyla; kod DUZELTILMEDI):

  * KIMLIK: y-degismez ortamda motorun 2B TM_z Yee'si (`yee.run_2d`) bu
    modulun s cozumunun `ky -> 2 sin(ky/2)` konmus halidir, BIT-YAKIN.
    Yayilan kiplerde (tasiyicida K < sqrt(eps_kaynak) Om) zaman kaydinda
    tepeye gore <= 1.5e-14: eps 1->2.25, S 0.4 en buyuk 1.414e-14; 4->1,
    S 0.6, m = 1, 2'de 5.64e-15. eps 1->4 (S 0.5, 50 der) yalniz TEK
    FREKANSTA olculdu: |rho_2B - rho_egik(K)| = 6.6e-16 (tepeye gore
    degil). 4->1'de m = 3 kismen sonumlu (tasiyicida K 1.111 > 0.895, ama
    genis bantli darbenin ust kuyrugu yayilir; tepe 1.5e-4): 2.9e-13; tam
    sonumlu m >= 4 (tepe ~1e-8) bagil 2.8e-9.
  * GENLIK BEDELI, ayni omega ve ayni ANALITIK ky ile: |r(ky)| -
    |r(2 sin(ky/2))|. Karmasik fark DEGILDIR (kismi yansimada isareti ters,
    bantta buyuklugu farkli); FAZ BEDELI ayri bir buyukluktur ve yalniz
    referans duzlemiyle yazilir (asagida TIY maddesi). Kapali form, arayuz
    yuzune (I-1/2) referansli: r_s = sin((k1-k2)/2) / sin((k1+k2)/2),
    sin(k_j/2) = sqrt(eps_j Om^2 - K^2)/2, Om = 2 sin(omega S/2)/S. Sabit
    acida O(dx^2); sevk edilen aletle (eps 1 -> 2.25, lam 20, S 0.4)
    30/45/60 der: 1.00e-4 / 6.62e-4 / 3.13e-3.
  * KRITIK ACI BANDI (n1 > n2): iki semanin kritik acisi AYRILIR; bantta bu
    modul |r| = 1, 2B kismi yansima verir. 2.25 -> 1, lam 20: bant 0.21 der,
    bedel kapali formda 0.147'ye kadar (cozucuyle 41.8 der'de 0.140), yalniz
    O(dx) (lam 10 -> 160 usler 0.82 -> 0.98). "O(dx^2)" noktasaldir.
  * TIY FAZ BEDELI yalniz REFERANS DUZLEMIYLE yazilir (iki semanin k1'i
    farkli): 2.25 -> 1, 60 der, lam 20.78'de |arg(r/r_2B)| arayuz yuzunde
    1.99 der, prob duzleminde 4.73 der; `reflection_coefficient` prob
    referansli okur.
  * Surekli Fresnel'e hangi semanin yakin oldugu aciya gore isaret
    degistirir (kapali form: 1 -> 2.25'te ~75, 1 -> 4'te ~80 der); bir
    "daha dogru sema" hukmu VERILMEZ.
  * p (TE_z) KOLU (G1-p, 10.10.2026): motorun 2B yolu yalniz TM_z'dir;
    TE_z ona DUAL ORTAM verilerek kosulur -- eps' = 1, mu' = eps,
    sogurucuda sigma' = sigma/eps, (e, hx, hy) = (Hz, -Ex, -Ey). Bu, dual
    izgarada bir 2B TE_z Yee'dir (Ex/Ey'nin eps'i `Medium.mu_yari_2d`, hep
    aritmetik; testte elle yazilmis yerli TE_z ile tam 2B rastgele eps'te
    bit bit ayni). KIMLIK: y-degismez ortamda dual 2B, bu modulun p
    cozumunun `ky -> 2 sin(ky/2)` halidir; yayilan kiplerde tepeye gore
    <= 2.3e-14 (yedi duzenek, eps 1..9, S 0.3-0.6, PML ve PEC). Egik p ile
    kimligin kosullari (TE_z olmanin degil):
    mu = 1; `ey_kural` = `mu_yari_2d`nin kurali (sevk: aritmetik;
    harmonik/sol/sag ile >= 2.7e-2 kirilir); sogurucu icinde eps sabit
    (rampada 9e-7..6e-5 sapar); duvar ve ic maske PMC -- TE_z'de PEC cisim
    bu yolla KURULAMAZ. Kapali form (aritmetik Ey, I-1/2 referansli):
    r_p = (eps2 a1 c2 - eps1 a2 c1) / (eps2 a1 c2 + eps1 a2 c1), a_j ve
    c_j = cos(k_j/2) s'tekiyle ayni. HER `ey_kural` ICIN GENEL FORM (bilgi
    notu, 11.10.2026; KILITSIZ -- testte yalniz aritmetik form elle yazili):
    ea arayuz yari-hucresinin kuraldan gelen eps'i, delta = ea -
    (eps1+eps2)/2 iken Q = (eps2 a1 c2 - 2i delta a1 a2) / (eps1 a2 c1),
    r_p = (Q-1)/(Q+1); delta = 0 (aritmetik) ustteki forma iner. Sevk aleti
    (`reflection_coefficient`, p, T2p penceresi yalniz `ey_kural` degisik)
    dort kuralda da |r|'de <= 1.5e-7 oturur (1 -> 2.25 40 der lam 10,
    1 -> 6.25 30 der lam 12, 2.25 -> 1 20 der lam 20); aritmetik form
    aritmetik-disi kurallarda 2.5e-3..0.21 sapar.
    Genlik bedelinin ISARETI Brewster'de
    doner (sevk aleti, 1 -> 2.25, lam 20, 30/45/60 der: -1.05e-4 /
    -7.28e-4 / +3.82e-3; s'de hep pozitif). Kritik aci bandi s ile ORTAK
    (dispersiyon ayni); bantta p'nin bedel tepesi 0.303 (2.25 -> 1, lam 20;
    s'de 0.147; lam -> sonsuzda oran eps1/eps2), yalniz O(dx). TIY'de
    delta_ps bedeli referans duzleminden BAGIMSIZDIR (iki kutuplanma ayni
    semada ayni k1): 4 -> 1, 45 der, lam 20'de -0.390 der (boru hatti
    uzlasimi). KILIT: `tests/em/test_g1p_egik_2b_te.py` -- T0p yerli TE_z,
    T1p kimlik, T2p kapali form + isaret (hizli); T3p dogrudan 2B TIY
    delta_ps sureklige O(dx^2), 2B TE_z yolunun FIZIK CIPASI (`slow`).
  * KILIT, `tests/em/test_g1_egik_2b.py`:
    `test_2B_YEE_y_KIPI_EGIK_s_NIN_K_2SIN_HALI_BIT_YAKIN` (T1, kimlik,
    zaman kaydi), `test_EGIK_BEDELI_AYRIK_FRESNEL_KAPALI_FORMUNA_OTURUR`
    (T2, genlik bedeli) ve
    `test_KRITIK_ACI_BANDINDA_SEMALAR_AYRI_TARAFTA_ve_BEDEL_O_dx` (T3,
    kritik bant, `slow`). r uzerindeki kilit (T2, T3) YALNIZ |r|'dir: faz
    sayilari (TIY maddesi) ve kapali formun isareti KILITSIZDIR (kapali
    formun payinda isaret cevrilince uc test de gecer; dogrulayici olctu).
    Faz kapisi eklenmedi: T2 penceresinde faz protokol hatasi olculmedi.

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
from scipy.linalg import eigvalsh_tridiagonal

from qlab.engine.em.grid import FieldRecord, FieldState, Window, dx_kapisi
from qlab.engine.em.medium import Medium, _kurucudan_indirgeme, _salt_okunur_kopya
from qlab.engine.em.yee import YeeCoefficients, _adim, _etkin_courant_kapisi, yee_coefficients
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

    KAPI TIPIN KENDISINDEDIR (G13, 12.10.2026): `__post_init__` her kurulusta
    `_katsayi_kapisi`ni kosar, yani `dataclasses.replace` ve dogrudan kurulus
    da kapidan gecer. ESKIDEN kapilar yalniz `oblique_coefficients`teydi;
    OLCULDU (HEAD e262cab): vakum L = 201, PEC, S = 0.9, ky = 0'in katsayilari
    `replace(co, ky=1.2)` ile kapisiz kuruluyor ve `oblique_step` tohumlu
    baslangici 45. adimda 1.1e+12 katina cikariyordu (ayni S ve ky dogrudan
    istenince REDDEDILIYOR). Kapi ortami gormez; sinadigi sey ADIMLANAN
    DIZILERDIR. Kapiyi atlamak zorunda olan esik olcumu
    (`tests/em/test_arayuz.py`) tipi `object.__new__` ile kurar -- acik ve yerel.
    """

    main: YeeCoefficients
    aux_a: np.ndarray
    aux_b: np.ndarray
    sign: float
    polarization: str
    ky: float

    def __post_init__(self) -> None:
        # Kapi kurulusta kosar; sonradan YERINDE yazma onu atliyordu (C2, grup 9; olcum
        # `yee.YeeCoefficients` dokstringinde: `co.main.cb[:] *= 1.5` 33. adimda 1e12,
        # `co.aux_b[:] *= -1` p, ky = 0.8, PML 15'te 76. adimda 1e12). `main` kendi
        # `__post_init__`iyle salt okunurdur; yardimci yuvalar burada kopyalanir.
        if not isinstance(self.main, YeeCoefficients):
            raise InvalidInputError(
                f"`main` bir YeeCoefficients olmali (tur kapisi ve salt okunur diziler), "
                f"verilen: {type(self.main).__name__}"
            )
        for ad in ("aux_a", "aux_b"):
            object.__setattr__(self, ad, _salt_okunur_kopya(getattr(self, ad), ad))
        _katsayi_kapisi(self)

    def __reduce__(self):
        return _kurucudan_indirgeme(self)


#: `_katsayi_kapisi`nin kayan nokta payi: S^2 lambda_max <= 4 (1 + pay). Fabrikanin
#: en buyuk kabul ettigi S'de kurdugu katsayilarda olculen S^2 lambda_max / 4 - 1 en
#: cok 4.4e-16 idi (asagidaki olcum); pay bundan ~2e4 kat buyuk, S'de 5e-12.
_KATSAYI_KAPISI_PAYI = 1e-11


def _katsayi_kapisi(co: ObliqueCoefficients) -> None:
    """ADIMLANAN DIZILERIN kararlilik kapisi -- `ObliqueCoefficients`in kendi kapisi.

    `oblique_step` iki ikinci-mertebe fark verir: e^{n+1} - 2 e^n + e^{n-1} =
    -M e^n, M = diag(cb) [D^T diag(db) D + ky^2 diag(aux_b)] (ic dugumler; PEC
    duvar uclari sabitler). Kayipli yuvalarda kayipsiz katsayi `2 c / (1 + a)`
    ile geri okunur (c = cb, db, aux_b; a = ca, da, aux_a): yari-adim ustel
    formunda tam olarak S/eps'tir. Kararlilik M'nin ozdegerlerinin [0, 4]te
    olmasidir; cb^{1/2} K cb^{1/2} simetrik uc-kosegen oldugu icin lambda_max
    `eigvalsh_tridiagonal` ile alinir. Bu `_egik_p_kesin_esik`in operatorunun
    AYNISIDIR, ama ORTAMDAN degil KATSAYILARDAN kurulur ve sonlu izgara payini
    tasir (fabrikanin politika kapilarindan GEVSEK, asla siki degil:
    fabrikanin kabul ettigi her katsayi buradan gecer -- s'de Titu siniri, p'de
    ayni operator). Ayrica: kutuplanma, sonlu ky, sign = +-1, sekiller ve
    `main.courant <= oblique_courant_limit(ky)` (fabrikanin egik politika
    kapisi; `replace(ky=...)` ayni hukmu alir).

    SONUM YUVALARI -- KAZANC KAPISI (G13 duzeltmesi, 12.10.2026): `ca`, `da`,
    `aux_a` her yerde |a| <= 1 olmali. Geri okuma ANCAK o zaman bir kayipli
    ortamdir: a, (-1, 1]'de her (a, b) cifti negatif olmayan bir kayip d = (1 -
    a)/(1 + a) ve c = 2 b/(1 + a) tasir; a > 1 NEGATIF kayiptir (kazanc) ve
    ozdeger okumasi onu goremez (2 c/(1 + a) yalniz kuculur). Fabrika (1 - d)/
    (1 + d), d >= 0 ile |a| <= 1'i TAM kurar (|1 - d| <= 1 + d, yuvarlama
    monoton): sahte ret yok. OLCULDU (kapiyi `object.__new__` ile atlayan
    `oblique_step`, 800 adim, tohumlu baslangic): vakum L = 201, PEC, S = 0.9,
    s, ky = 0'da `ca` x 1.02 -> 4.1e+03 kat, `da` x 1.02 -> 4.1e+03 kat; p, ky =
    0.8, PML 15'te `aux_a` x 1.05 -> 1e12 ustu; `ca` = -3 ile `cb` = 0.45 (1 +
    ca) (geri okunan 2 cb/(1 + ca) = 0.9 > 0) -> 1e12 ustu. Kazanc kapisindan
    ONCE dordu de bu kapidan geciyordu.

    ESIGIN NE KADAR KESIN OLDUGU: tekdize kayipta (L = 41, d = 0.3 / 1 / 3; ky =
    0'da s, ky = 0.8'de s ve p) bu kapinin esiginin 0.99'u soner, 1.01'i buyur -- yani
    orada kayipli semanin esigidir. ky = 0.8 hucresi aux yuvasinin geri
    okumasini da sinar: ham `aux_b` okunsaydi esik 1.036 kat gevserdi ve 1.01
    gecerdi. Tekdize OLMAYAN kayip (PML) icin ISPAT yazilmadi ama OLCULDU (C3,
    grup 9, 14.10.2026; bagimsiz yontem: sevk edilen `oblique_step`in tek adim
    buyutme matrisinin spektral yaricapi, politika kapilari kapali): PML 15 /
    28 / 40 ve LossyLayer 15, L = 61 / 121, s ve p, ky 0 - 0.8, vakum, eps 1|4
    ve eps 4|0.25 x mu 0.25|4 (harmonik / sag) -- kayipli semanin GERCEK esigi
    bu kapinin esiginin en cok 1 + 1.3e-10 kati (ikiye bolme cozunurlugunde
    esit); 0.99 / 0.999 katinda rho - 1 <= 2e-14 ve 20000 adim sinirli, 1.001
    katinda rho - 1 = 0.067 - 0.094 ve patliyor (V10, 23.10.2026, olcut yazili: L 61,
    tohum 3'lu rasgele baslangic, |E| maksimumu 1e12 katini 297 - 419., kayan nokta
    tasmasini 7904 - 10973. adimda asar; 14.10'un "5100 - 7100. adim" sayisinin olcutu
    yazilmamisti ve yeniden uretilemedi). Yani PML'de
    de kapi tam esiktir: ne gevsek ne asiri temkinli. KAYAN NOKTA PAYININ
    BEDELI (ayni olcum): kapinin KABUL SINIRINDA (S^2 lambda = 4 (1 + 1e-11))
    rho - 1 en cok 6.3e-6 (tepe kipi kayipsiz bolgedeyse; PML'de sondugunde
    1e-9 - 1e-10) -- payin karekoku mertebesi; fabrika ciktisi bu sinira
    yaklasmaz (asagidaki 4.4e-16). Kilit: `tests/em/test_g9_kopya_ve_tur_kapisi.py`
    (`test_C3_...`; pay 1e-9 ve 1e-6 mutantlarini oldurur).

    OLCULDU (12.10.2026): fabrikanin p kapisinin TAM ESIKTE kurdugu katsayilarda
    (`test_EGIK_P_KESIN_KAPI_...` ortamlari, dort kural x ky = 0 / 0.8 + ikinci
    capa) S^2 lambda_max / 4 - 1 <= 2.2e-16; 3000 rastgele ortamda (L 8-80,
    eps ve yarisinda mu 10^U(-1, 1.3), dort kural, ky 0-2, s/p, PEC/PML)
    fabrikanin kabul ettigi EN BUYUK S'de 0 ret (kazanc kapisi eklenince yeniden
    kosuldu: yine 0), en cok 4.4e-16. Maliyet:
    N = 3600, PML 60 kurulusu 1.76 ms, bunun 0.91 ms'i bu kapi. Kayipli esigin
    kayipsizina esit oldugu `_egik_p_kesin_esik` dokstringinde olculmustur.
    """
    if co.polarization not in POLARIZATIONS:
        raise InvalidInputError(
            f"polarization {POLARIZATIONS} icinden olmali, verilen: {co.polarization!r}"
        )
    if not np.isfinite(co.ky):
        raise InvalidInputError(f"ky sonlu olmali, verilen: {co.ky}")
    if co.sign not in (-1.0, 1.0):
        raise InvalidInputError(f"sign +1 ya da -1 olmali, verilen: {co.sign!r}")
    ana = co.main
    n = int(np.shape(ana.ca)[0]) if np.ndim(ana.ca) == 1 else -1
    sekiller = (np.shape(ana.ca), np.shape(ana.cb), np.shape(co.aux_a), np.shape(co.aux_b),
                np.shape(ana.da), np.shape(ana.db))
    if ana.dim != 1 or n < 3 or sekiller != ((n,),) * 4 + ((n - 1,),) * 2:
        raise InvalidInputError(
            f"egik katsayi sekilleri tutarsiz (ca, cb, aux_a, aux_b, da, db): {sekiller}, "
            f"dim = {ana.dim}"
        )
    limit = oblique_courant_limit(co.ky)
    if not ana.courant <= limit:
        raise InvalidInputError(
            f"S = {ana.courant!r} EGIK indirgemenin kararlilik sinirini asiyor: ky = "
            f"{co.ky} icin S_max = {limit} (katsayi kapisi; bkz. oblique_coefficients)."
        )
    for ad, sonum in (("ca", ana.ca), ("da", ana.da), ("aux_a", co.aux_a)):
        with np.errstate(invalid="ignore"):
            buyuk = float(np.max(np.abs(np.asarray(sonum))))
        if not buyuk <= 1.0:
            raise InvalidInputError(
                f"egik katsayi `{ad}` KAZANC veriyor: her yerde |{ad}| <= 1 olmali, en buyugu "
                f"{buyuk!r} (fabrika (1 - d)/(1 + d), d >= 0 ile tam olarak <= 1 kurar; > 1 "
                "negatif kayiptir) -- kapi kapali yonde reddeder."
            )
    with np.errstate(all="ignore"):
        e0 = 2.0 * np.asarray(ana.cb) / (1.0 + np.asarray(ana.ca))
        h0 = 2.0 * np.asarray(ana.db) / (1.0 + np.asarray(ana.da))
        y0 = 2.0 * np.asarray(co.aux_b) / (1.0 + np.asarray(co.aux_a))
        ky2 = float(co.ky) * float(co.ky)
        d = e0[1:-1] * (h0[:-1] + h0[1:] + ky2 * y0[1:-1])
        kok = np.sqrt(e0[1:-1])
        e = -kok[:-1] * kok[1:] * h0[1:-1]
    gecerli = all(np.all(np.isfinite(x)) and np.all(np.real(x) > 0.0) for x in (e0, h0, y0))
    lam = float("nan")
    if gecerli and np.isrealobj(d):
        try:
            lam = float(eigvalsh_tridiagonal(d, e, select="i", select_range=(d.size - 1, d.size - 1))[0])
        except (ValueError, np.linalg.LinAlgError):
            pass
    if not np.isfinite(lam):
        raise InvalidInputError(
            "egik katsayilar sonlu ve pozitif bir kayipsiz operator vermiyor (cb, db, aux_b "
            "ve 1 + ca, 1 + da, 1 + aux_a > 0 olmali) -- kapi kapali yonde reddeder."
        )
    if lam > 4.0 * (1.0 + _KATSAYI_KAPISI_PAYI):
        raise InvalidInputError(
            f"egik katsayilar KARARSIZ: S^2 lambda_max = {lam!r} > 4 (adimlanan dizilerden "
            f"geri okunan operator; etkin S/S_krit = {float(np.sqrt(lam / 4.0))!r}). Kapi tipin "
            "kendisindedir: `dataclasses.replace` ya da dogrudan kurulus da buradan gecer."
        )


def _egik_p_kesin_esik(medium: Medium, ky: float) -> float:
    """Egik p semasinin KESIN kararlilik esigi -- kayipsiz ayrik operatorun ozdegeri.

    p'de Hz tam hucrede `mu_r` ile, Ey yari hucrede `eps_yari` (`ey_kural`) ile
    eslenir; hucre carpimi kapisi (`yee._etkin_courant_kapisi`) bu ESLESMEYI
    goremez. Ic dugumlerde (PEC duvar e[0], e[-1]'i sabitler) w = mu_r,
    K = D^T diag(1/eps_yari) D + ky^2 diag(1/eps_r); W^-1/2 K W^-1/2 simetrik
    uc-kosegendir ve S_krit = 2/sqrt(lambda_max). OLCULDU 02.10.2026, uc
    bagimsiz yol: tek adim buyutme matrisiyle 8960 vakada bagil fark <= 5.5e-13
    (ikiye bolme aracinin tek yonlu payi dahil);
    zaman adimlamasiyla 22/22 (artik pay adim^-2.08 ile iner); PML ve LossyLayer'da
    kayipli esik BUNA ESIT (CN ortalamali kayip lambda = -1'de duser -- kayip pay
    KAZANDIRMAZ, operator butun izgara uzerinden kurulur). s'de GEREKMEZ: H yari
    hucrede HEP aritmetik `mu_yari` ile oturur ve hucre carpimi kapisi yeter
    (Titu siniri; binlerce rastgele ortamda 0 siki). Esik sonlu izgaranin payini
    TASIR (vakumda 1/cos(pi/(2(L-1))) kadar), bu yuzden oteki kapilarin YERINE
    degil YANINA konur -- yerine konsaydi o payin icindeki bugunku retler (orn.
    test_yee ETKIN, p, ky = 0.8) kabule donerdi. Ortam ~1e+-150 olcegindeyse
    hesap tasar ya da coker: kapi KAPALI yonde InvalidInputError atar (hucre
    carpimi kapisinin alt tasma kuralinin esi; c4c66ad bunlari kabul ediyordu).
    """
    w = np.sqrt(medium.mu_r[1:-1])
    c = medium.eps_yari()
    with np.errstate(all="ignore"):
        d = (1.0 / c[:-1] + 1.0 / c[1:] + ky * ky / medium.eps_r[1:-1]) / (w * w)
        e = -1.0 / (w[:-1] * w[1:] * c[1:-1])
    n = d.size
    try:
        lam = float(eigvalsh_tridiagonal(d, e, select="i", select_range=(n - 1, n - 1))[0])
    except (ValueError, np.linalg.LinAlgError):
        lam = float("nan")
    if not (np.isfinite(lam) and lam > 0.0):
        raise InvalidInputError(
            f"egik p kesin kararlilik esigi bu ortam/ky olceginde HESAPLANAMIYOR "
            f"(lambda_max = {lam!r}; tasma ya da alt tasma) -- kapi kapali yonde reddeder."
        )
    return 2.0 / float(np.sqrt(lam))


def oblique_coefficients(
    medium: Medium, *, courant: float, boundary, polarization: str, ky: float
) -> ObliqueCoefficients:
    """Egik katsayilar. COURANT DENETIMI `yee_coefficients` uzerinden kosar.

    `yee_coefficients` HER IKI kutuplanma icin de cagrilir -- p'de sonucun
    yalniz bir kismi kullanilsa bile. Sebep kapinin tek kaynaktan gecmesidir:
    ikinci bir kararlilik denetimi yazilsaydi iki denetim birbirinden
    sessizce ayrilabilirdi. BILINCLI ISTISNA (02.10.2026): p'deki kesin kapi
    (`_egik_p_kesin_esik`) operatoru katsayilardan degil ORTAMDAN yeniden kurar,
    yani semanin yuvalari degisirse sessizce ayrilabilir. Bekcisi
    `test_EGIK_P_KESIN_KAPI_ESLESMEYI_KESER_ve_TAM_ESIGE_OTURUR`in ky = 0.8 dinamik
    capasidir: kapi atlaninca esigin 1.001'i patlamali, 0.999'u kararli kalmali.

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
    yeni bir sey reddetmezdi ve OLU bir kod yolu olurdu. 01-02.10.2026'da iki
    kapi daha geldi ve ikisi de CANLI: etkin kapi (eps_r*mu_r < 1) duzgun eps 0.25'te
    S in (0.464238, 0.464251] bandini (ky = 0.8, L = 200) YALNIZ kendisi keser; p'deki kesin kapi
    (`_egik_p_kesin_esik`) her hucrede eps*mu = 1 olan eps 4|0.25, mu 0.25|4,
    L = 200, ky = 0 ortaminda harmonikte (0.627587, 1], sagda (0.470588, 1]
    bandini YALNIZ kendisi keser.

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
    # Ayni sira ETKIN kapida da (01.10.2026): `base` 1B'nin etkin kapisini kosturdu,
    # bu satir eps_r*mu_r < 1'de egik BANDI keser (olculdu ky = 0.8, s ve p, eps ya da
    # mu 0.25: esik 0.464255 = 0.5 * S_max(ky) * 1.000036). p'deki mu_r / eps_yari
    # ESLESMESINI goremez -- onu asagidaki kesin kapi keser (02.10.2026).
    _etkin_courant_kapisi(medium, courant, limit, f"EGIK (ky = {ky})")
    if polarization == "p":
        esik = _egik_p_kesin_esik(medium, float(ky))
        if courant > esik:
            raise InvalidInputError(
                f"S = {courant!r} EGIK p semasinin KESIN kararlilik esigini ({esik!r}) "
                f"asiyor (ky = {ky}): Hz tam hucrede mu_r, Ey yari hucrede eps_yari ile "
                "eslenir ve hucre carpimi kapisi bu eslesmeyi goremez. Ayrinti: "
                "oblique._egik_p_kesin_esik."
            )

    cells = medium.cells
    sigma = np.asarray(boundary.sigma_1d(cells), dtype=float)
    eps = medium.eps_r
    d_e = sigma * courant / (2.0 * eps)

    # TAM HUCREDEKI H (s'te Hx, p'de Hz). Manyetik kaybin ESLESME kosulu
    # sigma_m/mu = sigma_e/eps oldugu icin `d_h` burada `d_e`nin AYNISIDIR --
    # yari hucredeki gibi bir ortalama alinmaz, cunku ortalanacak bir sey yok.
    # (28.09) ESLESME her mu'da korunur (tum kayip hizlari sigma/eps); degisen PML SONUMU:
    # sigma*sqrt(mu/eps) ile olceklenir, sigma_max tasarimi eta = 1 (olculdu, duz ortam: eps = 4
    # -> yansima 7.2e-4, mu = 4 -> 6.2e-7 ayriklik tabani). Dualite eta'yi 1/eta yapar: p'deki
    # mu = M, s'deki eps = M ikizinden farkli sonum gorur (M 1.0654: 12376. adimdan sonra bagil
    # 1.9e-3). Yalniz Hz'i mu'dan kurmak eslesmeyi BOZAR (6.2e-7 -> 9.0e-7); tum p hizlari
    # mu'dan kurulursa ikizin yansimasi (7.2e-4) gelir. Sonuma bagli olan s<->p ikiz esdegerligi.
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
    False) bir imkan kapisi YOKTU ve `dx != 1`in yanlis sayisi SESSIZCE geri
    donerdi. (11.10.2026: bu yol artik yukaridaki `dx_kapisi` ile GIRISTE
    kapalidir; kismi yansimada |rho| < 1 fizigin kendisidir, imkan kapisi
    gerekmez. Kismi yansimada OLCULEN yanlis sayi kaynagi -- KAYIT UZUNLUGU
    -- `reflection_coefficient`in KUYRUK ve IKINCI VARIS kapilarindadir
    (10.10.2026; kuyruk kapisi kaydin tamligini, ikinci varis kapisi kayit
    ici sogurucu yankisini sinar), cevaplanan acinin
    dogrulugunu degil -- buradaki aci hatasi `dx`ten geliyordu.) Ustelik
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
