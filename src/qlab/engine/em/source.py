"""Darbeler. `sgf` PENCERENIN ADLI PARCASI -- ve IKI DARBEDE IKI AYRI SEY.

BU DOSYANIN EN ONEMLI CUMLESI: bu modulde iki darbe var ve `sgf` ikisinde
AYNI SEYI YAPMAZ.

    gauss_cos : sg = sgf * wavelength      -> sgf ZARF GENISLIGI, yani BANT
                                              GENISLIGI dugmesi
    ricker    : t0 = sgf * wavelength      -> sgf yalniz GECIKME; Ricker'in
                                              bant genisligi f_p'ye KILITLI

Bu ayrimin kendisi olculdu ve testle kilitlendi (test_yee.py). Ayrimi
yazmamak, plan taslaginin yaptigi hatanin tekrari olurdu: taslak tek bir
`ricker(t, *, wavelength, sgf)` tanimlayip dokstringine `sg = sgf *
wavelength` yaziyor ve 1560 katlik bant genisligi olcumunu ona bagliyordu --
ama o olcum Gauss x cos darbesiyle yapilmisti ve Ricker'da o dugme YOKTUR.
"""

from __future__ import annotations

import numpy as np

from qlab.errors import InvalidInputError


def gauss_cos(
    t: float | np.ndarray, *, wavelength: float, sgf: float, t0: float
) -> float | np.ndarray:
    """Gauss zarfli tasiyici: exp(-(t-t0)^2 / (2 sg^2)) * cos(2 pi (t-t0)/lam),
    `sg = sgf * wavelength`.

    `sgf` PENCERENIN ADLI PARCASIDIR, ANALIZ AYRINTISI DEGIL -- ve bu bir
    tercih degil OLCUM. Egik geliste (eps1 = 4, eps2 = 1, theta = 55 der,
    lam = 40, S = 0.4, N = 3600, npml = 60, iface = 1500, isrc = 600,
    probe = 1494, t0 = 5*sg, DIKDORTGEN FFT penceresi) yalniz `sgf`yi
    3 -> 6 yapmak fiziksel olabilirlik olcusunu

        ||rho_s| - 1| :  4.2533e-02  ->  2.7313e-05      (1557 KAT)

    kadar oynatiyor. AYNI taramada `npml` 60 -> 120 ve 60 -> 200, `N`
    3600 -> 5000, `S` 0.4 -> 0.2 sirasiyla 1.0000 / 1.0000 / 1.0000 / 1.0001
    kat, yani HICBIR SEY yapmiyor. Bu sayi ancak `sgf` adlandirilarak
    yazilabilir.

    DUGMENIN KENDISI de olculdu: bagil -3 dB bant genisligi (df/f_p),
    `sgf` = 1/2/3/6/9 icin 0.261230 / 0.129395 / 0.085449 / 0.041504 /
    0.026855, yani 1/sgf ile gidiyor.

    MEKANIZMA: egik indirgemede `ky` kosum boyunca SABITTIR, ama `f`
    bininin gordugu ETKIN aci asin(ky / (2 pi f n1))'dir. Genis bant, KOMSU
    BINLERIN farkli acilarinin r'sini hedef bine karistirir. Bu, olcum
    aletinin buyuk acilarda `R > 1` (fiziksel olarak imkansiz) dondurmesinin
    SEBEBIDIR -- bir sinir kusuru degil, bant genisligi arizasi.

    BU DOKSTRING BIR ZAMANLAR *"theta >= 75 derecede R > 1"* DIYORDU ve o
    CUMLE, egik cozucu SEVK EDILINCE (Task 3) OLCULEREK CURUDU. Sevk edilen
    boruda (eps1 = 4, eps2 = 1, lam = 40, S = 0.4, N = 3600, npml = 60,
    iface = 1500, isrc = 600, probe = 1494, t0 = 5 sg, DIKDORTGEN, kapi
    KAPALI) olculen |rho_s|:

        theta   45      55      65      70      72.5    75      80      85
        sgf=9   1.0000  1.0000  0.9986  1.0206  1.0291  0.9947  0.9420  0.9406
        sgf=3   0.9964  1.0425  0.9475  0.9159  0.9087  0.9126  0.9213  0.9344

    Yani `R > 1` bolgesi `sgf` = 9'da theta = 70-72.5, `sgf` = 3'te ise
    theta = 50-55'tir; theta = 75'te IKISINDE DE R < 1'dir. ARIZANIN ACISI
    BIR THETA ESIGI DEGIL, `sgf`NIN FONKSIYONUDUR -- ve bu, mekanizma
    iddiasini CURUTMEZ, tam tersine dogrular. Kapi bir ACI kapisi olarak
    degil, nokta nokta bir IMKAN kapisi olarak kurulmustur (bkz.
    `observable/em_obs.reflection_coefficient`).
    """
    if wavelength <= 0.0:
        raise InvalidInputError(f"wavelength > 0 olmali, verilen: {wavelength}")
    if sgf <= 0.0:
        raise InvalidInputError(f"sgf > 0 olmali, verilen: {sgf}")
    sg = sgf * wavelength
    u = np.asarray(t, dtype=float) - t0
    return np.exp(-(u**2) / (2.0 * sg**2)) * np.cos(2.0 * np.pi * u / wavelength)


def ricker(
    t: float | np.ndarray, *, wavelength: float, sgf: float = 1.0
) -> float | np.ndarray:
    """Ricker (Gauss'un ikinci turevi), tepe frekansi `f_p = 1/wavelength`.

        a = (pi * (t - t0) / wavelength)^2 ,   t0 = sgf * wavelength
        ricker = (1 - 2a) * exp(-a)

    BURADA `sgf` YALNIZ GECIKMEDIR ve `gauss_cos`taki bant genisligi
    dugmesiyle KARISTIRILAMAZ. Sebep cebirseldir, olcum degil: Ricker'in
    spektrumu |F(f)| ~ (f/f_p)^2 exp(-(f/f_p)^2)'dir, yani BAGIL bandi
    yalniz `f_p` belirler ve `t0` sadece faz tasir. Olculdu (ornekleme
    dt = lambda/160, M = 65536 nokta) -- bagil -3 dB bant genisligi df/f_p:

        wavelength 20 / 40 / 80 / 160 : 0.822754 hepsinde
        sgf         1 /  2 /  4 /   8 : 0.822754 hepsinde

    yani DEGISIM 0.0e+00. Ayni olcut `gauss_cos`ta `sgf` ile 0.261230'dan
    0.026855'e iniyor -- ayrim bir yorum degil, sayidir.

    Sevk gerekcesi: sogurucu yansimasi olcumu (bkz. boundary.PML) genis
    bantli TEK bir kosumda butun lambda'lari birden okumak icin bunu
    kullanir; oradaki dugme `wavelength` (= 1/f_p), zarf degildir.
    """
    if wavelength <= 0.0:
        raise InvalidInputError(f"wavelength > 0 olmali, verilen: {wavelength}")
    if sgf <= 0.0:
        raise InvalidInputError(f"sgf > 0 olmali, verilen: {sgf}")
    t0 = sgf * wavelength
    a = (np.pi * (np.asarray(t, dtype=float) - t0) / wavelength) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)
