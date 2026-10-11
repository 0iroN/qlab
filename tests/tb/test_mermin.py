"""Mermin dolumu ve tek orbitalli TB makinesi -- spec md-tb ss-sigma G2.

SURE (olculdu, 24.09.2026, bu makine, Python 3.14.7, numpy 2.5.1): 97 test,
duvar saati 1.81-1.85 s (uc kosu; toplama ve import dahil), `--durations=0`
toplami ~1.2 s. En yavaslar: M10 alt surec 0.38-0.40 s, C2 0.21-0.27 +
0.19-0.21 s, R1 0.04-0.06 s x2, M1 0.04-0.05 s; digerlerinin hepsi
< 0.03 s. Bulgu duzeltme turundan sonra (ayni gun) 101 test, pytest raporu
1.94-2.11 s; yeni testlerin (C3 S_gen x2, KAYAN NOKTA DIBI, ne-None) her
biri < 0.05 s. Hicbiri `slow` DEGILDIR (depo emsali 0.3 s'lik bir cipayi
`slow` saymiyor, spec 4.1).

Kapsam: spec 4.2 M1, M3'un dimer kismi, M4, M5 (elestirmen E3 dahil), M9,
M10'un t' = 0 kolu ve import kilitleri; spec 4.3 C1-C4 ve negatif testleri;
R1 (gevsemis q_dom regresyonu); prototip hatalarinin kapanisi (fermi D2-D4,
goreli dejenerelik toleransi, kararli entropi, kT = 0 kesirli -> kuvvet None,
d = 0 kapisi, L/2 kapisi, yalniz-x en-yakin-goruntu).

ETIKETLER. MAKINE: uygulama tutarliligi (M*). MODEL CIPASI: parametreden
bagimsiz, boyutsuz, girdide yazili olmayan; modelin KENDI kuramina karsi
sinanir, dogaya karsi degil (C*). Bu kenarda DOGA CIPASI YOKTUR: yasa
turetilmedi, secildi (spec B4). Parametrelerden geri okunan sayilar (denge
bagi, dimerlesme genligi, k_c) cipa DEGILDIR; R1 yalniz q'yu tasir.

Ortak pencere (spec basi): ss_sigma genlik -0.1, d0 2, us 2; itme genlik
0.05, d0 2, us 6; r_on 3.0, r_cut 3.8; a = 2.

ARAC TUZAKLARI (spec 4.1). `assert_allclose` (rtol = 1e-7 varsayilani) ve
`pytest.approx` (abs ile rel'in BUYUGU) kullanilmaz; her karsilastirma acik
`abs(x - y) < tol`dur ve her tolerans olculmus tabana baglidir. Sabite karsi
assert yazilmaz: beklenen degerler modulun sabitlerinden (G_S vb.) okunmaz,
kuramdan literal yazilir. Duzgun zincirde merkezi fark simetrik uyelik
dusmelerine kordur: orada tek yanli fark kullanilir. Satirlar konumsal
indeksle degil anlamla secilir (Fermi cifti, en kucuk yarilma).

BAGLAYICILIK OLCULDU (24.09.2026): her tolerans en az bir mutantla kirildi.
Mutantlar `src/` agacinin bir kopyasinda `mermin.py`ye (M1'in `dfc` ve
turev mutantlari icin `radyal.py`ye) uygulanip bu dosya
`PYTHONPATH=<kopya>/src` ile kosturuldu; hangi mutantin hangi testi kirdigi
ilgili testin dokstringinde yazili.
"""

from __future__ import annotations

import ast
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq, minimize
from scipy.special import expit

import qlab.engine.tb.mermin as mermin_modulu
import qlab.engine.tb.radyal as radyal_modulu
from qlab.engine.tb.mermin import mermin_dolum, tek_orbital_mermin
from qlab.engine.tb.radyal import UsselRadyalYasa
from qlab.errors import InvalidInputError

R_ON = 3.0
R_CUT = 3.8

# Kilitler KOSAN kodu okur (G1'de olculdu: sabit depo yolu PYTHONPATH
# kopyasindaki mutantlari gormuyordu).
MERMIN_DOSYASI = Path(mermin_modulu.__file__).resolve()
RADYAL_DOSYASI = Path(radyal_modulu.__file__).resolve()


def _t(us: float = 2.0, genlik: float = -0.1, r_on: float = R_ON) -> UsselRadyalYasa:
    return UsselRadyalYasa(genlik, 2.0, us, r_on, R_CUT)


def _itme(genlik: float = 0.05, us: float = 6.0, r_on: float = R_ON, r_cut: float = R_CUT):
    return UsselRadyalYasa(genlik, 2.0, us, r_on, r_cut)


SS_SIGMA = _t()
ITME = _itme()


def _zincir(N: int, a: float, genlik: float = 0.0, tohum: int = 0) -> np.ndarray:
    """x ekseninde 1B zincir; istege bagli N(0, genlik^2) x bozulmasi."""
    R = np.zeros((N, 3))
    R[:, 0] = a * np.arange(N)
    if genlik:
        R[:, 0] += genlik * np.random.default_rng(tohum).standard_normal(N)
    return R


def _om(R, ne, kT, L=None, t=SS_SIGMA, p=ITME):
    return tek_orbital_mermin(R, t, p, float(ne), kT, L)


def _ciftler(R: np.ndarray, L: float | None):
    """(i, j, d) -- i < j, en-yakin-goruntu (yalniz x) BAGIMSIZ hesap."""
    fark = R[None, :, :] - R[:, None, :]
    if L is not None:
        fark[..., 0] -= L * np.round(fark[..., 0] / L)
    i, j = np.triu_indices(R.shape[0], 1)
    return i, j, np.linalg.norm(fark[i, j], axis=1)


def _dikise_uzaklik(R, L, r_on: float = R_ON) -> float:
    """Her ciftin kuyruk dikislerine (r_on, r_cut) en kucuk uzakligi (spec B3).

    Dikisi kapsayan merkezi fark O(h)'dir (C^1, C^2 degil); sonlu fark
    kullanan her test sinadigi geometride bunu >= 1e-3 (en buyuk h'nin
    10 kati) diye KENDISI dogrular.
    """
    _, _, d = _ciftler(R, L)
    return float(np.min(np.minimum(np.abs(d - r_on), np.abs(d - R_CUT))))


# --------------------------------------------------------------------- M1 --

_RNG7 = np.random.default_rng(7)
_KUME8 = _RNG7.uniform(0.0, 4.5, size=(8, 3))
_ZINCIR16_3B = _zincir(16, 2.0, 0.1, 3) + np.c_[np.zeros(16), 0.1 * _RNG7.standard_normal((16, 2))]


@pytest.mark.parametrize(
    ("R", "ne", "L"),
    [
        pytest.param(_KUME8, 8, None, id="kume8-kutusuz"),
        pytest.param(_ZINCIR16_3B, 16, 32.0, id="zincir16-3B-L32"),
    ],
)
def test_M1_kuvvet_OMEGANIN_eksi_gradyani_3N_bilesenin_HEPSI(R, ne, L):
    """F = -grad(Omega), 3N bilesen, merkezi fark h = 1e-5, kT = 1e-3 (MAKINE).

    Pencere (spec M1): 3B kume N = 8 (U[0, 4.5]^3, tohum 7), n_e = 8,
    kutusuz; periyodik zincir(16, 2, 0.1, 3) + y, z'ye 0.1 N(0, 1) (ayni
    uretecin devami), n_e = 16, L = 32. Test ON KOSULU KENDISI dogrular:
    her cift r_on ve r_cut'tan >= 1e-3 uzakta (B3: dikiste merkezi fark
    O(h)) ve HOMO-LUMO >= 4e-4. Olculen on kosul: dikise 8.97e-02 / 7.36e-03,
    bosluk 1.75e-02 / 1.42e-03.
    Olculen max|F + grad_FD|: 1.32e-09 / 7.78e-11 (h = 1e-5); h taramasi
    1e-4'te 1.5e-07 / 7.9e-09 (kesme hatasi), 1e-6'da 1.2e-09 / 7.9e-10
    (yuvarlama). Tolerans 1e-8, en kotuye 7.6x pay.
    KIRMA (olculdu, kume8 / zincir16, max|F + grad_FD|): kuvvet isareti
    ters 17.1 / 5.7e-01; E_band'da spin 1 3.4e-01 / 9.6e-02; yogunlukta
    (rho) spin 1 3.4e-01 / 1.0e-01 (cift sayimi 1x ayni); E_rep x2 9.2 /
    4.9e-01; entropi isareti ters (E + kTS) 2.0e-04 / 2.1e-02; entropide
    spin 1 5.0e-05 / 5.4e-03; dfc unutulmus (radyal.py) 2.3e-02 / 1.3e-03;
    turev x1.1 (radyal.py) 8.6e-01 / 2.9e-02; bant kuvvetine t sizintisi
    (t' + 1e-3 t) 6.8e-04 / 2.0e-04. KOR KALDIGI (olculdu, spec ile ayni):
    tutarli ama yanlis yasa (us -> 1.3 us) geciyor; onu test_radyal M2
    yakalar.
    """
    s = _om(R, ne, 1e-3, L)
    assert _dikise_uzaklik(R, L) >= 1e-3
    eps = np.asarray(s.eps)
    homo = ne // 2 - 1
    assert eps[homo + 1] - eps[homo] >= 4e-4
    h = 1e-5
    gradyan = np.zeros_like(R)
    for i in range(R.shape[0]):
        for c in range(3):
            arti = R.copy()
            arti[i, c] += h
            eksi = R.copy()
            eksi[i, c] -= h
            gradyan[i, c] = (_om(arti, ne, 1e-3, L).omega - _om(eksi, ne, 1e-3, L).omega) / (2 * h)
    assert s.kuvvet is not None and s.kuvvet.shape == R.shape
    hata = float(np.max(np.abs(s.kuvvet + gradyan)))
    assert hata < 1e-8, hata


# --------------------------------------------------------------------- M3 --


def test_M3_dimer_MERDIVENI_kapanir_mermin_dolumuyla():
    """dE(e) = Omega(r_cut - e) - Omega(r_cut + e); oran dE(1e-3)/dE(1e-4) in [90, 110].

    Pencere (spec M3): dimer, n_e = 2, kT = 1e-3, kutusuz. Depo bugun
    (`uniform_model`, sert kesme) ayni dimerde sabit 0.2000 sicrama verir
    (2|t|, girdiden geri okuma); burada basamak e^2 ile kapanir.
    Olculen: dE(1e-3) = 8.71222e-10, dE(1e-4) = 8.70883e-12, oran 100.0389
    (spec 100.04). KIRMA (olculdu, radyal.py mutantlari): C^0 (dogrusal)
    kuyruk oran 2.65 (dE 6.92e-08 / 2.61e-08); sert kesme oran 1.001 (dE
    -3.009e-02 / -3.006e-02, basamak kapanmiyor).
    """

    def omega(d: float) -> float:
        R = np.zeros((2, 3))
        R[1, 0] = d
        return _om(R, 2, 1e-3).omega

    dE = [omega(R_CUT - e) - omega(R_CUT + e) for e in (1e-3, 1e-4)]
    assert 90.0 <= dE[0] / dE[1] <= 110.0, dE


# --------------------------------------------------------------------- M4 --


def _gap(R: np.ndarray) -> float:
    eps = np.asarray(_om(R, 16, 0.0, 32.0).eps)
    return float(eps[8] - eps[7])


@pytest.mark.parametrize("x", [0.2, 1.0, 3.086809276836417, 10.0, 20.0])
def test_M4_OMEGA_E_farki_EKSI_P_x_grad_gap(x):
    """F_HF - F_FD(E) = kT grad S = -P(x) grad(gap), P(x) = x/(4 cosh^2(x/4)).

    MAKINE (iki seviyeli yasanin kapali bicimi; isaret YASASI). Pencere
    (spec M4): zincir(16, 2, 0.05, 3), n_e = 16 (yari dolu: Fermi ciftinde
    2 ELEKTRON -- P yasasi yalniz bu durumda; 1 ya da 3 elektronda P_1,
    `mermin.py` dokstringi), L = 32; gap
    kT = 0 ozdegerlerinden (7.5019e-04), kT = gap/x; bilesen (atom 1, x);
    grad(gap) merkezi h = 1e-7 (4.575665e-02), FD(E) merkezi h = 1e-6.
    On kosul (spec B3, test dogrular): her cift dikisten >= 1e-3 uzak
    (olculen 1.62e-02). KIRMA (olculdu, TEST tarafi, sk/duz/olc_m4_dikis.log):
    (1, 3) cifti r_cut + 3e-7'ye tasininca on kosulsuz test x_max ve x = 10
    hucrelerinde oran - 1 = 1.25e-07 / 2.03e-07 ile SESSIZCE geciyor (0.2,
    1 ve 20'de kirmizi); on kosulla bes hucrenin besi kirmizi.
    Olculen oran: 1.000000614 / 0.999999986 / 0.999999989 / 1.000000063 /
    1.000000211 (x = 0.2 / 1 / 3.0868 / 10 / 20); en kotu sapma 6.1e-07.
    Tolerans 1e-5. x = x_max'ta fark -2.048723e-02 hartree/bohr (spec B1).
    Curuyen iddianin (+P) isareti bu testte oran -1 verir (olculdu, spec
    -0.99999999): test isarete KOR DEGIL.
    KIRMA (olculdu, x_max hucresinde oran): kuvvet isareti ters 13.0;
    E_band'da spin 1 2.45; rho'da spin 1 -0.952; E_rep x2 10.9; turev x1.1
    0.400; bant kuvvetine t sizintisi 0.996 (sapma 3.9e-03). DIKKAT
    (olculdu): x = 0.2 hucresi yasaya duyarli -- tutarli us -> 1.3 us
    yasasinda sapma 1.14e-05 (> 1e-5); x = 0.2'de kT = 3.75e-03 bosluktan
    buyuk ve iki seviye yaklasiminin artigi 1e-5 mertebesinde.
    """
    R = _zincir(16, 2.0, 0.05, 3)
    assert _dikise_uzaklik(R, 32.0) >= 1e-3
    gap = _gap(R)
    kT = gap / x
    s = _om(R, 16, kT, 32.0)
    i, c = 1, 0
    h = 1e-7
    arti = R.copy()
    arti[i, c] += h
    eksi = R.copy()
    eksi[i, c] -= h
    grad_gap = (_gap(arti) - _gap(eksi)) / (2 * h)
    h = 1e-6
    arti = R.copy()
    arti[i, c] += h
    eksi = R.copy()
    eksi[i, c] -= h

    def enerji(Rx):
        q = _om(Rx, 16, kT, 32.0)
        return q.E_band + q.E_rep

    F_E = -(enerji(arti) - enerji(eksi)) / (2 * h)
    P = x / (4.0 * math.cosh(x / 4.0) ** 2)
    oran = (s.kuvvet[i, c] - F_E) / (-P * grad_gap)
    assert abs(oran - 1.0) < 1e-5, oran


def _P1(x: float) -> float:
    """P_1(x) = 2 x df_1/dx, f_1 = f(a), f(a) + f(a + x) = 1/2 (`mermin.py` dokstringi).

    TESTTE ELLE, kuramdan: f(y) = 1/(1 + e^y) = expit(-y), f' = -f(1 - f).
    Kisit turevlenince a'(x) = -f'(a + x)/(f'(a) + f'(a + x)), yani
    df_1/dx = -f'(a) f'(a + x)/(f'(a) + f'(a + x)) > 0. Kok `brentq`la;
    parantez [-x - 50, 50] kokun iki yaninda (toplam ~2 ve ~0).
    """
    a = brentq(lambda a: expit(-a) + expit(-(a + x)) - 0.5, -x - 50.0, 50.0)
    d0 = -expit(-a) * expit(a)
    d1 = -expit(-(a + x)) * expit(a + x)
    return 2.0 * x * (-(d0 * d1) / (d0 + d1))


def test_M4_P1_kapali_bicimin_TEPESI_dokstringdeki_sayi():
    """`_P1` yardimcisinin kendisi: P_max = 0.23967, x = 2.0873'te (dokstring).

    OLCULDU (11.10.2026): tepe 0.2396657050, x = 2.0872913189'da (sinirli
    `minimize_scalar` ile; burada Nelder-Mead ayni noktaya iner). Bu bir
    CEVIRI denetimidir (yardimci <-> dokstring);
    fizik isirigi asagidaki M4-P1'dedir.
    """
    tepe = minimize(lambda v: -_P1(float(v[0])), x0=[2.0], method="Nelder-Mead",
                    options={"xatol": 1e-9, "fatol": 1e-15})
    assert abs(-tepe.fun - 0.23967) < 5e-6, -tepe.fun
    assert abs(float(tepe.x[0]) - 2.0873) < 5e-5, tepe.x


@pytest.mark.parametrize("ne", [15, 17])
@pytest.mark.parametrize("x", [0.2, 1.0, 2.0872913188686844, 3.086809276836417, 10.0])
def test_M4_P1_TEK_Ne_ciftte_1_ya_da_3_elektron_P1_YASASI(ne, x):
    """M4'un TEK ELEKTRONLU KOLU: n_e = 15 ve 17'de F_HF - F_FD(E) = -P_1(x) grad(gap).

    MAKINE (iki seviyeli yasanin kapali bicimi; D8, 11.10.2026). Pencere M4'un
    AYNISI (zincir(16, 2, 0.05, 3), L = 32, bilesen (atom 1, x), Fermi cifti
    8. ve 9. seviye, ayni sonlu farklar); degisen yalniz n_e: 15'te ciftte 1,
    17'de 3 elektron. mu artik bosluk ortasinda DEGIL (P_1'in kisiti).
    OLCULDU (11.10.2026): oran - 1 = +1.03e-06 / +1.6e-08 / +5.2e-09 /
    -8.6e-09 / +7.04e-07 (x = 0.2 / 1 / 2.087 / 3.087 / 10); 15 ile 17 basilan
    dort hanede AYNI. Tolerans M4'unki, 1e-5. x = 20 PENCERE DISI: P_1(20)
    ustel kucuk ve fark sonlu fark tabanina iner (olculdu oran - 1 = 7.0e-02
    / 1.1e-02); M4'un P(20)'si ise hala okunur.
    KARSIT (ayni olcum, iki elektron yasasina gore): P_olculen/P = 0.749 /
    0.715 / 0.596 / 0.437 / 0.0137 -- 1'den en az 0.25 uzak; test iki yasayi
    AYIRIR. KIRMA (olculdu, `mut.py`): dolumda tek n_e'yi en yakin cifte
    yuvarlayan kod mutanti (tek elektronlu kolu SILER) burada kirmizi, M4'un
    bes hucresinde YESIL; testte P_1 yerine P, ya da kisitta 1/2 yerine 1
    (iki elektron kosulu) yazan mutantlar da kirmizi.
    """
    R = _zincir(16, 2.0, 0.05, 3)
    assert _dikise_uzaklik(R, 32.0) >= 1e-3
    gap = _gap(R)
    kT = gap / x
    s = _om(R, ne, kT, 32.0)
    i, c = 1, 0
    h = 1e-7
    arti = R.copy()
    arti[i, c] += h
    eksi = R.copy()
    eksi[i, c] -= h
    grad_gap = (_gap(arti) - _gap(eksi)) / (2 * h)
    h = 1e-6
    arti = R.copy()
    arti[i, c] += h
    eksi = R.copy()
    eksi[i, c] -= h

    def enerji(Rx):
        q = _om(Rx, ne, kT, 32.0)
        return q.E_band + q.E_rep

    F_E = -(enerji(arti) - enerji(eksi)) / (2 * h)
    olculen = (s.kuvvet[i, c] - F_E) / (-grad_gap)
    P1 = _P1(x)
    assert abs(olculen / P1 - 1.0) < 1e-5, olculen / P1
    P = x / (4.0 * math.cosh(x / 4.0) ** 2)
    assert abs(olculen / P - 1.0) > 0.2, olculen / P


# --------------------------------------------------------------------- M5 --


def _dimer_modu(N: int) -> np.ndarray:
    v = np.zeros((N, 3))
    v[:, 0] = (-1.0) ** np.arange(N)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize(
    ("ne", "beklenen"),
    [
        pytest.param(16, -0.25, id="Ne16-f=1/2"),
        pytest.param(15, -0.75, id="Ne15-f=1/4"),
        pytest.param(17, -0.75, id="Ne17-f=3/4"),
    ],
)
@pytest.mark.parametrize("kT", [1e-6, 1e-7])
def test_M5_smearing_egriligi_GENEL_yasa_ve_uc_degerleri(ne, beklenen, kT):
    """E''(0) kT / c^2 = -g_s f(1-f) sum(v_i - v_bar)^2 / c^2 (MAKINE).

    Pencere: duzgun N16 halka (a = 2, L = 32), normalize dimer modu, Fermi
    ciftinin (2 katli, simetrik yarilan) dolulugu f. c = kT = 0 tek yanli
    egim (Richardson, h = 1e-5 ve 5e-6 ileri fark; merkezi fark burada 0
    okur), v_i = cift ozdegerlerinin tek yanli turevi (h = 1e-6). E''
    merkezi ikinci farkin RICHARDSON birlesimidir: (4 o(h/2) - o(h))/3,
    h = 0.1 kT/c. On kosul (spec B3, test dogrular): her cift dikisten
    >= 1e-3 uzak (olculen 2.0e-01).
    -1/4 GENEL YASA DEGILDIR (elestirmen E3): yalniz 2 katli, simetrik ve
    ciftinde 2 elektron olan uc (f = 1/2, c = 2 v0) -1/4 verir; 1 ya da 3
    elektronda (f = 1/4, 3/4, c = v0) -3/4. Beklenenler kuramdan literal
    (g_s = 2 modulden OKUNMAZ).

    NEDEN RICHARDSON (olculdu, 24.09.2026; sk/duz/b3_m5.log). Ilk surum tek
    adim h = 1e-2 kT/c kullaniyordu ve kT = 1e-7 hucresi YUVARLAMA
    TABANINDAYDI: oran Delta^2 Omega'nin ulp kuantumunu okuyordu (adim
    ~ulp(|Omega|) x 1e4 / kT = 2.2e-05) ve olculen -6.66e-05 / +6.66e-05
    (Ne16 / Ne15) birkac basamaktan ibaretti, fizik degil. Fizigi
    degistirmeyen 200 x-otelemesinde o yontemin gurultusu kT = 1e-7'de std
    7.4e-05 / 8.6e-05, en kotu 2.9e-04 (tolerans 1e-3'un yuvarlamaya karsi
    payi ~3.5x; sayi makineye/BLAS'a bagli; eski dokstringin "13x pay"i
    iyimserdi ve 7.5e-05 aslinda kT = 1e-6 Ne15 hucresine aitti). Ayni 200
    otelemede Richardson: kT = 1e-6'da std 5.5e-07 / 3.0e-07, kT = 1e-7'de
    3.7e-06 / 5.4e-06 (en kotu 2.2e-05). h ile h/2'yi 1e-2 kT/c'de almak
    yuvarlamayi BUYUTURDU (olculdu, b3_m5_kucuk_h.log: kT = 1e-7'de std
    4.0e-04 / 4.1e-04, tek adimin ~5 kati); olcek bu yuzden 0.1 kT/c.
    Olculen sapma (oteleme yok; c = 0.20000000 / 0.10000000): Ne16
    +1.62e-05 (kT 1e-6), -3.57e-06 (1e-7); Ne15 ve Ne17 +6.83e-05 /
    +1.10e-05. kT = 1e-6'daki fark FIZIKTIR (bant sizintisi, kT ile
    dogrusal; 1e-7'de ~1/10'u ve yuvarlama gurultusuyle ayni mertebe).
    kT = 1e-5'te Ne15 -0.74932 (sapma 6.8e-4; kT/c^2 buyudukce duzenli bant
    payi sizar) -- bu yuzden 1e-6 ve 1e-7. Tolerans 1e-3: oteleme dahil en
    kotu sapmaya (6.9e-05) ~14x pay. Makineler arasi SINANMADI.
    KIRMA: dosya sonundaki M5 kaydi (Richardson ile yeniden olculdu).
    """
    N = 16
    v = _dimer_modu(N)
    R0 = _zincir(N, 2.0)
    assert _dikise_uzaklik(R0, 32.0) >= 1e-3

    def e0(s: float) -> float:
        return _om(R0 + s * v, ne, 0.0, 32.0).omega

    h0 = 1e-5
    c = abs(2.0 * (e0(h0 / 2) - e0(0.0)) / (h0 / 2) - (e0(h0) - e0(0.0)) / h0)
    taban = _om(R0, ne, 0.0, 32.0)
    kayik = _om(R0 + 1e-6 * v, ne, 0.0, 32.0)
    # Fermi cifti: kismen dolu iki seviye (konumla degil dolulukla secilir)
    f0 = np.asarray(taban.f)
    cift = np.flatnonzero((f0 > 0.0) & (f0 < 1.0))
    assert cift.size == 2 and taban.kesirli
    f = float(f0[cift[0]])
    v_i = (np.asarray(kayik.eps)[cift] - np.asarray(taban.eps)[cift]) / 1e-6
    genel = -2.0 * f * (1.0 - f) * float(np.sum((v_i - v_i.mean()) ** 2)) / c**2

    def e(s: float) -> float:
        return _om(R0 + s * v, ne, kT, 32.0).omega

    def ikinci_fark(h: float) -> float:
        return (e(h) - 2.0 * e(0.0) + e(-h)) / h**2 * kT / c**2

    h = 0.1 * kT / c
    oran = (4.0 * ikinci_fark(0.5 * h) - ikinci_fark(h)) / 3.0
    assert abs(oran - beklenen) < 1e-3, (oran, beklenen)
    assert abs(oran - genel) < 1e-3, (oran, genel)


# --------------------------------------------------------------------- M9 --


def test_M9_SARMA_nL_otelemesi_Omega_ve_F_degistirmez():
    """Bir atom +4L, sonra 50 rastgele +-nL (n <= 5) oteleme (MAKINE).

    Pencere (spec M9): zincir(16, 1.8, 0.05, 3), L = 28.8 (a != d0),
    n_e = 16, kT = 2e-3. Olculen: +4L'de |dOmega| 0.0, max|dF| 5.1e-15
    (tolerans 1e-13, spec); 50 rastgele otelemede (koordinat 5L'ye kadar)
    max 1.9e-14 / 3.7e-14 (bit bit ayni DEGIL, spec B5). Rastgele kolun
    toleransi 1e-12: OLCULDU, fiziksel olarak zararsiz bir yasa degisikligi
    (us -> 1.3 us, tutarli) ayni kolda 1.13e-13 GURULTU uretti; 1e-13 orada
    gurultuye dayanikli degil.
    KIRMA (olculdu): en-yakin-goruntu kapali (sarma yok) +4L'de dOmega =
    7.85e-02. KOR KALDIGI (olculdu): floor ile sarma (en yakin goruntu
    DEGIL ama nL-periyodik) M9'u gecer; onu C2, C3, C4, M4, M5, R1 yakalar.
    """
    N = 16
    L = 1.8 * N
    R = _zincir(N, 1.8, 0.05, 3)
    ref = _om(R, 16, 2e-3, L)
    R4 = R.copy()
    R4[7, 0] += 4.0 * L
    kayik = _om(R4, 16, 2e-3, L)
    assert abs(kayik.omega - ref.omega) < 1e-13, ('4L omega', kayik.omega - ref.omega)
    dF = float(np.max(np.abs(kayik.kuvvet - ref.kuvvet)))
    assert dF < 1e-13, ('4L F', dF)
    rng = np.random.default_rng(0)
    for _ in range(50):
        Rn = R.copy()
        Rn[:, 0] += L * rng.integers(-5, 6, N)
        q = _om(Rn, 16, 2e-3, L)
        assert abs(q.omega - ref.omega) < 1e-12, ('nL omega', q.omega - ref.omega)
        dF = float(np.max(np.abs(q.kuvvet - ref.kuvvet)))
        assert dF < 1e-12, ('nL F', dF)


def test_M9_L_YARI_siniri_KESIN_ve_IKI_yasanin_BUYUGU():
    """2 max(r_cut) >= L hata; L = 2 r_cut + 1e-7 kabul (spec B5, M9).

    Kesme iki yasanin BUYUGUDUR: itmenin r_cut'i 4.2 iken L = 8.0 (ss_sigma
    icin yeterli) de reddedilir. KIRMA (olculdu): kapi yok, esitligi kabul
    eden kapi (<=) ve yalniz ss_sigma'nin r_cut'ini okuyan kapi -- ucu de
    kiriyor.
    """
    R = _zincir(4, 1.9)
    with pytest.raises(InvalidInputError):
        _om(R, 4, 2e-3, 2.0 * R_CUT)
    _om(R, 4, 2e-3, 2.0 * R_CUT + 1e-7)
    genis_itme = _itme(r_cut=4.2)
    with pytest.raises(InvalidInputError):
        _om(R, 4, 2e-3, 8.0, p=genis_itme)
    _om(R, 4, 2e-3, 8.4 + 1e-7, p=genis_itme)


def test_yalniz_X_periyodik_Y_ve_Z_ACIK():
    """y ya da z'de L kadar oteleme BASKA bir geometridir (spec B5).

    Dimer d = 2, L = 32: atom 1'i y'de (ya da z'de) +L kaydirmak cifti
    kesmenin disina atar; Omega serbest iki atomun degerine (E_rep = 0,
    kuvvet 0) gider. x'te ayni oteleme hicbir sey degistirmez.
    """
    R = np.zeros((2, 3))
    R[1, 0] = 2.0
    ref = _om(R, 2, 2e-3, 32.0)
    for eksen in (1, 2):
        Ry = R.copy()
        Ry[1, eksen] += 32.0
        q = _om(Ry, 2, 2e-3, 32.0)
        assert q.E_rep == 0.0 and abs(q.omega - ref.omega) > 1e-2
        assert float(np.max(np.abs(q.kuvvet))) == 0.0
    Rx = R.copy()
    Rx[1, 0] += 32.0
    assert abs(_om(Rx, 2, 2e-3, 32.0).omega - ref.omega) < 1e-15


# -------------------------------------------------------------------- M10 --


def test_M10_t_turevi_SIFIRKEN_bant_kuvveti_bit_bit_SIFIR():
    """us = 0 ve kuyrukta cift yokken bant kuvveti 48/48 TAM 0.0 (spec B4, M10).

    Bant kuvvetinin TAMAMI yazilan yasanin turevinden gelir. Pencere:
    zincir(16, 2, 0.05, 3), n_e = 16, kT = 2e-3, L = 32; itme genligi 0
    (yalniz bant). On kosul (test dogrular): (r_on, r_cut) kuyrugunda cift
    YOK. Kontrol: us = 2'de ayni pencerede max|F| = 1.054e-01 (test bos
    gecmiyor). KIRMA (olculdu): bant kuvvetine t sizintisi (t' + 1e-3 t)
    48 bilesenden 32'sini sifir birakiyor, en buyugu 2.2e-19 -- bit bit
    iddia 2e-19'luk sizintiyi da yakaliyor.
    """
    R = _zincir(16, 2.0, 0.05, 3)
    _, _, d = _ciftler(R, 32.0)
    assert int(np.sum((d > R_ON) & (d < R_CUT))) == 0
    sifir_itme = _itme(genlik=0.0)
    s = _om(R, 16, 2e-3, 32.0, t=_t(us=0.0), p=sifir_itme)
    sifir = int(np.sum(s.kuvvet == 0.0))
    assert sifir == 48, (sifir, float(np.max(np.abs(s.kuvvet))))
    kontrol = _om(R, 16, 2e-3, 32.0, p=sifir_itme)
    assert float(np.max(np.abs(kontrol.kuvvet))) > 1e-2


def _importlar(yol: Path) -> set[str]:
    agac = ast.parse(yol.read_text(encoding="utf-8"), filename=str(yol))
    adlar: set[str] = set()
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.Import):
            adlar.update(a.name for a in dugum.names)
        elif isinstance(dugum, ast.ImportFrom):
            assert dugum.level == 0, "goreli import tarayicidan kacar"
            adlar.add(str(dugum.module))
    return adlar


def test_M10_mermin_IMPORTLARI_kilitli():
    """Spec 2.1(2): numpy, scipy.special, qlab.errors, qlab.engine.tb.radyal.

    Tam kume esitligi: her yeni import bilincli bir test degisikligi ister.
    KIRMA (olculdu): birim modulu, downfold ve `math` importu eklenmis uc
    mutant da kume esitligini kiriyor.
    """
    assert _importlar(MERMIN_DOSYASI) == {
        "__future__",
        "dataclasses",
        "numpy",
        "scipy.special",
        "qlab.errors",
        "qlab.engine.tb.radyal",
    }


_YASAK_ONEKLER = (
    "qlab.engine.hf",
    "qlab.engine.tb.downfold",
    "qlab.engine.tb.factory",
    "qlab.units",
)


def _yasakli_mi(modul: str) -> bool:
    return any(modul == y or modul.startswith(y + ".") for y in _YASAK_ONEKLER)


@pytest.mark.parametrize("yol", [RADYAL_DOSYASI, MERMIN_DOSYASI], ids=["radyal", "mermin"])
def test_M10_HF_DOWNFOLD_FACTORY_ve_birim_modulu_IMPORT_EDILMEZ(yol):
    """Spec 2.2 ve M10: kenar HF'e cagri duzeyinde dokunmaz.

    `tb.factory` downfold uzerinden 8, md fabrikasi 6 hf modulu yukluyor
    (spec 2.2); downfold'dan gelen yasa yasak (TB payi %0.000, spec 5.2).
    Birim modulu yasak: units ekseninde tb 2'den 3'e cikar (spec 5.2).
    """
    assert [m for m in _importlar(yol) if _yasakli_mi(m)] == []


def test_M10_mermin_yasanin_IKINCI_KOPYASINI_yazmaz():
    """mermin.py'de exp/cos/sin cagrisi YOK: yasa yalniz radyal.py'den gelir.

    Spec M10'un md/tb_kuvvet.py icin istedigi kilidin ayni deseni; MD yolu
    yasayi burada degerlendirdigi icin ikinci kopya burada da yasak (V4
    tam da bunu yapmisti). Entropi `logaddexp` ve `expit` ile yazilir.
    KIRMA (olculdu): entropiyi `np.log1p(np.exp(-ax))` ile yazan mutant
    `[('exp', satir)]` veriyor.
    """
    agac = ast.parse(MERMIN_DOSYASI.read_text(encoding="utf-8"))
    cagrilar = []
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.Call):
            fonk = dugum.func
            ad = fonk.attr if isinstance(fonk, ast.Attribute) else getattr(fonk, "id", "")
            if ad in {"exp", "expm1", "exp2", "cos", "sin", "cosh", "sinh", "power"}:
                cagrilar.append((ad, dugum.lineno))
    assert cagrilar == []


def test_M10_mermin_importu_CALISMA_ZAMANINDA_motor_yuklemez():
    """Alt surecte `import qlab.engine.tb.mermin` -> yuklenen qlab modulleri TAM kume.

    `ast` kilidi yalniz dogrudan importu gorur; bu test gecisli yuklemeyi
    olcer (paket `__init__`leri dahil). Alt surec ayni PYTHONPATH'i
    devralir ve yukledigi dosyanin bu testin sinadigi dosya oldugunu dogrular.
    """
    kod = (
        "import sys, qlab.engine.tb.mermin as m\n"
        "print(m.__file__)\n"
        "print(','.join(sorted(k for k in sys.modules if k == 'qlab' or k.startswith('qlab.'))))\n"
    )
    cikti = subprocess.run(
        [sys.executable, "-c", kod],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        check=True,
    ).stdout.splitlines()
    assert Path(cikti[0]).resolve() == MERMIN_DOSYASI
    assert set(cikti[1].split(",")) == {
        "qlab",
        "qlab.engine",
        "qlab.engine.tb",
        "qlab.engine.tb.mermin",
        "qlab.engine.tb.radyal",
        "qlab.errors",
    }


# --------------------------------------------------------------------- C1 --


def _iki_parcali_mi(R: np.ndarray, L: float) -> bool:
    """Kesme icindeki HER cift farkli alt orgude mi (i + j tek)? H iki parcali."""
    i, j, d = _ciftler(R, L)
    icerde = d < R_CUT
    return bool(np.all((i[icerde] + j[icerde]) % 2 == 1))


@pytest.mark.parametrize(
    ("a", "genlik", "tohum"),
    [(2.0, 0.05, 3), (2.0, 0.05, 1), (2.0, 0.02, 7), (2.1, 0.05, 3)],
)
def test_C1_PARCACIK_BOSLUK_ayni_geometride(a, genlik, tohum):
    """Omega(N_e) = Omega(2N - N_e) ve F(N_e) = F(2N - N_e) (MODEL CIPASI).

    Iki parcali H (kosegen 0, onsite yok) spektrumu eps <-> -eps simetrik
    yapar. Pencere: zincir(16, a, genlik, tohum), L = 16 a, N_e 8 <-> 24,
    kT = 2e-3. On kosul (test dogrular): gercek geometride kesme icindeki
    her cift iki alt orgu arasinda (2. komsu >= r_cut; spec C1).
    Olculen: |dOmega| <= 8.9e-16, max|dF| <= 9.2e-16 (dort pencere; spec
    4.4e-16 / 2.6e-16). Tolerans 1e-13.
    KIRMA (olculdu): oz-cifti kosegene yazan mutant (H_ii = t(0), tekbicim
    sahte onsite) dOmega = -11.82; yarim entropi (-f ln f, f <-> 1-f
    simetrisini bozar) dOmega = -6.6e-11..-6.2e-10; g_s = 1 (global).
    """
    R = _zincir(16, a, genlik, tohum)
    L = 16 * a
    assert _iki_parcali_mi(R, L)
    s8 = _om(R, 8, 2e-3, L)
    s24 = _om(R, 24, 2e-3, L)
    assert abs(s24.omega - s8.omega) < 1e-13, s24.omega - s8.omega
    kuvvet_farki = float(np.max(np.abs(s24.kuvvet - s8.kuvvet)))
    assert kuvvet_farki < 1e-13, kuvvet_farki


@pytest.mark.parametrize(
    ("a", "genlik", "tohum", "omega_farki", "kuvvet_farki"),
    [
        pytest.param(1.8, 0.05, 3, 1e-2, 1e-2, id="a1.8-2.komsu-kuyrukta"),
        pytest.param(2.0, 0.1, 5, 1e-3, 1e-2, id="a2-bozulma0.1-2.komsu-3.667"),
    ],
)
def test_C1_NEGATIF_ikinci_komsu_KUYRUKTA_simetri_DUSER(
    a, genlik, tohum, omega_farki, kuvvet_farki
):
    """2. komsu kesme icine girince H iki parcali degil ve C1 duser (olculdu).

    a = 1.8: dOmega = 7.659e-02, max|dF| = 1.002e-01 (spec ile ayni).
    a = 2, zincir(16, 2, 0.1, 5): bozulma 2. komsuyu 3.667 < r_cut'a
    sokuyor; dOmega = 3.226e-03, max|dF| = 3.606e-02. On kosul denetimi bu
    yuzden testin PARCASIDIR: onsuz bu pencere C1'in pozitif listesine
    girebilirdi.
    """
    R = _zincir(16, a, genlik, tohum)
    L = 16 * a
    assert not _iki_parcali_mi(R, L)
    s8 = _om(R, 8, 2e-3, L)
    s24 = _om(R, 24, 2e-3, L)
    assert abs(s24.omega - s8.omega) > omega_farki
    assert float(np.max(np.abs(s24.kuvvet - s8.kuvvet))) > kuvvet_farki


# --------------------------------------------------------------------- C2 --


def _K_eff(N: int, ne: int, kT: float, h: float = 1e-5) -> np.ndarray:
    """K_eff(q) = lambda(q)/(4 sin^2(pi q/N)), q = 1..N//2; duzgun halka a = 2.

    Hessiyen kuvvetin merkezi farkindan (h = 1e-5), simetriklestirilmis.
    On kosul (spec B3): her cift dikisten >= 1e-3 uzak (olculen 2.0e-01:
    ikinci komsu 4.0, r_cut 3.8).
    """
    R = _zincir(N, 2.0)
    L = 2.0 * N
    assert _dikise_uzaklik(R, L) >= 1e-3
    H = np.zeros((N, N))
    for j in range(N):
        arti = R.copy()
        arti[j, 0] += h
        eksi = R.copy()
        eksi[j, 0] -= h
        H[:, j] = -(_om(arti, ne, kT, L).kuvvet[:, 0] - _om(eksi, ne, kT, L).kuvvet[:, 0]) / (2 * h)
    H = 0.5 * (H + H.T)
    q = np.arange(1, N // 2 + 1)
    jj = np.arange(N)
    lam = np.array([float(np.sum(H[0] * np.cos(2 * np.pi * qq * jj / N))) for qq in q])
    return lam / (4.0 * np.sin(np.pi * q / N) ** 2)


def _argmin_ve_marj(K: np.ndarray) -> tuple[int, float]:
    sirali = np.sort(K)
    return int(np.argmin(K)) + 1, float(sirali[1] - sirali[0])


@pytest.mark.parametrize("kT", [1e-4, 2e-3])
def test_C2_KOHN_argmin_q_F_duzgun_zincirde(kT):
    """argmin_q K_eff(q) = q_F = min(N_e/2, N - N_e/2) (MODEL CIPASI; tamsayi).

    PENCERE (spec C2 + elestirmen E1): N = 16 (CIFT), N_e = 2..30 CIFT,
    kT in {1e-4, 2e-3}, a = d0 = 2, us = 2, yalniz en yakin komsu. TEK N_e
    ve TEK N pencere DISIDIR: orada formul tamsayi degil ve secim kurali
    farkli (asagidaki negatif test). a = d0'da (k, A, krep) taramasi bir
    OZDESLIKTIR, taranmaz. Bagimsiz olgu kT basina ~8 (parcacik-bosluk
    yariya indirir; acik kabukta momentum secim kurali, gercek duyarlilik
    sinavi yalniz kapali kabukta: kT basina 4).
    Olculen: 30/30; en dar marj 5.5221e-02 (FD gurultusu ~2e-9). Her
    hucrede marj > 1e-3 da istenir (argmin gurultuyle karar verilmiyor).
    KIRMA (olculdu): g_s = 1 (spinsiz dolum; spec'te "fikir") N_e = 2'de
    argmin 2, q_F 1; kuvvet isareti ters N_e = 2'de argmin 8; en-yakin-
    goruntu kapali ya da floor ile sarma N_e = 14'te argmin 8. KOR KALDIGI
    (olculdu): bant kuvvetini olcekleyen mutantlar (rho'da spin 1, cift
    sayimi 1x, E_band'da spin 1) argmin'i KAYDIRMIYOR -- Kohn anomalisi 2k_F'de
    durur, ciftlenim buyuklugunden bagimsiz; onlari M1, M4, C3, C4 yakalar.
    """
    for ne in range(2, 32, 2):
        argmin, marj = _argmin_ve_marj(_K_eff(16, ne, kT))
        assert argmin == min(ne // 2, 16 - ne // 2), (ne, argmin)
        assert marj > 1e-3, (ne, marj)


@pytest.mark.parametrize("ne", [4, 28])
def test_C2_NEGATIF_tau_BUYUK_argmin_1(ne):
    """tau = kT/|t| buyuk: N16, kT = 1e-2, N_e in {4, 28} -> argmin 1, q_F = 2 (olculdu).

    Olculen (ikisinde ayni, parcacik-bosluk): K1 = 0.12895940,
    K2 = 0.13125500 (spec ile ayni). Marj 2.3e-03 > 1e-4.
    """
    argmin, marj = _argmin_ve_marj(_K_eff(16, ne, 1e-2))
    assert argmin == 1 and min(ne // 2, 16 - ne // 2) == 2
    assert marj > 1e-4


@pytest.mark.parametrize(
    ("N", "ne", "kT", "olculen"),
    [
        pytest.param(20, 3, 2e-3, 1, id="N20-Ne3-kT2e-3"),
        pytest.param(9, 11, 1e-4, 3, id="N9-Ne11-kT1e-4"),
    ],
)
def test_C2_NEGATIF_TEK_Ne_TEK_N_pencere_DISI_kural_FARKLI(N, ne, kT, olculen):
    """Tek N_e'de q_F formulu tamsayi degil; secim kurali cift-N kucuk-kT kuralindan FARKLI.

    Elestirmen E1: 300 tek hucrede argmin hep katlanmis {floor, ceil}(N_e/2)
    icinde; cift N ve kT = 1e-4'te ikisinden CIFT olani seciliyor. Bu iki
    hucrede o kural DUSUYOR (olculdu): N20/N_e3/kT 2e-3 -> 1 (kural 2 der;
    marj 4.18e-02), N9/N_e11/kT 1e-4 -> 3 (kural 4 der; marj 1.66e+01).
    Yani tek N_e / tek N icin cipa bir sey ONGORMUYOR; pencere disi.
    """
    assert ne % 2 == 1
    yarim = ne / 2.0
    aday = sorted(
        {min(math.floor(yarim), N - math.floor(yarim)), min(math.ceil(yarim), N - math.ceil(yarim))}
    )
    cift_kurali = [q for q in aday if q % 2 == 0]
    argmin, marj = _argmin_ve_marj(_K_eff(N, ne, kT))
    assert argmin == olculen and argmin in aday
    assert cift_kurali and argmin not in cift_kurali
    assert marj > 1e-3


# ----------------------------------------------------------------- C3, C4 --


def _dimer_egriligi(
    N: int, us: float, kT: float, h: float, r_on: float = R_ON, r_on_itme: float | None = None
) -> float:
    """dOmega/d delta / h, x_j += h (-1)^j (dimer modu, bag a +- 2h), halka a = 2.

    `r_on` hoplama yasasinin, `r_on_itme` itmenin kuyruk basi (verilmezse
    ikisi ayni). On kosul (spec B3): yer degistirmeden onceki halkada her
    cift iki yasanin dikislerinden >= 1e-3 uzak (olculen: r_on = 3.0'da
    2.0e-01, 1.9'da 1.0e-01, 1.5'te 2.0e-01).
    """
    r_on_itme = r_on if r_on_itme is None else r_on_itme
    R = _zincir(N, 2.0)
    for kuyruk_basi in {r_on, r_on_itme}:
        assert _dikise_uzaklik(R, 2.0 * N, kuyruk_basi) >= 1e-3
    isaret = (-1.0) ** np.arange(N)
    R[:, 0] += h * isaret
    F = _om(R, N, kT, 2.0 * N, t=_t(us=us, r_on=r_on), p=_itme(r_on=r_on_itme)).kuvvet
    return -float(np.sum(F[:, 0] * isaret)) / h


def _S(N: int, fermi_haric: bool = False) -> float:
    """S(N) = sum_{j < N/2} 1/|cos(2 pi j/N)|; 4n'de cos = 0 terimi dislanir (S')."""
    c = np.abs(np.cos(2 * np.pi * np.arange(N // 2) / N))
    if fermi_haric:
        c = c[c > 1e-12]
    return float(np.sum(1.0 / c))


# Hepsi ortak pencereden: |t(a)| = 0.1 (a = d0), phi''(a) = A (krep/d0)^2.
_T0, _D0, _A_ITME, _US_ITME = 0.1, 2.0, 0.05, 6.0
_PHI2 = _A_ITME * (_US_ITME / _D0) ** 2


def _huckel_esigi(N: int, r_on: float) -> float:
    """Lambda_c S(N) - 1; k_c egriligin sifiri (brentq), kT = 0, yari dolu 4n+2."""
    k_kapali = math.sqrt(_D0**2 * N * _PHI2 / (4 * _T0 * _S(N)))
    k_c = brentq(
        lambda us: _dimer_egriligi(N, us, 0.0, 1e-4, r_on),
        0.3 * k_kapali,
        3.0 * k_kapali,
        xtol=1e-13,
    )
    t_turev = k_c * _T0 / _D0
    return 4.0 * t_turev**2 / (_T0 * N * _PHI2) * _S(N) - 1.0


@pytest.mark.parametrize("N", [10, 14, 18, 22])
def test_C3_HUCKEL_4n_arti_2_esigi_Lambda_c_S_N_BIR(N):
    """Lambda_c S(N) = 1, Lambda = 4 t'^2/(|t| N phi'') (MODEL CIPASI).

    Pencere (spec C3): yalniz en yakin komsu, a = 2 <= r_on (fc = 1),
    yari dolu, onsite 0, kT = 0 (4n+2 kapali kabuk: kesirli degil, kuvvet
    var), N in {10, 14, 18, 22}; egrilik g(h)/h, h = 1e-4; k_c brentq.
    Olculen: +1.95e-07, +2.89e-07, +3.98e-07, +5.21e-07 (spec ile ayni).
    Tolerans 2e-6. Yanlilik N^2 h^2 ile buyur; N <= 22 bu yuzden.
    KIRMA (olculdu): bant kuvvetinde g_s = 1 (rho'da spin 1; spec'te
    "fikir") Lambda_c S(N) = 2.0000007 (N10) .. 2.0000020 (N22); cift
    sayimi 1x ayni; bant kuvvetine t sizintisi +9.4e-04..+1.05e-03. KOR
    KALDIGI (olculdu): tutarli us -> 1.3 us iki yasada birden t'^2/phi''
    oranini degistirmez, esik ayni kalir (C3 negatif testi yakalar).
    """
    assert N % 4 == 2
    sapma = _huckel_esigi(N, R_ON)
    assert abs(sapma) < 2e-6, sapma


def test_C3_NEGATIF_fc1_VEKIL_formulu_kuyruk_a_da_AKTIFKEN_GECERSIZ():
    """fc = 1 VEKIL formulu kuyruk a'da aktifken (r_on < a = 2, iki yasada birden) tutmaz.

    `_huckel_esigi`nin Lambda'si modelin GERCEK t(a), t'(a), phi''(a)
    degerlerini degil, fc(a) = 1 varsayan vekilleri kullanir: t' = k T0/D0,
    |t| = T0, phi'' = A (krep/d0)^2. Bu test YALNIZ o vekilin kuyruk a'da
    aktifken gecersiz oldugunu gosterir; spec C3'un fizik iddiasini (S(N)
    bicimi duser, S_gen tutar) SINAMAZ -- o bir sonraki testin isidir.
    Olculen (N 10/14/18/22; sk/curut_c3, sk/duz/olc_c3_mut.log): r_on =
    1.9'da +5.518e-02, +3.857e-02, +2.761e-02, +1.946e-02; r_on = 1.5'te
    vekil ISARET DEGISTIRIR: -3.874e-01, -4.257e-01, -4.519e-01,
    -4.717e-01. Pencere bu yuzden ISARETLIDIR: 1.9'da > +1e-2, 1.5'te
    < -1e-2. (Iki yanli |.| > 1e-2 dfc'yi unutan mutanta KOR kalirdi:
    orada 1.9'un degerleri -0.078..-0.092'ye doner -- olculdu.) Sayiyi
    spec'in +0.393'unden ayiran pencere degil Lambda'nin tanimidir: yalniz
    t'den iki yasaya gecmek GERCEK Lambda_c S(N) - 1'i yalniz +0.393'ten
    +0.421'e tasir (r_on = 1.9, N = 10); +0.055'e indiren VEKIL Lambda'dir
    (ayni vekil yalniz-t penceresinde +0.114 verir).
    KIRMA: dosya sonundaki C3 kaydi.
    """
    for r_on, isaret in ((1.9, 1.0), (1.5, -1.0)):
        for N in (10, 14, 18, 22):
            assert isaret * _huckel_esigi(N, r_on) > 1e-2, (r_on, N)


def _kuyruklu_t(us: float, r_on: float, d: float = 2.0) -> tuple[float, float, float]:
    """t(d), t'(d), t''(d) -- KAPALI BICIMDEN, test icinde; `radyal.py`den OKUNMAZ.

    t = A e^(-us (d/d0 - 1)) fc(d), A = -0.1, d0 = 2; r_on < d < r_cut'ta
    fc = sin^2(u), u = q (r_cut - d), q = pi/(2 (r_cut - r_on)):
    fc' = -q sin(2u), fc'' = 2 q^2 cos(2u). Neden yasanin kendi
    `deger_ve_turev`i DEGIL (olculdu, sk/duz/olc_c3_mut.log): turevlerin
    HEPSI (phi'' dahil) yasadan alinsaydi tutarli us -> 1.3 us mutanti
    modeli ve testi BIRLIKTE kaydirir ve KACARDI (S_gen - 1 <= 1.35e-06).
    Gercek kodda kapali bicim yasayla t ve t'de bit bit, t''de (yasanin
    turevinin merkezi farki, h = 1e-5) 1e-11 icinde ayni.
    """
    g = -0.1 * math.exp(-us * (d / 2.0 - 1.0))
    g1, g2 = -(us / 2.0) * g, (us / 2.0) ** 2 * g
    if d <= r_on:
        fc, fc1, fc2 = 1.0, 0.0, 0.0
    else:
        q = math.pi / (2.0 * (R_CUT - r_on))
        u = q * (R_CUT - d)
        fc, fc1, fc2 = math.sin(u) ** 2, -q * math.sin(2.0 * u), 2.0 * q * q * math.cos(2.0 * u)
    return g * fc, g1 * fc + g * fc1, g2 * fc + 2.0 * g1 * fc1 + g * fc2


def _S_gen(N: int, rho: float) -> float:
    """S_gen(rho) = sum_{j < N/2} [rho c_j + (1 - c_j^2)/c_j], c_j = |cos(2 pi j/N)|.

    rho = t t''/t'^2; ussel yasada (fc = 1) rho = 1 ve S_gen = S(N).
    """
    c = np.abs(np.cos(2 * np.pi * np.arange(N // 2) / N))
    return float(np.sum(rho * c + (1.0 - c**2) / c))


@pytest.mark.parametrize("r_on", [1.9, 1.5])
def test_C3_NEGATIF_kuyruk_a_da_AKTIFKEN_S_N_bicimi_DUSER_S_gen_TUTAR(r_on):
    """Spec C3'un dustugu yer, spec'in penceresiyle: YALNIZ t yasasinin kuyrugu a'da aktif.

    Pencere: t yasasinda r_on in {1.9, 1.5} < a = 2, itmede r_on = 3.0
    (fc = 1: phi''(a) = A (krep/d0)^2), yari dolu 4n+2 halka N in {10, 14,
    18, 22}, kT = 0; k_c egriligin sifiri (brentq, h = 1e-4). Lambda = 4
    t'(a)^2 / (|t(a)| N phi''(a)) GERCEK turevlerle (`_kuyruklu_t`, kapali
    bicim), rho = t t''/t'^2. (1) S(N) bicimi DUSER: Lambda_c S(N) - 1 =
    +0.393 / +0.268 (spec ile ayni; N boyunca +0.3932..+0.3881 ve
    +0.2683..+0.2648); esik > 0.2. (2) Genel bicim S_gen(rho) TUTAR:
    |Lambda_c S_gen - 1| <= 1.02e-06 (C3'un pozitif testiyle ayni N^2 h^2
    yanliligi); tolerans 2e-6 (pozitif testinki). On kosul `_dimer_egriligi`
    icinde (dikise 1.0e-01 / 2.0e-01).
    KIRMA: dosya sonundaki C3 kaydi.
    """
    for N in (10, 14, 18, 22):
        k_kapali = math.sqrt(_D0**2 * N * _PHI2 / (4 * _T0 * _S(N)))
        k_c = brentq(
            lambda us: _dimer_egriligi(N, us, 0.0, 1e-4, r_on, R_ON),
            0.1 * k_kapali,
            5.0 * k_kapali,
            xtol=1e-13,
        )
        t, t1, t2 = _kuyruklu_t(k_c, r_on)
        lam = 4.0 * t1**2 / (abs(t) * N * _PHI2)
        rho = t * t2 / t1**2
        assert lam * _S(N) - 1.0 > 0.2, (N, lam * _S(N) - 1.0)
        assert abs(lam * _S_gen(N, rho) - 1.0) < 2e-6, (N, lam * _S_gen(N, rho) - 1.0)


@pytest.mark.parametrize("N", [8, 16])
@pytest.mark.parametrize("kT", [1e-4, 1e-3])
def test_C4_4n_ve_kT_esigi_Lambda_4n_BIR(N, kT):
    """Lambda_4n = 4 t'^2 (1/kT + S'(N)/|t|)/(N phi'') = 1 (MODEL CIPASI).

    Pencere (spec C4): en yakin komsu, yari dolu 4n halka, N in {8, 16},
    kT in {1e-4, 1e-3}, h = 1e-4 kT; k_c brentq [1e-3, 3]. Olculen:
    +2.69e-08, +3.70e-10, -1.33e-08, +2.48e-11. Tolerans 1e-6.
    KIRMA (olculdu): rho'da spin 1 Lambda_4n = 2.0000000 (+-4e-8); bant
    kuvvetine t sizintisi Lambda_4n - 1 = 1.06e-02 .. 2.5e-03.
    """
    assert N % 4 == 0
    k_c = brentq(lambda us: _dimer_egriligi(N, us, kT, 1e-4 * kT), 1e-3, 3.0, xtol=1e-14)
    t_turev = k_c * _T0 / _D0
    lam = 4.0 * t_turev**2 * (1.0 / kT + _S(N, fermi_haric=True) / _T0) / (N * _PHI2)
    assert abs(lam - 1.0) < 1e-6, lam - 1.0


# --------------------------------------------------------------------- R1 --


def _bag_deseni(R: np.ndarray, L: float) -> np.ndarray:
    x = np.sort(np.mod(R[:, 0], L))
    return np.diff(np.append(x, x[0] + L))


def _baskin_fourier(bag: np.ndarray) -> tuple[int, np.ndarray]:
    """GENLIK-NORMLU: A cos(2 pi q j/N) her q'da genlik A verir.

    Ham |rfft| 0 < q < N/2'de A N/2, q = N/2'de (Nyquist) A N verir: Nyquist
    2x SISER (spec R1). Normlu: 2|X_q|/N, Nyquist'te |X_q|/N.
    """
    X = np.abs(np.fft.rfft(bag - bag.mean()))
    N = bag.size
    genlik = 2.0 * X / N
    if N % 2 == 0:
        genlik[-1] = X[-1] / N
    genlik[0] = 0.0
    return int(np.argmax(genlik)), genlik


def test_R1_baskin_fourier_GENLIK_NORMLU_Nyquist_SISMEZ():
    """Esit genlikli q = N/2 ve q = N/4 bilesenleri esit genlik verir.

    Ham |rfft| ile Nyquist 2x cikar (kontrol olarak yazili). R1 penceresinde
    diger bilesenler TAM 0 oldugu icin ham rfft de dogru q'yu seciyor
    (olculdu): normun baglayiciligi R1'de degil burada sinanir.
    """
    N = 16
    j = np.arange(N)
    for q in (4, 8):
        _, genlik = _baskin_fourier(0.03 * np.cos(2 * np.pi * q * j / N))
        assert abs(genlik[q] - 0.03) < 1e-15

    def dalga(q: int) -> np.ndarray:
        return np.cos(2 * np.pi * q * j / N)

    ham = np.abs(np.fft.rfft(0.03 * dalga(8) + 0.03 * dalga(4)))
    assert abs(ham[8] / ham[4] - 2.0) < 1e-12
    q, _ = _baskin_fourier(0.03 * dalga(4) + 0.02 * dalga(8))
    assert q == 4


@pytest.mark.parametrize(("ne", "q_beklenen"), [(16, 8), (8, 4)], ids=["yari-dolu", "ceyrek-dolu"])
def test_R1_GEVSEMIS_q_dom_REGRESYONU(ne, q_beklenen):
    """Gevsemis zincirin baskin bag modu (REGRESYON; cipa DEGIL, spec 4.3).

    Pencere: N16, a = 2, L = 32, kT = 2e-3, baslangic zincir(16, 2, 0.02, 1),
    yalniz x uzerinde L-BFGS (gtol 1e-10). On kosul (test dogrular): max|F_x|
    < 1e-7 ve TEMIZ rejim (baglar < r_cut, 2. komsu >= r_cut). Olculen:
    yari dolu q = 8 (bag [1.8658, 2.1342], genlik 0.1342), ceyrek dolu q = 4
    (bag [1.9319, 2.0681]); max|F_x| 2.9e-09 / 2.1e-09. YALNIZ q tasinir:
    genlik girdiden geri okunur (0.2684, spec B4). Gevsemis q_dom cipa
    degildir: kapali kabukta bozulma yok, tek N_e'de 5/171 duser
    (elestirmen E2), N >= 48'de polaron.
    """
    y = np.zeros((16, 2))

    def fonk(x):
        s = _om(np.column_stack([x, y]), ne, 2e-3, 32.0)
        return s.omega, -s.kuvvet[:, 0]

    x0 = _zincir(16, 2.0, 0.02, 1)[:, 0]
    secenek = dict(gtol=1e-10, ftol=1e-16, maxiter=20000)
    sonuc = minimize(fonk, x0, jac=True, method="L-BFGS-B", options=secenek)
    R = np.column_stack([sonuc.x, y])
    assert float(np.max(np.abs(_om(R, ne, 2e-3, 32.0).kuvvet[:, 0]))) < 1e-7
    bag = _bag_deseni(R, 32.0)
    assert float(bag.max()) < R_CUT and float(np.min(bag + np.roll(bag, -1))) >= R_CUT
    q, _ = _baskin_fourier(bag)
    assert q == q_beklenen


# ------------------------------------------------ B2: kT = 0 ve dejenerelik --


def test_kT0_KESIRLI_ise_kuvvet_NONE_sifir_DEGIL():
    """kT = 0 ve kesirli -> kuvvet None (spec 2.1(2), B2). 0.0 DEGIL.

    Duzgun N16 yari dolu halka: Fermi cifti 2 katli, f = 1/2; orada tek yanli
    egimler +-0.05/atom, gradyan yok. Tek elektronlu dimer: dejenere olmayan
    tek seviye kesirli; gradyan aslinda var ama kural muhafazakar (modul
    dokstringi). Omega ikisinde de sonlu. KIRMA (olculdu): kesirliyken 0.0
    donduren mutant; kesirli kapisi olmayan mutant (duzgun halkada +-2.8e-17
    'lik sahte bir "denge" kuvveti donuyor).
    """
    s = _om(_zincir(16, 2.0), 16, 0.0, 32.0)
    assert s.kesirli and s.kuvvet is None and math.isfinite(s.omega)
    assert sorted(np.asarray(s.f)[7:9]) == [0.5, 0.5]
    R = np.zeros((2, 3))
    R[1, 0] = 2.0
    tek = _om(R, 1, 0.0)
    assert tek.kesirli and tek.kuvvet is None and math.isfinite(tek.omega)


def test_kT0_KAPALI_KABUKTA_kuvvet_VAR():
    """N10 yari dolu (4n+2) kT = 0: kesirli degil, kuvvet (N, 3) sonlu."""
    s = _om(_zincir(10, 2.0), 10, 0.0, 20.0)
    assert not s.kesirli and s.kuvvet is not None
    assert s.kuvvet.shape == (10, 3) and np.all(np.isfinite(s.kuvvet))


def test_DEJENERELIK_toleransi_EIGH_gurultusunu_KAPSAR():
    """Duzgun N16 (a = 2) Fermi cifti eigh'te 7.1e-17 yarilir; yine de TEK kume.

    Olculen yarilma / (eps_mach max|eps|) = 1.61 (sifir DEGIL: tam esitlik
    kumeyi boler). KIRMA (olculdu): katsayi 0 (tam esitlik) ve mutlak
    tolerans yerine g_s = 1 mutantlari.
    """
    s = _om(_zincir(16, 2.0), 16, 0.0, 32.0)
    eps = np.asarray(s.eps)
    assert eps[8] - eps[7] > 0.0
    assert s.kesirli


def test_DEJENERELIK_toleransi_KOORDINAT_yuvarlamasini_KAPSAR():
    """a = 1.8 halka x ~ 100 bohr otelenmis: Fermi cifti koordinat yuvarlamasiyla yarilir.

    Olculen yarilma / (eps_mach max|eps|): 50 tohumda en cok 41.6 (medyan
    37.1); katsayi 256. Bu kumenin dejenere sayilmasi bir TASARIM
    kararidir (modul yorumu): altinda kalsa kT = 0 tek yanli bir kuvvet
    donerdi. KIRMA (olculdu): katsayi 22 (`eigh` gurultusunun olculmus
    tavani) tohum 0'da cifti boler; katsayi 0 da.
    """
    for tohum in range(5):
        R = _zincir(16, 1.8)
        R[:, 0] += 100.0 * (1.0 + np.random.default_rng(tohum).uniform())
        s = _om(R, 16, 0.0, 28.8)
        assert s.kesirli, tohum


@pytest.mark.parametrize("e", [1e-5, 1e-6, 1e-7])
def test_DEJENERELIK_toleransi_MUTLAK_DEGIL_rcut_alti_dimer(e):
    """Dimer d = r_cut - e, 2 elektron, kT = 0: gercek aufbau (f = [1, 0]).

    Eski MUTLAK 1e-10 tolerans r_cut altinda 2.8e-05 bohr'luk bantta bu
    iki seviyeyi dejenere sayip f = 1/2 yaziyordu (spec B2-2). Yarilma
    (olculdu) 1.275e-11 / 1.275e-13 / 1.275e-15; goreli tolerans max|eps|
    ile olcekli oldugu icin ayirir. Omega = 2 eps_0 + itme tam.
    KIRMA (olculdu): mutlak 1e-10 uc hucrede de kesirli yazar; prototip
    D1 bicimi 64 eps max(1, max|eps|) (1 hartree tabanli) e = 1e-7'de
    kesirli yazar (tolerans 1.4e-14 > yarilma 1.275e-15).
    """
    R = np.zeros((2, 3))
    R[1, 0] = R_CUT - e
    s = _om(R, 2, 0.0)
    eps = np.asarray(s.eps)
    assert not s.kesirli and s.kuvvet is not None
    assert list(np.asarray(s.f)) == [1.0, 0.0]
    assert s.omega == 2.0 * float(eps[0]) + s.E_rep


def test_DEJENERELIK_toleransi_SAHTE_BANDI_dar():
    """40 bohr otede r_cut - 1e-6'daki zayif cift, 16 atomlu zincirin yaninda.

    n = 18: Fermi seviyesi zayif ciftin bag seviyesi. Olculen sahte bant
    (kesirli sayilan en buyuk r_cut - d): katsayi 256'da 2.7e-07, 1024'te
    5.5e-07, 4096'da 1.15e-06; eski mutlak 1e-10'da 2.8e-05.
    KIRMA (olculdu): katsayi 4096 ve mutlak 1e-10 kesirli yaziyor;
    katsayi 1024 geciyor (bant 5.5e-07 < 1e-6). Yani bu test ve
    KOORDINAT testi katsayiyi ~42 ile ~3500 arasina kilitler.
    """
    R = np.zeros((18, 3))
    R[:16, 0] = 2.0 * np.arange(16)
    R[16, 0] = 80.0
    R[17, 0] = 80.0 + R_CUT - 1e-6
    s = _om(R, 18, 0.0)
    assert not s.kesirli and s.kuvvet is not None


@pytest.mark.parametrize("olcek", [1e-9, 1e8])
def test_DEJENERELIK_toleransi_BIRIMSIZ(olcek):
    """mermin_dolum(olcek * eps) ayni dolumu verir: toleransin birimi yok.

    Mutlak bir esik 1e8'de eigh gurultusunu (7.1e-09) ayri seviye sayar,
    1e-9'da ayri seviyeleri (araliklari 7.6e-11) birlestirir.
    """
    eps = np.asarray(_om(_zincir(16, 2.0), 16, 0.0, 32.0).eps)
    ref = mermin_dolum(eps, 16.0, 0.0)
    q = mermin_dolum(olcek * eps, 16.0, 0.0)
    assert q.kesirli == ref.kesirli is True
    assert np.array_equal(np.asarray(q.f), np.asarray(ref.f))


# ------------------------------------------------ kT > 0: fermi D2-D4 --


def test_MU_platonun_ORTASI_bosluk_merkezinde():
    """Bosluklu sistemde kucuk kT'de mu bosluk merkezinden < 2 kT (fermi D2).

    Pencere: 3B kume N8 (M1'inki), n_e = 8, kutusuz, bosluk 1.7532e-02.
    Olculen (mu - orta)/kT = +0.1438 (kT 1e-6, 1e-5, 1e-4) = ln(4/3)/2.
    Bu bir KAYAN NOKTA ARTEFAKTIDIR, fiziksel bir O(kT) duzeltme DEGIL
    (olculdu, 24.09.2026, sk/duz/b2_mu_plato.log): sum f == n olan platonun
    iki ucu HOMO'nun 35.638 kT ustunde ve LUMO'nun 35.351 kT altinda --
    yuvarlama esiklerinin asimetrisi; mu platonun ortasi oldugu icin
    farkin yarisi kadar kayar. HOMO/LUMO dejenere olmadigi icin FIZIKSEL
    kanonik mu bosluk ortasidir (ustel kucuk duzeltmeyle; log uzayinda
    kesin kok ve 120 haneli Decimal ikisi de ortadan <= 3.5e-12 kT uzakta --
    ortanin kendi kayan nokta yuvarlamasi duzeyi).
    O(kT) bir terim yalniz esit olmayan HOMO/LUMO katliliklarinda dogar:
    (kT/2) ln(g_H/g_L). kT = 1e-3'te kuyruklar temsil edilebilir ve kayma
    0. Platonun ALT UCU (prototip) mu'yu HOMO'nun ~36 kT ustune koyar.
    Tolerans 2 kT artefakti AYIRT ETMEZ; isi prototipin hatasini yakalamak.
    Artefaktin penceresi (400 rastgele spektrumda en cok 0.394 kT) ve
    Omega/F'nin ondan etkilenmedigi `mermin.py` dokstringinde.
    KIRMA (olculdu): platonun ALT UCU (prototip) kT = 1e-6'da
    (mu - orta)/kT = -8730.6.
    """
    for kT in (1e-6, 1e-5, 1e-4):
        s = _om(_KUME8, 8, kT)
        eps = np.asarray(s.eps)
        orta = 0.5 * (eps[3] + eps[4])
        assert abs(s.mu - orta) < 2.0 * kT, (kT, (s.mu - orta) / kT)


@pytest.mark.parametrize("x", [0.5, 5.0, 40.0, 100.0, 700.0])
def test_ENTROPI_KARARLI_bicim(x):
    """Iki seviye +-1, 2 elektron, kT = 1/x: entropi bagimsiz referansa goreli < 1e-12.

    Referans seviye basina delta = expit(-|x_i|) (kucuk olan doluluk) ile
    s = -delta ln delta - (1 - delta) log1p(-delta); modulun bicimi
    (log(1 + e^-|x|) + |x| expit(-|x|)) ondan FARKLIDIR. Olculen goreli
    hata <= 1.4e-15. Naif -[f ln f + (1-f) ln(1-f)] (f'den) x >= 40'ta
    %50 hatali (f = 1 - delta 1'e yuvarlanir). KIRMA (olculdu): naif
    bicim x = 40 / 100 / 700'de goreli -0.512 / -0.505 / -0.501; entropide
    spin 1 -0.5.
    """
    kT = 1.0 / x
    d = mermin_dolum(np.array([-1.0, 1.0]), 2.0, kT)
    referans = 0.0
    for e in (-1.0, 1.0):
        kucuk = math.exp(-abs(e - d.mu) / kT)  # tasmaz: alta tasarsa 0
        delta = kucuk / (1.0 + kucuk)
        if delta > 0.0:
            referans += -delta * math.log(delta) - (1.0 - delta) * math.log1p(-delta)
    referans *= 2.0
    assert referans > 0.0
    assert abs(d.entropi / referans - 1.0) < 1e-12, d.entropi / referans - 1.0


@pytest.mark.parametrize(
    ("ne", "kT", "genlik"),
    [(8.0, 1e-7, 0.0), (8.0, 1e-6, 0.0), (8.0, 2e-3, 0.05), (13.3, 1e-3, 0.05)],
)
def test_OMEGA_mu_ya_gore_DURAGAN_buyuk_kanonik_KIMLIK(ne, kT, genlik):
    """Omega = -g_s kT sum ln(1 + e^{-(eps - mu)/kT}) + mu n + E_rep (fermi D4).

    IKI PARCA. (1) KIMLIK: Omega_impl = E_band - kT S + mu (n - n_hesap),
    f = f(mu) oldugu surece HER mu icin cebirsel olarak bu buyuk-kanonik
    ifadeye esittir -- yani kimlik tek basina mu'ya KORDUR; D4 teriminin
    varligini ve entropinin bicimini baglar. Biseksiyon artigi dn'yi
    Omega'ya + mu (n - n_hesap) duzeltmesi sizdirmaz. Metalik N16 N_e = 8
    (Fermi cifti kismen dolu) kT = 1e-7'de dn = -1.4e-10 olculdu: duzeltme
    olmasa sizinti mu dn = 2.0e-11. Olculen fark <= 2.2e-16; tolerans
    1e-14. (2) DURAGANLIK mu'nun DOGRU olmasini ister: N(mu) = g_s sum
    expit((mu - eps)/kT) icin N(mu^-) <= n <= N(mu^+) (mu^-+ bitisik kayan
    noktalar) -- TOLERANSSIZ plato kosulu, modulun mu tanimiyla ayni (plato
    yoksa mu bitisik iki kayan noktadan biridir).
    KIRMA (olculdu): duzeltme kaldirilinca (1) fark -1.96e-11 (kT 1e-7) ve
    -1.96e-12 (kT 1e-6); entropi isareti / spin / yarim entropi mutantlari
    dort hucrede de (1)'i kiriyor. mu'yu +1e-6 kT / +0.3 kT / 4 ulp
    kaydiran mutantlarda (1)'in farki <= 4.4e-16 (kimlik mu'dan bagimsiz);
    (2) uc mutanti da dort hucrede kiriyor.
    """
    s = _om(_zincir(16, 2.0, genlik, 3), ne, kT, 32.0)
    eps = np.asarray(s.eps)
    buyuk = -2.0 * kT * float(np.sum(np.logaddexp(0.0, -(eps - s.mu) / kT)))
    fark = s.omega - (buyuk + s.mu * ne + s.E_rep)
    assert abs(fark) < 1e-14, fark

    def N(m: float) -> float:
        return 2.0 * float(np.sum(expit(-(eps - m) / kT)))

    alt, ust = np.nextafter(s.mu, -np.inf), np.nextafter(s.mu, np.inf)
    assert N(alt) <= ne <= N(ust), (N(alt) - ne, N(s.mu) - ne, N(ust) - ne)


def test_KAYAN_NOKTA_DIBI_parantez_KATLANIR_entropi_KIRPILIR():
    """Mermin'in iki kayan-nokta dibi dali -- fizik penceresi DISI, ama kodun iddiasi.

    (a) kT y ulp(eps)'in altinda: eps = [1, 2], n = 1, kT = 1e-20. mu
    parantezinin ilk siniri uca YUVARLANIR (1 - kT y == 1) ve n = 1 alt uca
    1'den yakin: adim ikiye KATLANMADAN parantez kapanmaz. Beklenen: mu ==
    1.0 ve f == [0.5, 0.0] TAM (plato tek kayan nokta, 2 kT ulp'un
    altinda); model cipasi, doga cipasi degil. (b) kT kayan noktanin
    dibinde (5e-324): x = (eps - mu)/kT tasar (inf) ve |x| kirpilmazsa
    inf * 0 = nan olurdu; entropi == 0.0. kT = 1e-308 bu dali TETIKLEMEZ
    (x sonlu kalir; olculdu). (c) Ortak pencerenin dimeri (d = 2), kT =
    5e-324, n = 1, 2, 3: Omega sonlu. Tasma uyarilari (overflow in divide)
    beklenir ve susturulur.
    KIRMA (olculdu, sk/duz/b8_mut.log): katlamayan mutant (adim *= 1.0)
    (a)'yi ve (c)'nin n = 1 ve 3 hucrelerini kiriyor (`InvalidInputError`:
    parantez kapanmadi); kirpmasiz mutant (ax = |x|) (b)'yi (entropi nan)
    ve (c)'nin uc hucresini (Omega nan) kiriyor. SINANMADI: buyuk kT'de
    adimin tasip sonsuz oldugu yol -- kT = 1e308'de dolum nan donuyor
    (olculdu; acik kalan, `mermin.py` yorumu).
    """
    with np.errstate(over="ignore"):
        d = mermin_dolum(np.array([1.0, 2.0]), 1.0, 1e-20)
        assert d.mu == 1.0 and list(np.asarray(d.f)) == [0.5, 0.0]
        assert mermin_dolum(np.array([1.0, 2.0]), 2.0, 5e-324).entropi == 0.0
        R = np.zeros((2, 3))
        R[1, 0] = 2.0
        for n in (1, 2, 3):
            assert math.isfinite(_om(R, n, 5e-324).omega), n


def test_kT0_MU_tanimi():
    """kT = 0: kapali kabukta mu bosluk ortasi, kesirli kumede kumenin enerjisi, uclarda -+inf."""
    eps = np.array([-2.0, -1.0, 1.0, 3.0])
    assert mermin_dolum(eps, 4.0, 0.0).mu == 0.0
    assert mermin_dolum(eps, 3.0, 0.0).mu == -1.0
    assert mermin_dolum(eps, 0.0, 0.0).mu == -math.inf
    assert mermin_dolum(eps, 8.0, 0.0).mu == math.inf
    assert mermin_dolum(eps, 0.0, 1e-3).mu == -math.inf
    assert mermin_dolum(eps, 8.0, 1e-3).mu == math.inf


@pytest.mark.parametrize("kT", [0.0, 2e-3])
@pytest.mark.parametrize("ne", [0, 32])
def test_UC_DOLULUKLAR_nan_URETMEZ(ne, kT):
    """n = 0 ve n = 2N: f tam 0 / 1, S = 0, Omega sonlu (mu = -+inf ile 0 carpimi yok).

    KIRMA (olculdu): duzeltmeyi korumasiz uygulayan mutant (mu = -+inf
    carpi 0 = nan) ve uc ozel durumu olmayan mutant kT > 0 hucrelerinde.
    """
    s = _om(_zincir(16, 2.0, 0.05, 3), ne, kT, 32.0)
    assert np.all(np.asarray(s.f) == (1.0 if ne else 0.0))
    assert s.entropi == 0.0 and not s.kesirli
    assert math.isfinite(s.omega) and s.kuvvet is not None and np.all(np.isfinite(s.kuvvet))
    assert s.omega == s.E_band + s.E_rep
    if ne == 0:
        assert s.E_band == 0.0


def test_f_EPS_ile_AYNI_sirada():
    """mermin_dolum sirasiz eps alir, f'yi ayni sirada dondurur."""
    eps = np.array([0.3, -0.2, 0.1, -0.5])
    f = np.asarray(mermin_dolum(eps, 3.0, 0.1).f)
    assert f[3] > f[1] > f[2] > f[0]
    assert list(np.asarray(mermin_dolum(eps, 3.0, 0.0).f)) == [0.0, 0.5, 0.0, 1.0]


# ---------------------------------------------------------- girdi kapilari --


def test_CAKISIK_cift_REDDEDILIR():
    """d = 0 cift (dogrudan ya da x goruntusu) InvalidInputError (spec B9).

    KIRMA (olculdu): kapisi olmayan mutant hata firlatmiyor (ve prototip
    gibi yonu tanimsiz bir kuvvet donduruyor).
    """
    R = _zincir(4, 2.0)
    R[2] = R[1]
    with pytest.raises(InvalidInputError):
        _om(R, 4, 2e-3)
    R = _zincir(4, 2.0)
    R[3, 0] = R[0, 0] + 16.0
    with pytest.raises(InvalidInputError):
        _om(R, 4, 2e-3, 16.0)
    # kontrol: ayni geometri kutusuz gecerli (uzakta, d = 16)
    _om(R, 4, 2e-3)


@pytest.mark.parametrize(
    ("degisiklik", "deger"),
    [
        pytest.param("R", np.array([[0.0, 0.0], [2.0, 0.0]]), id="R-sekil-N2"),
        pytest.param("R", np.zeros((0, 3)), id="R-bos"),
        pytest.param("R", np.array([[math.nan, 0.0, 0.0]]), id="R-tek-atom-nan"),
        pytest.param("ne", -1.0, id="ne-negatif"),
        pytest.param("ne", 4.5, id="ne-fazla"),
        pytest.param("ne", math.nan, id="ne-nan"),
        pytest.param("ne", "2", id="ne-dizge"),
        pytest.param("ne", None, id="ne-None"),
        pytest.param("kT", -1e-3, id="kT-negatif"),
        pytest.param("kT", math.inf, id="kT-inf"),
        pytest.param("L", 0.0, id="L-sifir"),
        pytest.param("L", -16.0, id="L-negatif"),
        pytest.param("L", math.nan, id="L-nan"),
        pytest.param("t", "yasa", id="yasa-turu"),
    ],
)
def test_GECERSIZ_girdi_REDDEDILIR(degisiklik, deger):
    """Her ornek, YALNIZ kendi kapisinin yakalayacagi bicimde kurulur.

    OLCULDU: ilk surumde sekil ornegi `np.zeros((3, 2))` ve nan ornegi iki
    atomluydu; R'nin sekil ve sonluluk kapilari mutantla kaldirilinca test
    YINE geciyordu (sifir matris d = 0 kapisina, nan ise radyal yasanin
    mesafe kapisina takiliyordu -- yanlis sebeple yesil). Simdi sekil ornegi
    ayri iki atom (kapisiz: IndexError), nan ornegi TEK atom (cift yok,
    kapisiz: nan konum sessizce yutulur). Bos R'yi mermin_dolum'un eps
    kapisi da yakalar (ikinci kapi). KIRMA (olculdu): R sekil / R sonluluk /
    yasa turu / skaler / kT isareti / L2 kapisi olmayan mutantlarin her biri
    kendi orneginde kiriyor. `ne-None` hucresi `_skaler`in try/except
    dalini baglar (float(None) -> TypeError -> `InvalidInputError`); o dal
    ilk surumde hic calismiyordu. KIRMA (olculdu, 24.09.2026):
    try/except'i kaldiran mutant ham TypeError sizdiriyor -> `ne-None`
    kirmizi.
    """
    R = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    arg = dict(R=R, ss_sigma=SS_SIGMA, itme=ITME, n_elektron=2.0, kT_hartree=1e-3, kutu_x_bohr=None)
    adlar = {"ne": "n_elektron", "kT": "kT_hartree", "L": "kutu_x_bohr", "t": "ss_sigma"}
    anahtar = adlar.get(degisiklik, degisiklik)
    arg[anahtar] = deger
    with pytest.raises(InvalidInputError):
        tek_orbital_mermin(**arg)


@pytest.mark.parametrize(
    ("eps", "ne", "kT"),
    [
        pytest.param(np.array([]), 0.0, 0.0, id="bos"),
        pytest.param(np.zeros((2, 2)), 2.0, 0.0, id="iki-boyutlu"),
        pytest.param(np.array([0.0, math.nan]), 2.0, 1e-3, id="nan-kT"),
        pytest.param(np.array([0.0, math.nan]), 2.0, 0.0, id="nan-kT0"),
        pytest.param(np.array([0.0, 1.0]), 4.1, 1e-3, id="ne-fazla-kT"),
        pytest.param(np.array([0.0, 1.0]), 4.1, 0.0, id="ne-fazla-kT0"),
        pytest.param(np.array([0.0, 1.0]), -1.0, 0.0, id="ne-negatif-kT0"),
        pytest.param(np.array([0.0, 1.0]), 2.0, -1e-9, id="kT-negatif"),
        pytest.param(np.array([0.0, 1.0]), True, 1e-3, id="ne-bool"),
    ],
)
def test_mermin_dolum_GECERSIZ_girdi_REDDEDILIR(eps, ne, kT):
    """kT = 0 ornekleri ayrica var: aufbau aralik disi n'de ve nan eps'te
    hata FIRLATMAZ, yalniz giris kapisi yakalar (olculdu). kT > 0'da mu
    parantezinin katlama tavani ikinci kapidir: n-araligi ya da eps-sonluluk
    kapisi kaldirilinca ilk surumde dongu SONSUZA gidiyordu (300 s zaman
    asimi, olculdu); simdi `InvalidInputError`.
    """
    with pytest.raises(InvalidInputError):
        mermin_dolum(eps, ne, kT)


def test_SONUC_dondurulmus_ve_diziler_SALT_OKUNUR():
    s = _om(_zincir(4, 2.0), 4, 1e-3)
    with pytest.raises(AttributeError):
        s.omega = 0.0  # type: ignore[misc]
    for dizi in (s.eps, s.f, s.kuvvet):
        with pytest.raises(ValueError):
            dizi[0] = 1.0


# ---------------------------------------------------------------------------
# KIRMA KAYDI -- BULGU DUZELTME TURU (24.09.2026, bu makine; sk/duz/mh.py
# duzenegi: src/qlab + reference + scenarios + tests + tools + pyproject
# kopyasi, testler kopyanin icinden PYTHONPATH=<kopya>/src; kopyanin
# yuklendigi once dogrulanir; log sk/duz/mut_mermin*.log). Esdeger kontrol
# mutanti (biseksiyon tavani 400 -> 400 + 0) 101/101 yesil.
#   M5 (Richardson ile yeniden): entropi isareti ters / entropide spin 1 /
#     yarim entropi (-f ln f) -> alti hucrenin altisi (+ DURAGAN kimligi 4/4);
#     E_band'da spin 1 -> alti hucrenin altisi
#   DURAGAN plato satiri: mu + 1e-6 kT / + 0.3 kT / + 4 ulp -> 4/4 hucre
#     (kimlik farki bu mutantlarda <= 4.4e-16: kimlik mu'ya kor)
#   C3 vekil (isaretli pencere): us -> 1.3 us (tutarli) ve dfc unutulmus
#   C3 S_gen (yeni): dfc unutulmus / us x1.3 (tutarli) / rho'da spin 1 ->
#     iki hucrenin ikisi; rho'da spin 1 ayrica C3 pozitif 4/4
#   KAYAN NOKTA DIBI: adim *= 1.0 / ax = |x| -> kirmizi
#   GECERSIZ girdi [ne-None]: _skaler try/except yok -> ham TypeError, kirmizi
#   M4 on kosulu (TEST tarafi): (1, 3) cifti r_cut + 3e-7 -> 5/5 kirmizi;
#     on kosulsuz 2/5 SESSIZCE gecer (x_max, 10)
# ---------------------------------------------------------------------------
