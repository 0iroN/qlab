"""D15 -- `mermin_dolum`un kT > 0'da raporladigi mu KANONIK KOKTUR (log uzayi kok adimi).

ESKI DAVRANIS (OLCULDU, HEAD e262cab, kamu yolu `mermin_dolum`, Decimal 60
hane dogrusal-uzay Newton capraz kontrolu): mu kayan-nokta platosunun
ortasiydi. Bu dosyanin havuzlarinda |mu - mu*|/kT en cok
  derin bosluk (400 spektrum, bosluk/kT 80-2000, 1-3 katlilik)  1.040
  orta bosluk (bosluk/kT 1-30)                                     6.6e-10
  kesirli dolum                                                    5.9e-12
idi; eski dokstringin "en cok 0.394 kT"si katliliksiz havuzundu (ayni
tarifle yeniden olculdu: 0.3942; katlilik ln(3)/2'ye kadar katar). Simdi
ayni uc havuzda 1.5e-11 / 3.5e-13 / 4.2e-12 kT (katliliksizde 4.2e-13).

PENCERE VE TABAN. Seviyeler [-1, 1] hartree'de, bosluk >= 0.01 hartree; yani
kT >= 5e-6 ve ulp(mu)/kT <= 4.4e-11. mu bir double'dir: kT ulp(mu)'ya
yaklasinca sapma ulp(mu) duzeyinde kalir ve kT birimiyle buyur (kayan nokta
dibi, `test_mermin.py`nin KAYAN NOKTA DIBI testi) -- 1e-10 kT iddiasi bu
pencerenin icindir.

REFERANSLAR BAGIMSIZDIR: (1) `brentq` + `scipy.special.logsumexp` /
`log_expit` ile ayni kesin kimlik, ama bolme tamsayi dolumdan (k = round(n/g_s)
en alt seviye), kodunki platonun ortasindan; (2) bir alt kumede Decimal (50
hane) dogrusal uzayda Newton -- log uzayi yok, kuyruklar Decimal'in us
araliginda tam; (3) analitik cipa: derin boslukta mu* = orta + (kT/2)
ln(g_H/g_L).

KIRMA (olculdu, `mut.py`; sayilar commit mesajinda): kok adimini kaldiran
mutant (mu = platonun ortasi) uc testi de kiriyor.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_expit, logsumexp

from qlab.engine.tb.mermin import mermin_dolum

TOL_KT = 1e-10


def _spektrumlar(tur: str, adet: int, tohum: int) -> list[tuple[np.ndarray, float, float]]:
    """(eps, n, kT). derin/orta: tamsayi dolum boslukta; kesirli: rastgele n."""
    rng = np.random.default_rng(tohum)
    out: list[tuple[np.ndarray, float, float]] = []
    while len(out) < adet:
        L = int(rng.integers(4, 33))
        if tur == "kesirli":
            eps = rng.uniform(-1.0, 1.0, size=L)
            out.append((eps, float(rng.uniform(0.2, 2 * L - 0.2)), float(10 ** rng.uniform(-5, -1))))
            continue
        g = rng.integers(1, 4, size=L)
        seviye = np.sort(rng.uniform(-1.0, 1.0, size=L))
        k = int(rng.integers(1, L))
        bosluk = float(seviye[k] - seviye[k - 1])
        if bosluk < 0.01:
            continue
        r = rng.uniform(80.0, 2000.0) if tur == "derin" else rng.uniform(1.0, 30.0)
        out.append((rng.permutation(np.repeat(seviye, g)), 2.0 * float(np.sum(g[:k])), bosluk / r))
    return out


def _kok_brentq(eps: np.ndarray, n: float, kT: float) -> float:
    """Kesin kimlik sum_B f - sum_A h = m, A = en alttaki round(n/2) seviye; log uzayi, brentq.

    Degisken y = (mu - c)/kT, c A/B sinirinin ortasi: `brentq`in rtol tabani
    (4 eps) |y|'ye uygulanir, |mu|'ya degil (mu'da 8.9e-16 |mu| / kT = 1.8e-10
    kT olurdu). (eps - c)/kT bir kez hesaplanir; mu = c + kT y tek yuvarlama.
    """
    nu = n / 2.0
    sira = np.sort(eps)
    k = int(round(nu))
    c = 0.5 * (float(sira[max(k - 1, 0)]) + float(sira[min(k, sira.size - 1)]))
    d_a = (c - sira[:k]) / kT  # x_a = (mu - eps_a)/kT = y + d_a
    d_b = (sira[k:] - c) / kT  # x_b = (eps_b - mu)/kT = d_b - y
    m = nu - k

    def g(y: float) -> float:
        L_h = logsumexp(log_expit(-(y + d_a))) if k else -math.inf
        L_f = logsumexp(log_expit(y - d_b)) if d_b.size else -math.inf
        if m > 0.0:
            return L_f - float(np.logaddexp(math.log(m), L_h))
        if m < 0.0:
            return float(np.logaddexp(math.log(-m), L_f)) - L_h
        return L_f - L_h

    y = brentq(g, (float(sira[0]) - c) / kT - 50.0, (float(sira[-1]) - c) / kT + 50.0,
               xtol=1e-13, rtol=8.9e-16, maxiter=500)
    return c + kT * y


def _kok_decimal(eps: np.ndarray, n: float, kT: float, mu0: float) -> Decimal:
    """Decimal 50 hane, DOGRUSAL uzayda Newton: g_s sum 1/(1 + e^((eps - mu)/kT)) = n."""
    with localcontext() as c:
        c.prec = 50
        E = [Decimal(float(e)) for e in eps]
        T = Decimal(kT)
        nu = Decimal(n) / 2
        mu = Decimal(mu0)
        bir = Decimal(1)
        A = [e for e in E if e < mu]
        B = [e for e in E if e >= mu]
        m = nu - len(A)
        for _ in range(40):
            f = [bir / (bir + ((e - mu) / T).exp()) for e in B]
            h = [bir / (bir + ((mu - e) / T).exp()) for e in A]
            G = sum(f, Decimal(0)) - sum(h, Decimal(0)) - m
            dG = (sum((x * (bir - x) for x in f), Decimal(0)) + sum((x * (bir - x) for x in h), Decimal(0))) / T
            adim = G / dG
            mu -= adim
            if abs(adim) < T * Decimal("1e-30"):
                break
        return mu


def test_D15_RASTGELE_SPEKTRUMLARDA_mu_KANONIK_KOKTE():
    """400 derin + 100 orta bosluklu + 100 kesirli spektrum: |mu - mu*| <= 1e-10 kT.

    mu* bagimsiz `brentq` referansi. HEAD e262cab'de derin havuzda 330/400
    spektrum esigi asiyordu (en cok 1.040 kT), orta havuzda 16/200 (6.6e-10).
    """
    kotu = []
    for tur, adet, tohum in (("derin", 400, 1012), ("orta", 100, 1013), ("kesirli", 100, 1014)):
        for eps, n, kT in _spektrumlar(tur, adet, tohum):
            mu = mermin_dolum(eps, n, kT).mu
            sapma = abs(mu - _kok_brentq(eps, n, kT)) / kT
            if not sapma <= TOL_KT:
                kotu.append((tur, len(eps), n, kT, sapma))
    assert kotu == [], kotu[:5]


def test_D15_DECIMAL_CAPRAZ_KONTROL_ve_REFERANSIN_KENDISI():
    """12 spektrumda (dort tur x uc) Decimal mu* ile kod da brentq referansi da <= 1e-10 kT.

    Decimal dogrusal uzaydadir (log yok); referansin kendi dogrulugunu da baglar.
    """
    for tur, tohum in (("derin", 7), ("orta", 8), ("kesirli", 9)):
        for eps, n, kT in _spektrumlar(tur, 4, tohum):
            mu = mermin_dolum(eps, n, kT).mu
            dec = _kok_decimal(eps, n, kT, mu)
            for ad, deger in (("kod", mu), ("brentq", _kok_brentq(eps, n, kT))):
                sapma = abs(float((Decimal(deger) - dec) / Decimal(kT)))
                assert sapma <= TOL_KT, (tur, ad, len(eps), n, kT, sapma)


def test_D15_ANALITIK_CIPA_esit_olmayan_katlilik_kT_2_ln_gH_gL():
    """eps = [-1, -1, 1, 1, 1] (HOMO 2 kat, LUMO 3 kat), n = 4: mu* = (kT/2) ln(2/3).

    Bosluk/kT = 2/kT >= 200: duzeltme e^(-1/kT) (< 1e-43) temsil edilemez,
    yani mu*/kT = ln(2/3)/2 = -0.2027 tam. Eski plato ortasi (HEAD e262cab)
    burada ortaya (0) yakin oturuyordu ve O(kT) terimi kaciriyordu.
    Ikinci satir ters katlilik: [-1, -1, -1, 1], n = 6 -> (kT/2) ln(3).
    """
    for eps, n, beklenen in (([-1.0, -1.0, 1.0, 1.0, 1.0], 4.0, 0.5 * math.log(2.0 / 3.0)),
                             ([-1.0, -1.0, -1.0, 1.0], 6.0, 0.5 * math.log(3.0))):
        for kT in (1e-2, 3e-3, 1e-3):
            mu = mermin_dolum(np.array(eps), n, kT).mu
            assert abs(mu / kT - beklenen) <= TOL_KT, (eps, kT, mu / kT, beklenen)
