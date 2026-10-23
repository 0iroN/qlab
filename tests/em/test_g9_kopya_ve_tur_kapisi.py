"""Grup 9 serit C -- EM: kopyalar kurucudan gecer (C1), katsayi tiplerinin tur kapisi (C2),
PML'de katsayi kapisinin esigi kayipli semanin gercek esigidir (C3).

C1, C2 SINIF: TRANSKRIPSIYON DENETIMI (kapi kilidi). Fizik iddiasi tasimaz; kapattiklari yollar
HEAD 448f808'de KAMU YOLUNDAN olculdu (sayilar `Medium` ve `YeeCoefficients` dokstringlerinde):
  * `copy.deepcopy(Medium)` (ve pickle protokol 0-4) yazilabilir dizi veriyordu; kopyaya yazilan
    cift negatif hucre `run_1d`de son 50 kaydi 4.2e+40'a cikariyordu.
  * `dataclasses.replace(YeeCoefficients)` kapisizdi; `oblique_coefficients(...).main.cb[:] *= 1.5`
    YERINDE yazma kamu `oblique_step`ini 33. adimda 1e12 katina cikariyordu.
C3 SINIF: SEMA CIPASI, BAGIMSIZ YONTEM -- kapinin tridiagonal geri okumasini, sevk edilen
`oblique_step`in yogun tek adim buyutme matrisinin spektral yaricapiyla karsilastirir.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from qlab.engine.em import oblique, yee
from qlab.engine.em.boundary import PEC, PML
from qlab.engine.em.grid import FieldState
from qlab.engine.em.medium import Medium
from qlab.engine.em.oblique import ObliqueCoefficients, oblique_coefficients, oblique_step
from qlab.engine.em.yee import YeeCoefficients, YeeCoefficients2D, run_1d, yee_coefficients, yee_coefficients_2d
from qlab.errors import InvalidInputError


def _ricker(t: float, lam: float = 20.0) -> float:
    a = (math.pi * (t / lam - 1.5)) ** 2
    return (1.0 - 2.0 * a) * math.exp(-a)


_KOPYALAYICILAR = {
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
    **{f"pickle{p}": (lambda x, p=p: pickle.loads(pickle.dumps(x, protocol=p)))
       for p in range(pickle.HIGHEST_PROTOCOL + 1)},
}


def _salt_okunur_ve_acilamaz(dizi: np.ndarray) -> None:
    assert not dizi.flags.writeable
    zincir = dizi
    while isinstance(zincir, np.ndarray):
        with pytest.raises(ValueError, match="WRITEABLE"):
            zincir.setflags(write=True)
        zincir = zincir.base
    with pytest.raises(ValueError, match="read-only"):
        dizi[1] = -4.0


# ------------------------------------------------------------------------------------- C1 --


def test_C1_MEDIUM_KOPYALARI_KURUCUDAN_GECER_SALT_OKUNUR_ve_KAPILI():
    e2 = np.r_[np.ones(100), np.full(100, 4.0)]
    m = Medium(e2, np.ones(200), "harmonik")
    kw = dict(courant=0.5, boundary=PML(thickness=15), steps=300, source=_ricker, source_index=40,
              probes=(60, 150), wavelength=20.0, sgf=1.0)
    referans = run_1d(m, **kw).probes
    for ad, kopyala in _KOPYALAYICILAR.items():
        k = kopyala(m)
        assert type(k) is Medium and k is not m and k.ey_kural == "harmonik", ad
        for dizi, asil in ((k.eps_r, m.eps_r), (k.mu_r, m.mu_r)):
            _salt_okunur_ve_acilamaz(dizi)
            assert np.array_equal(dizi, asil), ad
        assert np.array_equal(run_1d(k, **kw).probes, referans), ad  # kopya ayni kosumu verir
    # Kurucuyu atlayan bir durum (object.__setattr__) kopyalanirken KAPIYA takilir.
    bozuk = copy.copy(m)
    object.__setattr__(bozuk, "eps_r", np.full(200, -4.0))
    for ad, kopyala in _KOPYALAYICILAR.items():
        with pytest.raises(InvalidInputError, match="pozitif"):
            kopyala(bozuk)


# ------------------------------------------------------------------------------------- C2 --


def test_C2_YEE_KATSAYILARI_TUR_KAPISI_replace_ve_dogrudan_kurulus():
    yc = yee_coefficients(Medium.vacuum(201), courant=0.9, boundary=PEC())
    assert dataclasses.replace(yc).courant == 0.9  # fabrika ciktisi ve kapiya uyan replace gecer
    vakalar = (
        ({"ca": yc.ca * 1.02}, "`ca` KAZANC"),
        ({"da": -1.02 * yc.da}, "`da` KAZANC"),
        ({"cb": -yc.cb}, "`cb` her yerde pozitif"),
        ({"db": 0.0 * yc.db}, "`db` her yerde pozitif"),
        ({"ca": np.full(201, np.nan)}, "`ca` sonlu degil"),
        ({"cb": np.full(201, np.inf)}, "`cb` sonlu degil"),
        ({"courant": -3.0}, "courant"), ({"courant": math.nan}, "courant"), ({"courant": True}, "courant"),
        ({"courant": "0.9"}, "courant"),
        ({"dim": 2}, "dim etiketi"), ({"dim": True}, "dim etiketi"),
        ({"ca": yc.ca[:-1]}, "sekilleri tutarsiz"), ({"db": yc.db[:-1]}, "sekilleri tutarsiz"),
        ({"ca": yc.ca[:2], "cb": yc.cb[:2], "da": yc.da[:1], "db": yc.db[:1]}, "sekilleri tutarsiz"),
        ({"cb": yc.cb + 0j}, "gercel sayi"), ({"ca": np.ones(201, dtype=bool)}, "gercel sayi"),
    )
    for degisen, desen in vakalar:
        with pytest.raises(InvalidInputError, match=desen):
            dataclasses.replace(yc, **degisen)
    with pytest.raises(InvalidInputError, match="KAZANC"):
        YeeCoefficients(ca=np.full(5, 1.5), cb=np.ones(5), da=np.ones(4), db=np.ones(4), dim=1, courant=0.5)
    # Kararlilik (ozdeger) bu kapida YOK (beyanli): etiketi S = 0.9 kalan 1.5 kat cb gecer, kamu
    # tuketicisi `ObliqueCoefficients` onu kendi kesin kapisiyla reddeder.
    hizli = dataclasses.replace(yc, cb=1.5 * yc.cb)
    co = oblique_coefficients(Medium.vacuum(201), courant=0.9, boundary=PEC(), polarization="s", ky=0.0)
    with pytest.raises(InvalidInputError, match="KARARSIZ"):
        dataclasses.replace(co, main=hizli)
    with pytest.raises(InvalidInputError, match="YeeCoefficients olmali"):
        dataclasses.replace(co, main=SimpleNamespace(**{f.name: getattr(co.main, f.name)
                                                          for f in dataclasses.fields(YeeCoefficients)}))


def _buyume(co: ObliqueCoefficients, adim: int = 800) -> float:
    L = co.aux_a.size
    rng = np.random.default_rng(3)
    st = FieldState(rng.standard_normal(L) + 0j, rng.standard_normal(L - 1) + 0j, 1)
    aux = rng.standard_normal(L) + 0j
    n0 = float(np.max(np.abs(st.e)))
    for _ in range(adim):
        st, aux = oblique_step(st, aux, co)
        m = float(np.max(np.abs(st.e)))
        if not np.isfinite(m) or m > 1e12 * n0:
            return math.inf
    return m / n0


def test_C2_KATSAYI_DIZILERI_SALT_OKUNUR_YERINDE_YAZMA_KAPIYI_ATLAYAMAZ():
    vak = Medium.vacuum(201)
    co = oblique_coefficients(vak, courant=0.9, boundary=PEC(), polarization="s", ky=0.0)
    co8 = oblique_coefficients(vak, courant=0.5, boundary=PML(thickness=15), polarization="p", ky=0.8)
    for c in (co, co8):
        for ad in ("ca", "cb", "da", "db"):
            _salt_okunur_ve_acilamaz(getattr(c.main, ad))
        for ad in ("aux_a", "aux_b"):
            _salt_okunur_ve_acilamaz(getattr(c, ad))
    yc = yee_coefficients(vak, courant=0.9, boundary=PEC())
    for ad in ("ca", "cb", "da", "db"):
        _salt_okunur_ve_acilamaz(getattr(yc, ad))
    # Kapatilan yol GERCEKTI: ayni yazma kapisiz bir kopyada (yazilabilir dizi) 1e12'yi asar.
    kapisiz_main = object.__new__(YeeCoefficients)
    for f in dataclasses.fields(YeeCoefficients):
        object.__setattr__(kapisiz_main, f.name, np.array(getattr(co.main, f.name), copy=True)
                           if isinstance(getattr(co.main, f.name), np.ndarray) else getattr(co.main, f.name))
    kapisiz = object.__new__(ObliqueCoefficients)
    for f in dataclasses.fields(ObliqueCoefficients):
        object.__setattr__(kapisiz, f.name, kapisiz_main if f.name == "main" else getattr(co, f.name))
    assert _buyume(kapisiz) < 2.0  # yazmadan once kararli (1.163606)
    kapisiz_main.cb[:] *= 1.5
    assert _buyume(kapisiz) == math.inf


def test_C2_KATSAYI_KOPYALARI_KURUCUDAN_GECER():
    vak = Medium.vacuum(61)
    co8 = oblique_coefficients(vak, courant=0.5, boundary=PML(thickness=15), polarization="p", ky=0.8)
    yc2 = yee_coefficients_2d(Medium.uniform_2d(11, 11), courant=0.5, boundary=PEC())
    for ad, kopyala in _KOPYALAYICILAR.items():
        k = kopyala(co8)
        assert type(k) is ObliqueCoefficients and type(k.main) is YeeCoefficients, ad
        for dizi, asil in ((k.aux_a, co8.aux_a), (k.main.ca, co8.main.ca), (k.main.db, co8.main.db)):
            _salt_okunur_ve_acilamaz(dizi)
            assert np.array_equal(dizi, asil), ad
        k2 = kopyala(yc2)
        assert type(k2) is YeeCoefficients2D and np.array_equal(k2.dby, yc2.dby), ad
    bozuk = copy.copy(co8.main)
    object.__setattr__(bozuk, "ca", np.full(61, 1.5))
    bozuk2 = copy.copy(yc2)
    object.__setattr__(bozuk2, "dax", np.full((11, 10), 1.5))
    for ad, kopyala in _KOPYALAYICILAR.items():
        with pytest.raises(InvalidInputError, match="`ca` KAZANC"):
            kopyala(bozuk)
        with pytest.raises(InvalidInputError, match="`dax` KAZANC"):
            kopyala(bozuk2)


def test_C2_YEE_2B_TUR_KAPISI():
    yc2 = yee_coefficients_2d(Medium.uniform_2d(11, 9), courant=0.5, boundary=PEC())
    assert dataclasses.replace(yc2).dim == 2
    vakalar = (
        ({"ca": 1.02 * yc2.ca}, "2B guncelleme katsayisi `ca` KAZANC"),
        ({"day": -1.02 * yc2.day}, "`day` KAZANC"),
        ({"dby": -yc2.dby}, "`dby` her yerde pozitif"),
        ({"dbx": np.full((11, 8), np.nan)}, "`dbx` sonlu degil"),
        ({"dax": yc2.dax[:, :-1]}, "sekilleri tutarsiz"),
        ({"day": yc2.day.T}, "sekilleri tutarsiz"),
        ({"pec_mask": yc2.pec_mask.astype(int)}, "pec_mask"),
        ({"pec_mask": yc2.pec_mask[:-1]}, "pec_mask"),
        ({"dim": 1}, "dim etiketi"), ({"courant": 0.0}, "courant"),
    )
    for degisen, desen in vakalar:
        with pytest.raises(InvalidInputError, match=desen):
            dataclasses.replace(yc2, **degisen)


# ------------------------------------------------------------------------------------- C3 --


@pytest.fixture
def politika_kapilari_kapali(monkeypatch):
    """YALNIZ politika kapilari kapatilir (izotropik S <= 1, egik S_max(ky), etkin Courant, p'nin
    ortamdan kurulan kesin esigi). `ObliqueCoefficients`in KENDI ozdeger kapisi (`_katsayi_kapisi`)
    ACIK kalir -- sinanan odur. Esik olcumu kapiyi atlamak zorundadir (`grid.courant_limit` gerekcesi)."""
    monkeypatch.setattr(yee, "courant_limit", lambda d: math.inf)
    monkeypatch.setattr(yee, "_etkin_courant_kapisi", lambda *a, **k: None)
    monkeypatch.setattr(oblique, "_etkin_courant_kapisi", lambda *a, **k: None)
    monkeypatch.setattr(oblique, "oblique_courant_limit", lambda ky: math.inf)
    monkeypatch.setattr(oblique, "_egik_p_kesin_esik", lambda m, ky: math.inf)


def _spektral_yaricap(co: ObliqueCoefficients) -> float:
    """Sevk edilen `oblique_step`in tek adim buyutme matrisi -- (e, h, aux) birim vektorlerinden yogun kurulur."""
    L = co.aux_a.size
    n = 3 * L - 1
    A = np.empty((n, n), dtype=complex)
    for j in range(n):
        v = np.zeros(n, dtype=complex)
        v[j] = 1.0
        st, aux = oblique_step(FieldState(v[:L].copy(), v[L:2 * L - 1].copy(), 1), v[2 * L - 1:].copy(), co)
        A[:, j] = np.concatenate((st.e, st.h, aux))
    return float(np.max(np.abs(np.linalg.eigvals(A))))


_L3 = 41
_YARI = np.r_[np.full(_L3 // 2, 4.0), np.full(_L3 - _L3 // 2, 0.25)]


@pytest.mark.parametrize(
    ("ortam", "pol", "ky"),
    [
        pytest.param(lambda: Medium.vacuum(_L3), "s", 0.8, id="vakum-s-ky08"),
        pytest.param(lambda: Medium(_YARI, _YARI[::-1].copy(), "harmonik"), "p", 0.0, id="eps-mu-zit-harmonik-p"),
        pytest.param(lambda: Medium.half_space(_L3, interface=20, eps1=1.0, eps2=4.0), "p", 0.5, id="eps1-4-p-ky05"),
    ],
)
def test_C3_PML_KAPI_KABUL_SINIRI_KAYIPLI_SEMANIN_GERCEK_ESIGI(politika_kapilari_kapali, monkeypatch, ortam, pol, ky):
    """PML (derecelendirilmis, TEKDIZE OLMAYAN kayip) icin ispat yazilmamisti; olculdu (C3).

    Kapinin KABUL SINIRI ikiye bolmeyle bulunur (60 adim, gercek kapi acik). Bagimsiz yontem:
    `oblique_step`in yogun tek adim buyutme matrisi. OLCULDU (L = 41, PML 10):
      sinirda rho - 1          1.05e-09 / 6.32e-06 / 2.69e-10   (kapi: <= 2e-5)
      sinir x (1 + 1e-6)'da     2.10e-04 / 2.83e-03 / 5.39e-05   (kapi: >= 1e-5)
    6.32e-6 kayan nokta payinin (1e-11) bedelidir: tepe kipi kayipsiz bolgedeyken rho - 1 ~
    2 sqrt(pay). Pay 1e-9'a gevseyince ayni vakada 6.32e-05, 1e-6'da 2.0e-03 (oldurulen mutantlar).
    """
    med = ortam()
    sinir = PML(thickness=10)

    def kur(S: float) -> ObliqueCoefficients:
        return oblique_coefficients(med, courant=S, boundary=sinir, polarization=pol, ky=ky)

    def kabul(S: float) -> bool:
        try:
            kur(S)
        except InvalidInputError:
            return False
        return True

    lo, hi = 0.05, 3.0
    assert kabul(lo) and not kabul(hi)
    for _ in range(60):
        orta = 0.5 * (lo + hi)
        lo, hi = (orta, hi) if kabul(orta) else (lo, orta)
    co = kur(lo)
    assert np.min(co.main.ca) < 0.99, "oncul: PML gercekten kayipli (derecelendirilmis sigma)"
    assert _spektral_yaricap(co) - 1.0 <= 2e-5, "kapinin kabul ettigi S kararsiz"
    with monkeypatch.context() as mp:
        mp.setattr(oblique, "_katsayi_kapisi", lambda c: None)
        assert _spektral_yaricap(kur(hi * (1.0 + 1e-6))) - 1.0 >= 1e-5, "kapi gereksiz temkinli"
