"""G13 -- EM kararlilik kapilarinin ATLAMA YOLLARI kapandi (12.10.2026).

SINIF: TRANSKRIPSIYON DENETIMI (kapi kilidi). Fizik iddiasi tasimaz; her
testin kapattigi yol HEAD e262cab'de KAMU YOLUNDAN olculdu ve sayilari ilgili
kaynak dokstringindedir (`Medium`, `ObliqueCoefficients`, `boundary.
_katman_kapisi`, `yee._sonlu_katsayi_kapisi`):

  * Medium dizileri kurulumdan sonra yazilabiliyordu (cift negatif eps/mu
    etkin Courant kapisini gecip alani 2.46e+49'a cikariyordu) ve cagiranin
    dizisiyle takma addi; eps_r = inf kabul ediliyordu.
  * `dataclasses.replace(ObliqueCoefficients)` kapisizdi: S = 0.9, ky = 0
    katsayilari ky = 1.2'ye tasininca 45. adimda 1.1e+12 kat.
  * LossyLayer(sigma_max / m = nan, inf, <= 0), PML(m = nan / inf), kesirli /
    bool kalinlik kabul; CircularPEC(radius = inf) kabul.
  * Alt-normal eps x dev mu: hucre carpimi kapisi gecer, cb = inf, kayit nan.
  * (dogrulayici turu) Sonum yuvalari ca / da / aux_a > 1 (kazanc) ozdeger kapisini
    geciyordu; ozel sinirin negatif sigmasi 1B/2B'de kazanc kuruyordu; db / dbx / dby
    tasmasi kilitsizdi; `setflags(write=True)` Medium dizisini geri aciyordu.

PENCERE: vakum ya da iki bolgeli ortam, PEC ya da PML, S <= 0.9, en cok 800
adim. Dinamik kanitlar kapiyi `object.__new__` ile ATLAYARAK sevk edilen
`oblique_step`i kosar: kapinin reddettigi sey gercekten patlar.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qlab.engine.em import yee
from qlab.engine.em.boundary import PEC, PML, CircularPEC, LossyLayer
from qlab.engine.em.grid import FieldState
from qlab.engine.em.medium import Medium
from qlab.engine.em.oblique import ObliqueCoefficients, oblique_coefficients, oblique_step
from qlab.engine.em.yee import run_1d, run_2d
from qlab.errors import InvalidInputError


def _ricker(t: float, lam: float = 20.0) -> float:
    a = (math.pi * (t / lam - 1.5)) ** 2
    return (1.0 - 2.0 * a) * math.exp(-a)


def _kapisiz(co: ObliqueCoefficients, **degisen) -> ObliqueCoefficients:
    """`replace`in KAPISIZ esi -- yalniz dinamik kanit icin (kapi atlanir)."""
    yeni = object.__new__(ObliqueCoefficients)
    for alan in dataclasses.fields(ObliqueCoefficients):
        object.__setattr__(yeni, alan.name, degisen.get(alan.name, getattr(co, alan.name)))
    return yeni


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


def test_G13_MEDIUM_DIZILERI_SALT_OKUNUR_KOPYA_ve_SONLU():
    eps, mu = np.ones(200), np.ones(200)
    m = Medium(eps, mu)
    for dizi in (m.eps_r, m.mu_r):
        assert not dizi.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        m.eps_r[100] = -4.0
    with pytest.raises(ValueError, match="read-only"):
        m.mu_r[100] = -4.0
    eps[100] = -4.0  # cagiranin dizisi: takma ad kopuk
    assert m.eps_r[100] == 1.0 and m.eps_min == 1.0
    for kurulus in (Medium.vacuum(10), Medium.half_space(10, interface=5, eps2=4.0),
                    Medium.uniform_2d(5, 5, eps_r=2.25)):
        assert not kurulus.eps_r.flags.writeable and not kurulus.mu_r.flags.writeable
    for ad, e, u in (("eps_r", np.full(10, np.inf), np.ones(10)), ("mu_r", np.ones(10), np.full(10, np.inf))):
        with pytest.raises(InvalidInputError, match=f"{ad} her yerde sonlu"):
            Medium(e, u)
    with pytest.raises(InvalidInputError, match="pozitif"):
        Medium(np.full(10, np.nan), np.ones(10))
    # Kopya sayilari degistirmez: ayni diziyle iki kurulus bit bit ayni kosum.
    kw = dict(courant=0.5, boundary=PML(thickness=15), steps=300, source=_ricker, source_index=40,
              probes=(60, 150), wavelength=20.0, sgf=1.0)
    e2 = np.r_[np.ones(100), np.full(100, 4.0)]
    assert np.array_equal(run_1d(Medium(e2, np.ones(200)), **kw).probes,
                          run_1d(Medium(e2.copy(), np.ones(200)), **kw).probes)


def test_G13_OBLIQUE_REPLACE_ve_DOGRUDAN_KURULUS_KAPIDAN_GECER():
    vak = Medium.vacuum(201)
    co = oblique_coefficients(vak, courant=0.9, boundary=PEC(), polarization="s", ky=0.0)
    assert _buyume(co) < 2.0  # ky = 0'da kararli (buyume 1.163606)
    # (1) ky tasima: fabrikanin egik politika kapisinin aynisi -- ve gercekten patliyor.
    with pytest.raises(InvalidInputError, match="EGIK indirgemenin"):
        dataclasses.replace(co, ky=1.2)
    assert _buyume(_kapisiz(co, ky=1.2)) == math.inf
    # (2) Politika kapisini gecen ama ADIMLANAN DIZILERI kararsiz olan iki yol:
    #     etiketi S = 0.9 kalan 1.5 kat cb, ve isareti cevrilmis aux_b (ky = 0.8).
    hizli = dataclasses.replace(co.main, cb=co.main.cb * 1.5)
    with pytest.raises(InvalidInputError, match="KARARSIZ"):
        dataclasses.replace(co, main=hizli)
    assert _buyume(_kapisiz(co, main=hizli)) == math.inf
    co8 = oblique_coefficients(vak, courant=0.5, boundary=PML(thickness=15), polarization="p", ky=0.8)
    assert _buyume(co8) < 2.0
    with pytest.raises(InvalidInputError, match="operator"):
        dataclasses.replace(co8, aux_b=-co8.aux_b)
    assert _buyume(_kapisiz(co8, aux_b=-co8.aux_b)) == math.inf
    # ky^2 terimi: 20 kat aux_b -> S^2 lambda ~ 0.25 (4 + 0.64 * 20) = 4.2 (enine kol).
    with pytest.raises(InvalidInputError, match="KARARSIZ"):
        dataclasses.replace(co8, aux_b=20.0 * co8.aux_b)
    assert _buyume(_kapisiz(co8, aux_b=20.0 * co8.aux_b)) == math.inf
    # (3) Etiket ve sekil kapilari.
    for degisen, desen in (({"polarization": "q"}, "polarization"), ({"ky": math.nan}, "ky sonlu"),
                           ({"sign": 0.5}, "sign"), ({"aux_a": co.aux_a[:-1]}, "sekilleri")):
        with pytest.raises(InvalidInputError, match=desen):
            dataclasses.replace(co, **degisen)
    # Fabrikanin kendi ciktisi ve kapiya uyan replace GECER (sign bir gauge'dir).
    assert dataclasses.replace(co8, sign=-co8.sign).ky == 0.8
    # (4) KAYIPLI yuvalar: esik kayipsizinkine ESITTIR (olculdu, L = 41, tekdize
    #     kayip d = 0.3 / 1 / 3: 0.99 S_krit soner, 1.01 S_krit buyur) -- kapi
    #     kayipsiz katsayiyi 2 cb / (1 + ca) ile geri okur. Etiket S = 0.9.
    L, d = 41, 1.0
    s_krit = 1.0 / math.cos(math.pi / (2 * (L - 1)))

    def kayipli(S: float, kapili: bool = True) -> ObliqueCoefficients:
        a, b = (1 - d) / (1 + d), S / (1 + d)
        alan = dict(main=dataclasses.replace(co.main, ca=np.full(L, a), cb=np.full(L, b), da=np.full(L - 1, a),
                                             db=np.full(L - 1, b)),
                    aux_a=np.full(L, a), aux_b=np.full(L, b), sign=-1.0, polarization="s", ky=0.0)
        return ObliqueCoefficients(**alan) if kapili else _kapisiz(co, **alan)

    assert kayipli(0.99 * s_krit).ky == 0.0
    with pytest.raises(InvalidInputError, match="KARARSIZ"):
        kayipli(1.01 * s_krit)
    assert _buyume(kayipli(1.01 * s_krit, kapili=False), adim=2000) == math.inf


def test_G13_KATMAN_ve_DAIRE_PARAMETRE_KAPILARI():
    nan, inf = math.nan, math.inf
    for kur, desen in ((lambda: LossyLayer(sigma_max=nan), "sigma_max"),
                       (lambda: LossyLayer(sigma_max=inf), "sigma_max"),
                       (lambda: LossyLayer(sigma_max=-1.0), "sigma_max"),
                       (lambda: LossyLayer(m=nan), "m > 0"), (lambda: LossyLayer(m=-1.0), "m > 0"),
                       (lambda: LossyLayer(m=0.0), "m > 0"), (lambda: LossyLayer(m=inf), "m > 0"),
                       (lambda: LossyLayer(thickness=15.5), "tamsayi"),
                       (lambda: PML(m=nan), "m > 0"), (lambda: PML(m=inf), "m > 0"),
                       (lambda: PML(m=1e308, r0=1e-300), "sigma_max sonlu"),
                       (lambda: PML(thickness=15.5), "tamsayi"), (lambda: PML(thickness=True), "tamsayi"),
                       (lambda: PML(thickness=nan), "tamsayi"), (lambda: PML(thickness=-1), "kalinl"),
                       (lambda: CircularPEC(radius=inf), "sonlu")):
        with pytest.raises(InvalidInputError, match=desen):
            kur()
    # Sevk edilen degerler ve numpy tamsayisi gecer.
    assert PML(thickness=np.int64(15)).sigma_1d(60).shape == (60,)
    assert LossyLayer().sigma_max == 0.35 and PML().thickness == 15


def test_G13_SONLU_KATSAYI_KAPISI_ALT_NORMAL_eps_ve_nan_SIGMA():
    eps, mu = np.ones(200), np.ones(200)
    eps[100], mu[100] = 2e-309, 1.5e308  # carpim 0.3 >= S^2 = 0.25: etkin kapi gecer
    kw = dict(courant=0.5, steps=50, source=_ricker, source_index=40, probes=(60,), wavelength=20.0, sgf=1.0)
    e2, m2 = np.ones((41, 41)), np.ones((41, 41))
    e2[20, 20], m2[20, 20] = 1.2e-309, 1.79e308  # 2B: carpim 0.215, S = 0.3 -> etkin 0.647 < 0.707
    with np.errstate(over="ignore"):  # S/eps'in tasmasi beklenir; kapi onu reddeder
        with pytest.raises(InvalidInputError, match="`cb` sonlu degil"):
            run_1d(Medium(eps, mu), boundary=PEC(), **kw)
        with pytest.raises(InvalidInputError, match="`cb` sonlu degil"):
            run_2d(Medium(e2, m2), courant=0.3, boundary=PEC(), steps=10, source=_ricker,
                   source_index=(10, 10), probes=((30, 30),), wavelength=20.0, sgf=1.0)

    class NanSigma:  # kapisiz ozel sinir
        thickness = 0

        def sigma_1d(self, cells):
            s = np.zeros(cells)
            s[5] = math.nan
            return s

    with pytest.raises(InvalidInputError, match="sonlu degil"):
        run_1d(Medium.vacuum(200), boundary=NanSigma(), **kw)
    # Sonlu uc olcek reddedilmez: eps 1e-300 x mu 1e300 (cb = 5e299).
    eps[100], mu[100] = 1e-300, 1e300
    assert run_1d(Medium(eps, mu), boundary=PEC(), **kw).probes.shape == (1, 50)


# ----------------------------------------- G13 duzeltmesi: KAZANC kapisi --


def test_G13_KAZANC_YUVALARI_ca_da_aux_a_REDDEDILIYOR_ve_GERCEKTEN_PATLIYOR():
    """Sonum yuvalari |a| <= 1 (dogrulayici, 12.10.2026): ozdeger okumasi 2 c / (1 + a)
    kazanci goremiyordu -- ca x 1.02, da x 1.02, aux_a x 1.05 ve (ca = -3, cb < 0) kapidan
    geciyordu. Her yuva AYRI vaka: biri kapidan duserse yalniz onun vakasi kirmizi.
    """
    vak = Medium.vacuum(201)
    co = oblique_coefficients(vak, courant=0.9, boundary=PEC(), polarization="s", ky=0.0)
    co8 = oblique_coefficients(vak, courant=0.5, boundary=PML(thickness=15), polarization="p", ky=0.8)
    assert np.max(co8.aux_a) == 1.0 and np.min(co8.aux_a) < 1.0  # PML: kayipli aux yuvasi
    m = co.main
    vakalar = (
        ("ca", co, {"main": dataclasses.replace(m, ca=m.ca * 1.02)}),
        ("da", co, {"main": dataclasses.replace(m, da=m.da * 1.02)}),
        ("aux_a", co8, {"aux_a": co8.aux_a * 1.05}),
        # geri okunan 2 cb / (1 + ca) = 0.9 > 0: ozdeger kapisi gecerdi
        ("ca", co, {"main": dataclasses.replace(m, ca=np.full(201, -3.0), cb=np.full(201, 0.45 * (1.0 - 3.0)))}),
    )
    for ad, taban, degisen in vakalar:
        with pytest.raises(InvalidInputError, match=f"`{ad}` KAZANC"):
            dataclasses.replace(taban, **degisen)
        assert _buyume(_kapisiz(taban, **degisen)) > 1e3, ad  # olculdu: 4.1e3 / 4.1e3 / inf / inf
    # Fabrika ciktisi gecer: tam olarak |a| <= 1 (PML'li p dahil).
    for c in (co, co8):
        assert dataclasses.replace(c).ky == c.ky


def test_G13_KAYIPLI_AUX_YUVASI_GERI_OKUNUYOR_ky_08():
    """Tekdize kayip d = 1, L = 41, ky = 0.8, s ve p: kapinin esigi kayipli semanin esigidir
    (0.99 soner, 1.01 buyur). ky^2 terimi aux yuvasindan gelir; ham aux_b okunsaydi esik
    1.036 kat gevserdi ve 1.01 gecerdi (olculdu) -- bu vaka o geri okumayi kilitler.
    """
    co = oblique_coefficients(Medium.vacuum(201), courant=0.9, boundary=PEC(), polarization="s", ky=0.0)
    L, d, ky = 41, 1.0, 0.8
    s_krit = 2.0 / math.sqrt(4.0 * math.cos(math.pi / (2 * (L - 1))) ** 2 + ky * ky)  # 0.929094

    def kayipli(S: float, pol: str, kapili: bool = True) -> ObliqueCoefficients:
        a, b = (1 - d) / (1 + d), S / (1 + d)
        alan = dict(main=dataclasses.replace(co.main, ca=np.full(L, a), cb=np.full(L, b), da=np.full(L - 1, a),
                                             db=np.full(L - 1, b)),
                    aux_a=np.full(L, a), aux_b=np.full(L, b), sign=-1.0 if pol == "s" else 1.0,
                    polarization=pol, ky=ky)
        return ObliqueCoefficients(**alan) if kapili else _kapisiz(co, **alan)

    for pol in ("s", "p"):
        assert _buyume(kayipli(0.99 * s_krit, pol), adim=2000) < 1.0
        with pytest.raises(InvalidInputError, match="KARARSIZ"):
            kayipli(1.01 * s_krit, pol)
        assert _buyume(kayipli(1.01 * s_krit, pol, kapili=False), adim=2000) == math.inf


class _NegatifSigma:
    """Kapisiz ozel sinir: verilen dilimde negatif sigma (1B ve 2B)."""

    thickness = 0

    def __init__(self, dilim, deger: float = -0.5):
        self.dilim, self.deger = dilim, deger

    def sigma_1d(self, cells):
        s = np.zeros(cells)
        s[self.dilim] = self.deger
        return s

    def sigma_2d(self, shape):
        s = np.zeros(shape)
        s[self.dilim] = self.deger
        return s

    def pec_mask_2d(self, shape):
        return PEC().pec_mask_2d(shape)


def test_G13_YEE_KAZANC_KAPISI_NEGATIF_SIGMA_1B_2B(monkeypatch):
    """Negatif sigma donduren ozel sinir: ca = 1.286 > 1. Kapi 1B ve 2B'de reddeder;
    kapi kaldirilinca kayit gercekten patlar (olculdu: 1B 6.05e+21, 2B 841)."""
    kw1 = dict(courant=0.5, steps=600, source=_ricker, source_index=40, probes=(60, 150), wavelength=20.0, sgf=1.0)
    kw2 = dict(courant=0.5, steps=300, source=_ricker, source_index=(10, 10), probes=((30, 30),),
               wavelength=20.0, sgf=1.0)
    kosular = (
        (lambda b: run_1d(Medium.vacuum(200), boundary=b, **kw1), _NegatifSigma(slice(95, 106)), 1e20),
        (lambda b: run_2d(Medium(np.ones((41, 41)), np.ones((41, 41))), boundary=b, **kw2),
         _NegatifSigma((20, 20)), 1e3),
    )
    for kos, sinir, kat in kosular:
        referans = float(np.max(np.abs(kos(PEC()).probes)))
        with pytest.raises(InvalidInputError, match="`ca` KAZANC"):
            kos(sinir)
        with monkeypatch.context() as mp:
            mp.setattr(yee, "_SONUM_YUVALARI", ())
            with np.errstate(all="ignore"):
                assert float(np.max(np.abs(kos(sinir).probes))) > kat * referans
    # Her sonum yuvasi AYRI: fabrika yolunda da > 1 ancak ca > 1 ile birlikte olusur (d_h,
    # d_e'nin ortalamasi), yani `da`/`dax`/`day`i yalniz kapinin kendisi ayirir.
    bir = np.ones(4)
    for yol, ad, digerleri in (("1B", "da", ("ca", "cb", "db")),
                               ("2B", "dax", ("ca", "cb", "dbx", "day", "dby")),
                               ("2B", "day", ("ca", "cb", "dax", "dbx", "dby"))):
        for kat in (1.02, -1.02):  # a < -1: isaret degistirerek buyur
            with pytest.raises(InvalidInputError, match=f"`{ad}` KAZANC"):
                yee._sonlu_katsayi_kapisi(yol, **{k: bir for k in digerleri}, **{ad: kat * bir})
        yee._sonlu_katsayi_kapisi(yol, **{k: bir for k in digerleri}, **{ad: -bir})  # |a| = 1 gecer


def test_G13_SONLU_KAPI_db_ve_2B_dbx_dby_YUVALARI():
    """VA1: `cb` disindaki yuvalar da sinanir. Dev eps x alt-normal mu: carpim kapisi gecer,
    cb sonlu, ama db = S/mu tasar (1B: 3 hucre; 2B: j boyunca serit -> yalniz dbx, i boyunca
    serit -> yalniz dby)."""
    eps, mu = np.ones(200), np.ones(200)
    eps[100:103], mu[100:103] = 1.5e308, 2e-309  # carpim 0.3 >= S^2 = 0.25
    kw = dict(courant=0.5, steps=50, source=_ricker, source_index=40, probes=(60,), wavelength=20.0, sgf=1.0)
    kw2 = dict(courant=0.3, boundary=PEC(), steps=10, source=_ricker, source_index=(10, 10), probes=((30, 30),),
               wavelength=20.0, sgf=1.0)
    with np.errstate(over="ignore"):
        with pytest.raises(InvalidInputError, match="`db` sonlu degil"):
            run_1d(Medium(eps, mu), boundary=PEC(), **kw)
        for dilim, ad in (((20, slice(15, 26)), "dbx"), ((slice(15, 26), 20), "dby")):
            e2, m2 = np.ones((41, 41)), np.ones((41, 41))
            e2[dilim], m2[dilim] = 1.79e308, 1.2e-309  # carpim 0.215, S = 0.3 -> etkin 0.647 < 0.707
            with pytest.raises(InvalidInputError, match=f"`{ad}` sonlu degil"):
                run_2d(Medium(e2, m2), **kw2)


def test_G13_MEDIUM_SALT_OKUNURLUGU_setflags_ILE_GERI_ACILAMAZ():
    """`setflags(write=False)` yetmiyordu: `setflags(write=True)` diziyi geri aciyor ve cift
    negatif hucre alani 4.2e+40'a cikariyordu. Zincirdeki HER ndarray (gorunum ve tabani)
    yazmaya acilmayi reddetmeli -- salt okunur tabanin gorunumu `.base` uzerinden acilirdi."""
    for m in (Medium.vacuum(200), Medium.uniform_2d(5, 6, eps_r=2.25)):
        for dizi in (m.eps_r, m.mu_r):
            zincir = dizi
            while isinstance(zincir, np.ndarray):
                with pytest.raises(ValueError, match="WRITEABLE"):
                    zincir.setflags(write=True)
                zincir = zincir.base
            assert not dizi.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                dizi[0] = -4.0
    assert Medium.vacuum(200).eps_r.shape == (200,) and Medium.uniform_2d(5, 6).mu_r.shape == (5, 6)
