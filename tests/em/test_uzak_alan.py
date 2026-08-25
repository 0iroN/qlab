"""Uzak alan — **İKİ AYRI GÖZLEMLENEBİLİR**, ve TEK YARIK çıpası.

Bu dosyanın çekirdek iddiası spec §7.5'te ölçülmüştür ve burada yeniden
ölçülüyor: `aperture_far_field` ile `ntff_far_field` **"iki yöntem" değildir**,
çünkü aynı problemi çözmezler.

- **AÇIKLIKTA** (tek yarık) kapalı kontur **yapısal olarak kurulamaz**: ekran
  sonsuza uzanır, dolayısıyla ekran akımlarının bir kısmı **her sonlu
  konturun dışında** kalır. Zorlanınca kontur *kurulabilir* ama cevabı
  **konturun yerine bağlı** olur — ölçüldü, bu dosyada.
- **SAÇILMADA tam tersi**: `tests/em/test_sacilma.py`.

DOSYANIN İKİ SINIFI, ve karıştırılamazlar:

- **ÖLÇÜM ALETİNİN KAPILARI** — bin kapısı, Fraunhofer kapısı, kontur kapanma
  kapısı, işaret uzlaşımı. Bunlar şema/fizik çıpası **değil**, aletin kendi
  imkân koşullarıdır.
- **FİZİK ÇIPASI** — tek yarık `a·sin(θ₁)/λ = 1`. Bu sayı girilen
  parametrelerden okunamaz: `a` ve `λ` **ayrı ayrı** taranır ve çarpım 1'de
  kalır.

**ÇIPA MERTEBE HÜKMÜ OLMADAN SEVK EDİLİR.** Task 2'nin ölçütü (`convergence_
ladder`) bu çıpanın merdiveninde **YARGILANAMAZ** diyor ve sebebi bu dosyada
ölçülü: ardışık farklar **işaret değiştiriyor**. Panel bu çıpa için
"YAKINSAMA: YARGILANAMAZ" basar. (Modül 4'ün "bir merdiven bir tamsayıya
oturmak zorunda değildir" dersinin ikinci uygulaması.)

PENCERE (bu dosyadaki her yarık koşumu için):
`dx = 1`, `c = ε₀ = μ₀ = 1`, `S = 0.5`, **kaynak terimi YOK** — ızgara
+x'e giden bir Gauss darbeli **düzlem dalga ile BAŞLATILIYOR** (`run_2d`ye
`state=`), y'de düz tepeli (8. derece süper-Gauss, yarı genişlik
`w = max(3a, 120)`); `sgf = 1`, tek hücre kalınlığında PEC ekran, dört
kenarda `npml = 20`, açıklık yüzü ekranın **2 hücre** arkası, kayıt uzunluğu
`λ`'nın tam katı (FFT bini `1/λ`ya **tam** oturuyor — dosyadaki her koşumda
`f_bin·λ = 1.000000000`).

NEDEN NOKTA KAYNAK DEĞİL — ÖLÇÜLDÜ: yumuşak nokta kaynağı silindirik bir
dalga üretir ve Fraunhofer koşulunu **kaynak tarafında** da zorlar. Aynı
çıpa nokta kaynakla `d = 60/120/240/480` hücre uzaklıkta 0.911 / 0.940 /
0.975 / 0.996 veriyor (a = 40, λ = 16 ve 20), yani düzlem dalga sınırına
ancak `d ≳ 500`de yaklaşıyor ve o ızgara **16.5 s/yapılandırma** ediyor.
Düzlem dalga başlangıç koşulu aynı pencereyi **0.3–2 s**de veriyor.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.special import hankel2

from qlab.engine.em.boundary import PML
from qlab.engine.em.grid import FieldState2D
from qlab.engine.em.medium import Medium
from qlab.engine.em.source import gauss_cos
from qlab.engine.em.yee import run_2d
from qlab.errors import InvalidInputError
from qlab.observable.em_obs import (
    BIN_GATE,
    FRAUNHOFER_LAMBDA,
    _kapali_mi,
    aperture_far_field,
    convergence_ladder,
    ntff_far_field,
)

S = 0.5

#: Çıpanın penceresi. `a/λ < 2`'de bozuluyor (aşağıda ölçüldü).
PENCERE_ORANI = 2.0

#: Sevk edilen (a, λ) kümesi — hepsi `a/λ ≥ 2`.
AL = (
    (32, 12.0), (32, 16.0), (40, 12.0), (40, 16.0), (40, 20.0),
    (48, 16.0), (48, 20.0), (48, 24.0), (64, 16.0), (64, 20.0),
    (64, 24.0), (80, 20.0), (80, 24.0), (96, 24.0),
)
#: Pencerenin DIŞI — burada çıpa bozuluyor.
AL_DISI = ((24, 16.0), (32, 20.0), (32, 24.0), (40, 24.0))
#: Kontur yeri taraması (aynı fizik, kutunun yarı yüksekliği değişiyor).
KUTU_TARAMASI = (40, 60, 80, 120, 160)
#: Merdiven: `a` ve `λ` BİRLİKTE 1.5 kat ölçekleniyor, `a/λ = 8/3` SABİT.
MERDIVEN_R = 1.5
MERDIVEN = ((32, 12.0), (48, 18.0), (72, 27.0), (108, 40.5))


# ---------------------------------------------------------------------------
# Yardımcılar — GEOMETRİ BURADA, `src/`te DEĞİL
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class YarikEkrani:
    """PEC ekran (tek hücre kalınlığında) + yarık; dört kenarda PML.

    `CircularPEC` DIŞINDA bir `pec_mask_2d` geometrisi — ve bu, Task 4'te
    sevk edilen çekirdeğin "geometriyi bilmez" iddiasının **ilk bağımsız
    sınavıdır**: çekirdek değişmeden ikinci bir geometri takılıyor.
    """

    x: int
    y0: int
    y1: int
    npml: int = 20

    @property
    def thickness(self) -> int:
        return self.npml

    def sigma_2d(self, shape):
        return PML(thickness=self.npml).sigma_2d(shape)

    def pec_mask_2d(self, shape):
        m = np.zeros(tuple(int(n) for n in shape), dtype=bool)
        m[self.x, :] = True
        m[self.x, self.y0 : self.y1] = False
        return m

    def sigma_1d(self, cells):
        raise InvalidInputError("YarikEkrani yalniz 2B'de tanimlidir")


def _fazor(probes, *, wavelength):
    """Prob serilerinden hedef bindeki karmaşık genlik + binin GERÇEK frekansı."""
    F = np.fft.rfft(np.asarray(probes, dtype=float), axis=1)
    fr = np.fft.rfftfreq(np.asarray(probes).shape[1], d=S)
    j = int(np.argmin(np.abs(fr - 1.0 / wavelength)))
    return F[:, j], float(fr[j])


def _dikdortgen_kontur(x1, x2, y1, y2, *, kose_dusur=False):
    """Örnek noktaları YARIM tamsayıda; dört yüz konturu TAM örter.

    `kose_dusur=True` spec §7.5'in uyardığı kusurdur: her yüz `[y1, y2)` gibi
    yarım açık döşenir ve her köşede bir hücre düşer.
    """
    idx, nrm = [], []
    u = 1 if kose_dusur else 0
    for j in range(y1, y2 - u):
        idx.append(("x", x2, j, j + 1)); nrm.append((1.0, 0.0))
    for i in range(x1, x2 - u):
        idx.append(("y", y2, i, i + 1)); nrm.append((0.0, 1.0))
    for j in range(y1, y2 - u):
        idx.append(("x", x1, j, j + 1)); nrm.append((-1.0, 0.0))
    for i in range(x1, x2 - u):
        idx.append(("y", y1, i, i + 1)); nrm.append((0.0, -1.0))
    return idx, np.array(nrm, dtype=float)


def _kontur_noktalari(idx):
    return np.array(
        [(a, 0.5 * (b1 + b2)) if ax == "x" else (0.5 * (b1 + b2), a)
         for ax, a, b1, b2 in idx], dtype=float)


def _kontur_problari(idx):
    s = set()
    for ax, a0, b1, b2 in idx:
        for d in (-1, 0, 1):
            if ax == "x":
                s.add((a0 + d, b1)); s.add((a0 + d, b2))
            else:
                s.add((b1, a0 + d)); s.add((b2, a0 + d))
    return sorted(s)


def _kontur_alanlari(F, harita, idx, nrm):
    """Yüzde ORTALAMA `E`, DIŞA bakan normal yönünde MERKEZİ FARK `dE/dn`."""
    E, dn = [], []
    for (ax, a0, b1, b2), n in zip(idx, nrm):
        if ax == "x":
            def g(d, a0=a0, b1=b1, b2=b2):
                return 0.5 * (F[harita[(a0 + d, b1)]] + F[harita[(a0 + d, b2)]])
            s = n[0]
        else:
            def g(d, a0=a0, b1=b1, b2=b2):
                return 0.5 * (F[harita[(b1, a0 + d)]] + F[harita[(b2, a0 + d)]])
            s = n[1]
        E.append(g(0))
        dn.append(s * 0.5 * (g(1) - g(-1)))
    return np.array(E), np.array(dn)


def _duzlem_darbe(nx, ny, *, lam, sgf, x0, c, w, p=8):
    """+x'e giden Gauss darbeli, y'de DÜZ TEPELİ başlangıç alanı.

    `Hy(i+1/2)`nin argümanındaki `+0.75` sıçrama kaydırmasıdır: `h` zaman
    ekseninde yarım adım GERİDE (t = −dt/2 = −0.25) ve uzayda yarım hücre
    İLERİDE oturur, `E(x,t) = f(x − t)` için ikisi `f(i + 0.5 + 0.25)` eder.
    """
    def f(x):
        return gauss_cos(np.asarray(x, float), wavelength=lam, sgf=sgf, t0=x0)

    g = np.exp(-((np.abs(np.arange(ny) - c) / w) ** p))
    ex = np.arange(nx, dtype=float)
    return FieldState2D(
        f(ex)[:, None] * g[None, :],
        np.zeros((nx, ny - 1)),
        f(ex[:-1] + 0.75)[:, None] * g[None, :],
    )


def _yarik_kosumu(*, a, lam, sgf=1.0, npml=20, bosluk=24, kutu_x=40,
                  kutu_y=None, kuyruk=8.0):
    a = int(a)
    w = max(3 * a, 120)
    ky = int(kutu_y if kutu_y else max(2 * a, 80))
    ny = 2 * (max(w, ky + 8) + 40 + npml) + 1
    c = (ny - 1) // 2
    x0 = npml + int(round(4 * sgf * lam))
    xs = int(round(x0 + 4 * sgf * lam + bosluk))
    nx = xs + kutu_x + 12 + npml + 1
    steps = int(round(math.ceil((xs - x0 + kuyruk * sgf * lam + 6 * lam) / lam)
                      * lam / S))
    x1, x2, y1, y2 = xs + 2, xs + 2 + kutu_x, c - ky, c + ky
    pset = {(xs + d, j) for d in range(1, 5) for j in range(1, ny - 1)}
    pset |= set(_kontur_problari(_dikdortgen_kontur(x1, x2, y1, y2)[0]))
    probes = tuple(sorted(pset))
    rec = run_2d(
        Medium.uniform_2d(nx, ny), courant=S,
        boundary=YarikEkrani(x=xs, y0=c - a // 2, y1=c - a // 2 + a, npml=npml),
        steps=steps, source=None, source_index=None, probes=probes,
        wavelength=lam, sgf=sgf,
        state=_duzlem_darbe(nx, ny, lam=lam, sgf=sgf, x0=x0, c=c, w=w),
    )
    return rec, dict(nx=nx, ny=ny, c=c, xs=xs, steps=steps, a=a, lam=lam,
                     x1=x1, x2=x2, y1=y1, y2=y2,
                     harita={p: i for i, p in enumerate(probes)})


def _ilk_sifir(genlik_fn, *, n=4001, sin_max=0.95):
    """|F(θ)|'nın İLK yerel minimumu; log-parabol ile inceltilir."""
    th = np.linspace(1e-4, math.asin(sin_max), n)
    A = np.abs(genlik_fn(th))
    i = np.flatnonzero((A[1:-1] < A[:-2]) & (A[1:-1] < A[2:]))
    if i.size == 0:
        return None
    i = int(i[0]) + 1
    y0, y1, y2 = np.log(A[i - 1]), np.log(A[i]), np.log(A[i + 1])
    payda = y0 - 2.0 * y1 + y2
    d = 0.5 * (y0 - y2) / payda if payda != 0.0 else 0.0
    return float(th[i] + d * (th[1] - th[0]))


def _aciklik(g, F, *, ofset=2):
    jj = np.arange(1, g["ny"] - 1)
    return (np.array([F[g["harita"][(g["xs"] + ofset, j)]] for j in jj]),
            (jj - g["c"]).astype(float))


def _cipa(g, F, fb, *, ofset=2, n=4001, egiklik=True):
    """`a·sin(θ₁)/λ` — SEVK EDİLEN açıklık gözlemlenebiliriyle."""
    Fa, ys = _aciklik(g, F, ofset=ofset)
    lam = 1.0 / fb

    def genlik(th):
        out = aperture_far_field(Fa, wavelength=lam, bin_frequency=fb,
                                 angles=th, coordinates=ys)
        return out if egiklik else out / np.cos(th)

    th1 = _ilk_sifir(genlik, n=n)
    return None if th1 is None else g["a"] * math.sin(th1) / lam


def _kutu_cipasi(g, F, fb, *, kose_dusur=False, kutu_y=None, n=4001):
    """Aynı çıpa, ama KAPALI KUTU konturuyla — ve kusurlu köşe döşemesiyle."""
    y1 = g["c"] - kutu_y if kutu_y else g["y1"]
    y2 = g["c"] + kutu_y if kutu_y else g["y2"]
    idx, nrm = _dikdortgen_kontur(g["x1"], g["x2"], y1, y2,
                                  kose_dusur=kose_dusur)
    E, dn = _kontur_alanlari(F, g["harita"], idx, nrm)
    pts, lens = _kontur_noktalari(idx), np.ones(len(idx))
    lam = 1.0 / fb
    if kose_dusur:
        # SEVK EDİLEN kapı bunu REDDEDER (aşağıda ayrıca sınanıyor); kusurun
        # BEDELİNİ ölçebilmek için integral burada elle kuruluyor.
        k = 2.0 * math.pi / lam

        def genlik(th):
            rh = np.stack([np.cos(th), np.sin(th)], axis=1)
            return ((1j * k * (rh @ nrm.T) * E[None, :] - dn[None, :])
                    * np.exp(1j * k * (rh @ pts.T)) * lens[None, :]).sum(axis=1)
    else:
        def genlik(th):
            return ntff_far_field(points=pts, normals=nrm, lengths=lens,
                                  values=E, normal_derivatives=dn,
                                  wavelength=lam, bin_frequency=fb, angles=th)

    th1 = _ilk_sifir(genlik, n=n)
    return None if th1 is None else g["a"] * math.sin(th1) / lam


# ---------------------------------------------------------------------------
# Fixture'lar
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def yarik():
    """Bütün yarık koşumları — BİR KEZ koşulur, sekiz test tüketir.

    Anahtar `(a, λ, kutu_y)`. `kutu_y = None` varsayılan kutudur; kontur
    yeri taraması aynı `(a, λ)`yı farklı kutularla tekrar koşar (kutu
    konturunun problarını değiştirdiği için ayrı koşum gerekir — FDTD'nin
    KENDİSİ prob listesinden etkilenmez, ama kutu geometrisi prob kümesini
    belirler).
    """
    K = {}
    for a, lam in AL:
        K[(a, lam, None)] = _yarik_kosumu(a=a, lam=lam)
    for a, lam in AL_DISI:
        K[(a, lam, None)] = _yarik_kosumu(a=a, lam=lam)
    for a, lam in MERDIVEN:
        K.setdefault((a, lam, None), _yarik_kosumu(a=a, lam=lam))
    for ky in KUTU_TARAMASI:
        K[(40, 16.0, ky)] = _yarik_kosumu(a=40, lam=16.0, kutu_y=ky)
    return K


@pytest.fixture(scope="module")
def nokta_kaynak():
    """Vakumda merkezi nokta kaynak — İŞARET UZLAŞIMININ sınavı.

    Doğru cevap IZOTROPİKTİR ve bu bir modelleme varsayımı değil: 2B'de
    tek bir Ez kaynağı monopoldür. İki kontur yarıçapı AYNI koşumdan
    okunur (FDTD prob listesinden etkilenmez).
    """
    LAM, N, NPML, STEPS, SGF = 20.0, 241, 40, 2400, 4.0
    c = (N - 1) // 2
    konturlar = {}
    pset = set()
    for rc in (30, 45):
        idx, nrm = _dikdortgen_kontur(c - rc, c + rc, c - rc, c + rc)
        konturlar[rc] = (idx, nrm)
        pset |= set(_kontur_problari(idx))
    probes = tuple(sorted(pset))
    rec = run_2d(
        Medium.uniform_2d(N, N), courant=S, boundary=PML(thickness=NPML),
        steps=STEPS,
        source=lambda t: gauss_cos(t, wavelength=LAM, sgf=SGF, t0=5 * SGF * LAM),
        source_index=(c, c), probes=probes, wavelength=LAM, sgf=SGF)
    F, fb = _fazor(rec.probes, wavelength=LAM)
    return dict(F=F, fb=fb, lam=LAM,
                harita={p: i for i, p in enumerate(probes)},
                konturlar=konturlar)


ACI = np.linspace(0.0, 2.0 * math.pi, 121, endpoint=False)


def _anizotropi(F):
    a = np.abs(F)
    return float(a.std(ddof=1) / a.mean())


# ===========================================================================
# 1. İKİ UZAK ALAN — "iki yöntem" DEĞİL
# ===========================================================================
@pytest.mark.slow
def test_IKI_UZAK_ALAN_YONTEMI_AYNI_PROBLEMI_COZMUYOR_YARIK_TARAFI(yarik):
    """SINIF: ÖLÇÜM — iki gözlemlenebilirin AYRIMI, yarık tarafı.

    Yarıkta kapalı kontur **kurulabilir** ama doğru cevabı vermesi bir
    tesadüftür: ekran sonsuza uzandığı için ekran akımlarının bir kısmı
    her sonlu konturun dışında kalır. Bu dosyanın ölçtüğü ayrım **ortalama
    değil**, kutunun **konturun yerine bağlı** olmasıdır (bir sonraki test).

    n = 14, `a/λ ≥ 2`, aynı 14 kayıttan dört okuma:

        açıklık-FT (SEVK)     ort 0.998610   std 0.019357
        açıklık-FT (eğiksiz)  ort 0.997942   std 0.019400
        kapalı kutu           ort 0.996575   std 0.016353
        kutu, KÖŞE DÜŞÜK      ort 0.825877   std 0.302356

    **PLANIN "kapalı kutu dört kat kötü" SAYISI (std 0.065 / 0.077) BU
    PENCEREDE ÜRETİLEMEDİ**: sevk edilen köşe döşemesiyle kutunun yayılımı
    açıklık-FT'ninkinin ALTINDA (0.016 < 0.019). Planın sayısına yakın olan
    şey **köşesi düşürülmüş** kutudur (std 0.302, yani 19 kat kötü) —
    yani plandaki "kapalı kutu" büyük olasılıkla köşe döşemesi kusurlu bir
    konturdu. Ayrımın gerçek yeri ortalama/std değil, KONTUR BAĞIMLILIĞIDIR.
    """
    ft, kutu, acik = [], [], []
    for a, lam in AL:
        rec, g = yarik[(a, lam, None)]
        F, fb = _fazor(rec.probes, wavelength=lam)
        assert fb * lam == pytest.approx(1.0, abs=1e-12), (a, lam)
        ft.append(_cipa(g, F, fb))
        kutu.append(_kutu_cipasi(g, F, fb))
        acik.append(_kutu_cipasi(g, F, fb, kose_dusur=True))
    ft, kutu, acik = map(np.array, (ft, kutu, acik))
    assert ft.size == 14

    # Sevk edilen iki gözlemlenebilir AYNI hanede
    assert ft.mean() == pytest.approx(0.998610, abs=2e-4)
    assert ft.std(ddof=1) == pytest.approx(0.019357, rel=0.05)
    assert kutu.mean() == pytest.approx(0.996575, abs=2e-3)
    assert kutu.std(ddof=1) == pytest.approx(0.016353, rel=0.05)

    # KÖŞE DÜŞÜNCE kontur artık eşdeğerlik teoreminin konturu DEĞİL
    assert acik.std(ddof=1) == pytest.approx(0.302356, rel=0.05)
    assert acik.std(ddof=1) > 15.0 * kutu.std(ddof=1)
    assert acik.mean() == pytest.approx(0.825877, abs=5e-3)


@pytest.mark.slow
def test_KAPALI_KONTUR_YARIKTA_KONTURUN_YERINE_BAGLI(yarik):
    """SINIF: ÖLÇÜM — ayrımın GERÇEK yeri (yapısal, ortalama değil).

    Aynı fizik, aynı kayıt penceresi; değişen tek şey kutunun yarı
    yüksekliği. `a = 40`, `λ = 16`, kutu yarı yüksekliği 40/60/80/120/160:

        kapalı kutu : 0.984953 / 0.967527 / 0.971425 / 0.982944 / 0.983442
                      -> yayılım 1.743e-02
        açıklık-FT  : 0.964916 / 0.964916 / 0.964916 / 0.965335 / 0.965362
                      -> yayılım 4.460e-04

    yani **39 kat**. Açıklık-FT'nin küçük yayılımı bile kutunun yeriyle
    ilgili DEĞİL: açıklık satırı ilk üç kutuda birebir aynıdır (fark
    0.000e+00) ve yalnız `ky ≥ 120`de ızgaranın kendisi (`ny`) büyüdüğü
    için üçüncü hanede kımıldıyor.
    """
    kutu, ft = [], []
    for ky in KUTU_TARAMASI:
        rec, g = yarik[(40, 16.0, ky)]
        F, fb = _fazor(rec.probes, wavelength=16.0)
        kutu.append(_kutu_cipasi(g, F, fb))
        ft.append(_cipa(g, F, fb))
    kutu, ft = np.array(kutu), np.array(ft)
    assert np.ptp(kutu) == pytest.approx(1.743e-02, rel=0.10)
    assert np.ptp(ft) == pytest.approx(4.460e-04, rel=0.10)
    assert np.ptp(kutu) > 30.0 * np.ptp(ft)
    # ilk üç kutuda açıklık satırı BİREBİR aynı
    assert ft[0] == ft[1] == ft[2]


# ===========================================================================
# 2. TEK YARIK ÇIPASI
# ===========================================================================
@pytest.mark.slow
def test_TEK_YARIK_PENCERESI_ILE_ve_MERTEBE_HUKMU_OLMADAN(yarik):
    """SINIF: **FİZİK ÇIPASI**. `a·sin(θ₁)/λ = 1` (tam).

    ÖLÇÜLDÜ (n = 14, PENCERE `a/λ ≥ 2`):

        ort 0.998610 ± 0.019357,  aralık [0.964916, 1.029080]

    `a` 32–96 (**3.0 kat**) ve `λ` 12–24 (**2.0 kat**) AYRI AYRI değişti,
    çarpım 1'de kaldı. Bu, girilen parametrelerden okunamayan bir sayıdır:
    `a` ve `λ` girdidir, `sin θ₁` ÇÖZÜLÜR.

    HANE SAYISI **İKİ** (%1.9) — ve bu bir kusur beyanıdır. Sapmanın
    ölçülen bir parçası ızgara dispersiyonudur: `λ = 12`de sayısal
    dalgaboyu nominalden %0.86, `λ = 24`te %0.21 kısadır ve çıpa tam olarak
    `λ_sayısal/λ_nominal` okur.
    """
    v = []
    for a, lam in AL:
        rec, g = yarik[(a, lam, None)]
        F, fb = _fazor(rec.probes, wavelength=lam)
        v.append(_cipa(g, F, fb))
    v = np.array(v)
    assert v.mean() == pytest.approx(0.998610, abs=2e-4)
    assert v.std(ddof=1) == pytest.approx(0.019357, rel=0.05)
    assert v.min() == pytest.approx(0.964916, abs=1e-4)
    assert v.max() == pytest.approx(1.029080, abs=1e-4)
    # `a` ve `lambda` AYRI AYRI tarandı
    assert len({a for a, _ in AL}) == 6 and max(a for a, _ in AL) / min(
        a for a, _ in AL) == pytest.approx(3.0)
    assert len({l for _, l in AL}) == 4 and max(l for _, l in AL) / min(
        l for _, l in AL) == pytest.approx(2.0)


@pytest.mark.slow
def test_YARIK_CIPASI_PENCERESININ_DISINDA_BOZULUYOR(yarik):
    """SINIF: ÖLÇÜM — pencerenin ZORUNLU olduğu.

    `a/λ < 2`de ölçülen (a, λ, değer):

        (24, 16) a/λ=1.50 -> 0.903207
        (32, 20) a/λ=1.60 -> 0.924812
        (32, 24) a/λ=1.33 -> 0.828910
        (40, 24) a/λ=1.67 -> 0.960164

    Dördü de pencere içindeki en küçük değerin (0.964916) ALTINDA. Sebep
    fiziktir, sayısal değil: skaler Kirchhoff yaklaşımı `a ~ λ`da geçerli
    değildir ve PEC ekranın kenar akımları deseni kaydırır.
    """
    for (a, lam), beklenen in zip(
            AL_DISI, (0.903207, 0.924812, 0.828910, 0.960164)):
        rec, g = yarik[(a, lam, None)]
        F, fb = _fazor(rec.probes, wavelength=lam)
        v = _cipa(g, F, fb)
        assert a / lam < PENCERE_ORANI
        assert v == pytest.approx(beklenen, abs=1e-4), (a, lam)
        assert v < 0.964916


@pytest.mark.slow
def test_YARIK_MERDIVENI_YARGILANAMAZ_ve_SEBEBI_ISARET_DEGISIMI(yarik):
    """SINIF: ÖLÇÜT — çıpa MERTEBE HÜKMÜ OLMADAN sevk ediliyor.

    Merdiven `a` ile `λ`yı BİRLİKTE 1.5 kat ölçekler, `a/λ = 8/3` dört
    basamakta da SABİT: (32, 12) → (48, 18) → (72, 27) → (108, 40.5).
    `r = 1.5` bir tercih değil ÖLÇÜM: `λ/S` tam sayı olmak zorunda (FFT
    bini `1/λ`ya oturması için) ve `r = 1.25` dizisinde `λ = 18.75` bunu
    sağlamıyor — yuvarlanınca `a/λ` 2.667 / 2.667 / 2.632 / 2.681 olur,
    yani merdiven artık AYNI fiziği ölçmez.

    ÖLÇÜLEN (gürültü = işlem-sonrası pencere eksenlerinin bağıl yayılımı:
    açıklık yüzü ofseti 1/2/3 × açı ızgarası 2001/4001/8001):

        değer   0.972245 / 0.971090 / 0.979468 / 0.960488
        gürültü 7.80e-05 / 1.41e-03 / 1.40e-03 / 1.27e-04
        farklar −1.155e-03 / +8.378e-03 / −1.899e-02

    **HÜKÜM: YARGILANAMAZ**, ve sebebi tek satırda okunuyor: ardışık
    farklar İŞARET DEĞİŞTİRİYOR, dolayısıyla `convergence_ladder`ın imkân
    kapısı (aritmetiğin tanımlı olma koşulu) hiç geçilmiyor. Ardışık fark
    oranları 7.251 ve 2.265 — yakınsayan bir merdivende `r^{-p} < 1`
    beklenirken **ikisi de 1'in ÜSTÜNDE**, yani farklar küçülmüyor
    BÜYÜYOR. **Planın "ρ_son = 0.798" sayısı bu pencerede üretilemedi.**
    """
    deger, gurultu = [], []
    for a, lam in MERDIVEN:
        rec, g = yarik[(a, lam, None)]
        F, fb = _fazor(rec.probes, wavelength=lam)
        deger.append(_cipa(g, F, fb))
        havuz = [_cipa(g, F, fb, ofset=o, n=nn)
                 for o in (1, 2, 3) for nn in (2001, 4001, 8001)]
        gurultu.append(float(np.std(havuz, ddof=1) / abs(np.mean(havuz))))
    for (a, lam) in MERDIVEN:
        assert a / lam == pytest.approx(8.0 / 3.0)
        assert (lam / S) == int(lam / S)
    d = np.diff(deger)
    assert np.any(d[:-1] * d[1:] < 0.0), "işaret değişimi bekleniyordu"
    rho = [abs(d[i + 1] / d[i]) for i in range(len(d) - 1)]
    assert rho[0] == pytest.approx(7.251, rel=0.05)
    assert rho[1] == pytest.approx(2.265, rel=0.05)
    assert all(r > 1.0 for r in rho)
    lad = convergence_ladder(deger, [MERDIVEN_R ** -i for i in range(4)],
                             noise=gurultu, ratio=MERDIVEN_R)
    assert lad.verdict == "YARGILANAMAZ"
    assert lad.settled_order is None
    assert lad.p_inf is None


@pytest.mark.slow
def test_EGIKLIK_CARPANI_CIPAYI_OYNATMIYOR(yarik):
    """SINIF: ÖLÇÜM — RS-I'in `cos θ`sı DESENİN genliği içindir, çıpa için değil.

    Aynı 14 kayıtta `cos θ` çarpanı kaldırılınca `a·sin(θ₁)/λ`:

        ortalama 0.998610 -> 0.997942,  std 0.019357 -> 0.019400
        tek tek fark 5.26e-05 ... 2.94e-03 (ortalama 6.69e-04)

    Yani çarpan çıpanın hanesinin ALTINDA kalıyor. **DÜRÜST OLUMSUZ: bu
    çıpa eğiklik çarpanını SINAMAZ** — çarpan yanlış konsa (ör. RS-II'nin
    1'i ya da Kirchhoff'un (1+cos θ)/2'si) bu test görmezdi. Çarpanın
    doğruluğu burada ÖLÇÜLMÜYOR, BEYAN EDİLİYOR.
    """
    d = []
    for a, lam in AL:
        rec, g = yarik[(a, lam, None)]
        F, fb = _fazor(rec.probes, wavelength=lam)
        d.append(abs(_cipa(g, F, fb) - _cipa(g, F, fb, egiklik=False)))
    d = np.array(d)
    assert d.min() == pytest.approx(5.2551e-05, rel=0.1)
    assert d.max() == pytest.approx(2.9382e-03, rel=0.1)
    assert d.max() < 0.019357  # çıpanın kendi yayılımının altında


# ===========================================================================
# 3. ALETİN KAPILARI
# ===========================================================================
@pytest.mark.slow
def test_ISARET_UZLASIMI_OLCULDU_ters_isaret_deseni_taninmaz_yapiyor(
        nokta_kaynak):
    """SINIF: ÖLÇÜM — `ntff_far_field`in iki teriminin BAĞIL İŞARETİ.

    Vakumda merkezi bir Ez kaynağı monopoldür; doğru uzak alan
    IZOTROPİKTİR. Aynı kayıttan, `dE/dn`in işareti çevrilerek:

        kontur rc = 30 : doğru 0.003004   ters 0.264058   (87.9 kat)
        kontur rc = 45 : doğru 0.005507   ters 0.312722   (56.8 kat)

    (ölçü: |F(θ)|'nın std/ortalama'sı, 121 açı). Doğru işaretin bıraktığı
    %0.3–0.6 bir HATA DEĞİL, ızgaranın kendi açısal dispersiyonudur —
    `tests/em/test_sacilma.py`de aynı büyüklük TAM ÇÖZÜMLE karşılaştırılıyor
    ve orada tam çözümün kendi anizotropisiyle uyuşuyor.

    Bağıl işaret bu borunun uzlaşımından gelir: +x'e giden dalganın fazoru
    `e^{-ikx}` (Task 3'te arayüzsüz koşumla ölçüldü), dolayısıyla dışarı
    giden dalga `e^{-ikr}` ve çekirdek `exp(+i k r̂·r')`dir.
    """
    F, fb = nokta_kaynak["F"], nokta_kaynak["fb"]
    beklenen = {30: (0.003004, 0.264058), 45: (0.005507, 0.312722)}
    for rc, (dogru_b, ters_b) in beklenen.items():
        idx, nrm = nokta_kaynak["konturlar"][rc]
        E, dn = _kontur_alanlari(F, nokta_kaynak["harita"], idx, nrm)
        ortak = dict(points=_kontur_noktalari(idx), normals=nrm,
                     lengths=np.ones(len(idx)), values=E,
                     wavelength=1.0 / fb, bin_frequency=fb, angles=ACI)
        dogru = _anizotropi(ntff_far_field(normal_derivatives=dn, **ortak))
        ters = _anizotropi(ntff_far_field(normal_derivatives=-dn, **ortak))
        assert dogru == pytest.approx(dogru_b, rel=0.05), rc
        assert ters == pytest.approx(ters_b, rel=0.05), rc
        assert ters > 40.0 * dogru


def test_KONTUR_KAPANMA_KAPISI_uzaklasma_ozdesligi_bu_kusuru_GORMUYOR():
    """SINIF: ÖLÇÜM ALETİNİN KAPISI + neden UCUZ kapının yetmediği.

    Spec §7.5 "köşe döşemesi" uyarısı: dört yüzü `[y1, y2)` gibi yarım açık
    döşemek her köşede bir hücre düşürür. 20×20 bir kutuda ölçülen:

        örnek sayısı        80  ->  76
        SUM n_i dl_i (maks) 0.000000e+00  ->  0.000000e+00   (kapı KÖR)
        eşleşmeyen uç       0   ->  6                        (kapı GÖRÜYOR)

    Yani "kapalı konturda `∮ n̂ dl = 0`" özdeşliği bu kusuru **hiç
    görmüyor**, çünkü karşılıklı yüzler aynı miktarda kısalır ve katkılar
    birebir sadeleşir. Sevk edilen kapı UÇ EŞLEŞTİRMESİDİR.
    """
    tam_idx, tam_nrm = _dikdortgen_kontur(10, 30, 10, 30)
    kus_idx, kus_nrm = _dikdortgen_kontur(10, 30, 10, 30, kose_dusur=True)
    assert (len(tam_idx), len(kus_idx)) == (80, 76)
    for idx, nrm in ((tam_idx, tam_nrm), (kus_idx, kus_nrm)):
        dl = np.ones(len(idx))
        assert np.abs((nrm * dl[:, None]).sum(axis=0)).max() == 0.0
    assert _kapali_mi(_kontur_noktalari(tam_idx), tam_nrm, np.ones(80))
    assert not _kapali_mi(_kontur_noktalari(kus_idx), kus_nrm, np.ones(76))

    ortak = dict(wavelength=20.0, bin_frequency=0.05, angles=ACI)
    with pytest.raises(InvalidInputError, match="KAPALI DEGIL"):
        ntff_far_field(points=_kontur_noktalari(kus_idx), normals=kus_nrm,
                       lengths=np.ones(76), values=np.ones(76, dtype=complex),
                       normal_derivatives=np.zeros(76, dtype=complex), **ortak)
    # tam kontur GEÇİYOR
    out = ntff_far_field(points=_kontur_noktalari(tam_idx), normals=tam_nrm,
                         lengths=np.ones(80), values=np.ones(80, dtype=complex),
                         normal_derivatives=np.zeros(80, dtype=complex), **ortak)
    assert out.shape == ACI.shape


def test_FRAUNHOFER_KAPISI_YETERSIZ_BAGLAYICI_KOSUL_R_COK_BUYUK_LAMBDA():
    """SINIF: ÖLÇÜM ALETİNİN KAPISI — ve spec §4.4'ün DÜZELTİLMESİ.

    Spec §4.4 kapıyı bir zamanlar `R ≥ a²/λ` diye yazıyordu. ÖLÇÜLDÜ ve
    YETERSİZ. Bir sonraki test sayıları veriyor; burada kapının KENDİSİ
    sınanıyor:

    * `detector_radius = None` -> kapı KOŞMAZ (dönen şey zaten R → ∞
      asimptotik desendir),
    * `detector_radius < 100 λ` -> **`None`** (sessizce yanlış bir açı
      basmaz),
    * eski kapının geçirdiği bir R (`a²/λ`nın katları) yeni kapıda kalıyor.

    `a = 2` hücrelik bir saçıcıda `a²/λ` λ = 40'ta 0.10, λ = 100'de 0.04'tür;
    yani eski kapı `R/λ = 10`u **4000 ile 25000 kat** rahat geçirirdi — ve
    orada desen %14.5–37.4 yanlıştır (bir sonraki test).
    """
    alan = np.exp(-np.linspace(-2, 2, 41) ** 2).astype(complex)
    lam = 20.0
    ortak = dict(field=alan, wavelength=lam, bin_frequency=1.0 / lam,
                 angles=np.linspace(-1.0, 1.0, 21))
    assert FRAUNHOFER_LAMBDA == 100.0
    assert aperture_far_field(**ortak) is not None            # kapı koşmaz
    assert aperture_far_field(**ortak, detector_radius=None) is not None
    esik = FRAUNHOFER_LAMBDA * lam
    assert aperture_far_field(**ortak, detector_radius=esik) is not None
    assert aperture_far_field(**ortak, detector_radius=esik * (1 - 1e-9)) is None
    for R in (1.0, 10.0, 100.0, 1000.0):
        assert aperture_far_field(**ortak, detector_radius=R) is None
    # ESKİ KAPI: a = 2, lambda = 20 -> a^2/lambda = 0.2; R = 100 onu 500 kat
    # geçer ama YENİ kapıya takılır.
    a2lam = 2.0 ** 2 / lam
    assert 100.0 / a2lam == pytest.approx(500.0)
    assert aperture_far_field(**ortak, detector_radius=100.0) is None
    # NTFF'te de AYNI kapı, AYNI yerden koşar
    idx, nrm = _dikdortgen_kontur(0, 8, 0, 8)
    n = len(idx)
    kw = dict(points=_kontur_noktalari(idx), normals=nrm, lengths=np.ones(n),
              values=np.ones(n, dtype=complex),
              normal_derivatives=np.zeros(n, dtype=complex),
              wavelength=lam, bin_frequency=1.0 / lam, angles=ACI)
    assert ntff_far_field(**kw) is not None
    assert ntff_far_field(**kw, detector_radius=esik) is not None
    assert ntff_far_field(**kw, detector_radius=esik / 2) is None


@pytest.mark.slow
def test_FRAUNHOFER_HATASI_1_BOLU_R_GIDIYOR_ve_esikte_BEDELI_OLCULDU(yarik):
    """SINIF: ÖLÇÜM — kapının BEDELİ, tolerans değil.

    Aynı açıklık kaydından iki desen: (a) sevk edilen R → ∞ asimptotik
    desen, (b) aynı açıklığın TAM Green fonksiyonuyla sonlu R'de
    hesaplanmış deseni. Ölçü, tepesine normalize edilmiş |F(θ)| farkının
    RMS'i (a = 40, λ = 16, 121 açı, |θ| ≤ 1.2 rad):

        R/λ      10        30        100       300       1000
        RMS   1.539e-02 1.178e-02 4.286e-03 1.478e-03 4.464e-04

    Son üç basamakta hata **1/R** gidiyor (R on kat -> hata 2.9 ve 3.3 kat
    küçülüyor). Sevk edilen eşikte (`R = 100 λ`) artık hata **4.3e-03**tür
    ve bu eşiğin BEDELİDİR.

    `R = a²/λ` = 100 (yani `R/λ` = 6.25) eski kapının TAM SINIRIDIR ve
    orada hata **2.141e-02**tir — kapı "geçti" derken desen %2.1 yanlıştır.
    """
    rec, g = yarik[(40, 16.0, None)]
    F, fb = _fazor(rec.probes, wavelength=16.0)
    lam = 1.0 / fb
    Fa, ys = _aciklik(g, F)
    th = np.linspace(-1.2, 1.2, 121)
    inf = aperture_far_field(Fa, wavelength=lam, bin_frequency=fb, angles=th,
                             coordinates=ys)
    k = 2.0 * math.pi / lam

    def sonlu(R):
        rx = R * np.cos(th)[:, None]
        ry = R * np.sin(th)[:, None] - ys[None, :]
        d = np.hypot(rx, ry)
        return (Fa[None, :] * hankel2(1, k * d) * (rx / d)).sum(axis=1)

    def rms(A, B):
        A = np.abs(A) / np.abs(A).max()
        B = np.abs(B) / np.abs(B).max()
        return float(np.sqrt(((A - B) ** 2).mean()))

    olculen = {rl: rms(sonlu(rl * lam), inf)
               for rl in (10, 30, 100, 300, 1000)}
    beklenen = {10: 1.5388e-02, 30: 1.1782e-02, 100: 4.2859e-03,
                300: 1.4775e-03, 1000: 4.4642e-04}
    for rl, b in beklenen.items():
        assert olculen[rl] == pytest.approx(b, rel=0.05), rl
    # 1/R yasası
    assert olculen[100] / olculen[300] == pytest.approx(2.9, abs=0.3)
    assert olculen[300] / olculen[1000] == pytest.approx(3.3, abs=0.3)
    # ESKİ kapının tam sınırında (R = a^2/lambda = 100) hata BÜYÜK
    assert rms(sonlu(1600.0 / lam), inf) == pytest.approx(2.141e-02, rel=0.05)


def test_BIN_KAPISI_ETIKETLEME_KAPISIDIR_ve_uc_fonksiyonda_da_KOSAR():
    """SINIF: ÖLÇÜM ALETİNİN KAPISI — `lambda` etiketlemesi.

    Fazor bir BİNDE okunur; beyan edilen `wavelength` o binin üstüne
    oturmuyorsa dönen sayı **başka bir dalgaboyunun** sayısıdır. Kapı
    `|f_bin·λ − 1| ≤ 1e-9`dur (Task 3'ün `reflection_coefficient`indeki
    kapının aynısı) ve ihlalde `None` döner.

    Sayısal bedeli `tests/em/test_sacilma.py`de: nominal λ etiketiyle
    ölçülen Rayleigh üssü **+0.042543** kayıyor.
    """
    alan = np.ones(21, dtype=complex)
    ac = np.linspace(-1.0, 1.0, 11)
    assert BIN_GATE == 1e-9
    assert aperture_far_field(alan, wavelength=20.0, bin_frequency=0.05,
                              angles=ac) is not None
    assert aperture_far_field(alan, wavelength=20.0, bin_frequency=0.05 * 1.001,
                              angles=ac) is None
    idx, nrm = _dikdortgen_kontur(0, 8, 0, 8)
    n = len(idx)
    kw = dict(points=_kontur_noktalari(idx), normals=nrm, lengths=np.ones(n),
              values=np.ones(n, dtype=complex),
              normal_derivatives=np.zeros(n, dtype=complex),
              wavelength=20.0, angles=ACI)
    assert ntff_far_field(bin_frequency=0.05, **kw) is not None
    assert ntff_far_field(bin_frequency=0.0501, **kw) is None


def test_UZAK_ALAN_GIRDI_DENETIMLERI():
    """SINIF: ÇEVİRİ DENETİMİ — imkânsız girdiler SESSİZCE geçmiyor."""
    ac = np.linspace(-1.0, 1.0, 11)
    with pytest.raises(InvalidInputError, match="en az iki degerli"):
        aperture_far_field(np.ones(1, dtype=complex), wavelength=20.0,
                           bin_frequency=0.05, angles=ac)
    with pytest.raises(InvalidInputError, match="sonlu olmali"):
        aperture_far_field(np.array([1.0, np.nan], dtype=complex),
                           wavelength=20.0, bin_frequency=0.05, angles=ac)
    with pytest.raises(InvalidInputError, match="kesin ARTAN"):
        aperture_far_field(np.ones(3, dtype=complex), wavelength=20.0,
                           bin_frequency=0.05, angles=ac,
                           coordinates=np.array([0.0, 2.0, 1.0]))
    with pytest.raises(InvalidInputError, match="spacing"):
        aperture_far_field(np.ones(3, dtype=complex), wavelength=20.0,
                           bin_frequency=0.05, angles=ac, spacing=0.0)
    with pytest.raises(InvalidInputError, match="wavelength"):
        aperture_far_field(np.ones(3, dtype=complex), wavelength=-1.0,
                           bin_frequency=0.05, angles=ac)
    with pytest.raises(InvalidInputError, match="detector_radius"):
        aperture_far_field(np.ones(3, dtype=complex), wavelength=20.0,
                           bin_frequency=0.05, angles=ac, detector_radius=0.0)

    idx, nrm = _dikdortgen_kontur(0, 8, 0, 8)
    n = len(idx)
    tam = dict(points=_kontur_noktalari(idx), normals=nrm, lengths=np.ones(n),
               values=np.ones(n, dtype=complex),
               normal_derivatives=np.zeros(n, dtype=complex),
               wavelength=20.0, bin_frequency=0.05, angles=ACI)
    with pytest.raises(InvalidInputError, match="BIRIM vektor"):
        ntff_far_field(**{**tam, "normals": nrm * 2.0})
    with pytest.raises(InvalidInputError, match=r"\(M, 2\)"):
        ntff_far_field(**{**tam, "points": np.zeros((n, 3))})
    with pytest.raises(InvalidInputError, match="lengths"):
        ntff_far_field(**{**tam, "lengths": np.zeros(n)})
    with pytest.raises(InvalidInputError, match="en az dort"):
        ntff_far_field(**{**tam, "points": np.zeros((3, 2)),
                          "normals": np.array([[1.0, 0], [0, 1.0], [-1.0, 0]]),
                          "lengths": np.ones(3),
                          "values": np.ones(3, dtype=complex),
                          "normal_derivatives": np.zeros(3, dtype=complex)})


def test_YARIK_EKRANI_CIRCULARPEC_DISINDA_BIR_GEOMETRI():
    """SINIF: ÇEVİRİ DENETİMİ — çekirdek GEOMETRİYİ BİLMEZ, ikinci kanıt.

    Task 4'te `pec_mask_2d`in tek sevk edilmiş kullanıcısı `CircularPEC`ti
    ve "çekirdek geometriyi bilmez" iddiası TEK bir geometriyle
    sınanıyordu. `YarikEkrani` ikinci geometridir: maske bir DAİRE değil,
    tek hücre kalınlığında bir DOĞRU ve içinde bir boşluk. Çekirdekte tek
    satır değişmedi.
    """
    e = YarikEkrani(x=10, y0=4, y1=8, npml=3)
    m = e.pec_mask_2d((21, 15))
    assert m.shape == (21, 15) and m.dtype == bool
    assert m[10, :4].all() and m[10, 8:].all()
    assert not m[10, 4:8].any()
    assert not m[:10].any() and not m[11:].any()
    assert e.sigma_2d((21, 15)).shape == (21, 15)
    assert e.thickness == 3
    with pytest.raises(InvalidInputError):
        e.sigma_1d(21)


@pytest.mark.slow
def test_SLOW_ESIGI_OLCULDU(yarik, nokta_kaynak):
    """`slow` işareti bu dosyada FIXTURE'a bağlıdır ve eşik ÖLÇÜLDÜ.

    `yarik` fixture'ı 26 FDTD koşumu ve `nokta_kaynak` bir koşum yapar
    (ölçüldü: bu dosyanın tamamı **40.1 s**, `-m "not slow"` ile
    **0.5 s**); ikisini tüketen her test `slow`dur. Fixture maliyeti
    `pytest --durations` çıktısında `setup` olarak görünür ve koşum
    sırasına göre HERHANGİ BİRİNE yazılabilir (Modül 4'ün ölçtüğü tuzak),
    o yüzden eşik `setup + call`un en kötüsüne göre seçilmiştir.

    İşaretsiz kalanlar aletin kendi kapılarıdır (bin, Fraunhofer, kapanma,
    girdi denetimleri, geometri) — hepsi saf aritmetik, `-m "not slow"`
    koşumunda da koşarlar.
    """
    kaynak = open(__file__, encoding="utf-8").read()
    assert kaynak.count("@pytest.mark.slow\n") == 9
    for ad in ("IKI_UZAK_ALAN_YONTEMI_AYNI_PROBLEMI_COZMUYOR_YARIK_TARAFI",
               "KAPALI_KONTUR_YARIKTA_KONTURUN_YERINE_BAGLI",
               "TEK_YARIK_PENCERESI_ILE_ve_MERTEBE_HUKMU_OLMADAN",
               "YARIK_CIPASI_PENCERESININ_DISINDA_BOZULUYOR",
               "YARIK_MERDIVENI_YARGILANAMAZ_ve_SEBEBI_ISARET_DEGISIMI",
               "EGIKLIK_CARPANI_CIPAYI_OYNATMIYOR",
               "ISARET_UZLASIMI_OLCULDU_ters_isaret_deseni_taninmaz_yapiyor",
               "FRAUNHOFER_HATASI_1_BOLU_R_GIDIYOR_ve_esikte_BEDELI_OLCULDU"):
        assert f"@pytest.mark.slow\ndef test_{ad}" in kaynak, ad
    assert len(yarik) == 26
    assert set(nokta_kaynak["konturlar"]) == {30, 45}
    # Aletin kapıları FIXTURE'A DOKUNMUYOR: `-m "not slow"` koşumunda da
    # koşsunlar diye (Modül 4'ün `kucuk_kayit` ayrımıyla aynı gerekçe).
    for ad in ("KONTUR_KAPANMA_KAPISI", "FRAUNHOFER_KAPISI_YETERSIZ",
               "BIN_KAPISI_ETIKETLEME", "UZAK_ALAN_GIRDI_DENETIMLERI",
               "YARIK_EKRANI_CIRCULARPEC_DISINDA"):
        i = kaynak.index(f"def test_{ad}")
        govde = kaynak[i:kaynak.index("\ndef ", i + 1)] if "\ndef " in kaynak[i:] \
            else kaynak[i:]
        assert "yarik" not in govde.split("\n")[0], ad
