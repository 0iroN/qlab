"""Saçılma — **RAYLEIGH ÜSSÜ**, ve uzak alan ayrımının ÖBÜR yüzü.

`tests/em/test_uzak_alan.py` yarıkta ölçtü: kapalı kontur yapısal olarak
kurulamaz, cevabı konturun yerine bağlıdır. **BURADA TAM TERSİ ÖLÇÜLÜYOR:**
saçılmada doğru cevabı kapalı kontur verir, tek bir SATIRIN uzamsal FT'si ise
var olmayan bir açısal yapı **uydurur** (%21–47).

DOSYANIN SINIFLARI:

- **FİZİK ÇIPASI** — `W ~ λ^{−p}`, 2B'de `p = 3` / 3B'de `p = 4`. **BOYUT
  AYIRICI**: üs `a`'dan, `ε_r`'den ve `λ`'dan bağımsız olmalıdır ve bu
  dosyada ölçülen tam olarak bu bağımsızlığın **sınırıdır**.
- **ÖLÇÜM ALETİNİN KAPILARI** — `λ` etiketleme (bin) kapısı, `k` üssü,
  kontur yarıçapı bağımsızlığı.
- **ÇEVİRİ DENETİMİ** — TAM ÇÖZÜM (2B silindir serisi) ile karşılaştırma.

**`p = 3 ÖLÇÜLDÜ` YAZILAMAZ, ve σ CİNSİNDEN MARJ DA YAZILAMAZ.**
İkisinin de sebebi bu dosyada ölçülüdür (`test_SIGMA_FIT_BIR_BELIRSIZLIK_
DEGIL_PENCERENIN_EGRILIGIDIR`). Yazılabilecek olan **niteldir**:

    "2B saçılma üssü bu pencerede (ka = 0.094–0.398) ölçüldü ve AYNI
     PENCEREDEKİ TAM ÇÖZÜMLE uyuşuyor; 3B'nin üssü 4 geniş bir marjla
     dışlanıyor. `p = 3` ASİMPTOTİK bir değerdir ve bu pencere asimptotik
     DEĞİLDİR."

TAM ÇÖZÜM NASIL SERTİFİKA EDİLDİ — ve nasıl EDİLMEDİ: **optik teorem kapısı
bir TAUTOLOJİDİR** ve bu dosyada sayıyla gösteriliyor (Bessel yerine rastgele
reel fonksiyon konunca da 1e-16 bandında sağlanıyor). Serinin iki gerçek
sertifikası var: (a) `ka → 0` limitinde üssü **3.000002**'ye gidiyor
(bağımsız, bilinen asimptotik sonuç), (b) beş ayrı pencerede **bağımsız FDTD
ölçümüyle** 0.0185 içinde uyuşuyor.

PENCERE (ana koşum): `dx = 1`, `S = 0.5`, `N = 341`, `npml = 60`, saçıcı
yarıçap `a = 2` hücre `ε_r = 4`, kontur `rc = 40` (ayrıca 30 ve 50),
**kaynak terimi YOK** — ızgara +x'e giden geniş bantlı bir Ricker düzlem
dalgasıyla BAŞLATILIYOR (`λ_p = 40`, `x₀ = npml + 1.2 λ_p`), `T = 1200`
(2400 adım). Saçılan alan **saçıcılı ve saçıcısız iki koşumun FARKIDIR**
(Task 3'ün `reflection_coefficient`indeki aynı desen). `λ` değerleri
**BİN IZGARASINDAN** seçilir (`λ = T/j`), böylece etiketleme kapısı tam
geçer.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import h1vp, hankel1, hankel2, jv, jvp

from qlab.engine.em.boundary import PML
from qlab.engine.em.grid import FieldRecord, FieldState2D
from qlab.engine.em.medium import Medium
from qlab.engine.em.source import gauss_cos, ricker
from qlab.engine.em.yee import run_2d
from qlab.errors import InvalidInputError
from qlab.observable.em_obs import (
    convergence_ladder,
    mode_spectrum,
    ntff_far_field,
    scattering_width,
)

S = 0.5
ACI = np.linspace(0.0, 2.0 * math.pi, 121, endpoint=False)

#: Ana koşumun penceresi.
N_ANA, A_ANA, EPS_ANA, NPML_ANA, LAMP_ANA, T_ANA = 341, 2.0, 4.0, 60, 40.0, 1200.0
#: `lambda = T/j` — BİN IZGARASINDAN seçilen dalgaboyları.
JL = (9, 12, 15, 19, 24, 30, 38)
#: Fraunhofer tablosunun okunduğu binler (λ = 100 / 60 / 40).
JL_FRAUNHOFER = (12, 20, 30)
#: NOMİNAL (bin ızgarasına oturmayan) dalgaboyu kümesi — etiketleme sınavı.
NOMINAL = (128.0, 100.0, 80.0, 64.0, 50.0, 40.0, 32.0)

#: Merdiven: HER ŞEY `s` ile ölçekleniyor (a, λ, ızgara, npml, kayıt).
MERDIVEN_R = 1.25
MERDIVEN_JL = (16, 20, 25, 30, 35, 40)
MERDIVEN_TABAN = dict(N0=181, npml0=36, rc0=22, lamp0=22.5, T0=720.0, a0=1.08)


# ---------------------------------------------------------------------------
# TAM ÇÖZÜM — 2B TM_z dielektrik silindir serisi
# ---------------------------------------------------------------------------
def _mie_c(x, m):
    """Saçılma katsayıları `c_n`. OPTİK TEOREM KAPISI KULLANILMAZ (tautoloji)."""
    nmax = int(max(6, math.ceil(x + 6 * x ** (1 / 3) + 6)))
    n = np.arange(-nmax, nmax + 1)
    Jm, Jmp = jv(n, m * x), jvp(n, m * x)
    J, Jp = jv(n, x), jvp(n, x)
    H, Hp = hankel1(n, x), h1vp(n, x)
    return (m * Jmp * J - Jm * Jp) / (m * Jmp * H - Jm * Hp)


def _mie_W(x, m, a):
    """Toplam saçılma genişliği `W = (4/k) SUM |c_n|^2` (uzunluk)."""
    return float(4.0 * np.sum(np.abs(_mie_c(x, m)) ** 2) / x * a)


def _mie_desen(x, m, th):
    nmax = (len(_mie_c(x, m)) - 1) // 2
    return np.abs(np.exp(1j * np.outer(th, np.arange(-nmax, nmax + 1)))
                  @ _mie_c(x, m))


def _us(lam, W):
    """`W ~ lambda^{-p}` -> `p` (log-log en küçük kareler)."""
    return float(-np.polyfit(np.log(lam), np.log(W), 1)[0])


def _fit(lam, W):
    x, y = np.log(np.asarray(lam)), np.log(np.asarray(W))
    k, b = np.polyfit(x, y, 1)
    art = y - (k * x + b)
    sig = math.sqrt(np.sum(art ** 2) / (len(x) - 2) / np.sum((x - x.mean()) ** 2))
    return -float(k), float(sig), art


# ---------------------------------------------------------------------------
# Geometri ve koşum — GEOMETRİ BURADA, `src/`te DEĞİL
# ---------------------------------------------------------------------------
def _dikdortgen_kontur(x1, x2, y1, y2):
    idx, nrm = [], []
    for j in range(y1, y2):
        idx.append(("x", x2, j, j + 1)); nrm.append((1.0, 0.0))
    for i in range(x1, x2):
        idx.append(("y", y2, i, i + 1)); nrm.append((0.0, 1.0))
    for j in range(y1, y2):
        idx.append(("x", x1, j, j + 1)); nrm.append((-1.0, 0.0))
    for i in range(x1, x2):
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


def _silindir(N, c, a, eps_r):
    i = np.arange(N)[:, None] - c
    j = np.arange(N)[None, :] - c
    eps = np.ones((N, N))
    eps[(i * i + j * j) <= a * a] = eps_r
    return Medium(eps, np.ones((N, N)))


def _duzlem_ricker(N, *, lam_p, x0):
    """+x'e giden GENİŞ BANTLI düzlem dalga başlangıç alanı.

    Geniş bant bilinçlidir: tek koşumda BÜTÜN `λ`lar okunur (`boundary.PML`
    ölçümündeki aynı gerekçe). Yedi `λ` için yedi koşum yapmak aynı sayıyı
    yedi kat pahalıya alırdı.
    """
    def f(x):
        return ricker(np.asarray(x, float), wavelength=lam_p, sgf=x0 / lam_p)

    ex = np.arange(N, dtype=float)
    return FieldState2D(np.repeat(f(ex)[:, None], N, axis=1),
                        np.zeros((N, N - 1)),
                        np.repeat(f(ex[:-1] + 0.75)[:, None], N, axis=1))


def _sacilma_kosumu(*, N, a, eps_r, rc, npml, lam_p, steps, satir_yari=None,
                    binler=()):
    """SAÇICILI ve SAÇICISIZ iki koşum; saçılan alan ikisinin FARKIDIR.

    `x0 = npml + 1.2 λ_p` BİR KOŞULDUR, kolaylık değil: başlangıç darbesi
    saçıcıyla ÖRTÜŞMEMELİ. Örtüşünce ölçüm bozuluyor — ölçüldü, `W/W_tam`
    oranı λ boyunca 1.01'den 0.24'e düşüyor ve üs 3.24 yerine 2.21 çıkıyor.
    """
    c = (N - 1) // 2
    rcler = (int(rc),) if np.isscalar(rc) else tuple(int(r) for r in rc)
    konturlar, pset = [], {(c, c)}
    for r in rcler:
        idx, nrm = _dikdortgen_kontur(c - r, c + r, c - r, c + r)
        pset |= set(_kontur_problari(idx))
        konturlar.append(dict(rc=r, idx=idx, nrm=nrm,
                              pts=_kontur_noktalari(idx),
                              lens=np.ones(len(idx))))
    if satir_yari:
        pset |= {(c + rcler[0], j) for j in range(c - satir_yari, c + satir_yari)}
    probes = tuple(sorted(pset))
    st = _duzlem_ricker(N, lam_p=lam_p, x0=npml + 1.2 * lam_p)
    kayitlar = [
        run_2d(_silindir(N, c, a, eps), courant=S, boundary=PML(thickness=npml),
               steps=steps, source=None, source_index=None, probes=probes,
               wavelength=lam_p, sgf=1.0, state=st)
        for eps in (eps_r, 1.0)
    ]
    fr = np.fft.rfftfreq(steps, d=S)
    FS = np.fft.rfft(kayitlar[0].probes - kayitlar[1].probes, axis=1)
    FI = np.fft.rfft(kayitlar[1].probes, axis=1)
    b = np.asarray(sorted(set(binler)), dtype=int) if len(binler) else \
        np.arange(fr.size)
    return dict(N=N, c=c, a=a, eps_r=eps_r, steps=steps, npml=npml,
                konturlar=konturlar, satir_yari=satir_yari, rc=rcler[0],
                harita={p: i for i, p in enumerate(probes)},
                F={int(j): FS[:, j] for j in b},
                FI={int(j): FI[:, j] for j in b}, fr=fr,
                ham=kayitlar[0], ham_ref=kayitlar[1])


def _oku(g, jler, *, kontur=0, acilar=ACI, k_ussu=True, etiket=None):
    """Bin listesinden `(lambda, W)`; `etiket` verilirse NOMİNAL λ ile etiketler."""
    K = g["konturlar"][kontur]
    lam, W = [], []
    for t, j in enumerate(jler):
        L = 1.0 / g["fr"][j]
        E, dn = _kontur_alanlari(g["F"][j], g["harita"], K["idx"], K["nrm"])
        I = ntff_far_field(points=K["pts"], normals=K["nrm"], lengths=K["lens"],
                           values=E, normal_derivatives=dn, wavelength=L,
                           bin_frequency=g["fr"][j], angles=acilar)
        w = scattering_width(I, angles=acilar, wavelength=L,
                             bin_frequency=g["fr"][j],
                             incident_amplitude=g["FI"][j][
                                 g["harita"][(g["c"], g["c"])]])
        if not k_ussu:
            w = w * (2.0 * math.pi / L)     # `k` çarpanı DÜŞÜRÜLÜR (mutant)
        lam.append(etiket[t] if etiket is not None else L)
        W.append(w)
    return np.array(lam), np.array(W)


def _desen(g, j, *, kontur=0):
    K = g["konturlar"][kontur]
    L = 1.0 / g["fr"][j]
    E, dn = _kontur_alanlari(g["F"][j], g["harita"], K["idx"], K["nrm"])
    return np.abs(ntff_far_field(
        points=K["pts"], normals=K["nrm"], lengths=K["lens"], values=E,
        normal_derivatives=dn, wavelength=L, bin_frequency=g["fr"][j],
        angles=ACI))


def _seviye(s):
    """Merdivenin bir basamağı: HER ŞEY `s` ile ölçekleniyor."""
    t = MERDIVEN_TABAN
    N = 2 * int(round(t["N0"] * s / 2)) + 1
    return _sacilma_kosumu(N=N, a=t["a0"] * s, eps_r=EPS_ANA,
                           rc=int(round(t["rc0"] * s)),
                           npml=int(round(t["npml0"] * s)),
                           lam_p=t["lamp0"] * s,
                           steps=int(round(t["T0"] * s / S)))


# ---------------------------------------------------------------------------
# Fixture'lar
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ana():
    """Ana koşum — İKİ FDTD koşumu, yedi test tüketir (ölçüldü: 4.4 s)."""
    binler = list(JL) + list(JL_FRAUNHOFER) + [int(np.argmin(np.abs(
        np.fft.rfftfreq(int(T_ANA / S), d=S) - 1.0 / L))) for L in NOMINAL]
    return _sacilma_kosumu(N=N_ANA, a=A_ANA, eps_r=EPS_ANA, rc=(40, 30, 50),
                           npml=NPML_ANA, lam_p=LAMP_ANA,
                           steps=int(T_ANA / S), satir_yari=40, binler=binler)


@pytest.fixture(scope="module")
def merdiven():
    """Dört basamak (sekiz FDTD koşumu, ölçüldü: 10.0 s)."""
    return [_seviye(MERDIVEN_R ** i) for i in range(4)]


@pytest.fixture(scope="module")
def pencereler():
    """Farklı `a` ve `eps_r` pencereleri — üssün BAĞIMSIZLIK SINIRI."""
    return {
        (2.0, 2.25): _sacilma_kosumu(N=341, a=2.0, eps_r=2.25, rc=40, npml=60,
                                     lam_p=40.0, steps=2400, binler=JL),
        (3.0, 4.0): _sacilma_kosumu(N=341, a=3.0, eps_r=4.0, rc=40, npml=60,
                                    lam_p=40.0, steps=2400, binler=JL),
    }


@pytest.fixture(scope="module")
def sogurucu():
    """2B soğurucunun AÇI TARAMASI — Task 4'ün dürüst olumsuzu kapanıyor.

    Vakumda merkezi nokta kaynak; uzak alan deseni sevk edilen
    `ntff_far_field` ile okunur. REFERANS, soğurucusu konturdan o kadar
    uzakta ki yansıması kayıt penceresine YETİŞMİYOR — ve bu bir varsayım
    değil, ölçüm: iki bağımsız ızgara (N = 541 ve 641) deseni **3.07e-15**
    içinde aynı veriyor.
    """
    LAM, SGF, RC = 16.0, 3.0, 30
    steps = int(round(24 * LAM / S))

    def desen(N, npml):
        c = (N - 1) // 2
        idx, nrm = _dikdortgen_kontur(c - RC, c + RC, c - RC, c + RC)
        probes = tuple(_kontur_problari(idx))
        rec = run_2d(Medium.uniform_2d(N, N), courant=S,
                     boundary=PML(thickness=npml), steps=steps,
                     source=lambda t: gauss_cos(t, wavelength=LAM, sgf=SGF,
                                                t0=5 * SGF * LAM),
                     source_index=(c, c), probes=probes, wavelength=LAM,
                     sgf=SGF)
        F = np.fft.rfft(rec.probes, axis=1)
        fr = np.fft.rfftfreq(steps, d=S)
        j = int(np.argmin(np.abs(fr - 1.0 / LAM)))
        h = {p: i for i, p in enumerate(probes)}
        E, dn = _kontur_alanlari(F[:, j], h, idx, nrm)
        return np.abs(ntff_far_field(
            points=_kontur_noktalari(idx), normals=nrm,
            lengths=np.ones(len(idx)), values=E, normal_derivatives=dn,
            wavelength=1.0 / fr[j], bin_frequency=fr[j], angles=ACI))

    ref = {N: desen(N, 60) for N in (541, 641)}
    kucuk = {d: desen(2 * (RC + d + 12) + 1, d) for d in (8, 15, 25, 40)}
    return dict(ref=ref, kucuk=kucuk, lam=LAM, rc=RC, steps=steps)


# ===========================================================================
# 1. UZAK ALAN AYRIMININ ÖBÜR YÜZÜ
# ===========================================================================
@pytest.mark.slow
def test_IKI_UZAK_ALAN_YONTEMI_AYNI_PROBLEMI_COZMUYOR_SACILMA_TARAFI(ana):
    """SINIF: ÖLÇÜM — ayrımın saçılma tarafı, ve **TAM ÇÖZÜMLE tartılıyor**.

    Küçük bir TM_z dielektrik silindirin uzak alanı neredeyse izotropiktir.
    Aynı kayıttan iki okuma, ve ikisi de TAM ÇÖZÜMLE karşılaştırılıyor
    (normalize |F(θ)|'nın std/ort'u ve tam çözümden RMS farkı):

        ka        0.094   0.126   0.157   0.199   0.251   0.314   0.398
        NTFF      0.0011  0.0020  0.0048  0.0066  0.0095  0.0157  0.0231
        TAM       0.0015  0.0027  0.0041  0.0064  0.0098  0.0146  0.0222
        SATIR-FT  0.2130  0.3762  0.4697  0.4164  0.3742  0.4602  0.4042

        RMS(NTFF−TAM)  0.0004 ... 0.0025
        RMS(SATIR−TAM) 0.2121 ... 0.4678

    **NTFF'in kalan anizotropisi bir HATA DEĞİL, büyük ölçüde GERÇEK**:
    tam çözümün kendi anizotropisiyle aynı hanede ve aynı yönde gidiyor.
    Tek satırın FT'si ise 100–500 kat büyük, tamamen UYDURMA bir açısal
    yapı veriyor.
    """
    jj = np.arange(ana["c"] - ana["satir_yari"], ana["c"] + ana["satir_yari"])
    ys = (jj - ana["c"]).astype(float)
    ntff_rms, satir_rms = [], []
    for j in JL:
        L = 1.0 / ana["fr"][j]
        D = _desen(ana, j)
        D = D / D.mean()
        M = _mie_desen(2 * math.pi * ana["a"] / L, math.sqrt(ana["eps_r"]), ACI)
        M = M / M.mean()
        satir = np.array([ana["F"][j][ana["harita"][(ana["c"] + ana["rc"], q)]]
                          for q in jj])
        k = 2 * math.pi / L
        B = np.abs((satir[None, :]
                    * np.exp(1j * k * np.sin(ACI)[:, None] * ys[None, :])
                    ).sum(axis=1))
        B = B / B.mean()
        ntff_rms.append(float(np.sqrt(((D - M) ** 2).mean())))
        satir_rms.append(float(np.sqrt(((B - M) ** 2).mean())))
        # NTFF'in anizotropisi TAM ÇÖZÜMÜNKİYLE aynı hanede
        assert D.std(ddof=1) == pytest.approx(M.std(ddof=1), rel=0.35), L
        assert B.std(ddof=1) > 15.0 * D.std(ddof=1), L
    ntff_rms, satir_rms = np.array(ntff_rms), np.array(satir_rms)
    assert ntff_rms.max() < 3.0e-3
    assert satir_rms.min() > 0.21 and satir_rms.max() < 0.47
    assert (satir_rms / ntff_rms).min() > 100.0


@pytest.mark.slow
def test_KONTUR_YARICAPI_SACILMADA_SERBEST(ana):
    """SINIF: ÖLÇÜM — yarıkta konturun yeri belirleyiciydi, burada DEĞİL.

    Aynı koşumdan üç kontur (rc = 30 / 40 / 50):

        p = 3.242092 / 3.242894 / 3.241718     -> yayılım 1.175e-03
        `W`nin bağıl yayılımı (yedi λ, üç kontur) 3.682e-03

    Karşılaştır: yarıkta aynı türden bir tarama `a·sin θ₁/λ`yı **1.743e-02**
    oynatıyordu. Fark yapısal: burada bütün kaynaklar (saçıcı) her konturun
    İÇİNDE, yarıkta ekranın akımları her konturun DIŞINA taşıyor.
    """
    p, Wler = {}, {}
    for i, K in enumerate(ana["konturlar"]):
        lam, W = _oku(ana, JL, kontur=i)
        p[K["rc"]] = _us(lam, W)
        Wler[K["rc"]] = W
    assert p[40] == pytest.approx(3.242894, abs=1e-4)
    assert p[30] == pytest.approx(3.242092, abs=1e-4)
    assert p[50] == pytest.approx(3.241718, abs=1e-4)
    assert max(p.values()) - min(p.values()) == pytest.approx(1.175e-03, rel=0.1)
    yayilim = max(float(np.max(np.abs(Wler[r] / Wler[40] - 1.0)))
                  for r in Wler)
    assert yayilim == pytest.approx(3.682e-03, rel=0.1)


# ===========================================================================
# 2. RAYLEIGH ÜSSÜ
# ===========================================================================
@pytest.mark.slow
def test_RAYLEIGH_USSU_3B_YI_DISLIYOR_ama_SIGMA_CINSINDEN_MARJ_YOK(ana):
    """SINIF: **FİZİK ÇIPASI, BOYUT AYIRICI**.

    ÖLÇÜLDÜ (pencere ka = 0.0942–0.3979, yedi λ bin ızgarasından):

        p_FDTD = 3.242894      p_TAM = 3.238361      fark = +0.004533

    ve `W/W_tam` yedi λ'da 1.0733 … 1.0834 (yani ölçek de tutuyor; oranın
    dördüncü kökü `a_eff = 2.038` verir — merdivenlenmiş dairenin etkin
    yarıçapı, nominal 2'nin %1.9 üstünde).

    **`p = 3 ÖLÇÜLDÜ` YAZILAMAZ**, çünkü tam çözüm de bu pencerede 3
    vermiyor. `p = 3` ASİMPTOTİKTİR ve bu pencere asimptotik DEĞİLDİR —
    tam çözümün üssü ka penceresiyle:

        ka = 0.1 – 0.4     -> 3.246022
        ka = 0.01 – 0.04   -> 3.010256
        ka = 0.001 – 0.004 -> 3.000172
        ka = 1e-4 – 4e-4   -> 3.000002

    3B'NİN 4'Ü DIŞLANIYOR ve bu **σ cinsinden değil ARALIK olarak**
    yazılıyor: bu dosyada beş ayrı pencerede ölçülen `p` [2.924895,
    3.243596] aralığındadır ve 4 bu aralığın dışındadır — üstelik aralığın
    kendi genişliği (0.319) 4'e olan uzaklıktan (0.756) küçüktür.
    """
    lam, W = _oku(ana, JL)
    ka = 2 * math.pi * ana["a"] / lam
    Wm = np.array([_mie_W(x, math.sqrt(ana["eps_r"]), ana["a"]) for x in ka])
    assert ka.min() == pytest.approx(0.0942, abs=1e-4)
    assert ka.max() == pytest.approx(0.3979, abs=1e-4)
    p_f, p_m = _us(lam, W), _us(lam, Wm)
    assert p_f == pytest.approx(3.242894, abs=1e-4)
    assert p_m == pytest.approx(3.238361, abs=1e-6)
    assert abs(p_f - p_m) == pytest.approx(0.004533, abs=5e-4)
    assert np.all(np.abs(W / Wm - 1.078) < 0.01)
    assert float(np.mean((W / Wm) ** 0.25) * ana["a"]) == pytest.approx(
        2.0380, abs=1e-3)
    # `p = 3` ASİMPTOTİK
    for (lo, hi), beklenen in ((( 0.1, 0.4), 3.246022),
                               ((0.01, 0.04), 3.010256),
                               ((0.001, 0.004), 3.000172),
                               ((1e-4, 4e-4), 3.000002)):
        x = np.exp(np.linspace(math.log(lo), math.log(hi), 9))
        pp = _us(2 * math.pi * ana["a"] / x,
                 [_mie_W(xi, math.sqrt(ana["eps_r"]), ana["a"]) for xi in x])
        assert pp == pytest.approx(beklenen, abs=1e-5), (lo, hi)


@pytest.mark.slow
def test_SIGMA_FIT_BIR_BELIRSIZLIK_DEGIL_PENCERENIN_EGRILIGIDIR(ana):
    """SINIF: ÖLÇÜM — **σ cinsinden marjın neden YAZILAMAYACAĞI**.

    Aynı λ ızgarasında iki fit: FDTD'nin ve **analitik** tam çözümün.
    Analitik eğride ızgara yok, adım yok, gürültü yok:

        FDTD  p = 3.242894   sigma_fit = 0.016080
        TAM   p = 3.238361   sigma_fit = 0.015423      oran 1.0426

    yani fit artığının **%96'sı** gürültüsüz bir eğriden geliyor. Artıklar
    da birbirinin aynısı (ikisinde de üç işaret değişimi, düzgün bir eğri).
    `sigma_fit` bir ölçüm belirsizliği DEĞİL, pencerenin EĞRİLİĞİDİR.

    Dolayısıyla şu ifadelerin **hiçbiri yazılamaz** (hepsi bu σ ile
    kuruluyor ve hiçbiri çözücü hakkında bir şey söylemiyor):

        "3'ten 15.1 sigma"   "4'ten 47.1 sigma"   "tam çözümle 0.28 sigma"

    çünkü **TAM ÇÖZÜM DE** aynı ızgarada 3'ten **15.5 sigma** uzakta.
    """
    lam, W = _oku(ana, JL)
    Wm = np.array([_mie_W(2 * math.pi * ana["a"] / L,
                          math.sqrt(ana["eps_r"]), ana["a"]) for L in lam])
    p_f, s_f, art_f = _fit(lam, W)
    p_m, s_m, art_m = _fit(lam, Wm)
    assert s_f == pytest.approx(0.016080, rel=0.02)
    assert s_m == pytest.approx(0.015423, rel=0.02)
    assert s_f / s_m == pytest.approx(1.0426, rel=0.03)
    assert s_m / s_f > 0.95            # gürültüsüz payı %95'in ÜSTÜNDE
    # artıklar aynı eğri: işaret dizisi birebir aynı
    assert np.array_equal(np.sign(art_f), np.sign(art_m))
    assert int(np.sum(np.diff(np.sign(art_m)) != 0)) == 3
    # SAHTE marjlar: tam çözüm de 3'ten aynı kadar "uzak"
    assert abs(p_f - 3.0) / s_f == pytest.approx(15.1, abs=0.3)
    assert abs(p_m - 3.0) / s_m == pytest.approx(15.5, abs=0.3)
    assert abs(p_f - 4.0) / s_f == pytest.approx(47.1, abs=0.5)


@pytest.mark.slow
def test_USSUN_BAGIMSIZLIK_SINIRI_OLCULDU(ana, pencereler):
    """SINIF: ÖLÇÜM — üs `a`'dan ve `ε_r`'den TAM bağımsız DEĞİL, ve neden.

    Evrensellik (`p = 3`, girdilerden bağımsız) **asimptotik** bir
    ifadedir. Bu pencerede ölçülen:

        a = 2, eps_r = 4.00 : ka 0.094–0.398  p = 3.242894  (tam 3.238361)
        a = 2, eps_r = 2.25 : ka 0.094–0.398  p = 3.093334  (tam 3.090754)
        a = 3, eps_r = 4.00 : ka 0.141–0.597  p = 2.924895  (tam 2.943387)

    Üç pencerede `p` **0.318 oynuyor** — ama üçünde de FDTD, aynı
    penceredeki TAM ÇÖZÜMLE **0.0185 içinde** uyuşuyor. Yani oynayan şey
    çözücü değil PENCEREDİR; `ε_r` ve `a` bağımlılığı sonlu `ka` fiziğidir
    ve tam çözüm onu birebir üretiyor.

    Bu, çıpayı ZAYIFLATMIYOR, **PENCERESİNİ ADLANDIRIYOR** — spec §7.4'ün
    "parametre bağımsızlığı gerekli ama yeterli değil" dersinin karşı yüzü.
    """
    kayitlar = [(ana["a"], ana["eps_r"], ana)] + [
        (k[0], k[1], v) for k, v in pencereler.items()]
    ps, farklar = [], []
    for a, eps, g in kayitlar:
        lam, W = _oku(g, JL)
        Wm = np.array([_mie_W(2 * math.pi * a / L, math.sqrt(eps), a)
                       for L in lam])
        ps.append(_us(lam, W))
        farklar.append(abs(_us(lam, W) - _us(lam, Wm)))
    assert ps[0] == pytest.approx(3.242894, abs=1e-4)
    assert ps[1] == pytest.approx(3.093334, abs=1e-4)
    assert ps[2] == pytest.approx(2.924895, abs=1e-4)
    assert max(ps) - min(ps) == pytest.approx(0.318, abs=0.01)
    assert max(farklar) == pytest.approx(0.018491, abs=1e-3)
    # 4 HER pencerede dışarıda, ve marj aralığın kendi genişliğinden BÜYÜK
    assert 4.0 - max(ps) > max(ps) - min(ps)


@pytest.mark.slow
def test_RAYLEIGH_MERDIVENI_YARGILANAMAZ_ama_GENLIK_2_57_KAT_OYNUYOR(merdiven):
    """SINIF: ÖLÇÜT — ve çıpanın en güçlü tek cümlesi.

    Merdiven HER ŞEYİ 1.25 kat ölçekler: `a` 1.08 → 2.109, `λ` kümesi,
    ızgara 181 → 355, `npml` 36 → 70, kayıt 720 → 1406. `ka` penceresi
    dört basamakta da SABİT (0.1508–0.3770).

        s          1.0000     1.2500     1.5625     1.9531
        p        3.261441   3.332617   3.325972   3.296608
        gürültü  1.43e-02   3.21e-03   1.77e-03   1.38e-03
        W/W_tam    1.9719     0.7677     1.0386     0.8613

    **HÜKÜM: YARGILANAMAZ** — ardışık farklar işaret değiştiriyor.
    Sebebi Task 4'ün mutlak `j` merdiveninde ölçtüğünün AYNISI: saçıcının
    merdivenlenmiş etkin yarıçapı `a_eff/a` bir kuvvet yasası değil KAFES
    ARİTMETİĞİDİR ve düğüm sayısı sıçradıkça zıplar.

    **AMA ASIL SAYI BU:** `W`nin genliği merdiven boyunca **2.57 kat**
    oynarken (`W/W_tam` 0.7677 → 1.9719) üs yalnızca **0.0712** oynuyor.
    Yani `W` etkin yarıçapa dördüncü kuvvetten bağlıyken ÜS ona neredeyse
    kör — çıpanın "boyut ayırıcı" olmasının ölçülmüş karşılığı budur.
    """
    p, gurultu, oran = [], [], []
    for g in merdiven:
        lam, W = _oku(g, MERDIVEN_JL)
        Wm = np.array([_mie_W(2 * math.pi * g["a"] / L,
                              math.sqrt(g["eps_r"]), g["a"]) for L in lam])
        p.append(_us(lam, W))
        oran.append(float(np.mean(W / Wm)))
        havuz = []
        for na in (61, 121, 241):
            ac = np.linspace(0.0, 2 * math.pi, na, endpoint=False)
            l2, w2 = _oku(g, MERDIVEN_JL, acilar=ac)
            havuz += [_us(l2, w2), _us(l2[:-1], w2[:-1]), _us(l2[1:], w2[1:])]
        gurultu.append(float(np.std(havuz, ddof=1) / abs(np.mean(havuz))))
    for beklenen, olculen in zip(
            (3.261441, 3.332617, 3.325972, 3.296608), p):
        assert olculen == pytest.approx(beklenen, abs=2e-4)
    ka = [2 * math.pi * g["a"] / (1.0 / g["fr"][MERDIVEN_JL[0]])
          for g in merdiven]
    assert max(ka) - min(ka) < 1e-4          # pencere SABİT
    lad = convergence_ladder(p, [MERDIVEN_R ** -i for i in range(4)],
                             noise=gurultu, ratio=MERDIVEN_R)
    assert lad.verdict == "YARGILANAMAZ"
    assert lad.settled_order is None
    d = np.diff(p)
    assert np.any(d[:-1] * d[1:] < 0.0)
    # GENLİK oynuyor, ÜS oynamıyor
    assert max(oran) / min(oran) == pytest.approx(2.568, rel=0.02)
    assert max(p) - min(p) == pytest.approx(0.0712, abs=2e-3)


# ===========================================================================
# 3. ALETİN KAPILARI
# ===========================================================================
@pytest.mark.slow
def test_LAMBDA_ETIKETLEME_KAPISI_ve_MERDIVENIN_ONU_GOREMEDIGI(ana, merdiven):
    """SINIF: ÖLÇÜM ALETİNİN KAPISI — ve neden AYRI bir testin kilitlemesi gerek.

    `σ` FFT bininin GERÇEK frekansında ölçülür. Nominal bir λ kümesi
    istenirse binler oraya oturmaz:

        nominal 128    100   80   64      50   40   32
        gerçek  133.33 100   80   63.158  50   40   32.432   (en büyük %4.17)

    ve ölçülen üs **3.290854** çıkar; aynı `W`ler binin gerçek λ'sıyla
    etiketlenince **3.248311**. Fark **+0.042543**.

    **VE BU ARIZA ÇÖZÜNÜRLÜK MERDİVENİNİN TEK BAŞINA GÖREMEYECEĞİ BİR
    ARIZADIR.** `λ` ile `dx` birlikte ölçeklendiği için bağıl bin kayması
    merdivende sabit kalır; ölçülen etiketleme hatası dört basamakta

        +0.029448 / +0.032410 / +0.031798 / +0.031639

    yani mertebeler −0.4295 / +0.0854 / +0.0225 — hata sıfıra
    yakınsamıyor, hatta ilk adımda BÜYÜYOR. Merdiven "OTURDU" bile dese
    sayı yanlış kalırdı; kilitleyen şey **bin kapısıdır**.

    Kapı sevk edilen boruda 7 nominal λ'nın **3'ünü** reddediyor (`None`),
    kalan 4'ü zaten bin üstüne oturduğu için geçiyor — ve bir üs fiti
    BÜTÜN noktaları istediğinden, kapı çalıştığında fit KURULAMAZ.
    """
    fr = ana["fr"]
    jn = [int(np.argmin(np.abs(fr - 1.0 / L))) for L in NOMINAL]
    gercek = np.array([1.0 / fr[j] for j in jn])
    assert float(np.max(np.abs(gercek / np.array(NOMINAL) - 1.0))) == \
        pytest.approx(0.0417, abs=1e-3)
    _, Wn = _oku(ana, jn)
    assert _us(np.array(NOMINAL), Wn) == pytest.approx(3.290854, abs=1e-4)
    assert _us(gercek, Wn) == pytest.approx(3.248311, abs=1e-4)
    assert _us(np.array(NOMINAL), Wn) - _us(gercek, Wn) == pytest.approx(
        0.042543, abs=1e-4)
    # KAPI: nominal lambda ile `None`
    K = ana["konturlar"][0]
    reddedilen = 0
    for L, j in zip(NOMINAL, jn):
        E, dn = _kontur_alanlari(ana["F"][j], ana["harita"], K["idx"], K["nrm"])
        out = ntff_far_field(points=K["pts"], normals=K["nrm"],
                             lengths=K["lens"], values=E,
                             normal_derivatives=dn, wavelength=L,
                             bin_frequency=fr[j], angles=ACI)
        reddedilen += out is None
    assert reddedilen == 3

    # MERDİVEN BU ARIZAYI GÖRMÜYOR
    hata = []
    for i, g in enumerate(merdiven):
        s = MERDIVEN_R ** i
        nom = [MERDIVEN_TABAN["T0"] * s / j
               for j in (16.3, 20.7, 25.4, 30.6, 35.9, 41.3)]
        jj = [int(np.argmin(np.abs(g["fr"] - 1.0 / L))) for L in nom]
        _, Wg = _oku(g, jj)
        gg = np.array([1.0 / g["fr"][j] for j in jj])
        hata.append(_us(np.array(nom), Wg) - _us(gg, Wg))
    for beklenen, olculen in zip(
            (0.029448, 0.032410, 0.031798, 0.031639), hata):
        assert olculen == pytest.approx(beklenen, abs=1e-3)
    mert = [math.log(abs(hata[i]) / abs(hata[i + 1])) / math.log(MERDIVEN_R)
            for i in range(3)]
    assert mert[0] < 0.0                # ilk adımda hata BÜYÜYOR
    assert max(abs(m) for m in mert) < 0.5


@pytest.mark.slow
def test_K_USSU_DUSURULUNCE_US_TAM_BIR_BIRIM_KAYIYOR(ana):
    """SINIF: ÖLÇÜM — `scattering_width`teki `1/k` çarpanı fiziktir.

    `W = INT |I|^2 dθ / (8 π k |E_i|^2)`. `1/k ~ λ` olduğu için `k`
    çarpanı düşürülünce ölçülen üs TAM BİR BİRİM kayar:

        p = 3.242894  ->  4.242894   (fark 1.000000)

    Ve bu, **3B'nin üssünün tam üstüne** düşer — yani bu tek çarpan
    yanlış yazılsaydı modül "3B bulduk" derdi. Çıpanın boyut ayırıcı
    olması aletin bu satırına bağlıdır.
    """
    lam, W = _oku(ana, JL)
    _, Wk = _oku(ana, JL, k_ussu=False)
    assert _us(lam, Wk) - _us(lam, W) == pytest.approx(1.0, abs=1e-9)
    assert _us(lam, Wk) == pytest.approx(4.242894, abs=1e-4)


def test_OPTIK_TEOREM_KAPISI_BIR_TAUTOLOJIDIR():
    """SINIF: ÖLÇÜM — tam çözümü NASIL SERTİFİKA ETMEDİĞİMİZ.

    Ön ölçümün `p_MIE`yi doğrulamak için kullandığı optik teorem kapısı
    (`Re c_n = |c_n|²`) **fiziği değil, `H = J + iY`nin tutarlı
    yazıldığını** sınar. Cebir: `c = A/(A + iB)` ile `A`, `B` REEL olduğu
    sürece `c = 1/(1 + iq)`, `q = B/A`, dolayısıyla `Re c = 1/(1+q²) = |c|²`
    — **`A` ve `B`nin ne olduğundan bağımsız**.

    ÖLÇÜLDÜ: gerçek Bessel katsayılarında ihlal 1e-16 bandında; Bessel
    yerine RASTGELE REEL fonksiyonlar konunca da 1e-16 bandında. Kapı
    ikisini ayırt edemiyor, dolayısıyla serinin sertifikası olamaz.

    Serinin GERÇEK iki sertifikası bu dosyada: `ka → 0`da üssün 3'e
    gitmesi ve beş pencerede bağımsız FDTD ölçümüyle uyuşması.
    """
    for x in (0.1, 0.3, 0.9):
        c = _mie_c(x, 2.0)
        assert float(np.max(np.abs(c.real - np.abs(c) ** 2))) < 1e-14

    rng = np.random.default_rng(20260825)
    for _ in range(5):
        A = rng.normal(size=9)
        B = rng.normal(size=9)
        sahte = A / (A + 1j * B)        # FİZİK YOK, sadece "A + iB" formu
        assert float(np.max(np.abs(sahte.real - np.abs(sahte) ** 2))) < 1e-14


def test_SACILMA_GENISLIGI_GIRDI_DENETIMLERI():
    """SINIF: ÇEVİRİ DENETİMİ — `W` düzgün olmayan açı ızgarasını REDDEDER."""
    I = np.ones(16, dtype=complex)
    duzgun = np.linspace(0.0, 2 * math.pi, 16, endpoint=False)
    ortak = dict(wavelength=20.0, bin_frequency=0.05, incident_amplitude=1.0)
    assert scattering_width(I, angles=duzgun, **ortak) > 0.0
    with pytest.raises(InvalidInputError, match="DUZGUN"):
        scattering_width(I, angles=np.linspace(0.0, math.pi, 16), **ortak)
    with pytest.raises(InvalidInputError, match="en az 8 aci"):
        scattering_width(np.ones(4, dtype=complex),
                         angles=np.linspace(0.0, 2 * math.pi, 4,
                                            endpoint=False), **ortak)
    with pytest.raises(InvalidInputError, match="ayni sekilde"):
        scattering_width(np.ones(8, dtype=complex), angles=duzgun, **ortak)
    with pytest.raises(InvalidInputError, match="incident_amplitude"):
        scattering_width(I, angles=duzgun, wavelength=20.0, bin_frequency=0.05,
                         incident_amplitude=0.0)
    assert scattering_width(I, angles=duzgun, wavelength=20.0,
                            bin_frequency=0.0501,
                            incident_amplitude=1.0) is None
    # `1/k` gerçekten orada: aynı `I` ile lambda iki katına çıkınca W iki kat
    w1 = scattering_width(I, angles=duzgun, wavelength=20.0,
                          bin_frequency=0.05, incident_amplitude=1.0)
    w2 = scattering_width(I, angles=duzgun, wavelength=40.0,
                          bin_frequency=0.025, incident_amplitude=1.0)
    assert w2 / w1 == pytest.approx(2.0, rel=1e-12)


@pytest.mark.slow
def test_FRAUNHOFER_TABLOSUNUN_SACILMA_SATIRLARI(ana):
    """SINIF: ÖLÇÜM ALETİNİN KAPISI — `_uzaklik_kapisi`nin saçılma satırları.

    `tests/em/test_uzak_alan.py` aynı tabloyu AÇIKLIK tarafında ölçüyor;
    burada KAPALI KONTUR tarafı ölçülüyor. Sevk edilen R → ∞ deseni, aynı
    konturun TAM Green fonksiyonuyla sonlu R'de hesaplanmış deseniyle
    karşılaştırılıyor (a = 2, rc = 40, 121 açı):

        R/λ            10        30       100       300      1000
        λ = 40      3.743e-01 1.163e-01 3.305e-02 1.018e-02 2.978e-03
        λ = 60      2.358e-01 7.431e-02 2.096e-02 6.703e-03 1.972e-03
        λ = 100     1.453e-01 4.809e-02 1.412e-02 4.506e-03 1.273e-03

    Üç satır da 1/R gidiyor (ardışık oranlar 3.02–3.55, adım çarpanları 3
    ve 3.33). **ESKİ KAPI (`R ≥ a²/λ`) BURADA ANLAMINI YİTİRİYOR:** a = 2
    ile `a²/λ` sırasıyla 0.100 / 0.067 / 0.040'tır, yani `R = 10 λ`
    noktasında eski kapı **4000 / 9000 / 25000 kat** rahat geçer — ve orada
    desen %37.4 / %23.6 / %14.5 yanlıştır.
    """
    K = ana["konturlar"][0]
    beklenen = {12: (1.453e-01, 4.809e-02, 1.412e-02, 4.506e-03, 1.273e-03),
                20: (2.358e-01, 7.431e-02, 2.096e-02, 6.703e-03, 1.972e-03),
                30: (3.743e-01, 1.163e-01, 3.305e-02, 1.018e-02, 2.978e-03)}
    for j, satir in beklenen.items():
        L = 1.0 / ana["fr"][j]
        k = 2 * math.pi / L
        E, dn = _kontur_alanlari(ana["F"][j], ana["harita"], K["idx"], K["nrm"])
        inf = ntff_far_field(points=K["pts"], normals=K["nrm"],
                             lengths=K["lens"], values=E,
                             normal_derivatives=dn, wavelength=L,
                             bin_frequency=ana["fr"][j], angles=ACI)
        olculen = []
        for rl in (10, 30, 100, 300, 1000):
            R = rl * L
            r = np.stack([R * np.cos(ACI), R * np.sin(ACI)], axis=1)
            d = r[:, None, :] - K["pts"][None, :, :]
            dd = np.linalg.norm(d, axis=2)
            ndot = ((d / dd[:, :, None]) * K["nrm"][None, :, :]).sum(axis=2)
            son = ((-k * E[None, :] * hankel2(1, k * dd) * ndot
                    + hankel2(0, k * dd) * dn[None, :])
                   * K["lens"][None, :]).sum(axis=1)
            A = np.abs(son) / np.abs(son).max()
            B = np.abs(inf) / np.abs(inf).max()
            olculen.append(float(np.sqrt(((A - B) ** 2).mean())))
        for beklenen_deger, v in zip(satir, olculen):
            assert v == pytest.approx(beklenen_deger, rel=0.05), (j, v)
        oranlar = [olculen[i] / olculen[i + 1] for i in range(4)]
        assert min(oranlar) > 3.0 and max(oranlar) < 3.6, (j, oranlar)
        # ESKİ KAPININ geçirdiği kat sayısı
        a2lam = ana["a"] ** 2 / L
        assert (10 * L) / a2lam == pytest.approx(10 * L * L / ana["a"] ** 2)
        assert (10 * L) / a2lam > 3900.0


# ===========================================================================
# 4. SEVK EDİLMİŞ DOKSTRİNG DENETİMİ — Task 5'te SINANABİLİR olanlar
# ===========================================================================
@pytest.mark.slow
def test_2B_SOGURUCUNUN_ACI_TARAMASI_Task_4_UN_DURUST_OLUMSUZU(sogurucu):
    """SINIF: ÖLÇÜM — `boundary.PML.sigma_2d`'nin İDDİASI ilk kez AÇIYA GÖRE.

    Task 4 "2B soğurucuda açı taraması yapılmadı" diye bırakmıştı; sevk
    edilen `ntff_far_field` bunu ilk kez ölçülebilir yapıyor. Vakumda
    merkezi nokta kaynak, kontur `rc = 30`, `λ = 16`, 768 adım. REFERANS
    soğurucusu erişilemeyecek kadar uzakta — ve bu bir varsayım değil:
    N = 541 ile N = 641 deseni **3.07e-15** içinde aynı veriyor.

    Kalınlığa göre BAĞIL desen hatası:

        d        RMS        0° (yüze DİK)   45° (KÖŞEYE)    45°/0°
        8      5.787e-03    1.228e-04       1.326e-02       108.0
        15     3.013e-03    2.337e-03       6.068e-03         2.60
        25     2.194e-04    1.831e-04       5.001e-04         2.73
        40     1.802e-06    4.016e-06       4.217e-07         0.11

    `d ≤ 25`te **en büyük hata her zaman 45°'de**, yani skaler dereceli
    iletkenin en az eşleştiği yönde — `sigma_2d` dokstringinin
    "eşleşme koşulu yalnız dik bileşende sağlanır" cümlesinin ilk sayısal
    karşılığı.

    **DÜRÜST OLUMSUZ:** `d = 40`ta oran TERSİNE dönüyor (0.11). Bu bir
    referans tabanı artefaktı DEĞİL (referans 1e-15'te temiz); ölçülen
    gerçek bir davranış ve bu modülde kovalanmadı.
    """
    r541, r641 = sogurucu["ref"][541], sogurucu["ref"][641]
    fark = np.abs(r641) / np.abs(r541) - 1.0
    assert float(np.sqrt((fark ** 2).mean())) < 1e-13
    beklenen = {8: (5.787e-03, 1.228e-04, 1.326e-02, 108.0),
                15: (3.013e-03, 2.337e-03, 6.068e-03, 2.60),
                25: (2.194e-04, 1.831e-04, 5.001e-04, 2.73),
                40: (1.802e-06, 4.016e-06, 4.217e-07, 0.11)}
    k45 = len(ACI) // 8
    for d, (rms, dik, kose, oran) in beklenen.items():
        b = np.abs(sogurucu["kucuk"][d]) / np.abs(r541) - 1.0
        assert float(np.sqrt((b ** 2).mean())) == pytest.approx(rms, rel=0.05), d
        assert float(np.abs(b[0])) == pytest.approx(dik, rel=0.05), d
        assert float(np.abs(b[k45])) == pytest.approx(kose, rel=0.05), d
        assert float(np.abs(b[k45]) / np.abs(b[0])) == pytest.approx(
            oran, rel=0.05), d
        if d <= 25:
            # EN BÜYÜK hata KÖŞEDE: dört köşe simetriktir, `argmax` hangisini
            # seçtiği ızgaraya bağlıdır — sınanan şey DEĞERDİR.
            assert float(np.abs(b).max()) == pytest.approx(kose, rel=0.05), d


@pytest.mark.slow
def test_MODE_SPECTRUM_BIR_KIP_DEDEKTORU_DEGILDIR(ana):
    """SINIF: ÖLÇÜM — sevk edilmiş bir dokstringin BAŞLIĞI aşırı iddialı.

    `mode_spectrum`un ilk satırı "Kayittan kip frekanslari" der. Task 5'te
    ilk kez KİPİ OLMAYAN bir kayıt (açık uzayda saçılan alan; boşluk yok,
    duran kip yok) verilebiliyor — ve alet yine "tepe" döndürüyor:

        blackman, taban 0.01 : 6 tepe, oranlar 1.000 2.195 2.842 3.456 3.965
        hann,     taban 0.05 : 6 tepe, oranlar 1.000 1.836 2.350 2.880 3.265
        dikdortgen, taban 0.3: 4 tepe, oranlar 1.000 1.234 1.682 1.925

    Üç pencere üç farklı oran kümesi veriyor — yani bu sayılar fizik
    TAŞIMIYOR, pencerenin kendi tepelerini sayıyorlar. Aletin "adsız
    tepeler, teori yok" beyanı DOĞRU; eksik olan, **tepenin kip anlamına
    gelmediğinin** yazılmasıydı. `em_obs.mode_spectrum` dokstringi bu
    ölçümle güncellendi.
    """
    sac = FieldRecord(probes=ana["ham"].probes - ana["ham_ref"].probes,
                      probe_indices=ana["ham"].probe_indices,
                      window=replace(ana["ham"].window), final=ana["ham"].final)
    beklenen = {("blackman", 0.01): (6, (1.0, 2.1948, 2.8416, 3.4560, 3.9652)),
                ("hann", 0.05): (6, (1.0, 1.8362, 2.3501, 2.8803, 3.2647)),
                ("dikdortgen", 0.3): (4, (1.0, 1.2338, 1.6818, 1.9247))}
    oranlar = {}
    for (pencere, taban), (n_tepe, ilk) in beklenen.items():
        ms = mode_spectrum(sac, pad=4, window=pencere, peak_floor=taban)
        assert len(ms.frequencies) == n_tepe, pencere
        r = ms.ratios()
        for beklenen_oran, olculen in zip(ilk, r):
            assert olculen == pytest.approx(beklenen_oran, abs=2e-3), pencere
        oranlar[pencere] = r[:4]
    # ÜÇ PENCERE ÜÇ FARKLI CEVAP -> sayılar fizik taşımıyor
    assert oranlar["blackman"][1] != pytest.approx(oranlar["hann"][1], rel=0.05)
    assert oranlar["hann"][1] != pytest.approx(
        oranlar["dikdortgen"][1], rel=0.05)


def test_SLOW_ESIGI_OLCULDU():
    """`slow` işareti bu dosyada FIXTURE'a bağlıdır.

    Dört fixture da FDTD koşar: `ana` (2 koşum), `merdiven` (8), `pencereler`
    (4), `sogurucu` (6). Ölçüldü: bu dosyanın tamamı **35.4 s**,
    `-m "not slow"` ile **0.4 s**. Fixture
    maliyeti `pytest --durations` çıktısında `setup` olarak görünür ve koşum
    sırasına göre herhangi birine yazılabilir (Modül 4'ün ölçtüğü tuzak).

    İşaretsiz kalan üç test saf aritmetiktir (optik teorem tautolojisi,
    `scattering_width` girdi denetimleri, bu test) ve `-m "not slow"`
    koşumunda da koşar.
    """
    kaynak = open(__file__, encoding="utf-8").read()
    assert kaynak.count("@pytest.mark.slow\n") == 11
    for ad in ("OPTIK_TEOREM_KAPISI_BIR_TAUTOLOJIDIR",
               "SACILMA_GENISLIGI_GIRDI_DENETIMLERI",
               "SLOW_ESIGI_OLCULDU"):
        i = kaynak.index(f"def test_{ad}")
        assert "@pytest.mark.slow" not in kaynak[i - 40:i], ad
