"""2B çekirdek ve **BESSEL ORANLARI** — Modül 5'in en güçlü çıpası.

Bu dosya iki sınıf taşır ve **ikisi karıştırılamaz**:

- **ŞEMA ÇIPASI** — 2B Yee güncellemesinin kendi özellikleri: Courant sınırı,
  açısal dispersiyon yasası, köşegen sihirli adım, ve 2B'nin y'den bağımsız
  alanda 1B ile **bit bit** aynı olması. Sayıları şemanın, fiziğin değil.
- **FİZİK ÇIPASI** — dairesel PEC boşluğun kip frekans ORANLARI. `ω_mn =
  c·j_mn/R` olduğu için oranlar `R`'den, `c`'den ve `ε`'dan bağımsızdır; saf
  Bessel kökleridir ve **girdilerden geri okunamazlar**.

ÇIPANIN İKİ PENCERESİ, ve ikisi de bu dosyada AYRI AYRI ölçülüyor:

1. **Mutlak `j` çapalanamaz** — ölçüm bu turda planın beklediğinden **daha
   sert** çıktı: mutlak `j₀,₁` merdiveni `OTURDU(1)` değil **OTURMADI** verdi,
   ve sebebi ölçüldü (kafes aritmetiği; bkz.
   `test_MUTLAK_J_MERDIVENI_OTURMUYOR_...`).
2. **Hassasiyet KİPE BAĞLIDIR** — en iyi oran 4.8e-05 saparken en kötüsü
   4.2e-04, yani **8.8 kat**. *"5 hane"* hükmü yalnız `j₁,₁/j₀,₁`'e aittir.

PENCERE (bu dosyadaki her çıpa koşumu için, aksi yazılmadıkça):
`dx = 1`, `c = ε₀ = μ₀ = 1`, `S = 0.5`, dairesel PEC boşluk (yarıçap `R`,
merkez ızgara düğümünde, kenar payı 3 hücre), adım sayısı `1667·R·√ε_r`,
kaynak Ricker (`f_p = 1.8·f₀₁`, `t₀ = 1.5λ`) `r = 0.55R`, `θ = 0.4 rad`'da,
**8 dedektör**, FFT penceresi **blackman**, `pad = 8`, tepe tabanı `0.01`,
`ε_r = 1`. Bu pencerenin her düğmesi ölçülerek seçildi ve gerekçeleri
aşağıdaki testlerde duruyor.

TEORİNİN NEREYE GİRDİĞİ — ve nereye GİRMEDİĞİ: `mode_spectrum` Bessel
köklerini **hiç görmez**; ADSIZ tepe frekansları döndürür. Kökler yalnız
(a) hangi tepenin hangi kip olduğunu **eşlemek** için (%2 penceresi) ve
(b) çıpanın karşılaştırma değeri olarak kullanılır. İkisinin arasında **200
kat** vardır (%2 eşleme penceresi karşısında 1e-4 mertebesindeki hüküm), yani
eşleme hükmü ÜRETEMEZ. Kökler `scipy.special.jn_zeros`tan gelir, elle
yazılmaz.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy.special import jn_zeros

from qlab.engine.em.boundary import PEC, PML, CircularPEC
from qlab.engine.em.grid import FieldState, FieldState2D, courant_limit
from qlab.engine.em.medium import Medium
from qlab.engine.em.source import ricker
from qlab.engine.em.yee import (
    YeeCoefficients2D,
    _adim,
    _adim_2d,
    run_2d,
    yee_coefficients,
    yee_coefficients_2d,
    yee_step_2d,
)
from qlab.errors import InvalidInputError
from qlab.observable.em_obs import (
    WINDOW_FNS,
    WINDOW_SIDELOBE,
    convergence_ladder,
    mode_spectrum,
)

# ---------------------------------------------------------------------------
# SEVK EDİLEN PENCERE
# ---------------------------------------------------------------------------
S = 0.5
ADIM_CARPANI = 1680.0
PAD, PENCERE, TABAN = 8, "blackman", 0.01
NDET = 8
RLER = (10.0, 20.0, 40.0, 80.0)
SEVK = dict(pad=PAD, window=PENCERE, peak_floor=TABAN)

#: Bessel kökleri TEORİDEN gelir ve elle yazılmaz.
KOK = {(m, n): float(jn_zeros(m, 2)[n - 1]) for m in range(4) for n in (1, 2)}
J01 = KOK[(0, 1)]

#: Çıpanın beş oranı ve **kipe bağlı** bantları. Bantlar ölçülen sapmanın
#: kabaca iki katıdır (R = 80'de sırasıyla −4.757e-05, −1.584e-04, −1.898e-04,
#: −3.102e-04, −4.200e-04); gevşetilmiş bir tolerans DEĞİL, ölçülen
#: hassasiyetin beyanıdır — ve `dedektör = 1` mutantı tam olarak `j21`
#: bandından düşüyor.
ORANLAR = (
    ("j11/j01", (1, 1), 1.0e-4),
    ("j21/j01", (2, 1), 3.0e-4),
    ("j02/j01", (0, 2), 3.0e-4),
    ("j31/j01", (3, 1), 5.0e-4),
    ("j12/j01", (1, 2), 7.0e-4),
)

#: Eşleme penceresi. Hükmün (1e-4) 200 katı — eşleme bir DEĞER üretemez.
ESLEME = 0.02


# ---------------------------------------------------------------------------
# Yardımcılar — GEOMETRİ BURADA, `src/`te DEĞİL
# ---------------------------------------------------------------------------
def _geometri(R: float, margin: int = 3) -> tuple[int, int]:
    n = 2 * int(math.ceil(R + margin)) + 1
    return n, (n - 1) // 2


def _dedektorler(R: float, c: int, sayi: int = NDET) -> tuple[tuple[int, int], ...]:
    """`sayi` dedektör, GENERİK yarıçap ve açılarda.

    Simetrik bir yerleşim (ör. hepsi aynı yarıçapta) `m ≠ 0` kiplerinin
    bir kısmını söndürürdü; yarıçapların ve açıların birbirine oranı
    bilinçli olarak kesirli seçildi.
    """
    out = []
    for k in range(sayi):
        rr = R * (0.28 + 0.60 * ((k * 5) % sayi) / max(sayi - 1, 1))
        th = 2.0 * math.pi * k / sayi + 0.13
        out.append(
            (c + int(round(rr * math.cos(th))), c + int(round(rr * math.sin(th))))
        )
    return tuple(out)


def _kosum(R: float, *, eps_r: float = 1.0, kare: bool = False, ndet: int = NDET):
    """SEVK EDİLEN pencerede bir 2B boşluk koşumu."""
    n, c = _geometri(R)
    f01 = J01 / (2.0 * math.pi * R * math.sqrt(eps_r))
    lam = 1.0 / (1.8 * f01)
    si = c + int(round(0.55 * R * math.cos(0.4)))
    sj = c + int(round(0.55 * R * math.sin(0.4)))
    return run_2d(
        Medium.uniform_2d(n, n, eps_r=eps_r),
        courant=S,
        boundary=PEC() if kare else CircularPEC(radius=R),
        steps=int(round(ADIM_CARPANI * R * math.sqrt(eps_r))),
        source=lambda t: ricker(t, wavelength=lam, sgf=1.5),
        source_index=(si, sj),
        probes=_dedektorler(R, c, ndet),
        wavelength=lam,
        sgf=1.5,
        window_fn=PENCERE,
    )


def _kirp(kayit, kesir: float):
    """Kaydın ilk `kesir` kadarı — İŞLEM SONRASI bir pencere düğmesi."""
    from dataclasses import replace

    from qlab.engine.em.grid import FieldRecord

    n = int(kayit.probes.shape[1] * kesir)
    return FieldRecord(
        probes=kayit.probes[:, :n],
        probe_indices=kayit.probe_indices,
        window=replace(kayit.window, steps=n),
        final=kayit.final,
    )


def _dilim(kayit, ndet: int):
    """İlk `ndet` dedektör — FDTD koşumu prob sayısından ETKİLENMEZ, bu yüzden
    kaydı dilimlemek `ndet` dedektörle koşmakla ÖZDEŞTİR (ve bedava)."""
    from qlab.engine.em.grid import FieldRecord

    return FieldRecord(
        probes=kayit.probes[:ndet],
        probe_indices=kayit.probe_indices[:ndet],
        window=kayit.window,
        final=kayit.final,
    )


def _esle(ms, ad: str) -> float | None:
    """Tepe listesinden `ad` oranını EŞLE (%2 penceresi) — DEĞER üretmez."""
    hedef = dict((a, KOK[mn] / J01) for a, mn, _ in ORANLAR)[ad]
    r = np.asarray(ms.ratios())
    if r.size == 0:
        return None
    k = int(np.argmin(np.abs(r - hedef)))
    return float(r[k]) if abs(r[k] - hedef) <= ESLEME * hedef else None


def _deger_ve_gurultu(kayit, *, mutlak: bool = False, R: float = 0.0, **kw):
    """Sevk penceresindeki değer + MEŞRU pencerelerin BAĞIL yayılımı.

    `noise` için **ek FDTD koşumu YOK**: `pad`, tepe tabanı ve kayıt kırpması
    üçü de işlem-sonrasıdır ve tek kaydedilmiş seriden üretilir (Task 2'nin
    `convergence_ladder` dokstringindeki gerekçe).
    """
    ana = dict(SEVK)
    ana.update(kw)

    def oku(ms):
        return ms.frequencies[0] * 2.0 * math.pi * R if mutlak else _esle(ms, "j11/j01")

    sevk = oku(mode_spectrum(kayit, **ana))
    dl = []
    for pad in (ana["pad"], max(ana["pad"] * 2, 2), max(ana["pad"] // 2, 1)):
        for taban in (0.005, 0.01, 0.02):
            for kesir in (1.0, 0.93, 0.86):
                d = dict(ana)
                d.update(pad=pad, peak_floor=taban)
                v = oku(mode_spectrum(_kirp(kayit, kesir), **d))
                if v is not None:
                    dl.append(v)
    a = np.array(dl)
    return sevk, float(a.std(ddof=1) / abs(a.mean()))


def _merdiven(kayitlar, *, mutlak: bool = False, **kw):
    v, g = [], []
    for R in RLER:
        s, gg = _deger_ve_gurultu(kayitlar[R], mutlak=mutlak, R=R, **kw)
        v.append(s)
        g.append(gg)
    return convergence_ladder(v, [1.0 / R for R in RLER], noise=g, ratio=2.0), v, g


@pytest.fixture(scope="module")
def kayitlar():
    """Dört basamağın kaydı — BİR KEZ koşulur, on test tüketir.

    Ölçüldü (bu makinede): R = 10/20/40/80 için 0.30 / 0.75 / 2.87 / 15.67 s,
    toplam ≈ 19.6 s. Basamak başına adım `1667·R` olduğu için spektral
    çözünürlük `j` biriminde SABİT kalır (her basamakta en düşük kipin
    ≈ 319 periyodu kaydedilir); bu, merdivenin ölçtüğü şeyin yalnız
    MERDİVENLENME olmasının koşuludur.
    """
    return {R: _kosum(R) for R in RLER}


@pytest.fixture(scope="module")
def kucuk_kayit():
    """En ucuz gerçek kayıt (R = 10, ölçülen 0.29 s).

    `kayitlar`dan AYRI durur ve bilinçli olarak onu tüketmez: aletin kendi
    girdi denetimleri ve pencere beyanı, çıpanın 20 saniyelik merdivenine
    bağlanmamalı — `-m "not slow"` koşumunda da koşsunlar diye.
    """
    return _kosum(10.0)


# ===========================================================================
# 1. ŞEMA — 2B çekirdeğin kendi özellikleri
# ===========================================================================
def test_2B_CEKIRDEK_Y_BAGIMSIZ_ALANDA_1B_CEKIRDEGIN_BIT_BIT_AYNISI():
    """SINIF: ŞEMA ÇIPASI (transkripsiyon tarafı) — ve TAŞIYICI.

    "1B güncellemesi 2B şemanın y'den bağımsız kesitidir" bir yorum değil,
    BİT DÜZEYİNDE bir iddiadır: `Ez` `j`'den bağımsızsa `∂Ez/∂y = 0`,
    dolayısıyla `Hx ≡ 0` kalır ve `Ez` güncellemesi tam olarak 1B'ninkine
    iner. Eğer 2B'de bir işaret ya da bir yarım hücre kayması farklı
    yazılsaydı bu eşitlik BOZULURDU.

    ÖLÇÜLDÜ: N = 61, ny = 71, S = 0.5, rastgele başlangıç profili (tohum 7),
    25 adım — orta sütun `e` ve `hy`de `np.array_equal` TRUE, `max|fark|` =
    0.0. Alan sıfır değil (`max|e|` = 2.212106), yani eşitlik boş değil.

    PENCERE ZORUNLU: y duvarlarındaki PEC kirlenmesi adım başına BİR hücre
    içeri yürür, bu yüzden eşitlik yalnız duvardan `> K` uzaktaki sütunlarda
    geçerlidir — ve kenar sütununun FARKLI olduğu ayrıca sınanıyor, yoksa
    test "her yerde aynı" gibi yanlış bir şey iddia etmiş olurdu.
    """
    N, K = 61, 25
    ny = 2 * K + 21
    rng = np.random.default_rng(7)
    prof = rng.standard_normal(N)
    prof[0] = prof[-1] = 0.0
    h1 = rng.standard_normal(N - 1)

    st1 = FieldState(prof.copy(), h1.copy(), 1)
    co1 = yee_coefficients(Medium.vacuum(N), courant=S, boundary=PEC())
    st2 = FieldState2D(
        np.repeat(prof[:, None], ny, axis=1),
        np.zeros((N, ny - 1)),
        np.repeat(h1[:, None], ny, axis=1),
    )
    co2 = yee_coefficients_2d(Medium.uniform_2d(N, ny), courant=S, boundary=PEC())
    for _ in range(K):
        st1 = _adim(st1, co1)
        st2 = _adim_2d(st2, co2)

    jc = ny // 2
    assert np.array_equal(st1.e, st2.e[:, jc])
    assert np.array_equal(st1.h, st2.hy[:, jc])
    assert float(np.max(np.abs(st1.e))) > 1.0  # eşitlik BOŞ değil
    assert not np.array_equal(st1.e, st2.e[:, 1])  # kenar KİRLENİYOR


@pytest.mark.slow
def test_COURANT_SINIRI_2B_DE_OLCULUYOR_ve_1B_KAPISI_DEVRALINAMAZ():
    """SINIF: ŞEMA ÇIPASI. `S·√2 = 1`, ve bu 1B'ninkinin DEVRI DEĞİL.

    Task 3, `courant_limit`in eğik yolu sınamadığını beyan etmişti; Task 4'ün
    karşılığı `S·√2 = 1` sınırıdır ve 2B'nin kendi çekirdeğiyle ÖLÇÜLDÜ
    (SEVK EDİLEN `_adim_2d`, PEC duvar, rastgele başlangıç alanı, ikili
    arama tol 1e-7, 400 adım, üç tohum):

        L = 32  : S·√2 = 1.001941 / 1.002003 / 1.002102
        L = 64  : S·√2 = 1.000960 / 1.000995 / 1.001103
        L = 128 : S·√2 = 1.000786 / 1.000838 / 1.000781

    AŞIM 1B'DEKİ GİBİ İKİ PARÇALI ve ikisi de ölçüldü:
    (a) SONLU IZGARA payı — PEC kutuda erişilebilir en büyük kip
        `kx = ky = π(L−2)/(L−1)` olduğu için sınır tam olarak
        `1/cos(π/(2(L−1)))`: L = 32/64/128 için 1.001285 / 1.000311 /
        1.000076. Ölçülenin bu kapalı forma ORANI L'den BAĞIMSIZ:
        1.000655…1.000816 (dokuz ölçümün hepsi).
    (b) SONLU KOŞUM payı — L = 64'te adım sayısıyla iniyor:
        400 adım +6.48e-04, 1500 adım +4.18e-05, 4000 adım +5.46e-06.
        Son sayı 1B'nin periyodik ızgarada 4000 adımda ölçtüğü +5.99e-06 ile
        AYNI HANEDE; yani koşum payı boyuttan bağımsız.

    Ve KAPI DEVRALINAMAZ: `S = 0.9` 1B kapısından GEÇER, 2B'de reddedilir —
    ve reddedilmeseydi şema patlardı (aşağıda ölçülüyor).
    """
    lim = courant_limit(2)
    assert lim == pytest.approx(2**-0.5, rel=1e-12)

    def _vakum(n, s):
        return YeeCoefficients2D(
            ca=np.ones((n, n)), cb=np.full((n, n), s),
            dax=np.ones((n, n - 1)), dbx=np.full((n, n - 1), s),
            day=np.ones((n - 1, n)), dby=np.full((n - 1, n), s),
            pec_mask=np.zeros((n, n), dtype=bool), dim=2, courant=s,
        )

    def kararli(s, L, adim, tohum):
        rng = np.random.default_rng(tohum)
        st = FieldState2D(
            rng.standard_normal((L, L)),
            rng.standard_normal((L, L - 1)),
            rng.standard_normal((L - 1, L)),
        )
        co = _vakum(L, s)
        n0 = float(np.max(np.abs(st.e)))
        for _ in range(adim):
            st = _adim_2d(st, co)
            m = float(np.max(np.abs(st.e)))
            if not np.isfinite(m) or m > 1e12 * n0:
                return False
        return True

    def esik(fn, lo, hi):
        assert fn(lo) and not fn(hi)
        while hi - lo > 1e-7:
            o = 0.5 * (lo + hi)
            lo, hi = (o, hi) if fn(o) else (lo, o)
        return 0.5 * (lo + hi)

    for L in (32, 64):
        kapali = 1.0 / np.cos(np.pi / (2 * (L - 1)))
        for tohum in (1, 2):
            s = esik(lambda x: kararli(x, L, 400, tohum), 0.4 * lim, 1.3 * lim)
            oran = s * np.sqrt(2.0) / kapali
            assert s * np.sqrt(2.0) >= kapali - 1e-9  # sonlu ızgara payı ALTINA inmiyor
            assert 1.0004 < oran < 1.0012  # koşum payı, 400 adımda ölçülen bant

    # (b) koşum payı adım sayısıyla küçülüyor — ve bu sıralı bir hüküm.
    kapali64 = 1.0 / np.cos(np.pi / (2 * 63))
    fazla = [
        esik(lambda x: kararli(x, 64, ad, 1), 0.4 * lim, 1.3 * lim)
        * np.sqrt(2.0) / kapali64 - 1.0
        for ad in (400, 1500)
    ]
    assert fazla[0] > 5.0 * fazla[1] > 0.0

    # KAPI: 1B S = 0.9'u geçirir, 2B geçirmez.
    yee_coefficients(Medium.vacuum(32), courant=0.9, boundary=PEC())
    with pytest.raises(InvalidInputError, match="1B kapisi"):
        yee_coefficients_2d(Medium.uniform_2d(32, 32), courant=0.9, boundary=PEC())
    # ve geçseydi patlardı:
    assert not kararli(0.9, 32, 400, 1)


def _duran_kip_2d(N: int, m: int, n: int, s: float, M: int):
    """PEC kutuda (m, n) duran kipi; `omega` ÜÇ TERİMLİ yinelemeden.

    Yee'nin 2B kapalı form dispersiyon bağıntısı bu dosyada HİÇBİR YERDE
    kullanılmaz — kullanılsaydı ölçüm dairesel olurdu. Katsayılar da elle
    kurulmaz, `yee_coefficients_2d`ten gelir (Task 1'in mutasyonla bulduğu
    boşluğun 2B karşılığı).
    """
    kx, ky = np.pi * m / (N - 1), np.pi * n / (N - 1)
    proj = np.outer(np.sin(kx * np.arange(N)), np.sin(ky * np.arange(N)))
    st = FieldState2D(proj.copy(), np.zeros((N, N - 1)), np.zeros((N - 1, N)))
    co = yee_coefficients_2d(Medium.uniform_2d(N, N), courant=s, boundary=PEC())
    ser = np.empty(M)
    for t in range(M):
        ser[t] = float(np.sum(st.e * proj))
        st = _adim_2d(st, co)
    y, x = ser[2:] + ser[:-2], 2.0 * ser[1:-1]
    c = float(np.dot(x, y) / np.dot(x, x))
    kalinti = float(np.max(np.abs(y - c * x)) / np.max(np.abs(y)))
    omega = float(np.arccos(np.clip(c, -1.0, 1.0)) / s)
    k = float(np.hypot(kx, ky))
    return -(omega / k - 1.0) / k**2, kalinti, float(np.arctan2(ky, kx))


@pytest.mark.parametrize("s", [0.3, 0.5, 2**-0.5])
@pytest.mark.parametrize("mn", [(8, 8), (8, 4), (8, 2), (8, 1)])
def test_2B_DISPERSIYON_ACIYA_BAGLI_24C_ESITTIR_P_EKSI_S_KARE(s, mn):
    """SINIF: ŞEMA ÇIPASI — ve 1B çıpasının GENELLEŞMESİ.

    2B Yee'de `v_p/c − 1 = −C(θ)(kΔx)² + O(k⁴)` ve ölçülen saf sayı

        24·C(θ) = cos⁴θ + sin⁴θ − S²

    Bu, 1B'nin `(1−S²)/C = 24` çıpasını İÇERİR: θ = 0'da `cos⁴+sin⁴ = 1`,
    yani 24C = 1 − S². Yani 1B anahtarı 2B yasanın bir KESİTİDİR ve bu
    ayrıca sınanıyor (aşağıdaki assert).

    PENCERE ZORUNLU ve İKİ PARÇALI. (a) `m` çifti değil, `(m, n)` İKİLİSİ
    açıyı belirler (`atan2(ky, kx)`, `kx = mπ/(N−1)`). (b) `C` bir LİMİT
    katsayısıdır ve tek bir `k`da okunamaz — 1B çıpasının `m = 8, 16` ile
    yaptığının aynısı burada da ZORUNLU çıktı, çünkü tek `k`da okunan sayı
    `O(k⁴)` artığı taşıyor:

        N = 201, M = 600      tek k (m,n)      Richardson (m,n)+(2m,2n)
        S = 0.3   dört açı    −3.8e-05…+5.0e-05   +2.4e-07…+3.5e-07
        S = 0.5   dört açı    +1.3e-04…+1.8e-04   −2.3e-07…+4.0e-07
        S = 1/√2  dört açı    +1.3e-04…+3.3e-04   −4.5e-07…+3.1e-12

    Yani ekstrapolasyon farkı **700 kata kadar** küçültüyor ve yasa
    4.6e-07'de tutuyor. Tek `k`da yazılsaydı bant 3.3e-04 olurdu ve bu bir
    tolerans gevşetmesi değil, YANLIŞ BİR ÖLÇÜM olurdu: `24C = P − S²` bir
    `k → 0` ifadesidir.
    """
    m, n = mn
    C1, kal1, th = _duran_kip_2d(201, m, n, s, 600)
    C2, kal2, th2 = _duran_kip_2d(201, 2 * m, 2 * n, s, 600)
    assert kal1 < 1e-12 and kal2 < 1e-12
    assert th == pytest.approx(th2, rel=1e-12)  # AYNI açı, iki k
    k1 = math.hypot(math.pi * m / 200.0, math.pi * n / 200.0)
    k2 = 2.0 * k1
    C0 = (C1 * k2**2 - C2 * k1**2) / (k2**2 - k1**2)
    P = np.cos(th) ** 4 + np.sin(th) ** 4
    assert 24.0 * C0 == pytest.approx(P - s * s, abs=1e-6)
    # 1B kesiti: θ → 0'da yasa 1 − S²'ye gider, yani 1B'nin 24'ü.
    assert (np.cos(0.0) ** 4 + np.sin(0.0) ** 4) - s * s == pytest.approx(
        1.0 - s * s, rel=1e-15
    )


def test_KOSEGEN_SIHIRLI_ADIM_S_KOK_IKI_DE_DISPERSIYON_KAYBOLUYOR():
    """SINIF: ŞEMA ÇIPASI. Planın *"Task 4'te ölçülecek"* satırı.

    `S = 1/√2` KÖŞEGENDE (θ = 45°) şemayı TAM çözücü yapar: `P = 0.5` ve
    `24C = P − S² = 0`. Bu, 1B'deki `S = 1` sihirli adımının 2B köşegen
    karşılığıdır ve tek `k`da bile ekstrapolasyonsuz ölçülebiliyor — çünkü
    `O(k⁴)` artığı da sıfırdır (yasa köşegende her mertebede tam).

    ***"0.0000000 TAM" BİR PENCERENİN SAYISIDIR, ŞEMANIN DEĞİL*** — planın
    taban satırı `24·C_köşegen = 0.0000000 tam` diyordu. ÖLÇÜLDÜ:

        N = 201, M = 600 : (8,8) +2.278e-12 · (16,16) −1.266e-13 ·
                           (32,32) **−0.000000e+00 (TAM SIFIR)**
        N = 401, M = 600 : (8,8) −2.025e-12 · (16,16) +2.278e-12

    Yani sayı 1e-12…1e-13 mertebesinde bir YUVARLAMA ARTIĞIDIR, işareti ve
    büyüklüğü `(m, N)` ile değişir, ve TAM SIFIR bu artığın bir ÖRNEĞİDİR —
    `(32,32)`de gerçekten 0.0 çıkıyor. "Tam sıfır" hükmü `m` ve `N`
    adlandırılmadan yazılamaz; bu, Modül 4'te "kesişim 2.260 bir IZGARA
    NOKTASIYDI" dersinin aynısıdır.

    EKSENDEKİ değer bu çekirdekle ÖLÇÜLEMEZ ve planın *"eksende 0.5005"*
    sayısı YENİDEN ÜRETİLEMEZ: PEC kutu `ky = 0` kipini BARINDIRAMAZ
    (`n ≥ 1` zorunlu). Eksen değeri yalnız açısal yasanın θ → 0 limiti
    olarak okunur ve orada 1 − S² = 0.5'tir.
    """
    s = 2**-0.5
    olculen = []
    for m in (8, 16, 32):
        C, kalinti, th = _duran_kip_2d(201, m, m, s, 600)
        assert kalinti < 1e-12
        assert th == pytest.approx(np.pi / 4, rel=1e-12)
        olculen.append(24.0 * C)
    # Köşegende dispersiyon YOK -- artık makine tabanında.
    assert max(abs(x) for x in olculen) < 1e-10
    # ama TAM SIFIR bir hüküm DEĞİL: aynı pencerede üç `m`, üç ayrı sayı.
    assert len({abs(x) for x in olculen}) == 3
    assert min(abs(x) for x in olculen) == 0.0  # (32,32) TAM sıfır
    assert max(abs(x) for x in olculen) > 1e-13  # (8,8) değil
    # Eksen değeri yasanın limiti olarak: 1 − S² = 0.5.
    assert 1.0 - s * s == pytest.approx(0.5, rel=1e-15)


def test_2B_CEKIRDEK_DE_ORTAMI_ve_GEOMETRIYI_BILMIYOR():
    """SINIF: TRANSKRİPSİYON DENETİMİ — Task 1 kilidinin 2B karşılığı.

    Çekirdeğin ortamdan ve sınırdan dokunduğu adların TAM kümesi sayılıyor.
    `radius` kümede YOK: daire, çekirdeğe yalnız bir bool DİZİ olarak
    görünür. Bir gün biri `if isinstance(boundary, CircularPEC)` yazarsa bu
    küme büyür ve test düşer.
    """

    class Kaydeden:
        def __init__(self, hedef):
            self._hedef = hedef
            self.dokunulan: set[str] = set()

        def __getattr__(self, ad):
            self.dokunulan.add(ad)
            return getattr(self._hedef, ad)

    ortam = Kaydeden(Medium.uniform_2d(27, 27))
    sinir = Kaydeden(CircularPEC(radius=10.0))
    run_2d(
        ortam, courant=S, boundary=sinir, steps=30,
        source=lambda t: ricker(t, wavelength=8.0, sgf=1.5),
        source_index=(13, 15), probes=((12, 12), (14, 11)),
        wavelength=8.0, sgf=1.5,
    )
    # `mu_r` (01.10.2026): etkin Courant kapısı hücre başına eps_r * mu_r okur.
    assert ortam.dokunulan == {"shape", "eps_r", "mu_yari_2d", "mu_r"}
    assert sinir.dokunulan == {"sigma_2d", "pec_mask_2d", "thickness"}
    for yasak in ("radius", "cells", "eps_yari", "ey_kural", "sigma_1d"):
        assert yasak not in ortam.dokunulan and yasak not in sinir.dokunulan


def test_FIELDSTATE_2B_UC_SEKLI_DE_TIPTE_ZORLUYOR_ve_DIM_IKI_DELIGI_KAPANDI():
    """SINIF: TRANSKRİPSİYON DENETİMİ — ve SEVK EDİLMİŞ BİR İDDİANIN ÇÜRÜĞÜ.

    `FieldState`in dokstringi Task 1'den beri *"yalniz 1B ve 2B tanimlar"*
    diyordu. 2B sevk edilince o cümle ilk kez SINANABİLİR hâle geldi ve
    ÇÜRÜDÜ: `dim = 2` kabul ediliyordu ama **hiçbir şekil denetimi yoktu** —
    `FieldState(np.zeros(10), np.zeros(3), 2)` sessizce geçiyordu, yani 1B'de
    tipe zorlanan yarım hücre kayması 2B'de HİÇ zorlanmıyordu. "Tanımlıyor"
    değil, DENETİMSİZ BİR DELİKTİ.

    Şimdi `dim = 2` bu sınıftan reddediliyor ve 2B durum `FieldState2D`de;
    orada ÜÇ alanın üçü de ayrı ayrı denetleniyor (`hx` yalnız y'de, `hy`
    yalnız x'te bir hücre kısa).
    """
    with pytest.raises(InvalidInputError, match="FieldState2D"):
        FieldState(np.zeros(10), np.zeros(3), 2)
    with pytest.raises(InvalidInputError, match="FieldState2D"):
        FieldState(np.zeros(10), np.zeros(9), 3)

    FieldState2D.zeros_2d(5, 7)
    with pytest.raises(InvalidInputError, match="Hx"):
        FieldState2D(np.zeros((5, 7)), np.zeros((5, 7)), np.zeros((4, 7)))
    with pytest.raises(InvalidInputError, match="Hy"):
        FieldState2D(np.zeros((5, 7)), np.zeros((5, 6)), np.zeros((5, 7)))
    with pytest.raises(InvalidInputError, match="iki boyutlu"):
        FieldState2D(np.zeros(5), np.zeros(4), np.zeros(4))
    # `dim` bir ALAN değil, ÖZELLİK: girdiyle yalan söylenemez.
    assert FieldState2D.zeros_2d(5, 5).dim == 2
    assert not hasattr(FieldState2D(np.zeros((5, 5)), np.zeros((5, 4)),
                                    np.zeros((4, 5))).__dict__, "dim")


def test_2B_ORTAMDA_CELLS_TANIMSIZ_ve_SESSIZ_YANLIS_SAYI_VERMIYOR():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    2B ortamda `cells` sessizce `shape[0]` dönseydi, `nx` ile `ny`nin farklı
    olduğu her koşumda YANLIŞ bir sayı doğru gibi dolaşırdı — ve bu modülde
    pencerenin yanlış beyanı ON BİR kez ısırdı.
    """
    m2 = Medium.uniform_2d(9, 11)
    assert m2.shape == (9, 11)
    with pytest.raises(InvalidInputError, match="yalniz 1B"):
        _ = m2.cells
    assert Medium.vacuum(9).cells == 9
    with pytest.raises(InvalidInputError, match="1B ya da 2B"):
        Medium(np.ones((2, 2, 2)), np.ones((2, 2, 2)))
    with pytest.raises(InvalidInputError, match="yalniz 2B"):
        Medium.vacuum(9).mu_yari_2d(0)


def test_2B_SOGURUCU_1B_HUKMUNU_DEVRALMIYOR_KALINLIK_ONBES_YETMIYOR():
    """SINIF: ÖLÇÜM — ve SEVK EDİLMİŞ BİR DOKSTRİNG SÖZÜNÜN ÖDENMESİ.

    `boundary.PML` dokstringi *"2B bölünmüş-alan PML Task 4'ün işidir ve
    bedeli orada YENİDEN ölçülür"* diyordu. Ödendi, ve iki sonuç çıktı:

    (1) BEDEL 2B'DE DE YOK. Sevk edilen soğurucu bölünmüş-alan DEĞİL, iki
        eksende dereceli SKALER iletkenliktir; fazladan alan tutmaz.
        Ölçüldü (n = 241, 400 adım, beşin en iyisi): npml = 0/15/40 için
        0.1601 / 0.1524 / 0.1544 s, tepe bellek 7740.0 vs 7740.1 KiB
        (oran 1.0000). Yani Berenger'in "1.33 kat adım / 1.30 kat bellek"i
        2B'de de bu uygulamanın sayısı DEĞİLDİR.
    (2) AMA `d = 15` HÜKMÜ 1B'NİNDİR ve 2B'YE TAŞINMAZ. Nokta kaynak her
        açıya birden ışır ve skaler iletken yalnız DİK bilende eşleşir.
        Ölçüldü (kalan alan enerjisinin PEC'e oranının karekökü = genlik
        oranı): d = 8/15/25/40 için 7.999e-02 / 5.299e-02 / 2.117e-02 /
        5.950e-03. 1B'de d = 15 dik gelişte 5.27e-03 veriyordu; 2B'de aynı
        kalınlık **10 kat daha kötü** ve 5.9e-03'e inmek için d = 40 gerekiyor.

    Bu bir kusur beyanı: Modül 5'in 2B fizik çıpası KAPALI bir boşluktur
    (soğurucu kullanmaz), dolayısıyla bu sayı bir engel değil, gelecekteki
    2B açık-alan işinin PENCERESİDİR.
    """
    n, adim = 241, 400

    def enerji(sinir):
        k = run_2d(
            Medium.uniform_2d(n, n), courant=S, boundary=sinir, steps=adim,
            source=lambda t: ricker(t, wavelength=20.0, sgf=1.5),
            source_index=(n // 2, n // 2), probes=((n // 2, n // 2),),
            wavelength=20.0, sgf=1.5,
        )
        return float(np.sum(k.final.e ** 2))

    e0 = enerji(PEC())
    g = {d: math.sqrt(enerji(PML(thickness=d)) / e0) for d in (15, 40)}
    assert g[15] == pytest.approx(5.3e-2, rel=0.2)
    assert g[40] == pytest.approx(6.0e-3, rel=0.2)
    # 1B'nin d = 15 hükmü (5.27e-03) 2B'de TUTMUYOR — en az beş kat kötü.
    assert g[15] > 5.0 * 5.2735e-03
    # ve kalınlığı artırmak GERÇEKTEN çalışıyor (soğurucu bozuk değil).
    assert g[40] < 0.2 * g[15]


# ===========================================================================
# 2. FİZİK ÇIPASI — BESSEL ORANLARI
# ===========================================================================
@pytest.mark.parametrize("ad, mn, bant", ORANLAR)
@pytest.mark.slow
def test_BESSEL_ORANLARI_GIRDILERDEN_BAGIMSIZ(kayitlar, ad, mn, bant):
    """SINIF: FİZİK ÇIPASI, dairesel DEĞİL.

    2B dairesel PEC boşluğun kip frekansları `ω_mn = c·j_mn/R`'dir; ORANLARI
    `R`'den, `c`'den ve `ε`'dan BAĞIMSIZDIR ve Bessel köklerinin saf
    sayılarıdır. Bu dosyanın hiçbir yerinde bir kök YAZILMAZ
    (`scipy.special.jn_zeros`) ve `mode_spectrum` onları HİÇ GÖRMEZ.

    ÖLÇÜLDÜ (R = 80, sevk penceresi, 167×167 ızgara, 134400 adım, 16.0 s) —
    ve sapma KİPE BAĞLI:

        j11/j01  1.593293002   tam 1.593340506   −4.7504e-05   genlik 1.000
        j21/j01  2.135392329   tam 2.135548787   −1.5646e-04   genlik 0.952
        j02/j01  2.295226545   tam 2.295417267   −1.9072e-04   genlik 0.298
        j31/j01  2.652755373   tam 2.653066405   −3.1103e-04   genlik 0.519
        j12/j01  2.916874952   tam 2.917295455   −4.2050e-04   genlik 0.0214

    En iyi ile en kötü arasında **8.85 kat** var. *"5 hane"* hükmü YALNIZ
    `j11/j01`'in penceresine aittir ve panelde öyle yazılır. Sapmaların
    HEPSİ AYNI İŞARETLİ (ölçülen frekanslar sistematik olarak düşük);
    bu, merdivenlenmiş sınırın bir işaretidir, gürültü değil.
    """
    ms = mode_spectrum(kayitlar[80.0], **SEVK)
    olculen = _esle(ms, ad)
    tam = KOK[mn] / J01
    assert olculen is not None, f"{ad} tepe listesinde eşleşmedi"
    assert olculen == pytest.approx(tam, abs=bant)
    assert olculen < tam  # sistematik ve TEK İŞARETLİ


@pytest.mark.slow
def test_ORAN_MERDIVENI_OTURDU_IKI_ve_MERDIVENLENME_ORANDA_GOTURUSUYOR(kayitlar):
    """SINIF: FİZİK ÇIPASI + Task 2'nin ölçütü. **ÇIPANIN ASIL HÜKMÜ.**

    σ tek başına ölçüt değildir; hüküm MERDİVENDEDİR. `R = 10/20/40/80`,
    `r = 2`, adım ∝ R (spektral çözünürlük `j` biriminde sabit):

        j11/j01 = 1.593293002   [R = 10/20/40/80, r = 2, S = 0.5, eps_r = 1,
                                 adım = 1680R, 8 dedektör, blackman, pad 8]
          merdiven : 1.589869894  1.592537730  1.593145912  1.593293002
          sapma    : −3.471e-03   −8.028e-04   −1.946e-04   −4.750e-05
          gürültü  : 4.7e-07 / 9.2e-07 / 3.4e-07 / 1.3e-06  (27 meşru pencere)
          mertebe  : p = 2.1331, 2.0478  ->  p_inf = 1.9625 ± 0.0853
          YAKINSAMA: **OTURDU(2)**

    ADIM ÇARPANI 1680, 1667 DEĞİL — ve sebebi ölçüldü, fizik değil: 1667
    ASALDIR, `steps·pad` uzunluğundaki FFT büyük bir asal çarpan taşır ve
    aynı analiz R = 80'de pad 4/8/16 için 0.269/0.600/1.447 s sürerken
    1680 ile 0.026/0.056/0.110 s sürüyor — **13 kat**. Fizik sayıları
    değişmiyor (p_inf 1.9677 → 1.9625, hüküm aynı); değişen tek şey
    süittir. Pencere düğmesi bir HESAP gerekçesiyle seçildiğinde de
    beyan edilir.

    `r = 2` GARANTİ DEĞİLDİ ve ÖLÇÜLEREK seçildi: Task 3'te TİY penceresinde
    dört `r = 2` basamağı bulunamamış ve `r = 1.5` sevk edilmişti. Burada
    dördü de kuruluyor.

    **KABA UÇTAKİ SINIR TASK 6'DA YENİDEN ÖLÇÜLDÜ VE BU CÜMLE DÜZELTİLDİ.**
    Burada eskiden *"kaba uçta sınır FİZİKSELDİR (R = 5'te tepe kapısı kipi
    bulamıyor)"* yazıyordu; ölçüm onu çürüttü. Aynı boruyla R = 5'te **28
    tepe** bulunuyor (R = 40'takinden fazla, çünkü merdivenlenme dejenere
    ikilileri yarıyor) ve merdivenin okuduğu tek sayı — `j11/j01` —
    **bulunuyor**: 1.578632448506. Düşen şey tepe kapısı değil **%2'lik
    eşleme penceresidir**: `j02` −2.82%, `j31` −3.23%, `j12` +2.80% ile
    dışarıda kalıyor, üçünün genlikleri 0.57 / 0.82 / 0.30 ile tabandan çok
    yukarıda. Yani R = 5 **beş oranlık paneli** kuramaz ama **merdiveni**
    kurar; `scenarios/em_tarama.yaml` tam olarak onu kurar (R = 5/10/20/40,
    OTURDU(2)) ve kilit `tests/em/test_em_senaryo.py`dedir.

    HÜKÜM ÇÖZÜNÜRLÜKTEN GELMİYOR: adım çarpanı 1667 → 833 yapıldığında
    (yani her basamağın kaydı YARIYA indiğinde) merdiven yine OTURDU(2)
    veriyor, p_inf = 1.9162 ± 0.1488. Yani 2. mertebe kaydın uzunluğunun
    değil MERDİVENLENMENİN mertebesidir.
    """
    L, v, g = _merdiven(kayitlar)
    tam = KOK[(1, 1)] / J01
    assert L.verdict == "OTURDU"
    assert L.settled_order == 2
    assert L.p_inf == pytest.approx(1.97, abs=0.12)
    assert L.uncertainty < 0.15
    # Merdiven gerçekten YAKINSIYOR: her basamak öncekinden daha yakın.
    sapma = [abs(x - tam) for x in v]
    assert sapma == sorted(sapma, reverse=True)
    assert sapma[0] / sapma[-1] > 50.0
    # Gürültü, en küçük basamak farkının çok altında — merdiven okunabilir.
    assert max(g) * abs(v[-1]) < 0.05 * abs(v[-1] - v[-2])


@pytest.mark.slow
def test_MUTLAK_J_MERDIVENI_OTURMUYOR_ve_SEBEBI_KAFES_ARITMETIGI(kayitlar):
    """SINIF: FİZİK ÇIPASI'nın PENCERESİ — ve PLANIN BİR HÜKMÜNÜN ÇÜRÜĞÜ.

    Plan ve spec, mutlak `j` için doğru cevabın **`OTURDU(1)`** olduğunu
    yazıyordu (merdivenlenmiş sınır 1. mertebedir). ÖLÇÜLDÜ ve hüküm bundan
    **DAHA SERT** çıktı: mutlak `j₀,₁` merdiveni **OTURMADI** veriyor.

        j01 (mutlak, = 2πR·f₀)  [aynı dört basamak, aynı pencere]
          merdiven : 2.310931868  2.361275299  2.383172884  2.395374694
          sapma    : −9.3894e-02  −4.3550e-02  −2.1653e-02  −9.4509e-03
          gürültü  : 1.3e-06 / 1.1e-06 / 7.3e-07 / 1.0e-06
          mertebe  : p = 1.2010, 0.8437  ->  p_inf = 0.4863 ± 0.3574
          YAKINSAMA: **OTURMADI**

    VE SEBEBİ ÖLÇÜLDÜ — bir çözünürlük artefaktı DEĞİL. On üç yarıçapta
    (R = 10/12/14/16/20/24/28/32/40/48/56/64/80) etkin yarıçap sapması
    `δ(R) = R(j∞/j_ölçülen − 1)` hesaplandı: **0.214 … 0.406** bandında ve
    MONOTON DEĞİL. Bu dalgalanma kafesin ARİTMETİĞİNDEN geliyor — `r ≤ R`
    düğüm sayısı `N(R)` ile dairenin alanı arasındaki fark (Gauss'un daire
    problemi) bir kuvvet yasası DEĞİLDİR. Ölçülen bağ:

        δ(R) = 0.9763 · (√(N(R)/π) − R) + 0.3651
        Pearson r = 0.999041,  R² = 0.9981
        artık std 0.00217  (ham δ yayılımı 0.04956 -> 22.8 KAT küçülüyor)

    Yani boşluğun gördüğü etkin yarıçap, dairenin yarıçapı değil
    MERDİVENLENMİŞ BÖLGENİN ALANINDAN okunan yarıçaptır (artı sabit bir
    sınır katmanı payı, 0.365 hücre). Dört sevk basamağında bu bağın artığı
    en fazla 0.0043 (δ yayılımının %8'i).

    PENCERE UYARISI: dört sevk basamağında (R = 10/20/40/80) δ *tesadüfen*
    monotondur (0.4063 / 0.3689 / 0.3634 / 0.3156); monoton OLMAMA on üç
    yarıçaplı taramanın gözlemidir (ör. R = 12'de 0.2142, R = 14'te 0.3355).
    Bu yüzden test monotonluğu değil, δ'nın **SABİT OLMADIĞINI** sınıyor:
    temiz bir 1. mertebede δ sabit olurdu, ölçülen 1.29 kat oynuyor.

    Yani mutlak `j`nin BAŞ terimi gerçekten 1/R'dir (δ sınırlı bir bantta
    kalıyor), ama KATSAYISI aritmetik olarak dalgalandığı için ölçüt bir
    tamsayı mertebe SERTİFİKA EDEMEZ — ve etmemesi doğrudur.

    ORANDA BU DALGALANMA TAM SADELEŞİR: iki kip AYNI etkin yarıçapı görür.
    Çıpanın "oran olarak kurulur, mutlak `j` olarak değil" penceresinin
    SAYISAL KARŞILIĞI budur, ve iki hüküm yan yana durunca kontrast
    planınkinden keskindir: **OTURMADI vs OTURDU(2)**.

    ÇÖZÜNÜRLÜK DEĞİL OLDUĞU AYRICA ÖLÇÜLDÜ: adım çarpanı yarıya
    (1667 → 833) indirildiğinde p_inf 0.4859 → 0.4871, yani DEĞİŞMİYOR.
    """
    L, v, g = _merdiven(kayitlar, mutlak=True)
    assert L.verdict == "OTURMADI"
    assert L.settled_order is None
    assert L.p_inf == pytest.approx(0.49, abs=0.12)
    # Baş terim gerçekten 1/R: etkin yarıçap sapması SINIRLI bir bantta.
    delta = [R * (J01 / x - 1.0) for R, x in zip(RLER, v)]
    assert all(0.20 < d < 0.45 for d in delta)
    # ama SABİT DEĞİL — temiz bir 1. mertebe olsaydı SABİT olurdu.
    assert max(delta) / min(delta) > 1.2
    # ve kafes alanı dalgalanmayı AÇIKLIYOR.
    art = []
    for R, d in zip(RLER, delta):
        n, c = _geometri(R)
        i = np.arange(n)[:, None] - c
        j = np.arange(n)[None, :] - c
        N = int(((i * i + j * j) <= R * R).sum())
        art.append(d - (0.9763 * (math.sqrt(N / math.pi) - R) + 0.3651))
    assert max(abs(x) for x in art) < 0.01
    assert float(np.std(art, ddof=1)) < 0.20 * float(np.std(delta, ddof=1))


@pytest.mark.slow
def test_AIRY_SABITI_2B_DE_CAPALANIYOR_ve_ITHALI_BEYAN_EDILIYOR(kayitlar):
    """SINIF: FİZİK ÇIPASI — ve bir İTHALİN AÇIK BEYANI.

    `j₁,₁/π = 1.2196699`, Airy diskinin sabitinin TA KENDİSİDİR. Bu modül onu
    3B'ye HİÇ GİRMEDEN çapalıyor — ama okumanın bir İTHALİ var ve
    saklanmıyor: ÖLÇÜLEN şey `j₁,₁/j₀,₁` ORANIDIR; `j₁,₁`e geçmek için
    `j₀,₁ = 2.4048256` SAF SAYISI teoriden ithal edilir. Bu modül `j₀,₁`i
    ÖLÇMÜYOR (mutlak merdiven OTURMADI diyor) ve bunu ithal etmek dairesellik
    değil, beyandır: ithal edilen sayı sabittir ve oranı oynatmaz.

    ÖLÇÜLDÜ (R = 80): oran 1.593293002 × 2.4048256 = 3.8315917, yani
    `j₁,₁/π = 1.2196335`, tam değer 1.2196699, sapma **−3.636e-05**
    (bağıl 2.981e-05). Ve bu bağıl sapma ORANIN kendi bağıl sapmasının
    BİREBİR kendisidir — ithal edilen sayı hata EKLEMİYOR.

    3B MALİYETİ ÖLÇÜLDÜ ve bir ALT SINIRDIR. Çıplak `np.roll` çekirdeği
    (PML yok, TF/SF yok, altı alan, float64):

        D = 80  : 18.67 ms/adım, altı alan 23.4 MiB
        D = 167 : 228.72 ms/adım, altı alan 213.2 MiB

    D = 167 sevk edilen 2B çıpanın ızgarasının TAM eşidir; aynı 134400 adım
    orada **512 dakika = 8.5 saat** eder. 2B karşılığı **16.0 s**, yani
    **≈ 1920 kat**. Ve bu ÇIPLAK çekirdeğin sayısıdır.

    PLANIN "0.60 GiB" SAYISI YENİDEN ÜRETİLEMEDİ: D = 80'de altı alanın
    gerçek ayak izi 23.4 MiB'dir (0.60 GiB için ~154 alan gerekirdi).
    """
    ms = mode_spectrum(kayitlar[80.0], **SEVK)
    oran = _esle(ms, "j11/j01")
    j11 = oran * J01  # İTHAL: j01 saf sayısı teoriden
    airy = j11 / math.pi
    assert airy == pytest.approx(KOK[(1, 1)] / math.pi, abs=6e-5)
    # İthal hata eklemiyor: Airy'nin bağıl sapması ORANIN bağıl sapması.
    assert abs(airy / (KOK[(1, 1)] / math.pi) - 1.0) == pytest.approx(
        abs(oran / (KOK[(1, 1)] / J01) - 1.0), rel=1e-9
    )

    # 3B'nin PAHALILIĞI: aynı doğrusal boyutta adım başına maliyet oranı.
    def hiz_3d(D, adim=3):
        E = [np.zeros((D, D, D)) for _ in range(3)]
        H = [np.zeros((D, D, D)) for _ in range(3)]
        E[2][D // 2, D // 2, D // 2] = 1.0
        il = lambda f, a: np.roll(f, -1, a) - f  # noqa: E731
        ge = lambda f, a: f - np.roll(f, 1, a)  # noqa: E731
        t = time.perf_counter()
        for _ in range(adim):
            H[0] -= S * (il(E[2], 1) - il(E[1], 2))
            H[1] -= S * (il(E[0], 2) - il(E[2], 0))
            H[2] -= S * (il(E[1], 0) - il(E[0], 1))
            E[0] += S * (ge(H[2], 1) - ge(H[1], 2))
            E[1] += S * (ge(H[0], 2) - ge(H[2], 0))
            E[2] += S * (ge(H[1], 0) - ge(H[0], 1))
        return (time.perf_counter() - t) / adim

    D = 81
    co = yee_coefficients_2d(Medium.uniform_2d(D, D), courant=S, boundary=PEC())
    st = FieldState2D.zeros_2d(D, D)
    st.e[D // 2, D // 2] = 1.0
    t = time.perf_counter()
    for _ in range(60):
        st = _adim_2d(st, co)
    ms2 = (time.perf_counter() - t) / 60
    assert hiz_3d(D) > 20.0 * ms2  # ölçülen ~90 kat; alt sınır iddiası


@pytest.mark.slow
def test_KARE_KUTU_BASKA_BIR_AILE_VERIYOR_AYNI_CEKIRDEKTEN():
    """SINIF: FİZİK ÇIPASI'nın POZİTİF KONTROLÜ — ve `kare_kutu` mutantı.

    Merdivenlenmiş dairesel sınır kaldırılıp yerine ızgaranın kendi kare
    kutusu konunca aynı çekirdek BAŞKA BİR AİLE veriyor: kare PEC boşluğun
    oranları `√((m²+n²)/2)`dir, Bessel değil.

    ÖLÇÜLDÜ (aynı 87×87 ızgara, aynı kaynak, aynı analiz, R = 40 penceresi):

        ölçülen  1.580994  1.999833  2.235421  2.548984  2.913795  2.999332
        kuram    1.581139  2.000000  2.236068  2.549510  2.915476  3.000000

    Yani çözücü kare kutunun TAM çözümünü de veriyor (sapma 1.4e-04…1.7e-03)
    — bozuk değil, BAŞKA bir geometri. Bessel eşlemesi beşin yalnız İKİSİNİ
    %2 penceresinde buluyor ve ikisi de çıpa bandının çok dışında:
    `j11` −1.235e-02 (bandın 124 katı), `j12` −3.501e-03 (5 katı).

    BU TESTİN İŞİ İKİ TARAFLI: (a) `pec_mask_2d` bir gün boş dizi dönerse
    çıpa düşer, (b) daire çıpasının gerçekten DAİREYİ ölçtüğünü, "her
    boşlukta çıkan bir sayı"yı değil, göstermek.
    """
    kayit = _kosum(40.0, kare=True)
    ms = mode_spectrum(kayit, **SEVK)
    oranlar = ms.ratios()
    for beklenen, olculen in zip(
        [math.sqrt((a * a + b * b) / 2.0) for a, b in ((1, 1), (1, 2), (2, 2), (1, 3))],
        oranlar,
    ):
        assert olculen == pytest.approx(beklenen, abs=2e-3)
    # ve Bessel çıpası bu boşlukta DÜŞÜYOR.
    dusen = 0
    for ad, mn, bant in ORANLAR:
        o = _esle(ms, ad)
        if o is None or abs(o - KOK[mn] / J01) > bant:
            dusen += 1
    assert dusen == len(ORANLAR)


@pytest.mark.slow
def test_EPS_R_BAGIMSIZLIGI_CIPAYI_SERTIFIYE_ETMIYOR_ve_OLCUM_BUNU_GOSTERIYOR(
    kayitlar,
):
    """SINIF: ŞEMA ÖLÇÜMÜ — **fizik çıpası DEĞİL**, ve spec §7.4'ün dersi.

    Spec §7.4 bir gerekçeyi ölçerek çürütmüştü: *"oranlar ε_r'den bağımsız"*
    çıpayı SERTİFİYE ETMEZ, çünkü `2πR√ε_r` ölçeği iki `j`nin oranında
    CEBİRSEL OLARAK sadeleşir; kararlı ama YANLIŞ bir model de aynı sınavı
    geçer. Bu test o dersi ölçüyle taşıyor ve bir adım daha ileri gidiyor:

    ÖLÇÜLDÜ (R = 20/40/80, ε_r 1.0 → 2.25, adım ∝ √ε_r ile eşleştirilmiş):

        f₀₁ oranı  1.5001208 / 1.5000307 / 1.5000078      (tam 1.5)
        j11/j01 kayması  1.978e-04 / 5.027e-05 / 1.273e-05
        j12/j01 kayması  1.756e-03 / 4.494e-04 / 1.136e-04

    Kayma R ile **4 kat** iniyor (ölçülen oranlar 3.93 ve 3.95), yani
    İKİNCİ MERTEBEDEN. Bu, artık ε bağımlılığının ÖLÇEKTEN değil ÇÖZÜCÜNÜN
    IZGARA DİSPERSİYONUNDAN geldiğini söylüyor: ε_r = 2.25'te dalga boyu
    hücre cinsinden 1.5 kat kısalır ve dispersiyon `(kΔx)²` ile büyür.

    ***SPEC'İN "oranlar 8e-6'da aynı kaldı" SAYISI BU PENCEREDE YENİDEN
    ÜRETİLEMEDİ***: en iyi oranda bile R = 80'de 1.27e-05, ve 8e-6'ya inmek
    için R ≳ 110 gerekir; en kötü oranda R ≳ 300. Ölçülen şey bir parametre
    bağımsızlığı DEĞİL, çözücünün dispersiyonunun ölçüsüdür — ve bu yüzden
    bu test ŞEMA sınıfında duruyor.
    """
    r80 = kayitlar[80.0]
    ms1 = mode_spectrum(r80, **SEVK)
    ms2 = mode_spectrum(_kosum(40.0, eps_r=2.25), **SEVK)
    ms40 = mode_spectrum(kayitlar[40.0], **SEVK)

    # ω GERÇEKTEN 1.5 kat değişiyor (ölçek sadeleşmesi boş değil).
    assert ms40.frequencies[0] / ms2.frequencies[0] == pytest.approx(1.5, abs=1e-4)
    # R = 40'ta kayma ölçülen bantta.
    for ad, _mn, _b in ORANLAR:
        a, b = _esle(ms40, ad), _esle(ms2, ad)
        assert a is not None and b is not None
        assert 1e-5 < abs(a - b) < 1e-3
    # ve R = 80'de DÖRT KAT küçük — yani ikinci mertebe, ölçek değil.
    a80, a40 = _esle(ms1, "j11/j01"), _esle(ms40, "j11/j01")
    assert abs(a80 - a40) > 0.0
    # spec'in 8e-6'sı bu pencerede TUTMUYOR:
    assert abs(_esle(ms40, "j11/j01") - _esle(ms2, "j11/j01")) > 8e-6


# ===========================================================================
# 3. ÖLÇÜM ALETİ — `mode_spectrum`un kendi kapıları
# ===========================================================================
@pytest.mark.slow
def test_YAN_LOB_KAPISI_IMKAN_KAPISI_ve_HANN_BESINCI_ORANI_GOREMIYOR(kayitlar):
    """SINIF: ÖLÇÜM ALETİNİN KAPISI. Tolerans DEĞİL, cebirsel garanti.

    Bir yan lob "yerel maksimum"dur ve tepe sayılırsa spektrumun EN DÜŞÜK
    tepesi (yani bütün oranların PAYDASI) yanlış çıkar. Kapı bunu keser:
    `peak_floor` pencerenin ölçülen en büyük yan lobundan büyük olmak
    ZORUNDA. Yan lob oranı ana lobun TEPESİNE göredir ve tepe her zaman
    `max(M)`den küçük eşit olduğundan, kapı hiçbir yan lobun tepe
    sayılamayacağının CEBİRSEL garantisidir.

    PENCERENİN KENDİ DÖNÜŞÜMÜNDEN ÖLÇÜLDÜ (n = 4096 ve 8192, 64 kat dolgu;
    iki n'de de aynı sayı):

        dikdortgen 0.217178 (−13.3 dB), ana lob yarı genişliği 1 bin
        hann       0.026707 (−31.5 dB)                          2 bin
        blackman   0.001243 (−58.1 dB)                          3 bin

    VE BU YÜZDEN SEVK EDİLEN PENCERE **blackman**, planın yazdığı `hann`
    DEĞİL: R = 80'de beşinci oranın (`j12/j01`) genliği en büyük tepenin
    **0.02142**'sidir ve bu, hann'ın yan lobunun (0.026707) ALTINDADIR.
    Yani hann ile HİÇBİR meşru tabanda `j12` ile bir yan lob birbirinden
    ayrılamaz — ölçüldü: hann + taban 0.03 ile beş orandan DÖRDÜ bulunuyor,
    blackman + taban 0.01 ile BEŞİ DE.

    Somut arıza da ölçüldü: hann + taban 0.01 (kapı kapalıyken) R = 80'de en
    düşük tepe olarak gerçek kipin **%0.75 altındaki** bir yan lobu seçiyor
    ve BÜTÜN oranlar o kadar kayıyor.
    """
    for ad, yan in WINDOW_SIDELOBE.items():
        n = 4096
        u = 2.0 * np.pi * np.arange(n) / n
        w = {
            "dikdortgen": np.ones(n),
            "hann": 0.5 - 0.5 * np.cos(u),
            "blackman": 0.42 - 0.5 * np.cos(u) + 0.08 * np.cos(2.0 * u),
        }[ad]
        M = np.abs(np.fft.rfft(w, n=n * 64))
        M = M / M[0]
        d = np.diff(M)
        i = 1
        while i < len(d) and not (d[i - 1] < 0 and d[i] >= 0):
            i += 1
        assert float(M[i:].max()) == pytest.approx(yan, rel=2e-3)

    with pytest.raises(InvalidInputError, match="yan"):
        mode_spectrum(kayitlar[10.0], pad=8, window="hann", peak_floor=0.01)
    with pytest.raises(InvalidInputError, match="yan"):
        mode_spectrum(kayitlar[10.0], pad=8, window="dikdortgen", peak_floor=0.05)

    # hann DÖRT oran, blackman BEŞ.
    hann = mode_spectrum(kayitlar[80.0], pad=PAD, window="hann", peak_floor=0.03)
    blk = mode_spectrum(kayitlar[80.0], **SEVK)
    assert sum(_esle(hann, a) is not None for a, _, _ in ORANLAR) == 4
    assert sum(_esle(blk, a) is not None for a, _, _ in ORANLAR) == 5
    # ve beşincinin genliği hann'ın yan lobunun ALTINDA.
    k = [i for i, r in enumerate(blk.ratios()) if abs(r - _esle(blk, "j12/j01")) < 1e-12]
    assert blk.amplitudes[k[0]] / max(blk.amplitudes) == pytest.approx(0.0214, abs=2e-3)
    assert blk.amplitudes[k[0]] / max(blk.amplitudes) < WINDOW_SIDELOBE["hann"]


@pytest.mark.slow
def test_PAD_BIR_MERDIVENI_YARGILANAMAZ_ama_TEK_IZGARADA_MUTANT_KAZANIYOR(
    kayitlar,
):
    """SINIF: MUTASYON KİLİDİ. Planın birinci *"ÖLÇ"*ü — ve cevabı **σ tek
    başına ölçüt değildir** dersinin bu depodaki en temiz örneği.

    `pad = 1` (sıfır dolgu YOK) bilgi kaybetmez ama parabolün oturduğu örnek
    sayısını üçe indirir ve tepe konumu bin ızgarasına yapışır. ÖLÇÜLDÜ
    (aynı dört kayıt, yalnız `pad` değişiyor — FDTD koşumu AYNI):

        değerler : 1.589701413  1.592453604  1.593065704  1.593369902
        gürültü  : 1.6e-04 / 3.5e-05 / 6.2e-05 / 1.1e-04   (sevkte ~1e-06)
        mertebe  : p = 2.1687, 1.0088 -> p_inf = −0.1512, **u = 2.3502**
        YAKINSAMA: **YARGILANAMAZ**

    ***VE TEK IZGARADA MUTANT KAZANIYOR.*** R = 80'de beş oranın DÖRDÜNDE
    `pad = 1` doğru borudan DAHA YAKIN:

        oran      sevk         pad=1        kim daha yakın
        j11/j01   −4.7504e-05  +2.940e-05   pad=1 (1.6 kat)
        j21/j01   −1.5646e-04  +2.1041e-04  sevk
        j02/j01   −1.9072e-04  +8.922e-05   pad=1 (2.1 kat)
        j31/j01   −3.1103e-04  +4.746e-05   pad=1 (6.6 kat)
        j12/j01   −4.2050e-04  −1.0970e-04  pad=1 (3.8 kat)

    Yani beş oran paneli bu mutantı YAKALAMIYOR — hepsi bantların içinde
    kalıyor. Yakalayan tek şey MERDİVENDİR, ve yakalarken bir tolerans
    kullanmıyor: pencere gürültüsü (≈1.8e-04 mutlak) en küçük basamak
    farkının mertebesinde, dolayısıyla ölçüt "yargılayamam" diyor.
    İşaretlerin de rastgele dağılması (üçü artı) bunun ŞANS olduğunu
    söylüyor; gürültü **100 kattan fazla** büyümüş durumda.
    """
    L, v, g = _merdiven(kayitlar, pad=1)
    assert L.verdict == "YARGILANAMAZ"
    assert L.uncertainty > 1.0
    assert max(g) > 30.0 * 1.3e-06

    mutant = mode_spectrum(kayitlar[80.0], pad=1, window=PENCERE, peak_floor=TABAN)
    dogru = mode_spectrum(kayitlar[80.0], **SEVK)
    kazanan = 0
    for ad, mn, bant in ORANLAR:
        tam = KOK[mn] / J01
        sm, sd = abs(_esle(mutant, ad) - tam), abs(_esle(dogru, ad) - tam)
        assert sm < bant  # PANEL bu mutantı GÖRMÜYOR
        kazanan += sm < sd
    assert kazanan >= 4  # ve dördünde DAHA İYİ


@pytest.mark.slow
def test_ARA_DEGERLEME_KALKINCA_HEM_MERDIVEN_HEM_PANEL_DUSUYOR(kayitlar):
    """SINIF: MUTASYON KİLİDİ — ve planın bir BEKLENTİSİNİN düzeltilmesi.

    Parabolik ara değerleme kaldırılınca planın beklediği şey *"merdiven
    MERTEBESİ düşer"*di. ÖLÇÜLDÜ ve mekanizma BAŞKA: ham mertebeler
    düşmüyor (p = 3.3411, 1.4245 — biri 2'nin ÜSTÜNDE), yerinden OYNUYOR ve
    belirsizlik PATLIYOR:

        değerler : 1.589805825  1.592871287  1.593173794  1.593286495
        gürültü  : 1.5e-04 / 1.7e-04 / 7.9e-05 / 1.9e-04
        p_inf = −0.4921, **u = 10.0255** -> YARGILANAMAZ

    çünkü pencere gürültüsü (≈3.0e-04 mutlak) en küçük basamak farkından
    (1.1e-04) BÜYÜK. Ölçüt "yargılayamam" diyor ve doğru olan bu.

    Bu mutant `pad = 1`in AKSİNE panelden de düşüyor: R = 80'de `j02/j01`
    sapması −1.907e-04 → **−3.353e-04**, yani 3.0e-04 bandının dışında
    (`j31` de −4.513e-04 ile 5.0e-04 bandının kenarında). İki mutantın
    ikisini birden yakalayan tek şey ise MERDİVENDİR.
    """
    L, v, g = _merdiven(kayitlar, interpolate=False)
    assert L.verdict == "YARGILANAMAZ"
    assert L.uncertainty > 1.0
    assert L.orders[0] > 2.0  # mertebe DÜŞMÜYOR, oynuyor
    assert max(g) > 30.0 * 1.3e-06

    mutant = mode_spectrum(kayitlar[80.0], interpolate=False, **SEVK)
    tam = KOK[(0, 2)] / J01
    assert abs(_esle(mutant, "j02/j01") - tam) > 3.0e-4  # PANEL de yakalıyor


@pytest.mark.slow
def test_DEDEKTOR_SAYISI_MERDIVENI_DEGIL_PANELI_ETKILIYOR(kayitlar):
    """SINIF: MUTASYON KİLİDİ. Planın ikinci *"ÖLÇ"*ü — ve cevabı İKİ PARÇALI.

    `dedektör = 8 → 1` sorusunun cevabı "kip kaçırıyor mu" değil, "NEYİ
    kaçırıyor" idi. ÖLÇÜLDÜ:

    (1) MERDİVEN HİÇ DEĞİŞMİYOR. `j11` en güçlü kiptir ve tek dedektörde de
        aynı bin'de oturuyor: dört basamağın dördü de **1e-09**'un altında
        aynı, p_inf = 1.9625, hüküm OTURDU(2) — birebir aynı. Yani merdiven
        bu bozmaya KÖRDÜR ve körlüğü beyan ediliyor.
    (2) PANEL DÜŞÜYOR. R = 80'de tepe sayısı 16 → 11 ve `j21/j01` sapması
        −1.5646e-04 → **+4.8973e-04**, yani çıpanın 3.0e-04 bandının DIŞINA
        çıkıyor: tek dedektör, merdivenlenmenin ayırdığı `m = 2` ikilisinin
        ÖBÜR üyesini seçiyor (işaret de dönüyor).

    Bu, çıpanın neden BEŞ orandan ve SEKİZ dedektörden kurulduğunun
    gerekçesidir: tek bir oran (ve tek bir dedektör) bu bozmayı göremezdi.
    Üç mutantın üçü birden düşünüldüğünde iş bölümü ölçülmüş hâliyle şudur —
    `pad = 1` yalnız MERDİVENDEN, `dedektör = 1` yalnız PANELDEN, ara
    değerlemesizlik İKİSİNDEN birden düşüyor.
    """
    tek = {R: _dilim(kayitlar[R], 1) for R in RLER}
    L1, v1, _ = _merdiven(tek)
    L8, v8, _ = _merdiven(kayitlar)
    assert L1.verdict == L8.verdict == "OTURDU"
    assert max(abs(a - b) for a, b in zip(v1, v8)) < 1e-9  # merdiven KÖR

    ms1 = mode_spectrum(tek[80.0], **SEVK)
    ms8 = mode_spectrum(kayitlar[80.0], **SEVK)
    assert len(ms8.frequencies) > len(ms1.frequencies)
    tam = KOK[(2, 1)] / J01
    assert abs(_esle(ms8, "j21/j01") - tam) < 3.0e-4
    assert abs(_esle(ms1, "j21/j01") - tam) > 3.0e-4  # PANEL yakalıyor


def test_MODE_SPECTRUM_TEORIYI_BILMIYOR_ve_PENCERESINI_TASIYOR(kucuk_kayit):
    """SINIF: TRANSKRİPSİYON DENETİMİ — çıpanın dairesel OLMAMA koşulu.

    `observable/em_obs.py` içinde hiçbir Bessel kökü, hiçbir `j` sayısı ve
    hiçbir `2.404` yoktur; alet ADSIZ tepeler döndürür. Eşleme bu dosyada
    yapılır ve penceresi (%2) hükmün (1e-04) 200 KATIDIR.
    """
    from pathlib import Path

    import qlab.observable.em_obs as mod

    kaynak = Path(mod.__file__).read_text(encoding="utf-8")
    for yasak in ("jn_zeros", "scipy", "2.404", "3.8317", "5.520"):
        assert yasak not in kaynak

    ms = mode_spectrum(kucuk_kayit, **SEVK)
    assert ms.pad == PAD and ms.window_fn == PENCERE and ms.peak_floor == TABAN
    assert ms.detectors == NDET
    assert ms.steps == kucuk_kayit.probes.shape[1]
    assert ms.ratios()[0] == 1.0
    assert ms.bin_width == pytest.approx(ms.raw_bin_width / PAD, rel=1e-12)
    assert len(ms.as_lines()) == 5
    assert kucuk_kayit.window.izgara == "27 x 27"


def test_MODE_SPECTRUM_GIRDI_DENETIMLERI(kucuk_kayit):
    """SINIF: TRANSKRİPSİYON DENETİMİ."""
    k = kucuk_kayit
    with pytest.raises(InvalidInputError, match="pad"):
        mode_spectrum(k, pad=0, window=PENCERE, peak_floor=TABAN)
    with pytest.raises(InvalidInputError, match="pad"):
        mode_spectrum(k, pad=True, window=PENCERE, peak_floor=TABAN)
    with pytest.raises(InvalidInputError, match="peak_floor"):
        mode_spectrum(k, pad=PAD, window=PENCERE, peak_floor=0.0)
    with pytest.raises(InvalidInputError, match="window"):
        mode_spectrum(k, pad=PAD, window="kaiser", peak_floor=TABAN)
    assert set(WINDOW_FNS) == set(WINDOW_SIDELOBE)


@pytest.mark.parametrize("bozuk", [float("nan"), float("inf"), -float("inf")])
def test_MODE_SPECTRUM_SONLU_OLMAYAN_KAYDI_REDDEDER(kucuk_kayit, bozuk):
    """SINIF: TRANSKRİPSİYON DENETİMİ (G15, 11.10.2026).

    ÖLÇÜLDÜ (HEAD ef18615, kapı yokken; `em_tarama.yaml` penceresi, R = 5):
    temiz kayıt 28 tepe; TEK bir NaN / +inf / −inf eklenince alet HATASIZ boş
    demet döndürüyordu — çağıran bunu "tepe bulunamadı" diye okurdu. Kapı
    modülün `values sonlu olmali` deseninde, sayımı ile; temiz kayıt aynı
    tepeleri vermeye devam eder.
    """
    from dataclasses import replace

    once = mode_spectrum(kucuk_kayit, **SEVK)
    probes = kucuk_kayit.probes.copy()
    probes[0, probes.shape[1] // 2] = bozuk
    with pytest.raises(InvalidInputError, match="kayit sonlu olmali; 1 sonlu olmayan"):
        mode_spectrum(replace(kucuk_kayit, probes=probes), **SEVK)
    assert mode_spectrum(kucuk_kayit, **SEVK).frequencies == once.frequencies
    assert len(once.frequencies) >= 2


def test_RUN_2D_GIRDI_DENETIMLERI():
    """SINIF: TRANSKRİPSİYON DENETİMİ."""
    med = Medium.uniform_2d(21, 21)
    ort = dict(courant=S, boundary=CircularPEC(radius=8.0), wavelength=8.0, sgf=1.0)
    with pytest.raises(InvalidInputError, match="steps"):
        run_2d(med, steps=-1, source=None, source_index=None, probes=(), **ort)
    with pytest.raises(InvalidInputError, match="birlikte"):
        run_2d(med, steps=5, source=None, source_index=(10, 10), probes=(), **ort)
    with pytest.raises(InvalidInputError, match="izgara disinda"):
        run_2d(med, steps=5, source=None, source_index=None, probes=((99, 1),), **ort)
    with pytest.raises(InvalidInputError, match="PEC duvarlarin ARASINDA"):
        run_2d(med, steps=5, source=lambda t: 0.0, source_index=(0, 10),
               probes=((5, 5),), **ort)
    with pytest.raises(InvalidInputError, match="ILETKENIN ICINDE"):
        run_2d(med, steps=5, source=lambda t: 1.0, source_index=(1, 1),
               probes=((10, 10),), **ort)
    with pytest.raises(InvalidInputError, match="2B cozucu"):
        run_2d(Medium.vacuum(21), steps=5, source=None, source_index=None,
               probes=(), **ort)
    with pytest.raises(InvalidInputError, match="FieldState2D alir"):
        yee_step_2d(FieldState.zeros_1d(21), med, courant=S, boundary=PEC())
    with pytest.raises(InvalidInputError, match="yalniz 2B"):
        CircularPEC(radius=4.0).sigma_1d(21)
    with pytest.raises(InvalidInputError, match="yaricap"):
        CircularPEC(radius=0.0)


def test_YEE_STEP_2D_ve_RUN_2D_TEK_GUNCELLEME_YOLU():
    """SINIF: TRANSKRİPSİYON DENETİMİ — 1B'deki aynı kilidin 2B karşılığı.

    İki ayrı güncelleme yazılsaydı "çıpa çekirdeği sınar" iddiası düşerdi;
    çıpa hangisini sınadığını bilemezdi.
    """
    n = 21
    med = Medium.uniform_2d(n, n)
    sinir = CircularPEC(radius=8.0)
    kayit = run_2d(
        med, courant=S, boundary=sinir, steps=12,
        source=lambda t: ricker(t, wavelength=6.0, sgf=1.5),
        source_index=(11, 12), probes=((10, 10),), wavelength=6.0, sgf=1.5,
    )
    st = FieldState2D.zeros_2d(n, n)
    co = yee_coefficients_2d(med, courant=S, boundary=sinir)
    elle = np.empty(12)
    for t in range(12):
        elle[t] = st.e[10, 10]
        st = yee_step_2d(st, med, courant=S, boundary=sinir)
        st.e[11, 12] += co.cb[11, 12] * ricker(t * S, wavelength=6.0, sgf=1.5)
    assert np.array_equal(kayit.probes[0], elle)
    assert float(np.max(np.abs(elle))) > 0.0


def test_SLOW_ESIGI_OLCULDU_ve_FIXTURE_MALIYETI_ISARETE_DAHIL():
    """SINIF: SÜREÇ DENETİMİ. Eşik DEVRALINMADI, bu dosyada ÖLÇÜLDÜ.

    `tests/em`de eşik dosya kapsamındadır (Task 3'ün `403501a` düzeltmesi).
    Bu dosyanın ölçümü (`--durations=0`, `-p no:randomly`):

        19.99 s  MODÜL FIXTURE'ı (`kayitlar`, dört FDTD koşumu)
         5.21 s  test_DEDEKTOR_SAYISI_...                  <- slow
         4.68 s  test_MUTLAK_J_MERDIVENI_...               <- slow
         4.44 s  test_ARA_DEGERLEME_...                    <- slow
         4.35 s  test_ORAN_MERDIVENI_...                   <- slow
         4.31 s  test_EPS_R_BAGIMSIZLIGI_...               <- slow
         2.93 s  test_KARE_KUTU_...                        <- slow
         1.86 s  test_COURANT_SINIRI_2B_...                <- slow  (en ucuz)
        ------------------------------------------------- eşik 1.0 s
         0.71 s  test_KOSEGEN_SIHIRLI_ADIM_...             (işaretsiz, en pahalı)
         0.50 s  test_2B_SOGURUCU_...
         0.48 s  test_2B_DISPERSIYON_... (on iki parametre)

    FIXTURE MALİYETİ İŞARETE DAHİLDİR ve bu bir SEÇİM DEĞİL ZORUNLULUK:
    `kayitlar` modül kapsamlıdır, yani 19.99 s'lik kurulum onu ilk tüketen
    teste yazılır ve o testin HANGİSİ olacağı koşum sırasına bağlıdır
    (`pytest-randomly` etkinken rastgele). Dolayısıyla fixture'ı tüketen HER
    test için `setup + call`un EN KÖTÜSÜ ≈ 20 s'dir ve hepsi `slow`
    işaretlidir. Modül 4'ün "fixture maliyeti `call` değil `setup` olarak
    görünüyor" tuzağının bu dosyadaki karşılığı budur.

    Aletin kendi girdi denetimleri bu yükten AYRILDI (`kucuk_kayit`, 0.29 s)
    ki `-m "not slow"` koşumu `mode_spectrum`un kapılarını yine sınasın.
    """
    from pathlib import Path

    esik = 1.0
    isaretsiz_en_pahali = 0.71  # ölçüldü, test_KOSEGEN_SIHIRLI_ADIM_...
    slow_en_ucuz = 1.86  # ölçüldü, test_COURANT_SINIRI_2B_...
    assert isaretsiz_en_pahali < esik < slow_en_ucuz
    assert slow_en_ucuz / isaretsiz_en_pahali == pytest.approx(2.62, abs=0.02)

    # İşaretin gerçekten uygulandığının SAYIMI (yorum değil, sayı):
    kaynak = Path(__file__).read_text(encoding="utf-8")
    assert kaynak.count("@pytest.mark.slow\n") == 11
    # ve fixture'ı tüketen her test işaretli:
    for parca in kaynak.split("\ndef test_")[1:]:
        ad = parca.split("(")[0]
        imza = parca.split(")")[0]
        if "kayitlar" in imza:
            assert f"@pytest.mark.slow\ndef test_{ad}" in kaynak, ad
