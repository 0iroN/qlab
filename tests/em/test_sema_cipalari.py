"""DÖRT ŞEMA ÇIPASI — ve hepsi "ŞEMA ÇIPASI" diye etiketli.

Bu dosyadaki hiçbir sayı FİZİK ÇIPASI DEĞİLDİR. Hepsi Yee şemasının kendi
özelliğidir: Courant sınırı, yakınsama mertebesi, dispersiyon katsayısı,
sihirli adım. Bunlar "çözücü doğru mu" sorusunu yanıtlar; "doğa böyle mi"
sorusunu DEĞİL. Modül 5'in fizik çıpaları (Bessel oranları, Rayleigh üssü,
tek yarık, TİY faz farkı) Task 3–5'te gelir ve o dosyalarda etiketlenir.

Bir şema çıpasını fizik çıpası diye sunmak, bu depoda üç kez düzeltilmiş
hatanın kılık değiştirmiş hâlidir (spec §3.4).

PENCERE — hepsi burada adlandırılıyor, çünkü bu modülde tohumun karşılığı
pencerenin tam beyanıdır (bu depoda hükmün pencerenin dışına taşınması ON
KEZ ölçüldü):

- 1B ölçümlerin tamamı SEVK EDİLEN çekirdekten (`yee._adim`) geçer.
- Duran-kip ölçümü: PEC duvarlı ızgarada `sin(k i)`, `k = m·pi/(N-1)`.
  Bu, Dirichlet sınır koşullu ikinci farkın TAM özvektörüdür, dolayısıyla
  seri tek frekanslıdır ve `omega` üç terimli yinelemeden okunur —
  KAPALI FORM HİÇ KULLANILMAZ.
- d = 2 ve d = 3 basamakları için sevk edilmiş çekirdek YOK (2B Task 4'te
  gelir, 3B kapsam dışıdır). O iki basamak bu dosyadaki çıplak periyodik
  çekirdekle ölçülür ve bu, testin kendi beyanıdır.

`slow` İŞARETİ ÖLÇÜLEREK KONDU, devralınmadı. Modül 4'te ölçülen "call >=
2.0 s" eşiği burada TEK BAŞINA yanlış sınıflandırırdı; ayrıca oradaki
`setup`/`call` tuzağı bu dosyada YOKTUR ve bu da ölçüldü: `tests/em`de
HİÇBİR `setup` 0.005 s'yi geçmiyor (fixture kullanılmıyor), yani
setup + call = call. Ölçülen call süreleri:
    1.65s  test_S_EN_HIZLI_DALGA_HIZIYLA_TANIMLI...      <- slow
    0.68s  test_PML_KALINLIGI_ONBES_ONIKI_DEGIL...       <- slow
    0.38s  test_COURANT_ESIGI_UC_BOYUTTA_DA_OLCULUYOR[3]
    0.36s  test_COURANT_SINIRI_1B_DE_OLCULUYOR...
    (kalan 41 test toplam 0.6 s)
Sınır deponun kendi kalibrasyonundan: `hf/test_density_observable.py` 0.35 s
ve `hf/test_rhf_sikismasi.py` 0.46 s AÇIKÇA "slow DEĞİL" diye yazılı; mevcut
slow işaretli en ucuz gövde 0.69 s. Bu dosya o boşluğu kullanıyor.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.em.boundary import PEC, PML, LossyLayer
from qlab.engine.em.grid import FieldState, courant_limit
from qlab.engine.em.medium import Medium
from qlab.engine.em.source import ricker
from qlab.engine.em.yee import (
    YeeCoefficients,
    _adim,
    run_1d,
    yee_coefficients,
)


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def _vakum_katsayilari(cells: int, courant: float) -> YeeCoefficients:
    """Vakum katsayıları, KARARLILIK KAPISINI ATLAYARAK.

    Kapı S > 1'i reddeder; ölçülecek olan şey tam olarak O REDDİN NEREDE
    olduğudur, dolayısıyla eşik ölçümü kapıyı atlamak ZORUNDADIR. Koşan
    güncelleme yine sevk edilen `_adim`dir — ölçülen şema sevk edilen
    şemadır, yalnız kapı devre dışıdır.
    """
    return YeeCoefficients(
        ca=np.ones(cells),
        cb=np.full(cells, courant),
        da=np.ones(cells - 1),
        db=np.full(cells - 1, courant),
        dim=1,
        courant=courant,
    )


def _kararli_mi_1d(S: float, L: int, steps: int, seed: int) -> bool:
    """SEVK EDİLEN çekirdek, PEC duvar, rastgele başlangıç alanı."""
    rng = np.random.default_rng(seed)
    st = FieldState(rng.standard_normal(L), rng.standard_normal(L - 1), 1)
    co = _vakum_katsayilari(L, S)
    n0 = float(np.max(np.abs(st.e)))
    for _ in range(steps):
        st = _adim(st, co)
        m = float(np.max(np.abs(st.e)))
        if not np.isfinite(m) or m > 1e12 * n0:
            return False
    return True


def _kararli_mi_2d(S: float, L: int, steps: int, seed: int) -> bool:
    """ÇIPLAK periyodik 2B — sevk edilmiş karşılığı YOK (Task 4)."""
    rng = np.random.default_rng(seed)
    ez = rng.standard_normal((L, L))
    hx = rng.standard_normal((L, L))
    hy = rng.standard_normal((L, L))
    n0 = float(np.max(np.abs(ez)))
    for _ in range(steps):
        hx -= S * (np.roll(ez, -1, 1) - ez)
        hy += S * (np.roll(ez, -1, 0) - ez)
        ez += S * ((hy - np.roll(hy, 1, 0)) - (hx - np.roll(hx, 1, 1)))
        m = float(np.max(np.abs(ez)))
        if not np.isfinite(m) or m > 1e12 * n0:
            return False
    return True


def _kararli_mi_3d(S: float, L: int, steps: int, seed: int) -> bool:
    """ÇIPLAK periyodik 3B — sevk edilmiş karşılığı YOK ve OLMAYACAK."""
    rng = np.random.default_rng(seed)
    E = [rng.standard_normal((L, L, L)) for _ in range(3)]
    H = [rng.standard_normal((L, L, L)) for _ in range(3)]
    n0 = max(float(np.max(np.abs(e))) for e in E)

    def ileri(f, ax):
        return np.roll(f, -1, ax) - f

    def geri(f, ax):
        return f - np.roll(f, 1, ax)

    for _ in range(steps):
        H[0] -= S * (ileri(E[2], 1) - ileri(E[1], 2))
        H[1] -= S * (ileri(E[0], 2) - ileri(E[2], 0))
        H[2] -= S * (ileri(E[1], 0) - ileri(E[0], 1))
        E[0] += S * (geri(H[2], 1) - geri(H[1], 2))
        E[1] += S * (geri(H[0], 2) - geri(H[2], 0))
        E[2] += S * (geri(H[1], 0) - geri(H[0], 1))
        m = max(float(np.max(np.abs(e))) for e in E)
        if not np.isfinite(m) or m > 1e12 * n0:
            return False
    return True


def _esik(kararli, lo: float, hi: float, tol: float) -> float:
    """İkili arama: en büyük kararlı S."""
    assert kararli(lo), "alt sınır zaten kararsız — pencere bozuk"
    assert not kararli(hi), "üst sınır hâlâ kararlı — pencere bozuk"
    while hi - lo > tol:
        orta = 0.5 * (lo + hi)
        if kararli(orta):
            lo = orta
        else:
            hi = orta
    return 0.5 * (lo + hi)


def _duran_kip(N: int, m: int, S: float, M: int):
    """PEC duvarlı ızgarada m. duran kip; `c = cos(omega dt)` ve `omega`.

    `omega` ÜÇ TERİMLİ YİNELEMEDEN okunur: tek frekanslı bir seri
    `s[n+1] + s[n-1] = 2 cos(omega dt) s[n]` sağlar. Yee'nin kapalı form
    dispersiyon bağıntısı bu dosyada HİÇBİR YERDE kullanılmaz — kullanılsaydı
    ölçüm daireselleşirdi.

    KATSAYILAR SEVK EDİLEN `yee_coefficients`TEN GELİR, elle kurulmaz. İlk
    yazımda elle kuruluyordu ve MUTASYON bunu yakaladı: `cb`ye %2'lik bir
    çarpan enjekte edildiğinde dört şema çıpasının HİÇBİRİ görmüyordu, çünkü
    ölçüm o fonksiyonu hiç çağırmıyordu. Kapıyı atlamak yalnız Courant EŞİĞİ
    ölçümünde zorunludur (orada S > 1 gerekir); burada değil.
    """
    k = np.pi * m / (N - 1)
    proj = np.sin(k * np.arange(N))
    st = FieldState(proj.copy(), np.zeros(N - 1), 1)
    co = yee_coefficients(Medium.vacuum(N), courant=S, boundary=PEC())
    s = np.empty(M)
    for n in range(M):
        s[n] = float(np.dot(st.e, proj))
        st = _adim(st, co)
    y, x = s[2:] + s[:-2], 2.0 * s[1:-1]
    c = float(np.dot(x, y) / np.dot(x, x))
    kalinti = float(np.max(np.abs(y - c * x)) / np.max(np.abs(y)))
    omega = float(np.arccos(np.clip(c, -1.0, 1.0)) / S)
    return c, omega, k, kalinti


# ---------------------------------------------------------------------------
# ÇIPA 1 — COURANT
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim, beklenen", [(1, 1.0), (2, 2**-0.5), (3, 3**-0.5)])
def test_COURANT_SINIRI_BOYUTLA_KOK_D_OLCEKLENIYOR(dim, beklenen):
    """SINIF: ŞEMA ÇIPASI — çözücüyü doğrular, fiziği DEĞİL.

    `S·sqrt(d) = 1` saf bir sayıdır ve GİRDİ DEĞİLDİR: şema kararlı olmaktan
    çıkar. Ama `d` AYRIKTIR (1, 2, 3) ve sürekli bir düğmeye çevrilemez —
    spec §3.5'in ikincil ayırt edici ölçütü.

    Bu test iki şey yapar ve ikisi de gereklidir:
      (a) TRANSKRİPSİYON KİLİDİ — `courant_limit` gerçekten `1/sqrt(d)` mi;
      (b) ÖLÇÜM — o sayı şemanın kendi eşiği mi (ayrı testte, aşağıda).

    NOT: `courant_limit` d = 3'ü KABUL EDER ama çözücü REDDEDER. Sebep
    ölçüldü ve dokstringde yazılı: şema SAYISI bedava, 3B KOŞUM değil
    (D = 80'de bellek 156 kat ve bu oran kararlı; süre oranı ise üç ölçümde
    258 / 443 / 896 kat oynuyor — sayısı yok, alt sınırı birkaç yüz kat).
    """
    assert courant_limit(dim) == pytest.approx(beklenen, rel=1e-12)


def test_COURANT_SINIRI_1B_DE_OLCULUYOR_ve_ASIMIN_IKI_PARCASI_VAR():
    """SINIF: ŞEMA ÇIPASI. Eşik GİRİLMEZ, ikili aramayla ÖLÇÜLÜR.

    PENCERE: sevk edilen `_adim`, PEC duvar, L = 200, 4000 adım, tohum 3,
    ikili arama toleransı 1e-7.

    ÖLÇÜLEN EŞİK 1'İN AZ ÜSTÜNDEDİR (+3.8e-05) ve plan taslağı bunu tek bir
    şeye bağlıyordu: "sonlu koşum artefaktıdır, 20000 adımda +2.1e-7'ye
    iniyor". ÖLÇÜLDÜ — bu YANLIŞ; aşım İKİ parçadır:

    (a) SONLU IZGARA payı, yalnız PEC duvarda. PEC'te ayrık kipler
        k_m = m·pi/(L-1)'dir ve ızgara k = pi'ye HİÇ ULAŞMAZ; sonlu ızgaranın
        TAM sınırı bu yüzden 1/cos(pi/(2(L-1)))'dir. Bu test o kapalı formu
        doğrudan sınıyor.
    (b) SONLU KOŞUM payı, L'den bağımsız; periyodik ızgarada tek kalan pay
        budur (4000 adımda +5.99e-06, 20000 adımda +2.12e-07 ölçüldü).

    Toleransı gevşetmek yerine MEKANİZMA ölçüldüğü için sınır dar
    tutulabiliyor: ölçülen eşik ile kapalı form arasındaki fark 2.3e-07
    (L = 200, 20000 adım) ve bu, (b)'nin kendisidir.
    """
    L, adim = 200, 4000
    S = _esik(lambda s: _kararli_mi_1d(s, L, adim, 3), 0.5, 1.5, 1e-7)
    izgara_siniri = 1.0 / np.cos(np.pi / (2 * (L - 1)))

    # (a) eşik ham 1'in ÜSTÜNDE — ve az.
    assert 1.0 < S < 1.0 + 1e-4

    # (a) kapalı form: sonlu ızgaranın TAM sınırı. Kalan fark = koşum payı.
    assert S == pytest.approx(izgara_siniri, abs=1e-5)
    assert S > izgara_siniri  # koşum payı POZİTİFTİR, işareti de bir iddiadır

    # (b) ızgara payı L^-2 ile küçülür: L dört katına çıkınca ~16 kat.
    L4 = 4 * L
    izgara_siniri_4 = 1.0 / np.cos(np.pi / (2 * (L4 - 1)))
    oran = (izgara_siniri - 1.0) / (izgara_siniri_4 - 1.0)
    assert oran == pytest.approx(16.0, rel=0.02)


@pytest.mark.parametrize(
    "dim, kararli, L, adim",
    [(1, _kararli_mi_1d, 64, 400), (2, _kararli_mi_2d, 32, 400),
     (3, _kararli_mi_3d, 16, 300)],
)
def test_COURANT_ESIGI_UC_BOYUTTA_DA_OLCULUYOR(dim, kararli, L, adim):
    """SINIF: ŞEMA ÇIPASI — BOYUT MERDİVENİ, spec §3.6'nın BİRİNCİL bağı.

    Üç basamağın üçü de burada ÖLÇÜLÜYOR (ikili arama, tolerans 1e-5).
    d = 1 sevk edilen çekirdekle, d = 2 ve d = 3 bu dosyadaki çıplak
    PERİYODİK çekirdeklerle — ve fark önemlidir: periyodik ızgarada
    (çift L) k = pi TAM olarak vardır, dolayısıyla sonlu-ızgara payı YOKTUR
    ve `S·sqrt(d)` doğrudan 1'e oturur. PEC'li d = 1 basamağında ise
    1/cos(pi/(2(L-1))) payı vardır ve L = 64'te bu +3.1e-04'tür — bu yüzden
    d = 1'in beklenen değeri 1 değil, O KAPALI FORMDUR.

    Bu testin işi `1/sqrt(d)`yi ÖLÇMEKTİR; `courant_limit`in o sayıyı
    döndürdüğü ayrı testte kilitlenir.
    """
    S = _esik(lambda s: kararli(s, L, adim, 1), 0.4 * courant_limit(dim),
              1.3 * courant_limit(dim), 1e-5)
    if dim == 1:
        beklenen = 1.0 / np.cos(np.pi / (2 * (L - 1)))
    else:
        beklenen = 1.0
    assert S * np.sqrt(dim) == pytest.approx(beklenen, abs=2e-3)
    assert S * np.sqrt(dim) >= beklenen - 1e-9


@pytest.mark.slow
def test_S_EN_HIZLI_DALGA_HIZIYLA_TANIMLI_bu_bir_MODELLEME_SECIMI():
    """SINIF: ŞEMA ÇIPASI'nın BEYAN EDİLEN ÖN KOŞULU.

    Ham kararlılık eşiği ORTAMA BAĞLIDIR — `1/sqrt(d)` hükmü `S`'in en hızlı
    dalga hızıyla normalize edilmesine dayanır ve bu, ölçümden ÖNCE yapılan
    bir SEÇİMDİR, ölçümün sonucu değil.

    ÖLÇÜLDÜ (sevk edilen `_adim`, kapı atlanmış, L = 200, 3000 adım,
    tohum 5, tolerans 1e-7):
        eps_r      1.0          2.25         4.0          9.0
        S_max      1.000040301  1.500060657  2.000081036  3.000121823
        /sqrt(eps) 1.000040301  1.000040438  1.000040518  1.000040608
    Değişmeyen şey S_max/sqrt(eps_r)'dir.

    Bu test o değişmezliği ÖLÇER; sevk edilen kapının `eps_min`i KULLANMADIĞI
    (yani normalize eşiği zorladığı) ayrı testte kilitlenir.
    """
    L, adim = 200, 3000

    def kararli(S, epsr):
        rng = np.random.default_rng(5)
        st = FieldState(rng.standard_normal(L), rng.standard_normal(L - 1), 1)
        co = YeeCoefficients(
            ca=np.ones(L), cb=np.full(L, S / epsr),
            da=np.ones(L - 1), db=np.full(L - 1, S), dim=1, courant=S,
        )
        n0 = float(np.max(np.abs(st.e)))
        for _ in range(adim):
            st = _adim(st, co)
            m = float(np.max(np.abs(st.e)))
            if not np.isfinite(m) or m > 1e12 * n0:
                return False
        return True

    olculen = {}
    for epsr in (1.0, 2.25, 4.0, 9.0):
        kok = np.sqrt(epsr)
        olculen[epsr] = _esik(
            lambda s: kararli(s, epsr), 0.5 * kok, 1.4 * kok, 1e-7
        )
    normalize = np.array([olculen[e] / np.sqrt(e) for e in olculen])
    # HAM eşik ortama bağlı — dört değer birbirinden UZAK.
    ham = np.array(list(olculen.values()))
    assert ham.max() / ham.min() == pytest.approx(3.0, rel=1e-3)
    # NORMALİZE eşik değişmiyor — dördü de aynı sayı.
    assert normalize.max() - normalize.min() < 1e-6
    assert normalize.mean() == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# ÇIPA 2 — YAKINSAMA MERTEBESİ
# ---------------------------------------------------------------------------
def test_YEE_YAKINSAMA_MERTEBESI_IKI_yani_HATA_ORANI_DORT():
    """SINIF: ŞEMA ÇIPASI. Saf sayı DÖRT'tür (mertebe 2 <=> oran 4).

    PENCERE: sevk edilen `_adim`, PEC duvarlı duran kip, m = 8 SABİT,
    N = 201 -> 401 -> 801 -> 1601 -> 3201 (yani `dx` yarılanıyor), 600 adım,
    S = 0.5. Ölçülen büyüklük `|v_p/c - 1|`.

    ÖLÇÜLEN MERDİVEN (S = 0.5):
        N=201   |v_p/c-1| = 4.936019e-04
        N=401   1.233777e-04   oran 4.000740
        N=801   3.084299e-05   oran 4.000185
        N=1601  7.710659e-06   oran 4.000046
        N=3201  1.927667e-06   oran 3.999995   <- EN YAKIN
        N=6401  4.819398e-07   oran 3.999809   <- GERİ DÖNÜYOR

    "4.000009" YAZILAMAZ: bu penceredeki en yakın değer 3.999995'tir ve
    merdiven ORADAN SONRA 4'ten UZAKLAŞIYOR. Sebep yuvarlamadır ve
    ölçülebilir: `omega`, `arccos(c)` ile okunuyor ve `c -> 1` iken
    `d(arccos)/dc` patlar; N = 6401'de `k·dx = 3.9e-03` ve türev ~5e+02.
    Yani merdivenin sonu bir FİZİK sonucu değil, TAHMİNCİNİN TABANIDIR.

    S = 0.7'de aynı merdiven: 4.002020 / 4.000505 / 4.000125 / 4.000059 /
    3.999614 — iki S de aynı deseni veriyor.
    """
    for S, beklenen_son in ((0.5, 4.000046), (0.7, 4.000125)):
        hatalar = []
        for kat in (1, 2, 4, 8):
            N = 200 * kat + 1
            _, omega, k, kalinti = _duran_kip(N, 8, S, 600)
            assert kalinti < 1e-12, "seri tek frekanslı değil — pencere bozuk"
            hatalar.append(abs(omega / k - 1.0))
        oranlar = [hatalar[i] / hatalar[i + 1] for i in range(len(hatalar) - 1)]
        # Merdiven 4'e YAKINSIYOR: her basamak bir öncekinden 4'e daha yakın.
        for i in range(len(oranlar) - 1):
            assert abs(oranlar[i + 1] - 4.0) < abs(oranlar[i] - 4.0)
        assert oranlar[-1] == pytest.approx(beklenen_son, abs=5e-5)
        assert oranlar[-1] == pytest.approx(4.0, abs=2e-4)


# ---------------------------------------------------------------------------
# ÇIPA 3 — DİSPERSİYON KATSAYISI
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("S", [0.3, 0.5, 0.7, 0.9])
def test_DISPERSIYON_KATSAYISI_SAF_SAYI_YIRMIDORT(S):
    """SINIF: ŞEMA ÇIPASI. Saf sayı 24'tür ve `S`'ten BAĞIMSIZDIR.

    Yee 1B'de `v_p/c - 1 = -C (k dx)^2 + O((k dx)^4)` ve `C = (1-S^2)/24`.
    Sınanan şey `(1-S^2)/C = 24`, yani `S` DÖRT FARKLI değer alırken
    değişmeyen saf sayı.

    PENCERE: sevk edilen `_adim`, PEC duvarlı duran kip, N = 801, 800 adım,
    Richardson `k dx -> 0` ekstrapolasyonu m = 8 ve m = 16 noktalarından.

    m ÇİFTİ PENCERENİN PARÇASIDIR ve ÖLÇÜLDÜ — küçük m'de `arccos`
    yuvarlaması, büyük m'de `O(k^4)` artığı baskın:
        m çifti   2,4        4,8        8,16       16,32      32,64
        S=0.3     24.000326  23.999997  24.000002  24.000000  23.999991
        S=0.5     24.000337  24.000019  24.000000  23.999999  23.999987
        S=0.7     24.000478  24.000014  23.999999  24.000001  24.000019
        S=0.9     23.999671  23.999980  24.000001  24.000009  24.000136
    Seçilen (8,16) sütununda dördü de 24'ten 1.4e-06'dan az sapıyor. Plan
    taslağının "23.999963 … 24.000025" bandı BAŞKA bir m çiftinin bandıdır;
    m adlandırılmadan o bant yazılamaz.
    """
    N, M = 801, 800
    C = {}
    for m in (8, 16):
        _, omega, k, kalinti = _duran_kip(N, m, S, M)
        assert kalinti < 1e-12
        C[m] = (k, -(omega / k - 1.0) / k**2)
    (k1, C1), (k2, C2) = C[8], C[16]
    C0 = (C1 * k2**2 - C2 * k1**2) / (k2**2 - k1**2)
    assert (1.0 - S * S) / C0 == pytest.approx(24.0, abs=1e-5)


# ---------------------------------------------------------------------------
# ÇIPA 4 — SİHİRLİ ADIM
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [4, 16, 64, 128])
def test_SIHIRLI_ADIM_S_BIR_DE_DISPERSIYON_TAM_KAYBOLUYOR(m):
    """SINIF: ŞEMA ÇIPASI. 1B'de `S = 1` şemayı TAM ÇÖZÜCÜ yapar.

    PENCERE: sevk edilen `_adim`, PEC duvarlı duran kip, N = 801, 400 adım,
    S = 1.0 (kararlılık sınırının TAM üstünde).

    "0.000e+00 (m = 128)" YAZILAMAZ. Ölçülen `|v_p/c - 1|`:
        m=4    4.943e-13     m=16   5.562e-14
        m=64   2.887e-15     m=128  1.332e-15
    yani sıfır değil, MAKİNE HASSASİYETİ. Ve artığın kaynağı ŞEMA DEĞİL,
    `arccos` yuvarlamasıdır — bu ölçülerek gösterildi: yuvarlamayı hiç
    kullanmayan ölçüt `|c - cos(k·S)|` (üç terimli yinelemeden gelen ham
    `c`) aynı dört kipte 1.110e-16 / 2.220e-16 / 2.220e-16 / 3.331e-16, yani
    tamamı 1–2 ULP.

    Bu test HER İKİ ölçütü de sınar; sadece `v_p` sınansaydı "m küçüldükçe
    şema bozuluyor" gibi YANLIŞ bir okuma mümkün olurdu (4.9e-13 vs
    1.3e-15 — 370 kat), oysa şema dördünde de TAM.
    """
    N = 801
    c, omega, k, kalinti = _duran_kip(N, m, 1.0, 400)
    assert kalinti < 1e-12
    # Yuvarlamasız ölçüt: ham `c` ile kapalı formun TAM değeri.
    assert abs(c - np.cos(k * 1.0)) < 5e-16
    # arccos üzerinden okunan v_p: makine hassasiyeti, ama m'ye bağlı.
    assert abs(omega / k - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# BEŞİNCİ ÇIPA DEĞİL — SINIR KARARININ YENİDEN ÖLÇÜMÜ
# ---------------------------------------------------------------------------
def _yansima_dik(sinir, d, lamlar=(60, 90, 110, 165, 200)):
    """R(lambda) = |FFT(test - referans)| / |FFT(referans)|, probda.

    FARK YÖNTEMİ: referans koşumda soğurucu YOK ve ızgara çok daha uzundur,
    öyle ki kayıt penceresi içinde prob hiçbir dönüşü görmez. Fark TAM
    OLARAK soğurucudan dönen dalgadır; zaman kapısı yok -> sızıntı yok.
    """
    S, XS, XP, XA, ADIM, LAM = 0.5, 1500, 2200, 2800, 6800, 110.0

    def kosum(cells, snr):
        return run_1d(
            Medium.vacuum(cells), courant=S, boundary=snr, steps=ADIM,
            source=lambda t: ricker(t, wavelength=LAM, sgf=1.0),
            source_index=XS, probes=(XP,), wavelength=LAM, sgf=1.0,
            window_fn="dikdortgen",
        ).probes[0]

    t = kosum(XA + d, sinir)
    r = kosum(2 * XA + 800, PEC())
    F_r, F_d = np.fft.fft(r), np.fft.fft(t - r)
    fr = np.fft.fftfreq(ADIM, d=S)
    return {
        lam: abs(F_d[int(np.argmin(np.abs(fr - 1.0 / lam)))])
        / abs(F_r[int(np.argmin(np.abs(fr - 1.0 / lam)))])
        for lam in lamlar
    }


@pytest.mark.slow
def test_PML_KALINLIGI_ONBES_ONIKI_DEGIL_ve_ONIKI_GERCEKTEN_GECMIYOR():
    """SINIF: ÇEVİRİ DENETİMİ (sınır tasarımının kendi ölçütü).

    Plan taslağı bir zamanlar "12 hücre PML, 30 hücre kayıplı katmanı HER
    açıda geçer" diyordu. Bu ÇÜRÜDÜ ve burada YENİDEN ölçülüyor — dik
    gelişte (theta = 0), sevk edilen kodla.

    PENCERE: S = 0.5, dx = 1, kaynak 1500, prob 2200, soğurucunun iç yüzü
    2800, 6800 adım, Ricker wavelength = 110 (sgf = 1), referans ızgara
    6400 hücre, DİKDÖRTGEN pencere.

    KAPI (pazarlık yok): soğurucu yerine PEC konursa R = 1 çıkmalı. Ölçülen
    |R - 1| en büyük 4.602e-07. Çıkmasaydı alttaki hiçbir sayı okunmazdı.

    ÖLÇÜLEN (lam = 110):
        R(PML,     d = 12) = 8.3162e-03
        R(kayıplı, d = 30) = 7.6132e-03   -> PML 1.0923 kat DAHA KÖTÜ
        R(PML,     d = 15) = 5.2735e-03   -> 0.6927 kat, GEÇİYOR

    Ve hüküm lambda boyunca da kararlı: d = 12'nin oranı 60/90/110/165/200'de
    1.0166 / 1.0550 / 1.0923 / 1.1672 / 1.2006 — BEŞİ DE 1'in üstünde.

    AMA HÜKÜM `m`E BAĞLIDIR ve bunu ne plan ne spec adlandırıyordu; mutasyon
    turundaki "PML profili m = 3 -> m = 1" mutantı sordurdu. Ölçüldü
    (theta = 0, lam = 110, kayıplı d = 30 = 7.6132e-03):
        m = 1 : d=12 5.9197e-03 GEÇER    m = 3 : d=12 8.3162e-03 GEÇMEZ
        m = 2 : d=12 6.5265e-03 GEÇER    m = 4 : d=12 1.0728e-02 GEÇMEZ
    `d = 15` ise DÖRT `m` değerinde de geçiyor. Bu yüzden test hem DEĞERİ
    hem de HÜKMÜ sınıyor: değer `m = 3`e özgüdür, hüküm değildir.

    `LossyLayer` bu yüzden depoda duruyor ve SEVK EDİLMİYOR: kıyas nesnesi
    kodda olmasaydı bu hüküm plan metnine dayanan bir iddia olarak kalırdı.
    """
    kapi = _yansima_dik(PEC(), 0)
    assert max(abs(v - 1.0) for v in kapi.values()) < 1e-6

    kay = _yansima_dik(LossyLayer(thickness=30, yuzler="sag"), 30)
    p12 = _yansima_dik(PML(thickness=12, yuzler="sag"), 12)
    p15 = _yansima_dik(PML(thickness=15, yuzler="sag"), 15)

    assert p12[110] == pytest.approx(8.3162e-03, rel=2e-3)
    assert kay[110] == pytest.approx(7.6132e-03, rel=2e-3)
    assert p15[110] == pytest.approx(5.2735e-03, rel=2e-3)

    for lam in kay:
        assert p12[lam] > kay[lam], f"d=12 lam={lam}'de geçiyor — hüküm değişti"
        assert p15[lam] < kay[lam], f"d=15 lam={lam}'de geçmiyor — hüküm değişti"
    assert PML().thickness == 15
