"""Metrik, `w` formu ve KURAM SEÇİCİ — Modül 6 Task 1.

PENCERE — bu dosyadaki HER sayı aşağıdaki tek pencerede ölçüldü ve başka
hiçbir yerden devralınmadı (bu depoda hükmün pencerenin dışına taşınması ON
BİRDEN fazla kez ölçüldü):

- Birim seçimi `G = c = 1`; `M = 1.0` (aksi yazılmadıkça).
- Yörünge parametreleri `p = a(1-e²)` ve `e`; dönüş noktaları
  `u1 = (1-e)/p` (afel), `u2 = (1+e)/p` (perihel).
- Kareleme `psi` İKAMESİYLE: `u = (u1+u2)/2 + ((u2-u1)/2)·sin(psi)`,
  `psi ∈ [-pi/2, pi/2]`, integrand `1/sqrt(w(u))`. İkame kök tekilliğini
  ANALİTİK olarak kaldırır; `sqrt`ın altında yalnız `w` kalır.
- `scipy.integrate.quad`, `epsabs=1e-14`, `epsrel=1e-13`, `limit=400`.
  ÖLÇÜLDÜ: sonuç bu düğmelerden BAĞIMSIZ — `epsabs` 1.49e-8'den 1e-16'ya
  altı ayarda C bit bit AYNI (18.84955591968662) çıkıyor. Sınırlayıcı şey
  kareleme hatası değil, `dphi - 2pi` farkındaki kayan nokta sadeleşmesi.
  1e-16/1e-15 seçilseydi `IntegrationWarning` çıkardı ve HİÇBİR hane
  kazandırmazdı; o yüzden 1e-14/1e-13 sevk ediliyor.
- Richardson çifti (alt 3e5, üst 1e6), `1/p`'de doğrusal dışdeğerleme.
  ÖLÇÜLDÜ, ve bu çift bir OPTİMUMDUR — dışdeğerleme sonsuza kadar
  iyileşmiyor (GR, e = 0.3, dürüst hane):
      (1e5, 3e5) -> 9.12 · (3e5, 1e6) -> 10.01 · (1e6, 3e6) -> 9.94 ·
      (3e6, 1e7) -> 8.98 · (3e7, 1e8) -> 8.93

BU DOSYADAKİ KARELEME SEVK EDİLMİŞ KOD DEĞİLDİR. Task 2 `geodesic.py`de
gerçek `perihelion_coefficient`ı sevk edecek. Buradaki yirmi satırlık tüketici
tek bir iş için var: `radial_w`in FİZİK KAPSAMASINI boş bırakmamak. Yalnız
`lambda = 0`'da 1 döndüğünü sınayan bir süit, `a = 2*lam*M/3` katsayısını HİÇ
sınamaz — o katsayıyı yarıya indiren bir mutant hayatta kalırdı (ölçüldü:
yarılanmış katsayı C = 9.424777960515788 veriyor, doğrusu 18.84955592153876).

`slow` İŞARETİ YOK ve bu ölçülerek karar verildi: dosyanın tamamı 0.35 s'nin
(deponun "slow DEĞİL" diye açıkça yazılı en pahalı gövdesi) çok altında;
tek bir Richardson çağrısı 10 µs.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning, quad

from qlab.engine.grav.metric import (
    IsotropicPotentials,
    Theory,
    closed_form_coefficient,
    isotropic_potential,
    radial_w,
)
from qlab.errors import InvalidInputError

IKI_PI = 2.0 * math.pi

# Kareleme düğmeleri — dosya dokstringindeki pencere.
EPSABS = 1e-14
EPSREL = 1e-13
LIMIT = 400
RICHARDSON = (3e5, 1e6)


# ---------------------------------------------------------------------------
# Test-içi tüketici: SEVK EDİLMİŞ `radial_w`i çözücü olarak koşturur
# ---------------------------------------------------------------------------
def _donus_noktalari(p: float, e: float) -> tuple[float, float]:
    return (1.0 - e) / p, (1.0 + e) / p


def _dphi_psi(*, theory: Theory, M: float, p: float, e: float) -> float:
    """Bir radyal periyotta süpürülen açı — `psi` ikamesiyle."""
    u1, u2 = _donus_noktalari(p, e)
    orta = 0.5 * (u1 + u2)
    yari = 0.5 * (u2 - u1)

    def integrand(psi: float) -> float:
        u = orta + yari * math.sin(psi)
        return 1.0 / math.sqrt(radial_w(u, u1, u2, theory=theory, M=M))

    deger, _ = quad(
        integrand, -math.pi / 2, math.pi / 2,
        epsabs=EPSABS, epsrel=EPSREL, limit=LIMIT,
    )
    return 2.0 * deger


def _dphi_ham_u(*, theory: Theory, M: float, p: float, e: float) -> float:
    """ELENEN boru: ikamesiz, `u` üzerinden kareleme.

    Sevk edilmiyor; yalnız `e -> 0` kapısının gerekçesini ÖLÇMEK için var
    (aşağıya bak). Kök tekilliği integrandda duruyor, bu yüzden `quad`
    `IntegrationWarning` basar — bu bir kaza değil, ÖLÇÜLEN ŞEYİN TA
    KENDİSİDİR (elenen borunun yakınsamaması), o yüzden burada susturuluyor.
    """
    u1, u2 = _donus_noktalari(p, e)

    def integrand(u: float) -> float:
        s = (u - u1) * (u2 - u) * radial_w(u, u1, u2, theory=theory, M=M)
        return 1.0 / math.sqrt(s) if s > 0.0 else 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        deger, _ = quad(integrand, u1, u2, epsabs=EPSABS, epsrel=EPSREL, limit=LIMIT)
    return 2.0 * deger


def _C_tek_basamak(*, theory: Theory, M: float, p: float, e: float, dphi=_dphi_psi) -> float:
    return (dphi(theory=theory, M=M, p=p, e=e) - IKI_PI) * p / M


def _C_richardson(*, theory: Theory, M: float = 1.0, e: float = 0.3,
                  cift: tuple[float, float] = RICHARDSON) -> float:
    """`C = lim_{p->inf} (dphi - 2pi)·p/M`, `1/p`'de doğrusal dışdeğerleme."""
    alt, ust = cift
    f1 = _C_tek_basamak(theory=theory, M=M, p=alt, e=e)
    f2 = _C_tek_basamak(theory=theory, M=M, p=ust, e=e)
    h1, h2 = 1.0 / alt, 1.0 / ust
    return (f1 * h2 - f2 * h1) / (h2 - h1)


def _K_tek_basamak(*, theory: Theory, M: float, p: float, e: float, dphi=_dphi_psi) -> float:
    """İkinci mertebe katsayısı; `C = 2*pi*lam` TAM değeri çıkarılarak."""
    d = dphi(theory=theory, M=M, p=p, e=e)
    return (d - IKI_PI - IKI_PI * theory.lam * M / p) * (p / M) ** 2


# ---------------------------------------------------------------------------
# Step 1 — TAŞIYICI TEST
# ---------------------------------------------------------------------------
def test_W_FORMU_LAMBDA_SIFIRDA_TAM_BIR_VERIYOR_ve_BU_4_3_UN_ON_KOSULU():
    """SINIF: transkripsiyon denetimi — ama spec §4.3'ün ÖN KOŞULU.

    Spec §4.3 "lambda = 0 koşusu HATA ÇUBUĞUDUR" diyor. Kök formunda yazılmış
    bir kareleme bunu KARŞILAYAMAZ: çekirdek `u3 = 3/(2*lam*M) - u1 - u2`
    formundayken lambda = 0'da tanımsız olur.

    `w(u) = 1 - a*(u1 + u2 + u)`, `a = 2*lam*M/3` formu lambda = 0'da TAM 1
    verir ve AYNI borudan geçer.
    """
    for u in (0.0, 0.3, 0.9):
        # NEWTON NOKTASI (gamma, beta)'DAN KURULUR — `Theory(lam=0.0)` DEĞİL:
        # `lam` bir alan değil TÜRETİLMİŞ bir özellik, ve
        # lam_eff = 2 + 2*(-1) - 0 = 0.
        assert radial_w(u, 0.1, 0.5, theory=Theory(gamma=-1.0, beta=0.0), M=1.0) == 1.0


def test_W_FORMU_NEWTON_NOKTASINDA_HER_M_ve_HER_U_ICIN_TAM_BIR():
    """Yukarıdaki üç nokta bir tesadüf değil: `a = 0` olduğu için `u` ve `M`
    ÇARPANI TAMAMEN DÜŞÜYOR. On beş (u, M) çiftinde ölçüldü."""
    newton = Theory(gamma=-1.0, beta=0.0)
    assert newton.lam == 0.0
    for u in (0.0, 0.3, 0.9, 1e-6, 123.456):
        for M in (1.0, 3.7, 1e6):
            assert radial_w(u, 0.1, 0.5, theory=newton, M=M) == 1.0


def test_KOK_FORMU_AYNI_NOKTADA_COKUYOR_bu_bir_TERCIH_DEGIL():
    """`w` formunun bir üslup tercihi OLMADIĞININ ölçümü.

    Kök formu `w_kok = (2*lam*M/3)·(3/(2*lam*M) - u1 - u2 - u)`; lambda = 0'da
    `3/(2*lam*M)` ifadesi ÖLÇÜLDÜ ve Python float aritmetiğinde
    `ZeroDivisionError` atıyor (nan bile dönmüyor, yani "sessizce yanlış" da
    değil — koşu DURUYOR). Yani spec §4.3'ün hata çubuğu makinesi kök
    formunda ilk çağrıda çöker.
    """
    lam, M = 0.0, 1.0
    with pytest.raises(ZeroDivisionError):
        _ = 3.0 / (2.0 * lam * M)


def test_W_FORMU_GR_DE_KOK_FORMUNDAN_AYIRT_EDILEMIYOR():
    """`w` formu λ = 0'ı KURTARIRKEN λ = 3'te hiçbir şey DEĞİŞTİRMİYOR.

    BU TESTİN İLK HÂLİ NOKTASAL BİT BİT EŞİTLİK İSTİYORDU VE KIRMIZI DÜŞTÜ —
    ölçüm hükmü daralttı, eşik gevşetilmedi. Ölçülen (GR, M = 1,
    p ∈ {1e2, 1e3, 3e5, 1e6} × e ∈ {0.05, 0.3, 0.7, 0.95}, aralıkta 21 nokta,
    336 örnek):

        noktasal bit bit eşit : 248 / 336
        en kötü bağıl fark    : 1.205e-16   (1 ulp = 2.220e-16, yani YARIM ulp)

    Yani iki form noktasal olarak son bitte ayrılabiliyor; AYRILAMADIKLARI yer
    HÜKMÜN kendisi: dışdeğerlenmiş `C` üç dışmerkezlikte de **bit bit aynı**
    (fark 0.000e+00, e ∈ {0.3, 0.7, 0.95}).
    """
    gr = Theory()
    a = 2.0 * gr.lam * 1.0 / 3.0

    def w_kok(u, u1, u2):
        u3 = 3.0 / (2.0 * gr.lam * 1.0) - u1 - u2
        return a * (u3 - u)

    esit = toplam = 0
    enkotu = 0.0
    for p in (1e2, 1e3, 3e5, 1e6):
        for e in (0.05, 0.3, 0.7, 0.95):
            u1, u2 = _donus_noktalari(p, e)
            for k in range(21):
                u = u1 + (u2 - u1) * k / 20.0
                yeni, eski = radial_w(u, u1, u2, theory=gr, M=1.0), w_kok(u, u1, u2)
                toplam += 1
                esit += int(yeni == eski)
                enkotu = max(enkotu, abs(yeni - eski) / abs(eski))
    assert toplam == 336
    assert esit == 248
    assert enkotu < 2.3e-16          # ölçülen 1.205e-16

    # Ve HÜKÜM düzeyinde fark HİÇ YOK:
    def _C_kok(e: float) -> float:
        def dphi(p: float) -> float:
            u1, u2 = _donus_noktalari(p, e)
            orta, yari = 0.5 * (u1 + u2), 0.5 * (u2 - u1)
            deger, _ = quad(
                lambda psi: 1.0 / math.sqrt(w_kok(orta + yari * math.sin(psi), u1, u2)),
                -math.pi / 2, math.pi / 2,
                epsabs=EPSABS, epsrel=EPSREL, limit=LIMIT,
            )
            return 2.0 * deger

        alt, ust = RICHARDSON
        f1, f2 = (dphi(alt) - IKI_PI) * alt, (dphi(ust) - IKI_PI) * ust
        h1, h2 = 1.0 / alt, 1.0 / ust
        return (f1 * h2 - f2 * h1) / (h2 - h1)

    for e in (0.3, 0.7, 0.95):
        assert _C_richardson(theory=gr, e=e) == _C_kok(e)


# ---------------------------------------------------------------------------
# Step 3 — KURAM SEÇİCİ
# ---------------------------------------------------------------------------
def test_THEORY_VARSAYILANI_GR():
    """Varsayılan kuram GR'dir: gamma = beta = 1 <=> lam = 3.

    Bu test bir MUTANT BEKÇİSİDİR ve bunu ölçerek biliyorum: `Theory`nin
    `gamma` varsayılanını 0.5'e çeviren mutant, `gamma`yı AÇIKÇA veren
    testlerin hiçbirinde görünmez (Step 1, Step 4 ve kareleme çıpası
    (gamma, beta)'yı elle verir). Varsayılanı okuyan tek yer burasıdır.
    """
    t = Theory()
    assert (t.gamma, t.beta) == (1.0, 1.0)
    assert t.lam == 3.0
    assert closed_form_coefficient(t) == 6.0 * math.pi


def test_LAM_BIR_ALAN_DEGIL_KURUCU_ARGUMANI_DA_DEGIL():
    """İKİ ALAN AYNI ŞEYİ BELİRLERSE SESSİZ BİR KAZANAN DOĞAR.

    Tek kaynak `(gamma, beta)`dır; `lam` bir `@property`dir. Gerçek
    varsayılanlı bir alan olsaydı ("verildi" ile "verilmedi" ayırt
    edilemeyeceği için) tutarlılık kuralı Step 1'in taşıyıcı testini
    (lam_eff = 0), Step 4'ün beş satırından DÖRDÜNÜ ve Task 2'nin
    `Theory(gamma=g, beta=2+2*g)` enjeksiyonunu öldürürdü.
    """
    with pytest.raises(TypeError):
        Theory(lam=3.0)  # type: ignore[call-arg]
    assert "lam" not in {alan for alan in Theory().__dataclass_fields__}


def test_GAMMA_SIFIR_BETA_BIR_ATMIYOR_bu_bir_BEKCIDIR():
    """Step 4'ün dört parametre satırını ve Task 2'nin enjeksiyonunu koruyan
    bekçi. Bir "tutarlılık kuralı" onları sessizce öldürmeye kalkarsa tam
    buradan görünür."""
    for gamma, beta in ((0.0, 1.0), (1.0, 0.0), (2.0, 1.0), (1.0, 3.0), (-1.0, 0.0)):
        t = Theory(gamma=gamma, beta=beta)
        assert t.lam == 2.0 + 2.0 * gamma - beta
        assert math.isfinite(closed_form_coefficient(t))


# ---------------------------------------------------------------------------
# Step 4 — KAPALI FORM
# ---------------------------------------------------------------------------
PPN_NOKTALARI = [
    (1.0, 1.0, 3.0),   # GR   -> 6*pi
    (0.0, 1.0, 1.0),   #      -> 2*pi
    (1.0, 0.0, 4.0),   #      -> 8*pi
    (2.0, 1.0, 5.0),   #      -> 10*pi
    (1.0, 3.0, 1.0),   #      -> 2*pi
]


@pytest.mark.parametrize("gamma, beta, beklenen_kat", PPN_NOKTALARI)
def test_KAPALI_FORM_2PI_2_2GAMMA_EKSI_BETA(gamma, beta, beklenen_kat):
    """SINIF: TRANSKRİPSİYON KİLİDİ — ve planın "FİZİK ÇIPASI" etiketi
    Task 1'de YANLIŞTIR.

    Bu gövde `2*pi*(2+2*gamma-beta)`yı çağırıp `2*pi*kat` ile karşılaştırır;
    ortada çözülen bir model YOKTUR. Deponun kendi ölçütü — *"simülasyon bu
    sayı hakkında YANLIŞ olabilir miydi?"* — burada HAYIR yanıtını verir.
    Fizik kapsamasını veren gövde bir alttaki karelemedir.

    ÖLÇÜLDÜ: beş noktanın beşinde de eşitlik BİT BİTtir (`==`), yaklaşık
    değil; `rel` bir güvenlik payıdır. `6*pi = 18.84955592153876`.
    """
    t = Theory(gamma=gamma, beta=beta)
    assert closed_form_coefficient(t) == pytest.approx(
        2 * np.pi * beklenen_kat, rel=1e-15
    )
    assert closed_form_coefficient(t) == 2 * math.pi * beklenen_kat


@pytest.mark.parametrize("gamma, beta, beklenen_kat", PPN_NOKTALARI)
def test_KARELEME_KAPALI_FORMU_URETIYOR_FIZIK_CIPASI(gamma, beta, beklenen_kat):
    """SINIF: FİZİK ÇIPASI — `radial_w` ile yörüngeyi ÇÖZMEK zorunlu.

    Simülasyon bu sayı hakkında YANLIŞ OLABİLİRDİ: `a`nın 2/3'ü, `w`nin
    içindeki `u1 + u2 + u` toplamı, ikamedeki yarıçap, dışdeğerlemenin
    ağırlıkları — hepsi sayıyı kaydırır. (Ölçüldü: `a`yı yarıya indirmek
    GR'de C = 9.424777960515788 veriyor, doğrusu 18.84955592153876.)

    ÖLÇÜLEN (dosya penceresi: psi ikamesi, quad 1e-14/1e-13/400,
    Richardson (3e5, 1e6), e = 0.3, M = 1):

        gamma beta  lam   C                     bağıl fark
        1.0   1.0   3.0   18.84955591968662     9.826e-11
        0.0   1.0   1.0    6.283185307915622    1.171e-10
        1.0   0.0   4.0   25.132741225508678    1.277e-10
        2.0   1.0   5.0   31.415926529605134    2.003e-10
        1.0   3.0   1.0    6.283185307915622    1.171e-10

    GR için `6*pi`den fark -1.852e-09, yani **10.01 DÜRÜST HANE**.
    Eşik 1e-8 = en kötü ölçümün ~50 katı; sayı değil HÜKÜM sabit.

    ORTAK DÜĞME UYARISI — ölçüldü, varsayılmadı: aşağıdaki İKİ assert'ten
    BİRİNCİSİ (kareleme ↔ kapalı form) `Theory.lam`i PAYLAŞIR, dolayısıyla
    `lam`ın formülünü bozan bir mutantta iki taraf BİRLİKTE kayar ve o assert
    sessiz kalır. Mutantı öldüren şey İKİNCİ assert'teki ELLE yazılmış
    çarpandır. Bu, spec §7.3'ün "ortak düğme çapraz kontrolü körleştiriyor"
    maddesinin bu dosyadaki örneğidir ve iki assert bu yüzden ayrı durur.
    """
    t = Theory(gamma=gamma, beta=beta)
    olculen = _C_richardson(theory=t)
    assert olculen == pytest.approx(closed_form_coefficient(t), rel=1e-8)
    assert olculen == pytest.approx(2 * math.pi * beklenen_kat, rel=1e-8)


def test_KARELEME_e_DEN_BAGIMSIZ():
    """`C` dışmerkezlikten OKUNAMAZ — `e` yalnız pencereyi değiştirir.

    ÖLÇÜLDÜ (GR, dosya penceresi), e ∈ {0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95}:
    yayılım **2.462e-09**; en uzak nokta `6*pi`den -2.258e-09 (e = 0.9).
    Yayılım dışdeğerlemenin kendi gürültü tabanı mertebesindedir, yani bu
    testin ölçtüğü şey "e'nin etkisi yok" değil, "e'nin etkisi TABANIN
    ALTINDA"dır.
    """
    gr = Theory()
    degerler = [_C_richardson(theory=gr, e=e)
                for e in (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95)]
    assert max(degerler) - min(degerler) < 1e-7
    for d in degerler:
        assert d == pytest.approx(6.0 * math.pi, rel=1e-8)


def test_NEWTON_NOKTASINDA_KARELEME_TAM_2PI_hata_cubugunun_ON_KOSULU():
    """`lambda = 0` koşusu AYNI borudan geçiyor ve TAM 2*pi veriyor.

    ÖLÇÜLDÜ (dosya penceresi, e ∈ {0.3, 0.7}, p ∈ {3e5, 1e6}):
    `dphi - 2*pi` **tam 0.0** (bit bit), dolayısıyla dışdeğerlenmiş taban da
    tam 0.0 ve S/G sonsuz.

    DÜRÜST NİTELİK — ve bu Task 2 için bir UYARIDIR: bu taban, borunun
    aritmetiğini ölçmüyor. `w ≡ 1` olduğu için integrand SABİT 1.0'dır ve
    `quad` sabiti tam integre eder; yani sıfır, gürültünün küçüklüğünden
    değil, `psi` ikamesinin bu noktada yapacak iş bırakmamasından geliyor.
    Planın 2.284e-10'u BU boruda ÜRETİLEMEDİ (bkz. rapor).
    """
    newton = Theory(gamma=-1.0, beta=0.0)
    for e in (0.3, 0.7):
        for p in (3e5, 1e6):
            assert _dphi_psi(theory=newton, M=1.0, p=p, e=e) - IKI_PI == 0.0
        assert _C_richardson(theory=newton, e=e) == 0.0


# ---------------------------------------------------------------------------
# Step 5 — `e = 0` KAPISI
# ---------------------------------------------------------------------------
def test_E_SIFIR_KAPISI_dairesel_yorunge_REDDEDILIYOR():
    """`u1 == u2` (e = 0) ve `u1 > u2` reddediliyor — `InvalidInputError`."""
    gr = Theory()
    with pytest.raises(InvalidInputError) as bilgi:
        radial_w(0.3, 0.3, 0.3, theory=gr, M=1.0)
    assert "e = 0" in str(bilgi.value)
    with pytest.raises(InvalidInputError):
        radial_w(0.3, 0.5, 0.1, theory=gr, M=1.0)
    # Kapı DAR: u1 < u2 olan her şey geçer, `u`nun aralık dışında olması
    # dahil (bükülme karelemesi u1 = 0'dan başlar).
    assert math.isfinite(radial_w(0.9, 0.0, 0.5, theory=gr, M=1.0))


def test_M_KAPISI():
    """`M <= 0` fiziksel olarak anlamsız — sessizce Newton'a dönmez."""
    gr = Theory()
    for M in (0.0, -1.0):
        with pytest.raises(InvalidInputError):
            radial_w(0.3, 0.1, 0.5, theory=gr, M=M)


def test_E_SIFIR_KAPISI_GEREKCESI_OLCULDU_ve_PLANDAN_FARKLI():
    """Kapının gerekçesi ÖLÇÜLDÜ — ve spec §4.6'nın hükmü DARALTILDI.

    ÖLÇÜLEN (GR, M = 1, p = 3e5, dosya penceresi; hedef ikinci mertebe
    katsayısı `3*pi*lam² = 84.82300164692441`):

        e        ham-u karelemesi K      psi ikamesi K
        1e-12    -4.141e+09              +84.82444
        1e-06    -5.314e+03              +84.82444
        1e-03    +7.278e+01              +84.82444
        0.3      +8.563e+01              +85.249

    İKİ HÜKÜM ÇIKIYOR ve ikisi de spec §4.6'yı düzeltiyor:
    (1) `e > 0` kapısı ham-u borusu için YETMEZ — e = 1e-6 kapıdan GEÇİYOR
        ve sonuç zaten üç mertebe yanlış. Çöküş e = 0'da bir SIÇRAMA değil,
        e -> 0'da SÜREKLİ bir bozulma.
    (2) `psi` ikamesi için kapı GEREKLİ DEĞİL — e = 0'da bile doğru sayıyı
        (84.82444) veriyor.

    Kapı yine de sevk ediliyor, ama gerekçesi SAYISAL DEĞİL SÖZLEŞMESELDİR:
    `radial_w`in girdisi bağlı bir yörüngenin İKİ AYRI dönüş noktasıdır;
    `u1 == u2` bir dairedir ve orada "perihel kayması" ölçülecek bir şey
    değildir. Spec §4.6'nın "kareleme dairesel yörüngede çöküyor" cümlesi
    SEVK EDİLEN boru için geçerli değildir.
    """
    gr = Theory()
    hedef_K = 3.0 * math.pi * gr.lam ** 2
    assert hedef_K == pytest.approx(84.82300164692441, rel=1e-15)

    K_ham = _K_tek_basamak(theory=gr, M=1.0, p=3e5, e=1e-6, dphi=_dphi_ham_u)
    assert K_ham < 0.0 and abs(K_ham) > 1e3          # ölçülen -5.314e+03

    K_psi = _K_tek_basamak(theory=gr, M=1.0, p=3e5, e=1e-6, dphi=_dphi_psi)
    assert K_psi == pytest.approx(hedef_K, abs=1e-2)  # ölçülen +84.82444


# ---------------------------------------------------------------------------
# İZOTROPİK PPN AYARI
# ---------------------------------------------------------------------------
def test_IZOTROPIK_POTANSIYELLER_METRIK_SATIRINI_VERIYOR():
    """`ds² = -A dt² + B (dx²+dy²+dz²)`, `A = 1 - 2Mu + 2β(Mu)²`,
    `B = 1 + 2γMu`. GR'de u = 0.01, M = 1 için A = 0.9802, B = 1.02."""
    pot = isotropic_potential(0.01, theory=Theory(), M=1.0)
    assert isinstance(pot, IsotropicPotentials)
    assert pot.time_factor == pytest.approx(0.9802, rel=1e-15)
    assert pot.space_factor == pytest.approx(1.02, rel=1e-15)
    # beta yalnız ikinci mertebede: gamma sabit tutulup beta değişince
    # `space_factor` KIPIRDAMAMALI.
    for beta in (0.0, 1.0, 3.0):
        p2 = isotropic_potential(0.01, theory=Theory(gamma=1.0, beta=beta), M=1.0)
        assert p2.space_factor == pot.space_factor
    assert isotropic_potential(0.01, theory=Theory(beta=0.0), M=1.0).time_factor != pot.time_factor


def test_IZOTROPIK_NEWTON_NOKTASINDA_IKI_POTANSIYEL_BIT_BIT_ESIT():
    """Newton noktasının izotropik ayardaki KARŞILIĞI — ve `w ≡ 1` ile aynı
    işi görür.

    (gamma, beta) = (-1, 0)'da `A = 1 - 2Mu` ve `B = 1 - 2Mu`; iki potansiyel
    ÖZDEŞ olunca jeodezik denklemi cebirsel olarak Newton'un Kepler denklemine
    iner. ÖLÇÜLDÜ: 2000 rastgele (u, M) noktasında (u ∈ [-3, 3],
    M ∈ [1e-6, 1e6], tohum 20260828) eşitsizlik sayısı **0** — bit bit.

    DÜRÜST NİTELİK: bu eşitlik `beta`yı TEK BAŞINA kilitlemez. Ölçüldü:
    (gamma, beta) = (-1, 1e-16)'da da A == B çıkıyor, çünkü `2*beta*(Mu)²`
    terimi 0.98'in ulp'sinin altında kalıyor. `beta`nın değerini gören test
    bir üsttekidir.
    """
    newton = Theory(gamma=-1.0, beta=0.0)
    rng = np.random.default_rng(20260828)
    esitsiz = 0
    for _ in range(2000):
        u = float(rng.uniform(-3.0, 3.0))
        M = float(10.0 ** rng.uniform(-6.0, 6.0))
        pot = isotropic_potential(u, theory=newton, M=M)
        if pot.time_factor != pot.space_factor:
            esitsiz += 1
    assert esitsiz == 0
    for gamma, beta in ((1.0, 1.0), (0.0, 1.0)):
        pot = isotropic_potential(0.01, theory=Theory(gamma=gamma, beta=beta), M=1.0)
        assert pot.time_factor != pot.space_factor


@pytest.mark.parametrize("gamma, beta", [
    (1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (2.0, 1.0), (1.0, 3.0),
    (-1.0, 0.0), (0.5, -0.5), (-0.5, 2.0),
])
def test_IZOTROPIK_ACILIM_KAPALI_FORMU_URETIYOR(gamma, beta):
    """SINIF: KAPALI-FORM TÜREV — ve `2*pi*lam` ile PPN'in TEK FORM oluşunun
    İKİNCİ, BAĞIMSIZ kod yolundan okunuşu.

    Jeodezik denkleminde teori bağımlılığının tamamı `B/A` oranından gelir:

        B/A = 1 + (2 + 2*gamma)·x + (4 - 2*beta + 4*gamma)·x² + O(x³),
        x = M*u

    İkinci mertebe katsayısı `4 - 2*beta + 4*gamma = 2*lam`, yani
    `closed_form_coefficient(theory)/pi`. Bu test o katsayıyı `B/A`dan SAYISAL
    olarak söküp kapalı formla karşılaştırır; iki taraf farklı kod yollarından
    gelir (biri metrik cebiri, öteki tek satırlık kapalı form).

    PENCERE: `x = 1e-3`, iki kademeli Richardson
    `R1(x) = 2q(x/2) - q(x)`, `R2(x) = (4R1(x/2) - R1(x))/3` ile
    `q(x) = (B/A - 1 - (2+2*gamma)x)/x²`. ÖLÇÜLDÜ: sekiz (gamma, beta)
    noktasında en kötü bağıl fark **2.302e-09**, ve `x = 1e-3` bir
    OPTİMUMDUR — x = 1e-2'de 1.845e-06, x = 1e-5'te 3.442e-05. Eşik 1e-6,
    yani en kötü ölçümün ~430 katı.
    """
    t = Theory(gamma=gamma, beta=beta)
    c1 = 2.0 + 2.0 * gamma

    def q(x: float) -> float:
        pot = isotropic_potential(x, theory=t, M=1.0)
        return (pot.space_factor / pot.time_factor - 1.0 - c1 * x) / (x * x)

    def R1(x: float) -> float:
        return 2.0 * q(x / 2.0) - q(x)

    x = 1e-3
    c2 = (4.0 * R1(x / 2.0) - R1(x)) / 3.0
    assert c2 == pytest.approx(closed_form_coefficient(t) / math.pi, rel=1e-6, abs=1e-9)
    assert c2 == pytest.approx(2.0 * t.lam, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("gamma", [1.0, 0.0, 2.0, -1.0, 0.5])
def test_IZOTROPIK_ACILIM_BIRINCI_MERTEBE_IKI_ARTI_IKI_GAMMA(gamma):
    """`B/A`nın birinci mertebe katsayısı `2 + 2*gamma` — ve `beta` oraya
    HİÇ girmiyor. ÖLÇÜLDÜ (x = 1e-8, M = 1): GR'de 4.000000064507958."""
    for beta in (0.0, 1.0, 3.0):
        t = Theory(gamma=gamma, beta=beta)
        x = 1e-8
        pot = isotropic_potential(x, theory=t, M=1.0)
        c1 = (pot.space_factor / pot.time_factor - 1.0) / x
        assert c1 == pytest.approx(2.0 + 2.0 * gamma, rel=1e-6, abs=1e-7)


def test_RADIAL_W_NDARRAY_UZERINDE_ELEMAN_ELEMAN_CALISIYOR():
    """`u` bir ndarray olabilir — Task 4'ün yörünge motorunun ön koşulu."""
    gr = Theory()
    u = np.array([0.1, 0.2, 0.3])
    cikti = radial_w(u, 0.1, 0.5, theory=gr, M=1.0)
    assert isinstance(cikti, np.ndarray)
    for i, tek in enumerate(u):
        assert cikti[i] == radial_w(float(tek), 0.1, 0.5, theory=gr, M=1.0)
