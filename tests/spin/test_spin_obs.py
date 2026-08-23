"""Spin gözlemlenebilirleri — HATA ÇUBUĞUNUN KENDİSİ ÖLÇÜLÜYOR.

Bu dosya dört ayrı iş yapar:

1. KAPALI FORM ÇIPALARI (sınıf: DAİRESEL DEĞİL, cebir) — `binning_error`ın
   `tau_int`i AR(1) sürecinin TAM `τ = (1+φ)/(2(1−φ))` değeriyle, hata
   çubuğu ise TAM `SE = σ·√(2τ/N)` ile sınanır. Bunlar MC ölçümü değil
   CEBİRDİR: tahminci, Monte Carlo'ya hiç sorulmadan doğru ya da yanlış
   çıkar. Aynı şekilde `binder_cumulant` bağımsız Gauss dizisinde 0, tam
   düzenli dizide TAM 2/3 verir.

2. TAM SAYIM ÇIPALARI (sınıf: DAİRESEL DEĞİL) — L = 4 Ising'in 2^16 = 65536
   durumu sayılıp `U`, `χ`, `C` Boltzmann ağırlığıyla TAM hesaplanır ve MC
   ile karşılaştırılır. Sayımın enerjisi BAĞIMSIZ yazılır (her bağ tam bir
   kez), yani `lattice.total_energy`in çift-sayım düzeltmesini kullanmaz.
   `χ`nin `L²` katsayısı ve Binder'ın `3`ü ancak burada gerçekten kilitlenir.

3. TAŞIYICI HÜKÜM — naif SEM YANILIYOR ve plato oluşmadıysa dönen sayı
   hatanın kendisi DEĞİL bir alt sınırıdır.

4. TAHMİNCİNİN KIRILDIĞI BÖLGELER — `correlation_length`in düzenli fazda
   mıknatıslanmayı okuduğu, ilişkisiz kafeste yarı yarıya `None` döndüğü,
   ve `heat_capacity`nin ölçülmemiş enerjiyi sıfır saymadığı.

────────────────────────────────────────────────────────────────────────────
PLATO ÖLÇÜTÜ — PLANIN ÖLÇÜTÜ ÖLÇÜLDÜ ve KENDİ İDDİASINI GEÇMEDİ
────────────────────────────────────────────────────────────────────────────

Plan (Task 4, Step 3) platoyu "son ÜÇ blok boyutunun hataları %5 içinde mi"
diye tanımlıyor, `min_blocks = 16` kullanıyor ve zorunlu testinde
`sonuc_w.plateau_reached is True` diyordu. ÖLÇÜLDÜ (L = 32, T = T_c,
8000 ölçüm, 24 bağımsız zincir, tohumlar 20260824 + 1000·k):

    ölçüt                       Metropolis      Wolff
    plan (%5, min_blocks=16)      0/24 True     3/24 True   ← plan YANLIŞ
    plan (%2)                     0/24          1/24
    plan (%10)                    1/24         11/24
    plan (%20)                    8/24         23/24

Yani planın kendi zorunlu testi, planın kendi kodu ve kendi eşiğiyle 24
tohumun 21'inde KIRMIZI olurdu. Sebep ölçüldü ve bu depoda tanıdıktır: bir
std tahmininin bağıl saçılımı ≈ 1/√(2(K−1)) ve `min_blocks = 16`ta son üç
basamakta yalnız K = 125, 62, 31 blok kalır —

    b       K      1/√(2(K−1))
    64     125        6.35%
    128     62        9.05%
    256     31       12.91%

— yani %5'lik eşik, KENDİ SAÇILIMI eşiğin iki-üç katı olan üç sayıyı
karşılaştırıyor. Task 2'nin 6 zincirle gördüğü sahte "yanlılık" ve Task 3'ün
K = 8'de %42 dar çıkan SE'si ile AYNI tuzak.

ÖLÇÜLEREK SEÇİLEN ÖLÇÜT: son nokta, üç önceki noktadan İSTATİSTİKSEL OLARAK
anlamlı biçimde yüksek değilse plato —

    e[-1] − e[-3]  ≤  z · e[-1] · √(σ[-1]² + σ[-3]²),   σ_i = 1/√(2(K_i−1))

`min_blocks` ve `z` ELLE SEÇİLMEDİ; ikisi de tarandı (aynı 24 zincir):

    min_blocks   z      Metropolis (plato OLMAMALI)   Wolff (plato OLMALI)
        16      1.0            4/24  ✗                    19/24  ✗
        16      2.0           19/24  ✗✗                   24/24  ✓
        32      1.0            0/24  ✓                    17/24  ✗
        32      2.0            0/24  ✓                    24/24  ✓   ← SEÇİLEN
        64      1.0            0/24  ✓                     7/24  ✗
        64      2.0            0/24  ✓                    19/24  ✗

Altı kombinasyondan YALNIZ BİRİ iki tarafı da ayırıyor. `min_blocks = 16`
son basamakta çok gürültülü (z = 2 bandı gerçek tırmanışı yutuyor,
Metropolis'te 19/24 yanlış "plato"), `min_blocks = 64` ise eğriyi Wolff'un
plato bölgesine varmadan kesiyor.

BAYRAĞIN NE İDDİA ETTİĞİ, BAĞIMSIZ BİR REFERANSLA ÖLÇÜLDÜ. L = 32, T = T_c,
8000 ölçüm; gerçek hata = BAĞIMSIZ zincir ortalamalarının std'si
(Metropolis 200 zincir → ±%5.0, Wolff 100 zincir → ±%7.1):

    algoritma    plato   binning `error` / GERÇEK    naif SEM'in yanılması
    Wolff        True            1.0572                  2.37 kat küçük
    Metropolis   False           0.8538                  8.81 kat küçük

Yani bayrak `True` dediğinde sayı gerçeğin %6 içinde ve YUKARI yanlı (hata
çubuğunda doğru yön); `False` dediğinde %15 AŞAĞI — tam da "alt sınır"
demek. Bayrağın hükmü ile sayının hatası AYNI YÖNÜ gösteriyor.

────────────────────────────────────────────────────────────────────────────
PLATO BAYRAĞI ALGORİTMANIN DEĞİL KOŞUMUN ÖZELLİĞİDİR — ÖLÇÜLDÜ
────────────────────────────────────────────────────────────────────────────

Planın anlatısı "Metropolis plato yapmaz, Wolff yapar" diyor. Bu, ALGORİTMA
hakkında bir cümle olarak YANLIŞ; ölçülen (24 zincir, T = T_c, seçilen
ölçüt):

    algoritma     L    N ölçüm    plato    binning error / naif SEM
    Wolff         8      2000     23/24      1.74
    Wolff         8      4000     24/24      1.75
    Wolff        16      2000     20/24      2.03
    Wolff        16      8000     24/24      2.11
    Wolff        32      8000     24/24      2.53
    Metropolis    8      2000     23/24      2.37   ← Metropolis de PLATO YAPIYOR
    Metropolis    8      4000     24/24      2.50
    Metropolis   16      2000      0/24      3.84
    Metropolis   16      4000      6/24      4.22
    Metropolis   16      8000     23/24      4.39   ← ve yine yapıyor
    Metropolis   32      2000      0/24      4.88
    Metropolis   32      8000      0/24      7.45

Bayrak "bu ALGORİTMA iyi mi" değil, "BU KOŞUM kendi otokorelasyonunu çözecek
kadar uzun mu" sorusunu cevaplıyor. Bu yüzden aşağıdaki her testin
dokstringinde L, T, seri uzunluğu, tohum ve algoritma AÇIKÇA yazılıdır.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.spin.lattice import SpinState
from qlab.engine.spin.metropolis import run_metropolis
from qlab.engine.spin.models import HeisenbergModel, IsingModel
from qlab.engine.spin.wolff import run_wolff
from qlab.errors import InvalidInputError
from qlab.observable.spin_obs import (
    BinningResult,
    binder_cumulant,
    binning_error,
    correlation_length,
    heat_capacity,
    susceptibility,
)

TOHUM = 20260824
TC = 2.0 / np.log(1.0 + np.sqrt(2.0))
ZINCIR24 = tuple(TOHUM + 1000 * k for k in range(24))


# --------------------------------------------------------------------------
# Yardımcılar
# --------------------------------------------------------------------------


def _ar1(n: int, phi: float, tohum: int) -> np.ndarray:
    """AR(1): x_t = φ x_{t−1} + ε_t, DURAĞAN başlatılır (ε ~ N(0,1)).

    Bu sürecin iki büyüklüğü KAPALI FORMDUR ve hiçbir yere GİRDİ değildir:

        tau_int = (1+φ)/(2(1−φ))          σ_x = 1/√(1−φ²)
        gerçek SE(ortalama) = σ_x·√(2·tau_int/N)

    φ = 0 bağımsız seridir ve `tau_int = 1/2` verir.
    """
    r = np.random.default_rng(tohum)
    e = r.normal(size=n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1.0 - phi**2) if phi else e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _ar1_tau(phi: float) -> float:
    return (1.0 + phi) / (2.0 * (1.0 - phi))


def _ar1_gercek_se(phi: float, n: int) -> float:
    return float(np.sqrt(1.0 / (1.0 - phi**2)) * np.sqrt(2.0 * _ar1_tau(phi) / n))


def _tau_acf(series: np.ndarray, c: float = 5.0) -> float:
    """BAĞIMSIZ `tau_int`: otokorelasyon fonksiyonundan, otomatik pencereyle.

    `binning_error`ın hesabıyla ORTAK HİÇBİR SATIRI YOKTUR — o blok
    ortalamalarının std'sinden gider, bu FFT ile otokovaryansı kurup
    `tau = 1/2 + Σ ρ(t)`yi ilk `W ≥ c·tau(W)` noktasında keser (Sokal
    penceresi).
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    x = x - x.mean()
    nf = 1
    while nf < 2 * n:
        nf *= 2
    f = np.fft.rfft(x, nf)
    acf = np.fft.irfft(f * np.conjugate(f), nf)[:n].real
    acf /= acf[0]
    tau = 0.5
    for W in range(1, n):
        tau += acf[W]
        if W >= c * tau:
            return float(tau)
    return float(tau)


def _tam_sayim(L: int, T: float, J: float = 1.0) -> dict[str, float]:
    """2^(L·L) durumun HEPSİ — U, χ, C ve ⟨|m|⟩ TAM değerleri.

    Enerji BAĞIMSIZ yazılır: her bağ TAM BİR KEZ (her sitenin sağ ve aşağı
    komşusu), `lattice.neighbour_sum` + 0.5 düzeltmesi KULLANILMAZ.
    """
    n = L * L
    idx = np.arange(2**n, dtype=np.int64)
    bit = (idx[:, None] >> np.arange(n)) & 1
    s = (2 * bit - 1).astype(float).reshape(-1, L, L)
    bag = np.sum(s * np.roll(s, -1, axis=1), axis=(1, 2)) + np.sum(
        s * np.roll(s, -1, axis=2), axis=(1, 2)
    )
    E = -J * bag
    m = s.mean(axis=(1, 2))
    w = np.exp(-(E - E.min()) / T)
    w /= w.sum()

    def ort(v):
        return float((w * v).sum())

    return {
        "C": (ort(E**2) - ort(E) ** 2) / (n * T * T),
        "U": 1.0 - ort(m**4) / (3.0 * ort(m**2) ** 2),
        "chi_abs": n * (ort(m**2) - ort(np.abs(m)) ** 2) / T,
        "chi_isaretli": n * (ort(m**2) - ort(m) ** 2) / T,
        "mabs": ort(np.abs(m)),
    }


def _wolff_seri(L: int, T: float, tohum: int, n_burn: int, n_sweep: int) -> np.ndarray:
    return run_wolff(
        SpinState.uniform(L, 1), temperature=T, coupling=1.0, field=0.0,
        rng=np.random.default_rng(tohum), n_sweep=n_sweep, n_burn=n_burn,
    ).magnetisation_series


def _metro_kosum(L: int, T: float, tohum: int, n_burn: int, n_sweep: int):
    return run_metropolis(
        SpinState.uniform(L, 1), IsingModel(), temperature=T, coupling=1.0,
        field=0.0, rng=np.random.default_rng(tohum), n_sweep=n_sweep, n_burn=n_burn,
    )


def _beyaz_kafes(L: int, tohum: int) -> np.ndarray:
    """Tamamen İLİŞKİSİZ ±1 kafes — gerçek ξ = 0."""
    r = np.random.default_rng(tohum)
    return 2.0 * r.integers(0, 2, (L, L, 1)).astype(float) - 1.0


def _se(v) -> float:
    return float(np.std(v, ddof=1) / np.sqrt(len(v)))


# ==========================================================================
# 1. KAPALI FORM ÇIPALARI — tahminci MC'ye SORULMADAN sınanıyor
# ==========================================================================


def test_BAGIMSIZ_SERIDE_BINNING_NAIF_SEMe_OTURUYOR_ve_tau_YARIM():
    """SINIF: DAİRESEL DEĞİL (cebir). PENCERE: AR(1) φ = 0, N = 8000,
    24 tohum (20260824 + 31·k), `min_blocks` varsayılan 32.

    İlişkisiz bir seride blok ortalaması hiçbir şey öğrenmez: binning hatası
    naif SEM'e OTURMALI ve `tau_int` KAPALI FORMDAN 1/2 olmalıdır. Bu bir
    ölçüm değil cebirdir — tahminci, hiçbir Monte Carlo koşumuna sorulmadan
    burada doğru ya da yanlış çıkar.

    ÖLÇÜLDÜ: `error`/naif SEM = 1.039123 ± 0.046507 (24 tohum),
    `tau_int` = 0.541 ± 0.050 (tam değer 0.500), plato 24/24 True.

    `error = max(hatalar)` SEÇİMİNİN BEDELİ BURADA GÖRÜNÜYOR ve BEYAN
    EDİLİYOR: 8 gürültülü sayının maksimumu ~%4 YUKARI yanlıdır. Alternatif
    `error = hatalar[-1]` ise 0.965151 ± 0.095293 — yani %3.5 AŞAĞI yanlı ve
    saçılımı İKİ KAT. Bir hata çubuğunda yukarı yanlılık DOĞRU yöndür;
    aşağı yanlılık, çubuğu olduğundan dar göstermenin kibar halidir.
    """
    oranlar, taular, plato = [], [], 0
    for k in range(24):
        sonuc = binning_error(_ar1(8000, 0.0, TOHUM + 31 * k))
        oranlar.append(sonuc.error / sonuc.errors[0])
        taular.append(sonuc.tau_int)
        plato += sonuc.plateau_reached

    assert plato == 24, f"bağımsız seride plato HER ZAMAN oluşmalı: {plato}/24"
    # Tolerans elle seçilmedi: 24 tohumun kendi standart hatasının 4 katı.
    assert abs(np.mean(oranlar) - 1.039123) < 4 * _se(oranlar)
    assert 1.0 < np.mean(oranlar) < 1.10, np.mean(oranlar)
    assert abs(np.mean(taular) - 0.5) < 4 * _se(taular) + 0.05


@pytest.mark.parametrize("phi", [0.5, 0.8])
def test_AR1in_KAPALI_FORM_tau_INTINI_binning_GERI_OKUYOR(phi):
    """SINIF: DAİRESEL DEĞİL (cebir). PENCERE: N = 8000, 24 tohum
    (20260824 + 31·k), `min_blocks` = 32, plato ölçütü z = 2.

    `tau_int = (1+φ)/(2(1−φ))` AR(1)'in TAM değeridir ve hiçbir yere GİRDİ
    değildir. `binning_error` onu blok ortalamalarından geri okumak zorunda.

    ÖLÇÜLDÜ:
        φ = 0.5:  tam 1.500   binning 1.552 ± 0.195   oran 1.035   plato 24/24
        φ = 0.8:  tam 4.500   binning 4.446 ± 0.649   oran 0.988   plato 24/24
    """
    tam = _ar1_tau(phi)
    taular, plato = [], 0
    for k in range(24):
        sonuc = binning_error(_ar1(8000, phi, TOHUM + 31 * k))
        taular.append(sonuc.tau_int)
        plato += sonuc.plateau_reached

    assert plato == 24, f"φ={phi}: plato oluşmalı, {plato}/24"
    assert abs(np.mean(taular) - tam) < 4 * _se(taular), (
        f"φ={phi}: tam {tam}, ölçülen {np.mean(taular):.4f} ± {_se(taular):.4f}"
    )


def test_PLATO_OLUSMADIYSA_ALT_SINIR_diye_isaretleniyor():
    """EN ÖNEMLİ TEST. SINIF: DAİRESEL DEĞİL (cebir) + MC.

    Plato yapmamış bir binning tahmincisi hatanın KENDİSİNİ değil bir ALT
    SINIRINI verir. Kod bunu tespit etmezse "hata çubuğu hesapladım" demek,
    onu olduğundan küçük göstermenin kibar hali olur.

    KAPALI FORM PENCERESİ: AR(1), N = 8000, 24 tohum (20260824 + 31·k).
    Burada "alt sınır" bir iddia değil ÖLÇÜLEN bir sayıdır, çünkü gerçek
    `tau_int` BİLİNMEKTEDİR:

        φ       tam τ     binning τ    oran     plato
        0.8       4.500      4.446     0.988    24/24 True
        0.95     19.500     16.333     0.838    10/24
        0.995   199.500     52.833     0.265     0/24 False

    Bayrak tam olarak işini yapıyor: `True` dediği yerde `tau_int` %1 doğru,
    `False` dediği yerde gerçeğin DÖRTTE BİRİ. φ = 0.95 ARA BÖLGEDİR ve
    10/24 ile bayrağın da bir geçiş bölgesi olduğunu gösterir — bu bir kusur
    değil, ölçütün gerçek çözünürlüğüdür ve burada beyan ediliyor.

    MC PENCERESİ (T = T_c, tohum 20260824):
        Metropolis L = 16, n_burn = 500, N = 2000 → plato False (24/24 tohum)
        Wolff      L =  8, n_burn = 400, N = 4000 → plato True  (24/24 tohum)
    İki farklı L bilinçlidir: bayrak ALGORİTMANIN değil KOŞUMUN özelliğidir
    (dosya başlığındaki tablo) ve bu ikisi, 24 tohumun HEPSİNDE aynı hükmü
    veren en ucuz iki penceredir. Aynı iddia planın kendi L = 32 penceresinde
    `test_PLANIN_D_PENCERESI_L32_Tc_yeniden_olculuyor` ile bir kez daha
    kuruluyor (yavaş şerit, 6.5 s).
    """
    # --- kapalı form: plato YOK, ve alt sınır olduğu ÖLÇÜLEBİLİYOR ---
    kot = [binning_error(_ar1(8000, 0.995, TOHUM + 31 * k)) for k in range(24)]
    assert all(s.plateau_reached is False for s in kot)
    tau_kot = float(np.mean([s.tau_int for s in kot]))
    assert tau_kot < 0.5 * _ar1_tau(0.995), (
        f"plato yokken tau_int gerçeğin çok altında olmalı: {tau_kot:.3f} "
        f"vs tam {_ar1_tau(0.995):.3f}"
    )
    assert abs(tau_kot - 52.833) < 4 * _se([s.tau_int for s in kot])

    # --- kapalı form: plato VAR, ve orada sayı gerçekten hatanın kendisi ---
    iyi = [binning_error(_ar1(8000, 0.8, TOHUM + 31 * k)) for k in range(24)]
    assert all(s.plateau_reached is True for s in iyi)

    # --- MC: aynı hüküm gerçek Ising verisinde ---
    metro = binning_error(_metro_kosum(16, TC, TOHUM, 500, 2000).magnetisation_series)
    assert metro.plateau_reached is False
    wolff = binning_error(_wolff_seri(8, TC, TOHUM, 400, 4000))
    assert wolff.plateau_reached is True


@pytest.mark.slow
def test_PLANIN_D_PENCERESI_L32_Tc_yeniden_olculuyor():
    """SINIF: transkripsiyon denetimi. PENCERE: L = 32, T = T_c, N = 8000,
    tohum 20260824; Metropolis n_burn = 2000, Wolff n_burn = 1000.
    `slow` GEREKÇESİ: ölçülen call süresi 6.5 s, eşik 2.0 s.

    Planın D tablosunun penceresi budur. Aynı iddia hızlı şeritte
    `test_PLATO_OLUSMADIYSA_ALT_SINIR_diye_isaretleniyor` içinde daha kısa
    zincirlerle bir kez daha kuruluyor; bu test onu PLANIN KENDİ
    penceresinde tekrarlar. ÖLÇÜLDÜ (24 zincir): Metropolis 0/24 plato,
    Wolff 24/24 plato.
    """
    metro = binning_error(_metro_kosum(32, TC, TOHUM, 2000, 8000).magnetisation_series)
    wolff = binning_error(_wolff_seri(32, TC, TOHUM, 1000, 8000))
    assert metro.plateau_reached is False
    assert wolff.plateau_reached is True
    # Wolff'un otokorelasyon zamanı Metropolis'inkinden BİR MERTEBE küçük.
    assert wolff.tau_int < 0.2 * metro.tau_int


# ==========================================================================
# 2. TAŞIYICI HÜKÜM — NAİF SEM YANILIYOR
# ==========================================================================


@pytest.mark.parametrize(
    "phi, N, kat_beklenen",
    [(0.8, 2000, 3.0173), (0.8, 8000, 3.0197), (0.95, 8000, 6.3292)],
)
def test_NAIF_HATA_BIR_MERTEBE_YANILIYOR_ve_bunu_ISPATLIYORUZ(phi, N, kat_beklenen):
    """SINIF: DAİRESEL DEĞİL (cebir) — referans KAPALI FORMDUR.
    PENCERE: AR(1), 24 tohum (20260824 + 31·k).

    "Naif SEM" = `std(seri, ddof=1)/√N`, yani ardışık örnekleri BAĞIMSIZ
    sayan tahmin. Örnekler ilişkiliyse bu tahmin gerçeği KÜÇÜK gösterir ve
    yanılma miktarı AR(1)'de TAM OLARAK BİLİNİR: `√(2·tau_int)`.

    ÖLÇÜLDÜ (gerçek SE = σ_x·√(2τ/N), simülasyona GİRDİ DEĞİL):

        φ      N      gerçek SE      naif SEM      gerçek/naif   √(2τ)
        0.80   2000   1.118034e-01   3.705431e-02    3.0173      3.0000
        0.80   8000   5.590170e-02   1.851222e-02    3.0197      3.0000
        0.95   8000   2.236068e-01   3.532917e-02    6.3292      6.2450

    `binning_error` bunu yakalıyor: `error`/gerçek SE = 0.9162 / 0.9853 /
    0.9014.

    DÜRÜST OLUMSUZ, AYNI ÖLÇÜMDEN: 24 zincirlik AMPİRİK referans (zincir
    ortalamalarının std'si) kapalı formun 0.9612 / 0.8875 / 0.8918 katı
    çıkıyor — yani referansın KENDİSİ %4-11 aşağıda. Kendi bağıl saçılımı
    1/√(2·23) = %14.7 olduğu için bu tutarlı; ama depoda tekrar tekrar
    çıkan dersi bir kez daha ölçüyor: bir referans da bir tahmindir.

    GERÇEK MC PENCERESİNDE (L = 32, T_c, 8000 ölçüm) aynı hüküm, BAĞIMSIZ
    zincirlerle ölçüldü — Metropolis 200 zincir, Wolff 100 zincir:
        Metropolis: gerçek 1.713639e-02, naif 1.946079e-03 → 8.81 KAT
        Wolff:      gerçek 4.642595e-03, naif 1.955908e-03 → 2.37 KAT
    PLANIN SAYISI BAYAT: plan (D tablosu) "naif 2.193e-03, gerçek 2.275e-02,
    10.4 KAT" diyor ve adlandırdığı testin adında "ON KAT" geçiyordu.
    Ölçülen 8.81 (±%5.0); ad ÖLÇÜLEN büyüklüğe göre "BİR MERTEBE" olarak
    düzeltildi — "on kat" yazan bir ad, düzeltilmemiş bir sayıyı adın içinde
    saklamak olurdu.

    NAİF SEM BU DEPODA HİÇBİR YERDE RAPORLANMAZ; burada yalnızca YANLIŞ
    olduğunu göstermek için hesaplanıyor.
    """
    gercek = _ar1_gercek_se(phi, N)
    naifler, binningler, ortalamalar = [], [], []
    for k in range(24):
        x = _ar1(N, phi, TOHUM + 31 * k)
        sonuc = binning_error(x)
        naifler.append(sonuc.errors[0])
        binningler.append(sonuc.error)
        ortalamalar.append(float(x.mean()))

    kat = gercek / float(np.mean(naifler))
    assert abs(kat - kat_beklenen) < 0.02 * kat_beklenen, kat
    assert abs(kat - np.sqrt(2 * _ar1_tau(phi))) < 0.03 * kat, kat
    # binning gerçeğe oturuyor; sınır 24 zincirin kendi saçılımından (%14.7).
    assert abs(float(np.mean(binningler)) / gercek - 1.0) < 0.147


def test_NAIF_SEM_GERCEK_ISING_VERISINDE_de_KUCUK_cikiyor():
    """SINIF: transkripsiyon denetimi (taşıyıcı iddianın MC tekrarı).
    PENCERE: L = 16, T = T_c, Metropolis, n_burn = 500, N = 2000,
    tohum 20260824. Ölçülen call süresi 0.2 s.

    Yukarıdaki kapalı-form ispatının gerçek Ising verisindeki ucuz tekrarı.
    Burada gerçek hata BİLİNMEDİĞİ için hüküm YÖNSELDİR ve öyle beyan
    ediliyor: `binning_error.error` naif SEM'in en az 3 katı olmalı.
    ÖLÇÜLDÜ (24 zincir ortalaması, bu pencere): oran 3.84.

    Bağımsız zincirlerle kurulan MUTLAK hüküm yavaş şeritte
    (`test_NAIF_HATA_BAGIMSIZ_ZINCIRLERLE_L16`, 6.5 s).
    """
    sonuc = binning_error(_metro_kosum(16, TC, TOHUM, 500, 2000).magnetisation_series)
    assert sonuc.plateau_reached is False
    assert sonuc.error / sonuc.errors[0] > 3.0, sonuc.error / sonuc.errors[0]


@pytest.mark.slow
def test_NAIF_HATA_BAGIMSIZ_ZINCIRLERLE_L16():
    """SINIF: DAİRESEL DEĞİL — hüküm BAĞIMSIZ ZİNCİRLERDEN geliyor.
    PENCERE: L = 16, T = T_c, Metropolis n_burn = 500, N = 2000, 24 bağımsız
    zincir (tohumlar 20260824 + 1000·k).
    `slow` GEREKÇESİ: ölçülen call süresi 6.5 s, eşik 2.0 s.

    Gerçek hata = 24 BAĞIMSIZ zincirin ortalamalarının std'si; binning ile
    ORTAK HİÇBİR SATIRI YOKTUR.

    ÖLÇÜLDÜ: bağımsız referans 1.9875e-02, naif SEM 4.4533e-03 → naif SEM
    4.46 KAT KÜÇÜK. binning `error` 1.7157e-02, yani referansın 0.863 katı —
    yani `plateau_reached` False derken sayı gerçekten AŞAĞIDA.

    DÜRÜST OLUMSUZ: 24 zincirlik referansın kendi bağıl belirsizliği %14.7,
    ve L = 32 penceresinde 24 zincirin verdiği referans (1.3666e-02)
    200 zincirinkinin (1.7136e-02) yalnız 0.80 katı çıktı. Bu testin
    toleransı bu yüzden 3 × %14.7'dir, daha dar değil.
    """
    ortalamalar, naifler, binningler = [], [], []
    for tohum in ZINCIR24:
        s = _metro_kosum(16, TC, tohum, 500, 2000).magnetisation_series
        ortalamalar.append(float(s.mean()))
        sonuc = binning_error(s)
        naifler.append(sonuc.errors[0])
        binningler.append(sonuc.error)

    referans = float(np.std(ortalamalar, ddof=1))
    assert referans / float(np.mean(naifler)) > 3.0
    assert abs(float(np.mean(binningler)) / referans - 1.0) < 3 * 0.147


def test_tau_int_IKI_BAGIMSIZ_YOLDAN_okunuyor():
    """SINIF: ÇAPRAZ KONTROL + KAPALI FORM. PENCERE: AR(1), N = 8000,
    24 tohum (20260824 + 31·k).

    ÜÇ yol karşılaştırılıyor ve üçüncüsü HAKEMDİR:
      (a) `binning_error.tau_int` — blok ortalamalarının std'sinden,
          `tau = ½(SEM_plato/SEM_naif)²`
      (b) `_tau_acf` — FFT ile otokovaryans + Sokal penceresi
      (c) `(1+φ)/(2(1−φ))` — TAM DEĞER
    (a) ile (b)'nin ortak hiçbir satırı yok; (c) ikisine de girdi değil.

    BİREBİR UYUŞMALARI BEKLENMİYOR — ölçülen, NE KADAR uyuştuklarıdır:

        φ       tam        binning        ACF          bin/ACF   plato
        0.000     0.500    0.541±0.050    0.494±0.019   1.0945   24/24
        0.500     1.500    1.552±0.195    1.472±0.050   1.0548   24/24
        0.800     4.500    4.446±0.649    4.409±0.386   1.0083   24/24
        0.950    19.500   16.333±2.492   18.870±3.998   0.8656   10/24
        0.995   199.500   52.833±2.748  169.274±80.775  0.3121    0/24

    DESEN ÖLÇÜLDÜ ve tesadüf değil: platonun OLDUĞU yerde iki yol %10
    içinde buluşuyor, OLMADIĞI yerde binning sistematik olarak DÜŞÜK kalıyor
    (φ = 0.995'te ACF'nin ÜÇTE BİRİ). Yani bayrak ile uyuşmazlığın yönü AYNI
    ŞEYİ söylüyor — plato yoksa `tau_int` de bir alt sınırdır. ACF'nin
    kendisi de φ = 0.995'te 0.85 katla ve ±%48 saçılımla zorlanıyor; hiçbir
    tahminci N = 8000'de τ = 199.5'i temiz okuyamıyor.
    """
    for phi, esik in ((0.0, 0.15), (0.5, 0.15), (0.8, 0.15)):
        b = [binning_error(_ar1(8000, phi, TOHUM + 31 * k)).tau_int for k in range(24)]
        a = [_tau_acf(_ar1(8000, phi, TOHUM + 31 * k)) for k in range(24)]
        oran = float(np.mean(b) / np.mean(a))
        assert abs(oran - 1.0) < esik, (phi, oran)

    # Plato YOKSA binning AŞAĞI kalıyor -- yön ölçüldü, tesadüfi değil.
    b995 = [binning_error(_ar1(8000, 0.995, TOHUM + 31 * k)).tau_int for k in range(24)]
    a995 = [_tau_acf(_ar1(8000, 0.995, TOHUM + 31 * k)) for k in range(24)]
    assert float(np.mean(b995) / np.mean(a995)) < 0.5


@pytest.mark.slow
def test_tau_int_CAPRAZ_KONTROLU_GERCEK_MC_VERISINDE():
    """SINIF: ÇAPRAZ KONTROL. PENCERE: T = T_c, 12 tohum
    (20260824 + 1000·k); Wolff L = 8, n_burn = 400, N = 2000 (plato VAR) ve
    Metropolis L = 16, n_burn = 500, N = 2000 (plato YOK).
    `slow` GEREKÇESİ: ölçülen call süresi 3.6 s, eşik 2.0 s.

    Kapalı formda gösterilen desenin gerçek Ising verisindeki karşılığı.
    ÖLÇÜLDÜ (12 tohum ortalaması):

        pencere                              binning     ACF      oran
        Wolff L=8   N=2000  (plato 11/12)      1.564     1.521   1.0284
        Metropolis L=16 N=2000 (plato 0/12)    7.486    11.456   0.6535

    Plato varken %3 içinde buluşuyorlar, yokken binning ACF'nin 0.65 katı.
    """
    wb, wa, mb, ma = [], [], [], []
    for tohum in ZINCIR24[:12]:
        sw = _wolff_seri(8, TC, tohum, 400, 2000)
        wb.append(binning_error(sw).tau_int)
        wa.append(_tau_acf(sw))
        sm = _metro_kosum(16, TC, tohum, 500, 2000).magnetisation_series
        mb.append(binning_error(sm).tau_int)
        ma.append(_tau_acf(sm))

    w_oran = float(np.mean(wb) / np.mean(wa))
    m_oran = float(np.mean(mb) / np.mean(ma))
    assert abs(w_oran - 1.0) < 0.15, w_oran
    assert m_oran < 0.8, m_oran
    assert m_oran < w_oran


# ==========================================================================
# 3. TAM SAYIM ÇIPALARI — U, χ, C'nin MUTLAK değerleri
# ==========================================================================


@pytest.mark.parametrize(
    "T, U_tam, chi_tam, mabs_tam",
    [
        (2.0, 0.644810791, 0.195719624, 0.918943267),
        (TC, 0.617199318, 0.347320832, 0.843860445),
        (3.5, 0.380777825, 0.383037135, 0.488437184),
    ],
)
def test_BINDER_ve_SUSEPTIBILITE_TAM_SAYIMLA_ucusuyor(T, U_tam, chi_tam, mabs_tam):
    """SINIF: DAİRESEL DEĞİL — 2^16 = 65536 durumun TAM sayımı.
    PENCERE: L = 4, Wolff, n_burn = 500, N = 4000, 12 tohum
    (20260824 + 1000·k). Ölçülen call süresi 0.9 s. Tolerans ELLE SEÇİLMEZ:
    12 zincir ortalamasının standart hatasının 4 katı.

    Binder'ın `3` çarpanı ve χ'nin `L²` katsayısı ancak MUTLAK bir değere
    oturunca kilitlenir; oran testleri ikisini de kaçırır.

    ÖLÇÜLDÜ (bu pencerede, sapmalar SE cinsinden):
        T = 2.0   U 0.645033104±0.000386 (0.58×SE)  χ 0.194550±0.004059 (0.29×SE)
        T = T_c   U 0.617325185±0.000680 (0.19×SE)  χ 0.343738±0.005108 (0.70×SE)
        T = 3.5   U 0.381130846±0.001963 (0.18×SE)  χ 0.380614±0.001413 (1.71×SE)

    Tam sayım değerleri (bu testin parametreleri) BAĞIMSIZ yazılmış bir
    enerji ile üretilir ve simülasyona hiç girmez; `_tam_sayim` her koşumda
    yeniden hesaplanıp parametrelerle karşılaştırılır (transkripsiyon
    kilidi).
    """
    tam = _tam_sayim(4, T)
    assert abs(tam["U"] - U_tam) < 1e-9
    assert abs(tam["chi_abs"] - chi_tam) < 1e-9
    assert abs(tam["mabs"] - mabs_tam) < 1e-9

    Uler, chiler, mler = [], [], []
    for tohum in ZINCIR24[:12]:
        m = _wolff_seri(4, T, tohum, 500, 4000)
        Uler.append(binder_cumulant(m))
        chiler.append(susceptibility(m, L=4, T=T))
        mler.append(float(m.mean()))

    assert abs(np.mean(Uler) - U_tam) < 4 * _se(Uler), (
        f"U: tam {U_tam}, MC {np.mean(Uler):.9f} ± {_se(Uler):.9f}"
    )
    assert abs(np.mean(chiler) - chi_tam) < 4 * _se(chiler), (
        f"χ: tam {chi_tam}, MC {np.mean(chiler):.9f} ± {_se(chiler):.9f}"
    )
    assert abs(np.mean(mler) - mabs_tam) < 4 * _se(mler)


@pytest.mark.parametrize(
    "T, C_tam", [(2.0, 0.605532657), (TC, 0.783266826), (3.5, 0.375755537)]
)
def test_ISI_SIGASI_TAM_SAYIMLA_ucusuyor(T, C_tam):
    """SINIF: DAİRESEL DEĞİL — TAM sayım. PENCERE: L = 4, Metropolis,
    n_burn = 500, N = 1200, 12 tohum (20260824 + 1000·k). Ölçülen call
    süresi 1.5 s — hızlı şeritte, eşiğin (2.0 s) altında.

    Isı sığası Metropolis'ten gelmek ZORUNDA: Wolff `energy_series = None`
    döner (`test_OLCULMEMIS_ENERJI_SIFIR_SAYILMIYOR` onu kilitliyor) ve
    Metropolis L = 4'te sweep başına ~75 µs ile Wolff'un küme çevirmesinden
    4 kat pahalıdır. Zincir sayısı bu yüzden 12'de, seri 1200'de tutuldu.

    BEDELİ ÖLÇÜLDÜ ve BEYAN EDİLİYOR: aynı çıpa 24 zincir × 20000 ölçümle
    koşulduğunda sapmalar 0.74 / 1.25 / 1.69 × SE ve SE ~4 kat dardı
    (T_c'de 0.00215 vs 0.00880). Bu bir GEVŞETME DEĞİL — sınır yine hata
    çubuğundan geliyor ve çubuk kısa zincirde GERÇEKTEN daha geniş.

    ÖLÇÜLDÜ (bu pencerede): T = 2.0 → 0.602308898 ± 0.015851 (0.20×SE),
    T_c → 0.780450671 ± 0.008802 (0.32×SE), T = 3.5 → 0.385049216 ±
    0.006434 (1.44×SE).
    """
    assert abs(_tam_sayim(4, T)["C"] - C_tam) < 1e-9
    Cler = [
        heat_capacity(_metro_kosum(4, T, tohum, 500, 1200).energy_series, L=4, T=T)
        for tohum in ZINCIR24[:12]
    ]
    assert abs(np.mean(Cler) - C_tam) < 4 * _se(Cler), (
        f"C: tam {C_tam}, MC {np.mean(Cler):.9f} ± {_se(Cler):.9f}"
    )


def test_BINDER_GAUSSTA_SIFIR_TAM_DUZENLIDE_UCTE_IKI():
    """SINIF: DAİRESEL DEĞİL (cebir). PENCERE: numpy `default_rng(20260824)`,
    n = 2_000_000 Gauss örneği; ve sabit dizi.

    Gauss'ta ⟨m⁴⟩ = 3⟨m²⟩² olduğu için U = 0; tam düzenli (m sabit) durumda
    ⟨m⁴⟩ = ⟨m²⟩² olduğu için U = 1 − 1/3 = 2/3. İkisi de MC'ye hiç
    sorulmadan `3` çarpanını kilitler.

    ÖLÇÜLDÜ: Gauss n = 10⁴ → +0.037629249, n = 2·10⁵ → +0.001511486,
    n = 2·10⁶ → −0.001646572; yani 1/√n ile 0'a gidiyor. Sabit dizi
    0.666666666667 (TAM).

    Binder ÇİFT momentlerden kurulur, dolayısıyla işaretli `m` ile `|m|`
    AYNI sayıyı verir — bu, motorun `|m|` taşıyan serisinin doğrudan
    verilebilmesinin gerekçesidir.
    """
    g = np.random.default_rng(TOHUM).normal(size=2_000_000)
    assert abs(binder_cumulant(g)) < 0.005, binder_cumulant(g)
    assert abs(binder_cumulant(np.abs(g)) - binder_cumulant(g)) < 1e-12

    sabit = np.full(1000, 0.37)
    assert abs(binder_cumulant(sabit) - 2.0 / 3.0) < 1e-12


def test_SUSEPTIBILITE_MUTLAK_m_KULLANIYOR_ve_ISARETLI_m_COK_FARKLI():
    """SINIF: transkripsiyon denetimi + BEYAN. PENCERE: L = 4 TAM SAYIM
    (65536 durum), T ∈ {2.0, T_c, 3.5}. RNG YOK.

    `susceptibility` kendisine VERİLEN seriden `L²(⟨x²⟩ − ⟨x⟩²)/T` hesaplar.
    Motorun `magnetisation_series`i `|m|` taşır, dolayısıyla pratikte
    hesaplanan `χ' = L²(⟨m²⟩ − ⟨|m|⟩²)/T`dir. Sonlu kafeste bu, işaretli
    `χ = L²(⟨m²⟩ − ⟨m⟩²)/T`den AYRI BİR GÖZLEMLENEBİLİRDİR ve fark küçük
    değildir — TAM SAYIMDAN ölçüldü:

        T        χ(|m|)        χ(m)        oran
        2.000   0.195720     6.951373    35.52
        T_c     0.347321     5.368333    15.46
        3.500   0.383037     1.473647     3.85

    Sebep: sonlu kafeste ⟨m⟩ simetriden ötürü 0'a gider ve χ(m) mıknatısın
    +m ile −m arasındaki TÜNELLEMESİNİ ölçer, denge duyarlılığını değil.
    Bu yüzden `⟨|m|⟩` kullanılır; seçim burada AÇIKÇA BEYAN EDİLİYOR.

    Ayrıca `L²` katsayısı burada kilitleniyor: `susceptibility`ye tam
    sayımdan gelen ağırlıklı seri değil, aynı momentleri veren bir seri
    verilirse `L` yazan bir sürüm TAM 4 kat sapar (L = 4).
    """
    for T, oran_beklenen in ((2.0, 35.52), (TC, 15.46), (3.5, 3.85)):
        tam = _tam_sayim(4, T)
        oran = tam["chi_isaretli"] / tam["chi_abs"]
        assert abs(oran - oran_beklenen) < 0.01 * oran_beklenen, (T, oran)

    # `L²` katsayısının doğrudan kilidi: varyansı ELDE bilinen bir seri.
    seri = np.array([0.2, 0.4, 0.2, 0.4])  # varyans (ddof=0) tam 0.01
    assert abs(susceptibility(seri, L=4, T=2.0) - 16 * 0.01 / 2.0) < 1e-15
    assert abs(heat_capacity(seri, L=4, T=2.0) - 0.01 / (16 * 4.0)) < 1e-18


# ==========================================================================
# 4. TAHMİNCİNİN KIRILDIĞI BÖLGELER
# ==========================================================================


def test_OLCULMEMIS_ENERJI_SIFIR_SAYILMIYOR():
    """SINIF: transkripsiyon denetimi — ama TAŞIYICI.

    Wolff `energy_series = None` döner (`SamplingResult` dokstringi: `None`
    "ÖLÇÜLMEDİ" demektir, SIFIR değil). Isı sığası bunu sıfır sanarsa
    "C = 0" diye okunur, oysa gerçek hüküm "ölçülmedi".

    Bu, Modül 1B'nin kararlılık alanı dersinin aynısı: analiz edilmedi ile
    kararlı aynı değildir ve aynı değerle gösterilmemelidir.

    PENCERE: L = 8, T = T_c, Wolff, tohum 20260824, n_burn = 100, N = 200 —
    kapsam sınırının kendisi sınandığı için kısa koşum yeterli.
    """
    w = run_wolff(
        SpinState.uniform(8, 1), temperature=TC, coupling=1.0, field=0.0,
        rng=np.random.default_rng(TOHUM), n_sweep=200, n_burn=100,
    )
    assert w.energy_series is None

    with pytest.raises(InvalidInputError, match="olculmedi"):
        heat_capacity(w.energy_series, L=8, T=TC)
    with pytest.raises(InvalidInputError, match="olculmedi"):
        heat_capacity(None, L=16, T=2.0)


def test_KORELASYON_UZUNLUGU_KIRILDIGINDA_None_donuyor():
    """SINIF: DAİRESEL DEĞİL (cebir) + ÖLÇÜM.

    `S(0)/S(k_min) − 1` NEGATİF çıkabilir; o zaman ξ TANIMSIZDIR ve fonksiyon
    `None` döner. `max(oran, 0)` ile kırpmak YASAK: sıfır "korelasyon yok"
    diye okunur, oysa gerçek hüküm "bu tahminci burada bir şey söylemez".
    İkisi çok farklı iddiadır.

    CEBİRSEL DURUM: dama tahtası (antiferromanyetik) kafeste Σs = 0 olduğu
    için S(0) TAM SIFIRDIR, oran TAM −1'dir. Tohum yok, gürültü yok.

    ÖLÇÜLEN DURUM: tamamen İLİŞKİSİZ ±1 kafes (gerçek ξ = 0), 200 tohum
    (10000 + k):
        L = 16 → 123/200 tohum `None`; kalan 77'nin ortalama ξ'si 4.418
        L = 32 → 124/200 tohum `None`; kalan 76'nın ortalama ξ'si 7.791
    Yani gerçek ξ = 0 iken tahminci yarı yarıya `None`, yarı yarıya SAHTE
    bir 4-8 üretiyor. Bu, `None`un neden 0'a kırpılamayacağının ölçüsüdür:
    kırpılsaydı bu kafeslerin YARISI "ξ = 0, doğru!" diye okunurdu ve doğru
    olması TESADÜF olurdu.

    İLK NEGATİF TOHUMLAR (deterministik): L = 16 tohum 10000 → oran
    −0.969563713; L = 32 tohum 10000 → oran −1.000000000 (o kafeste Σs tam
    0 çıkmış).

    PENCERE (pozitif taraf): Ising L = 32, T = T_c, Wolff, tohum 909,
    n_burn = 500, N = 200.
    """
    # --- cebirsel: dama tahtası, oran TAM -1 ---
    L = 8
    i, j = np.indices((L, L))
    dama = ((-1.0) ** ((i + j) % 2))[..., None]
    assert correlation_length(dama) is None

    # --- ölçülen: ilişkisiz kafesler ---
    assert correlation_length(_beyaz_kafes(16, 10000)) is None
    assert correlation_length(_beyaz_kafes(32, 10000)) is None
    none_sayisi = sum(
        correlation_length(_beyaz_kafes(16, 10000 + k)) is None for k in range(200)
    )
    assert none_sayisi == 123, none_sayisi

    # --- ve gerçekten uzun korelasyonlu bir kafeste bir şey SÖYLÜYOR ---
    tc_konfig = run_wolff(
        SpinState.uniform(32, 1), temperature=TC, coupling=1.0, field=0.0,
        rng=np.random.default_rng(909), n_sweep=200, n_burn=500,
    ).final_state.spins
    xi = correlation_length(tc_konfig)
    assert xi is not None and xi > 5.0, xi


def test_XI_DUZENLI_FAZDA_MIKNATISLANMAYI_OKUYOR_ve_bunu_BEYAN_EDIYORUZ():
    """SINIF: DÜRÜST OLUMSUZ — tahmincinin GEÇERSİZ olduğu bölge ölçülüyor.
    PENCERE: Ising L = 32, Wolff, n_burn = 300, N = 100, tohumlar 909 + 77·k
    (k < 4); T = 1.5, yani T_c'nin %34 altı. Ölçülen call süresi 1.4 s.

    İkinci moment tahmincisi `S(0) = |Σs|²` kullanır ve DÜZENLİ FAZDA bu,
    mıknatıslanmanın karesiyle DOMİNE OLUR. ξ o zaman korelasyonu değil
    ⟨m⟩'yi okur ve kafesten büyük çıkar.

    ÖLÇÜLDÜ (bu pencere, tek konfigürasyon): ξ = 1513.064, 996.211, 894.439,
    1286.127 — yani ξ/L 28 ile 47 arasında.

    TOPLULUK ORTALAMASI SORUNU ÇÖZMÜYOR, yalnızca yanlış sayıyı
    kesinleştiriyor. Ayrı bir pencerede ölçüldü (L = 32, T = 1.5, Wolff,
    n_burn = 500, 200 konfigürasyon 5 küme çevirmesi arayla, tohumlar
    909 + 77·k): ξ = 832.207, 832.975, 850.974, 837.174, 812.367 — saçılım
    daralıyor ama değer hâlâ kafesin 26 katı. L = 32'lik bir kafeste ξ = 830
    bir korelasyon uzunluğu DEĞİLDİR; tahmincinin geçerlilik penceresi
    T ≳ T_c'dir ve bu dokstringde beyan ediliyor.

    'BAĞLI' (connected) SÜRÜM NEDEN YOK — ÖLÇÜLDÜ ve CEBİRSEL: tek bir
    konfigürasyondan site ortalamasını çıkarmak Σ(s − s̄) = 0 yapar, yani
    S(0) TAM SIFIR olur ve oran HER ZAMAN tam −1 çıkar. Ölçüldü (L = 32,
    Wolff, n_burn = 500, N = 200, tohumlar 777 + 100·k): T = 1.5, T_c ve
    3.5'te 5'er tohumun 15'inde de oran = −1.000000e+00, yani bağlı sürüm
    HER KONFİGÜRASYONDA `None` döndürür ve hiçbir şey söylemez. 'Bağlı'
    için ⟨s⟩ TOPLULUK ortalaması olmak zorundadır; tek konfigürasyonda
    böyle bir şey yoktur.
    """
    for k in range(4):
        soguk = run_wolff(
            SpinState.uniform(32, 1), temperature=1.5, coupling=1.0, field=0.0,
            rng=np.random.default_rng(909 + 77 * k), n_sweep=100, n_burn=300,
        ).final_state.spins
        xi = correlation_length(soguk)
        assert xi is not None
        assert xi / 32 > 20.0, f"düzenli fazda ξ/L kafesten çok büyük: {xi / 32}"

        # Bağlı sürüm cebirsel olarak ÖLÜ: site ortalaması çıkarılınca S(0)=0.
        bagli = soguk - soguk.mean(axis=(0, 1), keepdims=True)
        assert correlation_length(bagli) is None


def test_TOPLULUK_ORTALAMASI_TEK_KONFIGURASYONDAN_KESKIN():
    """SINIF: ÖLÇÜM — 'tek akıştan okunan oran paydası kadar gürültülüdür'.
    PENCERE: Ising L = 16, T = T_c, Wolff; topluluk 40 BAĞIMSIZ koşumun son
    konfigürasyonu (tohumlar 909 + 7·k, k < 40), n_burn = 200, N = 1.
    Ölçülen call süresi 1.2 s.

    `S(k)` bir TOPLULUK ortalamasıdır. Tek konfigürasyondan okunan oran,
    paydası (`S(k_min)`) kadar gürültülüdür ve ξ tohumdan tohuma savrulur.

    ÖLÇÜLDÜ (bu pencerede): topluluk ξ = 14.2272 (ξ/L = 0.8892); tek
    konfigürasyonlar 1/40 `None`, kalanların aralığı 4.948 … 99.161 — yani
    yayılım topluluk değerinin 6.62 KATI. Aynı ölçüm L = 32, T_c'de daha da
    keskin: TEK konfigürasyonda ξ = 3.644, 17.735, 30.337, 36.768, 150.562
    (41 kat yayılım), 200 konfigürasyonluk toplulukta 29.437, 30.773,
    26.668, 30.975, 33.078 (%11 yayılım).

    Bu yüzden `correlation_length` `(n_config, L, L, n_component)` şeklini de
    kabul eder ve yapı faktörünü ÖNCE ortalar, ORANI SONRA kurar. (Planın
    imzası yalnız tek konfigürasyon alıyordu; bu bir EKLEMEDİR ve gerekçesi
    yukarıdaki yayılım farkıdır.)
    """
    konfigler = np.array(
        [
            run_wolff(
                SpinState.uniform(16, 1), temperature=TC, coupling=1.0, field=0.0,
                rng=np.random.default_rng(909 + 7 * k), n_sweep=1, n_burn=200,
            ).final_state.spins
            for k in range(40)
        ]
    )
    tekler = [correlation_length(c) for c in konfigler]
    gecerli = [x for x in tekler if x is not None]
    toplu = correlation_length(konfigler)

    assert toplu is not None
    assert 0.4 < toplu / 16 < 2.0, f"T_c'de ξ/L ~ 1 beklenir: {toplu / 16}"
    assert (max(gecerli) - min(gecerli)) / toplu > 3.0


def test_HEISENBERG_YUKSEK_Tde_TAHMINCI_TOHUMDAN_TOHUMA_KIRILIYOR():
    """SINIF: DÜRÜST OLUMSUZ + PLAN DÜZELTMESİ.
    PENCERE: Heisenberg L = 16, Metropolis (step = 0.6), n_burn = 400,
    N = 200, 5 tohum (555 + 100·k), T ∈ {1.0, 1.4, 1.8}. Ölçülen call
    süresi 0.9 s.

    PLAN BAYAT ÇIKTI. Plan (E tablosu) "Heisenberg T ≥ 1.0'da oran
    −0.0511 / −0.0125 / −0.0383, ÜÇÜ DE None" diyor — yani hükmü kesin bir
    işaret iddiası gibi kuruyor. Yeniden ölçüldü, iki ayrı pencerede:

        L = 32, n_burn = 800, N = 400, tohum 555 + 100·k:
            T = 1.0  −0.37590, +1.39614, +3.24936, +0.27039, −0.79288 → 2/5 None
            T = 1.4  +0.72027, +0.66235, +0.65145, −0.40021, −0.30780 → 2/5 None
            T = 1.8  +0.12789, −0.31965, +1.64226, −0.19789, +0.64164 → 2/5 None
        L = 16, n_burn = 400, N = 200 (bu testin penceresi):
            T = 1.0 → 1/5 None,  T = 1.4 → 2/5 None,  T = 1.8 → 2/5 None

    İşaret TOHUMDAN TOHUMA DEĞİŞİYOR. Planın üç negatif sayısı bir tohumun
    şansıydı; doğru hüküm "orada oran negatiftir" değil, "orada tahminci
    kararsızdır ve verdiği işaret gürültüdür". Test bu ZAYIF ama DOĞRU
    hükmü kilitler: her üç sıcaklıkta da 5 tohumun en az biri `None`, en az
    biri sayı döndürür.
    """
    for T in (1.0, 1.4, 1.8):
        sonuclar = [
            correlation_length(
                run_metropolis(
                    SpinState.uniform(16, 3), HeisenbergModel(), temperature=T,
                    coupling=1.0, field=0.0,
                    rng=np.random.default_rng(555 + 100 * k),
                    n_sweep=200, n_burn=400,
                ).final_state.spins
            )
            for k in range(5)
        ]
        assert any(x is None for x in sonuclar), (T, sonuclar)
        assert any(x is not None for x in sonuclar), (T, sonuclar)


# ==========================================================================
# 5. YAPISAL KİLİTLER ve GİRDİ DENETİMİ
# ==========================================================================


def test_BinningResult_DONDURULMUS_ve_ALANLARI_TUTARLI():
    """SINIF: transkripsiyon denetimi. PENCERE: AR(1) φ = 0.8, N = 8000,
    tohum 20260824.

    `block_sizes` 1'den başlayıp İKİYE KATLANIR, `errors` onunla aynı
    uzunluktadır, `error` `errors`in maksimumudur ve `tau_int`
    `½(error/errors[0])²`dir. Alanlar birbirinden türetilebilir olmalı,
    yoksa `BinningResult` bir kayıt değil bir iddia yığını olur.
    """
    sonuc = binning_error(_ar1(8000, 0.8, TOHUM))
    assert isinstance(sonuc, BinningResult)
    assert sonuc.block_sizes == (1, 2, 4, 8, 16, 32, 64, 128)
    assert len(sonuc.errors) == len(sonuc.block_sizes)
    assert sonuc.error == max(sonuc.errors)
    assert abs(sonuc.tau_int - 0.5 * (sonuc.error / sonuc.errors[0]) ** 2) < 1e-12
    with pytest.raises(AttributeError):
        sonuc.error = 1.0  # type: ignore[misc]


def test_BLOK_SAYISI_min_blocks_ALTINA_DUSMUYOR():
    """SINIF: transkripsiyon denetimi. PENCERE: N = 1000, `min_blocks` = 32
    ve 64; AR(1) φ = 0, tohum 20260824.

    Son blok boyutunda blok sayısı `min_blocks`un ALTINA düşerse plato
    ölçütünün dayandığı 1/√(2(K−1)) saçılımı kontrolsüz büyür.
    """
    s = _ar1(1000, 0.0, TOHUM)
    for mb in (32, 64):
        sonuc = binning_error(s, min_blocks=mb)
        assert all(1000 // b >= mb for b in sonuc.block_sizes)
        assert 1000 // (sonuc.block_sizes[-1] * 2) < mb


@pytest.mark.parametrize(
    "fonksiyon, arglar",
    [(binder_cumulant, ()), (susceptibility, (8, 2.0)), (heat_capacity, (8, 2.0))],
)
def test_OLCULMEMIS_SERI_HER_YERDE_REDDEDILIYOR(fonksiyon, arglar):
    """SINIF: transkripsiyon denetimi. `None` "ölçülmedi" demektir ve
    hiçbir gözlemlenebilir onu sessizce sıfır saymaz."""
    with pytest.raises(InvalidInputError, match="olculmedi"):
        fonksiyon(None, *arglar)


@pytest.mark.parametrize("gecersiz_T", [0.0, -1.0, np.inf, np.nan])
def test_GECERSIZ_SICAKLIK_reddediliyor(gecersiz_T):
    """SINIF: transkripsiyon denetimi. T bölende; sıfır veya sonsuz sıcaklık
    sessizce inf/nan döndürmemeli."""
    m = np.array([0.5, 0.4, 0.6, 0.5])
    with pytest.raises(InvalidInputError):
        susceptibility(m, L=4, T=gecersiz_T)
    with pytest.raises(InvalidInputError):
        heat_capacity(np.array([-1.0, -2.0, -1.5, -1.2]), L=4, T=gecersiz_T)


@pytest.mark.parametrize("gecersiz_L", [0, -3])
def test_GECERSIZ_L_reddediliyor(gecersiz_L):
    m = np.array([0.5, 0.4, 0.6, 0.5])
    with pytest.raises(InvalidInputError):
        susceptibility(m, L=gecersiz_L, T=2.0)
    with pytest.raises(InvalidInputError):
        heat_capacity(m, L=gecersiz_L, T=2.0)


def test_BOS_veya_SONSUZ_SERI_reddediliyor():
    """SINIF: transkripsiyon denetimi. Boş seri ortalamasında `nan` doğar;
    `nan`lı bir hata çubuğu sessizce raporlanamaz."""
    with pytest.raises(InvalidInputError):
        binder_cumulant(np.array([]))
    with pytest.raises(InvalidInputError):
        susceptibility(np.array([0.1, np.nan, 0.2]), L=4, T=2.0)
    with pytest.raises(InvalidInputError):
        binning_error(np.array([1.0, 2.0, 3.0]))  # min_blocks = 32'nin altında
    with pytest.raises(InvalidInputError):
        binning_error(np.ones((4, 4)))  # 1B olmalı
    with pytest.raises(InvalidInputError):
        binning_error(np.ones(100), min_blocks=2)  # plato ölçütü üç nokta ister


def test_BINDER_SIFIR_MOMENTTE_HATA_veriyor():
    """SINIF: transkripsiyon denetimi. Tümüyle sıfır bir seride ⟨m²⟩ = 0 ve
    U tanımsızdır; 0/0 sessizce `nan` dönmemeli."""
    with pytest.raises(InvalidInputError):
        binder_cumulant(np.zeros(100))


def test_KORELASYON_UZUNLUGU_SEKIL_DENETIMI():
    """SINIF: transkripsiyon denetimi. Kabul edilen iki şekil:
    `(L, L, n_component)` ve `(n_config, L, L, n_component)`."""
    with pytest.raises(InvalidInputError):
        correlation_length(np.ones((8, 8)))
    with pytest.raises(InvalidInputError):
        correlation_length(np.ones((8, 6, 1)))
    with pytest.raises(InvalidInputError):
        correlation_length(np.ones((2, 2, 1)))  # L < 3: k_min ile Nyquist çakışır


# --------------------------------------------------------------------------
# `slow` İŞARETİNİN GEREKÇESİ — ÖLÇÜM, DEVRALINAN EŞİK DEĞİL
# --------------------------------------------------------------------------
#
# Depoda devralinabilir bir esik YOKTUR; `test_metropolis.py` esigi 22.08'de
# yeniden olcup "kendi call suresi 2.0 s ve ustunde olan durum `slow`" diye
# sabitledi ve o kural burada da uygulandi. Bu dosyanin OLCULEN call sureleri
# (`--durations`, 23.08.2026, bu makinede):
#
#     5.38 s  planin D penceresi L=32 T_c            <- slow
#     4.56 s  naif hata, 24 bagimsiz zincir L=16     <- slow
#     3.65 s  tau_int capraz kontrolu, gercek MC     <- slow
#     1.41 s  isi sigasi tam sayim T=3.5             <- ISARETSIZ
#     1.39 s  isi sigasi tam sayim T_c               <- ISARETSIZ
#     1.38 s  isi sigasi tam sayim T=2.0             <- ISARETSIZ
#     1.36 s  xi duzenli fazda                       <- ISARETSIZ
#     1.16 s  topluluk vs tek konfigurasyon          <- ISARETSIZ
#     0.99 s  Binder+chi tam sayim T=2.0             <- ISARETSIZ
#     0.93 s  Binder+chi tam sayim T_c               <- ISARETSIZ
#     0.90 s  Heisenberg kirilmasi                   <- ISARETSIZ
#     0.61 s  Binder+chi tam sayim T=3.5             <- ISARETSIZ
#     < 0.5 s kalan 27 durum
#
# Esigi gecen HER durum isaretlendi, gecmeyen HICBIRI isaretlenmedi.
#
# TASIYICI IDDIALARIN HICBIRI TAMAMEN YAVAS SERITTE DEGIL, ve bu bilincli:
#   - "naif SEM yaniliyor" hizli seritte AR(1)'in KAPALI FORMUYLA (gercek
#     SE = sigma*sqrt(2 tau/N)) ispatlaniyor -- MC referansindan DAHA GUCLU
#     bir dayanak, ve 0.03 s. Yavas serit ayni hukmu gercek Ising verisinde
#     24 bagimsiz zincirle tekrarliyor.
#   - "plato yoksa alt sinir" hizli seritte hem AR(1) hem kisa MC ile
#     kuruluyor (0.46 s); yavas serit onu planin L = 32 penceresinde
#     tekrarliyor.
#   - tau_int capraz kontrolu hizli seritte AR(1) uzerinde UC yolla
#     (binning / ACF / kapali form) kuruluyor; yavas serit gercek MC.
# Bedeli olculdu: hizli seritteki MC pencereleri daha kisa oldugu icin
# ORADA mutlak hata cubugu bilinmez ve hukum YONSELDIR; bu, ilgili
# dokstringlerde acikca yazilidir.
#
# tests/spin/ serit dagilimi bu dosyayla birlikte OLCULDU:
#   -m "not slow"  36 test  12.51 s  (bu dosya)
#   -m slow         3 test  13.59 s  (bu dosya)
