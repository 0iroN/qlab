"""Wolff küme algoritması — ve İKİ ALGORİTMANIN ÇAPRAZ KONTROLÜ.

Bu dosya üç ayrı iş yapar:

1. ÇAPRAZ KONTROL (sınıf: DAİRESEL DEĞİL) — Wolff ile Metropolis aynı
   `⟨|m|⟩`yi veriyor mu. İki kod yolunun ORTAK HİÇBİR KARAR NOKTASI yoktur:
   Wolff hiçbir tek-spin kabul/red kararı vermez, Metropolis hiçbir küme
   kurmaz. Aynı sayıyı vermeleri, ikisinin de doğru dağılımı örneklediğini
   söyler. Bu, Plan F'deki `bloch_hamiltonian` ↔ `finite_ring_matrix`
   deseninin aynısıdır.

2. MUTLAK ÇIPA (sınıf: DAİRESEL DEĞİL) — L = 3 ve L = 4 kafeslerin BÜTÜN
   durumları (512 ve 65536) sayılır ve `⟨|m|⟩` Boltzmann ağırlığıyla TAM
   olarak hesaplanır. Çapraz kontrol tek başına "iki algoritma da aynı
   şekilde bozuk" ihtimalini dışlayamaz; tam sayım dışlar.

3. YAPISAL KİLİTLER (sınıf: transkripsiyon denetimi) — kümenin gerçekten
   tohumun BAĞLI AYNI-İŞARETLİ bölgesi olduğu, T → 0'da tüm kafesi yuttuğu,
   T → ∞'da tek siteye indiği, ve üç kapsam sınırının sessiz olmadığı.

ÇAPRAZ KONTROLÜN PENCERESİ — BEYAN EDİLİYOR (bu depoda "pencere
adlandırılmazsa hüküm pencerenin dışına taşar" dersi altı kez çıktı):

    T_c = 2/ln(1+√2) = 2.269185314 (Onsager; hiçbir yere GİRDİ değil)

    L    T           başlangıç   metro(n_burn,n_sweep)   wolff(n_burn,n_sweep)
    8    1.800000    soğuk       (400, 1200)             (200, 900)
    8    2.269185    soğuk       (400, 1200)             (200, 900)
    8    2.800000    soğuk       (400, 1200)             (200, 900)
    16   2.269185    soğuk       (1000, 2000)            (400, 1500)
    16   2.800000    SICAK       (500, 1500)             (200, 1000)
    8    2.269185    soğuk       (200, 500)              (100, 400)   ← hızlı şerit

Her satır K = 24 BAĞIMSIZ tohumlu zincirdir; tohumlar `20260823 + 1000·k`,
k = 0…23. Tahminci zincir ortalamalarının ortalamasıdır, hata çubuğu zincir
ortalamalarının standart hatasıdır. Bu, zincir-İÇİ otokorelasyondan
BAĞIMSIZDIR ve bu modülde şart: Metropolis'in binning eğrisi plato yapmıyor
(planın D tablosu: L=32, T_c'de `τ_int` = 53.78, naif SEM 10.4 kat yanlış),
yani ardışık örnekleri bağımsız sayan hiçbir hata tahmini kullanılamaz.

SINIR ELLE SEÇİLMEDİ: |ŵ − m̂| < 3·√(SE_w² + SE_m²).

K = 24 NEDEN — ÖLÇÜLDÜ, DEVRALINMADI. Bu depoda SE tahmincisinin KENDİSİNİN
gürültülü olduğu Task 2'de pahalıya öğrenildi (6 zincirle 5.52×SE "yanlılık",
32 zincirle yanlılık yok). Aynı tuzak burada da ÖLÇÜLDÜ — zincir
ortalamalarının std tahmini K ile şöyle geziniyor:

    zincir ortalamalarının std'si          K =    8     12     16     20     24
    L = 8, T = 1.8, uzun zincirler, metro  ×10⁻³  2.07   2.93   2.70   2.49   3.57
    L = 8, T = 1.8, uzun zincirler, wolff  ×10⁻³  1.67   1.70   1.64   1.70   2.39
    L = 8, T_c,     kısa zincirler, metro  ×10⁻²  2.64   2.18   2.15   2.01   2.11
    L = 8, T_c,     kısa zincirler, wolff  ×10⁻²  0.82   0.95   1.12   1.08   1.36

Hiçbir satır "oturmuş" değil ve olması da beklenmez: std tahmininin bağıl
saçılımı ≈ 1/√(2(K−1)) — K = 8 için %26.7, K = 24 için %14.7. ÖNEMLİ OLAN
YÖN: dört satırın üçünde K = 8 tahmini K = 24'ünkinden KÜÇÜK çıkıyor
(uzun-metro 2.07 vs 3.57, yani %42 daha dar; kısa-wolff 0.82 vs 1.36, %40
daha dar). Yani K = 8 ile kurulan bir 3σ sınırı sistematik olarak FAZLA DAR
olur ve gürültüyü "gerileme" diye raporlar. Task 2'nin
5.52×SE yanlış alarmı tam olarak buydu. K = 24, ölçülen dalgalanmanın
tepesini (3.57e-03, 1.36e-02) kullanır ve saçılımı %14.7'ye indirir. Daha
büyük K'nın bedeli doğrudan test süresidir ve zincirler bağımsız koştuğu
için K'da DOĞRUSALDIR: altı çapraz kontrol satırı K = 24'te 29.8 s sürüyor,
K = 32'de 39.7 s sürerdi — +9.9 s karşılığında saçılım %14.7'den yalnızca
%12.6'ya iner.

ADIM EŞİTLİĞİ İDDİA EDİLMİYOR: Wolff'un bir adımı BİR KÜME ÇEVİRMESİ,
Metropolis'in bir sweep'i L² SİTE GÜNCELLEMESİDİR. Karşılaştırma eşit adımda
değil, ikisi de dengeye oturmuş uzunlukta yapılır (yukarıdaki tablo).

ÖLÇÜLEN DÜRÜST OLUMSUZ — VE İKİ AŞAMALI DERSİ:

L = 16, T = 2.8'de SOĞUK başlangıç ve kısa zincirlerle (metro (1000, 2000),
wolff (400, 1500)) iki tahminci ayrışıyor gibi göründü. İLK ölçüm turunda
fark 3σ sınırının K = 12'de 0.872'si, K = 16'da 0.804'ü, K = 24'te 0.555'i
idi — sınırı geçmiyor ama işaret her K'da aynı yönde. Sebep arandı ve İKİ
şey bulundu, biri gerçek biri değil:

(a) GERÇEK OLAN — METROPOLIS O NOKTADA DENGEYE OTURMAMIŞ. Aynı Metropolis
    kodu, aynı tohumlar: (1000, 2000) → 0.218639 (SE 1.461e-03), ama
    (4000, 8000), K = 32 → 0.214438 (SE 6.260e-04). Yani kısa koşum
    +4.2e-03 (2.6σ) YUKARI yanlı; soğuk başlangıçtan gelen artık düzen
    T_c'nin yalnız %23 üstünde hâlâ sönmemiş. Wolff aynı noktada kısa
    koşumda ZATEN oturmuştu: (400, 1500) → 0.217129, (2000, 6000), K = 32 →
    0.216382 ± 8.653e-04.

(b) GERÇEK OLMAYAN — ORANIN KENDİSİ GÜRÜLTÜLÜ. Kümeyi büyüten iç döngünün
    rastgele akışı değiştirilince (bkz. `wolff.py`, çürütülen "toplu rng"
    iyileştirmesi) AYNI parametrelerdeki oran 0.872 → 0.242 (K = 12),
    0.804 → 0.274 (K = 16), 0.555 → 0.221 (K = 24) oldu. Algoritma
    değişmedi, yalnız akış değişti. DERS: tek bir akıştan okunan bir
    "fark/sınır" oranı, sınırın kendisi kadar gürültülüdür; tekrarlanabilir
    olan (a)'ydı çünkü Metropolis kodu iki turda da aynıydı.

Bu yüzden o satır testte SICAK (rastgele ±1) başlangıçla koşuyor: T > T_c'de
sıcak başlangıç dengeye yakındır, Metropolis'in artık düzeni yoktur ve oran
0.208'e iner. Gevşetme DEĞİL — aynı iddia, doğru başlangıç koşuluyla.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.spin.lattice import SpinState
from qlab.engine.spin.metropolis import run_metropolis
from qlab.engine.spin.models import HeisenbergModel, IsingModel
from qlab.engine.spin.wolff import _kume_buyut, run_wolff
from qlab.errors import InvalidInputError, UnsupportedSystemError

TOHUM = 20260823
TC = 2.0 / np.log(1.0 + np.sqrt(2.0))
ZINCIR24 = tuple(TOHUM + 1000 * k for k in range(24))
ZINCIR16 = ZINCIR24[:16]


# --------------------------------------------------------------------------
# Yardimcilar
# --------------------------------------------------------------------------


def _tam_sayim(L: int, temperature: float, coupling: float) -> float:
    """2^(L·L) durumun HEPSİNİ sayar; `⟨|m|⟩` döndürür.

    Enerji BAĞIMSIZ yazılır: her bağ TAM BİR KEZ (her sitenin sağ ve aşağı
    komşusu). `lattice.neighbour_sum` + 0.5 düzeltmesi KULLANILMAZ. L ≥ 3
    gerekir; L = 2'de `np.roll(-1)` ile `np.roll(+1)` çakışır.
    """
    n = L * L
    idx = np.arange(2**n, dtype=np.int64)
    bitler = (idx[:, None] >> np.arange(n)) & 1
    s = (2 * bitler - 1).astype(float).reshape(-1, L, L)
    bag = np.sum(s * np.roll(s, -1, axis=1), axis=(1, 2)) + np.sum(
        s * np.roll(s, -1, axis=2), axis=(1, 2)
    )
    enerji = -coupling * bag
    mutlak_m = np.abs(s.mean(axis=(1, 2)))
    # Emin cikarilmasi TASMA korumasidir; oran degismez.
    agirlik = np.exp(-(enerji - enerji.min()) / temperature)
    return float((agirlik * mutlak_m).sum() / agirlik.sum())


def _se(v: np.ndarray) -> float:
    return float(np.std(v, ddof=1) / np.sqrt(len(v)))


def _baslangic(L: int, tohum: int, sicak: bool) -> SpinState:
    """Soğuk = tekbiçim; sıcak = tohuma bağlı rastgele ±1 kafes."""
    if not sicak:
        return SpinState.uniform(L=L, n_component=1)
    r = np.random.default_rng(tohum ^ 0x5EED)
    return SpinState(2.0 * r.integers(0, 2, (L, L, 1)) - 1.0)


def _seritli_kafes(L: int) -> SpinState:
    """Satırları dönüşümlü ±1 — L çift olduğunda 3 ayrık `+1`, 3 ayrık `−1`
    bileşen verir (satırlar yatayda halka, dikeyde zıt işaretle çevrili)."""
    spins = np.empty((L, L, 1))
    for i in range(L):
        spins[i, :, 0] = 1.0 if i % 2 == 0 else -1.0
    return SpinState(spins)


# --------------------------------------------------------------------------
# 1. CAPRAZ KONTROL -- TASIYICI TEST
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "L, temperature, sicak, m_nb, m_ns, w_nb, w_ns, metro_ort, wolff_ort, oran",
    [
        # HIZLI SERIT temsilcisi -- kisa zincirler, ayni iddia, genis cubuk.
        (8, TC, False, 200, 500, 100, 400, 0.773513, 0.780911, 0.481),
        pytest.param(
            8, 1.8, False, 400, 1200, 200, 900, 0.957263, 0.956978, 0.109,
            marks=pytest.mark.slow,
        ),
        pytest.param(
            8, TC, False, 400, 1200, 200, 900, 0.776324, 0.780441, 0.393,
            marks=pytest.mark.slow,
        ),
        pytest.param(
            8, 2.8, False, 400, 1200, 200, 900, 0.422572, 0.419061, 0.313,
            marks=pytest.mark.slow,
        ),
        pytest.param(
            16, TC, False, 1000, 2000, 400, 1500, 0.710550, 0.714768, 0.416,
            marks=pytest.mark.slow,
        ),
        pytest.param(
            16, 2.8, True, 500, 1500, 200, 1000, 0.212972, 0.214793, 0.208,
            marks=pytest.mark.slow,
        ),
    ],
)
def test_WOLFF_ve_METROPOLIS_AYNI_dengeyi_veriyor(
    L, temperature, sicak, m_nb, m_ns, w_nb, w_ns, metro_ort, wolff_ort, oran
):
    """SINIF: DAİRESEL DEĞİL — iki BAĞIMSIZ kod yolu, tek hüküm.

    Wolff hiçbir tek-spin kabul/red kararı VERMEZ; Metropolis hiçbir küme
    KURMAZ. Aynı `⟨|m|⟩`yi vermeleri, ikisinin de doğru dağılımı
    örneklediğini söyler.

    SINIR HATA ÇUBUĞUNDAN GELİR, ELLE SEÇİLMEZ: iki tahmincinin farkı, iki
    hata çubuğunun karelerinin toplamının karekökünün ÜÇ katını geçmemeli.
    Hata çubuğu K = 24 bağımsız tohumlu zincirin ortalamalarından çıkar, yani
    zincir-içi otokorelasyondan bağımsızdır (Metropolis'in binning eğrisi
    plato yapmadığı için bu şart).

    ÖLÇÜLDÜ (23.08.2026, bu depoda; tohumlar 20260823 + 1000·k, k = 0…23):

        L   T         başl.   metro       SE          wolff       SE          fark/3σ
        8   2.269185  soğuk   0.773513    4.314e-03   0.780911    2.769e-03   0.481
        8   1.800000  soğuk   0.957263    7.293e-04   0.956978    4.873e-04   0.109
        8   2.269185  soğuk   0.776324    2.899e-03   0.780441    1.939e-03   0.393
        8   2.800000  soğuk   0.422572    2.041e-03   0.419061    3.139e-03   0.313
        16  2.269185  soğuk   0.710550    2.993e-03   0.714768    1.564e-03   0.416
        16  2.800000  SICAK   0.212972    1.976e-03   0.214793    2.146e-03   0.208

    Üç sıcaklık BİLİNÇLİ seçildi: T_c'nin ALTI (1.8, düzenli faz), T_c'ye
    yakın (2.269185, korelasyon uzunluğu L'yi geçiyor ve Metropolis'in
    kritik yavaşlaması burada en kötü), ve ÜSTÜ (2.8, düzensiz faz). Tek
    sıcaklıkta koşup "iki algoritma uyuşuyor" demek, hükmü penceresinin
    dışına taşımak olurdu.

    3σ SINIRININ GÜCÜ ÖLÇÜLDÜ, tahmin edilmedi. `p`deki 2 çarpanı düşerse
    Wolff etkin olarak 2T'yi örnekler; L = 8, T_c, K = 24, (200, 900)
    koşumunda mutant `⟨|m|⟩ = 0.186040` veriyor, oysa doğru kod 0.780441.
    Fark 5.94e-01, o satırın 3σ sınırı 1.046e-02 — yani sınırın 57 KATI.
    Mutantın gerçekten 2T'yi örneklediği ayrıca doğrulandı: DOĞRU kod
    T = 2·T_c'de koşturulunca 0.186040 (SE 2.576e-03) veriyor, mutantla
    BİRE BİR aynı sayı.
    """
    metro = np.array(
        [
            run_metropolis(
                _baslangic(L, t, sicak),
                IsingModel(),
                temperature=temperature,
                coupling=1.0,
                field=0.0,
                rng=np.random.default_rng(t),
                n_sweep=m_ns,
                n_burn=m_nb,
            ).magnetisation_series.mean()
            for t in ZINCIR24
        ]
    )
    wolff = np.array(
        [
            run_wolff(
                _baslangic(L, t, sicak),
                temperature=temperature,
                coupling=1.0,
                field=0.0,
                rng=np.random.default_rng(t),
                n_sweep=w_ns,
                n_burn=w_nb,
            ).magnetisation_series.mean()
            for t in ZINCIR24
        ]
    )

    m_o, m_se = float(metro.mean()), _se(metro)
    w_o, w_se = float(wolff.mean()), _se(wolff)
    fark = abs(w_o - m_o)
    sinir = 3.0 * float(np.hypot(m_se, w_se))

    # FIZIK IDDIASI ONCE: elle secilmis bir tolerans degil, hata cubugu.
    assert fark < sinir, (
        f"L={L} T={temperature:.6f}: Wolff {w_o:.6f} (SE {w_se:.3e}) ile "
        f"Metropolis {m_o:.6f} (SE {m_se:.3e}) ayrisiyor -- "
        f"fark {fark:.3e} = {fark / sinir * 3.0:.2f} sigma"
    )
    # Sonra transkripsiyon denetimi: olculen sayilar tohumlariyla kilitli.
    assert m_o == pytest.approx(metro_ort, abs=1e-5)
    assert w_o == pytest.approx(wolff_ort, abs=1e-5)
    assert fark / sinir == pytest.approx(oran, abs=2e-3)


# --------------------------------------------------------------------------
# 2. MUTLAK CIPA -- TAM SAYIM
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "L, temperature, tam_deger, olculen, se_beklenen, sapma_se",
    [
        (4, 2.0, 0.918943267, 0.921494141, 2.429e-03, 1.05),
        (4, TC, 0.843860445, 0.842158203, 2.598e-03, 0.66),
        (3, 2.0, 0.925984778, 0.925850694, 1.747e-03, 0.08),
        (3, TC, 0.871071935, 0.869722222, 1.723e-03, 0.78),
    ],
)
def test_WOLFF_TAM_SAYIM_CIPASI_ile_ucusuyor(
    L, temperature, tam_deger, olculen, se_beklenen, sapma_se
):
    """SINIF: DAİRESEL DEĞİL — bütün durumlar sayıldı, MC'ye sorulmadı.

    Çapraz kontrol tek başına "iki algoritma da AYNI şekilde bozuk"
    ihtimalini dışlayamaz. Tam sayım dışlar: L = 3'te 512, L = 4'te 65536
    durumun hepsi Boltzmann ağırlığıyla toplanır ve enerji BAĞIMSIZ yazılır
    (her bağ tam bir kez), yani `lattice.total_energy`in çift-sayım
    düzeltmesi de sınanır.

    L = 3 AYRICA BİR KAPSAM İDDİASIDIR: `metropolis.py` tek L'yi REDDEDER
    (checkerboard periyodik kare kafesin iki-parçalı olmasını ister). Wolff'un
    böyle bir şartı yoktur ve bu, Metropolis'in HİÇ KOŞAMADIĞI bir kafeste
    ölçülerek gösterilmiştir — yani bu çıpa çapraz kontrolle KURULAMAZDI.

    ÖLÇÜLDÜ (K = 16 zincir, tohum 20260823 + 1000·k, n_burn = 300,
    n_sweep = 800, soğuk başlangıç):

        L   T           tam           Wolff         SE          sapma
        4   2.000000    0.918943267   0.921494141   2.429e-03   1.05 × SE
        4   2.269185    0.843860445   0.842158203   2.598e-03   0.66 × SE
        3   2.000000    0.925984778   0.925850694   1.747e-03   0.08 × SE
        3   2.269185    0.871071935   0.869722222   1.723e-03   0.78 × SE

    TOLERANS ELLE SEÇİLMEDİ: 4 × SE, ve SE testin kendi zincirlerinden çıkar
    (`test_metropolis.py`nin tam sayım çıpasıyla aynı kural).
    """
    beklenen = _tam_sayim(L, temperature, 1.0)
    assert beklenen == pytest.approx(tam_deger, abs=1e-9)

    v = np.array(
        [
            run_wolff(
                SpinState.uniform(L=L, n_component=1),
                temperature=temperature,
                coupling=1.0,
                field=0.0,
                rng=np.random.default_rng(t),
                n_sweep=800,
                n_burn=300,
            ).magnetisation_series.mean()
            for t in ZINCIR16
        ]
    )
    ort, se = float(v.mean()), _se(v)

    assert abs(ort - beklenen) < 4.0 * se, (
        f"L={L} T={temperature:.6f}: Wolff {ort:.9f}, tam {beklenen:.9f}, "
        f"sapma {abs(ort - beklenen) / se:.2f} × SE"
    )
    assert ort == pytest.approx(olculen, abs=1e-6)
    assert se == pytest.approx(se_beklenen, rel=1e-3)
    assert abs(ort - beklenen) / se == pytest.approx(sapma_se, abs=5e-3)


def test_METROPOLIS_TEK_Lyi_REDDEDERKEN_WOLFF_KOSUYOR():
    """İki çekirdeğin kapsamı AYNI DEĞİL ve bu ölçülerek beyan ediliyor.

    Metropolis checkerboard'dur ve tek L'de periyodik kare kafes iki-parçalı
    olmadığı için reddeder. Wolff'un küme kurulumu iki-parçalılığı hiç
    sormaz. Bu test iki iddiayı tek yerde tutuyor ki, ileride biri
    değişirse diğerinin sessizce yanlış varsayılmasın.
    """
    with pytest.raises(InvalidInputError, match="CIFT"):
        run_metropolis(
            SpinState.uniform(L=3, n_component=1),
            IsingModel(),
            temperature=2.0,
            coupling=1.0,
            field=0.0,
            rng=np.random.default_rng(TOHUM),
            n_sweep=1,
            n_burn=0,
        )

    sonuc = run_wolff(
        SpinState.uniform(L=3, n_component=1),
        temperature=2.0,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(TOHUM),
        n_sweep=5,
        n_burn=1,
    )
    assert sonuc.magnetisation_series.shape == (5,)
    assert sonuc.final_state.spins.shape == (3, 3, 1)


# --------------------------------------------------------------------------
# 3. YAPISAL KILITLER -- KUME GERCEKTEN KUME MU
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tohum", [TOHUM, TOHUM + 1, TOHUM + 2, TOHUM + 3, TOHUM + 17])
def test_KUME_TOHUMUN_BAGLI_AYNI_ISARETLI_BOLGESI(tohum):
    """Küme zıt işaretli bir bölgenin ÜSTÜNDEN ATLAMAZ — deterministik ölçüm.

    Kafes: L = 6, satırları dönüşümlü ±1. Her satır yatayda bir halkadır ve
    dikey komşuları ZIT işaretlidir, yani üç ayrık `+1` bileşen ve üç ayrık
    `−1` bileşen vardır. T = 0.05'te `p = 1 − exp(−40) ≈ 1`, dolayısıyla
    tohumun satırı tamamen kümeye girer ve BAŞKA hiçbir satır giremez.

    Sonuç TOHUMDAN BAĞIMSIZ ve KAPALI FORM: bir satır çevrilince toplam spin
    0'dan ∓12'ye gider, yani `|m| = 12/36 = 1/3`. ÖLÇÜLDÜ — beş ayrı tohumun
    beşinde de tam 0.333333333.

    MUTANT DAVRANIŞI ÖLÇÜLDÜ (bu testin neden var olduğu):
    - işaret kontrolü kalkarsa küme bütün kafese yayılır ve `+1` satırlarının
      HEPSİ birden çevrilir → `|m| = 1.0`.
    - yalnız tohum site çevrilirse → `|m| = 2/36 = 0.0556`.
    Yani üç davranış üç FARKLI sayı verir; test hangisinin gerçekleştiğini
    ayırt edebiliyor.
    """
    durum = _seritli_kafes(6)
    sonuc = run_wolff(
        durum,
        temperature=0.05,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(tohum),
        n_sweep=1,
        n_burn=0,
    )
    assert sonuc.magnetisation_series[0] == pytest.approx(1.0 / 3.0, abs=1e-12)

    # Maskenin kendisi de kilitleniyor: TAM OLARAK bir satir.
    spins = durum.spins[..., 0]
    rng = np.random.default_rng(tohum)
    i0, j0 = (int(v) for v in rng.integers(0, 6, 2))
    maske = _kume_buyut(spins, i0, j0, 1.0 - np.exp(-2.0 / 0.05), rng)
    beklenen = np.zeros((6, 6), dtype=bool)
    beklenen[i0, :] = True
    assert np.array_equal(maske, beklenen)


def test_COK_DUSUK_SICAKLIKTA_KUME_TUM_KAFESI_YUTUYOR():
    """T → 0: `p → 1`, tekbiçim kafeste küme her adımda TÜM kafestir.

    ÖLÇÜLDÜ (L = 8, T = 0.05, tohum 20260823, 20 adım): `|m|` serisinin hem
    en küçüğü hem en büyüğü tam 1.000000000 — yani her adımda kafesin
    tamamı çevriliyor ve mıknatıslanmanın BÜYÜKLÜĞÜ korunuyor. Kümenin bir
    kısmı çevrilseydi `|m|` 1'in altına inerdi.
    """
    durum = SpinState.uniform(L=8, n_component=1)
    sonuc = run_wolff(
        durum,
        temperature=0.05,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(TOHUM),
        n_sweep=20,
        n_burn=0,
    )
    assert np.allclose(sonuc.magnetisation_series, 1.0, atol=1e-12)

    rng = np.random.default_rng(TOHUM)
    i0, j0 = (int(v) for v in rng.integers(0, 8, 2))
    maske = _kume_buyut(
        durum.spins[..., 0], i0, j0, 1.0 - np.exp(-2.0 / 0.05), rng
    )
    assert int(maske.sum()) == 64


def test_COK_YUKSEK_SICAKLIKTA_KUME_TEK_SITE():
    """T → ∞: `p → 0`, küme tohum sitenin kendisidir ve o site çevrilir.

    ÖLÇÜLDÜ (L = 8, T = 1e8, tohum 20260823, tek adım): `|m| = 0.968750000`,
    yani tam olarak (64 − 2)/64. Bu bir KAPALI FORMDUR: tek site çevrilince
    toplam 64'ten 62'ye iner. İki site çevrilseydi 0.9375 olurdu.

    Bu, Metropolis'in `test_COK_YUKSEK_SICAKLIKTA_HER_ONERI_KABUL_EDILIYOR`
    testinin KARŞITIDIR ve karşıtlık bilinçlidir: T → ∞'da Metropolis bir
    sweep'te L² spinin HEPSİNİ çevirir, Wolff BİR tanesini. İki modülün
    "adım" kelimesi aynı şeyi saymaz.
    """
    durum = SpinState.uniform(L=8, n_component=1)
    sonuc = run_wolff(
        durum,
        temperature=1e8,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(TOHUM),
        n_sweep=1,
        n_burn=0,
    )
    assert sonuc.magnetisation_series[0] == pytest.approx(62.0 / 64.0, abs=1e-12)

    rng = np.random.default_rng(TOHUM)
    i0, j0 = (int(v) for v in rng.integers(0, 8, 2))
    maske = _kume_buyut(
        durum.spins[..., 0], i0, j0, 1.0 - np.exp(-2.0 / 1e8), rng
    )
    assert int(maske.sum()) == 1


def test_KUME_BOYUT_KESRI_SICAKLIKLA_DUSUYOR():
    """Küme boyutu `p`nin TEKDÜZE azalan bir fonksiyonu — ve geçiş keskin.

    ÖLÇÜLDÜ (L = 16, tohum 20260823, 200 burn + 200 ölçüm, tekbiçim
    başlangıç):

        T           p           ⟨|C|⟩/L²
        0.500000    0.981684    1.000000
        1.800000    0.670807    0.915605
        2.269185    0.585786    0.481875   ← T_c
        2.800000    0.510458    0.064551
        10.000000   0.181269    0.006211

    İKİ HÜKÜM: (1) küme kesri tekdüze düşüyor; (2) DÜŞÜŞ T_c CİVARINDA
    KESKİN — T_c'nin %23 üstünde (2.8) küme kesri 7.46 kat iniyor, oysa `p`
    yalnız %12.9 düşüyor. Bu, Wolff'un neden T_c'de Metropolis'ten hızlı
    olduğunun doğrudan ölçüsüdür: kritik bölgede tek adım kafesin yaklaşık
    yarısını çeviriyor.

    Bu test FAZ GEÇİŞİ İDDİA ETMEZ — o Task 5'indir. Burada ölçülen şey
    algoritmanın kendi davranışıdır.
    """
    beklenen = {
        0.5: 1.000000,
        1.8: 0.915605,
        TC: 0.481875,
        2.8: 0.064551,
        10.0: 0.006211,
    }
    olculen = []
    for T, hedef in beklenen.items():
        L = 16
        rng = np.random.default_rng(TOHUM)
        spins = np.ones((L, L))
        p = 1.0 - np.exp(-2.0 / T)
        boyutlar = []
        for adim in range(400):
            i0, j0 = (int(v) for v in rng.integers(0, L, 2))
            isaret = spins[i0, j0]
            maske = _kume_buyut(spins, i0, j0, p, rng)
            spins[maske] = -isaret
            if adim >= 200:
                boyutlar.append(maske.sum() / (L * L))
        kesir = float(np.mean(boyutlar))
        assert kesir == pytest.approx(hedef, abs=1e-6)
        olculen.append(kesir)

    assert olculen == sorted(olculen, reverse=True), "kesir tekduze dusmedi"
    # T_c ile 2.8 arasindaki KESKIN dusus: kesir 7.46 kat, `p` yalniz %12.9
    assert olculen[2] / olculen[3] == pytest.approx(7.465053, abs=1e-5)


# --------------------------------------------------------------------------
# 4. UC KAPSAM SINIRI -- UCU DE SESSIZ OLSAYDI TEHLIKELIYDI
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alan", [0.3, -0.3, 1e-12])
def test_WOLFF_ALAN_VARSA_HATA_firlatiyor(alan):
    """Küme çevirme dış alanı ihlal eder; sessizce YANLIŞ dağılım örneklerdi.

    BU SINIR SESSİZ OLSAYDI en tehlikeli türden bir hata olurdu: sonuç makul
    görünür, yakınsar, ve YANLIŞ olur. Kümenin alan enerjisi çevrildiğinde
    değişir ama Wolff'un kabul olasılığı 1 kalır; yani `h ≠ 0`'da ayrıntılı
    denge kurulmaz.

    `1e-12` de reddediliyor: "çok küçük alan" diye bir muafiyet YOKTUR,
    çünkü muafiyetin sınırı fiziksel değil keyfi olurdu.
    """
    with pytest.raises(UnsupportedSystemError, match="alan"):
        run_wolff(
            SpinState.uniform(L=4, n_component=1),
            temperature=2.0,
            coupling=1.0,
            field=alan,
            rng=np.random.default_rng(TOHUM),
            n_sweep=1,
            n_burn=0,
        )


@pytest.mark.parametrize("n_component", [2, 3])
def test_WOLFF_ISING_DISI_SPIN_UZAYINI_reddediyor(n_component):
    """Küme kurulumu "aynı işaret" kavramına dayanır; sürekli spinde yok.

    Sürekli spin uzayında Wolff'un karşılığı bir YANSITMA DÜZLEMİ seçip
    izdüşümlerin işaretine bakmaktır ve bu depoda YAZILMADI.

    SINIR OLMASAYDI İSTİSNA DA ALMAZDIK — ÖLÇÜLDÜ. Gövde `state.spins[..., 0]`
    alıyor, yani İLK bileşeni Ising sanıp kalanları ATIYOR ve `(L, L, 1)` bir
    durum döndürüyor: n_component 3 iken 1 oluyor. Üstelik
    `SpinState.uniform(L, n_component=3)`ün ilk bileşeni tümüyle 0.0'dır
    (`uniform` SON bileşeni 1 yapar), dolayısıyla küme kurulumu "hepsi aynı
    işaret" görür, tüm kafesi yutar ve `-0.0` ile doldurur — mıknatıslanma
    serisi baştan sona 0.0 çıkar. Yani hata ham bir numpy istisnası bile
    olmazdı; sessiz ve makul görünen bir sıfır olurdu.
    """
    with pytest.raises(UnsupportedSystemError, match="n_component"):
        run_wolff(
            SpinState.uniform(L=4, n_component=n_component),
            temperature=2.0,
            coupling=1.0,
            field=0.0,
            rng=np.random.default_rng(TOHUM),
            n_sweep=1,
            n_burn=0,
        )

    # Ayni kafesi Metropolis KOSTURUR (uygun modelle) -- sinir Wolff'a ozel.
    if n_component == 3:
        sonuc = run_metropolis(
            SpinState.uniform(L=4, n_component=3),
            HeisenbergModel(),
            temperature=2.0,
            coupling=1.0,
            field=0.0,
            rng=np.random.default_rng(TOHUM),
            n_sweep=2,
            n_burn=0,
        )
        assert sonuc.magnetisation_series.shape == (2,)


@pytest.mark.parametrize("eslesme", [-1.0, 0.0, -1e-9])
def test_ANTIFERROMANYETIK_ESLESME_REDDEDILIYOR(eslesme):
    """PLANDA OLMAYAN SINIR — ve ölçülen sonucu 20 KAT hata.

    `p = 1 − exp(−2J/T)` J ≤ 0'da pozitif değildir (J = −1, T = 1.5 için
    −2.793668). `rng.random() < p` hiçbir zaman doğru olmaz, hiçbir bağ
    kurulmaz, her adımda rastgele TEK bir site KOŞULSUZ çevrilir — yani
    algoritma sessizce SONSUZ SICAKLIĞI örnekler.

    ÖLÇÜLDÜ — KORUMA GERÇEKTEN KALDIRILARAK (L = 4, 16 zincir, n_burn = 300,
    n_sweep = 800): korumasız Wolff T = 1.5'te `⟨|m|⟩ = 0.201171875`,
    T = 2.0'da `0.201171875`. AYNI SAYI: sıcaklık sonuca HİÇ girmiyor, çünkü
    T yalnız `p`ye giriyor ve `p` her negatif değerinde aynı davranıyor. Tam
    sayımla karşılaştırma: T = ∞'un TAM değeri kapalı formdan
    6435/32768 = 0.196380615234375 — yani örneklenen şey gerçekten sonsuz
    sıcaklık. (Tam sayım büyük ama SONLU T'de bu değeri vermez: T = 1e8'de
    0.196380619 çıkar ve sapma 1/T ile ölçeklenir — 1e6/1e7/1e8/1e9 için
    +4.189e-07/-08/-09/-10. "Tam değer" ile "büyük T'de sayılan değer"
    aynı şey değildir.) Oysa J = −1, T = 1.5'in gerçek değeri
    0.009542350. Hata 20 KAT ve hiçbir istisna atılmıyor.

    `J = 0` da reddediliyor: orada `p = 0` ve tek-site örnekleyici tesadüfen
    DOĞRU olur (bağsız sistemde her konfigürasyon eşit olasılıklı), ama
    "Wolff küme algoritması" adının arkasında hiçbir küme kalmaz. Sınırı
    J > 0'da tutmak, doğru-ama-yanıltıcı olanı da dışarıda bırakır.
    """
    with pytest.raises(InvalidInputError, match="FERROMANYETIK"):
        run_wolff(
            SpinState.uniform(L=4, n_component=1),
            temperature=1.5,
            coupling=eslesme,
            field=0.0,
            rng=np.random.default_rng(TOHUM),
            n_sweep=1,
            n_burn=0,
        )


@pytest.mark.parametrize("gecersiz", [0.0, -1.0, float("nan"), float("inf")])
def test_GECERSIZ_SICAKLIK_reddediliyor(gecersiz):
    """T ≤ 0 Boltzmann çarpanını anlamsız kılar; T = 0'da sıfıra bölünür."""
    with pytest.raises(InvalidInputError, match="sicaklik"):
        run_wolff(
            SpinState.uniform(L=4, n_component=1),
            temperature=gecersiz,
            coupling=1.0,
            field=0.0,
            rng=np.random.default_rng(TOHUM),
            n_sweep=1,
            n_burn=0,
        )


@pytest.mark.parametrize("n_sweep, n_burn", [(0, 10), (-1, 0), (10, -1)])
def test_GECERSIZ_KOSUM_UZUNLUGU_reddediliyor(n_sweep, n_burn):
    """`n_sweep = 0` boş bir seri döndürürdü ve `mean()` NaN verirdi."""
    with pytest.raises(InvalidInputError):
        run_wolff(
            SpinState.uniform(L=4, n_component=1),
            temperature=2.0,
            coupling=1.0,
            field=0.0,
            rng=np.random.default_rng(TOHUM),
            n_sweep=n_sweep,
            n_burn=n_burn,
        )


# --------------------------------------------------------------------------
# 5. SOZLESME KILITLERI
# --------------------------------------------------------------------------


def test_AYNI_TOHUM_AYNI_SONUC_ve_FARKLI_TOHUM_FARKLI_SONUC():
    """Tekrarlanabilirlik pazarlık konusu değildir (Global Constraints)."""
    ortak = dict(
        temperature=2.4, coupling=1.0, field=0.0, n_sweep=25, n_burn=5
    )
    durum = SpinState.uniform(L=8, n_component=1)
    a = run_wolff(durum, rng=np.random.default_rng(TOHUM), **ortak)
    b = run_wolff(durum, rng=np.random.default_rng(TOHUM), **ortak)
    c = run_wolff(durum, rng=np.random.default_rng(TOHUM + 1), **ortak)
    assert np.array_equal(a.magnetisation_series, b.magnetisation_series)
    assert np.array_equal(a.final_state.spins, b.final_state.spins)
    assert not np.array_equal(a.magnetisation_series, c.magnetisation_series)


def test_WOLFF_GIRDIYI_DEGISTIRMIYOR():
    """`state.spins` kopyalanır; aynı başlangıçtan iki koşum bağımsızdır."""
    durum = SpinState.uniform(L=6, n_component=1)
    onceki = durum.spins.copy()
    sonuc = run_wolff(
        durum,
        temperature=2.0,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(TOHUM),
        n_sweep=10,
        n_burn=2,
    )
    assert np.array_equal(durum.spins, onceki)
    assert sonuc.final_state is not durum
    assert not np.array_equal(sonuc.final_state.spins, onceki)


def test_WOLFF_SONUCU_OLCULMEYEN_ALANLARI_None_birakiyor():
    """`None` = "ÖLÇÜLMEDİ", SIFIR DEĞİL.

    Wolff kümeyi çevirdiği için enerjiyi adım başına ucuza güncelleyemez ve
    kabul oranı anlamsızdır (küme HER ZAMAN çevrilir). Task 2'nin
    `test_SONUC_HASHLENEBILIR_ve_OLCULMEYEN_ALANLAR_None` testi bu satırın
    geleceğini yazmıştı; burada kapanıyor. Tüketici taraf `None`u sıfır
    SANMAMALIDIR — Modül 1B'nin kararlılık alanı dersinin aynısı.
    """
    sonuc = run_wolff(
        SpinState.uniform(L=6, n_component=1),
        temperature=2.0,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(TOHUM),
        n_sweep=7,
        n_burn=3,
    )
    assert sonuc.algorithm == "wolff"
    assert sonuc.energy_series is None
    assert sonuc.acceptance_rate is None
    assert sonuc.n_sweep == 7
    assert sonuc.magnetisation_series.shape == (7,)
    assert sonuc.final_state.spins.shape == (6, 6, 1)
    assert isinstance(hash(sonuc), int)


def test_BURN_IN_GERCEKTEN_ATILIYOR():
    """`n_burn` adım GERÇEKTEN atılıyor ve seri burn-in SONRASINI taşıyor.

    Aynı tohumla (n_burn = 0, n_sweep = 12) koşumunun SON 7 ölçümü,
    (n_burn = 5, n_sweep = 7) koşumunun serisiyle BİRE BİR aynı olmalıdır.
    Bu tek test şunları birden kilitler: burn-in gerçekten atılıyor, seri
    uzunluğu `n_sweep`, ölçüm çevirmeden SONRA alınıyor, ve rng akışı
    kaydırılmıyor.
    """
    ortak = dict(temperature=2.4, coupling=1.0, field=0.0)
    durum = SpinState.uniform(L=8, n_component=1)
    tam = run_wolff(
        durum, rng=np.random.default_rng(TOHUM), n_sweep=12, n_burn=0, **ortak
    )
    kirpik = run_wolff(
        durum, rng=np.random.default_rng(TOHUM), n_sweep=7, n_burn=5, **ortak
    )
    assert np.array_equal(tam.magnetisation_series[5:], kirpik.magnetisation_series)
    assert np.array_equal(tam.final_state.spins, kirpik.final_state.spins)


# --------------------------------------------------------------------------
# MUTASYON SINAVI -- 23.08.2026, 8 mutant, HAYATTA KALAN YOK
# --------------------------------------------------------------------------
#
# Her kosum `PYTHONDONTWRITEBYTECODE=1` ile ve ilgili `__pycache__` dizinleri
# SILINEREK yapildi (bayat .pyc tuzagi; bu dosyanin testleri hizli oldugu icin
# ozellikle riskli). Orijinal `atexit` + SIGTERM/SIGINT ile geri yukleniyor.
#
#   mutant                                    yakalayan test(ler)
#   M1 p = 1 - exp(-J/T)  (2 duser)           capraz kontrol (110.73 sigma),
#                                             tam sayim
#   M2 kume ayni-isaret sartini kaybeder      capraz kontrol (26.45 sigma),
#                                             kume bagli bolge, kume boyutu,
#                                             tam sayim
#   M3 h != 0 kontrolu kalkar                 alan siniri (3 parametre)
#   M4 yalniz tohum site cevrilir             capraz kontrol (109.06 sigma),
#                                             T->0 tum kafes, kume bagli
#                                             bolge, tam sayim
#   M5 J <= 0 korumasi kalkar                 antiferromanyetik sinir
#   M6 n_component != 1 korumasi kalkar       Ising disi spin uzayi
#   M7 burn-in atilmiyor                      burn-in testi, capraz kontrol,
#                                             tam sayim, sozlesme kilitleri
#   M8 p = 1 - exp(+2J/T) (ustelin isareti)   capraz kontrol, T->0 tum kafes,
#                                             kume bagli bolge, tam sayim
#
# UC MUTANTTA (M1, M2, M4) DUSEN IDDIA OLCULDU: hata mesaji `fark < sinir`
# satirindan geliyor, yani KILITLENMIS SAYIDAN degil FIZIK IDDIASINDAN.
# Bu, assert sirasinin bilincli sonucudur -- 3-sigma once, transkripsiyon
# kilidi sonra. Ters sirada mutantlar yine yakalanirdi ama "capraz kontrol
# yakaladi" demek olcume dayanmazdi.


# --------------------------------------------------------------------------
# `slow` ISARETININ GEREKCESI -- OLCUM, DEVRALINAN ESIK DEGIL
# --------------------------------------------------------------------------
#
# Depoda devralinabilir bir esik YOKTUR; `test_metropolis.py` esigi 22.08'de
# yeniden olcup "kendi call suresi 2.0 s ve ustunde olan durum `slow`" diye
# sabitledi ve o kural burada da uygulandi. Bu dosyanin OLCULEN call sureleri
# (`--durations`, 23.08.2026, bu makinede):
#
#     12.08 s  capraz kontrol  L=16 T_c            <- slow
#      4.46 s  capraz kontrol  L=16 T=2.8 SICAK    <- slow
#      4.30 s  capraz kontrol  L=8  T=1.8          <- slow
#      3.95 s  capraz kontrol  L=8  T_c   (uzun)   <- slow
#      3.30 s  capraz kontrol  L=8  T=2.8          <- slow
#      1.72 s  capraz kontrol  L=8  T_c   (kisa)   <- ISARETSIZ
#      0.33 s  tam sayim L=4 T=2.0
#      0.32 s  tam sayim L=4 T_c
#      0.23 s  kume boyut kesri
#      0.22 s  tam sayim L=3 T=2.0
#      0.22 s  tam sayim L=3 T_c
#      < 0.005 s  kalan 27 durum
#
# Esigi gecen HER durum isaretlendi, gecmeyen HICBIRI isaretlenmedi.
#
# TEK KARAR NOKTASI capraz kontrolun hangi satirinin varsayilan seritte
# kalacagiydi. Uzun zincirli bes satirin hepsi 3.3-12.1 s surer; hepsi
# isaretlenseydi TASIYICI TEST varsayilan seritten TAMAMEN duserdi -- bu,
# `test_mermin_wagner.py`de bir kez odenen bedelin aynisi olurdu. Bu yuzden
# AYNI IDDIA kisa zincirlerle bir kez daha kuruldu (L = 8, T_c, K = 24,
# metro (200, 500), wolff (100, 400)) ve YALNIZ o satir isaretsiz birakildi.
# Bedeli olculdu: 1.72 s, esigin altinda. Kaybettigi de olculdu: 3-sigma
# siniri uzun zincirlerde 1.046e-02, kisa zincirlerde 1.538e-02 -- yani %47
# daha genis. Bu bir GEVSETME DEGIL: sinir yine hata cubugundan gelir ve
# cubuk kisa zincirde GERCEKTEN daha genistir. Ayni iddia iki farkli
# zincir uzunlugunda iki kez kuruluyor ve ikisi de geciyor.
#
# tests/spin/ serit dagilimi bu dosyayla birlikte OLCULDU:
#   -m "not slow"  78 test  11.53 s   (Task 3 oncesi 45 test 9.18 s)
#   -m slow         7 test  32.84 s   (Task 3 oncesi  2 test 4.82 s)
