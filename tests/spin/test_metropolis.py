"""Metropolis çekirdeği: MODELİ BİLMEZ, ve ÖRNEKLEDİĞİ DAĞILIM DOĞRUDUR.

Bu dosya iki ayrı iş yapar ve ikisi de gereklidir:

1. YAPISAL KİLİTLER (sınıf: transkripsiyon denetimi) — çekirdeğin modele
   yalnız `n_component` + `propose` protokolünden baktığını, checkerboard'un
   gerçekten checkerboard olduğunu, `run_metropolis`in elle koşulan
   sweep'lerle bire bir aynı olduğunu kilitler.

2. DAĞILIM ÇIPALARI (sınıf: DAİRESEL DEĞİL) — üç bağımsız tam çözümle
   karşılaştırma. Hiçbiri simülasyona girdi değildir:

   a. TAM SAYIM: L = 4 Ising'in 2^16 = 65536 durumunun HEPSİ sayılır ve
      `⟨|m|⟩` ile `⟨E⟩` Boltzmann ağırlığıyla tam olarak hesaplanır. Enerji
      bu sayımda BAĞIMSIZ bir formülle (her bağ tam bir kez, sağ ve aşağı
      komşu) yazılır — `lattice.total_energy`in çift-sayım düzeltmesini
      KULLANMAZ, dolayısıyla onu de sınar.
   b. tanh ve LANGEVIN: J = 0'da siteler bağımsızlaşır ve tek-spin
      çözümü kapalı formdur — Ising'de `tanh(h/T)`, klasik Heisenberg'de
      Langevin fonksiyonu `coth(h/T) - T/h`. İKİ MODEL de AYNI çekirdekten
      geçer; farklı sayıları vermeleri, çekirdeğin `propose`u gerçekten
      çağırdığının kanıtıdır.
   c. ONSAGER-YANG taban mıknatıslanması `m = [1 - sinh^-4(2J/T)]^(1/8)`
      (Yang 1952) — 2B Ising'in TAM çözümü, T < T_c için.

   (a) ve (b)'nin toleransı ELLE SEÇİLMEZ: bağımsız zincirlerin
   ortalamalarından bir standart hata (SE) hesaplanır ve sınır 4 x SE'dir.

ÖLÇÜLEN BİR TUZAK — SE TAHMİNCİSİNİN KENDİSİ GÜRÜLTÜLÜDÜR. Langevin çıpası
önce 6 zincirle koşuldu ve sapma 5.52 x SE çıktı; "yanlılık var" gibi
göründü. 32 zincirle yeniden ölçüldü: nb=300/ns=1000'de +0.76 x SE,
nb=2000/ns=4000'de -0.86 x SE, sıcak başlangıçta -0.50 x SE, yani YANLILIK
YOK — 6 zincirin (5 serbestlik derecesi) SE tahmini tesadüfen küçük çıkmıştı.
Ders: SE'yi tolerans yapan bir testte zincir sayısı da ölçülmüş olmalıdır;
bu dosya 8 ve 10 zincir kullanır.

GEÇERLİLİK PENCERESİ — BEYAN EDİLİYOR:
- Çekirdek YALNIZ ÇİFT L için doğrudur ve tek L'yi REDDEDER. Sebep sayıldı:
  periyodik kare kafes tek L'de iki-parçalı (bipartite) DEĞİLDİR. Aynı
  alt-örgüdeki komşu bağ sayısı — L=5: 50 bağın 10'u (%20.0), L=9: 162'nin
  18'i (%11.1), L=17: 578'in 34'ü (%5.9); çift L'de 0. O bağların iki ucu
  aynı anda güncellenir ve ayrıntılı denge bozulur.
- Klasik spin, kare kafes, `k_B = 1` birim seçimi (J bir girdidir), sonlu L.
- Bu dosya faz geçişi, kritik üs veya T_c hakkında HİÇBİR ŞEY iddia etmez;
  onlar Task 5'indir.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import numpy as np
import pytest

from qlab.engine.spin.lattice import SpinState, magnetisation
from qlab.engine.spin.metropolis import (
    SamplingResult,
    metropolis_sweep,
    run_metropolis,
)
from qlab.engine.spin.models import HeisenbergModel, IsingModel
from qlab.errors import InvalidInputError

TOHUM = 20260822
ZINCIR8 = (20260822, 7, 424242, 991, 1, 31337, 555555, 2718281)
ZINCIR10 = ZINCIR8 + (11235, 8675309)


# --------------------------------------------------------------------------
# Yardimcilar
# --------------------------------------------------------------------------


def _tam_sayim(L: int, temperature: float, coupling: float, field: float):
    """2^(L*L) durumun HEPSİNİ sayar; `⟨|m|⟩` ve `⟨E⟩` döndürür.

    Enerji burada BAĞIMSIZ yazılır: her bağ TAM BİR KEZ (her sitenin sağ ve
    aşağı komşusu). `lattice.neighbour_sum` + 0.5 düzeltmesi KULLANILMAZ, yani
    o düzeltme de bu çıpa tarafından sınanır.
    """
    n = L * L
    idx = np.arange(2**n, dtype=np.int64)
    bitler = (idx[:, None] >> np.arange(n)) & 1
    s = (2 * bitler - 1).astype(float).reshape(-1, L, L)
    bag = np.sum(s * np.roll(s, -1, axis=1), axis=(1, 2)) + np.sum(
        s * np.roll(s, -1, axis=2), axis=(1, 2)
    )
    enerji = -coupling * bag - field * s.sum(axis=(1, 2))
    mutlak_m = np.abs(s.mean(axis=(1, 2)))
    # Emin cikarilmasi TASMA korumasidir; oran degismez.
    agirlik = np.exp(-(enerji - enerji.min()) / temperature)
    z = agirlik.sum()
    return float((agirlik * mutlak_m).sum() / z), float((agirlik * enerji).sum() / z)


def _zincir_ortalamalari(tohumlar, kosucu):
    """Bağımsız zincirlerin ortalamaları + standart hataları."""
    v = np.array([kosucu(t) for t in tohumlar], dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def _paritye(L: int) -> np.ndarray:
    i, j = np.indices((L, L))
    return (i + j) % 2


# --------------------------------------------------------------------------
# 1. DAGILIM CIPALARI
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alan, tam_m, tam_e, olculen_m, olculen_e",
    [
        (0.0, 0.918943, -28.086085, 0.920039, -28.126250),
        (0.3, 0.954355, -34.029807, 0.954115, -33.947083),
    ],
)
def test_TAM_SAYIM_CIPASI_ornekleme_DOGRU_dagilimi_veriyor(
    alan, tam_m, tam_e, olculen_m, olculen_e
):
    """SINIF: DAİRESEL DEĞİL — 65536 durumun tamamı sayıldı, MC'ye sorulmadı.

    L = 4, T = 2.0, J = 1; 8 bağımsız zincir, n_burn = 400, n_sweep = 1200.
    ÖLÇÜLDÜ (22.08.2026):

        h = 0.0: ⟨|m|⟩ tam 0.918943, MC 0.920039, SE 1.588e-03 → 0.69 x SE
                 ⟨E⟩   tam -28.086085, MC -28.126250, SE 5.686e-02 → 0.71 x SE
        h = 0.3: ⟨|m|⟩ tam 0.954355, MC 0.954115, SE 1.563e-03 → 0.15 x SE
                 ⟨E⟩   tam -34.029807, MC -33.947083, SE 1.274e-01 → 0.65 x SE

    TOLERANS ELLE SEÇİLMEDİ: 4 x SE, ve SE testin kendi zincirlerinden çıkar.

    BU ÇIPANIN GÜCÜ ÖLÇÜLDÜ, tahmin edilmedi:
    - `field` terimi düşerse h = 0.3 koşumu 0.918943'ü görür — doğru değerden
      23 x SE uzakta.
    - checkerboard maskesi kalkarsa (tüm siteler aynı anda) ölçülen
      ⟨|m|⟩ = 0.000000 ve ⟨E⟩ = +29.0 olur (h=0'da); yani paralel güncelleme
      bir limit döngüsüne düşüyor ve enerjinin İŞARETİ bile ters çıkıyor.
    """
    beklenen_m, beklenen_e = _tam_sayim(4, 2.0, 1.0, alan)
    assert beklenen_m == pytest.approx(tam_m, abs=1e-6)
    assert beklenen_e == pytest.approx(tam_e, abs=1e-6)

    def _kos(tohum):
        return run_metropolis(
            SpinState.uniform(L=4, n_component=1),
            IsingModel(),
            temperature=2.0,
            coupling=1.0,
            field=alan,
            rng=np.random.default_rng(tohum),
            n_sweep=1200,
            n_burn=400,
        )

    sonuclar = [_kos(t) for t in ZINCIR8]
    m_ort, m_se = _zincir_ortalamalari(
        range(len(sonuclar)),
        lambda k: sonuclar[k].magnetisation_series.mean(),
    )
    e_ort, e_se = _zincir_ortalamalari(
        range(len(sonuclar)),
        lambda k: sonuclar[k].energy_series.mean(),
    )

    assert m_ort == pytest.approx(olculen_m, abs=1e-5)
    assert e_ort == pytest.approx(olculen_e, abs=1e-4)
    assert abs(m_ort - beklenen_m) < 4.0 * m_se, (
        f"⟨|m|⟩ sapmasi {abs(m_ort - beklenen_m) / m_se:.2f} x SE"
    )
    assert abs(e_ort - beklenen_e) < 4.0 * e_se, (
        f"⟨E⟩ sapmasi {abs(e_ort - beklenen_e) / e_se:.2f} x SE"
    )


@pytest.mark.parametrize(
    "model, ad, tam_deger, olculen",
    [
        (IsingModel(), "tanh", 0.761594156, 0.760799479),
        (HeisenbergModel(), "Langevin", 0.313035285, 0.311940436),
    ],
)
def test_ALANSIZ_ESLESME_TEK_SPIN_COZUMUNU_veriyor(model, ad, tam_deger, olculen):
    """SINIF: DAİRESEL DEĞİL — J = 0'da tek-spin çözümü KAPALI FORMDUR.

    J = 0 siteleri bağımsızlaştırır ve `⟨s^z⟩` tam olarak bilinir:

        Ising      (s = ±1)        : tanh(h/T)              = 0.761594156
        Heisenberg (s in S^2)      : coth(h/T) - T/h        = 0.313035285

    İKİ MODEL DE AYNI ÇEKİRDEKTEN GEÇER. Farklı sayı vermeleri, çekirdeğin
    `model.propose`u gerçekten çağırdığının doğrudan kanıtıdır: `propose`
    yerine `-spins` yazan mutant Heisenberg'de de tanh'ı görür ve 0.31 yerine
    0.76'ya gider: 10 zincirde olculen 0.760799, dogru degerden 333 x SE uzakta.

    ÖLÇÜLDÜ (L = 16, h = 1.0, T = 1.0, 10 zincir, n_burn = 300, n_sweep = 1200):
        Ising      MC 0.760799479, SE 3.844e-04 → 2.07 x SE
        Heisenberg MC 0.311940436, SE 1.346e-03 → 0.81 x SE

    NEDEN `⟨s^z⟩`, `⟨|m|⟩` DEĞİL: `|m|` üç bileşenin normudur ve enine
    bileşenlerin dalgalanması onu YUKARI kaydırır. ÖLÇÜLDÜ (L = 16, 8 tohum):
    ⟨|m|⟩ = 0.317270 ama ⟨s^z⟩ = 0.313317, bağıl yanlılık %1.26.
    Kapalı form `⟨s^z⟩` içindir; norm alan bir tahminci bu çıpayı sessizce
    yanlı yapardı. Bu yüzden test `magnetisation(...)[-1]`i kendisi toplar.

    ÖNERİ SİMETRİSİ (dairesellik denetimi): Heisenberg önerisi
    `s + step * N(0, 1)^3` normalize edilerek üretilir; bu önerinin yoğunluğu
    yalnız `s . s'`ye bağlıdır, dolayısıyla SİMETRİKTİR ve Metropolis kabul
    kuralı (Hastings düzeltmesi olmadan) geçerlidir. Bu argüman ölçümle
    desteklendi: 32 zincirle sapma +0.76 / -0.86 / -0.50 x SE arasında,
    yani sistematik yanlılık yok.
    """

    def _kos(tohum):
        durum = SpinState.uniform(L=16, n_component=model.n_component)
        rng = np.random.default_rng(tohum)
        toplam = 0.0
        for adim in range(300 + 1200):
            durum = metropolis_sweep(
                durum,
                model,
                temperature=1.0,
                coupling=0.0,
                field=1.0,
                rng=rng,
            )
            if adim >= 300:
                toplam += float(magnetisation(durum)[-1])
        return toplam / 1200

    ort, se = _zincir_ortalamalari(ZINCIR10, _kos)
    assert ort == pytest.approx(olculen, abs=1e-6)
    assert abs(ort - tam_deger) < 4.0 * se, (
        f"{ad}: sapma {abs(ort - tam_deger) / se:.2f} x SE "
        f"(MC {ort:.9f}, tam {tam_deger:.9f}, SE {se:.3e})"
    )


@pytest.mark.parametrize(
    "temperature, tam_deger, olculen",
    [
        (0.8, 0.999906681, 0.999927083),
        (1.0, 0.999275752, 0.999250000),
        (1.4, 0.991386865, 0.991493490),
    ],
)
def test_ISING_TABAN_MIKNATISLANMASI_ONSAGER_YANG_ile_ucusuyor(
    temperature, tam_deger, olculen
):
    """SINIF: DAİRESEL DEĞİL — Yang'ın (1952) tam çözümü simülasyona GİRMEZ.

        m(T) = [1 - sinh^-4(2J/T)]^(1/8),   T < T_c

    ÖLÇÜLDÜ (L = 32, tohum 20260822, n_burn = 500, n_sweep = 1500):

        T      tam           MC            fark
        0.8    0.999906681   0.999927083   +2.040e-05
        1.0    0.999275752   0.999250000   -2.575e-05
        1.4    0.991386865   0.991493490   +1.066e-04

    TOLERANS 1e-3'ÜN GEREKÇESİ ÖLÇÜM: MC değerinin 8 tohumdaki std'si
    8.75e-06 (T=0.8), 2.26e-05 (T=1.0), 1.79e-04 (T=1.4). 1e-3 bunların en
    büyüğünün 5.6 katıdır. L = 32'de sonlu-boyut düzeltmesi tohum
    gürültüsünün ALTINDA kalıyor (8 tohumun ortalaması tam değerden
    +2.8e-06 / +2.1e-06 / -8.3e-06 sapıyor).

    BU TEST `np.clip` SINIRININ BEKÇİSİDİR — planın "ölç" dediği mutant
    burada ölçüldü ve cevap NET: üstel sınır ±50'den ±5'e indirilirse
    ölçülen `⟨|m|⟩` 0.985978 / 0.985251 / 0.982147 olur, yani tam değerden
    -1.39e-02 / -1.40e-02 / -9.24e-03 sapar — toleransın 9 ilâ 14 katı.
    Sınır ±1'e inerse mıknatıslanma tamamen çöker (0.040389).
    Yani ±5 FİZİĞİ BOZAR ve bu test onu görür; Mermin-Wagner testi ise
    GÖRMEZ (aynı mutantta Heisenberg eğimi -1.0062'den -0.9864'e gidiyor,
    hâlâ sınırın içinde). İki testin biri olmadan mutant kaçardı.
    """
    beklenen = float((1.0 - np.sinh(2.0 / temperature) ** -4) ** 0.125)
    assert beklenen == pytest.approx(tam_deger, abs=1e-9)

    sonuc = run_metropolis(
        SpinState.uniform(L=32, n_component=1),
        IsingModel(),
        temperature=temperature,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(TOHUM),
        n_sweep=1500,
        n_burn=500,
    )
    olculen_mc = float(sonuc.magnetisation_series.mean())
    assert olculen_mc == pytest.approx(olculen, abs=1e-6)
    assert olculen_mc == pytest.approx(beklenen, abs=1e-3)


def test_KABUL_ORANI_ADIMLA_DUSUYOR_ve_BIR_TABANA_DOYUYOR():
    """TASK 1'DEN DEVROLAN BORCU KAPATIR — ve hükmü NİTELİKLİ çıkıyor.

    Task 1'in `HeisenbergModel` dokstringi "varsayılan 0.6 ölçüldü: T ~ 1'de
    kabul oranı makul" diyordu ve bu ORADA ÖLÇÜLEMEZDİ (kabul/red kararı
    yoktu); `OLCULMEDI -- BEYAN` diye bırakılmıştı. Burada ölçüldü.

    ÖLÇÜLDÜ (L = 16, T = 1.0, J = 1, h = 0, tohum 20260822, 500/1500):

        step     kabul oranı
        0.05     0.965898
        0.60     0.676958   (varsayılan)
        2.00     0.499771
        100.0    0.417086   ← DOYUM

    İKİ HÜKÜM, VE İKİNCİSİ BEKLENMİYORDU:

    (1) Kabul oranı `step` ile TEKDÜZE düşüyor — dokstringin "step kabul
        oranını ayarlar" cümlesi doğru.

    (2) AMA BİR TABANA DOYUYOR. `step` büyüdükçe öneri, kürede DÜZGÜN
        DAĞILMIŞ rastgele bir yöne yakınsar ve kabul oranı 0.42'nin altına
        İNEMEZ. Yani bu öneri ailesiyle T = 1.0'da erişilebilir aralık
        [0.42, 1.00)'dir ve Metropolis folklorundaki "%30-50 hedefle"
        kuralı burada ancak ucundan tutturulabilir (step ≳ 2, yani "küçük
        dönme" olmaktan çıktığı yerde).

    Dolayısıyla dürüst hüküm şudur: 0.6 için ölçülen 0.677, klasik %30-50
    bandının ÜSTÜNDEDİR; "makul" ancak "ne ~0 ne ~1, erişilebilir aralığın
    ortasına yakın" anlamında doğrudur. `models.py` dokstringi bu ölçülmüş
    sayıyla düzeltildi.

    Sıcaklığa bağlılığı da ölçüldü (step = 0.6, 4 tohum): T = 0.8 → 0.6008,
    T = 1.0 → 0.6770, T = 1.4 → 0.7700. Ising'in kabul oranı `step`
    TAŞIMAZ ve çok küçüktür (T = 1.0 → 0.000802, T = T_c → 0.196302);
    iki modeli aynı bantta beklemek yanlış olurdu.
    """

    def _kabul(step):
        sonuc = run_metropolis(
            SpinState.uniform(L=16, n_component=3),
            HeisenbergModel(step=step),
            temperature=1.0,
            coupling=1.0,
            field=0.0,
            rng=np.random.default_rng(TOHUM),
            n_sweep=1500,
            n_burn=500,
        )
        return sonuc.acceptance_rate

    kucuk = _kabul(0.05)
    varsayilan = _kabul(0.6)
    buyuk = _kabul(2.0)
    devasa = _kabul(100.0)

    assert kucuk == pytest.approx(0.965898, abs=1e-5)
    assert varsayilan == pytest.approx(0.676958, abs=1e-5)
    assert buyuk == pytest.approx(0.499771, abs=1e-5)
    assert devasa == pytest.approx(0.417086, abs=1e-5)

    # (1) TEKDUZE dusuyor
    assert kucuk > varsayilan > buyuk > devasa

    # (2) TABANA doyuyor: step 50 kat buyurken kabul orani %17 duser
    assert devasa > 0.4
    assert (buyuk - devasa) < 0.5 * (varsayilan - buyuk)

    # Ising ayni cekirdekten gecer ama `step` TASIMAZ ve bambaska bir bantta
    ising = run_metropolis(
        SpinState.uniform(L=16, n_component=1),
        IsingModel(),
        temperature=1.0,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(TOHUM),
        n_sweep=1500,
        n_burn=500,
    )
    assert ising.acceptance_rate == pytest.approx(0.000802, abs=1e-6)


# --------------------------------------------------------------------------
# 2. YAPISAL KILITLER
# --------------------------------------------------------------------------


@dataclass
class _KaydeciModel:
    """`propose` çağrılarını kaydeden casus — çekirdeğe HİÇBİR ek yol açmaz."""

    ic: object
    cagrilar: list = dataclass_field(default_factory=list)

    @property
    def n_component(self) -> int:
        return int(self.ic.n_component)

    def propose(self, spins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        aday = self.ic.propose(spins, rng)
        self.cagrilar.append((spins.copy(), aday.copy()))
        return aday


@dataclass(frozen=True)
class _XYModel:
    """ÜÇÜNCÜ spin uzayı — `n_component = 2`, depoda hiç yok, TESTTE tanımlı."""

    step: float = 0.6
    n_component: int = 2

    def propose(self, spins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        aday = spins + self.step * rng.normal(size=spins.shape)
        return aday / np.linalg.norm(aday, axis=2, keepdims=True)


def test_CEKIRDEK_MODELE_YALNIZ_PROTOKOLDEN_bakiyor_ve_CHECKERBOARD_gercek():
    """Çekirdeğin `propose`u çağırdığını ve maskenin GERÇEK olduğunu kilitler.

    Üç iddia, hepsi tek bir casus koşumundan:

    1. `propose` sweep başına TAM İKİ KEZ çağrılır (iki alt-örgü). Bir kez
       çağrılsaydı sitelerin yarısı hiç güncellenmezdi.
    2. Sonuç dizisi ELEMAN BAZINDA ya eski spin ya öneridir — `np.where`
       üçüncü bir değer üretmez. Heisenberg kullanılır çünkü Ising'de "eski
       ya da -eski" iddiası her işaret dizisi için doğrudur, yani hiçbir şey
       söylemez.
    3. Bir yarı-adımda DEĞİŞEN sitelerin hepsi AYNI paritededir ve bir
       sweep'in iki yarı-adımı ZIT paritelidir. Bu, checkerboard'un adının
       hakkını verdiğinin doğrudan ölçüsüdür; maske kalkarsa iki yarı-adım
       da her iki pariteyi birden değiştirir.
    """
    casus = _KaydeciModel(HeisenbergModel())
    L = 8
    sonuc = run_metropolis(
        SpinState.uniform(L=L, n_component=3),
        casus,
        temperature=1.0,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(TOHUM),
        n_sweep=3,
        n_burn=2,
    )

    assert len(casus.cagrilar) == 2 * (3 + 2)

    parite = _paritye(L)
    sonraki_durumlar = [eski for eski, _ in casus.cagrilar[1:]]
    sonraki_durumlar.append(sonuc.final_state.spins)

    gorulen_pariteler = []
    for (eski, oneri), yeni in zip(casus.cagrilar, sonraki_durumlar):
        eskiydi = np.all(np.isclose(yeni, eski), axis=2)
        onerivdi = np.all(np.isclose(yeni, oneri), axis=2)
        assert np.all(eskiydi | onerivdi), "np.where UCUNCU bir deger uretti"

        degisen = ~eskiydi
        assert degisen.any(), "yari-adimda hic site degismedi (T=1'de beklenmez)"
        pariteler = set(np.unique(parite[degisen]).tolist())
        assert len(pariteler) == 1, (
            f"tek yari-adimda IKI parite birden degisti: {pariteler}"
        )
        gorulen_pariteler.append(pariteler.pop())

    for k in range(0, len(gorulen_pariteler), 2):
        assert gorulen_pariteler[k] != gorulen_pariteler[k + 1], (
            "bir sweep'in iki yari-adimi AYNI alt-orguyu guncelledi"
        )


def test_ORNEKLEYICI_UCUNCU_SPIN_UZAYINI_da_kosturuyor():
    """`n_component = 2` bir model, çekirdekte TEK SATIR değişmeden koşuyor.

    XY modeli depoda yoktur ve burada tanımlanmıştır. Çekirdek Ising'i de
    Heisenberg'i de adıyla bilmediği için bunu kabul eder; `n_component`i
    1 veya 3 varsayan her uygulama burada düşer.

    Bu test FİZİK İDDİA ETMEZ ve etmemesi bilinçlidir: 2B XY modeli T = 1.0
    civarında BKT geçişinin yakınındadır, `⟨|m|⟩` orada yavaş bir kuvvet
    yasasıyla söner ve 8 tohumda ölçülen yayılım büyüktür (L=32'de std
    6.0e-02, m(8)/m(32) = 1.24…1.79). Yani "XY de sönüyor" demek burada
    ölçülebilir bir hüküm değildir; Mermin-Wagner çıpası Heisenberg'le
    kurulur.

    KİLİTLENEN ŞEY: çekirdek son bileşene (`[..., -1]`) alan uygularken
    n_component = 2'de son bileşen İLK bileşen DEĞİLDİR — Task 1'de ölçülen
    "n_component = 1'de son ile ilk aynı şeydir" körlüğünü kapatan üçüncü
    tanık budur.
    """
    durum = SpinState.uniform(L=8, n_component=2)
    sonuc = run_metropolis(
        durum,
        _XYModel(),
        temperature=1.0,
        coupling=1.0,
        field=0.25,
        rng=np.random.default_rng(TOHUM),
        n_sweep=20,
        n_burn=5,
    )
    son = sonuc.final_state.spins
    assert son.shape == (8, 8, 2)
    assert np.allclose(np.linalg.norm(son, axis=2), 1.0, atol=1e-12)
    assert not np.allclose(son, durum.spins), "zincir hic hareket etmedi"
    assert sonuc.magnetisation_series.shape == (20,)


def test_RUN_METROPOLIS_ELLE_KOSULAN_SWEEPLERLE_BIRE_BIR_AYNI():
    """`run_metropolis` bir plumbing katmanıdır ve KENDİ fiziği yoktur.

    Aynı tohumla `metropolis_sweep`i elle `n_burn + n_sweep` kez koşmak,
    `run_metropolis`in döndürdüğü seriyi ve son durumu BİRE BİR vermelidir.
    Bu tek test şunları birden kilitler: burn-in gerçekten ATILIYOR, seri
    uzunluğu `n_sweep`, ölçüm sweep'ten SONRA alınıyor, ve rng akışı
    kaydırılmıyor.
    """
    model = HeisenbergModel()
    baslangic = SpinState.uniform(L=6, n_component=3)
    ortak = dict(temperature=1.1, coupling=1.0, field=0.2)

    sonuc = run_metropolis(
        baslangic, model, rng=np.random.default_rng(TOHUM), n_sweep=7, n_burn=4, **ortak
    )

    rng = np.random.default_rng(TOHUM)
    durum = baslangic
    seri = []
    for adim in range(4 + 7):
        durum = metropolis_sweep(durum, model, rng=rng, **ortak)
        if adim >= 4:
            seri.append(float(np.linalg.norm(magnetisation(durum))))

    assert sonuc.n_sweep == 7
    assert sonuc.algorithm == "metropolis"
    assert sonuc.magnetisation_series.shape == (7,)
    assert np.array_equal(sonuc.magnetisation_series, np.array(seri))
    assert np.array_equal(sonuc.final_state.spins, durum.spins)


def test_AYNI_TOHUM_AYNI_SONUC_ve_FARKLI_TOHUM_FARKLI_SONUC():
    """Tekrarlanabilirlik pazarlık konusu değildir (Global Constraints)."""
    ortak = dict(temperature=1.2, coupling=1.0, field=0.0, n_sweep=15, n_burn=5)
    durum = SpinState.uniform(L=8, n_component=3)
    a = run_metropolis(
        durum, HeisenbergModel(), rng=np.random.default_rng(TOHUM), **ortak
    )
    b = run_metropolis(
        durum, HeisenbergModel(), rng=np.random.default_rng(TOHUM), **ortak
    )
    c = run_metropolis(
        durum, HeisenbergModel(), rng=np.random.default_rng(TOHUM + 1), **ortak
    )
    assert np.array_equal(a.magnetisation_series, b.magnetisation_series)
    assert not np.array_equal(a.magnetisation_series, c.magnetisation_series)


def test_SWEEP_GIRDIYI_DEGISTIRMIYOR():
    """Reddedilen bir öneriyi geri alabilmenin ön koşulu budur."""
    durum = SpinState.uniform(L=6, n_component=3)
    onceki = durum.spins.copy()
    yeni = metropolis_sweep(
        durum,
        HeisenbergModel(),
        temperature=1.0,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(TOHUM),
    )
    assert np.array_equal(durum.spins, onceki)
    assert yeni is not durum
    assert not np.array_equal(yeni.spins, onceki)


def test_COK_YUKSEK_SICAKLIKTA_HER_ONERI_KABUL_EDILIYOR():
    """T → ∞: `exp(-ΔE/T) → 1`, yani TEK sweep'te TÜM Ising spinleri çevrilir.

    ÖLÇÜLDÜ (L = 6, T = 1e8, tek sweep, tohum 20260822): kabul oranı
    1.000000 ve kafesin tamamı -1. Bir alt-örgü atlansaydı yarısı +1 kalırdı,
    yani bu test aynı zamanda "sweep iki yarı-adımdır" iddiasının ikinci
    tanığıdır.
    """
    sonuc = run_metropolis(
        SpinState.uniform(L=6, n_component=1),
        IsingModel(),
        temperature=1e8,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(TOHUM),
        n_sweep=1,
        n_burn=0,
    )
    assert sonuc.acceptance_rate == pytest.approx(1.0)
    assert np.allclose(sonuc.final_state.spins, -1.0)


def test_COK_DUSUK_SICAKLIKTA_ZINCIR_DONUYOR():
    """T → 0: taban durumdan çıkış maliyetli, kabul oranı 0.

    ÖLÇÜLDÜ (L = 6, T = 1e-3, üç sweep): kabul oranı 0.000000 ve kafes
    tekbiçim +1 kalıyor. `delta`nın işareti ters çevrilirse bu testin tersi
    olur (taban durumdan hemen kaçılır), yani işaret burada da kilitli.
    """
    sonuc = run_metropolis(
        SpinState.uniform(L=6, n_component=1),
        IsingModel(),
        temperature=1e-3,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(TOHUM),
        n_sweep=3,
        n_burn=0,
    )
    assert sonuc.acceptance_rate == pytest.approx(0.0)
    assert np.allclose(sonuc.final_state.spins, 1.0)
    assert np.allclose(sonuc.magnetisation_series, 1.0)


def test_TEK_L_REDDEDILIYOR():
    """Tek L'de kare kafes iki-parçalı DEĞİLDİR; checkerboard sessizce bozulur.

    SAYILDI: aynı alt-örgüdeki komşu bağ sayısı L=5'te 50 bağın 10'u (%20.0),
    L=9'da 162'nin 18'i (%11.1), L=17'de 578'in 34'ü (%5.9). Çift L'de 0.
    O bağların İKİ UCU da aynı yarı-adımda güncellenir ve ayrıntılı denge
    bozulur — hem de sessizce, sonuç makul görünerek.
    """
    for L in (5, 9):
        with pytest.raises(InvalidInputError, match="CIFT"):
            metropolis_sweep(
                SpinState.uniform(L=L, n_component=1),
                IsingModel(),
                temperature=1.0,
                coupling=1.0,
                field=0.0,
                rng=np.random.default_rng(TOHUM),
            )


def test_BILESEN_SAYISI_UYUSMAZLIGI_REDDEDILIYOR():
    """Model 3 bileşenli, kafes 1 bileşenliyse sonuç SESSİZCE yanlış olurdu.

    `HeisenbergModel().propose` bir (L, L, 1) diziye uygulanınca ±1 döndürür
    — yani Ising gibi davranır ama adı Heisenberg kalır. Mermin-Wagner
    çıpası tam olarak bu karışıklığa dayanamaz.
    """
    with pytest.raises(InvalidInputError, match="n_component"):
        metropolis_sweep(
            SpinState.uniform(L=4, n_component=1),
            HeisenbergModel(),
            temperature=1.0,
            coupling=1.0,
            field=0.0,
            rng=np.random.default_rng(TOHUM),
        )
    with pytest.raises(InvalidInputError, match="n_component"):
        metropolis_sweep(
            SpinState.uniform(L=4, n_component=3),
            IsingModel(),
            temperature=1.0,
            coupling=1.0,
            field=0.0,
            rng=np.random.default_rng(TOHUM),
        )


@pytest.mark.parametrize("gecersiz", [0.0, -1.0, float("nan"), float("inf")])
def test_GECERSIZ_SICAKLIK_REDDEDILIYOR(gecersiz):
    """T <= 0 Boltzmann çarpanını anlamsız kılar; T = 0'da sıfıra bölünür."""
    with pytest.raises(InvalidInputError, match="sicaklik"):
        metropolis_sweep(
            SpinState.uniform(L=4, n_component=1),
            IsingModel(),
            temperature=gecersiz,
            coupling=1.0,
            field=0.0,
            rng=np.random.default_rng(TOHUM),
        )


@pytest.mark.parametrize(
    "ad, dizi",
    [
        ("field", 0.3 * (-1.0) ** np.add.outer(np.arange(8), np.arange(8))),
        ("field", np.array([0.3])),
        ("coupling", np.ones((8, 8))),
        ("field", [[0.3 * (-1.0) ** (i + j) for j in range(8)] for i in range(8)]),  # liste: ndarray değil
        ("coupling", (1.0,)),
    ],
)
def test_SKALER_OLMAYAN_field_ve_coupling_REDDEDILIYOR(ad, dizi):
    """Dizi SESSİZCE yayınlanırdı. ÖLÇÜLDÜ (27.09.2026, kapı YOKKEN, L = 8,
    T = 2, tohum 5, 50 sweep): kademeli (L, L) alan istisnasız koşuyor ve
    `energy_series` (50, 8, 8) çıkıyordu (değerler −124.2 / −91.8, doğru
    skaler −108.6); (1,) alan (50, 1) şekilli seri, (L, L) coupling (50, 8, 8).
    Depoda meşru dizi kullanımı YOK (grep: her `field=`/`coupling=` skaler).
    Kapı iki giriş kapısında da (`run_metropolis`, `metropolis_sweep`) durur;
    liste/demet de reddedilir (`isinstance(ndarray)` diye daraltan mutant düşer)."""
    girdi = {"coupling": 1.0, "field": 0.0, ad: dizi}
    ortak = dict(temperature=2.0, rng=np.random.default_rng(5), **girdi)
    with pytest.raises(InvalidInputError, match=f"{ad} SKALER"):
        run_metropolis(SpinState.uniform(L=8, n_component=1), IsingModel(), n_sweep=1, n_burn=0, **ortak)
    with pytest.raises(InvalidInputError, match=f"{ad} SKALER"):
        metropolis_sweep(SpinState.uniform(L=8, n_component=1), IsingModel(), **ortak)


@pytest.mark.parametrize("n_sweep, n_burn", [(0, 10), (-1, 0), (10, -1)])
def test_GECERSIZ_KOSUM_UZUNLUGU_REDDEDILIYOR(n_sweep, n_burn):
    """`n_sweep = 0` boş bir seri döndürürdü ve `mean()` NaN verirdi."""
    with pytest.raises(InvalidInputError):
        run_metropolis(
            SpinState.uniform(L=4, n_component=1),
            IsingModel(),
            temperature=1.0,
            coupling=1.0,
            field=0.0,
            rng=np.random.default_rng(TOHUM),
            n_sweep=n_sweep,
            n_burn=n_burn,
        )


def test_SONUC_HASHLENEBILIR_ve_OLCULMEYEN_ALANLAR_None():
    """`eq=False` tuzağı (Task 1) burada da geçerli; ve `None` = "ölçülmedi".

    `SamplingResult`in `energy_series` ve `acceptance_rate` alanları
    varsayılan olarak `None`dur. `None` SIFIR DEĞİLDİR: Modül 1B'nin
    kararlılık alanı dersinin aynısı — "analiz edilmedi" ile "kararlı" aynı
    değerle gösterilemez. Metropolis ikisini de doldurur; Wolff (Task 3)
    `energy_series`i dolduramaz ve `None` bırakacaktır.
    """
    bos = SamplingResult(
        magnetisation_series=np.zeros(3),
        final_state=SpinState.uniform(L=4, n_component=1),
        n_sweep=3,
        algorithm="test",
    )
    assert bos.energy_series is None
    assert bos.acceptance_rate is None
    assert isinstance(hash(bos), int)

    dolu = run_metropolis(
        SpinState.uniform(L=4, n_component=1),
        IsingModel(),
        temperature=2.0,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(TOHUM),
        n_sweep=5,
        n_burn=1,
    )
    assert dolu.energy_series is not None
    assert dolu.energy_series.shape == (5,)
    assert dolu.acceptance_rate is not None


# --------------------------------------------------------------------------
# `slow` ISARETLERININ GEREKCESI -- OLCUM, DEVRALINAN ESIK DEGIL
# --------------------------------------------------------------------------
#
# Depoda devralinabilir bir esik YOKTUR (`tests/tb/test_downfold_cipa.py`de
# olculdu: isaretli ve isaretsiz bantlar tamamen cakisiyor). Bu yuzden esik
# yeniden OLCULDU (22.08.2026, bu depoda, Task 2 dahil):
#
#     -m "not slow" : 966 test,  94.31 s -- en yavas call 5.06 s
#     -m slow       :  53 test, 341.59 s -- en yavas call 34.64 s
#
# ESIK SECILDI: kendi call suresi 2.0 s ve USTUNDE olan durum `slow`
# isaretlenir. Task 2'nin olculen call sureleri:
#
#     2.45 s  test_mermin_wagner  T=1.0     <- ISARETSIZ (asagidaki istisna)
#     2.39 s  test_mermin_wagner  T=1.4     <- slow
#     2.30 s  test_mermin_wagner  T=0.8     <- slow
#     1.59 s  test_ALANSIZ_ESLESME (Langevin)
#     1.20 s  test_KABUL_ORANI
#     1.05 s  test_TAM_SAYIM h=0.0
#     1.04 s  test_TAM_SAYIM h=0.3
#     0.95 s  test_ALANSIZ_ESLESME (tanh)
#     0.35 s  test_ISING_MIKNATISLANMASI_DORT_HANEDE...
#     0.21 s  test_ISING_TABAN_MIKNATISLANMASI (her T)
#
# TEK ISTISNA VE GEREKCESI: `test_mermin_wagner.py`nin T = 1.0 hali. O, beyan
# edilen gecerlilik penceresinin (T >~ 1.0) MERKEZIDIR ve modulun en guclu
# cipasidir; varsayilan seritte hic kosmamasi, "yesil gorunen ama olcmeyen"
# durumun kendisi olurdu. Bedeli olculdu: 2.34 s ile hizli seridin BESINCI
# en yavas testi, ve tavani (5.06 s) yarisindan az. Uc parametrenin ucu birden
# isaretlenseydi Mermin-Wagner varsayilan seritten TAMAMEN duserdi.
#
# tests/spin/ serit dagilimi: -m "not slow" 45 test 9.18 s, -m slow 2 test
# 4.82 s.
