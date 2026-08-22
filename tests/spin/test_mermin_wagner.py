"""MERMIN-WAGNER — Modül 4'ün en güçlü çıpası.

SINIF: DAİRESEL DEĞİL. İddia bir TEOREMDİR (Mermin & Wagner 1966): 2B'de
SÜREKLİ simetri T > 0'da uzun menzilli düzen vermez; AYRIK simetri verir.

Çıpayı dairesel olmaktan çıkaran şey şudur: iki model AYNI kafesten, AYNI
Metropolis çekirdeğinden, AYNI sıcaklıklardan geçer. Tek fark spin uzayıdır.
Yani farkı yaratan şey KOD DEĞİL, SİMETRİDİR — ve bunu `metropolis.py`nin
modeli bilmemesi garanti eder. O bilmeme `test_metropolis.py`de ayrıca
kilitlenir (`test_ORNEKLEYICI_UCUNCU_SPIN_UZAYINI_da_kosturuyor`,
`test_CEKIRDEK_MODELE_YALNIZ_PROTOKOLDEN_bakiyor`).

ÖLÇÜLDÜ (22.08.2026, bu depoda; `⟨|m|⟩`, L = 8, 12, 16, 24, 32, J = 1, h = 0,
n_burn = 500, n_sweep = 1500, tohum 20260822, soğuk (tekbiçim) başlangıç):

    T      Ising eğimi   Heisenberg eğimi   m_I(32)/m_H(32)
    0.8     +0.000033        -0.9196              6.99
    1.0     +0.000002        -1.0545             11.35
    1.4     +0.000174        -0.9580             16.04

Ising'in mıknatıslanması DÖRT HANEDE L'den bağımsız: T = 1.0'da L = 8 için
0.999208, L = 32 için 0.999250.

TOHUM YAYILIMI ÖLÇÜLDÜ (8 tohum, aynı koşum uzunluğu) — sınırlar buradan
gerekçelendirilir, elle seçilmedi:

    T     Heisenberg eğimi ort ± std   en GEVŞEK tohum   sınır   sınırı geçen
    0.8      -0.8690 ± 0.0754            -0.7438         -0.6        8/8
    1.0      -1.0250 ± 0.0846            -0.8690         -0.8        8/8
    1.4      -1.0084 ± 0.0422            -0.9580         -0.8        8/8

    m_I(32)/m_H(32) en küçük tohum: T=1.0 → 9.18, T=1.4 → 16.04 (sınır 8.0).
    Ising eğiminin 8 tohumdaki en büyük mutlak değeri 0.00079 (T=1.4);
    sınır 0.02, yani 25 kat marj.

PLANDAN SAPMA — VE SEBEBİ ÖLÇÜLDÜ. Plan referansı: T = 0.8/1.0/1.4 için
Heisenberg eğimi -0.7365 / -0.9644 / -1.0659, oran 4.94 / 10.60 / 17.55.
Yeniden ölçüldü ve TUTMADI. Sebep aranıp bulundu ve bu kez Task 1'dekinden
FARKLI: Task 1'de sapma bir kat hatasıydı (cebirsel, tohumdan bağımsız);
burada sapma TOHUM GÜRÜLTÜSÜ mertebesindedir. Planın üç eğiminden ikisi
(-0.9644 ve -1.0659) yukarıdaki 8-tohumluk aralığın İÇİNDE, üçüncüsü
(-0.7365) aralığın gevşek ucunun 0.007 dışındadır. Oranlar ise sistematik
olarak planınkinin ÜSTÜNDE çıktı (6.4 / 11.4 / 17.8 vs 4.94 / 10.60 / 17.55);
bu da beklenen yönde: Heisenberg `⟨|m|⟩`si soğuk başlangıçtan aşağı doğru
gevşer, yani daha uzun bir burn-in daha KÜÇÜK m_H ve daha BÜYÜK oran verir.
Hüküm: planın sayıları yanlış değil, YENİDEN ÜRETİLEBİLİR DEĞİL — ve bu
dosyadaki her sayı tohumuyla birlikte yazılmıştır.

GEÇERLİLİK PENCERESİ — BEYAN EDİLİYOR, GEVŞETME DEĞİL:

Çıpa T ≳ 1.0'da geçerlidir. T = 0.8 AYRI bir parametre olarak ve GEVŞEK
sınırla (-0.6) girer, ve oran iddiası orada HİÇ kurulmaz. Sebep ölçüldü:
2B Heisenberg'de korelasyon uzunluğu ξ ~ exp(2πJ/T) ile patlar ve düşük
T'de "L >> ξ" koşulu L = 32'de SAĞLANMAZ. Ölçülen imzası da bu: T = 0.8'de
eğim -0.8690 ± 0.0754 ile -1'e ulaşamıyor ve tohum yayılımı T = 1.4'ün
(±0.0422) neredeyse iki katı. Oran ise T = 0.8'de 8 tohumda 5.52'ye kadar
iniyor — yani `> 8.0` iddiası orada YANLIŞ olurdu ve kurulmadı.

Bu, sınırın "test geçsin diye" gevşetilmesi DEĞİLDİR: gevşetme aynı hükmü
zayıf sınırla tekrarlamaktır, pencere beyanı ise hükmün NEREDE kurulduğunu
söylemektir. Bu depoda "pencere adlandırılmazsa hüküm dışına taşınır" dersi
beş kez çıktı; burada önlem olarak yazıldı.

KAPSAM DIŞI: bu dosya Ising'in mutlak mıknatıslanmasını Onsager-Yang ile
karşılaştırmaz — o çıpa `test_metropolis.py`dedir ve `np.clip` sınırı gibi
kabul kuralını bozan mutantları orada yakalar. Burada Ising yalnızca
"L'den BAĞIMSIZ" iddiasını taşır.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.spin.lattice import SpinState
from qlab.engine.spin.metropolis import run_metropolis
from qlab.engine.spin.models import HeisenbergModel, IsingModel

LADDER = (8, 12, 16, 24, 32)
TOHUM = 20260822
N_BURN = 500
N_SWEEP = 1500


def _ortalama_mutlak_m(model, temperature: float, L: int) -> float:
    """Tek bir (model, T, L) koşumunun `⟨|m|⟩`si — tohum SABİT ve AÇIK."""
    sonuc = run_metropolis(
        SpinState.uniform(L=L, n_component=model.n_component),
        model,
        temperature=temperature,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(TOHUM),
        n_sweep=N_SWEEP,
        n_burn=N_BURN,
    )
    return float(sonuc.magnetisation_series.mean())


def _log_log_egim(ms: list[float]) -> float:
    """`d ln⟨|m|⟩ / d ln L` — en küçük kareler, `np.polyfit`e bağımlı DEĞİL."""
    x = np.log(np.asarray(LADDER, dtype=float))
    y = np.log(np.asarray(ms, dtype=float))
    return float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))


@pytest.mark.parametrize(
    "temperature, heisenberg_sinir, oran_sinir, olculen_egim, olculen_oran",
    [
        pytest.param(0.8, -0.6, None, -0.9196, 6.99, marks=pytest.mark.slow),
        pytest.param(1.0, -0.8, 8.0, -1.0545, 11.35),
        pytest.param(1.4, -0.8, 8.0, -0.9580, 16.04, marks=pytest.mark.slow),
    ],
)
def test_SUREKLI_SIMETRI_SONUYOR_AYRIK_SIMETRI_KALIYOR(
    temperature, heisenberg_sinir, oran_sinir, olculen_egim, olculen_oran
):
    """İki model, tek çekirdek, zıt hüküm.

    `slow` İŞARETİ ÖLÇÜM: üç parametrenin call süreleri dosyanın sonundaki
    notta. T = 1.0 BİLEREK işaretsiz bırakıldı — o, beyan edilen geçerlilik
    penceresinin (T ≳ 1.0) merkezidir ve modülün en güçlü çıpasının varsayılan
    şeritte hiç koşmaması, "yeşil görünen ama ölçmeyen" durumun ta kendisi
    olurdu.

    Sınırların GERİ OKUNAMAZLIĞI: -0.8 ve 8.0 hiçbir yerden girdi değildir.
    Simülasyona giren şeyler J, h, T, L ve `step`tir; eğim bunların hiçbirinden
    okunamaz, örneklemeden ÇIKAR. Mermin-Wagner teoremi de simülasyona
    girmez — kod 2B'de sürekli simetrinin düzen vermeyeceğini hiçbir yerde
    bilmez.
    """
    ising_m = [_ortalama_mutlak_m(IsingModel(), temperature, L) for L in LADDER]
    heisenberg_m = [
        _ortalama_mutlak_m(HeisenbergModel(), temperature, L) for L in LADDER
    ]

    ising_egim = _log_log_egim(ising_m)
    heisenberg_egim = _log_log_egim(heisenberg_m)
    oran = ising_m[-1] / heisenberg_m[-1]

    # AYRIK simetri: L'den bagimsiz (8 tohumda en buyuk |egim| 0.00079).
    assert abs(ising_egim) < 0.02, f"Ising egimi L'ye bagli cikti: {ising_egim}"

    # SUREKLI simetri: sonuyor.
    assert heisenberg_egim < heisenberg_sinir, (
        f"T={temperature}: Heisenberg egimi {heisenberg_egim:+.4f}, "
        f"sinir {heisenberg_sinir}"
    )

    # Yeniden uretilebilirlik bandi (FIZIK IDDIASI DEGIL, tohum kilidi):
    # 8-tohumluk std en buyuk 0.0846 idi; 0.25 onun ~3 katidir.
    assert heisenberg_egim == pytest.approx(olculen_egim, abs=0.25)

    if oran_sinir is not None:
        assert oran > oran_sinir, (
            f"T={temperature}: m_I(32)/m_H(32) = {oran:.2f}, sinir {oran_sinir}"
        )
        assert oran == pytest.approx(olculen_oran, rel=0.35)
    else:
        # T = 0.8 PENCERE DISI: oran iddiasi burada KURULMAZ ve neden
        # kurulmadigi olculdu -- 8 tohumda 5.52'ye kadar iniyor.
        assert oran < 8.0, (
            f"T={temperature} pencere disi kabul edilmisti ama oran {oran:.2f} "
            "cikti; pencere beyani yeniden olculmeli"
        )


def test_ISING_MIKNATISLANMASI_DORT_HANEDE_L_DEN_BAGIMSIZ():
    """Eğimden ayrı, DOĞRUDAN okunabilir hâli: iki uç L aynı sayıyı veriyor.

    Eğim testi bir uyum (fit) üzerinden konuşur; bu test ham sayıyı gösterir.
    ÖLÇÜLDÜ (T = 1.0, tohum 20260822): L = 8 → 0.999208, L = 32 → 0.999250.
    Fark 4.2e-05, yani `⟨|m|⟩`nin kendisinin 8 tohumdaki std'si (2.3e-05)
    mertebesinde.

    Bu test `slow` DEĞİL: ölçüldü, 0.35 s sürüyor (iki Ising koşumu).
    """
    kucuk = _ortalama_mutlak_m(IsingModel(), 1.0, 8)
    buyuk = _ortalama_mutlak_m(IsingModel(), 1.0, 32)
    assert kucuk == pytest.approx(buyuk, abs=1e-3)
    assert kucuk == pytest.approx(0.999208, abs=5e-5)
    assert buyuk == pytest.approx(0.999250, abs=5e-5)


# SURELER OLCULDU (pytest --durations, 22.08.2026): T=1.0 2.45 s, T=1.4
# 2.39 s, T=0.8 2.30 s, Ising-L-bagimsizligi 0.35 s. Esik 2.0 s secildi ve
# T=1.0 BILEREK isaretsiz birakildi; her ikisinin de tam gerekcesi
# `test_metropolis.py`nin sonundaki blokta (depo geneli serit olcumleriyle
# birlikte: -m "not slow" 966 test / 94.31 s / tavan 5.06 s).
