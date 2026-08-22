"""Kafes: spin uzayından BAĞIMSIZ.

SINIF: TRANSKRİPSİYON DENETİMİ. Bu dosya tanımları kilitler, fizik ölçmez.
Buradaki hiçbir sayı bir öngörü değildir; hepsi ya elle sayılabilir bir
kombinatorik (dört komşu, çift sayım) ya da tanımın kendisidir.

GEÇERLİLİK PENCERESİ — BEYAN EDİLİYOR:
- Kafes kare ve periyodiktir; testler L >= 4 kullanır. L = 2'de karşıt
  komşular AYNI siteye denk gelir (`np.roll(+1)` ile `np.roll(-1)` çakışır)
  ve bağ çift katsayıyla sayılır. Bu, `lattice.py` dokstringinde beyan edilen
  bir yaklaşımdır; testler onu ölçmez, ondan kaçınır.
- `magnetisation` ve `total_energy` KLASİK spinler içindir; kuantum
  beklenen değer değildir (spec §1.1).
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.spin.lattice import (
    SpinState,
    magnetisation,
    neighbour_sum,
    total_energy,
)
from qlab.errors import InvalidInputError


def test_KOMSU_TOPLAMI_periyodik_ve_DORT_terimli():
    """Tekbiçim kafeste her sitenin komşu toplamı tam 4 spindir.

    Dört yönün biri düşerse bu sayı 3 olur; kenar sarmazsa köşelerde 2 veya 3
    olur. Yani tek satırlık bu iddia iki ayrı hatayı birden görür.
    """
    state = SpinState.uniform(L=6, n_component=1)
    toplam = neighbour_sum(state.spins)
    assert toplam.shape == state.spins.shape
    assert np.allclose(toplam, 4.0)


def test_KOMSU_TOPLAMI_KENARI_SARIYOR():
    """Periyodik sınır: (0,0)'ın komşuları (L-1,0) ve (0,L-1)'i İÇERİR.

    Tek bir siteyi -1 yapıp komşu toplamlarına bakmak bunu doğrudan gösterir:
    (L-1,0), (1,0), (0,L-1), (0,1) sitelerinin toplamı 4 yerine 2 olmalıdır
    (bir komşu +1'den -1'e döndü). Uzaktaki (2,2) sitesi 4'te kalır.

    Bu test aynı zamanda "np.roll yönü ters çevrildi" mutantını da öldürür:
    yön ters çevrilince bir komşu İKİ KEZ sayılır ve toplam 2 yerine 0 olur.
    """
    state = SpinState.uniform(L=5, n_component=1)
    spins = state.spins.copy()
    spins[0, 0, 0] = -1.0
    toplam = neighbour_sum(spins)[..., 0]
    for i, j in ((4, 0), (1, 0), (0, 4), (0, 1)):
        assert toplam[i, j] == pytest.approx(2.0)
    assert toplam[2, 2] == pytest.approx(4.0)


def test_TEKBICIM_ISING_ENERJISI_analitik():
    """E = -J * 2 * N  (her site iki bağ taşır, çift sayım yok) - h * N.

    L=6, J=1, h=0: N = 36, E = -72. Komşu toplamı her bağı iki kez sayar;
    0.5 çarpanı düşerse -144 çıkar ve bu test kırmızıya döner.
    h=0.5 eklendiğinde alan terimi -0.5 * 36 = -18.
    """
    state = SpinState.uniform(L=6, n_component=1)
    assert total_energy(state, coupling=1.0, field=0.0) == pytest.approx(-72.0)
    assert total_energy(state, coupling=1.0, field=0.5) == pytest.approx(-72.0 - 18.0)


def test_ALAN_SON_BILESENE_yani_z_YE_etki_ediyor():
    """Alan terimi s^z'ye, yani SON bileşene bağlanır.

    Bu testin varlık sebebi ölçülmüş bir kördür: n_component = 1'de son bileşen
    ile İLK bileşen aynı şeydir, dolayısıyla yalnız Ising'le çalışan bir test
    `spins[..., 0]` mutantını GÖREMEZ. Heisenberg durumunda (ilk iki bileşen
    sıfır) fark ortaya çıkar: doğru kodda alan terimi -h*N, mutantta 0'dır.

    L=4, n_component=3, J=1, h=0.25: N = 16,
    etkileşim = -0.5 * 1 * (16 sitede s.(4s) = 64) = -32,
    alan = -0.25 * 16 = -4  =>  E = -36.
    """
    state = SpinState.uniform(L=4, n_component=3)
    assert total_energy(state, coupling=1.0, field=0.0) == pytest.approx(-32.0)
    assert total_energy(state, coupling=1.0, field=0.25) == pytest.approx(-36.0)


def test_MIKNATISLANMA_bilesen_bazinda():
    """Ortalama (toplam DEĞİL) spin, ve YÖNÜ de kilitleniyor.

    Yalnız normu sınamak yetmez: `uniform` ilk bileşeni 1 yapsaydı norm yine
    1 olurdu. Yön iddiası o mutantı öldürür.
    """
    state = SpinState.uniform(L=4, n_component=3)
    m = magnetisation(state)
    assert m.shape == (3,)
    assert np.linalg.norm(m) == pytest.approx(1.0)
    assert np.allclose(m, [0.0, 0.0, 1.0])


def test_MIKNATISLANMA_ISARETLERI_GOTURUYOR():
    """Yarısı ters çevrilmiş kafeste mıknatıslanma sıfırdır.

    `abs` alan veya kareleyen bir tahminci burada 0 yerine 1 verirdi.
    """
    state = SpinState.uniform(L=4, n_component=1)
    spins = state.spins.copy()
    spins[:2, :, 0] = -1.0
    assert magnetisation(SpinState(spins)) == pytest.approx(np.zeros(1))


def test_KUCUK_L_REDDEDILIYOR():
    """L = 1'de site kendi komşusudur; sessizce anlamsız enerji üretirdi."""
    with pytest.raises(InvalidInputError, match="L en az 2"):
        SpinState.uniform(L=1, n_component=1)


def test_GECERSIZ_SEKIL_REDDEDILIYOR():
    """(L, L, n_component) dışındaki her şey erken ve gürültülü reddedilir."""
    with pytest.raises(InvalidInputError, match="L, n_component"):
        SpinState(np.zeros((4, 4)))
    with pytest.raises(InvalidInputError, match="L, n_component"):
        SpinState(np.zeros((4, 5, 1)))


def test_DURUM_HASHLENEBILIR_kaliyor():
    """`eq=False` bilinçlidir; deponun iki kez ödediği bir tuzağı kapatır.

    ndarray alanlı bir `frozen=True` dataclass'ta üretilen `__eq__` bir DİZİ
    döndürür ve üretilen `__hash__` `TypeError` verir (aynı gerekçe
    `engine/md/state.py` ve `engine/hf/integrals.py`'de yazılı). `eq=False`
    kaldırılırsa bu test kırmızıya döner.
    """
    state = SpinState.uniform(L=4, n_component=3)
    assert isinstance(hash(state), int)
    assert state == state
    assert state != SpinState.uniform(L=4, n_component=3)


def test_L_ve_BILESEN_SAYISI_dizinin_seklinden_okunuyor():
    """Şekil tek doğruluk kaynağıdır; ayrıca liste girdisi de kabul edilir."""
    state = SpinState([[[1.0], [-1.0]], [[-1.0], [1.0]]])
    assert state.L == 2
    assert state.n_component == 1
    assert state.spins.dtype == np.float64
