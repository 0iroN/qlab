"""İki model, TEK protokol. Farkı yaratan şey kod değil SİMETRİ.

SINIF: TRANSKRİPSİYON DENETİMİ. Bu dosya iki modelin TANIMINI ve ortak
protokolü kilitler; hiçbir fiziksel hüküm vermez. Mermin-Wagner çıpası
(Task 2) bu dosyanın kilitlediği paylaşıma DAYANIR: Ising ile Heisenberg aynı
kafesten ve aynı örnekleyiciden geçmezse "aynı kod, zıt hüküm" iddiası düşer.

GEÇERLİLİK PENCERESİ — BEYAN EDİLİYOR:
- Ölçülen nokta çarpımı sayıları YALNIZCA şu pencerede geçerlidir: L = 16
  (256 site), başlangıç durumu kuzey kutbu (+z), RNG tohumu 20260821, ve
  TEK bir öneri turu. Tohum değişirse sayı değişir — L=16'da step=2.0 için
  sekiz tohumda ölçülen aralık 0.196833 … 0.298552'dir. Testin SINIRLARI
  (0.99 ve 0.6) bu yayılımın çok dışında seçilmiştir; sayıların kendisi
  tohuma bağlı, HÜKÜM değildir.
- Bu dosya kabul oranı hakkında HİÇBİR ŞEY iddia etmez; kabul/red kararı
  `metropolis.py`nindir ve bu görevde henüz yoktur.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.spin.models import HeisenbergModel, IsingModel, SpinModel
from qlab.errors import InvalidInputError


def _kutup_kafesi(L: int, n_component: int) -> np.ndarray:
    """Hepsi son bileşene (kuzey kutbuna) bakan kafes."""
    spins = np.zeros((L, L, n_component))
    spins[..., -1] = 1.0
    return spins


def test_ISING_onerisi_HER_ZAMAN_isareti_ceviriyor():
    rng = np.random.default_rng(20260821)
    spins = np.ones((4, 4, 1))
    onerilen = IsingModel().propose(spins, rng)
    assert np.all(onerilen == -spins)


def test_HEISENBERG_onerisi_BIRIM_KUREDE_kaliyor():
    """Sürekli spin normu 1 olmak ZORUNDA -- normalize edilmemiş bir öneri
    enerjiyi sessizce kaydırırdı."""
    rng = np.random.default_rng(20260821)
    spins = _kutup_kafesi(5, 3)
    onerilen = HeisenbergModel().propose(spins, rng)
    assert onerilen.shape == spins.shape
    assert np.allclose(np.linalg.norm(onerilen, axis=2), 1.0)


def test_HEISENBERG_ADIMI_kucukse_oneri_YAKIN_kaliyor():
    """`step` kabul oranını ayarlar; küçük adım = küçük sapma.

    ÖLÇÜLDÜ (L=16, kutup başlangıcı, tohum 20260821 -- iki çağrı da AYNI
    tohumu kullanır, yani aralarındaki TEK fark `step`'tir):

        step=0.05 -> ortalama nokta çarpımı 0.997451
        step=0.60 -> 0.692146   (varsayılan)
        step=2.00 -> 0.259781

    Sınırlar 0.99 ve 0.6'dır ve şu ölçümle gerekçelendirilir: varsayılan
    step=0.6'nın ölçülen değeri 0.692146, yani İKİ sınırın da dışında kalır.
    Dolayısıyla `step` parametresini sessizce yok sayan (hep 0.6 kullanan)
    bir mutant her iki iddiada da yakalanır.

    NOT -- PLANDAN SAPMA: plan bu sayıları 0.99875 ve 0.34 diye veriyordu;
    yeniden ölçüldü ve tutmadı. Küçük adım açılımı bağımsız olarak doğruyu
    söylüyor: nokta çarpımı ~ 1 - step^2 (0.05 -> 0.997500, ölçülen L=1024'te
    0.997502), yani 0.99875 = 1 - step^2/2 bir kat hatasıdır.
    """
    yakin = HeisenbergModel(step=0.05).propose(
        _kutup_kafesi(16, 3), np.random.default_rng(20260821)
    )
    uzak = HeisenbergModel(step=2.0).propose(
        _kutup_kafesi(16, 3), np.random.default_rng(20260821)
    )
    kutup = _kutup_kafesi(16, 3)

    yakin_carpim = float(np.mean(np.sum(yakin * kutup, axis=2)))
    uzak_carpim = float(np.mean(np.sum(uzak * kutup, axis=2)))

    assert yakin_carpim == pytest.approx(0.997451, abs=5e-6)
    assert uzak_carpim == pytest.approx(0.259781, abs=5e-6)
    assert yakin_carpim > 0.99
    assert uzak_carpim < 0.6


def test_HEISENBERG_ONERISI_SITE_BASINA_BAGIMSIZ():
    """Gürültü site başına çekilir; tek bir dönmeyi tüm kafese uygulamak
    ayrıntılı dengeyi bozardı ve `size=spins.shape` düşerse bu olur."""
    onerilen = HeisenbergModel(step=0.6).propose(
        _kutup_kafesi(8, 3), np.random.default_rng(20260821)
    )
    duzlestirilmis = onerilen.reshape(-1, 3)
    assert not np.allclose(duzlestirilmis, duzlestirilmis[0])


def test_ONERI_GIRDIYI_DEGISTIRMIYOR():
    """İki model de YENİ dizi döndürür; yerinde değişiklik yapan bir uygulama
    Metropolis'in eski durumu geri koyabilmesini sessizce imkânsız kılardı."""
    for model in (IsingModel(), HeisenbergModel()):
        spins = _kutup_kafesi(4, model.n_component)
        onceki = spins.copy()
        model.propose(spins, np.random.default_rng(20260821))
        assert np.array_equal(spins, onceki)


def test_BILESEN_SAYISI_modelin_ozelligi():
    assert IsingModel().n_component == 1
    assert HeisenbergModel().n_component == 3


def test_ADIM_POZITIF_OLMALI():
    """step <= 0 hiç hareket etmeyen (veya ters yönde büyüyen) bir zincir
    demektir; sessizce kabul edilirse örnekleme donar."""
    for gecersiz in (0.0, -0.3):
        with pytest.raises(InvalidInputError, match="step pozitif"):
            HeisenbergModel(step=gecersiz)


def test_IKI_MODEL_TEK_CAGRI_YOLUNDAN_geciyor():
    """MERMIN-WAGNER ÇIPASININ ÖN KOŞULU, testle kilitleniyor.

    Aşağıdaki yardımcı, modelin ne olduğunu BİLMEZ: yalnız `n_component` ve
    `propose` görür. İki model de aynı imzadan geçip aynı şekli korumak
    zorundadır. Bu kilit düşerse Task 2'nin "aynı kod, zıt hüküm" iddiası
    dayanaksız kalır.
    """

    def tek_yol(model: SpinModel, L: int) -> np.ndarray:
        spins = _kutup_kafesi(L, model.n_component)
        return model.propose(spins, np.random.default_rng(20260821))

    for model in (IsingModel(), HeisenbergModel()):
        onerilen = tek_yol(model, 6)
        assert onerilen.shape == (6, 6, model.n_component)
        assert np.all(np.isfinite(onerilen))
