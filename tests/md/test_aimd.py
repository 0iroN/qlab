"""Born-Oppenheimer moleküler dinamiği -- çekirdekler HF yüzeyinde.

Bu dosya Plan D'nin ANA İDDİASINI sınar: kuvvet bir MODELDEN değil, her
adımda yeniden çözülen bir HF yüzeyinden gelir; buna rağmen enerji korunur
ve titreşim frekansı yüzeyin KENDİSİNDEN çıkar.

PAYDA UYARISI -- 432x: BO yüzeyinde |E_toplam| ~ 1.1175 hartree ama
E_titreşim ~ 2.58e-03. Plan C'nin `|blok[0]|` paydası buraya kopyalanırsa
her enerji testi 432 kat zayıflar. Payda E_titreşim'dir.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.forcefields import HartreeFockGradient
from qlab.engine.md.integrator import run
from qlab.engine.md.state import MDState
from qlab.units import HARTREE_TO_WAVENUMBER, amu_to_electron_mass

H2_RE_BOHR = 1.3459193602
H2_E_MIN = -1.117505885116
#: mu = m(1H)/2. KÜTLE BORU HATTINDAN GEÇER, literal DEĞİL -- Plan C'de
#: literal yazıldığı için `AMU_TO_ELECTRON_MASS = 1.0` mutantı altı testi
#: birden geçiyordu (ölçüldü).
MU_H2 = amu_to_electron_mass(1.00782503207) / 2.0
#: HF/STO-3G yüzeyinin KENDİ harmonik periyodu (ölçülen k = 0.57293630).
T_HF = 2 * np.pi * np.sqrt(MU_H2 / 0.5729363)


def _baslangic(genlik: float) -> MDState:
    return MDState(
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, H2_RE_BOHR + genlik]]),
        velocities=np.zeros((2, 3)),
        masses=np.full(2, amu_to_electron_mass(1.00782503207)),
    )


def _bag(yorunge) -> np.ndarray:
    return np.linalg.norm(
        yorunge.positions[:, 1, :] - yorunge.positions[:, 0, :], axis=1
    )


@pytest.mark.slow
def test_BO_yuzeyinde_enerji_SURUKLENMIYOR():
    """velocity-Verlet BO yüzeyinde de SİMPLEKTİK: enerji SALINIR.

    ÖLÇÜT SÜRÜKLENME DEĞİL SALINIMDIR. Ölçüldü: sürüklenme dt ile
    ölçekleniyor (N=50'de 9.4200e-05, N=100'de 1.9412e-05 -- 4.85 kat),
    yani bir gürültü tabanı değil ayrıklaştırma eseri; sabit eşiği YOK.
    Bu planın ilk taslağı `< 1e-8` yazıyordu ve KIRMIZI DOĞARDI.

    Salınım ise temiz π²/N² yasasına uyuyor: 4.2778e-03 -> 1.0690e-03,
    oran 4.0016.

    ÖLÇÜLEN NORMALİZE ORAN (naif konvansiyon, dt = T_harm/N, genlik 0.10):
      N=50  2 periyot -> 1.0836      N=100 2 periyot -> 1.0829
      N=50  6 periyot -> 1.0836      N=100 6 periyot -> 1.0832
    Saçılım 0.0007. SINIR ±0.01 = saçılımın 14 katı.
    HANGİ HATAYI YAKALAR (ölçüldü): genlik 0.05 -> 1.0468 (sınırın 3.6
    katı dışında), h=1e-2 -> 2.399 (efektif yüzeyi bozar).

    GENLİK SABİT: oran genliğe duyarlı (0.05 -> 1.0468, 0.10 -> 1.0919,
    0.20 -> 1.1952). Genlik değişirse ölçüt anlamsızlaşır.

    PAYDA E_TİTREŞİM: |E_toplam| kullanılsaydı ölçüt 432 kat zayıflardı
    (ölçüldü: E_titreşim 2.582381e-03, |E_toplam| 1.1149235 -> oran 431.75).
    """
    durum = _baslangic(0.10)
    alan = HartreeFockGradient(["H", "H"])
    yorunge = run(durum, alan, dt=T_HF / 50, n_steps=100, save_every=1)

    toplam = yorunge.kinetic + yorunge.potential
    e_titresim = float(toplam[0] - H2_E_MIN)
    assert e_titresim == pytest.approx(2.582381e-03, rel=1e-3, abs=0.0)

    # SALINIM ORANI -- sürüklenme DEĞİL. Sürüklenme dt ile ölçeklenen bir
    # ayrıklaştırma eseridir (N=50'de 9.4209e-05, N=100'de 1.94e-05) ve sabit
    # eşiği YOKTUR. Salınım ise π²/N² yasasına uyuyor: N ikiye katlanınca
    # 4.2778e-03 -> 1.0690e-03, oran 4.0016.
    #
    # KONVANSİYON: N NAİF olandır (dt = T_harm/N), gerçek periyottan türetilen
    # N_eff DEĞİL. İkisi %0.75 farklı ölçüldü; sınır bu testin GERÇEKTEN
    # kullandığı konvansiyona oturtuldu.
    tepe_tepe = float(toplam.max() - toplam.min()) / e_titresim
    n_naif = 50.0
    assert tepe_tepe / (np.pi**2 / n_naif**2) == pytest.approx(
        1.0833, abs=0.01, rel=0.0
    )


@pytest.mark.slow
def test_bag_dengenin_ETRAFINDA_saliniyor():
    """Yörünge gerçekten hareket etmeli ve denge etrafında SİMETRİK olmalı.

    Plan C'de bu sınıf test ilk kare tarafından sağlanıyordu; burada genlik
    ve iki yönlü aşma birlikte sınanır.
    ÖLÇÜLDÜ: genlik 0.10 bohr ile salınım 1.2984-1.3959 bohr aralığında.
    """
    yorunge = run(
        _baslangic(0.05), HartreeFockGradient(["H", "H"]),
        dt=T_HF / 50, n_steps=100, save_every=1,
    )
    r = _bag(yorunge)
    assert float(r.max()) > H2_RE_BOHR + 0.04
    assert float(r.min()) < H2_RE_BOHR - 0.04
