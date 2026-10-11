"""Anharmoniklik ve ayrışma -- manşet hikâyenin klasik sürümü.

BU DOSYANIN İDDİA ETMEDİĞİ ŞEY: burada ölçülen hiçbir sayı H2 FİZİĞİNİ
doğrulamaz. Morse'un `a` parametresi omega_e'den türetilmiştir (dairesellik,
tasarım §4.3). Doğrulanan şey, Morse potansiyelinin klasik dinamiğinin
analitik sonucuna uymasıdır.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.forcefields import Morse
from qlab.engine.md.integrator import run
from qlab.engine.md.state import MDState

from .h2_morse_sabitleri import H2_A, H2_DE, H2_MU, H2_RE

M_H = 1837.152647
T0 = 313.322876909  # küçük genlik periyodu, atomik zaman birimi


def _iki_atom(uzunluk: float, bagil_hiz: float = 0.0) -> MDState:
    return MDState(
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, uzunluk]]),
        velocities=np.array([[0.0, 0.0, -bagil_hiz / 2], [0.0, 0.0, bagil_hiz / 2]]),
        masses=np.array([M_H, M_H]),
    )


def _bag_uzunlugu(yorunge) -> np.ndarray:
    return np.linalg.norm(
        yorunge.positions[:, 1, :] - yorunge.positions[:, 0, :], axis=1
    )


def _olculen_periyot(yorunge) -> float:
    """TAM periyot: AYNI YÖNDEKİ geçişler arası.

    Task 4'ün harmonik `_periyot`u `2*mean(diff)` kullanır ve orada doğrudur,
    çünkü harmonik salınım simetriktir. MORSE ASİMETRİKTİR -- ölçüldü,
    E/De=0.10'da dış yarı 198.959, iç yarı 131.308 a.u. Geçiş aralıkları
    [dış, iç, dış, iç, ...] diye alternatiflenir; `2*mean` ancak aralık sayısı
    ÇİFT iken doğru sonucu verir. TEK sayıda bir yarı fazla sayılır ve tahmin
    kayar: ölçüldü, 41 aralıkta oran 0.943739 çıkıyor (analitik 0.948683).
    Bu yüzden birer atlayarak TAM periyotlar alınır.
    """
    r = _bag_uzunlugu(yorunge) - H2_RE
    t = yorunge.times
    gecis = np.nonzero(np.diff(np.sign(r)) != 0)[0]
    anlar = [
        t[i] + (t[i + 1] - t[i]) * abs(r[i]) / (abs(r[i]) + abs(r[i + 1]))
        for i in gecis
    ]
    return float(np.mean(np.diff(np.asarray(anlar)[::2])))


@pytest.mark.parametrize(
    ("enerji_orani", "beklenen_oran"),
    [(0.10, 0.948683), (0.50, 0.707107), (0.90, 0.316228)],
)
def test_anharmoniklik_frekansi_dusuruyor(enerji_orani, beklenen_oran):
    """TAM klasik sonuç: omega(E) = omega0 * sqrt(1 - E/De).

    ÖLÇÜLDÜ (800 adım/periyot, düzeltilmiş tahminciyle):
    E/De = 0.10 -> 0.948686 (analitik 0.948683), sapma 3.39e-06, marj 590x
    E/De = 0.50 -> 0.707120 (analitik 0.707107), sapma 1.29e-05, marj 155x
    E/De = 0.90 -> 0.316508 (analitik 0.316228), sapma 2.80e-04, marj 7.1x
    EN DAR MARJ E/De=0.90'DADIR ve adım sayısına duyarlıdır: 400 adım/periyotta
    marj 1.8x'e düşer (sapma 1.12e-03), 1600'de 6.99e-05, 3200'de 1.73e-05.
    Bu yüzden 800 sabitlendi. Sınır 2e-3, çünkü dönüş noktasına yaklaştıkça
    periyot ölçümü doğal olarak duyarlılaşır.

    BAĞIMSIZ KUADRATÜR: T = 2*integral dr/sqrt(2(E-V)/mu) ile aynı üç oran
    0.948684 / 0.707107 / 0.316228 çıkıyor -- yani Task 4 ve Task 5 sağlam,
    eskiden buradaki kırmızılığın sebebi tahminciydi.

    HANGİ HATAYI YAKALAR: Morse'u harmonikle karıştırmak -- harmonikte frekans
    genlikten BAĞIMSIZDIR, oran 1.000000 çıkar ve üç durum da kırılır.
    """
    r_donus = H2_RE - np.log(1.0 + np.sqrt(enerji_orani)) / H2_A
    durum = _iki_atom(r_donus)
    alan = Morse(d_e=H2_DE, r_e=H2_RE, a=H2_A)
    periyot_tahmini = T0 / np.sqrt(1.0 - enerji_orani)
    yorunge = run(durum, alan, dt=periyot_tahmini / 800, n_steps=800 * 20, save_every=1)
    assert T0 / _olculen_periyot(yorunge) == pytest.approx(
        beklenen_oran, abs=2e-3, rel=0.0
    )


def test_esigin_ALTINDA_bag_kopmuyor():
    """E < De: bağ bağlıdır ve dönüş noktası analitikten okunur.

    ÖLÇÜLEN dönüş noktaları (tasarım §4.3): E/De = 0.90 -> r_dis = 4.28717
    bohr. Bağ o noktadan öteye GİTMEZ.

    KOŞU UZUNLUĞU ÖLÇÜT BELİRLER: periyot = T0/sqrt(1-E/De) ıraksar; bir tam
    periyot görmeden "bağlı" denemez. E/De = 0.90 için periyot 3.16x uzar.
    """
    orani = 0.90
    r_donus = H2_RE - np.log(1.0 + np.sqrt(orani)) / H2_A
    alan = Morse(d_e=H2_DE, r_e=H2_RE, a=H2_A)
    periyot = T0 / np.sqrt(1.0 - orani)
    yorunge = run(_iki_atom(r_donus), alan, dt=periyot / 400, n_steps=400 * 5, save_every=1)
    r_max = float(_bag_uzunlugu(yorunge).max())
    # ÇİFT TARAFLI: ölçüldü, doğru Morse 4.280174 (analitik dış dönüş 4.287169);
    # aynı k'li HARMONİK bağ mutantı r_max = 2.049471 verir ve tek taraflı
    # "< 4.4" sınırını geçerdi -- alt sınır onu kırmızı yapıyor.
    assert 4.2 < r_max < 4.4, f"ölçülen dış dönüş noktası {r_max:.4f} bohr"


def test_esigin_USTUNDE_bag_kopuyor():
    """E > De: atomlar ayrılır ve geri dönmez.

    "Kopmuş" tanımı bilinçlidir: TOPLAM ENERJİ > De ölçütü KULLANILMAZ, çünkü
    o tanım dinamikten değil başlangıç koşulundan gelir ve testi totolojiye
    çevirirdi. Ölçüt DAVRANIŞSAL: bağ uzunluğu monoton büyür.

    SAHTE AYRIŞMA TUZAĞI (tasarım §4.3): gevşek dt ile bağ SAHTE kopar --
    ölçüldü, E/De = 0.99 ve dt = 20 (157 adım/periyot, "makul" görünüyor) ile
    max|dE| De'nin yalnızca %2.9'u ve r_max 479.69 bohr. Bu test bu yüzden
    dt = T0/400 kullanır VE enerji korunumunu ayrıca sınar.
    """
    orani = 1.20
    hiz = float(np.sqrt(2.0 * orani * H2_DE / H2_MU))
    alan = Morse(d_e=H2_DE, r_e=H2_RE, a=H2_A)
    yorunge = run(
        _iki_atom(H2_RE, bagil_hiz=hiz), alan, dt=T0 / 400, n_steps=400 * 20, save_every=10
    )
    r = _bag_uzunlugu(yorunge)
    assert r[-1] > 20.0, f"bağ kopmalıydı, son uzunluk {r[-1]:.3f} bohr"
    assert np.all(np.diff(r[len(r) // 2 :]) > 0.0), "kopan bağ geri dönmemeli"
    toplam = yorunge.kinetic + yorunge.potential
    sapma = float((toplam.max() - toplam.min()) / H2_DE)
    assert sapma < 1e-3, f"sahte ayrışma riski: enerji sapması De'nin %{sapma*100:.2f}'i"
