"""MD durumu ve kuvvet sağlayıcı sözleşmesi."""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.state import MDState
from qlab.errors import InvalidInputError


def _iki_atom() -> MDState:
    return MDState(
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        velocities=np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]),
        masses=np.array([1837.15, 1837.15]),
    )


def test_kinetik_enerji_elle_hesaplananla_ayni():
    """KE = sum 0.5 m v^2. Sayı LİTERAL: 2 * 0.5 * 1837.15 * 0.25 = 459.2875."""
    assert _iki_atom().kinetic_energy() == pytest.approx(459.2875, abs=1e-9, rel=0.0)


def test_kutle_merkezi_hareketi_cikarilinca_toplam_momentum_sifir():
    """Kütle merkezi sürüklenmesi sıcaklığa SAHTE katkı yapar.

    ndof = 3N-3 kullanılacaksa (Task 8) o üç serbestlik derecesi gerçekten
    çıkarılmış olmalı; yoksa sıcaklık sistematik olarak yüksek okunur.

    HIZLAR BİLEREK FARKLI: kütle-ağırlıklı v_com = (1*1+2*2+3*3)/6 = 2.333333,
    ağırlıksız düz ortalama 2.0. ÖLÇÜLDÜ -- üç hız da 1.0 iken (testin eski
    hali) kütle ağırlığını atlayan mutant 5 testin hepsini geçiyordu; farklı
    hızlarla o mutantın artık momentumu [2,0,0] oluyor ve test kırmızı.
    """
    durum = MDState(
        positions=np.zeros((3, 3)),
        velocities=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        masses=np.array([1.0, 2.0, 3.0]),
    )
    temiz = durum.remove_center_of_mass_motion()
    momentum = (temiz.masses[:, None] * temiz.velocities).sum(axis=0)
    np.testing.assert_allclose(momentum, np.zeros(3), rtol=0.0, atol=1e-12)


def test_uyusmayan_sekiller_reddediliyor():
    """Sessiz yayın (broadcast) yanlış sistemi simüle ettirir."""
    with pytest.raises(InvalidInputError, match="şekl"):
        MDState(
            positions=np.zeros((2, 3)),
            velocities=np.zeros((2, 3)),
            masses=np.ones(3),
        )
    with pytest.raises(InvalidInputError, match="şekl"):
        MDState(
            positions=np.zeros((2, 3)),
            velocities=np.zeros((3, 3)),
            masses=np.ones(2),
        )


@pytest.mark.parametrize("alan", ["positions", "velocities", "masses"])
@pytest.mark.parametrize("bozuk", [np.nan, np.inf])
def test_sonlu_olmayan_deger_reddediliyor(alan, bozuk):
    """`types.Atom` ile aynı gerekçe: nan/inf sessizce yayılır ve teşhisi bozar.

    ÜÇ ALAN DA SINANIR. ÖLÇÜLDÜ -- yalnız `positions` sınanırken sonluluk
    döngüsünden `hızlar` ve `kütleler` satırlarını silen mutant beş testi de
    geçiyordu.
    """
    kwargs = {
        "positions": np.zeros((1, 3)),
        "velocities": np.zeros((1, 3)),
        "masses": np.ones(1),
    }
    kwargs[alan] = kwargs[alan] + bozuk if alan == "masses" else kwargs[alan].copy()
    if alan != "masses":
        kwargs[alan][0, 0] = bozuk
    with pytest.raises(InvalidInputError, match="sonlu"):
        MDState(**kwargs)


def test_kutle_pozitif_olmali():
    with pytest.raises(InvalidInputError, match="kütle"):
        MDState(
            positions=np.zeros((1, 3)),
            velocities=np.zeros((1, 3)),
            masses=np.array([0.0]),
        )


def test_diziye_cevrilmeyen_girdi_tipli_hata_veriyor():
    """Liste kabul edilirse `kinetic_energy()` ham `TypeError` veriyor.

    ÖLÇÜLDÜ: liste girdi doğrulamayı GEÇİYOR, sonra
    `TypeError: list indices must be integers or slices, not tuple` --
    QLabError DEĞİL, yani Global Constraint "ham hata tüketiciye ulaşmaz"
    ihlali. `__post_init__` alanları ndarray'e çevirir.
    """
    durum = MDState(
        positions=[[0.0, 0.0, 0.0]],
        velocities=[[0.0, 0.0, 1.0]],
        masses=[2.0],
    )
    assert durum.kinetic_energy() == pytest.approx(1.0, abs=1e-12, rel=0.0)
    assert isinstance(durum.positions, np.ndarray)
