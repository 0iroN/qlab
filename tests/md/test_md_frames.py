"""MD zaman serisi çizim yükü."""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.forcefields import Harmonic
from qlab.engine.md.integrator import run
from qlab.engine.md.state import MDState
from qlab.observable.md_obs import instantaneous_temperature
from qlab.viz.md_frames import build_md_payload


def _yorunge():
    """İKİ parçacık, TEK değil.

    ÖLÇÜLDÜ: tek parçacıkla ndof = 3*1 - 3 = 0 oluyor ve
    `build_md_payload`'ın VARSAYILAN dalı (`n_constraints=3`) her karede
    `InvalidInputError: serbestlik derecesi kalmadı` fırlatıyordu -- iki test
    kırmızı doğuyordu. İki parçacıkla ndof = 6 - 3 = 3 ve varsayılan dal
    GERÇEKTEN koşuluyor (aşağıdaki parametre-iletimi testinin ön koşulu).
    Gerçek H2 ile de tutarlı.
    """
    durum = MDState(
        positions=np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        velocities=np.zeros((2, 3)),
        masses=np.full(2, 918.576324),
    )
    alan = Harmonic(k=0.3693947966, center=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))
    return run(durum, alan, dt=313.322876909 / 200, n_steps=400, save_every=1), durum.masses


def test_zaman_femtosaniyeye_cevriliyor():
    """Dönüşüm SINIRDA yapılır; motor içi atomik zaman birimidir.

    Sayı LİTERAL: 1 atomik zaman birimi = 0.024188843265857 fs.
    """
    yorunge, kutleler = _yorunge()
    yuk = build_md_payload(yorunge, kutleler)
    assert yuk.times_fs[-1] == pytest.approx(
        yorunge.times[-1] * 0.024188843265857, rel=1e-8, abs=0.0
    )


def test_toplam_enerji_bilesenlerin_toplami():
    """Alanların KİMLİĞİ de kilitlenir, yalnız toplamı değil.

    ÖLÇÜLDÜ: `MDPayload` içinde `kinetic` ile `potential` TAKAS edilse üç
    testin hiçbiri kırılmıyordu -- `total == kinetic + potential` aynı
    ifadeyi kendisiyle karşılaştırıyor (bit farkı 0.0, atol=1e-15 hiç
    devreye girmiyor). Yön testi takası kırar: x0 != 0, v0 = 0 ile ilk
    karede kinetik TAM sıfır, potansiyel pozitif.
    """
    yorunge, kutleler = _yorunge()
    yuk = build_md_payload(yorunge, kutleler)
    np.testing.assert_allclose(
        yuk.total, yuk.kinetic + yuk.potential, rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(yuk.kinetic, yorunge.kinetic, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(yuk.potential, yorunge.potential, rtol=0.0, atol=0.0)
    assert yuk.kinetic[0] == 0.0
    assert yuk.potential[0] > 0.0


def test_sicaklik_serisi_gozlem_katmaniyla_AYNI_tanimi_kullaniyor():
    """İki ayrı yerde tanımlanmış bir sıcaklık, ekranda gösterilen ile teste
    giren sayının sessizce ayrışması demektir."""
    yorunge, kutleler = _yorunge()
    yuk = build_md_payload(yorunge, kutleler, n_constraints=0)
    durum = MDState(
        positions=yorunge.positions[3],
        velocities=yorunge.velocities[3],
        masses=kutleler,
    )
    assert yuk.temperature_kelvin[3] == pytest.approx(
        instantaneous_temperature(durum, n_constraints=0), rel=1e-12, abs=0.0
    )


def test_kisit_sayisi_gercekten_iletiliyor():
    """`n_constraints` parametresini YOK SAYAN mutant onarılmış tabanda bile
    3 testi geçiyordu -- çünkü tek sıcaklık testi zaten `n_constraints=0`
    veriyor ve varsayılan dal hiç koşulmuyordu.

    İki parçacıkta ndof oranı 6/3 = 2 tam olarak; LİTERAL kilitleniyor.
    """
    yorunge, kutleler = _yorunge()
    yuk3 = build_md_payload(yorunge, kutleler)
    yuk0 = build_md_payload(yorunge, kutleler, n_constraints=0)
    assert yuk3.temperature_kelvin[3] / yuk0.temperature_kelvin[3] == pytest.approx(
        6 / 3, rel=1e-12, abs=0.0
    )
