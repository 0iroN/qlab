"""Sıcaklık ve hız dağılımı göstergeleri."""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.state import MDState
from qlab.errors import InvalidInputError
from qlab.observable.md_obs import (
    SPEED_SHAPE_RATIO_MB,
    instantaneous_temperature,
    speed_shape_ratio,
    speed_shape_ratio_noise,
)
from qlab.units import KB_HARTREE_PER_KELVIN


def test_sicaklik_esbolusum_tanimini_izliyor():
    """T = 2*KE / (ndof * k_B). İki parçacık, ndof = 3*2 - 3 = 3.

    Hızlar öyle seçildi ki KE tam olarak 1.5 * k_B * 300 K olsun -> T tam
    olarak 300 K çıkmalı.
    """
    hedef = 300.0
    ke = 1.5 * KB_HARTREE_PER_KELVIN * hedef
    m = 1000.0
    v = float(np.sqrt(ke / m))  # iki parçacık zıt yönde: KE = 2 * 0.5 * m * v^2
    durum = MDState(
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        velocities=np.array([[0.0, 0.0, v], [0.0, 0.0, -v]]),
        masses=np.array([m, m]),
    )
    assert instantaneous_temperature(durum) == pytest.approx(hedef, rel=1e-12, abs=0.0)


def test_kisit_sayisi_sicakligi_degistiriyor():
    """ndof = 3N - n_constraints; kütle merkezi çıkarıldıysa 3 olmalı.

    ÖLÇÜLMÜŞ UYARI (tasarım §4.4): 3N ile 3N-3 arasındaki fark -1/N'dir
    (N=256'da %0.39) ve <T>'nin tohum saçılımı %2.15 -- yani bu hata
    İSTATİSTİKLE HİÇBİR N'DE yakalanamaz. Bu yüzden burada, doğrudan ve
    deterministik olarak kilitlenir.
    """
    durum = MDState(
        positions=np.zeros((4, 3)),
        velocities=np.array(
            [[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, -0.01, 0.0]]
        ),
        masses=np.ones(4) * 100.0,
    )
    serbest = instantaneous_temperature(durum, n_constraints=0)
    kisitli = instantaneous_temperature(durum, n_constraints=3)
    assert kisitli / serbest == pytest.approx(12 / 9, rel=1e-12, abs=0.0)


def test_kutle_merkezi_surukleniyorken_sicaklik_SAHTE_yuksek_okunuyor():
    """`n_constraints=3` varsayılanının ÖN KOŞULU: COM gerçekten çıkarılmış olmak.

    Eski fixture SAF ÖTELEMEYDİ (dört hız da [0.01,0.01,0.01]) -- iç hareket
    SIFIR, yani gerçek sıcaklık 0 K. Modül yine de 4210.33 K rapor ediyordu ve
    yukarıdaki 12/9 testi bunu GÖRMÜYORDU, çünkü oran her iki durumda da aynı.

    ÖLÇÜLDÜ: saf ötelemede instantaneous_temperature = 4210.333664 K;
    remove_center_of_mass_motion() sonrası KE = 0.0 ve T = 0.0 K.
    """
    suruklenen = MDState(
        positions=np.zeros((4, 3)),
        velocities=np.ones((4, 3)) * 0.01,
        masses=np.ones(4) * 100.0,
    )
    # SAYILAR LİTERAL ve HANGİ DALDAN geldikleri yazılı:
    #   varsayılan dal (n_constraints=3, ndof=9)  -> 4210.333664 K
    #   kısıtsız dal   (n_constraints=0, ndof=12) -> 3157.750248 K
    # Gerçek iç sıcaklık ise SIFIR -- hiçbiri onu göstermiyor.
    assert instantaneous_temperature(suruklenen) == pytest.approx(
        4210.333664, rel=1e-9, abs=0.0
    )
    assert instantaneous_temperature(suruklenen, n_constraints=0) == pytest.approx(
        3157.750248, rel=1e-9, abs=0.0
    )
    temiz = suruklenen.remove_center_of_mass_motion()
    assert instantaneous_temperature(temiz, n_constraints=0) == pytest.approx(
        0.0, abs=1e-9, rel=0.0
    )


def test_R4_maxwell_boltzmann_orneginde_5_bolu_3_veriyor():
    """R4 = <v^4>/<v^2>^2 -> 5/3, ÖLÇEKTEN BAĞIMSIZ şekil testi.

    NEDEN KS TESTİ DEĞİL (tasarım §4.4): ölçek aynı örnekten kestirilirse KS
    gerçek MB örneklerinde %5 yerine %0.10-0.53 red oranı veriyor -- yani
    "KS p > 0.05" ölçütü SONSUZA DEK YEŞİL kalır.

    Analitik gürültü: std(R4) = 1.217161/sqrt(n), altı örneklem boyutunda
    ölçülen/analitik oranı 0.958-1.002 ile doğrulandı. Sınır 4 sigma.
    """
    uretici = np.random.default_rng(20260811)
    n = 20000
    hizlar = uretici.normal(size=(n, 3)) * 3.7
    assert abs(speed_shape_ratio(hizlar) - SPEED_SHAPE_RATIO_MB) < 4.0 * speed_shape_ratio_noise(n)


def test_MB_sabiti_kendi_turetiminden_geliyor():
    """5/3 LİTERAL olarak kilitlenir -- yoksa sabit hiçbir yerde sınanmıyor.

    ÖLÇÜLDÜ: `SPEED_SHAPE_RATIO_MB = 1.69` yazan mutant bu dosyadaki BEŞ
    testi de geçiyordu, çünkü sabit yalnızca KENDİSİNE karşı assert ediliyor
    (geçiş aralığı ~[1.628, 1.697]). Türetim: s ~ chi_3 için
    E[s^4]/E[s^2]^2 = 15/9 = 5/3.
    """
    assert SPEED_SHAPE_RATIO_MB == pytest.approx(
        1.6666666666666667, rel=1e-15, abs=0.0
    )


def test_gecersiz_serbestlik_ve_ornek_sayisi_tipli_hata_veriyor():
    """İki koruma da SİLİNSE beş test yeşil kalıyordu -- ölçüldü.

    Korumasız zarar: N=1, n_constraints=3 -> ZeroDivisionError;
    n_constraints=4 -> T = -5797629.455 K (NEGATİF sıcaklık, sessiz);
    speed_shape_ratio_noise(0) -> ZeroDivisionError; (-4) -> nan.
    """
    tek = MDState(
        positions=np.zeros((1, 3)),
        velocities=np.ones((1, 3)) * 0.01,
        masses=np.ones(1) * 100.0,
    )
    with pytest.raises(InvalidInputError, match="serbestlik"):
        instantaneous_temperature(tek, n_constraints=3)
    with pytest.raises(InvalidInputError, match="serbestlik"):
        instantaneous_temperature(tek, n_constraints=4)
    for kotu in (0, -4):
        with pytest.raises(InvalidInputError, match="örnek"):
            speed_shape_ratio_noise(kotu)


def test_R4_gurultusu_analitik_forma_uyuyor():
    """std(R4) = 1.217161/sqrt(n) -- toleransın kendisi ölçüme dayanmalı."""
    assert speed_shape_ratio_noise(108) == pytest.approx(0.117121, abs=1e-5, rel=0.0)
    assert speed_shape_ratio_noise(20000) == pytest.approx(0.008607, abs=1e-6, rel=0.0)


def test_R4_maxwell_OLMAYAN_dagilimi_reddediyor():
    """Sınırın diğer tarafı: her dağılım 5/3 vermez."""
    uretici = np.random.default_rng(1)
    n = 20000
    yonler = uretici.normal(size=(n, 3))
    yonler /= np.linalg.norm(yonler, axis=1)[:, None]
    hizlar = yonler * uretici.uniform(0.0, 1.0, size=(n, 1))
    assert abs(speed_shape_ratio(hizlar) - SPEED_SHAPE_RATIO_MB) > 10.0 * speed_shape_ratio_noise(n)
