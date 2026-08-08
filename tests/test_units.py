import numpy as np
import pytest
from hypothesis import given, strategies as st

from qlab import units


def test_bohr_to_angstrom_known_value():
    # rel=0.0: fiilen uygulanan, varsayılan rel=1e-6'nın (=5.3e-7) yazılan
    # abs=1e-12'yi ezdiği haldi. Ölçülen sapma tam 0.0 (bit-bit eşit; 1.0
    # çarpımı sabiti değiştirmiyor) olduğundan yazarın abs=1e-12'si korunuyor.
    assert units.bohr_to_angstrom(1.0) == pytest.approx(0.529177210903, abs=1e-12, rel=0.0)


def test_hartree_to_ev_known_value():
    # rel=0.0: fiilen uygulanan, varsayılan rel=1e-6'nın (=2.7e-5) yazılan
    # abs=1e-9'u ezdiği haldi. Ölçülen sapma tam 0.0 (bit-bit eşit) olduğundan
    # yazarın abs=1e-9'u korunuyor.
    assert units.hartree_to_ev(1.0) == pytest.approx(27.211386245988, abs=1e-9, rel=0.0)


def test_hydrogen_ground_state_in_ev():
    """H atomu taban durumu -0.5 hartree = -13.6057 eV."""
    assert units.hartree_to_ev(-0.5) == pytest.approx(-13.6057, abs=1e-3)


def test_conversion_works_on_arrays():
    result = units.bohr_to_angstrom(np.array([1.0, 2.0, 3.0]))
    assert isinstance(result, np.ndarray)
    # rel=0.0: fiilen uygulanan, varsayılan rel=1e-6'nın (~1.06e-6) yazılan
    # abs=1e-12'yi ezdiği haldi. Ölçülen sapma tam 0.0 (bit-bit eşit)
    # olduğundan yazarın abs=1e-12'si korunuyor.
    assert result[1] == pytest.approx(2 * 0.529177210903, abs=1e-12, rel=0.0)


@given(st.floats(min_value=1e-6, max_value=1e6, allow_nan=False))
def test_length_roundtrip(value):
    # abs=0.0: yazılan rel=1e-12, varsayılan abs=1e-12 ile birleşince küçük
    # uçta (value ~1e-6) fiilen ~1e-6 göreli sapmaya izin veriyordu. Dönüşüm
    # sadece iki float çarpımı (C ve 1/C) olduğundan göreli sapma büyüklükten
    # bağımsız: [1e-6, 1e6] aralığında 40000+ değerlik taramada (uçlar, 2'nin
    # kuvvetleri, rastgele) en kötü ölçülen ~2.22e-16. Yazarın 1e-12'si bu
    # ölçüme göre ~4500x gevşek olurdu (gerçekten bağlayıcı olmazdı), bu
    # yüzden rel=1e-15'e sıkıştırıldı: ~4.5x pay, "genuinely binding".
    assert units.angstrom_to_bohr(units.bohr_to_angstrom(value)) == pytest.approx(
        value, rel=1e-15, abs=0.0
    )


@given(st.floats(min_value=1e-6, max_value=1e6, allow_nan=False))
def test_energy_roundtrip(value):
    # abs=0.0: test_length_roundtrip ile aynı gerekçe. En kötü ölçülen göreli
    # sapma ~1.89e-16; rel=1e-15 ~5.3x payla bağlayıcı.
    assert units.ev_to_hartree(units.hartree_to_ev(value)) == pytest.approx(
        value, rel=1e-15, abs=0.0
    )
