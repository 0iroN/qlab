import numpy as np
import pytest
from hypothesis import given, strategies as st

from qlab import units


def test_bohr_to_angstrom_known_value():
    assert units.bohr_to_angstrom(1.0) == pytest.approx(0.529177210903, abs=1e-12)


def test_hartree_to_ev_known_value():
    assert units.hartree_to_ev(1.0) == pytest.approx(27.211386245988, abs=1e-9)


def test_hydrogen_ground_state_in_ev():
    """H atomu taban durumu -0.5 hartree = -13.6057 eV."""
    assert units.hartree_to_ev(-0.5) == pytest.approx(-13.6057, abs=1e-3)


def test_conversion_works_on_arrays():
    result = units.bohr_to_angstrom(np.array([1.0, 2.0, 3.0]))
    assert isinstance(result, np.ndarray)
    assert result[1] == pytest.approx(2 * 0.529177210903, abs=1e-12)


@given(st.floats(min_value=1e-6, max_value=1e6, allow_nan=False))
def test_length_roundtrip(value):
    assert units.angstrom_to_bohr(units.bohr_to_angstrom(value)) == pytest.approx(value, rel=1e-12)


@given(st.floats(min_value=1e-6, max_value=1e6, allow_nan=False))
def test_energy_roundtrip(value):
    assert units.ev_to_hartree(units.hartree_to_ev(value)) == pytest.approx(value, rel=1e-12)
