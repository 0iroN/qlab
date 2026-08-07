import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf, factorial2

from qlab.engine.hf.boys import boys


def boys_by_quadrature(n: int, T: float) -> float:
    """Tanimin dogrudan sayisal integrali -- bagimsiz referans."""
    value, _ = quad(lambda t: t ** (2 * n) * np.exp(-T * t**2), 0.0, 1.0, epsabs=1e-13)
    return value


def double_factorial_odd(n: int) -> float:
    """(2n-1)!! -- n=0 icin (-1)!! = 1 kuraldir.

    scipy.special.factorial2(-1) 0.0 doner (scipy >= 1.11 davranisi, 1.18'de
    olculdu), bu yuzden n=0 durumu elle verilmelidir; dogrudan factorial2
    kullanmak asimptot testini n=0'da sessizce yanlis yapar.
    """
    return 1.0 if n == 0 else float(factorial2(2 * n - 1))


def test_f0_at_zero_is_one():
    assert boys(0, 0.0) == pytest.approx(1.0, abs=1e-14)


def test_fn_at_zero_is_one_over_2n_plus_1():
    for n in range(6):
        assert boys(n, 0.0) == pytest.approx(1.0 / (2 * n + 1), rel=1e-12)


def test_f0_closed_form():
    """F_0(T) = sqrt(pi/(4T)) * erf(sqrt(T))."""
    for T in (0.1, 1.0, 5.0, 30.0):
        expected = np.sqrt(np.pi / (4 * T)) * erf(np.sqrt(T))
        assert boys(0, T) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("T", [1e-8, 0.05, 0.5, 2.0, 12.0, 40.0, 120.0])
def test_matches_numerical_quadrature(n, T):
    assert boys(n, T) == pytest.approx(boys_by_quadrature(n, T), rel=1e-10)


def test_downward_recursion_consistency():
    """F_n(T) = (2T F_{n+1}(T) + exp(-T)) / (2n+1)."""
    for T in (0.3, 1.7, 9.0, 55.0):
        for n in range(5):
            rhs = (2 * T * boys(n + 1, T) + np.exp(-T)) / (2 * n + 1)
            assert boys(n, T) == pytest.approx(rhs, rel=1e-11)


def test_monotonically_decreasing_in_T():
    values = [boys(2, T) for T in (0.0, 1.0, 5.0, 20.0, 100.0)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_accepts_array_input():
    result = boys(1, np.array([0.0, 1.0, 10.0]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert result[0] == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_large_T_asymptote():
    """Buyuk T'de F_n(T) -> (2n-1)!! / 2^{n+1} * sqrt(pi / T^{2n+1})."""
    T = 500.0
    for n in (0, 1, 3):
        expected = double_factorial_odd(n) / 2 ** (n + 1) * np.sqrt(np.pi / T ** (2 * n + 1))
        assert boys(n, T) == pytest.approx(expected, rel=1e-6)
