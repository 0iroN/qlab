import dataclasses
import pytest

from qlab.errors import InvalidInputError, QLabError, UnitarityViolationError
from qlab.provenance import Provenance


def make_provenance(**overrides) -> Provenance:
    defaults = dict(
        method="TDSE split-operator (Strang), 1B",
        approximations=("tek parçacık", "periyodik sınır koşulu"),
        neglected=("elektron-elektron etkileşimi", "spin"),
        expected_error="O(dt^2) ~ 4.0e-04 hartree",
        converged=True,
        diagnostics={"norm_deviation": 3.2e-13, "steps": 8000},
    )
    defaults.update(overrides)
    return Provenance(**defaults)


def test_provenance_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_provenance().method = "başka bir şey"


def test_as_lines_returns_expected_labels():
    labels = [label for label, _ in make_provenance().as_lines()]
    assert labels == ["Yöntem", "Yaklaşımlar", "İhmal edilen", "Beklenen hata", "Yakınsama"]


def test_as_lines_joins_lists_with_separator():
    assert dict(make_provenance().as_lines())["Yaklaşımlar"] == "tek parçacık · periyodik sınır koşulu"


def test_as_lines_marks_converged_state():
    assert dict(make_provenance(converged=True).as_lines())["Yakınsama"].startswith("✓")
    assert dict(make_provenance(converged=False).as_lines())["Yakınsama"].startswith("✗")


def test_unitarity_error_carries_diagnosis():
    err = UnitarityViolationError(step=412, norm_deviation=2.5e-6, dt=0.5)
    assert err.step == 412
    assert err.norm_deviation == pytest.approx(2.5e-6)
    assert err.dt == pytest.approx(0.5)
    assert "412" in str(err)
    assert "0.5" in str(err)


def test_error_hierarchy():
    assert issubclass(UnitarityViolationError, QLabError)
    assert issubclass(InvalidInputError, QLabError)
