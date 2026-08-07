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


def test_unmasked_convergence_shows_norm_deviation():
    """Maskesiz (üniterlik kontrolü yapılan) koşular norm_deviation etiketi
    göstermelidir, bu kontrol kanıtıdır."""
    prov = make_provenance(
        converged=False,
        diagnostics={"norm_deviation": 1.5e-2, "steps": 100},
    )
    yakinsama_line = dict(prov.as_lines())["Yakınsama"]
    assert "norm_deviation=" in yakinsama_line
    assert "observed_norm_drop=" not in yakinsama_line


def test_masked_convergence_shows_observed_norm_drop():
    """Maskeli (üniterlik kontrolü yapılmayan) koşular observed_norm_drop
    etiketi göstermelidir, bu bilgilendirici ölçüdür ve kontrol kanıtı değildir."""
    prov = make_provenance(
        converged=True,
        diagnostics={"observed_norm_drop": 7.0e-2, "steps": 100},
    )
    yakinsama_line = dict(prov.as_lines())["Yakınsama"]
    assert "observed_norm_drop=" in yakinsama_line
    assert "norm_deviation=" not in yakinsama_line


def test_masked_case_cannot_be_misread_as_convergence_evidence():
    """Maskeli koşuda kontrol yapılmadığı için, panel okuyucu norm değerini
    yakınsama kanıtı olarak yanlış okumamalı. observed_norm_drop etiketinin
    varlığı ve norm_deviation'ın YOKLUĞU bunu garanti eder."""
    prov = make_provenance(
        converged=True,  # Maske varken always True (kontrol yapılmadığı için)
        diagnostics={"observed_norm_drop": 7.0e-2, "steps": 100},
    )
    yakinsama_line = dict(prov.as_lines())["Yakınsama"]
    # Başlangıç ✓ kontrol geçti anlamı değildir maskede, ama norm_deviation
    # görmemek kontrol yapılmadığını kanıtlar.
    assert yakinsama_line.startswith("✓")
    assert "norm_deviation" not in yakinsama_line
