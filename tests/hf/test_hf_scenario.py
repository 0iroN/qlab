import textwrap
from pathlib import Path

import numpy as np
import pytest

from qlab.engine.hf.factory import run_scan
from qlab.engine.hf.scf import solve
from qlab.errors import BasisNotAvailableError, ScenarioError, SCFConvergenceError
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import HFScanScenario
from qlab.types import Molecule

VALID = textwrap.dedent(
    """
    kind: hf_scan
    name: H2 bag taramasi
    molecule_template: "H 0 0 0; H 0 0 {value}"
    basis: sto-3g
    parameter: bond_length
    values: [0.5, 0.6, 0.7, 0.74, 0.8, 1.0]
    unit: angstrom
    """
)


# Depo koku test dosyasina GORE cozulur: "scenarios/harmonic_trap.yaml" gibi
# cwd'ye bagli bir yol yalnizca pytest depo kokunden kosuldugunda gecerdi
# (tests/hf/ icinden kosuldugunda dosya bulunamaz, test ScenarioError ile
# kirilirdi).
_SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios"


def write(tmp_path, content):
    path = tmp_path / "s.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_hf_scenario(tmp_path):
    scenario = load_scenario(write(tmp_path, VALID))
    assert isinstance(scenario, HFScanScenario)
    assert scenario.basis == "sto-3g"
    assert len(scenario.values) == 6


def test_existing_tdse_scenario_still_loads():
    """Mevcut TDSE senaryolari kirilmamali — kind anahtari opsiyonel."""
    scenario = load_scenario(_SCENARIO_DIR / "harmonic_trap.yaml")
    assert scenario.__class__.__name__ == "TDSEScenario"


def test_rejects_template_without_placeholder(tmp_path):
    """Yer tutucu yoksa tarama ayni molekulu m kez cozerdi — sessiz anlamsizlik."""
    broken = VALID.replace("{value}", "0.74")
    with pytest.raises(ScenarioError, match="value"):
        load_scenario(write(tmp_path, broken))


def test_rejects_unknown_kind(tmp_path):
    broken = VALID.replace("kind: hf_scan", "kind: banana")
    with pytest.raises(ScenarioError, match="banana"):
        load_scenario(write(tmp_path, broken))


def test_rejects_empty_values(tmp_path):
    broken = VALID.replace("values: [0.5, 0.6, 0.7, 0.74, 0.8, 1.0]", "values: []")
    with pytest.raises(ScenarioError, match="values"):
        load_scenario(write(tmp_path, broken))


def test_scan_produces_energy_for_each_value(tmp_path):
    result = run_scan(load_scenario(write(tmp_path, VALID)))
    assert result.energies.shape == (6,)
    assert np.all(np.isfinite(result.energies))


def test_scan_minimum_is_near_equilibrium(tmp_path):
    result = run_scan(load_scenario(write(tmp_path, VALID)))
    assert result.values[int(np.argmin(result.energies))] == pytest.approx(0.74, abs=0.1)


def test_scan_result_carries_provenance(tmp_path):
    result = run_scan(load_scenario(write(tmp_path, VALID)))
    assert result.provenance.converged is True
    assert "Born-Oppenheimer" in " ".join(result.provenance.approximations)


def test_scan_and_single_point_share_one_provenance_source(tmp_path):
    """Tarama ile tek nokta AYNI RHF'i kosar, dolayisiyla ayni beyani basmali.

    Bu blok (dort yaklasim, ihmal listesi, beklenen-hata metni) daha once
    scf.py ve factory.py'de birebir KOPYALANMISTI; biri duzeltilip digeri
    unutuldugunda panel iki farkli sey iddia ederdi. Tek kaynak
    scf.provenance_for'dur ve bu test o esitligi kilitler -- metni
    tekrarlamaz, iki uretici arasinda karsilastirir.
    """
    scan = run_scan(load_scenario(write(tmp_path, VALID)))
    single = solve(Molecule.from_string("H 0 0 0; H 0 0 0.74"), "sto-3g")

    assert scan.provenance.approximations == single.provenance.approximations
    assert scan.provenance.neglected == single.provenance.neglected
    assert scan.provenance.expected_error == single.provenance.expected_error
    # Yontem satiri ve diagnostics bilincli olarak FARKLI olmali (biri tarama,
    # digeri tek nokta) -- aksi halde test yanlislikla her seyi ayni yapardi.
    assert scan.provenance.method != single.provenance.method


# --- Ek testler: brief'in dispatcher/tur-bazli dallarinin her biri, ve
# run_scan'in nokta-basarisizligi tasarim karari, testle kanitlanir. ---


def test_rejects_unknown_top_level_key_in_hf_scenario(tmp_path):
    broken = VALID + "unknown_key: 3\n"
    with pytest.raises(ScenarioError, match="unknown_key"):
        load_scenario(write(tmp_path, broken))


def test_rejects_missing_required_key_in_hf_scenario(tmp_path):
    broken = VALID.replace("basis: sto-3g\n", "")
    with pytest.raises(ScenarioError, match="basis"):
        load_scenario(write(tmp_path, broken))


def test_rejects_molecule_template_not_a_string(tmp_path):
    """molecule_template sayi/liste gibi bir metin-disi deger olursa "{value}
    in template" kontrolu ciplak TypeError firlatirdi -- bu tur yerine temiz
    bir ScenarioError vermeli."""
    broken = VALID.replace(
        'molecule_template: "H 0 0 0; H 0 0 {value}"', "molecule_template: 42"
    )
    with pytest.raises(ScenarioError, match="molecule_template"):
        load_scenario(write(tmp_path, broken))


def test_rejects_values_not_a_list(tmp_path):
    broken = VALID.replace("values: [0.5, 0.6, 0.7, 0.74, 0.8, 1.0]", "values: 5")
    with pytest.raises(ScenarioError, match="values"):
        load_scenario(write(tmp_path, broken))


def test_rejects_non_numeric_value_in_values(tmp_path):
    broken = VALID.replace(
        "values: [0.5, 0.6, 0.7, 0.74, 0.8, 1.0]", "values: [0.5, banana, 0.7]"
    )
    with pytest.raises(ScenarioError, match="values"):
        load_scenario(write(tmp_path, broken))


def test_scan_marks_provenance_unconverged_when_a_point_fails(tmp_path, monkeypatch):
    """Tasarim karari: SCF bir noktada yakinsamazsa tarama DEVAM eder, o
    noktanin enerjisi NaN olur ve provenance.converged sahte bicimde True
    kalmaz. solve() ucuncu cagrida (values[2] = 0.7) yakinsamiyormus gibi
    monkeypatch'lenir; gercek fizige gerek yok, sadece kontrol akisi test
    ediliyor."""
    import qlab.engine.hf.factory as hf_factory

    real_solve = hf_factory.solve
    calls = {"n": 0}

    def flaky_solve(molecule, basis):
        calls["n"] += 1
        if calls["n"] == 3:
            raise SCFConvergenceError(
                iterations=5, energy_history=[-1.0, -1.0], final_residual=1.0
            )
        return real_solve(molecule, basis)

    monkeypatch.setattr(hf_factory, "solve", flaky_solve)

    result = run_scan(load_scenario(write(tmp_path, VALID)))

    assert result.provenance.converged is False
    assert np.isnan(result.energies[2])
    assert np.sum(np.isnan(result.energies)) == 1
    assert result.provenance.diagnostics["n_failed"] == 1
    assert result.provenance.diagnostics["failed_values"] == [pytest.approx(0.7)]


def test_scan_rejects_template_with_extra_placeholder(tmp_path):
    """{value} disinda cozulemeyen bir alan (kopyala-yapistir kalintisi gibi)
    str.format()'da ciplak KeyError firlatirdi -- ScenarioError'a cevrilmeli."""
    broken = VALID.replace(
        'molecule_template: "H 0 0 0; H 0 0 {value}"',
        'molecule_template: "H 0 0 0; H 0 0 {value} {oops}"',
    )
    scenario = load_scenario(write(tmp_path, broken))
    with pytest.raises(ScenarioError, match="molecule_template"):
        run_scan(scenario)


def test_scan_propagates_configuration_error_immediately(tmp_path):
    """Tasarim karari: SCFConvergenceError DISINDAKI hatalar (orn. gecersiz
    baz adi) tek bir noktaya ozgu degildir -- her noktada tekrar olusur, bu
    yuzden yakalanip NaN'a cevrilmez; tarama hemen durur."""
    broken = VALID.replace("basis: sto-3g", "basis: not-a-real-basis")
    scenario = load_scenario(write(tmp_path, broken))
    with pytest.raises(BasisNotAvailableError):
        run_scan(scenario)
