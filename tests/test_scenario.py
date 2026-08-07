import textwrap

import numpy as np
import pytest

from qlab.engine.tdse import factory
from qlab.errors import ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import TDSEScenario

VALID_YAML = textwrap.dedent(
    """
    name: Test tünelleme
    grid:
      n: 512
      length_bohr: 200.0
    potential:
      kind: barrier
      params:
        height: 0.8
        width: 2.0
        center: 0.0
    packet:
      x0_bohr: -40.0
      k0: 1.0
      sigma_bohr: 8.0
    dt: 0.02
    n_steps: 100
    save_every: 10
    """
)


def write(tmp_path, content, name="s.yaml"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_valid_scenario(tmp_path):
    scenario = load_scenario(write(tmp_path, VALID_YAML))
    assert isinstance(scenario, TDSEScenario)
    assert scenario.name == "Test tünelleme"
    assert scenario.grid.n == 512
    assert scenario.potential.kind == "barrier"
    assert scenario.potential.params["height"] == pytest.approx(0.8)
    assert scenario.packet.k0 == pytest.approx(1.0)


def test_rejects_unknown_top_level_key(tmp_path):
    with pytest.raises(ScenarioError, match="unknown_key"):
        load_scenario(write(tmp_path, VALID_YAML + "\nunknown_key: 3\n"))


def test_rejects_unknown_potential_kind(tmp_path):
    broken = VALID_YAML.replace("kind: barrier", "kind: banana")
    with pytest.raises(ScenarioError, match="banana"):
        load_scenario(write(tmp_path, broken))


def test_rejects_missing_required_section(tmp_path):
    broken = VALID_YAML.replace("dt: 0.02\n", "")
    with pytest.raises(ScenarioError, match="dt"):
        load_scenario(write(tmp_path, broken))


def test_reports_missing_file():
    with pytest.raises(ScenarioError, match="bulunamadı"):
        load_scenario("olmayan_dosya.yaml")


def test_factory_builds_consistent_objects(tmp_path):
    scenario = load_scenario(write(tmp_path, VALID_YAML))
    grid, potential, psi0, prop = factory.build(scenario)

    assert grid.n == 512
    assert potential.shape == (512,)
    assert psi0.shape == (512,)
    assert psi0.dtype == np.complex128
    assert np.sum(np.abs(psi0) ** 2) * grid.dx == pytest.approx(1.0, abs=1e-12)
    assert prop.dt == pytest.approx(0.02)
    assert prop.mask is None


def test_factory_attaches_mask_when_requested(tmp_path):
    scenario = load_scenario(write(tmp_path, VALID_YAML + "absorbing_width_bohr: 20.0\n"))
    _, _, _, prop = factory.build(scenario)
    assert prop.mask is not None


def test_provenance_declares_periodic_boundary(tmp_path):
    scenario = load_scenario(write(tmp_path, VALID_YAML))
    prov = factory.provenance_for(scenario, norm_deviation=1e-13)
    assert "periyodik" in " ".join(prov.approximations + prov.neglected).lower()
    assert prov.converged is True
