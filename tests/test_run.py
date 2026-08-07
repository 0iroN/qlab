import json
import textwrap

import numpy as np
import pytest

from qlab import run

SMALL_SCENARIO = textwrap.dedent(
    """
    name: Küçük serbest paket
    grid:
      n: 256
      length_bohr: 100.0
    potential:
      kind: free
    packet:
      x0_bohr: -10.0
      k0: 0.5
      sigma_bohr: 5.0
    dt: 0.05
    n_steps: 40
    save_every: 10
    """
)


@pytest.fixture
def scenario_file(tmp_path):
    path = tmp_path / "small.yaml"
    path.write_text(SMALL_SCENARIO, encoding="utf-8")
    return path


def test_run_scenario_returns_frames_and_provenance(scenario_file):
    result = run.run_scenario(scenario_file)
    assert result.frames.shape == (5, 256)  # t=0 dahil
    assert result.times[-1] == pytest.approx(2.0)
    assert result.provenance.converged is True
    assert "periyodik" in " ".join(result.provenance.approximations).lower()


def test_run_scenario_preserves_norm(scenario_file):
    result = run.run_scenario(scenario_file)
    final_norm = np.sum(np.abs(result.frames[-1]) ** 2) * result.grid.dx
    assert final_norm == pytest.approx(1.0, abs=1e-10)


def test_save_writes_readable_npz(scenario_file, tmp_path):
    result = run.run_scenario(scenario_file)
    output = tmp_path / "out.npz"
    run.save(result, output)

    with np.load(output, allow_pickle=False) as data:
        assert data["frames"].shape == (5, 256)
        assert data["x"].shape == (256,)
        assert data["potential"].shape == (256,)
        assert data["times"].shape == (5,)
        provenance = json.loads(str(data["provenance_json"]))

    assert provenance["method"].startswith("TDSE split-operator")
    assert provenance["converged"] is True


def test_main_returns_zero_and_writes_output(scenario_file, tmp_path):
    output = tmp_path / "cli.npz"
    assert run.main([str(scenario_file), "-o", str(output)]) == 0
    assert output.is_file()


def test_main_reports_missing_scenario(capsys):
    assert run.main(["yok.yaml", "-o", "x.npz"]) == 1
    assert "bulunamadı" in capsys.readouterr().err
