import io
import json
import sys
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


def test_main_outputs_turkish_text_through_cp1252_stream(scenario_file, tmp_path):
    """Ana fonksiyon CP1252 kodlaması (Türkçe 'ı' karakteri yapamaz) tarafından
    yapılamayan akış üzerinden Türkçe metni çıkış yapamalıdır."""
    output = tmp_path / "cli.npz"

    # CP1252 akışı oluştur; 'ı' karakteri bu kodlamada yapılamaz.
    # Başarılı bir çalıştırma, main() içindeki UTF-8 yeniden yapılandırmasını kanıtlar.
    cp1252_buffer = io.BytesIO()
    cp1252_stream = io.TextIOWrapper(cp1252_buffer, encoding="cp1252", errors="strict")

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = cp1252_stream
        sys.stderr = io.TextIOWrapper(
            io.BytesIO(), encoding="cp1252", errors="strict"
        )

        # main() Türkçe çıktı yapmak istediğinde, fix olmadan UnicodeEncodeError
        # ile başarısız olmalı (scenario adında ve provenance'da 'ı' vardır).
        # Fix ile, main() stdout'u UTF-8'e yapılandırır ve başarılı olur.
        result = run.main([str(scenario_file), "-o", str(output)])
        assert result == 0
        assert output.is_file()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
