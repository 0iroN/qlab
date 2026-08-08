import dataclasses
import textwrap
from pathlib import Path

import numpy as np
import pytest

from qlab.engine.hf.factory import run_scan
from qlab.engine.tdse import factory
from qlab.errors import ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import HFScanScenario, TDSEScenario

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


def test_loads_scenario_with_explicit_kind_tdse(tmp_path):
    """B1 Task 12: 'kind' artık isteğe bağlı bir üst-seviye anahtar. Mevcut
    dosyalar bu anahtarı hiç taşımadığı için (yukarıdaki test) o yol zaten
    kapsanıyor, ama 'kind: tdse' AÇIKÇA verilirse de aynı şekilde yüklenmeli
    -- 'kind' _TOP_LEVEL kümesine eklenmezse bu, sahte bir 'bilinmeyen
    anahtar' ScenarioError'ıyla kırılırdı."""
    explicit = "kind: tdse\n" + VALID_YAML
    scenario = load_scenario(write(tmp_path, explicit))
    assert isinstance(scenario, TDSEScenario)
    assert scenario.name == "Test tünelleme"


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


def test_rejects_potential_as_string(tmp_path):
    """potential: <string> yerine potential: {dict} gerekirse ScenarioError verilir."""
    broken = VALID_YAML.replace(
        "potential:\n  kind: barrier\n  params:\n    height: 0.8\n    width: 2.0\n    center: 0.0",
        "potential: banana"
    )
    with pytest.raises(ScenarioError, match="potential.*sözlük"):
        load_scenario(write(tmp_path, broken))


def test_rejects_potential_as_list(tmp_path):
    """potential: [...] yerine potential: {dict} gerekirse ScenarioError verilir."""
    broken = VALID_YAML.replace(
        "potential:\n  kind: barrier\n  params:\n    height: 0.8\n    width: 2.0\n    center: 0.0",
        "potential: [1, 2, 3]"
    )
    with pytest.raises(ScenarioError, match="potential.*sözlük"):
        load_scenario(write(tmp_path, broken))


def test_rejects_non_numeric_dt(tmp_path):
    """dt: <non-numeric> ScenarioError verip 'dt' adını belirtir."""
    broken = VALID_YAML.replace("dt: 0.02", "dt: not_a_number")
    with pytest.raises(ScenarioError, match="dt.*dönüştürülemedi"):
        load_scenario(write(tmp_path, broken))


def test_rejects_non_numeric_n_steps(tmp_path):
    """n_steps: <non-numeric> ScenarioError verip 'n_steps' adını belirtir."""
    broken = VALID_YAML.replace("n_steps: 100", "n_steps: not_a_number")
    with pytest.raises(ScenarioError, match="n_steps.*dönüştürülemedi"):
        load_scenario(write(tmp_path, broken))


def test_rejects_non_numeric_mass(tmp_path):
    """mass: <non-numeric> ScenarioError verip 'mass' adını belirtir."""
    broken = VALID_YAML + "mass: not_a_number\n"
    with pytest.raises(ScenarioError, match="mass.*dönüştürülemedi"):
        load_scenario(write(tmp_path, broken))


def test_rejects_non_numeric_save_every(tmp_path):
    """save_every: <non-numeric> ScenarioError verip 'save_every' adını belirtir."""
    broken = VALID_YAML.replace("save_every: 10", "save_every: not_a_number")
    with pytest.raises(ScenarioError, match="save_every.*dönüştürülemedi"):
        load_scenario(write(tmp_path, broken))


def test_provenance_declares_periodic_boundary(tmp_path):
    scenario = load_scenario(write(tmp_path, VALID_YAML))
    prov = factory.provenance_for(scenario, norm_deviation=1e-13)
    assert "periyodik" in " ".join(prov.approximations + prov.neglected).lower()
    assert prov.converged is True


def test_provenance_masked_run_not_marked_unconverged_by_norm_drop(tmp_path):
    """Soğurucu maskeli bir senaryoda norm bilinçli olarak azalır (evolve()
    da bu yüzden maske varken norm kontrolünü atlar, bkz. propagator.py
    check_norm). Büyük bir norm_deviation TEK BAŞINA converged=False'a yol
    açmamalı — aksi halde panel doğru çalışan bir koşuyu '✗ Yakınsama' diye
    gösterir."""
    scenario = load_scenario(write(tmp_path, VALID_YAML + "absorbing_width_bohr: 20.0\n"))
    prov = factory.provenance_for(scenario, norm_deviation=0.2)
    assert prov.converged is True


def test_provenance_unmasked_run_with_large_norm_deviation_reports_unconverged(tmp_path):
    """Maskesiz bir koşuda norm sapması hâlâ tek üniterlik kanıtıdır; eşiği
    (1e-8) aşan büyük bir sapma converged=False raporlamalı. Bu test, maskeli
    koşu için yukarıdaki düzeltmenin kör bir 'her zaman yakınsadı'ya
    dönüşmediğini kanıtlar."""
    scenario = load_scenario(write(tmp_path, VALID_YAML))
    prov = factory.provenance_for(scenario, norm_deviation=0.2)
    assert prov.converged is False


def test_rejects_params_as_string(tmp_path):
    """params: banana yerine params: {dict} gerekirse ScenarioError verilir."""
    broken = VALID_YAML.replace(
        "potential:\n  kind: barrier\n  params:\n    height: 0.8\n    width: 2.0\n    center: 0.0",
        "potential:\n  kind: barrier\n  params: banana"
    )
    with pytest.raises(ScenarioError, match="params|dictionary"):
        load_scenario(write(tmp_path, broken))


def test_rejects_params_as_list(tmp_path):
    """params: [1, 2, 3] yerine params: {dict} gerekirse ScenarioError verilir."""
    broken = VALID_YAML.replace(
        "potential:\n  kind: barrier\n  params:\n    height: 0.8\n    width: 2.0\n    center: 0.0",
        "potential:\n  kind: barrier\n  params:\n    - 1\n    - 2\n    - 3"
    )
    with pytest.raises(ScenarioError, match="params|dictionary"):
        load_scenario(write(tmp_path, broken))


def test_rejects_non_utf8_scenario_file(tmp_path):
    """UTF-8 olmayan dosya ScenarioError verip UTF-8 hatasını belirtir."""
    path = tmp_path / "non_utf8.yaml"
    # İçinde geçersiz UTF-8 bayt dizisi olan dosya yaz
    path.write_bytes(b"name: test\ngrid:\n  n: 512\n  length_bohr: 200\npotential:\n  kind: barrier\n  params:\n    height: 0.8\n\xff\xfe")
    with pytest.raises(ScenarioError, match="UTF-8"):
        load_scenario(path)


def test_factory_rejects_absorbing_width_as_string(tmp_path):
    """absorbing_width_bohr: <string> factory.build() içinde ScenarioError verip
    'absorbing_width_bohr' adını belirtir."""
    scenario = load_scenario(write(tmp_path, VALID_YAML + "absorbing_width_bohr: cok\n"))
    with pytest.raises(ScenarioError, match="absorbing_width_bohr"):
        factory.build(scenario)


def test_factory_rejects_absorbing_width_as_list(tmp_path):
    """absorbing_width_bohr: [...] factory.build() içinde ScenarioError verip
    hatayı belirtir."""
    scenario = load_scenario(write(tmp_path, VALID_YAML + "absorbing_width_bohr: [1, 2, 3]\n"))
    with pytest.raises(ScenarioError, match="absorbing_width_bohr"):
        factory.build(scenario)


# ---------------------------------------------------------------------------
# Depoyla BİRLİKTE DAĞITILAN senaryo dosyaları
#
# Bu dosyalara hiçbir test dokunmuyordu: `scenarios/water_angle_scan.yaml`
# tam da bu boşluktan, bağ uzunluğu z koordinatına yazılmış hâlde yayınlandı
# ve elle ölçümle yakalandı, süiti kırmadan. Aşağıdaki testler dizini
# KEŞFEDEREK çalışır — yeni bir senaryo eklendiğinde kapsam kendiliğinden
# genişler, listeyi güncellemeyi unutmak diye bir şey olmaz.
#
# Yollar test dosyasına GÖRE çözülür, cwd'ye göre değil: cwd-bağımlı bir yol
# yalnızca pytest depo kökünden koşulduğunda geçerdi.
# ---------------------------------------------------------------------------

_SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"
_SHIPPED_SCENARIOS = sorted(_SCENARIO_DIR.glob("*.yaml"))
_HF_SCAN_SAMPLE_POINTS = 2


def test_scenario_directory_is_not_empty():
    """Keşfe dayalı parametrizasyonun sessizce boşa düşmemesi için güvenlik ağı:
    glob hiçbir şey bulmasaydı aşağıdaki test 0 kez koşar ve YEŞİL görünürdü."""
    assert len(_SHIPPED_SCENARIOS) >= 2, f"{_SCENARIO_DIR} altında senaryo bulunamadı"


@pytest.mark.parametrize(
    "scenario_path", _SHIPPED_SCENARIOS, ids=lambda p: p.name
)
def test_shipped_scenario_loads_and_runs(scenario_path):
    """Her dağıtılan senaryo yüklenebilmeli VE koşturulabilir olmalı.

    "Ayrıştırılabilir" yetmez: TDSE tarafında motor nesneleri gerçekten
    kurulur (`factory.build`, potansiyel parametreleri dahil), HF tarafında
    taramanın ilk birkaç noktası gerçekten çözülür. Tam ızgara koşulmaz —
    `h2_bond_scan.yaml` 19 nokta taşır ve burada aranan şey fizik değil,
    dosyanın çalışır olmasıdır.
    """
    scenario = load_scenario(scenario_path)

    if isinstance(scenario, HFScanScenario):
        kisa = dataclasses.replace(
            scenario, values=scenario.values[:_HF_SCAN_SAMPLE_POINTS]
        )
        result = run_scan(kisa)
        assert result.energies.shape == (len(kisa.values),)
        assert np.all(np.isfinite(result.energies)), (
            f"{scenario_path.name}: ilk {_HF_SCAN_SAMPLE_POINTS} noktada SCF "
            "yakınsamadı (enerji NaN)"
        )
        assert result.provenance.converged is True
        return

    assert isinstance(scenario, TDSEScenario)
    grid, potential, psi0, _ = factory.build(scenario)
    assert potential.shape == (scenario.grid.n,)
    assert np.all(np.isfinite(potential))
    norm = float(np.sum(np.abs(psi0) ** 2) * grid.dx)
    assert norm == pytest.approx(1.0, abs=1e-10, rel=0.0)
