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


def test_scan_marks_lowest_eigenvalue_nan_for_a_failed_point(tmp_path, monkeypatch):
    """Yakinsamayan noktanin kararlilik bilgisi YOKTUR -- 0.0 gibi bir sayi
    basmak "kararli" diye okunurdu; NaN durust olanidir."""
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

    assert np.isnan(result.lowest_eigenvalues[2])
    assert np.sum(np.isnan(result.lowest_eigenvalues)) == 1
    # Yakinsamayan nokta "kararsiz" olarak SAYILMAMALI -- olculmemistir.
    assert result.provenance.diagnostics["n_unstable"] == 0


def test_scan_reports_no_stability_verdict_when_every_point_fails(tmp_path, monkeypatch):
    """Hicbir nokta olculemediyse kararlilik hakkinda hicbir sey iddia
    edilmemeli: verdict None, diagnostics'te n_unstable YOK. "0 kararsiz"
    yazmak, hicbir sey olculmemisken "hepsi kararli" gibi okunurdu."""
    import qlab.engine.hf.factory as hf_factory

    def always_fails(molecule, basis):
        raise SCFConvergenceError(
            iterations=5, energy_history=[-1.0, -1.0], final_residual=1.0
        )

    monkeypatch.setattr(hf_factory, "solve", always_fails)
    result = run_scan(load_scenario(write(tmp_path, VALID)))

    assert result.provenance.stability is None
    assert "n_unstable" not in result.provenance.diagnostics
    assert np.all(np.isnan(result.lowest_eigenvalues))


UNSTABLE_SCAN = textwrap.dedent(
    """
    kind: hf_scan
    name: H2 asiri gerilmis tarama
    molecule_template: "H 0 0 0; H 0 0 {value}"
    basis: sto-3g
    parameter: bond_length
    values: [0.74, 12.0]
    unit: angstrom
    """
)


def test_scan_says_when_a_point_is_an_unreported_saddle_point(tmp_path):
    """Duzeltilen kusur: 12 A'daki nokta yakinsar (n_failed=0) ama bir EYER
    NOKTASIDIR. Tarama bunu soylemek zorundadir; "n_failed=0" tek basina
    egrinin guvenilir oldugu anlamina GELMEZ."""
    result = run_scan(load_scenario(write(tmp_path, UNSTABLE_SCAN)))

    assert result.provenance.diagnostics["n_failed"] == 0
    assert result.provenance.diagnostics["n_unstable"] == 1
    assert result.provenance.diagnostics["unstable_values"] == [pytest.approx(12.0)]
    assert result.provenance.stability.stable is False
    assert result.provenance.stability.lowest_eigenvalue < 0.0
    assert result.lowest_eigenvalues[0] > 0.0
    assert result.lowest_eigenvalues[1] < 0.0


def test_scan_reports_no_verdict_when_points_were_solved_without_analysis(
    tmp_path, monkeypatch
):
    """Nokta COZULDU ama kararlilik ANALIZ EDILMEDI dali.

    run_scan bugun solve()'u varsayilanlariyla cagirir, yani bu durum
    normalde olusmaz; ama HFResult.stability tur olarak None olabilir ve o
    dal testsiz kalirsa, ileride run_scan'e "hizli mod" eklendiginde
    sessizce yanlis rapor uretirdi. Beklenen: yakinsama TRUE (noktalar
    cozuldu), kararlilik hakkinda HICBIR iddia yok.
    """
    import qlab.engine.hf.factory as hf_factory

    real_solve = hf_factory.solve
    monkeypatch.setattr(
        hf_factory, "solve", lambda molecule, basis: real_solve(
            molecule, basis, stability=False
        )
    )
    result = run_scan(load_scenario(write(tmp_path, VALID)))

    assert result.provenance.converged is True
    assert result.provenance.stability is None
    assert "n_unstable" not in result.provenance.diagnostics
    assert np.all(np.isnan(result.lowest_eigenvalues))


HELIUM_SCAN = textwrap.dedent(
    """
    kind: hf_scan
    name: He tek atom konum taramasi
    molecule_template: "He 0 0 {value}"
    basis: sto-3g
    parameter: position
    values: [0.0, 1.0]
    unit: angstrom
    """
)


def test_scan_of_a_system_without_virtual_orbitals_reports_no_eigenvalue(tmp_path):
    """He/STO-3G'de bos orbital yoktur -> hicbir noktada ozdeger olculemez.

    Hukum "kararli"dir (dondurulecek yon yok) ama raporlanan ozdeger None
    olmalidir; 0.0 yazmak "sinirda kararli" diye okunurdu. Bu, _scan_verdict
    icindeki "olculmus ozdeger listesi bos" dalidir.
    """
    result = run_scan(load_scenario(write(tmp_path, HELIUM_SCAN)))

    assert np.all(np.isfinite(result.energies))
    assert np.all(np.isnan(result.lowest_eigenvalues))
    assert result.provenance.stability.stable is True
    assert result.provenance.stability.lowest_eigenvalue is None
    assert result.provenance.stability.n_rotations == 0
    assert result.provenance.diagnostics["n_unstable"] == 0


def test_scan_with_only_stable_points_reports_a_stable_verdict(tmp_path):
    result = run_scan(load_scenario(write(tmp_path, VALID)))

    assert result.provenance.diagnostics["n_unstable"] == 0
    assert "unstable_values" not in result.provenance.diagnostics
    assert result.provenance.stability.stable is True
    assert np.all(result.lowest_eigenvalues > 0.0)


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
