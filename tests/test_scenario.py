import dataclasses
import math
import textwrap
from pathlib import Path

import numpy as np
import pytest

from qlab.engine.hf.factory import run_scan
from qlab.engine.md import factory as md_factory
from qlab.engine.tdse import factory
from qlab.errors import ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import (
    BridgeScenario,
    POTENTIAL_KINDS,
    DownfoldScenario,
    EMScenario,
    GravScenario,
    HFScanScenario,
    MDScenario,
    SpinScenario,
    TBScenario,
    TDSEScenario,
)

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


def test_POTENTIAL_KINDS_FACTORY_SEVKINDEN_TURETILIYOR(sevk_tarayici):
    """`POTENTIAL_KINDS`in bayatlamasını yasaklar.

    Yukarıdaki `test_rejects_unknown_potential_kind` KAPIYI sınıyor; bu
    test kapının LİSTESİNİ sınıyor, ve ikisi ayrı şeylerdir. Kapı, demet
    yanlış olsa bile yeşil kalır: `banana` her hâlükârda reddedilir. Bir
    ALTINCI potansiyel `tdse/factory.py`ye eklenip demete eklenmezse kapı
    yine yeşildir ama o potansiyel YAML'dan HİÇ yüklenemez -- sessiz ve
    yanlış. Bu testin yakaladığı şey tam olarak odur.

    `MD_FORCE_FIELD_KINDS`/`MD_THERMOSTAT_KINDS` kilitleriyle AYNI
    sözleşme ve AYNI tarayıcı (kök `conftest.py`).

    ÖLÇÜLDÜ (30.08.2026): beş sevk dalı -- free, barrier, well, harmonic,
    uniform_field; ve demet BAYAT DEĞİLDİ, yani burada onarılan mevcut bir
    kusur yok, kapatılan şey aynı tuzağın üçüncü kopyası.
    """
    sevk = sevk_tarayici(factory, "_build_potential", "kind")
    assert sevk == set(POTENTIAL_KINDS), (
        f"factory {sorted(sevk)} sevk ediyor, şema {sorted(POTENTIAL_KINDS)} "
        f"diyor -- demet bayatladı"
    )
    # Sayı LİTERAL: tarayıcı sessizce boşalırsa (ör. sevk `kind` yerine
    # başka bir değişkene taşınırsa) boş küme boş demete karşı yeşil kalırdı.
    assert len(sevk) == 5


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
    # rel=0.0: varsayılan rel=1e-6 yazılan abs=1e-12'yi ezmesin. Ölçülen sapma
    # 2.22e-16; abs=1e-15 ~4.5x payla bağlayıcı.
    assert np.sum(np.abs(psi0) ** 2) * grid.dx == pytest.approx(1.0, abs=1e-15, rel=0.0)
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
#: MD senaryolarinda TAM kosu YAPILMAZ: argon_nvt.yaml 20000 adimdir ve bu
#: test dosyanin CALISIRLIGINI sinar, fizigi degil. `_HF_SCAN_SAMPLE_POINTS`
#: ile ayni gerekce.
_MD_SAMPLE_STEPS = 5
_TB_SAMPLE_K_POINTS = 10
#: İndirgeme senaryosunda TAM koşu YAPILMAZ: `halka_indirgeme.yaml` beş
#: mesafe × 12 tohum UHF araması taşır (ölçüldü, 5.5 s). Burada aranan şey
#: fizik değil dosyanın çalışır olmasıdır; `_HF_SCAN_SAMPLE_POINTS` ile aynı
#: gerekçe. Tam koşum `tests/tb/test_downfold_senaryo.py`de yapılır ve orada
#: `@pytest.mark.slow` ile işaretlidir.
_DOWNFOLD_SAMPLE_DISTANCES = 1
_DOWNFOLD_SAMPLE_SEEDS = 4
#: Spin senaryosunda TAM koşu YAPILMAZ. `_HF_SCAN_SAMPLE_POINTS` ile aynı
#: gerekçe: burada aranan şey fizik değil dosyanın çalışır olmasıdır. Tam
#: koşum `tests/spin/test_spin_senaryo.py::test_CLI_UCTAN_UCA`dadır.
#: `n_sweep` 32'nin ALTINA İNDİRİLEMEZ ve bu bir ölçüm sonucudur:
#: `binning_error` plato ölçütü için en az `min_blocks = 32` örnek ister ve
#: daha azında `InvalidInputError` fırlatır — yani "hızlandıralım" diye 16
#: yazmak bu testi kırardı, sessizce zayıflatmazdı.
_SPIN_SAMPLE_TEMPERATURES = 1
_SPIN_SAMPLE_SWEEPS = 64
_SPIN_SAMPLE_BURN = 10
#: EM senaryosunda TAM koşu YAPILMAZ. `_HF_SCAN_SAMPLE_POINTS` ile aynı
#: gerekçe; tam koşum `tests/em/test_em_senaryo.py::test_CLI_UCTAN_UCA`dadır
#: (ölçüldü, 1.49 s). Burada merdiven DERİNLİĞİ de kısaltılır ve bunun bir
#: yan etkisi vardır: iki basamaklı bir merdiven `p_inf = None` döndürür,
#: yani panelin hükmü YARGILANAMAZ olur. **Bu bir kusur değil, testin ikinci
#: işi:** sevk edilen senaryo ✓ dalını gösteriyor, burası ✗ dalını gösteriyor
#: ve iki dal ancak birlikte kilitlenir.
_EM_SAMPLE_RADII = 2
_EM_SAMPLE_STEP_FACTOR = 120.0

#: Yerçekimi senaryosunda yörünge örneklemesi KISALTILIR (2000 → 64) ama
#: Richardson çiftleri ve N-cisim toleransları KISALTILMAZ, ve bu bilinçli:
#: `trajectory_points` bir ÖRNEKLEME yoğunluğudur (doğruluk düğmesi değil),
#: oysa Richardson çifti ile `rtol` doğrudan hükme girer — kısaltılsalardı bu
#: test panelin ✓ dalını başka bir pencerede sınıyor olurdu. Tam koşum
#: `tests/grav/test_grav_senaryo.py::test_CLI_UCTAN_UCA`dadır (ölçüldü,
#: ≈ 0.7 s + yorumlayıcı başlangıcı).
_GRAV_SAMPLE_POINTS = 64


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

    if isinstance(scenario, MDScenario):
        kisa = dataclasses.replace(scenario, n_steps=_MD_SAMPLE_STEPS, save_every=1)
        durum, alan, termostat = md_factory.build(kisa)
        enerji, kuvvetler = alan(durum.positions)
        assert kuvvetler.shape == durum.positions.shape
        assert np.all(np.isfinite(kuvvetler)), (
            f"{scenario_path.name}: ilk kuvvet hesabı sonlu değil"
        )
        assert np.isfinite(enerji)
        return

    if isinstance(scenario, TBScenario):
        # TB tarafinda "kosturulabilir" demek: model kurulabiliyor, bant
        # yapisi sonlu cikiyor ve iletkenlik hukmu verilebiliyor. Izgara
        # KUCULTULUR (senaryonun kendi k_points'i degil), cunku burada
        # aranan sey fizik degil dosyanin calisir olmasidir.
        from qlab.engine.tb.bands import band_structure, uniform_k_grid
        from qlab.engine.tb.factory import build_tb
        from qlab.observable.tb_obs import is_metal

        model, provenance = build_tb(
            dataclasses.replace(scenario, k_points=_TB_SAMPLE_K_POINTS)
        )
        enerjiler = band_structure(
            model, uniform_k_grid(model.lattice, _TB_SAMPLE_K_POINTS)
        )
        assert enerjiler.shape[1] == model.n_orbital
        assert np.all(np.isfinite(enerjiler)), (
            f"{scenario_path.name}: bant enerjileri sonlu degil"
        )
        assert isinstance(
            is_metal(enerjiler, scenario.n_electron_per_cell), bool
        )
        assert provenance.method.startswith("tight-binding")
        return

    if isinstance(scenario, DownfoldScenario):
        # Indirgeme tarafinda "kosturulabilir" demek: UHF aramasi kosuyor,
        # parametreler sonlu cikiyor ve panel dolu. Mesafe sayisi ve tohum
        # sayisi KUCULTULUR (senaryonun kendi degerleri degil).
        from qlab.engine.tb.factory import build_downfold

        kisa = dataclasses.replace(
            scenario,
            distances_bohr=scenario.distances_bohr[:_DOWNFOLD_SAMPLE_DISTANCES],
            seeds=_DOWNFOLD_SAMPLE_SEEDS,
        )
        parametreler, provenance = build_downfold(kisa)
        assert len(parametreler) == len(kisa.distances_bohr)
        for p in parametreler:
            assert np.isfinite(p.epsilon_bar) and np.isfinite(p.delta)
            assert np.all(np.isfinite(p.hoppings)), (
                f"{scenario_path.name}: hoplamalar sonlu degil"
            )
            assert len(p.hoppings) == kisa.n_neighbour
            assert p.n_site == kisa.ring_size
        assert provenance.method.startswith("UHF indirgemesi")
        return

    if isinstance(scenario, SpinScenario):
        # Spin tarafinda "kosturulabilir" demek: ornekleyici koşuyor,
        # gozlemlenebilirler sonlu cikiyor ve panel dolu. Sicaklik sayisi ve
        # zincir uzunlugu KUCULTULUR (senaryonun kendi degerleri degil).
        from qlab.engine.spin.factory import build_spin, provenance_for
        from qlab.viz.spin_frames import build_spin_payload

        kisa = dataclasses.replace(
            scenario,
            temperatures=scenario.temperatures[:_SPIN_SAMPLE_TEMPERATURES],
            n_sweep=_SPIN_SAMPLE_SWEEPS,
            n_burn=_SPIN_SAMPLE_BURN,
        )
        yuk = build_spin_payload(build_spin(kisa))
        assert yuk.temperatures.shape == (len(kisa.temperatures),)
        assert np.all(np.isfinite(yuk.magnetisation)), (
            f"{scenario_path.name}: miknatislanma sonlu degil"
        )
        assert np.all(np.isfinite(yuk.magnetisation_error))
        assert np.all(np.isfinite(yuk.binder))
        provenance = provenance_for(
            kisa, plateau_reached=yuk.plateau_reached, tau_int=yuk.tau_int
        )
        assert provenance.method.startswith("spin Monte Carlo")
        # Bu senaryo turunun panelinde IKI EK SATIR ZORUNLUDUR ve baska
        # hicbir turde yoktur; kesif-tabanli bu testte kilitlenmesi, yeni
        # bir spin senaryosu eklendiginde de aranacagi anlamina gelir.
        assert provenance.circularity is not None
        assert provenance.sampling is not None
        return

    if isinstance(scenario, EMScenario):
        # EM tarafinda "kosturulabilir" demek: FDTD kosuyor, spektrum tepe
        # buluyor, merdiven kuruluyor ve panel dolu. Basamak sayisi ve adim
        # carpani KUCULTULUR (senaryonun kendi degerleri degil).
        from qlab.engine.em.factory import build_em, provenance_for
        from qlab.viz.em_frames import build_em_payload

        kisa = dataclasses.replace(
            scenario,
            radii=scenario.radii[:_EM_SAMPLE_RADII],
            step_factor=_EM_SAMPLE_STEP_FACTOR,
        )
        yuk = build_em_payload(build_em(kisa))
        assert yuk.ratios.shape == (len(kisa.radii),)
        assert np.all(np.isfinite(yuk.ratios)), (
            f"{scenario_path.name}: kip orani sonlu degil"
        )
        assert np.all(yuk.noise >= 0.0)
        assert np.all(yuk.peaks >= 2)
        provenance = provenance_for(
            kisa,
            windows=yuk.windows,
            ratios=tuple(float(x) for x in yuk.ratios),
            noise=tuple(float(x) for x in yuk.noise),
            peaks=tuple(int(x) for x in yuk.peaks),
            orders=yuk.ladder.orders,
            p_inf=yuk.ladder.p_inf,
            uncertainty=yuk.ladder.uncertainty,
            verdict=yuk.ladder.verdict,
        )
        assert provenance.method.startswith("Maxwell FDTD")
        # Bu senaryo turunun panelinde DORT EK SATIR ZORUNLUDUR; ucu baska
        # hicbir turde yoktur. Kesif-tabanli bu testte kilitlenmesi, yeni bir
        # EM senaryosu eklendiginde de aranacagi anlamina gelir.
        assert provenance.circularity is not None
        assert provenance.anchor_class is not None
        assert provenance.window is not None
        assert provenance.convergence_diagnosis is not None
        # SIG BIR MERDIVEN "OTURDU" DIYEMEZ -- ve panel bunu SOYLER.
        assert yuk.ladder.verdict == "YARGILANAMAZ"
        assert yuk.ladder.settled_order is None
        assert provenance.converged is False
        assert "OTURDU(" not in provenance.convergence_diagnosis.split("[BEYAN")[0]
        return

    if isinstance(scenario, BridgeScenario):
        # Köprü tarafında "koşturulabilir" demek: tablo koda karşı taranıyor
        # (tutarsızlık YOK), kenarın İKİ tabanı da okunuyor ve panel dolu.
        # YOL B penceresi KÜÇÜLTÜLÜR (N = 50, 2 periyot; ölçüldü, iki taban
        # da SAGLAM kalıyor); ikiz `a` ve genlikler senaryonun kendisidir.
        from qlab.viz.bridge_frames import build_bridge_payload, provenance_for

        kisa = dataclasses.replace(scenario, n_bolme=50, periyot_sayisi=2)
        yuk = build_bridge_payload(kisa)
        assert yuk.tutarsizliklar == ()
        # 24.09.2026: `md-tb` kenarı yazıldı (OLCULMEDI -> GERCEK_KOPRU).
        # 27.09.2026: `hf-spin`; 28.09.2026: `tdse-spin`, `md-em`, `tb-em`
        # kenarsız, hükümle (OLCULMEDI -> TAUTOLOJI).
        assert yuk.kova_sayim == {
            "GERCEK_KOPRU": 3, "TAUTOLOJI": 8, "YAPISAL_YOK": 8,
            "BENZERLIK": 0, "OLCULMEDI": 2,
        }
        assert len(yuk.kenarlar) == 4
        for taban in (yuk.a1, yuk.a2):
            assert math.isfinite(taban.taban) and taban.taban > 0.0, (
                f"{scenario_path.name}: kenar tabanı sonlu/pozitif değil"
            )
            assert taban.hukum == "SAGLAM"
        assert yuk.a1.tur != yuk.a2.tur
        provenance = provenance_for(yuk)
        assert provenance.method.startswith("Moduller arasi yokluk haritasi")
        # Bu senaryo türünün panelinde altı ek satır ZORUNLUDUR.
        assert provenance.circularity is not None
        assert provenance.anchor_class is not None
        assert provenance.theory_selector is not None
        assert provenance.window is not None
        assert provenance.convergence_diagnosis is not None
        assert provenance.converged is True
        return

    if isinstance(scenario, GravScenario):
        # Yerçekimi tarafında "koşturulabilir" demek: iki çözücü de koşuyor
        # (jeodezik ODE + N-cisim BVP), her çıpanın KENDİ borusundan bir
        # hata çubuğu okunuyor ve panel dolu. Yalnız yörünge ÖRNEKLEMESİ
        # kısaltılır; hükme giren hiçbir düğmeye dokunulmaz.
        from qlab.engine.grav.factory import build_grav, provenance_for
        from qlab.viz.grav_frames import build_grav_payload

        kisa = dataclasses.replace(
            scenario, trajectory_points=_GRAV_SAMPLE_POINTS
        )
        setup = build_grav(kisa)
        yuk = build_grav_payload(setup)
        assert yuk.phi.shape == (_GRAV_SAMPLE_POINTS,)
        assert np.all(np.isfinite(yuk.r)) and np.all(yuk.r > 0.0)
        for sayi in (yuk.coefficient, yuk.second_order, yuk.deflection,
                     yuk.eight, *yuk.angles):
            assert math.isfinite(sayi), f"{scenario_path.name}: {sayi} sonlu degil"
        provenance = provenance_for(
            kisa,
            setup=setup,
            floors=yuk.floors,
            second_order=yuk.second_order,
            closed_forms=yuk.closed_forms,
            eight=yuk.eight,
            angles=yuk.angles,
        )
        assert provenance.method.startswith("Yercekimi ve uzay-zaman")
        # Bu senaryo türünün panelinde BEŞ EK SATIR ZORUNLUDUR ve `Kuram
        # seçici` başka HİÇBİR türde yoktur. Keşif-tabanlı bu testte
        # kilitlenmesi, yeni bir yerçekimi senaryosu eklendiğinde de
        # aranacağı anlamına gelir.
        assert provenance.circularity is not None
        assert provenance.anchor_class is not None
        assert provenance.theory_selector is not None
        assert provenance.window is not None
        assert provenance.convergence_diagnosis is not None
        # DEJENERE bir boru PANELDE DURUR ama ÇIPA hükmüne girmez.
        assert [f.verdict for f in yuk.floors].count("DEJENERE") == 1
        assert provenance.converged is True
        return

    assert isinstance(scenario, TDSEScenario)
    grid, potential, psi0, _ = factory.build(scenario)
    assert potential.shape == (scenario.grid.n,)
    assert np.all(np.isfinite(potential))
    norm = float(np.sum(np.abs(psi0) ** 2) * grid.dx)
    assert norm == pytest.approx(1.0, abs=1e-10, rel=0.0)


# STO-3G'nin H2 icin GERCEK denge uzunlugu; Brent ile olculur ve
# tests/hf/test_physical.py::test_coarse_scan_fit_sits_above_the_true_sto3g_minimum
# tarafindan kilitlenir. Burada yalnizca referans olarak kullanilir.
_TRUE_STO3G_MINIMUM_ANGSTROM = 0.7122299


@pytest.mark.slow
def test_shipped_h2_scan_fit_is_grid_biased_but_less_than_the_uniform_grid():
    """Kabul kriteri 2'nin bildirdigi sayi BU senaryodan cikmali.

    README bir zamanlar bu komutu kanit gosterip 0.7156 A bildiriyordu; o
    sayi baska bir izgaranin (test_physical'daki linspace(0.4, 3.0, 40),
    adim 0.0667 A) ciktisiydi. Dagitilan senaryo minimum civarinda daha
    sik ortuyor (0.65 / 0.70 / 0.74) ve baska bir sayi verir.

    Test ayni zamanda "izgara siklastikca es kaybolur" iddiasini ISPATLAR:
    daha sik izgaranin yanliligi daha kucuk cikmali. Bu bir aciklama degil,
    olculen bir siralamadir.
    """
    scenario = load_scenario(_SCENARIO_DIR / "h2_bond_scan.yaml")
    assert isinstance(scenario, HFScanScenario)

    result = run_scan(scenario)
    values = np.asarray(scenario.values, dtype=float)
    index = int(np.argmin(result.energies))
    assert 0 < index < len(values) - 1, "minimum taranan aralikta olmali"

    a, b, _ = np.polyfit(
        values[index - 1 : index + 2], result.energies[index - 1 : index + 2], 2
    )
    fit = -b / (2 * a)

    assert fit == pytest.approx(0.7134824, abs=2e-5), (
        "kabul kriteri 2'nin bildirdigi sayi; degistiyse README de degismeli"
    )
    assert abs(fit - 0.741) / 0.741 < 0.08, "kriterin kendisi: deneyselden ±%8"

    bias = fit - _TRUE_STO3G_MINIMUM_ANGSTROM
    uniform_grid_bias = 0.0033483  # linspace(0.4, 3.0, 40), adim 0.0667 A
    assert bias > 0, "parabol fiti gercek minimumun USTUNDE kalir"
    assert bias < uniform_grid_bias, (
        "minimum civarinda daha sik ortulen izgara daha kucuk yanlilik vermeli; "
        f"olculen {bias:.7f} A, kaba izgara {uniform_grid_bias:.7f} A"
    )
