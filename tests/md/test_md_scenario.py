"""MD senaryosu: YAML'dan motora."""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.factory import build, provenance_for
from qlab.errors import ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import MDScenario


def _morse_senaryo(dt: float = 0.78) -> MDScenario:
    return MDScenario(
        name="t",
        force_field={"kind": "morse", "params": {"d_e": 0.174434333, "r_e": 1.401118538, "a": 1.028997701}},
        initial={
            "positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
            "masses_amu": [1.00782503207, 1.00782503207],
        },
        thermostat=None,
        dt=dt,
        n_steps=10,
    )


def test_md_senaryosu_yamldan_yukleniyor(tmp_path):
    yol = tmp_path / "md.yaml"
    yol.write_text(
        "kind: md\n"
        "name: H2 Morse testi\n"
        "force_field:\n"
        "  kind: morse\n"
        "  params: {d_e: 0.174434333, r_e: 1.401118538, a: 1.028997701}\n"
        "initial:\n"
        "  positions_bohr: [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]\n"
        "  masses_amu: [1.00782503207, 1.00782503207]\n"
        "dt: 0.78\n"
        "n_steps: 100\n"
        "save_every: 10\n",
        encoding="utf-8",
    )
    senaryo = load_scenario(str(yol))
    assert isinstance(senaryo, MDScenario)
    assert senaryo.n_steps == 100


def test_bilinmeyen_anahtar_sessizce_yutulmuyor(tmp_path):
    """Mevcut kural (loader dokstring'i): yazım hatası varsayılana DÜŞMEZ."""
    yol = tmp_path / "md.yaml"
    yol.write_text(
        "kind: md\nname: t\nforce_field: {kind: morse, params: {}}\n"
        "initial: {positions_bohr: [[0,0,0]], masses_amu: [1.0]}\n"
        "dt: 1.0\nn_steps: 1\nn_stepss: 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ScenarioError, match="n_stepss"):
        load_scenario(str(yol))


def test_kutleler_amu_verilip_m_e_olarak_kuruluyor():
    """SENARYODAN MOTORA GEÇERKEN DÖNÜŞÜM -- modülün en tehlikeli hatası.

    ÖLÇÜLDÜ (tasarım §3): dönüşüm atlanırsa H2'nin periyodu 42.86x yanlış
    çıkar ve HİÇBİR ŞEY PATLAMAZ. Sayı LİTERAL: 1.00782503207 amu =
    1837.1526 m_e.
    """
    durum, _, _ = build(_morse_senaryo())
    np.testing.assert_allclose(
        durum.masses, np.array([1837.1526, 1837.1526]), rtol=0.0, atol=1e-3
    )


def test_dt_kararlilik_sinirini_asarsa_REDDEDILIYOR():
    """ÖLÇÜLDÜ: velocity-Verlet omega*dt = 2.0'da patlar (1.999 kararlı,
    2.001 -> 4.28e+54, 2.05 -> nan).

    Sessizce nan üreten bir koşu yerine tipli hata verilir. Harmonik alanda
    omega analitiktir; kontrol orada yapılabilir.
    """
    senaryo = MDScenario(
        name="t",
        force_field={"kind": "harmonic", "params": {"k": 1.0}},
        initial={
            "positions_bohr": [[0.1, 0.0, 0.0]],
            "masses_amu": [1.0 / 1822.888486209],  # m = 1 m_e -> omega = 1
        },
        thermostat=None,
        dt=2.5,
        n_steps=10,
    )
    with pytest.raises(ScenarioError, match="kararlılık"):
        build(senaryo)


def test_provenance_kutleleri_ve_yaklasimlari_beyan_ediyor():
    """Kullanılan kütleler EKRANDA görünmeli: 1837.15 gibi bir sayıyı gören
    biri "1 amu = 1" hatasını fark edebilir."""
    p = provenance_for(_morse_senaryo(), energy_drift=1e-9)
    metin = " ".join(p.approximations) + " " + " ".join(p.neglected)
    assert "klasik" in metin.lower()
    assert "nokta parçacık" in metin.lower()
    assert "1837" in str(p.diagnostics["masses_electron_mass"])


def test_bilinmeyen_kuvvet_alani_reddediliyor():
    senaryo = MDScenario(
        name="t",
        force_field={"kind": "yok_boyle_bir_sey", "params": {}},
        initial={"positions_bohr": [[0.0, 0.0, 0.0]], "masses_amu": [1.0]},
        thermostat=None,
        dt=1.0,
        n_steps=1,
    )
    with pytest.raises(ScenarioError, match="kuvvet alanı"):
        build(senaryo)
def test_fcc_kafes_kurulumu_dogru_sayida_parcacik_veriyor():
    """`lattice: fcc` dalı: N = 4 * hucre^3 olmalı ve kutu kenarı yoğunluktan
    türetilmeli. Sayı LİTERAL: 108 = 4 * 3^3.
    """
    senaryo = MDScenario(
        name="t",
        force_field={
            "kind": "lennard_jones",
            "params": {
                "epsilon": 3.7938402530e-04, "sigma": 6.4345174543,
                "box_length": 33.0656, "cutoff": 16.0862936,
            },
        },
        initial={
            "lattice": "fcc", "n_particles": 108,
            "density_reduced": 0.8442, "masses_amu": [39.948],
        },
        thermostat=None,
        dt=445.732231,
        n_steps=1,
    )
    durum, _, _ = build(senaryo)
    assert durum.positions.shape == (108, 3)
    np.testing.assert_allclose(
        durum.masses, np.full(108, 39.948 * 1822.888486209), rtol=0.0, atol=1e-6
    )


def test_hartree_fock_kuvvet_alani_fabrikadan_kuruluyor():
    """`hartree_fock` dalı olmadan Plan D YAML'dan çağrılamaz.

    Dağıtım KAPALI bir sayımdır (`else: raise ScenarioError`), yani dal
    eklenmeden senaryo ham hata verir.
    """
    senaryo = MDScenario(
        name="t",
        force_field={
            "kind": "hartree_fock",
            "params": {"symbols": ["H", "H"], "basis": "sto-3g", "step": 1e-5},
        },
        initial={
            "positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3959193602]],
            "masses_amu": [1.00782503207, 1.00782503207],
        },
        thermostat=None,
        dt=5.0,
        n_steps=1,
    )
    durum, alan, termostat = build(senaryo)
    assert alan.__class__.__name__ == "HartreeFockGradient"
    assert alan.step == pytest.approx(1e-5, rel=0.0, abs=0.0)
    assert termostat is None


def test_HF_dalinda_yaklasim_listesi_ANALITIK_MODEL_DEMIYOR():
    """Şeffaflık panelinin YALAN söylememesi.

    `_YAKLASIMLAR` listesindeki "kuvvet alanı analitik bir modeldir,
    hesaplanmış bir yüzey değildir" satırı Plan D için DÜZ YALANDIR ve
    panelde basılır. HF dalı kendi listesini taşımak zorunda.

    Bu test o yalanı kilitler: HF provenance'ında o cümle GEÇMEMELİ, ve
    HF'ye özgü dört şey GEÇMELİ.
    """
    senaryo = MDScenario(
        name="t",
        force_field={
            "kind": "hartree_fock",
            "params": {"symbols": ["H", "H"], "basis": "sto-3g", "step": 1e-5},
        },
        initial={
            "positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3959193602]],
            "masses_amu": [1.00782503207, 1.00782503207],
        },
        thermostat=None,
        dt=5.0,
        n_steps=1,
    )
    prov = provenance_for(senaryo, energy_drift=1e-10)
    metin = " ".join(prov.approximations + prov.neglected).lower()

    # İKİ yalan da gitmiş olmalı (factory.py:16 ve :18 -- kaynak okundu)
    assert "analitik bir model" not in metin
    assert "elektron yok" not in metin
    for beklenen in ("born-oppenheimer", "tek slater", "sonlu baz", "sayısal gradyan"):
        assert beklenen in metin, f"HF beyanında eksik: {beklenen}"
    assert "sto-3g" in " ".join(str(v) for v in prov.diagnostics.values()).lower()
    assert prov.diagnostics["gradient_step_bohr"] == pytest.approx(1e-5)


def test_HF_dalinda_kararlilik_denetiminin_ATLANDIGI_beyan_ediliyor():
    """`omega*dt < 2` denetimi yalnız `harmonic` dalındadır; HF'de omega
    analitik DEĞİLDİR ve denetim atlanır.

    Atlamak meşru bir seçimdir; SESSİZ atlamak değildir.
    """
    senaryo = MDScenario(
        name="t",
        force_field={
            "kind": "hartree_fock",
            "params": {"symbols": ["H", "H"], "basis": "sto-3g"},
        },
        initial={
            "positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3959193602]],
            "masses_amu": [1.00782503207, 1.00782503207],
        },
        thermostat=None,
        dt=5.0,
        n_steps=1,
    )
    prov = provenance_for(senaryo, energy_drift=1e-10)
    metin = " ".join(prov.approximations + prov.neglected).lower()
    assert "kararlılık" in metin and "denetlenmiyor" in metin
