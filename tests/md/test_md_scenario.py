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
