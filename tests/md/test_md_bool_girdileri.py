"""Grup 6 serit A (11.10.2026): MD senaryosunda bool / karmasik / tasan sayi ve D14 kamu yolu.

OLCULDU (HEAD a9ec48a; `load_scenario` + `build`, sevk edilen
`tb_zincir_md.yaml` / `argon_nvt.yaml`'in tek satiri bozularak):
- `masses_amu: [true]` (ve YAML 1.1 `yes`) 1 amu'luk koşu kuruyordu,
  `positions_bohr`daki `true` 1.0 bohr okunuyordu (`np.asarray(..., float)`).
- fcc `n_particles: 108.5` / `108.9` / `'108'` SESSIZCE 108 atom.
- `tb_ss_sigma`: `genlik_hartree: true` -> genlik 1.0 hartree; 401 haneli
  `n_electron` / `electron_temperature_kelvin` / `box_x_bohr` ham OverflowError.
- `MDScenario(...)` dogrudan: Kelvin `np.True_` / `np.array(True)` 1 K diye
  geciyordu, `np.complex128` uyariyla gercel kismina iniyordu.
- D14: atom 1 atom 0'a 5e-10 bohr yaklasinca TB MD yolu kosuyordu (rc = 0).
Katman: `initial` icerigi ve Kelvin `md/factory.build`de `ScenarioError`;
yasa ve kuvvet sinifinin deger kapilari `InvalidInputError` (ikisi de
`run.main`de rc = 1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import qlab.run as run_modulu
from qlab.engine.md.factory import build
from qlab.engine.md.forcefields import Harmonic, HartreeFockGradient, LennardJones, Morse
from qlab.engine.md.thermostats import Langevin, NoseHoover
from qlab.errors import InvalidInputError, ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import MDScenario
from qlab.units import KB_HARTREE_PER_KELVIN

from .h2_morse_sabitleri import H2_A, H2_DE, H2_RE

_SEVK = Path(__file__).resolve().parents[2] / "scenarios"
_TB = (_SEVK / "tb_zincir_md.yaml").read_text(encoding="utf-8").replace("n_steps: 400", "n_steps: 4")
_ARGON = (_SEVK / "argon_nvt.yaml").read_text(encoding="utf-8").replace("n_steps: 20000", "n_steps: 4")
_X0 = "0.10204595606925913"
_ATOM1 = "    - [1.8722167484342909, 0.0, 0.0]\n"
_KUTLE = "  masses_amu: [1.00782503207]\n"
_DEV = "1" + "0" * 400  # 10**400: float'a sigmaz

#: (ad, kaynak, eski, yeni, mesajda beklenen parca)
_BOZUK = [
    ("d14_cakisik_5e-10", _TB, _ATOM1, f"    - [{float(_X0) + 5e-10!r}, 0.0, 0.0]\n", "cakisik atom cifti (0, 1)"),
    ("konum_true", _TB, _ATOM1, "    - [true, 0.0, 0.0]\n", "initial.positions_bohr"),
    ("kutle_true", _TB, _KUTLE, "  masses_amu: [true]\n", "initial.masses_amu"),
    ("kutle_yes", _TB, _KUTLE, "  masses_amu: [yes]\n", "initial.masses_amu"),
    ("kutle_skaler_true", _TB, _KUTLE, "  masses_amu: true\n", "initial.masses_amu"),
    ("hiz_true", _TB, _KUTLE, _KUTLE + "  velocities: [[true, 0, 0]" + ", [0, 0, 0]" * 15 + "]\n",
     "initial.velocities"),
    ("konum_dev", _TB, _ATOM1, f"    - [{_DEV}, 0.0, 0.0]\n", "initial.positions_bohr"),
    ("kutle_dev", _TB, _KUTLE, f"  masses_amu: [{_DEV}]\n", "initial.masses_amu"),
    ("genlik_true", _TB, "genlik_hartree: -0.1", "genlik_hartree: true", "genlik_hartree tek bir sayi"),
    ("itme_us_true", _TB, "us: 6.0", "us: true", "us tek bir sayi"),
    ("genlik_dev", _TB, "genlik_hartree: -0.1", f"genlik_hartree: -{_DEV}", "genlik_hartree sayiya cevrilemiyor"),
    ("n_electron_dev", _TB, "n_electron: 16", f"n_electron: {_DEV}", "n_elektron sayiya cevrilemiyor"),
    ("kelvin_dev", _TB, "electron_temperature_kelvin: 631.6", f"electron_temperature_kelvin: {_DEV}",
     "electron_temperature_kelvin tek bir say"),
    ("kutu_dev", _TB, "box_x_bohr: 32.0", f"box_x_bohr: {_DEV}", "kutu_x_bohr sayiya cevrilemiyor"),
    ("fcc_108.5", _ARGON, "n_particles: 108", "n_particles: 108.5", "n_particles tam sayı olmalı"),
    ("fcc_108.9", _ARGON, "n_particles: 108", "n_particles: 108.9", "n_particles tam sayı olmalı"),
    ("fcc_tirnakli", _ARGON, "n_particles: 108", "n_particles: '108'", "n_particles tam sayı olmalı"),
    ("fcc_true", _ARGON, "n_particles: 108", "n_particles: true", "n_particles tam sayı olmalı"),
    ("fcc_yogunluk_true", _ARGON, "density_reduced: 0.8442", "density_reduced: true", "density_reduced sayı olmalı"),
    # Grup 6 duzeltme turu (11.10.2026; HEAD a9ec48a'da ham OverflowError /
    # ZeroDivisionError / TypeError ya da yanlis nedenli rc = 1 -- olculdu).
    ("fcc_n_dev", _ARGON, "n_particles: 108", f"n_particles: {_DEV}", "n_particles float'a sığmıyor"),
    ("fcc_n_sifir", _ARGON, "n_particles: 108", "n_particles: 0", "n_particles pozitif olmalı"),
    ("fcc_n_negatif", _ARGON, "n_particles: 108", "n_particles: -108", "n_particles pozitif olmalı"),
    ("fcc_yogunluk_dev", _ARGON, "density_reduced: 0.8442", f"density_reduced: {_DEV}",
     "density_reduced float'a sığmıyor"),
    ("fcc_yogunluk_sifir", _ARGON, "density_reduced: 0.8442", "density_reduced: 0",
     "density_reduced pozitif ve sonlu olmalı"),
    ("fcc_yogunluk_negatif", _ARGON, "density_reduced: 0.8442", "density_reduced: -0.8442",
     "density_reduced pozitif ve sonlu olmalı"),
    ("fcc_yogunluk_nan", _ARGON, "density_reduced: 0.8442", "density_reduced: .nan",
     "density_reduced pozitif ve sonlu olmalı"),
    ("fcc_yogunluk_inf", _ARGON, "density_reduced: 0.8442", "density_reduced: .inf",
     "density_reduced pozitif ve sonlu olmalı"),
    ("fcc_sigma_true", _ARGON, "sigma: 6.4345174543", "sigma: true", "sigma sayı olmalı, bool verildi"),
    ("fcc_sigma_dev", _ARGON, "sigma: 6.4345174543", f"sigma: {_DEV}", "sigma float'a sığmıyor"),
]


@pytest.mark.parametrize(("ad", "kaynak", "eski", "yeni", "parca"), _BOZUK, ids=[b[0] for b in _BOZUK])
def test_bozuk_MD_girdisi_KAMU_YOLUNDA_rc1(tmp_path, capsys, ad, kaynak, eski, yeni, parca):
    """`run.main` rc = 1 ve nedeni soyler; .npz yazilmaz (HEAD'de ya rc = 0 ya ham istisna)."""
    assert kaynak.count(eski) == 1, ad
    yol = tmp_path / f"{ad}.yaml"
    yol.write_text(kaynak.replace(eski, yeni), encoding="utf-8")
    cikti = tmp_path / f"{ad}.npz"
    assert run_modulu.main([str(yol), "-o", str(cikti)]) == 1
    hata = capsys.readouterr().err
    assert parca in hata, hata
    assert not cikti.exists()


def test_fcc_TAM_DEGERLI_float_ve_tam_sayi_KABUL(tmp_path):
    """Kontrol: `108.0` (yukleyicinin `_yaml_tam_sayisi` kurali) ve `108` ayni kafesi kurar."""
    konumlar = []
    for n in ("108", "108.0"):
        yol = tmp_path / f"a{n}.yaml"
        yol.write_text(_ARGON.replace("n_particles: 108", f"n_particles: {n}"), encoding="utf-8")
        konumlar.append(build(load_scenario(yol))[0].positions)
    assert konumlar[0].shape == (108, 3) and np.array_equal(konumlar[0], konumlar[1])


def _tb_senaryo(kelvin=631.6, initial=None) -> MDScenario:
    yasa = dict(d0_bohr=2.0, r_on_bohr=3.0, r_cut_bohr=3.8)
    R = np.zeros((8, 3))
    R[:, 0] = 2.0 * np.arange(8)
    return MDScenario(
        name="t",
        force_field={"kind": "tb_ss_sigma", "params": {
            "ss_sigma": dict(genlik_hartree=-0.1, us=2.0, **yasa),
            "itme": dict(genlik_hartree=0.05, us=6.0, **yasa),
            "n_electron": 8, "electron_temperature_kelvin": kelvin, "box_x_bohr": 16.0,
        }},
        initial=initial or {"positions_bohr": R.tolist(), "masses_amu": [1.0]},
        thermostat=None, dt=10.0, n_steps=2, save_every=1,
    )


@pytest.mark.parametrize("kelvin", [
    pytest.param(np.True_, id="np.True_"),
    pytest.param(np.array(True), id="0d-bool-dizi"),
    pytest.param(np.complex128(631.6), id="np.complex128"),
    pytest.param(10**400, id="10**400"),
    pytest.param([[631.6], [631.6, 1.0]], id="duzensiz-liste"),
    # B4 (grup 9): 0 boyutlu NESNE dizisi; HEAD 448f808'de 1 K ve 631.6 K diye geciyordu.
    pytest.param(np.array(True, dtype=object), id="0d-nesne-bool"),
    pytest.param(np.array("631.6", dtype=object), id="0d-nesne-dize"),
])
def test_KELVIN_numpy_bool_karmasik_tasma_DOGRUDAN_senaryoda_ScenarioError(kelvin):
    """`MDScenario(...)` dogrudan kurulunca `build` tek bekcidir (YAML bu tipleri uretmez)."""
    with pytest.raises(ScenarioError, match="electron_temperature_kelvin"):
        build(_tb_senaryo(kelvin))


def test_KELVIN_numpy_GERCEL_KABUL_deger_aynen():
    """Kontrol: `np.float64`, 0 boyutlu gercel dizi (B4: nesne dizisi de) gecer; kT = K * k_B."""
    for kelvin in (np.float64(631.6), np.array(631.6), 631.6, np.array(631.6, dtype=object)):
        alan = build(_tb_senaryo(kelvin))[1]
        assert alan.kT_hartree == 631.6 * KB_HARTREE_PER_KELVIN


# ------------------------------------------- kuvvet alani / termostat `true` --

#: Kosan (rc = 0) taban senaryolar: `_ARGON`un `save_every: 50`si 4 adimda tek
#: kayit birakir ve rc = 1'i BASKA bir neden verir; burada taban gecerli.
_MORSE = (_SEVK / "h2_morse.yaml").read_text(encoding="utf-8").replace("n_steps: 4000", "n_steps: 8")
_ARGON_KOSAR = _ARGON.replace("save_every: 50", "save_every: 1")
_HARMONIK = _MORSE.replace("kind: morse", "kind: harmonic").replace(
    "params: {d_e: 0.174434333, r_e: 1.401118538, a: 1.028997701}", "params: {k: 0.1}"
)

#: (ad, kaynak, eski, yeni, mesajda beklenen parca). OLCULDU (HEAD a9ec48a,
#: `run.main`): HEPSI rc = 0 ve .npz yaziyordu -- sinif kapisi
#: `isfinite(x) and x > 0` bool'u 1.0 / 1 diye geciriyordu.
_PARAM_TRUE = [
    ("morse_d_e_true", _MORSE, "d_e: 0.174434333", "d_e: true", "morse parametresi `d_e`"),
    ("morse_a_true", _MORSE, "a: 1.028997701", "a: true", "morse parametresi `a`"),
    ("morse_r_e_yes", _MORSE, "r_e: 1.401118538", "r_e: yes", "morse parametresi `r_e`"),
    ("lj_epsilon_true", _ARGON_KOSAR, "epsilon: 3.7938402530e-04", "epsilon: true",
     "lennard_jones parametresi `epsilon`"),
    ("lj_cutoff_true", _ARGON_KOSAR, "cutoff: 16.0862936", "cutoff: true", "lennard_jones parametresi `cutoff`"),
    ("langevin_friction_true", _ARGON_KOSAR, "friction: 2.2434e-05", "friction: true",
     "langevin parametresi `friction`"),
    ("langevin_seed_true", _ARGON_KOSAR, "seed: 2026", "seed: true", "langevin parametresi `seed`"),
    ("harmonik_k_true", _HARMONIK, "params: {k: 0.1}", "params: {k: true}", "harmonic parametresi `k`"),
    ("harmonik_merkez_true", _HARMONIK, "params: {k: 0.1}", "params: {k: 0.1, center: [[true, 0, 0]]}",
     "harmonic parametresi `center`"),
]


@pytest.mark.parametrize(("ad", "kaynak"), [("morse", _MORSE), ("argon", _ARGON_KOSAR), ("harmonik", _HARMONIK)],
                         ids=["morse", "argon", "harmonik"])
def test_PARAM_tabanlari_KOSAR_rc0(tmp_path, ad, kaynak):
    """Kontrol: bozulmamis taban rc = 0 ve .npz yazar -- asagidaki rc = 1'i kapi verir."""
    yol = tmp_path / f"{ad}.yaml"
    yol.write_text(kaynak, encoding="utf-8")
    cikti = tmp_path / f"{ad}.npz"
    assert run_modulu.main([str(yol), "-o", str(cikti)]) == 0
    assert cikti.exists()


@pytest.mark.parametrize(("ad", "kaynak", "eski", "yeni", "parca"), _PARAM_TRUE, ids=[b[0] for b in _PARAM_TRUE])
def test_PARAM_true_KAMU_YOLUNDA_rc1_ANAHTARI_ADIYLA(tmp_path, capsys, ad, kaynak, eski, yeni, parca):
    """Kuvvet alani / termostat parametresinde `true`/`yes`: rc = 1, anahtar adiyla, .npz yok."""
    assert kaynak.count(eski) == 1, ad
    yol = tmp_path / f"{ad}.yaml"
    yol.write_text(kaynak.replace(eski, yeni), encoding="utf-8")
    cikti = tmp_path / f"{ad}.npz"
    assert run_modulu.main([str(yol), "-o", str(cikti)]) == 1
    hata = capsys.readouterr().err
    assert parca in hata and "bool (true/yes)" in hata, hata
    assert not cikti.exists()


def _harmonik_senaryo(params: dict, kind: str = "harmonic") -> MDScenario:
    return MDScenario(
        name="t", force_field={"kind": kind, "params": params},
        initial={"positions_bohr": [[0.0, 0, 0], [1.4, 0, 0]], "masses_amu": [1.0, 1.0]},
        thermostat=None, dt=1.0, n_steps=2, save_every=1,
    )


@pytest.mark.parametrize("k", [
    pytest.param(np.True_, id="np.True_"),
    pytest.param(np.array(True), id="0d-bool-dizi"),
    pytest.param(np.complex128(0.1), id="np.complex128"),
])
def test_PARAM_numpy_bool_ve_karmasik_DOGRUDAN_senaryoda_ScenarioError(k):
    """`MDScenario(...)` dogrudan: numpy bool / karmasik yay sabiti de anahtar adiyla reddedilir."""
    with pytest.raises(ScenarioError, match="harmonic parametresi `k`"):
        build(_harmonik_senaryo({"k": k}))


def test_PARAM_TIPI_bool_OLAN_alan_MUAF_ve_bilinmeyen_anahtar_SINIFA_kalir():
    """Asiri reddetmeye karsi: `HartreeFockGradient.stability: false` gecerli (tipi `bool`).

    Sinifta olmayan `de: true` bool kapisina DEGIL sinifin "unexpected keyword"
    hatasina duser (kullaniciya yazim hatasini soyler, bool'u degil).
    """
    alan = build(_harmonik_senaryo({"symbols": ["H", "H"], "stability": False}, kind="hartree_fock"))[1]
    assert alan.stability is False
    with pytest.raises(ScenarioError, match="unexpected keyword argument 'de'"):
        build(_harmonik_senaryo({"de": True, "r_e": 1.4, "a": 1.0}, kind="morse"))


@pytest.mark.parametrize(("anahtar", "deger"), [
    pytest.param("positions_bohr", [[0.0, 0, 0], [np.True_, 0, 0]], id="konum-np.True_"),
    pytest.param("positions_bohr", [[0.0, 0, 0], [np.complex128(2.0), 0, 0]], id="konum-np.complex128"),
    pytest.param("positions_bohr", np.array([[0.0, 0, 0], [True, 0, 0]], dtype=object), id="konum-object-dizi-bool"),
    pytest.param("masses_amu", np.array([True]), id="kutle-bool-dizi"),
    pytest.param("velocities", np.zeros((2, 3), dtype=bool), id="hiz-bool-dizi"),
])
def test_initial_numpy_bool_ve_karmasik_OGE_DOGRUDAN_senaryoda_ScenarioError(anahtar, deger):
    """`_senaryo_dizisi` ic ice liste, nesne dizisi ve bool/karmasik dtype'li diziyi tarar."""
    ilk = {"positions_bohr": [[0.0, 0, 0], [2.0, 0, 0]], "masses_amu": [1.0, 1.0]} | {anahtar: deger}
    senaryo = MDScenario(
        name="t", force_field={"kind": "harmonic", "params": {"k": 0.1}}, initial=ilk,
        thermostat=None, dt=1.0, n_steps=2, save_every=1,
    )
    with pytest.raises(ScenarioError, match=f"initial.{anahtar}"):
        build(senaryo)


# ------------------------- B3 (grup 9): sinif DOGRUDAN kurulunca bool kapisi --

#: Her sinifin gecerli kurulumu. OLCULDU (HEAD 448f808): bu alanlarda bool
#: (True / np.True_ / np.array(True)) 54 hucrenin 43'unde SESSIZCE kabul ediliyordu.
_SINIF_GECERLI = {
    Harmonic: dict(k=0.5, center=np.zeros((1, 3))),
    Morse: dict(d_e=H2_DE, r_e=H2_RE, a=H2_A),
    LennardJones: dict(epsilon=3.79e-4, sigma=6.43, box_length=32.4, cutoff=16.0),
    HartreeFockGradient: dict(symbols=("H", "H"), step=1e-5, max_bond_bohr=21.0),
    Langevin: dict(target_temperature_kelvin=300.0, friction=1e-3, seed=1),
    NoseHoover: dict(target_temperature_kelvin=300.0, relaxation_time=100.0, chain_length=1, n_dof=3),
}
_SINIF_ALANLARI = [(s, a) for s, d in _SINIF_GECERLI.items() for a in d if a != "symbols"]


@pytest.mark.parametrize(("sinif", "alan"), _SINIF_ALANLARI, ids=[f"{s.__name__}.{a}" for s, a in _SINIF_ALANLARI])
@pytest.mark.parametrize("deger", [
    pytest.param(True, id="True"),
    pytest.param(np.True_, id="np.True_"),
    pytest.param(np.array(True), id="0d-bool-dizi"),
    pytest.param(np.array(True, dtype=object), id="0d-nesne-bool"),
])
def test_B3_SINIF_DOGRUDAN_kurulunca_BOOL_InvalidInputError_ALANI_ADIYLA(sinif, alan, deger):
    """Kuvvet alani ve termostat siniflari kamu API'sidir (`observable/bridge_obs.py`
    `Morse`u dogrudan kurar): kapi fabrikada (`_senaryo_nesnesi`) ve SINIFTA.

    HEAD'de: 43 hucre kabul (deger 1 diye kullanilir); LJ `box_length` YANLIS nedenle
    (L/2 kapisi), NH `chain_length` / Langevin `seed` (numpy bool) ham `TypeError`.
    KIRMA (mut.py): herhangi bir sinifin `_bool_kapisi` cagrisi kaldirilinca kirmizi.
    """
    with pytest.raises(InvalidInputError, match=rf"{sinif.__name__}\.{alan} sayı olmalı"):
        sinif(**(_SINIF_GECERLI[sinif] | {alan: deger}))


@pytest.mark.parametrize(("sinif", "alan"), _SINIF_ALANLARI, ids=[f"{s.__name__}.{a}" for s, a in _SINIF_ALANLARI])
@pytest.mark.parametrize("deger", [
    pytest.param("0.5", id="dize"),
    pytest.param(b"0.5", id="bytes"),
    pytest.param(np.array("0.5"), id="0d-U-dizi"),
    pytest.param(np.array(b"0.5"), id="0d-S-dizi"),
    pytest.param(np.array("0.5", dtype=object), id="0d-nesne-dize"),
])
def test_V13_SINIF_DOGRUDAN_kurulunca_DIZE_InvalidInputError_ALANI_ADIYLA(sinif, alan, deger):
    """V13 (grup 9 duzeltici, 23.10.2026): B3'un kapsami bool/karmasikti. OLCULDU (HEAD 448f808
    ve duzeltmeden onceki agac ayni): `Morse(d_e='0.17')`, `LennardJones(cutoff='4')`,
    `Langevin(friction='2e-5')`, `NoseHoover(relaxation_time='100')` ham `TypeError` ("ufunc
    'isfinite' not supported"), `QLabError` DEGIL. Artik adli `InvalidInputError`, alanin adiyla.
    KIRMA: `_bool_kapisi`ndaki dize dali kaldirilinca kirmizi."""
    with pytest.raises(InvalidInputError, match=rf"{sinif.__name__}\.{alan} sayı olmalı: dize"):
        sinif(**(_SINIF_GECERLI[sinif] | {alan: deger}))


def test_V13_ic_ice_DIZE_merkez_SESSIZCE_SAYI_OLMAZ():
    """`Harmonic(center=[['0', 0, 0]])` duzeltmeden once HATASIZ kuruluyordu: `np.asarray(...,
    dtype=float)` dizeyi 0.0 bohr yapiyordu (olculdu 23.10.2026, HEAD'de de)."""
    with pytest.raises(InvalidInputError, match=r"Harmonic\.center sayı olmalı: dize"):
        Harmonic(k=0.5, center=[["0", 0, 0]])
    with pytest.raises(InvalidInputError, match=r"Harmonic\.center sayı olmalı: dize"):
        Harmonic(k=0.5, center=np.array([[b"0", b"0", b"0"]]))


def test_B3_SINIF_KAPISI_ic_ice_BOOL_ve_KARMASIK_ve_GECERLI_deger_AYNEN():
    """`Harmonic(center=[[True, 0, 0]])` HEAD'de 1.0 bohr'luk merkez kuruyordu; karmasik da
    reddedilir. Asiri reddetmeye karsi: gecerli numpy sayilari kabul ve SAKLANAN deger
    verilenin KENDISI (kapi donusturmez -> sevk ciktilari bit bit ayni)."""
    with pytest.raises(InvalidInputError, match=r"Harmonic\.center sayı olmalı"):
        Harmonic(k=0.5, center=[[True, 0, 0]])
    with pytest.raises(InvalidInputError, match=r"Morse\.d_e sayı olmalı"):
        Morse(d_e=np.complex128(0.17), r_e=1.4, a=1.0)
    for sinif, gecerli in _SINIF_GECERLI.items():
        nesne = sinif(**gecerli)
        for alan, deger in gecerli.items():
            if alan not in ("center", "symbols"):
                assert getattr(nesne, alan) is deger, (sinif.__name__, alan)
    assert Morse(d_e=np.float64(0.17), r_e=np.int64(2), a=np.array(1.0)).r_e == 2
