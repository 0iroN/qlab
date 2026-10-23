"""C6 (11.10.2026): MD senaryo girdileri HAM hata veriyordu -- artik `ScenarioError`.

OLCULDU (HEAD 80eb21c, `python -m qlab.run`, sevk edilen `h2_morse.yaml` /
`argon_nvt.yaml`'in tek satiri bozularak): yanlis yazilmis anahtar
(`de:` -> `TypeError: Morse.__init__() got an unexpected keyword argument
'de'`; LJ'de `sigmaa`, Langevin'de `surtunme`, harmonikte `kk` -> `KeyError:
'k'`), sayi yerine dize (`positions_bohr`daki `"np.float64(0.0)"`,
`masses_amu: [abc, ...]`, `velocities`, `d_e: abc` -> numpy `isfinite`
TypeError'i, `n_particles: yuz`, `dt: abc`, `n_steps: dort`), eksik
`masses_amu` (`KeyError`) ve sozluk yerine liste (`initial: [1, 2]`,
`params: [1, 2]`) traceback'le dusuyordu; `n_steps: 40.7` ise SESSIZCE 40'a
kesilip rc = 0 veriyordu. Katman: YAML'in kendi alanlari yukleyicide
(`_load_md_scenario`), kuvvet alani / termostat / `initial` icerigi
`md/factory.build`de -- `build` `MDScenario(...)` dogrudan kurulunca da tek
bekcidir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

import qlab.run as run_modulu
from qlab.engine.md.factory import build
from qlab.errors import ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import MDScenario

from .h2_morse_sabitleri import H2_A, H2_DE, H2_RE

_SEVK = Path(__file__).resolve().parents[2] / "scenarios"
_MORSE = (_SEVK / "h2_morse.yaml").read_text(encoding="utf-8")
_ARGON = (_SEVK / "argon_nvt.yaml").read_text(encoding="utf-8")
_PARAMS = "params: {d_e: 0.174434333, r_e: 1.401118538, a: 1.028997701}"
_KONUM = "positions_bohr: [[0.0, 0.0, 0.0], [0.0, 0.0, 1.701118538]]"
_KUTLE = "  masses_amu: [1.00782503207, 1.00782503207]\n"

#: (ad, kaynak, eski, yeni, mesajda beklenen parca)
_BOZUK = [
    ("morse_anahtar", _MORSE, "d_e: 0.174434333", "de: 0.174434333", "unexpected keyword argument 'de'"),
    # V13 (grup 9 duzeltici): eskiden siniftan ham TypeError ("morse parametreleri gecersiz:
    # ufunc 'isfinite' ..."); artik bool'unki gibi anahtarin adiyla.
    ("morse_dize", _MORSE, "d_e: 0.174434333", "d_e: abc", "morse parametresi `d_e`: dize"),
    ("lj_anahtar", _MORSE, "kind: morse\n  " + _PARAMS,
     "kind: lennard_jones\n  params: {epsilon: 0.1, sigmaa: 2.0}", "unexpected keyword argument 'sigmaa'"),
    ("harmonik_anahtar", _MORSE, "kind: morse\n  " + _PARAMS, "kind: harmonic\n  params: {kk: 0.1}",
     "unexpected keyword argument 'kk'"),
    # Konumlu dalda `sigma` harmonik icin bilinmeyen anahtardir (yalniz fcc onu kafese harcar).
    ("harmonik_sigma_konumlu", _MORSE, "kind: morse\n  " + _PARAMS, "kind: harmonic\n  params: {k: 0.1, sigma: 2.0}",
     "unexpected keyword argument 'sigma'"),
    ("morse_dizi", _MORSE, "d_e: 0.174434333", "d_e: [1.0, 2.0]", "morse parametreleri"),
    ("konum_dize", _MORSE, _KONUM, _KONUM.replace("[[0.0,", '[["np.float64(0.0)",'), "initial.positions_bohr"),
    ("kutle_dize", _MORSE, _KUTLE, "  masses_amu: [abc, 1.0]\n", "initial.masses_amu"),
    ("hiz_dize", _MORSE, _KUTLE, "  velocities: [[x, 0, 0], [0, 0, 0]]\n" + _KUTLE, "initial.velocities"),
    ("kutle_yok", _MORSE, _KUTLE, "", "masses_amu"),
    ("initial_liste", _MORSE, "initial:\n  " + _KONUM + "\n" + _KUTLE, "initial: [1, 2]\n", "'initial'"),
    ("params_liste", _MORSE, _PARAMS, "params: [1, 2]", "'force_field.params'"),
    ("dt_dize", _MORSE, "dt: 0.783307192", "dt: abc", "'dt'"),
    ("n_steps_dize", _MORSE, "n_steps: 4000", "n_steps: dort", "'n_steps'"),
    ("n_steps_kesir", _MORSE, "n_steps: 4000", "n_steps: 40.7", "'n_steps' tam"),
    ("fcc_dize", _ARGON, "n_particles: 108", "n_particles: yuz", "n_particles"),
    ("termostat_anahtar", _ARGON, "friction", "surtunme", "unexpected keyword argument 'surtunme'"),
]


@pytest.mark.parametrize(("ad", "kaynak", "eski", "yeni", "parca"), _BOZUK, ids=[b[0] for b in _BOZUK])
def test_C6_bozuk_MD_girdisi_KAMU_YOLUNDA_ScenarioError(tmp_path, capsys, ad, kaynak, eski, yeni, parca):
    """`run.main` rc = 1 ve nedeni soyler; ham istisna yukari SIZMAZ, .npz yazilmaz."""
    assert kaynak.count(eski) == 1, ad
    yol = tmp_path / f"{ad}.yaml"
    yol.write_text(kaynak.replace(eski, yeni), encoding="utf-8")
    cikti = tmp_path / f"{ad}.npz"
    assert run_modulu.main([str(yol), "-o", str(cikti)]) == 1
    hata = capsys.readouterr().err
    assert parca in hata, hata
    assert not cikti.exists()


def test_C6_build_DOGRUDAN_kurulan_senaryoda_da_ScenarioError():
    """YAML kapisi olmadan (`MDScenario(...)`) `build` tek bekcidir."""
    taban = dict(
        name="h2",
        initial={"positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]], "masses_amu": [1.0078, 1.0078]},
        thermostat=None, dt=0.78, n_steps=10, save_every=1,
    )
    with pytest.raises(ScenarioError, match="unexpected keyword argument 'de'"):
        build(MDScenario(force_field={"kind": "morse", "params": {"de": H2_DE, "r_e": H2_RE, "a": H2_A}}, **taban))
    with pytest.raises(ScenarioError, match="initial.positions_bohr"):
        build(MDScenario(
            force_field={"kind": "morse", "params": {"d_e": H2_DE, "r_e": H2_RE, "a": H2_A}},
            **{**taban, "initial": {"positions_bohr": [["np.float64(0.0)", 0, 0], [0, 0, 1.7]],
                                    "masses_amu": [1.0078, 1.0078]}},
        ))


def test_C6_H2_sabitleri_TEK_kaynak_ve_SEVK_senaryosuyla_ayni():
    """`h2_morse_sabitleri` iki test dosyasinin ortak kaynagi; sevk YAML'i ayni sayilari tasir."""
    params = load_scenario(_SEVK / "h2_morse.yaml").force_field["params"]
    assert params == {"d_e": H2_DE, "r_e": H2_RE, "a": H2_A}


# --------------------------------------------- dogrulama turu (11.10.2026) --

#: harmonic + fcc: `sigma` kafesi kurar, yaya gitmez. HEAD 80eb21c'de rc = 0
#: kosuyordu; C6'nin ilk hali rc = 1 "unexpected keyword argument 'sigma'"
#: verip mesajda fabrikanin ekledigi `box_length`'i "Verilen" diye gosterdi.
_FCC_HARMONIK = """kind: md
name: t
force_field:
  kind: harmonic
  params: {k: 0.0001, sigma: 6.4345174543}
initial:
  lattice: fcc
  n_particles: 32
  density_reduced: 0.8442
  masses_amu: [39.948]
dt: 10.0
n_steps: 4
"""


def _kos(tmp_path, metin, ad="s"):
    yol = tmp_path / f"{ad}.yaml"
    yol.write_text(metin, encoding="utf-8")
    cikti = tmp_path / f"{ad}.npz"
    return run_modulu.main([str(yol), "-o", str(cikti)]), cikti


def test_C6_harmonik_FCC_k_sigma_KOSAR_ve_merkez_ORIJIN(tmp_path, capsys):
    """Gecerli girdi: rc = 0, npz yazilir; yay merkezi verilmeyince orijin (HEAD ile ayni)."""
    rc, cikti = _kos(tmp_path, _FCC_HARMONIK)
    assert rc == 0, capsys.readouterr().err
    assert cikti.exists()
    durum, alan, termostat = build(load_scenario(tmp_path / "s.yaml"))
    assert durum.positions.shape == (32, 3) and termostat is None
    assert alan.center.shape == (32, 3) and not alan.center.any()


@pytest.mark.parametrize(
    ("ad", "metin", "parca"),
    [
        ("harmonik_kk", _FCC_HARMONIK.replace("sigma: 6.4345174543}", "sigma: 6.4345174543, kk: 3}"),
         "unexpected keyword argument 'kk'"),
        ("lj_cutof", _FCC_HARMONIK.replace("kind: harmonic\n  params: {k: 0.0001,",
                                           "kind: lennard_jones\n  params: {epsilon: 3.79e-04, cutof: 10.0,"),
         "unexpected keyword argument 'cutof'"),
        # B5 (grup 9): HF (AIMD) ve Morse dallari da yalniz `yaml_p`yi gosterir; ikisi de
        # HEAD 448f808'de TESTSIZDI (mut.py: mesaja `p`yi veren mutant md hizli seridini geciyordu).
        ("hf_sigma", _FCC_HARMONIK.replace("kind: harmonic\n  params: {k: 0.0001,",
                                           "kind: hartree_fock\n  params: {symbols: [H, H],"),
         "unexpected keyword argument 'sigma'"),
        ("morse_sigma", _FCC_HARMONIK.replace("kind: harmonic\n  params: {k: 0.0001,",
                                              "kind: morse\n  params: {d_e: 0.17, r_e: 1.4, a: 1.0,"),
         "unexpected keyword argument 'sigma'"),
    ],
)
def test_C6_FCC_bilinmeyen_anahtar_mesaji_YALNIZ_kullanicinin_anahtarlari(tmp_path, capsys, ad, metin, parca):
    """fcc dalinin ekledigi `box_length` "Verilen" listesinde GORUNMEZ (B5: HF ve Morse dallari da)."""
    rc, cikti = _kos(tmp_path, metin, ad)
    hata = capsys.readouterr().err
    assert rc == 1 and parca in hata, hata
    assert "Verilen: {" in hata and "box_length" not in hata, hata
    assert not cikti.exists()


def test_C6_tb_ss_sigma_FCC_bilinmeyen_listesinde_box_length_YOK():
    """`_tb_ss_sigma_alani` kullanicinin sozlugunu gorur; fcc'nin ekledigi anahtari degil."""
    senaryo = MDScenario(
        name="t", force_field={"kind": "tb_ss_sigma", "params": {"sigma": 6.43}},
        initial={"lattice": "fcc", "n_particles": 4, "density_reduced": 0.8442, "masses_amu": [1.0]},
        thermostat=None, dt=1.0, n_steps=2,
    )
    with pytest.raises(ScenarioError, match=r"bilinmeyen \['sigma'\]") as hata:
        build(senaryo)
    assert "box_length" not in str(hata.value)


def test_C6_harmonik_YAML_center_KULLANILIR(tmp_path):
    """YAML'in `center`i varsayilan orijinin USTUNE yazilir (tersi olsa sessizce yutulurdu)."""
    yol = tmp_path / "merkez.yaml"
    yol.write_text(_MORSE.replace(
        "kind: morse\n  " + _PARAMS, "kind: harmonic\n  params: {k: 0.1, center: [[0, 0, 5.0], [0, 0, 5.0]]}"
    ), encoding="utf-8")
    _, alan, _ = build(load_scenario(yol))
    assert alan.center.tolist() == [[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]]


def test_C6_FCC_eksik_density_reduced_ScenarioError(tmp_path, capsys):
    """`density_reduced` yok -> `KeyError` yakalanir, nedeni soyleyen `ScenarioError`."""
    rc, cikti = _kos(tmp_path, _ARGON.replace("  density_reduced: 0.8442\n", ""), "fcc_eksik")
    hata = capsys.readouterr().err
    assert rc == 1 and "density_reduced" in hata and "Traceback" not in hata, hata
    assert not cikti.exists()


@pytest.mark.parametrize(
    ("ad", "eski", "yeni", "parca"),
    [
        # save_every yukleyicide tam sayi kapisindan gecer (eskiden int(2.5) = 2).
        ("save_every_kesir", "save_every: 50", "save_every: 2.5", "'save_every' tam"),
        # thermostat.params sozluk olmali (liste ham TypeError'du).
        ("termostat_params_liste",
         "params: {target_temperature_kelvin: 87.2, friction: 2.2434e-05, seed: 2026}",
         "params: [1, 2]", "'thermostat.params'"),
    ],
)
def test_C6_yukleyici_save_every_ve_termostat_params(tmp_path, ad, eski, yeni, parca):
    assert _ARGON.count(eski) == 1
    yol = tmp_path / f"{ad}.yaml"
    yol.write_text(_ARGON.replace(eski, yeni), encoding="utf-8")
    with pytest.raises(ScenarioError, match=parca):
        load_scenario(yol)


# ------------------------------------- B1 (grup 9): `initial`de TIRNAKLI sayilar --

_TB = (_SEVK / "tb_zincir_md.yaml").read_text(encoding="utf-8")
_ARGON_KISA = _ARGON.replace("n_steps: 20000\nsave_every: 50", "n_steps: 4\nsave_every: 1")

#: (ad, kaynak, eski, yeni, mesajda beklenen anahtar). OLCULDU (HEAD 448f808,
#: `python -m qlab.run`): sekiz hucrenin sekizi rc = 0, sayilar tirnaksizla bit bit
#: ayni (`np.asarray(..., dtype=float)` dizeyi sessizce sayi yapiyordu).
_TIRNAKLI = [
    ("konum", _MORSE, _KONUM, _KONUM.replace("1.701118538]]", "'1.701118538']]"), "initial.positions_bohr"),
    ("konum_hepsi", _MORSE, _KONUM, "positions_bohr: [['0', '0', '0'], ['0', '0', '1.701118538']]",
     "initial.positions_bohr"),
    ("kutle", _MORSE, _KUTLE, "  masses_amu: ['1.00782503207', \"1.00782503207\"]\n", "initial.masses_amu"),
    ("kutle_skaler", _MORSE, _KUTLE, "  masses_amu: '1.00782503207'\n", "initial.masses_amu"),
    ("hiz", _MORSE, _KUTLE, _KUTLE + "  velocities: [['0', '0', '0'], ['0', '0', '0.0']]\n", "initial.velocities"),
    ("fcc_yogunluk", _ARGON_KISA, "density_reduced: 0.8442", "density_reduced: '0.8442'", "initial.density_reduced"),
    ("fcc_kutle", _ARGON_KISA, "masses_amu: [39.948]", "masses_amu: ['39.948']", "initial.masses_amu"),
    ("tb_konum", _TB, "- [0.10204595606925913, 0.0, 0.0]", "- ['0.10204595606925913', 0.0, 0.0]",
     "initial.positions_bohr"),
    # Tirnakli ama tirnaksiz yazilsa PyYAML'in SAYI yapacagi ustel bicim: tirnak bilinclidir.
    ("konum_ustel_tirnakli", _MORSE, _KONUM, _KONUM.replace("1.701118538]]", "'1.701118538e+0']]"),
     "initial.positions_bohr"),
]


@pytest.mark.parametrize(("ad", "kaynak", "eski", "yeni", "anahtar"), _TIRNAKLI, ids=[t[0] for t in _TIRNAKLI])
def test_B1_initial_TIRNAKLI_SAYI_KAMU_YOLUNDA_rc1_ANAHTARI_ADIYLA(tmp_path, capsys, ad, kaynak, eski, yeni,
                                                                   anahtar):
    """rc = 1, anahtar adiyla, traceback ve .npz yok (C6 / grup 6 `initial true` emsali)."""
    assert kaynak.count(eski) == 1, ad
    yol = tmp_path / f"{ad}.yaml"
    yol.write_text(kaynak.replace(eski, yeni), encoding="utf-8")
    cikti = tmp_path / f"{ad}.npz"
    assert run_modulu.main([str(yol), "-o", str(cikti)]) == 1
    hata = capsys.readouterr().err
    assert anahtar in hata and "Traceback" not in hata, hata
    assert not cikti.exists()


@pytest.mark.parametrize(
    ("ad", "kaynak", "eski", "ustel", "ondalik"),
    [
        ("konum", _MORSE, "1.701118538]]", "1.701118538e0]]", "1.701118538]]"),
        ("kutle", _MORSE, "masses_amu: [1.00782503207,", "masses_amu: [1.00782503207e0,",
         "masses_amu: [1.00782503207,"),
        ("hiz", _MORSE, _KUTLE, _KUTLE + "  velocities: [[0, 0, 0], [0, 0, 1e-3]]\n",
         _KUTLE + "  velocities: [[0, 0, 0], [0, 0, 0.001]]\n"),
        ("fcc_yogunluk", _ARGON_KISA, "density_reduced: 0.8442", "density_reduced: 8442e-4",
         "density_reduced: 0.8442"),
    ],
)
def test_B1_TIRNAKSIZ_USTEL_yazim_SAYI_ve_KURULUM_AYNI(tmp_path, ad, kaynak, eski, ustel, ondalik):
    """Asiri reddetmeye karsi: PyYAML'in TIRNAKSIZ da DIZE dondurdugu ustel yazim (`1e-3`,
    `1.701118538e0`, `8442e-4`) sayidir (`_yaml_sayisi` istisnasi) ve kurulum ondalik yazimla
    BIT BIT ayni. KIRMA: yukleyici ustel istisnayi tanimazsa ya da dizeyi cevirmeden
    birakirsa (fabrikanin dize kapisi onu reddeder) -> ScenarioError, kirmizi.
    """
    assert kaynak.count(eski) == 1, ad
    assert any(isinstance(x, str) for x in _yapraklar(yaml.safe_load(kaynak.replace(eski, ustel))["initial"]))
    kurulumlar = []
    for i, yeni in enumerate((ustel, ondalik)):
        yol = tmp_path / f"{ad}{i}.yaml"
        yol.write_text(kaynak.replace(eski, yeni), encoding="utf-8")
        durum, alan, _ = build(load_scenario(yol))
        kurulumlar.append((durum.positions, durum.velocities, durum.masses, getattr(alan, "box_length", None)))
    (p0, v0, m0, l0), (p1, v1, m1, l1) = kurulumlar
    assert np.array_equal(p0, p1) and np.array_equal(v0, v1) and np.array_equal(m0, m1) and l0 == l1
    assert ad != "hiz" or v0.any()


def _yapraklar(deger):
    if isinstance(deger, dict):
        for x in deger.values():
            yield from _yapraklar(x)
    elif isinstance(deger, list):
        for x in deger:
            yield from _yapraklar(x)
    else:
        yield deger


@pytest.mark.parametrize(("anahtar", "deger"), [
    ("positions_bohr", [[0.0, 0.0, 0.0], [0.0, 0.0, "1.7"]]),
    ("positions_bohr", np.array([["0", "0", "0"], ["0", "0", "1.7"]])),
    ("masses_amu", ["1.0078", 1.0078]),
    ("velocities", [[0, 0, 0], [0, 0, np.str_("0.001")]]),
    ("density_reduced", "0.8442"),
    ("positions_bohr", [[0.0, 0.0, 0.0], [0.0, 0.0, b"1.7"]]),
    ("positions_bohr", np.array([[b"0", b"0", b"0"], [b"0", b"0", b"1.7"]])),
    ("density_reduced", b"0.8442"),
], ids=["konum-liste", "konum-U-dizi", "kutle", "hiz-np.str_", "fcc-yogunluk",
        "konum-bytes", "konum-S-dizi", "fcc-yogunluk-bytes"])
def test_B1_DOGRUDAN_senaryoda_DIZE_ScenarioError(anahtar, deger):
    """YAML kapisi olmadan (`MDScenario(...)`) fabrika dizeyi kabul etmez -- her dize oraya
    dogrudan kurulumdan ulasir (YAML'in ustel istisnasini yukleyici sayiya cevirir).

    V8 (grup 9 duzeltici, 23.10.2026): bytes oge ve 'S' turlu dizi hucreleri. Kapinin
    bytes / 'S' kolu kilitsizdi: `(str, bytes)` -> `str` ve `"SU"` -> `"U"` mutanti md + tb
    + scenario hizli seridinin 607 testinden geciyordu (`float(b'1.7')` 1.7 verir)."""
    if anahtar == "density_reduced":
        ilk = {"lattice": "fcc", "n_particles": 4, "density_reduced": deger, "masses_amu": [39.948]}
        ff = {"kind": "lennard_jones", "params": {"epsilon": 3.79e-4, "sigma": 6.43, "cutoff": 4.0}}
    else:
        ilk = {"positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]], "masses_amu": [1.0078, 1.0078]}
        ilk[anahtar] = deger
        ff = {"kind": "morse", "params": {"d_e": H2_DE, "r_e": H2_RE, "a": H2_A}}
    senaryo = MDScenario(name="t", force_field=ff, initial=ilk, thermostat=None, dt=0.78, n_steps=2)
    with pytest.raises(ScenarioError, match=anahtar):
        build(senaryo)


# ---------------------------------------------- B2 (grup 9): fcc boyut tavani --


def test_B2_FCC_TAVANI_SINIRDA_KABUL_USTUNDE_ScenarioError(tmp_path):
    """Tavan (2048 = 4 x 8^3) KURULUR, bir sonraki fcc sayisi (2916 = 4 x 9^3) REDDEDILIR;
    kafes kurulmadan once (build 2048'de ~4 ms; kuvvet cagrilmaz).

    KIRMA: tavan yok / tavan 4096 -> 2916 kurulur; `>` yerine `>=` -> 2048 reddedilir.
    """
    def senaryo(n):
        yol = tmp_path / f"a{n}.yaml"
        yol.write_text(_ARGON_KISA.replace("n_particles: 108", f"n_particles: {n}"), encoding="utf-8")
        return load_scenario(yol)

    durum, alan, _ = build(senaryo(2048))
    assert durum.positions.shape == (2048, 3) and alan.box_length > 2.0 * alan.cutoff
    with pytest.raises(ScenarioError, match="n_particles = 2916 fcc boyut tavanını"):
        build(senaryo(2916))


def test_B2_FCC_DEV_SAYI_KAMU_YOLUNDA_ScenarioError_ham_MemoryError_DEGIL(tmp_path, capsys):
    """N = 4 x 60^3 = 864000: HEAD 448f808'de LJ kuvveti 16.3 TiB isteyip ham
    `_ArrayMemoryError` traceback'i veriyordu (olculdu, 0.9 s). Artik rc = 1, bellek
    kestirimiyle; .npz yok."""
    rc, cikti = _kos(tmp_path, _ARGON_KISA.replace("n_particles: 108", "n_particles: 864000"), "dev")
    hata = capsys.readouterr().err
    assert rc == 1 and "fcc boyut tavanını" in hata and "GB" in hata and "Traceback" not in hata, hata
    assert not cikti.exists()
