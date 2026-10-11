"""LOADER-VE (11.10.2026): TB yukleyicisi sayi olmayan girdide HAM hata veriyordu.

OLCULDU (HEAD 80eb21c, `python -m qlab.run`, sevk edilen `zincir_tb.yaml`in
tek satiri bozularak): `lattice_vectors_bohr: [[abc, ...]]`, `basis_bohr`,
`onsite_ev: [foo]`, `n_electron_per_cell: bir`, `k_points`, `t_ev`, `i` ham
`ValueError`; liste yerine sayi (`lattice_vectors_bohr: 2.5`, `hoppings: 5`)
ve `onsite_ev: [null]` ham `TypeError` (rc = 1, traceback). Ayni fonksiyonda
SESSIZ donusumler de vardi ve rc = 0 ile panel basiliyordu:
`n_electron_per_cell: 1.5` -> 1 (yari dolu bant hukmu), `k_points: 100.7`
-> 100, `lattice_vectors_bohr: [[true, ...]]` -> 1.0, tirnakli `'-5.3'` ->
-5.3. Kapilar spin yukleyicisinin `_yaml_sayisi` / `_yaml_tam_sayisi`si.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import qlab.run as run_modulu

_SEVK = Path(__file__).resolve().parents[2] / "scenarios" / "zincir_tb.yaml"
_ZINCIR = _SEVK.read_text(encoding="utf-8")
_HOP = "  - {i: 0, j: 0, cell: [1, 0, 0], t_ev: -1.7}\n  - {i: 0, j: 0, cell: [-1, 0, 0], t_ev: -1.7}\n"

#: (ad, eski, yeni, mesajda beklenen parca)
_BOZUK = [
    ("orgu_dize", "[[2.5, 0.0, 0.0]]", "[[abc, 0.0, 0.0]]", "'lattice_vectors_bohr[0][0]'"),
    ("orgu_skaler", "lattice_vectors_bohr: [[2.5, 0.0, 0.0]]", "lattice_vectors_bohr: 2.5", "bir liste"),
    ("orgu_bool", "[[2.5, 0.0, 0.0]]", "[[true, 0.0, 0.0]]", "(bool)"),
    ("taban_dize", "basis_bohr: [[0.0, 0.0, 0.0]]", "basis_bohr: [[x, 0.0, 0.0]]", "'basis_bohr[0][0]'"),
    ("onsite_dize", "onsite_ev: [-5.3]", "onsite_ev: [foo]", "'onsite_ev[0]'"),
    ("onsite_null", "onsite_ev: [-5.3]", "onsite_ev: [null]", "'onsite_ev[0]'"),
    ("onsite_tirnak", "onsite_ev: [-5.3]", "onsite_ev: ['-5.3']", "'onsite_ev[0]'"),
    ("elektron_dize", "n_electron_per_cell: 1", "n_electron_per_cell: bir", "'n_electron_per_cell'"),
    ("elektron_kesir", "n_electron_per_cell: 1", "n_electron_per_cell: 1.5", "'n_electron_per_cell' tam"),
    ("k_dize", "k_points: 100", "k_points: yuz", "'k_points'"),
    ("k_kesir", "k_points: 100", "k_points: 100.7", "'k_points' tam"),
    ("t_dize", "t_ev: -1.7}\n  - {i: 0, j: 0, cell: [-1", "t_ev: eksi}\n  - {i: 0, j: 0, cell: [-1",
     "'hoppings[0].t_ev'"),
    ("i_dize", "{i: 0, j: 0, cell: [1, 0, 0]", "{i: sifir, j: 0, cell: [1, 0, 0]", "'hoppings[0].i'"),
    # Kesirli indis SESSIZCE kesilmez (int(0.5) = 0 olurdu; dogrulama turu).
    ("i_kesir", "{i: 0, j: 0, cell: [1, 0, 0]", "{i: 0.5, j: 0, cell: [1, 0, 0]", "'hoppings[0].i' tam"),
    ("hoplama_skaler", "hoppings:\n" + _HOP, "hoppings: 5\n", "'hoppings' bir liste"),
]


@pytest.mark.parametrize(("ad", "eski", "yeni", "parca"), _BOZUK, ids=[b[0] for b in _BOZUK])
def test_LOADER_VE_bozuk_TB_girdisi_KAMU_YOLUNDA_ScenarioError(tmp_path, capsys, ad, eski, yeni, parca):
    """`run.main` rc = 1 ve alani adiyla soyler; ham istisna SIZMAZ, sessiz donusum yok."""
    assert _ZINCIR.count(eski) == 1, ad
    yol = tmp_path / f"{ad}.yaml"
    yol.write_text(_ZINCIR.replace(eski, yeni), encoding="utf-8")
    cikti = tmp_path / f"{ad}.npz"
    assert run_modulu.main([str(yol), "-o", str(cikti)]) == 1
    hata = capsys.readouterr().err
    assert parca in hata, hata
    assert not cikti.exists()


def test_LOADER_VE_tam_degerli_float_ve_ussel_yazim_KABUL(tmp_path):
    """Kapi daraltmaz: `1.0` tam sayi alanda, tirnaksiz `-1.7e0` sayi alaninda gecer."""
    from qlab.scenario.loader import load_scenario

    metin = _ZINCIR.replace("n_electron_per_cell: 1", "n_electron_per_cell: 1.0").replace(
        "onsite_ev: [-5.3]", "onsite_ev: [-5.3e0]"
    )
    yol = tmp_path / "gecerli.yaml"
    yol.write_text(metin, encoding="utf-8")
    senaryo = load_scenario(yol)
    assert senaryo.n_electron_per_cell == 1 and senaryo.onsite_ev == (-5.3,)
