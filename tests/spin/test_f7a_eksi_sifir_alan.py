"""F7-a (11.10.2026): `field: -0.0` panelin Yontem satirinda "h = -0" basmaz.

OLCULDU (HEAD 80eb21c, kamu CLI): L = 4, `field: -0.0` -> "Yontem : ... J = 1
(girdi), h = -0; kT/J = T/J: 2". Yukleyici -0.0'i normallestirmiyor ve
`provenance_for` `f"{field:g}"` basiyordu. Duzeltme bicimdedir (`+ 0.0`), yani
YAML yolu da Python yolu da (`SpinScenario` dogrudan) ayni satiri verir.
SAYILAR DEGISMEZ: -0.0 ile kosumun npz'si +0.0 ile kosumunkiyle bit bit ayni
(asagida kilitli) -- duzeltme yalniz metindir.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from qlab.engine.spin.factory import provenance_for
from qlab.run import main
from qlab.scenario.loader import load_scenario

YAML = (
    "kind: spin\nname: eksi sifir\nmodel: ising\nalgorithm: metropolis\n"
    "lattice_size: 4\ntemperatures: [2.0]\nn_sweep: 50\nn_burn: 10\nseed: 1\n"
    "coupling: 1.0\nfield: {alan}\n"
)


def _yontem(cikti: str) -> str:
    return next(s for s in cikti.splitlines() if s.startswith("Yöntem"))


def test_KAMU_YOLU_eksi_sifir_h_eq_0_basar_ve_SAYILAR_ayni(tmp_path, capsys):
    satirlar, npz = {}, {}
    for ad, alan in (("eksi", "-0.0"), ("arti", "0.0")):
        yol = tmp_path / f"{ad}.yaml"
        yol.write_text(YAML.format(alan=alan), encoding="utf-8")
        assert main([str(yol), "-o", str(tmp_path / f"{ad}.npz")]) == 0
        satirlar[ad] = _yontem(capsys.readouterr().out)
        with np.load(tmp_path / f"{ad}.npz", allow_pickle=False) as veri:
            npz[ad] = {k: veri[k] for k in veri.files}
    assert "h = -0" not in satirlar["eksi"]
    assert "h = 0; kT/J" in satirlar["eksi"]
    assert satirlar["eksi"] == satirlar["arti"]
    assert npz["eksi"].keys() == npz["arti"].keys()
    for k in npz["arti"]:
        if k != "provenance_json":
            assert np.array_equal(npz["eksi"][k], npz["arti"][k], equal_nan=True), k


def test_PYTHON_YOLU_ve_SIFIR_DISI_alan_DEGISMEDI(tmp_path):
    yol = tmp_path / "s.yaml"
    yol.write_text(YAML.format(alan="0.0"), encoding="utf-8")
    temel = load_scenario(str(yol))
    tani = dict(plateau_reached=np.array([True]), tau_int=np.array([1.0]))

    def yontem(alan: float) -> str:
        return provenance_for(dataclasses.replace(temel, field=alan), **tani).method

    assert "h = 0;" in yontem(-0.0) and "h = -0" not in yontem(-0.0)
    for alan, metin in ((-0.1, "h = -0.1;"), (0.25, "h = 0.25;"), (-5e-324, "h = -4.94066e-324;")):
        assert metin in yontem(alan), alan
