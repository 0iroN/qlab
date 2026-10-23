"""Spin Python yolu: düzensiz (ragged) liste ham ValueError değil, adlı InvalidInputError.

Grup 6 doğanı (14.10.2026'da kapandı). HEAD'de (448f808) ölçüldü:
`SpinScenario`ya doğrudan `coupling=[1, [2, 3]]` gibi iç içe, boyları tutmayan
bir liste verilince `_skaler` içindeki `np.ndim` dizi KURAMIYOR ve
`build_spin` / `provenance_for` ham `ValueError` ("inhomogeneous shape")
fırlatıyordu — dört alanda, iki girişte 8/8, `QLabError` DEĞİL. YAML yolu
aynı girdiyi zaten `ScenarioError` ile adıyla reddediyordu (`_yaml_sayisi`).

SINIF: GİRDİ KAPISI kilidi (fizik değil). Kapının dili bu yolun mevcut dili
`InvalidInputError`dır ve mesaj ALANI adlandırır.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pytest

from qlab.engine.spin import factory as fabrika
from qlab.engine.spin.factory import build_spin, provenance_for
from qlab.errors import InvalidInputError
from qlab.scenario.schema import SpinScenario

_TABAN = SpinScenario(
    name="duzensiz liste kapisi",
    model="heisenberg",
    algorithm="metropolis",
    lattice_size=4,
    temperatures=(3.0, 4.0),
    n_sweep=20,
    n_burn=0,
    seed=20261014,
)


@pytest.mark.parametrize(
    "alanlar,ad",
    [
        (dict(coupling=[1, [2, 3]]), "coupling"),
        (dict(field=[0.1, [0.2]]), "field"),
        (dict(heisenberg_step=[[0.5], 0.6, [0.7, 0.8]]), "heisenberg_step"),
        (dict(temperatures=(3.0, [1.0, [3.0]])), "temperatures"),
    ],
)
def test_DUZENSIZ_LISTE_adli_InvalidInputError_HESAPTAN_ONCE(monkeypatch, alanlar, ad):
    """Her iki giriş (`build_spin`, `provenance_for`) aynı kapıdan geçer; hata
    `InvalidInputError`dır (ham `ValueError` DEĞİL) ve mesaj alanın adını ve
    'duzensiz' kelimesini taşır. Kapı sweep'lerden ÖNCE durur."""
    monkeypatch.setattr(fabrika, "run_metropolis", lambda *a, **k: pytest.fail("kapi hesaptan sonra"))
    senaryo = dataclasses.replace(_TABAN, **alanlar)
    desen = re.escape(f"spin senaryosu: {ad} tek bir gercel sayi olmali, verilen duzensiz")
    with pytest.raises(InvalidInputError, match=desen):
        build_spin(senaryo)
    with pytest.raises(InvalidInputError, match=desen):
        provenance_for(senaryo, plateau_reached=np.ones(2, bool), tau_int=np.ones(2))


def _ic_ice_nesne(oge):
    """Öğesi yine bir 0 boyutlu nesne DİZİSİ olan 0 boyutlu nesne dizisi (`np.array(x,
    dtype=object)` iç diziyi açar; sarmalamak için `np.empty` + atama gerekir)."""
    dis = np.empty((), dtype=object)
    dis[()] = oge
    return dis


@pytest.mark.parametrize("alan", ["coupling", "field", "heisenberg_step", "temperatures"])
@pytest.mark.parametrize(
    "uret",
    [
        pytest.param(lambda: np.array(True, dtype=object), id="0d-nesne-True"),
        pytest.param(lambda: np.array("2.5", dtype=object), id="0d-nesne-dize"),
        pytest.param(lambda: _ic_ice_nesne(np.array(True, dtype=object)), id="0d-nesne-ic-ice-nesne"),
        pytest.param(lambda: np.array("2.5"), id="0d-U-dizi"),
        pytest.param(lambda: np.array(b"2.5"), id="0d-S-dizi"),
    ],
)
def test_V6_V7_0d_NESNE_ve_DIZE_dizisi_adli_InvalidInputError(monkeypatch, alan, uret):
    """V6 + V7 (grup 9 düzeltici, 23.10.2026). ÖLÇÜLDÜ (düzeltmeden önce, Python yolu):
    `_skaler` nesne-True'yu 1.0, nesne-'2.5'i ve 'U'/'S' '2.5'i 2.5 okuyordu;
    `build_spin(coupling=np.array(True, dtype=object))` KOŞUYORDU, 'U'/'S' dizide ham
    `UFuncTypeError`, `provenance_for` ham `ValueError`. B4 (grup 9) nesne dizisini öteki
    dört skaler kapıda kapatmıştı, spin'de açık kalmıştı. Kapı sweep'lerden ÖNCE durur."""
    monkeypatch.setattr(fabrika, "run_metropolis", lambda *a, **k: pytest.fail("kapi hesaptan sonra"))
    deger = (3.0, uret()) if alan == "temperatures" else uret()
    senaryo = dataclasses.replace(_TABAN, **{alan: deger})
    desen = re.escape(f"spin senaryosu: {alan} tek bir gercel sayi olmali (bool, dizge")
    with pytest.raises(InvalidInputError, match=desen):
        build_spin(senaryo)
    with pytest.raises(InvalidInputError, match=desen):
        provenance_for(senaryo, plateau_reached=np.ones(2, bool), tau_int=np.ones(2))


def test_V7_0d_NESNE_GERCEL_dizi_ve_BUYUK_TAMSAYI_asiri_reddedilmez():
    """Aşırı reddetmeye karşı (`mermin._skaler` ile aynı kural): 0 boyutlu GERÇEL nesne
    dizisi açılıp sayısı döner; float'a sığmayan tamsayı (tür 'O' ama dizi DEĞİL) eski
    mesajını korur."""
    assert fabrika._skaler("coupling", np.array(2.0, dtype=object)) == 2.0
    assert fabrika._skaler("coupling", np.array(np.int64(3), dtype=object)) == 3.0
    with pytest.raises(InvalidInputError, match="coupling sayiya cevrilemiyor"):
        fabrika._skaler("coupling", 10**400)

