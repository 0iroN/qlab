"""B7dog-3 / B7dog-4 (11.10.2026): indirgeme panelinin mesafe etiketi ve npz'nin
mesafe basina kararlilik hukmu -- kamu yolu (`qlab.run.main`, YAML'dan), SCF'SIZ.

`tb.factory.downfold_uhf` yamalanir: her mesafenin sonucu elle kurulur ve
arama ozetleri mesafeye gore AYRISIR (panel o zaman her mesafeyi etiketiyle
yazar). Neden SCF'siz: Rc yakininda hangi durumun dondugu BLAS cekirdegine
baglidir; bu dosya transkripsiyonu (etiket, npz anahtari) sinar, fizigi degil.

OLCULDU (HEAD 80eb21c, kamu CLI, SCF'li): N=6, distances [2.2340, 2.23401,
2.5], seeds 1 -> arama satirinda "d=2.234:" IKI kez, tablo satirinda
"d=2.234 bohr" IKI kez; npz anahtarlari delta, distances_bohr, energy_uhf,
epsilon_bar, hoppings, provenance_json, s_squared, staggering_residual --
`search_stable` YOK.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import qlab.engine.tb.factory as factory_modulu
from qlab.engine.tb.downfold import DownfoldedParameters
from qlab.engine.tb.factory import mesafe_etiketi
from qlab.run import main

YAKIN = (2.234, 2.23401, 2.5)


def _yaml(tmp_path, mesafeler) -> str:
    yol = tmp_path / "indirgeme.yaml"
    yol.write_text(
        "kind: downfold\nname: etiket denetimi\nring_size: 6\n"
        f"distances_bohr: [{', '.join(repr(d) for d in mesafeler)}]\n"
        "basis: sto-3g\nn_neighbour: 2\nseeds: 1\n",
        encoding="utf-8",
    )
    return str(yol)


def _yama(monkeypatch, sonuc: dict[float, tuple[float, bool | None]]) -> None:
    """d -> (delta, search_stable). Mesafe molekulden okunur (yardimci ithal edilmez)."""

    def sahte(molekul, *a, **k):
        r = molekul.atoms[0].position - molekul.atoms[1].position
        d = min(sonuc, key=lambda x: abs(x - float(np.linalg.norm(r))))
        delta, kararli = sonuc[d]
        return DownfoldedParameters(
            epsilon_bar=0.0, delta=delta, hoppings=(-0.3, 0.02), n_site=6,
            basis_name="sto-3g", s_squared=0.0 if delta == 0.0 else 0.9,
            energy_uhf=-3.0, staggering_residual=0.0,
            search_line=f"elle {d!r}", search_stable=kararli,
        )

    monkeypatch.setattr(factory_modulu, "downfold_uhf", sahte)


def test_YAKIN_MESAFELER_AYRI_ETIKET_alir_ARAMA_TANI_ve_TABLODA(tmp_path, monkeypatch, capsys):
    """B7dog-4: uc yerde de (arama satiri, "Yakinsama tanisi", tablo satiri)
    her mesafe AYRI bir etiketle yazilir ve etiket mesafeyi TAM geri verir."""
    # Iki yakin mesafe spin-KIRILMAMIS (Delta = 0) -> tani satirinda ikisi de adlanir.
    _yama(monkeypatch, {2.234: (0.0, True), 2.23401: (0.0, True), 2.5: (0.2, True)})
    assert main([_yaml(tmp_path, YAKIN), "-o", str(tmp_path / "x.npz")]) == 0
    cikti = capsys.readouterr().out

    tablo = re.findall(r"^  d=(\S+) bohr ", cikti, flags=re.M)
    assert tablo == ["2.234", "2.23401", "2.500"]
    arama = re.search(r"arama=(.*)$", cikti, flags=re.M).group(1)
    assert re.findall(r"d=(\S+): elle", arama) == ["2.234", "2.23401", "2.500"]
    tani = next(s for s in cikti.splitlines() if s.startswith("Yakınsama tanısı"))
    assert re.findall(r"d=(\S+): \|Delta\|", tani) == ["2.234", "2.23401"]
    for etiketler in (tablo, re.findall(r"d=(\S+): \|Delta\|", tani)):
        assert [float(e) for e in etiketler] == list(YAKIN[: len(etiketler)])


def test_ETIKET_BIREBIR_ve_SEVK_EDILEN_ETIKETLER_DEGISMEDI():
    """Uc ondalik `d`yi geri veriyorsa aynen (depodaki her senaryo), vermiyorsa
    `repr`. Birebirlik: farkli mesafeler farkli etiket, etiket mesafeyi verir."""
    for d, beklenen in ((2.0, "2.000"), (2.6, "2.600"), (1.8, "1.800"), (3.2, "3.200"),
                        (2.234, "2.234"), (2.23401, "2.23401"), (2.2338185, "2.2338185"), (3, "3.000")):
        assert mesafe_etiketi(d) == beklenen
    rc = 2.233818500
    adaylar = [rc + k * 1e-9 for k in range(-50, 51)] + [2.234, 2.2340000000000004, 2.2345]
    etiketler = [mesafe_etiketi(d) for d in adaylar]
    assert len(set(etiketler)) == len(set(adaylar))
    assert all(float(e) == d for e, d in zip(etiketler, adaylar))


def test_NPZ_MESAFE_BASINA_search_stable_TASIR(tmp_path, monkeypatch, capsys):
    """B7dog-3: npz `search_stable` bool dizisi; True YALNIZ kanitli kararlida."""
    _yama(monkeypatch, {2.6: (0.2, True), 2.8: (0.21, False), 3.0: (0.22, None)})
    cikti = tmp_path / "x.npz"
    assert main([_yaml(tmp_path, (2.6, 2.8, 3.0)), "-o", str(cikti)]) == 0
    capsys.readouterr()
    with np.load(cikti, allow_pickle=False) as veri:
        assert "search_stable" in veri.files
        assert veri["search_stable"].dtype == np.bool_
        assert veri["search_stable"].tolist() == [True, False, False]
        assert veri["search_stable"].shape == veri["distances_bohr"].shape
