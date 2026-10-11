"""K3 (11.10.2026): element listeleri bir DESTEK iddiasi tasimaz; baz listesi baza ozgu.

OLCULDU (HEAD 80eb21c, kamu CLI, `kind: hf_scan`): Na -> InvalidInputError
"Desteklenenler: ['B', 'Be', 'C', 'F', 'H', 'He', 'Li', 'N', 'Ne', 'O']";
Be2 / B2 / Ne2 (sto-3g) -> BasisNotAvailableError "'Be' elementi icin
'sto-3g' baz setinde veri yok" -- LISTESIZ. Yani "desteklenen" diye sayilan
uc element hicbir bazda cozulemiyordu (6-31G'de de veri yok).

KARAR (urun davranisi DEGISMEDI): `types.ATOMIC_NUMBERS` DARALTILMADI -- o bir
atom numarasi tablosudur ve `Molecule` HF disinda da kurulur; liste
"Taninan elementler" diye adlandirildi. Baz verisini bilen katman
(`build_basis`) kendi listesini basar.

SINAV DOGRULUK TABLOSUDUR: her baz x her taninan element icin `build_basis`
(SCF yok, tek atom) ya basarir ya da mesajindaki liste TAM OLARAK basaran
elementlerin kumesidir.
"""

from __future__ import annotations

import ast

import pytest

from qlab.engine.hf.basis import AVAILABLE_BASES, build_basis
from qlab.errors import BasisNotAvailableError, InvalidInputError
from qlab.types import ATOMIC_NUMBERS, Molecule


def _liste(mesaj: str) -> list[str]:
    return ast.literal_eval(mesaj[mesaj.index("[") : mesaj.index("]") + 1])


def test_BILINMEYEN_ELEMENT_mesaji_DESTEK_IDDIA_ETMEZ():
    with pytest.raises(InvalidInputError) as hata:
        Molecule.from_string("Na 0 0 0")
    mesaj = str(hata.value)
    assert "Bilinmeyen element: 'Na'" in mesaj
    assert "Desteklenen" not in mesaj
    assert "Taninan elementler" in mesaj and "HF destegi DEGILDIR" in mesaj
    assert _liste(mesaj) == sorted(ATOMIC_NUMBERS)


@pytest.mark.parametrize("baz", AVAILABLE_BASES)
def test_BAZ_MESAJININ_LISTESI_BASARAN_ELEMENTLERIN_KENDISI(baz):
    basaran, mesajlar = [], {}
    for el in sorted(ATOMIC_NUMBERS):
        try:
            build_basis(Molecule.from_string(f"{el} 0 0 0"), baz)
        except BasisNotAvailableError as hata:
            mesajlar[el] = str(hata)
        else:
            basaran.append(el)
    assert {"Be", "B", "Ne"} <= set(mesajlar), "veri olmayan uc element artik cozuluyor -- belgeyi guncelle"
    assert basaran, "hicbir element cozulmuyor"
    for el, mesaj in mesajlar.items():
        assert f"{el!r} elementi icin {baz!r} baz setinde veri yok" in mesaj
        assert _liste(mesaj) == basaran, (el, mesaj)
