"""B7-d (11.10.2026): `fit_distance_laws` EYER donen bir mesafeyi yasaya sokmaz.

SINIF: KAPI DENETIMI, SCF'SIZ. `downfold.downfold_uhf` yamalanir ve her
mesafenin sonucu elle kurulur (`test_downfold_senaryo.py`deki esik
testlerinin kalibi). Neden SCF'siz: Rc yakininda hangi durumun dondugu BLAS
cekirdegine baglidir (`downfold.SPIN_KIRIK_ESIGI` notu). Kamu yolundaki
tanik varsayilan OpenBLAS'ta OLCULDU ve yalniz bir KAYITTIR (kilitlenmez):
N=6, pencere (2.2338285, 2.6, 3.0) bohr, seeds = 2 -- ilk mesafede spin-KIRIK
eyer (lambda_min -7.184e-06, |Delta| 2.127e-04 >= 2.5e-5, yani Delta esigi
onu gormez); kapi yokken `fit_distance_laws` hatasiz donuyordu
(delta_growth 9.40, ln|Delta| artigi 1.677 RMS).

Elle kurulan pencere spin-KIRIKTIR (|Delta| esigin cok ustunde) ve isaretleri
tutarlidir: kapi olmasa yasa uydurulurdu (kontrol testi bunu kosturur). Eyer
BASTA, ORTADA ve SONDA ayri ayri sinanir -- yalniz bir ucuna bakan ya da ilk
mesafeyi atlayan bir kapi gormezdi (kamu yolundaki tanikta eyer ILK
mesafedeydi; dogrulama turu `zip(d[1:], p[1:])` mutantinin [2.8, 3.0]
parametresiyle hayatta kaldigini olctu).
"""

from __future__ import annotations

import pytest

import qlab.engine.tb.downfold as downfold_modulu
from qlab.engine.tb.downfold import DistanceLaws, DownfoldedParameters
from qlab.errors import UnsupportedSystemError

PENCERE = (2.6, 2.8, 3.0)


def _parametre(d: float, kararli: bool | None) -> DownfoldedParameters:
    """SCF'siz, spin-kirik, isaretleri tutarli bir mesafe sonucu."""
    return DownfoldedParameters(
        epsilon_bar=-0.02 + 0.1 * (0.3 * 2.718281828 ** (-0.9 * d)) ** 2,
        delta=0.2 * 2.718281828 ** (0.15 * (d - 2.6)),
        hoppings=(-0.3 * 2.718281828 ** (-0.9 * (d - 2.6)), 0.02 * 2.718281828 ** (-1.5 * (d - 2.6))),
        n_site=6,
        basis_name="sto-3g",
        s_squared=0.9,
        energy_uhf=-3.0,
        staggering_residual=0.0,
        search_line=f"elle d={d}" + ("" if kararli else " -- EYER (elle)"),
        search_stable=kararli,
    )


def _yama(monkeypatch, kararlilik: dict[float, bool | None]) -> None:
    def sahte(molekul, *a, **k):
        # Mesafe halkanin ilk iki atomunun arasindan okunur (yardimci ithal edilmez).
        import numpy as np

        r = molekul.atoms[0].position - molekul.atoms[1].position
        d = round(float(np.linalg.norm(r)), 9)
        return _parametre(d, kararlilik[d])

    monkeypatch.setattr(downfold_modulu, "downfold_uhf", sahte)


def test_KONTROL_hepsi_KARARLIYKEN_yasa_UYDURULUR(monkeypatch):
    _yama(monkeypatch, {d: True for d in PENCERE})
    yasa = downfold_modulu.fit_distance_laws(6, PENCERE)
    assert isinstance(yasa, DistanceLaws)
    assert yasa.window_bohr == PENCERE


@pytest.mark.parametrize("eyer", [2.6, 2.8, 3.0])
def test_EYER_donen_MESAFE_yasaya_GIRMEZ(monkeypatch, eyer):
    """Spin-KIRIK eyer (|Delta| 0.2): Delta esigi onu GORMEZ, kararlilik kapisi gorur."""
    _yama(monkeypatch, {d: d != eyer for d in PENCERE})
    with pytest.raises(UnsupportedSystemError, match="UHF-kararli OLMAYAN") as hata:
        downfold_modulu.fit_distance_laws(6, PENCERE)
    mesaj = str(hata.value)
    assert f"d={eyer!r}: elle d={eyer} -- EYER (elle)" in mesaj
    assert "(1/3)" in mesaj
    for kararli in (d for d in PENCERE if d != eyer):
        assert f"d={kararli!r}:" not in mesaj, "kararli mesafe eyer diye adlandirildi"


def test_OLCULMEMIS_KARARLILIK_da_KABUL_EDILMEZ(monkeypatch):
    """`search_stable` None ("olculmedi", `DistanceLaws.at`in degeri) kararlilik
    KANITI degildir; kapi `is True` ister."""
    _yama(monkeypatch, {2.6: True, 2.8: None, 3.0: True})
    with pytest.raises(UnsupportedSystemError, match="UHF-kararli OLMAYAN"):
        downfold_modulu.fit_distance_laws(6, PENCERE)
