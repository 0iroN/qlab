"""Zamana bagli yogunluk farki filmi -- YUK uretir, goruntu DEGIL.

Bu deponun `viz/` katmani hep yuk uretir (`viz/frames.py`, `viz/md_frames.py`);
`.mp4` yazmak yeni bagimlilik ve yeni katman demektir, kapsam disidir.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.forcefields import Morse
from qlab.engine.md.integrator import run
from qlab.engine.md.state import MDState
from qlab.errors import InvalidInputError
from qlab.units import amu_to_electron_mass
from qlab.viz.density_frames import build_density_film

M_H = amu_to_electron_mass(1.00782503207)
H2_RE_BOHR = 1.3459193602


def _gerilme_yorungesi(n_kare: int = 6, v_com: float = 0.0):
    """Bagi gererek acan basit bir yorunge -- kuvvet alani ONEMSIZ.

    OLCULDU (Plan D on olcumu): film ab initio kuvvet GEREKTIRMIYOR;
    Morse yorungesiyle 20 karelik kopus filmi 0.64 s ve SIFIR SCF.
    Buradaki yorunge yalnizca geometri uretir.

    `v_com` iki atoma da eklenen ortak surukleme hizidir. Varsayilan 0'da
    kutle merkezi HIC kaymaz (olculdu: her karede tam 0.000e+00) -- yani
    varsayilan yorunge merkezleme kodunu SINAMAZ, sifirdan farkli bir
    `v_com` gerekir.
    """
    durum = MDState(
        positions=np.array([[0.0, 0.0, -H2_RE_BOHR / 2], [0.0, 0.0, H2_RE_BOHR / 2]]),
        velocities=np.array(
            [[0.0, 0.0, -0.004 + v_com], [0.0, 0.0, 0.004 + v_com]]
        ),
        masses=np.full(2, M_H),
    )
    alan = Morse(d_e=0.174434333, r_e=H2_RE_BOHR, a=1.028997701)
    return run(durum, alan, dt=40.0, n_steps=(n_kare - 1) * 4, save_every=4)


def _tum_diziler(nesne, yol="yuk", derinlik=0):
    """Yukun ICINDEKI her NumPy dizisini gezer -- listelerin icine DE girer.

    Bu ozyineleme sus pus bir ayrinti degil, testin ta kendisi. Yalniz
    `vars(yuk)` taramak OLCULEREK yetersiz bulundu: `meshes` bir `list`
    oldugu icin `isinstance(deger, np.ndarray)` onu atliyordu ve "alan
    kupu saklaniyor" mutanti (ag yerine (41,41,41) dizisi biriktiren
    surum) besi testten de gecti.
    """
    if isinstance(nesne, np.ndarray):
        yield yol, nesne
    elif isinstance(nesne, (list, tuple)):
        for i, oge in enumerate(nesne):
            yield from _tum_diziler(oge, f"{yol}[{i}]", derinlik + 1)
    elif hasattr(nesne, "__dict__") and derinlik < 4:
        for ad, deger in vars(nesne).items():
            yield from _tum_diziler(deger, f"{yol}.{ad}", derinlik + 1)


def test_film_ORTA_NOKTA_serisi_cokuyor():
    """Filmin ANLATTIGI sey bu seridir: kovalent birikim cokuyor.

    OLCULDU (gercek HF-MD kopus yorungesi): 0.043821 (r=1.346) ->
    0.000577 (r=6.198) -> 0.000008 (r=9.415) -> sayisal sifir.
    Buradaki kisa yorunge ayni yonu gostermek zorunda.
    """
    yuk = build_density_film(
        _gerilme_yorungesi(), symbols=("H", "H"), half_width_bohr=8.0, n=41
    )
    assert yuk.midpoint.shape == yuk.times_fs.shape
    assert float(yuk.midpoint[0]) > float(yuk.midpoint[-1])
    assert float(yuk.midpoint[0]) == pytest.approx(0.043821, abs=2e-3, rel=0.0)


def test_film_ALAN_KUPU_SAKLAMIYOR():
    """Bellek sozlesmesi -- olcume dayanir.

    OLCULDU: (M,n,n,n) float64, M=100 ve n=80 icin 0.410 GB. Bir film icin
    kabul edilemez. Saklanmasi gereken alan degil AG: kare basina 354-1370
    tepe (n=41...81), yani kilobayt mertebesi.

    Bu test yukun alan kupu tasimadigini kilitler: hicbir alani
    (M,n,n,n) seklinde olamaz.
    """
    yuk = build_density_film(
        _gerilme_yorungesi(4), symbols=("H", "H"), half_width_bohr=8.0, n=41
    )
    diziler = list(_tum_diziler(yuk))
    assert diziler, "yukte hic dizi bulunamadi -- tarama bozuk"
    for yol, dizi in diziler:
        assert dizi.ndim <= 2, f"{yol} alan kupu tasiyor: {dizi.shape}"

    toplam = sum(dizi.nbytes for _, dizi in diziler)
    assert toplam < 300_000, f"yuk sismis: {toplam:,} bayt"


def test_film_KUTUYU_SABIT_tutuyor():
    """Kare basina kutu secimi filmi kadraj hareketine cevirir.

    OLCULDU (Plan D on olcumu): `around()` benzeri otomatik kutu kare basina
    kayiyor. Sabit kutuda ardisik kareler AYNI uzayi gosterir.
    """
    yuk = build_density_film(
        _gerilme_yorungesi(4), symbols=("H", "H"), half_width_bohr=8.0, n=41
    )
    assert yuk.grid_origin.tolist() == [-8.0, -8.0, -8.0]
    assert yuk.grid_shape == (41, 41, 41)


def test_film_KAYAN_kutle_merkezini_KUTUYA_TASIYOR():
    """Kutu sabitse molekul kutuya tasinmali -- yoksa film kadraj kayar.

    Varsayilan test yorungesi bunu SINAYAMAZ: olculdu, orada kutle merkezi
    her karede tam 0.000e+00, yani `konum - merkez` satiri olu koddur ve
    onu silen mutant besi testten de geciyordu. Ortak surukleme hizi
    eklenince kutle merkezi bes karede 1.92 bohr kayiyor.

    OLCULDU, merkezleme silinirse ne olur: Task 1'in izgara muhafazasi
    devreye girer ve `InvalidInputError: orta nokta izgarada degil (sapma
    8.000e-02 bohr)` firlar -- yani hata sessizce yanlis kare uretmez,
    yuksek sesle durur. Merkezleme varken ag agirlik merkezi orijinde
    kalir (olculdu: max |z| ~ 9e-09 bohr).
    """
    yorunge = _gerilme_yorungesi(5, v_com=0.003)
    com_z = yorunge.positions.mean(axis=1)[:, 2]
    assert float(np.abs(com_z).max()) == pytest.approx(1.92, abs=1e-2)

    yuk = build_density_film(
        yorunge, symbols=("H", "H"), half_width_bohr=8.0, n=41
    )
    merkez_z = np.array(
        [
            float(np.concatenate([arti.vertices, eksi.vertices])[:, 2].mean())
            for arti, eksi in yuk.meshes
        ]
    )
    assert float(np.abs(merkez_z).max()) < 1e-6


def test_film_CIFT_izgarayi_reddediyor():
    """Task 1'in parite muhafazasi buraya da gecmeli."""
    with pytest.raises(InvalidInputError, match="TEK"):
        build_density_film(
            _gerilme_yorungesi(3), symbols=("H", "H"), half_width_bohr=8.0, n=40
        )


def test_film_sembol_sayisi_uyusmazligini_reddediyor():
    with pytest.raises(InvalidInputError, match="sembol"):
        build_density_film(
            _gerilme_yorungesi(3), symbols=("H",), half_width_bohr=8.0, n=41
        )
