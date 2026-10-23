"""Grup 6 serit A (11.10.2026): D14 MD yolunun cakisik atom kapisi ve NP-BOOL ailesi.

Her kapinin testi HEAD a9ec48a'da KIRMIZIDIR (olculdu, kamu fonksiyonlari):
- D14: 8 atomlu zincirde (a = 2, L = 16, n_e = 8, kT = 2e-3) atom 1 atom 0'a
  d = 1e-9 / 5e-10 / 1e-12 yaklasinca `tek_orbital_mermin` ve
  `SsSigmaTBKuvveti` Omega = 18.318899 HESAPLIYORDU; kafes yolu
  (`neighbour_shells`) ayni bantta `InvalidInputError` veriyordu. Osman
  karari (11.10): MD yoluna da kafes yolunun 1e-9 bohr kapisi (iki yol
  tutarli; bu esige inen bir yorunge zaten fiziksiz bolgededir -- B9'un
  V-kuyusu d'yi 0'a cekebilir -- ve oraya girip girmedigi OLCULMEDI).
- NP-BOOL: `np.True_` ve `np.array(True)` `tb/mermin.py::_skaler`,
  `md/tb_kuvvet.py::_sayi` ve `tb/radyal.py::_sonlu_skaler`den 1.0 diye
  geciyordu; radyal Python `True`yu da geciriyordu (YAML
  `genlik_hartree: true` -> genlik 1.0 hartree). `np.complex128` bir
  ComplexWarning ile gercel kismina iniyordu; `10**400` ham OverflowError,
  duzensiz ic ice liste ham ValueError veriyordu. Desen
  `spin/factory.py::_skaler`in (grup 2) aynisidir.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from qlab.engine.md import tb_kuvvet
from qlab.engine.md.tb_kuvvet import SsSigmaTBKuvveti
from qlab.engine.tb import lattice, mermin, radyal
from qlab.engine.tb.mermin import tek_orbital_mermin
from qlab.engine.tb.radyal import UsselRadyalYasa, kosinus_kuyrugu
from qlab.errors import InvalidInputError

_SS = UsselRadyalYasa(-0.1, 2.0, 2.0, 3.0, 3.8)
_ITME = UsselRadyalYasa(0.05, 2.0, 6.0, 3.0, 3.8)
_L = 16.0


def _zincir(d: float) -> np.ndarray:
    """8 atom, a = 2; atom 1 atom 0'dan d uzakta (x)."""
    R = np.zeros((8, 3))
    R[:, 0] = 2.0 * np.arange(8)
    R[1, 0] = R[0, 0] + d
    return R


# ---------------------------------------------------------------------- D14 --


@pytest.mark.parametrize("kutu", [_L, None], ids=["kutulu", "kutusuz"])
@pytest.mark.parametrize("d", [1e-9, 5e-10, 1e-12, 0.0])
def test_D14_MD_yolu_KAFES_esiginde_HATA(d, kutu):
    """0 < d <= 1e-9 bandi (ve d = 0) iki MD girisinde de, KUTULU ve KUTUSUZ, `InvalidInputError`.

    `d = 1e-9` TAM esittir (kafes yolu da `<=`). KIRMA (olculdu): esigi 0.0'a
    indiren mutant (HEAD'in davranisi) 1e-9 / 5e-10 / 1e-12'de hata vermiyor;
    `<=` yerine `<` yazan mutant d = 1e-9'da hata vermiyor. Kutusuz kol
    (grup 6 dogrulayicisi, 11.10.2026): esigi yalniz `kutu_x_bohr is not None`
    iken uygulayip kutusuzda eski `d == 0` kapisina donen mutant, kutulu
    kollarin hepsinden SAG cikiyordu (serit A'nin 372 testi); burada olur.
    """
    R = _zincir(d)
    with pytest.raises(InvalidInputError, match=r"cakisik atom cifti \(0, 1\)"):
        tek_orbital_mermin(R, _SS, _ITME, 8.0, 2e-3, kutu)
    alan = SsSigmaTBKuvveti(_SS, _ITME, 8.0, 2e-3, kutu)
    with pytest.raises(InvalidInputError, match="cakisik atom"):
        alan(R)


def test_D14_x_GORUNTUSU_sonrasi_bantta_HATA_kutusuz_GECERLI():
    """Atom 7, atom 0'in +L goruntusune 5e-10 yaklasir: kutuyla hata, kutusuz uzak (d ~ 16).

    Kapi en-yakin-goruntu SONRASI mesafeye bakar (kafes yolunun goruntu
    cakismasinin esi).
    """
    R = _zincir(2.0)
    R[7, 0] = R[0, 0] + _L - 5e-10
    with pytest.raises(InvalidInputError, match=r"cakisik atom cifti \(0, 7\)"):
        tek_orbital_mermin(R, _SS, _ITME, 8.0, 2e-3, _L)
    sonuc = tek_orbital_mermin(R, _SS, _ITME, 8.0, 2e-3)
    assert np.isfinite(sonuc.omega) and sonuc.kuvvet is not None


@pytest.mark.parametrize("kutu", [_L, None], ids=["kutulu", "kutusuz"])
def test_D14_esigin_USTU_2e9_HESAPLANIR(kutu):
    """d = 2e-9 > esik: iki giris de, kutulu ve kutusuz, hesaplar (kafes yoluyla esitligi `test_distance_model`de).

    KIRMA (olculdu): esigi 1e-8'e cikaran mutant burada hata veriyor.
    """
    R = _zincir(2e-9)
    sonuc = tek_orbital_mermin(R, _SS, _ITME, 8.0, 2e-3, kutu)
    omega, kuvvet = SsSigmaTBKuvveti(_SS, _ITME, 8.0, 2e-3, kutu)(R)
    assert np.isfinite(sonuc.omega) and abs(omega - sonuc.omega) == 0.0
    assert kuvvet.shape == (8, 3) and np.all(np.isfinite(kuvvet))


def test_D14_IKIZ_esik_KAFES_yoluyla_AYNI_sabit():
    """`mermin` import kilidi (spec 2.1(2), M10) yuzunden `lattice`ten import edemez: ikiz sabit.

    Biri degisip oteki kalirsa kirmizi (iki yolun tutarliligi tek sayiya bagli).
    """
    assert mermin._CAKISMA_ESIGI_BOHR == lattice._CAKISMA_ESIGI_BOHR == 1e-9


# ------------------------------------------------------------------ NP-BOOL --


_MERMIN_GECERLI = {"n_elektron": 8.0, "kT_hartree": 2e-3, "kutu_x_bohr": _L}


def _mermin_cagir(**degis):
    a = _MERMIN_GECERLI | degis
    return tek_orbital_mermin(_zincir(2.0), _SS, _ITME, a["n_elektron"], a["kT_hartree"], a["kutu_x_bohr"])


def _kuvvet_kur(**degis):
    a = _MERMIN_GECERLI | degis
    return SsSigmaTBKuvveti(_SS, _ITME, a["n_elektron"], a["kT_hartree"], a["kutu_x_bohr"])


@pytest.mark.parametrize("kur", [_mermin_cagir, _kuvvet_kur], ids=["mermin", "tb_kuvvet"])
@pytest.mark.parametrize("alan", list(_MERMIN_GECERLI))
@pytest.mark.parametrize(("deger_fn", "parca"), [
    pytest.param(lambda g: np.True_, "tek bir sayi olmali", id="np.True_"),
    pytest.param(lambda g: np.array(True), "tek bir sayi olmali", id="0d-bool-dizi"),
    pytest.param(lambda g: np.complex128(g), r"sayiya cevrilemiyor \(karmasik\)", id="np.complex128"),
    pytest.param(lambda g: 10**400, "sayiya cevrilemiyor", id="10**400"),
    pytest.param(lambda g: [[g], [g, g]], "tek bir sayi olmali", id="duzensiz-liste"),
    # B4 (grup 9): 0 boyutlu NESNE dizisi -- HEAD 448f808'de 1.0 / dizenin sayisi diye geciyordu.
    pytest.param(lambda g: np.array(True, dtype=object), "tek bir sayi olmali", id="0d-nesne-bool"),
    pytest.param(lambda g: np.array(np.True_, dtype=object), "tek bir sayi olmali", id="0d-nesne-np.True_"),
    pytest.param(lambda g: np.array(str(g), dtype=object), "tek bir sayi olmali", id="0d-nesne-dize"),
    pytest.param(lambda g: np.array(complex(g), dtype=object), r"sayiya cevrilemiyor \(karmasik\)",
                 id="0d-nesne-karmasik"),
    # V6 (grup 9 duzeltici): 0 boyutlu DIZE dizisi -- HEAD 448f808'de ve duzeltmeden once dizenin sayisi.
    pytest.param(lambda g: np.array(str(g)), "tek bir sayi olmali", id="0d-U-dizi"),
    pytest.param(lambda g: np.array(str(g).encode()), "tek bir sayi olmali", id="0d-S-dizi"),
])
def test_NP_BOOL_mermin_ve_tb_kuvvet_REDDEDER(kur, alan, deger_fn, parca):
    """`_skaler` / `_sayi`: numpy bool, karmasik, tasan tamsayi, duzensiz liste -> tipli hata.

    Mesaj parcasi KAPIYI da sinar: `np.True_` kT'de 1 hartree'lik Kelvin
    kapisina, L'de L/2 kapisina takilirsa bu test yine kirmizidir.
    """
    deger = deger_fn(_MERMIN_GECERLI[alan])
    with pytest.raises(InvalidInputError, match=f"{alan} {parca}"):
        kur(**{alan: deger})


_YASA_GECERLI = {"genlik_hartree": -0.1, "d0_bohr": 2.0, "us": 2.0, "r_on_bohr": 3.0, "r_cut_bohr": 3.8}


@pytest.mark.parametrize("alan", list(_YASA_GECERLI))
@pytest.mark.parametrize(("deger_fn", "parca"), [
    pytest.param(lambda g: True, "tek bir sayi olmali", id="True"),
    pytest.param(lambda g: np.True_, "tek bir sayi olmali", id="np.True_"),
    pytest.param(lambda g: np.array(True), "tek bir sayi olmali", id="0d-bool-dizi"),
    pytest.param(lambda g: np.complex128(g), r"sayiya cevrilemiyor \(karmasik\)", id="np.complex128"),
    pytest.param(lambda g: 10**400, "sayiya cevrilemiyor", id="10**400"),
    pytest.param(lambda g: [[g], [g, g]], "tek bir sayi olmali", id="duzensiz-liste"),
    pytest.param(lambda g: np.array(True, dtype=object), "tek bir sayi olmali", id="0d-nesne-bool"),  # B4
    pytest.param(lambda g: np.array(str(g), dtype=object), "tek bir sayi olmali", id="0d-nesne-dize"),  # B4
    pytest.param(lambda g: np.array(str(g)), "tek bir sayi olmali", id="0d-U-dizi"),  # V6
    pytest.param(lambda g: np.array(str(g).encode()), "tek bir sayi olmali", id="0d-S-dizi"),  # V6
])
def test_NP_BOOL_radyal_yasa_BES_alanda_REDDEDER(alan, deger_fn, parca):
    """`UsselRadyalYasa`: Python bool da (HEAD'de HIC reddedilmiyordu) -> tipli hata."""
    with pytest.raises(InvalidInputError, match=f"{alan} {parca}"):
        UsselRadyalYasa(**(_YASA_GECERLI | {alan: deger_fn(_YASA_GECERLI[alan])}))


@pytest.mark.parametrize(("konum", "ad"), [(1, "r_on_bohr"), (2, "r_cut_bohr")], ids=["r_on", "r_cut"])
@pytest.mark.parametrize("deger_fn", [lambda g: True, lambda g: np.True_, lambda g: np.complex128(g)],
                         ids=["True", "np.True_", "np.complex128"])
def test_NP_BOOL_kosinus_kuyrugu_sinirlari_REDDEDER(konum, ad, deger_fn):
    """Kamu fonksiyonu `kosinus_kuyrugu(d, r_on, r_cut)` ayni kapidan gecer."""
    argumanlar = [1.0, 3.0, 3.8]
    argumanlar[konum] = deger_fn(argumanlar[konum])
    with pytest.raises(InvalidInputError, match=f"{ad} (tek bir sayi|sayiya cevrilemiyor)"):
        kosinus_kuyrugu(*argumanlar)


def test_NP_BOOL_GECERLI_numpy_sayilari_KABUL():
    """Asiri reddetmeye karsi: numpy tam/gercel skalerleri ve 0 boyutlu gercel dizi gecer, deger aynen."""
    # B4 (grup 9): 0 boyutlu GERCEL nesne dizisi ve float'a sigan buyuk tamsayi (dtype 'O')
    # GECER -- nesne dtype'ini toptan reddeden kapi bunlari da reddederdi.
    for n in (np.float64(8.0), np.int64(8), np.float32(8.0), np.array(8.0), 8, np.array(8.0, dtype=object)):
        assert _kuvvet_kur(n_elektron=n).n_elektron == 8.0
        assert np.isfinite(_mermin_cagir(n_elektron=n).omega)
    assert radyal._sonlu_skaler("x", 2**70) == float(2**70) and np.asarray(2**70).dtype.kind == "O"
    yasa = UsselRadyalYasa(np.float64(-0.1), np.int64(2), np.array(2.0), np.float32(3.0), 3.8)
    assert (yasa.genlik_hartree, yasa.d0_bohr, yasa.us, yasa.r_on_bohr) == (-0.1, 2.0, 2.0, 3.0)


def _govde(fonksiyon) -> str:
    """Dokstringsiz govde + argumanlar, ad haric (`ast.dump`); `test_dusuk_tb_kapilari`nin esi."""
    tanim = ast.parse(inspect.getsource(fonksiyon)).body[0]
    govde = tanim.body[1:] if isinstance(tanim.body[0], ast.Expr) and isinstance(
        getattr(tanim.body[0], "value", None), ast.Constant
    ) else tanim.body
    return ast.dump(ast.Module(body=govde, type_ignores=[])) + ast.dump(tanim.args)


def test_NP_BOOL_UC_KAPI_AYNI_govde():
    """`tb/radyal._sonlu_skaler` de D17 ikizlerinin (`_skaler`, `_sayi`) govdesini tasir.

    HEAD'de radyal'in govdesi farkliydi (bool'u hic denetlemiyordu): aile
    ancak uc govde birlikte degisirse tutarli kalir.
    """
    assert _govde(radyal._sonlu_skaler) == _govde(mermin._skaler) == _govde(tb_kuvvet._sayi)
