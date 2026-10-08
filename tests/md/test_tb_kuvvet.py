"""ss-sigma TB kuvvet alani (`engine/md/tb_kuvvet.py`) -- spec md-tb ss-sigma G4.

Kapsam: spec 4.2 M10'un geri kalani (tb_kuvvet `ast` import kilidi, yasanin
ikinci kopyasi yasagi, alt surecte import -> SIFIR hf modulu, TB MD
kosusunda `hf.scf.solve` SIFIR cagri + HF pozitif kontrolu), M11 (NVE,
KE + Omega salinimi O(dt^2)), B1'in dinamik kaniti (KE + E KORUNMAZ, kT
dS'de doyar -- M11'in kor kaldigi mutant icin, spec'e EK),
`ForceField` sozlesmesi, kT > 0 kapisi ve
Kelvin -> hartree donusumu (spec B7: "Kelvin sanilip hartree girilen 300"
tuzagina karsi savunma ve yakalanmayan ters tuzagin acik beyani), ve CLI
uctan uca kosumu (loader -> factory -> integrator -> panel). G5 (spec
2.1(7), B8, elestirmen P0-2): TB surukleme olcutunun enerji sifirindan
bagimsizligi (Omega + c), esigin TB'ye ozgu suruklenmeyi (Fermi gecisi)
yakalamasi, max(KE) paydasinin duragan olmayan kosudaki gerekcesi ve sevk
edilen `scenarios/tb_zincir_md.yaml`in CLI uctan uca kosumu.

ETIKETLER. Buradaki testlerin HEPSI MAKINE testidir (uygulama tutarliligi);
model cipalari G2'de (`tests/tb/test_mermin.py`). Bu kenarda DOGA CIPASI
YOKTUR (spec B4): yasa turetilmedi, secildi.

ARAC TUZAKLARI (spec 4.1). `assert_allclose`/`pytest.approx` kullanilmaz;
her karsilastirma acik `abs(x - y) < tol` ya da bit bit `==`dir. Kilitler
dosyayi KOSAN modulden okur (`modul.__file__`), sabit depo yolundan degil
(G1'de olculdu: sabit yol PYTHONPATH kopyasindaki mutantlari gormuyordu).

SURE (olculdu, 24.09.2026, bu makine, `--durations`): 32 test, dosyanin
tamami 2.9 s (G4'te 28 test, 1.8-1.9 s); en yavaslar G5 sifir-bagimsizlik
0.54 s (6 kosu x 100-200 adim), B1 0.43-0.46 s (3 x 1.5 T_min, adim basina
iki cagri), alt surec 0.36-0.38 s, G5 Fermi 0.35 s (2 x 400 adim), G5 sevk
edilen senaryo 0.25 s, M11 0.24 s (329 adim + L-BFGS), G5 duragan-olmayan
0.06 s, CLI 0.02-0.04 s; digerleri < 0.03 s. Bulgu duzeltme turundan
sonra (ayni gun) 38 test, 3.5 s (yuk altindaki makinede; yeni B9 ve kapi
hucreleri < 0.01 s). Hicbiri `slow` DEGILDIR (spec
4.1: depo emsali 0.3 s'lik cipayi slow saymiyor; depo esigi 2 s).

BAGLAYICILIK OLCULDU (24.09.2026): her tolerans ve kilit en az bir mutantla
kirildi -- src kopyasina uygulanan mutant + `PYTHONPATH=<kopya>/src`;
hangi mutantin hangi testi kirdigi ilgili testin dokstringinde yazili.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

import qlab.engine.hf.scf as hf_scf
import qlab.engine.md.forcefields as forcefields_modulu
import qlab.engine.md.tb_kuvvet as tb_kuvvet_modulu
from qlab.engine.md.factory import build, energy_drift, energy_oscillation, provenance_for
from qlab.engine.md.integrator import run, velocity_verlet_step
from qlab.engine.md.state import MDState
from qlab.engine.md.tb_kuvvet import SsSigmaTBKuvveti
from qlab.engine.tb.mermin import tek_orbital_mermin
from qlab.engine.tb.radyal import UsselRadyalYasa
from qlab.errors import InvalidInputError
from qlab.scenario.schema import MDScenario
from qlab.units import KB_HARTREE_PER_KELVIN, amu_to_electron_mass

TB_KUVVET_DOSYASI = Path(tb_kuvvet_modulu.__file__).resolve()

#: Ortak pencere (spec basi): ss_sigma -0.1/2/2, itme 0.05/2/6, r_on 3.0,
#: r_cut 3.8, a = 2.
SS_SIGMA = UsselRadyalYasa(-0.1, 2.0, 2.0, 3.0, 3.8)
ITME = UsselRadyalYasa(0.05, 2.0, 6.0, 3.0, 3.8)
M_H = float(amu_to_electron_mass(1.00782503207))


def _zincir(N: int, a: float, genlik: float = 0.0, tohum: int = 0) -> np.ndarray:
    R = np.zeros((N, 3))
    R[:, 0] = a * np.arange(N)
    if genlik:
        R[:, 0] += genlik * np.random.default_rng(tohum).standard_normal(N)
    return R


def _alan(ne: float = 16.0, kT: float = 2e-3, L: float | None = 32.0) -> SsSigmaTBKuvveti:
    return SsSigmaTBKuvveti(SS_SIGMA, ITME, ne, kT, L)


#: 3B bozulmus periyodik zincir: yasalar, kutu ve elektron sayisi ayri
#: ayri iz birakir (duzgun zincirde simetri bazi hatalari yutardi).
_RNG = np.random.default_rng(11)
_R3B = _zincir(16, 2.0, 0.1, 3) + np.c_[np.zeros(16), 0.1 * _RNG.standard_normal((16, 2))]


# ------------------------------------------------------------- sozlesme --


def test_SOZLESME_cagri_TANI_ve_tek_orbital_mermin_BIT_BIT_AYNI():
    """`__call__` -> (float, (N, 3)); tani() ve dogrudan cagri ile BIT BIT ayni.

    Sinif hicbir sey hesaplamaz, yalniz sarar: Omega ve kuvvet
    `tek_orbital_mermin`inkilerle `==`. Pencere 3B bozulmus zincir (N16,
    L = 32, n_e = 16, kT = 2e-3). KIRMA (olculdu): tani()'de iki yasanin
    yeri degistirilince, kutu None gecirilince ve n_elektron + 1
    gecirilince uc mutant da kirmizi.
    """
    alan = _alan()
    omega, kuvvet = alan(_R3B)
    assert type(omega) is float
    assert isinstance(kuvvet, np.ndarray) and kuvvet.shape == (16, 3)
    tani = alan.tani(_R3B)
    ref = tek_orbital_mermin(_R3B, SS_SIGMA, ITME, 16.0, 2e-3, 32.0)
    assert omega == tani.omega == ref.omega
    assert np.array_equal(kuvvet, ref.kuvvet) and np.array_equal(tani.kuvvet, ref.kuvvet)
    assert np.array_equal(np.asarray(tani.eps), np.asarray(ref.eps))
    # Kontrol: kutu gercekten iz birakiyor (kutulu != kutusuz).
    assert _alan(L=None)(_R3B)[0] != omega


def test_SOZLESME_integrator_run_ile_KOSUYOR_ve_POTANSIYEL_OMEGA():
    """Degismemis `integrator.run` alani ayirt etmez; kayitli potansiyel Omega'dir.

    Pencere: zincir(16, 2, 0.05, 3), M_H, dt = 20, 10 adim. Her kayitta
    potansiyel == alan(konum)[0] (E degil Omega, spec B1).
    """
    alan = _alan()
    durum = MDState(
        positions=_zincir(16, 2.0, 0.05, 3), velocities=np.zeros((16, 3)),
        masses=np.full(16, M_H),
    )
    yorunge = run(durum, alan, dt=20.0, n_steps=10)
    assert yorunge.positions.shape == (11, 16, 3)
    assert np.all(np.isfinite(yorunge.potential)) and np.all(np.isfinite(yorunge.kinetic))
    assert not np.array_equal(yorunge.positions[0], yorunge.positions[-1])
    for k in (0, 5, 10):
        assert yorunge.potential[k] == alan(yorunge.positions[k])[0]
        s = alan.tani(yorunge.positions[k])
        assert yorunge.potential[k] == s.E_band - 2e-3 * s.entropi + s.E_rep


def test_SOZLESME_DONDURULMUS_ve_ESITLIK_KIMLIK():
    """frozen (alan sessizce degistirilemez) ve eq=False (kardes siniflar gibi)."""
    alan = _alan()
    with pytest.raises(dataclasses.FrozenInstanceError):
        alan.kT_hartree = 1e-3  # type: ignore[misc]
    assert alan != _alan() and alan == alan
    assert isinstance(hash(alan), int)


# ------------------------------------------------------ kT > 0 kapisi --


@pytest.mark.parametrize(
    ("kT", "desen"),
    [
        (0.0, "kT_hartree > 0"),
        (-1e-3, "kT_hartree > 0"),
        (math.nan, "sonlu olmali"),
        (math.inf, "sonlu olmali"),
        (True, "tek bir sayi"),
        ("0.002", "tek bir sayi"),
        (np.array([2e-3]), "tek bir sayi"),
    ],
    ids=["sifir", "negatif", "nan", "inf", "bool", "dizge", "dizi"],
)
def test_kT_SIFIR_ve_GECERSIZ_REDDEDILIR(kT, desen):
    """kT_hartree > 0 ZORUNLU (spec 2.1(4), B2): MD'de kT = 0 yok.

    Her hucre KENDI kapisinin mesajiyla eslesir (`match=`). OLCULDU
    (24.09.2026, bulgu duzeltme turu): ilk surumde `match` yoktu ve `bool`
    hucresi YANLIS SEBEPLE yesildi -- `_sayi`dan bool'u silen mutantta
    True 1.0'a cevriliyor ve Kelvin kapisina (kT >= 1) takiliyordu.
    KIRMA (olculdu): `not kT > 0.0` kapisi kaldirilinca `sifir` ve
    `negatif` hucreleri kirmizi (DID NOT RAISE; kT = 0 sinif kurulur --
    asagidaki ikinci-bekci testinin var olma sebebi); `_sayi`dan bool'u
    silen mutant `bool` hucresini kiriyor (mesaj Kelvin kapisinin).
    """
    with pytest.raises(InvalidInputError, match=desen):
        _alan(kT=kT)


def test_kT_SIFIR_ikinci_BEKCI_kesirli_kuvvet_None_TIPLI_HATA():
    """Kurulus kapisi asilirsa (dogrudan `object.__setattr__`) `__call__` ham
    TypeError degil `InvalidInputError` verir: kT = 0 yari dolu duzgun
    halkada Fermi cifti kesirli ve `tek_orbital_mermin` kuvveti None
    dondurur (G2). KIRMA (olculdu): ikinci bekci silinince cagri (Omega,
    None) dondurur -> kirmizi (DID NOT RAISE); integrator o None'u
    `kuvvet / kutle`de ham TypeError olarak dusururdu.
    """
    alan = _alan()
    object.__setattr__(alan, "kT_hartree", 0.0)
    assert alan.tani(_zincir(16, 2.0)).kuvvet is None
    with pytest.raises(InvalidInputError, match="kT > 0"):
        alan(_zincir(16, 2.0))


@pytest.mark.parametrize(
    ("degisiklik", "desen"),
    [
        (dict(n_elektron=-1.0), "n_elektron"),
        (dict(n_elektron="16"), "n_elektron"),
        (dict(n_elektron=True), "n_elektron tek bir sayi"),
        (dict(n_elektron=1 + 1j), "n_elektron sayiya cevrilemiyor"),
        (dict(ss_sigma=(-0.1, 2.0, 2.0, 3.0, 3.8)), "ss_sigma"),
        (dict(itme=None), "itme"),
        (dict(kutu_x_bohr=7.6), "L/2"),
        (dict(kutu_x_bohr=math.nan), "kutu_x_bohr"),
        (dict(kutu_x_bohr=True), "kutu_x_bohr tek bir sayi"),
    ],
    ids=[
        "ne-negatif", "ne-dizge", "ne-bool", "ne-karmasik", "yasa-demet", "itme-None",
        "L-2rcut", "L-nan", "L-bool",
    ],
)
def test_GECERSIZ_KURULUS_REDDEDILIR(degisiklik, desen):
    """Her kapi kendi ornegiyle: yalniz o kapinin yakalayacagi bicimde kurulu.

    `_sayi`nin bool kapisi ve try/except dali ilk surumde BAGLANMAMISTI
    (test-korlugu incelemesi, olculdu): bool'u silen mutant her yerde
    yesildi -- `n_elektron = True` sinifi sessizce TEK elektronla kuruyor,
    `kutu_x_bohr = True` L/2 kapisina takiliyordu -- ve try/except dali hic
    calismiyordu (1 + 1j icin mutantta ham TypeError sizar). KIRMA
    (olculdu, 24.09.2026, sk/duz/mut_md.log): bool'u silen mutant `ne-bool`
    ve `L-bool` hucrelerini, try/except'i silen mutant `ne-karmasik`
    hucresini kiriyor.
    """
    alanlar = dict(ss_sigma=SS_SIGMA, itme=ITME, n_elektron=16.0, kT_hartree=2e-3, kutu_x_bohr=32.0)
    alanlar.update(degisiklik)
    with pytest.raises(InvalidInputError, match=desen):
        SsSigmaTBKuvveti(**alanlar)


def test_L_YARI_siniri_KURULUSTA_KESIN():
    """2 max(r_cut) < L kurulusta: L = 7.6 red, L = 7.6000001 kabul (spec B5, M9).

    Kesme iki yasanin BUYUGUDUR ve bu KURULUSTA denetlenir: asimetrik iki
    hucre sinifi dogrudan kurar ve HIC CAGIRMAZ -- hata mermin'in cagri
    kapisindan degil kurulustan gelmek zorunda. (a) itme genis (ss_sigma
    r_cut 3.8, itme 4.2): L = 8.0 red, L = 8.4 + 1e-7 kabul; (b) roller
    ters (ss_sigma 4.2, itme 3.8): L = 8.0 red.
    KIRMA (olculdu, 24.09.2026, sk/duz/mut_md.log): `kesme =
    self.ss_sigma.r_cut_bohr` mutanti (a)'yi, `kesme = self.itme.r_cut_bohr`
    mutanti (b)'yi kiriyor; esit r_cut'li 7.6 hucreleri ikisine de KORDU
    (ilk surum yalniz onlari tasiyordu).
    """
    with pytest.raises(InvalidInputError):
        _alan(L=7.6)
    assert _alan(L=7.6000001).kutu_x_bohr == 7.6000001
    dar = UsselRadyalYasa(-0.1, 2.0, 2.0, 3.0, 3.8)
    genis = UsselRadyalYasa(0.05, 2.0, 6.0, 3.0, 4.2)
    with pytest.raises(InvalidInputError, match="L/2"):
        SsSigmaTBKuvveti(dar, genis, 16.0, 2e-3, 8.0)
    assert SsSigmaTBKuvveti(dar, genis, 16.0, 2e-3, 8.4 + 1e-7).kutu_x_bohr == 8.4 + 1e-7
    genis_t = UsselRadyalYasa(-0.1, 2.0, 2.0, 3.0, 4.2)
    dar_itme = UsselRadyalYasa(0.05, 2.0, 6.0, 3.0, 3.8)
    with pytest.raises(InvalidInputError, match="L/2"):
        SsSigmaTBKuvveti(genis_t, dar_itme, 16.0, 2e-3, 8.0)


# ------------------------------------------------ B9: sert cekirdek yok --

#: d izgarasi (hepsi < r_on = 3.0: fc = 1 bolgesi).
_B9_IZGARA = (1e-3, 0.05, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9)


def _dimer_egrisi(itme: UsselRadyalYasa) -> tuple[list[float], float]:
    alan = SsSigmaTBKuvveti(SS_SIGMA, itme, 2.0, 2e-3)
    omega, F0 = [], 0.0
    for d in _B9_IZGARA:
        R = np.zeros((2, 3))
        R[1, 0] = d
        o, f = alan(R)
        omega.append(o)
        if d == _B9_IZGARA[0]:
            F0 = float(f[1, 0])
    return omega, F0


@pytest.mark.parametrize(
    ("A", "krep", "cokme"),
    [(0.05, 6.0, False), (0.05, 2.0, True), (0.04, 3.0, True)],
    ids=["ortak-us6-DUVAR", "krep-esit-k-COKER", "krep-buyuk-k-yine-COKER"],
)
def test_B9_V_KUYUSU_kosulu_ve_SONLU_DUVAR(A, krep, cokme):
    """Spec B9 (MAKINE; B4 geregi parametreler SECIMDIR, cipa degil).

    Dimer, n_e = 2, kT = 2e-3, kutusuz; hoplama t0 = -0.1, d0 = 2, k = 2;
    itme genligi A, ussu krep; d izgarasi {1e-3 .. 2.9} (< r_on = 3,
    fc = 1). Kosul (d = 0'da Omega'nin egimi pozitif): 2|t0| k e^k >
    A krep e^krep -- testte kapali bicimden hesaplanir ve hucrenin
    sinifiyla uyusmasi ayrica sinanir. (ortak) krep = 6: kosul tutmaz
    (2.956 < 121.0); duvar SONLU ve yuksek: Omega(1e-3) - Omega(2) = +18.78
    (> 0 ve sonlu). (krep = k = 2) yeter kosul (krep <= k, 2|t0| k = 0.4 >
    A krep = 0.1): Omega izgarada KESIN tekduze artar (-1.107'den -0.061'e)
    ve F_x(atom 1, d = 1e-3) = -1.107 < 0 -- atom d = 0'a cekilir, bariyer
    yok. (A = 0.04, krep = 3) krep > k ama kosul tutar (2.956 > 2.410):
    yine tekduze, F = -0.273 -- dar kosul GEREK kosul DEGIL. Sinif uc
    hucrede de kurulur (parametreleri reddetmez; beyanli). Olculdu
    24.09.2026, sk/duz/olc_b9.log.
    KIRMA (olculdu, sk/duz/mut_md.log): itmeyi E_rep'e ters isaretle
    katan mutant (`mermin.py`) DUVAR hucresini kiriyor (duvar kuyuya
    doner); cokme hucrelerinde Omega yine tekduze (iki terim de negatif
    ustel) ve kuvvet ayni -- o mutanta KORDURLER. Kuvvetin isaretini ters
    ceviren mutant (`mermin.py`) ve radyal yasanin ussunu ters isaretleyen
    mutant (`radyal.py`) iki cokme hucresini kiriyor, duvar hucresine kor.
    Yani uc hucre birlikte baglayicidir, hicbiri tek basina degil.
    """
    kosul = 2.0 * 0.1 * 2.0 * math.exp(2.0) > A * krep * math.exp(krep)
    assert kosul is cokme
    omega, F0 = _dimer_egrisi(UsselRadyalYasa(A, 2.0, krep, 3.0, 3.8))
    assert all(math.isfinite(o) for o in omega)
    if cokme:
        assert all(b > a for a, b in zip(omega, omega[1:])), omega
        assert F0 < 0.0, F0
    else:
        assert omega[0] - omega[_B9_IZGARA.index(2.0)] > 0.0, omega


# ------------------------------------------------------ Kelvin siniri --


def test_KELVIN_TUZAGI_300_hartree_REDDEDILIR_ve_TUZAGIN_BUYUKLUGU():
    """Spec B7: Kelvin sayisi hartree sanilip verilirse kuvvet SESSIZCE kuculur.

    Tuzagin kendisi makinede sessizdir -- `tek_orbital_mermin` kT = 300'u
    kabul eder. Olculen (zincir(16, 2, 0.05, 3), n_e = 16, L = 32, itme
    genligi 0 -> yalniz bant): max|F| kT = 0.002'de 1.053524e-01, kT = 300'de
    2.802562e-05, oran 3759.1. Savunma sinifta: kT_hartree >= 1 hartree
    (315775 K) reddedilir ve mesaj Kelvin'i adiyla soyler. Sinir kesin:
    0.999 kabul, 1.0 red. KIRMA (olculdu): kapi kaldirilinca 300 ve 1.0
    hucreleri sinif kurar -> kirmizi; esik 1e3'e cekilince 300 gecer ->
    kirmizi.
    """
    R = _zincir(16, 2.0, 0.05, 3)
    bant = UsselRadyalYasa(0.0, 2.0, 6.0, 3.0, 3.8)
    dogru = float(np.max(np.abs(tek_orbital_mermin(R, SS_SIGMA, bant, 16.0, 2e-3, 32.0).kuvvet)))
    tuzak = float(np.max(np.abs(tek_orbital_mermin(R, SS_SIGMA, bant, 16.0, 300.0, 32.0).kuvvet)))
    assert abs(dogru - 1.053524e-01) < 1e-6 and abs(tuzak - 2.802562e-05) < 1e-10
    assert dogru / tuzak > 3000.0
    for kT in (300.0, 1.0):
        with pytest.raises(InvalidInputError, match="Kelvin"):
            _alan(kT=kT)
    assert _alan(kT=0.999).kT_hartree == 0.999


def _tb_senaryo(kelvin: object, **ek) -> MDScenario:
    yasa = lambda g, u: dict(genlik_hartree=g, d0_bohr=2.0, us=u, r_on_bohr=3.0, r_cut_bohr=3.8)  # noqa: E731
    params = dict(
        ss_sigma=yasa(-0.1, 2.0), itme=yasa(0.05, 6.0), n_electron=16,
        electron_temperature_kelvin=kelvin, box_x_bohr=32.0,
    )
    params.update(ek)
    return MDScenario(
        name="tb",
        force_field={"kind": "tb_ss_sigma", "params": params},
        initial={"positions_bohr": _zincir(16, 2.0, 0.05, 3).tolist(), "masses_amu": [1.00782503207]},
        thermostat=None,
        dt=20.0,
        n_steps=20,
    )


def test_KELVIN_HARTREE_DONUSUMU_FABRIKADA():
    """Senaryo Kelvin, sinif hartree; donusum `md/factory.py`de (spec 2.1(5), B7).

    Referans (spec B7): 0.002 hartree = 631.6 K, 0.0005 = 157.9 K. Olculen
    631.6 K -> 0.002000158183478562, 157.9 K -> 0.0005000395458696405
    (sabit CODATA k_B / E_h). Ikinci assert donusumun IFADESINI kilitler
    (`kelvin * KB_HARTREE_PER_KELVIN`, bit bit); ilki ondan bagimsiz,
    referans sicakliga karsi. KIRMA (olculdu): donusum silinince (kT =
    Kelvin sayisi) 631.6 >= 1 -> sinif Kelvin kapisindan hata verir;
    ters donusum (Kelvin / k_B) ayni kapiya takilir; k_B x 1.001 degerle
    kirmizi (|kT - 0.002| = 2.16e-06 > 2e-07; bit bit ifade de dusuyor).
    """
    for kelvin, hartree, tol in ((631.6, 0.002, 2e-7), (157.9, 0.0005, 1e-7)):
        _, alan, _ = build(_tb_senaryo(kelvin))
        assert isinstance(alan, SsSigmaTBKuvveti)
        assert abs(alan.kT_hartree - hartree) < tol
        assert alan.kT_hartree == kelvin * KB_HARTREE_PER_KELVIN
    with pytest.raises(InvalidInputError, match="kT_hartree > 0"):
        build(_tb_senaryo(0.0))


def test_TERS_TUZAK_YAKALANMAZ_acik_BEYAN():
    """Hartree sayisini Kelvin diye yazmak (0.002 K) YAKALANMAZ -- beyan kilidi.

    0.002 K = 6.334e-09 hartree: kT > 0 ve < 1, iki kapidan da gecer.
    Bu bir kusur DEGIL bir BEYANDIR (modul dokstringi, B7); bu test beyanin
    dogru kalmasini kilitler -- bir gun bir alt esik eklenirse kirmizi olur
    ve dokstring de guncellenmek zorunda kalir.
    """
    _, alan, _ = build(_tb_senaryo(0.002))
    assert abs(alan.kT_hartree - 6.334e-09) < 1e-12
    assert "YAKALANMAZ" in (tb_kuvvet_modulu.__doc__ or "")


# ------------------------------------------------------------------ M10 --


def _importlar(yol: Path) -> set[str]:
    agac = ast.parse(yol.read_text(encoding="utf-8"), filename=str(yol))
    adlar: set[str] = set()
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.Import):
            adlar.update(a.name for a in dugum.names)
        elif isinstance(dugum, ast.ImportFrom):
            assert dugum.level == 0, "goreli import tarayicidan kacar"
            adlar.add(str(dugum.module))
    return adlar


def test_M10_tb_kuvvet_IMPORTLARI_kilitli():
    """Spec 2.1(4): tb.mermin, tb.radyal, numpy, qlab.errors (+ __future__, dataclasses).

    TAM kume esitligi. Yasak olanlar ayrica adiyla: hf, tb.downfold,
    tb.factory (downfold uzerinden 8 hf modulu yukler, spec 2.2) ve birim
    modulu (md units ekseni 3'te kalmali, spec 5.2; metin sayimi da -- adi
    kaynakta metin olarak da gecmez). KIRMA (olculdu): birim modulu ve
    tb.factory importu eklenmis iki mutant kirmizi.
    """
    adlar = _importlar(TB_KUVVET_DOSYASI)
    assert adlar == {
        "__future__",
        "dataclasses",
        "numpy",
        "qlab.engine.tb.mermin",
        "qlab.engine.tb.radyal",
        "qlab.errors",
    }
    yasak = ("qlab.engine.hf", "qlab.engine.tb.downfold", "qlab.engine.tb.factory", "qlab.units")
    assert [m for m in adlar if any(m == y or m.startswith(y + ".") for y in yasak)] == []
    assert "qlab.units" not in TB_KUVVET_DOSYASI.read_text(encoding="utf-8")


def test_M10_tb_kuvvet_yasanin_IKINCI_KOPYASINI_yazmaz():
    """tb_kuvvet.py'de exp/cos/sin (ve akrabalari) cagrisi SIFIR (spec M10).

    Yasa yalniz `radyal.py`den gelir; on olcumun V4 varyanti yasayi burada
    yeniden yaziyordu. KIRMA (olculdu): `np.exp` cagrisi eklenen mutant
    `[('exp', satir)]` veriyor.
    """
    agac = ast.parse(TB_KUVVET_DOSYASI.read_text(encoding="utf-8"))
    cagrilar = []
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.Call):
            fonk = dugum.func
            ad = fonk.attr if isinstance(fonk, ast.Attribute) else getattr(fonk, "id", "")
            if ad in {"exp", "expm1", "exp2", "cos", "sin", "cosh", "sinh", "power", "expit"}:
                cagrilar.append((ad, dugum.lineno))
    assert cagrilar == []


def test_M10_tb_kuvvet_importu_CALISMA_ZAMANINDA_SIFIR_hf_modulu():
    """Alt surecte `import qlab.engine.md.tb_kuvvet` -> yuklenen qlab modulleri TAM kume.

    Spec 2.2 V4: kenar dosyasinin importu tek basina SIFIR hf modulu
    yukler (`md.factory` forcefields uzerinden 6, `tb.factory` downfold
    uzerinden 8 yukler). `ast` kilidi yalniz dogrudan importu gorur; bu
    test gecisli yuklemeyi olcer. Alt surec ayni PYTHONPATH'i devralir ve
    yukledigi dosyanin bu testin sinadigi dosya oldugunu dogrular. KIRMA
    (olculdu): tb_kuvvet'e forcefields importu eklenince kume 6 hf modulu
    (hf, basis, boys, integrals, scf, stability), tb.factory importu
    eklenince 8 hf modulu ile buyuyor -> kirmizi.
    """
    kod = (
        "import sys, qlab.engine.md.tb_kuvvet as m\n"
        "print(m.__file__)\n"
        "print(','.join(sorted(k for k in sys.modules if k == 'qlab' or k.startswith('qlab.'))))\n"
    )
    cikti = subprocess.run(
        [sys.executable, "-c", kod],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        check=True,
    ).stdout.splitlines()
    assert Path(cikti[0]).resolve() == TB_KUVVET_DOSYASI
    yuklenen = set(cikti[1].split(","))
    assert [m for m in yuklenen if m.startswith("qlab.engine.hf")] == []
    assert yuklenen == {
        "qlab",
        "qlab.engine",
        "qlab.engine.md",
        "qlab.engine.md.tb_kuvvet",
        "qlab.engine.tb",
        "qlab.engine.tb.mermin",
        "qlab.engine.tb.radyal",
        "qlab.errors",
    }


def test_M10_TB_MD_KOSUSUNDA_hf_scf_solve_SIFIR_cagri_HF_POZITIF_KONTROL(monkeypatch):
    """Bagimsizlik CAGRI duzeyinde (spec 2.2): TB MD kosusunda `solve` 0 kez.

    Sayac hem `qlab.engine.hf.scf.solve`a hem de forcefields'in bagladigi
    ada takilir. Pencere: fabrikadan kurulan TB alani, 20 adim NVE.
    POZITIF KONTROL (sayac gercekten sayiyor): ayni sayacla HF alani (H2,
    STO-3G) 1 adim -> 14 cagri (baslangic + 1 adim, her biri 7 SCF;
    spec 2.2'nin olcumuyle ayni). SAYACIN KENDISI OLCULDU: yalniz
    `hf_scf.solve`a takilan sayac HF kontrolunde 0 sayiyor (forcefields
    adi import aninda baglamis) -- ikinci `setattr` bu yuzden sart; o
    olmadan TB'nin 0'i bir sey kanitlamazdi.
    """
    sayac = {"n": 0}
    asil = hf_scf.solve

    def sayan(*a, **k):
        sayac["n"] += 1
        return asil(*a, **k)

    monkeypatch.setattr(hf_scf, "solve", sayan)
    monkeypatch.setattr(forcefields_modulu, "solve", sayan)

    senaryo = _tb_senaryo(631.6)
    durum, alan, _ = build(senaryo)
    yorunge = run(durum, alan, dt=senaryo.dt, n_steps=senaryo.n_steps)
    assert np.all(np.isfinite(yorunge.potential))
    assert sayac["n"] == 0

    hf = MDScenario(
        name="hf",
        force_field={"kind": "hartree_fock", "params": {"symbols": ["H", "H"], "basis": "sto-3g"}},
        initial={
            "positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3959193602]],
            "masses_amu": [1.00782503207, 1.00782503207],
        },
        thermostat=None,
        dt=5.0,
        n_steps=1,
    )
    durum, alan, _ = build(hf)
    run(durum, alan, dt=hf.dt, n_steps=hf.n_steps)
    assert sayac["n"] == 14


# ------------------------------------------------------------------ M11 --

#: Dimer Peierls minimumunun en yuksek x-modu frekansi (on olcum, spec B4'un
#: omega_max'i). Burada bir CIPA DEGIL bir PENCERE dugmesidir: yalniz dt'yi
#: ve pencere uzunlugunu (1.5 T_min) tanimlar; degeri girdiden geri okunur.
_OMEGA_MAX = 0.023147211225190607


def _peierls_minimumu(alan: SsSigmaTBKuvveti) -> np.ndarray:
    y = np.zeros((16, 2))

    def fonk(x):
        omega, kuvvet = alan(np.column_stack([x, y]))
        return omega, -kuvvet[:, 0]

    x0 = _zincir(16, 2.0, 0.02, 1)[:, 0]
    secenek = dict(gtol=1e-10, ftol=1e-16, maxiter=20000)
    sonuc = minimize(fonk, x0, jac=True, method="L-BFGS-B", options=secenek)
    return np.column_stack([sonuc.x, y])


def _tepe_tepe_KE_arti_OMEGA(alan: SsSigmaTBKuvveti, R0: np.ndarray, V0: np.ndarray, wdt: float) -> float:
    dt = wdt / _OMEGA_MAX
    n = int(round(1.5 * 2.0 * math.pi / _OMEGA_MAX / dt))
    durum = MDState(positions=R0.copy(), velocities=V0.copy(), masses=np.full(16, M_H))
    omega, kuvvet = alan(durum.positions)
    toplam = [durum.kinetic_energy() + omega]
    for _ in range(n):
        durum, omega, kuvvet = velocity_verlet_step(durum, kuvvet, alan, dt)
        toplam.append(durum.kinetic_energy() + omega)
    return float(np.max(toplam) - np.min(toplam))


def test_M11_NVE_KE_arti_OMEGA_salinimi_O_dt_KARE():
    """KE + Omega'nin tepe-tepe salinimi dt^2 ile iner: oranlar 4 +- 0.1.

    Pencere (spec M11): N = 16, a = 2, L = 32, n_e = 16, kT = 0.002; x-uzerinde
    L-BFGS Peierls minimumundan (zincir(16, 2, 0.02, 1), gtol 1e-10;
    on kosul -- test dogrular -- max|F_x| < 1e-7) baslangic; 100 K
    x-hizlari (tohum 7), kutle merkezi cikarilmis, M_H; omega_max*dt in
    {0.2, 0.1, 0.05}, pencere 1.5 T_min (329 adim). Olculen tepe-tepe
    6.2430141783e-06 / 1.5544147109e-06 / 3.8853552198e-07, oranlar
    4.016312 ve 4.000702 (prototiple 6 hane ayni). Pencere boyunca
    (T in {10, 1000, 3000} K x tohum {7, 11, 23}) oranlar 3.998 ile 4.028
    arasinda. Baslangic minimumdan olmali (genel baslangicta tt pencere
    boyuyla buyuyor, spec M11).

    KIRMA (olculdu, sk/g4_m11_E_mutant.log): tutarsiz kuvvet (F x 1.001)
    oranlari 5.955 / 2.114'e atiyor -> kirmizi. KOR KALDIGI MUTANT
    (olculdu): Omega yerine E donduren alan (entropi terimi dusurulmus)
    ayni baslangictan 3.998 / 3.924 veriyor -- PENCERE ICINDE. Sebep: bu
    pencerede bosluk kT'nin cok ustunde, kT*ptp(S) = 2.6e-08 << tt; M11'in
    ORAN SINAVI E ile Omega'yi AYIRAMAZ (mutasyon kosusunda bu test o
    mutantta yalniz ON KOSULUNDA kirmizi: mutantin enerjisiyle gevsetme
    max|F_x| = 3.0e-07'de kaliyor). O mutanti sozlesme testleri ve
    asagidaki B1 testi (kT = 0.02, metalik) ORANLA yakalar.
    """
    alan = _alan()
    R0 = _peierls_minimumu(alan)
    assert float(np.max(np.abs(alan(R0)[1][:, 0]))) < 1e-7
    rng = np.random.default_rng(7)
    V0 = np.zeros((16, 3))
    V0[:, 0] = math.sqrt(KB_HARTREE_PER_KELVIN * 100.0 / M_H) * rng.standard_normal(16)
    V0 = MDState(positions=R0, velocities=V0, masses=np.full(16, M_H)).remove_center_of_mass_motion().velocities
    tt = [_tepe_tepe_KE_arti_OMEGA(alan, R0, V0, wdt) for wdt in (0.2, 0.1, 0.05)]
    oran1, oran2 = tt[0] / tt[1], tt[1] / tt[2]
    assert abs(oran1 - 4.0) < 0.1, (oran1, tt)
    assert abs(oran2 - 4.0) < 0.1, (oran2, tt)


def test_B1_KE_arti_E_KORUNMAZ_kT_dS_de_DOYAR_KE_arti_OMEGA_dt_KARE():
    """Spec B1'in DINAMIK kaniti: integrator KE + Omega'yi korur, KE + E'yi DEGIL.

    KE + E = KE + Omega + kT S; KE + Omega'nin salinimi dt^2 ile inerken
    KE + E'ninki kT * ptp(S)'de DOYAR (dt'den bagimsiz). M11 bunu goremez
    (orada kT * ptp(S) = 2.6e-08, bosluk kT'nin cok ustunde); bu yuzden
    pencere METALIK: kT = 0.02, N16 a = 2 L = 32 n_e = 16. Bu kT'de M11'in
    gevsetmesi DUZGUN zincire iniyor (olculdu: baglar [2.0000, 2.0000],
    max|F_x| 2.8e-09) ve ayni sayilari 5 hanede veriyor; test bu yuzden
    dogrudan duzgun zincirden baslar (on kosul -- test dogrular -- simetriden
    max|F| < 1e-12, olculen 1.1e-16). 100 K x-hizlari (tohum 7), kutle
    merkezi cikarilmis, omega_max*dt in {0.2, 0.1, 0.05}, 1.5 T_min.
    Olculen (24.09.2026): tt(KE+Omega) 4.5415e-06 / 1.1304e-06 /
    2.8228e-07 (oranlar 4.0178, 4.0044); tt(KE+E) 4.3838e-05 / 4.0289e-05 /
    3.9440e-05 (oranlar 1.0881, 1.0215); omega_max*dt = 0.05'te
    tt(KE+E) / (kT ptp S) = 1.0072 ve tt(KE+E) / tt(KE+Omega) = 139.7.
    Toleranslar: KE+Omega oranlari 4 +- 0.1 (M11'inki), kimlik 2e-2 (2.8x
    pay), ayrisma > 50 (2.8x pay), KE+E orani < 1.2.

    KIRMA (olculdu, sk/g4_b1_duzgun.log ve g4_mutasyon_2.log): Omega yerine
    E donduren mutant (M11'in kor kaldigi) KE+potansiyel oranlarini
    1.0881 / 1.0215'e cekiyor; Omega'da entropiyi iki kez sayan mutant
    (kuvvet degismez) 0.9136 / 0.9785'e -- ikisi de kirmizi.
    """
    alan = _alan(kT=0.02)
    R0 = _zincir(16, 2.0)
    assert float(np.max(np.abs(alan(R0)[1]))) < 1e-12
    rng = np.random.default_rng(7)
    V0 = np.zeros((16, 3))
    V0[:, 0] = math.sqrt(KB_HARTREE_PER_KELVIN * 100.0 / M_H) * rng.standard_normal(16)
    V0 = MDState(positions=R0, velocities=V0, masses=np.full(16, M_H)).remove_center_of_mass_motion().velocities

    def kosu(wdt: float) -> tuple[float, float, float]:
        dt = wdt / _OMEGA_MAX
        n = int(round(1.5 * 2.0 * math.pi / _OMEGA_MAX / dt))
        son: list = []

        def kaydeden(R):
            # Potansiyel ve kuvvet SINANAN `__call__`dan; E ve S tani()'den
            # (yalniz kayit). tani() ayri cagrilir ki `__call__`daki bir
            # mutant (ornegin Omega yerine E) bu testte gorunsun.
            son.append(alan.tani(R))
            return alan(R)

        durum = MDState(positions=R0.copy(), velocities=V0.copy(), masses=np.full(16, M_H))
        potansiyel, kuvvet = alan(durum.positions)
        s = alan.tani(durum.positions)
        ke_om, ke_e, S = [durum.kinetic_energy() + potansiyel], [durum.kinetic_energy() + s.E_band + s.E_rep], [s.entropi]
        for _ in range(n):
            durum, potansiyel, kuvvet = velocity_verlet_step(durum, kuvvet, kaydeden, dt)
            s = son[-1]
            ke = durum.kinetic_energy()
            ke_om.append(ke + potansiyel)
            ke_e.append(ke + s.E_band + s.E_rep)
            S.append(s.entropi)
        return float(np.ptp(ke_om)), float(np.ptp(ke_e)), 0.02 * float(np.ptp(S))

    (om1, e1, _), (om2, e2, _), (om3, e3, kts3) = (kosu(w) for w in (0.2, 0.1, 0.05))
    assert abs(om1 / om2 - 4.0) < 0.1 and abs(om2 / om3 - 4.0) < 0.1, (om1, om2, om3)
    assert abs(e3 / kts3 - 1.0) < 2e-2, (e3, kts3)
    assert e3 / om3 > 50.0, (e3, om3)
    assert e1 / e2 < 1.2 and e2 / e3 < 1.2, (e1, e2, e3)


# ----------------------------------------------------------- uctan uca --


def test_CLI_TB_SENARYOSU_UCTAN_UCA(tmp_path, capsys):
    """YAML -> loader kapisi -> factory (Kelvin -> hartree) -> integrator -> panel.

    Panel TB'nin KENDI beyanini basar (Mermin, Kelvin/hartree, ss-sigma) ve
    ortak listenin yalanlarini basmaz. Surukleme olcutu G5'ten beri TB'ye
    ozgudur (max(KE) paydasi); o dal asagidaki G5 testlerinde ve sevk edilen
    senaryonun uctan uca testinde sinanir. Bu pencere (20 adim) hukum icin
    DEGIL yalniz boru hatti icin.
    """
    from qlab.run import main

    # `float(x)!r`: numpy 2'de `repr(np.float64)` "np.float64(...)" verir ve
    # YAML onu dizge okur (forcefields.py'deki olculmus tuzagin aynisi).
    konum = ", ".join(f"[{float(x)!r}, 0.0, 0.0]" for x in _zincir(16, 2.0, 0.05, 3)[:, 0])
    yol = tmp_path / "tb.yaml"
    yol.write_text(
        "kind: md\nname: tb uctan uca\n"
        "force_field:\n  kind: tb_ss_sigma\n  params:\n"
        "    ss_sigma: {genlik_hartree: -0.1, d0_bohr: 2.0, us: 2.0, r_on_bohr: 3.0, r_cut_bohr: 3.8}\n"
        "    itme: {genlik_hartree: 0.05, d0_bohr: 2.0, us: 6.0, r_on_bohr: 3.0, r_cut_bohr: 3.8}\n"
        "    n_electron: 16\n    electron_temperature_kelvin: 631.6\n    box_x_bohr: 32.0\n"
        "initial:\n  positions_bohr: [" + konum + "]\n  masses_amu: [1.00782503207]\n"
        "dt: 20.0\nn_steps: 20\n",
        encoding="utf-8",
    )
    cikti = tmp_path / "tb.npz"
    assert main([str(yol), "-o", str(cikti)]) == 0
    assert cikti.exists()
    ekran = capsys.readouterr().out
    assert "klasik MD / velocity-Verlet / tb_ss_sigma" in ekran
    for parca in ("Omega (Mermin)", "electron_temperature_kelvin", "ss-sigma", "kT_hartree=2.000e-03"):
        assert parca in ekran, parca
    assert "elektron yok" not in ekran and "analitik bir model" not in ekran


# ------------------------------------------------- G5 surukleme olcutu --
#
# `run.py` G5'e kadar suruklenmeyi |blok[0]|'a (enerjinin KEYFI sifirina)
# boluyordu; TB'de tekbicim bir onsite c Omega'yi n_e c kaydirir ve yorungeyi
# degistirmez (spec B6, B8). Olcut artik `md/factory.py::energy_drift`te:
# pay ayni (blok farki), TB paydasi max(KE). Olcumler:
# sk/g5_olcum*.py + .log (depoya girmez); sayilar bu pencerelerin.


class _Kaydirilmis:
    """Omega'ya sabit c ekler, kuvvete dokunmaz: tekbicim onsite'in karsiligi.

    Sinif onsite ALMAZ (B6); tekbicim onsite c'nin etkisi Omega + n_e c ve
    ayni kuvvettir -- yuvarlama duzeyinde; bu oncul
    `tests/tb/test_distance_model.py`nin M6 onsite kolunda, AYNI c'lerle
    (+1, -Omega0/16, -16) kilitlidir (c = -16'da Omega <= 2.9e-14, kuvvet
    <= 1.3e-14; |c| ile olcekli tolerans). Sarmalayici bunu tam yapar:
    yorunge BIT BIT ayni kalir (testler dogrular), yalniz potansiyel kayar.
    """

    def __init__(self, alan, c: float):
        self.alan, self.c = alan, c

    def __call__(self, R):
        omega, kuvvet = self.alan(R)
        return omega + self.c, kuvvet


def _ortak(senaryo: MDScenario) -> MDScenario:
    """Ayni kayitlari ORTAK dalin olcutuyle (|blok[0]| paydasi) okumak icin."""
    return dataclasses.replace(senaryo, force_field={"kind": "morse", "params": {}})


def _hukum(senaryo: MDScenario, yorunge) -> tuple[float, bool]:
    """Sürüklenme oranı ve panelin BİRLEŞİK hükmü (D2: TB NVE'de S de girer)."""
    oran = energy_drift(senaryo, yorunge.kinetic, yorunge.potential)
    salinim = energy_oscillation(senaryo, yorunge)
    return oran, provenance_for(
        senaryo, energy_drift=oran, energy_oscillation=salinim
    ).converged


def _suruklenme_hukmu(senaryo: MDScenario, oran: float) -> bool:
    """YALNIZ sürüklenme hükmü: S eşiğin altında (0.0) sabitlenir.

    D2'den beri birleşik hüküm iki ölçütün VE'sidir; bir pencerede ikisi de
    ✗ derse birleşik hükme bakan test sürüklenme eşiğinin mutantına kör
    kalır (Fermi penceresinde 3.16 K: sürüklenme 5.48e-02, S 7.30e-02).
    """
    return provenance_for(senaryo, energy_drift=oran, energy_oscillation=0.0).converged


def test_G5_SURUKLENME_HUKMU_ENERJI_SIFIRINDAN_BAGIMSIZ():
    """Omega + c: yorunge bit bit ayni, TB orani ve HUKMU ayni; eski olcut kayar.

    Pencere: zincir(16, 2, 0.05, 3), n_e = 16, 631.6 K, L = 32, M_H, v0 = 0
    (sevk yolu: `build`), iki kol: dt = 15 x 267 adim ve dt = 40 x 100 adim
    (~4000 a.u.). c in {-Omega0, +16} (-Omega0: blok[0] ~ 0, eski paydanin
    patolojisi; +16: n_e x 1 hartree onsite). Olculen (dt 40 kolu
    g5_olcum7.log; dt 15 kolu D2 dogrulamasi, 09.10.2026): TB orani dt = 15'te 3.8919877720e-04, S
    2.2971250111e-02 (ikisi de GECER); dt = 40'ta 7.0039486469e-02, S
    3.4599e-01 (KALIR); c boyunca bagil fark oranda <= 5.1e-11, S'de <=
    2.9e-13 (yuvarlama). Tolerans 1e-9 bagil. Eski olcut (ortak dal, AYNI
    kayitlar): dt = 15'te c = 0 -> 1.0935e-05 (GECER), c = -Omega0 ->
    5.3519e-02 (KALIR); dt = 40'ta 1.8311e-03 (GECER) -> 1.6733 (KALIR) --
    hukum enerjinin sifirina bagliydi (spec B8'in 4.6e+11'i ayni olgunun
    prototip penceresi).

    D2 (09.10.2026): kol dt = 20 x 200'den dt = 15'e TASINDI. dt = 20'de S
    4.1384e-02, esigin yalniz 1.2x alti; dt = 15 2.2x pay birakir. Hukum
    birlesiktir (suruklenme VE S); S de c'den bagimsizdir ve ayrica sinanir.

    KIRMA (olculdu, sk/g5_mutasyon.log; G5'te dt 20 kolunda): TB paydasi
    |blok[0]|'a geri cevrilince kirmizi (c = -Omega0'da oran 3.3e-05 ->
    9.1e-02; dt 15 kolunda eski olcut 1.09e-05 -> 5.35e-02); TB paydasi
    |ortalama(potansiyel)| yapilinca kirmizi. Ortalama KE paydasi (o da
    sifirdan bagimsiz) burada YESIL kalir; onu asagidaki duragan-olmayan
    test yakalar.
    """
    senaryo = _tb_senaryo(631.6)
    durum, alan, _ = build(senaryo)
    omega0 = alan(durum.positions)[0]
    for dt, n, beklenen in ((15.0, 267, True), (40.0, 100, False)):
        ref = run(durum, alan, dt=dt, n_steps=n)
        ref_oran, ref_hukum = _hukum(senaryo, ref)
        ref_salinim = energy_oscillation(senaryo, ref)
        assert ref_hukum is beklenen, (dt, ref_oran, ref_salinim)
        _, eski_c0 = _hukum(_ortak(senaryo), ref)
        eski = {eski_c0}
        for c in (-omega0, 16.0):
            yor = run(durum, _Kaydirilmis(alan, c), dt=dt, n_steps=n)
            assert np.array_equal(yor.positions, ref.positions)
            assert np.array_equal(yor.kinetic, ref.kinetic)
            assert not np.array_equal(yor.potential, ref.potential)
            oran, hukum = _hukum(senaryo, yor)
            assert abs(oran - ref_oran) < 1e-9 * ref_oran, (dt, c, oran, ref_oran)
            salinim = energy_oscillation(senaryo, yor)
            assert abs(salinim - ref_salinim) < 1e-9 * ref_salinim, (dt, c, salinim)
            assert hukum is beklenen
            eski.add(_hukum(_ortak(senaryo), yor)[1])
        # Pozitif kontrol: AYNI kayitlarda eski olcutun hukmu c ile DEGISIYOR,
        # yani pencere sifir-bagimliligini gercekten gosterebiliyor.
        assert eski == {True, False}, (dt, eski)


def test_G5_ESIK_TB_OZGU_SURUKLENMEYI_YAKALAR_FERMI_GECISI():
    """Kucuk T_e'de Fermi seviyesindeki gecis adimda cozulmuyor: TB KALIR, eski GECER.

    Pencere: zincir(16, 2, 0.05, 3), n_e = 15 (tek: Fermi seviyesinde
    seviye gecisi), L = 32, M_H, v0 = 0, dt = 10 x 400 adim. Olculen
    (g5_olcum6/7.log): T_e = 631.6 K -> 5.972399e-04 (GECER, esigin 16.7x
    alti); T_e = 3.16 K -> 5.480276e-02 (KALIR, 5.5x ustu). Kaynak (sk/
    g5_olcum9.log): SOMO ile komsu seviye adim uclarinda 5.8 kT'ye kadar
    yaklasiyor, adim basina aralik degisimi / kT medyani 349 (631.6 K'de
    1.9) -- spec B2(b)'nin "<~ 1.3-4" kosulunun disi; en buyuk enerji
    sicramasi -1.52e-03 bu yaklasmanin ardinda. dt ile temiz bir kuvvet
    yasasi yok (3.16 K: dt 5/10/20 -> 5.71e-03 / 5.48e-02 / 1.89e-01);
    ayni geometri 631.6 K'de dt = 20'de bile 2.87e-04. Eski olcut
    (ortak dal, ayni kayitlar) ikisinde de GECER: 9.96e-06 ve 1.03e-03 (ortak
    esik 5e-03) -- TB'ye ozgu suruklenmeyi KACIRIYORDU.

    KIRMA (olculdu): esik 1e-1 -> 3.16 K kolu GECER (kirmizi); TB paydasi
    |blok[0]| -> 3.16 K kolu GECER (kirmizi).

    D2 (09.10.2026): 3.16 K kolunda S de esigin ustunde (7.2979e-02;
    631.6 K'de 1.1357e-02). Birlesik hukum o yuzden suruklenme esiginin
    mutantina KORDUR; iddia (suruklenme TB'ye ozgu sicramayi yakalar)
    `_suruklenme_hukmu` ile S esik altinda sabitlenerek okunur.
    """
    for kelvin, beklenen in ((631.6, True), (3.16, False)):
        senaryo = _tb_senaryo(kelvin, n_electron=15)
        durum, alan, _ = build(senaryo)
        yorunge = run(durum, alan, dt=10.0, n_steps=400)
        oran, hukum = _hukum(senaryo, yorunge)
        assert _suruklenme_hukmu(senaryo, oran) is beklenen, (kelvin, oran)
        assert hukum is beklenen, (kelvin, oran)
        assert _hukum(_ortak(senaryo), yorunge)[1] is True, kelvin


def test_G5_DURAGAN_OLMAYAN_KOSU_max_KE_ORTALAMA_KE_DEGIL():
    """Kararsizligin pencere icinde buyudugu kosu: max(KE) GECER, ortalama KE KALIR derdi.

    Pencere: duzgun zincir + 1e-6 bohr gauss gurultu (tohum 5), n_e = 16,
    631.6 K, L = 32, M_H, v0 = 0, dt = 10 x 100 adim (pencerenin kendi yol
    omega_max'iyla omega_max dt ~ 0.27; M11 sabitiyle 0.23 -- ince dt). Yari dolu duzgun zincir Peierls'e kararsiz: KE[0] = 0,
    KE[50] = 7.78e-03, max 2.23e-02. Olculen (g5_olcum7.log): pay/max(KE)
    = 3.851995e-03 (GECER, 2.6x alti), pay/ortalama(KE) = 1.785206e-02 (KALIR) --
    sessiz baslangic ortalamayi seyreltiyor. On kosul (test dogrular):
    pencere iki paydayi GERCEKTEN ayiriyor (ortalama paydayla oran esigin
    ustunde); yoksa test ortalama-KE mutantina kor kalirdi.

    KIRMA (olculdu): TB paydasi ortalama KE -> kirmizi; esik 1e-3 ->
    kirmizi (3.85e-03 > 1e-3).
    """
    R = _zincir(16, 2.0, 1e-6, 5)
    senaryo = dataclasses.replace(
        _tb_senaryo(631.6),
        initial={"positions_bohr": R.tolist(), "masses_amu": [1.00782503207]},
    )
    durum, alan, _ = build(senaryo)
    yorunge = run(durum, alan, dt=10.0, n_steps=100)
    oran, hukum = _hukum(senaryo, yorunge)
    assert yorunge.kinetic[0] == 0.0 and yorunge.kinetic.max() > 1e-2
    ortalama_ile = oran * float(yorunge.kinetic.max()) / float(yorunge.kinetic.mean())
    tolerans = provenance_for(
        senaryo, energy_drift=oran, energy_oscillation=0.0
    ).diagnostics["energy_drift_tolerance"]
    assert ortalama_ile > tolerans, ortalama_ile
    # D2: birlesik hukum de GECER (S 1.3672e-02, buyume evresine ragmen).
    assert hukum is True, oran


_SEVK_EDILEN_TB = Path(__file__).resolve().parents[2] / "scenarios" / "tb_zincir_md.yaml"


def test_G5_SEVK_EDILEN_TB_SENARYOSU_CLI_UCTAN_UCA(tmp_path, capsys):
    """`python -m qlab.run scenarios/tb_zincir_md.yaml -o x.npz`: rc 0, GECER, TB beyanlari.

    Senaryo: zincir(16, 2, 0.05, 3), n_e = 16, 631.6 K, L = 32, 1H, v0 = 0,
    dt = 10, 400 adim, save_every 2 (201 kayit). Olculen (24.09.2026, CLI):
    energy_drift = 4.513e-04 (GECER, esigin 22x alti), duvar saati 0.70 s
    (yorumlayici baslangici dahil); D2 (09.10.2026): salinim S = 1.0054e-02
    her adimdan (GECER, esigin 5x alti; kayitli 201 kareden okunsaydi
    9.69e-03). Panel TB'nin kendi beyanini, surukleme ve salinim paydasini
    ve esiklerini basar; ortak listenin yalanlarini basmaz.
    """
    from qlab.run import main

    cikti = tmp_path / "tb.npz"
    assert main([str(_SEVK_EDILEN_TB), "-o", str(cikti)]) == 0
    assert np.load(cikti)["total"].shape == (201,)
    ekran = capsys.readouterr().out
    # Kaynak ASCII kalsin diye aksanli dizgeler kacis dizisiyle: "Yakinsama"
    # satiri ve onay isareti U+2713, "doga cipasi yok" beyani.
    satirlar = {s.split(":", 1)[0].strip(): s for s in ekran.splitlines() if ":" in s}
    yakinsama = satirlar["Yak\u0131nsama"]
    assert "\u2713" in yakinsama and "energy_drift_scale=max_kinetic" in yakinsama
    for parca in (
        "Omega (Mermin)", "ss-sigma", "do\u011fa \xe7\u0131pas\u0131 yok",
        "kT_hartree=2.000e-03", "max(KE)", "energy_drift_tolerance=1.000e-02",
    ):
        assert parca in ekran, parca
    # D2 (T2): salinim olcutu panelde ve bu kosuda ESIGIN 5x ALTINDA -- olculen
    # S = 1.0054e-02 (her adimdan). Esigi 1e-2'ye ceken mutant burada \u2717 basar.
    for parca in (
        "energy_oscillation=1.005e-02", "energy_oscillation_scale=max_kinetic_all_steps",
        "energy_oscillation_tolerance=5.000e-02",
    ):
        assert parca in yakinsama, parca
    assert "ikisi de e\u015fi\u011fin alt\u0131nda" in satirlar["Yak\u0131nsama tan\u0131s\u0131"]
    assert "elektron yok" not in ekran and "analitik bir model" not in ekran


# ---------------------------------------------------------------------------
# KIRMA KAYDI (24.09.2026, bu makine). Duzenek: src/qlab + reference +
# scenarios + tests + pyproject bir kopyaya alinir, mutant uygulanir, testler
# kopyanin icinden PYTHONPATH=<kopya>/src ile kosar (sabit yol okuyan
# kilitler de mutanti gorur); kopyanin yuklendigi once dogrulanir. Esdeger
# kontrol mutanti (1.0 -> 1.0 + 0.0) 86/86 yesil. Bu dosya icin:
#   tani: yasa takasi / kutu None / n_e + 1 -> sozlesme testi
#   kT > 0 kapisi yok -> kT[sifir], kT[negatif], Kelvin donusumu testi
#   Kelvin kapisi yok / esik 1e3 -> Kelvin tuzagi testi
#   ikinci bekci yok -> ikinci-bekci testi
#   F x 1.001 -> M11 (5.955 / 2.114), B1, sozlesme
#   Omega yerine E -> B1 (1.0881 / 1.0215), sozlesme x2; M11 oranla KOR
#   Omega'da entropi x2 (mermin.py) -> B1 (0.9136 / 0.9785), run sozlesmesi
#   np.exp cagrisi -> ikinci-kopya kilidi ([('exp', 161)])
#   birim modulu / tb.factory / forcefields importu -> ast ve alt surec kilitleri
#     (birim modulunde ayrica test_kova_tablosu units ekseni: md 3 -> 4)
#   mermin importu silindi -> 9 test kirmizi, NameError (kod kenari yuk tasiyor)
#   factory: donusum yok / ters / k_B x 1.001 -> Kelvin donusumu testi (+ CLI)
#   TEST araci: sayac yalniz hf_scf'ye takili -> HF pozitif kontrolu 0 == 14
#
# G5 KIRMA KAYDI (24.09.2026, ayni duzenek, sk/g5_mutasyon.py + .log; bu dosya,
# test_md_scenario.py ve test_run.py birlikte). Esdeger kontrol (esik 1e-2 ->
# 1e-2 + 0.0) 135/135 yesil. factory.energy_drift / provenance_for:
#   TB paydasi |blok[0]| -> sifir-bagimsizlik, Fermi, duragan-olmayan, hareketsiz
#   TB paydasi |ortalama(potansiyel)| -> ayni dort test
#   TB paydasi ortalama KE -> YALNIZ duragan-olmayan (1.785e-02)
#   TB paydasi ptp(Omega) -> 97/97 YESIL: ESDEGER ADAY, beyanli (factory
#     dokstringi: sabit hizla oteleyen sistemde sifir oldugu icin secilmedi)
#   hareketsizde nan yerine 0.0 -> hareketsiz testi
#   esik 1e-1 -> sifir-bagimsizlik (dt 40: 7.0e-02), Fermi (5.48e-02),
#     duragan-olmayan (on kosul), sevk edilen senaryo (panel sayisi)
#   esik 1e-3 -> sifir-bagimsizlik (dt 20: 1.20e-03), duragan-olmayan (3.85e-03),
#     panel testi, sevk edilen senaryo
#   esik 2e-2 -> duragan-olmayan (on kosul) + sevk edilen senaryo; esik 5e-3 ->
#     panel testi (morse karsilastirmasi) + sevk edilen senaryo
#   TB ortak esigi kullanir / taniya baska sayi -> panel testi (+ NVT testi)
#   TB beklenen metni ortak metne / payda cumlesi silindi -> TB panel testi
#     (test_md_scenario) + sevk edilen senaryo
#   kayit kapisi yok -> kapi testi + test_run; sekil kapisi yok -> kapi testi
#   ortak dal |toplam[0]| -> ortak-dal kilidi x4
#   sakli Subscript dali -> sakli-dal kilidi; run.py'ye eski formul -> ast kilidi
#   sevk edilen YAML dt 10 -> 40 -> sevk edilen senaryo testi (1.668e-02 KALIR)
#
# BULGU DUZELTME TURU (24.09.2026; sk/duz/mh.py + mut_md.log). Esdeger kontrol
# (Kelvin esigi 1.0 -> 1.0 + 0.0) 119/119 yesil (bu dosya + test_md_scenario +
# test_tb_beyan).
#   _sayi'dan bool silindi -> kT[bool] (match), GECERSIZ[ne-bool], [L-bool];
#     ilk surumde YESILDI (test-korlugu incelemesinin olcumu: 110/110; kT =
#     True Kelvin kapisina takiliyordu)
#   _sayi'da try/except yok -> GECERSIZ[ne-karmasik] (ham TypeError)
#   kurulus kesmesi yalniz ss_sigma / yalniz itme -> L/2 KURULUSTA (a) / (b)
#   E_rep isareti ters (mermin) -> B9[ortak-us6-DUVAR]
#   kuvvet isareti ters (mermin), radyal ussun isareti ters -> B9 iki cokme
#     hucresi
# ---------------------------------------------------------------------------
