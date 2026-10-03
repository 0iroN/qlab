"""Mesafeye bagli hoplama (`distance_model`) -- spec md-tb ss-sigma G3.

Kapsam: spec 4.2 M6 (kafes yolu == MD yolu, AYNI yasa nesnesi), M7 (sabit
yasa ile geriye uyum), M8 (max_cell ters orgudan); cakisik atom kapisi
(elestirmen E6: `lattice.py` d <= 1e-9 ciftlerini SESSIZCE atiyordu);
`hamiltonian.py` ve `lattice.py` beyanlarinin koda karsi kilitleri.

SURE (olculdu, 24.09.2026, bu makine, Python 3.14.7, numpy 2.5.1): 25 test,
pytest raporu 0.66-0.70 s (uc kosu; toplama ve import dahil). En yavaslar:
BEYAN `ast` taramasi (tum src/qlab) 0.14 s, M8 triklinik 0.03 s, M8 2B taban
disarida ve M6 x2 0.01'er s; digerleri < 0.005 s. Bulgu duzeltme turundan
sonra (ayni gun) 31 test, 0.79 s; M6 onsite kolunun alti hucresi 0.01'er s.
Hicbiri `slow` DEGILDIR (depo emsali 0.3 s'lik bir cipayi `slow` saymiyor,
spec 4.1).

ETIKET: buradaki her test MAKINE testidir (uygulama tutarliligi). Bu kenarda
DOGA CIPASI YOKTUR: yasa turetilmedi, secildi (spec B4). M7 bir GERIYE UYUM
testidir ve MESAFE HATALARINA KORDUR (sabit yasada t mesafeden bagimsizdir;
korluk ayrica olculup kilitlendi). Mesafe yolunu DEGER olarak sinayan M6'dir.

Ortak pencere (spec basi): ss_sigma genlik -0.1, d0 2, us 2; itme genlik
0.05, d0 2, us 6; r_on 3.0, r_cut 3.8.

ARAC TUZAKLARI (spec 4.1): `assert_allclose` ve `pytest.approx` kullanilmaz;
her karsilastirma acik `abs(x - y) < tol`dur ve her tolerans olculmus tabana
baglidir. Beklenen degerler modulun sabitlerinden okunmaz (g_s = 2 literal).
Satirlar konumsal indeksle degil anahtarla ((i, j, cell) kumesi) karsilastirilir.

BAGLAYICILIK OLCULDU (24.09.2026): her tolerans ve kapi en az bir mutantla
kirildi. Mutantlar `src/` agacinin bir kopyasinda `hamiltonian.py` ya da
`lattice.py`ye uygulanip bu dosya `PYTHONPATH=<kopya>/src` ile kosturuldu;
hangi mutantin hangi testi kirdigi ilgili testin dokstringinde ve dosya
sonundaki listede yazili.
"""

from __future__ import annotations

import ast
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

import qlab
import qlab.engine.tb.hamiltonian as hamiltonian_modulu
from qlab.engine.tb.hamiltonian import (
    Hopping,
    TBModel,
    bloch_hamiltonian,
    distance_model,
    uniform_model,
)
from qlab.engine.tb.lattice import (
    Lattice,
    max_cell_for_cutoff,
    neighbour_shells,
)
from qlab.engine.tb.mermin import mermin_dolum, tek_orbital_mermin
from qlab.engine.tb.radyal import UsselRadyalYasa
from qlab.errors import InvalidInputError

# Kilitler KOSAN kodu okur (G1'de olculdu: sabit depo yolu PYTHONPATH
# kopyasindaki mutantlari gormuyordu).
KAYNAK = Path(qlab.__file__).resolve().parent
HAMILTONIAN_DOSYASI = Path(hamiltonian_modulu.__file__).resolve()

R_ON = 3.0
R_CUT = 3.8
SS_SIGMA = UsselRadyalYasa(-0.1, 2.0, 2.0, R_ON, R_CUT)
ITME = UsselRadyalYasa(0.05, 2.0, 6.0, R_ON, R_CUT)


def _zincir(N: int, a: float, genlik: float = 0.0, tohum: int = 0) -> np.ndarray:
    """x ekseninde 1B zincir; istege bagli N(0, genlik^2) x bozulmasi."""
    R = np.zeros((N, 3))
    R[:, 0] = a * np.arange(N)
    if genlik:
        R[:, 0] += genlik * np.random.default_rng(tohum).standard_normal(N)
    return R


def _hex_listesi(model: TBModel) -> list[tuple[int, int, tuple[int, int, int], str]]:
    """Hoplama listesi SIRASIYLA, t bit bit (`float.hex`)."""
    return [(h.i, h.j, h.cell, float(h.t).hex()) for h in model.hoppings]


# --------------------------------------------------------------------- M6 --


def _kafes_yolu(
    R: np.ndarray, L: float, ss_sigma, itme, n_e: float, kT: float, onsite: np.ndarray | None = None
):
    """KAFES YOLU: supercell (x'te L, taban = R), `distance_model` -> H(Gamma).

    Dolum MD yolunun AYNI `mermin_dolum`u; Omega = E_band - kT S + E_rep
    (+ mu (n - n_hesap), `tek_orbital_mermin` ile ayni tanim). Kuvvet
    Hellmann-Feynman: her YONLU hoplama (i, j, cell) icin yer degistirme
    vektoru ORGUDEN yeniden kurulur (`Hopping` yon tasimaz) ve t'(d) ayni
    yasa nesnesinden okunur. Itme ayni `neighbour_shells` listesinden,
    yonlu cift basina 1/2 ile. `onsite` verilmezse sifir; ozdegerler de
    doner (onsite kolu icin).
    """
    N = R.shape[0]
    orgu = Lattice(vectors=np.array([[L, 0.0, 0.0]]), basis=R)
    x = np.array([L, 0.0, 0.0])
    model = distance_model(orgu, np.zeros(N) if onsite is None else onsite, ss_sigma)
    H = bloch_hamiltonian(model, np.zeros(3))
    assert np.all(H.imag == 0.0), "Gamma'da H reel olmali"
    eps, C = np.linalg.eigh(H.real)
    dolum = mermin_dolum(eps, n_e, kT)
    E_band = 2.0 * float(np.sum(dolum.f * eps))
    itme_ciftleri = neighbour_shells(
        orgu, itme.r_cut_bohr, max_cell=max_cell_for_cutoff(orgu, itme.r_cut_bohr)
    )
    phi, dphi = itme.deger_ve_turev(np.array([c.distance for c in itme_ciftleri]))
    E_rep = 0.5 * float(np.sum(phi))
    omega = E_band - kT * dolum.entropi + E_rep
    if kT > 0.0 and dolum.n_hesap != n_e:
        omega += dolum.mu * (n_e - dolum.n_hesap)
    rho = (C * (2.0 * dolum.f)) @ C.T
    gradyan = np.zeros_like(R)
    for h in model.hoppings:
        vek = R[h.j] + h.cell[0] * x - R[h.i]
        d = float(np.linalg.norm(vek))
        _, turev = ss_sigma.deger_ve_turev(d)
        katki = rho[h.i, h.j] * float(turev) * vek / d
        gradyan[h.j] += katki
        gradyan[h.i] -= katki
    for c, dp in zip(itme_ciftleri, dphi):
        vek = R[c.j] + c.cell[0] * x - R[c.i]
        katki = 0.5 * float(dp) * vek / c.distance
        gradyan[c.j] += katki
        gradyan[c.i] -= katki
    return omega, -gradyan, E_band, E_rep, model, eps


@pytest.mark.parametrize("tohum", [3, 5])
def test_M6_KAFES_yolu_MD_yolu_ile_AYNI_yasa_nesnesiyle_AYNI(tohum):
    """distance_model -> bloch_hamiltonian(Gamma) == tek_orbital_mermin (MAKINE).

    Pencere (spec M6): zincir(16, 1.8, 0.05, tohum), L = 28.8, n_e = 16,
    kT = 2e-3; IKI yol ayni `SS_SIGMA` ve `ITME` NESNELERINI ve ayni
    `mermin_dolum`u kullanir. a = 1.8 != d0 BILEREK: t mesafeyle degisir
    (olculen t araligi [-0.1537, -3.2e-05] / [-0.1345, -2.3e-04]) ve ikinci
    komsular KUYRUKTA (16 cift, r_on < d < r_cut); a = 2.0'da kuyrukta cift
    yok. On kosul bunlari testin kendisi dogrular.
    Olculen |dOmega| 2.2e-16 / 0.0, max|dF| 2.6e-16 / 1.5e-16; tolerans 1e-14
    (en kotuye 38x pay). Sure iki yol birlikte ~7 ms.
    KIRMA (olculdu, tohum 3 / 5): distance_model mesafeyi x(1 + e) okuyan
    mutant |dOmega| e = 1e-14'te 4.5e-14 / 4.5e-14, 1e-12'de 4.4e-12,
    1e-9'da 4.4e-09 veriyor -- yani tolerans 1e-14 bagil mesafe hatasini
    cozuyor; t yerine t' yazan 1.3e-02 / 1.1e-02; -t 2.6e-01 / 1.3e-01;
    kesmeyi r_on'dan alan mutant on kosulu (t yayilimi 0.055 / 0.027 < 0.1)
    kiriyor. Hucreyi ters isaretle ya da (i, j)'yi ters yazan mutantlar
    Gamma'daki H'yi DEGISTIRMEZ ama kuvvet yonunu bozar: max|dF| 0.115 /
    0.128. Tam liste dosya sonunda.
    """
    N, a = 16, 1.8
    L = N * a
    R = _zincir(N, a, 0.05, tohum)
    om, F, E_band, E_rep, model, _ = _kafes_yolu(R, L, SS_SIGMA, ITME, 16.0, 2e-3)
    md = tek_orbital_mermin(R, SS_SIGMA, ITME, 16.0, 2e-3, L)

    # on kosul: pencere mesafe yolunu DEGER olarak sinar
    t = np.array([h.t for h in model.hoppings])
    assert float(t.max() - t.min()) > 0.1
    fark = R[None, :, :] - R[:, None, :]
    fark[..., 0] -= L * np.round(fark[..., 0] / L)
    d = np.linalg.norm(fark[np.triu_indices(N, 1)], axis=1)
    assert int(np.sum((d > R_ON) & (d < R_CUT))) >= 8

    assert abs(om - md.omega) < 1e-14
    assert abs(E_band - md.E_band) < 1e-14
    assert abs(E_rep - md.E_rep) < 1e-14
    assert float(np.max(np.abs(F - md.kuvvet))) < 1e-14


@pytest.mark.parametrize("c_adi", ["+1", "-Omega0/16", "-16"])
@pytest.mark.parametrize("tohum", [3, 5])
def test_M6_TEKBICIM_ONSITE_kafes_yolunda_Omega_n_e_c_kayar_kuvvet_ayni(tohum, c_adi):
    """distance_model'in `onsite`i modele gecer; tekbicim c: Omega + n_e c, ayni F (MAKINE).

    M6'nin penceresi (zincir(16, 1.8, 0.05, tohum), L = 28.8, n_e = 16,
    kT = 2e-3); kafes yolu onsite = c ones(N), MD yolu `tek_orbital_mermin`
    (onsite ALMAZ). c degerleri `tests/md/test_tb_kuvvet.py`nin G5
    `_Kaydirilmis` sarmalayicisininkilerle eslesir: +1 (G5'in "+16 = n_e x
    1 hartree"i), -Omega0/16 (G5'in -Omega0'i) ve buyuk |c| kolu -16.
    Uc kol: (i) Omega_kafes(c) - Omega_MD - n_e c, (ii) max|F_kafes(c) -
    F_MD|, (iii) max|eps(c) - eps_MD - c|. Olculen sapma YUVARLAMADIR ve |c|
    ile buyur -- "tam c kayar" DEGIL; toleranslar bu yuzden |c| ile
    olceklidir: (i) 1e-15 n_e max(1, |c|), (ii) 1e-14 max(1, |c|), (iii)
    4e-15 max(1, |c|). Olculen en kotu (iki tohum, 24.09.2026;
    sk/duz/olc_onsite.log): (i) 3.6e-15 (c = +1) / 2.8e-14 (c = -16);
    (ii) 4.2e-15 / 1.2e-14; (iii) 7.8e-16 / 1.4e-14 -- en dar pay (ii)'de
    c = +1: 2.4x. (Pencere disi kontrol: c = +16'da 5.7e-14 / 3.6e-14 /
    7.1e-15.)
    KIRMA (olculdu): distance_model'de onsite'i yok sayan mutant (onsite =
    np.zeros) (i)'de n_e |c| (c = +1'de 16.0), (iii)'te |c| sapma verir --
    mutanti baglayan (i) ve (iii)'tur. (ii) o mutanti KIRMAZ (<= 2.7e-16:
    tekbicim onsite kuvveti degistirmez); (ii)'nin isi G5 `_Kaydirilmis`in
    "ayni kuvvet" onculunu kilitlemektir ve bagi AYRI bir mutantla olculdu:
    onsite'i tekbicim olmayan yapan mutant (onsite x (1 + 1e-3 j)) (ii)'yi
    de kiriyor (3.3e-04 .. 1.2e-01).
    """
    N, a = 16, 1.8
    L = N * a
    R = _zincir(N, a, 0.05, tohum)
    md = tek_orbital_mermin(R, SS_SIGMA, ITME, 16.0, 2e-3, L)
    c = {"+1": 1.0, "-Omega0/16": -md.omega / 16.0, "-16": -16.0}[c_adi]
    om, F, _, _, _, eps = _kafes_yolu(R, L, SS_SIGMA, ITME, 16.0, 2e-3, onsite=c * np.ones(N))
    olcek = max(1.0, abs(c))
    assert abs(om - md.omega - 16.0 * c) < 1e-15 * 16.0 * olcek, om - md.omega - 16.0 * c
    assert float(np.max(np.abs(F - md.kuvvet))) < 1e-14 * olcek
    assert float(np.max(np.abs(eps - np.asarray(md.eps) - c))) < 4e-15 * olcek


# --------------------------------------------------------------------- M7 --

# Sabit yasa: us = 0 -> exp(-0 * x) = 1.0 TAM, r_on = 3.7 -> komsular fc = 1.
SABIT = UsselRadyalYasa(-0.1, 2.0, 0.0, 3.7, 3.8)
_FCC_49 = np.array([[0.0, 2.45, 2.45], [2.45, 0.0, 2.45], [2.45, 2.45, 0.0]])
M7_PENCERELERI = [
    pytest.param(Lattice(vectors=np.array([[2.0, 0, 0]]), basis=np.zeros((1, 3))), id="1B-a2.0"),
    pytest.param(Lattice(vectors=np.array([[1.8, 0, 0]]), basis=np.zeros((1, 3))), id="1B-a1.8"),
    pytest.param(
        Lattice(vectors=np.array([[3.4, 0, 0]]), basis=np.array([[0.0, 0, 0], [1.8, 0, 0]])),
        id="SSH-1.8-1.6",
    ),
    pytest.param(Lattice(vectors=2.5 * np.eye(3), basis=np.zeros((1, 3))), id="kubik-2.5"),
    pytest.param(Lattice(vectors=_FCC_49, basis=np.zeros((1, 3))), id="fcc-4.9"),
]


@pytest.mark.parametrize("orgu", M7_PENCERELERI)
def test_M7_GERIYE_UYUM_sabit_yasa_uniform_model_ile_BIT_BIT_ayni(orgu):
    """us = 0 ve komsu kabuklari [r_on, r_cut] DISINDA -> uniform_model ile ayni liste.

    GERIYE UYUM testidir, MESAFE HATALARINA KORDUR (asagidaki korluk testi).
    Karsilastirma SIRALI ve `float.hex` ile: (i, j, cell, t) birebir.
    Yeni API'de pencere yeniden kuruldu (kesme artik yasadan: r_cut = 3.8,
    r_on = 3.7). Olculen: bes pencerenin besi bit bit ayni; kabuklar
    1B-a2.0 {2.0}, 1B-a1.8 {1.8, 3.6}, SSH {1.6, 1.8, 3.4}, kubik
    {2.5, 3.536}, fcc {3.465}. Bes pencere toplam 2.9 ms.
    On kosul (spec M7 uyarisi): max_cell = 2 bu pencerelerde kabuk
    kaybetmez (asagidaki ikinci assert). 03.10.2026'dan beri `uniform_model`
    de max_cell'i `max_cell_for_cutoff`tan turetir (D12).
    KIRMA (olculdu): -t, t yerine t' ve hucreyi ters isaretle yazan
    mutantlar bes pencerenin besinde; (i, j)'yi ters yazan mutant yalniz
    iki atomlu SSH penceresinde kiriliyor (tek atomlu tabanda i = j,
    mutant ESDEGER). Son ikisi Gamma'daki H'yi degistirmez.
    """
    onsite = np.zeros(orgu.n_atom)
    tam = neighbour_shells(orgu, R_CUT, max_cell=max_cell_for_cutoff(orgu, R_CUT))
    assert all(not (3.7 <= c.distance <= 3.8) for c in tam)
    assert len(neighbour_shells(orgu, R_CUT, max_cell=2)) == len(tam)
    assert _hex_listesi(distance_model(orgu, onsite, SABIT)) == _hex_listesi(
        uniform_model(orgu, onsite, R_CUT, -0.1)
    )


def test_M7_KORLUK_sabit_yasada_YANLIS_mesafe_GORUNMEZ():
    """M7'nin neden mesafe yolunu sinamadigi -- korlugun hala orada oldugunu kilitler.

    Mesafeyi x1.02 ya da x0.5 okuyan bir kurucu, sabit yasada (us = 0, fc = 1
    bolgesinde kalan olcekler: 3.6 x 1.02 = 3.672 < r_on = 3.7) dogru
    kurucuyla BIT BIT ayni listeyi veriyor (olculdu; on-olcumde de 6/6).
    Yani M7'nin yesili mesafe hakkinda hicbir sey soylemez; o is M6'nindir.
    """
    orgu = Lattice(vectors=np.array([[1.8, 0, 0]]), basis=np.zeros((1, 3)))
    dogru = _hex_listesi(distance_model(orgu, np.zeros(1), SABIT))
    ciftler = neighbour_shells(orgu, R_CUT, max_cell=max_cell_for_cutoff(orgu, R_CUT))
    for olcek in (1.02, 0.5):
        t, _ = SABIT.deger_ve_turev(np.array([olcek * c.distance for c in ciftler]))
        yanlis = TBModel(
            lattice=orgu,
            onsite=np.zeros(1),
            hoppings=tuple(
                Hopping(c.i, c.j, c.cell, float(tt)) for c, tt in zip(ciftler, t)
            ),
        )
        assert _hex_listesi(yanlis) == dogru


def test_M7_KONTROL_kuyruk_AKTIFKEN_listeler_AYRISIR():
    """us = 0 ama ikinci kabuk kuyrukta (a = 1.8, r_on = 3.0): ayni OLMAMALI.

    Bit bit karsilastirmanin bos gecmedigini gosterir. Fark kapali bicimden:
    3.6'da fc = sin^2(pi/8), yani |dt| = 0.1 cos^2(pi/8) = 0.1 (2 + sqrt 2)/4
    = 0.0853553390593274 (olculdu, spec M7 ile ayni). Tolerans 1e-15.
    KIRMA (olculdu): kuyruk mesafeye duyarli oldugu icin bu kontrol de
    mesafe mutantini yakaliyor -- x(1 + 1e-14) 5.0e-15 sapma; -t 0.115,
    t yerine t' 0.153, kesmeyi r_on'dan alan mutant 0.0854 (kuyruk cifti
    dusuyor, fark uniform'un kendi t'si oluyor).
    """
    orgu = Lattice(vectors=np.array([[1.8, 0, 0]]), basis=np.zeros((1, 3)))
    kuyruklu = UsselRadyalYasa(-0.1, 2.0, 0.0, R_ON, R_CUT)
    dm = distance_model(orgu, np.zeros(1), kuyruklu)
    un = uniform_model(orgu, np.zeros(1), R_CUT, -0.1)
    assert _hex_listesi(dm) != _hex_listesi(un)
    en_buyuk = max(abs(x.t - y.t) for x, y in zip(dm.hoppings, un.hoppings))
    assert abs(en_buyuk - 0.1 * (2.0 + math.sqrt(2.0)) / 4.0) < 1e-15


def test_M7_pencere_SINIRI_uniform_model_3a_da_KABUK_KAYBETMIYOR():
    """D12 KAPANDI (03.10.2026): uniform_model max_cell'i `max_cell_for_cutoff`tan turetir.

    1B a = 1.2, kesme 3.8: kabuklar 1.2, 2.4, 3.6 = 3a. Eskiden (max_cell = 2
    SABIT) uniform_model 4 komsu buluyordu, distance_model 6; simdi ikisi de
    6. Kayip 2B/3B'de 3|a|'dan ONCE basliyordu ve simetriyi kiriyordu (fcc
    2.449|a|'da 86 yerine 80, C4z 0.398 Ha) -- o pencere asagidaki fcc kolu.
    max_cell = 2 ile `neighbour_shells` hala 4 bulur: pencere duyarlidir.
    """
    orgu = Lattice(vectors=np.array([[1.2, 0, 0]]), basis=np.zeros((1, 3)))
    assert len(neighbour_shells(orgu, R_CUT, max_cell=2)) == 4
    assert len(uniform_model(orgu, np.zeros(1), R_CUT, -0.1).hoppings) == 6
    assert len(distance_model(orgu, np.zeros(1), SABIT).hoppings) == 6
    fcc = Lattice(vectors=0.5 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]), basis=np.zeros((1, 3)))
    kesme = 6**0.5 * 0.5**0.5  # sqrt(6) * en yakin komsu
    tam = neighbour_shells(fcc, kesme, max_cell=max_cell_for_cutoff(fcc, kesme))
    assert len(neighbour_shells(fcc, kesme, max_cell=2)) < len(tam)
    assert len(uniform_model(fcc, np.zeros(1), kesme, -0.1).hoppings) == len(tam)


# --------------------------------------------------------------------- M8 --


def _kaba_kuvvet_komsular(orgu: Lattice, kesme: float, M: int):
    """BAGIMSIZ referans: |n_i| <= M kupundeki HER hucre, vektorize.

    `neighbour_shells` ile ayni kabul kurali (bagil tolerans 1e-9; kendisi
    haric) ama ayri kod yolu ve ayri (buyuk, sabit) hucre araligi.
    """
    d = orgu.dimension
    hucreler = np.array(list(itertools.product(range(-M, M + 1), repeat=d)), dtype=float)
    kayma = hucreler @ orgu.vectors  # (n_hucre, 3)
    taban = orgu.basis
    fark = taban[None, None, :, :] + kayma[:, None, None, :] - taban[None, :, None, :]
    mesafe = np.linalg.norm(fark, axis=-1)  # (hucre, i, j)
    secili = (mesafe > 1e-9) & (mesafe <= kesme * (1.0 + 1e-9))
    anahtar = []
    for h, i, j in zip(*np.nonzero(secili)):
        hucre = tuple(int(v) for v in hucreler[h]) + (0,) * (3 - d)
        anahtar.append((int(i), int(j), hucre))
    return sorted(anahtar)


_EGIK_15 = np.array([[1.0, 0.0, 0.0], [math.cos(math.pi / 12), math.sin(math.pi / 12), 0.0]])
_TRIKLINIK = np.array([[3.0, 0.0, 0.0], [0.7, 2.4, 0.0], [0.3, 0.9, 4.1]])
M8_PENCERELERI = [
    pytest.param(_EGIK_15, np.zeros((1, 3)), 0.79, 3, 4, 12, id="2B-15derece-tek-atom"),
    pytest.param(
        _EGIK_15, np.array([[0.0, 0.0, 0.0], [2.3, 0.4, 0.0]]), 0.79, 4, 18, 16,
        id="2B-15derece-taban-DISARIDA",
    ),
    pytest.param(
        _TRIKLINIK,
        np.array([[0.0, 0.0, 0.0], 2.7 * _TRIKLINIK[0] - 1.3 * _TRIKLINIK[2] + [0.11, -0.07, 0.05]]),
        4.5, 4, 34, 9,
        id="triklinik-taban-DISARIDA",
    ),
]


@pytest.mark.parametrize(("vektorler", "taban", "kesme", "gereken", "iki_hucrede", "M"), M8_PENCERELERI)
def test_M8_max_cell_TERS_ORGUDEN_hicbir_komsu_KAYBOLMAZ(vektorler, taban, kesme, gereken, iki_hucrede, M):
    """distance_model'in (i, j, cell) kumesi == bagimsiz kaba-kuvvet referansi.

    Pencere (spec M8): 15 derece egik 2B orgu (|a_1| = |a_2| = 1), kesme
    0.79; en kisa orgu vektoru a_2 - a_1 (0.261) ve komsular (-3, 3) hucresine
    uzaniyor. Olculen: naif max_cell = ceil(kesme/min|a|) + 1 = 2 ile 4 komsu,
    ters orgu formulu (max_cell 4) ile 6, referans (max_cell 12 ve 20) 6.
    Ek pencereler (spec "triklinik ve tabani hucre disina tasmis orguler"):
    taban atomu hucrenin disinda; max_cell 2 ile 18/30 ve 34/50 cift
    buluyor, ters formul (13 ve 6) hepsini. Formul UST SINIRDIR, siki degil
    (gereken en buyuk |hucre| 3, 4, 4). On kosul: referans M ve M + 4'te
    ayni (yakinsamis) ve max_cell = 2 GERCEKTEN kaybediyor (pencere duyarli).
    Sure: uc pencere toplam ~0.05 s (triklinik ~0.03 s, referans dahil).
    KIRMA (olculdu): naif formul ve sabit max_cell = 2 mutantlari uc
    pencerenin ucunde; ters orguyu 2 pi yerine 4 pi'ye bolen mutant tek
    atomlu 2B ve triklinik pencerede; taban YAYILIMINI unutan mutant YALNIZ
    triklinik-taban-DISARIDA penceresinde (bu pencere o yuzden var).
    HAYATTA KALAN (olculdu, beklenen): ceil yerine floor -- sinir gevsek
    oldugu icin uc pencerede de kayip uretmiyor; test "kayip yok"u sinar,
    sinirin degerini degil.
    """
    orgu = Lattice(vectors=vektorler, basis=taban)
    referans = _kaba_kuvvet_komsular(orgu, kesme, M)
    assert referans == _kaba_kuvvet_komsular(orgu, kesme, M + 4)
    assert max(max(abs(v) for v in hucre) for _, _, hucre in referans) == gereken
    assert len(neighbour_shells(orgu, kesme, max_cell=2)) == iki_hucrede < len(referans)

    yasa = UsselRadyalYasa(-0.1, 0.5, 2.0, 0.5 * kesme, kesme)
    model = distance_model(orgu, np.zeros(orgu.n_atom), yasa)
    assert sorted((h.i, h.j, h.cell) for h in model.hoppings) == referans
    assert max_cell_for_cutoff(orgu, kesme) >= gereken


def test_M8_YETMEYEN_acik_max_cell_REDDEDILIR_yetenler_AYNI_listeyi_verir():
    """Acik max_cell ters orgu sinirindan kucukse hata (sessiz kabuk kaybi yok).

    Egik 2B pencerede sinir 4; max_cell = 2 kabul edilseydi 4/6 komsu
    donerdi (olculdu). Sinir ve ustu ayni sirali listeyi verir.
    KIRMA (olculdu): sinir kapisini kaldiran mutant burada kiriliyor.
    """
    orgu = Lattice(vectors=_EGIK_15, basis=np.zeros((1, 3)))
    yasa = UsselRadyalYasa(-0.1, 0.5, 2.0, 0.5, 0.79)
    sinir = max_cell_for_cutoff(orgu, 0.79)
    with pytest.raises(InvalidInputError, match="max_cell"):
        distance_model(orgu, np.zeros(1), yasa, max_cell=2)
    with pytest.raises(InvalidInputError, match="max_cell"):
        distance_model(orgu, np.zeros(1), yasa, max_cell=sinir - 1)
    varsayilan = _hex_listesi(distance_model(orgu, np.zeros(1), yasa))
    assert _hex_listesi(distance_model(orgu, np.zeros(1), yasa, max_cell=sinir)) == varsayilan
    assert _hex_listesi(distance_model(orgu, np.zeros(1), yasa, max_cell=sinir + 3)) == varsayilan


# ------------------------------------------------------- cakisik atom kapisi --

_CAKISIK_ORGULER = [
    pytest.param(
        Lattice(vectors=np.array([[4.0, 0, 0]]), basis=np.array([[0.0, 0, 0], [0.0, 0, 0]])),
        id="iki-taban-ayni-yerde",
    ),
    pytest.param(
        Lattice(vectors=np.array([[4.0, 0, 0]]), basis=np.array([[0.0, 0, 0], [4.0, 0, 0]])),
        id="taban-bir-GORUNTUYLE-cakisik",
    ),
    pytest.param(
        Lattice(vectors=np.array([[0.3, 0, 0], [0, 5.0, 0]]), basis=np.array([[0.0, 0, 0], [0.1 * 3, 0, 0]])),
        id="goruntu-YUVARLAMA-5.6e-17",
    ),
    pytest.param(
        Lattice(vectors=np.array([[4.0, 0, 0]]), basis=np.array([[0.0, 0, 0], [5e-10, 0, 0]])),
        id="esik-ALTI-5e-10",
    ),
]


@pytest.mark.parametrize("orgu", _CAKISIK_ORGULER)
def test_CAKISIK_atom_UC_kurucuda_da_HATA(orgu):
    """Cakisik taban atomlari ya da goruntuleri InvalidInputError (spec 2.1(3)).

    Kapi `neighbour_shells`te: iki kurucu (uniform_model, distance_model)
    ayni kapidan gecer. Kapinin ORADA kurulabilecegi olculdu: sessiz atmayi
    hataya cevirmek `tests/tb`'nin 342 testinin hicbirini kirmadi.
    `0.1 * 3` ornegi goruntu cakismasinin kayan noktada TAM 0 olmadigini
    (5.55e-17) gosterir; MD yolunun `d == 0.0` kapisi bunu gormezdi.
    KIRMA (olculdu): kapiyi `continue`ya geri ceviren mutant dort ornegin
    dordunde; esigi 0.0'a indiren mutant 5e-10 ve 5.55e-17 orneklerinde;
    "atomun kendisi"ni yalniz `cell == 0` ile taniyan mutant ayni-yerde ve
    5e-10 orneklerinde kiriliyor.
    """
    for kur in (
        lambda: neighbour_shells(orgu, 3.0),
        lambda: uniform_model(orgu, np.zeros(orgu.n_atom), 3.0, -0.1),
        lambda: distance_model(orgu, np.zeros(orgu.n_atom), SS_SIGMA),
    ):
        with pytest.raises(InvalidInputError, match="cakisik atom"):
            kur()


def test_CAKISIK_atom_KENDI_goruntusuyle_PATOLOJIK_orgu():
    """|a| = 5e-10 <= esik: her atom kendi goruntusuyle cakisir -> hata.

    Kapi "atomun kendisi"ni (i = j VE cell = 0) yalniz sifir hucrede atar;
    `i == j` tek basina yetmez. KIRMA (olculdu): kosulu `if i == j:`e
    indiren mutant yalniz bu ornekte yakalaniyor (diger her orgude esdeger).
    max_cell = 2 ile `neighbour_shells` cakismayi gorur. `uniform_model` ve
    `distance_model` max_cell'i `max_cell_for_cutoff`tan alir (~6e9): eskiden
    `distance_model`in Python dongusu pratikte bitmiyordu (~96 GB'lik aralik
    tahsisi); 03.10.2026'dan beri ikisi de hucre sayisi kapisina ANINDA takilir.
    """
    orgu = Lattice(vectors=np.array([[5e-10, 0.0, 0.0]]), basis=np.zeros((1, 3)))
    with pytest.raises(InvalidInputError, match="cakisik atom"):
        neighbour_shells(orgu, 3.0)
    with pytest.raises(InvalidInputError, match="hucre sayisi"):
        uniform_model(orgu, np.zeros(1), 3.0, -0.1)
    with pytest.raises(InvalidInputError, match="hucre sayisi"):
        distance_model(orgu, np.zeros(1), SABIT)
    # |a| = 1e-300: ters orgu sonsuza tasar -- ham OverflowError degil.
    asiri = Lattice(vectors=np.array([[1e-300, 0.0, 0.0]]), basis=np.zeros((1, 3)))
    with pytest.raises(InvalidInputError, match="ters orgu sonlu degil"):
        uniform_model(asiri, np.zeros(1), 3.0, -0.1)


def test_HUCRE_SAYISI_KAPISI_mesru_en_buyugu_gecirir_siniri_asani_reddeder():
    """`_HUCRE_SINIRI` = 1e7 hucre, (2 max_cell + 1)^d -- ATOM SAYISINA BAKMAZ.

    OLCULDU: depo testlerinde en buyuk 2197 hucre (3B, max_cell 6); 1 derece egik
    2B orgu kesme 8|a|'da 844561. Sinirin hemen ustu (1B, max_cell 5000000 ->
    10000001 hucre) taramadan ONCE reddedilir. max_cell TAMSAYI olmali (bool ve
    float reddedilir); np.uint8 dar tipi int'e cevrilir -- eskiden -max_cell
    tasip SESSIZCE bos sonuc veriyordu (03.10 dogrulayicisi).
    """
    iki_atom = Lattice(vectors=np.eye(3) * 2.0, basis=np.array([[0.0, 0, 0], [1.0, 0, 0]]))
    assert len(neighbour_shells(iki_atom, 1.0, max_cell=6)) > 0  # 13^3 = 2197 hucre
    zincir = Lattice(vectors=np.array([[1.0, 0, 0]]), basis=np.zeros((1, 3)))
    with pytest.raises(InvalidInputError, match="hucre sayisi .* = 10000001 > 10000000"):
        neighbour_shells(zincir, 1.0, max_cell=5_000_000)
    for kotu in (2.9, 2.0, True, float("inf")):
        with pytest.raises(InvalidInputError, match="tamsayi"):
            neighbour_shells(zincir, 1.0, max_cell=kotu)
    alanlar = lambda ciftler: [(c.i, c.j, c.cell, c.distance) for c in ciftler]  # NeighbourPair eq=False
    beklenen = alanlar(neighbour_shells(zincir, 1.0, max_cell=3))
    assert len(beklenen) == 2 and alanlar(neighbour_shells(zincir, 1.0, max_cell=np.uint8(3))) == beklenen
    assert len(neighbour_shells(zincir, 1.0, max_cell=np.uint8(255))) == 2


@pytest.mark.slow
def test_HUCRE_SAYISI_KAPISI_SUPERCELLI_REDDETMEZ():
    """Ilk surumun kusuru (03.10 dogrulayicisi): kapi hucre x n_atom^2'ye bakiyordu
    ve HEAD'in hesapladigi mesru supercell'leri reddediyordu (1B N >= 448, 2B 15x15,
    3B 5^3, Si 3x3x3 216 atom). 600 atomlu 1B supercell: max_cell 2 -> 5 hucre;
    eski yuk 5 * 600^2 = 1.8e6 > 1e6 RED derdi, hucre kapisi gecirir. Her atomun
    iki komsusu (+-2 bohr) -> 1200 yonlu hoplama.
    """
    n = 600
    zincir = Lattice(vectors=np.array([[2.0 * n, 0, 0]]), basis=np.array([[2.0 * k, 0, 0] for k in range(n)]))
    model = uniform_model(zincir, np.zeros(n), 3.8, -0.1)
    assert len(model.hoppings) == 2 * n

def test_CAKISMA_esigi_ESKI_sessiz_atma_esigi_ve_MD_yolu_ile_uyum():
    """Esik 1e-9 bohr: eskiden ATILAN her cift artik hata, TUTULAN her cift aynen tutulur.

    Pencere (elestirmen E6): 8 atomlu zincir, a = 2, L = 16, n_e = 8,
    kT = 2e-3; atom 1 = atom 0 + d (x). OLCULDU: eski sessiz atma d <= 1e-9'da
    kafes yolunun Omega'sini -0.5318024310 yapiyordu, MD yolu +18.3188990775;
    fark -18.85 hartree. d = 2e-9'da cift tutuluyordu ve iki yol ayniydi
    (|dOmega| 3.6e-15, Omega ~ 18.3; tolerans 1e-13). Yeni: 1e-9 hata,
    2e-9 eskisi gibi. MD yolu yalniz d == 0.0'i reddeder: 0 < d <= 1e-9
    bandinda kafes yolu hata verir, MD yolu hesaplar (beyanli fark).
    KIRMA (olculdu): esigi 0.0'a indiren mutant d = 1e-9'da hata vermiyor;
    1e-8'e cikaran mutant d = 2e-9'da hata veriyor; distance_model mesafeyi
    x(1 + 1e-13) okuyan mutant 2e-9 kolunda |dOmega| 1.5e-13 veriyor.
    """
    for d, hata_beklenir in ((1e-9, True), (2e-9, False)):
        R = _zincir(8, 2.0)
        R[1, 0] = R[0, 0] + d
        if hata_beklenir:
            with pytest.raises(InvalidInputError, match="cakisik atom"):
                _kafes_yolu(R, 16.0, SS_SIGMA, ITME, 8.0, 2e-3)
            # MD yolu bu bantta hesaplar (beyanli fark)
            tek_orbital_mermin(R, SS_SIGMA, ITME, 8.0, 2e-3, 16.0)
        else:
            om = _kafes_yolu(R, 16.0, SS_SIGMA, ITME, 8.0, 2e-3)[0]
            md = tek_orbital_mermin(R, SS_SIGMA, ITME, 8.0, 2e-3, 16.0)
            assert abs(om - md.omega) < 1e-13


def test_ATOMUN_KENDISI_haric_tutulmaya_DEVAM_eder():
    """Kapi atomun kendisini (i = j, cell = 0, d = 0) cakisma saymaz.

    Tek atomlu zincirde yalniz +-1 hucresi (test_lattice ile ayni sayi).
    """
    zincir = Lattice(vectors=np.array([[2.5, 0.0, 0.0]]), basis=np.zeros((1, 3)))
    assert sorted(c.cell[0] for c in neighbour_shells(zincir, 2.5)) == [-1, 1]


# --------------------------------------------------------- girdi kapilari --


def test_distance_model_GECERSIZ_girdi_REDDEDILIR():
    """Her ornek yalniz kendi kapisinin yakalayacagi bicimde kurulur.

    Yasa turu: yalniz `UsselRadyalYasa` (ayni nesnenin iki yolda kullanilmasi
    sozlesmesi; cagrilabilir bir fonksiyon kabul edilmez). max_cell tamsayi
    olmali (bool degil). Kesme pozitif (max_cell_for_cutoff kapisi).
    """
    orgu = Lattice(vectors=np.array([[2.0, 0, 0]]), basis=np.zeros((1, 3)))
    with pytest.raises(InvalidInputError, match="UsselRadyalYasa"):
        distance_model(orgu, np.zeros(1), lambda d: -0.1)
    for kotu in (True, 2.5, "3"):
        with pytest.raises(InvalidInputError, match="tamsayi"):
            distance_model(orgu, np.zeros(1), SS_SIGMA, max_cell=kotu)
    with pytest.raises(InvalidInputError, match="kesme"):
        max_cell_for_cutoff(orgu, -1.0)
    with pytest.raises(InvalidInputError, match="kesme"):
        max_cell_for_cutoff(orgu, math.nan)
    # kontrol: numpy tamsayisi kabul edilir
    distance_model(orgu, np.zeros(1), SS_SIGMA, max_cell=np.int64(5))


# ---------------------------------------------- beyanlar ve import kilidi --


def _tuketiciler_ve_okuyucular() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """src/qlab'da (1) `neighbour_shells(...)` CAGIRAN ve (2) `.distance` OKUYAN
    (dosya, fonksiyon) ciftleri. Tek gecis; `rglob` (alt paketler dahil)."""
    cagiranlar: set[tuple[str, str]] = set()
    okuyanlar: set[tuple[str, str]] = set()
    for yol in sorted(KAYNAK.rglob("*.py")):
        agac = ast.parse(yol.read_text(encoding="utf-8"), filename=str(yol))
        for fonk in ast.walk(agac):
            if not isinstance(fonk, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dugum in ast.walk(fonk):
                if isinstance(dugum, ast.Call):
                    f = dugum.func
                    cagrilan = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                    if cagrilan == "neighbour_shells":
                        cagiranlar.add((yol.name, fonk.name))
                elif (
                    isinstance(dugum, ast.Attribute)
                    and dugum.attr == "distance"
                    and isinstance(dugum.ctx, ast.Load)
                ):
                    okuyanlar.add((yol.name, fonk.name))
    return cagiranlar, okuyanlar


def test_BEYAN_neighbour_shells_IKI_tuketicisi_ve_distance_TEK_okuyucusu():
    """`hamiltonian.py` beyanlari koda karsi (elestirmen P3.16: bayatlayan beyan).

    Eski beyan "`neighbour_shells`in TEK tuketicisi uniform_model" idi ve
    distance_model'le YANLIS oldu; `NeighbourPair.distance` src'de 0 kez
    okunuyordu (spec 2.1(3)). Simdi: iki tuketici, tek okuyucu. Sayim `ast`
    ile ve KOSAN paketten (`qlab.__file__`). Dokstring ikisini de adlandirir.
    Not: `max_cell_for_cutoff` bir tuketici DEGILDIR (cagirmaz).
    """
    cagiranlar, okuyanlar = _tuketiciler_ve_okuyucular()
    assert cagiranlar == {
        ("hamiltonian.py", "uniform_model"),
        ("hamiltonian.py", "distance_model"),
    }
    assert okuyanlar == {("hamiltonian.py", "distance_model")}
    belge = uniform_model.__doc__ or ""
    assert "TEK tuketicisi" not in belge
    assert "IKI tuketicisinden" in belge and "`distance_model`" in belge


def test_BEYAN_distance_model_YALNIZ_ss_sigma_satiri_SK_modeli_DEGIL():
    """Spec 1'in zorunlu beyani ve `uniform_model` dokstringindeki SK cumlesinin yeni bicimi.

    (Spec bu cumleyi degisiklik oncesi satir numarasiyla anar; dokstring
    buyudukce kaydigi icin numara burada BILEREK yazilmaz.)
    """
    belge = " ".join((distance_model.__doc__ or "").split())
    for parca in (
        "Slater-Koster tablosunun YALNIZ ss-sigma satiri",
        "aci carpani OZDES 1, bu bir SK modeli DEGILDIR",
        "p/d yok",
        "'-t' konvansiyonu kullanilmaz",
        "KESME YASADAN TURETILIR",
        "DOGA CIPASI YOKTUR",
    ):
        assert parca in belge, parca
    uniform = " ".join((uniform_model.__doc__ or "").split())
    assert "ss-sigma satiri `distance_model`dedir" in uniform
    assert "Yone bagli Slater-Koster (p/d orbitalleri, yon kosinusleri) depoda YOKTUR" in uniform
    assert "Plan G" not in uniform


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


def test_hamiltonian_IMPORTLARI_kilitli_HF_DOWNFOLD_FACTORY_BIRIM_yok():
    """hamiltonian.py radyal'i import eder; hf/downfold/tb.factory/birim modulu ETMEZ.

    Tam kume esitligi (spec 2.2: tb.factory downfold uzerinden 8 hf modulu
    yukler; spec 5.2: birim modulu tb'nin units derecesini 2'den 3'e cikarir).
    """
    assert _importlar(HAMILTONIAN_DOSYASI) == {
        "__future__",
        "dataclasses",
        "numpy",
        "qlab.engine.tb.lattice",
        "qlab.engine.tb.radyal",
        "qlab.errors",
    }


# ------------------------------------------------------------------------------
# KIRMA KANITLARI (olculdu, 24.09.2026; bu dosya + test_lattice.py +
# test_hamiltonian.py, src kopyasi + PYTHONPATH; 26 mutant, biri esdeger
# kontrol; 25 gercek mutantin 24'u kirildi):
#
#   mesafe x(1 + 1e-14 | 1e-13 | 1e-12 | 1e-9)   -> M6 [3][5], M7 KONTROL
#                                                   (1e-13 ve ustu: CAKISMA esigi de)
#   t yerine t' ; -t                              -> M6, M7 (5/5), M7 KONTROL, KORLUK
#   kesme = r_on                                  -> M6 on kosulu, M7 KONTROL, M8 (3/3), YETMEYEN
#   hucre ters isaretli                           -> M6 (kuvvet), M7 (5/5), KORLUK, M8 taban-DISARIDA x2
#   (i, j) ters                                   -> M6 (kuvvet), M7 SSH, M8 taban-DISARIDA x2
#   max_cell naif ; max_cell = 2 sabit            -> M8 (3/3), YETMEYEN (sabit 2: M7 SINIRI da)
#   ters orgu / 4 pi                              -> M8 tek-atom + triklinik, YETMEYEN, M7 SINIRI
#   taban yayilimi yok                            -> M8 triklinik-taban-DISARIDA (YALNIZ)
#   max_cell kapisi yok ; tamsayi kapisi yok ; yasa turu kapisi yok -> YETMEYEN ; GECERSIZ girdi (x2)
#   cakisik -> continue (eski davranis)           -> CAKISIK (4/4), KENDI goruntusu, CAKISMA esigi
#   esik 0.0 ; esik 1e-8                          -> CAKISIK 5e-10 + yuvarlama, KENDI goruntusu,
#                                                    CAKISMA ; CAKISMA
#   "kendisi" = yalniz cell == 0                  -> CAKISIK ayni-yerde + 5e-10, CAKISMA
#   "kendisi" = yalniz i == j                     -> KENDI goruntusuyle PATOLOJIK (YALNIZ)
#   dokstringe "TEK tuketicisi" ; yeni tuketici   -> BEYAN (iki tuketici)
#   hamiltonian.py'ye birim modulu importu        -> import kilidi
#
# HAYATTA KALAN: (1) esdeger kontrol mutanti (kesme = float(r_cut)); (2) ceil
# yerine floor (sinir gevsek, kayip yok -- M8 dokstringi).
# Duzenek: scratchpad sk/g3_mutasyon.py (depoya girmez).
#
# BULGU DUZELTME TURU (24.09.2026; sk/duz/mut_ek.log). Ilk surumde 20 cagrinin
# 20'si sifir onsite veriyordu ve `onsite=onsite`i `np.zeros`a ceviren mutant
# butun tests/tb'de YESILDI (test-korlugu incelemesi). Simdi:
#   onsite yok sayilir (np.zeros)   -> M6 onsite kolu 6/6 (Omega ve eps kollari;
#                                      tests/tb'nin geri kalani 357 yesil)
#   onsite kademeli (x (1 + 1e-3 j)) -> M6 onsite kolu 6/6 (kuvvet kolu da)
