"""Dusuk TB kod maddeleri (11.10.2026): D16, E4, H0-1, H0-2 kapilari; D1 ve D17 beyan kilitleri.

Her kapinin testi HEAD 80eb21c'de KIRMIZIDIR (olculdu, kamu fonksiyonlari):
- D16: `mermin_dolum([-1, 0, 1, 2], 4, 1e308)` -> f = [nan x 4], mu = nan,
  tipli hata yok (yalniz RuntimeWarning).
- E4: `bloch_hamiltonian` k + 1e12 G'de ozdegeri 2.0e-04 eV kaydiriyordu,
  uyarisiz. Kapinin ilk olcusu (kesirli koordinat) iki delik birakiyordu --
  egik taban ve atom gauge'unda dik k; olcu terim basina Kartezyen carpim
  buyuklugune cevrildi (dogrulama turu, ayni gun), delik testleri asagida.
- H0-1: `neighbour_shells` (i, j, R) ve (j, i, -R) mesafesini 1 ulp ayri
  hesapliyordu; kesme o pencereye dusunce `uniform_model` kendi listesini
  H0 kapisinda reddediyor ve sucu kullaniciya yukluyordu.
- H0-2: R = 0'da 1e11 hartree'lik terim olcegi buyutup TEK YONLU 0.05
  hartree'lik R != 0 hoplamasini kabul ettiriyordu (bant genisligi 0.1 -> 0.05).
"""

from __future__ import annotations

import ast
import inspect
import math
import warnings

import numpy as np
import pytest

from qlab.engine.md import tb_kuvvet
from qlab.engine.tb import mermin
from qlab.engine.tb.bands import band_structure
from qlab.engine.tb.hamiltonian import Hopping, TBModel, bloch_hamiltonian, uniform_model
from qlab.engine.tb.lattice import Lattice, max_cell_for_cutoff, neighbour_shells, reciprocal_vectors
from qlab.engine.tb.mermin import mermin_dolum, tek_orbital_mermin
from qlab.engine.tb.radyal import UsselRadyalYasa
from qlab.errors import InvalidInputError

# ---------------------------------------------------------------------- D16 --


def test_D16_DEV_kT_nan_DEGIL_tipli_hata_ve_sinirin_alti_DOGRU():
    """Parantez siniri tasinca (kT y > 1.8e308) InvalidInputError; 6.5e307 hala sonlu ve dogru.

    kT >> bant: f = 1/2 her seviyede, S = g_s m ln 2 = 8 ln 2 (n = 4, m = 4).
    Uyari da cikmaz (`float(y)`: Python float'i uyarisiz tasar).
    """
    eps = [-1.0, 0.0, 1.0, 2.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for kT in (7e307, 1e308, 1.79e308):
            with pytest.raises(InvalidInputError, match="parantezi kayan nokta"):
                mermin_dolum(eps, 4, kT)
        d = mermin_dolum(eps, 4, 6.5e307)
    assert list(np.asarray(d.f)) == [0.5] * 4 and d.n_hesap == 4.0
    assert d.entropi == pytest.approx(8.0 * math.log(2.0), rel=1e-14)


# ----------------------------------------------------------------------- E4 --


def _iki_orbitalli_zincir() -> TBModel:
    """E4 olcum modeli: a = 2.5, taban 0 ve 1.1, ilk ve ikinci komsu (HARTREE)."""
    orgu = Lattice(vectors=np.array([[2.5, 0, 0]]), basis=np.array([[0, 0, 0], [1.1, 0, 0]]))
    liste = [(0, 1, 0, -0.06), (1, 0, 0, -0.06), (1, 0, 1, -0.045), (0, 1, -1, -0.045),
             (0, 0, 1, -0.01), (0, 0, -1, -0.01)]
    return TBModel(
        lattice=orgu, onsite=np.array([-0.1, 0.05]),
        hoppings=tuple(Hopping(i, j, (c, 0, 0), t) for i, j, c, t in liste),
    )


@pytest.mark.parametrize("gauge", ["cell", "atom"])
def test_E4_DEV_k_SESSIZ_kayma_DEGIL_tipli_hata(gauge):
    """k0 + 1e12 G: faz hatasi siniri asilir -> InvalidInputError (eskiden 2.0e-04 eV kayma)."""
    model = _iki_orbitalli_zincir()
    G = reciprocal_vectors(model.lattice)[0]
    k0 = np.array([0.37, 0.0, 0.0])
    with pytest.raises(InvalidInputError, match="k cok buyuk"):
        bloch_hamiltonian(model, k0 + 1e12 * G, gauge=gauge)
    for kotu in (np.nan, np.inf):
        with pytest.raises(InvalidInputError, match="sonlu"):
            bloch_hamiltonian(model, np.array([kotu, 0.0, 0.0]), gauge=gauge)


def test_E4_atom_gauge_TABAN_kolu_R0_hoplamasinda_da_SAYILIR():
    """Yalniz R = 0 hoplamasi: hucre gauge'unda faz YOK (H k'dan bagimsiz, dev k zararsiz),
    atom gauge'unda faz k . (tau_j - tau_i) -- dev k'de ayni yuvarlama, kapi oraya da bakar."""
    orgu = Lattice(vectors=np.array([[2.5, 0, 0]]), basis=np.array([[0, 0, 0], [1.1, 0, 0]]))
    model = TBModel(lattice=orgu, onsite=np.zeros(2),
                    hoppings=(Hopping(0, 1, (0, 0, 0), -0.06), Hopping(1, 0, (0, 0, 0), -0.06)))
    dev = np.array([1e12, 0.0, 0.0])
    assert bloch_hamiltonian(model, dev).tobytes() == bloch_hamiltonian(model, np.zeros(3)).tobytes()
    with pytest.raises(InvalidInputError, match="k cok buyuk"):
        bloch_hamiltonian(model, dev, gauge="atom")


def _egik_kare(N: float) -> TBModel:
    """Indirgenmemis taban a2 = (N a, a), a = 2.5; ilk komsu hoplamalari (HARTREE)."""
    orgu = Lattice(vectors=np.array([[2.5, 0, 0], [2.5 * N, 2.5, 0]]), basis=np.zeros((1, 3)))
    hop = []
    for c, t in (((1, 0, 0), -0.0367), ((0, 1, 0), -0.0184)):
        hop += [Hopping(0, 0, c, t), Hopping(0, 0, tuple(-x for x in c), t)]
    return TBModel(lattice=orgu, onsite=np.zeros(1), hoppings=tuple(hop))


def test_E4_INDIRGENMEMIS_taban_ilk_bolgede_KABUL_uzak_k_EGIK_carpimla_RED():
    """a2 = (N a, a), N = 3e4. Olcu terim basina Kartezyen carpim buyuklugu.

    Ilk bolgedeki izgara kabul: P ~ 4 pi N |c| = 3.6e5 < 8.2e5 (gercek hata
    1.9e-12 hartree; kaba Kartezyen sinir max|k| sum|c| ||a||_1 burada 5e9'a
    cikip izgarayi reddederdi). k 1e5 hucre oteyken RED -- ilk olcu (kesirli
    koordinat, 1e5 x 2 pi = 6.3e5) bunu GECIRIYORDU: k_x a_2,x ile k_y a_2,y
    birbirini goturur, yuvarlamalari goturmez (olculdu: faz hatasi 1.5e-06 rad,
    tam faz referansi; dogrulayicinin k'sinde 1.68e-06).
    """
    model = _egik_kare(30000)
    b = reciprocal_vectors(model.lattice)
    for f in ((0.0, 0.0), (0.35, 0.95), (0.95, 0.5)):
        E = float(np.linalg.eigvalsh(bloch_hamiltonian(model, f[0] * b[0] + f[1] * b[1]))[0])
        ref = sum(2 * t * math.cos(2 * math.pi * f[e]) for e, t in ((0, -0.0367), (1, -0.0184)))
        assert abs(E - ref) < 1e-10
    with pytest.raises(InvalidInputError, match="k cok buyuk"):
        bloch_hamiltonian(model, (0.95 + 1e5) * b[0] + 0.5 * b[1])


def test_E4_atom_gauge_DUZLEM_DISI_taban_farki_DIK_k_REDDEDILIR_hucre_gauge_TAM():
    """1B orgu, tau_1 = (1.1, 0, 0.7), k = (0.37, 0, 1e14).

    Atom gauge'unda faz k_z tau_z icerir: ilk olcu (ters orgu vektorlerine
    izdusum) k_z'yi GORMUYORDU ve 2.3e-03 eV sessiz kayma geciyordu (olculdu,
    faz hatasi 3.7e-03 rad). Hucre gauge'unda kayma_z TAM 0: H k_z'den bit bit
    bagimsiz, kapi da reddetmez.
    """
    orgu = Lattice(vectors=np.array([[2.5, 0, 0]]), basis=np.array([[0, 0, 0], [1.1, 0.0, 0.7]]))
    liste = [(0, 1, 0, -0.06), (1, 0, 0, -0.06), (1, 0, 1, -0.045), (0, 1, -1, -0.045)]
    model = TBModel(lattice=orgu, onsite=np.array([-0.1, 0.05]),
                    hoppings=tuple(Hopping(i, j, (c, 0, 0), t) for i, j, c, t in liste))
    dik = np.array([0.37, 0.0, 1e14])
    with pytest.raises(InvalidInputError, match="k cok buyuk"):
        bloch_hamiltonian(model, dik, gauge="atom")
    duz = np.array([0.37, 0.0, 0.0])
    assert bloch_hamiltonian(model, dik).tobytes() == bloch_hamiltonian(model, duz).tobytes()
    # Sinirin alti (k_z tau_z = 7e5 < 8.2e5) atom gauge'unda da kabul.
    bloch_hamiltonian(model, np.array([0.37, 0.0, 1e6]), gauge="atom")


def _kare_capraz(capraz: bool = True) -> TBModel:
    """a = (2.5, 3.0) dikdortgen; (1,0), (0,1) ve istege bagli (1,1) capraz hoplama."""
    orgu = Lattice(vectors=np.array([[2.5, 0, 0], [0, 3.0, 0]]), basis=np.zeros((1, 3)))
    hucreler = [(1, 0, 0), (0, 1, 0)] + ([(1, 1, 0)] if capraz else [])
    hop = []
    for c in hucreler:
        hop += [Hopping(0, 0, c, -0.03), Hopping(0, 0, tuple(-x for x in c), -0.03)]
    return TBModel(lattice=orgu, onsite=np.zeros(1), hoppings=tuple(hop))


def test_E4_2B_IKINCI_eksende_dev_k_REDDEDILIR():
    """k yalniz ikinci eksende uzak: (0, 1) hoplamasinin P'si 2 pi 2e5 = 1.26e6 > 8.2e5."""
    model = _kare_capraz()
    b = reciprocal_vectors(model.lattice)
    with pytest.raises(InvalidInputError, match="k cok buyuk"):
        bloch_hamiltonian(model, 0.3 * b[0] + 2e5 * b[1])


def test_E4_capraz_hoplamanin_P_si_BILESENLERIN_TOPLAMI():
    """k = 1e5 (b0 + b1): (1,0) ve (0,1) terimleri P = 2 pi 1e5 = 6.3e5 (kabul);
    capraz (1,1) teriminde |k_x| a + |k_y| b = 1.26e6 -> RED. Capraz terim
    yokken ayni k kabul (olcu terim basina; tek eksen ya da max degil, TOPLAM)."""
    k = 1e5 * reciprocal_vectors(_kare_capraz().lattice).sum(axis=0)
    with pytest.raises(InvalidInputError, match=r"k cok buyuk.*cell=\[-?1, -?1, 0\]"):
        bloch_hamiltonian(_kare_capraz(), k)
    bloch_hamiltonian(_kare_capraz(capraz=False), k)


def test_E4_esik_2_KAT_icinde():
    """Tek orbitalli zincir: P = 2 pi N. N = 1.2e5 (7.5e5) kabul, 2.6e5 (1.6e6) red."""
    orgu = Lattice(vectors=np.array([[2.5, 0, 0]]), basis=np.zeros((1, 3)))
    m = TBModel(lattice=orgu, onsite=np.zeros(1),
                hoppings=(Hopping(0, 0, (1, 0, 0), -0.03), Hopping(0, 0, (-1, 0, 0), -0.03)))
    G = reciprocal_vectors(orgu)[0]
    bloch_hamiltonian(m, 1.2e5 * G)
    bloch_hamiltonian(m, -1.2e5 * G)
    for isaret in (1.0, -1.0):
        with pytest.raises(InvalidInputError, match="k cok buyuk"):
            bloch_hamiltonian(m, isaret * 2.6e5 * G)


def test_E4_olcu_FORMULU_mesajda_bagimsiz_hesapla_AYNI():
    """Mesajdaki sayi = max_h sum_x |k_x| (sum_e |c_e| |a_e,x| [+ |tau_j,x| + |tau_i,x|]).

    Negatif bilesenli egik 3B orgu, orijinden uzak taban, negatif k bilesenleri:
    mutlak deger, i/j taban terimleri ve eksen toplami ayri ayri gorunur.
    """
    orgu = Lattice(
        vectors=np.array([[2.1, -0.4, 0.3], [-0.9, 2.6, 0.2], [0.5, -0.7, 3.1]]),
        basis=np.array([[0.3, -0.2, 0.1], [-1.4, 0.9, 2.2]]),
    )
    liste = [(0, 1, (1, -2, 0), -0.05), (1, 0, (-1, 2, 0), -0.05),
             (1, 1, (0, 1, -3), -0.02), (1, 1, (0, -1, 3), -0.02)]
    model = TBModel(lattice=orgu, onsite=np.zeros(2),
                    hoppings=tuple(Hopping(i, j, c, t) for i, j, c, t in liste))
    k = np.array([-2.0e5, 7.0e4, -1.3e5])
    for gauge in ("cell", "atom"):
        P = 0.0
        for i, j, c, _ in liste:
            M = [sum(abs(c[e]) * abs(orgu.vectors[e][x]) for e in range(3)) for x in range(3)]
            if gauge == "atom":
                M = [M[x] + abs(orgu.basis[j][x]) + abs(orgu.basis[i][x]) for x in range(3)]
            P = max(P, sum(abs(k[x]) * M[x] for x in range(3)))
        with pytest.raises(InvalidInputError, match="k cok buyuk") as hata:
            bloch_hamiltonian(model, k, gauge=gauge)
        assert f"= {P:.3e} >" in str(hata.value), (gauge, P, str(hata.value))


def test_E4_hoplamasiz_model_her_k_da_kosegen():
    """Hoplama yok: carpim listesi bos, olcu 0 (kapi `max`'in bos dizide cokmesine dusmez)."""
    orgu = Lattice(vectors=np.array([[2.5, 0, 0]]), basis=np.zeros((1, 3)))
    m = TBModel(lattice=orgu, onsite=np.array([-0.2]), hoppings=())
    for gauge in ("cell", "atom"):
        assert bloch_hamiltonian(m, np.array([1e300, 0.0, 0.0]), gauge=gauge)[0, 0] == -0.2


def test_E4_kapinin_ALTINDA_hata_BEYANLI_sinirda():
    """k0 + 1e4 G kabul ve ozdeger farki <= 1e-9 x sum|t| (beyanli faz hatasi siniri).

    Olculen fark 3.9e-13 hartree (1.0e-11 eV); sinir 2.3e-10. Ust sinir: kapi
    k + 1e5 G'yi de kabul eder (faz olcegi 6.3e5 < 8.2e5), 1e6 G'yi reddeder.
    """
    model = _iki_orbitalli_zincir()
    G = reciprocal_vectors(model.lattice)[0]
    k0 = np.array([0.37, 0.0, 0.0])
    E0 = np.linalg.eigvalsh(bloch_hamiltonian(model, k0))
    E1 = np.linalg.eigvalsh(bloch_hamiltonian(model, k0 + 1e4 * G))
    assert float(np.max(np.abs(E1 - E0))) <= 1e-9 * sum(abs(h.t) for h in model.hoppings)
    bloch_hamiltonian(model, k0 + 1e5 * G)
    with pytest.raises(InvalidInputError, match="k cok buyuk"):
        bloch_hamiltonian(model, k0 + 1e6 * G)


# -------------------------------------------------------------------- H0-1 --

#: Asimetrik cift tasiyan orgu (olculdu): (1, 1, (-1, 1)) eski sirada
#: 2.6400757564888173, (1, 1, (1, -1)) 2.640075756488817 bohr.
_EGIK = Lattice(
    vectors=np.array([[2.3, 0.0, 0.0], [0.7, 2.1, 0.0]]),
    basis=np.array([[0.0, 0.0, 0.0], [0.37, 0.29, 0.0]]),
)


def _ters(cift_anahtari):
    i, j, hucre = cift_anahtari
    return (j, i, tuple(-c for c in hucre))


@pytest.mark.parametrize(
    "orgu",
    [
        _EGIK,
        Lattice(vectors=np.array([[2.7, 0, 0], [1.1, 2.3, 0]]), basis=np.array([[0, 0, 0], [0.53, 0.61, 0]])),
        Lattice(vectors=np.diag([2.1, 2.9, 3.3]), basis=np.array([[0, 0, 0], [0.7, 1.3, 0.9], [1.9, 0.2, 2.4]])),
    ],
)
def test_H0_1_ters_yonun_mesafesi_BIT_BIT_ayni(orgu):
    ciftler = neighbour_shells(orgu, 4.0, max_cell=max_cell_for_cutoff(orgu, 4.0))
    mesafe = {(c.i, c.j, c.cell): c.distance for c in ciftler}
    assert len(mesafe) > 20
    farkli = [(k, d, mesafe[_ters(k)]) for k, d in mesafe.items() if mesafe[_ters(k)] != d]
    assert farkli == []


def test_H0_1_ULP_kesmesinde_uniform_model_TEK_YONLU_liste_kurmaz():
    """Kesme, eski siradaki iki mesafenin ARASINA (sinir = kucugu) konur.

    HEAD: `uniform_model` kendi listesini "her hoplama iki yonde yazilmali"
    diye reddediyordu. Simdi iki yon ya birlikte girer ya birlikte kalir.
    """
    v, b = _EGIK.vectors, _EGIK.basis
    K = (-1) * v[0] + 1 * v[1]
    eski = (float(np.linalg.norm(b[1] + K - b[1])), float(np.linalg.norm(b[1] + (-K) - b[1])))
    assert eski[0] != eski[1]  # pencere gercek
    kucuk = min(eski)
    kesme = kucuk / (1.0 + 1e-9)
    for _ in range(8):
        sinir = kesme * (1.0 + 1e-9)
        if sinir == kucuk:
            break
        kesme = np.nextafter(kesme, np.inf if sinir < kucuk else -np.inf)
    assert kesme * (1.0 + 1e-9) == kucuk
    model = uniform_model(_EGIK, np.zeros(2), float(kesme), -0.1)
    anahtarlar = {(h.i, h.j, h.cell) for h in model.hoppings}
    assert all(_ters(k) in anahtarlar for k in anahtarlar)


# -------------------------------------------------------------------- H0-2 --


def test_H0_2_ESI_YOK_hoplama_DEV_R0_terimiyle_de_REDDEDILIR():
    """Esi yazilmamis sifirdan farkli anahtar olcekten bagimsiz red; esi VAR olan asimetri olcege gore."""
    orgu = Lattice(vectors=np.array([[2.5, 0, 0]]), basis=np.array([[0, 0, 0], [1.2, 0, 0]]))
    dev = (Hopping(0, 1, (0, 0, 0), 1e11), Hopping(1, 0, (0, 0, 0), 1e11))
    with pytest.raises(InvalidInputError, match="Hermityen degil.*YOK"):
        TBModel(lattice=orgu, onsite=np.zeros(2), hoppings=dev + (Hopping(0, 0, (1, 0, 0), -0.05),))
    iki_yon = dev + (Hopping(0, 0, (1, 0, 0), -0.05), Hopping(0, 0, (-1, 0, 0), -0.05))
    model = TBModel(lattice=orgu, onsite=np.zeros(2), hoppings=iki_yon)
    k = np.array([[x, 0, 0] for x in np.linspace(-np.pi / 2.5, np.pi / 2.5, 201)])
    alt = band_structure(model, k)[:, 0]
    assert float(alt.max() - alt.min()) == pytest.approx(0.1, rel=1e-2)  # yarilanmis olsa 0.05; eigh gurultusu ~eps x 1e11
    # Esi VAR, degeri 1e-13 bagil farkli: olcege gore tolerans (beyan) -- kabul.
    TBModel(lattice=orgu, onsite=np.zeros(2),
            hoppings=dev + (Hopping(0, 0, (1, 0, 0), -0.05), Hopping(0, 0, (-1, 0, 0), -0.05 * (1 + 1e-13))))


def test_H0_2_ESI_YOK_COK_KUCUK_hoplama_da_REDDEDILIR_tam_SIFIR_kabul():
    """Esi yok dalinda olcut TAM sifirdir, mutlak tolerans degil: tek yonlu 1e-14 red, 0.0 kabul."""
    orgu = Lattice(vectors=np.array([[2.5, 0, 0]]), basis=np.zeros((1, 3)))
    with pytest.raises(InvalidInputError, match="Hermityen degil"):
        TBModel(lattice=orgu, onsite=np.zeros(1), hoppings=(Hopping(0, 0, (1, 0, 0), 1e-14),))
    TBModel(lattice=orgu, onsite=np.zeros(1), hoppings=(Hopping(0, 0, (1, 0, 0), 0.0),))


# ----------------------------------------------------- D1 / D17 beyan kilitleri --


def test_D1_BEYANI_kuvvet_bastirilmasi_kT_bolu_GENLIGE_bagli():
    """Sabit hartree esigi neden yok: oran F(kT = |g|)/F(kT = 1e-3 |g|) her genlikte AYNI (0.51336).

    `tb_kuvvet.py` modul dokstringindeki D1 beyaninin sayisi; olcek degismezligi
    matematikte tamdir (H = g H1), sayisal fark ~1e-15.
    """
    R = np.zeros((16, 3))
    R[:, 0] = 2.0 * np.arange(16) + 0.05 * np.random.default_rng(3).standard_normal(16)
    itmesiz = UsselRadyalYasa(0.0, 2.0, 6.0, 3.0, 3.8)
    oranlar = []
    for g in (0.01, 0.1, 1.0):
        ss = UsselRadyalYasa(-g, 2.0, 2.0, 3.0, 3.8)
        F = [float(np.max(np.abs(tek_orbital_mermin(R, ss, itmesiz, 16.0, kT, 32.0).kuvvet)))
             for kT in (g, 1e-3 * g)]
        oranlar.append(F[0] / F[1])
    assert oranlar == pytest.approx([oranlar[0]] * 3, rel=1e-12)
    assert abs(oranlar[0] - 0.51336) < 1e-5


def _govde(fonksiyon) -> str:
    """Dokstringsiz govde, ad haric (`ast.dump`)."""
    tanim = ast.parse(inspect.getsource(fonksiyon)).body[0]
    govde = tanim.body[1:] if isinstance(tanim.body[0], ast.Expr) and isinstance(
        getattr(tanim.body[0], "value", None), ast.Constant
    ) else tanim.body
    return ast.dump(ast.Module(body=govde, type_ignores=[])) + ast.dump(tanim.args)


def test_D17_BILINCLI_KOPYA_tb_kuvvet_sayi_ile_mermin_skaler_IKIZ():
    """Import edilmez (gerekce `tb_kuvvet._sayi` dokstringinde); biri degisip oteki kalirsa kirmizi."""
    assert _govde(tb_kuvvet._sayi) == _govde(mermin._skaler)
