"""`t`'nin ISARETI ne zaman gozlenebilir -- ve neden bir CIPA DEGILDIR.

SINIF: TRANSKRIPSIYON DENETIMI. Bu dosya bir teoremi kilitler; hicbir fizik
olcmez ve oyle beyan edilmez.

NEDEN VAR
---------
Cekirdek spec §7'nin 3. acik karari "t'nin isaretini sabitleyen sinav"di.
Plan G on olcumunde onerilen aday (ozvektorun dugum sayisini HF'in taban
MO'suna capalamak) sahte-cipa sinavinda dustu. Ikinci bir aday da olculdu ve
o da dustu; bu dosya IKINCI ADAYIN OLUM RAPORUDUR ve karari kapatir:

    Iki-parcali (bipartite) bir orguде t'nin isareti bir AYAR SECIMIDIR.
    Bir "isaret sinavi" ancak tek-parite bir cevrim varsa anlamlidir --
    ve orada bile HF'i sinamaz.

AYAR DONUSUMU NE YAPAR (olculdu, en onemli duzeltme)
----------------------------------------------------
D = diag((-1)^i) donusumu H uzerinde "HALKA MESAFESI TEK olan TUM elemanlari
cevir" islemidir; yalnizca en yakin komsu (NN) elemanlarini degil. Gercek,
YOGUN bir H_eff'te (HF'ten Lowdin indirgemesiyle, duzgun cokgen halka,
yaricap 1.7 bohr, STO-3G) olculdu:

    N     yalniz NN cevrilince   tum tek-mesafe cevrilince   D H D
    4         0.000000e+00              0.000000e+00       0.000000e+00
    6         2.216264e-02              0.000000e+00       0.000000e+00
    8         2.244237e-01              0.000000e+00       0.000000e+00
   12         7.648898e-01              0.000000e+00       0.000000e+00

ve |D H D - (tum tek-mesafe cevrilmis H)| = 0.000e+00 dort N'de de. N=4
OZELDIR: halka mesafeleri {1,2} ve tek olan yalnizca 1 = NN, yani iki kume
cakisir.

Sonuc: "t'nin isareti hicbir gozlenebiliri degistirmez" onermesi NN-KESILMIS
model icin dogru, gercek yogun H_eff icin YANLISTIR. Bu dosya iki durumu da
ayri ayri kilitler; ikisini karistirmak bu depoda bir kez yapildi.

DOGRU DEGISMEZ
--------------
"Cevrimin uzunlugu" DEGIL, "cevrimde gecilen NN kenar sayisinin PARITESI".
Karsi-ornek olculdu (saf cebir, t1 ve t3 hoplamali ACIK zincir): yalnizca t1
cevrilince spektrum kayiyor (n=4 2.888624e-01 ... n=12 5.432251e-01), t1 VE
t3 birlikte cevrilince 0.000000e+00. Acik zincirin hic cevrimi yoktur, yani
"tek uzunlukta cevrim" olcutu burada bir sey soylemez ama isaret yine de
gozlenebilir hale gelir.

Ayar-degismez buyukluk, kapali bir cevrim boyunca hoplamalarin CARPIMIDIR.

NEDEN CIPA DEGIL (bu kismi TEST ETMIYORUZ, OLCUP YAZIYORUZ)
-----------------------------------------------------------
"Tek halkada cevrim carpimi negatiftir" hukmu FIZIK TASIMAZ:

  - Fizik TAMAMEN cikarilinca da tutuyor. S_fake[i,j] = exp(-a*d_ij),
    kosegen 1, H_eff = S_fake^-1 -- Hamiltoniyen yok, elektron yok, integral
    yok, baz seti yok. N = 3,5,7,9,11 x a = 0.3,0.5,1,2,4 -> **25/25 geciyor**
    (N=3, a=0.5: cevrim -8.4989e-03, taban katliligi 1).
  - SCF gereksiz: cekirdek Hamiltoniyen tek basina, hatta -S tek basina ayni
    isareti veriyor (N=3,5,7'de olculdu).
  - Kok mekanizma temsil secimi: dolasik S icin
    (S^(-1/2))[0,1] = [(1+2s)^(-1/2) - (1-s)^(-1/2)]/3, s>0 iken CEBIRSEL
    olarak daima negatif; STO-3G 1s kontraksiyonunun katsayilari pozitif
    oldugu icin s>0 zorunludur.
  - N=3'te cipa ile gozlenebilir AYNI SAYIDIR: t = (e1-e2)/3 ve
    cevrim = (e1-e2)^3/27; sekiz yaricapta oran 1.000000000. TB'nin iki
    parametresi H3+'in iki ayrik ozdegerine oturur, kalan serbestlik
    derecesi SIFIRDIR -- yani "TB, HF ozdegerlerini 1.2e-15 ile veriyor"
    ifadesi bir sinav degil, bir yeniden okumadir.
  - `t = H_eff[0,1]` Lowdin disinda TANIMSIZDIR: kanonik ortogonallestirmede
    (X = V s^(-1/2), tamamen mesru) butun HF gozlenebilirleri 1e-15'te ayni
    kalir ama t OZDES SIFIR olur (N=3 ve N=7'de ~1e-49).

Bu yuzden asagidaki testlerin HICBIRI "isaret dogru cikti" diye bir sey
iddia etmez; yalnizca ayar cebrini kilitler.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from qlab.engine.tb.hamiltonian import (
    Hopping,
    TBModel,
    bloch_hamiltonian,
    finite_ring_matrix,
)
from qlab.engine.tb.lattice import Lattice

T = 0.7


def _nn_modeli(t: float, eps: float = 0.0) -> TBModel:
    """1B, tek orbitalli, YALNIZCA en yakin komsu hoplamali model."""
    lattice = Lattice(
        vectors=np.array([[1.0, 0.0, 0.0]]),
        basis=np.array([[0.0, 0.0, 0.0]]),
    )
    return TBModel(
        lattice=lattice,
        onsite=np.array([eps]),
        hoppings=(Hopping(0, 0, (1, 0, 0), t), Hopping(0, 0, (-1, 0, 0), t)),
    )


def _acik_zincir(n: int, t1: float, t3: float = 0.0) -> np.ndarray:
    """Ucu ACIK zincir: hic cevrimi yoktur."""
    H = np.zeros((n, n))
    for i in range(n - 1):
        H[i, i + 1] = H[i + 1, i] = t1
    for i in range(n - 3):
        H[i, i + 3] = H[i + 3, i] = t3
    return H


def _gauge(n: int) -> np.ndarray:
    return np.diag([(-1.0) ** i for i in range(n)])


def _spektrum(H: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh(H))


def _cevrim_carpimi(H: np.ndarray, n: int) -> float:
    carpim = 1.0
    for i in range(n):
        carpim *= H[i, (i + 1) % n]
    return carpim


# --- 1. NN-KESILMIS modelde isaret AYAR ----------------------------------


@pytest.mark.parametrize("n", [4, 6, 8, 12])
def test_NN_MODELINDE_cift_halkada_isaret_bir_AYAR_secimidir(n):
    """OLCULDU: D H(t) D - H(-t) = 0.000e+00 TAM; spektrum farki 0.000e+00.

    BEYAN: bu hukum NN-KESILMIS model icindir. Gercek yogun H_eff icin
    YANLISTIR ve o durum test_GERCEK_HEFF_te_YALNIZ_NN_cevirmek_YETMIYOR
    tarafindan ayrica kilitlenmistir.
    """
    art = finite_ring_matrix(_nn_modeli(-T), n)
    eksi = finite_ring_matrix(_nn_modeli(+T), n)
    D = _gauge(n)
    assert np.abs(D @ art @ D - eksi).max() == 0.0
    assert np.abs(_spektrum(art) - _spektrum(eksi)).max() == 0.0


@pytest.mark.parametrize("n", [3, 5, 7, 12])
def test_NN_MODELINDE_acik_zincir_uzunluk_ne_olursa_olsun_AYAR(n):
    """Acik zincirin cevrimi yoktur; TEK uzunlukta bile ayar vardir.

    Plan G'nin butun aday sinavlari acik/periyodik ZINCIR uzerindeydi --
    hepsinin isarete kor cikmasinin sebebi budur.
    """
    D = _gauge(n)
    art, eksi = _acik_zincir(n, -T), _acik_zincir(n, +T)
    assert np.abs(D @ art @ D - eksi).max() == 0.0
    assert np.abs(_spektrum(art) - _spektrum(eksi)).max() == 0.0


@pytest.mark.parametrize("n", [3, 5, 7, 9])
def test_NN_MODELINDE_tek_halkada_HICBIR_ayar_isareti_ceviremiyor(n):
    """TUKETICI arama: 2^n adet +-1 kosegen ayarin hicbiri t -> -t yapmiyor."""
    art = finite_ring_matrix(_nn_modeli(-T), n)
    eksi = finite_ring_matrix(_nn_modeli(+T), n)
    for isaretler in itertools.product((1.0, -1.0), repeat=n):
        assert np.abs(np.diag(isaretler) @ art @ np.diag(isaretler) - eksi).max() > 1e-12


@pytest.mark.parametrize(
    "n,fark", [(3, 1.400), (5, 0.8652), (7, 0.6231), (9, 0.4862)]
)
def test_NN_MODELINDE_tek_halkada_spektrum_GERCEKTEN_farkli(n, fark):
    """OLCULDU (t = -+0.7): N=3 1.400, N=5 0.8652, N=7 0.6231, N=9 0.4862.

    Fark N buyudukce KUCULUYOR: uzun halka yerel olarak zincire benzer ve
    frustrasyon seyrelir.
    """
    art = finite_ring_matrix(_nn_modeli(-T), n)
    eksi = finite_ring_matrix(_nn_modeli(+T), n)
    assert np.abs(_spektrum(art) - _spektrum(eksi)).max() == pytest.approx(
        fark, abs=1e-3
    )


# --- 2. DOGRU DEGISMEZ: cevrim uzunlugu DEGIL, NN PARITESI ---------------


@pytest.mark.parametrize(
    "n,yalniz_t1", [(4, 2.888624e-01), (6, 3.694823e-01), (8, 4.981517e-01),
                    (12, 5.432251e-01)]
)
def test_UZUN_MENZIL_varsa_YALNIZ_t1_cevirmek_SPEKTRUMU_kaydirir(n, yalniz_t1):
    """KARSI-ORNEK -- "tek uzunlukta cevrim" etiketini curutur.

    t1 = -0.7 ve t3 = -0.2 hoplamali ACIK zincir (hic cevrimi YOK):
      yalnizca t1 cevrilince spektrum kayiyor (yukaridaki sayilar),
      t1 VE t3 birlikte cevrilince 0.000000e+00.
    Yani belirleyici olan sey cevrimin uzunlugu degil, ayarin TUM tek
    pariteli hoplamalari birlikte cevirmesidir.
    """
    taban = _acik_zincir(n, -0.7, -0.2)
    tek_cevrilmis = _acik_zincir(n, +0.7, -0.2)
    ikisi_cevrilmis = _acik_zincir(n, +0.7, +0.2)

    assert np.abs(
        _spektrum(taban) - _spektrum(tek_cevrilmis)
    ).max() == pytest.approx(yalniz_t1, rel=1e-5)
    assert np.abs(_spektrum(taban) - _spektrum(ikisi_cevrilmis)).max() == 0.0


# --- 3. Ayar-degismez buyukluk: cevrim carpimi ---------------------------


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_CEVRIM_CARPIMI_her_ayar_altinda_DEGISMEZ(n):
    """Ayar-degismez olan sey t'nin kendisi degil, cevrim carpimidir."""
    H = finite_ring_matrix(_nn_modeli(-T), n)
    referans = _cevrim_carpimi(H, n)
    for isaretler in itertools.product((1.0, -1.0), repeat=n):
        D = np.diag(isaretler)
        assert _cevrim_carpimi(D @ H @ D, n) == pytest.approx(referans, rel=1e-12)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_CEVRIM_CARPIMININ_ISARETI_yalniz_TEK_N_de_bilgi_tasir(n):
    """t^N: N cift ise isaret her zaman +, yani sifir bit bilgi."""
    neg = _cevrim_carpimi(finite_ring_matrix(_nn_modeli(-T), n), n)
    poz = _cevrim_carpimi(finite_ring_matrix(_nn_modeli(+T), n), n)
    if n % 2 == 0:
        assert neg > 0 and poz > 0
        assert neg == pytest.approx(poz, rel=1e-12)
    else:
        assert neg < 0 < poz


# --- 4. Mekanizma: t -> -t, BZ'de k -> k + pi ----------------------------


def test_ISARET_CEVIRMEK_BZ_de_k_arti_pi_KAYMASIDIR():
    """Bant NOKTA BAZINDA degisiyor, KUME olarak degismiyor; gozlenebilir
    her sey (DOS, toplam enerji, dolum) kumeye baglidir.

    OLCULDU (401 noktali k izgarasi): nokta bazinda max fark 2.800e+00;
    k -> k+pi kaydirildiktan sonra 0'a iniyor.
    """
    ks = np.linspace(-np.pi, np.pi, 401)

    def bant(t, kaydir=0.0):
        return np.array(
            [
                float(
                    np.linalg.eigvalsh(
                        bloch_hamiltonian(
                            _nn_modeli(t), np.array([k + kaydir, 0.0, 0.0])
                        )
                    ).min()
                )
                for k in ks
            ]
        )

    assert np.abs(bant(-T) - bant(+T)).max() == pytest.approx(2.800, abs=1e-3)
    assert np.abs(bant(-T) - bant(+T, kaydir=np.pi)).max() < 1e-12


# --- 5. TAUTOLOJI KANITI: fizik cikarilinca da tutuyor -------------------


def test_ISARET_HUKMU_FIZIKSIZ_bir_cekirdekte_de_TUTUYOR():
    """Bu test "isaret sinavi bir CIPA DEGILDIR"i kilitler.

    S_fake[i,j] = exp(-a * d_ij), kosegen 1, H_eff = S_fake^-1.
    Hamiltoniyen YOK, elektron YOK, integral YOK, baz seti YOK -- yalnizca
    halka GEOMETRISI. OLCULDU: N = 3,5,7,9,11 x a = 0.3,0.5,1,2,4 -> 25/25
    "cevrim carpimi negatif VE taban katliligi 1" cikiyor
    (N=3, a=0.5: -8.4989e-03; N=11, a=2.0: -8.2737e-10).

    Bir sinavin ciktisi girdi uzayinin tamaminda sabitse sifir bit bilgi
    tasir. Bu yuzden hicbir yerde "isaret dogru cikti" diye capalama
    YAPILMAZ.
    """
    gecen = 0
    toplam = 0
    for n in (3, 5, 7, 9, 11):
        for alpha in (0.3, 0.5, 1.0, 2.0, 4.0):
            aci = 2.0 * np.pi * np.arange(n) / n
            noktalar = np.stack([1.7 * np.cos(aci), 1.7 * np.sin(aci)], axis=1)
            mesafe = np.linalg.norm(
                noktalar[:, None, :] - noktalar[None, :, :], axis=2
            )
            s_fake = np.exp(-alpha * mesafe)
            np.fill_diagonal(s_fake, 1.0)
            H = np.linalg.inv(s_fake)
            ozdeger = _spektrum(H)
            katlilik = int(np.sum(np.abs(ozdeger - ozdeger[0]) < 1e-9))
            toplam += 1
            gecen += (_cevrim_carpimi(H, n) < 0.0) and (katlilik == 1)
    assert gecen == toplam == 25
