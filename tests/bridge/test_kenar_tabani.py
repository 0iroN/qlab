"""Modül 7 / Task 2 — tek gerçek kenarın tabanı, ve o tabanın İKİ SAYI oluşu.

Bu dosyanın taşıyıcı iddiası tek cümledir: **fiziği kapatmak _tabanı_
ölçer; fiziği analitik bir ikizle değiştirmek _hata çubuğunu_ ölçer.**
İkisi ayrı sayıdır, ayrı kovadır, ve panel ikisini de yazar. Yalnız
birincisi sevk edilseydi panel gerçek hatayı **426 kat iyimser**
gösterirdi — ve bu bir tahmin değil, aşağıda ölçülen bir orandır.

PENCERE (her sayı bunun içindedir ve dışına taşınamaz):
  boru        `tests/md/test_aimd.py`'nin YOL B'si, harfi harfine
  yüzey       `Morse(d_e = K/(2a²), r_e = 1.3459193602, a = a)`, `K = 0.5729363`
  `h`         0.002 (statik ikinci fark, `ω_A` → `dt`)
  `dt`/`N`    `dt = (2π/ω_A)/N`, `N = 200`, 6 periyot
  genlikler   (0.01, 0.02, 0.04), `A²`'de doğrusal uydurma
  `a`         A1: (0.03, 0.02, 0.01, 0.003, 0.001, 1e-4, 0.0) · A2: 0.9137
  SCF         **0** — kenar yalnız `ForceField` nesnesi değiştirilerek kesilir
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from qlab.engine.md.forcefields import Harmonic, HartreeFockGradient, Morse
from qlab.engine.md.state import ForceField
from qlab.errors import InvalidInputError
from qlab.observable.bridge_obs import (
    A1_MERDIVENI,
    A_ESLESEN,
    GENLIKLER,
    GERCEK_HATA_CM,
    K_HF,
    MU_H2,
    N_BOLME,
    PERIYOT_SAYISI,
    R_E_BOHR,
    TAM_AU,
    TAM_CM,
    TURLER,
    EdgeFloor,
    a2_egimi,
    edge_noise_floor,
    omega_b,
)
from qlab.observable.bridge_obs import (
    _a2_uydurmasi,
    _artik_au,
    _boru_kur,
    _donmus_yuzey,
    _omega_a,
)
from qlab.observable.grav_obs import VERDICTS, newton_noise_floor
from qlab.units import HARTREE_TO_WAVENUMBER, amu_to_electron_mass

PROJE_KOKU = Path(__file__).resolve().parents[2]
KAYNAK_DOSYA = PROJE_KOKU / "src" / "qlab" / "observable" / "bridge_obs.py"

M_H = float(amu_to_electron_mass(1.00782503207))


class _Olcekli:
    """Kuvveti sabit bir çarpanla bozan sarmalayıcı — mutant düğmesi.

    `test_aimd.py`'nin bağımsızlık kanıtındaki `×1.001` bozulmasının aynısı;
    orada `ω_A` birebir aynı kalır, `ω_B` kayar.
    """

    def __init__(self, alan: ForceField, c: float) -> None:
        self.alan = alan
        self.c = c

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        enerji, kuvvet = self.alan(positions)
        return enerji, self.c * kuvvet


# --- TAŞIYICI ----------------------------------------------------------------


def test_KENAR_TABANI_IKI_SAYIDIR_ve_BIRI_426_KAT_IYIMSER():
    """BU DOSYANIN EN ÖNEMLİ TESTİ — Modül 6'nın KUSUR BEYANININ ONARIMI.

    Modül 6 şunu beyan etmişti: *taban yalnız aritmetiği ölçer; kesme hatası
    fizikle birlikte kapandığı için sınav onu yapısal olarak göremez.* Orada
    bu bir **kusur olarak kaldı**. Burada kapanıyor — çünkü burada baskın
    hata bileşeninin bir **adı** (anharmoniklik) ve bir **düğmesi**
    (Morse'un `a`'sı) var.

    ÖLÇÜLDÜ (yukarıdaki pencerede, 30.08.2026):
        A1 (fizik kapalı, `a → 0`)      3.398670651222e-05 cm⁻¹, S/G 1.6128e+08
        A2 (analitik ikiz, `a = 0.9137`) 1.202599227508e-02 cm⁻¹, S/G 4.5578e+05
        gerçek hata                      1.447280860793e-02 cm⁻¹

        gerçek/A1 = 425.837  → dürüst hane farkı 2.6292
        A2/gerçek = 0.83094  → dürüst hane farkı 0.0804

    Yani A1 tek başına sevk edilseydi panel gerçek hatayı 426 kat iyimser
    yazardı; A2 gerçeğin %83'ünü tutturuyor. İKİSİ AYRI SAYIDIR.

    NOT — PLANIN SAYILARI YENİDEN ÖLÇÜLDÜ ve ikisi kıl payı kaydı: plan
    `A2/gerçek = 0.8312` diyordu, ölçülen 0.830937; plan gerçek hatayı
    `1.4467853e-02` yazıyordu, ölçülen `1.4472809e-02` (fark %0.034, ve
    kaynağı `ω_ref`'in Richardson çifti — bkz. `GERCEK_HATA_CM`).
    """
    a1 = edge_noise_floor(tur="aritmetik")
    a2 = edge_noise_floor(tur="hata-cubugu")

    assert isinstance(a1, EdgeFloor) and isinstance(a2, EdgeFloor)
    assert (a1.tur, a2.tur) == ("aritmetik", "hata-cubugu")
    assert set(TURLER) == {a1.tur, a2.tur}

    assert a1.taban == pytest.approx(3.398670651222e-05, rel=1e-9, abs=0.0)
    assert a2.taban == pytest.approx(1.202599227508e-02, rel=1e-9, abs=0.0)
    assert a1.hukum == "SAGLAM" and a2.hukum == "SAGLAM"
    assert set(VERDICTS) >= {a1.hukum, a2.hukum}

    # Sinyal AYNI boru, AYNI pencere -- karışık boru YOK.
    assert a1.sinyal == a2.sinyal == pytest.approx(5481.241299201833, rel=1e-9,
                                                   abs=0.0)
    assert a1.oran == pytest.approx(1.612760e08, rel=1e-5, abs=0.0)
    assert a2.oran == pytest.approx(4.557829e05, rel=1e-5, abs=0.0)

    # VE ASIL SAYI: ikisinin gerçek hataya UZAKLIĞI aynı değil.
    assert GERCEK_HATA_CM / a1.taban == pytest.approx(425.837, rel=1e-4, abs=0.0)
    assert a2.taban / GERCEK_HATA_CM == pytest.approx(0.830937, rel=1e-4, abs=0.0)
    assert math.log10(GERCEK_HATA_CM / a1.taban) == pytest.approx(2.6292, abs=1e-3)
    assert math.log10(GERCEK_HATA_CM / a2.taban) == pytest.approx(0.0804, abs=1e-3)

    # İKİSİ TEK HÜCREYE YAZILAMAZ: boru etiketleri de AYRI.
    assert a1.boru != a2.boru
    assert "fizik KAPALI" in a1.boru
    assert "IKIZLE DEGISTIRILDI" in a2.boru


def test_A_SIFIR_NOKTASI_DEJENERE_DEGIL_MODUL_6_NIN_TERSI():
    """Modül 6'da fiziği kapatmak bir NOKTA OLAMAZDI: `w == 1` olduğu için
    integrand sabit 1.0'a düşüyor, `quad` sabiti tam integre ediyor ve
    `dphi - 2*pi` BİT BİT 0.0 çıkıyordu. Onarım `lambda → 0` MERDİVENİYDİ.

    BURADA TERSİ, ve ölçülerek: cebirsel `a = 0` noktası
    `1.4670405515393448e-10` a.u. döndürüyor, `== 0.0` → **False**; ve
    `a → 0` merdiveni tam bu noktaya yakınsıyor
    (`a = 1e-4` → `1.4674508136414133e-10`, bağıl fark 2.8e-04).

    Yani MERDİVEN BURADA SİGORTADIR, ZORUNLULUK DEĞİL — ve bunu ölçerek
    biliyoruz. `a = 0` noktasının var olabilmesinin sebebi `Morse`'un
    `a → 0` CEBİRSEL limitinin sonlu olması (`_BagHarmonik`); `Morse`'un
    kendisi orada tanımsızdır ve bu da ayrıca ölçülüyor.
    """
    nokta = _artik_au(a=0.0)
    assert nokta != 0.0
    assert math.isfinite(nokta)
    assert nokta == pytest.approx(1.4670405515393448e-10, rel=1e-9, abs=0.0)
    assert _artik_au(a=1e-4) == pytest.approx(1.4674508136414133e-10, rel=1e-9,
                                              abs=0.0)
    # Merdiven NOKTAYA yakınsıyor: 1e-4 ile 0.0 arasındaki bağıl fark.
    assert abs(_artik_au(a=1e-4) / nokta - 1.0) == pytest.approx(2.796e-04,
                                                                rel=1e-2, abs=0.0)
    # ...ve merdivenin BÜTÜN basamakları CANLI; hiçbiri elenmiyor.
    a1 = edge_noise_floor(tur="aritmetik")
    assert f"{len(A1_MERDIVENI)}/{len(A1_MERDIVENI)} canli" in a1.boru
    assert 0.0 in A1_MERDIVENI

    # `Morse` a = 0'da TANIMSIZ; noktayı var eden şey CEBİRSEL LİMİTTİR.
    with pytest.raises(ZeroDivisionError):
        Morse(d_e=K_HF / (2.0 * 0.0 * 0.0), r_e=R_E_BOHR, a=0.0)
    with pytest.raises(InvalidInputError):
        Morse(d_e=1.0, r_e=R_E_BOHR, a=0.0)


def test_A1_MERDIVENI_ICINDE_MONOTON_ama_DISINDA_DEGIL():
    """A1 tabanı merdivenin **tepesinden** okunuyor, sınırından değil.

    ÖLÇÜLDÜ (cm⁻¹): a = 0.0 / 1e-4 / 0.001 / 0.003 / 0.01 / 0.02 / 0.03 →
    3.219782e-05 / 3.220682e-05 / 3.226580e-05 / 3.240329e-05 /
    3.290616e-05 / 3.354177e-05 / 3.398671e-05 — merdiven içinde MONOTON.

    Ama merdivenin DIŞINDA monotonluk YOK: a = 0.05'te 3.346971e-05'e,
    a = 0.1'de 1.757641e-05'e DÜŞÜYOR. Yani `0.03` bir sınır değil bir
    TEPE; merdiveni büyütmek tabanı büyütmez ve bu bir tercih değil,
    ölçülmüş bir şekildir.
    """
    artiklar = [
        _artik_au(a=a) * HARTREE_TO_WAVENUMBER for a in reversed(A1_MERDIVENI)
    ]
    assert artiklar == pytest.approx(
        [3.219782e-05, 3.220682e-05, 3.226580e-05, 3.240329e-05,
         3.290616e-05, 3.354177e-05, 3.398671e-05], rel=1e-5, abs=0.0
    )
    assert all(artiklar[i] < artiklar[i + 1] for i in range(len(artiklar) - 1))
    # ...ve merdivenin ÖTESİ düşüyor.
    for disari in (0.05, 0.1):
        assert _artik_au(a=disari) * HARTREE_TO_WAVENUMBER < artiklar[-1]
    assert _artik_au(a=0.05) * HARTREE_TO_WAVENUMBER == pytest.approx(
        3.346971e-05, rel=1e-5, abs=0.0
    )
    assert _artik_au(a=0.1) * HARTREE_TO_WAVENUMBER == pytest.approx(
        1.757641e-05, rel=1e-5, abs=0.0
    )

    # VE BURADA ÖLÇÜLEN BİR SINIR VAR — bu test onu YAZARAK kilitliyor.
    # `edge_noise_floor` artığı cm⁻¹'de çıkarır (`|ω_B·HTW − TAM_CM|`),
    # `_artik_au` a.u.'da çıkarıp sonra çevirir. İkisi AYNI SAYI DEĞİL:
    # bağıl fark 1.3779737351171661e-08 ölçüldü. Sebep bir hata değil bir
    # SADELEŞME: `ulp(TAM_CM)/taban = 2.6760307415442646e-08`, yani cm⁻¹'de
    # çıkarmak tabanın son ~8 hanesini yok ediyor. TABAN SEKİZ ANLAMLI
    # HANEDEN FAZLASINA YAZILAMAZ; tolerans bu ölçümden geliyor, gevşetmeden.
    ikisi = edge_noise_floor(tur="aritmetik").taban
    assert ikisi / artiklar[-1] - 1.0 == pytest.approx(1.37797e-08, rel=1e-4,
                                                       abs=0.0)
    assert abs(ikisi / artiklar[-1] - 1.0) < math.ulp(TAM_CM) / artiklar[-1]


# --- TABAN KIPIRDIYOR: DÖRT DÜĞME --------------------------------------------


def test_TABAN_KIPIRDIYOR_dt_MERDIVENI_ve_KARELEMEDEN_BAGIMSIZ_KAPALI_FORM():
    """Bir sayı hiçbir düğmeye tepki vermiyorsa o bir ölçüm değil bir sabittir.

    ÖLÇÜLDÜ (`a = 0`, öteki pencere sabit), artık cm⁻¹:
        N=50   4.159394e-03
        N=100  4.418242e-04    oran 9.41
        N=200  3.219782e-05    oran 13.72
        N=400  2.162532e-06    oran 14.89
        N=800  1.399601e-07    oran 15.45
        N=1600 8.826032e-09    oran 15.86   → **16 = 2⁴**

    VE KARELEMEDEN BAĞIMSIZ BİR KAPALI FORM ÖNGÖRÜSÜ VAR:
    `artık/ω = (ω·dt)⁴/144`. Ölçülen/öngörü aynı merdivende
    0.4382 / 0.7448 / 0.8684 / 0.9332 / 0.9663 / 0.9750 — yani öngörüye
    MONOTON yakınsıyor. Bu öngörü tabanın bir aritmetik artık olduğunun
    bağımsız kanıtıdır.

    PLANIN SAYISI ÇÜRÜDÜ: plan `ölçülen/öngörü = 0.9959 @ N = 1600` diyordu;
    ölçülen **0.9750**. Sapma yönü de bilgi taşıyor — dizi 1'in ALTINDAN
    yaklaşıyor ve N = 3200'de dönüyor (0.9125), yani orada kayan nokta
    tabanı devralıyor.
    """
    yuzey = _donmus_yuzey(0.0)
    wa = _omega_a(yuzey)
    artiklar: list[float] = []
    oranlar: list[float] = []
    for n in (50, 100, 200, 400, 800, 1600):
        dt = (2.0 * np.pi / wa) / n
        artik = abs(omega_b(yuzey, n_bolme=n) - TAM_AU)
        ongoru = TAM_AU * (TAM_AU * dt) ** 4 / 144.0
        artiklar.append(artik * HARTREE_TO_WAVENUMBER)
        oranlar.append(artik / ongoru)

    assert artiklar[0] == pytest.approx(4.159394e-03, rel=1e-5, abs=0.0)
    assert artiklar[-1] == pytest.approx(8.826032e-09, rel=1e-5, abs=0.0)
    # DÖRDÜNCÜ MERTEBE: ardışık oranlar 16'ya MONOTON tırmanıyor.
    ardisik = [artiklar[i] / artiklar[i + 1] for i in range(len(artiklar) - 1)]
    assert ardisik == pytest.approx([9.41, 13.72, 14.89, 15.45, 15.86],
                                    rel=1e-3, abs=0.0)
    assert all(ardisik[i] < ardisik[i + 1] < 16.0 for i in range(len(ardisik) - 1))
    # KAPALI FORM: monoton yakınsama, ve N = 1600'de 0.9750.
    assert oranlar == pytest.approx([0.4382, 0.7448, 0.8684, 0.9332, 0.9663,
                                     0.9750], rel=1e-3, abs=0.0)
    assert all(oranlar[i] < oranlar[i + 1] for i in range(len(oranlar) - 1))
    assert oranlar[-1] != pytest.approx(0.9959, rel=1e-3, abs=0.0)


def test_TABAN_KIPIRDIYOR_KUVVET_ve_KUTLE_dugmelerinde():
    """Üçüncü ve dördüncü düğme — ve kütle düğmesi kapalı formu İKİNCİ
    KEZ, bağımsız bir yoldan doğruluyor.

    KUVVET ×1.001 (ölçüldü): taban `3.219782e-05` → `2.739889e+00` cm⁻¹,
    yani **85095.5 kat**. Kapalı form bunu bekliyor:
    `TAM·(√1.001 − 1) = 2.7399418` cm⁻¹, ölçülenle bağıl fark 1.9e-05.
    (Planın sayısı `84 916`'ydı; ölçülen **85095.5** — %0.21 fark.)

    KÜTLE ×4, İKİ AYRI KULLANIM ve İKİSİ AYNI SAYIYI VERMEZ:
      (i)  `dt` boru kuralından YENİDEN türetilirse çarpan **tam 0.5**.
           Sebep kapalı formda yazılı: `ω·dt` değişmez, `artık = ω(ω·dt)⁴/144`
           yalnız `ω` ile ölçeklenir ve `ω` tam yarıya iner.
      (ii) `dt` DONDURULURSA taban `1.123208e-06` cm⁻¹'e düşer,
           yani **28.666 kat** küçülür. (Planın "28.6 kat"ı bu kullanımdır;
           yön plana yazılmamıştı — taban BÜYÜMÜYOR, KÜÇÜLÜYOR.)
    """
    yuzey = _donmus_yuzey(0.0)
    taban_au = abs(omega_b(yuzey) - TAM_AU)
    assert taban_au * HARTREE_TO_WAVENUMBER == pytest.approx(3.219782e-05,
                                                             rel=1e-5, abs=0.0)

    # --- KUVVET
    bozuk = abs(omega_b(_Olcekli(yuzey, 1.001)) - TAM_AU) * HARTREE_TO_WAVENUMBER
    assert bozuk / (taban_au * HARTREE_TO_WAVENUMBER) == pytest.approx(
        85095.5, rel=1e-4, abs=0.0
    )
    assert bozuk == pytest.approx(TAM_CM * (math.sqrt(1.001) - 1.0), rel=1e-4,
                                  abs=0.0)

    # --- KÜTLE
    mu4 = 4.0 * MU_H2
    tam4 = math.sqrt(K_HF / mu4)
    kutle4 = np.full(2, 4.0 * M_H)
    # (i) `dt` boru kuralından YENİDEN: çarpan tam 0.5 -- ve BU BİT DÜZEYİNDE
    # beklenir, çünkü `mu` ×4, `dt` ×2 ve `masses` ×4 ikinin kuvvetleridir,
    # yani mantisleri değiştirmez. Ölçüldü: tam 0.5.
    yeniden = abs(omega_b(yuzey, mu=mu4, masses=kutle4) - tam4)
    assert yeniden / taban_au == 0.5
    # (ii) `dt` DONDURULMUŞ: taban 28.666 kat KÜÇÜLÜR.
    dt0 = (2.0 * np.pi / _omega_a(yuzey)) / N_BOLME
    donmus = abs(omega_b(yuzey, mu=mu4, masses=kutle4, dt=dt0) - tam4)
    assert taban_au / donmus == pytest.approx(28.666, rel=1e-4, abs=0.0)


# --- YANLIŞ İKİZ -------------------------------------------------------------


def test_HARMONIC_YANLIS_IKIZDIR_ve_PENCEREYI_SESSIZCE_DEGISTIRIR():
    """`Morse` bir TERCİH DEĞİL — ve tehlikenin şekli ölçüldü.

    Sevk edilen `Harmonic` MERKEZ-TABANLIDIR: her atomu kendi merkezine
    bağlar, bağı indirgenmiş kütle üzerinden GÖRMEZ. Tehlikeli olan şu:
    **statik yol bu farkı GÖREMİYOR.** İki yüzeyin `ω_A`'sı BİT BİT AYNI
    (ölçüldü — `float` eşitliği), çünkü `V''(r_e)` ikisinde de `K`.
    Ayrışan şey DİNAMİK: `ω_B` `3875.83132712` cm⁻¹, yani `TAM/√2`.

    Sonuç bir sayı değil bir HÜKÜM farkı: yanlış ikizle "taban"
    `1605.42` cm⁻¹ çıkardı — sevk edilen A1'in **4.7e+07 katı**.

    PLANIN SAYISI ÇÜRÜDÜ VE YÖNÜ TERS: plan `7751.66279129` cm⁻¹ yazıyordu.
    O sayı `√2 × TAM`'dır, yani ölçülen `3875.83132712`'nin TAM İKİ KATI;
    yanlış ikiz `ω_B`'yi √2 BÜYÜTMÜYOR, √2 KÜÇÜLTÜYOR.
    """
    dogru = _donmus_yuzey(0.0)
    yanlis = Harmonic(k=K_HF, center=np.array([[0.0, 0.0, 0.0],
                                               [0.0, 0.0, R_E_BOHR]]))

    # STATİK YOL KÖR: bit bit aynı.
    assert _omega_a(yanlis) == _omega_a(dogru)

    # DİNAMİK YOL AYIRIYOR.
    wb_yanlis = omega_b(yanlis) * HARTREE_TO_WAVENUMBER
    assert wb_yanlis == pytest.approx(3875.83132712, rel=1e-9, abs=0.0)
    assert wb_yanlis == pytest.approx(TAM_CM / math.sqrt(2.0), rel=1e-7, abs=0.0)
    # Plan'ın sayısı ölçülenin TAM İKİ KATI -- ve bu tesadüf değil.
    assert 2.0 * wb_yanlis == pytest.approx(7751.66279129, rel=1e-7, abs=0.0)

    yanlis_taban = abs(wb_yanlis - TAM_CM)
    assert yanlis_taban == pytest.approx(1605.42, rel=1e-4, abs=0.0)
    assert yanlis_taban / edge_noise_floor(tur="aritmetik").taban == pytest.approx(
        4.7e07, rel=5e-2, abs=0.0
    )


# --- ELENEN ADAYLAR ----------------------------------------------------------


def test_ELENEN_UC_ADAY_OLCUMLE_ELENDI_ve_SAYILARI_BEYAN_EDILDI():
    """Bir eleme gerekçesiz yazılırsa bir tercihtir; sayısıyla yazılırsa ölçüm.

    (b) `step → 0` — KIPIRDIYOR AMA FİZİĞİ KAPATMIYOR. `ω_ref`'ten sapma
        HF gradyan adımı 1e-4 / 5e-5 / 1e-5 için
        1.4472299906628905e-02 / 1.4472808607933985e-02 /
        1.4473752252342820e-02 cm⁻¹; limitte SABİT (bağıl oynama 1.0e-04).
        Üstelik TAM'ı yok: tek TAM adayı aynı HF çözümünün öteki tüketicisi
        olurdu ve o KARIŞIK BORUDUR. **Bu üç sayı süite ALINMADI** çünkü
        her biri ayrı bir HF YOL B koşumu, yani ~28 s (ölçüldü); süitede
        `step = 5e-5` noktası `slow` testte canlı üretiliyor, ötekiler bu
        beyanda kalıyor.
    (c) baz daraltma — YÖN BOŞ; burada ölçülüyor.
    (d) kenarı tam kesmek — boru patlıyor; burada ölçülüyor.
    """
    from qlab.engine.hf.basis import AVAILABLE_BASES, build_basis
    from qlab.errors import BasisNotAvailableError
    from qlab.types import Molecule

    # (c) YÖN BOŞ: sto-3g zaten minimal, tek alternatif GENİŞLETİYOR.
    h2 = Molecule.from_string("H 0 0 0; H 0 0 0.71223")
    assert len(build_basis(h2, "sto-3g")) == 2
    assert len(build_basis(h2, "6-31g")) == 4
    assert AVAILABLE_BASES == ("sto-3g", "6-31g")
    with pytest.raises(BasisNotAvailableError):
        build_basis(h2, "sto-1g")

    # (d) KENARI TAM KESMEK: iki farklı aritmetikte iki farklı ölüm.
    with pytest.raises(ZeroDivisionError):
        K_HF / (2.0 * 0.0 * 0.0)
    with np.errstate(divide="ignore"):
        assert not math.isfinite(float(np.float64(K_HF) / np.float64(0.0)))
    with pytest.raises(InvalidInputError):
        _donmus_yuzey(-1.0)


# --- ÖLÜ BASAMAK ÖLÇÜTÜ ------------------------------------------------------


def test_OLU_BASAMAK_OLCUTU_GENISLETILDI_nan_SIZAMAZ():
    """Modül 6'nın ölçütü BURADA YETMİYOR — ve yetmediği ölçüldü.

    Eski ölçüt `deger == 0.0` idi. `nan == 0.0` → **False**, yani bir `nan`
    ölü SAYILMIYOR ve `max()`'e sızıyor. Sızması tek başına da kötü değil;
    asıl arıza şu: sonuç ARGÜMAN SIRASINA bağlanıyor —
        `max([nan, 1.0, 2.0])` → nan
        `max([1.0, nan, 2.0])` → 2.0
        `max([2.0, nan, 1.0])` → 2.0
    Bir hükmün probe sırasına bağlı olması, hüküm olmamasıdır.

    GENİŞLETİLMİŞ ÖLÇÜT: `not math.isfinite(deger) or deger == 0.0`.
    """
    # ÖNCE ÖLÇÜT-ÖNCESİ GERÇEK: `nan` ölü sayılmıyor ve `max` sıraya bağlı.
    assert (math.nan == 0.0) is False
    assert math.isnan(max([math.nan, 1.0, 2.0]))
    assert max([1.0, math.nan, 2.0]) == 2.0
    assert max([2.0, math.nan, 1.0]) == 2.0

    def boru(*, kotu: bool = False, sifir: bool = False) -> float:
        if kotu:
            return math.nan
        if sifir:
            return 0.0
        return 10.0

    boru.PIPELINE = "oyuncak boru -- olu basamak olcutu"
    # `nan` probe'u BAŞTA: eski ölçütle `max()` `nan` döndürürdü.
    boru.NEWTON_PROBES = (
        ({"kotu": True}, 0.0),
        ({"sifir": True}, 0.0),
        ({}, 9.0),
    )
    kayit = newton_noise_floor(boru)
    assert math.isfinite(kayit.value)
    assert kayit.value == 1.0
    assert "1/3 canli" in kayit.pipeline
    assert kayit.verdict == "SAGLAM"

    # HEPSİ ÖLÜ (biri `nan`, biri 0.0) → DEJENERE, `ratio` nan.
    def hepsi_olu(*, kotu: bool = False) -> float:
        return math.nan if kotu else 7.0

    hepsi_olu.PIPELINE = "oyuncak boru -- hepsi olu"
    hepsi_olu.NEWTON_PROBES = (({"kotu": True}, 1.0),)
    olu = newton_noise_floor(hepsi_olu)
    assert olu.verdict == "DEJENERE"
    assert olu.value == 0.0 and math.isnan(olu.ratio)
    assert "0/1 canli" in olu.pipeline


def test_SEVK_EDILEN_OLCUT_KAYNAKTA_DA_YAZILI():
    """Ölçüt bir dokstring iddiası değil, KODDA duran bir satır olmalı.

    Bu test `grav_obs.py`'nin `newton_noise_floor` gövdesini `ast` ile
    okuyup genişletilmiş ölçütün gerçekten orada olduğunu doğrular; ölçüt
    daraltılırsa (yalnız `deger == 0.0`) bu test KIRMIZI olur.
    """
    yol = PROJE_KOKU / "src" / "qlab" / "observable" / "grav_obs.py"
    agac = ast.parse(yol.read_text(encoding="utf-8"))
    govde = next(
        d for d in agac.body
        if isinstance(d, ast.FunctionDef) and d.name == "newton_noise_floor"
    )
    kaynaklar = {
        ast.unparse(d.test) for d in ast.walk(govde) if isinstance(d, ast.If)
    }
    assert "not math.isfinite(deger) or deger == 0.0" in kaynaklar


# --- KARIŞIK BORU YASAĞI -----------------------------------------------------


def test_KENAR_KESILIRKEN_hf_scf_solve_SIFIR_KEZ_CAGRILIYOR(monkeypatch):
    """"Aynı boru" iddiasının tek dürüst sınavı budur.

    Kenar YALNIZ `ForceField` nesnesi değiştirilerek kesilir: `engine/md`'nin
    tek satırı değişmez ve HF hiç çözülmez. `solve` patlayıcıyla
    değiştirilir; taban ölçümü yine de tamamlanıyorsa SCF sayısı **sıfırdır**.

    İkinci kilit `ast` düzeyinde: `bridge_obs` `qlab.engine.hf`'ten HİÇBİR
    ŞEY import etmiyor — yani karışık boru bir çalışma zamanı tesadüfü
    değil, YAPISAL olarak imkânsız.
    """

    def patlayici(*_a, **_k):  # pragma: no cover - çağrılmamalı
        raise AssertionError("hf.scf.solve cagrildi -- KARISIK BORU")

    monkeypatch.setattr("qlab.engine.md.forcefields.solve", patlayici)
    a1 = edge_noise_floor(tur="aritmetik")
    a2 = edge_noise_floor(tur="hata-cubugu")
    assert a1.hukum == "SAGLAM" and a2.hukum == "SAGLAM"

    agac = ast.parse(KAYNAK_DOSYA.read_text(encoding="utf-8"))
    adlar: set[str] = set()
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.ImportFrom) and dugum.module:
            adlar.add(dugum.module)
        elif isinstance(dugum, ast.Import):
            adlar.update(ad.name for ad in dugum.names)
    assert not any(ad.startswith("qlab.engine.hf") for ad in adlar), adlar
    assert "qlab.engine.md.integrator" in adlar


def test_BORU_ETIKETI_PENCEREYI_YAZIYOR_ve_IKI_SINAV_AYNI_ETIKETI_TASIMIYOR():
    """Pencere adlandırılmazsa hüküm dışına taşınır — Modül 6'nın dersi.

    `newton_noise_floor`'un `PIPELINE_KEYS` mekanizması burada `n_bolme`,
    `periyot_sayisi` ve `genlikler`i etikete yazıyor; A1 ile A2 ayrıca
    `PIPELINE` metninde ayrışıyor, yani iki AYRI sınav aynı etiketi
    taşımıyor (Task 3'te `gauge` yüzünden düşen tam bu ayrımdı).
    """
    a1 = edge_noise_floor(tur="aritmetik")
    a2 = edge_noise_floor(tur="hata-cubugu")
    for kayit in (a1, a2):
        assert f"n_bolme={N_BOLME}" in kayit.boru
        assert f"periyot_sayisi={PERIYOT_SAYISI}" in kayit.boru
        assert f"genlikler={GENLIKLER}" in kayit.boru
    assert f"a = {A_ESLESEN}" in a2.boru
    assert str(A_ESLESEN) not in a1.boru
    # Farklı ikiz → farklı etiket VE farklı taban.
    yarim = edge_noise_floor(tur="hata-cubugu", a=0.5)
    assert yarim.boru != a2.boru
    assert yarim.taban / a2.taban == pytest.approx(0.16684, rel=1e-3, abs=0.0)


def test_edge_noise_floor_SOZLESMESI_cagiran_fizigi_KAPATAMAZ():
    """Modül 6'nın sözleşmesi devralındı: fiziği kapatma noktasını ÇAĞRILAN
    belirler. A1'de `a` bir düğme DEĞİLDİR; A2'de `a` HANGİ İKİZ olduğunu
    söyler ve bu bir kaçak değil, sınavın kendisidir.
    """
    with pytest.raises(InvalidInputError):
        edge_noise_floor(tur="aritmetik", a=0.5)
    with pytest.raises(InvalidInputError):
        edge_noise_floor(tur="yok-boyle-bir-tur")
    with pytest.raises(InvalidInputError):
        edge_noise_floor(tur="hata-cubugu", a=-1.0)
    with pytest.raises(InvalidInputError):
        edge_noise_floor(tur="hata-cubugu", a=float("nan"))

    # VE `newton_noise_floor` katmanında da: probe anahtarı verilemez.
    fn = _boru_kur(tur="aritmetik", a_acik=A_ESLESEN,
                   probeler=(({"a": 0.01}, TAM_CM),), etiket="sinav borusu")
    with pytest.raises(InvalidInputError):
        newton_noise_floor(fn, a=0.02)

    # Pencere doğrulaması da yerinde.
    with pytest.raises(InvalidInputError):
        omega_b(_donmus_yuzey(0.0), n_bolme=0)
    with pytest.raises(InvalidInputError):
        omega_b(_donmus_yuzey(0.0), genlikler=(0.01,))


def test_A2_IKIZI_A_KARE_EGIMIYLE_ESLESTIRILDI_bir_TERCIH_DEGIL():
    """`a = 0.9137` uydurma bir sayı değil: borunun KENDİ ölçtüğü büyüklükten
    geliyor. `omega_b` `A²` doğrusunun KESİŞİMİ; anharmoniklik EĞİMDEDİR.

    ÖLÇÜLDÜ: dondurulmuş Morse `a = 0.9137`'de eğim `-0.010023011049457173`;
    HF yüzeyinin kendi eğimi `-0.010024824447837994` (`slow` testte canlı
    üretiliyor). Bağıl fark **1.81e-04**.
    """
    egim = a2_egimi(_donmus_yuzey(A_ESLESEN))
    assert egim == pytest.approx(-0.010023011049457173, rel=1e-9, abs=0.0)
    assert abs(egim / -0.010024824447837994 - 1.0) == pytest.approx(
        1.81e-04, rel=2e-2, abs=0.0
    )
    # Eğim `a` ile GERÇEKTEN değişiyor -- eşleştirme bir düğmeye oturuyor.
    assert a2_egimi(_donmus_yuzey(0.5)) == pytest.approx(-0.0030548612112175836,
                                                        rel=1e-9, abs=0.0)
    # ...ve `a = 0`da CEBİRSEL OLARAK sıfıra düşüyor (anharmoniklik yok).
    assert abs(a2_egimi(_donmus_yuzey(0.0))) < 1e-12


# --- CANLI HF DOĞRULAMASI (PAHALI) -------------------------------------------


@pytest.mark.slow
def test_GERCEK_HATA_HF_BORUSUNDAN_CANLI_URETILIYOR_ve_BORU_AYNI_BORU():
    """`GERCEK_HATA_CM` bir literal değil, BU KOŞUMUN kaydı.

    İki şeyi birden ölçer:

    1. **Boru gerçekten `test_aimd.py`'nin YOL B'si mi?** `omega_b` HF
       alanıyla çağrıldığında o testin kilitlediği sayıyı BİT BİT vermeli:
       `ω_B/ω_A − 1 = -4.360844408468623e-06`. Vermezse dondurulmuş yüzeyde
       ölçülen taban BAŞKA bir borunun tabanıdır.
    2. **Gerçek hata.** `ω_B(HF, step = 5e-5) = 5481.229456091029` cm⁻¹;
       yüzeyin kendi harmonik frekansı `ω_ref = 5481.243928899637` cm⁻¹
       (statik ikinci farkın Richardson kesişimi, `h` çifti `(0.002, 0.004)`).
       Fark **1.4472808607933985e-02** cm⁻¹.

    ÇİFT BİR PENCERE DÜĞMESİDİR ve bu ölçüldü: `(0.001, 0.002)` →
    1.4478119341220008e-02, `(0.004, 0.008)` → 1.4471104747826757e-02.
    Yayılım ±4e-06 cm⁻¹, yani gerçek hata dört haneye yazılabilir, beşe
    yazılamaz — planın `1.4467853e-02`'si bu yayılımın **dışında**.

    MALİYET: bu test tek başına ~28 s (üç genlikte 1200 adım × 7 SCF).
    """
    alan = HartreeFockGradient(["H", "H"])
    wa = _omega_a(alan)
    # TEK koşum: `omega_b` ile `a2_egimi` aynı uydurmayı paylaşır, yoksa
    # bu test 28 s yerine 56 s sürerdi (ölçüldü).
    egim_hf, wb = _a2_uydurmasi(alan)

    # (1) BORU AYNI BORU -- bit bit.
    assert wb / wa - 1.0 == -4.360844408468623e-06
    assert wa * HARTREE_TO_WAVENUMBER == pytest.approx(5481.2533589840905,
                                                       rel=1e-12, abs=0.0)
    assert wb * HARTREE_TO_WAVENUMBER == pytest.approx(5481.229456091029,
                                                       rel=1e-12, abs=0.0)

    # (2) omega_ref: Richardson, `h` çifti (0.002, 0.004).
    k_h = {h: _omega_a(alan, h=h) ** 2 * MU_H2 for h in (0.002, 0.004)}
    assert k_h[0.002] == pytest.approx(K_HF, rel=1e-7, abs=0.0)
    k_r = (4.0 * k_h[0.002] - k_h[0.004]) / 3.0
    w_ref = math.sqrt(k_r / MU_H2) * HARTREE_TO_WAVENUMBER
    assert w_ref == pytest.approx(5481.243928899637, rel=1e-11, abs=0.0)

    # İKİ BÜYÜK SAYININ FARKI: `ulp(5481) ~ 9.1e-13`, yani bu fark bağıl
    # 1.3e-10'dan daha ince okunamaz; tolerans BU YÜZDEN 1e-8, gevşetmeden.
    gercek = w_ref - wb * HARTREE_TO_WAVENUMBER
    assert gercek == pytest.approx(GERCEK_HATA_CM, rel=1e-8, abs=0.0)
    assert gercek == pytest.approx(1.4472808607933985e-02, rel=1e-8, abs=0.0)
    # PLANIN SAYISI PENCERENİN DIŞINDA.
    assert gercek != pytest.approx(1.4467853e-02, rel=1e-4, abs=0.0)

    # (3) HF'nin kendi A²-eğimi -- A2 ikizinin eşleştirildiği büyüklük.
    assert egim_hf == pytest.approx(-0.010024824447837994, rel=1e-9, abs=0.0)
    assert abs(a2_egimi(_donmus_yuzey(A_ESLESEN)) / egim_hf - 1.0) < 2e-04

    # (4) VE İKİ SAYININ ANLAMI: A1 iyimser, A2 gerçeğe yakın.
    assert GERCEK_HATA_CM / edge_noise_floor(tur="aritmetik").taban == (
        pytest.approx(425.837, rel=1e-4, abs=0.0)
    )
