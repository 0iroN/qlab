"""SEVK EDILMIS KODDA OLCULMEMIS IKI KUSUR -- once onlar olculuyor.

`scf.solve()` bazi geometrilerde TABAN DURUM OLMAYAN bir RHF cozumune
yakinsiyor ve `converged=True` diyor. Olculdu ve sebep TEK DEGIL, IKI:

  (1) DIIS. N2 R=1.10 A'da DIIS'li kosum -106.769673858 verirken DIIS'siz
      kosum -107.496500512 buluyor -- yani orada tohum degil ITERASYON
      SEMASI belirleyici.
  (2) BASLANGIC TAHMINI. R=1.30...2.00 A'da DIIS'li ve DIIS'siz kosumlar
      BIREBIR AYNI sonucu veriyor (fark 0.000000), ama farkli baslangic
      sapmalariyla 2-5 AYRI cozum bulunuyor ve en dusugu 0.278...0.498
      hartree asagida.

Onemi: Modul 2D'nin ab initio MD'si bu `solve()` uzerine kuruludur.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.hf.multistart import search_rhf
from qlab.engine.hf.scf import solve
from qlab.types import Molecule


def _n2(r_angstrom: float) -> Molecule:
    return Molecule.from_string(f"N 0 0 0; N 0 0 {r_angstrom}")


def _su() -> Molecule:
    return Molecule.from_string("O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587")


# OLCULDU: her R icin (tek baslangicli solve, cok baslangicli arama).
# Arama 8 tohumla kosuldu; 16 ve 32 tohum AYNI sonucu veriyor, yani 8
# yeterlidir ve bu SECILMEDI, OLCULDU.
_N2_OLCUM = {
    1.10: (-106.769673858, -107.496500512),
    1.30: (-106.935653933, -107.433870690),
    1.50: (-106.959406679, -107.282763473),
    1.80: (-106.728309698, -107.138245990),
    2.00: (-106.772612763, -107.067294617),
}


@pytest.mark.slow
def test_TEK_baslangicli_solve_TABAN_DURUM_BULMUYOR():
    """OLCULDU: N2 1.10 A'da depo solve()'u gercek RHF taban durumundan
    0.726827 hartree (19.78 eV) YUKARIDA bir cozume yakinsiyor -- ve
    converged=True, sekiz iterasyonda.

    Bu bir UHF meselesi DEGIL: bulunan cozumde spin kavrami hic devreye
    girmiyor, SCF yalnizca baska bir duragan noktaya oturuyor.
    """
    tek = solve(_n2(1.10), "sto-3g", stability=False, max_iterations=5000)
    arama = search_rhf(_n2(1.10), "sto-3g", seeds=8)

    assert tek.provenance.converged is True  # SESSIZCE yakinsiyor
    assert tek.energy_total == pytest.approx(-106.769673858, abs=1e-8)
    assert arama.energy == pytest.approx(-107.496500512, abs=1e-8)
    assert (tek.energy_total - arama.energy) == pytest.approx(0.726827, abs=1e-5)


def test_R_110_daki_sebep_DIIS_TEK_BASINA():
    """AYIRICI OLCUM: 1.10 A'daki kusurun sebebi TOHUM DEGIL, DIIS.

    `@pytest.mark.slow` DEGIL, bilerek: olculdu, bu test 0.46 s suruyor
    cunku arama YAPMIYOR -- yalnizca iki solve() cagiriyor. Kural
    "saniye alti cipalar varsayilan seritte" oldugu icin buraya aittir,
    ve modulun asil AYIRICI olcumu budur.

    OLCULDU: ayni molekulde, ayni (varsayilan) baslangic tahminiyle
      DIIS'li  -> -106.769673858
      DIIS'siz -> -107.496500512  (= aramanin buldugu taban durum)
    Yani orada hicbir tohum degisikligine gerek yok; DIIS'i kapatmak
    yetiyor.

    Bu test o ayrimi kilitler -- yoksa iki farkli kusur tek bir "sikisma"
    hikayesine karisir.
    """
    diisli = solve(_n2(1.10), "sto-3g", stability=False, max_iterations=5000)
    diissiz = solve(
        _n2(1.10), "sto-3g", stability=False, max_iterations=20000, use_diis=False
    )
    assert diissiz.energy_total == pytest.approx(-107.496500512, abs=1e-8)
    assert (diisli.energy_total - diissiz.energy_total) == pytest.approx(
        0.726827, abs=1e-5
    )


@pytest.mark.slow
@pytest.mark.parametrize("r", [1.30, 1.50, 1.80, 2.00])
def test_R_130_USTUNDE_sebep_BASLANGIC_TAHMINI(r):
    """AYIRICI OLCUM: 1.30 A ve otesinde DIIS MASUM.

    OLCULDU: bu dort R'nin dordunde de DIIS'li ve DIIS'siz kosumlar BIREBIR
    ayni enerjiyi veriyor (fark < 1e-9). Buna ragmen cok-baslangicli arama
    0.278...0.498 hartree asagida cozumler buluyor. Yani buradaki kusurun
    sebebi iterasyon semasi degil BASLANGIC TAHMINIDIR.
    """
    diisli = solve(_n2(r), "sto-3g", stability=False, max_iterations=5000)
    diissiz = solve(
        _n2(r), "sto-3g", stability=False, max_iterations=20000, use_diis=False
    )
    assert abs(diisli.energy_total - diissiz.energy_total) < 1e-9  # DIIS masum

    tek_beklenen, arama_beklenen = _N2_OLCUM[r]
    arama = search_rhf(_n2(r), "sto-3g", seeds=8)
    assert diisli.energy_total == pytest.approx(tek_beklenen, abs=1e-8)
    assert arama.energy == pytest.approx(arama_beklenen, abs=1e-8)
    assert 0.27 < (diisli.energy_total - arama.energy) < 0.50


@pytest.mark.slow
def test_ARAMA_kendi_kendini_denetliyor():
    """Bir enerjinin "en dusuk" oldugu iddiasi, KAC ADAY arasindan
    secildigi bilinmeden anlamsizdir.

    OLCULDU (8 tohum): N2 R=2.00'de birden cok ayri cozum bulunuyor.
    """
    arama = search_rhf(_n2(2.00), "sto-3g", seeds=8)
    assert arama.n_seeds == 8
    assert arama.n_converged == 8
    assert arama.n_distinct == 4
    # Yayilim ALT VE UST sinirla kilitli. Yalniz alt sinir yazmak yetmiyor:
    # olculdu, yakinsamayan tohumlari 0.0 enerjiyle sayan mutant yayilimi
    # 107'ye ciKariyor ve "spread > 0.2" ondan da geciyordu.
    assert arama.spread == pytest.approx(0.294682, abs=1e-5)
    assert "ayri cozum" in arama.as_line()


@pytest.mark.slow
def test_VARSAYILAN_tohum_sayisi_YETIYOR():
    """Varsayilan `seeds` degeri de kilitli olmali.

    OLCULDU: varsayilan 8 tohum R=1.10'da taban durumu buluyor
    (-107.496500512); `seeds=1` ise yalniz sifir bias'i kosar ve depo
    solve()'unun sikistigi degeri (-106.769673858) verir.

    Bu test olmadan varsayilani 1'e dusuren mutant on bir testten de
    geciyordu -- cunku butun testler `seeds=` acikca veriyordu.
    """
    varsayilan = search_rhf(_n2(1.10), "sto-3g")
    assert varsayilan.n_seeds >= 8
    assert varsayilan.energy == pytest.approx(-107.496500512, abs=1e-8)

    tek = search_rhf(_n2(1.10), "sto-3g", seeds=1)
    assert tek.energy == pytest.approx(-106.769673858, abs=1e-8)


def test_YAKINSAMAYAN_tohumlar_RAPORA_GIRMIYOR():
    """`enerjiler` YALNIZ yakinsayan kosumlari icermeli.

    OLCULDU: N2 R=2.00'de `max_iterations=10` ile sekiz tohumun UCU
    yakinsiyor, ucu de AYNI cozume -- yayilim 7.105e-14. Yakinsamayanlari
    sessizce 0.0 enerjiyle sayan bir surumde yayilim ~106.77'ye firlardi.

    (Sinir 0.0 DEGIL 1e-12: yayilim tam sifir sanilmisti, cunku ilk olcum
    `%.6f` ile basilmisti ve 7.105e-14'u "0.000000" gosteriyordu. Sayi
    dogruydu, okunusu yanlisti.)

    Bu testin ayri durmasi gerekiyordu: normal kosuda (max_iterations=2000)
    N2 R=2.00'de SEKIZ tohumun SEKIZI de yakinsiyor, dolayisiyla o testte
    hatali muhasebe HIC TETIKLENMIYOR -- olculdu, ilgili mutant oradan
    geciyordu.
    """
    arama = search_rhf(_n2(2.00), "sto-3g", seeds=8, max_iterations=10)
    assert arama.n_converged == 3
    assert arama.n_distinct == 1
    assert arama.spread < 1e-12
    assert arama.energy == pytest.approx(-106.772612763, abs=1e-8)


def test_HICBIRI_yakinsamazsa_HATA_firlatiliyor():
    """Sessizce bos rapor donmez.

    OLCULDU: `max_iterations=5` ile sekiz tohumun hicbiri yakinsamiyor.
    """
    from qlab.errors import SCFConvergenceError

    with pytest.raises(SCFConvergenceError):
        search_rhf(_n2(2.00), "sto-3g", seeds=8, max_iterations=5)


def test_KAPALI_KABUK_dengede_sikisma_YOK():
    """DURUST DENGE: kusur her yerde degil -- ve bu HIZLI seritte kalir.

    OLCULDU: su molekulunde tek baslangicli solve() zaten taban durumu
    buluyor; 16 tohumluk arama ile fark 5.68e-14.
    """
    tek = solve(_su(), "sto-3g", stability=False)
    arama = search_rhf(_su(), "sto-3g", seeds=4)
    assert abs(tek.energy_total - arama.energy) < 1e-12
    assert arama.n_distinct == 1


def test_initial_bias_VARSAYILANI_DAVRANISI_DEGISTIRMIYOR():
    """REGRESYON MUHAFIZI: `initial_bias=None` yolu birebir eski davranis.

    Yeni bir parametre eklemenin bedeli, mevcut 801 testin sessizce baska
    bir sayiya kaymasi olabilirdi. Bu test onu kilitler.
    """
    for mol in (_su(), Molecule.from_string("H 0 0 0; H 0 0 0.74")):
        acik = solve(mol, "sto-3g", stability=False)
        sifir = solve(mol, "sto-3g", stability=False, initial_bias=None)
        assert acik.energy_total == sifir.energy_total  # BIREBIR


def test_initial_bias_SEKLI_denetleniyor():
    from qlab.errors import InvalidInputError

    with pytest.raises(InvalidInputError, match="initial_bias"):
        solve(_su(), "sto-3g", stability=False, initial_bias=np.zeros(3))


def test_TOHUMLAR_TEKRARLANABILIR():
    """Bir arama protokolu TEKRARLANABILIR olmak zorunda.

    `seed_biases` icindeki rastgele desenlerin tohumu SABITTIR. Sabit
    olmasaydi ayni cagri farkli sonuc verebilir ve "en dusuk" iddiasi
    kosudan kosuya degisirdi -- olculdu, mutasyon sinavinda serbest
    tohumlu surum sekiz testten de geciyordu, yani baska hicbir test bunu
    korumuyor.
    """
    from qlab.engine.hf.multistart import seed_biases

    # DETERMINISTIK desen sayisi OLCULDU: n_basis = 7 ve 10 icin 19 tane.
    # Bu yuzden 12 tohum istemek rastgele kisma HIC ULASMAZ ve test bosuna
    # gecerdi -- olculdu, serbest tohumlu mutant o halinden geciyordu.
    n_det = 19
    a = seed_biases(10, n_det + 6)
    b = seed_biases(10, n_det + 6)
    assert len(a) == n_det + 6
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    # ...ve rastgele kisim GERCEKTEN kosuldu (deterministik desenlerin otesi)
    assert not np.array_equal(a[n_det], np.zeros(10))
    assert not np.array_equal(a[n_det], a[n_det + 1])
