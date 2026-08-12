import numpy as np
import pytest
from hypothesis import given, strategies as st

from qlab import units


def test_bohr_to_angstrom_known_value():
    # rel=0.0: fiilen uygulanan, varsayılan rel=1e-6'nın (=5.3e-7) yazılan
    # abs=1e-12'yi ezdiği haldi. Ölçülen sapma tam 0.0 (bit-bit eşit; 1.0
    # çarpımı sabiti değiştirmiyor) olduğundan yazarın abs=1e-12'si korunuyor.
    assert units.bohr_to_angstrom(1.0) == pytest.approx(0.529177210903, abs=1e-12, rel=0.0)


def test_hartree_to_ev_known_value():
    # rel=0.0: fiilen uygulanan, varsayılan rel=1e-6'nın (=2.7e-5) yazılan
    # abs=1e-9'u ezdiği haldi. Ölçülen sapma tam 0.0 (bit-bit eşit) olduğundan
    # yazarın abs=1e-9'u korunuyor.
    assert units.hartree_to_ev(1.0) == pytest.approx(27.211386245988, abs=1e-9, rel=0.0)


def test_hydrogen_ground_state_in_ev():
    """H atomu taban durumu -0.5 hartree = -13.6057 eV."""
    assert units.hartree_to_ev(-0.5) == pytest.approx(-13.6057, abs=1e-3)


def test_conversion_works_on_arrays():
    result = units.bohr_to_angstrom(np.array([1.0, 2.0, 3.0]))
    assert isinstance(result, np.ndarray)
    # rel=0.0: fiilen uygulanan, varsayılan rel=1e-6'nın (~1.06e-6) yazılan
    # abs=1e-12'yi ezdiği haldi. Ölçülen sapma tam 0.0 (bit-bit eşit)
    # olduğundan yazarın abs=1e-12'si korunuyor.
    assert result[1] == pytest.approx(2 * 0.529177210903, abs=1e-12, rel=0.0)


@given(st.floats(min_value=1e-6, max_value=1e6, allow_nan=False))
def test_length_roundtrip(value):
    # abs=0.0: yazılan rel=1e-12, varsayılan abs=1e-12 ile birleşince küçük
    # uçta (value ~1e-6) fiilen ~1e-6 göreli sapmaya izin veriyordu. Dönüşüm
    # sadece iki float çarpımı (C ve 1/C) olduğundan göreli sapma büyüklükten
    # bağımsız: [1e-6, 1e6] aralığında 40000+ değerlik taramada (uçlar, 2'nin
    # kuvvetleri, rastgele) en kötü ölçülen ~2.22e-16. Yazarın 1e-12'si bu
    # ölçüme göre ~4500x gevşek olurdu (gerçekten bağlayıcı olmazdı), bu
    # yüzden rel=1e-15'e sıkıştırıldı: ~4.5x pay, "genuinely binding".
    assert units.angstrom_to_bohr(units.bohr_to_angstrom(value)) == pytest.approx(
        value, rel=1e-15, abs=0.0
    )


@given(st.floats(min_value=1e-6, max_value=1e6, allow_nan=False))
def test_energy_roundtrip(value):
    # abs=0.0: test_length_roundtrip ile aynı gerekçe. En kötü ölçülen göreli
    # sapma ~1.89e-16; rel=1e-15 ~5.3x payla bağlayıcı.
    assert units.ev_to_hartree(units.hartree_to_ev(value)) == pytest.approx(
        value, rel=1e-15, abs=0.0
    )


def test_md_sabitleri_codata_2018_ile_uyusuyor():
    """MD'nin üç sabiti; hepsi türetilmiş, hiçbiri elle yazılmış değil.

    HARTREE_TO_WAVENUMBER bir zamanlar çıplak literaldi ve bu dokstring
    yalan söylüyordu; artık E_h/(h c)'den türetiliyor. ÖLÇÜLDÜ: türetilen
    değer 219474.63136319647, CODATA tablo değerinden bağıl farkı 1.60e-14
    -- aşağıdaki rel=1e-10 sınırının 6000 katı altında.

    HANGİ HATAYI YAKALAR (ölçüldü): kB ters çevrilmiş (h/kB) 1 test;
    dalga sayısında c unutulmuş (E_h/h) 1 test.
    """
    from qlab.units import (
        AMU_TO_ELECTRON_MASS,
        HARTREE_TO_WAVENUMBER,
        KB_HARTREE_PER_KELVIN,
    )

    assert KB_HARTREE_PER_KELVIN == pytest.approx(3.166811563455608e-06, rel=1e-12, abs=0.0)
    assert AMU_TO_ELECTRON_MASS == pytest.approx(1822.888486209, rel=1e-10, abs=0.0)
    assert HARTREE_TO_WAVENUMBER == pytest.approx(219474.6313632, rel=1e-10, abs=0.0)


def test_amu_donusumu_gercek_cekirdek_kutlelerini_veriyor():
    """Kütle dönüşümü MD'nin en tehlikeli hatasının tek savunmasıdır.

    ÖLÇÜLDÜ: `amu_to_electron_mass` çağrılmayan kod 1822.888486x hafif bir
    molekül simüle eder. H2'nin periyodu 313.322877 a.u. yerine 7.338580
    a.u. çıkar -- çarpan 42.695298 = sqrt(AMU_TO_ELECTRON_MASS) -- ve
    HİÇBİR ŞEY PATLAMAZ. dt periyoda göre seçildiği için MD "çalışır
    görünür", üstelik enerjiyi de KORUR: N=200 adım/periyotta
    max|dE|/E0 = 2.467401e-04 = pi^2/N^2, ve bu sayı doğru ile hatalı
    kütlede BİREBİR AYNIDIR (bağıl salınım yalnız N'ye bağlıdır, kütleye
    değil). Titreşim frekansı 4401.39 cm^-1 yerine 187918.70 cm^-1 okunur.

    Sayılar LİTERAL: sabite karşı yazılsalardı sabitle birlikte hareket
    ederlerdi ve hiçbir zaman kırılamazlardı. ÖLÇÜLDÜ:
    `AMU_TO_ELECTRON_MASS = 1.0` mutantı bu dosyada 2 testi kırıyor.

    TOLERANS: proton için abs=1e-4; ölçülen hata 4.41e-07, marj 227x. Bir
    önceki hali (kırpılmış 1.007276 + abs=1e-3) bütçesinin %87'sini kırpma
    hatasına harcıyordu ve CODATA protonunu kırpılmış protondan AYIRT
    EDEMİYORDU -- ölçüldü: kırpılmış sabit artık 1 testi kırmızı yapıyor.
    """
    from qlab.units import amu_to_electron_mass

    assert amu_to_electron_mass(1.007276466621) == pytest.approx(
        1836.152673, abs=1e-4, rel=0.0
    )
    assert amu_to_electron_mass(1.00782503207) == pytest.approx(1837.1526, abs=1e-3, rel=0.0)
    assert amu_to_electron_mass(39.948) == pytest.approx(72820.749, abs=1e-2, rel=0.0)
    np.testing.assert_allclose(
        amu_to_electron_mass(np.array([1.00782503207, 39.948])),
        np.array([1837.1526, 72820.749]),
        rtol=0.0,
        atol=1e-2,
    )
