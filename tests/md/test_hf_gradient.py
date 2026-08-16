"""HF sayısal gradyanı -- Plan D'nin dördüncü kuvvet alanı.

Bu dosyanın sınadığı şey KUVVETİN DOĞRULUĞU, dinamik değil (o test_aimd.py).
Referans, gradyanın KENDİSİNDEN bağımsız üretilir: Richardson ekstrapolasyonu
(h ve h/2 ile 4. mertebe). Analitik HF gradyanı (Pulay kuvvetleri) bu depoda
YOKTUR, bu yüzden analitik bir referansa karşı sınama İMKÂNSIZDIR ve
sınanmıyor -- dürüst olumsuz.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.forcefields import HartreeFockGradient
from qlab.errors import InvalidInputError

#: STO-3G'nin H2 için GERÇEK denge bağ uzunluğu (Brent, xtol 1e-12).
#: 0.7156 A DEĞİL -- o, tarama ızgarasının eseridir (bkz. README).
H2_RE_BOHR = 1.3459193602
H2_E_MIN = -1.117505885116


def _h2(r_bohr: float) -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r_bohr]])


def test_hf_enerjisi_bilinen_minimumu_veriyor():
    """Kuvvet alanının döndürdüğü enerji SCF'nin kendisiyle aynı olmalı.

    Sayı LİTERAL: STO-3G'nin H2 minimumu -1.117505885116 hartree.
    """
    alan = HartreeFockGradient(["H", "H"])
    enerji, _ = alan(_h2(H2_RE_BOHR))
    assert enerji == pytest.approx(H2_E_MIN, abs=1e-9, rel=0.0)


def test_gradyan_Richardson_referansina_oturuyor():
    """Merkezi fark, 4. mertebeden BAĞIMSIZ referansa karşı.

    ÖLÇÜLDÜ (h=1e-5, referans = Richardson h=1e-3/5e-4):
      r = 1.3459 bohr -> hata 1.739e-12
      r = 1.6         -> hata 1.421e-11
      r = 2.0         -> hata 7.809e-12
    SINIR 1e-9 LİTERAL: ölçülen en kötü hata 1.42e-11, marj 70x.
    HANGİ HATAYI YAKALAR (ölçüldü): h=1e-3 (1.34e-07, sınırın 134 katı),
    h=1e-2 (1.34e-05), ileri fark (H2O'da 1.44e-07).
    """
    alan = HartreeFockGradient(["H", "H"], step=1e-5)

    def _kuvvet_z(r: float, h: float) -> float:
        return float(HartreeFockGradient(["H", "H"], step=h)(_h2(r))[1][1, 2])

    for r in (H2_RE_BOHR, 1.6, 2.0):
        kaba = _kuvvet_z(r, 1e-3)
        ince = _kuvvet_z(r, 5e-4)
        referans = (4.0 * ince - kaba) / 3.0
        assert abs(float(alan(_h2(r))[1][1, 2]) - referans) < 1e-9


def test_dengede_kuvvet_h_ye_BAGLI_olarak_sifira_gidiyor():
    """|F(r_e)| TEK BİR SAYI DEĞİLDİR -- h ile ölçeklenir.

    ÖLÇÜLDÜ: h=1e-3'te |F| = 3.06e-07, h=1e-4'te 3.84e-09. Sabit eşikli bir
    assert h değişince kırılır; bu yüzden sınır h'ye BAĞLANIR. Kesme hatası
    h^2 ile gider ve ölçülen katsayı ~0.38, yani `|F| < 5*h^2` sınırı 13x
    marj bırakır.
    """
    for h in (1e-3, 1e-4):
        _, kuvvet = HartreeFockGradient(["H", "H"], step=h)(_h2(H2_RE_BOHR))
        assert float(np.abs(kuvvet).max()) < 5.0 * h**2


def test_kuvvetler_newtonun_ucuncu_yasasina_uyuyor():
    """Toplam kuvvet sıfır olmalı; yoksa kütle merkezi kendiliğinden hızlanır.

    Gradyanın İŞARET ve EŞLEŞTİRME hatalarını yakalar -- enerji testi yakalamaz.
    """
    _, kuvvet = HartreeFockGradient(["H", "H"])(_h2(1.6))
    np.testing.assert_allclose(kuvvet.sum(axis=0), np.zeros(3), rtol=0.0, atol=1e-9)


def test_adim_basina_SCF_sayisi_oteleme_kisayolunu_kullaniyor():
    """Maliyet sözleşmesi: 6(N-1)+1 SCF, 6N+1 DEĞİL.

    Enerji yalnız BAĞIL konumlara bağlı olduğundan son atomun kuvveti
    ötekilerden çıkar. ÖLÇÜLDÜ: H2'de 13 -> 7 SCF (0.341x maliyet, MD adımı
    15.9 -> 8.5 ms); H2O'da 19 -> 13 (0.687x).

    KISAYOL SADECE UCUZ DEĞİL, DAHA DOĞRU: tam alanda |sum F| H2'de
    1.110e-12, H2O'da 9.237e-10; kısayolda TAM 0.000e+00.

    Bu sayı sessizce büyürse maliyet ikiye katlanır ve hiçbir fizik testi
    bunu görmez.
    """
    alan = HartreeFockGradient(["H", "H"])
    alan(_h2(H2_RE_BOHR))
    assert alan.n_scf_calls == 7


def test_kisayol_TAM_alanla_ayni_kuvveti_veriyor():
    """Kısayol bir yaklaşım DEĞİL, özdeşliktir -- ölçülerek gösterilir.

    ÖLÇÜLDÜ: maks|F_tam - F_kısayol| H2'de 1.110e-12, H2O'da 9.237e-10, ve
    bu fark TAM OLARAK |sum F_tam|'a eşit. SINIR 1e-8: H2O ölçümünün 10.8 katı.
    """
    alan = HartreeFockGradient(["H", "H"])
    _, kisayol = alan(_h2(1.6))

    tam = np.zeros((2, 3))
    h = alan.step
    for i in range(2):
        for k in range(3):
            a = _h2(1.6); a[i, k] += h
            b = _h2(1.6); b[i, k] -= h
            tam[i, k] = -(alan._energy(a) - alan._energy(b)) / (2.0 * h)

    assert float(np.abs(tam - kisayol).max()) < 1e-8


def test_kararlilik_analizi_KAPALI():
    """Her adımda kararlılık analizi anlamsız ve ÖLÇÜLDÜ: H2'de +%31 maliyet.

    Açık kalırsa bir AIMD koşusu sessizce üçte bir yavaşlar.
    """
    assert HartreeFockGradient(["H", "H"]).stability is False


def test_geometri_kapisi_iyonik_dala_atlamayi_engelliyor():
    """R > 21.109 bohr'da RHF iyonik H-/H+ dalına atlıyor ve PES 0.3636
    hartree SIÇRIYOR (ölçüldü).

    Bağlı H2 dinamiği oraya asla girmez (ölçülen salınım 1.26-1.45 bohr),
    yani bu bir bloker DEĞİL -- ama sessiz kalmamalı.
    """
    with pytest.raises(InvalidInputError, match="21"):
        HartreeFockGradient(["H", "H"])(_h2(25.0))


def test_gecersiz_girdileri_reddediyor():
    with pytest.raises(InvalidInputError, match="sembol"):
        HartreeFockGradient(["H"])
    for kotu in (0.0, -1e-5, np.nan, np.inf):
        with pytest.raises(InvalidInputError, match="adım"):
            HartreeFockGradient(["H", "H"], step=kotu)
    alan = HartreeFockGradient(["H", "H"])
    for kotu in (np.zeros((2, 2)), np.zeros(3), np.zeros((3, 3))):
        with pytest.raises(InvalidInputError):
            alan(kotu)
