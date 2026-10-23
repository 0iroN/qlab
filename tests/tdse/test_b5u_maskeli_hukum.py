"""B5-U (14.10.2026): maskeli TDSE koşusunun GERÇEK üniterlik hükmü.

Soğurucu maske normu bilerek düşürür; |N - 1| maskeli koşuda üniterlik
hakkında hiçbir şey söylemez ve grup 1'den (B5) beri panel orada
✗ + YARGILANMADI basıyordu. Ölçülebilen şey maskeden ÖNCEKİ üniter adımdır:
`evolve_olcumlu` her adımda ‖Uψ‖²/‖ψ‖² oranını ve norm bütçesini
(N + soğurulan − N0) ölçer, `tdse.factory.provenance_for` maskesiz yolun
eşiğiyle (1e-8) hüküm kurar.

SINIFLAR (dürüst etiket):
- ÇAPA (Parseval): kinetik çarpana (1 + ε) enjekte edilince adım oranı TAM
  OLARAK (1 + ε)² − 1 olmalıdır — numpy'nin fft/ifft çifti normu korur, yani
  beklenen sayı yöntemin kendisinden değil Parseval özdeşliğinden gelir.
- ÇAPA (üniterlik): kusursuz Strang adımı normu yuvarlama düzeyinde korur;
  paket %99.99 soğurulurken de adım oranı yuvarlama tabanında kalmalıdır.
- ÖLÇÜT KİLİDİ: iki ölçünün her biri ötekinin göremediği bir kusuru yakalar
  (yavaş birikim -> yalnız bütçe; soğurma sonrası kalan küçük normdaki kusur
  -> yalnız göreli adım oranı). Bunlar ölçütün TANIMINI kilitler.

Senaryo kamu yükleyicisinden (`load_scenario`) ve fabrikadan (`build`)
kurulur: 256 nokta, bariyerli, 800 adım; paket bariyerden geçip iki kenarda
soğurulur (N 1 -> 6.9e-05). Koşu ~10 ms.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.tdse import factory
from qlab.engine.tdse.propagator import (
    SplitOperatorPropagator,
    UniterlikOlcumu,
    evolve,
    evolve_olcumlu,
)
from qlab.errors import InvalidInputError
from qlab.scenario.loader import load_scenario

_YAML = """\
name: maskeli kucuk bariyer
grid: {n: 256, length_bohr: 100.0}
potential: {kind: barrier, params: {height: 0.8, width: 2.0, center: 0.0}}
packet: {x0_bohr: -10.0, k0: 2.0, sigma_bohr: 4.0}
dt: 0.05
n_steps: 800
save_every: 100
"""

_ESIK = 1e-8  # tdse/factory.py::_NORM_TOLERANCE -- maskesiz yolun eşiği


@pytest.fixture
def maskeli(tmp_path):
    yol = tmp_path / "maskeli.yaml"
    yol.write_text(_YAML + "absorbing_width_bohr: 15.0\n", encoding="utf-8")
    return load_scenario(yol)


@pytest.fixture
def maskesiz(tmp_path):
    yol = tmp_path / "maskesiz.yaml"
    yol.write_text(_YAML, encoding="utf-8")
    return load_scenario(yol)


def _kos(senaryo, duzelt=None, sinif=None):
    grid, _, psi0, prop = factory.build(senaryo)
    if sinif is not None:
        prop = sinif(
            grid=prop.grid, potential=prop.potential, dt=prop.dt, mass=prop.mass, mask=prop.mask
        )
    if duzelt is not None:
        duzelt(prop)
    kareler, zamanlar, olcum = evolve_olcumlu(
        prop, psi0, n_steps=senaryo.n_steps, save_every=senaryo.save_every
    )
    son_norm = float(np.sum(np.abs(kareler[-1]) ** 2) * grid.dx)
    prov = factory.provenance_for(senaryo, abs(son_norm - 1.0), olcum)
    return kareler, zamanlar, olcum, son_norm, dict(prov.as_lines()), prov


def test_KUSURSUZ_adim_paket_sogurulurken_hukum_TIK_ve_sogurulan_AYRI(maskeli):
    """ÇAPA (üniterlik): paket %99.99 soğurulur, maske öncesi adım normu korunur.

    Ölçüldü (14.10.2026): adım oranı en çok 6.7e-16, bütçe 8.0e-14,
    soğurulan 0.99993 ve N_son + soğurulan − 1 ~ 1e-14. Eşikler ölçümün
    ~150 ve ~120 katı (OpenBLAS çekirdeğine bağlı son-ulp KİLİTLENMEZ).
    KIRMIZIYA DÖNDÜRENLER: oranı maskeden SONRA ölçmek (oran ~ soğurma),
    bütçeye soğurulanı katmamak (bütçe ~ 1), oranı adıma giren norm yerine
    N0'a bölmek, yarım adım fazında `j` düşmüş (gerçel üstel) bir `_half_v`.
    """
    _, _, olcum, son_norm, satirlar, prov = _kos(maskeli)

    assert isinstance(olcum, UniterlikOlcumu) and olcum.adim_sayisi == 800
    assert son_norm < 1e-3, "paket maskeye varmadı -- senaryo sınavı kaybeder"
    assert olcum.en_buyuk_adim_sapmasi <= 1e-13
    assert olcum.en_buyuk_butce_sapmasi <= 1e-11
    assert olcum.sogurulan_norm == pytest.approx(1.0 - son_norm, abs=1e-11, rel=0.0)

    assert prov.converged is True
    assert satirlar["Yakınsama"].startswith("✓")
    assert "premask_step_deviation=" in satirlar["Yakınsama"]
    assert "absorbed_norm=" in satirlar["Yakınsama"]
    tani = satirlar["Yakınsama tanısı"]
    assert tani.startswith("maske öncesi üniter adım normu yargılandı")
    assert "EŞİĞİ AŞTI" not in tani and "YARGILANMADI" not in tani
    assert "fizik değil" in tani


def test_PARSEVAL_capasi_enjekte_kinetik_carpan_TAM_olculur_ve_hukum_X(maskeli):
    """ÇAPA (Parseval): kinetik çarpan (1 + ε) ile ölçeklenince her adımın
    oranı tam (1 + ε)² − 1'dir; ölçülen sayı ona yuvarlama düzeyinde eşit
    olmalı ve hüküm ✗ olmalıdır (2e-7 > 1e-8).

    Ölçüldü: 2.0000001077e-07 / beklenen 2.0000001011e-07 (bağıl 3.3e-9 =
    yuvarlama 6.7e-16 / 2e-7); tolerans bağıl 1e-7. KIRMIZIYA DÖNDÜRENLER:
    ölçülen hükmü her zaman ✓ yapan, eşiği 1e-6'ya gevşeten, oranı karesiz
    (‖Uψ‖/‖ψ‖) alan mutantlar.
    """
    eps = 1e-7

    def bozuk_kinetik(prop):
        prop._kinetic = prop._kinetic * (1.0 + eps)

    _, _, olcum, _, satirlar, prov = _kos(maskeli, duzelt=bozuk_kinetik)

    beklenen = (1.0 + eps) ** 2 - 1.0
    assert olcum.en_buyuk_adim_sapmasi == pytest.approx(beklenen, rel=1e-7, abs=0.0)
    assert prov.converged is False
    assert satirlar["Yakınsama"].startswith("✗")
    tani = satirlar["Yakınsama tanısı"]
    assert tani.startswith("maske öncesi üniter adım normu yargılandı ve EŞİĞİ AŞTI")
    assert "KORUMUYOR" in tani


def test_YAVAS_BIRIKIM_yalniz_BUTCEYE_gorunur(maskeli):
    """ÖLÇÜT KİLİDİ: adım başına eşiğin ALTINDA ama biriken bir sızıntı.

    ε = 2e-9 -> adım oranı 4.0e-9 (eşiğin 2.5 kat altı) ama 800 adımda bütçe
    2.0e-6 (200 kat üstü). Maskesiz yolun |N − 1| ölçütü de birikimlidir; bu
    test bütçenin o ölçütün maskeli karşılığı olduğunu kilitler. KIRMIZIYA
    DÖNDÜREN: hükmü yalnız adım oranına bağlayan mutant.
    """
    eps = 2e-9

    def sizinti(prop):
        prop._kinetic = prop._kinetic * (1.0 + eps)

    _, _, olcum, _, satirlar, prov = _kos(maskeli, duzelt=sizinti)

    assert olcum.en_buyuk_adim_sapmasi < 0.5 * _ESIK
    assert olcum.en_buyuk_butce_sapmasi > 20.0 * _ESIK
    assert prov.converged is False
    assert satirlar["Yakınsama"].startswith("✗")


class _SogurmaSonrasiKusur(SplitOperatorPropagator):
    """Yalnız adıma giren norm 1e-3'ün altına inince (1 + 5e-8) ölçekleyen U."""

    def _uniter_kisim(self, psi):
        sonuc = super()._uniter_kisim(psi)
        if float(np.vdot(psi, psi).real) * self.volume_element < 1e-3:
            sonuc = sonuc * (1.0 + 5e-8)
        return sonuc


def test_SOGURMA_SONRASI_kusur_yalniz_GORELI_ADIM_oranina_gorunur(maskeli):
    """ÖLÇÜT KİLİDİ: kusur yalnız paketin %99.9'u soğurulduktan sonra etkin.

    Kalan küçük norm üzerinde adım oranı 1.0e-7 (eşiğin 10 katı) ama bütçe
    mutlak olasılık biriminde 2.5e-9 (eşiğin 4 kat altı): bütçe tek başına
    bu kusuru GÖRMEZ. KIRMIZIYA DÖNDÜREN: hükmü yalnız bütçeye bağlayan
    mutant.
    """
    _, _, olcum, _, satirlar, prov = _kos(maskeli, sinif=_SogurmaSonrasiKusur)

    assert olcum.en_buyuk_adim_sapmasi > 5.0 * _ESIK
    assert olcum.en_buyuk_butce_sapmasi < 0.5 * _ESIK
    assert prov.converged is False
    assert satirlar["Yakınsama"].startswith("✗")


def test_OLCUMLU_YOL_kareleri_evolve_ile_BIT_BIT_ayni_ve_maskesizde_olcum_YOK(maskeli, maskesiz):
    """Ölçüm dinamiği değiştirmez: maskeli koşuda `evolve_olcumlu` kareleri
    `evolve` (yani `step`) kareleriyle BİT BİT aynıdır; maskesiz koşuda ölçüm
    None'dır ve kareler yine bit bit aynıdır. KIRMIZIYA DÖNDÜRENLER: maskeyi
    `_maskeli_adim`de unutmak ya da iki kez uygulamak."""
    for senaryo, maske_var in ((maskeli, True), (maskesiz, False)):
        grid, _, psi0, prop = factory.build(senaryo)
        k1, z1 = evolve(prop, psi0, n_steps=senaryo.n_steps, save_every=senaryo.save_every)
        k2, z2, olcum = evolve_olcumlu(
            prop, psi0, n_steps=senaryo.n_steps, save_every=senaryo.save_every
        )
        assert np.array_equal(k1, k2) and np.array_equal(z1, z2)
        assert (olcum is not None) is maske_var


def test_PROVENANCE_kapilari_olcum_YANLIS_kosuya_verilemez(maskeli, maskesiz):
    """Maskesiz koşuya ölçüm verilemez (orada ölçüt |N − 1|'dir); adım sayısı
    tutmayan ölçüm reddedilir; maskeli koşuda ölçüm verilmezse hüküm hâlâ
    ✗ YARGILANMADI'dır (= denenmedi; YARGILANAMAZ değil)."""
    olcum = UniterlikOlcumu(800, 1e-16, 1e-14, 0.5)
    with pytest.raises(InvalidInputError, match="maskesiz"):
        factory.provenance_for(maskesiz, 1e-12, olcum)
    # V3 (23.10.2026): eksik DE fazla adımlı ölçüm reddedilir (`!=`; `<` olsaydı 801 geçerdi)
    for adim in (799, 801):
        with pytest.raises(InvalidInputError, match=f"{adim} adım"):
            factory.provenance_for(maskeli, 0.5, UniterlikOlcumu(adim, 1e-16, 1e-14, 0.5))

    satirlar = dict(factory.provenance_for(maskeli, 0.5).as_lines())
    assert satirlar["Yakınsama"].startswith("✗")
    assert satirlar["Yakınsama tanısı"].startswith("YARGILANMADI:")
    assert "YARGILANAMAZ" not in satirlar["Yakınsama tanısı"]

    # NaN ölçüm ✓ basamaz (karşılaştırma `<=` ile kurulur; `not >` olsaydı geçerdi) --
    # V3: İKİ alanda da ayrı ayrı (eskiden yalnız adım alanı sınanıyordu).
    for nan_olcum in (UniterlikOlcumu(800, float("nan"), 1e-14, 0.5),
                      UniterlikOlcumu(800, 1e-16, float("nan"), 0.5)):
        assert factory.provenance_for(maskeli, 0.5, nan_olcum).converged is False, nan_olcum


# ------------------------------------------------- düzeltici (grup 9, 23.10.2026): V1 V2 V3 V5


def _kinetik_carpan(eps):
    def duzelt(prop):
        prop._kinetic = prop._kinetic * (1.0 + eps)

    return duzelt


def _sogurma_sonrasi(oran):
    """Adıma giren norm 1e-3'ün altına inince oranı TAM `oran` olan U (genlik √(1 + oran))."""
    carpan = np.sqrt(1.0 + oran)

    class _Kusur(SplitOperatorPropagator):
        def _uniter_kisim(self, psi):
            sonuc = super()._uniter_kisim(psi)
            if float(np.vdot(psi, psi).real) * self.volume_element < 1e-3:
                sonuc = sonuc * carpan
            return sonuc

    return _Kusur


@pytest.mark.parametrize(
    "eps,gecer",
    [pytest.param(2e-11, False, id="butce-2x-esik-X"), pytest.param(5e-12, True, id="butce-0.5x-esik-TIK")],
)
def test_ESIK_KILIDI_BUTCE_1e8_esigin_IKI_katinda_X_YARISINDA_TIK(maskeli, eps, gecer):
    """ÖLÇÜT KİLİDİ (V1): bütçe eşiği TAM OLARAK maskesiz yolun 1e-8'i.

    Kinetik çarpana (1 + ε) -> adım oranı 2ε (eşiğin ≥ 250 kat altı), bütçe ε ile
    doğrusal: ölçüldü (23.10.2026) ε 5e-12 / 1e-11 / 2e-11 -> 5.07e-9 / 1.015e-8 /
    2.03e-8. Bu iki hücre eşiği [0.51, 2.03] × 1e-8 aralığına kilitler. Eskiden
    kilit penceresi ~1e-11 ile 2e-6 arasıydı: eşiği 10 kat gevşeten mutant
    (V_A1_02) ilgili 176 testin hepsinden geçiyordu.
    """
    _, _, olcum, _, satirlar, prov = _kos(maskeli, duzelt=_kinetik_carpan(eps))

    assert olcum.en_buyuk_adim_sapmasi < 0.01 * _ESIK  # yalnız bütçe yargılanıyor
    oran = olcum.en_buyuk_butce_sapmasi / _ESIK
    assert (1.5 < oran < 3.0) if not gecer else (0.3 < oran < 0.7), oran  # öncül
    assert prov.converged is gecer
    assert satirlar["Yakınsama"].startswith("✓" if gecer else "✗")


@pytest.mark.parametrize(
    "oran,gecer",
    [pytest.param(1.5e-8, False, id="adim-1.5x-esik-X"), pytest.param(5e-9, True, id="adim-0.5x-esik-TIK")],
)
def test_ESIK_KILIDI_ADIM_1e8_esigin_1_5_katinda_X_YARISINDA_TIK_ve_KOR_PENCERE(maskeli, oran, gecer):
    """ÖLÇÜT KİLİDİ (V1) + BEYANLI KÖR PENCERE (V2).

    Kusur yalnız soğurmadan sonra (N < 1e-3) etkin, adım oranı TAM `oran`; bütçe
    eşiğin ~30 kat altında kalır (ölçüldü 3.7e-10 / 1.2e-10), yani hükmü YALNIZ
    adım ölçütü verir. İki hücre adım eşiğini [0.5, 1.5] × 1e-8 aralığına kilitler
    (eşiği 5 kat gevşeten V_A1_03 eskiden hayattaydı).

    0.5x hücresi ✓'dir ve bu BEYANDIR, kusur değil: adım ölçütü ADIM BAŞINA
    ölçer, birikimi görmez. Kalan paketin normu kusursuza göre GÖRELİ ~4.6e-7
    kayar (eşiğin ~46 katı), MUTLAK olasılıkta ~3e-11 (maskesiz yolun mutlak 1e-8
    ölçütünün ~300 kat altı) -- README ve `UniterlikOlcumu` dokstringi bunu yazar.
    """
    _, _, kusursuz, son0, _, _ = _kos(maskeli)
    _, _, olcum, son, satirlar, prov = _kos(maskeli, sinif=_sogurma_sonrasi(oran))

    assert olcum.en_buyuk_adim_sapmasi == pytest.approx(oran, rel=1e-6, abs=0.0)
    assert olcum.en_buyuk_butce_sapmasi < 0.1 * _ESIK
    assert prov.converged is gecer
    assert satirlar["Yakınsama"].startswith("✓" if gecer else "✗")
    if gecer:  # kör pencerenin ölçüsü: göreli birikim eşiği aşar, mutlak birikim aşmaz
        assert abs(son / son0 - 1.0) > 10.0 * _ESIK
        assert abs(son - son0) < 0.01 * _ESIK


@pytest.mark.parametrize("olcek", [1e-4, 1e-100], ids=["psi0x1e-4", "psi0x1e-100"])
def test_GORELI_ADIM_ORANI_OLCEK_DEGISMEZ_kucuk_normda_da_PARSEVAL(maskeli, olcek):
    """ÇAPA (Parseval, ölçek): U doğrusaldır; ψ0 × c'de oran c'den bağımsızdır.

    N0 = 1e-8 ve 1e-200'de de adım oranı (1 + ε)² − 1 ve hüküm ✗ (ölçüldü 23.10.2026:
    üç ölçekte de 2.0000001077e-07). KIRMIZIYA DÖNDÜREN: "küçük normda oranı atla"
    (V_A1_09, `onceki > 1e-6`; eskiden göreli ölçüm yalnız N ≈ 6.9e-5'e kadar
    sınanıyordu ve bu mutant hayattaydı).
    """
    eps = 1e-7
    grid, _, psi0, prop = factory.build(maskeli)
    _kinetik_carpan(eps)(prop)
    kareler, _, olcum = evolve_olcumlu(prop, psi0 * olcek, n_steps=800, save_every=100)

    assert olcum.en_buyuk_adim_sapmasi == pytest.approx((1.0 + eps) ** 2 - 1.0, rel=1e-7, abs=0.0)
    son_norm = float(np.sum(np.abs(kareler[-1]) ** 2) * grid.dx) / olcek**2
    assert factory.provenance_for(maskeli, abs(son_norm - 1.0), olcum).converged is False


class _NaNEnjekte(SplitOperatorPropagator):
    """400. çağrıda U'nun çıktısına tek bir NaN koyan propagatör (sonrası tümüyle NaN)."""

    def _uniter_kisim(self, psi):
        sonuc = super()._uniter_kisim(psi)
        self._cagri = getattr(self, "_cagri", 0) + 1
        if self._cagri == 400:
            sonuc = sonuc.copy()
            sonuc[0] = np.nan
        return sonuc


def test_MOTORDA_NaN_IKI_OLCUYE_de_YAYILIR_hukum_X(maskeli):
    """V5 (koşan maksimum) + V3: koşunun ortasında doğan NaN iki ölçüde de KALIR.

    400. adımdan sonra ψ NaN'dır; NaN'dan önceki 399 adımın ölçüsü yuvarlama
    tabanındadır. NaN'ı yutan bir maksimum (`nanmax`, ya da `max(x, nan)` sırası,
    ya da koşan maksimumda `yeni != yeni` kolu yok) önceki sonlu en büyüğü döndürür
    ve hüküm ✓ olurdu -- NaN'lı karelerin yanında. V_A1_05 (`nanmax`) bu testten
    önce ilgili hızlı şeritte hayattaydı.
    """
    _, _, olcum, son_norm, satirlar, prov = _kos(maskeli, sinif=_NaNEnjekte)

    assert np.isnan(son_norm)
    assert np.isnan(olcum.en_buyuk_adim_sapmasi) and np.isnan(olcum.en_buyuk_butce_sapmasi)
    assert np.isnan(olcum.sogurulan_norm)
    assert prov.converged is False and satirlar["Yakınsama"].startswith("✗")


@pytest.mark.parametrize(
    "alanlar,desen",
    [
        pytest.param(dict(en_buyuk_adim_sapmasi=-5.0, en_buyuk_butce_sapmasi=-1.0), "negatif olamaz",
                     id="negatif-sapmalar"),
        pytest.param(dict(en_buyuk_butce_sapmasi=-1e-30), "negatif olamaz", id="negatif-butce"),
        pytest.param(dict(sogurulan_norm=-0.1), "negatif olamaz", id="negatif-sogurulan"),
        pytest.param(dict(en_buyuk_adim_sapmasi="0"), "gerçel bir sayı", id="dize-sapma"),
        pytest.param(dict(en_buyuk_butce_sapmasi=1e-14 + 0j), "gerçel bir sayı", id="karmasik-butce"),
        pytest.param(dict(sogurulan_norm=np.array(0.5)), "gerçel bir sayı", id="0d-dizi-sogurulan"),
        pytest.param(dict(en_buyuk_adim_sapmasi=False), "gerçel bir sayı", id="bool-sapma"),
        pytest.param(dict(adim_sayisi=True), "pozitif bir tam sayı", id="bool-adim"),
        pytest.param(dict(adim_sayisi=800.0), "pozitif bir tam sayı", id="float-adim"),
        pytest.param(dict(adim_sayisi=0), "pozitif bir tam sayı", id="sifir-adim"),
    ],
)
def test_UniterlikOlcumu_TUR_KAPISI_dogrudan_kurulus(alanlar, desen):
    """V3 (23.10.2026): doğrudan kuruluş (Python yolu) ölçünün tanımına uymalı.

    Ölçüldü (kapı yokken): `UniterlikOlcumu(800, -5, -1, 0.5)` panelde ✓ basıyordu,
    dize sapma `provenance_for`da ham `TypeError`, `800.0` adım kabul ediliyordu.
    Adlı `InvalidInputError` ve alanın adı.
    """
    taban = dict(adim_sayisi=800, en_buyuk_adim_sapmasi=1e-16, en_buyuk_butce_sapmasi=1e-14,
                 sogurulan_norm=0.5)
    with pytest.raises(InvalidInputError, match=f"UniterlikOlcumu\\.\\w+ {desen}"):
        UniterlikOlcumu(**(taban | alanlar))


def test_UniterlikOlcumu_TUR_KAPISI_asiri_reddetmez():
    """Aşırı reddetmeye karşı: numpy tam sayı/float, NaN ve +inf (taşan bir U'nun dürüst
    ölçümü) kabul; değer DEĞİŞTİRİLMEZ. NaN ve inf hükmü ✗ yapar (önceki testler)."""
    for olcum in (UniterlikOlcumu(np.int64(800), np.float64(1e-16), 1e-14, np.float32(0.5)),
                  UniterlikOlcumu(800, float("nan"), float("inf"), float("nan")),
                  UniterlikOlcumu(1, 0, 0.0, 0)):
        assert olcum.adim_sayisi >= 1
    assert UniterlikOlcumu(800, np.float64(1e-16), 1e-14, 0.5).en_buyuk_adim_sapmasi.__class__ is np.float64
