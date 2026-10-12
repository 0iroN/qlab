"""Arayüz: TİY faz kilidi (FİZİK ÇIPASI), Fresnel'in ÇEVİRİ DENETİMİ,
ve ölçüm aletinin kendi TAVANI.

Bu dosya Modül 5'in **ilk fizik çıpasını** taşır. Task 1 çekirdeği, Task 2
ölçütü kurdu; burada ilk kez *girilen parametrelerden okunamayan* bir sayı
ölçülüyor.

ÜÇ SINIF, ve karıştırılmıyorlar:

* **FİZİK ÇIPASI** — `delta_ps = arg(rho_p) - arg(rho_s)` tam iç yansımada.
  İki kutuplanmanın evanesan bölgede FARKLI sınır koşullarına uymasından
  doğar; dik gelişte bu sayı BİLGİSİZDİR (orada `r` reeldir).
* **ÇEVİRİ DENETİMİ** — Fresnel `|rho|`. Kırılma indisi GİRDİDİR ve `R` onun
  cebirsel sonucudur; bu sayı çözücünün arayüzü çözdüğünü gösterir, yeni bir
  fizik bilgisi üretmez (spec §2.3).
* **ŞEMA ÇIPASI / TRANSKRİPSİYON DENETİMİ** — `|rho| = 1`, `ky = 0`
  indirgemesi, dokunulan ad kümesi.

KİLİDİN ÇEKİRDEĞİ VE NEDEN GERİ OKUMA DEĞİL: ortam 1'de her iki kutuplanma
AYNI `kx1` ile yayılır, dolayısıyla prob→arayüz gidiş-dönüş fazı
`exp(-2i kx1 L)` ikisinde de aynıdır ve FARKTA TAM SADELEŞİR. Beyan edilen
hiçbir konum hesaba girmez. Bu, elenen `x_eff` adayının TAM KARŞITIDIR:
orada dal belirsizliği `round((iface-0.5-x_eff)/per)` satırıyla BEYAN EDİLEN
`iface`'ten kapatılıyordu — kilit ölçmüyor, GERİ OKUYORDU.

PENCERE (aksi yazılmadıkça bu dosyadaki her koşum): `eps1 = 4` (n1 = 2),
`eps2 = 1`, `theta = 45°` (`theta_c = 30°`), `lambda = 40` hücre, `dx = 1`,
`S = 0.4`, `N = 3600`, `npml = 60` (`m = 3`, `r0 = 1e-6`), `iface = 1500`,
`isrc = 600`, `probe_offset = 6`, **`sgf = 9`**, `t0 = 5 sg`, FFT penceresi
DİKDÖRTGEN, kayıt uzunluğu açıya göre seçilir (yankı-dışı kapı), kapı
`||rho| - 1| <= 1e-2`, kuyruk kapısı 1.6e-3 (kaydın son periyodundaki
genlik / tepe; 10.10.2026, bkz. `test_kayit_tamligi.py`). Tohum YOK: FDTD deterministtir ve tekrarlanabilirliğin
karşılığı pencerenin tam beyanıdır.

`sigma_analiz = 0.002187°` — 15 analiz/ayrıklaştırma ekseninin standart
sapması, `ey_kural` HARİÇ (o bir modelleme seçimidir; havuza katılınca sigma
1446 kat şişiyor: 3.162925°).

MALİYET: `tir_phase_difference` çağrısı başına 4 FDTD koşumu; sevk
penceresinde lambda = 16/24/36/40/54 için 1.414/1.629/1.999/2.131/2.595 s.
Bu dosyadaki ağır testler `@pytest.mark.slow` ile işaretli — eşik ölçülerek
seçildi (aşağıdaki `test_SLOW_ESIGI_OLCULDU`).
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache

import numpy as np
import pytest

from qlab.engine.em.boundary import PEC, PML
from qlab.engine.em.grid import FieldRecord, FieldState, courant_limit
from qlab.engine.em.medium import Medium
from qlab.engine.em.oblique import (
    POLARIZATIONS,
    ObliqueCoefficients,
    _egik_p_kesin_esik,
    oblique_coefficients,
    oblique_courant_limit,
    oblique_step,
    run_oblique_1d,
)
from qlab.engine.em.source import gauss_cos
from qlab.engine.em.yee import YeeCoefficients, run_1d
from qlab.errors import InvalidInputError
from qlab.observable.em_obs import (
    ObliqueWindow,
    _record_steps,
    _spektrum,
    convergence_ladder,
    fresnel_coefficients,
    fresnel_phase_difference,
    reflection_coefficient,
    tir_phase_difference,
    tir_phase_ladder,
    total_internal_reflection,
)

EPS1 = 4.0
EPS2 = 1.0
THETA = 45.0
LAM = 40.0
SEVK = ObliqueWindow()
#: 15 analiz/ayrıklaştırma ekseninin standart sapması (ölçüldü; `ey_kural`
#: HARİÇ). Bir tolerans değil, pencerenin kendi çözünürlüğü.
SIGMA_ANALIZ = 0.002187
#: Merdiven için BAĞIL pencere gürültüsü = sigma_analiz / |delta| =
#: 0.002187 / 70.685 = 3.09e-05. ÖLÇÜLEN sayı, seçilen değil.
MERDIVEN_GURULTUSU = 3.09e-05
#: Sevk edilen merdiven. `r = 2` bu pencerede DÖRT basamağa ulaşmıyor
#: (bkz. `tir_phase_ladder` dokstringi); dördünün de `lambda/S` değeri tam
#: sayıdır (40/60/90/135), yani FFT bini `1/lambda`ya oturur.
MERDIVEN = (16.0, 24.0, 36.0, 54.0)


# ---------------------------------------------------------------------------
# Önbellek: aynı pencere+parametre ikinci kez ÇAĞRILMASIN. FDTD determinist
# olduğu için bu bir yaklaşım değil, birebir aynı sayının yeniden
# kullanılmasıdır; `ObliqueWindow` donmuş ve hashlenebilir olduğu için de
# anahtar güvenlidir.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _rho(pol, eps2=EPS2, theta=THETA, lam=LAM, window=SEVK, mirror=False):
    return reflection_coefficient(
        eps1=EPS1, eps2=eps2, theta_deg=theta, wavelength=lam,
        polarization=pol, window=window, mirror=mirror,
    )


def _delta(eps2=EPS2, theta=THETA, lam=LAM, window=SEVK, mirror=False):
    rs = _rho("s", eps2, theta, lam, window, mirror)
    rp = _rho("p", eps2, theta, lam, window, mirror)
    if rs is None or rp is None:
        return None
    return float(np.rad2deg(np.angle(rp / rs)))


# ===========================================================================
# 1. TAŞIYICI — KİLİT GERİ OKUMA YAPMIYOR
# ===========================================================================
@pytest.mark.slow
def test_TIY_FAZ_FARKI_BEYAN_EDILEN_KONUMU_KULLANMIYOR():
    """SINIF: FİZİK ÇIPASI — ve bu testin işi kilidin GERİ OKUMA
    OLMADIĞINI göstermektir.

    ÇEKİRDEK CEBİRSEL: ortam 1'de her iki kutuplanma AYNI `kx1` ile yayılır
    (aynı `n1`, aynı `theta`), dolayısıyla prob→arayüz gidiş-dönüş fazı
    `exp(-2i kx1 L)` İKİSİNDE DE AYNIDIR ve FARKTA TAM SADELEŞİR.

    ÖLÇÜLDÜ (sevk penceresi, `probe_offset` = arayüze uzaklık):

        L        arg(rho_s)[°]   arg(rho_p)[°]   delta_ps[°]
        6         -70.195356       0.489827      70.685182494
        50       -112.197620     -41.512438      70.685182756
        94       -154.199939     -83.514756      70.685183778
        300        -7.200807      63.484374      70.685181356

    Tek kutuplanmanın fazı 150°'den fazla dolaşıyor (ve 300'de sarıyor);
    `delta_ps` yayılımı **2.421e-06°** = 1.11e-03 sigma_analiz.

    Elenen `x_eff` adayında dal belirsizliği BEYAN EDİLEN `iface`'ten
    kapatılıyordu; burada `probe_offset` yalnızca MODELİ KURAR, hesaba
    girmez.
    """
    ds, args = [], []
    for L in (6, 50, 94, 300):
        w = dataclasses.replace(SEVK, probe_offset=L)
        rs = _rho("s", window=w)
        rp = _rho("p", window=w)
        assert rs is not None and rp is not None
        args.append(np.rad2deg(np.angle(rs)))
        ds.append(float(np.rad2deg(np.angle(rp / rs))))

    # Tek kutuplanma CANLI: fazı gerçekten dolaşıyor.
    assert max(args) - min(args) > 100.0
    assert args[0] == pytest.approx(-70.195356, abs=1e-4)
    assert args[2] == pytest.approx(-154.199939, abs=1e-4)

    # Fark ise ÖLÜ: dört değerin hepsi aynı.
    #
    # ÜST SINIR, EŞİTLİK DEĞİL — ve bu bilinçli: ölçülen yayılım 2.421e-06°
    # bir SIFIRIN kalıntısıdır, taşıdığı tek bilgi "küçük"tür. Değerini
    # çapalamak, tahmincinin gürültüsünü ölçüt yapmak olurdu (bu depoda üç
    # kez ısıran desen). Kapı `sigma_analiz`ten gelir: 1e-2 sigma.
    yayilim = max(ds) - min(ds)
    assert yayilim < 1e-4, f"delta_ps konuma bağımlı çıktı: yayılım {yayilim:.3e}°"
    assert yayilim / SIGMA_ANALIZ < 1e-2
    for d in ds:
        assert d == pytest.approx(70.685182, abs=1e-4)


# ===========================================================================
# 2. SINAV A — FRESNEL İÇERİĞİ
# ===========================================================================
@pytest.mark.slow
def test_KILIT_FRESNEL_ICERIGI_TASIYOR_ve_DIK_GELIS_TASIMAZDI():
    """SINAV A. SINIF: FİZİK ÇIPASI.

    Fiziği çıkar, boru hattını YERİNDE BIRAK: `n2`'yi tara ve kilidin
    okuduğu sayının gerçekten `n2` ile değiştiğini göster.

    ÖLÇÜLDÜ (sevk penceresi):

        n2      delta_ölç[°]   delta_kapalı form[°]   fark[°]
        1.00     70.685182          70.528779         +0.156403
        1.15     60.403887          60.400722         +0.003165
        1.35     33.078530          33.178576         -0.100046

    Ölçülen yayılım **37.606652°**, kapalı formun yayılımı **37.350204°**
    (oran 1.006866). `sigma_analiz` = 0.002187° ile yayılım **17193 sigma**.

    KARŞITI: aynı `n2` aralığında DİK GELİŞ Fresnel yansıması yalnızca
    3.0 kat değişir ve `arg(r)` dik gelişte SABİTTİR (r reeldir) — yani
    kilidin okuduğu şey dik gelişte OKUNAMAYAN bir sayıdır.

    0.25° KAPISI nereden geliyor: `lambda` = 40'ta ölçülen ızgara sistematiği
    0.156403°'dir ve merdiven onun ~1/lambda^2 ile sıfıra gittiğini gösterir
    (bkz. `test_HUKUM_SIGMA_DEGIL_YAKINSAMA`). En yakın mutant `dt_carpan`
    aynı ölçüte 2.217231° verir. Kapı ikisinin arasında ve ikisine de pay
    bırakıyor (aşağıda 1.6 kat, yukarıda 8.9 kat) — ve HÜKÜM kapıda değil
    MERDİVENDEDİR; bu test İÇERİĞİ ölçer.
    """
    n2ler = (1.00, 1.15, 1.35)
    olc, teo = [], []
    for n2 in n2ler:
        d = _delta(eps2=n2**2)
        assert d is not None
        olc.append(d)
        teo.append(fresnel_phase_difference(EPS1, n2**2, THETA))
    olc, teo = np.array(olc), np.array(teo)

    # (a) İÇERİK: sayı gerçekten n2 ile değişiyor, ve sigma'nın çok üstünde.
    yayilim = float(olc.max() - olc.min())
    assert yayilim == pytest.approx(37.606652, abs=1e-4)
    assert yayilim / SIGMA_ANALIZ > 1000.0

    # (b) İçerik KAPALI FORMUN içeriği: iki yayılım %1'de uyuşuyor.
    teo_yay = float(teo.max() - teo.min())
    assert teo_yay == pytest.approx(37.350204, abs=1e-5)
    assert yayilim / teo_yay == pytest.approx(1.006866, abs=1e-4)

    # (c) Nokta nokta uyum; kapı ölçülen (0.156) ile en yakın mutantın
    #     (2.217) arasında.
    assert float(np.max(np.abs(olc - teo))) == pytest.approx(0.156403, abs=1e-4)
    assert np.all(np.abs(olc - teo) < 0.25)

    # (d) Aynı aralıkta dik geliş Fresnel'i çok daha az bilgi taşıyor.
    n1 = np.sqrt(EPS1)
    Rd = ((n1 - np.array(n2ler)) / (n1 + np.array(n2ler))) ** 2
    assert Rd.max() / Rd.min() == pytest.approx(3.0, abs=0.05)


# ===========================================================================
# 3. SINAV B — AYNA KONTROLÜ, `x_eff`in TAM TERSİ
# ===========================================================================
@pytest.mark.slow
def test_AYNA_KONTROLU_MALZEME_BILGISI_TASIMIYOR_ve_BU_x_eff_in_TERSI():
    """SINAV B. SINIF: KONTROL (fizik çıpası DEĞİL).

    Malzeme adımını kaldır, boru hattını yerinde bırak: kilidin okuduğu sayı
    da kaybolmalıdır. Sevk edilen ayna, ızgarayı arayüzde bitirir; sağ uç
    `_adim`in kendi PEC duvarıdır ve `eps2` KODA HİÇ GİRMEZ.

    ÖLÇÜLDÜ: `delta_ayna` = **0.000000000000°**, `eps2` argümanı 1.0 ve 9.0
    için AYNI (fark 0.000e+00). Dahası `rho_s` ile `rho_p` BİREBİR eşit.

    SEBEP CEBİRSEL, ve planın beklediğinden SERTTİR: aynada iki duvar TAM
    OLARAK AYNI hücrededir (s'te `e` = Ez → PEC; p'de `e` = Hz → PMC) ve
    ortam düzgündür. O durumda iki kutuplanmanın güncellemesi sabit bir
    ölçek çarpanı dışında AYNI özyinelemedir; `rho` bir ORAN olduğu için
    çarpan sadeleşir. Yani ayna kipinde kilit `dt_carpan`ı 116 sigma değil
    **0.0 sigma** ile ayırıyor (ölçüldü; `kaydir` ve `sinir_pec` de 0.0).
    Dielektrikte aynı mutant 249.8 sigma verir.

    `x_eff` adayı PEC'te 12 kat daha KESKİNDİ — yani Fresnel'i hiç
    sınamıyordu. Bu tablo ikisi arasındaki farkın en temiz ölçüsüdür.

    BORU HATTI AYNADA DA CANLI (aksi hâlde 0.0 bir arıza olurdu):
    `|rho| = 1`'e 4.8e-09'da oturuyor ve arg(rho_s) `probe_offset` ile
    26.999696° → 89.994242° dolaşıyor.
    """
    a = _delta(eps2=1.0, mirror=True)
    b = _delta(eps2=9.0, mirror=True)
    assert a is not None and b is not None
    assert a == 0.0 and b == 0.0
    assert abs(a - b) == 0.0

    # Sebep: iki kutuplanma aynada BİREBİR aynı sayıyı veriyor.
    rs = _rho("s", eps2=1.0, mirror=True)
    rp = _rho("p", eps2=1.0, mirror=True)
    assert rs == rp

    # Ama boru hattı ÖLÜ DEĞİL: |rho| = 1 (şema çıpası) ve tek kutuplanmanın
    # fazı konumla dolaşıyor.
    assert abs(abs(rs) - 1.0) < 1e-8
    assert abs(rs) == pytest.approx(0.999999995209, abs=1e-9)
    w300 = dataclasses.replace(SEVK, probe_offset=300)
    uzak = _rho("s", eps2=1.0, window=w300, mirror=True)
    assert np.rad2deg(np.angle(rs)) == pytest.approx(26.999696, abs=1e-4)
    assert np.rad2deg(np.angle(uzak)) == pytest.approx(89.994242, abs=1e-4)
    # ve orada da fark ÖLÜ:
    assert _delta(eps2=1.0, window=w300, mirror=True) == 0.0

    # Dielektrikte AYNI kilit malzeme taşıyor -- karşıtlığın sayısı.
    d1 = _delta(eps2=1.00)
    d2 = _delta(eps2=1.35**2)
    assert abs(d1 - d2) / SIGMA_ANALIZ > 1000.0


# ===========================================================================
# 4. HÜKÜM SIGMA DEĞİL, YAKINSAMA
# ===========================================================================
@pytest.mark.slow
def test_HUKUM_SIGMA_DEGIL_YAKINSAMA_ve_TEK_IZGARADA_MUTANT_KAZANIR():
    """EN ÖNEMLİ TEST. SINIF: FİZİK ÇIPASI + ÖLÇÜT.

    Doğru çözücünün KENDİ sistematiği `lambda` = 40'ta +0.156403°'dir, yani
    **71 sigma_analiz**. Sigma tek başına bir ölçüt OLAMAZ.

    Ve tek ızgarada karar VERİLEMEZ — ölçüldü: `lambda` = 16'da `dt_carpan`
    +0.432869° sapma verir, doğru çözücü +0.931942°. **Tek nokta bakılsaydı
    MUTANT KAZANIRDI.** Bu, Task 2'nin merdiven zorunluluğunun bağımsız bir
    gerekçesidir.

    SEVK EDİLEN MERDİVEN (r = 1.5, lambda = 16/24/36/54, `noise` = 3.09e-05
    ölçülen bağıl pencere gürültüsü) ve `delta - delta_Fresnel`:

        çözücü        lam=16      lam=24      lam=36      lam=54    hüküm
        doğru        +0.931942   +0.427803   +0.192706   +0.086305  OTURDU(2)
        eps_orta    -10.138681   -5.928302   -3.577321   -2.217855  OTURDU(1)
        dt_carpan    +0.432869   -0.103535   -0.352029   -0.464442  OTURDU(2)
        sinir_pec    +0.931942   +0.427803   +0.192706   +0.055924  YARGILANAMAZ
        ey=harmonik +14.496587   +9.228048   +5.974611   +3.909029  OTURDU(1)
        kaydir       +0.932006   +0.427799   +0.192705   +0.086413  OTURDU(2)
        kaynak_çift  (doğru ile BİT BİT aynı)                       OTURDU(2)

    İKİ ÖLÇÜT, VE İKİSİ DE TOLERANSSIZ:

    1. `verdict == OTURDU` — merdiven bir mertebeye oturuyor mu.
       `sinir_pec` tam buradan düşüyor (p_inf 2.103 → 0.244).
    2. **TEORİ SON ARTIŞIN İÇİNDE**: `|v[-1] - delta_Fresnel|` son basamak
       artışı `|v[-1] - v[-2]|`den küçük olmalı. Ölçek merdivenin KENDİSİNDEN
       gelir, elle seçilmez. Ölçüldü: doğru 0.086305 <= 0.106401 (GEÇER);
       `eps_orta`, `dt_carpan` ve `ey=harmonik` DÜŞER.

    İkisi birlikte `eps_orta`, `dt_carpan`, `sinir_pec` ve `ey_kural`
    değişimini yakalar; `kaydir` ve `kaynak_cift`e BEYAN EDİLDİĞİ GİBİ
    kördür.
    """
    L = tir_phase_ladder(
        eps1=EPS1, eps2=EPS2, theta_deg=THETA, wavelengths=MERDIVEN,
        noise=MERDIVEN_GURULTUSU,
    )
    assert L is not None
    teo = fresnel_phase_difference(EPS1, EPS2, THETA)

    beklenen = (71.460721, 70.956582, 70.721486, 70.615084)
    for olc, bek in zip(L.values, beklenen):
        assert olc == pytest.approx(bek, abs=1e-4)

    # (1) Merdiven bir mertebeye OTURUYOR, ve o mertebe 2.
    assert L.verdict == "OTURDU"
    assert L.settled_order == 2
    assert L.p_inf == pytest.approx(2.1028, abs=1e-3)
    assert L.uncertainty == pytest.approx(0.33642, abs=1e-4)
    # Ve PAY İNCEDİR -- dürüst olumsuz, dokstringde yazılı:
    assert abs(L.p_inf - 2) + L.uncertainty == pytest.approx(0.4392, abs=1e-3)

    # (2) TEORİ SON ARTIŞIN İÇİNDE -- ölçek merdivenin kendisinden.
    son_artis = abs(L.values[-1] - L.values[-2])
    assert son_artis == pytest.approx(0.106401, abs=1e-4)
    assert abs(L.values[-1] - teo) == pytest.approx(0.086305, abs=1e-4)
    assert abs(L.values[-1] - teo) <= son_artis

    # (3) Artışlar küçülüyor -- yakınsamanın kendisi.
    artislar = np.abs(np.diff(L.values))
    assert np.all(np.diff(artislar) < 0.0)

    # (4) VE TEK IZGARADA KARAR VERİLEMEZ: en kaba basamakta doğru çözücünün
    #     sapması, `dt_carpan`ınkinden BÜYÜK (0.931942 > 0.432869).
    assert L.values[0] - teo == pytest.approx(0.931942, abs=1e-4)
    assert abs(L.values[0] - teo) > 0.432869


def test_MERDIVEN_KAPIYA_TAKILAN_BASAMAKTA_NONE_DONER():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    Eksik bir merdivenden hüküm çıkarılmaz. `lambda` = 10 `|rho| = 1`
    kapısına takılıyor (ölçüldü) ve merdiven bütünüyle `None` dönmeli --
    üç basamakla devam edip "derinlik yetmedi" demek, kapının düştüğünü
    GİZLERDİ.
    """
    L = tir_phase_ladder(
        eps1=EPS1, eps2=EPS2, theta_deg=THETA, wavelengths=(10.0, 20.0),
        noise=MERDIVEN_GURULTUSU,
    )
    assert L is None


def test_MERDIVEN_GIRDI_DENETIMI():
    """SINIF: TRANSKRİPSİYON DENETİMİ. Azalan `lambda` = kalınlaşan ızgara;
    `convergence_ladder` bunu `p`nin işaretini çevirerek SESSİZCE geçerdi."""
    with pytest.raises(InvalidInputError, match="ARTAN"):
        tir_phase_ladder(eps1=EPS1, eps2=EPS2, theta_deg=THETA,
                         wavelengths=(54.0, 36.0, 24.0, 16.0), noise=1e-5)
    with pytest.raises(InvalidInputError, match="en az iki"):
        tir_phase_ladder(eps1=EPS1, eps2=EPS2, theta_deg=THETA,
                         wavelengths=(40.0,), noise=1e-5)


# ===========================================================================
# 5. ÜÇ KÖRLÜK — kusur değil, aynı özelliğin bedeli
# ===========================================================================
@pytest.mark.slow
def test_KORLUK_1_ARAYUZ_KAYDIRMASI_ve_BU_KANITLANABILIR():
    """SINIF: KÖRLÜĞÜN KİLİDİ. Bu test kilidin GÖRMEDİKLERİNİ kilitler.

    Arayüzü bir hücre kaydırmak, ortam 1 içinde bir ÖTELEMEDİR (kaynak ile
    arayüz arası homojen) ve her iki kutuplanma AYNI çarpanı alır. GERİ OKUMA
    YOKLUĞU İLE ÖTELEMEYE DUYARLILIK AYNI ŞEYİN İKİ YÜZÜDÜR; ikisi birden
    istenemez, ve bu KANITLANABİLİR — testin 1. maddesi zaten aynı özelliği
    ölçüyor.

    ÖLÇÜLDÜ: `iface` 1500 → 1499 (prob 1494'te YERİNDE):

        arg(rho_s)   -70.195356°  →  -44.695300°   (25.500° oynadı)
        delta_ps      70.685182494 → 70.685181162  (1.332e-06° oynadı)

    Öteleme fazının kendisi `2 kx1 dx` = 25.46°'dir ve ölçülen 25.500° ona
    oturuyor — yani kaydırma GERÇEKTEN oldu, kilit onu GÖRMÜYOR.

    BEDELİN KARŞILIĞI: Fresnel `|rho|` ÇEVİRİ DENETİMİ olarak kalıyor.
    DÜRÜST OLUMSUZ: o da zayıf görüyor — dik gelişte aynı kaydırma `|R|`yi
    bağıl 2.4e-06 oynatıyor (spec §2.3 "7 hanede birebir aynı" diyordu;
    ölçülen 6 hane). GÜÇLÜ gören tek şey `arg(rho)`dur (36.14°) ve o bir
    GERİ OKUMADIR.
    """
    kaydirilmis = dataclasses.replace(SEVK, interface=1499, probe_offset=5)
    assert kaydirilmis.probe == SEVK.probe  # prob YERİNDE

    rs0 = _rho("s")
    rs1 = _rho("s", window=kaydirilmis)
    d0 = _delta()
    d1 = _delta(window=kaydirilmis)

    # Kaydırma GERÇEKTEN oldu -- tek kutuplanma bunu 2*kx1*dx kadar görüyor.
    faz = float(np.rad2deg(np.angle(rs1 / rs0)))
    beklenen = np.rad2deg(2.0 * (2 * np.pi / LAM) * np.sqrt(EPS1)
                          * np.cos(np.deg2rad(THETA)))
    assert faz == pytest.approx(25.500, abs=1e-2)
    assert faz == pytest.approx(beklenen, rel=2e-3)

    # Kilit ise görmüyor. ÜST SINIR, eşitlik DEĞİL: ölçülen 1.332e-06° bir
    # SIFIRIN kalıntısıdır. (Bu satır ilk hâlde `== 1.332e-06 ± %35` idi ve
    # MUTASYON TURU onu düşürdü: `kaydir` mutantı altında aynı büyüklük
    # 1.013e-07 çıkıyor — yani test, KÖR OLDUĞUNU İDDİA ETTİĞİ bozmayı bir
    # gürültü değerini çapaladığı için yakalıyordu. Bir NULL'ın değeri
    # çapalanamaz; çapalanabilecek olan üst sınırıdır.)
    assert abs(d1 - d0) < 1e-4
    assert abs(d1 - d0) / SIGMA_ANALIZ < 1e-2


@pytest.mark.slow
def test_KORLUK_2_SOGURUCU_TEK_IZGARADA_KOR_MERDIVENDE_DEGIL():
    """SINIF: KÖRLÜĞÜN KİLİDİ + ÇÖZÜNÜRLÜĞÜN kilidi.

    Soğurucuyu kaldırmak (`npml` = 0 → ızgaranın iki ucundaki PEC duvar
    çıplak kalır) sınır yansıması üretir. ÖLÇÜLDÜ:

        lambda = 40 : 70.685182494 → 70.685110980   fark 7.15e-05° (0.033 s)
        lambda = 54 : 70.615084    → 70.584703      fark 3.04e-02° (13.9 s)

    Yani körlük İZGARAYA BAĞLI ve merdivenin işi tam olarak budur: aynı
    bozma `lambda` = 40'ta 0.03 sigma, `lambda` = 54'te 13.9 sigma.
    Merdivenin kendi hükmü de düşüyor (ölçüldü, aynı dört basamakta:
    p_inf 2.1028 → 0.2440, OTURDU(2) → YARGILANAMAZ).

    `|rho| = 1` şema çıpası aynı bozmayı 4.53e-09 → 3.17e-06 (700 kat) ile
    görüyor — ama BU BİR SIGMA DEĞİLDİR: analiz eksenlerindeki ||rho|-1|
    yayılımının standart sapması 2.8e-05'tir, yani 3.17e-06 o ölçekte
    **0.11 sigma**'dır. Planın "4.2 sigma" sayısı bu pencerede YENİDEN
    ÜRETİLEMEDİ ve yazılmıyor.

    KUYRUK KAPISI (10.10.2026) körlüğü YARIYA İNDİRİYOR: `lambda` = 54'te
    soğurucusuz kayıt TAM DEĞİL — PEC yankısı kayda giriyor, kaydın son
    periyodunda genlik tepenin 1.75e-01'i — ve alet artık `None` döndürüyor;
    yani bozma orada TEK IZGARADA görünür. 13.9 sigma sayısı ve merdivenin
    YARGILANAMAZ hükmü kuyruk kapısı KAPALIYKEN (`tail_gate` 1) üretilir
    (RC-a'dan beri yankı kapısı da kapalı, `echo_gate` 1: PEC yankısı
    gelen kaydında, spektral ikinci varış 8.9e-03).
    `lambda` = 40'ta soğurucusuz kuyruk 1.97e-04 (eşiğin altında) ve körlük
    orada SÜRÜYOR.
    """
    cipsiz = dataclasses.replace(SEVK, npml=0)

    d40 = _delta()
    d40p = _delta(window=cipsiz)
    assert abs(d40p - d40) == pytest.approx(7.15e-05, rel=0.2)
    assert abs(d40p - d40) / SIGMA_ANALIZ < 0.1  # tek ızgarada KÖR

    d54 = _delta(lam=54.0)
    assert _delta(lam=54.0, window=cipsiz) is None  # kuyruk kapısı görüyor
    d54p = _delta(lam=54.0, window=dataclasses.replace(cipsiz, tail_gate=1.0, echo_gate=1.0))
    assert abs(d54p - d54) == pytest.approx(3.04e-02, rel=0.2)
    assert abs(d54p - d54) / SIGMA_ANALIZ > 10.0  # daha ince ızgarada GÖRÜNÜR

    # Ve şema çıpası da görüyor -- ama 700 kat, "sigma" olarak değil.
    assert abs(abs(_rho("s")) - 1.0) < 1e-8
    assert abs(abs(_rho("s", window=cipsiz)) - 1.0) > 1e-6


def test_KORLUK_3_KAYNAK_GENLIGI_cunku_RHO_BIR_ORANDIR():
    """SINIF: KÖRLÜĞÜN KİLİDİ. Ve mekanizması ÇÖZÜCÜNÜN DOĞRUSALLIĞIDIR.

    Hiçbir yansıma-ORANI kilidi kaynağın genliğindeki bir hatayı göremez:
    `rho = FFT(yansıyan)/FFT(gelen)` ve her iki taraf aynı çarpanı alır.

    Burada mekanizmanın kendisi kilitleniyor: kaynak 2 katına çıkarılınca
    kayıt BİT BİT 2 katı (2 bir ikinin kuvveti olduğu için kayan noktada
    TAM). Sevk penceresinde tam boru hattıyla da ölçüldü: `rho_s` BİREBİR
    aynı, |fark| = 0.000e+00.

    Bu test UCUZ tutuldu (N = 400, 400 adım) çünkü iddia cebirseldir; pahalı
    olan ölçüm yukarıda yazılı.
    """
    ortam = Medium.half_space(400, interface=250, eps1=1.0, eps2=4.0)
    ortak = dict(
        courant=0.5, boundary=PML(thickness=15), steps=400, source_index=100,
        probes=(200,), wavelength=20.0, sgf=1.5, ky=0.1, polarization="p",
    )
    tek = run_oblique_1d(
        ortam, source=lambda t: gauss_cos(t, wavelength=20.0, sgf=1.5, t0=60.0),
        **ortak)
    cift = run_oblique_1d(
        ortam,
        source=lambda t: 2.0 * gauss_cos(t, wavelength=20.0, sgf=1.5, t0=60.0),
        **ortak)
    assert np.array_equal(cift.probes, 2.0 * tek.probes)


# ===========================================================================
# 6. ÖLÇÜM ALETİNİN TAVANI
# ===========================================================================
@pytest.mark.slow
def test_OLCUM_ALETININ_TAVANI_VAR_ve_KAPI_ONU_KESIYOR():
    """SINIF: ŞEMA ÇIPASI (`|rho| = 1`) + KAPI.

    TİY'de `|r| = 1` FİZİKSEL OLARAK ZORUNLUDUR. Büyük açılarda 1B-açılı
    fark yöntemi bunu sağlayamaz ve `R > 1` gibi İMKANSIZ sayılar döndürür.
    Kapı onları keser ve `None` döner — sessizce imkansız bir sayı basmaz
    (Modül 4'ün `correlation_length → None` dersinin aynısı).

    ÖLÇÜLDÜ (sevk penceresi, kapı KAPALIYKEN `|rho_s|`):

        theta   45      55      65      70       72.5     75      80      85
        sgf=9  1.0000  1.0000  0.9986  1.0206   1.0291  0.9947  0.9420  0.9406
        sgf=3  0.9964  1.0425  0.9475  0.9159   0.9087  0.9126  0.9213  0.9344

    ÜÇ ŞEY BİRDEN OKUNUYOR:

    1. `theta` = 70 ve 72.5'te `R = |rho|^2` **1'in ÜSTÜNDE** (1.041710 ve
       1.059033) — fiziksel olarak imkansız. Kapı ikisini de kesiyor.
    2. Arıza bir DÜĞMEDİR, bir açı eşiği değil: `R > 1` bölgesi `sgf` = 9'da
       70-72.5, `sgf` = 3'te 50-55'e kayıyor. Mekanizma
       `source.gauss_cos`ta: `ky` sabittir, `f` bininin etkin açısı
       `asin(ky/(2 pi f n1))`dir, geniş bant komşu binlerin farklı
       açılarının `r`sini karıştırır.
    3. **DÜRÜST OLUMSUZ:** bozulma `theta` ile MONOTON DEĞİL — `theta` = 75
       iki kesilen bölgenin arasından GEÇİYOR (`||rho|-1|` = 5.3e-03).
       Planın *"theta >= 75'te R > 1 (1.006/1.306/1.933)"* cümlesi bu
       pencerede YENİDEN ÜRETİLEMEDİ; sevk edilen şey bir açı tavanı değil,
       nokta nokta bir imkan kapısıdır. (Aynı cümle Task 1'in
       `source.gauss_cos` ve `boundary.PML` dokstringlerinde de vardı ve
       Task 3'te düzeltildi.)

    KUYRUK KAPISI (10.10.2026): yukarıdaki tablo KAPILAR KAPALIYKEN
    (`unity_gate` 1e9, `tail_gate` 1, `echo_gate` 1) okunur. Kuyruk kapısı açıkken 65..85
    hepsi `None` (kaydın son periyodunda genlik tepenin 2.35e-02 .. 0.99'u;
    ayrık-tam referansa hata 1.6e-03 .. 7.4e-02); theta = 75'in `|rho| = 1`
    kapısından geçmesi bir rastlantıydı (hata 4.7e-02). Mekanizma kesik
    kayıttır: bkz. `reflection_coefficient` ve `test_kayit_tamligi.py`.
    """
    # theta = 80: kapı kesiyor.
    assert _rho("s", theta=80.0) is None
    assert tir_phase_difference(
        eps1=EPS1, eps2=EPS2, theta_deg=80.0, wavelength=LAM) is None

    acik = dataclasses.replace(SEVK, unity_gate=1e9, tail_gate=1.0, echo_gate=1.0)
    r80 = _rho("s", theta=80.0, window=acik)
    assert abs(r80) == pytest.approx(0.941959, abs=1e-4)
    assert abs(abs(r80) - 1.0) > SEVK.unity_gate

    # theta = 70: İMKANSIZ tarafta -- R > 1.
    r70 = _rho("s", theta=70.0, window=acik)
    assert abs(r70) ** 2 == pytest.approx(1.041710, abs=1e-4)
    assert abs(r70) ** 2 > 1.0
    assert _rho("s", theta=70.0) is None

    # Sevk açısında kapı GEÇİYOR ve şema çıpası 1e-8'de oturuyor.
    assert abs(abs(_rho("s")) - 1.0) == pytest.approx(4.5278e-09, rel=0.2)


def test_KAPI_YALNIZ_TIY_ve_AYNADA_KOSAR():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    `|r| = 1` yalnız tam iç yansımada ve mükemmel aynada ZORUNLUDUR. Kısmi
    yansımada |r| < 1 fiziğin kendisidir; kapı orada koşsaydı DOĞRU
    ölçümleri reddederdi.
    """
    assert total_internal_reflection(4.0, 1.0, 45.0)      # theta_c = 30
    assert not total_internal_reflection(4.0, 1.0, 25.0)
    assert not total_internal_reflection(1.0, 4.0, 45.0)  # yoğun ortama geçiş
    assert not total_internal_reflection(4.0, 4.0, 45.0)


@pytest.mark.slow
def test_TAVANIN_SEBEBI_sgf_ve_TASK1_DOKSTRINGI_HALA_DOGRU():
    """SINIF: ÇAPRAZ DENETİM — Task 1'in SEVK EDİLMİŞ bir sayısı, ilk kez
    sevk edilen egik boruyla sınanıyor.

    `source.gauss_cos` dokstringi şunu iddia ediyor: `theta` = 55, `lambda` =
    40, N = 3600, npml = 60, iface = 1500, isrc = 600, probe = 1494,
    t0 = 5 sg, DİKDÖRTGEN penceresinde yalnız `sgf` 3 → 6 ile `||rho_s|-1|`
    4.2533e-02 → 2.7313e-05 (1557 KAT); `npml` 60 → 120/200 ise 1.0000 kat.

    O ölçüm sevk edilmemiş bir prototiple yapılmıştı. SEVK EDİLEN boruyla
    yeniden ölçüldü ve **birebir tuttu**: 4.2533e-02 / 2.7313e-05 /
    7.4694e-08 (sgf = 9), ve npml 200'de yine 4.2533e-02.

    Kapılar kapalı (`unity_gate` 1e9, `tail_gate` 1, `echo_gate` 1): sgf = 3 ve 6'da
    kayıt TAM DEĞİL (kuyruk 4.75e-01 ve 2.96e-03, 10.10.2026) ve kuyruk
    kapısı açıkken ikisi de `None` dönerdi.
    """
    acik = dataclasses.replace(SEVK, unity_gate=1e9, tail_gate=1.0, echo_gate=1.0)
    w3 = dataclasses.replace(acik, sgf=3.0)
    w6 = dataclasses.replace(acik, sgf=6.0)
    w3n = dataclasses.replace(acik, sgf=3.0, npml=200)

    e3 = abs(abs(_rho("s", theta=55.0, window=w3)) - 1.0)
    e6 = abs(abs(_rho("s", theta=55.0, window=w6)) - 1.0)
    e3n = abs(abs(_rho("s", theta=55.0, window=w3n)) - 1.0)

    assert e3 == pytest.approx(4.2533e-02, rel=1e-3)
    assert e6 == pytest.approx(2.7313e-05, rel=1e-3)
    assert e3 / e6 == pytest.approx(1557.0, rel=0.01)
    # npml HİÇBİR ŞEY yapmıyor -- arıza sınırın değil, bandın.
    assert e3n / e3 == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# 7. FRESNEL'İN ÇEVİRİ DENETİMİ (spec §2.3) — FİZİK ÇIPASI DEĞİL
# ===========================================================================
@pytest.mark.slow
def test_FRESNEL_R_CEVIRI_DENETIMI_ve_HATASI_2_MERTEBEDEN_KAPANIYOR():
    """SINIF: ÇEVİRİ DENETİMİ. **FİZİK ÇIPASI DEĞİL** — `n` girdidir, `R`
    onun cebirsel sonucudur (spec §2.1/§2.3). Bu sayı çözücünün arayüzü
    çözdüğünü gösterir, yeni bir fizik bilgisi üretmez.

    ÖLÇÜLDÜ (`eps1` = 4, `n2` = 1.5, `theta` = 0, sevk geometrisi):

        lambda = 20 : R = 0.023802680   R/R_Fresnel = 1.166331
        lambda = 40 : R = 0.021185092   R/R_Fresnel = 1.038070

    Bağıl hata 0.166331 → 0.038070, oran **4.3691** — 2. mertebeye yakın
    (tam 2. mertebe 4.0 verirdi). Kalan sapma arayüzün basamak
    ayrıklaştırmasının 1. mertebeli katkısıdır ve bir mertebe SAYISI
    yazılmıyor.
    """
    RF = abs(fresnel_coefficients(EPS1, 2.25, 0.0)[0]) ** 2
    h = []
    for lam in (20.0, 40.0):
        r = _rho("s", eps2=2.25, theta=0.0, lam=lam)
        assert r is not None
        h.append(abs(r) ** 2 / RF - 1.0)
    assert h[0] == pytest.approx(0.166331, abs=1e-4)
    assert h[1] == pytest.approx(0.038070, abs=1e-4)
    assert h[0] / h[1] == pytest.approx(4.3691, abs=0.05)


def test_KISMI_YANSIMADA_KAPI_KOSMAZ_ve_R_1_DEN_KUCUK():
    """SINIF: TRANSKRİPSİYON DENETİMİ -- `n2 > n1`de |r| < 1 ve kapı susar."""
    r = reflection_coefficient(
        eps1=1.0, eps2=2.25, theta_deg=0.0, wavelength=40.0, polarization="s",
        window=dataclasses.replace(SEVK, cells=1200, interface=600,
                                   source_index=200, sgf=2.0),
    )
    assert r is not None
    assert abs(r) < 1.0


# ===========================================================================
# 8. ŞEMA / TRANSKRİPSİYON — EGİK ÇÖZÜCÜ İKİNCİ BİR ÇÖZÜCÜ DEĞİL
# ===========================================================================
def test_KY_SIFIRDA_EGIK_COZUCU_run_1d_ILE_BIT_BIT_AYNI():
    """SINIF: TRANSKRİPSİYON DENETİMİ — ve Modül 5'in TAŞIYICI mimari kilidi.

    `oblique.py` ikinci bir çözücü DEĞİLDİR: her iki kutuplanma da
    `yee._adim`in KENDİSİNDEN geçer, üstüne yalnız `i*ky` terimi eklenir.
    `ky = 0`da s-kutuplanma bu yüzden `run_1d`nin kaydını BİT BİT
    üretmelidir — ve bu yalnız güncellemeyi değil, kaynak zamanlamasını ve
    KAYIT HİZASINI de sınar (kayıt sonra yapılsaydı bütün seri bir adım
    kayardı).

    ÖLÇÜLDÜ: `max|fark|` = 0.000e+00, `array_equal` True.
    """
    ortam = Medium.half_space(400, interface=250, eps1=1.0, eps2=4.0)
    ortak = dict(
        courant=0.5, boundary=PML(thickness=15), steps=600,
        source=lambda t: gauss_cos(t, wavelength=20.0, sgf=1.5, t0=60.0),
        source_index=100, probes=(200, 300), wavelength=20.0, sgf=1.5,
    )
    duz = run_1d(ortam, **ortak)
    egik = run_oblique_1d(ortam, ky=0.0, polarization="s", **ortak)
    assert np.array_equal(duz.probes.astype(complex), egik.probes)

    # Ama ky ve kutuplanma GERÇEKTEN kullanılıyor (çözücü onları yutmuyor).
    assert not np.allclose(
        egik.probes, run_oblique_1d(ortam, ky=0.2, polarization="s", **ortak).probes)
    assert not np.allclose(
        egik.probes, run_oblique_1d(ortam, ky=0.0, polarization="p", **ortak).probes)


def test_EGIK_CEKIRDEK_ORTAMDAN_BES_AD_GORUYOR_ve_GEOMETRI_ONLARDA_YOK():
    """SINIF: TRANSKRİPSİYON DENETİMİ — Task 1'in sayımının egik uzantısı.

    Task 1'de 1B TM çekirdeği ortamdan ÜÇ ad görüyordu: `{cells, eps_r,
    mu_yari}`. Egik indirgeme İKİ ad daha ekliyor -- `mu_r` (tam hücredeki
    H'nin katsayısı; 01.10.2026'dan beri 1B kümesinde de var, etkin Courant
    kapısı yüzünden) ve **`eps_yari`** (yarım hücredeki TEĞET E, yani
    `ey_kural`ın ilk gerçek çağrısı; Task 1 raporunda "1B TM'de ölü kod"
    diye işaretlenmişti ve doğruydu).

    GEOMETRİ hâlâ kümede DEĞİL: `interface`, `half_space`, `eps_min` yok.
    Bu, "aynı çözücü, bağımsız hükümler" iddiasının ön koşuludur.
    """
    class Kaydeden:
        def __init__(self, hedef):
            self._hedef = hedef
            self.dokunulan: set[str] = set()

        def __getattr__(self, ad):
            self.dokunulan.add(ad)
            return getattr(self._hedef, ad)

    for pol in POLARIZATIONS:
        ortam = Kaydeden(Medium.half_space(240, interface=120, eps1=1.0, eps2=4.0))
        sinir = Kaydeden(PML(thickness=15))
        run_oblique_1d(
            ortam, courant=0.4, boundary=sinir, steps=40,
            source=lambda t: gauss_cos(t, wavelength=20.0, sgf=1.5, t0=60.0),
            source_index=80, probes=(120,), wavelength=20.0, sgf=1.5,
            ky=0.1, polarization=pol,
        )
        assert ortam.dokunulan == {"cells", "eps_r", "eps_yari", "mu_r", "mu_yari"}
        assert sinir.dokunulan == {"sigma_1d", "thickness"}
        for yasak in ("ey_kural", "eps_min", "half_space", "interface"):
            assert yasak not in ortam.dokunulan


def test_UZAMSAL_ISARET_UZLASIMI_ARAYUZSUZ_OLCULDU():
    """SINIF: ŞEMA ÇIPASI. `delta_ps`in İŞARETİ bir tercih değil ÖLÇÜMDÜR.

    Ölçüm ARAYÜZSÜZ yapılır — `n2` ve arayüz hesaba HİÇ girmez; yalnız ortam
    1'deki iki prob arasındaki faz ilerlemesine bakılır. Ölçüldü
    (`eps1` = 4, `theta` = 45, `lambda` = 40, prob x = 1000 ve 1200):
    `arg(F2/F1)` = **-0.523604090 rad**.

    Karşılaştırma İNDİRGENMİŞ ŞEMANIN SAYISAL `kx`iyle yapılır:

        eps (2/S sin(wS/2))^2 = (2 sin(kx/2))^2 + ky^2   ->   kx = 0.222529927

    (analitik 0.222144147; oran 1.0017366). Beklenenler ve ayrım:

        e^{-i kx x} : -0.523688309   |fark| = 8.4e-05    <- BU
        e^{+i kx x} : +0.523688309   |fark| = 1.0e+00

    yani 12400 kat. ANALİTİK `kx` kullanılsaydı ayrım yine doğru çıkardı ama
    kalıntı 7.7e-02 olurdu — 900 kat kaba. Fark ŞEMANIN KENDİ dispersiyonudur.

    SONUÇ: ölçülen `rho`, klasik Fresnel `r`sinin EŞLENİĞİDİR; bu yüzden
    `fresnel_phase_difference` bir eksi işareti taşır.
    """
    w = 2 * np.pi / LAM
    ky = w * np.sqrt(EPS1) * np.sin(np.deg2rad(THETA))
    lhs = EPS1 * (2.0 / SEVK.courant * np.sin(w * SEVK.courant / 2.0)) ** 2
    kx_sayisal = 2.0 * np.arcsin(0.5 * np.sqrt(lhs - ky**2))
    kx_analitik = w * np.sqrt(EPS1) * np.cos(np.deg2rad(THETA))
    assert kx_sayisal == pytest.approx(0.222529927296, abs=1e-9)
    assert kx_sayisal / kx_analitik == pytest.approx(1.0017366, abs=1e-6)

    steps = _record_steps(theta_deg=THETA, wavelength=LAM, eps1=EPS1, window=SEVK)
    ortam = Medium(np.full(SEVK.cells, EPS1), np.ones(SEVK.cells))
    sg = SEVK.sgf * LAM
    kayit = run_oblique_1d(
        ortam, courant=SEVK.courant, boundary=PML(thickness=SEVK.npml),
        steps=steps,
        source=lambda t: gauss_cos(t, wavelength=LAM, sgf=SEVK.sgf, t0=5 * sg),
        source_index=SEVK.source_index, probes=(1000, 1200), wavelength=LAM,
        sgf=SEVK.sgf, ky=ky, polarization="s",
    )
    F1, _ = _spektrum(kayit.probes[0], SEVK.courant, 1.0 / LAM)
    F2, _ = _spektrum(kayit.probes[1], SEVK.courant, 1.0 / LAM)
    olculen = float(np.angle(F2 / F1))
    assert olculen == pytest.approx(-0.523604090, abs=1e-6)

    def sar(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    eksi = abs(olculen - sar(-kx_sayisal * 200))
    arti = abs(olculen - sar(+kx_sayisal * 200))
    assert eksi < 1e-3
    assert arti / eksi > 1000.0

    # Ve sevk edilen kapalı form BU uzlaşımda yazıldı: eksi işaretli.
    rs, rp = fresnel_coefficients(EPS1, EPS2, THETA)
    assert fresnel_phase_difference(EPS1, EPS2, THETA) == pytest.approx(
        float(-np.rad2deg(np.angle(rp / rs))), abs=1e-12)
    assert fresnel_phase_difference(EPS1, EPS2, THETA) == pytest.approx(
        70.528779366, abs=1e-6)


def test_EGIK_KATSAYILAR_KUTUPLANMAYA_GORE_DUALITE_ILE_SLOTLANIYOR():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    s'te `_adim`in (ca, cb) slotu Ez'nin, (da, db) slotu Hy'nindir; p'de
    (ca, cb) Hz'nin, (da, db) Ey'nindir. Bu bir yorum değil sayıdır:
    s-katsayıları `yee_coefficients`ınkilerle BİREBİR aynı olmalı, p'ninki
    ise `eps_yari` üzerinden kurulmalı ve `ey_kural`a BAĞLI olmalıdır.
    """
    from qlab.engine.em.yee import yee_coefficients

    eps = np.ones(200)
    eps[100:] = 4.0
    a = Medium(eps, np.ones(200), "aritmetik")
    b = Medium(eps, np.ones(200), "harmonik")
    sinir = PML(thickness=15)

    temel = yee_coefficients(a, courant=0.4, boundary=sinir)
    s = oblique_coefficients(a, courant=0.4, boundary=sinir, polarization="s",
                             ky=0.1)
    assert np.array_equal(s.main.ca, temel.ca)
    assert np.array_equal(s.main.cb, temel.cb)
    assert np.array_equal(s.main.da, temel.da)
    assert np.array_equal(s.main.db, temel.db)
    assert s.sign == -1.0

    pa = oblique_coefficients(a, courant=0.4, boundary=sinir, polarization="p",
                              ky=0.1)
    pb = oblique_coefficients(b, courant=0.4, boundary=sinir, polarization="p",
                              ky=0.1)
    assert pa.sign == +1.0
    # p'nin (da, db) slotu YARIM HÜCREDEKİ TEĞET E'dir -> `ey_kural`a bağlı.
    assert not np.array_equal(pa.main.db, pb.main.db)
    # s ise `ey_kural`a bağlı DEĞİL (yarım hücrede E yok).
    sb = oblique_coefficients(b, courant=0.4, boundary=sinir, polarization="s",
                              ky=0.1)
    assert np.array_equal(s.main.db, sb.main.db)


def _vakum_egik_katsayilari(L: int, S: float, ky: float, pol: str):
    """Vakum + PEC eğik katsayıları, KARARLILIK KAPISINI ATLAYARAK.

    Eşik ölçümü kapıyı atlamak ZORUNDADIR — ölçülecek olan şey tam olarak
    O REDDİN NEREDE olduğudur, ve kapı devredeyken ölçüm kendi kapısını
    ölçerdi (daireselleşirdi). `test_sema_cipalari._vakum_katsayilari` ile
    AYNI gerekçe; koşan güncelleme yine SEVK EDİLEN `oblique_step`tir.

    Sayılar `oblique_coefficients`in vakumda ürettiğinin AYNISIDIR:
    `sigma = 0` olduğu için `d_e = 0`, dolayısıyla ca = da = 1, cb = db = S;
    `mu_r = 1` ve `eps_yarı = 1` olduğu için yardımcı slot da (1, S). İki
    kutuplanmada SAYILAR AYNI, yalnız `sign` farklı — ve bu, `sign`in bir
    GAUGE olduğu iddiasının bu testteki ölçülen karşılığıdır.

    G13'ten (12.10.2026) beri kapı TİPİN KENDİSİNDEDİR (`__post_init__`); atlama
    bu yüzden `object.__new__` ile, açık ve yerel yapılır.
    """
    base = YeeCoefficients(
        ca=np.ones(L), cb=np.full(L, S),
        da=np.ones(L - 1), db=np.full(L - 1, S), dim=1, courant=S,
    )
    co = object.__new__(ObliqueCoefficients)
    for ad, deger in (("main", base), ("aux_a", np.ones(L)), ("aux_b", np.full(L, S)),
                      ("sign", -1.0 if pol == "s" else +1.0), ("polarization", pol),
                      ("ky", float(ky))):
        object.__setattr__(co, ad, deger)
    return co


def _egik_kararli_mi(S: float, ky: float, L: int, adim: int, tohum: int,
                     pol: str = "s") -> bool:
    """SEVK EDİLEN `oblique_step`, PEC duvar, tohumlu rastgele başlangıç."""
    rng = np.random.default_rng(tohum)
    st = FieldState(rng.standard_normal(L) + 0j,
                    rng.standard_normal(L - 1) + 0j, 1)
    aux = rng.standard_normal(L) + 0j
    co = _vakum_egik_katsayilari(L, S, ky, pol)
    n0 = float(np.max(np.abs(st.e)))
    for _ in range(adim):
        st, aux = oblique_step(st, aux, co)
        m = float(np.max(np.abs(st.e)))
        if not np.isfinite(m) or m > 1e12 * n0:
            return False
    return True


def _yogun_egik_p_esigi(med: Medium, ky: float) -> float:
    """Kapının modelini BAĞIMSIZ kurar: açık D matrisi + yoğun eigvalsh (üç-köşegen kısayolu yok)."""
    n = med.cells
    D = (np.eye(n - 1, n, 1) - np.eye(n - 1, n))[:, 1:-1]
    K = D.T @ np.diag(1.0 / med.eps_yari()) @ D + ky**2 * np.diag(1.0 / med.eps_r[1:-1])
    s = 1.0 / np.sqrt(med.mu_r[1:-1])
    return float(2.0 / np.sqrt(np.linalg.eigvalsh(s[:, None] * K * s[None, :]).max()))


def _egik_patlar_mi(co: ObliqueCoefficients, L: int, adim: int = 3000, tohum: int = 3) -> bool:
    rng = np.random.default_rng(tohum)
    st = FieldState(rng.standard_normal(L) + 0j, rng.standard_normal(L - 1) + 0j, 1)
    aux = rng.standard_normal(L) + 0j
    n0 = float(np.max(np.abs(st.e)))
    for _ in range(adim):
        st, aux = oblique_step(st, aux, co)
        m = float(np.max(np.abs(st.e)))
        if not np.isfinite(m) or m > 1e12 * n0:
            return True
    return False


def test_EGIK_P_KESIN_KAPI_ESLESMEYI_KESER_ve_TAM_ESIGE_OTURUR(monkeypatch):
    """SINIF: ŞEMA ÇIPASI — kapanmış ESKİ açığın kilidi (02.10.2026).

    p'de Hz tam hücrede mu_r, Ey yarı hücrede `eps_yari` (`ey_kural`) ile
    eşlenir. Bu ortamda her hücrede eps·mu = 1, yani hücre çarpımı kapısı
    S <= 1'e izin veriyordu; şema harmonikte 0.627587'de, sağda 0.470588'de
    patlıyordu (c4c66ad'de public `run_oblique_1d` eşiğin hemen üstündeki S'yi
    kabul edip patlıyordu). Beklenen sayılar TEK ADIM BÜYÜTME MATRİSİNİN eşiğidir
    (modelle <= 5.5e-13, bölme aracının payı dahil), kodun kendi çıktısı DEĞİL;
    model ayrıca yoğun matrisle bağımsız kurulur. Kapı tam eşikte açılır,
    0.999'u kararlı, kapı atlanınca 1.001'i patlar; aritmetikte ve mu = 1'de
    bugünkü kapıdan SIKI değildir.

    eps·mu = 1 ortamında 1/eps_r = mu_r OLDUĞU İÇİN ky² terimini mu_r'den kuran
    mutant ilk dört çapayı geçiyordu (doğrulayıcı yakaladı) — İKİNCİ ÇAPA
    eps·mu != 1, konuma bağlı mu ve ky = 0.8'dedir: eps 2|0.5, mu 0.3|1, sağ,
    L = 40. Eşiği kendi büyütme matrisi aracımla 0.5775927064 (Jordan noktasında
    özdeğer duyarlılığı sqrt(eps) olduğu için 1e-8 hassasiyetinde; model
    0.5775927075); dinamik: 0.999 sınırlı (tepe 2.3), kapı atlanınca 1.001 312.
    adımda, mutantın eşiğinin 0.999'u 74. adımda patlıyor. Bu çapa aynı zamanda
    kapının ORTAMDAN kurulmasının (tek kaynak istisnası) bekçisidir.
    """
    L = 200
    eps = np.r_[np.full(L // 2, 4.0), np.full(L // 2, 0.25)]
    mu = eps[::-1].copy()
    matris = {("harmonik", 0.0): 0.627586660138914, ("sag", 0.0): 0.470588235294272,
              ("harmonik", 0.8): 0.608700014742208, ("sag", 0.8): 0.462466377568198}
    for (kural, ky), sayi in matris.items():
        med = Medium(eps, mu, kural)
        esik = _egik_p_kesin_esik(med, ky)
        assert esik == pytest.approx(sayi, rel=1e-12)
        assert esik == pytest.approx(_yogun_egik_p_esigi(med, ky), rel=1e-12)
        oblique_coefficients(med, courant=esik, boundary=PEC(), polarization="p", ky=ky)
        with pytest.raises(InvalidInputError, match="KESIN kararlilik"):
            oblique_coefficients(med, courant=esik * (1 + 1e-12), boundary=PEC(), polarization="p", ky=ky)
        oblique_coefficients(med, courant=0.9 * oblique_courant_limit(ky), boundary=PEC(), polarization="s", ky=ky)
    med = Medium(eps, mu, "harmonik")
    esik = _egik_p_kesin_esik(med, 0.0)
    kur = lambda S: oblique_coefficients(med, courant=S, boundary=PEC(), polarization="p", ky=0.0)
    assert not _egik_patlar_mi(kur(0.999 * esik), L)
    monkeypatch.setattr("qlab.engine.em.oblique._egik_p_kesin_esik", lambda *a: np.inf)
    with pytest.raises(InvalidInputError, match="KARARSIZ"):  # G13: tipin kendi kapisi da keser
        kur(1.001 * esik)
    monkeypatch.setattr("qlab.engine.em.oblique._katsayi_kapisi", lambda *a: None)
    assert _egik_patlar_mi(kur(1.001 * esik), L)
    monkeypatch.undo()
    # İKİNCİ ÇAPA: eps·mu != 1, konuma bağlı mu, ky != 0 (aux yolu canlı).
    L2, ky2 = 40, 0.8
    med2 = Medium(np.r_[np.full(20, 2.0), np.full(20, 0.5)], np.r_[np.full(20, 0.3), np.full(20, 1.0)], "sag")
    esik2 = _egik_p_kesin_esik(med2, ky2)
    assert esik2 == pytest.approx(0.5775927064071993, rel=1e-8)
    kur2 = lambda S: oblique_coefficients(med2, courant=S, boundary=PEC(), polarization="p", ky=ky2)
    assert not _egik_patlar_mi(kur2(0.999 * esik2), L2)
    monkeypatch.setattr("qlab.engine.em.oblique._egik_p_kesin_esik", lambda *a: np.inf)
    monkeypatch.setattr("qlab.engine.em.oblique._katsayi_kapisi", lambda *a: None)
    assert _egik_patlar_mi(kur2(1.001 * esik2), L2)
    monkeypatch.undo()
    # Sayısal aralık dışı ölçek: kapalı yönde ve DOĞRU istisna türüyle.
    with pytest.raises(InvalidInputError, match="HESAPLANAMIYOR"):
        oblique_coefficients(Medium(np.full(10, 1e170), np.full(10, 1e170)), courant=0.5,
                             boundary=PEC(), polarization="p", ky=0.0)
    # Bugünkü kapıdan SIKI olmadığı yerler: aritmetik (aynı ortam) ve em_obs biçimi (mu = 1).
    oblique_coefficients(Medium(eps, mu), courant=1.0, boundary=PEC(), polarization="p", ky=0.0)
    for kural in ("harmonik", "sol", "sag"):
        yari = Medium.half_space(L, interface=L // 2, eps1=4.0, eps2=1.0, ey_kural=kural)
        oblique_coefficients(yari, courant=oblique_courant_limit(0.55536), boundary=PEC(), polarization="p", ky=0.55536)


def test_KARARLILIK_SINIRI_ky_ILE_DUSUYOR_ve_KAPI_ARTIK_SINIYOR():
    """SINIF: ŞEMA ÇIPASI — ve KAPANMIŞ bir dürüst olumsuzun kilidi.

    Sevk edilen izotropik kapı `courant_limit(1)` = 1'dir ve `ky`yi HESABA
    KATMAZ. Gerçek sınır `ky` ile düşer: `oblique_courant_limit(ky)` =
    `1/sqrt(1 + (ky/2)^2)`. İKİNCİ BİR KAPI artık `oblique_coefficients`te
    duruyor; türetimi, iki parçalı artık aşımı ve tam tablosu orada.

    BU TEST BİR ZAMANLAR `..._KAPI_BUNU_SINAMIYOR` ADINI TAŞIYORDU ve
    dürüst olumsuzu kilitliyordu: *"`ky` yeterince büyütülürse
    `courant_limit(1)` geçer ve şema yine de patlar."* O cümle DOĞRUYDU ve
    artık GEÇERSİZDİR — kapı kondu, ve bu testin işi artık kapının YERİNİ
    ölçmek.

    ÖLÇÜM KAPIYI ATLAMAK ZORUNDA (`_vakum_egik_katsayilari`), yoksa ölçülen
    şey şema değil kapının kendisi olurdu — `test_sema_cipalari`nin Courant
    eşiği ölçümüyle aynı gerekçe.

    PENCERE: vakum, PEC duvarlı L = 201, 800 adım, tohumlu rastgele
    başlangıç (tohum 3), ikili arama 30 adım (60 kısa koşum), tespit ölçütü
    `max|e| > 1e12 * başlangıç`. ÖLÇÜLEN:

        ky           0.2         1.2
        ölçülen      0.99522660  0.85764172
        kapalı(inf)  0.99503719  0.85749293
        kapalı(PEC)  0.99506758  0.85751237

    TESPİT ÖLÇÜTÜ DE BİR PENCERE DÜĞMESİDİR ve bu ölçüldü: aynı pencerede
    ölçüt 1e12 yerine 1e6 olsaydı ky = 1.2'de 0.85754172 çıkardı (fark
    1.0e-04). Bu dosyada bir zamanlar sevk edilen tablo (kaynak sürülen
    koşum, 1e6 mutlak eşiği) tam da o yüzden 0.85755547 diyordu — sayı
    yanlış değildi, PENCERESİ farklıydı. Ayrıntı ve yakınsama:
    `oblique.oblique_courant_limit`.
    """
    L, ADIM, TOHUM = 201, 800, 3
    for ky, beklenen in ((0.2, 0.99522660), (1.2, 0.85764172)):
        lo, hi = 0.05, 1.5
        for _ in range(30):
            m = 0.5 * (lo + hi)
            if _egik_kararli_mi(m, ky, L, ADIM, TOHUM):
                lo = m
            else:
                hi = m
        olculen = 0.5 * (lo + hi)
        kapali = oblique_courant_limit(ky)
        kapali_pec = 1.0 / np.sqrt(
            np.cos(np.pi / (2 * (L - 1))) ** 2 + (ky / 2.0) ** 2
        )
        assert olculen == pytest.approx(beklenen, abs=1e-7)
        # ÖLÇÜLEN eşik kapalı formun ÜSTÜNDE — işareti de bir iddiadır.
        assert olculen > kapali_pec > kapali
        # ve artık aşım 2e-04'ün altında (sonlu ızgara + sonlu koşum payı).
        assert olculen / kapali_pec == pytest.approx(1.00016, abs=1e-5)

    # KAPALI FORM `courant_limit(1)`İ GENELLİYOR, ONU ÇELİŞMİYOR.
    assert oblique_courant_limit(0.0) == courant_limit(1)
    assert oblique_courant_limit(1.2) == pytest.approx(0.85749293, abs=5e-9)
    with pytest.raises(InvalidInputError, match="ky sonlu"):
        oblique_courant_limit(float("nan"))

    # KAPI ARTIK SINIYOR: S = 0.9 izotropik kapıyı GEÇER (ve ky = 0'da haklı
    # olarak geçer, çünkü orada KARARLIDIR) ama ky = 1.2'de KESİLİR.
    vak = Medium.vacuum(64)

    def kur(S, ky):
        return oblique_coefficients(vak, courant=S, boundary=PEC(),
                                    polarization="s", ky=ky)

    assert 0.9 < courant_limit(1)
    assert kur(0.9, 0.0).main.courant == 0.9  # ky = 0'da geçiyor
    assert _egik_kararli_mi(0.9, 0.0, L, ADIM, TOHUM)  # ve GERÇEKTEN kararlı
    with pytest.raises(InvalidInputError, match="EGIK indirgemenin"):
        kur(0.9, 1.2)
    assert not _egik_kararli_mi(0.9, 1.2, L, ADIM, TOHUM)  # kesilen HAKLI

    # İKİ KAPI DA CANLI: izotropik olan S > 1'i, eğik olan BANDI kesiyor.
    with pytest.raises(InvalidInputError, match="1B'de S_max"):
        kur(1.5, 0.1)
    with pytest.raises(InvalidInputError, match="EGIK indirgemenin"):
        kur(0.99, 0.4)
    assert kur(0.98, 0.4).ky == 0.4  # bandın hemen altı GEÇİYOR

    # KAPI SINIRIN TAM ÜSTÜNDE AÇILIYOR — izotropik kapıyla AYNI uzlaşım
    # (`test_yee.test_COURANT_KAPISI_SINIRIN_TAM_USTUNDE_ACILIYOR...`).
    # `S == S_max` GEÇER, bir sonraki temsil edilebilir sayı KESİLİR — ve
    # HANGİ KAPININ kestiği `ky`ye bağlı: `ky = 0`da iki eşik ÇAKIŞTIĞI için
    # kesen şey izotropik kapıdır, `ky > 0`da eğik kapı.
    for ky, desen in ((0.0, "1B'de S_max"), (0.4, "EGIK indirgemenin"),
                      (1.2, "EGIK indirgemenin")):
        tam = oblique_courant_limit(ky)
        assert kur(tam, ky).ky == ky
        with pytest.raises(InvalidInputError, match=desen):
            kur(np.nextafter(tam, 2.0), ky)

    # SEVK EDİLEN PENCEREDE PAY BÜYÜK — kapı hiçbir şeyi kesmiyor.
    ky_sevk = (2.0 * np.pi / 16.0) * np.sqrt(EPS1) * np.sin(np.deg2rad(45.0))
    assert ky_sevk == pytest.approx(0.555360, abs=1e-6)
    assert oblique_courant_limit(ky_sevk) == pytest.approx(0.963542, abs=1e-6)
    assert oblique_courant_limit(ky_sevk) / SEVK.courant == pytest.approx(
        2.41, abs=0.01
    )


def test_EGIK_KOSUM_PENCERESINI_DONDURMEK_ZORUNDA():
    """SINIF: TRANSKRİPSİYON DENETİMİ. Penceresiz bir kayıt, sayının hangi
    ayrıklaştırmaya ait olduğunu çağrıda kaybederdi."""
    ortam = Medium.vacuum(200)
    k = run_oblique_1d(
        ortam, courant=0.4, boundary=PML(thickness=15), steps=10,
        source=None, source_index=None, probes=(100,), wavelength=20.0,
        sgf=9.0, ky=0.1, polarization="p",
    )
    assert isinstance(k, FieldRecord)
    assert k.window.npml == 15
    assert k.window.sgf == 9.0
    assert k.window.window_fn == "dikdortgen"
    assert k.window.courant == 0.4
    assert k.probes.dtype == complex
    # PEC sınırda `thickness` 0 -> pencere onu da doğru taşımalı.
    k2 = run_oblique_1d(
        ortam, courant=0.4, boundary=PEC(), steps=10, source=None,
        source_index=None, probes=(100,), wavelength=20.0, sgf=9.0, ky=0.1,
        polarization="s",
    )
    assert k2.window.npml == 0


def test_EGIK_GIRDI_DENETIMI():
    """SINIF: TRANSKRİPSİYON DENETİMİ."""
    ortam = Medium.vacuum(200)
    sinir = PML(thickness=15)
    with pytest.raises(InvalidInputError, match="polarization"):
        oblique_coefficients(ortam, courant=0.4, boundary=sinir,
                             polarization="TE", ky=0.1)
    with pytest.raises(InvalidInputError, match="ky sonlu"):
        oblique_coefficients(ortam, courant=0.4, boundary=sinir,
                             polarization="s", ky=float("nan"))
    # Courant kapısı EGİK yolda da koşuyor -- tek kaynaktan.
    with pytest.raises(InvalidInputError, match="kararlilik"):
        oblique_coefficients(ortam, courant=1.5, boundary=sinir,
                             polarization="p", ky=0.1)
    with pytest.raises(InvalidInputError, match="steps"):
        run_oblique_1d(ortam, courant=0.4, boundary=sinir, steps=-1,
                       source=None, source_index=None, probes=(10,),
                       wavelength=20.0, sgf=9.0, ky=0.0, polarization="s")
    with pytest.raises(InvalidInputError, match="prob indisi"):
        run_oblique_1d(ortam, courant=0.4, boundary=sinir, steps=1,
                       source=None, source_index=None, probes=(500,),
                       wavelength=20.0, sgf=9.0, ky=0.0, polarization="s")
    with pytest.raises(InvalidInputError, match="polarization"):
        reflection_coefficient(eps1=4.0, eps2=1.0, theta_deg=45.0,
                               wavelength=40.0, polarization="x")
    with pytest.raises(InvalidInputError, match="window_fn"):
        reflection_coefficient(eps1=4.0, eps2=1.0, theta_deg=45.0,
                               wavelength=40.0, polarization="s",
                               window=dataclasses.replace(SEVK,
                                                          window_fn="kaiser"))
    with pytest.raises(InvalidInputError, match="pozitif"):
        fresnel_coefficients(4.0, -1.0, 45.0)


def test_KAYIT_UZUNLUGU_GUVENLI_DEGILSE_NONE_ve_IKI_AYRI_KAPI_VAR():
    """SINIF: TRANSKRİPSİYON DENETİMİ + `None` disiplini.

    Yansıyan darbeyi İÇEREN ama sol soğurucudan dönen yankıyı İÇERMEYEN bir
    kayıt uzunluğu yoksa ölçüm yapılmaz. `_record_steps` bunu **İKİ AYRI
    kapıyla** yapar ve ikisi de ayrı girdilerde açılır:

    1. `lo >= hi` — hiç aralık yok. Ölçüldü: sevk geometrisinde `lambda` =
       160'ta böyle (10 sigma = 14400, prob-duvar gidiş-dönüşü 8451). Bu,
       `r = 2`lik bir merdivenin bu pencerede neden kurulamadığının sebebi.
    2. `T > hi` — aralık VAR ama `lambda`nın tam katına yuvarlanan `T`
       aralığın dışına düşüyor. Bu kapı bir MUTASYON TURUNDA bulundu: ilk
       hâlde onu açan hiçbir test yoktu (mutant hayatta kaldı). Arandı ve
       BULUNDU -- sevk `sgf`inde, `theta` = 34°, `lambda` = 80'de
       lo = 9385.67 < hi = 9394.03 ama T = 9440 > hi. Aralık yalnız 8.4
       zaman birimi; yuvarlama onu yutuyor.

    İkinci kapı olmasaydı ölçüm SESSİZCE yankıyı içeren bir kayıt üzerinde
    yapılırdı -- yani `None` dönmesi gereken yerde bir sayı basılırdı.
    """
    # 1. kapı
    assert _record_steps(theta_deg=THETA, wavelength=160.0, eps1=EPS1,
                         window=SEVK) is None
    assert _record_steps(theta_deg=THETA, wavelength=40.0, eps1=EPS1,
                         window=SEVK) == 15500
    assert reflection_coefficient(eps1=EPS1, eps2=EPS2, theta_deg=THETA,
                                  wavelength=160.0, polarization="s") is None

    # 2. kapı: aralık VAR (lo < hi) ama T onu aşıyor.
    sg = SEVK.sgf * 80.0
    t0 = 5.0 * sg
    vx = np.cos(np.deg2rad(34.0)) / np.sqrt(EPS1)
    t_yans = t0 + (2 * SEVK.interface - SEVK.source_index - SEVK.probe) / vx
    lo = t_yans + 5 * sg
    hi = t_yans + 2.0 * SEVK.probe / vx - 5 * sg
    assert lo < hi                       # birinci kapı GEÇERDİ
    assert lo == pytest.approx(9385.67, abs=0.01)
    assert hi == pytest.approx(9394.03, abs=0.01)
    assert np.ceil(lo / 80.0) * 80.0 == 9440.0 > hi
    assert _record_steps(theta_deg=34.0, wavelength=80.0, eps1=EPS1,
                         window=SEVK) is None


@pytest.mark.slow
def test_FFT_BINI_OTURMAZSA_NONE_cunku_ETKIN_ACI_KAYAR():
    """SINIF: TRANSKRİPSİYON DENETİMİ + `None` disiplini.

    `ky` koşum boyunca SABİTTİR, ama bir FFT bininin gördüğü ETKİN açı
    `asin(ky/(2 pi f n1))`dir. Hedef bin `1/lambda`ya tam oturmazsa ölçülen
    şey BAŞKA BİR AÇININ `r`sidir -- ve bu sessizce olur.

    Bin ancak `T`nin `lambda`nın, `steps = T/S`nin de TAM SAYI olduğu
    yerde oturur. Ölçüldü (sevk penceresi, `S` = 0.4):

        lambda   lambda/S    |f_bin*lambda - 1|   rho
          30      75.000        0.000e+00         1.000000
          40     100.000        0.000e+00         1.000000
          45     112.500        3.023e-05         None
          47     117.500        2.935e-05         None
          67.5   168.750        0.000e+00         1.000498

    Son satır DÜRÜST OLUMSUZDUR: `lambda/S` tam sayı olmadığı hâlde bin
    oturabiliyor (yuvarlama dört periyodun katına düşerse). Yani kapı
    `lambda/S`yi değil BİNİN KENDİSİNİ sınıyor -- doğru olan da budur.

    Bu kapı da MUTASYON TURUNDA bulundu: kaldırıldığında hiçbir test
    düşmüyordu, çünkü sevk edilen bütün `lambda`ların `lambda/S`si tam sayı.
    """
    assert reflection_coefficient(eps1=EPS1, eps2=EPS2, theta_deg=THETA,
                                  wavelength=45.0, polarization="s") is None
    # Ama kayıt uzunluğu VARDI -- yani düşen kapı bu değil, bin kapısı.
    assert _record_steps(theta_deg=THETA, wavelength=45.0, eps1=EPS1,
                         window=SEVK) == 16538
    steps = 16538
    fr = np.fft.fftfreq(steps, d=SEVK.courant)
    i = int(np.argmin(np.abs(fr - 1.0 / 45.0)))
    assert abs(fr[i] * 45.0 - 1.0) == pytest.approx(3.023e-05, rel=0.02)
    # Sevk edilen merdivenin dört basamağında bin TAM oturuyor.
    for lam in MERDIVEN:
        st = _record_steps(theta_deg=THETA, wavelength=lam, eps1=EPS1,
                           window=SEVK)
        fr = np.fft.fftfreq(st, d=SEVK.courant)
        i = int(np.argmin(np.abs(fr - 1.0 / lam)))
        assert abs(fr[i] * lam - 1.0) == 0.0


def test_PENCERE_BEYANI_TAM():
    """SINIF: TRANSKRİPSİYON DENETİMİ. Pencere adlandırılmazsa hüküm
    pencerenin dışına taşınır -- bu depoda ON BİR kez oldu."""
    satirlar = SEVK.as_lines()
    metin = " ".join(satirlar)
    for parca in ("3600", "0.4", "1500", "600", "1494", "60", "m = 3", "9",
                  "dikdortgen", "aritmetik", "0.01", "0.0016", "0.00045"):
        assert parca in metin
    assert SEVK.probe == SEVK.interface - SEVK.probe_offset
    assert dataclasses.replace(SEVK, probe_offset=300).probe == 1200


def test_SLOW_ESIGI_OLCULDU():
    """SINIF: SÜREÇ KİLİDİ -- devralınabilir bir eşik YOK, ölçülür.

    Bu dosyada `@pytest.mark.slow` eşiği **setup+call > 1.0 s** olarak
    ölçüldü, devralınmadı. Ölçülen iki küme ve ARALARINDAKİ BOŞLUK:

        işaretsiz, en pahalı : 0.67 s
            (`test_MERDIVEN_KAPIYA_TAKILAN...` ve
             `test_KARARLILIK_SINIRI_ky_ILE_DUSUYOR...` — aynı koşumda
             İKİSİ DE 0.67 s, yani gürültü içinde BERABERE)
        `slow`, en ucuz      : 1.52 s
            (`test_FFT_BINI_OTURMAZSA_NONE...`)

    Eşik ikisinin ARASINDA ve ikisine de pay bırakıyor (aşağıda 1.49 kat,
    yukarıda 1.52 kat). Sebep yapısaldır: `tir_phase_difference` çağrısı
    başına 4 FDTD koşumudur ve sevk penceresinde 1.414-2.595 s sürer, yani
    BİR çağrı bile eşiği aşar.

    İKİ ATIF DA YENİLENDİ ve sebepleri FARKLI:
      * Alt uç, `oblique_courant_limit` kapısı konurken DEĞİŞTİ: bu dosyanın
        kararlılık kilidi kaynak sürülen koşumdan tohumlu koşuma geçti
        (kapıyı atlamak zorunlu olduğu için) ve 0.69 s → 0.58-0.67 s'ye
        indi; artık en pahalı işaretsiz test TEK BAŞINA o değil.
      * Üst uç ATIFI ise ÖLÇÜLDÜĞÜNDE ZATEN BAYATTI: aynı koşumda
        `test_FRESNEL_R_CEVIRI_DENETIMI...` 1.84 s, `test_FFT_BINI...` ise
        1.52 s çıktı — yani `slow` kümesinin en ucuzu artık ikincisidir.
        Ne zaman değiştiği bilinmiyor ve UYDURULMUYOR; yazılan şey bugün
        ölçülendir.

    MAKİNE GÜRÜLTÜSÜ BEYAN EDİLİR: aynı dosyanın iki ardışık koşumunda
    bu dört sayı 0.61/0.58/1.27/1.70 ve 0.67/0.67/1.52/1.84 çıktı, yani
    %20'ye varan oynama var. Eşiğin ikisine de bıraktığı pay (1.49 ve 1.52
    kat) bu oynamanın ÜSTÜNDEDİR — sınıflandırma gürültüyle değişmiyor.

    Test eşiğin ARİTMETİĞİNİ kilitler, SÜREYİ ÖLÇMEZ: süre ölçen bir test
    makinenin yüküne bağlı olurdu ve bu depoda o tuzağa daha önce girilmişti.
    """
    isaretsiz_en_pahali = 0.67  # ölçüldü
    slow_en_ucuz = 1.52  # ölçüldü
    esik = 1.0
    assert isaretsiz_en_pahali < esik < slow_en_ucuz
    assert esik / isaretsiz_en_pahali == pytest.approx(1.49, abs=0.02)
    assert slow_en_ucuz / esik == pytest.approx(1.52, abs=0.02)

    # Ve `slow` işaretinin gerçekten uygulandığının SAYIMI:
    import inspect

    kaynak = inspect.getsource(inspect.getmodule(test_SLOW_ESIGI_OLCULDU))
    dekorator = [s for s in kaynak.splitlines() if s == "@pytest.mark.slow"]
    assert len(dekorator) == 10
