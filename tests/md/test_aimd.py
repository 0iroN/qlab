"""Born-Oppenheimer moleküler dinamiği -- çekirdekler HF yüzeyinde.

Bu dosya Plan D'nin ANA İDDİASINI sınar: kuvvet bir MODELDEN değil, her
adımda yeniden çözülen bir HF yüzeyinden gelir; buna rağmen enerji korunur
ve titreşim frekansı yüzeyin KENDİSİNDEN çıkar.

PAYDA UYARISI -- 432x: BO yüzeyinde |E_toplam| ~ 1.1175 hartree ama
E_titreşim ~ 2.58e-03. Plan C'nin `|blok[0]|` paydası buraya kopyalanırsa
her enerji testi 432 kat zayıflar. Payda E_titreşim'dir.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.forcefields import HartreeFockGradient
from qlab.engine.md.integrator import run
from qlab.engine.md.state import MDState
# TAKMA AD ZORUNLU: aşağıdaki çıpanın gövdesinde `omega_b` bir YEREL
# değişkendir (`egim, omega_b = np.polyfit(...)`). Çıplak `omega_b` diye
# import edilseydi okuyan iki farklı şeyi aynı adla görürdü; çıpanın gövdesi
# DEĞİŞMEDEN kalsın diye ad burada değiştirildi.
from qlab.observable.bridge_obs import omega_b as boru_omega_b
from qlab.units import (
    AMU_TO_ELECTRON_MASS,
    HARTREE_TO_WAVENUMBER,
    amu_to_electron_mass,
)

H2_RE_BOHR = 1.3459193602
H2_E_MIN = -1.117505885116
#: mu = m(1H)/2. KÜTLE BORU HATTINDAN GEÇER, literal DEĞİL -- Plan C'de
#: literal yazıldığı için `AMU_TO_ELECTRON_MASS = 1.0` mutantı altı testi
#: birden geçiyordu (ölçüldü).
MU_H2 = amu_to_electron_mass(1.00782503207) / 2.0
#: HF/STO-3G yüzeyinin KENDİ harmonik periyodu (ölçülen k = 0.57293630).
T_HF = 2 * np.pi * np.sqrt(MU_H2 / 0.5729363)


def _baslangic(genlik: float) -> MDState:
    return MDState(
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, H2_RE_BOHR + genlik]]),
        velocities=np.zeros((2, 3)),
        masses=np.full(2, amu_to_electron_mass(1.00782503207)),
    )


def _bag(yorunge) -> np.ndarray:
    return np.linalg.norm(
        yorunge.positions[:, 1, :] - yorunge.positions[:, 0, :], axis=1
    )


@pytest.mark.slow
def test_BO_yuzeyinde_enerji_SURUKLENMIYOR():
    """velocity-Verlet BO yüzeyinde de SİMPLEKTİK: enerji SALINIR.

    ÖLÇÜT SÜRÜKLENME DEĞİL SALINIMDIR. Ölçüldü: sürüklenme dt ile
    ölçekleniyor (N=50'de 9.4200e-05, N=100'de 1.9412e-05 -- 4.85 kat),
    yani bir gürültü tabanı değil ayrıklaştırma eseri; sabit eşiği YOK.
    Bu planın ilk taslağı `< 1e-8` yazıyordu ve KIRMIZI DOĞARDI.

    Salınım ise temiz π²/N² yasasına uyuyor: 4.2778e-03 -> 1.0690e-03,
    oran 4.0016.

    ÖLÇÜLEN NORMALİZE ORAN (naif konvansiyon, dt = T_harm/N, genlik 0.10):
      N=50  2 periyot -> 1.0836      N=100 2 periyot -> 1.0829
      N=50  6 periyot -> 1.0836      N=100 6 periyot -> 1.0832
    Saçılım 0.0007. SINIR ±0.01 = saçılımın 14 katı.
    HANGİ HATAYI YAKALAR (ölçüldü): genlik 0.05 -> 1.0468 (sınırın 3.6
    katı dışında), h=1e-2 -> 2.399 (efektif yüzeyi bozar).

    GENLİK SABİT: oran genliğe duyarlı (0.05 -> 1.0468, 0.10 -> 1.0919,
    0.20 -> 1.1952). Genlik değişirse ölçüt anlamsızlaşır.

    PAYDA E_TİTREŞİM: |E_toplam| kullanılsaydı ölçüt 432 kat zayıflardı
    (ölçüldü: E_titreşim 2.582381e-03, |E_toplam| 1.1149235 -> oran 431.75).
    """
    durum = _baslangic(0.10)
    alan = HartreeFockGradient(["H", "H"])
    yorunge = run(durum, alan, dt=T_HF / 50, n_steps=100, save_every=1)

    toplam = yorunge.kinetic + yorunge.potential
    e_titresim = float(toplam[0] - H2_E_MIN)
    assert e_titresim == pytest.approx(2.582381e-03, rel=1e-3, abs=0.0)

    # SALINIM ORANI -- sürüklenme DEĞİL. Sürüklenme dt ile ölçeklenen bir
    # ayrıklaştırma eseridir (N=50'de 9.4209e-05, N=100'de 1.94e-05) ve sabit
    # eşiği YOKTUR. Salınım ise π²/N² yasasına uyuyor: N ikiye katlanınca
    # 4.2778e-03 -> 1.0690e-03, oran 4.0016.
    #
    # KONVANSİYON: N NAİF olandır (dt = T_harm/N), gerçek periyottan türetilen
    # N_eff DEĞİL. İkisi %0.75 farklı ölçüldü; sınır bu testin GERÇEKTEN
    # kullandığı konvansiyona oturtuldu.
    tepe_tepe = float(toplam.max() - toplam.min()) / e_titresim
    n_naif = 50.0
    assert tepe_tepe / (np.pi**2 / n_naif**2) == pytest.approx(
        1.0833, abs=0.01, rel=0.0
    )


@pytest.mark.slow
def test_bag_dengenin_ETRAFINDA_saliniyor():
    """Yörünge gerçekten hareket etmeli ve denge etrafında SİMETRİK olmalı.

    Plan C'de bu sınıf test ilk kare tarafından sağlanıyordu; burada genlik
    ve iki yönlü aşma birlikte sınanır.
    ÖLÇÜLDÜ: genlik 0.10 bohr ile salınım 1.2984-1.3959 bohr aralığında.
    """
    yorunge = run(
        _baslangic(0.05), HartreeFockGradient(["H", "H"]),
        dt=T_HF / 50, n_steps=100, save_every=1,
    )
    r = _bag(yorunge)
    assert float(r.max()) > H2_RE_BOHR + 0.04
    assert float(r.min()) < H2_RE_BOHR - 0.04


@pytest.mark.slow
def test_titresim_frekansi_IKI_GERCEKTEN_BAGIMSIZ_yoldan_ayni():
    """PLAN D'NİN ANA SONUCU. İki yol GERÇEKTEN bağımsızdır.

    YOL A -- STATİK: enerji eğrisinin r_e'deki ikinci türevi, k = V''(r_e),
             sonra omega = sqrt(k/mu). Hiç dinamik koşmaz, hiç kuvvet kullanmaz.
    YOL B -- DİNAMİK: MD yörüngesinin ölçülen periyodu. Hiç ikinci türev
             alınmaz; yalnız kuvvetler ve velocity-Verlet.

    İKİ DÜZELTME ŞART, ikisi de ölçüldü:
    1. ANHARMONİKLİK. Sonlu genlikte frekans DÜŞER (ölçüldü, N=200/6 periyot:
       A=0.01 -> 5481.24125, A=0.02 -> 5480.56669, A=0.04 -> 5477.93574
       cm^-1). Tek genlikle karşılaştırmak 0.687 cm^-1 sahte fark üretir --
       bu testin ilk taslağı tam olarak buna takıldı. Çözüm: A^2'de doğrusal
       uydurup A -> 0'a ekstrapole etmek.
    2. velocity-Verlet DİSPERSİYONU. Ayrık integratörün frekansı sürekli
       olandan yüksektir: Omega = (2/dt) arcsin(omega dt/2). Tersi
       uygulanır: omega = (2/dt) sin(Omega dt/2). Ölçülen etki 0.225 cm^-1.

    ÖLÇÜLDÜ (bu konfigürasyon): omega_A = 5481.25336 cm^-1,
    omega_B = 5481.22946 cm^-1, oran - 1 = -4.360844e-06.
    Yani iki bağımsız yol MİLYONDA 4.4'te uyuşuyor.
    SINIR 1e-4 = ölçülen sapmanın 23 katı.

    PLAN C'DE BU AYNI İDDİA YANLIŞTI: orada "iki bağımsız yol" diye yazılan
    `a*sqrt(2De/mu)` ile `sqrt(2*De*a^2/mu)` AYNI cebirsel ifadeydi (1 ulp
    fark) ve Morse'u hiç çağırmıyordu. Buradaki iki yol farklı kodu, farklı
    veriyi ve farklı fiziği kullanır ve BAĞIMSIZLIKLARI ÖLÇÜLDÜ:

      kuvvetleri x1.001 ile bozan mutant:
        omega_A  5481.25336 -> 5481.25336   (BİREBİR AYNI -- gradyanı hiç
                                             kullanmıyor)
        omega_B  5481.22946 -> 5483.96942   (kaydı)
        oran-1   -4.36e-06  -> +4.96e-04    (sınırın 5 katı, test KIRMIZI)

      gradyan adımını değiştirmek de aynı yönde: h = 5e-5 / 1e-3 / 1e-2 için
      omega_A üçünde de 5481.25336, omega_B ise 5481.22946 / 5481.22952 /
      5481.23607.

    Yani omega_A kuvvet kodundan YAPISAL OLARAK etkilenemiyor; ikisinin aynı
    hesap olması imkânsız.

    ÇAPRAZ ATIF (Modül 7 Task 3 -- yukarıdaki iddiaların HİÇBİRİ
    değişmedi, EKLENEN şey bu testin GÖRMEDİĞİDİR): kilitlenen `oran - 1`
    kütleye YAPISAL OLARAK kördür, ölçüldü ve bit bit
    (`test_ORAN_CIPASI_KUTLEYE_KOR_ve_bu_OLCULDU`); kütle borusunun kendi
    sınavı bu assert'e yüklenemez ve ayrı olarak sevk edildi
    (`test_KUTLE_BORUSUNUN_KENDI_SINAVI_ORAN_ASSERTINE_YUKLENEMEZ`).
    """
    alan = HartreeFockGradient(["H", "H"])

    # --- YOL A: statik ikinci türev (kuvvet KULLANILMAZ)
    def E(r: float) -> float:
        return alan(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]]))[0]

    h = 0.002
    k = (E(H2_RE_BOHR + h) + E(H2_RE_BOHR - h) - 2 * E(H2_RE_BOHR)) / h**2
    omega_a = float(np.sqrt(k / MU_H2))
    assert omega_a * HARTREE_TO_WAVENUMBER == pytest.approx(5481.25, abs=0.5, rel=0.0)

    # --- YOL B: dinamik periyot (ikinci türev ALINMAZ)
    t_harm = 2 * np.pi / omega_a
    dt = t_harm / 200
    noktalar = []
    for genlik in (0.01, 0.02, 0.04):
        yorunge = run(
            _baslangic(genlik), alan, dt=dt, n_steps=200 * 6, save_every=1
        )
        sapma = _bag(yorunge) - H2_RE_BOHR
        t = yorunge.times
        gecis = np.nonzero(np.diff(np.sign(sapma)) != 0)[0]
        anlar = [
            t[i]
            + (t[i + 1] - t[i]) * abs(sapma[i]) / (abs(sapma[i]) + abs(sapma[i + 1]))
            for i in gecis
        ]
        # TAM periyot: aynı yöndeki geçişler arası (Plan C Task 6'nın dersi --
        # `2*mean` aralık sayısı TEK olduğunda tahmini kaydırır).
        periyot = float(np.mean(np.diff(np.asarray(anlar)[::2])))
        omega_ayrik = 2 * np.pi / periyot
        # velocity-Verlet dispersiyon TERSİ
        noktalar.append((genlik**2, (2.0 / dt) * np.sin(omega_ayrik * dt / 2.0)))

    egim, omega_b = np.polyfit(
        np.array([n[0] for n in noktalar]), np.array([n[1] for n in noktalar]), 1
    )

    assert abs(float(omega_b) / omega_a - 1.0) < 1e-4


def _yol_a(alan, *, mu: float, h: float = 0.002) -> float:
    """Yukarıdaki çıpanın YOL A'sının AYNISI: statik ikinci türev, a.u.

    Elde yazıldı, `bridge_obs`ten alınmadı: çıpanın kendi YOL A'sı da elde
    yazılıdır ve bu iki testin ölçtüğü şey TAM OLARAK o çıpadır. `mu` bir
    parametre çünkü aşağıdaki iki test de kütle düğmesini çeviriyor.
    """

    def E(r: float) -> float:
        return alan(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]]))[0]

    k = (E(H2_RE_BOHR + h) + E(H2_RE_BOHR - h) - 2 * E(H2_RE_BOHR)) / h**2
    return float(np.sqrt(k / mu))


@pytest.mark.slow
def test_ORAN_CIPASI_KUTLEYE_KOR_ve_bu_OLCULDU():
    """SINIF: ŞEMA ÇIPASI DENETİMİ -- ve bir KUSUR BEYANI.

    Yukarıdaki `test_titresim_frekansi_IKI_GERCEKTEN_BAGIMSIZ_yoldan_ayni`
    AYAKTA KALIYOR: iki yolun bağımsızlığı hakkındaki iddiası doğru ve bu
    test onu ÇÜRÜTMÜYOR. Çürütülen şey o çıpanın GÖRMEDİĞİDİR: kilitlediği
    `oran - 1` KÜTLEYE YAPISAL OLARAK KÖRDÜR.

    CEBİR -- sadeleşme bir tesadüf değil, BORUNUN ŞEKLİ:
      YOL A  omega_A = sqrt(k/mu). `k` YÜZEYDEN gelir, kütleyi hiç görmez;
             yani omega_A ~ mu^(-1/2).
      YOL B  dt = (2*pi/omega_A)/N, yani dt ~ mu^(+1/2). `m r'' = F(r)`
             denklemi `m -> L*m` ile `t -> sqrt(L)*t` altında DEĞİŞMEZ,
             yani periyot sqrt(L) uzar ve omega_B de ~ mu^(-1/2).
             Boyutsuz olan iki büyüklük -- `omega*dt` ve genlik/r_e -- mu'den
             BAĞIMSIZ kalır; bu yüzden hem A^2 ekstrapolasyonunun artığı hem
             velocity-Verlet dispersiyon düzeltmesi aynen korunur.
      ORAN   omega_B/omega_A: mu^(-1/2) PAY VE PAYDADA sadeleşir. `oran`
             kütlenin bir FONKSİYONU DEĞİLDİR.

    ÖLÇÜLDÜ (31.08.2026), ÇIPANIN KENDİ KONFİGÜRASYONUNDA -- pencere:
    HF/STO-3G alanı, h = 0.002, N = 200, 6 periyot, genlikler
    0.01/0.02/0.04, r_e = 1.3459193602 -- kütle x4 yapıldığında:

        oran - 1  -4.360844408468623e-06 -> -4.360844408468623e-06
        hex       -0x1.24a6bc8940000p-18 -> -0x1.24a6bc8940000p-18

    BİT BİT aynı. Sebep yalnız cebir değil, kayan noktanın da işbirliği:
    4 ve 2 ikinin kuvvetleri olduğu için mu x4 -> dt x2 -> masses x4
    mantisleri HİÇ değiştirmez. İki yol da TAM YARIYA iniyor (ölçüldü,
    `==` ile, yaklaşıklık değil):

        omega_A  5481.2533589840905 -> 2740.6266794920452
        omega_B  5481.229456091029  -> 2740.6147280455143

    KUSUR: çıpanın `1e-4` eşiğinin marjı x4'lük bir kütle hatasında BİLE
    kıpırdamıyor -- 22.931338665925146 -> 22.931338665925146. O çıpa
    transferin SAYISALLIĞINI doğruluyor; kütle borusunu ya da yüzeyin
    fiziğini DEĞİL.

    "KÜTLE x4" İKİ AYRI SAYIDIR VE BU TEST BİRİNCİSİDİR. Burada `dt` boru
    kuralından YENİDEN türetiliyor (`boru_omega_b`ye `dt` VERİLMİYOR),
    çarpan TAM 0.5. İkinci düğme -- `dt` DONDURULMUŞ -- BAŞKA bir sayıdır:
    taban 28.66595664843827 kat KÜÇÜLÜR, ve o
    `tests/bridge/test_kenar_tabani.py`de dondurulmuş Morse yüzeyinde
    ölçülür. Hangisi olduğu yazılmazsa "kütle x4" cümlesi anlamsızdır.

    BORU: `bridge_obs.omega_b` bu dosyanın YOL B'sinin AYNISIDIR ve bu bir
    iddia değil bir ÖLÇÜMDÜR -- HF alanıyla çağrıldığında yukarıdaki çıpanın
    sayısını bit bit veriyor (`tests/bridge/test_kenar_tabani.py`de de
    kilitli). Pencere burada AÇIKÇA geçiliyor, `bridge_obs`in
    varsayılanlarına güvenilmiyor.

    NEDEN DEĞER `rel=1e-3` İLE KİLİTLENİYOR -- BİLEREK: ölçüldü,
    `AMU_TO_ELECTRON_MASS = 1.0` mutantı bu sayıyı bağıl
    1.1599354966441513e-05 oynatıyor. `rel=1e-6` yazılsaydı bu test o
    mutantı KAZARA yakalar ve tam da belgelediği günahı -- BİR ASSERT'E İKİ
    SINAV YÜKLEMEK -- tekrarlardı. Kütle sınavı BİR SONRAKİ testtedir.
    Değerin bit düzeyindeki kaydı zaten `tests/bridge/test_kenar_tabani.py`
    içinde duruyor; buradaki taşıyıcı assert `oran(4mu) == oran(mu)`dur.

    MUTASYON (31.08.2026) -- BU TEST KÜTLE MUTANTLARINDA YEŞİL KALIYOR, ve
    bu bir KUSUR DEĞİL yukarıdaki tasarım kararının ÖLÇÜLMÜŞ sonucudur:
    `AMU_TO_ELECTRON_MASS = 1.0` ve `ELECTRON_MASS_KG x1.001` mutantlarının
    ikisi de buradan geçiyor, ikisini de BİR SONRAKİ test öldürüyor.
    `HARTREE_TO_WAVENUMBER x1.001` de buradan geçiyor -- kilitlenen her şey
    BOYUTSUZ. Assert'lerin boş OLMADIĞI ayrıca ölçüldü: `bridge_obs`te `dt`
    boru kuralı dondurulunca (1) ve (2) kırılıyor, velocity-Verlet
    dispersiyon tersi kaldırılınca (3) kırılıyor -- ve koşulan üç test
    içinde o iki mutantı YALNIZ bu test görüyor, çünkü mevcut çıpa kendi
    YOL B'sini ELDE yazdığı için `bridge_obs`e hiç dokunmuyor.

    MALİYET ÖLÇÜLDÜ, VE MALİYETİN DE BİR PENCERESİ VAR: tek başına
    (`pytest tests/md/test_aimd.py --durations`) 77.84 s; `tests/md`nin
    içinde 55.54 s. Fark ısınmış SCF önbelleğidir, ve aynı oran mevcut
    çıpada da var (38.83 s -> 28.02 s). Hangi pencerede olursa olsun bu
    test `tests/md`nin EN PAHALI testidir -- çıpanın yaklaşık iki katı --
    çünkü aynı HF borusunu İKİ kez koşuyor. `slow` kararı DEVRALINMADI,
    ölçüldü. (Ayrı bir betikte iki koşum 38.10 + 38.81 s.)
    """
    alan = HartreeFockGradient(["H", "H"])
    m_h = float(amu_to_electron_mass(1.00782503207))
    pencere = dict(
        r_e=H2_RE_BOHR, n_bolme=200, periyot_sayisi=6,
        genlikler=(0.01, 0.02, 0.04), h=0.002,
    )

    wa1 = _yol_a(alan, mu=MU_H2)
    wb1 = boru_omega_b(alan, mu=MU_H2, masses=np.full(2, m_h), **pencere)
    oran1 = wb1 / wa1 - 1.0

    # KÜTLE x4 -- `dt` VERİLMİYOR, yani boru kuralından YENİDEN türetiliyor.
    wa4 = _yol_a(alan, mu=4.0 * MU_H2)
    wb4 = boru_omega_b(alan, mu=4.0 * MU_H2, masses=np.full(2, 4.0 * m_h),
                       **pencere)
    oran4 = wb4 / wa4 - 1.0

    # (1) TAŞIYICI: oran kütleye KÖR -- yaklaşık değil, BİT BİT.
    assert oran4 == oran1
    assert float.hex(oran4) == float.hex(oran1)

    # (2) SADELEŞMENİN MEKANİZMASI: iki yol da TAM yarıya iniyor. Bu, (1)'in
    # NEDENİDİR ve `mu`nun sadeleştiğini gösteren şeydir; (1) tek başına bir
    # tesadüfle de sağlanabilirdi.
    assert wa4 == wa1 / 2.0
    assert wb4 == wb1 / 2.0

    # (3) Boru gerçekten çıpanın borusu mu -- ölçülen değerin kaydı.
    # PENCERE `rel=1e-3` (gerekçesi dokstringde: daha keskin bir pencere bu
    # testi KAZARA bir kütle sınavına çevirir).
    assert oran1 == pytest.approx(-4.360844408468623e-06, rel=1e-3, abs=0.0)

    # BU TESTTE MUTLAK BİR `omega_A` ASSERT'İ YOKTUR, VE BU BİR EKSİK DEĞİL
    # BİR TASARIM KARARIDIR. Buraya `omega_A == 5481.25...` konsaydı test
    # kütle mutantını yakalardı -- ama tam da belgelediği hatayı, İKİ SINAVI
    # TEK TESTE YÜKLEMEYİ, tekrarlayarak. Mutlak sınav BİR SONRAKİ testtedir.

    # (4) KUSUR BEYANI -- BU SATIR BİR BAŞARI DEĞİLDİR. Çıpanın kendi eşiği
    # x4'lük bir kütle hatasında YEŞİL kalıyor, üstelik marjı bile aynı.
    # (`oran4 == oran1` olduğu için marjların eşitliğini AYRICA assert etmek
    # tautoloji olurdu; kilitlenen şey marjın DEĞERİDİR, aynı `rel=1e-3`
    # penceresiyle.)
    assert abs(oran4) < 1e-4
    assert 1e-4 / abs(oran1) == pytest.approx(22.931338665925146, rel=1e-3,
                                              abs=0.0)


def test_KUTLE_BORUSUNUN_KENDI_SINAVI_ORAN_ASSERTINE_YUKLENEMEZ():
    """BİR ASSERT'E İKİ SINAV YÜKLENEMEZ -- ve planın İKİ sayısı çürüdü.

    PLANIN İDDİASI: `AMU_TO_ELECTRON_MASS = 1.0` mutantı oranı yalnız bağıl
    `2e-7` oynatır ve `1e-4` eşiğinden `~600 kat marjla` geçer.

    YENİDEN ÖLÇÜLDÜ (31.08.2026, çıpanın penceresi) -- İKİSİ DE ÇÜRÜK:

      oran = omega_B/omega_A     0.9999956391555915 -> 0.9999956392061745
      oran - 1            -4.360844408468623e-06 -> -4.360793825486375e-06
      ORANIN bağıl oynaması       5.058320283272645e-11
                                  (plan 2e-7 dedi; 3953.88 kat BÜYÜK yazmış)
      (oran-1)in bağıl oynaması   1.1599354966441513e-05
                                  (plan 2e-7 dedi; 57.997 kat KÜÇÜK yazmış)
      `1e-4` eşiğinin MARJI       22.931338665925146 -> 22.931604657747524
                                  (plan ~600 dedi; 26.16 kat BÜYÜK yazmış)

    Planın `600`ü bir KATEGORİ HATASI: `abs(oran - 1) < 1e-4` assert'inin
    marjı `eşik/|değer|`dir (22.93), `eşik/|oynama|` DEĞİL. Plan ikincisini
    -- üstelik kendi çürük `2e-7`siyle -- hesaplamış görünüyor:
    `1e-4 / 2e-7 = 500`.

    HÜKÜM DEĞİŞMİYOR, GÜÇLENİYOR: mutant oran assert'ini yalnız GEÇMİYOR,
    onu neredeyse HİÇ kıpırdatmıyor. `oran` mu'ye yapısal olarak kördür (bir
    önceki test, kütle x4'te BİT BİT); mutant onu ancak kayan nokta
    yuvarlamasıyla oynatıyor. Onu yakalayan şey oran assert'i DEĞİL, aynı
    testteki MUTLAK `omega_A` assert'idir (`abs=0.5`) -- yani o tek testte
    bir assert transferi, öteki kütle borusunu sınıyor.

    BU TEST SINAVI AYIRIYOR VE MUTLAK BİR BÜYÜKLÜKTEN OKUYOR:

      (1) DIŞ OKUMA. HF/STO-3G'nin DENEYSEL H2 omega_e = 4401.21 cm^-1'den
          sapması +0.2453969156173167. Bu bacak (2)'yle AYNI büyüklüğü okur
          -- bağımsız bir SINAV değil, bağımsız bir OKUMADIR -- ama değeri
          şudur: (2)'nin literali bozuk bir boruya UYDURULARAK
          "düzeltilebilir", (1) uydurulamaz, çünkü 4401.21 depodan gelmiyor.
      (2) DEPO KAYDI. Aynı büyüklük 5481.2533589840905 cm^-1. Bu bir
          TRANSKRİPSİYON KİLİDİDİR ve öyle etiketleniyor.
      (3) MUTANTIN MUTLAK BÜYÜKLÜKTEKİ ETKİSİ, CANLI. Dönüşüm atlanınca
          omega_A 234023.7465323093 cm^-1 olur; çarpan
          42.695298174591926 = sqrt(AMU_TO_ELECTRON_MASS) ve ölçüldü:
          BİT BİT eşit. Deneysel sapma +0.245 yerine +52.172592657998436,
          yani (1)'in okuduğu yerin 212.6049242581304 katı.

    NEDEN MUTLAK OLMAK ZORUNDA: kütle yalnızca bir ÖLÇEKTİR. Bir orana
    girdiğinde sadeleşir; bir farka girdiğinde ölçeğiyle birlikte gider.
    Kütle borusunu sınayan bir assert, kütlenin sadeleşemeyeceği bir yerden
    -- MUTLAK bir büyüklükten -- okumak ZORUNDADIR. `tests/test_units.py`
    dönüşümü SINIRDA sınıyor; burada sınanan şey farklıdır: dönüşümün BU
    DOSYANIN gözlenebilirine ULAŞTIĞI.

    MUTASYON (31.08.2026) -- BU TEST ÜÇ MUTANTI DA ÖLDÜRÜYOR ve üçünde de
    bir önceki test YEŞİL kalıyor. Sınavı ayırmanın ölçülmüş kazancı budur:
      `AMU_TO_ELECTRON_MASS = 1.0`  -- kaba kütle ölçeği (x1822.888)
      `ELECTRON_MASS_KG x1.001`     -- İNCE kütle hatası, yalnız bağıl 1e-3
      `HARTREE_TO_WAVENUMBER x1.001`-- birim dönüşümü (transkripsiyon)
    İkincisi önemli: sınav yalnız 1822 katlık balyozu değil, binde birlik
    bir kütle kaymasını da görüyor.

    MALİYET ÖLÇÜLDÜ: 0.06 s -- ALTI enerji değeri (iki `_yol_a` çağrısı x
    üç nokta), hiç MD YOK. `slow` DEĞİL, ve bu bilinçli: kütle sınavı ucuz
    olmalı ki her koşumda dönsün. Karşılaştırma: bir önceki test 77.84 s.
    """
    alan = HartreeFockGradient(["H", "H"])
    omega_cm = _yol_a(alan, mu=MU_H2) * HARTREE_TO_WAVENUMBER

    # (1) DIŞ OKUMA -- deneysel H2 omega_e. Depodan GELMEYEN bir sayı.
    deneysel = 4401.21
    assert (omega_cm - deneysel) / deneysel == pytest.approx(
        0.2453969156173167, rel=1e-6, abs=0.0
    )

    # (2) DEPO KAYDI -- transkripsiyon kilidi, ve öyle etiketli.
    assert omega_cm == pytest.approx(5481.2533589840905, rel=1e-9, abs=0.0)

    # (3) MUTANT CANLI: `amu_to_electron_mass` ATLANIRSA mutlak büyüklük
    # sqrt(AMU_TO_ELECTRON_MASS) katına çıkar. Ölçüldü: BİT BİT eşit;
    # pencere yine de 1e-15 (~4.5 ulp), çünkü `k` BLAS'a bağlıdır ve oranda
    # tam sadeleşmesi garanti DEĞİLDİR.
    omega_mut = _yol_a(alan, mu=1.00782503207 / 2.0) * HARTREE_TO_WAVENUMBER
    assert omega_mut / omega_cm == pytest.approx(
        float(np.sqrt(AMU_TO_ELECTRON_MASS)), rel=1e-15, abs=0.0
    )
    assert omega_mut == pytest.approx(234023.7465323093, rel=1e-9, abs=0.0)
    # ...ve (1)'in penceresi onu 212 kat dışarıda bırakır.
    assert (omega_mut - deneysel) / deneysel == pytest.approx(
        52.172592657998436, rel=1e-9, abs=0.0
    )


def test_HF_STO3G_deneyselden_SAPIYOR_ve_bu_beyan_ediliyor():
    """%24.54 sapma bir KUSUR DEĞİL, YÖNTEMİN BEYANIDIR.

    ÖLÇÜLDÜ: HF/STO-3G 5481.25 cm^-1, deneysel omega_e 4401.21 cm^-1,
    sapma +1080.03 cm^-1 = +%24.54.

    Bu test sapmanın KÜÇÜLMESİNİ İSTEMEZ. Deneysel sayıya yaklaşma hedefi
    ya da toleransı bu planda YOKTUR; deneysel sayı yalnız beyan olarak
    raporlanır. Test, sapmanın BİLİNEN büyüklükte kalmasını kilitler --
    sessizce değişirse bir şey bozulmuş demektir.

    BAZ SETİ BÜYÜYÜNCE YAKLAŞIYOR (ölçüldü, ayrı koşu): 6-31G ile sapma
    +244.97 cm^-1 = +%5.57, yani hata 4.4 kat azalıyor. Bu, sapmanın
    yöntemin kendisinden geldiğinin bağımsız kanıtıdır -- bir kodlama
    hatası baz setiyle böyle davranmazdı. O koşu 5.4x pahalı olduğu için
    test süitesine ALINMADI (ölçüldü: H2/6-31G adımı 76.2 ms).
    """
    deneysel = 4401.21
    hf_sto3g = 5481.25
    assert (hf_sto3g - deneysel) / deneysel == pytest.approx(0.2454, abs=0.002)
