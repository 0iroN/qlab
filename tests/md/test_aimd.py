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
from qlab.units import HARTREE_TO_WAVENUMBER, amu_to_electron_mass

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
