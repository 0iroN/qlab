"""G7 (09.10.2026) -- md hızlı şeridinin kör kaldığı çekirdek fizik için ucuz çıpalar.

Envanter G7: "-m not slow" şeridi modülün çekirdek fiziğini bozan bir mutantı
öldürebilmeli. md'de hızlı şerit çıpa bakımından ZENGİNDİR (velocity-Verlet'in
analitik periyodu/salınımı, Morse/LJ/harmonik analitik noktaları, Morse'un
omega(E) kapalı formu, Langevin sürtünme sönümü, NH zaman-tersinirliği, TB
O(dt^2), HF Richardson) -- ama G7 ölçümü (`.superpowers/sdd/g7-olcum/md/`)
ÜÇ boşluk buldu:

* Nosé-Hoover'ın `n_dof` kanalı ve termostat yarım-adımı: iki mutantı da
  YALNIZ yavaş şeridin ergodiklik testleri (~200 s, yük altında) görüyordu.
* Langevin'in gürültü genliğindeki KÜTLE bağımlılığı: hızlı ve yavaş
  şeridin bütün termostat testleri TEK kütle kullanıyor; sigma'yı ortalama
  kütleyle yazan mutant iki şeritten de geçiyordu.
* velocity-Verlet'in parçacık başına kütlesi: ivmeyi ortalama kütleyle alan
  mutantı hızlı şeritte YALNIZ bir regresyon kilidi (D2'nin kütle
  safsızlığı panel sayısı, "kendi önceki çıktısına eşit") öldürüyordu;
  fizik çıpası yoktu. Yavaş şeridin bütün MD'si eşit kütleli.

Denetim (09.10.2026, `.superpowers/sdd/g7-olcum/denetim/md/`) DÖRDÜNCÜ bir
boşluk buldu: termostatların KENDİ tekmelerinin kütle kanalı. Üçüncü
maddenin termostat ikizleri -- Langevin'in B adımları (Y6) ve NH'nin iki
tekmesi (Y7) ortalama kütleyle -- ilk üç testten, md hızlı şeridinin
tamamından (257 test) ve 4 yavaş termostat testinden sağ çıkıyordu: iki
termostat çıpası F = 0 kullandığı için tekmeleri yapısal olarak göremez.

Buradaki beş test bu boşlukları BAĞIMSIZ analitik referanslarla kapatır:
serbest parçacık NH'sinin kapalı çözümü, eşbölüşüm teoremi, iki cisim
problemi (indirgenmiş kütle + momentum korunumu), NH'de dP/dt = -zeta P
(P0 = 0 -> P = 0 tam) ve Langevin'de kütle merkezi momentumunun kapalı-form
Ornstein-Uhlenbeck varyansı. Hiçbiri `slow` değildir.

KÖR NOKTA (dürüst olumsuz, ölçüldü): bu dosyanın bütün gözlenebilirleri
boyutsuz ORANDIR, kütle ÖLÇEĞİNE ve k_B'ye kördür (aşağıda M_H yorumuna
bakınız). Ölçek sınavları başka testlerdedir.

SÜRE (ölçüldü, 09.10.2026, bu makine, başka ajanlar aynı anda test koşarken
-- yük altında; dört koşu): `--durations=0` ile dosya sırasıyla
0.04-0.07 / 0.02 / 0.05-0.07 / 0.02-0.03 / 0.02-0.03 s, dosya 0.57-0.75 s
(toplama dahil). İlk sürümün yük altı ölçümü 0.10 / 0.09 / 0.04 s, 0.94 s
idi. Mutasyon ölçümü ve logları:
`.superpowers/sdd/g7-olcum/md/` (ilk üç test) ve
`.superpowers/sdd/g7-olcum/duzeltme/md/` (denetim düzeltmesi, son iki test).
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.md.forcefields import Morse
from qlab.engine.md.integrator import run
from qlab.engine.md.state import MDState
from qlab.engine.md.thermostats import Langevin, NoseHoover
from qlab.units import KB_HARTREE_PER_KELVIN, amu_to_electron_mass

#: Kütleler kütle borusundan (`amu_to_electron_mass`) geçer -- gerçekçi ORAN
#: (1H : 12C : 16O) için. Bu bir ÖLÇEK koruması DEĞİLDİR: dosyanın bütün
#: gözlenebilirleri (<2K>/(n k_B T), <m v^2>/k_B T, T/T_an -- mu aynı borudan
#: gelir ve sadeleşir --, |P|/(mu v_bağıl), |P|/sigma_n) kütle ölçeğine
#: kördür. ÖLÇÜLDÜ (09.10.2026, denetim düzeltmesi, mut.py;
#: `.superpowers/sdd/g7-olcum/duzeltme/md/mut_kapsam_disi*.log`):
#: AMU_TO_ELECTRON_MASS = 1.0 mutantı bu dosyanın beş testini de GEÇER;
#: k_B'yi %1 kaydıran mutant da geçer. Hızlı şeritte AMU = 1.0'ı
#: test_aimd.py::test_KUTLE_BORUSUNUN_KENDI_SINAVI_ORAN_ASSERTINE_YUKLENEMEZ,
#: ikisini de test_d2_salinim.py::test_T1_IRI_dt_... öldürür (ikincisinin
#: fizik çıpası mı regresyon kilidi mi olduğu burada incelenmedi).
M_H = amu_to_electron_mass(1.00782503207)
M_C = amu_to_electron_mass(12.0)
M_O = amu_to_electron_mass(15.99491461957)

#: H2 Morse parametreleri (test_dissociation.py / test_forcefields.py ile
#: aynı; burada MODEL yüzeyi, H2 fiziği iddiası yok) ve E/D_e.
MORSE_DE, MORSE_RE, MORSE_A = 0.174434333, 1.401118538, 1.028997701
E_ORAN = 0.10


class _KuvvetYok:
    """Serbest parçacık: V = 0, F = 0. Kapalı çözümlerin sahnesi."""

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        return 0.0, np.zeros_like(positions)


def _hc_morse() -> tuple[MDState, Morse, float, float]:
    """H + C iki cisim sahnesi: Morse yüzeyi, iç dönüş noktasından DURGUN (P0 = 0).

    Döner (durum, alan, mu, T_an). mu = m1 m2/(m1 + m2); T_an = 2 pi/omega(E),
    omega(E) = a sqrt(2 D_e/mu) sqrt(1 - E/D_e) -- klasik Morse'un kapalı
    formu (test_dissociation.py'deki formun eşit olmayan kütleli hâli).
    İç dönüş noktası r0 = r_e - ln(1 + sqrt(E/D_e))/a, yani V(r0) = E.
    """
    kutle = np.array([M_H, M_C])
    mu = M_H * M_C / (M_H + M_C)
    t_an = 2.0 * np.pi / (MORSE_A * np.sqrt(2.0 * MORSE_DE / mu) * np.sqrt(1.0 - E_ORAN))
    r0 = MORSE_RE - np.log(1.0 + np.sqrt(E_ORAN)) / MORSE_A
    durum = MDState(
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r0]]),
        velocities=np.zeros((2, 3)),
        masses=kutle,
    )
    return durum, Morse(d_e=MORSE_DE, r_e=MORSE_RE, a=MORSE_A), mu, t_an


def _yuvarlama_payi(n_adim: int) -> float:
    """Momentum korunumunun YUVARLAMA bütçesi, |P| / (mu max|v_bağıl|) biriminde.

    P = 0 iken |m_i v_i| = mu |v_bağıl|. Bir adımın birkaç işlemi (iki tekme;
    NH'de ayrıca iki ölçekleme) P'ye en çok ~10 eps göreli hata ekler. İptal
    VARSAYMAYAN doğrusal birikim n_adim * 10 eps; 10 kat güvenlikle
    100 * n_adim * eps (800 adımda 1.78e-11, 2400 adımda 5.33e-11).
    Bu bir ölçüm sabiti DEĞİL, kayan nokta bütçesidir: ölçülen değerler
    bunun 1e-3'ünün altında, kütle kanalı mutantları O(1) (aşağıda) --
    ayrım >= 1e10. Böylece meşru yeniden yazımlar (ters kütle çarpımı,
    Taylor konumu, farklı toplama sırası) yanlış kırmızı vermez.
    """
    return 100.0 * n_adim * float(np.finfo(float).eps)


def test_G7_NH_SERBEST_PARCACIK_kapali_cozum_KE_ortalamasi_ve_periyot():
    """Serbest parçacıklarda NH denklemleri KAPALI bir 2B sisteme iner.

    F = 0 iken K = toplam kinetik enerji ve zeta için (Q = k_B T tau^2,
    n = n_dof):  dK/dt = -2 zeta K ,  dzeta/dt = (2K - n k_B T) / Q.
    İki analitik sonuç (yalnız bu denklemlerden, koddan değil):
      (1) Tam periyot boyunca zeta başa döner, yani zaman ortalaması
          <2K> = n k_B T TAM OLARAK (eşbölüşüm, n_dof ile).
      (2) Küçük genlikte doğrusallaştırma: omega^2 = 2 n k_B T / Q
          = 2n / tau^2, yani periyot T = 2 pi tau / sqrt(2n).
    Pencere: N = 64 parçacık, YALNIZ x'te hareket (y, z tam sıfır) ve
    n_dof = 64 -- bu yüzden `n_dof` gerçekten okunmak ZORUNDA (v.size = 192).
    tau = 100 a.u., T = 300 K, K0 = 1.02 * n k_B T / 2, zeta0 = 0,
    dt = T_an/200, 10 periyot (2000 adım). Hızlar RNG'siz: K(t) yalnız K0'a
    bağlıdır (serbest parçacıkta bütün hızlar aynı çarpanla ölçeklenir).

    ÖLÇÜLDÜ (09.10.2026, bu makine): <2K>/(n k_B T) - 1 = 4.94e-07;
    T_ölçülen/T_an - 1 = -2.47e-05. Sapmanın kaynakları: pencere tam olarak
    10 ANALİTİK periyot, gerçek periyot 2.5e-05 kısa (ortalamada artık) ve
    periyottaki O((omega dt)^2) ayrıklaştırma + O(eps^2) genlik terimi
    (eps = 0.01/0.05/0.2'de -3.7e-05/+6.0e-05/+1.43e-03; n_per = 100/400'de
    -1.48e-04/+6.2e-06).
    TOLERANSLAR: (1) 5e-06 = ölçülenin ~10 katı; (2) 2.5e-04 = ölçülenin
    ~10 katı. Mutantların ayrımı bunun çok üstündedir (aşağıda).

    KIRMA (ölçüldü, 09.10.2026; mut.py, yalnız hızlı şerit; ikisini de
    önceden YALNIZ yavaş şeridin ergodiklik testi öldürüyordu):
      `_serbestlik` n_dof'u yok sayıp v.size döndürür: (1) +2.025 (sabit
        nokta 3 n k_B T/2'ye kayar), (2) -0.402 -> İKİ assert de kırmızı.
      termostat yarım-adımları h yerine dt ile: (1) +2.96e-06 -- GEÇER (bu
        mutant serbest parçacıkta zamanın yeniden ölçeklenmesidir, sabit
        nokta ve tam-periyot ortalaması değişmez; dürüst olumsuz), (2)
        -0.50007 (periyot TAM yarıya) -> kırmızı. Mutantı (2) öldürür.
    """
    n, sicaklik, tau, eps = 64, 300.0, 100.0, 0.02
    kt = KB_HARTREE_PER_KELVIN * sicaklik
    hiz = np.zeros((n, 3))
    hiz[:, 0] = np.linspace(-1.0, 1.0, n) + 0.3
    ke = 0.5 * M_H * float((hiz**2).sum())
    hiz *= np.sqrt((1.0 + eps) * n * kt / 2.0 / ke)
    durum = MDState(positions=np.zeros((n, 3)), velocities=hiz, masses=np.full(n, M_H))
    termostat = NoseHoover(target_temperature_kelvin=sicaklik, relaxation_time=tau, n_dof=n)

    t_an = 2.0 * np.pi * tau / np.sqrt(2.0 * n)
    dt = t_an / 200
    alan = _KuvvetYok()
    _, kuvvet = alan(durum.positions)
    iki_k, zeta = [], []
    for _ in range(200 * 10):
        durum, _, kuvvet = termostat.step(durum, kuvvet, alan, dt)
        iki_k.append(float((durum.masses[:, None] * durum.velocities**2).sum()) / (n * kt))
        zeta.append(float(termostat.zeta[0]))
    zeta_d = np.asarray(zeta)
    t = dt * np.arange(1, len(zeta_d) + 1)
    gecis = np.nonzero(np.diff(np.sign(zeta_d)) != 0)[0]
    anlar = np.array([
        t[i] + (t[i + 1] - t[i]) * abs(zeta_d[i]) / (abs(zeta_d[i]) + abs(zeta_d[i + 1]))
        for i in gecis
    ])
    # TAM periyot: aynı yöndeki geçişler arası (yarım periyotlar asimetrik olabilir).
    periyot = float(np.mean(np.diff(anlar[::2])))

    assert abs(float(np.mean(iki_k)) - 1.0) < 5e-06, float(np.mean(iki_k))
    assert abs(periyot / t_an - 1.0) < 2.5e-04, periyot / t_an


def test_G7_LANGEVIN_ESBOLUSUM_IKI_KUTLE_her_tur_kT():
    """Eşbölüşüm teoremi: her ikinci dereceden serbestlik derecesi
    <m v_a^2> = k_B T taşır -- KÜTLEDEN BAĞIMSIZ.

    Serbest parçacıkta BAOAB'ın B ve A alt adımları hızı değiştirmez; O adımı
    Ornstein-Uhlenbeck'in TAM geçişidir. Sıfır hızdan n adım sonra
    Var(v_a) = (1 - c1^(2n)) k_B T / m, c1 = exp(-gamma dt). Pencere:
    gamma dt = 0.5, n = 40 -> c1^80 = e^-40 ~ 4e-18, yani tam denge.
    4000 H + 4000 O (16x kütle oranı), T = 300 K, dt = 10 a.u., tohum 3;
    yalnız SON adımın anlık görüntüsü (replikalar bağımsız).
    KAPSAM: F = 0 olduğu için bu test B adımlarının kütle kanalını GÖREMEZ;
    onu test_G7_LANGEVIN_ESIT_OLMAYAN_KUTLE_TEKMESI... sınar.

    ÖLÇÜLDÜ (09.10.2026): <m v^2>/k_B T = 0.99471 (H), 0.98615 (O).
    Tohum saçılımı (tohum 1..8): H 0.9815..1.0079, O 0.9716..1.0028; en büyük
    sapma %2.84. İstatistiksel taban sqrt(2/(3R)) = %1.29.
    TOLERANS %5 = 3.9 sigma; bu tohumun en kötüsünün (%1.39) 3.6 katı, 8
    tohumun en kötüsünün 1.76 katı. İstatistiksel bir çıpadır; marjı
    replika sayısı belirler (R'yi artırmak tabanı sqrt ile daraltır).

    KIRMA (ölçüldü, 09.10.2026; mut.py, yalnız hızlı şerit): O adımında
    sigma = sqrt(k_B T / <m>) (ortalama kütle) yazan mutant H 0.1179, O 1.8554
    verir (beklenen m/<m> = 0.118 ve 1.882) -> iki assert de kırmızı. Bu
    mutantı ÖNCEDEN HİÇBİR test görmüyordu: hızlı ve yavaş şeridin bütün
    termostat testleri (MB dahil) tek kütleli; iki şeritte de 0 kırmızı.
    """
    r, sicaklik, dt = 4000, 300.0, 10.0
    kt = KB_HARTREE_PER_KELVIN * sicaklik
    kutle = np.concatenate([np.full(r, M_H), np.full(r, M_O)])
    durum = MDState(positions=np.zeros((2 * r, 3)), velocities=np.zeros((2 * r, 3)), masses=kutle)
    termostat = Langevin(target_temperature_kelvin=sicaklik, friction=0.5 / dt, seed=3)
    alan = _KuvvetYok()
    _, kuvvet = alan(durum.positions)
    for _ in range(40):
        durum, _, kuvvet = termostat.step(durum, kuvvet, alan, dt)
    mv2 = durum.masses[:, None] * durum.velocities**2 / kt
    assert float(mv2[:r].mean()) == pytest.approx(1.0, rel=0.05, abs=0.0), float(mv2[:r].mean())
    assert float(mv2[r:].mean()) == pytest.approx(1.0, rel=0.05, abs=0.0), float(mv2[r:].mean())


def test_G7_VV_ESIT_OLMAYAN_KUTLE_momentum_KORUNUR_ve_INDIRGENMIS_KUTLE_periyodu():
    """İki cisim problemi, EŞİT OLMAYAN kütlelerle (H + C, Morse yüzeyi).

    İki bağımsız analitik referans:
      (1) Yalıtılmış sistemde toplam momentum KORUNUR: Morse kuvveti eşit
          ve zıttır (Newton III), integratör her parçacığı KENDİ kütlesiyle
          ilerletirse sum m_i v_i = 0 makine hassasiyetinde kalır.
      (2) Bağıl hareket indirgenmiş kütleyle olur, mu = m1 m2/(m1 + m2);
          klasik Morse frekansı omega(E) = a sqrt(2 D_e/mu) sqrt(1 - E/D_e)
          (_hc_morse). mu m1 <-> m2 takasında simetriktir: kütle TAKASI
          mutantını (2) göremez, yalnız (1) görür -- (1) bu yüzden gerekli.
    Pencere: _hc_morse (m1 = 1H, m2 = 12C, E/D_e = 0.10, iç dönüş
    noktasından durgun), dt = T_an/400, 6 periyot (2400 adım).

    ÖLÇÜLDÜ (09.10.2026): max|P| / (mu * max|v_bağıl|) = 1.85e-15
    (n_per 200/800'de 2.31e-15/4.27e-15 -- yuvarlama birikimi); kütle
    merkezi kayması 3.6e-15 bohr; T_ölçülen/T_an - 1 = -1.30e-05
    (n_per 200/800'de -5.2e-05/-3.3e-06, O(dt^2)). Meşru eşdeğer yeniden
    yazımlarda (denetim düzeltmesi, ölçüldü): ivme = F * (1/m) ile (1)
    7.98e-15, konum x + v dt + a dt^2/2 ile (1) 5.20e-15.
    TOLERANSLAR: (1) _yuvarlama_payi(2400) = 5.33e-11 -- kayan nokta
    bütçesi; eski 5e-14 eşiği meşru yeniden yazımın bile payın 1/6'sını
    tükettiği bir gürültü sınırıydı (denetim bulgusu), bu yüzden
    gevşetildi; (2) 1e-04 = ölçülenin 7.7 katı.

    KIRMA (ölçüldü, 09.10.2026; mut.py, yalnız hızlı şerit): ivmeyi iki
    tekmede de ORTALAMA kütleyle alan velocity-Verlet mutantı (1) 5.91
    (kütle merkezi 0.269 bohr kayar), (2) +0.870 verir -> iki assert de
    kırmızı. Önceden hızlı şeritte onu YALNIZ bir regresyon kilidi
    (`test_d2_salinim.py`nin kütle safsızlığı panel sayısı) öldürüyordu;
    yavaş şerit (bütün MD'si eşit kütleli) hiç görmüyordu. Denetimin
    mutantları: yalnız ikinci tekme ortalama kütleyle (1) 1.33, (2) +0.247;
    iki tekmede kütle takası (1) 11.8, (2) -1.30e-05 (DEĞİŞMEZ, mu simetrik).
    """
    durum, alan, mu, t_an = _hc_morse()
    n_adim = 400 * 6
    yorunge = run(durum, alan, dt=t_an / 400, n_steps=n_adim, save_every=1)

    momentum = (durum.masses[None, :, None] * yorunge.velocities).sum(axis=1)
    v_bagil = np.linalg.norm(yorunge.velocities[:, 1] - yorunge.velocities[:, 0], axis=1)
    oran = float(np.abs(momentum).max()) / (mu * float(v_bagil.max()))
    assert oran < _yuvarlama_payi(n_adim), oran

    sapma = np.linalg.norm(yorunge.positions[:, 1] - yorunge.positions[:, 0], axis=1) - MORSE_RE
    t = yorunge.times
    gecis = np.nonzero(np.diff(np.sign(sapma)) != 0)[0]
    anlar = np.array([
        t[i] + (t[i + 1] - t[i]) * abs(sapma[i]) / (abs(sapma[i]) + abs(sapma[i + 1]))
        for i in gecis
    ])
    # TAM periyot: Morse asimetrik, yarım periyotlar eşit değil (test_dissociation dersi).
    periyot = float(np.mean(np.diff(anlar[::2])))
    assert abs(periyot / t_an - 1.0) < 1e-04, periyot / t_an


def test_G7_NH_ESIT_OLMAYAN_KUTLE_TEKMESI_momentum_TAM_SIFIR_kalir():
    """Nosé-Hoover'ın KENDİ tekmeleri, eşit olmayan kütlelerle (H + C, Morse).

    NH denklemleri m_i dv_i/dt = F_i - zeta m_i v_i; i üzerinden toplam
    dP/dt = sum F_i - zeta P = -zeta P (Newton III). P0 = 0 ise P(t) = 0
    TAM OLARAK, zeta ne olursa olsun. Ayrık adım da bunu korur: termostat
    yarım-adımları bütün hızları AYNI çarpanla ölçekler (P -> s P = 0),
    tekmeler P'ye h sum F_i = 0 ekler -- ama YALNIZ her parçacık KENDİ
    kütlesiyle tekmelenirse. Ortalama kütleyle tekme P'ye
    h F_1 (m_1 - m_2)/<m> = O(1) ekler.
    Pencere: _hc_morse (P0 = 0), T = 300 K, n_dof = 1 (tek titreşim
    serbestliği), tau = 5 T_an, dt = T_an/400, 2 periyot (800 adım).
    Termostat ETKİN, sahne VV'ye dejenere DEĞİL: E/k_B T = 18.4 (sistem
    hedeften sıcak), NH 2 periyotta enerjinin üçte ikisini alır.

    ÖLÇÜLDÜ (09.10.2026, denetim düzeltmesi): max|P| / (mu max|v_bağıl|)
    = 3.84e-15; E_son/E_0 = 0.342. tau = 0.5/1/2/50 T_an'da oran
    7.6e-16/9.1e-16/1.3e-15/2.3e-15 (E_son/E_0 0.0000/0.0001/0.016/0.987):
    termostat gücünden bağımsız, yuvarlama düzeyinde.
    TOLERANS: _yuvarlama_payi(800) = 1.78e-11 (kayan nokta bütçesi, ölçüm
    sabiti değil).

    KIRMA (ölçüldü, 09.10.2026; mut.py + sonda, yalnız hızlı şerit; logları
    `.superpowers/sdd/g7-olcum/duzeltme/md/`): NH'nin iki tekmesi ortalama
    kütleyle (Y7) 5.91; yalnız ikinci tekme ortalama kütleyle 1.30; iki
    tekmede kütle takası 11.8 -> kırmızı. Y7'yi bu test eklenmeden ÖNCE
    md hızlı şeridinin tamamı (257 test) ve 4 yavaş termostat testi
    görmüyordu (denetim: mut_kick_hizli.log, mut_kick_yavas.log).
    KAPSAM: yalnız tekmelerin kütle kanalı. n_dof ve termostat yarım-adımı
    test_G7_NH_SERBEST_PARCACIK...'ın işidir (P her zeta için 0 kalır).
    """
    durum, alan, mu, t_an = _hc_morse()
    n_adim = 400 * 2
    termostat = NoseHoover(target_temperature_kelvin=300.0, relaxation_time=5.0 * t_an, n_dof=1)
    _, kuvvet = alan(durum.positions)
    p_max = v_max = 0.0
    for _ in range(n_adim):
        durum, _, kuvvet = termostat.step(durum, kuvvet, alan, t_an / 400)
        p_max = max(p_max, float(np.abs((durum.masses[:, None] * durum.velocities).sum(axis=0)).max()))
        v_max = max(v_max, float(np.linalg.norm(durum.velocities[1] - durum.velocities[0])))
    oran = p_max / (mu * v_max)
    assert oran < _yuvarlama_payi(n_adim), oran


def test_G7_LANGEVIN_ESIT_OLMAYAN_KUTLE_TEKMESI_kutle_merkezi_OU_kapali_formu():
    """Langevin'in KENDİ B adımları, eşit olmayan kütlelerle (H + C, Morse).

    BAOAB'da kütle merkezi momentumu P = sum m_i v_i iç kuvvetlerden
    BAĞIMSIZ bir Ornstein-Uhlenbeck sürecidir: B adımları P'ye
    (dt/2) sum F_i = 0 ekler (Newton III -- YALNIZ her parçacık KENDİ
    kütlesiyle tekmelenirse), A adımları hızı değiştirmez, O adımı
    P -> c1 P + c2 sqrt(k_B T) sum sqrt(m_i) xi_i uygular. P0 = 0'dan n adım
    sonra her bileşen KAPALI FORMLA N(0, sigma_n^2):
    sigma_n^2 = (1 - c1^(2n)) k_B T M, M = m1 + m2 (test 2'deki OU
    geçişinin kütle merkezi hâli).
    Pencere: _hc_morse, T = 300 K, gamma dt = 1e-12 (c1^n = 1 - 8e-10:
    pratikte rastgele yürüyüş), dt = T_an/400, 800 adım, tohum 1. Gürültü
    titreşimin yanında küçüktür: sqrt(2 mu E) / sigma_n = 7.69 / 1.90e-04
    = 4.05e4, yani kütle kanalı hatası gürültüden 4 mertebe yukarıda görünür.
    Assert: bütün yolda max_k |P_k| < 6 sigma_n (üç bileşen). Lévy
    eşitsizliği (simetrik bağımsız artımlar): P(max_k |S_k| >= a)
    <= 2 P(|S_n| >= a) -> üç bileşende yanlış alarm <= 12 Q(6) = 1.2e-08
    (numpy RNG akışı değişse bile).

    ÖLÇÜLDÜ (09.10.2026, denetim düzeltmesi; dogrula_ou.log): tohum 1'de
    max|P|/sigma_n = 1.78. Kapalı formun kendisi, tohum 1..400:
    P_son/sigma_n'nin std'si 0.970/0.949/0.990 (taban 1/sqrt(800) = 0.035),
    max|P|/sigma_n medyan 1.59, en büyük 3.71; >= 4 olan 0/400 (Lévy
    sınırı 3.8e-04). TOLERANS 6 sigma = 400 tohumun en kötüsünün 1.6 katı.

    KIRMA (ölçüldü, 09.10.2026; mut.py + sonda, yalnız hızlı şerit; logları
    `.superpowers/sdd/g7-olcum/duzeltme/md/`): B adımları ortalama kütleyle
    (Y6) max|P|/sigma_n = 1.28e5; yalnız ilk B ortalama kütleyle 4.33e4;
    B'lerde kütle takası 4.79e5 -> kırmızı. Y6'yı bu test eklenmeden ÖNCE
    md hızlı şeridinin tamamı ve 4 yavaş termostat testi görmüyordu
    (denetim: mut_kick_hizli.log, mut_kick_yavas.log).
    KAPSAM (dürüst olumsuz): yalnız B adımlarının kütle kanalı. O adımının
    kütle kanalı (sigma = sqrt(k_B T/<m>)) bu tek yolda GÖRÜNMEZ: kapalı
    formda sigma_n'yi yalnız sqrt(2 (m1^2 + m2^2))/(m1 + m2) = 1.31 kat
    büyütür; ÖLÇÜLDÜ 2.36 < 6, GEÇER. Onu test_G7_LANGEVIN_ESBOLUSUM... sınar.
    """
    durum, alan, _, t_an = _hc_morse()
    sicaklik, dt, gamma_dt, n_adim = 300.0, t_an / 400, 1e-12, 400 * 2
    termostat = Langevin(target_temperature_kelvin=sicaklik, friction=gamma_dt / dt, seed=1)
    # Kapalı form; expm1, 1 - exp(-1.6e-09)'daki iptali önler.
    kt = KB_HARTREE_PER_KELVIN * sicaklik
    sigma_n = float(np.sqrt(-np.expm1(-2.0 * gamma_dt * n_adim) * kt * durum.masses.sum()))
    _, kuvvet = alan(durum.positions)
    p_max = 0.0
    for _ in range(n_adim):
        durum, _, kuvvet = termostat.step(durum, kuvvet, alan, dt)
        p_max = max(p_max, float(np.abs((durum.masses[:, None] * durum.velocities).sum(axis=0)).max()))
    assert p_max / sigma_n < 6.0, p_max / sigma_n
