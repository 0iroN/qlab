"""MDScenario -> (durum, kuvvet alanı, termostat) ve Provenance."""

from __future__ import annotations

import numpy as np

from qlab.engine.md.forcefields import (
    Harmonic,
    HartreeFockGradient,
    LennardJones,
    Morse,
)
from qlab.engine.md.state import MDState
from qlab.engine.md.tb_kuvvet import SsSigmaTBKuvveti
from qlab.engine.md.thermostats import Langevin, NoseHoover
from qlab.engine.tb.radyal import UsselRadyalYasa
from qlab.errors import InvalidInputError, ScenarioError
from qlab.provenance import Provenance, StabilityVerdict
from qlab.scenario.schema import (
    RADIAL_LAW_FIELDS,
    TB_SS_SIGMA_LAWS,
    TB_SS_SIGMA_PARAMS,
    TB_SS_SIGMA_REQUIRED,
    MDScenario,
)
from qlab.units import AMU_TO_ELECTRON_MASS, KB_HARTREE_PER_KELVIN, amu_to_electron_mass

#: HF dalı KENDİ yaklaşım listesini taşır. `_YAKLASIMLAR`'ın ÜÇ satırından
#: İKİSİ Plan D için YALAN olur ve şeffaflık panelinde basılırdı (ölçüldü,
#: Plan D §0.2): "elektron yok" -- oysa elektronlar her adımda çözülüyor --
#: ve "kuvvet alanı analitik bir modeldir" -- tam tersi.
#:
#: DİKKAT -- aranabilirlik: bu dizgiler testte `.lower()` sonrası aranır ve
#: Python'da "SAYISAL".lower() ASCII "sayisal" üretir ("sayısal" DEĞİL).
#: Aranan ifadeler bu yüzden küçük harfle yazılıdır.
_HF_YAKLASIMLAR = (
    "Born-Oppenheimer: çekirdekler klasik, elektronlar her adımda anlık çözülür",
    "RHF: tek Slater determinantı, kapalı kabuk (çift işgal)",
    "sonlu baz seti",
    "çekirdekler klasik: sıfır nokta enerjisi ve çekirdek tünellemesi yok",
    "kuvvetler sayısal gradyan ile alınır (merkezi fark); analitik (Pulay) "
    "gradyan bu depoda yoktur",
    "dt kararlılık denetimi HF dalında yapılmaz: omega analitik değildir, "
    "bu yüzden omega*dt < 2 kuralı denetlenmiyor -- dt'yi kullanıcı seçer",
)

_YAKLASIMLAR = (
    "klasik dinamik: çekirdekler nokta parçacık, elektron yok",
    "sıfır nokta enerjisi ve tünelleme yok (kuantum titreşim klasikleştirildi)",
    "kuvvet alanı analitik bir modeldir, hesaplanmış bir yüzey değildir",
)
_IHMALLER = (
    "kuantum istatistiği (Maxwell-Boltzmann klasik limittir)",
    "elektronik uyarılma ve Born-Oppenheimer yüzeyler arası geçiş",
)

#: TB dalı da KENDİ listesini taşır -- HF dalının dersinin (yukarıda) aynısı,
#: ve bu kez ÖNCEDEN: ön ölçümün V4 kopyasında TB türü ortak listeyi
#: basıyordu ve panel "elektron yok", "kuvvet alanı analitik bir modeldir"
#: ve (NVE'de) "enerji SÜRÜKLENMEZ" diyordu (spec eleştirmeni E4, ölçüldü).
#: Üçü de TB için yanlış: elektronlar her adımda çözülüyor; yasa seçilmiş
#: bir model ama yüzey her adımda HESAPLANIYOR; ve korunan büyüklük KE + E
#: DEĞİL, KE + Omega. `tests/md/test_md_scenario.py` elektron çözen her
#: türde bu yalanın yokluğunu `MD_FORCE_FIELD_KINDS` üzerinden kilitler.
#:
#: SPEC md-tb §3'ÜN ON ZORUNLU BEYANI (B1-B10) bu üç sabitte ve
#: `engine/md/tb_kuvvet.py`nin dokstringindedir; hangi beyanın hangi
#: cümlede durduğu `tests/md/test_tb_beyan.py`de parça parça kilitlidir
#: (G6). B2'nin bedeli (tek global mu), B3 (pürüzsüz kesme) ve B5'in küçük
#: kutu / enine hareket yarısı G6'da eklendi -- G4'ün panelinde yoktular.
#: Bulgu düzeltme turunda (24.09.2026) iki değişiklik: (1) B1'in KORUNUM
#: yarısı ("integratör KE + Omega'yı korur, KE + E korunmaz") bu listeden
#: ÇIKARILDI -- liste termostattan bağımsız basılıyordu ve termostatlı
#: panelde "Beklenen hata" satırı "NVT: toplam enerji KORUNMAZ" derken
#: "Yaklaşımlar" satırı korunum iddia ediyordu; aynı içerik zaten yalnız
#: NVE'de basılan `_TB_BEKLENEN`dedir. (2) B9'a spec'in V-kuyusu koşulu ve
#: "duvar sonludur" eklendi (ilk sürümde yalnız "d = 0'da sonlu" vardı).
_TB_YAKLASIMLAR = (
    "Born-Oppenheimer: çekirdekler klasik; elektronlar her adımda tek "
    "orbitalli sıkı-bağ Hamiltoniyeniyle çözülür ve ısıl dengede "
    "(Fermi-Dirac, kanonik) doldurulur",
    "potansiyel Omega (Mermin) serbest enerjisidir: Omega = E_band - kT*S + "
    "E_rep, kuvvet F = -grad(Omega), E'nin gradyanı değil",
    "elektron sıcaklığı senaryoda Kelvin, kuvvet sınıfında hartree "
    "cinsindendir (tanılarda ikisi de adıyla basılır); dönüşüm fabrikada "
    "yapılır ve kT > 0 zorunludur -- kT = 0 MD yoktur",
    "dolumun kimyasal potansiyeli TEK ve globaldir (kanonik, N sabit): "
    "birbirinden uzak parçalar mu üzerinden çiftlenir ve bu yerel olmayan "
    "etki 1/kT ile büyür -- kT > 0'ın bedelidir",
    "atom başına tek s orbitali: Slater-Koster tablosunun yalnız ss-sigma "
    "satırı, açı çarpanı özdeş 1; p/d orbitali ve açıya bağlı kuvvet yok",
    "spin katlanması g_s = 2 sabit (manyetizma yok); yük öz-uyumu (SCC/U) "
    "yok; tek atom türü, onsite yok",
    "radyal yasa seçildi, türetilmedi: bu kenarda doğa çıpası yok; yasadan "
    "geri okunan bağ uzunluğu, genlik ve eşikler fizik iddiası değildir",
    "radyal kesme pürüzsüzdür (r_on ile r_cut arasında sin^2 kuyruğu): "
    "enerji ve kuvvet r_cut'ta süreklidir, sert kesmenin enerji basamağı "
    "yoktur; kuyruk C^1'dir, C^2 değil -- ikinci türev r_on ve r_cut'ta sıçrar",
    "sert çekirdek yok: üstel yasa d = 0'da sonludur ve duvar sonludur (Morse "
    "beyanının eşi); çakışık atom hata verir. d = 0'da bir V-kuyusu olabilir: "
    "dimerde (n_e = 2, kT -> 0, d < r_on) koşul 2|t0|*k*e^k > A*krep*e^krep'tir; "
    "krep <= k ve 2|t0|*k > A*krep bunun yeter koşuludur ve o zaman Omega "
    "d0'dan d = 0'a bariyersiz iner. Sınıf bu parametreleri reddetmez; kuyunun "
    "dibinde kuvvet gradyan değildir",
    "periyodik kutu yalnız x ekseninde (en-yakın-görüntü), y ve z açık; "
    "max(r_cut) < L/2 kesin koşul",
    "dt kararlılık denetimi TB dalında yapılmaz: omega analitik değildir ve "
    "O(dt^2) davranış ancak bir adımdaki aralık değişimi / kT küçükken geri "
    "gelir -- dt'yi kullanıcı seçer",
)
_TB_IHMALLER = (
    "çekirdeklerin kuantum istatistiği (Maxwell-Boltzmann klasik limittir), "
    "sıfır nokta enerjisi ve tünelleme",
    "adyabatik olmayan geçiş: elektronlar her adımda anlık ısıl dengededir",
    "p/d orbitalleri, çok türlü sistemler, 3B periyodik kristal ve k-noktası "
    "örneklemesi",
    "küçük kutu ve enine hareket: Gamma'da tek hücre bir halkadır, kristal "
    "değil; 1B pencereler y = z = 0 alt-uzayında kalır ve enine kararsızlık "
    "bu depoda ölçülmedi (ön ölçümde a = 2 zincirinin yalnız x'te "
    "gevşetilmiş geometrisi 3B'de bir eyerdi)",
)
#: TB dalının NVE `expected_error`ı. Ortak metnin "enerji SÜRÜKLENMEZ"
#: cümlesi burada yanlış olurdu: potansiyel sütunu Omega'dır ve KE + E
#: korunmaz. Eşiğin SAYISI bu metinde YAZILMAZ: tanılarda
#: `energy_drift_tolerance` olarak sabitin kendisinden basılır (metinde
#: literal olsaydı sabit değişince bayatlardı).
_TB_BEKLENEN = (
    "velocity-Verlet simplektiktir ve kT > 0'da Omega pürüzsüz bir yüzeydir: "
    "KE + Omega SALINIR ve salınım dt^2 ile küçülür -- ancak bir adımdaki "
    "aralık değişimi / kT küçükken; Fermi geçişli yörüngede birinci mertebe "
    "sıçramalar olabilir. KE + E KORUNMAZ: fark kT*dS'dir, bir hata değil "
    "tanımdır. energy_drift burada |blok[-1] - blok[0]| / max(KE)'dir, "
    "|blok[0]|'a BÖLÜNMEZ: TB'de enerjinin sıfırı keyfidir (tekbiçim bir "
    "onsite Omega'yı n_e*c kaydırır, yörüngeyi değiştirmez) ve o payda hükmü "
    "bu sıfıra bağlıyordu. Eşik (energy_drift_tolerance) hareketin en büyük "
    "kinetik enerjisinin bir kesridir ve TB'ye özgü sürüklenmeyi -- Fermi "
    "seviyesindeki geçiş adımda çözülmediğinde (adım başına aralık "
    "değişimi / kT büyük) doğan enerji sıçramalarını -- yakalar; iri dt'nin "
    "SALINIMINI yargılamaz: ✓, dt'nin yeterince küçük olduğu anlamına "
    "gelmez. Hareketsiz koşu (max KE = 0) YARGILANAMAZ ve ✗ basılır."
)


def _tb_ss_sigma_alani(p: dict) -> SsSigmaTBKuvveti:
    """`tb_ss_sigma` dalı: iki yasa ve Kelvin -> hartree dönüşümü BURADA.

    Anahtar kapısı YAML tarafında `loader.py`dedir; burada `MDScenario(...)`
    doğrudan kurulduğunda (testler bunu yapıyor) tekrar edilir, aksi hâlde
    yazım hatası ham `KeyError`/`TypeError` olurdu. Değer kapıları
    (sonluluk, kT > 0, L/2) sınıflardadır.
    """
    bilinmeyen = set(p) - set(TB_SS_SIGMA_PARAMS)
    eksik = set(TB_SS_SIGMA_REQUIRED) - set(p)
    if bilinmeyen or eksik:
        raise ScenarioError(
            f"tb_ss_sigma parametreleri: bilinmeyen {sorted(bilinmeyen)}, "
            f"eksik {sorted(eksik)}; geçerli: {list(TB_SS_SIGMA_PARAMS)}"
        )
    yasalar = {}
    for rol in TB_SS_SIGMA_LAWS:
        alanlar = p[rol]
        if not isinstance(alanlar, dict) or set(alanlar) != set(RADIAL_LAW_FIELDS):
            raise ScenarioError(
                f"tb_ss_sigma: '{rol}' tam olarak {list(RADIAL_LAW_FIELDS)} "
                f"anahtarlarını taşıyan bir sözlük olmalı, verilen: {alanlar!r}"
            )
        yasalar[rol] = UsselRadyalYasa(**alanlar)
    kelvin = p["electron_temperature_kelvin"]
    try:
        if isinstance(kelvin, (bool, str, bytes)) or np.ndim(kelvin) != 0:
            raise TypeError
        kelvin = float(kelvin)
    except (TypeError, ValueError) as hata:
        raise ScenarioError(
            f"electron_temperature_kelvin tek bir sayı (KELVİN) olmalı: {kelvin!r}"
        ) from hata
    # Kelvin -> hartree SINIRDA: sınıf yalnız hartree görür (spec B7).
    return SsSigmaTBKuvveti(
        ss_sigma=yasalar["ss_sigma"],
        itme=yasalar["itme"],
        n_elektron=p["n_electron"],
        kT_hartree=kelvin * KB_HARTREE_PER_KELVIN,
        kutu_x_bohr=p.get("box_x_bohr"),
    )


def _fcc_kafes(n_particles: int, density_reduced: float, sigma: float):
    """FCC kafes ve kutu kenarı. L = sigma * (N/rho*)^(1/3).

    ÖLÇÜLDÜ: N=108, rho*=0.8442, sigma=6.4345174543 -> L = 32.422173 bohr.
    Kutu kenarı YOĞUNLUKTAN TÜRETİLİR, YAML'dan okunmaz -- tek doğruluk
    kaynağı olsun diye. (Eski YAML 33.0656 yazıyordu ve o değer beyan edilen
    yoğunlukla çelişiyordu: gerçek rho* 0.795871, -%5.72.)
    """
    hucre = round((n_particles / 4) ** (1 / 3))
    if 4 * hucre**3 != n_particles:
        raise ScenarioError(
            f"n_particles bir FCC sayısı olmalı (4*n^3): {n_particles}. "
            f"En yakınları {4 * hucre**3} ve {4 * (hucre + 1) ** 3}."
        )
    kenar = (n_particles / density_reduced) ** (1 / 3) * sigma
    a = kenar / hucre
    taban = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    konumlar = np.concatenate(
        [
            taban + np.array([i, j, k]) * a
            for i in range(hucre)
            for j in range(hucre)
            for k in range(hucre)
        ]
    )
    return konumlar, float(kenar)


def build(scenario: MDScenario):
    """Senaryodan motoru kurar. Kütle dönüşümü BURADA, sınırda yapılır."""
    tur = scenario.force_field["kind"]
    p = dict(scenario.force_field.get("params", {}))
    ilk = scenario.initial

    # İKİ BAŞLANGIÇ BİÇİMİ. Bu dal eskiden YOKTU ve `argon_nvt.yaml` ham
    # `KeyError: 'positions_bohr'` veriyordu -- Global Constraint "sessiz
    # başarısızlık yok / ham hata tüketiciye ulaşmaz" ihlali.
    if "positions_bohr" in ilk:
        konumlar = np.asarray(ilk["positions_bohr"], dtype=float)
    elif ilk.get("lattice") == "fcc":
        konumlar, kenar = _fcc_kafes(
            int(ilk["n_particles"]), float(ilk["density_reduced"]), float(p["sigma"])
        )
        p["box_length"] = kenar  # yoğunluktan türetilen kenar YAML'ı EZER
    else:
        raise ScenarioError(
            "initial bloğu ya `positions_bohr` ya da "
            "`lattice: fcc` + `n_particles` + `density_reduced` içermeli; "
            f"verilen anahtarlar: {sorted(ilk)}"
        )

    ham_kutle = np.asarray(ilk["masses_amu"], dtype=float)
    if ham_kutle.size == 1 and konumlar.shape[0] > 1:
        ham_kutle = np.full(konumlar.shape[0], ham_kutle.item())
    kutleler = amu_to_electron_mass(ham_kutle)

    hizlar = np.asarray(
        ilk.get("velocities", np.zeros_like(konumlar)), dtype=float
    )
    durum = MDState(positions=konumlar, velocities=hizlar, masses=kutleler)

    if tur == "harmonic":
        alan = Harmonic(k=p["k"], center=p.get("center", np.zeros_like(konumlar)))
        # velocity-Verlet omega*dt = 2.0'da patlar (ölçüldü: 2.001 -> 4.3e+54).
        # Harmonikte omega analitiktir, o yüzden burada denetlenebilir.
        omega = float(np.sqrt(alan.k / kutleler.min()))
        if omega * scenario.dt >= 2.0:
            raise ScenarioError(
                f"dt kararlılık sınırını aşıyor: omega*dt = "
                f"{omega * scenario.dt:.3f} >= 2.0. velocity-Verlet bu sınırda "
                f"patlar. dt < {2.0 / omega:.4g} seçin."
            )
    elif tur == "morse":
        alan = Morse(**p)
    elif tur == "lennard_jones":
        alan = LennardJones(**p)
    elif tur == "hartree_fock":
        # dt kararlılık denetimi BU DALDA YOK: omega analitik değil.
        # Atlandığı _HF_YAKLASIMLAR'da beyan edilir (sessiz atlama değil).
        alan = HartreeFockGradient(**p)
    elif tur == "tb_ss_sigma":
        # dt kararlılık denetimi BU DALDA DA YOK (spec B10): omega analitik
        # değil. Atlandığı _TB_YAKLASIMLAR'da beyan edilir.
        alan = _tb_ss_sigma_alani(p)
    else:
        raise ScenarioError(f"bilinmeyen kuvvet alanı: {tur!r}")

    termostat = None
    if scenario.thermostat is not None:
        t_tur = scenario.thermostat["kind"]
        t_p = dict(scenario.thermostat.get("params", {}))
        if t_tur == "langevin":
            termostat = Langevin(**t_p)
        elif t_tur == "nose_hoover":
            termostat = NoseHoover(**t_p)
        else:
            raise ScenarioError(f"bilinmeyen termostat: {t_tur!r}")

    return durum, alan, termostat


#: NVE koşusunda kabul edilen bağıl enerji sürüklenmesi. LİTERAL ve ölçülmüş:
#: dt = 40 a.u.'da 1.739e-03, dt = 70'te 1.342e-02, dt = 90'da 2.397e-01.
#: Eskiden `converged=True` SABİTTİ ve panel %24 kaymış enerjinin yanında bile
#: ✓ basıyordu -- deponun `tdse/factory.py` ve `hf/factory.py` desenine aykırı.
#: HF dalı (03.10.2026, C5) bu eşiği max(KE) paydasıyla kullanır. Eşik
#: Morse kalibrasyonundan TAŞINDI ve HF'de ayrıca ölçülmedi: Morse'ta
#: |blok[0]|/max(KE) = 1.09 / 1.23 / 0.81 (yukarıdaki üç dt), yani eşik fiilen
#: max(KE) ölçeğinde ölçülmüştü. HF'de hüküm omega*dt ve koşu uzunluğuyla
#: MONOTON DEĞİLDİR (bkz. `energy_drift`, SINIR).
_MD_DRIFT_TOLERANCE: float = 5.0e-03

#: TB dalının NVE sürüklenme eşiği. ORTAK EŞİKLE KARŞILAŞTIRILAMAZ: KLASİK
#: dallardan farklı paydayla (`energy_drift`: |blok[0]| değil, kaydedilen EN
#: BÜYÜK kinetik enerji; HF de 03.10'dan beri bu paydada) ve TB'ye özgü
#: pencerelerde ölçüldü, yani bu sayı "sürüklenme, hareketin en büyük kinetik enerjisinin
#: %1'i" demektir. ÖLÇÜLDÜ (G5, 24.09.2026, bu makine, 115 hücre; pencere
#: aileleri `energy_drift` dokstringinde):
#:
#: * omega_max*dt <= 0.46 olan 62 hücrenin 62'si eşiğin altında; en büyüğü
#:   7.04e-03 (1.4x pay; düzgün zincirde yuvarlamadan büyüyen kararsızlık).
#: * YAKALADIĞI TB'ye özgü sürüklenmedir -- Fermi seviyesindeki geçişin kT
#:   ölçeği adımda çözülmeyince (spec B2(b), B10): zincir(16, 2, 0.05, 3),
#:   n_e = 15, v0 = 0, dt = 10'da T_e = 3.16 K -> 5.48e-02 (✗), aynı pencere
#:   T_e = 631.6 K -> 5.97e-04 (✓). Ortak dalın |blok[0]| oranı ikisinde de
#:   ✓ derdi (1.03e-03 ve 9.96e-06; ortak eşik 5e-03).
#: * YAKALAMADIĞI iri dt'nin SALINIMIDIR: omega_max*dt >= 0.93 olan 33
#:   hücrenin 8'i eşiğin altında kaldı (1.2'de 5.35e-05'e kadar). Verlet
#:   orada hâlâ kararlı ve simplektik: enerji büyük salınır, sürüklenmez.
#:   Bu eşik dt'nin yeterince küçük olduğunu ONAYLAMAZ (spec B10: dt'yi
#:   kullanıcı seçer).
_TB_DRIFT_TOLERANCE: float = 1.0e-02


def energy_drift(
    scenario: MDScenario, kinetic: np.ndarray, potential: np.ndarray
) -> float:
    """NVE enerji sürüklenmesi: blok ortalamalarının baş-son farkı / ölçek.

    PAY -- bütün türlerde AYNI: toplam = KE + potansiyel, kayıtlar
    min(10, M) bloğa bölünür, pay = |blok[-1] - blok[0]|. SÜRÜKLENME,
    salınım DEĞİL. BLOK SAYISI VERİYE GÖRE: ÖLÇÜLDÜ -- 10 sabit blokla
    M < 10 kayıtta `reshape` boş dilim üretiyor, sürüklenme `nan` çıkıyor ve
    panel onun YANINDA ✓ basıyordu (M=3 ve M=9'da görüldü; ilk sağlam değer
    M=10). Tek kayıtta (M=1) tek blok kalır ve pay 0.0 olurdu -- ölçülmemiş
    bir şey "sıfır sürüklenme ✓" diye basılırdı; bu yüzden M < 2 HATADIR.
    (Hesap G5'e kadar `run.py::_run_md`in içindeydi; payda türe bağlı
    olunca sevkle birlikte buraya, `provenance_for`un yanına taşındı.)

    PAYDA TÜRE BAĞLIDIR:

    * Ortak dal (klasik kuvvet alanları: harmonic, morse, lennard_jones):
      |blok[0]|. Değişmedi.
    * `hartree_fock` (03.10.2026, C5): max(KE), TB ile aynı. |blok[0]| HF'de
      ELEKTRONİK enerjidir (~-1.117 Ha): titreşim enerjisinin 1644 katı
      (h2_aimd genliği 0.05; 0.10'da 432), yani eski oran sürüklenmeye bu
      kadar kördü -- ✗ ancak pay titreşim enerjisinin ~8 katını aşınca
      basılıyordu (Morse-eşdeğer dt 40/70/90 x n 50-800 ızgarasında 15/15 ✓).
      ÖLÇÜLDÜ (public yol, A/B): sevk edilen h2_aimd penceresinde (200 adım)
      yalnız gradyan adımı 1e-5 -> 1e-13 değişince (sonlu fark gürültüsü;
      başlangıç, dt ve Verlet salınımı aynı) yeni oran 1.465e-04 (✓) ->
      2.825e-01 (✗); eski oran 8.78e-08 -> 2.62e-04, ikisinde de ✓.

    SINIR -- BLOK ÖLÇÜTÜ SALINIMI SÜRÜKLENMEDEN AYIRAMAZ (bütün dallar,
    03.10.2026 doğrulaması): kısa koşuda (birkaç periyot) ya da örnekleme
    katlanmasında pay Verlet salınımının kendisi olabilir. HF h2_aimd
    geometrisi omega*dt = 0.25, n = 20: ✗ 1.488e-02 ~ (omega*dt)^2/4, n >= 30
    ✓; omega*dt = 1.4037: n = 100/400/1600 ✗, 3200 ✓ (tepe-tepe sabit). Morse
    aynı hücrelerde ✗ verir. Bu yüzden ✗ "korunum DOĞRULANAMADI" demektir,
    "sürüklenme kanıtlandı" değil: koşuyu uzatın ya da save_every/dt'yi
    değiştirip yeniden bakın. ✓ payın eşiğin altında kaldığını söyler.
    * `tb_ss_sigma`: kaydedilen EN BÜYÜK kinetik enerji, max(KE).
      |blok[0]| TB'de KEYFİ BİR SIFIRA bölmektir (spec B8, eleştirmen
      P0-2): tekbiçim bir onsite c Omega'yı n_e*c kaydırır, kuvveti ve
      yörüngeyi değiştirmez (spec B6), ama eski oranı oynatır. ÖLÇÜLDÜ
      (G5, Omega + c sarmalayıcısıyla, yörünge bit bit aynı): zincir(16,
      2, 0.05, 3), n_e = 16, M_H, v0 = 0, dt = 20, 200 adım -- c = 0'da
      eski oran 3.26e-05 (✓), c = -Omega0'da 9.13e-02 (✗); dt = 40, 100
      adımda c = 0'da 1.83e-03 (✓), c = -Omega0'da 1.67 (✗). Hüküm
      enerjinin sıfırına bağlıydı. Üstelik GERÇEK alarmı kaçırıyordu:
      Fermi geçişli pencerede (aynı zincir, n_e = 15, T_e = 3.16 K, dt =
      10, 400 adım) eski oran c = 0'da 1.03e-03 (✓), max(KE) oranı
      5.48e-02 (✗). Kaynağı ölçüldü: SOMO ile komşu seviye adım uçlarında
      5.8 kT'ye kadar yaklaşıyor ve adım başına aralık değişimi / kT'nin
      medyanı 349 -- spec B2(b)'nin "<~ 1.3-4" koşulunun çok dışında; en
      büyük enerji sıçraması (-1.52e-03) bu yaklaşmanın hemen ardında ve
      FD(Omega) orada h = 1e-5'te 1.1e-07'ye bozuluyor (başka noktalarda
      ~5e-11: yüzey kT ölçeğinde keskin). 631.6 K'de aynı oran medyan 1.9,
      en büyük sıçrama 9.1e-05. Sürüklenme dt ile temiz bir kuvvet yasasına
      uymuyor (3.16 K: dt 5 / 10 / 20 -> 5.71e-03 / 5.48e-02 / 1.89e-01).
      KE mutlaktır ve c'den etkilenmez: yeni oran c in {0, -Omega0, +-16}
      boyunca en çok 8.5e-11 bağıl farkla aynı (yuvarlama).

    PENCERE AİLELERİ (eşiğin ölçümü, `_TB_DRIFT_TOLERANCE`; hepsi ortak
    yasa penceresi, a = 2, L = 2N, M_H): M11 penceresi (Peierls minimumu +
    x-hızları, T 10-3000 K, 3 tohum, 5 T_min), metalik kT = 0.02 düzgün
    zincir, zincir(16, 2, 0.05, 1..6) v0 = 0 4000 a.u., düzgün zincir
    (gürültüsüz ve 1e-6 gürültülü) v0 = 0, çeyrek dolu N16 ve yarı dolu N8;
    dt merdiveni omega_max*dt = 0.05 ... 2.1 (omega_max = 0.02315, M11
    minimumunun en yüksek x-modu; öteki ailelerde yaklaşık).

    NEDEN ORTALAMA KE DEĞİL (ÖLÇÜLDÜ, G5): KE'nin zaman ortalaması
    DURAĞAN OLMAYAN koşuda yanlış ✗ verdi. Düzgün zincir + 1e-6 bohr
    gürültü (v0 = 0) Peierls kararsızlığıyla pencerenin içinde çöküyor;
    sessiz başlangıç ortalamayı seyreltiyor: 1000 a.u., dt = 10'da
    pay/ortalama(KE) = 1.79e-02 (✗), pay/max(KE) = 3.85e-03 (✓); gürültüsüz
    düzgün zincirde 2000 a.u., dt = 5'te (omega_max*dt ~ 0.12) 3.24e-02'ye
    karşı 2.94e-04. max(KE) hareketin ulaştığı ölçeği okur. Omega'nın
    tepe-tepe salınımı aynı pencerelerde max(KE)'ye yakın çıktı (eşdeğer
    bir aday; seçilmedi, çünkü sabit hızla öteleyen bir sistemde sıfırdır).

    HAREKETSİZ KOŞU YARGILANAMAZ: max(KE) = 0 ise oran `nan` döner ve
    panel ✗ basar -- sıfır sürüklenme diye ✓ DEĞİL. max(KE) sıfır değil
    ama yuvarlama düzeyindeyse (ör. gürültüsüz düzgün zincir, kararsızlık
    henüz büyümemiş) oran anlamsız derecede büyür; hüküm yine ✗'dir.
    """
    kinetic = np.asarray(kinetic, dtype=float)
    potential = np.asarray(potential, dtype=float)
    if kinetic.ndim != 1 or kinetic.shape != potential.shape:
        raise InvalidInputError(
            "kinetic ve potential aynı uzunlukta tek boyutlu diziler olmalı: "
            f"{kinetic.shape} ve {potential.shape}"
        )
    toplam = kinetic + potential
    if len(toplam) < 2:
        raise InvalidInputError(
            f"sürüklenme ölçmek için en az 2 kayıt gerekir, {len(toplam)} var; "
            "n_steps/save_every oranını büyütün"
        )
    n_blok = min(10, len(toplam))
    blok = toplam[: (len(toplam) // n_blok) * n_blok].reshape(n_blok, -1).mean(axis=1)
    pay = float(abs(blok[-1] - blok[0]))
    tur = scenario.force_field["kind"]
    if tur == "tb_ss_sigma" or tur == "hartree_fock":
        olcek = float(np.max(kinetic))
        return pay / olcek if olcek > 0.0 else float("nan")
    # numpy bölmesi BİLEREK (G5 öncesi `run.py` ile bit bit aynı): blok[0] = 0
    # ise inf + RuntimeWarning, Python float'ının ZeroDivisionError'ı değil.
    return float(pay / abs(blok[0]))


def provenance_for(
    scenario: MDScenario,
    *,
    energy_drift: float,
    n_constraints: int = 3,
    stability: StabilityVerdict | None = None,
    external_stability: StabilityVerdict | None = None,
    kararlilik_tanisi: dict | None = None,
) -> Provenance:
    """MD koşusunun şeffaflık bloğu.

    `stability` / `external_stability` (B10, 03.10.2026): HF kuvvet alanının
    koşu boyunca biriktirdiği EN KÖTÜ iç / dış kararlılık hükmü
    (`HartreeFockGradient.en_kotu_ic` / `en_kotu_dis`). `converged`a GİRMEZ --
    kararlılık yakınsamadan ayrı bir kavramdır (Osman'ın kararı, 03.10.2026);
    panelde kendi satırlarını alır.
    """
    kutleler = amu_to_electron_mass(
        np.asarray(scenario.initial["masses_amu"], dtype=float)
    )
    # SEVK DEĞİŞKENİ `tur`: dallar `tur == "..."` biçiminde yazılır ki
    # türetme kilitleri (kök `conftest.py`) onları görsün. Eskiden
    # `scenario.force_field["kind"] == ...` yazılıydı; o bir `Subscript`tir ve
    # `sevk_tarayici` onu GÖRMÜYORDU (spec eleştirmeni P0-1) -- yani bu
    # dalların sevkle eşleştiğini hiçbir kilit denetlemiyordu.
    tur = scenario.force_field["kind"]
    p = scenario.force_field.get("params", {})
    if tur == "hartree_fock":
        yaklasimlar, ihmaller = _HF_YAKLASIMLAR, _IHMALLER
    elif tur == "tb_ss_sigma":
        yaklasimlar, ihmaller = _TB_YAKLASIMLAR, _TB_IHMALLER
    else:
        yaklasimlar, ihmaller = _YAKLASIMLAR, _IHMALLER
    termostat = scenario.thermostat
    if termostat is None:
        yontem = f"klasik MD / velocity-Verlet / {tur}"
        beklenen = (
            "velocity-Verlet simplektiktir: enerji SÜRÜKLENMEZ, SALINIR. "
            "Salınım genliği (omega*dt)^2/4 mertebesindedir ve bir hata değildir."
        )
        esik = _MD_DRIFT_TOLERANCE
        if tur == "tb_ss_sigma":
            # TB'nin eşiği TB'ye özgü pencerelerde ölçüldü (`_TB_DRIFT_TOLERANCE`);
            # HF ortak eşiği max(KE) paydasıyla kullanır (C5, `_MD_DRIFT_TOLERANCE`).
            beklenen, esik = _TB_BEKLENEN, _TB_DRIFT_TOLERANCE
        yakinsadi = bool(np.isfinite(energy_drift)) and energy_drift <= esik
    else:
        parametreler = ", ".join(
            f"{ad}={deger}" for ad, deger in sorted(termostat["params"].items())
        )
        yontem = (
            f"klasik MD / velocity-Verlet + {termostat['kind']}({parametreler})"
            f" / {tur}"
        )
        # NVT'DE ENERJİ KORUNMAZ -- bu bir kusur değil, tanımdır. Eskiden
        # termostatlı koşuda da "enerji SÜRÜKLENMEZ" yazıyordu ve ölçülen
        # sürüklenme %22.4 iken panel ✓ basıyordu.
        beklenen = (
            "NVT: toplam enerji KORUNMAZ, sistem banyoyla enerji alışverişi "
            "yapar. Ölçülen büyüklük kinetik sıcaklıktır, enerji sürüklenmesi "
            "değil; aşağıdaki energy_drift bir tanı değeridir, bir hata payı DEĞİL."
        )
        yakinsadi = True
        esik = None  # NVT'de sürüklenme hükme girmez: eşik YOK (0.0 değil)

    tanilar = {
        "dt": scenario.dt,
        "n_steps": scenario.n_steps,
        "energy_drift": energy_drift,
        "basis": p.get("basis"),
        # YAML `step` vermezse koşu HartreeFockGradient'ın VARSAYILANIYLA yapılır;
        # panel eskiden `None` basıyordu (03.10.2026, B10 ön ölçümü).
        "gradient_step_bohr": p.get(
            "step",
            HartreeFockGradient.__dataclass_fields__["step"].default if tur == "hartree_fock" else None,
        ),
        "n_constraints": n_constraints,
        "masses_electron_mass": np.round(kutleler, 4).tolist(),
        "amu_to_electron_mass": AMU_TO_ELECTRON_MASS,
    }
    if tur == "hartree_fock":
        # C5 (03.10.2026): paydası TB'ninki gibi max(KE); panelde AÇIK. ✗ burada
        # "korunum DOĞRULANAMADI" demektir (`energy_drift`, SINIR).
        tanilar.update(energy_drift_scale="max_kinetic", energy_drift_tolerance=esik)
    if tur == "tb_ss_sigma":
        # Eleştirmen E4: V4'ün tanılarında kT, n_electron ve r_cut YOKTU.
        # Sıcaklık İKİ birimde basılır -- tuzağın kendisi bu (spec B7).
        tanilar.update(
            electron_temperature_kelvin=float(p["electron_temperature_kelvin"]),
            kT_hartree=float(p["electron_temperature_kelvin"]) * KB_HARTREE_PER_KELVIN,
            n_electron=p["n_electron"],
            r_cut_bohr=max(
                float(p[rol]["r_cut_bohr"]) for rol in TB_SS_SIGMA_LAWS
            ),
            box_x_bohr=p.get("box_x_bohr"),
            # G5: sürüklenmenin paydası ve eşiği panelde AÇIK -- ortak dalın
            # |blok[0]| paydasıyla karıştırılmasın diye (`energy_drift`).
            energy_drift_scale="max_kinetic",
            energy_drift_tolerance=esik,
        )
    if kararlilik_tanisi:
        tanilar.update(kararlilik_tanisi)
    return Provenance(
        method=yontem,
        approximations=yaklasimlar,
        neglected=ihmaller,
        expected_error=beklenen,
        converged=yakinsadi,
        diagnostics=tanilar,
        stability=stability,
        external_stability=external_stability,
    )
