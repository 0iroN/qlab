"""MDScenario -> (durum, kuvvet alanı, termostat) ve Provenance."""

from __future__ import annotations

import dataclasses

import numpy as np

from qlab.engine.md.forcefields import (
    Harmonic,
    HartreeFockGradient,
    LennardJones,
    Morse,
)
from qlab.engine.md.integrator import Trajectory
from qlab.engine.md.state import MDState
from qlab.engine.md.tb_kuvvet import SsSigmaTBKuvveti
from qlab.engine.md.thermostats import Langevin, NoseHoover
from qlab.engine.tb.radyal import UsselRadyalYasa
from qlab.errors import InvalidInputError, QLabError, ScenarioError
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
    "bu yüzden omega*dt < 2 kuralı denetlenmiyor -- dt'yi kullanıcı seçer, "
    "koşu reddedilmez; seçilen dt'yi NVE'de koşudan SONRA salınım hükmü "
    "(energy_oscillation) ve kare-0 geometrisinin ve yolun her adımda izlenen "
    "en kısa çift mesafesi karesinin (element çifti türü başına da) sayısal "
    "Hessian'ından ölçülen omega*dt yargılar (omega_dt_frame0, "
    "omega_dt_path_frames; yalnız bu karelere bakar)",
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
    "gelir -- dt'yi kullanıcı seçer, koşu reddedilmez; seçilen dt'yi koşudan "
    "SONRA NVE'nin salınım hükmü yargılar (energy_oscillation), koşudan ÖNCE "
    "de kare-0 geometrisinin sayısal Hessian'ından ölçülen omega0*dt hükme "
    "girer (omega_dt_frame0), koşudan SONRA da yolun her adımda izlenen iki "
    "karesininki (omega_dt_path_frames; yalnız seçilen karelere bakar)",
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
    "SALINIMINI ise ayrı ölçüt yargılar: energy_oscillation = tepe-tepe(KE + "
    "Omega) / max(KE), uçlar kayıtlardan değil her adımdan, eşiği "
    "energy_oscillation_tolerance. S'nin paydası tüm KE'dir ve titreşime "
    "katılmayan KE onu seyreltir; onu omega*dt tanısı yakalar: "
    "omega_dt_frame0 = kare-0 geometrisinin (koşudan ÖNCE) ve "
    "omega_dt_path_frames = yolun her adımda izlenen iki karesinin (en kısa "
    "çift mesafesi, kütle ağırlıklı sertlik tepesi; koşudan SONRA) en hızlı "
    "kütle ağırlıklı modu x dt, ikisinin eşiği omega_dt_tolerance. Yakınsama "
    "dördünün VE'sidir. Hareketsiz koşu (max KE = 0) YARGILANAMAZ ve ✗ basılır."
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
    # NP-BOOL (11.10.2026, grup 6; ölçüldü HEAD a9ec48a, `MDScenario(...)`
    # doğrudan): `np.True_` ve `np.array(True)` 1 K diye geçiyordu,
    # `np.complex128` uyarıyla gerçel kısmına iniyordu, `10**400` (YAML'da
    # 401 haneli tamsayı da) ham `OverflowError` veriyordu.
    try:
        if (
            isinstance(kelvin, (bool, str, bytes))
            or np.ndim(kelvin) != 0
            or np.asarray(kelvin).dtype.kind in "bc"
        ):
            raise TypeError
        kelvin = float(kelvin)
    except (TypeError, ValueError, OverflowError) as hata:
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


def _bool_iceriyor(deger: object) -> bool:
    """İç içe liste/demet/dizide bool ya da karmaşık öğe (Python ya da numpy) var mı."""
    if isinstance(deger, (bool, np.bool_, complex, np.complexfloating)):
        return True
    if isinstance(deger, np.ndarray):
        if deger.dtype.kind == "O":
            return any(_bool_iceriyor(x) for x in deger.ravel())
        return deger.dtype.kind in "bc"
    if isinstance(deger, (list, tuple)):
        return any(_bool_iceriyor(x) for x in deger)
    return False


def _fcc_tam_sayisi(deger: object) -> int:
    """fcc `n_particles`: tam sayı ya da tam değerli float; bool, dize, kesir REDDEDİLİR.

    Ölçüldü (11.10.2026, grup 6, HEAD a9ec48a, kamu yolu): `int()`
    `n_particles: 108.5`i ve `108.9`u SESSİZCE 108'e kesiyor, `'108'`i 108
    yapıyordu (koşu 108 atomla rc = 0). Yükleyicinin `_yaml_tam_sayisi`
    kuralının aynısı (`108.0` kabul); hata `ScenarioError`a çağıran çevirir.

    Grup 6 düzeltmesi (11.10.2026, ölçüldü HEAD a9ec48a, kamu yolu):
    `n_particles: 10**400` (401 haneli) `_fcc_kafes`te ham `OverflowError`,
    `0` ham `ZeroDivisionError`, `-108` ham `TypeError` (karmaşık küp kök)
    veriyordu. Kafes hesabı float'tır: sayı pozitif ve float'a sığmalı.
    """
    if isinstance(deger, (bool, np.bool_)) or not isinstance(
        deger, (int, np.integer, float, np.floating)
    ):
        raise TypeError(f"n_particles tam sayı olmalı, verilen: {deger!r}")
    if isinstance(deger, (float, np.floating)) and not float(deger).is_integer():
        raise ValueError(f"n_particles tam sayı olmalı, kesir kesilmez: {deger!r}")
    if deger <= 0:
        raise ValueError(f"n_particles pozitif olmalı: {deger!r}")
    try:
        float(deger)
    except OverflowError as hata:
        raise ValueError(
            f"n_particles float'a sığmıyor ({len(str(int(deger)))} hane)"
        ) from hata
    return int(deger)


def _bool_degil(ad: str, deger: object) -> float:
    """`float(deger)`; bool (Python ya da numpy) `TypeError` -- `true` 1.0 okunmaz."""
    if isinstance(deger, (bool, np.bool_)):
        raise TypeError(f"{ad} sayı olmalı, bool verildi: {deger!r}")
    return float(deger)


def _fcc_olcegi(ad: str, deger: object) -> float:
    """fcc kafes ölçeği (`density_reduced`, `sigma`): bool değil, sonlu ve > 0.

    Ölçüldü (11.10.2026, grup 6, HEAD a9ec48a, `argon_nvt.yaml`ın tek satırı
    bozularak): `density_reduced: 10**400` ham `OverflowError`, `0` ham
    `ZeroDivisionError`, `-0.8442` ham `TypeError` (karmaşık kenar) veriyordu;
    `.nan` / `.inf` rc = 1 ama yanlış nedenle ("konumlar sonlu olmalı",
    "kutu kenarı ... 0.0"); `sigma: true` kafesi 1 bohr ölçekle kuruyordu.
    Hata `ScenarioError`a çağıran çevirir (`ValueError` / `TypeError`).
    """
    try:
        sayi = _bool_degil(ad, deger)
    except OverflowError as hata:
        raise ValueError(f"{ad} float'a sığmıyor ({len(str(deger))} hane)") from hata
    if not (np.isfinite(sayi) and sayi > 0.0):
        raise ValueError(f"{ad} pozitif ve sonlu olmalı: {deger!r}")
    return sayi


def _senaryo_dizisi(ad: str, deger: object) -> np.ndarray:
    """`initial` bloğundan sayısal dizi; çevrilemeyen girdi `ScenarioError`.

    C6 (11.10.2026, ölçüldü `python -m qlab.run`): `positions_bohr`daki bir
    dize (`"np.float64(0.0)"`), `masses_amu: [abc, ...]` ve `velocities`
    içindeki bir harf ham `ValueError` traceback'i veriyordu.

    NP-BOOL (11.10.2026, grup 6; ölçüldü HEAD a9ec48a, kamu yolu): YAML
    `true`/`yes` (bool) `dtype=float`ta SESSİZCE 1.0 oluyordu --
    `masses_amu: [true]` 1 amu'luk koşu, `positions_bohr`da `true` 1.0 bohr.
    bool ve karmaşık öğe (Python ya da numpy) artık `ScenarioError`.
    Float'a sığmayan öğe (401 haneli konum / kütle / hız) ham `OverflowError`
    veriyordu (ölçüldü HEAD a9ec48a, `tb_zincir_md.yaml`); o da artık burada.
    """
    if _bool_iceriyor(deger):
        raise ScenarioError(
            f"initial.{ad} sayısal bir dizi olmalı; bool ve karmaşık öğe sayıya "
            f"çevrilmez (true → 1.0), verilen: {deger!r}"
        )
    try:
        return np.asarray(deger, dtype=float)
    except (TypeError, ValueError, OverflowError) as hata:
        raise ScenarioError(
            f"initial.{ad} sayısal bir dizi olmalı, verilen: {deger!r} ({hata})"
        ) from hata


def _senaryo_nesnesi(sinif, ad: str, params: dict, yaml_params: dict | None = None):
    """Kuvvet alanı / termostat kurulumu; YAML `params` hatası `ScenarioError`.

    C6 (11.10.2026, ölçüldü `python -m qlab.run`): yanlış yazılmış anahtar
    (`de:` -> `Morse.__init__() got an unexpected keyword argument 'de'`,
    `sigmaa`, termostatta `surtunme`) ve sayı yerine dize (`d_e: abc` ->
    numpy `isfinite` TypeError'ı) ham traceback veriyordu. Sınıfların değer
    kapıları (`InvalidInputError`) olduğu gibi geçer. `yaml_params` mesajda
    gösterilen (YAML'ın yazdığı) sözlüktür; verilmezse `params`.

    NP-BOOL (11.10.2026, grup 6; ölçüldü HEAD a9ec48a, kamu yolu): sınıfların
    `isfinite(x) and x > 0` kapısı bool'u sayı sanıyordu -- `h2_morse.yaml`da
    `d_e: true`, `a: true`, `r_e: yes`; `argon_nvt.yaml`da LJ `epsilon: true`,
    `cutoff: true`, Langevin `friction: true`, `seed: true` koşusu rc = 0
    bitip .npz YAZIYORDU (1.0 / 1 ile). Tipi `bool` OLMAYAN bir sınıf
    alanında bool ya da karmaşık değer (iç içe de: `center`) artık
    `ScenarioError` ve anahtarı ADIYLA söylenir. Tipi `bool` olan alan
    (`HartreeFockGradient.stability`) muaf: `false` geçerli bir değerdir.
    Sınıfta olmayan anahtar burada denetlenmez, sınıfın kendi `TypeError`ı
    ("unexpected keyword argument") ona düşer.
    """
    verilen = params if yaml_params is None else yaml_params
    alan_tipleri = {f.name: f.type for f in dataclasses.fields(sinif)}
    for anahtar, deger in params.items():
        tip = alan_tipleri.get(anahtar)
        if tip is not None and tip not in (bool, "bool") and _bool_iceriyor(deger):
            raise ScenarioError(
                f"{ad} parametresi `{anahtar}`: bool (true/yes) ve karmaşık değer "
                f"sayıya çevrilmez (true → 1.0). Verilen: {verilen!r}"
            )
    try:
        return sinif(**params)
    except (TypeError, ValueError) as hata:
        raise ScenarioError(
            f"{ad} parametreleri geçersiz: {hata}. Verilen: {verilen!r}"
        ) from hata


def build(scenario: MDScenario):
    """Senaryodan motoru kurar. Kütle dönüşümü BURADA, sınırda yapılır."""
    tur = scenario.force_field["kind"]
    # `yaml_p` KULLANICININ yazdigidir; `p` fabrikanin ekledigini de tasir
    # (fcc dalinda `box_length`). Hata mesajlari yalniz `yaml_p`'yi gosterir.
    yaml_p = dict(scenario.force_field.get("params", {}))
    p = dict(yaml_p)
    ilk = scenario.initial

    # İKİ BAŞLANGIÇ BİÇİMİ. Bu dal eskiden YOKTU ve `argon_nvt.yaml` ham
    # `KeyError: 'positions_bohr'` veriyordu -- Global Constraint "sessiz
    # başarısızlık yok / ham hata tüketiciye ulaşmaz" ihlali.
    fcc = "positions_bohr" not in ilk and ilk.get("lattice") == "fcc"
    if "positions_bohr" in ilk:
        konumlar = _senaryo_dizisi("positions_bohr", ilk["positions_bohr"])
    elif fcc:
        try:
            fcc_sayilari = (
                _fcc_tam_sayisi(ilk["n_particles"]),
                _fcc_olcegi("density_reduced", ilk["density_reduced"]),
                _fcc_olcegi("sigma", p["sigma"]),
            )
        except (KeyError, TypeError, ValueError) as hata:
            raise ScenarioError(
                "lattice: fcc için `n_particles` (tam sayı), `density_reduced` ve "
                f"kuvvet alanının `sigma`sı sayı olmalı: {hata!r}"
            ) from hata
        konumlar, kenar = _fcc_kafes(*fcc_sayilari)
        p["box_length"] = kenar  # yoğunluktan türetilen kenar YAML'ı EZER
    else:
        raise ScenarioError(
            "initial bloğu ya `positions_bohr` ya da "
            "`lattice: fcc` + `n_particles` + `density_reduced` içermeli; "
            f"verilen anahtarlar: {sorted(ilk)}"
        )

    if "masses_amu" not in ilk:
        raise ScenarioError(f"initial bloğunda `masses_amu` yok: {sorted(ilk)}")
    ham_kutle = _senaryo_dizisi("masses_amu", ilk["masses_amu"])
    if ham_kutle.size == 1 and konumlar.shape[0] > 1:
        ham_kutle = np.full(konumlar.shape[0], ham_kutle.item())
    kutleler = amu_to_electron_mass(ham_kutle)

    hizlar = _senaryo_dizisi("velocities", ilk.get("velocities", np.zeros_like(konumlar)))
    durum = MDState(positions=konumlar, velocities=hizlar, masses=kutleler)

    if tur == "harmonic":
        # `center` verilmezse orijin; bilinmeyen anahtar artık SESSİZCE
        # yutulmaz (eskiden yalnız `k` ve `center` okunuyordu, `kk:` yok sayılıp
        # ham `KeyError: 'k'` geliyordu). fcc dalında `sigma` KAFESİ kurar
        # (yukarıda tüketildi) ve `box_length` fabrikanın kendi eklemesidir:
        # ikisi de yaya GEÇMEZ -- HEAD de geçirmiyordu. Doğrulama turu
        # (11.10.2026, ölçüldü): onlar geçince harmonic + fcc + {k, sigma}
        # rc = 1 "unexpected keyword argument 'sigma'" veriyor ve mesaj
        # kullanıcının yazmadığı `box_length`'i "Verilen" diye gösteriyordu.
        yay_p = dict(yaml_p)
        if fcc:
            yay_p.pop("sigma", None)
        alan = _senaryo_nesnesi(
            Harmonic, "harmonic", {"center": np.zeros_like(konumlar), **yay_p}, yaml_p
        )
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
        alan = _senaryo_nesnesi(Morse, "morse", p, yaml_p)
    elif tur == "lennard_jones":
        alan = _senaryo_nesnesi(LennardJones, "lennard_jones", p, yaml_p)
    elif tur == "hartree_fock":
        # dt kararlılık denetimi BU DALDA YOK: omega analitik değil.
        # Atlandığı _HF_YAKLASIMLAR'da beyan edilir (sessiz atlama değil).
        alan = _senaryo_nesnesi(HartreeFockGradient, "hartree_fock", p, yaml_p)
    elif tur == "tb_ss_sigma":
        # dt kararlılık denetimi BU DALDA DA YOK (spec B10): omega analitik
        # değil. Atlandığı _TB_YAKLASIMLAR'da beyan edilir. Koşu reddedilmez;
        # kare-0 ve yol karelerinin omega*dt'si NVE panelinin hükmüne girer
        # (D2-i `omega_max_frame0`, D2-vi `omega_max_path_frames`). Kutusu
        # `box_x_bohr`dir: fcc'nin eklediği `box_length` ona geçmez (geçseydi
        # "bilinmeyen" listesinde kullanıcının yazmadığı anahtar görünürdü).
        alan = _tb_ss_sigma_alani(yaml_p)
    else:
        raise ScenarioError(f"bilinmeyen kuvvet alanı: {tur!r}")

    termostat = None
    if scenario.thermostat is not None:
        t_tur = scenario.thermostat["kind"]
        t_p = dict(scenario.thermostat.get("params", {}))
        if t_tur == "langevin":
            termostat = _senaryo_nesnesi(Langevin, "langevin", t_p)
        elif t_tur == "nose_hoover":
            termostat = _senaryo_nesnesi(NoseHoover, "nose_hoover", t_p)
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
#: MONOTON DEĞİLDİR (bkz. `energy_drift`, SINIR) ve iri dt'nin salınımını
#: yargılamaz: D2-ii ön ölçümünde iri-dt sınıfının 105 hücresinden 49'u bu
#: eşiğin altındaydı. D2-ii'den (10.10.2026) beri HF NVE'de o yarıyı S ve
#: omega*dt yargılar (`_HF_OSCILLATION_TOLERANCE`, `_HF_OMEGA_DT_TOLERANCE`).
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
#:   Bu eşik TEK BAŞINA dt'nin yeterince küçük olduğunu ONAYLAMAZ (spec
#:   B10: dt'yi kullanıcı seçer). D2 (09.10.2026): o yarıyı
#:   `_TB_OSCILLATION_TOLERANCE` yargılar, S'nin seyreltildiği koşuları
#:   `_TB_OMEGA_DT_TOLERANCE` (D2-i, D2-vi); TB NVE'de Yakınsama dördünün
#:   VE'sidir.
#:   (Bu 33 hücrenin ekseni M11 minimumunun omega_max'ıdır; D2
#:   ön ölçümünde pencerelerin kendi yol boyu omega_max'ı D ve Q
#:   ailelerinde 1.2-1.6 kat büyük, U/U2'de 0.86-1.20, metalik W3'te
#:   0.87-0.88 kat -- yani "33 hücre" ile salınım eşiğinin sınıfları AYNI
#:   eksende değildir; U 1000 a.u. dt 40 hücresi kendi ekseninde gri, 0.83.)
_TB_DRIFT_TOLERANCE: float = 1.0e-02

#: TB dalının NVE SALINIM eşiği (D2, 09.10.2026). Ölçüt `energy_oscillation`:
#: S = [max(KE + Omega) - min(KE + Omega)] / max(KE), uçlar integratörün HER
#: adımından. ÖLÇÜLDÜ (D2 ön ölçümü wf_e71a1dcb-1dc, bu makine; düz aileler,
#: N = 16, 871 hücre; EKSEN: her pencerenin KENDİ yol boyu omega_max'ı,
#: omega_ref): omega_ref*dt <= 0.463 sınıfının en büyük S'i 3.56e-02,
#: omega_ref*dt >= 0.925 sınıfının en küçüğü 7.29e-02 -- 2.05 katlık bir
#: ayırma bandı; 5e-02 onun geometrik ortasına (0.051) yakın. Bu kümede
#: sahte ✗ 0/405, sahte ✓ 0/235 (Fermi ve U0 aileleri hariç); G5'in
#: sürüklenme eşiğinden geçen 8 iri-dt hücresinin 8'i burada ✗. 1e-01
#: eşiği bu kümenin iri 235 hücresinden 3'ünü (en küçüğü S 7.29e-02) ve
#: G5'in U 1000 a.u. dt 40 hücresini (S 7.66e-02) kaçırırdı. Sevk edilen
#: `tb_zincir_md` S = 1.0054e-02 (5x pay). Küçük dt'de (dt 2.5-10,
#: omega_ref*dt <~ 0.3) S temiz O(dt^2) ölçeklenir (pencere başına log-log
#: eğim medyanı 2.00-2.01; Fermi sıçramalı pencere sapar); iri dt'de
#: S/(omega*dt)^2 büyür ve S dt ile monoton bile olmayabilir.
#:
#: BANT, ÖLÇÜLEN AİLELERİN DEĞERİDİR, SINIR DEĞİL: tek bir harmonik modda
#: Verlet'in S'i analitik x^2/(4 - x^2)'dir (x = omega*dt; T8'in S = 1.0'ı
#: x = sqrt(2)); eşik 0.05 orada x = 0.436'ya düşer. Doğrulamada (D2,
#: 09.10.2026) yalnız en yüksek modu uyarılmış Peierls minimumu yol
#: omega_max*dt 0.441'de ✗ verdi (S 5.086e-02, analitikle aynı).
#:
#: GRİ BÖLGE (ölçülen ailelerde 0.463 < omega_ref*dt < 0.925): S = 0.05'in
#: hangi omega*dt'ye düştüğü pencereye ve tohuma bağlıdır (S/(omega*dt)^2
#: küçük sınıfta 0.074-0.20); hükmü tohum, aile, başlangıç ve koşu uzunluğu
#: değiştirir. Eşik orada bir sınırın yeri DEĞİLDİR.
#:
#: SEYRELME: payda TÜM kinetik enerjidir; titreşime katılmayan ya da az
#: katılan KE S'yi seyreltir (`energy_oscillation`, KÖR AİLELER). Onu S
#: değil, KE'nin dağılımına bakmayan omega*dt tanısı yakalar
#: (`_TB_OMEGA_DT_TOLERANCE`, D2-i ve D2-vi).
_TB_OSCILLATION_TOLERANCE: float = 5.0e-02

#: TB NVE'nin omega*dt TANISININ eşiği (D2-i 10.10.2026, D2-vi aynı gün).
#: AYNI eşik iki yere uygulanır: kare-0 geometrisinin (`omega_max_frame0`,
#: koşudan ÖNCE) ve yolun izleyicinin seçtiği karelerinin
#: (`omega_max_path_frames`, koşudan SONRA) TAM 3N kütle ağırlıklı
#: Hessian'ından (kuvvetin merkezi farkı, h = 1e-4) en hızlı mod; tanı
#: omega*dt. NEDEN: S'nin paydası tüm KE'dir ve titreşime katılmayan KE onu
#: seyreltir -- kamu yolunda ayrışan iki zincir (yol omega_max*dt 1.09) S
#: 0.021 ✓, kütle safsızlığı (omega*dt 1.0) S 0.036 ✓ basıyordu. omega*dt
#: KE'nin nereye dağıldığına bakmaz.
#:
#: D2-i (yalnız kare 0, eşik 0.54): D2 bandının 1170 hücresi S'leri bit bit
#: aynı yeniden üretilerek, sınıf ekseni her pencerenin yol boyu omega_ref'i
#: -- düz 871 hücrede küçük-dt sınıfının (omega_ref*dt <= 0.463) en büyük
#: omega0*dt'si 0.430, iri sınıfın (>= 0.925) en küçüğü 0.682, geometrik orta
#: 0.542. Bilinen kör ailelerde D2 panelinin sahte ✓ verdiği 24 iri-dt
#: hücresinin 19'unu yakaladı. KALAN 5'i kare 0'ın YOLU görmemesiydi:
#: çarpışan parçalar (kare 0 yalnız parça içi modları görür, omega0*dt
#: 0.405-0.470, yol boyu 0.93-1.0) ve 6000 K ısıl hız (0.533, yol 0.930).
#:
#: D2-vi: YOL KARELERİ. Koşu, alanı saran `YolKaresiIzleyici` ile yapılır;
#: o HER ADIMDA iki kare seçer -- en kısa çift mesafesi ve kütle ağırlıklı
#: köşegen sertlik vekilinin tepesi -- ve koşudan sonra o karelerde aynı
#: Hessian alınır. ÖLÇÜLDÜ (D2-vi ön ölçümü ve eleştirmeni, bu makine):
#: * Adaylar aynı 1170 + 121 panel hücresinde yan yana: yalnız SON kare
#:   (önceki adayın) çarpışıp AYRILAN parçalarda kendi eşiğinde (0.57) D2
#:   panelinin sahte ✓ verdiği 13 hücrenin 10'unu kaçırdı -- son kare olayı
#:   değil, olaydan sonrasını görür; 5/9 eşit aralıklı kare kısa olayı
#:   aralarında kaçırdı (koşu 4 kat uzayınca 9 kare de); yalnız kayıtlı
#:   karelerden seçim save_every 50'de 37 hücrenin 3'ünü kaçırdı. Yalnız en
#:   kısa mesafe kütleyi görmez: içinde H olan ağır parçalar çarpışırken en
#:   kısa mesafe ağır çiftlerde, H'nin yerel modu başka anda tepe yapar
#:   (seçilen karenin omega'sı yol boyununkinin 0.425'i; 6 hücre kaçtı).
#:   İki seçici birlikte: seçilen karelerin omega'sı yol boyununkinin en az
#:   0.973'ü (63 hücre, yol boyu = koşunun HER karesi).
#: * Eşik bu tanının KENDİ bandından, S'nin ve D2-i'nin yöntemiyle: düz 871
#:   hücrede küçük-dt sınıfının en büyük omega*dt'si 0.478, iri sınıfın en
#:   küçüğü 0.930; geometrik orta 0.667 -> 0.67 (kenarlar sürekli dt'de
#:   alınırsa 0.6875; 0.67 ile 0.68 aynı sayıları verir). 0.67'de: küçük-dt
#:   sınıfında ✗ 0/552 (U0 dahil en büyük 0.552), iri sınıfın 302'sinin
#:   302'si tek başına, gri bölgede S'nin ✓ verdiği 53 hücrenin 4'ü ✗
#:   (D2-i'de 9'u). Bilinen kör ailelerde (D2 şüphecisi ve doğrulaması, D2-i
#:   ve D2-vi eleştirmenleri: kütle safsızlığı 12/40 amu, ağır parça + H,
#:   karışık kütleli gaz -- kare 0'da omega0 = 0 --, çarpışan ve çarpışıp
#:   ayrılan parçalar, yüksek T_ion, zikzak; kamu yolunun `_run_md`ıyla) D2
#:   panelinin sahte ✓ verdiği 95 iri-dt hücresinden D2-i 40'ını kaçırıyordu
#:   (eşik 0.54); bu tanı HİÇBİRİNİ kaçırmaz (en küçük omega*dt 0.723).
#:   Eşiği 0.67'ye çıkarmak D2-i'nin yakaladığı 55 hücreden hiçbirini
#:   kaçırmaz: omega0*dt'si 0.54-0.67 arasındaki 22'sinde yol kareleri en az
#:   0.774.
#: * YAPISAL GÜVENCE KOŞUNUN KENDİ YOLUNA GÖREDİR: seçilen kare
#:   integratörün bir karesidir, yani omega <= koşunun yol boyu omega_max'ı.
#:   Sınıf ekseni omega_ref ise dt 2.5 referans koşusunun ÖRNEKLENMİŞ
#:   karelerinden gelir; yuvarlamadan büyüyen kararsızlıkta (gürültüsüz
#:   düzgün zincir) koşunun yolu onu 1.25 kata kadar aşar.
#:
#: KÖRLÜĞÜ (`_TB_SALINIM_BEYANI`, KALAN KÖRLÜK): tanı yolun en çok üç
#: karesine bakar. En hızlı mod bu karelerde değilse (sertliği çift
#: mesafesinden ya da köşegenden okunmayan, ör. elektronik ya da toplu bir mod
#: başka bir anda tepe yaparsa) VE S seyreltilmişse ✓ dt'yi kanıtlamaz. Sertlik
#: vekili ss-sigma'nın iki üstel yasasına özgüdür. Ölçülen Fermi penceresinde
#: omega(t) sivri uç vermedi (en büyük / medyan 1.10). BEDEL: kare başına 6N
#: kuvvet çağrısı (N 16'da ~100 MD adımı, N 32'de ~190, N 64'te ~380), en çok
#: üç Hessian (iki seçici aynı kareyi seçerse iki) ve adım başına izleme
#: (~%10); sevk edilen senaryonun CLI süresi (N 16, 400 adım, yorumlayıcı
#: açılışı dahil) 0.76 s'den 0.90 s'ye.
_TB_OMEGA_DT_TOLERANCE: float = 0.67

#: TB NVE "Yakınsama tanısı" satırının SABİT yarısı (D2). Canlı yarı
#: (`_tb_yakinsama_tanisi`) ölçütlerin bu koşudaki hükmünü söyler; bu yarı
#: hükmün nerede bir şey söylemediğini. `diagnostics`e konmaz: ✓/✗'nin
#: hemen ardına basılırdı (`Provenance.convergence_diagnosis` gerekçesi).
_TB_SALINIM_BEYANI = (
    "GRİ BÖLGE: ölçülen ailelerde omega_max*dt kabaca 0.46 ile 0.93 arasında "
    "(her pencerenin kendi yol boyu omega_max'ıyla) salınım hükmünü tohum, "
    "aile, başlangıç ve koşu uzunluğu değiştirir; alt kenar aileye bağlıdır "
    "(tek harmonik modda S = x^2/(4 - x^2), eşik x = 0.436'da). BÜYÜME "
    "EVRESİ: kararsızlığın pencere içinde büyüdüğü koşuda ✗ 'korunum "
    "doğrulanamadı' diye okunur, 'dt iri' diye değil. YUVARLAMA TABANI: max KE "
    "yuvarlama düzeyindeyse S gürültüdür ve dt'yi küçültmek çare değildir. KÖR "
    "AİLELER: payda TÜM kinetik enerjidir; titreşime katılmayan ya da az "
    "katılan her KE S'yi seyreltir -- ayrışan parçaların ötelemesi, kütle "
    "safsızlığı, yalnız düşük modun uyarıldığı koşu, yüksek T_ion (ön ölçümde "
    "3000-6000 K), enine yumuşak modlu geometri, sonradan çarpışan parçalar. "
    "OMEGA*DT TANISI onları yakalar: kare 0'ın ve yolun her adımda izlenen iki "
    "karesinin (en kısa çift mesafesi, kütle ağırlıklı sertlik tepesi) en "
    "hızlı kütle ağırlıklı modu için omega*dt eşiğin üstündeyse ✗ (panelin "
    "sahte ✓ verdiği ölçülmüş 95 iri-dt hücresinin hepsi). KALAN KÖRLÜK: tanı "
    "yolun en çok üç karesine bakar -- en hızlı mod bu karelerde değilse "
    "(sertliği çift mesafesinden ya da köşegenden okunmayan, ör. elektronik ya "
    "da toplu bir mod başka bir anda tepe yaparsa) VE S seyreltilmişse ✓ "
    "dt'nin yeterince küçük olduğunu kanıtlamaz. Güvence koşunun kendi yoluna "
    "göredir. Liste kapsayıcı değildir."
)


def _tb_yakinsama_tanisi(
    suruklenme_gecti: bool,
    salinim_gecti: bool,
    salinim_sonlu: bool,
    kare0_gecti: bool,
    kare0_sonlu: bool,
    yol_gecti: bool,
    yol_sonlu: bool,
) -> str:
    """TB NVE "Yakınsama tanısı": dört ölçütün AYRI hükmü + sabit beyan.

    Hükümler `provenance_for`da BİR KEZ hesaplanır ve buraya geçer -- eşik
    karşılaştırması iki yerde yazılsaydı biri kayınca panel hükmüyle
    açıklaması ayrışırdı.
    """
    if suruklenme_gecti and salinim_gecti and kare0_gecti and yol_gecti:
        canli = (
            "sürüklenme, salınım, kare-0 omega0*dt ve yol karelerinin "
            "omega*dt'si dördü de eşiğin altında: KE + Omega bu koşuda ne "
            "sürükleniyor ne iri salınıyor; dt başlangıç geometrisinin ve "
            "yolun seçilen iki anının (en kısa çift mesafesi, sertlik tepesi) "
            "en hızlı modu için de eşiğin altında."
        )
    else:
        parcalar = []
        if not salinim_sonlu:
            parcalar.append(
                "salınım YARGILANAMAZ (S sonlu değil: max KE = 0, uçlar "
                "biriktirilmedi ya da enerji sonlu değil)"
            )
        elif not salinim_gecti:
            parcalar.append(
                "salınım eşiğin üstünde: dt bu hareket için iri olabilir, "
                "yörünge bir büyüme evresinde olabilir ya da max KE yuvarlama "
                "düzeyindedir -- dt'yi küçültüp yeniden koşun; S dt^2 ile "
                "küçülmüyorsa sebep dt değildir"
            )
        if not kare0_sonlu:
            parcalar.append(
                "a priori tanı YARGILANAMAZ (kare-0 Hessian'ı sonlu değil)"
            )
        elif not kare0_gecti:
            parcalar.append(
                "kare-0 omega0*dt eşiğin üstünde: dt başlangıç geometrisinin "
                "en hızlı titreşimini çözmüyor -- S eşiğin altında olsa bile "
                "(titreşime katılmayan KE onu seyreltir) dt'yi küçültün"
            )
        if not yol_sonlu:
            parcalar.append(
                "yol karelerinin tanısı YARGILANAMAZ (konum ya da Hessian "
                "sonlu değil)"
            )
        elif not yol_gecti:
            parcalar.append(
                "yol karelerinin omega*dt'si eşiğin üstünde: dt yolun en "
                "sıkışık ya da en sert anındaki en hızlı titreşimi çözmüyor "
                "(kare 0 onu görmez: sonradan çarpışan parçalar, ısıl keşif) "
                "-- S eşiğin altında olsa bile dt'yi küçültün"
            )
        if not suruklenme_gecti:
            parcalar.append(
                "sürüklenme eşiğin üstünde ya da yargılanamaz: koşuyu uzatın "
                "ya da dt'yi küçültün (energy_drift, SINIR)"
            )
        canli = "korunum DOĞRULANAMADI -- " + "; ".join(parcalar) + "."
    return canli + " " + _TB_SALINIM_BEYANI


#: HF dalının NVE SALINIM eşiği (D2-ii, 10.10.2026). Ölçüt TB'ninkiyle AYNI
#: formül (`energy_oscillation`; potansiyel sütunu HF'de E): S = ptp(KE + E) /
#: max(KE), uçlar her adımdan. TB'nin 0.05'i burada SAHTE ✗ verir: tek
#: titreşim modunda Verlet'in S'i x^2/(4 - x^2)'dir (x = omega*dt), 0.05
#: x = 0.436'ya düşer ve küçük-dt sınıfının kenarındaki (x 0.463) HF
#: hücreleri S 0.050'de eşiği aşıyordu. ÖLÇÜLDÜ (D2-ii ön ölçümü,
#: `.superpowers/sdd/d2-olcum/D2ii/`, bu makine; EKSEN: her ailenin KENDİ yol
#: boyu omega_ref'i -- ince dt referans koşusunun örnek karelerinin ve en kısa
#: çift karesinin Hessian'ından, TB D2 konvansiyonu): 334 hücre, 16 aile (H2
#: STO-3G gerilmiş / sıkışık / bağ boyu hızlı, 6-31G, LiH, H2O; dönen H2; H2 +
#: H2 çarpışması) ve eleştirmenin kendi hücreleri (küçük genlik 1e-1 ... 1e-6
#: bohr, n 1-8, Rc'yi aşan yörünge, 6-31G, LiH):
#: * KÜÇÜK-dt TAVANI harmonik sınırdır: küçük genlikte x 0.463'te üç bazda S
#:   0.0568 (analitik 0.0566); ön ölçümün 0.0503'ü yalnız >= 0.05 bohr genlikli
#:   ailelerdendi. 8.5e-02'nin payı 1.49x; küçük sınıfta S'nin kendi sahte ✗'i
#:   yok. TB'nin 0.05'i her küçük genlikli HF koşusunda x 0.44-0.463'te aşılır
#:   (analitik eşik x 0.436) -- ön ölçümde 3/129.
#: * İRİ TABAN genlikle düşer (seyrelme OLMADAN da): iri genlikli anharmonik
#:   H2'de S tek mod analitiğinin 0.41-0.60'ı (x 0.925'te 0.147), Rc'yi aşan
#:   yörüngede x 0.94'te 0.093 (1.10x pay). Bant ~[0.057, 0.093], geometrik
#:   orta 0.073; 8.5e-02 bandın içinde ama SAHTE ✗'ten KAÇINMA yönünde -- HF'de
#:   iri dt'yi birincil yakalayan omega*dt'dir (aşağıda), S'nin işi değil.
#: * dt'Yİ TEK BAŞINA YARGILAYAMAZ: dönme ve parçaların ötelemesi S'yi
#:   seyreltir -- dönen H2'de iri sınıfın en küçük S'i 1.39e-04, H2 + H2
#:   çarpışmasında 7.9e-03; S tek başına iri-dt'nin 24/105 hücresini kaçırırdı.
#: * KENDİ KATKISI omega*dt'nin göremediğidir: yuvarlama düzeyindeki KE
#:   (dengede hareketsiz H2: bu makinede sürüklenme tam 0 ve D2-ii öncesi panel
#:   ✓, omega*dt 0.126 ✓, S 4.4e+07 ✗) ve bir Hessian karesinde görünmeyen
#:   korunum bozulması. Sevk edilen `h2_aimd` S = 4.113e-03 (20.7x pay).
_HF_OSCILLATION_TOLERANCE: float = 8.5e-02

#: HF NVE'nin omega*dt TANISININ eşiği (D2-ii). TB'nin tarifi (`_omega_max`:
#: TAM 3N kütle ağırlıklı Hessian, kuvvetin merkezi farkı, h = 1e-4) iki
#: kareye: kare 0 (`omega_max_frame0`, koşudan ÖNCE) ve yolun her adımda
#: izlenen EN KISA ÇİFT MESAFESİ karesi (`omega_max_path_frames`, koşudan
#: SONRA). ÖLÇÜLDÜ (D2-ii ön ölçümü, `_HF_OSCILLATION_TOLERANCE` ile aynı 334
#: hücre ve eksen):
#: * En kısa çift karesinin omega'sı her ailede yol boyununkinin KENDİSİ
#:   (örnek karelerle oran 1.0000, 16/16 aile; H2O ve H2 + H2 dahil). İki
#:   atomluda yapısaldır: sertlik sıkışmada artar.
#: * Kare 0 TEK BAŞINA AYIRAMAZ: gerilmiş başlangıç yolun en YUMUŞAK anıdır
#:   (omega0 yol tepesinin 0.37-0.86'sı), sıkışık başlangıç en sert anı --
#:   küçük sınıfın en büyük omega0*dt'si 0.463, iri sınıfın en küçüğü 0.353.
#: * K = max(kare 0, yol karesi): küçük sınıfın en büyüğü 0.465, iri sınıfın
#:   en küçüğü 0.919 (1.98 katlık bant), geometrik orta 0.654. TB'nin 0.67'si
#:   (`_TB_OMEGA_DT_TOLERANCE`, kendi bandından 0.667) bu bandın içinde (+%2.4;
#:   alt kenara 1.44x, üste 1.37x pay): AYNI fiziksel nicelik (en hızlı kütle
#:   ağırlıklı modun omega*dt'si) iki dalda tek sayıyla okunur. 0.67'de
#:   küçük-dt sahte ✗ 0/129, iri kaçan 0/105 (seyrelen 42'si dahil), gri
#:   bölgede ✗ 49/100. Sevk edilen `h2_aimd` kare-0 0.116, yol karesi 0.1356.
#: * BU AYRIM BÜYÜK ÖLÇÜDE YAPISALDIR (eleştirmen): sınıf ekseni omega_ref
#:   aynı seçicinin karesinden kurulur, iki atomluda K*dt ~ x_ref tanım
#:   gereği. Asıl kanıt seçicinin yol tepesini bulmasıdır: yoğun örneklemeyle
#:   He + H2 (doğrusal, T-biçimli, sıcak, duvara bastırılmış), H2 + H2
#:   (T-biçimli, yan yana) ve dönen H2O'da seçilen karenin omega'sı yoğun
#:   yol tepesinin 0.9999-1.0000'i. BAĞIMSIZ EKSENDE (bağ uzunluğunun ölçülen
#:   Verlet frekans hatasının harmonik eşdeğeri x_eff) de 0.67 hatasız: küçük
#:   sınıfın en büyük K'sı 0.643, iri sınıfın en küçüğü 1.000 -- ama alt
#:   kenara payı yalnız 1.04x: iri genlikli anharmonik titreşimde ✗ temkinlidir
#:   (x_eff / x_ref aile sabiti 0.68-0.93; bağımsız doğrulamada Rc'yi aşan
#:   genlikte 0.38-0.43: r0 2.8 bohr, yol omega*dt 0.754 ✗, x_eff 0.29).
#: * Kare-0 tanısı ölçülen hiçbir hücrede tek başına bir şey yakalamadı
#:   (izleyici kare 0'ı zaten görür; 334 + 9 sistemde omega0 <= yol karesi);
#:   kuramsal değeri en kısa çift olmayan sert bir çiftle BAŞLAYAN koşudur.
#:   Tutuldu (TB ile aynı panel); seçilen kare kare 0'ın kendisiyse Hessian
#:   ikinci kez alınmaz (`omega_max_path_frames(omega_frame0=)`).
#: * TÜR BAŞINA EN KISA AN (eleştirmenin karşı örneğinden): genel en kısa çift
#:   en sert çift olmayabilir -- soğuk H2 + sıcak F-H'de en kısa çift hep H-H,
#:   yolun en sert anı F-H'nin sıkışması; genel en kısa çiftin karesinin
#:   omega'sı yolunkinin en çok 0.675'iydi ve iri dt'de (F-H'nin x'i 0.955)
#:   kamu yolunda S 0.060 ✓, K 0.640 ✓ -> sahte ✓. İzleyici bu yüzden HF'de en
#:   kısa çifti ELEMENT ÇİFTİ TÜRÜ başına da izler; aynı kamu yolu hücresi
#:   `pair_type_steps=F-H@32 H-H@0`, yol karesi omega*dt 0.955 -> ✗.
#:   Seçilen kare yolun bir karesidir: sahte ✗ doğuramaz, yalnız yakalar;
#:   tek türlü sistemde (H2, H4) seçim öncekiyle aynı.
#: BEDEL: Hessian başına 6N kuvvet çağrısı (H2'de 12 ~ 0.1 s; H2O'da 18 ~ 11 s);
#: sevk CLI'si 2.19 s -> 2.39 s (x1.09, yorumlayıcı açılışı dahil); kısa
#: çok-atomlu koşuda koşunun katları (LiH 10 adım x3.6; H2O 4 adım x9.2 --
#: kare 0'ın Hessian'ının tekrarı kalktı ama H-H türünün karesi (adım 4) bir
#: Hessian ekledi, toplam yine 2); her ek element çifti türü en çok +1 Hessian
#: (F-H + H2, 43 adım: 2 dk 32 s -> 2 dk 39 s). Sondaların her merkez SCF'i
#: kararlılık analizi de yapar (alan bunu sabit yapar; H2'de ~+%12, boşa).
#: TB'nin kütle ağırlıklı sertlik vekili HF'de kurulamaz (yasa yok); kalan
#: körlük `_HF_SALINIM_BEYANI`nda.
_HF_OMEGA_DT_TOLERANCE: float = 0.67

#: HF dalının NVE `expected_error`ı (D2-ii). Ortak metnin "salınım bir hata
#: değildir" cümlesi HF'de artık yanlış: iri dt'nin salınımını S ve omega*dt
#: yargılar. Eşiklerin SAYISI metinde YAZILMAZ (tanılarda sabitin kendisinden
#: basılır; `_TB_BEKLENEN` ile aynı gerekçe).
_HF_BEKLENEN = (
    "velocity-Verlet simplektiktir: KE + E SÜRÜKLENMEZ, SALINIR ve salınım "
    "dt^2 ile küçülür (tek titreşim modunda S = x^2/(4 - x^2), x = omega*dt). "
    "energy_drift burada |blok[-1] - blok[0]| / max(KE)'dir (|blok[0]| HF'de "
    "elektronik enerjidir, max(KE)'nin onlarca-binlerce katı) ve korunumun "
    "bozulmasını -- ör. sonlu fark kuvvet gürültüsünü -- yakalar; iri dt'nin "
    "SALINIMINI ise ayrı ölçüt yargılar: energy_oscillation = tepe-tepe(KE + E) / "
    "max(KE), uçlar kayıtlardan değil her adımdan, eşiği "
    "energy_oscillation_tolerance. S'nin paydası tüm KE'dir ve titreşime "
    "katılmayan KE (dönme, parçaların ötelemesi) onu seyreltir; onu omega*dt "
    "tanısı yakalar: omega_dt_frame0 = kare-0 geometrisinin (koşudan ÖNCE) ve "
    "omega_dt_path_frames = yolun her adımda izlenen en kısa çift mesafesi "
    "karesinin ve element çifti türü başına en kısa anlarının (koşudan SONRA) "
    "en hızlı kütle ağırlıklı modu x dt, ikisinin eşiği omega_dt_tolerance. "
    "Yakınsama dördünün VE'sidir. Hareketsiz koşu (max KE = 0) YARGILANAMAZ ve "
    "✗ basılır."
)

#: HF NVE "Yakınsama tanısı" satırının SABİT yarısı (D2-ii); TB'nin
#: `_TB_SALINIM_BEYANI`nın HF karşılığı. Canlı yarı `_hf_yakinsama_tanisi`.
_HF_SALINIM_BEYANI = (
    "GRİ BÖLGE: ölçülen HF ailelerinde omega*dt kabaca 0.46 ile 0.93 arasında "
    "(her ailenin kendi yol boyu omega_max'ıyla, yani yolun EN SERT anına göre) "
    "salınım hükmünü genlik, aile ve koşu uzunluğu değiştirir; tek titreşim "
    "modunda S = x^2/(4 - x^2), anharmonik genlikte bunun 0.4-0.9 katı, genlik "
    "Rc'yi aşınca daha da az. EKSEN TEMKİNİ: omega*dt yolun en sert anını okur; "
    "iri genlikli anharmonik titreşimde ✗'in karşılığı olan gerçek frekans "
    "hatası, harmonik eşdeğeri 0.68-0.93 kat daha küçük bir x'tir (genlik Rc'nin "
    "altında); Rc'yi aşan genlikte 0.38-0.43 kata kadar (ölçüldü) -- orada ✗ "
    "fiziksel olarak küçük bir frekans hatasına karşılık gelebilir. YUVARLAMA "
    "TABANI: max KE yuvarlama düzeyindeyse (ör. dengede hareketsiz molekül; "
    "ölçülen x 0.3'te genlik ~3e-7 bohr ve altı; 1e-6 civarında S şişer) S "
    "gürültüdür ve ✗ basar; dt'yi küçültmek çare değildir. KÖR AİLELER: payda TÜM kinetik enerjidir; dönme ve "
    "parçaların ötelemesi S'yi seyreltir (ölçülen dönen H2'de bin kata kadar). "
    "OMEGA*DT TANISI onları yakalar: kare 0'ın ve yolun her adımda izlenen en "
    "kısa çift mesafesi karesinin -- element çifti türü başına da: farklı bağ "
    "türlerinde en kısa çift en sert çift değildir (ölçüldü: soğuk H2 + sıcak "
    "F-H'de genel en kısa çiftin karesi yolun omega'sının en çok 0.675'iydi) "
    "-- en hızlı kütle ağırlıklı modu için omega*dt eşiğin üstündeyse ✗ "
    "(ölçülmüş 105 iri-dt hücresinin ve F-H karşı örneğinin hepsi). KALAN "
    "KÖRLÜK: tanı yolun yalnız seçilen karelerine bakar ve seçici MESAFEYE "
    "bakar -- en hızlı mod bunların hiçbirinde değilse (bir türün en kısa anı o "
    "türün en sert anı değilse: aynı türden birden çok bağın toplu modu, açısal "
    "sertlik; ölçülen ailelerde örnek yok) VE S seyreltilmişse ✓ dt'nin "
    "yeterince küçük olduğunu kanıtlamaz; TB'nin kütle ağırlıklı sertlik "
    "vekili HF'de kurulamaz (yasa yok). SÜRÜKLENME: kısa koşuda blok ölçütü salınımı "
    "sürüklenme sanar (ölçüldü: 2 periyotluk koşuda küçük dt'de de ✗; 5 adım "
    "ve altında sürüklenme S'ye eşit) ve save_every'ye bağlıdır. Güvence "
    "koşunun kendi yoluna göredir. Liste kapsayıcı değildir."
)


def _hf_yakinsama_tanisi(
    suruklenme_gecti: bool,
    salinim_gecti: bool,
    salinim_sonlu: bool,
    kare0_gecti: bool,
    kare0_sonlu: bool,
    yol_gecti: bool,
    yol_sonlu: bool,
) -> str:
    """HF NVE "Yakınsama tanısı": dört ölçütün AYRI hükmü + sabit beyan
    (D2-ii). `_tb_yakinsama_tanisi` ile aynı yapı, HF'nin metni: korunan
    büyüklük KE + E, yol karesi yalnız en kısa çift mesafesi; Fermi ve
    sertlik vekili yok. Hükümler `provenance_for`da bir kez hesaplanır."""
    if suruklenme_gecti and salinim_gecti and kare0_gecti and yol_gecti:
        canli = (
            "sürüklenme, salınım, kare-0 omega0*dt ve yol karesinin "
            "omega*dt'si dördü de eşiğin altında: KE + E bu koşuda ne "
            "sürükleniyor ne iri salınıyor; dt başlangıç geometrisinin ve "
            "yolun en kısa çift mesafesi anlarının (genel ve element çifti "
            "türü başına) en hızlı modu için de eşiğin altında."
        )
    else:
        parcalar = []
        if not salinim_sonlu:
            parcalar.append(
                "salınım YARGILANAMAZ (S sonlu değil: max KE = 0, uçlar "
                "biriktirilmedi ya da enerji sonlu değil)"
            )
        elif not salinim_gecti:
            parcalar.append(
                "salınım eşiğin üstünde: dt bu titreşim için iri olabilir ya "
                "da max KE yuvarlama düzeyindedir -- dt'yi küçültüp yeniden "
                "koşun; S dt^2 ile küçülmüyorsa sebep dt değildir"
            )
        if not kare0_sonlu:
            parcalar.append(
                "a priori tanı YARGILANAMAZ (kare-0 Hessian'ı sonlu değil; "
                "sondası geometri kapısına ya da SCF yakınsamamasına "
                "takıldıysa da -- koşu yine yapıldı)"
            )
        elif not kare0_gecti:
            parcalar.append(
                "kare-0 omega0*dt eşiğin üstünde: dt başlangıç geometrisinin "
                "en hızlı titreşimini çözmüyor -- S eşiğin altında olsa bile "
                "(dönme ya da öteleme KE'si onu seyreltir) dt'yi küçültün"
            )
        if not yol_sonlu:
            parcalar.append(
                "yol karesinin tanısı YARGILANAMAZ (konum ya da Hessian "
                "sonlu değil; sonda kapıya ya da SCF'e takıldıysa da)"
            )
        elif not yol_gecti:
            parcalar.append(
                "yol karesinin omega*dt'si eşiğin üstünde: dt yolun en "
                "sıkışık anındaki (genel ya da bir element çifti türünün) en "
                "hızlı titreşimi çözmüyor (kare 0 onu görmez: gerilmiş "
                "başlangıç, sonradan çarpışan parçalar) -- S eşiğin altında "
                "olsa bile dt'yi küçültün"
            )
        if not suruklenme_gecti:
            parcalar.append(
                "sürüklenme eşiğin üstünde ya da yargılanamaz: koşuyu uzatın, "
                "save_every'yi ya da dt'yi değiştirip yeniden bakın "
                "(energy_drift, SINIR)"
            )
        canli = "korunum DOĞRULANAMADI -- " + "; ".join(parcalar) + "."
    return canli + " " + _HF_SALINIM_BEYANI


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
    minimumunun en yüksek x-modu; öteki ailelere de bu sayı uygulandı ve
    orada YAKLAŞIK bile değil: D/U/Q pencerelerinin kendi yol boyu
    omega_max'ı 1.2-1.6 kat büyük -- D2 ön ölçümü, 09.10.2026).

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
    henüz büyümemiş) bu oran HER ŞEY olabilir, sıfır dahil: ÖLÇÜLDÜ (D2,
    09.10.2026) -- bu makinede (numpy 2.5.1'in OpenBLAS'ı, SkylakeX
    çekirdeği) kamu yolunda o zincirde dt 10 x 80'de blok farkı tam 0.0
    çıkıyor ve sürüklenme tek başına ✓ derdi. "Tam 0" bir YUVARLAMA
    RASTLANTISIDIR: OPENBLAS_CORETYPE ile zorlanan Haswell / Sandybridge /
    Katmai çekirdeklerinde ve zincir 0.37 bohr ötelenince sürüklenme kendisi
    ✗ veriyor (doğrulama, aynı gün). TB NVE'de o koşuyu her durumda bir
    ölçüt yakalar: burada salınım hükmü (yuvarlama düzeyindeki max KE'ye
    bölünen tepe-tepe farkı, ✗).
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


def energy_oscillation(scenario: MDScenario, trajectory: Trajectory) -> float | None:
    """TB ve HF NVE enerji SALINIMI S; öteki koşularda `None` (hesaplanmaz).

    S = [max_t(KE + Omega) - min_t(KE + Omega)] / max_t(KE). Uçlar
    integratörün HER ADIMINDAN gelir (`Trajectory.total_min`, `total_max`,
    `kinetic_max`; adım 0 dahil), kayıtlı karelerden DEĞİL: ÖLÇÜLDÜ (D2 ön
    ölçümü, 09.10.2026) -- karelerden okunan S save_every ile hem alta hem
    üste kayıyor, çünkü paydadaki max(KE) de seyreltiliyor (sevk
    geometrisi, dt 40 x 75, save_every 20: kareler 0.039, her adım 0.278 --
    S'nin kendi eşiğinde sahte ✓; panel aynı karelerin sürüklenmesiyle,
    2.07e-02, yine ✗ derdi). Pay ve payda `energy_drift`in TB dalıyla aynı
    ölçekte; tepe-tepe farkı enerjinin keyfi sıfırından bağımsızdır (spec B8).

    NE YARGILAR: iri dt'nin Verlet salınımını -- sürüklenme ölçütünün kör
    olduğu yarıyı (`_TB_DRIFT_TOLERANCE`, 33 hücrenin 8'i). Eşik ve ölçüm
    `_TB_OSCILLATION_TOLERANCE`ta. Kararsızlığın pencere içinde büyüdüğü
    koşuda ✗ "korunum DOĞRULANAMADI" diye okunur, "dt iri" diye değil.

    KÖR AİLELER (beyan: `_TB_SALINIM_BEYANI`): payda TÜM kinetik enerjidir;
    titreşime katılmayan ya da az katılan her KE S'yi seyreltir. Kamu
    yolunda ölçülenler: ayrışan iki zincir (titreşim T1 hücresininki, yol
    omega_max*dt 1.09) S = 0.021 ✓, öteleme yokken aynı çift ✗ (doğrulama,
    09.10.2026); kütle safsızlığı (15 x 12 amu + tek H) omega*dt = 1.0'da
    S = 0.036 ✓; ön ölçümde yalnız en düşük mod uyarılmışken omega*dt
    1.5'te S 0.0425, T_ion 3000-6000 K'de 0.045-0.050. Liste kapsayıcı değil.
    D2-i (10.10.2026): bu seyrelmeyi S DÜZELTMEZ -- payda değişmedi; panel
    onu omega*dt tanısıyla yakalar: kare 0 (`omega_max_frame0`) ve D2-vi'den
    beri yolun izlenen iki karesi (`omega_max_path_frames`; D2 panelinin sahte
    ✓ verdiği ölçülmüş 95 iri-dt hücresinin hepsi). Kalan körlük
    `_TB_OMEGA_DT_TOLERANCE`ta.

    Hareketsiz koşu (max KE = 0) ve uçları biriktirmeyen bir kurucu (`nan`,
    ör. `run.py`nin termostat döngüsü) `nan` döner, panel ✗ basar.

    YALNIZ `tb_ss_sigma` ve `hartree_fock` NVE: NVT'de toplam enerji tanım
    gereği korunmaz. HF dalı D2-ii'de (10.10.2026) girdi -- formül AYNI
    (potansiyel sütunu HF'de E, TB'de Omega), eşik HF'nin KENDİ bandından
    (`_HF_OSCILLATION_TOLERANCE`): TB'nin 0.05'i omega*dt 0.463'te HF/LiH/H2O
    hücrelerinde sahte ✗ veriyordu. Klasik dallar hükme alınmadı.
    """
    tur = scenario.force_field["kind"]
    if (tur == "tb_ss_sigma" or tur == "hartree_fock") and scenario.thermostat is None:
        olcek = trajectory.kinetic_max
        if not olcek > 0.0:  # 0 ve nan (karşılaştırma nan'da False)
            return float("nan")
        return float((trajectory.total_max - trajectory.total_min) / olcek)
    return None


def omega_max_frame0(
    scenario: MDScenario, state: MDState, force_field, step: float = 1.0e-4
) -> float | None:
    """TB ve HF NVE'de kare-0 geometrisinin en hızlı kütle ağırlıklı modu omega0;
    öteki koşularda `None` (hesaplanmaz).

    omega0 = sqrt(max(lambda_max, 0)), lambda_max = M^-1/2 H M^-1/2'nin en
    büyük özdeğeri; H TAM 3N Hessian, kuvvetin merkezi farkından (6N kuvvet
    çağrısı, adım `step` bohr) ve simetrikleştirilmiş. Panelin tanısı
    omega0*dt; eşik ve ölçüm `_TB_OMEGA_DT_TOLERANCE`ta.

    NE YARGILAR: S'nin seyreltildiği koşuları (envanter D2-i): S'nin paydası
    tüm KE'dir, omega0 ise KE'nin nereye dağıldığına bakmaz -- dt'nin
    başlangıç geometrisinin en hızlı titreşimini çözüp çözmediğini söyler.
    KOŞUDAN ÖNCE, `state`in konumlarında hesaplanır (COM hızı çıkarılmış ya
    da çıkarılmamış: konum aynı). Koşuyu REDDETMEZ: dt'yi kullanıcı seçer
    (spec B10); hüküm panelde.

    TAM 3N, kütle ağırlıklı: 1B zincirde enine modlar daha yumuşak olsa da
    zincir y ya da z boyunca uzanırsa yalnız x altuzayı en hızlı modu
    göremezdi; karışık kütlede tek bir kütleye bölmek safsızlığın yerel
    modunu yanlış ölçerdi. Negatif özdeğer (eyer) omega0'a girmez.

    KÖR: yol boyu omega'yı görmez -- kare 0 yolun en hızlı modunu temsil
    etmiyorsa (sonradan çarpışan parçalar, yüksek T_ion); yalnız serbest
    atomlardan oluşan bir başlangıç omega0 = 0 verir. O yarıyı D2-vi'den beri
    `omega_max_path_frames` (yolun izlenen iki karesi) yargılar; panel ikisini
    AYNI eşikle okur. Hessian sonlu değilse `nan` döner, panel ✗ basar
    (YARGILANAMAZ).

    HF NVE (D2-ii, 10.10.2026): aynı tarif, eşik `_HF_OMEGA_DT_TOLERANCE`.
    Hessian'ın 6N kuvvet çağrısı `_HFHessianSondasi` ile yapılır: alanın TAZE
    kopyası (koşunun alanı her merkez SCF'i kararlılık sayaçlarına ve
    `max_bag_bohr`a yazar -- B10 -- ve sondalar yolun kareleri DEĞİLDİR) ve
    sonda hatası (geometri kapısı, SCF yakınsamaması) `nan` -> YARGILANAMAZ.
    """
    tur = scenario.force_field["kind"]
    if (tur == "tb_ss_sigma" or tur == "hartree_fock") and scenario.thermostat is None:
        return _omega_max(state.positions, state.masses, _hessian_alani(force_field), step)
    return None


class _HFHessianSondasi:
    """HF Hessian'ının kuvvet sondası (D2-ii): alanın AYNI parametreli TAZE
    kopyası (`dataclasses.replace`: baz, gradyan adımı, geometri kapısı aynı,
    sayaçlar sıfır) ve sonda hatası -> `nan`.

    NEDEN KOPYA: HF alanı her çağrının merkez SCF'ini B10 kararlılık
    sayaçlarına (`kararlilik_cagri`, `en_kotu_ic` ...) ve `max_bag_bohr`a işler;
    panel o sayaçları KOŞUNUN adımları diye basar ve Hessian'ın 6N sondası
    onları kirletirdi (ölçüldü, D2-ii eleştirmeni: kopyasız +12 merkez SCF ve
    sondanın +1e-4'ü max_bag'de). Değer kopyayla bit bit aynı (SCF durumsuz).
    NEDEN `nan`: sonda koşunun yolunda DEĞİLDİR; geometri kapısının 1e-4
    yakınındaki bir kare (ölçüldü: r0 20.99995 bohr -- koşu geçer, sonda
    21.00005'te kapıya çarpar) ya da sondanın SCF'inin yakınsamaması koşuyu
    KOŞMADAN (kare 0) ya da koşu bittikten sonra (yol karesi; .npz kaybolurdu)
    düşürürdü. Tanı o zaman YARGILANAMAZ'dır (panel ✗), koşu sürer. Yalnız
    `QLabError` yakalanır; programlama hatası yükselir.
    """

    def __init__(self, force_field: HartreeFockGradient):
        self.alan = dataclasses.replace(force_field)

    def __call__(self, positions):
        try:
            return self.alan(positions)
        except QLabError:
            return float("nan"), np.full(np.shape(positions), np.nan)


def _hessian_alani(force_field):
    """Hessian sondaları için alan: `HartreeFockGradient`te `_HFHessianSondasi`
    (taze kopya + sonda hatası `nan`), öteki alanlarda alanın kendisi (TB'nin
    davranışı değişmedi)."""
    if isinstance(force_field, HartreeFockGradient):
        return _HFHessianSondasi(force_field)
    return force_field


def _omega_max(positions, masses, force_field, step: float) -> float:
    """Bir geometrinin en hızlı kütle ağırlıklı modu: sqrt(max(lambda_max, 0)),
    lambda_max = M^-1/2 H M^-1/2'nin en büyük özdeğeri; H TAM 3N Hessian,
    kuvvetin merkezi farkından (6N kuvvet çağrısı) ve simetrikleştirilmiş.
    Sonlu değilse `nan`. `omega_max_frame0` ve `omega_max_path_frames`
    AYNI tarifi kullanır -- iki kare aynı ölçütle karşılaştırılır."""
    konum = np.asarray(positions, dtype=float)
    n3 = konum.size
    hessian = np.empty((n3, n3))
    for k in range(n3):
        arti = konum.copy().ravel()
        arti[k] += step
        eksi = konum.copy().ravel()
        eksi[k] -= step
        kuvvet_arti = force_field(arti.reshape(konum.shape))[1].ravel()
        kuvvet_eksi = force_field(eksi.reshape(konum.shape))[1].ravel()
        hessian[:, k] = -(kuvvet_arti - kuvvet_eksi) / (2.0 * step)
    hessian = 0.5 * (hessian + hessian.T)
    olcek = 1.0 / np.sqrt(np.repeat(np.asarray(masses, dtype=float), 3))
    agirlikli = hessian * olcek[:, None] * olcek[None, :]
    if not np.all(np.isfinite(agirlikli)):
        return float("nan")
    return float(np.sqrt(max(float(np.linalg.eigvalsh(agirlikli)[-1]), 0.0)))


#: `YolKaresiIzleyici.tanilar`ın ve panelin yol karesi anahtarları (sıra panel sırasıdır).
_YOL_KARESI_TANILARI = ("closest_pair_step", "closest_pair_bohr", "stiffest_frame_step")
#: HF'nin yol karesi anahtarları (D2-ii): sertlik vekili yerine element çifti
#: türü başına en kısa anın adımı ("F-H@32 H-H@0" biçiminde, virgülsüz).
_HF_YOL_KARESI_TANILARI = ("closest_pair_step", "closest_pair_bohr", "pair_type_steps")


class YolKaresiIzleyici:
    """TB (ve HF) NVE'de kuvvet alanını SARAR ve yolun iki karesini SEÇER; dönüşü
    DEĞİŞTİRMEZ (alanın (enerji, kuvvet)'i aynen geçer, yörünge bit bit aynı).

    Her çağrıda iki seçici (x'te en yakın görüntü, kutu alanın kendi
    `kutu_x_bohr`u):
    * EN KISA ÇİFT MESAFESİ -- çarpışmanın, sıkışmanın anı;
    * KÜTLE AĞIRLIKLI KÖŞEGEN SERTLİK VEKİLİ, en büyüğü: atom başına
      s_i = sum_j [(us/d0)^2 |itme(d_ij)| + 2 (us/d0)^2 |ss_sigma(d_ij)|] / m_i
      (üstel yasanın ikinci türevi (us/d0)^2 x değer; 2 = spin dahil bağ
      mertebesinin üst sınırı). Mesafeyi görmeyen KÜTLEYİ görür: ağır
      parçalar çarpışırken içlerindeki H'nin yerel modu başka bir anda tepe
      yapar ve en kısa mesafe karesi onu kaçırıyordu (ölçüm:
      `_TB_OMEGA_DT_TOLERANCE`).
    Seçici yalnız HANGİ karenin Hessian'lanacağını seçer: hükmü yine o karenin
    TAM Hessian'ı verir (`omega_max_path_frames`), seçilen kare yolun bir
    karesidir -- vekil yanlışsa yakalama gücü düşer, sahte ✗ doğmaz.

    NEDEN HER ÇAĞRI: velocity-Verlet alanı adım başına TAM BİR KEZ çağırır
    (`integrator.velocity_verlet_step`), koşudan önce bir kez de kare 0 için;
    izleyici böylece HER ADIMI görür, kayıtlı kareleri değil -- kayıtlı
    karelerden seçim save_every büyüdükçe kısa olayı kaçırıyordu. Yalnız
    `integrator.run`a verilir: kare-0 Hessian'ının sondaları yolun kareleri
    DEĞİLDİR. İlk çağrı her durumda kaydedilir (tek atomda çift yok: kare 0
    seçilir); eşitlikte İLK kare kalır. Sonlu olmayan bir konum `sonlu`yu
    False yapar (tanı YARGILANAMAZ).

    HF NVE (D2-ii, 10.10.2026): `turler` (atomların element sembolleri)
    verilir -- sertlik vekili YOK (ss-sigma'nın iki üstel yasasından
    kuruluyordu; HF'de yasa yok), kutu yok (HF periyodik değil). Onun yerine
    en kısa çift mesafesi ELEMENT ÇİFTİ TÜRÜ başına ayrıca izlenir (`tur_kareleri`:
    "F-H", "H-H" ...). NEDEN: ölçülen H/He ailelerinde (iki atomlu, H2O, H2 +
    H2 ve He + H2 çarpışmaları) genel en kısa çift karesinin omega'sı yol
    boyununkinin kendisiydi, ama farklı bağ türlerinde DEĞİL -- soğuk H2 +
    sıcak F-H'de en kısa çift hep H-H, yolun en sert anı F-H'nin sıkışması;
    seçilen karenin omega'sı yolunkinin en çok 0.675'iydi ve iri dt'de panel
    sahte ✓ basıyordu (D2-ii eleştirmeni, kamu yolu). Türün en kısa anı o
    türün en sıkışık bağıdır. `stiffest_frame_step` HF panelinde yok,
    `pair_type_steps` var.
    """

    def __init__(self, force_field, masses, turler: tuple[str, ...] | None = None):
        self._alan = force_field
        self._kutle = np.asarray(masses, dtype=float)
        self._tur_maskeleri: dict[str, np.ndarray] = {}
        if turler is None:  # TB: alanın kutusu ve iki yasası
            self._kutu = force_field.kutu_x_bohr
            self._yasalar = tuple(
                (yasa, carpan * (yasa.us / yasa.d0_bohr) ** 2)
                for yasa, carpan in ((force_field.itme, 1.0), (force_field.ss_sigma, 2.0))
            )
            self._kesme = max(force_field.itme.r_cut_bohr, force_field.ss_sigma.r_cut_bohr)
        else:  # HF: en kısa çift (genel ve tür başına), kutusuz
            self._kutu, self._yasalar, self._kesme = None, (), 0.0
            semboller = [str(s) for s in turler]
            for i in range(len(semboller)):
                for j in range(i + 1, len(semboller)):
                    ad = "-".join(sorted((semboller[i], semboller[j])))
                    maske = self._tur_maskeleri.setdefault(ad, np.zeros((len(semboller),) * 2, dtype=bool))
                    maske[i, j] = True
        self.tur_kareleri: dict[str, list] = {}  # ad -> [mesafe, adım, konum]
        self.cagri = 0
        self.sonlu = True
        self.en_kisa = float("inf")
        self.en_kisa_adim: int | None = None
        self.en_kisa_konum: np.ndarray | None = None
        self.sertlik = float("-inf")
        self.sertlik_adim: int | None = None
        self.sertlik_konum: np.ndarray | None = None

    def __call__(self, positions):
        sonuc = self._alan(positions)
        adim = self.cagri
        self.cagri += 1
        konum = np.array(positions, dtype=float)  # KOPYA: çağıran diziyi değiştirse de kare korunur
        if not np.all(np.isfinite(konum)):
            self.sonlu = False
            return sonuc
        fark = konum[:, None, :] - konum[None, :, :]
        if self._kutu is not None:
            fark[..., 0] -= self._kutu * np.round(fark[..., 0] / self._kutu)
        uzaklik = np.sqrt((fark * fark).sum(axis=2))
        np.fill_diagonal(uzaklik, np.inf)
        en_kisa = float(uzaklik.min())
        if self.en_kisa_konum is None or en_kisa < self.en_kisa:
            self.en_kisa, self.en_kisa_adim, self.en_kisa_konum = en_kisa, adim, konum
        for ad, maske in self._tur_maskeleri.items():
            mesafe = float(uzaklik[maske].min())
            eski = self.tur_kareleri.get(ad)
            if eski is None or mesafe < eski[0]:
                self.tur_kareleri[ad] = [mesafe, adim, konum]
        if self._yasalar:
            yakin = uzaklik < self._kesme
            sertlik = np.zeros(len(konum))
            if yakin.any():
                for yasa, carpan in self._yasalar:
                    deger = np.zeros_like(uzaklik)
                    deger[yakin] = yasa.deger_ve_turev(uzaklik[yakin])[0]
                    sertlik += carpan * np.abs(deger).sum(axis=1)
            sertlik_tepe = float((sertlik / self._kutle).max())
            if self.sertlik_konum is None or sertlik_tepe > self.sertlik:
                self.sertlik, self.sertlik_adim, self.sertlik_konum = sertlik_tepe, adim, konum
        return sonuc

    def tanilar(self) -> dict:
        """Panelin `path_frames` tanıları: TB'de `_YOL_KARESI_TANILARI` (seçilen
        iki karenin adımı ve en kısa çift mesafesi, bohr); HF'de
        `_HF_YOL_KARESI_TANILARI` (`stiffest_frame_step` yerine tür başına en
        kısa anın adımı, "F-H@32 H-H@0")."""
        if self._yasalar:
            return dict(zip(_YOL_KARESI_TANILARI, (self.en_kisa_adim, self.en_kisa, self.sertlik_adim)))
        turler = " ".join(f"{ad}@{kare[1]}" for ad, kare in self.tur_kareleri.items()) or None
        return dict(zip(_HF_YOL_KARESI_TANILARI, (self.en_kisa_adim, self.en_kisa, turler)))

    def secilen_kareler(self) -> list[np.ndarray]:
        """Hessian'lanacak yol kareleri, adımına göre TEKİL ve sırayla: en kısa
        çift, (TB) sertlik tepesi, (HF) tür başına en kısa anlar."""
        kareler = {self.en_kisa_adim: self.en_kisa_konum}
        if self.sertlik_konum is not None:
            kareler.setdefault(self.sertlik_adim, self.sertlik_konum)
        for _, adim, konum in self.tur_kareleri.values():
            kareler.setdefault(adim, konum)
        return list(kareler.values())


def path_frame_tracker(scenario: MDScenario, force_field, masses) -> YolKaresiIzleyici | None:
    """TB ve HF NVE'de alanı saran `YolKaresiIzleyici` (HF'de sertlik vekili
    yerine element çifti türü başına en kısa an, D2-ii); öteki koşularda
    `None` (izlenmez -- koşu doğrudan alanla yapılır)."""
    tur = scenario.force_field["kind"]
    if scenario.thermostat is None:
        if tur == "tb_ss_sigma":
            return YolKaresiIzleyici(force_field, masses)
        if tur == "hartree_fock":
            return YolKaresiIzleyici(force_field, masses, turler=tuple(scenario.force_field["params"]["symbols"]))
    return None


def omega_max_path_frames(
    scenario: MDScenario,
    tracker: YolKaresiIzleyici | None,
    state: MDState,
    force_field,
    step: float = 1.0e-4,
    *,
    omega_frame0: float | None = None,
) -> float | None:
    """TB ve HF NVE'de izleyicinin seçtiği yol karelerinin (en kısa çift mesafesi,
    sertlik tepesi) en hızlı kütle ağırlıklı modunun EN BÜYÜĞÜ (`_omega_max`,
    kare 0 ile aynı tarif; iki seçici aynı kareyi seçtiyse tek Hessian);
    öteki koşularda `None`. Koşu BİTTİKTEN sonra, alanın KENDİSİyle (izleyici
    değil) çağrılır. İzleyici sonlu olmayan bir konum gördüyse, hiç
    çağrılmadıysa ya da bir Hessian sonlu değilse `nan` (panel ✗,
    YARGILANAMAZ). Panelin tanısı omega*dt; eşik ve ölçüm
    `_TB_OMEGA_DT_TOLERANCE`ta.

    HF NVE (D2-ii): yalnız en kısa çift karesi (izleyicinin sertlik vekili
    yok), Hessian `_HFHessianSondasi` ile (taze kopya, sonda hatası `nan`);
    eşik `_HF_OMEGA_DT_TOLERANCE`.

    `omega_frame0` (D2-ii): `omega_max_frame0()`ın AYNI `state` için değeri.
    Seçilen kare kare 0'ın kendisiyse (izleyicinin ilk çağrısı koşunun
    başlangıcıdır; konum bit bit `state.positions`) Hessian ikinci kez
    alınmaz, bu değer kullanılır -- sonuç bit bit aynı olurdu. ÖLÇÜLDÜ (D2-ii
    eleştirmeni): sıkışık başlangıçta ve kısa H2O koşusunda aynı karenin
    Hessian'ı iki kez alınıyordu (H2O 4 adımda iki Hessian'ın biri; tür
    izlemesi orada H-H karesi için yeni bir Hessian ekledi)."""
    tur = scenario.force_field["kind"]
    if (tur == "tb_ss_sigma" or tur == "hartree_fock") and scenario.thermostat is None:
        if tracker is None:
            raise InvalidInputError(
                f"{tur} NVE koşusunda yol karesi izleyicisi zorunludur: "
                "path_frame_tracker(scenario, alan, kütleler) ile koşun (D2-vi, D2-ii)"
            )
        if not tracker.sonlu or tracker.en_kisa_konum is None:
            return float("nan")
        alan = _hessian_alani(force_field)

        def kare_omega(konum):
            if omega_frame0 is not None and np.array_equal(konum, state.positions):
                return float(omega_frame0)
            return _omega_max(konum, state.masses, alan, step)

        # Seçilen kareler adımına göre tekil: iki seçici (ya da bir tür) aynı
        # kareyi seçtiyse tek Hessian.
        omegalar = [kare_omega(konum) for konum in tracker.secilen_kareler()]
        # max() nan'ı sırasına göre yutar: biri sonlu değilse sonuç nan.
        if not all(np.isfinite(omegalar)):
            return float("nan")
        return float(max(omegalar))
    return None


#: NVT (termostatlı) koşunun "Yakınsama tanısı" (C1/D3/D2-iv, 11.10.2026).
#: NVT'de enerji korunmaz, yani NVE'nin korunum hükmü burada YOKTUR; NVT'nin
#: kendi hükmü (kinetik sıcaklık hedefe oturdu mu, ⟨T⟩ hedefe karşı) ise bu
#: koşuda HİÇ HESAPLANMIYOR. Eskiden panel burada koşulsuz ✓ basıyordu (ör.
#: `argon_nvt`: energy_drift 3.46e-02'nin yanında ✓). Artık ✗ + bu tanı.
#: YARGILANMADI (= denenmedi) bilerek YARGILANAMAZ'dan (= denendi, sonuçsuz)
#: ayrı bir kelimedir. ⟨T⟩ ölçümleri (üç tohum) README'dedir -- ölçümdür,
#: hüküm değil; blok SEM'i tek koşuda hatayı küçük gösterdiği için gelecekteki
#: bir hüküm çok tohum ya da öz-ilinti tahmini ister.
_NVT_YARGILANMADI = (
    "YARGILANMADI: NVT'de toplam enerji korunmaz (banyo enerji alıp verir), "
    "bu yüzden korunum hükmü yoktur; kinetik sıcaklığın hedefe oturup "
    "oturmadığı (⟨T⟩ hedefe karşı) bu koşuda YARGILANMADI -- sıcaklık serisi "
    "npz'de (temperature_kelvin) durur ama ne ortalaması basılır ne hedefle "
    "karşılaştırılır; energy_drift bir tanı değeridir, hüküm değil"
)


def provenance_for(
    scenario: MDScenario,
    *,
    energy_drift: float,
    energy_oscillation: float | None = None,
    omega_max_frame0: float | None = None,
    omega_max_path_frames: float | None = None,
    path_frames: dict | None = None,
    n_constraints: int = 3,
    stability: StabilityVerdict | None = None,
    external_stability: StabilityVerdict | None = None,
    kararlilik_tanisi: dict | None = None,
) -> Provenance:
    """MD koşusunun şeffaflık bloğu.

    `energy_oscillation` (D2, 09.10.2026): `energy_oscillation()`ın S'i.
    TB NVE'de ve (D2-ii, 10.10.2026) HF NVE'de ZORUNLUDUR ve Yakınsama =
    (sürüklenme eşik altı) VE (S eşik altı) VE (kare-0 omega0*dt eşik altı)
    VE (yol karelerinin omega*dt'si eşik altı); verilmezse InvalidInputError --
    "yok say" sürüklenmenin kör olduğu iri-dt salınımına yeni bir sahte ✓
    kapısı açardı. Eşikler türe göre: TB `_TB_*`, HF `_HF_OSCILLATION_TOLERANCE`
    ve `_HF_OMEGA_DT_TOLERANCE`. Öteki koşularda (NVT, klasik) S hesaplanmaz
    ve verilirse yine hata: hükme girmeyen bir sayının panelde hüküm gibi
    durmaması için.

    `omega_max_frame0` (D2-i, 10.10.2026): `omega_max_frame0()`ın omega0'ı;
    S'yle AYNI kapı (TB NVE'de zorunlu, öteki koşularda yasak, negatif
    hata). Verilmezse S'nin seyreltildiği iri-dt koşusu yine ✓ alırdı.

    `omega_max_path_frames` (D2-vi): `omega_max_path_frames()`ın yol
    karelerindeki en büyük omega'sı; omega0'la AYNI kapı ve AYNI eşik
    (`_TB_OMEGA_DT_TOLERANCE`). Verilmezse kare 0'ın görmediği yol (sonradan
    çarpışan parçalar, ısıl keşif) yine ✓ alırdı. `path_frames`
    (`YolKaresiIzleyici.tanilar()`) yalnız tanıdır: seçilen karelerin adımı
    ve en kısa çift mesafesi; TB NVE'de isteğe bağlı, öteki koşularda yasak.

    NVT (termostatlı koşu; C1/D3/D2-iv, 11.10.2026): Yakınsama ✗ ve
    `convergence_diagnosis` = `_NVT_YARGILANMADI`. NVT'nin hükmü (⟨T⟩ hedefe
    oturdu mu) bu koşuda hesaplanmaz; ölçülmemiş bir hükmün yanına ✓
    basılmaz. Langevin ve Nose-Hoover aynı dalı alır (termostat türüne göre
    dallanma yok).

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
    # D2-ii (10.10.2026): HF NVE de AYNI dört ölçütle yargılanır (eşikleri
    # kendi bandından, `_HF_OSCILLATION_TOLERANCE`, `_HF_OMEGA_DT_TOLERANCE`).
    salinim_hukme_girer = (tur == "tb_ss_sigma" or tur == "hartree_fock") and termostat is None
    if salinim_hukme_girer and energy_oscillation is None:
        raise InvalidInputError(
            f"{tur} NVE koşusunda salınım ölçütü zorunludur: provenance_for(..., "
            "energy_oscillation=energy_oscillation(scenario, yorunge)); "
            "sürüklenme iri dt'nin salınımına kördür (D2, D2-ii)"
        )
    if not salinim_hukme_girer and energy_oscillation is not None:
        raise InvalidInputError(
            "energy_oscillation yalnız TB NVE ve HF NVE koşusunda hükme girer; bu koşu "
            f"({tur}, termostat {termostat['kind'] if termostat else None}) için None verin"
        )
    if energy_oscillation is not None and energy_oscillation < 0.0:
        # Tepe-tepe farkı negatif olamaz; `energy_oscillation()` bunu üretmez.
        # Kapı yoksa doğrudan API çağrısında negatif sayı ✓ alırdı (doğrulama).
        raise InvalidInputError(f"energy_oscillation negatif olamaz: {energy_oscillation}")
    # D2-i: a priori tanı S'yle AYNI kapıdan geçer -- verilmezse seyreltilmiş
    # S'nin iri-dt koşusu yine ✓ alırdı; hükme girmediği yerde verilirse hata.
    if salinim_hukme_girer and omega_max_frame0 is None:
        raise InvalidInputError(
            f"{tur} NVE koşusunda a priori tanı zorunludur: provenance_for(..., "
            "omega_max_frame0=omega_max_frame0(scenario, durum, alan)); S'nin "
            "paydası tüm KE'dir ve titreşime katılmayan KE onu seyreltir (D2-i)"
        )
    if not salinim_hukme_girer and omega_max_frame0 is not None:
        raise InvalidInputError(
            "omega_max_frame0 yalnız TB NVE ve HF NVE koşusunda hükme girer; bu koşu "
            f"({tur}, termostat {termostat['kind'] if termostat else None}) için None verin"
        )
    if omega_max_frame0 is not None and omega_max_frame0 < 0.0:
        # sqrt(max(lambda, 0)) negatif olamaz; `omega_max_frame0()` bunu üretmez.
        raise InvalidInputError(f"omega_max_frame0 negatif olamaz: {omega_max_frame0}")
    # D2-vi: yol karelerinin omega'sı kare 0'ınkiyle AYNI kapıdan geçer --
    # verilmezse kare 0'ın görmediği yol (çarpışma, ısıl keşif) yine ✓ alırdı.
    if salinim_hukme_girer and omega_max_path_frames is None:
        raise InvalidInputError(
            f"{tur} NVE koşusunda yol karelerinin tanısı zorunludur: "
            "provenance_for(..., omega_max_path_frames=omega_max_path_frames("
            "scenario, izleyici, durum, alan)); kare 0 yolun en hızlı modunu "
            "görmez (D2-vi, D2-ii)"
        )
    if not salinim_hukme_girer and (omega_max_path_frames is not None or path_frames is not None):
        raise InvalidInputError(
            "omega_max_path_frames ve path_frames yalnız TB NVE ve HF NVE koşusunda "
            f"hükme girer; bu koşu ({tur}, termostat "
            f"{termostat['kind'] if termostat else None}) için None verin"
        )
    # HF'nin izleyicisinde sertlik vekili yok (D2-ii): `stiffest_frame_step` TB'ye
    # özgü, HF'de onun yerine tür başına en kısa anlar (`pair_type_steps`).
    yol_anahtarlari = _YOL_KARESI_TANILARI if tur == "tb_ss_sigma" else _HF_YOL_KARESI_TANILARI
    if path_frames is not None and set(path_frames) - set(yol_anahtarlari):
        raise InvalidInputError(
            f"path_frames yalnız {yol_anahtarlari} anahtarlarını taşır: "
            f"{sorted(set(path_frames) - set(yol_anahtarlari))}"
        )
    if omega_max_path_frames is not None and omega_max_path_frames < 0.0:
        raise InvalidInputError(
            f"omega_max_path_frames negatif olamaz: {omega_max_path_frames}"
        )
    omega_dt = omega_dt_yol = omega_esigi = None  # NVT'de ve TB/HF dışında hükme girmez (0.0 DEĞİL)
    salinim_esigi = None  # NVT'de ve TB/HF dışında salınım hükme girmez (0.0 DEĞİL)
    yakinsama_tanisi = None
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
            salinim_esigi, omega_esigi = _TB_OSCILLATION_TOLERANCE, _TB_OMEGA_DT_TOLERANCE
            tani = _tb_yakinsama_tanisi
        elif tur == "hartree_fock":
            # D2-ii: ortak metnin "salınım bir hata değildir" cümlesi HF'de
            # artık yanlış -- iri dt'nin salınımını S ve omega*dt yargılar.
            beklenen = _HF_BEKLENEN
            salinim_esigi, omega_esigi = _HF_OSCILLATION_TOLERANCE, _HF_OMEGA_DT_TOLERANCE
            tani = _hf_yakinsama_tanisi
        yakinsadi = bool(np.isfinite(energy_drift)) and energy_drift <= esik
        if salinim_hukme_girer:
            # D2: sürüklenme iri dt'nin SALINIMINA kördür (33 hücrenin 8'i);
            # S onu yargılar. `nan` (hareketsiz koşu) -> ✗, YARGILANAMAZ.
            salinim_sonlu = bool(np.isfinite(energy_oscillation))
            salinim_gecti = salinim_sonlu and energy_oscillation <= salinim_esigi
            # D2-i: S seyreltilebilir (payda tüm KE); omega*dt KE'nin
            # dağılımına bakmaz. D2-vi: AYNI eşik iki yere -- kare 0 ve yolun
            # izleyicinin seçtiği kareleri (en kısa çift mesafesi, TB'de bir de
            # sertlik tepesi). `nan` (Hessian sonlu değil) -> ✗, YARGILANAMAZ.
            omega_dt = float(omega_max_frame0 * scenario.dt)
            omega_dt_yol = float(omega_max_path_frames * scenario.dt)
            kare0_sonlu = bool(np.isfinite(omega_dt))
            kare0_gecti = kare0_sonlu and omega_dt <= omega_esigi
            yol_sonlu = bool(np.isfinite(omega_dt_yol))
            yol_gecti = yol_sonlu and omega_dt_yol <= omega_esigi
            yakinsama_tanisi = tani(
                yakinsadi, salinim_gecti, salinim_sonlu,
                kare0_gecti, kare0_sonlu, yol_gecti, yol_sonlu,
            )
            yakinsadi = yakinsadi and salinim_gecti and kare0_gecti and yol_gecti
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
        # sürüklenme %22.4 iken panel ✓ basıyordu. Sonra metin "Ölçülen
        # büyüklük kinetik sıcaklıktır" dedi ve ✓ kaldı -- ama ⟨T⟩ ne basılıyor
        # ne hedefle karşılaştırılıyordu, yani ✓ ölçülmemiş bir hükmün
        # yanındaydı. C1/D3/D2-iv (11.10.2026): ✗ + YARGILANMADI tanısı.
        # Termostat türüne göre dallanma YOK (AST kilidi, `tur` dışı karşılaştırma
        # yasak): Langevin ve Nose-Hoover aynı hükmü alır.
        beklenen = (
            "NVT: toplam enerji KORUNMAZ, sistem banyoyla enerji alışverişi "
            "yapar. NVT'nin sorusu kinetik sıcaklığın hedefe oturup oturmadığıdır "
            "ve bu koşu onu yargılamaz (Yakınsama tanısı); aşağıdaki energy_drift "
            "bir tanı değeridir, bir hata payı DEĞİL."
        )
        yakinsadi = False
        yakinsama_tanisi = _NVT_YARGILANMADI
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
        # D2-ii (10.10.2026): TB'nin S ve omega*dt anahtarlarının AYNISI (aynı
        # adlar, aynı kural: NVT'de hepsi None); yol kareleri en kısa çift ve
        # tür başına en kısa anlar (`stiffest_frame_step` HF'de yok -- sertlik
        # vekili TB yasalarından).
        tanilar.update(
            energy_oscillation=energy_oscillation,
            energy_oscillation_scale="max_kinetic_all_steps" if salinim_hukme_girer else None,
            energy_oscillation_tolerance=salinim_esigi,
            omega_max_frame0=omega_max_frame0,
            omega_dt_frame0=omega_dt,
            omega_max_path_frames=omega_max_path_frames,
            omega_dt_path_frames=omega_dt_yol,
            **{ad: (path_frames or {}).get(ad) for ad in yol_anahtarlari},
            omega_dt_tolerance=omega_esigi,
        )
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
            # D2: salınımın paydası "max_kinetic_all_steps" -- kayıtların değil
            # HER adımın max KE'si (sürüklenmeninkinden farklı; ikisi ayrı ad).
            # NVT'de S hesaplanmaz: üç anahtar da None (hesaplanmamış bir
            # sayının paydası ✓'nin yanında basılmasın -- doğrulama, 09.10).
            energy_oscillation=energy_oscillation,
            energy_oscillation_scale="max_kinetic_all_steps" if salinim_hukme_girer else None,
            energy_oscillation_tolerance=salinim_esigi,
            # D2-i / D2-vi: kare 0'ın ve yol karelerinin omega'sı a.u.
            # (1/zaman), hükme giren omega*dt'ler; eşik ikisine de aynı.
            # Seçilen karelerin adımı ve en kısa çift mesafesi (bohr) hangi
            # karelerin Hessian'landığını söyler (`YolKaresiIzleyici.tanilar`).
            # NVT'de hesaplanmaz: hepsi None (S'nin anahtarlarıyla aynı kural).
            omega_max_frame0=omega_max_frame0,
            omega_dt_frame0=omega_dt,
            omega_max_path_frames=omega_max_path_frames,
            omega_dt_path_frames=omega_dt_yol,
            **{ad: (path_frames or {}).get(ad) for ad in _YOL_KARESI_TANILARI},
            omega_dt_tolerance=omega_esigi,
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
        convergence_diagnosis=yakinsama_tanisi,
        stability=stability,
        external_stability=external_stability,
    )
