"""MD senaryosu: YAML'dan motora."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import qlab.engine.md.factory
from qlab.engine.md.factory import build, provenance_for
from qlab.errors import ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import (
    MD_FORCE_FIELD_KINDS,
    MD_THERMOSTAT_KINDS,
    MDScenario,
)


def _morse_senaryo(dt: float = 0.78) -> MDScenario:
    return MDScenario(
        name="t",
        force_field={"kind": "morse", "params": {"d_e": 0.174434333, "r_e": 1.401118538, "a": 1.028997701}},
        initial={
            "positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
            "masses_amu": [1.00782503207, 1.00782503207],
        },
        thermostat=None,
        dt=dt,
        n_steps=10,
    )


def test_md_senaryosu_yamldan_yukleniyor(tmp_path):
    yol = tmp_path / "md.yaml"
    yol.write_text(
        "kind: md\n"
        "name: H2 Morse testi\n"
        "force_field:\n"
        "  kind: morse\n"
        "  params: {d_e: 0.174434333, r_e: 1.401118538, a: 1.028997701}\n"
        "initial:\n"
        "  positions_bohr: [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]\n"
        "  masses_amu: [1.00782503207, 1.00782503207]\n"
        "dt: 0.78\n"
        "n_steps: 100\n"
        "save_every: 10\n",
        encoding="utf-8",
    )
    senaryo = load_scenario(str(yol))
    assert isinstance(senaryo, MDScenario)
    assert senaryo.n_steps == 100


def test_bilinmeyen_anahtar_sessizce_yutulmuyor(tmp_path):
    """Mevcut kural (loader dokstring'i): yazım hatası varsayılana DÜŞMEZ."""
    yol = tmp_path / "md.yaml"
    yol.write_text(
        "kind: md\nname: t\nforce_field: {kind: morse, params: {}}\n"
        "initial: {positions_bohr: [[0,0,0]], masses_amu: [1.0]}\n"
        "dt: 1.0\nn_steps: 1\nn_stepss: 5\n",
        encoding="utf-8",
    )
    with pytest.raises(ScenarioError, match="n_stepss"):
        load_scenario(str(yol))


def test_kutleler_amu_verilip_m_e_olarak_kuruluyor():
    """SENARYODAN MOTORA GEÇERKEN DÖNÜŞÜM -- modülün en tehlikeli hatası.

    ÖLÇÜLDÜ (tasarım §3): dönüşüm atlanırsa H2'nin periyodu 42.86x yanlış
    çıkar ve HİÇBİR ŞEY PATLAMAZ. Sayı LİTERAL: 1.00782503207 amu =
    1837.1526 m_e.
    """
    durum, _, _ = build(_morse_senaryo())
    np.testing.assert_allclose(
        durum.masses, np.array([1837.1526, 1837.1526]), rtol=0.0, atol=1e-3
    )


def test_dt_kararlilik_sinirini_asarsa_REDDEDILIYOR():
    """ÖLÇÜLDÜ: velocity-Verlet omega*dt = 2.0'da patlar (1.999 kararlı,
    2.001 -> 4.28e+54, 2.05 -> nan).

    Sessizce nan üreten bir koşu yerine tipli hata verilir. Harmonik alanda
    omega analitiktir; kontrol orada yapılabilir.
    """
    senaryo = MDScenario(
        name="t",
        force_field={"kind": "harmonic", "params": {"k": 1.0}},
        initial={
            "positions_bohr": [[0.1, 0.0, 0.0]],
            "masses_amu": [1.0 / 1822.888486209],  # m = 1 m_e -> omega = 1
        },
        thermostat=None,
        dt=2.5,
        n_steps=10,
    )
    with pytest.raises(ScenarioError, match="kararlılık"):
        build(senaryo)


def test_provenance_kutleleri_ve_yaklasimlari_beyan_ediyor():
    """Kullanılan kütleler EKRANDA görünmeli: 1837.15 gibi bir sayıyı gören
    biri "1 amu = 1" hatasını fark edebilir."""
    p = provenance_for(_morse_senaryo(), energy_drift=1e-9)
    metin = " ".join(p.approximations) + " " + " ".join(p.neglected)
    assert "klasik" in metin.lower()
    assert "nokta parçacık" in metin.lower()
    assert "1837" in str(p.diagnostics["masses_electron_mass"])


def test_bilinmeyen_kuvvet_alani_reddediliyor():
    senaryo = MDScenario(
        name="t",
        force_field={"kind": "yok_boyle_bir_sey", "params": {}},
        initial={"positions_bohr": [[0.0, 0.0, 0.0]], "masses_amu": [1.0]},
        thermostat=None,
        dt=1.0,
        n_steps=1,
    )
    with pytest.raises(ScenarioError, match="kuvvet alanı"):
        build(senaryo)
def test_fcc_kafes_kurulumu_dogru_sayida_parcacik_veriyor():
    """`lattice: fcc` dalı: N = 4 * hucre^3 olmalı ve kutu kenarı yoğunluktan
    türetilmeli. Sayı LİTERAL: 108 = 4 * 3^3.
    """
    senaryo = MDScenario(
        name="t",
        force_field={
            "kind": "lennard_jones",
            "params": {
                "epsilon": 3.7938402530e-04, "sigma": 6.4345174543,
                "box_length": 33.0656, "cutoff": 16.0862936,
            },
        },
        initial={
            "lattice": "fcc", "n_particles": 108,
            "density_reduced": 0.8442, "masses_amu": [39.948],
        },
        thermostat=None,
        dt=445.732231,
        n_steps=1,
    )
    durum, _, _ = build(senaryo)
    assert durum.positions.shape == (108, 3)
    np.testing.assert_allclose(
        durum.masses, np.full(108, 39.948 * 1822.888486209), rtol=0.0, atol=1e-6
    )


def test_hartree_fock_kuvvet_alani_fabrikadan_kuruluyor():
    """`hartree_fock` dalı olmadan Plan D YAML'dan çağrılamaz.

    Dağıtım KAPALI bir sayımdır (`else: raise ScenarioError`), yani dal
    eklenmeden senaryo ham hata verir.
    """
    senaryo = MDScenario(
        name="t",
        force_field={
            "kind": "hartree_fock",
            "params": {"symbols": ["H", "H"], "basis": "sto-3g", "step": 1e-5},
        },
        initial={
            "positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3959193602]],
            "masses_amu": [1.00782503207, 1.00782503207],
        },
        thermostat=None,
        dt=5.0,
        n_steps=1,
    )
    durum, alan, termostat = build(senaryo)
    assert alan.__class__.__name__ == "HartreeFockGradient"
    assert alan.step == pytest.approx(1e-5, rel=0.0, abs=0.0)
    assert termostat is None


def test_HF_dalinda_yaklasim_listesi_ANALITIK_MODEL_DEMIYOR():
    """Şeffaflık panelinin YALAN söylememesi.

    `_YAKLASIMLAR` listesindeki "kuvvet alanı analitik bir modeldir,
    hesaplanmış bir yüzey değildir" satırı Plan D için DÜZ YALANDIR ve
    panelde basılır. HF dalı kendi listesini taşımak zorunda.

    Bu test o yalanı kilitler: HF provenance'ında o cümle GEÇMEMELİ, ve
    HF'ye özgü dört şey GEÇMELİ.
    """
    senaryo = MDScenario(
        name="t",
        force_field={
            "kind": "hartree_fock",
            "params": {"symbols": ["H", "H"], "basis": "sto-3g", "step": 1e-5},
        },
        initial={
            "positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3959193602]],
            "masses_amu": [1.00782503207, 1.00782503207],
        },
        thermostat=None,
        dt=5.0,
        n_steps=1,
    )
    prov = provenance_for(senaryo, energy_drift=1e-10)
    metin = " ".join(prov.approximations + prov.neglected).lower()

    # İKİ yalan da gitmiş olmalı (factory.py:16 ve :18 -- kaynak okundu)
    assert "analitik bir model" not in metin
    assert "elektron yok" not in metin
    for beklenen in ("born-oppenheimer", "tek slater", "sonlu baz", "sayısal gradyan"):
        assert beklenen in metin, f"HF beyanında eksik: {beklenen}"
    assert "sto-3g" in " ".join(str(v) for v in prov.diagnostics.values()).lower()
    assert prov.diagnostics["gradient_step_bohr"] == pytest.approx(1e-5)


def test_HF_dalinda_kararlilik_denetiminin_ATLANDIGI_beyan_ediliyor():
    """`omega*dt < 2` denetimi yalnız `harmonic` dalındadır; HF'de omega
    analitik DEĞİLDİR ve denetim atlanır.

    Atlamak meşru bir seçimdir; SESSİZ atlamak değildir.
    """
    senaryo = MDScenario(
        name="t",
        force_field={
            "kind": "hartree_fock",
            "params": {"symbols": ["H", "H"], "basis": "sto-3g"},
        },
        initial={
            "positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3959193602]],
            "masses_amu": [1.00782503207, 1.00782503207],
        },
        thermostat=None,
        dt=5.0,
        n_steps=1,
    )
    prov = provenance_for(senaryo, energy_drift=1e-10)
    metin = " ".join(prov.approximations + prov.neglected).lower()
    assert "kararlılık" in metin and "denetlenmiyor" in metin


# --- Kuvvet alanı kapısı ve TÜRETME KİLİDİ -----------------------------------
# `MD_FORCE_FIELD_KINDS` bu görevden önce hem ÖLÜdü (hiçbir yerde okunmuyordu)
# hem BAYATtı (`hartree_fock` eksikti, oysa `scenarios/h2_aimd.yaml` onu
# kullanıyor). Yani biri onu bir doğrulama kapısı olarak bağlasaydı depo KENDİ
# senaryosunu reddederdi. Aşağıdaki DÖRT test (parametrelenmeyle altı vaka) hem
# kapıyı kuruyor hem de demetin bir daha bayatlamasını İMKÂNSIZ kılıyor.


def test_MD_FORCE_FIELD_KINDS_FACTORY_SEVKINDEN_TURETILIYOR(sevk_tarayici):
    """BU BÖLÜMÜN TAŞIYICI TESTİ — kapının bayatlamasını yasaklar.

    Katman sözleşmesi `scenario`nun `engine`i import etmesini yasakladığı
    için demet ELLE yazılmak ZORUNDA (ölçüldü: `scenario -> qlab.engine`
    enjeksiyonu `lint-imports`ta rc=1 ve "Katmanlı mimari BROKEN" veriyor;
    30.08.2026'da `loader.py`ye `import qlab.engine` konarak DOĞRULANDI).
    Elle yazılan bir yokluk
    beyanı bayatlarsa bir YALANDIR ve burada yalan PAHALIDIR: bayat demet
    artık bir KAPI olduğu için, factory'ye eklenen ama demete eklenmeyen
    bir kuvvet alanı YAML'dan HİÇ yüklenemez.

    Bu yüzden doğruluk kaynağı demet değil, factory'nin sevk dallarıdır ve
    demet ona karşı sınanır. Beşinci bir kuvvet alanı eklenip demet
    güncellenmezse BU TEST KIRMIZI OLUR.

    ÖLÇÜLDÜ (30.08.2026): dört sevk dalı --
    harmonic, morse, lennard_jones, hartree_fock.
    """
    sevk = sevk_tarayici(qlab.engine.md.factory, "build", "tur")
    assert sevk == set(MD_FORCE_FIELD_KINDS), (
        f"factory {sorted(sevk)} sevk ediyor, şema {sorted(MD_FORCE_FIELD_KINDS)} "
        f"diyor -- demet bayatladı"
    )
    # Sayı LİTERAL: dördün altına düşerse sevk taraması sessizce boşalmış
    # demektir (ör. değişken adı `tur` değiştirilirse), ve boş küme boş
    # demete karşı yeşil kalırdı.
    assert len(sevk) == 4


@pytest.mark.parametrize(
    "senaryo_adi", ["h2_aimd.yaml", "h2_morse.yaml", "argon_nvt.yaml"]
)
def test_SEVK_EDILEN_MD_SENARYOLARI_KAPIDAN_GECIYOR(senaryo_adi):
    """Deponun KENDİ senaryolarını reddetmediğinin sınavı.

    `h2_aimd.yaml` bu listenin taşıyıcı üyesidir: kapı kurulmadan önceki
    bayat demet tam olarak onu (`kind: hartree_fock`) reddederdi.
    """
    yol = Path(__file__).resolve().parents[2] / "scenarios" / senaryo_adi
    senaryo = load_scenario(str(yol))
    assert isinstance(senaryo, MDScenario)
    assert senaryo.force_field["kind"] in MD_FORCE_FIELD_KINDS
    # `argon_nvt.yaml` termostat TAŞIR (`langevin`); ötekiler NVE'dir ve
    # `thermostat is None` olur. İki dal da burada geçiliyor.
    if senaryo.thermostat is not None:
        assert senaryo.thermostat["kind"] in MD_THERMOSTAT_KINDS


def test_YUKLEYICI_BILINMEYEN_KUVVET_ALANINI_YAMLDA_REDDEDIYOR(tmp_path):
    """Kapı YAML'ı okurken kapanır, motor kurulmadan; ve geçerli türleri
    SAYAR (`POTENTIAL_KINDS`/`SPIN_MODEL_KINDS` kapılarının deseni).

    Factory'nin `else: raise` dalı bunun YERİNE GEÇMEZ: o dal yalnızca
    `build()` çağrıldığında ateşlenir ve hangi türlerin geçerli olduğunu
    SÖYLEMEZ.
    """
    yol = tmp_path / "md.yaml"
    yol.write_text(
        "kind: md\nname: t\nforce_field: {kind: yok_boyle_bir_sey, params: {}}\n"
        "initial: {positions_bohr: [[0,0,0]], masses_amu: [1.0]}\n"
        "dt: 1.0\nn_steps: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ScenarioError, match="Bilinmeyen kuvvet alanı") as hata:
        load_scenario(str(yol))
    # Hata metni geçerli türleri SAYMALI -- yoksa kullanıcı yazım hatasını
    # düzeltmek için kaynağa bakmak zorunda kalır.
    for tur in MD_FORCE_FIELD_KINDS:
        assert tur in str(hata.value)


def test_KUVVET_ALANI_BLOGU_SOZLUK_DEGILSE_HAM_HATA_SIZMIYOR(tmp_path):
    """`force_field: morse` (blok yerine düz dizge) eskiden `dict(...)`
    içinde ham `ValueError` veriyordu; Global Constraint "ham hata
    tüketiciye ulaşmaz" gereği `ScenarioError`a çevrilir.
    """
    yol = tmp_path / "md.yaml"
    yol.write_text(
        "kind: md\nname: t\nforce_field: morse\n"
        "initial: {positions_bohr: [[0,0,0]], masses_amu: [1.0]}\n"
        "dt: 1.0\nn_steps: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ScenarioError, match="sözlük olmalı"):
        load_scenario(str(yol))


# --- Termostat kapısı: aynı tuzağın İKİNCİ KOPYASI ---------------------------
# `MD_THERMOSTAT_KINDS` de ÖLÜydü (yalnız tanımı ve `bridge_map.py`nin
# dokstringindeki bir anma). Kuvvet alanı demetinden TEK FARKI: BAYAT
# DEĞİLDİ -- ölçüldü, factory tam olarak `langevin` + `nose_hoover` sevk
# ediyor. Yani burada onarılan mevcut bir kusur YOK; kapatılan şey aynı
# tuzağın bir üçüncü termostat eklendiği gün patlayacak ikinci kopyası.


def test_MD_THERMOSTAT_KINDS_FACTORY_SEVKINDEN_TURETILIYOR(sevk_tarayici):
    """`MD_FORCE_FIELD_KINDS` kilidiyle AYNI sözleşme, aynı tarayıcı.

    Üçüncü bir termostat eklenip demet güncellenmezse BU TEST KIRMIZI OLUR
    -- ve kırmızı olması ŞART, çünkü demet artık bir KAPI: güncellenmeyen
    demet, factory'nin desteklediği bir termostatı YAML'dan yüklenemez
    kılar.

    ÖLÇÜLDÜ (30.08.2026): iki sevk dalı -- langevin, nose_hoover.
    """
    sevk = sevk_tarayici(qlab.engine.md.factory, "build", "t_tur")
    assert sevk == set(MD_THERMOSTAT_KINDS), (
        f"factory {sorted(sevk)} sevk ediyor, şema {sorted(MD_THERMOSTAT_KINDS)} "
        f"diyor -- demet bayatladı"
    )
    # Sayı LİTERAL: tarayıcı sessizce boşalırsa (ör. değişken adı `t_tur`
    # değiştirilirse) boş küme boş demete karşı yeşil kalırdı.
    assert len(sevk) == 2

    # İKİ KİLİT BİRBİRİNİ GÖRMÜYOR OLMALI: kuvvet alanı ve termostat
    # dalları aynı `build()` gövdesinde yan yana duruyor, ve tarayıcı
    # değişken adına göre ayırıyor. Karışsalardı iki iddia da anlamsızlaşırdı.
    assert sevk.isdisjoint(set(MD_FORCE_FIELD_KINDS))


def test_YUKLEYICI_BILINMEYEN_TERMOSTATI_YAMLDA_REDDEDIYOR(tmp_path):
    """Kapı YAML'ı okurken kapanır ve geçerli termostatları SAYAR.

    Factory'nin `else: raise` dalı bunun YERİNE GEÇMEZ: yalnızca `build()`
    çağrıldığında ateşlenir ve hangi türlerin geçerli olduğunu SÖYLEMEZ.
    """
    yol = tmp_path / "md.yaml"
    yol.write_text(
        "kind: md\nname: t\nforce_field: {kind: morse, params: {}}\n"
        "initial: {positions_bohr: [[0,0,0]], masses_amu: [1.0]}\n"
        "thermostat: {kind: yok_boyle_bir_termostat, params: {}}\n"
        "dt: 1.0\nn_steps: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ScenarioError, match="Bilinmeyen termostat") as hata:
        load_scenario(str(yol))
    for tur in MD_THERMOSTAT_KINDS:
        assert tur in str(hata.value)


def test_TERMOSTAT_BLOGU_SOZLUK_DEGILSE_HAM_HATA_SIZMIYOR(tmp_path):
    """`thermostat: langevin` (blok yerine düz dizge) ham `ValueError`
    vermez; Global Constraint gereği `ScenarioError`a çevrilir.
    """
    yol = tmp_path / "md.yaml"
    yol.write_text(
        "kind: md\nname: t\nforce_field: {kind: morse, params: {}}\n"
        "initial: {positions_bohr: [[0,0,0]], masses_amu: [1.0]}\n"
        "thermostat: langevin\n"
        "dt: 1.0\nn_steps: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ScenarioError, match="'thermostat' bir sözlük olmalı"):
        load_scenario(str(yol))


def test_TERMOSTATSIZ_SENARYO_HALA_NVE_KOSUYOR(tmp_path):
    """KAPININ DARALTMADIĞININ sınavı -- bu testin işi bir gerilemeyi
    yakalamak.

    `thermostat` İSTEĞE BAĞLIDIR ve yükleyicinin eski semantiği şuydu:
    anahtar yoksa DA boşsa (`{}`) `thermostat is None`, yani NVE koşusu.
    Kapı bu iki yolu da kapatabilirdi (`None`ın `MD_THERMOSTAT_KINDS`te
    olmaması yeterdi). Kapatmıyor, ve bu bilinçli.
    """
    for govde in ("", "thermostat: {}\n"):
        yol = tmp_path / f"md_{len(govde)}.yaml"
        yol.write_text(
            "kind: md\nname: t\nforce_field: {kind: morse, params: {}}\n"
            "initial: {positions_bohr: [[0,0,0]], masses_amu: [1.0]}\n"
            + govde
            + "dt: 1.0\nn_steps: 1\n",
            encoding="utf-8",
        )
        senaryo = load_scenario(str(yol))
        assert senaryo.thermostat is None
