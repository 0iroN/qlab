"""MD senaryosu: YAML'dan motora."""

from __future__ import annotations

import dataclasses
import types
from pathlib import Path

import numpy as np
import pytest

import qlab.engine.md.factory
from qlab.engine.md.factory import build, provenance_for
from qlab.engine.md.tb_kuvvet import SsSigmaTBKuvveti
from qlab.engine.tb.radyal import UsselRadyalYasa
from qlab.errors import ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import (
    MD_FORCE_FIELD_KINDS,
    MD_THERMOSTAT_KINDS,
    RADIAL_LAW_FIELDS,
    TB_SS_SIGMA_LAWS,
    TB_SS_SIGMA_PARAMS,
    TB_SS_SIGMA_REQUIRED,
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

    # İKİ yalan da gitmiş olmalı (factory.py `_YAKLASIMLAR`ın birinci ve
    # üçüncü cümlesi -- kaynak okundu; eski `:16`/`:18` satır numaraları
    # dosya büyüdükçe kaydı, bu yüzden yazılmıyor)
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
    demet ona karşı sınanır. Yeni bir kuvvet alanı eklenip demet
    güncellenmezse BU TEST KIRMIZI OLUR.

    ÖLÇÜLDÜ (30.08.2026): dört sevk dalı --
    harmonic, morse, lennard_jones, hartree_fock. 24.09.2026: BEŞ -- md<->tb
    ss-sigma kenarı `tb_ss_sigma`yı ekledi (`engine/md/tb_kuvvet.py`);
    altıncı bir tür eklenip demet güncellenmezse bu test kırmızı olur.
    """
    sevk = sevk_tarayici(qlab.engine.md.factory, "build", "tur")
    assert sevk == set(MD_FORCE_FIELD_KINDS), (
        f"factory {sorted(sevk)} sevk ediyor, şema {sorted(MD_FORCE_FIELD_KINDS)} "
        f"diyor -- demet bayatladı"
    )
    # Sayı LİTERAL: beşin altına düşerse sevk taraması sessizce boşalmış
    # demektir (ör. değişken adı `tur` değiştirilirse), ve boş küme boş
    # demete karşı yeşil kalırdı.
    assert len(sevk) == 5


@pytest.mark.parametrize(
    "senaryo_adi",
    ["h2_aimd.yaml", "h2_morse.yaml", "argon_nvt.yaml", "tb_zincir_md.yaml"],
)
def test_SEVK_EDILEN_MD_SENARYOLARI_KAPIDAN_GECIYOR(senaryo_adi):
    """Deponun KENDİ senaryolarını reddetmediğinin sınavı.

    `h2_aimd.yaml` bu listenin taşıyıcı üyesidir: kapı kurulmadan önceki
    bayat demet tam olarak onu (`kind: hartree_fock`) reddederdi.
    `tb_zincir_md.yaml` (G5) TB parametre kapısından da geçer.
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


# --- tb_ss_sigma dalı (md<->tb ss-sigma kenarı, 24.09.2026) ------------------
# Spec eleştirmeni P0-1 / P0-3 (ÖLÇÜLDÜ, V4 kopyası): TB türü ortak yaklaşım
# listesini basıyordu -- "elektron yok", "kuvvet alanı analitik bir modeldir"
# -- ve NVE'de "enerji SÜRÜKLENMEZ" diyordu. Bu dosyanın HF testi yalnız HF
# dalını kilitliyordu, yani o yalanı HİÇBİR TEST yakalamıyordu. Aşağıdaki
# kilit tür başına değil demet üzerinden kurulur: elektron çözen her tür --
# bugün iki -- aynı sınavdan geçer.

#: Elektronları ÇÖZEN türler. LİTERAL, bir modül sabitine karşı DEĞİL
#: (sabite karşı yazılan assert sabitle birlikte kayar). Her tür aşağıdaki
#: iki kümeden TAM BİRİNDE olmak zorunda: altıncı bir tür sınıflandırılmadan
#: bu dosya yeşil kalamaz.
ELEKTRON_COZEN = {"hartree_fock", "tb_ss_sigma"}
KLASIK = {"harmonic", "morse", "lennard_jones"}


def _yasa(genlik: float, us: float) -> dict:
    return {
        "genlik_hartree": genlik, "d0_bohr": 2.0, "us": us,
        "r_on_bohr": 3.0, "r_cut_bohr": 3.8,
    }


def _tb_parametreleri(**degisiklik) -> dict:
    """Ortak pencere (spec md-tb başı): ss_sigma -0.1/2/2, itme 0.05/2/6.

    Değeri `None` verilen anahtar SİLİNİR (eksik anahtar vakası için).
    """
    p = {
        "ss_sigma": _yasa(-0.1, 2.0),
        "itme": _yasa(0.05, 6.0),
        "n_electron": 16,
        "electron_temperature_kelvin": 631.6,
        "box_x_bohr": 32.0,
    }
    p.update(degisiklik)
    return {k: v for k, v in p.items() if v is not None}


def _senaryo(tur: str, params: dict | None = None) -> MDScenario:
    """Her tür için en küçük GEÇERLİ senaryo (yalnız `provenance_for`/`build`)."""
    iki_atom = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.3959193602]]
    tablo = {
        "harmonic": ({"k": 0.1}, iki_atom),
        "morse": ({"d_e": 0.174434333, "r_e": 1.401118538, "a": 1.028997701}, iki_atom),
        "lennard_jones": (
            {"epsilon": 3.79e-04, "sigma": 6.43, "box_length": 40.0, "cutoff": 16.0},
            [[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        ),
        "hartree_fock": ({"symbols": ["H", "H"], "basis": "sto-3g"}, iki_atom),
        "tb_ss_sigma": (_tb_parametreleri(), [[2.0 * i, 0.0, 0.0] for i in range(16)]),
    }
    varsayilan, konum = tablo[tur]
    return MDScenario(
        name="t",
        force_field={"kind": tur, "params": varsayilan if params is None else params},
        initial={"positions_bohr": konum, "masses_amu": [1.00782503207]},
        thermostat=None,
        dt=5.0,
        n_steps=1,
    )


@pytest.mark.parametrize("tur", MD_FORCE_FIELD_KINDS)
def test_ELEKTRON_COZEN_HER_TURDE_panel_ELEKTRON_YOK_DEMIYOR(tur):
    """Şeffaflık paneli elektron çözen hiçbir türde "elektron yok" DEMEZ.

    Demet üzerinden parametreli: her tür ya ELEKTRON_COZEN ya KLASIK
    kümesinde (tam birinde) olmak zorunda, ve elektron çözen her türde
    ortak listenin iki yalanı -- "elektron yok", "analitik bir model" --
    panelde GEÇMEZ. Klasik türlerde aynı satırların GEÇTİĞİ de sınanır:
    kontrol, testin boş geçmediğini gösterir.

    KIRMA (ölçüldü, 24.09.2026): `provenance_for`da TB dalı ortak listeye
    geri bağlanınca (`_TB_YAKLASIMLAR` -> `_YAKLASIMLAR`) `tb_ss_sigma`
    hücresi kırmızı; TB dalı tamamen silinince de (tür `else`e düşer).
    """
    assert (tur in ELEKTRON_COZEN) != (tur in KLASIK), f"{tur} sınıflandırılmamış"
    # D2: TB NVE'de salınım S'i zorunlu, öteki türlerde verilmez (kapı);
    # D2-i: a priori tanının omega0'ı da aynı kapıdan geçer; D2-vi: yol karelerininki de.
    salinim = 1e-9 if tur == "tb_ss_sigma" else None
    prov = provenance_for(
        _senaryo(tur), energy_drift=1e-9, energy_oscillation=salinim, omega_max_frame0=salinim,
        omega_max_path_frames=salinim,
    )
    metin = " ".join(prov.approximations + prov.neglected).lower()
    if tur in ELEKTRON_COZEN:
        assert "elektron yok" not in metin
        assert "analitik bir model" not in metin
    else:
        assert "elektron yok" in metin
        assert "analitik bir model" in metin


def test_TB_PANELI_MERMIN_KELVIN_SS_SIGMA_DOGA_CIPASI_ve_dt_BEYANI():
    """TB dalının KENDİ beyanı (spec 2.1(5), B1/B2/B6/B7/B10, eleştirmen P0-1/P0-3).

    Beş iddia panelde ADIYLA: (1) potansiyel Omega (Mermin), kuvvet E'nin
    gradyanı değil -- korunum yarısı ("KE + E KORUNMAZ") yaklaşımlarda
    DEĞİL, yalnız NVE'nin beklenen hatasında (termostatlı panel kendisiyle
    çelişmesin diye, `tests/md/test_tb_beyan.py`); (2) sıcaklık senaryoda
    Kelvin, sınıfta hartree; (3) yalnız
    ss-sigma, açı çarpanı özdeş 1; (4) doğa çıpası yok; (5) dt kararlılık
    denetimi yok. Ve NVE `expected_error`ı ortak metnin "enerji
    SÜRÜKLENMEZ" cümlesini TAŞIMAZ -- TB'de korunan KE + Omega'dır.
    Tanılarda kT İKİ birimde, n_electron ve r_cut (eleştirmen E4: V4'te
    üçü de yoktu).

    KIRMA (ölçüldü): TB `expected_error`ı ortak metne geri bağlanınca
    kırmızı; panelin kT'si Kelvin'den çevrilmeden basılınca kırmızı.
    """
    prov = provenance_for(
        _senaryo("tb_ss_sigma"), energy_drift=1e-9, energy_oscillation=1e-9, omega_max_frame0=1e-9,
        omega_max_path_frames=1e-9,
    )
    metin = " ".join(prov.approximations + prov.neglected).lower()
    # G6: anahtar adı (`electron_temperature_kelvin`) cümleden TANILARA
    # taşındı -- `tests/md/test_tb_beyan.py`nin parça-kopya eşiği ölçümü
    # (şemadaki anahtar sabiti cümlenin içinde geçiyordu). Adın panelde
    # kaldığı aşağıda tanılardan sınanır.
    for parca in (
        "omega (mermin)",
        "e'nin gradyanı değil",
        "senaryoda kelvin, kuvvet sınıfında hartree",
        "tanılarda ikisi de adıyla basılır",
        "kt > 0",
        "yalnız ss-sigma",
        "açı çarpanı özdeş 1",
        "doğa çıpası yok",
        "dt kararlılık denetimi tb dalında yapılmaz",
        # D2: koşuyu reddeden denetim yok, a posteriori hüküm var.
        "koşu reddedilmez",
        "koşudan sonra nve'nin salınım hükmü yargılar",
        # D2-i: koşudan önce kare-0 omega0*dt de hükme girer; D2-vi: koşudan
        # sonra yolun izlenen iki karesininki de; körlüğü yazılı.
        "koşudan önce de kare-0 geometrisinin sayısal hessian'ından ölçülen omega0*dt",
        "koşudan sonra da yolun her adımda izlenen iki karesininki",
        "yalnız seçilen karelere bakar",
    ):
        assert parca in metin, f"TB beyanında eksik: {parca!r}"
    beklenen = prov.expected_error.lower()
    assert "sürüklenmez" not in beklenen
    assert "ke + e korunmaz" in beklenen and "ke + omega" in beklenen
    # G5 (spec B8): sürüklenme max(KE)'ye bölünür, |blok[0]|'a DEĞİL; eşiğin
    # neyi yakalayıp neyi yargılamadığı yazılı. (Büyük "I" `.lower()` ile
    # ASCII "i" olur -- aranan parçalar bu yüzden büyük I içermeyen yerden.)
    for parca in (
        "max(ke)",
        "|blok[0]|'a bölünmez",
        "sıfırı keyfidir",
        "fermi seviyesindeki geçiş adımda çözülmediğinde",
        # D2 (09.10.2026): "salınımı yargılamaz / ✓ dt'nin küçük olduğu
        # anlamına gelmez" cümlesi bayatladı -- salınımı artık ayrı ölçüt
        # yargılıyor ve hüküm ikisinin VE'si.
        "ayrı ölçüt yargılar",
        "kayıtlardan değil her adımdan",
        # D2-i / D2-vi: S'nin seyrelmesini omega*dt tanısı yakalar (kare 0 ve
        # yol kareleri, tek eşik); hüküm dördünün VE'si.
        "titreşime katılmayan ke onu seyreltir",
        "omega_dt_frame0", "omega_dt_path_frames", "omega_dt_tolerance",
        "yakınsama dördünün ve'sidir",
    ):
        assert parca in beklenen, f"TB sürüklenme beyanında eksik: {parca!r}"
    assert "yargılamaz" not in beklenen and "anlamına gelmez" not in beklenen
    t = prov.diagnostics
    assert t["electron_temperature_kelvin"] == 631.6
    assert abs(t["kT_hartree"] - 0.002) < 2e-7  # 631.6 K = 0.0020002 hartree
    assert t["n_electron"] == 16 and t["r_cut_bohr"] == 3.8 and t["box_x_bohr"] == 32.0
    # Sevk edilen alanın kT'si ile panelin kT'si BİT BİT aynı (iki yerde
    # hesaplanıyor; biri kayarsa panel başka bir sıcaklık basardı).
    _, alan, _ = build(_senaryo("tb_ss_sigma"))
    assert alan.kT_hartree == t["kT_hartree"]


@pytest.mark.parametrize(
    ("ss_rc", "itme_rc"), [(3.8, 4.2), (4.2, 3.8)], ids=["itme-genis", "ss_sigma-genis"]
)
def test_TB_PANELI_r_cut_IKI_yasanin_BUYUGU_ve_KURULAN_KESMEYLE_AYNI(ss_rc, itme_rc):
    """Tanıdaki `r_cut_bohr` iki yasanın BÜYÜĞÜDÜR -- L/2 koşulunun kendisi.

    Yukarıdaki panel testi yalnız iki yasanın da 3.8 olduğu pencerede
    bakıyordu; yalnız `ss_sigma`yı okuyan panel orada yeşil kalıyordu
    (eleştirmen E4: V4'te r_cut hiç yoktu, sonra geldi ama bağlanmamıştı).
    İki asimetrik sıra: beklenen LİTERAL (4.2), test içinde max(...) yeniden
    YAZILMAZ; ikinci satır kurulan alanın kesmesinin panelle aynı olduğunu
    sınar. box_x_bohr = 32 L/2 koşulunu sağlar.
    KIRMA (ölçüldü, 24.09.2026, sk/duz/mut_md.log): yalnız `ss_sigma`yı
    okuyan panel `itme-genis` hücresinde, yalnız `itme`yi okuyan ve min()
    alan paneller `ss_sigma-genis` hücresinde kırmızı. Yalnız itme = 4.2
    senaryosuyla yalnız-itme mutantı hayatta kalırdı.
    """
    params = _tb_parametreleri(
        ss_sigma={**_yasa(-0.1, 2.0), "r_cut_bohr": ss_rc},
        itme={**_yasa(0.05, 6.0), "r_cut_bohr": itme_rc},
    )
    senaryo = _senaryo("tb_ss_sigma", params)
    t = provenance_for(
        senaryo, energy_drift=1e-9, energy_oscillation=1e-9, omega_max_frame0=1e-9,
        omega_max_path_frames=1e-9,
    ).diagnostics
    assert t["r_cut_bohr"] == 4.2
    _, alan, _ = build(senaryo)
    assert max(alan.ss_sigma.r_cut_bohr, alan.itme.r_cut_bohr) == t["r_cut_bohr"]


def test_PROVENANCE_DALLARI_SEVKLE_ESLESIYOR_ve_SAKLI_DAL_YOK(
    sevk_tarayici, dize_karsilastirma_tarayici
):
    """`sevk_tarayici`nin kör noktası KAPALI (eleştirmen P0-1).

    `provenance_for` dallarını `scenario.force_field["kind"] == ...` diye
    yazıyordu: solu bir `Subscript`, `sevk_tarayici`ye görünmez. Bu test
    (a) işlevdeki HER dizge karşılaştırmasının solunun sevk değişkeni `tur`
    olduğunu, (b) kendi dalı olan türlerin tam olarak ELEKTRON_COZEN
    olduğunu ve (c) `build`de saklı bir dal olmadığını -- her dizge
    karşılaştırmasının ya bir kuvvet alanı ya bir termostat sevki ya da
    bilinen kafes dalı olduğunu -- kilitler.

    KIRMA (ölçüldü, 24.09.2026): (1) TB'nin yaklaşım dalını
    `scenario.force_field["kind"] == "tb_ss_sigma"` diye yazan mutant
    (a)'yı kırar; (2) `scenario.force_field["kind"] == "morse"` biçiminde
    SAKLI bir dal ekleyip morse'un `expected_error`ını değiştiren mutant
    41 testin YALNIZ bu testinde kırmızı -- `sevk_tarayici` o dalı görmez,
    panel testleri morse'un beklenen hatasına bakmaz.
    """
    fabrika = qlab.engine.md.factory
    prov = dize_karsilastirma_tarayici(fabrika, "provenance_for")
    assert {sol for sol, _ in prov} == {"tur"}, prov
    assert {deger for _, deger in prov} == ELEKTRON_COZEN
    assert sevk_tarayici(fabrika, "provenance_for", "tur") == ELEKTRON_COZEN
    assert ELEKTRON_COZEN <= set(MD_FORCE_FIELD_KINDS)
    kur = dize_karsilastirma_tarayici(fabrika, "build")
    assert kur == (
        {("tur", k) for k in MD_FORCE_FIELD_KINDS}
        | {("t_tur", k) for k in MD_THERMOSTAT_KINDS}
        | {("ilk.get('lattice')", "fcc")}
    ), kur


def test_dize_karsilastirma_tarayici_SAKLI_BICIMLERI_GORUYOR(
    tmp_path, sevk_tarayici, dize_karsilastirma_tarayici
):
    """Tarayıcının kendisi ölçülür: `sevk_tarayici`nin göremediği dört biçim.

    Sentetik bir modülde `x["kind"] == "a"`, `"b" == tur`, `tur in ("c",
    "d")` ve `tur != "e"`. `sevk_tarayici` yalnız `tur == ...` biçimini
    görür (burada hiç yok -> boş küme); yeni tarayıcı dördünü de görür.
    """
    yol = tmp_path / "sahte.py"
    yol.write_text(
        "def f(x, tur):\n"
        "    if x['kind'] == 'a':\n        pass\n"
        "    if 'b' == tur:\n        pass\n"
        "    if tur in ('c', 'd'):\n        pass\n"
        "    if tur != 'e':\n        pass\n",
        encoding="utf-8",
    )
    modul = types.SimpleNamespace(__file__=str(yol))
    assert sevk_tarayici(modul, "f", "tur") == set()
    assert dize_karsilastirma_tarayici(modul, "f") == {
        ("x['kind']", "a"), ("tur", "b"), ("tur", "c"), ("tur", "d"), ("tur", "e"),
    }


def test_RADIAL_LAW_FIELDS_YASA_SINIFINDAN_TURETILIYOR():
    """Şemanın yasa alanları `UsselRadyalYasa`nın alanlarıyla AYNI (sırasıyla).

    Factory yasayı `UsselRadyalYasa(**alanlar)` ile kurar; şema elle
    yazılmak zorunda (katman sözleşmesi) ve bu kilit olmadan sınıfa bir
    alan eklendiğinde YAML kapısı onu "bilinmeyen anahtar" diye reddederdi.
    Rol adları (`ss_sigma`, `itme`) da kuvvet sınıfının ilk iki alanıyla aynı.
    """
    assert RADIAL_LAW_FIELDS == tuple(f.name for f in dataclasses.fields(UsselRadyalYasa))
    assert TB_SS_SIGMA_LAWS == tuple(f.name for f in dataclasses.fields(SsSigmaTBKuvveti))[:2]
    assert set(TB_SS_SIGMA_LAWS) <= set(TB_SS_SIGMA_REQUIRED) <= set(TB_SS_SIGMA_PARAMS)
    assert set(TB_SS_SIGMA_PARAMS) - set(TB_SS_SIGMA_REQUIRED) == {"box_x_bohr"}


def _tb_yaml(tmp_path, params_satiri: str) -> Path:
    yol = tmp_path / "tb.yaml"
    yol.write_text(
        "kind: md\nname: tb\n"
        "force_field:\n  kind: tb_ss_sigma\n  params: " + params_satiri + "\n"
        "initial: {positions_bohr: [[0,0,0],[2,0,0]], masses_amu: [1.00782503207]}\n"
        "dt: 20.0\nn_steps: 2\n",
        encoding="utf-8",
    )
    return yol


_YASA_YAML = "{genlik_hartree: -0.1, d0_bohr: 2.0, us: 2.0, r_on_bohr: 3.0, r_cut_bohr: 3.8}"
_ITME_YAML = "{genlik_hartree: 0.05, d0_bohr: 2.0, us: 6.0, r_on_bohr: 3.0, r_cut_bohr: 3.8}"
_GECERLI = (
    "{ss_sigma: " + _YASA_YAML + ", itme: " + _ITME_YAML
    + ", n_electron: 2, electron_temperature_kelvin: 631.6}"
)


def test_TB_YAML_YUKLENIYOR_ve_KUTU_ISTEGE_BAGLI(tmp_path):
    """Geçerli TB YAML'ı kapıdan geçer ve motor kurulur; `box_x_bohr` yoksa kutusuz."""
    senaryo = load_scenario(str(_tb_yaml(tmp_path, _GECERLI)))
    assert senaryo.force_field["kind"] == "tb_ss_sigma"
    _, alan, _ = build(senaryo)
    assert alan.kutu_x_bohr is None
    assert alan.ss_sigma.genlik_hartree == -0.1 and alan.itme.us == 6.0


@pytest.mark.parametrize(
    ("params", "desen"),
    [
        (_GECERLI.replace("n_electron", "n_electrons"), "bilinmeyen parametre"),
        (_GECERLI.replace(", electron_temperature_kelvin: 631.6", ""), "eksik parametre"),
        (
            _GECERLI.replace("r_cut_bohr: 3.8}, itme", "r_cut_bhor: 3.8}, itme"),
            "'ss_sigma' yasasının anahtarları",
        ),
        (_GECERLI.replace("us: 6.0, ", ""), "'itme' yasasının anahtarları"),
        (_GECERLI.replace("itme: " + _ITME_YAML, "itme: 0.05"), "'itme' bir sözlük olmalı"),
        ("[1, 2]", "bir sözlük olmalı"),
    ],
    ids=["param-yazim", "param-eksik", "yasa-yazim", "yasa-eksik", "yasa-skaler", "params-liste"],
)
def test_TB_PARAMETRE_KAPISI_YAMLDA_REDDEDIYOR(tmp_path, params, desen):
    """Anahtar kapısı YAML okunurken kapanır (`loader.py`nin genel kuralı:
    yazım hatası varsayılana DÜŞMEZ) -- motor kurulmadan, `ScenarioError`
    olarak. Değer kapıları motordadır (ayrı tür)."""
    with pytest.raises(ScenarioError, match=desen):
        load_scenario(str(_tb_yaml(tmp_path, params)))


@pytest.mark.parametrize(
    "params",
    [
        _tb_parametreleri(n_electron=None),
        _tb_parametreleri(sicaklik=300.0),
        _tb_parametreleri(itme={"genlik_hartree": 0.05}),
        _tb_parametreleri(electron_temperature_kelvin="631.6"),
        _tb_parametreleri(electron_temperature_kelvin=True),
    ],
    ids=["eksik", "bilinmeyen", "yasa-eksik", "kelvin-dizge", "kelvin-bool"],
)
def test_TB_FABRIKA_DOGRUDAN_KURULUMDA_HAM_HATA_SIZMIYOR(params):
    """`MDScenario(...)` doğrudan kurulduğunda (YAML kapısı yok) factory aynı
    anahtarları denetler: ham `KeyError`/`TypeError` tüketiciye ulaşmaz."""
    with pytest.raises(ScenarioError):
        build(_senaryo("tb_ss_sigma", params))


# ---------------------------------------------------------------------------
# G5 -- SÜRÜKLENME ÖLÇÜTÜ (md-tb spec 2.1(7), B8, eleştirmen P0-2)
#
# `run.py` sürüklenmeyi |blok[0]|'a bölüyordu; TB'de enerjinin sıfırı
# keyfidir. Ölçüt artık `md/factory.py::energy_drift`te: pay bütün türlerde
# aynı, TB paydası max(KE). Dinamik kanıtlar (c kayması, Fermi geçişi,
# durağan olmayan koşu, sevk edilen senaryo) `tests/md/test_tb_kuvvet.py`de;
# burada panelin ve ortak dalın kilitleri.
# ---------------------------------------------------------------------------


def test_G5_TB_ESIGI_PANELDE_ve_HUKUMDE_AYNI_SAYI():
    """Tanılarda basılan eşik, hükmün GERÇEKTEN kullandığı eşiktir.

    Sabite karşı değil HÜKME karşı: panelin `energy_drift_tolerance`ının bir
    ulp altı ✓, bir ulp üstü ✗. Ve TB dalı ORTAK eşiği kullanmıyor: aynı
    değerde (≈ TB eşiği) morse ✗ -- payda farklı olduğu için eşikler
    karşılaştırılamaz (factory yorumu). Payda adı panelde (`max_kinetic`);
    ortak dalın paneli G5'te DEĞİŞMEDİ (iki anahtar yok).

    KIRMA (ölçüldü, sk/g5_mutasyon.log): TB dalı `_MD_DRIFT_TOLERANCE`'a
    bağlanınca kırmızı (panel de 5e-3 basar ve ulp sınavı tutarlı kalır;
    yakalayan morse karşılaştırması); tanıya başka bir sayı basılınca
    kırmızı (ulp üstü ✓); eşik 1e-3 ve 5e-3 mutantları kırmızı.
    """
    tb = _senaryo("tb_ss_sigma")
    # D2: S eşiğin altında sabit (1e-9): bu test YALNIZ sürüklenme eşiğini sınar.
    # D2-i / D2-vi: omega*dt tanısı da geçer durumda sabit (omega 1e-9).
    w = dict(energy_oscillation=1e-9, omega_max_frame0=1e-9, omega_max_path_frames=1e-9)
    t = provenance_for(tb, energy_drift=1e-9, **w).diagnostics
    esik = t["energy_drift_tolerance"]
    assert t["energy_drift_scale"] == "max_kinetic"
    assert provenance_for(tb, energy_drift=float(np.nextafter(esik, 0.0)), **w).converged
    assert not provenance_for(tb, energy_drift=float(np.nextafter(esik, 1.0)), **w).converged
    morse = provenance_for(_senaryo("morse"), energy_drift=float(np.nextafter(esik, 0.0)))
    assert not morse.converged
    assert "energy_drift_scale" not in morse.diagnostics
    assert "energy_drift_tolerance" not in morse.diagnostics


def test_G5_TERMOSTATLI_TB_de_SURUKLENME_ESIGI_YOK():
    """NVT'de sürüklenme hükme girmez: eşik `None` basılır (0.0 DEĞİL), hüküm ✓.

    Termostatlı TB koşusu HİÇ koşulmadı (spec 6, eleştirmen P2-10); bu test
    yalnız panelin ne dediğini kilitler, dinamiği değil.
    """
    nvt = dataclasses.replace(
        _senaryo("tb_ss_sigma"),
        thermostat={"kind": "langevin", "params": {"temperature": 300.0, "gamma": 1e-3}},
    )
    prov = provenance_for(nvt, energy_drift=5.0)
    assert prov.converged is True
    assert prov.diagnostics["energy_drift_tolerance"] is None
    assert "NVT" in prov.expected_error


def test_G5_HAREKETSIZ_TB_KOSUSU_YARGILANAMAZ():
    """max(KE) = 0 -> oran `nan`, panel ✗ -- "sıfır sürüklenme ✓" DEĞİL.

    Aynı kayıtlarda ortak dal 0.0 verir (pay 0, payda |blok[0]| > 0) ve ✓
    basardı: hareketsiz bir koşuda enerjinin korunduğunu söylemek ölçülmemiş
    bir şeyi onaylamaktır. KIRMA (ölçüldü): TB dalında `nan` yerine 0.0
    döndüren mutant kırmızı.
    """
    kinetik = np.zeros(12)
    potansiyel = np.full(12, -1.2)
    tb = _senaryo("tb_ss_sigma")
    oran = qlab.engine.md.factory.energy_drift(tb, kinetik, potansiyel)
    assert np.isnan(oran)
    # S ve omega'lar eşik altında sabit: ✗'yi sürüklenmenin `nan`ı veriyor.
    assert provenance_for(
        tb, energy_drift=oran, energy_oscillation=1e-9, omega_max_frame0=1e-9,
        omega_max_path_frames=1e-9,
    ).converged is False
    assert qlab.engine.md.factory.energy_drift(_senaryo("morse"), kinetik, potansiyel) == 0.0


@pytest.mark.parametrize("tur", sorted(KLASIK))
def test_G5_ORTAK_DAL_OLCUTU_DEGISMEDI(tur):
    """Klasik türlerde ölçüt G5 öncesinin AYNISI (bit bit).

    `hartree_fock` 03.10.2026'da (C5) bu listeden ÇIKTI: paydası artık
    max(KE) -- bkz. `test_C5_HF_NVE_...`.

    Hesap `run.py`den `factory.energy_drift`e taşındı; bu kilit taşımanın
    ortak dalı değiştirmediğini söyler. Formül aşağıda AÇIKÇA yazılı (G5
    öncesi `run.py`): min(10, M) blok, |blok[-1] - blok[0]| / |blok[0]|.
    M = 23 (10 blok, son 3 kayıt atılır) ve M = 7 (7 blok). Kayıtlar
    sentetik: kinetik sabit olmayan, potansiyel negatif (|blok[0]| ≠
    |toplam[0]|).
    """
    for m in (23, 7):
        k = np.linspace(0.0, 0.3, m) ** 2
        v = -1.0 + 0.01 * np.sin(np.arange(m))
        toplam = k + v
        n = min(10, m)
        blok = toplam[: (m // n) * n].reshape(n, -1).mean(axis=1)
        beklenen = float(abs(blok[-1] - blok[0]) / abs(blok[0]))
        assert qlab.engine.md.factory.energy_drift(_senaryo(tur), k, v) == beklenen


@pytest.mark.slow
def test_C5_HF_NVE_kuvvet_gurultusu_max_KE_paydasiyla_KALIR_eski_payda_GECIRIRDI(tmp_path, capsys):
    """C5 (03.10.2026): HF dalının paydası max(KE); |blok[0]| ELEKTRONİK enerjiydi.

    A/B: sevk edilen `scenarios/h2_aimd.yaml` (200 adım) ve AYNI senaryo yalnız
    gradyan adımı 1e-5 -> 1e-13 (sonlu fark gürültüsü). Başlangıç, dt ve Verlet
    salınımı aynı; farkı yalnız KUVVET HATASI yaratır, yani bu katlanma değil
    gerçek bir korunum hatasıdır. ÖLÇÜLDÜ: yeni oran 1.465e-04 (✓) ->
    2.825e-01 (✗); eski |blok[0]| oranı 8.78e-08 -> 2.62e-04, İKİSİNDE DE ✓.
    """
    import qlab.run as run_modulu

    sevk = Path(__file__).resolve().parents[2] / "scenarios" / "h2_aimd.yaml"
    metin = sevk.read_text(encoding="utf-8")
    assert "step: 1.0e-5" in metin
    esik = qlab.engine.md.factory._MD_DRIFT_TOLERANCE
    oranlar = {}
    for ad, adim in (("temiz", "1.0e-5"), ("gurultulu", "1.0e-13")):
        yol = tmp_path / f"{ad}.yaml"
        yol.write_text(metin.replace("step: 1.0e-5", f"step: {adim}"), encoding="utf-8")
        cikti = tmp_path / f"{ad}.npz"
        assert run_modulu.main([str(yol), "-o", str(cikti)]) == 0
        satir = next(s for s in capsys.readouterr().out.splitlines() if s.startswith("Yakınsama"))
        assert "energy_drift_scale=max_kinetic" in satir and f"energy_drift_tolerance={esik:.3e}" in satir
        kayit = np.load(cikti)
        toplam = kayit["kinetic"] + kayit["potential"]
        blok = toplam[: (len(toplam) // 10) * 10].reshape(10, -1).mean(axis=1)
        pay = abs(blok[-1] - blok[0])
        oranlar[ad] = (pay / np.max(kayit["kinetic"]), pay / abs(blok[0]), satir)
    assert oranlar["temiz"][0] < esik and "✓" in oranlar["temiz"][2]
    assert oranlar["gurultulu"][0] > 10 * esik and "✗" in oranlar["gurultulu"][2]
    assert oranlar["gurultulu"][1] < esik  # eski payda gerçek korunum hatasını GEÇİRİRDİ


def test_C5_HF_ESIGI_PANELDE_ve_HUKUMDE_AYNI_SAYI_ve_HAREKETSIZ_YARGILANAMAZ():
    """TB'nin ulp kilidinin HF karşılığı (03.10 doğrulayıcısı: üç mutant geçiyordu).

    Panelin `energy_drift_tolerance`ı hükmün GERÇEKTEN kullandığı eşik ve
    `_MD_DRIFT_TOLERANCE`; bir ulp altı ✓, bir ulp üstü ✗. HF dalında max(KE) = 0
    ise oran `nan` ve hüküm ✗ (YARGILANAMAZ), 0.0 = sahte ✓ değil.
    """
    hf = _senaryo("hartree_fock")
    t = provenance_for(hf, energy_drift=1e-9).diagnostics
    esik = t["energy_drift_tolerance"]
    assert t["energy_drift_scale"] == "max_kinetic"
    assert esik == qlab.engine.md.factory._MD_DRIFT_TOLERANCE
    assert provenance_for(hf, energy_drift=float(np.nextafter(esik, 0.0))).converged
    assert not provenance_for(hf, energy_drift=float(np.nextafter(esik, 1.0))).converged
    oran = qlab.engine.md.factory.energy_drift(hf, np.zeros(12), np.full(12, -1.1))
    assert np.isnan(oran)
    assert provenance_for(hf, energy_drift=oran).converged is False


def test_B10_KARARLILIK_YAKINSAMAYA_GIRMEZ_ve_VARSAYILAN_ADIM_PANELDE():
    """Osman'ın kararı (03.10.2026): kararlılık hükmü `converged`ı DEĞİŞTİRMEZ,
    kendi satırını alır. Ayrıca YAML `step` vermezse panel artık `None` değil
    koşunun gerçekten kullandığı varsayılanı (5e-5) basar.
    """
    from qlab.provenance import StabilityVerdict

    hf = _senaryo("hartree_fock")
    eyer = StabilityVerdict(stable=False, lowest_eigenvalue=-0.45, n_rotations=1, channel="dis")
    ic = StabilityVerdict(stable=True, lowest_eigenvalue=3.037, n_rotations=1, channel="ic")
    duz = provenance_for(hf, energy_drift=1e-9)
    izli = provenance_for(
        hf, energy_drift=1e-9, stability=ic, external_stability=eyer, kararlilik_tanisi={"dis_kararsiz": "3/5"}
    )
    assert duz.converged is izli.converged is True
    satirlar = dict(izli.as_lines())
    assert "SPİN EYERİ" in satirlar["Dış kararlılık"] and "Dış kararlılık" not in dict(duz.as_lines())
    assert satirlar["Kararlılık"].startswith("✓") and "Kararlılık" not in dict(duz.as_lines())
    assert izli.diagnostics["dis_kararsiz"] == "3/5"
    assert "step" not in hf.force_field.get("params", {})
    assert duz.diagnostics["gradient_step_bohr"] == 5e-5


_RC_USTU_YAML = """kind: md
name: rc_ustu
force_field:
  kind: hartree_fock
  params:
    symbols: ["H", "H"]
    basis: sto-3g
    step: 1.0e-5
initial:
  positions_bohr: [[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]]
  masses_amu: [1.00782503207, 1.00782503207]
dt: 5.031487
n_steps: 20
save_every: 1
"""


def test_B10_RC_USTU_KISA_KOSU_run_YOLUNDAN_PANELE(tmp_path, capsys):
    """Hızlı şerit: `run.py`nin izlemeyi okuyup panele taşıdığı yol (2 adım).

    Durgun R0 = 2.5 bohr Rc'nin üstünde -- her merkez SCF dış kanalda
    kararsız (k = n), iç kanal kararlı. Yavaş testin run.py yarısını hızlı
    şeritte tutar (mutasyon turunda M6/M10/M11 yalnız yavaş testte ölüyordu).
    """
    import qlab.run as run_modulu

    yol = tmp_path / "rc_ustu_kisa.yaml"
    yol.write_text(_RC_USTU_YAML.replace("n_steps: 20", "n_steps: 2"), encoding="utf-8")
    assert run_modulu.main([str(yol), "-o", str(tmp_path / "kisa.npz")]) == 0
    panel = {s[:14].strip(): s[16:] for s in capsys.readouterr().out.splitlines() if len(s) > 16}
    assert panel["Dış kararlılık"].startswith("✗ SPİN EYERİ") and panel["Kararlılık"].startswith("✓")
    k, n = map(int, panel["Yakınsama"].split("dis_kararsiz=")[1].split(",")[0].split("/"))
    assert 0 < k == n
    assert "ic_kararsiz=0/" in panel["Yakınsama"] and "max_bag_bohr=2.5" in panel["Yakınsama"]


@pytest.mark.slow
def test_B10_RC_USTU_YORUNGE_PANELDE_SPIN_EYERI_SEVK_EDILEN_TEMIZ(tmp_path, capsys):
    """B10: R > Rc = 2.1796947938 bohr'da H2/STO-3G RHF yüzeyi SPİN EYERİ.

    Durgun R0 = 2.5 bohr koşusu: eskiden panel hiçbir şey demiyordu (✓);
    şimdi "Dış kararlılık ✗ SPİN EYERİ" (en kötü λ = -0.4501, ölçüldü), iç
    kanal ✓. Sevk edilen h2_aimd Rc'nin 0.784 bohr altında kalır: iki satır ✓.
    """
    import qlab.run as run_modulu

    yol = tmp_path / "rc_ustu.yaml"
    yol.write_text(_RC_USTU_YAML, encoding="utf-8")
    sevk = Path(__file__).resolve().parents[2] / "scenarios" / "h2_aimd.yaml"
    paneller = {}
    for ad, senaryo in (("rc_ustu", yol), ("sevk", sevk)):
        assert run_modulu.main([str(senaryo), "-o", str(tmp_path / f"{ad}.npz")]) == 0
        paneller[ad] = {s[:14].strip(): s[16:] for s in capsys.readouterr().out.splitlines() if len(s) > 16}
    rc = paneller["rc_ustu"]
    assert rc["Dış kararlılık"].startswith("✗ SPİN EYERİ") and "-4.501e-01" in rc["Dış kararlılık"]
    assert rc["Kararlılık"].startswith("✓")
    k, n = map(int, rc["Yakınsama"].split("dis_kararsiz=")[1].split(",")[0].split("/"))
    assert 0 < k <= n
    sevk_p = paneller["sevk"]
    assert sevk_p["Dış kararlılık"].startswith("✓") and sevk_p["Kararlılık"].startswith("✓")
    assert "dis_kararsiz=0/" in sevk_p["Yakınsama"]


def test_G5_energy_drift_KAYIT_ve_SEKIL_KAPISI():
    """M < 2 kayıt hata (tek blokta pay 0.0 olurdu: ölçülmemiş ✓); şekil hatası da.

    Mesaj `run.py`nin G5 öncesi mesajıyla aynı ("en az 2 kayıt") --
    `tests/test_run.py::test_MD_SURUKLEME_dali_DUZGUN_ISTISNA_firlatiyor`
    o dalı `_run_md` üzerinden tetikler.

    Şekil kapısının İKİ yarısı ayrı hücrelerle: (5,) / (4,) UZUNLUK
    uyuşmazlığını, EŞİT şekilli 2B girdi ise "tek boyutlu" yarısını sınar.
    O yarı silinirse (2, 12) ile -2 x (2, 12)'de sonuç 0.0 = sahte ✓ olur
    (TB dalında da ortak dalda da); (5, 2) çiftinde nan; 0-B girdide ham
    TypeError sızar. KIRMA (ölçüldü, 24.09.2026, sk/duz/mut_md.log ve
    mut_ndim_hucre.log): `kinetic.ndim != 1 or` yarısını silen mutant ilk
    sürümde bu testi GEÇİYORDU (yalnız uzunluk hücresi vardı); üç yeni
    hücrenin her biri onu İKİ türde de ayrı ayrı kırıyor.
    """
    from qlab.errors import InvalidInputError

    fabrika = qlab.engine.md.factory
    for tur in ("tb_ss_sigma", "morse"):
        with pytest.raises(InvalidInputError, match="en az 2"):
            fabrika.energy_drift(_senaryo(tur), np.array([0.1]), np.array([-1.0]))
        with pytest.raises(InvalidInputError, match="aynı uzunlukta"):
            fabrika.energy_drift(_senaryo(tur), np.zeros(5), np.zeros(4))
        for kin, pot in (
            (np.ones((2, 12)), -2.0 * np.ones((2, 12))),
            (np.zeros((5, 2)), np.zeros((5, 2))),
            (np.float64(1.0), np.float64(-1.0)),
        ):
            with pytest.raises(InvalidInputError, match="tek boyutlu"):
                fabrika.energy_drift(_senaryo(tur), kin, pot)


def test_G5_energy_drift_SAKLI_DAL_YOK_ve_run_py_KENDISI_HESAPLAMIYOR(
    sevk_tarayici, dize_karsilastirma_tarayici
):
    """(a) `energy_drift`in dizge karşılaştırmaları TAM OLARAK `tur == "tb_ss_sigma"`
    ve (03.10.2026, C5) `tur == "hartree_fock"` (saklı bir `x["kind"] == ...`
    dalı kırmızı olur); (b) `run.py::_run_md`
    sürüklenmeyi KENDİSİ hesaplamaz: `energy_drift`i çağırır ve gövdesinde
    `reshape`/`mean` yoktur -- G5 öncesinin |blok[0]| formülü oraya geri
    kopyalanırsa kırmızı (TB'nin sıfırdan bağımsızlığı sessizce kaybolurdu).
    """
    import ast

    import qlab.run as run_modulu

    fabrika = qlab.engine.md.factory
    # C5 (03.10.2026): `hartree_fock` de max(KE) dalında; ikisi de ELEKTRON ÇÖZEN tür.
    assert dize_karsilastirma_tarayici(fabrika, "energy_drift") == {
        ("tur", "tb_ss_sigma"), ("tur", "hartree_fock")
    }
    assert sevk_tarayici(fabrika, "energy_drift", "tur") == ELEKTRON_COZEN
    kaynak = Path(run_modulu.__file__).read_text(encoding="utf-8")
    govde = next(
        d for d in ast.walk(ast.parse(kaynak))
        if isinstance(d, ast.FunctionDef) and d.name == "_run_md"
    )
    cagrilar = [d.func for d in ast.walk(govde) if isinstance(d, ast.Call)]
    adlar = {f.id for f in cagrilar if isinstance(f, ast.Name)}
    oznitelikler = {f.attr for f in cagrilar if isinstance(f, ast.Attribute)}
    assert "energy_drift" in adlar
    assert not {"reshape", "mean"} & oznitelikler, oznitelikler
    # D2 (09.10.2026): salınım da fabrikada. `energy_oscillation`ın TEK dalı
    # TB'dir; `_run_md` onu çağırır, S'yi KENDİSİ hesaplamaz (kayıtlı
    # karelerden ptp/max/min = save_every'ye bağlı sahte hüküm, D2 ön ölçümü)
    # ve sonucu `provenance_for`a `energy_oscillation=` ile geçirir.
    assert dize_karsilastirma_tarayici(fabrika, "energy_oscillation") == {
        ("tur", "tb_ss_sigma")
    }
    assert "energy_oscillation" in adlar
    assert not {"ptp", "max", "min"} & oznitelikler, oznitelikler
    # Doğrulama (09.10.2026, N05/N12): yerleşik min/max ya da
    # `dataclasses.replace` ile kayıtlı karelerden S kuran bir `_run_md` yalnız
    # yukarıdaki öznitelik yasağından geçiyordu. Yerleşikler ve `replace` de
    # yasak; davranış tarafı T1'in save_every 2 hücresinde (test_d2_salinim).
    assert not {"min", "max", "sorted"} & adlar, adlar
    assert "replace" not in oznitelikler and "replace" not in adlar
    prov_cagrisi = next(
        d for d in ast.walk(govde)
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
        and d.func.id == "provenance_for"
    )
    assert "energy_oscillation" in {k.arg for k in prov_cagrisi.keywords}
    # D2-i (10.10.2026): a priori tanı da fabrikada; TEK dalı TB NVE'dir ve
    # `_run_md` onu çağırıp sonucunu `omega_max_frame0=` ile geçirir (Hessian'ı
    # kendisi kurmaz: `eigvalsh` yasak).
    assert dize_karsilastirma_tarayici(fabrika, "omega_max_frame0") == {
        ("tur", "tb_ss_sigma")
    }
    assert "omega_max_frame0" in adlar
    assert "eigvalsh" not in oznitelikler
    assert "omega_max_frame0" in {k.arg for k in prov_cagrisi.keywords}
    # D2-vi: yol kareleri de fabrikada; izleyicinin ve Hessian'ın TEK dalı TB
    # NVE'dir. `_run_md` izleyiciyi kurar, koşuyu onunla yapar, sonucu
    # `omega_max_path_frames=` ve `path_frames=` ile geçirir.
    for islev in ("path_frame_tracker", "omega_max_path_frames"):
        assert dize_karsilastirma_tarayici(fabrika, islev) == {("tur", "tb_ss_sigma")}, islev
        assert islev in adlar, islev
    assert {"omega_max_path_frames", "path_frames"} <= {k.arg for k in prov_cagrisi.keywords}
