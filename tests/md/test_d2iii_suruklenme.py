"""D2-iii, D2-ii-c, D2-v (12.10.2026) — NVE korunum hükmünün üç düşük maddesi.

D2-iii: `energy_drift` KAYITLI KARELERDEN bloklanıyordu, yani save_every
hükmü çeviriyordu. ÖLÇÜLDÜ (HEAD e262cab, kamu yolu): h2_morse dt 20 x 400
save_every 1 / 2 / 4 / 10 / 37 / 100 -> 6.147e-04 / 6.786e-04 / 2.641e-03 /
4.155e-03 (✓) / 3.249e-02 / 1.884e-02 (✗); tb_zincir_md 1 / 2 / 5 / 10 / 20 ->
4.525e-04 / 4.513e-04 / 3.926e-04 / 4.072e-04 / 1.484e-03; h2_aimd 1 / 2 / 10
-> 7.511e-05 / 1.465e-04 / 7.966e-04. Kod: integratör her adımın KE +
potansiyelini biriktirir (`Trajectory.total_all_steps`), `energy_drift(
scenario, yorunge)` onu bloklar (TB/HF paydası her adımın max KE'si); seri
save_every 1'in karelerinin kendisidir, yani save_every 1 panelleri bit bit
aynı kalır. Kamu yolu `run.py`nin `energy_drift(scenario, yorunge)` çağrısına
dayanır.

D2-ii-c: kısa koşuda blok ölçütü Verlet salınımını sürüklenme sanıyor (HEAD:
h2_aimd geometrisi dt 12.014 x 42 -> ✗, sürüklenme 1.278e-02; S ve omega*dt
geçiyor). ✓'ye çevrilemez (pay <= tepe-tepe; `energy_drift` SINIR'ı); tanı
artık salınımın payının farkı açıklayıp açıklamadığını söyler: "AYRILAMIYOR
(YARGILANAMAZ)" ya da eski "eşiğin üstünde ya da yargılanamaz". Hüküm ✗ kalır.

D2-v: max KE yuvarlama düzeyindeyken S gürültüdür (ölçülen gürültü tepe-tepesi
eps x |E|'nin 15 katına kadar). Taban: max KE <= 3000 x eps x max|KE + potansiyel|
-> S `nan`, "salınım YARGILANAMAZ". HEAD'de TB dimeri r_e + 1e-7 bohr'da S
4.859e-02 (gerçeği 1.010e-02) ile ✓ basıyordu, sürüklenme tam 0.

Sayılar yalnız deterministik ve gürültüden uzak hücrelerde kilitlenir; taban
altındaki gürültü değerleri kilitlenmez (makineye bağlı). SÜRE: her test
saniyenin altında (en uzunu HF kısa koşusu ~0.6 s).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import yaml

import qlab.engine.md.factory as fabrika
from qlab.engine.md.factory import build, energy_drift, energy_oscillation, omega_max_frame0, provenance_for
from qlab.engine.md.forcefields import HartreeFockGradient
from qlab.engine.md.integrator import Trajectory, run
from qlab.engine.md.state import MDState
from qlab.errors import InvalidInputError
from qlab.run import main
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import MDScenario
from qlab.units import amu_to_electron_mass

_SENARYOLAR = Path(__file__).resolve().parents[2] / "scenarios"
_M_H = 1.00782503207
#: TB dimeri (ss-sigma ortak yasaları, n_e 2, 631.6 K, kutusuz): kuvvetin
#: sıfırı (brentq, xtol 1e-15) ve kare-0 omega0 (`omega_max_frame0`).
_TB_RE = 1.8561589637741096
_TB_OMEGA0 = 0.022423688972326262
_YASA = dict(genlik_hartree=-0.1, d0_bohr=2.0, us=2.0, r_on_bohr=3.0, r_cut_bohr=3.8)
_ITME = dict(genlik_hartree=0.05, d0_bohr=2.0, us=6.0, r_on_bohr=3.0, r_cut_bohr=3.8)


def _sevk(ad: str, **degisen) -> dict:
    """Sevk edilen senaryo + değişen anahtarlar (`initial.positions_bohr` ve
    `force_field.params` için `konum=` / `parametre=`)."""
    d = yaml.safe_load((_SENARYOLAR / f"{ad}.yaml").read_text(encoding="utf-8"))
    if "konum" in degisen:
        d["initial"]["positions_bohr"] = degisen.pop("konum")
    d["force_field"]["params"].update(degisen.pop("parametre", {}))
    d.update(degisen)
    return d


def _tb_dimer(r: float, n: int = 94) -> dict:
    return {
        "kind": "md", "name": "d2v", "force_field": {"kind": "tb_ss_sigma", "params": {
            "ss_sigma": _YASA, "itme": _ITME, "n_electron": 2, "electron_temperature_kelvin": 631.6}},
        "initial": {"positions_bohr": [[0.0, 0.0, 0.0], [r, 0.0, 0.0]], "masses_amu": [_M_H]},
        "dt": 0.2 / _TB_OMEGA0, "n_steps": n, "save_every": 1,
    }


def _yaz(yol: Path, d: dict) -> Path:
    """PyYAML float'ları repr ile yazar: YAML'dan bit bit geri döner."""
    yol.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    return yol


def _panel(d: dict, tmp_path: Path, capsys, ad: str = "d2iii") -> dict[str, str]:
    """Kamu yolu: `python -m qlab.run <yaml> -o x.npz`; etiket -> satır."""
    assert main([str(_yaz(tmp_path / f"{ad}.yaml", d)), "-o", str(tmp_path / f"{ad}.npz")]) == 0
    ekran = capsys.readouterr().out
    return {s.split(":", 1)[0].strip(): s for s in ekran.splitlines() if ":" in s}


def _canli(satir: dict[str, str]) -> str:
    return satir["Yakınsama tanısı"].split(":", 1)[1].strip().split(" GRİ BÖLGE")[0]


def _kosu(d: dict, tmp_path: Path, save_every: int):
    """`run.py::_run_md` sırası (kamu yükleyicisi, `build`, COM çıkarma, integratör)."""
    senaryo = load_scenario(_yaz(tmp_path / "k.yaml", d))
    durum, alan, _ = build(senaryo)
    durum = durum.remove_center_of_mass_motion()
    return senaryo, run(durum, alan, dt=senaryo.dt, n_steps=senaryo.n_steps, save_every=save_every)


# ------------------------------------------------- D2-iii: her adım serisi --


def test_D2iii_INTEGRATOR_HER_ADIMI_BIRIKTIRIR_save_every_BAGIMSIZ(tmp_path):
    """`total_all_steps`: n_steps + 1 değer (adım 0 dahil), save_every 1 / 4 / 37'de
    BİT BİT aynı ve save_every 1'in karelerinin KE + potansiyelinin kendisi; uçları
    `total_min` / `total_max`. Pencere h2_morse, dt 20 x 400.
    KIRMA: seri yalnız kayıtlı adımlarda biriktirilir (se 37'de 11 değer) ya da
    adım 0 atlanır (400 değer) -> kırmızı.
    """
    d = _sevk("h2_morse", dt=20.0, n_steps=400)
    seriler = {}
    for se in (1, 4, 37):
        _, yorunge = _kosu(d, tmp_path, se)
        seriler[se] = yorunge
        assert yorunge.total_all_steps.shape == (401,)
        assert yorunge.total_all_steps.min() == yorunge.total_min
        assert yorunge.total_all_steps.max() == yorunge.total_max
    bir = seriler[1]
    assert np.array_equal(bir.total_all_steps, bir.kinetic + bir.potential)
    for se in (4, 37):
        assert np.array_equal(seriler[se].total_all_steps, bir.total_all_steps), se
    # Termostat döngüsünün kurucusu biriktirmez: varsayılan None ("ölçülmedi").
    assert Trajectory(np.zeros(2), np.zeros((2, 1, 3)), np.zeros((2, 1, 3)), np.ones(2),
                      np.ones(2)).total_all_steps is None


def test_D2iii_energy_drift_YORUNGEDEN_save_every_BAGIMSIZ_se1_ILE_BIT_BIT(tmp_path):
    """`energy_drift(scenario, yorunge)`: save_every 1 / 2 / 4 / 10 / 37 / 100'de AYNI
    sayı ve save_every 1'in karelerinden okunanla bit bit aynı (6.147135e-04, ortak
    dal); karelerden okunan (eski yol) aynı yörüngede 6.786e-04 / 2.641e-03 /
    4.155e-03 / 3.249e-02 / 1.884e-02 -- hüküm se 37 ve 100'de ✗'e dönüyordu
    (pozitif kontrol: kareler gerçekten ayrışıyor). TB dalı (T1 hücresi, tohum 4,
    dt 40 x 100): se 1 / 2 / 5 -> 8.787803e-03; se 5'in kareleri max KE'yi kaçırır
    (0.014648 < 0.015320) -- payda `kinetic_max` olmalı.
    KIRMA: yörünge biçimi kareleri bloklar -> morse eşitliği kırmızı; TB paydası
    karelerin max KE'si -> se 5 kırmızı (9.19e-03).
    """
    d = _sevk("h2_morse", dt=20.0, n_steps=400)
    degerler, kare = {}, {}
    for se in (1, 2, 4, 10, 37, 100):
        senaryo, yorunge = _kosu(d, tmp_path, se)
        degerler[se] = energy_drift(senaryo, yorunge)
        kare[se] = energy_drift(senaryo, yorunge.kinetic, yorunge.potential)
    assert len(set(degerler.values())) == 1, degerler
    assert degerler[1] == kare[1]
    assert abs(degerler[1] - 6.147135e-04) < 1e-9
    assert kare[37] > fabrika._MD_DRIFT_TOLERANCE > degerler[37], kare  # hüküm çevriliyordu
    x = 2.0 * np.arange(16) + 0.05 * np.random.default_rng(4).standard_normal(16)
    tb = _sevk("tb_zincir_md", dt=40.0, n_steps=100, konum=[[float(v), 0.0, 0.0] for v in x])
    tb_deger = {}
    for se in (1, 2, 5):
        senaryo, yorunge = _kosu(tb, tmp_path, se)
        tb_deger[se] = energy_drift(senaryo, yorunge)
        if se == 5:
            assert yorunge.kinetic.max() < yorunge.kinetic_max  # kareler tepeyi kaçırıyor
    assert len(set(tb_deger.values())) == 1, tb_deger
    assert abs(tb_deger[1] - 8.787803e-03) < 1e-9


def test_D2iii_KAMU_YOLU_MORSE_save_every_37_ARTIK_GECER(tmp_path, capsys):
    """Kamu yolu: h2_morse dt 20 x 400. HEAD save_every 37'de ✗ (sürüklenme
    3.249e-02), save_every 1'de ✓ (6.147e-04) -- aynı yörünge. Şimdi ikisi de ✓
    6.147e-04. `run.py` sürüklenmeyi yörüngeden okumalı (`energy_drift(scenario,
    yorunge)`); karelerden okuyan bir `run.py` burada ✗ basar.
    """
    for se in (37, 1):
        satir = _panel(_sevk("h2_morse", dt=20.0, n_steps=400, save_every=se), tmp_path, capsys)
        yakinsama = satir["Yakınsama"]
        assert "✓" in yakinsama and "energy_drift=6.147e-04," in yakinsama, (se, yakinsama)


def test_D2iii_BICIM_KAPILARI(tmp_path):
    """Yörünge biçiminde potential verilmez; seri biçiminde zorunludur; biriktirmeyen
    kurucunun yörüngesi (termostat döngüsü, `total_all_steps` None) kareleri okur;
    tek karelik koşu iki biçimde de "en az 2 kayıt"; bozuk her adım serisi hata.
    """
    senaryo, yorunge = _kosu(_sevk("h2_morse", dt=20.0, n_steps=40), tmp_path, 1)
    with pytest.raises(InvalidInputError, match="potential verilmez"):
        energy_drift(senaryo, yorunge, yorunge.potential)
    with pytest.raises(InvalidInputError, match="potential zorunludur"):
        energy_drift(senaryo, yorunge.kinetic)
    kurulan = Trajectory(yorunge.times, yorunge.positions, yorunge.velocities,
                         yorunge.kinetic, yorunge.potential)
    assert energy_drift(senaryo, kurulan) == energy_drift(senaryo, yorunge.kinetic, yorunge.potential)
    tek, tek_yorunge = _kosu(_sevk("h2_morse", dt=20.0, n_steps=1), tmp_path, 5)
    assert tek_yorunge.total_all_steps.shape == (2,)
    with pytest.raises(InvalidInputError, match="en az 2"):
        energy_drift(tek, tek_yorunge)
    for bozuk in (np.ones((2, 3)), np.ones(1)):
        with pytest.raises(InvalidInputError, match="total_all_steps"):
            energy_drift(senaryo, Trajectory(yorunge.times, yorunge.positions, yorunge.velocities,
                                             yorunge.kinetic, yorunge.potential,
                                             total_all_steps=bozuk))


# ------------------------------------------- D2-ii-c: kısa koşu, AYRILAMIYOR --


def test_D2iic_KAMU_YOLU_KISA_HF_KOSUSU_SALINIMDAN_AYRILAMIYOR(tmp_path, capsys):
    """h2_aimd geometrisi, dt 12.014 (yol omega*dt 0.324), 42 adım (~2 periyot),
    save_every 1: sürüklenme 1.278e-02 > 5e-3 ✗; S 2.374e-02 ve omega*dt geçer.
    Salınımın uç bloklara bırakabileceği pay S x 1/(0.3236 x 4) = 1.834e-02 >
    eşiği aşan 7.778e-03 -> tanı "AYRILAMIYOR (YARGILANAMAZ ...)"; hüküm ✗ kalır.
    HEAD'in tanısı "sürüklenme eşiğin üstünde ya da yargılanamaz: ... save_every'yi
    ... değiştirip" diyordu.
    KIRMA: kapı yok sayılır -> "AYRILAMIYOR" yok, kırmızı.
    """
    satir = _panel(_sevk("h2_aimd", dt=12.014, n_steps=42, save_every=1), tmp_path, capsys)
    yakinsama = satir["Yakınsama"]
    assert "✗" in yakinsama and "✓" not in yakinsama
    for parca in ("energy_drift=1.278e-02,", "energy_oscillation=2.374e-02,", "omega_dt_path_frames=3.236e-01,"):
        assert parca in yakinsama, parca
    canli = _canli(satir)
    assert ("sürüklenme eşiğin üstünde ama bu koşuda Verlet salınımından AYRILAMIYOR (YARGILANAMAZ: "
            "salınımın uç bloklara bırakabileceği pay 1.834e-02 (S x 0.773), eşiği aşan kısım "
            "7.778e-03)") in canli, canli
    assert "salınım eşiğin" not in canli and "omega" not in canli and "save_every" not in canli


def test_D2iic_KAMU_YOLU_GERCEK_SURUKLENME_ESIGIN_USTUNDE_KALIR(tmp_path, capsys):
    """TB Fermi penceresi (zincir(16, 2, 0.05, 3), n_e 15, 3.16 K, dt 10 x 400,
    save_every 1): Fermi geçişinin GERÇEK sürüklenmesi 5.480e-02; salınımın payı
    S 7.298e-02 x 1/(0.2886 x 40) = 6.3e-03 farkı açıklamaz -> tanı eski "sürüklenme
    eşiğin üstünde ya da yargılanamaz" (AYRILAMIYOR DEĞİL).
    KIRMA: pay her zaman kaba sınır S (kat 1) -> AYRILAMIYOR, kırmızı.
    """
    x = 2.0 * np.arange(16) + 0.05 * np.random.default_rng(3).standard_normal(16)
    d = _sevk("tb_zincir_md", n_steps=400, save_every=1, konum=[[float(v), 0.0, 0.0] for v in x],
              parametre={"n_electron": 15, "electron_temperature_kelvin": 3.16})
    satir = _panel(d, tmp_path, capsys)
    assert "✗" in satir["Yakınsama"] and "energy_drift=5.480e-02," in satir["Yakınsama"]
    canli = _canli(satir)
    assert "sürüklenme eşiğin üstünde ya da yargılanamaz" in canli and "AYRILAMIYOR" not in canli


def _tb_senaryo(n_steps: int, dt: float = 10.0) -> MDScenario:
    return MDScenario(
        name="d2iic",
        force_field={"kind": "tb_ss_sigma", "params": dict(
            ss_sigma=_YASA, itme=_ITME, n_electron=16, electron_temperature_kelvin=631.6, box_x_bohr=32.0)},
        initial={"positions_bohr": [[2.0 * i, 0.0, 0.0] for i in range(16)], "masses_amu": [_M_H]},
        thermostat=None, dt=dt, n_steps=n_steps,
    )


def test_D2iic_PAY_FORMULU_S_x_min_1_bolu_omega_dt_L_ve_HUKUM_DEGISMEZ():
    """Pay = S x min(1, 1/(omega*dt x L)), L = (n_steps + 1) // min(10, n_steps + 1).
    n_steps 99 -> L 10; omega_yol 0.05, dt 10 -> omega*dt 0.5 -> kat 0.2; S 0.04 ->
    pay 0.008. Sürüklenme eşik + 0.99 pay -> AYRILAMIYOR, eşik + 1.01 pay -> eski
    metin; ikisinde de ✗ (kapı hükmü DEĞİŞTİRMEZ). omega*dt x L <= 1 -> kaba sınır
    (kat 1); omega `nan` -> B8'den (grup 9) beri kaba sınır DEĞİL, "ayrılıp
    ayrılamadığı da YARGILANAMAZ" (test_B8_...); S `nan` -> "payı bilinmiyor";
    sürüklenme geçerse kapı sessiz. HF dalı aynı metni basar.
    Sınırın kendisi (sürüklenme - pay == eşik, IEEE'de tam kurulur) AYRILAMIYOR,
    bir ulp üstü değil.
    KIRMA: L = n_steps // 10 (9, pay 0.0089) / 2 / (omega*dt x L) / kapı ✓ basar /
    kaba sınırda kat 0 / '>' yerine '>=' -> kırmızı.
    """
    tb = _tb_senaryo(99)
    esik = fabrika._TB_DRIFT_TOLERANCE
    ortak = dict(energy_oscillation=0.04, omega_max_frame0=1e-9, omega_max_path_frames=0.05)
    for carpan, ayrilamaz in ((0.99, True), (1.01, False)):
        p = provenance_for(tb, energy_drift=esik + carpan * 0.008, **ortak)
        canli = p.convergence_diagnosis.split(" GRİ BÖLGE")[0]
        assert p.converged is False
        assert ("AYRILAMIYOR" in canli) is ayrilamaz, (carpan, canli)
        assert "sürüklenme eşiğin üstünde" in canli
        if ayrilamaz:
            assert "pay 8.000e-03 (S x 0.2)" in canli, canli
    # Kaba sınır: omega*dt x L <= 1 (omega_yol 1e-3 -> 0.1) -> kat 1 (omega nan: test_B8_...).
    p = provenance_for(tb, energy_drift=esik + 0.99 * 0.04, **(ortak | dict(omega_max_path_frames=1e-3)))
    assert "pay 4.000e-02 (S x 1)" in p.convergence_diagnosis
    p = provenance_for(tb, energy_drift=esik + 0.01, **(ortak | dict(energy_oscillation=float("nan"))))
    assert "S sonlu değil, salınımın payı bilinmiyor" in p.convergence_diagnosis
    gecen = provenance_for(tb, energy_drift=esik, **ortak).convergence_diagnosis.split(" GRİ BÖLGE")[0]
    s_tam = 2.0 ** -10  # esik + s_tam - s_tam == esik tam (önkoşul aşağıda)
    d_tam = esik + s_tam
    assert d_tam - s_tam == esik
    ince = dict(energy_oscillation=s_tam, omega_max_frame0=1e-9, omega_max_path_frames=1e-9)
    canlilar = [provenance_for(tb, energy_drift=d, **ince).convergence_diagnosis.split(" GRİ BÖLGE")[0]
                for d in (d_tam, float(np.nextafter(d_tam, 1.0)))]
    assert "AYRILAMIYOR" in canlilar[0] and "AYRILAMIYOR" not in canlilar[1], canlilar
    assert "AYRILAMIYOR" not in gecen and "sürüklenme eşiğin" not in gecen
    hf = MDScenario(
        name="d2iic", force_field={"kind": "hartree_fock", "params": {"symbols": ["H", "H"], "basis": "sto-3g"}},
        initial={"positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], "masses_amu": [_M_H, _M_H]},
        thermostat=None, dt=10.0, n_steps=99,
    )
    p = provenance_for(hf, energy_drift=fabrika._MD_DRIFT_TOLERANCE + 0.99 * 0.008, **ortak)
    assert p.converged is False and "AYRILAMIYOR (YARGILANAMAZ: " in p.convergence_diagnosis


# ------------------------------------------------ D2-v: yuvarlama tabanı --


def test_D2v_KAMU_YOLU_TB_DIMER_YUVARLAMA_TABANINDA_S_YARGILANAMAZ(tmp_path, capsys):
    """TB dimeri dengede + bağ boyunca 1e-7 bohr, v0 0, omega0*dt 0.2, 94 adım:
    max KE / (eps |E|) ~ 67. HEAD ✓ basıyordu: S 4.859e-02 (gerçeği 1.010e-02;
    eşiğin hemen altında, gürültüyle) ve sürüklenme tam 0 (yuvarlama rastlantısı).
    Şimdi S `nan`, "salınım YARGILANAMAZ (... yuvarlama tabanında ...)", ✗.
    Kontrol: 1e-4 bohr'da S 1.010e-02 ✓ (sonlu; tek mod analitiği x^2/(4 - x^2)
    = 1.010e-02). Gürültü değerleri kilitlenmez.
    KIRMA: taban yok (c 0) -> 1e-7 hücresi sonlu S; c 1e9 -> kontrol nan.
    """
    satir = _panel(_tb_dimer(_TB_RE + 1e-7), tmp_path, capsys)
    assert "✗" in satir["Yakınsama"] and "energy_oscillation=nan," in satir["Yakınsama"]
    canli = _canli(satir)
    assert ("salınım YARGILANAMAZ (S sonlu değil: max KE = 0 ya da yuvarlama tabanında -- "
            "max KE <= 3000 x eps x max|KE + Omega|") in canli, canli
    kontrol = _panel(_tb_dimer(_TB_RE + 1e-4), tmp_path, capsys)["Yakınsama"]
    assert "✓" in kontrol and "energy_oscillation=1.010e-02," in kontrol


def test_D2v_TABAN_FORMULU_c_eps_max_mutlak_ENERJI_SINIRDA_YARGILANAMAZ():
    """S = nan <=> max KE <= 3000 x eps x max(|total_min|, |total_max|). Sınırın
    kendisi nan, bir ulp üstü sonlu; ölçek iki uçtan büyük olanı (|min| > |max| ve
    tersi); uçlardan biri nan -> nan; KE 0 -> nan. Sabit literal 3.0e03.
    KIRMA: c 0 / c x 10 / ölçek yalnız total_max / '>' yerine '>=' -> kırmızı.
    """
    assert fabrika._OSCILLATION_ROUNDING_FLOOR == 3.0e03
    tb = _tb_senaryo(10)
    eps = float(np.finfo(float).eps)
    bos = np.zeros(2)

    def s(ke, tmin, tmax):
        y = Trajectory(bos, np.zeros((2, 1, 3)), np.zeros((2, 1, 3)), bos, bos,
                       total_min=tmin, total_max=tmax, kinetic_max=ke)
        return energy_oscillation(tb, y)

    for tmin, tmax in ((-2.0, -1.999), (1.999, 2.0)):
        sinir = 3.0e03 * eps * 2.0
        assert np.isnan(s(sinir, tmin, tmax)), (tmin, tmax)
        ust = float(np.nextafter(sinir, 1.0))
        assert s(ust, tmin, tmax) == pytest.approx(1e-3 / ust, rel=1e-12)
    assert np.isnan(s(1.0, float("nan"), 2.0)) and np.isnan(s(1.0, -2.0, float("nan")))
    assert np.isnan(s(0.0, 0.0, 0.0))


# ------------------------------------- B8 (grup 9): ölçülemeyen omega, gürültülü HF --


def _hf_h2(r: float) -> tuple[MDScenario, MDState]:
    hf = MDScenario(
        name="b8", force_field={"kind": "hartree_fock", "params": {"symbols": ["H", "H"], "basis": "sto-3g"}},
        initial={"positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, r]], "masses_amu": [_M_H, _M_H]},
        thermostat=None, dt=5.0, n_steps=2,
    )
    m = amu_to_electron_mass(np.array([_M_H, _M_H]))
    return hf, MDState(positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]]), velocities=np.zeros((2, 3)), masses=m)


def test_B8_GURULTULU_HF_HESSIANI_OLCULEMEDI_nan_TEMIZ_HF_SAYI():
    """`_omega_max`: HF'de lambda_max <= 60 x u (u = eps |E| / (h s m_min), sayısal
    gradyanın gürültü tabanı) ise omega ÖLÇÜLEMEDİ -> `nan`. ÇIPA: bağımsız referans
    aynı geometrinin temiz (h 1e-5) Hessian'ı. ÖLÇÜLDÜ (bu makine, H2/STO-3G):
    r 1.2 -- h 1e-12 / 1e-13'te lambda / u 2.7 / 1.6; HEAD omega 0.0604 / 0.1486
    basıyordu (temizi 0.03156: 1.9 / 4.7 kat). r 1.8 -- h 1e-10'da lambda / u 12.5
    (HEAD 0.01277, %4.5 hata, ama tabanda hata öngörülemez: aynı tabanda %93900'e
    kadar ölçüldü), h 1e-9'da 112 (0.012118, temizden %0.8) -> SAYI kalır.
    BANT KİLİDİ (12.5 < c < 112, ölçülmüş iki hücreden; c = 60): tabanın katını
    10'a indiren ya da 1000'e çıkaran mutant kırmızı. Sayılar gürültüdür ve OpenBLAS'a
    bağlıdır, ama iki hücrenin c'ye uzaklığı 4.8 ve 1.9 kat. 5 Hessian, ~0.5 s.
    KIRMA (mut.py): taban yok (yalnız lambda <= 0) / c 1 / c 10 / c 1000 -> kırmızı.
    """
    for r, temiz_h, olculemez, sayi in ((1.2, 1e-5, (1e-12, 1e-13), None), (1.8, 1e-5, (1e-10,), 1e-9)):
        hf, durum = _hf_h2(r)
        temiz = omega_max_frame0(hf, durum, HartreeFockGradient(symbols=("H", "H"), step=temiz_h))
        assert np.isfinite(temiz) and temiz > 0.0
        for h in olculemez:
            w = omega_max_frame0(hf, durum, HartreeFockGradient(symbols=("H", "H"), step=h))
            assert np.isnan(w), (r, h, w, temiz)
        if sayi is not None:
            w = omega_max_frame0(hf, durum, HartreeFockGradient(symbols=("H", "H"), step=sayi))
            assert abs(w / temiz - 1.0) < 0.05, (r, sayi, w, temiz)


_M_T = 3.01604928  # trityum, amu


def test_B8_TABAN_EN_KUCUK_KUTLEYI_okur_ESIT_OLMAYAN_KUTLE_HT():
    """V9 (grup 9 düzeltici, 23.10.2026): tabandaki m_min KİLİDİ. Kütle ağırlıklı Hessian'da
    gürültü d H / sqrt(m_i m_j) en çok en HAFİF atomun bloğunda büyür; taban onu okumalı.
    Yukarıdaki testler yalnız EŞİT kütleli H2 kullanıyordu: `kutle.min()` -> `kutle.max()`
    mutantı (V_B8_01) hayattaydı. HT (H2'nin elektronik yapısı, kütleler 1.0078 / 3.0160,
    oran 2.993; ucuz -- LiH aynı kilidi Hessian başına ~3 s'de verirdi, ölçüldü).
    ÖLÇÜLDÜ (bu makine, r 1.8): h 5e-10'da lambda / u_min 38.1 -> nan (c = 60'ın 1.57 kat
    altı); m_max ile u 2.993 kat küçülür, lambda / u_max 114 -> SAYI (60'ın 1.90 kat üstü).
    h 1e-9'da lambda / u_min 75.3 -> SAYI, temizden %0.7. ÇAPA: aynı geometrinin temiz
    (h 1e-5) Hessian'ı, 0.009980. 3 Hessian, ~0.6 s.
    """
    r = 1.8
    hf = MDScenario(
        name="b8-ht", force_field={"kind": "hartree_fock", "params": {"symbols": ["H", "H"], "basis": "sto-3g"}},
        initial={"positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, r]], "masses_amu": [_M_H, _M_T]},
        thermostat=None, dt=5.0, n_steps=2,
    )
    m = amu_to_electron_mass(np.array([_M_H, _M_T]))
    durum = MDState(positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]]), velocities=np.zeros((2, 3)), masses=m)

    def omega(h):
        return omega_max_frame0(hf, durum, HartreeFockGradient(symbols=("H", "H"), step=h))

    temiz = omega(1e-5)
    assert np.isfinite(temiz) and temiz > 0.0
    w = omega(5e-10)
    assert np.isnan(w), (w, temiz)
    w = omega(1e-9)
    assert abs(w / temiz - 1.0) < 0.05, (w, temiz)


def test_B8_TB_POZITIF_MODSUZ_HESSIAN_OLCULEMEDI_nan(tmp_path):
    """TB dimeri kesmenin ötesinde (d 5 > r_cut 3.8): Hessian tam sıfır, pozitif mod yok
    -> `nan` (YARGILANAMAZ). HEAD 0.0 döndürüyordu ve omega*dt ölçütü onunla ✓ geçiyordu.
    Kesmenin içinde (d = r_e) sayı kalır. KIRMA: `lam > taban` yerine `lam >= taban` ya da
    eski `sqrt(max(lam, 0))` -> 0.0, kırmızı."""
    for r, beklenen_sayi in ((5.0, False), (_TB_RE, True)):
        senaryo = load_scenario(_yaz(tmp_path / f"d{r}.yaml", _tb_dimer(r)))
        durum, alan, _ = build(senaryo)
        w = omega_max_frame0(senaryo, durum, alan)
        assert bool(np.isfinite(w)) is beklenen_sayi, (r, w)
    assert w == pytest.approx(_TB_OMEGA0, rel=1e-9)


@pytest.mark.parametrize("w", [0.0, -0.0, float("nan")], ids=["sifir", "eksi-sifir", "nan"])
def test_B8_provenance_OLCULEMEYEN_omega_YARGILANAMAZ_AYRIM_da_YARGILANAMAZ(w):
    """Doğrudan API: omega <= 0 (-0.0 dahil) ya da `nan` ölçüt ✓ GEÇEMEZ (HEAD'de 0 ve -0.0
    eşiğin altında sayılıyordu); sürüklenme ✗ iken "AYRILAMIYOR" basılmaz -- pay omega'dan
    kestirilir, omega ölçülemeyince tanı "ayrılıp ayrılamadığı da YARGILANAMAZ" der. TB
    ve HF aynı metni basar. Kontrol: ölçülmüş omega (0.05) ile D2-ii-c değişmedi.
    KIRMA: `> 0.0` koşulu kare 0'dan ya da yoldan kalkar / AYRILAMIYOR kapısı yol_sonlu'ya
    bakmaz (HEAD) / tanılardan yeni dal kalkar -> kırmızı.
    """
    hf, _ = _hf_h2(1.4)
    hf = dataclasses.replace(hf, dt=10.0, n_steps=99)
    for sen, esik in ((_tb_senaryo(99), fabrika._TB_DRIFT_TOLERANCE), (hf, fabrika._MD_DRIFT_TOLERANCE)):
        ortak = dict(energy_oscillation=0.04, omega_max_frame0=1e-9, omega_max_path_frames=0.05)
        assert provenance_for(sen, energy_drift=1e-9, **ortak).converged
        kare0 = provenance_for(sen, energy_drift=1e-9, **(ortak | dict(omega_max_frame0=w)))
        assert not kare0.converged and "a priori tanı YARGILANAMAZ" in kare0.convergence_diagnosis
        yol = provenance_for(sen, energy_drift=1e-9, **(ortak | dict(omega_max_path_frames=w)))
        assert not yol.converged and "tanısı YARGILANAMAZ" in yol.convergence_diagnosis
        surukle = provenance_for(sen, energy_drift=esik + 0.99 * 0.008, **(ortak | dict(omega_max_path_frames=w)))
        canli = surukle.convergence_diagnosis.split(" GRİ BÖLGE")[0]
        assert "AYRILAMIYOR" not in canli and "ayrılıp ayrılamadığı da YARGILANAMAZ" in canli, canli
        olculmus = provenance_for(sen, energy_drift=esik + 0.99 * 0.008, **ortak).convergence_diagnosis
        assert "AYRILAMIYOR (YARGILANAMAZ: " in olculmus.split(" GRİ BÖLGE")[0]


def test_B8_KAMU_YOLU_GURULTULU_HF_20_ADIM_omega_nan_AYRILAMIYOR_YOK(tmp_path, capsys):
    """h2_aimd + gradyan adımı 1e-13 (sonlu fark gürültüsü), 20 adım. HEAD 448f808: kare-0 ve
    yol omega*dt 0.000e+00 (ölçüt ✓ geçiyordu) ve sürüklenme ✗'ine (1.115e-01) "AYRILAMIYOR"
    deniyordu (kaba sınır). Şimdi ikisi `nan` (YARGILANAMAZ), AYRILAMIYOR yok, hüküm ✗.
    """
    satir = _panel(_sevk("h2_aimd", n_steps=20, parametre={"step": 1e-13}), tmp_path, capsys, "b8")
    yakinsama = satir["Yakınsama"]
    assert "✗" in yakinsama and "energy_drift=1.115e-01," in yakinsama
    assert "omega_dt_frame0=nan," in yakinsama and "omega_dt_path_frames=nan," in yakinsama
    canli = _canli(satir)
    assert "AYRILAMIYOR" not in canli and "ayrılıp ayrılamadığı da YARGILANAMAZ" in canli, canli
    assert "a priori tanı YARGILANAMAZ" in canli and "yol karesinin tanısı YARGILANAMAZ" in canli


@pytest.mark.slow
def test_B8_KAMU_YOLU_GURULTULU_HF_400_ADIM_EKSI_SIFIR_BASILMAZ(tmp_path, capsys):
    """Grup 7 c.md madde 2'nin kendi hücresi: h2_aimd + gradyan adımı 1e-13, 400 adım. HEAD
    448f808 yol karesi omega'sı için `-0.000e+00` basıyordu (lambda_max -0.0, max(-0.0, 0.0)
    = -0.0) ve gerçek 400 adımlık sürüklenmeye (4.295e-01) "AYRILAMIYOR" diyordu. ~3.5 s."""
    satir = _panel(_sevk("h2_aimd", n_steps=400, parametre={"step": 1e-13}), tmp_path, capsys, "b8u")
    yakinsama = satir["Yakınsama"]
    assert "✗" in yakinsama and "energy_drift=4.295e-01," in yakinsama and "-0.000e+00" not in yakinsama
    assert "omega_max_path_frames=nan," in yakinsama
    canli = _canli(satir)
    assert "AYRILAMIYOR" not in canli and "ayrılıp ayrılamadığı da YARGILANAMAZ" in canli, canli
