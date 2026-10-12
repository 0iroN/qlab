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

from pathlib import Path

import numpy as np
import pytest
import yaml

import qlab.engine.md.factory as fabrika
from qlab.engine.md.factory import build, energy_drift, energy_oscillation, provenance_for
from qlab.engine.md.integrator import Trajectory, run
from qlab.errors import InvalidInputError
from qlab.run import main
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import MDScenario

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
    metin; ikisinde de ✗ (kapı hükmü DEĞİŞTİRMEZ). omega*dt x L <= 1 ya da omega
    `nan` -> kaba sınır (kat 1); S `nan` -> "payı bilinmiyor"; sürüklenme geçerse
    kapı sessiz. HF dalı aynı metni basar.
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
    # Kaba sınır: omega*dt x L <= 1 (omega_yol 1e-3 -> 0.1) ve omega nan -> kat 1.
    for w in (1e-3, float("nan")):
        p = provenance_for(tb, energy_drift=esik + 0.99 * 0.04, **(ortak | dict(omega_max_path_frames=w)))
        assert "pay 4.000e-02 (S x 1)" in p.convergence_diagnosis, w
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
