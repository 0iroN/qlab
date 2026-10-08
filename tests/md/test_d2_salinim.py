"""D2 (09.10.2026) — TB NVE'de iri dt'nin SALINIM hükmü.

Envanter D2: TB dalının yakınsama hükmü yalnız sürüklenmeye bakıyordu ve
sürüklenme iri dt'nin Verlet salınımına kördür (`_TB_DRIFT_TOLERANCE`
yorumu: omega_max*dt >= 0.93 olan 33 hücrenin 8'i eşiğin altında). Ön ölçüm
(wf_e71a1dcb-1dc, `.superpowers/sdd/d2-olcum/`) tasarımı kesinleştirdi:

* S = [max(KE + Omega) - min(KE + Omega)] / max(KE), uçlar integratörün
  HER adımından (kayıtlı karelerden değil -- save_every'ye göre sahte ✓/✗);
* eşik 5e-02, yalnız `tb_ss_sigma` NVE; Yakınsama = sürüklenme VE S;
* `provenance_for` TB NVE'de S almazsa hata verir; öteki koşularda S verilirse
  de hata verir.

T1/T1b, T4, iki dürüst olumsuz ve `nan` kilidi KAMU YOLUNDAN (YAML ->
`qlab.run.main`) koşar; T3 kamu yükleyicisi + `build` + integratörle, T5-T9
birim/kapı testleridir. Sayılar eleştirmenin kamu yolu logundan
(`d2-olcum/elestirmen/e1_kamu_yolu.log`), doğrulama koşusundan
(wf_6b0da0f9-95f, `d2-olcum/dogrulama/`) ve bu oturumda yeniden üretildi.
T2 (sevk edilen senaryo) `test_tb_kuvvet.py`nin CLI testindedir.

SÜRE (ölçüldü, 09.10.2026, bu makine): 13 test 1.39 s, en yavaşı 0.26 s;
hiçbiri `slow` değildir.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import qlab
import qlab.engine.md.factory as fabrika
from qlab.engine.md.factory import build, energy_oscillation, provenance_for
from qlab.engine.md.forcefields import Harmonic
from qlab.engine.md.integrator import Trajectory, run
from qlab.engine.md.state import MDState
from qlab.errors import InvalidInputError
from qlab.run import main
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import MDScenario
from qlab.units import amu_to_electron_mass

_M_H = 1.00782503207


def _zincir(tohum: int | None) -> np.ndarray:
    """a = 2 bohr, 16 atomlu zincirin x'leri; `test_tb_kuvvet._zincir` ile
    AYNI işlemler (aynı tohum -> bit bit aynı konum). `None`: gürültüsüz."""
    x = 2.0 * np.arange(16)
    if tohum is not None:
        x = x + 0.05 * np.random.default_rng(tohum).standard_normal(16)
    return x


def _yaml(
    yol: Path,
    x: np.ndarray,
    dt: float,
    n: int,
    save_every: int = 1,
    hizlar: np.ndarray | None = None,
    kutleler: tuple[float, ...] = (_M_H,),
    n_e: int = 16,
) -> Path:
    """Ortak pencere (ss_sigma -0.1/2/2, itme 0.05/2/6, r_on 3.0, r_cut 3.8,
    631.6 K, L 32). `x`/`hizlar` (N,) ise yalnız x bileşeni, (N, 3) ise tam.
    Sayılar `repr` ile: YAML'dan bit bit geri döner."""

    def uc(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        return np.c_[a, np.zeros((len(a), 2))] if a.ndim == 1 else a

    satirlar = [
        "kind: md", "name: d2", "force_field:", "  kind: tb_ss_sigma", "  params:",
        "    ss_sigma: {genlik_hartree: -0.1, d0_bohr: 2.0, us: 2.0, r_on_bohr: 3.0, r_cut_bohr: 3.8}",
        "    itme: {genlik_hartree: 0.05, d0_bohr: 2.0, us: 6.0, r_on_bohr: 3.0, r_cut_bohr: 3.8}",
        f"    n_electron: {n_e}", "    electron_temperature_kelvin: 631.6", "    box_x_bohr: 32.0",
        "initial:", "  positions_bohr:",
    ]
    satirlar += [f"    - [{a!r}, {b!r}, {c!r}]" for a, b, c in uc(x).tolist()]
    if hizlar is not None:
        satirlar += ["  velocities:"] + [f"    - [{a!r}, {b!r}, {c!r}]" for a, b, c in uc(hizlar).tolist()]
    satirlar += [
        "  masses_amu: [" + ", ".join(repr(float(m)) for m in kutleler) + "]",
        f"dt: {dt!r}", f"n_steps: {n}", f"save_every: {save_every}", "",
    ]
    yol.write_text("\n".join(satirlar), encoding="utf-8")
    return yol


def _panel(yol: Path, tmp_path: Path, capsys) -> dict[str, str]:
    """Kamu yolu: `python -m qlab.run <yaml> -o x.npz`; etiket -> satır."""
    assert main([str(yol), "-o", str(tmp_path / "d2.npz")]) == 0
    ekran = capsys.readouterr().out
    return {s.split(":", 1)[0].strip(): s for s in ekran.splitlines() if ":" in s}


def _tb_senaryo(thermostat: dict | None = None) -> MDScenario:
    """Yalnız `provenance_for` için en küçük geçerli TB senaryosu."""
    yasa = lambda g, u: dict(genlik_hartree=g, d0_bohr=2.0, us=u, r_on_bohr=3.0, r_cut_bohr=3.8)  # noqa: E731
    return MDScenario(
        name="d2",
        force_field={"kind": "tb_ss_sigma", "params": dict(
            ss_sigma=yasa(-0.1, 2.0), itme=yasa(0.05, 6.0), n_electron=16,
            electron_temperature_kelvin=631.6, box_x_bohr=32.0,
        )},
        initial={"positions_bohr": [[2.0 * i, 0.0, 0.0] for i in range(16)], "masses_amu": [_M_H]},
        thermostat=thermostat,
        dt=10.0,
        n_steps=1,
    )


_NVT = {"kind": "langevin", "params": {"temperature": 300.0, "gamma": 1e-3}}


# --------------------------------------------------- T1 / T1b: sahte ✓ kapandı --


@pytest.mark.parametrize(
    ("tohum", "dt", "se", "drift", "salinim"),
    [(4, 40.0, 2, "1.401e-03", "2.310e-01"), (2, 35.0, 1, "7.806e-04", "1.767e-01")],
    ids=["T1-D2-hucresi-se2", "T1b-K1-se1"],
)
def test_T1_IRI_dt_SURUKLENME_GECIRIYORDU_SALINIM_YAKALIYOR(tohum, dt, se, drift, salinim, tmp_path, capsys):
    """İri dt: sürüklenme eşiğin ALTINDA, panel yine ✗ -- sebebi salınım.

    Pencere: zincir(16, 2, 0.05, tohum), n_e 16, 631.6 K, L 32, M_H, v0 = 0,
    100 adım. ÖLÇÜLDÜ (eleştirmen kamu yolu logu, bu oturumda yeniden
    üretildi): T1 (tohum 4, dt 40) save_every 1'de sürüklenme 8.788e-03,
    save_every 2'de 1.401e-03, S her ikisinde 2.3100e-01; T1b (tohum 2, dt
    35, save_every 1) sürüklenme 7.806e-04, S 1.7673e-01. D2 öncesi panel
    hepsinde ✓ basıyordu (sürüklenme eşiği 1e-02).

    T1 BİLEREK save_every 2: S'yi kayıtlı karelerden okuyan bir `run.py`
    (yerleşik min/max ya da `dataclasses.replace` ile, AST kilidini atlatan
    biçimde -- doğrulama N05/N12) burada 2.296e-01 basardı. save_every 1'de
    kare = adım olduğu için bu mutantlar görünmezdi.

    Tanı satırı YALNIZ salınımı suçlar: sürüklenme bu hücrelerde gerçekten
    geçiyor -- hüküm iki ölçütün VE'sidir, birinin yerine geçmez.

    KIRMA: S yok sayılır (yakinsadi yalnız sürüklenme) -> ikisi de ✓,
    kırmızı; eşik 0.5 -> ikisi de ✓ (0.231, 0.177 < 0.5), kırmızı.
    """
    yol = _yaml(tmp_path / "t1.yaml", _zincir(tohum), dt, 100, save_every=se)
    satir = _panel(yol, tmp_path, capsys)
    yakinsama = satir["Yakınsama"]
    assert "✗" in yakinsama and "✓" not in yakinsama, yakinsama
    assert f"energy_drift={drift}," in yakinsama
    assert f"energy_oscillation={salinim}," in yakinsama
    tani = satir["Yakınsama tanısı"]
    assert "salınım eşiğin üstünde" in tani
    assert "sürüklenme eşiğin üstünde" not in tani


# ---------------------------------------------- T3: S kayıt seyreltmesinden bağımsız --


def test_T3_S_HER_ADIMDAN_save_every_ile_BIT_BIT_AYNI(tmp_path):
    """Aynı hücre save_every 1 / 2 / 5 / 20: S BİT BİT aynı.

    T1 hücresi (tohum 4, dt 40 x 100), kamu yükleyicisi + `build` +
    integratör (`run.py::_run_md` ile aynı sıra: COM hızı çıkarılır). ÖLÇÜLDÜ:
    her adım S = 0.2309989852933888 dört save_every'de de; kayıtlı
    karelerden okunan S ise 2.3100e-01 / 2.2964e-01 / 2.1174e-01 /
    2.0806e-01. Yalnız HÜKMÜ karşılaştırmak "S karelerden" mutantını
    öldürmezdi (dördü de ✗); değer karşılaştırılır. Pozitif kontrol: aynı
    kayıtlarda kare-S save_every 20'de gerçekten ayrışıyor.
    """
    senaryo = load_scenario(_yaml(tmp_path / "t3.yaml", _zincir(4), 40.0, 100))
    durum, alan, _ = build(senaryo)
    durum = durum.remove_center_of_mass_motion()
    degerler = {}
    for se in (1, 2, 5, 20):
        yorunge = run(durum, alan, dt=senaryo.dt, n_steps=senaryo.n_steps, save_every=se)
        degerler[se] = energy_oscillation(senaryo, yorunge)
        kare = float(np.ptp(yorunge.kinetic + yorunge.potential) / yorunge.kinetic.max())
    assert len(set(degerler.values())) == 1, degerler
    assert abs(degerler[1] - 0.2309989852933888) < 1e-12
    assert abs(kare - degerler[1]) > 0.05 * degerler[1], (kare, degerler[1])


# ------------------------------------------ T4: yuvarlama düzeyinde KE, sahte ✓ --


def test_T4_DUZGUN_ZINCIR_yuvarlama_KE_HER_CEKIRDEKTE_X_bu_makinede_S_YAKALAR(tmp_path, capsys):
    """Gürültüsüz düzgün zincir, v0 = 0, dt 10 x 80: KE yuvarlama düzeyinde, panel ✗.

    Simetri kuvveti sıfırlar; KE yuvarlama düzeyinde kalır (ölçülen max KE
    4.6e-18). BU MAKİNEDE (OpenBLAS SkylakeX çekirdeği) blok farkı tam 0.0
    çıkar ve D2 öncesi panel bunun yanında ✓ basıyordu. "Tam 0" bir yuvarlama
    rastlantısıdır (doğrulama, 09.10.2026): Haswell / Sandybridge / Katmai
    çekirdeklerinde sürüklenme kendisi ✗ verir ve S'nin değeri de değişir
    (Katmai'de 3.861). Bu yüzden kilit taşınabilir biçimde yazılır: panel HER
    çekirdekte ✗; sürüklenme eşiğin altındaysa (bu makinede olduğu gibi) ✗'yi
    salınım vermiştir. Sayıların kendisi kilitlenmez.
    """
    satir = _panel(_yaml(tmp_path / "u0.yaml", _zincir(None), 10.0, 80), tmp_path, capsys)
    yakinsama = satir["Yakınsama"]
    assert "✗" in yakinsama, yakinsama
    drift = float(yakinsama.split("energy_drift=", 1)[1].split(",", 1)[0])
    if drift <= fabrika._TB_DRIFT_TOLERANCE:
        tani = satir["Yakınsama tanısı"]
        assert "salınım eşiğin üstünde" in tani or "salınım YARGILANAMAZ" in tani, tani


# ------------------------------------------------------- T5: eşik ulp kilidi --


def test_T5_SALINIM_ESIGI_PANELDE_ve_HUKUMDE_AYNI_SAYI():
    """Panelin `energy_oscillation_tolerance`ının bir ulp altı ✓, üstü ✗.

    Sürüklenme eşiğin altında sabit (1e-9): hükmü yalnız S belirler. Payda adı
    panelde ve sürüklenmeninkinden AYRI (`max_kinetic_all_steps`). Klasik
    dalın panelinde salınım anahtarı yok.
    KIRMA: tanıya başka sayı basılır -> ulp üstü ✓; eşik 1e-2 / 1e-1 ->
    kırmızı (literal 5e-02).
    """
    tb = _tb_senaryo()
    t = provenance_for(tb, energy_drift=1e-9, energy_oscillation=1e-9).diagnostics
    esik = t["energy_oscillation_tolerance"]
    assert esik == 5.0e-02
    assert t["energy_oscillation_scale"] == "max_kinetic_all_steps"
    assert provenance_for(
        tb, energy_drift=1e-9, energy_oscillation=float(np.nextafter(esik, 0.0))
    ).converged
    assert not provenance_for(
        tb, energy_drift=1e-9, energy_oscillation=float(np.nextafter(esik, 1.0))
    ).converged
    # Eşiğin KENDİSİ ✓ ('<='; sürüklenme dalıyla aynı anlam -- doğrulama N03).
    assert provenance_for(tb, energy_drift=1e-9, energy_oscillation=esik).converged
    morse = dataclasses.replace(tb, force_field={"kind": "morse", "params": {
        "d_e": 0.174434333, "r_e": 1.401118538, "a": 1.028997701}})
    assert not any(k.startswith("energy_oscillation") for k in provenance_for(
        morse, energy_drift=1e-9).diagnostics)


# ------------------------------------------------ T6: kapı ve NVT'de eşik yok --


def test_T6_KAPI_TB_NVE_S_ZORUNLU_ve_OTEKI_KOSULARDA_YASAK():
    """TB NVE'de S verilmezse HATA ("yok say" yeni bir sahte ✓ kapısıydı);
    NVT'de ve klasik dalda S verilirse HATA; NVT'de eşik `None` (0.0 DEĞİL),
    hüküm ✓ ve `energy_oscillation()` NVT'de None döner.

    Termostatlı TB koşusu HİÇ koşulmadı (G5 beyanı); bu test yalnız panelin
    ne dediğini kilitler.
    """
    tb = _tb_senaryo()
    with pytest.raises(InvalidInputError, match="salınım ölçütü zorunludur"):
        provenance_for(tb, energy_drift=1e-9)
    nvt = _tb_senaryo(_NVT)
    with pytest.raises(InvalidInputError, match="yalnız TB NVE"):
        provenance_for(nvt, energy_drift=5.0, energy_oscillation=1e-9)
    prov = provenance_for(nvt, energy_drift=5.0)
    assert prov.converged is True
    assert prov.diagnostics["energy_oscillation"] is None
    assert prov.diagnostics["energy_oscillation_tolerance"] is None
    # Hesaplanmayan S'nin paydası ✓'nin yanında basılmaz (doğrulama, 09.10).
    assert prov.diagnostics["energy_oscillation_scale"] is None
    assert prov.convergence_diagnosis is None
    with pytest.raises(InvalidInputError, match="negatif olamaz"):
        provenance_for(tb, energy_drift=1e-9, energy_oscillation=-1e-3)
    bos = Trajectory(*(np.zeros(2),) * 5, total_min=0.0, total_max=1.0, kinetic_max=1.0)
    assert energy_oscillation(nvt, bos) is None
    morse = dataclasses.replace(tb, force_field={"kind": "morse", "params": {}})
    assert energy_oscillation(morse, bos) is None
    with pytest.raises(InvalidInputError, match="yalnız TB NVE"):
        provenance_for(morse, energy_drift=1e-9, energy_oscillation=1e-9)


# ----------------------------------------------- T7: hareketsiz / ölçülmemiş --


def test_T7_HAREKETSIZ_ve_UCLARI_BIRIKTIRMEYEN_KOSU_YARGILANAMAZ():
    """max KE = 0 ve varsayılan `nan` uçlar (ör. `run.py`nin termostat
    döngüsünün kurduğu `Trajectory`) -> S `nan`, panel ✗ ve tanı
    "YARGILANAMAZ" -- sürüklenme geçse bile. KIRMA: `nan` yerine 0.0 dönen
    mutant ✓ basar, kırmızı."""
    tb = _tb_senaryo()
    sifir = Trajectory(*(np.zeros(3),) * 5, total_min=-1.2, total_max=-1.2, kinetic_max=0.0)
    olculmemis = Trajectory(*(np.zeros(3),) * 5)
    # Varsayılan ÜÇ alan da nan ("ölçülmedi"), 0.0 değil: panel için eşdeğer
    # olurdu (kinetic_max 0 da nan verir) ama alanları doğrudan okuyan bir
    # tüketici için değil (doğrulama N06).
    assert all(np.isnan(v) for v in (
        olculmemis.total_min, olculmemis.total_max, olculmemis.kinetic_max))
    for yorunge in (sifir, olculmemis):
        s = energy_oscillation(tb, yorunge)
        assert np.isnan(s)
        prov = provenance_for(tb, energy_drift=1e-9, energy_oscillation=s)
        assert prov.converged is False
        assert "salınım YARGILANAMAZ" in prov.convergence_diagnosis
        assert "sürüklenme eşiğin üstünde" not in prov.convergence_diagnosis


# ------------------------------------------- T8: integratörün uçları, analitik --


def test_T8_HARMONIK_omega_dt_KOK2_S_TAM_1_ve_save_every_KOR_DEGIL():
    """Tek parçacık, k = 2, m = 1, dt = 1 (omega*dt = sqrt(2), sayısal faz
    adımı pi/2), x0 = 1, v0 = 0: velocity-Verlet ELLE iki adım --
    (x, v) = (1, 0) -> (0, -1) -> (-1, 0); toplam enerji 1, 0.5, 1, 0.5 ...
    ve max KE 0.5 -> S = (1 - 0.5) / 0.5 = 1.0 TAM (bütün ara sayılar ikili
    tabanda kesin). save_every 2'de kayıtlar yalnız dönüm noktalarını görür
    (KE = 0, toplam 1): kareden okunan S 0/0'dır. Her adım uçları ikisinde
    de AYNI. Integratör katmanında sınanır: `energy_oscillation` yalnız TB'de
    hesaplar.
    """
    alan = Harmonic(k=2.0, center=np.zeros((1, 3)))
    durum = MDState(
        positions=np.array([[1.0, 0.0, 0.0]]), velocities=np.zeros((1, 3)), masses=np.array([1.0])
    )
    for se in (1, 2):
        y = run(durum, alan, dt=1.0, n_steps=8, save_every=se)
        assert (y.total_min, y.total_max, y.kinetic_max) == (0.5, 1.0, 0.5), se
        assert (y.total_max - y.total_min) / y.kinetic_max == 1.0
    assert y.kinetic.max() == 0.0 and np.ptp(y.kinetic + y.potential) == 0.0
    # ADIM 0 DAHİL: tek adımlık koşuda toplam 1 yalnız adım 0'da, 0.5 adım
    # 1'de -- adım 0'ı dışarıda bırakan mutant (0.5, 0.5, 0.5), S = 0 verir.
    # (Mutasyon turunda bu mutantı yalnız kütle safsızlığı testi öldürüyordu;
    # o test beyanlı bir körlüğü kilitler ve düzeltilince değişecektir.)
    y1 = run(durum, alan, dt=1.0, n_steps=1)
    assert (y1.total_min, y1.total_max, y1.kinetic_max) == (0.5, 1.0, 0.5)
    # KE'nin tepesi YALNIZ adım 0'da: x0 = 0, v0 = 1 -> (x, v) = (1, 0) tek
    # adımda; KE 0.5 -> 0. `k_max`ı adım 0'sız başlatan mutant 0 verir
    # (doğrulama N01: önceden yalnız kütle safsızlığı testi öldürüyordu).
    hizli = MDState(
        positions=np.zeros((1, 3)), velocities=np.array([[1.0, 0.0, 0.0]]), masses=np.array([1.0])
    )
    y2 = run(hizli, alan, dt=1.0, n_steps=1)
    assert (y2.total_min, y2.total_max, y2.kinetic_max) == (0.5, 1.0, 0.5)
    assert y2.kinetic.tolist() == [0.5, 0.0]


def test_T8b_ARADA_nan_ENERJI_UCLARA_YAYILIR_sessizce_DUSMEZ():
    """Bir ARA adımda enerji `nan` (kuvvet sonlu): uçlar `nan` olmalı.

    Python'un `min`/`max`'ı ve `<` karşılaştırması nan'ı SIRAYA BAĞLI olarak
    düşürür -- ölçülmemiş bir adım "salınım küçük" diye geçerdi. Uçlar
    `np.minimum`/`np.maximum` ile tutulur. KIRMA: builtin `min`/`max`'a dönen
    mutant burada sonlu uç üretir.
    """
    yay = Harmonic(k=2.0, center=np.zeros((1, 3)))
    cagri = {"n": 0}

    def alan(R):
        enerji, kuvvet = yay(R)
        cagri["n"] += 1
        return (float("nan") if cagri["n"] == 4 else enerji), kuvvet

    durum = MDState(
        positions=np.array([[1.0, 0.0, 0.0]]), velocities=np.zeros((1, 3)), masses=np.array([1.0])
    )
    y = run(durum, alan, dt=1.0, n_steps=8)
    assert np.isfinite(y.kinetic_max)
    assert np.isnan(y.total_min) and np.isnan(y.total_max), (y.total_min, y.total_max)


# ----------------------------------------------- dürüst olumsuz: kör aile --


#: Şüphecinin kamu yolu YAML'ı (`d2-olcum/A_supheci/s8_kutle_safsizligi.yaml`)
#: AYNEN: Peierls minimumu, 15 x 12 amu + tek H (indis 7), x-hızları, dt
#: 52.35 x 57 adım. Sayılar YAML'daki `repr`lerdir.
_KS_X = np.array([
    0.06710839136830482, 1.9328916086316943, 4.067108391368305,
    5.932891608631695, 8.067108391368304, 9.932891608631696,
    12.067108391368304, 13.932891608631696, 16.067108391368304,
    17.932891608631696, 20.067108391368304, 21.932891608631696,
    24.067108391368304, 25.932891608631696, 28.067108391368304,
    29.932891608631696,
])
_KS_V = np.array([
    7.20203011353725e-05, 0.00017122654181515927, 6.886361783746799e-05,
    -0.0002715800618255416, 0.00018867761792834095, 9.30251784051583e-05,
    -0.00011190191737419112, 0.0004178914856219186, 7.597747339682488e-05,
    6.129768510609042e-05, 5.923240777524243e-06, 0.00011393586522344705,
    -0.0001534782155437614, -3.3950695031235695e-05, -0.00010047443974486399,
    0.0001248005133279877,
])
_KS_KUTLE = (12.0,) * 7 + (_M_H,) + (12.0,) * 8
_KS_DT, _KS_N = 52.35, 57


def test_DURUST_OLUMSUZ_KUTLE_SAFSIZLIGI_iri_dt_de_SALINIM_GECIRIR(tmp_path, capsys):
    """BEYANLI KÖR AİLE: 15 x 12 amu zincirde tek H; omega_ref*dt = 1.000
    (yol boyu 51 karenin en büyük kütle ağırlıklı Hessian modu; D2 şüphecisi
    s8) -- iri dt, ama S = 3.6286e-02 eşiğin ALTINDA ve panel ✓.

    Bu test bir başarıyı değil BİLİNEN BİR KÖRLÜĞÜ kilitler: düzeltilirse
    (ör. a priori omega*dt tanısı -- envanterde yeni madde) kırmızıya döner ve
    beyan (`_TB_SALINIM_BEYANI`, `energy_oscillation` dokstringi) güncellenmelidir.
    Pencere yukarıdaki `_KS_*` sabitleri (şüphecinin kamu yolu YAML'ı, aynen).
    """
    yol = _yaml(tmp_path / "ks.yaml", _KS_X, _KS_DT, _KS_N, hizlar=_KS_V, kutleler=_KS_KUTLE)
    satir = _panel(yol, tmp_path, capsys)
    yakinsama = satir["Yakınsama"]
    assert "✓" in yakinsama, yakinsama
    assert "energy_oscillation=3.629e-02," in yakinsama
    tani = satir["Yakınsama tanısı"]
    assert "KÖR AİLELER" in tani and "kütle safsızlığı" in tani


def test_DURUST_OLUMSUZ_AYRISAN_PARCALAR_otelemesi_S_yi_SEYRELTIR(tmp_path, capsys):
    """BEYANLI KÖR AİLE, en genel hâli (doğrulama, 09.10.2026): S'nin paydası
    TÜM kinetik enerjidir; titreşime katılmayan KE paydayı büyütür, payı değil.

    Pencere: T1 zinciri (tohum 4) İKİ kopya, y = 0 ve y = 10 (r_cut 3.8'in
    çok ötesinde, etkileşmez), n_e 32, dt 40 x 100. Kontrol: v_y = 0 -> iki
    kopya tek zincirle AYNI sayıları verir (sürüklenme 8.788e-03, S
    2.310e-01, ✗). Ayrışan: zincirler -/+ v_y (öteleme KE'si = 10 x T1'in
    max KE'si x 2) -> titreşim aynı, yol omega_max*dt 1.09 (doğrulayıcının 3N
    Hessian ölçümü), ama S 2.100e-02, panel ✓. Kütleler hep M_H, hızlar
    sıfırdan: beyandaki öteki ailelerin hiçbiri değil. D2 öncesi panel de ✓
    basıyordu (sürüklenme 7.989e-04 -- o da aynı biçimde seyrelir).

    Aynı gözlenebilir üzerinde çürütme: iki koşu yalnız v_y'de ayrılır.
    Düzeltilirse (ör. parça başına COM'a göre KE paydası ya da a priori
    omega*dt tanısı -- envanterde yeni madde) ikinci assert kırmızıya döner.
    """
    zA = np.zeros((16, 3))
    zA[:, 0] = _zincir(4)
    zB = zA.copy()
    zB[:, 1] = 10.0
    R = np.vstack([zA, zB])
    # Öteleme hızı: 10 x 2 x (T1 hücresinin max KE'si) = 0.5 x 32 x M_H x v_y^2.
    ke1 = 0.015320024303224319
    vy = float(np.sqrt(10.0 * 2.0 * ke1 / (0.5 * 32.0 * amu_to_electron_mass(np.array([_M_H]))[0])))
    V = np.zeros((32, 3))
    V[:16, 1], V[16:, 1] = -vy, vy
    sonuc = {}
    for ad, hiz in (("kontrol", np.zeros((32, 3))), ("ayrisan", V)):
        yol = _yaml(tmp_path / f"{ad}.yaml", R, 40.0, 100, hizlar=hiz, n_e=32)
        sonuc[ad] = _panel(yol, tmp_path, capsys)["Yakınsama"]
    assert "✗" in sonuc["kontrol"] and "energy_oscillation=2.310e-01," in sonuc["kontrol"]
    assert "✓" in sonuc["ayrisan"] and "energy_oscillation=2.100e-02," in sonuc["ayrisan"]


def test_KAMU_YOLU_nan_S_PANELDE_nan_ve_YARGILANAMAZ(tmp_path, capsys, monkeypatch):
    """`run.py` S'yi olduğu gibi geçirir: `nan` panelde `nan` ve
    "salınım YARGILANAMAZ" olarak basılır, ✗ -- 0.0'a ya da sonlu bir sayıya
    çevrilmez. `energy_oscillation` sarılır: sevk edilen senaryoda gerçek S
    sonludur, `nan`ı yalnız bu sarmalayıcı üretir. KIRMA: `run.py`de `nan`ı
    0.0'a çeviren mutant (doğrulama N11) burada ✓ basar."""
    monkeypatch.setattr(fabrika, "energy_oscillation", lambda senaryo, yorunge: float("nan"))
    sevk = Path(qlab.__file__).resolve().parents[2] / "scenarios" / "tb_zincir_md.yaml"
    satir = _panel(sevk, tmp_path, capsys)
    assert "✗" in satir["Yakınsama"] and "energy_oscillation=nan," in satir["Yakınsama"]
    assert "salınım YARGILANAMAZ" in satir["Yakınsama tanısı"]


# ------------------------------------------------------- T9: metin kilitleri --


def test_T9_TANI_SATIRI_GRI_BOLGE_BUYUME_KOR_AILE_ve_TEK_KAYNAK():
    """"Yakınsama tanısı" satırı: canlı yarı iki ölçütün AYRI hükmü, sabit
    yarı üç beyan (gri bölge, büyüme evresi, kör aileler). Sabit yarı
    `src/` ağacında TEK kez (`md/factory.py`); tanı klasik dalda ve NVT'de
    basılmaz. Sürüklenme ✗ / S ✓ hücresinde tanı yalnız sürüklenmeyi suçlar.
    """
    beyan = fabrika._TB_SALINIM_BEYANI
    for parca in (
        "GRİ BÖLGE", "her pencerenin kendi yol boyu omega_max'ıyla",
        "tohum, aile, başlangıç ve koşu uzunluğu", "alt kenar aileye bağlıdır",
        "S = x^2/(4 - x^2), eşik x = 0.436",
        "BÜYÜME EVRESİ", "'korunum doğrulanamadı' diye okunur, 'dt iri' diye değil",
        "YUVARLAMA TABANI", "dt'yi küçültmek çare değildir",
        "KÖR AİLELER", "payda TÜM kinetik enerjidir", "ayrışan parçaların ötelemesi",
        "kütle safsızlığı", "yalnız düşük modun uyarıldığı koşu",
        "yüksek T_ion", "enine yumuşak modlu geometri", "liste kapsayıcı değildir",
        "kanıtlamaz",
    ):
        assert parca in beyan, parca
    # Tek mod analitiği: eşiğin karşılığı x = 0.436 (beyandaki sayı bundan).
    x = 0.436
    assert abs(x * x / (4.0 - x * x) - fabrika._TB_OSCILLATION_TOLERANCE) < 5e-4
    kaynak = Path(qlab.__file__).resolve().parent
    sayim = sum(p.read_text(encoding="utf-8").count(beyan[:60]) for p in kaynak.rglob("*.py"))
    assert sayim == 1, sayim
    tb = _tb_senaryo()
    iyi = provenance_for(tb, energy_drift=1e-9, energy_oscillation=1e-9).convergence_diagnosis
    assert iyi.startswith("sürüklenme ve salınım ikisi de eşiğin altında") and iyi.endswith(beyan)
    kayan = provenance_for(tb, energy_drift=1.0, energy_oscillation=1e-9)
    assert kayan.converged is False
    assert "sürüklenme eşiğin üstünde" in kayan.convergence_diagnosis
    assert "salınım" not in kayan.convergence_diagnosis.split(" GRİ BÖLGE")[0]
    # İKİSİ BİRDEN ✗ (ya da ikisi birden nan): canlı yarı ikisini de söyler --
    # biri öbürünü yutmaz (doğrulama N04: `if` -> `elif` mutantı yaşıyordu).
    for d, s, salinim_parcasi in (
        (1.0, 1.0, "salınım eşiğin üstünde"),
        (float("nan"), float("nan"), "salınım YARGILANAMAZ"),
        (1.0, float("inf"), "salınım YARGILANAMAZ"),
    ):
        canli = provenance_for(tb, energy_drift=d, energy_oscillation=s).convergence_diagnosis
        canli = canli.split(" GRİ BÖLGE")[0]
        assert salinim_parcasi in canli and "sürüklenme eşiğin üstünde" in canli, (d, s, canli)
    morse = dataclasses.replace(tb, force_field={"kind": "morse", "params": {}})
    assert provenance_for(morse, energy_drift=1e-9).convergence_diagnosis is None
