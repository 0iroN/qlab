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

T1/T1b, T4, kapanış ve dürüst olumsuz testleri ve `nan` kilidi KAMU
YOLUNDAN (YAML -> `qlab.run.main`) koşar; T3 kamu yükleyicisi + `build` +
integratörle, T5-T11 birim/kapı testleridir. Sayılar eleştirmenin kamu yolu
logundan (`d2-olcum/elestirmen/e1_kamu_yolu.log`), doğrulama koşusundan
(wf_6b0da0f9-95f, `d2-olcum/dogrulama/`) ve bu oturumda yeniden üretildi.
T2 (sevk edilen senaryo) `test_tb_kuvvet.py`nin CLI testindedir.

D2-i (10.10.2026): S'nin paydası TÜM kinetik enerjidir ve titreşime
katılmayan KE onu seyreltir (iki dürüst olumsuz bunu kilitliyordu). Ön ölçüm
(`.superpowers/sdd/d2-olcum/D2i/`) iki adayı ölçtü; kod a priori tanıyı aldı:
kare-0 geometrisinin 3N kütle ağırlıklı Hessian'ından omega0, Yakınsama =
sürüklenme VE S VE omega0*dt <= 0.54. Eski iki dürüst olumsuz KAPANIŞ
testine döndü (kontrolleri küçük dt'de ✓); kalan körlük (kare 0 yolun en
hızlı modunu görmez: çarpışan parçalar, yüksek T_ion) iki yeni dürüst
olumsuzla kilitli.

SÜRE (ölçüldü, 10.10.2026, bu makine): bkz. commit; hiçbiri `slow` değildir.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import qlab
import qlab.engine.md.factory as fabrika
from qlab.engine.md.factory import build, energy_oscillation, omega_max_frame0, provenance_for
from qlab.engine.md.forcefields import Harmonic
from qlab.engine.md.integrator import Trajectory, run
from qlab.engine.md.state import MDState
from qlab.engine.md.tb_kuvvet import SsSigmaTBKuvveti
from qlab.engine.tb.radyal import UsselRadyalYasa
from qlab.errors import InvalidInputError
from qlab.run import main
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import MDScenario
from qlab.units import KB_HARTREE_PER_KELVIN, amu_to_electron_mass

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
    kutu: float = 32.0,
) -> Path:
    """Ortak pencere (ss_sigma -0.1/2/2, itme 0.05/2/6, r_on 3.0, r_cut 3.8,
    631.6 K, L `kutu`, varsayılan 32). `x`/`hizlar` (N,) ise yalnız x
    bileşeni, (N, 3) ise tam. Sayılar `repr` ile: YAML'dan bit bit geri döner."""

    def uc(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        return np.c_[a, np.zeros((len(a), 2))] if a.ndim == 1 else a

    satirlar = [
        "kind: md", "name: d2", "force_field:", "  kind: tb_ss_sigma", "  params:",
        "    ss_sigma: {genlik_hartree: -0.1, d0_bohr: 2.0, us: 2.0, r_on_bohr: 3.0, r_cut_bohr: 3.8}",
        "    itme: {genlik_hartree: 0.05, d0_bohr: 2.0, us: 6.0, r_on_bohr: 3.0, r_cut_bohr: 3.8}",
        f"    n_electron: {n_e}", "    electron_temperature_kelvin: 631.6", f"    box_x_bohr: {kutu!r}",
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
    t = provenance_for(tb, energy_drift=1e-9, energy_oscillation=1e-9, omega_max_frame0=1e-9).diagnostics
    esik = t["energy_oscillation_tolerance"]
    assert esik == 5.0e-02
    assert t["energy_oscillation_scale"] == "max_kinetic_all_steps"
    # D2-i: a priori tanı geçer durumda sabit (omega0 1e-9) -- hükmü yalnız S belirler.
    w = dict(energy_drift=1e-9, omega_max_frame0=1e-9)
    assert provenance_for(tb, energy_oscillation=float(np.nextafter(esik, 0.0)), **w).converged
    assert not provenance_for(tb, energy_oscillation=float(np.nextafter(esik, 1.0)), **w).converged
    # Eşiğin KENDİSİ ✓ ('<='; sürüklenme dalıyla aynı anlam -- doğrulama N03).
    assert provenance_for(tb, energy_oscillation=esik, **w).converged
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
    # D2-i: a priori tanının üç anahtarı da NVT'de None (0.0 DEĞİL).
    for anahtar in ("omega_max_frame0", "omega_dt_frame0", "omega_dt_frame0_tolerance"):
        assert anahtar in prov.diagnostics and prov.diagnostics[anahtar] is None, anahtar
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
        prov = provenance_for(tb, energy_drift=1e-9, energy_oscillation=s, omega_max_frame0=1e-9)
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


def _a_priori_suclar(satir: dict[str, str]) -> bool:
    """Tanının CANLI yarısı ✗'i yalnız a priori tanıya mı bağlıyor (S ve
    sürüklenme geçerken)? Sabit beyan (" GRİ BÖLGE"den sonrası) ayrılır."""
    canli = satir["Yakınsama tanısı"].split(" GRİ BÖLGE")[0]
    return (
        "kare-0 omega0*dt eşiğin üstünde" in canli
        and "salınım eşiğin üstünde" not in canli
        and "sürüklenme eşiğin üstünde" not in canli
    )


def test_D2i_KAPANDI_KUTLE_SAFSIZLIGI_S_SEYRELIR_A_PRIORI_YAKALAR(tmp_path, capsys):
    """D2'nin dürüst olumsuzu KAPANDI (D2-i): 15 x 12 amu zincirde tek H;
    omega_ref*dt = 1.000 (yol boyu 51 karenin en büyük kütle ağırlıklı
    Hessian modu; D2 şüphecisi s8). S = 3.629e-02 hâlâ eşiğin ALTINDA (S
    düzeltilmedi, payda tüm KE) ve sürüklenme 1.559e-03 de; D2'nin paneli ✓
    basıyordu. Şimdi ✗: kare-0 omega0*dt = 9.266e-01 > 0.54 (H'nin yerel
    modu kare 0'da da hızlı).

    Aynı gözlenebilir üzerinde çürütme: AYNI pencere dt yarıya (26.175 x 114
    adım, aynı süre) -> ✓ (S 8.047e-03, omega0*dt 4.633e-01). ✗'i dt
    vermiştir, pencere değil. Pencere `_KS_*` (şüphecinin kamu yolu YAML'ı).
    KIRMA: a priori tanı yok sayılır ya da eşik >= 0.927 -> ✓, kırmızı.
    """
    sonuc = {}
    for ad, dt, n in (("iri", _KS_DT, _KS_N), ("kontrol", _KS_DT / 2.0, 2 * _KS_N)):
        yol = _yaml(tmp_path / f"{ad}.yaml", _KS_X, dt, n, hizlar=_KS_V, kutleler=_KS_KUTLE)
        sonuc[ad] = _panel(yol, tmp_path, capsys)
    iri = sonuc["iri"]["Yakınsama"]
    assert "✗" in iri and "✓" not in iri, iri
    for parca in ("energy_drift=1.559e-03,", "energy_oscillation=3.629e-02,", "omega_dt_frame0=9.266e-01,",
                  "omega_dt_frame0_tolerance=5.400e-01"):
        assert parca in iri, parca
    assert _a_priori_suclar(sonuc["iri"]), sonuc["iri"]["Yakınsama tanısı"]
    kontrol = sonuc["kontrol"]["Yakınsama"]
    assert "✓" in kontrol and "omega_dt_frame0=4.633e-01," in kontrol, kontrol
    assert "energy_oscillation=8.047e-03," in kontrol


def test_D2i_KAPANDI_AYRISAN_PARCALAR_otelemesi_S_yi_seyreltir_A_PRIORI_YAKALAR(tmp_path, capsys):
    """D2'nin dürüst olumsuzu KAPANDI (D2-i), seyrelmenin en genel hâli: S'nin
    paydası TÜM kinetik enerjidir; titreşime katılmayan KE paydayı büyütür,
    payı değil.

    Pencere: T1 zinciri (tohum 4) İKİ kopya, y = 0 ve y = 10 (r_cut 3.8'in
    çok ötesinde, etkileşmez), n_e 32, dt 40 x 100. Kontrol: v_y = 0 -> iki
    kopya tek zincirle AYNI sayıları verir (sürüklenme 8.788e-03, S
    2.310e-01, ✗ -- S'den). Ayrışan: zincirler -/+ v_y (öteleme KE'si = 10 x
    T1'in max KE'si x 2) -> titreşim aynı, yol omega_max*dt 1.09, S hâlâ
    2.100e-02 (seyreltilmiş; D2'nin paneli ✓ basıyordu). Şimdi ✗: kare-0
    omega0*dt 9.451e-01 (iki koşuda aynı -- KE'ye bakmaz) ve tanı YALNIZ a
    priori tanıyı suçlar.

    Aynı gözlenebilir üzerinde çürütme: ayrışan çift dt 20 x 200'de ✓ (S
    6.077e-03, omega0*dt 4.725e-01).
    KIRMA: a priori tanı yok sayılır -> ayrışan ✓, kırmızı.
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
    for ad, hiz, dt, n in (("kontrol", np.zeros((32, 3)), 40.0, 100), ("ayrisan", V, 40.0, 100),
                           ("ayrisan_ince", V, 20.0, 200)):
        yol = _yaml(tmp_path / f"{ad}.yaml", R, dt, n, hizlar=hiz, n_e=32)
        sonuc[ad] = _panel(yol, tmp_path, capsys)
    kontrol, ayrisan = sonuc["kontrol"]["Yakınsama"], sonuc["ayrisan"]["Yakınsama"]
    assert "✗" in kontrol and "energy_oscillation=2.310e-01," in kontrol
    assert "✗" in ayrisan and "✓" not in ayrisan, ayrisan
    assert "energy_oscillation=2.100e-02," in ayrisan and "omega_dt_frame0=9.451e-01," in ayrisan
    assert "omega_dt_frame0=9.451e-01," in kontrol
    assert _a_priori_suclar(sonuc["ayrisan"]), sonuc["ayrisan"]["Yakınsama tanısı"]
    ince = sonuc["ayrisan_ince"]["Yakınsama"]
    assert "✓" in ince and "omega_dt_frame0=4.725e-01," in ince, ince


# ------------------------------------------------- dürüst olumsuz: KALAN körlük --


#: Şüphecinin s3 penceresi "H Pmin T=6000 t2" (D2-i ön ölçümü i8, kamu yolunda
#: yeniden kuruldu): Peierls minimumu (`_KS_X` ile AYNI konumlar) + 6000 K ısıl
#: x-hızları (tohum 2). Sayılar `repr`.
_KK_V = np.array([
    0.0006079929805280287, -0.0016811515350165151, -0.0013284064665321778,
    -0.00785172429475127, 0.005787833284536401, 0.0036796211321841674,
    -0.0010465552038377924, 0.0024885509508133256, 0.0009043695048696799,
    -0.0017810863461288511, 0.0031438429857150404, -0.0009987454264742954,
    -0.001057492987432516, -0.002547532671656966, 0.0014631386708072785,
    -0.00031901952006388977,
])


def test_DURUST_OLUMSUZ_KALAN_YUKSEK_T_ION_kare0_yolu_temsil_etmez(tmp_path, capsys):
    """BEYANLI KALAN KÖRLÜK (D2-i): kare 0 yolun en hızlı modunu temsil etmez.

    Peierls minimumu + 6000 K ısıl hız, dt 23.02 x 130: yol boyu
    omega_ref*dt 0.930 (şüpheci s3, 51 kare) -- iri dt; ama kare-0 omega0*dt
    5.328e-01 eşiğin (0.54) ALTINDA, S 4.511e-02 de (anharmonik keşif yol
    boyu omega'yı kare 0'ın 1.75 katına çıkarır; S'yi seyrelten düşük modlar).
    Panel ✓ -- dt'yi KANITLAMAZ.

    Bu test bir başarıyı değil BİLİNEN BİR KÖRLÜĞÜ kilitler: eşik bu hücreye
    göre 0.53'e çekilirse ya da tanı yol boyu omega'ya genişletilirse kırmızıya
    döner ve beyan (`_TB_SALINIM_BEYANI`, KALAN KÖRLÜK) güncellenmelidir.
    """
    yol = _yaml(tmp_path / "kk.yaml", _KS_X, 23.02, 130, hizlar=_KK_V)
    satir = _panel(yol, tmp_path, capsys)
    yakinsama = satir["Yakınsama"]
    assert "✓" in yakinsama, yakinsama
    for parca in ("energy_drift=3.046e-03,", "energy_oscillation=4.511e-02,", "omega_dt_frame0=5.328e-01,"):
        assert parca in yakinsama, parca
    assert "KALAN KÖRLÜK" in satir["Yakınsama tanısı"] and "yüksek T_ion" in satir["Yakınsama tanısı"]


def test_DURUST_OLUMSUZ_KALAN_CARPISAN_PARCALAR_kare0_yalniz_parca_ici_modlari_gorur(tmp_path, capsys):
    """BEYANLI KALAN KÖRLÜK (D2-i eleştirmeni): kare 0'da 10 bohr ayrı iki 8'li
    zincir (x = 0..14 ve 24..38, L 64) +/-0.004 hızla birbirine koşar; dt 21.61
    x 139. Yol boyu omega_ref*dt 0.930 (eleştirmen e3, dt 2.5 referansı) --
    iri dt; kare 0 yalnız parça içi modları görür: omega0*dt 4.371e-01, S
    3.961e-02, sürüklenme 3.007e-03 -- panel ✓, dt'yi KANITLAMAZ.

    Aynı aile dt yarıya (10.805 x 278): S 1.111e-02 (1/3.6). Kilit bilinen
    körlüğü tutar: tanı yol boyu omega'ya genişletilirse kırmızıya döner.
    """
    x = np.r_[2.0 * np.arange(8), 24.0 + 2.0 * np.arange(8)]
    v = np.r_[np.full(8, 0.004), np.full(8, -0.004)]
    satir = _panel(_yaml(tmp_path / "carpisma.yaml", x, 21.61, 139, hizlar=v, kutu=64.0), tmp_path, capsys)
    yakinsama = satir["Yakınsama"]
    assert "✓" in yakinsama, yakinsama
    for parca in ("energy_drift=3.007e-03,", "energy_oscillation=3.961e-02,", "omega_dt_frame0=4.371e-01,"):
        assert parca in yakinsama, parca
    assert "çarpışması" in satir["Yakınsama tanısı"]


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


def test_KAMU_YOLU_nan_OMEGA0_PANELDE_nan_ve_YARGILANAMAZ(tmp_path, capsys, monkeypatch):
    """D2-i: `run.py` omega0'ı olduğu gibi geçirir -- sevk edilen senaryoda
    gerçek omega0*dt 2.555e-01 (✓); sarmalayıcının `nan`ı panelde `nan` ve "a
    priori tanı YARGILANAMAZ", ✗. KIRMA: `run.py`de omega0'ı hiç hesaplamayıp
    0.0 geçen ya da `nan`ı 0.0'a çeviren mutant burada ✓ basar."""
    monkeypatch.setattr(fabrika, "omega_max_frame0", lambda senaryo, durum, alan: float("nan"))
    sevk = Path(qlab.__file__).resolve().parents[2] / "scenarios" / "tb_zincir_md.yaml"
    satir = _panel(sevk, tmp_path, capsys)
    assert "✗" in satir["Yakınsama"] and "omega_dt_frame0=nan," in satir["Yakınsama"]
    assert "a priori tanı YARGILANAMAZ" in satir["Yakınsama tanısı"]
    assert "salınım" not in satir["Yakınsama tanısı"].split(" GRİ BÖLGE")[0]


# ------------------------------------------------------- T9: metin kilitleri --


def test_T9_TANI_SATIRI_GRI_BOLGE_BUYUME_KOR_AILE_ve_TEK_KAYNAK():
    """"Yakınsama tanısı" satırı: canlı yarı üç ölçütün AYRI hükmü, sabit
    yarı beyanlar (gri bölge, büyüme evresi, yuvarlama tabanı, kör aileler,
    a priori tanı, kalan körlük). Sabit yarı `src/` ağacında TEK kez
    (`md/factory.py`); tanı klasik dalda ve NVT'de basılmaz. Sürüklenme ✗ /
    S ✓ hücresinde tanı yalnız sürüklenmeyi suçlar.
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
        "yüksek T_ion", "enine yumuşak modlu geometri",
        # D2-i: seyrelmeyi a priori tanı yakalar; kalanı açıkça beyanlı.
        "A PRIORI TANI", "en hızlı kütle ağırlıklı modu", "24 ölçülmüş iri-dt hücresinin 19'u",
        "sayı eşiğe duyarlı", "KALAN KÖRLÜK", "kare 0 yolun en hızlı modunu temsil etmiyorsa",
        "parçaların sonradan çarpışması", "omega0 = 0", "Liste kapsayıcı değildir.",
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
    w = dict(omega_max_frame0=1e-9)  # a priori tanı geçer; hükmü öteki ikisi verir
    iyi = provenance_for(tb, energy_drift=1e-9, energy_oscillation=1e-9, **w).convergence_diagnosis
    assert iyi.startswith("sürüklenme, salınım ve kare-0 omega0*dt üçü de eşiğin altında")
    assert iyi.endswith(beyan)
    kayan = provenance_for(tb, energy_drift=1.0, energy_oscillation=1e-9, **w)
    assert kayan.converged is False
    assert "sürüklenme eşiğin üstünde" in kayan.convergence_diagnosis
    canli = kayan.convergence_diagnosis.split(" GRİ BÖLGE")[0]
    assert "salınım" not in canli and "omega0" not in canli
    # İKİSİ BİRDEN ✗ (ya da ikisi birden nan): canlı yarı ikisini de söyler --
    # biri öbürünü yutmaz (doğrulama N04: `if` -> `elif` mutantı yaşıyordu).
    for d, s, salinim_parcasi in (
        (1.0, 1.0, "salınım eşiğin üstünde"),
        (float("nan"), float("nan"), "salınım YARGILANAMAZ"),
        (1.0, float("inf"), "salınım YARGILANAMAZ"),
    ):
        canli = provenance_for(tb, energy_drift=d, energy_oscillation=s, **w).convergence_diagnosis
        canli = canli.split(" GRİ BÖLGE")[0]
        assert salinim_parcasi in canli and "sürüklenme eşiğin üstünde" in canli, (d, s, canli)
        assert "omega0" not in canli, canli
    morse = dataclasses.replace(tb, force_field={"kind": "morse", "params": {}})
    assert provenance_for(morse, energy_drift=1e-9).convergence_diagnosis is None


# ------------------------------------------- T10: a priori tanının kapısı --


def test_T10_A_PRIORI_ESIGI_PANELDE_ve_HUKUMDE_AYNI_SAYI_KAPI_ve_TANI():
    """D2-i: omega0*dt eşiği panelde ve hükümde aynı sayı (literal 0.54); eşiğin
    KENDİSİ ✓ ('<=', öteki eşiklerle aynı anlam), bir ulp üstü ✗. Kapı S'ninkiyle
    aynı: TB NVE'de zorunlu, NVT'de ve klasik dalda yasak, negatif hata. `nan`
    (Hessian sonlu değil) ✗ ve "YARGILANAMAZ". Üç ölçüt birden ✗ iken canlı
    yarı üçünü de söyler.

    Eşik dt üzerinden sınanır (omega0 sabit 1.0 -> omega0*dt = dt): a priori
    tanı `omega0 * scenario.dt`yi yargılar, omega0'ı değil.
    KIRMA: tanı yok sayılır -> ulp üstü ✓; '<' -> eşiğin kendisi ✗; eşik 0.6
    ya da 0.5 -> kırmızı; `nan` -> ✓ (`not omega_dt > esik` biçimi) kırmızı.
    """
    tb = _tb_senaryo()
    esik = provenance_for(tb, energy_drift=1e-9, energy_oscillation=1e-9,
                          omega_max_frame0=1e-9).diagnostics["omega_dt_frame0_tolerance"]
    assert esik == 0.54 == fabrika._TB_OMEGA_DT_TOLERANCE

    def hukum(dt, omega0=1.0, s=1e-9, d=1e-9):
        sen = dataclasses.replace(tb, dt=dt)
        return provenance_for(sen, energy_drift=d, energy_oscillation=s, omega_max_frame0=omega0)

    assert hukum(esik).converged
    assert hukum(float(np.nextafter(esik, 0.0))).converged
    ust = hukum(float(np.nextafter(esik, 1.0)))
    assert not ust.converged
    assert ust.diagnostics["omega_dt_frame0"] == float(np.nextafter(esik, 1.0))
    assert ust.diagnostics["omega_max_frame0"] == 1.0
    canli = ust.convergence_diagnosis.split(" GRİ BÖLGE")[0]
    assert "kare-0 omega0*dt eşiğin üstünde" in canli
    assert "salınım" not in canli and "sürüklenme" not in canli
    # nan -> ✗ ve YARGILANAMAZ; inf de sonlu değil.
    for omega0 in (float("nan"), float("inf")):
        p = hukum(0.1, omega0=omega0)
        assert p.converged is False
        assert "a priori tanı YARGILANAMAZ" in p.convergence_diagnosis
    # Üçü birden ✗: canlı yarı üçünü de söyler (biri öbürünü yutmaz).
    canli = hukum(10.0, s=1.0, d=1.0).convergence_diagnosis.split(" GRİ BÖLGE")[0]
    for parca in ("salınım eşiğin üstünde", "kare-0 omega0*dt eşiğin üstünde", "sürüklenme eşiğin üstünde"):
        assert parca in canli, (parca, canli)
    # Kapı: TB NVE'de zorunlu ...
    with pytest.raises(InvalidInputError, match="a priori tanı zorunludur"):
        provenance_for(tb, energy_drift=1e-9, energy_oscillation=1e-9)
    with pytest.raises(InvalidInputError, match="negatif olamaz"):
        provenance_for(tb, energy_drift=1e-9, energy_oscillation=1e-9, omega_max_frame0=-1e-3)
    # ... NVT'de ve klasik dalda yasak.
    with pytest.raises(InvalidInputError, match="omega_max_frame0 yalnız TB NVE"):
        provenance_for(_tb_senaryo(_NVT), energy_drift=5.0, omega_max_frame0=1e-9)
    morse = dataclasses.replace(tb, force_field={"kind": "morse", "params": {}})
    with pytest.raises(InvalidInputError, match="omega_max_frame0 yalnız TB NVE"):
        provenance_for(morse, energy_drift=1e-9, omega_max_frame0=1e-9)
    assert not any(k.startswith("omega_") for k in provenance_for(morse, energy_drift=1e-9).diagnostics)


# ----------------------------------- T11: omega0 bağımsız yoldan, analitik --


_SS = UsselRadyalYasa(-0.1, 2.0, 2.0, 3.0, 3.8)
_IT = UsselRadyalYasa(0.05, 2.0, 6.0, 3.0, 3.8)


def test_T11_OMEGA0_ENERJININ_IKINCI_FARKIYLA_UYUSUR_DONME_ve_KUTLE_OLCEGI():
    """`omega_max_frame0`: kuvvetin merkezi farkından 3N kütle ağırlıklı Hessian.

    BAĞIMSIZ YOL: dimer (n_e 2, 631.6 K, kutusuz), Brent ile kuvvetin sıfır
    olduğu bağ uzunluğu d0; Omega'nın bağ boyunca İKİNCİ FARKI (h 1e-3, kuvvet
    kullanılmaz) k; omega = sqrt(k / (m/2)). ÖLÇÜLDÜ (D2-i, i9): bağıl fark
    -5.3e-07 (enerji farkının O(h^2) kesmesi); tolerans 1e-5.
    DÖNME: bağ x, y ya da z boyunca -> aynı omega0 (yalnız x altuzayını
    alan bir mutant y/z'de 8.5e-07 verirdi). KÜTLE ÖLÇEĞİ: aynı geometri,
    bütün kütleler 4 amu -> omega0 tam sqrt(M_H / 4) katı (kütle ağırlıksız ya
    da M_H'ye bölen mutant oranı 1 verirdi). NVT ve klasik dalda None.
    """
    from scipy.optimize import brentq

    dimer = SsSigmaTBKuvveti(_SS, _IT, 2.0, 631.6 * KB_HARTREE_PER_KELVIN, None)
    d0 = brentq(lambda d: dimer(np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]]))[1][1, 0], 1.0, 3.5, xtol=1e-14)
    E = lambda d: dimer(np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]]))[0]  # noqa: E731
    h = 1e-3
    k = (E(d0 + h) - 2.0 * E(d0) + E(d0 - h)) / (h * h)
    m_h = float(amu_to_electron_mass(np.array([_M_H]))[0])
    beklenen = float(np.sqrt(k / (0.5 * m_h)))
    tb = _tb_senaryo()
    omegalar = []
    for eksen in range(3):
        R = np.zeros((2, 3))
        R[1, eksen] = d0
        durum = MDState(positions=R, velocities=np.zeros((2, 3)), masses=np.full(2, m_h))
        omegalar.append(omega_max_frame0(tb, durum, dimer))
    for w in omegalar:
        assert abs(w / beklenen - 1.0) < 1e-5, (w, beklenen)
    assert max(omegalar) - min(omegalar) < 1e-9 * beklenen, omegalar
    R = np.zeros((2, 3))
    R[1, 0] = d0
    agir = MDState(positions=R, velocities=np.zeros((2, 3)),
                   masses=np.full(2, float(amu_to_electron_mass(np.array([4.0]))[0])))
    oran = omega_max_frame0(tb, agir, dimer) / omegalar[0]
    assert abs(oran / np.sqrt(_M_H / 4.0) - 1.0) < 1e-12, oran
    # EYER: gerilmiş dimer (d 2.8, bükülme noktasının ötesi) -- boyuna eğrilik
    # NEGATİF ve enine katılıktan büyük; omega0 enine moddur, sqrt(E'(d)/d / mu)
    # (enerjinin BİRİNCİ farkı; kuvvet kullanılmaz). ÖLÇÜLDÜ (D2-i): k_boyuna
    # -4.904e-02, k_enine +2.724e-02, omega0 5.445106e-03, bağıl fark ~1e-7.
    # KIRMA: |lambda|'nın en büyüğünü alan mutant boyuna modu (7.3e-03) verirdi.
    d = 2.8
    k_boyuna = (E(d + h) - 2.0 * E(d) + E(d - h)) / (h * h)
    k_enine = (E(d + h) - E(d - h)) / (2.0 * h) / d
    assert k_boyuna < 0.0 and -k_boyuna > k_enine > 0.0, (k_boyuna, k_enine)
    gerili = np.zeros((2, 3))
    gerili[1, 0] = d
    eyer = MDState(positions=gerili, velocities=np.zeros((2, 3)), masses=np.full(2, m_h))
    w_enine = float(np.sqrt(k_enine / (0.5 * m_h)))
    assert abs(omega_max_frame0(tb, eyer, dimer) / w_enine - 1.0) < 1e-5
    durum = MDState(positions=R, velocities=np.zeros((2, 3)), masses=np.full(2, m_h))
    assert omega_max_frame0(_tb_senaryo(_NVT), durum, dimer) is None
    morse = dataclasses.replace(tb, force_field={"kind": "morse", "params": {}})
    assert omega_max_frame0(morse, durum, dimer) is None
    # Hessian sonlu değil -> nan (panel ✗, YARGILANAMAZ; T10).
    bozuk = lambda R: (0.0, np.full(R.shape, np.nan))  # noqa: E731
    assert np.isnan(omega_max_frame0(tb, durum, bozuk))
