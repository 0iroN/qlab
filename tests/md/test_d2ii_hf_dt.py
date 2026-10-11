"""D2-ii (10.10.2026) — HF NVE'de dt doğruluk hükmü.

Envanter D2-ii: HF (`hartree_fock`) MD dalının yakınsama hükmü yalnız
sürüklenmeye bakıyordu ve sürüklenme iri dt'nin Verlet SALINIMINA kördür. Ön
ölçüm (`.superpowers/sdd/d2-olcum/D2ii/`, 334 hücre, 16 aile: H2 STO-3G/6-31G,
dönen H2, LiH, H2O, H2 + H2 çarpışması) bugünkü panelin iri-dt sınıfının
(kendi yol omega_ref'inde omega*dt >= 0.925) 49/105 hücresine ✓ bastığını
ölçtü. Kod: TB'nin dört ölçütlü yapısı (D2, D2-i, D2-vi) HF NVE'ye --
Yakınsama = sürüklenme VE S VE kare-0 omega0*dt VE yolun her adımda izlenen EN
KISA ÇİFT karesinin omega*dt'si. Eşikler HF'nin KENDİ bandından
(`_HF_OSCILLATION_TOLERANCE`, `_HF_OMEGA_DT_TOLERANCE`); TB'nin S eşiği 0.05
x 0.463'te sahte ✗ veriyordu. İzleyicinin sertlik vekili HF'de yok (ss-sigma
yasalarından kuruluyordu). Hessian alanın TAZE kopyasıyla alınır: B10
kararlılık sayaçları yalnız koşunun adımlarını sayar.

K1-K4 KAMU YOLUNDAN (YAML -> `qlab.run.main`) koşar; sayılar ön ölçümün
`p5_kapanis.log` ve `p5b_kontrol.log`undan, bu oturumda yeniden üretildi. Her
kapanış hücresinin yanında aynı pencerenin dt yarıda ✓ kontrolü vardır.
Sevk edilen senaryonun tam koşusu (200 adım) `slow`dur; öteki testlerin her
biri saniyenin altındadır.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import qlab
import qlab.engine.md.factory as fabrika
from qlab.engine.md.factory import (
    YolKaresiIzleyici,
    build,
    energy_oscillation,
    omega_max_frame0,
    omega_max_path_frames,
    path_frame_tracker,
    provenance_for,
)
from qlab.engine.md.forcefields import HartreeFockGradient
from qlab.engine.md.integrator import run
from qlab.engine.md.state import MDState
from qlab.errors import InvalidInputError, SCFConvergenceError
from qlab.run import main
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import MDScenario
from qlab.units import amu_to_electron_mass

_M_H = 1.00782503207
_RE = 1.3459193616  # H2/STO-3G dengesi (ön ölçüm p0, kuvvetten Newton)
_SEVK = 1.3959193602  # scenarios/h2_aimd.yaml başlangıcı (denge + 0.05)
_SEVK_DT = 5.031487


def _yaml(yol: Path, r: float, dt: float, n: int, v_dik: float = 0.0, save_every: int = 1) -> Path:
    """H2/STO-3G, gradyan adımı 1e-5 (sevk senaryosuyla aynı); bağ z boyunca,
    `v_dik` x boyunca zıt işaretli (DÖNEN H2). Sayılar `repr` ile."""
    satirlar = [
        "kind: md", "name: d2ii", "force_field:", "  kind: hartree_fock", "  params:",
        '    symbols: ["H", "H"]', "    basis: sto-3g", "    step: 1.0e-5", "initial:",
        f"  positions_bohr: [[0.0, 0.0, 0.0], [0.0, 0.0, {r!r}]]",
        f"  masses_amu: [{_M_H!r}, {_M_H!r}]",
    ]
    if v_dik:
        satirlar.append(f"  velocities: [[{v_dik!r}, 0.0, 0.0], [{-v_dik!r}, 0.0, 0.0]]")
    satirlar += [f"dt: {dt!r}", f"n_steps: {n}", f"save_every: {save_every}", ""]
    yol.write_text("\n".join(satirlar), encoding="utf-8")
    return yol


def _panel(yol: Path, tmp_path: Path, capsys) -> dict[str, str]:
    """Kamu yolu: `python -m qlab.run <yaml> -o x.npz`; etiket -> satır."""
    assert main([str(yol), "-o", str(tmp_path / "d2ii.npz")]) == 0
    ekran = capsys.readouterr().out
    return {s.split(":", 1)[0].strip(): s for s in ekran.splitlines() if ":" in s}


def _canli(satir: dict[str, str]) -> str:
    """Tanı satırının canlı yarısı: etiketten sonrası, sabit beyandan ("GRİ BÖLGE") önce."""
    return satir["Yakınsama tanısı"].split(":", 1)[1].strip().split(" GRİ BÖLGE")[0]


def _hf_senaryo(thermostat: dict | None = None, dt: float = 10.0) -> MDScenario:
    """Yalnız `provenance_for` için en küçük geçerli HF senaryosu."""
    return MDScenario(
        name="d2ii",
        force_field={"kind": "hartree_fock", "params": {"symbols": ["H", "H"], "basis": "sto-3g"}},
        initial={"positions_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, _SEVK]], "masses_amu": [_M_H, _M_H]},
        thermostat=thermostat,
        dt=dt,
        n_steps=1,
    )


_NVT = {"kind": "langevin", "params": {"temperature": 300.0, "gamma": 1e-3}}
_GECER = dict(energy_oscillation=1e-9, omega_max_frame0=1e-9, omega_max_path_frames=1e-9)

#: Kapanış hücreleri: (r, v_dik, dt, n) -> panelde basılan sayılar (ön ölçüm, 4 hane).
_DORT = ("energy_drift", "energy_oscillation", "omega_dt_frame0", "omega_dt_path_frames")


# ------------------------------------------------- K1: iri dt, sevk geometrisi --


@pytest.mark.parametrize(
    ("dt", "n", "hukum", "sayilar", "scf"),
    [
        (37.111651551205455, 38, "✗", ("1.107e-03", "2.813e-01", "8.557e-01", "1.000e+00"), 39),
        (37.111651551205455 / 2, 60, "✓", ("4.290e-04", "5.902e-02", "4.278e-01", "5.001e-01"), 61),
    ],
    ids=["K1-iri-x1.0", "K1-kontrol-dt-yarida"],
)
def test_K1_IRI_dt_SURUKLENME_GECIRIYORDU_UC_TANI_YAKALIYOR(dt, n, hukum, sayilar, scf, tmp_path, capsys):
    """Sevk geometrisi (gerilmiş +0.05, durgun), dt = 1.0 / omega_ref (ön ölçüm A05
    hücresi, 6 periyot): sürüklenme 1.107e-03 eşiğin ALTINDA ve D2-ii öncesi panel
    ✓ basıyordu; S 0.281, kare-0 0.856 ve yol karesi 1.000 üçü de eşiğin üstünde
    -> ✗. Tanı yalnız bu üçünü suçlar (sürüklenme geçiyor; hüküm VE'dir). Kontrol:
    aynı pencere dt yarıda, 60 adım ✓. (50-57 adımda aynı kontrolün sürüklenmesi
    9.714e-03 ✗ -- blok ölçütünün kısa koşu SINIR'ı, beyanlı; ölçüm p5c.)

    B10 SAYAÇLARI: `kararlilik_merkez_scf` = n + 1 (kare 0 + her adım). Hessian
    alanın kendisiyle alınsaydı her biri 12 merkez SCF daha sayardı (iki Hessian:
    +24) -- panel kararlılığı koşunun olmayan adımlarıyla basardı.
    KIRMA: S / kare-0 / yol karesi yok sayılır -> tanı eksik; Hessian kopyasız ->
    scf 63; HF eşikleri TB'ninkine bağlanır (S 0.05) -> kontrol ✗.
    """
    satir = _panel(_yaml(tmp_path / "k1.yaml", _SEVK, dt, n), tmp_path, capsys)
    yakinsama = satir["Yakınsama"]
    assert hukum in yakinsama and ({"✓", "✗"} - {hukum}).pop() not in yakinsama, yakinsama
    for ad, deger in zip(_DORT, sayilar):
        assert f"{ad}={deger}," in yakinsama, (ad, yakinsama)
    assert f"kararlilik_merkez_scf={scf}," in yakinsama and f"ic_kararsiz=0/{scf}," in yakinsama
    assert "stiffest_frame_step" not in yakinsama  # HF izleyicisinde sertlik vekili yok
    canli = _canli(satir)
    if hukum == "✗":
        for parca in ("salınım eşiğin üstünde", "kare-0 omega0*dt eşiğin üstünde",
                      "yol karesinin omega*dt'si eşiğin üstünde"):
            assert parca in canli, parca
        assert "sürüklenme eşiğin üstünde" not in canli
    else:
        assert canli.startswith("sürüklenme, salınım, kare-0 omega0*dt ve yol karesinin omega*dt'si dördü de")


# ------------------------------------------ K2: dönen H2, S seyrelir (kör nokta) --


@pytest.mark.parametrize(
    ("dt", "n", "hukum", "sayilar"),
    [
        (36.53216, 41, "✗", ("8.953e-04", "6.019e-03", "8.423e-01", "9.251e-01")),
        (18.26608, 28, "✓", ("9.682e-04", "1.489e-03", "4.212e-01", "4.625e-01")),
    ],
    ids=["K2-iri-x0.925", "K2-kontrol-dt-yarida"],
)
def test_K2_DONEN_H2_S_SEYRELIR_OMEGA_DT_YAKALAR(dt, n, hukum, sayilar, tmp_path, capsys):
    """Envanterin "dönen H2 kör noktası": sevk geometrisi + bağa dik zıt hızlar
    (v 0.002 bohr/a.u.t.). Dönme KE'si S'nin paydasını şişirir: iri dt'de (yol
    omega*dt 0.925) S 6.0e-03 -- eşiğin 14 kat ALTINDA; sürüklenme de geçer, D2-ii
    öncesi panel ✓. omega*dt KE'nin dağılımına bakmaz: kare-0 0.842, yol karesi
    0.925 -> ✗; tanı salınımı DEĞİL iki omega*dt tanısını suçlar. Kontrol: dt
    yarıda ✓. KIRMA: omega*dt tanıları yok sayılır -> iri hücre ✓.
    """
    satir = _panel(_yaml(tmp_path / "k2.yaml", _SEVK, dt, n, v_dik=0.002), tmp_path, capsys)
    yakinsama = satir["Yakınsama"]
    assert hukum in yakinsama and ({"✓", "✗"} - {hukum}).pop() not in yakinsama, yakinsama
    for ad, deger in zip(_DORT, sayilar):
        assert f"{ad}={deger}," in yakinsama, (ad, yakinsama)
    if hukum == "✗":
        canli = _canli(satir)
        assert "kare-0 omega0*dt eşiğin üstünde" in canli and "yol karesinin omega*dt'si eşiğin üstünde" in canli
        assert "salınım" not in canli and "sürüklenme eşiğin üstünde" not in canli


# ------------------------------ K3: kare 0 yolu görmez, yalnız yol karesi yakalar --


@pytest.mark.parametrize(
    ("dt", "n", "hukum", "sayilar", "adim", "mesafe"),
    [
        (30.0, 14, "✗", ("9.825e-04", "5.423e-02", "4.660e-01", "9.265e-01"), 13, "1.213e+00"),
        (15.0, 28, "✓", ("2.981e-04", "1.239e-02", "2.330e-01", "4.604e-01"), 9, "1.217e+00"),
    ],
    ids=["K3-iri", "K3-kontrol-dt-yarida"],
)
def test_K3_GERILMIS_DONEN_H2_YALNIZ_YOL_KARESI_YAKALAR(dt, n, hukum, sayilar, adim, mesafe, tmp_path, capsys):
    """Gerilmiş başlangıç (denge + 0.30) + dönme (v 0.003): kare 0 yolun EN
    YUMUŞAK anıdır (omega0*dt 0.466 ✓), S dönmeyle seyrelir (0.054 ✓), sürüklenme
    geçer -- yalnız yolun her adımda izlenen en kısa çift karesi (adım 13, bağ
    1.213 bohr; omega*dt 0.926) ✗ verir. Ön ölçümde gerilmiş başlangıçlı ailelerde
    omega0 yol tepesinin 0.37-0.86'sıydı (A35, A20, G20). Kontrol: dt yarıda ✓.
    KIRMA: yol karesi yok sayılır ya da izleyici yalnız kare 0'ı tutar -> ✓.
    """
    satir = _panel(_yaml(tmp_path / "k3.yaml", _RE + 0.30, dt, n, v_dik=0.003), tmp_path, capsys)
    yakinsama = satir["Yakınsama"]
    assert hukum in yakinsama and ({"✓", "✗"} - {hukum}).pop() not in yakinsama, yakinsama
    for ad, deger in zip(_DORT, sayilar):
        assert f"{ad}={deger}," in yakinsama, (ad, yakinsama)
    assert f"closest_pair_step={adim}," in yakinsama and f"closest_pair_bohr={mesafe}," in yakinsama
    if hukum == "✗":
        canli = _canli(satir)
        assert "yol karesinin omega*dt'si eşiğin üstünde" in canli
        assert "salınım" not in canli and "kare-0" not in canli and "sürüklenme eşiğin" not in canli


# ----------------------------------------- K4: dengede hareketsiz, yuvarlama KE --


def test_K4_DENGEDE_HAREKETSIZ_H2_yuvarlama_KE_HER_CEKIRDEKTE_X(tmp_path, capsys):
    """H2 dengede (r_e), hız 0, sevk dt'si x 10: kuvvet yuvarlama düzeyinde, KE de.
    BU MAKİNEDE (OpenBLAS) blok farkı tam 0.0 çıkar ve D2-ii öncesi panel bunun
    yanında ✓ basıyordu; omega*dt 0.126 bunu GÖREMEZ. S (yuvarlama düzeyindeki max
    KE'ye bölünen tepe-tepe) ✗ verir. "Tam 0" yuvarlama rastlantısıdır (TB'nin T4
    ikizi): kilit taşınabilir yazılır -- panel HER ortamda ✗; sürüklenme eşiğin
    altındaysa ✗'yi salınım vermiştir. Sayıların kendisi kilitlenmez.
    KIRMA: HF'de S yok sayılır -> bu makinede ✓.
    """
    satir = _panel(_yaml(tmp_path / "k4.yaml", _RE, _SEVK_DT, 10), tmp_path, capsys)
    yakinsama = satir["Yakınsama"]
    assert "✗" in yakinsama, yakinsama
    drift = float(yakinsama.split("energy_drift=", 1)[1].split(",", 1)[0])
    if drift <= fabrika._MD_DRIFT_TOLERANCE:
        canli = _canli(satir)
        assert "salınım eşiğin üstünde" in canli or "salınım YARGILANAMAZ" in canli, canli


# --------------------- K5: seçilen kare kare 0'ın kendisi -> tek Hessian (bedel) --


def test_K5_SIKISIK_BASLANGIC_YOL_KARESI_KARE0_TEK_HESSIAN(tmp_path, capsys, monkeypatch):
    """Sıkışık başlangıç (denge - 0.20, durgun): kare 0 yolun en kısa çift anıdır
    (`closest_pair_step=0`), yol karesinin omega*dt'si kare-0'ınkiyle aynı sayı
    ve `_run_md` Hessian'ı TEK kez alır (`omega_frame0=`). Eleştirmen ölçtü: bu
    tekilleştirme yokken aynı karenin Hessian'ı iki kez alınıyordu (sıkışık
    başlangıç, kısa H2O koşusu). Hüküm burada sürüklenmeden ✗ (10 adımlık kısa
    koşu, blok ölçütünün SINIR'ı) -- bu test yalnız bedeli ve eşitliği kilitler.
    TAM EŞİTLİK: kare 0'dan 1e-6 bohr farklı bir seçili kare yeniden KULLANILMAZ.
    KIRMA: tekilleştirme kaldırılır -> 2 Hessian; `array_equal` yerine
    `allclose` -> yakın kare kare-0 değerini alır.
    """
    sayac = {"n": 0}
    gercek = fabrika._omega_max

    def sayan(*a, **k):
        sayac["n"] += 1
        return gercek(*a, **k)

    monkeypatch.setattr(fabrika, "_omega_max", sayan)
    satir = _panel(_yaml(tmp_path / "k5.yaml", _RE - 0.20, 10.0, 10), tmp_path, capsys)
    yakinsama = satir["Yakınsama"]
    assert "closest_pair_step=0," in yakinsama
    assert "omega_dt_frame0=3.445e-01," in yakinsama and "omega_dt_path_frames=3.445e-01," in yakinsama
    assert sayac["n"] == 1
    monkeypatch.setattr(fabrika, "_omega_max", gercek)
    # Tam eşitlik (sentetik ucuz alan, T8b'nin): yalnız konumu bit bit kare 0 olan kare yeniden kullanılır.
    sen4, m4, R0 = _iki_morse_senaryosu()
    durum4 = MDState(positions=R0, velocities=np.zeros_like(R0), masses=m4)
    for kayma, beklenen_123 in ((0.0, True), (1e-6, False)):
        R = R0.copy()
        R[2, 0] += kayma  # 10 bohr'luk koordinatta 1e-7 bağıl: göreli toleranslı bir eşitlik bunu yutardı
        iz = path_frame_tracker(sen4, _iki_morse, m4)
        iz(R)
        w = omega_max_path_frames(sen4, iz, durum4, _iki_morse, omega_frame0=123.0)
        assert (w == 123.0) is beklenen_123, (kayma, w)


# ---------------------- K6: Hessian sondası geometri kapısına çarpar -> YARGILANAMAZ --


def test_K6_KAPI_YAKINI_KOSU_YAPILIR_SONDA_HATASI_YARGILANAMAZ(tmp_path, capsys):
    """Eleştirmenin kenar bulgusu: başlangıç bağı geometri kapısının (21.0 bohr)
    5e-5 içinde (20.99995). Koşu geçer (HEAD de koşar), ama kare-0 Hessian'ının
    +1e-4 sondası 21.00005'te kapıya çarpar: taslağın ilk sürümü CLI'yi koşudan
    ÖNCE hatayla bitiriyordu. Şimdi sonda hatası `nan` -> "a priori tanı
    YARGILANAMAZ" ✗; koşu yapılır (`kararlilik_merkez_scf` = n + 1, .npz
    yazılır). Yol karesi (adım 3, bağ biraz kısalmış) sonludur. Kontrol: 20.9998'de
    sonda kapıya çarpmaz, dördü de ✓ (ön ölçüm p5d).
    KIRMA: `_HFHessianSondasi` hatayı yakalamaz -> main InvalidInputError'la 1 döner.
    """
    satir = _panel(_yaml(tmp_path / "k6.yaml", 20.99995, 5.0, 3), tmp_path, capsys)
    yakinsama = satir["Yakınsama"]
    assert "✗" in yakinsama and "omega_dt_frame0=nan," in yakinsama
    assert "kararlilik_merkez_scf=4," in yakinsama and (tmp_path / "d2ii.npz").exists()
    assert np.isfinite(float(yakinsama.split("omega_dt_path_frames=", 1)[1].split(",", 1)[0]))
    canli = _canli(satir)
    assert "a priori tanı YARGILANAMAZ" in canli and "koşu yine yapıldı" in canli
    assert "yol karesinin" not in canli and "salınım" not in canli
    kontrol = _panel(_yaml(tmp_path / "k6k.yaml", 20.9998, 5.0, 3), tmp_path, capsys)["Yakınsama"]
    assert "✓" in kontrol and "omega_dt_frame0=1.212e-03," in kontrol


# ------------------------------------------------- sevk edilen senaryo (yavaş) --


@pytest.mark.slow
def test_SEVK_h2_aimd_CLI_DORT_TANI_DE_GECER_ve_PAYLAR(tmp_path, capsys):
    """`scenarios/h2_aimd.yaml` (200 adım) kamu yolundan: ✓; S 4.113e-03 (eşiğin
    20.7 katı altında), kare-0 0.116, yol karesi 0.1356 (adım 25, bağ 1.298 bohr;
    eşiğin 4.9 katı altında); B10 sayacı 201 = n + 1. Beklenen hata satırı HF'nin
    kendi metni (ortak metnin "bir hata değildir" cümlesi yok).
    """
    sevk = Path(qlab.__file__).resolve().parents[2] / "scenarios" / "h2_aimd.yaml"
    satir = _panel(sevk, tmp_path, capsys)
    yakinsama = satir["Yakınsama"]
    assert "✓" in yakinsama, yakinsama
    for parca in ("energy_drift=1.465e-04,", "energy_oscillation=4.113e-03,", "omega_dt_frame0=1.160e-01,",
                  "omega_dt_path_frames=1.356e-01,", "closest_pair_step=25,", "kararlilik_merkez_scf=201,"):
        assert parca in yakinsama, parca
    assert "bir hata değildir" not in satir["Beklenen hata"] and "KE + E" in satir["Beklenen hata"]


# ---------------------------------------------- T5: iki eşiğin ulp kilidi (HF) --


def test_T5_HF_ESIKLERI_PANELDE_ve_HUKUMDE_AYNI_SAYI():
    """Panelin `energy_oscillation_tolerance` ve `omega_dt_tolerance`ı hükmün
    GERÇEKTEN kullandığı eşikler; literal 8.5e-02 ve 0.67; eşiğin kendisi ✓
    ('<='), bir ulp üstü ✗ -- her biri tek başına (öteki ikisi geçer). omega*dt
    tanısı dt üzerinden sınanır (omega 1.0 -> omega*dt = dt): kare-0 ve yol
    karesi AYRI ayrı. HF sürüklenme eşiği değişmedi (5e-3, C5).
    KIRMA: TB'nin eşiği (0.05) HF'ye bağlanır; '<'; tanıya başka sayı basılır.
    """
    hf = _hf_senaryo()
    t = provenance_for(hf, energy_drift=1e-9, **_GECER).diagnostics
    s_esik, w_esik = t["energy_oscillation_tolerance"], t["omega_dt_tolerance"]
    assert s_esik == 8.5e-02 == fabrika._HF_OSCILLATION_TOLERANCE
    assert w_esik == 0.67 == fabrika._HF_OMEGA_DT_TOLERANCE
    assert t["energy_oscillation_scale"] == "max_kinetic_all_steps"
    assert t["energy_drift_tolerance"] == fabrika._MD_DRIFT_TOLERANCE

    def hukum(**degisen):
        ortak = dict(energy_drift=1e-9, **_GECER)
        sen = dataclasses.replace(hf, dt=degisen.pop("dt", 10.0))
        return provenance_for(sen, **(ortak | degisen)).converged

    assert hukum(energy_oscillation=s_esik)
    assert hukum(energy_oscillation=float(np.nextafter(s_esik, 0.0)))
    assert not hukum(energy_oscillation=float(np.nextafter(s_esik, 1.0)))
    for anahtar in ("omega_max_frame0", "omega_max_path_frames"):
        assert hukum(dt=w_esik, **{anahtar: 1.0}), anahtar
        assert not hukum(dt=float(np.nextafter(w_esik, 1.0)), **{anahtar: 1.0}), anahtar


# ------------------------------------------------------- T6: kapı ve NVT (HF) --


def test_T6_HF_KAPI_NVE_ZORUNLU_NVT_YASAK_ve_YOL_ANAHTARLARI():
    """HF NVE'de S, kare-0 ve yol karesi tanıları ZORUNLU (verilmezse "yok say"
    yeni bir sahte ✓ kapısı olurdu); HF NVT'de verilirse HATA, anahtarlar None
    (0.0 değil), hüküm ✗ ve tanı satırı YARGILANMADI ile başlar (NVT'nin
    sorusu, ⟨T⟩ hedefe oturdu mu, bu koşuda hesaplanmaz -- C1/D3/D2-iv,
    11.10.2026; eskiden ✓ ve tanı satırı yoktu). Tanı HF NVE'nin salınım/
    omega*dt tanısı DEĞİLDİR. Negatif değer hata. HF'nin
    `path_frames`ı en kısa çift ve tür başına en kısa an anahtarlarını taşır:
    `stiffest_frame_step` HF'de BİLİNMEYEN anahtardır (sertlik vekili TB
    yasalarından).
    """
    hf = _hf_senaryo()
    for eksik, mesaj in (("energy_oscillation", "salınım ölçütü zorunludur"),
                         ("omega_max_frame0", "a priori tanı zorunludur"),
                         ("omega_max_path_frames", "yol karelerinin tanısı zorunludur")):
        with pytest.raises(InvalidInputError, match=mesaj):
            provenance_for(hf, energy_drift=1e-9, **{k: v for k, v in _GECER.items() if k != eksik})
    for anahtar in _GECER:
        with pytest.raises(InvalidInputError, match="negatif olamaz"):
            provenance_for(hf, energy_drift=1e-9, **(_GECER | {anahtar: -1e-3}))
    yk = {"closest_pair_step": 25, "closest_pair_bohr": 1.298, "pair_type_steps": "H-H@25"}
    t = provenance_for(hf, energy_drift=1e-9, path_frames=yk, **_GECER).diagnostics
    assert [t[k] for k in yk] == [25, 1.298, "H-H@25"] and "stiffest_frame_step" not in t
    assert list(t)[list(t).index("omega_dt_path_frames") + 1:][:4] == [*yk, "omega_dt_tolerance"]
    with pytest.raises(InvalidInputError, match="path_frames yalnız"):
        provenance_for(hf, energy_drift=1e-9, path_frames=yk | {"stiffest_frame_step": 3}, **_GECER)
    nvt = _hf_senaryo(_NVT)
    for anahtar in _GECER:
        with pytest.raises(InvalidInputError, match="yalnız TB NVE ve HF NVE"):
            provenance_for(nvt, energy_drift=5.0, **{anahtar: 1e-9})
    prov = provenance_for(nvt, energy_drift=5.0)
    assert prov.converged is False and prov.convergence_diagnosis.startswith("YARGILANMADI:")
    assert "salınım" not in prov.convergence_diagnosis
    for anahtar in ("energy_oscillation", "energy_oscillation_scale", "energy_oscillation_tolerance",
                    "omega_max_frame0", "omega_dt_frame0", "omega_max_path_frames", "omega_dt_path_frames",
                    "closest_pair_step", "closest_pair_bohr", "pair_type_steps", "omega_dt_tolerance"):
        assert anahtar in prov.diagnostics and prov.diagnostics[anahtar] is None, anahtar
    durum = MDState(positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, _SEVK]]), velocities=np.zeros((2, 3)),
                    masses=amu_to_electron_mass(np.array([_M_H, _M_H])))
    alan = HartreeFockGradient(symbols=("H", "H"), basis="sto-3g", step=1e-5)
    assert omega_max_frame0(nvt, durum, alan) is None
    assert path_frame_tracker(nvt, alan, durum.masses) is None
    assert omega_max_path_frames(nvt, None, durum, alan) is None
    with pytest.raises(InvalidInputError, match="yol karesi izleyicisi zorunludur"):
        omega_max_path_frames(hf, None, durum, alan)


# ------------------------------------- T7: Hessian TAZE kopyayla (B10 sayaçları) --


def test_T7_HF_HESSIAN_TAZE_KOPYA_KARARLILIK_SAYACLARINI_KIRLETMEZ(monkeypatch):
    """Koşunun alanı her merkez SCF'i B10 sayaçlarına işler (`kararlilik_cagri`,
    `max_bag_bohr`); Hessian'ın 6N sondası (H2'de 12 kuvvet çağrısı) bunları
    kirletmemeli. Kare-0 ve yol karesi Hessian'larından sonra sayaçlar koşunun
    kendi değerinde; kopya aynı parametreli (baz, adım, kapı) ve sayaçları sıfır.
    Sonuç kopyasız Hessian'la AYNI sayı (yüzey değişmedi).
    KIRMA: `_hessian_alani` alanın kendisini döner -> sayaç 1 + 12.
    """
    hf = _hf_senaryo()
    alan = HartreeFockGradient(symbols=("H", "H"), basis="sto-3g", step=1e-5)
    R = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, _SEVK]])
    m = amu_to_electron_mass(np.array([_M_H, _M_H]))
    durum = MDState(positions=R, velocities=np.zeros((2, 3)), masses=m)
    alan(R)  # koşunun kare 0'ı
    once = (alan.kararlilik_cagri, alan.max_bag_bohr, alan.en_kotu_ic, alan.en_kotu_dis)
    w0 = omega_max_frame0(hf, durum, alan)
    iz = path_frame_tracker(hf, alan, m)
    iz(R)
    wy = omega_max_path_frames(hf, iz, durum, alan)
    assert (alan.kararlilik_cagri, alan.max_bag_bohr, alan.en_kotu_ic, alan.en_kotu_dis) == (
        once[0] + 1, once[1], once[2], once[3])  # +1: yalnız izleyicinin kendi çağrısı
    sonda = fabrika._hessian_alani(alan)
    kopya = sonda.alan
    assert kopya is not alan and kopya.kararlilik_cagri == 0
    assert (kopya.symbols, kopya.basis, kopya.step, kopya.max_bond_bohr) == (
        alan.symbols, alan.basis, alan.step, alan.max_bond_bohr)
    assert w0 == wy == fabrika._omega_max(R, m, alan, 1e-4)
    # Sonda hatası (geometri kapısı) -> nan, yükselmez; alanın kendisi yükselir.
    uzak = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 21.5]])
    enerji, kuvvet = sonda(uzak)
    assert np.isnan(enerji) and np.isnan(kuvvet).all() and kuvvet.shape == uzak.shape
    with pytest.raises(InvalidInputError, match="geometri kapısı"):
        alan(uzak)
    baska = lambda X: (0.0, np.zeros_like(X))  # noqa: E731
    assert fabrika._hessian_alani(baska) is baska  # TB ve öteki alanlar: alanın kendisi (davranış değişmedi)
    # SCF yakınsamaması da (QLabError) nan; QLabError olmayan bir hata (programlama
    # hatası) YUTULMAZ -- doğrulama N08 / N47.
    for hata, yutulur in ((SCFConvergenceError(100, [], 1e-1), True), (RuntimeError("hata"), False)):
        def bozuk(self, positions, kararlilik=False, _hata=hata):
            raise _hata

        monkeypatch.setattr(HartreeFockGradient, "_energy", bozuk)
        if yutulur:
            enerji, kuvvet = sonda(R)
            assert np.isnan(enerji) and np.isnan(kuvvet).all()
        else:
            with pytest.raises(RuntimeError, match="hata"):
                sonda(R)


# ------------------------------------------------ T8: HF izleyicisi (D2-ii) --


def test_T8_HF_IZLEYICI_YORUNGEYI_DEGISTIRMEZ_YALNIZ_EN_KISA_CIFT_TEK_HESSIAN(monkeypatch):
    """HF NVE'nin izleyicisi alanın dönüşünü AYNEN geçirir (yörünge bit bit aynı),
    adım başına tam bir kez + kare 0 çağrılır (n + 1), seçtiği kare kayıtlı
    karelerden bağımsız yoldan (her kare) en kısa çift mesafesinin karesi;
    sertlik vekili YOK (`sertlik_adim` None, `stiffest_frame_step` anahtarı
    yok); tek türlü H2'de tür başına en kısa an genel en kısa çiftle AYNI kare
    ("H-H@adım") ve `omega_max_path_frames` TEK Hessian alır.
    KIRMA: HF'ye sertlik vekili (TB yasaları) -> AttributeError; ikinci Hessian.
    """
    hf = _hf_senaryo()
    sen = dataclasses.replace(hf, dt=20.0, n_steps=20)
    durum, alan, _ = build(sen)
    durum = durum.remove_center_of_mass_motion()
    yalin = run(durum, alan, dt=20.0, n_steps=20)
    iz = path_frame_tracker(sen, alan, durum.masses)
    y = run(durum, iz, dt=20.0, n_steps=20)
    for ad in ("positions", "velocities", "kinetic", "potential"):
        assert np.array_equal(getattr(y, ad), getattr(yalin, ad)), ad
    assert iz.cagri == 21 and iz.sonlu and iz.sertlik_adim is None and iz.sertlik_konum is None
    d = np.linalg.norm(yalin.positions[:, 1] - yalin.positions[:, 0], axis=1)
    assert iz.en_kisa_adim == int(np.argmin(d)) and abs(iz.en_kisa - float(d.min())) < 1e-12
    assert 0 < iz.en_kisa_adim < 20
    assert np.array_equal(iz.en_kisa_konum, yalin.positions[iz.en_kisa_adim])
    assert iz.tanilar() == {"closest_pair_step": iz.en_kisa_adim, "closest_pair_bohr": iz.en_kisa,
                            "pair_type_steps": f"H-H@{iz.en_kisa_adim}"}
    sayac = {"n": 0}
    gercek = fabrika._omega_max

    def sayan(*a, **k):
        sayac["n"] += 1
        return gercek(*a, **k)

    monkeypatch.setattr(fabrika, "_omega_max", sayan)
    omega_max_path_frames(sen, iz, durum, alan)
    assert sayac["n"] == 1
    assert isinstance(iz, YolKaresiIzleyici)


# ----------------------- T8b: tür başına en kısa an (eleştirmenin F-H karşı örneği) --


def _iki_morse(R: np.ndarray) -> tuple[float, np.ndarray]:
    """Sentetik, ucuz alan: F0-H1 ve H2-H3 iki ayrı Morse bağı (parçalar 10 bohr
    arada, aralarında etkileşim yok). F-H sıkıştıkça H-H'den SERT olur ama hep
    ondan UZUN kalır -- eleştirmenin HF'de ölçtüğü geometri sınıfı."""
    enerji, kuvvet = 0.0, np.zeros_like(R)
    for i, j, d_e, a, r_e in ((0, 1, 0.22, 1.2, 1.8), (2, 3, 0.174, 1.03, 1.40)):
        fark = R[j] - R[i]
        r = float(np.sqrt(fark @ fark))
        x = np.exp(-a * (r - r_e))
        enerji += d_e * (1.0 - x) ** 2
        turev = 2.0 * d_e * a * x * (1.0 - x)
        kuvvet[i] += turev * fark / r
        kuvvet[j] -= turev * fark / r
    return enerji, kuvvet


def _iki_morse_senaryosu() -> tuple[MDScenario, np.ndarray, np.ndarray]:
    """F + 3 H'lik HF senaryosu (izleyici türleri sembollerden okur), kütleler
    (elektron kütlesi) ve başlangıç: F-H gerilmiş (2.1, durgun), H2 hafif sıkışık."""
    sen = MDScenario(
        name="d2ii", thermostat=None, dt=5.0, n_steps=60,
        force_field={"kind": "hartree_fock", "params": {"symbols": ["F", "H", "H", "H"], "basis": "sto-3g"}},
        initial={"positions_bohr": [[0, 0, 0], [0, 0, 2.1], [10, 0, 0], [10, 0, 1.39]],
                 "masses_amu": [18.998403, _M_H, _M_H, _M_H]},
    )
    m = amu_to_electron_mass(np.array(sen.initial["masses_amu"]))
    return sen, m, np.array(sen.initial["positions_bohr"], dtype=float)


def test_T8b_HF_IZLEYICI_TUR_BASINA_EN_KISA_AN_FARKLI_BAG_TURU():
    """Eleştirmenin kamu yolu karşı örneği (sıcak F-H + dönen soğuk H2, `kamu/
    fh_h2don_iri_tur.log`, CLI 2.6 dk -- bu yüzden burada sentetik alanla): genel
    en kısa çift HEP H-H, yolun en sert anı F-H'nin sıkışması. Yalnız genel en
    kısa çifti izleyen seçici yol omega'sının bir kısmını görürdü (HF'de en çok
    0.675'i; iri dt'de sahte ✓). HF izleyicisi en kısa anı ELEMENT ÇİFTİ TÜRÜ
    başına da izler: F-H türünün karesi bağımsız yoldan (her kare) F-H bağının en
    kısa anı, `omega_max_path_frames` iki karenin büyüğü = F-H karesininki;
    `pair_type_steps` ikisini adıyla basar. Hüküm: bu omega'yla eşiği aşan bir
    dt'de ✗, genel karenin omega'sıyla aynı dt ✓ olurdu. EŞİTLİK: aynı kare iki
    kez -> tür başına da İLK kare kalır (TB seçicileriyle aynı kural).
    KIRMA: tür maskesi kaldırılır / tür seçimi Hessian'a girmez -> küçük omega;
    tür karesi '<=' ile güncellenir -> eşitlikte son kare.
    """
    sen, m, R0 = _iki_morse_senaryosu()
    durum = MDState(positions=R0, velocities=np.zeros((4, 3)), masses=m)
    esit = path_frame_tracker(sen, _iki_morse, m)
    esit(R0)
    esit(R0.copy())
    assert {ad: kare[1] for ad, kare in esit.tur_kareleri.items()} == {"F-H": 0, "H-H": 0}
    yalin = run(durum, _iki_morse, dt=5.0, n_steps=60)
    iz = path_frame_tracker(sen, _iki_morse, m)
    run(durum, iz, dt=5.0, n_steps=60)
    assert set(iz.tur_kareleri) == {"F-H", "H-H"}
    P = yalin.positions
    d_fh = np.linalg.norm(P[:, 1] - P[:, 0], axis=1)
    d_hh = np.linalg.norm(P[:, 3] - P[:, 2], axis=1)
    assert d_fh.min() > d_hh.max()  # F-H hep daha uzun: genel en kısa çift hep H-H
    adim_fh, adim_hh = iz.tur_kareleri["F-H"][1], iz.tur_kareleri["H-H"][1]
    assert adim_fh == int(np.argmin(d_fh)) and adim_hh == int(np.argmin(d_hh)) == iz.en_kisa_adim
    assert np.array_equal(iz.tur_kareleri["F-H"][2], P[adim_fh])
    assert iz.tanilar()["pair_type_steps"] == f"F-H@{adim_fh} H-H@{adim_hh}"
    w_genel = fabrika._omega_max(iz.en_kisa_konum, m, _iki_morse, 1e-4)
    w_fh = fabrika._omega_max(P[adim_fh], m, _iki_morse, 1e-4)
    w_yol = omega_max_path_frames(sen, iz, durum, _iki_morse)
    assert w_yol == max(w_genel, w_fh) == w_fh and w_genel < 0.9 * w_fh, (w_genel, w_fh)
    yogun = max(fabrika._omega_max(P[k], m, _iki_morse, 1e-4) for k in range(0, 61, 2))
    assert w_yol >= 0.999 * yogun, (w_yol, yogun)
    dt = 0.8 / w_fh
    iri = dataclasses.replace(sen, dt=dt)
    w0 = omega_max_frame0(iri, durum, _iki_morse)
    ortak = dict(energy_drift=1e-9, energy_oscillation=1e-9, omega_max_frame0=w0)
    assert w0 * dt < 0.67 and w_genel * dt < 0.67
    assert not provenance_for(iri, omega_max_path_frames=w_yol, **ortak).converged
    assert provenance_for(iri, omega_max_path_frames=w_genel, **ortak).converged


# -------------------------------- T9: omega0 bağımsız yoldan (enerjinin 2. farkı) --


def test_T9_HF_OMEGA0_ENERJININ_IKINCI_FARKIYLA_UYUSUR_ve_DONME():
    """HF'de `omega_max_frame0`: kuvvetin (kendisi enerjinin sayısal farkı) merkezi
    farkından 3N kütle ağırlıklı Hessian. BAĞIMSIZ YOL: dengede ENERJİNİN bağ
    boyunca ikinci farkı (h 1e-3, kuvvet kullanılmaz) k, omega = sqrt(k / mu).
    Ön ölçüm (p0): k 0.5729343, omega 2.497438e-02 (kuvvetten) / 2.497439e-02
    (Hessian'dan) -- README'nin T_HF = 251.585 a.u.'su. Bağ x, y ya da z boyunca:
    aynı omega0 (yalnız z altuzayı mutantı x/y'de ~0 verirdi). Tolerans 1e-5.
    """
    hf = _hf_senaryo()
    alan = HartreeFockGradient(symbols=("H", "H"), basis="sto-3g", step=1e-5)
    E = lambda r: alan(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]]))[0]  # noqa: E731
    h = 1e-3
    k = (E(_RE + h) - 2.0 * E(_RE) + E(_RE - h)) / (h * h)
    m = amu_to_electron_mass(np.array([_M_H, _M_H]))
    beklenen = float(np.sqrt(k / (0.5 * m[0])))
    assert abs(beklenen * 251.585 / (2.0 * np.pi) - 1.0) < 1e-4  # README'nin periyodu
    for eksen in range(3):
        R = np.zeros((2, 3))
        R[1, eksen] = _RE
        w = omega_max_frame0(hf, MDState(positions=R, velocities=np.zeros((2, 3)), masses=m), alan)
        assert abs(w / beklenen - 1.0) < 1e-5, (eksen, w, beklenen)


# -------------------------------------------------- T10: metin kilitleri (HF) --


def test_T10_HF_BEKLENEN_BEYAN_TANI_ve_TEK_KAYNAK():
    """HF NVE "Beklenen hata" HF'nin kendi metni (ortak metnin "salınım bir hata
    değildir" cümlesi HF'de artık yanlış); "Yakınsama tanısı" canlı yarısı dört
    ölçütün AYRI hükmü, sabit yarısı HF beyanı (gri bölge, yuvarlama tabanı, kör
    aileler, omega*dt tanısı, kalan körlük, sürüklenmenin sınırı); beyan `src/`
    ağacında TEK kez. Dördü birden ✗ iken canlı yarı dördünü de söyler; NVT'de
    tanı yok ve NVT'nin metni değişmedi.
    """
    hf = _hf_senaryo()
    iyi = provenance_for(hf, energy_drift=1e-9, **_GECER)
    beklenen = iyi.expected_error
    assert "bir hata değildir" not in beklenen
    for parca in ("KE + E SÜRÜKLENMEZ, SALINIR", "S = x^2/(4 - x^2)", "max(KE)'dir",
                  "ayrı ölçüt yargılar", "uçlar kayıtlardan değil her adımdan",
                  "titreşime katılmayan KE (dönme, parçaların ötelemesi) onu seyreltir",
                  "en kısa çift mesafesi karesinin", "Yakınsama dördünün VE'sidir",
                  "YARGILANAMAZ"):
        assert parca in beklenen, parca
    beyan = fabrika._HF_SALINIM_BEYANI
    for parca in ("GRİ BÖLGE", "kendi yol boyu omega_max'ıyla", "yolun EN SERT anına göre",
                  "EKSEN TEMKİNİ", "0.68-0.93 kat daha küçük", "Rc'yi aşan genlikte 0.38-0.43",
                  "~3e-7 bohr", "YUVARLAMA TABANI",
                  "dengede hareketsiz molekül", "KÖR AİLELER", "dönme ve parçaların ötelemesi",
                  "OMEGA*DT TANISI", "en kısa çift mesafesi karesinin", "105 iri-dt hücresinin",
                  "element çifti türü başına da", "soğuk H2 + sıcak F-H", "en çok 0.675'iydi",
                  "F-H karşı örneğinin hepsi", "KALAN KÖRLÜK", "yalnız seçilen karelerine bakar",
                  "seçici MESAFEYE bakar", "aynı türden birden çok bağın toplu modu",
                  "sertlik vekili HF'de kurulamaz",
                  "kanıtlamaz", "SÜRÜKLENME", "2 periyotluk koşuda küçük dt'de de ✗", "save_every'ye",
                  "Liste kapsayıcı değildir."):
        assert parca in beyan, parca
    kaynak = Path(qlab.__file__).resolve().parent
    assert sum(p.read_text(encoding="utf-8").count(beyan[:60]) for p in kaynak.rglob("*.py")) == 1
    assert iyi.convergence_diagnosis.startswith(
        "sürüklenme, salınım, kare-0 omega0*dt ve yol karesinin omega*dt'si dördü de eşiğin altında: KE + E"
    )
    assert iyi.convergence_diagnosis.endswith(beyan)
    # YALNIZ sürüklenme ✗: tanı "dördü de eşiğin altında" DEMEZ (panel kendisiyle
    # çelişmesin -- doğrulama N17: ✓ dalı sürüklenmeyi unutan mutant yaşıyordu).
    kayan = provenance_for(hf, energy_drift=1.0, **_GECER)
    canli = kayan.convergence_diagnosis.split(" GRİ BÖLGE")[0]
    assert kayan.converged is False and canli.startswith("korunum DOĞRULANAMADI")
    assert "sürüklenme eşiğin üstünde" in canli and "dördü de" not in canli
    assert "salınım" not in canli and "omega" not in canli
    sen = dataclasses.replace(hf, dt=10.0)
    hepsi = provenance_for(sen, energy_drift=1.0, energy_oscillation=1.0, omega_max_frame0=1.0,
                           omega_max_path_frames=1.0)
    canli = hepsi.convergence_diagnosis.split(" GRİ BÖLGE")[0]
    for parca in ("salınım eşiğin üstünde", "kare-0 omega0*dt eşiğin üstünde",
                  "yol karesinin omega*dt'si eşiğin üstünde", "sürüklenme eşiğin üstünde"):
        assert parca in canli, parca
    for kotu, parca in ((dict(energy_oscillation=float("nan")), "salınım YARGILANAMAZ"),
                        (dict(omega_max_frame0=float("nan")), "a priori tanı YARGILANAMAZ"),
                        (dict(omega_max_path_frames=float("inf")), "yol karesinin tanısı YARGILANAMAZ")):
        q = provenance_for(hf, energy_drift=1e-9, **(_GECER | kotu))
        assert q.converged is False and parca in q.convergence_diagnosis, parca
    nvt = provenance_for(_hf_senaryo(_NVT), energy_drift=5.0)
    assert nvt.expected_error.startswith("NVT: toplam enerji KORUNMAZ")


# ---------------------------------------- T11: kamu yolunda nan -> YARGILANAMAZ --


@pytest.mark.parametrize(
    ("islev", "imza", "anahtar", "parca"),
    [
        ("energy_oscillation", lambda senaryo, yorunge: float("nan"), "energy_oscillation",
         "salınım YARGILANAMAZ"),
        ("omega_max_frame0", lambda senaryo, durum, alan: float("nan"), "omega_dt_frame0",
         "a priori tanı YARGILANAMAZ"),
        ("omega_max_path_frames", lambda senaryo, izleyici, durum, alan, **_: float("nan"),
         "omega_dt_path_frames", "yol karesinin tanısı YARGILANAMAZ"),
    ],
    ids=["S", "kare0", "yol"],
)
def test_T11_HF_KAMU_YOLU_nan_PANELDE_nan_ve_YARGILANAMAZ(islev, imza, anahtar, parca, tmp_path, capsys, monkeypatch):
    """`run.py` HF'de de üç tanıyı olduğu gibi geçirir: sarmalayıcının `nan`ı
    panelde `nan` ve "YARGILANAMAZ", ✗ -- 0.0'a çevrilmez. Pencere sevk
    geometrisi, sevk dt'si x 10 (gerçek değerler sonlu ve geçer).
    KIRMA: `run.py`de tanıyı hesaplamayıp 0.0 geçen ya da `nan`ı 0.0'a çeviren mutant ✓.
    """
    monkeypatch.setattr(fabrika, islev, imza)
    satir = _panel(_yaml(tmp_path / "nan.yaml", _SEVK, _SEVK_DT, 10), tmp_path, capsys)
    assert "✗" in satir["Yakınsama"] and f"{anahtar}=nan," in satir["Yakınsama"]
    assert parca in _canli(satir)


def test_T12_HF_S_her_adimdan_save_every_ile_BIT_BIT_AYNI(tmp_path):
    """HF'de de S kayıtlı karelerden değil her adımın uçlarından: K1 iri hücresinin
    ilk 19 adımı save_every 1 ve 19 ile S bit bit aynı (ve eşiğin üstünde);
    kayıtlı karelerden okunan S ise save_every 19'da ayrışır (pozitif kontrol).
    KIRMA: S'yi kayıtlı karelerden kuran bir `energy_oscillation`."""
    senaryo = load_scenario(_yaml(tmp_path / "t12.yaml", _SEVK, 37.111651551205455, 19))
    durum, alan, _ = build(senaryo)
    durum = durum.remove_center_of_mass_motion()
    degerler = {}
    for se in (1, 19):
        yorunge = run(durum, alan, dt=senaryo.dt, n_steps=senaryo.n_steps, save_every=se)
        degerler[se] = energy_oscillation(senaryo, yorunge)
        kare = float(np.ptp(yorunge.kinetic + yorunge.potential) / yorunge.kinetic.max())
    assert len(set(degerler.values())) == 1, degerler
    assert degerler[1] > 2.0 * fabrika._HF_OSCILLATION_TOLERANCE
    assert abs(kare - degerler[1]) > 0.05 * degerler[1], (kare, degerler[1])
