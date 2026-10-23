"""Senaryoları GUI olmadan koşan giriş noktası.

    python -m qlab.run scenarios/tunneling.yaml -o out.npz
    python -m qlab.run scenarios/h2_bond_scan.yaml -o h2_curve.npz

Toplu parametre taramaları ve doğrulama koşuları bu yoldan yapılır; app katmanı
hiçbir zaman gerekmez. Bu, spec §3 İlke 1'in pratik karşılığıdır.

Dokuz senaryo türüne göre dallanır: TDSEScenario split-operator yoluna,
HFScanScenario engine.hf.factory.run_scan'e gider (ikisi `_run_loaded`);
MD, TB, Downfold, Spin, EM, Grav ve Bridge kendi `_run_*` dallarına. Her dal
şeffaflık bloğunu Provenance'tan (provenance.as_lines()) basar.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qlab.engine.hf.factory import ScanResult, run_scan
from qlab.engine.tdse import factory
from qlab.engine.tdse.grid import Grid1D
from qlab.engine.tdse.propagator import evolve_olcumlu
# `_run_md`in surukleme kapisi ("en az 2 kayit gerekir", `InvalidInputError`)
# G5'ten beri `md/factory.py::energy_drift`te firlatilir; Modul 3H'de burada
# import EDILMEMIS oldugu icin yakalanmayan bir `NameError` veriyordu ve
# mesaj kullaniciya HIC ulasmiyordu (`tests/test_run.py` o dali tetikler).
from qlab.errors import QLabError
from qlab.provenance import Provenance, etiket_genisligi
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import (
    BridgeScenario,
    DownfoldScenario,
    EMScenario,
    GravScenario,
    HFScanScenario,
    MDScenario,
    SpinScenario,
    TBScenario,
    TDSEScenario,
)


@dataclass
class RunResult:
    scenario: TDSEScenario
    grid: Grid1D
    potential: np.ndarray
    frames: np.ndarray
    times: np.ndarray
    provenance: Provenance


def run_scenario(path: str | Path) -> RunResult | ScanResult:
    return _run_loaded(load_scenario(path))


def _run_loaded(scenario) -> RunResult | ScanResult:
    if isinstance(scenario, HFScanScenario):
        return run_scan(scenario)

    grid, potential, psi0, propagator = factory.build(scenario)

    frames, times, uniterlik = evolve_olcumlu(
        propagator, psi0, n_steps=scenario.n_steps, save_every=scenario.save_every
    )

    final_norm = float(np.sum(np.abs(frames[-1]) ** 2) * grid.dx)
    provenance = factory.provenance_for(scenario, abs(final_norm - 1.0), uniterlik)
    return RunResult(scenario, grid, potential, frames, times, provenance)


def save(result: RunResult | ScanResult, output_path: str | Path) -> None:
    provenance_json = json.dumps(
        dataclasses.asdict(result.provenance), ensure_ascii=False
    )

    if isinstance(result, ScanResult):
        np.savez_compressed(
            output_path,
            values=result.values,
            energies=result.energies,
            # Nokta bazında kararlılık: tek bir özet bayrağı, eğrinin HANGİ
            # noktasının eyer noktası olduğunu söylemez. Ölçülemeyen
            # noktalarda NaN (bkz. factory.ScanResult).
            lowest_eigenvalues=result.lowest_eigenvalues,
            # Dış (RHF→UHF) kanal aynı gerekçeyle nokta bazında yazılır: bir
            # ayrışma taramasında işaret değiştiren kanal tipik olarak budur.
            external_lowest_eigenvalues=result.external_lowest_eigenvalues,
            provenance_json=provenance_json,
        )
        return

    np.savez_compressed(
        output_path,
        frames=result.frames,
        times=result.times,
        x=result.grid.x,
        potential=result.potential,
        provenance_json=provenance_json,
    )


def reconfigure_stdio_utf8() -> None:
    """Windows ve diğer CP1252 varsayılan konsollarında Türkçe karakterleri
    yazdırabilmek için stdout/stderr'i UTF-8'e yeniden yapılandırır.
    reconfigure() var olmayan ortamlarda (ender), stdout/stderr'i wrapping'e
    geri döneriz. Her iki durumda da, başarısız olursa orijinal akış
    kullanılır ve hiçbir ek hata yükseltilmez.

    Paylaşılan tek kaynak: hem bu modülün `main()`'i hem de `qlab.app.main`
    aynı GUI-öncesi/iş-öncesi noktada bunu çağırır — desen iki kez elle
    kopyalanmaz.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        try:
            if hasattr(sys.stdout, "buffer"):
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="replace"
                )
        except (AttributeError, ValueError, OSError):
            pass

    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        try:
            if hasattr(sys.stderr, "buffer"):
                sys.stderr = io.TextIOWrapper(
                    sys.stderr.buffer, encoding="utf-8", errors="replace"
                )
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    reconfigure_stdio_utf8()

    parser = argparse.ArgumentParser(
        prog="qlab.run", description="Senaryoyu (dokuz türün herhangi biri) GUI olmadan koştur"
    )
    parser.add_argument("scenario", help="YAML senaryo dosyası")
    parser.add_argument("-o", "--output", required=True, help="çıktı .npz dosyası")
    args = parser.parse_args(argv)

    try:
        scenario = load_scenario(args.scenario)
        # MD DALI: `_run_md` kendi panelini basar ve kendi .npz'sini yazar,
        # bu yuzden `save()` yolundan gecmez. Bu dagitim eskiden YOKTU ve
        # `python -m qlab.run scenarios/h2_morse.yaml` yakalanmayan bir
        # `AttributeError: 'MDScenario' object has no attribute 'grid'`
        # veriyordu (main yalniz QLabError yakaliyor).
        if isinstance(scenario, MDScenario):
            return _run_md(scenario, args.output)
        # TB DALI: `_run_md` ile ayni gerekce -- `TBScenario` da `grid`
        # tasimaz, dolayisiyla `_run_loaded` yolundan gecerse ayni
        # yakalanmayan AttributeError'i verirdi.
        if isinstance(scenario, TBScenario):
            return _run_tb(scenario, args.output)
        # INDIRGEME DALI: `TBScenario` ile ayni gerekce -- `DownfoldScenario`
        # da `grid` tasimaz.
        if isinstance(scenario, DownfoldScenario):
            return _run_downfold(scenario, args.output)
        # SPIN DALI: `TBScenario`/`DownfoldScenario` ile ayni gerekce --
        # `SpinScenario` da `grid` tasimaz.
        if isinstance(scenario, SpinScenario):
            return _run_spin(scenario, args.output)
        # EM DALI: `SpinScenario` ile ayni gerekce -- `EMScenario` da `grid`
        # tasimaz ve `_run_loaded` yolundan gecerse yakalanmayan bir
        # `AttributeError` verirdi.
        if isinstance(scenario, EMScenario):
            return _run_em(scenario, args.output)
        # YERÇEKİMİ DALI: `EMScenario` ile aynı gerekçe — `GravScenario` da
        # `grid` taşımaz ve `_run_loaded` yolundan geçerse yakalanmayan bir
        # `AttributeError` verirdi.
        if isinstance(scenario, GravScenario):
            return _run_grav(scenario, args.output)
        # KÖPRÜ DALI: `GravScenario` ile aynı gerekçe — `BridgeScenario` da
        # `grid` taşımaz ve `_run_loaded` yolundan geçerse yakalanmayan bir
        # `AttributeError` verirdi (ölçüldü, Task 5: dal eklenmeden önce
        # tam olarak bu hata çıkıyordu).
        if isinstance(scenario, BridgeScenario):
            return _run_bridge(scenario, args.output)
        result = _run_loaded(scenario)
    except QLabError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    save(result, args.output)
    if isinstance(result, ScanResult):
        print(f"{result.scenario.name}: {len(result.values)} nokta -> {args.output}")
    else:
        print(f"{result.scenario.name}: {len(result.times)} kare -> {args.output}")
    genislik = etiket_genisligi()  # C5: GUI ile ayni kaynak (provenance.etiket_genisligi)
    for label, value in result.provenance.as_lines():
        print(f"  {label:<{genislik}} {value}")
    return 0


def _run_tb(scenario: TBScenario, output: str | None) -> int:
    """TB senaryosunu koşturur ve şeffaflık panelini basar.

    `_run_md` gibi kendi panelini basar ve kendi .npz'sini yazar; `save()`
    yolundan geçmez, çünkü `RunResult` bir zaman serisi varsayar ve bant
    yapısı zaman serisi değildir.
    """
    from qlab.engine.tb.bands import (
        band_structure,
        grid_lipschitz_bound,
        hamiltonian_rounding_scale,
        k_path,
        uniform_k_grid,
    )
    from qlab.engine.tb.factory import build_tb
    from qlab.engine.tb.lattice import reciprocal_vectors
    from qlab.observable.tb_obs import certified_conduction, fermi_level
    from qlab.units import HARTREE_TO_EV

    model, provenance = build_tb(scenario)
    izgara = uniform_k_grid(model.lattice, scenario.k_points)
    enerjiler = band_structure(model, izgara)
    # E13 (04.10.2026): panel ızgaranın kendi hükmünü (`is_metal`) DEĞİL
    # kanıtlı hükmü basar -- grafen her k_points'te "yalıtkan ✓" çıkıyordu.
    hukum = certified_conduction(
        enerjiler,
        scenario.n_electron_per_cell,
        grid_lipschitz_bound(model, scenario.k_points),
        hamiltonian_rounding_scale(model),
    )
    e_fermi = fermi_level(enerjiler, scenario.n_electron_per_cell)

    print(f"{scenario.name}: {model.n_orbital} bant, {len(izgara)} k-noktası")
    print(f"  İletkenlik    : {hukum.describe()}")
    # C5'in kalanı (V12, grup 9 düzeltici, 23.10.2026): bu dal ve `_run_md` de sütunu
    # `etiket_genisligi()`den okur. Eskiden sabit 14'tü: `_run_md`nin sevk panellerinde
    # (tb_zincir_md, h2_aimd, argon_nvt) ':' {14, 16} iki sütundaydı ("Yakınsama tanısı"
    # 16 karakter); bu daldaki zincir_tb paneli 14'te, elle hizalı "İletkenlik" başlığı
    # 16'daydı. Bekleten iki test ayrıştırıcısı (`tests/tb/test_tb_scenario.py::_panel`
    # sabit `[16:]`, `tests/md/test_md_scenario.py` B10 `s[:14]`/`s[16:]`) ilk ": "e çevrildi.
    genislik = etiket_genisligi()
    for etiket, deger in provenance.as_lines():
        print(f"{etiket:<{genislik}}: {deger}")

    if output is not None:
        # Task 6'nın modülü BURADA import edilir -- Task 7 tek başına commit
        # edildiğinde CLI'ın kırık olmaması için (aynı gerekçe `_run_md`'de
        # ölçülmüştü: koşulsuz import `ModuleNotFoundError` veriyordu).
        from qlab.viz.band_frames import build_band_payload

        b = reciprocal_vectors(model.lattice)
        yol = k_path((np.zeros(3), 0.5 * b[0]), n_per_segment=scenario.k_points)
        yuk = build_band_payload(
            model, yol, tick_labels=("G", "X"), fermi_hartree=e_fermi
        )
        np.savez(
            output,
            k_distance=yuk.k_distance,
            energies_ev=yuk.energies_ev,
            orbital_weights=yuk.orbital_weights,
            degenerate=yuk.degenerate,
            tick_distances=yuk.tick_distances,
            tick_labels=np.array(yuk.tick_labels),
            # `fermi_ev` k_points IZGARASININ değeridir (yol değil) ve panelin
            # ✓'si onu SERTİFİKALAMAZ (✓ bant uçlarına bakar): zincirde k = 82'de
            # -5.4302 eV, doğrusu -5.3. Kanıtlı sınır |fermi_ev - E_F| <=
            # iletkenlik_cozunurluk_ev (B1), yalnız n_e·n_k çiftken
            # (bkz. `tb_obs.fermi_level`).
            fermi_ev=np.array(yuk.fermi_ev),
            # `is_metal` yalnız KANITLI metalde True; False "yalıtkan"
            # DEMEK DEĞİLDİR -- tam hüküm `iletkenlik` alanındadır.
            is_metal=np.array(hukum.verdict == "metal"),
            iletkenlik=np.array(hukum.verdict),
            iletkenlik_cozunurluk_ev=np.array(hukum.bound * HARTREE_TO_EV),
        )
    return 0


def _run_downfold(scenario: DownfoldScenario, output: str | None) -> int:
    """UHF indirgemesini koşturur ve şeffaflık panelini basar.

    `_run_tb`/`_run_md` ile aynı kalıp: kendi panelini basar ve kendi
    `.npz`'sini yazar, `save()` yolundan geçmez.

    BU FONKSİYON HİÇBİR BEYAN METNİ YAZMAZ. Yöntem/Yaklaşımlar/İhmal
    edilen/Beklenen hata satırlarının tek kaynağı
    `engine/tb/factory.py::build_downfold`'dur ve buradan yalnızca
    `provenance.as_lines()` basılır. İki kopya metin bu depoda bir kez
    panelin iki farklı şey iddia etmesine yol açtı (bkz.
    `tests/hf/test_hf_scenario.py`); kopya sayısı testle kilitlidir.
    """
    from qlab.engine.tb.factory import build_downfold, mesafe_etiketi

    parametreler, provenance = build_downfold(scenario)

    print(
        f"{scenario.name}: N={scenario.ring_size} halka, "
        f"{len(parametreler)} mesafe"
    )
    for d, p in zip(scenario.distances_bohr, parametreler):
        hoplamalar = "  ".join(
            f"t{m + 1}={t:+.6f}" for m, t in enumerate(p.hoppings)
        )
        print(
            f"  d={mesafe_etiketi(d)} bohr  eps={p.epsilon_bar:+.6f}  "
            f"Delta={p.delta:+.6f}  {hoplamalar}  "
            f"<S^2>={p.s_squared:.6f}  E_UHF={p.energy_uhf:+.6f}"
        )
    genislik = etiket_genisligi()
    for etiket, deger in provenance.as_lines():
        print(f"{etiket:<{genislik}}: {deger}")

    if output is not None:
        np.savez(
            output,
            distances_bohr=np.array(scenario.distances_bohr),
            epsilon_bar=np.array([p.epsilon_bar for p in parametreler]),
            delta=np.array([p.delta for p in parametreler]),
            hoppings=np.array([p.hoppings for p in parametreler]),
            s_squared=np.array([p.s_squared for p in parametreler]),
            energy_uhf=np.array([p.energy_uhf for p in parametreler]),
            staggering_residual=np.array(
                [p.staggering_residual for p in parametreler]
            ),
            # B7dog-3 (11.10.2026): mesafe basina UHF kararlilik hukmu, bool.
            # True YALNIZ kanitli kararlida (`search_stable is True`); eyer
            # (False) da olculmemis (None) da False yazilir. Eskiden bu hukum
            # yalniz `provenance_json`daki arama dizgesinden okunabiliyordu.
            search_stable=np.array(
                [p.search_stable is True for p in parametreler], dtype=bool
            ),
            provenance_json=json.dumps(
                dataclasses.asdict(provenance), ensure_ascii=False
            ),
        )
    return 0


def _run_spin(scenario: SpinScenario, output: str | None) -> int:
    """Spin Monte Carlo taramasını koşturur ve şeffaflık panelini basar.

    `_run_downfold`/`_run_tb`/`_run_md` ile aynı kalıp: kendi panelini basar
    ve kendi `.npz`'sini yazar, `save()` yolundan geçmez.

    BU FONKSİYON HİÇBİR BEYAN METNİ YAZMAZ. Yöntem / Yaklaşımlar / İhmal
    edilen / Beklenen hata / Dairesellik / Örnekleme satırlarının tek kaynağı
    `engine/spin/factory.py`dir ve buradan yalnızca `provenance.as_lines()`
    basılır. İki kopya metin bu depoda bir kez panelin iki farklı şey iddia
    etmesine yol açtı (bkz. `tests/hf/test_hf_scenario.py`); kopya sayısı
    `tests/spin/test_spin_senaryo.py` ile kilitlidir.

    Aşağıdaki tablo BEYAN DEĞİL SAYIDIR (`_run_downfold`ın mesafe tablosuyla
    aynı sınıf) — tek kelimelik istisna `ALT_SINIR_ISARETI`dir ve o da
    yazılmaz, fabrikadan İTHAL EDİLİR: iki yerde elle yazılsaydı biri
    değiştiğinde tablo ile beyan satırı farklı şey söylerdi.
    """
    from qlab.engine.spin.factory import (
        ALT_SINIR_ISARETI,
        build_spin,
        provenance_for,
    )
    from qlab.viz.spin_frames import build_spin_payload

    sonuc = build_spin(scenario)
    yuk = build_spin_payload(sonuc)
    provenance = provenance_for(
        scenario, plateau_reached=yuk.plateau_reached, tau_int=yuk.tau_int
    )

    print(
        f"{scenario.name}: {scenario.model}/{scenario.algorithm}, "
        f"L={scenario.lattice_size}, {len(yuk.temperatures)} sıcaklık"
    )
    for i, T in enumerate(yuk.temperatures):
        isaret = "" if yuk.plateau_reached[i] else f"  <- {ALT_SINIR_ISARETI}"
        isi = (
            "C=ÖLÇÜLMEDİ"
            if yuk.heat_capacity is None
            else f"C={yuk.heat_capacity[i]:.6f}"
        )
        print(
            f"  T={T:.4f}  <|m|>={yuk.magnetisation[i]:.6f} "
            f"+-{yuk.magnetisation_error[i]:.3e}  U={yuk.binder[i]:+.6f}  "
            f"chi={yuk.susceptibility[i]:.6f}  {isi}  "
            f"tau={yuk.tau_int[i]:.2f}{isaret}"
        )
    genislik = etiket_genisligi()
    for etiket, deger in provenance.as_lines():
        print(f"{etiket:<{genislik}}: {deger}")

    if output is not None:
        alanlar = {
            "temperatures": yuk.temperatures,
            "magnetisation": yuk.magnetisation,
            "magnetisation_error": yuk.magnetisation_error,
            "plateau_reached": yuk.plateau_reached,
            "tau_int": yuk.tau_int,
            "binder": yuk.binder,
            "susceptibility": yuk.susceptibility,
            "coupling": np.float64(scenario.coupling),  # kT/J = T/coupling, J*chi
            "provenance_json": json.dumps(
                dataclasses.asdict(provenance), ensure_ascii=False
            ),
        }
        # ISI SIGASI OLCULMEDIYSE ANAHTAR HIC YAZILMAZ. Sifir dizisi yazmak
        # "C = 0 cikti" diye okunurdu; `None` "olculmedi" demektir ve ikisi
        # ayni degerle gosterilemez (Modul 1B'nin kararlilik alani dersi).
        if yuk.heat_capacity is not None:
            alanlar["heat_capacity"] = yuk.heat_capacity
        np.savez(output, **alanlar)
    return 0


def _run_em(scenario: EMScenario, output: str | None) -> int:
    """Maxwell FDTD yarıçap taramasını koşturur ve şeffaflık panelini basar.

    `_run_spin`/`_run_downfold`/`_run_tb`/`_run_md` ile aynı kalıp: kendi
    panelini basar ve kendi `.npz`'sini yazar, `save()` yolundan geçmez.

    BU FONKSİYON HİÇBİR BEYAN METNİ YAZMAZ. Yöntem / Yaklaşımlar / İhmal
    edilen / Beklenen hata / Dairesellik / Çıpa sınıfı / Pencere / Yakınsama
    tanısı satırlarının tek kaynağı `engine/em/factory.py`dir ve buradan
    yalnızca `provenance.as_lines()` basılır. İki kopya metin bu depoda bir
    kez panelin iki farklı şey iddia etmesine yol açtı (bkz.
    `tests/hf/test_hf_scenario.py`); kopya sayısı
    `tests/em/test_em_senaryo.py` ile kilitlidir.

    Aşağıdaki tablo BEYAN DEĞİL SAYIDIR (`_run_downfold`ın mesafe tablosu ve
    `_run_spin`in sıcaklık tablosuyla aynı sınıf). Tek kelimelik istisna
    `HUKUM_VERILMEDI`dir ve o da yazılmaz, fabrikadan İTHAL EDİLİR: iki yerde
    elle yazılsaydı biri değiştiğinde tablo ile beyan satırı farklı şey
    söylerdi.

    ÜÇ DEĞERLİ HÜKÜM TABLODA DA KISALTILMAZ: merdiven oturmadıysa satır
    `settled_order` yerine hükmün kendisini basar. `YARGILANAMAZ`ı sessizce
    "oturdu" diye basmak bu modülün açıkça yasakladığı şeydir.
    """
    from qlab.engine.em.factory import HUKUM_VERILMEDI, build_em, provenance_for
    from qlab.viz.em_frames import build_em_payload

    sonuc = build_em(scenario)
    yuk = build_em_payload(sonuc)
    merdiven = yuk.ladder
    provenance = provenance_for(
        scenario,
        windows=yuk.windows,
        ratios=tuple(float(x) for x in yuk.ratios),
        noise=tuple(float(x) for x in yuk.noise),
        peaks=tuple(int(x) for x in yuk.peaks),
        orders=merdiven.orders,
        p_inf=merdiven.p_inf,
        uncertainty=merdiven.uncertainty,
        verdict=merdiven.verdict,
    )

    print(
        f"{scenario.name}: {len(scenario.radii)} yarıçap, "
        f"r = {scenario.ratio:g}, S = {scenario.courant:g}"
    )
    for i, R in enumerate(scenario.radii):
        pencere = yuk.windows[i]
        print(
            f"  R={R:<6g} ızgara={pencere.izgara:<9} adım={pencere.steps:<6} "
            f"oran={yuk.ratios[i]:.9f}  tepe={int(yuk.peaks[i]):<3} "
            f"gürültü={yuk.noise[i]:.3e}"
        )
    hukum = (
        f"OTURDU({merdiven.settled_order})"
        if merdiven.settled_order is not None
        else f"{merdiven.verdict} — {HUKUM_VERILMEDI}"
    )
    print(f"  merdiven hükmü: {hukum}")
    genislik = etiket_genisligi()
    for etiket, deger in provenance.as_lines():
        print(f"{etiket:<{genislik}}: {deger}")

    if output is not None:
        alanlar = {
            "radii": yuk.radii,
            "spacings": yuk.spacings,
            "ratios": yuk.ratios,
            "noise": yuk.noise,
            "peaks": yuk.peaks,
            "orders": np.array(merdiven.orders, dtype=float),
            "verdict": np.array(merdiven.verdict),
            "field": yuk.field,
            "field_conductor": yuk.field_conductor,
            "provenance_json": json.dumps(
                dataclasses.asdict(provenance), ensure_ascii=False
            ),
        }
        # MERTEBE OKUNAMADIYSA ANAHTAR HIC YAZILMAZ -- `_run_spin`in isi
        # sigasi kuralinin aynisi. `p_inf = 0.0` yazmak "mertebe sifir cikti"
        # diye okunurdu; `None` "okunamadi" demektir ve ikisi ayni degerle
        # gosterilemez.
        if merdiven.p_inf is not None:
            alanlar["p_inf"] = np.array(merdiven.p_inf)
        if merdiven.uncertainty is not None:
            alanlar["uncertainty"] = np.array(merdiven.uncertainty)
        np.savez(output, **alanlar)
    return 0


def _run_grav(scenario: GravScenario, output: str | None) -> int:
    """Yerçekimi senaryosunu koşturur ve şeffaflık panelini basar.

    `_run_em`/`_run_spin`/`_run_downfold`/`_run_tb`/`_run_md` ile aynı kalıp:
    kendi panelini basar ve kendi `.npz`'sini yazar, `save()` yolundan
    geçmez.

    BU FONKSİYON HİÇBİR BEYAN METNİ YAZMAZ. Yöntem / Yaklaşımlar / İhmal
    edilen / Beklenen hata / Dairesellik / Çıpa sınıfı / **Kuram seçici** /
    Pencere / Yakınsama tanısı satırlarının tek kaynağı
    `engine/grav/factory.py`dir ve buradan yalnızca `provenance.as_lines()`
    basılır. İki kopya metin bu depoda bir kez panelin iki farklı şey iddia
    etmesine yol açtı (bkz. `tests/hf/test_hf_scenario.py`); kopya sayısı
    `tests/grav/test_grav_senaryo.py` ile kilitlidir — ve kilit bu modülde
    TAM CÜMLE değil **PARÇA** düzeyindedir, çünkü Modül 5'te bir mutant 55
    karakterlik bir parça kopyalayarak tam-cümle sayımından kaçmıştı.

    Aşağıdaki tablo BEYAN DEĞİL SAYIDIR (`_run_em`in yarıçap tablosu ve
    `_run_spin`in sıcaklık tablosuyla aynı sınıf). Tek kelimelik istisna
    `GR_DEGIL_ISARETI`dir ve o da yazılmaz, fabrikadan İTHAL EDİLİR.

    GÜRÜLTÜ TABANI SATIRLARI TABLODA DA KISALTILMAZ ve `DEJENERE` olan boru
    SAKLANMAZ: oranı okunamayan bir borunun yerine bir sayı basmak
    ("nan", "inf" ya da boş) o borunun ölçtüğü şeyi gizlerdi. Panelin işi
    ölçülmüş kusurları taşımaktır.
    """
    from qlab.engine.grav.factory import (
        GR_DEGIL_ISARETI,
        KAPALI_FORM_YOK,
        KURAMA_BAGLI_KAPALI_FORMLAR,
        build_grav,
        provenance_for,
    )
    from qlab.viz.grav_frames import build_grav_payload

    setup = build_grav(scenario)
    yuk = build_grav_payload(setup)
    provenance = provenance_for(
        scenario,
        setup=setup,
        floors=yuk.floors,
        second_order=yuk.second_order,
        closed_forms=yuk.closed_forms,
        eight=yuk.eight,
        angles=yuk.angles,
    )

    gr = (scenario.gamma, scenario.beta) == (1.0, 1.0)
    print(
        f"{scenario.name}: ayar {scenario.gauge}, "
        f"gamma={scenario.gamma:g}, beta={scenario.beta:g}, "
        f"lam={setup.theory.lam:g}"
        + ("" if gr else f"  <- {GR_DEGIL_ISARETI}")
    )
    print(f"  perihel katsayısı C  = {yuk.coefficient:.14f}")
    print(f"  ikinci mertebe   K   = {yuk.second_order:.12f}")
    print(f"  bükülme oranı        = {yuk.deflection:.14f}")
    print(f"  sekiz sabiti         = {yuk.eight:.14f}")
    print(
        f"  kararlılık (eliptik çiftin AÇISI, max|λ| DEĞİL) = "
        f"±{yuk.angles[0]:.12f} ve ±{yuk.angles[1]:.12f} rad"
    )
    # OKUNAMAYAN KAPALI-FORM TÜREV SESSİZCE ATLANMAZ: atlansaydı tablo "bu
    # kuram noktasında ISCO yok" ile "hesaplamayı unuttum" arasındaki farkı
    # göstermezdi. İşaret burada YAZILMAZ, fabrikadan İTHAL EDİLİR.
    for ad in KURAMA_BAGLI_KAPALI_FORMLAR:
        if ad in yuk.closed_forms:
            print(f"  kapalı-form türev    {ad:<18} = {yuk.closed_forms[ad]:.14f}")
        else:
            print(f"  kapalı-form türev    {ad:<18} = {KAPALI_FORM_YOK}")
    for ad, deger in yuk.closed_forms.items():
        if ad not in KURAMA_BAGLI_KAPALI_FORMLAR:
            print(f"  kapalı-form türev    {ad:<18} = {deger:.14f}")
    for f in yuk.floors:
        oran = "S/G OKUNAMADI" if math.isnan(f.ratio) else f"S/G={f.ratio:.4e}"
        isaret = "" if f.anchor else "   (ÇIPA DEĞİL)"
        print(
            f"  taban {f.value:.6e}  {oran:<18} {f.verdict:<9}"
            f"{f.label}{isaret}"
        )
    genislik = etiket_genisligi()
    for etiket, deger in provenance.as_lines():
        print(f"{etiket:<{genislik}}: {deger}")

    if output is not None:
        np.savez(
            output,
            phi=yuk.phi,
            u=yuk.u,
            r=yuk.r,
            x=yuk.x,
            y=yuk.y,
            coefficient=np.array(yuk.coefficient),
            second_order=np.array(yuk.second_order),
            deflection=np.array(yuk.deflection),
            eight=np.array(yuk.eight),
            angles=np.array(yuk.angles),
            closed_forms=np.array(list(yuk.closed_forms.values())),
            closed_form_names=np.array(list(yuk.closed_forms)),
            floor_values=np.array([f.value for f in yuk.floors]),
            floor_ratios=np.array([f.ratio for f in yuk.floors]),
            floor_verdicts=np.array([f.verdict for f in yuk.floors]),
            floor_pipelines=np.array([f.pipeline for f in yuk.floors]),
            floor_is_anchor=np.array([f.anchor for f in yuk.floors]),
            provenance_json=json.dumps(
                dataclasses.asdict(provenance), ensure_ascii=False
            ),
        )
    return 0


def _run_bridge(scenario: BridgeScenario, output: str | None) -> int:
    """Köprü haritası senaryosunu koşturur ve şeffaflık panelini basar.

    `_run_grav`/`_run_em`/`_run_spin`/`_run_downfold`/`_run_tb`/`_run_md` ile
    aynı kalıp: kendi panelini basar ve kendi `.npz`'sini yazar, `save()`
    yolundan geçmez.

    BU FONKSİYON HİÇBİR BEYAN METNİ YAZMAZ. Yöntem / Yaklaşımlar / İhmal
    edilen / Beklenen hata / Dairesellik / Çıpa sınıfı / Kuram seçici /
    Pencere / Yakınsama tanısı satırlarının tek kaynağı `qlab.bridge_map`tir
    (saf tanım katmanı — bir motor fabrikası DEĞİL, çünkü bu modülün motoru
    yok); `viz/bridge_frames.py` sayıları toplar, buradan yalnızca
    `provenance.as_lines()` basılır. Kopya sayısı
    `tests/bridge/test_bridge_senaryo.py` ile kilitlidir ve kilit Modül 6'daki
    gibi PARÇA düzeyindedir — eşik orada yeniden ölçüldü.

    Aşağıdaki tablo BEYAN DEĞİL SAYIDIR (`_run_grav`ın taban tablosuyla aynı
    sınıf). Kova adları yazılmaz, `bridge_map.KOVALAR`dan İTHAL EDİLİR.

    `None` İLE `0.0` AYNI HÜCREYE BASILMAZ: ölçülmemiş ya da iki sayıya
    bölünmüş bir büyüklük tabloda `None` diye görünür, `tb -> spin`in bit
    bit ölümü `0.000000e+00` diye — ikisi ayrı şeydir (`OLCULMEDI` kovasının
    gerekçesi). `.npz`'de de aynı ayrım: `magnitude` `nan` taşır ama yanında
    `magnitude_is_none` maskesi vardır; `nan` tek başına "ölçüldü, tanımsız
    çıktı" diye okunabilirdi.

    İKİ TABAN İKİ SATIR: A1 ile A2 tek satıra sığdırılmaz, çünkü ikisi
    ayrı soruya cevaptır (aritmetik / hata çubuğu) ve tek satır "taban"
    diye okunurdu — bu modülün taşıyıcı hükmü tam olarak bunun yasağıdır.
    """
    from qlab.bridge_map import KOVA_TABLOSU, KOVALAR
    from qlab.viz.bridge_frames import build_bridge_payload, provenance_for

    yuk = build_bridge_payload(scenario)
    provenance = provenance_for(yuk)

    print(
        f"{scenario.name}: {len(KOVA_TABLOSU)} çift, "
        f"{len(yuk.kenarlar)} kod kenarı, "
        f"{sum(1 for n in yuk.eksen.values() if n > 0)} eksenli motor"
    )
    for kova in KOVALAR:
        girdiler = [f for f in KOVA_TABLOSU if f.kova == kova]
        print(f"  {kova:<13} {len(girdiler)}")
        for f in girdiler:
            buyukluk = "None" if f.buyukluk is None else f"{f.buyukluk:.6e}"
            print(f"      {f.kaynak:>4} -> {f.hedef:<4}  {buyukluk}")
    print(
        "  kod kenarı       : "
        + (", ".join(f"{a} -> {b}" for a, b in sorted(yuk.kenarlar)) or "YOK")
    )
    print(
        "  units ekseni     : "
        + " ".join(f"{ad}={n}" for ad, n in sorted(yuk.eksen.items()))
    )
    for etiket, f in (("A1", yuk.a1), ("A2", yuk.a2)):
        print(
            f"  taban {etiket} {f.tur:<12} {f.taban:.8e}  S/G={f.oran:.4e}  "
            f"{f.hukum:<9}[{f.boru}]"
        )
    print(
        f"  gerçek hata          {yuk.gercek_hata:.8e}  "
        f"gerçek/A1={yuk.gercek_hata / yuk.a1.taban:.6g}  "
        f"A2/gerçek={yuk.a2.taban / yuk.gercek_hata:.6g}"
    )
    if yuk.tutarsizliklar:
        for sorun in yuk.tutarsizliklar:
            print(f"  TABLO ≠ KOD          {sorun}")
    genislik = etiket_genisligi()
    for etiket, deger in provenance.as_lines():
        print(f"{etiket:<{genislik}}: {deger}")

    if output is not None:
        kenarlar = sorted(yuk.kenarlar)
        eksen = sorted(yuk.eksen.items())
        np.savez(
            output,
            pair_a=np.array([f.kaynak for f in KOVA_TABLOSU]),
            pair_b=np.array([f.hedef for f in KOVA_TABLOSU]),
            kova=np.array([f.kova for f in KOVA_TABLOSU]),
            magnitude=np.array(
                [math.nan if f.buyukluk is None else f.buyukluk
                 for f in KOVA_TABLOSU]
            ),
            magnitude_is_none=np.array([f.buyukluk is None for f in KOVA_TABLOSU]),
            edges=np.array(kenarlar, dtype=str).reshape(-1, 2),
            units_axis_names=np.array([ad for ad, _ in eksen]),
            units_axis_counts=np.array([n for _, n in eksen]),
            floor_turler=np.array([yuk.a1.tur, yuk.a2.tur]),
            floor_values=np.array([yuk.a1.taban, yuk.a2.taban]),
            floor_signals=np.array([yuk.a1.sinyal, yuk.a2.sinyal]),
            floor_ratios=np.array([yuk.a1.oran, yuk.a2.oran]),
            floor_verdicts=np.array([yuk.a1.hukum, yuk.a2.hukum]),
            floor_pipelines=np.array([yuk.a1.boru, yuk.a2.boru]),
            real_error=np.array(yuk.gercek_hata),
            inconsistencies=np.array(list(yuk.tutarsizliklar), dtype=str),
            converged=np.array(provenance.converged),
            provenance_json=json.dumps(
                dataclasses.asdict(provenance), ensure_ascii=False
            ),
        )
    return 0


def _run_md(scenario: MDScenario, output: str | None) -> int:
    """MD senaryosunu koşturur ve şeffaflık panelini basar."""
    from qlab.engine.md.factory import (
        build,
        energy_drift,
        energy_oscillation,
        omega_max_frame0,
        omega_max_path_frames,
        path_frame_tracker,
        provenance_for,
    )
    from qlab.engine.md.integrator import Trajectory, run

    durum, alan, termostat = build(scenario)
    durum = durum.remove_center_of_mass_motion()
    # A PRIORI TANI (D2-i, 10.10.2026): koşudan ÖNCE, kare-0 geometrisinin en
    # hızlı kütle ağırlıklı modu (TB ve -- D2-ii -- HF NVE; öteki koşularda
    # None). S'nin paydası tüm KE'dir ve titreşime katılmayan KE onu
    # seyreltir; omega0*dt bakmaz. HF'de Hessian alanın TAZE kopyasıyla alınır
    # (fabrikada): `alan`ın kararlılık sayaçları yalnız koşunun adımlarını sayar.
    omega0 = omega_max_frame0(scenario, durum, alan)
    # YOL TANISI (D2-vi): kare 0 yolun en hızlı modunu görmez (sonradan
    # çarpışan parçalar, ısıl keşif; HF'de gerilmiş başlangıç). TB ve HF
    # NVE'de koşu alanı saran izleyiciyle yapılır; o HER ADIMI görür (kayıtlı
    # kareleri değil), en kısa çift mesafesinin (TB'de bir de kütle ağırlıklı
    # sertlik tepesinin) karesini seçer; koşudan sonra o karelerde aynı
    # Hessian alınır. Öteki koşularda None.
    # Kare-0 Hessian'ından SONRA kurulur ve yalnız `run`a verilir: `alan`
    # adı değişmez (HF'nin kararlılık sayaçları `alan`dan okunur).
    izleyici = path_frame_tracker(scenario, alan, durum.masses)

    if termostat is None:
        yorunge = run(
            durum, alan if izleyici is None else izleyici, dt=scenario.dt,
            n_steps=scenario.n_steps, save_every=scenario.save_every,
        )
    else:
        # Termostatlı koşu: `step` imzası velocity_verlet_step ile aynıdır.
        _, kuvvetler = alan(durum.positions)
        zamanlar, konumlar, hizlar, kinetik, potansiyel = [0.0], [durum.positions], \
            [durum.velocities], [durum.kinetic_energy()], [alan(durum.positions)[0]]
        for adim in range(1, scenario.n_steps + 1):
            durum, enerji, kuvvetler = termostat.step(durum, kuvvetler, alan, scenario.dt)
            if adim % scenario.save_every == 0:
                zamanlar.append(adim * scenario.dt)
                konumlar.append(durum.positions)
                hizlar.append(durum.velocities)
                kinetik.append(durum.kinetic_energy())
                potansiyel.append(enerji)
        yorunge = Trajectory(
            times=np.array(zamanlar), positions=np.array(konumlar),
            velocities=np.array(hizlar), kinetic=np.array(kinetik),
            potential=np.array(potansiyel),
        )

    # SÜRÜKLENME, salınım DEĞİL: blok ortalamalarının baş-son farkı. Hesap
    # (blok sayısı, "en az 2 kayıt" kapısı, TÜRE BAĞLI payda) G5'ten beri
    # `md/factory.py::energy_drift`te: TB'de enerjinin sıfırı keyfidir ve
    # |blok[0]|'a bölmek hükmü o sıfıra bağlıyordu (spec B8; ölçüm orada).
    suruklenme = energy_drift(scenario, yorunge)
    # SALINIM (D2, 09.10.2026): TB ve (D2-ii) HF NVE'de iri dt'yi yargılar,
    # öteki koşularda None. Uçlar integratörün HER adımından gelir (`Trajectory.total_min` ...);
    # buradaki termostat döngüsü onları biriktirmez ve NVT'de S hesaplanmaz.
    salinim = energy_oscillation(scenario, yorunge)
    # D2-ii: seçilen kare kare 0'ın kendisiyse fabrika Hessian'ı ikinci kez almaz.
    omega_yol = omega_max_path_frames(scenario, izleyici, durum, alan, omega_frame0=omega0)
    yol_kareleri = None if izleyici is None else izleyici.tanilar()

    # KISIT SAYISI TERMOSTATA BAĞLIDIR: ÖLÇÜLDÜ -- Langevin kütle merkezini
    # YENİDEN ISITIYOR (<KE_COM>/(1.5kT) = 1.098, çıkarılmış kalsaydı 0.000),
    # yani NVT koşusunda 3N-3 kullanmak <T>'yi %0.935 yukarı kaydırıyor
    # (89.393 K yerine 88.565 K). Task 8'in varsayılanı COM'un çıkarılmış
    # KALDIĞI ön koşuluna dayanır ve o ön koşul burada geçerli değil.
    kisit = 3 if termostat is None else 0

    # B10 (03.10.2026): HF kuvvet alanı her adımın merkez SCF'inde iç/dış
    # kararlılığı ölçer; en kötü hüküm panele kendi satırıyla girer.
    kararlilik = {}
    if getattr(alan, "kararlilik_cagri", 0):
        n = alan.kararlilik_cagri
        kararlilik = {
            "stability": alan.en_kotu_ic,
            "external_stability": alan.en_kotu_dis,
            "kararlilik_tanisi": {
                "kararlilik_merkez_scf": n,
                "ic_kararsiz": f"{alan.ic_kararsiz_cagri}/{n}",
                "dis_kararsiz": f"{alan.dis_kararsiz_cagri}/{n}",
                "max_bag_bohr": round(alan.max_bag_bohr, 6),
            },
        }
    provenance = provenance_for(
        scenario, energy_drift=suruklenme, energy_oscillation=salinim,
        omega_max_frame0=omega0, omega_max_path_frames=omega_yol,
        path_frames=yol_kareleri, n_constraints=kisit, **kararlilik
    )
    genislik = etiket_genisligi()  # V12: `_run_tb` ile aynı (gerekçe orada)
    for etiket, deger in provenance.as_lines():
        print(f"{etiket:<{genislik}}: {deger}")

    if output is not None:
        # Task 13'ün modülü BURADA import edilir: Task 12 tek başına
        # commit edildiğinde CLI'ın kırık olmaması için (ölçüldü -- koşulsuz
        # import `ModuleNotFoundError` veriyordu).
        from qlab.viz.md_frames import build_md_payload

        yuk = build_md_payload(yorunge, durum.masses, n_constraints=kisit)
        np.savez(
            output,
            times_fs=yuk.times_fs, positions=yorunge.positions,
            velocities=yorunge.velocities, kinetic=yuk.kinetic,
            potential=yuk.potential, total=yuk.total,
            temperature_kelvin=yuk.temperature_kelvin,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
