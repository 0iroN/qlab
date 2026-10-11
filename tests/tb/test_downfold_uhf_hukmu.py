"""B7-a / B7-c: downfold panelinin UHF hukmu -- "YEREL" ve isaret uyumu.

SINIF: PANEL DENETIMI (transkripsiyon) + bir OLCUM KILIDI. Hicbiri bir fizik
cipasi DEGILDIR: tutulan sayi bir referansa degil, panelin soyledigi seyin
olculen duruma UYUP UYMADIGINA bakar.

B7-a. Downfold paneli `Provenance.stability` satirini BASMAZ; UHF hukmunu
yalniz `UHFSearchReport.as_line()` uzerinden, "Yakinsama" satirinin ICINDE
tasir. O satir once "donen cozum UHF-kararli" diyordu -- "YEREL" yok, kuresel
iddia sinirlamasi yok. Kamu yolunda olculdu (B7-a on olcumu, 210 arama):
H8 halkasi d = 3.6 bohr, seeds = 1 UHF-kararli (lambda_min ~ +0.83 ... +0.89)
ama seeds = 2'nin buldugu kararli durumun ~0.0328 Ha USTUNDE. Inilen yerel
minimum OpenBLAS cekirdegine bagli (Prescott/Haswell -3.743722958588,
varsayilan/Nehalem -3.743707950431) -- bu yuzden kesin enerji degil yalniz
"aralik > 0.01 Ha" kilitlenir.

B7-c. Ayni satirda isaret celiskisi: N=6 halkasinin RHF -> UHF catallanmasinin
(Rc = 2.233818500 bohr) hemen ustunde seeds = 1 bir UHF EYERI dondurur
(lambda_min -7.68e-6 at Rc + 1e-5; izlemenin 1e-9 kabul esiginin beyanli kor
bolgesi) ve kademeli yapi esigi gecildigi icin panel
"Yakinsama : ✓ ... ✗ donen cozum UHF EYERI" basiyordu. Alti cekirdek
yapilandirmasinda (varsayilan, tek is parcacigi, Sandybridge, Prescott,
Haswell, Nehalem) ayni lambda olculdu.
"""

from __future__ import annotations

import pytest

import qlab.engine.tb.factory as factory_modulu
from qlab.engine.hf.multistart import search_uhf
from qlab.engine.tb.downfold import (
    SPIN_KIRIK_ESIGI,
    STAGGERING_TOLERANCE,
    DistanceLaws,
    hydrogen_ring,
)
from qlab.engine.tb.factory import build_downfold
from qlab.scenario.schema import DownfoldScenario

#: N=6 halkasinin RHF -> UHF gecis noktasi (bohr; tests/tb/test_downfold_senaryo.py).
RC6 = 2.233818500


def test_B7a_UHF_KARARLI_kuresel_DEGIL_ve_satir_YEREL_diyor():
    """Kamu `search_uhf`: iki kararli sonuc, farkli enerji -> biri kuresel degil.

    seeds = 1'in sonucu kararli AMA seeds = 2'ninkinden > 0.01 Ha yukarida;
    satiri "YEREL" der ve kuresel iddia etmez. Downfold yolunda ayni durum
    `search_stable is True` tasir ve panelin ✗'inin IKI sebebi vardir: YAPI
    kapisi (kademeli artik ~0.75) VE Delta kapisi (B7dog-2: |Delta| 6.8e-12
    < `SPIN_KIRIK_ESIGI`) -- B7-c kapisi DEGIL. Durum antiferromanyetik
    DEGIL ama RHF limiti de DEGIL: <S^2> 3.5606, yani tani "spin kirik ama
    kademeli degil" der, "RHF limiti" DEMEZ (|Delta| < esik RHF limitini
    gerektirmez; etiket bu yuzden <S^2>'den secilir). Bu bir GOZLEMDIR:
    olculen kuresel-olmayan minimumlarin hepsi AF degildi; AF olan
    kuresel-olmayan bir minimumun var olmadigi iddia EDILMEZ.
    """
    mol = hydrogen_ring(8, 3.6)
    tek, iki = search_uhf(mol, "sto-3g", seeds=1), search_uhf(mol, "sto-3g", seeds=2)
    assert tek.stability.stable and iki.stability.stable
    assert tek.energy - iki.energy > 0.01
    satir = tek.as_line()
    assert "donen cozum UHF YEREL minimum" in satir
    assert "KURESEL minimum iddia EDILMEZ" in satir

    (p,), kaynak = build_downfold(
        DownfoldScenario(name="B7-a tek tohum", ring_size=8, distances_bohr=(3.6,), seeds=1)
    )
    assert p.energy_uhf == pytest.approx(tek.energy, abs=1e-9)
    assert p.search_stable is True
    assert p.staggering_residual > 0.1
    assert kaynak.converged is False
    assert "UHF YEREL minimum" in kaynak.diagnostics["arama"]
    assert abs(p.delta) < SPIN_KIRIK_ESIGI and p.s_squared > 1.0
    assert "spin kirik ama kademeli degil" in kaynak.convergence_diagnosis
    assert "RHF limiti" not in kaynak.convergence_diagnosis


def test_B7c_EYER_donen_mesafe_YAKINSAMA_isaretini_dusuruyor(monkeypatch):
    """Uc mesafe, EYER ORTADA: kararli / eyer / kararli.

    Eyerin ortada olmasi bir secimdir: (eyer, kararli) "yalniz ilk mesafeye
    bak", (kararli, eyer) "yalniz sona bak" mutantini hayatta birakirdi.
    On kosul (`search_stable` deseni ve kademeli artik) ayrica kilitlenir:
    izleme bir gun o eyerden inmeyi ogrenirse test SESLI kirilir.

    DELTA KAPISI BU TESTTE KAPATILIR (B7dog-2, 11.10.2026) ve sebebi
    olculmustur: 2.0 (Rc'nin altinda) ve Rc + 1e-5 (seeds = 1'in dondurdugu
    eyer) IKISI DE spin-KIRILMAMISTIR (|Delta| ~ 3.4e-16). Kapi acik kalsaydi
    ✗ onlardan da gelirdi ve `search_stable` kosulunu dusuren bir mutant bu
    testi YESIL birakirdi. Fabrikanin esigi -1'e cekilir (|Delta| < -1 hic
    dogru degil) -- yani kalan tek ✗ kaynagi eyer kapisidir; on kosul
    (iki mesafe gercekten kirilmamis) ayrica kilitlenir ki patch'in gerekcesi
    bayatlarsa test sesli kirilsin.
    """
    senaryo = DownfoldScenario(
        name="B7-c catallanma kenari",
        ring_size=6,
        distances_bohr=(2.0, RC6 + 1e-5, 2.6),
        seeds=1,
    )
    monkeypatch.setattr(factory_modulu, "SPIN_KIRIK_ESIGI", -1.0)
    parametreler, kaynak = build_downfold(senaryo)

    assert [abs(p.delta) < SPIN_KIRIK_ESIGI for p in parametreler] == [True, True, False]
    assert kaynak.convergence_diagnosis is None  # Delta kapisi gercekten kapali
    assert [p.search_stable for p in parametreler] == [True, False, True]
    assert max(p.staggering_residual for p in parametreler) < STAGGERING_TOLERANCE
    assert kaynak.converged is False
    satir = dict(kaynak.as_lines())["Yakınsama"]
    assert satir.startswith("✗")
    assert "✓" not in satir
    assert "✗ donen cozum UHF EYERI" in kaynak.diagnostics["arama"]


def test_B7c_MESAFE_YASASI_hukum_UYDURMUYOR():
    """`DistanceLaws.at` SCF kosmaz -> `search_stable` None ("olculmedi").

    Yasa ELLE kurulur (uydurma yok, SCF yok); hizli yoldadir. True dondurse
    yasadan gelen bir parametre seti "UHF-kararli" diye okunurdu.
    """
    yasa = DistanceLaws(
        t_prefactors=(1.0, 0.1),
        t_decays=(1.0, 1.5),
        t_signs=(-1.0, 1.0),
        epsilon_infinity=0.0,
        epsilon_coefficient=0.5,
        delta_prefactor=0.01,
        delta_growth=1.0,
        window_bohr=(2.6, 3.0),
        ring_size=6,
        basis_name="sto-3g",
        t_log_residual_rms=(0.0, 0.0),
        delta_log_residual_rms=0.0,
        epsilon_residual_rms=0.0,
    )
    p = yasa.at(2.8)
    assert "SCF KOSULMADI" in p.search_line
    assert p.search_stable is None


def test_B7c_KOR_BANT_beyani_yumusak_modda_eyer_KARARLI_sayiliyor():
    """BEYANLI SINIRIN kilidi -- istenen davranis DEGIL, dogru BEYAN.

    lambda (-1e-8, 0) yumusak mod bandinda ayirt edilmez: N=6 halkasinda
    Rc + 5e-9'da lambda ~ -3.76e-9 (gercekte eyer) ama `search_stable` True
    ve arama satiri "UHF YEREL minimum" der; Rc + 3e-8'de lambda ~ -2.30e-8
    bandin DISINDA ve kararsiz. Esigi 0'a ceken bir degisiklik
    (`lowest_eigenvalue > 0`) ya da bandi genisleten bir degisiklik README'nin
    kor bant cumlesini bayatlatir -- bu test o cumleyi tutar. Paylar:
    -3.76e-9 sinira 6.2e-9, -2.30e-8 1.3e-8.

    PANEL ARTIK ✗ BASAR (B7dog-2, 11.10.2026), ama sebebi kararlilik DEGIL
    Delta kapisidir: o banttaki eyer spin-KIRILMAMISTIR (olculdu |Delta|
    1.7e-16, <S^2> 2.7e-15; Rc + 1e-9 ... 1.3e-8 ve seeds 1-20'nin hepsinde
    |Delta| <= 4.2e-16). Tani bu yuzden "RHF limiti" der. Kararlilik kor
    bandinin KENDISI durur -- arama satiri hala "UHF YEREL minimum" der.
    Eskiden burada `converged is True` (panel ✓) kilitliydi.
    """
    ic = search_uhf(hydrogen_ring(6, RC6 + 5e-9), "sto-3g", seeds=1)
    dis = search_uhf(hydrogen_ring(6, RC6 + 3e-8), "sto-3g", seeds=1)
    assert -1e-8 < ic.stability.lowest_eigenvalue < 0.0
    assert ic.stability.stable
    assert dis.stability.lowest_eigenvalue < -1e-8
    assert not dis.stability.stable

    (p,), kaynak = build_downfold(
        DownfoldScenario(name="B7-c kor bant", ring_size=6, distances_bohr=(RC6 + 5e-9,), seeds=1)
    )
    assert p.search_stable is True
    assert "UHF YEREL minimum" in kaynak.diagnostics["arama"]
    assert abs(p.delta) < SPIN_KIRIK_ESIGI
    assert kaynak.converged is False
    assert "RHF limiti" in kaynak.convergence_diagnosis
    assert dict(kaynak.as_lines())["Yakınsama"].startswith("✗")
