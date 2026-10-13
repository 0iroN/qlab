"""Modül 7 / Task 4 — düşen adaylar (27.09.2026'dan beri üç, 28.09.2026'dan beri dört: üç spin hedefli
+ `hf → tb`; iki EM tautolojisi `md → em`/`tb → em` burada SINANMAZ) SESSİZCE DÜŞÜRÜLMEZ; tautoloji olarak,
SAYILARIYLA sevk edilir.

`bridge_map.py`de `hf → tb` ve `tb → spin` girdileri Task 1'den beri
TAUTOLOJİ kovasındaydı ama `buyukluk` alanları `None` idi ("Task 4'ün işi").
Bu dosya o alanları dolduran sayıları MOTORU KOŞTURARAK üretir ve
`bridge_map.py`deki literallere karşı sınar — `bridge_map` hiçbir motoru
import edemez, o yüzden sayılar orada literal, doğrulama burada canlı.

DÖRT TAUTOLOJİ (Task 4'ün ikisi + 27.09.2026 `hf → spin` + 28.09.2026 `tdse → spin`), İKİ MEKANİZMA + BİR AYAR ETİKETİ:

  `hf → tb`   `t_m` bir `np.mean` olarak TANIMLANIYOR (`downfold.py:259-271`)
              ve aynı ortalama `==` ile geri okunuyor. Uyum değil, tanımın
              yeniden koşulması: aynı elemanlar TERS sırayla toplanınca 22
              kabuğun 10'unda `==` düşüyor (en çok 1 ulp). "On altı hane"
              iddiası ÖLÇÜLDÜ ve DARALTILDI: özdeşlik referans pencerede
              15 hane (bağıl 2.276e-15) ve SİSTEME BAĞLI (N=8 d=2.0'da
              1.090e-08 — yedi mertebe). Pencere (N, d, baz, seeds) her
              sayının yanında yazılıdır.

  `tb → spin` `J = 4t²/U` köprüsü BİT BİT ölü: `coupling` çıktıya YALNIZ
              `coupling/temperature` oranı olarak giriyor
              (`metropolis.py:189-191`), `h = 0`'da başka enerji ölçeği yok.
              Beş `J` (6.70 kat aralık) → max|m farkı| = 0.0. `U`'yu
              1016 kat büyütmek de, `z`'yi uyuşturmak da kenarı AÇMIYOR,
              çünkü engel oran-sadeleşmesidir. Kontrol: `h = 0.1`'de aynı
              iki `J` 9.961e-02 ayrışıyor — motor ikinci bir ölçek varsa
              `J`'yi görür; TB o ölçeği üretmez.

  `hf → spin` (27.09.2026, ÜÇÜNCÜ aday, kenarsız): H2 Yamaguchi `J`sinin
              büyüklüğü aynı oranda sadeleşir; İŞARETİ `h = 0` çift L'de bir
              AYAR ETİKETİDİR — AFM kademeli seri FM tekbiçim seriyle bit
              bit aynı. İşaret ŞARTLIDIR: İZLEMESİZ `seeds = 1` kırık
              simetriyi bulamaz ve formül sessizce +J verir (B7, 09.10.2026:
              varsayılan kararlılık izlemesi onu buluyor, J < 0).

  `tdse → spin` (28.09.2026, kenarsız): 1B soft-Coulomb H2 `J`si GERÇEK ama aynı oranda sadeleşir;
              300 K'de R = 3/4/5 |m| = 1.0 (donar ya da doyar), R = 6 aynı T/|J|'de J = 1'in etiketi.

PENCERELER (her sayı bunların içindedir, dışına taşınamaz):
  hf → tb   altı sistem (N, d) = (6, 2.6) (6, 3.0) (6, 3.6) (8, 2.0) (8, 2.6)
            (10, 2.6) bohr, STO-3G, `seeds = 12`, tam menzil `n_neighbour = N/2`
            (seeds = 16 ile (6, 2.6) ve (8, 2.0)'da her sayı BİT BİT AYNI çıktı;
            iki ayrı süreçte de aynı — ölçüldü 02.09.2026, numpy 2.5.1).
  tb → spin Ising, L = 32, h = 0, tohum 20260822, n_burn = 500, n_sweep = 1500,
            T = 2.269185 · J. PLANIN 0.200322266 KABUL ORANI BU PENCEREDE
            DEĞİL (0.187216796875) ve 288 pencerelik taramada da bulunamadı
            (L ∈ {8, 16, 32} × 4 burn × 5 sweep × 10 tohum): planın sayısı
            penceresiz yazılmıştı ve taşınamaz.
  hf → spin H2 R = 5/6/7 bohr (`%.10g`, unit="bohr"), STO-3G, `search_uhf(seeds
            = 12)` varsayılan eşikler; Ising L = 32, h = 0, tohum 20260822,
            T = 2.269185 · |J|, 2000 sweep `n_burn = 0` (burn-in DAHİL), Néel.
  tdse → spin `TDSE_J` LİTERAL (Grid2D sanal zaman, n = 256, L = 40, dt = 0.01, a = b = 1, R = 3/4/5/6);
            spin: Protokol A `hf → spin` penceresi, Protokol B `spin_tarama.yaml` + 300 K.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

import qlab.engine.spin.lattice as lattice_mod
import qlab.engine.spin.metropolis as metropolis_mod
import qlab.engine.tb.downfold as downfold_mod
from qlab.bridge_map import (
    KOVA_SAYILARI,
    KOVA_TABLOSU,
    YENIDEN_URETILMEDI,
    Kova,
    _HF_SPIN_M_FARKI,
    _HF_TB_OZDESLIK_BAGIL,
    _SPIN_BUYUKLUGU_SILER,
    _TB_SPIN_M_FARKI,
    _TDSE_SPIN_M_FARKI,
)
from qlab.engine.spin.factory import build_spin
from qlab.errors import ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.engine.hf.multistart import search_uhf
from qlab.engine.spin.lattice import SpinState, magnetisation, neighbour_sum
from qlab.engine.spin.metropolis import run_metropolis
from qlab.engine.spin.models import IsingModel
from qlab.engine.tb.downfold import (
    DownfoldedParameters,
    _effective_hamiltonians,
    downfold_uhf,
    model_from_downfold,
)
from qlab.engine.tb.hamiltonian import finite_ring_matrix
from qlab.types import Molecule
from qlab.units import BOHR_TO_ANGSTROM, KB_HARTREE_PER_KELVIN

PROJE_KOKU = Path(__file__).resolve().parents[2]
KAYNAK = PROJE_KOKU / "src" / "qlab"
TESTLER = PROJE_KOKU / "tests"

# --- hf -> tb penceresi -------------------------------------------------------
BAZ = "sto-3g"
SEEDS = 12
ALTI_SISTEM = ((6, 2.6), (6, 3.0), (6, 3.6), (8, 2.0), (8, 2.6), (10, 2.6))
#: `slow` ÖLÇÜLDÜ, devralınmadı — dosyanın sonundaki nota bakınız.
SISTEM_PARAMLARI = [
    pytest.param(n, d, id=f"N{n}_d{d}", marks=([pytest.mark.slow] if (n, d) in ((8, 2.6), (10, 2.6)) else []))
    for n, d in ALTI_SISTEM
]

#: Ters sırayla toplanınca `==`'i düşen kabuk sayısı (02.09.2026, numpy 2.5.1).
#: Toplam 10/22. Taşıyıcı iddia "en az birinde düşüyor"dur; sayının kendisi
#: bu ortamın kaydıdır (numpy'nin ikili toplama sırasına bağlı).
TERS_SIRA_DUSEN = {
    (6, 2.6): 2, (6, 3.0): 0, (6, 3.6): 2, (8, 2.0): 0, (8, 2.6): 2, (10, 2.6): 4,
}

#: max|model − H_eff^α| ile en büyük ÇİFT kabuk yayılımının yarısı arasındaki
#: BAĞIL fark, tam menzilde. 15 hane yalnız N = 6'da; N ≥ 8'de 1e-8.
OZDESLIK_BAGIL = {
    (6, 2.6): 2.276006188306662e-15,
    (6, 3.0): 2.024221721222578e-15,
    (6, 3.6): 4.728665243959321e-14,
    (8, 2.0): 1.0898620259967446e-08,
    (8, 2.6): 1.3100825408206874e-08,
    (10, 2.6): 3.768953577346613e-08,
}

#: max|eig(H_eff^α) − ε_UHF^α| — Löwdin adımının kaybı. En küçüğü (6, 2.6),
#: en büyüğü (8, 2.6); N ile MONOTON DEĞİL ((8, 2.0) (6, 3.0)'dan küçük).
LOWDIN_FARKI = {
    (6, 2.6): 3.490441824460788e-10,
    (6, 3.0): 2.6319992851853158e-09,
    (6, 3.6): 1.391987247689741e-09,
    (8, 2.0): 7.194357332096502e-10,
    (8, 2.6): 2.8906645410664566e-09,
    (10, 2.6): 2.403742593859448e-09,
}

# --- tb -> spin penceresi -----------------------------------------------------
TC_BOLU_J = 2.269185
L = 32
TOHUM = 20260822
N_BURN = 500
N_SWEEP = 1500
#: Planın beş `J`si (spec §7.3). İlk dördü canlı `t1`den türetilir ve aşağıda
#: doğrulanır; BEŞİNCİSİNİN `U`SU PLANDA YAZMIYOR — geri hesaplandığında
#: U/|t| = 13.630622 çıkıyor ve gerekçede öyle yazılıdır.
BES_J = (0.801032362, 0.750154440, 0.407144652, 0.203572326, 0.119479403)
#: Boşluk denkleminden `U/|t|` (N = 8, d = 2.0, seeds = 12); canlı doğrulanır.
U_BOLU_T = 2.0330996395880065


# --- yardımcılar (motorun yardımcıları İTHAL EDİLMEZ) -------------------------


def _halka(n: int, d: float) -> Molecule:
    """`downfold.hydrogen_ring`in KOPYASI — paylaşılan yardımcıyı bozan bir
    mutant iki tarafı da aynı yönde bozar (kesme testinin gerekçesi)."""
    yaricap = d / (2 * np.sin(np.pi / n))
    a = yaricap * BOHR_TO_ANGSTROM
    return Molecule.from_string(
        "; ".join(
            f"H {a*np.cos(2*np.pi*k/n):.17g} {a*np.sin(2*np.pi*k/n):.17g} 0"
            for k in range(n)
        )
    )


def _halka_mesafesi(i: int, j: int, n: int) -> int:
    return min(abs(i - j), n - abs(i - j))


def _kabuk(h: np.ndarray, n: int, m: int, *, ters: bool = False) -> list[float]:
    """Halka mesafesi `m` olan elemanlar. `ters=False` `downfold_uhf`un kendi
    sırasıdır (önce i, sonra j); `ters=True` aynı elemanları j-i sırasında
    verir — SAYILAR AYNI, SIRA FARKLI."""
    if ters:
        return [h[i, j] for j in range(n) for i in range(n) if _halka_mesafesi(i, j, n) == m]
    return [h[i, j] for i in range(n) for j in range(n) if _halka_mesafesi(i, j, n) == m]


def _girdi(kaynak: str, hedef: str):
    esler = [f for f in KOVA_TABLOSU if f.cift == tuple(sorted((kaynak, hedef)))]
    assert len(esler) == 1, f"{kaynak}→{hedef} için tabloda {len(esler)} girdi"
    return esler[0]


def _fonksiyon(agac: ast.AST, ad: str) -> ast.FunctionDef:
    return next(
        d for d in ast.walk(agac) if isinstance(d, ast.FunctionDef) and d.name == ad
    )


def _yukler(govde: ast.AST, ad: str) -> list[ast.Name]:
    return [
        d for d in ast.walk(govde)
        if isinstance(d, ast.Name) and d.id == ad and isinstance(d.ctx, ast.Load)
    ]


def _atama(govde: ast.AST, ad: str) -> ast.Assign:
    return next(
        d for d in ast.walk(govde)
        if isinstance(d, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == ad for t in d.targets)
    )


def _kos(J: float, *, alan: float = 0.0):
    """`coupling = +J`: motorun konvansiyonunda (E = −c Σ s·s) FERROMANYETİK.

    İŞARET BEYANI (27.09.2026): kaynak ANTİFERROMANYETİKTİR (`downfold.py:20`,
    `tb/factory.py:143` ve `tb/factory.py:183-187`); +J bir AYAR SEÇİMİDİR — `h = 0` çift
    L'de AFM(−J) Néel başlangıçlı koşum bununla bit bit aynı, kilitli 0.687930
    doğru işaretli AFM'nin KADEMELİ ortalamasıdır (tekbiçimde 0.009724).
    İŞARETİ ÇEVİRME: kilit bozulur, hiçbir şey kazanılmaz. `h ≠ 0` penceresi
    işarete VE çarpana bağlıdır (9.961e-02 FM'nin; doğru işaretle, çarpansız
    c = −J: 5.078e-02 / 8.809e-01; c = −J/4'te 6.250e-02 / 7.812e-01).
    """
    return run_metropolis(
        SpinState.uniform(L=L, n_component=1),
        IsingModel(),
        temperature=TC_BOLU_J * J,
        coupling=J,
        field=alan,
        rng=np.random.default_rng(TOHUM),
        n_sweep=N_SWEEP,
        n_burn=N_BURN,
    )


# =============================================================================
# hf -> tb
# =============================================================================


@pytest.mark.parametrize("n,d", SISTEM_PARAMLARI)
def test_HF_TB_t_m_GERI_OKUMASI_ALTI_SISTEMDE_TAM_ESIT_ve_bu_bir_TANIMDIR(n, d):
    """SINIF: TAUTOLOJİ BEYANI. Bir fizik iddiası TAŞIMAZ; tam tersini yazar.

    `downfold_uhf` `t_m`'yi halka kabuğu üzerinde `np.mean` olarak TANIMLAR
    (`downfold.py:259-271`). Bu test aynı kabuğu aynı sırada toplayıp
    `p.hoppings[m-1]` ile Python `==`'inde karşılaştırır: ALTI sistemde, HER
    kabukta True, |fark| = 0.0 — `eps_bar` ve `delta` da öyle.

    NEDEN BU BİR UYUM DEĞİL: aynı elemanlar TERS sırayla toplanınca `==`
    22 kabuğun 10'unda DÜŞÜYOR (fark en çok 1 ulp; ölçüldü 02.09.2026,
    numpy 2.5.1). Yani "on altı hanede aynı" cümlesi bir doğa gözlemi
    değil, toplama sırasının kaydıdır. Modül 3'ün silisyum kalıbının
    aynısı: tanımı geri okumak.

    UHF İKİ KEZ KOŞAR (`_effective_hamiltonians` + `downfold_uhf`) ve bu
    bilinçlidir: `downfold_uhf`un içine `H_eff` enjekte edilseydi geri
    okuma KURGUYLA tautoloji olurdu; burada iki bağımsız çağrının aynı
    çözüme indiği de ölçülüyor (seeds = 12 ile bit bit).
    """
    mol = _halka(n, d)
    h_a, _, _, _, _ = _effective_hamiltonians(mol, BAZ, SEEDS)
    p = downfold_uhf(mol, BAZ, n_neighbour=n // 2, seeds=SEEDS)
    assert len(p.hoppings) == n // 2

    dusen = 0
    for m in range(1, n // 2 + 1):
        hop = p.hoppings[m - 1]
        ayni_sira = float(np.mean(_kabuk(h_a, n, m)))
        assert ayni_sira == hop, f"N={n} d={d} m={m}: {ayni_sira!r} != {hop!r}"
        assert float.hex(ayni_sira) == float.hex(hop)
        ters_sira = float(np.mean(_kabuk(h_a, n, m, ters=True)))
        assert abs(ters_sira - hop) <= np.spacing(abs(hop)), (
            f"N={n} d={d} m={m}: ters sıra 1 ulp'tan fazla sapıyor"
        )
        dusen += int(ters_sira != hop)
    assert dusen == TERS_SIRA_DUSEN[(n, d)], (
        f"N={n} d={d}: ters sırada düşen kabuk {dusen}, kayıt {TERS_SIRA_DUSEN[(n, d)]}"
    )

    kosegen = np.diag(h_a)
    assert float(np.mean(kosegen)) == p.epsilon_bar
    assert float(np.mean(kosegen[0::2] - kosegen[1::2])) / 2.0 == p.delta


@pytest.mark.parametrize("n,d", SISTEM_PARAMLARI)
def test_HF_TB_OZDESLIK_ON_BES_HANE_ve_SISTEME_BAGLI_yedi_mertebe(n, d):
    """"ON ALTI HANE" İDDİASI DARALTILDI — VE PENCERESİZ TAŞINAMAZ.

    Tam menzilde `max|model − H_eff^α|`, en büyük ÇİFT kabuk yayılımının
    yarısıdır (model ortalamayı koyar; A-A ile B-B elemanları ortalamanın
    ± yarım yayılım uzağındadır). Spec §3.1 bunu (6, 2.6)'da "on altı hane"
    diye yazmıştı; ölçülen bağıl fark 2.276006188306662e-15 = ON BEŞ hane,
    ve sisteme bağlı:

        (6, 2.6)  2.276e-15      (8, 2.0)  1.090e-08
        (6, 3.0)  2.024e-15      (8, 2.6)  1.310e-08
        (6, 3.6)  4.729e-14      (10, 2.6) 3.769e-08

    N = 6'dan N = 8'e YEDİ MERTEBE (log10 = 6.68). Sebebi kabuk-içi
    gürültü: N = 6'da UHF artığı 2e-15, N ≥ 8'de 1e-9 (`solve_uhf`in
    artık toleransı 1e-8'dir), ve özdeşlik "A-A elemanlarının hepsi aynı"
    varsayımına o gürültü kadar dayanır.

    İKİ AYRI KİLİT: (1) her sistemin MERTEBESİ (N = 6 < 1e-13, N ≥ 8 > 1e-9)
    ve N ≥ 8'in referansa oranı > 1e6 — bunlar taşıyıcı iddia; (2) sayının
    kendisi `rel=1e-3` ile — bu ortamın gerileme kaydı (iki süreçte ve
    seeds = 12/16'da bit bit aynı çıktı; başka BLAS'ta aynı çıkıp
    çıkmayacağı ÖLÇÜLMEDİ). Referans pencerenin sayısı `bridge_map`in
    literaliyle ve gerekçe metniyle de karşılaştırılır.

    E6 (11.10.2026): o sayı tabloda `buyukluk` DEĞİLDİR (None) — K3
    `tdse → tb`ye nasıl uygulandıysa buraya da: bağıl artık bir TOPLAMA-SIRASI
    KAYDIDIR. Referans pencerede aynı kabuk elemanları ters ve 50 rastgele
    sırayla ortalanır; ayrı değerlerin sayısı >= 2 kilitlenir (değerlerin
    kendisi son-ulp kaydıdır, kilitlenmez).
    """
    mol = _halka(n, d)
    h_a, _, _, _, _ = _effective_hamiltonians(mol, BAZ, SEEDS)
    p = downfold_uhf(mol, BAZ, n_neighbour=n // 2, seeds=SEEDS)
    matris = finite_ring_matrix(model_from_downfold(p, spin="alpha"), n // 2)

    fark = np.abs(matris - h_a)
    artik = float(fark.max())
    i, j = np.unravel_index(int(np.argmax(fark)), fark.shape)
    assert _halka_mesafesi(int(i), int(j), n) % 2 == 0, "artık ÇİFT kabukta değil"

    yayilim = {}
    for m in range(1, n // 2 + 1):
        el = np.array(_kabuk(h_a, n, m))
        yayilim[m] = float(el.max() - el.min())
    yarim = max(v for m, v in yayilim.items() if m % 2 == 0) / 2.0
    bagil = abs(artik - yarim) / artik

    # (1) taşıyıcı: mertebe ve sisteme bağlılık
    if n == 6:
        assert bagil < 1e-13
    else:
        assert bagil > 1e-9
        assert bagil > 1e6 * _HF_TB_OZDESLIK_BAGIL, "yedi mertebe kayboldu"

    # (2) kayıt
    assert bagil == pytest.approx(OZDESLIK_BAGIL[(n, d)], rel=1e-3, abs=0.0)
    gerekce = _girdi("hf", "tb").gerekce
    if (n, d) == (6, 2.6):
        assert bagil == pytest.approx(_HF_TB_OZDESLIK_BAGIL, rel=1e-3, abs=0.0)
        # E6 (11.10.2026): sayi bir TOPLAMA-SIRASI KAYDIDIR -> tabloda buyukluk
        # None (K3, `tdse -> tb` ile ayni). Kanit canli: ayni kabuk elemanlari
        # baska sirayla ortalaninca ayni tanim BASKA bir sayi verir. Kilit
        # yalniz "en az iki ayri deger"dir; degerlerin kendisi son-ulp
        # kaydidir ve kilitlenmez (gerekcedeki 2.113e-15 / 3 deger bu makinenin).
        sira_sayilari = {bagil}
        rastgele = np.random.default_rng(0)
        for sirala in [lambda k: np.arange(k)[::-1]] + [rastgele.permutation] * 50:
            hop = []
            for m in range(1, n // 2 + 1):
                el = np.array(_kabuk(h_a, n, m))
                hop.append(float(np.mean(el[sirala(len(el))])))
            yeni = finite_ring_matrix(
                model_from_downfold(dataclasses.replace(p, hoppings=tuple(hop)), spin="alpha"), n // 2
            )
            artik_i = float(np.abs(yeni - h_a).max())
            sira_sayilari.add(abs(artik_i - yarim) / artik_i)
        assert len(sira_sayilari) >= 2, "bagil artik siradan BAGIMSIZ cikti -- E6 gerekcesi curudu"
        assert _girdi("hf", "tb").buyukluk is None
        assert "TOPLAMA-SIRASI KAYDIDIR" in gerekce and "E6" in gerekce
        assert repr(_HF_TB_OZDESLIK_BAGIL) in gerekce
        assert artik == pytest.approx(0.010670502034857853, rel=1e-9, abs=0.0)
        assert yarim == pytest.approx(0.010670502034857829, rel=1e-9, abs=0.0)
    else:
        assert f"{bagil:.3e}" in gerekce, f"gerekçe {bagil:.3e} sayısını taşımıyor"
    if (n, d) == (8, 2.0):
        assert artik == pytest.approx(0.025530603398030302, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("n,d", SISTEM_PARAMLARI)
def test_HF_TB_LOWDIN_ADIMI_KAYIPSIZ_ama_farki_SISTEME_BAGLI(n, d):
    """Löwdin adımı tam bir benzerlik dönüşümü: `eig(H_eff^α)` ile UHF alfa
    orbital enerjileri arasındaki fark her sistemde 1e-8'in altında.

    Ama "3.49e-10" sayısı da penceresiz taşınamaz: o, ALTI sistemin EN
    KÜÇÜĞÜdür ((6, 2.6)); en büyüğü (8, 2.6) 2.890665e-09 (8.28 kat), ve
    N ile MONOTON DEĞİL — (8, 2.0) 7.19e-10 iken (6, 3.0) 2.63e-09.
    Spec'in "UHF α enerjileriyle fark 3.49e-10" cümlesi yalnız bir sistemin
    cümlesidir. Bu test altısını da ölçer ve en küçük/en büyük iddiasını
    her sistemde iki literal arasında sıkıştırarak sınar.
    """
    mol = _halka(n, d)
    h_a, _, _, _, _ = _effective_hamiltonians(mol, BAZ, SEEDS)
    rep = search_uhf(mol, BAZ, seeds=SEEDS)
    hedef = np.sort(rep.result.orbital_energies_alpha)
    lowdin = float(np.abs(np.sort(np.linalg.eigvalsh(h_a)) - hedef).max())

    assert lowdin < 1e-8, "Löwdin adımı kayıpsız değil"
    assert lowdin == pytest.approx(LOWDIN_FARKI[(n, d)], rel=1e-3, abs=0.0)
    en_kucuk, en_buyuk = LOWDIN_FARKI[(6, 2.6)], LOWDIN_FARKI[(8, 2.6)]
    assert en_kucuk * (1 - 1e-3) <= lowdin <= en_buyuk * (1 + 1e-3), (
        f"N={n} d={d}: {lowdin:.6e} en küçük/en büyük aralığının dışında"
    )
    gerekce = _girdi("hf", "tb").gerekce
    if (n, d) in ((6, 2.6), (8, 2.6)):
        assert repr(LOWDIN_FARKI[(n, d)]) in gerekce
    if (n, d) == (8, 2.6):
        assert lowdin / en_kucuk == pytest.approx(8.28, rel=1e-3, abs=0.0)


def test_model_from_downfold_SRC_ICINDE_HIC_CAGRILMIYOR_ast_ile_ve_SATIRLAR_GERCEK():
    """BORU HATTINDA `hf → tb` AKIŞI YOK: `model_from_downfold` `src/`
    ağacında TANIMLANIR ama HİÇ ÇAĞRILMAZ; tüketicileri yalnız testlerdir.
    Olan şey `downfold.py`nin `tb/` klasöründe OTURMASIDIR.

    `ast` ile: `src/qlab/**/*.py` içinde `Call` düğümü 0, `Name` yükü 0,
    import 0 — ve tarayıcının kör olmadığı testlerde ölçülür (çağıran test
    dosyaları kümesi TAM OLARAK kesme testi + bu dosya). `bridge_map.py`
    adı yalnız bir DİZGE içinde anar; `ast` dizgeyi saymaz (Task 1'in
    dersi: regex sayardı).

    Gerekçenin yazdığı satır numaraları da gerçektir: tanım
    `downfold.py:287`, `t_m` ataması `downfold.py:259-271` — ikisi de
    gerekçe METNİNDEN ayrıştırılıp `ast`nin `lineno`suyla karşılaştırılır,
    yani gerekçe bayatlarsa bu test kırmızı olur.
    """
    def cagrilar(agac):
        return [
            d for d in ast.walk(agac)
            if isinstance(d, ast.Call)
            and (
                (isinstance(d.func, ast.Name) and d.func.id == "model_from_downfold")
                or (isinstance(d.func, ast.Attribute) and d.func.attr == "model_from_downfold")
            )
        ]

    def importlar(agac):
        return [
            a for d in ast.walk(agac) if isinstance(d, ast.ImportFrom)
            for a in d.names if a.name == "model_from_downfold"
        ]

    src_cagri, src_yuk, src_import, tanimlar = 0, 0, 0, []
    for yol in sorted(KAYNAK.rglob("*.py")):
        agac = ast.parse(yol.read_text(encoding="utf-8"), filename=str(yol))
        src_cagri += len(cagrilar(agac))
        src_yuk += len(_yukler(agac, "model_from_downfold"))
        src_import += len(importlar(agac))
        tanimlar += [
            (yol, d) for d in ast.walk(agac)
            if isinstance(d, ast.FunctionDef) and d.name == "model_from_downfold"
        ]
    assert src_cagri == 0, f"src/ içinde {src_cagri} çağrı var"
    assert src_yuk == 0 and src_import == 0
    assert len(tanimlar) == 1
    tanim_yolu, tanim = tanimlar[0]
    assert tanim_yolu == KAYNAK / "engine" / "tb" / "downfold.py"

    cagiranlar = set()
    for yol in sorted(TESTLER.rglob("*.py")):
        agac = ast.parse(yol.read_text(encoding="utf-8"), filename=str(yol))
        if cagrilar(agac):
            cagiranlar.add(yol.relative_to(PROJE_KOKU).as_posix())
    assert cagiranlar == {
        "tests/tb/test_downfold_kesme.py",
        "tests/bridge/test_dusen_adaylar.py",
    }, f"tarayıcı kör değil ama küme değişmiş: {cagiranlar}"

    gerekce = _girdi("hf", "tb").gerekce
    for ad in cagiranlar:
        assert ad in gerekce
    assert int(re.search(r"tanim `downfold\.py:(\d+)`", gerekce).group(1)) == tanim.lineno

    bas, son = map(int, re.search(r"`downfold\.py:(\d+)-(\d+)`\)", gerekce).groups())
    agac = ast.parse(Path(downfold_mod.__file__).read_text(encoding="utf-8"))
    atama = _atama(_fonksiyon(agac, "downfold_uhf"), "hoppings")
    assert (atama.lineno, atama.end_lineno) == (bas, son)
    assert any(
        isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "mean"
        for d in ast.walk(atama.value)
    ), "t_m artık np.mean ile tanımlanmıyor — gerekçe bayat"


# =============================================================================
# tb -> spin
# =============================================================================


def test_TB_SPIN_KAYNAK_SAYILARI_U_bolu_t_2_033100_t1_Delta_ve_bes_J():
    """`U/|t| = 2.033100`, `t1 = −0.407144651592`, `Δ = +0.193177917796`
    — üçü de N = 8, d = 2.0, STO-3G, seeds = 12 halkasından CANLI üretilir.

    `U` boşluk denkleminden gelir (`downfold.py` dokstringi):
        1 = (U/N) Σ_k 1/(2 √(ε_k² + Δ²)),   ε_k = 2 t1 cos k
    → U = 0.827765644, U/|t| = 2.033100. Güçlü bağlaşım şartı (U ≫ |t|)
    İHLAL — ve bu, gerekçede AYRI SATIR DEĞİL, tek satırın içindedir.

    Planın beş `J`si: ilk dördü canlı `t1`den (U_gap, Löwdin (00|00)
    integrali 0.883907411 — `downfold.py` dokstringinin sevk edilmiş sayısı
    —, 4|t|, 8|t|) 1e-8 bağıl hassasiyetle yeniden üretiliyor. BEŞİNCİSİNİN
    `U`su PLANDA YAZMIYOR; geri hesaplanınca U/|t| = 13.630622, gerekçe
    öyle yazar. `J` aralığı 6.70 kat.
    """
    p = downfold_uhf(_halka(8, 2.0), BAZ, seeds=SEEDS)
    t1, delta = p.hoppings[0], p.delta
    assert t1 == pytest.approx(-0.407144651592, rel=1e-11, abs=0.0)
    assert delta == pytest.approx(+0.193177917796, rel=1e-11, abs=0.0)

    k = 2 * np.pi * np.arange(8) / 8
    eps = 2 * t1 * np.cos(k)
    u_gap = 1.0 / float(np.mean(1.0 / (2.0 * np.sqrt(eps**2 + delta**2))))
    assert u_gap == pytest.approx(0.827765644, rel=1e-8, abs=0.0)
    assert u_gap / abs(t1) == pytest.approx(U_BOLU_T, rel=1e-9, abs=0.0)
    assert f"{u_gap / abs(t1):.6f}" == "2.033100"

    assert "0.883907411" in downfold_mod.__doc__, "U_int kaynağı dokstringden silinmiş"
    u_int = 0.883907411
    for u, J in ((u_gap, BES_J[0]), (u_int, BES_J[1]), (4 * abs(t1), BES_J[2]), (8 * abs(t1), BES_J[3])):
        assert 4 * t1**2 / u == pytest.approx(J, rel=1e-8, abs=0.0)
    u_bes = 4 * t1**2 / BES_J[4]
    assert u_bes / abs(t1) == pytest.approx(13.630622, rel=1e-6, abs=0.0)
    assert BES_J[0] / BES_J[4] == pytest.approx(6.704355, rel=1e-5, abs=0.0)

    gerekce = _girdi("tb", "spin").gerekce
    for parca in (
        "U/|t| = 2.033100", "t1 = -0.407144651592", "Delta = +0.193177917796",
        "U = 0.827765644", "13.630622", "6.70 kat",
        "0.801032362 / 0.750154440 / 0.407144652 / 0.203572326 / 0.119479403",
    ):
        assert parca in gerekce, f"gerekçe {parca!r} taşımıyor"


def test_TB_SPIN_BES_J_BIT_BIT_OLU_ve_E_esittir_J_carpi_E1():
    """`J` 6.70 KAT DEĞİŞTİ, ÇIKTI `0.000e+00` KIPIRDADI.

    Beş `J`, her biri kendi `T = 2.269185 · J`siyle, aynı tohum. Beşinde de:
      * mıknatıslanma serisi `J = 1`inkiyle BİT BİT aynı (`np.array_equal`)
      * `E == J · E(1)` BİT BİT — bir yaklaşıklık değil, aritmetik:
        `total_energy` `(-0.5·J)·S`i tek yuvarlamayla kurar (S tamsayı
        değerli), `J·(-0.5·S)` de aynı gerçek sayının tek yuvarlamasıdır.
        `E/J` ise İKİNCİ bir yuvarlama alır: max|E/J − E(1)| = 2⁻⁴²
        (|E| ≈ 1483 için tam 1 ulp). Spec'in 5.684e-14'ü (= 2⁻⁴⁴) bu
        sayının L = 16 penceresindeki hâlidir — orada da bit bit değildi.
      * kabul oranı `==` (tamsayı oranı: 287565/1536000 = 0.187216796875).

    MEKANİZMA FLOAT DÜZEYİNDE DE ÖLÇÜLDÜ: Ising'de `delta = -J·s`,
    s ∈ {0, ±4, ±8} (ikinin kuvvetleri → çarpım TAM), üstel argümanı
    `(-J·s)/fl(2.269185·J)`. Bu, `(-s)/2.269185` ile 20 çiftin 12'sinde
    bit bit aynı, 8'inde TAM 1 ulp farklı (J = 0.801032362 ve 0.750154440,
    dört s'de). DÜZELTİLDİ (27.09.2026, hf-spin eleştirmeni E1/E1b; eski
    metin "2⁻⁵², 1 536 000 karar, 3.4e-10" idi — burn-in sayılmamıştı ve
    olasılık iki kat yazılmıştı): `rng.random()` 2⁻⁵³ ızgarasında, karar
    eşiği `ceil(üstel·2⁵³)`. Burn-in DAHİL 2 048 000 aktif karar, bunların
    1 346 053'ü y = −8 sınıfında; eşik YALNIZ o sınıfta ve YALNIZ o iki J'de
    1 birim farklı. Karar başına dönüş olasılığı tam 2⁻⁵³, beklenen dönüş
    2/5 J'de 1 346 053 · 2⁻⁵³ = 1.494e-10, öteki 3/5 J'de TAM 0 — ve
    ölçülen 0. "Bit bit" bir ÖLÇÜM sonucudur; oranın sadeleşmesi ise
    cebirsel bir KESİNLİKtir. Bu sayılar KOŞTURULUR ve dokstringe bağlıdır — ÖZEL YOL:
    yarı-adım REPLİKASI (seri motora `==`); eşik tablosu bu makinenin `np.exp`ine bağlı.

    `bridge_map`in `buyukluk = 0.0`ı burada canlı ölçümle eşitlenir.
    """
    ref = _kos(1.0)
    assert ref.acceptance_rate == 0.187216796875
    assert round(ref.acceptance_rate * N_SWEEP * L * L) == 287565
    assert float(ref.magnetisation_series.mean()) == pytest.approx(0.687930, rel=1e-5, abs=0.0)

    m_farki_max = 0.0
    e_bolu_j_max = 0.0
    for J in BES_J:
        r = _kos(J)
        assert np.array_equal(r.magnetisation_series, ref.magnetisation_series), J
        assert np.array_equal(r.energy_series, J * ref.energy_series), J
        assert r.acceptance_rate == ref.acceptance_rate
        m_farki_max = max(m_farki_max, float(np.abs(r.magnetisation_series - ref.magnetisation_series).max()))
        e_bolu_j_max = max(e_bolu_j_max, float(np.abs(r.energy_series / J - ref.energy_series).max()))
    assert m_farki_max == 0.0
    assert m_farki_max == _TB_SPIN_M_FARKI == _girdi("tb", "spin").buyukluk
    assert 0.0 < e_bolu_j_max <= 2.0**-42, "E/J ikinci yuvarlamayı kaybetti ya da büyüttü"

    # float düzeyinde mekanizma: üstel argümanı en çok 1 ulp, 20'de 8 eşitsiz
    esitsiz = 0
    for J in BES_J:
        for s in (-8.0, -4.0, 4.0, 8.0):
            a = (-J * s) / (TC_BOLU_J * J)
            b = (-1.0 * s) / TC_BOLU_J
            assert abs(a - b) <= np.spacing(abs(b))
            esitsiz += int(a != b)
    assert esitsiz == 8

    # Karar sayımı ve eşik sınıfı (ÖZEL YOL, dokstring): replika seri motora `==`.
    rng, s, m, y8 = np.random.default_rng(TOHUM), SpinState.uniform(L=L, n_component=1).spins, [], 0
    for _ in range(N_BURN + N_SWEEP):
        for maske in (ETA[..., 0] > 0, ETA[..., 0] < 0):
            y = np.sum(-2.0 * s * neighbour_sum(s), axis=2)
            y8 += int(np.count_nonzero(y[maske] == -8.0))
            s = np.where(((rng.random((L, L)) < np.exp(y / TC_BOLU_J)) & maske)[..., None], -s, s)
        m.append(abs(float(s.mean())))
    assert np.array_equal(np.array(m[N_BURN:]), ref.magnetisation_series), "replika ayrıştı"
    assert ((N_BURN + N_SWEEP) * L * L, y8, f"{y8 * 2.0**-53:.3e}") == (2_048_000, 1_346_053, "1.494e-10")
    esik = {(J, y) for J in BES_J for y in (-8.0, -4.0)  # dizi np.exp: motorun yolu
            if np.ceil(np.exp(np.full((L, L), J * y) / (TC_BOLU_J * J)) * 2.0**53)[0, 0]
            != np.ceil(np.exp(np.full((L, L), y) / TC_BOLU_J) * 2.0**53)[0, 0]}
    assert esik == {(0.801032362, -8.0), (0.750154440, -8.0)}
    assert all(p in test_TB_SPIN_BES_J_BIT_BIT_OLU_ve_E_esittir_J_carpi_E1.__doc__
               for p in ("2 048 000 aktif karar", "1 346 053'ü y = −8", "1.494e-10")), "dokstring bayat"

    gerekce = _girdi("tb", "spin").gerekce
    for parca in (
        repr(ref.acceptance_rate), "287565/1536000", "L = 32", "tohum 20260822",
        "n_burn = 500", "n_sweep = 1500", "T = 2.269185 * J", "max|m farki| = 0.0",
        "E == J * E(J=1)",
    ):
        assert parca in gerekce, f"gerekçe {parca!r} taşımıyor"


def test_TB_SPIN_KOK_NEDEN_ORAN_SADELESMESI_ast_ile_KILITLI():
    """KÖK NEDEN `metropolis.py:189-191` — `coupling` çıktıya YALNIZ
    `coupling/temperature` oranı olarak girer. `ast` ile:

      `metropolis_sweep` içinde, `_dogrula(...)` çağrısı DIŞINDA (o yalnız
      doğrular; 27.09.2026'dan beri `coupling`/`field`in skalerliğini de):
        * `coupling` TAM BİR kez yüklenir, o da `delta = ...` atamasında
        * `temperature` TAM BİR kez,
          `ustel` atamasındaki TEK bölmenin paydası olarak, payı `delta`
        * `field` TAM BİR kez, yine `delta` atamasında (h = 0'da 0 · x)
        * satırlar (`delta`, `ustel`, bölme) ardışıktır ve gerekçedeki
          `metropolis.py:A-B` atfından AYRIŞTIRILIR; numara burada BİLEREK yazılmaz
      `_dogrula` gövdesi izin listesiyle: `state` yalnız `.L`/`.n_component`, `coupling`/`field`
      yalnız skalerlik döngüsünde, `temperature` yalnız sonluluk/işarette — yan etkiyle J SOKULAMAZ.
      `run_metropolis` içinde `coupling` (`_dogrula` dışında) iki kez: sweep'e
      ve `total_energy`ye geçirilir, başka hiçbir yerde okunmaz.
      `lattice.total_energy` içinde `coupling` TAM BİR kez, çarpan olarak.

    Yani `(J, T) → (λJ, λT)` altında kabul olasılığı `exp(-delta/T)`
    değişmez, `E` λ ile ölçeklenir, `m` `J`'yi hiç görmez. Kodda başka
    bir enerji ölçeği YOKTUR; olsaydı bu sayımlar değişirdi.
    """
    agac = ast.parse(Path(metropolis_mod.__file__).read_text(encoding="utf-8"))
    sweep = _fonksiyon(agac, "metropolis_sweep")
    delta = _atama(sweep, "delta")
    ustel = _atama(sweep, "ustel")

    def _dogrula_disi(govde, ad):
        args = [n for d in ast.walk(govde) if isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                and d.func.id == "_dogrula" for n in ast.walk(d)]
        return [n for n in _yukler(govde, ad) if n not in args]

    coupling = _dogrula_disi(sweep, "coupling")
    assert len(coupling) == 1 and coupling[0] in list(ast.walk(delta.value))
    field = _dogrula_disi(sweep, "field")
    assert len(field) == 1 and field[0] in list(ast.walk(delta.value))
    temperature = _dogrula_disi(sweep, "temperature")
    assert len(temperature) == 1
    bolmeler = [
        d for d in ast.walk(ustel.value) if isinstance(d, ast.BinOp) and isinstance(d.op, ast.Div)
    ]
    assert len(bolmeler) == 1
    bolme = bolmeler[0]
    assert isinstance(bolme.left, ast.Name) and bolme.left.id == "delta"
    assert bolme.right is temperature[0]
    assert not any(isinstance(d, ast.Name) and d.id == "coupling" for d in ast.walk(ustel.value))

    gerekce = _girdi("tb", "spin").gerekce
    bas, son = map(int, re.search(r"`engine/spin/metropolis\.py:(\d+)-(\d+)`", gerekce).groups())
    assert (delta.lineno, ustel.lineno, bolme.lineno) == (bas, bas + 1, son)

    kosum = _fonksiyon(agac, "run_metropolis")
    assert len(_dogrula_disi(kosum, "coupling")) == 2

    dog = _fonksiyon(agac, "_dogrula")
    ebeveyn = {c: p for p in ast.walk(dog) for c in ast.iter_child_nodes(p)}
    dongu = next(d for d in ast.walk(dog) if isinstance(d, ast.For))
    assert not any(isinstance(d, (ast.Global, ast.Nonlocal)) for d in ast.walk(dog))
    assert all(getattr(ebeveyn[n], "attr", None) in ("L", "n_component") for n in _yukler(dog, "state"))
    for ad in ("coupling", "field"):
        assert [n in list(ast.walk(dongu.iter)) for n in _yukler(dog, ad)] == [True], ad
    assert all(ast.unparse(ebeveyn[n]) in ("np.ndim(deger)", "np.shape(deger)") for n in _yukler(dog, "deger"))
    assert all(isinstance(ebeveyn[n], (ast.Compare, ast.FormattedValue)) or ast.unparse(ebeveyn[n])
               == "np.isfinite(temperature)" for n in _yukler(dog, "temperature"))

    kafes = ast.parse(Path(lattice_mod.__file__).read_text(encoding="utf-8"))
    enerji = _fonksiyon(kafes, "total_energy")
    etkilesim = _atama(enerji, "etkilesim")
    c = _yukler(enerji, "coupling")
    assert len(c) == 1 and c[0] in list(ast.walk(etkilesim.value))
    assert any(
        isinstance(d, ast.BinOp) and isinstance(d.op, ast.Mult) and c[0] in (d.left, d.right)
        for d in ast.walk(etkilesim.value)
    )


def test_TB_SPIN_U_BUYUTULSE_DE_z_UYUSTURULSA_DA_KENAR_ACILMIYOR_ve_KONTROL():
    """AYRI SATIR OLMAZ — çünkü ayrı satır "bunu düzeltirsen kenar açılır"
    diye okunur ve AÇILMIYOR. Bu test o cümleyi ölçer.

    (a) `U`'yu büyütmek: U/|t| = 2.0331 × (2, 4, 8, 50, 500) → 4.07 / 8.13
        / 16.26 / 101.65 / 1016.55. Güçlü bağlaşım şartı ne kadar sağlanırsa
        sağlansın çıktı `J = 1`inkiyle BİT BİT aynı. Engel `U/|t|` değil.
    (b) `z`'yi uyuşturmak: spin motoru kodda 2B (`neighbour_sum` DÖRT
        `np.roll`, tekbiçim durumda komşu toplamı 4.0), TB halkası 1B
        (`model_from_downfold`un site matrisinde her satırda TAM İKİ komşu).
        `J`'yi `z_tb/z_spin = 2/4` ile ölçeklemek de bir ölçekleme, o da
        sadeleşiyor: BİT BİT aynı. Engel `z` de değil.
    (c) KONTROL — motor `J`'ye gerçekten kör mü, yoksa test mi kör?
        `h = 0.1` sabitken aynı iki `J` 9.961e-02 ayrışıyor: ikinci bir
        enerji ölçeği doğar doğmaz `J` görünür oluyor. TB o ölçeği (`h`)
        ÜRETMEZ; köprü `J`'den ibarettir. Bu yüzden hüküm "BU BORUDA
        ÖLÇÜLEMEZ"dir, "spin motoru bozuk" değil.
    """
    ref = _kos(1.0)
    gerekce = _girdi("tb", "spin").gerekce

    # (a)
    for kat in (2, 4, 8, 50, 500):
        J = BES_J[0] / kat
        r = _kos(J)
        assert np.array_equal(r.magnetisation_series, ref.magnetisation_series), kat
        assert np.array_equal(r.energy_series, J * ref.energy_series), kat
        assert r.acceptance_rate == ref.acceptance_rate
        assert f"{U_BOLU_T * kat:.2f}" in gerekce, f"gerekçe U/|t| = {U_BOLU_T * kat:.2f} taşımıyor"

    # (b)
    assert float(neighbour_sum(SpinState.uniform(L=4, n_component=1).spins)[0, 0, 0]) == 4.0
    kafes = ast.parse(Path(lattice_mod.__file__).read_text(encoding="utf-8"))
    roll = [
        d for d in ast.walk(_fonksiyon(kafes, "neighbour_sum"))
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "roll"
    ]
    assert len(roll) == 4
    elle = DownfoldedParameters(
        epsilon_bar=0.0, delta=0.0, hoppings=(-0.3,), n_site=8, basis_name=BAZ,
        s_squared=0.0, energy_uhf=0.0, staggering_residual=0.0, search_line="elle",
    )
    site = finite_ring_matrix(model_from_downfold(elle), 8 // 2)  # 4 hücre x 2 site
    assert site.shape == (8, 8)
    assert all(int(np.count_nonzero(site[i])) == 2 for i in range(8)), "TB halkası z = 2 değil"
    r = _kos(BES_J[0] * 2 / 4)
    assert np.array_equal(r.magnetisation_series, ref.magnetisation_series)
    for parca in ("z = 2", "z = 4", "dort `np.roll`", "J * 2/4", "BU BORUDA OLCULEMEZ", "ORAN SADELESMESIDIR"):
        assert parca in gerekce, f"gerekçe {parca!r} taşımıyor"

    # (c)
    a = _kos(1.0, alan=0.1)
    b = _kos(BES_J[0], alan=0.1)
    fark = float(np.abs(a.magnetisation_series - b.magnetisation_series).max())
    assert fark > 1e-2, "h ≠ 0'da bile J görünmüyor — motor değil test kör"
    assert fark == pytest.approx(9.961e-02, rel=1e-3, abs=0.0)
    assert f"h = 0.1'de ayni iki J'nin ciktisi {fark:.3e} ayrisiyor" in gerekce


# =============================================================================
# hf -> spin (27.09.2026) — üçüncü düşen aday, KENARSIZ
# =============================================================================

HF_R = (5.0, 6.0, 7.0)
#: `search_uhf(seeds = 12)` varsayılan eşikler, STO-3G, `%.10g` + `unit="bohr"`
#: (Ångström gidiş-dönüşü R = 5 − 1 ulp üretir, J bağıl 1.22e-9 kayar). 17
#: hane bu pencerenin kaydıdır; fiziksel olarak ~7 anlamlı hane.
HF_J = (-0.001308824217584225, -0.00017495439344641936, -1.9411358087445856e-05)
#: İZLEMESİZ `seeds = 1` (`follow_instability=False`) kırık simetriyi bulamaz;
#: formül SESSİZCE +J (FM) verir. B7 (09.10.2026) öncesinin varsayılanıydı.
HF_J_SEEDS1 = (0.12292785673387831, 0.14398775927612392, 0.15520099760670322)
#: İZLEMELİ (varsayılan) `seeds = 1`in HF_J'den bağıl farkı -- ÖLÇÜLDÜ 09.10.2026
#: 4.71e-10 / 1.02e-9 / 2.04e-11. J1 izlemenin açı ızgarasına bağlı, `==` YAZILMAZ.
HF_J1_IZLEMELI_BAGIL = 1e-8
#: `tdse -> spin` (28.09.2026): TDSE TARAFI hizli seritte KOSMAZ (R = 3: I6 slow testi) -- LITERAL J'ler (1B
#: soft-Coulomb H2, `Grid2D` sanal zaman, n = 256, L = 40, dt = 0.01, a = b = 1, R = 3/4/5/6).
TDSE_J = (-0.03679538951052319, -0.010055270069717404, -0.0021743343691629224, -0.00041083190968227434)
#: "S2_BS > 0" LİTERAL yazılamaz: izlemesiz seeds = 1'in RHF-benzeri çözümü
#: 2.2e-16 / 2.2e-16 / 1.1e-15 taşıyor ve `> 0`'ı GEÇİYOR. Kapı sayısal sıfırın
#: üstünde -- ve ŞART KALIR: izlemesiz yol ve tek `solve_uhf` o sıfırı hâlâ taşır.
S2_BS_KAPISI = 1e-8
ETA = np.where(np.indices((L, L)).sum(axis=0) % 2 == 0, 1.0, -1.0)[..., None]
NEEL = ETA * SpinState.uniform(L=L, n_component=1).spins


def _yamaguchi(R: float, seeds: int, *, follow: bool = True) -> tuple[float, float]:
    """J = (E_BS − E_T)/(S2_T − S2_BS) — formül `src/`de YOK, burada elle."""
    mol = Molecule.from_string("H 0 0 0; H 0 0 %.10g" % R, unit="bohr")
    bs = search_uhf(mol, BAZ, seeds=seeds, follow_instability=follow)
    uc = search_uhf(mol, BAZ, seeds=SEEDS, n_alpha=2)
    s2 = bs.result.s_squared
    return (bs.energy - uc.energy) / (uc.result.s_squared - s2), s2


def _bs_kapisi(s2_bs: float) -> bool:
    return s2_bs > S2_BS_KAPISI


def _tam(J: float, s0: np.ndarray, *, alan: float = 0.0):
    """Burn-in DAHİL 2000 sweep'in hepsi ölçülür (`n_burn = 0`): özdeşlik
    sweep 1'den sınanır. Burn-in'den sonra bakmak yetmez — ortak rastgele
    sayılarla zincirler birleşir ve DÖNÜŞTÜRÜLMEMİŞ başlangıç da aynı seriyi
    verir (ölçüldü 27.09.2026: o mutant 500/1500 penceresinde HAYATTA kaldı)."""
    return run_metropolis(SpinState(s0), IsingModel(), temperature=TC_BOLU_J * abs(J), coupling=J,
                          field=alan, rng=np.random.default_rng(TOHUM), n_sweep=N_BURN + N_SWEEP, n_burn=0)


def _zincir(J: float, s0: np.ndarray, *, alan: float = 0.0):
    """`run_metropolis(n_sweep=1, n_burn=0)` ZİNCİRİ (aynı rng) — sweep başına
    durum için public yol. Dönüş: (tekbiçim, kademeli) seri, burn-in DAHİL;
    tekbiçim seri TEK çağrıyla bit bit karşılaştırılır (aracın kendisi sınanır)."""
    rng, st, uni, kad = np.random.default_rng(TOHUM), SpinState(s0), [], []
    for _ in range(N_BURN + N_SWEEP):
        r = run_metropolis(st, IsingModel(), temperature=TC_BOLU_J * abs(J), coupling=J,
                           field=alan, rng=rng, n_sweep=1, n_burn=0)
        st = r.final_state
        uni.append(r.magnetisation_series[0])
        kad.append(float(np.linalg.norm(magnetisation(SpinState(ETA * st.spins)))))
    assert np.array_equal(np.array(uni), _tam(J, s0, alan=alan).magnetisation_series), "zincir ayrıştı"
    return np.array(uni), np.array(kad)


def test_HF_SPIN_J_UC_R_DE_BIT_BIT_BS_KAPISI_ve_seeds1_ISARETI_SESSIZCE_CEVIRIR():
    """İŞARET ŞARTLI. `seeds = 12`de üç J literal ile `==` ve NEGATİF (AFM);
    İZLEMESİZ `seeds = 1` BS'yi bulamaz, S2_BS sayısal sıfırdır ama `> 0`'ı
    GEÇER ve formül +J verir. Kapı `S2_BS > 1e-8` o çözümü reddeder. KIRMA: kapı
    eşiği 0.0 yapılınca bu test kırmızı (seeds = 1 kapıdan geçer); eşik
    gerekçedeki sayıdan AYRIŞTIRILIR, yani ikisi ayrı kayamaz (1e-10 / 0.5 düşer).
    B7 (09.10.2026): izlemesiz yol bit bit eskisi -- J1 literal ile `==`."""
    gerekce = _girdi("hf", "spin").gerekce
    for R, J_lit, J1_lit in zip(HF_R, HF_J, HF_J_SEEDS1):
        J, s2 = _yamaguchi(R, SEEDS)
        assert _bs_kapisi(s2) and 0.99 < s2 < 1.0, (R, s2)
        assert J == J_lit and J < 0.0, (R, J)
        assert repr(J_lit) in gerekce
        J1, s21 = _yamaguchi(R, 1, follow=False)
        assert 0.0 < s21 < 1e-14, "seeds = 1 artık '> 0'ı geçmiyor — kapı gerekçesi bayat"
        assert not _bs_kapisi(s21), f"R={R}: kırık simetrisiz çözüm kapıdan geçti"
        assert J1 > 0.0 and J1 == J1_lit, (R, J1)
        assert f"+{J1_lit:.4f}" in gerekce
    for parca in ("seeds = 12", "%.10g", 'unit="bohr"', "~7 anlamli hane"):
        assert parca in gerekce, parca
    esik = re.search(r"S2_BS > (\S+) olmali", gerekce)
    assert esik and float(esik.group(1)) == S2_BS_KAPISI, "gerekçe eşiği != test kapısı"


def test_HF_SPIN_seeds1_IZLEMEYLE_KIRIK_SIMETRIYI_BULUYOR_J_NEGATIF():
    """B7 (09.10.2026): varsayılan kararlılık izlemesiyle `seeds = 1` RHF eyerinden
    BS'ye iniyor (her R'de 1 iniş). ÖLÇÜLDÜ: S2_BS 0.9945903905 / 0.9992972155 /
    0.9999246042 (kapıdan GEÇER), J1 = -0.0013088242182004426 /
    -0.00017495439362406981 / -1.9411358087049468e-05, `seeds = 12` literalinden
    bağıl 4.71e-10 / 1.02e-9 / 2.04e-11. `==` KULLANILMAZ: J1 izlemenin açı
    ızgarasına ve aday sırasına bağlı (eleştirmenin bağımsız izleme kopyası, e04:
    ön ölçümün uygulamasından 3.4e-13 ... 2.3e-11 farklı J1 verdi)."""
    gerekce = _girdi("hf", "spin").gerekce
    for R, J_lit in zip(HF_R, HF_J):
        J1, s21 = _yamaguchi(R, 1)
        assert _bs_kapisi(s21) and 0.99 < s21 < 1.0, (R, s21)
        assert J1 < 0.0, (R, J1)
        assert abs(J1 - J_lit) / abs(J_lit) < HF_J1_IZLEMELI_BAGIL, (R, J1)
    assert "izlemeli (varsayilan) `search_uhf(seeds = 1)`" in gerekce


def test_HF_SPIN_BUYUKLUK_PROTOKOL_A_BIT_BIT_SADELESIR():
    """Üç HF J'si, T = 2.269185·|J|, Néel başlangıç: `m` serisi J = −1'inkiyle
    BİT BİT, E == |J|·E(−1) bit bit, kabul oranı `==`. `buyukluk = 0.0` burada
    canlı ölçümle eşitlenir. R = 6/7'de karar eşiği 1 birim farklı (eleştirmen
    E1c) — eşitlik bir ÖLÇÜM sonucudur. 28.09.2026: dört TDSE J'si (`tdse → spin`)
    AYNI pencerede LITERAL parametre olarak koşar — TDSE tarafı bu testte KOŞMAZ."""
    ref = _tam(-1.0, NEEL)
    fark = 0.0
    for J in HF_J + TDSE_J:
        r = _tam(J, NEEL)
        assert np.array_equal(r.magnetisation_series, ref.magnetisation_series), J
        assert np.array_equal(r.energy_series, abs(J) * ref.energy_series), J
        assert r.acceptance_rate == ref.acceptance_rate
        fark = max(fark, float(np.abs(r.magnetisation_series - ref.magnetisation_series).max()))
    tdse = _girdi("tdse", "spin")
    assert fark == 0.0 == _HF_SPIN_M_FARKI == _girdi("hf", "spin").buyukluk == _TDSE_SPIN_M_FARKI == tdse.buyukluk
    assert isinstance(_girdi("hf", "spin").buyukluk, float) and isinstance(tdse.buyukluk, float)
    assert _girdi("hf", "spin").kova == tdse.kova == Kova.TAUTOLOJI
    assert all(repr(J) in tdse.gerekce for J in TDSE_J) and "`hf -> spin` penceresinde canli" in tdse.gerekce
    # 22.09 P8: kök neden TEK sabit, üç spin girdisi onu okur, kaynakta tek kopya.
    assert {f.cift for f in KOVA_TABLOSU if _SPIN_BUYUKLUGU_SILER in f.gerekce} == {
        ("spin", "tb"), ("hf", "spin"), ("spin", "tdse")}
    assert (KAYNAK / "bridge_map.py").read_text(encoding="utf-8").count("ORANI olarak giriyor") == 1
    assert "J'nin BUYUKLUGUNU SILER. Bit bit esitlik bir OLCUM sonucudur, ozdeslik degil" in _SPIN_BUYUKLUGU_SILER
    gerekce = _girdi("hf", "spin").gerekce  # sayılar YALNIZ bu pencereyle taşınır
    for parca in (f"T = {TC_BOLU_J} * |J|", f"Ising L = {L}, h = 0", f"tohum {TOHUM}",
                  f"{N_BURN + N_SWEEP} sweep burn-in DAHIL", "Neel baslangic"):
        assert parca in gerekce, f"gerekçe {parca!r} taşımıyor"


def test_TDSE_SPIN_PROTOKOL_B_300K_R3_R5_DONAR_R6_J1_IMGESI(tmp_path):
    """Protokol B (T = 300 K mutlak), sevk edilmiş `spin_tarama.yaml` penceresi, PUBLIC YAML yolu
    (`load_scenario` -> `build_spin`); yükleyici TDSE'nin gerçek (negatif) J'sini REDDEDER, koşum
    coupling = |J| (FM imgesi). R = 3/4/5'te T/|J| < 0.5: |m| serisi tamamen 1.0, kabul 0.0, J = 1 /
    T = 1e-3 koşumuyla (TDSE içeriği SIFIR) bit bit aynı — zincir donar ya da doyar, pencere ayırmaz.
    R = 6 kıpırdar ama aynı T/|J|'de J = 1 koşumunun bit bit etiketidir. KIRMA: 3000 K'de R = 5 çözülür."""
    sevk = (PROJE_KOKU / "scenarios" / "spin_tarama.yaml").read_text(encoding="utf-8")
    T300 = 300.0 * KB_HARTREE_PER_KELVIN

    def kos(J: float, T: float):
        metin = re.sub(r"(?m)^temperatures: .*$", f"temperatures: [{T!r}]", sevk)
        (yol := tmp_path / "b.yaml").write_text(re.sub(r"(?m)^coupling: .*$", f"coupling: {J!r}", metin),
                                                encoding="utf-8")
        s = build_spin(load_scenario(yol)).samples[0]
        return s.magnetisation_series, s.acceptance_rate

    with pytest.raises(ScenarioError, match=r"'coupling' \(J\) sonlu ve pozitif"):
        kos(TDSE_J[0], T300)
    donuk, kabul0 = kos(1.0, 1e-3)
    assert np.all(donuk == 1.0) and kabul0 == 0.0
    oran = [T300 / abs(J) for J in TDSE_J]
    for J, r in zip(TDSE_J[:3], oran):
        m, kabul = kos(abs(J), T300)
        assert r < 0.5 and np.array_equal(m, donuk) and kabul == 0.0, J
    m6, k6 = kos(abs(TDSE_J[3]), T300)
    m1, k1 = kos(1.0, oran[3])
    assert np.array_equal(m6, m1) and k6 == k1 > 0.0 and not np.array_equal(m6, donuk)
    gerekce = _girdi("tdse", "spin").gerekce
    assert f"300 K = {T300:.3e} Ha" in gerekce
    assert "R = 3/4/5'te T/|J| = " + " / ".join(f"{r:.3f}" for r in oran[:3]) + " < 0.5" in gerekce
    assert f"R = 6'da T/|J| = {oran[3]:.3f}, <|m|> = {m6.mean():.3f}" in gerekce
    for parca in ("J(R = 3/4/5/6 bohr)", "J <= 0'i ScenarioError", "coupling = |J|", "sevk listesinde yok, 0/4"):
        assert parca in gerekce, parca


def test_HF_SPIN_ISARET_AYAR_ETIKETI_AFM_KADEMELI_eq_FM_TEKBICIM_ve_h01_BOZAR():
    """h = 0 çift L: AFM(−1) Néel başlangıçlı koşumun KADEMELİ serisi, sevk
    edilmiş FM(+1) koşumunun (`_kos`) TEKBİÇİM serisiyle BİT BİT aynı —
    tekbiçim çıktı işaretle değişir (0.009724) ama değişim FM koşumunun
    ayar-imgesidir. `tb → spin`in kilitli 0.687930'u bu yüzden doğru işaretli
    AFM'nin kademeli ortalamasıdır. KONTROL: h = 0.1'de (iki tarafta aynı
    tekbiçim alan) özdeşlik BOZULUR — test kör değil."""
    uni = SpinState.uniform(L=L, n_component=1).spins
    fm = _tam(1.0, uni)
    afm_uni, afm_kad = _zincir(-1.0, NEEL)
    assert np.array_equal(afm_kad, fm.magnetisation_series)  # sweep 1'den
    assert np.array_equal(fm.magnetisation_series[N_BURN:], _kos(1.0).magnetisation_series)
    assert float(afm_kad[N_BURN:].mean()) == pytest.approx(0.687930, rel=1e-5, abs=0.0)
    assert float(afm_uni[N_BURN:].mean()) == pytest.approx(0.009724, rel=1e-4, abs=0.0)
    for g in (_girdi("hf", "spin").gerekce, _girdi("tb", "spin").gerekce):
        assert "0.687930" in g
    assert "tekbicim cikti isaretle degisir ama degisim FM kosumunun bit bit ayar-imgesidir" in _girdi("hf", "spin").gerekce
    assert "0.009724" in _girdi("hf", "spin").gerekce

    _, kad_h = _zincir(-1.0, NEEL, alan=0.1)
    fm_h = _kos(1.0, alan=0.1)
    bozulma = float(np.abs(kad_h[N_BURN:] - fm_h.magnetisation_series).max())
    assert bozulma > 1e-2, "h ≠ 0'da da özdeş — kontrol kör"
    assert bozulma == 0.82421875
    assert f"max|fark| {bozulma!r}" in _girdi("hf", "spin").gerekce


def test_TB_SPIN_ISARET_BEYANI_KAYNAK_AFM_ve_KONTROL_DOGRU_ISARETLE():
    """`tb → spin` gerekçesinin işaret beyanı: atıflanan satırlar gerçekten
    antiferromanyetikliği yazıyor (bayatlarsa kırmızı) ve `h = 0.1`
    kontrolü DOĞRU işaretle (tekbiçim başlangıç) çarpansız c = −J'de
    5.078e-02 / 8.809e-01, c = −J/4'te 6.250e-02 / 7.812e-01: `h ≠ 0`'da
    çarpan SADELEŞMEZ ("sadeleşir" diye −J yazan mutant düşer)."""
    gerekce = _girdi("tb", "spin").gerekce
    atif = re.search(r"`engine/tb/downfold\.py:(\d+)(?:-(\d+))?`", gerekce)
    fab = re.search(r"`engine/tb/factory\.py:(\d+)` ve `engine/tb/factory\.py:(\d+)-(\d+)`", gerekce)
    kaynak = (KAYNAK / "engine" / "tb" / "downfold.py").read_text(encoding="utf-8").splitlines()
    fabrika = (KAYNAK / "engine" / "tb" / "factory.py").read_text(encoding="utf-8").splitlines()
    bas, son = int(atif.group(1)), int(atif.group(2) or atif.group(1))
    assert "antiferromanyetik" in " ".join(kaynak[bas - 1:son]).lower()
    t, b2, s2 = map(int, fab.groups())
    assert "antiferromanyetik" in fabrika[t - 1].lower()
    assert "antiferromanyetik" in " ".join(fabrika[b2 - 1:s2]).lower()

    uni = SpinState.uniform(L=L, n_component=1).spins
    a_uni, a_kad = _zincir(-1.0, uni, alan=0.1)
    b_uni, b_kad = _zincir(-BES_J[0], uni, alan=0.1)
    tek = float(np.abs(a_uni - b_uni)[N_BURN:].max())
    kad = float(np.abs(a_kad - b_kad)[N_BURN:].max())
    assert (f"{tek:.3e}", f"{kad:.3e}") == ("5.078e-02", "8.809e-01")
    assert f"tekbicimde {tek:.3e}, kademelide {kad:.3e}" in gerekce
    c_uni, c_kad = _zincir(-0.25, uni, alan=0.1)
    d_uni, d_kad = _zincir(-BES_J[0] / 4, uni, alan=0.1)
    ceyrek = (f"{np.abs(c_uni - d_uni)[N_BURN:].max():.3e}", f"{np.abs(c_kad - d_kad)[N_BURN:].max():.3e}")
    assert ceyrek == ("6.250e-02", "7.812e-01") and "c = -J/4'te %s / %s" % ceyrek in gerekce


# =============================================================================
# tablo: tek satır, damga kalktı, etiket içeride
# =============================================================================


def test_TABLODA_IKI_GIRDI_DOLDU_TEK_SATIR_DAMGA_KALKTI_ETIKET_ICERIDE():
    """Task 4'ün `bridge_map` üzerindeki izi TAM OLARAK şudur:

      * `hf → tb` ve `tb → spin` hâlâ TAUTOLOJİ kovasında, hâlâ birer
        satır (Task 4'te kova sayıları 1/4/8/0/8'di ve DEĞİŞMEDİ; 27.09.2026
        sayımı 3/5/8/0/5, 28.09.2026'da 3/8/8/0/2 -- TAUTOLOJİ 8: `hf → spin`,
        `tdse → spin`, `md → em`, `tb → em` kenarsız hükümle eklendi; bu test
        yalnız TAUTOLOJİ sayısını kilitler)
      * Task 4'te ikisinin `buyukluk`u doldu — biri 2.276e-15 (bağıl
        özdeşlik artığı), öteki 0.0 (bit bit ölüm). E6 (11.10.2026): ilki
        bir TOPLAMA-SIRASI KAYDI olduğu için yeniden `None` (K3, `tdse → tb`
        ile aynı kural); sayı gerekçede kayıt olarak durur. `tb → spin`in
        0.0'ı bir ölçüm sonucudur ve `float` kalır.
      * ikisinden `YENIDEN_URETILMEDI` damgası KALKTI; damga TAUTOLOJİ
        kovasının öteki üç girdisinden ikisinde (`md → spin`, `hf → em`)
        DURUYOR; `hf → spin` damgasız (27.09.2026'da ölçüldü)
      * "BU BORUDA OLCULEMEZ" etiketi, `U/|t|` ve `z` sayıları TEK bir
        girdinin içindedir — tabloda başka hiçbir girdi `z = 4` ya da
        `U/|t|` yazmaz. Ayrı satır YOK.
    """
    hf_tb, tb_spin = _girdi("hf", "tb"), _girdi("tb", "spin")
    assert hf_tb.kova == tb_spin.kova == Kova.TAUTOLOJI
    assert KOVA_SAYILARI[Kova.TAUTOLOJI] == 8
    assert sum(1 for f in KOVA_TABLOSU if f.kova == Kova.TAUTOLOJI) == 8

    assert hf_tb.buyukluk is None  # E6: toplama-sirasi kaydi (K3)
    # K3 iki kenara AYNI uygulanir: `tdse -> tb`nin 7.03e-16'si de tasinmadi.
    assert _girdi("tdse", "tb").buyukluk is None and "`tdse -> tb`" in hf_tb.gerekce
    assert isinstance(tb_spin.buyukluk, float) and tb_spin.buyukluk == _TB_SPIN_M_FARKI == 0.0
    assert repr(_HF_TB_OZDESLIK_BAGIL) in hf_tb.gerekce and _HF_TB_OZDESLIK_BAGIL > 0.0

    assert YENIDEN_URETILMEDI not in hf_tb.gerekce
    assert YENIDEN_URETILMEDI not in tb_spin.gerekce
    damgali = {f.cift for f in KOVA_TABLOSU if YENIDEN_URETILMEDI in f.gerekce}
    assert damgali == {("md", "spin"), ("em", "hf")}

    etiketli = [f for f in KOVA_TABLOSU if "BU BORUDA OLCULEMEZ" in f.gerekce]
    assert etiketli == [tb_spin]
    for parca in ("z = 4", "U/|t|", "ORAN SADELESMESIDIR"):
        assert [f for f in KOVA_TABLOSU if parca in f.gerekce] == [tb_spin], parca
    assert "Task 4" in hf_tb.gerekce and "Task 4" in tb_spin.gerekce
    # Gerekçenin sayılı cümleleri bu dosyanın ölçtüğü sayılarla aynı olmalı.
    assert "22 kabugun 10'unda" in hf_tb.gerekce
    assert sum(TERS_SIRA_DUSEN.values()) == 10 and sum(n // 2 for n, _ in ALTI_SISTEM) == 22
    assert "seeds = 12" in hf_tb.gerekce and "seeds = 12" in tb_spin.gerekce
    # 28.09: iki EM girdisi, TÜR etiketleri ve paylaşılan TÜR B sabiti LİTERAL (sabite karşı değil).
    md_em, tb_em = _girdi("md", "em"), _girdi("tb", "em")
    assert md_em.buyukluk is None and tb_em.buyukluk is None
    assert "AKIM KANALI (J = sum q v) TUR A" in md_em.gerekce and "TEKBICIM ortam TUR A" in tb_em.gerekce
    assert [f for f in KOVA_TABLOSU if "TUR B, HUKMU YOK" in f.gerekce and "serbest bir beyana baglidir" in f.gerekce] == [md_em, tb_em]
    assert "HIC KOSMADI" in md_em.gerekce and md_em.gerekce.count("DYADIK J'de") == 2 and "YARGILANAMAZ der" in tb_em.gerekce
    assert "TUR A -- DUZELTILEMEZ (h = 0 motorunda)" in _girdi("tdse", "spin").gerekce


@pytest.mark.slow
def test_TDSE_SPIN_J_R3_GRID2D_SANAL_ZAMANDA_CANLI_URETILIR_ve_EIGSH_ILE_BAGIMSIZ():
    """I6 (13.10.2026): `tdse → spin`in KAYNAK tarafı artık testte KOŞAR — R = 3.

    `TDSE_J` 28.09'dan beri LİTERALDİ ve onu üreten birleşim (`Grid2D` + sanal
    zaman `SplitOperatorPropagator`) süitte hiç koşmuyordu. Bu test J(R = 3)'ü
    o yoldan YENİDEN ÜRETİR. İki karşılaştırma, iki SINIF:

    (a) SANAL ZAMAN ↔ LİTERAL — GERİLEME KİLİDİ (pin), çıpa DEĞİL: literal aynı
        yöntemle aynı pencerede üretilmişti; bu kıyas literalin bugünkü motordan
        yeniden üretilebildiğini kilitler. ÖLÇÜLDÜ (13.10.2026, J − literal,
        adım sayısına göre): 2000 → −2.08e-09, 2250 → −2.92e-10, 2500 →
        −4.21e-11, 2750 → −6.54e-12, **3000 → −1.18e-12**, 4000 → −7.4e-15
        (her 250 adımda ~6-7 kat; tekil kip aralığının üstel kuyruğu). 3000
        adımda pin YAKINSAMIŞ J'yi ölçer: `2.3e-10` tolerans (literalin beyan
        edilen hassasiyeti, gerekçedeki "bağımsız eigsh ile fark ≤ 2.3e-10")
        kalan artığın ~190 katı; sınıra ancak ~2270 etkin adımda varılır, yani
        yakınsamayı ~%24 yavaşlatan bir değişiklik gerekir. (İlk sürüm 2500
        adımdaydı: pay 5.5 kat, ~%10 yavaşlama testi fiziksiz kırmızıya
        çeviriyordu — doğrulayıcı, 13.10.2026.)
    (b) EIGSH ↔ LİTERAL — BAĞIMSIZ YÖNTEM ÇIPASI (aynı ayrık H, Lanczos;
        propagatör HİÇ kullanılmaz): ölçülen 1.4e-12. FİZİK çıpası DEĞİLDİR —
        yumuşatma `a = b = 1` bir model seçimidir; çıpa YÖNTEMİ sınar.
        Sanal zaman ↔ eigsh enerjileri 3000 adımda 4.4e-11 / 5.0e-11 ayrışır
        (Strang `dt²` sabit noktası; tolerans 1e-9).
        Sayılar OPENBLAS_CORETYPE=Sandybridge ve NPY_DISABLE_CPU_FEATURES=X86_V3
        (2500 adım) ve Prescott (3000 adım) altında da aynı (J_sanal bit bit;
        J_eigsh ≤ 4e-15 oynar).

    SÖZLEŞME: sanal zaman her adımdan sonra 2B hacim öğesiyle (`cell_area`)
    yeniden normalize eder — ham adımdan sonra `Σ|ψ|²·dA = 1` (yuvarlama).

    KIRMA (mut.py). Bugünkü süitte (tests/tdse + tests/bridge + kök senaryo /
    gözlenebilir testleri, 384 test) HAYATTA kalan sanal-zaman normu
    `volume_element` yerine `grid.dx` ile burada norm kapısında düşer — GERÇEK
    bir 2B kusuru (J'ye kör; yalnız 2B'de görünür). `Grid2D`nin y ekseni yarım
    hücre kayınca J −6.06e-05 kayar ve burada da düşer, ama o mutantı bugün
    çift yarığın simetri testi (slow) zaten öldürüyor; iki eksen birlikte
    kayınca (öteleme, spektral kinetik) mutant EŞDEĞERDİR, hayatta kalır.

    YAKINSAMA BÜTÇESİ ARTEFAKTI — kusur öldürme DEĞİL: sanal zaman adımını
    yarıya indiren mutant (`-0.5j·dt`) AYNI taban hâline iki kat yavaş
    yakınsar. ÖLÇÜLDÜ (13.10.2026, J − literal): 3000 adımda −1.06e-07,
    4500'de −2.84e-10, 4750'de −1.05e-10, 6500'de +1.2e-12 (Strang `dt²`
    sabit noktası). Burada düşmesi yalnız 3000 adımlık bütçenin ona
    yetmemesindendir; 4750 adımlık bütçede hayatta kalırdı. Aynası, adım İKİ
    KATI (`-2j·dt`), ölçüldü: HAYATTA (≤ 2000 adımda yakınsar, sabit nokta
    −2.13e-11, tolerans içinde). Bu test sanal-zaman adımının ÖLÇEĞİNİ
    KİLİTLEMEZ (1B sanal zaman testinin abs = 1e-2 toleransı da kilitlemiyor).
    Süre ~27 s (2 × 3000 adım 256² FFT + 2 eigsh) -> slow.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    from qlab.engine.tdse.grid import Grid2D
    from qlab.engine.tdse.propagator import SplitOperatorPropagator

    n, uzunluk, dt, R, adim = 256, 40.0, 0.01, 3.0, 3000
    grid = Grid2D(nx=n, ny=n, lx=uzunluk, ly=uzunluk)
    X, Y = grid.X, grid.Y
    V = 1.0 / np.sqrt((X - Y) ** 2 + 1.0)  # e-e, yumusatma b = 1
    for XA in (-0.5 * R, 0.5 * R):  # e-cekirdek, yumusatma a = 1
        V = V - 1.0 / np.sqrt((X - XA) ** 2 + 1.0) - 1.0 / np.sqrt((Y - XA) ** 2 + 1.0)

    def g(u, c):
        return np.exp(-((u - c) ** 2))

    def baslangic(isaret):
        return g(X, -R / 2) * g(Y, R / 2) + isaret * g(X, R / 2) * g(Y, -R / 2)

    def enerji(psi):
        pk = np.abs(np.fft.fft2(psi)) ** 2
        w = np.abs(psi) ** 2
        return float(np.sum(pk * grid.k2) / (2.0 * np.sum(pk))) + float(np.sum(w * V) / np.sum(w))

    prop = SplitOperatorPropagator(grid=grid, potential=V, dt=dt, imaginary_time=True)
    E_sanal = {}
    for isaret in (1.0, -1.0):  # singlet (simetrik) / triplet (antisimetrik izdusum)
        psi = baslangic(isaret).astype(complex)
        for _ in range(adim):
            psi = prop.step(psi)
            psi = 0.5 * (psi + isaret * psi.T)
        ham = prop.step(psi)
        assert abs(float(np.sum(np.abs(ham) ** 2) * grid.cell_area) - 1.0) < 1e-12
        E_sanal[isaret] = enerji(psi)
    J_sanal = 0.5 * (E_sanal[1.0] - E_sanal[-1.0])
    assert abs(J_sanal - TDSE_J[0]) < 2.3e-10, J_sanal  # (a) PIN

    E_eigsh = {}
    for isaret in (1.0, -1.0):
        def mv(x, s=isaret):
            p = x.reshape(n, n)
            p = 0.5 * (p + s * p.T)
            hp = np.fft.ifft2(0.5 * grid.k2 * np.fft.fft2(p)).real + V * p
            return (0.5 * (hp + s * hp.T)).ravel()
        op = LinearOperator((n * n, n * n), matvec=mv, dtype=float)
        E_eigsh[isaret] = float(eigsh(op, k=1, which="SA", v0=baslangic(isaret).ravel(), tol=1e-13)[0][0])
    J_eigsh = 0.5 * (E_eigsh[1.0] - E_eigsh[-1.0])
    assert abs(J_eigsh - TDSE_J[0]) < 2.3e-10, J_eigsh  # (b) BAGIMSIZ YONTEM CIPASI
    for isaret in (1.0, -1.0):  # olculen (3000 adim) 4.4e-11 / 5.0e-11
        assert abs(E_sanal[isaret] - E_eigsh[isaret]) < 1e-9
    assert J_sanal < 0.0 and "J = (E_S - E_T)/2" in _girdi("tdse", "spin").gerekce


# ---------------------------------------------------------------------------
# `slow` İŞARETLERİNİN GEREKÇESİ — ÖLÇÜM, KURAL DEVRALMA DEĞİL.
#
# Deponun eşiği Task 2'de seçilmişti: kendi call süresi 2 s ve ÜSTÜNDE olan
# durum işaretlenir. Bu dosyanın `--durations=0` çıkışı (02.09.2026, depoda
# başka hiçbir şey koşmazken, her sistem için UHF İKİ kez):
#
#     N10_d2.6  4.40 / 4.07 / 3.96 s   (üç parametrik test)   -> slow
#     N8_d2.6   2.42 / 2.39 / 2.38 s                          -> slow
#     N8_d2.0   1.49 / 1.39 / 1.22 s                          -> DEĞİL
#     N6_*      0.50 ... 0.70 s                               -> DEĞİL
#     tb->spin  1.39 / 1.38 / 0.74 s  (L = 32, 2000 sweep)     -> DEĞİL
#     27.09.2026 izole: tb-işaret (−J/4 dahil) 1.64, hf-ayar 1.40, BES_J (replika dahil) 1.26,
#               hf-protokol A 0.79, hf-J 0.17 s -> DEĞİL; kalabalık koşuda en ağırı 1.96 s.
#               Görevin eşiği 3 s, bu dosyanınki 2 s: ikisinin de altında (sınıra YAKIN).
#     ast/tablo < 0.5 s                                       -> DEĞİL
#     13.10.2026 I6: tdse-spin J(R = 3) Grid2D sanal zaman (3000 adim) + eigsh ~27 s -> slow
#
# Eşik sınırın iki yanında iki komşu var (N8_d2.0 1.49 s, N8_d2.6 2.38 s);
# ikisi de eşiğin kendi tarafında ve işaret ona göre. Hızlı şeritte kalan
# dört sistem tautolojinin BÜTÜN iddialarını taşır (geri okuma, 15 hane,
# yedi mertebe (8, 2.0) ile görünür, Löwdin'in en küçüğü (6, 2.6)); yavaş
# şerit yalnız "en büyük" uçları ekler ((8, 2.6) Löwdin, (10, 2.6) özdeşlik).
# ---------------------------------------------------------------------------
