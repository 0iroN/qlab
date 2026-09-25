"""Modül 7 / Task 4 — düşen iki aday SESSİZCE DÜŞÜRÜLMEZ; tautoloji olarak,
SAYILARIYLA sevk edilir.

`bridge_map.py`de `hf → tb` ve `tb → spin` girdileri Task 1'den beri
TAUTOLOJİ kovasındaydı ama `buyukluk` alanları `None` idi ("Task 4'ün işi").
Bu dosya o alanları dolduran sayıları MOTORU KOŞTURARAK üretir ve
`bridge_map.py`deki literallere karşı sınar — `bridge_map` hiçbir motoru
import edemez, o yüzden sayılar orada literal, doğrulama burada canlı.

İKİ TAUTOLOJİ, İKİ AYRI MEKANİZMA:

  `hf → tb`   `t_m` bir `np.mean` olarak TANIMLANIYOR (`downfold.py:256-268`)
              ve aynı ortalama `==` ile geri okunuyor. Uyum değil, tanımın
              yeniden koşulması: aynı elemanlar TERS sırayla toplanınca 22
              kabuğun 10'unda `==` düşüyor (en çok 1 ulp). "On altı hane"
              iddiası ÖLÇÜLDÜ ve DARALTILDI: özdeşlik referans pencerede
              15 hane (bağıl 2.276e-15) ve SİSTEME BAĞLI (N=8 d=2.0'da
              1.090e-08 — yedi mertebe). Pencere (N, d, baz, seeds) her
              sayının yanında yazılıdır.

  `tb → spin` `J = 4t²/U` köprüsü BİT BİT ölü: `coupling` çıktıya YALNIZ
              `coupling/temperature` oranı olarak giriyor
              (`metropolis.py:164-166`), `h = 0`'da başka enerji ölçeği yok.
              Beş `J` (6.70 kat aralık) → max|m farkı| = 0.0. `U`'yu
              1016 kat büyütmek de, `z`'yi uyuşturmak da kenarı AÇMIYOR,
              çünkü engel oran-sadeleşmesidir. Kontrol: `h = 0.1`'de aynı
              iki `J` 9.961e-02 ayrışıyor — motor ikinci bir ölçek varsa
              `J`'yi görür; TB o ölçeği üretmez.

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
"""

from __future__ import annotations

import ast
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
    _HF_TB_OZDESLIK_BAGIL,
    _TB_SPIN_M_FARKI,
)
from qlab.engine.hf.multistart import search_uhf
from qlab.engine.spin.lattice import SpinState, neighbour_sum
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
from qlab.units import BOHR_TO_ANGSTROM

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
    (`downfold.py:256-268`). Bu test aynı kabuğu aynı sırada toplayıp
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
        assert _girdi("hf", "tb").buyukluk == _HF_TB_OZDESLIK_BAGIL
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
    `downfold.py:283`, `t_m` ataması `downfold.py:256-268` — ikisi de
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
    dört s'de). 1 ulp, 53 bitlik `rng.random()` karşılaştırmasını
    2⁻⁵² olasılıkla döndürür; 1 536 000 karşılaştırmada beklenen dönüş
    3.4e-10 — ve ölçülen 0. "Bit bit" bir ÖLÇÜM sonucudur; oranın
    sadeleşmesi ise cebirsel bir KESİNLİKtir.

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

    gerekce = _girdi("tb", "spin").gerekce
    for parca in (
        repr(ref.acceptance_rate), "287565/1536000", "L = 32", "tohum 20260822",
        "n_burn = 500", "n_sweep = 1500", "T = 2.269185 * J", "max|m farki| = 0.0",
        "E == J * E(J=1)",
    ):
        assert parca in gerekce, f"gerekçe {parca!r} taşımıyor"


def test_TB_SPIN_KOK_NEDEN_ORAN_SADELESMESI_ast_ile_KILITLI():
    """KÖK NEDEN `metropolis.py:164-166` — `coupling` çıktıya YALNIZ
    `coupling/temperature` oranı olarak girer. `ast` ile:

      `metropolis_sweep` içinde
        * `coupling` TAM BİR kez yüklenir, o da `delta = ...` atamasında
        * `temperature` — `_dogrula(...)` çağrısı dışında — TAM BİR kez,
          `ustel` atamasındaki TEK bölmenin paydası olarak, payı `delta`
        * `field` TAM BİR kez, yine `delta` atamasında (h = 0'da 0 · x)
        * satırlar: `delta` 164, `ustel` 165, bölme 166 — gerekçenin
          `metropolis.py:164-166`sından AYRIŞTIRILIR, elle yazılmaz
      `run_metropolis` içinde `coupling` iki kez: sweep'e ve `total_energy`ye
      geçirilir, başka hiçbir yerde okunmaz.
      `lattice.total_energy` içinde `coupling` TAM BİR kez, çarpan olarak.

    Yani `(J, T) → (λJ, λT)` altında kabul olasılığı `exp(-delta/T)`
    değişmez, `E` λ ile ölçeklenir, `m` `J`'yi hiç görmez. Kodda başka
    bir enerji ölçeği YOKTUR; olsaydı bu sayımlar değişirdi.
    """
    agac = ast.parse(Path(metropolis_mod.__file__).read_text(encoding="utf-8"))
    sweep = _fonksiyon(agac, "metropolis_sweep")
    delta = _atama(sweep, "delta")
    ustel = _atama(sweep, "ustel")

    coupling = _yukler(sweep, "coupling")
    assert len(coupling) == 1 and coupling[0] in list(ast.walk(delta.value))
    field = _yukler(sweep, "field")
    assert len(field) == 1 and field[0] in list(ast.walk(delta.value))

    dogrula_args = [
        n for d in ast.walk(sweep)
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "_dogrula"
        for n in ast.walk(d)
    ]
    temperature = [n for n in _yukler(sweep, "temperature") if n not in dogrula_args]
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
    assert len(_yukler(kosum, "coupling")) == 2

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
# tablo: tek satır, damga kalktı, etiket içeride
# =============================================================================


def test_TABLODA_IKI_GIRDI_DOLDU_TEK_SATIR_DAMGA_KALKTI_ETIKET_ICERIDE():
    """Task 4'ün `bridge_map` üzerindeki izi TAM OLARAK şudur:

      * `hf → tb` ve `tb → spin` hâlâ TAUTOLOJİ kovasında, hâlâ birer
        satır (Task 4'te kova sayıları 1/4/8/0/8'di ve DEĞİŞMEDİ; bugünkü
        sayım 3/4/8/0/6 -- TAUTOLOJİ yine 4 ve bu test yalnız onu kilitler)
      * ikisinin `buyukluk`u artık `None` değil — biri 2.276e-15 (bağıl
        özdeşlik artığı), öteki 0.0 (bit bit ölüm) — ve ikisi de `float`
      * ikisinden `YENIDEN_URETILMEDI` damgası KALKTI; damga TAUTOLOJİ
        kovasının öteki iki girdisinde (`md → spin`, `hf → em`) DURUYOR
      * "BU BORUDA OLCULEMEZ" etiketi, `U/|t|` ve `z` sayıları TEK bir
        girdinin içindedir — tabloda başka hiçbir girdi `z = 4` ya da
        `U/|t|` yazmaz. Ayrı satır YOK.
    """
    hf_tb, tb_spin = _girdi("hf", "tb"), _girdi("tb", "spin")
    assert hf_tb.kova == tb_spin.kova == Kova.TAUTOLOJI
    assert KOVA_SAYILARI[Kova.TAUTOLOJI] == 4
    assert sum(1 for f in KOVA_TABLOSU if f.kova == Kova.TAUTOLOJI) == 4

    assert isinstance(hf_tb.buyukluk, float) and hf_tb.buyukluk == _HF_TB_OZDESLIK_BAGIL
    assert isinstance(tb_spin.buyukluk, float) and tb_spin.buyukluk == _TB_SPIN_M_FARKI == 0.0
    assert hf_tb.buyukluk > 0.0

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
#     ast/tablo < 0.5 s                                       -> DEĞİL
#
# Eşik sınırın iki yanında iki komşu var (N8_d2.0 1.49 s, N8_d2.6 2.38 s);
# ikisi de eşiğin kendi tarafında ve işaret ona göre. Hızlı şeritte kalan
# dört sistem tautolojinin BÜTÜN iddialarını taşır (geri okuma, 15 hane,
# yedi mertebe (8, 2.0) ile görünür, Löwdin'in en küçüğü (6, 2.6)); yavaş
# şerit yalnız "en büyük" uçları ekler ((8, 2.6) Löwdin, (10, 2.6) özdeşlik).
# ---------------------------------------------------------------------------
