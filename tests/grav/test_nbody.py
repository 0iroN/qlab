"""N-body çözücüsü, sekiz yörüngesinin BVP'si, monodromi ve `m2 = 0` hata
çubuğu — Modül 6 Task 5.

Beş bölüm, ve dosyanın omurgası şu ayrımdır: **hangi sayı fiziği ölçüyor,
hangisi kendi arama yordamını geri okuyor?**

  A. **`nbody.py` METRİĞİ BİLMİYOR** (spec §4.2'nin ön koşulu). Kaydeden
     vekil kilidi + vekilin **göremediği** yüzü kapatan kaynak düzeyi
     denetimi. Vekilin sınırı burada ÖLÇÜLEREK gösteriliyor.
  B. **BVP — IC'ler depo içinde aranıyor**, literatürden alınmıyor. Alınsaydı
     çıpa dairesel olurdu; ve BVP ölçülerek DAHA İYİ çıkıyor.
  C. **ÇIPA `|E|^{3/2}T/(G²m^{5/2})`** ve **aramadan sonra daireselleşen üç
     iddia**. Üçü de burada "ölçüm DEĞİL" diye etiketli.
  D. **MONODROMİ — defektif özdeğer tuzağı.** Kararlılık `max|λ|` ile
     raporlanmaz; eliptik çiftin açısıyla raporlanır.
  E. **`m2 = 0` — N-body tarafının hata çubuğu** (spec §4.3'ün N-body
     yarısı). **İKİ gözlenebilir ayrı ayrı ölçülüyor:** spec'in kendi
     gözlenebiliri (monodrominin etki bloğu → `DEJENERE`) ve sevk edilen
     ikame (sarılma açısı kusuru → `SAGLAM`, gerçek kesişimli). Spec'in
     `∝ m2²`si **ikisinde de** düşüyor; öteki iki sayısı **çürütülmüyor**,
     spec'in gözlenebilirinde **üretilemiyor**.

PENCERE — her sayı sevk edilen kodla ölçüldü:

- `G = 1` (birim seçimi `metric.py`de beyan edilir; `nbody.py` onu import
  EDEMEZ, o yüzden orada `G` diye bir sabit yoktur — bkz. modül dokstringi).
- İntegratör `scipy.integrate.solve_ivp`, varsayılan **DOP853**,
  `rtol = 1e-13`, `atol = 1e-15`. Aksi yazılmadıkça bu pencere geçerlidir.
- BVP: `scipy.optimize.least_squares`, `xtol = ftol = gtol = 1e-15`,
  başlangıç tahmini `(0.35, 0.53, 6.3)`, koşul `Φ_{T/3}(y₀) = P_{(2,0,1)} y₀`
  (12 artık). Yakınsama 5 artık değerlendirmesinde.
- Kuvvet üssü `p = 2` (aksi yazılmadıkça); `p ≠ 2`de potansiyel
  `U = -m_i m_j / ((p-1) r^{p-1})` — **bir SEÇİM**, ve planın `p`-tablosu bu
  seçim beyan edilmediği için yeniden üretilemedi.
- Monodromi **varyasyonel denklemle** (sonlu farkla DEĞİL), aynı pencere.

`slow` EŞİĞİ ÖLÇÜLDÜ, DEVRALINMADI — VE BU DOSYADA TEMİZ BİR BOŞLUK YOK.
`--durations`, tek koşum (`-p no:randomly`):

    `slow` (8)  : 8.26 · 3.78 · 2.09 · 1.53 · 1.28 · 1.23 · 1.18 · **0.84** s
    işaretsiz   : **0.61** · 0.58 · 0.48 · 0.44 · 0.31 · 0.30 · 0.20 · … s

Eşik **0.8 s** seçildi. Aşağıda **1.31 kat**, yukarıda **1.05 kat** pay var —
yani `test_kapali_form.py`nin 4.8/2.8 katlık boşluğunun yanında BU BOŞLUK
DAR, ve makine gürültüsünün (%20'ye varıyor) içinde. Bunu gizlemek yerine
yazıyorum, çünkü sebebi yapısal: **BVP maliyeti bir SÜREKLİLİKTİR**
(çözüm başına 0.2–0.4 s), yani testler kesikli iki kümeye ayrılmıyor.

O yüzden sınıra düşen test (0.84 s) kronometreyle DEĞİL, bir YAPISAL
ÖLÇÜTLE `slow` işaretlendi: *BVP'yi ikiden fazla kez çözen, dörtten fazla
varyasyonel akış koşan, ya da bir boyutlu bir minimizasyon koşan test
`slow`dur.* İşaretsiz kümenin hiçbir üyesi bu ölçütü karşılamıyor
(`lru_cache` sayesinde çoğu hiç BVP çözmüyor).

ÖLÇÜLEN TOPLAM: bu dosya 30 test **23.7 s**; `tests/grav` 267 test
**28.8 s**, `-m "not slow"` ile 258 test **4.5 s** (9 deselect).

**RADAU MONODROMİSİ SEVK EDİLMİYOR** ve sebebi ölçüldü: 12×12 varyasyonel
sistem Radau'da **61.2 s** sürüyor (DOP853 0.23 s). Değeri
(`max|λ|−1 = 3.923e-06`) dokstringde yazılı ama süitte koşulmuyor; süitin
tamamını 13 kat büyütecek tek bir sayı için ödenecek bedel değil.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from qlab.engine.grav import nbody
from qlab.engine.grav.nbody import (
    EULER_PERMUTATION,
    FIGURE_EIGHT_GUESS,
    NBODY_METHODS,
    FigureEight,
    NBodyState,
    accelerations,
    angular_momentum,
    euler_state,
    figure_eight_residual,
    figure_eight_state,
    integrate,
    monodromy,
    permute,
    return_defect,
    solve_figure_eight,
    total_energy,
    variational_flow,
)
from qlab.errors import InvalidInputError
from qlab.observable.grav_obs import (
    CIRCULAR_PAIR_PERIOD,
    MONODROMY_GAP,
    MONODROMY_PERTURBATION,
    NBODY_PROBE_MASSES,
    PAIR_INDICES,
    THIRD_BODY_INDICES,
    circular_defect_closed_form,
    circular_frequency_defect,
    circular_pair_state,
    eight_invariant,
    monodromy_angles,
    newton_noise_floor,
    third_body_influence,
)

CIPA = 9.237681250723917
VX, VY, T = 0.347116888119, 0.532724945388, 6.325913982926


def _taban(**kw) -> float:
    """Merdivenin en büyük artığı — `newton_noise_floor`un yaptığının aynısı."""
    return max(abs(circular_frequency_defect(m2=m, **kw)
                   - circular_defect_closed_form(m))
               for m in NBODY_PROBE_MASSES)


# ---------------------------------------------------------------------------
# A — `nbody.py` METRİĞİ BİLMİYOR
# ---------------------------------------------------------------------------
class _Kaydeden:
    """Erişilen her öznitelik adını kaydeden şeffaf vekil."""

    def __init__(self, hedef):
        self._hedef = hedef
        self.dokunulan: set[str] = set()

    def __getattr__(self, ad):
        # `_hedef` ve `dokunulan` __dict__'te olduğu için buraya düşmezler.
        self.dokunulan.add(ad)
        return getattr(self._hedef, ad)


def test_NBODY_METRIGI_BILMIYOR_dokunulan_ad_kumesi_SAYILIYOR():
    """SINIF: transkripsiyon denetimi — spec §4.2'nin ÖN KOŞULU.

    İki çözücü ortak bir kod yolundan geçerse "biri ötekinin YARISINI
    üretiyor" iddiası düşer: `geodesic.py` eğri uzay-zamanda bir test
    parçacığı taşır, `nbody.py` düz uzayda birbirini çeken cisimleri.

    KAYDEDEN VEKİLİN SINIRI BİLİNİYOR (spec §5, ölçüldü): vekil çekirdeğin
    bir alanı OKUMADIĞINI kanıtlar; İKİ TÜKETİCİNİN ORTAK BİR TÜREVDEN
    gelmediğini kanıtlamaz. Buradaki iddia BİRİNCİ türdendir — `nbody`
    `Theory`ye HİÇ dokunmuyor — dolayısıyla vekil GEÇERLİDİR.
    """
    st = euler_state(VX, VY)
    for cagri in (lambda v: integrate(v, t_span=0.1),
                  lambda v: variational_flow(v, t_span=0.1)):
        vekil = _Kaydeden(st)
        cagri(vekil)
        assert vekil.dokunulan == {"masses", "positions", "velocities"}
        for yasak in ("lam", "gamma", "beta", "radial_w", "theory"):
            assert yasak not in vekil.dokunulan


def test_VEKILIN_KOR_NOKTASI_OLCULDU_ve_KAYNAK_DENETIMIYLE_KAPATILDI():
    """SINIF: ÇIPA DEĞİL — bir MİMARİ ÖLÇÜMÜ, ve vekilin sınırının SAYISI.

    Vekil yalnız `NBodyState`in okunan alanlarını sayar. Bir modül düzeyi
    `from ...metric import Theory` satırını ve o `Theory`nin gövde içinde
    kullanılmasını GÖREMEZ — aşağıdaki `sahte_cekirdek` tam bunu yapıyor ve
    vekil yine de TAM AYNI üç adı rapor ediyor.

    Bu yüzden iddia İKİ kilitle taşınır: vekil (çalışma zamanı) + kaynak
    düzeyi import denetimi (aşağısı). Modül 5'in dersi burada da geçerli:
    *bir tespit ölçütü de bir pencere düğmesidir.*
    """
    from qlab.engine.grav.metric import Theory  # modül düzeyi taklidi

    def sahte_cekirdek(state):
        gizli = Theory(gamma=0.5, beta=0.25)          # vekil BUNU GÖRMEZ
        return (np.asarray(state.masses).sum()
                + np.asarray(state.positions).sum()
                + np.asarray(state.velocities).sum()
                + gizli.lam * 0.0)

    vekil = _Kaydeden(euler_state(VX, VY))
    sahte_cekirdek(vekil)
    assert vekil.dokunulan == {"masses", "positions", "velocities"}

    # KAYNAK DÜZEYİ DENETİMİ — vekilin göremediği yüz.
    kaynak = open(nbody.__file__, encoding="utf-8").read()
    for satir in kaynak.splitlines():
        s = satir.strip()
        if s.startswith("import ") or s.startswith("from "):
            assert "metric" not in s, f"nbody.py metriği import ediyor: {s!r}"
            assert "geodesic" not in s, f"nbody.py geodesic import ediyor: {s!r}"
    for yasak in ("Theory", "radial_w", "closed_form_coefficient",
                  "isotropic_potential", "null_kernel"):
        assert yasak not in vars(nbody), (
            f"{yasak!r} `nbody` ad uzayında; Task 1 TÜKETİLMEMELİ.")


# ---------------------------------------------------------------------------
# B — BVP: IC'ler DEPO İÇİNDE aranıyor
# ---------------------------------------------------------------------------
def test_PERMUTASYON_OLCULEREK_SECILDI_ALTISININ_HEPSI_SINANDI():
    """SINIF: transkripsiyon denetimi — ama seçim bir ÖLÇÜMDÜR.

    `(2, 0, 1)` bir varsayım değil: altı permütasyonun tamamı BAŞLANGIÇ
    TAHMİNİNDE koşuldu ve beşi O(1) artık verdi. ÖLÇÜLEN
    (`max|Φ_{T/3}(y₀) − P y₀|`, guess `(0.35, 0.53, 6.3)`):

        (0,1,2) 1.992164e+00   (0,2,1) 1.577843e+00   (1,0,2) 1.595127e+00
        (1,2,0) 1.996448e+00   (2,1,0) 1.996448e+00
        (2,0,1) 4.391674e-02   <-- iki mertebe küçük, TEK aday

    PLANIN "(2,0,1) → 1.65e-08" SAYISI ÜRETİLEMEDİ ve sebebi ölçüldü:
    o sayı bir permütasyon özelliği DEĞİL, DEĞERLENDİRME NOKTASININ
    özelliğidir. Tahminde 4.39e-02, çözümün kendisinde ise 2.50e-14
    (BVP onu oraya indirdiği için). Plan iki noktayı ayırmamıştı.
    """
    olculen = {p: float(np.max(np.abs(figure_eight_residual(*FIGURE_EIGHT_GUESS,
                                                            perm=p))))
               for p in itertools.permutations((0, 1, 2))}
    assert olculen[EULER_PERMUTATION] == pytest.approx(4.391674e-02, rel=1e-5)
    otekiler = [v for p, v in olculen.items() if p != EULER_PERMUTATION]
    assert min(otekiler) > 1.5
    assert min(otekiler) / olculen[EULER_PERMUTATION] > 30.0


def test_BVP_ICLERI_DEPO_ICINDE_BULUYOR_LITERATURDEN_ALMIYOR():
    """SINIF: FİZİK ÇIPASININ ÖN KOŞULU — dairesellik burada kesiliyor.

    IC'ler literatürden alınsaydı `T` bir GİRDİ olurdu ve `|E|^{3/2}T`
    girilen sayının geri okunması olurdu. Burada `T` ARANIYOR.

    ÖLÇÜLEN: `vx = 0.347116888119`, `vy = 0.532724945388`,
    `T = 6.325913982926`, `max|artık| = 2.498002e-14`. Üçü de planın üç
    sayısıyla 12 hanede AYNI çıktı.
    """
    fe = solve_figure_eight()
    assert fe.vx == pytest.approx(VX, abs=5e-13)
    assert fe.vy == pytest.approx(VY, abs=5e-13)
    assert fe.period == pytest.approx(T, abs=5e-13)
    assert fe.residual == pytest.approx(2.498002e-14, rel=0.2)
    # ARTIK GERÇEKTEN SIFIRA GİDİYOR: `abs=0.0` ZORUNLU, yoksa varsayılan
    # `abs=1e-12` penceresi 2.5e-14'ü hiç ölçmezdi (Task 4'ün dersi).
    assert fe.residual == pytest.approx(0.0, abs=1e-13)
    assert fe.residual != 0.0


@pytest.mark.slow
def test_BVP_LITERATUR_IC_SINDEN_878_DEGIL_874_KAT_DAHA_IYI():
    """SINIF: ÇIPA DEĞİL — iki YORDAMIN karşılaştırması.

    Literatürün 8 haneli IC'si (Chenciner–Montgomery) EN İYİ `T`siyle bile
    tam periyot dönüşünde `max|Φ_T(y₀) − y₀| = 7.594684e-10` veriyor;
    BVP'ninki **8.690826e-13**, yani **873.9 kat** daha iyi.

    Plan 7.595e-10 / 8.646e-13 / 878 diyordu: **literatür yarısı BİREBİR**,
    BVP yarısı son hanesinde ayrılıyor (873.9 ↔ 878). Yani planın bu satırı
    ayakta, ama üç haneden fazlası taşınmıyor.

    NOT: literatür IC'sinin `|x₁| = 1.000000002830` — yani bu ansatz'ın
    `|q₁| = 1` normalizasyonu literatürün kendi 8 hanesi içinde. Ansatz
    literatürden bir SAYI almıyor, bir SİMETRİ SINIFI alıyor.
    """
    from scipy.optimize import minimize_scalar

    x1 = np.array([0.97000436, -0.24308753])
    v3 = np.array([-0.93240737, -0.86473146])
    lit = NBodyState(masses=np.ones(3),
                     positions=np.array([x1, -x1, [0.0, 0.0]]),
                     velocities=np.array([-v3 / 2, -v3 / 2, v3]))

    def hata(tt):
        s = integrate(lit, t_span=float(tt))
        return float(max(np.max(np.abs(s.positions - lit.positions)),
                         np.max(np.abs(s.velocities - lit.velocities))))

    r = minimize_scalar(hata, bracket=(6.3258, 6.3259, 6.3260),
                        options={"xtol": 1e-14})
    bvp = return_defect(solve_figure_eight())
    assert r.fun == pytest.approx(7.594684e-10, rel=1e-3)
    assert bvp == pytest.approx(8.690826e-13, rel=0.15)
    assert r.fun / bvp == pytest.approx(873.9, rel=0.2)
    assert np.linalg.norm(x1) == pytest.approx(1.000000002830, abs=1e-11)


@pytest.mark.slow
def test_ARAMA_NOKTASI_BIR_CEVAP_TASIMIYOR_DORT_UZAK_TAHMIN():
    """SINIF: ÇIPA DEĞİL — aramanın SAĞLAMLIĞI, ve HAVZA BİR KUTUYA BAĞLIDIR.

    Dört ayrı köşe tahmininden de aynı çözüm çıkıyor (rtol 1e-9), yani
    `(0.35, 0.53, 6.3)` ince ayarlı bir başlangıç değil.

    PLANIN "%58.2 HAVZA"SI YENİDEN ÜRETİLEMEZ — ve sebebi bir eksiklik
    değil, bir TANIMSIZLIK: havza yüzdesi KUTUYA göre tanımlıdır ve plan
    kutuyu hiç yazmamıştı. İki kutu İLAN EDİLEREK ölçüldü (3×3×3 eşit aralı
    ızgara, rtol 1e-9, yakınsama ölçütü `residual < 1e-7` ve
    `|T − T_ref| < 1e-5` ve `||vx| − |vx_ref|| < 1e-5`):

        vx [0.25, 0.45] × vy [0.45, 0.62] × T [5.8, 6.8]  →  26/27 = %96.3
        vx [0.10, 0.60] × vy [0.30, 0.80] × T [4.5, 8.5]  →  17/27 = %63.0

    Yani "havza %X" beyanı kutusuz bir sayı DEĞİLDİR. Bu test dar kutunun
    dört köşesini koşuyor (tam ızgara 9 s, dört köşe 1.5 s).
    """
    for g in [(0.25, 0.45, 5.8), (0.45, 0.62, 6.8),
              (0.25, 0.62, 6.8), (0.45, 0.45, 5.8)]:
        fe = solve_figure_eight(guess=g, rtol=1e-9)
        assert fe.residual < 1e-7
        assert fe.period == pytest.approx(T, abs=1e-8)
        assert abs(fe.vx) == pytest.approx(VX, abs=1e-8)


# ---------------------------------------------------------------------------
# C — ÇIPA ve ARAMADAN SONRA DAİRESELLEŞENLER
# ---------------------------------------------------------------------------
def test_SEKIZ_CIPASI_9_237681250724():
    """SINIF: FİZİK ÇIPASI — statik kökü YOK, `6π` ile aynı raf.

    ÖLÇÜLEN: **9.237681250723917**, planın 9.237681250724'ü ile BİREBİR.
    `T` yörünge çözülmeden bilinemez (BVP); `E` o çözümün başlangıç
    durumundan okunur.

    İKİNCİ KÜTLE NOKTASI MUTASYONLA GELDİ, TASARIMLA DEĞİL. `m^{5/2}`yi
    `m^{2}`ye çeviren mutant TAM SÜİTTE ölüyor ama `-m "not slow"`
    koşumunda **258 testin hepsinden sağ çıkıyordu** — çünkü tek kütle
    noktası `m = 1`di ve orada `1^{2.5} = 1^{2}`. `m = 2` noktası (bir BVP
    çözümü, 0.35 s) o boşluğu HIZLI süitte de kapatıyor.
    """
    assert eight_invariant(solve_figure_eight()) == pytest.approx(CIPA,
                                                                  rel=1e-12)
    ikinci = solve_figure_eight(guess=(0.35 * math.sqrt(2.0),
                                       0.53 * math.sqrt(2.0),
                                       6.3 / math.sqrt(2.0)), mass=2.0)
    assert eight_invariant(ikinci) == pytest.approx(CIPA, rel=1e-12)


@pytest.mark.slow
def test_CIPA_RTOL_ILE_YAKINSIYOR_ve_PLANIN_IKI_UCU_DA_KAYDI():
    """SINIF: FİZİK ÇIPASININ HATA ÇUBUĞU (yakınsama yolu).

    BVP her `rtol`de YENİDEN çözülüyor; referans rtol 1e-13.

        rtol      1e-9       1e-10      1e-11      1e-12
        |fark|    1.027e-08  6.437e-10  6.864e-11  4.967e-12

    PLANIN "7.1e-9 → 5.0e-13"Ü ÜRETİLEMEDİ: iki uç da birer mertebe kaydı
    (1.027e-08 ve 4.967e-12). Yön ve şekil aynı — monoton, ~`rtol` ile.
    """
    ref = eight_invariant(solve_figure_eight(rtol=1e-13))
    farklar = {r: abs(eight_invariant(solve_figure_eight(rtol=r)) - ref)
               for r in (1e-9, 1e-10, 1e-11, 1e-12)}
    assert farklar[1e-9] == pytest.approx(1.027e-08, rel=0.05)
    assert farklar[1e-12] == pytest.approx(4.967e-12, rel=0.2)
    siralı = [farklar[r] for r in (1e-9, 1e-10, 1e-11, 1e-12)]
    assert siralı == sorted(siralı, reverse=True)


@pytest.mark.slow
def test_CIPA_KUTLEDEN_BAGIMSIZ_ama_G_KARE_SINANMIYOR():
    """SINIF: FİZİK ÇIPASI — ve PAYDANIN İKİ ÇARPANI AYNI STATÜDE DEĞİL.

    `m^{5/2}` GERÇEKTEN İŞ YAPIYOR: `m = 0.5/1/2/7`de BVP yeniden çözülüyor,
    `T` = 8.946193349060 / 6.325913982926 / 4.473096674530 / 2.390970744858
    (yani `T ~ m^{-1/2}`), bölme sonrası dördünün yayılımı **6.573e-14**.

    `G²` İSE SAYISAL OLARAK İŞ YAPMIYOR: `nbody.py`de `G` diye bir sabit
    yok (o dosya birim beyanını taşıyan `metric.py`yi import EDEMEZ).
    Formülde BOYUTSAL olarak durur, ama sınanmış DEĞİLDİR — ve panel bunu
    böyle yazar. Bir formülün her çarpanı aynı ölçüde sevk edilmiş olmaz.
    """
    degerler, periyotlar = [], []
    for m in (0.5, 1.0, 2.0, 7.0):
        g = (0.35 * math.sqrt(m), 0.53 * math.sqrt(m), 6.3 / math.sqrt(m))
        fe = solve_figure_eight(guess=g, mass=m)
        degerler.append(eight_invariant(fe))
        periyotlar.append(fe.period)
    assert max(degerler) - min(degerler) == pytest.approx(6.573e-14, rel=0.5)
    assert all(v == pytest.approx(CIPA, rel=1e-12) for v in degerler)
    # `T ~ m^{-1/2}`: ölçek yasası da geri okunuyor.
    for m, p in zip((0.5, 1.0, 2.0, 7.0), periyotlar):
        assert p * math.sqrt(m) == pytest.approx(T, rel=1e-11)


@pytest.mark.slow
def test_CIPA_YANILABILIR_KUVVET_USSUYLE_KAYIYOR_ve_P_3_TE_COZUM_YOK():
    """SINIF: FİZİK ÇIPASI — YANILABİLİRLİK SINAVI.

    `power ≠ 2` bir yaklaşım değil BAŞKA BİR KURAMDIR (`metric.py`nin
    `Theory`si gibi bir kuram seçici). Çıpa onunla kayıyor:

        p     1.7        1.9        2.0       2.1       2.3       2.6    3.0
        çıpa  23.956761  12.586202  9.237681  6.778916  3.556908  1.095405 YOK

    `p = 3`te BVP'nin çözümü KALMIYOR (`InvalidInputError`).

    PLANIN AYNI TABLOSU (35.47 / 13.26 / 9.24 / 6.7560 / 3.91 / 1.73)
    ÜRETİLEMEDİ ve sebebi ölçüldü: `p ≠ 2`de potansiyelin kapalı formu bir
    SEÇİMDİR ve plan onu HİÇ BEYAN ETMEMİŞTİ. Buradaki seçim
    `U = -m_i m_j / ((p-1) r^{p-1})` (kuvvetin integrali). `p = 2`de iki
    seçim çakışır, o yüzden ÇIPANIN KENDİSİ etkilenmez.

    *Adlandırılmamış bir pencere düğmesi yine bir tabloyu hüküm dışına
    taşıdı* — bu modülde üçüncü kez.
    """
    beklenen = {1.7: 23.956761, 1.9: 12.586202, 2.0: 9.237681,
                2.1: 6.778916, 2.3: 3.556908, 2.6: 1.095405}
    olculen = {p: eight_invariant(solve_figure_eight(power=p))
               for p in beklenen}
    for p, b in beklenen.items():
        assert olculen[p] == pytest.approx(b, rel=1e-6)
    # MONOTON AZALAN: kuvvet sertleştikçe çıpa küçülüyor.
    assert list(olculen.values()) == sorted(olculen.values(), reverse=True)
    with pytest.raises(InvalidInputError, match="integrasyonu basarisiz"):
        solve_figure_eight(power=3.0)


def test_ARAMADAN_SONRA_DAIRESELLESEN_UC_IDDIA():
    """SINIF: ÇIPA DEĞİL — ve bu testin İŞİ tam olarak "çıpa değil" demek.

    Spec §3.6 üçünü de "PANELDE YAZILACAK" diye ayırıyor; burada ÜÇÜ DE
    ölçülerek kilitleniyor:

    1. **"üçü de aynı eğride `T/3` kaymayla"** = BVP KOŞULUNUN TA KENDİSİ.
       Aramadan sonra bunu "bulduk" demek, koştuğumuz denklemi geri okumaktır.
    2. **tam periyot dönüşü** = `Φ_T = (Φ_{T/3})³` CEBİRSEL SONUCU. Ölçüldü:
       üç kez `T/3` uygulamak ile bir kez `T` uygulamak arasındaki fark
       **7.861e-14** — yani sayı İNTEGRATÖRÜ ölçüyor, fiziği değil.
    3. **`L = 0` BİR ÖLÇÜM DEĞİL**: simetrik ansatz'ta `euler_state(0, 0)`
       bile TAM 0 veriyor. `abs=0.0` ile sınanıyor — varsayılan `abs=1e-12`
       burada hiçbir şey ölçmezdi.

    Girdi olarak kalan tek şey **SİMETRİ SINIFIDIR** — bir sayı değil.
    """
    fe = solve_figure_eight()
    y0 = figure_eight_state(fe)

    # (1) BVP koşulunun kendisi: T/3 sonrası durum P y0'a EŞİT.
    hedef = permute(y0, EULER_PERMUTATION)
    son = integrate(y0, t_span=fe.period / 3.0)
    assert np.max(np.abs(son.positions - hedef.positions)) < 1e-13

    # (2) Tam periyot CEBİRSEL SONUÇ.
    s = y0
    for _ in range(3):
        s = integrate(s, t_span=fe.period / 3.0)
    bir_kerede = integrate(y0, t_span=fe.period)
    assert float(np.max(np.abs(s.positions - bir_kerede.positions))) == (
        pytest.approx(7.861e-14, rel=0.6))

    # (3) L = 0 ansatz'ın kendisinden geliyor.
    assert angular_momentum(y0) == pytest.approx(0.0, abs=0.0)
    assert angular_momentum(euler_state(0.0, 0.0)) == pytest.approx(0.0,
                                                                    abs=0.0)
    assert angular_momentum(euler_state(0.9, -0.4)) == pytest.approx(0.0,
                                                                     abs=0.0)


def test_ENERJI_KORUNUMU_CIPA_DEGIL_BES_KUVVET_USSUNDE_DE_GECIYOR():
    """SINIF: ÇIPA DEĞİL — "çıpa değil" rafının üçüncü üyesi (spec §9/7).

    Enerji korunumu bir TAUTOLOJİDİR: yanlış kuvvet üssünde de geçer, yani
    modeli çözmüş olmayı hiç ölçmez. Ölçüldü — `p = 1.7 … 2.6` beşinde de
    bağıl sapma 1e-11'in altında, ve `p = 2` diğerlerinden AYRIŞMIYOR.
    """
    y0 = euler_state(VX, VY)
    for p in (1.7, 1.9, 2.0, 2.3, 2.6):
        E0 = total_energy(y0, power=p)
        E1 = total_energy(integrate(y0, t_span=2.0, power=p), power=p)
        assert abs(E1 - E0) / abs(E0) < 1e-11


# ---------------------------------------------------------------------------
# D — MONODROMİ: DEFEKTİF ÖZDEĞER TUZAĞI
# ---------------------------------------------------------------------------
def test_MONODROMI_LAMBDA_1_SEKIZ_KATLI_ve_DEFEKTIF():
    """SINIF: transkripsiyon denetimi — ama TUZAĞIN KİLİDİ.

    `max|λ| − 1` bir kararsızlık ölçüsü DEĞİLDİR, çünkü `λ = 1` sekiz katlı
    ve DEFEKTİFTİR. Bu bir varsayım değil, iki bağımsız sayım:

        cebirsel kat = 8   (12 özdeğerin 8'i `|arg| < 1e-3`)
        geometrik kat = 4  (`(M − I)`nin tekil değerlerinin 4'ü < 1e-6:
                            3.55e-13, 1.81e-13, 8.04e-15, 3.75e-15;
                            kalan 8'i 1.72e-02 … 7.26e+01)

    yani DÖRT TANE 2×2 Jordan bloğu. Boşluk 10 mertebe — bir eşik seçimi
    değil, bir uçurum.

    Bu tuzak bu depoda İKİNCİ kez çıktı (ilki L4/L5 eşiğinde).
    """
    M = monodromy(solve_figure_eight())
    lam = np.linalg.eigvals(M)
    acilar = np.abs(np.angle(lam))
    assert int(np.sum(acilar < 1e-3)) == 8
    sv = np.linalg.svd(M - np.eye(12), compute_uv=False)
    assert int(np.sum(sv < 1e-6)) == 4
    assert sv[7] > 1e-2          # 8. tekil değer HÂLÂ büyük
    assert sv[8] < 1e-11         # 9.'su on mertebe aşağıda
    assert sv[7] / sv[8] > 1e8


def test_KARARLILIK_ELIPTIK_CIFTIN_ACISIYLA_RAPORLANIR():
    """SINIF: FİZİK ÇIPASI (kararlılık hükmü) — ve `max|λ|` ile DEĞİL.

    ÖLÇÜLEN: **±1.872970598421** ve **±0.052921540137** rad. İkisi de planın
    yazdığı değerlerle BİREBİR (12 hane). Dördü de birim çember üzerinde,
    yani sekiz LİNEER KARARLI.
    """
    buyuk, kucuk = monodromy_angles(solve_figure_eight())
    assert buyuk == pytest.approx(1.87297059842, abs=1e-11)
    assert kucuk == pytest.approx(0.05292154014, abs=1e-11)
    # Dört özdeğer de birim çemberde.
    lam = np.linalg.eigvals(monodromy(solve_figure_eight()))
    eliptik = [z for z in lam if abs(np.angle(z)) > 1e-3]
    assert len(eliptik) == 4
    for z in eliptik:
        assert abs(abs(z) - 1.0) < 1e-9


@pytest.mark.slow
def test_MAX_LAMBDA_YAKINSAMIYOR_ACILAR_YAKINSIYOR():
    """SINIF: ÇIPA DEĞİL — ama bu dosyanın EN ÖNEMLİ ÖLÇÜMÜ.

    `max|λ| − 1` düğmelere tepki verir ve YAKINSAMAZ; eliptik açılar
    yakınsar. Aynı koşumlarda ölçüldü:

        rtol      1e-13      1e-12      1e-11      1e-10      1e-9
        max|λ|−1  1.500e-07  2.560e-07  1.920e-07  1.503e-07  2.195e-07
        metot     DOP853     RK45       LSODA      (Radau 3.923e-06,
        max|λ|−1  1.500e-07  9.266e-07  2.482e-07   61.2 s — koşulmuyor)

    Dört mertebe daralan bir `rtol` bu sayıyı HİÇ küçültmüyor. AYNI
    koşumlarda büyük açı 1.872970598421 / …8423 / …8420 ve küçük açı
    0.052921540137 / …133 / …143 — yani 11-12 hane SABİT.

    PLANIN "8.0e-8"İ: ölçülen bandın (1.5e-07 … 3.9e-06) hemen altında ve
    aynı mertebede — ama asıl bulgu SAYININ DEĞERİ DEĞİL, YAKINSAMAMASIDIR.
    Plan onu tek bir sayı gibi yazıyordu; tek bir sayı değil.
    """
    tepe = {}
    for rtol in (1e-13, 1e-12, 1e-11, 1e-10):
        M = monodromy(solve_figure_eight(rtol=rtol), rtol=rtol)
        tepe[rtol] = float(np.max(np.abs(np.linalg.eigvals(M)))) - 1.0
        b, k = monodromy_angles(solve_figure_eight(rtol=rtol), rtol=rtol)
        assert b == pytest.approx(1.87297059842, abs=1e-8)
        assert k == pytest.approx(0.05292154014, abs=1e-8)
    # YAKINSAMIYOR: `rtol` 1000 KAT daralıyor, sayı 3 kattan az oynuyor.
    assert 1e-8 < tepe[1e-13] < 1e-5
    assert max(tepe.values()) / min(tepe.values()) < 3.0
    # VE BÜYÜKLÜĞÜ KAREKÖK İMZASI TAŞIYOR: integrasyon hatası ~1e-13 iken
    # `max|λ|−1` ~1.5e-07, yani `sqrt(1e-13) = 3.16e-07`in 2.1 katı içinde.
    assert 0.2 < tepe[1e-13] / math.sqrt(1e-13) < 5.0

    metotlar = {}
    for meth in ("DOP853", "RK45", "LSODA"):
        M = monodromy(solve_figure_eight(method=meth), method=meth)
        metotlar[meth] = float(np.max(np.abs(np.linalg.eigvals(M)))) - 1.0
        b, k = monodromy_angles(solve_figure_eight(method=meth), method=meth)
        assert b == pytest.approx(1.87297059842, abs=1e-9)
        assert k == pytest.approx(0.05292154014, abs=1e-9)
    assert max(metotlar.values()) / min(metotlar.values()) > 3.0


def test_KAREKOK_TEPKISI_US_YARIM_ve_BIRIM_MATRIS_OLMAZDI():
    """SINIF: kapalı-form türev — defektifliğin İMZASI.

    Defektif bir özdeğer bozulmaya `sqrt` ile tepki verir. `M + δN` ile
    ölçüldü (`N` = `MONODROMY_PERTURBATION`, DETERMİNİSTİK ve JENERİK;
    RNG yok):

        δ         1e-4       1e-6       1e-8       1e-10
        max|λ|−1  1.201e-02  1.199e-03  1.199e-04  1.205e-05

    log-log eğimi **0.4998** — plan 0.4981 diyordu, aynı hüküm.

    VE `N = I` OLMAZDI, ÖLÇÜLEREK: `M + δI`nin özdeğerleri TAM `λ + δ`dir,
    yani üs 1 çıkar ve sınav defektifliği HİÇ GÖRMEZ. Karşılaştırma
    δ ∈ {1e-2, 1e-3, 1e-4}'te yapılıyor — `I` yönünde daha küçük δ'lar
    KULLANILAMAZ, çünkü `max|λ|−1`in kendi tabanı 1.5e-07'dir ve δ onun
    altına düştüğünde eğim ölçülemez hale gelir (ölçüldü: δ = 1e-10'a kadar
    inen bir merdivende `I` eğimi 0.4625 çıkıyor — yani taban SINAVIN
    KENDİSİNİ bozuyor). Ölçülen: `N` eğimi **0.5044**, `I` eğimi **0.9997**.
    """
    M = monodromy(solve_figure_eight())
    ds = np.array([1e-4, 1e-6, 1e-8, 1e-10])
    ys = np.array([np.max(np.abs(np.linalg.eigvals(M + d * MONODROMY_PERTURBATION)))
                   - 1.0 for d in ds])
    egim = float(np.polyfit(np.log(ds), np.log(ys), 1)[0])
    assert egim == pytest.approx(0.4998, abs=0.01)
    assert ys[0] == pytest.approx(1.201e-02, rel=0.05)

    # BİRİM MATRİS BİR SINAV DEĞİL: üs 1 verir. Aynı δ aralığında iki yön.
    ds2 = np.array([1e-2, 1e-3, 1e-4])
    egim2 = {}
    for ad, P in (("N", MONODROMY_PERTURBATION), ("I", np.eye(12))):
        y = np.array([np.max(np.abs(np.linalg.eigvals(M + d * P))) - 1.0
                      for d in ds2])
        egim2[ad] = float(np.polyfit(np.log(ds2), np.log(y), 1)[0])
    assert egim2["N"] == pytest.approx(0.5044, abs=0.02)
    assert egim2["I"] == pytest.approx(0.9997, abs=0.02)


def test_MONODROMI_KAPISI_SABIT_ESIK_DEGIL_BOSLUK_SINAVI():
    """SINIF: transkripsiyon kilidi — kapının KENDİSİ ölçülmüş.

    `monodromy_angles` eliptik çifti sabit bir açı eşiğiyle DEĞİL, 4. ve 5.
    en büyük `|arg|` arasındaki BOŞLUKLA ayırıyor. Ölçülen boşluk
    **1.048e+04** kat (0.052921540137 ↔ 5.047990e-06); eşik
    `MONODROMY_GAP = 100`.

    Kapının tetiklenebilir olduğu ÖLÇÜLEREK gösteriliyor: periyodu bozulmuş
    bir `FigureEight`te `Φ_T` artık bir monodromi değildir, `λ = 1` kümesi
    dağılır ve boşluk kapanır — kapı reddediyor. İnandırıcı görünen bir
    açı çifti döndürmek yerine.
    """
    M = monodromy(solve_figure_eight())
    acilar = np.sort(np.abs(np.angle(np.linalg.eigvals(M))))[::-1]
    assert acilar[3] / acilar[4] == pytest.approx(1.048e4, rel=0.3)
    assert acilar[3] / acilar[4] > MONODROMY_GAP
    with pytest.raises(InvalidInputError, match="BOSLUK sinavidir"):
        monodromy_angles(FigureEight(vx=VX, vy=VY, period=6.0, residual=0.0))


# ---------------------------------------------------------------------------
# E — `m2 = 0`: N-BODY BORUSUNUN KENDİ HATA ÇUBUĞU
# ---------------------------------------------------------------------------
def test_M2_SIFIR_NOKTASI_DEJENERE_DEGIL_KARELEMENIN_TAM_TERSI():
    """SINIF: HATA ÇUBUĞU — ve Task 2'nin açığının BU BORUDA OLMADIĞI ölçümü.

    Karelemede `λ = 0` NOKTASI dejenereydi: integrand sabit 1.0'a düşüyor,
    `dφ − 2π` BİT BİT 0.0 çıkıyordu, ve Task 2 tabanı bir LİMİTTEN okumak
    zorunda kalmıştı. N-BODY BORUSUNDA NOKTA ÇALIŞIYOR — ölçüldü:

        rtol      1e-13            1e-11            1e-9
        defect(0) -3.874678e-14    -3.872458e-12    -3.988428e-10

    Üçü de bit bit sıfırdan FARKLI ve `rtol` ile ölçekleniyor. Yani spec
    §4.3'ün N-body yarısının yazdığı `m2 = 0` burada gerçekten
    kullanılabilir; merdivenin ilk basamağı TAM SIFIRDIR ve ölüdür DEĞİL.
    """
    assert NBODY_PROBE_MASSES[0] == 0.0
    assert circular_defect_closed_form(0.0) == 0.0
    olculen = {r: circular_frequency_defect(m2=0.0, rtol=r)
               for r in (1e-13, 1e-11, 1e-9)}
    for r, v in olculen.items():
        assert v != 0.0, "m2 = 0 basamağı ÖLÜ; boru aritmetik yapmıyor"
        assert abs(v) / r == pytest.approx(0.39, rel=0.2)
    nf = newton_noise_floor(circular_frequency_defect)
    assert "7/7 canli" in nf.pipeline


def test_NBODY_TABANI_KENDI_BORUSUNDAN_ve_ETIKETI_AYRI():
    """SINIF: HATA ÇUBUĞU — spec §4.3, KARIŞIK BORU YASAK.

    Dört boru, dört etiket: `kareleme` · `bukulme` · `ode` · **`nbody`**.
    N-body'nin tabanı ne karelemeden ne ODE'den okunur.

    ÖLÇÜLEN (DOP853, rtol 1e-13, atol 1e-15, revolutions 1, samples 256):
    taban **3.996803e-14**, sinyal **1.236067977500** (= `√5 − 1`,
    `m2 = 1`in kapalı formu), S/G **3.0926e+13**, 7/7 basamak canlı,
    hüküm `SAGLAM`.

    Karelemenin tabanından (1.594187e-09) **4.0e+04 kat**, ODE'ninkinden
    (1.921807e-06) **4.8e+07 kat** uzakta — karışık boru olsaydı S/G dört
    ile yedi mertebe arasında yalan söylerdi.
    """
    nf = newton_noise_floor(circular_frequency_defect)
    assert nf.pipeline.startswith("nbody | method=DOP853 |")
    assert nf.verdict == "SAGLAM"
    assert nf.value == pytest.approx(3.996803e-14, rel=0.2)
    assert nf.signal == pytest.approx(math.sqrt(5.0) - 1.0, rel=1e-11)
    assert nf.ratio > 1e13
    # `method` BORUYU SEÇER, `rtol` seçmez — Task 4'ün ölçütü.
    # (Etiket sınavı LSODA ile koşuluyor: Radau aynı sonucu verir ama tek
    # başına 4.9 s sürer ve bu test `slow` DEĞİLDİR.)
    assert "method=LSODA" in newton_noise_floor(
        circular_frequency_defect, method="LSODA").pipeline
    assert "rtol" not in nf.pipeline


@pytest.mark.slow
def test_TABAN_KIPIRDIYOR_IKI_YON_ve_REVOLUTIONS_KIPIRDATMIYOR():
    """SINIF: HATA ÇUBUĞU — "kıpırdıyor mu" sınavı (Task 2'nin dersi).

    Kıpırdamayan bir taban bir hata çubuğu değil bir SABİTTİR. Ölçüldü:

        rtol   1e-13      1e-12      1e-11      1e-10      1e-9       1e-6
        taban  3.997e-14  3.888e-13  3.915e-12  3.942e-11  3.993e-10  4.456e-07
        atol   1e-15      1e-13      1e-11      1e-9       1e-7
        taban  3.997e-14  8.171e-14  4.046e-12  3.934e-10  3.963e-08
        metot  DOP853     RK45       LSODA      (Radau 7.550e-15, 4.9 s)
        taban  3.997e-14  1.629e-13  7.595e-11

    `rtol` yönü **1.1e+07 kat**, `atol` yönü **9.9e+05 kat**, metot yönü
    **1.90e+03 kat** oynatıyor.

    `revolutions` İSE OYNATMIYOR — ve bu da bir ölçümdür: 1/4/16/64 →
    3.997e-14 / 3.408e-14 / 1.132e-14 / 9.459e-14, yani 8.4 kat içinde ve
    YÖNSÜZ (16'da düşüyor, 64'te geri çıkıyor). Test yalnız 1/4/16'yı
    koşuyor: `revolutions = 64` tek başına **7.2 s** ve hükmü değiştirmiyor.
    Sebebi yapısal: gözlenen açı DEVİR BAŞINA normalize edilir,
    birikmiş hata da ~doğrusal biriktiği için oran sabit kalır. Bir
    düğmenin tabanı kıpırdatMAMASI, o düğmenin `PIPELINE_KEYS`e neden
    girmediğinin değil, tabanın neyi ölçtüğünün kanıtıdır.
    """
    rtol_yon = {r: _taban(rtol=r) for r in (1e-13, 1e-11, 1e-9, 1e-6)}
    assert rtol_yon[1e-13] == pytest.approx(3.996803e-14, rel=0.2)
    assert rtol_yon[1e-11] == pytest.approx(3.914646e-12, rel=0.2)
    assert rtol_yon[1e-6] / rtol_yon[1e-13] > 1e6

    atol_yon = {a: _taban(atol=a) for a in (1e-15, 1e-11, 1e-7)}
    assert atol_yon[1e-7] / atol_yon[1e-15] > 1e5

    metot_yon = {m: _taban(method=m) for m in ("DOP853", "RK45", "LSODA")}
    assert metot_yon["LSODA"] / metot_yon["DOP853"] > 1e3

    devir = {n: _taban(revolutions=n) for n in (1.0, 4.0, 16.0)}
    assert max(devir.values()) / min(devir.values()) < 20.0


def test_SPEC_IN_M2_KARE_IDDIASI_DUSTU_oteki_ikisi_BASKA_GOZLENEBILIRDE():
    """SINIF: HATA ÇUBUĞU — ve DAR TUTULMUŞ bir çürütme.

    **BU BORU SPEC'İN BORUSU DEĞİLDİR.** Spec §4.3'ün N-body gözlenebiliri
    *"`m2 = 0` → DOĞRUSAL HARİTA BİRİM"*, yani **monodromi**; burası
    **sarılma açısı** kusuru. Dolayısıyla spec'in **taban 3.651e-15** ve
    **kesişim 3e-4** sayıları burada ÇÜRÜTÜLMÜŞ OLMAZ — bu boruda tanımlı
    bile değiller. (Bir çürütme, çürüttüğü iddiayla AYNI gözlenebilir
    üzerinde ölçülmelidir; ilk yazdığım hâli bu kuralı çiğniyordu.)

    Spec'in **iki gözlenebilirde de sınanabilen tek sayısı** `sinyal ∝ m2²`
    ve BURADA düşüyor: sinyal `m2` ile DOĞRUSAL — bir fit değil, KAPALI
    FORMDAN çıkan sonuç (`√(1+4m2) − 1 = 2m2 − 2m2² + …`). Ölçülen
    `defect/m2`: 1.832160 / 1.980390 / 1.998004 / 1.999800 / 1.999980 /
    1.999998 (m2 = 1e-1 … 1e-6), log-log eğimi **0.99418**. Aynı sayı
    spec'in KENDİ gözlenebilirinde de ölçüldü ve orada da düştü — bkz.
    `test_SPEC_IN_KENDI_GOZLENEBILIRI_DEJENERE_CIKTI`.

    BU BORUNUN KENDİ tabanı ve kesişimi (spec'inkiyle karşılaştırılamaz,
    ayrı gözlenebilir): taban rtol 1e-11'de **3.914646e-12**; kesişim
    `m2 ≈ taban/2` — rtol 1e-13'te **2.0e-14**, 1e-11'de **2.0e-12**,
    1e-9'da **2.0e-10**. Kesişim böyle tanımlanır çünkü sinyal eğrisi
    `2m2`dir ve taban `m2`den BAĞIMSIZDIR (merdiven boyunca plato).
    """
    ms = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    oranlar = [circular_frequency_defect(m2=m) / m for m in ms]
    assert oranlar[0] == pytest.approx(1.832160, rel=1e-5)
    assert oranlar[-1] == pytest.approx(1.999998, rel=1e-5)
    egim = float(np.polyfit(np.log(ms), np.log([o * m for o, m in
                                                zip(oranlar, ms)]), 1)[0])
    assert egim == pytest.approx(0.99418, abs=0.005)
    assert egim < 1.5, "üs 2 OLSAYDI burası ~2 çıkardı"

    # BU BORUNUN kendi tabanı ve kesişimi — spec'in sayılarıyla DEĞİL,
    # kendi kapalı formuyla karşılaştırılıyor.
    for rtol, kesisim in ((1e-13, 2.0e-14), (1e-11, 2.0e-12), (1e-9, 2.0e-10)):
        taban = _taban(rtol=rtol)
        assert taban / 2.0 == pytest.approx(kesisim, rel=0.3)


@pytest.mark.slow
def test_SPEC_IN_KENDI_GOZLENEBILIRI_DEJENERE_CIKTI():
    """SINIF: HATA ÇUBUĞU — spec §4.3'ün KENDİ gözlenebiliri, ÖLÇÜLDÜ.

    Spec'in cümlesi: *"`m2 = 0` (üçüncü cisim kuvvet uygulamaz → DOĞRUSAL
    HARİTA BİRİM olmalı)"*. Gözlenebilir `third_body_influence`:

        B(m2) = Φ_t[çift satırları, üçüncü cisim sütunları]   (8 × 4)

    yani üçüncü cismin durumundaki bir bozulmanın çifti ne kadar oynattığı.
    `m2 = 0`da kuvvet yok → blok tam sıfır → harita blok-üçgen.

    PENCERE: DOP853, `rtol = 1e-13`, `atol = 1e-15`, `t_span = 4π`
    (`CIRCULAR_PAIR_PERIOD`), norm = `max` mutlak eleman,
    `circular_pair_state` kurulumu. Tek koşum **0.141 s**.

    **(1) SPEC'İN `∝ m2²`Sİ BURADA DA DÜŞTÜ.** `max|B|/m2` ON HANEDE sabit:
    **1.123391199e+10**, m2 = 1e-8 … 1e-16 (sekiz dekat). `m2²` terimi VAR
    ama ALT mertebe (m2 = 1e-2'de orana %1.25 katkı → `c₁/c₀ ≈ 1.25`).
    Yapısal sebep: üçüncü cismin çift denklemlerine girdiği tek yol
    `∂a_{0,1}/∂r₂ = m2·(…)`, yani her yol TAM BİR çarpan `m2` taşır.

    **(2) AMA SPEC'İN ÖTEKİ İKİ SAYISI ÇÜRÜTÜLMEDİ — ÜRETİLEMİYORLAR:**
    `B(0)` **bit bit sıfır** (32 elemanın 32'si tam 0.0), çünkü varyasyonel
    kaynak terimi `m2` ile çarpılıyor ve `0.0 * x` tam 0.0. Karşı kontrol:
    `max|Φ_t| = 2.246782e+10` — boru ÇALIŞIYOR, ölü olan yalnız blok. Ve
    LİMİT de kurtarmıyor: `m2` 16 dekat küçülünce `max|B|` de 16 dekat
    küçülüyor, oran sabit → **gürültü sinyalle BİRLİKTE ölçekleniyor,
    KESİŞİM YOK.** `newton_noise_floor` bu boruya **DEJENERE** diyor.

    İŞTE `circular_frequency_defect`İN İKAME OLARAK SEVK EDİLME GEREKÇESİ:
    bir tercih değil, bu ölçüm. Task 2'nin karelemede yaşadığı arızanın
    aynısı — *"fiziği kapat" bir NOKTA olarak dejenere* — ve burada LİMİT
    onarımı da çalışmıyor.
    """
    assert third_body_influence(m2=0.0) == 0.0
    Phi = variational_flow(circular_pair_state(0.0), t_span=CIRCULAR_PAIR_PERIOD)
    B0 = Phi[np.ix_(PAIR_INDICES, THIRD_BODY_INDICES)]
    assert B0.size == 32 and np.all(B0 == 0.0)
    assert float(np.max(np.abs(Phi))) == pytest.approx(2.246782e+10, rel=0.05)

    # (1) ÜS 1, ve oran ON HANEDE sabit.
    oran = {m: third_body_influence(m2=m) / m
            for m in (1e-8, 1e-12, 1e-16)}
    for v in oran.values():
        assert v == pytest.approx(1.123391199e+10, rel=1e-6)
    assert third_body_influence(m2=1e-2) / 1e-2 == pytest.approx(
        1.137459541e+10, rel=1e-5)          # m2^2 terimi: ALT mertebe

    # (2) MAKİNE "DEJENERE" DİYOR — iddia laf değil, kayıt.
    nf = newton_noise_floor(third_body_influence)
    assert nf.verdict == "DEJENERE"
    assert nf.value == 0.0
    assert math.isnan(nf.ratio)
    assert nf.pipeline.startswith("nbody-dogrusal-harita | method=DOP853 |")
    assert "0/1 canli" in nf.pipeline
    # Karşılaştır: AÇI borusu aynı makinede SAGLAM.
    assert newton_noise_floor(circular_frequency_defect).verdict == "SAGLAM"


def test_CAGIRAN_M2_YI_SECEMEZ_ve_CURUK_BANDI_GENIS_DEGIL():
    """SINIF: HATA ÇUBUĞU — ve SEVK EDİLMİŞ BİR İDDİANIN DÜŞÜŞÜ.

    `newton_noise_floor`un dokstringi Task 4'te şöyle yazmıştı: *"Geniş ve
    güvenli bir CURUK bandı hâlâ N-body'den (`m2 = 0`, sinyal `~m2²`)
    bekleniyor — ama artık 'erişilemez' diye değil, 'dar' diye."* Bu cümle
    ilk kez SINANABİLİR hâle geldi ve DÜŞTÜ.

    ÖLÇÜLEN (`atol = rtol/100`, merdiven aynı):

        rtol  1e-6      1e-3      1e-2    1e-1   2e-1   3e-1     4e-1  7e-1
        S/G   2.75e+06  1.92e+03  216.97  21.64  11.95  0.993068 9.07  5.73

    yani `CURUK` TEK BİR düğme noktasında beliriyor, iki yanında kayboluyor.
    GENİŞ BİR BANT YOK.

    SEBEBİ N-BODY'DE DEĞİL, `newton_noise_floor`UN SÖZLEŞMESİNDE — ve
    sözleşmenin o maddesi DOĞRUDUR: çağıran bir `NEWTON_PROBES` anahtarını
    veremez (aşağıda kilitli), yani `m2`yi seçemez. Dolayısıyla `signal`
    her zaman VARSAYILAN `m2 = 1`in O(1) fizik değeridir ve taban ona ancak
    integrasyon çöpe döndüğünde yetişir — o noktada "sinyal" de artık fizik
    değildir (1.236067977500 yerine 0.0856).

    *Aynı borudan okunan bir hata çubuğunda `CURUK` yapısal olarak DAR bir
    olaydır.* Üç boruda üç kez ölçüldü; bu bir boru özelliği değil, ölçütün
    kendi özelliğiymiş.
    """
    with pytest.raises(InvalidInputError, match="verilemez"):
        newton_noise_floor(circular_frequency_defect, m2=0.5)

    sg = {}
    for rtol in (1e-2, 1e-1, 2e-1, 3e-1, 4e-1):
        nf = newton_noise_floor(circular_frequency_defect, rtol=rtol,
                                atol=rtol * 1e-2)
        sg[rtol] = nf.ratio
    assert sg[1e-2] == pytest.approx(216.97, rel=0.05)
    assert sg[3e-1] == pytest.approx(0.993068, rel=0.05)
    assert sg[3e-1] < 1.0 < sg[2e-1] and sg[4e-1] > 1.0
    hukum = newton_noise_floor(circular_frequency_defect, rtol=3e-1,
                               atol=3e-3).verdict
    assert hukum == "CURUK"


# ---------------------------------------------------------------------------
# KAPILAR — geçersiz girdi normal sonuç gibi dönmez
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kw, kalip", [
    ({"masses": np.array([1.0]), "positions": np.zeros((1, 2)),
      "velocities": np.zeros((1, 2))}, "en az iki cisim"),
    ({"masses": np.ones(3), "positions": np.zeros((2, 2)),
      "velocities": np.zeros((3, 2))}, "sekli"),
    ({"masses": np.array([1.0, -1.0]), "positions": np.array([[0.0, 0.0], [1.0, 0.0]]),
      "velocities": np.zeros((2, 2))}, "negatif kutle"),
    ({"masses": np.ones(2), "positions": np.zeros((2, 2)),
      "velocities": np.zeros((2, 2))}, "ayni noktada"),
])
def test_INTEGRATE_KAPILARI(kw, kalip):
    """SINIF: transkripsiyon kilidi — hatalı girdi sessizce geçmez."""
    with pytest.raises(InvalidInputError, match=kalip):
        integrate(NBodyState(**kw), t_span=0.1)


def test_DIGER_KAPILAR():
    """SINIF: transkripsiyon kilidi."""
    st = euler_state(VX, VY)
    with pytest.raises(InvalidInputError, match="sevk edilen listede yok"):
        integrate(st, t_span=0.1, method="Euler")
    with pytest.raises(InvalidInputError, match="p <= 1"):
        integrate(st, t_span=0.1, power=1.0)
    with pytest.raises(InvalidInputError, match="uc bilesenlidir"):
        solve_figure_eight(guess=(0.3, 0.5))
    with pytest.raises(InvalidInputError, match="periyot tahmini pozitif"):
        solve_figure_eight(guess=(0.3, 0.5, -1.0))
    with pytest.raises(InvalidInputError, match="permutasyon degil"):
        permute(st, (0, 0, 1))
    with pytest.raises(InvalidInputError, match="en az 8"):
        circular_frequency_defect(samples=4)
    with pytest.raises(InvalidInputError, match="pozitif olmali"):
        circular_frequency_defect(revolutions=0.0)
    with pytest.raises(InvalidInputError, match="negatif olamaz"):
        circular_frequency_defect(m2=-1.0)
    with pytest.raises(InvalidInputError, match="negatif olamaz"):
        circular_defect_closed_form(-1.0)


def test_IVME_CEKIRDEGI_ETKI_TEPKI_ve_KOSEGEN_ARITMETIKLE_ELENIYOR():
    """SINIF: transkripsiyon kilidi — çekirdeğin iki değişmezi.

    (1) Eşit kütlelerde toplam ivme SIFIR (etki-tepki). `abs=0.0` ile
        sınanıyor: varsayılan `abs=1e-12` penceresi burada hiçbir şey
        ölçmezdi (Task 4'ün dersi).
    (2) Köşegen bir `if` ile değil `inf ** negatif = 0.0` aritmetiğiyle
        eleniyor; dallanma olsaydı bir mutant o dalı değiştirip sessizce
        geçebilirdi. `1/inf` terimi bit bit 0.0 olmalı.
    """
    q = np.array([[1.0, 0.0], [-1.0, 0.3], [0.2, -0.7]])
    a = accelerations(np.ones(3), q)
    toplam = np.sum(a, axis=0)
    assert float(toplam[0]) == pytest.approx(0.0, abs=1e-15)
    assert float(toplam[1]) == pytest.approx(0.0, abs=1e-15)
    # Kütle ağırlıklı toplam da sıfır (farklı kütlelerde).
    m = np.array([0.5, 2.0, 3.0])
    am = accelerations(m, q)
    assert float(np.max(np.abs(np.sum(m[:, None] * am, axis=0)))) == (
        pytest.approx(0.0, abs=1e-15))
    # Sıfır kütleli cisim kuvvet UYGULAMAZ ama kuvvet ALIR.
    a0 = accelerations(np.array([1.0, 1.0, 0.0]), q)
    a1 = accelerations(np.array([1.0, 1.0, 0.0]),
                       np.array([q[0], q[1], [5.0, 5.0]]))
    assert np.allclose(a0[:2], a1[:2], atol=0.0, rtol=0.0) or np.allclose(
        a0[:2], a1[:2], atol=1e-18)
    assert float(np.max(np.abs(a0[2]))) > 0.1


def test_NBODY_METHODS_KAPALI_LISTE():
    """SINIF: transkripsiyon kilidi — sevk edilen integratörler yazılı."""
    assert NBODY_METHODS == ("DOP853", "RK45", "Radau", "LSODA")
    assert EULER_PERMUTATION == (2, 0, 1)
    assert FIGURE_EIGHT_GUESS == (0.35, 0.53, 6.3)
