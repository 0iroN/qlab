"""Senaryo veri yapıları.

Bu modül hiçbir şey import etmez (numpy dahil): saf tanım katmanıdır.
Uzunluklar bohr, enerjiler hartree, zamanlar atomik birimdir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: `engine/tdse/factory.py::_build_potential`un SEVK ETTİĞİ potansiyeller.
#: `loader.py`de zaten bir KAPIYA bağlıydı; 30.08.2026'da eksik olan
#: TÜRETME KİLİDİ eklendi (`tests/test_scenario.py`). Kapı ile kilit ayrı
#: şeylerdir: kapı, demet bayat olsa bile yeşil kalır -- geçersiz bir ad
#: her hâlükârda reddedilir. Bayat demetin bedeli, factory'nin
#: desteklediği bir potansiyelin YAML'dan HİÇ yüklenememesidir.
#: Demet ölçüldü ve BAYAT DEĞİLDİ; kilit o günü beklemiyor.
POTENTIAL_KINDS = ("free", "barrier", "well", "harmonic", "uniform_field")


@dataclass(frozen=True)
class GridSpec:
    n: int
    length_bohr: float


@dataclass(frozen=True)
class PotentialSpec:
    kind: str
    params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PacketSpec:
    x0_bohr: float
    k0: float
    sigma_bohr: float


@dataclass(frozen=True)
class TDSEScenario:
    name: str
    grid: GridSpec
    potential: PotentialSpec
    packet: PacketSpec
    dt: float
    n_steps: int
    save_every: int = 1
    mass: float = 1.0
    absorbing_width_bohr: float | None = None


@dataclass(frozen=True)
class HFScanScenario:
    name: str
    molecule_template: str
    basis: str
    parameter: str
    values: tuple[float, ...]
    unit: str = "angstrom"


#: `engine/md/factory.py`nin SEVK ETTİĞİ kuvvet alanları. Bu demet ELLE
#: yazılır, çünkü katman sözleşmesi `scenario`nun `engine`i import etmesini
#: yasaklar (ölçüldü: `scenario -> qlab.engine` enjeksiyonu rc=1 veriyor).
#: Ama artık BAYATLAYAMAZ: `tests/md/test_md_scenario.py`deki türetme
#: kilidi factory'nin sevk dallarını `ast` ile tarayıp bu demete karşı
#: sınar. Kilit, demetin ÖLÜ ve BAYAT olduğu dönemde (`hartree_fock`
#: eksikti, oysa `scenarios/h2_aimd.yaml:10` tam olarak onu kullanıyor)
#: yokluğu ölçülerek eklendi -- 30.08.2026. 24.09.2026: beşinci tür
#: `tb_ss_sigma` (md<->tb ss-sigma kenarı, `engine/md/tb_kuvvet.py`)
#: eklendi; kilit onu da factory'nin sevkinden türetir.
MD_FORCE_FIELD_KINDS = ("harmonic", "morse", "lennard_jones", "hartree_fock", "tb_ss_sigma")

#: `tb_ss_sigma` kuvvet alanının PARAMETRE ŞEMASI. Öteki MD türlerinin
#: parametreleri `**p` ile doğrudan sınıfa gider; bu türünkiler iki iç içe
#: yasa taşıdığı ve biri BİRİM DÖNÜŞÜMÜ gerektirdiği için (Kelvin -> hartree,
#: `engine/md/factory.py`) YAML tarafında adıyla denetlenir: bilinmeyen ya da
#: eksik anahtar sessizce yutulmaz (`loader.py`nin genel kuralı).
#:
#: `electron_temperature_kelvin` KELVİNDİR ve sınıf HARTREE alır
#: (`kT_hartree`); iki adın ayrı olması tuzağın kendisini adlandırır:
#: Kelvin sayısı hartree sanılıp verilirse kuvvet sessizce küçülür (spec
#: B7). `n_electron` toplam elektron sayısıdır (spin dahil). `box_x_bohr`
#: İSTEĞE BAĞLIDIR: yoksa kutusuz, varsa YALNIZ x ekseni periyodiktir.
TB_SS_SIGMA_PARAMS = (
    "ss_sigma", "itme", "n_electron", "electron_temperature_kelvin", "box_x_bohr",
)
TB_SS_SIGMA_REQUIRED = ("ss_sigma", "itme", "n_electron", "electron_temperature_kelvin")
#: İki yasa rolü. Hoplama rolündeki yasanın adı `ss_sigma`, çift itmesi
#: rolündekinin adı `itme` (spec 1: rol çağrı yerinde adlandırılır).
TB_SS_SIGMA_LAWS = ("ss_sigma", "itme")
#: Her yasanın alanları. `engine/tb/radyal.py::UsselRadyalYasa`nın dataclass
#: alanlarıyla AYNI olmak ZORUNDADIR (factory yasayı `**alanlar` ile kurar);
#: katman sözleşmesi yüzünden elle yazılır ve `tests/md/test_md_scenario.py`
#: onu sınıfın alanlarına karşı sınar -- MD_FORCE_FIELD_KINDS'in türetme
#: kilidiyle aynı desen.
RADIAL_LAW_FIELDS = ("genlik_hartree", "d0_bohr", "us", "r_on_bohr", "r_cut_bohr")
#: `engine/md/factory.py`nin SEVK ETTİĞİ termostatlar. `MD_FORCE_FIELD_KINDS`
#: demetiyle aynı gerekçe ve aynı türetme kilidi. TEK FARK ÖLÇÜLDÜ (30.08.2026): bu
#: demet ÖLÜydü ama BAYAT DEĞİLDİ -- factory tam olarak `langevin` +
#: `nose_hoover` sevk ediyor. Yani burada kapatılan şey mevcut bir kusur
#: değil, aynı tuzağın İKİNCİ KOPYASI: bir üçüncü termostat eklendiğinde
#: bayatlayacaktı ve o an kapıya bağlıysa `argon_nvt.yaml` tipi geçerli
#: senaryolar reddedilirdi. Kilit o günü beklemiyor.
MD_THERMOSTAT_KINDS = ("langevin", "nose_hoover")


@dataclass(frozen=True)
class MDScenario:
    """Klasik moleküler dinamik senaryosu.

    `initial["masses_amu"]` AMU'dur; motor içi elektron kütlesine `factory`
    tarafından çevrilir (bkz. qlab.units.amu_to_electron_mass).
    """

    name: str
    force_field: dict
    initial: dict
    thermostat: dict | None
    dt: float
    n_steps: int
    save_every: int = 1


@dataclass(frozen=True)
class TBHopping:
    """Tek bir hoplama yolu. `t_ev` ELEKTRONVOLTTUR.

    Depo içi birim hartree'dir; dönüşüm `engine/tb/factory.py` sınırında
    yapılır (bkz. md/factory.py'nin amu_to_electron_mass çağrısı).
    """

    i: int
    j: int
    cell: tuple[int, int, int]
    t_ev: float


@dataclass(frozen=True)
class TBScenario:
    """Tight-binding senaryosu.

    `onsite_ev` ve `hoppings[*].t_ev` ELEKTRONVOLTTUR; örgü ve taban
    BOHR'dur. TB literatürü eV'de yayınlandığı için giriş birimi eV
    seçilmiştir ve dönüşüm tek bir sınırda yapılır.
    """

    name: str
    lattice_vectors_bohr: tuple[tuple[float, float, float], ...]
    basis_bohr: tuple[tuple[float, float, float], ...]
    onsite_ev: tuple[float, ...]
    hoppings: tuple[TBHopping, ...]
    n_electron_per_cell: int
    k_points: int


@dataclass(frozen=True)
class DownfoldScenario:
    """UHF indirgemesi senaryosu. `ring_size` ÇİFT olmalı.

    `distances_bohr` BOHR'dur ve dönüşüm yoktur: indirgeme motorunun kendi
    birimi de bohr'dur (`downfold.hydrogen_ring` komşu aralığını bohr alır).
    Bu, `TBScenario`'nun eV sınırından bilinçli olarak FARKLIDIR — orada
    dönüşüm vardı çünkü TB literatürü eV'de yayınlanır; burada girdi bir
    geometridir ve geometrinin atomik birimi bohr'dur.

    `ring_size` çift olmak ZORUNDADIR: iki-siteli antiferromanyetik hücre tek
    sayıda sitede kurulamaz (frustrasyon). Kontrol hem yükleyicide hem
    motorda vardır ve bu bir tekrar değil bir katman ayrımıdır — yükleyici
    YAML'ın hatasını, motor çağrının hatasını yakalar.
    """

    name: str
    ring_size: int
    distances_bohr: tuple[float, ...]
    basis: str = "sto-3g"
    n_neighbour: int = 2
    seeds: int = 12


SPIN_MODEL_KINDS = ("ising", "heisenberg")
SPIN_ALGORITHM_KINDS = ("metropolis", "wolff")


@dataclass(frozen=True)
class SpinScenario:
    """Spin Monte Carlo sıcaklık taraması.

    BİRİM DÖNÜŞÜMÜ YOKTUR VE BU BİR EKSİKLİK DEĞİL BİR BEYANDIR: bu modülde
    `J = k_B = 1` bir BİRİM SEÇİMİDİR. `temperatures` bu yüzden `kT/J`
    cinsindendir ve KELVİN DEĞİLDİR; `coupling` ile `field` de boyutsuzdur.
    `TBScenario` eV'den hartree'ye çevirir çünkü TB literatürü eV'de
    yayınlanır; burada çevrilecek bir şey yoktur, çünkü ölçülen büyüklükler
    (`kT_c/J`, kritik üsler) zaten boyutsuzdur. Bu, `DownfoldScenario`nun
    "girdi bir geometridir, atomik birimi zaten bohr'dur" gerekçesiyle aynı
    aile: dönüşümün yokluğu bir unutma değil, bir ölçüm sonucudur.

    `seed` ZORUNLUDUR ve varsayılanı YOKTUR. Bir Monte Carlo senaryosunun
    tekrarlanabilir olmaması kabul edilemez; varsayılan bir tohum, tohumu
    yazmayı unutan bir senaryonun sessizce "tekrarlanabilir" görünmesine yol
    açardı.

    `heisenberg_step` YALNIZCA `model = "heisenberg"` ile birlikte
    verilebilir (yükleyici bunu reddeder). Ising'de öneri işaret çevirmedir
    ve adım parametresi hiçbir şeyi değiştirmez; anahtarın sessizce
    yutulması, kullanıcının ayarladığını sandığı bir düğmenin bağlı olmaması
    demekti.
    """

    name: str
    model: str
    algorithm: str
    lattice_size: int
    temperatures: tuple[float, ...]
    n_sweep: int
    n_burn: int
    seed: int
    coupling: float = 1.0
    field: float = 0.0
    heisenberg_step: float = 0.6


#: FFT pencereleri. Bu tuple bir KOPYA DEĞİL bir KATMAN AYRIMIDIR: asıl
#: otorite `observable/em_obs.py::WINDOW_SIDELOBE`tir ve pencerenin ÖLÇÜLEN
#: yan lobunu da o taşır, ama `scenario` katmanı `observable`ı import EDEMEZ
#: (katman sözleşmesi: run → viz → observable → engine → scenario). Yükleyici
#: burada yalnızca "böyle bir pencere var mı" diye bakar; "seçilen tepe tabanı
#: o pencerenin yan lobunun üstünde mi" İMKÂN KAPISI `mode_spectrum`ta durur
#: ve orada kalmalıdır — çünkü o kapı bir ÖLÇÜME dayanır, bir isim listesine
#: değil. İki denetim ayrı tür fırlatır (`ScenarioError` ile
#: `InvalidInputError`) ve testler bu farkı kilitler.
EM_WINDOW_FNS = ("dikdortgen", "hann", "blackman")

#: Merdivenin en küçük derinliği — `observable/em_obs.py::MIN_DEPTH` ile aynı
#: sayı, aynı gerekçeyle ve aynı katman ayrımıyla tekrar edilir: n değer →
#: n−1 fark → n−2 mertebe → n−3 Richardson noktası, yani `u`nun ölçülebildiği
#: en küçük derinlik DÖRTTÜR. Yükleyici bunu YAML tarafında reddeder ki
#: panelin yakınsama satırı hiçbir zaman "yapısal olarak yargılanamaz" bir
#: koşumu göstermesin.
EM_MIN_DEPTH = 4


@dataclass(frozen=True)
class EMScenario:
    """Maxwell FDTD — 2B dairesel PEC boşlukta yarıçap merdiveni.

    TOHUM YOKTUR VE BU BİR EKSİKLİK DEĞİL BİR BEYANDIR: FDTD deterministtir,
    RNG hiç kullanılmaz. Tekrarlanabilirliğin karşılığı burada **pencerenin
    tam beyanıdır** ve bu yüzden aşağıdaki alanların hepsi ZORUNLUDUR —
    ızgara (`radii` + `margin`), `courant`, adım (`step_factor`),
    dalga boyu (`wavelength_per_radius`), `sgf`, FFT penceresi (`window_fn`),
    `pad`, `peak_floor` ve `detectors`. `SpinScenario`nun `seed`i neden
    varsayılansızsa bunlar da o yüzden varsayılansızdır: pencereyi yazmayı
    unutan bir senaryo sessizce "tekrarlanabilir" görünürdü ve bu depoda bir
    hükmün penceresinin dışına taşınması ON KEZ ölçüldü.

    BİRİM DÖNÜŞÜMÜ YOKTUR, çünkü çevrilecek bir şey yoktur: bu modülde
    `c = ε₀ = μ₀ = 1` bir BİRİM SEÇİMİDİR (Modül 4'ün `J = k_B = 1`'iyle aynı
    statüde). `radii` ve `wavelength_per_radius` HÜCRE cinsindendir, metre
    değil; frekanslar `1/(hücre)` cinsindendir, hertz değil.

    `npml` VE `dx` BİRER ALAN DEĞİLDİR, ve ikisi de bilinçli — ama SEBEPLERİ
    FARKLI:

    * `npml`: dairesel PEC boşluk TAM YANSITICIDIR, soğurucu katman yoktur ve
      `CircularPEC.thickness` sabit sıfırdır. Yani düğme **bağlı değildir**.
    * `dx`: düğme bağlıydı ama **sessizce yanlış** çalışıyordu ve bu ÖLÇÜLDÜ.
      `run_2d` kaynağı `t = n·S·Δx` zamanında besliyordu, oysa `wavelength`
      **hücre** cinsinden verilir; ikisi yalnız `Δx = 1`'de örtüşüyordu.
      Ölçüm (R = 5/10, `step_factor` = 120): `Δx` = 1 → tepe sayısı
      **20 / 13**, `Δx` = 0.5 → **4 / 4**, `Δx` = 2 → **31 / 43**, `λ`
      üçünde de aynı (7.257613) kalarak. Bugün düğme **motorda da kapalı**:
      `engine/em/grid.py::dx_kapisi` `Δx ≠ 1`'i reddediyor, çünkü ölçüldüğünde
      `Δx`in AYRI BİR SERBESTLİK DERECESİ OLMADIĞI çıktı — davranış yalnız
      `wavelength/Δx` oranına bağlı ve bu BİT BİT doğrulandı. Yani bu satır
      artık "senaryo bu düğmeyi taşımıyor" değil, "böyle bir düğme YOK"
      demektir.

    İki durumda da kural aynı: bağlanmamış ya da sessizce yanlış çalışan bir
    düğmeyi kabul etmek bu depoda düzeltilmiş bir hatadır (`heisenberg_step` +
    Ising). Panel `npml`i ve `Δx`'i yine de basar — değerleri koşumdan,
    `Window`dan gelir.

    `ratio` TÜRETİLMEZ, BEYAN EDİLİR ve yükleyici onu `radii` ile karşılaştırır.
    Türetilseydi (ör. `radii[1]/radii[0]`) yamuk bir merdiven sessizce kabul
    edilir ve `p = log(d_i/d_{i+1})/log r` bir MERTEBE OLMAYAN bir sayı
    döndürürdü.
    """

    name: str
    radii: tuple[float, ...]
    ratio: float
    courant: float
    step_factor: float
    wavelength_per_radius: float
    sgf: float
    detectors: int
    pad: int
    window_fn: str
    peak_floor: float
    eps_r: float = 1.0
    margin: int = 3


#: Sevk edilen iki AYAR. `engine/grav/metric.py::GAUGE_NAMES` ile AYNI liste, aynı
#: katman ayrımıyla tekrar edilir (`EM_WINDOW_FNS`in gerekçesinin aynısı):
#: asıl otorite motordadır ve iki gövdeyi o taşır, ama `scenario` katmanı
#: `engine`i import EDEMEZ. Yükleyici burada yalnızca "böyle bir ayar var mı"
#: diye bakar; ayarın hangi çekirdeği seçtiği motorda kalır. İki denetim ayrı
#: tür fırlatır (`ScenarioError` ile `InvalidInputError`).
GRAV_GAUGES = ("schwarzschild", "isotropic")

#: Sevk edilen N-cisim integratörleri — `engine/grav/nbody.py::NBODY_METHODS`
#: ile aynı liste, `GRAV_GAUGES` ile aynı katman gerekçesiyle.
GRAV_NBODY_METHODS = ("DOP853", "RK45", "Radau", "LSODA")


@dataclass(frozen=True)
class GravScenario:
    """Yerçekimi ve uzay-zaman — jeodezik çıpalar + Newtonyen N-cisim.

    PYDANTIC DEĞİL: bu dosyanın kendi dokstringi *hiçbir şey import etmediğini*
    söyler ve bu depoda `pydantic` KURULU DEĞİLDİR (ölçüldü: `import pydantic`
    → `ModuleNotFoundError`, ağaç genelinde 0 eşleşme). Bu sınıf da öteki yedi
    `*Scenario` gibi düz bir `@dataclass(frozen=True)`tir.

    TOHUM YOKTUR VE BU BİR EKSİKLİK DEĞİL BİR BEYANDIR: bu modülde RNG hiç
    kullanılmaz (kareleme deterministtir, `solve_ivp` deterministtir, BVP
    `least_squares` sabit bir başlangıç tahmininden koşar). `EMScenario`nun
    gerekçesinin aynısı: tekrarlanabilirliğin karşılığı **pencerenin tam
    beyanıdır** ve bu yüzden aşağıdaki alanların hepsi ZORUNLUDUR,
    varsayılanı olan yoktur — Richardson çiftleri, `p`/`b` basamakları,
    N-cisim toleransları ve integratör adı dahil. Bu depoda bir hükmün kendi
    penceresinin dışına taşınması ON BİRDEN FAZLA kez ölçüldü.

    BİRİM DÖNÜŞÜMÜ YOKTUR, çünkü çevrilecek bir şey yoktur: `G = c = 1` bir
    BİRİM SEÇİMİDİR (Modül 4'ün `J = k_B = 1`'i ve Modül 5'in
    `c = ε₀ = μ₀ = 1`'i ile aynı statüde). `mass` bu yüzden kilogram DEĞİLDİR
    ve `richardson`ın basamakları metre değildir; ölçülen büyüklüklerin hepsi
    (`2π(2+2γ−β)`, `πλ²(3+e²/6)`, `1+γ`, `|E|^{3/2}T/(G²m^{5/2})`) zaten
    boyutsuzdur. Bu, `SpinScenario`nun "dönüşümün yokluğu bir unutma değil bir
    ölçüm sonucudur" gerekçesiyle aynı ailedir.

    `gamma` VE `beta` BİRER PENCERE DÜĞMESİ DEĞİL, BİRER KURAM SEÇİCİDİR.
    `γ = β = 1` genel göreliliktir (`λ = 2+2γ−β = 3`); **herhangi başka değer
    GR DEĞİLDİR** ve panel bunu kendi satırında öyle yazar
    (`Provenance.theory_selector`). Yükleyici GR olmayan bir seçimi
    REDDETMEZ — reddetseydi §4.3'ün hata çubuğu makinesi (`λ = 0` koşusu)
    senaryo yolundan hiç görünmezdi.

    `lam` BİR ALAN DEĞİLDİR ve senaryoda bir anahtar da değildir: tek kaynak
    `(gamma, beta)`dır ve `λ` motorda türetilmiş bir `@property`dir. İki alan
    aynı şeyi belirlerse sessiz bir kazanan doğar ve o kazanan adlandırılmamış
    yeni bir pencere düğmesidir.

    `richardson` VE `deflection_richardson` AYRI ALANLARDIR, ve bu bir tekrar
    değil bir ÖLÇÜM SONUCUDUR: iki borunun optimum çifti aynı yerde değildir
    (kareleme (3e5, 1e6)'da 10.01 dürüst hane verir, bükülme borusu `b`
    cinsinden çalışır ve ölçeği tamamen başkadır). Tek alan paylaşsalardı
    biri için seçilen çift ötekinin hükmünü sessizce penceresinin dışına
    taşırdı.
    """

    name: str
    gauge: str
    gamma: float
    beta: float
    mass: float
    eccentricity: float
    richardson: tuple[float, float]
    second_order_p: float
    deflection_richardson: tuple[float, float]
    trajectory_p: float
    trajectory_points: int
    nbody_rtol: float
    nbody_atol: float
    nbody_method: str


@dataclass(frozen=True)
class BridgeScenario:
    """Modüller arası yokluk haritası + tabanı ölçülen tek kenarın (`hf -> md`) tabanı (Modül 7).

    (Modül 7 kapandığında `hf -> md` TEK gerçek köprüydü; 22.09.2026'dan beri
    `md -> tdse`, 24.09.2026'dan beri `tb -> md` de GERCEK_KOPRU'dur. Tabanı
    ölçülen kenar yine yalnız `hf -> md`dir ve bu senaryonun penceresi onundur.)

    PYDANTIC DEĞİL — `GravScenario` ile aynı gerekçe: bu depoda `pydantic`
    kurulu değildir (ölçüldü, Task 5'te yeniden: `import pydantic` →
    `ModuleNotFoundError`, ağaçta 0 eşleşme) ve bu dosya hiçbir şey import
    etmez. Öteki sekiz `*Scenario` gibi düz bir `@dataclass(frozen=True)`.

    BU SENARYO BİR ÇIPA KURMAZ. Kova tablosu YAML'dan gelmez, koddan
    türetilir (`qlab.bridge_map`); senaryonun taşıdığı tek şey KENAR
    TABANININ PENCERESİDİR — YOL B'nin `N`i, periyot sayısı, genlikleri ve
    A2 probunun ikizi (`a_ikiz`). Hepsi ZORUNLUDUR, varsayılanı olan alan
    YOKTUR (`EMScenario`/`GravScenario` gerekçesi: pencereyi yazmayı unutan
    bir senaryo sessizce "tekrarlanabilir" görünürdü; bu depoda bir hükmün
    penceresinin dışına taşınması ON BİRDEN FAZLA kez ölçüldü ve bu modülde
    spec'in bir sayısı daha tam bu yüzden yeniden üretilemedi — 6-31G'nin
    `ω_A`sı, `r_e` yazılmadığı için).

    `a` BİR ALAN DEĞİLDİR ve YAML'da bir anahtar da değildir: A1'in
    fizik-kapatma merdiveni ÇAĞRILANIN tasarım kararıdır
    (`bridge_obs.A1_MERDIVENI`), çağıranın düğmesi değil — Modül 6'nın
    "fiziği kapatma noktasını çağrılan belirler" sözleşmesi. `a_ikiz` ise
    A2'nin HANGİ ikiz olduğunu söyler ve bu bir kaçak değil sınavın
    kendisidir: ikiz değişince taban değişmelidir (ölçüldü: `a = 0.5`te
    sevk edilenin 0.1668'i).

    TOHUM YOKTUR: YOL B deterministtir (velocity-Verlet + `polyfit`), RNG
    hiç kullanılmaz; tekrarlanabilirliğin karşılığı pencerenin tam beyanıdır.
    BİRİM DÖNÜŞÜMÜ tek sınırda: taban cm⁻¹ raporlanır, boru atomik birimde
    koşar (`HARTREE_TO_WAVENUMBER`, `observable/bridge_obs.py`).
    """

    name: str
    a_ikiz: float
    n_bolme: int
    periyot_sayisi: int
    genlikler: tuple[float, ...]
