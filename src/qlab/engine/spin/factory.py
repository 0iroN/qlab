"""Senaryo -> spin taramasi. PANEL BEYANININ TEK KAYNAGI BURASIDIR.

`run.py::_run_spin` bu metinlerin HICBIRINI yeniden yazmaz; yalnizca
`Provenance.as_lines()` ciktisini basar. Gerekcesi bu depoda OLCULEREK
ogrenildi: ayni beyan blogu bir kez `scf.py` ve `tb/factory.py`de birebir
KOPYALANMISTI ve biri duzeltilip digeri unutuldugunda panel iki farkli sey
iddia ediyordu (bkz. tests/hf/test_hf_scenario.py). Buradaki kopya sayisi
`tests/spin/test_spin_senaryo.py::test_BEYAN_TEK_KAYNAKTAN_geliyor` ile
kilitlidir: her beyan cumlesi `src/` agacinda TAM BIR KEZ gecer.

IKI ADIMLI FABRIKA -- VE NEDEN. `build_tb`/`build_downfold` (model, provenance)
ciftini TEK cagrida dondurur; burada olmuyor, cunku panelin ZORUNLU ornekleme
tanisi (binning platosu, tau_int) `observable/spin_obs.py`den gelir ve
`engine` katmani `observable`i IMPORT EDEMEZ (import-linter sozlesmesi:
run -> viz -> observable -> engine -> scenario). Bu yuzden desen `md/factory.py`
ninkidir: `build_spin` fizigi kosar, `provenance_for` olculmus tanilari ALIR ve
beyani kurar. Sayilar disaridan gelir, METIN burada kalir.

DAIRESELLIK BEYANI YENIDEN OLCULDU (23.08.2026, bu depoda). Yol haritasinin
"demir Curie sicakligi 1043 K" hedefi bir TAUTOLOJIDIR: Ising modeli tek bir
enerji olcegi tasir ve `T_c` ile `J` DOGRU ORANTILIDIR. Onsager'in tam
cozumu
`kT_c/J = 2/ln(1+sqrt(2)) = 2.269185314` konunca

    1043.0000 K icin gereken J = 39.608394 meV
    o J geri konulunca      T_c = 1043.0000 K   (fark 0.000e+00)

yani sayi bir tahmin degil, secilen parametrenin geri okunmasidir. Ayni kadran
her malzemeyi verir: Ni (627 K) -> 23.810607 meV, Co (1388 K) -> 52.709924 meV,
Gd (292 K) -> 11.088831 meV. (`k_B = 8.617333262e-2 meV/K`, CODATA 2018.)
Modul 3F'nin silisyum bant araligi hedefiyle AYNI AILE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.spin.lattice import SpinState
from qlab.engine.spin.metropolis import SamplingResult, run_metropolis
from qlab.engine.spin.models import HeisenbergModel, IsingModel, SpinModel
from qlab.engine.spin.wolff import run_wolff
from qlab.errors import InvalidInputError
from qlab.provenance import Provenance
from qlab.scenario.schema import SpinScenario

#: Plato olusmamis bir hata cubugunun panel isareti. TEK bir sabit olarak
#: durur ve `run.py` onu BURADAN import eder; iki yerde elle yazilsaydi biri
#: degistiginde tablo ile beyan satiri farkli sey soylerdi -- bu modulun
#: dokstringindeki kopya dersinin aynisi, kelime olceginde.
ALT_SINIR_ISARETI = "ALT SINIR"

_SPIN_YAKLASIMLAR: tuple[str, ...] = (
    "klasik spin: her site bir vektordur, kuantum operatoru DEGIL; "
    "sifir noktasi dalgalanmasi ve komutasyon bagintilari YOKTUR",
    "kare kafes, periyodik sinir kosulu",
    "k_B = 1 BIR BIRIM SECIMIDIR, FIZIK DEGIL: sicaklik ENERJI birimindedir "
    "ve bu modulun hicbir ciktisi kelvin tasimaz. J (coupling) bir GIRDIDIR, "
    "1'e kilitli DEGILDIR: Onsager'in kT_c/J'siyle karsilastirilacak sayi T "
    "degil kT/J = T/coupling'dir ve Yontem satirinda basilir; chi = L^2 "
    "var(|m|)/T ENERJI^-1 birimindedir: (J, T) -> (lJ, lT) altinda <|m|>, "
    "Binder U ve C degismez (l ikinin kuvvetiyken bit bit), chi 1/l ile "
    "olceklenir, coupling'ler arasi karsilastirilacak sayi J*chi'dir",
    "denge ornekleme: raporlanan her sayi bir DENGE ortalamasidir; "
    "histerezis, tavlama ve butun denge-disi surecler kapsam disindadir",
    "Ising checkerboard Metropolis T/J -> sonsuzda DONAR, iliskisiz tabana "
    "gitmez: kabul ~1 olunca her yari-adim bir alt-orguyu butunuyle cevirir, "
    "<|m|> baslangici okur ve karisma suresi T/J ile buyur. O rejimde binning "
    "platosu OLUSMAZ ve panel 'yakinsadi' DEMEZ -- olculdu (27.09.2026, "
    "L = 8/16/32, T/J = 33 ... 2e9, 1000-16000 olcum): tabanin 1.5 katini "
    "asan 128 satirin 128'inde plato yok. Heisenberg donmuyor (T/J >= 1e3'te "
    "tabanda, plato var)",
)

_SPIN_IHMAL: tuple[str, ...] = (
    "gercek kristal yapisi: koordinasyon sayisi 4'te sabitlenmistir, "
    "orgu geometrisi ve kafes sabiti girdide YOKTUR",
    "spin-yorunge etkilesimi ve manyetokristal anizotropi",
    "kuantum dalgalanmalari -- yani S = 1/2 Heisenberg antiferromanyetinin "
    "butun fizigi",
    "3B: bu modul 2B'dir ve 3B BILINCLI OLARAK kapsam disindadir, cunku 3B "
    "Ising'in tam cozumu yoktur ve cipa Onsager yerine literaturun sayisal "
    "degerlerine dayanirdi",
    "korelasyon uzunlugu bu senaryoda RAPORLANMAZ: ikinci moment tahmincisi "
    "tek konfigurasyonda 41 KAT yayiliyor (T_c, L = 32, 5 tohum: 3.644 ... "
    "150.562) ve iliskisiz bir kafeste tohumlarin ~%38'i SAHTE bir xi "
    "donduruyor, yani None hikayenin yalnizca yarisidir",
)

# Kritik usun OLCULMUS beyani. Sayilar tests/spin/test_cipalar.py'de
# kilitlidir ve oradan ALINTILANIR; bu senaryo bunlari YENIDEN OLCMEZ.
_SPIN_BEKLENEN_HATA = (
    "kritik us sonlu-boyut duzeltmesi tasir ve bu senaryo onu OLCMEZ "
    "(olcum tests/spin/test_cipalar.py'dedir): L = 8...32, Wolff, T_c "
    "penceresinde beta/nu = 0.122941 +- 0.003225 olculdu; tam deger 0.125 "
    "(sapma -0.64 sigma, yani belirsizligin icinde) ve ORTALAMA ALAN "
    "(beta/nu = 1.0) 272.0 kat marjla dislandi. Kalan sapma sifirlanmaz: "
    "L alt kumeleri ayri ayri uydurulunca 0.121346 / 0.129350 / 0.123540 "
    "cikiyor, yani log-log dogrusu TAM DOGRU DEGILDIR. Binder kesisiminde "
    "de ayni yonde bir sonlu-boyut kaymasi var: uc tohum blogu havuzlaninca "
    "kT/J = 2.266504 +- 0.000642, Onsager'den -4.18 sigma."
)

_DAIRESELLIK_BEYANI = (
    "T_c'nin SI birimlerindeki degeri bir tahmin DEGILDIR: T_c, J ile DOGRU "
    "ORANTILIDIR ve J girdidir. "
    "Bu modul kT_c/J oranini ve kritik usleri olcer; ikisi de J'den, "
    "kafes sabitinden ve birim seciminden BAGIMSIZDIR. OLCULDU: demirin "
    "1043 K'i icin gereken J = 39.608394 meV geri konuldugunda T_c BIREBIR "
    "1043.0000 K okunuyor (fark 0.000e+00) -- yani yol haritasinin hedefi "
    "bir dogrulama degil, girilen parametrenin geri okunmasiydi. Ayni kadran "
    "Ni'yi (627 K -> 23.810607 meV), Co'yu (1388 K -> 52.709924 meV) ve "
    "Gd'yi (292 K -> 11.088831 meV) de verir. BU MODUL HICBIR MALZEMENIN "
    "T_c'sini IDDIA ETMEZ."
)


@dataclass(frozen=True, eq=False)
class SpinScanResult:
    """Bir sicaklik taramasinin ham sonucu -- gozlemlenebilir HESAPLANMAMIS.

    `samples` senaryodaki `temperatures` ile AYNI SIRADADIR. Gozlemlenebilirler
    burada hesaplanmaz cunku `engine` katmani `observable`i import edemez;
    onlar `viz/spin_frames.py`de kurulur.

    `eq=False`: alanlar ndarray tasiyan dataclass'lardir; uretilen `__eq__`
    dizi dondurur ve `__hash__` `TypeError` verir. Depo bu tuzagi
    `engine/hf/integrals.py`, `engine/md/state.py` ve `spin/lattice.py`de
    zaten odedi.
    """

    scenario: SpinScenario
    samples: tuple[SamplingResult, ...]


def _model_kur(scenario: SpinScenario) -> SpinModel:
    if scenario.model == "heisenberg":
        return HeisenbergModel(step=scenario.heisenberg_step)
    return IsingModel()


def _coupling_kapisi(scenario: SpinScenario) -> None:
    if not (np.isfinite(scenario.coupling) and scenario.coupling > 0.0):
        raise InvalidInputError(
            f"spin senaryosu sonlu ve pozitif coupling ister, verilen: "
            f"{scenario.coupling}. J <= 0 bu modulun KAPSAM DISIDIR (dort cipanin "
            "dordu de ferromanyetik); motor J < 0'i kosar ama DOGRULANMAMISTIR."
        )


def build_spin(scenario: SpinScenario) -> SpinScanResult:
    """Her sicaklikta bir zincir kosar. TOHUM AKISLARI BAGIMSIZDIR.

    BIRIM DONUSUMU YOKTUR ve bu bir eksiklik degil bir OLCUM sonucudur:
    `TBScenario` eV -> hartree cevirir cunku TB literaturu eV'de yayinlanir;
    burada cevrilecek bir sey yoktur, cunku `k_B = 1` bir birim secimidir
    (sicaklik ENERJI birimindedir; `coupling` bir GIRDIDIR, kT/J = T/coupling)
    ve olculen buyuklukler (kT_c/J, kritik usler) zaten boyutsuzdur.

    KAPSAM KAPISI PYTHON YOLUNDA DA: yukleyicinin `J <= 0` kapsam karari
    (`loader.py::_load_spin_scenario`) `build_spin` ve `provenance_for`
    girislerinde de uygulanir (`_coupling_kapisi`), sonluluk dahil;
    `SpinScenario(coupling=-1)` eskiden panele kadar kosuyor ve tekbicim
    m'den AFM icin anlamsiz bir Binder basiyordu. Motor (`run_metropolis`)
    `J < 0`i BILEREK kabul etmeye devam eder.

    HER SICAKLIK AYRI BIR AKIS ALIR. Tek bir `default_rng(seed)` butun
    sicakliklarda yeniden kullanilsaydi tarama noktalari AYNI rastgele diziyi
    goruntusunden gecirirdi; egri sahte bir duzgunluk kazanir ve komsu iki
    noktanin farki kendi gurultusunden kucuk gorunurdu. `SeedSequence.spawn`
    tek bir tohumdan bagimsiz akislar uretir ve tekrarlanabilirligi bozmaz --
    ayni `seed` her zaman ayni akis listesini verir.
    """
    _coupling_kapisi(scenario)
    model = _model_kur(scenario)
    akislar = np.random.SeedSequence(scenario.seed).spawn(
        len(scenario.temperatures)
    )

    sonuclar: list[SamplingResult] = []
    for sicaklik, akis in zip(scenario.temperatures, akislar):
        rng = np.random.default_rng(akis)
        durum = SpinState.uniform(
            L=scenario.lattice_size, n_component=model.n_component
        )
        if scenario.algorithm == "wolff":
            sonuclar.append(
                run_wolff(
                    durum,
                    temperature=sicaklik,
                    coupling=scenario.coupling,
                    field=scenario.field,
                    rng=rng,
                    n_sweep=scenario.n_sweep,
                    n_burn=scenario.n_burn,
                )
            )
        else:
            sonuclar.append(
                run_metropolis(
                    durum,
                    model,
                    temperature=sicaklik,
                    coupling=scenario.coupling,
                    field=scenario.field,
                    rng=rng,
                    n_sweep=scenario.n_sweep,
                    n_burn=scenario.n_burn,
                )
            )

    return SpinScanResult(scenario=scenario, samples=tuple(sonuclar))


def provenance_for(
    scenario: SpinScenario,
    *,
    plateau_reached: np.ndarray,
    tau_int: np.ndarray,
) -> Provenance:
    """Olculmus ornekleme tanilarindan panel beyanini kurar.

    `plateau_reached` ve `tau_int` senaryodaki `temperatures` ile AYNI
    SIRADADIR ve `viz/spin_frames.py`den gelir.

    `converged` OLCULEBILIR bir seye baglanir -- ve bu modulde o sey SCF'in
    durmasi ya da bir izgaranin yakinsamasi DEGILDIR, cunku Monte Carlo'da
    boyle bir olcut yoktur. Baglandigi sey su: RAPORLANAN HER HATA CUBUGU
    GERCEKTEN BIR HATA CUBUGU MU? Binning egrisi plato yapmamis bir sicaklikta
    donen sayi hatanin kendisi degil ALT SINIRIDIR; panel o zaman "yakinsadi"
    diyemez. Yani buradaki tik "hesap bitti" degil, "raporladigim belirsizlik
    belirsizligin kendisidir" demektir.

    BAYRAK ALGORITMANIN DEGIL KOSUMUN OZELLIGIDIR ve bu olculdu: Metropolis
    L = 8 / 4000 olcumde 24/24, L = 16 / 8000'de 23/24 plato yapiyor,
    L = 32 / 8000'de 0/24. "Metropolis plato yapmaz" bir ALGORITMA cumlesi
    olarak YANLIStir.
    """
    _coupling_kapisi(scenario)  # Yontem satiri T/coupling boler (0'a, eksiye)
    plato = np.asarray(plateau_reached, dtype=bool)
    tau = np.asarray(tau_int, dtype=float)
    n_t = len(scenario.temperatures)
    if plato.shape != (n_t,) or tau.shape != (n_t,):
        raise ValueError(
            f"tani dizileri {n_t} sicaklikla ayni uzunlukta olmali, verilen: "
            f"plateau_reached {plato.shape}, tau_int {tau.shape}"
        )

    eksikler = [
        f"{t:.3f}" for t, p in zip(scenario.temperatures, plato) if not p
    ]
    ornekleme = (
        f"{n_t} sicaklik x {scenario.n_sweep} olcum "
        f"(+{scenario.n_burn} burn-in sweep), algoritma {scenario.algorithm}, "
        f"L = {scenario.lattice_size}, tohum {scenario.seed}; "
        f"tau_int araligi {tau.min():.2f} ... {tau.max():.2f} "
        f"(binning'den turetildi, naif SEM hicbir yerde raporlanmaz); "
    )
    if eksikler:
        ornekleme += (
            f"binning platosu {len(eksikler)}/{n_t} sicaklikta OLUSMADI "
            f"(T = {', '.join(eksikler)}) -- o satirlarin hata cubugu "
            f"hatanin kendisi DEGIL bir {ALT_SINIR_ISARETI}IDIR"
        )
    else:
        ornekleme += (
            f"binning platosu {n_t}/{n_t} sicaklikta olustu -- hicbir hata "
            "cubugu alt sinir degil"
        )

    return Provenance(
        method=(
            f"spin Monte Carlo ({scenario.name}); {scenario.model} modeli, "
            f"{scenario.algorithm}, L = {scenario.lattice_size}, "
            f"{n_t} sicaklik, J = {scenario.coupling:g} (girdi), "
            f"h = {scenario.field:g}; kT/J = T/J: "
            + ", ".join(f"{t / scenario.coupling:.6g}" for t in scenario.temperatures)
        ),
        approximations=_SPIN_YAKLASIMLAR
        + (
            f"sonlu L: bu kosum L = {scenario.lattice_size} kullanir ve "
            "termodinamik limit ALINMAZ; sonlu kafeste keskin bir faz gecisi "
            "YOKTUR, gorunen sey yumusatilmis bir gecidir",
        ),
        neglected=_SPIN_IHMAL,
        expected_error=_SPIN_BEKLENEN_HATA,
        converged=bool(plato.all()),
        circularity=_DAIRESELLIK_BEYANI,
        sampling=ornekleme,
        diagnostics={
            "n_temperature": float(n_t),
            "n_sweep": float(scenario.n_sweep),
            "n_burn": float(scenario.n_burn),
            "tau_int_max": float(tau.max()),
            "plateau_count": f"{int(plato.sum())}/{n_t}",
            "algorithm": scenario.algorithm,
            "model": scenario.model,
        },
    )
