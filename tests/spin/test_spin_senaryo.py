"""`kind: spin` senaryosu, panel beyanı ve uçtan uca koşum.

SINIF: TRANSKRİPSİYON DENETİMİ. Bu dosyanın hiçbir testi bir fizik iddiası
taşımaz; senaryo sınırının (YAML → dataclass → motor → gözlemlenebilir →
panel) doğru bağlandığını ve panelin **beş zorunlu beyanı** bastığını ölçer.
Modülün fiziksel çıpaları başka yerdedir: Onsager ve kritik üs
`test_cipalar.py`de, Mermin-Wagner `test_mermin_wagner.py`de, çapraz kontrol
`test_wolff.py`de.

TEK İSTİSNA — VE O DA FİZİK DEĞİL, KOŞUM ÖZELLİĞİ: plato bayrağının sevk
edilen senaryoda **hangi sıcaklıklarda** düştüğü. O bir Ising iddiası değil,
"bu koşum kendi otokorelasyonunu çözecek kadar uzun mu" sorusunun cevabıdır
ve aşağıda 24 tohumla ölçülmüştür.

BU DOSYANIN EN ÖNEMLİ TESTİ BİR SAYI DEĞİL BİR SAYIM DENETLER:
`test_BEYAN_TEK_KAYNAKTAN_geliyor`. Ölçülmüş gerekçe bu depodadır — aynı
beyan bloğu bir kez `scf.py` ile `factory.py`de birebir kopyalanmıştı ve biri
düzeltilip öteki unutulduğunda panel iki farklı şey iddia ediyordu (bkz.
`tests/hf/test_hf_scenario.py::test_scan_and_single_point_share_one_provenance_source`
ve `tests/tb/test_downfold_senaryo.py::test_BEYAN_TEK_KAYNAKTAN_geliyor`).

===========================================================================
ÖLÇÜLEN SAYILAR — HEPSİ 23.08.2026'DA, BU DEPODA, AŞAĞIDAKİ PENCEREDE
===========================================================================

**Sevk edilen senaryo** (`scenarios/spin_tarama.yaml`): Ising · Metropolis ·
L = 16 · T = (1.6, 2.0, 2.2, 2.4, 2.8, 3.4) · `n_burn = 200` ·
`n_sweep = 1000` · tohum 20260823 · soğuk (tekbiçim) başlangıç. **0.56 s.**

Plato bayrağının tohum yayılımı — `build_spin` boru hattının kendisiyle,
24 tohumla (20260823 + 1000·k) ölçüldü:

    T      plato
    1.6    23/24
    2.0    16/24     <- GEÇİŞ BÖLGESİ, testlerde KİLİTLENMEZ
    2.2     0/24     <- her tohumda ALT SINIR
    2.4     0/24     <- her tohumda ALT SINIR
    2.8    13/24     <- GEÇİŞ BÖLGESİ, testlerde KİLİTLENMEZ
    3.4    24/24
    tau_int aralığı 0.68 … 6.44

Yani "ALT SINIR" işaretinin İKİ YÖNÜ de aynı sevk edilmiş senaryodan
kilitlenebiliyor: T = 2.2/2.4 işareti ALMAK, T = 3.4 ALMAMAK zorunda. İkisi
de 0/24 ve 24/24 ile desteklidir; T = 1.6 (23/24) ve geçiş bölgesindeki iki
sıcaklık bilinçli olarak assert edilmez — sevk edilen tohumda değerleri
sabittir ama hüküm tohum gürültüsüne açıktır ve o gürültüyü fiziğe yazmak bu
depoda üç kez düzeltilmiş bir hatadır.

**Ucuz test senaryoları** (uçtan uca koşum dışındaki her testin kullandığı):

    UCUZ_METRO  metropolis L=8 T=(3.0,4.0) ns=800 nb=200 -> 0.13 s
                12/12 tohumda HER sıcaklıkta plato -> `converged is True`
    UCUZ_WOLFF  wolff      L=8 T=(1.8,3.0) ns=800 nb=200 -> 0.05 s
                `energy_series is None` -> ısı sığası ÖLÇÜLMEDİ

`UCUZ_METRO`nun 12/12'si önemlidir: `converged is True` dalını sevk edilen
senaryo HİÇBİR ZAMAN göstermez (orada bayrak ✗'tir), yani o dal ancak burada
kilitlenebilir.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import qlab.engine.spin.factory as fabrika
from qlab.engine.spin.factory import (
    ALT_SINIR_ISARETI,
    build_spin,
    provenance_for,
)
from qlab.errors import InvalidInputError, ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import SpinScenario
from qlab.viz.spin_frames import build_spin_payload

PROJE_KOKU = Path(__file__).resolve().parents[2]
SEVK_EDILEN = PROJE_KOKU / "scenarios" / "spin_tarama.yaml"
SEVK_YAML = SEVK_EDILEN.read_text(encoding="utf-8")

# Onsager'in sayisi. BU DOSYADA YALNIZCA "YAML'da GECMEMELI" iddiasi icin
# kullanilir; hicbir kosuma girdi degildir.
ONSAGER = "2.269185"

UCUZ_METRO = SpinScenario(
    name="ucuz metropolis denetimi",
    model="ising",
    algorithm="metropolis",
    lattice_size=8,
    temperatures=(3.0, 4.0),
    n_sweep=800,
    n_burn=200,
    seed=20260823,
)

UCUZ_WOLFF = dataclasses.replace(
    UCUZ_METRO,
    name="ucuz wolff denetimi",
    algorithm="wolff",
    temperatures=(1.8, 3.0),
)

# Beyan listesinin CUMLE degil TERIM olan girdisi YOK; hepsi tam cumledir,
# dolayisiyla `test_downfold_senaryo.py`deki STANDART_TERIMLER muafiyetinin
# bir karsiligi burada gerekmiyor. Bu bir varsayim degil, asagidaki sayimin
# kendisi tarafindan dogrulaniyor: muafiyet olmadan sayim 1/0 tutuyor.


def _yaz(tmp_path, metin: str) -> Path:
    yol = tmp_path / "spin.yaml"
    yol.write_text(metin, encoding="utf-8")
    return yol


def _kaynak_metni(yol: Path) -> str:
    """Dosya metni, BİTİŞİK DİZE SABİTLERİ BİRLEŞTİRİLMİŞ hâlde.

    Gerekçesi ölçülmüş bir tuzak (`tests/tb/test_downfold_senaryo.py`):
    `"birinci parca "\\n"ikinci parca"` kaynakta İKİ parçadır ama çalışma
    zamanında TEK dizedir. Ham metinde aramak 0 eşleşme verirdi ve "kopya
    yok" diye okunurdu — yani test kendi iddiasının tersini kanıtlardı.
    """
    return re.sub(r'"\s*\n\s*"', "", yol.read_text(encoding="utf-8"))


def _tum_beyanlar() -> tuple[str, ...]:
    return (
        fabrika._SPIN_YAKLASIMLAR
        + fabrika._SPIN_IHMAL
        + (fabrika._SPIN_BEKLENEN_HATA, fabrika._DAIRESELLIK_BEYANI)
    )


def _panel(senaryo: SpinScenario) -> dict[str, str]:
    """Bir senaryoyu koşturup panel satırlarını sözlük olarak döndürür."""
    yuk = build_spin_payload(build_spin(senaryo))
    kaynak = provenance_for(
        senaryo, plateau_reached=yuk.plateau_reached, tau_int=yuk.tau_int
    )
    return dict(kaynak.as_lines())


# ---------------------------------------------------------------------------
# YAPISAL testler -- hiç sweep koşmazlar, milisaniyedirler.
# ---------------------------------------------------------------------------


def test_SEVK_EDILEN_senaryo_yukleniyor_ve_BIRIM_DONUSUMU_YOK():
    """Depodaki `scenarios/spin_tarama.yaml` gerçekten yükleniyor.

    BİRİM DÖNÜŞÜMÜ YOKLUĞU DA BİR İDDİADIR ve burada kilitlenir: `TBScenario`
    eV → hartree çevirir (TB literatürü eV'de yayınlanır), bu senaryo ise
    HİÇBİR ŞEY çevirmez çünkü `J = k_B = 1` bir birim seçimidir ve ölçülen
    büyüklükler zaten boyutsuzdur. YAML'daki 2.2 dataclass'ta da 2.2'dir.
    """
    senaryo = load_scenario(SEVK_EDILEN)
    assert isinstance(senaryo, SpinScenario)
    assert senaryo.model == "ising"
    assert senaryo.algorithm == "metropolis"
    assert senaryo.lattice_size == 16
    assert senaryo.temperatures == (1.6, 2.0, 2.2, 2.4, 2.8, 3.4)
    assert senaryo.n_sweep == 1000
    assert senaryo.n_burn == 200
    assert senaryo.seed == 20260823
    assert senaryo.coupling == 1.0
    assert senaryo.field == 0.0


def test_ONSAGER_SEVK_EDILEN_YAML_DA_HIC_GECMIYOR():
    """Sıcaklık ızgarası YUVARLAK sayılardan kurulu — çıpanın ön koşulu.

    `2.269185314` senaryoya girdi olsaydı "geçiş T_c'de görünüyor" cümlesi
    dairesel olurdu: ızgaranın kendisi o sayıyı taşırdı. Izgara
    (1.6 … 3.4) yuvarlak adımlardan kuruludur ve geçiş 2.2 ile 2.4 arasında
    KENDİLİĞİNDEN doğar. Bu test o dairesellik kapısını kapatır.
    """
    assert ONSAGER not in SEVK_YAML
    assert "2.269" not in SEVK_YAML
    senaryo = load_scenario(SEVK_EDILEN)
    assert all(round(T, 1) == T for T in senaryo.temperatures)


def test_VARSAYILANLAR_yalnizca_SEKIZ_anahtar_zorunlu(tmp_path):
    """`coupling`, `field` ve `heisenberg_step` isteğe bağlıdır; varsayılanları
    şemadan gelir ve yükleyici kendi kopyasını taşımaz. `seed` ise ZORUNLUDUR
    ve bir sonraki test onu ayrıca kilitler."""
    senaryo = load_scenario(
        _yaz(
            tmp_path,
            "kind: spin\nname: kisa\nmodel: ising\nalgorithm: wolff\n"
            "lattice_size: 6\ntemperatures: [3.0]\nn_sweep: 40\n"
            "n_burn: 5\nseed: 7\n",
        )
    )
    assert senaryo.coupling == 1.0
    assert senaryo.field == 0.0
    assert senaryo.heisenberg_step == 0.6


def test_TOHUM_ZORUNLU_ve_VARSAYILANI_YOK(tmp_path):
    """Tohumsuz bir Monte Carlo senaryosu KABUL EDİLMEZ.

    Varsayılan bir tohum, tohumu yazmayı unutan bir senaryonun sessizce
    "tekrarlanabilir" görünmesine yol açardı — ve o senaryo bir kez koşup
    bir daha aynı sayıyı vermeyebilirdi. Hata mesajı eksik anahtarı ADIYLA
    söylemeli.
    """
    with pytest.raises(ScenarioError, match="seed"):
        load_scenario(_yaz(tmp_path, SEVK_YAML.replace("seed: 20260823\n", "")))


def test_BILINMEYEN_kind_listesi_spin_iceriyor(tmp_path):
    """Hata mesajı geçerli türleri sayar; `spin` orada görünmeli — aksi
    hâlde kullanıcı sevk edilmiş bir türü keşfedemez."""
    with pytest.raises(ScenarioError, match="spin") as bilgi:
        load_scenario(_yaz(tmp_path, "kind: muz\nname: x\n"))
    assert "muz" in str(bilgi.value)


@pytest.mark.parametrize(
    "bozuk,desen",
    [
        ("model: ising", "model: potts"),
        ("algorithm: metropolis", "algorithm: gibbs"),
        ("lattice_size: 16", "lattice_size: 1"),
        ("lattice_size: 16", "lattice_size: 15"),
        ("lattice_size: 16", "lattice_size: onalti"),
        ("temperatures: [1.6, 2.0, 2.2, 2.4, 2.8, 3.4]", "temperatures: []"),
        ("temperatures: [1.6, 2.0, 2.2, 2.4, 2.8, 3.4]", "temperatures: [2.0, 2.0]"),
        ("temperatures: [1.6, 2.0, 2.2, 2.4, 2.8, 3.4]", "temperatures: [0.0]"),
        ("temperatures: [1.6, 2.0, 2.2, 2.4, 2.8, 3.4]", "temperatures: [-1.0]"),
        ("temperatures: [1.6, 2.0, 2.2, 2.4, 2.8, 3.4]", "temperatures: [.inf]"),
        ("temperatures: [1.6, 2.0, 2.2, 2.4, 2.8, 3.4]", "temperatures: [.nan]"),
        ("temperatures: [1.6, 2.0, 2.2, 2.4, 2.8, 3.4]", "temperatures: 2.0"),
        ("n_sweep: 1000", "n_sweep: 0"),
        ("n_burn: 200", "n_burn: -1"),
        ("coupling: 1.0", "coupling: 0.0"),
        ("coupling: 1.0", "coupling: -1.0"),
    ],
)
def test_BOZUK_senaryolar_REDDEDILIYOR(tmp_path, bozuk, desen):
    """Sessizce varsayılana düşmek YASAK: her biri ScenarioError vermeli.

    `lattice_size: 15` özellikle önemli — checkerboard Metropolis TEK L'de
    ayrıntılı dengeyi bozar ve bozulma SESSİZDİR (sonuç makul görünür).
    `coupling: -1.0` de öyle: Wolff'ta `p = 1 - exp(-2J/T)` negatife düşer,
    hiçbir bağ kurulmaz ve algoritma sonsuz sıcaklığı örnekler — ölçüldü,
    20 kat hata ve sıfır istisna (`engine/spin/wolff.py`).
    """
    with pytest.raises(ScenarioError):
        load_scenario(_yaz(tmp_path, SEVK_YAML.replace(bozuk, desen)))


def test_BILINMEYEN_anahtar_ve_EKSIK_anahtar_reddediliyor(tmp_path):
    fazla = SEVK_YAML + "kafes_boyu: 3\n"
    with pytest.raises(ScenarioError, match="kafes_boyu"):
        load_scenario(_yaz(tmp_path, fazla))
    eksik = SEVK_YAML.replace("temperatures: [1.6, 2.0, 2.2, 2.4, 2.8, 3.4]\n", "")
    with pytest.raises(ScenarioError, match="temperatures"):
        load_scenario(_yaz(tmp_path, eksik))


def test_HEISENBERG_ADIMI_ISING_ILE_verilemez(tmp_path):
    """Bağlı olmayan bir düğme sessizce yutulmaz.

    Ising'de öneri işaret çevirmedir ve `step` hiçbir şeyi değiştirmez
    (`models.py::IsingModel.propose` `rng`yi bile kullanmaz). Anahtar
    sessizce kabul edilseydi kullanıcı ayarladığını sanırdı.
    """
    with pytest.raises(ScenarioError, match="heisenberg_step"):
        load_scenario(_yaz(tmp_path, SEVK_YAML + "heisenberg_step: 0.3\n"))
    # Heisenberg ile GECERLI -- ve gercekten modele ulasiyor.
    senaryo = load_scenario(
        _yaz(
            tmp_path,
            SEVK_YAML.replace("model: ising", "model: heisenberg") + "heisenberg_step: 0.3\n",
        )
    )
    assert senaryo.heisenberg_step == 0.3
    assert fabrika._model_kur(senaryo).step == 0.3
    assert fabrika._model_kur(senaryo).n_component == 3


@pytest.mark.parametrize(
    "bozuk,desen,mesaj",
    [
        ("model: ising", "model: heisenberg", "wolff"),
        ("field: 0.0", "field: 0.3", "field"),
    ],
)
def test_WOLFF_SINIRLARI_YAML_TARAFINDA_da_yakalaniyor(tmp_path, bozuk, desen, mesaj):
    """Yükleyici ile motor AYNI koşulu ayrı ayrı denetler ve bu bir tekrar
    değil bir KATMAN AYRIMIDIR: biri YAML'ın hatasını (`ScenarioError`),
    öteki çağrının hatasını (`UnsupportedSystemError`) yakalar. Hata SWEEP
    KOŞMADAN önce verilir.
    """
    metin = SEVK_YAML.replace("algorithm: metropolis", "algorithm: wolff").replace(
        bozuk, desen
    )
    with pytest.raises(ScenarioError, match=mesaj):
        load_scenario(_yaz(tmp_path, metin))


def test_WOLFF_SINIRI_MOTORDA_DA_duruyor():
    """Yükleyici kaldırılsa motor hâlâ tutar — ama hata TÜRÜ değişir.

    Bu iki satır o farkı kilitler: yükleyici `ScenarioError`, motor
    `UnsupportedSystemError` fırlatır ve biri ötekinin yerine sessizce
    geçemez.
    """
    from qlab.engine.spin.lattice import SpinState
    from qlab.engine.spin.wolff import run_wolff
    from qlab.errors import UnsupportedSystemError

    with pytest.raises(UnsupportedSystemError):
        run_wolff(
            SpinState.uniform(L=4, n_component=1),
            temperature=3.0,
            coupling=1.0,
            field=0.3,
            rng=np.random.default_rng(1),
            n_sweep=5,
            n_burn=1,
        )


def test_BEYAN_TEK_KAYNAKTAN_geliyor():
    """BU DOSYANIN EN ÖNEMLİ TESTİ. Beyan metninin İKİ KOPYASI OLAMAZ.

    Ölçülmüş gerekçe bu depodadır: `scf.py` ile `factory.py` aynı bloğu
    birebir taşıyordu ve biri düzeltilip öteki unutulduğunda panel iki
    farklı şey iddia ediyordu. Kilit iki yönlüdür:

      * `src/` ağacında her cümle tam BİR kez — kaynakta kopya yok, panel
        iki farklı şey iddia edemez. Bu, `run.py::_run_spin`in beyan metnini
        kendisi yazmasını da YASAKLAR: yazsaydı sayım 2 olurdu.
      * `tests/` ağacında, BU DOSYA hariç, sıfır kez — ikinci bir test
        dosyası açıklama cümlesini elle yeniden yazamaz.

    STANDART TERİM MUAFİYETİ YOK, VE BU ÖLÇÜLDÜ: `test_downfold_senaryo.py`
    dört girdiyi muaf tutmak zorundaydı çünkü onlar cümle değil TERİMDİ
    ("Lowdin ortogonallestirmesi" gerçekten başka bir dosyada düz yazı
    olarak geçiyor). Buradaki on bir girdinin on biri de tam cümledir ve
    muafiyetsiz sayım 1/0 tutuyor — yani muafiyet listesi eklemek denetimi
    gereksiz yere gevşetmek olurdu.

    BU DOSYANIN MUAF OLMASI BİR SAPMA DEĞİL, KİLİDİN KENDİSİDİR: tam cümle
    aramak için cümleyi bir kez yazmak gerekir ve o tek kopya
    `test_MODULUN_ETIK_CEKIRDEGI_metinde_YAZILI`dadır. Kaynak metni
    değişirse İKİSİ BİRDEN kırmızı olur.
    """
    beyanlar = _tum_beyanlar()
    assert len(beyanlar) == 11, "beyan sayisi degisti; testi bilerek guncelleyin"
    bu_dosya = Path(__file__).resolve()

    for cumle in beyanlar:
        src_sayim = sum(
            _kaynak_metni(yol).count(cumle)
            for yol in (PROJE_KOKU / "src").rglob("*.py")
        )
        assert src_sayim == 1, f"src/ agacinda {src_sayim} kopya: {cumle[:60]!r}"

        test_sayim = sum(
            _kaynak_metni(yol).count(cumle)
            for yol in (PROJE_KOKU / "tests").rglob("*.py")
            if yol.resolve() != bu_dosya
        )
        assert test_sayim == 0, f"BASKA bir test dosyasinda kopya var: {cumle[:60]!r}"


def test_MODULUN_ETIK_CEKIRDEGI_metinde_YAZILI():
    """Anahtar kelime değil TAM CÜMLE aranır.

    Gerekçesi `test_tb_scenario.py`de ölçüldü: kelime bazlı bir test,
    bitişik dize sabitlerinin yalnızca İLK satırını değiştiren mutantları
    kaçırıyor. Aşağıdaki sekiz parça bu modülün etik çekirdeğidir ve
    sessizce yeniden yazılmamalıdır: ikisi birim seçimini ve sonlu kafesi
    (yani "burada keskin bir geçiş yok"u), ikisi 3B ile ξ'nin neden
    kapsam dışı olduğunu, ikisi kritik üsün sonlu-boyut kalıntısını, ikisi
    de daireselliğin kendisini yazıyor.

    Bu test bir totoloji DEĞİLDİR: sağ taraf `factory` modülünden okunur,
    sol taraftaki cümleler burada BAĞIMSIZ olarak yazılmıştır.

    ARANAN METİN PANELİN KENDİSİDİR, sabit listesi değil: `sonlu L` satırı
    çalışma zamanında kurulur (`lattice_size`ı okur) ve sabit listede
    GEÇMEZ. Yalnız sabitlere bakan bir sürüm o cümleyi hiç denetlemezdi.
    """
    kaynak = provenance_for(
        UCUZ_METRO,
        plateau_reached=np.array([True, True]),
        tau_int=np.array([1.0, 2.0]),
    )
    metin = " ".join(deger for _, deger in kaynak.as_lines())
    for cumle in (
        "J = k_B = 1 BIR BIRIM SECIMIDIR, FIZIK DEGIL",
        "sonlu kafeste keskin bir faz gecisi",
        "3B Ising'in tam cozumu yoktur",
        "iliskisiz bir kafeste tohumlarin ~%38'i SAHTE bir xi donduruyor",
        "log-log dogrusu TAM DOGRU DEGILDIR",
        "ORTALAMA ALAN (beta/nu = 1.0) 272.0 kat marjla dislandi",
        "T_c'nin SI birimlerindeki degeri bir tahmin DEGILDIR",
        "BU MODUL HICBIR MALZEMENIN T_c'sini IDDIA ETMEZ",
    ):
        assert cumle in metin, f"beyan kayip: {cumle!r}"


def test_SONLU_L_BEYANI_SENARYODAN_gelir_SABIT_DEGIL():
    """`sonlu L: bu kosum L = N kullanir` satırı gerçekten `lattice_size`ı
    okumalı.

    Sabit yazılsaydı `lattice_size: 8` ile koşan bir kullanıcı panelde
    "L = 16" yazdığını görürdü — yani panel koşulmayan bir hesabı beyan
    ederdi. Bu test hiç sweep KOŞMAZ: yalnızca beyan bloğunu kurar.
    """
    for L in (6, 8, 32):
        senaryo = dataclasses.replace(UCUZ_METRO, lattice_size=L)
        kaynak = provenance_for(
            senaryo,
            plateau_reached=np.array([True, True]),
            tau_int=np.array([1.0, 2.0]),
        )
        metin = " ".join(deger for _, deger in kaynak.as_lines())
        assert f"sonlu L: bu kosum L = {L} kullanir" in metin
        assert f"L = {L}" in dict(kaynak.as_lines())["Yöntem"]


def test_YAKLASIMLAR_ile_IHMAL_EDILEN_KARISMIYOR():
    """İki liste AYNI panelde ama AYNI SATIRDA değil.

    Tersine çevrilmiş bir atama (`approximations=_SPIN_IHMAL`) bütün
    cümleleri hâlâ ekrana basardı ve "beyan var mı" diye bakan her test
    yeşil kalırdı. Ayıran şey cümlelerin HANGİ satırda olduğudur: bir
    yaklaşım ile bir ihmal aynı şey değildir. Aynı ayrım DAİRESELLİK için
    de kurulur — o cümle `Yaklaşımlar` satırına karışmamalı, çünkü bir
    yaklaşım YAPILAN bir şeydir, dairesellik beyanı ise çıktının NE
    OLMADIĞINI söyler.
    """
    kaynak = provenance_for(
        UCUZ_METRO,
        plateau_reached=np.array([True, True]),
        tau_int=np.array([1.0, 2.0]),
    )
    satirlar = dict(kaynak.as_lines())

    for cumle in fabrika._SPIN_YAKLASIMLAR:
        assert cumle in satirlar["Yaklaşımlar"]
        assert cumle not in satirlar["İhmal edilen"]
    for cumle in fabrika._SPIN_IHMAL:
        assert cumle in satirlar["İhmal edilen"]
        assert cumle not in satirlar["Yaklaşımlar"]
    assert fabrika._SPIN_BEKLENEN_HATA == satirlar["Beklenen hata"]
    assert fabrika._DAIRESELLIK_BEYANI == satirlar["Dairesellik"]
    assert fabrika._DAIRESELLIK_BEYANI not in satirlar["Yaklaşımlar"]
    assert fabrika._DAIRESELLIK_BEYANI not in satirlar["İhmal edilen"]


def test_PANEL_BES_ZORUNLU_BEYANIN_BESINI_de_basiyor():
    """Spec §6'nın beş zorunlu beyanı, HER BİRİ KENDİ SATIRINDA.

    Etiket listesi tam olarak kilitlenir: bir satır kaybolursa ya da
    yeri değişirse test kırılır. `Dairesellik` ile `Örnekleme` ayrı
    satırlardır ve bu bir biçim tercihi değil — `provenance.py`de
    gerekçesi yazılı: dairesellik beyanı `Yaklaşımlar` listesine
    konsaydı modülün varlık sebebi beş maddeden biri olarak akıp
    giderdi, örnekleme tanısı `diagnostics`e konsaydı "✓ … ALT SINIR"
    diye görünürdü.
    """
    kaynak = provenance_for(
        UCUZ_METRO,
        plateau_reached=np.array([True, False]),
        tau_int=np.array([1.0, 2.0]),
    )
    etiketler = [etiket for etiket, _ in kaynak.as_lines()]
    assert etiketler == [
        "Yöntem",
        "Yaklaşımlar",
        "İhmal edilen",
        "Beklenen hata",
        "Dairesellik",
        "Yakınsama",
        "Örnekleme",
    ]
    satirlar = dict(kaynak.as_lines())
    assert satirlar["Dairesellik"].strip()
    assert satirlar["Örnekleme"].strip()


def test_ORNEKLEME_TANISI_BES_SEYI_de_soyluyor():
    """Sweep sayısı · ölçüm sayısı · algoritma · `τ_int` · plato.

    Beşi de spec §6'da ADIYLA istenmiştir; biri düşerse panel eksik
    beyan etmiş olur. Sayılar SENARYODAN ve ÖLÇÜMDEN gelir, sabit
    değildir — bu yüzden alışılmadık değerlerle sınanır.
    """
    senaryo = dataclasses.replace(UCUZ_METRO, n_sweep=777, n_burn=13, seed=4242)
    satir = provenance_for(
        senaryo,
        plateau_reached=np.array([True, True]),
        tau_int=np.array([1.25, 9.75]),
    ).as_lines()
    ornekleme = dict(satir)["Örnekleme"]
    assert "777 olcum" in ornekleme
    assert "+13 burn-in sweep" in ornekleme
    assert "algoritma metropolis" in ornekleme
    assert "tohum 4242" in ornekleme
    assert "1.25 ... 9.75" in ornekleme
    assert "2/2 sicaklikta olustu" in ornekleme


def test_PLATO_YOKSA_PANEL_ALT_SINIR_diyor_VARSA_DEMIYOR():
    """"ALT SINIR" işareti KOŞULLUDUR — iki yön de kilitlenir.

    Hep basan bir mutant ("her hata çubuğu alt sınırdır") ikinci assert'i,
    hiç basmayan bir mutant birinciyi düşürür. Tek yönlü bir test
    ikisinden birini kaçırırdı.

    Ayrıca `converged` bayrağı da buraya bağlıdır ve bu bilinçlidir: bu
    modülde "yakınsadı" demek "SCF durdu" değil, RAPORLANAN HER HATA
    ÇUBUĞU GERÇEKTEN BİR HATA ÇUBUĞU demektir.
    """
    eksik = provenance_for(
        UCUZ_METRO,
        plateau_reached=np.array([True, False]),
        tau_int=np.array([1.0, 40.0]),
    )
    assert ALT_SINIR_ISARETI in dict(eksik.as_lines())["Örnekleme"]
    assert "T = 4.000" in dict(eksik.as_lines())["Örnekleme"]
    assert eksik.converged is False
    assert dict(eksik.as_lines())["Yakınsama"].startswith("✗")

    tam = provenance_for(
        UCUZ_METRO,
        plateau_reached=np.array([True, True]),
        tau_int=np.array([1.0, 2.0]),
    )
    assert ALT_SINIR_ISARETI not in dict(tam.as_lines())["Örnekleme"]
    assert tam.converged is True
    assert dict(tam.as_lines())["Yakınsama"].startswith("✓")


def test_TANI_DIZILERI_SICAKLIK_SAYISIYLA_uyusmak_zorunda():
    """Yanlış uzunlukta bir tanı dizisi SESSİZCE kabul edilemez.

    `zip` kısa olanda durur: üç sıcaklığa iki bayrak verilseydi son
    sıcaklık hiç denetlenmemiş olur ve panel eksik bir hüküm basardı.
    """
    with pytest.raises(ValueError):
        provenance_for(
            UCUZ_METRO,
            plateau_reached=np.array([True]),
            tau_int=np.array([1.0]),
        )


# ---------------------------------------------------------------------------
# FİZİK koşan testler -- ölçülen süreler dosya dokstringindedir, hiçbiri
# 2 s eşiğini geçmez ve bu yüzden HİÇBİRİ `slow` DEĞİLDİR. Tek istisna
# uçtan uca CLI koşumudur ve o da ölçülerek işaretsiz bırakılmıştır.
# ---------------------------------------------------------------------------


def test_FABRIKA_IKINCI_BIR_FIZIK_YOLU_ACMIYOR():
    """`build_spin`, `run_metropolis`in ürettiğinden BAŞKA bir sayı üretmez.

    Fabrikanın işi model kurmak, tohum akışlarını dağıtmak ve beyanı
    toplamaktır; bir gözlemlenebilirin kendisini hesaplasaydı iki yol
    ayrışabilirdi ve panel motorun söylemediğini yazardı.

    Karşılaştırma BİREBİR (`==`) yapılır, tolerans YOKTUR: aynı tohum
    akışı aynı diziyi verir.
    """
    from qlab.engine.spin.lattice import SpinState
    from qlab.engine.spin.metropolis import run_metropolis
    from qlab.engine.spin.models import IsingModel

    sonuc = build_spin(UCUZ_METRO)
    akislar = np.random.SeedSequence(UCUZ_METRO.seed).spawn(
        len(UCUZ_METRO.temperatures)
    )
    for T, akis, olculen in zip(UCUZ_METRO.temperatures, akislar, sonuc.samples):
        beklenen = run_metropolis(
            SpinState.uniform(L=UCUZ_METRO.lattice_size, n_component=1),
            IsingModel(),
            temperature=T,
            coupling=UCUZ_METRO.coupling,
            field=UCUZ_METRO.field,
            rng=np.random.default_rng(akis),
            n_sweep=UCUZ_METRO.n_sweep,
            n_burn=UCUZ_METRO.n_burn,
        )
        assert np.array_equal(
            olculen.magnetisation_series, beklenen.magnetisation_series
        )


def test_AYNI_TOHUM_AYNI_SONUC_FARKLI_TOHUM_FARKLI():
    """Tekrarlanabilirlik, ve tohumun GERÇEKTEN okunduğu.

    İkinci yarı olmadan `seed`i yok sayan bir mutant (sabit tohum, ya da
    `default_rng()` ile tohumsuz) birinci yarıyı GEÇEBİLİRDİ — sabit
    tohum da tekrarlanabilirdir. Ayrışma iddiası o deliği kapatır.
    """
    bir = build_spin_payload(build_spin(UCUZ_METRO))
    iki = build_spin_payload(build_spin(UCUZ_METRO))
    assert np.array_equal(bir.magnetisation, iki.magnetisation)
    assert np.array_equal(bir.magnetisation_error, iki.magnetisation_error)
    assert np.array_equal(bir.binder, iki.binder)

    baska = build_spin_payload(
        build_spin(dataclasses.replace(UCUZ_METRO, seed=UCUZ_METRO.seed + 1))
    )
    assert not np.array_equal(bir.magnetisation, baska.magnetisation)


def test_HER_SICAKLIK_AYRI_AKIS_aliyor():
    """Tarama noktaları AYNI rastgele diziyi paylaşmaz.

    Tek bir `default_rng(seed)` bütün sıcaklıklarda yeniden kullanılsaydı
    komşu noktaların gürültüsü ilişkili olurdu ve eğri sahte bir düzgünlük
    kazanırdı. Ölçüt: AYNI sıcaklığı iki kez içeren bir taramada iki
    satırın serileri FARKLI olmalı (aynı akış olsaydı birebir aynı
    olurlardı).
    """
    senaryo = dataclasses.replace(UCUZ_METRO, temperatures=(3.0, 3.0000001, 3.0000002))
    sonuc = build_spin(senaryo)
    a, b, c = (s.magnetisation_series for s in sonuc.samples)
    assert not np.array_equal(a, b)
    assert not np.array_equal(b, c)


def test_ISI_SIGASI_WOLFF_TA_OLCULMEDI_SIFIR_DEGIL():
    """`energy_series is None` → ısı sığası `None` KALIR, 0.0 OLMAZ.

    Wolff küme çevirdiği için enerjiyi adım başına ucuza güncelleyemez.
    `None`u sıfır sanan bir tüketici "C = 0 çıktı" diye okunurdu ve hiçbir
    uyarı vermezdi — Modül 1B'nin kararlılık alanı dersinin aynısı.
    Metropolis tarafı aynı testte ölçülür, yoksa "hep None döndür" diyen
    bir mutant hayatta kalırdı.
    """
    wolff = build_spin_payload(build_spin(UCUZ_WOLFF))
    assert wolff.heat_capacity is None
    assert all(s.energy_series is None for s in build_spin(UCUZ_WOLFF).samples)

    metro = build_spin_payload(build_spin(UCUZ_METRO))
    assert metro.heat_capacity is not None
    assert np.all(metro.heat_capacity > 0.0)


def test_YUK_SICAKLIK_SIRASINI_KORUYOR_ve_TUM_DIZILER_AYNI_UZUNLUKTA():
    """Sıra bozan bir mutant, sıcaklıklar birbirine yakınsa fark edilmezdi.

    Bu yüzden hem sıra hem de fiziksel yön birlikte kilitlenir: `⟨|m|⟩`
    sıcaklıkla DÜŞER (T = 3.0 → 4.0). Bu bir Ising iddiası değil, dizilerin
    doğru satırla eşleştiğinin kontrolüdür.
    """
    yuk = build_spin_payload(build_spin(UCUZ_METRO))
    assert np.array_equal(yuk.temperatures, np.array(UCUZ_METRO.temperatures))
    n = len(UCUZ_METRO.temperatures)
    for dizi in (
        yuk.magnetisation,
        yuk.magnetisation_error,
        yuk.plateau_reached,
        yuk.tau_int,
        yuk.binder,
        yuk.susceptibility,
        yuk.heat_capacity,
    ):
        assert dizi.shape == (n,)
    assert yuk.magnetisation[0] > yuk.magnetisation[1]


def test_KARISIK_ENERJI_DURUMU_REDDEDILIYOR():
    """Yarısı ölçülmüş bir ısı sığası dizisi olamaz.

    `energy_series` bir KOŞUM özelliğidir (Wolff ölçmez, Metropolis ölçer).
    Karışık bir sonuç oluşursa tek bir dizi "bu satır ölçüldü, bu satır
    ölçülmedi"yi anlatamaz ve sessizce yarısı doğru bir dizi dönerdi.
    """
    metro = build_spin(UCUZ_METRO)
    wolff = build_spin(UCUZ_WOLFF)
    karisik = dataclasses.replace(
        metro, samples=(metro.samples[0], wolff.samples[0])
    )
    with pytest.raises(InvalidInputError, match="isi sigasi"):
        build_spin_payload(karisik)


def test_SEVK_EDILEN_KOSUMDA_ALT_SINIR_IKI_YONDE_de_dogru():
    """Sevk edilen senaryo işaretin İKİ YÖNÜNÜ de kendisi gösteriyor.

    ÖLÇÜLDÜ (24 tohum, dosya dokstringindeki tablo): T = 2.2 ve T = 2.4'te
    plato **0/24**, T = 3.4'te **24/24**. Yani bu iki assert tohum
    şansına değil, kritik yavaşlamanın kendisine dayanıyor.

    T = 1.6 (23/24) ile geçiş bölgesindeki T = 2.0 (16/24) ve T = 2.8
    (13/24) BİLİNÇLİ OLARAK assert EDİLMEZ: sevk edilen tohumda değerleri
    sabittir ama hüküm tohum gürültüsüne açıktır ve bu depoda "tek akıştan
    okunan bir oranı fiziğe yazmak" üç kez düzeltilmiş bir hatadır.
    """
    senaryo = load_scenario(SEVK_EDILEN)
    yuk = build_spin_payload(build_spin(senaryo))
    indis = {round(T, 4): i for i, T in enumerate(senaryo.temperatures)}
    assert yuk.plateau_reached[indis[2.2]] is np.False_
    assert yuk.plateau_reached[indis[2.4]] is np.False_
    assert yuk.plateau_reached[indis[3.4]] is np.True_
    # Ve bu, panelde de goruluyor -- iki yon birden.
    kaynak = provenance_for(
        senaryo, plateau_reached=yuk.plateau_reached, tau_int=yuk.tau_int
    )
    ornekleme = dict(kaynak.as_lines())["Örnekleme"]
    assert ALT_SINIR_ISARETI in ornekleme
    assert "2.200" in ornekleme and "2.400" in ornekleme
    assert "3.400" not in ornekleme


def test_CLI_UCTAN_UCA(tmp_path):
    """`python -m qlab.run scenarios/spin_tarama.yaml -o spin.npz`.

    ÖLÇÜLDÜ: taramanın kendisi 0.56 s, yorumlayıcı başlangıcıyla birlikte
    alt süreç ~2 s civarı. Deponun `slow` eşiği "kendi `call` süresi 2 s ve
    üstünde" der; bu test ölçülerek işaretsiz bırakıldı ve ölçülen süre
    rapor bölümündedir.

    YEDİ panel satırının YEDİSİ de aranır. Metinler bu dosyada
    TEKRARLANMAZ; `factory` modülünden okunur, yani beyan düzeltilirse test
    onu KENDİLİĞİNDEN takip eder ve ikinci bir kopya doğmaz.
    """
    cikti = tmp_path / "spin.npz"
    sonuc = subprocess.run(
        [sys.executable, "-m", "qlab.run", str(SEVK_EDILEN), "-o", str(cikti)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(PROJE_KOKU),
    )
    assert sonuc.returncode == 0, sonuc.stderr
    assert cikti.exists()

    for etiket in (
        "Yöntem",
        "Yaklaşımlar",
        "İhmal edilen",
        "Beklenen hata",
        "Dairesellik",
        "Yakınsama",
        "Örnekleme",
    ):
        assert etiket in sonuc.stdout, f"panel satiri kayip: {etiket}"
    for cumle in _tum_beyanlar():
        assert cumle in sonuc.stdout, f"beyan EKRANA basilmadi: {cumle[:50]!r}"
    assert "sonlu L: bu kosum L = 16 kullanir" in sonuc.stdout
    assert ALT_SINIR_ISARETI in sonuc.stdout

    with np.load(cikti, allow_pickle=False) as veri:
        assert np.allclose(veri["temperatures"], [1.6, 2.0, 2.2, 2.4, 2.8, 3.4])
        assert veri["magnetisation"].shape == (6,)
        assert np.all(veri["magnetisation_error"] > 0.0)
        assert veri["plateau_reached"].dtype == bool
        assert not veri["plateau_reached"].all()
        # Metropolis enerji OLCER -> anahtar VAR.
        assert "heat_capacity" in veri
        kaynak = json.loads(str(veri["provenance_json"]))
    assert kaynak["converged"] is False
    assert kaynak["circularity"] == fabrika._DAIRESELLIK_BEYANI
    assert ALT_SINIR_ISARETI in kaynak["sampling"]
    assert kaynak["diagnostics"]["plateau_count"] == "4/6"


def test_WOLFF_KOSUMUNDA_ISI_SIGASI_ANAHTARI_NPZ_DE_HIC_YOK(tmp_path):
    """Ölçülmemiş bir büyüklük diske SIFIR olarak yazılmaz — HİÇ yazılmaz.

    Sıfır dizisi yazılsaydı `.npz`yi okuyan biri "C = 0 çıktı" diye
    okurdu ve dosyada hiçbir uyarı olmazdı. Aynı gerekçe
    `observable/spin_obs.py::_seri`de `None`un hata fırlatmasının da
    sebebidir.
    """
    yaml_yolu = _yaz(
        tmp_path,
        "kind: spin\nname: wolff denetimi\nmodel: ising\nalgorithm: wolff\n"
        "lattice_size: 8\ntemperatures: [1.8, 3.0]\nn_sweep: 800\n"
        "n_burn: 200\nseed: 20260823\n",
    )
    cikti = tmp_path / "wolff.npz"
    sonuc = subprocess.run(
        [sys.executable, "-m", "qlab.run", str(yaml_yolu), "-o", str(cikti)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(PROJE_KOKU),
    )
    assert sonuc.returncode == 0, sonuc.stderr
    assert "C=ÖLÇÜLMEDİ" in sonuc.stdout
    with np.load(cikti, allow_pickle=False) as veri:
        assert "heat_capacity" not in veri
        assert "magnetisation" in veri


def test_BOZUK_SENARYO_CLI_DA_TEMIZ_HATA_veriyor(tmp_path):
    """Yakalanmayan bir `AttributeError` değil, `rc = 1` ve stderr'de mesaj.

    `n_sweep` 32'nin altına düştüğünde `binning_error` bir `InvalidInputError`
    fırlatır (plato ölçütü en az `min_blocks` örnek ister). Bu hata
    `viz` katmanının derinindedir ve `main`in `QLabError` dalına düşmesi
    gerekir; düşmezse kullanıcı bir yığın izi görürdü.
    """
    yaml_yolu = _yaz(
        tmp_path,
        "kind: spin\nname: cok kisa\nmodel: ising\nalgorithm: wolff\n"
        "lattice_size: 6\ntemperatures: [3.0]\nn_sweep: 8\n"
        "n_burn: 2\nseed: 1\n",
    )
    sonuc = subprocess.run(
        [sys.executable, "-m", "qlab.run", str(yaml_yolu), "-o", str(tmp_path / "x.npz")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(PROJE_KOKU),
    )
    assert sonuc.returncode == 1
    assert "min_blocks" in sonuc.stderr
    assert "Traceback" not in sonuc.stderr


# ---------------------------------------------------------------------------
# `slow` İŞARETİNİN GEREKÇESİ -- ÖLÇÜM.
#
# Deponun eşiği: kendi `call` süresi 2 s ve ÜSTÜNDE olan test işaretlenir
# (devralınabilir bir eşik YOK; gerekçesi `tests/spin/test_metropolis.py` ve
# `tests/tb/test_downfold_senaryo.py`de yazılı). Bu dosyada işaretli test
# YOKTUR ve gerekçesi ölçüldü: en pahalı iki test alt süreç başlatan CLI
# koşumlarıdır; taramanın kendisi 0.56 s, ucuz senaryolar 0.13 s ve 0.05 s.
# Ölçülen dosya süresi rapor bölümündedir.
# ---------------------------------------------------------------------------
