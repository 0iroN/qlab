"""`kind: spin` senaryosu, panel beyanı ve uçtan uca koşum.

SINIF: TRANSKRİPSİYON DENETİMİ. Bu dosyanın hiçbir testi bir fizik iddiası
taşımaz; senaryo sınırının (YAML → dataclass → motor → gözlemlenebilir →
panel) doğru bağlandığını ve panelin **beş zorunlu beyanı** bastığını ölçer.
Modülün fiziksel çıpaları başka yerdedir: Onsager ve kritik üs
`test_cipalar.py`de, Mermin-Wagner `test_mermin_wagner.py`de, çapraz kontrol
`test_wolff.py`de.

BİRİNCİ İSTİSNA — VE O DA FİZİK DEĞİL, KOŞUM ÖZELLİĞİ: plato bayrağının sevk
edilen senaryoda **hangi sıcaklıklarda** düştüğü. O bir Ising iddiası değil,
"bu koşum kendi otokorelasyonunu çözecek kadar uzun mu" sorusunun cevabıdır
ve aşağıda 24 tohumla ölçülmüştür.

İKİNCİ İSTİSNA (09.10.2026, F7 kapanışı): `test_ALAN_MIKNATISLANMAYI_AZALTAMAZ`
bir fizik iddiası taşır (alan ⟨|m|⟩'yi `h = 0`'ın altına indiremez; eşik 11
tohumla ölçülmüş bandın içinde) ve `test_BINNING_PLATOSU_METASTABILITEYI_DISLAMAZ`
bir tanının SINIRINI, yani karşı örneğini kilitler. İkisi de kamu yolundan
koşar; ölçümler kendi dokstringlerindedir.

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
import inspect
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
    HİÇBİR ŞEY çevirmez çünkü `k_B = 1` bir birim seçimidir (sıcaklık enerji
    birimindedir, coupling bir girdidir). YAML'daki 2.2 dataclass'ta da 2.2'dir.
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
        ("coupling: 1.0", "coupling: .inf"),
        ("field: 0.0", "field: .nan"),
        ("field: 0.0", "field: .inf"),
        ("model: ising", "model: heisenberg\nheisenberg_step: .nan"),
    ],
)
def test_BOZUK_senaryolar_REDDEDILIYOR(tmp_path, bozuk, desen):
    """Sessizce varsayılana düşmek YASAK: her biri ScenarioError vermeli.

    `lattice_size: 15` özellikle önemli — checkerboard Metropolis TEK L'de
    ayrıntılı dengeyi bozar ve bozulma SESSİZDİR (sonuç makul görünür).
    `coupling: -1.0` de öyle: Wolff'ta `p = 1 - exp(-2J/T)` negatife düşer,
    hiçbir bağ kurulmaz ve algoritma sonsuz sıcaklığı örnekler — ölçüldü,
    20 kat hata ve sıfır istisna (`engine/spin/wolff.py`). `coupling: .inf`
    27.09.2026'ya kadar GEÇİYORDU (`not coupling > 0`) ve hata geç geliyordu;
    `field: .nan/.inf` de (adsız geç hata) ve `heisenberg_step: .nan` (SESSİZ donma).
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
    olarak geçiyor). Buradaki on iki girdinin on ikisi de tam cümledir ve
    muafiyetsiz sayım 1/0 tutuyor — yani muafiyet listesi eklemek denetimi
    gereksiz yere gevşetmek olurdu.

    BU DOSYANIN MUAF OLMASI BİR SAPMA DEĞİL, KİLİDİN KENDİSİDİR: tam cümle
    aramak için cümleyi bir kez yazmak gerekir ve o tek kopya
    `test_MODULUN_ETIK_CEKIRDEGI_metinde_YAZILI`dadır. Kaynak metni
    değişirse İKİSİ BİRDEN kırmızı olur.
    """
    beyanlar = _tum_beyanlar()
    assert len(beyanlar) == 12, "beyan sayisi degisti; testi bilerek guncelleyin"
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
    kaçırıyor. Aşağıdaki on bir parça bu modülün etik çekirdeğidir ve
    sessizce yeniden yazılmamalıdır: ikisi birim seçimini ve sonlu kafesi
    (yani "burada keskin bir geçiş yok"u), ikisi 3B ile ξ'nin neden
    kapsam dışı olduğunu, ikisi kritik üsün sonlu-boyut kalıntısını, ikisi
    de daireselliğin kendisini yazıyor. 27.09.2026'da iki parça eklendi:
    kT/J = T/coupling (J bir girdidir) ve Ising'in T/J -> sonsuz donması.
    09.10.2026'da (F7) bir parça daha: başlangıç alanla hizalı tekbiçimdir —
    F7'nin "pencere uyarısı YOK" hükmü bu öncüle dayanır
    (`test_KAMU_BASLANGICI_ALANLA_HIZALI_ve_TEK`). Cümle sayısı DEĞİŞMEDİ
    (12): parça mevcut "denge ornekleme" cümlesine eklendi.

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
        "k_B = 1 BIR BIRIM SECIMIDIR, FIZIK DEGIL",
        "T degil kT/J = T/coupling'dir ve Yontem satirinda basilir",
        "Ising checkerboard Metropolis T/J -> sonsuzda DONAR",
        "sonlu kafeste keskin bir faz gecisi",
        "3B Ising'in tam cozumu yoktur",
        "iliskisiz bir kafeste tohumlarin ~%38'i SAHTE bir xi donduruyor",
        "log-log dogrusu TAM DOGRU DEGILDIR",
        "ORTALAMA ALAN (beta/nu = 1.0) 272.0 kat marjla dislandi",
        "T_c'nin SI birimlerindeki degeri bir tahmin DEGILDIR",
        "BU MODUL HICBIR MALZEMENIN T_c'sini IDDIA ETMEZ",
        "baslangic alanla hizali tekbicimdir",
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
# BİRİM, KAPSAM KAPISI VE DONMA (27.09.2026). Ölçüm kayıtları depo dışında (gitignore):
# `.superpowers/sdd/hf-spin-uygulama-betikleri-2026-09-27/` (p2_olc_*.py + .out). Hepsi PUBLIC panel yolundan.
# ---------------------------------------------------------------------------


def test_YONTEM_SATIRI_coupling_i_GIRDI_ve_GERCEK_kT_J_yi_basiyor():
    """Panel `coupling`i ve gerçek `kT/J = T/coupling`i basar. Sweep koşmaz.

    27.09.2026'ya kadar beyan "sıcaklık kT/J cinsindendir" diyordu; coupling
    ≠ 1'de YANLIŞTI. ÖLÇÜLDÜ (L = 16, tohum 5, 500 burn + 4000, T = 2.269185):
    coupling 0.5 / 1 / 2 → ⟨|m|⟩ 0.089652 / 0.734195 / 0.998039 — aynı "T",
    üç ayrı faz. `T` basan (bölmeyen) ya da `T·J` basan mutant düşer; basamak
    kaybı mutantı (.6g → .2g / .3g / .5g) üçüncü durumda düşer.
    """
    for J, T, beklenen in ((2.0, (3.0, 4.0), "kT/J = T/J: 1.5, 2"), (0.5, (3.0, 4.0), "kT/J = T/J: 6, 8"),
                           (2.0, (4.53837, 6.0), "kT/J = T/J: 2.26918, 3")):
        yontem = dict(
            provenance_for(
                dataclasses.replace(UCUZ_METRO, coupling=J, temperatures=T),
                plateau_reached=np.array([True, True]),
                tau_int=np.array([1.0, 2.0]),
            ).as_lines()
        )["Yöntem"]
        assert f"J = {J:g} (girdi)" in yontem
        assert yontem.endswith(beklenen)


@pytest.mark.parametrize("taban", [UCUZ_METRO, UCUZ_WOLFF], ids=["metro", "wolff"])
def test_SICAKLIK_ENERJI_BIRIMINDE_yalniz_T_bolu_coupling_sayiyor(taban):
    """`(coupling, T) = (2, 2T')` koşumu `(1, T')` koşumuyla BİT BİT aynı.

    2'nin kuvvetiyle ölçekleme IEEE'de kesindir: `dE/T` aynı sayıdır, kabul
    kararları da. Beyanın ("kT/J = T/coupling") MOTORA bağlandığı yer burası:
    fabrika `T`yi kT/J sanıp motora `T·coupling` geçirseydi seriler ayrışırdı
    (iki dalda da sınandı: metropolis A3, wolff W1). Enerji serisi tam 2
    katıdır; Wolff enerji TUTMAZ, orada yalnız `m` karşılaştırılır ve o da
    kesindir (`wolff.py::run_wolff` `p_bond = 1 - exp(-2J/T)`).
    """
    bir = build_spin(taban)
    iki = build_spin(dataclasses.replace(taban, coupling=2.0, temperatures=tuple(2.0 * t for t in taban.temperatures)))
    for a, b in zip(bir.samples, iki.samples, strict=True):
        assert np.array_equal(a.magnetisation_series, b.magnetisation_series)
        if a.energy_series is None:
            assert b.energy_series is None and taban is UCUZ_WOLFF
        else:
            assert np.array_equal(2.0 * a.energy_series, b.energy_series)


@pytest.mark.parametrize("J", [-1.0, 0.0, float("inf"), float("-inf"), float("nan")])
def test_PYTHON_YOLU_da_coupling_KAPSAM_KAPISINDAN_geciyor(J):
    """Yükleyicinin `J <= 0` kapsam kararı + sonluluk Python yolunda da geçerli.

    ÖLÇÜLDÜ (kapı YOKKEN): `SpinScenario(coupling=-1)` panele kadar koşuyordu
    — T = 3 / 4'te ⟨|m|⟩ 0.055 / 0.064, Binder −0.046 / −0.040,
    `converged=True`, AFM için anlamsız ve uyarısız. `coupling=inf`/`nan`
    hatayı ancak `build_spin_payload`da, enerji serisi mesajıyla veriyordu.
    Motorun kendisi J < 0'ı koşmaya DEVAM EDER; onu hf-spin ayar testi
    (`tests/bridge/test_dusen_adaylar.py`) kullanır ve kilitler. Public
    `provenance_for` da aynı kapıdan geçer: kapı yokken J = 0'da ham
    ZeroDivisionError, J < 0 / inf / nan'da hatasız anlamsız kT/J basıyordu.
    """
    with pytest.raises(InvalidInputError, match="coupling"):
        build_spin(dataclasses.replace(UCUZ_METRO, coupling=J))
    with pytest.raises(InvalidInputError, match="coupling"):
        provenance_for(dataclasses.replace(UCUZ_METRO, coupling=J), plateau_reached=np.ones(2, bool), tau_int=np.ones(2))


def test_DONMA_YAML_dan_ULASILIR_ve_PANEL_YAKINSADI_DEMIYOR(tmp_path):
    """Ising checkerboard T/J → ∞'da DONAR; `converged` bunu yakalıyor.

    `coupling: 1.0e-6` kapıdan geçer (pozitif, sonlu: kapsam ihlali değil,
    pencere sorunu). T/J = 1e6'da kabul ~1, her yarı-adım bir alt-örgüyü
    bütünüyle çevirir: ⟨|m|⟩ ilişkisiz taban 0.099347 (L = 8, tam binom)
    yerine ~1. ÖLÇÜLDÜ: bu pencerede 12/12 tohumda ⟨|m|⟩ ≥ 0.968 ve 12/12
    `converged=False`; geniş taramada tabanın 1.5 katını aşan 128/128 satırda
    plato yok (L = 8/16/32). Bayrak zaten düştüğü için KAPI DEĞİL BEYAN
    seçildi. Kabulü yarıya indiren mutant donmayı çözer (ilk assert),
    `converged`i sabitleyen mutant ikinciyi düşürür; aynı pencerede J = 0.2
    (T/J = 5, 12/12 tohumda plato) `True` — dedektör gerçek bir koşumu ayırıyor.
    """
    senaryo = load_scenario(
        _yaz(
            tmp_path,
            "kind: spin\nname: donma\nmodel: ising\nalgorithm: metropolis\n"
            "lattice_size: 8\ntemperatures: [1.0]\nn_sweep: 800\nn_burn: 200\n"
            "seed: 20260823\ncoupling: 1.0e-6\n",
        )
    )
    yuk = build_spin_payload(build_spin(senaryo))
    kaynak = provenance_for(
        senaryo, plateau_reached=yuk.plateau_reached, tau_int=yuk.tau_int
    )
    assert yuk.magnetisation[0] > 0.9
    assert kaynak.converged is False
    assert dict(kaynak.as_lines())["Yakınsama"].startswith("✗")
    ayni = dataclasses.replace(senaryo, coupling=0.2)
    yuk = build_spin_payload(build_spin(ayni))
    assert provenance_for(ayni, plateau_reached=yuk.plateau_reached, tau_int=yuk.tau_int).converged is True


def test_KHI_ENERJI_EKSI_BIR_BIRIMINDE_ve_NPZ_coupling_TASIYOR(tmp_path, capsys):
    """`(J, T) → (J/2, T/2)`: ⟨|m|⟩, Binder, C BİT BİT aynı, χ tam 2 KATI (enerji⁻¹; karşılaştırılacak
    sayı J·χ). Beyan panelde; `coupling` `.npz`de ayrı anahtar (kT/J metin ayrıştırmadan kurulur)."""
    from qlab import run
    bir = build_spin_payload(build_spin(UCUZ_METRO))
    yol = _yaz(tmp_path, "kind: spin\nname: khi\nmodel: ising\nalgorithm: metropolis\nlattice_size: 8\n"
               "temperatures: [1.5, 2.0]\nn_sweep: 800\nn_burn: 200\nseed: 20260823\ncoupling: 0.5\n")
    assert run.main([str(yol), "-o", str(tmp_path / "khi.npz")]) == 0
    with np.load(tmp_path / "khi.npz", allow_pickle=False) as v:
        assert float(v["coupling"]) == 0.5
        for ad in ("magnetisation", "binder", "heat_capacity"):
            assert np.array_equal(v[ad], getattr(bir, ad)), ad
        assert np.array_equal(0.5 * v["susceptibility"], bir.susceptibility)
        yaklasim = json.loads(str(v["provenance_json"]))["approximations"]
    assert any("chi 1/l ile olceklenir, coupling'ler arasi karsilastirilacak sayi J*chi'dir" in a for a in yaklasim)


# ---------------------------------------------------------------------------
# F7 KAPANIŞI (09.10.2026): BAŞLANGIÇ ALANLA HİZALI. Ölçüm kayıtları depo dışında (gitignore):
# `.superpowers/sdd/f7-olcum/` (olcum/ ön ölçüm, elestirmen/ e1-e6; uygulama/ u0 düzeltme öncesi
# kırmızı, u1-u2 e-sayılarının kamu yolundan yeniden üretimi, u3 mutantlar).
# HEPSİ KAMU YOLUNDAN: YAML -> `qlab.run.main` -> `.npz`.
# Ortak pencere: Ising · Metropolis · L = 16 · T = (1.0, 1.5) · 200 + 1000 sweep · tohum 20260823.
# ---------------------------------------------------------------------------

F7_ANAHTARLAR = ("magnetisation", "magnetisation_error", "tau_int", "binder", "susceptibility", "heat_capacity")

F7_MESAJ = (
    "F7 (pencere uyarisi YOK) bu onculle duruyor; baslangic secenegi eklenirse F7'yi yeniden ac: "
    "capraz serit ✓ veriyor (.superpowers/sdd/f7-olcum/elestirmen/e2, e6)"
)


def _f7_kos(tmp_path, field, *, model="ising", algoritma="metropolis", L=16, sicakliklar=(1.0, 1.5),
            n_sweep=1000, n_burn=200, tohum=20260823) -> dict:
    """Kamu yolu: YAML yaz, `qlab.run.main([yaml, '-o', npz]) == 0`, `.npz`yi sözlük olarak döndür.

    Sözlükte `.npz` anahtarlarının kopyası, `kaynak` (çözülmüş `provenance_json`) ve `yaml` (yol) var.
    """
    from qlab import run

    ad = f"f7_{model}_{algoritma}_L{L}_h{float(field):+.3f}_t{tohum}"
    yaml_yolu = tmp_path / f"{ad}.yaml"
    yaml_yolu.write_text(
        f"kind: spin\nname: f7 kapanis\nmodel: {model}\nalgorithm: {algoritma}\n"
        f"lattice_size: {L}\ntemperatures: [{', '.join(repr(float(t)) for t in sicakliklar)}]\n"
        f"n_sweep: {n_sweep}\nn_burn: {n_burn}\nseed: {tohum}\nfield: {float(field)!r}\n",
        encoding="utf-8",
    )
    npz = tmp_path / f"{ad}.npz"
    assert run.main([str(yaml_yolu), "-o", str(npz)]) == 0, f"kamu yolu cikis != 0: {ad}"
    with np.load(npz, allow_pickle=False) as v:
        sonuc = {k: v[k].copy() for k in v.files}
    sonuc["kaynak"] = json.loads(str(sonuc["provenance_json"]))
    sonuc["yaml"] = yaml_yolu
    return sonuc


def test_NEGATIF_ALAN_kamu_yolunda_AYNA_bit_bit(tmp_path):
    """`h → −h` altında kamu çıktısı BİT BİT aynı olmalı (Z2 aynası).

    Ising'de `(s, h) → (−s, −h)` enerjiyi ve her Metropolis kabul kararını
    değiştirmez; aynı tohum akışıyla, BAŞLANGIÇ DA AYNALANIRSA, `h < 0` koşumu
    `h > 0` koşumunun bit bit aynasıdır (eleştirmen e1 B: `|m|`, E ve son durum
    3/3 tohumda ayna). 09.10.2026'ya kadar başlangıç HEP +1'di: `h < 0` alana
    TERS başlıyor, Ising metastabil kalıyor ve panel ✓ basıyordu. ÖLÇÜLDÜ (e5,
    bu pencere, tohum 20260823 + 1..10): düzeltmesiz 0/11, düzeltmeyle 11/11.

    Yöntem satırı GİRDİYİ basmalı: `|h|`yi motora geçirip aynalığı "kazanan"
    bir mutantın senaryosu da `h = 0.2` derdi.

    TEK BAŞINA YETMEZ: işareti TERS çeviren mutant (`h < 0 → +1`, `h ≥ 0 → −1`)
    aynayı da korur ve bu testi 11/11 GEÇER (e5 "ters"); onu
    `test_ALAN_MIKNATISLANMAYI_AZALTAMAZ` ile `test_POZITIF_VE_SIFIR_ALAN_DEGISMEDI` öldürür.
    """
    eksi = _f7_kos(tmp_path, -0.2)
    arti = _f7_kos(tmp_path, +0.2)
    for ad in F7_ANAHTARLAR:
        assert np.array_equal(eksi[ad], arti[ad]), f"{ad}: h = -0.2 ile +0.2 bit bit ayna DEGIL"
    assert "h = -0.2;" in eksi["kaynak"]["method"]
    assert "h = 0.2;" in arti["kaynak"]["method"]


def test_ALAN_MIKNATISLANMAYI_AZALTAMAZ(tmp_path):
    """`|h| = 0.2` her sıcaklıkta ve HER İŞARETTE `⟨|m|⟩`yi `h = 0`ın ÜSTÜNE çıkarır.

    Eşik 0'dır ve ÖLÇÜLMÜŞ bandın içindedir (eleştirmen e5b, bu pencere, 11
    tohum 20260823 + 1..10; z = (⟨|m|⟩(h) − ⟨|m|⟩(0)) / hypot(hata çubukları)):

        h = −0.2     düzeltmesiz (+1 başlangıç)    düzeltmeyle (−1 başlangıç)
        T = 1.0      z ∈ [−3.37, −2.54]            z ∈ [+1.80, +2.97]
        T = 1.5      z ∈ [−8.81, −1.39]            z ∈ [+6.17, +7.93]

    `h = +0.2` iki kuralda da düzeltilmiş sütunla aynıdır. Yani 0 eşiği iki
    bandın ARASINDA kalır: düzeltmesiz kod 11/11 tohumda kırmızı, düzeltme
    11/11'de yeşil. 3σ eşiği SEÇİLMEDİ: T = 1.0'da düzeltilmiş bant +1.80'de
    başlıyor ve 3σ doğru kodu 11'in 11'inde kırmızı yapardı (e5 K2 0/11).
    """
    sifir = _f7_kos(tmp_path, 0.0)["magnetisation"]
    for h in (-0.2, +0.2):
        m = _f7_kos(tmp_path, h)["magnetisation"]
        for i, T in enumerate((1.0, 1.5)):
            assert m[i] > sifir[i], f"T = {T}, h = {h}: <|m|> {m[i]!r} <= h = 0'daki {sifir[i]!r}"


@pytest.mark.parametrize("h", [0.2, 0.0], ids=["h+0.2", "h0"])
def test_POZITIF_VE_SIFIR_ALAN_DEGISMEDI(tmp_path, h):
    """`h ≥ 0`'da düzeltme kamu çıktısını BİT BİT değiştirmez.

    `test_FABRIKA_IKINCI_BIR_FIZIK_YOLU_ACMIYOR`in `h = +0.2` ve `0`a
    genişletilmiş hâli, kamu yolundan: beklenen değer, motoru `+1` tekbiçim
    başlangıçla, AYNI `SeedSequence.spawn` akışlarıyla doğrudan koşup aynı
    `build_spin_payload`dan geçirmektir. Literal YOK; kıyas birebir `==`.
    ÖLÇÜLDÜ (e5 K3): düzeltme `h ≥ 0` çıktılarını 11/11 tohumda bit bit korur;
    işareti TERS çeviren mutant `h = +0.2`'de 0/11 (alana ters başlar).
    """
    from qlab.engine.spin.lattice import SpinState
    from qlab.engine.spin.metropolis import run_metropolis
    from qlab.engine.spin.models import IsingModel

    kamu = _f7_kos(tmp_path, h)
    senaryo = load_scenario(kamu["yaml"])
    assert senaryo.field == h
    akislar = np.random.SeedSequence(senaryo.seed).spawn(len(senaryo.temperatures))
    dogrudan = tuple(
        run_metropolis(
            SpinState.uniform(L=senaryo.lattice_size, n_component=1),
            IsingModel(),
            temperature=T,
            coupling=senaryo.coupling,
            field=senaryo.field,
            rng=np.random.default_rng(akis),
            n_sweep=senaryo.n_sweep,
            n_burn=senaryo.n_burn,
        )
        for T, akis in zip(senaryo.temperatures, akislar)
    )
    beklenen = build_spin_payload(fabrika.SpinScanResult(scenario=senaryo, samples=dogrudan))
    for ad in F7_ANAHTARLAR:
        assert np.array_equal(kamu[ad], getattr(beklenen, ad)), f"{ad}: h = {h} ciktisi degisti"
    for olculen, ref in zip(build_spin(senaryo).samples, dogrudan, strict=True):
        assert np.array_equal(olculen.magnetisation_series, ref.magnetisation_series)
        assert np.array_equal(olculen.energy_series, ref.energy_series)


def test_KAMU_BASLANGICI_ALANLA_HIZALI_ve_TEK(tmp_path, monkeypatch):
    """F7 KAPSAM KİLİDİ: kamu yolunun TEK başlangıcı alanla hizalı tekbiçimdir.

    F7'nin hükmü ("düşük T burn-in tohuma bağlı → panele pencere uyarısı?"
    sorusu UYARISIZ kapanır) bir tanıya değil bu öncüle dayanır: binning
    platosu metastabiliteyi DIŞLAMAZ (karşı örnek
    `test_BINNING_PLATOSU_METASTABILITEYI_DISLAMAZ`), ama kamu yolunda metastabil
    bir başlangıç YOKTUR — soğuk kamu yolunda takılma ÖLÇÜLDÜ SIFIR (eleştirmen
    e3 C 0/100, ön ölçüm 0/480). Öncül üç parçadır ve üçü de burada kilitli:

      * motora giden HER durum tekbiçimdir: son bileşen `h < 0`'da tümü −1,
        aksi hâlde +1, diğer bileşenler 0 — Ising/Heisenberg × Metropolis,
        Ising × Wolff; motora giden alan GİRDİNİN kendisidir;
      * başlangıç SEÇENEĞİ yoktur: `SpinScenario` alan kümesi ve `build_spin`
        imzası sabit (bir seçenek eklemek ikisinden birini değiştirir);
      * öncül yazılı: `build_spin` ve `SpinState.uniform` dokstringleri.

    `h = 0`'da +1 bir KONVANSİYONDUR: `field <= 0 → −1` mutantı ölçülmüş
    eşdeğerdir (Ising'de Z2; e5 "h0eksi" 11/11 bit bit aynı; Heisenberg'de bit
    düzeyinde değil, O(3) ile dağılımca -- doğrulama, 2 tohum ~1-1.5σ) ve
    yalnız bu kilit yakalar. Eşik mutantlarına karşı tablo en küçük alt-normal
    negatif alanı da içerir (doğrulama X1: `field < -0.1`).
    Heisenberg'de hizalama davranışça neredeyse eşdeğerdir (`+z` başlangıç
    `h < 0`'da 6/6 gevşiyor, e4 H); onu da yalnız bu kilit yakalar.
    """
    from qlab.engine.spin.lattice import SpinState

    kayit: list[tuple[str, np.ndarray, float]] = []
    asil_metro, asil_wolff = fabrika.run_metropolis, fabrika.run_wolff

    def casus_metro(state, model, **kw):
        kayit.append(("metropolis", state.spins.copy(), kw["field"]))
        return asil_metro(state, model, **kw)

    def casus_wolff(state, **kw):
        kayit.append(("wolff", state.spins.copy(), kw["field"]))
        return asil_wolff(state, **kw)

    monkeypatch.setattr(fabrika, "run_metropolis", casus_metro)
    monkeypatch.setattr(fabrika, "run_wolff", casus_wolff)

    for model, algoritma, h in (
        ("ising", "metropolis", -0.2),
        # EN KÜÇÜK alt-normal negatif alan: `< 0.0` dışındaki HER eşik mutantı
        # (ör. `field < -0.1`, doğrulama X1 -- küçük |h|'de sahte ✓ kanalını
        # yeniden açıyordu) burada +1 başlatır ve ölür; `<= 0.0` M4'tür.
        ("ising", "metropolis", -5e-324),
        ("ising", "metropolis", 0.0),
        ("ising", "metropolis", 0.2),
        ("heisenberg", "metropolis", -0.2),
        ("heisenberg", "metropolis", 0.2),
        ("ising", "wolff", 0.0),
    ):
        kayit.clear()
        _f7_kos(tmp_path, h, model=model, algoritma=algoritma, L=8, sicakliklar=(3.0, 4.0), n_sweep=200, n_burn=0)
        assert [k[0] for k in kayit] == [algoritma, algoritma], F7_MESAJ
        beklenen = np.zeros((8, 8, 3 if model == "heisenberg" else 1))
        beklenen[..., -1] = -1.0 if h < 0 else 1.0
        for _, spins, alan in kayit:
            assert np.array_equal(spins, beklenen), f"{model}/{algoritma} h = {h}: {F7_MESAJ}"
            assert alan == h, f"{model}/{algoritma}: motora giden alan {alan!r}, girdi {h!r}"

    assert {f.name for f in dataclasses.fields(SpinScenario)} == {
        "name", "model", "algorithm", "lattice_size", "temperatures", "n_sweep", "n_burn", "seed",
        "coupling", "field", "heisenberg_step",
    }, F7_MESAJ
    assert list(inspect.signature(build_spin).parameters) == ["scenario"], F7_MESAJ
    assert "BASLANGIC ALANLA HIZALIDIR -- F7 KARARI BUNA DAYANIR" in " ".join(build_spin.__doc__.split()), F7_MESAJ
    assert "h < 0'da alana TERS olur ve hizalama cagiranin isidir" in " ".join(
        SpinState.uniform.__doc__.split()
    ), F7_MESAJ


def test_BINNING_PLATOSU_METASTABILITEYI_DISLAMAZ(tmp_path, monkeypatch):
    """Beyanın KARŞI ÖRNEĞİ kilitli: plato ✓ ile yanlış `⟨|m|⟩` AYNI satırda.

    Kamu başlangıcı yerine `L/4` genişliğinde çapraz şerit (`(i + j) mod L <
    L/4` → +1, aksi −1) konur. Checkerboard Metropolis çapraz şeridi ΔE = 0
    kabulüyle KATI öteler (eleştirmen e3 A: her sweep'te 2 satır kayma), `|m|`
    sabit kalır ve seri plato yapar. ÖLÇÜLDÜ (e6, bu pencere): ⟨|m|⟩ = 0.499998
    ± 1.969e-06, U = +0.666667, Yakınsama ✓ — denge ⟨|m|⟩ ise ~1.0 (e2 REF,
    soğuk kamu, T = 0.6). Genel sayım (e2): 658 takılmanın 68'inde panel ✓.

    Bu yüzden F7 bir tanıya değil başlangıca dayanır
    (`test_KAMU_BASLANGICI_ALANLA_HIZALI_ve_TEK`). Binning ileride bunu
    yakalayacak biçimde değişirse bu test KIRMIZI olur ve `build_spin`
    dokstringindeki "binning platosu metastabiliteyi DISLAMAZ" beyanı
    güncellenir — sessizce eskimez.
    """

    class CaprazSerit(fabrika.SpinState):
        @classmethod
        def uniform(cls, L, n_component):
            i, j = np.indices((L, L))
            return cls(np.where(((i + j) % L) < L // 4, 1.0, -1.0)[..., None])

    monkeypatch.setattr(fabrika, "SpinState", CaprazSerit)
    v = _f7_kos(tmp_path, 0.0, L=32, sicakliklar=(0.6,), n_sweep=1000, n_burn=0, tohum=1)
    assert v["plateau_reached"][0]
    assert v["kaynak"]["converged"] is True
    assert v["magnetisation"][0] < 0.51


# ---------------------------------------------------------------------------
# F2 (11.10.2026): bool/metin reddi, TAŞMA KAPISI, Python yolunda sonluluk.
# Her "ÖLÇÜLDÜ" HEAD ef18615'te, kapı YOKKEN, kamu yolundan alındı (kayıtlar
# depo dışında). Ret testleri hiç sweep koşmaz.
# ---------------------------------------------------------------------------


def _f2_yaml(**ek: str) -> str:
    alanlar = {
        "kind": "spin", "name": "f2", "model": "ising", "algorithm": "metropolis",
        "lattice_size": "4", "temperatures": "[2.0]", "n_sweep": "64",
        "n_burn": "0", "seed": "7",
    }
    alanlar.update(ek)
    return "".join(f"{k}: {v}\n" for k, v in alanlar.items())


@pytest.mark.parametrize(
    "ek,ad",
    [
        (dict(coupling="true"), "coupling"),
        (dict(coupling="yes"), "coupling"),
        (dict(coupling="'2'"), "coupling"),
        (dict(coupling="'1.0e-6'"), "coupling"),
        (dict(field="'0.1'"), "field"),
        (dict(model="heisenberg", heisenberg_step="'0.5'"), "heisenberg_step"),
        (dict(temperatures="['2.0']"), "temperatures[0]"),
        (dict(temperatures="[3.0, true]"), "temperatures[1]"),
        (dict(lattice_size="'4'"), "lattice_size"),
        (dict(lattice_size="4.7"), "lattice_size"),
        (dict(n_sweep="true"), "n_sweep"),
        (dict(n_burn="false"), "n_burn"),
        (dict(seed="'5'"), "seed"),
    ],
)
def test_F2_SAYISAL_ALAN_bool_ve_METNI_SESSIZCE_CEVIRMIYOR(tmp_path, ek, ad):
    """YAML'ın bayrağı ve metni sayıya ÇEVRİLMEZ; mesaj alanı ADIYLA anar.

    ÖLÇÜLDÜ: her satır `load_scenario`dan GEÇİYORDU — `coupling: true`/`yes`
    → 1.0 (CLI rc = 0, panel "J = 1 (girdi)"), `'2'` → 2.0, `'1.0e-6'` →
    1e-06, `[true]` → T = 1.0, `n_sweep: true` → 1, `lattice_size: 4.7` → 4,
    `seed: '5'` → 5.
    """
    with pytest.raises(ScenarioError, match=re.escape(f"'{ad}'")):
        load_scenario(_yaz(tmp_path, _f2_yaml(**ek)))


def test_F2_GERCEK_YAML_SAYILARI_hala_GECIYOR(tmp_path):
    """Kapı fazla sıkı değil — ve PyYAML'ın üstel yazım açığı KIRILMADI.

    PyYAML (YAML 1.1) `1e-6` ve `-5e-324`ü tırnaksız da DİZE döndürür
    (`repr(float)` tam bu biçimi yazar; F7 kilidi `field: -5e-324` yazar).
    Onlar eskiden doğru okunuyordu ve hâlâ sayıdır; tam değerli float da
    tam sayı alanına geçer.
    """
    s = load_scenario(
        _yaz(tmp_path, _f2_yaml(coupling="1e-6", field="-5e-324", n_sweep="64.0"))
    )
    assert (s.coupling, s.field, s.n_sweep) == (1e-6, -5e-324, 64)
    assert s.field < 0.0 and type(s.n_sweep) is int and type(s.coupling) is float
    assert load_scenario(_yaz(tmp_path, _f2_yaml(coupling="1.0e-6"))).coupling == 1e-6


@pytest.mark.parametrize(
    "ek", [dict(coupling="1.0e+308"), dict(field="1.0e+308"), dict(field="-1.0e+308")]
)
def test_F2_TASMA_KAPISI_YAML_da_ve_CLI_da_ADIYLA(tmp_path, capsys, ek):
    """Sonlu ama TAŞAN coupling/field YAML okunurken, adıyla düşer.

    ÖLÇÜLDÜ (CLI, L = 4, 64 ölçüm): üçü de yükleniyordu ve `qlab.run` rc = 1
    ile YALNIZ "enerji serisi sonlu olmali; 64 sonlu olmayan deger var"
    diyordu — hangi girdinin taştığını anmadan, sweep'lerden SONRA.
    """
    yol = _yaz(tmp_path, _f2_yaml(**ek))
    with pytest.raises(ScenarioError, match="enerji ölçeği taşıyor"):
        load_scenario(yol)
    from qlab.run import main

    assert main([str(yol), "-o", str(tmp_path / "x.npz")]) == 1
    hata = capsys.readouterr().err
    assert "enerji ölçeği taşıyor" in hata and "enerji serisi" not in hata


def test_F2_TASMA_ESIGI_GECERLI_KOSUMU_KESMIYOR_ve_TASANI_KESIYOR(tmp_path):
    """Eşik `n_sweep · E_max² <= float max`, `E_max = L²(2J + |h|)`; iki yanı.

    `(J, T) → (2^500 J, 2^500 T)` IEEE'de KESİNDİR: geçen koşumun paneli J = 1
    koşumuyla BİT BİT aynı (kapı geçerli bir koşumu kesmiyor). 2^510'da
    `E_max² · 64` taşar; ÖLÇÜLDÜ (kapı yokken, aynı pencere, J = 1.0e+153):
    rc = 0 ve panelde C=nan, uyarısız. Kapı onu yükleyicide adıyla keser.
    """
    def panel(J, T):
        senaryo = load_scenario(
            _yaz(tmp_path, _f2_yaml(coupling=repr(J), temperatures=f"[{T!r}]"))
        )
        yuk = build_spin_payload(build_spin(senaryo))
        return yuk.magnetisation[0], yuk.binder[0], yuk.heat_capacity[0]

    taban = panel(1.0, 2.0)
    assert panel(2.0**500, 2.0**501) == taban and np.isfinite(taban[2])
    with pytest.raises(ScenarioError, match="enerji ölçeği taşıyor"):
        panel(2.0**510, 2.0**511)


@pytest.mark.parametrize(
    "alanlar,ad",
    [
        (dict(field=float("nan")), "sonlu field"),
        (dict(field=float("inf")), "sonlu field"),
        (dict(model="heisenberg", heisenberg_step=float("nan")), "heisenberg_step"),
        (dict(model="heisenberg", heisenberg_step=float("inf")), "heisenberg_step"),
        (dict(coupling=True), "coupling"),
        (dict(coupling="2"), "coupling"),
        (dict(field="0.1"), "field"),
        (dict(temperatures=(True, 4.0)), "temperatures"),
        (dict(n_sweep=True), "n_sweep"),
        (dict(seed="5"), "seed"),
        (dict(coupling=1e308), "coupling/field"),
        (dict(field=1e308), "coupling/field"),
        (dict(coupling=np.True_), "coupling tek bir gercel"),
        (dict(field=np.array(True)), "field tek bir gercel"),
        (dict(temperatures=(np.True_, 4.0)), "temperatures tek bir gercel"),
        (dict(coupling=np.complex128(2.0)), "coupling tek bir gercel"),
        (dict(n_sweep=np.True_), "n_sweep tam sayi"),
        (dict(seed=np.True_), "seed tam sayi"),
    ],
)
def test_F2_PYTHON_YOLU_SONLULUK_TIP_ve_TASMA_KAPISI(monkeypatch, alanlar, ad):
    """Yükleyici kapılarının Python eşi (`InvalidInputError`), HESAPTAN ÖNCE.

    ÖLÇÜLDÜ (`build_spin` doğrudan, L = 4, 20 ölçüm): field nan/inf ve
    heisenberg_step nan/inf HATASIZ geçiyordu (alan: enerji serisi nan/-inf;
    adım: Heisenberg önerisi DONUK, |m| hep 1); `coupling=True` J = 1 diye
    koşuyordu; `'2'`/`'0.1'` ham `TypeError`; 1e308 enerji serisini -inf
    yapıyordu. `provenance_for` da aynı kapıdan geçer.

    field nan/inf satırları SONLULUK mesajını ("sonlu field") eşler: taşma
    kapısı da nan/inf alanı reddeder ve mesajı "coupling/field" der, yani
    yalnız "field" eşlemek sonluluk denetiminin silinmesini (doğrulayıcının
    M28'i) yakalamıyordu. numpy bool'u (`np.True_`, `np.array(True)`) Python
    `bool`unun alt sınıfı değildir ve kapının ilk sürümünden coupling /
    field / temperatures için 1.0 diye geçiyordu (tam sayı alanlarında zaten
    reddediliyordu — o iki satır gerileme kilidi); `np.complex128` bir
    uyarıyla gerçel kısmına iniyordu.
    """
    monkeypatch.setattr(fabrika, "run_metropolis", lambda *a, **k: pytest.fail("kapi hesaptan sonra"))
    senaryo = dataclasses.replace(UCUZ_METRO, **alanlar)
    with pytest.raises(InvalidInputError, match=re.escape(ad)):
        build_spin(senaryo)
    with pytest.raises(InvalidInputError, match=re.escape(ad)):
        provenance_for(senaryo, plateau_reached=np.ones(2, bool), tau_int=np.ones(2))


#: Eşik testlerinin penceresi: L = 4, 64 ölçüm, tek sıcaklık. Kapı sınırı
#: E_max = 32 J <= sqrt(float max / 64) = 1.676e153, yani J <= 5.237e151.
KUCUK_METRO = SpinScenario(
    name="esik", model="ising", algorithm="metropolis", lattice_size=4,
    temperatures=(2.0,), n_sweep=64, n_burn=0, seed=7,
)
KUCUK_WOLFF = dataclasses.replace(KUCUK_METRO, algorithm="wolff", temperatures=(1.7,))


def _panel(sonuc):
    """(<|m|>, Binder, C); Wolff enerji kaydetmez, C orada `None`."""
    yuk = build_spin_payload(sonuc)
    C = None if yuk.heat_capacity is None else yuk.heat_capacity[0]
    return yuk.magnetisation[0], yuk.binder[0], C


def test_F2_TASMA_ESIGI_IKI_CARPANI_DA_SAYIYOR_YAML_ve_PYTHON(tmp_path):
    """Eşik SIKI: J = 5.2e151 geçer, J = 1.0e152 düşer (L = 4, n = 64).

    E_max = L²·2J: 5.2e151 → 1.664e153 <= 1.676e153 (geçer), 1e152 → 3.2e153
    (düşer). Kapıdan `n_sweep` çarpanı düşerse tavan 1.34e154'e, `L²` düşerse
    E_max 2e152'ye iner ve 1e152 GEÇER — doğrulayıcının M02/M03 (yükleyici)
    ve M18/M27 (fabrika) mutantları 2^500/2^510 çatalından ve 1e308'den
    sağ çıkıyordu, çünkü o çatal iki çarpanı da fark etmeyecek kadar geniş.
    ÖLÇÜLDÜ (kapı yokken, aynı pencere): J = 1e152'de C = inf, rc = 0.

    Geçen koşum J = 1 ile aynı fiziktir (T = 2J): <|m|> ve Binder BİT BİT
    aynı (kabul kararları ΔE/T = 2, 4 oranlarına bağlı), C yalnız son
    basamaklarda ayrışır (ölçüldü: göreli 2.9e-15; J 2'nin kuvveti değil).
    """
    J_gecer, J_duser = 5.2e151, 1.0e152
    taban = _panel(build_spin(KUCUK_METRO))
    assert np.isfinite(taban[2])

    def yaml_senaryo(J):
        return load_scenario(
            _yaz(tmp_path, _f2_yaml(coupling=repr(J), temperatures=f"[{2.0 * J!r}]"))
        )

    for senaryo in (
        yaml_senaryo(J_gecer),
        dataclasses.replace(KUCUK_METRO, coupling=J_gecer, temperatures=(2.0 * J_gecer,)),
    ):
        m, U, C = _panel(build_spin(senaryo))
        assert (m, U) == taban[:2]
        assert C == pytest.approx(taban[2], rel=1e-12)

    with pytest.raises(ScenarioError, match="enerji ölçeği taşıyor"):
        yaml_senaryo(J_duser)
    with pytest.raises(InvalidInputError, match="enerji olcegi tasiyor"):
        build_spin(
            dataclasses.replace(KUCUK_METRO, coupling=J_duser, temperatures=(2.0 * J_duser,))
        )


def test_F2_WOLFF_2J_KAPISI_YAML_da_ve_PYTHON_da(tmp_path, monkeypatch):
    """2J taşarsa Wolff'un bağ olasılığı p = 1 - exp(-2J/T) sessizce 1 olur.

    ÖLÇÜLDÜ (doğrulayıcı, kapının ilk sürümü, kT/J = 1.7): coupling >= 9e307
    Wolff'ta rc = 0 ile <|m|> = 1.000000 ve tau = inf basıyordu — E_max kapısı
    "Wolff enerji ölçmez" diye yalnız Metropolis'teydi. 8e307'de 2J sonludur
    ve panel J = 1 ile BİT BİT aynıdır (bu testte ölçülür); 1e308 iki yolda
    da adıyla, HİÇBİR küme çevrilmeden düşer.
    """
    taban = _panel(build_spin(KUCUK_WOLFF))

    def yaml_senaryo(J):
        return load_scenario(
            _yaz(
                tmp_path,
                _f2_yaml(algorithm="wolff", coupling=repr(J), temperatures=f"[{1.7 * J!r}]"),
            )
        )

    gecen = (
        yaml_senaryo(8e307),
        dataclasses.replace(KUCUK_WOLFF, coupling=8e307, temperatures=(1.7 * 8e307,)),
    )
    for senaryo in gecen:
        assert _panel(build_spin(senaryo))[:2] == taban[:2]

    monkeypatch.setattr(fabrika, "run_wolff", lambda *a, **k: pytest.fail("kapi hesaptan sonra"))
    with pytest.raises(ScenarioError, match=r"'coupling' çok büyük: 2J = inf"):
        yaml_senaryo(1.0e308)
    senaryo = dataclasses.replace(KUCUK_WOLFF, coupling=1.0e308, temperatures=(1.7e308,))
    with pytest.raises(InvalidInputError, match=r"coupling cok buyuk, 2J = inf"):
        build_spin(senaryo)
    with pytest.raises(InvalidInputError, match=r"2J = inf"):
        provenance_for(senaryo, plateau_reached=np.ones(1, bool), tau_int=np.ones(1))


_COK_BUYUK_N = 10**309  # > float max (~1.8e308): float'a çevrilirken OverflowError
_COK_BUYUK_L = 10**155  # L² = 1e310 > float max; çift (Metropolis kuralı)


@pytest.mark.parametrize(
    "alan,deger,desen",
    [
        ("n_sweep", _COK_BUYUK_N, "n_sweep' float aralığını aşıyor"),
        ("lattice_size", _COK_BUYUK_L, "lattice_size' çok büyük"),
    ],
    ids=["n_sweep_1e309", "lattice_size_1e155"],
)
def test_F2_TASMA_KAPISI_DEV_TAM_SAYIDA_HAM_OVERFLOW_SIZDIRMAZ(
    tmp_path, monkeypatch, alan, deger, desen
):
    """Taşma kapısının KENDİSİ float'a çevirirken taşmamalı — alan adıyla düşer.

    ÖLÇÜLDÜ (doğrulayıcı, kapının ilk sürümü): `n_sweep` tam sayı yazımı
    > ~1.8e308 ya da `lattice_size` > ~1.3e154, kapının içinden ham
    `OverflowError: int too large to convert to float` sızdırıyordu (CLI'da
    traceback; `build_spin`/`provenance_for` aynı).
    """
    with pytest.raises(ScenarioError, match=desen):
        load_scenario(_yaz(tmp_path, _f2_yaml(**{alan: str(deger)})))
    monkeypatch.setattr(fabrika, "run_metropolis", lambda *a, **k: pytest.fail("kapi hesaptan sonra"))
    senaryo = dataclasses.replace(KUCUK_METRO, **{alan: deger})
    python_deseni = {"n_sweep": "n_sweep float araligini asiyor", "lattice_size": "lattice_size cok buyuk"}
    with pytest.raises(InvalidInputError, match=python_deseni[alan]):
        build_spin(senaryo)
    with pytest.raises(InvalidInputError, match=python_deseni[alan]):
        provenance_for(senaryo, plateau_reached=np.ones(1, bool), tau_int=np.ones(1))


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
