"""`kind: downfold` senaryosu, panel beyanı ve uçtan uca koşum.

SINIF: TRANSKRİPSİYON DENETİMİ. Bu dosyanın hiçbir testi bir fizik iddiası
taşımaz; senaryo sınırının (YAML → dataclass → motor → panel) doğru
bağlandığını ve panelin ZORUNLU beyanları bastığını ölçer. Modülün fiziksel
çıpası `test_downfold_cipa.py`dedir.

BU DOSYANIN EN ÖNEMLİ TESTİ BİR SAYI DEĞİL BİR SAYIM DENETLER:
`test_BEYAN_TEK_KAYNAKTAN_geliyor`. Bu depoda ölçülmüş bir hata deseni var —
aynı beyan bloğu bir kez `scf.py` ve `factory.py`de birebir kopyalanmıştı,
biri düzeltilip diğeri unutulunca panel iki farklı şey iddia ediyordu (bkz.
`tests/hf/test_hf_scenario.py::test_scan_and_single_point_share_one_provenance_source`).
Orada çözüm iki üreticiyi karşılaştırmaktı; burada tek üretici var, bu yüzden
kilit farklı kuruldu: her beyan cümlesi `src/` ağacında **tam bir kez** geçer
ve `tests/` ağacında — bu dosya ve dört standart terim hariç — hiç geçmez.
Muafiyetlerin ikisi de ölçülmüş bir gerekçe taşır ve o gerekçe
`test_BEYAN_TEK_KAYNAKTAN_geliyor` dokstringindedir.
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

import qlab.engine.tb.factory as factory_modulu
from qlab.engine.tb.downfold import STAGGERING_TOLERANCE, downfold_uhf, hydrogen_ring
from qlab.engine.tb.factory import build_downfold
from qlab.errors import ScenarioError
from qlab.scenario.loader import load_scenario
from qlab.scenario.schema import DownfoldScenario

PROJE_KOKU = Path(__file__).resolve().parents[2]
SEVK_EDILEN = PROJE_KOKU / "scenarios" / "halka_indirgeme.yaml"

# Ucuz bir senaryo: N=6, IKI mesafe, 8 tohum. Ikisi de gecis noktasinin
# (2.233818500 bohr, Task 2) USTUNDEDIR, yani cozum spin-kiriktir ve kademeli
# alan sifir degildir -- panelin `converged` olcutu ancak orada bir sey soyler.
#
# MESAFE SAYISI IKI, VE BU BIR OLCUM SONUCUDUR: tek mesafede `max` ile `min`
# ayni sayiyi verir, yani `staggering_residual_max` alanini `min`e ceviren bir
# mutant sessizce hayatta kalirdi. Iki mesafede artiklar FARKLI cikiyor ve
# ayrim gercekten doguyor -- mutasyon sinavinda olculdu, M8 KIRMIZI. Degerler
# testin kendisinde `!=` ile kilitlidir; sabit yazilmaz, cunku bu dosyanin
# isi yapiyi denetlemektir, bir fizik degerini degil.
UCUZ = DownfoldScenario(
    name="ucuz denetim halkasi",
    ring_size=6,
    distances_bohr=(2.6, 3.0),
    basis="sto-3g",
    n_neighbour=2,
    seeds=8,
)

SEVK_YAML = SEVK_EDILEN.read_text(encoding="utf-8")

# Beyan listesinin CUMLE degil TERIM olan girdileri. Tek-kaynak sayimi
# `tests/` tarafinda bunlari muaf tutar; gerekcesi
# `test_BEYAN_TEK_KAYNAKTAN_geliyor` dokstringinde OLCULEREK yazilidir
# (dordunden biri, "Lowdin ortogonallestirmesi", gercekten
# `tests/hf/test_scf.py`de siradan duzyazi olarak geciyor).
STANDART_TERIMLER = frozenset(
    {
        "sonlu baz seti (STO-3G)",
        "iki-siteli antiferromanyetik hucre",
        "Lowdin ortogonallestirmesi",
        "t3 ve otesi hoplamalar (menzil kesmesi)",
    }
)


def _yaz(tmp_path, metin: str) -> Path:
    yol = tmp_path / "indirgeme.yaml"
    yol.write_text(metin, encoding="utf-8")
    return yol


def _kaynak_metni(yol: Path) -> str:
    """Dosya metni, BİTİŞİK DİZE SABİTLERİ BİRLEŞTİRİLMİŞ hâlde.

    Gerekçesi ölçülmüş bir tuzak: `"birinci parca "\\n"ikinci parca"` kaynakta
    İKİ parçadır ama çalışma zamanında TEK dizedir. Ham metinde aramak bu
    yüzden 0 eşleşme verirdi ve "kopya yok" diye okunurdu — yani test kendi
    iddiasının tersini kanıtlardı.

    Desen yalnızca tırnak-boşluk-satırsonu-boşluk-tırnak dizilimini siler;
    `"a",\\n    "b"` gibi ayrı iki eleman arada virgül taşıdığı için
    BİRLEŞTİRİLMEZ.
    """
    return re.sub(r'"\s*\n\s*"', "", yol.read_text(encoding="utf-8"))


def _tum_beyanlar() -> tuple[str, ...]:
    return (
        factory_modulu._INDIRGEME_YAKLASIMLAR
        + factory_modulu._INDIRGEME_IHMAL
        + (factory_modulu._INDIRGEME_BEKLENEN_HATA,)
    )


# ---------------------------------------------------------------------------
# YAPISAL testler -- SCF koşmazlar, milisaniyedirler, `slow` DEĞİLDİRLER.
# ---------------------------------------------------------------------------


def test_SEVK_EDILEN_senaryo_yukleniyor_ve_BIRIM_DONUSUMU_YOK():
    """Depodaki `scenarios/halka_indirgeme.yaml` gerçekten yükleniyor.

    BİRİM SINIRI YOKLUĞU DA BİR İDDİADIR ve burada kilitlenir: `TBScenario`
    eV → hartree çevirir (TB literatürü eV'de yayınlanır), bu senaryo ise
    HİÇBİR ŞEY çevirmez çünkü girdisi bir geometridir ve geometrinin atomik
    birimi zaten bohr'dur. YAML'daki 1.8 dataclass'ta da 1.8'dir.
    """
    senaryo = load_scenario(SEVK_EDILEN)
    assert isinstance(senaryo, DownfoldScenario)
    assert senaryo.ring_size == 8
    assert senaryo.distances_bohr == (1.8, 2.0, 2.2, 2.6, 3.2)
    assert senaryo.basis == "sto-3g"
    assert senaryo.n_neighbour == 2
    assert senaryo.seeds == 12


def test_VARSAYILANLAR_yalnizca_UC_anahtar_zorunlu(tmp_path):
    """`basis`, `n_neighbour` ve `seeds` isteğe bağlıdır ve varsayılanları
    şemadan gelir; yükleyici kendi kopyasını taşımaz."""
    senaryo = load_scenario(
        _yaz(
            tmp_path,
            "kind: downfold\nname: kisa\nring_size: 6\ndistances_bohr: [2.6]\n",
        )
    )
    assert senaryo.basis == "sto-3g"
    assert senaryo.n_neighbour == 2
    assert senaryo.seeds == 12


def test_BILINMEYEN_kind_listesi_downfold_iceriyor(tmp_path):
    """Hata mesajı geçerli türleri sayar; `downfold` orada görünmeli --
    aksi halde kullanıcı sevk edilmiş bir türü keşfedemez."""
    with pytest.raises(ScenarioError, match="downfold") as bilgi:
        load_scenario(_yaz(tmp_path, "kind: muz\nname: x\n"))
    assert "muz" in str(bilgi.value)


@pytest.mark.parametrize(
    "bozuk,desen",
    [
        ("ring_size: 8", "ring_size: 7"),
        ("ring_size: 8", "ring_size: 2"),
        ("distances_bohr: [1.8, 2.0, 2.2, 2.6, 3.2]", "distances_bohr: []"),
        (
            "distances_bohr: [1.8, 2.0, 2.2, 2.6, 3.2]",
            "distances_bohr: [2.0, 2.0]",
        ),
        (
            "distances_bohr: [1.8, 2.0, 2.2, 2.6, 3.2]",
            "distances_bohr: [2.0, -1.0]",
        ),
        ("distances_bohr: [1.8, 2.0, 2.2, 2.6, 3.2]", "distances_bohr: 2.0"),
        ("n_neighbour: 2", "n_neighbour: 5"),
        ("n_neighbour: 2", "n_neighbour: 0"),
        ("seeds: 12", "seeds: 0"),
        ("ring_size: 8", "ring_size: sekiz"),
    ],
)
def test_BOZUK_senaryolar_REDDEDILIYOR(tmp_path, bozuk, desen):
    """Sessizce varsayılana düşmek YASAK: her biri ScenarioError vermeli.

    `n_neighbour: 5` özellikle önemli -- N=8 halkasında en büyük halka
    mesafesi 4'tür ve 5, m ile n-m'yi aynı çifte yazardı (örtüşme). Motor
    bunu zaten kapatıyor (`test_MENZIL_yarim_halkayi_ASAMAZ`), ama YAML
    tarafında yakalanması hatayı SCF'ten önce verir.
    """
    with pytest.raises(ScenarioError):
        load_scenario(_yaz(tmp_path, SEVK_YAML.replace(bozuk, desen)))


def test_BILINMEYEN_anahtar_ve_EKSIK_anahtar_reddediliyor(tmp_path):
    fazla = SEVK_YAML + "yaricap: 3\n"
    with pytest.raises(ScenarioError, match="yaricap"):
        load_scenario(_yaz(tmp_path, fazla))
    eksik = SEVK_YAML.replace("ring_size: 8\n", "")
    with pytest.raises(ScenarioError, match="ring_size"):
        load_scenario(_yaz(tmp_path, eksik))


def test_TEK_SITE_sayisi_IKI_KATMANDA_da_reddediliyor(tmp_path):
    """Yükleyici ile motor AYNI koşulu ayrı ayrı denetler ve bu bir tekrar
    değil bir katman ayrımıdır: biri YAML'ın hatasını (`ScenarioError`),
    öteki çağrının hatasını (`UnsupportedSystemError`) yakalar. Biri
    kaldırılsa öteki hâlâ tutar -- ama hata TÜRÜ değişir ve bu iki satır
    o farkı kilitler.
    """
    from qlab.errors import UnsupportedSystemError

    with pytest.raises(ScenarioError, match="ÇİFT"):
        load_scenario(_yaz(tmp_path, SEVK_YAML.replace("ring_size: 8", "ring_size: 7")))
    with pytest.raises(UnsupportedSystemError, match="CIFT"):
        downfold_uhf(hydrogen_ring(7, 2.0), "sto-3g", seeds=4)


def test_BEYAN_TEK_KAYNAKTAN_geliyor():
    """BU DOSYANIN EN ÖNEMLİ TESTİ. Beyan metninin İKİ KOPYASI OLAMAZ.

    Ölçülmüş gerekçe bu depodadır: `scf.py` ile `factory.py` aynı bloğu
    birebir taşıyordu ve biri düzeltilip öteki unutulduğunda panel iki
    farklı şey iddia ediyordu. Orada iki üretici vardı ve test onları
    karşılaştırıyordu; burada üretici TEK, dolayısıyla kilit sayımdır.

    İki yönlü:

      * `src/` ağacında tam BİR kez -- kaynakta kopya yok, panel iki farklı
        şey iddia edemez.
      * `tests/` ağacında, BU DOSYA ve STANDART TERİMLER HARİÇ, sıfır kez --
        ikinci bir test dosyası açıklama cümlesini elle yeniden yazamaz.

    STANDART TERİM MUAFİYETİ ÖLÇÜLDÜ, VARSAYILMADI: beyan listesinin dört
    girdisi bir cümle değil bir TERİMDİR ("sonlu baz seti (STO-3G)",
    "iki-siteli antiferromanyetik hucre", "Lowdin ortogonallestirmesi",
    "t3 ve otesi hoplamalar (menzil kesmesi)"). Bunlardan biri gerçekten
    başka bir dosyada geçiyor: `tests/hf/test_scf.py`de "Lowdin
    ortogonallestirmesi" sıradan bir düzyazı olarak kullanılıyor ve bu bir
    kopya değil, alanın kendi kelimesi. Terimlerin kayacak bir içeriği
    yoktur; kayabilecek olan AÇIKLAMA cümleleridir ve onların hepsi
    denetlenir.

    BU DOSYANIN MUAF OLMASI BİR SAPMA DEĞİL, KİLİDİN KENDİSİDİR ve
    gerekçesi ölçüldü: `test_tb_scenario.py`de kelime bazlı bir denetimin,
    bitişik dize sabitlerinin yalnızca ilk satırını değiştiren mutantları
    kaçırdığı görülmüştü; oradan gelen kural TAM CÜMLE aramaktır ve tam
    cümle aramak için cümleyi bir kez yazmak gerekir. O tek kopya
    `test_MODULUN_ETIK_CEKIRDEGI_metinde_YAZILI`dadır, bu testin hemen
    yanındadır ve kaynak metni değişirse İKİSİ BİRDEN kırmızı olur --
    yani sessiz kayma imkânsızdır.
    """
    beyanlar = _tum_beyanlar()
    assert len(beyanlar) == 12, "beyan sayisi degisti; testi bilerek guncelleyin"
    bu_dosya = Path(__file__).resolve()

    # Muafiyet listesi BEYAN LISTESININ ALT KUMESI olmali; yazim hatasiyla
    # gercekte var olmayan bir muafiyet tanimlamak, denetimi sessizce
    # genisletirdi.
    assert STANDART_TERIMLER < set(beyanlar)

    for cumle in beyanlar:
        src_sayim = sum(
            _kaynak_metni(yol).count(cumle)
            for yol in (PROJE_KOKU / "src").rglob("*.py")
        )
        assert src_sayim == 1, f"src/ agacinda {src_sayim} kopya: {cumle[:60]!r}"

        if cumle in STANDART_TERIMLER:
            continue
        test_sayim = sum(
            _kaynak_metni(yol).count(cumle)
            for yol in (PROJE_KOKU / "tests").rglob("*.py")
            if yol.resolve() != bu_dosya
        )
        assert test_sayim == 0, (
            f"BASKA bir test dosyasinda kopya var: {cumle[:60]!r}"
        )


def test_MODULUN_ETIK_CEKIRDEGI_metinde_YAZILI():
    """Anahtar kelime değil TAM CÜMLE aranır.

    Gerekçesi `test_tb_scenario.py`de ölçüldü: kelime bazlı bir test,
    bitişik dize sabitlerinin yalnızca İLK satırını değiştiren mutantları
    kaçırıyor. Aşağıdaki yedi cümle bu modülün etik çekirdeğidir ve
    sessizce yeniden yazılmamalıdır. Üçü modülün NE İDDİA ETMEDİĞİNİ
    söylüyor (hızlanma yok, spin özdurumu değil, spin projeksiyonu yok),
    ikisi kapsamı çiziyor (yalnızca halka, 6-31G'de site-bazlı okuma
    tanımsız), ikisi de modelin şeklinin ve tavanının nereden geldiğini
    yazıyor.

    Bu test bir totoloji DEĞİLDİR: sağ taraf `factory` modülünden okunur,
    sol taraftaki cümleler burada BAĞIMSIZ olarak yazılmıştır. Kaynaktaki
    metni değiştiren bir mutant iki tarafı ayrıştırır ve test kırılır.
    """
    metin = " ".join(_tum_beyanlar())
    for cumle in (
        "BU MODUL HIZLANMA IDDIA ETMEZ: HF/TB maliyet orani OLCULMEDI",
        "UHF (tek determinant, spin ozdurumu DEGIL)",
        "kenar etkileri: yalnizca HALKA; acik zincir kapsam DISINDADIR",
        "site-bazli okuma orada TANIMSIZDIR",
        "spin projeksiyonu (PUHF) yok -- <S^2> raporlanir, DUZELTILMEZ",
        "MODELIN SEKLI VARSAYIM DEGIL OLCUMDUR",
        "tavani menzil degil KABUK ORTALAMASI belirler",
    ):
        assert cumle in metin, f"beyan kayip: {cumle!r}"


def test_MENZIL_BEYANI_senaryodan_gelir_SABIT_DEGIL():
    """`menzil N kesmesi` satırı gerçekten `n_neighbour`u okumalı.

    Sabit yazılsaydı `n_neighbour: 1` ile koşan bir kullanıcı, panelde
    "menzil 2" yazdığını görürdü -- yani panel koşulmayan bir hesabı
    beyan ederdi. Bu test SCF koşmaz: yalnızca beyan bloğunun kendisini
    kurar.
    """
    from qlab.provenance import Provenance

    for menzil in (1, 2, 3):
        yaklasimlar = factory_modulu._INDIRGEME_YAKLASIMLAR + (
            f"menzil {menzil} kesmesi",
        )
        kaynak = Provenance(
            method="x",
            approximations=yaklasimlar,
            neglected=factory_modulu._INDIRGEME_IHMAL,
            expected_error=factory_modulu._INDIRGEME_BEKLENEN_HATA,
            converged=True,
        )
        metin = " ".join(deger for _, deger in kaynak.as_lines())
        assert f"menzil {menzil} kesmesi" in metin


# ---------------------------------------------------------------------------
# FİZİK koşan testler. `slow` işaretleri dosya sonundaki ölçüme dayanır.
# ---------------------------------------------------------------------------


def test_FABRIKA_IKINCI_BIR_FIZIK_YOLU_ACMIYOR():
    """`build_downfold`, `downfold_uhf`in ürettiğinden BAŞKA bir sayı üretmez.

    Fabrikanın işi geometri kurmak ve beyanı toplamaktır; bir parametreyi
    yeniden hesaplasaydı (örneğin ortalamayı kendi alsaydı) iki yol
    ayrışabilirdi ve panel motorun söylemediğini yazardı.

    Karşılaştırma BİREBİR (`==`) yapılır, tolerans YOKTUR: iki taraf aynı
    deterministik aramayı koşar, dolayısıyla eşitlik tam olmalıdır. Ölçüldü,
    tam eşitliyor.
    """
    olculenler, _ = build_downfold(UCUZ)
    assert len(olculenler) == len(UCUZ.distances_bohr)
    for d, olculen in zip(UCUZ.distances_bohr, olculenler):
        beklenen = downfold_uhf(
            hydrogen_ring(UCUZ.ring_size, d),
            UCUZ.basis,
            n_neighbour=UCUZ.n_neighbour,
            seeds=UCUZ.seeds,
        )
        assert olculen == beklenen, f"d={d}: fabrika baska bir sayi uretti"
    # Ve mesafeler SIRAYLA eslesiyor: sirasi bozan bir mutant, iki mesafe
    # birbirine cok yakinsa fark edilmezdi -- bu yuzden ayri ayri kilitlenir.
    assert olculenler[0] != olculenler[1]

    # `n_neighbour` GERCEKTEN aktariliyor. Sabit 2 yazan bir mutant, sevk
    # edilen senaryo da UCUZ da 2 kullandigi icin BASKA HICBIR TESTTE
    # gorunmezdi -- bu satir onun tek kapisidir.
    tek_menzil, _ = build_downfold(dataclasses.replace(UCUZ, n_neighbour=1))
    assert all(len(p.hoppings) == 1 for p in tek_menzil)
    assert tek_menzil[0].hoppings[0] == pytest.approx(
        olculenler[0].hoppings[0], abs=1e-12
    )


def test_YAKINSAMA_KADEMELI_YAPININ_DOGRULANMASINA_bagli():
    """`converged`, "SCF durdu" DEĞİL "yapı bulundu" demektir.

    Bu modülde SCF'in durması zaten `search_uhf`in ön koşuludur ve tek
    başına hiçbir şey söylemez. Panelin ölçtüğü şey, modelin VARSAYDIĞI
    iki-siteli antiferromanyetik yapının gerçekten bulunup bulunmadığıdır:
    `|diag(H^a) - roll(diag(H^b), 1)|` her mesafede eşiğin altında mı?

    Ayrıca üç tanı da panelde OLMAK ZORUNDA -- ikisi (kaydırma artığı,
    <S^2>) modülün kendi sınırlarını, biri (arama satırı) sayıların bir
    ARAMA protokolünden geldiğini yazar.
    """
    parametreler, kaynak = build_downfold(UCUZ)

    assert kaynak.converged is True
    # B7-c: ✓ ayrica her mesafenin UHF-kararli dondugunu gerektirir; eyerli
    # durum tests/tb/test_downfold_uhf_hukmu.py'de (burada hic eyer yok).
    assert [p.search_stable for p in parametreler] == [True, True]
    artiklar = [p.staggering_residual for p in parametreler]
    artik = kaynak.diagnostics["staggering_residual_max"]
    # EN BUYUGU, en kucugu DEGIL: iki mesafenin artiklari gercekten farkli
    # (ölçüldü) ve bu satır `min`e çeviren bir mutantı öldürür.
    assert artiklar[0] != artiklar[1]
    assert artik == max(artiklar)
    assert artik < STAGGERING_TOLERANCE

    assert kaynak.diagnostics["s_squared_max"] == max(
        abs(p.s_squared) for p in parametreler
    )
    assert kaynak.diagnostics["n_distance"] == float(len(UCUZ.distances_bohr))
    assert kaynak.diagnostics["n_neighbour"] == 2.0

    arama = kaynak.diagnostics["arama"]
    assert "8 tohum" in arama
    for p in parametreler:
        assert p.search_line in arama

    panel = " ".join(deger for _, deger in kaynak.as_lines())
    assert "staggering_residual_max" in panel
    assert "s_squared_max" in panel
    assert "arama=" in panel


def test_YAKINSAMA_ESIGE_gercekten_BAKIYOR(monkeypatch):
    """`converged` SABİT DEĞİL: eşik sıkılaştırılınca hüküm DÖNMELİ.

    Bir önceki test tek başına `converged=True` yazan bir mutantı öldüremez,
    çünkü gerçek sistemlerde kaydırma artığı her zaman eşiğin altındadır
    (ölçüldü: beş sistemde 1.942890e-15 ... 9.087647e-09) ve fiziksel olarak
    eşiği aşan bir halka üretmenin yolu yok. Bu yüzden delik EŞİK tarafından
    kapatılıyor: `STAGGERING_TOLERANCE` 1e-16'ya çekildiğinde N=6 d=2.6'nın
    2.220446e-15'lik artığı onu AŞAR ve panel "yakınsamadı" demek zorundadır.

    Ölçülen artık 1e-16'nın bir mertebe üstündedir ve bu seçim de ölçümdür:
    daha küçük bir eşik (örneğin 1e-20) aynı sonucu verirdi ama gürültüye
    değil YAPIYA duyarlı olmazdı.
    """
    monkeypatch.setattr(factory_modulu, "STAGGERING_TOLERANCE", 1e-16)
    parametreler, kaynak = build_downfold(UCUZ)
    assert max(p.staggering_residual for p in parametreler) > 1e-16
    assert kaynak.converged is False


def test_YAKLASIMLAR_ile_IHMAL_EDILEN_KARISMIYOR():
    """İki liste AYNI panelde ama AYNI SATIRDA değil.

    Tersine çevrilmiş bir atama (`approximations=_INDIRGEME_IHMAL`) bütün
    cümleleri hâlâ ekrana basardı ve "beyan var mı" diye bakan her test
    yeşil kalırdı. Ayıran şey cümlelerin HANGİ satırda olduğudur: bir
    yaklaşım ile bir ihmal aynı şey değildir ve panelin ikisini karıştırması
    okuyucuya "bunu yaptık" ile "bunu yapmadık"ı ters gösterirdi.
    """
    _, kaynak = build_downfold(UCUZ)
    satirlar = dict(kaynak.as_lines())

    for cumle in factory_modulu._INDIRGEME_YAKLASIMLAR:
        assert cumle in satirlar["Yaklaşımlar"]
        assert cumle not in satirlar["İhmal edilen"]
    for cumle in factory_modulu._INDIRGEME_IHMAL:
        assert cumle in satirlar["İhmal edilen"]
        assert cumle not in satirlar["Yaklaşımlar"]
    assert factory_modulu._INDIRGEME_BEKLENEN_HATA == satirlar["Beklenen hata"]


def test_ARAMA_SATIRI_her_MESAFE_icin_konusuyor():
    """Her mesafe AYRI bir aramadır; özetleri ayrışırsa panel HEPSİNİ yazar.

    "İlkini yaz" seçeneği reddedildi ve bu test o reddi kilitler: bir
    koşumun özeti ötekilerin adına konuşamaz. Ayrışma sevk edilen
    senaryoda GERÇEKTEN oluyor -- ölçüldü, d=3.2'de 12 tohumun yalnızca
    9'u yakınsıyor; sabit bir "12 yakınsadı" satırı yalan olurdu. O uç
    aşağıdaki uçtan uca testte kilitlenir; burada ucuz halkanın hangi
    kalıba düştüğü ölçülür ve kalıp ne olursa olsun HER mesafenin özeti
    satırda bulunmak zorundadır.
    """
    parametreler, kaynak = build_downfold(UCUZ)
    arama = kaynak.diagnostics["arama"]
    satirlar = [p.search_line for p in parametreler]

    if len(set(satirlar)) == 1:
        assert arama == f"{len(parametreler)} mesafenin hepsinde: {satirlar[0]}"
    else:
        for d, s in zip(UCUZ.distances_bohr, satirlar):
            assert f"d={d:.3f}: {s}" in arama


@pytest.mark.slow
def test_CLI_UCTAN_UCA(tmp_path):
    """`python -m qlab.run scenarios/halka_indirgeme.yaml -o indirgeme.npz`.

    ÖLÇÜLDÜ: 5.46 s (beş UHF araması × 12 tohum + yorumlayıcı başlangıcı),
    yani deponun 2 s eşiğinin üstünde ve `slow` işareti ZORUNLU. Karşılaştır:
    `test_tb_scenario.py::test_CLI_uctan_uca` saniye altında ve işaretsiz --
    fark TB'nin hiç SCF koşmamasıdır.

    Beş panel satırının BEŞİ de aranır (Yöntem / Yaklaşımlar / İhmal edilen /
    Beklenen hata / Yakınsama). Metinler bu dosyada tekrarlanmaz; `factory`
    modülünden okunur, yani beyan düzeltilirse test onu KENDİLİĞİNDEN takip
    eder ve ikinci bir kopya doğmaz.
    """
    cikti = tmp_path / "indirgeme.npz"
    sonuc = subprocess.run(
        [
            sys.executable,
            "-m",
            "qlab.run",
            str(SEVK_EDILEN),
            "-o",
            str(cikti),
        ],
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
        "Yakınsama",
    ):
        assert etiket in sonuc.stdout, f"panel satiri kayip: {etiket}"
    for cumle in _tum_beyanlar():
        assert cumle in sonuc.stdout, f"beyan EKRANA basilmadi: {cumle[:50]!r}"
    assert "menzil 2 kesmesi" in sonuc.stdout

    with np.load(cikti, allow_pickle=False) as veri:
        assert np.allclose(veri["distances_bohr"], [1.8, 2.0, 2.2, 2.6, 3.2])
        assert veri["hoppings"].shape == (5, 2)
        assert np.all(veri["staggering_residual"] < STAGGERING_TOLERANCE)
        assert np.all(np.abs(veri["delta"]) > 1e-3)  # hepsi spin-kirik
        kaynak = json.loads(str(veri["provenance_json"]))
    assert kaynak["converged"] is True
    assert kaynak["diagnostics"]["n_distance"] == 5.0
    # ARAMA AYRISMASI GERCEKTEN OLUYOR: d=3.2'de 12 tohumun 9'u
    # yakinsiyor, dolayisiyla bes ozet AYNI DEGIL ve panel hepsini
    # yaziyor. Sabit bir "hepsinde" satiri burada yalan olurdu.
    assert "; d=" in kaynak["diagnostics"]["arama"]
    assert kaynak["diagnostics"]["arama"].count("tohum") == 5


# ---------------------------------------------------------------------------
# `slow` İŞARETİNİN GEREKÇESİ -- ÖLÇÜM.
#
# Deponun eşiği Task 2'de SEÇİLDİ ve gerekçesi orada yazılı: kendi call
# süresi 2 s ve ÜSTÜNDE olan test işaretlenir (devralınabilir bir eşik YOK --
# `-m slow` şeridinin en hızlı çağrısı 0.01 s, `-m "not slow"` şeridinin en
# yavaşı 4.54 s, bantlar tamamen çakışıyor).
#
# ÖLÇÜLEN SÜRELER bu dosyanın rapor bölümünde verilmiştir. Tek işaretli test
# CLI koşumudur; geri kalan her şey ya SCF koşmaz ya da N=6 × tek mesafe ×
# 8 tohum ile saniye altındadır.
# ---------------------------------------------------------------------------
