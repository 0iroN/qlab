"""Modül 7 / Task 1 — kova tablosunun KODDAN TÜRETİLDİĞİNİN sınavı.

Bu dosyanın tek işi şu cümleyi makineye zorlatmak: **bayat bir yokluk
beyanı bir yalandır.** Tablo elle listelenirse ilk değişiklikte sessizce
yanlış olur; bu yüzden `olculen_kenarlar()` kaynak ağacını tarar ve tablo
ona karşı sınanır.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest

from qlab.scenario.loader import load_scenario
from qlab.bridge_map import (
    HICBIR_SAYI_GELMEDI,
    KOVA_SAYILARI,
    KOVA_TABLOSU,
    KOVALAR,
    MODULLER,
    EdgeFact,
    Kova,
    olculen_kenarlar,
)

PROJE_KOKU = Path(__file__).resolve().parents[2]
KAYNAK = PROJE_KOKU / "src" / "qlab"
MOTOR_KOKU = KAYNAK / "engine"

#: 30.08.2026'da (HEAD = b221a9c) ölçülen çapraz-motor kenar kümesi.
BEKLENEN_KENARLAR = {("md", "hf"), ("tb", "hf")}


# --- Bağımsız tarayıcı -------------------------------------------------------
# Bilinçli olarak `bridge_map`in iç yardımcılarını KULLANMIYOR: `units`
# ölçümü ile kenar ölçümü aynı makineye dayanırsa, o makinenin bir hatası
# iki ölçümü de aynı yöne kaydırır ve çıpa dairesel olur.


def _import_edilen(dosya: Path) -> set[str]:
    """Bir dosyanın GERÇEKTEN import ettiği mutlak modül adları."""
    agac = ast.parse(dosya.read_text(encoding="utf-8"), filename=str(dosya))
    adlar: set[str] = set()
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.Import):
            adlar.update(ad.name for ad in dugum.names)
        elif isinstance(dugum, ast.ImportFrom) and dugum.level == 0 and dugum.module:
            adlar.add(dugum.module)
    return adlar


def _motor_paketleri() -> list[Path]:
    return sorted(
        p for p in MOTOR_KOKU.iterdir() if p.is_dir() and not p.name.startswith("__")
    )


def _tutarsizliklar(tarama: set[tuple[str, str]]) -> list[str]:
    """Taranan kenar kümesi ile tablonun ÇELİŞTİĞİ noktalar.

    İki yönlü kilit:

    1. Taranan her kod kenarı tabloda bir çift olmalı ve o çift
       `YAPISAL_YOK` ya da `OLCULMEDI` kovasında OLAMAZ — kodun fiilen
       bağladığı bir çifti "yapısal olarak yok" ya da "hiç ölçülmedi" diye
       ilan etmek düpedüz yalandır.
    2. `GERCEK_KOPRU` kovasındaki her çiftin bir kod kenarı OLMALI — kodda
       karşılığı olmayan bir köprü iddia edilemez.
    """
    kova_of = {f.cift: f.kova for f in KOVA_TABLOSU}
    sorunlar: list[str] = []
    for kaynak, hedef in sorted(tarama):
        cift = tuple(sorted((kaynak, hedef)))
        if cift not in kova_of:
            sorunlar.append(f"taranan kenar {cift} tabloda YOK")
        elif kova_of[cift] in (Kova.YAPISAL_YOK, Kova.OLCULMEDI):
            sorunlar.append(
                f"taranan kenar {cift} tabloda {kova_of[cift]} kovasında — "
                "kodun bağladığı bir çift 'yok' ya da 'ölçülmedi' olamaz"
            )
    yonsuz_tarama = {tuple(sorted(k)) for k in tarama}
    for f in KOVA_TABLOSU:
        if f.kova == Kova.GERCEK_KOPRU and f.cift not in yonsuz_tarama:
            sorunlar.append(
                f"GERCEK_KOPRU {f.cift} için kaynak ağacında kod kenarı YOK"
            )
    return sorunlar


def _sahte_motor_agaci(kok: Path, dosyalar: dict[str, str]) -> Path:
    """Sentetik bir `engine/` ağacı kur: {'spin/x.py': '<kaynak>'}."""
    for yol, govde in dosyalar.items():
        hedef = kok / yol
        hedef.parent.mkdir(parents=True, exist_ok=True)
        hedef.write_text(govde, encoding="utf-8")
    return kok


# --- 1. TAŞIYICI TEST --------------------------------------------------------


def test_KOVA_TABLOSU_KODDAN_TURETILIYOR_elle_listelenmiyor():
    """BU DOSYANIN EN ÖNEMLİ TESTİ.

    Bir yokluk beyanı BAYATLARSA BİR YALANDIR. "Çapraz-motor import sayısı
    2'dir" iddiası elle yazılırsa ilk değişiklikte sessizce yanlış olur.

BU TUZAĞIN BİR ÖRNEĞİ DEPODA ÖLÇÜLDÜ (`test_MD_FORCE_FIELD_KINDS_...`
    testine bakınız): `scenario/schema.py`nin `MD_FORCE_FIELD_KINDS` demeti
    hem ölüydü hem bayattı. 30.08.2026'da kapatıldı, ve kapatma biçimi bu
    dosyanın fikrini doğruluyor: demet SİLİNMEDİ, kaynağa karşı sınanan bir
    TÜRETME KİLİDİNE bağlandı. Aynı desen burada TEKRARLANMAYACAK.

    ÖLÇÜLDÜ (30.08.2026, HEAD = b221a9c): tam İKİ kenar,
    `md → hf` (`engine/md/forcefields.py:13`) ve
    `tb → hf` (`engine/tb/downfold.py:130-132`). 21 çiftin 19'unda sıfır
    bağ, ve iki kenarın ikisi de aynı düğüme gidiyor — yapı bir merdiven
    değil, merkezi HF olan İKİ ÇUBUKLU BİR YILDIZ.

    PLANIN "tablo = `SIFIR_KENAR` olmayan girdiler" FİLTRESİ ÇÜRÜDÜ, iki
    kez: (a) beş kovanın hiçbirinin adı `SIFIR_KENAR` değil — plan Step 1
    o adı kullanıyor, Step 3 ise beş kovayı `GERCEK_KOPRU · TAUTOLOJI ·
    YAPISAL_YOK · BENZERLIK · OLCULMEDI` diye sayıyor, yani o satır
    `AttributeError` verirdi; (b) ad, spec §3'ün "hükümsüz (sıfır kod
    kenarı)" satırı (8 çift) sayılsa bile filtre 21 − 8 = 13 çift
    döndürür ve taranan 2 kenarla ASLA eşleşmez — test yazıldığı hâliyle
    hiçbir depoda yeşil olamazdı. Türetme kilidinin doğru biçimi
    `_tutarsizliklar()` içindeki İKİ YÖNLÜ ilişkidir.
    """
    tarama = set(olculen_kenarlar())
    assert tarama == BEKLENEN_KENARLAR, (
        f"kaynak ağacı {sorted(tarama)} diyor, beklenen "
        f"{sorted(BEKLENEN_KENARLAR)}"
    )
    assert _tutarsizliklar(tarama) == []


def test_YENI_BIR_CAPRAZ_MOTOR_IMPORTU_TABLOYU_KIRMIZIYA_CEVIRIR(tmp_path):
    """Türetme kilidinin GERÇEKTEN ısırdığını göster.

    Kilit yalnız bugünün ağacında yeşil kalıyorsa hiçbir şey kilitlemiyor
    demektir. Burada sentetik bir ağaçta `spin → em` import'u yaratılıyor;
    o çift tabloda `OLCULMEDI` kovasında oturuyor ve tarama onu bulunca
    tutarsızlık raporu DOLU dönmeli.

    KENAR BİLİNÇLİ OLARAK `__init__.py` İÇİNDE: gerçek ağaçta hiçbir
    `engine/*/__init__.py` çapraz-motor import etmiyor, yani taramadan
    `__init__.py`yi eleyen bir mutant o ağaçta HAYATTA KALIRDI. Kenarı
    oraya koyarak o açık kapatılıyor.
    """
    kok = _sahte_motor_agaci(
        tmp_path / "engine",
        {
            "spin/__init__.py": "from qlab.engine.em.grid import Grid\n",
            "em/__init__.py": "",
        },
    )
    tarama = set(olculen_kenarlar(kok))
    assert tarama == {("spin", "em")}
    sorunlar = _tutarsizliklar(tarama)
    assert sorunlar, "yeni kenar tabloyu kırmızıya çevirmedi"
    assert "OLCULMEDI" in sorunlar[0]


# --- 2. Kapsama ve sayım -----------------------------------------------------


def test_YIRMI_BIR_CIFTIN_TAMAMI_BIR_KOVADA():
    """Hiçbir çift SESSİZCE düşmemeli, ve hiçbiri İKİ KEZ sayılmamalı."""
    ciftler = {tuple(sorted(p)) for p in combinations(MODULLER, 2)}
    assert len(ciftler) == 21
    kapsanan = [f.cift for f in KOVA_TABLOSU]
    assert len(kapsanan) == 21, "tabloda 21'den farklı sayıda girdi var"
    assert len(set(kapsanan)) == 21, "aynı çift birden çok kez listelenmiş"
    assert set(kapsanan) == ciftler


def test_KOVA_SAYILARI_1_4_8_0_8_ve_BENZERLIK_BOS_KALMASI_BIR_OLCUM():
    """ÖLÇÜLDÜ: GERÇEK KÖPRÜ 1 · TAUTOLOJİ 4 · YAPISAL YOK 8 ·
    BENZERLİK 0 · ÖLÇÜLMEDİ 8. Toplam 21 = C(7, 2).

    BENZERLİK kovasının boş olması bir ihmal değil bir ÖLÇÜM SONUCUdur;
    sayısı da bu yüzden kilitli — biri oraya "biçimsel analoji" diye bir
    satır eklerse bu test kırmızı olur ve o satırın ölçümü sorulur.
    """
    sayim = Counter(f.kova for f in KOVA_TABLOSU)
    for kova in KOVALAR:
        assert sayim.get(kova, 0) == KOVA_SAYILARI[kova], f"{kova} sayısı tutmuyor"
    assert sum(KOVA_SAYILARI.values()) == 21


def test_KOVA_ADLARI_KOVALAR_ICINDE_ve_sabitin_degeri_ADIYLA_AYNI():
    """Bir yazım hatası (`TAUTOLOJII`) sessizce yeni bir kova yaratamaz."""
    assert len(KOVALAR) == 5
    assert len(set(KOVALAR)) == 5
    for f in KOVA_TABLOSU:
        assert f.kova in KOVALAR, f"bilinmeyen kova: {f.kova}"
    # Sabitin ADI ile DEĞERİ aynı olmalı; aksi hâlde paneldeki metin ile
    # koddaki dal ayrışır ve fark sessiz kalır.
    for ad in ("GERCEK_KOPRU", "TAUTOLOJI", "YAPISAL_YOK", "BENZERLIK", "OLCULMEDI"):
        assert getattr(Kova, ad) == ad


# --- 3. ÖLÇÜLMEDİ ayrı bir kovadır -------------------------------------------


def test_OLCULMEDI_AYRI_BIR_KOVADIR_ve_HICBIR_SAYI_TASIMAZ():
    """Spec §7.2: "ÖLÇÜLDÜ, SIFIR ÇIKTI" ile "HİÇ ÖLÇÜLMEDİ" AYNI HÜCREYE
    YAZILAMAZ.

    Sekiz çift (tdse-md, tdse-tb, tdse-spin, hf-spin, md-tb, md-em, tb-em,
    spin-em) ikincidedir: o soruyu ölçen ajan API bağlantısı kopunca öldü
    ve hiçbir sayı gelmedi. Bu bir YOKLUK BEYANIDIR, bir sonuç DEĞİL.

    Kilit üç parçalı: (a) çiftler tam olarak o sekiz, (b) hepsinin
    `buyukluk`u `None` — çünkü hiçbir sayı gelmedi, (c) beyan dizgesi
    YALNIZCA bu kovada geçiyor. (c) olmadan bu kovayı `YAPISAL_YOK`a
    katlayan bir mutant sadece bir sayaç değiştirir; (c) ile birlikte
    metnin kendisi de taşınmak zorunda kalır.
    """
    olculmedi = [f for f in KOVA_TABLOSU if f.kova == Kova.OLCULMEDI]
    assert {f.cift for f in olculmedi} == {
        ("md", "tdse"),
        ("tb", "tdse"),
        ("spin", "tdse"),
        ("hf", "spin"),
        ("md", "tb"),
        ("em", "md"),
        ("em", "tb"),
        ("em", "spin"),
    }
    assert len(olculmedi) == 8
    # Beyan dizgesi HARFİ HARFİNE aranıyor, `HICBIR_SAYI_GELMEDI` sabiti
    # ÜZERİNDEN DEĞİL: gerekçeler o sabitten kuruluyor, dolayısıyla sabiti
    # "ölçüldü, sıfır çıktı" diye değiştiren bir mutant sabit üzerinden
    # yazılan bir assert'i BİRLİKTE kaydırır ve hayatta kalırdı.
    assert HICBIR_SAYI_GELMEDI == "HICBIR SAYI GELMEDI"
    for f in olculmedi:
        assert f.buyukluk is None, f"{f.cift} ölçülmedi ama bir sayı taşıyor"
        assert "HICBIR SAYI GELMEDI" in f.gerekce
        assert "SIFIR CIKTI" not in f.gerekce.replace(
            "'OLCULDU, SIFIR CIKTI' ile 'HIC OLCULMEDI' AYNI HUCREYE YAZILAMAZ", ""
        )
    digerleri = [f for f in KOVA_TABLOSU if f.kova != Kova.OLCULMEDI]
    for f in digerleri:
        assert "HICBIR SAYI GELMEDI" not in f.gerekce, (
            f"{f.cift} ÖLÇÜLMEDİ kovasında değil ama o kovanın beyanını taşıyor"
        )


def test_YAPISAL_YOK_MERTEBELERI_DEPODAKI_BEYANLA_ve_CODATA_ILE_AYNI():
    """Taşınan her sayı bu depoda GERÇEKTEN var mı — iki bağımsız yoldan.

    (a) `F_g/F_e = 8.093550856608016e-37` dizgesi `engine/grav/factory.py`
        içindeki sevk edilmiş YOKLUK BEYANINDA birebir geçiyor (metin
        olarak okundu, import edilmedi — `bridge_map` motorlara bağlanamaz).
    (b) Aynı sayı CODATA 2018 sabitlerinden burada yeniden türetildi.
        İKİ PROTON BİR SEÇİMDİR: elektron-proton için 4.407885560760525e-40.
    (c) `hf → tdse` mertebesi 1600³ = 4.096e+09 nokta; complex128 tek dizi
        4.096e9 × 16 = 65 536 000 000 bayt = 61.04 GiB. Spec "61.0 GB"
        yazıyor; ölçüm o sayının GiB olduğunu söylüyor (ondalık GB
        karşılığı 65.536). SPAN VE ÇÖZÜNÜRLÜK BİRER DÜĞMEDİR.

    `em → tdse` bilinçli olarak `None`: eksik olan sayı değil KARŞILIK.

    HER `approx` ÇAĞRISINDA `abs=0.0` YAZILI — VE BU MUTASYONLA ÖLÇÜLDÜ.
    `pytest.approx`ın varsayılan MUTLAK toleransı `1e-12`dir; buradaki
    sayılar 1e-36 ve 1e-40 mertebesinde olduğu için `rel=1e-12` tek başına
    HİÇBİR ŞEY kilitlemiyordu. İlk turda `_GRAV_ORANI`yi `e-37`den `e-36`ya
    çeviren mutant (on kat!) **HAYATTA KALDI**; `abs=0.0` eklendikten sonra
    öldü. Modül 6'nın "`pytest.approx`ın varsayılanı bir pencere
    düğmesidir" dersinin bu depodaki ikinci örneği.
    """
    beyan = (MOTOR_KOKU / "grav" / "factory.py").read_text(encoding="utf-8")
    assert "8.093550856608016e-37" in beyan

    G = 6.67430e-11
    m_p = 1.67262192369e-27
    m_e = 9.1093837015e-31
    e = 1.602176634e-19
    eps0 = 8.8541878128e-12
    k_e = 1.0 / (4.0 * 3.141592653589793 * eps0)
    iki_proton = G * m_p * m_p / (k_e * e * e)
    assert iki_proton == pytest.approx(8.093550856608016e-37, rel=1e-12, abs=0.0)
    assert G * m_p * m_e / (k_e * e * e) == pytest.approx(
        4.407885560760525e-40, rel=1e-12, abs=0.0
    )

    grav_girdileri = [f for f in KOVA_TABLOSU if "grav" in f.cift]
    assert len(grav_girdileri) == 6
    for f in grav_girdileri:
        assert f.kova == Kova.YAPISAL_YOK
        assert f.buyukluk == pytest.approx(iki_proton, rel=1e-12, abs=0.0)

    (hf_tdse,) = [f for f in KOVA_TABLOSU if f.cift == ("hf", "tdse")]
    assert hf_tdse.buyukluk == 1600**3 == 4.096e09
    assert 1600**3 * 16 / 2**30 == pytest.approx(61.04, abs=0.01)

    (em_tdse,) = [f for f in KOVA_TABLOSU if f.cift == ("em", "tdse")]
    assert em_tdse.kova == Kova.YAPISAL_YOK
    assert em_tdse.buyukluk is None

    yapisal = [f for f in KOVA_TABLOSU if f.kova == Kova.YAPISAL_YOK]
    assert len(yapisal) == 8
    assert sum(1 for f in yapisal if f.buyukluk is not None) == 7


# --- 4. `ast` mi regex mi ----------------------------------------------------


def test_AST_YORUMDAKI_ve_DIZGEDEKI_IMPORTU_SAYMIYOR_REGEX_SAYIYOR(tmp_path):
    """`ast` seçiminin regex'e karşı ÜSTÜNLÜĞÜ — ölçülerek gösterilir.

    Sentetik ağaçta `spin/sahte.py` hiçbir şey import etmiyor; içinde
    yalnızca (a) bir yorum satırı, (b) bir dokstring ve (c) bir dizge
    sabiti `qlab.engine.em` metnini taşıyor. `olculen_kenarlar()` SIFIR
    kenar bulmalı; aynı ağaca uygulanan düz metin taraması ise
    `spin → em` uydurmalı.

    NEDEN ÖNEMLİ: uydurulmuş bir kenar tabloyu KIRMIZIYA çevirirdi (o çift
    `OLCULMEDI` kovasında) — yani regex tabanlı bir tarama, hiç kimse
    hiçbir şey yapmadan yokluk beyanını bozardı.
    """
    govde = (
        '"""Bu dokstring qlab.engine.em adini yaziyor ama import etmiyor."""\n'
        "# from qlab.engine.em.grid import Grid  <- yorum satiri\n"
        'YOL = "qlab.engine.em.medium"\n'
    )
    kok = _sahte_motor_agaci(
        tmp_path / "engine",
        {"spin/__init__.py": "", "spin/sahte.py": govde, "em/__init__.py": ""},
    )

    assert olculen_kenarlar(kok) == frozenset()

    desen = re.compile(r"qlab\.engine\.([A-Za-z_][A-Za-z_0-9]*)")
    regex_kenarlar = set()
    for paket in sorted(p for p in kok.iterdir() if p.is_dir()):
        for dosya in sorted(paket.glob("*.py")):
            for m in desen.finditer(dosya.read_text(encoding="utf-8")):
                if m.group(1) != paket.name:
                    regex_kenarlar.add((paket.name, m.group(1)))
    assert regex_kenarlar == {("spin", "em")}, "regex yanlış-pozitif üretmedi"
    assert _tutarsizliklar(regex_kenarlar), (
        "regex'in uydurduğu kenar tabloyu kırmızıya çevirmiyor — "
        "o hâlde bu testin gerekçesi yanlış"
    )


# --- 5. `units` ekseni -------------------------------------------------------


def test_UNITS_EKSENI_IKI_KENARIN_DUGUM_KUMESIYLE_AYNI():
    """Spec §2.2'nin TAŞIYICI İDDİASI — ve planın bir sayısının çürümesi.

    Bir merdiven tüm basamakların TEK BİR MUTLAK EKSENE dizilmesini ister.
    `qlab.units`e dokunan modül kümesi ölçüldü ve tam olarak `{hf, md, tb}`
    çıktı — yani iki çapraz-motor kenarının kapsadığı düğüm kümesinin
    AYNISI. Kalan dört modül (tdse, spin, em, grav) hem eksensiz hem
    bağsız: merdiven fizikten ÖNCE TANIMDAN yok.

    ÇÜRÜYEN SAYI: plan ve spec `md` için **4** dosya yazıyor
    (`factory.py`, `forcefields.py`, `thermostats.py`, `__init__.py`).
    `ast` ile ölçüm **3** diyor. Sebep `engine/md/__init__.py:10`: o
    dokstring `qlab.units.amu_to_electron_mass` ADINI yazıyor ama modülü
    import ETMİYOR. Düz metin taraması 4, `ast` taraması 3 sayar — yani
    planın sayısı bir METİN TARAMASI ARTIFAKTIDIR ve bu, `ast` seçiminin
    gerçek depodaki ölçülmüş kanıtıdır.

    YÜK TAŞIYAN İDDİA AYAKTA: küme iki yöntemde de `{hf, md, tb}`.
    Çürüyen şey modül başına SAYIDIR, kümenin kendisi değil.
    """
    ast_sayim: dict[str, int] = {}
    metin_sayim: dict[str, int] = {}
    for paket in _motor_paketleri():
        ast_hit = 0
        metin_hit = 0
        for dosya in sorted(paket.glob("*.py")):
            if any(m.split(".")[:2] == ["qlab", "units"] for m in _import_edilen(dosya)):
                ast_hit += 1
            if "qlab.units" in dosya.read_text(encoding="utf-8"):
                metin_hit += 1
        ast_sayim[paket.name] = ast_hit
        metin_sayim[paket.name] = metin_hit

    assert ast_sayim == {
        "em": 0,
        "grav": 0,
        "hf": 1,
        "md": 3,
        "spin": 0,
        "tb": 2,
        "tdse": 0,
    }
    # Planın/spec'in sayısı: metin taraması. Farkın TEK kaynağı md.
    assert metin_sayim["md"] == 4
    assert {k: v for k, v in metin_sayim.items() if k != "md"} == {
        k: v for k, v in ast_sayim.items() if k != "md"
    }
    govde = (MOTOR_KOKU / "md" / "__init__.py").read_text(encoding="utf-8")
    assert "qlab.units.amu_to_electron_mass" in govde
    assert "import" not in govde.replace("import etmez", "")

    eksenli = {k for k, v in ast_sayim.items() if v > 0}
    assert eksenli == {"hf", "md", "tb"}
    dugumler = {m for kenar in olculen_kenarlar() for m in kenar}
    assert eksenli == dugumler, (
        "spec §2.2'nin taşıyıcı iddiası: eksen kümesi = kenarların düğüm kümesi"
    )
    assert {"tdse", "spin", "em", "grav"} == set(MODULLER) - eksenli


# --- 6. `observable/` bağlaşım eklemiyor -------------------------------------


def test_OBSERVABLE_KATMANI_BAGLASIM_EKLEMIYOR_ve_dosya_sayisi_DOKUZ():
    """Bağlaşım bir kat yukarıda gizlenmiyor — ve spec'in bir sayısı daha
    çürüyor.

    ÖLÇÜLDÜ (Task 1, HEAD = b221a9c): `observable/` içinde `__init__.py`
    hariç **8** dosya vardı (em_obs, grav_obs, hf_obs, md_obs, spin_obs,
    tb_obs, tdse_obs, tdse_obs_2d). Spec §2.1 "yedi gözlenebilir dosyasının
    yedisi" diyor ama kendi parantezinde bu SEKİZ adı tek tek sayıyor —
    yani sayı yanlış, liste doğru.

    TASK 2'DE SAYI DOKUZ OLDU ve BU TEST ONU YAKALADI — istenen davranış tam
    da budur: `bridge_obs` eklendiğinde test KIRMIZI doğdu ve tablo
    güncellenmeye ZORLANDI. Bayat bir beyanın yalana dönüşmesi burada
    yapısal olarak engelleniyor.

    Yük taşıyan iddia AYAKTA KALDI ve yeni dosyada da ölçüldü: dokuzun
    HİÇBİRİ birden çok motor import etmiyor; yedisi yalnız kendi motorunu
    (`bridge_obs` → `md`), `spin_obs` ve `tb_obs` hiçbirini. `bridge_obs`
    ayrıca `observable.grav_obs`u import ediyor ama o bir MOTOR DEĞİL,
    aynı katmanın kardeşidir — bağlaşım motorlar arasında değil.
    """
    dosyalar = sorted(
        p for p in (KAYNAK / "observable").glob("*.py") if p.name != "__init__.py"
    )
    assert len(dosyalar) == 9
    motorlar = {p.name for p in _motor_paketleri()}
    harita: dict[str, set[str]] = {}
    for dosya in dosyalar:
        hedefler = {
            m.split(".")[2]
            for m in _import_edilen(dosya)
            if m.split(".")[:2] == ["qlab", "engine"] and len(m.split(".")) >= 3
        }
        harita[dosya.stem] = hedefler
        assert len(hedefler) <= 1, f"{dosya.name} birden çok motoru import ediyor"
        assert hedefler <= motorlar

    assert harita == {
        "bridge_obs": {"md"},
        "em_obs": {"em"},
        "grav_obs": {"grav"},
        "hf_obs": {"hf"},
        "md_obs": {"md"},
        "spin_obs": set(),
        "tb_obs": set(),
        "tdse_obs": {"tdse"},
        "tdse_obs_2d": {"tdse"},
    }


# --- 7. Depodaki tuzağın kendisi -- ve kapanışı ------------------------------


def test_MD_FORCE_FIELD_KINDS_TUZAGI_KAPANDI_VE_KILITLENDI():
    """Bu görevin taşıyıcı fikrinin depodaki KANITI — ve artık ÇÖZÜMÜ.

    TUZAK NEYDİ (ölçüldü 29-30.08.2026, `scenario/schema.py:56`):
        `MD_FORCE_FIELD_KINDS = ("harmonic", "morse", "lennard_jones")`
    hem ÖLÜydü (adın `src/`+`tests/` ağacındaki tek geçişi bir ATAMAydı;
    hiçbir yerde okunmuyordu) hem BAYATtı (`hartree_fock` listede yoktu,
    oysa `engine/md/factory.py` onu sevk ediyor ve `scenarios/h2_aimd.yaml`
    onu kullanıyor). Yani biri o demeti bir doğrulama kapısı olarak
    bağlasaydı, depo KENDİ senaryosunu reddederdi.

    NASIL KAPANDI (30.08.2026): demete `hartree_fock` eklendi ve demet
    `scenario/loader.py`de GERÇEK bir kapıya bağlandı. Ama asıl iş bu
    değil: elle yazılan bir demeti kapıya bağlamak tuzağı ÇÖZMEZ,
    PAHALILAŞTIRIR — bayat demet artık sessizce yanlış olmakla kalmaz,
    geçerli senaryoları REDDEDER. Bu yüzden demet bir TÜRETME KİLİDİNE
    bağlandı: `tests/md/test_md_scenario.py`deki
    `test_MD_FORCE_FIELD_KINDS_FACTORY_SEVKINDEN_TURETILIYOR` factory'nin
    sevk dallarını `ast` ile tarar ve demeti ona karşı sınar.

    DERS, `bridge_map.py`NİN TAŞIYICI FİKRİNİN TA KENDİSİ: elle yazılmak
    ZORUNDA olan bir yokluk beyanı (burada katman sözleşmesi yüzünden --
    `scenario`, `engine`i import EDEMEZ) silinmek zorunda değildir; KAYNAĞA
    KARŞI SINANMAK zorundadır. Kova tablosu da tam olarak böyle yaşıyor.

    PLANIN BİR SATIR NUMARASI ÇÜRÜDÜ: plan `scenarios/h2_aimd.yaml:13`
    diyor; ölçüldü, `kind: hartree_fock` **10.** satırda, 13. satır
    `basis: sto-3g`. (13 numarası spec §4'ün `basis` maddesinden
    kopyalanmış.)
    """
    schema_metni = (KAYNAK / "scenario" / "schema.py").read_text(encoding="utf-8")

    # (a) ARTIK BAYAT DEĞİL: `hartree_fock` demette.
    assert 'MD_FORCE_FIELD_KINDS = ("harmonic", "morse", "lennard_jones", "hartree_fock")' in schema_metni

    # (b) ARTIK ÖLÜ DEĞİL. Ölçüt yine `ast` ile okunuyor, düz metinle değil:
    # bu dosya ve `bridge_map.py` adı DOKSTRINGDE anıyor, ve düz metin
    # taraması onları "kullanım" sayıp testi yanlış yere yeşile çevirirdi.
    # Gerçek ölçüt adın bir DEĞER olarak okunması (Load) ya da import
    # edilmesidir.
    yazma: list[tuple[str, int]] = []
    okuma: list[tuple[str, int]] = []
    for kok in (KAYNAK, PROJE_KOKU / "tests"):
        for dosya in kok.rglob("*.py"):
            if "__pycache__" in dosya.parts:
                continue
            agac = ast.parse(dosya.read_text(encoding="utf-8"), filename=str(dosya))
            for dugum in ast.walk(agac):
                if (
                    isinstance(dugum, ast.Name)
                    and dugum.id == "MD_FORCE_FIELD_KINDS"
                ):
                    hedef = yazma if isinstance(dugum.ctx, ast.Store) else okuma
                    hedef.append((dosya.name, dugum.lineno))
                elif isinstance(dugum, ast.Attribute) and (
                    dugum.attr == "MD_FORCE_FIELD_KINDS"
                ):
                    okuma.append((dosya.name, dugum.lineno))
                elif isinstance(dugum, ast.ImportFrom) and any(
                    ad.name == "MD_FORCE_FIELD_KINDS" for ad in dugum.names
                ):
                    okuma.append((dosya.name, dugum.lineno))
    assert [ad for ad, _ in yazma] == ["schema.py"]
    okuyan_dosyalar = {ad for ad, _ in okuma}
    assert "loader.py" in okuyan_dosyalar, (
        f"kapıya bağlı değil, demet yine ÖLÜ: {sorted(okuyan_dosyalar)}"
    )

    # (c) KAPI GERÇEKTEN KAPI: `loader.py` demeti bir RED ölçütü olarak
    # kullanıyor, yalnızca import etmiş olmak için değil.
    loader_metni = (KAYNAK / "scenario" / "loader.py").read_text(encoding="utf-8")
    assert "if ff_kind not in MD_FORCE_FIELD_KINDS:" in loader_metni

    # (d) Tuzağın kurbanı olacak senaryo ARTIK GEÇİYOR.
    yaml_satirlari = (
        (PROJE_KOKU / "scenarios" / "h2_aimd.yaml")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert yaml_satirlari[9].strip() == "kind: hartree_fock"
    assert yaml_satirlari[12].strip() == "basis: sto-3g"
    fabrika = (MOTOR_KOKU / "md" / "factory.py").read_text(encoding="utf-8")
    assert '"hartree_fock"' in fabrika
    senaryo = load_scenario(str(PROJE_KOKU / "scenarios" / "h2_aimd.yaml"))
    assert senaryo.force_field["kind"] == "hartree_fock"


# --- 8. Saf tanım katmanı ----------------------------------------------------


def test_MODULLER_KAYNAK_AGACINDAN_DOGRULANIYOR():
    """`MODULLER` de bayatlayabilir: sekizinci bir motor eklenirse 21 çift
    28 olur ve tablo sessizce eksik kalır. Bu yüzden dizin listesine karşı
    sınanıyor."""
    assert MODULLER == tuple(p.name for p in _motor_paketleri())
    assert len(MODULLER) == 7
    assert len(set(MODULLER)) == 7


def test_bridge_map_HICBIR_QLAB_MODULUNU_IMPORT_ETMIYOR():
    """`bridge_map` SAF TANIM KATMANIDIR (`scenario/schema.py`nin kalıbı).

    Hiçbir motoru import etmemesi bir üslup tercihi değil bir KATMAN
    SÖZLEŞMESİDİR: beyanın tek kaynağı, beyan ettiği şeye bağlanamaz.
    Ölçüm import-linter'a bırakılmıyor, burada da bağımsızca yapılıyor —
    `pyproject.toml`daki katman listesine eklenmesi bu testin yerini
    tutmaz (orada bir sözleşme adı unutulursa sessizce geçer).
    """
    dosya = KAYNAK / "bridge_map.py"
    icerik = _import_edilen(dosya)
    qlab_olanlar = {m for m in icerik if m.split(".")[0] == "qlab"}
    assert qlab_olanlar == set(), f"saf tanım katmanı qlab import ediyor: {qlab_olanlar}"
    assert icerik <= {"__future__", "ast", "dataclasses", "pathlib"}


def test_bridge_map_DOKSTRINGLERI_AKSANSIZ_ASCII():
    """Kaynak dokstringleri Türkçe ama AKSANSIZ ASCII (global kısıt).

    Testler tam Türkçedir; kaynak değil. Bu dosyanın kendisi bilinçli
    olarak aksanlıdır ve o da burada ölçülüyor — iki tarafın karışması
    kuralı görünmez kılar.
    """
    kaynak = (KAYNAK / "bridge_map.py").read_text(encoding="utf-8")
    aksanli = [
        (no, s) for no, s in enumerate(kaynak.splitlines(), 1) if not s.isascii()
    ]
    assert aksanli == [], f"bridge_map.py ASCII dışı satır taşıyor: {aksanli}"

    bu_dosya = Path(__file__).read_text(encoding="utf-8")
    assert not bu_dosya.isascii(), "test dosyası tam Türkçe olmalı"


def test_bridge_map_GRAV_BEYANININ_PARCASINI_KOPYALAMIYOR():
    """MODÜL 6'NIN PARÇA-KOPYA YASAĞI BU DOSYAYI DA BAĞLAR — VE BİR KEZ
    GERÇEKTEN ISIRDI (ölçüldü, 30.08.2026).

    `tests/grav/test_grav_senaryo.py::test_BEYAN_PARCASININ_KOPYASI_da_YASAK`
    16 karakter eşiğiyle şunu yasaklar: `src/` ağacında (grav fabrikası
    hariç) hiçbir dize sabiti bir grav beyan cümlesinin İÇİNDE geçemez —
    çünkü korunan şey "aynı cümle iki kez yazılmasın" değil, PANELİN İKİ
    FARKLI ŞEY İDDİA EDEMEMESİdir.

    `bridge_map.py`nin ilk sürümündeki `YENIDEN_URETILMEDI` damgası
    "BU DEPODA YENIDEN URETILMEDI" idi ve grav beyanının Kozai cümlesinin
    tam bir parçasıydı; tam süit bu yüzden KIRMIZI oldu (1 failed / 1719
    passed). Damga "BU GOREVDE YENIDEN OLCULMEDI" olarak değiştirildi.

    Bu test aynı yasağı BURADA, saniyeler içinde koşan bir süitte tekrar
    kurar: aynı hata bir daha 16 dakikalık tam süite kadar saklanamasın.
    """
    fabrika = (MOTOR_KOKU / "grav" / "factory.py").read_text(encoding="utf-8")
    agac = ast.parse(
        (KAYNAK / "bridge_map.py").read_text(encoding="utf-8"),
        filename="bridge_map.py",
    )
    kopyalar = [
        (d.lineno, d.value[:60])
        for d in ast.walk(agac)
        if isinstance(d, ast.Constant)
        and isinstance(d.value, str)
        and len(d.value) >= 16
        and d.value in fabrika
    ]
    assert kopyalar == [], f"grav beyanının parçası kopyalanmış: {kopyalar}"
    # Kontrol: eşiğin altındaki kısa parçalar zaten serbest, yani bu test
    # boş bir kural değil — ölçülmüş eşiği (16) aynen devralıyor.
    assert "BU GOREVDE YENIDEN OLCULMEDI" not in fabrika
    assert "BU DEPODA YENIDEN URETILMEDI" in fabrika


def test_EdgeFact_DONDURULMUS_ve_cift_YONDEN_BAGIMSIZ():
    """`cift` yönden bağımsız kimliktir: `hf → md` ile `md → hf` AYNI çift.

    Bu, tablo ile taramanın karşılaştırılabilmesinin ön koşulu — tarama
    `(md, hf)` yönünü görür (import eden md'dir), tablo `hf → md` yazar
    (parametreyi üreten hf'tir)."""
    f = EdgeFact("hf", "md", Kova.GERCEK_KOPRU, "gerekçe", None)
    assert f.cift == ("hf", "md")
    assert EdgeFact("md", "hf", Kova.GERCEK_KOPRU, "x", None).cift == f.cift
    with pytest.raises(Exception):
        f.kaynak = "tb"  # type: ignore[misc]
