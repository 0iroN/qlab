"""Moduller arasi YOKLUK HARITASI -- kova tablosunun TEK KAYNAGI (Modul 7).

Bu modul hicbir motoru import etmez: saf tanim katmanidir
(`scenario/schema.py`nin kalibi). Import edilen HER SEY standart
kutuphanedir ve tam listesi sudur: `__future__`, `ast`, `dataclasses`,
`pathlib`. Bir tek `qlab.*` import'u bile yoktur; bu hem bir testle hem de
`pyproject.toml`daki katman sozlesmesiyle kilitlidir.

BU MODULUN TASIYICI FIKRI: KOVA TABLOSU KODDAN TURETILIR, ELLE
LISTELENMEZ. Bir yokluk beyani BAYATLARSA BIR YALANDIR. "Capraz-motor
import sayisi 2'dir" iddiasi elle yazilirsa ilk degisiklikte sessizce
yanlis olur; burada `olculen_kenarlar()` kaynak agacini TARAR ve tablo ona
karsi sinanir. (Ornek varsayimsal kalmadi: 30.08.2026'da sayi 2'ydi,
22.09.2026'da 3, 24.09.2026'da 4 oldu -- elle yazilmis olsaydi iki kez
bayatlardi.)

BU TUZAGIN BIR ORNEGI DEPODA OLCULDU VE KAPATILDI (30.08.2026):
`scenario/schema.py`nin `MD_FORCE_FIELD_KINDS` demeti hem OLUydu (`src/` ve
`tests/` icinde YALNIZ tanimi bulunuyordu, hicbir yerde okunmuyordu) hem
BAYATti (`hartree_fock` listede YOKtu, oysa `scenarios/h2_aimd.yaml:10` tam
olarak onu kullaniyor ve `engine/md/factory.py`nin `hartree_fock` dali --
30.08.2026'da `:126`; satir numarasi o gunden beri kaydi -- onu sevk
ediyor). Biri o
demeti bir dogrulama kapisi olarak baglasaydi, depo KENDI senaryosunu
reddederdi.

KAPATMA BICIMI BU MODULUN FIKRINI DOGRULUYOR: demet SILINMEDI. Katman
sozlesmesi `scenario`nun `engine`i import etmesini yasakladigi icin elle
yazilmak ZORUNDA; o yuzden `loader.py`de gercek bir kapiya baglandi VE
factory'nin sevk dallarini `ast` ile tarayan bir TURETME KILIDINE
(`tests/md/test_md_scenario.py`) bagli kilindi. Elle yazilan bir yokluk
beyani silinmek zorunda degildir; KAYNAGA KARSI SINANMAK zorundadir.
Kova tablosu da tam olarak boyle yasiyor. Ayni desen burada
TEKRARLANMAYACAK.

TARAMA `ast` ILE YAPILIR, REGEX ILE DEGIL. Yorum satirindaki ya da dizge
icindeki bir import METNI tabloyu kirletir. Bunun bedeli bu depoda
OLCULDU ve bu planin bir sayisini curuttu: `engine/md/__init__.py`
dokstringi (satir numarasi BILEREK yazilmaz: 24.09.2026'da dokstringe
satir eklenince eski `:10` sessizce `:13`e kaydi)
`qlab.units.amu_to_electron_mass` ADINI yaziyor ama `units`i
IMPORT ETMIYOR. Duz metin taramasi bu yuzden md icin 4 dosya sayar
(planin/spec'in sayisi), `ast` taramasi 3 sayar. Yuk tasiyan iddia --
`units`e dokunan kumenin `{hf, md, tb}` olmasi -- iki yontemde de ayni
ciktigi icin AYAKTA KALIR; curuyen sey MODUL BASINA SAYIDIR.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

#: Yedi fizik motoru. Sira `src/qlab/engine/` altindaki paket dizinlerinin
#: alfabetik siralamasidir ve bir testle o dizin listesine karsi
#: dogrulanir -- elle tutulan bir sira degil.
MODULLER: tuple[str, ...] = ("em", "grav", "hf", "md", "spin", "tb", "tdse")


class Kova:
    """Bes kova. `enum.Enum` DEGIL, cunku bu depoda `enum` modulu hic
    kullanilmiyor -- olculdu 30.08.2026: bu dosya disinda `src/` agacinda
    "enum" gecen 13 satirin 13'u de `enumerate`, `import enum` /
    `from enum import` sayisi SIFIR. Ve `scenario/schema.py`nin kalibi duz
    dizge sabitleridir (`POTENTIAL_KINDS`, `MD_THERMOSTAT_KINDS`; satir
    numarasi BILEREK yazilmaz -- 24.09.2026'da ikincisi kaymisti).

    OLCULMEDI AYRI BIR KOVADIR -- spec 7.2.
    "OLCULDU, SIFIR CIKTI" ile "HIC OLCULMEDI" AYNI HUCREYE YAZILAMAZ.
    Onu YAPISAL_YOK'a katlamak hukum verilmemis bir seyi verilmis gibi gosterirdi;
    kova TARIHCESI (hangi cift ne zaman, neden gecti) TEK YERDE: `KOVA_SAYILARI` yorumu.
    """

    GERCEK_KOPRU = "GERCEK_KOPRU"
    TAUTOLOJI = "TAUTOLOJI"
    YAPISAL_YOK = "YAPISAL_YOK"
    BENZERLIK = "BENZERLIK"
    OLCULMEDI = "OLCULMEDI"


#: Gecerli kova adlarinin tamami. Bir yazim hatasi (`TAUTOLOJII`) sessizce
#: yeni bir kova yaratamasin diye tablo buna karsi sinanir.
KOVALAR: tuple[str, ...] = (
    Kova.GERCEK_KOPRU,
    Kova.TAUTOLOJI,
    Kova.YAPISAL_YOK,
    Kova.BENZERLIK,
    Kova.OLCULMEDI,
)

#: OLCULMEDI kovasinin her girdisinin gerekcesinde ARANAN dizge. Bu bir
#: susleme degil bir KILIT: bir girdi bu kovaya konup da "olculdu" diye
#: yazilamaz, ve baska hicbir kovanin girdisi bu dizgeyi tasiyamaz.
#:
#: 22.09.2026 NOTU -- DIZGE AYNI KALDI, ANLAMI TARIHE CIPALANDI. Bu kilit
#: OLCULMEDI icin TEK bir mesru gerekce taniyacak sekilde yazilmisti; spec
#: 7.2'nin on olcumu ikinci bir tane uretti ("sayi geldi, HUKUM verilemedi").
#: Dizgeyi silmek ya da gevsetmek yerine, gerekce metni onu 29.08.2026
#: hakkinda bir TARIHSEL cumle olarak tasir ve yeni durumu ayrica yazar --
#: boylece kilit ayakta kalir ve hicbir cumle bayatlamaz.
HICBIR_SAYI_GELMEDI = "HICBIR SAYI GELMEDI"

#: Bir hukmu 29.08.2026 on olcumunden DEVRALIP burada yeniden olcmeyen
#: girdilerin tasidigi damga. `engine/grav/factory.py`nin alinti damgasiyla
#: AYNI ISI gorur ama ONUN SOZCUKLERIYLE YAZILAMAZ: Modul 6'nin parca-kopya
#: yasagi (`tests/grav/test_grav_senaryo.py`, esik 16 karakter) src/ agacinda
#: bir beyan cumlesinin ICINDE gecen her dize sabitini reddeder. OLCULDU
#: 30.08.2026: ilk yazilan "BU DEPODA YENIDEN URETILMEDI" damgasi grav
#: beyaninin bir PARCASIYDI ve tam suiti KIRMIZIYA cevirdi; bu damga
#: `factory.py` metninde gecmiyor.
YENIDEN_URETILMEDI = "BU GOREVDE YENIDEN OLCULMEDI"


@dataclass(frozen=True)
class EdgeFact:
    """Bir modul ciftinin hukmu.

    `kaynak`/`hedef` KAVRAMSAL yondur (parametreyi ureten -> tuketen);
    ciftin kimligi yonden bagimsizdir ve `cift` ozelliginden okunur.

    `buyukluk` YALNIZCA bu depoda olculmus ya da bu depodaki bir sevk
    edilmis beyandan birebir dogrulanmis bir sayi tasir. Olculmemisse
    `None`'dir ve `gerekce` bunun NEDEN `None` oldugunu yazar. Bir sayiyi
    "elde var" gibi gostermek, bu depoda yapilabilecek en agir hatadir.
    """

    kaynak: str
    hedef: str
    kova: str
    gerekce: str
    buyukluk: float | None = None

    @property
    def cift(self) -> tuple[str, str]:
        """Yonden bagimsiz cift kimligi (alfabetik siralanmis ikili)."""
        a, b = sorted((self.kaynak, self.hedef))
        return (a, b)


# --- Kaynak agaci taramasi ---------------------------------------------------

_MOTOR_KOKU = Path(__file__).resolve().parent / "engine"

#: KENAR SINIFLARI (26.09.2026). Bir kod kenari ya CALISMA ZAMANI ya YALNIZ
#: TIP kenaridir; kural `olculen_kenar_siniflari()` dokstringindedir.
KENAR_CALISMA_ZAMANI = "calisma_zamani"
KENAR_YALNIZ_TIP = "yalniz_tip"

#: MOTOR AGACINDA YASAK BICIMLER (26.09.2026) -- bir ANALIZ degil bir
#: POLITIKA. Tarayici yalniz DUZ import bicimlerini cozer; ADI SAYILAN
#: yukleme ve erisim bicimleri (`_YASAK_*`, ciplak `qlab` adi, zincirsiz
#: `qlab.engine`) motor agacinda YASAKTIR ve `taranamayanlar()` onlari
#: adiyla raporlar (fail-closed: `tablo_tutarsizliklari` tiki dusurur).
#: Liste SONLUDUR; bilinen artik yollar `taranamayanlar()` dokstringinde.
#: Gercek agacta hepsinin sayisi 0'dir (olculdu). Yeni bir yukleme bicimi
#: gerekirse once tarayici onu cozecek sekilde genisletilir, sonra yasak
#: kalkar. `getattr` SERBESTTIR: modul yukleyemez ve motor agacinda mesru
#: kullanimlari var (`qlab`/`qlab.engine` uzerinde ise ad kuralina takilir).
_YASAK_MODULLER = frozenset(
    {"importlib", "runpy", "pkgutil", "pydoc", "imp", "zipimport", "builtins", "sys"}
)
_YASAK_ADLAR = frozenset(
    {"__import__", "exec", "eval", "compile", "vars", "globals", "locals",
     "__builtins__", "TYPE_CHECKING"}
)
#: `from M import ad` biciminde yasak (M, ad) ciftleri; M COZULMUS addir,
#: yani `from .. import *` de `from qlab.engine import *` sayilir.
_YASAK_CIFTLER = frozenset(
    {("qlab", "engine"), ("qlab", "*"), ("qlab.engine", "*")}
)


@dataclass(frozen=True)
class _Tarama:
    """Motor agacinin TEK bir taramasinin sonucu (ic yapi)."""

    kenarlar: frozenset[tuple[str, str]]
    siniflar: dict[tuple[str, str], str]
    eksen: dict[str, int]
    taranamayan: tuple[str, ...]


def _motor_paketleri(kok: Path) -> list[Path]:
    return sorted(
        p for p in kok.iterdir() if p.is_dir() and not p.name.startswith("__")
    )


def _tip_dugumleri(agac: ast.AST) -> set[int]:
    """Annotasyon baglamindaki butun dugumlerin kimlikleri: arguman
    annotasyonu, fonksiyon donus annotasyonu, `AnnAssign` annotasyonu."""
    kimlikler: set[int] = set()
    for d in ast.walk(agac):
        ann = None
        if isinstance(d, ast.arg):
            ann = d.annotation
        elif isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ann = d.returns
        elif isinstance(d, ast.AnnAssign):
            ann = d.annotation
        if ann is not None:
            kimlikler.update(id(n) for n in ast.walk(ann))
    return kimlikler


def _agaci_tara(kok: Path | str | None) -> _Tarama:
    """Motor agacini BIR KEZ tarar: kenarlar, siniflar, eksen, yasaklar.

    Dosyalar `rglob` ile toplanir (alt paketler DAHIL) ve her dosya UST
    DUZEY motor paketine atfedilir. `ast` ile ayristirilir: yorum ve dizge
    icindeki import METNI bir dugum uretmez, sayilmaz. Cozulen bicimler:
    `import qlab.engine.X...`, `from qlab.engine.X... import ...`,
    `from qlab.engine import X` (adlar alt modul/paket olarak cozulur),
    goreli importlar (taban dosyanin `__package__`idir; alt pakette de
    dogru) ve `qlab.engine.X` oznitelik zinciri (`import qlab.units` de
    `qlab` adini baglar; bu yuzden `qlab` adi YALNIZ `qlab.engine.X` ya da
    `qlab.units` zincirinin basi olabilir). ADI SAYILAN yukleme bicimleri
    YASAKTIR (liste sonlu, `taranamayanlar()`). `engine/` KOKUNDEKI
    dosyalar (`__init__.py` dahil) yalniz yasaklar ve (iii) disari-cikis
    icin taranir; kenar ve eksen uretmez.
    """
    taban = Path(kok) if kok is not None else _MOTOR_KOKU
    paketler = _motor_paketleri(taban)
    eksen: dict[str, int] = {p.name: 0 for p in paketler}
    yasak: list[tuple[str, int, str]] = []
    site_siniflari: dict[tuple[str, str], set[str]] = {}
    adaylar: list[tuple[tuple[str, str], str, str]] = []
    alinan: set[tuple[str, str]] = set()
    anilan: set[str] = set()
    dosyalar = [(None, d) for d in sorted(taban.glob("*.py"))]  # kok: motor YOK
    dosyalar += [(p.name, d) for p in paketler for d in sorted(p.rglob("*.py"))]
    for motor, dosya in dosyalar:
        yer = dosya.relative_to(taban).as_posix()
        ic = list(dosya.relative_to(taban).with_suffix("").parts)
        paket_adi = ["qlab", "engine", *ic[:-1]]
        modul_adi = ".".join(paket_adi if ic[-1] == "__init__" else ["qlab", "engine", *ic])
        try:
            agac = ast.parse(dosya.read_bytes(), filename=str(dosya))
        except (SyntaxError, ValueError):
            yasak.append((yer, 0, "ayristirilamadi"))
            continue
        gelecek = any(
            isinstance(d, ast.ImportFrom) and d.module == "__future__"
            and any(a.name == "annotations" for a in d.names)
            for d in agac.body
        )
        tipte_id = _tip_dugumleri(agac)
        calisan: set[str] = set()
        tipte: set[str] = set()
        # (tam modul adi, baglanan ad) -- ad None ise site CALISMA ZAMANI.
        hedefler: list[tuple[str, str | None]] = []
        units_var = False
        # id(Attribute.value) -> attr. `ast.walk` genislik-once yurur: bir
        # oznitelik dugumu, degeri olan cocuktan ONCE ziyaret edilir.
        zincir: dict[int, str] = {}
        for d in ast.walk(agac):
            if isinstance(d, ast.Name):
                if d.id in _YASAK_ADLAR:
                    yasak.append((yer, d.lineno, f"`{d.id}`"))
                if d.id == "qlab" and zincir.get(id(d)) not in ("engine", "units"):
                    yasak.append((yer, d.lineno, "ciplak `qlab` adi"))
                if not isinstance(d.ctx, ast.Store):  # Load ve Del adi OKUR
                    (tipte if id(d) in tipte_id else calisan).add(d.id)
            elif isinstance(d, ast.AugAssign) and isinstance(d.target, ast.Name):
                calisan.add(d.target.id)  # `ad += ...` de adi okur
            elif isinstance(d, ast.Attribute):
                zincir[id(d.value)] = d.attr
                anilan.add(d.attr)
                if d.attr == "TYPE_CHECKING":
                    yasak.append((yer, d.lineno, "`TYPE_CHECKING`"))
                u = d.value
                if isinstance(u, ast.Name) and u.id == "qlab" and d.attr == "units":
                    units_var = True
                if d.attr == "engine" and isinstance(u, ast.Name) and u.id == "qlab" and id(d) not in zincir:
                    yasak.append((yer, d.lineno, "zincirsiz `qlab.engine`"))
                if (isinstance(u, ast.Attribute) and u.attr == "engine"
                        and isinstance(u.value, ast.Name) and u.value.id == "qlab"):
                    hedefler.append((f"qlab.engine.{d.attr}", None))
            elif isinstance(d, ast.Constant) and isinstance(d.value, str):
                anilan.add(d.value)
            elif isinstance(d, ast.Import):
                for a in d.names:
                    if a.name.split(".")[0] in _YASAK_MODULLER or a.name in ("qlab", "qlab.engine"):
                        yasak.append((yer, d.lineno, f"`import {a.name}`"))
                    hedefler.append((a.name, None))
            elif isinstance(d, ast.ImportFrom):
                if d.level:
                    kes = d.level - 1
                    if kes >= len(paket_adi):
                        yasak.append((yer, d.lineno, "goreli import ust paketin disina tasiyor"))
                        continue
                    m = ".".join(paket_adi[: len(paket_adi) - kes] + ([d.module] if d.module else []))
                else:
                    m = d.module or ""
                if m.split(".")[0] in _YASAK_MODULLER:
                    yasak.append((yer, d.lineno, f"`from {m} import`"))
                for a in d.names:
                    alinan.add((m, a.name))
                    if (m, a.name) in _YASAK_CIFTLER or a.name == "TYPE_CHECKING":
                        yasak.append((yer, d.lineno, f"`from {m} import {a.name}`"))
                    if a.name == "*":
                        hedefler.append((m, None))
                    else:
                        hedefler.append((f"{m}.{a.name}", a.asname or a.name))
        if motor is None:
            continue
        for tam, ad in hedefler:
            p = tam.split(".")
            units_var = units_var or p[:2] == ["qlab", "units"]
            if p[:2] != ["qlab", "engine"] or len(p) < 3 or p[2] == motor:
                continue
            kenar = (motor, p[2])
            site_siniflari.setdefault(kenar, set())
            if ad is not None and gelecek and ad in tipte and ad not in calisan:
                adaylar.append((kenar, modul_adi, ad))
            else:
                site_siniflari[kenar].add(KENAR_CALISMA_ZAMANI)
        eksen[motor] += int(units_var)
    # YALNIZ TIP ADI DOSYADAN CIKMAMALI: motor agacinda baska bir dosya onu o
    # modulden adiyla / yildizla aliyorsa, ya ad bir oznitelik ya da tam bir
    # dizge sabiti olarak geciyorsa (`getattr(M, "ad")`, `__all__`) site
    # CALISMA ZAMANI sayilir -- muhafazakar yon.
    for kenar, modul, ad in adaylar:
        disari = (modul, ad) in alinan or (modul, "*") in alinan or ad in anilan
        site_siniflari[kenar].add(KENAR_CALISMA_ZAMANI if disari else KENAR_YALNIZ_TIP)
    siniflar = {
        k: KENAR_CALISMA_ZAMANI if KENAR_CALISMA_ZAMANI in s else KENAR_YALNIZ_TIP
        for k, s in site_siniflari.items()
    }
    return _Tarama(
        kenarlar=frozenset(siniflar),
        siniflar=siniflar,
        eksen=eksen,
        taranamayan=tuple(f"{y}:{n} {b}" for y, n, b in sorted(yasak)),
    )


def olculen_kenarlar(kok: Path | str | None = None) -> frozenset[tuple[str, str]]:
    """`src/qlab/engine/*/` altindaki CAPRAZ-MOTOR import kenarlarini olcer.

    Her motor paketinin `.py` dosyalari `ast` ile ayristirilir,
    `qlab.engine.<X>` bicimindeki import'lar cikarilir ve `<X>` dosyanin
    kendi paketinden FARKLIYSA `(paket, X)` kenari uretilir.

    `kok` yalnizca sinama icindir (sentetik bir agac verilebilir);
    varsayilan bu paketin yanindaki gercek `engine/` dizinidir.

    OLCULDU (30.08.2026, b221a9c): tam IKI kenar --
    `md -> hf` (`engine/md/forcefields.py:13`) ve
    `tb -> hf` (`engine/tb/downfold.py:130-132`). O gun 21 ciftin 19'unda
    SIFIR bag vardi ve iki kenarin ikisi de ayni dugume (hf) gittigi icin
    yapi merkezi HF olan IKI CUBUKLU BIR YILDIZDI.

    YENIDEN OLCULDU (22.09.2026): UCUNCU KENAR YAZILDI --
    `tdse -> md` (`engine/tdse/from_md.py`). Dereceler artik
    hf 2, md 2, tb 1, tdse 1; yani yapi bir YILDIZ DEGIL, DORT DUGUMLU BIR
    ZINCIR: `tb - hf - md - tdse`. 21 ciftin 18'inde sifir bag.

    VE BU, MERDIVEN SORUSUNU YENIDEN ACAR GIBI GORUNUR -- ACMAZ, AMA
    GEREKCE DARALIR. Bir zincir topolojik olarak bir yildizdan daha cok
    merdivene benzer. Yine de bu bir OLCEK MERDIVENI DEGILDIR, cunku
    zincirin ucu OLCEKSIZDIR: `olculen_units_ekseni()["tdse"] == 0`, yani
    TDSE mutlak olcek eksenine hic dokunmuyor (bkz.
    `OLCEKSIZ_KENAR_DUGUMLERI`). Merdiven hukmu ayakta; DAYANAGI degisti --
    artik "yapi yildiz" degil "bagli olmak mutlak eksene dokunmayi
    GEREKTIRMIYOR". (Yukaridaki iki paragraf 22.09.2026'nin TARIHSEL
    kaydidir.)

    YENIDEN OLCULDU (24.09.2026): DORDUNCU KENAR YAZILDI -- `md -> tb`
    (`engine/md/tb_kuvvet.py` -> `qlab.engine.tb.{mermin, radyal}` ve
    `engine/md/factory.py` -> `qlab.engine.tb.radyal`). Dereceler hf 2,
    md 3, tb 2, tdse 1; yapi artik bir zincir de DEGIL: `hf - md - tb`
    UCGENI arti `md - tdse` KUYRUGU. 21 ciftin 17'sinde sifir bag.
    UCGEN IMPORT GRAFIGINDE BIR DONGUDUR, VERI AKISINDA DEGIL: `tb -> hf`
    kenarinda veri akmaz (`model_from_downfold` src'de 0 cagri), yeni kenar
    SECILMIS bir yasa tasir ve downfold'dan gelen yasa yasaktir. Uc dugum de
    ayni mutlak eksende (hartree/bohr); tek olcekteki bir ucgen bir OLCEK
    merdiveni degildir ve merdiven hukmu ayakta kalir.

    TARAMA GENISLEDI (26.09.2026, yalin surum): alt paketler, `from
    qlab.engine import X`, capraz goreli import ve `qlab.engine.X` zinciri
    artik cozulur; ADI SAYILAN yukleme bicimleri motor agacinda YASAKTIR
    (liste sonlu) ve `taranamayanlar()` onlari adiyla raporlar. Gercek
    agacta sonuc AYNI: ayni dort kenar, taranamayan 0 (olculdu; gercek
    agacta 648f03b tarayicisiyla karsilastirma
    `tests/bridge/test_kenar_tarayici.py::test_c_GERCEK_AGAC_648f03b_ile_AYNI`).
    """
    return _agaci_tara(kok).kenarlar


def olculen_kenar_siniflari(
    kok: Path | str | None = None,
) -> dict[tuple[str, str], str]:
    """Her kod kenarinin SINIFI: `calisma_zamani` ya da `yalniz_tip`.

    KURAL (ad duzeyinde): capraz-motor bir `from ... import ad` sitesi
    YALNIZ TIPtir ancak (i) dosyada `from __future__ import annotations`
    varsa (yoksa annotasyon 3.12'de tanim aninda degerlendirilir), (ii) ad en
    az bir kez ve YALNIZ annotasyon baglaminda (arguman annotasyonu, donus
    annotasyonu, `AnnAssign` annotasyonu) yukleniyorsa (`del ad` ve
    `ad += ...` de calisma zamani kullanimidir) ve (iii) ad dosyadan
    CIKMIYORSA (motor agacinda o modulden adiyla ya da yildizla alinmiyor,
    oznitelik ya da tam dizge sabiti olarak gecmiyor). Modul importu
    (`import qlab.engine.x as m`), yildiz ve `qlab.engine.X` zinciri her
    zaman CALISMA ZAMANIdir. Kenar, sitelerinden BIRI calisma zamaniysa
    calisma zamanidir.

    OLCULDU (26.09.2026): `md -> hf`, `md -> tb`, `tb -> hf` CALISMA ZAMANI;
    `tdse -> md` YALNIZ TIP (`ForceField` from_md.py'de yalniz bir arguman
    annotasyonu; import satiri silinince modul bit bit ayni calisiyor --
    `tests/bridge/test_kenar_tarayici.py` yuk tasima cipasi).

    HUKUM KOSULLUDUR: annotasyonu DEGERLENDIREN bir tuketici
    (`get_type_hints`, `get_annotations`, `eval_str`, `singledispatch` ve
    benzerleri; tam liste `tests/bridge/test_kenar_tarayici.py` TUKETICILER)
    yalniz tip bir adi calisma zamanina tasir; `src/qlab` capinda sayilari 0
    ve bu bir testle kilitlidir. Sinifin anlami `GERCEK_KOPRU_KENAR_SINIFLARI`
    yorumunda (tek yonludur).
    """
    return dict(_agaci_tara(kok).siniflar)


def taranamayanlar(kok: Path | str | None = None) -> tuple[str, ...]:
    """Motor agacindaki YASAK bicimlerin her gecisi, `dosya:satir bicim`.

    Yasaklar (`_YASAK_*` ve iki ad kurali): yukleyici modul importu
    (importlib, runpy, pkgutil, pydoc, imp, zipimport, builtins, sys);
    `__import__`, `exec`, `eval`, `compile`, `vars`, `globals`, `locals`,
    `__builtins__` adlari; `TYPE_CHECKING`; ciplak `import qlab` /
    `import qlab.engine`, `from qlab import engine|*`, `from qlab.engine
    import *`; `qlab.engine.X` / `qlab.units` zincirinin basi OLMAYAN `qlab`
    adi (takma ad, `getattr(qlab, ...)`) ve zincirsiz `qlab.engine`
    (`E = qlab.engine`, `getattr(qlab.engine, 'tb')`); ust paketin disina
    tasan goreli import; ayristirilamayan dosya. `engine/` kokundeki
    dosyalar (`__init__.py` dahil) yalniz yasaklar icin taranir, kenar
    uretmez. Bos demet OLCULMUS bir sifirdir (26.09.2026: gercek agacta 0).

    LISTE SONLUDUR -- bu bir analiz degil bir yasaklama politikasidir.
    Bilinen ARTIK YOLLAR burada gorunmez: izinli bir kenar uzerinden yeniden
    aktarim (AKLAMA: tdse, hf'yi md modulunun ozniteligiyle ya da md'nin
    disa aktardigi bir adla kullanir -- `getattr` dizgesi literal de olsa
    hesaplanmis da olsa -- ve `tdse -> hf` kenari cikmaz); adi sayilmamis
    cozucu/yukleyiciler (`os.sys.modules`, `unittest.mock.patch` dizgesi);
    `.pth`, C uzantisi, `.pyw`.
    """
    return _agaci_tara(kok).taranamayan


# --- Kova tablosu ------------------------------------------------------------

_GRAV_ORANI = 8.093550856608016e-37
_HF_TDSE_NOKTA = 4.096e09

# --- Dusen adaylarin OLCULMUS buyuklukleri (Task 4: iki; 27.09: hf -> spin; 28.09: tdse -> spin) ---
# Bu sayilar bir motor kosturularak uretildi ama BURAYA LITERAL yazildi:
# bu modul hicbir motoru import etmez. Testler (`tests/bridge/
# test_dusen_adaylar.py`) motoru kosturup bu literalleri canli olcume karsi
# sinar. Pencereleri asagidaki gerekcelerde yazilidir; pencere olmadan
# tasinamazlar.

#: `hf -> tb` ozdesliginin BAGIL artigi, REFERANS PENCEREDE:
#: N = 6, d = 2.6 bohr, STO-3G, seeds = 12, TAM menzil (n_neighbour = 3).
#: max|model - H_eff^alfa| = 0.010670502034857853; en buyuk CIFT kabugun
#: yayiliminin yarisi = 0.010670502034857829; bagil fark 2.276006188306662e-15,
#: yani ON BES anlamli hane -- spec'in "on alti hane"si DARALTILDI. Bu sayi
#: SISTEME BAGLIDIR (N = 8 d = 2.0'da 1.0898620259967446e-08, yedi mertebe
#: kotu); gerekcede alti sistemin altisi da yazilidir.
_HF_TB_OZDESLIK_BAGIL = 2.276006188306662e-15

#: `tb -> spin` koprusunun CIKTIYI NE KADAR KIPIRDATTIGI: bes farkli `J`de
#: miknatislanma serisinin max|farki| = 0.0, BIT BIT. Sifir bir OLCUM
#: SONUCUDUR, "olculmedi" DEGIL -- OLCULMEDI kovasi `None` tasir, bu girdi
#: `0.0` tasir ve ikisi ayni sey degildir.
_TB_SPIN_M_FARKI = 0.0

#: `hf -> spin` (27.09.2026): uc HF `J`sinin (H2, R = 5/6/7 bohr) protokol
#: A'da `m` serisini `J = -1`inkinden ne kadar ayirdigi: 0.0, BIT BIT --
#: bir OLCUM sonucu, `tests/bridge/test_dusen_adaylar.py` canli sinar.
_HF_SPIN_M_FARKI = 0.0

_TDSE_SPIN_M_FARKI = 0.0  #: `tdse -> spin` (28.09.2026): dort TDSE `J`si, ayni test ve pencere.

#: SPIN HEDEFININ ORTAK KOK NEDENI (28.09.2026; 22.09 P8: ayni olgu uc ayri
#: gerekcede uc bagimsiz olgu gibi gorunmesin). `tb/hf/tdse -> spin` bunu okur.
_SPIN_BUYUKLUGU_SILER = (
    "KOK NEDEN `engine/spin/metropolis.py:189-191` (`ast` ile kilitli): `coupling` yalniz "
    "`delta`yi carpiyor ve kabul olcutune `delta / temperature` olarak giriyor, yani ciktiya "
    "SADECE `coupling/temperature` ORANI olarak giriyor; `h = 0`'da baska enerji olcegi YOK -- "
    "spin hedefi, kaynak hangi motor olursa olsun, J'nin BUYUKLUGUNU SILER. Bit bit esitlik bir "
    "OLCUM sonucudur, ozdeslik degil (float'ta esik 1 ulp oynayabilir; olcumu her gerekcede). "
)

_HF_SPIN_GEREKCE = (
    "SIFIR kod kenari, ama HUKUM VAR (27.09.2026 on olcumu, 10 ajan; "
    "`.superpowers/sdd/hf-spin-on-olcum-2026-09-27.md`). 29.08.2026'dan beri "
    "OLCULMEDI'deydi, 22.09'dan beri sebep: sayi geldi, hukum yok; kenar "
    "YAZILMADI, hukum verildi. "
    "KAYNAK: H2 Yamaguchi J = (E_BS - E_T)/(S2_T - S2_BS), public `search_uhf` "
    "ve testte elle yazilmis formul (`src/`de Yamaguchi YOK). PENCERE: STO-3G, "
    "`search_uhf(seeds = 12)` varsayilan esikler, `Molecule.from_string(\"H 0 0 "
    "0; H 0 0 %.10g\", unit=\"bohr\")`; J(R = 5/6/7) = -0.001308824217584225 / "
    "-0.00017495439344641936 / -1.9411358087445856e-05 Ha. 17 hane bu pencerenin "
    "kaydidir, fiziksel olarak ~7 anlamli hane (bu uc J'de esik oynatilinca "
    "bagil <= 1.89e-9, DIIS kapali 2.73e-9, tohum yayilimi <= 4.95e-9; eksenin "
    "en kotusu 7.35e-8, 6-31G R = 3). " + _SPIN_BUYUKLUGU_SILER +
    "BUYUKLUK SADELESIYOR: protokol A (T = 2.269185 * |J|, Ising L = 32, h = 0, "
    "tohum 20260822, 2000 sweep burn-in DAHIL olculur, Neel baslangic) uc J'de `m` serisi "
    "J = -1'inkiyle bit bit ayni ve E == |J| * E(-1) bit bit (bu pencerede 3/3, "
    "testte canli; on olcumun 1080/1080'i AYRI pencereden: L = 16, tekbicim "
    "baslangic, tohum 11/12/13, 200 + 1000 sweep, c = 1.5 / Tc / 3.5, "
    "[1e-9, 1e9]'de 60 rastgele K, E bit bit sinanmadi); "
    "protokol B'de (300 K) HF basamagi tb basamagiyla ve HF icerigi sifir "
    "sentetik bir merdivenle yorunge duzeyinde bit bit ayni (on olcum, sha256 "
    "6/6; testte kosturulmaz). OLCUMUN KANITI "
    "(R = 6/7'de karar esigi 1 birim farkli, elestirmen E1c). ISARET "
    "BIR AYAR ETIKETIDIR: h = 0 ve cift L'de (J, s0) ile (-J, eta * s0) "
    "kosumlari her sweep'te s(-J) == eta * s(J), eta = (-1)^(i+j) -- AFM(-J) "
    "kosumunun kademeli serisi FM(+J) kosumunun tekbicim serisiyle bit bit ayni, "
    "SWEEP 1'DEN (yalniz burn-in sonrasina bakmak yetmez: ortak rastgele "
    "sayilarla zincirler birlesir, donusturulmemis baslangic da ayni seriyi verir). "
    "TAUTOLOJI 'cikti kipirdamiyor' diye okunursa isaret onu zorlar; bu yuzden: "
    "tekbicim cikti isaretle degisir ama degisim FM kosumunun bit bit "
    "ayar-imgesidir (AFM tekbicim ortalamasi 0.009724, kademeli 0.687930). "
    "Motorun ciktisi 'FM kosumu + girdinin tek biti'dir ve o bit girdiden geri "
    "okunur. ISARET SARTLI: izlemesiz `search_uhf(seeds = 1, "
    "follow_instability = False)` ve tek `solve_uhf` kirik simetriyi BULAMAZ "
    "(S2_BS sayisal sifir, 2.2e-16 / 2.2e-16 / 1.1e-15 -- '> 0' kapisini "
    "GECER) ve formul sessizce "
    "+J verir (R = 5/6/7: +0.1229 / +0.1440 / +0.1552, FM); izlemeli (varsayilan) "
    "`search_uhf(seeds = 1)` ise B7'den (09.10.2026) beri BS'yi buluyor (S2_BS "
    "0.99459 / 0.99930 / 0.99992, J `seeds = 12` literalinden bagil <= 1.02e-9). "
    "Kapi S2_BS > 1e-8 "
    "olmali ve SART KALIR (testte kilitli; hf -> spin kodu YOK, yazilirsa bu kapi SART). "
    "HF h URETMEZ: `core_hamiltonian`da manyetik terim yok; UHF'in "
    "tek spin alani kademelidir. KONTROL: h = 0.1'de ozdeslik bozulur (AFM "
    "kademeli ile FM tekbicim max|fark| 0.82421875) -- motor ikinci bir olcek "
    "varsa isareti gorur, HF o olcegi uretmez. ARGUMAN (olcum degil): iki "
    "elektronlu spin-bagimsiz bir sistemin taban hali singlettir, yani J < 0 "
    "bir teoremdir, HF bilgisi degil. tb-spin'deki 'duzeltirsen kenar acilir' "
    "cumlesinin esi: AFM'yi YAML'a acmak ya da kademeli gozlemlenebilir yazmak "
    "da kenari ACMAZ; cikan seri FM ciktisinin ayar-imgesidir. TUR A -- "
    "DUZELTILEMEZ (h = 0 motorunda)."
)

_ON_OLCUM_28_09 = "(28.09.2026 on olcumu; `.superpowers/sdd/olculmedi-bosaltma-on-olcum-2026-09-28.md`)"

_TDSE_SPIN_GEREKCE = (
    "SIFIR kod kenari, HUKUM VAR " + _ON_OLCUM_28_09 + "; 22.09'un YAPISAL_YOK hukmu DUSTU. "
    "KAYNAK GERCEK: 1B soft-Coulomb H2 (e-e dahil), iki elektronun uzayi `Grid2D` + sanal zaman "
    "`SplitOperatorPropagator` (SEVK EDILMIS ama bu birlesim TESTSIZ; triplet izdusumle), "
    "J = (E_S - E_T)/2. PENCERE: n = 256, L = 40, dt = 0.01, yumusatma a = b = 1; J(R = 3/4/5/6 "
    "bohr) = -0.03679538951052319 / -0.010055270069717404 / -0.0021743343691629224 / "
    "-0.00041083190968227434 Ha, bagimsiz eigsh ile fark <= 2.3e-10, yani ~9-10 anlamli hane; "
    "YUMUSATMA DUGMEDIR: a = b 0.707 -> 1.414'te J R = 3/4/6'da 1.88/4.52/23.2 kat oynar. TDSE "
    "tarafi testte KOSMAZ, J'ler LITERAL. " + _SPIN_BUYUKLUGU_SILER + "PROTOKOL A (T = t * |J|): "
    "on olcumde spin_tarama'da 42/42 seri ve E == |J| * E(1) bit bit (14/48 kabul sinifi 1 ulp "
    "farkli); testte `hf -> spin` penceresinde canli. PROTOKOL B (300 K = 9.500e-04 Ha, ayni "
    "pencere, public YAML yolu J <= 0'i ScenarioError ile reddeder, coupling = |J|, sayilar FM "
    "imgesidir): R = 3/4/5'te T/|J| = 0.026 / 0.094 / 0.437 < 0.5, zincir DONAR ya da DOYAR "
    "(pencere ayirmaz) -- |m| serisi tamamen 1.0, kabul 0.0, J = 1 / T = 1e-3 kosumuyla bit bit "
    "ayni; tekbicim, -tekbicim, Neel baslangic ayni seriyi verir, denge |m| = 1 - O(e^(-8|J|/T)) "
    "(R = 5'te 1.1e-8): cikti J'den bagimsiz; R = 6'da T/|J| = 2.312, <|m|> = 0.612 kipirdar ama "
    "ayni T/|J|'de J = 1 kosumunun bit bit etiketidir (o T/|J| sevk listesinde yok, 0/4; ikisi "
    "testte canli). BASLANGIC KANALI (sign(Re psi); yalniz motor yolu, `build_spin` HEP tekbicim): "
    "T = 3.4'te 0.060307 keyfi baslangiclarin disinda ama 25 zincir tohumunda sira kesri tekduze "
    "(KS p = 0.56) -- TASIMAZ. ISARET BIR AYAR ETIKETIDIR (`hf -> spin` gibi; K2: L = 16, h = 0, "
    "T/|J| = 1.6/2.2/3.4 x 4 R, 1200 sweep'in hepsinde bit bit) ve bir TEOREMDIR, TDSE bilgisi "
    "degil (40/40 rastgele sistemde E_S < E_T). TUR A -- DUZELTILEMEZ (h = 0 motorunda)."
)

#: EM HEDEFININ ORTAK TUR B YOLU (28.09.2026): `md -> em` ve `tb -> em`.
_EM_ORTAM_HUKUMSUZ = (
    "TUR B, HUKMU YOK: bilgi tasiyan tek yol EM'e konuma ya da yone gore degisen bir ortam "
    "vermektir (sevk edilen yuzey TEKBICIM tek bir eps_r alir); o yolun buyuklugu serbest bir "
    "beyana baglidir (ARGUMAN). "
)

_MD_EM_GEREKCE = (
    "SIFIR kod kenari, HUKUM VAR " + _ON_OLCUM_28_09 + ". AKIM KANALI (J = sum q v) TUR A: "
    "durust (esit) yukle sevk edilmis uc NVE senaryosunda (h2_morse, h2_aimd, tb_zincir_md) EM "
    "kaydi BIT BIT 0.0 -- momentum korunumu; argon_nvt'de cikti Langevin termostatinin "
    "tohumunun goruntusu (MD'siz bir ozyineleme 1.99e-15 farkla verir, kuvvet alani degisince "
    "fark 1.85e-15); q girdiden geri okunur (K5); S = 1'de DYADIK J'de prob[n] + prob[n-1] == "
    "J[n-d-1] bit bit (MD'nin float J'sinde 15/3989 nokta 1-8 ulp sapar: olcum, ozdeslik degil), "
    "ama MD'siz rastgele DYADIK J'de de (0/3989). Her 0.0 semanin ya da korunumun sifiridir, "
    "koprunun degil -> buyukluk None. ORTAM KANALI (MD konumu -> eps_r) bilgi tasir ve dogrusal "
    "degildir (bagil artik D = 3'te 0.633, D = 0.003'te 0.00105; tb_zincir_md dimerlesmesi "
    "0.075 -> 0.182 bohr yansimayi 0.175 oynatir); D, w, L serbest beyandir (1B pencere L = 0.05 "
    "bohr/hucre, D = 3, w = 1, N = 2001, S = 0.5, PML 15, 7000 adim); sevk (yaricap) merdiveni "
    "bu ortami HIC KOSMADI, D, w, L'ye K5 uygulanmadi, K4 icin md importu SART (ordek tipli "
    "kopru tarayiciya gorunmez). " + _EM_ORTAM_HUKUMSUZ + "Sayilar testte kosturulmaz."
)

_TB_EM_GEREKCE = (
    "SIFIR kod kenari, HUKUM VAR " + _ON_OLCUM_28_09 + ". TEKBICIM ortam TUR A: eps_r = K "
    "kosumu, eps_r = 1, S/sqrt(K), adim * sqrt(K) kosumunun ETIKETIDIR -- K = 4 ve 16'da "
    "panelin her alani bit bit ayni (public `load_scenario` -> `build_em` -> payload; tek dugme "
    "kontrolu, yalniz S, 4.38e-6 ayrisir); eps_r frekanslardan TAM geri okunur (K3/K5) -> "
    "buyukluk None. KONTRAST (ic disk, 0.5R) yolu: TB icerigi ciktiyi gurultunun 1350-4437 kati "
    "kipirdatir ama b serbest beyani TB iceriginden BASKINDIR (b 10 -> 20: -9.0e-2; |t| x 2: "
    "+2.6e-2 / +5.1e-2), metal zincir_tb'de taslak uretec sessizce eps = 1 verir (fail-open) ve "
    "kontrastli kosumlarda sevk merdiveni YARGILANAMAZ der (p_inf None), K3/K4 uygulanmadi. "
    + _EM_ORTAM_HUKUMSUZ + "R^-2 KAYDI: 22.09'un Bessel orani |kayma| serisi (eps_r x4, R = "
    "5/10/20/40: 1.37e-02 .. 4.45e-04) bugun URETILEMIYOR; sevk borusu bugun -3.6e-03 / -1.0e-03 "
    "/ -2.7e-04 / -6.8e-05 verir (ardisik oranlar 3.58 / 3.80 / 3.94): R^-2 yine tutuyor, "
    "tasinan bugunkudur. Sayilar on olcumun kaydidir, testte kosturulmaz."
)

_GRAV_GEREKCE = (
    "Yercekimi atom olcegine BAGLANMAZ ve engel BOYUT DEGIL ENERJIDIR: "
    "iki proton arasinda F_g/F_e = 8.093550856608016e-37. Bu sayi "
    "`engine/grav/factory.py` icindeki sevk edilmis YOKLUK BEYANINDAN "
    "birebir dogrulandi ve CODATA sabitlerinden bagimsiz olarak yeniden "
    "turetildi (ayni bitler). IKI PROTON BIR SECIMDIR: elektron-proton "
    "icin ayni oran 4.407885560760525e-40'tir. Ikinci mertebe (Task 5, "
    "CODATA 2018'den bu gorevde yeniden turetildi, ayni grav beyaniyla "
    "birebir): protonun Schwarzschild yaricapi 2*G*m_p/c^2 = 2.484232e-54 m, "
    "ona inmek icin kenari bir Bohr yaricapi olan bir kupte 9.665562e+129 "
    "nokta gerekir; kenar 1 angstrom secilseydi 6.522644e+130 -- span bir "
    "dugmedir, adlandirilmadan bu sayi yazilamaz. TUR A -- DUZELTILEMEZ."
)

#: `md -> tdse` kenarinin sayisi (22.09.2026, spec 7.2 on olcumunun kapanisi):
#: PLATODA iki-motor Ehrenfest artigi EKSI DONDURULMUS-MD tabani, bohr.
#: Bu iki sayi bir motor kosturularak uretildi ama BURAYA LITERAL yazildi --
#: bu modul hicbir motoru import etmez. `tests/tdse/test_from_md.py` motoru
#: kosturup bu literali canli olcume karsi sinar.
#: SEKIZ ANLAMLI HANE SINIRI: on olcumde iki bagimsiz harness ayni buyuklugu
#: 7 hanede tutturup 8.'de ayristi, o yuzden test `rtol=1e-8` tasir.
_TDSE_MD_KENAR_SAYISI = 0.13167174221268158

#: Gomme katlanmasinin DINAMIGE bedeli (ayni pencere), bohr. Sifir DEGIL.
_TDSE_MD_KATLANMA_BEDELI = 2.4982917457005982e-08

_TDSE_MD_GEREKCE = (
    "KOD KENARI VAR ve tarandi: `engine/tdse/from_md.py` -> "
    "`qlab.engine.md.state.ForceField`. KENAR 22.09.2026'DA YAZILDI; o "
    "tarihe kadar bu cift OLCULMEDI kovasindaydi ve gerekcesi 'hicbir sayi "
    "gelmedi'ydi. spec 7.2 on olcumu (34 ajan) o gerekceyi BAYATLATTI: sayilar "
    "geldi. Ve ortaya KOVA SOZLUGUNDE BIR DELIK cikti -- ciftin olculen "
    "durumu 'karsilik var + veri akabilir + cikti kipirdar + kod kenari "
    "YOK' idi ve bes kovanin hicbiri bunu tasiyamiyordu. Delik kovayi "
    "degistirerek degil KENARI KURARAK kapatildi. "
    "KARSILIK BIT BITTIR: MD'nin `Harmonic(k)` alani ile "
    "`tdse.potentials.harmonic(omega, mass)` ayni izgarada AYNI diziyi "
    "veriyor -- 11 (omega, mass) cifti x 4 izgarada `np.array_equal` True; "
    "cebirsel sebebi 0.5 ile carpmanin ikilik kayan noktada TAM olmasi. "
    "Iki motor bir ANALOJI degil bir NESNE paylasiyor. "
    "BUYUKLUK NEDIR: platoda iki-motor Ehrenfest artigi EKSI dondurulmus-MD "
    "tabani. TABAN CIKARILMAK ZORUNDADIR -- ham artigin %31.9'u TEK "
    "MOTORLUKTUR (kuantum paketinin Morse minimumundaki sifir-nokta "
    "bag-uzama salinimi) ve kenar hakkinda HICBIR SEY soylemez. "
    "PENCERE: Morse(d_e = K_HF/(2a^2), r_e = R_E_BOHR, a = A_ESLESEN), "
    "L = 4.0 n = 512, A = 0.1, sigma = uyumlu, N_BOLME = 200; periyot 18 ve "
    "24'te BIT BIT ayni. Periyot 30'da 0.13164903'e KAYAR cunku "
    "DONDURULMUS-MD TABANI kendi platosunda surukleniyor -- sayinin periyot "
    "ekseni bir dugmedir ve pencereye yazilmadan tasinamaz. "
    "KUSUR BEYANI, GOMME KATLANMASI: `ForceField` sozlesmesi KONUM alir ve "
    "ayrim iceride `r = |r_2 - r_1|`, yani bir NORM'dur ve negatif olamaz; "
    "izgara ise isaretlidir. L = 4.0'da 84 nokta (x < -r_e) KATLANIR ve "
    "potansiyel duvarin tepesinden geri asagi doner. Elle yazilmis bir V "
    "satirinin YAPAMAYACAGI bir hatadir; kenar onu sozlesmeye saygi "
    "duydugu icin MIRAS ALIR. Bedeli OLCULDU: "
    + repr(_TDSE_MD_KATLANMA_BEDELI) + " bohr. Gizlenmiyor, SAYILIYOR "
    "(`MdTdsePotansiyeli.katlanan_nokta`). "
    "NE SEMA CIPASI NE FIZIK CIPASI: `hf -> md`den farki, iki tuketicinin "
    "ayni cozumun soyundan gelmemesi; olculen sey TEK bir yuzeyin IKI ayri "
    "motorda kosulmasinin AYRISMASIDIR. "
    "KENAR SINIFI (26.09.2026, olculdu): YALNIZ TIP KENARI -- `ForceField` "
    "from_md.py'de yalniz bir arguman annotasyonunda gecer (dosyada `from "
    "__future__ import annotations` var); veri akisi ORDEK TIPLEMEYLE yurur "
    "(`(konum) -> (enerji, kuvvet)` donduren her cagrilabilir kabul edilir); "
    "import satiri silinince modul BIT BIT ayni calisiyor -- olculdu "
    "(tests/bridge/test_kenar_tarayici.py, yuk tasima cipasi). Import "
    "ifadesi yine calisir ve kenari TASIYAN odur: kilit 2 yalniz tip kenarini "
    "da kod kenari sayar. Kilit 2'nin calisma zamani kenari isteyip "
    "istemeyecegi Osman'in karari; varsayilan HAYIR. Evet denirse bu cift "
    "bugunku kaynakla GERCEK_KOPRU'da KALAMAZ; `calisma_zamani` muhafazakar "
    "statik bir siniftir ve yuku garanti etmez (dizge annotasyon, future "
    "silme ya da `_ = ForceField` sinifi veri akisi olmadan cevirir)."
)

#: `tb -> md` kenarinin gerekcesi (24.09.2026, spec md-tb ss-sigma 5.1).
#: BUYUKLUK YOK ve sebebi `hf -> md`ninkinden FARKLIDIR: orada taban iki
#: sayidir ve panel onu canli okur; burada kenar tabani HIC OLCULMEDI.
_TB_MD_GEREKCE = (
    "KOD KENARI VAR ve tarandi: `engine/md/tb_kuvvet.py` -> "
    "`qlab.engine.tb.mermin` ve `qlab.engine.tb.radyal`; `engine/md/factory.py` "
    "-> `qlab.engine.tb.radyal` (yasa nesnesini senaryodan kurar). KENAR "
    "24.09.2026'DA YAZILDI; o tarihe kadar bu cift OLCULMEDI kovasindaydi ve "
    "kod kenari yazilinca orada KALAMAZ (`tablo_tutarsizliklari` birinci "
    "kilidi). GERCEK_KOPRU OLCUTU (`tdse -> md`de konan olcut, DORT PARCA): "
    "karsilik var + veri AKABILIR + cikti kipirdar + kod kenari VAR. Dordu de "
    "burada saglanir: KARSILIK -- `SsSigmaTBKuvveti` `ForceField` "
    "sozlesmesini (N,3) -> (Omega, F) doldurur; veri AKABILIR -- nesne bir MD "
    "kuvvet alanidir ve fabrika onu senaryodan kurar; CIKTI KIPIRDAR -- bant "
    "kuvvetinin TAMAMI yazilan yasanin turevinden gelir (`us = 0` ve kuyrukta "
    "cift yokken bant kuvveti 48/48 bilesende bit bit 0.0, "
    "tests/tb/test_mermin.py M10); kod kenari VAR -- yukaridaki uc import. "
    "BU KENARDA OLCUTTEN GUCLU IKI OLGU DA OLCULDU ve ikisi olcutun PARCASI "
    "DEGILDIR (24.09.2026 metni onlari olcut diye yazmisti -- 'veri akabilir' "
    "'VERI AKAR', 'kod kenari var' 'YUK TASIR' diye kaymisti ve kaymis "
    "olcutu `tdse -> md` gecemezdi; 26.09.2026 duzeltmesi): VERI AKAR -- "
    "konum -> H -> Omega, F -> integrator; KOD KENARI YUK TASIR -- "
    "`tb_kuvvet.py`deki mermin importu silinince cagri NameError verir "
    "(olculdu, mutasyonla). KENAR SINIFI CALISMA ZAMANI: `tek_orbital_mermin` "
    "bir cagri yuvasinda yuklenir; `TekOrbitalSonuc` adi yalniz bir donus "
    "annotasyonudur. BU BIR MODEL CIPASIDIR, DOGA CIPASI YOK: "
    "radyal yasa SECILDI, turetilmedi; testlerdeki cipalar (Kohn argmin, "
    "parcacik-bosluk simetrisi, Huckel esikleri) modelin KENDI kuramina "
    "karsi sinanir, dogaya karsi degil, ve yasadan geri okunan bag, genlik "
    "ve esikler fizik iddiasi olarak sevk edilemez. kenar tabani OLCULMEDI: "
    "`hf -> md`nin A1/A2 tabaninin karsiligi (ornegin dimer optik modunun "
    "iki bagimsiz yoldan frekansi) bu kenar icin KURULMADI; bu yuzden "
    "buyukluk bos -- iki sayi oldugu icin degil, taban hic olculmedigi icin. "
    "BAGIMSIZLIK (spec M10): kenar HF'e cagri duzeyinde dokunmaz -- "
    "`tb_kuvvet.py`nin importu tek basina SIFIR hf modulu yukler, TB MD "
    "kosusunda `hf.scf.solve` SIFIR kez cagrilir ve `tb.downfold`dan yasa "
    "almak `ast` ile yasaktir (on olcumde o yasayla TB payi %0.000 cikti: "
    "totoloji; o sayi G4'te tekrar kosturulmadi)."
)

#: OLCULMEDI girdilerinin ORTAK TARIHSEL BASI; NEDEN 28.09.2026'dan beri CIFT BASINADIR.
_OLCULMEDI_TARIHCE = (
    "SIFIR KOD KENARI, HUKUM YOK, buyukluk None. Bu cifti olcen ajan 29.08.2026'da API "
    "baglantisi kopunca oldu ve o gun " + HICBIR_SAYI_GELMEDI + "; 22.09.2026'da sayi geldi ama "
    "HUKUM gelmedi (spec 7.2 on olcumu, `.superpowers/sdd/modul7-72-on-olcum-2026-09-22.md`); "
    + _ON_OLCUM_28_09 + " hukum yine VERILEMEDI ama NEDENI OLCULDU: "
)

_TDSE_TB_NEDEN = (
    "sentezin 'iki-motor artigi eksi motorsuz taban <= 2.2e-16' olcutu bir TANIM SECIMIDIR "
    "(depo `md -> tdse`de tabani DONDURULMUS-MD, yani tek-motor tabani diye tanimlar) ve "
    "POZITIF KONTROLU YAPILMADI -- ayni olcut `md -> tdse`ye uygulansa bagimsiz bir cozucu "
    "0.1317'yi de uretir ve sevk edilmis bir GERCEK_KOPRU 'sifir' cikar (ARGUMAN). Tutulmus "
    "k'larda artik geri okuma degil FIZIGE BAGLI yakinsamadir (Mathieu s = 6/8/10 -> 8.08e-7 / "
    "4.22e-8 / 2.77e-9) ve ters-sira K3 testi KORDUR (artik gurultu duzeyindeyken de 20/20 kip "
    "`==`). Kova, tablo yazilip SEVK YOLUNDAN cagrilir mi sorusuna baglidir (`hf -> tb` emsali: "
    "akis yok -> TAUTOLOJI); o sinav kosturulmadi."
)

_SPIN_EM_NEDEN = (
    "p-kutuplanmada (`oblique.py`, TE_z; Ising'in s^z ekseni 2B TM_z bosluguna oturmaz) mu = M "
    "ile eps = M yari-uzay kosumlari prob kaydinda bit bit ayni (12/12; M = 4 / "
    "1.0653663504464286 / 2.25 / 1.001, ky = 0 / 0.08 / 0.14; ObliqueWindow varsayilanlari, 9000 "
    "adim -- 20000 adimda M = 1.0654'te 12376. adimdan sonra ayrisir, bagil 1.9e-3: ikizlerin PML "
    "sonumu farkli, `oblique.py` notu) -- OLCULDU AMA TEK ve CURUTULMEMIS bir olcum. Duzlem ici "
    "anizotrop yol YON-BAGIMLIDIR: daire boslukta fiziksel olarak esdeger x ve y eksenleri farkli "
    "hukum verir (c = 0.017'de x OTURDU, p_inf 3.007; y OTURMADI, p_inf 1.612; c = 0.065 ve "
    "0.5'te iki eksen de YARGILANAMAZ) ve perturbasyon uyumu tek bir c'de (0.065; 0.02 / 0.15 / "
    "0.3'te tutmuyor). "
    "Hukum Ising ekseninin ve Lambda'nin BEYANINA baglidir: (mu - 1)/Lambda bit bit chi'dir ve "
    "Lambda'nin hicbir bileseni depoda yok."
)

KOVA_TABLOSU: tuple[EdgeFact, ...] = (
    # --- GERCEK KOPRU (3) ---------------------------------------------------
    EdgeFact(
        kaynak="md",
        hedef="tdse",
        kova=Kova.GERCEK_KOPRU,
        gerekce=_TDSE_MD_GEREKCE,
        buyukluk=_TDSE_MD_KENAR_SAYISI,
    ),
    EdgeFact(
        kaynak="hf",
        hedef="md",
        kova=Kova.GERCEK_KOPRU,
        gerekce=(
            "KOD KENARI VAR ve tarandi: `engine/md/forcefields.py:13` -> "
            "`qlab.engine.hf.scf.solve`. `scenarios/h2_aimd.yaml`in GIRDI "
            "ALANLARI ne `k` ne `omega` tasir (`scenarios/h2_aimd.yaml:3`teki "
            "YORUM `k = 0.5729363` sayisini anar, ama o bir girdi degil "
            "bir not); yuzey SCF cozulerek uretilir. AMA BU BIR SEMA "
            "CIPASIDIR, FIZIK CIPASI DEGIL: iki tuketici de ayni HF "
            "cozumunun soyundan gelir, uyum dogayi degil IKI TUKETICI "
            "UYGULAMASINI olcer. Kenarin TABANI `observable/bridge_obs.py`de "
            "olculur ve IKI SAYIDIR (A1 aritmetik, A2 hata cubugu); tek bir "
            "`buyukluk` alanina sigmaz ve burada literal tutulmaz -- panel "
            "onu her kosumda CANLI okur (Task 5). Bu yuzden burada buyukluk "
            "None: olculmedigi icin degil, iki sayi oldugu icin."
        ),
        buyukluk=None,
    ),
    EdgeFact(
        kaynak="tb",
        hedef="md",
        kova=Kova.GERCEK_KOPRU,
        gerekce=_TB_MD_GEREKCE,
        buyukluk=None,
    ),
    # --- TAUTOLOJI (8) ------------------------------------------------------
    EdgeFact(
        kaynak="hf",
        hedef="tb",
        kova=Kova.TAUTOLOJI,
        gerekce=(
            "KOD KENARI VAR ve tarandi: `engine/tb/downfold.py:130-132` -> "
            "`qlab.engine.hf.{basis, integrals, multistart}`. Ama BORU "
            "HATTINDA AKIS YOK: `model_from_downfold` (tanim "
            "`downfold.py:287`) `src/` icinde hic cagrilmiyor -- `ast` ile "
            "kilitli (Task 4); tuketicileri YALNIZ testlerdir "
            "(`tests/tb/test_downfold_kesme.py` ve "
            "`tests/bridge/test_dusen_adaylar.py`); olan sey `downfold.py`nin "
            "`tb/` klasorunde OTURMASIDIR. TAUTOLOJI YENIDEN OLCULDU (Task 4, "
            "02.09.2026): `t_m` halka kabugu uzerinde bir `np.mean` olarak "
            "TANIMLANIYOR (`downfold.py:259-271`), sonra `p.hoppings[m-1]` "
            "ile Python `==`'inde geri okunuyor -- ALTI sistemde, (N, d) = "
            "(6, 2.6) (6, 3.0) (6, 3.6) (8, 2.0) (8, 2.6) (10, 2.6), STO-3G, "
            "seeds = 12, HER kabukta True, |fark| = 0.0; `eps_bar` ve "
            "`delta` da ayni. Bu bir UYUM degil TANIMIN YENIDEN "
            "KOSULMASIDIR: ayni elemanlar TERS sirayla toplaninca 22 kabugun "
            "10'unda `==` DUSUYOR (fark en cok 1 ulp). 'ON ALTI HANE' "
            "DARALTILDI: tam menzilde max|model - H_eff^alfa| ozdesligi (en "
            "buyuk CIFT kabuk yayiliminin yarisi) referans pencerede "
            "(6, 2.6) bagil 2.276006188306662e-15 = ON BES hane, ve SISTEME "
            "BAGLI: (6, 3.0) 2.024e-15 / (6, 3.6) 4.729e-14 / (8, 2.0) "
            "1.090e-08 / (8, 2.6) 1.310e-08 / (10, 2.6) 3.769e-08 -- yedi "
            "mertebe. Lowdin adimi tam bir benzerlik donusumu: "
            "eig(H_eff^alfa) ile UHF alfa orbital enerjileri arasindaki fark "
            "(6, 2.6)'da 3.490441824460788e-10, ALTI sistemin EN KUCUGU; en "
            "buyugu (8, 2.6) 2.8906645410664566e-09 (8.28 kat). PENCERE "
            "(N, d, baz, seeds) YAZILMADAN bu sayilar tasinamaz; buyukluk, "
            "referans pencerenin bagil ozdeslik artigidir."
        ),
        buyukluk=_HF_TB_OZDESLIK_BAGIL,
    ),
    EdgeFact(
        kaynak="tb",
        hedef="spin",
        kova=Kova.TAUTOLOJI,
        gerekce=(
            "SIFIR kod kenari, ama HUKUM VAR: `J = 4t^2/U` koprusu BIT BIT "
            "olu. YENIDEN OLCULDU (Task 4, 02.09.2026). Kaynak: N = 8 "
            "d = 2.0 STO-3G seeds = 12 halkasi, t1 = -0.407144651592, "
            "Delta = +0.193177917796; bosluk denklemi U = 0.827765644 -> "
            "U/|t| = 2.033100 (guclu baglasim sarti IHLAL -- BU BORUDA "
            "OLCULEMEZ). Bes U'dan bes J: 0.801032362 / 0.750154440 / "
            "0.407144652 / 0.203572326 / 0.119479403 (U/|t| = 2.033100 / "
            "2.170991 / 4.000000 / 8.000000 / 13.630622), her biri kendi "
            "T = 2.269185 * J'siyle, AYNI pencere: Ising, L = 32, h = 0, "
            "tohum 20260822, n_burn = 500, n_sweep = 1500. Besinde de "
            "max|m farki| = 0.0 (bit bit), E == J * E(J=1) (bit bit), kabul "
            "orani 0.187216796875 = 287565/1536000 besinde `==`. J 6.70 kat "
            "degisti, cikti SIFIR kipirdadi. " + _SPIN_BUYUKLUGU_SILER +
            "KONTROL: h = 0.1'de ayni iki J'nin ciktisi "
            "9.961e-02 ayrisiyor -- motor ikinci bir olcek varsa J'yi GORUR, "
            "TB o olcegi (h) URETMEZ. BU YUZDEN TEK SATIR, kovasi TAUTOLOJI: "
            "U'yu buyutmek kenari ACMAZ (U/|t| = 4.07 / 8.13 / 16.26 / "
            "101.65 / 1016.55'te de bit bit ayni) ve z uyusmazligi (TB "
            "halkasi 1B z = 2 <-> spin motoru kodda 2B z = 4, "
            "`lattice.py::neighbour_sum` dort `np.roll`) da ACMAZ (J * 2/4 "
            "ile de bit bit ayni) -- BU BORUDA OLCULEMEZ, cunku engel ORAN "
            "SADELESMESIDIR. Ayri satir 'duzeltirsen kenar acilir' diye "
            "okunur ve ACILMIYOR. TUR A -- DUZELTILEMEZ (bu boruda). "
            "ISARET BEYANI (27.09.2026): kaynak ANTIFERROMANYETIKTIR "
            "(`engine/tb/downfold.py:20`; TB paneli `converged` tikini "
            "iki-siteli antiferromanyetik yapinin bulunmasina baglar, "
            "`engine/tb/factory.py:143` ve `engine/tb/factory.py:183-187`), motor ise c = +J ile "
            "FM kosar (E = -c * sum s s). +J bir AYAR SECIMIDIR: h = 0 cift L'de "
            "AFM(-J) Neel baslangicli kosum her sweep'te bununla bit bit ayni "
            "(s(-J) == eta * s(J)); kilitli 0.687930 dogru isaretli AFM'nin "
            "KADEMELI ortalamasidir. Carpan (-J/4 ya da -J/2, esleme olcutune "
            "gore) h = 0 oran protokolunde sadelesir. Yukaridaki 9.961e-02 KONTROLU FM "
            "penceresine aittir: dogru isaretle ayni iki J h = 0.1'de tekbicimde "
            "5.078e-02, kademelide 8.809e-01 ayrisiyor (tekbicim baslangic; "
            "carpansiz c = -J penceresi -- h != 0'da carpan SADELESMEZ: "
            "c = -J/4'te 6.250e-02 / 7.812e-01) -- "
            "hukum ayakta, sayi isarete ve carpana bagli. Isaret CEVRILMEDI: cevrilseydi "
            "kilitli 0.687930 bozulurdu ve hicbir sey kazanilmazdi."
        ),
        buyukluk=_TB_SPIN_M_FARKI,
    ),
    EdgeFact(
        kaynak="hf",
        hedef="spin",
        kova=Kova.TAUTOLOJI,
        gerekce=_HF_SPIN_GEREKCE,
        buyukluk=_HF_SPIN_M_FARKI,
    ),
    EdgeFact(kaynak="tdse", hedef="spin", kova=Kova.TAUTOLOJI,
             gerekce=_TDSE_SPIN_GEREKCE, buyukluk=_TDSE_SPIN_M_FARKI),
    EdgeFact(kaynak="md", hedef="em", kova=Kova.TAUTOLOJI, gerekce=_MD_EM_GEREKCE),
    EdgeFact(kaynak="tb", hedef="em", kova=Kova.TAUTOLOJI, gerekce=_TB_EM_GEREKCE),
    EdgeFact(
        kaynak="md",
        hedef="spin",
        kova=Kova.TAUTOLOJI,
        gerekce=(
            "SIFIR kod kenari. Kova hukmu 29.08.2026 on olcumunden "
            "DEVRALINDI ve " + YENIDEN_URETILMEDI + ": bu depoda bu cifti "
            "kosturan bir kod yoktur, bu yuzden buyukluk None. Alinti "
            "oldugu yazilmasaydi tablo kendi olcmedigi bir hukmu kendi "
            "olcumu gibi gosterirdi."
        ),
        buyukluk=None,
    ),
    EdgeFact(
        kaynak="hf",
        hedef="em",
        kova=Kova.TAUTOLOJI,
        gerekce=(
            "SIFIR kod kenari. Kova hukmu 29.08.2026 on olcumunden "
            "DEVRALINDI ve " + YENIDEN_URETILMEDI + ". Kirilma indisi "
            "uretecinin kendisi KAPSAM DISI (spec 8) -- TUR B, ERTELENDI. "
            "Bu yuzden buyukluk None."
        ),
        buyukluk=None,
    ),
    # --- YAPISAL OLARAK YOK (8) ---------------------------------------------
    EdgeFact(
        kaynak="hf",
        hedef="tdse",
        kova=Kova.YAPISAL_YOK,
        gerekce=(
            "3B TDSE govdesi YOK; olsaydi O 1s'i cozecek izgara "
            "1600^3 = 4.096e+09 nokta ister ve tek bir complex128 dizi "
            "61.04 GiB (65.536 GB) tutar. IKI SAYI DA BURADA YENIDEN "
            "TURETILDI (1600^3 ve 4.096e+09 * 16 bayt). SPAN VE COZUNURLUK "
            "BIRER DUGMEDIR: adlandirilmadan bu sayi yazilamaz. "
            "TUR B -- ERTELENDI (spec 8, kapsam disi)."
        ),
        buyukluk=_HF_TDSE_NOKTA,
    ),
    EdgeFact(
        kaynak="em",
        hedef="tdse",
        kova=Kova.YAPISAL_YOK,
        gerekce=(
            "Alan genligi koprusunun donusum carpani BUYUK DEGIL, "
            "MEVCUT DEGIL: TDSE motoru bir elektrik alani girdisi tasimaz, "
            "EM motoru bir dalga fonksiyonu tasimaz. Yazilacak bir MERTEBE "
            "yoktur -- eksik olan sayi degil KARSILIK -- bu yuzden "
            "buyukluk None. TUR A -- DUZELTILEMEZ (bu boruda)."
        ),
        buyukluk=None,
    ),
    EdgeFact(
        kaynak="grav",
        hedef="tdse",
        kova=Kova.YAPISAL_YOK,
        gerekce=_GRAV_GEREKCE,
        buyukluk=_GRAV_ORANI,
    ),
    EdgeFact(
        kaynak="grav",
        hedef="hf",
        kova=Kova.YAPISAL_YOK,
        gerekce=_GRAV_GEREKCE,
        buyukluk=_GRAV_ORANI,
    ),
    EdgeFact(
        kaynak="grav",
        hedef="md",
        kova=Kova.YAPISAL_YOK,
        gerekce=_GRAV_GEREKCE,
        buyukluk=_GRAV_ORANI,
    ),
    EdgeFact(
        kaynak="grav",
        hedef="tb",
        kova=Kova.YAPISAL_YOK,
        gerekce=_GRAV_GEREKCE,
        buyukluk=_GRAV_ORANI,
    ),
    EdgeFact(
        kaynak="grav",
        hedef="spin",
        kova=Kova.YAPISAL_YOK,
        gerekce=_GRAV_GEREKCE,
        buyukluk=_GRAV_ORANI,
    ),
    EdgeFact(
        kaynak="grav",
        hedef="em",
        kova=Kova.YAPISAL_YOK,
        gerekce=_GRAV_GEREKCE,
        buyukluk=_GRAV_ORANI,
    ),
    # --- BENZERLIK (0) ------------------------------------------------------
    # Bicimsel analoji kovasi BOS: bu turda olculen yok. Bos kalmasi bir
    # ihmal degil bir OLCUM SONUCUDUR ve sayisi (0) testte kilitlidir.
    # --- OLCULMEDI (2) ------------------------------------------------------
    EdgeFact(
        kaynak="tdse",
        hedef="tb",
        kova=Kova.OLCULMEDI,
        gerekce=_OLCULMEDI_TARIHCE + _TDSE_TB_NEDEN,
        buyukluk=None,
    ),
    EdgeFact(
        kaynak="spin",
        hedef="em",
        kova=Kova.OLCULMEDI,
        gerekce=_OLCULMEDI_TARIHCE + _SPIN_EM_NEDEN,
        buyukluk=None,
    ),
)

#: Kova basina beklenen girdi sayisi -- KOVA TARIHCESININ TEK YERI (dokstring ve yorumlar
#: onu tekrarlamaz). OLCULDU (30.08.2026): 1 / 4 / 8 / 0 / 8; OLCULMEDI'deki SEKIZ cift
#: (tdse-md, tdse-tb, tdse-spin, hf-spin, md-tb, md-em, tb-em, spin-em) 29.08'de ajan olunce dustu.
#: 22.09.2026'DA DEGISTI -> 2 / 4 / 8 / 0 / 7: `md -> tdse` kenari YAZILDI ve
#: cift OLCULMEDI'den GERCEK_KOPRU'ya gecti. Toplam 21 = C(7, 2) KORUNUR.
#: 24.09.2026'DA YINE DEGISTI -> 3 / 4 / 8 / 0 / 6: `tb -> md` kenari
#: YAZILDI (`engine/md/tb_kuvvet.py`) ve cift ayni yolu izledi. Kova
#: sozlugu BUKULMEDI: yeni bir kova acilmadi, cift var olan GERCEK_KOPRU
#: olcutunun dort parcasini sagladigi icin oraya gecti.
#: 27.09.2026'DA YINE DEGISTI -> 3 / 5 / 8 / 0 / 5: `hf -> spin` KENARSIZ
#: gecti -- HUKUM verildi (buyukluk sadelesir, isaret h = 0'da bir ayar
#: etiketi). Yine yeni kova ACILMADI: "yalniz isaret" kovasi bir girdi geri
#: okumasini kova duzeyinde odullendirirdi.
#: 28.09.2026'DA YINE DEGISTI -> 3 / 8 / 8 / 0 / 2 (olculmedi bosaltma on olcumu):
#: `tdse -> spin`, `md -> em`, `tb -> em` KENARSIZ, HUKUMLE TAUTOLOJI'ye gecti; `tdse -> tb`
#: ve `spin -> em` OLCULMEDI'de KALDI ama gerekceleri "sayi geldi, hukum gelmedi" degil
#: OLCULMUS NEDENDIR. Yeni kova yine ACILMADI. KOR KOL BEYANI: tasinan uc cift kilit 1'in
#: kenar tuzagini kaybeder -- TAUTOLOJI'de kod kenari yazilirsa tablo kirmiziya DONMEZ.
#: Bu esleme bir testle hem tabloya hem `MODULLER`in kombinasyonlarina karsi
#: sinanir.
KOVA_SAYILARI: dict[str, int] = {
    Kova.GERCEK_KOPRU: 3,
    Kova.TAUTOLOJI: 8,
    Kova.YAPISAL_YOK: 8,
    Kova.BENZERLIK: 0,
    Kova.OLCULMEDI: 2,
}


#: KENAR DUGUMU OLUP MUTLAK OLCEK EKSENINE DOKUNMAYAN MOTORLER.
#:
#: Modul 7 (30.08.2026) bir OZDESLIK olctu: "`units.py`ye dokunan kume
#: {hf, md, tb} = bagli kume" -- yani mutlak olcege dokunanlarla koprulerle
#: bagli olanlar AYNI kumeydi, ve kalan dort motor hem eksensiz hem bagsizdi.
#:
#: O OZDESLIK 22.09.2026'DA COTU, ve curuten sey bu depodaki bir olcumdur:
#: `md -> tdse` kenari yazilinca `tdse` bir kenar dugumu oldu ama
#: `olculen_units_ekseni()["tdse"]` hala 0 -- TDSE motoru `qlab.units`e hic
#: dokunmuyor. Yani ozdeslik bir YASA DEGIL, eski topolojinin OLCULMUS BIR
#: RASTLANTISIYMIS.
#:
#: Kilit GEVSETILMEDI, DARALTILDI: "eksenli kume kenar dugumlerinin ALT
#: KUMESIDIR" hala zorlanir (eksene dokunup bagsiz kalan bir motor cikarsa
#: KIRMIZI), ve eksensiz kalan kenar dugumleri ADIYLA burada sayilir. Yeni
#: bir olceksiz dugum eklenirse bu kume guncellenmek ZORUNDA kalir --
#: sessizce buyuyemez.
#:
#: 24.09.2026: `tb -> md` kenari eklendi ve kume DEGISMEDI -- iki ucu da
#: zaten eksenliydi (md 3, tb 2 dosya). Bu bir SART olarak korunuyor:
#: Kelvin -> hartree donusumu `md/factory.py`de kalir ve kenarin yeni
#: dosyalari birim modulunu import etmez (on olcumde olculdu, spec 5.2: md
#: dosyasi import edince md 4 oluyor, tb dosyasi import edince tb 3).
OLCEKSIZ_KENAR_DUGUMLERI: frozenset[str] = frozenset({"tdse"})

#: GERCEK_KOPRU KOD KENARLARININ BEYANLI SINIFI (26.09.2026). Anahtar YONLU
#: kod kenaridir (import eden -> import edilen), `EdgeFact`in kavramsal yonu
#: DEGIL: `hf -> md` girdisinin kod kenari `md -> hf`dir. Kilit 5
#: (`tablo_tutarsizliklari`) GERCEK_KOPRU ciftlerinin OLCULEN kenar
#: siniflarini bu sozlukle karsilastirir; bir sinif sessizce degisemez.
#: `tb -> hf` TAUTOLOJI kovasinda, bu beyanin disinda; sinifi yalniz canli
#: satirda gorunur. `tdse -> md` YALNIZ TIP ve kilit 2 onu yine kod kenari
#: sayar: kilit 2'nin calisma zamani kenari isteyip istemeyecegi Osman'in
#: karari; varsayilan HAYIR. EVET denirse bu cift bugunku kaynakla
#: GERCEK_KOPRU'da KALAMAZ (kova sayimi degisir, 22.09'un "delik kenari
#: kurarak kapatildi" hukmu yeniden acilir). SINIFIN ANLAMI TEK YONLUDUR:
#: `yalniz_tip` yukun YOKLUGUNU iddia eder ve yuk tasima cipasiyla kilitlidir;
#: `calisma_zamani` muhafazakar bir STATIK sinif, yuku garanti etmez -- dizge
#: annotasyon, future importunu silmek ya da modul duzeyinde `_ = ForceField`
#: sinifi veri akisini degistirmeden cevirir (olculdu, sentetik agac; sonuncusu
#: import silme olcusunu de cevirir). "Evet" kalici bir sart olacaksa bu
#: kacislari kapatan bir olcute baglanmalidir.
GERCEK_KOPRU_KENAR_SINIFLARI: dict[tuple[str, str], str] = {
    ("md", "hf"): KENAR_CALISMA_ZAMANI,
    ("md", "tb"): KENAR_CALISMA_ZAMANI,
    ("tdse", "md"): KENAR_YALNIZ_TIP,
}


# --- `qlab.units` ekseni (Task 5) ------------------------------------------


def olculen_units_ekseni(kok: Path | str | None = None) -> dict[str, int]:
    """Her motor paketinde `qlab.units`i GERCEKTEN import eden dosya sayisi.

    Spec 2.2'nin tasiyici iddiasinin KOD TARAFI: bir merdiven tum basamaklarin
    tek bir mutlak eksene dizilmesini ister, ve mutlak eksene dokunan motor
    kumesi `{hf, md, tb}`. Bu kume 30.08.2026'da iki capraz-motor kenarinin
    dugum kumesinin AYNISIYDI; 22.09.2026'dan beri kenar dugumlerinin ALT
    KUMESIDIR (`tdse` bagli ama olceksiz, bkz. `OLCEKSIZ_KENAR_DUGUMLERI`) ve
    24.09.2026'daki `tb -> md` kenari bunu degistirmedi. Panel bu kumeyi elle
    yazmaz, buradan okur; `olculen_kenarlar()`
    ile ayni `ast` makinesini kullanir ve `tests/bridge/test_kova_tablosu.py`
    ayni kumeyi BAGIMSIZ bir tarayiciyla da olcer (iki tarayici ayni
    makineye dayansaydi cipa dairesel olurdu).

    OLCULDU (30.08.2026, b221a9c; Task 5'te yeniden): hf 1 (`analysis.py`),
    md 3 (`factory.py`, `forcefields.py`, `thermostats.py`), tb 2
    (`downfold.py`, `factory.py`), tdse/spin/em/grav 0. Duz metin taramasi
    md icin 4 sayar (`engine/md/__init__.py` dokstringi adi yaziyor ama
    import etmiyor); bu fonksiyon 3 sayar, cunku bir dokstring bir
    `ImportFrom` dugumu degildir. 26.09.2026'dan beri alt paket dosyalari
    da sayilir ve ust duzey motora atfedilir (gercek agacta alt paket yok;
    sayilar ayni).
    """
    return dict(_agaci_tara(kok).eksen)


def tablo_tutarsizliklari(
    kenarlar: frozenset[tuple[str, str]] | set[tuple[str, str]],
    eksen: dict[str, int],
    *,
    taranamayan: tuple[str, ...],
    siniflar: dict[tuple[str, str], str],
) -> tuple[str, ...]:
    """Kova tablosunun KODLA CELISTIGI noktalar -- bos demet tutarlilik demek.

    Panelin `converged` hukmu buna baglidir: bayat bir yokluk beyani bir
    yalandir ve bir yalanin yaninda tik basilmaz. Bes kilit:

      1. taranan her kod kenari tabloda olmali ve YAPISAL_YOK ya da
         OLCULMEDI kovasinda OLAMAZ,
      2. GERCEK_KOPRU kovasindaki her ciftin bir kod kenari OLMALI (yalniz
         tip kenari da kod kenari SAYILIR; kilit 2'nin calisma zamani
         kenari isteyip istemeyecegi Osman'in karari: HAYIR, 08.10.2026),
      3. kova basina sayim `KOVA_SAYILARI` ile ayni olmali,
      4. `qlab.units` ekseni (dosya sayisi > 0 olan motorlar) kenar
         dugumlerinin ALT KUMESI olmali ve eksensiz kalan kenar dugumleri
         tam olarak `OLCEKSIZ_KENAR_DUGUMLERI` olmali (spec 2.2'nin
         ozdesligi 22.09.2026'da DARALTILDI; kod bunu uyguluyor),
      5. (26.09.2026) `siniflar`in anahtarlari `kenarlar` olmali ve
         GERCEK_KOPRU ciftlerinin olculen kenar siniflari
         `GERCEK_KOPRU_KENAR_SINIFLARI` ile AYNI olmali.

    FAIL-CLOSED: `taranamayan`daki her yasak bicim `TARANAMAYAN: ...` diye
    ADIYLA raporlanir -- goremedigi bir yolu olan tarama tabloyu dogrulayamaz.
    `taranamayan` ve `siniflar` keyword-only ve ZORUNLUDUR: varsayilan
    olsaydi onlari unutan bir cagiran "olculmedi"yi "olculdu, sifir" diye
    okurdu.

    `tests/bridge/test_kova_tablosu.py` ilk iki kilidi kendi bagimsiz
    kopyasiyla da kurar; bu fonksiyon panel icindir ve testin yerini tutmaz.
    """
    kova_of = {f.cift: f.kova for f in KOVA_TABLOSU}
    sorunlar: list[str] = []
    for kaynak, hedef in sorted(kenarlar):
        cift = tuple(sorted((kaynak, hedef)))
        if cift not in kova_of:
            sorunlar.append(f"taranan kenar {cift} tabloda YOK")
        elif kova_of[cift] in (Kova.YAPISAL_YOK, Kova.OLCULMEDI):
            sorunlar.append(
                f"taranan kenar {cift} tabloda {kova_of[cift]} kovasinda"
            )
    yonsuz = {tuple(sorted(k)) for k in kenarlar}
    for f in KOVA_TABLOSU:
        if f.kova == Kova.GERCEK_KOPRU and f.cift not in yonsuz:
            sorunlar.append(f"GERCEK_KOPRU {f.cift} icin kod kenari YOK")
    sayim: dict[str, int] = {k: 0 for k in KOVALAR}
    for f in KOVA_TABLOSU:
        sayim[f.kova] = sayim.get(f.kova, 0) + 1
    if sayim != KOVA_SAYILARI:
        sorunlar.append(f"kova sayimi {sayim} != {KOVA_SAYILARI}")
    eksenli = {ad for ad, n in eksen.items() if n > 0}
    dugumler = {m for kenar in kenarlar for m in kenar}
    if not eksenli <= dugumler:
        sorunlar.append(
            f"units eksenli {sorted(eksenli - dugumler)} kenar dugumu DEGIL"
        )
    olceksiz = dugumler - eksenli
    if olceksiz != OLCEKSIZ_KENAR_DUGUMLERI:
        sorunlar.append(
            f"olceksiz kenar dugumleri {sorted(olceksiz)} != "
            f"{sorted(OLCEKSIZ_KENAR_DUGUMLERI)} -- bir kenar dugumu mutlak "
            "olcek eksenine sessizce girdi ya da cikti"
        )
    if set(siniflar) != set(kenarlar):
        sorunlar.append("KENAR SINIFI: sinif sozlugu kenar kumesiyle ayni degil")
    gk = {f.cift for f in KOVA_TABLOSU if f.kova == Kova.GERCEK_KOPRU}
    olculen = {k: s for k, s in siniflar.items() if tuple(sorted(k)) in gk}
    for a, b in sorted(set(olculen) | set(GERCEK_KOPRU_KENAR_SINIFLARI)):
        beyan = GERCEK_KOPRU_KENAR_SINIFLARI.get((a, b), "YOK")
        if olculen.get((a, b), "YOK") != beyan:
            sorunlar.append(
                f"KENAR SINIFI: {a} -> {b} beyan {beyan}, olculen "
                f"{olculen.get((a, b), 'YOK')}"
            )
    sorunlar.extend(f"TARANAMAYAN: {t}" for t in taranamayan)
    return tuple(sorunlar)


# --- PANEL BEYANLARI (Task 5) -- TEK KAYNAK BURASI ---------------------------
#
# `run.py::_run_bridge` ve `viz/bridge_frames.py` bu metinlerin HICBIRINI
# yeniden yazmaz: `viz` sayilari toplar ve `panel_metinleri()`ni cagirir,
# `run` yalnizca `Provenance.as_lines()` ciktisini ve bir SAYI tablosunu basar.
# Gerekce Modul 6'nin `engine/grav/factory.py`sindekiyle aynidir ve bu depoda
# OLCULEREK ogrenildi: ayni beyan blogu bir kez iki dosyada kopyalanmisti ve
# biri duzeltilip oteki unutuldugunda panel iki farkli sey iddia ediyordu.
# Kilit iki katmanlidir (tam cumle + PARCA) ve
# `tests/bridge/test_bridge_senaryo.py`dedir; parca esigi orada YENIDEN
# OLCULDU, Modul 6'dan devralinmadi.
#
# SAYILAR: bu blokta gecen her sayi ya bu kosumda CANLI uretilir
# (`canli_satir`) ya da bir testte KILITLI bir literaldir; ucuncu bir kaynak
# yoktur. Kenar tabanlari SEKIZ anlamli haneyle yazilir -- olculdu, taban
# sekiz haneden fazlasina yazilamaz (`observable/bridge_obs.py`).

_YONTEM = (
    "Moduller arasi YOKLUK HARITASI ve TABANI OLCULEN kenarin (`hf -> md`) "
    "taban olcumu; "
    "bu modul SIFIR YENI FIZIK sevk eder. Kova tablosu ELLE LISTELENMEZ, "
    "KODDAN TURETILIR: `olculen_kenarlar()` `src/qlab/engine/*/` agacini "
    "`ast` ile tarar, `olculen_units_ekseni()` ayni yontemle `qlab.units`e "
    "dokunan motor kumesini olcer ve tablo HER KOSUMDA ikisine karsi sinanir "
    "(`tablo_tutarsizliklari`); celisirse panel tik basmaz. Kenar tabani "
    "`tests/md/test_aimd.py`nin YOL B borusu harfi harfine kosturularak "
    "okunur: yalniz `ForceField` nesnesi degistirilir, `engine/md`nin tek "
    "satiri degismez ve `hf.scf.solve` SIFIR kez cagrilir"
)

_YAKLASIMLAR: tuple[str, ...] = (
    "KENAR = STATIK IMPORT KENARI: `ast` ile `Import`/`ImportFrom` dugumleri "
    "sayilir (alt paketler ve goreli importlar dahil); calisma zamani cagri "
    "grafigi ve dizge icinde anilan modul adlari SAYILMAZ, ADI SAYILAN "
    "yukleme bicimleri motor agacinda YASAKTIR (liste sonlu; aklama, "
    "`os.sys`, `.pth` gorunmez) ve taranamayan yol olarak ADIYLA "
    "raporlanir -- dizgedeki ad sayilsaydi tablo bir yanlis "
    "pozitifle bozulurdu (olculdu, sentetik agac) ve `md`nin `units` sayisi "
    "3 yerine 4 cikardi (olculdu, gercek agac)",
    "KENAR TABANI ICIN HF YUZEYI DONDURULDU: SCF cozumu yerine ayni `K` ve "
    "`r_e`li bir Morse ikizi (A2) ya da onun `a -> 0` cebirsel limiti (A1) "
    "kosturulur; boru `test_aimd.py`nin YOL B'sinin AYNISIDIR ve bu bir "
    "iddia degil bir olcumdur (HF alaniyla cagrildiginda cipanin sayisini "
    "bit bit veriyor, testte kilitli)",
    "DUSEN ADAYLARIN SAYILARI TASINIYOR, YENIDEN KOSTURULMUYOR: `hf -> tb`, "
    "`tb -> spin`, `hf -> spin` (27.09) ve `tdse -> spin` (28.09; TDSE yani KOSMAZ) sayilari "
    "`KOVA_TABLOSU` gerekcelerinde literal olarak "
    "durur ve `tests/bridge/test_dusen_adaylar.py` onlari motoru kosturarak "
    "dogrular; bu senaryo onlari yeniden OLCMEZ, yeniden OKUR",
    "GERCEK HATA ve KUTLE KORLUGU sayilari HF borusundan gelir ve o boru bu "
    "senaryoda KOSMAZ (olculdu: tek YOL B kosumu ~28 s, kutle testi ~78 s); "
    "ikisi de `slow` testlerde canli uretilir, burada KAYITTIR",
)

_YONTEM_HEDEF_A = (
    "YONTEM/HEDEF UYUSMAZLIGI TUR A -- BU BORUDA DUZELTILEMEZ, dugme yok: "
    "(1) `grav` atom olcegine BAGLANMAZ, engel boyut degil ENERJIDIR "
    "(iki proton icin F_g/F_e = 8.093550856608016e-37); (2) `spin` `J` "
    "hakkinda YANILAMAZ, cunku `coupling` ciktiya yalniz "
    "`coupling/temperature` orani olarak girer (`metropolis.py:189-191`) ve "
    "`U`yu 1000 kat buyutmek de `z`yi uyusturmak da kenari acmaz. Ikisi de "
    "bir tolerans ya da izgara ayariyla KAPANMAZ; A'yi B diye yazmak "
    "'sonraki gorevde acilir' yalanini uretirdi"
)

_YONTEM_HEDEF_B = (
    "YONTEM/HEDEF UYUSMAZLIGI TUR B -- ERTELENDI, engel yok, spec 8 kapsam "
    "disi: (1) `hf -> tdse` icin 3B TDSE govdesi (O 1s icin 1600^3 = "
    "4.096e+09 nokta, tek complex128 dizi 61.04 GiB = 65.536 GB; span ve "
    "cozunurluk birer dugme); (2) `hf -> em` icin kirilma indisi ureteci. "
    "(3) `md -> em` ve `tb -> em` icin EM'e konuma bagli ortam (28.09.2026: bilgi tasiyor, "
    "hukmu YOK -- `tb -> em`de sevk merdiveni YARGILANAMAZ, `md -> em`de hic kosmadi). "
    "B'yi A diye yazmak yapilabilir isi gereksiz yere gomerdi"
)

_IHMAL: tuple[str, ...] = (
    "BENZERLIK (bicimsel analoji) kovasi BOS: bu turda olculen yok; bos "
    "kalmasi bir ihmal degil bir OLCUM SONUCUDUR ve sayisi (0) testte "
    "kilitlidir",
    "KENARI GUCLENDIRECEK UCUNCU BIR TUKETICI YAZILMADI (spec 8): yeni "
    "tuketici de ayni HF soyundan gelirdi, sema cipasi sema cipasi kalirdi -- "
    "daireyi buyutur, kirmaz",
    "IKINCI BIR BORU YOK: tabani olculen kenarin (`hf -> md`) tek borusu "
    "var -- 22.09.2026'da ikinci bir GERCEK_KOPRU (`md -> tdse`), "
    "24.09.2026'da ucuncusu (`tb -> md`, tabani olculmedi) yazildi ama "
    "ikisi de AYRI birer kenardir, bu kenarin ikinci borusu DEGILDIR; iki-ayarli "
    "capraz kontrol (Modul 6) burada kurulamaz ve karisik boru yasagi bir "
    "ikinci boruyu HF'in oteki tuketicisinden odunc almayi da yasaklar",
    _YONTEM_HEDEF_A,
    _YONTEM_HEDEF_B,
)

_YOKLUK_HARITASI_BEYANI = (
    "YOKLUK HARITASI: yedi motorun 21 cifti BES kovada -- GERCEK_KOPRU / "
    "TAUTOLOJI / YAPISAL_YOK / BENZERLIK / OLCULMEDI -- ve sayim bu kosumda "
    "tablodan CANLI yapilip `KOVA_SAYILARI` ile karsilastirilir. Her "
    "yapisal yoklugun MERTEBESI tablodadir: `grav` x 6 icin F_g/F_e = "
    "8.093550856608016e-37 (iki proton; elektron-proton 4.407885560760525e-40) "
    "ve protonun Schwarzschild yaricapina inmek icin a_0 kenarli kupte "
    "9.665562e+129 nokta (kenar 1 angstrom olsaydi 6.522644e+130 -- span bir "
    "dugmedir); `hf -> tdse` icin O 1s izgarasi 1600^3 = 4.096e+09 nokta = "
    "61.04 GiB; `em -> tdse` icin yazilacak bir mertebe YOK, eksik olan sayi "
    "degil KARSILIKTIR. OLCULMEDI AYRI BIR KOVADIR: IKI cift (tdse-tb, "
    "spin-em) orada durur. "
    "29.08.2026'da SEKIZDILER ve sebep 'hicbir sayi gelmedi'ydi; "
    "22.09.2026'da spec 7.2'nin on olcumu kosuldu, sekizinin sekizinde de "
    "sayi GELDI ve `tdse-md` kenari YAZILARAK GERCEK_KOPRU'ya gecti; "
    "24.09.2026'da `md-tb` kenari da YAZILDI ve ayni yolu izledi (tabani "
    "olculmedi); 27.09.2026'da `hf-spin` kenar YAZILMADAN, HUKUM verilerek "
    "TAUTOLOJI'ye gecti (buyukluk sadelesir, isaret h = 0'da bir ayar "
    "etiketi); 28.09.2026'da `tdse-spin`, `md-em`, `tb-em` kenarsiz, HUKUMLE TAUTOLOJI'ye gecti. "
    "Kalan ikinin NEDENI OLCULMUSTUR: `tdse-tb`de TANIM SECIMINE ve KOSTURULMAMIS sevk-yolu "
    "sinavina, `spin-em`de EKSEN ve Lambda BEYANINA bagli. 'olculdu, sifir cikti' ile ayni "
    "hucreye YAZILAMAZ -- bu bir yokluk beyanidir, bir sonuc degil, ve spec "
    "7.2 IKI CIFTTE ACIK kalir"
)

#: SEKLIN ADI (`hf - md - tb` UCGENI arti `md - tdse` KUYRUGU) bir testle
#: `olculen_kenarlar()`dan TURETILEN sekle karsi sinanir
#: (`tests/bridge/test_bridge_senaryo.py`); kenar sayisi, dereceler ve sifir
#: bagli cift sayisi bu metinde YAZILMAZ, `canli_satir` onlari her kosumda
#: taramadan basar. Gerekce olculdu (spec md-tb 5.3): 13 literal duzeltilince
#: testler yesil oluyordu ama bu beyanin sayilari BAYAT kaliyordu, cunku
#: testler cumleyi yalniz metin olarak kilitliyordu.
_YAPI_BEYANI = (
    "YAPININ SEKLI: bir MERDIVEN DEGIL -- `hf - md - tb` UCGENI arti "
    "`md - tdse` KUYRUGU. Capraz-motor import kenarlari: "
    "`engine/md/forcefields.py:13` -> `qlab.engine.hf.scf`, "
    "`engine/tb/downfold.py:130-132` -> "
    "`qlab.engine.hf.{basis, integrals, multistart}`, 22.09.2026'da YAZILAN "
    "`engine/tdse/from_md.py` -> `qlab.engine.md.state` ve 24.09.2026'da "
    "YAZILAN `engine/md/tb_kuvvet.py` -> `qlab.engine.tb.{mermin, radyal}` ile "
    "`engine/md/factory.py` -> `qlab.engine.tb.radyal` (yasa nesnesini "
    "senaryodan kurar); "
    "kenar sayisi, dereceler ve sifir bagli cift sayisi bu kosumda taranir ve "
    "CANLI satirdadir. Modul 7 kapandiginda yapi merkezi HF olan iki cubuklu "
    "bir YILDIZDI, 22.09.2026'da DORT DUGUMLU BIR ZINCIR oldu (`tb - hf - md "
    "- tdse`), 24.09.2026'da zincirin `tb` ve `md` dugumleri baglandi ve "
    "UCGEN kapandi. MERDIVEN HUKMU YINE DE AYAKTA, DAYANAGI YINE DEGISTI: "
    "ucgen IMPORT grafiginde bir dongudur, VERI AKISINDA degil -- `tb -> hf` "
    "kenarinda veri akmaz (`model_from_downfold` src'de hic cagrilmiyor), "
    "yeni kenar SECILMIS bir yasa tasir ve downfold'dan yasa almak yasaktir; "
    "yani hf -> tb -> md diye bir veri yolu YOKTUR. Uc dugum de ayni mutlak "
    "eksende (hartree/bohr) ve tek olcekteki bir ucgen bir OLCEK merdiveni "
    "degildir; kuyrugun ucu `tdse` hala OLCEKSIZDIR. `qlab.units`e dokunan "
    "motor kumesi `ast` ile hala {hf, md, tb} ve kenar dugumlerinin AYNISI "
    "DEGIL, ALT KUMESI: `tdse` bagli ama olceksiz; Modul 7'nin olctugu "
    "ozdeslik bir yasa degil, eski topolojinin bir RASTLANTISIYMIS. DURUST "
    "KAYIT: biri ileride downfold'u MD yoluna baglarsa ortaya cikan sey "
    "merdiven degil TOTOLOJI olur (on olcumde TB payi %0.000); kenarin M10 kilitleri "
    "(`tests/md/test_tb_kuvvet.py`, `tests/tb/test_mermin.py`) o ani yakalar. "
    "KENAR SINIFI (26.09.2026): tarayici her kenari CALISMA ZAMANI ya da "
    "YALNIZ TIP diye olcer ve canli satira basar; `tdse -> md` YALNIZ TIPtir "
    "(veri akisi ordek tiplemeyle, import satiri silinince modul bit bit ayni "
    "calisiyor -- olculdu) ve kilit 2 onu yine kod kenari sayar. GERCEK_KOPRU "
    "kenarlarinin sinifi beyanlidir ve kilit 5 onu olcume karsi sinar; kilit "
    "2'nin calisma zamani kenari isteyip istemeyecegi Osman'in karari: "
    "HAYIR, 08.10.2026 (evet denilseydi `tdse -> md` bugunku kaynakla "
    "GERCEK_KOPRU'da kalamazdi)"
)

_KENAR_TABANI_BEYANI = (
    "KENAR TABANI IKI SAYIDIR VE IKISI AYRI ETIKET TASIR. A1, fizik-kapali "
    "limit (`a -> 0` merdiveni; `a = 0.03` bir sinir degil bir TEPE, disinda "
    "artik duser), ARITMETIGI olcer. A2, anharmoniklik-eslestirilmis prob "
    "(`a = 0.9137`, borunun kendi olctugu A^2-egimine eslestirildi: HF "
    "-0.0100248, ikiz -0.0100230), HATA CUBUGUNU olcer. Sevk edilen "
    "pencerede olculdu (tests/bridge/test_kenar_tabani.py): A1 3.3986707e-05 "
    "cm^-1, S/G 1.6128e+08, SAGLAM; A2 1.2025992e-02 cm^-1, S/G 4.5578e+05, "
    "SAGLAM; gercek hata 1.4472809e-02 cm^-1 (HF borusu canli, `omega_ref` "
    "Richardson `h` cifti (0.002, 0.004); cift bir dugmedir, dort haneye "
    "yazilabilir) -> A1 gercegi 425.84 KAT IYIMSER gosterir, A2 gercegin "
    "0.8309'udur. YALNIZ A1 YAZILSAYDI PANEL YALAN SOYLERDI. Ve iki sayi "
    "iki AYRI dugmeye tepki verir (olculdu, Task 5): N = 200 -> 400'de A1 "
    "15.18 kat duser, A2 1.0008 kat -- A1 `dt`nin aritmetigini, A2 yuzeyin "
    "fizigini okur. Taban SEKIZ anlamli haneden fazlasina yazilamaz: "
    "`ulp(TAM_CM)/taban = 2.68e-08`, ayni taban cm^-1'de mi a.u.'da mi "
    "cikarildigina gore bagil 1.38e-08 oynar. Bu kosumun iki sayisi canli "
    "satirdadir"
)

_DURUST_OLUMSUZLAR = (
    "DURUST OLUMSUZLAR -- BEYAN EDILDI, DUZELTILMEDI, PANEL TASIR: (1) spec "
    "7.2 HALA ACIK ama DARALDI: 22.09.2026'da sekiz ciftin sekizi de "
    "olculdu ve `tdse-md` kenari yazildi, 24.09.2026'da `md-tb` kenari da "
    "yazildi (kenar tabani OLCULMEDI, DOGA CIPASI YOK), 27.09.2026'da "
    "`hf-spin`, 28.09.2026'da `tdse-spin`, `md-em`, `tb-em` kenarsiz, hukumle "
    "TAUTOLOJI'ye gecti; kalan IKI cift OLCULMEDI kovasinda kalir cunku hukum bir TANIM "
    "SECIMINE ve kosturulmamis sevk-yolu sinavina (`tdse-tb`) ve bir EKSEN ve Lambda BEYANINA "
    "(`spin-em`) bagli, EM ciftlerinin konuma bagli ortam yolu ise TUR B, hukmu YOK; (2) `md -> "
    "spin` ve `hf -> em` tautoloji hukumleri 29.08.2026 on olcumunden "
    "DEVRALINDI ve bu gorevde yeniden olculmedi (damgalari tabloda "
    "yazilidir); (3) makineler arasi tekrarlanabilirlik OLCULMEDI -- bit bit "
    "sayilar (22 kabugun 10'u, 2.276e-15) bu makinenin ve numpy 2.5.1'in "
    "kaydidir; (4) `omega_B`nin 6-31G degeri (spec 4: 4646.16252) bu depoda "
    "suite ALINMADI (HF YOL B'nin 6-31G kosumu ~5.4 kat pahali), suit yalniz "
    "statik `omega_A`(6-31G)'yi kilitler; (5) izotropik hicbir sey yok bu "
    "modulde ama bir ikinci boru da yok -- tabani olculen tek kenarin "
    "(`hf -> md`) capraz kontrolu "
    "kurulamaz, hukum tek borunun tabanindan okunur"
)

_DUSEN_ADAYLAR_BEYANI = (
    "DUSEN ADAYLAR, TAUTOLOJI OLARAK, PENCERESIYLE (Task 4'te iki, 7a26ed6; "
    "sayilar `KOVA_TABLOSU` gerekcelerinde ve tests/bridge/"
    "test_dusen_adaylar.py'de motorla dogrulanir; (4)'te TDSE ve (5)-(6) kosmaz): (1) `hf -> tb` "
    "-- `t_m` halka kabugu uzerinde `np.mean` olarak TANIMLANIP `==` ile "
    "geri okunuyor, 6/6 sistemde; ozdeslik 15 hane yalniz N = 6'da ((6, 2.6) "
    "bagil 2.276e-15), N >= 8'de ~1e-8, yani SISTEME BAGLI; ayni elemanlar "
    "ters sirayla toplaninca `==` 22 kabugun 10'unda dusuyor (1 ulp) -- 'on "
    "alti hane' toplama sirasinin kaydiydi; `model_from_downfold` `src/`de 0 "
    "cagri. Pencere: (N, d) altili, STO-3G, seeds = 12. (2) `tb -> spin` -- "
    "bes `J`de (U/|t| = 2.033100 ... 13.630622) `m` serisi bit bit "
    "ayni, `E == J * E(1)` bit bit, kabul orani `==` 0.187216796875 = "
    "287565/1536000; kok neden `metropolis.py:189-191` oran sadelesmesi; `U` "
    "1000 kat buyutulse de kenar ACILMIYOR, o yuzden TEK SATIR: `U/|t| = "
    "2.033100` ve `z` (spin kodda 4, TB halkasi 2) o satirin ICINDE, bu "
    "boruda olculemez. Pencere: Ising L = 32, h = 0, tohum 20260822, burn "
    "500, sweep 1500. Ikisi de sessizce dusurulmedi. 27.09.2026'DA UCUNCU "
    "ADAY ayni dosyada: (3) `hf -> spin` -- H2 J(R = 5/6/7) protokol A'da "
    "`J = -1` ile bit bit ayni (buyukluk sadelesir; bit bit bir OLCUM sonucu, "
    "ozdeslik degil. Pencere: STO-3G, `search_uhf` seeds = 12, Ising L = 32, "
    "h = 0, tohum 20260822, T = 2.269185 * |J|, 2000 sweep burn-in DAHIL, Neel "
    "baslangic), isaret h = 0 cift L'de "
    "bir AYAR ETIKETI (AFM kademeli seri == FM tekbicim seri, bit bit); o da "
    "TAUTOLOJI, sessizce dusurulmedi. 28.09.2026'DA UC ADAY DAHA, kenarsiz: (4) `tdse -> spin` "
    "-- ayni kok neden, dort TDSE J'si ayni pencerede bit bit (olcum, ozdeslik degil; testte "
    "LITERAL; 300 K'de R = 3/4/5 donar ya da doyar); (5) `md -> em`, (6) `tb -> em` -- "
    "tekbicim ortamda korunumun sifiri ya da bir eps kosumunun etiketi (pencereleri gerekcelerde)"
)

_CIPA_SINIFI_BEYANI = (
    "CIPA SINIFI: `hf -> md` kenari bir SEMA CIPASIDIR, fizik cipasi DEGIL. "
    "Iki tuketici de (statik ikinci turev `omega_A` ve MD periyodu `omega_B`) "
    "AYNI HF cozumunun soyundan gelir; uyum dogayi degil IKI UYGULAMAYI olcer "
    "-- bu yazilmasaydi Modul 3/4/5/6'nin hatasi yedinci kez tekrarlanirdi. "
    "`omega_A` SINAVDAN CIKARILDI: kuvvet x1.001 mutantinda `omega_A` BIT BIT "
    "AYNI kalir (statik yol gradyani hic kullanmaz; testte `==`), `omega_B` "
    "ise +4.955181e-04 kayar (5481.229456 -> 5483.969419 cm^-1; karekok "
    "yasasi sqrt(1.001) - 1 = 4.99875e-04) -- sinav yalniz `omega_B` "
    "uzerinden kosar. RAFTA: `omega_B/omega_A - 1` orani da bir cipa DEGIL, "
    "`mu`ye yapisal olarak kordur (asagida)"
)

_MU_KORLUGU_BEYANI = (
    "MEVCUT CIPANIN KUTLE KORLUGU (tests/md/test_aimd.py, Task 3, b36cdd6): "
    "kilitlenen `omega_B/omega_A - 1` kutle x4'te -4.360844408468623e-06 -> "
    "-4.360844408468623e-06, hex -0x1.24a6bc8940000p-18, BIT BIT ayni -- "
    "`mu` pay ve paydada sadelesir. Tasiyici kanit oranin esitligi degil "
    "MEKANIZMASI: `omega_A(4mu) == omega_A(mu)/2` ve `omega_B(4mu) == "
    "omega_B(mu)/2`, ikisi de tam esitlikle. Cipanin 1e-4 esiginin marji "
    "22.93 kat (= esik/|deger|; planin '600'u bir KATEGORI HATASIYDI, "
    "esik/|oynama| hesaplanmisti). O cipa transferin SAYISALLIGINI dogrular, "
    "kutle borusunu DEGIL; kutle borusu artik kendi MUTLAK sinavina sahiptir "
    "(`AMU_TO_ELECTRON_MASS = 1.0` mutanti orada olur, oran assert'inde "
    "degil)"
)

_KURAM_SECICI_BEYANI = (
    "KURAM SECICI: dondurulan yuzeyin `K`si HF/STO-3G'den okunur ve baz seti "
    "HF'in kuram duzeyini secer; kenar ona karsi YANILABILIR. 6-31G'de "
    "`omega_A` 4646.19 cm^-1 (k = 0.411661 Ha/bohr^2) -- ama YALNIZ 6-31G'nin "
    "KENDI `r_e`sinde (1.3794239789 bohr, Brent); sevk edilen pencerede "
    "(`r_e` STO-3G'nin, 1.3459193602) ayni yuzey 4921.26 cm^-1 verir "
    "(k = 0.461848). Spec 4'un 4646.18'i yanlis degildi, PENCERESI (`r_e`) "
    "yazilmamisti: ayni sayi iki pencerede 275 cm^-1 ayrisiyor. Iki deger de "
    "testte ucuz (statik, 3 SCF) kilitlidir; `omega_B`nin 6-31G kosumu "
    "suite alinmadi"
)

_PENCERE_BEYANI = (
    "BU MODULUN PENCERESI: HF/STO-3G `K = 0.5729363` Ha/bohr^2 (`h = 0.002`; "
    "`h` bir dugmedir: 0.008 -> 0.57296588, 0.0005 -> 0.57293446), `r_e = "
    "1.3459193602` bohr (on haneye yazilamaz, yontem penceresi ~5e-08), "
    "`mu = m(1H)/2` boru hattindan; YOL B `dt = (2*pi/omega_A)/N`, uc genlikte "
    "A^2'de dogrusal uydurma, velocity-Verlet dispersiyon tersi. A1 merdiveni "
    "(0.03, 0.02, 0.01, 0.003, 0.001, 1e-4, 0.0) CAGIRANA KAPALI, A2 ikizi "
    "senaryo dugmesidir. Taban `dt`ye DORDUNCU mertebeden bagli (N = 50 -> "
    "1600: 4.159394e-03 -> 8.826032e-09 cm^-1, ardisik oranlar 9.41 / 13.72 / "
    "14.89 / 15.45 / 15.86 -> 16; kapali form artik/omega = (omega*dt)^4/144, "
    "olculen/ongoru 0.9750 @ N = 1600, N = 3200'de donuyor). Kutle x4 IKI "
    "AYRI SAYIDIR: `dt` yeniden turetilirse carpan TAM 0.5, `dt` "
    "dondurulursa taban 28.67 kat KUCULUR. Kuvvet x1.001'de taban 85095.49 "
    "kat buyur. Pencere adlandirilmadan sayi yazilmaz"
)


def bridge_declaration() -> str:
    """PANELIN ZORUNLU BEYANLARININ TEK KAYNAGI, tek metin halinde.

    `panel_metinleri()` ayni sabitleri panelin AYRI satirlarina dagitir; bu
    fonksiyon onlari tek bir dizge isteyen cagiranlar icin birlestirir. IKI
    YOL DA AYNI SABITLERI OKUR, metnin ikinci bir kopyasi YOKTUR ve
    `tests/bridge/test_bridge_senaryo.py` bunu hem tam cumle hem PARCA
    duzeyinde kilitler. Canli sayilar burada YOKTUR; onlar `canli_satir()`da.
    """
    return " ".join(
        (_YONTEM,)
        + _YAKLASIMLAR
        + _IHMAL
        + (
            _KENAR_TABANI_BEYANI,
            _DURUST_OLUMSUZLAR,
            _DUSEN_ADAYLAR_BEYANI,
            _CIPA_SINIFI_BEYANI,
            _MU_KORLUGU_BEYANI,
            _KURAM_SECICI_BEYANI,
            _PENCERE_BEYANI,
            _YOKLUK_HARITASI_BEYANI,
            _YAPI_BEYANI,
        )
    )


def canli_satir(
    *,
    kova_sayim: dict[str, int],
    kenarlar: frozenset[tuple[str, str]] | set[tuple[str, str]],
    eksen: dict[str, int],
    a1: tuple[float, float, float, str, str],
    a2: tuple[float, float, float, str, str],
    gercek_hata: float,
    tutarsizliklar: tuple[str, ...],
    taranamayan: tuple[str, ...],
    siniflar: dict[tuple[str, str], str],
) -> str:
    """Bu kosumun CANLI satiri -- yalniz sayi ve etiket, beyan cumlesi yok.

    `a1`/`a2` `(taban, sinyal, oran, hukum, boru)` demetleridir: bu modul
    `observable.bridge_obs.EdgeFloor`u goremez (katman sozlesmesi), ceviriyi
    `viz/bridge_frames.py` yapar. Tabanlar SEKIZ anlamli haneyle basilir.
    `taranamayan N` sifirken de basilir (olculmus sifir); yollarin adlari
    `tutarsizliklar`da `TARANAMAYAN: ...` olarak gelir. `kenar siniflari`
    butun kenarlari (TAUTOLOJI'deki `tb -> hf` dahil) taramadan basar.
    """
    sayim = " / ".join(f"{k} {kova_sayim.get(k, 0)}" for k in KOVALAR)
    kenar_metni = ", ".join(f"{a} -> {b}" for a, b in sorted(kenarlar)) or "YOK"
    sinif_metni = ", ".join(f"{a} -> {b} {s}" for (a, b), s in sorted(siniflar.items())) or "YOK"
    # YAPININ SAYILARI BURADA, `_YAPI_BEYANI`nda DEGIL: taramadan her kosumda
    # turetilir ve bayatlayamaz (spec md-tb 5.3).
    yonsuz = {tuple(sorted(k)) for k in kenarlar}
    derece: dict[str, int] = {}
    for cift in yonsuz:
        for m in cift:
            derece[m] = derece.get(m, 0) + 1
    derece_metni = " ".join(f"{m}={n}" for m, n in sorted(derece.items())) or "YOK"
    toplam_cift = len(MODULLER) * (len(MODULLER) - 1) // 2
    sifir_bag = toplam_cift - len(yonsuz)
    eksen_metni = " ".join(f"{ad}={n}" for ad, n in sorted(eksen.items()))
    eksenli = ", ".join(sorted(ad for ad, n in eksen.items() if n > 0))

    def taban(etiket: str, t: tuple[float, float, float, str, str]) -> str:
        deger, sinyal, oran, hukum, boru = t
        return (
            f"{etiket}: taban {deger:.8g} cm^-1, sinyal {sinyal:.8g}, "
            f"S/G {oran:.4e}, hukum {hukum} [{boru}]"
        )

    hukum = (
        "TABLO KODLA TUTARLI"
        if not tutarsizliklar
        else "TABLO KODLA CELISIYOR: " + "; ".join(tutarsizliklar)
    )
    return (
        f"[CANLI -- bu kosum] kova sayimi {sayim} (toplam "
        f"{sum(kova_sayim.values())}); kod kenari {len(kenarlar)}: "
        f"{kenar_metni}; dereceler {derece_metni}, sifir bagli cift "
        f"{sifir_bag}/{toplam_cift}; `qlab.units` ekseni {eksen_metni} -> {{{eksenli}}}; "
        f"taranamayan {len(taranamayan)}; kenar siniflari {sinif_metni}; "
        f"KENAR TABANI IKI SAYI -- {taban('A1 (aritmetik)', a1)} | "
        f"{taban('A2 (hata-cubugu)', a2)}; gercek hata {gercek_hata:.8g} cm^-1 "
        f"-> gercek/A1 = {gercek_hata / a1[0]:.6g}, A2/gercek = "
        f"{a2[0] / gercek_hata:.6g}; {hukum}."
    )


def panel_metinleri(
    *,
    ad: str,
    a_ikiz: float,
    n_bolme: int,
    periyot_sayisi: int,
    genlikler: tuple[float, ...],
    kova_sayim: dict[str, int],
    kenarlar: frozenset[tuple[str, str]] | set[tuple[str, str]],
    eksen: dict[str, int],
    a1: tuple[float, float, float, str, str],
    a2: tuple[float, float, float, str, str],
    gercek_hata: float,
    tutarsizliklar: tuple[str, ...],
    taranamayan: tuple[str, ...],
    siniflar: dict[tuple[str, str], str],
) -> dict[str, object]:
    """`Provenance` alanlarinin METNI -- bu modul `Provenance`i kuramaz.

    Katman sozlesmesi `bridge_map`i `provenance`in ALTINA koyar; nesneyi
    `viz/bridge_frames.py::provenance_for` kurar, metin burada kalir. Sayilar
    disaridan gelir, hicbiri burada yeniden hesaplanmaz.

    `converged` OLCULEBILIR bir seye baglidir ve bu modulde o sey su ikisidir:
    (1) kova tablosu bu kosumda taranan kodla TUTARLI mi, (2) kenarin IKI
    tabani da okunabildi mi (`SAGLAM`). Bayat bir yokluk beyani bir yalandir
    ve yalanin yaninda tik basilmaz; `CURUK`/`DEJENERE` bir taban da tiki
    dusurur.
    """
    tutarli = not tutarsizliklar
    okunabildi = a1[3] == "SAGLAM" and a2[3] == "SAGLAM"
    pencere = (
        f"BU KOSUMUN PENCERESI: N = {n_bolme}, {periyot_sayisi} periyot, "
        f"genlikler {tuple(genlikler)}, A2 ikizi a = {a_ikiz:g}, "
        f"h = 0.002 (sabit), SCF = 0. " + _PENCERE_BEYANI
    )
    canli = canli_satir(
        kova_sayim=kova_sayim, kenarlar=kenarlar, eksen=eksen, a1=a1, a2=a2,
        gercek_hata=gercek_hata, tutarsizliklar=tutarsizliklar,
        taranamayan=taranamayan, siniflar=siniflar,
    )
    return dict(
        method=f"Moduller arasi yokluk haritasi ({ad}); " + _YONTEM,
        approximations=_YAKLASIMLAR
        + (
            f"sonlu pencere: bu kosum YOL B'yi N = {n_bolme}, "
            f"{periyot_sayisi} periyot ve {len(genlikler)} genlikle kosar; "
            "taban bu pencerenin tabanidir, sonsuz ince adimin degil",
        ),
        neglected=_IHMAL,
        expected_error=_KENAR_TABANI_BEYANI + " " + _DURUST_OLUMSUZLAR,
        converged=tutarli and okunabildi,
        circularity=_DUSEN_ADAYLAR_BEYANI,
        anchor_class=_CIPA_SINIFI_BEYANI + " " + _MU_KORLUGU_BEYANI,
        theory_selector=_KURAM_SECICI_BEYANI,
        window=pencere,
        convergence_diagnosis=(
            canli + " [BEYAN] " + _YOKLUK_HARITASI_BEYANI + " " + _YAPI_BEYANI
        ),
        diagnostics={
            "kova": "/".join(str(kova_sayim.get(k, 0)) for k in KOVALAR),
            "kenar": str(len(kenarlar)),
            "eksen": ",".join(sorted(ad for ad, n in eksen.items() if n > 0)),
            "taranamayan": str(len(taranamayan)),
            "A1": float(a1[0]),
            "A2": float(a2[0]),
            "scf": "0",
        },
    )
