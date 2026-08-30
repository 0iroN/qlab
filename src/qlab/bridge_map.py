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
karsi sinanir.

BU TUZAGIN BIR ORNEGI ZATEN DEPODA -- OLCULDU (30.08.2026):
`scenario/schema.py:56` satirindaki
`MD_FORCE_FIELD_KINDS = ("harmonic", "morse", "lennard_jones")` hem OLU
(`src/` ve `tests/` icinde YALNIZ tanimi bulunuyor, hicbir yerde okunmuyor)
hem BAYAT (`hartree_fock` listede YOK, oysa `scenarios/h2_aimd.yaml:10`
tam olarak onu kullaniyor ve `engine/md/factory.py:126` onu sevk ediyor).
Biri o demeti bir dogrulama kapisi olarak baglasaydi, depo KENDI
senaryosunu reddederdi. Ayni desen burada TEKRARLANMAYACAK.

TARAMA `ast` ILE YAPILIR, REGEX ILE DEGIL. Yorum satirindaki ya da dizge
icindeki bir import METNI tabloyu kirletir. Bunun bedeli bu depoda
OLCULDU ve bu planin bir sayisini curuttu: `engine/md/__init__.py:10`
dokstringi `qlab.units.amu_to_electron_mass` ADINI yaziyor ama `units`i
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
    dizge sabitleridir (`POTENTIAL_KINDS:11`, `MD_THERMOSTAT_KINDS:57`).

    OLCULMEDI AYRI BIR KOVADIR -- spec 7.2.
    "OLCULDU, SIFIR CIKTI" ile "HIC OLCULMEDI" AYNI HUCREYE YAZILAMAZ.
    Sekiz sifir-kod-kenarli cift (tdse-md, tdse-tb, tdse-spin, hf-spin,
    md-tb, md-em, tb-em, spin-em) IKINCIDEDIR: o soruyu olcen ajan API
    baglantisi kopunca oldu ve HICBIR SAYI GELMEDI. Bu bir YOKLUK
    BEYANIDIR, bir sonuc DEGIL. Onlari YAPISAL_YOK'a katlamak, olculmemis
    bir seyi olculmus gibi gostermek olurdu.
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


def _motor_paketleri(kok: Path) -> list[Path]:
    return sorted(
        p for p in kok.iterdir() if p.is_dir() and not p.name.startswith("__")
    )


def _import_edilen_moduller(dosya: Path, paket_adi: str) -> list[str]:
    """Bir kaynak dosyanin GERCEKTEN import ettigi modul adlari.

    `ast` ile ayristirilir. Yorum satirlari, dokstringler ve dizge
    sabitleri agacta bir `Import`/`ImportFrom` dugumu URETMEZ, bu yuzden
    sayilmazlar -- regex tabanli bir tarama onlari sayardi.
    """
    agac = ast.parse(dosya.read_text(encoding="utf-8"), filename=str(dosya))
    adlar: list[str] = []
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.Import):
            adlar.extend(ad.name for ad in dugum.names)
        elif isinstance(dugum, ast.ImportFrom):
            if dugum.level == 0:
                if dugum.module:
                    adlar.append(dugum.module)
                continue
            # Goreli import: taban, dosyayi ICEREN pakettir (`__package__`).
            taban = ["qlab", "engine", paket_adi]
            for _ in range(dugum.level - 1):
                if taban:
                    taban.pop()
            adlar.append(".".join(taban + ([dugum.module] if dugum.module else [])))
    return adlar


def olculen_kenarlar(kok: Path | str | None = None) -> frozenset[tuple[str, str]]:
    """`src/qlab/engine/*/` altindaki CAPRAZ-MOTOR import kenarlarini olcer.

    Her motor paketinin `.py` dosyalari `ast` ile ayristirilir,
    `qlab.engine.<X>` bicimindeki import'lar cikarilir ve `<X>` dosyanin
    kendi paketinden FARKLIYSA `(paket, X)` kenari uretilir.

    `kok` yalnizca sinama icindir (sentetik bir agac verilebilir);
    varsayilan bu paketin yanindaki gercek `engine/` dizinidir.

    OLCULDU (30.08.2026, b221a9c): tam IKI kenar --
    `md -> hf` (`engine/md/forcefields.py:13`) ve
    `tb -> hf` (`engine/tb/downfold.py:130-132`).
    21 ciftin 19'unda SIFIR bag; iki kenarin ikisi de ayni dugume (hf)
    gidiyor. Yapi bir MERDIVEN degil, merkezi HF olan IKI CUBUKLU BIR
    YILDIZDIR.
    """
    taban = Path(kok) if kok is not None else _MOTOR_KOKU
    kenarlar: set[tuple[str, str]] = set()
    for paket in _motor_paketleri(taban):
        for dosya in sorted(paket.glob("*.py")):
            for modul in _import_edilen_moduller(dosya, paket.name):
                parcalar = modul.split(".")
                if len(parcalar) < 3:
                    continue
                if parcalar[0] != "qlab" or parcalar[1] != "engine":
                    continue
                hedef = parcalar[2]
                if hedef != paket.name:
                    kenarlar.add((paket.name, hedef))
    return frozenset(kenarlar)


# --- Kova tablosu ------------------------------------------------------------

_GRAV_ORANI = 8.093550856608016e-37
_HF_TDSE_NOKTA = 4.096e09

_GRAV_GEREKCE = (
    "Yercekimi atom olcegine BAGLANMAZ ve engel BOYUT DEGIL ENERJIDIR: "
    "iki proton arasinda F_g/F_e = 8.093550856608016e-37. Bu sayi "
    "`engine/grav/factory.py` icindeki sevk edilmis YOKLUK BEYANINDAN "
    "birebir dogrulandi ve CODATA sabitlerinden bagimsiz olarak yeniden "
    "turetildi (ayni bitler). IKI PROTON BIR SECIMDIR: elektron-proton "
    "icin ayni oran 4.407885560760525e-40'tir. TUR A -- DUZELTILEMEZ."
)

_OLCULMEDI_GEREKCE = (
    "SIFIR KOD KENARI, AMA HUKUM YOK. Bu cifti olcen ajan 29.08.2026'da "
    "API baglantisi kopunca oldu ve " + HICBIR_SAYI_GELMEDI + ". "
    "'OLCULDU, SIFIR CIKTI' ile 'HIC OLCULMEDI' AYNI HUCREYE YAZILAMAZ: "
    "bu bir YOKLUK BEYANIDIR, bir sonuc DEGIL. Bu yuzden buyukluk None."
)

KOVA_TABLOSU: tuple[EdgeFact, ...] = (
    # --- GERCEK KOPRU (1) ---------------------------------------------------
    EdgeFact(
        kaynak="hf",
        hedef="md",
        kova=Kova.GERCEK_KOPRU,
        gerekce=(
            "KOD KENARI VAR ve tarandi: `engine/md/forcefields.py:13` -> "
            "`qlab.engine.hf.scf.solve`. `scenarios/h2_aimd.yaml`in GIRDI "
            "ALANLARI ne `k` ne `omega` tasir (dosyanin 3. satirindaki "
            "YORUM `k = 0.5729363` sayisini anar, ama o bir girdi degil "
            "bir not); yuzey SCF cozulerek uretilir. AMA BU BIR SEMA "
            "CIPASIDIR, FIZIK CIPASI DEGIL: iki tuketici de ayni HF "
            "cozumunun soyundan gelir, uyum dogayi degil IKI TUKETICI "
            "UYGULAMASINI olcer. Kenarin TABANI Task 2'de olculur; bu "
            "yuzden burada buyukluk None."
        ),
        buyukluk=None,
    ),
    # --- TAUTOLOJI (4) ------------------------------------------------------
    EdgeFact(
        kaynak="hf",
        hedef="tb",
        kova=Kova.TAUTOLOJI,
        gerekce=(
            "KOD KENARI VAR ve tarandi: `engine/tb/downfold.py:130-132` -> "
            "`qlab.engine.hf.{basis, integrals, multistart}`. Ama BORU "
            "HATTINDA AKIS YOK: `model_from_downfold` (tanim "
            "`downfold.py:283`) `src/` icinde hic cagrilmiyor -- 30.08.2026 "
            "tarandi, tek tuketicisi `tests/tb/test_downfold_kesme.py`; "
            "olan sey `downfold.py`nin `tb/` klasorunde OTURMASIDIR. Ve "
            "`t_m` halka kabugu uzerinde bir `np.mean` olarak TANIMLANIYOR "
            "(`downfold.py:256-268`, dogrulandi), sonra Python `==`'inde "
            "geri okunuyor. Sayilari ve "
            "pencereleri Task 4 sevk eder; " + YENIDEN_URETILMEDI + " "
            "oldugu icin burada buyukluk None."
        ),
        buyukluk=None,
    ),
    EdgeFact(
        kaynak="tb",
        hedef="spin",
        kova=Kova.TAUTOLOJI,
        gerekce=(
            "SIFIR kod kenari, ama HUKUM VAR: `J = 4t^2/U` koprusu BIT BIT "
            "olu. Kok neden `engine/spin/metropolis.py:164-166` "
            "(dogrulandi): `delta` yalniz `coupling` ile carpiliyor ve "
            "kabul olcutune `exp(-delta/temperature)` olarak giriyor, yani "
            "`coupling` ciktiya SADECE `coupling/temperature` ORANI olarak "
            "giriyor; `h = 0`'da baska enerji olcegi yok -- yani spin "
            "motoru `J` hakkinda "
            "YANILAMAZ. `z` uyusmazligi (TB halkasi 1B z=2 <-> spin motoru "
            "kodda 2B z=4) BU SATIRIN ICINDEDIR, ayri satir DEGIL: ayri "
            "satir 'duzeltirsen kenar acilir' diye okunur ve acilmiyor. "
            "Sayilari Task 4 sevk eder; " + YENIDEN_URETILMEDI + " oldugu "
            "icin burada buyukluk None."
        ),
        buyukluk=None,
    ),
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
    # --- OLCULMEDI (8) ------------------------------------------------------
    EdgeFact(
        kaynak="tdse",
        hedef="md",
        kova=Kova.OLCULMEDI,
        gerekce=_OLCULMEDI_GEREKCE,
        buyukluk=None,
    ),
    EdgeFact(
        kaynak="tdse",
        hedef="tb",
        kova=Kova.OLCULMEDI,
        gerekce=_OLCULMEDI_GEREKCE,
        buyukluk=None,
    ),
    EdgeFact(
        kaynak="tdse",
        hedef="spin",
        kova=Kova.OLCULMEDI,
        gerekce=_OLCULMEDI_GEREKCE,
        buyukluk=None,
    ),
    EdgeFact(
        kaynak="hf",
        hedef="spin",
        kova=Kova.OLCULMEDI,
        gerekce=_OLCULMEDI_GEREKCE,
        buyukluk=None,
    ),
    EdgeFact(
        kaynak="md",
        hedef="tb",
        kova=Kova.OLCULMEDI,
        gerekce=_OLCULMEDI_GEREKCE,
        buyukluk=None,
    ),
    EdgeFact(
        kaynak="md",
        hedef="em",
        kova=Kova.OLCULMEDI,
        gerekce=_OLCULMEDI_GEREKCE,
        buyukluk=None,
    ),
    EdgeFact(
        kaynak="tb",
        hedef="em",
        kova=Kova.OLCULMEDI,
        gerekce=_OLCULMEDI_GEREKCE,
        buyukluk=None,
    ),
    EdgeFact(
        kaynak="spin",
        hedef="em",
        kova=Kova.OLCULMEDI,
        gerekce=_OLCULMEDI_GEREKCE,
        buyukluk=None,
    ),
)

#: Kova basina beklenen girdi sayisi. OLCULDU (30.08.2026): 1 / 4 / 8 / 0 / 8,
#: toplam 21 = C(7, 2). Bu esleme bir testle hem tabloya hem `MODULLER`in
#: kombinasyonlarina karsi sinanir.
KOVA_SAYILARI: dict[str, int] = {
    Kova.GERCEK_KOPRU: 1,
    Kova.TAUTOLOJI: 4,
    Kova.YAPISAL_YOK: 8,
    Kova.BENZERLIK: 0,
    Kova.OLCULMEDI: 8,
}
