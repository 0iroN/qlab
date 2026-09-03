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

BU TUZAGIN BIR ORNEGI DEPODA OLCULDU VE KAPATILDI (30.08.2026):
`scenario/schema.py`nin `MD_FORCE_FIELD_KINDS` demeti hem OLUydu (`src/` ve
`tests/` icinde YALNIZ tanimi bulunuyordu, hicbir yerde okunmuyordu) hem
BAYATti (`hartree_fock` listede YOKtu, oysa `scenarios/h2_aimd.yaml:10` tam
olarak onu kullaniyor ve `engine/md/factory.py:126` onu sevk ediyor). Biri o
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
    dizge sabitleridir (`POTENTIAL_KINDS:18`, `MD_THERMOSTAT_KINDS:79`).

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

# --- Dusen iki adayin OLCULMUS buyuklukleri (Task 4, 02.09.2026) -----------
# Bu iki sayi bir motor kosturularak uretildi ama BURAYA LITERAL yazildi:
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
            "`downfold.py:283`) `src/` icinde hic cagrilmiyor -- `ast` ile "
            "kilitli (Task 4); tuketicileri YALNIZ testlerdir "
            "(`tests/tb/test_downfold_kesme.py` ve "
            "`tests/bridge/test_dusen_adaylar.py`); olan sey `downfold.py`nin "
            "`tb/` klasorunde OTURMASIDIR. TAUTOLOJI YENIDEN OLCULDU (Task 4, "
            "02.09.2026): `t_m` halka kabugu uzerinde bir `np.mean` olarak "
            "TANIMLANIYOR (`downfold.py:256-268`), sonra `p.hoppings[m-1]` "
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
            "degisti, cikti SIFIR kipirdadi. KOK NEDEN "
            "`engine/spin/metropolis.py:164-166` (`ast` ile kilitli): "
            "`coupling` yalniz `delta`yi carpiyor ve kabul olcutune "
            "`delta / temperature` olarak giriyor, yani ciktiya SADECE "
            "`coupling/temperature` ORANI olarak giriyor; `h = 0`'da baska "
            "enerji olcegi YOK. KONTROL: h = 0.1'de ayni iki J'nin ciktisi "
            "9.961e-02 ayrisiyor -- motor ikinci bir olcek varsa J'yi GORUR, "
            "TB o olcegi (h) URETMEZ. BU YUZDEN TEK SATIR, kovasi TAUTOLOJI: "
            "U'yu buyutmek kenari ACMAZ (U/|t| = 4.07 / 8.13 / 16.26 / "
            "101.65 / 1016.55'te de bit bit ayni) ve z uyusmazligi (TB "
            "halkasi 1B z = 2 <-> spin motoru kodda 2B z = 4, "
            "`lattice.py::neighbour_sum` dort `np.roll`) da ACMAZ (J * 2/4 "
            "ile de bit bit ayni) -- BU BORUDA OLCULEMEZ, cunku engel ORAN "
            "SADELESMESIDIR. Ayri satir 'duzeltirsen kenar acilir' diye "
            "okunur ve ACILMIYOR. TUR A -- DUZELTILEMEZ (bu boruda)."
        ),
        buyukluk=_TB_SPIN_M_FARKI,
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
