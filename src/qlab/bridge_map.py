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
    "icin ayni oran 4.407885560760525e-40'tir. Ikinci mertebe (Task 5, "
    "CODATA 2018'den bu gorevde yeniden turetildi, ayni grav beyaniyla "
    "birebir): protonun Schwarzschild yaricapi 2*G*m_p/c^2 = 2.484232e-54 m, "
    "ona inmek icin kenari bir Bohr yaricapi olan bir kupte 9.665562e+129 "
    "nokta gerekir; kenar 1 angstrom secilseydi 6.522644e+130 -- span bir "
    "dugmedir, adlandirilmadan bu sayi yazilamaz. TUR A -- DUZELTILEMEZ."
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
            "UYGULAMASINI olcer. Kenarin TABANI `observable/bridge_obs.py`de "
            "olculur ve IKI SAYIDIR (A1 aritmetik, A2 hata cubugu); tek bir "
            "`buyukluk` alanina sigmaz ve burada literal tutulmaz -- panel "
            "onu her kosumda CANLI okur (Task 5). Bu yuzden burada buyukluk "
            "None: olculmedigi icin degil, iki sayi oldugu icin."
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


# --- `qlab.units` ekseni (Task 5) ------------------------------------------


def olculen_units_ekseni(kok: Path | str | None = None) -> dict[str, int]:
    """Her motor paketinde `qlab.units`i GERCEKTEN import eden dosya sayisi.

    Spec 2.2'nin tasiyici iddiasinin KOD TARAFI: bir merdiven tum basamaklarin
    tek bir mutlak eksene dizilmesini ister, ve mutlak eksene dokunan motor
    kumesi `{hf, md, tb}` -- yani iki capraz-motor kenarinin dugum kumesinin
    AYNISI. Panel bu kumeyi elle yazmaz, buradan okur; `olculen_kenarlar()`
    ile ayni `ast` makinesini kullanir ve `tests/bridge/test_kova_tablosu.py`
    ayni kumeyi BAGIMSIZ bir tarayiciyla da olcer (iki tarayici ayni
    makineye dayansaydi cipa dairesel olurdu).

    OLCULDU (30.08.2026, b221a9c; Task 5'te yeniden): hf 1 (`analysis.py`),
    md 3 (`factory.py`, `forcefields.py`, `thermostats.py`), tb 2
    (`downfold.py`, `factory.py`), tdse/spin/em/grav 0. Duz metin taramasi
    md icin 4 sayar (`engine/md/__init__.py:10` dokstringi adi yaziyor ama
    import etmiyor); bu fonksiyon 3 sayar, cunku bir dokstring bir
    `ImportFrom` dugumu degildir.
    """
    taban = Path(kok) if kok is not None else _MOTOR_KOKU
    sayim: dict[str, int] = {}
    for paket in _motor_paketleri(taban):
        dosya_sayisi = 0
        for dosya in sorted(paket.glob("*.py")):
            adlar = _import_edilen_moduller(dosya, paket.name)
            if any(ad.split(".")[:2] == ["qlab", "units"] for ad in adlar):
                dosya_sayisi += 1
        sayim[paket.name] = dosya_sayisi
    return sayim


def tablo_tutarsizliklari(
    kenarlar: frozenset[tuple[str, str]] | set[tuple[str, str]],
    eksen: dict[str, int],
) -> tuple[str, ...]:
    """Kova tablosunun KODLA CELISTIGI noktalar -- bos demet tutarlilik demek.

    Panelin `converged` hukmu buna baglidir: bayat bir yokluk beyani bir
    yalandir ve bir yalanin yaninda tik basilmaz. Dort kilit:

      1. taranan her kod kenari tabloda olmali ve YAPISAL_YOK ya da
         OLCULMEDI kovasinda OLAMAZ,
      2. GERCEK_KOPRU kovasindaki her ciftin bir kod kenari OLMALI,
      3. kova basina sayim `KOVA_SAYILARI` ile ayni olmali,
      4. `qlab.units` ekseni (dosya sayisi > 0 olan motorlar) iki kenarin
         dugum kumesiyle AYNI olmali (spec 2.2).

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
    if eksenli != dugumler:
        sorunlar.append(
            f"units ekseni {sorted(eksenli)} != kenar dugumleri {sorted(dugumler)}"
        )
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
    "Moduller arasi YOKLUK HARITASI ve tek gercek kenarin taban olcumu; "
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
    "sayilir; calisma zamani cagri grafigi, dinamik yukleme ve dizge icinde "
    "anilan modul adlari SAYILMAZ -- dizgedeki ad sayilsaydi tablo bir yanlis "
    "pozitifle bozulurdu (olculdu, sentetik agac) ve `md`nin `units` sayisi "
    "3 yerine 4 cikardi (olculdu, gercek agac)",
    "KENAR TABANI ICIN HF YUZEYI DONDURULDU: SCF cozumu yerine ayni `K` ve "
    "`r_e`li bir Morse ikizi (A2) ya da onun `a -> 0` cebirsel limiti (A1) "
    "kosturulur; boru `test_aimd.py`nin YOL B'sinin AYNISIDIR ve bu bir "
    "iddia degil bir olcumdur (HF alaniyla cagrildiginda cipanin sayisini "
    "bit bit veriyor, testte kilitli)",
    "DUSEN IKI ADAYIN SAYILARI TASINIYOR, YENIDEN KOSTURULMUYOR: `hf -> tb` "
    "ve `tb -> spin` sayilari `KOVA_TABLOSU` gerekcelerinde literal olarak "
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
    "`coupling/temperature` orani olarak girer (`metropolis.py:164-166`) ve "
    "`U`yu 1000 kat buyutmek de `z`yi uyusturmak da kenari acmaz. Ikisi de "
    "bir tolerans ya da izgara ayariyla KAPANMAZ; A'yi B diye yazmak "
    "'sonraki gorevde acilir' yalanini uretirdi"
)

_YONTEM_HEDEF_B = (
    "YONTEM/HEDEF UYUSMAZLIGI TUR B -- ERTELENDI, engel yok, spec 8 kapsam "
    "disi: (1) `hf -> tdse` icin 3B TDSE govdesi (O 1s icin 1600^3 = "
    "4.096e+09 nokta, tek complex128 dizi 61.04 GiB = 65.536 GB; span ve "
    "cozunurluk birer dugme); (2) `hf -> em` icin kirilma indisi ureteci. "
    "B'yi A diye yazmak yapilabilir isi gereksiz yere gomerdi"
)

_IHMAL: tuple[str, ...] = (
    "BENZERLIK (bicimsel analoji) kovasi BOS: bu turda olculen yok; bos "
    "kalmasi bir ihmal degil bir OLCUM SONUCUDUR ve sayisi (0) testte "
    "kilitlidir",
    "KENARI GUCLENDIRECEK UCUNCU BIR TUKETICI YAZILMADI (spec 8): yeni "
    "tuketici de ayni HF soyundan gelirdi, sema cipasi sema cipasi kalirdi -- "
    "daireyi buyutur, kirmaz",
    "IKINCI BIR BORU YOK: tek gercek kenarin tek borusu var; iki-ayarli "
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
    "degil KARSILIKTIR. OLCULMEDI AYRI BIR KOVADIR: sekiz cift (tdse-md, "
    "tdse-tb, tdse-spin, hf-spin, md-tb, md-em, tb-em, spin-em) orada durur, "
    "cunku 29.08.2026 on olcumunde o soruyu olcen ajan API baglantisi "
    "kopunca oldu ve hicbir sayi gelmedi; 'olculdu, sifir cikti' ile ayni "
    "hucreye YAZILAMAZ -- bu bir yokluk beyanidir, bir sonuc degil, ve spec "
    "7.2 ACIK kalir"
)

_YAPI_BEYANI = (
    "YAPININ SEKLI: bir MERDIVEN DEGIL, merkezi HF olan IKI CUBUKLU BIR "
    "YILDIZ. Capraz-motor import kenari tam IKI -- `engine/md/forcefields.py:13` "
    "-> `qlab.engine.hf.scf` ve `engine/tb/downfold.py:130-132` -> "
    "`qlab.engine.hf.{basis, integrals, multistart}` -- 21 ciftin 19'unda "
    "SIFIR bag, iki kenarin ikisi de ayni dugume gidiyor. Ve merdiven "
    "fizikten ONCE TANIMDAN yok: `qlab.units`e dokunan motor kumesi `ast` "
    "ile {hf, md, tb} (hf 1, md 3, tb 2 dosya; tdse/spin/em/grav 0) -- iki "
    "kenarin dugum kumesinin AYNISI; kalan dort motor hem eksensiz hem "
    "bagsiz. Iki kume de bu kosumda CANLI taranir ve canli satirda yazilidir"
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
    "7.2 ACIK: sekiz cift OLCULMEDI kovasinda, hicbir sayi yok; (2) `md -> "
    "spin` ve `hf -> em` tautoloji hukumleri 29.08.2026 on olcumunden "
    "DEVRALINDI ve bu gorevde yeniden olculmedi (damgalari tabloda "
    "yazilidir); (3) makineler arasi tekrarlanabilirlik OLCULMEDI -- bit bit "
    "sayilar (22 kabugun 10'u, 2.276e-15) bu makinenin ve numpy 2.5.1'in "
    "kaydidir; (4) `omega_B`nin 6-31G degeri (spec 4: 4646.16252) bu depoda "
    "suite ALINMADI (HF YOL B'nin 6-31G kosumu ~5.4 kat pahali), suit yalniz "
    "statik `omega_A`(6-31G)'yi kilitler; (5) izotropik hicbir sey yok bu "
    "modulde ama bir ikinci boru da yok -- tek kenarin capraz kontrolu "
    "kurulamaz, hukum tek borunun tabanindan okunur"
)

_DUSEN_ADAYLAR_BEYANI = (
    "DUSEN IKI ADAY, TAUTOLOJI OLARAK, PENCERESIYLE (Task 4, 7a26ed6; "
    "sayilar `KOVA_TABLOSU` gerekcelerinde ve tests/bridge/"
    "test_dusen_adaylar.py'de motor kosturularak dogrulanir): (1) `hf -> tb` "
    "-- `t_m` halka kabugu uzerinde `np.mean` olarak TANIMLANIP `==` ile "
    "geri okunuyor, 6/6 sistemde; ozdeslik 15 hane yalniz N = 6'da ((6, 2.6) "
    "bagil 2.276e-15), N >= 8'de ~1e-8, yani SISTEME BAGLI; ayni elemanlar "
    "ters sirayla toplaninca `==` 22 kabugun 10'unda dusuyor (1 ulp) -- 'on "
    "alti hane' toplama sirasinin kaydiydi; `model_from_downfold` `src/`de 0 "
    "cagri. Pencere: (N, d) altili, STO-3G, seeds = 12. (2) `tb -> spin` -- "
    "bes `J`de (U/|t| = 2.033100 ... 13.630622) `m` serisi bit bit "
    "ayni, `E == J * E(1)` bit bit, kabul orani `==` 0.187216796875 = "
    "287565/1536000; kok neden `metropolis.py:164-166` oran sadelesmesi; `U` "
    "1000 kat buyutulse de kenar ACILMIYOR, o yuzden TEK SATIR: `U/|t| = "
    "2.033100` ve `z` (spin kodda 4, TB halkasi 2) o satirin ICINDE, bu "
    "boruda olculemez. Pencere: Ising L = 32, h = 0, tohum 20260822, burn "
    "500, sweep 1500. Ikisi de sessizce dusurulmedi"
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
) -> str:
    """Bu kosumun CANLI satiri -- yalniz sayi ve etiket, beyan cumlesi yok.

    `a1`/`a2` `(taban, sinyal, oran, hukum, boru)` demetleridir: bu modul
    `observable.bridge_obs.EdgeFloor`u goremez (katman sozlesmesi), ceviriyi
    `viz/bridge_frames.py` yapar. Tabanlar SEKIZ anlamli haneyle basilir.
    """
    sayim = " / ".join(f"{k} {kova_sayim.get(k, 0)}" for k in KOVALAR)
    kenar_metni = ", ".join(f"{a} -> {b}" for a, b in sorted(kenarlar)) or "YOK"
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
        f"{kenar_metni}; `qlab.units` ekseni {eksen_metni} -> {{{eksenli}}}; "
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
            "A1": float(a1[0]),
            "A2": float(a2[0]),
            "scf": "0",
        },
    )
