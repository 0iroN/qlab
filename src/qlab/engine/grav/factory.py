"""Senaryo -> yercekimi kosumu. PANEL BEYANININ TEK KAYNAGI BURASIDIR.

`run.py::_run_grav` bu metinlerin HICBIRINI yeniden yazmaz; yalnizca
`Provenance.as_lines()` ciktisini ve bir SAYI tablosunu basar. Gerekcesi bu
depoda OLCULEREK ogrenildi: ayni beyan blogu bir kez `scf.py` ve
`tb/factory.py`de birebir KOPYALANMISTI ve biri duzeltilip digeri
unutuldugunda panel iki farkli sey iddia ediyordu. Kural Modul 5'te bir kez
daha sertlestirildi ve burada AYNI sertlikte uygulanir: sayim TAM CUMLE
duzeyinde YETMEZ -- Modul 5'te bir mutant 55 karakterlik bir PARCA
kopyalayarak tam-cumle sayimindan KACTI. Kilit bu yuzden iki katmanlidir ve
ikisi de `tests/grav/test_grav_senaryo.py`dedir.

IKI ADIMLI FABRIKA -- VE NEDEN. `build_grav` fizigi kosar (ODE yorungesi ve
sekizin BVP'si), `provenance_for` OLCULMUS sayilari ALIR ve beyani kurar.
Sebep import-linter sozlesmesidir: panelin zorunlu GURULTU TABANI satiri
(spec 6.6) `observable/grav_obs.py::newton_noise_floor`tan gelir ve `engine`
katmani `observable`i IMPORT EDEMEZ (katman: run -> viz -> observable ->
engine -> scenario -> provenance). Desen Modul 4'un `spin/factory.py`si ve
Modul 5'in `em/factory.py`sinin aynisidir: SAYILAR disaridan gelir, METIN
burada kalir. Tabanlar buraya `FloorLine` olarak, duz veri halinde girer --
`NoiseFloor` bir `observable` turudur ve bu dosya onu goremez.

BU DOSYA CIPA KURMAZ. Sevk edilen senaryonun isi bir cipa kurmak degil,
PANELIN BEYANINI gostermektir (Modul 4'un `spin_tarama.yaml`inda ve Modul
5'in `em_tarama.yaml`inda ayni cumle yazili); cipalarin olcumu `tests/grav/`
altindadir ve bu senaryo onlari YENIDEN OLCMEZ -- yeniden okur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qlab.engine.grav.geodesic import Trajectory, run_trajectory
from qlab.engine.grav.metric import GAUGE_NAMES, Theory
from qlab.engine.grav.nbody import FigureEight, solve_figure_eight
from qlab.errors import InvalidInputError
from qlab.provenance import Provenance
from qlab.scenario.schema import GravScenario

#: Genel goreliligin kuram secici noktasi. TEK bir sabit olarak durur ve
#: `tests/grav/test_grav_senaryo.py` onu BURADAN import eder; iki yerde elle
#: yazilsaydi biri degistiginde test ile panel farkli sey soylerdi. Modul
#: 4'un `ALT_SINIR_ISARETI` ve Modul 5'in `HUKUM_VERILMEDI` isaretiyle ayni
#: statude.
GR_NOKTASI: tuple[float, float] = (1.0, 1.0)

#: GR olmayan bir kuram secildiginde panelin bastigi isaret. Ayni statu.
GR_DEGIL_ISARETI = "COZULEN KURAM GENEL GORELILIK DEGIL"


_GRAV_YAKLASIMLAR: tuple[str, ...] = (
    "TEST PARCACIGI: yorungeyi cizen cismin kutlesi metrige GERI TEPMEZ, "
    "alan kaynagi tek basina M'dir; iki-cisim problemi bu modulde COZULMEZ "
    "ve bu bir kapsam tercihi degil YONTEMIN SINIRIDIR",
    "SCHWARZSCHILD AILESI: kaynak KURESEL, DONMEYEN ve YUKSUZ alinir. "
    "Izotropik ayar ayni fizigin baska bir KOORDINAT yazilisidir, baska bir "
    "kaynak degil -- ve tam da bu yuzden iki ayarin ayni sayiyi vermesi bir "
    "ayar-bagimsizligi olcumudur",
    "G = c = 1 BIR BIRIM SECIMIDIR, FIZIK DEGIL (Modul 4'un J = k_B = 1'i ve "
    "Modul 5'in c = eps_0 = mu_0 = 1'i ile AYNI STATUDE): bu modulun hicbir "
    "ciktisi metre, saniye, kilogram ya da yay saniyesi tasimaz",
    "ISIMA SONUMU YOKTUR: yorunge korunumludur, gravitasyonel dalga kaybi "
    "hicbir denklemde gecmez",
    "N-CISIM TARAFI NEWTONYEN'DIR VE METRIGI HIC GORMEZ: `engine/grav/"
    "nbody.py` `Theory`yi import etmez ve kaydeden vekil dokunulan ad "
    "kumesini {masses, positions, velocities} olarak SAYAR",
    "SONLU KARELEME VE SONLU TOLERANS: her sayi bir `quad` cagrisinin ya da "
    "bir `solve_ivp` adiminin icinden okunur; surekli limit ALINMAZ, yalniz "
    "Richardson ile otelenir -- ve disdegerleme optimumdan sonra BOZULUR "
    "(kareleme (3e5, 1e6)'da 10.01 hane verirken (3e6, 1e7)'de 8.98'e "
    "duser), yani daha ince basamak daha iyi cevap DEMEK DEGILDIR",
)

_YONTEM_HEDEF_A = (
    "YONTEM/HEDEF UYUSMAZLIGI, TUR A -- YONTEM HEDEFI KAPSAMIYOR, "
    "DUZELTILEMEZ (baska bir KURAM secmek gerekir): (1) Newtonyen N-cisim "
    "cozucusu perihel kaymasini YANLIS uretmez, HIC uretmez -- kapali form "
    "Newton noktasinda (lam = 0) TAM SIFIRDIR ve bu bir yaklasim hatasi "
    "degil YAPISAL bir yokluktur; (2) jeodezik cozucu TEST PARCACIGI "
    "tasidigi icin geri-tepkiyi (yorungedeki cismin metrige etkisini) "
    "ureten bir denklem TASIMAZ. Ikisi de bir tolerans siki lastirmasiyla ya "
    "da daha ince bir izgarayla KAPANMAZ"
)

_YONTEM_HEDEF_B = (
    "YONTEM/HEDEF UYUSMAZLIGI, TUR B -- HEDEF BU MODULUN DISINDA, "
    "ERTELENDI, ENGEL YOK: Kerr metrigi (donen kaynak), isima sonumu ve "
    "gelgit deformasyonu. Kerr'i ertelemenin gerekcesi maliyet DEGIL: "
    "tum ISCO(a) tablosu tasarim belgesinin on olcumunde 0.27 ms'de, "
    "hicbir ODE kosmadan cikiyor (BU SAYI DA ALINTIDIR, bu depoda "
    "yeniden uretilmedi -- Kerr govdesi sevk edilmedi) -- yani "
    "maliyeti sifir ama CIPA DEGERI DE SIFIR, cunku ucundaki saf sayilar "
    "dahil hepsi kapali formdur. TUR A ILE TUR B KARISTIRILAMAZ: A'yi B "
    "sanmak 'sonraki modulde ekleriz' yalanini uretir, B'yi A sanmak "
    "yapilabilir isi gereksiz yere gomer"
)

_YOKLUK_BEYANI = (
    "MODULLER ARASI BAG -- YOKLUK BEYANI: bu modul atom olcegindeki "
    "modullerle (TDSE, Hartree-Fock, spin, Maxwell) FIZIKSEL OLARAK "
    "BAGLANMAZ, ve engel BOYUT DEGIL ENERJIDIR. OLCULDU (CODATA 2018 / SI "
    "2019, bu depoda): iki proton arasinda F_g/F_e = 8.093550856608016e-37; "
    "yercekimsel 'atom'un (alpha_g = G*m_p^2/(hbar*c) = 5.906149e-39, "
    "E_1 = m_p*c^2*alpha_g^2/2, INDIRGENMIS KUTLE KULLANILMADI -- bu bir "
    "SECIMDIR ve indirgenmis kutleyle sayi TAM YARIYA duser) taban durum "
    "enerjisi 2.621912e-87 J, yani kozmik ardalan foton banyosunun "
    "k_B*T = 3.762931e-23 J'unun 6.967737e-65 KATI (T = 2.72548 K "
    "secildi; 2.725 K yuvarlamasiyla ayni sayi 6.968950e-65 olur -- "
    "efemeris duyarliligiyla AYNI AILEDEN bir tablo secimi). Ve protonun "
    "Schwarzschild yaricapina (2*G*m_p/c^2 = 2.484232e-54 m) inmek, kenari "
    "bir Bohr yaricapi olan bir kupte 9.665562e+129 izgara noktasi ister "
    "-- SPAN BIR DUGMEDIR ve adlandirilmadan bu sayi yazilamaz: kenar "
    "1 angstrom secilseydi 6.522644e+130 cikardi. Kurulabilen tek bag "
    "1/r'nin YAPI ORTAKLIGIDIR ve o da bir KAPI, bir cipa degil"
)

_GRAV_IHMAL: tuple[str, ...] = (
    "KUANTUM YERCEKIMI: bu modulde ne bir dalga fonksiyonu ne bir hbar "
    "vardir; olcek yoklugu OLCULDU ve asagida YOKLUK BEYANI olarak yazili",
    "KOZMOLOJIK SABIT ve genisleyen evren: metrik asimptotik olarak DUZDUR",
    "KOZAI-LIDOV salinimi -- ve reddedilme gerekcesi bir TERCIH DEGIL bir "
    "OLCUMDUR: kapali-form kok ile dinamik yol AYNI MODELI cozmuyor "
    "(3/5 esigi tam N-cismin degil ikili-ortalamali dortkutup "
    "YAKLASIMININ kokudur), gorunen yakinsama iki ters isaretli "
    "sistematigin KARSILIKLI SADELESMESIDIR ve ucuncu hane yordama "
    "bagli; hukum cos^2 i_c = 0.602 +- 0.003 ve |fark| < 1e-4 icin "
    "maliyet 4.5 GUN. BU UC SAYI TASARIM BELGESININ ON OLCUMUNDEN "
    "ALINTIDIR VE BU DEPODA YENIDEN URETILMEDI -- Kozai govdesi hic "
    "sevk edilmedigi icin onlari kosturacak bir kod da yok; alinti "
    "oldugu yazilmasaydi panel kendi olcmedigi bir sayiyi kendi olcumu "
    "gibi gosterirdi",
    "GERCEK EFEMERIS VERISI ve gozlemle karsilastirma: bu modul hicbir "
    "gok cismine ait bir tabloyu okumaz ve hicbir gozlem artigini aciklamaz",
    "N > 3 kaotik sistemler, Lyapunov usleri, 3B yorungeler ve Simo'nun "
    "oteki koreografileri",
    _YONTEM_HEDEF_A,
    _YONTEM_HEDEF_B,
    _YOKLUK_BEYANI,
)

_DAIRESELLIK_43_SANIYE = (
    "(a) BU MODUL MERKUR'UN PERIHEL KAYMASINI ONGORMEZ. 43 yay saniyesi/"
    "yuzyil M'den, a'dan VE e'den BIRE BIR geri okunur -- olculdu (bu "
    "depoda, float64): M 1.01 kat buyutulunce sonuc 1.0099999999999996 kat "
    "buyuyor (bagil sapma 4.40e-16), a 1.01 kat buyutulunce 0.99009900990099 "
    "kat -- yani tam 1/1.01 (1.12e-16) --, ve e 0.20563069 -> 0.3 "
    "degistirilince oran 1.0524351860770593, kapali form "
    "(1 - e0^2)/(1 - e^2) = 1.052435186077059 ile 2.11e-16 icinde. Uc "
    "dugmenin ucu de tam olarak geri okunuyor. VE DEGERIN KENDISI HANGI "
    "TABLONUN SECILDIGINE BAGLIDIR: dort ADI KONMUS secim olculdu -- "
    "(A) (a, e, T) tablodan + Julyen yuzyil 42.980644, (B) T Kepler "
    "III'ten 42.980473, (C) a Kepler III'ten 42.980758, (D) A ile ayni ama "
    "TROPIK yuzyil 42.979726 -- yayilim 0.0010320 yay saniyesi, yani "
    "43'ten olculen farkin (0.019600) yuzde 5.27'si. BU MODULUN OLCTUGU "
    "SEYLER SUNLARDIR: 2*pi*(2 + 2*gamma - beta) katsayisi, "
    "pi*lam^2*(3 + e^2/6) ikinci mertebesi, (1 + gamma) bukulme orani ve "
    "|E|^{3/2}*T/(G^2*m^{5/2}) sekiz sabiti -- dordu de BOYUTSUZDUR ve "
    "hicbir efemeris tablosundan okunmaz"
)

_DAIRESELLIK_SEKIZ = (
    "(b) SEKIZ YORUNGESINDE ARAMADAN SONRA DAIRESELLESEN UC IDDIA VARDIR "
    "VE UCU DE CIPA DEGILDIR: (1) 'ucu de ayni egride T/3 kaymayla "
    "dolasiyor' cumlesi BVP KOSULUNUN TA KENDISIDIR (arama tam olarak "
    "Phi_{T/3}(y0) = P_{(2,0,1)} y0'i sifirlar), (2) tam periyot donusu "
    "Phi_T = (Phi_{T/3})^3 CEBIRSEL SONUCUDUR ve integratoru olcer, fizigi "
    "degil, (3) L = 0 BIR OLCUM DEGILDIR -- simetrik ansatz'ta hiz sifir "
    "verilse bile (`build(0, 0)`) acisal momentum sifir cikar. Arama "
    "bittikten sonra GIRDI OLARAK KALAN TEK SEY SIMETRI SINIFIDIR (Euler "
    "dizilimi + T/3 devri) ve o bir SAYI DEGILDIR. Cipa bu ucunden hicbiri "
    "degil, aramanin BULDUGU T ile hesaplanan boyutsuz sabittir"
)


_CIPA_SINIFI_BEYANI = (
    "FIZIK CIPASI (statik koku YOKTUR, sayi ancak yorunge/jeodezik "
    "cozulerek cikar): perihel katsayisi 2*pi*(2 + 2*gamma - beta), ikinci "
    "mertebe pi*lam^2*(3 + e^2/6), bukulme orani 1 + gamma, ve sekiz "
    "yorungesinin |E|^{3/2}*T/(G^2*m^{5/2}) sabiti. "
    "KAPALI-FORM TUREV (fizik cipasi DEGIL -- hicbirinde yorunge cozulmez, "
    "hepsi bir cebirsel kokun ya da bir NOKTADAKI dogrusallastirmanin "
    "sonucudur): ISCO r/r_s = 3, foton kuresi r/r_s = 1.5, "
    "b_krit/M = 3*sqrt(3), L4/L5 esigi mu_c, apsidal us -1/2. Statik ve "
    "dinamik yollarin orani OLCULDU ve hukum ayni yonde: ISCO'da 40418 kat, "
    "b_krit'te 483 kat -- dinamik yol her ikisinde de daha PAHALI ve daha "
    "AZ hassas. "
    "SEMA CIPASI (COZUCUYU dogrular, FIZIGI DEGIL): Richardson merdiveninin "
    "kendi optimumu ve `t0 = 0` sozlesmesi. "
    "CIPA DEGIL -- RAFTA UC UYE VE UCUNUN SEBEBI AYRI: (1) Kepler 3/2 ussu, "
    "cunku [GM] = L^3*T^-2 onu ZORUNLU kilar ve simulasyon o sayida "
    "YANILAMAZ; (2) <U>/|E| = 2, cunku viryal teoremi her bagli 1/r "
    "yorungesi icin zorunludur; (3) ENERJI KORUNUMU, cunku kuvvet ussu "
    "n = -2, -2.5, -1.3, 0, +3'un HEPSINDE geciyor -- yani bir INTEGRATOR "
    "kapisi, bir fizik kapisi DEGIL, ve deponun kendi olcutuyle bir "
    "TAUTOLOJI. AMA TAUTOLOJI OLMAK MUTASYON KAPSAMASI OLMAMAK DEMEK "
    "DEGILDIR ve bu OLCULDU: enerji korunumu testi bu turda bir mutanti "
    "OLDURDU -- bir cipanin fizik DEGERI ile bir testin AYIRT ETME GUCU "
    "iki ayri buyukluktur"
)

_KURAM_SECICI_BEYANI = (
    "KURAM SECICI YENI BIR KUTUDUR ve bu modulde IKI UYESI VAR. Bir kuram "
    "secici bir YAKLASIM DEGILDIR (yaklasim YAPILAN bir seydir: 'test "
    "parcacigi aldik') ve bir BIRIM SECIMI de degildir (G = c = 1 odur); "
    "HANGI KURAMIN cozuldugunu secer. (1) `Theory(gamma, beta)`: genel "
    "gorelilik gamma = beta = 1, yani lam = 2 + 2*gamma - beta = 3; "
    "HERHANGI BASKA DEGER GENEL GORELILIK DEGILDIR. `lam` BIR ALAN DEGIL "
    "TURETILMIS BIR OZELLIKTIR -- tek kaynak (gamma, beta)dir, cunku iki "
    "alan ayni seyi belirlerse sessiz bir kazanan dogar ve o kazanan "
    "adlandirilmamis yeni bir pencere dugmesidir. (2) N-cisim tarafindaki "
    "`power` (kuvvet ussu): p = 2 Newton'dur ve p != 2 BIR YAKLASIM DEGIL "
    "BASKA BIR KURAMDIR -- dahasi p != 2'de potansiyelin kapali formu bir "
    "SECIMDIR (sevk edilen secim U = -m_i*m_j/((p-1)*r^(p-1)), yani "
    "kuvvetin integrali) ve bu secim adlandirilmadigi icin planin butun bir "
    "us tablosu yeniden URETILEMEDI. p = 2'de iki secim CAKISIR, o yuzden "
    "cipanin kendisi etkilenmez"
)

_GURULTU_TABANI_BEYANI = (
    "HER FIZIK SAYISININ HATA CUBUGU KENDI BORUSUNDAN OKUNUR -- KARISIK "
    "BORU YASAK. Yordam: fizigi KAPAT (lam = 0, yani beta = 2 + 2*gamma; "
    "N-cisimde m2 = 0), AYNI boruyu AYNI pencerede kostur, cikan sapmayi "
    "taban say. Uc borunun UCU DE FARKLI DAVRANIYOR ve panel bunu yazar, "
    "cunku 'S/G buyuk' cumlesi bu farklar olmadan yaniltir: KARELEME "
    "borusunda w - 1 ile dphi - 2*pi ayni buyuklugun iki yuzudur, bu yuzden "
    "hukum SAGLAM'dan dogrudan DEJENERE'ye gecer ve CURUK'e ERISILEMEZ; ODE "
    "borusunda hicbir basamak olmez, bu yuzden CURUK erisilebilir ama "
    "DEJENERE erisilemez -- karelemenin TAM TERSI; N-cisim tarafinda ise "
    "iki gozlenebilir iki AYRI dejenerelik yuzeyi tasir (dairesel frekans "
    "kusuru 7/7 canli basamakla SAGLAM, spec 4.3'un kendi dogrusal-harita "
    "gozlenebiliri ise 0/1 ile DEJENERE ve orani `nan`). Ve bir hata "
    "cubugunda CURUK YAPISAL OLARAK DAR BIR OLAYDIR: sinyal her zaman "
    "varsayilan fizik noktasinin O(1) degeridir ve taban ona ancak boru "
    "cope dondugunde yetisir -- uc boruda uc kez olculdu"
)

_PENCERE_BEYANI = (
    "BUKULME ORANININ PENCERESI AYAR-BAGIMLIDIR ve sertifika hangi "
    "koordinatta oldugunu YAZAR: (oran - 2)*b/M Schwarzschild ayarinda "
    "7*pi/8 = 2.748894'e, izotropik ayarda pi/2 = 1.570796'ya gidiyor, fark "
    "TAM 3*pi/8 = 1.178097. Bu sayi bir zamanlar 15*pi/8 ve 3*pi/2 diye "
    "yaziliydi; ikisi de TAM BIR pi kadar yanlisti (c2(gamma = 0) = pi "
    "ortuk olarak sifir sanilmisti) ve FARKIN DOGRU KALMASI tam bu yuzden "
    "yanilticiydi -- ortak bir kayma farkta sadelesir, yani fark dogruyken "
    "iki mutlak sayi da yanlis olabilir. "
    "RICHARDSON'IN KENDI PENCERESI VAR VE OPTIMUMDAN SONRA BOZULUYOR: "
    "kareleme (1e5, 3e5)'te 9.12, (3e5, 1e6)'da 10.01, (1e6, 3e6)'da 9.94, "
    "(3e6, 1e7)'de 8.98 durust hane veriyor. ODE'nin optimumu ise `rtol` "
    "ILE YER DEGISTIRIYOR (rtol 1e-12'de (3e3, 1e4), rtol 1e-13'te "
    "(1e4, 3e4)), yani 'ODE 7.20 hane verir' cumlesi bir pencere beyani "
    "olmadan yazilamaz. "
    "e > 0 BIR KAPIDIR ve gerekcesi SAYISAL DEGIL SOZLESMESELDIR: u1 == u2 "
    "bir DAIREDIR ve orada olculecek bir perihel kaymasi yoktur. "
    "Pencere adlandirilmadan sayi yazilmaz -- bu depoda hukmun penceresinin "
    "disina tasinmasi ON BIRDEN FAZLA kez olculdu"
)

_GRAV_BEKLENEN_HATA = (
    "Bu modulun baskin hatasi bir ayriklastirma hatasi degil bir "
    "DISDEGERLEME artigidir ve buyuklugu OLCULDU (olcum tests/grav/ "
    "altindadir; bu senaryo onu YENIDEN OLCMEZ): kareleme borusunda GR "
    "perihel katsayisi 18.84955591968662, 6*pi'den -1.852140e-09 sapiyor "
    "(10.01 durust hane) ve izotropik ayarda 18.849555921564487, "
    "+2.572875e-11 (11.86 hane). Iki ayarin farki 1.4972e-09 ... "
    "3.3116e-09 araligindadir ve ORTAK KOD YOLU YOKTUR -- kaydeden vekil "
    "Schwarzschild yolunun `Theory`den yalniz {lam}, izotropik yolun yalniz "
    "{gamma, beta} okudugunu gosteriyor, KESISIM BOS. Bukulme orani "
    "1.9999999996887523 (Schwarzschild) ve 1.999999999804698 (izotropik), "
    "yani 9.81 ve 10.01 hane; sekiz sabiti 9.237681250723917 ve dort "
    "integratorde yayilimi 1.972e-12"
)

_DURUST_OLUMSUZLAR = (
    "DURUST OLUMSUZLAR -- BEYAN EDILDI, DUZELTILMEDI, VE PANEL BUNLARI "
    "SAKLAMAZ: (1) TABAN YALNIZ ARITMETIGI OLCER. Kesme hatasi fizikle "
    "BIRLIKTE kapandigi icin 'fizigi kapat, ayni boruyu kostur' sinavi onu "
    "YAPISAL OLARAK GOREMEZ -- tek basamakli bir pencerede gercek hata "
    "2.842e-04 iken taban 2.453e-10, yani gercegin 1/1158000'i. S/G orani "
    "ancak YUVARLAMA SINIRLI bir pencerede bir hata cubugudur. (2) IKINCI "
    "MERTEBE K AYAR-BAGIMLIDIR (GR, e = 0.3, p = 3e5: Schwarzschild "
    "85.248585, izotropik 26.577757), dolayisiyla IKI AYARLI CAPRAZ KONTROL "
    "YALNIZ BIRINCI MERTEBEYI belgeler. (3) CAPRAZ KONTROLUN GORMEDIGI "
    "YUZEY OLCULDU: iki ayar donus noktalarini, `quad` dugmelerini, psi "
    "ikamesini ve Richardson ciftini PAYLASIR, ve paylasilan yuzeye sokulan "
    "bir bozulma (or. 1 + e) iki tarafi BIRLIKTE kaydirip farki BUYUTMEZ, "
    "KUCULTUR -- orada capraz kontrol kor degil YANILTICIDIR; olduren sey "
    "6*pi cipasidir. (4) SPEC 4.3'UN KENDI N-CISIM GOZLENEBILIRINDE TABAN "
    "VE KESISIM URETILEMIYOR: m2 = 0'da dogrusal haritanin etki blogu BIT "
    "BIT sifirdir (32/32 eleman) ve limit de kurtarmaz, cunku gurultu "
    "sinyalle birlikte olcekleniyor -- sevk edilen dairesel-frekans kusuru "
    "bir IKAMEDIR ve gerekcesi bu OLCUMDUR, bir tercih degil. (5) IZOTROPIK "
    "K'NIN KAPALI FORMU TURETILMEDI, ve izotropik lam = 0 cizgisinin model "
    "artigi -24*pi*(1 + gamma)^2 OLCULDU ama TURETILMEDI"
)


@dataclass(frozen=True, eq=False)
class GravSetup:
    """Bir yercekimi kosumunun ham kaydi -- gozlemlenebilir HESAPLANMAMIS.

    `trajectory` ODE yorunge motorundan, `figure_eight` N-cisim BVP'sinden
    gelir; ikisi de `engine` icindedir. Cipa sayilari (katsayilar, oranlar,
    gurultu tabanlari) BURADA hesaplanmaz, cunku `engine` katmani
    `observable`i import edemez -- onlar `viz/grav_frames.py`de kurulur.
    Modul 5'in `EMScanResult`iyle ayni sozlesme.

    `eq=False`: `Trajectory` ndarray tasir; uretilen `__eq__` dizi dondurur
    ve `__hash__` `TypeError` verir. Depo bu tuzagi `engine/hf/integrals.py`,
    `engine/md/state.py`, `spin/lattice.py` ve `em/factory.py`de zaten odedi.
    """

    scenario: GravScenario
    theory: Theory
    trajectory: Trajectory
    figure_eight: FigureEight


@dataclass(frozen=True)
class FloorLine:
    """Bir gurultu tabaninin PANEL SATIRI -- duz veri, `NoiseFloor` DEGIL.

    `NoiseFloor` bir `observable` turudur ve bu dosya onu goremez (katman
    sozlesmesi). `viz/grav_frames.py` olculen tabani buraya cevirir; boylece
    panel metni burada kalirken sayilar disaridan gelir.

    `anchor` alani, tabanin bir CIPAYA mi ait oldugunu soyler ve panelin
    `converged` hukmune YALNIZ cipa tabanlari girer. Sebep aciktir ve
    saklanmaz: spec 4.3'un kendi dogrusal-harita gozlenebiliri YAPISAL
    OLARAK dejeneredir, cipa DEGILDIR ve sevk edilen ikamenin GEREKCESIDIR;
    hukme katilsaydi panel hicbir kosumda tik basamazdi ve tik'in anlami
    kaybolurdu. Satirin kendisi yine de BASILIR.
    """

    label: str
    value: float
    signal: float
    ratio: float
    pipeline: str
    verdict: str
    anchor: bool = True


def build_grav(scenario: GravScenario) -> GravSetup:
    """Senaryodan fizigi kosar: ODE yorungesi + sekizin BVP'si.

    TOHUM YOK, PENCERE VAR. Bu modulde RNG hic kullanilmaz; yorunge motoru
    determinist bir `solve_ivp` cagrisidir ve BVP sabit bir baslangic
    tahmininden koslar. Tekrarlanabilirligin karsiligi senaryodaki
    pencerenin TAM BEYANIDIR.

    IKI COZUCU BURADA DA AYRI KALIR: `run_trajectory` egri uzay-zamanda bir
    TEST PARCACIGI tasir, `solve_figure_eight` duz uzayda BIRBIRINI CEKEN uc
    cisim tasir. Bu fonksiyon ikisini yan yana kosar ama BIRLESTIRMEZ --
    araligin ortak tek seyi senaryonun kendisidir.

    `theory` BURADA KURULUR ve tek kaynak (gamma, beta)dir; `lam` senaryoda
    bir anahtar DEGILDIR ve burada da bir alan degil turetilmis bir
    ozelliktir.

    BVP `nbody.solve_figure_eight` icinde `lru_cache`lidir, dolayisiyla ayni
    (guess, rtol, atol, method, mass, power) icin ikinci cagri bedavadir --
    ama bu bir onbellek OPTIMIZASYONU degil bir SOZLESMEDIR: ayni pencerede
    iki farkli cozum donmesi cipanin kendisini belirsiz kilardi.
    """
    if scenario.gauge not in GAUGE_NAMES:
        raise InvalidInputError(
            f"bilinmeyen ayar: {scenario.gauge!r}. Sevk edilen ayarlar: "
            f"{GAUGE_NAMES}."
        )
    theory = Theory(gamma=scenario.gamma, beta=scenario.beta)
    trajectory = run_trajectory(
        theory=theory,
        M=scenario.mass,
        p=scenario.trajectory_p,
        e=scenario.eccentricity,
        n_points=scenario.trajectory_points,
    )
    figure_eight = solve_figure_eight(
        rtol=scenario.nbody_rtol,
        atol=scenario.nbody_atol,
        method=scenario.nbody_method,
    )
    return GravSetup(
        scenario=scenario,
        theory=theory,
        trajectory=trajectory,
        figure_eight=figure_eight,
    )


def _kuram_secici_metni(setup: GravSetup) -> str:
    """Bu KOSUMUN kuram secicisi -- ve GR olup olmadigi.

    `GR_DEGIL_ISARETI` burada yazilmaz, sabitten okunur; iki yerde elle
    yazilsaydi biri degistiginde panel ile test farkli sey soylerdi.
    """
    t = setup.theory
    gr = (t.gamma, t.beta) == GR_NOKTASI
    hukum = (
        "bu kosum GENEL GORELILIK cozuyor (gamma = beta = 1, lam = 3)"
        if gr
        else f"{GR_DEGIL_ISARETI}: gamma = {t.gamma:g}, beta = {t.beta:g}, "
        f"lam = {t.lam:g} (GR icin 3 olmali)"
    )
    return (
        f"[CANLI -- bu kosum] {hukum}; N-cisim tarafinda kuvvet ussu "
        f"power = {setup.figure_eight.power:g} "
        f"({'Newton' if setup.figure_eight.power == 2.0 else 'NEWTON DEGIL'}). "
        + _KURAM_SECICI_BEYANI
    )


def grav_declaration(setup: GravSetup) -> str:
    """PANELIN ZORUNLU BEYANLARININ TEK KAYNAGI, tek metin halinde.

    `provenance_for` ayni sabitleri `Provenance`in AYRI alanlarina dagitir;
    bu fonksiyon onlari tek bir dizge isteyen cagiranlar icin (test, ileride
    bir GUI paneli) birlestirir. IKI YOL DA AYNI SABITLERI OKUR, yani metnin
    ikinci bir kopyasi YOKTUR -- ve `tests/grav/test_grav_senaryo.py` bunu
    hem tam cumle hem PARCA duzeyinde kilitler.

    Sirasi panelin sirasidir: yaklasimlar, ihmal edilenler (yontem/hedef
    uyusmazliginin IKI TURU ve MODULLER ARASI YOKLUK BEYANI dahil -- ucu de
    `_GRAV_IHMAL`in ogeleridir ve burada AYRICA sayilmazlar, yoksa metin iki
    kez akardi), beklenen hata + durust olumsuzlar, dairesellik beyaninin IKI
    PARCASI, cipa sinifi, kuram secici, gurultu tabani ve pencere.
    """
    parcalar = (
        _GRAV_YAKLASIMLAR
        + _GRAV_IHMAL
        + (
            _GRAV_BEKLENEN_HATA,
            _DURUST_OLUMSUZLAR,
            _DAIRESELLIK_43_SANIYE,
            _DAIRESELLIK_SEKIZ,
            _CIPA_SINIFI_BEYANI,
            _kuram_secici_metni(setup),
            _GURULTU_TABANI_BEYANI,
            _PENCERE_BEYANI,
        )
    )
    return " ".join(parcalar)


#: Kurama BAGLI kapali-form turevler. `beta > gamma`da bu ucu de YOKTUR --
#: 1PN'de kesilmis A(u) orada ufuk yapmaz, foton kuresi bulunmaz ve donen
#: ISCO koku fiziksel degildir. `mu_c` ile apsidal us bu listede DEGILDIR
#: cunku onlar kuramdan bagimsizdir (biri Newtonyen kisitli uc-cisim
#: probleminin, oteki merkezi kuvvet ailesinin sayisi).
KURAMA_BAGLI_KAPALI_FORMLAR: tuple[str, ...] = (
    "isco_r_over_rs", "foton_r_over_rs", "b_krit_over_M",
)

#: Kapali-form turevi OKUNAMADIGINDA panelin bastigi isaret. TEK bir sabit
#: olarak durur ve `tests/grav/test_grav_senaryo.py` onu BURADAN import eder.
KAPALI_FORM_YOK = (
    "OKUNAMADI (beta > gamma: 1PN'de kesilmis A(u) ufuk YAPMIYOR, foton "
    "kuresi YOK; inanilir gorunen bir sayi dondurmek yerine reddedildi)"
)


def _kapali_form_metni(closed_forms: dict[str, float]) -> str:
    """Kapali-form turevlerin panel metni -- EKSIK OLANI DA YAZAR.

    Bir anahtarin YOKLUGU sessizce atlanmaz: atlansaydi panel "bu kuram
    noktasinda ISCO yok" ile "ISCO'yu hesaplamayi unuttum" arasindaki farki
    gosteremezdi. Modul 1B'nin kararlilik alani dersi ve `_run_spin`in isi
    sigasi kuralinin aynisi -- ama ters yonde uygulanmis: orada anahtar HIC
    yazilmiyordu cunku kayit makine-okunurdu; burada PANEL METNI okunuyor ve
    okuyucunun gormesi gereken sey tam olarak YOKLUGUN KENDISIDIR.
    """
    parcalar = []
    for ad in KURAMA_BAGLI_KAPALI_FORMLAR:
        if ad in closed_forms:
            parcalar.append(f"{ad}={closed_forms[ad]:.12g}")
        else:
            parcalar.append(f"{ad}={KAPALI_FORM_YOK}")
    for ad, deger in closed_forms.items():
        if ad not in KURAMA_BAGLI_KAPALI_FORMLAR:
            parcalar.append(f"{ad}={deger:.12g}")
    return "  ".join(parcalar)


def _taban_satiri(f: FloorLine) -> str:
    """Bir gurultu tabaninin tek satirlik hali. TEK metin kaynagi.

    `ratio` DEJENERE boruda `nan`dir ve o zaman bir SAYI BASILMAZ: "nan"
    yazmak bir oran okunmus gibi gorunurdu, "inf" yazmak ise "sonsuz
    hassasiyet" diye okunurdu -- ikisi de yalan olurdu. Basilan sey hukmun
    kendisidir.
    """
    if math.isnan(f.ratio):
        oran = "S/G OKUNAMADI (boru dejenere)"
    else:
        oran = f"S/G = {f.ratio:.4e}"
    return (
        f"{f.label}: taban {f.value:.6e}, sinyal {f.signal:.12g}, {oran}, "
        f"hukum {f.verdict} [{f.pipeline}]"
    )


def provenance_for(
    scenario: GravScenario,
    *,
    setup: GravSetup,
    floors: tuple[FloorLine, ...],
    second_order: float,
    closed_forms: dict[str, float],
    eight: float,
    angles: tuple[float, float],
) -> Provenance:
    """Olculmus sayilardan panel beyanini kurar. HICBIRINI YENIDEN HESAPLAMAZ.

    `floors` / `second_order` / `closed_forms` / `eight` / `angles`
    `viz/grav_frames.py`den gelir ve o da `observable/grav_obs.py`den okur.
    Bu fonksiyon hicbirini yeniden hesaplamaz -- hesaplasaydi panel ile kayit
    ayrisabilir ve panel motorun soylemedigini yazardi (Modul 5'in
    `em/factory.py::provenance_for` gerekcesinin aynisi).

    `converged` OLCULEBILIR bir seye baglanir, ve bu modulde o sey bir SCF
    dongusunun durmasi ya da bir merdivenin oturmasi DEGILDIR: BU KOSUMUN
    HER CIPA BORUSU KENDI HATA CUBUGUNU OKUYABILDI MI? Yani tik "hesap
    bitti" demez, "bastigim her cipa sayisinin YANINDA gecerli bir hata
    cubugu VAR" der. CURUK ya da DEJENERE bir cipa borusu tik'i dusurur.
    Cipa OLMAYAN borular (spec 4.3'un dogrusal-harita gozlenebiliri) hukme
    GIRMEZ ama satirlari BASILIR -- gerekcesi `FloorLine.anchor`da yazili.

    KARARLILIK HUKMU `max|lambda|` ILE VERILMEZ. Sekizin monodromisinde
    lambda = 1 SEKIZ KATLI ve DEFEKTIFTIR (cebirsel kat 8, geometrik kat 4,
    yani dort tane 2x2 Jordan blogu) ve defektif bir ozdeger bozulmaya
    KAREKOKLE tepki verir; `max|lambda| - 1` bu yuzden yakinsamaz ve
    raporlanabilir bir sayi degildir. Panel eliptik ciftin ACILARINI basar.
    """
    if not floors:
        raise InvalidInputError(
            "panel en az bir gurultu tabani satiri ister: spec 6.6 her fizik "
            "sayisinin yaninda KENDI borusundan okunmus bir taban sart kosar."
        )
    cipalar = tuple(f for f in floors if f.anchor)
    if not cipalar:
        raise InvalidInputError(
            "en az bir CIPA tabani gerekir; yalnizca cipa olmayan borularla "
            "kurulan bir panelin `converged` hukmu tanimsiz olurdu."
        )
    yakinsadi = all(f.verdict == "SAGLAM" for f in cipalar)

    t = setup.trajectory
    kayma = float(t.phi[-1]) - 2.0 * math.pi
    fe = setup.figure_eight
    kf = _kapali_form_metni(closed_forms)

    canli = (
        f"[CANLI -- bu kosum] perihel katsayisi "
        f"C = {floors[0].signal:.14g} (ayar {scenario.gauge}, Richardson "
        f"{scenario.richardson[0]:g}/{scenario.richardson[1]:g}); ikinci "
        f"mertebe K = {second_order:.12g} (p = {scenario.second_order_p:g}); "
        f"sekiz sabiti |E|^{{3/2}}T/(G^2 m^{{5/2}}) = {eight:.15g} "
        f"(vx = {fe.vx:.12f}, vy = {fe.vy:.12f}, T = {fe.period:.12f}, "
        f"BVP artigi {fe.residual:.6e}); kararlilik ELIPTIK CIFTIN ACISIYLA "
        f"+-{angles[0]:.12f} ve +-{angles[1]:.12f} rad; ODE yorungesinin bir "
        f"radyal periyodu {float(t.phi[-1]):.12f} rad, yani 2*pi'den "
        f"{kayma:.6e} buyuk; kapali-form turevler {kf}. "
        "GURULTU TABANLARI (her biri KENDI borusundan): "
        + " | ".join(_taban_satiri(f) for f in floors)
        + "."
    )
    beyan = (
        f"[BEYAN -- olcumu tests/grav/test_hata_cubugu.py ve "
        f"tests/grav/test_perihel.py'de] {_GURULTU_TABANI_BEYANI}"
    )

    pencere = (
        f"BU KOSUMUN PENCERESI: G = c = 1, M = {scenario.mass:g}, "
        f"e = {scenario.eccentricity:g}, ayar = {scenario.gauge}; kareleme "
        f"Richardson cifti ({scenario.richardson[0]:g}, "
        f"{scenario.richardson[1]:g}), ikinci mertebe tek basamak "
        f"p = {scenario.second_order_p:g}, bukulme Richardson cifti "
        f"({scenario.deflection_richardson[0]:g}, "
        f"{scenario.deflection_richardson[1]:g}); ODE yorunge motoru "
        f"p = {scenario.trajectory_p:g}, {scenario.trajectory_points} nokta; "
        f"N-cisim {scenario.nbody_method}, rtol = {scenario.nbody_rtol:g}, "
        f"atol = {scenario.nbody_atol:g}. " + _PENCERE_BEYANI
    )

    return Provenance(
        method=(
            f"Yercekimi ve uzay-zaman ({scenario.name}); IKI AYRI COZUCU -- "
            f"egri uzay-zamanda test parcacigi (kareleme = CIPA YOLU, "
            f"ODE = YORUNGE MOTORU, ayar {scenario.gauge}) ve duz uzayda "
            f"Newtonyen uc-cisim (sekiz yorungesinin BVP'si, "
            f"{scenario.nbody_method})"
        ),
        approximations=_GRAV_YAKLASIMLAR
        + (
            f"sonlu pencere: bu kosum kareleme tarafinda "
            f"({scenario.richardson[0]:g}, {scenario.richardson[1]:g}) "
            f"ciftini kullanir ve SUREKLI LIMIT ALINMAZ; okunan hane sayisi "
            f"bu ciftin kendi optimumuna aittir, sonsuz ince basamagin "
            f"degil",
        ),
        neglected=_GRAV_IHMAL,
        expected_error=_GRAV_BEKLENEN_HATA + " " + _DURUST_OLUMSUZLAR,
        converged=yakinsadi,
        circularity=_DAIRESELLIK_43_SANIYE + " " + _DAIRESELLIK_SEKIZ,
        anchor_class=_CIPA_SINIFI_BEYANI,
        theory_selector=_kuram_secici_metni(setup),
        window=pencere,
        convergence_diagnosis=canli + " " + beyan,
        diagnostics={
            "ayar": scenario.gauge,
            "gamma": float(scenario.gamma),
            "beta": float(scenario.beta),
            "e": float(scenario.eccentricity),
            "nbody": scenario.nbody_method,
            "cipa_borusu": str(len(cipalar)),
        },
    )
