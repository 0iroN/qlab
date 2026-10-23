"""Spin gozlemlenebilirleri -- ve HATA CUBUGUNUN KENDISI.

Bu modul motoru bilmez; yalniz seri ve konfigurasyon okur. Ana isi, MC
serilerinden sayi cikarmak DEGIL, cikardigi sayinin NE KADAR guvenilir
oldugunu da beraberinde dondurmektir.

NAIF SEM BU DEPODA HICBIR YERDE RAPORLANMAZ. MC ornekleri ILISKILIDIR ve
`std(seri)/sqrt(N)` gercek hatayi sistematik olarak KUCUK gosterir. Olculdu
(L = 32, T = T_c, 8000 olcum; BAGIMSIZ referans = 200 zincirin (Metropolis)
ve 100 zincirin (Wolff) ortalamalarinin std'si): Metropolis'te naif SEM
1.946079e-03, gercek 1.713639e-02 -- 8.81 KAT kucuk (referansin kendi bagil
belirsizligi %5.0); Wolff'ta 1.955908e-03 vs 4.642595e-03 -- 2.37 kat (%7.1).
Ayni referanslar YALNIZ 24 zincirle olculdugunde 1.3666e-02 ve 5.2827e-03
cikiyor, yani 24 zincir Metropolis'te gercegin 0.80 kati: referansin kendisi
de bir tahmindir ve bu depoda bu ders (Task 2, Task 3) ucuncu kez olculdu.
`binning_error` naif SEM'i yalniz `errors[0]` olarak tasir, `error` olarak
ASLA dondurmez.

PLATO OLCUTU -- PLANIN OLCUTU OLCULDU ve KENDI IDDIASINI GECMEDI:
Plan platoyu "son uc blok boyutunun hatalari %5 icinde mi" diye tanimliyor
ve `min_blocks = 16` kullaniyordu. Bir std tahmininin bagil sacilimi
~ 1/sqrt(2(K-1)); `min_blocks = 16`ta son uc basamakta yalniz K = 125, 62,
31 blok kalir, yani sacilimlar %6.35, %9.05, %12.91 -- ESIGIN IKI-UC KATI.
Olculdu (24 bagimsiz zincir): plan olcutu Wolff'un plato YAPTIGI pencerede
bile 24 tohumun yalniz 3'unde `True` diyor, yani planin kendi zorunlu testi
kendi koduyla 21/24 kirmizi olurdu.

SECILEN OLCUT -- son nokta, uc onceki noktadan ISTATISTIKSEL OLARAK anlamli
bicimde yuksek degilse plato:

    e[-1] - e[-3]  <=  z * e[-1] * sqrt(sig[-1]^2 + sig[-3]^2),
    sig_i = 1 / sqrt(2 (K_i - 1))

`min_blocks = 32` ve `z = 2` KEYFI DEGIL, ama BEDAVA da degil: iki BILINEN
tarafi (Metropolis'in cozulmemis kosumu ile Wolff'un cozulmus kosumu)
AYIRSIN diye alti kombinasyon tarandi (L = 32, T_c, 8000 olcum, 24
zincir) ve yalniz biri ayirdi:

    min_blocks  z     Metropolis (plato OLMAMALI)   Wolff (plato OLMALI)
        16     1.0            4/24                        19/24
        16     2.0           19/24  (yanlis plato)        24/24
        32     1.0            0/24                        17/24
        32     2.0            0/24                        24/24   <- SECILEN
        64     1.0            0/24                         7/24
        64     2.0            0/24                        19/24

BU BIR SECIMDIR VE SECILDIGI VERI UZERINDE DOGRULANAMAZ: iki dugme,
hukmu istenen yonde verecek sekilde TARANARAK bulundu, dolayisiyla ayni
kosumlarda dogru hukmu vermesi bir kanit degil bir tanimdir. Daireyi
kiran sey asagidaki sinavdir -- AR(1)'de `tau_int` KAPALI FORMDAN
bilinir ve olcut o sistemde HIC AYARLANMADAN dogru hukmu veriyor
(N = 8000, 24 tohum):

    phi     tam tau   binning tau   oran    plato
    0.0       0.500       0.541     1.082   24/24
    0.5       1.500       1.552     1.035   24/24
    0.8       4.500       4.446     0.988   24/24
    0.95     19.500      16.333     0.838   10/24  (gecis bolgesi)
    0.995   199.500      52.833     0.265    0/24

Bayrak tam olarak isini yapiyor: `True` dedigi yerde `tau_int` %1 dogru,
`False` dedigi yerde gercegin dortte biri.

BAYRAK ALGORITMANIN DEGIL KOSUMUN OZELLIGIDIR. Planin anlatisi "Metropolis
plato yapmaz, Wolff yapar" diyor; bu ALGORITMA hakkinda bir cumle olarak
YANLIS. Olculdu (T = T_c, 24 zincir): Metropolis L = 8, 4000 olcumde 24/24
plato yapiyor, L = 16, 8000 olcumde 23/24; L = 32, 8000 olcumde 0/24.
Bayrak "bu algoritma iyi mi" degil, "BU KOSUM kendi otokorelasyonunu
cozecek kadar uzun mu" sorusunu cevaplar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.errors import InvalidInputError

#: Son basamakta kalan blok sayisinin alt siniri. 32 OLCULEREK secildi
#: (modul dokstringindeki tarama); 16 son basamagi %12.9 sacilimla
#: gurultuye bogar, 64 ise egriyi Wolff'un plato bolgesine varmadan keser.
DEFAULT_MIN_BLOCKS: int = 32

#: Plato olcutunun sigma carpani. 2.0 OLCULEREK secildi; 1.0'da Wolff'un
#: gercek platosu 24 tohumun 7'sinde kaciriliyor.
_PLATO_Z: float = 2.0


@dataclass(frozen=True)
class BinningResult:
    """Blok-ortalamasi hata analizinin TAM kaydi.

    `error`: raporlanacak hata. **`plateau_reached` False ise bu sayi hatanin
    KENDISI DEGIL bir ALT SINIRIDIR** ve cagiran bunu bilmek zorundadir.

    `plateau_reached`: binning egrisi oturmus mu. Modul dokstringindeki
    olcutle hesaplanir.

    `block_sizes` / `errors`: egrinin kendisi. Cagiran hukmu kendi
    gozuyle denetleyebilsin diye tam olarak tasinir; `error` ve `tau_int`
    ikisinden TURETILEBILIR olmali (test bunu kilitler).

    `tau_int`: integre otokorelasyon zamani, binning'den TURETILIR --
    `SEM_plato / SEM_naif = sqrt(2 tau)` ozdesliginden. BAGIMSIZ bir
    otokorelasyon hesabiyla capraz kontrol edilir
    (`tests/spin/test_spin_obs.py::test_tau_int_IKI_BAGIMSIZ_YOLDAN_okunuyor`);
    olculen uyum plato VARSA %6-8, plato YOKSA binning sistematik olarak
    %11-28 DUSUK -- yani `plateau_reached` False iken `tau_int` de bir alt
    sinirdir.
    """

    error: float
    plateau_reached: bool
    block_sizes: tuple[int, ...]
    errors: tuple[float, ...]
    tau_int: float


def _seri(dizi, ad: str) -> np.ndarray:
    """Tek boyutlu, sonlu, bos olmayan bir seriye cevirir.

    `None` "OLCULMEDI" demektir ve SIFIR SAYILMAZ. Bu, Modul 1B'nin
    kararlilik alani dersinin aynisidir: analiz edilmedi ile kararli ayni
    degildir ve ayni degerle gosterilmemelidir. Wolff (`engine/spin/wolff.py`)
    `energy_series = None` dondurur; onu sifir sanan bir isi sigasi "C = 0"
    diye okunur ve hicbir uyari vermez.
    """
    if dizi is None:
        raise InvalidInputError(
            f"{ad} olculmedi (None). None SIFIR DEGILDIR: 'olculmedi' ile "
            "'sifir cikti' ayni sey degildir ve ayni degerle gosterilemez. "
            "Wolff kume algoritmasi enerjiyi olcmez; isi sigasi icin "
            "Metropolis kosumu gerekir."
        )
    a = np.asarray(dizi, dtype=float)
    if a.ndim != 1:
        raise InvalidInputError(f"{ad} tek boyutlu olmali, verilen sekil: {a.shape}")
    if a.size == 0:
        raise InvalidInputError(f"{ad} bos")
    if not np.all(np.isfinite(a)):
        raise InvalidInputError(
            f"{ad} sonlu olmali; {int(np.count_nonzero(~np.isfinite(a)))} "
            f"sonlu olmayan deger var"
        )
    return a


def _kafes_ve_sicaklik(L: int, T: float) -> None:
    if int(L) != L or L < 1:
        raise InvalidInputError(f"L pozitif tam sayi olmali, verilen: {L}")
    if not np.isfinite(T) or T <= 0.0:
        raise InvalidInputError(f"sicaklik pozitif ve sonlu olmali, verilen: {T}")


def binder_cumulant(m) -> float:
    """U = 1 - <m^4> / (3 <m^2>^2).

    CIFT momentlerden kurulur, dolayisiyla isaretli `m` ile `|m|` AYNI sayiyi
    verir -- motorun `magnetisation_series`i `|m|` tasidigi icin bu onemli.

    IKI KAPALI FORM CIPASI (MC'ye hic sorulmadan sinar):
    - Bagimsiz Gauss dizisinde <m^4> = 3 <m^2>^2, yani U -> 0. Olculdu
      (`default_rng(20260824)`): n = 1e4'te +0.037629249, n = 2e5'te
      +0.001511486, n = 2e6'da -0.001646572 -- 1/sqrt(n) ile sifire gidiyor.
    - Tam duzenli (m sabit) durumda <m^4> = <m^2>^2, yani U = 2/3. Olculdu:
      0.666666666667 (TAM).

    MUTLAK DEGER TAM SAYIMLA KILITLI (L = 4, 2^16 durum): T = 2.0 -> tam
    0.644810791, T_c -> 0.617199318, T = 3.5 -> 0.380777825.
    """
    a = _seri(m, "miknatislanma serisi")
    m2 = float(np.mean(a**2))
    if m2 <= 0.0:
        raise InvalidInputError(
            "Binder kumulanti <m^2> = 0 iken tanimsiz (seri tumuyle sifir); "
            "0/0 sessizce nan dondurulmez"
        )
    m4 = float(np.mean(a**4))
    return 1.0 - m4 / (3.0 * m2 * m2)


def susceptibility(m, L: int, T: float) -> float:
    """chi = L^2 (<x^2> - <x>^2) / T, `x` VERILEN seri.

    HANGI ORTALAMA KULLANILIYOR -- ACIKCA BEYAN: fonksiyon kendisine verilen
    serinin momentlerini alir. Motorun `magnetisation_series`i `|m|` tasir,
    dolayisiyla pratikte hesaplanan sey

        chi' = L^2 (<m^2> - <|m|>^2) / T

    yani MUTLAK DEGER suseptibilitesidir. Sonlu kafeste bu, isaretli
        chi  = L^2 (<m^2> - <m>^2) / T
    ile AYNI SEY DEGILDIR ve fark kucuk degildir. TAM SAYIMDAN olculdu
    (L = 4, 2^16 durum):

        T        chi(|m|)      chi(m)      oran
        2.000    0.195720     6.951373    35.52
        T_c      0.347321     5.368333    15.46
        3.500    0.383037     1.473647     3.85

    Sebep: sonlu kafeste <m> simetriden oturu 0'a gider ve chi(m) miknatisin
    +m ile -m arasindaki TUNELLEMESINI olcer, denge duyarliligini degil.
    Bu yuzden sonlu-boyut analizinde `<|m|>` kullanilir.

    `L^2` KATSAYISI kafes basina site sayisidir (kare kafes). Mutlak deger
    tam sayimla kilitlidir; `L` yazilsa sonuc L = 4'te tam 4 kat saparadi.

    Varyans POPULASYON varyansidir (ddof = 0) ve MERKEZLI hesaplanir
    (`_merkezli_varyans`, HC-SABIT): bir dagilimin momentleri, orneklem degil.
    """
    a = _seri(m, "miknatislanma serisi")
    _kafes_ve_sicaklik(L, T)
    varyans = _merkezli_varyans(a)
    return L * L * varyans / T


def _merkezli_varyans(a: np.ndarray) -> float:
    """Populasyon varyansi (ddof = 0), ILK ORNEGE KAYDIRILMIS iki gecisle.

    HC-SABIT (14.10.2026): `<x^2> - <x>^2` iki BUYUK ve neredeyse esit sayinin
    farkidir ve iptal hatasi tasir. KAMU YOLUNDAN OLCULDU (HEAD 448f808,
    `qlab.run`, Ising L = 8, T = 2, 64 olcum, enerji serisi tumuyle sabit,
    fizik C = 0): `field` = 1e10 -> C = 262144, 1e100 -> C = -2.7e185
    (negatif isi sigasi imkansiz). Ayni kusur `|m|` serisinde kucuk olcekte
    vardi: sabit 0.7 serisi (n = 64) varyans -2.2e-16, yani NEGATIF chi.

    Duz iki gecis (ortalamayi cikar, karelerin ortalamasi) YETMEZ: ortalamanin
    kendi yuvarlamasi kalir -- olculdu, sabit -6.4e101 serisinde n = 1000'de
    1.5e172. Ilk ornege kaydirmak sabit seride farki TAM sifir yapar
    (x - x = 0), dalgalanan seride merkezli iki gecisin dogrulugunu korur
    (ofset 1e10 + birim varyansli gurultu: eski formul 0.0, bu 1.000000;
    `np.var` ile ayni). Ikinin kuvvetiyle olcekleme ve ayni seri BIT BIT
    ayni sonucu verir (coupling olcek kilidi, F7 ayna kilidi korunur).
    """
    d = a - a[0]
    d = d - d.mean()
    return float(np.mean(d**2))


def heat_capacity(e, L: int, T: float) -> float:
    """C = (<E^2> - <E>^2) / (L^2 T^2), site basina.

    `e` TOPLAM enerji serisidir (`lattice.total_energy` ekstansiftir); `L^2`
    bolmesi sonucu site basina getirir.

    `e is None` ise `InvalidInputError` firlatilir -- Wolff enerjiyi OLCMEZ
    ve `None` SIFIR DEGILDIR (`_seri` dokstringi).

    MUTLAK DEGER TAM SAYIMLA KILITLI (L = 4, 2^16 durum, bagimsiz yazilmis
    enerji): T = 2.0 -> tam 0.605532657, T_c -> 0.783266826,
    T = 3.5 -> 0.375755537.

    Varyans MERKEZLI hesaplanir (`_merkezli_varyans`, HC-SABIT 14.10.2026):
    enerjinin sifiri keyfidir, C ona ofset eklenince degismemelidir. Eski
    `<E^2> - <E>^2` sabit seride `field` = 1e100'de C = -2.7e185 veriyordu.
    """
    a = _seri(e, "enerji serisi")
    _kafes_ve_sicaklik(L, T)
    varyans = _merkezli_varyans(a)
    return varyans / (L * L * T * T)


def correlation_length(spins) -> float | None:
    """Ikinci moment korelasyon uzunlugu. TAHMINCI KIRILIRSA None doner.

        xi = sqrt( S(0)/S(k_min) - 1 ) / (2 sin(pi/L))

    IKI SEKIL KABUL EDILIR:
    - `(L, L, n_component)` -- tek konfigurasyon
    - `(n_config, L, L, n_component)` -- TOPLULUK; yapi faktoru ONCE
      ortalanir, oran SONRA kurulur.

    TOPLULUK SEKLI PLANDA YOKTU ve OLCULEREK eklendi. `S(k)` bir topluluk
    ortalamasidir; tek akistan okunan bir oran paydasi kadar gurultuludur.
    Olculdu (Ising L = 32, T = T_c, 5 tohum): TEK konfigurasyonda
    xi = 3.644, 17.735, 30.337, 36.768, 150.562 -- 41 KAT yayilim; ayni 5
    tohumun 200'er konfigurasyonluk toplulugunda 29.437, 30.773, 26.668,
    30.975, 33.078 -- %11 yayilim.

    ORAN <= 0 ISE None. `max(oran, 0)` ile kirpmak YASAK: sifir "korelasyon
    yok" diye okunur, oysa gercek hukum "bu tahminci burada bir sey
    soylemez". Ikisi cok farkli iddiadir. Olculdu -- tamamen ILISKISIZ +-1
    kafeste (gercek xi = 0) 200 tohumun L = 16'da 123'u, L = 32'de 124'u
    `None` veriyor; kalan yarisi ise SAHTE bir xi = 4.418 / 7.791 uretiyor.
    Kirpma yapilsaydi bu kafeslerin yarisi "xi = 0, dogru!" diye okunurdu ve
    dogru olmasi TESADUF olurdu.

    GECERLILIK PENCERESI -- BEYAN EDILIYOR: tahminci `S(0) = |sum s|^2`
    kullanir ve DUZENLI FAZDA bu, miknatislanmanin karesiyle DOMINE OLUR.
    Olculdu (Ising L = 32, T = 1.5, tek konfigurasyon, 5 tohum):
    xi = 613.966, 893.997, 787.245, 1431.310, 524.014 -- yani xi/L 16 ile 45
    arasinda. L = 32'lik bir kafeste xi = 800 bir korelasyon uzunlugu
    DEGILDIR. Pencere T >~ T_c'dir.

    'BAGLI' (connected) SURUM NEDEN YOK -- CEBIRSEL: tek konfigurasyondan
    site ortalamasini cikarmak sum(s - s_ort) = 0 yapar, yani S(0) TAM SIFIR
    olur ve oran HER ZAMAN tam -1 cikar. Olculdu: T = 1.5 / T_c / 3.5'te
    5'er tohumun 15'inde de oran = -1.000000e+00, yani bagli surum HER
    konfigurasyonda `None` doner ve hicbir sey soylemez. 'Bagli' icin <s>
    TOPLULUK ortalamasi olmak zorundadir; tek konfigurasyonda oyle bir sey
    yoktur ve topluluk seklinde de S(0) ayni sekilde sifirlanacagi icin
    uygulanmaz.
    """
    a = np.asarray(spins, dtype=float)
    if a.ndim == 3:
        a = a[None, ...]
    if a.ndim != 4:
        raise InvalidInputError(
            "spins (L, L, n_component) veya (n_config, L, L, n_component) "
            f"olmali, verilen sekil: {np.asarray(spins).shape}"
        )
    if a.shape[1] != a.shape[2]:
        raise InvalidInputError(f"kafes kare olmali, verilen sekil: {a.shape}")
    L = int(a.shape[1])
    if L < 3:
        # L = 2'de k_min = 1 ile k_max cakisir (Nyquist); oran fizik degil
        # ortu (aliasing) okur.
        raise InvalidInputError(
            f"ikinci moment tahmincisi L >= 3 gerektirir, verilen: L = {L}. "
            "L = 2'de en kucuk sifirdan farkli k, Nyquist k'siyla cakisir."
        )
    if not np.all(np.isfinite(a)):
        raise InvalidInputError("spin dizisi sonlu olmali")

    donusum = np.fft.fft2(a, axes=(1, 2))
    guc = np.sum(np.abs(donusum) ** 2, axis=3)
    s_sifir = float(np.mean(guc[:, 0, 0]))
    s_kmin = 0.5 * float(np.mean(guc[:, 1, 0] + guc[:, 0, 1]))
    if s_kmin <= 0.0:
        return None
    oran = s_sifir / s_kmin - 1.0
    if oran <= 0.0:
        return None
    return float(np.sqrt(oran) / (2.0 * np.sin(np.pi / L)))


def binning_error(series, min_blocks: int = DEFAULT_MIN_BLOCKS) -> BinningResult:
    """Blok ortalamasiyla hata; PLATONUN OLUSUP OLUSMADIGINI da dondurur.

    MC ornekleri ILISKILIDIR. Blok boyutu buyudukce blok ortalamalari
    bagimsizlasir ve SEM artar; iliski tamamen silindiginde egri PLATOYA
    oturur. Plato OLUSMAMISSA dondurulen sayi hatanin kendisi degil bir ALT
    SINIRIDIR -- ve cagiran bunu `plateau_reached` alanindan bilmek zorunda.

    `error = max(errors)` -- OLCULEREK SECILDI, `errors[-1]` DEGIL:
    - Bagimsiz seride (gercek deger naif SEM'dir, cebirsel): `max` 1.039123
      +- 0.046507, `errors[-1]` 0.965151 +- 0.095293. Yani `max` %4 YUKARI
      yanli, `errors[-1]` %3.5 ASAGI yanli ve sacilimi IKI KAT.
    - Wolff L = 32, T_c, 8000 olcum (gercek deger 24 bagimsiz zincirden
      5.2827e-03): `max` 0.939 x gercek (sacilim %9.1), `errors[-1]`
      0.917 x gercek (sacilim %11.4). `max` hem daha yakin hem daha dar.
    - Metropolis'in ayni penceresinde egri tekduze arttigi icin ikisi ZATEN
      ayni sayidir.
    Bir hata cubugunda yukari yanlilik DOGRU yondur; asagi yanlilik, cubugu
    oldugundan dar gostermenin kibar halidir.

    `min_blocks` blok sayisinin ALT siniridir: son basamakta en az bu kadar
    blok kalir. Varsayilan 32'nin gerekcesi modul dokstringindeki taramadir.
    """
    a = _seri(series, "seri")
    if int(min_blocks) != min_blocks or min_blocks < 3:
        raise InvalidInputError(
            f"min_blocks en az 3 olmali (plato olcutu uc nokta ister), "
            f"verilen: {min_blocks}"
        )
    if a.size < min_blocks:
        raise InvalidInputError(
            f"seri en az min_blocks = {min_blocks} eleman icermeli, "
            f"verilen uzunluk: {a.size}"
        )

    boyutlar: list[int] = []
    hatalar: list[float] = []
    b = 1
    while a.size // b >= min_blocks:
        m = a.size // b
        bloklar = a[: m * b].reshape(m, b).mean(axis=1)
        boyutlar.append(b)
        hatalar.append(float(bloklar.std(ddof=1) / np.sqrt(m)))
        b *= 2

    plato = _plato_var_mi(a.size, boyutlar, hatalar)
    hata = max(hatalar)
    # tau_int = 1/2 (SEM_plato / SEM_naif)^2. hatalar[0] naif SEM'dir ve
    # yalniz burada, oran icinde kullanilir -- disariya `error` olarak ASLA
    # cikmaz.
    tau = 0.5 * (hata / hatalar[0]) ** 2 if hatalar[0] > 0.0 else float("inf")

    return BinningResult(
        error=hata,
        plateau_reached=plato,
        block_sizes=tuple(boyutlar),
        errors=tuple(hatalar),
        tau_int=tau,
    )


def _plato_var_mi(n: int, boyutlar: list[int], hatalar: list[float]) -> bool:
    """Son nokta, uc onceki noktadan ISTATISTIKSEL OLARAK anlamli mi yuksek?

        e[-1] - e[-3]  <=  z * e[-1] * sqrt(sig[-1]^2 + sig[-3]^2)

    `sig_i = 1/sqrt(2(K_i - 1))` bir std tahmininin bagil sacilimidir.
    Planin "son uc hata %5 icinde mi" olcutunun yerini alir: %5, kendi
    sacilimi %6-13 olan sayilari karsilastiran ELLE SECILMIS bir esikti ve
    olculdugunde Wolff'un gercek platosunu 24 tohumun 21'inde kaciriyordu.

    UC NOKTA GEREKIR; daha azinda plato iddia edilmez (`False`).
    """
    if len(hatalar) < 3 or hatalar[-1] <= 0.0:
        return False
    K_son = n // boyutlar[-1]
    K_onceki = n // boyutlar[-3]
    sig_son = 1.0 / np.sqrt(2.0 * (K_son - 1.0))
    sig_onceki = 1.0 / np.sqrt(2.0 * (K_onceki - 1.0))
    esik = _PLATO_Z * hatalar[-1] * float(np.hypot(sig_son, sig_onceki))
    return bool(hatalar[-1] - hatalar[-3] <= esik)
