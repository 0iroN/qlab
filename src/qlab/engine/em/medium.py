"""Ortam: `eps_r` / `mu_r` haritalari, ve ARAYUZ ORTALAMASI ACIK PARAMETRE.

Bu dosya GEOMETRI BILIR (nerede ne var), `yee.py` bilmez. Ayrim bilincli:
`yee.py`nin gordugu tek sey buradan cikan ALANLARDIR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.errors import InvalidInputError

EY_KURALLARI = ("aritmetik", "harmonik", "sol", "sag")


def _yari_hucre(dizi: np.ndarray, kural: str) -> np.ndarray:
    """`dizi[i]` ve `dizi[i+1]`den yarim hucre (i+1/2) degeri."""
    a, b = dizi[:-1], dizi[1:]
    if kural == "aritmetik":
        return 0.5 * (a + b)
    if kural == "harmonik":
        return 2.0 / (1.0 / a + 1.0 / b)
    if kural == "sol":
        return np.array(a, copy=True)
    if kural == "sag":
        return np.array(b, copy=True)
    raise InvalidInputError(
        f"ey_kural {EY_KURALLARI} icinden olmali, verilen: {kural!r}"
    )


@dataclass(frozen=True, eq=False)
class Medium:
    """Bagil izinlilik ve gecirgenlik haritalari.

    `eq=False`: alanlar ndarray; gerekce `grid.FieldState`dekiyle ayni.

    ARAYUZ ORTALAMASI ACIK PARAMETREDIR (`ey_kural`) ve bu bir MODELLEME
    SECIMIDIR, GURULTU EKSENI DEGIL. Yarim hucrede (i+1/2) oturan TEGET E
    bileseninin gordugu eps, iki komsu hucreden bir kuralla uretilir ve secim
    olculdu -- TIY faz farki `delta_ps`, sevk penceresinde (eps1 = 4,
    eps2 = 1, theta = 45 der, lam = 40, S = 0.4, N = 3600, npml = 60,
    sgf = 9, dikdortgen pencere) surekli-ortam degerinden su kadar sapiyor:

        aritmetik  +0.156403 der      <- SEVK EDILEN
        harmonik   +5.346570 der
        sol        -8.127598 der
        sag        +8.889958 der

    ARITMETIK TEGET BILESEN ICIN DOGRUDUR (teget E arayuzde SUREKLIDIR;
    sureksiz olan normal bilesendir, onun dogru kurali harmoniktir) ve olcum
    bunu soyluyor: dogru kural teoriye 0.156 der, otekiler 5.3 - 8.9 der
    sapiyor.

    BU EKSEN sigma'YA KATILMAZ. Olculdu: yalniz analiz/ayriklastirma
    dugmeleriyle sigma_analiz = 0.002189 der (n = 15); `ey_kural` dort
    degerini de havuza katinca sigma = 3.162927 der, yani 1445 KAT sisik.
    Bir modelleme secimini gurultu eksenine cevirmek, olcumun cozunurlugunu
    yok etmek demektir.

    `mu_r` SENARYO YOLUNDA her yerde 1'dir ve oyle SEVK EDILIYOR (kuruculari
    np.ones verir, EMScenario'da mu yok); alan olarak tutulmasinin sebebi
    `yee.py`nin ortami bilmemesi -- cekirdek "mu 1'dir" varsayimini iceride
    tasisaydi, manyetik ortam eklemek cekirdegi degistirmeyi gerektirirdi.
    Motor yolu mu_r != 1'i KABUL EDER ve kararlilik kapilari onu okur
    (`yee._etkin_courant_kapisi`, `oblique._egik_p_kesin_esik`).

    DIZILER KOPYALANIR VE SALT OKUNURDUR (G13, 12.10.2026). Kapi (pozitif ve
    sonlu) yalniz kurulusta kosar; ESKIDEN `np.asarray` cagiranin dizisini
    paylasiyordu ve dizi yazilabilirdi. OLCULDU (HEAD e262cab, kamu yolu
    `run_1d`, vakum N = 200, PML 15, S = 0.5, 600 adim): kurulumdan sonra
    `eps_r[100] = mu_r[100] = -4` (cift negatif: hucre carpimi +16, etkin
    Courant kapisi gecer) sessizce KABUL ediliyor ve son alan 2.46e+49'a
    cikiyordu (referans 8.3e-04); cagiranin `eps` dizisine sonradan yazilan
    -4 `m.eps_r`de gorunuyordu; `eps_r = inf` kurulusta geciyordu. Simdi
    yazma `ValueError` (numpy'nin salt-okunur hatasi), takma ad kopuk, inf
    `InvalidInputError`.

    SALT OKUNURLUK GERI ACILAMAZ: dizi bir `bytes` nesnesinin uzerine kurulur
    (`np.frombuffer`), yani zincirin dibinde yazilabilir bir ndarray yoktur.
    `setflags(write=False)` YETMIYORDU: `m.eps_r.setflags(write=True)` diziyi
    yeniden yazilabilir yapiyor ve ayni cift negatif hucre (PML 15, S = 0.5,
    600 adim) son 50 kaydi 4.2e+40'a cikariyordu (referans 3.7e-04). Salt
    okunur bir tabanin salt okunur GORUNUMU de yetmez: `.base.setflags(write=
    True)` once tabani, sonra gorunumu acar (numpy 2.5.1, olculdu). `bytes`
    tabaninda ikisi de `ValueError`. Degerler ve sekil kopyanin aynisidir.
    SINIR: `copy.deepcopy(m)` `__post_init__`i kosmaz ve yazilabilir dizi
    verir (numpy'nin kopyasi) -- bu yol kapali degildir.
    """

    eps_r: np.ndarray
    mu_r: np.ndarray
    ey_kural: str = "aritmetik"

    def __post_init__(self) -> None:
        for ad in ("eps_r", "mu_r"):
            kopya = np.array(getattr(self, ad), dtype=float)  # KOPYA: takma ad kopar
            # `bytes` tabani: salt okunurluk `setflags(write=True)` ile de geri acilamaz.
            dizi = np.frombuffer(kopya.tobytes(), dtype=float).reshape(kopya.shape)
            object.__setattr__(self, ad, dizi)
        if self.eps_r.shape != self.mu_r.shape:
            raise InvalidInputError(
                f"eps_r ve mu_r ayni sekilde olmali, verilen: "
                f"{self.eps_r.shape} ve {self.mu_r.shape}"
            )
        if self.ey_kural not in EY_KURALLARI:
            raise InvalidInputError(
                f"ey_kural {EY_KURALLARI} icinden olmali, verilen: "
                f"{self.ey_kural!r}"
            )
        if self.eps_r.ndim not in (1, 2):
            raise InvalidInputError(
                f"ortam 1B ya da 2B olmali, verilen sekil: {self.eps_r.shape}"
            )
        for ad, dizi in (("eps_r", self.eps_r), ("mu_r", self.mu_r)):
            if not np.all(dizi > 0.0):
                raise InvalidInputError(
                    f"{ad} her yerde pozitif olmali, en kucuk deger: "
                    f"{float(np.min(dizi)):.6g}"
                )
            if not np.all(np.isfinite(dizi)):
                raise InvalidInputError(
                    f"{ad} her yerde sonlu olmali (inf geldi; G13)"
                )

    @property
    def cells(self) -> int:
        """1B hucre sayisi. 2B ortamda TANIMSIZDIR ve oyle davranir.

        `shape[0]`i sessizce dondurmek 2B'de YANLIS bir sayiyi dogru gibi
        gosterirdi (nx, ny degil). Task 4'te 2B sevk edilince bu ayrim
        olculebilir hale geldi: `yee_coefficients_2d` `cells`e HIC dokunmaz,
        `shape`e dokunur -- ve iki kod yolunun ortamdan gordugu ad kumeleri
        testte ayri ayri sayiliyor.
        """
        if self.eps_r.ndim != 1:
            raise InvalidInputError(
                f"`cells` yalniz 1B ortamda tanimli; bu ortam {self.eps_r.ndim}B "
                f"(sekil {self.shape}). 2B'de `shape` kullanilir."
            )
        return int(self.eps_r.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        """Ortamin izgara sekli -- 1B'de `(N,)`, 2B'de `(nx, ny)`."""
        return tuple(int(n) for n in self.eps_r.shape)

    @property
    def eps_min(self) -> float:
        """En kucuk izinlilik -- mu_r = 1'de EN HIZLI dalganin gordugu. mu_r != 1'de
        en hizli dalgayi hucre basina eps_r*mu_r belirler (`yee._etkin_courant_kapisi`).

        `courant_limit` bu sayiyi KULLANMAZ ve kullanmamasi bir secimdir
        (bkz. grid.courant_limit): `S` en hizli dalga hiziyla tanimlidir,
        yani ham kararlilik esigi degil NORMALIZE edilmis esik sinanir.
        """
        return float(np.min(self.eps_r))

    @classmethod
    def vacuum(cls, cells: int, *, ey_kural: str = "aritmetik") -> Medium:
        return cls(np.ones(cells), np.ones(cells), ey_kural)

    @classmethod
    def half_space(
        cls,
        cells: int,
        *,
        interface: int,
        eps1: float = 1.0,
        eps2: float = 1.0,
        ey_kural: str = "aritmetik",
    ) -> Medium:
        """`interface` hucresinden itibaren eps2 olan iki-bolgeli ortam.

        Arayuzun HANGI hucreden basladigi bir ayriklastirma secimidir ve
        beyan edilir: `interface` INDISLI hucre ARTIK ortam 2'dir.
        """
        if not 1 <= interface < cells:
            raise InvalidInputError(
                f"interface 1 ile {cells - 1} arasinda olmali, verilen: {interface}"
            )
        eps = np.full(cells, float(eps1))
        eps[interface:] = float(eps2)
        return cls(eps, np.ones(cells), ey_kural)

    @classmethod
    def uniform_2d(
        cls, nx: int, ny: int, *, eps_r: float = 1.0, ey_kural: str = "aritmetik"
    ) -> Medium:
        """Duzgun 2B ortam. ADI `vacuum_2d` DEGIL -- `eps_r` girilebiliyor.

        Bessel oranlarinin `eps_r`den bagimsiz oldugu sinavi tam olarak bu
        kurucuyu 1.0 ve 2.25 ile cagirir; kurucu "vakum" diye adlandirilsaydi
        adi yaptigi seye YALAN SOYLERDI.
        """
        if nx < 3 or ny < 3:
            raise InvalidInputError(
                f"2B izgara her yonde en az 3 hucre olmali, verilen: {nx} x {ny}"
            )
        if eps_r <= 0.0:
            raise InvalidInputError(f"eps_r > 0 olmali, verilen: {eps_r}")
        return cls(np.full((nx, ny), float(eps_r)), np.ones((nx, ny)), ey_kural)

    def mu_yari_2d(self, axis: int) -> np.ndarray:
        """2B'de yarim hucre `mu` -- `axis` KAYDIRMANIN YONUDUR.

        `axis = 0` -> Hy(i+1/2, j), sekil (nx-1, ny)
        `axis = 1` -> Hx(i, j+1/2), sekil (nx, ny-1)

        1B'deki `mu_yari` gibi HER ZAMAN ARITMETIKTIR ve `ey_kural`a bagli
        DEGILDIR; gerekce `mu_yari`nin dokstringindedir ve burada tekrar
        edilmiyor.

        DUAL TE_z'DE (G1-p, 10.10.2026; mu' = eps) bu dizi Ex/Ey'nin gordugu
        eps'tir: aritmetik = teget E'nin kurali = egik p'nin sevk `ey_kural`i.
        x ekseninde (axis = 0) harmonik yapildiginda hizli EM seridinin TAMAMI
        geciyordu (10.10.2026'da 246 test; olculdu, elestirmen; dogrulayici
        yeniden olctu); artik `tests/em/test_g1p_egik_2b_te.py`de yerli TE_z
        (T0p), transpoz (T0y) ve kimlik (T1p) testleri duser (yavas seritte
        dogrudan 2B delta_ps testi T3p de). y ekseninde (axis = 1) harmonik
        ve 2B `dbx`te sol hucre mu'su 10.10.2026'da hizli seritten geciyordu;
        onlari T0p ve 11.10.2026'dan beri T0y dusurur (y-degismez T1p/T3p kor).
        """
        if self.mu_r.ndim != 2:
            raise InvalidInputError(
                f"mu_yari_2d yalniz 2B ortamda tanimli, verilen sekil: {self.shape}"
            )
        if axis == 0:
            return 0.5 * (self.mu_r[:-1, :] + self.mu_r[1:, :])
        if axis == 1:
            return 0.5 * (self.mu_r[:, :-1] + self.mu_r[:, 1:])
        raise InvalidInputError(f"axis 0 ya da 1 olmali, verilen: {axis}")

    def eps_yari(self) -> np.ndarray:
        """(i+1/2)'de TEGET E bileseninin gordugu eps -- kural `ey_kural`."""
        return _yari_hucre(self.eps_r, self.ey_kural)

    def mu_yari(self) -> np.ndarray:
        """(i+1/2)'de H'nin gordugu mu -- HER ZAMAN ARITMETIK.

        Bu `ey_kural`a BAGLI DEGILDIR ve olmamalidir: `ey_kural` teget E'nin
        arayuzdeki sureklilik kuraliyla ilgilidir, H ile degil. `mu_r` = 1'de dort
        kural sayi sayi ayni diziyi verir ve kuralin buraya baglanmasi SESSIZ bir
        bag olurdu. `mu_r` konuma bagliyken ise egik p'de Hz'in TAM hucre mu_r'si
        ile Ey'nin `eps_yari`si eslenir ve kararliligi o eslesme belirler (olculdu
        02.10.2026; kapi: `oblique._egik_p_kesin_esik`).
        """
        return _yari_hucre(self.mu_r, "aritmetik")
