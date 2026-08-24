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

    `mu_r` bu modulde her yerde 1'dir ve oyle SEVK EDILIYOR; alan olarak
    tutulmasinin sebebi `yee.py`nin ortami bilmemesi -- cekirdek "mu 1'dir"
    varsayimini iceride tasisaydi, manyetik ortam eklemek cekirdegi
    degistirmeyi gerektirirdi.
    """

    eps_r: np.ndarray
    mu_r: np.ndarray
    ey_kural: str = "aritmetik"

    def __post_init__(self) -> None:
        object.__setattr__(self, "eps_r", np.asarray(self.eps_r, dtype=float))
        object.__setattr__(self, "mu_r", np.asarray(self.mu_r, dtype=float))
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
        for ad, dizi in (("eps_r", self.eps_r), ("mu_r", self.mu_r)):
            if not np.all(dizi > 0.0):
                raise InvalidInputError(
                    f"{ad} her yerde pozitif olmali, en kucuk deger: "
                    f"{float(np.min(dizi)):.6g}"
                )

    @property
    def cells(self) -> int:
        return int(self.eps_r.shape[0])

    @property
    def eps_min(self) -> float:
        """EN HIZLI dalganin gordugu izinlilik.

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

    def eps_yari(self) -> np.ndarray:
        """(i+1/2)'de TEGET E bileseninin gordugu eps -- kural `ey_kural`."""
        return _yari_hucre(self.eps_r, self.ey_kural)

    def mu_yari(self) -> np.ndarray:
        """(i+1/2)'de H'nin gordugu mu -- HER ZAMAN ARITMETIK.

        Bu `ey_kural`a BAGLI DEGILDIR ve olmamalidir: `ey_kural` teget E'nin
        arayuzdeki sureklilik kuraliyla ilgilidir, H ile degil. Ayrica bu
        modulde `mu_r` her yerde 1 oldugu icin dort kural da sayi sayi ayni
        diziyi verir; kuralin buraya baglanmasi SESSIZ bir bag olurdu.
        """
        return _yari_hucre(self.mu_r, "aritmetik")
