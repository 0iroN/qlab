"""Ussel radyal yasa ve iptalsiz kosinus kuyrugu -- HARTREE ve BOHR biriminde.

    f(d)  = genlik * exp(-us * (d/d0 - 1)) * fc(d)
    fc(d) = 1                                   d <= r_on
          = sin^2(pi * (r_cut - d) / (2 w))     r_on < d < r_cut,  w = r_cut - r_on
          = 0                                   d >= r_cut

ROL CAGRI YERINDE ADLANDIRILIR. Bu genel bir yasadir; ayni bicim iki rolde
kullanilir: hoplama rolundeki ornegin adi `ss_sigma`, cift itmesi rolundekinin
adi `itme`dir. Hoplama rolunde bu, Slater-Koster tablosunun YALNIZ ss-sigma
satirinin radyal parcasidir: atom basina tek s orbitali oldugu icin yon
kosinusu carpani OZDES 1'dir ve bu dosya bir SK modeli DEGILDIR (p/d yok).
Itme rolunde ise bir SK integrali bile degildir. Bu yuzden hicbir sinif ya da
fonksiyon adinda "SlaterKoster" gecmez.

BIRIMLER ADDA YAZILIDIR (`genlik_hartree`, `d0_bohr`, `r_on_bohr`,
`r_cut_bohr`) ve bu modul deponun birim donusum modulunu IMPORT ETMEZ.
Gerekce olculdu (spec 5.2): tb paketinden bir dosya o modulu import edince
units ekseninde tb'nin derecesi 2'den 3'e cikiyor; Kelvin -> hartree
donusumu md fabrikasinin isidir. (Modulun tam adi burada BILEREK yazilmaz:
`tests/bridge/test_kova_tablosu.py` units eksenini duz metinle de sayar ve
md disindaki paketlerde metin sayiminin `ast` sayimina esit olmasini ister;
olculdu, adi dokstringde yazmak tb'nin metin sayimini 2'den 3'e cikariyor.)

BAGIMSIZLIK (spec M10). Bu dosya hf paketini, `tb.downfold`u ve
`tb.factory`yi import etmez: import kumesi tam esitlikle
`tests/tb/test_radyal.py`de, yasak adlar `tests/tb/test_mermin.py`de
(`test_M10_*`) kilitlidir.

IPTALSIZ BICIM. Ayni kuyruk `0.5 * (1 + cos(pi (d - r_on)/w))` diye de
yazilabilir ve matematikte OZDESTIR; ama r_cut yakininda 1 + cos(~pi) iki
buyuk sayinin farkidir ve birbirini goturur. OLCULDU (24.09.2026, Python
3.14.7, numpy 2.5.1; w = 0.8, r_cut = 3.8, d = r_cut - e, referans sin^2'nin
u^6'ya kadar Taylor acilimi): e = 1e-9'da iptalli bicim TAM 0.0 veriyor,
gercek deger 3.855315e-18; e = 1e-8'de goreli hatasi 7.90e-03, e = 1e-6'da
2.40e-06. Bu bicim e in {1e-9 ... 1e-3} boyunca referanstan en cok 3.3e-16
(goreli) sapiyor. Turev de iptalsiz yazilir:
dfc = -(pi/(2w)) * sin(pi (r_cut - d)/w).

C^1, C^2 DEGIL. Deger ve birinci turev r_on ve r_cut'ta sureklidir; ikinci
turev sicrar: fc''(r_on+) = -pi^2/(2 w^2), fc''(r_cut-) = +pi^2/(2 w^2).
Iki dikiste de f''(x0+) - f''(x0-) = -g(x0) * pi^2/(2 w^2), g = genlik *
exp(-us (x0/d0 - 1)). Ortak pencerede (t: -0.1/2/2, itme: 0.05/2/6) bu,
kT = 0 dimerde (E = 2 t + itme) r_on'da +0.54812, r_cut'ta +0.25317 eder.
Sonucu: dikisi kapsayan merkezi fark O(h)'dir, O(h^2) degil; sonlu fark
testleri dikisten uzak geometri kullanmak ve bunu kendisi dogrulamak
zorundadir. `tests/tb/test_radyal.py` dikisteki tek yanli turev farkinin h
ile DOGRUSAL indigini (C^1 kaniti) olcer.

SERT CEKIRDEK YOK. Ussel yasa d = 0'da sonludur; cakisik atom kapisi bu
yasanin degil, onu kullanan cagri yerinin isidir. Yasa yalniz d >= 0 ve sonlu
mesafe kabul eder; exp tasarsa (buyuk `us`, kucuk d) sonuc sessizce inf/nan
donmez, `InvalidInputError` firlatilir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.errors import InvalidInputError


def _sonlu_skaler(ad: str, deger: object) -> float:
    """Tek bir sonlu gercel sayi; aksi halde `InvalidInputError`."""
    if isinstance(deger, (str, bytes)) or np.ndim(deger) != 0:
        raise InvalidInputError(f"{ad} tek bir sayi olmali: {deger!r}")
    try:
        sayi = float(deger)  # type: ignore[arg-type]
    except (TypeError, ValueError) as hata:
        raise InvalidInputError(f"{ad} sayiya cevrilemiyor: {deger!r}") from hata
    if not np.isfinite(sayi):
        raise InvalidInputError(f"{ad} sonlu olmali: {sayi}")
    return sayi


def _kuyruk_sinirlari(r_on_bohr: object, r_cut_bohr: object) -> tuple[float, float]:
    r_on = _sonlu_skaler("r_on_bohr", r_on_bohr)
    r_cut = _sonlu_skaler("r_cut_bohr", r_cut_bohr)
    if not r_on < r_cut:
        raise InvalidInputError(
            f"r_on_bohr < r_cut_bohr olmali: r_on={r_on}, r_cut={r_cut} "
            "(kuyruk genisligi pozitif olmadan kesme surekli olamaz)"
        )
    return r_on, r_cut


def _mesafe_dizisi(d: object) -> np.ndarray:
    mesafe = np.asarray(d, dtype=float)
    if not np.all(np.isfinite(mesafe)):
        raise InvalidInputError("mesafe sonlu olmali (nan/inf geldi)")
    if np.any(mesafe < 0.0):
        raise InvalidInputError(
            f"mesafe negatif olamaz: en kucugu {float(np.min(mesafe))}"
        )
    return mesafe


def kosinus_kuyrugu(
    d: object, r_on_bohr: float, r_cut_bohr: float
) -> tuple[np.ndarray, np.ndarray]:
    """(fc, dfc/dd): d <= r_on'da (1, 0), d >= r_cut'ta (0, 0), arada sin^2.

    IPTALSIZ bicim (modul dokstringi): fc = sin^2(pi (r_cut - d)/(2w)),
    dfc = -(pi/(2w)) * sin(pi (r_cut - d)/w). Donen diziler `d` ile ayni
    sekildedir; skaler girdi 0 boyutlu dizi dondurur.
    """
    r_on, r_cut = _kuyruk_sinirlari(r_on_bohr, r_cut_bohr)
    mesafe = _mesafe_dizisi(d)
    yarim_dalga = np.pi / (2.0 * (r_cut - r_on))
    ic = mesafe <= r_on
    kuyruk = (mesafe > r_on) & (mesafe < r_cut)
    u = yarim_dalga * (r_cut - mesafe)
    fc = np.where(ic, 1.0, np.where(kuyruk, np.sin(u) ** 2, 0.0))
    dfc = np.where(kuyruk, -yarim_dalga * np.sin(2.0 * u), 0.0)
    return fc, dfc


@dataclass(frozen=True)
class UsselRadyalYasa:
    """f(d) = genlik * exp(-us * (d/d0 - 1)) * fc(d); rol cagri yerinde adlanir.

    Parametrelerin hepsi SECIMDIR, turetilmemistir (spec B4): bu yasadan geri
    okunan hicbir sayi (denge bagi, dimerlesme genligi, esik) fizik iddiasi
    olarak sevk edilemez. Dogaya karsi dogrulayan bir kaynak YOKTUR.
    """

    genlik_hartree: float
    d0_bohr: float
    us: float
    r_on_bohr: float
    r_cut_bohr: float

    def __post_init__(self) -> None:
        for ad in ("genlik_hartree", "d0_bohr", "us"):
            object.__setattr__(self, ad, _sonlu_skaler(ad, getattr(self, ad)))
        r_on, r_cut = _kuyruk_sinirlari(self.r_on_bohr, self.r_cut_bohr)
        object.__setattr__(self, "r_on_bohr", r_on)
        object.__setattr__(self, "r_cut_bohr", r_cut)
        if not self.d0_bohr > 0.0:
            raise InvalidInputError(f"d0_bohr > 0 olmali: {self.d0_bohr}")
        if not self.us >= 0.0:
            raise InvalidInputError(
                f"us >= 0 olmali: {self.us} (negatif us uzakta BUYUYEN bir yasa verir)"
            )

    def deger_ve_turev(self, d: object) -> tuple[np.ndarray, np.ndarray]:
        """(f, df/dd), `d` ile ayni sekilde. HARTREE ve HARTREE/BOHR."""
        fc, dfc = kosinus_kuyrugu(d, self.r_on_bohr, self.r_cut_bohr)
        mesafe = np.asarray(d, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            g = self.genlik_hartree * np.exp(-self.us * (mesafe / self.d0_bohr - 1.0))
            deger = g * fc
            turev = (-self.us / self.d0_bohr) * g * fc + g * dfc
        if not (np.all(np.isfinite(deger)) and np.all(np.isfinite(turev))):
            raise InvalidInputError(
                f"ussel yasa tasti: us={self.us}, d0={self.d0_bohr}, en kucuk "
                f"mesafe {float(np.min(mesafe))} (exp(us) sonlu degil)"
            )
        return deger, turev
