"""Tek orbitalli ss-sigma TB kuvvet alani -- `ForceField` sozlesmesinin besinci uygulamasi.

    (konumlar: (N, 3) bohr) -> (Omega: hartree, F = -grad(Omega): (N, 3) hartree/bohr)

NE SARAR. `qlab.engine.tb.mermin.tek_orbital_mermin`i; enerji ve kuvvetin
TAMAMI oradan gelir. Bu dosya yasanin ikinci bir kopyasini YAZAMAZ: icinde
ustel ya da trigonometrik cagri yoktur (`tests/md/test_tb_kuvvet.py` `ast`
ile kilitler; on olcumun V4 varyanti yasayi burada yeniden yaziyordu).
Radyal yasa `qlab.engine.tb.radyal.UsselRadyalYasa` nesnesidir ve kafes yolu
(`tb.hamiltonian.distance_model`) ile AYNI nesneyi kullanir (spec M6).

MODEL SINIRI (spec 1, B6). Hoplama rolundeki yasa Slater-Koster tablosunun
YALNIZ ss-sigma satiridir. Tek s orbitali/atom: aci carpani OZDES 1, bu bir
SK modeli DEGILDIR. p/d yok. Spin katlanmasi g_s = 2 sabit (manyetizma yok),
yuk oz-uyumu yok, tek tur; sinif onsite ALMAZ (tekbicim onsite Omega'yi tam
n_e c kaydirir, kuvveti degistirmez).

DONEN ENERJI MERMIN SERBEST ENERJISIDIR (spec B1): Omega = E_band - kT S +
E_rep ve kuvvet F = -grad(Omega)'dir, E = E_band + E_rep'in gradyani DEGIL.
NVE'de integrator bu yuzden KE + Omega'yi korur; KE + E KORUNMAZ, farki
kT dS'dir. (Termostatli kosuda toplam enerji zaten korunmaz; panel korunum
cumlesini bu yuzden yalniz NVE'nin beklenen hatasinda basar.)

kT_hartree > 0 ZORUNLUDUR (spec B2). kT = 0'da Fermi seviyesindeki dejenere
noktada gradyan yoktur, dejenerelik bir tolerans kararina baglanir, kopuk
parcalar arasinda kuvvet sicrar ve Fermi gecisli yorunge her geciste O(dt)
enerji sicramasi uretir. Statik kT = 0 `tek_orbital_mermin`de acik kalir;
MD sinifi onu kabul etmez. kT > 0'in bedeli beyanlidir: tek global mu
yerel olmayan bir ciftlenim uretir (buyuklugu 1/kT ile olceklenir) ve
O(dt^2) davranis ancak bir adimdaki aralik degisimi / kT kucukken geri
gelir; bu kosul analitik denetlenemez, dt'yi kullanici secer (spec B10).

BIRIM SINIRI (spec B7). Sinif HARTREE alir (`kT_hartree`); senaryo KELVIN
tasir (`electron_temperature_kelvin`) ve donusum `md/factory.py`de yapilir.
Bu dosya birim donusum modulunu IMPORT ETMEZ: md paketinin units ekseni 3
kalmalidir (spec 5.2; modulun tam adi `tests/bridge/test_kova_tablosu.py`
metin sayimi yuzunden burada BILEREK yazilmaz). TUZAK: Kelvin sayisi
hartree sanilip verilirse kuvvet SESSIZCE kuculur -- olculdu (24.09.2026):
zincir(16, 2, 0.05, 3), n_e = 16, L = 32, itme genligi 0 (yalniz bant),
max|F| kT = 0.002'de 1.054e-01, kT = 300'de 2.803e-05 (3759 kat). Savunma:
kT_hartree >= 1 hartree (315775 K) reddedilir. Kelvin cinsinden 1 K ve
ustu her sicaklik hartree yerine yazilinca bu kapiya takilir; 1 K'in altindaki
bir Kelvin sayisi (ornegin 0.5) ve ters tuzak (hartree sayisini Kelvin diye
yazmak: 0.002 K = 6.3e-09 hartree) YAKALANMAZ -- beyan.

PERIYODIK KUTU (spec B5). `kutu_x_bohr` verilirse yalniz x ekseni periyodik
(en-yakin-goruntu), y ve z acik; 2 max(r_cut) < L kesin kosuldur ve kurulusta
denetlenir. Kucuk kutu Gamma'da bir halka verir, kristal degil. Enine
hareket SINANMADI: 1B pencereler y = z = 0 alt-uzayinda kalir; on olcumde
a = 2 zincirinin yalniz x'te gevsetilmis geometrisi 3B'de bir eyerdi ve
enine kararsizlik bu depoda olculmedi.

PURUZSUZ KESME (spec B3). Yasanin kuyrugu (`tb/radyal.py`) r_on ile r_cut
arasinda sin^2'dir: enerji ve kuvvet r_cut'ta sureklidir ve sert kesmenin
enerji basamagi kapanir -- olculdu (tests/tb/test_mermin.py M3; dimer,
n_e = 2, kT = 1e-3, kutusuz): dE(e) = Omega(r_cut - e) - Omega(r_cut + e)
icin dE(1e-3)/dE(1e-4) = 100.0389, yani basamak e^2 ile kapaniyor; sert
kesmeli `uniform_model` ayni dimerde sabit 2|t| = 0.2000 sicrar. Kuyruk
C^1'dir, C^2 DEGIL: ikinci turev r_on ve r_cut'ta sicrar ve dikisi kapsayan
merkezi fark O(h)'dir.

SERT CEKIRDEK YOK (spec B9). Ussel yasa d = 0'da sonludur ve duvar da
sonludur (Morse beyaninin esi); cakisik cift (en-yakin-goruntu sonrasi
d = 0) `InvalidInputError`. d = 0'da bir V-KUYUSU olabilir: dimerde (n_e =
2, kT -> 0, d < r_on) kosul 2|t0| k e^k > A krep e^krep'tir (t0, k hoplama
yasasinin genligi ve ussu; A, krep itmeninki); krep <= k ve 2|t0| k > A
krep bunun YETER kosuludur ve o zaman Omega d0'dan d = 0'a bariyersiz
iner. Sinif bu parametreleri REDDETMEZ; kuyunun dibinde kuvvet gradyan
degildir. Makine testi (`tests/md/test_tb_kuvvet.py`, B9): ortak pencerede
(itme ussu 6) duvar sonlu ve yuksek; itme ussu 2 (krep = k) ve A = 0.04,
krep = 3 (krep > k: dar kosul GEREK kosul degil) dimerde Omega d0'dan
d = 0'a tekduze iner.

PARAMETRELER SECIMDIR (spec B4): bu kenarda DOGA CIPASI YOKTUR.

ENERJININ SIFIRI KEYFIDIR (spec B8). Tekbicim bir onsite Omega'yi n_e c
kaydirir, yorungeyi degistirmez; bu yuzden TB'nin NVE surukleme olcutu
|blok[0]|'a degil kaydedilen en buyuk kinetik enerjiye boler
(`md/factory.py::energy_drift`; esik ve olcumleri orada).

BAGIMSIZLIK (spec M10). Kenar HF'e CAGRI duzeyinde dokunmaz: bu dosya hf
paketini, `tb.downfold`u ve `tb.factory`yi import etmez (`tb.factory`
downfold uzerinden 8 hf modulu yukler); importu tek basina SIFIR hf modulu
yukler ve TB MD kosusunda `hf.scf.solve` SIFIR kez cagrilir. Downfold'dan
yasa almak bu yuzden yasaktir (on olcumde o yasayla TB payi %0.000 cikti:
totoloji). Kilitler `tests/md/test_tb_kuvvet.py`dedir (`test_M10_*`: `ast`
import kumesi, ikinci kopya yasagi, alt surecte yuklenen modul kumesi,
`solve` sayaci + HF pozitif kontrolu).

MALIYET (spec elestirmeni P0-4; olculdu 24.09.2026, bu makine, 5 tekrarin
en kucugu, 1B zincir a = 2, kT = 2e-3, L = 2N): cagri basina N = 16'da
0.62 ms, 32'de 0.92 ms, 64'te 1.43 ms, 128'de 4.31 ms (vektorize
en-yakin-goruntu + tam `eigh`). Kafes yolu (`distance_model`) N = 16..128
boyunca 9.5-51 kat pahali olculdu (G3; spec 2.3'te N = 64 icin 22 kat);
MD yolu bu yuzden kafes yolundan GECMEZ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qlab.engine.tb.mermin import TekOrbitalSonuc, tek_orbital_mermin
from qlab.engine.tb.radyal import UsselRadyalYasa
from qlab.errors import InvalidInputError

#: Kelvin tuzagi kapisi, HARTREE. kT_hartree bu degere esit ya da buyukse
#: girdi reddedilir: 1 hartree = 315775 K ve bu modelin bant genisligi
#: ortak pencerede 4|t| = 0.4 hartree; spec pencerelerinin kT'si 1e-7 ile
#: 1e-2 arasinda. Kapinin isi Kelvin sayisinin hartree yerine yazilmasini
#: yakalamaktir (modul dokstringi, B7).
_KELVIN_TUZAGI_HARTREE = 1.0


def _sayi(ad: str, deger: object) -> float:
    """Tek bir sonlu gercel sayi (bool ve dizge DEGIL); aksi halde hata."""
    if isinstance(deger, (str, bytes, bool)) or np.ndim(deger) != 0:
        raise InvalidInputError(f"{ad} tek bir sayi olmali: {deger!r}")
    try:
        sayi = float(deger)  # type: ignore[arg-type]
    except (TypeError, ValueError) as hata:
        raise InvalidInputError(f"{ad} sayiya cevrilemiyor: {deger!r}") from hata
    if not np.isfinite(sayi):
        raise InvalidInputError(f"{ad} sonlu olmali: {sayi}")
    return sayi


@dataclass(frozen=True, eq=False)
class SsSigmaTBKuvveti:
    """ss-sigma TB kuvvet alani: (N, 3) konum -> (Omega, F = -grad Omega).

    `ss_sigma` hoplama, `itme` cift itmesi rolundeki yasadir (ikisi de
    `UsselRadyalYasa`). `n_elektron` sistemin toplam elektron sayisi (spin
    dahil, g_s = 2); `kT_hartree` elektron sicakligi, HARTREE ve > 0
    ZORUNLU; `kutu_x_bohr` None ise kutusuz. Ayrintilar modul dokstringinde
    (B1, B2, B5, B7, B9, B10).

    `eq=False`: sonuc dizileri tasiyan kardes siniflarin (`MDState`,
    `HartreeFockGradient`) gerekcesi; esitlik kimliktir.
    """

    ss_sigma: UsselRadyalYasa
    itme: UsselRadyalYasa
    n_elektron: float
    kT_hartree: float
    kutu_x_bohr: float | None = None

    def __post_init__(self) -> None:
        for ad in ("ss_sigma", "itme"):
            yasa = getattr(self, ad)
            if not isinstance(yasa, UsselRadyalYasa):
                raise InvalidInputError(
                    f"{ad} bir UsselRadyalYasa olmali: {type(yasa).__name__}"
                )
        n = _sayi("n_elektron", self.n_elektron)
        if n < 0.0:
            raise InvalidInputError(f"n_elektron >= 0 olmali: {n}")
        kT = _sayi("kT_hartree", self.kT_hartree)
        if not kT > 0.0:
            raise InvalidInputError(
                f"kT_hartree > 0 ZORUNLU (MD'de smearing sart, spec B2): {kT}. "
                "kT = 0'da Fermi seviyesindeki dejenere noktada kuvvet tanimsizdir; "
                "statik kT = 0 hesabi `tek_orbital_mermin` ile yapilir."
            )
        if kT >= _KELVIN_TUZAGI_HARTREE:
            raise InvalidInputError(
                f"kT_hartree = {kT} >= {_KELVIN_TUZAGI_HARTREE} hartree (315775 K): "
                "bu bir Kelvin sayisina benziyor. Sinif HARTREE alir; senaryoda "
                "sicaklik `electron_temperature_kelvin` ile verilir ve fabrika "
                "cevirir (spec B7)."
            )
        object.__setattr__(self, "n_elektron", n)
        object.__setattr__(self, "kT_hartree", kT)
        if self.kutu_x_bohr is not None:
            L = _sayi("kutu_x_bohr", self.kutu_x_bohr)
            kesme = max(self.ss_sigma.r_cut_bohr, self.itme.r_cut_bohr)
            if not 2.0 * kesme < L:
                raise InvalidInputError(
                    f"max(r_cut) < L/2 olmali: max(r_cut) = {kesme}, L = {L} "
                    "(aksi halde birden cok goruntu kesme icine girer)"
                )
            object.__setattr__(self, "kutu_x_bohr", L)

    def tani(self, positions: np.ndarray) -> TekOrbitalSonuc:
        """Tum sonuc: Omega, kuvvet, E_band, E_rep, entropi, mu, eps, f."""
        return tek_orbital_mermin(
            positions,
            self.ss_sigma,
            self.itme,
            self.n_elektron,
            self.kT_hartree,
            self.kutu_x_bohr,
        )

    def __call__(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        sonuc = self.tani(positions)
        if sonuc.kuvvet is None:
            # kT > 0 kapisi kurulusta; bu ikinci bir bekcidir (kT = 0
            # kesirli durumda `tek_orbital_mermin` kuvveti None dondurur ve
            # integrator onu ham TypeError ile dusururdu).
            raise InvalidInputError(
                "kuvvet tanimsiz (kT = 0 ve Fermi kumesi kesirli); MD icin kT > 0 sart"
            )
        return sonuc.omega, sonuc.kuvvet
