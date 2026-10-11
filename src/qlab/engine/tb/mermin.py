"""Mermin dolumu ve tek orbitalli TB enerji/kuvvet makinesi -- HARTREE ve BOHR.

NE HESAPLANIR (spec md-tb ss-sigma B1). Donen enerji MERMIN SERBEST
ENERJISIDIR:

    Omega = E_band - kT * S + E_rep,   E_band = g_s sum_n f_n eps_n,
    S = -g_s sum_n [f ln f + (1 - f) ln(1 - f)],   E_rep = sum_{i<j} itme(d_ij)

ve kuvvet F = -grad(Omega)'dir (Hellmann-Feynman; kanonik -- N sabit --
Fermi dolulugunda Omega'nin TAM gradyani). Kuvvet E = E_band + E_rep'in
gradyani DEGILDIR. Fark F_HF - F_FD(E) = kT grad S; genel bicimi

    kT grad S = -g_s sum_n f_n (1 - f_n) ((eps_n - mu)/kT) (grad eps_n - grad mu).

Fermi ciftine (iki seviye, digerleri donmus: kT komsu bosluklardan cok
kucuk) indirgenince kT grad S = -P(x) grad(gap), x = gap/kT, ve P YASASI
CIFTTEKI ELEKTRON SAYISINA BAGLIDIR (genel yasa DEGILDIR):

* Ciftte 2 elektron (f_ust = 1 - f_alt, mu bosluk ortasinda):
  P(x) = x/(4 cosh^2(x/4)), P_max = 0.44774, x = 3.0868'de.
* Ciftte 1 ya da 3 elektron: P_1(x) = 2 x df_1/dx, f_1 = f(a) alt seviyenin
  dolulugu, f(a) + f(a + x) = 1/2 (f(y) = 1/(1 + e^y), a = (eps_alt - mu)/kT);
  P_max = 0.23967, x = 2.0873'te.

OLCULDU (24.09.2026; zincir(16, 2, 0.05, 3), L = 32, bilesen (1, x), Fermi
cifti 8. ve 9. seviye): n_e = 16'da P_olculen/P = 1 (M4, x in [0.2, 20],
sapma <= 6.1e-07); n_e = 15 ve 17'de P_olculen/P_1 = 1.000000 (x = 1,
2.087, 3.087; 0.2 ve 10'da 1.000001) ve P_olculen/P = 0.715 / 0.596 /
0.437 / 0.0137 (x = 1 / 2.087 / 3.087 / 10). Tek elektronlu kolun testi
M4-P1'dir (`tests/tb/test_mermin.py`). Iki durumda da P > 0: smearing
bosluk ACAN degil KAPATAN yone iter, yani Peierls'e karsi calisir. NVE
MD'de korunan buyukluk KE + Omega'dir, KE + E degil (KE + E = KE + Omega +
kT S); termostatli kosuda toplam enerji zaten korunmaz.

MODEL SINIRI (spec B6). Atom basina TEK s orbitali; hoplama rolundeki yasa
Slater-Koster tablosunun YALNIZ ss-sigma satiridir. Tek s orbitali/atom:
aci carpani OZDES 1, bu bir SK modeli DEGILDIR. p/d yok (V_sp-sigma,
V_pp-sigma, V_pp-pi kapsam disi). Isaret: H'nin kosegen disi elemani
dogrudan t'dir (hamiltonian.py `Hopping`); '-t' konvansiyonu kullanilmaz.
Spin katlanmasi g_s = 2 SABITTIR (manyetizma yok); yuk oz-uyumu (SCC/U) yok;
tek tur. Tekbicim onsite c bir AYARDIR: Omega'yi n_e c kaydirir, kuvveti
degistirmez -- ikisi de YUVARLAMA duzeyinde ve sapma |c| ile buyur (olculdu,
`tests/tb/test_distance_model.py` M6 onsite kolu, zincir(16, 1.8, 0.05, 3/5):
c = +-16'da Omega'da <= 5.7e-14, kuvvette <= 3.6e-14; |c| <= 1'de <= 3.6e-15
ve <= 4.2e-15). Bu yuzden `tek_orbital_mermin` onsite ALMAZ.

YASA TEK NESNEDIR. Radyal yasa `qlab.engine.tb.radyal.UsselRadyalYasa`dan
gelir; bu dosya yasanin ikinci bir kopyasini yazmaz (ustel/trigonometrik
cagri yok; `tests/tb/test_mermin.py` `ast` ile kilitler).

BAGIMSIZLIK (spec M10). Bu dosya hf paketini, `tb.downfold`u ve
`tb.factory`yi import etmez; importu tek basina yalniz `tb.radyal`i ve hata
modulunu yukler (hicbir hf modulu yuklenmez). Kilitler
`tests/tb/test_mermin.py`dedir (`test_M10_*`).

PARAMETRELER SECIMDIR (spec B4). Yasadan geri okunan hicbir sayi (denge
bagi, dimerlesme genligi, esik) fizik iddiasi olarak sevk edilemez. Bu
kenarda DOGA CIPASI YOKTUR; testlerdeki cipalar MODEL cipasidir.

kT = 0 (spec B2). Statik kT = 0 aufbau olarak acik kalir, ama Fermi
seviyesindeki kume KESIRLI dolunca kuvvet `None` doner (0.0 DEGIL): dejenere
noktada Omega'nin gradyani yoktur (tek yanli egimler +-, merkezi fark 0
okur). Kesirli ama dejenere olmayan tek seviyede (tek elektron) gradyan
aslinda vardir; ama bir seviyenin "dejenere olmayan" sayilmasi bir tolerans
KARARIDIR (asagida) ve o kararin bandinda kuvvet sicrar. Bu yuzden kural
muhafazakardir: kT = 0 ve kesirli ise kuvvet HER DURUMDA `None`. MD icin
kT > 0 zorunludur (kuvvet alani sinifinin isi).

DEJENERELIK TOLERANSI (kT = 0). Ayni enerjili kume `|eps_j - eps_i| <=
_DEJENERELIK_KATSAYISI * eps_mach * max|eps|` ile ayrilir: `eigh`in geri
hatasi |H|_2 = max|eps| ile olceklenir, tolerans da. Onceki MUTLAK 1e-10
(hartree) r_cut altinda 2.8e-05 bohr'luk bir bantta dejenere OLMAYAN iki
seviyeye sahte f = 1/2 yaziyordu (kuvvet 7.1e-06 sicriyor). Goreli
tolerans bandi YOK ETMEZ -- cozunurluk sinirinin altindaki yarilmalar
ayirt edilemez -- yalniz bandi `eigh`in kendi cozunurlugune indirir; band
|H| ile buyur (tekbicim onsite 1e4 ile 3.4e-05 bohr). Katsayinin olcumu
katsayinin yaninda yazili.

kT > 0 DOLUMU. mu, kayan noktada sum f = n'yi saglayan PLATONUN ORTASIDIR
(iki biseksiyon, erken durma: orta nokta bir uca esitlenince durur). Platonun
alt ucu (onceki prototip) bosluklu sistemde kucuk kT'de mu'yu bosluk
merkezinden -340 kT kaydiriyordu. Entropi kararli bicimde yazilir:
s(x) = ln(1 + e^-|x|) + |x| expit(-|x|), x = (eps - mu)/kT. Omega'ya
+ mu (n - n_hesap) eklenir: biseksiyon artigi dn'nin Omega'ya birinci
mertebeden sizmasini keser (Omega mu'ya gore duragan olur).

RAPORLANAN mu FIZIKSEL KANONIK KOKTEN kT'NIN BIR KESRI KADAR SAPABILIR.
mu platonun ortasidir; kuyruklar kayan nokta cozunurlugunun altina dusunce
(bosluk/kT >~ 70) platonun iki ucu yuvarlama esiklerince belirlenir ve
simetrik degildir. Dejenere olmayan HOMO/LUMO'da kesin kanonik mu bosluk
ortasidir (ustel kucuk duzeltmeyle); O(kT) bir terim yalniz esit olmayan
katliliklarda dogar ((kT/2) ln(g_H/g_L)). OLCULDU (24.09.2026, log uzayinda
kesin kok + 120 haneli Decimal capraz kontrolu): M1'in 3B kumesinde (n_e =
8) kT = 1e-6..1e-4'te mu - orta = +0.1438 kT = ln(4/3)/2 (plato HOMO +
35.638 kT'den LUMO - 35.351 kT'ye), kT = 1e-3'te 0 (kuyruklar temsil
edilebilir); 4 seviyeli oyuncakta ln2/2 = 0.3466 kT; 400 rastgele spektrumda
(4-32 seviye, bosluk/kT 80-2000) en cok 0.394 kT, bosluk/kT 1-30'da
<= 3.4e-10 kT. Omega, kuvvet ve n_hesap bundan kayan nokta duzeyinde
ETKILENMEZ: platoda butun f'ler birebir aynidir ve duzeltme terimi 0'dir.
Fiziksel kok gerekirse plato bulunduktan sonra log uzayinda bir kok adimi
eklenebilir (logsumexp(-logaddexp(0, (mu - eps_dolu)/kT)) =
logsumexp(-logaddexp(0, (eps_bos - mu)/kT))); YAPILMADI, acik kalan.

PERIYODIK KUTU (spec B5). Yalniz x ekseni periyodik, en-yakin-goruntu
(vektorize); y ve z ACIK. max(r_cut) < L/2 KESIN kosuldur (aksi halde birden
cok goruntu kesme icine girer ve cift SESSIZCE eksik sayilir):
`2 * max(r_cut) >= L` `InvalidInputError`. Kucuk kutu Gamma'da bir halka
verir, kristal degil.

SERT CEKIRDEK YOK (spec B9). Ussel yasa d = 0'da sonludur ve orada kuvvetin
yonu tanimsizdir (gradyan yok). Cakisik cift (en-yakin-goruntu sonrasi
d = 0.0) `InvalidInputError`. Duvar sonludur. d = 0'da bir V-KUYUSU
olabilir: dimerde (n_e = 2, kT -> 0, d < r_on; t = t0 e^(-k(d/d0 - 1)),
itme = A e^(-krep(d/d0 - 1))) kosul 2|t0| k e^k > A krep e^krep'tir (Omega'nin
d = 0'daki egimi pozitif); krep <= k ve 2|t0| k > A krep bunun YETER
kosuludur ve o zaman Omega d0'dan d = 0'a bariyersiz iner. Bu dosya o
parametreleri REDDETMEZ; kuyunun dibinde (d = 0) kuvvet gradyan degildir.
Ortak pencerede (t0 = -0.1, k = 2, A = 0.05, krep = 6) kosul tutmaz (2.96 <
121). Makine testi `tests/md/test_tb_kuvvet.py`de (B9).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from qlab.engine.tb.radyal import UsselRadyalYasa
from qlab.errors import InvalidInputError

#: Spin katlanmasi. SABIT: manyetizma yok (spec B6).
G_S = 2.0

#: kT = 0 dejenerelik toleransi, `eps_mach * max|eps|` biriminde.
#: OLCULDU (24.09.2026, Python 3.14.7, numpy 2.5.1). (1) `eigh`in TAM
#: simetrik girdide urettigi yarilma: duzgun halkalarda (N 4..512, a = 2,
#: 3612 ikiz grup; analitik eps_k ile eslenmis) ve 3B simetrik kumelerde
#: (tetrahedron/oktahedron/kup, 200'er rastgele donme, 1000 grup) en cok
#: 21.8 eps max|eps|. (2) KOORDINAT yuvarlamasinin urettigi GERCEK yarilma
#: koordinat buyuklugu X ile buyur: N16 halkada (a = 1.8) X ~ 1e2'de 41.6,
#: 1e3'te 99.7, 1e4'te 2400. 256, X <~ 1e3 bohr'u kapsar (payi 2.6x) ve
#: `eigh` gurultusune 11.7x pay birakir. Bedeli: 40 bohr otede r_cut
#: altina giren zayif bir cift, Fermi seviyesindeyken r_cut - d < 2.7e-07
#: bohr bandinda dejenere sayilir (kesirli -> kuvvet None); eski mutlak
#: 1e-10'da ayni bant 2.8e-05 idi ve orada sahte f = 1/2 ile KUVVET DE
#: donuyordu. Katsayi 64'te bant 1.4e-07, 1024'te 5.5e-07, 4096'da 1.15e-06.
_DEJENERELIK_KATSAYISI = 256.0

#: Biseksiyon adim tavani. Normal cikis erken durmadir (bitisik iki kayan
#: noktaya inince); tavan yalniz bir emniyettir.
_BISEKSIYON_TAVANI = 400

#: mu parantezini genisleten katlama tavani (5e-324'ten 1.8e308'e ~2100).
_PARANTEZ_TAVANI = 2200


def _skaler(ad: str, deger: object) -> float:
    """Tek bir sonlu gercel sayi; aksi halde `InvalidInputError`."""
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
class MerminDolum:
    """Spin-dejenere dolum. `f` seviye basina (0..1), `eps` ile ayni sirada.

    `n_hesap = g_s * sum(f)`; `entropi` boyutsuz (k_B biriminde, g_s dahil);
    `mu` HARTREE. kT = 0'da `entropi = 0.0`; `kesirli` yalniz kT = 0'da True
    olabilir (Fermi kumesi kismen dolu). Uc doluluklarda (n = 0 ya da
    n = g_s * seviye sayisi) kanonik mu sonsuza gider: -inf / +inf doner.
    kT > 0'da `mu` kayan-nokta platosunun ORTASIDIR: kuyruklar kayan nokta
    cozunurlugunun altina dusunce (bosluk/kT >~ 70) fiziksel kanonik kokten
    kT'nin bir kesri kadar sapabilir (olculdu en cok 0.394 kT; modul
    dokstringi). Omega, kuvvet ve `n_hesap` bundan etkilenmez.
    """

    f: np.ndarray
    mu: float
    entropi: float
    kesirli: bool
    n_hesap: float


def _aufbau(eps: np.ndarray, n_elektron: float) -> MerminDolum:
    """kT = 0: seviyeleri alttan doldur; son kume kismen doluysa `kesirli`."""
    sira = np.argsort(eps, kind="stable")
    e = eps[sira]
    tol = _DEJENERELIK_KATSAYISI * np.finfo(float).eps * float(np.max(np.abs(e)))
    f_sirali = np.zeros_like(e)
    kalan = n_elektron / G_S
    kesirli = False
    mu = -np.inf if n_elektron == 0.0 else np.inf
    i = 0
    while kalan > 0.0 and i < e.size:
        j = i
        while j + 1 < e.size and e[j + 1] - e[i] <= tol:
            j += 1
        boy = j - i + 1
        if kalan >= boy:
            f_sirali[i : j + 1] = 1.0
            kalan -= boy
            mu = 0.5 * (float(e[j]) + float(e[j + 1])) if j + 1 < e.size else np.inf
        else:
            f_sirali[i : j + 1] = kalan / boy
            kesirli = True
            mu = float(np.mean(e[i : j + 1]))
            kalan = 0.0
        i = j + 1
    f = np.empty_like(f_sirali)
    f[sira] = f_sirali
    return MerminDolum(
        f=f, mu=float(mu), entropi=0.0, kesirli=kesirli, n_hesap=G_S * float(np.sum(f))
    )


def _fermi(eps: np.ndarray, n_elektron: float, kT: float) -> MerminDolum:
    """kT > 0: kanonik Fermi-Dirac, mu kayan-nokta platosunun ortasi.

    Platonun ortasi fiziksel kanonik kok DEGILDIR: kuyruklar yuvarlamanin
    altindayken ondan kT'nin bir kesri kadar (olculdu en cok 0.394 kT)
    sapabilir; Omega, F ve n_hesap platoda degismez (modul dokstringi).
    """
    tam = G_S * eps.size
    if n_elektron == 0.0 or n_elektron == tam:
        dolu = 1.0 if n_elektron == tam else 0.0
        return MerminDolum(
            f=np.full(eps.size, dolu),
            mu=np.inf if dolu else -np.inf,
            entropi=0.0,
            kesirli=False,
            n_hesap=n_elektron,
        )

    def say(mu: float) -> float:
        return G_S * float(np.sum(expit((mu - eps) / kT)))

    # Kapali parantez: say(alt) < n < say(ust). 2m expit(-y) < n e^-2 icin
    # y = ln(2m/n) + 2 yeter; kayan nokta payi yoksa (kT y, ulp(eps)'in
    # altinda: sinir uca yuvarlanir) adim ikiye katlanir -- SINANDI
    # (`tests/tb/test_mermin.py`, KAYAN NOKTA DIBI testi: eps = [1, 2],
    # n = 1, kT = 1e-20; katlamayan mutant bu gecerli girdide hata veriyor).
    # Tavan (~2100 katlama: 5e-324'ten 1.8e308'e) asilirsa girdi gecersiz
    # sayilir. OLCULDU: `mermin_dolum`un n-araligi ya da eps-sonluluk
    # kapisi kaldirilinca sinirsiz dongu SONSUZA gidiyordu (300 s zaman
    # asimi); tavan ikinci bir kapidir. SINANMADI: adimin tasip sonsuz
    # oldugu yol (buyuk kT); kT = 1e308'de dolum nan donuyor (olculdu,
    # acik kalan).
    def parantez(uc: float, isaret: float, y: float, kosul) -> float:
        adim = kT * y
        for _ in range(_PARANTEZ_TAVANI):
            sinir = uc + isaret * adim
            if kosul(say(sinir)):
                return sinir
            adim *= 2.0
        raise InvalidInputError(
            f"mu parantezi kapanmadi: n = {n_elektron}, {eps.size} seviye, kT = {kT}"
        )

    alt = parantez(
        float(np.min(eps)), -1.0, np.log(tam / n_elektron) + 2.0, lambda s: s < n_elektron
    )
    ust = parantez(
        float(np.max(eps)), 1.0, np.log(tam / (tam - n_elektron)) + 2.0, lambda s: s > n_elektron
    )

    def ikiye_bol(kosul, lo: float, hi: float) -> tuple[float, float]:
        # kosul(lo) True, kosul(hi) False; bitisik kayan noktalara inince dur.
        for _ in range(_BISEKSIYON_TAVANI):
            orta = 0.5 * (lo + hi)
            if orta <= lo or orta >= hi:
                break
            if kosul(orta):
                lo = orta
            else:
                hi = orta
        return lo, hi

    # Plato [mu_alt, mu_ust]: say(mu) == n olan kayan-nokta araligi.
    lo_a, mu_alt = ikiye_bol(lambda m: say(m) < n_elektron, alt, ust)
    if say(mu_alt) > n_elektron:
        mu_ust = lo_a  # plato yok: n iki bitisik kayan noktanin arasinda
    else:
        mu_ust, _ = ikiye_bol(lambda m: say(m) <= n_elektron, mu_alt, ust)
    mu = 0.5 * (mu_alt + mu_ust)

    x = (eps - mu) / kT
    f = expit(-x)
    # |x| kirpilir: kT kayan noktanin dibindeyse x = inf olabilir ve
    # inf * 0 = nan olurdu; 1e300'de iki terim de zaten tam 0. SINANDI
    # (KAYAN NOKTA DIBI testi, kT = 5e-324; kT = 1e-308 bu dali TETIKLEMEZ).
    ax = np.minimum(np.abs(x), 1e300)
    s = np.logaddexp(0.0, -ax) + ax * expit(-ax)
    return MerminDolum(
        f=f,
        mu=float(mu),
        entropi=G_S * float(np.sum(s)),
        kesirli=False,
        n_hesap=G_S * float(np.sum(f)),
    )


def mermin_dolum(eps: object, n_elektron: object, kT_hartree: object) -> MerminDolum:
    """Spin-dejenere (g_s = 2 SABIT) kanonik dolum.

    kT > 0: Fermi-Dirac, mu platonun ORTASI, kararli entropi. kT = 0:
    aufbau; Fermi kumesi (goreli dejenerelik toleransiyla ayrilan) kismen
    doluysa `kesirli = True`. `f` `eps` ile ayni sirada doner.
    """
    e = np.asarray(eps, dtype=float)
    if e.ndim != 1 or e.size == 0:
        raise InvalidInputError(f"eps bos olmayan bir boyutlu dizi olmali: sekil {e.shape}")
    if not np.all(np.isfinite(e)):
        raise InvalidInputError("eps sonlu olmali (nan/inf geldi)")
    n = _skaler("n_elektron", n_elektron)
    if not 0.0 <= n <= G_S * e.size:
        raise InvalidInputError(
            f"n_elektron [0, {G_S * e.size}] araliginda olmali: {n} "
            f"({e.size} seviye, g_s = {G_S})"
        )
    kT = _skaler("kT_hartree", kT_hartree)
    if kT < 0.0:
        raise InvalidInputError(f"kT_hartree >= 0 olmali: {kT}")
    dolum = _aufbau(e, n) if kT == 0.0 else _fermi(e, n, kT)
    dolum.f.setflags(write=False)
    return dolum


@dataclass(frozen=True, eq=False)
class TekOrbitalSonuc:
    """`tek_orbital_mermin` sonucu. Enerjiler HARTREE, kuvvet HARTREE/BOHR.

    `omega` Mermin serbest enerjisi (kT > 0'da mu (n - n_hesap) duzeltmesi
    dahil); `kuvvet` (N, 3) = -grad(omega), kT = 0 ve `kesirli` ise `None`.
    `eps` artan sirada ozdegerler, `f` onlarin dolumu.
    """

    omega: float
    kuvvet: np.ndarray | None
    E_band: float
    E_rep: float
    entropi: float
    mu: float
    eps: np.ndarray
    f: np.ndarray
    kesirli: bool


def _konumlar(R: object) -> np.ndarray:
    konum = np.array(R, dtype=float)
    if konum.ndim != 2 or konum.shape[1] != 3 or konum.shape[0] == 0:
        raise InvalidInputError(f"R (N, 3) olmali, N >= 1: sekil {konum.shape}")
    if not np.all(np.isfinite(konum)):
        raise InvalidInputError("R sonlu olmali (nan/inf geldi)")
    return konum


def tek_orbital_mermin(
    R: object,
    ss_sigma: UsselRadyalYasa,
    itme: UsselRadyalYasa,
    n_elektron: object,
    kT_hartree: object,
    kutu_x_bohr: object = None,
) -> TekOrbitalSonuc:
    """(N, 3) konum -> Omega ve F = -grad(Omega); tek s orbitali/atom.

    H_ij = ss_sigma(d_ij) (i != j), H_ii = 0; E_rep = sum_{i<j} itme(d_ij).
    `kutu_x_bohr` verilirse x ekseninde en-yakin-goruntu (y, z acik) ve
    2 max(r_cut) < L sart. Cakisik cift (d = 0) hata. Ayrintilar modul
    dokstringinde (B1, B2, B5, B6, B9).
    """
    for ad, yasa in (("ss_sigma", ss_sigma), ("itme", itme)):
        if not isinstance(yasa, UsselRadyalYasa):
            raise InvalidInputError(f"{ad} bir UsselRadyalYasa olmali: {type(yasa).__name__}")
    konum = _konumlar(R)
    n_atom = konum.shape[0]
    n = _skaler("n_elektron", n_elektron)
    kT = _skaler("kT_hartree", kT_hartree)

    fark = konum[None, :, :] - konum[:, None, :]  # fark[i, j] = R_j - R_i
    if kutu_x_bohr is not None:
        L = _skaler("kutu_x_bohr", kutu_x_bohr)
        kesme = max(ss_sigma.r_cut_bohr, itme.r_cut_bohr)
        if not 2.0 * kesme < L:
            raise InvalidInputError(
                f"max(r_cut) < L/2 olmali: max(r_cut) = {kesme}, L = {L} "
                "(aksi halde birden cok goruntu kesme icine girer ve cift "
                "sessizce eksik sayilir)"
            )
        fark[..., 0] -= L * np.round(fark[..., 0] / L)
    i, j = np.triu_indices(n_atom, 1)
    vektor = fark[i, j]
    d = np.sqrt(np.sum(vektor * vektor, axis=1))
    cakisik = np.flatnonzero(d == 0.0)
    if cakisik.size:
        k = int(cakisik[0])
        raise InvalidInputError(
            f"cakisik atom cifti ({int(i[k])}, {int(j[k])}): d = 0 "
            "(en-yakin-goruntu sonrasi). Ussel yasada sert cekirdek yok; "
            "d = 0'da kuvvetin yonu tanimsiz."
        )

    t, dt = ss_sigma.deger_ve_turev(d)
    phi, dphi = itme.deger_ve_turev(d)
    H = np.zeros((n_atom, n_atom))
    H[i, j] = t
    H[j, i] = t
    eps, C = np.linalg.eigh(H)
    dolum = mermin_dolum(eps, n, kT)

    E_band = G_S * float(np.sum(dolum.f * eps))
    E_rep = float(np.sum(phi))
    omega = E_band - kT * dolum.entropi + E_rep
    if kT > 0.0 and dolum.n_hesap != n:
        omega += dolum.mu * (n - dolum.n_hesap)

    kuvvet = None
    if not (kT == 0.0 and dolum.kesirli):
        rho = (C * (G_S * dolum.f)) @ C.T
        # dOmega/dR_i = sum_j [2 rho_ij t'(d_ij) + itme'(d_ij)] * (-u_ij),
        # u_ij = (R_j - R_i)/d_ij; cift (i<j) i'ye -c u, j'ye +c u verir.
        c = 2.0 * rho[i, j] * dt + dphi
        katki = (c / d)[:, None] * vektor
        gradyan = np.empty((n_atom, 3))
        for eksen in range(3):
            gradyan[:, eksen] = np.bincount(
                j, weights=katki[:, eksen], minlength=n_atom
            ) - np.bincount(i, weights=katki[:, eksen], minlength=n_atom)
        kuvvet = -gradyan
        kuvvet.setflags(write=False)
    eps.setflags(write=False)
    return TekOrbitalSonuc(
        omega=float(omega),
        kuvvet=kuvvet,
        E_band=E_band,
        E_rep=E_rep,
        entropi=dolum.entropi,
        mu=dolum.mu,
        eps=eps,
        f=dolum.f,
        kesirli=dolum.kesirli,
    )
