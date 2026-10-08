"""Bant yapisi, k-yolu ve bant kenarlari.

BANT KENARLARI IZGARADAN OKUNMAZ. Olculdu ve bu modulun en onemli tasarim
karari budur: grafende 601x601 izgara BZ minimum araligini 2.471886e-15 eV
verirken 600x600 1.635969e-02 eV veriyor -- ve DAHA SIK olan 1000x1000
1.698157e-02 eV, yani 601'den 13 MERTEBE kotu. Siklastirmak duzeltmiyor.

Ayni aile 1B'de de olculdu ve iki ayri bicimde cikiyor:
- PARITE: tekbicim izgarada bant genisligi TEK n icin 2|t|(1+cos(pi/n))
  (n=3 -> gercek degerin %75'i), CIFT n icin tam 4|t|. Hata monoton
  kuculuyor, yani bir YAKINSAMA SORUNU gibi gorunuyor.
- MONOTON OLMAMA: ikinci komsulu zincirde bant minimumu yuksek simetri
  noktasinda degil ve hata izgara sikligiyla duzenli AZALMIYOR --
  n=5 -> 2.426e-06, n=7 -> 1.934e-02, n=12 -> 7.031e-03, n=13 -> 7.370e-03.
  n=5, n=13'ten yaklasik 3000 kat iyi.

Bu yuzden `band_edges` surekli minimizasyon yapar; `uniform_k_grid` yalniz
cizim ve DOS icindir ve cagiranin bildirdigi k noktalarini icerip
icermedigini DENETLER.

SIHIRLI SAYI YOK. On olcumde iki farkli izgara konvansiyonu iki farkli
"dogru N" verdi (601 ve N % 3 == 0); ikisi de kendi konvansiyonunda dogru.
Bu yuzden parite kodda sabitlenmez, `must_include` ile denetlenir.

`band_structure` ciktisi SIRALAMADIR, BANT KIMLIGI DEGILDIR. Olculdu (iki
bagimsiz zincir, kesisim ka* = 2.0427): kesisimin 0.05 bohr^-1 SAGINDA
sirali "bant 0"in etkin kutlesi +1.6735 -- obur zincirinki --, gercek zincir
0'inki -0.6276: ISARET BILE TERS, ve sonlu fark adimindan (1e-3 ... 1e-5)
bagimsiz. (Eski "12457 kat" sayisi YENIDEN URETILEMEDI: kesisimin tam
uzerinde sirali kutle adim h ile olceklenir -- oran 1/h, h = 1e-3/1e-4/1e-5
icin 674/6736/67361 --, yani sayi buyuk olasilikla orada alinmisti; ama
eski metnin "-0.800"u bu modelin kesisimdeki -0.859'uyla tutmuyor.) Cizim icin
dogru, bant-cozumlu nicelikler icin degil -- ve depo bant cozumlu nicelik
URETMEZ (`tests/tb/test_bands.py` cagiran listesini kilitler).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import minimize

from qlab.engine.tb.hamiltonian import TBModel, bloch_hamiltonian
from qlab.engine.tb.lattice import Lattice, reciprocal_vectors
from qlab.errors import InvalidInputError

# `must_include` uyelik denetiminin toleransi -- ters orgu TABANINDA
# kesirli koordinat uzerinden olculur, dolayisiyla boyutsuzdur ve mutlak
# bir enerji/uzunluk esigine bagli degildir.
_UYELIK_TOLERANSI = 1e-9

# E2 (04.10.2026): `band_edges`in tohum izgarasi ve cilasi -- gerekceler
# (olculmus) fonksiyonun dokstring'inde.
_TOHUM_YOGUNLUGU = 8  # eksen basina, birim hoplama menzili basina tohum noktasi
_ADAY_SAYISI = 3
_KOMPAS_TAVANI = 10_000
# Kompas bir adimi yalniz iyilesme bu kat eps * H olcegini asarsa kabul eder:
# yuvarlama gurultusunu iyilesme sayinca dar duzgun vadide 10 000 adimlik
# tavana dayanip zikzak ciziyordu (aralikli grafen, 2.6 s/arama).
_KOMPAS_GURULTU_KATSAYISI = 16.0


@dataclass(frozen=True, eq=False)
class KPath:
    """Yuksek simetri noktalarini birlestiren dogru parcalari."""

    points: np.ndarray
    distances: np.ndarray
    tick_distances: np.ndarray


def uniform_k_grid(
    lattice: Lattice, n: int, must_include: tuple[np.ndarray, ...] = ()
) -> np.ndarray:
    """Gamma merkezli tekbicim izgara, (n**d, 3). `must_include` DENETLENIR."""
    if n < 2:
        raise InvalidInputError(f"n en az 2 olmali, verilen: {n}")
    b = reciprocal_vectors(lattice)
    d = lattice.dimension
    kesirli = np.arange(n) / n
    noktalar = np.array(
        [
            sum(f[eksen] * b[eksen] for eksen in range(d))
            for f in product(kesirli, repeat=d)
        ]
    )
    for hedef in must_include:
        hedef = np.asarray(hedef, dtype=float)
        # Ters orgu periyodikligi: hedef, izgara noktasindan bir G kadar
        # uzakta olabilir. Bu yuzden fark ters orgu tabaninda KESIRLI
        # olarak alinip tam sayiya yakinligi sinanir.
        kesirli_fark = (noktalar - hedef) @ np.linalg.pinv(b)
        yakin = np.abs(kesirli_fark - np.rint(kesirli_fark)).max(axis=1)
        if float(yakin.min()) > _UYELIK_TOLERANSI:
            raise InvalidInputError(
                f"istenen k noktasi izgarada yok (en yakin kesirli sapma "
                f"{float(yakin.min()):.3e}). n'i degistirin; izgarayi "
                "SIKLASTIRMAK bu hatayi duzeltmez -- olculdu, ikinci komsulu "
                "zincirde n=5 hatasi 2.426e-06 iken n=13 hatasi 7.370e-03."
            )
    return noktalar


def band_structure(model: TBModel, kpoints: np.ndarray) -> np.ndarray:
    """(n_k, n_orb) SIRALI ozdegerler, hartree. Bant kimligi TASIMAZ."""
    kpoints = np.asarray(kpoints, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise InvalidInputError(f"kpoints (n_k,3) olmali: {kpoints.shape}")
    return np.array(
        [np.linalg.eigvalsh(bloch_hamiltonian(model, k)) for k in kpoints]
    )


def grid_lipschitz_bound(model: TBModel, n: int) -> float:
    """B1 (hartree): sirali her ozdeger, n izgarasi ile surekli k arasinda
    en cok bu kadar oynar -- KANITLI, olculmus degil.

    B1 = (pi / n) * max_i sum_{h: h.i = i} |t_h| * ||cell_h||_1.

    Turetme (E13, 04.10.2026): Weyl, |lambda_b(k) - lambda_b(k')| <=
    ||H(k) - H(k')||; Hermityen matriste ||A||_2 <= max satir toplami; hucre
    gauge'unda her eleman t (e^{i k.R} - e^{i k'.R}) ve |e^{ix} - e^{iy}| <=
    |x - y|. Gamma merkezli izgarada her k'ye kesirli koordinatta eksen
    basina en cok 1/(2n) uzakta bir nokta vardir (H ters orgude periyodik),
    yani |dk . R| <= (pi / n) ||cell||_1. Bant KESISIMINDE de gecerlidir
    (ikinci mertebe sinir orada gecersizdir -- kink). Olculdu: dogrulama
    turunda 690 model / 1157 (model, n) cifti (1B-3B, 1-10 orbital, |cell|
    <= 5, egik orgu, dejenere ikizler, en kotu durum aramasi) ihlal 0, en
    buyuk gercek/sinir orani 0.9992.

    ON KOSULLAR (`TBModel` denetler): Hermitlik -- tek yonlu listede ayni
    formul 36/900 vakada ihlal edildi; H0 toleransi icindeki asimetri
    (olcege gore 1e-12) en kotu dizilimde sinira ~1e-13 ekledi, sahte hukum
    kurulamadi -- ve TAM SAYI hucre (kesirli hucrede H periyodik degildir,
    sinir 128 kat asildi).
    """
    satir = _satir_toplami(model, kuvvet=1).max()
    sinir = float(np.pi / _izgara_n(n) * satir)
    # Alt-normal t'de sinir 0'a yuvarlanabiliyordu ve "H(k) sabit" dalina
    # (kanitli yalitkan) dusuyordu (dogrulama turu, t = 5e-324 hartree):
    # hoplama varken sinir ASLA sifir olmaz.
    return max(sinir, float(np.nextafter(0.0, 1.0))) if satir > 0.0 else sinir


def hamiltonian_rounding_scale(model: TBModel) -> float:
    """H(k)'nin KURULUS yuvarlamasinin olcegi (hartree):
    max_i (|onsite_i| + sum_{h: h.i = i} |t_h| (1 + 2 pi ||cell_h||_1)).

    `certified_conduction`in yuvarlama payi bunu |E|max'a ekler. Sebep
    (dogrulama turu, 04.10.2026, kamu yolundan): ayni (i, j, R) birbirini
    goturen parcalarla yazilinca (1.23 + 1.6 - 2.83 eV = 0) toplamada kalan
    kirinti DUZ bandi "metal (tanikli)" gosteriyordu; |R| ~ 1e3 hucrede faz
    argumaninin yuvarlamasi da ayni seyi yapiyordu. Ikisi de |E|max ile degil
    sum |t| ve |k . R| <= 2 pi ||cell||_1 ile olceklenir.
    """
    d = model.lattice.dimension
    satir = np.abs(np.asarray(model.onsite, dtype=float)).copy()
    for h in model.hoppings:
        satir[h.i] += abs(h.t) * (
            1.0 + 2.0 * np.pi * sum(abs(h.cell[eksen]) for eksen in range(d))
        )
    return float(satir.max())


def grid_extremum_bound(model: TBModel, n: int) -> float:
    """B2 (hartree): spektrumun GLOBAL alt ucu (en alt bandin minimumu) ve
    ust ucu (en ust bandin maksimumu) n izgarasinda en cok bu kadar
    kacirilir. KANITLI; izgara degeri hep ICERIDEDIR (gercek min <= izgara
    min <= gercek min + B2).

    B2 = (pi^2 / (2 n^2)) * max_i sum_{h: h.i = i} |t_h| * ||cell_h||_1^2.

    Turetme (E1, 04.10.2026): u, k*'daki en alt ozvektor olsun. Rayleigh:
    lambda_min(k* + dk) <= <u|H(k* + dk)|u> = f(dk). k* minimum oldugu icin
    f'nin birinci turevi SIFIRDIR -- dejenerelikte bile (degilse u yonunde
    daha asagi inilirdi); |f''| <= max satir toplami |t| (dk . R)^2 ve
    |dk . R| <= (pi / n) ||cell||_1 (bkz. `grid_lipschitz_bound`). Yalniz
    GLOBAL uclar icin gecerlidir: ic kenarlarda (VBM/CBM) bantlar
    kesisebilir ve egri kirilir; orada B1 kullanilir.

    SIKI: 1B zincirde tek n'de gercek hata ile B2'nin orani n buyudukce 1'e
    gider (cos'un ikinci mertebe acilimi). Olculdu: dogrulama turunda 1157
    (model, n) ciftinde ihlal 0, en buyuk oran 0.9995; IC bantta (kesisim,
    kacinilmis kesisim) B2 32-316 kat ihlal edildi -- yani "yalniz global
    uclar" kaydi suslemek degil.
    """
    return float(
        np.pi**2 / (2.0 * _izgara_n(n) ** 2) * _satir_toplami(model, kuvvet=2).max()
    )


def _izgara_n(n: int) -> int:
    if n < 2:
        raise InvalidInputError(f"n en az 2 olmali, verilen: {n}")
    return n


def _satir_toplami(model: TBModel, kuvvet: int) -> np.ndarray:
    """Satir basina sum |t| * ||cell||_1^kuvvet (ilk `dimension` bilesen)."""
    d = model.lattice.dimension
    satir = np.zeros(model.n_orbital)
    for h in model.hoppings:
        satir[h.i] += abs(h.t) * sum(abs(h.cell[eksen]) for eksen in range(d)) ** kuvvet
    return satir


def k_path(vertices: tuple[np.ndarray, ...], n_per_segment: int) -> KPath:
    """Koseleri birlestiren yol. Ortak noktalar bir kez sayilir."""
    if len(vertices) < 2:
        raise InvalidInputError(
            f"k-yolu en az iki kose ister, verilen: {len(vertices)}"
        )
    if n_per_segment < 2:
        raise InvalidInputError(f"n_per_segment en az 2 olmali: {n_per_segment}")

    parcalar: list[np.ndarray] = []
    for indis in range(len(vertices) - 1):
        bas = np.asarray(vertices[indis], dtype=float)
        son = np.asarray(vertices[indis + 1], dtype=float)
        t = np.linspace(0.0, 1.0, n_per_segment)
        parca = bas[None, :] + t[:, None] * (son - bas)[None, :]
        parcalar.append(parca if indis == 0 else parca[1:])
    noktalar = np.vstack(parcalar)

    adimlar = np.linalg.norm(np.diff(noktalar, axis=0), axis=1)
    uzakliklar = np.concatenate([[0.0], np.cumsum(adimlar)])
    kose_indisleri = [0]
    for _ in range(len(vertices) - 1):
        kose_indisleri.append(kose_indisleri[-1] + (n_per_segment - 1))
    return KPath(
        points=noktalar,
        distances=uzakliklar,
        tick_distances=uzakliklar[np.array(kose_indisleri)],
    )


def _kesirli_hamiltonyen(model: TBModel):
    """H(f) kurucusu, f KESIRLI koordinat (hucre gauge'u): H_ij += t exp(2 pi i
    f . c). `bloch_hamiltonian(model, sum f_a b_a)` ile matematikte AYNIDIR
    (b_a . a_b = 2 pi delta_ab), ama faz k . R uzerinden degil dogrudan f . c
    uzerinden hesaplanir. Sebep (E2 dogrulama turu): kotu kosullu (orn.
    indirgenmemis a2 = (N a, a)) tabanda k . R buyuk terimlerin birbirini
    goturmesiyle hesaplaniyordu; N = 3e4'te faz gurultusu 1.2e-7 W'ye cikti
    ve `band_edges` gercek minimumun 1.9e-8 W ALTINA indi -- "aralik icerde"
    garantisi bozuldu. f in [0, 1) ve c tam sayi oldugu icin buradaki faz
    hatasi kosuldan bagimsizdir.
    """
    d = model.lattice.dimension
    taban = np.diag(np.asarray(model.onsite, dtype=float)).astype(complex)
    if not model.hoppings:
        return lambda kesirli: taban.copy()
    i = np.array([h.i for h in model.hoppings])
    j = np.array([h.j for h in model.hoppings])
    hucre = np.array([[h.cell[eksen] for eksen in range(d)] for h in model.hoppings], dtype=float)
    t = np.array([h.t for h in model.hoppings], dtype=float)

    def kur(kesirli: np.ndarray) -> np.ndarray:
        H = taban.copy()
        np.add.at(H, (i, j), t * np.exp(2j * np.pi * (hucre @ np.asarray(kesirli, dtype=float))))
        return H

    return kur


def band_edges(model: TBModel, band_index: int) -> tuple[float, float]:
    """Bir bandin (min, max) enerjisi, hartree -- BULUNAN EN IYI uc degerler.

    KESIN OLAN YON (yuvarlama payi ~eps * H olcegi icinde): donen her deger
    gercek bir k noktasindaki gercek bir ozdegerdir; donen min gercek
    minimumun ustunde, donen max gercek maksimumun altindadir -- aralik
    gercek bandin ICINDEDIR (bant genisligi icin alt sinir, bant araligi
    icin ust sinir). H(f) kesirli koordinatta kurulur (`_kesirli_hamiltonyen`);
    bu yuzden garanti orgu tabaninin kosulundan bagimsizdir (dogrulama turu:
    k . R ile kurulan eski yol kosul 9e8'de gercek minimumun 1.9e-8 W ALTINA
    iniyordu). Kuresel uc KANITLANMAZ; kanitli dis sinir icin
    `grid_lipschitz_bound`, iletkenlik icin `certified_conduction`.

    ALGORITMA (E2, 04.10.2026): eksen basina 8 R_a noktali Gamma merkezli
    tohum izgarasi (R_a: o eksende sifir olmayan hoplamalarin en buyuk
    |hucre| bileseni; 0 ise tek nokta), izgaranin en iyi 3 yerel ucundan
    f mod 1 ile sarili L-BFGS-B, sonra kompas cilasi (3^d - 1 yon, adim
    1e-3'ten 1e-16'ya; bir adim yalniz iyilesme 16 eps * H olcegini asarsa
    kabul, 10 000 adim tavani). OLCULEREK SECILENLER (scratchpad e2/*.py):
    - kompas: kompassiz grafen 20/48, PT simetrik iki orbitalli 8/40 arama
      1e-12'yi asiyordu (L-BFGS-B nokta kinkinde ve duragan TRIM'de durur).
    - kompasin gurultu esigi: esiksiz kompas dar duzgun vadide 10 000 adim
      tavanina dayaniyordu (81 666 -> 1 882 eigvalsh, sonuc ayni); bedeli
      hassasiyetin 1e-16'dan ~1e-13 bagila inmesi.
    - tohum yogunlugu 8 R_a: Gamma merkezli SABIT 8 nokta esit genlikli uzun
      menzilli ("puruzlu") 1B'de 22/149 modeli bozuyordu (en kotu %39.5).
    - 3 aday: tek adayla uzun menzilli 2B 2/40, PT 2/40 (%5), puruzlu 2B
      3/160 arama kaciyordu.
    - kaydirma YOK: on olcumdeki 0.381966 adim kaydirilmis tohum puruzlu 1B'de
      bir dar tepeyi (%1.18) kaciriyordu; Gamma merkezli tohum bulur.
    TASARIM VARSAYILANI (olculmus etkisi yok): sarma (faz kucuk kalsin --
    ilke), L-BFGS-B (yalniz hiz; kompas tek basina ayni sonucu verir),
    capraz kompas yonleri ve yerel uc suzgeci.

    OLCULEN SONUC (eski 1e-9 / yeni 1e-12): orijinal 2B 14/50 / 0/50, 3B
    8/30 / 0/30, grafen 24/48 / 0/48, PT 20/40 / 0/40, uzun menzilli 2B
    39/40 / 0/40, puruzlu 1B 133/298 / 0/298, puruzlu 2B 153/160 / 0/160,
    puruzlu 3B 19/20 / 0/20 (eskinin en kotusu bant genisliginin %56'si).
    BILINEN BASARISIZLIK SINIFLARI (bagimsiz dogrulama turu, kamu
    yolundan; testle kilitli olan isaretli):
    - 2B/3B'de KESISIM CIZGISI/YUZEYI olan bantlarin IC kenarlari: uc, iki
      bandin kesistigi dikis uzerinde dar bir V tepesidir ve tohum
      izgarasinda yerel uc olarak GORUNMEZ -- aday ve yon sayisini artirmak
      cozmuyor (olculdu). Eslesmesiz 2B: 40 ic kenarin 25'i, en kotu %6;
      cok orbitalli %10; 3B 4/24. [kilitli]
    - zayif eslesmeli KACINILMIS KESISIMIN dar havzasi R'den bagimsizdir:
      1B'de bile %7.5 (164 kenarda 2); tohumu 32 R yapmak duzeltir.
    - kacinilmis kesisim araligi <= 1e-6 iken kompas tavana dayanir (cagri
      basina ~15 s; sonuc cogunlukla dogru).
    Maliyet tohumla (8 R)^d buyur: yogun 3B R=2/3/4'te 0.13 / 0.34 / 1.06 s.
    SURE (kesirli kurucu Python dongusunu kaldirdigi icin) eskinin
    0.94x'i (puruzlu 1B) ile 176x'i (puruzlu 3B) arasinda; 3B kumede 72x.

    IZGARA KENARI OKUMAZ: tohum yalniz BASLANGIC secer; ucu surekli arama
    bulur (modul dokstring'indeki parite tuzagi).
    """
    if not (0 <= band_index < model.n_orbital):
        raise InvalidInputError(
            f"bant indisi aralik disi: {band_index}, bant sayisi "
            f"{model.n_orbital}"
        )
    d = model.lattice.dimension
    kur = _kesirli_hamiltonyen(model)

    def enerji(kesirli: np.ndarray, isaret: float) -> float:
        return isaret * float(np.linalg.eigvalsh(kur(kesirli))[band_index])

    # Tohum yogunlugu eksen basina hoplama MENZILIYLE olceklenir: tek
    # orbitalde E(f) = sum t cos(2 pi f . c) ve eksen a'daki en yuksek
    # frekans R_a = max |c_a|; sabit 8 nokta R >= 3'te havza basina 3'ten az
    # nokta dusuruyor ve dogru havzayi eliyordu (olculdu, asagida).
    # Menzil 0 (o eksende k'ya bagimlilik yok) -> tek nokta yeter.
    boyutlar = [
        max(1, _TOHUM_YOGUNLUGU * max(
            (abs(h.cell[eksen]) for h in model.hoppings if h.t != 0.0), default=0
        ))
        for eksen in range(d)
    ]
    izgara = np.array(
        list(product(*[np.arange(n) / n for n in boyutlar]))
    )
    degerler = np.array([enerji(f, 1.0) for f in izgara])
    yonler = np.array([v for v in product((-1, 0, 1), repeat=d) if any(v)], dtype=float)

    gurultu = np.finfo(float).eps * hamiltonian_rounding_scale(model)
    uc_degerler: list[float] = []
    for isaret in (+1.0, -1.0):
        isaretli = isaret * degerler
        kup = isaretli.reshape(boyutlar)
        yerel = np.ones_like(kup, dtype=bool)
        for eksen in range(d):
            for kayma in (1, -1):
                yerel &= kup <= np.roll(kup, kayma, axis=eksen)
        adaylar = np.nonzero(yerel.ravel())[0]
        adaylar = adaylar[np.argsort(isaretli[adaylar])][:_ADAY_SAYISI]
        en_iyi = float(isaretli.min())
        for aday in adaylar:
            sonuc = minimize(
                lambda f, isaret=isaret: enerji(np.mod(f, 1.0), isaret),
                izgara[aday].astype(float),
                method="L-BFGS-B",
                options={"gtol": 1e-12, "ftol": 1e-15, "maxiter": 500},
            )
            nokta = np.mod(sonuc.x, 1.0)
            deger = enerji(nokta, isaret)
            adim, sayac = 1e-3, 0
            while adim > 1e-16 and sayac < _KOMPAS_TAVANI:
                sayac += 1
                komsular = [enerji(np.mod(nokta + v * adim, 1.0), isaret) for v in yonler]
                j = int(np.argmin(komsular))
                if komsular[j] < deger - _KOMPAS_GURULTU_KATSAYISI * gurultu:
                    deger, nokta = komsular[j], np.mod(nokta + yonler[j] * adim, 1.0)
                else:
                    adim /= 2.0
            en_iyi = min(en_iyi, deger)
        uc_degerler.append(isaret * en_iyi)
    return (uc_degerler[0], uc_degerler[1])
