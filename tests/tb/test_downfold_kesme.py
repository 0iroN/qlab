"""Iki-siteli modelin UHF'e ne kadar oturdugu -- KESME hatasinin BEYANI.

SINIF: TRANSKRIPSIYON DENETIMI. Bu dosyanin HICBIR testi bir fizik iddiasi
tasimaz; hepsi modelin ne kadar iyi oldugunu OLCER ve YAZAR. Modulun fiziksel
cipasi ayri dosyadadir (test_downfold_cipa.py) ve buradaki hicbir sayiya
dayanmaz.

DURUST OLUMSUZ: kesme hatasi kucuk DEGILDIR. N=8, d=2.0'da NN-only model UHF
alfa tayfini 0.153603 hartree kaciriyor -- yayilimin (1.693171) %9.1'i. t2
eklenince 0.066287'ye iniyor (%3.9) ama sifira gitmiyor, ve TAM menzilde bile
gitmiyor (0.063140). Sebebi olculdu ve asagida ayri bir teste baglandi.

OLCULEN KESME TABLOSU (21.08.2026, bu depoda, STO-3G, seeds=16; hata =
max|TB ozdegerleri - UHF alfa ozdegerleri|, hartree):

    N   d      yayilim    NN-only        +t2      oran
    6   2.60  1.113021   0.039420   0.012078      3.26
    6   3.00  0.947273   0.023748   0.009107      2.61
    6   3.60  0.834770   0.010104   0.003925      2.57
    8   2.00  1.693171   0.153603   0.066287      2.32
    8   2.60  1.134138   0.073229   0.032485      2.25
   10   2.60  1.140294   0.059631   0.020470      2.91

Alti sistemin altisi da planin referans degeriyle alti hanede AYNI cikti.
PLANIN TEK BAYAT SAYISI BASLIKTAYDI: plan (ve `downfold_uhf`in dokstringi)
"t2 hatayi ~3 kat dusurur" diyordu; olculen oran 2.25 ile 3.26 arasinda,
ortalamasi 2.65. "Uc kat" alti sistemin yalnizca BIRINDE tutuyor. Test bu
yuzden 3 degil, olculen alt sinira (2.0) gore kilitlenir ve kaynak dokstringi
duzeltildi.

BU DOSYA PLANIN GOVDESINDE BIR HATA BULDU (bkz. antipodal testi): plan
`model_from_downfold`un hoplama eslemesini "N=6/8/10 x menzil 1/2'de
0.000e+00 uyusuyor, blok kopyalanabilir" diye sevk ediyordu. Uyusma menzil
1-2'de gercekten TAM; ama menzil n/2'de (antipodal) plan govdesi her cifti
IKI KEZ bagliyor ve t yerine 2t yaziyor. `downfold_uhf` n_neighbour'a
n_site//2'ye kadar izin verdigi icin bu bozuk hal ACIK API'den erisilebilir.
Olculdu (plan govdesi, duzeltmeden once):

    N=6,  menzil 3, (0,3): model +0.006946  beklenen +0.003473
    N=8,  menzil 4, (0,4): model +0.006293  beklenen +0.003147
    N=10, menzil 5, (0,5): model -0.005788  beklenen -0.002894

Dersi planin kendi cumlesinde yaziyordu ve gorulmedi: dogrulama PENCERESI
menzil 1-2 idi, hukum ise pencerenin DISINA tasindi.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.hf.multistart import search_uhf
from qlab.engine.tb.downfold import (
    DownfoldedParameters,
    _effective_hamiltonians,
    downfold_uhf,
    model_from_downfold,
)
from qlab.engine.tb.hamiltonian import finite_ring_matrix
from qlab.errors import InvalidInputError
from qlab.types import Molecule
from qlab.units import BOHR_TO_ANGSTROM


def _halka(n: int, d: float) -> Molecule:
    yaricap = d / (2 * np.sin(np.pi / n))
    a = yaricap * BOHR_TO_ANGSTROM
    return Molecule.from_string(
        "; ".join(
            f"H {a*np.cos(2*np.pi*k/n):.17g} {a*np.sin(2*np.pi*k/n):.17g} 0"
            for k in range(n)
        )
    )


def _elle_parametre(n_site: int, hoppings: tuple[float, ...]) -> DownfoldedParameters:
    """SCF KOSMADAN parametre kurar -- yapisal testler saniye harcamasin.

    Sayilar KEYFIDIR ve oyle olmalidir: bu testler modelin YAPISINI
    denetler, bir fizik degerini degil. Gercek UHF sayilariyla kosmak
    testi yavaslatir ve ayrimi ARTIRMAZ.
    """
    return DownfoldedParameters(
        epsilon_bar=0.25,
        delta=-0.4,
        hoppings=hoppings,
        n_site=n_site,
        basis_name="sto-3g",
        s_squared=0.0,
        energy_uhf=0.0,
        staggering_residual=0.0,
        search_line="elle kuruldu",
    )


def _beklenen_site_matrisi(
    params: DownfoldedParameters, spin: str = "alpha"
) -> np.ndarray:
    """Modelin uretmesi GEREKEN site matrisi -- TB motorundan BAGIMSIZ kurulur.

    Halka mesafesi burada YENIDEN yazilir, `downfold._ring_distance` ithal
    EDILMEZ: iki taraf ayni yardimciyi paylassaydi o yardimciyi bozan bir
    mutant iki tarafi da ayni yonde bozar ve test sessizce yesil kalirdi.
    """
    n = params.n_site
    yon = 1.0 if spin == "alpha" else -1.0
    H = np.zeros((n, n))
    for i in range(n):
        H[i, i] = params.epsilon_bar + (1.0 if i % 2 == 0 else -1.0) * yon * params.delta
    for i in range(n):
        for j in range(n):
            mesafe = min(abs(i - j), n - abs(i - j))
            if 1 <= mesafe <= len(params.hoppings):
                H[i, j] += params.hoppings[mesafe - 1]
    return H


# ---------------------------------------------------------------------------
# YAPISAL testler -- SCF kosmazlar, milisaniyedirler, `slow` DEGILDIRLER.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [6, 8, 10, 12, 14, 16])
@pytest.mark.parametrize("spin", ["alpha", "beta"])
def test_HOPLAMA_ESLEMESI_site_matrisini_BIREBIR_veriyor(n, spin):
    """SINIF: TRANSKRIPSIYON DENETIMI -- hucre indislerinin site indislerine
    eslenmesini kilitler, bir fizik iddiasi TASIMAZ.

    `model_from_downfold` iki atomlu bir hucre kurar ve hoplamalari HUCRE
    KAYMASI olarak yazar; `finite_ring_matrix` onlari modulo baglar. Bu iki
    kodlamanin gercekten istenen SITE matrisini verdigi ancak eleman eleman
    karsilastirilarak gorulebilir.

    NEDEN OZDEGER YETMEZ -- BU IDDIA OLCULDU ve ilk yazildiginda FAZLA
    GENISTI, duzeltildi. `hamiltonian` modulunun dokstringi
    `np.linalg.eigvalsh`in varsayilan UPLO='L' ile yalniz ALT ucgeni
    okudugunu yaziyor. Ilk hali "ust ucgendeki bir kaydi dusuren mutant
    tayfi HIC degistirmez" diyordu; olculdugunde korluk MENZILE BAGLI cikti.

    Mutasyon M5 (hucre ici `Hopping(0, 1, (kayma,0,0))` kaydi silinir,
    yani her TEK menzilde bir kayit eksilir) uzerinde olculen
    max|dOzdeger|:

        menzil 1:  0.000e+00 (dordunde de, TAM)
        menzil 2:  0.000e+00 (dordunde de, TAM)
        menzil 3:  1.540e-02 ... 2.912e-02
        menzil 4:  1.598e-02 ... 2.951e-02
        menzil 5:  2.881e-02 ... 3.492e-02      menzil 6: 2.918e-02

    (N = 6, 8, 10, 12; ayni kosumda max|dMatris| = 3.000e-01 ve matrisin
    simetri ihlali 3.000e-01 -- on sekiz kosumun on sekizinde de -- yani
    matris FENA halde bozuk oldugu halde tayf menzil 1-2'de kipirdamiyor.

    BU IKI SATIR BIR KEZ YANLIS YAZILDI, DOGRULAMA TURUNDA YAKALANDI: once
    "menzil 3: 1.70e-02 ... 3.49e-02, menzil 4+: 1.60e-02 ... 2.95e-02"
    yaziyordu. Ikisi de tekrarlanmadi -- 3.49e-02 aslinda N=12'nin menzil 5
    degeri, menzil 3'un tabani ise 1.540e-02 (N=12). Yukaridakiler menzil
    menzil yeniden olculmustur.)

    Yani korluk TAM ama yalnizca menzil 1-2'de -- ve SEVK EDILEN VARSAYILAN
    MENZIL 2'DIR. Bunun sonucu mutasyon sinavinda isim isim goruldu: M5
    altinda `test_KESME_HATASI...`in DORT parametresi de ve
    `test_ALFA_ve_BETA...` GECTI (hepsi menzil <= 2 kosuyor); mutanti
    olduren sey bu test ile menzil 3-4'e ulasan iki testtir. Planin sevk
    ettigi iki test de menzil <= 2'de kaldigi icin, bu dosya yalnizca
    onlardan ibaret olsaydi M5 TAMAMEN KACARDI.

    Tarama SCF'siz oldugu icin genis tutuldu: 6 halka boyu x her menzil
    (1..n/2) x iki spin = 66 kombinasyon olculdu, BOZUK 0.
    """
    for menzil in range(1, n // 2 + 1):
        t = tuple(0.3 / (1.7**k) * (-1.0) ** k for k in range(menzil))
        params = _elle_parametre(n, t)
        matris = finite_ring_matrix(model_from_downfold(params, spin=spin), n // 2)
        beklenen = _beklenen_site_matrisi(params, spin=spin)
        fark = np.abs(matris - beklenen)
        if fark.max() > 1e-15:
            i, j = np.unravel_index(int(np.argmax(fark)), matris.shape)
            pytest.fail(
                f"N={n} menzil={menzil} spin={spin}: ({i},{j}) yanlis baglandi -- "
                f"model {matris[i, j]:+.9f}, beklenen {beklenen[i, j]:+.9f}"
            )
        assert np.abs(matris - matris.T).max() < 1e-15, (
            f"N={n} menzil={menzil}: matris simetrik degil -- eslenik hoplama eksik"
        )


@pytest.mark.parametrize("n", [6, 8, 10, 12])
def test_ANTIPODAL_menzil_hoplamayi_IKI_KEZ_BAGLAMIYOR(n):
    """SINIF: TRANSKRIPSIYON DENETIMI. PLANIN GOVDESINDEKI HATA BURADA.

    Halka mesafesi m < n/2 iken her sitenin IKI ortagi vardir (i+m ve i-m);
    m = n/2 iken bu ikisi CAKISIR ve ortak TEKTIR. Planin gonderdigi govde
    her menzil icin dort (tek m) ya da iki (cift m) hoplama kaydi yaziyordu;
    m = n/2'de bu kayitlar AYNI cifte dusuyor ve `finite_ring_matrix` onlari
    TOPLUYOR -- yani matrise t yerine 2t giriyor.

    OLCULDU (duzeltmeden once, t_{n/2} degerleri gercek UHF'ten):
        N=6,  menzil 3, (0,3): +0.006946 yazildi, +0.003473 olmaliydi
        N=8,  menzil 4, (0,4): +0.006293 yazildi, +0.003147 olmaliydi
        N=10, menzil 5, (0,5): -0.005788 yazildi, -0.002894 olmaliydi
    Ucunde de tam olarak IKI kat, yani hata bir isaret ya da kayma degil
    saf cift-sayimdir.

    Bu hal ACIK API'den erisilebilir: `downfold_uhf` n_neighbour'a
    n_site//2'ye kadar izin verir.
    """
    params = _elle_parametre(n, tuple(0.1 for _ in range(n // 2)))
    matris = finite_ring_matrix(model_from_downfold(params), n // 2)
    karsi = n // 2
    for i in range(n):
        assert matris[i, (i + karsi) % n] == pytest.approx(0.1, abs=1e-15), (
            f"N={n}: antipodal eleman ({i},{(i+karsi)%n}) = "
            f"{matris[i, (i+karsi)%n]:+.9f}, beklenen +0.100000000"
        )


def test_GECERSIZ_spin_REDDEDILIYOR():
    """Sessizce beta modeli dondurmek YASAK.

    `yon = 1.0 if spin == "alpha" else -1.0` satiri, dogrulama kaldirilirsa
    TANIMSIZ her degeri beta'ya cevirir -- yani 'Alpha' ya da 'a' yazan bir
    cagiran, hicbir uyari almadan TERS kademeli alanla calisir.
    """
    params = _elle_parametre(8, (0.3, -0.05))
    with pytest.raises(InvalidInputError, match="spin"):
        model_from_downfold(params, spin="gamma")
    with pytest.raises(InvalidInputError, match="spin"):
        model_from_downfold(params, spin="Alpha")


def test_MENZIL_yarim_halkayi_ASAMAZ():
    """m > n/2 diye bir kabuk YOKTUR: halka mesafesi n/2'de doyar.

    `downfold_uhf` bunu zaten kapatiyor, ama `DownfoldedParameters` cipak bir
    dataclass'tir ve elle kurulabilir. Kapatilmasaydi m ve n-m ayni cifte
    yazardi (ortusme) ve model sessizce yanlis olurdu.
    """
    with pytest.raises(InvalidInputError, match="menzil"):
        model_from_downfold(_elle_parametre(8, (0.3, -0.05, 0.01, 0.0, 0.0)))


# ---------------------------------------------------------------------------
# FIZIK sayilariyla kosan testler. `slow` isaretleri OLCULDU (asagida).
# ---------------------------------------------------------------------------


def test_ALFA_ve_BETA_modelleri_AYNI_spektrumu_veriyor():
    """Kademeli alanin YONU spin ile doner ama SPEKTRUM ayni kalir --
    antiferromanyetik cozumun iki spin kanali enerjide esdegerdir.

    OLCULDU (N=8, d=2.0, STO-3G, seeds=12):
        delta              = +0.193177918
        max|ozdeger farki| = 6.661e-16      (16 hanede ayni)
        max|kosegen farki| =  0.386355836   = 2*|delta|, BIREBIR
        max|hoplama farki| = 0.000e+00      (hoplama spinden bagimsiz)

    Kosegen farki neden ayrica kilitleniyor: `spin` argumanini tumden yok
    sayan bir mutant (yon her zaman +1) ozdeger testini GECERDI, cunku iki
    ozdes model elbette ayni tayfi verir. Ayirt eden sey kosegenin gercekten
    donmesidir; sinir 2*|delta|'nin kendisine baglanir, keyfi bir 1e-3'e
    degil.

    `slow` DEGIL: olculen sure asagida, dosyanin sonundaki not.
    """
    params = downfold_uhf(_halka(8, 2.0), "sto-3g", seeds=12)
    a = finite_ring_matrix(model_from_downfold(params, spin="alpha"), 4)
    b = finite_ring_matrix(model_from_downfold(params, spin="beta"), 4)

    assert (
        np.abs(np.sort(np.linalg.eigvalsh(a)) - np.sort(np.linalg.eigvalsh(b))).max()
        < 1e-12
    )
    assert np.abs(np.diag(a) - np.diag(b)).max() == pytest.approx(
        2.0 * abs(params.delta), abs=1e-12
    )
    kosegensiz_a = a - np.diag(np.diag(a))
    kosegensiz_b = b - np.diag(np.diag(b))
    assert np.abs(kosegensiz_a - kosegensiz_b).max() < 1e-15


@pytest.mark.parametrize(
    "n,d,yayilim,nn_hata,t2_hata",
    [
        (6, 2.60, 1.113021, 0.039420, 0.012078),
        (6, 3.60, 0.834770, 0.010104, 0.003925),
        pytest.param(8, 2.00, 1.693171, 0.153603, 0.066287, marks=pytest.mark.slow),
        pytest.param(10, 2.60, 1.140294, 0.059631, 0.020470, marks=pytest.mark.slow),
    ],
)
def test_KESME_HATASI_olculdu_ve_t2_onu_IKI_UC_KAT_dusuruyor(
    n, d, yayilim, nn_hata, t2_hata
):
    """SINIF: TRANSKRIPSIYON DENETIMI -- modelin iyiligini OLCER, fizik
    IDDIA ETMEZ. Amac gerilemeyi yakalamaktir, kesme hatasini kirmizi
    yapmak DEGIL: hata buyuktur ve buyuk oldugu YAZILIDIR.

    Olculen tam tablo dosyanin ust dokstringindedir (alti sistem). Bu test
    onun dordunu kosar.

    "UC KAT" DEGIL, IKI-UC KAT: plan ve `downfold_uhf`in ilk dokstringi
    "t2 hatayi ~3 kat dusurur" diyordu. Alti sistemde olculen oran
    2.25 / 2.32 / 2.57 / 2.61 / 2.91 / 3.26 -- yani 3 kat yalnizca BIR
    sistemde tutuyor. Test olculen ALT sinira baglanir (oran > 2.0) ve
    "uc kat" iddiasi kaynaktan cikarildi.

    Oran, HARD-CODE degerlerden degil KOSUMDA olculen iki hatadan hesaplanir;
    `assert t2_hata < nn_hata / 2.0` yazmak yalnizca yukaridaki iki sabitin
    birbirine oranini denetlerdi ve kodun bir tek satirini bile calistirmazdi.

    TOLERANS rel=1e-3'tur ve SECILMEDI: `search_uhf` sabit tohumlarla
    deterministiktir ve iki ayri surecte kosulan olcum 12 hanede birebir
    ayni cikti (dogrulama turunda olculdu, sapma 0.000e+00). rel=1e-3 bu
    yuzden gurultuye degil GERILEMEYE ayarlidir; planin onerdigi rel=0.05
    ise hatayi %5 buyuten bir kusuru gecirirdi.
    """
    molecule = _halka(n, d)
    hedef = np.sort(
        search_uhf(molecule, "sto-3g", seeds=16).result.orbital_energies_alpha
    )
    assert float(hedef.max() - hedef.min()) == pytest.approx(yayilim, rel=1e-3)

    olculenler = []
    for n_neighbour, beklenen in ((1, nn_hata), (2, t2_hata)):
        params = downfold_uhf(molecule, "sto-3g", n_neighbour=n_neighbour, seeds=16)
        matris = finite_ring_matrix(model_from_downfold(params, spin="alpha"), n // 2)
        olculen = float(np.abs(np.sort(np.linalg.eigvalsh(matris)) - hedef).max())
        assert olculen == pytest.approx(beklenen, rel=1e-3), (
            f"N={n}, d={d}, menzil={n_neighbour}: {olculen:.6f} != {beklenen:.6f}"
        )
        olculenler.append(olculen)

    assert olculenler[0] / olculenler[1] > 2.0, (
        f"N={n}, d={d}: t2 hatayi yalnizca {olculenler[0]/olculenler[1]:.2f} kat "
        f"dusuruyor; olculen alt sinir 2.25 idi"
    )


@pytest.mark.slow
def test_KESME_HATASININ_KAYNAGI_MENZIL_DEGIL_KABUK_ORTALAMASI():
    """SINIF: TRANSKRIPSIYON DENETIMI -- hatanin NEREDE oldugunu yerlestirir.

    Bu test bir sasirtmacayi kapatir: "menzili buyutunce hata sifira gider"
    beklentisi YANLIS ve olculdu. N=8 d=2.0'da TAM menzilde (m=4, yani hicbir
    sey kesilmemisken) hata hala 0.063140 -- menzil 2'nin 0.066287'sinden
    yalnizca %5 asagida.

    Sebep OLCULDU ve iki parcali:

    (a) INDIRGEMENIN KENDISI KAYIPSIZ. H_eff^alfa'nin ozdegerleri UHF alfa
        orbital enerjileriyle birebir ayni: N=6 3.490e-10, N=8 7.194e-10,
        N=10 2.404e-09. Yani Lowdin adimi hicbir sey kaybetmiyor.

    (b) KAYIP KABUK ORTALAMASINDA. Model her halka mesafesi m icin TEK bir
        t_m tasir, ama H_eff^alfa cift m'de A-A ile B-B'yi AYIRIYOR -- cunku
        alfa kanalinda A alt-orgusu alfa fazlasi, B beta fazlasi tasir ve
        ikisi esdeger DEGILDIR. Tek m'de kabuk tekbicim, cunku o hoplamalar
        A<->B'yi baglar ve AFM islemi altinda simetriktir. OLCULDU
        (N=8, d=2.0, kabuk ici yayilim):
            m=1  3.004e-09   (tekbicim, SCF artik tabani)
            m=2  5.106e-02   A-dan +0.021274, B-den +0.072335
            m=3  1.007e-09   (tekbicim)
            m=4  2.416e-02   A-dan +0.015226, B-den -0.008932

    Ikisi bir araya gelince SAYISAL bir ozdeslik veriyor ve o da olculdu:
    tam menzilde max|model - H_eff^alfa|, en buyuk CIFT kabugun yayiliminin
    TAM YARISIDIR (model ortalamayi koyar, elemanlar ortalamanin +-yarim
    yayilim uzagindadir):
        N=6  2.134e-02 / 2 = 1.067e-02   (olculen artik 1.067e-02)
        N=8  5.106e-02 / 2 = 2.553e-02   (olculen artik 2.553e-02)
        N=10 2.420e-02 / 2 = 1.210e-02   (olculen artik 1.210e-02)

    HUKUM: iki-siteli modelin bu sistemdeki tavani menzille DEGIL, model
    SEKLIYLE belirlidir. Hatayi gercekten dusurmek icin gereken sey t3
    degil, cift menzilde alt-orguya BAGLI iki ayri hoplamadir (t2^AA,
    t2^BB) -- ve o, dort parametreli modelin kapsami DISINDADIR.

    `slow` ISARETI OLCUM: dosyanin sonundaki nota bakiniz.
    """
    n, d = 8, 2.0
    molecule = _halka(n, d)
    h_alpha, _, _, _, _ = _effective_hamiltonians(molecule, "sto-3g", 16)
    hedef = np.sort(
        search_uhf(molecule, "sto-3g", seeds=16).result.orbital_energies_alpha
    )

    # (a) indirgeme kayipsiz
    assert np.abs(np.sort(np.linalg.eigvalsh(h_alpha)) - hedef).max() < 1e-8

    # (b) kabuklar: TEK m tekbicim, CIFT m degil
    yayilimlar = {}
    for m in range(1, n // 2 + 1):
        elemanlar = np.array(
            [
                h_alpha[i, j]
                for i in range(n)
                for j in range(n)
                if min(abs(i - j), n - abs(i - j)) == m
            ]
        )
        yayilimlar[m] = float(elemanlar.max() - elemanlar.min())
    for m in (1, 3):
        assert yayilimlar[m] < 1e-8, f"tek kabuk m={m} tekbicim degil"
    for m in (2, 4):
        assert yayilimlar[m] > 1e-2, f"cift kabuk m={m} beklenenden dar"

    # artik = en buyuk CIFT kabugun yariminin TA KENDISI
    params = downfold_uhf(molecule, "sto-3g", n_neighbour=n // 2, seeds=16)
    matris = finite_ring_matrix(model_from_downfold(params, spin="alpha"), n // 2)
    artik = float(np.abs(matris - h_alpha).max())
    assert artik == pytest.approx(max(yayilimlar[2], yayilimlar[4]) / 2.0, rel=1e-6)

    # ve tam menzil hatasi hala buyuk
    tam_hata = float(np.abs(np.sort(np.linalg.eigvalsh(matris)) - hedef).max())
    assert tam_hata == pytest.approx(0.063140, rel=1e-3)
    assert tam_hata > 0.9 * 0.066287, "tam menzil hatayi anlamli olcude dusurdu"


@pytest.mark.slow
def test_MAX_METRIGI_MENZIL_3E_KOR_RMS_DEGIL():
    """SINIF: TRANSKRIPSIYON DENETIMI -- yukaridaki testlerin KULLANDIGI
    metrigin sinirini yazar.

    Bu deponun tekrarlayan dersi: "max|fark| olcusu yaniltici olabilir"
    (Modul 2E'de bir kazanc tam bu yuzden elenmisti). Burada ayni sey
    OLCULDU ve bu kez metrigi ELEMIYORUZ, sinirini BEYAN EDIYORUZ.

    N=8, d=2.0'da menzil 2'den 3'e gecince:
        max  0.066286861 -> 0.066286861   (DOKUZ hanede DEGISMIYOR)
        rms  0.034019579 -> 0.032088498   (%5.7 iyilesiyor)

    Yani t3, sekiz ozdegerin ALTISINI kaydiriyor ama max metrigini tasiyan
    iki ozdegeri (menzil 2 ve 3'te birebir -0.229467 ve +0.156889) HIC
    kipirdatmiyor. max metrigine bakarak "t3 hicbir sey katmiyor" demek bu
    yuzden YANLIS olurdu; dogru cumle "t3'un kattigi sey max metrigine
    gorunmuyor ve rms'te %5.7"dir.

    Bu, menzili 2'de kesme kararini GECERSIZ KILMAZ -- %5.7 kucuktur ve
    tavani zaten kabuk ortalamasi belirliyor (bir onceki test) -- ama
    kararin gerekcesini duzeltir.
    """
    n, d = 8, 2.0
    molecule = _halka(n, d)
    hedef = np.sort(
        search_uhf(molecule, "sto-3g", seeds=16).result.orbital_energies_alpha
    )

    olcumler = {}
    for menzil in (2, 3):
        params = downfold_uhf(molecule, "sto-3g", n_neighbour=menzil, seeds=16)
        matris = finite_ring_matrix(model_from_downfold(params, spin="alpha"), n // 2)
        fark = np.sort(np.linalg.eigvalsh(matris)) - hedef
        olcumler[menzil] = (
            float(np.abs(fark).max()),
            float(np.sqrt((fark**2).mean())),
        )

    assert olcumler[2][0] == pytest.approx(olcumler[3][0], abs=1e-9), (
        f"max metrigi t3'e artik KOR DEGIL: {olcumler[2][0]:.9f} -> "
        f"{olcumler[3][0]:.9f}; dokstringdeki hukum yeniden olculmeli"
    )
    assert olcumler[3][1] < olcumler[2][1], "rms t3 ile iyilesmiyor"
    assert olcumler[2][1] == pytest.approx(0.034019579, rel=1e-3)
    assert olcumler[3][1] == pytest.approx(0.032088498, rel=1e-3)


# ---------------------------------------------------------------------------
# `slow` ISARETLERININ GEREKCESI -- OLCUM, KURAL DEVRALMA DEGIL.
#
# Bu deponun isaretleme siniri Task 2'de olculdu ve DEVRALINABILIR BIR ESIK
# OLMADIGI gorulmustu: `-m slow` seridinin en hizli cagrisi 0.01 s,
# `-m "not slow"` seridinin en yavasi 4.75 s -- bantlar tamamen cakisiyor.
# Isaret, testin suresinden cok ait oldugu fixture/parametre ailesinin
# maliyetini izliyor. Task 2 bu yuzden bir esik SECTI ve gerekcesini yazdi:
# kendi call suresi 2 s ve USTUNDE olan durum isaretlenir. Bu dosya ayni
# esigi kullanir; sayilar dosyanin kendi --durations cikisindan gelir ve
# rapor bolumunde yeniden verilmistir.
#
# Yapisal testlerin (esleme, antipodal, spin, menzil) hicbiri SCF kosmaz ve
# hepsi milisaniye mertebesindedir; isaretsiz seride kalmalari bilinclidir --
# `model_from_downfold`un yapisini bozan her mutant hizli seritte olur.
# ---------------------------------------------------------------------------
