"""YAKINSAMA MERDİVENİ — §4.3'ün pazarlık edilmeyen makinesi.

Bu dosyanın konusu bir ÖLÇÜTtür: ne şema çıpası ne fizik çıpası, ikisinin de
HÜKMÜNÜ veren alet. Task 3, 4 ve 5'in her fizik çıpası onu tüketecek;
dolayısıyla burada gevşek bırakılan her şey aşağıdaki üç görevin tamamına
sızar.

MERDİVENLER SENTETİK DEĞİL. Aşağıdaki `_merdiven_*` üreticileri Task 1'in
SEVK EDİLEN çekirdeğini (`yee._adim`, `yee_coefficients`) koşturur; PEC
duvarlı ızgarada duran kip başlatılır ve `omega` üç terimli yinelemeden ya da
FFT tepesinden okunur. Kapalı form Yee dispersiyonu HİÇBİR YERDE
kullanılmaz — kullanılsaydı hem ölçüm hem yer gerçeği daireselleşirdi.

YER GERÇEĞİ = BİLİNEN SÜREKLİ LİMİT. Ölçülen büyüklük `v_p/c`dir ve sürekli
limiti TAM olarak 1.0'dır. Merdivenin gerçek mertebesi bu limitten okunur;
`convergence_ladder` limiti HİÇ GÖRMEZ. Bu, ölçütün karnesinin dairesel
olmamasının koşuludur.

PENCERE — bu dosyadaki her koşum için:
- sevk edilen 1B Yee çekirdeği, `dx = 1` hücre, `c = eps0 = mu0 = 1`,
  PEC duvarlı ızgara, vakum (`eps_r = mu_r = 1`), tohum YOK (FDTD
  deterministtir; RNG yalnız tautoloji kapısında ve ortak kip ölçümünde var
  ve orada tohum açıkça verilir).
- merdiven derinliği 4, `r = 2`, `N_kaba` = 101 veya 201, dolayısıyla
  `N_ince <= 1601`. Bu üst sınır ÖLÇÜLMÜŞ bir sınırdır: 1B dispersiyon
  çıpasının ince ucu float tabanına çarpıyor.
- `spacings` = `1/(N-1)`, yani birim uzunlukta bir boşluk. Oran TAM 2'dir
  (korpusta ölçülen en büyük sapma 0.0e+00).

`slow` İŞARETİ ÖLÇÜLEREK KONDU. `tests/em`de fixture yok, yani `setup ~ 0`
ve `setup + call = call` (Task 1'de ölçülmüştü, bu dosyada yeniden
doğrulandı). Ölçülen call süreleri dosyanın sonundaki tabloda.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.em.boundary import PEC
from qlab.engine.em.grid import FieldState
from qlab.engine.em.medium import Medium
from qlab.engine.em.yee import _adim, yee_coefficients
from qlab.errors import InvalidInputError
from qlab.observable.em_obs import (
    MIN_DEPTH,
    VERDICTS,
    ConvergenceLadder,
    _gurultu_belirsizligi,
    convergence_ladder,
)

# ---------------------------------------------------------------------------
# GERÇEK MERDİVEN ÜRETİCİLERİ — sevk edilen çekirdekten
# ---------------------------------------------------------------------------
#: İŞLEM-SONRASI pencere seçenekleri. `noise` için EK FDTD KOŞUMU YOKTUR:
#: hepsi TEK kaydedilmiş seriden kırpılarak üretilir.
_PENCERELER = [(frac, atla) for frac in (1.0, 0.85, 0.7) for atla in (0, 7, 15)]
_PENCERELER_FFT = [
    (pad, pen, frac)
    for pad in (4, 8, 16)
    for pen in ("dikdortgen", "hann")
    for frac in (1.0, 0.85, 0.7)
]


def _seri(N: int, m: int, S: float, M: int) -> np.ndarray:
    """PEC duvarlı ızgarada m. duran kipin izdüşüm serisi — SEVK EDİLEN çekirdek."""
    k = np.pi * m / (N - 1)
    proj = np.sin(k * np.arange(N))
    st = FieldState(proj.copy(), np.zeros(N - 1), 1)
    co = yee_coefficients(Medium.vacuum(N), courant=S, boundary=PEC())
    s = np.empty(M)
    for n in range(M):
        s[n] = float(np.dot(st.e, proj))
        st = _adim(st, co)
    return s


def _omega_uc_terimli(s: np.ndarray, S: float) -> float:
    """Tek frekanslı seri `s[n+1] + s[n-1] = 2 cos(omega dt) s[n]` sağlar."""
    y, x = s[2:] + s[:-2], 2.0 * s[1:-1]
    c = float(np.dot(x, y) / np.dot(x, x))
    return float(np.arccos(np.clip(c, -1.0, 1.0)) / S)


def _omega_fft(s: np.ndarray, dt: float, *, pad: int, pencere: str) -> float:
    """FFT tepesinden `omega` (parabolik ara değer) — spektral çıpaların okuma biçimi."""
    x = np.asarray(s, float)
    if pencere == "hann":
        x = x * np.hanning(len(x))
    n = int(len(x) * pad)
    A = np.abs(np.fft.rfft(x, n=n))
    j = int(np.argmax(A))
    delta = 0.0
    if 0 < j < len(A) - 1:
        a, b, c = A[j - 1], A[j], A[j + 1]
        delta = 0.5 * (a - c) / (a - 2.0 * b + c)
    return float(2.0 * np.pi * (j + delta) / (n * dt))


def _deger_ve_gurultu(s, S, k_etiket):
    ler = np.array(
        [
            _omega_uc_terimli(s[atla : int(round(len(s) * frac))], S) / k_etiket
            for frac, atla in _PENCERELER
        ]
    )
    return float(ler[0]), float(np.std(ler, ddof=1) / abs(np.mean(ler)))


def _deger_ve_gurultu_fft(s, dt, k_etiket):
    ler = np.array(
        [
            _omega_fft(s[: int(round(len(s) * frac))], dt, pad=pad, pencere=pen)
            / k_etiket
            for pad, pen, frac in _PENCERELER_FFT
        ]
    )
    return float(ler[0]), float(np.std(ler, ddof=1) / abs(np.mean(ler)))


def _araliklar(N0: int, derinlik: int = 4) -> list[float]:
    return [1.0 / (N0 - 1) / 2**i for i in range(derinlik)]


def _merdiven_dispersiyon(m=8, S=0.5, M=600, N0=201):
    """GERÇEK MERTEBE 2 — Yee'nin kendi yakınsaması, limit TAM 1.0."""
    d, g = [], []
    for i in range(4):
        N = (N0 - 1) * 2**i + 1
        v, e = _deger_ve_gurultu(_seri(N, m, S, M), S, np.pi * m / (N - 1))
        d.append(v)
        g.append(e)
    return d, _araliklar(N0), g


def _merdiven_etiket_hatasi(m=8, S=0.5, M=600, N0=201):
    """GERÇEK MERTEBE 1 — `k` bir hücre yanlış etiketli (`pi m/N`, doğrusu `pi m/(N-1)`).

    Spec §3.2'nin "lambda etiketleme hatası" ailesinin 1B karşılığı. Limit
    yine TAM 1.0; sapma `1/(N-1) = h` ile gidiyor, yani birinci mertebe.
    """
    d, g = [], []
    for i in range(4):
        N = (N0 - 1) * 2**i + 1
        v, e = _deger_ve_gurultu(_seri(N, m, S, M), S, np.pi * m / N)
        d.append(v)
        g.append(e)
    return d, _araliklar(N0), g


def _merdiven_olceklenmeyen(m0=8, S=0.5, M=600, N0=201):
    """GERÇEK MERTEBE 0 — çözünürlük düğmesi ÖLÇEKLENMİYOR.

    `m` ile `N` birlikte iki katına çıkar, yani `k dx` SABİT kalır ve
    dispersiyon hatası hiç küçülmez.
    """
    d, g = [], []
    for i in range(4):
        N, m = (N0 - 1) * 2**i + 1, m0 * 2**i
        v, e = _deger_ve_gurultu(_seri(N, m, S, M), S, np.pi * m / (N - 1))
        d.append(v)
        g.append(e)
    return d, _araliklar(N0), g


def _merdiven_spektral(m=8, S=0.5, M0=600, N0=201, sabit_kayit=False):
    """AYNI FİZİK, SPEKTRAL OKUMA — ve limit ARTIK 1.0 DEĞİL.

    FFT tepesinin kendi yanlılığı `h`ten bağımsız bir taban bırakır; değer
    2. mertebeden ama YANLIŞ bir sayıya oturur. Ölçütün var oluş sebebi
    tam olarak bu ayrımdır.
    """
    d, g = [], []
    for i in range(4):
        N = (N0 - 1) * 2**i + 1
        M = M0 if sabit_kayit else M0 * 2**i
        v, e = _deger_ve_gurultu_fft(_seri(N, m, S, M), S, np.pi * m / (N - 1))
        d.append(v)
        g.append(e)
    return d, _araliklar(N0), g


def _p_yer_gercegi(degerler, limit=1.0, r=2.0):
    """YER GERÇEĞİ: BİLİNEN sürekli limitten okunan mertebe (en ince çift)."""
    e = np.abs(np.asarray(degerler, float) - limit)
    if e[-1] <= 0.0 or e[-2] <= 0.0:
        return 0.0
    return float(np.log(e[-2] / e[-1]) / np.log(r))


def _hukum(L: ConvergenceLadder) -> str:
    return f"OTURDU({L.settled_order})" if L.verdict == "OTURDU" else L.verdict


# ---------------------------------------------------------------------------
# 1. ÜÇÜNCÜ DEĞER — dosyanın en önemli testi
# ---------------------------------------------------------------------------
def test_UCUNCU_DEGER_YARGILANAMAZ_ve_ONSUZ_YANLIS_POZITIF_CIKIYOR():
    """SINIF: ÖLÇÜM — ve ölçütün TAŞIYICI iddiası.

    Bayrak İKİ değerli olsaydı "yargılayamam" ile "oturmadı" aynı sayılırdı.
    Modül 4'ün `correlation_length` kırıldığında `None` döndürmesiyle AYNI
    STATÜDEDİR: `max(oran, 0)` nasıl yasaksa, yargılayamadığını "oturmadı"
    diye raporlamak da yasaktır.

    BURADAKİ MERDİVEN GERÇEK: aynı fizik, SPEKTRAL okuma (FFT tepesi, pad
    ve pencere fonksiyonu işlem-sonrası düğmeler). Ölçüldü — değer sürekli
    limite (1.0) DEĞİL, ondan 3.899e-3 uzakta bir sayıya 2. mertebeden
    yakınsıyor; bilinen limitten okunan yer gerçeği |p_ref| < 0.3, yani
    fiziksel olarak YAKINSAMAMIŞTIR.

    `noise = 0` ile ölçüt "OTURDU(2)" diyor ve YANILIYOR. Ölçülen pencere
    gürültüsüyle "YARGILANAMAZ" diyor. 426 gerçek merdivende bu farkın
    bedeli: yanlış pozitif %16.4 (70/426) -> %0.0 (0/426).
    """
    degerler, araliklar, gurultu = _merdiven_spektral(m=8, S=0.5, M0=600, N0=201)

    # Bu merdiven fiziksel olarak YAKINSAMAMIŞTIR (yer gerçeği limitten okunur):
    assert abs(_p_yer_gercegi(degerler)) < 0.3
    assert abs(degerler[-1] - 1.0) > 3.0e-3

    L = convergence_ladder(degerler, araliklar, noise=gurultu)
    assert L.verdict == "YARGILANAMAZ"
    assert L.p_inf is not None  # hesaplanır ama HÜKÜM verilmez
    assert L.settled_order is None
    assert L.uncertainty >= 0.5

    # Aynı merdiven, gürültü beyan edilmezse: ÖLÇÜT YANILIYOR.
    yanlis = convergence_ladder(degerler, araliklar, noise=0.0)
    assert yanlis.verdict == "OTURDU"
    assert yanlis.settled_order == 2
    assert yanlis.p_inf == pytest.approx(L.p_inf, rel=1e-12)


def test_UC_DEGERIN_UCU_DE_GERCEK_MERDIVENDE_GORULUYOR():
    """SINIF: ÖLÇÜM. Bayrağın üç değeri de KULLANILIYOR, biri süs değil.

    Üç ayrı gerçek merdiven, üç ayrı hüküm — ve hepsi aynı çekirdekten:
        dispersiyon (gerçek mertebe 2)        -> OTURDU(2)
        ölçeklenmeyen düğme (gerçek mertebe 0)-> YARGILANAMAZ
        yüksek mertebe kirlenmesi (aşağıda)   -> OTURMADI
    """
    d1, a1, g1 = _merdiven_dispersiyon()
    d0, a0, g0 = _merdiven_olceklenmeyen()
    h = np.array(_araliklar(201))
    # Kirlenmiş merdiven: baş terim h^2, üstüne 500 h^3 (ölçüldü, aşağıda).
    kirli = 3.0 + 0.7 * h**2 + 500.0 * h**3

    assert _hukum(convergence_ladder(d1, a1, noise=g1)) == "OTURDU(2)"
    assert convergence_ladder(d0, a0, noise=g0).verdict == "YARGILANAMAZ"
    assert convergence_ladder(kirli, h, noise=0.0).verdict == "OTURMADI"
    assert set(VERDICTS) == {"OTURDU", "OTURMADI", "YARGILANAMAZ"}


# ---------------------------------------------------------------------------
# 2. KAPPA — TOLERANS DEĞİL, KARAR SINIRI
# ---------------------------------------------------------------------------
def test_KAPPA_TOLERANS_DEGIL_KARAR_SINIRI_ve_PLATOSU_OLCULDU():
    """`kappa = 0.5` elle seçilmiş bir TOLERANS DEĞİLDİR.

    Bu şemada rakip mertebeler AYRIKTIR: Yee 2, merdivenlenmiş/etiketi
    kaymış sınır 1, ölçeklenmeyen çözünürlük düğmesi 0. Dolayısıyla 0.5 iki
    tamsayı arasındaki ORTA NOKTADIR. Bu bir MODELLEME SEÇİMİDİR ve
    Courant'ın "S en hızlı dalga hızıyla tanımlıdır" beyanıyla aynı
    statüdedir; panelde öyle yazılır.

    VE KIRILGAN OLMADIĞI ÖLÇÜLDÜ (426 gerçek merdiven, sentetik YOK):
        kappa   doğru pozitif    yanlış pozitif
        0.20    %80.3            %0.0
        0.40    %84.1            %0.0
        0.50    %84.1            %0.0     <- SEÇİLEN
        0.60    %84.1            %0.0
        0.70    %84.1            %0.2
        0.80    %85.4            %0.5
    0.40 - 0.60 arasında DÜZ BİR PLATO var.

    Bu test o platoyu, iki UCUNU DA TAŞIYAN ÜÇ GERÇEK MERDİVENLE yeniden
    üretir (tam korpus `test_KORPUS_KARNESI...`ta, `slow`):
      A — dispersiyon, her kappa'da OTURDU(2)  (plato içinde de dışında da)
      B — etiket hatası (m=8, S=0.3, N0=101): u = 0.2127, |p_inf - 1| =
          0.0614. kappa <= 0.2'de KAYBOLUYOR, 0.3'ten sonra OTURDU(1).
      C — spektral (m=16, S=0.9, N0=101): u = 0.6193. kappa <= 0.6'da
          YARGILANAMAZ (doğrusu bu), kappa >= 0.7'de YANLIŞ POZİTİF.
    """
    A = _merdiven_dispersiyon(m=8, S=0.5, M=600, N0=201)
    B = _merdiven_etiket_hatasi(m=8, S=0.3, M=400, N0=101)
    C = _merdiven_spektral(m=16, S=0.9, M0=600, N0=101)

    # B ve C'nin ölçülen belirsizlikleri — platonun iki ucunu bunlar kuruyor.
    LB = convergence_ladder(*B[:2], noise=B[2])
    LC = convergence_ladder(*C[:2], noise=C[2])
    assert LB.uncertainty == pytest.approx(0.2127, abs=5e-4)
    assert abs(LB.p_inf - 1.0) == pytest.approx(0.0614, abs=5e-4)
    assert LC.uncertainty == pytest.approx(0.6193, abs=5e-4)

    # C'nin doğru cevabı OTURDU DEĞİLDİR: bilinen limitten okunan mertebe
    # hiçbir tamsayıya yakın değil.
    p_ref_C = _p_yer_gercegi(C[0])
    assert abs(p_ref_C - round(p_ref_C)) > 0.25

    beklenen = {
        0.10: ("OTURDU(2)", "YARGILANAMAZ", "YARGILANAMAZ"),
        0.20: ("OTURDU(2)", "YARGILANAMAZ", "YARGILANAMAZ"),
        0.40: ("OTURDU(2)", "OTURDU(1)", "YARGILANAMAZ"),
        0.50: ("OTURDU(2)", "OTURDU(1)", "YARGILANAMAZ"),
        0.60: ("OTURDU(2)", "OTURDU(1)", "YARGILANAMAZ"),
        0.70: ("OTURDU(2)", "OTURDU(1)", "OTURDU(2)"),
    }
    for kappa, (hA, hB, hC) in beklenen.items():
        assert _hukum(convergence_ladder(*A[:2], noise=A[2], kappa=kappa)) == hA
        assert _hukum(convergence_ladder(*B[:2], noise=B[2], kappa=kappa)) == hB
        assert _hukum(convergence_ladder(*C[:2], noise=C[2], kappa=kappa)) == hC

    # PLATO: 0.4 - 0.6 arasında üç merdivenin de hükmü DEĞİŞMİYOR.
    assert beklenen[0.40] == beklenen[0.50] == beklenen[0.60]
    # İki uç: altında doğru pozitif kayboluyor, üstünde yanlış pozitif geliyor.
    assert beklenen[0.20][1] != beklenen[0.50][1]
    assert beklenen[0.70][2] != beklenen[0.50][2]


@pytest.mark.slow
def test_KORPUS_KARNESI_GERCEK_MERDIVENLERDE_ve_GURULTUSUZUN_BEDELI():
    """SINIF: ÖLÇÜM. Kırk gerçek merdivenlik bir karne — sentetik YOK.

    Yer gerçeği BİLİNEN sürekli limitten (v_p/c -> 1) okunur; ölçüt limiti
    hiç görmez, yani karne dairesel değildir.

    PENCERE: m = 8/16, S = 0.3/0.5/0.7/0.9, M = 600, N = 101 -> 801,
    beş aile (dispersiyon 2, etiket hatası 1, ölçeklenmeyen 0, spektral,
    spektral + sabit kayıt). ÖLÇÜLDÜ (üretim 0.91 s):

        kappa   gürültülü            gürültüsüz
        0.10    DP  9/20  YP 0/40    DP  9  YP 5/40
        0.20    DP 11/20  YP 0/40    DP 11  YP 7/40
        0.40    DP 13/20  YP 0/40    DP 13  YP 8/40
        0.50    DP 13/20  YP 0/40    DP 13  YP 8/40
        0.60    DP 13/20  YP 0/40    DP 13  YP 8/40
        0.70    DP 13/20  YP 1/40    DP 13  YP 8/40

    Doğru pozitif oranı bu alt korpusta %65 (13/20); 426 merdivenlik tam
    korpusta %84.1 (201/239). Fark PENCEREDEN gelir — burada `N_kaba` = 101
    ve etiket-hatası ailesinin kaba ucu yüksek mertebe kirlenmesiyle dolu.
    """
    korpus = []
    for m in (8, 16):
        for S in (0.3, 0.5, 0.7, 0.9):
            korpus.append(_merdiven_dispersiyon(m, S, 600, 101))
            korpus.append(_merdiven_etiket_hatasi(m, S, 600, 101))
            korpus.append(_merdiven_olceklenmeyen(m, S, 600, 101))
            korpus.append(_merdiven_spektral(m, S, 600, 101))
            korpus.append(_merdiven_spektral(m, S, 600, 101, sabit_kayit=True))
    assert len(korpus) == 40
    for degerler, _, gurultu in korpus:
        assert np.all(np.isfinite(degerler)) and np.all(np.isfinite(gurultu))

    def karne(kappa, gurultulu):
        dp = yp = pay = 0
        for degerler, araliklar, gurultu in korpus:
            p_ref = _p_yer_gercegi(degerler)
            yg = (
                int(round(p_ref))
                if abs(p_ref - round(p_ref)) < 0.25 and 0 <= round(p_ref) <= 8
                else None
            )
            L = convergence_ladder(
                degerler,
                araliklar,
                noise=(gurultu if gurultulu else 0.0),
                kappa=kappa,
            )
            if yg is not None and yg >= 1:
                pay += 1
                dp += L.verdict == "OTURDU" and L.settled_order == yg
            yp += L.verdict == "OTURDU" and L.settled_order != yg
        return dp, pay, yp

    assert karne(0.10, True) == (9, 20, 0)
    assert karne(0.20, True) == (11, 20, 0)
    assert karne(0.40, True) == (13, 20, 0)
    assert karne(0.50, True) == (13, 20, 0)
    assert karne(0.60, True) == (13, 20, 0)
    assert karne(0.70, True) == (13, 20, 1)  # PLATONUN ÜST UCU

    # GÜRÜLTÜSÜZ: aynı korpus, aynı kappa — sekiz yanlış pozitif.
    assert karne(0.50, False) == (13, 20, 8)
    # Gürültünün doğru pozitife bedeli bu korpusta SIFIR.
    assert karne(0.50, False)[0] == karne(0.50, True)[0]


# ---------------------------------------------------------------------------
# 3. TAUTOLOJİ KAPISI — Plan H'nin dersi, BEŞİNCİ uygulama
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ad, uretici, ust_oturdu2, ust_oturdu",
    [
        ("saf gurultu", lambda r: r.normal(0.0, 1.0, size=4), 0.001, 0.005),
        (
            "rastgele yuruyus",
            lambda r: np.cumsum(r.normal(0.0, 1.0, size=4)),
            0.003,
            0.015,
        ),
    ],
)
def test_SAF_GURULTUDE_OTURDU_DEMIYOR_TAUTOLOJI_KAPISI(
    ad, uretici, ust_oturdu2, ust_oturdu
):
    """Fiziği çıkar, boru hattını YERİNDE BIRAK.

    Bu depoda bir "kapı"nın fizik-dışı girdide geçtiği DÖRT kez ölçüldü; en
    sonuncusu (Mie'nin optik teorem kapısı) bir muhasebenin TAMAMINI
    dayanaksız bıraktı. Bu yüzden her ölçüt kendi tautoloji kapısından geçer.

    VE PLANIN "%0,00"I YENİDEN ÜRETİLEMEDİ. 5 tohum x 2000 = 10000 deneme,
    `h = 1/200/2^i`, `noise = 1e-6` (en elverişsiz ayar) ile ölçülen:

        girdi                      OTURDU(2)   OTURDU(herhangi)
        saf gürültü                %0.01 (1)   %0.17 (17)
        rastgele yürüyüş           %0.09 (9)   %0.73 (73)
        monoton rastgele yürüyüş   %0.33 (33)  %3.12 (312)

    Kapı %99.7'yi çeviriyor ama SIFIR DEĞİL. Plan/spec'in "%0,00"ı 2000
    denemede örneklenmiş bir sıfırdır — bu oranlarda 2000 denemede sıfır
    görmek beklenen sonuçtur, dolayısıyla o sayı yanlış değil AZ ÖRNEKLENMİŞ.

    Eşikler ölçülen orandan gelir, ölçütü geçirmek için gevşetilmedi:
    üst sınır her iki satırda da ölçülen oranın ~10 katıdır ve tek başına
    bir tautoloji değildir — testin ikinci yarısı AYNI boru hattının gerçek
    bir 2. mertebe merdivende OTURDU(2) dediğini gösterir.
    """
    h = _araliklar(201)
    say2 = say = 0
    n = 0
    for tohum in (1, 2, 3, 4, 5):
        r = np.random.default_rng(tohum)
        for _ in range(2000):
            L = convergence_ladder(uretici(r), h, noise=1e-6)
            n += 1
            if L.verdict == "OTURDU":
                say += 1
                say2 += L.settled_order == 2
    assert say2 / n < ust_oturdu2, f"{ad}: OTURDU(2) orani {say2}/{n}"
    assert say / n < ust_oturdu, f"{ad}: OTURDU(herhangi) orani {say}/{n}"

    # KAPI SAĞIRLAŞTIRILMADI: aynı boru hattı gerçek merdivende hüküm veriyor.
    d, a, g = _merdiven_dispersiyon()
    assert _hukum(convergence_ladder(d, a, noise=g)) == "OTURDU(2)"


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_BILINEN_MERTEBE_BIRIM_DENETIMI(p):
    """Bilinen mertebeli birim denetimi — `v = A + C h^p` doğru okunuyor mu.

    ÖLÇÜLDÜ (h = 1/200/2^i, A = 3.0, C = 0.7, gürültüsüz):
        p = 1 -> 1.000000000    p = 3 -> 2.999999264
        p = 2 -> 2.000000001    p = 4 -> 4.000667062
    p = 4'teki 6.7e-04'lük sapma bir hata değil, TAHMİNCİNİN TABANI:
    `d_i` farkları 1e-13 mertebesine iniyor ve oran float artığını görüyor.
    """
    h = np.array(_araliklar(201))
    L = convergence_ladder(3.0 + 0.7 * h**p, h, noise=0.0)
    assert L.verdict == "OTURDU"
    assert L.settled_order == p
    assert L.p_inf == pytest.approx(float(p), abs=1e-3)


# ---------------------------------------------------------------------------
# 4. DERİNLİK — bir SAYIM KİMLİĞİ, tercih değil
# ---------------------------------------------------------------------------
def test_DERINLIK_EN_AZ_DORT_ve_bu_bir_SAYIM_KIMLIGI():
    """SINIF: ÖLÇÜM + sayım kimliği.

    n değer -> n-1 fark -> n-2 mertebe -> n-3 Richardson noktası. Yani:
      n = 2: mertebe SAYISI 0 — referanssız mertebe ÜRETİLEMEZ.
      n = 3: mertebe var (1 tane) ama Richardson noktası 0 — BELİRSİZLİK
             ÜRETİLEMEZ.
      n = 4: `u`nun ölçülebildiği EN KÜÇÜK derinlik.
    Bu bir tolerans değil ARİTMETİĞİN TANIMLI OLMA KOŞULUDUR ve korpusta
    doğrulandı: derinlik 2'de 426/426, derinlik 3'te 426/426 merdiven
    `p_inf = None, uncertainty = None` döndürüyor.

    Aynı merdivenin dört basamağı OTURDU(2) verirken, ilk üçü
    YARGILANAMAZ vermek ZORUNDADIR — "üç nokta da yeter" diyen bir
    uygulama, belirsizliğini ölçemediği bir sayıyı hüküm gibi basardı.
    """
    for n in (1, 2, 3, 4):
        assert (n - 2 if n >= 2 else 0) == max(n - 2, 0)
    assert MIN_DEPTH == 4

    d, a, g = _merdiven_dispersiyon()
    tam = convergence_ladder(d, a, noise=g)
    assert _hukum(tam) == "OTURDU(2)"
    assert tam.uncertainty is not None

    for n in (1, 2, 3):
        kisa = convergence_ladder(d[:n], a[:n], noise=g[:n])
        assert kisa.verdict == "YARGILANAMAZ"
        assert kisa.p_inf is None
        assert kisa.uncertainty is None
        assert kisa.orders == ()
        assert kisa.settled_order is None
        # Kayıt yine de TAM taşınıyor — çağıran merdiveni görebilmeli.
        assert kisa.values == tuple(d[:n])


# ---------------------------------------------------------------------------
# 5. ORTAK KİP — cebirsel bağışıklık, ve karşıtı
# ---------------------------------------------------------------------------
def test_ORTAK_KIPLI_GURULTU_TAHMINCIYI_OYNATAMAZ_CEBIRSEL():
    """`p` farkların ORANINDAN okunur, dolayısıyla ortak kip sadeleşir.

    Çarpımsal ortak kipte `d_i -> d_i (1+delta)` ve oran değişmez;
    toplamsal ortak kipte `delta` FARKTA DÜŞER (`d_i = v_{i+1} - v_i`).

    ÖLÇÜLDÜ (301 gerçek merdiven x 40 deneme, aynı sigma ile ortak kip vs
    BASAMAĞA ÖZGÜ):
        sigma    ortak kip     basamağa özgü    oran
        1e-10    6.760e-10     1.097e-03        1.6e+06
        1e-08    7.189e-10     1.286e-01        1.8e+08
        1e-06    6.704e-10     1.238e+01        1.9e+10
    `delta ~ N(0, 0.3)` gibi DEVASA bir ortak kipte bile çarpımsalda
    5.231e-10, toplamsalda 8.947e-10 (15050 deneme).

    Bu, `noise`un neden bir ALT SINIR olduğunun da gerekçesidir: görmediği
    tam-ortak bileşen zaten mertebeyi oynatamaz.
    """
    rng = np.random.default_rng(20260824)
    d, a, g = _merdiven_dispersiyon()
    v = np.asarray(d, float)
    L0 = convergence_ladder(v, a, noise=g)

    carpimsal = toplamsal = 0.0
    for _ in range(400):
        dl = float(rng.normal(0.0, 0.3))
        carpimsal = max(
            carpimsal,
            abs(convergence_ladder(v * (1.0 + dl), a, noise=g).p_inf - L0.p_inf),
        )
        c = float(rng.normal(0.0, 0.3)) * float(np.max(np.abs(v)))
        toplamsal = max(
            toplamsal, abs(convergence_ladder(v + c, a, noise=g).p_inf - L0.p_inf)
        )
    assert carpimsal < 1e-8
    assert toplamsal < 1e-8

    # KARŞITI: aynı sigma BASAMAĞA ÖZGÜ verildiğinde tahminci oynuyor.
    ozgu = 0.0
    for _ in range(400):
        vv = v * (1.0 + rng.normal(0.0, 1e-8, size=v.size))
        L = convergence_ladder(vv, a, noise=g)
        if L.p_inf is not None:
            ozgu = max(ozgu, abs(L.p_inf - L0.p_inf))
    assert ozgu > 1e4 * max(carpimsal, toplamsal)


def test_GURULTU_BELIRSIZLIGI_YAYILIM_ILE_UYUSUYOR():
    """SINIF: TRANSKRİPSİYON DENETİMİ — kapalı form Jacobi zinciri doğru mu.

    `_gurultu_belirsizligi` doğrusal yayılımdır (Monte Carlo DEĞİL: bu
    modülde RNG yoktur). Doğruluğu, basamağa özgü küçük bozmaların ampirik
    saçılımıyla karşılaştırılarak sınanır — iki yol %10 içinde uyuşmalı.

    Doğrusal bölgede kalmak için `sigma = 1e-10` seçildi; daha büyüğünde
    yayılım varsayımı zaten geçersizdir (yukarıdaki testte ölçüldü).
    """
    h = np.array(_araliklar(201))
    v = 3.0 + 0.7 * h**2
    sigma = 1e-10
    analitik = _gurultu_belirsizligi(v, np.diff(v), np.full(4, sigma), 2.0)

    rng = np.random.default_rng(4242)
    orn = []
    for _ in range(4000):
        vv = v * (1.0 + rng.normal(0.0, sigma, size=4))
        L = convergence_ladder(vv, h, noise=0.0)
        if L.p_inf is not None:
            orn.append(L.p_inf)
    ampirik = float(np.std(orn, ddof=1))
    assert ampirik == pytest.approx(analitik, rel=0.10)

    # ÖLÇEK: belirsizlik `noise` ile DOĞRUSAL — iki katı gürültü iki katı u.
    iki = _gurultu_belirsizligi(v, np.diff(v), np.full(4, 2 * sigma), 2.0)
    assert iki == pytest.approx(2.0 * analitik, rel=1e-12)


# ---------------------------------------------------------------------------
# 6. "HANGİ tamsayıya" — spec §4.3'ün dördüncü düzeltmesi
# ---------------------------------------------------------------------------
def test_SORU_IKIYE_OTURDU_MU_DEGIL_HANGI_TAMSAYIYA_OTURDU():
    """Merdivenlenmiş/etiketi kaymış sınır için doğru cevap 2 DEĞİL 1'dir.

    GERÇEK MERDİVEN: `k` bir hücre yanlış etiketli (`pi m/N`, doğrusu
    `pi m/(N-1)`) — spec §3.2'nin "lambda etiketleme hatası" ailesinin 1B
    karşılığı. Bilinen limitten okunan yer gerçeği 1'e yakın; ölçüt de
    OTURDU(1) demeli.

    `verdict` üç değerlidir ve tamsayıyı TAŞIMAZ; tamsayı `settled_order`
    üzerinden okunur ve OTURDU dışında None'dır. None SIFIR DEĞİLDİR ve
    2 de değildir.
    """
    d, a, g = _merdiven_etiket_hatasi(m=8, S=0.5, M=600, N0=201)
    p_ref = _p_yer_gercegi(d)
    assert round(p_ref) == 1  # YER GERÇEĞİ: bilinen limitten

    L = convergence_ladder(d, a, noise=g)
    assert L.verdict == "OTURDU"
    assert L.settled_order == 1
    assert L.settled_order != 2

    # Aynı fizik, doğru etiketle: 2. mertebe. Fark ETİKETTEN geliyor.
    d2, a2, g2 = _merdiven_dispersiyon(m=8, S=0.5, M=600, N0=201)
    assert convergence_ladder(d2, a2, noise=g2).settled_order == 2

    # OTURDU dışında tamsayı YOK.
    d0, a0, g0 = _merdiven_olceklenmeyen()
    assert convergence_ladder(d0, a0, noise=g0).settled_order is None


def test_OLCEKLENMEYEN_DUGME_MERTEBE_URETMIYOR():
    """SINIF: ÖLÇÜM. Çözünürlük düğmesi ölçeklenmezse ızgara "iyileşmez".

    `m` ile `N` birlikte iki katına çıkınca `k dx` SABİT kalır ve
    dispersiyon hatası HİÇ küçülmez. ÖLÇÜLDÜ (m0 = 8, S = 0.5, M = 600,
    N = 201 -> 1601): dört basamağın hepsi `v_p/c = 0.9995063980868`, yani
    sürekli limitten sapma dördünde de 4.936e-04 ve basamaklar arası fark
    2.83e-14 — MAKİNE HASSASİYETİ, fizik değil.

    Doğru cevap "mertebe 0" değil, "YARGILANAMAZ"dır: ölçüt makine
    artığından bir mertebe UYDURMAZ. Bu, bir merdivenin "yakınsıyor gibi"
    görünmesinin tek yolunun `dx` küçültmek OLMADIĞINI da gösterir —
    ölçeklenmeyen bir düğme ızgarayı iyileştirmez.
    """
    d, a, g = _merdiven_olceklenmeyen(m0=8, S=0.5, M=600, N0=201)
    sapma = np.abs(np.asarray(d) - 1.0)
    assert np.all(np.abs(sapma - 4.936e-4) < 1e-6), "hata hiç küçülmemeliydi"
    assert np.max(np.abs(np.diff(d))) < 1e-13, "fark makine hassasiyetinde olmalı"

    L = convergence_ladder(d, a, noise=g)
    assert L.verdict == "YARGILANAMAZ"
    assert L.p_inf is None and L.orders == ()


def test_PENCERE_DISI_MERDIVEN_YARGILANAMAZ_ISARET_KAPISI():
    """SINIF: ÖLÇÜM. Ardışık farklar işaret değiştirirse mertebe YOKTUR.

    GERÇEK MERDİVEN: spektral okuma + kayıt uzunluğu hücre cinsinden SABİT,
    yani ince basamaklarda fiziksel kayıt süresi kısalıyor ve FFT çözünürlüğü
    BOZULUYOR. Ölçülen değerler monoton değil; `d_i d_{i+1} <= 0` kapısı
    dönüyor. Bu, spec'in "tek yarık çıpası ölçütün penceresinin DIŞINDA"
    hükmüyle aynı ailedendir: yargı yok, ama SESSİZ bir sayı da yok.
    """
    d, a, g = _merdiven_spektral(m=8, S=0.5, M0=600, N0=201, sabit_kayit=True)
    fark = np.diff(d)
    assert np.any(fark[:-1] * fark[1:] <= 0.0), "merdiven monoton çıktı"
    L = convergence_ladder(d, a, noise=g)
    assert L.verdict == "YARGILANAMAZ"
    assert L.p_inf is None
    # Gürültü sıfırlansa bile hüküm DEĞİŞMEZ: bu bir İMKAN kapısı.
    assert convergence_ladder(d, a, noise=0.0).verdict == "YARGILANAMAZ"


def test_FARKLARI_BUYUYEN_MERDIVEN_ISARET_KAPISINI_GECER_ama_P_KAPISI_TUTAR():
    """SINIF: ÖLÇÜM. İki ayrı imkân kapısı var ve İKİSİ DE gerekli.

    İşaret kapısı yalnız SALINAN merdivenleri tutar. IRAKSAYAN bir merdiven
    (farklar monoton ama BÜYÜYOR) işaret kapısından geçer ve `p < 0` verir —
    spec §7.7'nin "tek yarık çıpası ölçütün penceresinin DIŞINDA,
    `rho_son` = 0,798" hükmüyle aynı aile.

    GERÇEK MERDİVEN (spektral okuma + kayıt hücre cinsinden sabit, m = 4,
    S = 0.5, M = 400, N = 101 -> 801): `rho` = 0.2829 ve 0.3600, yani ikisi
    de 1'in ALTINDA ve farklar iki kat büyüyor. Ölçüldü:
        p = -1.8217, -1.4740     p_inf = -1.1263     u = 0.3477
    `p in (0, 8]` kapısı OLMASAYDI ölçüt `|p_inf - round(p_inf)| + u` =
    0.4740 < 0.5 ile "OTURDU(-1)" derdi — NEGATİF bir mertebe.

    Üst uç (`p > 8`) için GERÇEK bir örnek ÜRETİLEMEDİ: bu anahtardaki üç
    terimli `omega` tahmincisi N = 12801'e kadar `rho = 4.000`'de duruyor,
    yani plan/spec'in "N = 3200'de rho = 25,76" sayısı BU okumada yeniden
    üretilemiyor. Üst kapı bu yüzden kurulmuş bir merdivenle kilitleniyor
    ve bu, ölçüm değil TRANSKRİPSİYON kilididir.
    """
    d, a, _ = _merdiven_spektral(m=4, S=0.5, M0=400, N0=101, sabit_kayit=True)
    fark = np.diff(d)
    rho = fark[:-1] / fark[1:]
    assert np.all(fark[:-1] * fark[1:] > 0.0), "işaret kapısını GEÇMELİ"
    assert rho == pytest.approx([0.2829, 0.3600], abs=1e-3)
    assert np.all(rho < 1.0), "farklar büyümeliydi (ıraksayan merdiven)"

    L = convergence_ladder(d, a, noise=0.0)
    assert L.verdict == "YARGILANAMAZ"
    assert L.p_inf == pytest.approx(-1.1263, abs=1e-3)
    # Kapı olmasa hüküm verilirdi — ve NEGATİF bir mertebeye:
    assert abs(L.p_inf - round(L.p_inf)) + L.uncertainty < 0.5

    # Üst kapı (kurulmuş merdiven): p = 9 > 8.
    h = [0.5, 0.25, 0.125, 0.0625]
    ust = convergence_ladder([3.0 + 0.7 * x**9 for x in h], h, noise=0.0)
    assert ust.orders == pytest.approx((9.0, 9.0), abs=1e-6)
    assert ust.verdict == "YARGILANAMAZ"


def test_U_ESIT_KAPPA_SINIRINDA_YARGILANAMAZ():
    """SINIF: TRANSKRİPSİYON DENETİMİ. Spec §7.7 `u >= kappa` yazıyor, `>` değil.

    Sınır ölçülemez değil: aynı merdiven `kappa = u` ile çağrılırsa `u`
    TAM OLARAK `kappa`ya eşit olur (hesap deterministtir, FDTD'de RNG yok).
    O noktada hüküm YARGILANAMAZ'dır — "belirsizliğim tam karar sınırı kadar"
    demek, karar verebiliyorum demek DEĞİLDİR.

    ÖLÇÜLDÜ (dispersiyon merdiveni, m = 8, S = 0.5, M = 600, N = 201..1601):
    u = 2.500213e-04; `kappa = u` ile YARGILANAMAZ, `kappa` bunun bir ulp
    üstünde OTURMADI. OTURDU'ya geçiş AYRI ve daha yüksek bir eşiktedir
    (`|p_inf - round(p_inf)| + u` = 4.171e-04), yani üç hüküm ARDIŞIK
    kappa aralıklarına oturuyor ve sınırların ikisi de tam olarak
    ölçülebiliyor.
    """
    d, a, g = _merdiven_dispersiyon()
    L = convergence_ladder(d, a, noise=g)
    assert _hukum(L) == "OTURDU(2)"
    assert L.uncertainty == pytest.approx(2.500213e-04, rel=1e-4)

    sinirda = convergence_ladder(d, a, noise=g, kappa=L.uncertainty)
    assert sinirda.uncertainty == L.uncertainty
    assert sinirda.verdict == "YARGILANAMAZ"

    ustunde = convergence_ladder(d, a, noise=g, kappa=np.nextafter(L.uncertainty, 1.0))
    assert ustunde.verdict == "OTURMADI"

    # OTURDU'ya geçiş AYRI bir eşiktedir: `|p_inf - round(p_inf)| + u < kappa`.
    esik = abs(L.p_inf - round(L.p_inf)) + L.uncertainty
    assert convergence_ladder(d, a, noise=g, kappa=esik).verdict == "OTURMADI"
    assert (
        _hukum(convergence_ladder(d, a, noise=g, kappa=np.nextafter(esik, 1.0)))
        == "OTURDU(2)"
    )


def test_STD_TERIMI_DERINLIK_DORTTE_OLU_BESTE_CANLI():
    """DÜRÜST OLUMSUZ + ÖLÇÜM: `u`nun üç teriminden biri derinlik 4'te ÖLÜDÜR.

    Sayım kimliği: n değer -> n-3 Richardson noktası. Derinlik 4'te bu SAYI
    BİRDİR, dolayısıyla `std(p_inf_dizi)` tanımsızdır ve ölçüt onu 0 sayar.
    Yani spec §7.7'nin D1 formülündeki üçüncü terim, sevk edilen çıpaların
    kullandığı derinlikte HİÇBİR ZAMAN baskın olamaz. Bu, terimin yanlış
    olduğu anlamına gelmez — DERİNLİĞE BAĞLI olduğu anlamına gelir ve
    burada beyan edilir.

    Derinlik 5 bu modülde MÜMKÜNDÜR (spec'in "derinlik 5 imkânsız" hükmü 2B
    Bessel çıpasına aittir; 1B'de N = 51 -> 801 merdiveni 5 basamak verir).
    Ölçüldü — GERÇEK derinlik-5 merdivenlerinde `|p_inf - p_son|` terimi
    `std`i her seferinde geçiyor (7 pencerenin 7'sinde), çünkü tek kuvvetli
    bir düzeltmede `std` cebirsel olarak SIFIRA gider:
        p_i = p* + c 2^-i  =>  std(p_inf_dizi) = 0,  |p_inf - p_son| = |c|/4.
    `std` ancak ÜÇÜNCÜ bir kuvvet devreye girince baskın olur; o zaman da
    hükmü değiştirir (aşağıda).
    """
    # (a) Derinlik 4: Richardson noktası sayısı 1 -> std terimi ölü.
    d, a, g = _merdiven_dispersiyon()
    assert len(d) - 3 == 1

    # (b) Derinlik 5, GERÇEK merdiven: `|p_inf - p_son|` baskın.
    vals, gur, hs = [], [], []
    for i in range(5):
        N = 50 * 2**i + 1
        v, e = _deger_ve_gurultu(_seri(N, 4, 0.5, 600), 0.5, np.pi * 4 / (N - 1))
        vals.append(v)
        gur.append(e)
        hs.append(1.0 / 50 / 2**i)
    L5 = convergence_ladder(vals, hs, noise=gur)
    assert _hukum(L5) == "OTURDU(2)"
    p = np.asarray(L5.orders)
    q = 2.0 * p[1:] - p[:-1]
    assert abs(L5.p_inf - p[-1]) > float(np.std(q, ddof=1))

    # (c) Derinlik 5, ÜÇ kuvvetli merdiven: `std` baskın VE hükmü değiştiriyor.
    #     v = 3 + 0.7 h + 980 h^2 + 2800 h^3 ; kaba basamakta h^2 terimi baş
    #     terimin 7 katı, h^3 terimi 0.1 katı. GERÇEK baş mertebe 1'dir.
    h = np.array([1.0 / 200 / 2**i for i in range(5)])
    v = 3.0 + 0.7 * h + 980.0 * h**2 + 2800.0 * h**3
    L = convergence_ladder(v, h, noise=0.0)
    p = np.asarray(L.orders)
    q = 2.0 * p[1:] - p[:-1]
    t1, t2 = abs(L.p_inf - p[-1]), float(np.std(q, ddof=1))
    assert t1 == pytest.approx(0.049919, abs=1e-4)
    assert t2 == pytest.approx(0.127992, abs=1e-4)
    assert t2 > t1
    assert L.uncertainty == pytest.approx(t2, rel=1e-12)
    assert L.verdict == "OTURMADI"
    # `std` terimi düşürülseydi: u = t1 ve ölçüt YANLIŞ tamsayıya otururdu.
    assert abs(L.p_inf - 2.0) + t1 < 0.5
    assert round(L.p_inf) == 2  # baş mertebe 1 olduğu HÂLDE


# ---------------------------------------------------------------------------
# 7. DÜRÜST OLUMSUZLAR
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p, gurultusuz_tamsayi", [(0.5, 0), (1.5, 1), (2.5, 3)])
def test_YARIM_TAMSAYI_KOR_NOKTASI_BICAK_SIRITIDIR(p, gurultusuz_tamsayi):
    """DÜRÜST OLUMSUZ — ve spec'in yazdığından KÖTÜ.

    Spec "p tam 0,5/1,5/2,5 ise ölçüt karar sınırının üstündedir" diyor.
    Ölçüldü: `noise = 0` iken bu bir BIÇAK SIRTIDIR ve son bit karar veriyor —
        p = 0.5 -> OTURDU(0)   p = 1.5 -> OTURDU(1)   p = 2.5 -> OTURDU(3)
    Çünkü `|p_inf - round(p_inf)|` tam 0.5 değil, 0.4999999...'dur.

    Kör noktayı KAPATAN şey `noise`un zorunlu olmasıdır: `noise >= 1e-12`
    verilen aynı merdivenlerde üçü de OTURMADI'ya dönüyor. Ve gerçek
    korpusta `u = 0` olan merdiven YOK (n = 301, min u = 1.357e-05), yani
    kör nokta yalnız gürültüsüz sentetik girdide açıktır.
    """
    h = np.array(_araliklar(201))
    v = 3.0 + 0.7 * h**p

    kor = convergence_ladder(v, h, noise=0.0)
    assert kor.verdict == "OTURDU"
    assert kor.settled_order == gurultusuz_tamsayi
    assert kor.p_inf == pytest.approx(p, abs=1e-7)

    for gurultu in (1e-12, 1e-9):
        L = convergence_ladder(v, h, noise=gurultu)
        assert L.verdict != "OTURDU"
        assert L.settled_order is None


def test_GERCEK_MERDIVENDE_BELIRSIZLIK_HIC_SIFIR_DEGIL():
    """DÜRÜST OLUMSUZ'un ikinci yarısı: kör nokta pratikte kapalı.

    426 merdivenlik korpusta `p_inf` üreten 301 merdivenin belirsizliği:
    min 1.357e-05, medyan 1.437e-02, max 3.118e+05 — ve `u == 0` olan
    merdiven sayısı SIFIR. Burada üç aileden birer örnekle yeniden
    doğrulanır.
    """
    for uret in (_merdiven_dispersiyon, _merdiven_etiket_hatasi):
        d, a, g = uret()
        L = convergence_ladder(d, a, noise=g)
        assert L.uncertainty > 0.0
        assert L.uncertainty >= abs(L.p_inf - L.orders[-1])


def test_KISA_KAYIT_BOZMASI_NO_OP_BIR_NEGATIF_ORNEK_CURUDU():
    """DÜRÜST OLUMSUZ: "kayıt uzunluğu ölçeklenmiyor" bir BOZMA DEĞİL.

    Spec §7.7 bir negatif örneğin çürüdüğünü yazıyor ("kayıt uzunluğu
    ölçeklenmiyor" bozması 2. mertebeden çıkmıştı). Burada SEBEBİ ölçüldü:
    üç terimli `omega` tahmincisi kayıt uzunluğuna DUYARSIZDIR, dolayısıyla
    bozma BİT BİT no-op'tur — merdiven de gürültü de aynen aynı çıkıyor.

    Bozma ancak SPEKTRAL okumada gerçek bir bozmadır (yukarıdaki
    `test_PENCERE_DISI_...`), çünkü orada frekans çözünürlüğü kayıt
    süresinden gelir. Yani "hangi bozma gerçek bozmadır" sorusunun cevabı
    TAHMİNCİYE bağlıdır ve bozmanın adına bakarak verilemez.
    """
    for m, S in ((8, 0.5), (16, 0.9)):
        d_uzun, a1, g1 = _merdiven_dispersiyon(m, S, 600, 201)
        # "Kayıt sabit" bozması: üç terimli okumada üretilen seri farklı
        # uzunlukta olsa bile okunan omega aynıdır.
        d_kisa, g2 = [], []
        for i in range(4):
            N = 200 * 2**i + 1
            v, e = _deger_ve_gurultu(_seri(N, m, S, 600), S, np.pi * m / (N - 1))
            d_kisa.append(v)
            g2.append(e)
        assert d_uzun == d_kisa
        assert g1 == g2


# ---------------------------------------------------------------------------
# 8. p_inf vs ham p_son — planın "ÖLÇ" dediği soru
# ---------------------------------------------------------------------------
def test_P_INF_HAM_P_SON_DEGIL_ve_AYRIM_KIRLENMEDE_ORTAYA_CIKIYOR():
    """SINIF: ÖLÇÜM. Planın "ÖLÇ" dediği mutantın cevabı burada.

    ÖLÇÜLDÜ: 426 merdivenlik korpusta `p_inf` üreten 301 merdivenin
    HİÇBİRİNDE `p_inf` yerine ham `p_son` kullanmak hükmü değiştirmiyor
    (0/301) — çünkü `u`nun birinci terimi tam olarak `|p_inf - p_son|`dur
    ve farkı soğurur. Yani mutant GERÇEK KORPUSTA HAYATTA KALIR.

    Ayrım YÜKSEK MERTEBE KİRLENMESİNDE ortaya çıkıyor ve orada karar
    TERSİNE dönüyor. `v = A + C h + B h^3`, `B` kaba basamakta kirlenme
    baş terimin `q` katı olacak şekilde seçilir; GERÇEK baş mertebe 1'dir.
    ÖLÇÜLDÜ (h = 1/200/2^i, A = 3.0, C = 0.7, gürültüsüz):

        q     p_inf    p_son    u       gerçek hüküm    p_son mutantı
        2     0.9799   1.6215   0.6415  YARGILANAMAZ    OTURDU(2)
        9     1.8679   2.3151   0.4472  OTURMADI        OTURDU(2)

    Yani mutant, baş terimi 1. mertebe olan bir merdivende "OTURDU(2)"
    diyor: `p_inf`i `p_son` ile değiştirmek `u`nun ilk terimini de
    SIFIRLADIĞI için belirsizlik kayboluyor ve ölçüt yanlış bir tamsayıya
    kilitleniyor. İki satır iki AYRI hüküm sınıfını (YARGILANAMAZ ve
    OTURMADI) mutanttan ayırıyor.

    DÜRÜST OLUMSUZ: `q = 12`'de gerçek ölçüt de OTURDU(2) diyor ve baş
    terim yine 1. mertebedir. Bu bir ölçüt kusuru DEĞİL, merdivenin
    penceresinin dışına çıkmasıdır: o kirlenmede `h^3` terimi merdivenin
    dört basamağında da baskındır ve merdiven asimptotik bölgeye HİÇ
    girmez. Çare daha iyi tahminci değil, daha ince merdivendir.
    """
    h = np.array(_araliklar(201))
    for q, bek_p_inf, bek_p_son, bek_u, bek_hukum in (
        (2.0, 0.9799, 1.6215, 0.6415, "YARGILANAMAZ"),
        (9.0, 1.8679, 2.3151, 0.4472, "OTURMADI"),
    ):
        B = q * 0.7 / h[0] ** 2
        v = 3.0 + 0.7 * h + B * h**3
        L = convergence_ladder(v, h, noise=0.0)

        assert L.p_inf == pytest.approx(bek_p_inf, abs=2e-3)
        assert L.orders[-1] == pytest.approx(bek_p_son, abs=2e-3)
        assert L.uncertainty == pytest.approx(bek_u, abs=2e-3)
        assert L.verdict == bek_hukum
        assert L.settled_order is None

        # MUTANT: `p_inf := p_son` (ve onunla birlikte `u`nun ilk terimi 0).
        p_son = L.orders[-1]
        u_mutant = 0.0
        assert abs(p_son - round(p_son)) + u_mutant < 0.5
        assert round(p_son) == 2  # baş terim 1. mertebe olduğu HÂLDE

    # Pencere dışı: kirlenme dört basamağın hepsinde baskınsa merdiven
    # gerçekten 2. mertebe GİBİ davranır ve ölçüt de öyle okur.
    v12 = 3.0 + 0.7 * h + (12.0 * 0.7 / h[0] ** 2) * h**3
    assert _hukum(convergence_ladder(v12, h, noise=0.0)) == "OTURDU(2)"


def test_YUKSEK_MERTEBE_KIRLENMESI_MERDIVENI_SINIRLIYOR():
    """SINIF: ÖLÇÜM. Merdiveni sınırlayan şey GÜRÜLTÜ DEĞİL, kirlenmedir.

    `v = 3 + 0.7 h^2 + D h^3` (gürültüsüz), h = 1/200/2^i:
        D = 5    p_son 2.014797  p_inf 2.000448  -> OTURDU(2)
        D = 50   p_son 2.130061  p_inf 2.030639  -> OTURDU(2)
        D = 500  p_son 2.594744  p_inf 2.444744  -> OTURMADI
    Richardson ötelemesi kirlenmeyi 33 kat (D = 5) ve 4.2 kat (D = 50)
    bastırıyor; yeterince büyük kirlenmede ölçüt DOĞRU biçimde pes ediyor.
    """
    h = np.array(_araliklar(201))
    beklenen = {5.0: "OTURDU(2)", 50.0: "OTURDU(2)", 500.0: "OTURMADI"}
    for D, hukum in beklenen.items():
        v = 3.0 + 0.7 * h**2 + D * h**3
        L = convergence_ladder(v, h, noise=0.0)
        assert _hukum(L) == hukum
        assert abs(L.p_inf - 2.0) < abs(L.orders[-1] - 2.0)


# ---------------------------------------------------------------------------
# 9. GİRDİ SÖZLEŞMESİ
# ---------------------------------------------------------------------------
def test_NOISE_ZORUNLU_ANAHTAR_KELIME_ARGUMANI():
    """`noise` unutulabilir bir varsayılan DEĞİLDİR.

    Spec §4.3'ün üçüncü düzeltmesi: `eps_i` ZORUNLU GİRDİDİR. Varsayılanı
    olsaydı çağıran onu vermeyi unutur ve ölçüt sessizce gürültüsüz kipe
    düşerdi — 426 merdivende %16.4 yanlış pozitif veren kip.
    """
    h = _araliklar(201)
    v = [3.0 + 0.7 * x**2 for x in h]
    with pytest.raises(TypeError):
        convergence_ladder(v, h)  # noqa: PGH005 — kasıtlı eksik argüman
    convergence_ladder(v, h, noise=0.0)  # anahtar kelimeyle çağrılabiliyor


@pytest.mark.parametrize(
    "kwargs, parca",
    [
        ({"values": [1.0, 2.0], "spacings": [1.0]}, "her degerin bir izgara"),
        ({"values": [], "spacings": []}, "bos"),
        ({"values": [1.0, np.nan, 3.0, 4.0]}, "sonlu olmali"),
        ({"spacings": [0.005, 0.0025, 0.00125, -1.0]}, "pozitif ve sonlu"),
        ({"spacings": [0.000625, 0.00125, 0.0025, 0.005]}, "KABADAN INCEYE"),
        ({"spacings": [0.005, 0.0025, 0.00125, 0.0009]}, "orani `ratio` ile"),
        ({"noise": -1e-3}, "negatif olmayan"),
        ({"noise": [1e-3, 1e-3]}, "skaler ya da 4 uzunlukta"),
        ({"ratio": 1.0}, "ratio > 1"),
        ({"ratio": 0.5}, "ratio > 1"),
        ({"kappa": 0.0}, "kappa pozitif"),
        ({"kappa": -0.5}, "kappa pozitif"),
    ],
)
def test_GECERSIZ_GIRDI_SESSIZCE_GECMIYOR(kwargs, parca):
    """Geçersiz girdi ASLA normal sonuç gibi dönmez (spec §3, İlke 3)."""
    h = _araliklar(201)
    cagri = {
        "values": [1.0, 1.5, 1.75, 1.875],
        "spacings": h,
        "noise": 1e-6,
        "ratio": 2.0,
        "kappa": 0.5,
    }
    cagri.update(kwargs)
    with pytest.raises(InvalidInputError, match=parca):
        convergence_ladder(
            cagri.pop("values"),
            cagri.pop("spacings"),
            **cagri,
        )


def test_ARALIK_ORANI_RATIO_ILE_TUTARLI_OLMAK_ZORUNDA():
    """`p = log(d_i/d_{i+1}) / log(ratio)` — oran uymuyorsa çıkan sayı mertebe DEĞİL.

    `spacings` yalnız süs değildir: `ratio` ile tutarlılığı KAPIDA sınanır.
    Ölçüldü — 426 merdivenlik korpusta `h_i/h_{i+1}` ile 2.0 arasındaki en
    büyük sapma TAM SIFIR (0.0e+00), çünkü ikili gösterimde ikiye bölme
    tamdır. `_RATIO_RTOL = 1e-9` bu yüzden bir tolerans değil güvenlik payıdır.
    """
    h = _araliklar(201)
    d, a, g = _merdiven_dispersiyon()
    assert a == h
    assert max(abs(a[i] / a[i + 1] - 2.0) for i in range(3)) == 0.0

    # `ratio = 3` verilirse aynı merdiven REDDEDİLİR (sessizce p/log3 basmaz).
    with pytest.raises(InvalidInputError, match="orani `ratio` ile"):
        convergence_ladder(d, a, noise=g, ratio=3.0)


def test_KAYIT_TAM_TASINIYOR_ve_PANELE_YAZILABILIYOR():
    """SINIF: TRANSKRİPSİYON DENETİMİ.

    Merdivenin kendisi (değerler + aralıklar + ham mertebeler) tam olarak
    taşınır; çağıran hükmü kendi gözüyle denetleyebilmelidir. `binning_error`
    `block_sizes`/`errors` eğrisini aynı gerekçeyle taşır.
    """
    d, a, g = _merdiven_dispersiyon()
    L = convergence_ladder(d, a, noise=g)

    assert L.values == tuple(d)
    assert L.spacings == tuple(a)
    assert L.ratio == 2.0
    assert len(L.orders) == len(d) - 2
    # `orders` gerçekten ham `p_i`ler: kayıttan yeniden hesaplanabiliyor.
    fark = np.diff(np.asarray(L.values))
    beklenen = np.log(fark[:-1] / fark[1:]) / np.log(L.ratio)
    assert L.orders == pytest.approx(tuple(beklenen), rel=1e-12)
    # `p_inf` de kayıttan türetilebilir olmalı (Richardson, r = 2).
    p_inf_dizi = (2.0 * beklenen[1:] - beklenen[:-1]) / 1.0
    assert L.p_inf == pytest.approx(float(np.mean(p_inf_dizi)), rel=1e-12)

    satirlar = L.as_lines()
    assert any("OTURDU(2)" in s for s in satirlar)
    assert any("r = 2" in s and "derinlik 4" in s for s in satirlar)

    with pytest.raises(AttributeError):
        L.p_inf = 3.0  # frozen


def test_SETTLED_ORDER_YALNIZ_OTURDU_ICIN_SAYI():
    """`None` SIFIR DEĞİLDİR — Modül 4'ün `correlation_length -> None` dersi."""
    h = np.array(_araliklar(201))
    oturdu = convergence_ladder(3.0 + 0.7 * h**2, h, noise=0.0)
    assert oturdu.settled_order == 2

    # OTURMADI ve YARGILANAMAZ: tamsayı YOK.
    kirli = 3.0 + 0.7 * h**2 + 500.0 * h**3
    oturmadi = convergence_ladder(kirli, h, noise=0.0)
    assert oturmadi.verdict == "OTURMADI"
    assert oturmadi.settled_order is None
    assert oturmadi.p_inf is not None  # hesap var, HÜKÜM yok

    yargilanamaz = convergence_ladder(3.0 + 0.7 * h**2, h, noise=1.0)
    assert yargilanamaz.verdict == "YARGILANAMAZ"
    assert yargilanamaz.settled_order is None


# ---------------------------------------------------------------------------
# ÖLÇÜLEN SÜRELER (bu dosya, `-p no:randomly`, 42 test toplam 2.44 s;
# `tests/em` bütünü 87 test / 6.30 s, tam süit 1246 test / 599.35 s).
# `setup` ÖLÇÜLDÜ ve DEVRALINMADI: bu dosyada fixture yok, 42 testin
# hiçbirinde `setup` 0.005 s'yi geçmiyor (pytest `--durations` hiçbirini
# listelemiyor), yani `setup + call = call`. Task 1'in aynı ölçümü
# `tests/em` için geçerliliğini koruyor.
#   0.94 s  test_KORPUS_KARNESI_...                      <- slow
#   0.28 s  test_SAF_GURULTUDE_..._[rastgele yuruyus]
#   0.26 s  test_SAF_GURULTUDE_..._[saf gurultu]
#   0.15 s  test_GURULTU_BELIRSIZLIGI_YAYILIM_ILE_UYUSUYOR
#   0.09 s  test_KAPPA_TOLERANS_DEGIL_...
#   kalan 37 test toplam 0.72 s
# `slow` eşiği deponun kendi kalibrasyonundan: `hf/test_rhf_sikismasi.py`
# 0.46 s AÇIKÇA "slow DEĞİL" diye yazılı, slow işaretli en ucuz gövde 0.69 s.
# Bu dosyanın tek 0.69 s üstü gövdesi korpus karnesidir ve o işaretli.
# ---------------------------------------------------------------------------
