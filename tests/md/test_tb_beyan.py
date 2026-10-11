"""TB ss-sigma kenarının ZORUNLU BEYANLARI — spec md-tb §3 (B1–B10) ve M10, G6.

SINIF: TRANSKRİPSİYON DENETİMİ. Bu dosyanın hiçbir testi yeni bir fizik
iddiası kurmaz; spec §3'ün "panelde VE dokstringde" dediği on beyanın ikisinde
de durduğunu, beyan metninin TEK kaynaktan geldiğini ve M10'un dokstring
beyanının kaynağa karşı doğru olduğunu ölçer. Beyanların fizik kanıtları başka
yerdedir: B1/M11 `test_tb_kuvvet.py`, M1–M5/C1–C4 `tests/tb/test_mermin.py`,
M2/M3 `tests/tb/test_radyal.py`, M6–M8 `tests/tb/test_distance_model.py`,
B8/G5 `test_tb_kuvvet.py` + `test_md_scenario.py`.

İKİ KATMANLI TEK-KAYNAK KİLİDİ (Modül 5/6/7'nin kalıbı, burada YENİDEN
ÖLÇÜLDÜ):

1. `test_TB_BEYANI_TEK_KAYNAKTAN_geliyor` — TB panelinin on altı beyan
   cümlesinin (`_TB_YAKLASIMLAR` 11, `_TB_IHMALLER` 4, `_TB_BEKLENEN` 1) her
   biri `src/` ağacında TAM BİR KEZ geçer ve `tests/` ağacında (bu dosya
   hariç) hiç geçmez.
2. `test_TB_BEYAN_PARCASININ_KOPYASI_da_YASAK` — tam cümle sayımı YETMEZ
   (Modül 5'te 55 karakterlik bir parça kopyası tam-cümle sayımından
   kaçmıştı).

===========================================================================
PARÇA EŞİĞİ BU KENARDA YENİDEN ÖLÇÜLDÜ — 16 DEĞİL 17 (24.09.2026)
===========================================================================

`src/` ağacındaki (`engine/md/factory.py` hariç, 3266 dize sabiti) her sabit
TB'nin on altı beyan cümlesinin içinde arandı (sk/g6_parca_esigi.py):

    eşik  4 ->  42 eşleşme   (" kesme ", "orbital", "onsite", "aralık" …)
    eşik  6 ->  14 eşleşme
    eşik  8 ->  10 eşleşme   (" k-noktası", " hartree" …)
    eşik 12 ->   7 eşleşme   ("YARGILANAMAZ" x4, "Born-Oppenheimer" x3)
    eşik 16 ->   3 eşleşme   ("Born-Oppenheimer": hf/scf.py, hf_obs.py x2)
    eşik 17 ->   0 eşleşme
    eşik 20 ->   0 eşleşme

BULGU DÜZELTME TURUNDA (24.09.2026) B1 ve B9 cümleleri değişince YENİDEN
ÖLÇÜLDÜ (sk/duz/g6_parca_esigi_yeniden.log): eşik 4'te 50 eşleşme (42
değil), 6 / 8 / 12 / 16 / 17 / 20'de aynı (14 / 10 / 7 / 3 / 0 / 0). Eşik
17 hâlâ ölçülmüş tabanın üstünde.

16'daki üç eşleşme bir kopya DEĞİL: "Born-Oppenheimer" TAM 16 karakterlik
standart bir TERİMDİR ve HF dalının yaklaşım listesinde de kendi adıyla
geçer; yanlış pozitif tabanı tam 16'dadır. Modül 5/6/7 16'yı seçmişti (orada
16'da sıfır eşleşme vardı); burada DEVRALINAMADI, 17 seçildi.

İLK TARAMADA 16 VE 20'DE İKİ EŞLEŞME DAHA VARDI: `scenario/schema.py`deki
`electron_temperature_kelvin` anahtarı (27 karakter) B7 cümlesinin içinde
geçiyordu. Eşik 28'e çekilmedi; bridge'in `miknatislanma serisi` kalıbıyla
BEYAN yeniden yazıldı: anahtar adı artık cümlede değil tanılarda
(`electron_temperature_kelvin=6.316e+02`) basılır ve B7'nin panel kilidi
onu orada arar.

===========================================================================
SÜRE (ölçüldü, 24.09.2026, bu makine, `--durations=0`)
===========================================================================

Dosya: 27 test, pytest 0.72–0.74 s (üç koşu; yorumlayıcı başlangıcı dahil
duvar saati 1.09–1.12 s); en yavaşı parça taraması 0.13–0.14 s, tek-kaynak
taraması 0.04 s, geri kalanı ≤ 0.01 s. Bulgu düzeltme turundan sonra (aynı
gün) 28 test, 0.76 s. Hiçbiri `slow` DEĞİLDİR (spec 4.1;
depo eşiği 2 s).

BAĞLAYICILIK ÖLÇÜLDÜ (24.09.2026): her kilit en az bir mutantla kırıldı —
src/qlab + reference + scenarios + tests + pyproject bir kopyaya alınır,
mutant uygulanır, testler kopyanın içinden `PYTHONPATH=<kopya>/src` ile
koşar (sk/g6_mutasyon.py + .log). Kayıt dosyanın sonunda.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import qlab
import qlab.engine.md.factory as fabrika
import qlab.engine.md.tb_kuvvet as tb_kuvvet_modulu
import qlab.engine.tb.mermin as mermin_modulu
import qlab.engine.tb.radyal as radyal_modulu
from qlab.engine.md.factory import provenance_for
from qlab.scenario.schema import MDScenario

#: Kilitler KOŞAN kodu ve KOŞAN test ağacını okur (G1'de ölçüldü: sabit depo
#: yolu PYTHONPATH kopyasındaki mutantları görmüyordu).
KAYNAK = Path(qlab.__file__).resolve().parent
TESTLER = Path(__file__).resolve().parents[1]
FABRIKA_DOSYASI = Path(fabrika.__file__).resolve()

#: Parça eşiği — SEÇİLMEDİ, ÖLÇÜLDÜ (dosya dokstringi: 16 -> 3, 17 -> 0).
_BEYAN_PARCASI_ESIGI = 17


def _tb_beyanlari() -> tuple[str, ...]:
    return fabrika._TB_YAKLASIMLAR + fabrika._TB_IHMALLER + (fabrika._TB_BEKLENEN,)


def _kaynak_metni(yol: Path) -> str:
    """Dosya metni, BİTİŞİK DİZE SABİTLERİ BİRLEŞTİRİLMİŞ hâlde
    (`test_grav_senaryo.py`nin ölçülmüş tuzağı: `"a "\\n"b"` kaynakta iki
    parça, çalışma zamanında tek dize)."""
    return re.sub(r'"\s*\n\s*"', "", yol.read_text(encoding="utf-8"))


def _yasa(genlik: float, us: float) -> dict:
    return {
        "genlik_hartree": genlik, "d0_bohr": 2.0, "us": us,
        "r_on_bohr": 3.0, "r_cut_bohr": 3.8,
    }


def _tb_senaryo(thermostat: dict | None = None) -> MDScenario:
    """Ortak pencere (spec md-tb başı): 16 atomlu a = 2 zincir, 631.6 K, L = 32."""
    return MDScenario(
        name="tb beyan",
        force_field={
            "kind": "tb_ss_sigma",
            "params": {
                "ss_sigma": _yasa(-0.1, 2.0),
                "itme": _yasa(0.05, 6.0),
                "n_electron": 16,
                "electron_temperature_kelvin": 631.6,
                "box_x_bohr": 32.0,
            },
        },
        initial={
            "positions_bohr": [[2.0 * i, 0.0, 0.0] for i in range(16)],
            "masses_amu": [1.00782503207],
        },
        thermostat=thermostat,
        dt=10.0,
        n_steps=1,
    )


def _panel_satirlari() -> dict[str, str]:
    return dict(
        provenance_for(
            _tb_senaryo(), energy_drift=1e-9, energy_oscillation=1e-9, omega_max_frame0=1e-9,
            omega_max_path_frames=1e-9,
        ).as_lines()
    )


def _dokstring(modul) -> str:
    return " ".join((modul.__doc__ or "").split())


# ---------------------------------------------------------------------------
# B1–B10: HER BEYAN PANELDE (DOĞRU SATIRDA) VE DOKSTRİNGDE
# ---------------------------------------------------------------------------
#
# Panel parçaları `(satır etiketi, parça)` çiftidir: parça yalnız panelde
# geçmekle kalmaz, KENDİ satırında geçer (tersine çevrilmiş bir atama --
# `approximations=_TB_IHMALLER` -- bütün cümleleri yine basardı). Dokstring
# parçaları `engine/md/tb_kuvvet.py`nin modül dokstringindendir: kuvvet
# alanının kullanıcısının okuduğu metin orasıdır ve on beyanın onu da orada,
# `(spec Bn)` etiketiyle durur. Parçalar BAĞIMSIZ yazılmıştır (bir modül
# sabitinden okunmaz); büyük/küçük harf korunur (`.lower()` Türkçe `I`/`İ`yi
# bozar -- `factory.py` yorumu).

YAK, IHM, BEK, YAKINSAMA = "Yaklaşımlar", "İhmal edilen", "Beklenen hata", "Yakınsama"

BEYANLAR: dict[str, tuple[tuple[tuple[str, str], ...], tuple[str, ...]]] = {
    "B1": (
        (
            (YAK, "potansiyel Omega (Mermin) serbest enerjisidir"),
            (YAK, "kuvvet F = -grad(Omega)"),
            (YAK, "E'nin gradyanı değil"),
            # Korunum yarısı YALNIZ NVE'nin beklenen hatasında (bulgu
            # düzeltme turu): termostatlı panelde korunum iddiası basılmaz.
            (BEK, "KE + E KORUNMAZ: fark kT*dS"),
            (BEK, "KE + Omega SALINIR"),
        ),
        (
            "DONEN ENERJI MERMIN SERBEST ENERJISIDIR (spec B1)",
            "E = E_band + E_rep'in gradyani DEGIL",
            "KE + E KORUNMAZ",
        ),
    ),
    "B2": (
        (
            (YAK, "kT > 0 zorunludur -- kT = 0 MD yoktur"),
            (YAK, "kimyasal potansiyeli TEK ve globaldir"),
            (YAK, "yerel olmayan etki 1/kT ile büyür"),
            (YAK, "aralık değişimi / kT küçükken"),
        ),
        (
            "kT_hartree > 0 ZORUNLUDUR (spec B2)",
            "Fermi seviyesindeki dejenere noktada gradyan yoktur",
            "tek global mu yerel olmayan bir ciftlenim uretir",
        ),
    ),
    "B3": (
        (
            (YAK, "radyal kesme pürüzsüzdür"),
            (YAK, "sert kesmenin enerji basamağı yoktur"),
            (YAK, "kuyruk C^1'dir, C^2 değil"),
        ),
        (
            "PURUZSUZ KESME (spec B3)",
            "dE(1e-3)/dE(1e-4) = 100.0389",
            "C^1'dir, C^2 DEGIL",
        ),
    ),
    "B4": (
        (
            (YAK, "radyal yasa seçildi, türetilmedi: bu kenarda doğa çıpası yok"),
            (YAK, "fizik iddiası değildir"),
        ),
        ("PARAMETRELER SECIMDIR (spec B4): bu kenarda DOGA CIPASI YOKTUR",),
    ),
    "B5": (
        (
            (YAK, "periyodik kutu yalnız x ekseninde (en-yakın-görüntü), y ve z açık"),
            (YAK, "max(r_cut) < L/2 kesin koşul"),
            (IHM, "Gamma'da tek hücre bir halkadır, kristal değil"),
            (IHM, "enine kararsızlık bu depoda ölçülmedi"),
        ),
        (
            "PERIYODIK KUTU (spec B5)",
            "2 max(r_cut) < L kesin kosuldur",
            "Kucuk kutu Gamma'da bir halka verir, kristal degil",
            "Enine hareket SINANMADI",
        ),
    ),
    "B6": (
        (
            (YAK, "Slater-Koster tablosunun yalnız ss-sigma satırı, açı çarpanı özdeş 1"),
            (YAK, "spin katlanması g_s = 2 sabit (manyetizma yok)"),
            (YAK, "yük öz-uyumu (SCC/U) yok"),
            (YAK, "tek atom türü, onsite yok"),
        ),
        (
            "MODEL SINIRI (spec 1, B6)",
            "Slater-Koster tablosunun YALNIZ ss-sigma satiridir",
            "aci carpani OZDES 1, bu bir SK modeli DEGILDIR",
            "sinif onsite ALMAZ",
        ),
    ),
    "B7": (
        (
            (YAK, "elektron sıcaklığı senaryoda Kelvin, kuvvet sınıfında hartree"),
            (YAK, "dönüşüm fabrikada yapılır"),
            (YAKINSAMA, "electron_temperature_kelvin=6.316e+02"),
            (YAKINSAMA, "kT_hartree=2.000e-03"),
        ),
        (
            "BIRIM SINIRI (spec B7)",
            "Sinif HARTREE alir",
            "YAKALANMAZ -- beyan",
        ),
    ),
    "B8": (
        (
            (BEK, "TB'de enerjinin sıfırı keyfidir"),
            (BEK, "|blok[0]|'a BÖLÜNMEZ"),
            (YAKINSAMA, "energy_drift_scale=max_kinetic"),
        ),
        (
            "ENERJININ SIFIRI KEYFIDIR (spec B8)",
            "|blok[0]|'a degil kaydedilen en buyuk kinetik enerjiye boler",
        ),
    ),
    "B9": (
        (
            (YAK, "sert çekirdek yok: üstel yasa d = 0'da sonludur"),
            (YAK, "duvar sonludur (Morse beyanının eşi)"),
            (YAK, "çakışık atom hata verir"),
            (YAK, "koşul 2|t0|*k*e^k > A*krep*e^krep"),
            (YAK, "krep <= k ve 2|t0|*k > A*krep bunun yeter koşuludur"),
            (YAK, "Sınıf bu parametreleri reddetmez"),
            (YAK, "kuyunun dibinde kuvvet gradyan değildir"),
        ),
        (
            "SERT CEKIRDEK YOK (spec B9)",
            "duvar da sonludur (Morse beyaninin esi)",
            "d = 0'da bir V-KUYUSU olabilir",
            "krep <= k ve 2|t0| k > A krep bunun YETER kosuludur",
            "Sinif bu parametreleri REDDETMEZ",
        ),
    ),
    "B10": (
        (
            (YAK, "dt kararlılık denetimi TB dalında yapılmaz"),
            (YAK, "dt'yi kullanıcı seçer, koşu reddedilmez"),
            # D2 (09.10.2026): koşuyu reddeden denetim yok, a posteriori hüküm var.
            (YAK, "seçilen dt'yi koşudan SONRA NVE'nin salınım hükmü yargılar"),
            # D2-i (10.10.2026): koşudan önce kare-0 omega0*dt de hükme girer.
            (YAK, "koşudan ÖNCE de kare-0 geometrisinin sayısal Hessian'ından ölçülen omega0*dt"),
            # D2-vi: koşudan sonra yolun izlenen iki karesininki de.
            (YAK, "koşudan SONRA da yolun her adımda izlenen iki karesininki"),
        ),
        ("dt'yi kullanici secer (spec B10)",),
    ),
}


def test_BEYAN_TABLOSU_ON_BEYANIN_ONUNU_de_TASIYOR():
    """B1–B10'un hepsi tabloda, her birinin en az bir panel ve bir dokstring
    parçası var — on birinci bir beyan sessizce eklenemez, biri sessizce
    düşemez."""
    assert list(BEYANLAR) == [f"B{n}" for n in range(1, 11)]
    for ad, (panel, dok) in BEYANLAR.items():
        assert panel and dok, ad
        assert {etiket for etiket, _ in panel} <= {YAK, IHM, BEK, YAKINSAMA}, ad


@pytest.mark.parametrize("ad", list(BEYANLAR))
def test_B_BEYANI_PANELDE_KENDI_SATIRINDA(ad):
    """Spec §3: beyan PANELDE (NVE TB koşusu) ve KENDİ satırında.

    KIRMA (ölçüldü, sk/g6_mutasyon.log): B3 cümlesi `_TB_YAKLASIMLAR`dan
    silinince B3 hücresi; B5'in enine cümlesi ihmallerden yaklaşımlara
    taşınınca B5 hücresi; `_TB_BEKLENEN` ortak NVE metnine geri bağlanınca
    B8 hücresi kırmızı.
    """
    satirlar = _panel_satirlari()
    for etiket, parca in BEYANLAR[ad][0]:
        assert parca in satirlar[etiket], f"{ad}: {etiket!r} satırında yok: {parca!r}"


@pytest.mark.parametrize("ad", list(BEYANLAR))
def test_B_BEYANI_KUVVET_ALANI_DOKSTRINGINDE(ad):
    """Spec §3: beyan DOKSTRİNGDE (`engine/md/tb_kuvvet.py`), `(spec Bn)` etiketiyle.

    B3 ve B8 G4'ün dokstringinde YOKTU (G6'da eklendi). KIRMA (ölçüldü):
    B8 paragrafı silinince B8 hücresi, B3 paragrafının ölçüm cümlesi
    değişince (100.0389 -> 100.04) B3 hücresi kırmızı.
    """
    dok = _dokstring(tb_kuvvet_modulu)
    for parca in BEYANLAR[ad][1]:
        assert parca in dok, f"{ad}: tb_kuvvet dokstringinde yok: {parca!r}"
    etiket = "(spec 1, B6)" if ad == "B6" else f"(spec {ad})"
    assert etiket in dok, f"{ad}: etiket yok"


def test_B1_P_x_YASASININ_PENCERESI_mermin_DOKSTRINGINDE():
    """B1'in -P(x) grad(gap) yasası GENEL YASA DEĞİLDİR: pencere `tb/mermin.py`de yazılı.

    Spec B1 yasayı "çiftte 2 elektron varken" geçerli sayar; 1 ya da 3
    elektronda P başka bir fonksiyondur (P_1, P_max 0.2397, x = 2.087).
    İlk sürümde `mermin.py` dokstringi yasayı penceresiz, koşulsuz
    yazıyordu (fizik incelemesi bulgusu). Bu bir METİN VARLIĞI kilididir,
    bir sayı çıpası değil: sayıların ölçümü dokstringin kendisinde ve
    `tests/tb/test_mermin.py` M4'te; tek elektronlu kolun testi aynı
    dosyada M4-P1'dir (11.10.2026, D8: n_e = 15 ve 17, P_ölçülen/P_1 = 1).
    KIRMA (ölçüldü, 24.09.2026, sk/duz/mut_md.log): dokstringden çift
    elektron sayısına bağlılık paragrafını silen mutant kırmızı.
    """
    dok = _dokstring(mermin_modulu)
    for parca in (
        "P YASASI CIFTTEKI ELEKTRON SAYISINA BAGLIDIR (genel yasa DEGILDIR)",
        "Ciftte 2 elektron (f_ust = 1 - f_alt, mu bosluk ortasinda): P(x) = x/(4 cosh^2(x/4))",
        "Ciftte 1 ya da 3 elektron: P_1(x) = 2 x df_1/dx",
        "P_max = 0.23967, x = 2.0873'te",
        "Tek elektronlu kolun testi M4-P1'dir (`tests/tb/test_mermin.py`)",
    ):
        assert parca in dok, f"mermin dokstringinde yok: {parca!r}"


def test_TB_BEYAN_SATIRLARI_KARISMIYOR():
    """Üç sabit AYNI panelde ama AYNI satırda değil; B1'in korunum yarısı ve B8 yalnız NVE'de.

    `_TB_YAKLASIMLAR` yalnız "Yaklaşımlar"da, `_TB_IHMALLER` yalnız "İhmal
    edilen"de, `_TB_BEKLENEN` NVE'de "Beklenen hata"nın TAMAMI. Termostatlı
    koşuda sürüklenme hükme girmez (G5) ve B8 metni basılmaz -- NVT'nin
    beklenen hatası kendi metnidir. B1'in KORUNUM yarısı ("KE + Omega
    korunur, KE + E korunmaz") yalnız NVE'nin beklenen hatasındadır:
    yaklaşım listesi termostattan bağımsız basılır ve korunum cümlesi
    orada dururken NVT paneli kendisiyle çelişiyordu ("Yaklaşımlar"
    korunum, "Beklenen hata" "NVT: toplam enerji KORUNMAZ" diyordu) --
    eski `nvt[YAK] == satirlar[YAK]` eşitliği bu çelişkiyi YEŞİL olarak
    kilitliyordu.
    KIRMA (ölçüldü, 24.09.2026, sk/duz/mut_md.log): eski B1 cümlesini
    ("... integratör KE + Omega'yı korur, KE + E korunmaz (fark kT*dS)")
    geri koyan mutant son satırda kırmızı; eski eşitlik assert'i aynı
    mutantta YEŞİL kalıyordu (ölçüldü).
    """
    satirlar = _panel_satirlari()
    for cumle in fabrika._TB_YAKLASIMLAR:
        assert cumle in satirlar[YAK] and cumle not in satirlar[IHM]
    for cumle in fabrika._TB_IHMALLER:
        assert cumle in satirlar[IHM] and cumle not in satirlar[YAK]
    assert satirlar[BEK] == fabrika._TB_BEKLENEN
    nvt = dict(
        provenance_for(
            _tb_senaryo({"kind": "langevin", "params": {"temperature": 300.0, "gamma": 1e-3}}),
            energy_drift=5.0,
        ).as_lines()
    )
    assert fabrika._TB_BEKLENEN not in nvt[BEK]
    assert nvt[IHM] == satirlar[IHM]
    for cumle in fabrika._TB_YAKLASIMLAR:
        assert cumle in nvt[YAK]
    assert "korur" not in nvt[YAK] and "KE + Omega" not in nvt[YAK], nvt[YAK]


# ---------------------------------------------------------------------------
# TEK KAYNAK — İKİ KATMANLI KİLİT
# ---------------------------------------------------------------------------


def test_TB_BEYANI_TEK_KAYNAKTAN_geliyor():
    """TB beyan metninin İKİ KOPYASI OLAMAZ.

    `src/` ağacında her cümle tam BİR kez (kaynakta kopya yok -- `run.py` ya
    da bir `viz` modülü beyanı kendisi yazamaz) ve `tests/` ağacında, BU
    DOSYA hariç, sıfır kez. Tek kaynağın dosyası da kilitli: on altısı da
    `engine/md/factory.py`de.

    KIRMA (ölçüldü): B9 cümlesinin tamamı `run.py`ye bir dize sabiti olarak
    kopyalanınca kırmızı (src sayımı 2).
    """
    beyanlar = _tb_beyanlari()
    assert len(beyanlar) == 16, "beyan sayısı değişti; testi bilerek güncelleyin"
    assert len(set(beyanlar)) == 16
    bu_dosya = Path(__file__).resolve()
    src_metinleri = [_kaynak_metni(yol) for yol in KAYNAK.rglob("*.py")]
    test_metinleri = [
        _kaynak_metni(yol) for yol in TESTLER.rglob("*.py") if yol.resolve() != bu_dosya
    ]
    fabrika_metni = _kaynak_metni(FABRIKA_DOSYASI)
    for cumle in beyanlar:
        src_sayim = sum(metin.count(cumle) for metin in src_metinleri)
        assert src_sayim == 1, f"src/ ağacında {src_sayim} kopya: {cumle[:60]!r}"
        test_sayim = sum(metin.count(cumle) for metin in test_metinleri)
        assert test_sayim == 0, f"BAŞKA bir test dosyasında kopya var: {cumle[:60]!r}"
        assert cumle in fabrika_metni


def test_TB_BEYAN_PARCASININ_KOPYASI_da_YASAK():
    """TAM CÜMLE SAYIMININ ÖLÇÜLMÜŞ DELİĞİ — ve bu kenarda YENİDEN ÖLÇÜLDÜ.

    `src/` ağacında (`engine/md/factory.py` hariç) hiçbir dize sabiti --
    dokstringler dahil -- bir TB beyan cümlesinin İÇİNDE geçemez. Korunan
    şey "aynı cümle iki kez yazılmasın" değil, PANELİN İKİ FARKLI ŞEY İDDİA
    EDEMEMESİdir.

    EŞİK DEVRALINMADI, YENİDEN ÖLÇÜLDÜ (dosya dokstringi): 16'da üç
    "Born-Oppenheimer" terimi, 17'de sıfır. KIRMA (ölçüldü): `run.py`ye B8
    cümlesinin 55 karakterlik bir parçası ve B2 cümlesinin 20 karakterlik
    bir parçası enjekte edilince ikisi de kırmızı; bu dosyada eşik 16'ya
    indirilince kırmızı (üç terim) -- eşik gerçekten ölçülmüş tabanın
    üstünde duruyor.
    """
    beyanlar = _tb_beyanlari()
    kopyalar = []
    for yol in KAYNAK.rglob("*.py"):
        if yol.resolve() == FABRIKA_DOSYASI:
            continue
        agac = ast.parse(yol.read_text(encoding="utf-8"))
        for dugum in ast.walk(agac):
            if not isinstance(dugum, ast.Constant) or not isinstance(dugum.value, str):
                continue
            if len(dugum.value) < _BEYAN_PARCASI_ESIGI:
                continue
            for cumle in beyanlar:
                if dugum.value in cumle:
                    kopyalar.append((yol.name, dugum.lineno, dugum.value[:60]))
    assert kopyalar == [], f"TB beyanının PARÇASI kopyalanmış: {kopyalar}"


# ---------------------------------------------------------------------------
# M10 DOKSTRİNG KİLİDİ — beyan KAYNAĞA karşı
# ---------------------------------------------------------------------------

#: Kenarın üç dosyası ve M10 kilitlerinin bulunduğu test dosyası. Dokstring
#: kilit dosyasını ADIYLA anar; anılan dosyanın var olduğu ve gerçekten
#: `test_M10_*` testleri taşıdığı burada sınanır (bayat bir gönderme -- test
#: silinip dokstring kalırsa -- kırmızı olur).
M10_DOSYALARI = [
    pytest.param(tb_kuvvet_modulu, "tests/md/test_tb_kuvvet.py", 4, id="tb_kuvvet"),
    pytest.param(mermin_modulu, "tests/tb/test_mermin.py", 4, id="mermin"),
    pytest.param(radyal_modulu, "tests/tb/test_mermin.py", 1, id="radyal"),
]

#: Beyanın yasakladığı paketler. `tb.factory` downfold üzerinden 8, `md`
#: fabrikası forcefields üzerinden 6 hf modülü yükler (spec 2.2).
_YASAK_ONEKLER = ("qlab.engine.hf", "qlab.engine.tb.downfold", "qlab.engine.tb.factory")


@pytest.mark.parametrize(("modul", "kilit", "en_az"), M10_DOSYALARI)
def test_M10_DOKSTRING_BEYANI_KAYNAGA_KARSI_DOGRU(modul, kilit, en_az):
    """Spec M10 + G6: bağımsızlık beyanı dokstringde VE kaynakta doğru.

    Üç parça: (1) dokstring `BAGIMSIZLIK (spec M10)` paragrafını ve üç
    yasak paketin adını taşır; (2) beyan KAYNAĞA karşı doğrudur -- dosyanın
    `ast` import kümesinde o paketlerden hiçbiri yok; (3) dokstringin andığı
    kilit dosyası var ve en az `en_az` `test_M10_*` testi taşıyor.
    `tb_kuvvet.py` ayrıca yasanın ikinci kopyasını yazmadığını beyan eder
    (`test_tb_kuvvet.py`nin `ast` kilidi).

    Parçalar dokstringin TAMAMINDA değil, `BAGIMSIZLIK (spec M10)`
    PARAGRAFININ İÇİNDE aranır. Ölçülmüş sebep: ilk sürüm tüm dokstringe
    bakıyordu ve paragrafın kilit dosyasını `tests/md/test_yok.py` diye
    değiştiren mutant YEŞİL kaldı -- `tb_kuvvet.py`nin dokstringi aynı dosyayı
    ilk paragrafında (ikinci kopya yasağı için) zaten anıyordu.

    KIRMA (ölçüldü, sk/g6_mutasyon.log): tb_kuvvet'e `import
    qlab.engine.hf.scf` eklenince (2) kırmızı; paragraf başlığı silinince
    (1); paragraf başka bir kilit dosyasını anınca (3); kilit dosyasındaki
    bir `test_M10_*` yeniden adlandırılınca (3).
    """
    paragraflar = [
        " ".join(p.split()) for p in (modul.__doc__ or "").split("\n\n")
        if "BAGIMSIZLIK (spec M10)" in p
    ]
    assert len(paragraflar) == 1, f"BAGIMSIZLIK paragrafı {len(paragraflar)} kez"
    (dok,) = paragraflar
    assert "hf paketini, `tb.downfold`u ve `tb.factory`yi import etmez" in dok
    agac = ast.parse(Path(modul.__file__).read_text(encoding="utf-8"))
    importlar = set()
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.Import):
            importlar.update(a.name for a in dugum.names)
        elif isinstance(dugum, ast.ImportFrom):
            assert dugum.level == 0, "göreli import tarayıcıdan kaçar"
            importlar.add(str(dugum.module))
    assert "qlab.errors" in importlar  # tarayıcı boş değil (kontrol)
    yasakli = [m for m in importlar if any(m == y or m.startswith(y + ".") for y in _YASAK_ONEKLER)]
    assert yasakli == [], yasakli
    assert f"`{kilit}`" in dok, f"dokstring kilit dosyasını anmıyor: {kilit}"
    kilit_yolu = TESTLER.parent / kilit
    assert kilit_yolu.is_file(), kilit
    m10 = [
        d.name for d in ast.walk(ast.parse(kilit_yolu.read_text(encoding="utf-8")))
        if isinstance(d, ast.FunctionDef) and d.name.startswith("test_M10_")
    ]
    assert len(m10) >= en_az, m10
    if modul is tb_kuvvet_modulu:
        assert "Bu dosya yasanin ikinci bir kopyasini YAZAMAZ" in _dokstring(modul)
        assert "importu tek basina SIFIR hf modulu yukler" in dok
        assert "`hf.scf.solve` SIFIR kez cagrilir" in dok


# ---------------------------------------------------------------------------
# KIRMA KAYDI (24.09.2026, bu makine; sk/g6_mutasyon.py + g6_mutasyon.log,
# g6_mutasyon_2.log). Düzenek G4/G5 ile aynı: src/qlab + reference +
# scenarios + tests + pyproject kopyası, testler kopyanın içinden. Eşdeğer
# kontrol mutantı (eşik 1e-2 -> 1e-2 + 0.0) 110/110 yeşil. Bu dosya için:
#   B3 cümlesi silindi          -> panel[B3] + tek kaynak (sayı 16 -> 15)
#   B2 bedel cümlesi silindi    -> panel[B2] + tek kaynak
#   B5 enine cümlesi yaklaşımlara taşındı -> panel[B5] (satır yanlış)
#   TB beklenen hata ortak NVE metnine -> panel[B8] + satır testi
#     (+ test_md_scenario TB paneli)
#   yaklaşım/ihmal ters atandı  -> panel[B1..B7, B9, B10] + satır testi
#   B7 eski cümle (anahtar adıyla) -> panel[B7] + PARÇA KOPYA (schema.py
#     `electron_temperature_kelvin` x2) + test_md_scenario TB paneli
#   dokstring B8 paragrafı silindi / B3 sayısı 100.04 / B6 etiketi
#     (spec B6) -> dokstring[B8] / [B3] / [B6]
#   run.py'ye B9 cümlesinin tamamı -> tek kaynak (src 2 kopya) + parça kopya
#   test_md_scenario.py'ye B9 cümlesi (yorum) -> tek kaynak (test kopyası)
#   run.py'ye 55 ve 20 karakterlik parça -> parça kopya
#   bu dosyada eşik 16 -> parça kopya (3 x "Born-Oppenheimer")
#   tb_kuvvet'e `import qlab.engine.hf.scf` -> M10 dokstring[tb_kuvvet]
#     (+ test_tb_kuvvet ast ve alt süreç kilitleri)
#   BAGIMSIZLIK başlığı silindi (tb_kuvvet / radyal) -> M10[tb_kuvvet] / [radyal]
#   paragrafın kilit dosyası `tests/md/test_yok.py` -> M10[tb_kuvvet]; İLK
#     SÜRÜMDE HAYATTA KALDI (dokstringin tamamı aranıyordu ve aynı dosya ilk
#     paragrafta da anılıyordu) -- arama paragrafa daraltıldı, artık kırmızı
#   test_tb_kuvvet'te bir `test_M10_*` yeniden adlandırıldı -> M10[tb_kuvvet]
#
# BULGU DÜZELTME TURU (24.09.2026; sk/duz/mh.py + mut_md*.log). Eşdeğer
# kontrol (Kelvin eşiği 1.0 -> 1.0 + 0.0) 119/119 yeşil.
#   eski B1 cümlesi (korunum yarısı yaklaşımlarda) -> panel[B1] + satır
#     testi (NVT'de "korur") + test_md_scenario TB paneli; aynı mutant
#     ESKİ `nvt[YAK] == satirlar[YAK]` eşitliğiyle YEŞİL (1 passed)
#   B9 "duvar sonludur" parçası silindi -> panel[B9]
#   B9 V-kuyusu koşul cümlesi silindi -> panel[B9]
#   tb_kuvvet dokstringinden "krep <= k ... YETER kosuludur" -> dokstring[B9]
#   mermin dokstringinden P'nin elektron sayısına bağlılığı -> B1 P(x) penceresi
# ---------------------------------------------------------------------------
