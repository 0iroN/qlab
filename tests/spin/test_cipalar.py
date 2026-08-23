"""MODÜL 4'ÜN İKİ SAYISAL ÇIPASI — ve ikisinin de ÇÖZÜNÜRLÜĞÜ.

SINIF: İKİSİ DE DAİRESEL DEĞİL.

1. **Onsager'in `kT_c/J = 2/ln(1+√2) = 2.269185314`'ü.** Modelin MATEMATİKSEL
   bir gerçeğidir ve simülasyona HİÇBİR YERDE girdi değildir: T ızgarası
   (2.24, 2.26, 2.28) yuvarlak sayılardan kurulur, örnekleyici yalnız `T`, `J`
   ve `h` görür, kesişimi bulan `_kesisim` yalnız (ızgara, ΔU) alır. Onsager
   sayısı bu dosyada YALNIZCA karşılaştırma değeri olarak geçer.

2. **Kritik üs `β/ν`.** Evrenseldir: `J`den, kafes sabitinden, birim
   seçiminden BAĞIMSIZDIR, dolayısıyla girdilerden geri okunamaz. Sonlu-boyut
   ölçeklemesiyle `⟨|m|⟩ ~ L^(-β/ν)` uyumundan çıkar.

===========================================================================
ÖLÇÜLEN SAYILAR — HEPSİ 23.08.2026'DA, BU DEPODA, AŞAĞIDAKİ PENCERELERDE
===========================================================================

**A. Binder kesişimi (Onsager çıpası).**
Pencere: L = 8, 16, 24 · T ızgarası (2.24, 2.26, 2.28), **adım 0.02** ·
Wolff · `n_burn = 300`, `n_sweep = 1500` · K = 24 bağımsız tohumlu zincir
(20260824 + 1000·k, k = 0…23) · soğuk (tekbiçim) başlangıç ·
tahminci: ΔU'nun işaret değiştirdiği iki komşu ızgara noktası arasında
**DOĞRUSAL ARA DEĞERLEME** · hata çubuğu: zincir etiketi üzerinden 3000
çekilişlik bootstrap (tohum 20260826).

    çift        kesişim     σ         Onsager sapması   3σ
    L=8 × L=16  2.265050    0.004118  −0.004135 (−1.00σ)  0.01235
    L=8 × L=24  2.265253    0.003020  −0.003932 (−1.30σ)  0.00906
    L=16 × L=24 2.265570    0.005206  −0.003616 (−0.69σ)  0.01562

    üç ikili kesişimin YAYILIMI: 0.000519 — en dar 3σ'nın (0.00906)
    **on yedide biri**, yani üç kesişim bu istatistikle AYIRT EDİLEMİYOR.

**B. Kritik üs.**
Pencere: L = 8, 12, 16, 24, 32 · T = T_c · Wolff · `n_burn = 300`,
`n_sweep = 1500` · aynı K = 24 zincir · aynı bootstrap.

    ⟨|m|⟩: 0.775523 / 0.740615 / 0.715627 / 0.678801 / 0.654571
    β/ν = 0.122941 ± 0.003225   (3σ = 0.009674)

===========================================================================
DÜRÜST HÜKÜMLER — "β = 1/8 ÇIKTI" YAZMAK YASAK
===========================================================================

Doğru cümle: **β/ν = 0.122941 ± 0.003225 ölçüldü; tam değer 0.125 (sapma
−0.002059 = −0.64σ, yani ölçüm belirsizliğinin içinde); ORTALAMA ALAN (β/ν =
1.0) 272.0 KAT marjla dışlandı.** Kalan sapma bir sonlu-boyut düzeltmesidir
ve sıfırlanmaz — bu, `L` alt kümeleri ayrı ayrı uydurulunca GÖRÜLÜR:
L = (8,12,16) için 0.121346, L = (16,24,32) için 0.129350, L = (24,32,48)
için 0.123540 (n_sweep = 3000 penceresinde, L = 48 dahil). Yani log-log
doğrusu **tam doğru değildir**; altı noktalık uyumun 0.125'e bu kadar yakın
düşmesi, düzeltmelerin bu L penceresinde birbirini götürmesindendir, "düzeltme
yok" demek değildir.

`β/ν` ile `β` SAYISAL OLARAK aynıdır çünkü 2B Ising'de `ν = 1`. **BU DOSYA
`ν`yü ÖLÇMEZ**; `ν = 1` bağı literatürdendir ve burada beyan edilir, çıpanın
parçası değildir.

Binder kesişimi için doğru cümle: **kesişim 2.265050…2.265570 aralığında
ölçüldü; Onsager'in 2.269185'i her üç kesişimin de 3σ'sı İÇİNDEDİR (en
kötüsü 1.30σ), ama üç kesişimin de sapması AYNI YÖNDE (aşağı).** Tek bir
tohum bloğu bunu ayıramaz; üç bağımsız tohum bloğu (20260824 / 20260825 /
19990101, aynı pencere) havuzlandığında:

    blok ortalamaları  2.265291 / 2.266747 / 2.267474
    havuz              2.266504 ± 0.000642
    Onsager sapması    −0.002681 = **−4.18σ**

yani sapma GERÇEK ve SİSTEMATİKTİR (sonlu-boyut kayması), tek blokta ise
gürültünün içinde kalır.

===========================================================================
ÜÇ TUZAK — ÖNCE ÖLÇÜLDÜ
===========================================================================

**TUZAK 1 — ÇÖZÜNÜRLÜK. Planın sayısı bir KESİŞİM DEĞİL, bir IZGARA
NOKTASIDIR.** Plan (satır 44-46) T ızgarası 2.10–2.46, adım 0.04 ile
Binder kesişimini **2.260** bulmuş ve sapmayı **%0.4** yazmış. %0.4 mutlak
olarak 0.0092'dir, yani **ızgara adımının dörtte biri**. Yeniden ölçüldü
(L = 8/16/24, aynı ızgara, K = 24 zincir, n_burn = 500, n_sweep = 3000):
o ızgarada `|ΔU|`yu en küçük yapan ızgara NOKTASI gerçekten 2.26'dır (üç
çiftte de), yani planın sayısı yeniden üretilebilir — **ama o bir kesişim
değil, bir ızgara noktasıdır ve çözünürlüğü 0.04'tür.** Aynı veriden doğrusal
ara değerlemeyle okunan kesişimler 2.26317 / 2.26363 / 2.26406'dır.
(KAPSAM: bu yeniden üretim **Wolff** ile yapıldı, plan Metropolis kullanmış.
Yani "2.26 çıkıyor" hükmü ızgara hakkındadır, algoritma hakkında değil.)

Bu dosyada kesişim **ara değerlemeyle** bulunur ve ara değerlemenin ızgara
sistematiği **gürültüsüz olarak** ölçülür (tam sayım, aşağıda): adım 0.02'de
sistematik **−2.156e-04**, yani bu dosyanın istatistiksel σ'sının (3.0e-03)
**on dörtte biri**. Yani sınırlayıcı olan ızgara değil, istatistiktir —
ve bu bir varsayım değil, ölçüm.

**TUZAK 2 — TEK KESİŞİM DİYE BİR ŞEY YOK.** Üç L → üç ikili kesişim, ve
sonlu-boyut kayması yüzünden çakışmaları beklenmez. ÖLÇÜLDÜ: L = 8/16/24'te
yayılım 0.000519, en dar 3σ 0.00906 — yani **bu istatistikle üçü ayırt
edilemiyor**. Ayrıca üçünün SIRASI tohuma göre değişiyor (üç blokta
True / False / True), yani "kesişim L ile şu yönde kayıyor" demek TEK bir
akıştan okunamaz — Task 3'ün "tek akıştan okunan oran, sınırın kendisi kadar
gürültülüdür" dersinin aynısı.

Kaymanın GERÇEK olduğu ise ayrı bir pencerede, gürültüsüz gösterilir:
L = 3 × L = 4 kafeslerin **TAM SAYIMINDA** (512 ve 65536 durum, hata çubuğu
YOK) kesişim **2.1692678**'dir — Onsager'den **−0.0999 (−%4.40)**. Yani
sonlu-boyut kayması gerçektir, L küçüldükçe DEVLEŞİR, ve L ≥ 8'de bu
istatistiğin çözünürlüğünün altına iner.

**TUZAK 3 — KENDİ DÜRÜSTLÜK MAKİNESİNİ KENDİ MANŞETİNE UYGULA.** Task 4'ün
`binning_error`ı plato tespitlidir ve `plateau_reached is False` iken dönen
sayının hatanın kendisi değil ALT SINIRI olduğu kilitlenmiştir. Çıpaları
besleyen HER zincirde bu bayrak okundu:

    Binder taşıyıcısı (ns = 1500): L=8 71/72 · L=16 68/72 · L=24 63/72
    FSS taşıyıcısı   (ns = 1500): 112/120 (L=32'de 18/24)
    FSS KISA sürüm   (ns =  160): **15/72** — PLATO YOK

Yani taşıyıcı koşumlarda plato %88-100 oranında oluşuyor; **kısa sürümde
OLUŞMUYOR ve orada zincir-içi binning hatası bir ALT SINIRDIR.** Bu dosyanın
kullandığı hata çubuğu zincir-içi DEĞİL, K = 24 bağımsız zincirin
ortalamalarının bootstrap'ıdır — dolayısıyla plato bayrağına bağlı değildir;
bayrak, her zincirin kendi otokorelasyonunu çözecek kadar uzun olduğunun
BAĞIMSIZ teyididir. Kısa sürümde bu teyit YOKTUR ve bu, sınırın gevşetilmesi
değil, sınırın nereden geldiğinin beyanıdır.

===========================================================================
ÖLÇÜLEN KÖR NOKTA — PLANIN BİR MUTANT GEREKÇESİ YANLIŞ
===========================================================================

Plan (satır 799) `⟨m⁴⟩/⟨m²⟩²` ters çevrilirse testin **kırmızı** olmasını
bekliyor — ve kırmızı oluyor, ama planın sandığı sebepten DEĞİL. Kesişimin
kendisi bu mutasyona kördür ve sebebi cebirseldir: kesişim
`U_L1(T) = U_L2(T)` noktasıdır, bu da `⟨m⁴⟩/⟨m²⟩²` oranlarının eşitliğine
denktir; oranın MONOTON herhangi bir yeniden parametrelemesi eşitlik noktasını
DEĞİŞTİRMEZ. Tam sayımla ölçüldü (hata çubuğu yok, tartışma yok):

    doğru                       kök 2.1692674
    ters oran 1 − ⟨m²⟩²/(3⟨m⁴⟩) kök 2.1692675
    3 çarpanı düşmüş            kök 2.1692674

Kesişim çıpasının **FİZİK İDDİASI** bu iki mutanta KÖRDÜR — ve bu, mutant
gerçekten koşularak doğrulandı: `abs(kok - TC) < 3*sigma` assert'i GEÇİYOR.
Düşen şey bir satır aşağıdaki **transkripsiyon kilidi**dir (`abs = 1e-5`).

Kilidin görme sebebi ARA DEĞERLEMEDİR, kesişimin yer değiştirmesi değil:
kaba ızgarada (adım 0.02) doğrusal ara değerleme farklı EĞRİLİKTEKİ bir
fonksiyonu kökler. Ölçüldü — ters mutantta bu dosyanın üç kesişimi
2.265233 / 2.265485 / 2.265881'e, yani **+1.83e-4 / +2.32e-4 / +3.11e-4**
kayıyor; istatistiksel σ (3.0-5.2e-3) ile karşılaştırıldığında KIRKTA BİR.
Tam sayımda ızgara 5e-4'e inince kayma 3.3e-8'e düşüyor, yani tamamen
ızgara artığıdır.

Mutantları **FİZİKLE** yakalayan şey LİMİT davranışıdır
(`test_TAM_SAYIM_BINDER_EGRISI_...`): ters oranda U sıcaklıkla ARTAR
(0.666667 → 0.689525 → 0.725398), 3 düşünce düzenli faz limiti 2/3 yerine
0 olur.

Yani planın beklentisi ("kesişim testi kırmızı olur") SONUÇ olarak doğru,
GEREKÇE olarak yanlış. Bu ayrım bu depoda önemlidir: **hangi assert'in
düştüğü, mutantın neyi bozduğunu söyler.**

===========================================================================
ÇIPALARIN AYIRMA GÜCÜ — T_c KAYMA TARAMASI (planın "ÖLÇ" dediği mutant)
===========================================================================

`TC` sabiti kaydırılırsa hangi kaymadan itibaren yakalanır?

**Kesişim çıpası** için `TC` YALNIZ karşılaştırma değeridir (T ızgarası
sabit yuvarlak sayılar), dolayısıyla tarama aynı koşumdan okunur:

    kayma    +0.002  +0.005  +0.010  +0.020  +0.050  +0.100
    yakalar?   hayır   hayır    EVET    EVET    EVET    EVET
    kayma    −0.002  −0.005  −0.010  −0.020  −0.050  −0.100
    yakalar?   hayır   hayır   hayır    EVET    EVET    EVET

    Ayrım gücü ASİMETRİKTİR (+0.010 / −0.020) çünkü ölçülen kesişim
    Onsager'in 0.0039 ALTINDADIR: negatif kayma T_c'yi ölçülen değere
    YAKLAŞTIRIR. **0.05'lik kayma 9.6 – 16.6σ ile yakalanır.**

**FSS çıpası** için `TC` gerçek bir GİRDİDİR (koşum T_c'de yapılır), yani
kayma bütün koşumu değiştirir. Aynı taşıyıcı pencerede ölçüldü:

    kayma     0.000   ±0.002        ±0.005        ±0.010        ±0.020        ±0.050
    β/ν     0.122941 0.128555/0.116326 0.135349/0.116517 0.144428/0.104954 0.171232/0.086804 0.263067/0.046682
    σ cinsi   −0.65   +1.09/−3.04   +3.35/−3.23   +7.35/−6.76  +12.45/−16.20 +29.56/−47.06
    yakalar?    —     hayır/EVET      EVET          EVET          EVET          EVET

    **FSS çıpasının ayırma gücü ≈ 0.005 (3σ ölçütüyle); 0.05'lik kayma
    29.6 – 47.1σ ile yakalanır.** Kısa (hızlı şerit) sürümün ayırma gücü
    10 kat kaba: ±0.01 ve ±0.02 YAKALANMIYOR, ±0.05 yakalanıyor (+5.23σ /
    −4.45σ).

    (Bu tarama AYRI bir bootstrap akışıyla koşuldu; oradaki kaymasız taban
    0.122941 ± 0.003178, bu dosyanın fixture'ında 0.003225. Aynı σ'nın iki
    tahmini — %1.5 fark, `1/√(2(N−1))` mertebesinde ve hükmü değiştirmiyor.)

**MUTASYON SINAVI BU AYRIMI ÖLÇEMEZ — VE BU YAZILMALI.** `TC` kaydırıldığında
bu dosya ±0.005'te bile kırmızıya döner, ama bunun sebebi FİZİK ASSERT'İ
DEĞİL, `TC` içeren TRANSKRİPSİYON KİLİTLERİDİR (ör. `(kök − T_c)/σ` oranı,
tam sayım kesişiminin `T_c`den uzaklığı). Kesişim çıpasının GERÇEK ayırma
gücü yukarıdaki tablodadır ve tam olarak hesaplanabilir, çünkü ölçülen
kesişim `TC`den TAMAMEN bağımsızdır: `|kesişim − (T_c + kayma)| > 3σ` eşiti
kayma > +0.010 ya da kayma < −0.020'de sağlanır. Mutasyon sınavının
"kırmızı" demesi bu sayıyı ispatlamaz; onu ölçen şey yukarıdaki taramadır.

===========================================================================
MODÜLLER ARASI BAĞ (spec §3.3)
===========================================================================

Modül 3H **aynı depoda**, UHF indirgemesinde, kademeli alanın çatallanma
üssünü **0.496190 / 0.497600 / 0.497811** (N = 6/10/14) ölçtü — çünkü UHF bir
ORTALAMA ALAN yöntemidir ve ortalama alanın çatallanma üssü tam 1/2'dir.
Bu dosya 2B Ising'in üssünü **0.122941 ± 0.003225** ölçüyor. **Aynı depo,
aynı tür kendiliğinden simetri kırılması, 4.04 KAT farklı üs** — ve fark
115.8σ. İki ölçüm birbirini bağlamlandırır: 3H'nin 1/2'ye yakınlığı UHF'in
ortalama-alan doğasının imzası, buradaki 1/8 ise 2B'de dalgalanmaların
ortalama alanı YIKTIĞININ imzasıdır. İkisi de tek başına "bir sayı tutturmuş"
gibi okunabilirdi; birlikte okunduklarında hangi yöntemin hangi rejimi
gördüğünü söylüyorlar.

===========================================================================
MALİYET VE `slow` KARARI
===========================================================================

Ölçüldü (pytest `--durations`, seri, tek süreç, bu makine):

    fixture              süre (iki koşum)   isteyen testler
    tam_sayım            0.04 s             5 test (hepsi işaretsiz)
    fss_kısa             1.36 s             2 test (biri işaretsiz)
    fss_kısa @ T_c+0.05  1.22 s             1 test (`slow`)
    fss_uzun            43.87 / 45.76 s     3 test (hepsi `slow`)
    binder_taraması     62.85 / 65.32 s     5 test (hepsi `slow`)

    dosyanın tamamı  18 test / 110.04 s
    `-m "not slow"`   8 test /   1.76 s
    `-m slow`        10 test / 114.03 s

(İki koşum arasındaki %4'lük oynama makine gürültüsüdür; sayılar iki ayrı
`--durations` koşumundan alındı ve ikisi de yazıldı.)

Bu dosya PAHALIDIR ve plan da öyle diyor. `slow` eşiği bu depoda ölçülmüştür
(call ≥ 2.0 s). İki taşıyıcı fixture `slow` işaretli testlerin ARKASINDADIR,
yani `-m "not slow"` onları hiç kurmaz. **Taşıyıcı iddia yine de tamamen
yavaş şeride düşmüyor:** kritik üs iddiası hızlı şeritte kısa zincirlerle bir
kez daha kuruluyor ve BEDELİ ÖLÇÜLDÜ — 3σ sınırı 0.009674 yerine 0.057869,
yani **%498 daha geniş (5.98 kat)**, ama sınır yine hata çubuğundan geliyor.
Onsager iddiasının hızlı şerit karşılığı MC değil TAM SAYIMDIR (0.04 s) ve
orada hata çubuğu YOKTUR — ucuzlatılmış bir MC değil, BAŞKA bir kanıt türü.

DÜRÜSTLÜK NOTU: fixture maliyeti pytest'in `--durations` çıktısında `call`
değil `setup` olarak görünür, yani deponun "call ≥ 2.0 s" eşiği tek başına
bu dosyayı DOĞRU sınıflandıramazdı. Burada setup + call birlikte ölçüldü ve
`slow` kararı ona göre verildi: pahalı iki fixture'ı isteyen HER test
işaretlidir. Hızlı şeride kalan toplam yük **1.76 s**tir ve hiçbir tekil
testin setup + call'u 2.0 s'yi geçmez.

===========================================================================
MUTASYON SINAVI — 13 MUTANT, HAYATTA KALAN YOK
===========================================================================

`PYTHONDONTWRITEBYTECODE=1`, her koşumdan önce `__pycache__` silindi
(planın Task 1'de ölçtüğü bayat `.pyc` tuzağı), `atexit` + SIGTERM/SIGINT
ile geri yükleme. Kontrol koşumu: 18 passed / 110.04 s.

    mutant                                       düşen  hızlı şeritte yakalayan
    M1  Binder oranı TERS çevrilir (kaynak)         4    TAM_SAYIM_AGIRLIKSIZ_LIMITI
    M2  Binder'da `3` çarpanı düşer (kaynak)        2    TAM_SAYIM_AGIRLIKSIZ_LIMITI
    M3  `plateau_reached` HEP True (kaynak)         3    KISA_KOSUMDA_PLATO_OLUSMUYOR
    M4  `binning_error` naif SEM döndürür (kaynak)  1    — (YALNIZ yavaş şerit)
    M5  Wolff `p_bond`da 2 çarpanı düşer (kaynak)   9    KRITIK_US_KISA + KISA_PLATO
    M6  FSS eğimi `+β/ν` okunur (test)              4    KRITIK_US_KISA
    M7  `TC` +0.050 (test)                         10    (bkz. NOT)
    M8  `TC` +0.020 (test)                         10    (bkz. NOT)
    M9  `TC` +0.010 (test)                         10    (bkz. NOT)
    M10 `TC` −0.010 (test)                         11    (bkz. NOT)
    M11 `TC` +0.005 (test)                         10    (bkz. NOT)
    M12 kesişim ara değerleme yerine IZGARA NOKTASI 6    IZGARA_SISTEMATIGI + 3 test
    M13 bootstrap Binder'ında `3 → 2` (test)        1    — (YALNIZ yavaş şerit)

    DÜRÜST OLUMSUZ: **M4 ve M13 hızlı şeritte GÖRÜNMEZ.** M4'ü yalnız
    `CIPAYI_BESLEYEN_KOSUMLARDA_PLATO_OLUSUYOR`un `τ_int` kilidi, M13'ü
    yalnız `BINDER_MOMENT_YOLU_ILE_KAYNAK_YOLU_AYNI` yakalıyor; ikisi de
    pahalı fixture'ın arkasında. Yani `-m "not slow"` koşan biri bu iki
    bozulmayı GÖREMEZ ve bu, hızlı şeridin kapsamı hakkında bir beyandır.

ÜÇ SONUÇ AYRICA KAYDEDİLDİ:

- **M1 ve M2, kesişim testini DEĞİL limit/tutarlılık testlerini düşürüyor.**
  Planın beklentisi ("Binder ters çevrilirse kesişim testi kırmızı") ölçümle
  çürüdü; sebebi yukarıda cebirsel olarak yazılı. M1'in düşürdüğü dört testin
  ikisi (`ONSAGER_BINDER_KESISIMINDEN_OKUNUYOR`,
  `UC_IKILI_KESISIM_...`) fizik assert'inden değil, U değerlerinin
  transkripsiyon kilidinden düşüyor.
- **M5 (Wolff `p_bond`) dosyayı 6.36 s'de düşürüyor** (kontrol koşumu
  110.04 s) — çünkü mutant etkin olarak `2T` örnekliyor, kümeler küçülüyor ve
  koşum hızlanıyor. Yani en tehlikeli fizik hatası aynı zamanda en hızlı
  yakalanan; bir mutasyon sınavında koşum SÜRESİ de bir sinyaldir.

- **M12 (ızgara noktası) altı testi düşürüyor ve dördü hızlı şeritte.**
  Yani "ara değerleme yapmak" bu dosyada süs değil, sınanan bir karardır;
  planın yaptığı gibi ızgara noktası raporlamak testleri kırar.

`git status --porcelain` mutasyon sonrası: yalnız bu dosyanın kendisi
(`?? tests/spin/test_cipalar.py`) — kaynak ağacı temiz.
"""

from __future__ import annotations

import numpy as np
import pytest

from qlab.engine.spin.lattice import SpinState
from qlab.engine.spin.wolff import run_wolff
from qlab.observable.spin_obs import binder_cumulant, binning_error

# --------------------------------------------------------------------------
# Sabitler -- HEPSI ACIK, RNG tohumlari dahil
# --------------------------------------------------------------------------

#: Onsager'in tam cozumu. BU DOSYADA YALNIZ KARSILASTIRMA DEGERIDIR:
#: hicbir kosumun sicakligi bundan turetilmez (T izgarasi yuvarlak sayilardir)
#: ve kesisimi bulan tahminci onu HIC gormez. Tek istisna FSS cipasidir --
#: o kosum T_c'de yapilir ve bu ACIKCA beyan edilir (modul dokstringi).
TC = 2.0 / np.log(1.0 + np.sqrt(2.0))

TAM_US = 0.125          # 2B Ising: beta = 1/8, nu = 1 -> beta/nu = 1/8
ORTALAMA_ALAN_US = 1.0  # Landau/ortalama alan: beta = 1/2, nu = 1/2
UHF_3H_US = 0.496190    # Modul 3H, N = 6 (tests/tb/test_downfold_cipa.py)

TOHUM = 20260824
K = 24
ZINCIR = tuple(TOHUM + 1000 * k for k in range(K))

BOOTSTRAP_TOHUM = 20260826
N_BOOTSTRAP = 3000

# Binder taramasi
LS_BINDER = (8, 16, 24)
TIZGARA = (2.24, 2.26, 2.28)   # YUVARLAK sayilar, adim 0.02
N_BURN_B, N_SWEEP_B = 300, 1500
CIFTLER = ((0, 1), (0, 2), (1, 2))

# FSS
LS_FSS = (8, 12, 16, 24, 32)
N_BURN_F, N_SWEEP_F = 300, 1500

# FSS kisa (hizli serit)
LS_KISA = (8, 12, 16)
N_BURN_K, N_SWEEP_K = 40, 160
KISA_KAYMA = 0.05


# --------------------------------------------------------------------------
# Yardimcilar
# --------------------------------------------------------------------------


def _wolff_serisi(L: int, T: float, tohum: int, n_burn: int, n_sweep: int):
    """Tek zincir. Soğuk (tekbiçim) başlangıç, tohum AÇIK ve SABİT.

    Wolff seçildi çünkü T_c'de Metropolis'in kritik yavaşlaması en kötü
    yerdedir (Task 4: L = 32, T_c'de τ_int Metropolis'te 53.78, Wolff'ta
    3.07). Sıcak başlangıçla ölçüldüğünde FSS üssü 0.122372 çıkıyor
    (soğukta 0.122941) — fark 0.000569 = 0.18σ, yani başlangıç koşulunun
    yanlılığı bu pencerede YOK.
    """
    return run_wolff(
        SpinState.uniform(L=L, n_component=1),
        temperature=T,
        coupling=1.0,
        field=0.0,
        rng=np.random.default_rng(tohum),
        n_sweep=n_sweep,
        n_burn=n_burn,
    ).magnetisation_series


def _binder_momentlerden(m2, m4):
    """`binder_cumulant`ın moment düzeyindeki AYNISI — bootstrap için.

    Merkezî değerler kaynağın `binder_cumulant`ından okunur; bootstrap 3000
    kez yeniden örneklediği için moment düzeyinde çalışır. İki yolun BİREBİR
    aynı sayıyı verdiği ayrı bir testte kilitlenir (ölçülen fark ≤ 2.220e-16),
    yani kaynağı mutasyona uğratan bir değişiklik her iki yolu da bozar.
    """
    return 1.0 - m4 / (3.0 * m2 * m2)


def _kesisim(izgara, fark) -> float:
    """ΔU'nun işaret değiştirdiği ilk aralıkta DOĞRUSAL ARA DEĞERLEME kökü.

    Yalnız `(izgara, fark)` görür — Onsager'in değerini GÖRMEZ. İşaret
    değişimi yoksa `nan` döner ve çağıran bunu `max(oran, 0)` gibi
    KIRPMAZ: "kesişim yok" ile "kesişim burada" aynı şey değildir
    (`spin_obs.correlation_length`ın aynı dersi).
    """
    for j in range(len(izgara) - 1):
        if fark[j] == 0.0:
            return float(izgara[j])
        if fark[j] * fark[j + 1] < 0.0:
            t = fark[j] / (fark[j] - fark[j + 1])
            return float(izgara[j] + t * (izgara[j + 1] - izgara[j]))
    return float("nan")


def _log_log_egim(ms, ls) -> float:
    """`d ln⟨|m|⟩ / d ln L` — en küçük kareler, `np.polyfit`e bağımlı DEĞİL."""
    x = np.log(np.asarray(ls, dtype=float))
    y = np.log(np.asarray(ms, dtype=float))
    return float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))


def _us(ms, ls) -> float:
    """β/ν = **−** d ln⟨|m|⟩ / d ln L. İşaret fizik: mıknatıslanma L ile DÜŞER."""
    return -_log_log_egim(ms, ls)


def _bootstrap_indisleri() -> np.ndarray:
    """(N_BOOTSTRAP, K) zincir etiketi — TEK sefer, TÜM tahminciler için aynı.

    Zincir etiketi üzerinden yeniden örnekleme BİLİNÇLİ: aynı `k` etiketi her
    (L, T) koşumunda aynı tohumu kullanır, dolayısıyla akışlar arasında artık
    bir bağ kalabilir. Etiketi bir bütün olarak yeniden örneklemek bu bağı
    korur (bağımsız varsaymak, hata çubuğunu olduğundan dar gösterirdi).
    """
    return np.random.default_rng(BOOTSTRAP_TOHUM).integers(0, K, (N_BOOTSTRAP, K))


# --------------------------------------------------------------------------
# TAM SAYIM yardimcilari -- L = 3 (512 durum) ve L = 4 (65536 durum)
# --------------------------------------------------------------------------


def _tam_tablo(L: int):
    """Bütün 2^(L·L) durumu sayar; `(|m| dizisi, E_tekil, |m|_tekil, katlılık)`.

    Enerji BAĞIMSIZ yazılır: her bağ TAM BİR KEZ (her sitenin sağ ve aşağı
    komşusu); `lattice.neighbour_sum` + 0.5 düzeltmesi KULLANILMAZ.
    `test_wolff.py::_tam_sayim` ile aynı desen.

    (E, |m|) ikilileri sıkıştırılır — L = 3'te 11, L = 4'te 45 kova kalır,
    böylece 40 001 noktalı ızgara merdiveni saniyenin altında koşar.
    """
    n = L * L
    idx = np.arange(2**n, dtype=np.int64)
    bitler = (idx[:, None] >> np.arange(n)) & 1
    s = (2 * bitler - 1).astype(float).reshape(-1, L, L)
    bag = np.sum(s * np.roll(s, -1, axis=1), axis=(1, 2)) + np.sum(
        s * np.roll(s, -1, axis=2), axis=(1, 2)
    )
    E = -bag
    mm = np.abs(s.mean(axis=(1, 2)))
    tekil, sayim = np.unique(np.stack([E, mm], axis=1), axis=0, return_counts=True)
    return mm, tekil[:, 0], tekil[:, 1], sayim.astype(float)


def _tam_binder(tablo, T, *, carpan: float = 3.0, ters: bool = False):
    """Boltzmann ağırlıklı TAM Binder; `T` skaler ya da dizi olabilir.

    `carpan` ve `ters` yalnız MUTASYON KÖRLÜĞÜNÜ ölçmek için vardır
    (modül dokstringi): `carpan = 1.0` üçün düştüğü mutant, `ters = True`
    oranın ters çevrildiği mutant.
    """
    _, E, mm, g = tablo
    t = np.atleast_1d(np.asarray(T, dtype=float))[:, None]
    w = g[None, :] * np.exp(-(E[None, :] - E.min()) / t)
    Z = w.sum(axis=1)
    m2 = (w * mm[None, :] ** 2).sum(axis=1) / Z
    m4 = (w * mm[None, :] ** 4).sum(axis=1) / Z
    u = 1.0 - (m2 * m2) / (carpan * m4) if ters else 1.0 - m4 / (carpan * m2 * m2)
    return u if np.ndim(T) else float(u[0])


def _tam_kesisim(adim: float, tablo3, tablo4, **kw) -> float:
    """`arange(1.0, 5.0 + adim/2, adim)` ızgarasında tam-sayım kesişimi."""
    izgara = np.arange(1.0, 5.0 + adim / 2, adim)
    fark = _tam_binder(tablo4, izgara, **kw) - _tam_binder(tablo3, izgara, **kw)
    return _kesisim(izgara, fark)


# --------------------------------------------------------------------------
# Fixture'lar -- modul kapsamli. PAHALI OLANLARI YALNIZ `slow` TESTLER ISTER.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tam_sayim():
    """L = 3 ve L = 4'ün tam sayım tabloları. Ölçülen süre 0.55 s."""
    return _tam_tablo(3), _tam_tablo(4)


@pytest.fixture(scope="module")
def binder_taramasi():
    """PAHALI (ölçülen seri süre 62.94 s) — YALNIZ `slow` testler ister.

    Döner: U ızgarası (kaynağın `binder_cumulant`ından), moment dizileri,
    plato bayrakları, τ_int'ler, üç ikili kesişim ve bootstrap σ'ları.
    """
    m2 = np.zeros((len(LS_BINDER), len(TIZGARA), K))
    m4 = np.zeros_like(m2)
    plato = np.zeros_like(m2, dtype=bool)
    tau = np.zeros_like(m2)
    u_kaynak = np.zeros((len(LS_BINDER), len(TIZGARA)))

    for a, L in enumerate(LS_BINDER):
        for b, T in enumerate(TIZGARA):
            seriler = []
            for k, tohum in enumerate(ZINCIR):
                s = _wolff_serisi(L, T, tohum, N_BURN_B, N_SWEEP_B)
                seriler.append(s)
                m2[a, b, k] = float(np.mean(s**2))
                m4[a, b, k] = float(np.mean(s**4))
                sonuc = binning_error(s)
                plato[a, b, k] = sonuc.plateau_reached
                tau[a, b, k] = sonuc.tau_int
            # MERKEZI DEGER KAYNAKTAN: zincirler esit uzunlukta oldugu icin
            # birlestirilmis serinin momentleri = zincir momentlerinin ortalamasi.
            u_kaynak[a, b] = binder_cumulant(np.concatenate(seriler))

    indis = _bootstrap_indisleri()
    kesisimler, sigmalar, gecerli_sayilar = [], [], []
    for (a, b) in CIFTLER:
        kok = _kesisim(TIZGARA, u_kaynak[b] - u_kaynak[a])
        ornekler = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            secim = indis[i]
            u_b = _binder_momentlerden(
                m2[:, :, secim].mean(axis=2), m4[:, :, secim].mean(axis=2)
            )
            ornekler[i] = _kesisim(TIZGARA, u_b[b] - u_b[a])
        gecerli = np.isfinite(ornekler)
        kesisimler.append(kok)
        sigmalar.append(float(ornekler[gecerli].std(ddof=1)))
        gecerli_sayilar.append(int(gecerli.sum()))

    return {
        "U": u_kaynak,
        "U_moment": _binder_momentlerden(m2.mean(axis=2), m4.mean(axis=2)),
        "plato": plato,
        "tau": tau,
        "kesisimler": kesisimler,
        "sigmalar": sigmalar,
        "gecerli": gecerli_sayilar,
    }


def _fss_kosumu(ls, n_burn, n_sweep, temperature):
    """K zincirlik FSS koşumu; `⟨|m|⟩` dizisi, plato bayrakları, üs ve σ."""
    m = np.zeros((len(ls), K))
    plato = np.zeros((len(ls), K), dtype=bool)
    for a, L in enumerate(ls):
        for k, tohum in enumerate(ZINCIR):
            s = _wolff_serisi(L, temperature, tohum, n_burn, n_sweep)
            m[a, k] = float(np.mean(s))
            plato[a, k] = binning_error(s).plateau_reached
    indis = _bootstrap_indisleri()
    us = _us(m.mean(axis=1), ls)
    ornekler = np.array(
        [_us(m[:, indis[i]].mean(axis=1), ls) for i in range(N_BOOTSTRAP)]
    )
    return {
        "m": m,
        "ort": m.mean(axis=1),
        "plato": plato,
        "us": us,
        "sigma": float(ornekler.std(ddof=1)),
    }


@pytest.fixture(scope="module")
def fss_uzun():
    """PAHALI (ölçülen seri süre 44.24 s) — YALNIZ `slow` testler ister."""
    return _fss_kosumu(LS_FSS, N_BURN_F, N_SWEEP_F, TC)


@pytest.fixture(scope="module")
def fss_kisa():
    """HIZLI ŞERİT (ölçülen seri süre 1.33 s) — aynı iddia, geniş çubuk."""
    return _fss_kosumu(LS_KISA, N_BURN_K, N_SWEEP_K, TC)


@pytest.fixture(scope="module")
def fss_kisa_kaymali():
    """Kısa konfigürasyon, `T_c + 0.05`te (ölçülen seri süre 1.19 s)."""
    return _fss_kosumu(LS_KISA, N_BURN_K, N_SWEEP_K, TC + KISA_KAYMA)


# ==========================================================================
# 1. ONSAGER CIPASI -- TAM SAYIM (hata cubugu YOK)
# ==========================================================================


def test_TAM_SAYIM_BINDER_EGRISI_UCTE_IKIDEN_SIFIRA_INIYOR(tam_sayim):
    """SINIF: DAİRESEL DEĞİL — Binder'ın iki limiti KAPALI FORMDAN bilinir.

    Düzenli fazda (T → 0) `m` sabittir, `⟨m⁴⟩ = ⟨m²⟩²`, yani U → 2/3.
    Düzensiz fazda U küçülür. İkisi de MC'ye hiç sorulmaz.

    ÖLÇÜLDÜ (23.08.2026, bu depoda; L = 3'ün 512, L = 4'ün 65536 durumunun
    TAMAMI, ağırlıklar `exp(-(E-E_min)/T)`):

        T          U(L=3)        U(L=4)        ΔU = U4 − U3
        0.5      0.666666614   0.666666634    +2.0e-08
        1.0      0.666479861   0.666553757    +7.389600e-05
        1.5      0.662468016   0.663958703    +1.490687e-03
        2.0      0.642125099   0.644810791    +2.685692e-03
        T_c      0.620146561   0.617199318    −2.947242e-03
        2.5      0.595374527   0.581527659    −1.384687e-02
        3.5      0.462022037   0.380777825    −8.124421e-02
        6.0      0.254156063   0.158615180    −9.554088e-02
        50.0     0.086798093   0.048835283    −3.796281e-02

    ÜÇ FARKLI İŞ YAPAN ÜÇ KİLİT:

    1. `U(0.5) = 2/3` — DÜZENLİ FAZ LİMİTİ. `3` çarpanı düşen mutantta bu
       −1.59e-07 olur (2/3 değil 0), yani limit o mutantı yakalar.
    2. U(T) TEKDÜZE AZALIR. Oran TERS çevrilen mutantta U ARTAR
       (0.666667 → 0.689525 → 0.725398), yani tekdüzelik onu yakalar.
       Not: ters mutantın `U(0.5)`si de 2/3'tür (0.666666720), yani limit
       tek başına yetmezdi — iki kilit AYRI mutantları yakalıyor.
    3. L = 4 sütununun üç değeri Task 4'ün BAĞIMSIZ olarak kilitlediği tam
       sayım değerleriyle BİREBİR aynı (0.644810791 / 0.617199318 /
       0.380777825, `tests/spin/test_spin_obs.py` satır 572-574). İki ayrı
       yazılmış sayıcı aynı sayıyı veriyor — transkripsiyon denetimi.
    """
    tab3, tab4 = tam_sayim
    izgara = np.array([0.5, 1.0, 1.5, 2.0, TC, 2.5, 3.5, 6.0, 50.0])
    u3 = _tam_binder(tab3, izgara)
    u4 = _tam_binder(tab4, izgara)

    # 1) duzenli faz limiti: FIZIK IDDIASI
    assert u3[0] == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert u4[0] == pytest.approx(2.0 / 3.0, abs=1e-6)
    # 2) tekduze azalma: FIZIK IDDIASI
    assert np.all(np.diff(u3) < 0.0), f"U(L=3) tekduze azalmiyor: {u3}"
    assert np.all(np.diff(u4) < 0.0), f"U(L=4) tekduze azalmiyor: {u4}"
    # 3) Task 4'un bagimsiz sayimiyla transkripsiyon denetimi
    assert u4[3] == pytest.approx(0.644810791, abs=1e-9)
    assert u4[4] == pytest.approx(0.617199318, abs=1e-9)
    assert u4[6] == pytest.approx(0.380777825, abs=1e-9)
    # 4) bu dosyanin kendi tablosu
    assert u3[3] == pytest.approx(0.642125099, abs=1e-9)
    assert u3[4] == pytest.approx(0.620146561, abs=1e-9)
    assert float(u4[8] - u3[8]) == pytest.approx(-3.796281e-02, abs=1e-8)


def test_TAM_SAYIM_AGIRLIKSIZ_LIMITI_KAYNAKLA_AYNI(tam_sayim):
    """SINIF: transkripsiyon denetimi — ağırlıklı formül ↔ `binder_cumulant`.

    `T → ∞`da bütün Boltzmann ağırlıkları eşitlenir, yani ağırlıklı Binder
    tam olarak `binder_cumulant(|m| dizisi)`ne dönüşür. Bu dosyanın tam
    sayım aritmetiğini Task 4'ün kaynağına BAĞLAR: kaynak mutasyona uğrarsa
    bu test de düşer.

    ÖLÇÜLDÜ: L = 3 → 0.074074074 (her iki yol), L = 4 → 0.041666667.
    (Bu sayılar 0 DEĞİLDİR: sonsuz sıcaklıkta `|m|` dağılımı küçük kafeste
    Gauss değildir; Gauss limiti L → ∞'da kurulur.)
    """
    for tab, beklenen in zip(tam_sayim, (0.074074074, 0.041666667)):
        agirliksiz = _tam_binder(tab, 1.0e12)
        kaynaktan = binder_cumulant(tab[0])
        assert agirliksiz == pytest.approx(kaynaktan, abs=1e-9)
        assert kaynaktan == pytest.approx(beklenen, abs=1e-9)


def test_TAM_SAYIM_KESISIMI_ONSAGER_DEN_YUZDE_4_ASAGIDA(tam_sayim):
    """SINIF: DAİRESEL DEĞİL — sonlu-boyut kayması GÜRÜLTÜSÜZ ölçülüyor.

    L = 3 × L = 4 kesişimi tam sayımdan gelir: hata çubuğu YOK, tohum YOK,
    tartışma yok. ÖLÇÜLDÜ: **2.1692678** (ızgara adımı 1e-4).

    Onsager 2.269185314 → sapma **−0.0999175**, yani **−%4.4032**.

    HÜKÜM: "Binder kesişimi T_c'yi verir" ifadesi L'den BAĞIMSIZ DEĞİLDİR.
    En küçük kafeslerde kesişim T_c'nin %4.4 altındadır ve bu bir gürültü
    değil, cebirsel bir gerçektir. Aynı kayma L = 8/16/24'te 0.0027'ye
    (%0.12) iner — yani 37 kat küçülür — ama SIFIRLANMAZ.
    """
    tab3, tab4 = tam_sayim
    kok = _tam_kesisim(0.0001, tab3, tab4)
    assert kok == pytest.approx(2.1692678, abs=1e-6)
    # FIZIK IDDIASI: kesisim Onsager DEGILDIR ve fark kucuk degildir.
    assert TC - kok > 0.09, f"kesisim {kok:.7f}, Onsager'den fark {TC-kok:.7f}"
    assert (kok - TC) / TC == pytest.approx(-0.044032, abs=1e-5)


#: Izgara merdiveni: (adim, kok, referanstan sistematik). Referans, adim
#: 1e-4'luk izgaranin koku (2.169267842248259) ve TESTIN ICINDE hesaplanir --
#: yuvarlanmis bir sabit kullanilsaydi en kucuk iki basamak (1e-6, 1e-7)
#: referansin kendi yuvarlama hatasi tarafindan yutulurdu. Bu, ilk yazimda
#: gercekten oldu: literal 2.1692678 ile son satirin sistematigi -4.213e-07
#: yerine -3.790e-07 okunuyordu.
IZGARA_MERDIVENI = (
    (0.5000, 2.1692678, 2.0812243, -8.8044e-02),
    (0.2500, 2.1692678, 2.1347509, -3.4517e-02),
    (0.1000, 2.1692678, 2.1643021, -4.9658e-03),
    (0.0500, 2.1692678, 2.1680085, -1.2594e-03),
    (0.0400, 2.1692678, 2.1686729, -5.9497e-04),   # PLANIN izgara adimi
    (0.0200, 2.1692678, 2.1690522, -2.1564e-04),   # BU DOSYANIN izgara adimi
    (0.0100, 2.1692678, 2.1692529, -1.4987e-05),
    (0.0050, 2.1692678, 2.1692610, -6.8344e-06),
    (0.0010, 2.1692678, 2.1692674, -4.2128e-07),
)


def test_ARA_DEGERLEMENIN_IZGARA_SISTEMATIGI_GURULTUSUZ_OLCULDU(tam_sayim):
    """SINIF: DAİRESEL DEĞİL — çözünürlük TAHMİN EDİLMİYOR, ÖLÇÜLÜYOR.

    Planın en zayıf noktası buydu: adım 0.04'lük bir ızgarada bulunan
    kesişimin sapması %0.4 (= 0.0092, adımın DÖRTTE BİRİ) diye yazılmıştı.
    Bir kesişimi ızgara adımının dörtte biri hassasiyetle konumlandırdığını
    iddia etmek, ancak ara değerlemenin sistematiği ÖLÇÜLMÜŞSE mümkündür.

    Burada ölçülüyor — ve MC gürültüsü OLMADAN, çünkü tam sayım kullanılıyor.
    Referans, adım 1e-4'lük ızgaranın kökü (2.1692678):

        ızgara adımı   kök          referanstan sistematik
        0.5000         2.0812243    −8.8044e-02
        0.2500         2.1347509    −3.4517e-02
        0.1000         2.1643021    −4.9658e-03
        0.0500         2.1680085    −1.2594e-03
        0.0400         2.1686729    −5.9497e-04    ← planın adımı
        0.0200         2.1690522    −2.1564e-04    ← bu dosyanın adımı
        0.0100         2.1692529    −1.4987e-05
        0.0050         2.1692610    −6.8344e-06
        0.0010         2.1692674    −4.2128e-07

    ÜÇ HÜKÜM:

    1. Sistematik ADIMLA TEKDÜZE küçülüyor — yani bu bir yakınsama, tek bir
       şanslı nokta değil.
    2. Adım 0.02'de sistematik **2.1564e-04**tür — bu dosyanın MC σ'sının
       (3.0e-03) **on dörtte biri**. Yani bu dosyada çözünürlüğü belirleyen
       şey ızgara DEĞİL istatistiktir; planın adımı 0.04'te bile sistematik
       yalnız 5.9e-04.
    3. Ama ara değerleme YAPILMAZSA çözünürlük adımın kendisidir. Planın
       raporladığı 2.260, adım 0.04'lük ızgaranın bir NOKTASIDIR; aynı
       veriden ara değerlemeyle 2.26317/2.26363/2.26406 okunur. Yani planın
       "%0.4 sapma" cümlesi, çözünürlüğü 0.04 olan bir ölçümden 0.0092'lik
       bir sapma okumaktır — sapma, çözünürlüğün dörtte biri.
    """
    tab3, tab4 = tam_sayim
    referans = _tam_kesisim(0.0001, tab3, tab4)
    assert referans == pytest.approx(2.169267842, abs=1e-8)

    sistematikler = []
    for adim, ref_beklenen, kok_beklenen, sistematik in IZGARA_MERDIVENI:
        kok = _tam_kesisim(adim, tab3, tab4)
        assert ref_beklenen == pytest.approx(referans, abs=5e-8)
        assert kok == pytest.approx(kok_beklenen, abs=1e-6)
        assert kok - referans == pytest.approx(sistematik, rel=2e-3)
        sistematikler.append(abs(kok - referans))

    # HUKUM 1: yakinsama TEKDUZE.
    assert all(
        sistematikler[i] > sistematikler[i + 1] for i in range(len(sistematikler) - 1)
    ), f"sistematik tekduze kuculmuyor: {sistematikler}"
    # HUKUM 2: bu dosyanin adiminda (0.02) sistematik, MC sigma'sinin cok altinda.
    assert sistematikler[5] < 3.0e-03 / 10.0


def test_KESISIM_MONOTON_YENIDEN_PARAMETRELEMEYE_KOR(tam_sayim):
    """SINIF: kapsam beyanı — PLANIN BİR MUTANT BEKLENTİSİ ÖLÇÜMLE ÇÜRÜDÜ.

    Plan (satır 799) `⟨m⁴⟩/⟨m²⟩²` ters çevrilirse kesişim testinin kırmızı
    olmasını bekliyor. Cebirsel olarak YANLIŞ: kesişim `U_3(T) = U_4(T)`
    noktasıdır, bu da `r_L = ⟨m⁴⟩_L/⟨m²⟩_L²` oranlarının EŞİTLİĞİNE denktir.
    `U = 1 − r/3` yerine `U = 1 − 1/(3r)` yazmak, ya da `3`ü düşürmek,
    `r`nin monoton bir fonksiyonudur ve eşitlik noktasını DEĞİŞTİRMEZ.

    ÖLÇÜLDÜ (tam sayım, adım 0.001):

        doğru                          kök 2.1692674
        ters oran 1 − ⟨m²⟩²/(3⟨m⁴⟩)    kök 2.1692675
        3 çarpanı düşmüş 1 − ⟨m⁴⟩/⟨m²⟩² kök 2.1692674

    Yani kesişim çıpası bu iki mutanta KÖRDÜR. Onları yakalayan şey
    `test_TAM_SAYIM_BINDER_EGRISI_...`in limit ve tekdüzelik kilitleridir
    (ve depo genelinde Task 4'ün Gauss limiti testi). Bu bir kusur değil,
    çıpanın KAPSAMIDIR — ve bu dosyada adlandırılmasa hüküm kapsamın dışına
    taşardı.
    """
    tab3, tab4 = tam_sayim
    dogru = _tam_kesisim(0.001, tab3, tab4)
    ters = _tam_kesisim(0.001, tab3, tab4, ters=True)
    ucsuz = _tam_kesisim(0.001, tab3, tab4, carpan=1.0)

    assert dogru == pytest.approx(2.1692674, abs=1e-6)
    assert ters == pytest.approx(dogru, abs=1e-6)
    assert ucsuz == pytest.approx(dogru, abs=1e-9)
    # ve mutantlarin gercekten FARKLI EGRILER oldugu -- korluk kesisime OZGU
    assert _tam_binder(tab3, 2.0, ters=True) == pytest.approx(0.689525276, abs=1e-9)
    assert _tam_binder(tab3, 2.0, carpan=1.0) == pytest.approx(-0.073624703, abs=1e-9)


# ==========================================================================
# 2. ONSAGER CIPASI -- MONTE CARLO (tasiyici)
# ==========================================================================


def test_ONSAGER_SIMULASYONA_GIRDI_DEGIL():
    """SINIF: yapısal — Onsager'in sayısı hiçbir koşumun parametresi DEĞİL.

    Üç ayrı şey kilitlenir:

    1. T ızgarası 0.02'nin katlarından oluşan YUVARLAK sayılardır; T_c'ye en
       yakın ızgara noktası **0.009185** uzaktadır, yani ızgara T_c'nin
       ÜSTÜNE oturtulmamıştır.
    2. Izgara T_c'yi KAPSAR (kapsamasaydı işaret değişimi hiç görülmezdi) —
       bu, "ızgara T_c etrafında kuruldu" itirafının kendisidir ve beyan
       ediliyor: bir kesişimi bulmak için onu içeren bir pencere gerekir.
       Daireselliği kıran şey pencerenin varlığı değil, kesişimin pencere
       İÇİNDE nereye düştüğünün T_c'den bağımsız okunmasıdır.
    3. `_kesisim` yalnız `(ızgara, ΔU)` görür. Bilinen bir kökü olan yapay
       veriyle sınanır: ΔU = (+1, −1) verilirse kök tam ortadadır.
    """
    assert all(abs(t / 0.02 - round(t / 0.02)) < 1e-9 for t in TIZGARA)
    en_yakin = min(abs(TC - t) for t in TIZGARA)
    assert en_yakin == pytest.approx(0.009185, abs=1e-5)
    assert min(TIZGARA) < TC < max(TIZGARA)

    # `_kesisim` T_c'yi gormez: yapay veriyle kok tam ortada.
    assert _kesisim((1.0, 2.0), np.array([+1.0, -1.0])) == pytest.approx(1.5)
    assert _kesisim((1.0, 2.0, 3.0), np.array([+2.0, +1.0, -1.0])) == pytest.approx(2.5)
    # isaret degisimi yoksa `nan` -- KIRPILMAZ.
    assert np.isnan(_kesisim((1.0, 2.0), np.array([+1.0, +2.0])))


@pytest.mark.slow
def test_BINDER_MOMENT_YOLU_ILE_KAYNAK_YOLU_AYNI(binder_taramasi):
    """SINIF: transkripsiyon denetimi — bootstrap kaynağı ATLAMIYOR.

    Merkezî U değerleri `observable.spin_obs.binder_cumulant`tan, bootstrap
    ise moment düzeyinden gelir. İkisinin aynı sayı olduğu burada kilitlenir,
    yani kaynağı bozan bir mutasyon her iki yolu da bozar.

    ÖLÇÜLEN en büyük fark: **2.220e-16** (çift duyarlıklı yuvarlama).

    `slow` GEREKÇESİ: pahalı fixture'ı ister (62.94 s).
    """
    fark = np.max(np.abs(binder_taramasi["U"] - binder_taramasi["U_moment"]))
    assert fark < 1e-14, f"iki yol ayrisiyor: {fark:.3e}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "cift, kesisim_beklenen, sigma_beklenen, sapma_sigma",
    [
        ((0, 1), 2.265050, 0.004118, -1.00),
        ((0, 2), 2.265253, 0.003020, -1.30),
        ((1, 2), 2.265570, 0.005206, -0.69),
    ],
)
def test_ONSAGER_BINDER_KESISIMINDEN_OKUNUYOR(
    binder_taramasi, cift, kesisim_beklenen, sigma_beklenen, sapma_sigma
):
    """TAŞIYICI TEST. SINIF: DAİRESEL DEĞİL.

    Binder kümülantı boyutsuzdur, yani `U_L(T_c)` kafes boyutundan bağımsızdır
    ve farklı `L` eğrileri T_c'de KESİŞİR. Kesişimin YERİ hiçbir yerden
    verilmedi: T ızgarası yuvarlak sayılar, `_kesisim` T_c'yi görmüyor.

    ÖLÇÜLDÜ (23.08.2026; L = 8/16/24, T = 2.24/2.26/2.28 adım 0.02, Wolff,
    n_burn = 300, n_sweep = 1500, K = 24 zincir, tohum 20260824 + 1000·k,
    soğuk başlangıç, bootstrap tohumu 20260826, 3000 çekiliş):

        çift          kesişim    σ         Onsager sapması   3σ      geçerli
        L=8 × L=16    2.265050   0.004118  −0.004135 (−1.00σ) 0.01235  3000/3000
        L=8 × L=24    2.265253   0.003020  −0.003932 (−1.30σ) 0.00906  3000/3000
        L=16 × L=24   2.265570   0.005206  −0.003616 (−0.69σ) 0.01562  2938/3000

    SINIR HATA ÇUBUĞUNDAN GELİR, ELLE SEÇİLMEZ: `|kesişim − T_c| < 3σ`,
    σ = zincir etiketi üzerinden bootstrap.

    DÜRÜST OLUMSUZ: L=16 × L=24 çiftinde 3000 bootstrap örneğinin **62'si**
    (%2.1) hiç işaret değişimi vermiyor, yani o çift bu istatistikle her
    yeniden örneklemede bir kesişim ÜRETMİYOR. `nan`lar kırpılmıyor, sayılıyor
    ve raporlanıyor.

    AYIRMA GÜCÜ ÖLÇÜLDÜ (modül dokstringi): `TC` sabiti +0.010 ya da −0.020
    kaydırılırsa bu test kırmızıya döner; ±0.005 kaydırılırsa DÖNMEZ.
    0.05'lik kayma 9.6 – 16.6σ ile yakalanır.

    `slow` GEREKÇESİ: pahalı fixture'ı ister (62.94 s).
    """
    a, b = cift
    i = CIFTLER.index(cift)
    kok = binder_taramasi["kesisimler"][i]
    sigma = binder_taramasi["sigmalar"][i]
    sinir = 3.0 * sigma

    # FIZIK IDDIASI ONCE: elle secilmis tolerans degil, hata cubugu.
    assert abs(kok - TC) < sinir, (
        f"L={LS_BINDER[a]}x{LS_BINDER[b]}: kesisim {kok:.6f}, Onsager {TC:.6f}, "
        f"fark {kok-TC:+.6f} = {(kok-TC)/sigma:+.2f} sigma (sinir 3 sigma = {sinir:.5f})"
    )
    # Sonra transkripsiyon denetimi.
    assert kok == pytest.approx(kesisim_beklenen, abs=1e-5)
    assert sigma == pytest.approx(sigma_beklenen, abs=1e-5)
    assert (kok - TC) / sigma == pytest.approx(sapma_sigma, abs=2e-2)


@pytest.mark.slow
def test_UC_IKILI_KESISIM_BU_ISTATISTIKLE_AYIRT_EDILEMIYOR(binder_taramasi):
    """SINIF: çözünürlük beyanı — "TEK kesişim" diye bir şey YOK.

    Üç `L` değeri üç ikili kesişim verir ve sonlu-boyut kayması yüzünden
    çakışmaları beklenmez. Bu pencerede ÖLÇÜLEN:

        kesişimler  2.265050 / 2.265253 / 2.265570
        YAYILIM     0.000519
        en dar 3σ   0.00906

    yani yayılım en dar sınırın **on yedide biri** — üçü AYIRT EDİLEMİYOR.
    "Kesişim L ile şu yöne kayıyor" demek bu veriden okunamaz; nitekim
    ÜÇ BAĞIMSIZ TOHUM BLOĞUNDA sıranın kendisi değişiyor:

        taban 20260824  2.265050 / 2.265253 / 2.265570   sıralı
        taban 20260825  2.270768 / 2.266854 / 2.262618   TERS
        taban 19990101  2.262914 / 2.266272 / 2.273238   sıralı

    Bu, Task 3'ün "tek akıştan okunan bir oran, sınırın kendisi kadar
    gürültülüdür" dersinin bir kere daha çıkmasıdır.

    AMA SAPMA GÜRÜLTÜ DEĞİL: aynı üç blok havuzlandığında
    (blok ortalamaları 2.265291 / 2.266747 / 2.267474)

        havuz 2.266504 ± 0.000642 → Onsager sapması −0.002681 = **−4.18σ**

    yani kesişim gerçekten ve sistematik olarak Onsager'in ALTINDADIR;
    ayırt edilemeyen şey ÇİFTLER ARASI fark, ölçülebilen şey ORTAK kayma.

    `slow` GEREKÇESİ: pahalı fixture'ı ister (62.94 s).
    """
    kokler = np.array(binder_taramasi["kesisimler"])
    yayilim = float(kokler.max() - kokler.min())
    en_dar_3sigma = 3.0 * min(binder_taramasi["sigmalar"])

    assert yayilim < en_dar_3sigma, (
        f"yayilim {yayilim:.6f} en dar 3 sigma sinirini ({en_dar_3sigma:.6f}) "
        "geciyor -- uc kesisim AYRISIYOR, dosyanin anlatisi guncellenmeli"
    )
    assert yayilim == pytest.approx(0.000519, abs=2e-5)
    assert en_dar_3sigma == pytest.approx(0.009060, abs=1e-5)
    # ucunun de sapmasi AYNI YONDE (asagi) -- havuzlanan hukmun tohumu
    assert all(k < TC for k in kokler)


@pytest.mark.slow
def test_CIPAYI_BESLEYEN_KOSUMLARDA_PLATO_OLUSUYOR(binder_taramasi):
    """SINIF: kendi dürüstlük makinesini kendi manşetine uygulamak.

    Task 4'ün `binning_error`ı, plato oluşmamışsa döndürdüğü sayının hatanın
    KENDİSİ değil ALT SINIRI olduğunu kilitler. Bu dosyanın çıpasını besleyen
    her zincirde o bayrak okundu.

    ÖLÇÜLDÜ (72 zincir/L = 3 sıcaklık × 24 tohum, n_sweep = 1500):

        L = 8    71/72 plato   τ_int 1.08 … 2.39  (ortalama 1.56)
        L = 16   68/72 plato   τ_int 1.44 … 3.27  (ortalama 2.09)
        L = 24   63/72 plato   τ_int 1.33 … 3.89  (ortalama 2.41)

    yani %88-99. HÜKÜM İKİ PARÇALI ve karıştırılmamalı:

    - Bu dosyanın hata çubuğu zincir-İÇİ binning'den GELMEZ; K = 24 bağımsız
      zincirin bootstrap'ından gelir. Dolayısıyla plato bayrağı sınırın
      geçerliliğinin ÖN KOŞULU değildir.
    - Bayrak yine de gereklidir çünkü BAĞIMSIZ bir teyittir: her zincir kendi
      otokorelasyonunu çözecek kadar uzun (τ_int ≤ 3.89 ≪ n_sweep = 1500).
      Bayrak `False` olsaydı, K-zincir sınırı hâlâ geçerli olurdu ama
      zincirlerin dengeye oturduğuna dair içeriden hiçbir kanıt kalmazdı.

    KARŞIT ÖRNEK AYNI DOSYADA: kısa (hızlı şerit) FSS koşumunda bayrak
    15/72 — ve orada bu AÇIKÇA beyan edilir
    (`test_KISA_KOSUMDA_PLATO_OLUSMUYOR_...`).

    `slow` GEREKÇESİ: pahalı fixture'ı ister (62.94 s).
    """
    plato = binder_taramasi["plato"]
    tau = binder_taramasi["tau"]
    sayilar = [int(plato[a].sum()) for a in range(len(LS_BINDER))]

    assert sayilar == [71, 68, 63], f"plato sayilari degisti: {sayilar}"
    # FIZIK/YONTEM IDDIASI: her L'de zincirlerin ezici cogunlugu plato yapiyor
    assert all(s >= 0.85 * plato[0].size for s in sayilar)
    # ve tau_int zincir uzunlugunun yaninda kucuk
    assert float(tau.max()) < 4.0
    assert float(tau.max()) * 100.0 < N_SWEEP_B
    assert float(tau.max()) == pytest.approx(3.89, abs=2e-2)


# ==========================================================================
# 3. KRITIK US CIPASI
# ==========================================================================


@pytest.mark.slow
def test_KRITIK_US_OLCULDU_ORTALAMA_ALAN_DISLANDI(fss_uzun):
    """TAŞIYICI TEST. SINIF: DAİRESEL DEĞİL — üs EVRENSELDİR.

    `β/ν` ne `J`den, ne kafes sabitinden, ne birim seçiminden okunabilir;
    yalnızca `⟨|m|⟩`nin `L` ile nasıl söndüğünden çıkar.

    ÖLÇÜLDÜ (23.08.2026; L = 8/12/16/24/32, T = T_c, Wolff, n_burn = 300,
    n_sweep = 1500, K = 24 zincir, tohum 20260824 + 1000·k, soğuk başlangıç,
    bootstrap tohumu 20260826, 3000 çekiliş):

        L        8         12        16        24        32
        ⟨|m|⟩  0.775523  0.740615  0.715627  0.678801  0.654571

        β/ν = 0.122941 ± 0.003225        (3σ = 0.009674)

    DOĞRU CÜMLE — VE "β = 1/8 ÇIKTI" YAZMAK YASAK:
    **β/ν = 0.122941 ölçüldü; tam değer 0.125 (sapma −0.002059 = −0.64σ);
    ORTALAMA ALAN (β/ν = 1.0) 272.0 KAT marjla DIŞLANDI.**

    Kalan sapma sonlu-boyut düzeltmesidir ve SIFIRLANMAZ. Kanıtı, aynı
    veriden `L` alt kümeleriyle yapılan uyumlardır (n_sweep = 3000
    penceresinde, L = 48 dahil): (8,12,16) → 0.121346 ± 0.002830,
    (16,24,32) → 0.129350 ± 0.003974, (24,32,48) → 0.123540 ± 0.004840.
    Bunlar birbirinden 2σ'ya kadar ayrışıyor, yani log-log doğrusu TAM DOĞRU
    DEĞİLDİR; altı noktalık uyumun 0.125'e çok yakın düşmesi düzeltmelerin
    bu pencerede birbirini götürmesindendir.

    `β/ν` ile `β` 2B Ising'de sayısal olarak aynıdır (`ν = 1`), AMA BU DOSYA
    `ν`yü ÖLÇMEZ; o bağ literatürdendir ve burada yalnızca beyan edilir.

    BAŞLANGIÇ KOŞULU YANLILIĞI ÖLÇÜLDÜ VE YOK: aynı pencere sıcak (rastgele
    ±1) başlangıçla 0.122372 veriyor; fark 0.000569 = 0.18σ.

    `slow` GEREKÇESİ: pahalı fixture'ı ister (44.24 s).
    """
    us = fss_uzun["us"]
    sigma = fss_uzun["sigma"]

    # FIZIK IDDIASI 1: tam deger hata cubugunun ICINDE.
    assert abs(us - TAM_US) < 3.0 * sigma, (
        f"beta/nu = {us:.6f} +- {sigma:.6f}, tam deger {TAM_US}: "
        f"sapma {us-TAM_US:+.6f} = {(us-TAM_US)/sigma:+.2f} sigma"
    )
    # FIZIK IDDIASI 2: ortalama alan DISLANIYOR.
    assert abs(ORTALAMA_ALAN_US - us) > 3.0 * sigma
    marj = (ORTALAMA_ALAN_US - us) / sigma
    assert marj > 100.0, f"ortalama alan yalnizca {marj:.1f} sigma ile dislaniyor"

    # Transkripsiyon denetimi.
    assert us == pytest.approx(0.122941, abs=1e-5)
    assert sigma == pytest.approx(0.003225, abs=1e-5)
    assert marj == pytest.approx(272.0, abs=1.0)
    for olculen, beklenen in zip(
        fss_uzun["ort"], (0.775523, 0.740615, 0.715627, 0.678801, 0.654571)
    ):
        assert float(olculen) == pytest.approx(beklenen, abs=1e-5)


@pytest.mark.slow
def test_FSS_TASIYICISINDA_PLATO_OLUSUYOR(fss_uzun):
    """SINIF: kendi dürüstlük makinesini kendi manşetine uygulamak (FSS).

    ÖLÇÜLDÜ (n_sweep = 1500, 24 zincir/L):

        L        8      12     16     24     32
        plato  24/24  24/24  23/24  23/24  18/24      toplam 112/120

    L = 32 en zayıf halka (18/24) ve bu beklenen yöndedir: kritik noktada
    otokorelasyon L ile büyür (Task 4: L = 32, T_c'de Wolff `τ_int` = 3.07).
    Yine de %75 ve toplamda %93. Hata çubuğu zincirler ARASI bootstrap'tan
    geldiği için sınır bu bayrağa bağlı değildir; bayrak bağımsız teyittir.

    `slow` GEREKÇESİ: pahalı fixture'ı ister (44.24 s).
    """
    plato = fss_uzun["plato"]
    sayilar = [int(plato[a].sum()) for a in range(len(LS_FSS))]
    assert sayilar == [24, 24, 23, 23, 18], f"plato sayilari degisti: {sayilar}"
    assert int(plato.sum()) == 112
    assert plato.sum() > 0.9 * plato.size


def test_KRITIK_US_KISA_KOSUMDA_da_KURULUYOR_BEDELI_OLCULDU(fss_kisa):
    """HIZLI ŞERİT KARŞILIĞI — aynı iddia, ÖLÇÜLMÜŞ bedelle.

    Task 3'ün dersi: taşıyıcı iddiayı tamamen yavaş şeride düşürme; aynı
    iddiayı kısa bir koşumla bir kez daha kur ve BEDELİNİ ÖLÇ.

    ÖLÇÜLDÜ (L = 8/12/16, T = T_c, Wolff, n_burn = 40, n_sweep = 160,
    K = 24 zincir, aynı tohumlar, aynı bootstrap):

        L        8         12        16
        ⟨|m|⟩  0.777441  0.747045  0.706736

        β/ν = 0.134989 ± 0.019290     (3σ = 0.057869)
        tam değer 0.125'ten +0.009989 = +0.52σ
        ORTALAMA ALAN 44.8 σ marjla dışlandı

    BEDEL: 3σ sınırı 0.009674 → 0.057869, yani **5.98 KAT daha geniş**
    (%498). Ama sınır YİNE hata çubuğundan geliyor; gevşetilmiş bir tolerans
    değil, daha az veriyle ölçülmüş bir sınır.

    İKİ KOŞUM TUTARLI: |0.134989 − 0.122941| = 0.012048, birleşik 3σ =
    3·√(0.019290² + 0.003225²) = 0.058679 → **0.62σ**. Yani kısa sürüm uzun
    sürümü tekrarlıyor, ayrı bir hüküm vermiyor.

    Ölçülen süre 1.33 s < 2.0 s eşiği, `slow` DEĞİL.
    """
    us = fss_kisa["us"]
    sigma = fss_kisa["sigma"]

    assert abs(us - TAM_US) < 3.0 * sigma, (
        f"kisa kosum: beta/nu = {us:.6f} +- {sigma:.6f}, "
        f"sapma {(us-TAM_US)/sigma:+.2f} sigma"
    )
    assert abs(ORTALAMA_ALAN_US - us) > 3.0 * sigma
    marj = (ORTALAMA_ALAN_US - us) / sigma
    assert marj > 20.0, f"ortalama alan yalnizca {marj:.1f} sigma ile dislaniyor"

    assert us == pytest.approx(0.134989, abs=1e-5)
    assert sigma == pytest.approx(0.019290, abs=1e-5)
    assert marj == pytest.approx(44.8, abs=0.2)
    # BEDEL: sinir gercekten genisledi ve bu bir SAYIDIR.
    assert 3.0 * sigma / 0.009674 == pytest.approx(5.98, abs=0.02)


def test_KISA_KOSUMDA_PLATO_OLUSMUYOR_HATA_ALT_SINIRDIR(fss_kisa):
    """DÜRÜST OLUMSUZ — ve bu dosyanın en önemli beyanlarından biri.

    ÖLÇÜLDÜ (n_sweep = 160): L = 8 → 5/24, L = 12 → 5/24, L = 16 → 5/24,
    toplam **15/72**. Yani kısa koşumda binning eğrisi OTURMUYOR ve Task 4'ün
    kilidi gereği o zincirlerin binning hatası hatanın KENDİSİ DEĞİL bir
    ALT SINIRIDIR.

    NEDEN — ölçülmüş, tahmin edilmemiş: `binning_error`ın varsayılan
    `min_blocks = 32`si ile N = 160'lık bir seride yalnız b = 1, 2, 4 üç blok
    boyutu kalır; plato ölçütü e[-1] ile e[-3]'ü karşılaştırdığı için elde
    K = 40 ve K = 160 bloklu iki nokta vardır ve bunların bağıl saçılımı
    %11.3 ve %5.6'dır. Yani bayrak "bu koşum kendi otokorelasyonunu çözecek
    kadar uzun DEĞİL" diyor — ki DOĞRU söylüyor.

    BU, KISA SÜRÜMÜN SINIRINI GEÇERSİZ KILMAZ, çünkü o sınır zincir-İÇİ
    binning'den değil K = 24 BAĞIMSIZ zincirin bootstrap'ından geliyor.
    Eksik olan şey bağımsız teyittir: uzun sürümde plato 112/120 ile o teyidi
    veriyor, kısa sürümde vermiyor. Fark burada YAZILDI, gizlenmedi.

    Ölçülen süre (fixture paylaşıldığı için) ~0 s, `slow` DEĞİL.
    """
    plato = fss_kisa["plato"]
    sayilar = [int(plato[a].sum()) for a in range(len(LS_KISA))]
    assert sayilar == [5, 5, 5], f"plato sayilari degisti: {sayilar}"
    assert int(plato.sum()) == 15
    # IDDIA: kisa kosumda plato AZINLIKTA -- bu bir beyan, bir kusur degil.
    assert plato.sum() < 0.5 * plato.size


@pytest.mark.slow
def test_FSS_CIPASI_TC_YI_GIRDI_ALIR_AYIRMA_GUCU_OLCULDU(fss_kisa, fss_kisa_kaymali):
    """SINIF: kapsam beyanı + ayırma gücü ÖLÇÜMÜ.

    Binder kesişiminden farklı olarak FSS çıpası T_c'yi GERÇEK BİR GİRDİ
    olarak alır: koşum `T = T_c`de yapılır. Bu daireselliğe yol açmaz (üs
    T_c'den okunamaz) ama bir DUYARLILIK yaratır ve bu ölçülmelidir.

    ÖLÇÜLDÜ (kısa konfigürasyon, L = 8/12/16, n_burn = 40, n_sweep = 160,
    aynı 24 tohum):

        T = T_c          β/ν = 0.134989 ± 0.019290   (0.125'ten +0.52σ)
        T = T_c + 0.05   β/ν = 0.214477 ± 0.016706   (0.125'ten +5.36σ)

    yani 0.05'lik bir T_c hatası kısa sürümde bile YAKALANIR. Kısa sürümün
    tam ayırma gücü ayrıca tarandı: ±0.01 ve ±0.02 YAKALANMIYOR, ±0.05
    yakalanıyor (+5.23σ / −4.45σ), ±0.10 yakalanıyor (+9.56σ / −11.22σ).

    TAŞIYICI (uzun) konfigürasyonun ayırma gücü 10 KAT daha keskin —
    modül dokstringindeki tam tarama: ±0.005'ten itibaren yakalıyor,
    0.05'lik kayma 29.6 – 47.1σ.

    ORTALAMA ALAN DIŞLAMASI T_c HATASINA DAYANIKLI: kaymış koşumda bile
    ortalama alan 47.0σ marjla dışlanıyor. Yani modülün "bu ortalama alan
    DEĞİLDİR" hükmü T_c'nin yerinden bağımsızdır; kırılgan olan yalnızca
    "0.125'e eşit" hükmüdür.

    `slow` GEREKÇESİ: bu test tek başına 1.22 s, AMA `fss_kisa` henüz
    kurulmamışsa ikisi birden 2.65 s eder — işaret EN KÖTÜ hale göre verildi
    (eşik 2.0 s).
    """
    kaymali_us = fss_kisa_kaymali["us"]
    kaymali_sigma = fss_kisa_kaymali["sigma"]

    # AYIRMA GUCU IDDIASI: 0.05'lik kayma tam degerden 3 sigma UZAGA itiyor.
    assert abs(kaymali_us - TAM_US) > 3.0 * kaymali_sigma, (
        f"T_c + 0.05'te beta/nu = {kaymali_us:.6f} +- {kaymali_sigma:.6f}, "
        f"tam degerden {(kaymali_us-TAM_US)/kaymali_sigma:+.2f} sigma -- "
        "cipa 0.05'lik kaymayi AYIRAMIYOR"
    )
    # ... ve kaymamis kosum tam degerin ICINDE kaliyor (karsit kontrol).
    assert abs(fss_kisa["us"] - TAM_US) < 3.0 * fss_kisa["sigma"]

    # ORTALAMA ALAN dislamasi kaymaya DAYANIKLI.
    kaymali_marj = (ORTALAMA_ALAN_US - kaymali_us) / kaymali_sigma
    assert kaymali_marj > 20.0

    # Transkripsiyon denetimi.
    assert kaymali_us == pytest.approx(0.214477, abs=1e-5)
    assert kaymali_sigma == pytest.approx(0.016706, abs=1e-5)
    assert (kaymali_us - TAM_US) / kaymali_sigma == pytest.approx(5.36, abs=2e-2)
    assert kaymali_marj == pytest.approx(47.0, abs=0.2)


@pytest.mark.slow
def test_MODULLER_ARASI_BAG_UHF_ORTALAMA_ALAN_2B_ISING_DEGIL(fss_uzun):
    """SINIF: DAİRESEL DEĞİL — aynı depoda İKİ üs, DÖRT KAT fark (spec §3.3).

    Modül 3H (`tests/tb/test_downfold_cipa.py`) UHF indirgemesinde kademeli
    alanın çatallanma üssünü ölçtü: N = 6/10/14 için **0.496190 / 0.497600 /
    0.497811** — ortalama alanın tam değeri 1/2'dir ve sapma N ile küçülüyor.
    UHF bir ORTALAMA ALAN yöntemi olduğu için bu beklenen sonuçtur.

    Bu dosya aynı depoda, aynı tür kendiliğinden simetri kırılmasını 2B
    Ising'de ölçüyor: **0.122941 ± 0.003225**.

        oran 0.496190 / 0.122941 = **4.036**
        fark 0.373249 = **115.8σ**

    İKİ ÖLÇÜM BİRBİRİNİ BAĞLAMLANDIRIR. Tek başına "0.4962 ölçtük, 1/2'ye
    yakın" bir tautoloji şüphesi taşırdı (ortalama alan zaten 1/2 verir);
    tek başına "0.1229 ölçtük, 1/8'e yakın" da öyle. Birlikte okunduklarında
    söyledikleri şey şudur: AYNI kod tabanı, AYNI tür çatallanma, ve yöntemin
    ortalama alan olup olmamasına göre DÖRT KAT farklı üs. Yani ölçülen şey
    "bir sayı tutturmak" değil, dalgalanmaların 2B'de ortalama alanı
    YIKMASIDIR.

    KAPSAM: 3H'nin ölçtüğü üs `β`, buradaki `β/ν`dir. 2B Ising'de `ν = 1`
    olduğu için ikisi sayısal olarak karşılaştırılabilir — ama BU DOSYA
    `ν`yü ölçmez ve `ν = 1` bağı literatürdendir. Bu, karşılaştırmanın
    zayıf halkasıdır ve gizlenmiyor.

    `slow` GEREKÇESİ: pahalı fixture'ı ister (44.24 s).
    """
    us = fss_uzun["us"]
    sigma = fss_uzun["sigma"]

    assert abs(UHF_3H_US - us) > 3.0 * sigma
    ayrim = (UHF_3H_US - us) / sigma
    assert ayrim > 50.0, f"iki us yalnizca {ayrim:.1f} sigma ile ayriliyor"
    assert ayrim == pytest.approx(115.8, abs=0.5)
    assert UHF_3H_US / us == pytest.approx(4.036, abs=2e-3)
    # 3H ORTALAMA ALANIN yaninda, bu dosya UZAGINDA.
    assert abs(UHF_3H_US - ORTALAMA_ALAN_US / 2.0) < 0.01
    assert abs(us - ORTALAMA_ALAN_US / 2.0) > 0.35


# --------------------------------------------------------------------------
# `slow` ISARETININ GEREKCESI -- OLCUM, DEVRALINAN ESIK DEGIL
# --------------------------------------------------------------------------
#
# Bu dosyanin butun maliyeti FIXTURE'lardadir; pytest onu `setup` olarak
# raporlar, `call` olarak degil. Depo esigi (call >= 2.0 s) bu yuzden tek
# basina yeterli olmazdi ve BURADA setup + call birlikte olculdu:
#
#   fixture             olculen sure   isteyen testler
#   tam_sayim                  0.04 s   5 test (hepsi isaretsiz)
#   fss_kisa                   1.36 s   2 test (biri isaretsiz)
#   fss_kisa_kaymali           1.22 s   1 test (`slow` -- fss_kisa ile
#                                       birlikte 2.65 s, esigin ustunde)
#   fss_uzun            43.87 / 45.76 s   3 test (hepsi `slow`)
#   binder_taramasi     62.85 / 65.32 s   5 test (hepsi `slow`)
#
# KARAR: pahali iki fixture'i isteyen HER test `slow` isaretlidir, yani
# `-m "not slow"` onlari HIC kurmaz. Hizli seritte olculen toplam 1.76 s
# (8 test) ve hicbir tekil testin setup + call'u 2.0 s'yi gecmez.
#
# `fss_kisa_kaymali`yi isteyen test bilincli olarak `slow`: tek basina 1.22 s
# ama `fss_kisa` henuz kurulmamissa ikisi birden 2.58 s'ye cikar, yani en
# KOTU haliyle esigi asar. Isaret en kotu hale gore verildi.
#
# TASIYICI IDDIA YAVAS SERIDE HAPSOLMUYOR:
#   - Onsager iddiasinin hizli serit karsiligi TAM SAYIMDIR (L = 3 x L = 4,
#     0.04 s, hata cubugu YOK) -- MC'nin ucuzlatilmis hali degil, BASKA
#     bir kanit turu.
#   - Kritik us iddiasi hizli seritte kisa zincirlerle bir kez daha kurulur
#     (1.36 s) ve bedeli olculdu: 3 sigma siniri 5.98 kat genis.
#
# BU DOSYANIN SUITE'E KATTIGI TOPLAM (olculdu): 110.04 s tam dosya
# (`-m slow` tek basina 114.03 s), bunun 1.76 s'si hizli seritte.
