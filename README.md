# qlab

Kuantum mekaniğini gerçek yöntemlerle çözen, sonucu görselleştiren ve **hangi
yaklaşımın yapıldığını her ekranda açıkça beyan eden** bir laboratuvar.

Tasarım belgesi: `../docs/superpowers/specs/2026-08-05-qlab-kuantum-cekirdegi-design.md`

## Durum

- **Modül 1A — TDSE motoru: tamam.** Tek parçacık, split-operator FFT.
  Motor çekirdeği — ızgara, potansiyeller, başlangıç paketi, propagatör ve
  observable katmanı — 1B/2B'dir ve ikisi de test edilmiştir (2B: `Grid2D`,
  `gaussian_packet_2d`, `double_slit`, `tdse_obs_2d`, çift yarık girişim
  testi). Senaryo şeması/yükleyici/fabrika, headless koşucu, görselleştirme
  veri katmanı, Qt kabuğu ve şeffaflık paneli ise yalnızca 1B'dir: `GridSpec`
  hiçbir 2B alan taşımaz, `factory.build()` sabit biçimde `Grid1D` üretir, Qt
  penceresi yalnızca 1B çizer. 2B'ye bu YAML → fabrika → koşucu/GUI hattından
  ulaşılamaz — yalnızca engine/observable modülleri doğrudan import edilerek
  kullanılabilir.
- **Modül 1B — Hartree-Fock: tamam.** Kısıtlı Hartree-Fock (RHF), keyfi s/p
  kabuğu için McMurchie-Davidson (MD) özyinelemesiyle: Boys fonksiyonu, tek-
  ve iki-elektron integralleri (örtüşme, kinetik, çekirdek-çekim, elektron
  itmesi), DIIS ile hızlandırılmış SCF, analiz katmanı (Mulliken/Löwdin
  yükleri, dipol, Mayer bağ mertebesi, HOMO/LUMO), PySCF referans testleri
  (`reference/*.json`, oturum kapsamlı fixture, koşarken PySCF'e bağımlı
  değil) ve YAML senaryo entegrasyonu (`h2_bond_scan.yaml`,
  `water_angle_scan.yaml`, headless koşucu üzerinden). MD, Obara-Saika
  yerine seçildi: dört integral tipi de aynı iki özyinelemeyi (Hermite
  genişletme katsayıları E, Hermite-Coulomb yardımcıları R) paylaşır, yani
  tek bir türetme doğrulandığında dördü birden doğrulanmış olur — gerekçe
  `engine/hf/integrals.py` modül dokstring'inde. Ölçülen sayılar aşağıdaki
  Doğrulama tablosundadır.
- **Modül 1B2 — orbital görselleştirme: tamam.** `observable/hf_obs.py`
  (moleküler orbitali kartezyen ızgarada değerlendirme, elektron yoğunluğu,
  promolekül ve yoğunluk farkı haritası), `engine/hf/analysis.py::analyse`
  (yedi analiz fonksiyonunu tek `Provenance` altında toplayan rapor),
  `viz/isosurface.py` (marching cubes ağ çıkarımı, ±faz için işaretli çift) ve
  `app/orbital_window.py` (döndürülebilir 3B pencere). Görselleştirme yığını
  **PyVista değil `pyqtgraph.opengl` + PyOpenGL**: ~150 MB yerine ~2 MB ve
  depoda zaten kullanılan tek render yığını — spec'ten bilinçli sapma.
  Kabul kriteri 3 kapandı ve bu, depodaki **ilk gerçek-ekran doğrulamasıdır**.
- **Modül 2 (Plan C) — klasik moleküler dinamik: tamam.** Kuvvetten bağımsız
  bir MD çekirdeği: `MDState` + `ForceField` protokolü, velocity-Verlet,
  üç analitik kuvvet alanı (harmonik, Morse, Lennard-Jones — periyodik kutu
  ve kaydırılmış kesmeyle), iki termostat (Langevin BAOAB, Nosé-Hoover +
  zincir), sıcaklık/hız-dağılımı gözlemleri, YAML senaryo katmanı ve headless
  koşucu (`h2_morse.yaml`, `argon_nvt.yaml`), zaman serisi çizim yükü.
  Ölçülen sayılar aşağıdaki Doğrulama tablosundadır.
- **Modül 2D (Plan D) — ab initio moleküler dinamik: tamam.** Hartree-Fock
  enerjisinin sayısal gradyanı **dördüncü kuvvet alanı** olarak takıldı
  (`HartreeFockGradient`); çekirdekler artık Born-Oppenheimer yüzeyinde
  hareket ediyor ve modül **dairesel olmayan** bir fiziksel çıpa kazandı —
  titreşim frekansı yüzeyin kendisinden, iki bağımsız yoldan çıkıyor.
  Mimari sözü tutuldu ve git ile ölçüldü: `integrator.py`, `thermostats.py`
  ve `md_obs.py` üçlüsünde **0 satır** değişti; `factory.py` ise değişmek
  zorundaydı (dağıtım dalı + HF'ye özgü şeffaflık beyanı). Ölçülen sayılar
  aşağıdaki Doğrulama tablosundadır.
  **Kendi planına bırakılan** zamana bağlı yoğunluk farkı filmi Modül 2E'de
  yapıldı (aşağıda).
- **Modül 2E — zamana bağlı yoğunluk farkı: tamam.** Bağ orta noktasındaki
  yoğunluk farkı (`bond_midpoint_difference`) ve kare dizisi yükü
  (`viz/density_frames.py`). Modülün asıl kazancı bir kod değil bir **eleme**:
  en doğal görünen aday `max|fark|` ölçülerek **elendi** — bağ zayıflarken
  ters yöne gidiyor, değeri ızgaraya bağlı, ve asimetrik bağda orta noktayla
  işareti bile zıt. Kalan gözlemlenebilir çözünürlükten de kutu boyutundan da
  bağımsız (0.043821, dört kutuda altı hane aynı) ve kopuş boyunca sayısal
  sıfıra tek yönlü oturuyor. Yük **alan küpü taşımaz** (60×–269× tasarruf) ve
  bu test edilir. **Filmden enerji okunmaz:** RHF'in ayrışma enerjisi 14.875 eV
  (deneysel 4.75) ile felakettir, yoğunluğu ise doğrudur. Ölçülen sayılar
  aşağıdaki Doğrulama tablosundadır.
- **Modül 3F — tight-binding çekirdeği: tamam.** Örgü/komşu kurulumu,
  `H(k)`, bant yapısı, elektron sayımı, bant yükü ve `kind: tb` senaryosu.
  Modülün asıl çıktısı bir sayı değil bir **hüküm**: yol haritasının
  "bakır/silisyum bant aralığı" hedefinin ilk yarısı ölçümle **geçersiz
  kılındı** — TB'de bant aralığının *değeri* bir tahmin değil, girilen fit
  parametresinin geri okunmasıdır (silisyumda 1.17 eV için gereken tek
  parametre 5.376670 eV; aynı kadran Ge'yi ve V=7.0'da metali de veriyor).
  Geriye kalan gerçek: **iletken yarısı sayımdan** gelir ve parametrelerden
  bağımsızdır (tek elektron → 300/300 metal), **yalıtkan yarısı fit'ten**
  gelir (çift sayım yalıtkanlığı garanti etmez: 1B %67, 2B %34, 3B %20 metal).
  Bant kenarları ızgaradan **okunmuyor** — ölçüldü, ızgarayı sıklaştırmak
  hatayı monoton küçültmüyor. Parametreleri qlab'ın kendi Hartree-Fock'undan
  türetmek Plan G'ye bırakıldı. Ölçülen sayılar aşağıdaki Doğrulama
  tablosundadır.

## Ne iddia etmiyor

Bu yazılım "tam kuantum, tamamen gerçekçi" bir simülatör değildir. Çok-elektronlu
Schrödinger denklemi 3N boyutludur; tam çözüm ~20 elektronda tıkanır. Her yöntem
bir yaklaşımdır. Buradaki duruş yaklaşımı gizlemek değil, ölçüp ekranda göstermektir.

Hartree-Fock (RHF) de bir yaklaşımdır, TDSE motoru kadar. Yaklaşımları: Born-
Oppenheimer (çekirdekler sabit), tek Slater determinantı, sonlu baz seti
(STO-3G/6-31G), kapalı kabuk (çift işgal). Korelasyon enerjisi eksiktir — bu
RHF'nin bilinen temel eksiğidir — ama bu depoda ölçülmüş bir korelasyon-enerjisi
referansı (örneğin bir CI/CCSD karşılaştırması) YOKTUR, dolayısıyla buraya bir
sayı yazılmıyor.

RHF ayrışmayı doğru tanımlayamaz, ve bu ölçülmüştür: H₂'yi 0.74 Å'dan 8.0 Å'ya
ayırmanın "bağlanma enerjisi" 14.6350 eV çıkar — deneysel ayrışma enerjisi
4.75 eV'nin yaklaşık üç katı (bkz. Doğrulama tablosu). Bunun nedeni eksik
korelasyon DEĞİLDİR; kısıtlı tek determinant H₂'yi iki nötr atoma ayrıştıramaz,
ayrışma limitinde %50 iyonik (H⁻ H⁺) kalır, bu yüzden E(sonsuz) olması
gerekenden çok yüksektir ve fazlalık doğrudan bağlanma enerjisine eklenir —
eksik korelasyon ters yönde çalışır ama bu hatanın yanında küçük kalır. Aynı
kusur ikinci, bağımsız bir gözlemlenebilirde de görülür: H₂ için Mayer bağ
mertebesi her ayrım mesafesinde tam 1.0'dir (çekirdekler ne kadar ayrılırsa
ayrılsın RHF bağ orbitalini çift işgal etmeye zorlar).

Çok büyük ayrımlarda (H₂, ~10-12 Å üstü) RHF çözümü sayısal olarak kırılır:
10 Å'da enerji asimptota uyar, 12 Å'da kopar. İntegraller sağlamdır (kontrol
edildi); kırılan adım özdeğer çözümüdür — iki atomik orbital arasındaki eşleşme
yuvarlama farkının altına düşünce `eigh` simetrik yerine yerelleşmiş, çok daha
yüksek enerjili bir çözüm seçer. Bu çözüm gerçek bir durağan noktadır
(`converged=True` yalan değildir) ama bir **eyer noktasıdır**. qlab bunu artık
iç (RHF→RHF) kararlılık analiziyle yakalar: yakınsamadan sonra tekil orbital
Hessian'ının özdeğerleri hesaplanır ve sonuç şeffaflık panelinde
"Kararlılık" satırı olarak basılır (H₂ 0.74 Å: +4.52, 10 Å: +2.89, **12 Å:
−2.92 → eyer noktası**). Analiz sonucu DÜZELTMEZ, yalnızca sessizliği
kaldırır — kırılma hâlâ oradadır ve kararsız bir nokta hata yükseltmez, çünkü
kullanıcı o çözümü meşru olarak isteyebilir. Sınanmayan: dış kararsızlıklar
(RHF→UHF, triplet, kompleks orbital) — "kararlı" hükmü çözümün küresel minimum
olduğunu söylemez.

Görselleştirme ızgarası çekirdek orbitallerini ÇÖZMEZ, ve bu ölçülmüştür:
su/STO-3G'de HOMO'nun ızgara üzerindeki normu 1.00000000 çıkarken çekirdek
1s orbitalininki **0.98760196**'dır — analitik değeri 1 olan bir büyüklükte
%1.24 eksik. Aynı orbitalin ızgaradaki maksimum genliği de yakınsamamıştır
(n=60'ta 4.9041, n=80'de 5.9271). Çekirdeğe yakın gösterilen her yüzey bu
yüzden niteldir; oradan sayı okunmaz.

Yoğunluk farkı haritasının promolekül referansı **gerçek taban durum
değildir**. RHF serbest atomları genel olarak çözemez (tek elektronlu H açık
kabuktur, oksijenin taban durumu üçlüdür), bu yüzden referans küresel
ortalamalı, kesirli işgalli bir serbest atomdur (SAD tipi: simetri blokları
ayrı köşegenleştirilir, kısmi dolu kabuk üç kartezyen yöne eşit bölünür —
oksijende 2p başına 4/3 elektron). Bu bir seçimdir ve `Provenance`'ta
yazılıdır; bağımsız bir dış referansı YOKTUR, çünkü karşılaştırılacak "doğru"
serbest atomu aynı yöntem üretemez.

Orbital penceresindeki **±faz renkleri mutlak bir anlam taşımaz**. Bir
özvektörün global işareti keyfidir; başka bir kütüphane aynı orbitali ters
işaretle verse kırmızı ile mavi yer değiştirirdi ve hiçbir fizik değişmezdi.
İki renk yalnızca aynı orbitalin kendi içindeki bağıl fazı gösterir. Aynı
şekilde izoyüzey seviyesi hesabın bir sonucu değil bakanın bir seçimidir:
aynı su HOMO'su 0.02'de 2850, 0.50'de 114 tepe verir. Pencere üçünü de
şeffaflık panelinde beyan eder.

Qt penceresi bu depoda **bir kez** gerçek ekranda açılıp gözle doğrulandı:
orbital penceresi (su/STO-3G HOMO), Windows platformunda, kabul kriteri 3
kapsamında — ayrıntı "Kabul kriterleri durumu" tablosunda. Onun DIŞINDA
projedeki her GUI testi `QT_QPA_PLATFORM=offscreen` ile ekransız çalışır
(bkz. `tests/test_transparency.py`, `tests/test_orbital_window.py`) — widget
durumu programatik olarak okunur, çizimin kendisi doğrulanmaz. TDSE penceresi
hâlâ hiçbir gerçek ekranda görülmedi.

## Kurulum

```bash
python -m pip install -e ".[dev]"
```

Orbital penceresi `pyqtgraph.opengl` üzerinden **PyOpenGL**'e bağlıdır; ikisi de
bağımlılık listesindedir. Eksikse `tests/test_orbital_window.py` sessizce
atlamaz, KIRMIZI olur — kurulu olmayan bir bağımlılık bozuk bir ortamdır,
atlanacak bir koşul değil.

## Kullanım

GUI olmadan senaryo koştur:

```bash
python -m qlab.run scenarios/tunneling.yaml -o out.npz
```

Klasik moleküler dinamik (Modül 2) — H₂'nin Morse potansiyelinde titreşimi
ve 108 argon atomunun Langevin termostatıyla dengelenmesi:

```bash
python -m qlab.run scenarios/h2_morse.yaml -o h2md.npz
```

```bash
python -m qlab.run scenarios/argon_nvt.yaml -o argon.npz
```

Çıktı `.npz`'si `times_fs`, `positions`, `velocities`, `kinetic`, `potential`,
`total` ve `temperature_kelvin` taşır; şeffaflık paneli koşunun yöntemini,
yaklaşımlarını ve enerji sürüklenmesini stdout'a basar.

Ab initio MD (Modül 2D) — H₂ Hartree-Fock yüzeyinde titreşiyor, kuvvetler
her adımda SCF ile yeniden çözülüyor (adım başına 7 SCF, ~8.5 ms):

```bash
python -m qlab.run scenarios/h2_aimd.yaml -o h2aimd.npz
```

Etkileşimli TDSE penceresi:

```bash
python -m qlab.app.main scenarios/tunneling.yaml
```

Döndürülebilir orbital penceresi (RHF koşar, HOMO ile açılır):

```bash
python -m qlab.app.main --molecule "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587" --basis sto-3g
```

İki giriş yolu birbirini dışlar; ikisi birden ya da hiçbiri verilmezse komut
hangisini kastettiğinizi tahmin etmeden hata verir.

Testler:

```bash
pytest tests/ -v
pytest tests/ -v -m slow
```

`pyproject.toml` bir `addopts` taşımaz, yani `pytest tests/` yavaş işaretli
testleri de KOŞAR — ikinci komut ayrı bir küme değil, birincinin alt kümesidir.

Bu makinede ölçüldü (tek oturum, dört ayrı koşu):

| Koşu | Test | Süre |
|---|---|---|
| tam süite | 801 | 239.62 s |
| `-m "not slow"` | 776 | 57.64 s |
| yalnız `-m slow` | 25 | 183.15 s |
| yalnız Modül 3F (`tests/tb/`) | 80 | 4.19 s |
| yalnız B2'nin üç dosyası (`test_orbital_window`, `test_isosurface`, `test_hf_obs`) | 165 | 29.84 s |
| yalnız Modül 2 + 2D + 2E (`tests/md/`) | 100 | 69.23 s |
| aynısı, `-m "not slow"` | 93 | 7.07 s |

Yavaş alt küme (25 test, 183.15 s) toplam sürenin **~%76'sını** yiyor. Mutlak
saniyeler bu makinede eşzamanlı yüke duyarlıdır — aynı tam süite ayrı
koşularda 215.53 s ile 258.73 s arasında ölçüldü; aşağıdaki yapısal gözlemler
ise duyarlı değildir.

**Modül 3F de hiç yavaş test eklemedi** — eklenen **84 testin 84'ü de**
hızlı şeritte (692 → 776) ve yavaş şerit 25'te sabit kaldı. En yavaş yeni test
0.80 s; dört ayrı yerde plan `@pytest.mark.slow` öneriyordu, dördü de ölçümle
kaldırıldı. Bir kez de tersi oldu: 2.90 s süren bir test referansı ızgaradan
**algebraya** çevrilerek 0.32 s'ye indi — yavaş bir testin çaresi her zaman
markerlemek değildir.

**Modül 2E hiç yavaş test eklemedi.** Plan E'nin planı üç testi
`@pytest.mark.slow` işaretliyordu; ölçüldü ve kaldırıldı — en yavaşı 0.35 s,
yani hepsi "saniye altı çıpa" kuralının içine düşüyor. Sonuç: yavaş şerit
25 testte sabit kaldı, eklenen 17 testin **17'si de** hızlı şeritte.

**Yavaş testlerin dağılımı (ölçüldü):** `hf/test_physical.py` ×5,
`hf/test_stability.py` ×5, `tdse/test_double_slit.py` ×4, `md/test_aimd.py`
×3, `md/test_maxwell_boltzmann.py` ×2, `md/test_thermostats.py` ×2,
`tdse/test_analytic.py` ×2, `tdse/test_tunneling.py` ×1,
`test_scenario.py` ×1.

**Süreyi kim yiyor:** en yavaş beş test tek başına ~169 s ediyor ve
**dördü MD değil TDSE**: `test_particle_in_box_ground_state_via_engine`
44.56 s, üç çift-yarık testi ~28.6 s'er; Nosé-Hoover ergodiklik fixture'ı
38.94 s ile ikinci sırada. Yani modül eklemek süreyi yiyen şey olmadı —
Plan D'nin **üç** yavaş testi ve tüm hızlı testleri toplam ~3 s ekledi.

**Düzeltme — "B2 süreye neredeyse hiçbir şey eklemedi" iddiası yanlıştı.**
Ölçüldü: B2'nin üç dosyasında **sıfır** yavaş test var, yani maliyeti tamamen
hızlı kısma düşüyor ve orada küçük değil: hızlı kısım 53.62 s, bunun
29.84 s'i (~%56) B2'nin üç dosyası. Eski iddia toplam sürenin sabit kalmasına
dayanıyordu (127.3 → 127.96 s), ama o sabitlik bir gürültü iptaliydi:
B2'nin hiç dokunmadığı yavaş alt küme aynı iki ölçüm arasında
112.6 → 106.67 s "hızlanmıştı".

**İlan edilmiş bir test bütçesi YOKTUR.** Plan D'nin ilk taslağı "`tests/md`
≤ 50 s, tam süite ≤ 240 s" diye bir kısıt varsayıyordu; `README.md` ve
`pyproject.toml` tarandı, öyle bir sınır **hiç tanımlı değil** — o sayılar
bu tablonun *ölçümleriydi* ve yanlışlıkla limite çevrilmişti. Yerine geçen
kural ölçüme dayanıyor: yeni testler `@pytest.mark.slow` şeridine girer,
varsayılan şeride yalnız saniye altı çıpalar konur; çünkü hızlı şerit
(53.62 s) günlük çalışma döngüsüdür ve korunması tam süiteden daha
değerlidir. Kural Modül 2E'de iki kez iş gördü: plan dört testi yavaş
işaretlemişti, ölçüm dördünü de hızlı şeride geri gönderdi.

## Doğrulama

Her fizik iddiası bilinen bir referansa karşı test edilir.

### Modül 1A — TDSE

Aşağıdaki sayılar bu görevde (Task 15) gerçekten çalıştırılıp ölçülmüştür —
plandaki hedef değerler değil, ölçülen değerlerdir:

| Test | Referans / tolerans | Ölçülen |
|---|---|---|
| Norm korunumu | Üniterlik garantisi, eşik 1e-8 | 8.427e-12 (headless CLI koşusu, `tunneling.yaml`, 32000 adım) |
| Kutuda parçacık, taban durum (motor seviyesi) | Sonlu-duvar transandant denklemi, tolerans %0.05 | sapma **%-0.017** (measured=0.17762085, referans=0.17765102) |
| Harmonik taban durum | E₀ = ω/2, tolerans %0.01 | pratikte tam örtüşme (measured=0.500000000002) |
| Serbest paket yayılımı | σ₀√(1+(t/2mσ₀²)²), tolerans %1 | pratikte tam örtüşme (measured=7.0710678119, analitik=7.0710678119 — Gauss paketinin yayılımı analitik olarak da tam bu formülü izler) |
| dt yakınsaması | O(Δt²), beklenen oran 4.0 | ölçülen oran **4.0001** (dt=0.02→0.01, hata 9.093e-05→2.273e-05) |
| Tünelleme oranı | Paket-integre analitik T(E), disiplin hedefi %2 | ölçülen fark **%0.07** (T_ölçülen=0.14782, T_analitik=0.14791) |
| Çift yarık girişimi | ≥3 girişim tepesi (%5 eşik) | 3 tepe (y=±9.77, 0); tek-yarık kontrolüyle ayrım oranı 2.94× |

Not: "Kutuda parçacık" satırı motor seviyesinde ölçülür (`SplitOperatorPropagator`
imajiner zamanda gerçekten evrimletilir); saf analitik formülün ikinci, bağımsız
bir sayısal yöntemle (merkezi-fark köşegenleştirme) çapraz doğrulaması ayrıca
`tests/tdse/test_analytic.py::test_particle_in_box_eigenvalues` içinde vardır
ama qlab.engine'i hiç çağırmaz — motorun kanıtı değildir.

### Modül 1B — Hartree-Fock

Aşağıdaki sayılar bu görevde (B1 Task 13) gerçekten çalıştırılıp ölçülmüştür,
`reference/hf_energies.json` içindeki `_pyscf_version` alanına göre PySCF
2.14.0'a karşı:

| Test | Referans / tolerans | Ölçülen |
|---|---|---|
| PySCF toplam enerji uyumu (9 sistem: He, H₂ @0.74/1.00/2.00 Å, LiH, HF, H₂O, NH₃, CH₄; STO-3G) | `tests/hf/test_scf.py::test_total_energy_matches_pyscf`, sınır `abs=1e-10, rel=0.0` (kabul kriteri 1e-6'dır; test ölçülen sapmanın ~7.6 katında bağlayıcı tutulur) | en kötü sapma **1.32e-11 hartree** (H₂, R=2.00 Å); su -74.9630631297 hartree (fark 4.56e-12) |
| İntegral uyumu (S, T, V, ERI; H₂ ve H₂O, STO-3G) | `tests/hf/test_integrals.py`, atol=1e-10..3e-10 | en kötü mutlak sapma **2.129e-10** (su çekirdek-çekim matrisi, [5,1]) |
| H₂ denge bağ uzunluğu | Deneysel 0.741 Å, tolerans ±%8 | Yöntemin gerçek minimumu **0.71223 Å** (sapma %3.9), E = -1.117505885116 hartree. Parabol fiti kullanan taramalar bunun biraz üstünü verir: dağıtılan senaryo **0.71348 Å** (%3.7), `test_physical`'ın 40 noktalı düzgün ızgarası **0.71558 Å** (%3.4). İkisi de denge değil, ızgara eseridir — aşağıdaki nota bakın |
| H₂ bağlanma enerjisi (RHF/STO-3G, 0.74→8.0 Å) | Deneysel ayrışma enerjisi 4.75 eV | **14.6350 eV** — RHF ayrışma kusuru yüzünden ~3 kat fazla, korelasyon eksikliği DEĞİL (bkz. "Ne iddia etmiyor") |
| H₂O denge açısı | Deneysel 104.5°, tolerans ±%8 | **100.97°**, sapma %3.4 |
| H₂O dipolü | Deneysel 1.85 D | **1.7255 D**, sapma %6.7 |
| SCF yakınsama hızlandırma (DIIS vs düz Roothaan-Hall, iterasyon sayısı) | `tests/hf/test_scf.py::test_diis_reduces_iteration_count` | H₂O 19→8, LiH 23→7, NH₃ 18→8, CH₄ 12→11 |
| Yakınsamayan SCF sessizce sayı döndürmemeli | `tests/hf/test_scf.py::test_non_convergence_raises_rather_than_returning` | `SCFConvergenceError` fırlatılır, iterasyon/enerji geçmişi hatanın içinde taşınır |
| Referans koruması gerçekten korur mu | `reference/hf_energies.json` geçici kaldırılıp `pytest tests/hf/` koşuldu | **kırmızı**, atlanmadı: **92 error + 2 failed, 234 passed, 0 skipped** (toplam 328). 92 hatanın hepsi `pytest.fail`: "Referans dosyasi bulunamadi"; 2 failed, alt süreçte pytest koşan yol-sırası testidir ve o alt süreç de aynı sebeple kırmızıdır |

Not (bağ uzunluğu — neden üç sayı): Bu README bir süre yalnızca 0.7156 Å'yı
bildirdi ve onu "denge bağ uzunluğu — ölçülen" diye sundu. İki ayrı hata
vardı. Birincisi: o sayı bir taramanın çıktısıdır, yöntemin dengesi değil.
İkincisi: kabul kriteri 2 kanıt olarak `h2_bond_scan.yaml` komutunu gösterip
o komutun **üretmediği** bir sayıyı bildiriyordu — 0.7156 Å, testteki
`linspace(0.4, 3.0, 40)` ızgarasından (adım 0.0667 Å) geliyordu.

Parabol fiti, minimumun etrafındaki üç ızgara noktasına oturtulur ve kaba
ızgarada minimumu yukarı kaydırır. Ölçülen üç sayı:

| Kaynak | Minimum civarındaki örnekleme | Fit | Gerçek minimumdan yanlılık |
|---|---|---|---|
| Brent (ızgarasız arama) | — | **0.7122299 Å** | 0 (tanım gereği) |
| `scenarios/h2_bond_scan.yaml` | 0.65 / 0.70 / 0.74 Å | **0.7134824 Å** | **+0.0012525 Å** |
| `test_h2_potential_curve_has_a_minimum_near_experiment` | 0.0667 Å düzgün adım | **0.7155781 Å** | **+0.0033483 Å** |

Sıralama iddiayı ölçüyor: minimum civarında daha sık örnekleyen ızgara daha
küçük yanlılık veriyor — yani eser gerçekten ızgaradan geliyor, yöntemden
değil. Bu sıralamanın kendisi
`test_shipped_h2_scan_fit_is_grid_biased_but_less_than_the_uniform_grid`
ile kilitli.

Fark akademik değil, çünkü kuvvet üç büyüklük mertebesi farklı: sayısal
|dE/dr| gerçek minimumda **8.7e-09 hartree/bohr** (yani sıfır, gürültü
seviyesi), 0.7156 Å'da **3.589e-03 hartree/bohr**. Modül 2'nin moleküler
dinamik testleri "denge" diye kaba fite demirlerse, molekülü dengede
sandıkları noktada iterken bulurlar. Gerçek minimum ve kaba fit
`test_coarse_scan_fit_sits_above_the_true_sto3g_minimum` ile ayrı ayrı
kilitli: hangisinin hangisi olduğu artık nesirle değil testle korunuyor.

Not: İntegral satırındaki 2.129e-10'luk artık özyinelemenin kendisinden değil,
referansı üreten PySCF derlemesinin bohr sabitinden (CODATA 2010,
0.52917721092 Å) qlab'ın kullandığı sabitten (CODATA 2018, 0.529177210903 Å)
gelir; bu fark çekirdek konumlarına geçer ve oradan her integrale kendi
çekirdek-konumu türevi üzerinden sızar. PySCF'in kendi bohr sabitiyle kurulan
aynı geometride dört tensör de rtol=0, atol=1e-12 ile eşleşir
(`test_water_matches_pyscf_to_machine_precision_at_pyscf_geometry`) — bu,
özyinelemeyi kilitleyen asıl kapıdır; yukarıdaki 1e-10 seviyesi birim-sabiti
farkını ölçer, integral hassasiyetini değil.

### Modül 1B2 — orbital görselleştirme

Bu tablodaki sayılar bu görevde (B2 Task 10) gerçekten koşturulup ölçülmüştür.
Modül 1B'nin aksine burada **bağımsız bir dış referans yoktur**: doğrulama
fikri, ızgara üzerinde sayısal olarak hesaplanan büyüklüğün motorun kendi
ANALİTİK sonucuna oturmasıdır. Bu, yeni kodu Modül 1B'nin PySCF'e karşı
doğrulanmış motoruna zincirler; "PySCF'ten daha doğru" gibi bir iddia
taşımaz.

| Test | Referans / tolerans | Ölçülen |
|---|---|---|
| Izgara örtüşmesi ↔ analitik `overlap_matrix` (H₂/STO-3G, padding 7 bohr, n=80, `np.einsum`) | Motorun kendi analitik integrali | en kötü fark **2.700e-09** |
| Elektron sayısı: ∫ρ dV (H₂/STO-3G, padding 6 bohr, n=120) | Tam olarak 2 | **1.9999993405** (kalan kutu kuyruğunun kesilmesinden) |
| Su HOMO'sunun ızgara normu | Analitik 1 | **1.00000000** |
| Su çekirdek 1s orbitalinin ızgara normu | Analitik 1 | **0.98760196** — %1.24 eksik; ızgaranın çekirdek tepesini çözemediğinin sayısal kanıtı |
| Yoğunluk farkı sıfıra integre oluyor mu (H₂) | Molekül ve promolekül aynı elektron sayısını taşır | **−1.411e-08** |
| **Kovalent bağda elektron birikimi** (H₂, bağ bölgesi \|x\|<1, \|y\|<1, 0.2<z<1.2 bohr) | Yön iddiası: çekirdekler arasında ARTMALI | bağ bölgesinde **+0.055005 elektron**, dış bölgede (z<−1.0 veya z>2.4) **−0.027068 elektron** (ikisi de bölge üzerinden İNTEGRE); haritanın ham tepe değeri ayrı bir büyüklüktür: **+0.042281 e/bohr³** (z=+0.643 bohr), ham dip **−0.064390 e/bohr³** |
| **Polar bağda yük transferi** (HF, r=0.917 Å, orta düzlem z=0.8664 bohr) | Yön iddiası: elektronegatif atoma GİTMELİ | H yarısı **−0.1841**, F yarısı **+0.1841** — üç ayrı ızgarada ±%0.2 |
| İzoyüzey yarıçapı (analitik küre `x²+y²+z²`, level=1.0, bohr) | Yarıçap 1.0 | ort **0.999215**, std 0.000363; indeks→bohr dönüşümü atlanırsa aynı büyüklük 35.600985 çıkar |
| İzoyüzey ağının yön tutarlılığı | Her iki fazın normali DIŞARI bakmalı | dışarı-bakan üçgen oranı **1.000 / 1.000** |

"Bağ", bu tabloda hiçbir yerde koda yazılmamıştır. `+0.055005` ve `0.184`
elektron, iki ayrı hesabın (molekül ve etkileşmeyen serbest atomlar) farkından
OKUNMUŞTUR. Elektronegatiflik de öyle: HF'nin hangi ucuna elektron gittiği
hiçbir tabloda tanımlı değildir, haritadan çıkar.

**İki birim karışmasın.** Bölge sayıları (`+0.055005`, `−0.027068`) bölge
üzerinden integre edilmiş ELEKTRON sayılarıdır (`values[maske].sum() *
volume_element`); `+0.042281` ve `−0.064390` ise haritanın tek bir noktadaki
HAM yoğunluk farkıdır (e/bohr³). Bunlar aynı büyüklüğün iki okuması değildir:
ölçüldü, tepe noktasının kendi hücresi üzerinden integrali 4.841e-05
elektrondur — bağ bölgesi sayısının ~1/1136'sı. Tablo bir süre üçünü aynı
hücrede, birim ayrımı yapmadan yan yana koyuyordu.

**Izgara örtüşmesi satırının kesinliği hakkında.** Orada `np.einsum` yolunun
yazılı olması gereksiz bir ayrıntı değil: bu büyüklükteki artıklar float64
toplama SIRASINA duyarlıdır. Ölçüldü — daha ince bir ızgarada (padding 8 bohr,
n=120) aynı hesap `np.einsum` ile 1.669e-11, `np.tensordot` ile 1.610e-11
verir; **%3.6 yayılım**, yani 4. anlamlı hane indirgeme yolunun eseridir.
Tablodaki satır bu yüzden deponun kendi testinin
(`test_izgara_ortusmesi_analitik_ortusmeye_oturuyor`) gerçekten koştuğu
ızgarayı ve kod yolunu bildirir; o ızgarada iki yol 5 haneye kadar örtüşür
(2.699635e-09 / 2.699516e-09).

### Modül 2 — klasik moleküler dinamik

Aşağıdaki sayılar bu görevde gerçekten çalıştırılıp ölçülmüştür. Modülün
iddiası tek cümleyle: **integratör kuvvetin nereden geldiğini bilmez** —
tek sözleşme `(konumlar) -> (enerji, kuvvetler)`, ve harmonik/Morse/
Lennard-Jones onun üç uygulamasıdır.

| Test | Referans / sınır | Ölçülen |
|---|---|---|
| Periyot analitiğe oturuyor | T = 2π√(m/k), sınır \|T/T_analitik − 1\| < 1e-3, N ≥ 200 şart | bağıl hata **4.11e-05**, kapalı form −π²/(6N²) ile 4 hanede uyuşuyor |
| Enerji SÜRÜKLENMİYOR | blok-ortalama baş-son farkı < 1e-6 (N=200, 1e5 adım, save_every=10) | **2.32e-10**, marj 4305× |
| Enerji SALINIMI kapalı forma uyuyor | tepe-tepe/E₀ = π²/N², sınır \|oran − 1\| < 0.05 | oran **1.0000** (üç N değerinde) |
| Global hata ikinci mertebeden | dt yarılanınca oran 4.0 ± %35, beş dt | **4.0018 → 4.0000** |
| Fiziksel çıpa — H₂ titreşimi | Deneysel ω_e = 4401.21 cm⁻¹, sınır ±5.0 | **4401.39 cm⁻¹** (ölçülen periyottan; ayrıklaştırma yanlılığı +π²/6N²) |
| Anharmoniklik: ω(E) = ω₀√(1−E/Dₑ) | sınır 2e-3, 800 adım/periyot | E/Dₑ=0.10 **0.948686** (marj 590×), 0.50 **0.707120** (155×), 0.90 **0.316508** (7.1×) |
| Morse dış dönüş noktası | analitik 4.287169 bohr, bant 4.2–4.4 | **4.280174** |
| LJ analitik çıpaları | r_min = 2^(1/6)σ, V(r_min) = −ε | **7.2225016416 bohr**, oran **1.00000000000000** |
| LJ kesme sürekliliği | sınır 1e-9 (prob 1e-6) | **2.299439e-12**, marj 435× |
| Langevin hedef sıcaklığa oturuyor | 300 K, ±%2 (400 replika, seed=7) | **297.785 K** (−%0.738), marj 2.71× |
| BAOAB sürtünmesiz limitte velocity-Verlet | γ=1e-14, 40 adım, sınır 1e-5 bohr | **1.08e-07** |
| Nosé-Hoover ergodiklik kusuru (1 dof) | düz NH < −4 %, zincirli \|·\| < 3 % | düz **−8.18 %**, zincirli **+0.15 %** (40 tohum) |
| Kusur KONUMDA, sıcaklıkta değil | \|düz ⟨v²⟩ sapması\| < 1 % | **+0.019 %** (marj 8.8×) |
| Maxwell-Boltzmann kendiliğinden çıkıyor | R4 = 5/3, 4σ; artı ⟨T⟩ = 87.2 K ±%10 | R4 sınır içinde; ⟨T⟩ **87.295 ± 1.763 K** (16 tohum) |
| MD senaryosu CLI'dan uçtan uca | `python -m qlab.run scenarios/h2_morse.yaml -o md.npz` | çıkış 0, 1001 kayıt, son an **75.78917958522622 fs** |

**Bu tablodaki her satır bir mutasyon sınavından geçti.** velocity-Verlet'in
yedi mutantı (forward Euler, simplektik Euler, yarım tekme eksik, a=F,
kuvvet işareti ters, a=F/m², bayat kuvvet) tek tek koşuldu ve **yedisi de**
en az bir testi kırmızı yapıyor.

**Kütle boru hattı — modülün en tehlikeli hatası.** Atomik birimde kütlenin
birimi elektron kütlesidir; "1 amu = 1" yazan kod 1822.888486× hafif bir
molekül simüle eder. Ölçüldü: periyot 313.322877 → 7.338580 a.u.
(çarpan 42.695298 = √AMU_TO_ELECTRON_MASS) ve **hiçbir şey patlamaz** —
enerji bile korunur, çünkü N=200 adım/periyotta bağıl salınım
π²/N² = 2.467401e-04'tür ve bu sayı doğru ile hatalı kütlede **birebir
aynıdır**. Yakalayan tek şey fiziksel çıpadır: 4401.39 yerine 187918.70 cm⁻¹.
Bu yüzden `MU_H2` testte literal DEĞİL, `amu_to_electron_mass`'ten geçer —
literal olduğu sürece `AMU_TO_ELECTRON_MASS = 1.0` mutantı Task 4'ün altı
testini de geçiyordu (ölçüldü).

**Nosé-Hoover'ın ergodiklik kusuru nerede DEĞİL.** 200–400 bağımsız osilatörü
tek global ζ'ya bağlayan kurulumda kusur ölçüm çözünürlüğünün altındadır
(Langevin kontrolü aynı kurulumda +0.017 ± 0.096 %). Kusur **tek** serbestlik
dereceli osilatörün özelliğidir ve testler orada kurulur. Ergodikliğin asıl
imzası sapmanın büyüklüğü bile değil, zaman ortalamalarının saçılımının
pencereyle küçülmemesidir:

| ölçüm penceresi | düz NH | zincirli (M=3) |
|---|---|---|
| 8 P | 21.203 % | 18.941 % |
| 40 P | 21.326 % | 8.824 % |
| 160 P | **21.170 %** | **4.186 %** |

20 kat pencerede düz NH'de saçılım hiç küçülmüyor; zincirlide 4.52 kat
küçülüyor (ideal √20 = 4.472).

**Dürüst olumsuzlar (ölçüldü, gizlenmiyor):**

- Maxwell-Boltzmann bir **kuvvet alanı sınavı değildir**. ε×10, ε×0.1, σ×0.9
  ve WCA kusursuz MB veriyor; dahası kuvvet alanı TAMAMEN kaldırılsa (ideal
  gaz) bile test geçiyor. Ölçek çıpası (kinetik sıcaklık) bu yüzden şart:
  termostat kaldırılınca ⟨T⟩ = **6.24e-26 K** ve test ancak o çıpayla kırmızı olur.
- Morse'un V(∞) testi `a`yı yalnızca **alttan** sınırlar; büyümesini asla
  yakalamaz (a=2.058 ve a=1e6 için sapma tam 0.0). Üst sınır V″(r_e)
  testinden gelir.
- Termostat kütlesi Q → 2Q hatası fiziksel gözlemlerin hiçbirinde eşiği
  aşmıyor; onu yalnızca `thermostat_mass`ın literal testi kırıyor.

### Modül 2D — ab initio moleküler dinamik (Plan D)

Modülün asıl kazancı burada: **çıpa artık dairesel değil.** Plan C'nin Morse
alanında `a` parametresi zaten deneysel ω_e'den türetilmişti, yani 4401.21
cm⁻¹ çıkması H2 fiziğini doğrulamıyordu — yalnız formüllerin tutarlılığını.
Plan D'de Hartree-Fock **kendi eğrisini üretir**; hiçbir deneysel sayı girdi
değildir.

| Test | Referans / sınır | Ölçülen |
|---|---|---|
| Kuvvet ↔ bağımsız referans | Richardson ekstrapolasyonu (h=1e-3/5e-4, 4. mertebe), sınır 1e-9 | hata **1.85e-12** (r_e), **2.52e-11** (1.6 bohr), **1.43e-11** (2.0 bohr) — marj 540× / 39.6× / 70× |
| Dengede kuvvet h ile sıfıra gidiyor | `\|F\|` < 5h² (sabit eşik YASAK — h'ye bağlı) | h=1e-3 → **3.06e-07**, h=1e-4 → **3.84e-09**; marj 16.3× / 13.0× |
| Newton III (toplam kuvvet sıfır) | atol 1e-9 | **tam 0.000e+00** (öteleme kısayolu sayesinde) |
| Adım maliyeti | sözleşme 6(N−1)+1 SCF | H2'de **7 SCF**, adım **8.5 ms** (tam alan 13 SCF / 15.9 ms olurdu) |
| BO yüzeyinde enerji SALINIMI | (tepe-tepe/E_titreşim)/(π²/N²) = 1.0833 ± 0.01 | **1.0836** (N=50), 1.0829–1.0836 dört konfigürasyonda; saçılım 0.0007, marj 14× |
| **ω — YOL A (statik)** | enerji eğrisinin ikinci türevi, kuvvet KULLANILMAZ | **5481.25336 cm⁻¹** |
| **ω — YOL B (dinamik)** | MD periyodu, ikinci türev ALINMAZ | **5481.22946 cm⁻¹** |
| **İki yolun uyuşması** | \|ω_B/ω_A − 1\| < 1e-4 | **−4.36e-06** — milyonda 4.4; marj 23× |
| Geometri kapısı | R > 21.1 bohr'da hata fırlatır | E(21.10) = −0.5695574096 → E(21.20) = −0.2057275665, **ΔE = +0.36383 hartree** |
| MD senaryosu CLI'dan uçtan uca | `python -m qlab.run scenarios/h2_aimd.yaml -o h2aimd.npz` | çıkış 0, `energy_drift=8.78e-08`, panelde `gradient_step_bohr=1e-05` |

**İki yolun bağımsızlığı iddia değil, ölçüm.** Kuvvetleri ×1.001 ile bozan
bir mutant çalıştırıldı:

| | bozulmamış | kuvvet ×1.001 |
|---|---|---|
| ω_A (statik) | 5481.25336 | **5481.25336** — birebir aynı |
| ω_B (dinamik) | 5481.22946 | 5483.96942 |
| ω_B/ω_A − 1 | −4.36e-06 | **+4.96e-04** → test kırmızı |

ω_A gradyan kodundan **yapısal olarak** etkilenemiyor (yalnız enerjileri
kullanır), ω_B ise ikinci türevi hiç almaz. İkisinin aynı hesap olması
imkânsızdır. Gradyan adımı değiştirildiğinde de aynı desen: h = 5e-5 / 1e-3 /
1e-2 için ω_A üçünde de 5481.25336, ω_B ise 5481.22946 / 5481.22952 /
5481.23607.

**İki düzeltme olmadan bu uyuşma elde edilemez** ve ikisi de ölçüldü.
(a) *Anharmoniklik:* sonlu genlikte frekans düşer — A=0.01 → 5481.24125,
0.02 → 5480.56669, 0.04 → 5477.93574 cm⁻¹. Tek genlikle karşılaştırmak
0.687 cm⁻¹ sahte fark üretir; A² → 0 ekstrapolasyonu şart.
(b) *velocity-Verlet dispersiyonu:* ayrık integratörün frekansı sürekli
olandan yüksektir, Ω = (2/dt)·arcsin(ω dt/2); tersi uygulanır. Etki 0.225 cm⁻¹.

**HF/STO-3G deneyselden %24.54 sapıyor — ve bu bir KUSUR değil, YÖNTEMİN
BEYANIDIR.** Ölçülen 5481.25 cm⁻¹, deneysel ω_e 4401.21 cm⁻¹. Bu depoda
deneysel sayıya yaklaşma hedefi ya da toleransı **yoktur**. Sapmanın
yöntemden geldiğinin bağımsız kanıtı baz setiyle gelir: 6-31G ile sapma
**+%5.57**'ye iner, yani hata 4.4 kat azalır — bir kodlama hatası baz
setiyle böyle davranmazdı.

**432× payda tuzağı — bir sonraki okuyucu için.** BO yüzeyinde toplam enerji
≈ −1.1175 hartree, titreşim enerjisi ise 2.582381e-03; oran **431.75**. Plan
C'nin enerji testleri paydaya `|E_toplam|` yazar. O satır buraya kopyalanırsa
her enerji ölçütü 432 kat zayıflar. Payda **E_titreşim**'dir.

**Sürüklenme bir kabul ölçütü DEĞİLDİR** ve bu ölçülerek anlaşıldı. Planın
ilk hâli `sürüklenme < 1e-8` yazıyordu; ölçülen değer **9.42e-05** (N=50) ve
**1.94e-05** (N=100) — yani dt ile ölçekleniyor, bir gürültü tabanı değil
ayrıklaştırma eseri. O sınır 9421 kat dışarıda kalırdı. Ölçüt salınım
oranıdır; salınım temiz π²/N² yasasına uyuyor (N ikiye katlanınca 4.0016).

**Dürüst olumsuzlar (ölçüldü, gizlenmiyor):**

- **Langevin 300 K'ya OTURMADI.** 200 adımda ⟨T⟩ = 218.4 K, std 106.1 K. Bu
  bir kod hatası değil: tek titreşim serbestliği olan iki atomlu sistem
  ergodik değil ve 200 adım istatistik için yetersiz. **Termostatın HF ile
  KOŞTUĞU kanıtlandı; hedef sıcaklığa OTURDUĞU kanıtlanmadı** ve iddia
  edilmiyor. (Nosé-Hoover'da ⟨T⟩ = 289.2 K ama T_max 974.3 K — aynı uyarı.)
- **"Isı bağı kırar" hikâyesi RHF'e TAŞINMAZ.** Ölçüldü: RHF/STO-3G ayrışma
  limiti −0.5709 hartree, doğrusu −1.000000. Plan C'nin Morse ayrışması nicel
  bir iddiaydı; RHF'te aynı iddia kurulamaz.
- **Varsayılan gradyan adımını hiçbir FİZİK testi korumuyor.** Ölçüldü:
  h=1e-3 yapan mutant bütün AIMD testlerini geçiyor (salınım oranı 1.0891,
  sınırın içinde) oysa gradyan hatası 1.34e-07 — 5e-5'tekinin 10⁴ katı. Hata
  sistematik ve kuvvete göre küçük olduğu için dinamik onu görmüyor. Bu
  yüzden varsayılanı **doğrudan kilitleyen** ayrı bir test var.
- **Analitik HF gradyanı (Pulay kuvvetleri) ÖLÇÜLEMEDİ** — bu depoda yok.
  "Analitik gradyan 6N+1 SCF'yi 1'e indirir" bir tahmindir ve buraya
  yazılmıyor. Aynı şekilde **sıcak başlangıç** (önceki yoğunluğu tohum verme)
  kazancı da ölçülemedi: `scf.solve` imzasında başlangıç yoğunluğu yok.
- **Maliyet kapsamı dar.** Ölçüldü: H2 1000 adım ~0.14 dk, H2O ~15.4 dk,
  CH4 ~77 dk. Test süitesine **yalnız H2** girer.

**Mimari sözü tutuldu ve git ile ölçüldü.** Plan D dördüncü kuvvet alanını
ekledi; `engine/md/integrator.py`, `engine/md/thermostats.py` ve
`observable/md_obs.py` üçlüsünde **0 satır** değişti (`git diff --stat
82c65c4..HEAD` boş). Ama iddia dardır: `engine/md/factory.py` değişmek
**zorundaydı** — dağıtım kapalı bir sayımdı ve `_YAKLASIMLAR` listesinin üç
satırından ikisi ("elektron yok", "analitik bir modeldir") HF dalında düz
yalan olurdu ve şeffaflık panelinde basılırdı.

**Ayrı plana bırakılan — zamana bağlı yoğunluk filmi.** Plan B2'nin kapanış
notu bunu istiyordu ama ölçüm onu Plan D'den ayırdı: bir karenin maliyeti
ızgarada baz değerlendirmesinden geliyor, kuvvet alanından değil; yörünge için
**sıfır SCF** gerekiyor. Yani film ab initio kuvvet gerektirmiyor. Modül 2E'de
yapıldı ve oradaki uçtan uca ölçüm bunu doğruladı: kare maliyetinin ~%64'ü
yoğunluk alanı, %31–35'i izoyüzey, SCF **%6'nın altında** ve n büyüdükçe %1'in
altına iniyor.

### Modül 2E — zamana bağlı yoğunluk farkı (Plan E)

Plan D'nin kapanışı bu filmi kapsam dışı bırakırken bir ölçüme dayanmıştı:
maliyet kuvvet alanından değil ızgaradan geliyor, dolayısıyla film ab initio
kuvvet **gerektirmiyor**. Plan E o ayrımı doğruladı ve filmin asıl sorusunu
değiştirdi: mesele kareleri üretmek değil, **hangi sayının bağın zayıflamasını
ölçtüğü**.

| Test | Referans / sınır | Ölçülen |
|---|---|---|
| Orta nokta değeri çözünürlükten bağımsız | n = 41 / 61 / 81, tolerans 1e-6 | üçünde de **0.043821** — altı hanede aynı |
| …ve kutu boyutundan da bağımsız | yarı genişlik 6.0 vs 7.0 | **0.043821** (7.0 / n=51'de de) |
| Gerilme serisi tek yönlü çöküyor | monoton azalan | 1.3459 → **0.043821**, 1.8 → **0.039467**, 2.5 → **0.028001**, 4.0 → **0.006847** |
| Kopuş serisi (sevk edilen modül) | sayısal sıfıra oturmalı | r = 1.3459 → 0.043821, 4.1649 → 0.005686, 6.2559 → **0.000541**, 8.2587 → 0.000044, 10.2504 → 0.000002, 12.24 ve ötesi → **0.000000** |
| Çift ızgara reddediliyor | `InvalidInputError` | n = 40 / 60 / 80 → 0.033640 / 0.038969 / 0.041031, yani **%23 / %11 / %6 eksik** |
| Kutu tüm kareler boyunca sabit | `origin` ve `shape` değişmez | `[-8,-8,-8]`, `(41,41,41)` |
| Kayan kütle merkezi kutuya taşınıyor | ağ ağırlık merkezi orijinde | yörünge COM'u 5 karede **1.92 bohr** kayarken ağ merkezi max \|z\| ≈ **9e-09** |
| Yük alan küpü taşımıyor | hiçbir dizi `ndim > 2`, toplam < 300 KB | 4 karede **36.792 bayt** (küple 2.205.472 → **60×**) |
| Alan kare başına bir kez hesaplanıyor | çağrı sayısı = kare sayısı | **4 / 4** (düzeltmeden önce 8) |

**Neden `max|fark|` değil.** En doğal görünen aday — fark haritasının mutlak
maksimumu — bir bağ gözlemlenebiliri **değildir**, ve bu iddia iki bağımsız
ölçümle kilitlenmiştir:

| | r = 1.3459 | r = 1.8 |
|---|---|---|
| bağ orta noktası | 0.043821 | **0.039467** (düştü) |
| `max\|fark\|` (n=61) | 0.064559 | **0.073424** (YÜKSELDİ) |

Yön yanlış olmakla kalmıyor, **değer ızgaraya bağlı**: aynı iki mesafede
`max|fark|` n = 41 / 61 / 81'de 0.061516 / 0.064559 / 0.063294 — n ile tek
yönlü bile değil. Orta nokta üç ızgarada da 0.043821'dir. Sebep fiziksel:
mutlak maksimum çekirdek yakınındaki keskin yapıdadır, bağ bölgesini hiç
görmez, ve kovaladığı çekirdek sivriliği ızgara inceldikçe sınırsız
keskinleşir. **Çözünürlüğe bağlı bir büyüklük gözlemlenebilir değildir.**

Asimetrik bağda ayrım daha da sert. HF molekülünde orta nokta **negatiftir**
(−0.194841, n = 41/61/81'in üçünde de altı hane sabit) — iyonik bağda orta
nokta zayıflar, yoğunluk F'ye kaçar — oysa `max` **+0.235735**, işaretleri
bile zıttır. Simetrik H2 bu ayrımı yapamaz: ölçüldü, orada işaretli maksimum
**tam olarak** bağ orta noktasındadır (argmax indisi (30,30,30), fark
0.00e+00, dört germe mesafesinin dördünde de).

**Izgara paritesi tuzağı.** Bağ orta noktası ancak `n` **tekse** gerçek bir
ızgara düğümüdür. Çift `n`'de en yakın düğüm okunur ve değer sessizce eksik
çıkar: n=40'ta %23, n=60'ta %11, n=80'de %6. Tuzağın asıl tehlikesi sayının
büyüklüğü değil **şekli**: hata n büyüdükçe küçüldüğü için bir *yakınsama
sorunu gibi görünür* ve insanı ızgarayı büyütmeye iter — maliyet n³ ile artar.
Oysa doğru düzeltme `n`'i tek yapmaktır ve bedavadır. Bu yüzden `symmetric_box`
çift `n`'i sessizce kabul etmez, hata fırlatır.

**Bellek kararı: ağ sakla, alan saklama.** (M,n,n,n) float64 bir küp M=100 /
n=80 için **0.410 GB** tutar; bir film için kabul edilemez. İzoyüzey ağları
ise kare başına yalnız **50–410 tepe**. Aynı 9 karenin ağ yükü 29.5 / 78.3 /
142.0 KB, alan küpleri 4.96 / 16.3 / 38.3 MB — **168× / 209× / 269×**. Alan
üretilir, ağ çıkarılır, alan atılır. Bu bir yorum değil test edilen bir
sözleşmedir: yükteki hiçbir dizi `ndim > 2` olamaz.

**Kare maliyeti** (uçtan uca, ölçüldü): n=41 → 21.6 ms, n=61 → 63.1 ms,
n=81 → 141.5 ms; 100 karelik film 2.16 / 6.31 / 14.15 s. Dağılım: yoğunluk
alanı ~%64, izoyüzey %31–35, SCF %6'dan az ve n büyüdükçe %1'in altına iner.
**Pahalı olan elektronik yapıyı çözmek değil, onu örneklemektir** — Plan D'nin
"film ab initio kuvvet gerektirmiyor" hükmünün sayısal karşılığı budur.

#### Dürüst olumsuzlar

- **RHF'in enerjisi ayrışmada felakettir, yoğunluğu doğrudur.** Ölçüldü:
  D_e = E(20 bohr) − E(r_e) = −0.570861 − (−1.117506) = **0.546645 hartree =
  14.875 eV**, deneysel değer **4.75 eV** — üç katından fazla. Doğru ayrışma
  limiti iki H atomu için −1.000000 hartree'dir; RHF 10 bohr'da −0.595971
  verir. Sebep yapısaldır: tek determinant kısıtlı dalga fonksiyonu H2'yi
  doğru ayrıştıramaz. Buna karşın *yoğunluk farkı* orta noktada doğru
  davranır ve sayısal sıfıra oturur. **Bu yüzden filmden enerji okunmaz** —
  film bir yoğunluk anlatısıdır, bir enerjetik iddia değil.
- **Video/görüntü üretimi kapsam dışıdır** ve bu bir eksiklik değil bir
  sınır. Deponun `viz/` katmanı hep *yük* üretir (`frames.py`,
  `md_frames.py`); `.mp4`/PNG yazmak yeni bir bağımlılık ve yeni bir katman
  demektir. `build_density_film` kare dizisini verir, onu neyin çizeceği
  çağıranın işidir.
- **H2 üstü ölçülmedi.** Film yalnız iki atomlu bir bağ için sınandı;
  `bond_midpoint_difference` iki atom indisi alır ve çok-bağlı bir molekülde
  hangi bağın anlatıldığı çağıranın kararıdır. H2O ve üstü için ne maliyet
  ne doğruluk ölçülmüştür.
- **Dış kararlılık analizi bu yolda koşulmaz** (`stability=False`). RHF
  çözümünün kısıtlamayı kaldırınca kararlı kalıp kalmadığı — ki gerilmiş H2'de
  kalmaz — sorulmamıştır.
- **Film ab initio kuvvet gerektirmiyor.** Buradaki yörüngeler Morse alanıyla
  üretildi ve kare başına SCF maliyeti %6'nın altında. Yani "kopuş filmi"
  görüntüsü, kuvvetlerin Hartree-Fock'tan geldiğini **kanıtlamaz**; onu
  kanıtlayan Modül 2D'nin çıpasıdır.

### Modül 3F — tight-binding çekirdeği (Plan F)

Bu modülün en önemli çıktısı bir sayı değil bir **hüküm**: yol haritasının
Modül 3 için ilan ettiği doğrulama hedefinin ilk yarısı — "bakır/silisyum bant
aralığı" — ölçümle **geçersiz kılındı**. Bir tight-binding modelinde bant
aralığının *değeri* bir tahmin değil, girilen fit parametresinin geri
okunmasıdır.

| Test | Referans / sınır | Ölçülen |
|---|---|---|
| **Gerçek-uzay ↔ Bloch çapraz kontrolü** (çekirdek çıpa) | hata/‖H‖ < 1e-13, N = 4…256 | N=12'de **8.882e-16**; hata √N ile büyüyor (log-log eğim **0.5025**), hata/‖H‖ N=4→4096 boyunca 3.165e-15 |
| Ters örgü konvansiyonu | b_i·a_j = 2π δ_ij, atol 1e-12 | eğik örgüde artık **1.178e-16** (yanlış sürümde 2.173) |
| Kesme yarıçapı, fcc | 12 komşu, 11 örgü sabitinde | **11/11** (bağıl toleranssız 5'inde **0** komşu seçiliyordu) |
| `eigh` UPLO tuzağı | üst üçgene +1000 | özdeğer değişimi **tam 0.000e+00** |
| Gauge (hücre vs atom) | özdeğerler eşit kalmalı | H elemanları farkı > 1e-3, özdeğerler 1e-14'te aynı |
| Faz konvansiyonu | H[0,1] işaretli | **−0.10362686 + 0.03983523j** (ters işarette özdeğerler **aynı**) |
| **Tek elektron sayımı → metal** | parametre-bağımsız | **300/300** (1B/2B/3B × 100 rastgele model) |
| Yapı oranı W²/μ₂ = 4z | ε ∈ (−500,500), t ∈ (−50,50) | 1B **8.000000000** (2.22e-16), 2B **16**, 3B **24** |
| Bant genişliği, tek n | kapalı form 2\|t\|(1+cos(π/n)) | fark **2.78e-17**; oranlar 0.750000 / 0.950484 / 0.991487 / 0.999758 |
| Ağırlıklar gauge-bağımsız | özvektörler değil | özvektör farkı **0.814887**, ağırlık farkı **3.331e-16** |
| Şeffaflık paneli daireselliği beyan ediyor | tam cümle | CLI çıktısında doğrulandı |
| Süite | — | **801 test yeşil**, `tests/tb/` 80 test / 4.19 s |

#### Bu modül bant aralığı TAHMİN ETMEZ

Vogl-Hjalmarson-Dow (1983) sp³s\* parametrizasyonuyla silisyum kuruldu ve
bağımsız bir uygulamayla yeniden üretildi:

| ölçüm | değer |
|---|---|
| dolaylı bant aralığı | 1.171338 eV (deneysel 1.12) |
| E_gap = 1.17 eV için gereken `V(s*a,pc)` | **5.376670 eV** — yayınlanan 5.3749, fark %0.033 |
| kazanç dE_gap/dV | −0.7561 eV/eV |
| aynı kadran | Ge'yi (0.66 eV), GaAs'ı ve V=7.0'da **metali** veriyor |
| Γ noktası özdeğerleri | analitik olarak E_s ± \|V_ss\| — **doğrudan girdiler** |

Aynı şey 1B'de doğrudan gösterildi: hedef aralık seçilip `t` ondan cebirsel
türetildiğinde çözücü hedefi geri veriyor (1.12 → 1.120000000000000, 0 ULP).

**Fit edilmeyen tek nicelik yanlış çıkıyor:** iletim bandı minimumunun konumu
**0.731066 X**, deneysel **0.85 X** — **%14.0 hata**. Model onu tahmin etmeye
çalıştı ve ıskaladı. Doğrulanabilir olan tek şey buydu ve doğrulanmadı.

#### İletken yarısı sayımdan, yalıtkan yarısı fit'ten

Modülün ölçülmüş asimetrisi budur. **Tek** elektron sayımı her zaman metal
verir — bir bandı ancak yarı doldurabilir, dolayısıyla hüküm bir hesap değil
bir **sayımdır** (300/300, iki bağımsız uygulamayla). Ama **çift** sayım
yalıtkanlığı **garanti etmez**: bant örtüşmesi yüzünden metal çıkma oranı
1B %67, 2B %34, 3B %20 ölçüldü. Silisyumun yarıiletkenliği sayımdan çıkmıyor.

#### Tautoloji-çıpa — bu depoda üçüncü kez

Ön ölçümde "çıpa" ilan edilenlerin çoğu karşıt doğrulamada düştü:

| iddia edilen çıpa | hüküm |
|---|---|
| "sayısal = analitik, 1 ULP" | **tautoloji** — `eigvalsh(1×1)` ile `H[0,0].real` arası 0 ULP; 1 ULP'in tamamı birleşme sırası |
| "bant katlanması, 0 ULP" | **tautoloji** — iki matris bit bit aynı |
| "durum sayımı" | 6 kasıtlı bozulmanın **0'ını** yakaladı |
| grafende K'de sıfır aralık "korunuyor" | **çürütüldü** — Haldane terimi A,B özdeş kalırken 0.519615242 eV açıyor |
| `E(Γ)/E(M) = 3` gibi oranlar | model kesiminin geri okunması |

Körlük doğrudan sergilendi ve testle kilitlendi: tek orbitalli zincirde
`eigvalsh` 1×1 matriste **yalnız gerçek kısmı** okur, dolayısıyla üç farklı
Hamiltoniyen aynı özdeğeri verir — doğrusu (−0.17251924+0j), hücre çarpanı
düşürülmüş olan (**−0.17251924 − 0.12299824j**, Hermityen bile değil) ve fazı
tamamen kaybetmiş olan; üçünün de `eigvalsh`'i −0.172519243044.

**Ayırıcı sınav ölçüldü:** aynı altı mutanta karşı çekirdek çıpa **6/6**,
tautoloji-çıpa **1/6** yakaladı.

> **Kural.** 1e-12 altında uyuşan iki "bağımsız yol" kanıt değil şüphedir.
> Sahte-çıpa sınavı (fiziğe ait olmayan rastgele girdi) şüpheyi *işaret eder*
> ama tek başına ayırmaz — **ayıran şey mutasyondur**: bir yolu bozunca öteki
> yakalıyor mu?

#### k-uzayı parite tuzağı — Modül 2E'nin kardeşi, ama daha kötü

Grafende ölçüldü: 601×601 ızgara Dirac noktasını yakalıyor (**2.471886e-15 eV**),
600×600 sahte bir **1.635969e-02 eV** aralık uyduruyor — ve **daha sık** olan
1000×1000 **1.698157e-02 eV**, yani 601'den **13 mertebe kötü**. Sıklaştırmak
düzeltmiyor.

1B'de iki ayrı biçimde çıkıyor. Birincisi parite: bant genişliği tek n'de
sistematik eksik, çift n'de tam. İkincisi daha sinsi — **hata monoton bile
değil**:

| n | bant minimumu hatası |
|---|---|
| 5 | **2.426e-06** |
| 7 | 1.934e-02 |
| 12 | 7.031e-03 |
| 13 | **7.370e-03** (n=12'den kötü) |
| 2001 | 1.936e-09 |
| 20001 | **3.849e-09** (on kat sık ızgara iki kat kötü) |

n=5, n=13'ten ~3000 kat iyi. Bu yüzden `band_edges` ızgarayı **hiç
kullanmıyor**, sürekli minimizasyon yapıyor — ve ölçüldü, 201×201 ızgaradan
(1.79 s) hem **daha doğru** hem 8× hızlı (0.23 s).

Parite kodda **sihirli sayıyla** sabitlenmiyor: ön ölçümde iki ajan iki farklı
ızgara konvansiyonunda iki farklı "doğru N" ölçtü (601 ve N%3==0), ikisi de
kendi konvansiyonunda doğru. Bunun yerine çağıran hangi k noktalarının ızgarada
bulunması gerektiğini söylüyor, kod **denetliyor** (`must_include`).

#### Dürüst olumsuzlar

- **`band_structure` çıktısı sıralamadır, bant kimliği değildir.** Ölçüldü:
  sıralı "bant 0"ın etkin kütlesi −6.42e-05, gerçek bandınki −8.000e-01 —
  **12457 kat** hata. Çizim için doğru, bant-çözümlü nicelikler için değil.
- **Dejenere alt uzayda bant başına orbital ağırlığı tanımlı değildir.**
  Doğrudan sergilendi: 2B'de iki bant tam dejenere (E farkı 0.0); numpy
  `[1,0]/[0,1]` döndürüyor ama 45° döndürülmüş çift de aynı ölçüde geçerli ve
  ağırlıkları `[0.5,0.5]`. Alt uzay toplamı iki tabanda da aynı.
- **`band_edges` çok boyutta bir SINIRDIR, kanıtlanmış küresel uç değer
  değil.** Ölçüldü: 50 aramada tek başlangıç 31/50 başarısız (en kötüsü bant
  genişliğinin **%51.5**'i), 5×5 çoklu başlangıç 8/50.
- **`converged` ölçütü kandırılabilir.** n vs 2n karşılaştırması "iki ızgara
  aynı sonucu veriyor" der, "sonuç doğru" demez; n=4 gibi çok kaba bir çift
  ızgara bile `True` veriyor çünkü sıfır yoğunluktan değil **pariteden** geliyor.
- **Ortogonal olmayan taban yoktur** (S = I). Bedeli ölçüldü: bant genişliğinin
  %45–65'i.
- **`viz` katmanının GUI yasağı YOKTUR** — ölçüldü, `viz/band_frames.py →
  PySide6` rc=0 ile geçiyor ve `viz/isosurface.py:55` zaten pyqtgraph import
  ediyor. "Bant yükü çizim yapmaz" bir **disiplindir, sözleşme değildir**.
  (`engine` tarafındaki `matplotlib` kaçış deliği ise bu modülde kapatıldı:
  enjeksiyon artık rc=1 veriyor ve taranan dosya 67 → 68'e çıkıyor.)
- **Büyük k'de argument reduction sessizce bozuluyor:** k + 1e12·G →
  özdeğer farkı 2.387652e-03 eV mertebesinde, hiçbir uyarı yok.
- **Bu ULP sayıları taşınabilir değildir** — tek makine, tek yığın
  (Python 3.14.7, numpy 2.5.1, scipy 1.18.0, Windows 11).

#### Ne yapılmadı

Silisyum ve bakır **sevk edilmedi** (ön ölçümde kuruldular, depoya girmediler);
spin, spin-yörünge etkileşimi ve manyetik alan yok; Peierls çarpılmasının
enerjetiği hesaplanmıyor; sıcaklık/Fermi-Dirac doluluğu yok. Ve en önemlisi:
**parametreler türetilmiyor**, dışarıdan giriyor. Onları qlab'ın kendi
Hartree-Fock'undan türetmek — daireselliği kırma hamlesinin TB karşılığı —
Plan G'ye bırakıldı ve ön ölçümü "KISMEN" dedi.

## Mimari kuralı

`engine` paketi `viz` veya `app` import **edemez** — ve GUI'nin kendisini de,
yani `PySide6` / `pyqtgraph` / `OpenGL` / `matplotlib`'i de. (`matplotlib`
Modül 3'te ölçülerek eklendi: `engine/tb/bands.py`ye enjekte edilen
`import matplotlib.pyplot` rc=0 ile **geçiyordu**, aynı dosyaya konan PySide6
kontrolü ise rc=1 veriyordu — yani araç çalışıyordu, eksik olan kapsamdı.) Aynısı `observable` için geçerli.
Katman sırası `run/app → viz → observable → engine → scenario →
provenance → types → units/errors`. Kural yorumla değil import-linter ile
korunur: `pytest tests/test_architecture.py`.

Sözleşmenin kapsamı ÖLÇÜLEREK genişletildi. Önceki hâlinde yasak listesi
yalnızca `qlab.viz` ve `qlab.app` idi ve şu altı ihlalin altısı da rc=0 ile
KAÇIYORDU: `engine → PySide6`, `engine → pyqtgraph`, `engine → OpenGL`,
`observable → PySide6`, `engine.hf.analysis → qlab.run`, `units → engine`.
Yani "engine görselleştirmeyi bilmez" diyen sözleşme, engine'in doğrudan Qt
import etmesine izin veriyordu.

`test_import_contracts_hold` tek başına bunu göremezdi: o yalnızca temiz
ağaçta rc==0 olduğunu görür ve sözleşme boşaltılsa bile yeşil kalırdı. Bu
yüzden yanına `test_import_sozlesmesi_gercekten_REDDEDIYOR` eklendi — kopya
bir ağaca ihlal enjekte eder ve **adı geçen sözleşmenin BROKEN raporlandığını**
şart koşar. O "BROKEN" şartı da ölçümle geldi: yalnızca sözleşme adının
çıktıda geçmesini aramak totolojiydi, çünkü import-linter her sözleşmenin adını
her koşuda basar.

## Kabul kriterleri durumu

Spec §15'teki yedi maddenin gerçek durumu (ölçüm ayrıntıları
`.superpowers/sdd/task-15-report.md` — Modül 1A — ve
`.superpowers/sdd/b1-task-13-report.md` — Modül 1B — içindedir):

| # | Kriter | Durum |
|---|---|---|
| 1 | `pytest` tam yeşil, PySCF referans testleri dahil | **Kapandı** — **801/801 test yeşil** (`pytest tests/`, 239.62 s), PySCF referans testleri dahil. Ayrıca `reference/` geçici olarak kaldırılıp tam süite yeniden koşuldu: testler ATLANMIYOR, KIRMIZI oluyor — ölçüldü, **128 failed + 132 error + 291 passed, 0 skipped** (toplam 551, 154.20 s). Yani 260 test kırmızıya dönüyor ve hiçbiri sessizce atlanmıyor. Dikkat: bu ikinci deney süite 551 testliyken yapıldı ve o günün sayılarını taşır; yeniden koşulmadı, sayıları 801'e uydurmak ölçüm değil uydurma olurdu |
| 2 | Headless koşucu çalışır + H₂ eğrisi bağ uzunluğu ±%8 | **Kapandı** — `python -m qlab.run scenarios/h2_bond_scan.yaml -o h2_curve.npz`; taramanın parabol fiti **0.71348 Å**, deneyselden sapma %3.7, kriterin istediği ±%8 sağlanıyor. Bu sayı yöntemin denge uzunluğu DEĞİL: STO-3G'nin gerçek minimumu 0.71223 Å, fark taramanın ızgara eseridir (Modül 1B tablosunun altındaki nota bakın). Bu satır bir zamanlar 0.7156 Å diyordu — o sayı bu komuttan değil, `test_physical`'ın ayrı ızgarasından geliyordu; artık `test_shipped_h2_scan_fit_is_grid_biased_but_less_than_the_uniform_grid` senaryonun kendi sayısını kilitliyor |
| 3 | GUI'de H₂O HOMO izoyüzeyi döndürülebilir | **Kapandı — bu depoda ilk gerçek-ekran doğrulaması.** `python -m qlab.app.main --molecule "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587"` gerçek Windows platformunda açıldı; su HOMO'su (1b₁) çizildi, ±faz iki renkte, 1764/1764 tepe, çekirdekler ve şeffaflık paneli görüldü. Döndürme ölçüldü: kamera azimutu değişince kare piksel bileşenlerinin **%14.1'i**, elevasyon değişince **%7.2'si** değişiyor — sahne gerçekten 3B ve görünüm tepki veriyor. **Ne ölçülmedi:** fareyle sürükleyerek döndürme; kamera programatik değiştirildi. Kriterin "döndürülebilir" kelimesi bu yüzden yalnızca sahne tarafında kapalıdır, girdi tarafında bir insanın bakması gerekir. Doğrulamanın kendisi bir kusuru da GÖRÜNÜR KILDI: 12 satırlık beyan 140 px'lik panelde katlanın altında kalıyordu (düzeltildi, `TransparencyPanel(max_height=…)`) |
| 4 | Kaydırıcı ile canlı iletim, analitiğin ±%2'si içinde | **Kapandı** — ölçülen fark %0.07 |
| 5 | Şeffaflık paneli her hesapta dolu | **Kapandı** — TDSE tarafında pencere paneli; HF tarafında headless koşucu her senaryoda Yöntem/Yaklaşımlar/İhmal edilen/Beklenen hata/Yakınsama satırlarını basar (`h2_bond_scan.yaml` çıktısı) |
| 6 | Yakınsamayan SCF sessizce sayı döndürmez, test edilmiş | **Kapandı** — `SCFConvergenceError`, `test_non_convergence_raises_rather_than_returning` ile test edilmiştir |
| 7 | `engine`'in `viz`/`app` bağımlılığı yok, import-linter ile doğrulandı | **Kapandı** — üç sözleşme (`engine görselleştirmeyi bilmez`, `observable görselleştirmeyi ve GUI'yi bilmez`, `Katmanlı mimari`), hepsi KEPT. Sözleşmenin kendisinin diş taşıdığı da ölçülüyor: enjekte edilen 8 ihlalin 8'i adı geçen sözleşmeyi BROKEN yapıyor (bkz. "Mimari kuralı") |

**7/7 madde kapalıdır.** Kriter 3 Modül 1B2 ile kapandı; yukarıdaki tablonun
yedi satırı da "Kapandı" der.

(Bu paragraf bir süre kendi tablosuyla çelişti: "6/7 madde kapalıdır, kapanmayan
tek kriter 3'tür" diyor ve "Qt penceresi bu depoda hiçbir gerçek ekranda henüz
açılmadı" diye ekliyordu — oysa aynı belgenin üç ayrı yeri, tablonun 3. satırı
dahil, kriterin B2'de kapandığını ve pencerenin gerçek ekranda **bir kez**
açıldığını yazıyordu. B2'den kalma güncellenmemiş bir özetti.)

Kriter 3'ün kapanışı kendi sınırını taşıyor ve o sınır hâlâ geçerlidir: sahne
tarafı ölçüldü, **girdi tarafı ölçülmedi** — fareyle sürükleyerek döndürme
denenmedi, kamera programatik değiştirildi. TDSE penceresi ise hâlâ hiçbir
gerçek ekranda görülmedi; onun dışındaki tüm GUI testleri ekransız
(`QT_QPA_PLATFORM=offscreen`) koşar.
