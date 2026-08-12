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
- **Modül 2 — moleküler dinamik: planlanıyor.** Henüz başlamadı. Bu modülün
  ürettiği yoğunluk farkı haritası orada zamana bağlı hâle gelir; bağın
  kopuşu görünür olur.

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
| tam süite | 601 | 180.65 s |
| `-m slow` | 18 | 117.61 s |
| `-m "not slow"` | 583 | 51.24 s |
| yalnız B2'nin üç dosyası (`test_orbital_window`, `test_isosurface`, `test_hf_obs`) | 165 | 32.34 s |

Yavaş alt küme toplam sürenin ~%65'ini yiyor (çift yarık, kutuda-parçacık ve
HF tarama koşuları). Mutlak saniyeler bu makinede eşzamanlı yüke duyarlıdır;
aşağıdaki yapısal gözlem ise duyarlı değildir.

**Düzeltme — "B2 süreye neredeyse hiçbir şey eklemedi" iddiası yanlıştı.**
Ölçüldü: 18 yavaş testin **hepsi** B2 ÖNCESİ dosyalarda
(`tests/hf/test_physical.py` ×5, `tests/hf/test_stability.py` ×5,
`tests/tdse/test_double_slit.py` ×4, `tests/tdse/test_analytic.py` ×2,
`tests/tdse/test_tunneling.py` ×1, `tests/test_scenario.py` ×1); B2'nin üç
dosyasında **sıfır** yavaş test var. Yani B2'nin maliyeti tamamen hızlı kısma
düşüyor ve orada küçük değil:
hızlı kısım 51.24 s, bunun 32.34 s'i (~%63) B2'nin üç dosyası. Eski iddia
toplam sürenin sabit kalmasına dayanıyordu (127.3 → 127.96 s), ama o sabitlik
bir gürültü iptaliydi: B2'nin hiç dokunmadığı yavaş alt küme aynı iki ölçüm
arasında 112.6 → 106.67 s "hızlanmıştı".

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

## Mimari kuralı

`engine` paketi `viz` veya `app` import **edemez** — ve GUI'nin kendisini de,
yani `PySide6` / `pyqtgraph` / `OpenGL`'i de. Aynısı `observable` için geçerli.
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
| 1 | `pytest` tam yeşil, PySCF referans testleri dahil | **Kapandı** — **601/601 test yeşil** (`pytest tests/`, 180.65 s), PySCF referans testleri dahil. Ayrıca `reference/` geçici olarak kaldırılıp tam süite yeniden koşuldu: testler ATLANMIYOR, KIRMIZI oluyor — ölçüldü, **128 failed + 132 error + 291 passed, 0 skipped** (toplam 551, 154.20 s). Yani 260 test kırmızıya dönüyor ve hiçbiri sessizce atlanmıyor. Dikkat: bu ikinci deney süite 551 testliyken yapıldı ve o günün sayılarını taşır; yeniden koşulmadı, sayıları 601'e uydurmak ölçüm değil uydurma olurdu |
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
