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
- **Modül 1C — kısıtlanmamış Hartree-Fock (UHF): tamam.** `engine/hf/uhf.py`
  (α/β Fock, birleşik DIIS, ⟨S²⟩), `engine/hf/multistart.py` (çok başlangıçlı
  RHF/UHF arama protokolü), `solve()`'a `initial_bias`, ve
  `stability.py::triplet_orbital_hessian` ile **dış (RHF→UHF, üçlü)
  kararlılık** analizi — panel ve kayıt artık iki kanalı ayrı ayrı basıyor.
  Modül bir özellik olarak değil **ön koşul** olarak yazıldı: Plan G'nin ön
  ölçümü referans sisteminin RHF çözümünün taban durum olmadığını gösterdi
  (çekirdek spec §13'ün "kapsam dışı" maddesi bu yüzden düzeltildi). Sevk
  edilmiş kodda **iki kusur** bulundu (N₂'de DIIS 19.78 eV, gerilmiş
  geometrilerde başlangıç tahmini 0.28–0.50 Ha) ve deponun kendi PySCF
  referans kaydı `h2_200`'ün gerçek HF taban durumundan **0.153420178832
  hartree** yukarıda olduğu ölçüldü. Modül 2D'nin etkilenmediği ölçülerek
  gösterildi. Ölçülen sayılar aşağıdaki Doğrulama bölümündedir.
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
- **Modül 3H — UHF indirgemesi (downfolding): tamam.** Modül 3F'nin açıkça
  yazılmış en büyük açığı kapandı: **parametreler artık dışarıdan girmiyor**,
  qlab'ın kendi UHF'inden türetiliyor. `engine/tb/downfold.py` spin-kırık UHF
  çözümünün α/β Fock matrislerini Löwdin ile ortogonalleştirip site bazında
  okuyor ve dört sayı üretiyor (`ε̄`, `Δ`, `t₁`, `t₂`); `model_from_downfold`
  onları TB motoruna besliyor, `fit_distance_laws` mesafe yasalarını
  uyduruyor, `kind: downfold` senaryosu uçtan uca koşuyor. **RHF üzerinde
  yapılamazdı ve gerekçesi Hückel'dir:** hidrojen halkalarında RHF çoğu
  geometride HF'in taban durumu değil (4n halkalarında HOMO/LUMO tam
  dejenere; 7/7 ölçüldü, N=8 ve N=12 test edilen hiçbir geometride kararlı
  değil). Modülün asıl kazancı bir sayı değil bir **çıpa**: kademeli alanın
  çatallanma üssü üç halkada bağımsız olarak **0.496190 / 0.497600 /
  0.497811** çıkıyor (beklenti ½) ve Modül 1C'nin H₂'de bambaşka bir
  yordamla ölçtüğü 0.4970 ile aynı yere düşüyor — `d_c` triplet Hessian'ın
  kökünden, `Δ` indirgemeden gelir, iki hesap birbirini hiç kullanmaz.
  Sahte-çıpa sınavı **geçti ama DAR**: 40 fizik-dışı bazın 0'ı gerçeği
  geçemedi, buna karşılık kuyruk marjı RHF tarafının üçte biri (1.49× vs
  4.29×) ve sınav hedef kümesine koşullu. **Hızlanma İDDİA EDİLMİYOR** —
  HF/TB maliyet oranı ölçülmedi. Ölçülen sayılar ve dört dürüst olumsuz
  aşağıdaki Doğrulama bölümündedir.
- **Modül 4 — spin Monte Carlo: tamam.** `engine/spin/` (kare kafes, iki
  model, checkerboard Metropolis, Wolff kümesi), `observable/spin_obs.py`
  (Binder, χ, C, ξ ve **hata çubuğunun kendisi** — plato denetimli binning),
  `viz/spin_frames.py` ve `kind: spin` senaryosu. Modülün asıl çıktısı yine
  bir sayı değil bir **hüküm**: yol haritasının "demir Curie sıcaklığı
  1043 K" hedefi ölçümle **dairesel çıktı** — Ising tek bir enerji ölçeği
  taşır ve `T_c ∝ J`, yani 1043 K için gereken `J = 39.608394 meV` geri
  konulduğunda `T_c` **birebir 1043.0000 K** okunur. Hedef düzeltildi ve
  yerine **dört dairesel olmayan çıpa** kondu: Onsager'in `kT_c/J`'si,
  kritik üs `β/ν = 0.122941 ± 0.003225` (ortalama alanı **272 kat** marjla
  dışlar), **Mermin-Wagner** (aynı kafes, aynı örnekleyici, zıt hüküm — ve
  sebebi bir teorem), ve Modül 3H'nin mean-field üssüyle (0.4962) karşılaştırma.
  Bu modülde depoya ilk kez bir **hata çubuğu** girdi ve ilk kuralı naif
  SEM'in hiçbir yerde raporlanmamasıdır (ölçüldü: 8.81 kat küçük gösteriyor).
  **Modül hiçbir malzemenin T_c'sini iddia etmez.** Ölçülen sayılar ve
  düzeltilen dört plan iddiası aşağıdaki Doğrulama bölümündedir.

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
kullanıcı o çözümü meşru olarak isteyebilir.

**Modül 1C'de bu cümle DEĞİŞTİ.** Eskiden burada *"Sınanmayan: dış
kararsızlıklar (RHF→UHF, triplet, kompleks orbital)"* yazıyordu. **Dış
(RHF→UHF, üçlü) kanal artık sınanıyor:** `solve(stability=True)` iki Hessian'ı
da kurar, panel iki ayrı satır basar ve tarama nokta bazında ikisini de
kaydeder. Bunun neden gerektiği ölçülmüştür — deponun kendi PySCF referans
kaydı `h2_200` iç kanalda **+2.5467 ile "kararlı"** derken dış kanalda
**−1.5995** ile eyerdir ve gerçek HF taban durumu **0.153420178832 hartree
aşağıdadır** (Modül 1C bölümü). **Hâlâ sınanmayan: kompleks (reel olmayan)
orbital kararsızlığı** — ve iki kanalda da "kararlı" hükmü çözümün küresel
minimum olduğunu söylemez.

RHF'nin yukarıda anlatılan ayrışma kusuru da artık yalnız anlatılmıyor,
**ölçülüyor**: UHF aynı limitte H₂'yi iki nötr hidrojene ayırıyor (20 bohr'da
2·E_H'den sapma **4.441e-16**) ve RHF'in aynı noktada **9.85877 eV** yukarıda
kaldığı ölçülüyor. UHF de bir yaklaşımdır ve kendi kusurunu taşır: spin
özdurumu değildir (`h2_200` çözümünde ⟨S²⟩ = 0.945862, singlet için 0
olmalıydı) ve bu depoda spin projeksiyonu (PUHF) yoktur.

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

**Modül 4 hiçbir malzemenin Curie sıcaklığını iddia etmez, ve bu bir alçak
gönüllülük değil bir cebir sonucudur.** 2B Ising modeli tek bir enerji ölçeği
taşır (`J`) ve `T_c ∝ J`; Onsager'in tam çözümü oranı sabitler
(`kT_c/J = 2/ln(1+√2) = 2.269185314`). Dolayısıyla SI birimindeki her `T_c`,
girilen `J`'nin geri okunmasıdır — ölçüldü: demirin 1043 K'i için gereken
`J = 39.608394 meV` konulduğunda `T_c` **birebir 1043.0000 K** çıkar (fark
0.000e+00). Aynı kadran her malzemeyi verir (Ni 627 K → 23.810607 meV,
Co 1388 K → 52.709924 meV, Gd 292 K → 11.088831 meV). Bu modülün ölçtüğü
şeyler `kT_c/J` oranı ve kritik üslerdir; ikisi de `J`'den, kafes sabitinden
ve birim seçiminden bağımsızdır. Beyan şeffaflık panelinde kendi satırında
(`Dairesellik`) basılır.

Aynı modeldeki başka iddialar da kurulmuyor: **spin kuantum değil klasiktir**
(sıfır noktası dalgalanması ve komütasyon bağıntıları yoktur), **kafes
gerçek bir kristal değildir** (koordinasyon sayısı 4'te sabit, örgü
geometrisi girdide yok), **3B kapsam dışıdır** (3B Ising'in tam çözümü yoktur
ve çıpa Onsager yerine literatürün sayısal değerlerine dayanırdı), ve
denge-dışı süreçler (histerezis, tavlama) yazılmadı. Korelasyon uzunluğu
sevk edilen senaryoda **raporlanmaz** — gerekçesi ölçüldü ve panelde yazılı.

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

UHF indirgemesi (Modül 3H) — H₈ halkasının tight-binding parametreleri beş
mesafede qlab'ın kendi UHF'inden türetiliyor (ölçüldü, 5.5 s):

```bash
python -m qlab.run scenarios/halka_indirgeme.yaml -o indirgeme.npz
```

Çıktı `.npz`'si `distances_bohr`, `epsilon_bar`, `delta`, `hoppings`,
`s_squared`, `energy_uhf`, `staggering_residual` ve `provenance_json` taşır.
Panelin `Yakınsama` satırı burada "SCF durdu" demez — modelin varsaydığı
iki-siteli antiferromanyetik yapının her mesafede gerçekten bulunduğunu söyler.

Spin Monte Carlo (Modül 4) — 2B Ising kafesinde altı sıcaklıklık bir tarama
(ölçüldü, 0.56 s):

```bash
python -m qlab.run scenarios/spin_tarama.yaml -o spin.npz
```

Çıktı `.npz`'si `temperatures`, `magnetisation`, `magnetisation_error`,
`plateau_reached`, `tau_int`, `binder`, `susceptibility`, `heat_capacity` ve
`provenance_json` taşır. **Sıcaklıklar `kT/J` cinsindendir, kelvin değil** —
`J = k_B = 1` bir birim seçimidir. Panel iki ek satır basar: `Dairesellik`
(SI'daki `T_c` bir tahmin değildir) ve `Örnekleme` (kaç sweep, kaç ölçüm,
hangi algoritma, `τ_int`, ve binning platosunun oluşup oluşmadığı). Plato
oluşmamış sıcaklıklarda hata çubuğu **"ALT SINIR" diye işaretlenir** ve
`Yakınsama` satırı ✗ basar — bu "hesap bitmedi" demek değil, "raporladığım
belirsizliklerin hepsi belirsizliğin kendisi değil" demektir.

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

**Bu makinede, Modül 4 kapanışında ölçüldü** (tek oturum, beş ayrı koşu):

| Koşu | Test | Süre |
|---|---|---|
| tam süite | **1159** | **585.36 s** |
| `-m "not slow"` | **1088** | **107.30 s** |
| yalnız `-m slow` | **71** | **475.33 s** |
| yalnız Modül 4 (`tests/spin/`) | **186** | **187.23 s** |
| aynısı, `-m "not slow"` | **166** | **29.82 s** |

Sevk edilen son hâl ayrıca bir kez daha doğrulandı: **1159 passed /
583.70 s**. Tablodaki 585.36 s ile arasındaki %0.3, aşağıda beyan edilen
makine gürültüsünün (%16'ya kadar) çok içindedir.

Modül 4'ün senaryo/panel görevi süiteye **44 test** ve **4.54 s** ekledi;
dosyanın en yavaş çağrısı 0.95 s, yani deponun 2 s eşiğinin altında ve
dosyada **hiç `slow` işareti yok**. Modül 4'ün toplam maliyeti 1114 → 1159
(+45; biri `tests/test_scenario.py`nin keşfe dayalı parametrizasyonundan
kendiliğinden geldi).

Aşağıdaki blok **Modül 3H kapanışının** ölçümüdür ve Modül 4'ü içermez;
yeniden koşulmadı, sayılarını bugüne uydurmak ölçüm değil uydurma olurdu:

| Koşu (Modül 3H oturumu) | Test | Süre |
|---|---|---|
| tam süite | **971** | **432.22 s** |
| `-m "not slow"` | **920** | **85.90 s** |
| yalnız `-m slow` | **51** | **352.41 s** |
| yalnız Modül 3F + 3H (`tests/tb/`) | **185** | **138.55 s** |
| aynısı, `-m "not slow"` | **170** | **16.73 s** |

Sevk edilen son hâl ayrıca bir kez daha doğrulandı: **971 passed / 424.45 s**.
Tablodaki 432.22 s ile arasındaki %1.8, aşağıda beyan edilen makine
gürültüsünün içindedir; tablo tek bir oturumun beş koşusu olduğu için o
oturumun sayısıyla bırakıldı.

Aşağıdaki satırlar **864 testlik önceki oturumun** ölçümleridir ve Modül 3H'yi
içermez; yeniden koşulmadılar ve sayılarını bugüne uydurmak ölçüm değil
uydurma olurdu. (O oturumun kendi toplamları: tam süite 864 / 264.98 s,
`-m "not slow"` 828 / 67.97 s, `-m slow` 36 / 210.65 s.)

| Koşu (önceki oturum) | Test | Süre |
|---|---|---|
| yalnız Modül 1C'nin beş dosyası (`test_uhf`, `test_uhf_cipalar`, `test_rhf_sikismasi`, `test_coulson_fischer`, `test_dis_kararlilik`) | 60 | 31.46 s |
| yalnız `tests/hf/` (1B + 1B2 + 1C) | 420 | 59.65 s |
| aynısı, `-m "not slow"` | 399 | 34.68 s |
| yalnız Modül 3F (`tests/tb/`) | 80 | 4.19 s |
| yalnız B2'nin üç dosyası (`test_orbital_window`, `test_isosurface`, `test_hf_obs`) | 165 | 29.84 s |
| yalnız Modül 2 + 2D + 2E (`tests/md/`) | 100 | 69.23 s |
| aynısı, `-m "not slow"` | 93 | 7.07 s |

Yavaş alt küme toplam sürenin **~%81'ini** yiyor (Modül 4 kapanışında
475.33 / 585.36 s); Modül 3H kapanışında oran %82, ondan önce %79'du — yani
Modül 4 yavaş şeridi **oransal olarak büyütmedi** (yavaş şerit 51 → 71,
ve eklenen 20 işaretin hepsi Task 2–5'ten gelir; Task 6 hiç eklemedi).
Mutlak saniyeler bu makinede eşzamanlı yüke
duyarlıdır ve bu ölçüldü: 864 testlik oturumda aynı tam süite yalıtılmış üç
koşuda **255.42 / 264.98 / 270.09 s**, makinede başka bir ölçüm koşarken ise
**296.58 s** verdi (%16 fark). Aşağıdaki yapısal gözlemler bu duyarlılıktan
etkilenmez.

**Modül 3F hiç yavaş test eklememişti** — eklenen **84 testin 84'ü de** hızlı
şeritte (692 → 776) ve yavaş şerit 25'te sabit kalmıştı. En yavaş yeni test
0.80 s; dört ayrı yerde plan `@pytest.mark.slow` öneriyordu, dördü de ölçümle
kaldırıldı. Bir kez de tersi oldu: 2.90 s süren bir test referansı ızgaradan
**algebraya** çevrilerek 0.32 s'ye indi — yavaş bir testin çaresi her zaman
markerlemek değildir.

**Modül 1C ise ekledi, ve gerekçesi ölçüldü:** yavaş şerit 25 → **36**
(+11), hepsi Task 1 ve Task 3'ten — N₂/F₂ üzerinde çok başlangıçlı arama
tanımı gereği pahalıdır (bir geometri × 8–12 tohum × tam SCF) ve ucuzlatmanın
tek yolu aramayı küçültmekti, ki o zaman test **ölçtüğü şeyi ölçmezdi**.
Task 4+5'in eklediği **20 testin 20'si** hızlı şeritte: iki dosya birlikte
2.76 s. Modül 1C'nin beş dosyası toplam 31.46 s'nin 60 testini taşıyor.

**Modül 2E hiç yavaş test eklemedi.** Plan E'nin planı üç testi
`@pytest.mark.slow` işaretliyordu; ölçüldü ve kaldırıldı — en yavaşı 0.35 s,
yani hepsi "saniye altı çıpa" kuralının içine düşüyor. Sonuç: yavaş şerit
25 testte sabit kaldı, eklenen 17 testin **17'si de** hızlı şeritte.

**Modül 3H, deponun en pahalı modülü oldu — ve gerekçesi ölçüldü.** Süite
864 → **971** (+107): hızlı şerit 828 → **920** (+92), yavaş şerit 36 → **51**
(+15). Yavaş şeridin süresi 210.65 → **352.41 s**; aradaki ~142 s'nin tamamını
Modül 3H'ye yazmak **iki ayrı oturumun** karşılaştırması olur ve bu makine
eşzamanlı yüke %16'ya kadar duyarlıdır — test SAYILARI kesin, saniye farkı
yaklaşıktır. Sebep ucuzlatılamaz: bu modülün her fiziksel testi bir
**arama protokolü** koşar (bir geometri × 12–20 tohum × tam UHF), çıpa testi
üstüne `brentq` ile bir kök araması ekler ve sahte-çıpa sınavı altı ayrı baz
seti üzerinde tam bir yasa uydurması yapar (36 `search_uhf` çağrısı). Aramayı
küçültmek testin **ölçtüğü şeyi ölçmemesi** demek olurdu.

Task 5'in (senaryo + panel + README) kendi payı küçük: eklenen **25 testin
24'ü** hızlı şeritte, tek yavaş test uçtan uca CLI koşumu (5.02 s). `tests/tb`
artık 185 test / 138.55 s; hızlı şeridi ise 170 test / **16.73 s** — yani
günlük çalışma döngüsü hâlâ ucuz, pahalı olan yalnızca işaretli şerit.

**Yavaş testlerin dağılımı (ölçüldü, 864 testlik oturum — Modül 3H'nin
eklediği 15 yavaş test bu listede YOKTUR):** `hf/test_physical.py` ×5,
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

### Modül 1C — kısıtlanmamış Hartree-Fock (UHF)

**Bu modülün kazancı enerji değil, DÜRÜSTLÜK.** UHF buraya bir "özellik"
olarak değil, bir **ön koşul** olarak girdi: Plan G (tight-binding
parametrelerini deponun kendi HF'inden türetme) yazılacaktı, ön ölçüm
referans sisteminin RHF çözümünün HF'in taban durumu **olmadığını** gösterdi
ve downfolding'in çıpası yanlış bir yüzeye oturacaktı. Plan ertelendi, yerine
bu modül yazıldı. (Çekirdek spec §13'te "kapsam dışı" yazan madde bu yüzden
üstü çizilerek düzeltildi.)

#### Deponun kendi referans kaydı taban durum değildi

`reference/hf_energies.json`'daki **11 PySCF kaydının 10'u** çok başlangıçlı
UHF aramasıyla **≤ 9.001e-12 hartree** uyuşuyor. Ayrılan tek kayıt `h2_200`
(H₂ @ 2.00 Å):

| büyüklük | değer |
|---|---|
| E_RHF (referans kaydı) | **−0.783792654277** |
| E_UHF (12 tohumlu arama) | **−0.937212833096** |
| fark | **0.153420178832 hartree** |
| iç (RHF→RHF) kararlılık | **+2.546683038854** → *"kararlı"* |
| dış (RHF→UHF) kararlılık | **−1.599532559297** → **spin eyeri** |

Yani depo, gerçek HF taban durumundan 0.153 hartree yukarıdaki bir noktaya
"iç minimum" diyordu ve bu çıktının **hiçbir yerinde** görünmüyordu. Modül 1C
o sessizliği kaldırıyor: `solve(stability=True)` artık iki kanalı da koşuyor
ve panel iki ayrı satır basıyor.

#### Coulson-Fischer noktası — modülün en güçlü çıpası

RHF çözümünün spin kanalında kararlı olmaktan çıktığı R. Fiziksel bir
**çatallanma** noktasıdır; deneysel hiçbir sayı içermez, hiçbir yerden
verilmez, **okunur**.

| yol | ne KULLANMIYOR | Rc (bohr) |
|---|---|---|
| (a) triplet Hessian'ın kökü | — | **2.179694793840** |
| (b) simetriden kurulmuş orbitaller + sonlu fark | **SCF yok, analitik Hessian yok** | **2.179694793998** |
| (c) SCF'in kendi davranışı (UHF/RHF ayrımı) | Hessian yok | aynı sınır |

Fark **1.578e-10**, ve bu fark **kesme değil YUVARLAMA** — ölçüldü: h
küçüldükçe fark *büyüyor* (3e-3 → 3.736e-10, 1e-3 → 1.275e-09, 3e-4 →
3.734e-09), yani h=1e-3 zaten yuvarlamanın hâkim olduğu bölgede.
Rc = 1.153444811624 Å.

Çatallanma üsleri hiçbir yerden verilmedi, uyumdan çıktı:

| pencere | \|ΔE\| | \|spin\| | ⟨S²⟩ |
|---|---|---|---|
| d ≤ 2e-1 | 1.9767 | 0.4825 | 0.9727 |
| **d ≤ 1.6e-2** | **1.9960** | **0.4970** | **0.9953** |
| çatallanma beklentisi | 2 | 1/2 | 1 |

Zincirlerde Rc kayıyor: n=2 **2.179694794**, n=4 1.951986163, n=6 1.804435695
bohr (ölçüldü ama iddia edilmiyor: n=8 1.695323976, n=10 1.609356683, n=12
1.538971647 — 1/n uyumu artığı 0.089'a çıkıyor, altı nokta n→∞ limitini
seçmeye **yetmez**).

#### Çıpa tablosu — SINIFIYLA birlikte

Tautolojiler gizlenmiyor; sınıflarıyla yazılıyor. Sınıflar:
*dairesel değil* (bağımsız ikinci yolu var) · *transkripsiyon denetimi*
(özdeşlik, ama mutasyona duyarlı) · *tautoloji* (fizik-dışı girdide de
tutuyor).

| iddia | ölçülen | sınıf |
|---|---|---|
| **Coulson-Fischer noktası** | üç bağımsız yol, fark 1.578e-10 | **dairesel değil — en güçlü** |
| **Çatallanma üsleri** | 1.9960 / 0.4970 / 0.9953 (beklenti 2 / ½ / 1) | **dairesel değil** |
| Parça toplamı (boyut tutarlılığı) | H₂ +5.551e-16, LiH −9.592e-14, N₂ +1.421e-14 | dairesel değil, **güçlü** |
| Ayrışma limiti | UHF(H₂, 20 bohr) − 2E_H = **4.441e-16**; RHF aynı yerde **9.85877 eV** sapıyor | dairesel değil, **zayıf** (iki mutant geçiyor) |
| Triplet Hessian = enerjinin 2. türevi | bağıl fark 1.305e-10 … 6.659e-10, beş sistemde | transkripsiyon denetimi (**taşıyıcı**) |
| λ_triplet < 0 ⟺ UHF kazancı var | gerçek sistemlerde **18/18**; fizik-dışı girdide de **51/53** | **muhafız** — sahte-çıpa sınavını geçiyor, çıpa DEĞİL |
| E_UHF ≤ E_RHF | 0 karşı örnek | transkripsiyon denetimi, **zayıf** (10 mutantın 7'sini kaçırıyor) |
| RHF kararlıyken UHF özdeş | fark < 1e-12 | **TAUTOLOJİ** — muhafız, "çıpa" değil |

Son satırdaki sahte-çıpa sınavı **iki ayrı soruyu** ayırıyor ve ikisinin
cevabı farklı:

| ölçüt | gerçek H₂ (18 ayrım) | fizik-dışı rastgele integraller (53 yakınsayan) |
|---|---|---|
| **triplet** kanal | **18/18** | 51/53 |
| **singlet** kanal | **8/18** | 50/53 |

Fizik-dışı girdide tuttuğu için hüküm **fizik değil** (cebirsel bir sonuç:
Hessian, SCF'in indiği aynı fonksiyonelin ikinci türevidir). Ama sahte girdi
iki kanalı **ayıramıyor**; ayrım gerçek moleküllerde çıkıyor. Yani hüküm
fizik değil ama **doğru cebir** — yanlış kanalla verildiğinde gerçek
moleküllerde 10/18 düşüyor.

#### Sevk edilmiş kodda bulunan kusur — ve Modül 2D'ye etkisi

Task 1 deponun **zaten sevk edilmiş** `solve()`'unda iki ayrı kusur ölçtü;
ikisi de `converged=True` diyordu:

| kusur | nerede | ölçülen |
|---|---|---|
| **DIIS** | N₂ R=1.10 Å | DIIS'li −106.769673858 vs DIIS'siz **−107.496500512** → **0.726827 Ha = 19.78 eV** |
| **başlangıç tahmini** | N₂ R=1.30–2.00 Å | 2–5 ayrı çözüm; en düşüğü **0.278–0.498 Ha** aşağıda (DIIS masum) |
| aynı kusur, ikinci molekül | F₂ R=1.80 / 2.00 / 2.40 Å | **0.4094 / 0.2513 / 0.0812 Ha**; 3–4 ayrı çözüm |

F₂'nin **kendi dengesi** etkilenmiyor, ve o denge deneysel bir sayı değil
**ölçüldü**: yöntemin STO-3G minimumu **R = 1.314612 Å**,
E = −195.981624421. Orada 12 tohum tek çözüme iniyor (fark 1.705e-13) ve iki
kanalda da kararlı (iç **+1.304533**, dış **+0.228957**). Kusur gerilmiş
geometrilerin kusurudur.

**Modül 2D bu `solve()` üzerine kuruludur, dolayısıyla etki ÖLÇÜLDÜ:**

| R (bohr) | çok başlangıçlı RHF farkı | ayrı çözüm | iç λ | dış λ | UHF kazancı |
|---|---|---|---|---|---|
| 1.2960 (MD dönüş noktası) | 4.441e-16 | **1** | +4.77335 | **+1.92166** | **0.000e+00** |
| 1.3459 (denge) | 4.441e-16 | **1** | +4.64526 | **+1.77054** | **0.000e+00** |
| 1.3960 (MD dönüş noktası) | 0.000e+00 | **1** | +4.52409 | **+1.62586** | **0.000e+00** |
| 2.1797 (= Rc) | 2.220e-16 | 1 | +3.30753 | −0.00001 | 4.733e-12 |
| 2.5000 | 6.661e-16 | 1 | +3.03687 | −0.45013 | 1.415e-02 |
| 4.0000 | 0.000e+00 | 1 | +2.51654 | −1.72551 | **1.748e-01** |

**Hüküm: Modül 2D'nin sonuçları etkilenmiyor, ve bu bir varsayım değil
ölçüm.** Ab initio MD H₂'yi 1.3959 bohr'dan başlatıp ~0.05 bohr genlikle
titreştiriyor; gezdiği [1.296, 1.396] aralığının her noktasında (i) çok
başlangıçlı RHF, **12 tohumun 12'sinde de aynı** çözüme iniyor (`n_distinct`
= 1, enerji farkı ≤ 6.7e-16) ve (ii) dış kararlılık özdeğeri **pozitif**,
UHF kazancı **tam sıfır**. Yani ω_A / ω_B uyuşması (milyonda 4.4) gerçek HF
taban durumu üzerindedir.

*(Bu, "H₂/STO-3G'de tek bir RHF durağan noktası vardır" demek DEĞİLDİR —
σu² işgali de bir durağan noktadır. Ölçülen şey, 12 tohumluk bu arama
protokolünün bu geometrilerde daha düşük bir çözüm bulamadığıdır.)*

**Ama modülün geometri kapısı 21.1 bohr'a kadar açık** ve R > 2.1797 bohr'da
aynı yüzey bir spin eyeridir — 4.0 bohr'da gerçek HF taban durumu
**0.1748 hartree aşağıdadır**. Modül 2D'nin BO yüzeyi bu yüzden yalnızca
Rc'nin altında HF taban durumudur; üstünde "RHF yüzeyi"dir ve öyle
okunmalıdır.

#### M9 deliği — tasarlanmış bir mutant dokuz denetimi de geçti

Task 3'ün kabul ölçütü şu mutanttı: doğru Fock, doğru enerji, ama artık
`E += 1e-3 · Σ(Mulliken spin popülasyonu)²`, ve yalnız N > 2'ye kapılı. Ön
ölçümde önerilen **dokuz denetimin dokuzunu da geçiyordu** — çünkü önerilen
muhafızın dört vakasının dördünde de spin popülasyonu sıfırdı.

Delik gerilmiş H₄ eklenerek kapatıldı, ve muhafızın **taşıyıcı** olduğu δ
küçültülerek kanıtlandı:

| δ (ceza katsayısı) | kırılan test |
|---|---|
| 1e-3 | yedi (atom enerjileri, parça toplamı, HF sınırı ve muhafız) |
| 1e-10 | iki: LiH parça toplamı ve muhafız |
| **1e-12** | **yalnız muhafız** |
| 1e-13 | hiçbiri (eşiğin altında) |

**Ders:** bir mutantın ölmesi, onu öldürmek için yazdığın testin çalıştığı
anlamına gelmez. *Hangi* testin öldürdüğünü sor.

#### Kök okuyan çıpa ÖLÇEĞE KÖRDÜR

Bu depoda ilk kez ölçüldü ve çıpa tasarımı hakkında geneldir. Triplet
Hessian'ın `4.0` çarpanı `2.0` yapıldığında **Rc hiç kıpırdamıyor** — küresel
bir çarpan bir sıfır geçişini yerinden oynatmaz:

| test | çarpan mutantı |
|---|---|
| `test_COULSON_FISCHER_noktasi` | **yeşil** |
| `test_ZINCIRLERDE_Rc_n_ile_KAYIYOR` | **yeşil** |
| `test_TRIPLET_OLCUTU_...SINGLETTEN_iyi` | **yeşil** |
| `test_Rc_IKINCI_BAGIMSIZ_YOLDAN_SCF_SUZ` (aralık boyu) | kırmızı |
| `test_TRIPLET_Hessian_SONLU_FARKLA_uyusuyor` | kırmızı |
| `test_TRIPLET_SINGLETTEN_AYRI_HUKUM_veriyor` | kırmızı |

**Kök = FİZİK, ölçek = TRANSKRİPSİYON.** İki ayrı test sınıfı gerekir ve biri
diğerinin yerine geçmez; yalnız çıpalarla yetinilseydi çarpanı yanlış yazılmış
bir Hessian sevk edilirdi.

Diğer mutantlarda Rc nereye gidiyor: Coulomb terimi **bırakılırsa** (yani
singlet yazılırsa) kök **hiç yok** — H₂'nin singlet Hessian'ı 0.6–12 bohr
arasında negatife inmiyor; yarım bırakılırsa yine kök yok; Fock bloklarının
işareti takas edilirse yine kök yok; `(ab|ij)` terimi düşerse kök
**4.570719109 bohr**'a kayıyor (2.39 bohr hata — ama VAR, yani "kök var mı"
diye soran bir test bunu kaçırırdı).

#### Dürüst olumsuzlar

- **⟨S²⟩ enerjiden ~6 mertebe yavaş yakınsıyor.** Ölçüldü: SCF toleransı
  1e-6'da enerji zaten 2.585e-12'ye oturmuşken ⟨S²⟩ hâlâ **1.457e-06**
  oynuyor. Sebebi yapısaldır — enerji durağan noktada ikinci mertebeden
  duyarlıdır, ⟨S²⟩ değildir. ⟨S²⟩'den yedi anlamlı haneden fazlası
  okunmamalıdır.
- **UHF bir spin özdurumu değildir.** `h2_200`'ün UHF çözümünde ⟨S²⟩ =
  **0.945862370628**, oysa bir singlet için 0 olmalıdır. Bu spin kirlenmesi
  ölçülüp raporlanıyor ama **düzeltilmiyor**: spin projeksiyonu (PUHF) yok.
- **Her UHF enerjisi bir ARAMA protokolüyle gelir** ve çıpanın gücü kodun
  değil aramanın özelliğidir. HF molekülünde (H + F, 20 bohr) parça toplamı
  −98.453086808293 ama bu tohum ailesi taban durumu **hiç bulmuyor**:
  8/12 tohum → +0.8824 Ha, 24/40/64 tohum → **+0.7749 Ha** ıskalıyor. Sekiz
  kat tohum farkın yalnız 0.11 Ha'ını kapatıyor. Ayrıca N atomunun UHF
  enerjisi tohuma bağlıdır: tekbiçim bias −53.719010163, alternan bias
  −53.259271159 (**0.46 Ha** fark). Aynı kırılganlık çatallanma üslerinde de
  görüldü — tek sabit bias ile UHF bazı R'lerde RHF'e geri çöküyor ve üs
  `nan` çıkıyor; yukarıdaki tablo `search_uhf` (12 tohum) ile ölçülmüştür.
- **PySCF bu makinede kurulu değil** (yalnız kaynak tarball); `reference/`
  kayıtları PySCF **2.14.0** ile başka bir yerde üretilmiştir. Dahası 11
  referans kaydının **hepsi kapalı kabuktur**, yani UHF'in **spin kırılan
  dalını hiç sınamıyor** — o dalın doğrulaması bu depoda yalnız iç tutarlılık
  (ayrışma limiti, boyut tutarlılığı, çatallanma) üzerinden yapılır.
- **Deneysel bağ uzunluğu hiçbir yerde girdi değildir.** F₂ için yapılan
  taramada yöntemin kendi dengesi (1.30 Å) kullanılmıştır.
- **Kompleks (reel olmayan) orbital kanalı hâlâ sınanmıyor.** İç ve dış
  kanalın ikisinde de "kararlı" hükmü, çözümün **küresel minimum** olduğunu
  söylemez.

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
| Süite | — | Plan F kapanışında **801 test yeşil**, `tests/tb/` 80 test / 4.19 s (o günün sayısı; bugünkü toplam için yukarıdaki süite tablosuna bakın) |

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

### Modül 3H — UHF indirgemesi / downfolding (Plan H)

Modül 3F'nin açıkça yazılmış en büyük açığı kapandı: **parametreler artık
dışarıdan girmiyor, qlab'ın kendi UHF'inden türetiliyor.**
`engine/tb/downfold.py`, `search_uhf`'in bulduğu spin-kırık çözümün α/β Fock
matrislerini Löwdin ile ortogonalleştirip site bazında okur

    H_eff^σ = S^(-1/2) F^σ S^(-1/2)

ve dört sayı üretir: `ε̄`, `Δ`, `t₁`, `t₂`. Sevk edilenler: `downfold_uhf`,
`model_from_downfold`, `fit_distance_laws` / `DistanceLaws`, `hydrogen_ring`;
`kind: downfold` senaryosu (`scenarios/halka_indirgeme.yaml`); testler
`test_downfold.py`, `test_downfold_cipa.py`, `test_downfold_kesme.py`,
`test_downfold_yasa.py`, `test_downfold_senaryo.py`.

**Modülün asıl kazancı bir sayı değil bir çıpa:** kademeli alanın çatallanma
üssü, üç halkada bağımsız olarak ½ çıkıyor. **Modülün asıl sınırı da yazılı:
hızlanma İDDİA EDİLMİYOR** — HF/TB maliyet oranı bu modülde hiç ölçülmedi.

#### Neden UHF, neden RHF değil — hüküm Hückel'den geliyor

Plan G'nin indirgemeyi RHF üzerine kurma denemesi düştü, çünkü hidrojen
halkalarında RHF çözümü çoğu geometride HF'in **taban durumu değildir**.
Desen 4n / 4n+2'dir ve **7/7 ölçüldü** (d = 1.6 bohr, dış — RHF→UHF, üçlü —
kararlılık özdeğeri ve UHF'in kazandırdığı enerji):

| N | tür | dış λ | UHF kazancı (hartree) |
|---|---|---|---|
| 6 | 4n+2 | **+1.26984** | 1.78e-15 |
| 8 | 4n | **−0.38237** | **0.099079** |
| 10 | 4n+2 | **+0.59956** | 7.11e-15 |
| 12 | 4n | **−0.30153** | **0.082805** |
| 14 | 4n+2 | **+0.22768** | 1.07e-14 |
| 16 | 4n | **−0.26614** | **0.077927** |
| 18 | 4n+2 | **+0.00983** | 2.84e-14 |

4n halkalarında TB spektrumunun HOMO/LUMO'su tam dejeneredir; RHF dejenere
seviyeyi çift işgale zorlayınca spin kararsızlığı doğar ve UHF gerçekten
aşağı iniyor (yaklaşık 0.08–0.10 hartree). 4n+2'de kazanç sayısal sıfır.

**N=8 ve N=12 test edilen hiçbir geometride kararlı değil** — altı mesafede
(1.6 / 1.8 / 2.0 / 2.2 / 2.6 / 3.0 bohr) **12/12** nokta negatif λ verdi.
N=8'in altı değeri: −0.382368, −0.390441, −0.399690, −0.409562, −0.428241,
−1.203061.

> **Plan H'nin "45 noktanın 35'i spin kararsız" sayısı tekrarlanamadı ve
> sebebi bu deponun kendi tekrar eden dersi:** ızgara yazılmamıştı. Adı konmuş
> bir ızgarayla yeniden ölçüldü — N ∈ {6, 8, 10, 12, 14} × d ∈ {1.6, 1.8, 2.0,
> 2.2, 2.6, 3.0, 3.4, 3.8, 4.2} bohr, yine 45 nokta — ve çıkan sayı
> **38/45**. Nitel hüküm (noktaların büyük çoğunluğu spin kararsız) aynı;
> kesin sayı **ızgaraya bağlı** ve ızgara adlandırılmadan tekrarlanamaz.
> Yukarıdaki 38/45 bu ızgaranın sayısıdır.
>
> Tamamlık için: özgün 35/45 **yanlış değildi**, ızgarası yazılmamıştı. O
> ölçüm N ∈ {6, 8, 10, 12, 14} × d ∈ {1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6,
> 3.2, 3.8} bohr üzerinde yapılmıştı ve o ızgarada tekrarlanıyor. İki sayının
> farkı, iki ızgaranın uzak mesafelerde farklı sıklıkta olmasıdır — 4.2
> bohr'a kadar giden ızgara daha çok kararsız nokta içerir. **Kusur sayıda
> değil, sayının yanında ızgaranın olmamasındaydı.**

#### Modelin şekli varsayım değil ÖLÇÜM

Spin-kırık halka çözümü antiferromanyetiktir ve bu her çağrıda **yeniden
doğrulanır**, varsayılmaz:

- Mulliken spin popülasyonu alternan (N=8, d=2.0: ∓0.517253)
- `diag(H_eff^α)` iki değer arasında alternan: **+0.250498 / −0.135858**
- `diag(H_eff^β)`, α'nın **bir site kaydırılmışı**. Kaydırma artığı beş
  sistemde ölçüldü ve sistemler **adlıdır** — aralık, hangi sistemlerden
  geldiği yazılmadan tekrarlanamaz:

  | sistem | kaydırma artığı |
  |---|---|
  | N=6, d=2.6 | 2.220446e-15 |
  | N=6, d=3.0 | 1.942890e-15 |
  | N=8, d=2.0 | 5.558044e-09 |
  | N=8, d=2.6 | 3.951947e-09 |
  | N=10, d=2.6 | 9.087647e-09 |

- Hoplama **spinden bağımsız ve tekbiçim**: N=8 d=2.0'da t = −0.407145,
  sekiz bağın sekizinde de aynı. "Aynı"nın ölçüsü:
  max|tᵅ − tᵅ[0]| = **3.003514e-09**, max|tᵅ − tᵝ| = **8.213723e-10**. Bu
  sayılar sıfır değildir ve olmaları da beklenmez — `solve_uhf`'in artık
  toleransı 1e-8'dir, Fock elemanlarının gürültü tabanı oradadır.

Dolayısıyla bağımsız parametre **dörttür**. `Δ`'nın **işareti bir ayar
seçimidir** — hangi alt-örgünün α fazlası taşıdığına UHF araması karar verir
ve karar d'den d'ye döner. Ölçüldü (N=8, d = 1.6 … 3.6, on nokta):
`+ − + − + − + + + +`.

#### ÇIPA — kademeli alanın çatallanma üssü · **SINIF: dairesel değil**

İki büyüklük **iki ayrı hesaptan** gelir ve birbirini hiç kullanmaz. `d_c`,
RHF çözümünün üçlü orbital Hessian'ının sıfır kökünden (Modül 1C,
`analysis.external_stability`); `Δ` indirgemeden. Üs hiçbir yerden verilmez,
log-log eğiminden okunur.

| N | d_c (bohr) | ölçülen üs | Δ(d_c + 0.032) | log-artık (max) |
|---|---|---|---|---|
| 6 | **2.233818500** | **0.496190** | 8.115373e-02 | 1.892e-03 |
| 10 | **1.957853161** | **0.497600** | 6.145320e-02 | 1.169e-03 |
| 14 | **1.757512489** | **0.497811** | 4.948334e-02 | 1.034e-03 |

Üçü de çatallanma beklentisi **½**'nin hemen altında: sapmalar 0.003810
(N=6), 0.002400 (N=10), 0.002189 (N=14) — ve N büyüdükçe küçülüyor. Ve
**üçüncü bir bağımsız okuma**: Modül 1C, H₂ molekülünde — başka bir sistem,
başka bir yordam, Coulson-Fischer noktasında |spin| ölçüsü — üssü **0.4970**
ölçmüştü (`tests/hf/test_coulson_fischer.py`). Aynı yere düşüyor.

> Task 2'nin dokstringi bu satırda *"üs ½'ye 0.0022'den yakın"* diyordu ve o
> sayı **yanlıştı** — N=6'nın sapması 0.003810, yani iddia edilenin 1.7 katı.
> Doğrulama turunda yakalandı ve dokstring düzeltildi; testin toleransı
> (abs=0.02) zaten gerçek sapmanın çok üstündeydi, yani hata testi kırmıyordu
> ve ancak **yeniden ölçülünce** görülebilirdi.

Çıpa **iki taraflıdır**: geçişin altında `Δ` sönüyor — N=6 d_c−0.20'de
2.886867e-09, d_c−0.02'de 1.539807e-08; N=10'da 9.908740e-16 ve
1.998401e-16. Bu artıklar fizik değil `solve_uhf`'in artık tabanıdır ve eşik
(1e-7) tam da bu yüzden planın önerdiği 1e-8 **değildir**: N=6 d_c−0.02
noktası 1.539807e-08 ile 1e-8'i aşıyor.

> **Üs testi tek başına ÖLÇEĞE KÖRDÜR** ve bu delik ayrıca kapatıldı: log-log
> eğimi `Δ → c·Δ` dönüşümü altında değişmez, yani üs testi `Δ`'yı sabit bir
> sayıyla çarpan bir kusuru göremez. Aynı testte mutlak bir değer de
> kilitlenir (yukarıdaki `Δ(d_c+0.032)` sütunu) ve o satır **transkripsiyon
> denetimidir**, çıpa değil.

#### Kesme hatasının dürüst tablosu

Hata = max|iki-siteli TB özdeğerleri − UHF α özdeğerleri|, hartree, STO-3G,
seeds=16. **Kesme hatası küçük DEĞİLDİR ve büyük olduğu yazılıdır.**

| N | d | UHF yayılımı | NN-only | +t₂ | oran | %NN | %t₂ |
|---|---|---|---|---|---|---|---|
| 6 | 2.60 | 1.113021 | 0.039420 | **0.012078** | 3.26 | 3.5% | 1.1% |
| 6 | 3.00 | 0.947273 | 0.023748 | **0.009107** | 2.61 | 2.5% | 1.0% |
| 6 | 3.60 | 0.834770 | 0.010104 | **0.003925** | 2.57 | 1.2% | 0.5% |
| 8 | 2.00 | 1.693171 | 0.153603 | **0.066287** | 2.32 | 9.1% | 3.9% |
| 8 | 2.60 | 1.134138 | 0.073229 | **0.032485** | 2.25 | 6.5% | 2.9% |
| 10 | 2.60 | 1.140294 | 0.059631 | **0.020470** | 2.91 | 5.2% | 1.8% |

Planın (ve ilk dokstringin) "t₂ hatayı ~3 kat düşürür" cümlesi **düzeltildi**:
ölçülen oran 2.25 … 3.26, ortalaması 2.65 — "üç kat" altı sistemin yalnızca
**birinde** tutuyor. Doğru cümle *"iki ila üç kat"*tır.

#### Modelin tavanı MENZİL değil KABUK ORTALAMASI

"Menzili büyütünce hata sıfıra gider" beklentisi **yanlış** ve ölçüldü:
N=8 d=2.0'da menzil tamamen açıkken (m=4, hiçbir şey kesilmemişken) hata hâlâ
**0.063140** — menzil 2'nin 0.066287'sinden yalnızca %5 aşağıda.

Sebep iki parçalı ve ikisi de ölçüldü. **(a) İndirgemenin kendisi kayıpsız:**
`H_eff^α`'nın özdeğerleri UHF α orbital enerjileriyle **7.194e-10**'da aynı.
**(b) Kayıp kabuk ortalamasında:** model her halka mesafesi m için tek bir
`t_m` taşır, ama `H_eff^α` **çift** m'de A-A ile B-B'yi ayırır (α kanalında A
alt-örgüsü α fazlası, B β fazlası taşır ve ikisi eşdeğer değildir). Kabuk içi
yayılım, N=8 d=2.0:

| kabuk | yayılım |
|---|---|
| m=1 | 3.003515e-09 (tekbiçim — SCF artık tabanı) |
| m=2 | **5.106121e-02** |
| m=3 | 1.007388e-09 (tekbiçim) |
| m=4 | **2.415794e-02** |

Tek m'de kabuk tekbiçim, çünkü o hoplamalar A↔B'yi bağlar ve AFM işlemi
altında simetriktir. **Hükmü:** hatayı gerçekten düşürmek için gereken şey t₃
değil, çift menzilde alt-örgüye bağlı **iki ayrı** hoplamadır (t₂^AA, t₂^BB) —
ve o, dört parametreli modelin **kapsamı dışındadır**.

#### Sahte-çıpa sınavı · **SINIF: dairesel değil, ama DAR**

Sınav: fiziği çıkar, **boru hattını yerinde bırak**; sonra aynı yasa uydurma
makinesini fizik-dışı girdiyle koştur. Ölçülen metrik **ekstrapolasyon
oranı** = (pencere dışı log artık RMS) / (pencere içi log artık RMS) — yani
"yasa, uydurduğu yerden ne kadar uzağa taşıyor".

N=8 halkası, pencere 1.8/2.0/2.2 bohr, hedef 2.4/2.6/3.0 bohr, seeds=12:

| kurulum | oran |
|---|---|
| **GERÇEK (STO-3G)** | **4.160383** |
| 40 rastgele düğümsüz baz — medyan | 29.902685 |
| aynı 40 — **en iyisi** | 6.216006 |
| aynı 40 — en kötüsü | 413.126243 |
| **gerçeği geçen** | **0/40** |
| F3 saf geometri çekirdeği (`S=exp(−a r)`, `H=S⁻¹`), beş a | 9.9647 … 29.3917, **0/5** |

45 fizik-dışı kurulumun 45'i de gerçekten kötü. **F2 (`eri*0`) koşulamadı ve
bu bir eksiklik değil bir ölçüm:** elektron-elektron kapanınca değiştokuş da
yok olur, UHF RHF'e çöker ve |Δ| = 1.110223e-16 çıkar; `fit_distance_laws`
bunu görüp `UnsupportedSystemError` fırlatıyor. İkame **beyanla değil kodun
kendi kapsam kontrolüyle** elendi. Aynı kontrol F1'de de gerçekten ateşledi:
41 çizimin 1'i "t₂ pencere içinde işaret değiştiriyor" diye reddedildi.

#### Dürüst olumsuzlar — dördü de ölçüldü

1. **Kuyruk marjı RHF tarafının üçte biri.** Burada gerçek 4.160383, en iyi
   sahte 6.216006 — yalnızca **1.49 kat**. Karşılaştırma noktası Plan H'nin
   RHF yolu üzerindeki ön ölçümüdür (2.80'e karşı en iyi sahte 12.01, yani
   **4.29 kat**); o rakam bu depoda **yeniden ölçülemez** çünkü RHF
   indirgemesi sevk edilmedi — kaynağı belirtilerek aktarılıyor. Sınav
   geçiyor ama ayırma gücü düşmüş durumda ve bu yazılmadan geçilemez.
2. **İkinci metrik aynı hükmü vermiyor.** "Yasa pencere dışını iyi öngörüyor"
   cümlesi tek başına fizik kanıtı **değildir**: ham bağıl öngörü hatasında
   40 sahtenin **2'si** (0.005409 ve 0.005571; gerçek 0.010811) ve saf
   geometri çekirdeklerinin **5'inden 2'si** (0.000997 ve 0.000297) gerçekten
   **daha iyi** öngörüyor. `exp(−a r)` tanımı gereği üsteldir; "üstel bir
   yasaya uyuyor" olmak bir çekirdeği fiziksel yapmaz.
3. **Mutlak hata tek başına da ayırmıyor, ters yönden.** 40 sahtenin **7'si**
   gerçekten küçük mutlak hata veriyor — ikisinin sebebi t₁'lerinin neredeyse
   **sıfır** olması. Bu yüzden her iki sınır da (bağıl VE mutlak) kilitlenir.
4. **Sınav HEDEF KÜMESİNE KOŞULLU.** Aynı yasa, pencere hiç değişmeden,
   yalnızca hedef kümesi uzatılarak:

   | hedef kümesi (bohr) | oran |
   |---|---|
   | (2.4, 2.6, 3.0) | **4.160383** |
   | + 3.2 | 5.997079 |
   | + 3.6 | **15.479892** |

   d = 3.6 hedefe konsaydı **gerçek sistem en iyi fizik-dışı bazdan
   (6.216006) daha kötü görünürdü ve sınav düşerdi.** Çıpanın geçtiği hüküm
   bu yüzden "yasa ekstrapole eder" değil, şu dar hükümdür: *1.8–2.2 bohr
   penceresinden uydurulan t₁ yasası, 3.0 bohr'a kadar fizik-dışı bazların
   üretemediği bir doğrulukla öngörüyor.* 3.0 bohr'un ötesi için bu modül bir
   şey iddia etmez.

#### `p = 2` ölçülmedi, SEÇİLDİ — ve bedeli ölçüldü

`ε̄ = ε_∞ + C·t₁²` yasasındaki üs Löwdin'in ikinci mertebe açılımından gelir
(yerinde düzeltme örtüşmede ikinci mertebedir), yani **teoriden**. Serbest
bırakıldığında N=8 halkasının on noktalı gridinde ölçülen üs **1.8887** —
yani serbest uyum 2'ye yakın ama onu **vermiyor**. (Plan H'nin RHF yolundaki
ön ölçümü 1.9869 diyordu; o sayı bu depoda yeniden ölçülemez, kaynağı
belirtilerek aktarılıyor. UHF tarafında serbest üs 2'den **daha uzak**.) p=2
zorlamasının pencere içi artığı 5.737638e-04 RMS.

#### `Δ` için temiz bir mesafe yasası YOK — ve sebebi ölçüldü

`ln|t_m|` doğrusal oturuyor (artık RMS 1.855587e-03 ve 4.104227e-04), ama
`|Δ| = A·exp(b·d)` **yalnızca pencere içinde** geçerli; dışarıda hızla
bozuluyor: bağıl hata 0.028079 (d=2.4) → 0.055498 (2.6) → 0.069184 (3.0),
yani t₁'in maksimumunun (0.010811) **altı katı** ve d ile monoton artıyor.
Sebep: **`Δ` doyuyor.** N=8'de |Δ| d=1.6'da 0.170033 iken d=3.6'da
0.363453'te doyuyor; pencere yasası orada %5.0 yüksek tahmin veriyor
(0.381747).

Doymanın sebebi **bağımsız bir hesapla** doğrulandı. Hubbard ortalama alan
boşluk denklemi, ölçülen (t₁, Δ) çiftinden U'yu çözdürür; çıkan U neredeyse
sabittir ve Löwdin bazında **bağımsız olarak** hesaplanan yerinde Coulomb
integrali (00|00) ile aynı yere düşer:

| d (bohr) | t₁ | \|Δ\| | U (boşluk denklemi) | (00\|00) | fark |
|---|---|---|---|---|---|
| 1.6 | −0.573180 | 0.170033 | 0.873471 | 0.946171 | 7.7% |
| 2.0 | −0.407145 | 0.193178 | 0.827766 | 0.883907 | 6.4% |
| 2.6 | −0.255455 | 0.265159 | 0.802284 | 0.828145 | 3.1% |
| 3.0 | −0.183164 | 0.318473 | 0.794279 | 0.806566 | 1.5% |
| 3.6 | −0.106972 | 0.363453 | 0.784102 | 0.788356 | **0.5%** |
| 6.0 | (ayrışmış atom sınırı) | — | — | 0.774783 | — |

U uçtan uca 0.873471 → 0.784102 (oran 1.114) oynuyor; uyum d büyüdükçe
iyileşiyor — ortalama alanın ve atomik sınırın en iyi çalıştığı yerde. Yani
`Δ` doyar çünkü U mesafeye kabaca duyarsızdır ve yerel moment 1'e doyar;
Δ = U·m/2.

#### Sınırlar — modülün beyan ettiği kapsam

- **Yalnız STO-3G.** 6-31G atom başına iki fonksiyon taşır ve site-bazlı okuma
  orada **tanımsızdır** (`H_eff[0,1]` atom-*içi* elemanı döndürürdü). Kod bunu
  sessizce geçmez, `UnsupportedSystemError` fırlatır.
- **Yalnız halka.** Açık zincir kapsam dışıdır ve gerekçesi ölçüldü: N=8 açık
  zincirde, halkadan gelen **bütün** menzillerle (m = 1…4; halka mesafesi
  orada doyar, bulk parametrizasyonu daha uzağını zaten tanımlamaz) kurulan
  modelin hatası d=1.8 bohr'da **0.095563 Ha**. Plan H'nin ön ölçümü aynı
  büyüklüğü "0.096 Ha'lık bir taban" diye kaydetmişti ve sayı tutuyor; ama
  **taban olduğu bu protokolde görünmüyor** — hata d ile düzgün küçülüyor
  (0.086963 / 0.058007 / 0.024325 @ 2.0 / 2.6 / 3.2 bohr). Yani o taban, eğer
  varsa, **menzil ekseninde** bir tabandır, mesafe ekseninde değil. Kenar
  etkisi bu modülde modellenmez ve bu satır onu modellemeye çalışmaz.
- **UHF bir spin özdurumu DEĞİLDİR.** ⟨S²⟩ sonuca iliştirilir ama
  **düzeltilmez** (spin projeksiyonu / PUHF yoktur). Ölçüldü: N=8 d=2.0'da
  ⟨S²⟩ = **1.220731** — ne singlet ne triplet. Sevk edilen senaryonun en uzak
  noktasında (d=3.2) **2.948899**'a çıkıyor.
- **Parametreler bir ARAMA protokolünden gelir**, tek bir SCF'ten değil.
  Arama özeti panelde yazılıdır ve ayrışma gerçektir: sevk edilen senaryoda
  d=1.8/2.0/2.2'de 12 tohumun 12'si yakınsıyor, d=2.6'da 11'i, **d=3.2'de
  yalnızca 9'u**.
- **`t`'nin işareti bir AYAR SEÇİMİDİR** ve bu modül onu sabitlemez
  (`tests/tb/test_ayar_ve_isaret.py`). Aynı şey `Δ`'nın işareti için de
  geçerlidir; `DistanceLaws.at()` bu yüzden her zaman pozitif `Δ` döndürür.
- **HIZLANMA İDDİA EDİLMİYOR.** Bu modül HF/TB maliyet oranını **ölçmedi**.
  İddia edilen tek şey parametrelerin türetilebilir ve çapalanabilir
  olduğudur — ve bu cümle şeffaflık panelinde de yazılıdır.

#### Panel ve senaryo

```bash
python -m qlab.run scenarios/halka_indirgeme.yaml -o indirgeme.npz
```

Çıkış 0; panel Yöntem / Yaklaşımlar / İhmal edilen / Beklenen hata / Yakınsama
satırlarını basar. `Yakınsama` burada "SCF durdu" **demek değildir**: ölçüt,
modelin varsaydığı iki-siteli antiferromanyetik yapının her mesafede
gerçekten bulunmasıdır (`staggering_residual_max` < 1e-7; sevk edilen
senaryoda ölçülen **7.526e-09**).

Beyan metninin **tek kaynağı** `engine/tb/factory.py`dir ve bu bir disiplin
değil bir **test**tir: her beyan cümlesi `src/` ağacında tam bir kez geçer;
`tests/` tarafında da — kilidi kuran tek dosya ve dört standart terim dışında
— hiç geçmez. `run.py::_run_downfold` metnin bir harfini bile yeniden yazmaz,
yalnızca `provenance.as_lines()` basar. Gerekçesi bu depoda ölçülmüştü: aynı
blok bir kez `scf.py` ile `factory.py`de kopyalanmış, biri düzeltilip öteki
unutulunca panel iki farklı şey iddia etmişti.

Sayım denetiminin kendisi de bir tuzağı kapatıyor ve o tuzak ölçüldü: bitişik
dize sabitleri (`"birinci parca "` + `"ikinci parca"`) kaynakta **iki**
parçadır ama çalışma zamanında **tek** dizedir; ham metinde aramak 0 eşleşme
verir ve "kopya yok" diye okunurdu — yani test kendi iddiasının tersini
kanıtlardı. Denetim bu yüzden metni önce birleştiriyor.

### Modül 4 — spin Monte Carlo / manyetizma (Plan I)

#### Yol haritasının hedefi ölçümle geçersiz kılındı: **demir tautolojisi**

Yol haritası "demirin Curie sıcaklığı 1043 K" diyordu. Bu bir doğrulama
değil, bir **tautoloji**: 2B Ising modeli **tek** bir enerji ölçeği taşır ve
`T_c ∝ J`. Onsager'in 1944 tam çözümü oranı sabitler:

    kT_c / J = 2 / ln(1 + √2) = 2.269185314

Dolayısıyla SI birimindeki her `T_c` girilen `J`'nin geri okunmasıdır. Modül 4
kapanışında **yeniden ölçüldü** (`k_B = 8.617333262e-2 meV/K`, CODATA 2018):

| hedef | gereken `J` | o `J` ile geri okunan `T_c` | fark |
|---|---|---|---|
| **demir, 1043 K** | **39.608394 meV** | **1043.0000 K** | **0.000e+00** |
| nikel, 627 K | 23.810607 meV | 627.0000 K | 0.000e+00 |
| kobalt, 1388 K | 52.709924 meV | 1388.0000 K | 0.000e+00 |
| gadolinyum, 292 K | 11.088831 meV | 292.0000 K | 5.684e-14 |

Aynı kadran her malzemeyi verir; hedef seçilir, parametre ondan türetilir,
sonra parametre "sonuç" diye geri okunur. (Gadolinyum satırındaki 5.684e-14
kayan nokta yuvarlamasıdır, fizik değil. Tabloda gösterilen **altı haneye
yuvarlanmış** `J`'ler kullanılırsa fark 7.2e-07 … 7.8e-06 K olur — yani
"1043.0000 K" hanesini bozan tek şey, sayının kaç basamağının yazıldığıdır.) **Modül 3F'nin silisyum bant
aralığı hedefiyle aynı aile** ve çekirdek spec'in aynı tablosundaki
ölçümle-düzeltilmiş **ikinci** hedef. Hedef düzeltildi; yerine aşağıdaki
dört çıpa kondu ve **dördü de girdilerden geri okunamaz**.

#### Dört doğrulama hedefi ve ÖLÇÜLEN sayıları

| # | Çıpa | Ölçülen | Referans | Sınıf |
|---|---|---|---|---|
| 1 | Binder kesişimi → `kT_c/J` | **2.265050 / 2.265253 / 2.265570** (üç ikili) | Onsager **2.269185314**, simülasyona **söylenmedi** | dairesel değil |
| 2 | Kritik üs `β/ν` | **0.122941 ± 0.003225** | tam **0.125** (−0.64σ); ortalama alan **1.0** → **272.0 kat** marjla dışlandı | dairesel değil (evrensel) |
| 3 | Mermin-Wagner | Ising eğimi **+0.000002**, Heisenberg **−1.0545** (T = 1.0) | teorem: 2B'de sürekli simetri düzenlenmez, ayrık düzenlenir | dairesel değil (teorem) |
| 4 | Modüller arası bağ | 2B Ising **0.122941** ↔ Modül 3H UHF **0.4962** | mean-field ½ ile tam çözümün ⅛'i **dört kat** farklı | dairesel değil |

#### Onsager — çözünürlük tuzağı ve gerçek sistematik kayma

Planın "kesişim 2.260, sapma %0.4" iddiası **yeniden ölçüldü ve çürüdü**:
%0.4 mutlak olarak 0.0092'dir, yani planın kendi ızgara adımının (0.04)
**dörtte biri**. O sayı bir kesişim değil, bir **ızgara noktasıydı**.

Bu modülde kesişim **doğrusal ara değerlemeyle** bulunur ve ara değerlemenin
ızgara sistematiği **gürültüsüz** (tam sayımla) ölçülür: adım 0.04 →
−5.950e-04, **0.02 → −2.156e-04**, 0.001 → −4.2e-07. Sistematik, istatistiksel
σ'nın (3.0–5.2e-03) **on dörtte biri** — yani sınırlayıcı olan ızgara değil,
istatistik.

| çift (L × L') | kesişim | σ | Onsager'den sapma | 3σ |
|---|---|---|---|---|
| 8 × 16 | 2.265050 | 0.004118 | −0.004135 (−1.00σ) | 0.01235 |
| 8 × 24 | 2.265253 | 0.003020 | −0.003932 (−1.30σ) | 0.00906 |
| 16 × 24 | 2.265570 | 0.005206 | −0.003616 (−0.69σ) | 0.01562 |

Üçünün yayılımı **0.000519**, en dar 3σ'nın on yedide biri: **bu istatistikle
üçü ayırt edilemiyor**. Ama üçü de **aynı yönde** sapıyor ve tek bir tohum
bloğu bunu ayıramaz. Üç bağımsız tohum bloğu (20260824 / 20260825 / 19990101)
havuzlanınca:

    blok ortalamalari  2.265291 / 2.266747 / 2.267474
    havuz              2.266504 ± 0.000642
    Onsager sapmasi    -0.002681 = **-4.18σ**

Yani kayma **gerçek ve sistematiktir** (sonlu boyut), tek blokta gürültünün
içinde kalır. Kaymanın gerçekliği ayrıca gürültüsüz de gösteriliyor: `L = 3` ×
`L = 4` kafeslerin **tam sayımında** (512 ve 65536 durum, hata çubuğu yok)
kesişim **2.1692678**, Onsager'den **−0.0999 (−%4.40)**. Küçük `L`'de kayma
devleşiyor, `L ≥ 8`'de bu istatistiğin çözünürlüğünün altına iniyor.

#### Kritik üs — ve "β = 1/8 çıktı" yazmanın neden yasak olduğu

Pencere: `L` = 8, 12, 16, 24, 32 · `T = T_c` · Wolff · `n_burn = 300`,
`n_sweep = 1500` · K = 24 bağımsız tohumlu zincir · 3000 çekilişlik bootstrap.

    <|m|>: 0.775523 / 0.740615 / 0.715627 / 0.678801 / 0.654571
    β/ν  = 0.122941 ± 0.003225      (3σ = 0.009674)

Doğru cümle: **β/ν = 0.122941 ± 0.003225 ölçüldü; tam değer 0.125 (sapma
−0.64σ, yani belirsizliğin içinde); ortalama alan (β/ν = 1.0) 272.0 kat marjla
dışlandı.** Kalan sapma bir sonlu-boyut düzeltmesidir ve **sıfırlanmaz** — bu,
`L` alt kümeleri ayrı ayrı uydurulunca görünür: (8,12,16) → 0.121346,
(16,24,32) → 0.129350, (24,32,48) → 0.123540. Yani log-log doğrusu **tam doğru
değildir**; altı noktanın 0.125'e bu kadar yakın düşmesi düzeltmelerin bu `L`
penceresinde birbirini götürmesindendir.

`β/ν` ile `β` sayısal olarak aynıdır çünkü 2B Ising'de `ν = 1`. **Bu depo
`ν`yü ÖLÇMEZ**; `ν = 1` bağı literatürdendir ve beyan edilir.

#### Mermin-Wagner — aynı kod, zıt hüküm, **ve penceresi**

Pencere: `⟨|m|⟩`, `L` = 8, 12, 16, 24, 32 · `J = 1`, `h = 0` · **aynı**
Metropolis çekirdeği · `n_burn = 500`, `n_sweep = 1500` · tohum 20260822 ·
soğuk (tekbiçim) başlangıç.

| T | Ising eğimi | Heisenberg eğimi | m_I(32)/m_H(32) |
|---|---|---|---|
| **0.8** | +0.000033 | **−0.9196** | 6.99 |
| 1.0 | +0.000002 | **−1.0545** | 11.35 |
| 1.4 | +0.000174 | **−0.9580** | 16.04 |

Ising'in mıknatıslanması **dört hanede `L`'den bağımsız** (T = 1.0'da L = 8
için 0.999208, L = 32 için 0.999250); Heisenberg'inki güç yasasıyla sönüyor.
Farkı yaratan şey **kod değil simetridir** ve bunu `metropolis.py`nin
koşturduğu modeli **bilmemesi** garanti eder (üç ayrı testle kilitli).

**GEÇERLİLİK PENCERESİ — beyan ediliyor, gevşetme değil: çıpa `T ≳ 1.0`'da
kurulur.** `T = 0.8` ayrı bir parametre olarak ve gevşek sınırla girer, oran
iddiası orada **hiç kurulmaz**. Sebebi ölçüldü: 2B Heisenberg'de `ξ ~
exp(2πJ/T)` ile patlar ve `L ≫ ξ` koşulu `L = 32`'de sağlanmaz. İmzası da bu —
8 tohumda eğim `T = 0.8`'de **−0.8690 ± 0.0754** ile −1'e ulaşamıyor ve
yayılım `T = 1.4`'ünkinin (±0.0422) neredeyse iki katı; oran orada **5.52**'ye
kadar iniyor, yani "> 8.0" iddiası **yanlış olurdu ve kurulmadı**.

#### Hata çubuğu — deponun ilk hata çubuğu, ve ilk kuralı

**Naif SEM bu depoda hiçbir yerde raporlanmaz.** MC örnekleri ilişkilidir ve
`std/√N` gerçek hatayı sistematik olarak küçük gösterir. Ölçüldü (L = 32,
T = T_c, 8000 ölçüm; bağımsız referans = 200 zincirin (Metropolis) ve 100
zincirin (Wolff) ortalamalarının std'si):

| algoritma | naif SEM | bağımsız referans | naif kaç kat küçük | referansın kendi belirsizliği |
|---|---|---|---|---|
| Metropolis | 1.946079e-03 | 1.713639e-02 | **8.81×** | %5.0 |
| Wolff | 1.955908e-03 | 4.642595e-03 | 2.37× | %7.1 |

Planın **"10.4 kat"**ı bayattı; 200 zincirlik referansla **8.81 kat** çıkıyor.
Ve aynı ölçüm 24 zincirle yapılınca gerçeğin **0.80 katı** oluyor
(1.3666e-02): **referansın kendisi de bir tahmindir** — bu ders bu modülde
üç kez ayrı ayrı ödendi.

**Plato bayrağı ALGORİTMANIN DEĞİL KOŞUMUN özelliğidir.** Planın "Metropolis
plato yapmaz, Wolff yapar" anlatısı bir algoritma cümlesi olarak **yanlış**;
ölçüldü (T = T_c, 24 zincir):

| algoritma | pencere | plato |
|---|---|---|
| Metropolis | L = 8, N = 4000 | **24/24** |
| Metropolis | L = 16, N = 8000 | 23/24 |
| Metropolis | L = 32, N = 8000 | **0/24** |

Bayrak "bu algoritma iyi mi" değil, "**bu koşum** kendi otokorelasyonunu
çözecek kadar uzun mu" sorusunu cevaplar. Plato oluşmamışsa dönen sayı hatanın
kendisi değil bir **ALT SINIRIDIR** ve panel onu öyle işaretler.

Ölçütün kendisi de düzeltildi: planın "son üç blok boyutunun hataları %5
içinde mi" ölçütü, kendi gürültüsü %6.35–12.91 olan sayıları %5'lik bir
eşikle karşılaştırıyordu ve Wolff'un plato **yaptığı** pencerede 24 tohumun
yalnız **3'ünde** `True` diyordu — yani planın kendi zorunlu testi kendi
koduyla **21/24 kırmızı** olurdu. Yerine konan ölçüt bir **seçimdir**
(iki düğme taranarak bulundu) ve daireyi kıran şey AR(1) sınavıdır:
`τ_int` orada kapalı formdan bilinir ve ölçüt hiç ayarlanmadan doğru hükmü
veriyor —

    φ       tam τ     binning τ   oran    plato
    0.0       0.500       0.541   1.082   24/24
    0.5       1.500       1.552   1.035   24/24
    0.8       4.500       4.446   0.988   24/24
    0.95     19.500      16.333   0.838   10/24   (geçiş bölgesi)
    0.995   199.500      52.833   0.265    0/24

#### ξ'nin kırılması — ve düzeltilmiş hüküm

Plan `ξ`nin `T ≥ 1.0`'da kırıldığını üç negatif oranla (−0.0511 / −0.0125 /
−0.0383) yazmıştı. **Yeniden ölçüldü: işaret tohumdan tohuma değişiyor** (her
sıcaklıkta 2/5). Doğru hüküm *"orada oran negatiftir"* değil,
***"orada tahminci KARARSIZDIR"***.

Ve `None` hikâyenin yalnızca yarısıdır: tamamen **ilişkisiz** bir ±1 kafeste
(gerçek `ξ = 0`) 200 tohumun `L = 16`'da 123'ü, `L = 32`'de 124'ü `None`
veriyor — kalan **~%38'i sahte bir `ξ` üretiyor** (4.418 ve 7.791). Yani
pozitif dönüş de orada güvenilir değil. `max(oran, 0)` ile kırpmak **yasak**:
sıfır "korelasyon yok" diye okunur, oysa gerçek hüküm "bu tahminci burada bir
şey söylemez".

İki ayrı geçerlilik sınırı daha ölçüldü ve beyan ediliyor:

- **Tek konfigürasyon yetmez.** T_c, L = 32, 5 tohum: `ξ` = 3.644 / 17.735 /
  30.337 / 36.768 / 150.562 — **41 kat** yayılım. Aynı 5 tohumun 200'er
  konfigürasyonluk topluluğunda 26.668 … 33.078 (%11).
- **Düzenli fazda ham tahminci mıknatıslanmayı okuyor** ve topluluk bunu
  **çözmüyor**: T = 1.5, L = 32'de `ξ` = 613.966 … 1431.310, yani `ξ/L` 16–45.
  L = 32'lik bir kafeste `ξ = 800` bir korelasyon uzunluğu değildir.
- **"Bağlı" (connected) sürüm cebirsel olarak ölüdür**: tek konfigürasyondan
  site ortalamasını çıkarmak `Σ(s−s̄) = 0` yapar, `S(0)` tam sıfır olur ve oran
  **her zaman tam −1** çıkar (15/15 ölçüldü).

Bu yüzden sevk edilen senaryo `ξ`yi **raporlamaz** ve gerekçesini panelde
yazar.

#### İki algoritma, tek hüküm — ve çapraz kontrolün kendi kör noktası

Metropolis ile Wolff'un ortak hiçbir karar noktası yoktur (biri kabul/red
verir, öteki hiç vermez), ve altı pencerede birbirini doğruluyorlar (fark /
3σ oranları 0.109 … 0.481). Ama çapraz kontrol *"ikisi de aynı şekilde
bozuk"* ihtimalini dışlayamaz; onu **tam sayım** dışlıyor: `L = 3` (512
durum) ve `L = 4` (65536), enerji bağımsız yazılmış, sapmalar
**0.08 / 0.66 / 0.78 / 1.05 × SE**. `L = 3` ayrıca bir kapsam iddiasıdır —
Metropolis tek `L`'yi reddeder, Wolff koşar; o çıpa çapraz kontrolle
**kurulamazdı**.

#### Modüller arası bağ

Modül 3H, UHF indirgemesinde kademeli alanın çatallanma üssünü üç halkada
bağımsız olarak **0.496190 / 0.497600 / 0.497811** ölçtü ve **UHF bir
ortalama-alan yöntemidir**, yani o üs ½'dir. Modül 4 aynı depoda, aynı tür
geçişte, gerçek 2B Ising'i koşuyor ve **0.122941** ölçüyor. İki sayı birbirini
bağlamlandırıyor: **mean-field ile tam çözümün farkı artık bu depoda
ölçülmüştür** ve ikisi de dairesel olmaktan çıkıyor.

#### Panel ve senaryo

```bash
python -m qlab.run scenarios/spin_tarama.yaml -o spin.npz
```

Çıkış 0; panel **yedi** satır basar — Yöntem / Yaklaşımlar / İhmal edilen /
Beklenen hata / **Dairesellik** / Yakınsama / **Örnekleme**. Son ikisi bu
modülde `Provenance`a eklendi ve ayrı alan olmalarının gerekçesi
`provenance.py`de yazılı: dairesellik beyanı `Yaklaşımlar` listesine konsaydı
modülün varlık sebebi beş maddeden biri olarak akıp giderdi, örnekleme tanısı
`diagnostics`e konsaydı panelde **"✓ … ALT SINIR"** diye görünürdü — yani
uyarının yanında bir onay işareti.

`Yakınsama` burada "hesap bitti" **demek değildir**: ölçüt, raporlanan **her**
hata çubuğunun gerçekten bir hata çubuğu olup olmadığıdır. Sevk edilen
senaryoda ✗ basar ve bu doğrudur — T = 2.2 ve T = 2.4'te binning platosu
oluşmuyor. Ölçüldü (24 tohum, aynı pencere): T = 1.6 → 23/24, T = 2.0 →
16/24, **T = 2.2 → 0/24**, **T = 2.4 → 0/24**, T = 2.8 → 13/24,
T = 3.4 → 24/24. Yani işaretin **iki yönü de** aynı sevk edilmiş senaryodan
kilitlenebiliyor.

Beyan metninin **tek kaynağı** `engine/spin/factory.py`dir ve bu bir disiplin
değil bir **test**tir: on bir beyan cümlesinin her biri `src/` ağacında tam bir
kez geçer, `tests/` tarafında da kilidi kuran tek dosya dışında hiç geçmez.
`run.py::_run_spin` metnin bir harfini bile yeniden yazmaz. Tek kelimelik
istisna `ALT SINIR` işaretidir ve o da yazılmaz, fabrikadan **ithal edilir**.

#### Dürüst olumsuzlar

- **`ν` ölçülmedi.** `β/ν = β` yalnızca literatürden gelen `ν = 1` bağı
  yüzündendir.
- **Plato bayrağının gerçek bir geçiş bölgesi var** (AR(1)'de φ = 0.95 →
  10/24; farklı tohumlarla bağımsız yeniden koşumda 8/24 — tohum gürültüsü,
  aynı hüküm). Kusur değil, çözünürlük. Ve bayrak **muhafazakârdır**: `False`
  derken binning değeri gerçeğin 0.854 katı — yani "plato yok", *serideki
  kanıtın eksikliğini* söylüyor, sayının yanlış olduğunun ispatını değil.
- **Hiçbir tahminci `τ = 199.5`'i temiz okumuyor.** ACF de kırılıyor
  (φ = 0.995'te 0.85× ve ±%48 saçılım); N = 8000 o otokorelasyona yetmiyor.
- **Planın "naif Metropolis + doğrudan fit → 0.0973" iddiası SINANMADI** —
  kritik üs her yerde Wolff ile ölçüldü.
- **Spec ön ölçümünün `τ_int` = 53.78 (Metropolis) / 3.07 (Wolff)
  karşılaştırması bu modülde YENİDEN ÖLÇÜLMEDİ.** Sevk edilen senaryonun
  kendi aralığı ölçüldü (L = 16, 24 tohum: 0.68 … 6.44); "17.5 kat kritik
  yavaşlama" cümlesi spec'in penceresine aittir, buraya taşınmadı.
- **Bir test tam sayı (123/200) sabitliyor**; deterministiktir ama numpy'nin
  RNG akışı değişirse kırılır.
- **Sevk edilen senaryonun T = 2.0 ve T = 2.8 satırları testlerde
  kilitlenmez** (16/24 ve 13/24 — geçiş bölgesi). Tek akıştan okunan bir
  oranı fiziğe yazmak bu depoda üç kez düzeltilmiş bir hatadır.

#### Planın bayat çıkan iddiaları

| planın dediği | ölçülen |
|---|---|
| `β/ν = 0.1152` | **yeniden üretilemedi**; 0.122941 ± 0.003225 |
| Binder kesişimi **2.260**, sapma **%0.4** | o bir **ızgara noktası**; ara değerlemeyle 2.265050 / 2.265253 / 2.265570, havuzlanınca −4.18σ |
| "Metropolis plato yapmaz, Wolff yapar" | algoritma cümlesi olarak **yanlış**; Metropolis L = 8 / N = 4000'de 24/24 plato yapıyor |
| naif SEM **10.4 kat** | **8.81 kat** (200 zincirlik referans; referansın kendi belirsizliği %5.0) |
| `ξ` `T ≥ 1.0`'da kırılıyor (−0.0511 / −0.0125 / −0.0383) | işaret **tohumdan tohuma değişiyor**; doğru hüküm "tahminci **kararsızdır**", artı ~%38 sahte pozitif |
| plato ölçütü: son üç hata %5 içinde, `min_blocks = 16` | kendi zorunlu testini **21/24 kırmızı** yapardı |
| demir 1043 K için `J = 39.608394 meV` → birebir 1043.0000 K | **doğrulandı** (fark 0.000e+00) — planın yeniden üretilebilen tek sayısı |

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
| 1 | `pytest` tam yeşil, PySCF referans testleri dahil | **Kapandı** — **1159/1159 test yeşil** (`pytest tests/`, 585.36 s; Modül 4 kapanışında ölçüldü). Önceki ölçüm 971/971 idi (432.22 s, Modül 3H kapanışı) ve aşağıdaki `reference/` deneyi o günden de eskidir. Ayrıca `reference/` geçici olarak kaldırılıp tam süite yeniden koşuldu: testler ATLANMIYOR, KIRMIZI oluyor — ölçüldü, **128 failed + 132 error + 291 passed, 0 skipped** (toplam 551, 154.20 s). Yani 260 test kırmızıya dönüyor ve hiçbiri sessizce atlanmıyor. Dikkat: bu ikinci deney süite 551 testliyken yapıldı ve o günün sayılarını taşır; yeniden koşulmadı, sayıları bugünkü toplama uydurmak ölçüm değil uydurma olurdu. Not: PySCF bu makinede **kurulu değildir** — `reference/*.json` kayıtları PySCF 2.14.0 ile başka bir yerde üretilmiştir ve testler koşarken PySCF'e bağımlı değildir; "PySCF referans testleri dahil" ifadesi kayıtlarla karşılaştırmayı anlatır, canlı bir PySCF çağrısını değil |
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
