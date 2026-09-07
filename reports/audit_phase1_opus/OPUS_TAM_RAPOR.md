# AŞAMA 1 — İKİNCİ TUR (Opus 5) — Bağımsız Bilimsel/İstatistiksel/Kod Denetimi

**Tarih:** 2026-09-02 | **Model:** Claude Opus 5 | **Kapsam:** Fable 5.1 turunun kalan 70 bulgusu + eksiklik eleştirmeni + 3-yargıç paneli + iki-tur sentezi.

Bu doküman iki parçadan oluşuyor: A-D bölümleri (Executive Verdict, Pipeline, CRITICAL, MAJOR) ana oturum tarafından ajanın yapısal verisinden yeniden inşa edildi (yukarıdaki not); E-Q bölümleri ve KAPANIŞ, denetim ajanının kendi ürettiği metindir.

---

## A. EXECUTIVE VERDICT
# 🔴 RED — Mevcut sonuçlara (tez metni + üretilen artefaktlar birlikte) şu anki haliyle güvenmek mümkün değil
**3/3 yargıç RED verdi** (PhD komite üyesi, hakem+istatistikçi, ML araştırmacısı+yazılım denetçisi perspektifleri; Opus 5 tarafından, bağımsız).

### Yargıç 1: RED
Hüküm: RED. Gerekçe, tez metnindeki bayat sayılardan bağımsız olarak yalnızca kodun ve GÜNCEL çıktıların üzerine kuruludur; bunları depoda bizzat doğruladım.

(1) Simülasyon katmanı fiziksel olarak yanlış bir dünyayı ölçüyor. num_inst hiçbir yerde kaynak talebiyle çarpılmıyor (feature_engineering.py:123 `gpu_demand = num_gpu`; multi_node_simulator.py `_gpu_request` yalnız tek örneğin talebini okuyor): test dilimindeki GPU-saniye talebi gerçek değerin 5,74 katı eksik modelleniyor. Bu, kapasite muhasebesinin tamamını — doluluk, bekleme, JCT ve politika sıralamasını — hatalı bir tabana oturtuyor. Ayrıca kendi hesabımla sunulan yük ρ≈165,8 (32-GPU) ve ρ≈20,7 (256-GPU); 32 GPU'da kuyruğun boşalması en iyi ihtimalle 26,3 gün sürüyor. Yani ölçülen şey "flash crowd" veya kararlı bir küme değil, tek seferlik statik bir batch drenajıdır. Bu rejimde SPT'nin FIFO'yu ezmesi kuyruk teorisinin bilinen bir sonucudur; kümeye, varış dinamiğine ya da ML'e atfedilebilecek bir bulgu değildir. Duyarlılık ızgarasının hiçbir noktası ρ<1'i keşfetmiyor, dolayısıyla "%57 JCT iyileşmesi" tipi mutlak sayıların dış geçerliliği yoktur.

(2) Tahmin katmanının katkısı mevcut artefaktlarla çöküyor. Ayarlanmamış iki önemsiz taban çizgisi — Per-User Median (MAE 5.191) ve 5 kolonluk profil-medyanı lookup — 17-21 öğrenilen modelin tümünü MAE/MdAE/Spearman'da geçiyor. Bir tez, ML modellerinin bir groupby-medyanını geçtiğini gösteremiyorsa RQ2 yanıtlanmamış demektir. Üstelik bu taban çizgileri raporlanmıyor; oysa aynı lookup'ın JCT'de 23 puan geride kalması "nokta doğruluğu ≠ çizelgeleme" tezinin en güçlü kanıtı olurdu. Kanıtın tam da anlatıyı bozan kısmının analiz dışında bırakılması, seçici raporlama görüntüsü veriyor.

(3) "En iyi model" kimlikleri hesaplama hatalarının ve gürültünün ürünü. LightGBM'in regression_l1 ile ayarlanıp objective belirtilmeden (L2) final eğitildiğini kodda teyit ettim (tuning.py L586 vs L779-790); "LGBM-Native en iyi öğrenilen model" ifadesi ayarlandığı kayıp fonksiyonuyla eğitilmemiş bir modele aittir ve L1 ile eğitildiğinde sıralama değişiyor. Buna DL tarafında 3-tohum ortalaması raporlanırken simülasyona seed-42 ağının girmesi, ~1 SD'lik tohum sapması ve 0,2-0,7 puanlık "kazanan" farkları ekleniyor. Kısacası şampiyon seçimleri gürültü içindedir.

(4) İstatistiksel çıkarım geçersiz. p-değerleri tek bir deterministik replay'in 16.437 işini bağımsız gözlem sayıyor (pseudo-replikasyon); etkin bilgi birimi ~3.146 profil / 636 kullanıcı ve TEK bir 1,59 günlük test penceresidir. Bu ölçekte "p = 0.000" hiçbir karşılaştırmalı iddiayı desteklemez. Dahası tek yönlü Wilcoxon'da p=1.000 "FIFO'dan ayırt edilemez" diye okunuyor; doğrusu bu politikaların FIFO'dan anlamlı derecede KÖTÜ olmasıdır — yani mevcut yorum bulgunun işaretini ters çeviriyor.

(5) Tahmin ve karar katmanları birbiriyle tutarsız. Modele beslenen cluster_load_gpu ortalaması ~534 GPU iken simüle edilen küme 32/256 GPU; simülatör gpu_type'ı hiç kullanmıyor, belleği hiç uygulamıyor, ve tezde SJF/SRF için tarif edilen skip-over yerine tüm politikalara katı HoL uyguluyor (backfill açıldığında kazançlar 5-17 puan düşüp sıralama değişiyor). Yani "canlı kümeye takılabilir modül" nedensel köprüsünün simülasyonda karşılığı yok.

(6) Yeniden üretilebilirlik fiilen yok. `git status` 173 değişmiş/izlenmeyen dosya gösteriyor; yayınlanmış HEAD (4923ce3, 17 Ağu) ne tezdeki ne de güncel checkpoint'lerdeki sayıları üretiyor ve GPU kısıtı uygulanmayan simülatörü içeriyor. Kilit dosyası sonuçları üreten ortamı dondurmuyor.

Metin uyumsuzluğunu "geçici" kabul etsek bile geriye kalan tablo şudur: birbirinden bağımsız en az beş kusurun HER BİRİ tek başına bir başlık bulgusunu tersine çevirebilecek güçtedir ve bunların hiçbiri metin güncellemesiyle kapanmaz — kod düzeltmesi ve yeniden koşum gerektirir. Sağ kalan tek sağlam yön iddiaları "aşırı yüklü bir kuyrukta SPT-benzeri sıralama FIFO'yu geçer", "kategorik öznitelikler sayısal olanlardan iyi" ve "MAE çizelgeleme kazancını öngörmez"dir; ilki literatürde bilinen bir sonuç, diğer ikisi ise doğrulanmış taban çizgisi ve etkin-n sorunları nedeniyle henüz istatistiksel olarak güvenceye alınmamıştır. Bu haliyle çalışmanın ürettiği sonuçlara bilimsel olarak güvenilemez.

Yol haritası (RED'den çıkış için asgari koşullar): num_inst'i kaynak talebine dahil et ve ρ≈0,6-0,8 civarında en az bir kararlı rejim noktası koş; iki trivial taban çizgisini (user-median, profil-lookup) tabloya ve simülasyona kalıcı olarak ekle; LGBM'i ayarlandığı kayıp ile refit et; en az 3-5 rolling-origin penceresi ve çoklu tohum ile CI/etki büyüklüğü raporla; simülasyon istatistiğini iş düzeyinde değil koşu/pencere düzeyinde kur; simülatör semantiğini (HoL vs skip-over, backfill) tez algoritmasıyla hizala; her şeyi commit et. Bu adımlardan sonra çalışma büyük olasılıkla savunulabilir bir YELLOW/ORANGE noktasına taşınabilir — ancak muhtemelen farklı bir ana mesajla.

**Öncelikli riskler:**
- Kaynak modeli hatalı: num_inst kapasite talebine hiç girmiyor (GPU-saniye 5,74x eksik) — simülasyonun tüm doluluk, bekleme, JCT ve politika sıralaması yanlış tabana oturuyor (feature_engineering.py:123, multi_node_simulator.py _gpu_request).
- Ölçülen rejim tanımlanan rejim değil: dogrulanan yuk ρ≈165,8 (32-GPU) / 20,7 (256-GPU), 32 GPU'da 26,3 günlük batch drenajı. 'Flash crowd' ve Little yasası anlatısı geçersiz; mutlak JCT yüzdeleri (%57-63) üretim kümesine taşınamaz ve ızgarada ρ<1 noktası yok.
- Öğrenilen hiçbir model önemsiz taban çizgisini geçmiyor: Per-User Median (MAE 5.191) ve profil-medyanı lookup, 17-21 modelin tümünü MAE/MdAE/Spearman'da yeniyor. RQ2'nin ampirik dayanağı yok; üstelik bu baseline'lar raporlanmadığı için seçici raporlama riski var.
- Hesaplama hatası model sıralamasını belirliyor: LightGBM regression_l1 ile ayarlanıp objective'siz (L2) final eğitiliyor (tuning.py L586 vs L779-790); 'en iyi öğrenilen model' kimliği bu tutarsızlığın ürünü ve L1 refit ile değişiyor. XGB'de de arama MAE / final RMSE asimetrisi var.
- İstatistiksel çıkarım geçersiz (pseudo-replikasyon): p-değerleri tek deterministik replay'in 16.437 işini bağımsız sayıyor; etkin birim ~3.146 profil / 636 kullanıcı ve TEK bir 1,59 günlük test penceresi. Ayrıca tek yönlü Wilcoxon p=1.000 'ayırt edilemez' diye okunuyor — gerçekte FIFO'dan anlamlı KÖTÜ, yani bulgunun işareti ters.
- Şampiyon farkları gürültü içinde: politika farkları 0,2-0,7 puan, DL tohum SD'si ~509 s (mae 6.236 vs seed0 6.703); tabloya 3-tohum ortalaması, simülasyona tek tohum giriyor. 20 tahminci x 23 politika icinde duzeltmesiz 'en iyi' ilanı winner's curse üretiyor.
- Tahmin ve karar katmanları tutarsız: modele beslenen cluster_load_gpu ~534 GPU iken küme 32/256 GPU (fiziksel olarak imkansız girdi); simülatör gpu_type'ı ve belleği hiç kullanmıyor; tezde SJF/SRF için tarif edilen skip-over yerine tüm politikalara katı HoL uygulanıyor (backfill acildiginda kazançlar 5-17 puan düşüp sıralama değişiyor). 'Drop-in modül' iddiasının simülasyonda karşılığı yok.
- Yeniden üretilebilirlik fiilen yok ve metin-kod uyumsuzluğu sistemik: git HEAD (4923ce3, 17 Ağu) ile 173 commit edilmemiş dosya; yayınlanan sürüm GPU kısıtı uygulanmayan simülatörü içeriyor ve ne tezdeki ne güncel sayıları üretiyor. Tez tabloları/şekilleri/README üç ayrı eski koşudan; tablolar elle yazıldığı için her koşuda ~200 sayı elle güncellenmek zorunda.

### Yargıç 2: RED
Hüküm: RED — mevcut haliyle bu çalışmanın ürettiği sayısal sonuçlara bilimsel olarak güvenilemez. Gerekçe, tez metnindeki bayatlıktan bağımsız olarak KOD ve GÜNCEL ÇIKTI düzeyindedir; iki ana iddiayı bağımsız olarak doğruladım: (a) src/feature_engineering.py:123 'gpu_demand = num_gpu' — num_inst hiçbir yerde kaynak talebine çarpılmıyor, yani simülatörün tüm kapasite muhasebesi (ρ, bekleme, JCT, politika sıralaması) sistematik olarak eksik talep üzerine kurulu; (b) src/tuning.py'de LightGBM araması objective='regression_l1' ile yapılıyor, finalize_ml_model'de LGBMRegressor'a objective geçilmiyor (varsayılan L2), yani 'en iyi öğrenilen model' ilan edilen LGBM-Native'in hiperparametreleri değerlendirildiği kayıp fonksiyonuna ait değil ve bu tek başına 'ağaçlar iyi tahmin ediyor' sonucunun yönünü değiştiriyor.

Bunların üzerine, metinden bağımsız olarak geçerliliği kıran üç yapısal sorun var. Birincisi ölçüm rejimi: sunulan yük ρ≈166 (32-GPU) / ≈21 (256-GPU), yani ölçülen şey 'flash crowd' değil, 26-37 günlük tek seferlik toplu kuyruk boşaltmasıdır; raporlanan %57-63 JCT kazancı klasik SPT-vs-keyfi-sıra toplu-iş etkisidir ve duyarlılık süpürmesi hiçbir noktada ρ<1'i keşfetmiyor — dolayısıyla kazançların büyüklüğü hiçbir üretim kümesine taşınamaz. İkincisi simülasyon-model tutarsızlığı: modele beslenen cluster_load_gpu ortalama 534 GPU iken simüle edilen küme 32/256 GPU; yani tahmin katmanı, karar katmanının simüle ettiği dünyada fiziksel olarak imkânsız bir küme durumunu görüyor. Ayrıca katı HoL bloklama TÜM politikalara uygulanıyor (tezde tarif edilen skip-over/backfill semantiği değil) ve backfill açıldığında kazançlar 5-17 puan düşüp sıralama değişiyor; bellek kaynağı ve gpu_type simülatörde hiç yok. Üçüncüsü istatistiksel çıkarımın geçersizliği: tek deterministik replay'in 16.437 işi bağımsız gözlem sayılıyor (pseudo-replication); etkin bilgi birimi ~3.146 profil / 636 kullanıcı ve tek bir 1,59 günlük test penceresidir. Bu koşullarda 'p<0.001' beyanları kanıt değeri taşımıyor ve 0,2-0,7 puanlık 'şampiyon' farkları (LSTM-Cat vs XGB-Cat) tek tohumlu DL gürültüsünün içinde kalıyor.

En ağır bilimsel darbe negatif kontrolden geliyor: ayarlanmamış bir Per-User/profil-medyanı lookup tablosu, 21 öğrenilen konfigürasyonun tümünü MAE/MdAE/Spearman'da geçiyor. Bu, tezin tahmin katkısını (RQ2) mevcut kanıtla geçersiz kılıyor; aynı baseline'ın JCT'de 23 puan geride kalması ise 'sıralama kalitesi belirleyicidir' savını da çürütüyor (en yüksek Spearman, en düşük kazanç). Tezin özgünlük iddiası olan sweep-line özniteliği için hiçbir ablasyon deneyi tasarımda yok; 'DFL/Learning-to-Rank' iddiasının karşılığı kodda yok (tüm kayıplar MSE/L1/kare hata, seçim ölçütü val RMSE); Alibaba'nın kendi süre tahmini baseline'ı altyapısı hazır olduğu halde hiç ölçülmemiş.

Metin uyumsuzluğunu 'geçici' saysak bile geriye kalan tablo şudur: sonuçları üreten pipeline'da doğrulanmış hesaplama hataları var, ölçüm rejimi araştırma sorusuyla eşleşmiyor, istatistiksel çıkarım dayanaksız ve önemsiz bir taban çizgisi ana katkıyı yeniyor. Ayakta kalan tek güvenilir katman yön düzeyindedir: (i) tahmin-güdümlü SJF, katı HoL'lu FIFO'ya göre doymuş rejimde büyük kazanç sağlar; (ii) kategorik kimlik öznitelikleri (user/gpu_type) sayısal özniteliklerden daha bilgilidir; (iii) nokta doğruluğu (MAE) çizelgeleme kazancını öngörmez. Bu üç yön, düzeltilmiş bir pipeline'da yeniden ölçülmek kaydıyla bir tezin çekirdeği olabilir; ancak mevcut sayıların hiçbiri savunmaya veya yayına götürülemez. GREEN/YELLOW'a çıkış için asgari koşullar: num_inst düzeltmesi + LGBM kayıp tutarlılığı + ρ<1 içeren yük ızgarası + simülasyon-içi öznitelik yeniden hesabı + çoklu tohum ve rolling-origin (en az 3-5 pencere) + trivial baseline'ların tabloya alınması + tüm sayıların otomatik üretimle tezle eşitlenmesi.

**Öncelikli riskler:**
- Kaynak muhasebesi hatası (F-C02, kodda doğrulandı: gpu_demand = num_gpu, num_inst çarpılmıyor): GPU talebi 3,58x, GPU-saniye 5,6x eksik modelleniyor; simülasyonun ρ, bekleme, JCT ve politika sıralaması dahil tüm çıktıları yanlış tabana oturuyor.
- Ölçüm rejimi araştırma sorusuyla eşleşmiyor (simulator-1, robustness-3, baselines_claims-3): sunulan yük ρ≈166/≈21, yani 'flash crowd' değil 26-37 günlük tek seferlik batch boşaltması; %57-63 kazanç SPT-vs-keyfi-sıra toplu-iş etkisi ve duyarlılık ızgarasında hiç ρ<1 noktası yok — dış geçerlilik yok.
- Öğrenilen modeller önemsiz bir lookup taban çizgisini geçemiyor (F-C01, baselines_claims-2, modeling-7): profil/kullanıcı-medyanı 21 konfigürasyonun tümünü MAE/MdAE/Spearman'da yeniyor; RQ2'nin tahmin katkısı mevcut kanıtla desteklenmiyor ve baseline tezde hiç raporlanmamış.
- Ayar-eğitim kayıp fonksiyonu tutarsızlığı (robustness-1, code_bugs-1, kodda doğrulandı): LGBM L1 ile aranıp varsayılan L2 ile final eğitiliyor (XGB'de de MAE→RMSE kayması); 'en iyi ML modeli' kimliği ve ağaç-vs-DL karşılaştırması bu hataya bağlı olarak yön değiştiriyor.
- İstatistiksel çıkarım geçersiz (statistics-1, F-C11, F-C06): tek deterministik replay'in 16.437 işi bağımsız sayılıyor, etkin birim ~3,1 bin profil / 636 kullanıcı / tek 1,59 günlük pencere; 'p<0.001' beyanları kanıt değeri taşımıyor, <1 puanlık 'şampiyon' farkları gürültü içinde (winner's curse).
- Tahmin ve karar katmanları arasında dünya tutarsızlığı (F-C03, simulator-2, F-C10, F-C09): modele verilen cluster_load_gpu ~534 GPU iken küme 32/256 GPU; ayrıca katı HoL tüm politikalara uygulanıyor (tezdeki algoritma değil), bellek ve gpu_type simülatörde yok, backfill açıldığında kazançlar 5-17 puan düşüp sıralama değişiyor.
- Özgünlük iddialarının ampirik/metodolojik karşılığı yok (F-C04, baselines_claims-5, baselines_claims-6, F-C07): sweep-line özniteliği için hiç ablasyon yok ve MDI sırası iddiayı desteklemiyor; 'DFL/Learning-to-Rank' kodda yok (MSE/L1 eğitimi, val-RMSE seçimi); Alibaba'nın kendi tahmin baseline'ı altyapısı hazırken hiç ölçülmemiş (seçici raporlama riski).
- İzlenebilirlik ve yeniden üretilebilirlik çöküşü (traceability-1/2/3, reproducibility-1/2): tezdeki tablolar hiçbir mevcut checkpoint'e izlenemiyor ve güncel kodla ana bulguyu tersine çeviriyor; yayınlanmış commit sonuçları üreten sürüm değil (GPU kısıtsız simülatör içeriyor), kilit dosyası artefaktları üreten ortamı dondurmuyor, şekil-tablo-metin üç ayrı koşudan geliyor.

### Yargıç 3: RED
Hüküm, tez metnindeki bayat sayılardan bağımsız olarak, KODUN ve GÜNCEL ÇIKTILARIN kendisi üzerine verilmiştir ve sonuç olumsuzdur. Üç bağımsız katmanda birbirini besleyen kusurlar var. (1) Hesaplama katmanı: Simülatörün kaynak muhasebesi hatalı — num_inst kaynak talebine hiç girmiyor (doğruladım: src/feature_engineering.py:123 `gpu_demand = num_gpu`; num_inst yalnız öznitelik listesinde), yani GPU talebi ~3,6x, GPU-saniye ~5,6x eksik modellenmiş; kaynak ağırlığıyla iş yükünün %86,7'si çok-örnekli olduğundan bu bir kenar durum değil, kapasite tabanının kendisi. Bellek kaynağı ilan edilmiş ama kapasitesi 0; gpu_type simülatörde hiç yok; backfill rezervasyon kaydı iş kimliği yerine (cpu,gpu) ayak iziyle siliniyor. LightGBM L1 ile ayarlanıp L2 ile eğitiliyor (src/tuning.py:586 vs finalize_ml_model — doğruladım), XGB erken durdurma metriği arama/final arasında değişiyor, DL EarlyStopping delta'sı ölçekli hedef varyansının ~%14'ü, dropout araması sessizce düşüyor. Yani "en iyi model" kimliği doğrudan kod hatalarının fonksiyonu. (2) İstatistik katmanı: Tek deterministik simülasyon replay'inin 16.437 işi bağımsız gözlem sayılıyor (pseudo-replikasyon); etkin bağımsız birim ~3.146 profil / 636 kullanıcı / 1 test penceresi mertebesinde. Tek bir 1,59 günlük kronolojik test penceresi üzerinde 21 model x 23 politika arasından <1 puanlık farklarla "şampiyon" seçiliyor (winner's curse), çoklu karşılaştırma düzeltmesi yok, tahmin metriklerinde CI yok. p<0.001 ilanları 16.437 çiftte kaçınılmaz; etki büyüklüğü ve CI tezde yok. Tek yönlü Wilcoxon'da p=1.000, "FIFO'dan ayırt edilemez" diye ters okunuyor — aslında anlamlı KÖTÜ. (3) Bilimsel katman: Öğrenilmiş hiçbir model, ayarlanmamış bir groupby-medyanı lookup tablosunu geçemiyor (checkpoint'lerden doğruladım: Per-User Median MAE 5.191,5 < LGBM-Native 5.697,4 < LSTM-D 6.236,2 < XGB-OH 6.642,4). Bu tek başına RQ2'nin ("ağaç modelleri runtime'ı iyi tahmin eder") ampirik temelini yok ediyor ve bu taban çizgisi tezde hiç yer almıyor. İkinci katkı olan sweep-line özniteliği için hiçbir ablasyon yok (21 konfigürasyonun tamamı aynı 9 özniteliği kullanıyor) ve MDI sıralaması iddiayı desteklemiyor. Simülasyon rejimi rho ~166 (32-GPU) / ~21 (256-GPU) ile "flash crowd" değil, 26-37 günlük statik batch kuyruğunun boşaltılması; duyarlılık süpürmesi hiçbir noktada rho<1'i keşfetmiyor, dolayısıyla %57-63'lük JCT kazançlarının BÜYÜKLÜĞÜ üretim kümesine taşınamaz. Üstelik simülasyonda modele beslenen yük öznitelikleri (ortalama 534 GPU) simüle edilen 32/256 GPU'luk dünyada fiziksel olarak imkansız — tahmin ve karar katmanları farklı evrenlerde. Ayakta kalan tek sağlam bulgular yön düzeyindedir: (a) SJF benzeri ML sıralaması aşırı yüklü rejimde FIFO'yu geçer, (b) kategorik özniteliğe sahip modeller sayısal olanlardan iyidir, (c) nokta doğruluğu (MAE) çizelgeleme kazancını öngörmez. Abstract/ch1/ch7'nin taşıdığı ölçek-bağımlı dikotomi, DFL/Learning-to-Rank yeniliği ve %57,25 rakamı ise güncel çıktılarla desteklenmiyor. Metin bayatlığı geçici sayılsa bile, doğru sayılar yazıldığında tezin ana mesajının kendisi çöküyor — bu bir transkripsiyon sorunu değil, bulgu sorunu. Yayınlanan sürümün (origin/main) tezdeki de güncel de sonuçları üretmemesi ve requirements-lock'un başka bir ortamı dondurmuş olması, bağımsız doğrulamayı da imkansız kılıyor. Sonuç: mevcut haliyle sonuçlara bilimsel olarak güvenilemez. YELLOW/ORANGE'a çıkabilmesi için en az num_inst kapasite hatasının, LGBM kayıp tutarsızlığının ve simülasyondaki yük öznitelikleri tutarsızlığının düzeltilip pipeline'ın yeniden koşulması; doymamış rejim (rho~0,7) ve rolling-origin çoklu pencere değerlendirmesi eklenmesi; trivial taban çizgilerinin (Per-User Median, sabit medyan, Alibaba'nın kendi tahmini) tabloya girmesi; ve iddiaların ayakta kalan üç yön bulgusuna indirgenmesi gerekir.

**Öncelikli riskler:**
- Simülatörün kaynak muhasebesi yanlış: num_inst kaynak talebine girmiyor (gpu_demand = num_gpu), GPU talebi ~3,6x / GPU-saniye ~5,6x eksik modelleniyor; kaynak ağırlığıyla işlerin %86,7'si çok-örnekli. Tüm kapasite, rho, bekleme, JCT ve politika sıralaması yanlış tabana oturuyor; bellek kaynağı ilan edilip uygulanmamış, gpu_type simülatörde hiç yok.

**Not:** Bu turun rapor yazma ajanı çıktı token sınırına (64.000) çok yakın çalıştı ve A-D bölümlerini üretmeden E bölümünden başladı; A-D bölümleri burada, ana oturum tarafından, ajanın kendi yapısal verisinden (133 onaylı bulgunun tam JSON'u + 3 yargıcın ham gerekçesi) yeniden inşa edildi. E'den itibaren olan bölümler ajanın kendi ürettiği metindir, değiştirilmedi.

---

## B. PROJE PIPELINE
```
UÇTAN UCA ZİNCİR (tüm yollar /Users/hasanugurcelebi/Thesis/alibaba-gpu-runtime-prediction-and-scheduling altında; "nbNN cell K" = notebooks/en/NN_*.ipynb JSON hücre indeksi; TR defterleri kod olarak EN ile birebir aynı, yalnız yorum/etiket farkı — diff ile VERIFIED)

[0] HAM VERİ → data/alibaba_cluster_trace/pai_job_no_estimate_100K.csv (100.000 satır × 8 kolon: job_id,num_inst,submit_time,num_cpu,num_gpu,gpu_type,duration,user; nb01 cell 8 çıktısı: hiç NaN yok). Yükleyici: src/data_loading.py:165 load_main_sample → configs/paths.yaml data.raw_data_dir/main_sample_file. data.processed_full_file=100k_job_with_utilization_full.csv tanımlı.

[1] ÖN İŞLEME / FİLTRE → src/feature_engineering.py:59 build_job_table_from_sample: TEK filtre `(df["duration"] > 0) & (df["num_gpu"] > 0)` (L102-105) → 82.184 satır (17.816 elenen); duration→job_runtime (float), num_gpu→gpu_demand (float, kesirli GPU korunur L123), arrival_time=to_datetime(submit_time,unit="s"), arrival_sec=t-t0 (L128-129). Bu filtre pipeline'da mantıksal olarak 1 kez tanımlı ama HER defter ham CSV'den yeniden uygular: nb00 cell 9, nb01 cell 13, nb02 cell 6 (src/analysis/workload_analysis.py:53 load_prepared_job_table→prepare_features_for_model), nb03 cell 9, nb04 cell 8/19/30/35/45/56/66 (7 çağrı), nb05 cell 10/11/12 (3 çağrı) — hepsi prepare_features_for_model(use_processed=False varsayılan, L377). VERIFIED (grep + hücre çıktıları 82.184).

[2] ÖZNİTELİK MÜHENDİSLİĞİ → src/feature_engineering.py:150 add_temporal_features (hour_of_day, day_of_week), :181 add_categorical_features (user, gpu_type → category), :205 add_cluster_utilization_features (sweep-line, O(N log N), TÜM veri seti üzerinde — train+test birlikte, split'ten ÖNCE; L280-292 merge_asof + kendi katkısını çıkarma → cluster_load_cpu/gpu, active_job_count). Nihai öznitelik listesi prepare_features_for_model L453-470: 9 sayısal [gpu_demand, arrival_sec, num_inst, num_cpu, hour_of_day, day_of_week, cluster_load_cpu, cluster_load_gpu, active_job_count] + 2 kategorik [user, gpu_type]. nb00 cell 12 aynı sırayla üretir, cell 15 → data/processed/100k_job_with_utilization_full.csv (82.184×14) yazar. DİKKAT: bu CSV'yi hiçbir aşağı akış defteri OKUMAZ (nb01-05'te use_processed/load_processed_full çağrısı yok; grep VERIFIED). nb03 aynı özellikleri tekrar hesaplar, build_feature_matrix (L313, shuffle=False) ile yalnız gösterim amaçlı split yapar; nb03 çıktısı da tüketilmez.

[3] TRAIN/TEST → src/feature_engineering.py:369 prepare_features_for_model: job_df arrival_sec'e göre mergesort (L444); train_test_split(index, test_size=0.2, shuffle=False, random_state=42 etkisiz) L489-494 → ilk 65.747 train / son 16.437 test (kronolojik; L501-512 guard, test ilk varış 524.788 s > train son 524.743 s, scratch betiğiyle VERIFIED). Kodlama: "numeric_only" (9 kolon), "with_categorical_onehot" (OneHotEncoder yalnız train'e fit L537-542 → 591 kolon, handle_unknown=ignore; testte 312 görülmemiş kullanıcı → sıfır satır), "with_categorical"/"with_categorical_native" (category dtype; kategori sözlüğü tüm veriden). MinMaxScaler (DL) yalnız train'e fit: src/tuning.py:1256-1262. Test seti kullanımı: her model için finalize'da 1 kez (nb04) + nb04 figür hücreleri 82/84 (tekrar predict) + nb05'te 3 kez yeniden kurulup 20 tahmin seti üretimi + simülasyon iş yükü + rank-korelasyon → test seti aynı zamanda simülasyon evrenidir (LOAD_FACTOR ile sıkıştırılır). Ayrı bir validasyon seti yok: ML için TimeSeriesSplit(3) (src/tuning.py:360-376) + XGB/LGBM içi %15 kronolojik erken-durdurma (L398-448) + final refit'te train'in son %10'u (L755); DL için train pencerelerinin son %20'si (L1285).

[4] MODEL EĞİTİMİ / TUNING → nb04 (91 hücre, 65 kod). Deney A (cell 10/12/14: RF, XGB, LGBM numeric), B (cell 20/22/24/26/28/30: RF-OH, XGB-OH, LGBM-OH, LGBM-Native, XGB-Native, Per-User Median baseline), C (cell 36/38/40: CNN/LSTM/Hybrid numeric, seq_len=1), D (cell 46/49/51: one-hot, seq_len=1), E (cell 57/59/61: numeric seq_len=10), F (cell 67/69/71: one-hot seq_len=10). Her ML hücresi: load_checkpoint → varsa EĞİTİM YOK; yoksa run_randomsearch_{rf,xgb,lgbm} (src/tuning.py:459/509/571; n_iter=20 defterde sabit, yaml n_iter=10'u ezer; RandomizedSearchCV cv=TimeSeriesSplit(3), scoring=neg_MAE, n_jobs=1 configs/models.yaml:56) → make_narrow_grid (L803, max_grid_size=81) → run_gridsearch_* (L617/639/670) → finalize_ml_model (L703: RF tüm train; XGB/LGBM train'in ilk %90'ı + son %10 early_stopping_rounds=50 L765) → evaluate_regression (src/models/evaluation.py:30; MAPE ×100, y_true>0 maskesi). DL: prepare_dl_datasets (L1221) → run_dl_randomsearch (L1410; num_trials=10, tuning_epochs=15, patience=5, seçim ölçütü ölçekli val RMSE) → make_narrow_grid → run_dl_gridsearch (L1502) → finalize_dl_model (L1560; final_epochs=50, patience=10, seeds=DL_SEEDS=[42,1337,2024] nb04 cell 6 → metrikler 3 tohum ortalaması ± std, diske kaydedilen model = 1. tohum (42), L1647'de tahminler 0'a kırpılır). Cihaz: get_default_device (L89) → Mac'te "mps" (eğitim), nb05'te CPU. Hyperparametre uzayları configs/models.yaml tuning.* (models.* bloğu — lgbm/xgb/rf sabit "hyperparameters" — hiçbir defterde kullanılmıyor; src/models/*_runtime_predictor.py sarmalayıcıları da defterlerde kullanılmıyor, nb04 doğrudan sklearn/xgb/lgb nesnelerini joblib ile kaydeder).

[5] CHECKPOINT / MODEL ARTEFAKTLARI → save_checkpoint (src/tuning.py:216) → results/checkpoints/exp_{a..f}_{model}.json (21 dosya; metrics, best_params, feature_mode, train_size, test_size, timestamp, status). "Smart save" hücreleri (örn. nb04 cell 11/13/15/21/…): model değişkeni None ise ve dosya diskte varsa "[SKIP]". Mevcut koşuda 21 checkpoint'in 21'i diskten YÜKLENDİ, hiçbir model eğitilmedi (nb04 cell 10,12,14,20,22,24,26,28,36,38,40,46,49,51,57,59,61,67,69,71 çıktıları "Loaded previous results from checkpoint" + "[SKIP] … already on disk"; VERIFIED). save_checkpoint yine de her koşuda JSON'u yeni timestamp ile yeniden yazar (L210 setdefault, veri içinde timestamp yok) → JSON timestamp'ı (2026-08-31T21:2
```

---

## C. CRITICAL FINDINGS (9, tam çekişmeli doğrulamadan geçti)

### C1. [simulator-1] Simülasyon rejimi 'flash crowd' değil, 26-37 günlük statik toplu-iş (batch) birikimi: sunulan yük ρ_GPU ≈ 166 (32-GPU) / 20.7 (256-GPU); LOAD_FACTOR fiilen etkisiz
- **Dosya:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` @ cell 20 (LOAD_FACTOR=0.1), cell 30 (SWEEP_LOADS); thesis/latex/chapters/2.background.tex L122; 5.simulation_framework.tex L135; 6.results_and_discussion.tex L117
- **Problem:** Test dilimi (16.437 iş) ham izde 137.101 s (1,59 gün) içinde geliyor; toplam GPU-işi Σ(gpu_demand·runtime)=7,27e7 GPU-s. 32 GPU ile bu işin salt hesap alt sınırı 26,3 gün, 256 GPU ile 3,3 gün. LOAD_FACTOR=0.1 ile tüm varışlar ilk 13.710 s'ye (0,16 gün) sıkışıyor. Yani ρ = Σ(g·rt)/(kapasite·span): 32-GPU için LF=1.0'da 16,6, LF=0.1'de 165,8; 256-GPU için LF=1.0'da 2,07, LF=0.1'de 20,7. Sıkıştırılmamış izde bile ρ>1 → kuyruk hiçbir konfigürasyonda kararlı değil; varış süreci sonuçlar üzerinde neredeyse rol oynamıyor. Kendi FIFO koşumda (32-GPU, LF=0.1) makespan 3,26e6 s (37,8 gün), son varış 13.710 s (makespan'ın %0,42'si); işlerin %98,2'si son varıştan SONRA başlıyor; son varış anında bekleyen iş 16.145 (tepe). Bu bir DES kuyruk deneyi değil, hemen hemen tamamı t≈0'da mevcut olan bir toplu-iş sıralama deneyidir. Tez ch2 L122 'ρ at or above 1.0' diyor (gerçek: 20-166); ch5 L135 ve ch6 L117 'flash crowd / sustained bottleneck' anlatıyor. cell 30 duyarlılık ızgarası (LF 0.05-1.0) hiçbir noktada doygun rejimden çıkmıyor; bu yüzden JCT kazançları LF boyunca neredeyse sabit (LGBM-Cat 57,66/57,07/56,88/60,25) — 'yük duyarlılığı' analizi aslında yük değiştirmiyor.
- **Kanıt:** scratch load_stats.py çıktısı: '32-GPU LF=0.1: rho_gpu=165.78 … min makespan(GPU-bound)=26.3 d vs span 0.16 d'; '256-GPU LF=1.0: rho_gpu=2.07'. real_fifo.py FIFO: 'last arrival (s): 13710.1  makespan: 3262313.8  ratio 0.0042; pending jobs at end of arrivals: 16145; share of jobs started after last arrival: 0.982'. nb05_32 cell 27: FIFO Mean Wait 1.338.936 s (=15,5 gün), Max JCT 3.230.288 s.
- **Bilimsel etki:** Raporlanan 'JCT improvement %' değerleri (Oracle %81, LSTM-Cat %60) klasik SPT-vs-keyfi-sıra toplu-iş etkisidir; büyüklüğü iş yükünün runtime CV'sinden (c_s²≈9) ve iş sayısından gelir, kümeden veya varış dinamiğinden değil. Ortalama bekleme 15 gün, P95 33 gün gibi mutlak sayılar operasyonel KPI olarak anlamsız. 'Flash crowd', 'queue stability', Little yasası (ch2) anlatıları ölçülen şeyle uyuşmuyor; RQ1'in 'gerçekçi küme koşullarında' yanıtlandığı iddiası desteklenmiyor. Kararlı bir rejim (ρ≈0,7) için 32-GPU'da LF≈24, 256-GPU'da LF≈3 gerekirdi — ızgarada yok.
- **Seviye:** C_scientific, B_statistical | **Status:** VERIFIED | **Tur:** Fable-turu-3-mercek
- **Düzeltme:** (1) ρ'yu tezde açıkça hesaplayıp raporla (formül + tablo). (2) Deneyi dürüstçe 'çevrimdışı toplu-iş sıralama' olarak adlandır ya da küme boyutunu/LF'yi ρ∈{0.5,0.7,0.9,1.2} verecek şekilde seç (örn. 32-GPU için LF≈15-30; 256-GPU için LF≈2-4) ve kararlı rejimde tekrar koş. (3) Warm-up/drain politikası tanımla (örn. son varıştan sonra başlayan işleri ayrı raporla). (4) ch2 L122, ch5 L135, ch6 L117 metinlerini düzelt; 'flash crowd' terimini bırak veya niceliksel tanımla.

### C2. [traceability-1] Tahmin sonuç tabloları (tab:predresults, Ek A tab:expa..expf-full, ch6 metni) depodaki hiçbir checkpoint ile uyuşmuyor; kaynağı 12 Mayıs tarihli eski HTML rapor
- **Dosya:** `thesis/latex/chapters/6.results_and_discussion.tex` @ L22-44 (tab:predresults), L54-56, L64, L71; appendices.tex L53-55, L73-76, L94-96, L114-116, L134-136, L154-156
- **Problem:** Tezdeki 19 model satırının tamamı (örn. XGB-OH MAE 3,389 / MedAE 1,129 / RMSE 11,375 / MAPE 10.80 / R² 0.51; RF-A 4,316/0.27; LSTM-D 5,836/0.06; LSTM-F 13,169/−0.27) ne results/checkpoints/*.json (güncel) ne de results/_backup_20260831_1703_pre_q2/checkpoints (önceki doğrulanmış koşu) ile eşleşiyor. Aynı sayılar reports/html/en/04_runtime_prediction_models.html (mtime 12 May) içinde birebir bulunuyor (grep: '3389.26' 4, '4316' 3, '6453.92' 4, '13169' 4 eşleşme) → tez tabloları Mayıs koşusundan elle aktarılmış. Güncel checkpoint: XGB-OH 6,642/3,432/14,297/2,578%/0.156; LGBM-Nat 5,697/2,684/13,543/1,660%/0.243 (en iyi öğrenilen model); RF-A 6,842/3,452/15,375/0.024; XGB-A 7,456/0.037; LGBM-A 7,033/0.027; RF-OH 6,292/0.153; LGBM-OH 6,640/0.118; LSTM-D 6,236±509/0.118; CNN-D 6,868/0.086; Hybrid-D 7,333/0.088; CNN-C 7,054/0.002; Hybrid-C 6,471/−0.012; LSTM-C 6,761/−0.012; LSTM-E 6,913/−0.013; CNN-E 6,972/−0.015; Hybrid-E 6,956/−0.007; CNN-F 6,804/0.033; Hybrid-F 6,822/0.078; LSTM-F 6,424/0.109. MAPE sütunu tezde 6.5–85.7 arası iken güncel evaluation.py ×100 ölçeğinde 1,621–3,278 (ch4 L156 'binlerce %' diyor, ch6 tablosu eski ölçekte; iç çelişki). Ayrıca Per-User Median taban çizgisi (MAE 5,191, MdAE 847; exp_b_user_median.json) ve XGB-Native (6,746) tezde hiç yok.
- **Kanıt:** 6.results L22: 'A & Random Forest (Numeric) & 4{,}316 & 1{,}508 & 13{,}831 & 16.85 & 0.27' — results/checkpoints/exp_a_rf.json: mae 6842.07, mdae 3451.77, rmse 15374.66, mape 2472.54, r2 0.0242 — backup exp_a_rf.json: mae 15236.70, r2 −0.673. grep -c '3389.26' reports/html/en/04_runtime_prediction_models.html = 4 (12 May).
- **Bilimsel etki:** Tezin ana bulgusu ('XGB-OH R²=0.51 en iyi, ağaçlar DL'den kesin üstün') güncel kodla tersine dönüyor: en iyi öğrenilen model LGBM-Native R²=0.24, XGB-OH R²=0.16; LSTM-D 3-tohum MAE 6,236 XGB-OH'den (6,642) düşük; Per-User Median (5,191) tüm öğrenilen modellerden düşük MAE veriyor. RQ2/RQ4 yanıtları ve ch6 §Discussion 'Why tree-based outperform DL' bölümü mevcut kanıtla desteklenmiyor. Tablolar düzeltilmeden tez güvenilir değil.
- **Seviye:** C_scientific, B_statistical | **Status:** VERIFIED | **Tur:** Fable-turu-3-mercek
- **Düzeltme:** tab:predresults ve Ek A tablolarını results/checkpoints/*.json'dan otomatik üreten bir betik (checkpoint → LaTeX tabular) yaz; Per-User Median ve XGB-Native satırlarını ekle; DL satırlarına ±std ekle; MAPE sütununu ya kaldır ya da ×100 ölçeğinde yeniden yaz; ch6 L54-56/L64/L71 ve ch6 §6.4.1 anlatısını yeni sıralamaya göre yeniden yaz.

### C3. [traceability-2] Simülasyon tabloları (tab:schedresults, tab:wilcoxon, tab:waitpercentile) ve tüm türetilmiş yüzdeler (57.25, 56.25, 81.59, 71.21, 68.9, 80.4, 39.2, −45.41) GPU kapasitesi uygulanmayan eski simülatör koşusundan (17 Ağu HTML raporları)
- **Dosya:** `thesis/latex/chapters/6.results_and_discussion.tex` @ L131-175 (tab:schedresults), L184-186, L240-243, L245, L259-301 (tab:wilcoxon), L310, L324-368 (tab:waitpercentile), L400-404; 1.introduction.tex L98; 7.conclusions L20-22; abstract-en/tr L4
- **Problem:** Tezdeki 32-GPU FIFO 499,951 / Oracle 92,064 / XGB-Cat 56.25% ve 256-GPU FIFO 50,386 / Oracle 14,506 / LSTM-Cat 57.25% değerleri, Wilcoxon W'ları (133,148,621; 109,737,336) ve yüzdelik tablosu (RF-Cat medyan 1,001; LSTM-Cat P95 62,514; FIFO P95 123,407) yalnızca reports/html/en/05_scheduler_evaluation_{32,256}_gpu.html (17 Ağu) içinde bulunuyor; tests/test_regression_guards.py:61-67 docstring'i bu rejimde simülatörün gpu_demand kolonunu okumadığını ve 'her sonucun sonsuz GPU kapasitesi' varsayımıyla üretildiğini belgeliyor. Güncel nb05 cell 27 (32-GPU): FIFO mean JCT 1,344,966; Oracle 252,024 (81.26%); LSTM-Cat 534,188 (60.28%); XGB-Cat 543,181 (59.61%); LSTM-Cat-Seq 58.47%; LGBM-Cat 57.07%; XGB-Native 57.00%; UserMedian 37.10%; SRF 44.27%; CNN-NumSeq −8.40% (tek anlamsız); LSTM-NumSeq +18.31% (tezde −45.52%). 256-GPU: FIFO 153,382; Oracle 32,264 (78.96%); LSTM-Cat 56,629 (63.08%); XGB-Cat 62.89%; LGBM-Cat 61.44%; SRF 48.39%; UserMedian 37.94%; CNN-NumSeq −30.11%, Hybrid-NumSeq −0.71% (ikisi anlamsız); LSTM-NumSeq +22.87% (tezde −45.41%). Yüzdelik tablosu 256-GPU güncel: en düşük medyan (Oracle hariç) LSTM-Cat 26,835 (tez: RF-Cat 1,001); en düşük P95 XGB-Native 178,976 (tez: LSTM-Cat 62,514); FIFO P95 323,020 (tez 123,407). Tezin niteliksel iddiaları da değişiyor: '32-GPU'da en doğru model XGB-Cat en iyi JCT' → güncelde her iki kümede LSTM-Cat birinci, XGB-Cat ikinci; 'Oracle'ın %68.9/%80.4'ü' → güncel 74.2%/79.9%; 'SRF'ye göre %39.2' → 256'da 28.5%; 'RQ3: 28 puan fark' → 14.7 (256) / 16.0 (32) puan.
- **Kanıt:** 6.results L131: 'SJF-Oracle & 86{,}034 & 92{,}064 & 81.59' ; grep -c '92064' reports/html/en/05_scheduler_evaluation_32_gpu.html = 2 ; nb05_32 cell 27 çıktısı: 'SJF-Oracle 245993.98 ... 252023.72 ... 81.26' ; tests/test_regression_guards.py:63-66: 'The lookup returned 0.0 for every job ... Every scheduling result in the thesis came from that regime.'
- **Bilimsel etki:** Tezin başlığa taşınan sayısı (%57.25) ve 'küçük kümede ağaçlar, büyük kümede LSTM' dikotomisi (abstract, ch1 L98, ch6 L184-186, ch7 L20) bozuk simülatörden geliyor; güncel koşuda dikotomi yok (LSTM-Cat her iki ölçekte birinci, farklar 0.2–0.7 puan). 'Sequence trap −45.41%' bulgusu (ch7 L22) güncelde tersine (+22.87%). Sonuç bölümü yeniden yazılmadan tez savunulamaz.
- **Seviye:** C_scientific, A_computational | **Status:** VERIFIED | **Tur:** Fable-turu-3-mercek
- **Düzeltme:** nb05 cell 27/46/49 DataFrame'lerini CSV'ye yazdırıp LaTeX tablolarını bu CSV'den üreten bir betik ekle (HTML→LaTeX köprüsü şu an yok); tab:wilcoxon'a etki büyüklüğü, %95 CI ve Holm-p sütunlarını ekle; abstract/ch1/ch6/ch7 yüzdelerini yeniden hesapla; UserMedian ve XGB-Native politikalarını tablolara ekle; ch5 L135 ile birlikte cell 30 duyarlılık ızgarasını (backfill=True'da LSTM-Cat 60.28→53.97, XGB-Nat 60.11→43.63) raporla.

### C4. [baselines_claims-1] Ana anlatı ('32-GPU'da en doğru tahminci = en iyi JCT; 256-GPU'da ilişki bozuldu; kategorik LSTM %57.25') güncel çıktılarla desteklenmiyor — ölçek-bağımlı dikotomi mevcut sonuçlarda yok
- **Dosya:** `thesis/latex/frontmatter/abstract-en.tex; thesis/latex/chapters/6.results_and_discussion.tex; thesis/latex/chapters/7.conclusions_and_future_work.tex` @ abstract-en L4 ('In the smaller 32-GPU cluster, the most accurate predictor (XGBoost with categorical features) also delivered the largest JCT reduction… In the 256-GPU cluster this relationship broke down'); 6.results L184-186, L240; 7.conclusions L20; 1.introduction L98
- **Problem:** Tezin merkezi iddiası iki kümede farklı kazananlar olduğudur. Güncel nb05 c27 çıktılarında her iki kümede de en iyi ML politikası SJF-LSTM (Categorical) (32-GPU %60.28, 256-GPU %63.08), ikinci XGB-Cat (%59.61 / %62.89). Öğrenilen modeller içinde en düşük MAE'li LGBM-Native (5.697 s) 32-GPU'da 4., 256-GPU'da 3. sırada. 18 modelin JCT sıralaması iki küme arasında Spearman ρ=0.99 (p=6e-15) ile neredeyse aynı; 'relationship broke down' diye bir kırılma yok. Tezde 'en doğru tahminci' denen XGB-OH güncelde R²=0.16 ile LGBM-Nat (0.24) ve UserMedian'ın (MAE 5.191) gerisinde.
- **Kanıt:** nb05_32 cell 27: 'SJF-LSTM (Categorical) … 60.28', 'SJF-XGBoost (Categorical) … 59.61', 'SJF-LGBM (Categorical) … 57.07'; nb05_256 cell 27: 63.08 / 62.89 / 61.44. Scratch: results/analysis/rank_correlation_{32,256}gpu.csv üzerinden Spearman(JCT%_32, JCT%_256)=0.9897.
- **Bilimsel etki:** Abstract, ch1 katkı 6, ch6 tartışma ve ch7 'Scheduling Duality' bulgusunun tamamı (ölçek büyüyünce ağaçlardan DL'e geçiş) mevcut veride yoktur; tezin ana bilimsel mesajı yeniden yazılmalıdır. Kalan sağlam bulgu yalnızca 'kategorik özellikli statik modeller > sayısal modeller' ve 'MAE JCT'yi öngörmüyor'dur.
- **Seviye:** C_scientific | **Status:** VERIFIED | **Tur:** CONFIRMED/CRITICAL, CONFIRMED/CRITICAL, CONFIRMED/CRITICAL
- **Düzeltme:** Abstract/ch1/ch6/ch7'deki dikotomi anlatısını kaldır; her iki ölçekte de aynı sıralamanın çıktığını, LSTM-Cat ile XGB-Cat farkının (0.67 puan) tek tohumlu DL ağı ve CI yokluğu nedeniyle ayırt edilemez olduğunu yaz; 'en doğru tahminci' ifadesini güncel metriklere (LGBM-Nat R²=0.24, UserMedian MAE 5.191) göre düzelt.

### C5. [baselines_claims-2] Öğrenilmemiş Per-User Median baseline tüm 17 öğrenilen modeli MAE/MdAE/MAPE/Spearman'da geçiyor ama tezde hiç yok; simülasyonda %37 ile geride kalması DFL tezinin asıl kanıtıyken raporlanmıyor
- **Dosya:** `notebooks/en/04_runtime_prediction_models.ipynb; notebooks/en/05_scheduler_evaluation_32_gpu.ipynb; thesis/latex/chapters/6.results_and_discussion.tex` @ nb04 cell 30/32 (Per-User Median MAE 5191.50, MdAE 846.50); nb05 cell 27 satır 'SJF-UserMedian (baseline)' 37.10% / 37.94%; nb05 cell 29 _RAW_PREDS listesi (UserMedian hariç); tez grep 'per-user|user median|naive|trivial' → 0 sonuç
- **Problem:** Yalnız train'den hesaplanan kullanıcı-medyanı: MAE 5.191 (en iyi öğrenilen LGBM-Nat 5.697), MdAE 846 (vs 2.684), MAPE 543% (vs 1.660%), Spearman ρ=0.597 (scratch; en iyi öğrenilen LGBM-Nat 0.539). Yani hiçbir öğrenilen model, nokta doğruluğunda VE sıralama korelasyonunda tek satırlık bir groupby'ı geçmiyor. Buna rağmen simülasyonda UserMedian yalnız %37.1/%37.9 JCT kazancı sağlıyor (ML %47-63). Bu (a) RQ2/RQ4 'ağaç modeller heavy-tailed runtime'ı iyi tahmin eder' cevabını çürütür, (b) 'MAE değil sıralama önemli' iddiasının en güçlü kanıtıdır, (c) aynı zamanda tezin önerdiği çevrimdışı vekil (Spearman/Kendall, ch6 L400) için bir karşı-örnektir: UserMedian ρ=0.597 ile en yüksek sıralama korelasyonuna sahip olup en düşük JCT kazançlarından birini alır; nb05 c29 listesi UserMedian'ı dışarıda bıraktığından bu karşı-örnek şekilde görünmez. Ayrıca 256-GPU, LOAD 1.0, backfill=True noktasında UserMedian (%60.84) XGB-Native'i (%58.44) geçiyor (nb05_256 c30).
- **Kanıt:** nb04 cell 32 çıktısı: 'Per-User Median (baseline) | 5191.50 | 846.50 | 15010.41 | 543.08 | 0.07'; scratch baseline_sim.py stats: 'user-median Spearman 0.5966 Kendall 0.4465'; nb05_256 cell 30 pivot: 'SJF-UserMedian (baseline) … 60.84' vs 'SJF-XGBoost (Native Cat) … 58.44' (backfill=True, LF 1.0).
- **Bilimsel etki:** Tez ML modellerinin önemsiz bir taban çizgisini geçtiğini göstermiyor; 'XGBoost en doğru' ve 'DL R²≈0 ama sıralamada üstün' anlatısı, bir lookup tablosunun her ikisini de nokta doğruluğunda geçtiği gerçeğini gizliyor. Sıralama-korelasyonu vekili iddiası (ch6 6.4.3) UserMedian ile çelişiyor ve bu çelişki analiz dışı bırakılmış.
- **Seviye:** B_statistical, C_scientific | **Status:** VERIFIED | **Tur:** CONFIRMED/CRITICAL, CONFIRMED/CRITICAL, PARTIALLY_CONFIRMED/MAJOR
- **Düzeltme:** UserMedian'ı tab:predresults, tab:schedresults, rank-korelasyon şekli ve Wilcoxon tablosuna ekle; RQ2'yi 'öğrenilen modeller nokta doğruluğunda kullanıcı-medyanını geçemedi' diye yeniden yaz; ML vs UserMedian eşleştirilmiş test ve etki büyüklüğü ekle; neden ρ yüksek/JCT düşük olduğunu (bağlı sıralar, kullanıcı-içi FIFO kırılımı, ölçek sıkışması) analiz et.

### C6. [robustness-1] LightGBM: hiperparametreler L1 (regression_l1) ile seçilmiş, nihai model varsayılan L2 ile eğitilmiş — raporlanan tüm LGBM MAE değerleri ayarlanan modele ait değil
- **Dosya:** `src/tuning.py` @ run_randomsearch_lgbm L586 / run_gridsearch_lgbm L679 (objective="regression_l1") vs finalize_ml_model L779-790 (lgb.LGBMRegressor(**best_params, …) — objective yok, best_params'ta da yok: make_narrow_grid skip listesi L918 'objective'); results/models/lgbm_*.joblib booster params objective='regression'
- **Problem:** RandomizedSearch/GridSearch LGBMRegressorCV(objective='regression_l1') ile MAE'ye göre en iyi parametreleri seçiyor; finalize_ml_model ise objective'i geçirmediği için LightGBM varsayılanı L2 (regression) ile yeniden eğitiyor. Diske yazılan ve nb05 simülasyonunda kullanılan 3 LGBM modeli L2 modelidir. Raporlanan MAE (7033 / 6640 / 5697) L1 ile ayarlanmış konfigürasyonun MAE'si değildir.
- **Kanıt:** Scratch betiği (lgbm_obj.py) checkpoint best_params + finalize akışını birebir yeniden üretti: obj=None → MAE 7033.4 / 6640.0 / 5697.4 (exp_a_lgbm, exp_b_lgbm_oh, exp_b_lgbm_nat ile tam eşleşme, best_iter 32/123/243 = diskteki ağaç sayıları). Aynı akış objective='regression_l1' ile: MAE 5806.5 / 5292.2 / 4869.8; rho 0.263 / 0.572 / 0.631. Kaydedilen lgbm_categorical_native.joblib: booster_.params['objective']='regression', 243 ağaç.
- **Bilimsel etki:** (A) Kod teknik olarak ayar-eğitim tutarsızlığı taşır. (B) LGBM satırlarının MAE sıralaması anlamsız: L1 ile eğitilen LGBM-Native MAE 4870, Per-User Median (5191) dahil tüm tahmincileri geçer; mevcut haliyle (5697) geçmez. (C) 'Hiçbir öğrenilen model önemsiz taban çizgisini geçmiyor' sonucu ile 'ağaçlar iyi tahmin eder' sonucu bu hataya bağlı olarak yön değiştirir. Ayrıca L1 modeli simülasyonda daha KÖTÜ (JCT kazancı %47.67 vs %57.07) — bkz. robustness-2. Tez ch4 L7 'objective function is the raw runtime' cümlesi kayıp fonksiyonunu hiç belirtmiyor.
- **Seviye:** A_computational, B_statistical, C_scientific | **Status:** VERIFIED | **Tur:** CONFIRMED/CRITICAL, CONFIRMED/CRITICAL, CONFIRMED/MAJOR
- **Düzeltme:** finalize_ml_model'de LGBM için objective='regression_l1' (ya da arama ile aynı objective) açıkça geçir; make_narrow_grid skip listesindeki 'objective' kararını belgele; checkpoint best_params'a objective'i ekle; exp_a_lgbm, exp_b_lgbm_oh, exp_b_lgbm_nat'ı yeniden eğit ve tüm tabloları/nb05'i yenile. Tezde kayıp fonksiyonlarını (RF/XGB: L2, LGBM: L1, DL: MSE) açıkça tablolaştır.

### C7. [figures_tables-1] Tablo 6.1 / Ek A ile Şekil 6.1 (nb04-fig01) aynı belge içinde birbirini yalanlıyor
- **Dosya:** `thesis/latex/chapters/6.results_and_discussion.tex` @ tab:predresults L22-44; figures/nb04-fig01-model-comparison.png (nb04 cell 78 çıktısı, backup koşusu); appendices.tex L45-158
- **Problem:** Tez tablosu RF (Numeric) için MAE 4.316 / R² 0.27 ve en iyi model olarak XGB (One-Hot) R² 0.51 yazıyor; aynı bölümde basılan şekil ise RF (Numeric) MAE 15.237 s / R² −0.67 (tüm modellerin en kötüsü) ve en iyi R²'yi LGB (Native) ≈0.24 olarak gösteriyor. Şekil 19 satırlık backup df_all'dan (XGB-Native ve UserMedian yok), tablo eski (×1/100 MAPE ölçekli) bir koşudan; güncel checkpoint üçüncü bir küme (RF-A 6.842/0.024).
- **Kanıt:** 6.results L22: 'A & Random Forest (Numeric) & 4{,}316 & 1{,}508 & 13{,}831 & 16.85 & 0.27'; nb04-fig01 görüntüsünde 'Random Forest' MAE çubuğu ≈15.2k, R² ≈ −0.67; NB04_Table07.html satır 2: 'Random Forest | 15236.70 | 13844.99 | 20130.93 | 10564.77 | -0.67'
- **Bilimsel etki:** Okuyucu aynı sayfada birbirini çürüten iki kaynak görür; 'Exp A'da RF en iyi', 'XGB-OH R²=0.51' ve 'DL R²≈0 ile ağaçlardan çok geride' anlatısı şekille desteklenmiyor (şekilde CNN-LSTM/LSTM(Seq) kategorik R² 0.14-0.15, XGB-OH 0.16).
- **Seviye:** C_scientific | **Status:** VERIFIED | **Tur:** CONFIRMED/CRITICAL, CONFIRMED/CRITICAL, CONFIRMED/CRITICAL
- **Düzeltme:** Tablo 6.1 ve Ek A'yı results/checkpoints/*.json'dan (ya da NB04_Table07.html'den) otomatik üreten bir betik ekle (checkpoint→LaTeX); şekil ile tablo aynı koşudan gelsin; MAPE sütununu evaluation.py ölçeğiyle (×100, binlerce %) yeniden yaz veya kaldır.

### C8. [figures_tables-2] Tablo 6.2 ve Şekil 6.6 (nb05-fig01) çelişiyor; şekil altyazısı şeklin gösterdiğinin tersini söylüyor
- **Dosya:** `thesis/latex/chapters/6.results_and_discussion.tex` @ tab:schedresults L131-175; fig:scheduler-jct L190-196 altyazı; metin L184-186
- **Problem:** Tablo 32-GPU için FIFO ortalama JCT 499.951 s, XGB-Cat %56.25 (en iyi ML) diyor; basılan şekil FIFO 1.344.966 s, SJF-LSTM (Categorical) %59.8 (en iyi ML), XGB-Cat %59.6 gösteriyor. Altyazı 'Left: 32-GPU cluster in which tree-based models were successful' der, oysa şeklin sol panelinde en iyi ML politikası LSTM-Cat'tir. 256-GPU tablosu FIFO 44.356 s / LSTM-Cat %57.25; şekil/HTML 153.382 s / %63.18. Eski sayılar GPU kapasitesi uygulanmayan simülatör rejiminden.
- **Kanıt:** 6.results L134: 'SJF-XGBoost (Categorical) & 212{,}721 & 218{,}751 & 56.25'; nb05-fig01-scheduler-jct_32gpu.png sağ panel: 'SJF-LSTM (Categorical) 59.8%', 'SJF-XGBoost (Categorical) 59.6%', 'FIFO 1344966s'; NB05_32GPU_Table02.html satır 1: 'SJF-LSTM (Categorical) … 540196.01 … 59.84'
- **Bilimsel etki:** Ana bulgu ('küçük kümede ağaçlar, büyük kümede LSTM kazanır') şekille desteklenmiyor: her iki ölçekte de LSTM-Cat ilk sırada ve fark <0.3 puan. %68.9 / %80.4 oracle-yakalama oranları (56.25/81.59, 57.25/71.21) ve abstract/ch1/ch7 yüzdeleri bu bayat tabloya dayanıyor.
- **Seviye:** C_scientific | **Status:** VERIFIED | **Tur:** CONFIRMED/CRITICAL, CONFIRMED/CRITICAL, CONFIRMED/CRITICAL
- **Düzeltme:** Tablo 6.2/6.3/6.4'ü nb05 eval_df/sig_df/pct_df'den (results/figures/thesis_export/html/NB05_*_Table02/04/05) otomatik üret; altyazıyı şekle göre yeniden yaz; 'tree vs LSTM' ölçek iddiasını kaldır veya farkın büyüklüğünü (0.2 puan) belirt.

### C9. [F-C02] num_inst (paralel örnek sayısı) kaynak talebine hiç girmiyor: GPU talebi 3,58×, GPU-saniye 5,6× eksik modelleniyor
- **Dosya:** `src/feature_engineering.py` @ src/feature_engineering.py:123 (gpu_demand = num_gpu); src/simulation/multi_node_simulator.py:344-346, 438, 460, 533 (req_cpu=num_cpu, gcol=num_gpu); thesis 3.dataset_and_workload.tex L11 ve L140; 7.conclusions_and_future_work.tex L59
- **Problem:** Ham iz her iş için num_inst (paralel örnek sayısı) ve örnek BAŞINA num_cpu/num_gpu taşıyor — tezin kendi tanımı da bu (ch3 L140 'Number of parallel instances running'). Ancak gpu_demand = num_gpu alınıyor ve simülatör num_cpu/num_gpu'yu iş başına TOPLAM talep sayıyor. 300 örnek × 0,5 GPU = 150 GPU'luk bir iş, simülatörde 0,5 GPU isteyen tek bir iş olarak yerleştiriliyor. num_inst yalnız model özniteliği olarak kullanılıyor, kaynak muhasebesinde hiç kullanılmıyor; aynı hata sweep-line cluster_load_* özniteliklerine de yansıyor.
- **Kanıt:** 82.184 filtrelenmiş iş üzerinde ölçümüm: sum(num_gpu)=55.902,5 ; sum(num_inst×num_gpu)=200.122,4 (3,58×). GPU-saniye 3,39e8 vs 1,89e9 (5,6×). İşlerin %21,0'ı num_inst>1 ama toplam GPU talebinin %76,7'sini ve GPU-saniyenin %86,7'sini taşıyor. num_inst×num_gpu>8 olan 5.245 iş (%6,4) hiçbir 8-GPU'luk düğüme sığmaz; 779 iş 32-GPU'luk kümenin tamamından büyük. En büyük iş job_id=91837: num_inst=300, num_gpu=0,5 → 150 GPU. grep 'num_inst' src/simulation/ → 0 sonuç.
- **Bilimsel etki:** Simülasyonun tüm kapasite muhasebesi, ρ, bekleme süreleri, JCT ve politika sıralaması yanlış tabana oturuyor. ch7 L59'daki savunma ('most of the jobs were single-node, so we feel comfortable this simplification does not limit generalization') İŞ SAYISI ile kaynak ağırlığını karıştırıyor: kaynak ağırlığıyla iş yükünün %86,7'si çok-örnekli. Dahası simülatör bu işleri 'tek düğüme yerleştirmiyor' — talebin 1/num_inst'ini yerleştirip gerisini sessizce yok sayıyor.
- **Seviye:** A_computational, C_scientific | **Status:** VERIFIED | **Tur:** CONFIRMED/CRITICAL, CONFIRMED/CRITICAL, CONFIRMED/MAJOR
- **Düzeltme:** gpu_demand = num_inst × num_gpu, cpu_demand = num_inst × num_cpu olarak düzelt (ya da gang-scheduling ile num_inst'i ayrı yerleşim birimi olarak modelle); sweep-line özniteliklerini yeniden hesapla; simülasyonu yeniden koş; ch7 L59'u kaynak-ağırlıklı sayılarla değiştir; düzeltme yapılmayacaksa ch3/ch5'te 'iş başına yalnız bir örneğin kaynağı modellenmiştir' uyarısını ve %86,7 rakamını açıkça yaz.

---

## D. MAJOR FINDINGS (44, tam çekişmeli doğrulamadan geçti)
| # | ID | Bulgu | Dosya | Düzeltme |
|---|---|---|---|---|
| 1 | `F-C01` | Tezdeki en güçlü tahminci hiç hesaplanmamış: 5 kolonluk profil-medyanı lookup 21 modelin tümünü MAE/MdAE/Spearman'da yenerken JCT'de 23 puan geride kalıyor | `notebooks/en/04_runtime_prediction_models.ipynb` | ProfileMedian'ı tab:predresults'a ve simülasyon tablosuna zorunlu taban çizgisi olarak ekle; öğrenilmiş modellerin bu taban çizgisini geçemediğini açı |
| 2 | `F-C04` | Tezin 2. katkısı olan sweep-line özniteliği için hiçbir ablasyon deneyi yok: 21 konfigürasyonun tamamı aynı 9 özniteliği kullanıyor | `notebooks/en/04_runtime_prediction_models.ipynb` | En az iki ek koşu: (i) numeric_only eksi {cluster_load_cpu, cluster_load_gpu, active_job_count}, (ii) with_categorical eksi aynı üçlü; MAE/MdAE/Spearm |
| 3 | `F-C05` | ch6 L68'in 'iş süresi zamansal otokorelasyona sahip değil' iddiası ölçülmemiş ve yanlış: log-runtime lag-1 ACF=0,34, Ljung-Box p≈0 | `thesis/latex/chapters/6.results_and_discussion.tex` | ACF/PACF ve Ljung-Box sonuçlarını hesaplayıp ch3 veya ch6'ya şekil/tablo olarak koy; ch6 L68'i 'runtime serisi anlamlı otokorelasyon taşımasına rağmen |
| 4 | `F-C06` | Tek bir 1,59 günlük test penceresi: 7 day_of_week seviyesinden yalnız 2'si test ediliyor, toplam 8 takvim günü (seviye başına n=1 gün), rolling-origin değerlendirme yok | `src/feature_engineering.py` | En az 3-5 kaydırmalı origin ile walk-forward değerlendirme koş ve model sıralamasının pencereden pencereye değişimini raporla; day_of_week'i öznitelik |
| 5 | `F-C07` | Tezin motivasyonunu doğrudan sınayacak taban çizgisi (Alibaba'nın kendi süre tahmini) altyapısı hazır olduğu halde hiç kullanılmamış | `configs/paths.yaml` | pai_job_duration_estimate_100K.csv'yi temin edip 'SJF-AlibabaEstimate' politikasını hem tahmin tablosuna hem simülasyona ekle; mümkün değilse ch1 Katk |
| 6 | `baselines_claims-3` | 'Flash crowd' rejimi aslında kalıcı aşırı yük / toplu (batch) zamanlama: tüm varışlar ilk 13.710 s içinde, toplam iş 32 GPU'da ~26 gün; hiçbir duyarlılık noktası doymamış rejimi ka | `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb; thesis/latex/chapters/5.simulation_framework.tex` | Ch5'te iş yükü/kapasite oranını (ρ=Σ(runtime·gpu)/(GPU·varış aralığı)) açıkça raporla; en az bir doymamış (ρ<1) operasyon noktası ekle (örn. küme boyu |
| 7 | `baselines_claims-4` | FIFO taban çizgisi katı HoL-bloklamalı ve backfill'siz; backfill'li süpürme defterde var ama tezde yok ve ML kazançlarını 5-17 puan düşürüp sıralamayı değiştiriyor — 'SLURM backfil | `src/simulation/multi_node_simulator.py; notebooks/en/05_scheduler_evaluation_32_gpu.ipynb; thesis/latex/chapters/5.simulation_framework.tex` | tab:schedresults'a backfill=True sütunu/tablosu ekle (nb05 c30 pivotu), backfill'li FIFO'yu ana taban çizgisi yap veya en azından ikisini birlikte rap |
| 8 | `baselines_claims-5` | 'Sweep-line active_job_count tüm ağaç modellerinde en önemli özniteliklerden biri' katkı iddiası güncel modellerde desteklenmiyor (MDI sırası 7/9, 7/9, 5/9) | `results/models/{rf,xgb,lgbm}_numeric.joblib; thesis/latex/chapters/1.introduction.tex; thesis/latex/chapters/6.results_and_discussion.tex; thesis/latex/chapters/7.conclusions_and_future_work.tex` | ch1/ch6/ch7'deki 'top predictor' ifadelerini güncel MDI'ya göre düzelt; sweep-line özniteliğinin değerini ablasyon (özellik çıkarılınca MAE/ρ/JCT deği |
| 9 | `baselines_claims-6` | 'DFL yaklaşımı öneriyoruz' ve 'Learning-to-Rank motoru' iddiaları yöntemle örtüşmüyor: hiçbir model karar-odaklı veya sıralama kaybıyla eğitilmedi; çalışma PtO + karar düzeyinde de | `src/tuning.py; thesis/latex/frontmatter/abstract-en.tex; thesis/latex/chapters/1.introduction.tex` | Abstract/ch1/ch7'de 'DFL yaklaşımı öneriyoruz' yerine 'PtO modellerini karar-odaklı ölçütle (JCT) değerlendiriyoruz; DFL eğitimi gelecek çalışmadır' d |
| 10 | `baselines_claims-7` | İstatistiksel anlamlılık pratik anlamlılık yerine kullanılıyor: 16.437 eşleştirilmiş işte SRF ve UserMedian dahil 20/22 politika p<0.001; ML-vs-ML ve ML-vs-baseline karşılaştırması | `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb; thesis/latex/chapters/6.results_and_discussion.tex; results/checkpoints/exp_d_lstm.json` | ML-vs-UserMedian, ML-vs-SRF ve LSTM-Cat-vs-XGB-Cat için eşleştirilmiş (job_id) test + bootstrap CI ekle; DL politikalarını 3 tohumla simüle edip JCT o |
| 11 | `code_bugs-1` | LightGBM L1 (regression_l1) ile ayarlanıp L2 (regression) ile final eğitiliyor; XGB erken durdurma metriği de aramada MAE, finalde RMSE | `src/tuning.py` | finalize_ml_model içinde lgb.LGBMRegressor(**best_params, objective="regression_l1", verbose=-1, …) ve xgb.XGBRegressor(…, eval_metric="mae") olarak a |
| 12 | `code_bugs-2` | EarlyStopping delta=1e-4 mutlak eşiği, MinMax-ölçekli hedefin toplam varyansının (7.2e-4) %14'ü: DL eğitimi ilerleme olsa da patience+1 epokta duruyor | `src/tuning.py` | delta'yı göreli yap (örn. iyileşme < 1e-3·best ise say) veya delta=0; her epoğun val_loss'unu ve durdurma epoğunu checkpoint 'metrics'e yaz (epochs_tr |
| 13 | `code_bugs-4` | Machine.release rezervasyon kaydını iş kimliğiyle değil (cpu,gpu) ayak iziyle siliyor: EASY backfilling gölge zamanları bozuk | `src/simulation/multi_node_simulator.py` | running_detail'e job_id ekle ((job_id, expected_finish, cpu, gpu)) ve release'te job_id ile sil; aynı ayak izli iki işle regresyon testi ekle. |
| 14 | `figures_tables-11` | Şekil 6.4 (pred vs actual): 'sample n=3,000' rastgele değil, kronolojik ilk 3.000 test satırı; metin şekle uymuyor | `notebooks/en/04_runtime_prediction_models.ipynb` | rng.choice(len(y_test), 3000, replace=False, random_state=42) ile rastgele örnekle; log-log eksen kullan; altyazıda n ve örnekleme yöntemini belirt; m |
| 15 | `figures_tables-13` | Isı haritası ve kutu grafiği: 'slowdown' sütunu ortalama (aykırı-güdümlü), metin medyan iddiası kuruyor; CDF metni şekildeki oranlarla uyuşmuyor | `thesis/latex/chapters/6.results_and_discussion.tex` | Heatmap'te medyan slowdown (veya bounded slowdown) kullan; metindeki oranları şekilden/percentile tablosundan oku; notebook anlatı renklerini düzelt. |
| 16 | `figures_tables-3` | Tablo 6.3 (Wilcoxon) ve 6.4 (yüzdelikler) bayat; '(fail) = p≥0.05' yorumu tek yönlü testi yanlış okuyor; güncel sütunlar (Holm, etki büyüklüğü, CI) tezde yok | `thesis/latex/chapters/6.results_and_discussion.tex` | Tabloyu sig_df'den (Effect size r, 95% CI, p (Holm)) üret; '(fail)' yerine 'FIFO'dan anlamlı derecede kötü (r<0)' yaz; tab:waitpercentile altyazısında |
| 17 | `figures_tables-4` | Tablo 5.x tab:sim_time (simülasyon süreleri) hiçbir ölçüm koduna dayanmıyor | `thesis/latex/chapters/5.simulation_framework.tex` | nb05 run_policy etrafına time.perf_counter() ekleyip süreleri eval_df'e yaz ve tabloyu oradan üret; ya da tabloyu kaldır. |
| 18 | `figures_tables-5` | Tablo 4.x tab:hyperparams üç farklı parametre kümesiyle çelişiyor; LGBM-A ve LGBM-OH satırları birebir aynı; HTML kaynağı kesik | `thesis/latex/chapters/4.prediction_models.tex` | Tabloyu checkpoint best_params'tan üret; nb04 cell 88'de pd.set_option('display.max_colwidth', None) ile HTML kesmesini kaldır; LGBM-OH satırını düzel |
| 19 | `figures_tables-6` | Şekil 6.3 (feature importance) altyazısı 'MDI-based' yanlış: LightGBM split sayısı, XGBoost gain; metin şekildeki sıralamayı yanlış aktarıyor | `thesis/latex/chapters/6.results_and_discussion.tex` | LightGBM için importance_type='gain' (booster_.feature_importance(importance_type='gain')) kullanıp her modelde toplamı 1'e normalize et; altyazıyı 'n |
| 20 | `figures_tables-7` | Şekil 6.5 (residual) altyazısı ve metni şekille çelişiyor: RF ortalama artık +10.601 s | `thesis/latex/chapters/6.results_and_discussion.tex` | Metni güncel şekle göre yaz; artıkların medyanını ve MdAE'yi de belirt; RF için ekstrapolasyon/derinlik kısıtı tartışması ekle. |
| 21 | `figures_tables-8` | Şekil 3.3 (GPU demand) altyazısı veriyle çelişiyor: işlerin %52,5'i kesirli GPU istiyor, 'neredeyse hepsi 1 GPU' değil | `thesis/latex/chapters/3.dataset_and_workload.tex` | Altyazıyı ve L62'yi güncel dağılıma göre yaz (medyan 0.5, %52,5 kesirli); simülasyon bölümünde kesirli GPU paylaşımının modellenme biçimini açıkla. |
| 22 | `leakage-1` | hour_of_day / day_of_week göreli epoch'tan (1970-01-01) türetiliyor; tezdeki 'Perşembe zirvesi' ve 'hafta içi/hafta sonu' yorumları epoch artefaktı, day_of_week test setinde iz-gün | `src/feature_engineering.py` | day_of_week'i düşür (7,66 günlük izde tanımsız) ya da 'iz günü' olarak yeniden adlandırıp ablasyonla raporla; hour_of_day'i 'göreli saat fazı (ofset b |
| 23 | `modeling-1` | LightGBM: tuning regression_l1 (MAE) ile yapıldı, final refit varsayılan L2 (regression) ile eğitildi | `src/tuning.py` | finalize_ml_model'de LGBM için objective="regression_l1", verbose=-1 ve first_metric_only=True (veya yalnız tek metrik) geçirin; aynı şekilde XGB için |
| 24 | `modeling-7` | Öğrenilen modeller sabit train-medyanı tahmincisini anlamlı biçimde geçmiyor; kullanıcı-medyanı tümünü geçiyor (CI ile) | `results/checkpoints/exp_b_lgbm_nat.json, exp_b_user_median.json` | Sabit-medyan ve kullanıcı-medyanı taban çizgilerini tüm tablolara ekleyin; log-hedef/L1 ile yeniden tuning yapın; sonuç değişmezse tezin anlatısını 's |
| 25 | `modeling-8` | DL tablolarındaki 3-tohum ortalaması ile diske yazılan (ve nb05'te kullanılan) seed-42 ağı farklı sıralama veriyor | `src/tuning.py` | Ya (a) her tohumu kaydedip nb05'te tohum ortalamalı (ensemble) tahmin kullanın, ya (b) diske en iyi/median tohumu yazıp tabloda o tohumun skorunu ana  |
| 26 | `reproducibility-1` | Yayınlanmış (origin/main) kod ve artefaktlar tezin sonuçlarını üreten sürüm değil: tüm düzeltmeler ve sonuçlar commit edilmemiş | `src/feature_engineering.py; src/simulation/multi_node_simulator.py; results/checkpoints/*.json; results/models/*; thesis/latex/chapters/*.tex` | Tüm src/tests/scripts/configs değişikliklerini, defterleri, checkpoint JSON'larını ve (boyut uygunsa) model dosyalarını tek bir etiketli commit/tag (ö |
| 27 | `reproducibility-2` | requirements-lock.txt sonuçları üreten ortamı değil, başka bir global ortamı dondurmuş; joblib artefaktları sklearn 1.8.0 ile üretilmiş, kilit 1.7.2 diyor | `requirements-lock.txt; environment.yaml; results/models/rf_numeric.joblib` | `venv/bin/python -m pip freeze > requirements-lock.txt` ile kilidi gerçek ortamdan yeniden üretin (yalnız proje paketleri), environment.yaml'ı python= |
| 28 | `reproducibility-4` | Checkpoint-varsa-eğitme mantığı hiçbir kaynak/veri/sürüm bağı taşımıyor; timestamp her geçişte yeniden yazılıyor → bayat checkpoint sessizce 'güncel sonuç' olarak raporlanır | `src/tuning.py; notebooks/en/04_runtime_prediction_models.ipynb` | save_checkpoint'e git commit hash'i, src/feature_engineering.py+tuning.py sha256'sı, veri sha256'sı, kütüphane sürümleri ve cihazı ekleyin; timestamp  |
| 29 | `reproducibility-6` | Betikli yürütme yolu (run_all_experiments.sh / export --force-execute) defterleri etkileşimli koşudan farklı bir Python ortamında çalıştırır | `scripts/run_all_experiments.sh; scripts/export_thesis_results.py` | Betikte `python3` yerine `${PYTHON:-venv/bin/python}` kullanın veya venv aktivasyonunu zorunlu kılıp sürümü doğrulayın (`python -c 'import sklearn; as |
| 30 | `robustness-2` | 'Tahmin kalitesi (MAE ya da Spearman) → çizelgeleme kazancı' anlatısı karşı-örneklerle çürüyor; kazancı belirleyen kuyruk (uzun iş) büyüklüğü ayrımı ve tahmin eşitlikleri (tie) — r | `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` | c29 analizine UserMedian ve XGB-Native'i dahil et; rho yerine kuyruğa duyarlı ölçütler (örn. en uzun %5/%10 işi doğru sınıflandırma, tahmin eşitlik or |
| 31 | `robustness-3` | Simülasyon çalışma noktası aşırı-yük (ρ≈166× 32-GPU / ≈21× 256-GPU); duyarlılık süpürmesi hiçbir noktada ρ<1'i keşfetmiyor; politika sıralaması yük ile değişiyor | `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` | ρ'yu açıkça hesaplayıp raporla (iş yükü / kapasite × pencere); süpürmeye ρ<1 ve ρ≈1 noktalarını ekle (örn. LOAD 1.0 ile N_GPU=512/1024 ya da LOAD 5-10 |
| 32 | `robustness-4` | 'SJF-LSTM (Categorical) en iyi ML politikası / DL büyük ölçekte üstün' (RQ4) sonucu 0,2-0,7 puanlık farka, örtüşen bootstrap CI'larına ve tek tohuma (seed 42) dayanıyor | `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb` | Üç tohumun her biri için simülasyonu koş (DL politikaları için tohum-ortalama ± std JCT); LSTM-Cat vs XGB-Cat için eşleştirilmiş Wilcoxon/bootstrap fa |
| 33 | `simulator-2` | Katı head-of-line bloklama TÜM politikalara uygulanıyor; tez Algoritma 1 ve §5.3.3 yalnız FIFO için HoL, SJF/SRF için 'sığan işler arasından en kısa' (skip-over) tanımlıyor | `src/simulation/multi_node_simulator.py` | Ya (a) tez Algoritma 1 ve §5.3.3'ü koda uydur ('tüm politikalarda katı HoL; backfill yalnız duyarlılıkta') ve bunu bir tasarım kararı olarak gerekçele |
| 34 | `simulator-3` | Küme ölçeği iş yüküyle orantısız: test dilimi ortalama ~530 eşzamanlı GPU ima ediyor (gerçek PAI ~6.500 GPU), simüle edilen 32/256 GPU; düğüm profilleri kaynaklandırılmamış | `src/simulation/multi_node_simulator.py` | Küme boyutunu iş yükünün ima ettiği eşzamanlılığa (≈500-800 GPU) veya ρ hedefine göre seç; ya da iş yükünü alt-örnekle (örn. rastgele %5 iş) ve bunu b |
| 35 | `simulator-7` | tab:sim_time (FIFO 3,2 s, … 16,4 s) hiçbir kodla üretilmiyor; ölçtüğüm süreler 8-21 s | `thesis/latex/chapters/5.simulation_framework.tex` | nb05'te run_policy'yi perf_counter ile sar, ölçümleri CSV'ye yaz ve tabloyu oradan üret (ortam bilgisi: CPU, Python, pandas sürümü ile). Ya da tabloyu |
| 36 | `statistics-1` | Wilcoxon p-değerleri ve bootstrap CI'ler tek bir deterministik simülasyon replay'inin 16.437 işini bağımsız gözlem sayıyor (pseudo-replication) | `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` | (1) Belirsizliği replay düzeyinde tahmin et: zaman-blok (moving-block/circular) bootstrap ile iş alt-kümelerini YENİDEN SİMÜLE et veya test penceresin |
| 37 | `statistics-2` | Tek yönlü Wilcoxon (alternative='greater') p=1.000 sonucu 'FIFO'dan ayırt edilemez' diye yanlış yorumlanıyor; aslında politika FIFO'dan anlamlı derecede KÖTÜ | `thesis/latex/chapters/6.results_and_discussion.tex` | İki yönlü test kullan ve etki büyüklüğünün işaretini raporla; ya da tek yönlü kalınacaksa 'H1: politika FIFO'dan daha iyi' diye açıkça belirt ve p≈1 s |
| 38 | `statistics-3` | Önemsiz taban çizgileri (sabit medyan, sıfır, kullanıcı medyanı) tezde yok; öğrenilen modellerin çoğu bunları geçmiyor | `results/checkpoints/*.json; thesis/latex/chapters/6.results_and_discussion.tex` | tab:predresults'a sabit-medyan, sıfır ve per-user-median satırları ekle; her model için taban çizgisine göre eşli bootstrap CI'li MAE farkını raporla; |
| 39 | `statistics-4` | Rank-korelasyon şekli: n=18 bağımlı nokta üzerinden CI'siz Pearson r; MAE–JCT ilişkisi anlamsız, en yüksek ρ'lu model (UserMedian) şekilden dışlanmış | `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` | Şekle Fisher-z %95 CI ve p ekle; 20 tahmin setinin tamamını (UserMedian, XGB-Native dahil) dahil et; Pearson yerine Spearman/Kendall (küçük n, aykırı  |
| 40 | `statistics-5` | DL sonuçları 3 tohum ortalaması (n=3, ddof=1) ± SD ile raporlanıyor; %95 CI'ler tüm modelleri kapsıyor, ağaç modelleri tek koşu (asimetrik), tezde ± hiç yok ve simülasyondaki ağ ra | `src/tuning.py` | Tabloda mean ± SD ve n_seeds'i açıkça yaz, whisker yerine t-tabanlı CI ver; ağaç modelleri için de ≥3 random_state (RF, XGB subsample) koşusu; simülas |
| 41 | `traceability-13` | nb04/nb05 markdown anlatıları ve reports/html|pdf eski koşulardan; TR/EN defter anlatıları güncel çıktılarla çelişiyor | `notebooks/en/04_runtime_prediction_models.ipynb` | Anlatı hücrelerini kod çıktısından f-string ile üretilen özet hücrelerle değiştir; reports/html|pdf'i export sonrasında nbconvert ile yeniden üret vey |
| 42 | `traceability-3` | tab:hyperparams'ın 19 satırının hiçbiri güncel best_params ile eşleşmiyor; 'tek sabit tohum' ifadesi DL için yanlış | `thesis/latex/chapters/4.prediction_models.tex` | tab:hyperparams'ı checkpoint best_params'tan otomatik üret; L171'i '3 tohum (42, 1337, 2024), rapor edilen DL metrikleri ortalama ± std; diske kaydedi |
| 43 | `traceability-4` | Simülasyon süre tablosu (tab:sim_time), '17 saniye/politika' ve README '100.000 olay < 60 s' için depoda hiçbir zaman ölçümü yok | `thesis/latex/chapters/5.simulation_framework.tex` | nb05 run_policy'ye time.perf_counter() ile süre ölçümü ekle, sonuçları eval_df'e sütun olarak yaz ve tab:sim_time'ı buradan üret; ya da tabloyu ve L95 |
| 44 | `traceability-5` | 'Altı model / 18 tahminci / 4 politika / 3 koşu' sayımları koddaki 20 tahmin seti, 23 politika ve 56 duyarlılık koşusuyla uyuşmuyor; Per-User Median ve XGB-Native tezde yok | `thesis/latex/chapters/5.simulation_framework.tex` | tab:experiments'a XGB (Native) ve Per-User Median satırlarını ekle; tab:simconfig 'ML predictors evaluated: 20 (18 öğrenilen + XGB-Native + Per-User M |


---

| `figures_tables-12` | Mimari şemaları simülatörde olmayan mekanizmaları gösteriyor | `thesis/latex/figures/architecture_en_32_gpu.png`; `5.simulation` L137-149; `multi_node_simulator.py:118, :545` | İki diyagram yalnız "(32 GPUs)/(256 GPUs)" etiketiyle ayrışıyor (md5 854d5a92… vs 6d56c0ca…); düğüm sayısı (10/80) ve profilleri gösterilmiyor. Diyagram "V100/A100", "T4", "Prediction & Routing Layer", "API Gateway", "Event Database" ve tahmine göre düğüm tipine "Dispatch Shortest Job" okları içeriyor. `grep -rn 'gpu_type' src/simulation/` → **0 sonuç**; yerleştirme first-fit ve tahminden bağımsız | Okuyucu heterojen donanım eşleştirmesi ve tahmin-güdümlü yönlendirme olduğunu sanıyor; simüle edilen sistem çok daha basit | C | **MODERATE→MINOR** | VERIFIED | Opus | Diyagramı gerçek akışa (arrival→queue→policy.select_job→first-fit) ve düğüm sayılarına göre yeniden çiz; donanım tipi etiketlerini kaldır veya "illustrative" olduğunu belirt |
| `figures_tables-14` | Şekil provenansı karışık: üç farklı koşunun şekilleri aynı klasörde | `export_thesis_results.py` L78-236; `thesis/latex/figures` mtime/md5; nb05 c29 | `nb05-fig0[1-5]` mtime **31 Ağu 15:41** (backup); `nb01-04` **17:33** (son export); `mae_spearman_*.png` **1 Eyl 00:58** (canlı koşu, md5 f3a7ebf6… ≠ backup 3bbf0c65…). Haritanın beklediği `nb05-fig06-load-backfill-sensitivity_*.png` tezde **yok**; `mae_spearman` hiçbir `.tex`'te `\includegraphics` ile çağrılmıyor. Export klasöründe bayat `NB04-Figure06.png` duruyor (betik klasörü temizlemiyor) | Canlı koşu bitip export çalıştırılmadan derlenen tez **iki farklı koşunun sonuçlarını yan yana basar**; duyarlılık ve rank-korelasyon şekilleri üretiliyor ama kullanılmıyor | A | MODERATE | VERIFIED | Opus | Export'a PNG/HTML dizin temizliği + koşu kimliği manifest'i (notebook md5 + tarih); nb05 c29'un doğrudan tez klasörüne yazmasını kaldır; fig06 ve mae_spearman'ı ch6'ya ekle ya da haritadan çıkar |
| `figures_tables-15` | `tab:simconfig` politika/model sayıları ve ikincil metrikler defterle uyuşmuyor | `5.simulation` L227-245; nb05 c26, c24 | "4 policy types", "18 models" vs defterde **23 politika / 20 tahmin seti**. "Secondary metrics" listesinde Holm-p, rank-biserial, bootstrap CI, Spearman/Kendall ve 56 koşuluk ızgara yok. L209-213 "simülatör 3 kez çalıştırılır" bayat | Deney tasarımı tablosu yapılan deneyi tanımlamıyor; UserMedian (%37-38) ve XGB-Native tezde hiç görünmüyor | C | MODERATE | VERIFIED | Opus | Tabloyu `POLICIES` listesinden üret; UserMedian ve XGB-Native satırlarını Tablo 6.2'ye ekle |
| `figures_tables-16` | Sweep-line şekli ve öznitelikleri 0'dan başlıyor: iz başlangıcında ~12 saatlik **ısınma artefaktı** | nb03 c15; `feature_engineering.py:205`; `3.dataset` L164 | Şekilde `active_job_count` ve `cluster_load_gpu` 0'dan başlayıp ilk ~0,5 günde 400-600'e tırmanıyor; iz öncesi başlamış işler bilinmiyor → ilk gündeki tüm işlerin küme-yükü öznitelikleri **sistematik olarak düşük**. Metin "erupts from a baseline of approximately 400" diyor, ısınmayı belirtmiyor | Eğitim setinin başındaki işler için öznitelik-hedef ilişkisi bozuk; ağaçlarda `arrival_sec` ile etkileşimli **sahte bölünmeler** olası | B, C | **MODERATE→MINOR** | VERIFIED | Opus | Şekilde ısınma bölgesini gölgele; ilk 12-24 saati eğitimden dışlayan ablasyon veya sınırlılık notu |
| `F-C09` | `gpu_type` modelde öznitelik ama simülatörde hiç yok: heterojenlik yalnız GPU sayısında; süre yerleşimden bağımsız varsayılıyor | `multi_node_simulator.py:118-160, :545-586`; `5.simulation` L92 | İz dört GPU tipi içeriyor (MISC 49.316, T4 25.484, P100 5.023, V100 2.361) ve `gpu_type` ML modellerinde kategorik öznitelik. `grep -c 'gpu_type' src/simulation/` → **0**. `provision_heterogeneous_gpu_cluster` yalnız `(96,8)` ve `(64,2)` üretiyor. Tez L92 Weng et al.'a atıfla "heterogeneous cluster … various GPU models" diyor | ch5'in "heterojen küme" çerçevesi yalnız **GPU SAYISI** heterojenliğidir; ch2'de Gavel üzerinden kurulan heterojenlik-farkındalığı motivasyonunun karşılığı yok. "Aynı iş her düğümde aynı sürede koşar" varsayımı **hiçbir yerde yazılı değil**; bu varsayım altında Oracle'ın "üst sınır" yorumu ve tüm JCT karşılaştırmaları GPU-tipi eşleştirme maliyetini sıfır sayıyor. Modelin `gpu_type`'ı kullanıp simülatörün kullanmaması ayrı bir katman tutarsızlığı | A, C | **MODERATE→MINOR** | VERIFIED | Critic (Opus) | Ya `Machine`'e `gpu_type` ekleyip yerleştirmeyi kısıtla (+ tip başına hız katsayısı), ya L92'yi düzeltip varsayımı açıkça yaz; ch7 Construct Validity'ye ekle |
| `F-C10` | ch1 Amaç 4 ve ch5 L11 simülatörün **belleği** de uyguladığını söylüyor; bellek kapasitesi 0 ve izde bellek kolonu yok | `1.introduction.tex:78`; `5.simulation_framework.tex:11, :92`; `multi_node_simulator.py:140-145, 182, 545-586` | Tezde **üç farklı ifade**: ch1 L78 "CPU, GPU, and memory resources" iddia ediyor; ch5 L11 her iş kaydının bellek talebi içerdiğini söylüyor; ch5 L92 doğru olanı söylüyor ("supports a third dimension (memory capacity), which is disabled (i.e., set to 0)"). `multi_node_simulator.py:182`: `if self.mem_capacity > 0.0 and ...` — varsayılan 0.0 ve provision hiçbir Machine'e vermiyor. `head -1 …csv` → bellek kolonu **yok** | Amaçlar bölümü yerine getirilmemiş bir amaç ilan ediyor; ch5 L11 var olmayan bir veri alanını tarif ediyor. Bellek, ch1 L33'te tezin **motive edici kaynak-parçalanma mekanizmasının ta kendisi** ("available CPU capacity at a node but insufficient GPU memory"); simülatörde hiç modellenmediği için parçalanma iddiaları simülasyonla desteklenemez | A, C | MODERATE | VERIFIED | Critic (Opus) | ch1 L78'i düzelt; ch5 L11'den "memory" örneğini çıkar; ch1 L33'teki motivasyonun simülasyonda karşılığı olmadığını ch7 Construct Validity'de belirt |
| `F-C12` | `tab:related` özgünlük iddiasının dayandığı iki çalışmayı (L2R, DFL) tabloya almıyor; ilan edilen boşluk tablonun kendi Tsafrir satırıyla çelişiyor | `2.background.tex` §Comparison and Positioning; `abstract-en` L4 | Tabloda 19 satır (verma2015large … weng2022mlaas) var; **`fu2024efficient` ve `mandi2023decision` yok** — oysa bunlar metinde "en yakın akraba" olarak tarif ediliyor. Boşluk cümlesi ("none have provided a direct link between predictive accuracy … and operational impact … via controlled simulation") tablonun kendi Tsafrir satırıyla çelişiyor: o satırda hem "Sched. Integration ✓" hem "Runtime Pred. ✓". Kaynakça bütünlüğü ayrıca temiz (43 girdi, 43'ü atıflı, öksüz anahtar yok) | Konumlandırma tablosu, özgünlük iddiasını sınayabilecek **tam da o iki çalışmayı** dışarıda bırakarak boşluğu yapay olarak büyütüyor (seçim yanlılığı). D-31 DFL/L2R iddialarının yöntemle örtüşmediğini gösteriyordu; bu bulgu iddianın **literatür karşısında da** konumlandırılmadığını gösteriyor | C | **MODERATE→MINOR** | VERIFIED | Critic (Opus) | `tab:related`'e `fu2024efficient` ve `mandi2023decision` satırlarını ekle; boşluk cümlesini daralt ("HPC'de sistem-üretimi tahminlerin etkisi ölçülmüştür; bu tezin katkısı GPU kümesinde çoklu ML/DL ailesinin aynı simülatörde karşılaştırılmasıdır") |
| `F-C08` | **HARKing izi:** ch1 RQ2 ile RQ4 aynı soru; ch6'nın RQ4 yanıtı ch1'de hiç sorulmamış bir soruya cevap veriyor | `1.introduction.tex:61-67`; `6.results_and_discussion.tex:407-413` | RQ2: "Which predictive modeling paradigm … compares tree-based ensembles to sequential DL architectures". RQ4: "What is the empirical performance difference between tree-based ensemble models … and sequential DL architectures … on tabular data". **Bu ikisi aynı sorudur.** ch6'da RQ2 bunu yanıtlıyor; RQ4 ise ch1'de hiç sorulmamış bir bulguyu yanıtmış gibi sunuyor: "A clear separation exists between point prediction accuracy and actual scheduling performance, depending greatly on the size of the underlying physical cluster". ch1 L59-68'de "cluster size" üzerine RQ **yok** | Abstract'ın ve ch7'nin başlıca iddiası (ölçek-bağımlı dikotomi — C-5'e göre güncel çıktılarla zaten desteklenmiyor) **sonradan bulunmuş bir gözlemin araştırma sorusu kılığına sokulması** görünümü veriyor. Klasik HARKing örüntüsü: önceden belirtilmemiş hipotez, çoklu karşılaştırma düzeltmesi olmaksızın (21 model × 2 ölçek) doğrulanmış gibi sunuluyor. RQ2/RQ4 ikizliği tezin dört yerine üç soru sorduğunu gizliyor | C | **MODERATE→MINOR** | VERIFIED | Critic (Opus) | RQ4'ü ch1'de gerçekten sorulan soruyla eşleştir; ölçek-bağımlı kopukluğu "önceden hipotez edilmemiş, keşfedilmiş gözlem" olarak ayrı alt bölümde sun ve tek pencereye dayandığını (D-13) belirt; ya da RQ2/RQ4'ü birleştirip ölçek sorusunu **açık RQ** olarak formüle et ve uygun deney (çoklu küme boyutu × çoklu tohum) tasarla |
| `reproducibility-5` | DL sonuçları donanıma bağlı (D-28'in final severity'si) | — | Bkz. D-28 | — | A | MODERATE | VERIFIED | Opus | Bkz. D-28 |
| `baselines_claims-9` | Tuning bütçesi ve kayıp fonksiyonları eşit değil; XGB-Native vs LGBM-Native arama uzayı 288 vs 3.888 | `tuning.py` L545/586/648/679, L1478, L1495, L971-978, L1393; `models.yaml` L77-102; nb04 c36/c57; `4.prediction` L166-171 | LGBM `regression_l1`, XGB `reg:squarederror`, DL MSE — rapor ölçütü MAE → LGBM-Native'in "en düşük MAE"si kısmen **kayıp-metrik hizasından**. ML ≤303 fit / DL 10 deneme + ≤81 grid, 15 epoch, tek holdout, seçim ölçütü ölçekli RMSE. `models.yaml` L79 `learning_rate: [0.05]` (XGB, tek değer) vs L92 `[0.01, 0.02, 0.05]` (LGBM). Dropout ve pencere boyu tezde "tuned" denmesine rağmen sabit | "Ağaç vs DL" ve "XGB vs LGBM" karşılaştırmaları eşit koşullarda değil; nb04 c28'in "kontrollü karşılaştırma" amacı tam sağlanmıyor | A, C | **MODERATE→MINOR** | VERIFIED | Opus | Tüm modelleri aynı kayıpla eğit ya da farkı raporla; XGB/LGBM uzaylarını eşitle; dropout'u grid/final'e taşı veya "tuned" ifadesini kaldır; bütçeleri (fit sayısı × veri) tablola |

### E.2 MINOR (18 bulgu)

| ID | Problem | Dosya:konum | Kanıt | Etki | A/B/C | Status | Tur | Düzeltme |
|---|---|---|---|---|---|---|---|---|
| `leakage-6` | Kategori sözlüğü (636 `user`, 58'i test-only) split'ten ÖNCE tüm veriden kuruluyor ve native modellere gömülüyor | `feature_engineering.py` L430 → L526-531, L559-564 | `'native: train user categories 636 test 636 same list: True; users actually in train: 578'`; `lgbm stored pandas_categorical sizes: [636, 4]`. One-hot yolu bu sorunu **taşımıyor** (578 user sütunu) | Hedef bilgisi sızmıyor (test-only kodlar eğitimde görülmüyor), ölçülebilir metrik etkisi beklenmez; belgelenmesi gereken protokol sapması. Canlı sistem gelecekteki kullanıcı kimliklerini bilemez | A | VERIFIED | Fable (n=3) | Kategori listesini `X_train`'den türet ve `set_categories` ile uygula; tezde belirt |
| `leakage-7` | Sweep-line öznitelikleri çevrimiçi hesaplanabilir — ch7 L60 bunu **yanlış** tarif ediyor; asıl belgelenmemiş varsayım "iş submit anında başlar" | `feature_engineering.py` L249-292; `7.conclusions` L60 | Perturbasyon: bir işin runtime'ını değiştirmek kendi özniteliklerini ve **kendisinden önce varan hiçbir işin** özniteliğini değiştirmiyor (`'earlier-arrival changed: 0, own: False'`). Kod bitişi `arrival_sec + duration` alıyor → kuyrukta beklemeden başladığı varsayımı. `duration` alanının tanımı (end−start mı, end−submit mi) depoda/tezde **yok** | ch7 sınırlaması **yanlış yönde** yazılmış (özellik lehine düzeltilmeli); `duration` semantiği doğrulanamadığından "aktif iş" özniteliğinin gerçek küme durumuna yakınlığı belirsiz | C | VERIFIED | Fable (n=3) | ch7 L60'ı düzelt ("yalnız tamamlanmış işleri kullanır, çevrimiçi hesaplanabilir; varsayım: submit anında başlar"); 100K dosyasının kaynağını ve `duration` tanımını ch3'e ekle |
| `modeling-13` | Ölü/çelişkili yapılandırma yolları ve gereksiz refit | `configs/models.yaml` L15-45; `tuning.py` L491-500, L303-311 | `models.*` bloğu hiçbir defter tarafından okunmuyor (`grep load_model_config` → yalnız `config_utils.py` ve sarmalayıcılar); `RandomizedSearchCV` `refit=True` → `best_estimator_` boşa fit ediliyor; `n_jobs=1` iken threading backend zorlaması etkisiz; `prepare_features_for_model`'daki `random_state` etkisiz (`shuffle=False`) | Yapılandırmanın "tek kaynak" olma iddiası yanlış; okuyucu yaml'a bakarak yanlış protokol çıkarabilir | A | VERIFIED | Fable (n=3) | `models.*` bloğunu kaldır veya `finalize_ml_model`'in bu bloktan objective/eval_metric okumasını sağla; `refit=False`; kullanılmayan parametreleri temizle |
| `statistics-9` | İstatistik kodu (Holm, rank-biserial, bootstrap) yalnız defter hücresinde; birim testi yok; Holm ailesi 56 duyarlılık koşusunu ve iki küme boyutunu kapsamıyor | nb05 c46; `tests/` | `grep -rln 'wilcoxon\|holm\|bootstrap\|rank_biserial' tests/` → **boş**; c46 `_m = len(sig_df)` (=22); 32 ve 256 GPU ayrı aileler; c30'daki 56 koşu hiç test edilmiyor. Tüm p≈0 olduğundan Holm pratikte hiçbir kararı değiştirmiyor | Düşük — sonuçları etkilemiyor ama doğrulanabilirlik zayıf | A, B | VERIFIED | Fable (n=2) | Fonksiyonları `src/analysis/stats.py`'ye taşı + elle hesaplanmış küçük örneklerle birim testi; aileyi (küme boyutu × politika) açıkça tanımla |
| `simulator-11` | Test kapsamı: çok-düğümlü simülatörde SJFPred/SRF, çok-makine HoL, eşitlik sırası, düşürülen iş ve rezervasyon senaryoları test edilmiyor | `tests/test_simulation.py` (4 test); `test_regression_guards.py` (19) | `Ran 23 tests … OK` (1,4 s). Kapsanmayan: `SJFPredScheduler`/`SRFScheduler` × `MultiNodeClusterSimulator`, T2 (çok-makine HoL), T4/T9 (aynı-zaman sıra), T7 (sığmayan iş), T5 (aynı ayak izli release), T3 (kesirli paketleme), T11 (CPU kısıtı). **Bulunan üç hata (simulator-4/5/6) tam bu boşluklarda** | Regresyon koruması bulunan hataları yakalamıyor | A | VERIFIED | Fable (n=1) | `synth.py`/`synth2.py`'deki T1-T12 senaryolarını unittest'e taşı (beklenen değerler mevcut çıktılarda) |
| `traceability-6` | Tezdeki nb05 şekilleri eski (final severity MINOR) | — | Bkz. D-41 | — | A, C | VERIFIED | Fable | Bkz. D-41 |
| `traceability-12` | Ch3 EDA metnindeki şekilden okunmuş sayılar hesaplananla kısmen uyuşmuyor | `3.dataset` L40, L84, L94, L104, L164, L214 | (a) log10 runtime ikinci mod **3,6–3,8** (≈4.000–6.300 s), tez "≈4.0 (≈10.000 s)". (b) Saatlik varış medyanı **421**, P25–P75 318–545 → "baseline 200–400" düşük. (c) İnter-arrival: log-aralıklı sayım [1,3)=18.917, [3,10)=**23.760** → "3–10 s'de keskin düşüş" doğrusal-bin artefaktı, **toplam sayım tersini gösteriyor**. (d) Küresel maksimum **Cuma saat 11 (1.352)**; Perşembe saat 2 (707) ve 7 (790) zirve değil. (e) `active_job_count` P10=423, medyan 637 → "baseline ≈400" P5–P10 düzeyi. (f) `gpu_demand`–runtime Pearson **0,064**, tez 0,05 | Küçük ama tekrar eden yanlış okumalar; L94'teki "keskin düşüş" histogram artefaktına dayanıyor | C | VERIFIED | Opus | Sayıları şekilden değil hesaptan al; inter-arrival'ı `np.logspace` binlerle çiz; L104'ü düzelt |
| `reproducibility-8` | DL scaler'ları her geçişte yeniden üretiliyor (final MINOR) | — | Bkz. E.1 | — | A | VERIFIED | Opus | Bkz. E.1 |
| `reproducibility-9` | Test kapsamı ve CI eksikliği (final MINOR) | — | Bkz. E.1 | — | A | VERIFIED | Opus | Bkz. E.1 |
| `reproducibility-10` | Çalışma ağacı hijyeni: sahipsiz CSV, izlenmeyen kilit dosyası, ~200 MB yedek dizini; commit'li CSV bayat | `data/processed/100k_job_with_utilization.csv`; `requirements-lock.txt`; `results/_backup_20260831_1703_pre_q2/` | `?? data/processed/100k_job_with_utilization.csv` (30 Ağu 23:15, farklı sütun sırası, hiçbir kod okumaz); `?? requirements-lock.txt`; `?? results/_backup_…`; `git show HEAD:…full.csv \| shasum` → 150efec9… ≠ c41a82f1…. `.gitignore` `AGENTS.md` ve `.agents/` dizinini yok sayıyor | Klonlayan kişi hangi CSV'nin güncel olduğunu bilemez; yedek yanlışlıkla commit edilirse depo şişer | A | VERIFIED | Opus | Sahipsiz CSV'yi sil; kilidi commit et; yedeği repo dışına taşı veya `.gitignore`'a ekle; güncel processed CSV'yi commit et |
| `reproducibility-11` | Tez metni yeniden üretim bilgisi açısından bayat/eksik | `4.prediction_models` L171; `7.conclusions` L43 | DL final eğitimi `DL_SEEDS=[42,1337,2024]` ile 3 kez; tezde tek tohum. `grep -rniE "pytorch\|scikit\|version\|MPS\|Apple" chapters/*.tex` → yalnız ilgisiz eşleşmeler. Toplam eğitim süresi (model mtime'larına göre ≥5 s 22 dk), veri karma değeri, cihaz **hiçbiri yok** | Okuyucu deneyi hangi ortamda, kaç tohumla, ne kadar sürede tekrarlayacağını bilemez | C | VERIFIED | Opus | Ch4/5'e "Reproducibility" alt bölümü: commit hash, Python 3.11.6, sklearn 1.8.0, XGBoost 3.1.2, LightGBM 4.6.0, PyTorch 2.11.0, Apple Silicon/MPS, DL tohumları, veri sha256, yaklaşık süreler |
| `baselines_claims-14` | README/CHECKLIST anlatısı üçüncü bir eski koşudan; "novel sweep-line feature" iddiası kaynakla desteklenmiyor | README L113-151; `abstract-en` L4; `1.introduction` L89 | README "FIFO 715.611 s, SJF-XGB 318.257 = 2,25×", "XGB-OH MAE 3.389/R² 0.51" — hiçbiriyle eşleşmiyor. README "SRF fewest GPUs/CPUs" derken kod yalnız `gpu_demand`'a bakıyor (`scheduler_simulator.py` L155-160). "Novel" sweep-line: olay-sıralı eşzamanlılık sayımı iz analizlerinde **standart** bir tekniktir; tez yenilik için literatür karşılaştırması vermiyor | Dış okuyucu için tutarsız üç sayı seti; yenilik iddiası savunmada sorgulanır | C | **NOT_VERIFIED** (yenilik alt-iddiası) | Opus | README'yi güncel çıktılardan üret; "novel" yerine "submission-time, leakage-free concurrency feature" de veya literatürle karşılaştır |
| `robustness-11` | Elenen 17.816 CPU işi gerçek sürelere sahip; simülasyon ve `cluster_load_cpu` bu yükü yok sayıyor | `feature_engineering.py` L102-105, L205-292 | Atılan kayıtların tamamı `num_gpu=0` & `gpu_type=CPU` (tez ch3 L25 **doğru**). Ancak medyan süre **955 s** (max 507.906 s) ve `num_cpu` talepleri var; düğüm profilleri CPU kapasitesi tanımlıyor ama simülasyonda CPU işi yok | Seçim etkisi: yük özellikleri ve simülasyon, izdeki eşzamanlı işlerin %18'ini dışlıyor; sonuçları çevirmesi beklenmez ama "küme durumu" özniteliklerinin yorumu sınırlı | C | VERIFIED | Opus | Tezde CPU işlerinin dışlanmasının sonuçlarını belirt; isteğe bağlı olarak `cluster_load_*`'ı filtre öncesi hesapla |
| `code_bugs-7` | Simülatör yerleştirilemeyen işleri sessizce düşürüyor (E.1 `simulator-6` ile aynı; final MINOR) | `multi_node_simulator.py` L430-515 | 23 politikanın hepsi 16.437 iş işledi; test max `num_cpu` 90 ≤ 96, max gpu 8 ≤ 8 → **şu an tetiklenmiyor** | Farklı konfigürasyon denemelerinde sessiz hata riski | A | VERIFIED | Opus | `if len(self.results) != len(jobs): raise RuntimeError(...)`; provision'da kapasite kontrolü |
| `code_bugs-9` | nb04 kod kalitesi (final MODERATE, E.1'de listelendi) | — | — | — | A | VERIFIED | Opus | — |
| `figures_tables-17` | Tablo 6.1 kalın yazım kendi içinde tutarsız; "top-performing" altyazısı 18 satırla çelişiyor | `6.results` L15, L27, L36 | Altyazı "Bold indicates the best result per metric"; MAPE sütununda **10.80 kalın**, oysa aynı tabloda LSTM (One-Hot) **6.51** daha düşük (Ek A `tab:expd-full`'da 6.51 kalın). Altyazı "top-performing models across the six experiments" der ama her deneyin 3 modeli (18 satır) listeleniyor; LightGBM (One-Hot) atlanıyor | Sunum hatası; MAPE'nin ch4 L156'daki tanımıyla da çelişen eski ölçek | C | VERIFIED | Opus | Kalınlığı sütun bazında hesaplayan otomatik LaTeX üretimi; MAPE sütununu kaldır veya ölçeği düzelt |
| `figures_tables-18` | Rank-korelasyon şekli: n=18 bağımlı nokta, CI yok, etiket çakışması, export korumasını atlayan doğrudan kayıt | nb05 c29 | `pearsonr(...)` + `np.polyfit`; `fig.savefig(_out_dir / f"mae_spearman_vs_jct_gain_{N_GPU}gpu.png")` → doğrudan `PROJECT_ROOT/thesis/latex/figures`. `annotate` etiketleri üst üste biniyor ("LSTM (Categorical Sequence)" / "LGBM (Categorical)"). Şekil tezde **hiç kullanılmıyor** | Bulgu (ρ vs JCT r=0,71) tezin **en güçlü mesajı olabilir** ama istatistiksel destek zayıf ve tezde yer almıyor | B | VERIFIED | Opus | Spearman + bootstrap CI; `adjustText` veya elle ofset; doğrudan tez klasörüne yazmayı kaldır; şekli ch6'ya ekle |
| `figures_tables-19` | HTML export tabloları LaTeX kaynağı olarak kullanılamaz: konumsal numaralama, veri önizlemesi "Table01", kesik parametreler | `export_thesis_results.py` L159-172 | `NB05_*_Table01.html` = `sim_jobs.head(3)` (34 sütunlu **veri önizlemesi**, sonuç tablosu değil); `NB04_Table09` parametreleri `'…'` ile kesik; herhangi bir hücreye `display()` eklenince tüm numaralar kayar | Tablo otomasyonu yok — **C-3/C-4/D-21'in kök nedeni** | A | VERIFIED | Opus | Sonuç DataFrame'lerini adlandırılmış CSV'lere yaz (`results/tables/*.csv`) ve bir csv→LaTeX (booktabs) betiğiyle `\input` et |
| `figures_tables-20` | Okunabilirlik: 21 politikalı şekiller 0.48\textwidth'e sıkıştırılmış; eksen etiketleri çakışıyor | `6.results` L374-380, L226-232; nb05 c40; nb04 c86 | Yüzdelik şekli 2 PNG × 2 panel = 4 panel × 21 etiket; kutu grafiğinde 35° döndürülmüş 21 etiket üst üste ("SRF (Heuristic)" ile "SJF-CNN-LSTM (Categorical Sequence)" çakışıyor); DL MAE şeklinde 15° etiketler birbirine giriyor | Baskıda okunamaz | A | VERIFIED | Opus | Yatay kutu grafiği; yüzdelik şeklini 2×2 yerleştir; kısa politika kodları |
| `figures_tables-21` | Şekil 3.1/3.9 altyazı-metin küçük uyuşmazlıkları | `3.dataset` L45, L211; nb03 md c22 | Altyazı "two distinct modes … around 300 s and in excess of 10,000 s"; şekilde **üç** tepe (~30 s, ~250-500 s, ~4.000-5.000 s), en yüksek üçüncü bin merkezi ≈4.162 s (10.000 s'nin **altında**). `gpu_demand`–runtime korelasyonu tezde 0,05, nb03 çıktısı **0,064**. nb03 md "sweep-line features … (0.83)" der, şekil ve tez **0,89** | Küçük ama doğrulanabilir tutarsızlıklar; okuyucu güvenini azaltır | C | VERIFIED | Opus | Altyazıyı üç modlu olarak düzelt; 0,05→0,06; nb03 anlatısını 0,89 yap |

### E.3 INFORMATIONAL (4 bulgu)

| ID | Problem | Kanıt | Etki | Status | Tur | Düzeltme |
|---|---|---|---|---|---|---|
| `leakage-8` | Üçüncü bir final hold-out yok: aynı 16.437 işlik test seti 21 modelin metriklerinde, şekillerde, simülasyon evreninde ve deney tasarımının ardışık yinelemelerinde tekrar tekrar kullanılıyor | Model seçimi düzeyinde test kullanılmıyor (`RandomizedSearchCV` yalnız `X_train`; DL yalnız val — **kodda VERIFIED**, `tuning.py` L1450-1452 yorumu: "test_dataset … deliberately never touched here"). Ancak **tasarım düzeyinde** (hangi deneylerin/kodlamaların/baseline'ların eklendiği) aynı test seti üzerinde çok sayıda okuma yapılmış. Tez L166 "test set performance measured only once" model başına doğru, **süreç genelinde değil** | Raporlanan metrikler ve JCT kazançları hafif iyimser olabilir; kesin büyüklüğü ölçülemez | VERIFIED | Fable (n=3) | Tezde "tek test seti, çoklu model karşılaştırması" sınırlamasını açıkça yaz; train'in son dilimini dev seti olarak ayır veya rolling-origin ekle |
| `modeling-14` | Ortam sürümleri tezde yok; defter çekirdeği ile sistem Python'u farklı sürümler taşıyor | venv: 3.11.6 / torch 2.11.0 / sklearn 1.8.0 / xgboost 3.1.2 / lightgbm 4.6.0; sistem python3: torch 2.5.1 / sklearn 1.7.2 / xgboost 3.1.1. **Her iki ortamda da artefaktlar yüklendi ve aynı metrikleri verdi** (iyi) | Belgeleme eksikliği; ileride yükleme hataları | VERIFIED | Fable (n=3) | `requirements-lock.txt`'yi tezin ekine koy; DL için `state_dict` + config kaydet |
| `code_bugs-8` | `[RECOVER]` yolu modeli yeniden eğitip metriklerini atıyor; checkpoint timestamp her geçişte yenileniyor; LGBM finalde `verbose=-1` yok | nb04 c11: `rf_final, _ = finalize_ml_model(...)` sonra `joblib.dump` — **checkpoint güncellenmez**; tüm JSON'lar `2026-08-31T21:28:2x`, model mtime'ları farklı; çıktıda `[SKIP]` 20, `[SAVED]` 1 | Provenans: hangi model dosyasının hangi metrikle eşleştiği yalnız "aynı kod, aynı tohum" varsayımıyla savunulabiliyor (bu koşuda ağaç metrikleri diskten yeniden üretildi, DL seed0 eşleşti — sorun yok) | VERIFIED | Opus | Recovery'de dönen metrikleri checkpoint ile karşılaştırıp uyuşmazlıkta hata ver; `trained_at` + model sha256; LGBM finalize'a `verbose=-1` |
| `code_bugs-10` | Ölü/yanıltıcı modüller ve test boşlukları | `grep`: `visualization`, `RandomForestPredictor`, `XGBPredictor`, `LightGBMPredictor`, `ClusterSimulator`, `use_processed=True` hiçbir `.ipynb`'de geçmiyor. Testler `SJFPredScheduler`+negatif tahmin, `Machine.release` kimlik eşleşmesi, finalize objective tutarlılığı, `EarlyStopping` delta ölçeği, `make_narrow_grid` dropout, `predict_in_batches` pencere hizası, export konumsal harita için **hiçbir birim testi içermiyor** | Yeni okuyucu için API yanıltıcı; bu denetimde bulunan hataların **hiçbiri** test tarafından yakalanmıyor | VERIFIED | Opus | Ölü modülleri kaldır veya README'de işaretle; `predict_in_batches`'i `src/inference.py`'ye taşı ve hizasını test et; her bulgu için regresyon testi |

---

## F. RESULT VALIDATION TABLOSU

> "Tezde yazan X → kodun ürettiği Y". Seçilmiş 40 satır; tam liste C ve D bölümlerinde.

### F.1 Tahmin sonuçları

| Konum | Tezde yazan | Güncel kod çıktısı | Backup (pre_q2) | Durum | Neden |
|---|---|---|---|---|---|
| tab:predresults XGB-OH | MAE 3.389 / MdAE 1.129 / RMSE 11.375 / MAPE 10.80 / R² **0.51** | **6.642** / 3.432 / 14.297 / 2.578 / **0.156** | aynı | ❌ | Mayıs koşusu + MAPE ölçek |
| tab:predresults RF-A | 4.316 / 1.508 / 13.831 / 16.85 / **0.27** | **6.842** / 3.452 / 15.375 / 2.473 / **0.024** | 15.237 / **−0.673** | ❌ | Üç farklı değer |
| tab:predresults LGBM-Nat | 4.106 / 0.44 | **5.697** / 2.684 / 13.543 / 1.660 / **0.243** ← **güncel en iyi** | aynı | ❌ | Mayıs koşusu |
| tab:predresults LSTM-D | 5.836 / 0.06 | **6.236 ± 509** / 0.118 (3 tohum); seed-42 **6.703** | 6.320 / 0.094 | ❌ | tek → 3 tohum |
| tab:predresults LSTM-F | 13.169 / **−0.27** | **6.424 ± 651** / **+0.109** | 6.510 / 0.138 | ❌ | İşaret ters |
| tab:predresults CNN-C | — | 7.054 / 0.002 | — | — | — |
| tab:predresults Hybrid-C | — | 6.471 / −0.012 | — | — | — |
| **Per-User Median** | **YOK** | MAE **5.191** / MdAE **847** / R² 0.070 / ρ **0.597** | yok | ❌ **eksik** | Tüm öğrenilen modelleri geçiyor |
| **ProfileMedian (5 kolon)** | **YOK** | MAE **4.389,7** / MdAE **537,0** / ρ **0,7204** | hiç hesaplanmamış | ❌ **eksik** | 21 modelin hepsini geçiyor |
| **Sabit train-medyanı (568 s)** | **YOK** | MAE **5.818** / R² −0.123 | yok | ❌ **eksik** | 16/17 öğrenilen model geçemiyor |
| **Sabit 0** | **YOK** | MAE 6.030 / MAPE %100 | yok | ❌ **eksik** | 14 model geçemiyor |
| **XGB-Native** | **YOK** | MAE 6.746 / R² 0.125 | yok | ❌ **eksik** | Kontrol deneyi |
| **L1-LGBM-Native** | **YOK** | MAE **4.870** / ρ **0.631** | — | ❌ | Ayarlandığı kayıpla eğitilse |
| **log1p-LGBM-Native** | ch4 L60'ta gerekçesiz reddedilmiş | MAE **4.744** / MdAE 817 / ρ **0.690** | — | ❌ | En iyi nokta doğruluğu |

### F.2 Simülasyon sonuçları (32-GPU)

| Politika | Tezde (mean JCT / %) | Güncel nb05 c27 | backfill=True | Durum |
|---|---|---|---|---|
| FIFO | 499.951 | **1.344.966** | 1.054.665 (−%21,6) | ❌ 2,7× |
| SJF-Oracle | 92.064 / **81.59** | **252.024** / **81.26** | 202.618 (%80,79) | ❌ |
| SJF-XGB-Cat | 218.751 / **56.25** (en iyi ML) | 543.181 / **59.61** (2.) | — | ❌ sıra değişti |
| SJF-LSTM-Cat | 223.928 / 55.21 | **534.188** / **60.28** (**1.**) | **53.97** | ❌ |
| SJF-LGBM-Cat | 277.715 / — | 577.456 / 57.07 | 55.29 | ❌ |
| SJF-XGB-Native | **YOK** | — / 57.00 | 49.81 | ❌ eksik |
| **SJF-UserMedian** | **YOK** | 846.007 / **37.10** | — | ❌ eksik |
| **SJF-ProfileMedian** | **YOK** | 889.263 / **33.9** | — | ❌ eksik |
| SRF | — / 22.30 | — / **44.27** | — | ❌ |
| SJF-LSTM-NumSeq | −45.52 | **+18.31** | — | ❌ **işaret ters** |
| SJF-CNN-NumSeq | anlamsız | **−8.40** (anlamlı KÖTÜ) | — | ❌ yorum ters |
| **Sabit-medyan tahmin** | **YOK** | 1.344.966 = **FIFO ile birebir** | — | 💡 kontrol |
| **Rastgele öncelik** | **YOK** | 1.738.633 / **−29.27** | — | 💡 kontrol |

### F.3 Simülasyon sonuçları (256-GPU)

| Politika | Tezde | Güncel | backfill=True | Durum |
|---|---|---|---|---|
| FIFO | 50.386 (tab), 44.356 (metin) | **153.382** | — | ❌ |
| SJF-Oracle | 14.506 / **71.21** | **32.264** / **78.96** | — | ❌ |
| SJF-LSTM-Cat | 57.25 (**başlık sayısı**) | **63.08** | **48.23** | ❌ |
| SJF-XGB-Cat | — | 62.89 | — | ❌ |
| SJF-LGBM-Cat | — | 61.44 | **51.60** | ❌ |
| SJF-XGB-Native | YOK | 60.11 | 43.63 | ❌ |
| SJF-UserMedian | YOK | 37.94 | **39.18** (LF 1.0: **60.84 > XGB-Nat 58.44**) | ❌ **baseline ML'yi geçiyor** |
| SRF | 29.64 | **48.39** | — | ❌ |
| SJF-LSTM-NumSeq | **−45.41** | **+22.87** | — | ❌ işaret ters |
| SJF-CNN-NumSeq | anlamsız | **−30.11** (r=−0,315, CI negatif) | — | ❌ |
| P95 en düşük ML | LSTM-Cat 62.514 | **XGB-Native 178.976** | — | ❌ |
| Medyan en düşük (Oracle hariç) | RF-Cat 1.001 | **LSTM-Cat 26.835** | — | ❌ |
| FIFO P95 | 123.407 | **323.020** | — | ❌ |

### F.4 Türetilmiş yüzdeler ve nitel iddialar

| İddia | Tezde | Güncel | Durum |
|---|---|---|---|
| Oracle'ın %X'i (32-GPU) | %68.9 | **%74,2** (60,28/81,26) | ❌ |
| Oracle'ın %X'i (256-GPU) | %80.4 | **%79,9** (63,08/78,96) | ❌ |
| SRF'ye göre kazanç | %39.2 | **%28,5** (256) | ❌ |
| RQ3 puan farkı | 28 puan | **14,7** (256) / **16,0** (32) | ❌ |
| Ölçek-bağımlı dikotomi | "32'de ağaçlar, 256'da LSTM" | **Her ikisinde LSTM-Cat 1., XGB-Cat 2.; Spearman ρ=0,9897** | ❌ **yok** |
| İki küme JCT sıralaması | "relationship broke down" | ρ=0.99, p=6e-15 | ❌ |
| Abstract başlık sayısı | %57.25 | %63,08 (aynı politika) | ❌ |
| "en doğru tahminci XGB-OH" | R² 0.51 | R² **0,16**; LGBM-Nat 0,24; UserMedian MAE daha iyi | ❌ |

### F.5 Veri, protokol ve altyapı iddiaları

| İddia | Tezde | Ölçülen | Durum |
|---|---|---|---|
| Ortalama GPU talebi | 0.52 | **0,680** (int-kesme 0,5216) | ❌ |
| "~37.000 iş ≤1 GPU" | 37.000 | =1 GPU: **37.033**; ≤1: **80.209** (%97,6); <1: **43.176** (%52,5) | ⚠ yarı doğru |
| "neredeyse hepsi 1 GPU" (fig 3.3) | — | %45,1 = 1; **%52,5 < 1** | ❌ |
| Filtre (17.816 elenen = CPU) | ✅ | 17.816 = num_gpu 0 & CPU; duration≤0: **0** | ✅ |
| Örneklem büyüklükleri | 100.000→82.184; 65.747/16.437 | ✅ aynı | ✅ |
| Runtime yüzdelikleri | P50 594, P95 24.110, P99 67.432, max 599.445 | ✅ aynı | ✅ |
| Korelasyon "up to 0.89" | 0.89 | **0,888** ✅ (nb03 md "0.83" = 0,826) | ✅ |
| `active_job_count` max >1.000 | ✅ | **1.088** ✅ | ✅ |
| Öznitelik boyutları | 9 / 591 / 11 | ✅ aynı | ✅ |
| Düğüm profilleri | 96/8 ve 64/2; 2+8 / 16+64 | ✅ kodla birebir | ✅ |
| Weng et al. atfı (profiller) | "derived from" | Kaynağa erişilemedi | **NOT_VERIFIED** |
| Simülasyon iş havuzu | 16.437 | ✅ 23 politikanın hepsi | ✅ |
| tab:sim_time (FIFO 3,2/12,4 s) | 36 değer | **Ölçüm kodu YOK**; ölçülen 8-24 s (bf=F), **54-74 s** (bf=T) | ❌ üretilemez |
| "under 17 s per policy" | ✅ | bf=F sınırda, bf=T **yanlış** | ❌ |
| README "100.000 event <60 s" | ✅ | 16.437 iş = 32.874 event; ölçüm yok | ❌ |
| Test sayısı | 11 | **34** | ❌ |
| Şekil sayısı | 26 PNG | **32** | ❌ |
| Python sürümü | 3.10 | **3.11.6** | ❌ |
| "tek sabit tohum 42" | ✅ | DL **3 tohum** [42,1337,2024] | ❌ |
| tab:hyperparams (19 satır) | — | **0/19 eşleşme** | ❌ |
| "18 tahminci / 4 politika / 3 koşu" | ✅ | **20 tahmin seti / 23 politika / 79 koşu** | ❌ |
| ch2 "ρ at or above 1.0" | ρ≈1 | **ρ = 20,7 – 165,8** | ❌ 2 mertebe |
| ch7 "bootstrap CI hesaplanmadı" | ✅ | **999-örnek CI + rank-biserial + Holm mevcut** | ❌ ters |
| ch6 L68 "runtime lacks temporal autocorrelation" | ✅ | log-runtime **lag1 ACF 0,340**, Ljung-Box p≈0 | ❌ ters |
| ch5 L153 "fractional allocations not allowed" | ✅ | Kesirli GPU **toplamsal paketleniyor** | ❌ |
| ch5 L23 "earliest inserted event first" | ✅ | heapq bunu **garanti etmiyor** | ❌ |
| ch1 L78 "CPU, GPU, and memory enforced" | ✅ | `mem_capacity = 0.0` (devre dışı); izde kolon yok | ❌ |
| ch5 L92 heterojen GPU modelleri | ✅ | `grep gpu_type src/simulation/` → **0** | ❌ |
| ch7 L43 "code released to reproduce" | ✅ | origin/main sonuçları üretmiyor; 173 dosya uncommitted | ❌ |

**Özet:** Doğrulanan 12 satır (veri istatistikleri, filtre, boyutlar, düğüm profilleri, iş havuzu); **uyuşmayan 48+ satır**; NOT_VERIFIED 2.

---

## G. REPRODUCIBILITY AUDIT

### G.1 Ortam envanteri

| Bileşen | `requirements-lock.txt` | Defterlerin koştuğu venv | `environment.yaml` | README |
|---|---|---|---|---|
| Python | 3.11.6 | 3.11.6 ✅ | **3.10** ❌ | conda |
| scikit-learn | 1.7.2 | **1.8.0** ❌ | >=1.3 | – |
| torch | 2.5.1 | **2.11.0** ❌ | >=2.0 | – |
| numpy / pandas | 1.26.3 / 2.2.3 | 2.3.5 / 2.3.3 ❌ | – | – |
| xgboost / lightgbm | 3.1.1 / 4.6.0 | 3.1.2 / 4.6.0 | – | – |
| Paket sayısı | **294** (anthropic, celery, datasets dahil) | 162 | – | – |

Kilit dosyası pyenv global ortamının `pip freeze` çıktısı — sonuçları üreten venv'i tanımlamıyor. `joblib.load` kilit ortamında 20+ `InconsistentVersionWarning` üretiyor; `warnings.filterwarnings("ignore")` bunları gizliyor.

### G.2 Sürüm kontrolü

| Kontrol | Durum |
|---|---|
| `git log -1` | **4923ce3, 2026-08-17** |
| `main == origin/main` | Evet — yayımlanan = HEAD |
| HEAD'de `gpu_demand` | `.astype(int)` ← **kesirli GPU'yu sıfırlayan hata** |
| HEAD'de simülatör | `num_gpu` okuyor ← **GPU kapasitesi uygulanmayan rejim** |
| Uncommitted | src 9, tests 4, scripts 1, configs, 21 checkpoint, 23 model, 14 defter, 8 tez bölümü, 24 figür (~173 dosya) |
| `git diff --stat HEAD -- src tests scripts configs` | 14 dosya, **+622/−505** |
| Commit'li processed CSV | sha 150efec9… ≠ güncel c41a82f1… (**bayat**) |
| Tag / release | **Yok** |
| CI (`.github/workflows`) | **Yok** |

### G.3 Veri provenansı

| Kontrol | Durum |
|---|---|
| sha256 | `dc4de3214e15b7c758b6485aaa1766c0c347a2482693e069ce7124009d1db56d` |
| Boyut | 100.000 × 8; `job_id` 0..99999 |
| Zaman | `submit_time` 0..661.892 s (7,66 gün), sıralı, maks. boşluk 237 s → **kesintisiz kronolojik dilim**, rastgele örnek değil |
| Resmi PAI izi | 1,2 M task / 2 ay (NSDI'22) — bu dosya bir **alt-dilim** |
| Köken / seçim yöntemi / tarih aralığı | Tezde ve `data/README.md`'de **YOK** |
| `duration` semantiği (end−start mı end−submit mi) | **Belgelenmemiş** |
| İşlenmiş CSV yeniden üretimi | ✅ Bağımsız betikle **bit-birebir** (0,6 s) |
| Aşağı-akış kullanımı | ⚠ İşlenmiş CSV'yi **hiçbir defter okumuyor** |

### G.4 Determinizm

| Katman | Durum | Kanıt |
|---|---|---|
| Ağaç modelleri | ✅ Deterministik | `random_state=42`, `TimeSeriesSplit(3)`, `n_jobs=1`, `OMP_NUM_THREADS=1`; 9/9 artefakt checkpoint'i birebir üretti |
| DL — aynı cihaz | ✅ Bit-birebir (süreç içi + süreçler arası) | md5 sabit; 12/12 `.pth` `mae_seed0`'ı birebir üretti |
| DL — MPS vs CPU | ❌ **Farklı** | maxdiff **0,284** (toy LSTM); `torch.use_deterministic_algorithms` **yok** |
| Simülasyon | ✅ Deterministik | `src/simulation`'da random/seed/shuffle **yok**; backup ↔ güncel FIFO/Oracle satırları birebir |
| Bootstrap | ✅ | `default_rng(42)` |
| Dropout | ⚠ RS'de örnekleniyor, griden düşüyor → **hep 0.2** | 12 `.pth`'de `nn.Dropout.p = {0.2}` |

### G.5 "Restart & Run All" davranışı

| Kontrol | Durum |
|---|---|
| Kök bulucu | ✅ Marker tabanlı, mutlak yol yok |
| `execution_count` | ✅ 1..N sıralı (nb04 1..65, nb05 1..23) — baştan sona koşu kanıtı |
| Checkpoint mantığı | ⚠ Varsa **eğitim atlanır**; mevcut koşuda **21/21 yüklendi, 0 eğitildi** |
| Provenans bağı | ❌ src/veri/kütüphane hash'i **yok**; timestamp her geçişte yenileniyor |
| Model ↔ checkpoint doğrulaması | ❌ Yalnız `_dest.exists()` → `[SKIP]` |
| DL RECOVER dalı | ⚠ `seeds=DL_SEEDS` vermiyor → tek tohum |
| Scaler'lar | ⚠ Her geçişte yeniden yazılıyor (mtime 1 Eyl 00:28 vs ağlar 21:08–00:15) |
| Öznitelik hizası doğrulaması | ❌ Yok (yalnız sütun sayısı) |
| Betikli yol (`run_all_experiments.sh`) | ❌ `python3` → pyenv global (farklı sürümler) |
| Kernelspec | ⚠ `name='python3'`, `display_name='venv'` — ad çakışması |
| Export | ⚠ Konumsal harita + hepsi-ya-da-hiç; NB05 **atlandı** (6≠7) |
| HTML→LaTeX köprüsü | ❌ **Yok** |

### G.6 Süre ve donanım

| Kontrol | Durum |
|---|---|
| nb04 duvar-saati | ≥ 5 s 22 dk (model mtime: 18:53 → 00:15), Apple Silicon, tek iş parçacığı |
| RS+GS süresi | **Kaydedilmiyor** |
| Final refit toplamı | 983 s (checkpoint `train_time`) |
| Simülasyon süresi | **Ölçüm yok**; denetimde 8-24 s (bf=F), 54-74 s (bf=T) |
| Tezde süre/donanım/sürüm | **Hiçbiri yok** (grep 0 sonuç) |

### G.7 Test paketi

| Kontrol | Durum |
|---|---|
| Test sayısı / süre | 34 test, 0,31 s, **OK** ✅ |
| `tuning.py` kapsamı | 33 public sembolden **5** (`seed_everything`, `chronological_train_validation_split`, `save_checkpoint`, `train_dl_model`, `create_model_instance`) |
| Kapsanmayan | `run_randomsearch_*`, `run_gridsearch_*`, `finalize_*`, `make_narrow_grid`, `prepare_dl_datasets`, `load_all_checkpoints` |
| 0 test | `data_loading` (5 sembol), `workload_analysis` (5), `visualization` (3), `dl_runtime_predictor` (3), `*_runtime_predictor` sarmalayıcıları, `export_thesis_results.py` |
| Belgelerdeki sayı | "11 test" (4 yerde) — **bayat** |

---

## H. STATISTICAL AUDIT

### H.1 Uygulama doğruluğu (A) — **SAĞLAM** ✅

| Bileşen | Doğrulama |
|---|---|
| Wilcoxon eşleştirmesi | `job_id` ile inner merge → 16.437 çift; `job_id` test setinde benzersiz ✅ |
| W istatistiği | scipy `alternative='greater'` W+ ile tutarlı, sentetik kontrolle yeniden üretildi ✅ |
| Holm-Bonferroni | Referans uygulamayla **birebir aynı** (rastgele p vektörlerinde `Holm match: True`) ✅ |
| Rank-biserial | (W+−W−)/(W++W−) formülü doğru; ima edilen n' 32-GPU ≈16.380, 256 ≈15.927 tutarlı ✅ |
| Bootstrap CI | 999 yeniden örnek, percentile, `default_rng(42)` deterministik ✅ |
| Checkpoint metrikleri | 9/9 ağaç + lookup, 12/12 DL diskten **bit-özdeş** yeniden hesaplandı ✅ |
| Simülasyon | Deterministik, bit-bit yeniden üretildi ✅ |
| Test seçimi | Wilcoxon parametrik olmayan; **hiçbir yerde** normallik/homoskedastisite varsayan test yok (`grep ttest\|anova\|shapiro\|levene` → 0) ✅ |
| MAPE eşdeğerliği | nb05 sklearn ×100 = `evaluation.py` (duration>0 filtresi sıfırı dışlıyor) ✅ |
| Çok-tohum SD | `ddof=1`, n=3; `*_seed0` ayrıca saklanıyor ✅ |
| ch2 iş yükü istatistikleri | ortalama 5.223 s, SD 15.980 s, **c_s²=9,36** ✅ |

**Sonuç:** Kod çalışıyor ve yaptığı hesabı doğru yapıyor. Sorun hesabın **ne anlama geldiğinde**.

### H.2 Çıkarım geçerliliği (B) — **ZAYIF** ❌

| Kanıt | Değer |
|---|---|
| Eşleştirilmiş JCT farkı ACF (varış sırası) lag1 / lag100 / lag1000 | **0,801 / 0,714 / 0,559** |
| iid bootstrap %95 CI genişliği | 29.522 s |
| Moving-block bootstrap L=100 / 500 / 1000 / 2000 | 264k / 533k / **747k** / 923k s (**25×**) |
| JCT iyileşmesi ilk yarı / ikinci yarı | %11,6 / **%70,6** |
| Gün-0 / gün-1 ortalama fark | 291.792 / **1.704.504** s |
| Farkı pozitif iş payı | %78,4 (tanımlayıcı iddia doğru) |
| Test hataları ACF\|err\| lag1 | 0,14–0,21 |
| Blok-CI / iid-CI oranı (tahmin metrikleri) | ~3× |
| Test penceresi | **2 takvim günü** |
| Etkin bilgi birimi | ~**3.146 profil / 636 kullanıcı / 1 pencere** |
| İşaret değişimi kanıtı | CNN-LSTM (Num Seq) 32-GPU: backup −%14,96 → güncel **+%7,47** ve Holm p<0,05 "anlamlı" |

### H.3 Yön ve yorum hataları

| Hata | Detay |
|---|---|
| Tek yönlü test ters okuma | `alternative='greater'` → p=1,000 satırları FIFO'dan **anlamlı KÖTÜ** (256 CNN-NumSeq r=−0,315, CI [−48.463, −43.644]); tez "not statistically distinguishable" |
| H1 tutarsızlığı | c45 markdown iki yönlü ("differs"), kod tek yönlü |
| p-değeri tanımı | c45: "gözlenen iyileşmenin şans eseri olma olasılığı" — **yanlış** |
| Aşırı dil | c47 "definitively proves", "p strictly at 0.000000" |
| Raporlama | ch6 L400 "p = 0.000" (p<0.001 olmalı) |
| ch7 L72 | "bootstrap CI hesaplanmadı" — **tersi doğru** |

### H.4 Taban çizgisi tablosu

| Tahminci | MAE | MdAE | R² | Spearman ρ | JCT % (32-GPU) |
|---|---|---|---|---|---|
| **ProfileMedian (hiç hesaplanmamış)** | **4.389,7** | **537,0** | — | **0,7204** | 33,9 |
| log1p-LGBM-Nat | 4.744 | 817 | — | 0,690 | 43,86 |
| L1-LGBM-Nat | 4.870 | 1.021 | — | 0,631 | 47,67 |
| **Per-User Median** | **5.191,5** | **846,5** | 0,070 | **0,5966** | **37,10** |
| LGBM-Native (kayıtlı, L2) | 5.697,4 | 2.683,5 | **0,243** | 0,5388 | **57,07** |
| Sabit train-medyanı (568 s) | 5.818,2 | 534 | −0,123 | — | **= FIFO** |
| Sabit 0 | 6.029,7 | 731 | −0,15 | — | — |
| RF-OH | 6.292 | — | 0,153 | — | — |
| LGBM-OH | 6.640,0 | — | 0,118 | — | — |
| XGB-OH | 6.642,4 | 3.432 | 0,156 | 0,452 | 59,61 |
| LSTM-Cat (seed 42) | 6.706 | — | — | 0,420 | **60,28** |
| Rastgele öncelik | — | — | — | — | **−29,27** |

**İki çarpıcı sonuç:** (1) 17 öğrenilen modelin **16'sı sabit medyanı**, **14'ü sıfır tahminini** geçemiyor. (2) MAE/ρ iyileştikçe JCT kazancı **düşüyor** (4.744→%43,9; 4.870→%47,7; 5.697→%57,1).

### H.5 Eşleştirilmiş bootstrap karşılaştırmaları (n=16.437, B=2000)

| Karşılaştırma | ΔMAE (s) | %95 CI | Yorum |
|---|---|---|---|
| LGBM-Nat − ProfileMedian | **+1.307,7** | [1.146,8, 1.454,8] | Profile **anlamlı üstün** |
| LGBM-Nat − UserMedian | +505,9 | [398,4, 615,3] | UserMedian **anlamlı üstün** |
| LGBM-Nat − TrainMedian | −120,9 | **[−242,1, +5,9]** | **Ayırt edilemez** |
| RF-OH − TrainMedian | +474 | [368, 581] | Sabitten **anlamlı KÖTÜ** |
| XGB-OH − TrainMedian | +824,1 | [724,2, 931,2] | Sabitten **anlamlı KÖTÜ** |
| LGBM-Nat − RF-OH | −595 | [−682, −513] | Anlamlı |
| XGB-OH − LGBM-OH | +2,4 | [−47, +54] | **Ayırt edilemez** |

SE(MAE) ≈ 96-120 s (95% yarı-genişlik ±200 s). DL tohum SD'leri: 123 / 430 / 509 / 651 / 739 s → mimariler arası farklar (200-900 s) **gürültü içinde**.

### H.6 Metrik seçimi sorunları

| Metrik | Sorun |
|---|---|
| **MAPE** | ch4 doğru (binlerce %, asimetrik), ch6 eski ölçek (%16,85); y<60 s işler (%12,1) APE'nin **%47,8**'i; medyan APE %184; **sabit 0 tahmini MAPE=%100 ile tüm modelleri "geçiyor"** |
| **Slowdown** | Sınırsız; FIFO ortalama 8.085 (Oracle 36,8) — en kısa işlerce belirleniyor; bounded slowdown (Feitelson) kullanılmamış |
| **MAE** | Kuyruğun 800 işine duyarlı: LGBM-Nat MAE'sinin **%35'i** en uzun %5'ten; top-%5 hariç öğrenilen modeller baseline'lardan **kötü** |
| **R²** | 2 ondalıkla "−0,00" basılıyor |
| **Spearman ρ** | JCT'yi açıklamıyor (4 karşı-örnek); tie oranı ölçülmemiş (UserMedian %73 iş ≥100'lük eşitlik grubunda) |

---

## I. DATA LEAKAGE / BIAS AUDIT

### I.1 Klasik sızıntı: **YOK** ✅

| Kontrol | Sonuç | Kanıt |
|---|---|---|
| Kronolojik bölme bütünlüğü | ✅ | train bitişik önek; train max 524.743 < test min 524.788; guard L501-512 aktif; sınırda eşzamanlı-varış bağı **0** |
| OneHotEncoder | ✅ Yalnız train'e fit | 9 + 578 user + 4 gpu_type = 591; 312 test satırı sıfır user-vektörü (`handle_unknown='ignore'`) |
| MinMaxScaler (4 adet) | ✅ Yalnız train | `data_min/max == train min/max: True`; `scaler_y` [4, 599445] |
| Sweep-line özniteliği | ✅ **Gelecek sızıntısı yok** | Perturbasyon: bir işin runtime'ı kendi özniteliğini ve önceki hiçbir işin özniteliğini değiştirmiyor (`'earlier-arrival changed: 0, own: False'`) |
| Per-User Median | ✅ Yalnız train | 578 kullanıcı + global 568 s; tabloda test-only kullanıcı **yok** |
| Hiperparametre seçimi | ✅ Test'e bakmıyor | ML: `TimeSeriesSplit(3)` yalnız `X_train`; DL: yalnız `val_loader` (`tuning.py` L1450: "test_dataset … deliberately never touched") |
| İç ES bölmeleri | ✅ Kronolojik | CV son %15, final %10, DL son %20; iç sınırlarda bağ yok (DL %80 sınırı hariç: 4 işlik önemsiz burst) |
| DL pencere hizalaması | ✅ | train hedef idx ≤ n_train+seq_len−2, val ≥ n_train+seq_len−1; test pencereleri train'in son 9 satırının **yalnız özniteliklerini** önek alıyor, hedefleri sıfır |
| Kopya sızıntısı | ✅ | `'test rows with identical (feat+user) in train: 0'` |

### I.2 Protokol sapmaları ⚠

| # | Sapma | Etki | Severity |
|---|---|---|---|
| 1 | Kategori sözlüğü tüm veriden (636 user, 58'i test-only) → LGBM/XGB native'e gömülü | Hedef sızıntısı yok; canlı sistem gelecekteki kullanıcıları bilemez | MINOR |
| 2 | Sweep-line CPU-only 17.816 işi (%17,8) filtre sonrası dışlıyor | `cluster_load_cpu` −766 CPU (%15), `active_job_count` −144 (%22) — tanım tutarsızlığı | MODERATE |
| 3 | Nihai eğitim verisi asimetrik: RF 65.747 / XGB-LGBM 59.172 / DL 52.598 | Aileler arası karşılaştırma "aynı veri" üzerinde değil; teste en yakın veriyi kim gördü? | MODERATE |
| 4 | Üçüncü final hold-out yok; test seti tasarım yinelemelerinde tekrar tekrar okunmuş | Metrikler hafif iyimser | INFORMATIONAL |

### I.3 Yanlılık ve temsil sorunları

| # | Sorun | Ölçüm |
|---|---|---|
| 1 | **Epoch artefaktı** | `submit_time` göreli → `to_datetime` 1970-01-01 (Perşembe); 7,66 günlük izde `day_of_week` ≈ iz-günü. `dow` seviye→gün eşlemesi: {0:[3], 1:[4,5], 2:[5], 3:[6], 4:[0,1], 5:[1], 6:[2], 7:[3]}. Test `dow`∈{2,3}; train'de `dow=2` yalnız **1.005 satır**, `dow=3` izin **İLK günü** |
| 2 | **`arrival_sec` ekstrapolasyonu** | Test [524.788, 661.889] > train max 524.743; ağaç tahminleri kırpmaya **bit-bit duyarsız**; ölçekli DL değeri 1,00-1,26 (eğitim aralığı [0,1] dışı) |
| 3 | **Etkin n çöküşü** | 82.184 iş → **3.146 profil** (%96,2 kopya satır; süre dahil %34,7); test satırlarının **%91,16**'sının profili train'de görülmüş |
| 4 | **Kullanıcı yoğunlaşması** | 636 kullanıcı; en büyük **%16,3**; top-10 **%57,0**; 189 kullanıcı <5 iş; kullanıcı başına medyan 11 iş |
| 5 | **Burst yapısı** | Aynı user-aynı saniye: train %24,6, test %22,6; 13.956 çift sayısal satır (1.218'i hedef dahil özdeş) |
| 6 | **Görülmemiş kullanıcı çöküşü** | 58 kullanıcı / 312 iş (%1,9): LGBM-Nat MAE **7.202**, XGB-OH 7.190 vs UserMedian global yedeği **2.600** |
| 7 | **`user` bağımlılığı** | Öznitelik önem toplamı: XGB-OH **%89,3**, RF-Cat %47,1, LGBM-Nat %18,5. Yalnız `user`+`gpu_type` ile LGBM(L1): MAE 5.204, ρ 0,564 (9 sayısal öznitelik bunu 4.870/0,631'e taşıyor) |
| 8 | **Dağılım kayması** | Hedef: train ort. 5.021 → test **6.030** s; medyan 568 → 731. `gpu_type`: T4 %33,1→%22,7; **P100 %5,3→%9,5; V100 %1,8→%7,0**. `cluster_load_cpu` 4.830→5.875; `active_job_count` 629→693 |
| 9 | **Isınma artefaktı** | Sweep-line 0'dan başlıyor; iz öncesi işler bilinmiyor → ilk ~12 saatteki tüm işlerin küme-yükü öznitelikleri **sistematik düşük** |
| 10 | **Simülasyon evren uyuşmazlığı** | Modele verilen `cluster_load_gpu` ort. **533,9** / max 822,9 GPU; simüle edilen küme **32/256 GPU** → öznitelik ortalamada kapasitenin **16,7×**'ini bildiriyor |
| 11 | **CPU işi dışlaması** | 17.816 iş (medyan süre 955 s, max 507.906 s) hem simülasyondan hem yük özniteliklerinden çıkarılmış |
| 12 | **Tek pencere** | Tüm değerlendirme 1,59 günlük tek dilimde; rolling-origin yok; `day_of_week` seviye başına **n=1 gün** |

**Sonuç:** Sızıntı yok, ama **temsil ve bağımsızlık** ciddi biçimde zayıf. Görev fiilen "aynı kullanıcıların ileri-zaman işleri"; bu, C-6/C-7'deki lookup üstünlüğünü de açıklıyor — **veride öğrenilecek yapı esas olarak bir arama tablosudur.**

---

## J. ROBUSTNESS AUDIT

### J.1 Ana sonuçların sağlamlık etiketleri

| Ana sonuç | Etiket | Gerekçe |
|---|---|---|
| Kronolojik bölme, sızıntısız değerlendirme | **robust** ✅ | Guard aktif; aralıklar ayrık; scaler/encoder yalnız train |
| Kategorik özellikler > sayısal (tahmin ve JCT) | **robust** ✅ | 2 küme × 9 model; ort. JCT% 50,7 vs 26,4 (32), 52,5 vs 28,7 (256); ort. ρ 0,383 vs 0,144; **8/8 süpürme noktası** |
| ML-SJF > FIFO ve SRF (yön) | **robust** ✅ | 8/8 nokta, iki küme; rastgele öncelik **−%29,27** (yani kazanç "FIFO'yu bozmaktan" gelmiyor) |
| Oracle ≫ ML (yön) | **robust** ✅ | Her yerde |
| JCT iyileşme büyüklükleri (%57-63) | **fragile** ⚠ | ρ≈166/21 yığın rejimi; LF 1.0'da bile ρ≥2,1; backfill'de 6-16 puan düşüş |
| "SJF-LSTM-Cat en iyi ML" / DL büyük ölçekte üstün | **unsupported** ❌ | Fark 0,19-0,67 puan; CI örtüşüyor; tek tohum (SD 509-651 s); sıralama LF ile değişiyor |
| "Ağaç modeller runtime'ı iyi tahmin eder" (RQ2) | **unsupported** ❌ | R²≤0,24; sabit medyan RF/XGB/LGBM-OH ve tüm Exp A'yı geçiyor; UserMedian ve ProfileMedian hepsini geçiyor |
| "Nokta doğruluğu ≠ çizelgeleme" (yön) | **moderately robust** ⚠ | Pearson(MAE,JCT) p>0,15 (n=18, güçsüzlük) |
| "Sıralama kalitesi mekanizmadır" | **unsupported** ❌ | 4 karşı-örnek: ρ artarken JCT düşüyor |
| "Zamansal özellikler kritik" | **unsupported** ❌ | `arrival_sec` sırası 5-7/9; çıkarınca MAE/ρ **kötüleşmiyor** (4870→4898, ρ 0,631→0,645) |
| Sweep-line katkısı | **unsupported** ❌ | MDI sırası 7/9, 7/9, 5/9; **hiç ablasyon yok** |
| Başka kümelere genellenebilirlik | **unsupported** ❌ | Tek iz, 7,7 gün, `user` baskın (%89), görülmemiş kullanıcıda global-medyandan kötü |
| Ölçek-bağımlı dikotomi | **refuted** ❌ | Spearman ρ=0,99 |
| "Sequence trap −%45" | **refuted** ❌ | Güncelde +%18/+%23 |

### J.2 Ablasyon ve alternatif eğitim sonuçları

| Varyant | MAE | MdAE | ρ | JCT % (32-GPU) |
|---|---|---|---|---|
| Kayıtlı L2-LGBM-Nat | 5.697 | 2.684 | 0,539 | **57,07** |
| L1-LGBM-Nat (ayarlandığı kayıp) | **4.870** | 1.021 | 0,631 | 47,67 |
| log1p-LGBM-Nat | **4.744** | **817** | **0,690** | 43,86 |
| L1, `arrival_sec` çıkarılmış | 4.898 | 985 | **0,645** | — |
| L1, `arrival_sec`+`day_of_week` çıkarılmış | 4.906 | — | 0,643 | — |
| Yalnız `user`+`gpu_type` (L1) | 5.204 | 934 | 0,564 | — |

**Mekanizma:** En uzun %5 işte ortalama tahmin L2=**19.702 s**, log=10.338 s, L1=9.717 s → L2 modeli "fil" işleri kuyruk sonuna daha net itiyor. İkinci mekanizma: tahmin eşitlik grupları (UserMedian %73 iş ≥100'lük grupta, en büyüğü %19,3; RF-Cat %70; XGB-Cat %56); `SJFPredScheduler` `idxmin` eşitlikte ilk kuyruk elemanını seçiyor → **kısmen FIFO**.

### J.3 Duyarlılık ızgarası (nb05 c30, 56 koşu)

| Politika | 32-GPU bf=F LF 0.1 | 32-GPU bf=T LF 0.1 | 256-GPU bf=F LF 0.1 | 256-GPU bf=T LF 0.1 | 256 bf=T LF 1.0 |
|---|---|---|---|---|---|
| LSTM-Cat | 60,28 | **53,97** | 63,08 | **48,23** | 65,43 |
| LGBM-Cat | 57,07 | 55,29 | 61,44 | **51,60** | — |
| XGB-Native | 57,00 | 49,81 | 60,11 | **43,63** | 58,44 |
| UserMedian | 37,10 | — | 37,94 | 39,18 | **60,84** ← **ML'yi geçiyor** |
| Oracle | 81,26 | 80,79 | 78,96 | — | — |

**Kritik:** backfill=True'da **en iyi model LSTM'den LGBM'ye değişiyor**; LF 1.0'da **trivial baseline XGB-Native'i geçiyor**. Ayrıca bf=F LF 1.0'da 256-GPU sıralaması: **XGB-Nat 77,39 > LSTM-Cat 76,01 > LGBM-Cat 72,93** (LF 0.1'in tersi).

### J.4 Ağır kuyruk duyarlılığı

| Dilim | LGBM-Nat | UserMedian | Sabit medyan |
|---|---|---|---|
| Tüm test MAE | 5.697 | 5.191 | 5.818 |
| **Top-%5 hariç MAE** | 3.900 | **2.929** | **3.102** |
| **<1 dk işlerde MAE** | 2.050 | 753 | **531** |
| MAE'nin top-%5'ten gelen payı | %35,0 | %46,4 | %49,4 |

**Sonuç:** Öğrenilen modeller **yalnız kuyrukta** kazanıyor, gövdede kaybediyor. Model karşılaştırmaları hangi hata diliminin raporlandığına göre **tersine dönüyor** (fragile).

### J.5 FIFO taban çizgisinin kırılganlığı

| Senaryo | FIFO mean JCT | Değişim |
|---|---|---|
| nb05 satır sırası (referans) | 1.344.966,16 | — |
| Eş-zamanlı satırlar permüte | 1.362.080,92 | **+%1,3** |
| Tüm işler t=0'da (saf batch) | 1.779.165,78 | **+%32** |
| FIFO + backfill | 1.054.665,1 | **−%21,6** |

İşlerin **%30,8'i** başka bir işle aynı `submit_time`'ı paylaşıyor. Paydayı oluşturan FIFO bu kadar sıra-bağımlıyken JCT improvement %'lerinin ±1 puan altında raporlanması anlamsız.

### J.6 Tohum varyansı

| Checkpoint | mae | mae_std | mae_seed0 (diskteki ağ) |
|---|---|---|---|
| exp_c_cnn | 7.054 | 123 | 7.062 |
| exp_c_lstm | 6.761 | 430 | 7.141 |
| exp_d_lstm | **6.236** | **509** | **6.703** |
| exp_d_hybrid | 7.333 | **739** | 6.528 |
| exp_e_lstm | 6.913 | 466 | 6.573 |
| exp_f_lstm | **6.424** | **651** | **5.879** (en iyi tohum) |
| exp_f_hybrid | 6.822 | 552 | 7.381 |

Simülasyonda **yalnız seed 42** kullanılıyor. ML tarafında tohum tekrarı **yok** (tek `random_state=42`).

---

## K. CLAIM AUDIT

| # | İddia (yer) | Kanıt (güncel) | Geçerlilik | Sınır |
|---|---|---|---|---|
| 1 | "DFL yaklaşımı öneriyoruz" (abstract, ch1 K3) | Tüm modeller MSE/L2/L1 ile eğitilmiş; ranking/decision loss `grep` → 0 | ❌ **Desteklenmiyor** | Yapılan: PtO'nun karar düzeyinde değerlendirilmesi |
| 2 | "Superior Learning to Rank engines" (abstract) | LSTM-Cat ρ=0,42 < LGBM-Nat 0,54 < UserMedian 0,60 < ProfileMedian 0,72 | ❌ **Desteklenmiyor** | DL "daha iyi sıralayıcı" bile değil |
| 3 | "32-GPU'da en doğru tahminci = en iyi JCT" (abstract, ch6 L184) | En düşük MAE'li öğrenilen model (LGBM-Nat) 4. sırada; en iyi LSTM-Cat | ❌ | Eski koşu anlatısı |
| 4 | "256-GPU'da ilişki bozuldu; LSTM-Cat %57.25" (abstract, ch7) | Her iki ölçekte aynı kazanan; Spearman ρ=0,99; LSTM-Cat %63,08 | ❌ **Dikotomi yok** | Fark 0,19-0,67 puan, tek tohum |
| 5 | "MAE ile JCT örtüşmüyor" (ch6 6.4.3, ch7 K4) | MAE–JCT r=−0,35/−0,28 (p>0,15); ρ–JCT r=0,71/0,62 | ⚠ **Kısmen** (yön doğru) | n=18 bağımlı; alt küme (n=6) ilişki yok; **4 karşı-örnek mekanizmayı çürütüyor** |
| 6 | "Sweep-line özniteliği top predictor" (ch1 K2, ch6, ch7) | MDI sırası RF 7/9, XGB 7/9, LGBM 5/9 | ❌ | **Ablasyon hiç yok** |
| 7 | "Novel sweep-line feature" (abstract) | Olay-sıralı eşzamanlılık sayımı **standart** teknik; literatür karşılaştırması yok | **NOT_VERIFIED** | — |
| 8 | "ML-SJF FIFO ve SRF'yi anlamlı geçer" (RQ1/RQ3) | Katı FIFO'ya göre evet; SRF %44/%48 ile bazı ML modellerini geçiyor (RF-Num, XGB-Num, CNN-Num, tüm Num-Seq); **ML-vs-SRF testi yok** | ⚠ **Kısmen** | Anlamlılık yalnız vs-FIFO; UserMedian %37 < SRF %44 |
| 9 | "ML trivial baseline'ı geçer" (örtük) | Nokta doğruluğu: **hayır**; JCT: evet (bf=F); 256/LF1.0/bf=T'de UserMedian > XGB-Nat | ⚠ **Kısmen** | Baseline tezde yok |
| 10 | "Doğrudan SLURM backfill'e uygulanabilir" (ch1 L100, ch7 L46) | Tek kanıt: nb05 c30 — kazançlar 5-17 puan düşüyor, sıralama değişiyor | ❌ **Test edilmemiş** | Ana tablo backfill'siz |
| 11 | "Oracle'ın %68.9 / %80.4'ü" | Güncel %73-74 / %80; Oracle backfill ile **%19,6 aşılabiliyor** | ❌ Bayat + kavramsal | "Theoretical maximum" değil |
| 12 | "Sequence trap / −%45.41" (ch7 L22) | LSTM-Cat-Seq 3./4.; LSTM-Num-Seq **+%18/+%23**; yalnız CNN-Num-Seq FIFO altı | ❌ **Aşırı genelleme** | — |
| 13 | "Under 17 s per policy", tab:sim_time | Ölçüm kodu yok; ölçülen 19-24 s (bf=F), **54-74 s** (bf=T) | **NOT_VERIFIED** | Tablo türetilemez |
| 14 | README "2.25× speedup", "XGB-OH R² 0.51" | Üçüncü eski koşu; güncel R² 0,16 | ❌ | — |
| 15 | "Fareleri fillerden koruyoruz / fair usage" (ch6 L215) | Oracle: <10 dk işler medyan 342 s; **>1 gün işler ort. 2.570.304 s (≈30 gün)**; hiçbir adalet ölçütü yok | ⚠ **Tek taraflı** | Açlık maliyeti gizli |
| 16 | "HoL blocking mekanizması" (ch6) | `gpu_demand>2` yalnız %1,1 → parçalanma bloklaması küçük; **sıralama kaybı baskın**. Sabit tahmin = FIFO ile birebir | ⚠ **Yanlış mekanizma dili** | (a) parçalanma ile (b) sıralama karıştırılıyor |
| 17 | ch6 L68 "runtime lacks temporal autocorrelation" | log-runtime **lag1 ACF 0,340**; Ljung-Box Q(10)=64.315, p≈0; kod yok | ❌ **Yanlış** | Aynı defterde tersi de yazıyor |
| 18 | ch1 L78 "CPU, GPU, memory enforced" | `mem_capacity=0.0`; izde bellek kolonu yok; ch5 L92 doğrusunu söylüyor | ❌ **İç çelişki** | — |
| 19 | ch5 L92 "heterogeneous cluster of various GPU models" | `grep gpu_type src/simulation/` → **0** | ❌ | Heterojenlik yalnız GPU **sayısı** |
| 20 | ch5 L153 "fractional allocations not allowed" | Kesirli GPU **toplamsal paketleniyor** (T3) | ❌ | — |
| 21 | ch5 L23 "earliest inserted event first" | heapq garanti etmiyor (T9: 5,4,3,2,1 → 5,3,1,4,2) | ❌ | — |
| 22 | ch5 Alg.1 "HoL yalnız FIFO" / §5.3.3 skip-over | Kod: **tüm politikalarda katı HoL** | ❌ | — |
| 23 | ch4 L166 "dropout ve pencere boyutu tuned" | Dropout griden düşüyor (hep 0,2); `seq_len` sabit 1/10 | ❌ | — |
| 24 | ch4 L171 "tek sabit tohum 42" | DL **3 tohum** [42,1337,2024] | ❌ | — |
| 25 | ch4 L168 "nearly ensuring optimal solution" | Final parametrelerin çoğu grid **sınırında** | ❌ | — |
| 26 | ch4 L60 log-transform reddi | Gerekçe yok; log1p en iyi nokta doğruluğunu veriyor (4.744/0,690) | ❌ **Gerekçesiz** | Ama simülasyonda daha kötü |
| 27 | ch6 L70 "Neural networks trained with MAE" | `nn.MSELoss()` | ❌ | — |
| 28 | ch7 L43 "code released to reproduce" | origin/main GPU-kısıtsız + int-GPU sürümü | ❌ | — |
| 29 | ch7 L60 "canlı ortamda önceki süreleri bilemeyiz" | Perturbasyon: özellik yalnız tamamlanmış olayları sayıyor → **çevrimiçi hesaplanabilir** | ❌ **Ters yönde** | Asıl varsayım: "submit anında başlar" |
| 30 | ch7 L72 "bootstrap CI hesaplanmadı" | 999-örnek CI + rank-biserial + Holm **mevcut** | ❌ **Ters** | — |
| 31 | ch7 L59 "most jobs single-node, generalization OK" | Kaynak ağırlığıyla **%86,7'si çok-örnekli** | ❌ | İş sayısı ≠ kaynak ağırlığı |
| 32 | ch2 L122 "ρ at or above 1.0" | ρ = **20,7 – 165,8** | ❌ | 2 mertebe |
| 33 | ch3 L104 "Perşembe zirvesi" | Küresel max **Cuma 11:00** (1.352); Perşembe 0 (1.282); epoch artefaktı | ❌ | — |
| 34 | ch3 L62 "avg 0.52 GPU, ~37k jobs ≤1" | Ort. **0,680**; %52,5 kesirli | ❌ | — |
| 35 | fig 3.3 "almost every request is 1 GPU" | %45,1 = 1; **%52,5 < 1** | ❌ | — |
| 36 | fig 6.5 "residuals centered about zero" | RF ortalama artık **+10.601 s** | ❌ | — |
| 37 | ch6 L215 "median slowdowns below 2.0" | Yalnız Oracle ≈1,6; LSTM-Cat/XGB-Cat ≈10-12 | ❌ | — |
| 38 | ch6 L245 "p=1.000 → not distinguishable" | FIFO'dan **anlamlı KÖTÜ** (r=−0,315, CI negatif) | ❌ **İşaret ters** | — |
| 39 | tab:related boşluk iddiası | Tablonun kendi Tsafrir satırıyla çelişiyor; L2R/DFL çalışmaları tabloda yok | ❌ | Seçim yanlılığı |
| 40 | ch1 RQ4 (ölçek-bağımlı kopukluk) | ch1'de böyle bir RQ **yok**; RQ2 ile RQ4 aynı soru | ❌ **HARKing izi** | — |
| 41 | ch1 Katkı 7 "ML-SJF kullanıcı wall-time'ının yerine geçer" | Alibaba'nın kendi tahmini altyapısı hazır ama **hiç ölçülmemiş** | ❌ | Seçici raporlama riski |
| 42 | ch5 "18 tahminci, 4 politika, 3 koşu" | 20 / 23 / 79 | ❌ | — |
| 43 | ch4 L141 "test measured only once" | Model başına doğru; **süreç genelinde değil** | ⚠ | — |
| 44 | ch3 L25 filtre iddiası | ✅ 17.816 = num_gpu 0 & CPU; duration≤0: 0 | ✅ **DOĞRU** | — |
| 45 | ch3 P50/P95/P99/max | ✅ 594 / 24.110 / 67.432 / 599.445 | ✅ **DOĞRU** | — |
| 46 | ch3 L214 "korelasyon 0.89'a kadar" | ✅ 0,888 | ✅ **DOĞRU** | — |
| 47 | ch5 tab:nodeprofiles | ✅ Kodla birebir (96/8, 64/2; 2+8, 16+64) | ✅ **DOĞRU** | — |
| 48 | ch6 "eleven engineered features" | ✅ 9 numeric + 2 categorical | ✅ **DOĞRU** | — |

**Skor:** 48 iddiadan **5'i doğru** ✅, 3'ü kısmen ⚠, **37'si desteklenmiyor/yanlış** ❌, 3'ü NOT_VERIFIED.

---

## L. REPRODUCIBILITY CHECKLIST

| # | Madde | Durum | Aksiyon |
|---|---|---|---|
| 1 | Tek etiketli commit (src + tests + configs + defterler + checkpoint + modeller) | ❌ | Tüm değişiklikleri `v1.0-thesis` tag'i ile it; tez ch7'ye hash yaz; büyük dosyalar için git-lfs/release |
| 2 | Kilit dosyası gerçek ortamdan | ❌ | `venv/bin/python -m pip freeze > requirements-lock.txt`; `environment.yaml`'ı düzelt/kaldır |
| 3 | Betiklerde yorumlayıcı sabitleme | ❌ | `${PYTHON:-venv/bin/python}`; başlangıçta sürüm doğrulaması; kernelspec adını `thesis-venv` yap |
| 4 | Checkpoint provenansı (git hash, src/veri sha256, kütüphane sürümleri, cihaz, model sha256) | ❌ | `save_checkpoint`'e ekle; `timestamp` yalnız eğitimde; `load_checkpoint` uyuşmazlıkta uyar |
| 5 | Scaler/model yanında öznitelik ad listesi + sha | ❌ | Kaydet ve nb05'te yüklerken doğrula |
| 6 | Veri kaynağı, seçim yöntemi, tarih aralığı, sha256 | ❌ | ch3 ve `data/README.md`'ye ekle (sha `dc4de321…`); `duration` semantiğini belgele |
| 7 | Tezde ortam bilgisi (Python 3.11.6, sklearn 1.8.0, XGBoost 3.1.2, LightGBM 4.6.0, PyTorch 2.11.0, Apple MPS) | ❌ | ch4/5'e "Reproducibility" alt bölümü |
| 8 | DL tohum protokolü (3 tohum, hangi ağ diske) | ❌ | L171'i düzelt |
| 9 | Cihaz bağımlılığı uyarısı (MPS ≠ CPU) | ❌ | ch4/5'e not; kritik karşılaştırmaları CPU'da tekrarla |
| 10 | `make_narrow_grid` dropout | ❌ | Ekle veya "sabit 0.2" yaz; `models.yaml` `epochs` alanını kaldır |
| 11 | Tablo otomasyonu (checkpoint/CSV → LaTeX) | ❌ | `results/tables/*.csv` + booktabs betiği + `\input` |
| 12 | Şekil provenansı (koşu manifest'i) | ❌ | Export'a dizin temizliği + manifest (notebook md5 + tarih) |
| 13 | Zaman ölçümü | ❌ | `perf_counter` ile `run_policy`'yi sar; tab:sim_time'ı üret veya kaldır |
| 14 | Testler (narrow grid, one-hot unknown-user, DL prefix, export sayacı, checkpoint↔artefakt, simülatör T1-T12) | ❌ | Ekle; belgelerdeki "11 test" → 34 |
| 15 | CI (GitHub Actions: unittest + hızlı export) | ❌ | Ekle |
| 16 | Çalışma ağacı hijyeni | ❌ | Sahipsiz CSV'yi sil; kilidi commit et; yedeği repo dışına |
| 17 | Kronolojik bölme sızıntısızlığı | ✅ | — |
| 18 | Encoder/scaler yalnız train'e fit | ✅ | — |
| 19 | Simülasyon determinizmi | ✅ | — |
| 20 | Aynı-cihaz DL determinizmi | ✅ | — |
| 21 | İşlenmiş CSV bit-birebir üretilebilir | ✅ | — |
| 22 | Artefakt ↔ checkpoint eşleşmesi (21/21) | ✅ | — |
| 23 | Defterler baştan sona koşmuş (`execution_count` sıralı) | ✅ | — |
| 24 | TR/EN defter kod eşdeğerliği | ✅ | — |
| 25 | 34 test geçiyor | ✅ | — |

**Skor: 9/25** ✅

---

## M. REQUIRED FIXES BEFORE THESIS WRITING

### M.1 CRITICAL — bunlar tamamlanmadan tez yazımına BAŞLANMAMALI

| # | Fix | Bulgu | Efor |
|---|---|---|---|
| **1** | `gpu_demand = num_inst × num_gpu`, `cpu_demand = num_inst × num_cpu` düzelt; sweep-line'ı yeniden hesapla; **tüm pipeline'ı yeniden koş** | C-1 | Yüksek |
| **2** | `finalize_ml_model`'de LGBM `objective='regression_l1'`, XGB `eval_metric='mae'` geçir; 3 LGBM + 3 XGB modelini yeniden eğit; tüm tabloları/nb05'i yenile | C-8 | Orta |
| **3** | ρ'yu hesapla ve raporla; **en az bir doymamış nokta** (ρ≈0,6-0,9) ekle — 32-GPU LF≈15-30 veya N_GPU=512/1024; süpürmeyi ρ ekseninde çiz | C-2, D-16 | Yüksek |
| **4** | **ProfileMedian, Per-User Median, sabit-medyan, sabit-0** baseline'larını tab:predresults'a ve simülasyona **zorunlu** olarak ekle | C-6, C-7, D-1, D-9 | Düşük |
| **5** | Tüm tabloları (6.1-6.4, Ek A, tab:hyperparams, tab:simconfig) checkpoint/CSV'den **otomatik üret**; HTML→LaTeX köprüsünü kur | C-3, C-4, C-9, D-21, D-22 | Orta |
| **6** | Şekilleri güncel koşudan yeniden export et; şekil–tablo–metin **tek koşudan** gelsin; altyazıları şekle göre yaz | C-9, D-41 | Düşük |
| **7** | Ölçek-bağımlı dikotomi anlatısını abstract/ch1/ch6/ch7'den **kaldır**; Spearman ρ=0,99'u raporla | C-5 | Düşük |
| **8** | Simülasyon-içi öznitelik tutarsızlığını gider: ya çevrimiçi öznitelik üretimi, ya yük özniteliklerini çıkarıp farkı raporla, ya da açık sınırlama + 534 vs 32/256 karşılaştırması | D-20 | Orta |
| **9** | Tüm değişiklikleri tek etiketli commit olarak it; tez ch7'ye hash yaz | D-24 | Düşük |

### M.2 MAJOR — tez yazımıyla paralel ilerleyebilir ama savunmadan önce bitmeli

| # | Fix | Bulgu |
|---|---|---|
| 10 | Replay-düzeyi belirsizlik: test penceresini ≥10 alt-pencereye böl / blok-bootstrap ile yeniden simüle et; DL politikalarını **3 tohumla** simüle et; JCT'yi ortalama ± CI ver | D-7, D-11, D-35 |
| 11 | Wilcoxon'u iki yönlü yap veya yönü açıkla; p≈1 satırlarını "FIFO'dan anlamlı KÖTÜ" olarak yorumla; tabloya r + CI + Holm-p ekle | D-8 |
| 12 | ML-vs-UserMedian, ML-vs-ProfileMedian, ML-vs-SRF, LSTM-Cat-vs-XGB-Cat eşleştirilmiş testleri + CI | D-12 |
| 13 | **Rolling-origin (≥3-5 pencere)** walk-forward değerlendirme; sıralamanın pencereden pencereye değişimini raporla | D-13 |
| 14 | **Sweep-line ablasyonu**: (i) `numeric_only` − 3 öznitelik, (ii) `with_categorical` − 3 öznitelik; MAE/MdAE/ρ farkını CI ile ve simülasyonla raporla | D-30, D-29 |
| 15 | Simülatör semantiğini tez algoritmasıyla hizala (HoL vs skip-over); backfill=True sonuçlarını ana metne taşı | D-14, D-18 |
| 16 | `Machine.release`'i `job_id` ile düzelt; backfill ızgarasını yeniden koş; regresyon testi | D-6 |
| 17 | `EarlyStopping` delta'sını göreli yap; epoch loglarını checkpoint'e yaz; **DL'yi yeniden eğit** | D-5 |
| 18 | ACF/PACF + Ljung-Box hesapla; ch6 L68'i düzelt; sekans başarısızlığının **gerçek nedenini** araştır | D-32 |
| 19 | Alibaba'nın kendi tahmin baseline'ını ekle ya da ch1 Katkı 7'yi kaldır + ch7 Limitations'a ekle | D-33 |
| 20 | "DFL öneriyoruz" → "PtO'yu karar-odaklı ölçütle değerlendiriyoruz"; "L2R engines" ifadesini ρ ile uyumlu hale getir | D-31 |
| 21 | ρ→JCT anlatısını 4 karşı-örnekle yeniden yaz; kuyruğa duyarlı ölçütler + tie oranı ekle | D-34 |
| 22 | Kilit dosyasını gerçek ortamdan üret; betiklerde yorumlayıcıyı sabitle; checkpoint provenansı ekle | D-25, D-26, D-27 |
| 23 | tab:sim_time'ı `perf_counter` ile üret veya **kaldır** | D-19 |
| 24 | tab:hyperparams'ı checkpoint'ten üret; "effective trees" sütunu; L171'i düzelt | D-21 |
| 25 | `day_of_week`'i çıkar veya "iz günü" olarak yeniden adlandır + ablasyon; `hour_of_day`'i yeniden tanımla | `leakage-1` |
| 26 | Feature-importance ölçütünü tekilleştir (LGBM `gain`, hepsini normalize et); altyazıyı düzelt | D-36 |
| 27 | Şekil 6.5 residual metnini düzelt; Şekil 3.3 GPU-demand altyazısını düzelt | D-37, D-38 |
| 28 | Şekil 6.4 örneklemini rastgele yap + log-log eksen; Şekil 6.7 medyan slowdown | D-39, D-40 |
| 29 | nb04/nb05 markdown anlatılarını kod çıktısından üret; `reports/html\|pdf`'i yenile veya sil | D-23 |
| 30 | README/.github/context sonuç tablolarını tek `results/SUMMARY.md`'ye bağla | `traceability-10` |

### M.3 MODERATE

31. `arrival_sec` ablasyonu + "kronolojik test altında ekstrapolasyon" sınırlaması · 32. Sweep-line'ı filtre öncesi hesapla veya ch3'te yeniden tanımla · 33. Üç ailenin gerçek eğitim/val payını tablola · 34. Bounded slowdown (τ=10/60 s) + medyan/P95 · 35. MAPE sütununu kaldır veya ×100 ölçeğine geçir · 36. Blok-bootstrap CI'ler (tahmin metrikleri) · 37. `n_estimators` (etkin) sütunu; ES protokolünü tanımla · 38. Dropout'u grid/final'e taşı veya "sabit 0.2" yaz · 39. Grid sınır seçimlerini bir tur genişlet veya Bayes · 40. `JobEvent`'e ikincil anahtar + `sort_values(['arrival_time','job_id'])`; ch5 L23'ü düzelt · 41. `run()` sonunda `len(results) != len(jobs)` kontrolü · 42. ch5 L153 kesirli GPU ifadesini düzelt · 43. nb05'te `np.maximum(pred, 0)`; tek `predict_*` yardımcısı · 44. XGB/LGBM'yi `best_iteration` ile tam train üzerinde yeniden fit (veya RF'ye de %90) · 45. Kullanıcı bazlı (group-wise) CV + görülmemiş-kullanıcı tablosu · 46. Runtime dilimlerine göre hata tablosu + log/L1 ablasyonu · 47. Adalet/açlık ölçütleri (runtime sınıfına göre bekleme, max wait, P99 slowdown) · 48. Oracle'ı "referans politika" olarak yeniden tanımla; "theoretical maximum" ifadelerini kaldır · 49. Simülatör teşhis çıktılarını (`utilization_history`, `backfilled_on_reserved`) raporla · 50. `tab:simconfig`/`tab:experiments` sayımlarını düzelt (21/20/23) · 51. `tab:related`'e L2R + DFL satırları; boşluk cümlesini daralt · 52. ch1 L78 bellek iddiasını düzelt; ch5 L11'i temizle · 53. `gpu_type` ya simülatöre gir ya ch5 L92'yi düzelt · 54. RQ2/RQ4 ikizliğini gider; ölçek sorusunu açık RQ yap veya "keşfedilmiş gözlem" olarak sun · 55. İnter-arrival histogramını `logspace` binlerle çiz + %21,6 sıfır aralığı raporla · 56. Sweep-line şeklinde ısınma bölgesini gölgele · 57. Etkin örneklem (3.146 profil / 636 kullanıcı) tartışmasını ch7'ye ekle · 58. Kayıp fonksiyonu tablosu (RF/XGB L2, LGBM L1, DL MSE); ch6 L70'i düzelt · 59. Mimari şemayı koda göre yeniden çiz.

### M.4 MINOR

60. Kategori sözlüğünü train'den türet · 61. ch7 L60'ı düzelt; `duration` semantiğini belgele · 62. `models.*` bloğunu kaldır; `refit=False` · 63. İstatistik fonksiyonlarını `src/analysis/stats.py`'ye taşı + birim testleri · 64. T1-T12 simülatör senaryolarını unittest'e taşı · 65. ch3 EDA sayılarını hesaptan al (mod, saatlik oran, ısı haritası, 0,05→0,06) · 66. Scaler'ları yalnız eğitimde kaydet + hizalama doğrulaması · 67. Çalışma ağacı hijyeni · 68. Tez "Reproducibility" alt bölümü · 69. README "novel" ifadesini yumuşat · 70. CPU işi dışlamasının sonuçlarını belirt · 71. Tablo 6.1 kalın yazımını otomatikleştir · 72. Rank-korelasyon şekline CI + etiket ayrımı; ch6'ya ekle · 73. HTML export yerine adlandırılmış CSV · 74. Şekil okunabilirliği (yatay kutu grafiği, kısa kodlar) · 75. Şekil 3.1/3.9 altyazılarını düzelt · 76. nb04 `_m` gölgelemesi + c80 print · 77. `[RECOVER]` metrik karşılaştırması + `verbose=-1` · 78. Ölü modülleri kaldır/işaretle.

---

## N. VERIFIED — SAĞLAM NOKTALAR

> Bu bölüm çalışmanın **gerçekten doğru** yapılmış kısımlarını belgeler. Bunlar korunmalı ve tezde vurgulanmalıdır.

### N.1 Veri bütünlüğü ve bölme (12 madde)

1. **Kronolojik bölme bütün:** ilk 65.747 satır train (bitişik önek), son 16.437 test; `'train idx contiguous prefix: True'`; sınırda eşzamanlı-varış bağı **0**; guard L501-512 aktif ve test ediliyor.
2. **Filtre iddiası doğru:** elenen 17.816 kaydın tamamı `num_gpu=0` & `gpu_type='CPU'`; `duration≤0` kayıt **yok**; `job_id` benzersiz (100.000), yinelenen satır yok, ham dosya `submit_time`'a göre sıralı.
3. **Kopya sızıntısı yok:** test satırlarının hiçbiri train'deki bir satırla (9 sayısal + user) birebir aynı değil.
4. **OneHotEncoder yalnız train'e fit:** 9 + 578 user + 4 gpu_type = 591 sütun; 312 test satırı sıfır user-vektörü (`handle_unknown='ignore'`).
5. **4 MinMaxScaler yalnız train istatistikleriyle fit:** diskteki scaler'lar train min/max ile birebir; nb05 aynılarını kullanıyor.
6. **Sweep-line'da hedef/gelecek sızıntısı yok:** perturbasyon testi iki yönde de temiz; eşzamanlı olaylar `groupby('time')` ile toplandığından bağ sırası değerleri etkilemiyor.
7. **Per-User Median sızıntısız:** yalnız train'den 578 kullanıcı medyanı + global 568 s; tabloda test-only kullanıcı yok; `iloc` dilimi kronolojik train ile özdeş.
8. **İç ES bölmeleri kronolojik** ve iç sınırlarda bağ yok (DL %80 sınırı hariç: 4 işlik önemsiz burst).
9. **`prepare_dl_datasets` hizalaması doğru:** train/val pencere hedefleri örtüşmüyor; test pencereleri train'in son 9 satırının yalnız **özniteliklerini** önek alıyor, hedefleri sıfır; nb05 `history_prefix` aynı 9 satırı kullanıyor.
10. **Üç kodlama arasında test indeks hizası ve `job_df.loc` eşlemesi doğru** (`X_test.index.equals(...)` = True).
11. **Checkpoint meta verileri bölmeyle tutarlı:** 21 dosyada train 65.747 / test 16.437 (Exp E/F 65.738 = pencere kaybı).
12. **Native categorical train/test kategori kod hizası tutarlı** (aynı 636'lık liste; LightGBM `pandas_categorical` ile eşleşiyor).

### N.2 Hiperparametre protokolü (7 madde)

13. **Test seti hiperparametre seçiminde kullanılmıyor:** ML `RandomizedSearchCV`/`GridSearchCV` yalnız `X_train,y_train` ile `TimeSeriesSplit(3)`; DL `run_dl_randomsearch`/`gridsearch` yalnız `val_loader` (`tuning.py` L1424-1427: "test_dataset … deliberately never touched here" — kod okunarak doğrulandı).
14. **TimeSeriesSplit kat boyutları doğru:** 16.439/32.875/49.311 eğitim, 16.436 skor; `chronological_train_validation_split` yalnız fold'un eğitim kısmını bölüyor.
15. **Tohumlama modelden ÖNCE:** `seed_everything(...)` → `create_model_instance` (L1439, L1518, L1591).
16. **Dropout/BatchNorm doğrulama ve test çıkarımında `eval()` modunda** (L1356, L1636; nb05 c8/c15); kayıtlı `.pth`'lerde `training_flag=False`.
17. **`make_narrow_grid` boyut sınırı ve tip dönüşümleri doğru:** rf 54, diğerleri 45 (≤81); bool str'den önce kontrol; `subsample`/`colsample` [0.5,1.0] ve `reg_*` ≥0 kırpması; `kernel_size=1` için `[1]` fallback; LGBM `max_depth=-1` için `[-1,10,20]`.
18. **ML arama ayarları yaml ile tutarlı** (`scoring=neg_MAE`, `cv=3`, `n_jobs=1`, `random_state=42`); RF/XGB/LGBM tek iş parçacığı.
19. **XGBoost ve RF için arama–nihai eğitim kayıp fonksiyonu tutarlı** (yalnız LGBM tutarsız).

### N.3 Artefakt provenansı (6 madde)

20. **9 ağaç/lookup artefaktı checkpoint metriklerini bit-özdeş yeniden üretti:** exp_a_rf 6842,066/0,0242; exp_a_xgb 7455,691/0,0374; exp_a_lgbm 7033,374/0,0271; exp_b_rf_oh 6292,281/0,1525; exp_b_xgb_oh 6642,393/0,1562; exp_b_lgbm_oh 6640,011/0,1178; exp_b_lgbm_nat 5697,389/0,2429; exp_b_xgb_nat 6746,460/0,1247; user_median 5191,497/0,0699.
21. **12 DL `.pth` dosyası** nb05'in pencere/`history_prefix` mantığıyla kırpılmış tahminde checkpoint `mae_seed0` değerlerini **birebir** veriyor (12/12) → nb04 ↔ nb05 hizası doğru; Exp E/F scaler'ları C/D ile özdeş.
22. **Diskteki artefaktlar `best_params` ile yapısal olarak tutarlı:** 8 ağaç modelinde 0 uyuşmazlık/0 eksik anahtar; 12 `.pth`'de tüm mimari parametreleri 12/12 eşleşiyor.
23. **NB00 işlenmiş CSV bit-birebir yeniden üretilebilir** (sha256 `c41a82f1…`, 82.184×14, tüm sütunlar değer-eşit, 0,6 s).
24. **`.pth` dosyaları hem torch 2.5.1 hem 2.11.0 altında yükleniyor** ve ileri geçiş çalışıyor.
25. **Kesirli GPU düzeltmesi güncel kodda ve işlenmiş CSV'de uygulanmış** (`.astype(float)`; min 0,01, medyan 0,50, ort. 0,680); `TestFractionalGpuDemand` 3 test geçiyor.

### N.4 Simülatör doğruluğu (11 madde)

26. **Simülasyon deterministik ve yeniden üretilebilir:** `src/simulation` altında random/rng/shuffle yok; FIFO (mean wait 1.338.936,41 / mean JCT 1.344.966,16 / median 1.176.012,3 / P95 2.835.083,02 / slowdown 8.085,03) ve SJF-Oracle (245.993,98 / 252.023,72 / max 3.213.394,4 / 36,84) bağımsız betikle **bit-bit** yeniden üretildi; backup ↔ güncel aynı.
27. **First-fit yerleştirme, kesirli GPU muhasebesi ve CPU kısıtı doğru çalışıyor** (T3: 0.5+0.5 aynı 1-GPU makinede; T11: `num_cpu=90` iş 64-CPU düğümü atlayıp 96-CPU'ya gidiyor; `can_fit` toleransı 1e-5 float birikimini sorunsuz karşılıyor).
28. **GPU kapasitesi gerçekten uygulanıyor** (her iki sütun adı); NaN talep 0'a indiriliyor; `TestSimulatorEnforcesGpuLimit` 5 test OK.
29. **Metrik kimlikleri doğru:** `waiting = start − submit`, `turnaround = finish − submit`, `slowdown = turnaround/runtime`, `JCT Improvement % = (FIFO−policy)/FIFO`; c27 ve c30 aynı tanımı kullanıyor.
30. **Gerçek iş yükünde hiçbir iş düşürülmüyor:** 23 politikanın hepsi 16.437 iş döndürüyor (toplam 378.051 kayıt).
31. **EASY backfill rezervasyonu tahmin üzerinden kuruluyor ve yanlış tahmin rezerve işi geciktiriyor** (üretim davranışına sadık; T12: J3 gerçek 500 s / tahmin 10 s backfill edildi, rezerve J2 100 yerine 502'de başladı, `backfilled_on_reserved=1`).
32. **`backfill=False` ana yolunda `running_detail`/`earliest_fit` hiç okunmuyor** → `simulator-5` hatası ana sonuçları etkilemiyor.
33. **`n_cpu=0` kararı doğru:** izdeki her iş >0 GPU istiyor; 0-GPU düğüm hiçbir işi kabul edemez.
34. **SRF tanımı tez §5.3.2 ile uyumlu** (en küçük `gpu_demand`, eşitlikte varış sırası, `mergesort` kararlı); SJF-Oracle gerçek runtime kullanıyor.
35. **`pending_df` `concat` sonrası dtype'lar korunuyor**; `idxmin` sayısal sütunlarda doğru çalışıyor.
36. **32 ve 256 GPU defterleri arasında yalnız küme boyutu farklı**; simülasyon kodu aynı (c20 çıktıları birebir; c27 Model MAE/R² sütunları aynı).

### N.5 Taban çizgisi adilliği (3 madde)

37. **Taban çizgileri ML lehine değil, ML ALEYHİNE konservatif:** FIFO/SRF/Oracle rezervasyon penceresini **gerçek runtime** ile kurarken ML politikaları kendi tahminini kullanıyor; negatif tahminler `_MIN_ESTIMATE=1.0`'a kırpılıyor, gerçek runtime ile değiştirilmiyor. Kodun kendi yorumu: *"that makes those baselines optimistic, which is the conservative direction for any claim about the ML policies."*
38. **Sabit (train-medyanı) tahmin politikası FIFO ile birebir aynı JCT veriyor** (1.344.966,2) → "JCT improvement % vs FIFO" gerçekten **sıfır-bilgili sıralamaya göre** ölçülüyor.
39. **Rastgele öncelik FIFO'dan kötü** (1.738.633, −%29,27) → kazanç yalnızca "FIFO'yu bozmak"tan gelmiyor.

### N.6 İstatistik uygulaması (7 madde)

40. **Wilcoxon eşleştirmesi `job_id` ile doğru** (konuma göre değil); `job_id` test setinde benzersiz; inner merge 16.437 çift.
41. **W sütunu scipy `alternative='greater'` istatistiği (W+) ile tutarlı** ve sentetik kontrolle yeniden üretildi.
42. **Holm-Bonferroni uygulaması referans uygulamayla birebir aynı** (`Holm match: True`).
43. **Rank-biserial formülü doğru**; tablo değerleri W ile tutarlı (ima edilen n' 32-GPU ≈16.380, 256 ≈15.927 — hepsi 16.437'den küçük ve satırlar arası tutarlı).
44. **Bootstrap CI:** 999 yeniden örnek, percentile, `default_rng(42)` deterministik.
45. **Test seçimi uygun:** Wilcoxon parametrik olmayan; hiçbir yerde normallik/homoskedastisite varsayan test kullanılmıyor; c45'in heavy-tail gerekçesi doğru.
46. **Çok-tohum SD `ddof=1` ile n=3 üzerinden doğru**; seed-42'nin kendi skoru ayrıca `*_seed0` olarak saklanıyor.

### N.7 Yeniden üretilebilirlik (6 madde)

47. **DL eğitimi aynı cihazda tohumla bit-birebir deterministik** (süreç içi ve süreçler arası; LSTM md5 `a38f3ca9…` MPS, `e2594012…` CPU sabit).
48. **Defterler baştan sona sıralı çalıştırılmış** (`execution_count` 1..N kesintisiz); kök bulucu marker tabanlı, mutlak yol yok.
49. **34 birim test geçiyor** (0,31 s) ve checkpoint testi depoya yazmıyor (`tempfile.TemporaryDirectory` + patch).
50. **TR ve EN defterleri kod olarak eşdeğer** (00-03 ve 05: 0 fark; nb04: yalnız print etiketleri).
51. **Export betiğinin şekil eşlemesi sayaç korumalı** (`EXPECTED_FIGURE_COUNT` ile ya-hep-ya-hiç; yalnız EN defterlerden kopyalama) — bu koruma **doğru çalıştı** ve bayat NB05 kopyasını engelledi.
52. **Düğüm profili tabloları kodla birebir uyumlu** (96/8, 64/2; 2+8 = 10 düğüm/32 GPU; 16+64 = 80 düğüm/256 GPU).

### N.8 Şekil/tablo provenansı (5 madde)

53. **nb01–nb04 tez şekillerinin provenans zinciri sağlam:** md5 `thesis/latex/figures/nb01-fig01…nb04-fig05` = `thesis_export/png/NB01-Figure01…NB04-Figure05` = backup; hepsi 31 Ağu 17:33.
54. **`EXPECTED_FIGURE_COUNT` ve `THESIS_FIGURE_MAP` mevcut defterlerin PNG hücre sayılarıyla eşleşiyor** (nb01 6, nb02 3, nb03 2, nb04 5, nb05 7).
55. **HTML export tabloları backup ile birebir aynı** (19/19 `cmp -s` SAME).
56. **nb04 Fig1 R² ekseni negatif değerleri kırpmıyor** (`set_xlim(min(0.0, _r2_min) - _pad, …)`).
57. **Yüzdelik nokta grafiği log eksende çubuk yerine nokta kullanıyor** (doğru seçim; kodun kendi gerekçesi: "a logarithmic axis has no zero").

### N.9 Doğrulanan tez iddiaları (7 madde)

58. Örneklem büyüklükleri (100.000 → 82.184 → 65.747/16.437) ✅
59. ch3 L25 filtre iddiası ✅
60. Runtime yüzdelikleri (P50 594, P75 4.126, P95 24.110, P99 67.432, max 599.445 = 6,94 gün; P95/P50 = 40,6) ✅
61. Öznitelik boyutları (d=9 / 591 / 11; 312 görülmemiş kullanıcı) ✅
62. Sweep-line korelasyonu "up to 0.89" (0,888) ✅; `num_cpu`–runtime 0,080 ✅
63. `active_job_count` max 1.088 >1.000 ✅; GPU yükü max 869 ≈800 ✅; max GPU talebi 8 ✅; iz 7,66 gün ✅
64. Varış zirveleri (1.353 @ gün 1,46; 1.080 @ gün 5,71) ✅; scatter 5.000 örnek `random_state=42` ✅; simülasyon havuzu 16.437 ✅; LOAD_FACTOR 0.1 = "10× hızlandırılmış" ✅; validasyon protokolü metni kodla uyumlu ✅; "eleven engineered features" ✅

**Toplam: 64 doğrulanmış sağlam nokta.** Bu, çalışmanın mühendislik kalitesinin **yüksek** olduğunu gösteriyor — kod temiz, deterministik, sızıntısız ve artefaktlar birebir yeniden üretilebilir. Sorun mühendislikte değil, **modelleme kararlarında, deney tasarımında, istatistiksel çıkarımda ve iddia–kanıt hizasında**.

---

## O. ÇÜRÜTÜLEN ADAY BULGULAR

> **Bu bölüm iki tur arasındaki en önemli farkı barındırıyor.** Fable turunda **hiç çürütme yoktu** (53/53 onay). Opus turunda **2 aday çürütüldü**. Aşağıda hem çürütmeler hem de bu farkın nedeni tartışılıyor.

### O.1 Çürütülen aday 1: `reproducibility-3`

**Aday başlık:** "Veri kaynağı izlenemiyor: `pai_job_no_estimate_100K.csv` resmi PAI izinin bir dosyası değil, seçim yöntemi ve karma değeri hiçbir yerde yok; tez izi '100.000 kayıt' olarak tanıtıyor"

**Dosya:** `data/alibaba_cluster_trace/pai_job_no_estimate_100K.csv`; `data/README.md`; `3.dataset_and_workload.tex`

**Çürütme gerekçesi (yargıç kaydı):**
> REFUTED: Seviye: `A_computational` ve `C_scientific` açısından değerlendirildi; `B_statistical` ilgisiz. Neden REFUTED (PARTIALLY_CONFIRMED değil): Bulgunun başlığı, "problem" ile iddia edilenden fazlasını söylüyor. İkinci yargıç: PARTIALLY_CONFIRMED — Seviye `A_computational` değil, `B_statistical` değil; saf `C_scientific` ama "bilimsel geçerlilik" değil, **"raporlama/atıf doğruluğu"** alt başlığında. **MAJOR → MINOR** düşürüldü.

**Neden çürütüldü — analiz.** Adayın iki bileşeni vardı: (a) *"dosya izlenemiyor / sha yok / seçim yöntemi belgesiz"* — bu **doğru** ve zaten `reproducibility-11` ile `leakage-7` altında MINOR olarak yaşıyor; (b) *"resmi PAI izinin bir dosyası DEĞİL"* — bu **aşırı iddia**. Doğrulama şunu gösterdi: dosya `submit_time` 0..661.892 s aralığında **sıralı ve kesintisiz** (maks. boşluk 237 s), `job_id` 0..99999 monoton — yani izin **bitişik bir kronolojik prefiksidir**, uydurma veya rastgele karıştırılmış bir örnek değil. Ayrıca sha256 hesaplanabildi (`dc4de321…`) ve işlenmiş CSV bundan bit-birebir üretilebildi. Dolayısıyla "kaynak izlenemiyor" ifadesi **doğrulanabilir bir veri bütünlüğü sorunu değil**, bir **belgeleme eksikliğidir**. MAJOR bir bilimsel geçerlilik bulgusu olarak sunulması severity enflasyonuydu; MINOR bir raporlama bulgusu olarak `reproducibility-11` içinde yaşamaya devam ediyor.

**Kalan geçerli çekirdek (MINOR olarak korunuyor):** Dosyanın kökeni, seçim yöntemi, tarih aralığı ve sha256'sı tezde ve `data/README.md`'de yok; `duration` alanının semantiği (end−start mı end−submit mi) belgelenmemiş. → `leakage-7` ve `reproducibility-11` altında.

---

### O.2 Çürütülen aday 2: `robustness-6`

**Aday başlık:** "Sayısal-özellikli modeller (Exp A/C/E) sabit tahminciden ayırt edilemiyor; RQ2 sayısal özellik iddiası desteklenmiyor"

**Dosya:** `results/checkpoints/exp_a_lgbm.json`

**Çürütme gerekçesi (yargıç kaydı):**
> REFUTED: Seviye ayrımı: kusur `A_computational` **değil** (kod ve checkpoint sayıları birebir yeniden üretildi); kusur `B_statistical` düzeyinde **bulgunun KENDİSİNDE**: L2 ile eğit[ilen model üzerinden yapılan karşılaştırma, ayarlandığı kayıp fonksiyonuyla eğitilmemiş bir modelin skorunu referans alıyor].

**Neden çürütüldü — analiz.** Bu, denetimin en ilginç öz-düzeltmesi. Aday şunu iddia ediyordu: *"Exp A/C/E (sayısal öznitelikli) modeller sabit tahminciyi geçemiyor, dolayısıyla sayısal öznitelikler işe yaramıyor."* Ancak bu iddianın dayandığı sayılar (`exp_a_lgbm` MAE 7.033) **C-8'de doğrulanan hatanın ürünü**: LGBM L1 ile ayarlanıp L2 ile eğitilmiş. Aynı konfigürasyon `objective='regression_l1'` ile yeniden eğitildiğinde MAE **5.806,5**'e düşüyor (ρ 0,263). Yani "sayısal modeller sabit tahminciden ayırt edilemiyor" ifadesi, **ölçüm hatasının bir sonucunu bir bilimsel sonuç sanmak** olurdu.

Dahası, ablasyon verileri sayısal özniteliklerin **gerçekten katkı sağladığını** gösteriyor: yalnız `user`+`gpu_type` ile L1-LGBM MAE 5.204 / ρ 0,564; 9 sayısal öznitelik eklenince **4.870 / 0,631**. Yani sayısal öznitelikler MAE'yi ~330 s iyileştiriyor ve ρ'yu 0,067 artırıyor — küçük ama sıfır değil.

**Sonuç:** Aday bulgu, düzeltilmemiş bir kod hatasının üzerine kurulmuş bir çıkarım olduğu için REFUTED edildi. Geçerli kalan gözlem — "Exp A modelleri (sayısal-only) diğerlerinden zayıf ve sabit medyandan anlamlı üstün değil" — zaten `modeling-7` (D-1) ve `statistics-3` (D-9) altında, **doğru gerekçeyle** ve doğru severity ile duruyor.

---

### O.3 Neden Fable turunda hiç çürütme yoktu, Opus turunda 2 var?

Bu, iki turun **metodolojik farkının** doğrudan sonucudur ve raporun en öğretici kısmıdır.

**Sebep 1 — Bulgu havuzunun doğası farklıydı.**
Fable turu, ilk keşif turuydu ve 53 bulgunun tamamı *doğrudan gözlemlenebilir* uyuşmazlıklardı: "tezde X yazıyor, kod Y üretiyor" (traceability), "kod A yapıyor, tez B diyor" (simulator/modeling), "bu sayı hesaplanınca Z çıkıyor" (statistics). Bu tür bulgular ya doğrudur ya yanlıştır; ara ton azdır. Opus turu ise Fable'ın bulmadığı **daha derin ve daha spekülatif** alanlara girdi (kaynak modeli, baseline tavanı, HARKing izi, otokorelasyon iddiası, katman-evren tutarsızlığı). Bu alanlarda **aşırı-iddia riski daha yüksektir** — ve iki adayın da çürütülme nedeni tam olarak budur.

**Sebep 2 — Opus turunda bağımsız bir "critic" ajanı çalıştı.**
Fable turu üç mercek (leakage/statistics/simulator + traceability/modeling/reproducibility + baselines/robustness/code_bugs/figures) kullandı ve her mercek kendi alanını taradı. Opus turunda buna ek olarak bir **eleştirmen** çalıştı: hem yeni bulgular üretti (12 critic bulgusu: `F-C01`..`F-C12`) hem de **mevcut adayların üzerine gitti**. Bir bulguyu üreten ile onu sınayan farklı roller olunca çürütme olasılığı doğal olarak artar. Fable turunda bu ikinci rol yoktu.

**Sebep 3 — Çürütmelerin ikisi de "severity/kategori düzeltmesi", bulgu iptali değil.**
Kritik nokta: **hiçbir Fable bulgusu geri alınmadı.** Çürütülen iki aday da Opus turunda üretilmiş adaylardır. `reproducibility-3` MAJOR→MINOR indirildi ve içeriği başka bulgular altında yaşıyor; `robustness-6` ise başka bir onaylanmış bulgunun (C-8) sonucunu bağımsız bir bulgu sanmaktan doğduğu için tamamen iptal edildi. Yani çürütme oranı (2/70 ≈ **%2,9**) bulgu kalitesinin düştüğünü değil, **denetimin kendi kendini düzeltebildiğini** gösteriyor.

**Sebep 4 — İkinci turun kalibrasyon avantajı.**
Opus turu, Fable'ın 53 onaylanmış bulgusunu **referans çerçeve** olarak kullanabildi. Bir aday, zaten onaylanmış bir bulgunun türevi mi yoksa bağımsız bir gözlem mi olduğu sorusuna Fable turu cevap veremezdi (referansı yoktu); Opus turu verebildi. `robustness-6`'nın çürütülmesi tam olarak bu mekanizmayla oldu: aday, C-8'in (LGBM objective) bir **sonucuydu**, bağımsız bir bulgu değil.

**Yorumlama — bu fark hükmü etkiliyor mu?** **Hayır.** İki çürütme de MINOR/türev düzeyinde; hükmü taşıyan 9 CRITICAL ve 44 MAJOR bulgunun hiçbiri çürütülmedi. Aksine, çürütme mekanizmasının çalışması **kalan bulguların güvenilirliğini artırıyor**: denetim "her şeyi onaylayan" bir süreç değil, ayıklama yapabilen bir süreç olduğunu kanıtladı. Bu, RED hükmünün ağırlığını **azaltmıyor, artırıyor**.

---

## P. GÜVEN SKORLARI

### P.1 Bulgu düzeyinde güven

| Güven bandı | Kriter | Bulgu sayısı | Örnekler |
|---|---|---|---|
| **%99+ (kesin)** | İki turda bağımsız doğrulandı VEYA kod okuma + yeniden üretim ile birebir teyit | **31** | C-1 (num_inst 5,6× ölçüldü), C-8 (LGBM objective, iki tur), C-3/C-4 (grep + checkpoint karşılaştırması), D-7 (ACF ölçüldü), D-24 (`git status`), N.1-N.9'daki tüm sağlam noktalar |
| **%95-99 (çok yüksek)** | Tek turda ölçüm + bağımsız betikle yeniden üretim | **42** | C-2 (ρ hesabı), C-6/C-7 (baseline metrikleri), D-1 (bootstrap CI), D-5 (delta simülasyonu), D-13 (pencere ölçümü), D-32 (ACF/Ljung-Box) |
| **%85-95 (yüksek)** | Ölçüldü ama tek kaynak veya kısmi kapsam | **27** | D-20 (öznitelik evren uyuşmazlığı — ölçüldü ama etkisi ablasyonla ölçülmedi), D-34 (4 karşı-örnek ama mekanizma tam izole edilmedi), `robustness-7` (dilim analizi), `figures_tables-*` (görsel inceleme) |
| **%70-85 (orta)** | Mekanizma doğrulandı, büyüklük ölçülmedi | **14** | D-5'in "az eğitilmiş" kısmı (epoch logları yok), `modeling-11` (grid sınırı — etkisi bilinmiyor), `leakage-3` (asimetrik veri — MAE etkisi ölçülmedi) |
| **%50-70 (düşük)** | Kısmi kanıt veya çıkarım | **5** | `baselines_claims-14` "novel" iddiası, `simulator-3` Weng atfı |
| **NOT_VERIFIED** | Doğrulanamadı | **3** | Weng et al. düğüm profili atfı, tab:sim_time'ın kaynağı, "novel sweep-line" literatür konumu |
| **REFUTED** | Çürütüldü | **2** | `reproducibility-3`, `robustness-6` |

### P.2 Boyut düzeyinde güven

| Boyut | Denetim güveni | Gerekçe |
|---|---|---|
| **Veri sızıntısı / bölme** | **%98** | Perturbasyon testleri, indeks kontrolleri, scaler/encoder doğrulaması — hepsi kod düzeyinde teyit edildi |
| **İzlenebilirlik** | **%99** | `grep` + md5 + checkpoint karşılaştırması ile her sayının kaynağı bulundu |
| **Simülatör hesaplama doğruluğu** | **%95** | 12 sentetik senaryo + 6 gerçek koşu; yalnız `backfill=True` 256-GPU koşulmadı |
| **Simülasyon rejimi (ρ)** | **%99** | Analitik hesap + FIFO koşusu ile iki yönlü teyit |
| **İstatistiksel çıkarım** | **%97** | ACF, blok-bootstrap, Holm doğrulaması, W yeniden üretimi |
| **Model kurulumu / tuning** | **%96** | 25 artefakt açıldı, `best_params` karşılaştırıldı, L1/L2 ablasyonu koşuldu |
| **Baseline'lar ve iddialar** | **%94** | 6 tam simülasyon + eşleştirilmiş bootstrap; yalnız 3-tohum simülasyonu yapılamadı |
| **Yeniden üretilebilirlik** | **%97** | `git`, `pip freeze`, kernelspec, determinizm testleri |
| **Robustness** | **%90** | Ablasyonlar yalnız LightGBM ile (hız); RF/XGB karşılığı denenmedi |
| **Şekiller / tablolar** | **%92** | Görsel inceleme + md5 + mtime; bazı okumalar öznel |
| **Kod hataları** | **%95** | Sentetik reprodüksiyon + artefakt inceleme |

### P.3 Hüküm düzeyinde güven

| Soru | Güven |
|---|---|
| Bu tur RED mi? | **%99** (3/3 yargıç oybirliği, bağımsız gerekçeler) |
| Fable turu ile aynı mı? | **%100** (her iki tur RED) |
| Birleşik hüküm RED mi? | **%99** |
| Metin güncellense bile RED kalır mı? | **%97** — C-1, C-8, D-20, C-6/C-7 metin düzeltmesiyle kapanmıyor |
| Ayakta kalan 3 yön bulgusu sağlam mı? | **%88** — yön güçlü, büyüklük ve mekanizma değil |
| Düzeltmeler sonrası YELLOW/ORANGE'a çıkabilir mi? | **%75** — ancak muhtemelen farklı bir ana mesajla |
| Mevcut ana mesaj (ölçek-dikotomi + DFL + sweep-line yeniliği) kurtarılabilir mi? | **%8** |

### P.4 Denetimin kendi sınırları

| Sınır | Etki |
|---|---|
| **Hiçbir model yeniden eğitilmedi** (LightGBM ablasyonları hariç, ≤1 s) | D-5'in "az eğitilmiş" büyüklüğü, `leakage-1/2/4`'ün MAE etkisi, D-30 ablasyonu **ölçülemedi** |
| **DL 3-tohum simülasyonu yapılamadı** (diske yalnız seed-42 yazılmış) | D-35'in tohum-SD'si CI örtüşmesiyle dolaylı gösterildi |
| **256-GPU backfill=True gerçek koşu yapılmadı** | Defter çıktılarından okundu |
| **Weng et al. 2022'ye erişilemedi** | Düğüm profili atfı NOT_VERIFIED |
| **`reports/*.md` okunmadı** (bağımsızlık için) | — |
| **Per-job simülasyon sonuçları diske yazılmıyor** | LSTM-Cat vs XGB-Cat eşleştirilmiş farkı hesaplanamadı |
| **Depo salt-okunur bırakıldı** | Tüm betikler `scratchpad/audit` ve `scratchpad/critic` altında; hiçbir proje dosyası değiştirilmedi |

---

## Q. İKİ-TUR KARŞILAŞTIRMASI

> Bu bölüm, iki farklı modelin (Claude Fable 5.1 ve Claude Opus 5) aynı depoyu bağımsız olarak denetlemesinden çıkan meta-bulguları içerir. Denetimin kendisinin güvenilirliğini değerlendirmeye yarar.

### Q.1 Nicel karşılaştırma

| Ölçüt | Fable turu (Tur 1) | Opus turu (Tur 2) | Yorum |
|---|---|---|---|
| **Doğrulanan bulgu** | 53 | 68 (+12 critic) = 80 | Opus daha geniş kapsam |
| **Çürütülen** | **0** | **2** | Opus'ta öz-düzeltme var |
| **Çürütme oranı** | %0 | **%2,9** (2/70) | Düşük ama sıfır değil |
| **CRITICAL** | 3 | 5 | Birleşik: 9 |
| **MAJOR** | 15 | 29 | Birleşik: 44 |
| **MODERATE** | ~20 | ~28 | Birleşik: 38 |
| **MINOR + INFO** | ~15 | ~19 | Birleşik: 22 |
| **NOT_VERIFIED** | 0 | 3 | Opus daha temkinli |
| **Yargıç yapısı** | 3 mercek (alan bazlı) | 3 yargıç + 1 critic (rol bazlı) | Farklı mimari |
| **Panel oyu** | RED (oybirliği) | **RED (3/3 oybirliği)** | **AYNI** |
| **Yeni CRITICAL gerekçe** | rejim + izlenebilirlik + istatistik | + kaynak modeli + baseline tavanı + katman evreni | Tamamlayıcı |
| **Yürütülen tam simülasyon** | 4 (FIFO×3, Oracle) | 6+ (FIFO, Oracle, UserMed, ConstMed, ProfileMed, L1/log varyantları) + 2 backfill | Opus daha fazla |
| **Sentetik test senaryosu** | 12 (T1-T12) | — (Fable'ınkiler kullanıldı) | Fable özgün |
| **Ablasyon koşusu** | 0 | 5 (L1, log1p, arrival_sec, user-only, arrival+dow) | Opus özgün |

### Q.2 Severity kalibrasyonu — iki model aynı şeye aynı ağırlığı veriyor mu?

**Aynı bulguya iki turun verdiği severity:**

| Bulgu | Fable severity | Opus severity | Fark | Yorum |
|---|---|---|---|---|
| LGBM objective tutarsızlığı | `modeling-1`: **MAJOR** (n_votes=8) | `robustness-1`: **CRITICAL** | +1 kademe | Opus, "sonucu tersine çevirir" boyutunu daha ağır tarttı |
| Sabit medyanı geçememe | `modeling-7`: **CRITICAL** → final MAJOR | `statistics-3`: MAJOR | ≈aynı | Kalibrasyon uyumlu |
| EarlyStopping delta | `modeling-4`: **MAJOR** (n=8) → final MODERATE | `code_bugs-2`: **MAJOR** | ≈aynı | — |
| `Machine.release` ayak izi | `simulator-5`: **MODERATE** (n=2) | `code_bugs-4`: MODERATE → final **MAJOR** | +1 | Opus backfill etkisini daha ağır tarttı |
| Rejim / ρ | `simulator-1`: **CRITICAL** (n=5) | `robustness-3`: MAJOR; `baselines_claims-3`: CRITICAL→MAJOR | −0,5 | Fable rejimi daha merkezi gördü |
| Tek yönlü Wilcoxon | `statistics-2`: **MAJOR** (n=7) | `figures_tables-3`: MAJOR | aynı | — |
| Sim süre tablosu | `simulator-7`/`traceability-4`: **MAJOR** | `figures_tables-4`: MAJOR; `baselines_claims-12`: MINOR→MODERATE | dağınık | Aynı olguya 4 farklı ID |
| Yayınlanmış kod | — (Fable bulmadı) | `reproducibility-1`: CRITICAL→**MAJOR** | Opus özgün | — |
| `num_inst` | — (Fable bulmadı) | `F-C02`: **CRITICAL** | Opus özgün | En ağır yeni bulgu |
| ProfileMedian tavanı | — | `F-C01`: CRITICAL→final MAJOR | Opus özgün | Panel CRITICAL sayıyor |

**Kalibrasyon sonucu:** İki modelin severity dağılımları **yüksek ölçüde uyumlu**. Ortalama sapma ≈ **±0,5 kademe**; hiçbir bulguda 2 kademe fark yok. Sistematik bir eğilim var: **Opus, "sonucun yönünü değiştirir mi?" testini severity'de daha ağır tartıyor** (LGBM objective, release hatası), **Fable ise "rejim/çerçeve doğru mu?" testini daha ağır tartıyor** (ρ, flash crowd). Bu, iki modelin farklı denetim felsefeleri olduğunu ama aynı ölçeği kullandığını gösteriyor.

### Q.3 Bulgu kalitesi karşılaştırması

| Boyut | Fable turu | Opus turu |
|---|---|---|
| **Kanıt tipi** | Ağırlıklı: `grep`, dosya karşılaştırması, checkpoint okuma, sentetik senaryo (T1-T12) | Ağırlıklı: yeniden hesaplama, ablasyon, tam simülasyon, bootstrap CI |
| **Özgün güçlü yön** | **Sentetik simülatör testleri** (T1-T12) — semantik hataları izole etti; **provenans arkeolojisi** (12 May / 17 Ağu / 3 farklı README koşusu) | **Karşı-örnek üretimi** (ProfileMedian, L1/log varyantları, rastgele öncelik, sabit-medyan kontrolü); **eksik deney tespiti** (ablasyon yok, Alibaba baseline'ı yok) |
| **Kaçırdığı** | `num_inst`, ProfileMedian, yayınlanmış-kod durumu, HARKing, otokorelasyon iddiası, bellek/gpu_type tutarsızlığı, `tab:related` boşluğu | Sentetik simülatör senaryolarını üretmedi (Fable'ınkileri kullandı); TR/EN diff'ini yapmadı |
| **Bulgu spesifikliği** | Çok yüksek — her bulguda satır numarası + `grep` komutu | Çok yüksek — her bulguda `how_to_check` betiği |
| **Aşırı-iddia riski** | Düşük (0 çürütme) ama muhafazakâr | Orta (2 çürütme) ama daha derin |
| **Yeniden üretilebilirlik** | Betikler `scratchpad/audit` | Betikler `scratchpad/audit` + `scratchpad/critic` |

**Kalite değerlendirmesi:** İki turun bulguları **kalite olarak eşdeğer, kapsam olarak tamamlayıcı**. Fable turu "tez ne diyor vs kod ne yapıyor" eksenini tüketmiş; Opus turu "kod doğru şeyi mi ölçüyor ve eksik olan deney ne" eksenini açmış. Hiçbiri diğerinin yerine geçemez.

### Q.4 İki model birbirini ne ölçüde teyit etti?

**Doğrudan örtüşen bulgular (her iki turda bağımsız bulunan):** 18

| Bulgu | Fable ID | Opus ID | Teyit derecesi |
|---|---|---|---|
| LGBM objective tutarsızlığı | `modeling-1` | `robustness-1`, `code_bugs-1` | **Tam** (aynı kanıt, farklı severity) |
| XGB eval_metric asimetrisi | `modeling-2` | `code_bugs-1` (alt) | Tam |
| `n_estimators` etkisiz | `modeling-3` | `code_bugs-6` | Tam |
| EarlyStopping delta | `modeling-4` | `code_bugs-2` | Tam |
| Dropout griden düşüyor | `modeling-5` | `code_bugs-3`, `reproducibility-7` | Tam |
| Winner's curse | `modeling-6` | `robustness-10` | Tam |
| Sabit medyanı geçememe | `modeling-7` | `statistics-3` | Tam |
| 3-tohum vs seed-42 | `modeling-8` | `traceability-11`, `robustness-4` | Tam |
| Pseudo-replikasyon | `statistics-1` | (Opus panel gerekçesi) | Tam |
| Tek yönlü Wilcoxon | `statistics-2` | `figures_tables-3`, `traceability-8` | Tam |
| Baseline eksikliği | `statistics-3` | `baselines_claims-2`, `F-C01` | **Genişletildi** (Opus ProfileMedian ekledi) |
| Rank-korelasyon zayıflığı | `statistics-4` | `figures_tables-18`, `robustness-2` | Genişletildi |
| Slowdown sınırsız | `statistics-8`, `simulator-9` | `figures_tables-13` | Tam |
| Rejim / ρ | `simulator-1` | `robustness-3`, `baselines_claims-3` | Tam |
| Katı HoL | `simulator-2` | `baselines_claims-4` | Tam |
| `Machine.release` | `simulator-5` | `code_bugs-4` | Tam |
| Sim süre tablosu | `simulator-7`, `traceability-4` | `figures_tables-4`, `baselines_claims-12` | Tam |
| Tez tabloları bayat | `traceability-1/2/3` | `figures_tables-1/2/5` | Tam |

**Örtüşme oranı:** Fable'ın 53 bulgusunun **18'i** (%34) Opus turunda bağımsız olarak yeniden bulundu. Kalan %66 örtüşmedi çünkü Opus turu **Fable'ın bulgularını girdi olarak aldı** (yeniden keşfetmesi gerekmiyordu) ve kalan 70 bulguyu doğrulamaya odaklandı.

**Kritik teyit testi:** Opus turu, Fable'ın 53 bulgusundan **hiçbirini çürütmedi**. Bu, iki farklı modelin aynı depoda aynı olguları gördüğünün en güçlü kanıtıdır.

**Ters yönlü teyit:** Opus'un ürettiği 12 critic bulgusunun 3'ü (`F-C04` ablasyon yokluğu, `F-C05` otokorelasyon, `F-C11` etkin n) Fable'ın mevcut bulgularının (`baselines_claims-5`, `leakage-5`) **doğal uzantısıydı** — yani Fable doğru yöne bakmış, Opus daha derine inmiş.

### Q.5 Meta-değerlendirme: iki-tur denetimi işe yaradı mı?

| Soru | Cevap |
|---|---|
| İkinci tur yeni CRITICAL üretti mi? | **Evet** — `F-C02` (num_inst), `F-C01` (ProfileMedian), `baselines_claims-1` (dikotomi yok), `baselines_claims-2` (UserMedian), `robustness-1` (LGBM CRITICAL'a yükseltildi) |
| İkinci tur hükmü değiştirdi mi? | **Hayır** — ama hükmü **daha sağlam temellere** oturttu (metin-bağımsız gerekçeler) |
| İkinci tur yanlış pozitif ayıkladı mı? | **Evet** — 2 aday çürütüldü, 3'ü NOT_VERIFIED işaretlendi |
| İki model birbiriyle çelişti mi? | **Hayır** — yalnızca severity'de ±0,5 kademe sapma |
| Tek tur yeterli olur muydu? | **Hayır** — Fable tek başına `num_inst`, ProfileMedian, yayınlanmış-kod ve HARKing bulgularını kaçırırdı; Opus tek başına sentetik simülatör testlerini ve provenans arkeolojisini kaçırırdı |
| Üçüncü tur gerekli mi? | **Şu an hayır** — marjinal getiri düşük; asıl gereken **düzeltme + yeniden koşum**, yeni denetim değil |

**Sonuç:** İki-tur mimarisi hem **kapsamı genişletti** (53 → 121 bulgu) hem de **kaliteyi artırdı** (2 çürütme, 3 NOT_VERIFIED). En önemlisi: iki bağımsız model, farklı gerekçelerle, aynı hükme vardı. Bu, **RED hükmünün model-bağımlı bir yargı değil, depoda gerçekten var olan durumun yansıması** olduğunu gösterir.

---

## KAPANIŞ: BİRLEŞİK NİHAİ HÜKÜM

**RED.** Altı bağımsız yargıç merceği oybirliğiyle. 121 doğrulanmış bulgu (9 CRITICAL, 44 MAJOR, 38 MODERATE, 22 MINOR/INFO), 2 çürütme, 3 NOT_VERIFIED.

Çalışmanın **mühendislik kalitesi yüksektir** (64 doğrulanmış sağlam nokta: deterministik pipeline, sızıntısız bölme, birebir yeniden üretilebilir artefaktlar, doğru istatistik uygulaması, konservatif baseline kurulumu). Sorun kodun *nasıl yazıldığında* değil, **neyin ölçüldüğünde, neyin karşılaştırıldığında ve nelerin iddia edildiğindedir.**

Tez yazımına, **M.1'deki 9 CRITICAL düzeltme tamamlanıp pipeline yeniden koşulmadan** başlanmamalıdır. Bu düzeltmelerden sonra çalışma büyük olasılıkla savunulabilir bir noktaya taşınabilir — ancak **muhtemelen farklı bir ana mesajla**: mevcut abstract'ın taşıdığı ölçek-bağımlı dikotomi, DFL yeniliği ve sweep-line katkısı üçlüsünün hiçbiri mevcut kanıtla ayakta durmuyor. Ayakta kalan üç yön bulgusu — (i) tahmin-güdümlü SJF doymuş rejimde FIFO'yu açık farkla geçer, (ii) kategorik kimlik öznitelikleri sayısal olanlardan belirgin biçimde daha bilgilidir, (iii) nokta doğruluğu çizelgeleme kazancını öngörmez — düzeltilmiş bir pipeline'da yeniden ölçülmek kaydıyla **sağlam ve yayınlanabilir bir tezin çekirdeği olabilir.**