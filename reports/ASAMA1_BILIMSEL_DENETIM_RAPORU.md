# AŞAMA 1 — Bağımsız Bilimsel/İstatistiksel/Kod Denetimi

**Tarih:** 2026-09-02
**Kapsam:** `alibaba-gpu-runtime-prediction-and-scheduling` deposunun tamamı — kaynak kod, 14 Jupyter defteri, veri, checkpoint/model artefaktları, tez LaTeX'i, README ve yardımcı belgeler.
**Yöntem:** 10 bağımsız denetçi ajan (her biri farklı bir boyuma odaklandı), ardından her bulgu için çekişmeli doğrulama (çürütme + yeniden üretim + kalibrasyon mercekleri). Depo salt okunur tutuldu; hiçbir dosya değiştirilmedi.

## ÖNEMLİ METODOLOJİK NOT — Bu raporun kapsamı ve sınırı

Bu denetim, kullanım kredisi tükenmesi yüzünden **planlanan çekişmeli doğrulama sürecinin tamamını bitiremedi**. Durum şeffaf olarak şöyle:

- **10/10 bulgu-arama boyutu tamamlandı.** Toplam **123 ham bulgu** ve **116 "sağlam" (doğru çalışan) nokta** tespit edildi.
- Bu 123 bulgunun **53'ü, planlanan çekişmeli doğrulama sürecinin tamamından geçti** (severity'ye göre 1-3 bağımsız doğrulayıcı mercek: çürütme denemesi, bağımsız yeniden üretim, kalibrasyon). **Sonuç: 53/53 ONAYLANDI, 0 çürütüldü.** Bu 53 bulgu arasında 3 CRITICAL ve 15 MAJOR var; ikisi de tam doğrulamadan geçti.
- **70 bulgu hiç ikinci-göz çekişmeli doğrulamadan geçmedi** (kredi tükendi). Bunlar yalnızca kendilerini bulan tek denetçinin kanıtına dayanıyor — o denetçi kendi iddiasını "VERIFIED" (bizzat okudu/çalıştırdı) ya da "NOT_VERIFIED" olarak etiketledi, ama **bağımsız bir ikinci ajan onu çürütmeye çalışmadı**. Bunlar raporda ayrı bir bölümde (K/L), "tek kaynaklı, ikinci-göz beklemede" etiketiyle listeleniyor. Birçoğu, tam doğrulamadan geçen 53 bulgudan biriyle **aynı olguyu bağımsız olarak** doğruluyor (çapraz teyit) — bunlar özellikle belirtildi.
- **Eksiklik eleştirmeni, nihai birleştirme, 3-yargıç paneli ve otomatik rapor sentezi çalışmadı.** Bu raporun **A. Executive Verdict** ve **P. Güven Skorları** bölümleri, ana oturum tarafından (üç ayrı denetim perspektifi benimsenerek) elle sentezlendi — bunu açıkça işaretliyorum; bir ajan-paneli oy birliği değil.
- Bir workflow çalıştırması sırasında (script hatası nedeniyle) bazı bulgular geçici olarak yanlış "çürütülmüş" sayıldı (eksik oy toplandığında otomatik red). Bu hata düzeltildi ve **gerçek toplanan oylara göre yeniden hesaplandı**; bu rapordaki 53/0 rakamı düzeltilmiş, doğru hesaptır.

**Sonuç:** Bu rapor **kanıt açısından eksiksiz değil** ama elindeki 53 tam-doğrulanmış bulgu (3 CRITICAL + 15 MAJOR dahil, sıfır çürütme) tek başına aşağıdaki hükmü desteklemeye yeterlidir. Kalan 70 bulgu ek risk taşıyor ama hükme dahil edilmedi; M ve L bölümlerinde "doğrulanmalı" olarak listelendi.

---

## A. EXECUTIVE VERDICT

# 🔴 RED — Mevcut sonuçlara (tez metni + üretilen artefaktlar birlikte) şu anki haliyle güvenmek mümkün değil

**Not:** Bu hüküm otomatik bir 3-yargıç panelinden değil, ana oturumun üç ayrı denetim perspektifinden (PhD komite üyesi / hakem+istatistikçi / ML araştırmacısı+yazılım denetçisi) elle sentezlenmesinden geliyor. Aşağıda üçü ayrı ayrı.

**PhD komite üyesi perspektifi:** Tezin savunulmuş halinin ana sonuç tabloları (Tablo 6.1–6.4, Ek A) hiçbir mevcut ya da önceden doğrulanmış kod koşusuyla eşleşmiyor (F-44, F-45 — CRITICAL, tam doğrulanmış); kaynakları Mayıs ve 17 Ağustos tarihli, artık geçersiz sayılan koşulardır. Tezin ana anlatı ekseni ("küçük kümede en doğru model en iyi JCT'yi verir, büyük kümede bu ilişki bozulur") güncel veriyle **tersine** dönüyor: her iki ölçekte de aynı model (kategorik LSTM) kazanıyor, fark 0,2–0,7 puan. Bu, savunma sırasında jüriye sunulan temel bulgunun artık geçerli olmadığı anlamına geliyor. Metin güncellenmeden hiçbir sayı arşive/yayına gönderilmemeli.

**Hakem + istatistikçi perspektifi:** Simülasyon deneyinin kendisi metodolojik olarak sorunlu — bu bir "tez metni eski" sorunu değil, **deney tasarımı** sorunu. Sunulan yük ρ, test aralığında 32-GPU kümede ~166, 256-GPU kümede ~21 (F-33, CRITICAL, 5 bağımsız doğrulayıcıyla tam doğrulandı); yani kuyruk hiçbir konfigürasyonda kararlı değil, "LOAD_FACTOR duyarlılık analizi" aslında hiçbir şey ölçmüyor çünkü sistem her ayarda zaten aşırı doygun. Wilcoxon testi 16.437 işi bağımsız gözlem sayıyor ama bunlar tek bir deterministik simülasyon replay'inin birbirine bağımlı çıktıları (F-23, MAJOR, pseudo-replication) — p-değerleri anlamsız derecede küçük çıkıyor. Tek yönlü test p=1.000 sonucu tezde "FIFO'dan ayırt edilemez" diye yanlış yorumlanmış; doğrusu "FIFO'dan anlamlı derecede kötü" (F-24, MAJOR). Rank-korelasyon şekli n=18 bağımlı nokta üzerinden CI'siz Pearson r kullanıyor (F-26, MAJOR). Bunların hiçbiri LaTeX güncellemesiyle çözülmez; deneyin bir kısmının yeniden tasarlanması gerekiyor.

**ML araştırmacısı + yazılım denetçisi perspektifi:** Kod altyapısı (sızıntı önleme, kronolojik bölme, checkpoint provenance, determinizm) genel olarak **sağlam** — 116 madde bağımsız olarak "doğru çalışıyor" diye doğrulandı, sıfır veri sızıntısı bulundu. Ama iki gerçek kod hatası var: (1) LightGBM hiperparametreleri L1 (MAE) kaybıyla seçiliyor, final model L2 (varsayılan) kaybıyla eğitiliyor — hiperparametreler ait olmadıkları bir kayıp fonksiyonuna göre seçilmiş (F-09/F-93, MAJOR/CRITICAL, çapraz teyitli). (2) 3-tohum ortalaması tabloya yazılıyor ama diske kaydedilen ve simülasyonda kullanılan ağ yalnızca tohum-42'nin ağı; ikisi farklı sıralama veriyor (F-16, MAJOR). Daha da önemlisi: **öğrenilen 17 modelin çoğu, tek satırlık bir "train medyanı" sabit tahmincisini istatistiksel olarak geçemiyor** (F-15/F-25, MAJOR, %95 CI ile doğrulandı); yalnız LGBM-Native marjinal olarak geçiyor, kullanıcı-medyanı (öğrenilmemiş) ise HEPSİNİ MAE'de geçiyor. Bu, tezin "ağaç modelleri runtime'ı iyi tahmin eder" (RQ2) iddiasını doğrudan sorguluyor.

**Sonuç:** İki ayrı sorun katmanı var ve ikisi de RED'i gerektiriyor, ama farklı çözüm yolları var:
1. **Bilinen/beklenen katman — tez metni güncel kodla senkron değil.** Bu, kullanıcının zaten bildiği ve düzeltme sürecinde olduğu bir durum; kod tarafı yeniden eğitilip yeniden çalıştırıldıktan sonra otomatik/yarı-otomatik bir LaTeX güncellemesiyle kapatılabilir.
2. **Yeni/beklenmeyen katman — deney tasarımı ve kodun kendisinde CRITICAL/MAJOR sorunlar.** Simülasyon rejiminin anlamsızlığı (F-33/F-35), istatistik metodolojisi (F-23/F-24/F-26/F-27), LightGBM kayıp fonksiyonu uyuşmazlığı (F-09/F-93), taban çizgisi karşılaştırması (F-15/F-25) — bunlar **yeniden LaTeX yazmakla değil, deney/kod düzeltmesi + yeniden koşuyla** çözülür.

**Aşama 2'ye (LaTeX/yazım) geçilebilir mi? HAYIR.** Önce M bölümündeki CRITICAL ve MAJOR maddeler (özellikle #2 katmanındakiler) kapatılmalı.

---

## B. PROJE PIPELINE

```
[0] HAM VERİ
    data/alibaba_cluster_trace/pai_job_no_estimate_100K.csv (100.000 satır × 8 kolon)
    → src/data_loading.py:165 load_main_sample

[1] FİLTRE  (src/feature_engineering.py:59 build_job_table_from_sample, L102-105)
    (duration>0) & (num_gpu>0) → 82.184 satır, 17.816 elenen (VERIFIED: hepsi gpu_type=CPU/num_gpu=0)
    Bu adım her defterde (00-05) ham CSV'den bağımsız olarak yeniden koşuluyor.

[2] ÖZNİTELİK MÜHENDİSLİĞİ  (feature_engineering.py:150-292)
    hour_of_day/day_of_week (göreli epoch'tan — bkz. F-01), user/gpu_type kategorik,
    sweep-line cluster_load_cpu/gpu + active_job_count (TÜM veri üzerinde, split'ten önce; sızıntı yok ama
    CPU-only işler filtrelendikten SONRA hesaplanıyor → F-04).
    nb00 çıktı CSV'sini (100k_job_with_utilization_full.csv) hiçbir aşağı-akış defteri okumuyor.

[3] TRAIN/TEST  (feature_engineering.py:369 prepare_features_for_model)
    Kronolojik mergesort + shuffle=False → ilk 65.747 train / son 16.437 test.
    Guard aktif (L501-512); sızıntı YOK (VERIFIED, çok kez bağımsız doğrulandı).
    Ayrı final hold-out yok; test seti aynı zamanda simülasyon evrenidir.

[4] MODEL EĞİTİMİ / TUNING  (nb04, 65 kod hücresi)
    Deney A-F: 7 ağaç + 12 DL + XGB-Native + Per-User Median = 21 konfigürasyon.
    RandomizedSearchCV/GridSearchCV (TimeSeriesSplit(3)) → finalize_ml_model / finalize_dl_model.
    DL: 3 tohum [42,1337,2024], ortalama±std raporlanır, diske seed-42 modeli yazılır (F-16).
    checkpoint-varsa-eğitme mantığı: mevcut koşuda 21/21 checkpoint diskten yüklendi, hiçbir model
    bu oturumda eğitilmedi (kullanıcının kendi paralel koşusu ayrı sürüyor).

[5] CHECKPOINT/ARTEFAKT  → results/checkpoints/*.json (21), results/models/* (25 dosya)

[6] DEĞERLENDİRME/ŞEKİL (nb04)  → Tablo A/B/C-F/7/8, Fig1-5 — hepsi load_all_checkpoints()'ten

[7] SİMÜLASYON  (nb05_32/256, 23 kod hücresi)
    8 model + lookup + 4 scaler + 12 .pth yüklenir; sim_jobs = test seti (16.437 iş, LOAD_FACTOR=0.1
    ile sıkıştırılmış varış); 23 politika (FIFO/SRF/Oracle + 20 SJF-Pred) × opsiyonel EASY backfill.
    DL tahminleri burada KIRPILMIYOR (nb04'ten farklı davranış, F-97).
    Wilcoxon + Holm-Bonferroni + bootstrap CI + rank-korelasyon (Spearman/Kendall/Pearson).
    LOAD×backfill duyarlılık ızgarası (cell 30, 56 ek koşu) — ρ hiçbir noktada kararlı rejime düşmüyor.

[8] TEZ  → thesis/latex/chapters/*.tex — TÜM sayısal tablolar elle yazılmış, otomatik üretim YOK
    (F-52). Kaynakları [4]-[7]'nin GÜNCEL çıktıları değil, Mayıs/17-Ağustos tarihli eski koşular (F-44,F-45).
```

**Tekrarlanan gözlem (10/10 boyutta çıktı):** Kod altyapısı → checkpoint/model artefaktları arasındaki zincir sağlam ve izlenebilir (birebir yeniden üretildi). Kırılma **[7]→[8]** geçişinde: hiçbir otomatik köprü (checkpoint/CSV → LaTeX tablo) yok, tüm sayılar elle taşınmış ve bayat.

---

## C. CRITICAL FINDINGS (tam çekişmeli doğrulamadan geçti — 3/3 onaylandı)

### C1. [F-33] Simülasyon rejimi "flash crowd" değil; sunulan yük ρ_GPU ≈ 166 (32-GPU) / 21 (256-GPU) — LOAD_FACTOR duyarlılık ızgarası fiilen etkisiz
- **Dosya:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` cell 20/30; `thesis/latex/chapters/2.background.tex:122`, `5.simulation_framework.tex:135`, `6.results_and_discussion.tex:117`
- **Kanıt:** Test dilimindeki iş yükü Σ(gpu_demand·runtime)=7,27×10⁷ GPU-s; 32 GPU ile bunun hesap-alt-sınırı 26,3 gün, ama tüm varışlar LOAD_FACTOR=0,1 ile 13.710 saniyeye (0,16 gün) sıkıştırılıyor. FIFO makespan 3,26×10⁶ s (37,8 gün); işlerin **%98,2'si son varıştan SONRA** başlıyor. Sıkıştırılmamış izde bile (LF=1.0) ρ=16,6 (32-GPU). Duyarlılık ızgarasının 4 LOAD_FACTOR noktasının hiçbiri kararlı rejime (ρ<1) düşmüyor; bu yüzden JCT kazançları LF boyunca neredeyse sabit kalıyor.
- **Bilimsel etki:** RQ1'in "gerçekçi küme koşullarında" yanıtlandığı iddiası desteklenmiyor. Ölçülen şey bir kuyruk-teorisi deneyi değil, klasik SPT-vs-keyfi-sıra toplu-iş sıralama deneyi; büyüklüğü iş yükünün varyasyon katsayısından geliyor, küme/varış dinamiğinden değil.
- **Seviye:** B_statistical + C_scientific (A_computational temiz — kod yazıldığı gibi çalışıyor).
- **Doğrulama:** 5/5 bağımsız doğrulayıcı CONFIRMED (3× CRITICAL, 2× MAJOR öneri — final CRITICAL).
- **Düzeltme:** ρ'yu tezde hesaplayıp raporla; küme boyutu/LF'yi ρ∈{0,5–1,2} verecek şekilde yeniden seç (32-GPU için LF≈15-30, 256-GPU için LF≈2-4) ve kararlı rejimde tekrar koş; warm-up/drain politikası tanımla; "flash crowd" terimini düzelt.

### C2. [F-44] Tahmin sonuç tabloları (Tablo 6.1, Ek A) depodaki hiçbir checkpoint ile uyuşmuyor — kaynağı 12 Mayıs tarihli eski HTML rapor
- **Dosya:** `thesis/latex/chapters/6.results_and_discussion.tex:22-44`; `appendices.tex:53-156`
- **Kanıt:** Tezdeki 19 satırın tamamı `reports/html/en/04_runtime_prediction_models.html` (mtime 12 Mayıs) içinde birebir bulunuyor. Güncel checkpoint'ler tamamen farklı: en iyi öğrenilen model artık **LGBM-Native (MAE 5.697, R²=0,243)**, tezdeki "XGB-OH R²=0,51" değil (güncel XGB-OH: MAE 6.642, R²=0,156). Per-User Median (MAE 5.191) ve XGB-Native (MAE 6.746) tezde hiç yok.
- **Bilimsel etki:** Tezin "XGB-OH en iyi, ağaçlar DL'den kesin üstün" ana bulgusu tersine dönüyor: LSTM-D (3-tohum ort. MAE 6.236) XGB-OH'den düşük; Per-User Median tüm öğrenilen modellerden düşük MAE veriyor. RQ2/RQ4 yanıtları desteklenmiyor.
- **Seviye:** C_scientific + B_statistical.
- **Doğrulama:** 3/3 CONFIRMED (hepsi CRITICAL).
- **Düzeltme:** Tabloları `results/checkpoints/*.json`'dan otomatik üreten bir betik yaz; UserMedian/XGB-Native satırlarını ekle; ch6 anlatısını yeni sıralamaya göre yeniden yaz.

### C3. [F-45] Simülasyon tabloları ve tüm türetilmiş yüzdeler GPU kapasitesi hiç uygulanmayan eski simülatör koşusundan (17 Ağustos)
- **Dosya:** `thesis/latex/chapters/6.results_and_discussion.tex:131-175,184-186,240-368,400-404`; `1.introduction.tex:98`; `7.conclusions:20-22`
- **Kanıt:** `tests/test_regression_guards.py:61-67`'nin kendi docstring'i, eski simülatörün GPU kısıtını hiç uygulamadığını belgeliyor ("her sonucun sonsuz GPU kapasitesi varsayımıyla üretildiği"). Tezdeki değerler yalnız `reports/html/en/05_scheduler_evaluation_{32,256}_gpu.html` (17 Ağu) içinde var. Güncel nb05: 32-GPU'da LSTM-Cat (%60,28) XGB-Cat'i (%59,61) geçiyor; 256-GPU'da da LSTM-Cat (%63,08) birinci — tezin "küçük kümede XGB, büyükte LSTM" dikotomisi **artık yok**, iki ölçekte de aynı model kazanıyor.
- **Bilimsel etki:** Başlığa taşınan %57,25 sayısı ve "sequence trap −45,41%" bulgusu (ch7) bozuk simülatörden geliyor; güncelde sequence-trap yönü bile TERSİNE dönüyor (+22,87%).
- **Seviye:** C_scientific + A_computational.
- **Doğrulama:** 3/3 CONFIRMED (hepsi CRITICAL).
- **Düzeltme:** nb05 DataFrame'lerinden CSV → LaTeX köprüsü kur; tüm ch1/ch6/ch7/abstract yüzdelerini yeniden hesapla.

---

## D. MAJOR FINDINGS (tam çekişmeli doğrulamadan geçti — 15/15 onaylandı)

| # | Bulgu | Dosya | Etki |
|---|---|---|---|
| F-01 | `day_of_week`/`hour_of_day` göreli epoch'tan (1970-01-01); test seti tek bir "iz-günü"ne düşüyor | `feature_engineering.py:176-177` | Tezdeki "Perşembe zirvesi" yorumu epoch artefaktı |
| F-09/F-93 | LightGBM: tuning `regression_l1` (MAE), final refit varsayılan L2 ile | `tuning.py:545-548,770-782` | En iyi model (LGBM-Native) yanlış kayıpla eğitilmiş olabilir |
| F-15/F-25 | Öğrenilen modeller train-medyanı sabitini geçemiyor; kullanıcı-medyanı hepsini geçiyor | checkpoint'ler + `exp_b_user_median.json` | RQ2 desteklenmiyor |
| F-16 | 3-tohum ortalaması ≠ diske yazılan (simülasyonda kullanılan) seed-42 ağı; sıralama değişiyor | `tuning.py:1560-1596` | nb04 ve nb05 farklı model kalitesi kullanıyor |
| F-23 | Wilcoxon/bootstrap 16.437 bağımlı JCT'yi bağımsız gözlem sayıyor (pseudo-replication) | nb05 cell 46 | p-değerleri anlamsız derecede küçük |
| F-24 | Tek yönlü Wilcoxon p=1,000 → "FIFO'dan ayırt edilemez" diye YANLIŞ yorumlanmış (doğrusu: FIFO'dan kötü) | ch6:245 | Sonuç bölümü yön hatası |
| F-26 | Rank-korelasyon: n=18 bağımlı nokta, CI'siz Pearson r; en yüksek-ρ model (UserMedian) şekle alınmamış | nb05 cell 29 | "MAE≠JCT" iddiası zayıf temelli |
| F-27 | DL: n=3 tohum, t-CI tüm modelleri kapsıyor; ağaç modelleri tek koşu (asimetrik belirsizlik) | `tuning.py` + nb04 cell 86 | Sıralama iddiaları tohum gürültüsünde |
| F-34 | Katı head-of-line bloklama TÜM politikalara uygulanıyor; tez yalnız FIFO için tanımlıyor | `multi_node_simulator.py:430-501` | Kod↔tez algoritma tanımı uyuşmuyor |
| F-35 | Küme ölçeği (32/256 GPU) iş yüküyle orantısız (iz ima ettiği eşzamanlılık ~530 GPU) | `multi_node_simulator.py:545-586` | "small vs large" aslında "aşırı doygun vs çok aşırı doygun" |
| F-39/F-47 | `tab:sim_time` hiçbir ölçüm koduna dayanmıyor; "<17s/politika" iddiası kod tarafından üretilmemiş | `5.simulation_framework.tex:254-284` | Doğrulanamaz performans tablosu |
| F-46 | `tab:hyperparams`'ın 19 satırının hiçbiri güncel `best_params` ile eşleşmiyor | `4.prediction_models.tex:184-207` | Yeniden üretilebilirlik iddiası boş |
| F-48 | "6 model/18 tahminci/4 politika" ↔ kod: 20 tahmin seti, 23 politika, 56 duyarlılık koşusu | çoklu tez bölümü | Deney sayımı tutarsız; en güçlü taban çizgisi (UserMedian) tezde yok |

*(Tam kanıt/etki/düzeltme metni her F-ID için `reports/` dizinindeki ham denetim verisinde mevcuttur; yukarıdaki tablo özet amaçlıdır.)*

---

## E. MODERATE / MINOR / INFORMATIONAL FINDINGS (tam doğrulamadan geçti)

**MODERATE (23, hepsi onaylandı):** F-02 (arrival_sec ekstrapolasyon dışı), F-03 (RF/XGB/LGBM eğitim verisi asimetrisi), F-04 (sweep-line CPU-only filtre sonrası), F-05 (bağımsız birim belirsizliği, burst'ler), F-10 (XGB erken durdurma ölçütü uyuşmazlığı), F-11 (n_estimators fiilen kullanılmıyor), F-12 (EarlyStopping delta çok büyük), F-13 (narrow grid dropout'u atıyor), F-14 (21 konfig aynı test setinde seçiliyor — winner's curse), F-17/F-18 (kayıp/bütçe tutarsızlığı), F-19 (grid yakınsamamış), F-20 (ch4 tuning anlatısı kodla uyuşmuyor), F-28 (tahmin CI yok), F-29 (MAPE ölçek çelişkisi), F-30/F-41 (bounded slowdown yok), F-32 (p-değeri tek başına), F-36 (eş-zamanlı varış sırası keyfi), F-37 (release() footprint eşleştirmesi), F-38 (sığmayan iş sessizce düşüyor), F-40 (kesirli GPU toplamsal, tez "izin yok" diyor), F-42 (teşhis çıktıları okunmuyor), F-50 (ch3 GPU-talep sayıları eski), F-51 (ch7/ch6 CI ifadeleri kodun tersi), F-52 (tüm tablolar elle, transkripsiyon hataları), F-53 (README/context 3 farklı eski koşu).

**MINOR (8, hepsi onaylandı):** F-06 (kategori sözlüğü tüm veriden — sızıntı değil ama not edilmeli), F-07 (sweep-line çevrimiçi hesaplanabilir, iyi haber), F-21 (ölü yapılandırma yolları), F-22 (ortam sürümleri tezde yok), F-31 (istatistik kodu test edilmemiş), F-43 (simülatör test kapsamı eksik), F-49 (nb05 şekilleri eski — kullanıcının koşusu bitince kendiliğinden düzelir).

**INFORMATIONAL (1):** F-08 (ayrı final hold-out yok, test seti her yerde tekrar kullanılıyor — 21 model + simülasyon + şekiller aynı 16.437 işte).

---

## F. RESULT VALIDATION

| Sonuç | Koddan doğrulandı mı? | İstatistiksel doğru mu? | Robust mu? | Bias riski | Güven |
|---|---|---|---|---|---|
| Kronolojik train/test bölmesi sızıntısız | ✅ VERIFIED (çoklu bağımsız yeniden üretim) | ✅ | Yüksek | Düşük | **HIGH** |
| Checkpoint metrikleri ↔ diskteki modeller | ✅ VERIFIED (bit-birebir) | ✅ | Yüksek | Düşük | **HIGH** |
| "En iyi öğrenilen model LGBM-Native" | ✅ VERIFIED | ⚠️ Sabit medyanı marjinal geçiyor (F-15) | Düşük | Orta | **LOW-MEDIUM** |
| "Ağaç modelleri runtime'ı iyi tahmin eder" (RQ2) | ✅ (kod) | ❌ Çoğu model sabit medyanı geçemiyor | **Fragile** | Yüksek | **LOW** |
| "Kategorik LSTM en iyi zamanlama politikası" | ✅ VERIFIED (güncel nb05) | ⚠️ Wilcoxon pseudo-replication (F-23) | Orta (yön iki ölçekte tutarlı) | Orta | **MEDIUM** |
| "Küçük kümede X, büyük kümede Y" dikotomisi | ❌ Tez ile güncel veri ÇELİŞİYOR | — | **Unsupported** (artık yok) | Yüksek | **VERY LOW** |
| JCT iyileşme yüzdeleri (%57-81) | ✅ (güncel kod) ama ❌ tez tabloları eski | ⚠️ ρ≫1 rejiminde anlamı sınırlı (F-33) | **Fragile** (rejim anlamsız) | Yüksek | **LOW** |
| Wilcoxon "p<0,001 anlamlı" iddiaları | ✅ (kod doğru hesaplıyor) | ❌ Pseudo-replication (F-23) | Düşük | Yüksek | **LOW** |
| Sızıntı yok (genel) | ✅ VERIFIED (10/10 boyutta çapraz teyit) | ✅ | Yüksek | Düşük | **HIGH** |

### "Tezde X → Kodda Y" izlenebilirlik tablosu (örnek, tam liste ~30 madde)

| Tez iddiası | Konum | Kodun/checkpoint'in söylediği | Kaynak |
|---|---|---|---|
| "XGB-OH R²=0,51, en doğru model" | ch6:22 | LGBM-Native R²=0,243 en iyi; XGB-OH R²=0,156 | F-44 |
| "GPU talebi ortalama 0,52" | ch3:62 | Ortalama 0,68 (kesirli GPU koruma düzeltmesi sonrası) | F-50 |
| "Bootstrap CI hesaplanmadı" | ch7:72 | nb05 cell 46'da hesaplanıyor (999 örnek) | F-51 |
| "p=1,000 → FIFO'dan ayırt edilemez" | ch6:245 | Tek yönlü test; doğrusu "FIFO'dan anlamlı kötü" | F-24 |
| "Simülasyon <17 s/politika" | ch1:95 | Hiçbir ölçüm kodu yok; ölçülen 8-21 s | F-39/F-47 |
| "6 model / 18 tahminci / 4 politika" | ch5:161,209 | 20 tahmin seti, 23 politika (kod) | F-48 |
| "Küçük kümede ağaçlar, büyükte LSTM üstün" | abstract, ch1:98 | Her iki ölçekte de LSTM-Cat kazanıyor | F-45 |
| "Sequence trap −45,41%" | ch7:22 | Güncel: +22,87% (yön tersine döndü) | F-45 |
| "MDI-based" feature importance | ch6 fig altyazı | LightGBM split-sayısı, XGBoost gain (farklı ölçütler) | F-108 (tek kaynak) |

---

## G. REPRODUCIBILITY AUDIT

**Olumlu (VERIFIED):**
- NB00 işlenmiş CSV bit-birebir yeniden üretilebilir.
- Checkpoint'ler ve diskteki model dosyaları yapısal olarak tutarlı; provenance zinciri sağlam.
- DL eğitimi tohumla (aynı cihazda) bit-birebir deterministik.
- Defterler baştan sona sıralı çalıştırılmış (Run All kanıtı); kök dizin bulucu sağlam.

**Sorunlu:**
- **F-58 (MAJOR, tek kaynaklı):** `requirements-lock.txt` sonuçları üreten ortamı değil, başka bir ortamı dondurmuş.
- **F-60 (MAJOR, tek kaynaklı):** Checkpoint timestamp'ı her defter geçişinde yeniden yazılıyor, gerçek eğitim zamanına bağlı değil — hangi koşunun hangi tarihte olduğu yalnız değer eşleştirmesiyle bulunabiliyor.
- **F-61 (MAJOR, tek kaynaklı, DOĞRULANMALI):** DL sonuçlarının donanıma (MPS vs CPU) bağlı olabileceği iddia ediliyor; `seed_everything` deterministik algoritma zorlamıyor.
- **F-22 (MINOR):** Ortam sürümleri (Python, sklearn, torch) tezde belgelenmemiş.
- **F-52:** Hiçbir LaTeX tablosu otomatik üretilmiyor — köprü eksik, elle taşıma transkripsiyon hatalarına açık.

---

## H. STATISTICAL AUDIT

| Yöntem | Uygulama doğru mu? | Sorun |
|---|---|---|
| Wilcoxon signed-rank (job_id eşleştirmesi) | ✅ Kod doğru | ❌ Bağımsızlık varsayımı ihlal ediliyor (F-23, pseudo-replication) |
| Holm-Bonferroni | ✅ Referans uygulamayla bit-birebir eşleşiyor (VERIFIED) | — |
| Rank-biserial etki büyüklüğü | ✅ Formül doğru | — |
| Bootstrap CI (999 örnek, percentile) | ✅ Uygulama doğru | ❌ iid varsayımı aynı pseudo-replication sorununu taşıyor |
| Tek yönlü test yorumu | — | ❌ p=1,000 sonucu tezde yanlış yönde okunmuş (F-24) |
| Pearson r (rank-korelasyon şekli) | ✅ Hesap doğru | ❌ n=18, bağımlı gözlemler, CI yok (F-26) |
| DL çoklu-tohum ± SD | ✅ Hesap doğru (ddof=1) | ⚠️ n=3 çok küçük, CI tüm modelleri kapsıyor (F-27) |
| Taban çizgisi karşılaştırması | ✅ CI ile doğru yapılmış (denetçi tarafından, tez tarafından DEĞİL) | ❌ Tezde hiç yok (F-25) |

**Genel:** İstatistiksel *hesaplama* kodu doğru (Holm, bootstrap, rank-biserial hepsi bağımsız referans uygulamayla teyit edildi). Sorun *tasarım* ve *yorumlama* katmanında: yanlış bağımsızlık varsayımı, yanlış yön yorumu, eksik taban çizgisi, küçük-n korelasyon.

---

## I. DATA LEAKAGE / BIAS AUDIT

**Sızıntı: BULUNMADI (yüksek güvenle).** 10 boyutun tamamı bağımsız olarak şunu doğruladı:
- Kronolojik bölme, guard'lı, sınırda örtüşme yok.
- OneHotEncoder/MinMaxScaler yalnız train'e fit.
- Sweep-line özniteliklerinde gelecek/hedef bilgisi yok (çevrimiçi hesaplanabilir).
- Hiperparametre seçimi test setine bakmıyor (TimeSeriesSplit, val-only DL seçimi).
- Per-User Median baseline sızıntısız (yalnız train'den).

**Bias/confounding riskleri (sızıntı değil ama sonucu etkileyebilir):**
- **F-04:** Küme yükü öznitelikleri CPU-only işler filtrelendikten sonra hesaplanıyor → sistematik eksik tahmin (~%15-22).
- **F-05:** Bağımsız gözlem birimi belirsiz; işlerin %24'ü aynı kullanıcı-saniye "burst" grupları — bu pseudo-replication'ın kaynağı.
- **F-90 (tek kaynaklı):** Model başarımı büyük ölçüde kullanıcı kimliğine (user lookup) bağlı; görülmemiş kullanıcılarda performans düşüyor olabilir.
- **F-01:** Zamansal özellikler epoch artefaktı taşıyor; "hafta içi/sonu" yorumu geçersiz.

---

## J. ROBUSTNESS AUDIT

| Sonuç | Etiket | Gerekçe |
|---|---|---|
| Sızıntısız kronolojik bölme | **ROBUST** | 10 boyutta bağımsız çapraz teyit, çürütülemedi |
| "ML-SJF > FIFO" yönü | **MODERATELY ROBUST** | 8 duyarlılık noktasında ve iki ölçekte korunuyor, ama mutlak büyüklük ρ rejimine bağlı |
| "En iyi tahminci = en iyi zamanlama" (tez ana tezi) | **FRAGILE / kısmen UNSUPPORTED** | Kaynak simülatör hatalıydı (F-45); güncel veride dikotomi kayboldu |
| "Ağaçlar runtime'ı iyi tahmin eder" (RQ2) | **UNSUPPORTED** | Sabit medyan taban çizgisini çoğu model geçemiyor (F-15) |
| JCT iyileşme yüzdelerinin mutlak büyüklüğü | **FRAGILE** | Aşırı doygun rejim (ρ≫1), duyarlılık ızgarası rejim değiştirmiyor (F-33) |
| Wilcoxon anlamlılık iddiaları | **UNSUPPORTED** (istatistiksel biçimde) | Pseudo-replication (F-23) |
| Simülatör hesaplama doğruluğu (yerleştirme, kaynak muhasebesi) | **ROBUST** | Bağımsız betiklerle bit-birebir yeniden üretildi |

---

## K. CLAIM AUDIT

| Claim | Evidence | Validity | Limitation |
|---|---|---|---|
| "Küçük kümede en doğru model en iyi JCT verir, büyükte bu bozulur" | Eski (17 Ağu) simülatör çıktısı | **DESTEKLENMİYOR** — güncel veri tersini gösteriyor | Kod düzeltildi, tez güncellenmedi (F-45) |
| "XGBoost one-hot en doğru model (R²=0,51)" | Eski (Mayıs) checkpoint | **DESTEKLENMİYOR** — güncel en iyisi LGBM-Native R²=0,24 | F-44 |
| "DL modelleri ağaçlardan kesin daha kötü tahmin eder" | Eski tablo | **KISMEN DESTEKLENMİYOR** — LSTM-D (3-tohum ort.) XGB-OH'yi geçiyor | F-44, F-16 |
| "Sıralama doğruluğu (MAE değil) zamanlama başarısını belirler" | nb05 rank-korelasyon + UserMedian JCT sonucu | **KISMEN DESTEKLENİYOR** ama kanıt zayıf (n=18, CI yok; F-26) | Tezin en güçlü potansiyel katkısı ama düzgün sunulmamış |
| "Veri sızıntısı yok, kronolojik bütünlük sağlandı" | 10 boyutta bağımsız doğrulama | **DESTEKLENİYOR** (yüksek güven) | — |
| "Simülasyon gerçekçi küme koşullarını modelliyor" | ρ hesabı | **DESTEKLENMİYOR** — aşırı doygun, dengesiz rejim | F-33, F-35 |
| "İstatistiksel olarak anlamlı JCT farkları" | Wilcoxon p<0,001 | **YANILTICI** — pseudo-replication, gerçek belirsizlik bilinmiyor | F-23 |

---

## L. REPRODUCIBILITY CHECKLIST

- [ ] `requirements-lock.txt` gerçekten sonuçları üreten ortamı yansıtacak şekilde güncellenmeli (F-58, doğrulanmalı)
- [ ] Ortam bilgisi (Python/sklearn/torch sürümü, donanım: MPS vs CPU) tez metnine eklenmeli (F-22)
- [ ] Checkpoint timestamp'ları gerçek eğitim zamanını yansıtmalı, ya da eğitim zamanı ayrı loglanmalı (F-60)
- [ ] LightGBM final refit'e `objective="regression_l1"` açıkça geçilmeli (F-09/F-93) — **kod düzeltmesi gerektiriyor**
- [ ] DL: hangi tohumun modelinin tabloya/simülasyona girdiği netleştirilmeli (F-16) — **kod/protokol kararı gerektiriyor**
- [ ] LaTeX tabloları için checkpoint/CSV → LaTeX otomatik üretim köprüsü kurulmalı (F-52)
- [ ] Simülasyon süre ölçümü koda eklenmeli, yoksa tab:sim_time kaldırılmalı (F-39/F-47)
- [ ] MPS/CPU determinizm iddiası test edilmeli (F-61)
- [ ] Yayınlanmış (git HEAD) sürüm ile working-tree değişiklikleri arasındaki fark netleştirilmeli (F-57, tek kaynaklı — DOĞRULANMALI, güncel git durumu ile çelişebilir, dikkatle kontrol edin)

---

## M. REQUIRED FIXES BEFORE THESIS WRITING

### CRITICAL (tam doğrulanmış — LaTeX'e geçmeden ÖNCE çözülmeli)
1. **[F-33]** Simülasyon rejimini ρ-farkındalıklı şekilde yeniden tasarla (küme boyutu / LOAD_FACTOR seçimi); "flash crowd" anlatısını düzelt veya deneyi buna göre çerçevele.
2. **[F-44]** Tahmin tablolarını güncel checkpoint'lerden otomatik üret; "en iyi model" anlatısını LGBM-Native lehine güncelle.
3. **[F-45]** Simülasyon tablolarını güncel nb05 çıktısından otomatik üret; "küçük/büyük küme dikotomisi" anlatısını kaldır veya güncel veriyle yeniden kur.

### MAJOR (tam doğrulanmış — kod/deney düzeltmesi gerektiriyor, sırayla)
4. **[F-09/F-93]** LightGBM final refit'e doğru objective'i geçir, ilgili modelleri yeniden eğit.
5. **[F-15/F-25]** Tüm tablolara sabit-medyan ve kullanıcı-medyanı taban çizgilerini ekle; "iyi tahmin" anlatısını taban çizgisine göre yeniden değerlendir.
6. **[F-16]** DL için tohum-tutarlılık protokolü belirle (ensemble ya da seçilen tohumun tabloya girmesi).
7. **[F-23/F-24]** Wilcoxon'un bağımsızlık varsayımı sorununu ele al (zaman-bloklu bootstrap veya çoklu bağımsız replay); tek yönlü test yorumunu düzelt.
8. **[F-26/F-27]** Rank-korelasyon şekline CI ekle, tüm modelleri dahil et; DL CI'lerini tabloda açıkça göster.
9. **[F-34]** Simülatör davranışını (katı HoL) tez algoritma tanımıyla uyumlu hale getir veya algoritma tanımını koda uydur.
10. **[F-35]** Küme boyutu seçimini iş yüküyle orantılı hale getir veya alt-örnekleme kullan.
11. **[F-39/F-46/F-47/F-48]** Simülasyon süresi, hiperparametre tablosu ve model/politika sayımı tablolarını koddan üret veya kaldır.

### MODERATE/MINOR
12. F-04 (sweep-line CPU-only filtre sırası), F-52 (LaTeX otomasyonu), F-53 (README/context senkronizasyonu), diğer 23 MODERATE ve 8 MINOR madde — bkz. Bölüm E.

### EK DENEYLER (bu denetimde tespit edilen, ayrı planlama gerektirir)
- Simülasyonu kararlı-yük rejiminde (ρ<1) yeniden koş.
- ≥3 bağımsız replay (farklı random tie-break veya bootstrap) ile Wilcoxon/CI'yi yeniden hesapla.
- Ağaç modelleri için de ≥3 `random_state` koşusu (DL ile simetrik belirsizlik raporlaması).
- **70 doğrulanmamış bulguyu** (Bölüm N'deki liste) ikinci-göz çekişmeli doğrulamadan geçir — özellikle 7 tek-kaynaklı CRITICAL aday (F-57, F-68, F-69, F-70, F-82, F-103, F-104).

---

## N. TEK-KAYNAKLI BULGULAR — İKİNCİ-GÖZ DOĞRULAMA BEKLİYOR (70 madde, kredi tükendiği için çekişmeli doğrulamaya alınamadı)

Bunlar bir denetçi ajanı tarafından bulunmuş ve VERIFIED (bizzat okundu/çalıştırıldı) olarak işaretlenmiş, ama **bağımsız bir ikinci ajan bunları çürütmeye çalışmadı**. Çoğu, yukarıdaki tam-doğrulanmış bulgularla aynı olguyu farklı açıdan teyit ediyor (çapraz tutarlılık — parantezde belirtildi).

**CRITICAL adayı (7) — öncelikli doğrulanmalı:**
- F-57: Yayınlanmış (git HEAD) kod tezin sonuçlarını üreten sürüm değil *(dikkat: bu iddia "Is a git repository: false" sistem bilgisiyle çelişebilir — önce git durumu netleştirilmeli)*
- F-68: Ana anlatı tersine döndü (F-45 ile çapraz teyit)
- F-69: Per-User Median tüm modelleri geçiyor ama tezde yok (F-15/F-25 ile çapraz teyit)
- F-70: "Flash crowd" aslında toplu-iş birikimi (F-33 ile çapraz teyit — **bu olgu artık fiilen ÇİFT doğrulanmış sayılabilir**)
- F-82: LGBM L1/L2 uyuşmazlığı (F-09/F-93 ile çapraz teyit — **çift doğrulanmış**)
- F-103/F-104: Tablo ile şekil AYNI belgede birbirini yalanlıyor (kendi içinde tutarlılık kontrolü, yüksek doğal güvenilirlik)

**MAJOR adayı (19):** F-58,59,60,61,71,72,73,74,83,84,85,93,94,105,106,107,108,109,110 — özet: ortam/veri izlenebilirliği, backfill'in tezde eksikliği, "DFL/Learning-to-Rank" iddialarının yöntemle örtüşmemesi, şekil-metin çelişkileri (residual, feature importance, GPU demand dağılımı).

**MODERATE (27), MINOR (16), INFORMATIONAL (1):** Tam liste ve dosya/konum bilgisi bu raporun üretiminde kullanılan ham veride mevcuttur; özet: checkpoint hijyeni, test kapsama boşlukları, şekil/tablo küçük tutarsızlıkları, kod kalitesi.

---

## O. ÇÜRÜTÜLEN ADAY BULGULAR

**Yok.** Çekişmeli doğrulamadan geçirilen 53 bulgunun tamamı onaylandı (0 çürütme). Bu, bulgu kalitesinin yüksek olduğunu gösteriyor ama aynı zamanda doğrulayıcıların "çürütme" merceğinde yeterince agresif olmamış olabileceği ihtimalini de akılda tutmak gerekir — 53/53'lük tam onay oranı istatistiksel olarak dikkat çekicidir ve M bölümündeki ek deneyler bunu bağımsız biçimde test etmelidir.

---

## P. GÜVEN SKORLARI

| Ana sonuç | Evidence strength | Reproducibility | Robustness | Risk of bias |
|---|---|---|---|---|
| Veri sızıntısı yok | **HIGH** | **HIGH** | **HIGH** | **LOW** |
| Checkpoint/model provenance | **HIGH** | **HIGH** | **HIGH** | **LOW** |
| "En iyi tahminci = en iyi JCT" (tez ana tezi) | **LOW** | **LOW** (tez tabloları eski) | **LOW** | **HIGH** |
| "Ağaçlar runtime'ı iyi tahmin eder" (RQ2) | **LOW** | **MEDIUM** (kod tekrarlanabilir) | **LOW** (baseline geçilemiyor) | **MEDIUM** |
| Simülasyon JCT kazanç yüzdeleri (mutlak) | **MEDIUM** (kod doğru hesaplıyor) | **MEDIUM** | **LOW** (ρ rejimi anlamsız) | **HIGH** |
| Wilcoxon anlamlılık iddiaları | **LOW** (pseudo-replication) | **HIGH** (kod deterministik) | **LOW** | **HIGH** |
| Sıralama>doğruluk hipotezi (tezin potansiyel en güçlü katkısı) | **LOW-MEDIUM** (n=18, CI yok) | **MEDIUM** | **MEDIUM** | **MEDIUM** |

*Gerekçe: "Evidence strength" bu denetimde toplanan kanıtın niteliğini; "Reproducibility" kodun/sonucun yeniden üretilebilirliğini; "Robustness" sonucun farklı koşullarda (rejim, tohum, split) ayakta kalıp kalmadığını; "Risk of bias" sistematik yanlılık olasılığını değerlendirir.*

---

*Bu rapor, 10 bağımsız denetçi ajanın tam taraması (123 bulgu, 116 sağlam nokta) ve 53 bulgunun tam çekişmeli doğrulaması (3 mercek: çürütme/yeniden-üretim/kalibrasyon, 0 çürütme) üzerine kurulmuştur. A ve P bölümleri, otomatik yargıç paneli tamamlanamadığı için ana oturum tarafından üç ayrı perspektiften elle sentezlenmiştir. Kod değişikliği bu raporun kapsamında YAPILMAMIŞTIR — Aşama 1 kuralına uygun olarak yalnız denetlendi.*
