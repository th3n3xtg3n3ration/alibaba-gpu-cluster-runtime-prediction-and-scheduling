# Kademe 1 + Kademe 2 (kısmi) — Son Durum Raporu

**Toplam bulgu:** 133 | **Düzeltilen:** 35 | **Kalan:** 98

Bu oturumda kod tarafındaki bulgular sistematik olarak işlendi, her biri izole commit'lendi. LaTeX-only bulgulara (rerun'a bağlı sayılar içerenler) kasıtlı olarak dokunulmadı — bunlar ayrı bir geçişte, gerçek sayılar elde edildikten sonra toplu işlenecek.

## ✅ DÜZELTİLENLER (35)

| ID | Önem | Commit | Ne yapıldı |
|---|---|---|---|
| `robustness-1` | CRITICAL | `6e68e52` | LightGBM final refit artık objective=regression_l1 (arama ile aynı) |
| `modeling-1` | MAJOR | `6e68e52` | aynı (robustness-1 ile aynı kök neden) |
| `code_bugs-1` | MAJOR | `6e68e52` | aynı (robustness-1 ile aynı kök neden) |
| `modeling-7` | MAJOR | `74161fe/cd76cd0` | ProfileMedian + UserMedian taban çizgileri NB04+NB05e eklendi |
| `F-C01` | MAJOR | `74161fe/cd76cd0` | aynı |
| `baselines_claims-2` | CRITICAL | `74161fe/cd76cd0` | aynı |
| `simulator-3` | MAJOR | `2861d93` | ~752 GPU kararlı-rejim (rho≈0,7) referans noktası eklendi |
| `robustness-3` | MAJOR | `2861d93` | aynı |
| `F-C03` | MODERATE | `2861d93` | aynı (kararlı-rejimde cluster_load artık fiziksel olarak mümkün) |
| `baselines_claims-1` | CRITICAL | `96b8622` | Ölçek-bağımlı dikotomi anlatısı LaTeX+notebook metninden kaldırıldı |
| `code_bugs-4` | MAJOR | `f54eded` | Machine.release() artık job_id ile eşleşiyor (cpu,gpu ayak izi değil) |
| `simulator-5` | MODERATE | `f54eded` | aynı |
| `code_bugs-2` | MAJOR | `88de5b8` | DL EarlyStopping delta artık göreli eşik |
| `modeling-4` | MODERATE | `88de5b8` | aynı |
| `leakage-1` | MAJOR | `4e45f74` | day_of_week artık iz-günü indeksi (mutlak sayaç), takvim günü değil |
| `figures_tables-11` | MAJOR | `9ed504e` | pred-vs-actual şekli rastgele örnekleme + log-log eksen kullanıyor |
| `F-C02` | CRITICAL | `Karar D` | num_inst — kod DEĞİŞMEDİ, tez metnine sınırlama notu olarak eklenecek |
| `reproducibility-4` | MAJOR | `3b552b7` | checkpoint provenance (git/sürüm/veri hash) + zaman damgası koruma eklendi |
| `reproducibility-2` | MAJOR | `5d5be50` | requirements-lock.txt gerçek venvden yeniden üretildi |
| `reproducibility-6` | MAJOR | `91015b6` | betikler artık venv/thesis-venv kerneline sabitlendi |
| `modeling-8` | MAJOR | `6fa9b80` | DL tablosunda artık seed0 (diske yazılan ağ) skoru da gösteriliyor |
| `statistics-5` | MAJOR | `6fa9b80` | aynı + ch4 metin düzeltmesi |
| `simulator-4` | MODERATE | `010a9be` | Eş-zamanlı olaylarda artık deterministik (seq sayaçlı) sıralama |
| `simulator-6` | MODERATE | `010a9be` | Sığmayan iş artık sessizce düşmüyor, ValueError/RuntimeError veriyor |
| `code_bugs-7` | MINOR | `010a9be` | aynı |
| `statistics-8` | MODERATE | `590f153` | bounded_slowdown (tau=10s) sütunu eklendi, unbounded yanında raporlanıyor |
| `simulator-9` | MODERATE | `590f153` | aynı |
| `statistics-7` | MODERATE | `9e49b2c` | evaluation.py docstring artık yanlış "tez ile tutarlı" iddiasını yapmıyor |
| `code_bugs-3` | MODERATE | `b34d9ed` | dropout artık dar-grid/final refite sabit değer olarak taşınıyor |
| `modeling-5` | MODERATE | `b34d9ed` | aynı + num_layers=1 artık filtrelenmiyor |
| `code_bugs-5` | MODERATE | `a4904d9` | Tüm tahminler (NB04+NB05, ağaç+DL) artık >=0a kırpılıyor |
| `code_bugs-9` | MODERATE | `ea81943` | Feature-importance döngüsü artık yazdırıyor; _m değişken gölgelenmesi giderildi |
| `leakage-6` | MINOR | `f599aff` | Native-categorical kategoriler artık yalnız trainden türetiliyor |
| `code_bugs-6` | MODERATE | `ad9c718` | XGB/LGBM final refit artık trainin %100ünü kullanıyor (RF ile eşit) |
| `figures_tables-9` | MODERATE | `9b429b6` | inter-arrival histogramı artık log-spaced bin kullanıyor |

## ⏳ BÜYÜK DENEY GEREKTİRENLER — yapılmadı (27)

Bunlar 'bug fix' değil, gerçek yeni deneyler (ablasyon, çoklu-pencere değerlendirme, yeni baseline eğitimi, istatistik metodolojisi yeniden tasarımı). Zorlanmadı; ayrı, planlı bir aşama (Grup 4) olarak ele alınmalı.

| ID | Önem | Konu | Dosya |
|---|---|---|---|
| `statistics-1` | MAJOR | Wilcoxon p-değerleri ve bootstrap CI'ler tek bir deterministik simülasyon replay'inin 16.4 | `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` |
| `statistics-4` | MAJOR | Rank-korelasyon şekli: n=18 bağımlı nokta üzerinden CI'siz Pearson r; MAE–JCT ilişkisi anl | `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` |
| `simulator-2` | MAJOR | Katı head-of-line bloklama TÜM politikalara uygulanıyor; tez Algoritma 1 ve §5.3.3 yalnız  | `src/simulation/multi_node_simulator.py` |
| `baselines_claims-3` | MAJOR | 'Flash crowd' rejimi aslında kalıcı aşırı yük / toplu (batch) zamanlama: tüm varışlar ilk  | `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb; thesis/latex/chapters/5.simulation_framework.tex` |
| `baselines_claims-4` | MAJOR | FIFO taban çizgisi katı HoL-bloklamalı ve backfill'siz; backfill'li süpürme defterde var a | `src/simulation/multi_node_simulator.py; notebooks/en/05_scheduler_evaluation_32_gpu.ipynb; thesis/latex/chapters/5.simulation_framework.tex` |
| `baselines_claims-6` | MAJOR | 'DFL yaklaşımı öneriyoruz' ve 'Learning-to-Rank motoru' iddiaları yöntemle örtüşmüyor: hiç | `src/tuning.py; thesis/latex/frontmatter/abstract-en.tex; thesis/latex/chapters/1.introduction.tex` |
| `baselines_claims-7` | MAJOR | İstatistiksel anlamlılık pratik anlamlılık yerine kullanılıyor: 16.437 eşleştirilmiş işte  | `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb; thesis/latex/chapters/6.results_and_discussion.tex; results/checkpoints/exp_d_lstm.json` |
| `robustness-2` | MAJOR | 'Tahmin kalitesi (MAE ya da Spearman) → çizelgeleme kazancı' anlatısı karşı-örneklerle çür | `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` |
| `robustness-4` | MAJOR | 'SJF-LSTM (Categorical) en iyi ML politikası / DL büyük ölçekte üstün' (RQ4) sonucu 0,2-0, | `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb` |
| `F-C04` | MAJOR | Tezin 2. katkısı olan sweep-line özniteliği için hiçbir ablasyon deneyi yok: 21 konfigüras | `notebooks/en/04_runtime_prediction_models.ipynb` |
| `F-C05` | MAJOR | ch6 L68'in 'iş süresi zamansal otokorelasyona sahip değil' iddiası ölçülmemiş ve yanlış: l | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `F-C06` | MAJOR | Tek bir 1,59 günlük test penceresi: 7 day_of_week seviyesinden yalnız 2'si test ediliyor,  | `src/feature_engineering.py` |
| `F-C07` | MAJOR | Tezin motivasyonunu doğrudan sınayacak taban çizgisi (Alibaba'nın kendi süre tahmini) alty | `configs/paths.yaml` |
| `leakage-2` | MODERATE | arrival_sec ham öznitelik: test değerleri eğitim aralığının tamamen dışında; ağaç modeller | `src/feature_engineering.py` |
| `leakage-5` | MODERATE | Bağımsız gözlem birimi belirsiz: işlerin ~%24'ü aynı kullanıcı-aynı saniye 'burst' gruplar | `notebooks/en/04_runtime_prediction_models.ipynb / 05_scheduler_evaluation_32_gpu.ipynb` |
| `modeling-6` | MODERATE | 21 konfigürasyon aynı test setinde karşılaştırılıp 'en iyi' test MAE'sine göre seçiliyor;  | `notebooks/en/04_runtime_prediction_models.ipynb` |
| `modeling-9` | MODERATE | Seçim ölçütü ve kayıp fonksiyonu modeller arasında tutarsız; hepsi MAE ile raporlanıyor | `src/tuning.py` |
| `statistics-6` | MODERATE | Tahmin metriklerinde hiç güven aralığı yok; test hataları zaman-korelasyonlu ve test pence | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `baselines_claims-10` | MODERATE | SJF-Oracle 'teorik maksimum/ütopik optimum' değil: katı HoL'lu, backfill'siz, kesintisiz S | `src/simulation/scheduler_simulator.py; thesis/latex/chapters/6.results_and_discussion.tex` |
| `baselines_claims-11` | MODERATE | 'HoL blocking / fillerden fareleri koruma' mekanizma iddiası ölçülmemiş; SJF politikaları  | `thesis/latex/chapters/6.results_and_discussion.tex; notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` |
| `robustness-5` | MODERATE | Zamansal öznitelikler (day_of_week, arrival_sec, hour_of_day) 7,7 günlük izde takvim indek | `src/feature_engineering.py` |
| `robustness-7` | MODERATE | Ağır kuyruk: MAE'nin %33-49'u en uzun %5 işten geliyor; işlerin %95'inde öğrenilen modelle | `thesis/latex/chapters/4.prediction_models.tex` |
| `robustness-9` | MODERATE | Model başarımı büyük ölçüde kullanıcı kimliğine bağlı (user lookup); görülmemiş kullanıcıl | `results/models/xgb_categorical.joblib` |
| `robustness-10` | MODERATE | Model/politika 'şampiyon' seçimi ve tüm raporlama aynı test seti üzerinde; 20 tahminci + 2 | `notebooks/en/04_runtime_prediction_models.ipynb` |
| `F-C11` | MODERATE | Etkin örneklem büyüklüğü raporlanan n'in çok altında: 82.184 iş yalnız 3.146 farklı öznite | `src/models/evaluation.py` |
| `baselines_claims-9` | MINOR | Tuning bütçesi ve kayıp fonksiyonları eşit değil: LGBM L1, XGB/RF kare hata, DL MSE ile eğ | `src/tuning.py; configs/models.yaml; thesis/latex/chapters/4.prediction_models.tex` |
| `leakage-8` | INFORMATIONAL | Üçüncü bir final hold-out yok: aynı 16.437 işlik test seti 21 modelin metriklerinde, şekil | `notebooks/en/04_runtime_prediction_models.ipynb; notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` |

## ❓ SENİN KARARINI GEREKTİRENLER — yapılmadı (13)

Kod tarafı mekanik olarak fixlenebilir ama sonucu değiştirecek bir tasarım kararı gerektiriyor (ör. cluster_load'u CPU işlerini de sayacak şekilde yeniden hesaplamak — bu TÜM downstream sayıları değiştirir). Onayın olmadan yapılmadı.

| ID | Önem | Konu | Dosya |
|---|---|---|---|
| `leakage-3` | MODERATE | Nihai modellerin eğitim verisi kullanımı model ailesine göre asimetrik (RF %100, XGB/LGBM  | `src/tuning.py` |
| `leakage-4` | MODERATE | Sweep-line küme yükü öznitelikleri CPU-only işler (17.816, %17,8) filtrelendikten SONRA he | `src/feature_engineering.py` |
| `modeling-2` | MODERATE | XGBoost: erken durdurma ölçütü tuning'de MAE, final refit'te varsayılan RMSE | `src/tuning.py` |
| `modeling-3` | MODERATE | n_estimators hiperparametresi fiilen kullanılmıyor: 200-1300 ayarlanıyor, 18-243 ağaçta du | `src/tuning.py` |
| `modeling-11` | MODERATE | Dar grid seçimleri sıkça komşuluğun sınırına düşüyor: arama yakınsamamış | `src/tuning.py` |
| `simulator-8` | MODERATE | Kesirli GPU paylaşımı 'toplamsal kapasite' olarak modelleniyor; tez §5.2 'fractional alloc | `src/simulation/multi_node_simulator.py` |
| `simulator-10` | MODERATE | Simülatörün kendi teşhis çıktıları (utilization_history, backfilled_on_reserved) hiçbir de | `src/simulation/multi_node_simulator.py` |
| `traceability-11` | MODERATE | Checkpoint timestamp'ı her defter geçişinde yeniden yazılıyor; aynı DL modeli için üç fark | `src/tuning.py` |
| `reproducibility-5` | MODERATE | DL sonuçları donanıma bağlı: MPS'te eğitilen ağırlıklar aynı tohumla CPU'da farklı çıkıyor | `src/tuning.py` |
| `reproducibility-7` | MODERATE | Dropout hiperparametresi rastgele aramada örneklenip dar ızgara/final aşamasında sessizce  | `src/tuning.py; configs/models.yaml` |
| `figures_tables-14` | MODERATE | Şekil provenansı karışık: nb05-fig01..05 31 Ağu 15:41 backup koşusundan, mae_spearman_*.pn | `scripts/export_thesis_results.py` |
| `F-C10` | MODERATE | ch1 Amaç 4 ve ch5 L11 simülatörün bellek kaynağını da uyguladığını söylüyor; bellek kapasi | `thesis/latex/chapters/1.introduction.tex` |
| `robustness-11` | MINOR | Elenen 17.816 CPU işi (gpu_type=CPU, num_gpu=0) gerçek sürelere sahip; simülasyon ve clust | `src/feature_engineering.py` |

## 🔄 RERUN'A BAĞLI — yapılmadı (12)

Doğası gereği güncel/gerçek sayı gerektiriyor (checkpoint-tablo tutarlılığı, şekil provenance'ı, README senkronizasyonu). Sen NB04/NB05'i koşturduktan sonra otomatik olarak ya çözülecek ya da doğru sayılarla yeniden yazılabilecek.

| ID | Önem | Konu | Dosya |
|---|---|---|---|
| `simulator-1` | CRITICAL | Simülasyon rejimi 'flash crowd' değil, 26-37 günlük statik toplu-iş (batch) birikimi: sunu | `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` |
| `traceability-1` | CRITICAL | Tahmin sonuç tabloları (tab:predresults, Ek A tab:expa..expf-full, ch6 metni) depodaki hiç | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `traceability-2` | CRITICAL | Simülasyon tabloları (tab:schedresults, tab:wilcoxon, tab:waitpercentile) ve tüm türetilmi | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `figures_tables-1` | CRITICAL | Tablo 6.1 / Ek A ile Şekil 6.1 (nb04-fig01) aynı belge içinde birbirini yalanlıyor | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `figures_tables-2` | CRITICAL | Tablo 6.2 ve Şekil 6.6 (nb05-fig01) çelişiyor; şekil altyazısı şeklin gösterdiğinin tersin | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `statistics-3` | MAJOR | Önemsiz taban çizgileri (sabit medyan, sıfır, kullanıcı medyanı) tezde yok; öğrenilen mode | `results/checkpoints/*.json; thesis/latex/chapters/6.results_and_discussion.tex` |
| `traceability-13` | MAJOR | nb04/nb05 markdown anlatıları ve reports/html/pdf eski koşulardan; TR/EN defter anlatıları | `notebooks/en/04_runtime_prediction_models.ipynb` |
| `reproducibility-1` | MAJOR | Yayınlanmış (origin/main) kod ve artefaktlar tezin sonuçlarını üreten sürüm değil: tüm düz | `src/feature_engineering.py; src/simulation/multi_node_simulator.py; results/checkpoints/*.json; results/models/*; thesis/latex/chapters/*.tex` |
| `traceability-9` | MODERATE | Tüm LaTeX tabloları elle yazılmış (otomatik üretim yok); transkripsiyon hataları mevcut | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `traceability-10` | MODERATE | README, .github/context ve scripts/README üç ayrı eski koşunun sayılarını taşıyor; hiçbiri | `README.md` |
| `traceability-6` | MINOR | Tezdeki nb05 şekilleri (nb05-fig01..05_{32,256}gpu.png) mevcut defter çıktılarından ESKİ;  | `thesis/latex/figures/` |
| `figures_tables-19` | MINOR | HTML export tabloları LaTeX kaynağı olarak kullanılamaz: konumsal numaralama, veri önizlem | `scripts/export_thesis_results.py` |

## 📄 SADECE LATEX — bilerek BEKLETİLDİ (30)

Talebin üzerine LaTeX'e dokunmadım; bunlar tamamen metin/tablo/altyazı düzeltmesi. Rerun sonrası, gerçek sayılarla TEK geçişte toplu işlenecek.

| ID | Önem | Konu | Dosya |
|---|---|---|---|
| `statistics-2` | MAJOR | Tek yönlü Wilcoxon (alternative='greater') p=1.000 sonucu 'FIFO'dan ayırt edilemez' diye y | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `simulator-7` | MAJOR | tab:sim_time (FIFO 3,2 s, … 16,4 s) hiçbir kodla üretilmiyor; ölçtüğüm süreler 8-21 s | `thesis/latex/chapters/5.simulation_framework.tex` |
| `traceability-3` | MAJOR | tab:hyperparams'ın 19 satırının hiçbiri güncel best_params ile eşleşmiyor; 'tek sabit tohu | `thesis/latex/chapters/4.prediction_models.tex` |
| `traceability-4` | MAJOR | Simülasyon süre tablosu (tab:sim_time), '17 saniye/politika' ve README '100.000 olay < 60  | `thesis/latex/chapters/5.simulation_framework.tex` |
| `traceability-5` | MAJOR | 'Altı model / 18 tahminci / 4 politika / 3 koşu' sayımları koddaki 20 tahmin seti, 23 poli | `thesis/latex/chapters/5.simulation_framework.tex` |
| `baselines_claims-5` | MAJOR | 'Sweep-line active_job_count tüm ağaç modellerinde en önemli özniteliklerden biri' katkı i | `results/models/{rf,xgb,lgbm}_numeric.joblib; thesis/latex/chapters/1.introduction.tex; thesis/latex/chapters/6.results_and_discussion.tex; thesis/latex/chapters/7.conclusions_and_future_work.tex` |
| `figures_tables-3` | MAJOR | Tablo 6.3 (Wilcoxon) ve 6.4 (yüzdelikler) bayat; '(fail) = p≥0.05' yorumu tek yönlü testi  | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `figures_tables-4` | MAJOR | Tablo 5.x tab:sim_time (simülasyon süreleri) hiçbir ölçüm koduna dayanmıyor | `thesis/latex/chapters/5.simulation_framework.tex` |
| `figures_tables-5` | MAJOR | Tablo 4.x tab:hyperparams üç farklı parametre kümesiyle çelişiyor; LGBM-A ve LGBM-OH satır | `thesis/latex/chapters/4.prediction_models.tex` |
| `figures_tables-6` | MAJOR | Şekil 6.3 (feature importance) altyazısı 'MDI-based' yanlış: LightGBM split sayısı, XGBoos | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `figures_tables-7` | MAJOR | Şekil 6.5 (residual) altyazısı ve metni şekille çelişiyor: RF ortalama artık +10.601 s | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `figures_tables-8` | MAJOR | Şekil 3.3 (GPU demand) altyazısı veriyle çelişiyor: işlerin %52,5'i kesirli GPU istiyor, ' | `thesis/latex/chapters/3.dataset_and_workload.tex` |
| `figures_tables-13` | MAJOR | Isı haritası ve kutu grafiği: 'slowdown' sütunu ortalama (aykırı-güdümlü), metin medyan id | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `modeling-12` | MODERATE | Tez ch4 tuning anlatısı ve tab:hyperparams kodla/checkpoint'lerle uyuşmuyor | `thesis/latex/chapters/4.prediction_models.tex` |
| `statistics-10` | MODERATE | Tez metni p-değerini tek başına kullanıyor: etki büyüklüğü ve CI (nb05 c46'da hesaplanıyor | `thesis/latex/chapters/7.conclusions_and_future_work.tex` |
| `traceability-7` | MODERATE | Ch3 GPU-talep sayıları (ortalama 0.52, '~37.000 iş ≤1 GPU') eski tam-sayı kesme koşusundan | `thesis/latex/chapters/3.dataset_and_workload.tex` |
| `traceability-8` | MODERATE | Ch7 'bootstrap CI hesaplanmadı' ve ch6 'p=1.000 → FIFO'dan ayırt edilemez' ifadeleri kodun | `thesis/latex/chapters/7.conclusions_and_future_work.tex` |
| `baselines_claims-8` | MODERATE | 'Altı model / 18 tahminci / 4 politika / üç koşu' anlatısı ile gerçek deney (21 konfigüras | `thesis/latex/chapters/5.simulation_framework.tex; thesis/latex/chapters/1.introduction.tex; thesis/latex/chapters/4.prediction_models.tex` |
| `baselines_claims-12` | MODERATE | Zamanlama iddiaları ('under 17 seconds per policy', tab:sim_time 3.2-16.4 s, README '100.0 | `thesis/latex/chapters/1.introduction.tex; thesis/latex/chapters/5.simulation_framework.tex; README.md` |
| `baselines_claims-13` | MODERATE | 'Sequence trap' iddiası kısmen destekleniyor: sekans modelleri JCT'de statiklerden yalnız  | `thesis/latex/chapters/7.conclusions_and_future_work.tex; thesis/latex/chapters/6.results_and_discussion.tex` |
| `figures_tables-10` | MODERATE | Şekil 6.2 (DL MAE çubukları): tezdeki sürüm tek tohum ve hata çubuksuz; güncel sürümde 3 t | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `figures_tables-15` | MODERATE | Tablo 5.x tab:simconfig politika/model sayıları ve ikincil metrikler defterle uyuşmuyor | `thesis/latex/chapters/5.simulation_framework.tex` |
| `traceability-12` | MINOR | Ch3 EDA metnindeki şekilden okunmuş sayılar hesaplanan değerlerle kısmen uyuşmuyor | `thesis/latex/chapters/3.dataset_and_workload.tex` |
| `reproducibility-11` | MINOR | Tez metni yeniden üretim bilgisi açısından bayat/eksik: tek tohum iddiası, sürüm/donanım/s | `thesis/latex/chapters/4.prediction_models.tex; 7.conclusions_and_future_work.tex` |
| `baselines_claims-14` | MINOR | README/CHECKLIST sonuç anlatısı üçüncü bir eski koşudan ('2.25x speedup', 'Optimal Perform | `README.md; thesis/latex/frontmatter/abstract-en.tex` |
| `figures_tables-17` | MINOR | Tablo 6.1 kalın yazım kendi içinde tutarsız; 'top-performing' altyazısı ile 18 satır çeliş | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `figures_tables-20` | MINOR | Okunabilirlik: 21 politikalı şekiller 0.48\textwidth'e sıkıştırılmış; eksen etiketleri çak | `thesis/latex/chapters/6.results_and_discussion.tex` |
| `figures_tables-21` | MINOR | Şekil 3.1/3.9 altyazı-metin küçük uyuşmazlıkları (runtime mod sayısı, gpu_demand korelasyo | `thesis/latex/chapters/3.dataset_and_workload.tex` |
| `F-C08` | MINOR | HARKing izi: ch1 RQ2 ile RQ4 aynı soru, ch6'nın RQ4 yanıtı ise ch1'de hiç sorulmamış bir s | `thesis/latex/chapters/1.introduction.tex` |
| `F-C12` | MINOR | tab:related, tezin özgünlük iddialarını dayandırdığı iki çalışmayı (L2R ve DFL) tabloya al | `thesis/latex/chapters/2.background.tex` |

## 🔲 SIRA GELMEDİ — henüz işlenmedi (16)

Kod tarafı, muhtemelen küçük/orta efor, ama bu oturumda sıra gelmedi (zaman/kapsam).

| ID | Önem | Konu | Dosya |
|---|---|---|---|
| `modeling-10` | MODERATE | Tuning bütçesi ve doğrulama protokolü ML ile DL arasında asimetrik; yaml değerleri defterd | `notebooks/en/04_runtime_prediction_models.ipynb` |
| `robustness-8` | MODERATE | DL kayıp fonksiyonu MSE iken tez 'MAE ile eğitilen sinir ağları' diyor; nb05 DL tahminleri | `src/tuning.py` |
| `leakage-7` | MINOR | Sweep-line öznitelikleri çevrimiçi hesaplanabilir (gelecek/kendi runtime bilgisi kullanılm | `src/feature_engineering.py` |
| `modeling-13` | MINOR | Ölü/çelişkili yapılandırma yolları ve gereksiz refit | `configs/models.yaml` |
| `modeling-14` | MINOR | Ortam sürümleri tezde yok; defter çekirdeği ile sistem Python'u farklı sürümler taşıyor | `notebooks/en/04_runtime_prediction_models.ipynb` |
| `statistics-9` | MINOR | İstatistik kodu (Holm, rank-biserial, bootstrap) yalnız defter hücresinde; birim testi yok | `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` |
| `simulator-11` | MINOR | Test kapsamı: çok-düğümlü simülatörde SJFPred/SRF, çok-makine HoL, eşitlik sırası, düşürül | `tests/test_simulation.py` |
| `reproducibility-8` | MINOR | DL scaler'ları ve test seti her defter geçişinde güncel src ile yeniden üretiliyor; ağla e | `notebooks/en/04_runtime_prediction_models.ipynb; notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` |
| `reproducibility-9` | MINOR | Test paketi 34/34 geçiyor ama tuning/arama, DL eğitim, one-hot yolu, veri yükleme, export  | `tests/; scripts/README.md; CHECKLIST.md; .agents/skills/thesis-auditor/scripts/audit_thesis.py` |
| `reproducibility-10` | MINOR | Çalışma ağacı hijyeni: sahipsiz işlenmiş CSV, izlenmeyen kilit dosyası ve büyük yedek dizi | `data/processed/100k_job_with_utilization.csv; requirements-lock.txt; results/_backup_20260831_1703_pre_q2/` |
| `figures_tables-12` | MINOR | Mimari şemaları (architecture_en_*.png) simülatörde olmayan mekanizmaları gösteriyor | `thesis/latex/figures/architecture_en_32_gpu.png` |
| `figures_tables-16` | MINOR | Sweep-line şekli (nb03-fig01) ve öznitelikleri 0'dan başlıyor: iz başlangıcındaki ~12 saat | `notebooks/en/03_feature_engineering.ipynb` |
| `figures_tables-18` | MINOR | Rank-korelasyon şekli (nb05 c29): n=18 bağımlı nokta üzerinde Pearson r, CI yok, etiket ça | `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb` |
| `F-C09` | MINOR | gpu_type modelde öznitelik ama simülatörde hiç yok: heterojenlik yalnız GPU sayısında; iş  | `src/simulation/multi_node_simulator.py` |
| `code_bugs-8` | INFORMATIONAL | [RECOVER] yolu modeli yeniden eğitip metriklerini atıyor; checkpoint zaman damgası her geç | `notebooks/en/04_runtime_prediction_models.ipynb` |
| `code_bugs-10` | INFORMATIONAL | Ölü/yanıltıcı modüller ve test boşlukları: hiçbir defter kullanmıyor, testler asıl hata sı | `src/visualization.py` |
