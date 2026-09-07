# LaTeX Turu — Kalan 74 Bulgu (Bölüme Göre Gruplu)

Toplam: 74 | Bu liste kod tarafinda zaten cozulmus 59 + karar-verilmis 7 haric kalanlar.

---

## 1.introduction.tex (2 bulgu)

- **baselines_claims-12** [MODERATE] — Depoda hiçbir simülasyon zamanlama ölçümü yok; tab:sim_time'daki 18 satır (ondalık hassasiyetle) türetilemez. Kendi ölçümüm (32-GPU, LOAD 0.1, paylaşımlı CPU): FIFO 18.8 s, Oracle 
- **F-C08** [MINOR] — ch1 RQ2: 'Which predictive modeling paradigm provides the best approach for forecasting the extremely heterogeneous, heavy-tailed nature of runtime... compares tree-based ensembles

## 2.background.tex (1 bulgu)

- **F-C12** [MINOR] — Tez özgünlüğünü iki kavram üzerine kuruyor: Decision-Focused Learning (mandi2023decision) ve scheduling için Learning-to-Rank (fu2024efficient). Bu iki çalışma metinde 'en yakın ak

## 3.dataset_and_workload.tex (4 bulgu)

- **figures_tables-8** [MAJOR] — Altyazı: 'almost every request is for a single GPU. Multi-GPU requests (2-8) occur rarely'. Şekil (log-x, kesirli değerler) ve ham veri: gpu_demand==1 %45,1; <1 (0.01–0.8) %52,5; >
- **traceability-7** [MODERATE] — Ham veriden yeniden hesapladım: gpu_demand ortalama 0.680, medyan 0.50, min 0.01, max 8 (nb01 cell 21 çıktısıyla aynı). np.floor(gpu_demand) ortalaması 0.5216 → tezdeki '0.52' tam-
- **figures_tables-21** [MINOR] — Altyazı 'two distinct modes … around 300 s and in excess of 10,000 s' der; şekilde üç tepe var (log10≈1.5 → ~30 s, ≈2.4 → ~250-500 s, ≈3.6 → ~4.000-5.000 s); en yüksek üçüncü bin m
- **traceability-12** [MINOR] — Yeniden hesap: (a) log10 runtime ikinci mod 3.6–3.8 (≈4,000–6,300 s), tez '≈4.0 (≈10,000 s)'; nb01 c16 markdown '3.5–4.0' diyor. (b) Saatlik varış medyanı 421, P25–P75 318–545 → 'b

## 4.prediction_models.tex (4 bulgu)

- **figures_tables-5** [MAJOR] — Tezde RF-A: n_estimators=150, max_depth=20; backup HTML Table09: max_depth 20, max_features 0.7; güncel exp_a_rf.json: max_depth=5, max_features='sqrt', n_estimators=100 (diskteki 
- **traceability-3** [MAJOR] — Güncel results/checkpoints best_params ile satır satır karşılaştırma: RF-A tez n_est=150/max_depth=20/max_features=0.7 → güncel 100/5/'sqrt'/min_samples_leaf=3; XGB-A 400/lr .025/d
- **modeling-12** [MODERATE] — (1) Pencere boyutu w ayarlanmıyor (deney başına 1/10 sabit: nb04 c35/45/56/66); dropout final modellerde ayarlanmıyor (modeling-5). (2) DL final refit 3 tohumla (nb04 c6 DL_SEEDS=[
- **reproducibility-11** [MINOR] — DL final eğitimi artık DL_SEEDS=[42,1337,2024] ile 3 kez koşuyor (nb04 cell 6; checkpoint 'seeds': [42,1337,2024]); tezde tek tohum yazıyor. Kütüphane sürümleri, cihaz (MPS), topla

## 5.simulation_framework.tex (5 bulgu)

- **figures_tables-4** [MAJOR] — 18 politika × 2 küme için ondalık saniye hassasiyetinde 'ölçülen' süreler veriliyor (FIFO 3.2 s / 12.4 s … CNN-LSTM 4.4 s / 16.4 s); depo genelinde (src/simulation, nb05) time.time
- **traceability-4** [MAJOR] — tab:sim_time'daki 36 değer (FIFO 3.2 s / 12.4 s … Hybrid-CatSeq 4.4 s / 16.4 s) ve ch1 L95 'all 16,437 test jobs in under 17 seconds per policy' ifadesinin üretildiği hiçbir kod yo
- **traceability-5** [MAJOR] — Tez: 'six models', '18 models (Experiments A–F)', 'eighteen unique ML predictor models', 'Running the simulator three times (FIFO, SRF, ML-SJF)', '4 policy types'. Kod: nb05 cell 2
- **baselines_claims-8** [MODERATE] — Tez 'six models' derken 6 mimari × kodlama × pencere = 18 konfigürasyonu kastediyor; ancak güncel nb04'te 21 checkpoint (XGB-Native ve Per-User Median eklenmiş), nb05'te 23 politik
- **figures_tables-15** [MODERATE] — '4 policy types: FIFO, SRF, SJF-Oracle, SJF-Pred' ve '18 models' yazıyor; defter 23 politika koşturuyor (SJF-XGBoost (Native Cat) ve SJF-UserMedian (baseline) dahil, 20 tahmin seti

## 6.results_and_discussion.tex (14 bulgu)

- **figures_tables-2** [CRITICAL] — Tablo 32-GPU için FIFO ortalama JCT 499.951 s, XGB-Cat %56.25 (en iyi ML) diyor; basılan şekil FIFO 1.344.966 s, SJF-LSTM (Categorical) %59.8 (en iyi ML), XGB-Cat %59.6 gösteriyor.
- **traceability-2** [CRITICAL] — Tezdeki 32-GPU FIFO 499,951 / Oracle 92,064 / XGB-Cat 56.25% ve 256-GPU FIFO 50,386 / Oracle 14,506 / LSTM-Cat 57.25% değerleri, Wilcoxon W'ları (133,148,621; 109,737,336) ve yüzde
- **F-C05** [MAJOR] — Tez, Deney E/F'deki (sekans modelleri) negatif R²'yi tek bir nedensel açıklamayla kapatıyor: 'Since GPU job runtime lacks temporal autocorrelation in this case, we can conclude tha
- **figures_tables-13** [MAJOR] — Heatmap 'Slowdown ↓ %' FIFO ortalama slowdown (32-GPU: 8.085; 256: 891) üzerinden hesaplanıyor; Oracle için '%99.5/99.6 azalma' ortalama-artefaktı, kutu grafiğindeki medyanlarla (O
- **figures_tables-3** [MAJOR] — Tablodaki W değerleri (örn. Oracle 133.148.621) ve JCT'ler HTML/notebook çıktısıyla (Oracle W 127.580.168, JCT 252.024) uyuşmuyor. Altyazı ''(fail)'' denotes p ≥ 0.05' ve metin 'no
- **figures_tables-6** [MAJOR] — nb04 c80 üç model için de model.feature_importances_ çiziyor; RandomForest'ta bu MDI (toplam 1.0), XGBRegressor'da importance_type=None→'gain' (toplam ≈1.0), LGBMRegressor'da impor
- **figures_tables-7** [MAJOR] — Altyazı 'Each model has residual values centered about zero', metin 'none of these models shows a significant positive or negative bias' ve 'all three models consistently underesti
- **statistics-2** [MAJOR] — Kod H1: FIFO > Policy tek yönlü testi kurar (`alternative="greater"`). Bu testte p≈1, farkların ters yönde toplandığını gösterir; 'ayırt edilemez' (iki yönlü p büyük) anlamına gelm
- **baselines_claims-11** [MODERATE] — Simülasyon çıktılarında (kendi koşum, 32-GPU) Oracle altında <10 dk işler medyan 342 s beklerken >1 gün işler ortalama 2.570.304 s (≈30 gün) bekliyor; UserMedian'da 1.766.256 s. Ya
- **figures_tables-10** [MODERATE] — Basılı şekil (backup) 12 çubuk, bıyık yok, değerler tek tohum (6.481, 7.186 …). Güncel kod bıyık = 1 SD (n=3; SEM/CI değil). Güncel checkpoint'lerde mae_std 123–739 s (exp_d_hybrid
- **statistics-6** [MODERATE] — Tek test seti üzerinden nokta MAE/MdAE/R² raporlanıyor; modeller arası farkların (ör. XGB-OH 6.642 vs LGBM-OH 6.640) belirsizliği yok. Mutlak hatalar zaman sırasında otokorelasyonl
- **traceability-9** [MODERATE] — Depoda checkpoint/CSV/HTML'den .tex üreten hiçbir betik yok; export_thesis_results.py yalnızca PNG kopyalar ve HTML tablo dosyası yazar (LaTeX'e köprü yok). Aynı büyüklük aynı bölü
- **figures_tables-17** [MINOR] — Altyazı 'Bold indicates the best result per metric' der; MAPE sütununda 10.80 kalın, oysa aynı tabloda LSTM (One-Hot) 6.51 daha düşük (Ek A tab:expd-full'da 6.51 kalın). Altyazı 't
- **figures_tables-20** [MINOR] — Yüzdelik şekli her biri 2 panelli iki PNG'nin yan yana konmasıyla 4 panel × 21 etiket; kutu grafiğinde 35° döndürülmüş 21 etiket üst üste biniyor ('SRF (Heuristic)' etiketi 'SJF-CN

## 7.conclusions_and_future_work.tex (3 bulgu)

- **baselines_claims-13** [MODERATE] — Güncel: LSTM-Cat-Seq JCT %58.47 (32) / %60.81 (256) — 3./4. sırada, statik LSTM-Cat'ten 1.8-2.3 puan geride; MAE'de seed-42 ağı 5.939 s (statik LSTM-Cat 6.706), 3-tohum 6.424 vs 6.
- **statistics-10** [MODERATE] — Tezde yalnız W ve p raporlanıyor; rank-biserial r ve bootstrap CI defterde var ama teze taşınmamış; ch7 tersini iddia ediyor. c45 p-değerini 'sonucun şans eseri olma olasılığı' ola
- **traceability-8** [MODERATE] — nb05 cell 46 her politika için 999-örnek percentile bootstrap %95 CI, rank-biserial etki büyüklüğü ve Holm-Bonferroni düzeltmesi hesaplıyor; ch7 L72 'this thesis did not calculate 

## en.tex (1 bulgu)

- **baselines_claims-1** [CRITICAL] — Tezin merkezi iddiası iki kümede farklı kazananlar olduğudur. Güncel nb05 c27 çıktılarında her iki kümede de en iyi ML politikası SJF-LSTM (Categorical) (32-GPU %60.28, 256-GPU %63

---

## Once gozden gecirilmeli — birincil konumu .tex degil (40 bulgu)

Bunlarin cogu "tez metni, ama kanit notebook/kod dosyasinda" turunde -- ama bazilarinin gercekten kod/config tarafinda da bir seyler gerektirip gerektirmedigini tek tek kontrol etmek lazim once.

- **baselines_claims-2** [CRITICAL] — konum: notebooks/en/04_runtime_prediction_models.ipynb; notebooks/en/05_scheduler_evalu
  Yalnız train'den hesaplanan kullanıcı-medyanı: MAE 5.191 (en iyi öğrenilen LGBM-Nat 5.697), MdAE 846 (vs 2.684), MAPE 543% (vs 1.660%), Spearman ρ=0.597 (scratch; en iyi öğrenilen 
- **baselines_claims-5** [MAJOR] — konum: results/models/{rf,xgb,lgbm}_numeric.joblib; thesis/latex/chapters/1.introductio
  Diskteki (nb05'in tükettiği, nb04 c47 _recover ile şekle giren) Exp-A modellerinin feature_importances_ değerleri: RF gpu_demand 0.305, num_cpu 0.240, num_inst 0.128, hour_of_day 0
- **baselines_claims-6** [MAJOR] — konum: src/tuning.py; thesis/latex/frontmatter/abstract-en.tex; thesis/latex/chapters/1
  Kodda DFL'nin tanımlayıcı unsuru (aşağı akış karar kaybının eğitime geri beslenmesi, türevlenebilir sıralama, regret vb.) yok; ranking loss/pairwise/listwise için grep 0 sonuç. Mod
- **modeling-7** [MAJOR] — konum: results/checkpoints/exp_b_lgbm_nat.json, exp_b_user_median.json
  Sabit tahmin ŷ=median(y_train)=568 s test MAE 5.818 s verir. En iyi öğrenilen model LGBM-Nat 5.697 s: fark −121 s, CI [−242, +6] → sıfırı içeriyor. RF-OH (+474), XGB-OH (+824), RF-
- **reproducibility-1** [MAJOR] — konum: src/feature_engineering.py; src/simulation/multi_node_simulator.py; results/chec
  GitHub'a itilmiş son commit 17 Ağustos tarihli. HEAD'deki src/feature_engineering.py:118 hâlâ `job["gpu_demand"] = job["num_gpu"].astype(int)` (kesirli GPU'yu sıfıra yuvarlayan hat
- **robustness-3** [MAJOR] — konum: notebooks/en/05_scheduler_evaluation_32_gpu.ipynb
  Test seti 16.437 iş, toplam GPU-iş yükü 72,73 M GPU-s; gönderim penceresi 137.101 s × 0.1 = 3,81 saat. Sunulan yük ρ = iş / (N_GPU × pencere): 32-GPU'da LOAD 0.1 → 166×, LOAD 1.0 →
- **simulator-2** [MAJOR] — konum: src/simulation/multi_node_simulator.py
  Kod her politika için scheduler.select_job ile tek bir 'en iyi' işi seçer; hiçbir makineye sığmıyorsa döngü kırılır ve bir sonraki FINISH'e kadar hiçbir başka pending iş yerleştiri
- **simulator-3** [MAJOR] — konum: src/simulation/multi_node_simulator.py
  Tam iz için Σ(g·rt)/span = 512,7 GPU, test dilimi için 530,5 GPU (LF=1.0). Yani iz, ortalama 500+ GPU'yu sürekli meşgul eden bir kümeden alınmış (tez ch3 L9: ~1.800 makine, >6.500 
- **statistics-3** [MAJOR] — konum: results/checkpoints/*.json; thesis/latex/chapters/6.results_and_discussion.tex
  İstatistiksel anlamda 'model öğreniyor' iddiası bir referans noktası gerektirir. Test setinde sabit train-medyanı (568 s) MAE=5.818, sabit 0 tahmini MAE=6.030, Per-User Median MAE=
- **statistics-4** [MAJOR] — konum: notebooks/en/05_scheduler_evaluation_32_gpu.ipynb
  Şekil 'MAE vs JCT gain' ve 'ρ vs JCT gain' için yalnızca r değeri gösterir; n=18, noktalar aynı test seti ve aynı simülasyon evreninden (bağımsız değil), CI/p yok. UserMedian (ρ=0,
- **traceability-13** [MAJOR] — konum: notebooks/en/04_runtime_prediction_models.ipynb
  nb04 markdown: 'Random Forest ... Test MAE (4316.39) ... R² 0.27', 'XGBoost One-Hot ... MAE (3389.26) ... R² 0.51', 'LSTM ... 13169.47 ... R² −0.27' — hemen üstündeki kod hücreleri
- **F-C03** [MODERATE] — konum: notebooks/en/05_scheduler_evaluation_32_gpu.ipynb
  Tahminler, gerçek iz zaman çizelgesinden hesaplanan cluster_load_cpu / cluster_load_gpu / active_job_count / arrival_sec / hour_of_day öznitelikleriyle üretiliyor. Simülasyon ise L
- **F-C11** [MODERATE] — konum: src/models/evaluation.py
  Tüm metrikler, istatistiksel testler ve bootstrap CI'ler 82.184 / 16.437 işi bağımsız gözlem sayıyor. Oysa (user, num_inst, num_cpu, num_gpu, gpu_type) beşlisi bakımından tüm izde 
- **baselines_claims-10** [MODERATE] — konum: src/simulation/scheduler_simulator.py; thesis/latex/chapters/6.results_and_discu
  Oracle, gerçek runtime ile SJF sırasıdır; first-fit + HoL + backfill=False altında koşar, kesinti/SRPT/paketleme yok. Kendi koşum: Oracle backfill=False 252.024 s, backfill=True 20
- **modeling-10** [MODERATE] — konum: notebooks/en/04_runtime_prediction_models.ipynb
  ML: 20 RS × 3 kat + 45-54 grid × 3 kat ≈ 195-222 fit, TimeSeriesSplit ile üç ayrı zaman dilimi üzerinden skor (iç eğitim boyutları 13.973/27.943/41.914 satır, sarmalayıcı %15'i erk
- **modeling-4** [MODERATE] — konum: src/tuning.py
  Hedef MinMaxScaler ile [0,1]'e ölçeklenmiş (data_max_=599.445 s); ölçekli val MSE 5.5e-4–6.2e-4 mertebesinde. delta=1e-4 mutlak eşiği, bir epoch'un 'iyileşme' sayılması için kaybın
- **modeling-6** [MODERATE] — konum: notebooks/en/04_runtime_prediction_models.ipynb
  Hiperparametre seçimi iç doğrulamayla yapılıyor (doğru), ancak deney/model seçimi (21 aday) doğrudan test seti üzerinden yapılıyor ve raporlanan 'en iyi' skoru bu seçimden dolayı i
- **modeling-9** [MODERATE] — konum: src/tuning.py
  DL modelleri MSELoss ile eğitilip MinMax-ölçekli val RMSE ile seçiliyor (ölçek data_max_=599.445 s → kuyruk baskın); ML modelleri neg_MAE ile seçiliyor ama RF/XGB L2, LGBM tuning L
- **robustness-10** [MODERATE] — konum: notebooks/en/04_runtime_prediction_models.ipynb
  Hiperparametre seçimi eğitim-içi CV ile (doğru), fakat hangi modelin/politikanın 'en iyi' olduğu kararı yalnız test setinde ve tek koşuda veriliyor; simülasyon evreni de aynı test 
- **robustness-8** [MODERATE] — konum: src/tuning.py
  DL modelleri MSE ile eğitilip ölçekli val-RMSE ile seçiliyor, tez ch6 L71 'Neural networks, especially when trained with MAE' diye açıklıyor. nb05 c17 tahminleri np.maximum(...,0) 
- **traceability-10** [MODERATE] — konum: README.md
  README 'Model Predictive Performance' (XGB-OH 3,389/0.51) 12 Mayıs koşusu; README 'Scheduling Optimization' (FIFO 715,611, Oracle 121,128, SJF-XGB 318,257 = 2.25×) yalnızca README'
- **F-C09** [MINOR] — konum: src/simulation/multi_node_simulator.py
  İz dört GPU tipi içeriyor (MISC 49.316, T4 25.484, P100 5.023, V100 2.361) ve gpu_type ML modellerinde kategorik öznitelik. Simülatörde ise gpu_type hiç geçmiyor: Machine sınıfının
- **baselines_claims-14** [MINOR] — konum: README.md; thesis/latex/frontmatter/abstract-en.tex
  README tabloları (FIFO 715.611 s, SJF-XGB 318.257 = 2.25×, XGB-OH MAE 3.389/R² 0.51) ne güncel checkpoint/nb05 ile ne tezle ne backup ile eşleşiyor; README 'SRF fewest GPUs/CPUs' d
- **baselines_claims-9** [MINOR] — konum: src/tuning.py; configs/models.yaml; thesis/latex/chapters/4.prediction_models.te
  (1) LightGBM objective=regression_l1 (medyan-benzeri hedef) ile eğitilirken XGB reg:squarederror ve DL MSE ile eğitiliyor; rapor ölçütü MAE → LGBM-Native'in 'en düşük MAE'si kısmen
- **figures_tables-12** [MINOR] — konum: thesis/latex/figures/architecture_en_32_gpu.png
  Elle yapılmış iki diyagram yalnız '(32 GPUs)/(256 GPUs)' etiketiyle ayrışıyor (md5 854d5a92… vs 6d56c0ca…); düğüm sayısı (10 / 80) ve profilleri (8 GPU/96 CPU, 2 GPU/64 CPU) göster
- **figures_tables-16** [MINOR] — konum: notebooks/en/03_feature_engineering.ipynb
  Şekilde active_job_count ve cluster_load_gpu 0'dan başlayıp ilk ~0.5 günde 400-600'e tırmanıyor; iz öncesi başlamış işler bilinmediğinden ilk gündeki tüm işlerin küme-yükü özniteli
- **figures_tables-18** [MINOR] — konum: notebooks/en/05_scheduler_evaluation_32_gpu.ipynb
  18 model aynı test seti üzerinde değerlendirildiğinden noktalar bağımsız değil; Pearson r (−0.352 / 0.713) güven aralığı ve p verilmeden 'MAE değil sıralama önemli' iddiasına kanıt
- **figures_tables-19** [MINOR] — konum: scripts/export_thesis_results.py
  NB05_*_Table01.html = sim_jobs.head(3) (34 sütunlu veri önizlemesi, sonuç tablosu değil); NB04_Table09 parametreleri '...' ile kesik; herhangi bir hücreye display() eklenince tüm n
- **leakage-7** [MINOR] — konum: src/feature_engineering.py
  Perturbasyon testi: bir işin runtime'ını değiştirmek kendi özniteliklerini değiştirmiyor ve kendisinden ÖNCE varan hiçbir işin özniteliğini değiştirmiyor (yalnız sonraki işler etki
- **modeling-13** [MINOR] — konum: configs/models.yaml
  models.* bloğu (örn. lgbm objective regression_l1, hyperparameters) hiçbir defter tarafından okunmuyor; gerçek protokol nb04 hücrelerinde ve tuning.py sabitlerinde. RandomizedSearc
- **modeling-14** [MINOR] — konum: notebooks/en/04_runtime_prediction_models.ipynb
  Artefaktlar tam nesne pickle'ı (.joblib, torch.save(model) + weights_only=False). Bu denetimde her iki ortamda da yüklendiler ve aynı metrikleri verdiler (iyi), ama sürüm kırılganl
- **reproducibility-10** [MINOR] — konum: data/processed/100k_job_with_utilization.csv; requirements-lock.txt; results/_ba
  data/processed/100k_job_with_utilization.csv (30 Ağu 23:15, farklı sütun sırası, hiçbir kod okumaz, izlenmiyor), requirements-lock.txt izlenmiyor, results/_backup_… (eski checkpoin
- **reproducibility-8** [MINOR] — konum: notebooks/en/04_runtime_prediction_models.ipynb; notebooks/en/05_scheduler_evalu
  nb04 cell 35/45 checkpoint yüklense bile scaler'ları yeniden fit edip diske yazar (4 scaler mtime 1 Eyl 00:28; ağlar 31 Ağu 21:08–1 Eyl 00:15). nb05 ağı diskteki .pth'den, scaler'ı
- **reproducibility-9** [MINOR] — konum: tests/; scripts/README.md; CHECKLIST.md; .agents/skills/thesis-auditor/scripts/a
  34 test 0,31 s'de geçer (VERIFIED). Kapsam: tuning.py'nin 33 public sembolünden 5'i (seed_everything, chronological_train_validation_split, save_checkpoint, train_dl_model, create_
- **simulator-11** [MINOR] — konum: tests/test_simulation.py
  23 test çalışıyor (OK, 1,4 s). Kapsanan: FIFO/SJF tek-sunucu ClusterSimulator, iki makineye yerleştirme, GPU limiti, NaN, EASY backfill temel senaryoları, n_cpu=0. Kapsanmayan: SJF
- **statistics-9** [MINOR] — konum: notebooks/en/05_scheduler_evaluation_32_gpu.ipynb
  Uygulamalar doğru (aşağıda VERIFIED) ama src/ altında değil ve test edilmiyor; Holm düzeltmesi 22 test için uygulanıyor, ancak 32 ve 256 GPU ayrı aileler olarak ele alınıyor ve c30
- **traceability-6** [MINOR] — konum: thesis/latex/figures/
  Export 31 Ağu 17:33'te koştu ve NB05_32GPU/NB05_256GPU için yalnızca Figure01..06 (6 dosya) üretti; EXPECTED_FIGURE_COUNT['NB05_*']=7 olduğundan _sync_thesis_figures NB05'i atladı.
- **code_bugs-10** [INFORMATIONAL] — konum: src/visualization.py
  grep: visualization/RandomForestPredictor/XGBPredictor/LightGBMPredictor/ClusterSimulator/use_processed=True hiçbir .ipynb'de geçmiyor (yalnız src/__init__, tests). Testler (34, OK
- **code_bugs-8** [INFORMATIONAL] — konum: notebooks/en/04_runtime_prediction_models.ipynb
  Recovery dalında disk modeli yeniden üretilirken checkpoint 'metrics' eski koşudan kalır → diskteki model ile JSON metrikleri farklı nesnelere ait olabilir; save_checkpoint her def
- **leakage-8** [INFORMATIONAL] — konum: notebooks/en/04_runtime_prediction_models.ipynb; notebooks/en/05_scheduler_evalu
  Model seçimi düzeyinde test seti kullanılmıyor (RandomizedSearchCV yalnız X_train; DL random/grid search yalnız val — kodda VERIFIED). Ancak tasarım düzeyinde (hangi deneylerin, ha
---

# EK: Şekil/Tablo Turunda Ortaya Çıkan YENİ LaTeX Bulguları

Bu bölüm 2026-09-06'daki şekil/tablo gözden geçirmesinde bulundu. Yukarıdaki
74 bulgunun DIŞINDA. Kod tarafı düzeltildi; aşağıdakiler LaTeX'te kaldı.

## L-01 [BLOCKER] — ch6 öznitelik-önemi paragrafı şekliyle çelişiyor

**Yer:** `chapters/6.results_and_discussion.tex:83`

Metin şunu iddia ediyor:
> *"Each model indicates `active_job_count` as one of the primary indicators of
> job runtime"* ve *"Random Forest's emphasis on arrival times, XGBoost's focus
> on resource requests, and LightGBM's reliance on global metrics"*

`NB04-Figure02` bunu göstermiyor:

| Model | 1. sıra | `active_job_count` sırası |
|---|---|---|
| Random Forest (Numeric) | `gpu_demand` | **7/9** |
| XGBoost (Numeric) | `gpu_demand` | **5/9** |
| LightGBM (Numeric) | `arrival_sec` | 4/9 |

- "Her model birincil gösterge olarak işaret ediyor" → hiçbirinde ilk 3'te değil.
- "Random Forest'ın varış zamanlarına vurgusu" → RF'de `arrival_sec` 5. sırada;
  RF'nin vurgusu kaynak taleplerinde (`gpu_demand`, `num_cpu`, `num_inst`).
- "XGBoost'un kaynak taleplerine odağı" → bu doğru.
- "LightGBM'in küresel ölçütlere dayanması" → LightGBM'in 1.'si `arrival_sec`
  (zamansal), `cluster_load_gpu` ise **sıfır** bölünme alıyor.

**Yapılacak:** Paragraf yeniden yazılacak. Rerun sonrası şekil değişebilir, o
yüzden metin rerun'dan SONRA yazılmalı — sayılara bakılarak.

## L-02 [MAJOR] — Tez, kodun üretmediği politika adlarını raporluyor

LaTeX'te geçip NB05'in hiç üretmediği etiketler:

| LaTeX'te yazan | Kodda gerçekte olan |
|---|---|
| `SJF-LGBM (Categorical)` | yok |
| `SJF-LGBM (Native)` | `SJF-LGBM (Native Cat)` |
| `SJF-LGBM (One-Hot)` | yok |
| `SJF-XGB (Categorical)` | `SJF-XGBoost (Categorical)` |
| `SJF-XGB (Numeric)` | `SJF-XGBoost (Numeric)` |
| `SJF-Pred`, `SJF-Pred (ML)`, `SJF-Random` | yok |

Kodda VAR ama tezde hiç geçmeyenler (yeni eklenenler, sonuçlara girmeli):
`SJF-AlibabaEstimate (baseline)`, `SJF-AlibabaEstimate-Group (baseline)`,
`SJF-AlibabaEstimate-GroupGPU (baseline)`, `SJF-ProfileMedian (baseline)`,
`SJF-UserMedian (baseline)`, `SJF-XGBoost (Native Cat)`,
`SJF-LGBM (No Cluster-Load Features)`.

**Yapılacak:** Rerun sonrası NB05 tablolarındaki adlar neyse tez onlara
uydurulacak. Ters yön DEĞİL — kod doğru, metin eski.

## L-03 [MAJOR] — NB05 politika adlandırması kendi içinde tutarsız (KOD tarafı)

`SJF-RF (...)` ve `SJF-LGBM (...)` kısaltılmış, `SJF-XGBoost (...)` tam yazılmış.
Aynı tabloda üç modelin ikisi kısaltma, biri tam ad.

**Not:** Bu aslında kod tarafı ama NB05 koşarken keşfedildi, rerun'dan sonra
düzeltilmeli (şimdi değiştirmek koşan simülasyonu bozar). Karar gerekiyor:
hepsi tam ad mı (`SJF-Random Forest`, `SJF-LightGBM`, `SJF-XGBoost`) yoksa
hepsi kısaltma mı (`SJF-RF`, `SJF-LGBM`, `SJF-XGB`)?
NB04 tarafında tam ad seçildi — tutarlılık için NB05'te de tam ad önerilir.

## L-04 [MODERATE] — Tablo adları/birimleri değişti, altyazılar güncellenmeli

Kod tarafında yapılan (commit `1a98ea5`):
- Her model tek kanonik adla: `<Kütüphane adı> (<öznitelik kodlaması>)`
- Her sayısal başlıkta birim: `Test MAE (s)`, `Test MAPE (%)`
- Ablasyon tablosundan iç denetim ID'leri (`leakage-2` vb.) çıkarıldı
- Tablo 13 başlıkları ham anahtarlardan (`mae_std`) okunabilir hale getirildi

Tez altyazıları/metni bu adlara referans veriyorsa güncellenmeli.

## L-05 [MODERATE] — Rerun sonrası TÜM sayılar değişecek

`cluster_load_cpu` (+%15,2) ve `active_job_count` (+%22,5) düzeltildi
(commit `09cf225`). Bu oturumda yeniden hesaplanan ilk tablo (`NB04_Table06`)
zaten farklı çıktı: RF pencere-1 MAE 7122,99 → 7105,66.

**Yapılacak:** Rerun bitmeden tezdeki HİÇBİR sayı güncellenmeyecek. Rerun
sonrası ch4/ch5/ch6'daki her sayı yeni çıktılarla karşılaştırılacak.

## L-06 [MINOR] — Şekil altyazısı gereksinimleri

`TEZ_METNI_NOTLARI.md` §5'teki tabloya bakılacak. Bu turda değişenler:
- nb02-fig03: n artık şeklin lejantında → altyazıda tekrar gerekmiyor
- nb04-fig01/05, nb05-fig01/02/03/04: ok anahtarı şeklin içinde → altyazıda
  "düşük/yüksek olan iyidir" cümlesi gerekmiyor
- nb01-fig06: beyaz hücreler = izin ulaşmadığı saatler → altyazıda MUTLAKA
  yazmalı (renk skalasının en düşüğü açık sarı, beyaz "sıfır iş" sanılabilir)

## L-09 [BLOCKER] — Tezin ana negatif sonucu karıştırıcı bir etkenle sakatlanmış

**Durum:** Kod tarafı yapıldı (yeni `Per-Group Median` baseline), tez metni
YENİDEN YAZILMALI.

### Bulgu

Alibaba'nın SJG/SJGG tahmincileri tezdeki 24 modelin hepsini yeniyor
(SJGG: MAE 3151 s, R² 0,347 · en iyi eğitilmiş model: 4973 s, R² 0,04).
Sebebi Alibaba'nın yönteminin gelişmişliği değil: bu tahminciler **`group`**
alanı (izin görev-kimliği alanı) üzerine kurulu ve **`group` birincil veri
dosyasında (`pai_job_no_estimate_100K.csv`) yok.** Yalnız tahmin dosyasında
var. Yani tezdeki hiçbir model bu özniteliği göremiyor.

### Ölçümler (bu oturumda yapıldı, hepsi yeniden üretilebilir)

Eğitimde görülmüş bir gruba ait test işlerinde (9.586 iş):

| Yöntem | MAE (s) | R² |
|---|---|---|
| Alibaba SJGG | 1588 | 0,778 |
| **Sızıntısız grup medyanı** (yalnız eğitim satırlarından) | **1532** | **0,835** |

Yani **sızıntı yok** — kanıtlanabilir şekilde nedensel bir grup medyanı
Alibaba'nın tahmincisini zaten geçiyor. Avantajın tamamı `group`'un kendisinden.

Tüm test seti üzerinde (eşleşmeyenler global medyana düşerek):
`Per-Group Median` MAE 3369 / R² 0,334 — SJGG'nin (3151 / 0,347) hemen ardında,
en iyi eğitilmiş modelin (4973 / 0,04) çok önünde.

`group`'u kategorik öznitelik olarak LightGBM'e verdim: 5313 → 5155. Neredeyse
hiç. 7.478 seviyeli bir kategoriden derinlik-5 bir ağaç grup-başına toplamı
çıkaramıyor; doğru fonksiyonel biçim ağaç bölünmesi değil, doğrudan toplam.

### Tez metnine etkisi

Şu anki anlatı — "çalışma süresi zor tahmin edilir, R² 0,19'u geçmiyor,
derin öğrenme katkı sağlamıyor" — **olduğu gibi savunulamaz**. Doğru ifade:

> Tahmin edilebilir sinyalin büyük kısmı görev-kimliği (`group`) alanında ve bu
> alan çalışmanın birincil veri dosyasında yer almıyor. Bu alana erişimi olan
> bir tahminci (Alibaba'nın kendi SJG/SJGG'si ya da eğitim setinden alınmış
> basit bir grup medyanı) R² 0,78–0,84'e ulaşıyor; erişimi olmayan 24 model
> 0,19'un altında kalıyor. Dolayısıyla düşük tahmin başarısı, çalışma süresinin
> özünde öngörülemez olduğunu değil, kullanılan öznitelik kümesinin en
> bilgilendirici alanı içermediğini gösteriyor.

**Yapılacak:** ch4 ve ch6'daki "ML katkı sağlamadı / süre öngörülemez" cümleleri
yukarıdaki çerçeveye çekilecek; ch3'e `group`'un birincil dosyada olmadığı ve
yalnız tahmin dosyası üzerinden kısmi (61.203/100.000) eşleşmeyle erişilebildiği
yazılacak; ch7 Future Work'e "group'u tam kapsamlı elde etmek" maddesi eklenecek.

## L-10 [MINOR] — nb01-fig04 artık log eksenli değil

GPU talebi dağılımı log ekseninden **eşit aralıklı kategorik** düzene geçti.
Sebebi kozmetik değil: log ekseninde çubuk genişliği veri biriminde sabit
tutulduğu için ekrandaki genişlik konuma göre değişiyordu; 0,2'deki çubuk
(~800 iş) 0,25'tekinden (~20.600 iş) dar çiziliyordu ve **çubuk alanı iş
sayısını temsil etmiyordu**. Ayrıca genişliğin yarısı hiçbir işin olamayacağı
aralıklara gidiyordu ve yakın seviyeler çakıştığı için 7 seviyenin etiketi
gizlenmek zorunda kalmıştı.

**LaTeX'te yapılacak:** Bu şekli "log ölçekli" diye anlatan cümleler
düzeltilecek. Eksen etiketi sade tutuldu (`GPU Demand per Job (GPUs)`), bu
yüzden şu bilgi **altyazıda MUTLAKA olmalı**: seviyeler ayrık ve eşit aralıklı
çiziliyor; yatay konum talep oranını göstermez (1 ile 2 arası, 0,1 ile 0,15
arası kadar yer kaplar).

## L-11 [MAJOR] — XGBoost öznitelik-önemi ekseni yanlış büyüklüğü adlandırıyordu

`nb04-fig02`'nin orta paneli "Share of total gain (sums to 1)" diyordu. Yanlış.
Diskteki modelle doğrulandı:

    feature_importances_ == get_score('gain')       -> True
    feature_importances_ == get_score('total_gain') -> False

XGBoost'un sklearn sarmalayıcısı `importance_type='gain'` raporluyor; bu, o
özniteliği kullanan bölünmelerin kazançlarının **ortalaması**, sonra 1'e
normalize ediliyor. Toplam kazancın payı DEĞİL. Etiket
`Average gain per split (normalised)` olarak düzeltildi.

**Not:** Bu etiketi bu oturumda önce ben "Share of total gain" yapmıştım
(denetim bulgusuna dayanarak); eski hâli "Average gain per split" daha
doğruydu, yalnız "normalise edilmiş" ibaresi eksikti.

**LaTeX'te yapılacak:** Bölüm 6'da bu şekli anlatan metin üç panelin ÜÇ FARKLI
büyüklük gösterdiğini söylemeli — paneller arası çubuk uzunluğu
karşılaştırılamaz. Ayrıca LightGBM paneli için doğru sayı: model toplam **30**
bölünme yapıyor (tek öznitelikte en fazla 8), dört öznitelik hiç kullanılmıyor.
