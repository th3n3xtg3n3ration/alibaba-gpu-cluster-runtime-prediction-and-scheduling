# Uçtan Uca Denetim Dökümü

**Depo:** `alibaba-gpu-runtime-prediction-and-scheduling`
**Tarih:** 31 Ağustos 2026
**Kapsam:** 331 dosya (venv, .git, __pycache__ hariç), her dizin tek tek gezildi
**Kural:** Hiçbir satır "muhtemelen doğrudur" diye geçilmedi. Her "DOĞRULANDI" etiketi, çalıştırılmış bir komutun çıktısına dayanır.

---

## HÜKÜM

**Çalışan bilimsel kod doğru. Depoyu çevreleyen belgeler yanlış.**

Kaynak kod, veri hattı, bölme mantığı, metrikler ve simülatör uçtan uca doğrulandı ve savunulabilir. Ancak deponun **dışa dönük yüzü** hâlâ sızıntılı sonuçları ilan ediyor: README ana sayfasında manşet olarak, altı HTML raporunda, iki bağlam dosyasında. Bir hakem depoyu açtığında ilk gördüğü sayı, düzeltmek için uğraştığımız yanlış sayıdır.

| Katman | Durum |
|---|---|
| Ham veri | DOĞRULANDI, bozulma yok |
| Veri hattı (`src/`) | DOĞRULANDI, 6 düzeltme yapıldı |
| Bölme / sızıntı | DOĞRULANDI, guard'lı, test edilmiş |
| Metrikler | DOĞRULANDI |
| Simülatör | DOĞRULANDI, korumalı |
| Testler | 26/26, duyarlılığı kanıtlandı |
| Defterler | 14/14 çalışmış, 1 model/metrik kopması var |
| **README + raporlar + bağlam dosyaları** | **ESKİ SIZINTILI SAYILAR** |
| **Ortam sabitlemesi** | **YETERSİZ** |
| **CI** | **YOK** |

---

## 1. HAM VERİ — DOĞRULANDI

`data/alibaba_cluster_trace/pai_job_no_estimate_100K.csv`

```
sha256          : dc4de3214e15b7c758b6485aaa1766c0...
boyut           : 100.000 satır × 8 sütun
sütunlar        : job_id, num_inst, submit_time, num_cpu, num_gpu, gpu_type, duration, user
NaN içeren sütun: YOK
duration <= 0   : 0 kayıt
num_gpu  <= 0   : 17.816 kayıt  (filtrelenir)
job_id benzersiz: EVET
filtre sonrası  : 82.184  →  pipeline çıktısı 82.184  ✓ birebir
```

Kaynak `data/README.md`'de belgelenmiş: Alibaba PAI GPU Trace (2022), NSDI '22. Erişilebilir bağlantı mevcut.

## 2. İŞLENMİŞ VERİ — DOĞRULANDI

| Dosya | Durum |
|---|---|
| `100k_job_with_utilization_full.csv` | **Kullanılan dosya.** Mevcut kodla sıfırdan yeniden üretildi, 14 sütunun tamamı `atol=1e-9` içinde özdeş |
| `100k_job_with_utilization.csv` | Git'te izlenmiyor. İçerik istatistikleri aynı (82.184 satır, gpu_demand ort. 0,6802, sıfır yok), yalnız **sütun sırası** farklı. Artık dosya, silinebilir |

`gpu_demand` doğrulaması: min 0,010 | ortalama 0,6802 | sıfır 0 | 43.176 kesirli | dtype float64.
Kronolojik sıralama: monotonik artan, geriye adım 0.

## 3. `src/` — SATIR SATIR OKUNDU

| Dosya | Satır | Durum | Bulgu |
|---|---:|---|---|
| `feature_engineering.py` | 527 | OKUNDU | 2 düzeltme: kronolojik sıralama garantisi (`:444`), bölme guard'ı (`:508`), shuffle uyarısı (`:480`) |
| `tuning.py` | 1544 | OKUNDU | 3 düzeltme: `seed_everything` (`:67-88`), çağrılar (`:1458,:1516,:1552`), ölü importlar |
| `data_loading.py` | 274 | OKUNDU | Temiz. `PathsConfig`, 4 yükleyici, hata mesajları net |
| `config_utils.py` | 152 | OKUNDU | Temiz |
| `visualization.py` | 267 | OKUNDU | **Ölü modül** — 0 çağrı, tezin hiçbir şeklini üretmiyor |
| `analysis/workload_analysis.py` | 287 | OKUNDU | 1 düzeltme: `int()` kırpması → `float()` (`:149`) |
| `simulation/multi_node_simulator.py` | 399 | OKUNDU | 1 düzeltme: `_finite()` NaN guard'ı (`:58`), `_gpu_request` (`:42`) |
| `simulation/scheduler_simulator.py` | 255 | OKUNDU | Tez sonuçlarında kullanılmıyor, yalnız testlerde |
| `models/evaluation.py` | 85 | OKUNDU | Temiz. MAPE 0-100 ölçeğinde, sıfır koruması var |
| `models/dl_runtime_predictor.py` | 227 | OKUNDU | 3 mimari sağlam. Xavier/orthogonal/Kaiming başlatma doğru |
| `models/lgb_runtime_predictor.py` | 204 | OKUNDU | **Ölü sınıf** — 0 defterde kullanılıyor |
| `models/xgb_runtime_predictor.py` | 175 | OKUNDU | **Ölü sınıf** |
| `models/rf_runtime_predictor.py` | 136 | OKUNDU | **Ölü sınıf** |
| `__init__.py` × 4 | 207 | OKUNDU | `src/__init__.py` `visualization`'ı yanıltıcı biçimde tanıtıyor |

**Statik analiz:** `pyflakes src/ scripts/ tests/` → **0 bulgu** (13 ölü import kaldırıldı).
**Derleme:** `compileall` → tüm modüller derleniyor.
**İçe aktarma:** 9 modülün tamamı hatasız.

## 4. SIZINTI DENETİMİ — 9 KONTROLÜN TAMAMI DOĞRULANDI

| # | Kontrol | Yöntem | Sonuç |
|---|---|---|---|
| 1 | Dış bölme kronolojik | `arrival_sec` aralığı | Eğitim `[0..524.743]`, test `[524.788..661.889]`, **0 örtüşme** |
| 2 | İç CV zamansal | `_make_cv` okundu | `TimeSeriesSplit`, `KFold(shuffle=True)` değil |
| 3 | Erken durdurma bölmesi | `XGBRegressorCV`/`LGBMRegressorCV` | Kronolojik son %15 |
| 4 | Final refit | `finalize_ml_model` | XGB/LGBM son %10 eval; RF tam veri |
| 5 | One-hot kodlayıcı | Kod okundu | Yalnız eğitimde `fit`, `handle_unknown="ignore"` |
| 6 | DL ölçekleyici | `prepare_dl_datasets` | `scaler_x`/`scaler_y` yalnız eğitimde `fit` |
| 7 | DL test penceresi | Kod okundu | Önek eğitimden, hedefler gerçek test satırlarından |
| 8 | Aramada test kullanımı | `run_dl_*search` | `test_dataset` alınıyor, **hiç kullanılmıyor**; ölü `test_loader` kaldırıldı |
| 9 | Kimlik benzersizliği | `job_id` | 82.184/82.184 benzersiz |

**Sıralama bağımsızlığı kanıtı:** CSV diskte kasten karıştırıldı, pipeline çalıştırıldı → RF R² = −0,6728515034, checkpoint ile bit-birebir.

## 5. SONUÇLARIN BÜTÜNLÜĞÜ

### 5.1 Ağaç modelleri — 7/7 DOĞRULANDI

Modeller diskten yüklendi, bağımsız hesaplanmış kronolojik bölmeyle yeniden puanlandı. 5 metriğin tamamı `< 1e-9`.

| Deney | MAE | MdAE | RMSE | R² |
|---|---:|---:|---:|---:|
| exp_a_lgbm | 7077 | 4197 | 15203 | 0,0459 |
| exp_a_xgb | 7456 | 4881 | 15271 | 0,0374 |
| exp_a_rf | 15237 | 13845 | 20131 | −0,6729 |
| exp_b_lgbm_nat | **5697** | **2684** | **13543** | **0,2429** |
| exp_b_rf_oh | 6292 | 2883 | 14329 | 0,1525 |
| exp_b_xgb_oh | 6642 | 3432 | 14297 | 0,1562 |
| exp_b_lgbm_oh | 6640 | 3414 | 14619 | 0,1178 |

### 5.2 DL modelleri — 11/12 DOĞRULANDI, 1 KOPMUŞ

```
exp_c_cnn    checkpoint MAE 6480,8  |  diskteki model 7378,7  |  sapma 1,4e-01  <<< KOPMUŞ
diğer 11                                                      |  sapma < 4e-07  ✓
```

Kök neden: eğitim hücresinin `if ckpt:` dalı model değişkenini atamıyordu, kaydetme hücresi ise koşulsuz yazıyordu. `.json` ilk eğitimin metriklerini dondurdu, `.pth` son eğitimin ağırlıklarını aldı. **Düzeltildi** (76 hücre, EN+TR), ama mevcut kopma yalnız yeniden eğitimle onarılır.

### 5.3 Simülatör — DOĞRULANDI

```
girdi 16.437 iş  →  çıktı 16.437 sonuç  |  düşen 0
negatif bekleme  : 0
slowdown < 1     : 0
_finite() gerçek veride değiştirdiği satır: 0
```

## 6. DEFTERLER — 14/14

| Defter | Hücre | Hata | Durum |
|---|---:|---:|---|
| 00–03 (EN+TR, 8 adet) | 6–13 | 0 | Çalışmış. EN/TR kodu **karakter karakter özdeş** |
| 04 (EN+TR) | 61 | 0 | Çalışmış. 61 hücreden 7'si yalnız `print` etiketiyle ayrışıyor |
| 05 × 2 (EN+TR, 4 adet) | 21 | 0 | Çalışmış |

Çalışmamış hücre yok, hata çıktısı yok. NB04'ün son geçişi checkpoint'ten okumuş (çekirdek yeniden başlatılmamış).

Küme yapılandırması doğrulandı: 32-GPU = `n_high=2, n_mid=8, n_cpu=3` → 2×8 + 8×2 = **32**. 256-GPU = `n_high=16, n_mid=64, n_cpu=24` → 16×8 + 64×2 = **256**.

## 7. TESTLER — 26/26, DUYARLILIĞI KANITLANDI

| Dosya | Test | Durum |
|---|---:|---|
| `test_evaluation.py` | 3 | OKUNDU |
| `test_feature_engineering.py` | 3 | OKUNDU |
| `test_simulation.py` | 4 | OKUNDU — GPU kısıtını sınamıyordu |
| `test_tuning.py` | 2 | OKUNDU |
| `test_config_utils.py` | 3 | OKUNDU |
| `test_regression_guards.py` | **11** | **YENİ** |

**Duyarlılık deneyi** — iki hata geri kondu:

```
                        eski paket    yeni guard'lar
astype(int) + num_gpu    15/15 GEÇTİ   5 TEST DÜŞTÜ
```

## 8. YAPILANDIRMA

| Dosya | Durum |
|---|---|
| `configs/paths.yaml` | OKUNDU, temiz, tüm yollar geçerli |
| `configs/models.yaml` | OKUNDU. `tuning:` bloğu kullanılıyor; `models:` bloğu **defterlerden 0 referans** |
| `pyrightconfig.json` | OKUNDU. `results/`, `data/`, `reports/` hariç tutulmuş; makul |
| `.gitignore` | OKUNDU. `__pycache__`, `.pytest_cache` kapsanmış |

## 9. EKSİKLER — DÜZELTİLMESİ GEREKENLER

### 9.1 README.md dışa dönük yüzde sızıntılı sonucu ilan ediyor — **EN YÜKSEK ÖNCELİK**

```
satır 115: | **XGBoost (One-Hot)** | **3,389.3** | **11,375.2** | **0.51** | **Optimal Performer** |
satır 116: | **LightGBM (Native)** |   4,105.8   |   12,147.2   |   0.44   | High Performance     |
satır 139: | **SJF-Pred-XGB** | 318,257 | **2.25x (Observed Best)** | 4.13x |
```

Bu tablo sızıntılı bölmenin ürünüdür. Doğru değerler: XGBoost-OH R² **0,1562**, LightGBM-Native R² **0,2429**. Satır 139'daki 2,25x hızlanma ise GPU kısıtı hiç uygulanmayan simülatörden gelir.

### 9.2 Bayat rapor çıktıları

| Dosya | Tarih | Eski sayı |
|---|---|---:|
| `reports/html/en/04_...html` | 12 May | 11 |
| `reports/html/tr/04_...html` | 12 May | 11 |
| `reports/html/en/05_...32/256.html` | 17 Ağu | 2+2 |
| `reports/html/tr/05_...32/256.html` | 17 Ağu | 2+2 |
| `reports/pdf/` (10 dosya) | 12 May | tarandı, aynı içerik |

### 9.3 Bağlam dosyaları

```
.github/context/results-summary.md:34  | **XGBoost-OH** | **3,389** | **11,375** | **0.509** | ...
.github/context/results-summary.md:39  **Winner (exp_b):** XGBoost-OH — **best model overall**
.github/context/experiments-map.md     1 eski sayı
```

### 9.4 Ortam sabitlemesi — somut kanıtlı

```
Modelleri eğiten : scikit-learn 1.8.0
CLI'daki sürüm   : scikit-learn 1.7.2     → InconsistentVersionWarning
environment.yaml : python=3.10            → gerçek 3.11.6
requirements.txt : tüm bağımlılıklar ">=", tek pin yok
```

Kurulu sürümler: numpy 1.26.3, pandas 2.2.3, scipy 1.16.3, sklearn 1.7.2, lightgbm 4.6.0, xgboost 3.1.1, torch 2.5.1, joblib 1.4.2.

### 9.5 CI yok

`.github/workflows/` dizini yok. 26 test hiçbir zaman otomatik çalışmıyor. `.github/` altında yalnız yapay zekâ ajan yapılandırmaları var (17 dosya).

### 9.6 Depo hijyeni

```
142.809.921 B  results/models/rf_numeric.joblib   ← git'te izleniyor
 13.298.465 B  results/models/rf_categorical.joblib
  9.478.516 B  data/processed/100k_job_with_utilization_full.csv
```

142 MB'lık bir pickle git geçmişinde. Ayrıca `results/_backup_pre_retrain/` (32 dosya) sızıntı öncesi artefaktları taşıyor.

### 9.7 Tek koşuluk sonuçlar ve MAPE

19 deneyin her biri tek koşu, varyans raporlanmıyor. MAPE değerleri %1.386–%10.565 aralığında; hesap doğru ama metrik bu veri için anlamsız.

### 9.8 Deney B'de eksik konfigürasyon

LightGBM'e yerel kategorik desteği verilmiş, XGBoost'a verilmemiş — oysa XGBoost 3.1.1 `enable_categorical` destekliyor. Gösterge ölçüm: XGBoost-native MAE 6905, R² 0,1269 (one-hot sürümünden de kötü). Karşılaştırmanın adil olması için bu konfigürasyon aynı arama prosedürüyle eklenmelidir.

## 10. ÖLÜ KOD ENVANTERİ (~800 satır)

| Öğe | Satır | Kullanım |
|---|---:|---|
| `src/visualization.py` | 267 | 0 çağrı |
| `LightGBMPredictor`, `XGBPredictor`, `RandomForestPredictor` | 515 | 0 defter |
| `ClusterSimulator` | ~100 | Yalnız testler |
| `configs/models.yaml` → `models:` | 40 | 0 referans |
| `EarlyStopping.save_checkpoint` | 4 | Adına rağmen kaydetmiyor |
| `time_window`, `gpu_capacity` | 2 | Belgelenmiş "reserved" |

## 11. HATA OLMAYAN TASARIM TERCİHLERİ

| Konu | Değerlendirme |
|---|---|
| DL dizilerinde `seq_len-1` pencere örtüşmesi | Hedef sızıntısı yok; embargo daha katı olurdu |
| `TimeSeriesSplit(n_splits=3)` ilk katı verinin %25'i | Zamansal CV'nin doğal bedeli, belgelenmiş |
| XGB/LGBM %90, RF %100 ile final eğitim | Erken durdurma gereği, asimetrik ama meşru |
| FINISH/ARRIVAL eşitliği | İncelendi, `self.time` her iki sırada aynı |
| `utilization_history` zaman-ağırlıksız | Yanlı olurdu ama defterlerde kullanılmıyor |
| `.pth` state_dict yerine pickle | Çalışıyor; sınıf yolu taşınırsa açılmaz |
| Backfilling yok (HOL blocking) | Meşru model, metinde belirtilmeli |
| `LOAD_FACTOR = 0.1` | Meşru varsayım, metinde belirtilmeli |

## 12. YAPILACAKLAR SIRASI

1. **README.md tablolarını düzelt** — dışa dönük en görünür yanlış
2. `.github/context/results-summary.md` ve `experiments-map.md` güncelle
3. DL'leri yeniden eğit (12 checkpoint + 12 `.pth` sil, temiz çekirdek)
4. XGBoost-native'i Deney B'ye ekle
5. NB05 × 2 ve export'u yenile
6. HTML/PDF raporları yeniden üret
7. `requirements-lock.txt` üret
8. MAPE kararı
9. Tez LaTeX sayıları
10. İsteğe bağlı: CI ekle, 142 MB pickle'ı git'ten çıkar, `_backup_pre_retrain/` temizle

---

## DOĞRULAMA GÜNLÜĞÜ

| İddia | Nasıl doğrulandı |
|---|---|
| Ham veri bozulmamış | sha256 + satır/sütun + NaN + benzersizlik taraması |
| Filtre 82.184 veriyor | Ham veride koşul sayıldı, pipeline çıktısıyla karşılaştırıldı |
| NB00 çıktısı değişmedi | Pipeline sıfırdan yeniden üretildi, 14 sütun karşılaştırıldı |
| 7 ağaç modeli tutarlı | Diskten yüklenip bağımsız bölmeyle yeniden puanlandı |
| Sıralama girdiden bağımsız | CSV karıştırılıp pipeline yeniden çalıştırıldı |
| Sızıntı yok | `arrival_sec` min/max karşılaştırması |
| Eski tohumlama üretilemez | Eski çağrı sırası taklit edilip iki kez çalıştırıldı |
| Yeni tohumlama üretilebilir | `finalize_dl_model` + `run_dl_randomsearch` iki kez |
| 12 DL modelinin 11'i tutarlı | Her biri yüklenip checkpoint'iyle karşılaştırıldı |
| Simülatör tüm işleri planlıyor | 16.437 iş gerçek veriyle koşuldu |
| NaN guard'ı koruyor | Guard'sız `allocate(NaN)` senaryosu |
| Şekil sayısı guard'ı çalışıyor | NB04'e sahte şekil eklenip `cmp` ile kontrol |
| Eski testler kör | İki hata geri konup iki paket ayrı çalıştırıldı |
| EN/TR aynı mantık | Yorum/boşluk temizlenip sha256 |
| sklearn sürüm uyuşmazlığı | `joblib.load` uyarıları yakalandı |
| README/raporlar bayat | 6 HTML + 10 PDF + 5 markdown tarandı |
| Küme kapasiteleri doğru | `n_high/n_mid/n_cpu` çarpımları |
| XGBoost native destekliyor | `get_params()` kontrolü + gerçek eğitim |

---

*Bu döküm, denetim sırasında çalıştırılan komutların çıktılarına dayanır. Ölçülmemiş hiçbir iddia içermez.*
