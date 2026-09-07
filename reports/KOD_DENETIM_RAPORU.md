# Kod Denetim Raporu

**Proje:** Multi-Paradigm Runtime Prediction and Heterogeneous Scheduling Optimization in Large-Scale GPU Clusters
**Depo:** `alibaba-gpu-runtime-prediction-and-scheduling`
**Denetim tarihi:** 31 Ağustos 2026
**Kapsam:** 8.521 satır Python (21 modül, 6 betik/test dosyası), 14 Jupyter defteri, 3 yapılandırma dosyası
**Yöntem:** Satır satır okuma + gerçek veriyle ampirik doğrulama. Hiçbir iddia yalnız akıl yürütmeyle kabul edilmedi; her biri çalıştırılarak kanıtlandı.

---

## 0. Özet hüküm

**Depo şu anda güvenilir bilimsel yöntem uyguluyor, ancak "dünya standardının üzerinde" değil.** Aradaki farkı üç kalem kapatıyor: tek koşuluk DL sonuçları, sürüm sabitlemesi olmayan ortam tanımı ve `exp_c_cnn` için çözülmemiş model/metrik uyuşmazlığı. Kritik sınıftaki tüm hatalar giderildi ve regresyon testleriyle kilitlendi.

| Kategori | Durum |
|---|---|
| Veri sızıntısı | Temiz, guard'lı, test edilmiş |
| Kronolojik bütünlük | Garanti altında, ihlalde çöküyor |
| Kaynak muhasebesi (simülatör) | Doğru, NaN'a karşı korumalı |
| Metrik hesabı | Doğru |
| Üretilebilirlik (ağaç modelleri) | Bit-birebir |
| Üretilebilirlik (DL) | Düzeltildi, yeniden eğitim bekliyor |
| Ortam sabitlemesi | **Yetersiz** |
| İstatistiksel güç (tek koşu) | **Yetersiz** |
| Ölü kod | ~800 satır, zararsız ama yanıltıcı |

---

## 1. Denetlenen dosyaların tam dökümü

### 1.1 `src/` — çekirdek kütüphane

| Dosya | Satır | Okundu | Sonuç |
|---|---:|---|---|
| `feature_engineering.py` | 527 | Tamamı | 2 düzeltme yapıldı |
| `tuning.py` | 1544 | Tamamı | 3 düzeltme yapıldı |
| `data_loading.py` | 274 | Tamamı | Temiz |
| `config_utils.py` | 152 | Tamamı | Temiz |
| `visualization.py` | 267 | Tamamı | Ölü modül |
| `analysis/workload_analysis.py` | 287 | Tamamı | 1 düzeltme yapıldı |
| `simulation/multi_node_simulator.py` | 399 | Tamamı | 1 düzeltme yapıldı |
| `simulation/scheduler_simulator.py` | 255 | Tamamı | Tezde kullanılmıyor |
| `models/evaluation.py` | 85 | Tamamı | Temiz |
| `models/dl_runtime_predictor.py` | 227 | Tamamı | Mimariler sağlam |
| `models/lgb_runtime_predictor.py` | 204 | Tamamı | Ölü sınıf |
| `models/xgb_runtime_predictor.py` | 175 | Tamamı | Ölü sınıf |
| `models/rf_runtime_predictor.py` | 136 | Tamamı | Ölü sınıf |
| `__init__.py` × 4 | 207 | Tamamı | Temiz |

### 1.2 `scripts/`, `tests/`, `configs/`

| Dosya | Satır | Sonuç |
|---|---:|---|
| `scripts/export_thesis_results.py` | 331 | 1 düzeltme yapıldı |
| `scripts/run_all_experiments.sh` | 101 | Temiz |
| `tests/test_*.py` (5 dosya) | 292 | Kritik boşluk vardı, kapatıldı |
| `tests/test_regression_guards.py` | 168 | **Yeni** |
| `configs/models.yaml` | 128 | `models:` bloğu kullanılmıyor |
| `configs/paths.yaml` | 33 | Temiz |

### 1.3 Defterler

14 defterin tamamı: sıfır hata, çalışmamış hücre yok, tümü baştan sona sıralı çalışmış.
EN/TR eşleştirmesi: 00–03 defterlerinin kodu **karakter karakter özdeş**; NB04'te 61 kod hücresinden 7'si yalnızca `print` etiketiyle ayrışıyor, mantık aynı.

---

## 2. Kritik bulgular ve düzeltmeler

### B1 — DL sonuçları üretilebilir değildi
**Yer:** `src/tuning.py:1315` (eski) → `seed_everything()` `tuning.py:67-88`, çağrılar `1458`, `1516`, `1552`

`torch.manual_seed(42)`, `train_dl_model` içinde çağrılıyordu; oysa PyTorch başlangıç ağırlıklarını **model kurulduğu anda** global RNG'den çeker. Kurulum bu çağrıdan önce olduğu için ağırlıklar bir önceki denemenin bıraktığı RNG durumuna bağlıydı.

Kanıt (eski sıralama birebir taklit edildi, aynı çağrı iki kez):
```
koşu 1: MAE 2.334  R² 0.355
koşu 2: MAE 2.886  R² 0.121   → üretilemez
```
Düzeltme sonrası, `finalize_dl_model` ve `run_dl_randomsearch` iki kez:
```
her iki koşu: MAE 2.4464638869, R² 0.2558581123  → bit-birebir
en iyi hiperparametre seti: iki koşuda da aynı
```

Yan tuzak: `seed_everything` global `random`'ı sıfırladığı için hiperparametre örneklemesi de dondu. Ayrı bir `random.Random(DL_SEED)` akışı verildi; doğrulandı, 6 denemede 3 farklı konfigürasyon örnekleniyor.

### B2 — Kronolojik sıralama garanti değildi
**Yer:** `src/feature_engineering.py:444` (sıralama), `:480` (uyarı), `:508` (guard)

`shuffle=False` yalnızca ilk %80 satırı alır. "Kronolojik" olması veri çerçevesinin zaten sıralı gelmesine bağlıydı; farklı sıralı bir kopya sessizce sızıntılı bölme üretirdi.

Kanıt: CSV diskte **kasten karıştırıldı**, pipeline çalıştırıldı.
```
karıştırılmış girdi → RF R² = -0.6728515034
checkpoint          → RF R² = -0.6728515034   → bit-birebir
```
Ek olarak `shuffle=True` artık `UserWarning` atıyor ve bölme sonrası ihlal `ValueError` ile çöküyor.

### B3 — Simülatörde NaN kaynak muhasebesini devre dışı bırakabiliyordu
**Yer:** `src/simulation/multi_node_simulator.py:58` (`_finite`), `:42` (`_gpu_request`)

NaN her `can_fit` karşılaştırmasını `False` yapar, dolayısıyla makine "sığıyor" der; `gpu_used` bir kez NaN olunca düğüm sınırsız iş kabul eder.
```
guard'sız: allocate(NaN) → gpu_used = nan → 1-GPU düğüm 5-GPU işi kabul ediyor
guard'lı : 3 iş tamamlandı, 1-GPU işler serileşti, muhasebe sonlu
```
Gerçek veride NaN yok (`_finite` 0 satırı değiştiriyor), yani latent bir açıktı.

### B4 — Export betiği yanlış tez şeklini sessizce yazabiliyordu
**Yer:** `scripts/export_thesis_results.py:116` (`EXPECTED_FIGURE_COUNT`), `:206`

`_sync_thesis_figures` yalnızca **eksik** konumu kontrol ediyordu. Bir deftere ortadan şekil eklenirse sonraki tüm konumlar kayar, hepsi "mevcut" görünür ve betik yanlış görselleri doğru tez dosya adlarıyla üzerine yazardı.

Kanıt: NB04'e sahte bir şekil eklendi.
```
⚠️ NB04: tez şekilleri GÜNCELLENMEDİ — 6 şekil üretti, beklenen 5
cmp sonucu: nb04-fig02-feature-importance.png DEĞİŞMEDİ
```

### B5 — `int()` kırpması, tezi bozan hatanın aynı ailesi
**Yer:** `src/analysis/workload_analysis.py:149`

`int(np.max(gpu_demand))`. Şu an maksimum tam 8,0 olduğu için zarar vermiyor, kesirli maksimumda kırpardı. `float`'a çevrildi; NB02 çıktısı (0,68 / 8,00) değişmedi.

---

## 3. Çözülmemiş bulgu: `exp_c_cnn` model/metrik kopması

**Bu, yeniden eğitim gerektiren tek kalemdir.**

12 DL modelinin tamamı checkpoint'leriyle karşılaştırıldı:

| Deney | Checkpoint MAE | Diskteki model MAE | Sapma |
|---|---:|---:|---:|
| **exp_c_cnn** | **6480,8** | **7378,7** | **1,4e-01** |
| exp_c_lstm | 7185,5 | 7185,5 | 1,5e-07 |
| exp_c_hybrid | 6697,6 | 6697,6 | 1,8e-08 |
| exp_d_cnn | 6295,7 | 6295,7 | 4,4e-09 |
| exp_d_lstm | 6320,4 | 6320,4 | 3,6e-07 |
| exp_d_hybrid | 6147,5 | 6147,5 | 5,8e-09 |
| exp_e_cnn | 6170,5 | 6170,5 | 4,4e-09 |
| exp_e_lstm | 6957,5 | 6957,5 | 1,6e-07 |
| exp_e_hybrid | 6791,2 | 6791,2 | 2,4e-09 |
| exp_f_cnn | 6734,2 | 6734,2 | 1,9e-09 |
| exp_f_lstm | 6509,6 | 6509,6 | 2,0e-07 |
| exp_f_hybrid | 6956,8 | 6956,8 | 4,7e-08 |

**Kök neden** — `notebooks/en/04_...ipynb` hücre 32:
```python
ckpt = load_checkpoint('exp_c_cnn')
if ckpt:
    cnn_metrics_num = ckpt['metrics']   # metrikler eski koşudan
    # cnn_model_num BU DALDA HİÇ ATANMIYOR
else:
    cnn_model_num, cnn_metrics_num = finalize_dl_model(...)
```
Hücre 33 ise `if 'cnn_model_num' in locals()` deyip koşulsuz kaydediyor. Checkpoint varken değişken önceki geçişten kalır, guard "var" der ve diske yazar. Sonuç: **`.json` ilk eğitimin metriklerini dondurur, `.pth` son eğitimin ağırlıklarını alır.**

Bu desen 19 deneyin **tamamında** vardır, yalnız DL'de değil. Sadece CNN'de patlamasının sebebi B1'dir: ağaç modelleri tohumludur, yeniden eğitim özdeş model üretir, dolayısıyla `.joblib` üzerine yazılsa bile fark oluşmaz. Tohumsuz DL'de iki koşu farklı model üretir. Bu iki bulgu aynı kökün iki dalıdır.

**Etkisi:** NB05 `cnn_numeric.pth`'i yükler. Tablo 6.1'de Exp C CNN için yazan MAE 6.480, zamanlayıcıdaki SJF-CNN politikasını besleyen modelin MAE'si değildir. Hangisinin doğru olduğu sorusunun cevabı yoktur; tutarlı bir çift ancak yeniden eğitmekle elde edilir.

**Düzeltme uygulandı.** Her eğitim hücresinin `if ckpt:` dalına model değişkenini boşaltan bir atama eklendi, kaydetme koruyucusu `'<var>' in locals()` yerine `locals().get('<var>') is not None` oldu. EN ve TR defterlerinde 19'ar eğitim ve 19'ar kaydetme hücresi, toplam 76 hücre. İzole simülasyonla doğrulandı:

```
ESKİ guard: checkpoint=metrics_A  model_file=weights_B  -> KOPMUŞ
YENİ guard: checkpoint=metrics_A  model_file=weights_A  -> TUTARLI
```

Düzeltme gelecekteki kopmayı engeller; mevcut `exp_c_cnn` kopmasını onarmaz. Onun için yeniden eğitim gerekir.

---

## 4. Değişmediği kanıtlanan çıktılar

Yapılan tüm düzeltmelerden sonra yayınlanmış hiçbir sayı değişmedi.

**NB00 ürünü** — mevcut kodla sıfırdan yeniden üretildi:
```
82.184 satır, 14 sütunun tamamı diskteki CSV ile atol=1e-9 içinde özdeş
```

**7 ağaç modeli** — bağımsız hesaplanmış kronolojik bölmeyle yeniden puanlandı, 5 metriğin tamamı (MAE/RMSE/R²/MdAE/MAPE) `< 1e-9`:

| Deney | MAE | R² |
|---|---:|---:|
| exp_a_rf | 15236,70 | −0,6729 |
| exp_a_lgbm | 7076,86 | 0,0459 |
| exp_a_xgb | 7455,69 | 0,0374 |
| exp_b_lgbm_nat | 5697,39 | **0,2429** |
| exp_b_rf_oh | 6292,28 | 0,1525 |
| exp_b_xgb_oh | 6642,39 | 0,1562 |
| exp_b_lgbm_oh | 6640,01 | 0,1178 |

**Tez şekilleri** — export dizini ile sha256 eşleşmesi doğrulandı (4 örnek, hepsi MATCH).

**Simülatör girdileri** — `_finite()` gerçek veride `num_cpu` ve `gpu_demand` sütunlarında **0 satır** değiştiriyor.

---

## 5. Sızıntı denetimi (ayrıntılı)

| Kontrol | Yöntem | Sonuç |
|---|---|---|
| Dış bölme kronolojik mi | `arrival_sec` aralık karşılaştırması | Eğitim `[0 .. 524.743]`, test `[524.788 .. 661.889]`, **0 örtüşme** |
| İç CV zamansal mı | `_make_cv` okundu | `TimeSeriesSplit`, `KFold(shuffle=True)` değil |
| Erken durdurma bölmesi | `XGBRegressorCV.fit` / `LGBMRegressorCV.fit` | Kronolojik son %15, sızıntı yok |
| Final refit | `finalize_ml_model` | XGB/LGBM son %10'u eval için ayırıyor; RF tam veri |
| One-hot kodlayıcı | `prepare_features_for_model` | Yalnız eğitimde `fit`, `handle_unknown="ignore"` |
| DL ölçekleyici | `prepare_dl_datasets` | `scaler_x`/`scaler_y` yalnız eğitimde `fit` |
| DL test penceresi | Kod okundu | Test önüne eğitimden son `seq_len-1` satır ekleniyor; hedefler gerçek test satırlarından, sızıntı yok |
| Hiperparametre aramada test kullanımı | `run_dl_randomsearch`/`gridsearch` | `test_dataset` alınıyor ama **hiç kullanılmıyor**; ölü `test_loader` satırı kaldırıldı |
| Test kimliği | `job_id` benzersizliği | 82.184/82.184 benzersiz |

---

## 6. Regresyon testleri: duyarlılık kanıtı

`tests/test_regression_guards.py` (11 test) eklendi. Değerini ölçmek için iki hata geri kondu:

```
                        eski paket    yeni guard'lar
astype(int) + num_gpu    15/15 GEÇTİ   5 TEST DÜŞTÜ
```

Eski paket hatalara kördü. `test_multinode_simulator_uses_additional_machine` iki 1-GPU işi iki 1-GPU makineye koyuyordu; ikisi de sığdığı için GPU sayılsa da sayılmasa da geçiyordu. Yeni test tek makineye iki iş koyar, serileşmek zorundalar.

Geri alma sonrası: **26/26 OK**.

---

## 7. Statik ve yapısal denetim

```
pyflakes src/ scripts/ tests/   → temiz (13 ölü import kaldırıldı)
python -m compileall             → tüm modüller derleniyor
tüm modüller import edildi       → hata yok
unittest discover tests          → 26/26 OK
```

---

## 8. Yayın standardını karşılamayan kalemler

### 8.1 Ortam sabitlemesi — **somut kanıtlı sorun**

```
Modelleri eğiten sürüm : scikit-learn 1.8.0
CLI'daki sürüm         : scikit-learn 1.7.2
environment.yaml       : python=3.10     |  gerçek: 3.11.6
requirements.txt       : tüm bağımlılıklar ">=", üst sınır ve pin yok
```
`.joblib` dosyaları yüklenirken `InconsistentVersionWarning` üretiyor. Jupyter çekirdeği ile terminal farklı ortamlar. Bir hakem bunu artefakt reddi sebebi sayabilir. Çözüm: `pip freeze` çıktısıyla tam pinli bir `requirements-lock.txt`.

### 8.2 Tek koşuluk sonuçlar
19 deneyin her biri tek koşu. Üst düzey mekânlar ≥3 tohumun ortalama ± standart sapmasını ister. Tohumlama artık düzeldiği için bu maliyetsiz yapılabilir.

### 8.3 MAPE
Tablolarda %1.386 ile %10.565 arası değerler var. Hesap doğru; sorun metrikte: 4 saniyelik işler paydada. Bir hakemin ilk soracağı şey budur. MAPE sütununu kaldırıp MdAE'yi öne çıkarmak, ya da MASE eklemek gerekir.

### 8.4 Belirtilmesi gereken modelleme varsayımları
- **Backfilling yok** (head-of-line blocking). En yüksek öncelikli iş sığmazsa hiçbir iş başlamaz.
- **`LOAD_FACTOR = 0.1`** varış aralıklarını 10 kat sıkıştırıyor.

SJF'in %60–80'lik JCT kazancının büyük kısmı bu iki varsayımdan gelir; metinde açıkça yazılmalıdır.

### 8.5 Model serileştirme
`.pth` dosyaları `state_dict` değil, komple pickle'lanmış model nesnesi saklıyor. Sınıf yolu dosyaya gömülü olduğu için `src.models.dl_runtime_predictor` taşınırsa dosyalar açılmaz.

---

## 9. Ölü kod envanteri (yaklaşık 800 satır)

| Öğe | Satır | Durum |
|---|---:|---|
| `src/visualization.py` | 267 | Hiçbir yerden çağrılmıyor; tezin hiçbir şeklini üretmiyor |
| `LightGBMPredictor` / `XGBPredictor` / `RandomForestPredictor` | 515 | 0 defterde kullanılıyor; defterler `finalize_ml_model` kullanıyor |
| `ClusterSimulator` (tek kuyruk) | ~100 | Yalnız testlerde; tez sonuçlarında değil |
| `configs/models.yaml` → `models:` bloğu | 40 | Defterlerden 0 referans |
| `EarlyStopping.save_checkpoint` | 4 | Adına rağmen hiçbir şey kaydetmiyor |
| `time_window`, `gpu_capacity` parametreleri | 2 | "Reserved, currently unused" olarak dürüstçe belgelenmiş |

Bunlar hata değildir; `src/__init__.py`'nin `visualization`'ı "Matplotlib-based plotting library" diye tanıtması yanıltıcıdır.

---

## 10. Hata olmayan, bilinçli tasarım tercihleri

| Konu | Değerlendirme |
|---|---|
| DL dizilerinde train/val arası `seq_len-1` pencere örtüşmesi | Hedef sızıntısı yok. Embargo/purge daha katı olurdu, standart uygulama bu |
| `TimeSeriesSplit(n_splits=3)` — ilk kat verinin %25'iyle eğitiliyor | Zamansal CV'nin doğal bedeli, docstring'de yazılı |
| XGB/LGBM %90, RF %100 veriyle final eğitim | Erken durdurma gereği; asimetrik ama meşru |
| Olay kuyruğunda FINISH/ARRIVAL eşitliği | İncelendi: `self.time` her iki sırada da aynı, sonuç değişmiyor |
| `utilization_history` zaman-ağırlıksız anlık görüntü ortalaması | Yanlı olurdu, ama defterlerde hiç kullanılmıyor |

---

## 11. Yapılması gerekenler (öncelik sırasıyla)

1. ~~NB04 kaydetme mantığını düzelt.~~ **Yapıldı** (bkz. Bölüm 3).
2. **DL'leri yeniden eğit.** `exp_{c,d,e,f}_*.json` (12) ve karşılık gelen 12 `.pth` silinip NB04 temiz çekirdekle çalıştırılmalı. Ağaç modelleri checkpoint'ten yüklenip hızlı geçer; bit-birebir doğrulandıkları için yeniden eğitmenin faydası yok.
3. **NB05 (32 ve 256) ve export'u yenile.**
4. **`requirements-lock.txt` üret.**
5. **MAPE kararını ver.**
6. **Tez metnindeki sayıları güncelle.**

---

## 12. Doğrulama günlüğü

Bu raporda geçen her iddia aşağıdaki komutlarla üretildi:

| İddia | Doğrulama |
|---|---|
| 7 ağaç modeli checkpoint'lerle eşleşiyor | Modeller diskten yüklendi, bağımsız bölmeyle yeniden puanlandı |
| Kronolojik bölmede sızıntı yok | `arrival_sec` min/max karşılaştırması |
| Sıralama girdi düzeninden bağımsız | CSV karıştırılıp pipeline yeniden çalıştırıldı |
| NB00 çıktısı değişmedi | Pipeline sıfırdan yeniden üretilip 14 sütun karşılaştırıldı |
| Eski tohumlama üretilemez | Eski çağrı sırası taklit edilip iki kez çalıştırıldı |
| Yeni tohumlama üretilebilir | `finalize_dl_model` ve `run_dl_randomsearch` iki kez çalıştırıldı |
| `_finite` gerçek veriyi değiştirmiyor | 82.184 satırda eleman eleman karşılaştırma |
| NaN guard'ı gerçekten koruyor | Guard'sız `allocate(NaN)` senaryosu çalıştırıldı |
| Şekil sayısı guard'ı çalışıyor | NB04'e sahte şekil eklenip `cmp` ile tez dosyası kontrol edildi |
| Eski testler hatalara kör | İki hata geri konup iki paket ayrı ayrı çalıştırıldı |
| 12 DL modelinin 11'i tutarlı | Her biri yüklenip checkpoint'iyle karşılaştırıldı |
| EN/TR defterleri aynı mantık | Yorumlar/boşluklar temizlenip sha256 karşılaştırması |
| sklearn sürüm uyuşmazlığı | `joblib.load` sırasında yakalanan uyarılar |

---

*Bu rapor, denetim sırasında çalıştırılan doğrulamaların çıktılarına dayanır. Ölçülmemiş hiçbir iddia içermez.*
