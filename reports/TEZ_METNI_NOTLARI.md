# Tez Metnine Eklenecek Notlar (henüz LaTeX'e işlenmedi)

Bu dosya, "şimdi tezi değiştirme, sadece not al" denen kararları biriktirir. LaTeX aşamasına geçildiğinde buradan işlenecek.

---

## 1. `num_inst` sınırlaması (Karar: D — kod değiştirilmeyecek, yalnız tez metnine sınırlama olarak eklenecek)

**Nereye eklenecek:** Threats to Validity / Limitations bölümü + Future Work.

**Taslak metin:**
> Bu çalışma her işi tek bir kaynak talebi (`num_gpu`, `num_cpu`) olarak modeller ve tek bir düğüme yerleştirir. Gerçek izde işlerin %21'i birden fazla paralel örnek (`num_inst`>1, ortalama 5, maksimum 512) içerir; Alibaba PAI mimarisinde bu örnekler gang-scheduling ile birden fazla makineye dağıtılabilir (Weng et al., NSDI'22). Bu basitleştirme özellikle büyük dağıtık işlerde kaynak tüketimini olduğundan düşük gösterir: `num_inst`'i dahil eden bir yeniden hesapta, işlerin yalnız %8,6'sı (ama toplam GPU-talebin %75,3'ü) tek bir düğüme sığmayacak kadar büyük çıkmaktadır. Çok-düğümlü gang-scheduling desteği bu çalışmanın kapsamı dışındadır ve gelecek çalışma olarak önerilir.

**Kanıt (gerekirse doğrulama için):** `python3 -W ignore -c "..."` ile `results/checkpoints` ve `data/` üzerinden yeniden üretilebilir; ölçüm bu oturumda yapıldı, sayılar: %8,63 sığmıyor, bu dilim toplam GPU-saniyenin %75,3'ü.

---

## 2. Kararlı-rejim (ρ≈0,7) referans noktası (Kademe 1 / Madde 3+8, kod DEĞİŞTİ)

**Durum:** Kod tarafı bitti (commit `2861d93`), tez metnine HENÜZ işlenmedi — NB04/NB05 yeniden koşulup gerçek sayılar üretilene kadar bekliyor.

**Ne yapıldı:** `run_policy`'nin küme boyutu (n_high/n_mid) artık parametre; yeni bir hücre ~752 GPU'luk (47 üst-düzey + 188 orta-düzey düğüm, mevcut 1:4/50-50 oranı korunarak) bir küme kurup izi doğal hızında (LOAD_FACTOR=1.0) yeniden oynatıyor ve ρ≈0,7'de FIFO/SRF/SJF-Oracle/SJF-KullanıcıMedyanı/SJF-ProfilMedyanı/SJF-LGBM(Kategorik)/SJF-LSTM(Kategorik) için Ortalama Bekleme/JCT/JCT-İyileşme-% raporluyor.

**Neden Madde 3 ile Madde 8 birlikte:** Test diliminin sunduğu yük (Σ gpu_talebi·çalışma_süresi / doğal-süre) ortalama ~530 eşzamanlı GPU'ya denk geliyor — mevcut 32/256-GPU kümelerinin İKİSİ de bunun çok altında, LOAD_FACTOR=1,0'da bile aşırı-doygun kalıyorlar. Bu yetersiz-boyutlandırma aynı zamanda Madde 8'in kaynağı: `cluster_load_gpu`/`cluster_load_cpu` öznitelikleri gerçek ~6.500 GPU'luk Alibaba izinden tek seferde hesaplanmış, yani 32/256 GPU'luk simüle küme bu özniteliklerin tarif ettiği yükü ASLA barındıramaz — tahmin ve karar farklı evrenlerde. ~752 GPU'luk kümede bu artık fiziksel olarak imkansız değil.

**Tez metnine eklenecek yer:** Bölüm 6 (Results) — yeni bir alt bölüm olarak "Stable-Regime Validation", Bölüm 7 Limitations'daki "Results Under Revision" notuna referans.

---

## 3. `cluster_load_cpu`/`active_job_count` CPU-only işleri saymıyordu (KARAR GÜNCELLENDİ: kod DEĞİŞTİ — `leakage-4` + `robustness-11`)

**Durum:** ~~Kod DEĞİŞMEYECEK~~ → **kod DÜZELTİLDİ** (commit `09cf225`). İlk kararda "rerun'dan hemen önce bir özellik daha değiştirmeyelim" denmişti; kullanıcı sonradan rerun'u tekrar yapmayı kabul edince düzeltme uygulandı.

**Yapılan:** `build_job_table_from_sample(include_cpu_only=True)` eklendi; sweep-line artık TÜM işler (CPU-only dahil) üzerinden hesaplanıyor, GPU işlerine kısıtlama ondan SONRA yapılıyor. Modelleme yine yalnız GPU işleriyle (82.184 satır, değişmedi). Ölçülen etki: `cluster_load_cpu` +%15,2, `active_job_count` +%22,5, `cluster_load_gpu` %0,0 (doğru — CPU işleri GPU kullanmıyor). Regresyon testi eklendi.

**Tez metnine:** Artık "sınırlama" notu GEREKMİYOR; bunun yerine Bölüm 3'te küme-yükü özniteliklerinin CPU-only işleri de kapsadığı belirtilmeli.


---

## 4. Diğer küçük/orta bulgular

Bu liste başta "kod değiştirilmeyecek, yalnız not" olarak açılmıştı. Kullanıcı sonradan rerun'u tekrar yapmayı kabul edince çoğu fiilen düzeltildi; aşağıda her maddenin güncel durumu var. **Hâlâ açık olan tek madde `reproducibility-5`.**

- **`reproducibility-5`** [HÂLÂ AÇIK — kod değişmeyecek] (DL sonuçları cihaza bağlı, MPS'te CPU'dan farklı çıkıyor): Tez metnine "sonuçlar Apple Silicon/MPS üzerinde üretildi, farklı donanımda küçük sapmalar beklenir" notu eklenecek. CPU'da yeniden eğitim yapılmayacak (pahalı, düşük değer).
- ~~`simulator-8`~~ **ÇÖZÜLDÜ** (commit `0d7d35d`): Bölüm 5.2/5.1 metni koda uyduruldu — "işler düğümler arasında bölünmez; kesirli GPU talepleri toplamsal olarak paketlenir, girişim modeli yok" diye yazıldı. Kod zaten doğruydu, yanlış olan cümleydi.
- ~~`F-C10`~~ **ÇÖZÜLDÜ** (commit `0d7d35d`): ch1 Amaç 4 ve ch5 iş-kaydı tanımından "memory" iddiası çıkarıldı, ikisi de Bölüm 5.2'deki doğru ifadeye (bellek boyutu izde veri olmadığı için devre dışı) referans verecek şekilde düzeltildi. Kod zaten doğruydu.
- ~~`modeling-11`~~ **ÇÖZÜLDÜ** (commit `a5d25ed` + `657bd87`): `grid_boundary_params()` (yalnız sayısal parametreler — bool/kategorik değerlerin "sınırı" olmaz) ve `run_gridsearch_iterative()` eklendi; 8 ML araması artık kazanan gridin ucuna düştüğünde bir tur genişletip yeniden arıyor, kaç tur döndüğünü ve hâlâ sınırda kalan parametreleri checkpoint'e yazıyor. 12 DL araması için (pahalı olduğundan yeniden arama yapılmadan) sınır durumu `run_dl_gridsearch` içinden raporlanıyor.
- ~~`figures_tables-14`~~ **KARAR GÜNCELLENDİ — kod değişti.** İlk kararda "manifest/temizlik sistemi eklenmeyecek (orantısız mühendislik)" denmişti; sonradan asıl kök nedenin tek satırlık bir eksik olduğu görüldü (`scripts/export_thesis_results.py` yeni dosyaları yazmadan önce eskilerini hiç silmiyordu) — tam manifest sistemi değil, üç satırlık bir `_clean_stale_exports()` yeterliydi. commit `5746076`'da düzeltildi, 3 regresyon testi eklendi. Artık tez metnine ayrıca not gerekmiyor (rerun sonrası export script zaten temiz çalışacak).

---

*(Bundan sonraki notlar buraya eklenecek.)*

---

## 5. Şekil altyazılarına TAŞINAN bilgiler (şekilden çıkarıldı, caption'a girmeli)

Şekiller sadeleştirilirken (yalnız birim + eksen adı kaldı) aşağıdaki bilgiler
şekil içinden çıkarıldı. **Bunlar kaybolmamalı — LaTeX altyazısında yazılmalı.**
Bir kısmı sadece stil değil, şeklin ne gösterdiğini belirleyen bilgi:

| Şekil | Altyazıda mutlaka olmalı |
|---|---|
| nb01-fig01 (runtime dağılımı) | n = 82.184 GPU işi (CPU-only ve sıfır-süreli satırlar hariç) |
| nb01-fig05 (varışlar arası süre) | Ardışık çiftlerin %21,6'sı aynı saniyede geldi, log eksende gösterilemediği için hariç; loglu bin genişliği |
| nb01-fig06 (ısı haritası) | y ekseni takvim günü değil **iz günü**; gri hücreler izin hiç ulaşmadığı saatler (son gün 15:51'de bitiyor) |
| nb02-fig03 (GPU talebi vs süre) | ~~5.000 işlik örneklem~~ → **artık tüm 82.184 iş çiziliyor**; **n şeklin lejantında yazıyor**, altyazıda tekrar GEREKMİYOR (yalnız jitter + düşük saydamlık kullanıldığı belirtilebilir) |
| nb03-fig01 (küme durumu) | Her iki panel de **gelen işin kendisini hariç tutar** (arka plan yükü) |
| nb03-fig02 (korelasyon) | Köşegen maskeli; renk skalası köşegen dışı en büyük |r|'ye göre ölçekli |
| nb04-fig01 (model karşılaştırma) | ~~yön bilgisi~~ → sembol anahtarı şekilde. **YENİ:** çubuk rengi artık öznitelik kümesi DEĞİL, model grubu: mavi/turuncu = bu tezde eğitilen (sayısal/kategorik), **gri = referans (bu tezde eğitilmedi)** (3 Alibaba tahmincisi + 4 medyan/sabit taban çizgisi). Lejant "Reference" diyor; grinin bu çalışmada eğitilmediği ALTYAZIDA yazmalı. **Metin için kritik:** MAE'de ilk üç sıra gri; en iyi eğitilmiş model 4. |
| nb04-fig02 (öznitelik önemi) | Üç panel **farklı birimlerde** (RF: impurity azalması, XGB: ortalama gain, LGBM: bölünme sayısı) — panelller arası çubuk uzunluğu karşılaştırılamaz |
| nb04-fig03 (tahmin vs gerçek) | Rastgele örneklem, n = 3.000 |
| nb04-fig04 (artıklar) | Üç panel ortak x aralığı, merkezi %99 |
| nb04-fig05 (DL karşılaştırma) | Çubuklar 3 tohumun ortalaması, bıyıklar ±1 standart sapma |
| nb05-fig01 (JCT) | ~~yön bilgisi~~ → şekilde ok anahtarı var |
| nb05-fig02 (bekleme CDF) | Sol-üst köşeye yakın eğri daha iyidir; gri demet = vurgulanmayan politikalar |
| nb05-fig03 (slowdown) | log ölçek (yön bilgisi şekildeki ok anahtarında) |
| nb05-fig04 (ısı haritası) | Sütunlar artık "Wait/JCT/Slowdown reduction (%)" — eskiden "Wait ↓ %" yazıyordu ve oradaki ↓ "azalma" demekti, eksen etiketlerindeki "düşük iyidir" ↓'sıyla çakışıyordu |
| mae_spearman_* (rank korelasyon) | 21 politika **tek bir test seti ve tek bir replay** üzerinde; noktalar bağımsız değil, r bu koşuyu betimler |
| Yük duyarlılık şekli | "Load factor" **düşükse daha doygun** küme demektir (sezgiye ters) |

---

## 6. Ok (↓/↑) gösterimi ve n bilgisi — şekil içi anahtar

Kullanıcı kararı: eksen etiketlerindeki ok gösterimi kalsın (alan kaplamıyor,
alandaki standart), ama **okuyucu bilmiyorsa anlayabilsin** diye her şeklin sağ
üst köşesinde küçük gri bir sembol anahtarı var:
`↓ lower is better` / `↑ higher is better` (yalnız o şekilde geçen oklar yazılır).

Aynı şekilde nb02-fig03'te **n lejantta** (`82,184 jobs`) yazıyor; okuyucu kaç
noktanın çizildiğini şekilden okuyabiliyor, altyazıya bağımlı değil.

**Bu bir stil tercihi değil, çakışma düzeltmesi de içeriyor:** NB05 ısı
haritasında `↓` "azalma yüzdesi" (yüksek = iyi) anlamında kullanılıyordu, eksen
etiketlerinde ise "düşük = iyi". Aynı sembol iki anlama geldiği için ısı
haritası sütunları açıkça `... reduction (%)` diye yeniden adlandırıldı.
