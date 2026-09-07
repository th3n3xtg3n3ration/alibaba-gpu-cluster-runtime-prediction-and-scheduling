# Düzeltme Planı — Onay Dokümanı

**Kaynak:** İki turlu denetim (Fable 5.1 + Opus 5), 133 onaylı bulgu, 3/3 yargıç RED.
**Amaç:** Düzeltmeye başlamadan önce ne yapılacağını, hangi sırayla, ve hangi kararların sana ait olduğunu netleştirmek.

---

## ÖNCE — 4 KARAR SENDE, BEN TEK BAŞIMA VEREMEM

### Karar 1: `num_inst` düzeltmesi ile küme boyutu çelişiyor — hangisini feda ediyoruz?

Kod şu an bir işin kaynak talebini yalnız `num_gpu` ile hesaplıyor; `num_inst` (paralel örnek sayısı, ortalama 5, işlerin %21'inde >1) hiç çarpılmıyor. Bunu düzelttiğimde ölçtüm:

```
rho=0,7 (kararlı, gerçekçi kuyruk rejimi) için gereken küme:
  düzeltmeden : 758 GPU
  düzeltince  : 4.348 GPU   (gerçek Alibaba kümesi ~6.500 GPU'ya çok yakın)
```

Yani `num_inst`'i düzeltirsem, "32-GPU küçük küme / 256-GPU büyük küme" karşılaştırmasının **ikisi de** aşırı-doygun kalır, aralarındaki fark anlamsızlaşır — kararlı rejime ulaşmak için pratikte tüm kümeyi simüle etmek gerekir.

**Üç seçenek:**
- **A.** `num_inst`'i düzelt, küme boyutunu 2-4 bin GPU'ya çıkar (tez tasarımının "küçük/büyük küme" çerçevesi değişir, yeni bir anlatı gerekir)
- **B.** `num_inst`'i düzelt, kararlı rejimi ARAMAKTAN vazgeç; bunun yerine "kasıtlı olarak aşırı yüklü, toplu-iş tipi bir rejimde SPT davranışı inceliyoruz" diye **dürüstçe çerçevele** (deney tasarımı değişmez, yalnız ne ölçtüğümüzün tanımı değişir)
- **C.** Önce `num_inst`'in gerçek anlamını iz şemasından/NSDI'22 makalesinden **teyit et** (ben bunu %100 doğrulayamadım — Opus 5,74× dedi, benim hesabım 3,7× çıktı, bu fark bile semantiğin net olmadığını gösteriyor), sonra karar ver

**Önerim: C, sonra muhtemelen B.** Yanlış anlaşılmış bir alanı düzeltip tüm tez tasarımını değiştirmek riskli; önce doğrulayalım.

### Karar 2: LOAD_FACTOR / küme boyutu — Karar 1'in cevabından bağımsız olarak da düzeltilmeli mi?

`num_inst`'i hiç düzeltmesek bile mevcut rejim aşırı doygun (ρ≈17-166). En az bir kararlı-rejim noktası (ρ≈0,6-0,9) eklemek CRITICAL bir düzeltme. Bunu Karar 1'den bağımsız, **her koşulda yapmayı öneriyorum.**

### Karar 3: Simülatör davranışı — HoL mü, tezin tanımladığı "skip-over" mu?

Kod tüm politikalara katı head-of-line bloklaması uyguluyor; tez metni SJF/SRF için "sığan işler arasından en kısayı seç" (skip-over) tanımlıyor. İkisi de savunulabilir, ama şu an **kod ile tez birbirini yalanlıyor**.
- **A.** Kodu tez tanımına uydur (skip-over ekle) — ana sonuçlar değişir, yeniden koşu gerekir
- **B.** Tez tanımını koda uydur (katı HoL olduğunu yaz) — kod değişmez, yalnız metin

**Önerim: B.** Katı HoL, gerçek zamanlayıcılarda da yaygın bir tasarım; sadece dürüstçe yazılması yeterli.

### Karar 4: Backfill sonuçları ana metne girsin mi?

Şu an backfill (EASY) yalnız duyarlılık ızgarasında var, ana tablolarda yok. Backfill açıldığında bazı politikaların kazancı 5-17 puan düşüyor. Bunu ana metne taşımak tezi daha gerçekçi ama daha alçakgönüllü sonuçlara taşır.

**Önerim: Evet, taşı** — zaten koşuyorsun, veri hazır, ve "gerçekçi zamanlayıcıda ne olur" sorusuna cevap veriyor.

---

## KADEME 1 — Kod düzeltmesi + zorunlu yeniden koşu (CRITICAL kaynaklı, 9 madde)

Yukarıdaki kararlar netleşince bunlar uygulanacak, sonra **tüm pipeline (04+05) yeniden koşulacak**:

| # | Ne | Bağlı olduğu karar |
|---|---|---|
| 1 | `num_inst` kaynak talebine dahil et (ya da dahil etmemenin gerekçesini yaz) | Karar 1 |
| 2 | LightGBM: `objective='regression_l1'` final refit'e geçir, XGB `eval_metric='mae'`; 3+3 modeli yeniden eğit | — (bağımsız, net) |
| 3 | ρ hesapla, raporla, en az bir kararlı-rejim noktası ekle | Karar 1+2 |
| 4 | Taban çizgilerini (ProfileMedian, Per-User Median, sabit medyan, sabit-0) tablo + simülasyona zorunlu ekle | — (bağımsız, net, düşük efor) |
| 5 | Tüm tabloları checkpoint/CSV'den otomatik üret (HTML→LaTeX köprüsü) | — (bağımsız) |
| 6 | Şekilleri güncel koşudan yeniden export et | — (zaten planlıydı) |
| 7 | Ölçek-bağımlı dikotomi anlatısını (abstract/ch1/ch6/ch7) kaldır, yerine Spearman ρ=0,99 | — (bağımsız) |
| 8 | Simülasyon-içi öznitelik tutarsızlığını gider (modele 534 GPU'luk yük besleniyor, küme 32/256) | Karar 1 |
| 9 | Tüm değişiklikleri tek etiketli commit'e al | — |

## KADEME 2 — MAJOR (44 madde, tez yazımıyla paralel ilerleyebilir)

En etkili 6 tanesi:
- Wilcoxon pseudo-replication çözümü (blok-bootstrap veya ≥3-5 rolling-origin pencere)
- Tek yönlü test yorumunu düzelt (p≈1 → "FIFO'dan anlamlı KÖTÜ", işaret ters)
- Sweep-line ablasyonu (özellik gerçekten katkı sağlıyor mu, ölç)
- `Machine.release()` job_id ile eşleştir (backfill rezervasyon hatası)
- DL `EarlyStopping` delta'sını göreli yap, yeniden eğit
- Simülatör semantiğini tez algoritmasıyla hizala (Karar 3)

Kalan 38 madde çoğunlukla metin/şekil/altyazı düzeltmesi — ayrı bir kod-koşu döngüsü gerektirmiyor.

## KADEME 3 — MODERATE + MINOR (74 madde)

Neredeyse tamamı yalnız metin/kozmetik (MAPE kararı, altyazı düzeltmeleri, README senkronizasyonu, test kapsamı). Kod/koşu etkisi yok, LaTeX aşamasında toplu halledilir.

---

## SÜREÇ

1. **Sen** Karar 1-4'ü ver
2. **Ben** Kademe 1'i kodla (kararlara göre şekillenir)
3. **Sen** temiz çekirdekle NB04→NB05'i yeniden koştur (~4-9 saat, karara göre değişir)
4. **Ben** Kademe 2'nin kod gerektiren kısmını uygularım, gerekirse ikinci bir koşu
5. **Ben** Kademe 3 + tüm LaTeX güncellemesini tek geçişte yaparım

---

*Tam bulgu listesi ve gerekçeler: `reports/audit_phase1_opus/OPUS_TAM_RAPOR.md` (bölüm C/D/M).*
