# ADİL KARŞILAŞTIRMA - EĞİTİM KOMUTLARI

## 🎯 Amaç
YOLOv8x Baseline ve P2H modellerini **tamamen adil koşullarda** eğitip karşılaştırmak.

## ✅ Yapılan Düzenlemeler

Her iki model için **aynı parametreler** kullanılacak:
- **Learning Rate:** 0.001 (dengeli)
- **Optimizer:** AdamW
- **Epochs:** 300
- **Patience:** 50
- **Augmentation:** Orta seviye (dengeli)
  - mosaic: 0.5
  - mixup: 0.1
  - copy_paste: 0.15
  - scale: 0.3 (±30%)
  - degrees: 10.0 (±10°)
  - translate: 0.1 (±10%)

## 🚀 1. BASELINE EĞİTİMİ

```bash
cd /home/ilaha/bitirmeprojesi

# Baseline modelini adil parametrelerle eğit
python train_yolov8_weighted.py \
  --epochs 300 \
  --batch 32 \
  --imgsz 896 \
  --device 0 \
  --data /home/ilaha/bitirmeprojesi/yolov8_config.yaml
```

**Beklenen Süre:** ~25-30 saat (GPU'ya bağlı)

**Sonuç Konumu:** `runs/detect/train/weights/best.pt`

---

## 🚀 2. P2H EĞİTİMİ

### Önce: Baseline Ağırlıklarını Kontrol Et
```bash
# Baseline eğitimi tamamlandıktan sonra
ls -lh runs/detect/train/weights/best.pt
```

### P2H Modelini Eğit
```bash
cd /home/ilaha/bitirmeprojesi

# P2H modelini baseline'dan transfer learning ile eğit
python train_p2h_ultra.py \
  --epochs 300 \
  --batch 24 \
  --imgsz 896 \
  --device 0 \
  --lr0 0.001 \
  --lr-strategy cosine \
  --baseline-weights runs/detect/train/weights/best.pt \
  --name p2h_fair
```

**Not:** Batch size 24 (P2H daha fazla memory kullanır)

**Beklenen Süre:** ~30-35 saat (P2H daha fazla parametre içerir)

**Sonuç Konumu:** `runs/detect/p2h_fair/weights/best.pt`

---

## 📊 3. DEĞERLENDİRME

### Karşılaştırmalı Değerlendirme
```bash
python evaluate_models.py \
  --baseline runs/detect/train/weights/best.pt \
  --p2h runs/detect/p2h_fair/weights/best.pt \
  --data /home/ilaha/bitirmeprojesi/yolov8_config.yaml
```

### Test Setinde Değerlendirme (SAHI ile)
```bash
# Baseline model
python inference_baseline_sahi.py

# P2H model
python inference_p2h_sahi.py
```

---

## 📈 SONUÇLARI GÖRÜNTÜLEME

### TensorBoard
```bash
tensorboard --logdir runs/detect --port 6006
```

Tarayıcıda: `http://localhost:6006`

### CSV Sonuçları
```bash
# Baseline sonuçları
cat runs/detect/train/results.csv | tail -5

# P2H sonuçları
cat runs/detect/p2h_fair/results.csv | tail -5
```

---

## 🎯 BEKLENEN SONUÇLAR

### Adil Karşılaştırma Beklentileri:

| Metrik | Baseline | P2H | Beklenen İyileşme |
|--------|----------|-----|-------------------|
| **mAP@50** | 0.78-0.82 | 0.81-0.86 | +3-5% |
| **mAP@50-95** | 0.48-0.52 | 0.50-0.54 | +2-4% |
| **Küçük Objeler** | Baseline | Daha iyi | +8-12% |

### P2H'nın Avantajları:
1. ✅ **P2 head** → 1/4 çözünürlükte detection (baseline 1/8)
2. ✅ **4-head mimarisi** → Daha zengin feature pyramid
3. ✅ **Küçük objeler** → Bird ve Drone için özellikle etkili

### Eğer P2H Hala Kötüyse:
- Weight transfer'i kontrol et
- Differential LR dene: `--differential-lr`
- LR stratejisini değiştir: `--lr-strategy plateau`

---

## ⚡ HIZLI TEST (5 Epoch)

Parametreleri test etmek için önce 5 epoch dene:

```bash
# Baseline quick test
python train_yolov8_weighted.py --epochs 5 --batch 32

# P2H quick test
python train_p2h_ultra.py --epochs 5 --batch 24 --name p2h_quick_test
```

Loss'un düzgün düştüğünü görürsen 300 epoch'a başla.

---

## 📝 NOTLAR

1. **GPU Memory:** P2H için en az 12GB GPU memory önerilir
2. **Disk Space:** ~10GB boş alan gerekli (checkpoint'ler için)
3. **Reproducibility:** seed=42 her iki modelde de ayarlandı
4. **Early Stopping:** patience=50 ile overfit önlenir
5. **Backup:** Her 10 epoch'ta checkpoint kaydedilir

---

## 🔧 SORUN GİDERME

### OOM (Out of Memory) Hatası:
```bash
# Batch size'ı düşür
--batch 16  # veya 12
```

### Çok Yavaş Eğitim:
```bash
# Workers'ı artır (CPU'ya bağlı)
# train_yolov8_weighted.py içinde workers=8 zaten ayarlı
```

### Connection Reset Hatası:
```bash
# P2H için workers=0 zaten ayarlı (script içinde)
```

---

## ✅ EĞİTİM KONTROL LİSTESİ

- [ ] Gereksiz test scriptleri silindi
- [ ] Baseline parametreleri güncellendi (adil)
- [ ] P2H parametreleri güncellendi (adil)
- [ ] GPU memory yeterli (>12GB)
- [ ] Dataset hazır ve doğrulanmış
- [ ] Baseline eğitimi başlatıldı
- [ ] Baseline eğitimi tamamlandı
- [ ] P2H eğitimi başlatıldı
- [ ] P2H eğitimi tamamlandı
- [ ] Sonuçlar karşılaştırıldı
- [ ] Sonuçlar dokümante edildi

---

## 🎓 BEKLENTİLER

**Normal koşullarda:**
- P2H, baseline'dan %3-8 daha iyi mAP almalı
- Özellikle küçük objeler (Bird, Drone) için fark belirgin olmalı
- Training loss benzer hızda düşmeli

**Eğer P2H daha kötüyse:**
- Transfer learning sorunu olabilir → `train_p2h_optimized.py` kontrol et
- LR çok düşük olabilir → `--lr0 0.002` dene
- Augmentation çok agresif olabilir → Parametreler zaten dengelendi

Bu dokümandaki komutlarla **bilimsel olarak geçerli** bir karşılaştırma yapabilirsin! 🚀
