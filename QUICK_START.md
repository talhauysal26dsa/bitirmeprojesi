# 🎯 ADİL KARŞILAŞTIRMA - HIZLI BAŞLANGIÇ

## ✅ YAPILAN DEĞİŞİKLİKLER (20 Aralık 2025)

### 1. ❌ Silinen Gereksiz Scriptler:
- `quick_test.py` (20 epoch test scripti - artık gereksiz)
- `pilot_batch_optimization.py` (batch size test scripti - artık gereksiz)

### 2. ✏️ Güncellenen Ana Scriptler:

#### `train_yolov8_weighted.py` (Baseline)
- ✅ Augmentation: Orta seviyeye getirildi (P2H ile aynı)
- ✅ lr0: 0.001 (P2H ile aynı)
- ✅ patience: 50 (P2H ile aynı)
- ✅ Tüm augmentation parametreleri dengelendi

#### `train_p2h_ultra.py` (P2H)
- ✅ Augmentation: Orta seviyeye indirildi (Baseline ile aynı)
- ✅ Default name: "p2h_fair"
- ✅ Tüm augmentation parametreleri dengelendi

### 3. 📚 Oluşturulan Dokümantasyon:
- `FAIR_COMPARISON_PLAN.md` - Detaylı karşılaştırma planı
- `TRAINING_COMMANDS.md` - Adım adım eğitim komutları
- `BEFORE_AFTER_COMPARISON.md` - Öncesi/sonrası parametre karşılaştırması
- `QUICK_START.md` (bu dosya) - Hızlı başlangıç

---

## 🚀 HEMEN BAŞLAMAK İÇİN

### 1️⃣ BASELINE EĞİTİMİNİ BAŞLAT (Önce Bu!)
```bash
cd /home/ilaha/bitirmeprojesi
python train_yolov8_weighted.py --epochs 300 --batch 32 --device 0
```
⏱️ Süre: ~25-30 saat

### 2️⃣ P2H EĞİTİMİNİ BAŞLAT (Baseline bittikten sonra)
```bash
python train_p2h_ultra.py \
  --epochs 300 \
  --batch 24 \
  --device 0 \
  --baseline-weights runs/detect/train/weights/best.pt \
  --name p2h_fair
```
⏱️ Süre: ~30-35 saat

### 3️⃣ SONUÇLARI KARŞILAŞTIR
```bash
python evaluate_models.py \
  --baseline runs/detect/train/weights/best.pt \
  --p2h runs/detect/p2h_fair/weights/best.pt
```

---

## 📊 PARAMETRELER (ADİL KARŞILAŞTIRMA)

| Parametre | Değer | Her İki Modelde de Aynı? |
|-----------|-------|---------------------------|
| epochs | 300 | ✅ |
| lr0 | 0.001 | ✅ |
| lrf | 0.01 | ✅ |
| optimizer | AdamW | ✅ |
| mosaic | 0.5 | ✅ |
| mixup | 0.1 | ✅ |
| copy_paste | 0.15 | ✅ |
| scale | 0.3 (±30%) | ✅ |
| degrees | 10.0 (±10°) | ✅ |
| translate | 0.1 (±10%) | ✅ |
| patience | 50 | ✅ |
| warmup | 3 epochs | ✅ |
| cos_lr | True | ✅ |
| batch | 32 / 24 | ⚠️ (memory limiti) |

---

## 🎯 BEKLENEN SONUÇLAR

### Adil Koşullarda:
- **Baseline mAP@50:** 0.75-0.80
- **P2H mAP@50:** 0.78-0.84
- **İyileşme:** +3-8% ✨

### P2H'nın Avantajları:
1. ✅ P2 head → 1/4 çözünürlükte detection
2. ✅ 4-head mimari → Zengin feature pyramid
3. ✅ Küçük objeler → Özellikle Bird ve Drone için

---

## ⚠️ SORUN YAŞARSAN

### Out of Memory (OOM):
```bash
# Batch size düşür
--batch 16  # veya 12
```

### Transfer Learning Sorunu:
```bash
# Baseline weights kontrol et
ls -lh runs/detect/train/weights/best.pt
```

### Yavaş Eğitim:
```bash
# Normal - P2H daha fazla parametre içerir
# GPU kullanımını kontrol et: nvidia-smi
```

---

## 📁 PROJE YAPISI

```
bitirmeprojesi/
├── train_yolov8_weighted.py      ← Baseline (GÜNCELLENDİ ✅)
├── train_p2h_ultra.py             ← P2H (GÜNCELLENDİ ✅)
├── train_p2h_optimized.py         ← Transfer learning yardımcısı
├── train_p2h_adaptive_callbacks.py ← Callbacks
├── train_p2h_callbacks.py         ← Differential LR
├── evaluate_models.py             ← Karşılaştırma
├── inference_*.py                 ← Test scriptleri
├── yolov8_config.yaml            ← Dataset config
├── yolov8x-p2-custom.yaml        ← P2H mimari
├── TRAINING_COMMANDS.md          ← Komutlar (YENİ 📚)
├── FAIR_COMPARISON_PLAN.md       ← Plan (YENİ 📚)
├── BEFORE_AFTER_COMPARISON.md    ← Karşılaştırma (YENİ 📚)
└── QUICK_START.md                ← Bu dosya (YENİ 📚)
```

---

## ✅ KONTROL LİSTESİ

- [x] Gereksiz scriptler silindi
- [x] Baseline güncellemesi yapıldı
- [x] P2H güncellemesi yapıldı
- [x] Parametreler eşitlendi
- [x] Dokümantasyon oluşturuldu
- [ ] **Baseline eğitimi başladı**
- [ ] Baseline eğitimi tamamlandı
- [ ] **P2H eğitimi başladı**
- [ ] P2H eğitimi tamamlandı
- [ ] Sonuçlar karşılaştırıldı

---

## 🎓 ÖNEMLİ NOTLAR

1. **Önce baseline'ı tamamla** - P2H için gerekli
2. **GPU memory'yi izle** - nvidia-smi
3. **Checkpoint'leri sakla** - Her 10 epoch'ta kaydediliyor
4. **TensorBoard kullan** - tensorboard --logdir runs/detect
5. **Sabırlı ol** - Her eğitim ~30 saat sürer

---

## 📞 DESTEK

Sorun yaşarsan şu dosyalara bak:
- `TRAINING_COMMANDS.md` - Detaylı komutlar
- `FAIR_COMPARISON_PLAN.md` - Detaylı plan
- `BEFORE_AFTER_COMPARISON.md` - Ne değişti?

**Başarılar! 🚀**
