# YOLOv8 Aerial Object Detection - Bitirme Projesi

## 📋 Proje Özeti

Bu proje, **havadan görüntüleme sistemlerinde nesne tespiti** için YOLOv8 tabanlı bir derin öğrenme modelinin geliştirilmesini içermektedir. RGB ve termal görüntülerde **uçak, kuş, drone ve helikopter** olmak üzere 4 sınıf tespit edilmektedir.

---

## 🎯 Hedef Sınıflar

| Sınıf ID | Sınıf Adı | Açıklama |
|----------|-----------|----------|
| 0 | Airplane | Uçaklar |
| 1 | Bird | Kuşlar |
| 2 | Drone | İnsansız hava araçları |
| 3 | Helicopter | Helikopterler |

---

## 📊 Dataset Bilgileri

### Dataset Yapısı

```
unified_dataset/
├── train/
│   ├── images/     # 15,086 görüntü
│   └── labels/     # 15,086 etiket
├── val/
│   ├── images/     # 4,076 görüntü
│   └── labels/     # 4,076 etiket
└── test/
    ├── images/     # 1,614 görüntü
    └── labels/     # 1,614 etiket
```

**Toplam:** 20,776 görüntü (31,973 dosya)

### Kaynak Dataset'ler

#### RGB Dataset'ler
- **Anti2(rgb)**: RGB görüntüler
- **flyingobject(rbg)**: Uçan nesneler RGB

#### Thermal Dataset'ler
- **AoD(white-hot-thermal)**: White-hot termal görüntüler
- **termal_drone(white-hot-thermal)**: Drone termal görüntüler
- **IVFlyingObjects(white-hot-thermal)**: Uçan nesneler termal

### Dataset Dağılımı

| Split | RGB | Thermal | Toplam |
|-------|-----|---------|--------|
| **Train** | ~7,543 | ~7,543 | 15,086 |
| **Val** | ~2,038 | ~2,038 | 4,076 |
| **Test** | ~807 | ~807 | 1,614 |

### Sınıf Dağılımı ve Ağırlıklar

Training data'daki sınıf sayıları:

| Sınıf | Instance Sayısı | Class Weight | Notlar |
|-------|----------------|--------------|--------|
| Airplane | 2,460 | 1.017 | Orta sıklıkta |
| Bird | 2,304 | 1.086 | Daha az |
| Drone | 2,501 | 1.000 | En yaygın (baseline) |
| Helicopter | 2,074 | 1.206 | En az (en yüksek ağırlık) |

**Not:** Class weighting şu anda kullanılmıyor (Ultralytics'in son versiyonunda desteklenmiyor).

---

## 🔧 Preprocessing

### Dataset Hazırlama
Script: `prepare_unified_dataset.py`

**İşlem adımları:**
1. **Dataset Merge:** RGB ve Thermal dataset'leri birleştirme
2. **Dosya İsimlendirme:** Unique prefix ekleme (rgb_, thermal_)
3. **Verification:** Etiket-görüntü eşleştirme kontrolü
4. **YAML Config:** YOLOv8 config dosyası oluşturma

**Özellikler:**
- ✅ Oversampling YOK
- ✅ Undersampling YOK
- ✅ Tüm data kullanılıyor (raw data)
- ✅ Dosya isimleri çakışmasını önlemek için prefix ekleniyor

### Veri Artırma (Data Augmentation)

**Minimal augmentation stratejisi** uygulanıyor:

| Augmentation | Değer | Açıklama |
|--------------|-------|----------|
| Mosaic | 0.0 | Devre dışı |
| Mixup | 0.0 | Devre dışı |
| Rotation | ±5° | Minimal dönüş |
| Translation | 5% | Minimal kaydırma |
| Scale | ±10% | Minimal ölçekleme |
| Horizontal Flip | 50% | Standart yatay çevirme |
| Vertical Flip | 0.0 | Devre dışı |
| HSV | 0.0 | Renk değişimi yok (termal için) |

**Neden minimal augmentation?**
- Termal görüntülerde renk değişimi anlamsız
- Havadan görüntüde aşırı augmentation gerçekçi değil
- Model gerçek dağılımı öğrenmeli

---

## 🏋️ Baseline Training Konfigürasyonu

### Model ve Donanım

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **Model** | YOLOv8x | En büyük YOLOv8 modeli |
| **Parameters** | 68.2M | 68 milyon parametre |
| **GFLOPs** | 258.1 | Hesaplama maliyeti |
| **GPU** | NVIDIA H100 80GB | 80GB VRAM |
| **Precision** | Mixed (AMP) | Otomatik mixed precision |

### Training Parametreleri

```python
# Model Configuration
MODEL_SIZE = 'yolov8x.pt'           # Largest model
EPOCHS = 300                         # Long training
BATCH_SIZE = 32                      # Memory optimized
IMG_SIZE = 896                       # Balanced resolution
DEVICE = 0                           # GPU:0

# Optimizer
optimizer = 'AdamW'
lr0 = 0.002                          # Initial learning rate (scaled for batch)
lrf = 0.001                          # Final learning rate
momentum = 0.937
weight_decay = 0.0005

# Loss Weights
box = 7.5                            # Bounding box loss
cls = 0.5                            # Classification loss
dfl = 1.5                            # Distribution focal loss

# Training Settings
patience = 100                       # Early stopping patience
workers = 8                          # Data loading workers
cache = 'disk'                       # Disk cache (VRAM tasarrufu)
seed = 42                            # Reproducibility
deterministic = True                 # Deterministic training
```

### Memory Optimization

**VRAM Kullanımı:**
- Model: ~15-20GB
- Batch processing (32 @ 896): ~15-20GB
- Cache: Disk (0GB VRAM)
- **Toplam:** ~35-40GB / 80GB ✅

**Optimizasyon stratejisi:**
1. `cache='disk'` → RAM cache yerine disk (38GB VRAM tasarrufu)
2. `batch=32` → 64 yerine 32 (15GB VRAM tasarrufu)
3. `imgsz=896` → 1280 yerine 896 (10GB VRAM tasarrufu)

---

## 📁 Proje Yapısı

```
bitirmeprojesi/
├── README.md                          # Bu dosya
├── requirements.txt                   # Python dependencies
├── yolov8_config.yaml                 # YOLOv8 dataset config
│
├── prepare_unified_dataset.py         # Dataset birleştirme script
├── train_yolov8_weighted.py           # Baseline training script
│
├── datasets/                          # Kaynak dataset'ler
│   ├── rgb/
│   │   ├── Anti2(rgb)/
│   │   └── flyingobject(rbg)/
│   └── thermal/
│       ├── AoD(white-hot-thermal)/
│       ├── termal_drone(white-hot-thermal)/
│       └── IVFlyingObjects. (white-hot-thermal)/
│
├── unified_dataset/                   # Birleştirilmiş dataset
│   ├── train/
│   ├── val/
│   └── test/
│
├── runs/                              # Training outputs
│   └── detect/
│       └── train/
│           ├── weights/
│           │   ├── best.pt           # En iyi model
│           │   └── last.pt           # Son checkpoint
│           ├── results.csv           # Training metrics
│           ├── results.png           # Metric grafikler
│           └── ...
│
└── venv/                             # Virtual environment
```

---

## 🚀 Kullanım

### 1. Environment Kurulumu

```bash
# Virtual environment oluştur
cd /home/talha/bitirmeprojesi
python3 -m venv venv
source venv/bin/activate

# Dependencies yükle
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dataset Hazırlama

```bash
# Dataset'leri birleştir ve hazırla
python prepare_unified_dataset.py
```

**Output:**
- ✅ `unified_dataset/` klasörü oluşur
- ✅ `yolov8_config.yaml` oluşur
- ✅ Dataset verification yapılır

### 3. Baseline Training

```bash
# Training'i başlat
python train_yolov8_weighted.py
```

**Training süresi:** ~8-12 saat (300 epoch, H100)

**Monitoring:**
```bash
# TensorBoard ile izle (başka terminal'de)
tensorboard --logdir runs/detect
```

### 4. Evaluation

```bash
# Validation set üzerinde değerlendirme
yolo val model=runs/detect/train/weights/best.pt data=yolov8_config.yaml

# Test set üzerinde değerlendirme
yolo val model=runs/detect/train/weights/best.pt data=yolov8_config.yaml split=test
```

### 5. Inference

```bash
# Tek görüntü
yolo predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg

# Klasör
yolo predict model=runs/detect/train/weights/best.pt source=path/to/images/

# Video
yolo predict model=runs/detect/train/weights/best.pt source=path/to/video.mp4
```

---

## 📈 Baseline Training Beklentileri

### Metrikler

Başarılı bir baseline training için beklenen metrikler:

| Metric | Hedef | Açıklama |
|--------|-------|----------|
| **mAP@50** | >0.75 | IoU=0.5'te ortalama precision |
| **mAP@50-95** | >0.50 | IoU=0.5-0.95 arası mAP |
| **Precision** | >0.80 | Positive predictions accuracy |
| **Recall** | >0.75 | Ground truth coverage |

### Checkpoint'ler

Training boyunca kaydedilen checkpoint'ler:

- `best.pt`: Validation mAP'i en yüksek model
- `last.pt`: Son epoch modeli
- `epoch_X.pt`: Her 10 epoch'ta kayıt (save_period=10)

---

## 🔮 Gelecek Adımlar: İleri Seviye Geliştirmeler

Baseline training tamamlandıktan sonra aşağıdaki geliştirmeler planlanmaktadır:

### 1. P2 Head Eklenmesi

**Amaç:** Küçük nesneleri daha iyi tespit etmek

**Neden gerekli:**
- Havadan görüntülerde nesneler genellikle küçüktür
- Standart YOLOv8: P3, P4, P5 (8x, 16x, 32x downsampling)
- P2 head: 4x downsampling → Daha yüksek çözünürlük

**Implementasyon:**
```python
# Model architecture'ı modifiye et
# P2 head ekle: 4x4 feature map
# Multi-scale detection: P2, P3, P4, P5
```

**Beklenen iyileşme:**
- Küçük drone/bird detection: +10-15% mAP
- Uzak mesafe detection: +20% recall

### 2. SAHI (Slicing Aided Hyper Inference)

**Amaç:** Yüksek çözünürlüklü görüntülerde detection accuracy artırmak

**Nasıl çalışır:**
1. Görüntüyü küçük parçalara (slice) böl
2. Her parçada inference yap
3. Sonuçları birleştir (NMS)

**Avantajları:**
- Küçük nesneleri daha iyi tespit
- Yüksek çözünürlük avantajı
- Memory efficient

**Implementasyon:**
```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# SAHI with YOLOv8
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='runs/detect/train/weights/best.pt',
    confidence_threshold=0.3,
    device='cuda:0'
)

result = get_sliced_prediction(
    image='path/to/image.jpg',
    detection_model=detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)
```

**Beklenen iyileşme:**
- Small object mAP: +15-20%
- Uzak mesafe detection: +25%

### 3. Super Resolution (SR)

**Amaç:** Düşük çözünürlüklü görüntüleri yükseltmek

**Pipeline:**
```
Input Image (low-res) 
    ↓
SR Model (ESRGAN/RealESRGAN)
    ↓
High-res Image
    ↓
YOLOv8 Detection
    ↓
Results
```

**SR Model seçenekleri:**
- **ESRGAN**: Enhanced Super-Resolution GAN
- **RealESRGAN**: Real-world super resolution
- **SwinIR**: Swin Transformer based

**Implementasyon:**
```python
# Pre-process with SR
from RealESRGAN import RealESRGAN

sr_model = RealESRGAN(device='cuda', scale=2)
sr_model.load_weights('RealESRGAN_x2.pth')

# SR + Detection pipeline
upscaled_image = sr_model.predict(low_res_image)
detections = yolo_model.predict(upscaled_image)
```

**Beklenen iyileşme:**
- Düşük kalite görüntülerde: +20-30% mAP
- Gece/kötü hava: +15% detection rate

### 4. Entegre Pipeline

**Final Architecture:**

```
Input Image
    ↓
[SR Model] ← (Opsiyonel, düşük kalite için)
    ↓
[SAHI Slicing] ← 512x512 patches
    ↓
[YOLOv8x + P2 Head] ← Her patch için detection
    ↓
[NMS + Merge] ← Sonuçları birleştir
    ↓
Final Detections
```

---

## 📊 Performans Karşılaştırması (Beklenen)

| Model | mAP@50 | mAP@50-95 | Small Objects | Inference Time |
|-------|--------|-----------|---------------|----------------|
| **Baseline (YOLOv8x)** | 0.75 | 0.50 | 0.45 | ~10ms |
| **+ P2 Head** | 0.82 | 0.58 | 0.60 | ~12ms |
| **+ SAHI** | 0.88 | 0.65 | 0.75 | ~150ms |
| **+ SR** | 0.90 | 0.68 | 0.78 | ~200ms |
| **Full Pipeline** | 0.92 | 0.72 | 0.82 | ~250ms |

---

## 🛠️ Dependencies

```txt
# Core
ultralytics>=8.3.0           # YOLOv8
torch>=2.5.0                 # PyTorch
torchvision>=0.18.0          # Vision utils

# Computer Vision
opencv-python>=4.10.0        # Image processing
pillow>=10.4.0               # Image handling

# Data & Config
numpy>=1.26.0                # Numerical operations
pandas>=2.2.0                # Data manipulation
pyyaml>=6.0.1                # YAML parsing

# Visualization
matplotlib>=3.9.0            # Plotting
seaborn>=0.13.0              # Statistical viz
tensorboard>=2.17.0          # Training monitoring

# Utils
tqdm>=4.66.0                 # Progress bars
albumentations>=1.4.0        # Advanced augmentation
ninja>=1.11.1                # Fast CUDA compilation
psutil>=6.0.0                # System monitoring

# Future additions (Phase 2)
# sahi>=0.11.0               # Slicing aided inference
# RealESRGAN>=0.3.0          # Super resolution
```

---

## 📝 Training Logs ve Monitoring

### TensorBoard Metrics

Training sırasında kaydedilen metrikler:

**Loss metrikleri:**
- `train/box_loss`: Bounding box regression loss
- `train/cls_loss`: Classification loss
- `train/dfl_loss`: Distribution focal loss

**Validation metrikleri:**
- `metrics/precision(B)`: Precision
- `metrics/recall(B)`: Recall
- `metrics/mAP50(B)`: mAP @ IoU=0.5
- `metrics/mAP50-95(B)`: mAP @ IoU=0.5:0.95

**Learning rate:**
- `x/lr0`, `x/lr1`, `x/lr2`: Layer-wise learning rates

### Results Files

```
runs/detect/train/
├── results.csv              # Tüm metrics (CSV)
├── results.png              # Metrics plots
├── confusion_matrix.png     # Confusion matrix
├── F1_curve.png            # F1 score curve
├── P_curve.png             # Precision curve
├── R_curve.png             # Recall curve
├── PR_curve.png            # Precision-Recall curve
└── labels.jpg              # Label distribution
```

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Batch size'ı azalt
BATCH_SIZE = 16  # 32 yerine

# veya Image size'ı küçült
IMG_SIZE = 640  # 896 yerine
```

**2. Dataset not found**
```bash
# Dataset'i hazırla
python prepare_unified_dataset.py
```

**3. Slow training**
```bash
# Workers sayısını artır (eğer CPU idle ise)
workers = 16

# veya cache'i RAM yap (eğer yeterli RAM varsa)
cache = 'ram'
```

**4. Label mismatch**
```bash
# Dataset verification
python prepare_unified_dataset.py  # Yeniden hazırla
```

---

## 📚 Referanslar

- **YOLOv8 Docs:** https://docs.ultralytics.com/
- **SAHI:** https://github.com/obss/sahi
- **RealESRGAN:** https://github.com/xinntao/Real-ESRGAN
- **PyTorch:** https://pytorch.org/docs/

---

## 👥 Proje Bilgileri

**Proje Türü:** Bitirme Projesi (Graduation Project)  
**Konu:** Havadan Görüntüleme Sistemlerinde Nesne Tespiti  
**Model:** YOLOv8x  
**Framework:** Ultralytics, PyTorch  
**Hardware:** NVIDIA H100 80GB  

**Tarih:** Aralık 2025

---

## 📄 License

Bu proje akademik amaçlı geliştirilmiştir.

---

## ✅ Checklist

### Phase 1: Baseline (Current)
- [x] Dataset hazırlama script
- [x] Unified dataset oluşturma
- [x] YOLOv8x model training config
- [x] Memory optimization (80GB VRAM)
- [x] Minimal augmentation strategy
- [ ] Baseline training (300 epochs)
- [ ] Validation ve test evaluation
- [ ] Model export (ONNX/TorchScript)

### Phase 2: Advanced Improvements
- [ ] P2 Head eklenmesi
- [ ] SAHI integration
- [ ] Super Resolution pipeline
- [ ] Full pipeline integration
- [ ] Performance comparison
- [ ] Production deployment

---

**Son Güncelleme:** 15 Aralık 2025
