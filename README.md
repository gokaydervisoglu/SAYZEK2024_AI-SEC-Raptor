# AI-SEC-Raptor: YOLOv5 Tabanlı Nesne Tespiti Projesi

Bu proje, SAYZEK2024 Datathon yarışması için geliştirilmiş bir nesne tespiti projesidir. YOLOv5 modeli kullanılarak **bina**, **yol kesişimi**, **halısaha** ve **silo** sınıflarını tespit etmek üzere tasarlanmıştır. Projede her sınıf için ayrı modeller kullanılarak daha yüksek tespit doğruluğu sağlanmıştır.

## İçindekiler
1. [Proje Hakkında](#1-proje-hakkında)
2. [Kurulum Talimatları](#2-kurulum-talimatları)
    1. [Gerekli Kütüphanelerin Yüklenmesi](#21-gerekli-kütüphanelerin-yüklenmesi)
    2. [CUDA ve GPU Desteği (Opsiyonel)](#22-cuda-ve-gpu-desteği-opsiyonel)
3. [Ortam Gereksinimleri](#3-ortam-gereksinimleri)
4. [Veri Setinin Tanımlanması](#4-veri-setinin-tanımlanması)
5. [Model Eğitimi](#5-model-eğitimi)
6. [Model Çıkarımı (Inference)](#6-model-çıkarımı-inference)
7. [Sonuçların Kaydedilmesi](#7-sonuçların-kaydedilmesi)
8. [Dosya Yapısı](#8-dosya-yapısı)
9. [Yazarlar](#9-yazarlar)

---

## 1. Proje Hakkında

AI-SEC-Raptor, YOLOv5 ile dört farklı sınıfı tespit etmek için tasarlanmıştır. Bu sınıflar şunlardır:

- **Bina**
- **Yol Kesişimi**
- **Halısaha**
- **Silo**

Bu projede, her sınıf için farklı modeller kullanılmış ve eğitim işlemleri optimize edilmiştir.

---

## 2. Kurulum Talimatları

Bu projeyi çalıştırabilmek için öncelikle ortam gereksinimlerini karşılamanız ve gerekli kütüphaneleri kurmanız gerekmektedir.

### 2.1 Gerekli Kütüphanelerin Yüklenmesi

Proje gereksinimlerini yüklemek için aşağıdaki komutu terminalde çalıştırın:

```bash
pip install -r requirements.txt

```

### 2.2 CUDA ve GPU Desteği (Opsiyonel)

Eğer NVIDIA GPU kullanıyorsanız, CUDA desteğini etkinleştirmeniz gerekmektedir. CUDA sürümünüz ile uyumlu olan PyTorch sürümünü yüklemek için [PyTorch'un resmi sitesindeki kurulum talimatlarına](https://pytorch.org/get-started/locally/) göz atabilirsiniz.

---

## 3. Ortam Gereksinimleri

- **Python Sürümü:** 3.8 ve üzeri
- **CUDA Sürümü:** 11.1 ve üzeri (GPU ile eğitim yapılacaksa)
- **PyTorch Sürümü:** 1.8.0 ve üzeri
- **Kütüphaneler:** Numpy, OpenCV, Matplotlib, PyYAML, Ultralytics YOLOv5, Pandas
- **Donanım Gereksinimleri:**
  - **RAM:** Minimum 16 GB
  - **Disk Alanı:** Minimum 50 GB
  - **GPU:** NVIDIA Tesla, T4, V100, A100 veya RTX serisi GPU önerilir

---

## 4. Veri Setinin Tanımlanması

Model eğitimi için kullanılacak veri setini config/ klasörü içerisinde yer alan config/data.yaml dosyasında tanımlamalısınız. Bu dosya, veri yollarını ve sınıf bilgilerini içerir. Aşağıda, örnek bir config/data.yaml dosyası gösterilmektedir:

```yaml
train: ../../dataset/train/images/    # Eğitim verilerinin yolu
val: ../../dataset/val/images/        # Doğrulama verilerinin yolu
test: ../../dataset/test/images/      # Test verilerinin yolu (opsiyonel)

nc: 4                              # Toplam sınıf sayısı (Bina, Yol Kesişimi, Halısaha, Silo)
names: ['Bina', 'Yol Kesisimi', 'Halisaha', 'Silo']  # Sınıf isimleri

```

---
## 5. Model Eğitimi

YOLOv5 modeli ile model eğitimi yapmak için **src/train.py** dosyasını kullanabilirsiniz. Aşağıdaki komut ile eğitimi başlatabilirsiniz:

```bash
python src/yolov5/train.py --weights src/yolov5/yolov5s.pt --data src/config/data.yaml --epochs 300 --batch-size 16 --img-size 640 --lr 0.01 --patience 60

```

*   **--weights**: Başlangıç model ağırlıkları (pre-trained weights)
*   **--data**: Veri seti ve sınıfları içeren **data.yaml** dosyası
*   **--epochs**: Eğitim süresince yapılacak epoch sayısı
*   **--batch-size**: Her adımda işlenecek veri örneği sayısı
*   **--img-size**: Eğitimde kullanılacak görüntü boyutu
*   **--lr**: Başlangıç öğrenme oranı
*   **--patience**: Erken durdurma sabrı (kaç epoch boyunca gelişme olmazsa eğitim durur)

Eğitim sonuçları **output/runs/train/** klasörüne kaydedilecektir.

## 6. Model Çıkarımı (Inference)

Model çıkarımı için **src/inference.py** dosyasını çalıştırabilirsiniz. Test verileri üzerinde tahmin yapmak için aşağıdaki komutu kullanın:

```bash
python src/inference.py

```

Bu komut, test görüntüleri üzerinde tahmin yapacak ve sonuçları **output/output_images/** klasörüne kaydedecektir. 

## 7. Sonuçların Kaydedilmesi


Ayrıca, tahmin sonuçları **JSON** formatında da kaydedilecektir. JSON dosyasının yolu: 

```bash
../output/AI-SEC-Raptor.json

```

Bu JSON dosyasında her tahmin edilen sınıf için aşağıdaki bilgiler yer alacaktır:

- **image_id**: Görüntünün ID'si
- **category_id**: Tahmin edilen sınıfın kategorisi
- **bbox**: Sınıraşan kutunun koordinatları (bounding box)
- **score**: Tahmin edilen sınıfın doğruluk skoru


## 8. Dosya Yapısı

Projenin dosya yapısı aşağıdaki gibidir:

```bash
SAYZEK2024_[TakimAdi/BireyselAd]/
│
├── src/
│   ├── train.py              # Model eğitimi için Python scripti
│   ├── inference.py          # Model çıkarımı için Python scripti
│   ├── models/               # Model dosyaları (YOLOv5 ağırlıkları)
│   ├── utils/                # Yardımcı fonksiyonlar (dataset_utils, metrics, visualization)
│   └── config/               # Hiperparametre ve veri yollarını içeren ayar dosyaları
│
├── notebooks/
│   └── model_development.ipynb # Model geliştirme süreci için Jupyter Notebook
│
├── requirements.txt          # Proje bağımlılıkları
│
└── README.md                 # Proje açıklama dosyası

```

## 9. Yazarlar

- **Takım Adı**: AI-SEC-Raptor
- **Yarışma**: SAYZEK2024 Datathon
- **Katkıda Bulunanlar**: Mehmet Ali Gümüşler, Gökay Dervişoğlu

Bu proje, SAYZEK Datathon yarışması için geliştirilmiş olup nesne tespiti alanında ileri seviye modeller ve optimizasyonlar içermektedir.
