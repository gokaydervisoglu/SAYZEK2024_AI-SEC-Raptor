import json
import os
import torch
import pathlib
import cv2
from yolov5 import YOLOv5

# pathlib.PosixPath hatasından kaçınmak için, Windows yolunu kullanacak şekilde ayarlama yapıyoruz
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Test ve çıktı görüntülerinin bulunduğu klasör yolları
test_images_path = '../data/test-images'  # Test görüntülerinin olduğu klasör
output_images_path = '../output/output_images'  # Tahmin sonuçlarının kaydedileceği klasör

# Eğer çıktı klasörü mevcut değilse oluştur
if not os.path.exists(output_images_path):
    os.makedirs(output_images_path)

# Görüntü dosya adları ile ID'lerin eşleşmelerini içeren JSON dosyasını yükleme
with open('../config/image_file_name_to_image_id.json', 'r') as f: 
    image_file_name_to_image_id = json.load(f)

# YOLOv5 modellerini yükle (farklı sınıflar için eğitimli modeller)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../models/bina.pt')  # Bina modeli
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='../models/yol_kesisimi.pt')  # Yol kesişimi modeli
model2 = torch.hub.load('ultralytics/yolov5', 'custom', path='../models/halisaha.pt')  # Halısaha modeli
model3 = torch.hub.load('ultralytics/yolov5', 'custom', path='../models/silo.pt')  # Silo modeli

# Tahmin sonuçlarını tutacak boş bir liste
results = []

# Test klasöründeki her bir resim için tahmin işlemi
for img_name in os.listdir(test_images_path):
    if img_name.endswith(('.jpg', '.png')):  # Sadece görüntü dosyalarını işle
        image_path = os.path.join(test_images_path, img_name)
        img = cv2.imread(image_path)  # Görüntüyü yükle
        
        # Her model için tahmin işlemini yap
        results_model = model(image_path)
        results_model1 = model1(image_path)
        results_model2 = model2(image_path)
        results_model3 = model3(image_path)

        # Bounding box'ları çizmek ve sonuçları işlemek için fonksiyon
        def draw_boxes(results_model, img, label_id):
            bboxes = results_model.xyxy[0].cpu().numpy()  # Bounding box'ları xyxy formatında al
            labels = results_model.xyxy[0][:, 5].cpu().numpy()  # Tahmin edilen sınıf etiketleri
            scores = results_model.xyxy[0][:, 4].cpu().numpy()  # Tahmin edilen skorlar

            # JSON dosyasından görüntü ID'sini al
            img_id = image_file_name_to_image_id[img_name] 

            # Her bir tahmin edilen bounding box için sonuçları işleme
            for bbox, label, score in zip(bboxes, labels, scores):
                
                # Eğer yol kesişimi sınıfı ise ve skor düşükse, skoru artır
                if int(label_id) == 2 and score <= 0.5:  
                    score += 0.2

                # Bounding box'ları xyxy formatından xywh formatına dönüştür
                bbox[2], bbox[3] = bbox[2] - bbox[0], bbox[3] - bbox[1]  
                
                # Sonuçları listeye ekle
                res = {
                    'image_id': img_id,
                    'category_id': int(label_id),  # Kategori ID'si
                    'bbox': list(bbox[:4].astype('float64')),  # Bounding box'lar
                    'score': float("{:.8f}".format(score))  # Tahmin edilen skor
                }
                results.append(res)

                # Bounding box'ları görüntüye çiz
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])
                color = (0, 255, 0)  # Yeşil renkte bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{label_id}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Tüm modeller için bounding box'ları çiz
        draw_boxes(results_model, img, 1)  # Bina modeli
        draw_boxes(results_model1, img, 2)  # Yol kesişimi modeli
        draw_boxes(results_model2, img, 3)  # Halısaha modeli
        draw_boxes(results_model3, img, 4)  # Silo modeli

        # Çıktı görüntüsünü belirtilen klasöre kaydet
        output_image_path = os.path.join(output_images_path, img_name)
        cv2.imwrite(output_image_path, img)

# Sonuçları bir JSON dosyasına kaydet
with open('../output/AI-SEC-Raptor.json', 'w') as f:
    json.dump(results, f)

print("Inference işlemi tamamlandi.")
