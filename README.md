# Derin Öğrenme Tabanlı Göz Hastalıkları Karar Destek Sistemi

## Özet
Diyabetik Retinopati, Glokom ve Retina Dekolmanı gibi göz hastalıkları erken teşhis edilmediğinde kalıcı görme kayıplarına yol açabilmektedir. Son yıllarda derin öğrenme tabanlı görüntü işleme yöntemleri, tıbbi görüntü analizinde başarılı sonuçlar vermektedir.  
Bu projede, retina fundus kamera görüntülerini kullanarak **Convolutional Neural Network (CNN)** tabanlı bir **Göz Hastalıkları Karar Destek Sistemi** geliştirilmiştir. Sistem, **MobileNetV2** mimarisi kullanılarak eğitilmiş olup, **Flask tabanlı bir web arayüzü** üzerinden kullanıcıya göz hastalığı tahmini sunmaktadır.

---

## 1. Giriş
Göz hastalıkları, dünya genelinde görme kaybının en önemli nedenlerinden biridir. Erken teşhis ve doğru tedavi, görme kaybını önlemede kritik rol oynamaktadır. Ancak göz hastalıklarının tanısı genellikle uzman hekimler tarafından yapılmakta ve bu süreç zaman alabilmektedir.

Bu çalışmanın amacı, derin öğrenme yöntemleri kullanılarak retina görüntülerinden otomatik hastalık tespiti yapabilen bir karar destek sistemi geliştirmektir. Geliştirilen sistem, sağlık profesyonellerine yardımcı olmayı ve tanı sürecini hızlandırmayı hedeflemektedir.

---

## 2. Veri Seti
Bu projede kullanılan veri seti, fundus kamera görüntülerinden elde edilen etiketli göz görüntülerinden oluşmaktadır. Veri setinde aşağıdaki göz hastalıklarına ait sınıflar bulunmaktadır:

- Diyabetik Retinopati
- Glokom
- Retina Dekolmanı
- Miyopi
- Sağlıklı Göz
- Diğer retina hastalıkları

Veri seti boyutunun büyük olması nedeniyle **GitHub deposuna eklenmemiştir**. Veri seti aşağıdaki klasör yapısına sahiptir:

veri seti/
├── eğitim/
├── doğrulama/
└── test/


Her klasör, ilgili hastalık sınıflarını içeren alt dizinlerden oluşmaktadır.

---

## 3. Yöntem

### 3.1 Veri Ön İşleme
Model eğitimi öncesinde görüntülere aşağıdaki ön işlemler uygulanmıştır:
- Görüntüler 224×224 boyutuna yeniden ölçeklendirilmiştir
- Piksel değerleri [0,1] aralığına normalize edilmiştir
- Veri artırma (Data Augmentation) teknikleri uygulanmıştır:
  - Döndürme
  - Yakınlaştırma
  - Yatay çevirme

### 3.2 Model Mimarisi
Model geliştirme sürecinde **MobileNetV2** mimarisi kullanılmıştır. ImageNet veri seti üzerinde önceden eğitilmiş model transfer öğrenme yöntemiyle adapte edilmiştir.

Model katmanları:
- MobileNetV2 (Önceden eğitilmiş)
- Global Average Pooling
- Dense (256 nöron, ReLU aktivasyon)
- Dropout (0.5)
- Softmax çıkış katmanı

### 3.3 Eğitim Parametreleri
- Optimizasyon algoritması: Adam
- Öğrenme oranı: 0.0001
- Batch size: 16
- Epoch sayısı: 10
- Kayıp fonksiyonu: Categorical Crossentropy

---

## 4. Sistem Mimarisi
Geliştirilen karar destek sistemi üç ana bileşenden oluşmaktadır:

1. **Model Eğitim Modülü** (`train_model.py`)
2. **Tahmin Modülü** (`predict.py`)
3. **Web Arayüzü** (`app.py` – Flask)

Kullanıcı web arayüzü üzerinden bir göz görüntüsü yüklediğinde, sistem eğitilmiş modeli kullanarak göz hastalığı tahminini gerçekleştirmektedir.

---

## 5. Sonuçlar
Model eğitimi sonucunda, eğitim ve doğrulama verileri üzerinde başarılı sonuçlar elde edilmiştir. Eğitim sürecine ait doğruluk (accuracy) ve kayıp (loss) grafiklerinin çıktıları `results/` klasöründe bulunmaktadır.

Elde edilen sonuçlar, derin öğrenme tabanlı yaklaşımların göz hastalıklarının otomatik tespitinde etkili olduğunu göstermektedir.

---

## 6. Kullanılan Teknolojiler
- Python
- TensorFlow / Keras
- MobileNetV2
- Flask
- OpenCV
- Matplotlib
- Git & GitHub

---

## 7. Projenin Çalıştırılması

### 7.1 Gerekli Kütüphanelerin Kurulumu
```bash
pip install -r requirements.txt

###7.2 Modelin Eğitilmesi
python train_model.py

###7.3 Web Uygulamasının Başlatılması
python app.py

---

## 8.Sonuç ve Gelecek Çalışmalar
Bu proje kapsamında geliştirilen karar destek sistemi, retina görüntülerinden otomatik göz hastalığı tespiti yapabilmektedir. Gelecek çalışmalarda daha büyük veri setleri, farklı derin öğrenme mimarileri ve model iyileştirme teknikleri kullanılarak sistemin başarımı artırılabilir.

---

## 9. Yazar Bilgisi

Ad Soyad: Berk Çabuk
Ders: Sinir Ağları
Proje Türü: Büt Projesi

