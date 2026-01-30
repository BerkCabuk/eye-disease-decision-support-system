import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# =========================
# AYARLAR
# =========================
MODEL_PATH = "model/eye_model.h5"
IMG_SIZE = 224

CLASS_NAMES = [
    "Central Serous Chorioretinopathy",
    "Diabetic Retinopathy",
    "Disc Edema",
    "Glaucoma",
    "Healthy",
    "Macular Scar",
    "Myopia",
    "Pterygium",
    "Retinal Detachment",
    "Retinitis Pigmentosa"
]

# =========================
# MODELİ YÜKLE
# =========================
model = tf.keras.models.load_model(MODEL_PATH)
print("✔ Model yüklendi")

# =========================
# TAHMİN FONKSİYONU
# =========================
def predict_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ Dosya bulunamadı: {img_path}")

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index] * 100

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence

# =========================
# TEST (ELLE ÇALIŞTIRMA)
# =========================
if __name__ == "__main__":
    test_image_path = "test.jpg"  # buraya test resmi koy

    disease, confidence = predict_image(test_image_path)

    print("🧠 Tahmin Sonucu")
    print("----------------------")
    print(f"Hastalık: {disease}")
    print(f"Güven: %{confidence:.2f}")
