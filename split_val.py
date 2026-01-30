import os
import random
import shutil

# Yollar
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

VAL_SPLIT = 0.2  # %20 validation

# val klasörü yoksa oluştur
if not os.path.exists(VAL_DIR):
    os.makedirs(VAL_DIR)

# Her sınıf için işlem
for class_name in os.listdir(TRAIN_DIR):
    class_train_path = os.path.join(TRAIN_DIR, class_name)
    class_val_path = os.path.join(VAL_DIR, class_name)

    if not os.path.isdir(class_train_path):
        continue

    # val/class klasörü yoksa oluştur
    if not os.path.exists(class_val_path):
        os.makedirs(class_val_path)

    images = os.listdir(class_train_path)
    random.shuffle(images)

    val_count = int(len(images) * VAL_SPLIT)
    val_images = images[:val_count]

    for img in val_images:
        src = os.path.join(class_train_path, img)
        dst = os.path.join(class_val_path, img)
        shutil.move(src, dst)

    print(f"{class_name}: {val_count} görüntü val klasörüne taşındı")

print("\n✅ Validation set başarıyla oluşturuldu!")
