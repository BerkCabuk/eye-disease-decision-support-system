import os
import shutil
import random

# Orijinal veri setinin olduğu klasör
SOURCE_DIR = "raw_dataset"   # BURAYA DİKKAT
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

SPLIT_RATIO = 0.8  # %80 train, %20 test

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

classes = os.listdir(SOURCE_DIR)

for cls in classes:
    cls_path = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_index]
    test_images = images[split_index:]

    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(TEST_DIR, cls, img)
        )

    print(f"{cls}: {len(train_images)} train, {len(test_images)} test")

print("✔ Veri seti başarıyla bölündü")
